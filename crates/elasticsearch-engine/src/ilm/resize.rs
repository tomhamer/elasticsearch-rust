//! `_clone`, `_shrink` and `_split` (`TransportResizeAction` +
//! `MetadataCreateIndexService.prepareResizeIndexSettings`).
//!
//! Every target shard starts as a local segment copy of its (first) source
//! shard (ES `selectCloneShard` / `selectSplitShard` / `selectShrinkShards`):
//! after a flush, the immutable segment files of the committed core index
//! are hard-linked (copied when the filesystem cannot link) into the
//! target's shard directory; translog and state files are copied. A split
//! target then deletes the documents that route to another target shard
//! (ES `ShardSplittingQuery`); a shrink target adds the documents of its
//! other source shards. Every ES precondition is enforced (write block,
//! shard-count factors, routing shards, data stream write index, ...).

use crate::error::{EsError, Result};
use crate::index::{new_uuid, IndexService};
use crate::node::Node;
use crate::settings::{self, FlatSettings};
use serde_json::{json, Map, Value};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResizeType {
    Shrink,
    Split,
    Clone,
}

impl ResizeType {
    pub fn parse(s: &str) -> Option<ResizeType> {
        match s {
            "_shrink" | "shrink" => Some(ResizeType::Shrink),
            "_split" | "split" => Some(ResizeType::Split),
            "_clone" | "clone" => Some(ResizeType::Clone),
            _ => None,
        }
    }
}

/// `ShrinkShardsCalculator.calculateAcceptableNumberOfShards`.
pub fn acceptable_shrink_shards(src: i64, min: i64) -> i64 {
    if src <= 0 || min <= 0 {
        return 1;
    }
    if src % min == 0 {
        return min;
    }
    let num = (src as f64).sqrt().floor() as i64;
    if min >= num {
        let mut i = num;
        while i >= 1 {
            if src % i == 0 && min <= src / i {
                return src / i;
            }
            i -= 1;
        }
    } else {
        for i in 1..=num {
            if src % i == 0 && min <= i {
                return i;
            }
        }
    }
    src
}

/// `IndexMetadata.getRoutingFactor`.
fn routing_factor(src: i64, tgt: i64) -> Result<i64> {
    if src < tgt {
        let f = tgt / src;
        if f * src != tgt || f <= 1 {
            return Err(EsError::illegal_argument(format!("the number of source shards [{src}] must be a factor of [{tgt}]")));
        }
        Ok(f)
    } else if src > tgt {
        let f = src / tgt;
        if f * tgt != src || f <= 1 {
            return Err(EsError::illegal_argument(format!("the number of source shards [{src}] must be a multiple of [{tgt}]")));
        }
        Ok(f)
    } else {
        Ok(1)
    }
}

/// `MetadataCreateIndexService.calculateNumRoutingShards` (indices >= 7.0).
fn default_routing_shards(n: i64) -> i64 {
    let log2_n = 64 - ((n - 1).max(0) as u64).leading_zeros() as i64;
    let splits = (10 - log2_n).max(1);
    n << splits
}

fn routing_shards(svc: &IndexService, shards: i64) -> i64 {
    svc.setting("index.number_of_routing_shards").and_then(|s| s.parse().ok()).unwrap_or_else(|| default_routing_shards(shards))
}

fn int_setting(flat: &FlatSettings, k: &str) -> Result<Option<i64>> {
    match flat.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let s = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            s.parse::<i64>().map(Some).map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}]")))
        }
    }
}

fn validate_target_name(name: &str) -> Result<()> {
    let bad = |why: &str| Err(EsError::invalid_index_name(name, why));
    if name.is_empty() {
        return bad("must not be empty");
    }
    if name != name.to_lowercase() {
        return bad("must be lowercase");
    }
    if name.starts_with(['-', '_', '+']) {
        return bad("must not start with '_', '-', or '+'");
    }
    if name == "." || name == ".." {
        return bad("must not be '.' or '..'");
    }
    if name.chars().any(|c| matches!(c, '\\' | '/' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' | ',' | '#' | ':')) {
        return bad("must not contain the following characters [ , \", *, \\, <, |, ,, >, /, ?]");
    }
    if name.len() > 255 {
        return bad("index name is too long, (> 255)");
    }
    Ok(())
}

/// Settings never copied from the source (private / `NotCopyableOnResize`).
fn not_copyable(k: &str) -> bool {
    matches!(
        k,
        "index.uuid" | "index.creation_date" | "index.provided_name" | "index.resize.source.name" | "index.resize.source.uuid"
            | "index.routing.allocation.initial_recovery._id" | "index.number_of_routing_shards" | "index.history.uuid"
    )
}

/// `POST /{source}/_{clone|shrink|split}/{target}`.
pub fn resize(node: &Node, source: &str, target: &str, ty: ResizeType, body: &Value) -> Result<Value> {
    resize_with(node, source, target, ty, body, None)
}

pub fn resize_with(node: &Node, source: &str, target: &str, ty: ResizeType, body: &Value, copy_settings: Option<bool>) -> Result<Value> {
    if copy_settings == Some(false) {
        return Err(EsError::illegal_argument("parameter [copy_settings] can not be explicitly set to [false]"));
    }
    let target = crate::node::resolve_date_math(target)?;
    let mut max_primary_shard_size: Option<u64> = None;
    let mut req_settings = FlatSettings::new();
    let mut req_aliases = Map::new();
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            match k.as_str() {
                "settings" => req_settings = settings::normalize_index_settings(v),
                "aliases" => req_aliases = v.as_object().cloned().unwrap_or_default(),
                "max_primary_shard_size" => {
                    let s = match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    max_primary_shard_size = Some(super::parse_size(&s, "max_primary_shard_size")?.0);
                }
                "mappings" => return Err(EsError::x_content_parse("[1:2] [resize_request] unknown field [mappings]")),
                other => return Err(EsError::x_content_parse(format!("[1:2] [resize_request] unknown field [{other}]"))),
            }
        }
    }
    // ResizeRequest.validate
    let requested = int_setting(&req_settings, "index.number_of_shards")?;
    if ty == ResizeType::Split && requested.is_none() {
        return Err(EsError::action_validation(&["index.number_of_shards is required for split operations"]));
    }
    if max_primary_shard_size.is_some() && ty != ResizeType::Shrink {
        return Err(EsError::action_validation(&["Unsupported parameter [max_primary_shard_size]"]));
    }
    if req_settings.keys().any(|k| k.starts_with("index.sort.")) {
        return Err(EsError::action_validation(&["can't override index sort when resizing an index"]));
    }
    validate_target_name(&target)?;
    let svc = node.get_index(source).ok_or_else(|| EsError::index_not_found(source))?;
    let src_shards: i64 = svc.setting("index.number_of_shards").and_then(|s| s.parse().ok()).unwrap_or(1);

    // prepareCreateIndexRequest: target shard count + calculator validation
    let target_shards = match ty {
        ResizeType::Shrink => {
            if requested.is_some() && max_primary_shard_size.is_some() {
                return Err(EsError::illegal_argument("Cannot set both index.number_of_shards and max_primary_shard_size for the target index"));
            }
            match (requested, max_primary_shard_size) {
                (Some(n), _) => n,
                (None, Some(max)) => {
                    let size = svc.local_shards().iter().map(|s| s.shard.store_size()).max().unwrap_or(0);
                    let mut min = (size / max.max(1)) as i64;
                    if (min as u64) * max < size {
                        min += 1;
                    }
                    if min > src_shards {
                        src_shards
                    } else {
                        acceptable_shrink_shards(src_shards, min)
                    }
                }
                (None, None) => 1,
            }
        }
        ResizeType::Clone => requested.unwrap_or(src_shards),
        ResizeType::Split => requested.unwrap_or(src_shards),
    };
    match ty {
        ResizeType::Shrink => {
            if src_shards < target_shards {
                return Err(EsError::illegal_argument(format!(
                    "the number of target shards [{target_shards}] must be less that the number of source shards [{src_shards}]"
                )));
            }
            routing_factor(src_shards, target_shards)?;
            let (docs, deleted) = svc.doc_count();
            if docs + deleted > i32::MAX as u64 - 128 {
                return Err(EsError::new(500, "illegal_state_exception", format!("Can't merge index with more than [{}] docs - too many documents in shards [[{source}][0]]", i32::MAX - 128)));
            }
        }
        ResizeType::Clone => {
            if src_shards != target_shards {
                return Err(EsError::illegal_argument(format!(
                    "the number of target shards ({target_shards}) must be the same as the number of  source shards ( {src_shards})"
                )));
            }
        }
        ResizeType::Split => {
            routing_factor(src_shards, target_shards)?;
            if src_shards > target_shards {
                return Err(EsError::illegal_argument(format!(
                    "the number of source shards [{src_shards}] must be less that the number of target shards [{target_shards}]"
                )));
            }
            let rs = if src_shards == 1 { target_shards } else { routing_shards(&svc, src_shards) };
            if rs % target_shards != 0 {
                return Err(EsError::new(500, "illegal_state_exception", format!(
                    "the number of routing shards [{rs}] must be a multiple of the target shards [{target_shards}]"
                )));
            }
        }
    }
    if req_settings.contains_key("index.routing_partition_size") {
        return Err(EsError::illegal_argument("cannot provide a routing partition size value when resizing an index"));
    }
    if req_settings.contains_key("index.number_of_routing_shards") && !(ty == ResizeType::Split && src_shards == 1) {
        return Err(EsError::illegal_argument("cannot provide index.number_of_routing_shards on resize"));
    }
    if svc.setting("index.soft_deletes.enabled").as_deref() != Some("false") && req_settings.get("index.soft_deletes.enabled").and_then(Value::as_str) == Some("false") {
        return Err(EsError::illegal_argument("Can't disable [index.soft_deletes.enabled] setting on resize"));
    }

    // MetadataCreateIndexService.validateResize
    if crate::esql::views::exists(node, &target) {
        return Err(EsError::invalid_index_name(&target, "already exists as an ESQL view"));
    }
    if let Some(existing) = node.get_index(&target) {
        return Err(EsError::index_already_exists(&target, &existing.uuid));
    }
    if let Some(ds) = node.data_stream_of_index(source) {
        if node.data_stream(&ds).map(|d| d.write_index() == source).unwrap_or(false) {
            return Err(EsError::illegal_argument(format!("cannot resize the write index [{source}] for data stream [{ds}]")));
        }
    }
    let blocked = ["index.blocks.write", "index.blocks.read_only", "index.blocks.read_only_allow_delete"].iter().any(|k| svc.setting(k).as_deref() == Some("true"));
    if !blocked {
        return Err(EsError::new(500, "illegal_state_exception", format!("index {source} must be read-only to resize index. use \"index.blocks.write=true\"")));
    }
    for k in ["index.mode", "index.mapping.source.mode", "index.recovery.use_synthetic_source", "index.sort.field", "index.sort.order", "index.sort.mode", "index.sort.missing"] {
        if req_settings.get(k).map(|v| !v.is_null()).unwrap_or(false) {
            return Err(EsError::illegal_argument(format!("can't change setting [{k}] during resize")));
        }
    }
    if ty == ResizeType::Shrink && src_shards == 1 {
        return Err(EsError::illegal_argument("can't shrink an index with only one shard"));
    }
    if !svc.is_open() {
        return Err(EsError::index_closed(&svc.name, &svc.uuid));
    }

    // prepareResizeIndexSettings (copy_settings defaults to true in 8.x)
    let src_meta = svc.meta();
    let mut flat: FlatSettings = FlatSettings::new();
    for (k, v) in &src_meta.settings {
        if not_copyable(k) || req_settings.contains_key(k) {
            continue;
        }
        flat.insert(k.clone(), v.clone());
    }
    let req_keys: Vec<String> = req_settings.keys().cloned().collect();
    for (k, v) in req_settings {
        if v.is_null() {
            flat.remove(&k);
        } else {
            flat.insert(k, v);
        }
    }
    flat.insert("index.number_of_shards".into(), json!(target_shards.to_string()));
    flat.insert("index.routing_partition_size".into(), json!(svc.setting("index.routing_partition_size").unwrap_or_else(|| "1".into())));
    flat.insert("index.resize.source.name".into(), json!(svc.name));
    flat.insert("index.resize.source.uuid".into(), json!(svc.uuid));
    if ty == ResizeType::Shrink {
        flat.insert("index.routing.allocation.initial_recovery._id".into(), json!(crate::node::NODE_ID));
    }
    // ES `getIndexNumberOfRoutingShards`: a target of a multi-shard source
    // keeps the source's routing shards (only recorded when it differs
    // from the target's default).
    if src_shards > 1 {
        let rs = routing_shards(&svc, src_shards);
        if rs != default_routing_shards(target_shards) {
            flat.insert("index.number_of_routing_shards".into(), json!(rs.to_string()));
        }
    }
    if ty == ResizeType::Split && src_shards == 1 {
        if let Some(rs) = int_setting(&flat, "index.number_of_routing_shards")? {
            if rs % target_shards != 0 {
                return Err(EsError::illegal_argument(format!("the number of routing shards [{rs}] must be a multiple of the target shards [{target_shards}]")));
            }
        }
    }
    settings::validate(&flat)?;
    // The body's similarities are new; the copied source ones are not.
    crate::similarity::check_changed_ranges(&flat, &req_keys)?;
    let uuid = new_uuid();
    let created = crate::date::now_millis();
    flat.insert("index.uuid".into(), json!(uuid));
    flat.insert("index.provided_name".into(), json!(target));
    flat.insert("index.creation_date".into(), json!(created.to_string()));
    let mut aliases = Map::new();
    for (a, def) in req_aliases {
        let mut m = def.as_object().cloned().unwrap_or_default();
        if let Some(r) = m.remove("routing") {
            m.insert("index_routing".into(), r.clone());
            m.insert("search_routing".into(), r);
        }
        if node.get_index(&a).is_some() {
            return Err(EsError::new(400, "invalid_alias_name_exception", format!("Invalid alias name [{a}]: an index or data stream exists with the same name as the alias")));
        }
        aliases.insert(a, Value::Object(m));
    }

    // Copy the committed shards.
    svc.flush()?;
    svc.persist()?;
    let raw: Value = serde_json::from_slice(&std::fs::read(svc.path().join("index_meta.json"))?)?;
    let dir = node.data.join("indices").join(new_uuid());
    // Source shards of each target shard.
    // `IndexMetadata.selectSplitShard` / `selectShrinkShards`: modulo-routed
    // (ES 9-created) indices interleave, routing-shard indices use ranges.
    let modulo = svc.index_routing().modulo;
    let sources_of = |t: i64| -> Vec<u32> {
        match ty {
            ResizeType::Clone => vec![t as u32],
            ResizeType::Split if modulo => vec![t.rem_euclid(src_shards) as u32],
            ResizeType::Split => vec![(t / (target_shards / src_shards).max(1)) as u32],
            ResizeType::Shrink if modulo => (t..src_shards).step_by(target_shards.max(1) as usize).map(|x| x as u32).collect(),
            ResizeType::Shrink => {
                let f = (src_shards / target_shards).max(1);
                ((t * f)..((t + 1) * f)).map(|x| x as u32).collect()
            }
        }
    };
    let result = (|| -> Result<Arc<IndexService>> {
        std::fs::create_dir_all(&dir)?;
        for t in 0..target_shards {
            let first = sources_of(t)[0];
            let sh = svc.shard(first).ok_or_else(|| EsError::shard_not_found(&svc.name, &svc.uuid, first))?;
            // Pin the files so no commit swaps them mid-copy.
            let _files = sh.shard.files_pinned();
            copy_tree(&svc.path().join(format!("shard-{first}")), &dir.join(format!("shard-{t}")))?;
        }
        let meta = json!({
            "name": target,
            "uuid": uuid,
            "creation_date": created,
            "settings": settings::as_flat_json(&flat),
            "mappings": raw["mappings"],
            "aliases": aliases,
            "open": true,
        });
        std::fs::write(dir.join("index_meta.json"), serde_json::to_vec_pretty(&meta)?)?;
        let opened = Arc::new(IndexService::open(&dir)?);
        for t in 0..target_shards {
            let tsh = opened.shard(t as u32).ok_or_else(|| EsError::shard_not_found(&target, &uuid, t as u32))?;
            match ty {
                ResizeType::Clone => {}
                ResizeType::Split => {
                    // Keep only the documents that route to this target shard.
                    tsh.retain_docs(&|id, routing| opened.route(id, routing).map(|s| s as i64 == t).unwrap_or(true))?;
                    tsh.shard.flush()?;
                }
                ResizeType::Shrink => {
                    for s in &sources_of(t)[1..] {
                        let src = svc.shard(*s).ok_or_else(|| EsError::shard_not_found(&svc.name, &svc.uuid, *s))?;
                        tsh.copy_docs_from(&src, &|_, _| true)?;
                    }
                    tsh.shard.flush()?;
                }
            }
        }
        // ES opens each target shard on the recovered Lucene index, so
        // a search sees every copied document right away.
        opened.refresh()?;
        node.register_index(opened.clone())?;
        Ok(opened)
    })();
    match result {
        Ok(_) => Ok(json!({"acknowledged": true, "shards_acknowledged": true, "index": target})),
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dir);
            Err(e)
        }
    }
}

/// Recursive copy; files of committed core generations (immutable once
/// written: every core write is tmp + rename) are hard-linked.
pub(crate) fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let p = e.path();
        let name = e.file_name();
        let to = dst.join(&name);
        if p.is_dir() {
            copy_tree(&p, &to)?;
            continue;
        }
        let in_core = src.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("core-")).unwrap_or(false);
        let lname = name.to_string_lossy();
        if lname.ends_with(".tmp") {
            continue;
        }
        if !(in_core && std::fs::hard_link(&p, &to).is_ok()) {
            match std::fs::copy(&p, &to) {
                Ok(_) => {}
                // Removed meanwhile (a seal's or merge's temporary file, still
                // encoding while the files are pinned): no commit refers to it.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_math() {
        assert_eq!(acceptable_shrink_shards(12, 5), 6);
        assert_eq!(acceptable_shrink_shards(8, 2), 2);
        assert_eq!(default_routing_shards(1), 1024);
        assert_eq!(default_routing_shards(2), 1024);
        assert_eq!(default_routing_shards(5), 640);
        assert!(routing_factor(2, 6).is_ok());
        assert_eq!(routing_factor(4, 3).unwrap_err().reason, "the number of source shards [4] must be a multiple of [3]");
    }

    #[test]
    fn clone_shrink_split() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index("source", &json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}})).unwrap();
        svc.write(
            vec![crate::index::WriteReq::Index { id: Some("1".into()), source: r#"{"foo":"hello world"}"#.into(), routing: None, cond: Default::default() }],
            true,
        );
        let e = resize(&node, "source", "t", ResizeType::Clone, &json!({})).unwrap_err();
        assert_eq!(e.kind, "illegal_state_exception");
        let e = resize(&node, "source", "t", ResizeType::Clone, &json!({"settings": {"index.number_of_shards": 6}})).unwrap_err();
        assert_eq!(e.kind, "illegal_argument_exception");
        svc.put_settings(settings::normalize_index_settings(&json!({"index.blocks.write": true})), false).unwrap();
        resize(&node, "source", "target", ResizeType::Clone, &json!({})).unwrap();
        let t = node.get_index("target").unwrap();
        assert!(t.get("1").unwrap().is_some());
        assert_eq!(t.setting("index.number_of_shards").as_deref(), Some("2"));
        assert_eq!(t.setting("index.blocks.write").as_deref(), Some("true"));
        resize(&node, "source", "shrunk", ResizeType::Shrink, &json!({})).unwrap();
        assert_eq!(node.get_index("shrunk").unwrap().setting("index.number_of_shards").as_deref(), Some("1"));
        resize(&node, "source", "split", ResizeType::Split, &json!({"settings": {"index.number_of_shards": 4}})).unwrap();
        assert!(node.get_index("split").unwrap().get("1").unwrap().is_some());
        let e = resize(&node, "source", "x", ResizeType::Split, &json!({"settings": {"index.number_of_shards": 6}})).unwrap_err();
        assert_eq!(e.kind, "illegal_state_exception");
        // the source keeps working independently of its copies
        drop(t);
        drop(node);
        let node = Node::open(tmp.path()).unwrap();
        assert!(node.get_index("target").unwrap().get("1").unwrap().is_some());
    }
}
