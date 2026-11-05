//! On-disk layout of an Elasticsearch 8.x node (`path.data`): the persisted
//! cluster state (a Lucene index under `_state/`, written by
//! `PersistedClusterStateService`) and per-shard Lucene indices under
//! `indices/<uuid>/<shard>/index`.

use crate::lucene::{self, stored, stored::Value as SV};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Returns the directory holding `_state/` and `indices/` (the data path
/// itself on 8.x; `nodes/0` on data directories first created by 7.x).
pub fn locate_node_dir(data: &Path) -> Result<PathBuf> {
    for cand in [data.to_path_buf(), data.join("nodes").join("0")] {
        let st = cand.join("_state");
        if st.is_dir() && lucene::latest_commit_file(&st)?.is_some() {
            return Ok(cand);
        }
    }
    bail!(
        "{} does not look like an Elasticsearch 8.x data path (no _state/segments_N); pass the node's path.data",
        data.display()
    )
}

pub type Fields = Vec<(String, SV)>;

pub fn get_str<'a>(f: &'a Fields, name: &str) -> Option<&'a str> {
    f.iter().find(|(n, _)| n == name).and_then(|(_, v)| match v {
        SV::Str(s) => Some(s.as_str()),
        _ => None,
    })
}

pub fn get_bytes<'a>(f: &'a Fields, name: &str) -> Option<&'a [u8]> {
    f.iter().find(|(n, _)| n == name).and_then(|(_, v)| match v {
        SV::Bytes(b) => Some(b.as_slice()),
        _ => None,
    })
}

pub fn get_int(f: &Fields, name: &str) -> Option<i64> {
    f.iter().find(|(n, _)| n == name).and_then(|(_, v)| match v {
        SV::Int(i) => Some(*i as i64),
        SV::Long(l) => Some(*l),
        SV::Str(s) => s.parse().ok(),
        _ => None,
    })
}

#[derive(Default, Debug, Clone)]
pub struct DocStats {
    pub segments: usize,
    pub max_doc: u64,
    pub hard_deleted: u64,
    pub soft_deleted: u64,
    pub live: u64,
}

/// Opens the latest commit of a Lucene directory and calls `f` for every live
/// document's stored fields.
pub fn scan_lucene_dir(dir: &Path, mut f: impl FnMut(&Fields) -> Result<()>) -> Result<(lucene::Commit, DocStats)> {
    let (_, path) = lucene::latest_commit_file(dir)?.ok_or_else(|| anyhow!("{}: no Lucene commit (segments_N)", dir.display()))?;
    let commit = lucene::read_commit(&path)?;
    let mut st = DocStats::default();
    for sc in &commit.segments {
        let seg = lucene::open_segment(dir, sc).with_context(|| format!("{}: segment {}", dir.display(), sc.name))?;
        st.segments += 1;
        st.max_doc += seg.max_doc as u64;
        st.hard_deleted += seg.hard_deleted as u64;
        st.soft_deleted += seg.soft_deleted as u64;
        st.live += seg.live_count() as u64;
        stored::for_each_doc(&seg, |_, fields| f(&fields)).with_context(|| format!("{}: segment {}", dir.display(), sc.name))?;
    }
    Ok((commit, st))
}

pub struct ClusterState {
    pub global: Value,
    /// index uuid -> IndexMetadata x-content (`{"<name>": {...}}`)
    pub indices: BTreeMap<String, Value>,
    pub mappings: HashMap<String, Value>,
    pub user_data: BTreeMap<String, String>,
}

pub fn read_cluster_state(node_dir: &Path) -> Result<ClusterState> {
    let dir = node_dir.join("_state");
    // (type, key) -> pages
    let mut docs: BTreeMap<(String, String), Vec<(i64, Vec<u8>)>> = BTreeMap::new();
    let (commit, _) = scan_lucene_dir(&dir, |f| {
        // 8.x docs carry no type field: the key field present identifies the kind
        let ty = match get_str(f, "type") {
            Some(t) => t.to_string(),
            None if get_str(f, "index_uuid").is_some() => "index".into(),
            None if get_str(f, "mapping_hash").is_some() => "mapping".into(),
            None => "global".into(),
        };
        let key = get_str(f, "index_uuid").or_else(|| get_str(f, "mapping_hash")).unwrap_or("").to_string();
        let page = get_int(f, "page").unwrap_or(0);
        let data = get_bytes(f, "data").ok_or_else(|| anyhow!("cluster state doc of type [{}] has no data", ty))?;
        docs.entry((ty, key)).or_default().push((page, data.to_vec()));
        Ok(())
    })?;
    let mut cs = ClusterState { global: Value::Null, indices: BTreeMap::new(), mappings: HashMap::new(), user_data: commit.user_data };
    for ((ty, key), mut pages) in docs {
        pages.sort_by_key(|p| p.0);
        let data: Vec<u8> = pages.into_iter().flat_map(|p| p.1).collect();
        let v = super::decode_blob(&data).with_context(|| format!("decode cluster state doc [{}] [{}]", ty, key))?;
        match ty.as_str() {
            "global" => cs.global = v,
            "index" => {
                cs.indices.insert(key, v);
            }
            "mapping" => {
                let v = match v {
                    Value::Object(mut o) if o.len() == 1 && o.contains_key("content") => o.remove("content").unwrap(),
                    v => v,
                };
                cs.mappings.insert(key, v);
            }
            other => eprintln!("warning: ignoring cluster state doc of unknown type [{}]", other),
        }
    }
    if cs.global.is_null() {
        bail!("{}: no global metadata document", dir.display());
    }
    Ok(cs)
}

#[derive(Debug, Clone)]
pub struct IndexMeta {
    pub uuid: String,
    pub name: String,
    pub state: String,
    pub settings: Map<String, Value>,
    pub mappings: Value,
    pub aliases: Map<String, Value>,
    pub num_shards: u32,
}

fn flatten(prefix: &str, v: &Value, out: &mut Map<String, Value>) {
    match v {
        Value::Object(m) => {
            for (k, v) in m {
                let key = if prefix.is_empty() { k.clone() } else { format!("{}.{}", prefix, k) };
                flatten(&key, v, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

/// Strips the legacy single type wrapper (`{"_doc": {...}}`).
pub fn unwrap_type(m: Value) -> Value {
    if let Value::Object(o) = &m {
        if o.len() == 1 {
            if let Some(inner @ Value::Object(_)) = o.get("_doc") {
                return inner.clone();
            }
        }
    }
    m
}

pub fn first_mapping(v: Option<&Value>) -> Option<Value> {
    match v? {
        Value::Array(a) => a.iter().find(|x| x.is_object()).cloned().map(unwrap_type),
        Value::Object(o) if !o.is_empty() => Some(unwrap_type(Value::Object(o.clone()))),
        _ => None,
    }
}

pub fn index_metas(cs: &ClusterState) -> Result<Vec<IndexMeta>> {
    let mut out = Vec::new();
    for (uuid, v) in &cs.indices {
        let o = v.as_object().ok_or_else(|| anyhow!("index metadata for {} is not an object", uuid))?;
        let (name, inner) = o.iter().next().ok_or_else(|| anyhow!("empty index metadata for {}", uuid))?;
        let mut settings = Map::new();
        flatten("", inner.get("settings").unwrap_or(&Value::Null), &mut settings);
        let mut mappings = first_mapping(inner.get("mappings"));
        if mappings.is_none() {
            let hashes: Vec<&str> = match inner.get("mappings_hash") {
                Some(Value::String(s)) => vec![s.as_str()],
                Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).collect(),
                _ => vec![],
            };
            for h in hashes {
                let m = cs.mappings.get(h).ok_or_else(|| anyhow!("index {}: mapping {} missing from cluster state", name, h))?;
                mappings = first_mapping(Some(m)).or_else(|| Some(unwrap_type(m.clone())));
            }
        }
        let num_shards = settings
            .get("index.number_of_shards")
            .and_then(|v| v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_u64().map(|n| n as u32)))
            .unwrap_or(1);
        out.push(IndexMeta {
            uuid: uuid.clone(),
            name: name.clone(),
            state: inner.get("state").and_then(|s| s.as_str()).unwrap_or("open").to_string(),
            settings,
            mappings: mappings.unwrap_or_else(|| Value::Object(Map::new())),
            aliases: inner.get("aliases").and_then(|a| a.as_object()).cloned().unwrap_or_default(),
            num_shards,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Global metadata object (`{"meta-data": {...}}` unwrapped).
pub fn global_meta(cs: &ClusterState) -> &Value {
    cs.global.get("meta-data").unwrap_or(&cs.global)
}

pub fn shard_index_dir(node_dir: &Path, uuid: &str, shard: u32) -> PathBuf {
    node_dir.join("indices").join(uuid).join(shard.to_string()).join("index")
}

#[derive(Debug, Clone, Default)]
pub struct TranslogState {
    /// highest seq_no held by any translog generation that still has operations
    pub max_seq_no: i64,
    /// global checkpoint recorded by the current generation
    pub global_checkpoint: i64,
}

fn read_checkpoint(p: &Path) -> Result<(i32, i64, i64)> {
    let b = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
    let mut inp = lucene::io::In::new(&b);
    let h = lucene::io::read_header(&mut inp)?;
    if h.codec != "ckp" {
        bail!("{}: unexpected codec {}", p.display(), h.codec);
    }
    // version 4+ is little-endian, 3 big-endian
    let le = h.version >= 4;
    let rd = |i: &mut lucene::io::In| if le { i.le_i64() } else { i.be_i64() };
    let _offset = rd(&mut inp)?;
    let ops = if le { inp.le_i32()? } else { inp.be_i32()? };
    let _gen = rd(&mut inp)?;
    let _min = rd(&mut inp)?;
    let max_seq_no = rd(&mut inp)?;
    let global_checkpoint = rd(&mut inp)?;
    Ok((ops, max_seq_no, global_checkpoint))
}

/// Reads `translog.ckp` and every rolled generation's `translog-N.ckp`
/// (ES `Checkpoint`).
pub fn read_translog_checkpoint(shard_dir: &Path) -> Result<Option<TranslogState>> {
    let dir = shard_dir.join("translog");
    let current = dir.join("translog.ckp");
    if !current.exists() {
        return Ok(None);
    }
    let (ops, max, gcp) = read_checkpoint(&current)?;
    let mut st = TranslogState { max_seq_no: if ops > 0 { max } else { -1 }, global_checkpoint: gcp };
    for e in std::fs::read_dir(&dir)? {
        let name = e?.file_name().to_string_lossy().into_owned();
        if name.starts_with("translog-") && name.ends_with(".ckp") {
            let (ops, max, _) = read_checkpoint(&dir.join(&name))?;
            if ops > 0 {
                st.max_seq_no = st.max_seq_no.max(max);
            }
        }
    }
    Ok(Some(st))
}
