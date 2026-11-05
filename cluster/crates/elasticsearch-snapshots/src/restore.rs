//! Restore request parsing/validation and planning. The node crate executes
//! the plan: creates (or reopens) each target index from the planned
//! metadata and calls [`SnapshotsService::restore_shard`] for every shard.

use crate::error::{Result, SnapshotError};
use crate::service::SnapshotsService;
use crate::snapshot::SnapshotInfo;
use crate::util::{resolve_names, simple_match};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// Flatten `metadata.settings` into `index.*` keys (ES normalises settings
/// keys with the `index.` prefix).
/// Whether an index keeps its complete `_source` (a source-only snapshot
/// needs it): not synthetic or disabled, by index setting or mapping, and
/// not filtered by `_source.includes` / `excludes`.
pub fn has_complete_source(index_metadata: &Value) -> bool {
    let mode = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_ascii_lowercase);
    if matches!(mode(flat_settings(index_metadata).get("index.mapping.source.mode")).as_deref(), Some("synthetic" | "disabled")) {
        return false;
    }
    let mappings = index_metadata.get("mappings").cloned().unwrap_or(Value::Null);
    let mappings = mappings.get("_doc").cloned().unwrap_or(mappings);
    let Some(src) = mappings.get("_source") else { return true };
    let disabled = matches!(src.get("enabled"), Some(Value::Bool(false))) || src.get("enabled").and_then(Value::as_str) == Some("false");
    let filtered = ["includes", "excludes"].iter().any(|k| src.get(*k).and_then(Value::as_array).is_some_and(|a| !a.is_empty()));
    !disabled && !filtered && !matches!(mode(src.get("mode")).as_deref(), Some("synthetic" | "disabled"))
}

pub fn flat_settings(index_metadata: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(s) = index_metadata.get("settings") {
        flatten_into("", s, &mut out);
    }
    normalize_index_keys(out)
}

fn flatten_into(prefix: &str, v: &Value, out: &mut Map<String, Value>) {
    match v {
        Value::Object(m) => {
            for (k, v) in m {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten_into(&key, v, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

fn normalize_index_keys(m: Map<String, Value>) -> Map<String, Value> {
    m.into_iter()
        .map(|(k, v)| if k.starts_with("index.") { (k, v) } else { (format!("index.{k}"), v) })
        .collect()
}

/// Parsed `POST _snapshot/{repo}/{snap}/_restore` body.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RestoreRequest {
    pub indices: Vec<String>,
    pub ignore_unavailable: bool,
    pub allow_no_indices: bool,
    pub include_global_state: bool,
    pub partial: bool,
    pub include_aliases: bool,
    pub rename_pattern: Option<String>,
    pub rename_replacement: Option<String>,
    /// Flat `index.*` settings to set on restored indices.
    pub index_settings: Map<String, Value>,
    pub ignore_index_settings: Vec<String>,
    pub feature_states: Vec<String>,
}

fn list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(),
        _ => vec![],
    }
}

fn boolean(k: &str, v: &Value) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" => Ok(true),
        Value::String(s) if s == "false" => Ok(false),
        _ => Err(SnapshotError::illegal_argument(format!("malformed {k} value, expected boolean"))),
    }
}

impl RestoreRequest {
    pub fn from_json(body: Option<&Value>) -> Result<Self> {
        let mut r = RestoreRequest { allow_no_indices: true, include_aliases: true, ..Default::default() };
        let Some(obj) = body.and_then(|b| b.as_object()) else { return Ok(r) };
        for (k, v) in obj {
            match k.as_str() {
                "indices" => r.indices = list(v),
                "ignore_unavailable" => r.ignore_unavailable = boolean(k, v)?,
                "allow_no_indices" => r.allow_no_indices = boolean(k, v)?,
                "expand_wildcards" => {}
                "include_global_state" => r.include_global_state = boolean(k, v)?,
                "partial" => r.partial = boolean(k, v)?,
                "include_aliases" => r.include_aliases = boolean(k, v)?,
                "rename_pattern" => {
                    r.rename_pattern = Some(v.as_str().ok_or_else(|| SnapshotError::illegal_argument("malformed rename_pattern"))?.into())
                }
                "rename_replacement" => {
                    r.rename_replacement =
                        Some(v.as_str().ok_or_else(|| SnapshotError::illegal_argument("malformed rename_replacement"))?.into())
                }
                "index_settings" => {
                    let m = v.as_object().ok_or_else(|| SnapshotError::illegal_argument("malformed index_settings section"))?;
                    let mut flat = Map::new();
                    flatten_into("", &Value::Object(m.clone()), &mut flat);
                    r.index_settings = normalize_index_keys(flat);
                }
                "ignore_index_settings" => {
                    r.ignore_index_settings = match v {
                        Value::String(_) | Value::Array(_) => list(v),
                        _ => return Err(SnapshotError::illegal_argument("malformed ignore_index_settings section, should be an array of strings")),
                    }
                    .into_iter()
                    .map(|k| if k.starts_with("index.") { k } else { format!("index.{k}") })
                    .collect()
                }
                "feature_states" => r.feature_states = list(v),
                "storage" | "quiet" => {}
                other => return Err(SnapshotError::illegal_argument(format!("Unknown parameter {other}"))),
            }
        }
        Ok(r)
    }
}

/// Convert a Java `Matcher.replaceAll` replacement (`$1`, `${name}`, `\$`)
/// to Rust `regex` syntax.
pub fn java_replacement_to_rust(rep: &str, group_count: usize) -> Result<String> {
    let mut out = String::new();
    let chars: Vec<char> = rep.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' {
            i += 1;
            match chars.get(i) {
                Some('$') => out.push_str("$$"),
                Some(c) => out.push(*c),
                None => return Err(SnapshotError::illegal_argument("character to be escaped is missing")),
            }
            i += 1;
        } else if c == '$' {
            i += 1;
            match chars.get(i) {
                Some('{') => {
                    let end = chars[i..].iter().position(|c| *c == '}').ok_or_else(|| {
                        SnapshotError::illegal_argument("named capturing group is missing trailing '}'")
                    })?;
                    let name: String = chars[i + 1..i + end].iter().collect();
                    out.push_str(&format!("${{{name}}}"));
                    i += end + 1;
                }
                Some(d) if d.is_ascii_digit() => {
                    let mut n = d.to_digit(10).unwrap() as usize;
                    i += 1;
                    // Java: keep consuming digits while the group number stays valid.
                    while let Some(d2) = chars.get(i).and_then(|c| c.to_digit(10)) {
                        let nn = n * 10 + d2 as usize;
                        if nn > group_count {
                            break;
                        }
                        n = nn;
                        i += 1;
                    }
                    if n > group_count {
                        return Err(SnapshotError::illegal_argument(format!("No group {n}")));
                    }
                    out.push_str(&format!("${{{n}}}"));
                }
                _ => return Err(SnapshotError::illegal_argument("Illegal group reference")),
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}

/// `index.replaceAll(pattern, replacement)` with Java semantics.
pub fn rename_index(name: &str, pattern: &str, replacement: &str) -> Result<String> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| SnapshotError::illegal_argument(format!("invalid rename_pattern [{pattern}]: {e}")))?;
    let rep = java_replacement_to_rust(replacement, re.captures_len() - 1)?;
    Ok(re.replace_all(name, rep.as_str()).into_owned())
}

/// State of an index with the target name in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExistingIndex {
    pub open: bool,
    pub number_of_shards: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestoreIndexPlan {
    pub source_index: String,
    pub target_index: String,
    /// Repository index id (for [`SnapshotsService::restore_shard`] callers that work by id).
    pub index_id: String,
    pub number_of_shards: u32,
    /// Shards to restore (successfully snapshotted ones).
    pub shards: Vec<u32>,
    /// Shards that failed in the snapshot (restored empty when `partial`).
    pub failed_shards: Vec<u32>,
    /// Index metadata with settings/aliases adjusted; `settings` is flat (`index.*` keys).
    pub metadata: Value,
    /// Target exists and is closed: restore replaces it in place.
    pub replaces_closed_index: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestorePlan {
    pub repository: String,
    pub snapshot: SnapshotInfo,
    pub indices: Vec<RestoreIndexPlan>,
    pub global_metadata: Option<Value>,
    pub feature_states: Vec<String>,
}

impl RestorePlan {
    /// `wait_for_completion=true` response, given the successfully restored shard count.
    pub fn response_json(&self, successful_shards: u32) -> Value {
        let total: u32 = self.indices.iter().map(|i| i.number_of_shards).sum();
        json!({"snapshot": {
            "snapshot": self.snapshot.snapshot,
            "indices": self.indices.iter().map(|i| i.target_index.clone()).collect::<Vec<_>>(),
            "shards": {"total": total, "failed": total.saturating_sub(successful_shards), "successful": successful_shards}
        }})
    }
}

const UNMODIFIABLE_SETTINGS: [&str; 5] =
    ["index.number_of_shards", "index.version.created", "index.uuid", "index.creation_date", "index.history.uuid"];
const UNREMOVABLE_EXTRA: [&str; 2] = ["index.number_of_replicas", "index.auto_expand_replicas"];

fn adjust_settings(snap_ref: &str, settings: &mut Map<String, Value>, req: &RestoreRequest) -> Result<()> {
    for ignored in &req.ignore_index_settings {
        if ignored.contains('*') {
            let keys: Vec<String> = settings
                .keys()
                .filter(|k| simple_match(ignored, k))
                .filter(|k| !UNMODIFIABLE_SETTINGS.contains(&k.as_str()) && !UNREMOVABLE_EXTRA.contains(&k.as_str()))
                .cloned()
                .collect();
            for k in keys {
                settings.remove(&k);
            }
        } else {
            if UNMODIFIABLE_SETTINGS.contains(&ignored.as_str()) || UNREMOVABLE_EXTRA.contains(&ignored.as_str()) {
                return Err(SnapshotError::snapshot_restore(snap_ref, format!("cannot remove setting [{ignored}] on restore")));
            }
            settings.remove(ignored);
        }
    }
    for (k, v) in &req.index_settings {
        if UNMODIFIABLE_SETTINGS.contains(&k.as_str()) {
            return Err(SnapshotError::snapshot_restore(snap_ref, format!("cannot modify setting [{k}] on restore")));
        }
        settings.insert(k.clone(), v.clone());
    }
    Ok(())
}

impl SnapshotsService {
    /// Validate a restore request and compute what to restore.
    /// `cluster_indices` maps existing index names to their state.
    pub async fn plan_restore(
        &self,
        repo: &str,
        snapshot: &str,
        req: &RestoreRequest,
        cluster_indices: &BTreeMap<String, ExistingIndex>,
    ) -> Result<RestorePlan> {
        let r = self.repository(repo)?;
        let data = r.repository_data().await?;
        let entry = data.snapshot_by_name(snapshot).ok_or_else(|| SnapshotError::snapshot_missing(repo, snapshot))?;
        let info = r.read_snapshot_info(&entry.uuid).await?;
        let snap_ref = info.snapshot_ref();
        let state = info.state.unwrap_or(crate::snapshot::SnapshotState::Incompatible);
        if !state.restorable() {
            return Err(SnapshotError::snapshot_restore(&snap_ref, format!("unsupported snapshot state [{state}]")));
        }

        let exprs = if req.indices.is_empty() { vec!["_all".to_string()] } else { req.indices.clone() };
        let (selected, missing) = resolve_names(&exprs, &info.indices);
        if let Some(m) = missing.first() {
            if !req.ignore_unavailable {
                return Err(SnapshotError::index_not_found(m));
            }
        }
        if selected.is_empty() && !req.allow_no_indices && !req.include_global_state {
            return Err(SnapshotError::index_not_found(&exprs.join(",")));
        }

        // Renames (collision check first, like ES).
        let mut renamed: BTreeMap<String, String> = BTreeMap::new();
        let mut targets = Vec::new();
        for index in &selected {
            let target = match (&req.rename_pattern, &req.rename_replacement) {
                (Some(p), Some(rep)) => rename_index(index, p, rep)?,
                _ => index.clone(),
            };
            if let Some(prev) = renamed.get(&target) {
                return Err(SnapshotError::snapshot_restore(
                    &snap_ref,
                    format!("indices [{index}] and [{prev}] are renamed into the same index [{target}]"),
                ));
            }
            renamed.insert(target.clone(), index.clone());
            targets.push((index.clone(), target));
        }

        let mut plans = Vec::new();
        for (index, target) in targets {
            let failed: Vec<u32> = info.failures.iter().filter(|f| f.index == index).map(|f| f.shard_id).collect();
            if !failed.is_empty() && !req.partial {
                return Err(SnapshotError::snapshot_restore(
                    &snap_ref,
                    format!("index [{index}] wasn't fully snapshotted - cannot restore"),
                ));
            }
            let ie = data.indices.get(&index).ok_or_else(|| SnapshotError::index_not_found(&index))?;
            let mut metadata = r.read_index_metadata(&data, &entry.uuid, &index).await?;
            let mut settings = flat_settings(&metadata);
            adjust_settings(&snap_ref, &mut settings, req)?;
            let number_of_shards = info
                .index_details
                .get(&index)
                .map(|d| d.shard_count)
                .filter(|n| *n > 0)
                .or_else(|| {
                    settings.get("index.number_of_shards").and_then(|v| {
                        v.as_u64().map(|n| n as u32).or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                    })
                })
                .unwrap_or(ie.shard_generations.len() as u32);
            if let Some(obj) = metadata.as_object_mut() {
                obj.insert("settings".into(), Value::Object(settings));
                if !req.include_aliases {
                    obj.remove("aliases");
                }
            }
            let mut replaces_closed = false;
            if let Some(existing) = cluster_indices.get(&target) {
                if existing.open {
                    return Err(SnapshotError::snapshot_restore(
                        &snap_ref,
                        format!(
                            "cannot restore index [{target}] because an open index with same name already exists in the cluster. Either close or delete the existing index or restore the index under a different name by providing a rename pattern and replacement name"
                        ),
                    ));
                }
                if existing.number_of_shards != number_of_shards {
                    return Err(SnapshotError::snapshot_restore(
                        &snap_ref,
                        format!(
                            "cannot restore index [{target}] with [{}] shards from a snapshot of index [{index}] with [{number_of_shards}] shards",
                            existing.number_of_shards
                        ),
                    ));
                }
                replaces_closed = true;
            }
            plans.push(RestoreIndexPlan {
                source_index: index.clone(),
                target_index: target,
                index_id: ie.id.clone(),
                number_of_shards,
                shards: (0..number_of_shards).filter(|s| !failed.contains(s)).collect(),
                failed_shards: failed,
                metadata,
                replaces_closed_index: replaces_closed,
            });
        }
        if req.include_global_state && info.include_global_state == Some(false) {
            return Err(SnapshotError::snapshot_restore(
                &snap_ref,
                "cannot restore global state since the snapshot was created without global state",
            ));
        }
        let global_metadata =
            if req.include_global_state { Some(r.read_global_metadata(&entry.uuid).await?) } else { None };
        Ok(RestorePlan {
            repository: repo.to_string(),
            snapshot: info,
            indices: plans,
            global_metadata,
            feature_states: req.feature_states.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn complete_source_detection() {
        use serde_json::json;
        assert!(has_complete_source(&json!({"settings": {"index": {"number_of_shards": "1"}}, "mappings": {"properties": {}}})));
        assert!(!has_complete_source(&json!({"settings": {"index": {"mapping": {"source": {"mode": "synthetic"}}}}})));
        assert!(!has_complete_source(&json!({"settings": {}, "mappings": {"_source": {"enabled": false}}})));
        assert!(!has_complete_source(&json!({"settings": {}, "mappings": {"_doc": {"_source": {"excludes": ["x"]}}}})));
        assert!(!has_complete_source(&json!({"settings": {}, "mappings": {"_source": {"mode": "synthetic"}}})));
        assert!(has_complete_source(&json!({"settings": {}, "mappings": {"_source": {"includes": []}}})));
    }

    use super::*;

    #[test]
    fn rename_patterns() {
        assert_eq!(rename_index("index_1", "index_(.+)", "restored_index_$1").unwrap(), "restored_index_1");
        // `$1a` in Java is group 1 followed by `a`.
        assert_eq!(rename_index("logs-2024", "logs-(\\d+)", "$1a").unwrap(), "2024a");
        assert_eq!(rename_index("abc", "(a)(b)(c)", "$3$2$1").unwrap(), "cba");
        // $12 with only 2 groups → group 1 then literal 2.
        assert_eq!(rename_index("ab", "(a)(b)", "$12").unwrap(), "a2");
        assert_eq!(rename_index("abc", "b", "\\$").unwrap(), "a$c");
        assert_eq!(rename_index("abc", "(?<x>b)", "[${x}]").unwrap(), "a[b]c");
        assert_eq!(rename_index("test", "nomatch", "x").unwrap(), "test");
        assert_eq!(rename_index("aaa", "a", "b").unwrap(), "bbb");
        assert!(rename_index("abc", "(a)", "$2").is_err());
        assert!(rename_index("abc", "(", "x").is_err());
    }

    #[test]
    fn parse_request() {
        let r = RestoreRequest::from_json(Some(&json!({
            "indices": "a,b",
            "index_settings": {"index": {"number_of_replicas": 0}, "refresh_interval": "5s"},
            "ignore_index_settings": ["index.refresh_interval"],
            "include_aliases": false
        })))
        .unwrap();
        assert_eq!(r.indices, vec!["a", "b"]);
        assert_eq!(r.index_settings.get("index.number_of_replicas"), Some(&json!(0)));
        assert_eq!(r.index_settings.get("index.refresh_interval"), Some(&json!("5s")));
        assert!(!r.include_aliases);
        let e = RestoreRequest::from_json(Some(&json!({"bogus": 1}))).unwrap_err();
        assert_eq!(e.reason, "Unknown parameter bogus");
    }

    #[test]
    fn settings_adjustment() {
        let mut s = flat_settings(&json!({"settings": {"index": {"number_of_shards": "2", "refresh_interval": "1s", "blocks": {"write": true}}}}));
        assert!(s.contains_key("index.blocks.write"));
        let req = RestoreRequest { ignore_index_settings: vec!["index.blocks.*".into()], ..Default::default() };
        adjust_settings("r:s/u", &mut s, &req).unwrap();
        assert!(!s.contains_key("index.blocks.write"));
        let req = RestoreRequest { ignore_index_settings: vec!["index.number_of_shards".into()], ..Default::default() };
        assert_eq!(
            adjust_settings("r:s/u", &mut s, &req).unwrap_err().reason,
            "[r:s/u] cannot remove setting [index.number_of_shards] on restore"
        );
        let mut m = Map::new();
        m.insert("index.number_of_shards".into(), json!(3));
        let req = RestoreRequest { index_settings: m, ..Default::default() };
        assert_eq!(
            adjust_settings("r:s/u", &mut s, &req).unwrap_err().reason,
            "[r:s/u] cannot modify setting [index.number_of_shards] on restore"
        );
    }
}
