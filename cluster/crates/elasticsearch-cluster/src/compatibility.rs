//! Internal segment format compatibility, independent of the advertised ES
//! product version. Unadvertised rolling-upgrade peers are limited to the
//! established v11 format; malformed advertisements never grant v12 support.
//!
//! The cluster's segment format floor (as ES upgrades on-disk formats once
//! every node can read them, and refuses a downgrade): a metadata custom
//! ([`SEGMENT_FORMAT_CUSTOM`]) owned by the elected master, which raises it
//! to the lowest segment read maximum over every member (data and
//! master-eligible) whenever that exceeds it, and never lowers it. Nodes
//! write `min(floor, own read max)`; a node reading less than the floor
//! cannot join, vote or be published to. The floor is a metadata custom,
//! not a new field: customs are an opaque map every build round-trips
//! (old masters carry unknown customs over, the wire diff and the gateway
//! store them as JSON), and it only exceeds v11 once no member is a
//! pre-advertisement build, so no older decoder ever has to understand it.
use crate::state::{ClusterState, DiscoveryNode, IndexMetadata, Metadata};

pub const CAPABILITIES_SCHEMA: &str = "rustic.internal.capabilities_schema";
pub const SEGMENT_READ_MAX: &str = "rustic.internal.segment_read_max";
pub const SEGMENT_WRITE: &str = "rustic.internal.segment_write";
/// Metadata custom holding the cluster's segment format floor:
/// `{"floor": N}`. Absent means the baseline ([`BASELINE_SEGMENT_FORMAT`]).
pub const SEGMENT_FORMAT_CUSTOM: &str = "_rustic_segment_format";
/// The format every build of this line reads: the read maximum assumed for
/// a node without (or with malformed) segment advertisements, and the
/// floor of a cluster that never negotiated one.
pub const BASELINE_SEGMENT_FORMAT: u32 = 11;

pub fn segment_formats(node: &DiscoveryNode) -> Result<(u32, u32), String> {
    let attrs = &node.attributes;
    if !attrs.contains_key(CAPABILITIES_SCHEMA) && !attrs.contains_key(SEGMENT_READ_MAX) && !attrs.contains_key(SEGMENT_WRITE) {
        return Ok((11, 11));
    }
    if attrs.get(CAPABILITIES_SCHEMA).map(String::as_str) != Some("1") {
        return Err(format!("node [{}] has an unsupported segment capability schema", node.name));
    }
    let parse = |key| attrs.get(key).and_then(|v| v.parse::<u32>().ok())
        .ok_or_else(|| format!("node [{}] has invalid [{key}]", node.name));
    let read = parse(SEGMENT_READ_MAX)?;
    let write = parse(SEGMENT_WRITE)?;
    if read < 11 || write < 11 || write > read {
        return Err(format!("node [{}] advertises inconsistent segment formats read={read}, write={write}", node.name));
    }
    Ok((read, write))
}

/// The segment format whose readers know DiskBBQ IVF files and mutable
/// graph code log chunks (`rustic_core::codec::VECTOR_FILES_SINCE`; this
/// crate does not depend on the codec, a node test pins them equal).
pub const VECTOR_FILES_SINCE: u32 = 13;

/// The segment format every holder of a copy of `index` must read, beyond
/// what its writers write. A `bbq_disk` (DiskBBQ) field is stored as IVF
/// files whatever the cluster's floor, so its copies need a v13 reader from
/// the start. Code log chunks need none of their own: only format 2 graph
/// checkpoints reference them, and those are written once the floor (which
/// every member reads) reached v13. Derived from the mapping, so it needs no
/// new metadata an older master would drop.
pub fn index_required_reader(index: &IndexMetadata) -> u32 {
    fn disk_bbq(value: &serde_json::Value, depth: usize) -> bool {
        let Some(object) = value.as_object() else { return false };
        let options = object.get("index_options");
        // dense_vector, and semantic_text's `index_options.dense_vector`.
        let ty = |o: Option<&serde_json::Value>| o.and_then(|o| o.get("type")).and_then(serde_json::Value::as_str) == Some("bbq_disk");
        if ty(options) || ty(options.and_then(|o| o.get("dense_vector"))) { return true; }
        depth < 64 && object.values().any(|child| disk_bbq(child, depth + 1))
    }
    if disk_bbq(&index.mappings, 0) { VECTOR_FILES_SINCE } else { BASELINE_SEGMENT_FORMAT }
}

const REFUSAL_TAG: &str = "segment_format_refusal[v";

/// Tag a refusal to place a copy on a node that cannot read segment format
/// `required`. A failed shard carrying it is not an allocation failure:
/// the master does not count it toward `max_retry`, and the
/// `segment_format` decider requires a `required` reader for the copy from
/// then on ([`refused_segment_format`]). Older masters count it, but keep
/// the text in the copy's unassigned details, so an upgraded master still
/// honours it.
pub fn segment_format_refusal(required: u32, reason: &str) -> String {
    format!("{REFUSAL_TAG}{required}]: {reason}")
}

/// How a later failure's details keep an earlier refusal's tag
/// ([`carry_segment_format_refusal`]).
const CARRIED_TAG: &str = "after an earlier segment_format_refusal[v";

/// Whether `details` describe a failure that *was* a segment format refusal
/// (not one that merely carries an earlier refusal's tag).
pub fn is_segment_format_refusal(details: &str) -> bool {
    refused_segment_format(details).is_some() && !details.contains(CARRIED_TAG)
}

/// The details of a failure that follows a tagged one (`previous`), keeping
/// the tag so the `segment_format` decider still steers the copy: an older
/// master counted every refusal, and a later failure (the target restarting
/// with an upgraded binary while its copy initialized, say) would otherwise
/// replace the only evidence that those counts were refusals.
pub fn carry_segment_format_refusal(previous: Option<&str>, details: String) -> String {
    match previous.and_then(refused_segment_format) {
        Some(required) if refused_segment_format(&details).is_none() => format!("{details}; {CARRIED_TAG}{required}]"),
        _ => details,
    }
}

/// The segment format a tagged refusal ([`segment_format_refusal`]) in
/// `text` requires.
pub fn refused_segment_format(text: &str) -> Option<u32> {
    let rest = &text[text.find(REFUSAL_TAG)? + REFUSAL_TAG.len()..];
    rest[..rest.find(']')?].parse().ok()
}

pub fn require_segment_reader(target: &DiscoveryNode, required: u32) -> Result<(), String> {
    let (read, _) = segment_formats(target)?;
    if read < required {
        return Err(format!("node [{}] reads segment formats through v{read}, but this shard requires v{required}", target.name));
    }
    Ok(())
}

/// No data member may advertise a write version another data member cannot
/// read. Advertised write versions are what each node wrote at startup (its
/// persisted floor, capped), so with the cluster floor ([`require_segment_floor`])
/// in force this holds by construction; it still catches inconsistent
/// advertisements.
pub fn validate_segment_members(state: &ClusterState) -> Result<(), String> {
    let mut required = 11;
    for node in state.data_nodes() { required = required.max(segment_formats(node)?.1); }
    for node in state.data_nodes() { require_segment_reader(node, required)?; }
    Ok(())
}

/// The persisted floor, if the custom is present and well formed.
pub fn segment_format_floor(md: &Metadata) -> Option<u32> {
    md.customs.get(SEGMENT_FORMAT_CUSTOM)?.get("floor")?.as_u64().and_then(|v| u32::try_from(v).ok())
}

/// The floor in force: the persisted one, else the baseline.
pub fn effective_segment_format_floor(md: &Metadata) -> u32 {
    segment_format_floor(md).unwrap_or(BASELINE_SEGMENT_FORMAT).max(BASELINE_SEGMENT_FORMAT)
}

/// The segment read maximum a node advertises; the baseline when it
/// advertises none or a malformed set (never more than it proves).
pub fn advertised_read_max(node: &DiscoveryNode) -> u32 {
    segment_formats(node).map_or(BASELINE_SEGMENT_FORMAT, |(read, _)| read)
}

/// The floor every current member can read: the lowest read maximum over
/// all nodes in `state` (`read_max` says what each one is trusted to read),
/// and never below the baseline.
pub fn negotiated_segment_format(state: &ClusterState, read_max: impl FnMut(&DiscoveryNode) -> u32) -> u32 {
    state.nodes.values().map(read_max).min().unwrap_or(BASELINE_SEGMENT_FORMAT).max(BASELINE_SEGMENT_FORMAT)
}

/// Raise `state`'s floor to `target` when that exceeds it (never lowers).
/// Returns whether it changed; the metadata version is bumped then.
pub fn raise_segment_format(state: &mut ClusterState, target: u32) -> bool {
    if target <= effective_segment_format_floor(&state.metadata) {
        return false;
    }
    state.metadata.customs.insert(SEGMENT_FORMAT_CUSTOM.to_string(), serde_json::json!({ "floor": target }));
    state.metadata.version += 1;
    true
}

/// A node may be a member only if it reads the cluster's floor: once the
/// floor rose past what an older build reads, that build cannot rejoin
/// (ES refuses a node older than the cluster's minimum the same way).
pub fn require_segment_floor(state: &ClusterState, node: &DiscoveryNode) -> Result<(), String> {
    let floor = effective_segment_format_floor(&state.metadata);
    if floor <= BASELINE_SEGMENT_FORMAT {
        return Ok(());
    }
    let read = segment_formats(node)?.0;
    if read < floor {
        return Err(format!(
            "node [{}] reads segment formats through v{read}, but this cluster writes segment format v{floor}; downgrading a node below the cluster's segment format is not supported",
            node.name
        ));
    }
    Ok(())
}

/// A publication may not lower the floor of the same cluster. `old` must be
/// a committed state (the leader's own base for its tasks); a follower
/// compares against the floor it applied from committed states
/// ([`validate_committed_segment_floor`]), never its last accepted state.
pub fn validate_segment_floor_publication(old: &ClusterState, next: &ClusterState) -> Result<(), String> {
    validate_committed_segment_floor(&old.metadata.cluster_uuid, effective_segment_format_floor(&old.metadata), next)
}

/// A publication may not lower `was`, the floor this node applied from a
/// committed state of cluster `cluster_uuid`. Only a committed floor binds:
/// an accepted but uncommitted raise may legitimately be superseded by a new
/// leader's publication without it, and refusing that would keep this node
/// out of the cluster (churning elections when it is needed for a quorum).
pub fn validate_committed_segment_floor(cluster_uuid: &str, was: u32, next: &ClusterState) -> Result<(), String> {
    if cluster_uuid.is_empty() || cluster_uuid != next.metadata.cluster_uuid {
        return Ok(());
    }
    let now = effective_segment_format_floor(&next.metadata);
    if now < was {
        return Err(format!("publication lowers the cluster's segment format from v{was} to v{now}; the segment format is never downgraded"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn node(id: &str, read: Option<u32>) -> DiscoveryNode {
        let mut attributes = BTreeMap::new();
        if let Some(read) = read {
            attributes.insert(CAPABILITIES_SCHEMA.to_string(), "1".to_string());
            attributes.insert(SEGMENT_READ_MAX.to_string(), read.to_string());
            attributes.insert(SEGMENT_WRITE.to_string(), "11".to_string());
        }
        DiscoveryNode { id: id.into(), name: id.into(), ephemeral_id: format!("{id}-e"),
            transport_address: "127.0.0.1:9300".into(), http_address: String::new(),
            host: "localhost".into(), ip: "127.0.0.1".into(), roles: crate::state::Role::defaults(),
            attributes, version: "9.5.5".into() }
    }

    fn state(nodes: &[DiscoveryNode]) -> ClusterState {
        let mut s = ClusterState::empty("c");
        s.metadata.cluster_uuid = "u".into();
        for n in nodes { s.nodes.insert(n.id.clone(), n.clone()); }
        s
    }

    #[test]
    fn floor_is_the_minimum_read_max_and_missing_attributes_count_as_v11() {
        let all_new = state(&[node("a", Some(13)), node("b", Some(13))]);
        assert_eq!(negotiated_segment_format(&all_new, advertised_read_max), 13);
        let mixed = state(&[node("a", Some(13)), node("b", Some(12))]);
        assert_eq!(negotiated_segment_format(&mixed, advertised_read_max), 12);
        let with_old = state(&[node("a", Some(13)), node("old", None)]);
        assert_eq!(negotiated_segment_format(&with_old, advertised_read_max), BASELINE_SEGMENT_FORMAT, "a pre-advertisement node counts as v11");
        let mut malformed = node("m", Some(13));
        malformed.attributes.insert(CAPABILITIES_SCHEMA.into(), "2".into());
        assert_eq!(advertised_read_max(&malformed), BASELINE_SEGMENT_FORMAT, "a malformed advertisement grants nothing");
        assert_eq!(negotiated_segment_format(&state(&[]), advertised_read_max), BASELINE_SEGMENT_FORMAT);
        // An unverified claim counts as what the caller trusts.
        assert_eq!(negotiated_segment_format(&all_new, |n| if n.id == "b" { 11 } else { advertised_read_max(n) }), 11);
    }

    #[test]
    fn floor_rises_and_never_lowers() {
        let mut s = state(&[node("a", Some(13)), node("old", None)]);
        let negotiated = negotiated_segment_format(&s, advertised_read_max);
        assert!(!raise_segment_format(&mut s, negotiated), "mixed cluster stays at the baseline");
        assert!(!s.metadata.customs.contains_key(SEGMENT_FORMAT_CUSTOM), "the baseline is never written");
        s.nodes.remove("old");
        let version = s.metadata.version;
        let negotiated = negotiated_segment_format(&s, advertised_read_max);
        assert!(raise_segment_format(&mut s, negotiated));
        assert_eq!(segment_format_floor(&s.metadata), Some(13));
        assert_eq!(s.metadata.version, version + 1);
        // A v12-only node arriving afterwards does not lower it.
        s.nodes.insert("b".into(), node("b", Some(12)));
        let negotiated = negotiated_segment_format(&s, advertised_read_max);
        assert!(!raise_segment_format(&mut s, negotiated));
        assert_eq!(effective_segment_format_floor(&s.metadata), 13);
        let mut lowered = s.clone();
        lowered.metadata.customs.insert(SEGMENT_FORMAT_CUSTOM.into(), serde_json::json!({"floor": 12}));
        assert!(validate_segment_floor_publication(&s, &lowered).is_err());
        lowered.metadata.customs.remove(SEGMENT_FORMAT_CUSTOM);
        assert!(validate_segment_floor_publication(&s, &lowered).is_err(), "dropping the custom lowers it too");
        assert!(validate_segment_floor_publication(&lowered, &s).is_ok());
        // Against a committed floor: an uncommitted raise binds nothing.
        assert!(validate_committed_segment_floor("u", BASELINE_SEGMENT_FORMAT, &lowered).is_ok(), "a superseded uncommitted raise");
        assert!(validate_committed_segment_floor("u", 13, &lowered).is_err());
        assert!(validate_committed_segment_floor("other", 13, &lowered).is_ok(), "another cluster's floor");
        assert!(validate_committed_segment_floor("", 13, &lowered).is_ok());
    }

    #[test]
    fn disk_bbq_mappings_require_the_vector_file_reader() {
        let index = |mappings: serde_json::Value| -> IndexMetadata {
            let mut meta: IndexMetadata = serde_json::from_value(serde_json::json!({
                "name": "i", "uuid": "u-i", "version": 1, "mapping_version": 1,
                "settings_version": 1, "aliases_version": 1, "state": "open",
                "settings": {}, "mappings": {}, "aliases": {}, "number_of_shards": 1,
                "number_of_replicas": 1, "routing_num_shards": 1, "primary_terms": [1],
                "in_sync_allocation_ids": {}, "creation_date": 0
            })).unwrap();
            meta.mappings = mappings.into();
            meta
        };
        let vector = |ty: &str| serde_json::json!({"properties": {"obj": {"properties": {"v": {"type": "dense_vector", "dims": 64, "index_options": {"type": ty}}}}}});
        assert_eq!(index_required_reader(&index(vector("bbq_disk"))), VECTOR_FILES_SINCE, "nested bbq_disk field");
        assert_eq!(index_required_reader(&index(vector("bbq_hnsw"))), BASELINE_SEGMENT_FORMAT);
        assert_eq!(index_required_reader(&index(serde_json::json!({"properties": {"n": {"type": "long"}}}))), BASELINE_SEGMENT_FORMAT);
        let semantic = serde_json::json!({"properties": {"s": {"type": "semantic_text", "index_options": {"dense_vector": {"type": "bbq_disk"}}}}});
        assert_eq!(index_required_reader(&index(semantic)), VECTOR_FILES_SINCE);

        let tagged = format!("failed recovery: [recovery_failed_exception] {}", segment_format_refusal(13, "node [old] reads segment formats through v12"));
        assert_eq!(refused_segment_format(&tagged), Some(13));
        assert_eq!(refused_segment_format("failed recovery: disk full"), None);
        assert_eq!(refused_segment_format("segment_format_refusal[vx]"), None);
        assert!(is_segment_format_refusal(&tagged));
        let carried = carry_segment_format_refusal(Some(&tagged), "failed shard on node [n0]: node restarted".into());
        assert_eq!(refused_segment_format(&carried), Some(13), "{carried}");
        assert!(!is_segment_format_refusal(&carried), "a carried tag is not a refusal");
        assert_eq!(carry_segment_format_refusal(Some(&carried), "again".into()), "again; after an earlier segment_format_refusal[v13]");
        assert_eq!(carry_segment_format_refusal(None, "plain".into()), "plain");
        assert_eq!(carry_segment_format_refusal(Some(&tagged), tagged.clone()), tagged, "a new refusal keeps its own tag");
    }

    #[test]
    fn members_below_the_floor_are_refused() {
        let mut s = state(&[node("a", Some(13))]);
        // Before the floor rises, old builds still join (mixed cluster).
        assert!(require_segment_floor(&s, &node("old", None)).is_ok());
        raise_segment_format(&mut s, 13);
        let err = require_segment_floor(&s, &node("old", None)).unwrap_err();
        assert!(err.contains("downgrading") && err.contains("v13"), "{err}");
        assert!(require_segment_floor(&s, &node("v12", Some(12))).is_err());
        assert!(require_segment_floor(&s, &node("new", Some(13))).is_ok());
        let mut malformed = node("m", Some(13));
        malformed.attributes.remove(SEGMENT_WRITE);
        assert!(require_segment_floor(&s, &malformed).is_err());
    }
}
