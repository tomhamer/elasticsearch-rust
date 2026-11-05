//! Read-only native checkpoint import for paired storage measurements.
//! Original source bytes and canonical codec bytes are recorded separately:
//! a validated legacy migration may change the latter without rebuilding HNSW.
use super::*;

pub(super) fn load(directory: &Path, expected: &Value) -> AnyResult<(MutableHnsw, Value)> {
    let complete: Value = serde_json::from_slice(&fs::read(directory.join("complete.json"))?)?;
    let saved: Value = serde_json::from_slice(&fs::read(directory.join("run.json"))?)?;
    if complete["completed"] != true || saved["mode"] != "mutable" {
        return Err("checkpoint must come from a completed mutable run".into());
    }
    for field in ["growth", "churn", "churn_rounds"] {
        if saved[field] != 0 {
            return Err(format!("checkpoint import requires zero {field}").into());
        }
    }
    for field in [
        "fixture_sha256",
        "query_sha256",
        "limit",
        "dims",
        "queries",
        "seed",
        "m",
        "ef_construction",
    ] {
        if saved[field].is_null() || saved[field] != expected[field] {
            return Err(format!("checkpoint input mismatch: {field}").into());
        }
    }
    let rows = expected["limit"]
        .as_u64()
        .ok_or("missing checkpoint limit")?;
    if rows == 0 || rows > u32::MAX as u64 {
        return Err("invalid checkpoint row count".into());
    }
    let bytes = fs::read(directory.join("mutable-graph.bin"))?;
    let graph = MutableHnsw::open(&bytes)?;
    let stats = graph.stats();
    let params = graph.params();
    // The codec validates unique source keys and key <= issued watermark.
    // All-live count plus watermark rows-1 therefore proves keys are 0..rows.
    if u64::from(graph.dim()) != expected["dims"].as_u64().unwrap_or(0)
        || graph.similarity() != Similarity::Cosine
        || u64::from(graph.active_nodes()) != rows
        || u64::from(graph.num_nodes()) != rows
        || stats.issued_key_watermark != Some(rows - 1)
        || stats.pending_reservations != 0
        || stats.retired_slots != 0
        || stats.visibility_retained != 0
        || params.m as u64 != expected["m"].as_u64().unwrap_or(0)
        || params.ef_construction as u64 != expected["ef_construction"].as_u64().unwrap_or(0)
        || Some(params.seed) != expected["seed"].as_u64()
    {
        return Err("checkpoint graph does not match its all-live contiguous-key manifest".into());
    }
    let canonical = graph.encode();
    {
        let reopened = MutableHnsw::open(&canonical)?;
        if reopened.encode() != canonical {
            return Err("checkpoint canonical encode/open is not stable".into());
        }
    }
    let report = json!({
        "directory":directory,"input_sha256":hash(&bytes),"input_bytes":bytes.len(),
        "input_codec_version":bytes[4],"source_revision":saved["source_revision"],
        "canonical_sha256":hash(&canonical),"canonical_bytes":canonical.len(),
        "canonical_codec_version":canonical[4],"canonical_roundtrip_verified":true,
        "source_construction":saved["mutable_construction"],
        "scope":"read-only native checkpoint; no graph construction, mutations or production recovery claim; source hash identifies shared topology while codec migration may change canonical bytes"
    });
    Ok((graph, report))
}

/// Prefer a self-contained graph/row pair when the completed run saved one.
/// Older recorded runs retain their explicit fixture-backed restoration path.
pub(super) fn restore_owned_run(
    directory: &Path, graph: MutableHnsw, source: &(impl MutableVectorSource + ?Sized), rows: usize,
) -> AnyResult<(Arc<MutableHnsw>, MutableVectorOwner, Vec<NodeHandle>)> {
    if directory.join("mutable-checkpoint.bin").try_exists()? {
        let storage = rustic_core::codec::directory::MmapDirectory::open(directory)?;
        let (restored, owner) = MutableVectorOwner::open_file(&storage, "mutable-checkpoint.bin")?;
        if restored.encode() != graph.encode() { return Err("atomic checkpoint differs from validated graph".into()); }
        let ordinals = ordinal_handles(&restored, rows)?;
        return Ok((restored, owner, ordinals));
    }
    let row_path = directory.join("mutable-rows.bin");
    if !row_path.try_exists()? { return restore_owned(graph, source, rows); }
    // Completed POC artifacts are immutable. Map rather than read_to_end so
    // restoration does not duplicate all vector cells in an intermediate Vec.
    let graph_file = File::open(directory.join("mutable-graph.bin"))?;
    let row_file = File::open(row_path)?;
    let graph_map = unsafe { Mmap::map(&graph_file)? };
    let row_map = unsafe { Mmap::map(&row_file)? };
    drop(graph); // load() already validated this topology and its manifest.
    let (graph, owner) = MutableVectorOwner::restore_checkpoint(&graph_map, &row_map)?;
    let ordinals = ordinal_handles(&graph, rows)?;
    Ok((graph, owner, ordinals))
}

fn ordinal_handles(graph: &MutableHnsw, rows: usize) -> AnyResult<Vec<NodeHandle>> {
    let mut identities: Vec<_> = graph.try_snapshot()?.visible_identities().collect();
    identities.sort_unstable_by_key(|(key, _)| *key);
    if identities.len() != rows || identities.iter().enumerate().any(|(i, (key, _))| *key != i as u64) {
        return Err("owned checkpoint source requires contiguous live source keys".into());
    }
    Ok(identities.into_iter().map(|(_, handle)| handle).collect())
}

/// Replay uses the same owned source implementation as fresh indexing. The
/// fixture supplies row values, but query distance lookup must not bypass the
/// generation-owned directory through a different flat-source adapter.
pub(super) fn restore_owned(
    graph: MutableHnsw,
    source: &(impl MutableVectorSource + ?Sized),
    rows: usize,
) -> AnyResult<(Arc<MutableHnsw>, MutableVectorOwner, Vec<NodeHandle>)> {
    let graph = Arc::new(graph);
    let ordinals = ordinal_handles(&graph, rows)?;
    let owner = MutableVectorOwner::restore(graph.clone(), source)?;
    Ok((graph, owner, ordinals))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rows;
    impl MutableVectorSource for Rows {
        fn get(&self, key: u64) -> Option<&[f32]> {
            match key {
                0 => Some(&[1., 0.]),
                1 => Some(&[0., 1.]),
                2 => Some(&[1., 1.]),
                _ => None,
            }
        }
    }

    #[test]
    fn saved_owned_rows_restore_without_consulting_fixture_values() {
        struct UnavailableFixture;
        impl MutableVectorSource for UnavailableFixture {
            fn get(&self, _: u64) -> Option<&[f32]> { panic!("saved rows must replace fixture lookup") }
        }
        let graph = MutableHnsw::new(2, Similarity::Cosine, HnswParams::default());
        for key in 0..3 { graph.insert(key, &Rows).unwrap(); }
        let (graph, owner, _) = restore_owned(graph, &Rows, 3).unwrap();
        let owner = Arc::new(owner);
        let capture = owner.capture_checkpoint().unwrap();
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("mutable-graph.bin"), capture.graph_bytes()).unwrap();
        capture.write_rows(&mut File::create(directory.path().join("mutable-rows.bin")).unwrap()).unwrap();
        capture.write_file(&rustic_core::codec::directory::MmapDirectory::open(directory.path()).unwrap(), "mutable-checkpoint.bin").unwrap();
        // The atomic file must supply the source even when the legacy row file
        // is absent. No fallback to fixture values is permitted.
        fs::remove_file(directory.path().join("mutable-rows.bin")).unwrap();
        let (restored, owner, handles) = restore_owned_run(directory.path(),
            MutableHnsw::open(&graph.encode()).unwrap(), &UnavailableFixture, 3).unwrap();
        assert_eq!(restored.encode(), graph.encode());
        let read = owner.pin().unwrap();
        for (key, handle) in handles.into_iter().enumerate() {
            assert_eq!(read.get_node(handle, key as u64), Rows.get(key as u64));
        }
    }

    #[test]
    fn checkpoint_owned_source_preserves_topology_handles_and_exact_rows() {
        let graph = MutableHnsw::new(2, Similarity::Cosine, HnswParams::default());
        for key in 0..3 { graph.insert(key, &Rows).unwrap(); }
        let encoded = graph.encode();
        let (restored, owner, ordinals) = restore_owned(MutableHnsw::open(&encoded).unwrap(), &Rows, 3).unwrap();
        assert_eq!(restored.encode(), encoded);
        let read = owner.pin().unwrap();
        for (key, &handle) in ordinals.iter().enumerate() {
            assert_eq!(read.get_node(handle, key as u64), Rows.get(key as u64));
        }
        assert_eq!(restored.search(&[1., 0.], 3, 10, &read, |_| true).hits,
            graph.search(&[1., 0.], 3, 10, &Rows, |_| true).hits);
        assert!(restore_owned(MutableHnsw::open(&encoded).unwrap(), &Rows, 4).is_err());
    }

    #[test]
    fn checkpoint_import_preserves_results_and_rejects_wrong_corpus_or_live_set() {
        let directory = tempfile::tempdir().unwrap();
        let params = HnswParams::default();
        let graph = MutableHnsw::new(2, Similarity::Cosine, params);
        let first = graph.insert(0, &Rows).unwrap();
        graph.insert(1, &Rows).unwrap();
        let manifest = json!({"mode":"mutable","growth":0,"churn":0,"churn_rounds":0,
            "fixture_sha256":"fixture-a","query_sha256":"queries-a","limit":2,"dims":2,
            "queries":1,"seed":params.seed,"m":params.m,"ef_construction":params.ef_construction});
        fs::write(
            directory.path().join("run.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        fs::write(
            directory.path().join("complete.json"),
            b"{\"completed\":true}",
        )
        .unwrap();
        fs::write(directory.path().join("mutable-graph.bin"), graph.encode()).unwrap();
        let (loaded, provenance) = load(directory.path(), &manifest).unwrap();
        assert_eq!(
            loaded.search(&[1., 0.], 2, 10, &Rows, |_| true).hits,
            graph.search(&[1., 0.], 2, 10, &Rows, |_| true).hits
        );
        assert_eq!(provenance["canonical_roundtrip_verified"], true);
        let source_bytes = fs::read(directory.path().join("mutable-graph.bin")).unwrap();
        loaded.insert(2, &Rows).unwrap();
        assert_eq!(loaded.active_nodes(), 3);
        assert_eq!(fs::read(directory.path().join("mutable-graph.bin")).unwrap(), source_bytes);
        assert_eq!(load(directory.path(), &manifest).unwrap().0.active_nodes(), 2);
        let mut other = manifest.clone();
        other["fixture_sha256"] = json!("fixture-b");
        assert!(load(directory.path(), &other)
            .err()
            .unwrap()
            .to_string()
            .contains("fixture_sha256"));
        assert!(graph.remove(first, &Rows).unwrap());
        fs::write(directory.path().join("mutable-graph.bin"), graph.encode()).unwrap();
        assert!(load(directory.path(), &manifest).is_err());
    }
}
