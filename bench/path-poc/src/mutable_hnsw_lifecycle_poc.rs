//! Standalone bounded-source lifecycle gate. Build/run on EC2 only.
//! Synthetic vectors, original f32 cosine, no Rally/production-memory claim.
mod mutable_hnsw_lifecycle_source;
#[path = "mutable_hnsw_memory.rs"]
mod graph_memory;
use mutable_hnsw_lifecycle_source::{PagedSource, PAGE_ROWS};
use rayon::prelude::*;
use rustic_core::vector::{
    hnsw::{MutableHnsw, MutableHnswLimits, MutableVectorSource, NodeHandle},
    HnswParams, Similarity,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
type AnyResult<T> = Result<T, Box<dyn Error>>;

fn vector(doc: usize, cycle: usize, dims: usize) -> Vec<f32> {
    let mut state = (doc as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
    let mut row: Vec<_> = (0..dims)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545F4914F6CDD1D);
            ((bits >> 40) as f32 / (1u32 << 24) as f32) * 2. - 1.
        })
        .collect();
    row[(doc.wrapping_mul(17) + cycle.wrapping_mul(31)) % dims] += 0.02 * (cycle % 7) as f32;
    let norm = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    for value in &mut row {
        *value = (*value as f64 / norm) as f32;
    }
    row
}
fn graph_stats(graph: &MutableHnsw) -> Value {
    let s = graph.stats();
    json!({"allocated_slots":s.allocated_slots,"active_nodes":s.active_nodes,"pending_reservations":s.pending_reservations,"reusable_slots":s.reusable_slots,"retired_slots":s.retired_slots,"visibility_retained":s.visibility_retained,"exhausted_slots":s.exhausted_slots,"issued_key_watermark":s.issued_key_watermark,"live_adjacency_bytes":s.live_adjacency_bytes,"retired_adjacency_bytes":s.retired_adjacency_bytes,"neighbor_size":s.neighbor_size,"storage":graph_memory::report(graph)})
}
fn memory() -> Value {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    json!({"proc_status":status.lines().filter(|line|["VmRSS:","VmHWM:","RssAnon:","RssFile:"].iter().any(|key|line.starts_with(key))).collect::<Vec<_>>(),"smaps_rollup":fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default()})
}
fn drain(graph: &MutableHnsw, source: &PagedSource) -> AnyResult<Value> {
    let started = Instant::now();
    let mut reclaimed = 0;
    loop {
        reclaimed += source.acknowledge(graph.collect_reclaimed())?;
        if !graph.reclamation_pending() {
            return Ok(json!({"reclaimed":reclaimed,"seconds":started.elapsed().as_secs_f64()}));
        }
        if started.elapsed().as_secs() >= 10 {
            return Err("graph epoch/source reclamation did not drain within 10 seconds".into());
        }
        std::thread::yield_now();
    }
}
fn insert_batch(
    graph: &MutableHnsw,
    source: &PagedSource,
    items: &[(usize, u64)],
    cycle: usize,
    dims: usize,
) -> AnyResult<(Vec<NodeHandle>, Value)> {
    let allocation = Instant::now();
    let mut reservations = Vec::with_capacity(items.len());
    let mut rows = Vec::with_capacity(items.len());
    for &(doc, key) in items {
        let row = source.allocate(vector(doc, cycle, dims))?;
        let reservation = graph.reserve_insert(key)?;
        rows.push((reservation.handle(), key, row));
        reservations.push(reservation);
    }
    source.install(rows)?;
    let allocate_seconds = allocation.elapsed().as_secs_f64();
    let view = source.pin();
    let prepare = Instant::now();
    let proposals = reservations
        .into_par_iter()
        .map(|reservation| graph.prepare(reservation, &view))
        .collect::<rustic_core::error::Result<Vec<_>>>()?;
    let prepare_seconds = prepare.elapsed().as_secs_f64();
    let proposal_bytes = proposals.iter().map(|p| p.estimated_bytes()).sum::<usize>();
    let commit = Instant::now();
    let handles = proposals
        .into_iter()
        .map(|proposal| graph.complete(proposal, &view))
        .collect::<rustic_core::error::Result<Vec<_>>>()?;
    Ok((
        handles,
        json!({"rows":items.len(),"allocate_and_publish_seconds":allocate_seconds,"prepare_seconds":prepare_seconds,"commit_seconds":commit.elapsed().as_secs_f64(),"prepared_bytes":proposal_bytes}),
    ))
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut an, mut bn) = (0., 0., 0.);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        an += a * a;
        bn += b * b;
    }
    (1. + dot / (an * bn).sqrt()) / 2.
}
fn audit(
    source: &PagedSource,
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
    floor: f64,
) -> AnyResult<Value> {
    let (graph, view) = source.read();
    let mut recalls = Vec::new();
    let mut details = Vec::new();
    for (qi, query) in queries.iter().enumerate() {
        let mut exact: Vec<_> = view
            .rows()
            .map(|(_, key, row)| (key, cosine(query, row)))
            .collect();
        exact.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        exact.truncate(k);
        let result = graph.search(query, k, ef, &view, |_| true);
        if result.missing_source_lookups != 0 || result.entry_fallback_scanned != 0 {
            return Err("query required missing source/entry fallback".into());
        }
        if result.hits.len() != exact.len() {
            return Err("query returned incomplete hits".into());
        }
        let mut seen = HashSet::new();
        let mut last = f32::INFINITY;
        for &(key, score) in &result.hits {
            if !score.is_finite() || score > last || !seen.insert(key) {
                return Err("invalid/duplicate/unsorted query score".into());
            }
            let row = view.get(key).ok_or("result key missing")?;
            if (score as f64 - cosine(query, row)).abs() > 2e-5 {
                return Err("incorrect original-f32 result score".into());
            }
            last = score;
        }
        let recall = result
            .hits
            .iter()
            .filter(|h| exact.iter().any(|t| t.0 == h.0))
            .count() as f64
            / exact.len() as f64;
        recalls.push(recall);
        details.push(json!({"query":qi,"recall":recall,"hits":result.hits,"truth":exact,"vector_ops":result.vector_ops}));
    }
    let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
    if mean + 1e-12 < floor {
        return Err(format!("recall {mean} below lifecycle gate {floor}").into());
    }
    Ok(
        json!({"mean_recall":mean,"min_recall":recalls.into_iter().fold(1.,f64::min),"floor":floor,"queries":details,"scope":"untimed exact f64 audit, latest graph topology and current source only"}),
    )
}
fn assert_plateau(
    graph: &MutableHnsw,
    source: &PagedSource,
    live: usize,
    initial_capacity: usize,
) -> AnyResult<()> {
    let g = graph.stats();
    let s = source.stats();
    if g.active_nodes as usize != live
        || g.pending_reservations != 0
        || g.retired_slots != 0
        || g.visibility_retained != 0
        || g.retired_adjacency_bytes != 0
    {
        return Err("graph live/pending/retired counts violate quiescent bound".into());
    }
    // Once per completed cycle, independently check the detailed stores. The
    // per-batch drain predicate must not scan all arrays on every replacement.
    let storage = graph.storage_memory();
    if storage.levels.held_payload_bytes != 0
        || storage.links.held_payload_bytes != 0
        || storage.node_held_payload_bytes != 0
        || storage.outstanding_ack_tokens != 0
    {
        return Err("graph stores or source acknowledgment tokens remain held".into());
    }
    for field in ["current_rows", "resident_rows"] {
        if s[field].as_u64() != Some(live as u64) {
            return Err(format!("source {field} did not return to live cardinality").into());
        }
    }
    if s["resident_roots"] != 1
        || s["retained_pages"] != 0
        || s["quarantined_tokens"] != 0
        || s["retained_or_pending_rows"] != 0
    {
        return Err("source readers/pages/quarantined descriptors did not drain".into());
    }
    if g.allocated_slots as usize > initial_capacity
        || s["directory_slot_capacity"].as_u64().unwrap() as usize
            > initial_capacity.div_ceil(PAGE_ROWS) * PAGE_ROWS
    {
        return Err("graph/source directory capacity grew with historical churn".into());
    }
    Ok(())
}
fn run(
    values: &BTreeMap<String, String>,
    events: &mut BufWriter<File>,
    dir: &PathBuf,
) -> AnyResult<()> {
    let number = |key: &str, default: usize| -> AnyResult<usize> {
        Ok(values
            .get(key)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let live = number("live", 100_000)?;
    let dims = number("dims", 96)?;
    let cycles = number("cycles", 10)?;
    let batch = number("batch", 64)?;
    let query_count = number("queries", 20)?;
    let k = number("k", 10)?;
    let ef = number("ef", 1000)?;
    let floor = values
        .get("recall-floor")
        .map(|v| v.parse::<f64>())
        .transpose()?
        .unwrap_or(0.95);
    if live < 100_000
        || dims == 0
        || cycles < 10
        || batch == 0
        || query_count == 0
        || k == 0
        || k > live
        || ef == 0
        || !(0.0..=1.0).contains(&floor)
    {
        return Err("performance lifecycle gate requires >=100K live rows, >=10 full cycles, positive dimensions/batch/queries/k/ef and valid floor".into());
    }
    let max_slots = live
        .checked_add(batch)
        .filter(|n| *n < u32::MAX as usize)
        .ok_or("slot limit overflow")?;
    let graph = Arc::new(MutableHnsw::with_limits(
        dims as u32,
        Similarity::Cosine,
        HnswParams {
            m: 16,
            ef_construction: 200,
            threads: number("threads", std::thread::available_parallelism()?.get())?,
            ..Default::default()
        },
        MutableHnswLimits {
            max_slots: max_slots as u32,
            max_pending: batch,
            max_retired: batch * 2,
            ..Default::default()
        },
    ));
    let source = PagedSource::new(graph.clone(), dims, live + batch, max_slots);
    let queries: Vec<_> = (0..query_count)
        .map(|i| vector(i * 997 % live, 0, dims))
        .collect();
    let mut next_key = 0u64;
    let mut handles = Vec::with_capacity(live);
    let mut keys = Vec::with_capacity(live);
    let mut emit = |event: Value| -> AnyResult<()> {
        serde_json::to_writer(&mut *events, &event)?;
        writeln!(events)?;
        events.flush()?;
        println!("{event}");
        Ok(())
    };
    emit(
        json!({"event":"configuration","live":live,"dims":dims,"cycles":cycles,"batch":batch,"queries":query_count,"k":k,"ef":ef,"recall_floor":floor,"source_revision":option_env!("RUSTIC_POC_SOURCE_REVISION"),"scope":"synthetic original-f32 lifecycle/correctness/ownership gate; no production ES, PIT, WAL, quantization, Rally or speedup claim","source":"safe Arc COW pages; graph epoch/publication captured before source root; outer root copy remains O(page count)"}),
    )?;
    let initial = Instant::now();
    for start in (0..live).step_by(batch) {
        let items: Vec<_> = (start..(start + batch).min(live))
            .map(|doc| {
                let key = next_key;
                next_key += 1;
                (doc, key)
            })
            .collect();
        let (new_handles, _) = insert_batch(&graph, &source, &items, 0, dims)?;
        handles.extend(new_handles);
        keys.extend(items.iter().map(|item| item.1));
    }
    let initial_drain = drain(&graph, &source)?;
    let initial_capacity = graph.stats().allocated_slots as usize;
    assert_plateau(&graph, &source, live, initial_capacity)?;
    fs::write(
        dir.join("audit-0.json"),
        serde_json::to_vec_pretty(&audit(&source, &queries, k, ef, floor)?)?,
    )?;
    emit(
        json!({"event":"initial","seconds":initial.elapsed().as_secs_f64(),"drain":initial_drain,"graph":graph_stats(&graph),"source":source.stats(),"memory":memory()}),
    )?;
    for cycle in 1..=cycles {
        let cycle_start = Instant::now();
        let mut remove_seconds = 0.;
        let mut drain_seconds = 0.;
        let mut allocation_seconds = 0.;
        let mut prepare_seconds = 0.;
        let mut commit_seconds = 0.;
        for start in (0..live).step_by(batch) {
            let end = (start + batch).min(live);
            let source_view = source.pin();
            let remove_start = Instant::now();
            for &handle in &handles[start..end] {
                if !graph.remove(handle, &source_view)? {
                    return Err("live removal rejected".into());
                }
            }
            remove_seconds += remove_start.elapsed().as_secs_f64();
            drop(source_view);
            drain_seconds += drain(&graph, &source)?["seconds"].as_f64().unwrap();
            let items: Vec<_> = (start..end)
                .map(|doc| {
                    let key = next_key;
                    next_key += 1;
                    (doc, key)
                })
                .collect();
            let (new_handles, phase) = insert_batch(&graph, &source, &items, cycle, dims)?;
            handles[start..end].copy_from_slice(&new_handles);
            for (offset, item) in items.iter().enumerate() {
                keys[start + offset] = item.1;
            }
            allocation_seconds += phase["allocate_and_publish_seconds"].as_f64().unwrap();
            prepare_seconds += phase["prepare_seconds"].as_f64().unwrap();
            commit_seconds += phase["commit_seconds"].as_f64().unwrap();
        }
        let final_drain = drain(&graph, &source)?;
        drain_seconds += final_drain["seconds"].as_f64().unwrap();
        assert_plateau(&graph, &source, live, initial_capacity)?;
        let wall = cycle_start.elapsed().as_secs_f64();
        let quality = audit(&source, &queries, k, ef, floor)?;
        fs::write(
            dir.join(format!("audit-{cycle}.json")),
            serde_json::to_vec_pretty(&quality)?,
        )?;
        emit(
            json!({"event":"cycle","cycle":cycle,"completed_replacements":cycle*live,"seconds_excluding_audit":wall,"remove_seconds":remove_seconds,"grace_period_source_ack_seconds":drain_seconds,"allocation_publication_seconds":allocation_seconds,"prepare_seconds":prepare_seconds,"commit_seconds":commit_seconds,"mean_recall":quality["mean_recall"],"capacity_plateau":true,"capacity_plateau_scope":"source rows/pages and graph slot universe after reader drain; packed-store capacity is reported separately, not asserted constant","graph":graph_stats(&graph),"source":source.stats(),"memory":memory()}),
        )?;
    }
    fs::write(
        dir.join("complete.json"),
        serde_json::to_vec_pretty(
            &json!({"completed":true,"cycles":cycles,"live":live,"historical_insertions":next_key,"bounded_quiescent_source_rows_and_graph_slots":true,"scope":"native source-row and graph-slot ownership gate; packed-store capacity recorded separately; excludes historical PIT/production durability","graph":graph_stats(&graph),"source":source.stats()}),
        )?,
    )?;
    Ok(())
}
fn main() -> AnyResult<()> {
    let mut values = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" {
            println!("mutable-hnsw-lifecycle-poc --output NEW_DIR [--live 100000 --dims 96 --cycles 10 --batch 64 --queries 20 --k 10 --ef 1000 --recall-floor .95 --threads N]");
            return Ok(());
        }
        let key = key
            .strip_prefix("--")
            .ok_or("expected --option")?
            .to_string();
        if ![
            "output",
            "live",
            "dims",
            "cycles",
            "batch",
            "queries",
            "k",
            "ef",
            "recall-floor",
            "threads",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown option {key}").into());
        }
        let value = args.next().ok_or("missing option value")?;
        if values.insert(key, value).is_some() {
            return Err("duplicate option".into());
        }
    }
    let dir = PathBuf::from(values.get("output").ok_or("--output is required")?);
    fs::create_dir(&dir)?;
    let threads = values
        .get("threads")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(std::thread::available_parallelism()?.get());
    if threads == 0 {
        return Err("positive thread count required".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let mut events = BufWriter::new(File::create(dir.join("events.jsonl"))?);
    match run(&values, &mut events, &dir) {
        Ok(()) => Ok(()),
        Err(error) => {
            serde_json::to_writer(
                &mut events,
                &json!({"event":"failure","error":error.to_string(),"memory":memory()}),
            )?;
            writeln!(events)?;
            events.flush()?;
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};
    fn fixture(live: usize, slack: usize) -> (Arc<MutableHnsw>, PagedSource) {
        let graph = Arc::new(MutableHnsw::with_limits(
            8,
            Similarity::Cosine,
            HnswParams {
                m: 8,
                ef_construction: 64,
                threads: 1,
                ..Default::default()
            },
            MutableHnswLimits {
                max_slots: (live + slack) as u32,
                max_pending: slack.max(live),
                max_retired: slack.max(live),
                ..Default::default()
            },
        ));
        let source = PagedSource::new(graph.clone(), 8, live + slack, live + slack);
        (graph, source)
    }
    #[test]
    fn graph_epoch_first_keeps_source_until_capture_after_concurrent_remove() {
        let (graph, source) = fixture(1, 1);
        let (handles, _) = insert_batch(&graph, &source, &[(0, 0)], 0, 8).unwrap();
        let old = handles[0];
        let snapshot = graph.snapshot(); // Deliberately pause BEFORE source pin.
        std::thread::scope(|scope| {
            let (done_tx, done_rx) = mpsc::channel();
            let graph = &graph;
            let source = &source;
            let writer = scope.spawn(move || {
                assert!(graph.remove(old, &source.pin()).unwrap());
                for _ in 0..1000 {
                    assert_eq!(source.acknowledge(graph.collect_reclaimed()).unwrap(), 0);
                    std::thread::yield_now();
                }
                let _ = done_tx.send(());
            });
            // A writer assertion disconnects the channel instead of stranding
            // the reader at a barrier. Release its epoch before asserting.
            let outcome = done_rx.recv_timeout(Duration::from_secs(10));
            let late_source = source.pin();
            let row_remained = late_source.get_node(old, 0).is_some();
            let resident = source.stats()["resident_rows"].clone();
            drop(late_source);
            drop(snapshot);
            let joined = writer.join();
            assert!(outcome.is_ok(), "writer failed or timed out: {outcome:?}");
            assert!(joined.is_ok(), "writer panicked");
            assert!(row_remained);
            assert_eq!(resident, 1);
        });
        drain(&graph, &source).unwrap();
        assert_eq!(source.stats()["resident_rows"], 0);
    }
    #[test]
    fn old_source_view_owns_row_and_rejects_reused_generation() {
        let (graph, source) = fixture(1, 1);
        let (handles, _) = insert_batch(&graph, &source, &[(0, 0)], 0, 8).unwrap();
        let old = handles[0];
        let view = source.pin();
        let bytes = view.get_node(old, 0).unwrap().to_vec();
        assert!(graph.remove(old, &source.pin()).unwrap());
        drain(&graph, &source).unwrap();
        assert_eq!(source.stats()["resident_rows"], 1);
        assert_eq!(source.stats()["current_rows"], 0);
        let (next, _) = insert_batch(&graph, &source, &[(0, 1)], 1, 8).unwrap();
        let new = next[0];
        assert_eq!(old.slot(), new.slot());
        assert_ne!(old.generation(), new.generation());
        assert_eq!(view.get_node(old, 0).unwrap(), bytes);
        assert!(view.get_node(new, 1).is_none());
        assert!(source.pin().get_node(old, 0).is_none());
        assert_eq!(source.stats()["resident_rows"], 2);
        drop(view);
        drain(&graph, &source).unwrap();
        assert_eq!(source.stats()["resident_rows"], 1);
    }
    #[test]
    fn relocation_shares_payload_bytes_and_preserves_old_reader_owner() {
        let (graph, source) = fixture(1, 1);
        let (handles, _) = insert_batch(&graph, &source, &[(0, 0)], 0, 8).unwrap();
        let handle = handles[0];
        let old = source.pin();
        let pointer = old.payload_address(handle, 0).unwrap();
        source.relocate(handle, 0, 42).unwrap();
        let new = source.pin();
        assert_eq!(new.payload_address(handle, 0), Some(pointer));
        assert_eq!(new.storage_identity(handle), Some(42));
        assert_eq!(old.storage_identity(handle), None);
        assert_eq!(source.stats()["resident_rows"], 1);
        graph.remove(handle, &new).unwrap();
        drop(new);
        drain(&graph, &source).unwrap();
        assert_eq!(source.stats()["resident_rows"], 1);
        assert_eq!(old.payload_address(handle, 0), Some(pointer));
        drop(old);
        assert_eq!(source.stats()["resident_rows"], 0);
    }
    #[test]
    fn cancelled_reservation_clears_source_before_reuse() {
        let (graph, source) = fixture(1, 1);
        let permit = graph.reserve_insert(0).unwrap();
        let old = permit.handle();
        source
            .install(vec![(old, 0, source.allocate(vector(0, 0, 8)).unwrap())])
            .unwrap();
        drop(permit);
        drain(&graph, &source).unwrap();
        assert_eq!(source.stats()["resident_rows"], 0);
        let next = graph.reserve_insert(1).unwrap();
        assert_eq!(old.slot(), next.handle().slot());
        assert_ne!(old, next.handle());
        drop(next);
        drain(&graph, &source).unwrap();
    }
    #[test]
    fn foreign_graph_reclamation_cannot_erase_matching_numeric_source_slot() {
        let (graph, source) = fixture(1, 1);
        let (ours, _) = insert_batch(&graph, &source, &[(0, 0)], 0, 8).unwrap();
        let (foreign, _) = fixture(1, 1);
        let permit = foreign.reserve_insert(0).unwrap();
        assert_eq!(permit.handle(), ours[0]);
        drop(permit);
        let started = Instant::now();
        let tokens = loop {
            let tokens = foreign.collect_reclaimed();
            if !tokens.is_empty() {
                break tokens;
            }
            assert!(started.elapsed() < Duration::from_secs(10));
            std::thread::yield_now();
        };
        assert!(source.acknowledge(tokens).is_err());
        assert!(source.pin().get_node(ours[0], 0).is_some());
        assert_eq!(source.stats()["quarantined_tokens"], 1);
        assert_eq!(foreign.stats().reusable_slots, 0);
    }
    #[test]
    fn held_source_budget_applies_backpressure_then_recovers() {
        let (graph, source) = fixture(1, 1);
        let (handles, _) = insert_batch(&graph, &source, &[(0, 0)], 0, 8).unwrap();
        let held = source.pin();
        graph.remove(handles[0], &source.pin()).unwrap();
        drain(&graph, &source).unwrap();
        let (next, _) = insert_batch(&graph, &source, &[(0, 1)], 1, 8).unwrap();
        assert!(source.allocate(vector(0, 2, 8)).is_err());
        drop(held);
        let allowed = source.allocate(vector(0, 2, 8)).unwrap();
        drop(allowed);
        assert_eq!(source.stats()["resident_rows"], 1);
        graph.remove(next[0], &source.pin()).unwrap();
        drain(&graph, &source).unwrap();
        assert_eq!(source.stats()["resident_rows"], 0);
    }
    #[test]
    fn ten_full_corpus_replacements_keep_payload_and_directory_bounded() {
        let live = 128;
        let batch = 16;
        let (graph, source) = fixture(live, batch);
        let initial: Vec<_> = (0..live).map(|doc| (doc, doc as u64)).collect();
        let (mut handles, _) = insert_batch(&graph, &source, &initial, 0, 8).unwrap();
        drain(&graph, &source).unwrap();
        let capacity = graph.stats().allocated_slots as usize;
        let queries = vec![vector(0, 0, 8), vector(77, 0, 8)];
        let mut key = live as u64;
        for cycle in 1..=10 {
            for start in (0..live).step_by(batch) {
                for &handle in &handles[start..start + batch] {
                    graph.remove(handle, &source.pin()).unwrap();
                }
                drain(&graph, &source).unwrap();
                let items: Vec<_> = (start..start + batch)
                    .map(|doc| {
                        let next = key;
                        key += 1;
                        (doc, next)
                    })
                    .collect();
                let (replacement, _) = insert_batch(&graph, &source, &items, cycle, 8).unwrap();
                handles[start..start + batch].copy_from_slice(&replacement);
            }
            drain(&graph, &source).unwrap();
            assert_plateau(&graph, &source, live, capacity).unwrap();
            audit(&source, &queries, 10, 1024, 1.).unwrap();
        }
        assert_eq!(graph.stats().issued_key_watermark, Some(key - 1));
    }
}
