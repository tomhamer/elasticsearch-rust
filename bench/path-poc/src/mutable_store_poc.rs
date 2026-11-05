//! Shared VectorStore + real SegmentWriter integration smoke. EC2 only.
//! Full ES/WAL/HTTP, exact routing and retained PIT admission are separate gates.
use rustic_core::{codec::directory::RamDirectory, segment::SegmentWriter,
    types::{Document, Field, FieldType}, vector::{HnswParams, Similarity,
    store::{VectorStore, VectorFieldOptions}, vespa_policy::*}};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{error::Error, fs, io::Read, path::Path, time::{Instant, Duration}};
use sha2::{Digest, Sha256};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=5).contains(&args.len()) { return Err("usage: mutable-store-poc FIXTURE NEW_OUTPUT_DIRECTORY [DOCUMENTS] [--profile-after-clean]".into()); }
    let profile_after_clean = args.get(4).is_some_and(|v| v == "--profile-after-clean");
    if args.len() == 5 && !profile_after_clean { return Err("unknown profiling option".into()); }
    let root = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    fs::create_dir(output)?;
    let stream_dim: Option<usize> = std::env::var("RUSTIC_POC_STREAM_DIMS").ok().map(|v| v.parse()).transpose()?;
    let dim = match stream_dim {
        Some(dim) => dim,
        None => serde_json::from_slice::<Value>(&fs::read(root.join("manifest.json"))?)?["dims"].as_u64().ok_or("missing dims")? as usize,
    };
    let count: usize = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(100_000);
    if count < 100_000 || dim == 0 { return Err("performance smoke requires at least 100K documents and nonzero dimensions".into()); }
    let bytes = count.checked_mul(dim).and_then(|n| n.checked_mul(4)).ok_or("fixture size overflow")?;
    let input_path = if root.join("vectors.f32").exists() { root.join("vectors.f32") } else { root.join("vectors.f32.part") };
    // A sequential read of completed batches avoids mapping unwritten pages.
    // The producer only appends; its final rename preserves this descriptor.
    let mut input = fs::File::open(input_path)?;
    let mut input_digest = Sha256::new();
    let mut input_wait_seconds = 0.0;
    let store = VectorStore::new();
    let field = store.mutable_field(1, dim as u32, Similarity::Cosine,
        HnswParams { m: 16, ef_construction: 200, threads: 32, ..Default::default() })?;
    let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: u16::try_from(dim)? })
        .with_similarity("cosine").with_vector_options(Some(VectorFieldOptions::f32_hnsw().with_hnsw(16, 200)))];
    let directory = RamDirectory::default();
    let started = Instant::now();
    let mut segments = Vec::new();
    for (sid, start) in (0..count).step_by(4096).enumerate() {
        let mut writer = SegmentWriter::new(sid as u32);
        let end = (start + 4096).min(count);
        let wait = Instant::now();
        while input.metadata()?.len() < (end * dim * 4) as u64 {
            if stream_dim.is_none() { return Err("truncated immutable fixture".into()); }
            if wait.elapsed() > Duration::from_secs(3600) { return Err("fixture producer stalled".into()); }
            std::thread::sleep(Duration::from_millis(250));
        }
        input_wait_seconds += wait.elapsed().as_secs_f64();
        let mut raw = vec![0u8; (end - start) * dim * 4];
        input.read_exact(&mut raw)?;
        input_digest.update(&raw);
        let batch_values: Vec<f32> = raw.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
        if batch_values.iter().any(|v| !v.is_finite()) { return Err("non-finite fixture batch".into()); }
        let row = |id: usize| &batch_values[(id-start)*dim..(id-start+1)*dim];
        for batch in (start..end).step_by(64) {
            for id in batch..(batch + 64).min(end) {
                let mut doc = Document::new();
                doc.add_vector(1, row(id).to_vec());
                writer.add_document(&doc, &schema);
            }
            writer.stage_mutable_field(1, &field)?;
        }
        let segment = writer.write_to_store_ref(&directory, Some(&store), false)?;
        for id in start..end {
            if segment.point(1, (id-start) as u32) != Some(id as u64) { return Err("point identity mismatch".into()); }
        }
        segments.push(segment);
        if end == count || start / 100_000 != end / 100_000 {
            println!("indexed {end}/{count} vectors ({dim} dimensions), elapsed {:.1}s", started.elapsed().as_secs_f64());
        }
        if !std::sync::Arc::ptr_eq(&field, &store.mutable_field(1, dim as u32, Similarity::Cosine,
            HnswParams { m: 16, ef_construction: 200, threads: 32, ..Default::default() })?) { return Err("field graph changed".into()); }
    }
    let indexing_seconds = started.elapsed().as_secs_f64();
    let wait = Instant::now();
    while !root.join("manifest.json").exists() {
        if stream_dim.is_none() || wait.elapsed() > Duration::from_secs(3600) { return Err("fixture manifest missing".into()); }
        std::thread::sleep(Duration::from_millis(250));
    }
    let manifest: Value = loop {
        match serde_json::from_slice(&fs::read(root.join("manifest.json"))?) {
            Ok(manifest) => break manifest,
            Err(error) if error.is_eof() && wait.elapsed() < Duration::from_secs(3600) => {
                std::thread::sleep(Duration::from_millis(250));
            }
            Err(error) => return Err(error.into()),
        }
    };
    if manifest["dims"].as_u64() != Some(dim as u64) || manifest["docs"].as_u64().is_none_or(|n| n < count as u64) {
        return Err("completed fixture dimensions/count mismatch".into());
    }
    let indexed_sha256 = format!("{:x}", input_digest.finalize());
    if manifest["docs"].as_u64() == Some(count as u64) && manifest["sha256"].as_str() != Some(indexed_sha256.as_str()) {
        return Err("streamed input differs from completed fixture checksum".into());
    }
    let queries = 100usize;
    let query_rows: Vec<Vec<f32>> = serde_json::from_slice(&fs::read(root.join("queries.json"))?)?;
    if query_rows.len() < queries || query_rows.iter().any(|row| row.len() != dim || row.iter().any(|v| !v.is_finite())) {
        return Err("invalid held-out queries".into());
    }
    let data = fs::File::open(root.join("vectors.f32"))?;
    if !cfg!(target_endian = "little") || data.metadata()?.len() < bytes as u64 { return Err("invalid fixture".into()); }
    // Only the finalized immutable file is mapped, after producer completion.
    let map = unsafe { memmap2::MmapOptions::new().len(bytes).map(&data)? };
    if (map.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 { return Err("unaligned fixture".into()); }
    let values = unsafe { std::slice::from_raw_parts(map.as_ptr().cast::<f32>(), count * dim) };
    let row = |i: usize| if i < count { &values[i * dim..(i + 1) * dim] } else { query_rows[i-count].as_slice() };
    if store.num_alive(1) != count { return Err("published count mismatch".into()); }
    let capture_start = Instant::now();
    let retained = store.capture_read_store()?;
    let capture_seconds = capture_start.elapsed().as_secs_f64();
    let mut measurements = Vec::new();
    for step in [1usize, 2, 10] {
        let matched = count.div_ceil(step);
        let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
            estimated_query_ratio: 1. / step as f64 }, PolicyConfig::default())?.finish(
                HitBudget { target_hits: 10, explore_additional_hits: 990 }, count as u32,
                PreparedGlobalFilter::Active(FilterStatistics { size: count as u32, count: matched as u32 }))?;
        // Independent f64 cosine oracle, outside search timers.
        let truth: Vec<Vec<u64>> = (0..queries).into_par_iter().map(|q| {
            let query = row(count + q);
            let qnorm = query.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let mut scores: Vec<_> = (0..count).step_by(step).map(|id| {
                let vector = row(id);
                let norm = vector.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let dot: f64 = vector.iter().zip(query).map(|(&x,&y)| x as f64 * y as f64).sum();
                (id as u64, dot/(norm*qnorm))
            }).collect();
            scores.sort_unstable_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scores.into_iter().take(10).map(|x|x.0).collect()
        }).collect();
        let accept = |key: u64| key < count as u64 && key % step as u64 == 0 && store.owner(key).is_some();
        let preparation_started = Instant::now();
        let prepared = field.prepare_ann(accept)?;
        let filter_preparation_ms = preparation_started.elapsed().as_secs_f64() * 1000.;
        let reference: Vec<_> = (0..queries).map(|q| store.search_mutable_ann(1, row(count+q), &plan, accept, None).map(|r| r.expect("enabled field")))
            .collect::<std::result::Result<Vec<_>,_>>()?;
        for (q, expected) in reference.iter().enumerate() {
            let actual = retained.search_mutable_ann(1, row(count+q), &plan,
                |key| key < count as u64 && key % step as u64 == 0 && retained.owner(key).is_some(), None)?.ok_or("missing retained field")?;
            if actual.hits != expected.hits { return Err("retained/live view differs at quiescent capture".into()); }
        }
        for (q, expected) in reference.iter().enumerate() {
            if prepared.search(row(count+q), &plan, None)?.hits != expected.hits {
                return Err("prepared node-ID filter differs from authoritative document predicate".into());
            }
        }
        let recall: Vec<f64> = reference.iter().zip(&truth).map(|(r,t)| r.hits.iter().filter(|h|t.contains(&h.0)).count() as f64/10.).collect();
        let signatures: Vec<Vec<_>> = reference.iter().map(|r|r.hits.iter().map(|h|(h.0,h.1.to_bits())).collect()).collect();
        for prepared_filter in [false, true] {
        for clients in [1usize, 8, 32] {
            let requests = 1024usize;
            let start = Instant::now();
            let samples = std::thread::scope(|scope| -> Result<Vec<f64>> {
                let jobs: Vec<_> = (0..clients).map(|client| {
                    let signatures = &signatures;
                    let store = &store;
                    let prepared = &prepared;
                    scope.spawn(move || -> Result<Vec<f64>> {
                        let mut samples = Vec::new();
                        for request in (client..requests).step_by(clients) {
                            let q = request % queries;
                            let begin = Instant::now();
                            let result = if prepared_filter { prepared.search(row(count+q), &plan, None)? }
                                else { store.search_mutable_ann(1, row(count+q), &plan, accept, None)?.ok_or("missing live field")? };
                            samples.push(begin.elapsed().as_secs_f64()*1000.);
                            let actual: Vec<_> = result.hits.iter().map(|h|(h.0,h.1.to_bits())).collect();
                            if actual != signatures[q] { return Err("concurrent query differs from validated reference".into()); }
                        }
                        Ok(samples)
                    })
                }).collect();
                let mut all = Vec::new();
                for job in jobs { all.extend(job.join().map_err(|_| "query worker panicked")??); }
                Ok(all)
            })?;
            let elapsed = start.elapsed().as_secs_f64();
            let mut samples = samples;
            samples.sort_by(f64::total_cmp);
            measurements.push(json!({"accepted_fraction":1./step as f64,"clients":clients,
                "requests":requests,"p50_ms":samples[samples.len()/2],"p99_ms":samples[samples.len()*99/100],
                "qps":requests as f64/elapsed,"mean_recall":recall.iter().sum::<f64>()/queries as f64,
                "min_recall":recall.iter().copied().fold(1.,f64::min),"route":format!("{:?}",plan.route),
                "all_timed_results_verified":true,"prepared_node_filter":prepared_filter,
                "filter_preparation_ms":filter_preparation_ms,
                "scope":if prepared_filter { "ANN with reusable fixture filter; preparation reported separately" } else { "live query with predicate" }}));
        }
        }
    }
    let result = json!({"source_revision":option_env!("RUSTIC_POC_SOURCE_REVISION"),
        "scope":"shared VectorStore, real text writers and chunk publication; no ES HTTP/WAL/PIT or capacity acceptance",
        "documents":count,"dimensions":dim,"text_segments":segments.len(),"indexing_seconds":indexing_seconds,
        "indexing_docs_per_second":count as f64/indexing_seconds,
        "input_wait_seconds":input_wait_seconds,"indexing_active_seconds":indexing_seconds-input_wait_seconds,
        "streamed_fixture":stream_dim.is_some(),"indexed_prefix_sha256":indexed_sha256,
        "indexing_timing_scope":if stream_dim.is_some(){"pipeline wall time; overlaps fixture producer, not standalone indexing acceptance"}else{"immutable input"},"retained_capture_seconds":capture_seconds,"fixture":manifest,"measurements":measurements,
        "rss_status":fs::read_to_string("/proc/self/status").unwrap_or_default()});
    fs::write(output.join("result.json"),serde_json::to_vec_pretty(&result)?)?;
    println!("{}",serde_json::to_string(&result)?);
    fs::write(output.join("clean.done"), b"unprofiled measurements saved\n")?;
    if profile_after_clean {
        // The runner attaches only AFTER all authoritative measurements finish.
        // Keep the same graph/source allocations; never rebuild for profiling.
        let waiting = Instant::now();
        while !output.join("profile.ready").exists() {
            if waiting.elapsed().as_secs() > 180 { return Err("profile attachment timeout".into()); }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let matched = count.div_ceil(10);
        let plan = choose_filter_preparation(PreparationInput { approximate: true, has_index: true,
            estimated_query_ratio: 0.1 }, PolicyConfig::default())?.finish(
            HitBudget { target_hits: 10, explore_additional_hits: 990 }, count as u32,
            PreparedGlobalFilter::Active(FilterStatistics { size: count as u32, count: matched as u32 }))?;
        let prepared = field.prepare_ann(|key| key < count as u64 && key % 10 == 0 && store.owner(key).is_some())?;
        let diagnostic = Instant::now();
        let mut requests = 0usize;
        while diagnostic.elapsed().as_secs() < 25 {
            let batch: Vec<_> = (0..32).into_par_iter().map(|i| {
                prepared.search(row(count + (requests+i)%queries), &plan, None)
            }).collect::<std::result::Result<_,_>>()?;
            if batch.iter().any(|r| r.missing_source_lookups != 0) { return Err("profile source missing".into()); }
            std::hint::black_box(&batch);
            requests += batch.len();
        }
        fs::write(output.join("profile-diagnostic.json"), serde_json::to_vec_pretty(&json!({
            "scope":"diagnostic only; same graph after unprofiled measurement; 10% filter C32",
            "requests":requests,"seconds":diagnostic.elapsed().as_secs_f64(),"acceptance_timing":false
        }))?)?;
    }
    Ok(())
}
