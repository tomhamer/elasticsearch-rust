//! Shared-field query dispatch across sealed text segments and published memtables.
//! Point owners are read lock-free from an owner-column snapshot taken once
//! per query: nothing holds the point table across the traversal or exact
//! scan, whose row reads take the store's state and table locks in that order.
use std::{collections::HashMap, sync::Arc};
use crate::{error::{Result, RusticError}, types::DocId, vector::{store::SearchStats, vespa_policy::*}};
use super::{query::{Query, SearchContext}, reader::LeafReader};

type Hits = (Vec<(f32, usize, DocId)>, SearchStats);

pub(crate) fn search(leaves: &[LeafReader<'_>], field: u16, vector: &[f32], k: usize, ef: usize,
    filter: Option<&Query>, ctx: &SearchContext, exact: bool, oversample: Option<f32>) -> Result<Option<Hits>>
{
    let store_of = |leaf: &LeafReader<'_>| match leaf {
        LeafReader::Segment(segment) => segment.vector_store().cloned(),
        LeafReader::Memtable(writer, ..) => writer.mutable_vector_store().cloned(),
    };
    // Sealed leaves first: a retained (PIT) reader view gives its segments a
    // captured store while its published memtables keep the live one, whose
    // points the captured graph already holds (they resolve through the
    // memtables' reservations below, not through the owner table).
    let segments_first = leaves.iter().filter(|leaf| matches!(leaf, LeafReader::Segment(_)))
        .chain(leaves.iter().filter(|leaf| matches!(leaf, LeafReader::Memtable(_, ..))));
    let Some((store, query_field)) = segments_first.filter_map(store_of).find_map(|store|
        store.mutable_query_field(field).map(|owner| (store, owner))) else {
        return Ok(None);
    };
    for leaf in leaves {
        let Some(other) = store_of(leaf) else { continue };
        let retained_memtable = matches!(leaf, LeafReader::Memtable(_, ..)) && store.is_read_capture();
        if !Arc::ptr_eq(&store, &other) && !retained_memtable {
            return Err(RusticError::Segment("mutable query spans inconsistent store views".into()));
        }
    }
    if k == 0 { return Ok(Some((Vec::new(), SearchStats::default()))); }
    if ctx.stopped() {
        return Err(RusticError::InvalidQuery("mutable query deadline exceeded".into()));
    }
    let requested_exact = exact;
    // Phase timings, reported when a query is slow (`slow_query_ms`): where
    // a kNN under concurrent writes spends its tail.
    let started = std::time::Instant::now();
    let mut phases = [std::time::Duration::ZERO; 4];
    let mut leaf_of = HashMap::new();
    let mut memory_points = HashMap::new();
    // Unfiltered: each memtable's reserved point ranges (resolved per hit by
    // binary search), not a map of every reserved point built per query.
    let mut memory_ranges: Vec<(usize, Vec<(u64, u64, usize)>)> = Vec::new();
    let mut masks = Vec::with_capacity(leaves.len());
    let mut filter_memory=Vec::new();
    let mut matched = 0usize;
    let mut accepted_points = Vec::new();
    // Memtable documents admitted while the field had no graph (see
    // `SegmentWriter::graphless_vector_docs`): scored exactly below.
    let mut graphless: Vec<(usize, DocId)> = Vec::new();
    // Mutable memtables and PIT retain the existing uncached preparation path.
    let cache_projection = query_field.supports_projection_cache()
        && leaves.iter().all(|leaf| matches!(leaf, LeafReader::Segment(_)));
    // A repeated filter over the same sealed leaves: its masks (after
    // deletions) from the kNN filter cache, not evaluated again.
    let mask_key = filter.filter(|_| cache_projection).and_then(|query| filter_mask_key(&store, field, query, leaves));
    let cached_masks = mask_key.as_ref().and_then(cached_filter_masks);
    for (li, leaf) in leaves.iter().enumerate() {
        let deleted = leaf.deletions_only();
        if let Some(cached) = &cached_masks {
            // All leaves are sealed segments here (`cache_projection`):
            // their docs resolve through the cached projection below.
            if let LeafReader::Segment(segment) = leaf { leaf_of.insert(segment.id, li); }
            masks.push(cached[li].clone());
            continue;
        }
        let mut mask = filter.map(|query| query.filter_bits(leaf, ctx)).transpose()?.map(|bits| {let(words,memory)=bits.into_parts();filter_memory.push(memory);words});
        // An unfiltered sealed leaf without deletions has no mask: its
        // projection key says "every doc" with an empty mask, instead of
        // filling, cloning and comparing `num_docs / 64` words per query.
        if mask.is_none() && deleted.is_some() {
            let memory=crate::admission::shared().reserve((leaf.num_docs() as u64).div_ceil(64).saturating_mul(8).saturating_add(128),"mutable visibility words")?;
            let mut words=vec![u64::MAX;(leaf.num_docs() as usize).div_ceil(64)];
            if let (Some(last),r @ 1..)=(words.last_mut(),leaf.num_docs()%64){*last=(1u64<<r)-1;}
            filter_memory.push(memory);mask=Some(words);
        }
        if let Some(deleted)=deleted {
            for(word,deleted) in mask.as_mut().expect("visibility mask").iter_mut().zip(deleted.words()){*word &= !deleted;}
        }
        let accepted = |doc: DocId| mask.as_ref().is_none_or(|words| words.get(doc as usize / 64)
            .is_some_and(|word| word & (1u64 << (doc % 64)) != 0));
        match leaf {
            LeafReader::Segment(segment) => {
                leaf_of.insert(segment.id, li);
                // Unfiltered with a memtable present (indexing): only the
                // count of visible vector docs is needed (the policy's
                // statistics), from the segment's cached doc -> point runs
                // and a popcount of the mask, not a point lookup per doc of
                // every segment on every query.
                let runs = if !cache_projection && filter.is_none() { segment.point_runs(field) } else { None };
                if let Some(runs) = runs {
                    matched += runs.iter().map(|r| match &mask {
                        Some(words) => count_bits(words, r.doc as usize, r.len as usize),
                        None => r.len as usize,
                    }).sum::<usize>();
                } else if !cache_projection {
                    // Bind the field column once per leaf. Opening the point map
                    // parses its directory; doing that for every accepted document
                    // makes filter projection allocate in the innermost loop.
                    let points = segment.point_map().column(field);
                    if let Some(points) = points {
                        if let Some(words) = &mask {
                            for (wi, &word) in words.iter().enumerate() {
                                let mut bits = word;
                                while bits != 0 {
                                    let doc = (wi * 64 + bits.trailing_zeros() as usize) as DocId;
                                    bits &= bits - 1;
                                    if let Some(point) = points.get(doc) {
                                        if filter.is_some() { accepted_points.push(point); }
                                        matched += 1;
                                    }
                                }
                            }
                        } else {
                            matched += (0..points.num_docs()).filter(|&doc| points.get(doc).is_some()).count();
                        }
                    }
                }
            }
            LeafReader::Memtable(writer, ..) if filter.is_none() => {
                let ranges = writer.reserved_point_ranges(field);
                for &(start, end, row) in &ranges {
                    for offset in 0..(end - start) as usize {
                        if writer.vector_row_doc(field, row + offset).is_some_and(|doc| accepted(doc)) { matched += 1; }
                    }
                }
                memory_ranges.push((li, ranges));
                graphless.extend(writer.graphless_vector_docs(field).into_iter().filter(|&doc| accepted(doc)).map(|doc| (li, doc)));
            }
            LeafReader::Memtable(writer, ..) => {
                for (doc, point) in writer.reserved_vector_points(field) {
                    if accepted(doc) {
                        matched += 1;
                        if filter.is_some() { accepted_points.push(point); }
                        memory_points.insert(point, (li, doc));
                    }
                }
                graphless.extend(writer.graphless_vector_docs(field).into_iter().filter(|&doc| accepted(doc)).map(|doc| (li, doc)));
            }
        }
        masks.push(mask);
    }
    if let (Some(key), None) = (mask_key, &cached_masks) { store_filter_masks(key, &masks); }
    // A retained view whose graph pin was released to bound memory: exact
    // search over the view's own vectors (correct, only slower).
    let exact_scan = || -> Result<Option<Hits>> {
        let similarity = query_field.similarity();
        let mut hits = Vec::new();
        let mut ops = 0u64;
        for (li, leaf) in leaves.iter().enumerate() {
            for doc in 0..leaf.num_docs() {
                if doc % 1024 == 0 && ctx.stopped() {
                    return Err(RusticError::InvalidQuery("mutable query deadline exceeded".into()));
                }
                let accepted = masks[li].as_ref().is_none_or(|words| words.get(doc as usize / 64)
                    .is_some_and(|word| word & (1u64 << (doc % 64)) != 0));
                if !accepted { continue; }
                if let Some(score) = leaf.with_vector(field, doc, |row| similarity.score(vector, row)) {
                    hits.push((score, li, doc));
                    ops += 1;
                }
            }
        }
        hits.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        hits.truncate(k);
        Ok(Some((hits, SearchStats { exact_segments: leaves.len(), vector_ops: ops, ..Default::default() })))
    };
    if query_field.is_released() {
        return exact_scan();
    }
    phases[0] = started.elapsed();
    let prepared_projection = if cache_projection {
        let key_bytes=masks.iter().fold(256u64,|n,mask|n.saturating_add(mask.as_ref().map_or(0,|m|m.len() as u64*8)).saturating_add(256));
        let key_memory=crate::admission::shared().reserve(key_bytes,"mutable projection key")?;
        let key = leaves.iter().zip(&masks).map(|(leaf, mask)| {
            let LeafReader::Segment(segment) = leaf else { unreachable!() };
            (segment.projection_id(), mask.clone().unwrap_or_default())
        }).collect();
        query_field.prepare_projection(key,key_memory, || leaves.iter().map(|leaf| {
            let LeafReader::Segment(segment) = leaf else { unreachable!() };
            segment.point_map().column(field)
        }).collect())?
    } else { None };
    if let Some(prepared) = &prepared_projection { matched = prepared.matched; }
    // Count only the captured searchable leaves. Detached sealing registers
    // vector payloads before replacing its published memtable; a global store
    // count would count those points again while the memtable is still visible.
    // It can also include sealed leaves outside a retained reader's snapshot.
    // ES's prefilter contract scores a set that fits within num_candidates
    // exactly. With one graph per shard this bound applies to the shard's
    // prepared vector matches, not an individual text segment. Keep this ES
    // adapter rule outside Vespa's graph traversal/planner implementation.
    // https://www.elastic.co/docs/solutions/search/vector/knn/filtered-knn-search
    let exact = exact || (filter.is_some() && matched <= ef);
    // A filter small enough that scanning its nodes' codes costs less than
    // the filtered graph search: the policy's exact route over the codes
    // (rescored), chosen by cost below (not `exact`, which would also drop
    // the rescore and scan f32 rows).
    let code_scan = !exact && filter.is_some() && rescores_codes(store.field_options(field), oversample)
        && (matched as f64) <= exact_scan_budget(ef, crate::vector::mutable_ingest::exact_scan_workers(matched));
    // The policy's document space: the graph's, read after the leaves were
    // projected (it only grows), and never below the rows this view holds.
    // Between an `index_options` migration retiring the current graph and
    // registering its successor, admissions go to the successor while
    // searches still use the current graph, so a view can hold more rows
    // than the graph it searches has nodes; those rows are not found by this
    // search, they must not fail it (`FilterCountExceedsSize`).
    let matched_count = u32::try_from(matched).map_err(|_| RusticError::InvalidQuery("mutable filter count overflow".into()))?;
    let limit = query_field.document_limit().max(matched_count);
    // An unknown query estimate uses the conservative pre-construction bound;
    // actual field-filter cardinality is supplied only to the second stage.
    let preparation = choose_filter_preparation(PreparationInput {
        approximate: !exact, has_index: true, estimated_query_ratio: 1.,
    }, PolicyConfig::default()).map_err(|e| RusticError::InvalidQuery(e.to_string()))?;
    let prepared = if exact { PreparedGlobalFilter::NotBuilt } else {
        PreparedGlobalFilter::Active(FilterStatistics {
            size: limit, count: matched_count,
        })
    };
    let options = store.field_options(field);
    let (candidate_k, rescore) = rescore_budget(k, exact, oversample, options)?;
    // B5: without rescoring ES returns the quantized estimate (Lucene scores
    // HNSW and exact candidates with the segment's quantized scorer). The
    // graph route returns each row's own code score: its codes and their
    // calibration epoch are what a Lucene segment's quantized vectors and
    // fitted quantizer are (no f32 read, no lock on the query path). An
    // exact route, which reads the candidates' f32 rows anyway, ranks
    // `max(k, num_candidates)` rows by f32 and then returns the same per-row
    // code scores, so a document's unrescored `_score` does not depend on
    // the route. Only a row with no codes yet (admitted outside the graph)
    // falls back to the field's cached score quantizer on its f32 row.
    let estimate = exact && !requested_exact && options.quantization.is_quantized() && oversample.unwrap_or(options.oversample) == 0.0;
    let candidate_k = if estimate { ef.max(k).min(OVERSAMPLE_LIMIT.max(k)) } else { candidate_k };
    // The beam is the request's: ES explores `num_candidates` (at least k
    // and the rescore gather) in every segment and returns k. An unfiltered
    // graph search of a quantized field explores a share of it
    // (`unfiltered_exploration`: one graph reaches its recall ceiling
    // earlier there), of a 1-bit field a wider beam (`binary_exploration`:
    // its codes steer the traversal coarsely); one of a raw field all of it
    // and at least a floor
    // (`raw_exploration`: one graph per shard against ES's per-segment
    // beams). A filtered one always explores exactly all of it.
    let ef = if filter.is_none() && !exact {
        if options.quantization == crate::vector::quantize::Quantization::Binary {
            crate::vector::vespa_policy::binary_exploration(candidate_k, ef, crate::vector::vespa_policy::binary_exploration_factor())
        } else if options.quantization.is_quantized() {
            crate::vector::vespa_policy::unfiltered_exploration(candidate_k, ef, crate::vector::vespa_policy::unfiltered_exploration_factor())
        } else {
            crate::vector::vespa_policy::raw_exploration(candidate_k, ef, crate::vector::vespa_policy::raw_exploration_floor())
        }
    } else { ef };
    let candidate_ef = ef.max(candidate_k);
    let mut plan = preparation.finish(HitBudget {
        target_hits: u32::try_from(candidate_k).map_err(|_| RusticError::InvalidQuery("mutable k overflow".into()))?,
        explore_additional_hits: u32::try_from(candidate_ef.saturating_sub(candidate_k)).map_err(|_| RusticError::InvalidQuery("mutable exploration overflow".into()))?,
    }, limit, prepared).map_err(|e| RusticError::InvalidQuery(e.to_string()))?;
    // The policy's exact route on a quantized field that rescores (a
    // selective filter): gather `candidate_k` by codes and rescore them by
    // f32 below, as Lucene's exact search on a quantized format and ES's
    // rescore do, instead of reading every filtered row's f32.
    if code_scan && rescore { plan.route = Route::Exact; }
    // The policy's exact route (Vespa's approximate-threshold: a filter
    // below `exact_lower_limit` of the field) on a quantized field that does
    // not rescore: Vespa's brute force over the field's attribute, here its
    // code plane in memory order, gathering `max(k, num_candidates)` and
    // ranking only those by f32, instead of reading every filtered node's
    // f32 row in key order. On vec1b (270M int8 rows, 1% filter, ~340k
    // filtered nodes a shard) those rows were cold random reads: 4-25 s a
    // shard search. The route still returns f32 scores, as before.
    let gather_codes = plan.route == Route::Exact && !exact && !rescore && options.quantization.is_quantized();
    if gather_codes {
        plan.requested_target_hits = u32::try_from(ef.max(k).min(OVERSAMPLE_LIMIT.max(k)))
            .map_err(|_| RusticError::InvalidQuery("mutable k overflow".into()))?;
    }
    plan.exact_codes = (rescore || gather_codes) && plan.route == Route::Exact;
    crate::vector::hnsw::note_route(match plan.route { Route::Exact => "exact", Route::FilterFirst => "filter-first", Route::Plain => "plain" });
    phases[1] = started.elapsed();
    let owners = store.mutable_point_owners();
    phases[2] = started.elapsed();
    // A reader view outlives the owner table it was published with: a merge
    // installed after the view was taken moves its points' owners to the
    // merged segment, which the view does not hold. Points never change, so
    // such a point is found through the view's own segments: each rejects a
    // point outside its point range at once (during indexing the traversal
    // meets points of documents this view does not hold on most queries)
    // and otherwise answers from a point -> doc index it builds once for
    // its lifetime, never a per-query map over every point.
    let by_point_map = |point: u64| -> Option<(usize, DocId)> {
        leaves.iter().enumerate().find_map(|(li, leaf)| match leaf {
            LeafReader::Segment(segment) => segment.doc_of_point(field, point).map(|doc| (li, doc)),
            LeafReader::Memtable(..) => None,
        })
    };
    // The owner table is only a fast path: it follows the live index, so a
    // point of this view may have no current owner (its segment merged
    // away and the merged segment's owner not installed, or detached) or
    // an owner outside the view. The view's own point maps decide then.
    let memory_range = |point: u64| -> Option<(usize, DocId)> {
        memory_ranges.iter().find_map(|(li, ranges)| {
            let i = ranges.partition_point(|r| r.0 <= point).checked_sub(1)?;
            let (start, end, row) = ranges[i];
            if point >= end { return None; }
            let LeafReader::Memtable(writer, ..) = &leaves[*li] else { return None };
            writer.vector_row_doc(field, row + (point - start) as usize).map(|doc| (*li, doc))
        })
    };
    let resolve = |point| memory_points.get(&point).copied().or_else(|| memory_range(point)).or_else(|| {
        match owners.owner(point) {
            Some((segment, doc)) => match leaf_of.get(&segment) {
                Some(&li) => Some((li, doc)),
                None => by_point_map(point),
            },
            None => by_point_map(point),
        }
    });
    // Unfiltered with memtables present (indexing): this view's visible
    // points as one bitset over point ids, built from the segments' cached
    // live points and the memtables' reserved rows, so the traversal
    // tests a bit per candidate instead of resolving its owner (a random
    // owner-table read and a hash lookup per scored node: a third of the
    // traversal's time under concurrent indexing).
    let visible = if filter.is_none() && prepared_projection.is_none() {
        visible_points(leaves, field, &masks, &memory_ranges, matched)?
    } else { None };
    let accept = |point: u64| match &visible {
        Some(visible) => visible.contains(point),
        None => resolve(point).is_some_and(|(li, doc)| masks[li].as_ref()
            .is_none_or(|words| words.get(doc as usize / 64).is_some_and(|word| word & (1u64 << (doc % 64)) != 0))),
    };
    let searched = if let Some(prepared) = prepared_projection {
        prepared.search(vector, &plan, ctx.deadline)
    } else if filter.is_some() {
        query_field.search_points(vector, &plan, &accepted_points, ctx.deadline)
    } else if let Some(visible) = &visible {
        query_field.search_shared(vector, &plan, &|point| visible.contains(point), ctx.deadline)
    } else {
        query_field.search(vector, &plan, accept, ctx.deadline)
    };
    phases[3] = started.elapsed();
    let result = match searched {
        Ok(result) => result,
        // Released between the check above and the search.
        Err(error) if crate::vector::mutable_ingest::is_released_capture(&error) => {
            return exact_scan();
        }
        Err(error) => return Err(error),
    };
    let mut hits = Vec::with_capacity(result.hits.len());
    let mut hit_points = Vec::with_capacity(if estimate { result.hits.len() } else { 0 });
    for (point, score) in result.hits {
        // Owners may have moved since the traversal (the snapshot reads live
        // owner entries); a point that is in none of this view's leaves
        // (neither owner nor point maps) does not match.
        let Some((li, doc)) = resolve(point) else { continue };
        hits.push((score, li, doc));
        if estimate { hit_points.push(point); }
    }
    // Sealed leaf access below takes the vector-store locks itself; no owner
    // lookup holds the table here.
    let mut rescore_ops = 0;
    // The graph scores each hit it returns once more (its final score);
    // a rescore replaces that score, so ES's profile count (traversal plus
    // `k * oversample` rescored rows, `210_knn_search_profile`) counts the
    // rescoring instead.
    let mut final_ops = 0;
    if (rescore && (plan.route != Route::Exact || plan.exact_codes)) || gather_codes {
        final_ops = result.final_score_ops;
        let similarity = query_field.similarity();
        // The rows to rescore are f32 in mapped vector segments: on a large
        // field most are not in the page cache. Ask for all of them first so
        // the kernel reads them concurrently, instead of one blocking fault
        // per row in the loop below. Only when they are not: an advice call
        // per row (`k * oversample` of them) cost about 0.5 ms of a k=100
        // so_vector query whose rows were all cached, so two probed rows
        // (the best hit and the middle one) decide.
        let cold = || [0, hits.len() / 2].iter().any(|&i| {
            let (_, li, doc) = hits[i];
            leaves[li].with_vector(field, doc, |row| !crate::vector::distance::resident(row)).unwrap_or(false)
        });
        if hits.len() > 1 && rescore_prefetch() && cold() {
            for (_, li, doc) in &hits {
                leaves[*li].with_vector(field, *doc, crate::vector::distance::will_need);
            }
        }
        for (score, li, doc) in &mut hits {
            if ctx.stopped() {
                return Err(RusticError::InvalidQuery("mutable rescore deadline exceeded".into()));
            }
            *score = leaves[*li].with_vector(field, *doc, |row| similarity.score(vector, row))
                .ok_or_else(|| RusticError::Segment("mutable rescore lost original vector".into()))?;
            rescore_ops += 1;
        }
    }
    if estimate {
        let codes = query_field.code_scores(vector, &hit_points)?;
        let mut fallback = None;
        for ((score, li, doc), code) in hits.iter_mut().zip(codes) {
            if ctx.stopped() {
                return Err(RusticError::InvalidQuery("mutable score estimate deadline exceeded".into()));
            }
            rescore_ops += 1;
            if let Some(code) = code { *score = code; continue }
            if fallback.is_none() {
                let Some(q) = query_field.score_quantizer() else { continue };
                fallback = Some(crate::vector::es_quant::EsQuantScorer::new(q, query_field.similarity(), vector));
            }
            let Some(scorer) = fallback.as_mut() else { continue };
            *score = leaves[*li].with_vector(field, *doc, |row| scorer.score(row))
                .ok_or_else(|| RusticError::Segment("mutable score estimate lost original vector".into()))?;
        }
    }
    // Rows the graph cannot hold yet: memtable documents admitted while the
    // field had no graph (a mapping just moved `flat`/`int8_flat` to a graph
    // type, before their memtable seals). Without them a kNN right after the
    // change finds only documents written since the graph was built.
    let mut graphless_ops = 0u64;
    if !graphless.is_empty() {
        let similarity = query_field.similarity();
        for &(li, doc) in &graphless {
            if ctx.stopped() {
                return Err(RusticError::InvalidQuery("mutable query deadline exceeded".into()));
            }
            if let Some(score) = leaves[li].with_vector(field, doc, |row| similarity.score(vector, row)) {
                hits.push((score, li, doc));
                graphless_ops += 1;
            }
        }
    }
    hits.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    hits.truncate(k);
    let total = started.elapsed();
    if total.as_millis() as u64 >= slow_query_ms() {
        #[cfg(test)]
        {
            let mut slow = SLOW_PHASES.lock();
            slow.0 += 1;
            for (sum, ms) in slow.1.iter_mut().zip([phases[0], phases[1] - phases[0], phases[2] - phases[1], phases[3] - phases[2], total - phases[3]]) {
                *sum += ms.as_secs_f64() * 1e3;
            }
        }
        // At most one report per second per process.
        static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        if LAST.swap(now, std::sync::atomic::Ordering::Relaxed) != now {
            tracing::warn!(field, total_ms = total.as_secs_f64() * 1e3,
                masks_ms = phases[0].as_secs_f64() * 1e3,
                projection_ms = (phases[1] - phases[0]).as_secs_f64() * 1e3,
                owners_lock_ms = (phases[2] - phases[1]).as_secs_f64() * 1e3,
                graph_ms = (phases[3] - phases[2]).as_secs_f64() * 1e3,
                finish_ms = (total - phases[3]).as_secs_f64() * 1e3,
                exact_phases_ms = ?crate::vector::mutable_ingest::take_exact_phases(),
                route = ?plan.route,
                leaves = leaves.len(), "slow mutable kNN query");
        }
    }
    Ok(Some((hits, SearchStats {
        exact_segments: usize::from(plan.route == Route::Exact),
        graph_segments: usize::from(plan.route != Route::Exact),
        vector_ops: traversal_ops(result.vector_ops.saturating_sub(final_ops), query_field.scored_node_bound()) + rescore_ops + graphless_ops, ..Default::default()
    })))
}

/// The kNN filter cache: a repeated filter's per-leaf masks (after
/// deletions) over the same sealed leaves, as ES's query cache keeps a
/// repeated filter's doc set per segment. Keyed by the store, field,
/// filter (only queries whose matches depend on nothing but the leaves'
/// contents, see [`deterministic_filter`]) and each leaf's segment
/// identity and deletion count, so any merge, refresh or delete makes a
/// new key. A few entries, LRU, on the node's accounting.
struct FilterMasks { key: FilterMaskKey, masks: Arc<Vec<Option<Vec<u64>>>>, _memory: crate::admission::Reservation }
type FilterMaskKey = (usize, u16, String, Vec<(u64, u32)>);
static FILTER_MASKS: parking_lot::Mutex<Vec<FilterMasks>> = parking_lot::const_mutex(Vec::new());
const FILTER_MASK_ENTRIES: usize = 16;

fn deterministic_filter(query: &Query) -> bool {
    match query {
        Query::Term(_) | Query::TermSet { .. } | Query::TermRange { .. } | Query::RangeI64 { .. } | Query::RangeU64 { .. }
            | Query::RangeF64 { .. } | Query::Exists { .. } | Query::MatchAll | Query::MatchNone => true,
        Query::ConstScore { query, .. } | Query::Boost { query, .. } => deterministic_filter(query),
        Query::Bool { must, should, must_not, filter, .. } =>
            must.iter().chain(should).chain(must_not).chain(filter).all(deterministic_filter),
        _ => false,
    }
}

fn filter_mask_key(store: &Arc<crate::vector::store::VectorStore>, field: u16, query: &Query, leaves: &[LeafReader<'_>]) -> Option<FilterMaskKey> {
    if !deterministic_filter(query) { return None; }
    let leaves = leaves.iter().map(|leaf| match leaf {
        LeafReader::Segment(segment) => Some((segment.projection_id(), leaf.deletions_only().map_or(0, |d| d.count() as u32))),
        LeafReader::Memtable(..) => None,
    }).collect::<Option<Vec<_>>>()?;
    Some((Arc::as_ptr(store) as usize, field, format!("{query:?}"), leaves))
}

fn cached_filter_masks(key: &FilterMaskKey) -> Option<Arc<Vec<Option<Vec<u64>>>>> {
    let mut cache = FILTER_MASKS.lock();
    let i = cache.iter().position(|e| &e.key == key)?;
    let entry = cache.remove(i);
    let masks = entry.masks.clone();
    cache.push(entry);
    Some(masks)
}

fn store_filter_masks(key: FilterMaskKey, masks: &[Option<Vec<u64>>]) {
    let bytes = masks.iter().map(|m| m.as_ref().map_or(0, |w| w.len() as u64 * 8)).sum::<u64>() + 256;
    let Ok(memory) = crate::admission::shared().reserve(bytes, "kNN filter masks") else { return };
    let mut cache = FILTER_MASKS.lock();
    if cache.iter().any(|e| e.key == key) { return; }
    if cache.len() >= FILTER_MASK_ENTRIES { cache.remove(0); }
    cache.push(FilterMasks { key, masks: Arc::new(masks.to_vec()), _memory: memory });
}

/// Filtered nodes up to which a quantized field's exact code scan replaces
/// the filtered graph search, per unit of `num_candidates`, on one thread:
/// on so_vector under saturating search load (16 clients, one worker a
/// scan) the scan of the `tags: java` filter (78k nodes a shard, k 100,
/// num_candidates 300) cost 13% more than the filtered graph search, so
/// the two break even near 230 nodes per unit of beam. Each further worker the
/// scan may borrow (idle cores, `exact_scan_workers`) adds
/// [`EXACT_SCAN_PER_EF_WORKER`]: with the cores free, the scan's latency is
/// its CPU over the workers, and its recall is the codes' (rescored), not
/// the graph's (so_vector, 10% filter, k 100: 0.999 against 0.986).
/// Vespa's choice between exact and approximate search by filter size,
/// with the threshold from cost instead of a fixed ratio.
/// `RUSTIC_KNN_EXACT_SCAN_PER_EF` scales it (0 disables).
const EXACT_SCAN_PER_EF: f64 = 200.0;
const EXACT_SCAN_PER_EF_WORKER: f64 = 300.0;
/// Highest budget per unit of beam however many workers are idle: at
/// num_candidates 50 the filter-first graph search (dense words and links,
/// prefetched) beat even a 9-worker scan of the `tags: java` filter (78k
/// nodes a shard: 2.02 against 2.25 ms end to end, force-merged); at 300 the
/// scan still wins wherever the budget allows it.
const EXACT_SCAN_PER_EF_MAX: f64 = 1200.0;
fn exact_scan_budget(ef: usize, workers: usize) -> f64 {
    static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let scale = *SCALE.get_or_init(|| std::env::var("RUSTIC_KNN_EXACT_SCAN_PER_EF").ok().and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v >= 0.0).map_or(1.0, |v| v / EXACT_SCAN_PER_EF));
    let per_ef = (EXACT_SCAN_PER_EF + EXACT_SCAN_PER_EF_WORKER * workers.saturating_sub(1) as f64).min(EXACT_SCAN_PER_EF_MAX);
    scale * per_ef * ef as f64
}

/// Whether an exact search of the field scores its codes (a quantized field
/// that rescores, `plan.exact_codes`): the scan the budget above prices.
fn rescores_codes(options: crate::vector::store::VectorFieldOptions, oversample: Option<f32>) -> bool {
    options.quantization.is_quantized() && oversample.unwrap_or(options.oversample) > 0.0
}

/// The profile's graph `vector_operations_count`, at most one per graph
/// node (`nodes`). Descent scores
/// an upper layer's neighbours and the layer-0 beam scores them again, so on
/// a tiny graph the raw count depended on which nodes the fixed-seed level
/// stream (and any repair) happened to put above layer 0: a 3-node graph gave
/// 3 or 4. ES's expected counts on such graphs are exact (its graph search
/// scores each of the N vectors once); the bound never binds on a graph of
/// realistic size, where a search scores a small fraction of the nodes.
pub(crate) fn traversal_ops(ops: u64, nodes: u32) -> u64 {
    ops.min(u64::from(nodes))
}

/// Whether rescoring asks for its f32 rows before reading them
/// (`RUSTIC_KNN_RESCORE_PREFETCH=0` turns it off; on by default).
fn rescore_prefetch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUSTIC_KNN_RESCORE_PREFETCH").map_or(true, |v| v != "0"))
}

/// ES `DenseVectorFieldMapper.OVERSAMPLE_LIMIT`: rescoring gathers at most
/// this many candidates (`adjustedK = min(ceil(k * oversample), 10_000)`).
pub(crate) const OVERSAMPLE_LIMIT: usize = 10_000;

// Keep request overrides distinct from mapping defaults, especially explicit
// zero (disabled). Validate before a float-to-integer cast can saturate.
// ES `DenseVectorFieldType.needsRescore`: only quantized index types rescore;
// a `rescore_vector` on `hnsw` / `flat` changes nothing. The gathered count
// is `min(ceil(k * oversample), OVERSAMPLE_LIMIT)` (never below `k`), and
// `num_candidates` grows to it (`numCands = max(adjustedK, numCands)`, the
// caller's `ef.max(candidate_k)`). `rescore_factor` (DiskBBQ recall parity)
// widens it within the same limit.
/// Test builds: how many queries were slow (`slow_query_ms`) and their
/// summed phase milliseconds (masks, projection, owner lock, graph, finish).
#[cfg(test)]
pub(crate) static SLOW_PHASES: parking_lot::Mutex<(u64, [f64; 5])> = parking_lot::const_mutex((0, [0.0; 5]));

/// `RUSTIC_KNN_SLOW_MS` (default 20): a mutable kNN at least this slow logs
/// its phase timings (at most once a second).
fn slow_query_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| std::env::var("RUSTIC_KNN_SLOW_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(20))
}

/// The points a reader view accepts, as bits over `base..base + 64 * words.len()`.
struct VisiblePoints {
    base: u64,
    words: Vec<u64>,
    _memory: crate::admission::Reservation,
}

impl VisiblePoints {
    #[inline]
    fn contains(&self, point: u64) -> bool {
        let Some(offset) = point.checked_sub(self.base) else { return false };
        self.words.get((offset / 64) as usize).is_some_and(|word| word & (1u64 << (offset % 64)) != 0)
    }

    fn set(&mut self, point: u64) {
        let offset = point - self.base;
        self.words[(offset / 64) as usize] |= 1u64 << (offset % 64);
    }
}

/// Points spanned by a view more sparsely than this many per visible vector
/// (plus a fixed allowance) keep per-candidate owner resolution: a bitset
/// over the span would cost more to build than the lookups it saves.
const VISIBLE_POINTS_SPAN_PER_MATCH: u64 = 8;
const VISIBLE_POINTS_SPAN_ALLOWANCE: u64 = 1 << 20;

/// The view's accepted points (`masks`: each leaf's live docs, `None` for
/// all of them; unfiltered), or `None` when they are spread too thin. Each
/// sealed leaf contributes its live points, cached on the segment until its
/// deletions change; the memtables (the docs since the last seal) are
/// projected row by row.
fn visible_points(leaves: &[LeafReader<'_>], field: u16, masks: &[Option<Vec<u64>>],
    memory_ranges: &[(usize, Vec<(u64, u64, usize)>)], matched: usize) -> Result<Option<VisiblePoints>>
{
    let mut parts = Vec::new();
    let (mut lo, mut hi) = (usize::MAX, 0usize);
    for (li, leaf) in leaves.iter().enumerate() {
        let LeafReader::Segment(segment) = leaf else { continue };
        let Some(column) = segment.point_map().column(field) else { continue };
        let Some(store) = segment.vector_store() else { return Ok(None) };
        let deleted = leaf.deletions_only().map_or(0, |d| d.count());
        let live = segment.cached_live_points(field, deleted, || {
            let every;
            let docs: &[u64] = match &masks[li] {
                Some(words) => words,
                None => {
                    let n = column.num_docs() as usize;
                    let mut words = vec![u64::MAX; n.div_ceil(64)];
                    if let (Some(last), r @ 1..) = (words.last_mut(), n % 64) { *last = (1u64 << r) - 1; }
                    every = words;
                    &every
                }
            };
            store.segment_point_words(docs, &column, segment.point_runs(field).as_deref())
        })?;
        if !live.words.is_empty() {
            lo = lo.min(live.first_word);
            hi = hi.max(live.first_word + live.words.len());
        }
        parts.push(live);
    }
    for (_, ranges) in memory_ranges {
        for &(start, end, _) in ranges {
            if start < end {
                lo = lo.min((start / 64) as usize);
                hi = hi.max(end.div_ceil(64) as usize);
            }
        }
    }
    if lo >= hi { return Ok(None); }
    if (hi - lo) as u64 * 64 > (matched as u64).saturating_mul(VISIBLE_POINTS_SPAN_PER_MATCH).saturating_add(VISIBLE_POINTS_SPAN_ALLOWANCE) {
        return Ok(None);
    }
    let memory = crate::admission::shared().reserve((hi - lo) as u64 * 8 + 64, "mutable visible points")?;
    let mut visible = VisiblePoints { base: lo as u64 * 64, words: vec![0; hi - lo], _memory: memory };
    for live in &parts {
        if live.words.is_empty() { continue; }
        for (dst, word) in visible.words[live.first_word - lo..].iter_mut().zip(&live.words) { *dst |= word; }
    }
    for (li, ranges) in memory_ranges {
        let LeafReader::Memtable(writer, ..) = &leaves[*li] else { continue };
        let mask = masks[*li].as_deref();
        for &(start, end, row) in ranges {
            for offset in 0..(end - start) as usize {
                let Some(doc) = writer.vector_row_doc(field, row + offset) else { continue };
                if mask.is_none_or(|words| words.get(doc as usize / 64).is_some_and(|word| word & (1u64 << (doc % 64)) != 0)) {
                    visible.set(start + offset as u64);
                }
            }
        }
    }
    Ok(Some(visible))
}

#[cfg(test)]
pub(crate) fn count_bits_for_test(words: &[u64], start: usize, len: usize) -> usize { count_bits(words, start, len) }

/// Set bits of `words` in docs `start..start + len`.
fn count_bits(words: &[u64], start: usize, len: usize) -> usize {
    let end = start + len;
    let mut count = 0usize;
    let mut doc = start;
    while doc < end {
        let w = doc / 64;
        let lo = doc % 64;
        let hi = (end - w * 64).min(64);
        let Some(&word) = words.get(w) else { break };
        let mask = if hi - lo == 64 { u64::MAX } else { ((1u64 << (hi - lo)) - 1) << lo };
        count += (word & mask).count_ones() as usize;
        doc = w * 64 + hi;
    }
    count
}

pub(crate) fn rescore_budget(k: usize, exact: bool, requested: Option<f32>,
    options: crate::vector::store::VectorFieldOptions) -> Result<(usize, bool)> {
    let oversample = requested.unwrap_or(options.oversample);
    if !oversample.is_finite() || oversample < 0.0 {
        return Err(RusticError::InvalidQuery("rescore oversample must be finite and nonnegative".into()));
    }
    if exact || oversample == 0.0 || !options.quantization.is_quantized() { return Ok((k, false)); }
    if !options.rescore_factor.is_finite() || options.rescore_factor <= 0.0 {
        return Err(RusticError::InvalidQuery("rescore factor must be finite and positive".into()));
    }
    let count = (k as f64 * f64::from(oversample) * f64::from(options.rescore_factor)).ceil();
    if count > u32::MAX as f64 || k > u32::MAX as usize {
        return Err(RusticError::InvalidQuery("mutable rescore candidate count overflow".into()));
    }
    Ok((k.max((count as usize).min(OVERSAMPLE_LIMIT)), true))
}

#[cfg(test)]
mod tests {
    #[test]
    fn graph_operation_count_is_at_most_one_per_node() {
        // 210_knn_search_profile: 3 vectors, descent re-scored one of them.
        assert_eq!(super::traversal_ops(4, 3), 3);
        assert_eq!(super::traversal_ops(3, 3), 3);
        assert_eq!(super::traversal_ops(120, 100_000), 120);
        assert_eq!(super::traversal_ops(5, 0), 0);
    }

    use super::*;
    use crate::vector::{store::VectorFieldOptions, quantize::Quantization};

    #[test]
    fn visible_points_test_their_own_bits() {
        let memory = crate::admission::shared().reserve(64, "test visible points").unwrap();
        let mut visible = VisiblePoints { base: 64, words: vec![0; 2], _memory: memory };
        for point in [64, 65, 127, 128, 191] { visible.set(point); }
        for point in 0..300u64 {
            assert_eq!(visible.contains(point), [64, 65, 127, 128, 191].contains(&point), "point {point}");
        }
    }

    #[test]
    fn rescore_budget_honors_mapping_request_disable_and_exact() {
        let options = VectorFieldOptions::quantized_hnsw(Quantization::Binary).with_rescore_factor(2.0);
        assert_eq!(rescore_budget(10, false, None, options).unwrap(), (60, true));
        assert_eq!(rescore_budget(10, false, Some(1.25), options).unwrap(), (25, true));
        assert_eq!(rescore_budget(10, false, Some(0.0), options).unwrap(), (10, false));
        assert_eq!(rescore_budget(10, true, None, options).unwrap(), (10, false));
        assert_eq!(rescore_budget(10, false, None, VectorFieldOptions::f32_hnsw()).unwrap(), (10, false));
        for value in [f32::NAN, f32::INFINITY, -1.0, f32::MAX] {
            assert!(rescore_budget(10, false, Some(value), options).is_err());
        }
        assert!(rescore_budget(u32::MAX as usize, false, None, options).is_err());
        // ES ignores rescore_vector on unquantized fields and caps the gathered count.
        assert_eq!(rescore_budget(10, false, Some(4.0), VectorFieldOptions::f32_hnsw()).unwrap(), (10, false));
        let int8 = VectorFieldOptions::quantized_hnsw(Quantization::Int8);
        assert_eq!(rescore_budget(4_000, false, Some(10.0), int8).unwrap(), (OVERSAMPLE_LIMIT, true));
        assert_eq!(rescore_budget(12_000, false, Some(2.0), int8).unwrap(), (12_000, true));
        assert_eq!(rescore_budget(7, false, Some(1.5), int8).unwrap(), (11, true), "ceil(k * oversample)");
    }
}
