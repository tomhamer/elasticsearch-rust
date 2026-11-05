//! Restricted reproduction of ES's shard KNN candidates -> global winners -> query phase.
use super::*;
use rustic_core::query::query::{CustomQuery, ScoredDocs};
use rustic_core::query::{Scorer, SearchContext, TopDocs};
use std::collections::HashMap;

#[derive(Debug)]
struct FixedDocs {
    by_leaf: HashMap<usize, Vec<(u32, f32)>>,
}
impl CustomQuery for FixedDocs {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let live = leaf.live_docs();
        let docs: Vec<_> = self
            .by_leaf
            .get(&leaf.id())?
            .iter()
            .copied()
            .filter(|(d, _)| live(*d))
            .collect();
        if docs.is_empty() {
            None
        } else {
            Some(Box::new(ScoredDocs::new(docs)))
        }
    }
    fn name(&self) -> &'static str {
        "knn_fixed"
    }
}

pub(super) fn search(
    leaves: &[LeafReader<'_>],
    query: &Query,
    options: &SearchOptions,
    trim_candidates: bool,
) -> (TopDocs, Value) {
    search_impl(leaves, query, options, trim_candidates, trim_candidates)
}

pub(super) fn search_collect_topk(
    leaves: &[LeafReader<'_>],
    query: &Query,
    options: &SearchOptions,
) -> (TopDocs, Value) {
    search_impl(leaves, query, options, true, false)
}

fn search_impl(
    leaves: &[LeafReader<'_>],
    query: &Query,
    options: &SearchOptions,
    trim_candidates: bool,
    trim_query: bool,
) -> (TopDocs, Value) {
    let Query::Knn {
        k, num_candidates, ..
    } = query
    else {
        return (
            Searcher::new(leaves.to_vec()).search(query, options),
            json!({}),
        );
    };
    let keep = if trim_candidates { *k } else { *num_candidates };
    let start = Instant::now();
    let mut shard_query = query.clone();
    if let Query::Knn {
        k: shard_k,
        num_candidates: shard_nc,
        ..
    } = &mut shard_query
    {
        *shard_k = if trim_query { keep } else { *num_candidates };
        *shard_nc = (*num_candidates).max(*k);
    }
    let candidates = Searcher::new(leaves.to_vec()).search(
        &shard_query,
        &SearchOptions {
            top_k: keep,
            track_total_hits: TrackTotalHits::False,
            ..Default::default()
        },
    );
    let candidate_us = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let mut winners = candidates.hits;
    winners.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.leaf.cmp(&b.leaf))
            .then(a.doc.cmp(&b.doc))
    });
    winners.truncate(*k);
    let mut by_leaf = HashMap::<usize, Vec<(u32, f32)>>::new();
    for h in winners {
        by_leaf
            .entry(leaves[h.leaf].id())
            .or_default()
            .push((h.doc, h.score));
    }
    let fixed = if by_leaf.is_empty() {
        Query::MatchNone
    } else {
        Query::Custom(Arc::new(FixedDocs { by_leaf }))
    };
    let merge_us = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let top = Searcher::new(leaves.to_vec()).search(&fixed, options);
    (
        top,
        json!({"candidate_us":candidate_us,"merge_us":merge_us,"fixed_query_us":start.elapsed().as_secs_f64()*1e6}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float_topk_candidates_preserve_global_winners_with_multiple_leaves_and_deletions() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = Index::create(
            dir.path(),
            "knn-test",
            schema(Some("vectors")),
            IndexSettings::default(),
        )
        .unwrap();
        let mut rng = 73u64;
        let mut vectors = Vec::new();
        for i in 0..300 {
            let v: Vec<f32> = (0..128)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 32) as u32) as f32 / u32::MAX as f32 - 0.5
                })
                .collect();
            let mut d = Document::new();
            d.add_keyword(ID, &format!("doc-{i}"));
            d.add_keyword(SOURCE, "{}");
            d.add_keyword(5, &format!("t{}", i % 3));
            d.add_vector(7, v.clone());
            index.add_document(&d).unwrap();
            vectors.push(v);
            if i % 100 == 99 {
                index.refresh().unwrap();
            }
        }
        index.force_merge_vectors(1, false).unwrap();
        index.commit().unwrap();
        assert_eq!(index.num_segments(), 3);
        assert!(
            index
                .vector_store()
                .segments(7)
                .iter()
                .all(|s| s.is_sealed() && s.has_graph())
        );
        index
            .delete_doc(&rustic_core::DocAddress::new(0, 1))
            .unwrap();
        index.refresh().unwrap();
        let leaves = index.leaves();
        let options = SearchOptions {
            top_k: 10,
            track_total_hits: TrackTotalHits::True,
            ..Default::default()
        };
        for nc in [10, 30, 100] {
            for filter in [None, Some(Box::new(Query::term(5, "t1")))] {
                for vector in [&vectors[2], &vectors[155], &vectors[270]] {
                    let query = Query::Knn {
                        field: 7,
                        vector: vector.clone(),
                        k: 5,
                        num_candidates: nc,
                        filter: filter.clone(),
                        similarity_threshold: None,
                        boost: 1.0,
                        oversample: None,
                        visit_percentage: None,
                    };
                    let old = search(&leaves, &query, &options, false).0;
                    let trimmed = search(&leaves, &query, &options, true).0;
                    let collected = search_collect_topk(&leaves, &query, &options).0;
                    assert_eq!(old.total_hits, trimmed.total_hits);
                    let tuples = |top: TopDocs| {
                        top.hits
                            .into_iter()
                            .map(|h| (h.leaf, h.doc, h.score))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(tuples(old.clone()), tuples(trimmed));
                    assert_eq!(tuples(old), tuples(collected));
                }
            }
        }
    }
}
