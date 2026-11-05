//! Encoded traversal must affect candidate selection, not just final scores.
use super::*;
use crate::vector::es_quant::{EsCodes, EsQuant, EsQuantScorer};

struct Rows(Vec<Vec<f32>>);
impl MutableVectorSource for Rows {
    fn get(&self, key: u64) -> Option<&[f32]> { self.0.get(key as usize).map(Vec::as_slice) }
}
struct Encoded<'a> {
    rows: &'a Rows,
    handles: &'a [NodeHandle],
    codes: EsCodes,
    similarity: Similarity,
    missing: Option<u64>,
}
struct Bound<'a> { source: &'a Encoded<'a>, scorer: EsQuantScorer }
impl MutableQueryScorer for Bound<'_> {
    fn score(&self, handle: NodeHandle, key: u64) -> Option<f32> {
        if self.source.missing == Some(key) || self.source.handles.get(key as usize) != Some(&handle) { return None; }
        Some(self.scorer.score_row(&self.source.codes, key as usize))
    }
}
impl MutableVectorSource for Encoded<'_> {
    fn get(&self, key: u64) -> Option<&[f32]> { self.rows.get(key) }
    fn bind_query(&self, query: &[f32]) -> Option<Box<dyn MutableQueryScorer + '_>> {
        Some(Box::new(Bound { source: self,
            scorer: EsQuantScorer::new(self.codes.q.clone(), self.similarity, query) }))
    }
}

#[test]
fn encoded_scores_drive_scalar_batch_plain_and_filtered_search() {
    let rows = Rows((0..48).map(|i| (0..32).map(|j|
        (((i * 37 + j * 19 + i * j * 3) % 101) as f32 - 50.0) / 90.0).collect()).collect());
    let query = &rows.0[13];
    for similarity in [Similarity::L2, Similarity::Cosine, Similarity::DotProduct, Similarity::MaxInnerProduct] {
        let graph = MutableHnsw::new(32, similarity, HnswParams { m: 32, ef_construction: 96, threads: 1, ..Default::default() });
        let handles: Vec<_> = (0..48).map(|key| graph.insert(key, &rows).unwrap()).collect();
        let view = graph.snapshot();
        for (bits, osq) in [(7, false), (4, false), (7, true), (4, true), (1, true)] {
            let spec = EsQuant { bits, osq, confidence_interval: None };
            let q = spec.fit(similarity, rows.0.len(), |i| &rows.0[i]);
            let source = Encoded { rows: &rows, handles: &handles,
                codes: EsCodes::build(q, similarity, 48, 32, |i| &rows.0[i]), similarity, missing: None };
            let scorer = EsQuantScorer::new(source.codes.q.clone(), similarity, query);
            let d = SourceDistance::for_query(&view, &source, query);
            let resolved: Vec<_> = handles.iter().map(|&h| d.resolve(h).unwrap()).collect();
            assert!(d.parallel_distances(&resolved, true).is_none(), "raw SIMD must not bypass codes");
            for row in &resolved {
                let expected = scorer.score_row(&source.codes, row.node.key() as usize);
                assert_eq!(d.result_score(row), Some(expected));
                let distance = d.distance_to(row);
                assert!((similarity.score_of_distance(distance as f32) - expected).abs() < 2e-6);
            }
            for filtered in [false, true] {
                let accept = |key: u64| !filtered || key % 2 == 0;
                let mut expected: Vec<_> = (0..48u64).filter(|&key| accept(key))
                    .map(|key| (key, scorer.score_row(&source.codes, key as usize))).collect();
                expected.sort_unstable_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                expected.truncate(8);
                let modes = [MutableSearchMode::VespaPlain(VespaPlainParams::default()),
                    MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                        filter_estimate: Some(VespaFilterEstimate { size: 48, count: if filtered {24} else {48} }),
                        ..Default::default()
                    })];
                for mode in modes {
                    let result = view.search_mode(query, 8, 96, &source, accept, mode).unwrap();
                    assert_eq!(result.missing_source_lookups, 0);
                    assert_eq!(result.hits, expected, "{similarity:?} bits={bits} osq={osq} filtered={filtered}");
                }
            }
        }
    }
}

#[test]
fn missing_encoded_row_never_falls_back_to_raw_scoring() {
    let rows = Rows(vec![vec![1., 0.], vec![0., 1.]]);
    let graph = MutableHnsw::new(2, Similarity::L2, HnswParams::default());
    let handles: Vec<_> = (0..2).map(|key| graph.insert(key, &rows).unwrap()).collect();
    let q = EsQuant { bits: 4, osq: false, confidence_interval: None }.fit(Similarity::L2, 2, |i| &rows.0[i]);
    let source = Encoded { rows: &rows, handles: &handles, codes: EsCodes::build(q, Similarity::L2, 2, 2, |i| &rows.0[i]),
        similarity: Similarity::L2, missing: Some(0) };
    let view = graph.snapshot();
    let d = SourceDistance::for_query(&view, &source, &rows.0[0]);
    let row = d.resolve(handles[0]).unwrap();
    assert!(d.distance_to(&row).is_infinite());
    assert_eq!(d.result_score(&row), None);
    assert_eq!(d.missing_source.get(), 2);
    // A raw (unbound) distance still scores f32 rows; a row resolved for a
    // bound quantized scorer carries no f32 (it is scored from its codes),
    // so each distance resolves its own rows.
    let construction = SourceDistance::new(&view, &source, &rows.0[0]);
    let raw = construction.resolve(handles[0]).unwrap();
    assert_eq!(construction.distance_to(&raw), 0.0);
}
