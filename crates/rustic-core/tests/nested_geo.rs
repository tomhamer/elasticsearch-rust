//! Block join (nested docs) and geo points against brute-force oracles,
//! across the memtable, sealed segments, the multi-threaded writer,
//! deletes/updates and merges.

use rustic_core::geo::{haversine_meters, point_in_polygon, quantize};
use rustic_core::nested::NestedScoreMode;
use rustic_core::query::{Query, SearchOptions, Searcher, SortField, SortKind, SortMode, SortMissing, SortValue};
use rustic_core::{DocAddress, Document, Field, FieldType, Index, IndexSettings, Term};
use std::collections::{BTreeMap, BTreeSet};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 31) as f64
    }
}

const NAME: u16 = 0;
const COLOR: u16 = 1;
const N: u16 = 2;
const LOC: u16 = 3;
const TAG: u16 = 4;

fn schema() -> Vec<Field> {
    vec![
        Field::new(NAME, "name", FieldType::Keyword),
        Field::new(COLOR, "c.color", FieldType::Keyword),
        Field::new(N, "c.n", FieldType::I64),
        Field::new(LOC, "loc", FieldType::GeoPoint),
        Field::new(TAG, "c.d.tag", FieldType::Keyword),
    ]
}

/// One parent of the oracle: its children `(color, n, grandchild tags)`.
#[derive(Clone, Debug)]
struct Parent {
    kids: Vec<(String, i64, Vec<String>)>,
}

const COLORS: [&str; 3] = ["red", "blue", "green"];

fn make_parent(r: &mut Rng) -> Parent {
    let k = r.below(5) as usize;
    let kids = (0..k)
        .map(|_| {
            let tags = (0..r.below(3)).map(|_| format!("t{}", r.below(3))).collect();
            (COLORS[r.below(3) as usize].to_string(), r.below(100) as i64, tags)
        })
        .collect();
    Parent { kids }
}

fn to_doc(name: &str, p: &Parent) -> Document {
    let mut d = Document::new();
    d.add_keyword(NAME, name);
    for (c, n, tags) in &p.kids {
        let mut k = Document::new();
        k.add_keyword(COLOR, c);
        k.add_i64(N, *n);
        for t in tags {
            let mut g = Document::new();
            g.add_keyword(TAG, t);
            k.add_nested("c.d", g);
        }
        d.add_nested("c", k);
    }
    d
}

/// name -> hits of a query (name, score, leaf, doc).
fn run(index: &Index, q: &Query, sort: Vec<SortField>) -> Vec<(String, f32, usize, u32, Vec<SortValue>)> {
    let leaves = index.leaves();
    let td = Searcher::new(leaves.clone()).search(q, &SearchOptions { top_k: 100_000, sort, ..Default::default() });
    td.hits
        .iter()
        .map(|h| {
            let d = leaves[h.leaf].get_doc(h.doc).unwrap().unwrap();
            (d.get(NAME).unwrap().as_text().unwrap().to_string(), h.score, h.leaf, h.doc, h.sort.clone())
        })
        .collect()
}

fn names(hits: &[(String, f32, usize, u32, Vec<SortValue>)]) -> BTreeSet<String> {
    hits.iter().map(|h| h.0.clone()).collect()
}

fn check_nested(index: &Index, live: &BTreeMap<String, Parent>) {
    check_nested_ranges(index, live);
    // Non-nested queries never see child docs.
    assert_eq!(names(&run(index, &Query::MatchAll, vec![])), live.keys().cloned().collect::<BTreeSet<_>>());
    assert!(run(index, &Query::term(COLOR, "red"), vec![]).is_empty(), "child docs leak into top-level queries");
    assert_eq!(Searcher::new(index.leaves()).count(&Query::MatchAll, Default::default()), live.len() as u64);

    for color in COLORS {
        let inner = Query::Term(Term::new_text(COLOR, color));
        let want: BTreeMap<&String, usize> = live
            .iter()
            .map(|(k, p)| (k, p.kids.iter().filter(|c| c.0 == color).count()))
            .filter(|(_, n)| *n > 0)
            .collect();
        let max = run(index, &Query::Nested { filter: false, path: "c".into(), query: Box::new(inner.clone()), score_mode: NestedScoreMode::Max }, vec![]);
        let sum = run(index, &Query::Nested { filter: false, path: "c".into(), query: Box::new(inner.clone()), score_mode: NestedScoreMode::Sum }, vec![]);
        let avg = run(index, &Query::Nested { filter: false, path: "c".into(), query: Box::new(inner.clone()), score_mode: NestedScoreMode::Avg }, vec![]);
        let none = run(index, &Query::Nested { filter: false, path: "c".into(), query: Box::new(inner.clone()), score_mode: NestedScoreMode::None }, vec![]);
        assert_eq!(names(&max), want.keys().map(|k| (*k).clone()).collect::<BTreeSet<_>>(), "color {color}");
        assert_eq!(names(&sum), names(&max));
        let maxs: BTreeMap<String, f32> = max.iter().map(|h| (h.0.clone(), h.1)).collect();
        for h in &sum {
            let n = want[&h.0] as f32;
            assert!((h.1 - n * maxs[&h.0]).abs() < 1e-3 * n.max(1.0), "sum {} vs {} x {}", h.1, n, maxs[&h.0]);
        }
        for h in &avg {
            assert!((h.1 - maxs[&h.0]).abs() < 1e-4);
        }
        assert!(none.iter().all(|h| h.1 == 0.0));

        // A conjunction seeks the join to the other clause's parents: the
        // same parents restricted to those names, same scores.
        for &pick in &[0usize, 3, 7] {
            let picked: Vec<&String> = live.keys().skip(pick).step_by(5).collect();
            let parents = Query::TermSet { field: NAME, terms: picked.iter().map(|n| n.as_bytes().to_vec()).collect(), boost: 1.0 };
            let conj = Query::Bool {
                must: vec![Query::Nested { filter: false, path: "c".into(), query: Box::new(inner.clone()), score_mode: NestedScoreMode::Max }],
                should: vec![],
                must_not: vec![],
                filter: vec![parents],
                minimum_should_match: rustic_core::query::boolean::MinShouldMatch::Count(0),
            };
            let got = run(index, &conj, vec![]);
            let want_names: BTreeSet<String> = names(&max).into_iter().filter(|n| picked.contains(&n)).collect();
            assert_eq!(names(&got), want_names, "conjunction color {color} pick {pick}");
            for h in &got {
                assert!((h.1 - maxs[&h.0]).abs() < 1e-4);
            }
        }

        // inner_hits: matching children with their offsets.
        let leaves = index.leaves();
        let ctx = rustic_core::query::SearchContext::for_query(&max_query(color), &leaves, Default::default());
        // A shared cursor over parents in (leaf, doc) order gives the same inner hits.
        let mut by_doc: Vec<_> = max.iter().collect();
        by_doc.sort_by_key(|h| (h.2, h.3));
        let mut cursor = None;
        for h in &by_doc {
            let a = max_query(color).nested_inner_hits_with(&mut cursor, &leaves[h.2], &ctx, h.3);
            let b = max_query(color).nested_inner_hits(&leaves[h.2], &ctx, h.3);
            assert_eq!(a.iter().map(|x| (x.doc, x.offset)).collect::<Vec<_>>(), b.iter().map(|x| (x.doc, x.offset)).collect::<Vec<_>>());
        }
        for h in &max {
            let ih = max_query(color).nested_inner_hits(&leaves[h.2], &ctx, h.3);
            let want_offsets: Vec<u32> =
                live[&h.0].kids.iter().enumerate().filter(|(_, c)| c.0 == color).map(|(i, _)| i as u32).collect();
            assert_eq!(ih.iter().map(|x| x.offset).collect::<Vec<_>>(), want_offsets, "{}", h.0);
            assert!(ih.iter().all(|x| leaves[h.2].nested_parent(x.doc) == Some(h.3)));
        }
    }

    // Nested range + bool inside the join (all conditions on one child).
    let q = Query::Nested { filter: false,
        path: "c".into(),
        query: Box::new(Query::bool_query().filter(Query::term(COLOR, "blue")).filter(Query::RangeI64 { field: N, lower: rustic_core::query::Bound::Included(30), upper: rustic_core::query::Bound::Excluded(70) }).build()),
        score_mode: NestedScoreMode::Avg,
    };
    let want: BTreeSet<String> = live.iter().filter(|(_, p)| p.kids.iter().any(|c| c.0 == "blue" && c.1 >= 30 && c.1 < 70)).map(|(k, _)| k.clone()).collect();
    assert_eq!(names(&run(index, &q, vec![])), want);

    // Two-level nesting: nested(c, nested(c.d, tag)).
    let q = Query::Nested { filter: false,
        path: "c".into(),
        query: Box::new(Query::Nested { filter: false, path: "c.d".into(), query: Box::new(Query::term(TAG, "t1")), score_mode: NestedScoreMode::Max }),
        score_mode: NestedScoreMode::Max,
    };
    let want: BTreeSet<String> = live.iter().filter(|(_, p)| p.kids.iter().any(|c| c.2.iter().any(|t| t == "t1"))).map(|(k, _)| k.clone()).collect();
    assert_eq!(names(&run(index, &q, vec![])), want);
    // At top level a deeper path joins straight to the root (ES parent
    // filter semantics).
    let q2 = Query::Nested { filter: false, path: "c.d".into(), query: Box::new(Query::term(TAG, "t1")), score_mode: NestedScoreMode::Max };
    assert_eq!(names(&run(index, &q2, vec![])), want);
    // A join to a level at or above the context matches nothing.
    let q3 = Query::Nested { filter: false, path: "c".into(), query: Box::new(Query::Nested { filter: false, path: "c".into(), query: Box::new(Query::MatchAll), score_mode: NestedScoreMode::Max }), score_mode: NestedScoreMode::Max };
    assert!(run(index, &q3, vec![]).is_empty());
    // reverse_nested style navigation.
    let leaves = index.leaves();
    for h in run(index, &q2, vec![]).iter().take(20) {
        for g in leaves[h.2].nested_children(h.3, "c.d") {
            assert_eq!(leaves[h.2].nested_ancestor(g, ""), Some(h.3));
            let c = leaves[h.2].nested_ancestor(g, "c").unwrap();
            assert_eq!(leaves[h.2].nested_path(c), "c");
            assert_eq!(leaves[h.2].nested_parent(g), Some(c));
        }
    }

    // Nested sort: min of the children's n ascending, missing last.
    let sort = vec![SortField::Nested { path: "c".into(), filter: None, sort: Box::new(SortField::field(N, SortKind::I64)) }];
    let hits = run(index, &Query::MatchAll, sort);
    let mut got = Vec::new();
    for h in &hits {
        got.push(match &h.4[0] {
            SortValue::I64(v) => Some(*v),
            SortValue::Missing => None,
            other => panic!("{other:?}"),
        });
        assert_eq!(got.last().unwrap(), &live[&h.0].kids.iter().map(|c| c.1).min(), "{}", h.0);
    }
    let present: Vec<i64> = got.iter().flatten().copied().collect();
    assert!(present.windows(2).all(|w| w[0] <= w[1]));
    assert!(got.iter().skip_while(|v| v.is_some()).all(|v| v.is_none()), "missing sort last");
    // Every reduction mode uses the same independent child-value oracle,
    // including a filter that leaves some parents with no eligible children.
    for mode in [SortMode::Min, SortMode::Max, SortMode::Sum, SortMode::Avg, SortMode::Median] {
        for descending in [false, true] {
            let sort = vec![SortField::Nested {
                path: "c".into(), filter: Some(Box::new(Query::term(COLOR, "red"))),
                sort: Box::new(SortField::Field { field: N, kind: SortKind::I64,
                    descending, missing: SortMissing::Value(SortValue::I64(-999)), mode: Some(mode) }),
            }];
            for hit in run(index, &Query::MatchAll, sort) {
                let mut values: Vec<_> = live[&hit.0].kids.iter().filter(|c| c.0 == "red").map(|c| c.1).collect();
                values.sort();
                let expected = if values.is_empty() { SortValue::I64(-999) } else {
                    match mode {
                        SortMode::Min => SortValue::I64(values[0]),
                        SortMode::Max => SortValue::I64(*values.last().unwrap()),
                        SortMode::Sum => SortValue::I64(values.iter().sum()),
                        SortMode::Avg => SortValue::I64((values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64 + 0.5).floor() as i64),
                        SortMode::Median => {
                            let middle = values.len() / 2;
                            SortValue::I64(if values.len() % 2 == 0 {
                                ((values[middle - 1] as f64 + values[middle] as f64) / 2.0 + 0.5).floor() as i64
                            } else { values[middle] })
                        }
                    }
                };
                assert_eq!(hit.4[0], expected, "parent {} mode {:?}", hit.0, mode);
            }
        }
    }
    // Filtered nested sort: max n among red children, descending.
    let sort = vec![SortField::Nested { path: "c".into(), filter: Some(Box::new(Query::term(COLOR, "red"))), sort: Box::new(SortField::field(N, SortKind::I64).desc()) }];
    for h in run(index, &Query::MatchAll, sort) {
        let want = live[&h.0].kids.iter().filter(|c| c.0 == "red").map(|c| c.1).max();
        match (&h.4[0], want) {
            (SortValue::I64(v), Some(w)) => assert_eq!(*v, w),
            (SortValue::Missing, None) => {}
            (g, w) => panic!("{}: {g:?} vs {w:?}", h.0),
        }
    }
}

/// Nested numeric (i64) ranges: the per-block verifier on sealed segments,
/// the generic join on memtables. Parents, per-mode and boosted scores
/// (folded in f64), the conjunction and filter contexts, bounded totals
/// with early termination, and inner hits, against the oracle.
fn check_nested_ranges(index: &Index, live: &BTreeMap<String, Parent>) {
    use rustic_core::query::{Bound, TotalHitsRelation, TrackTotalHits};
    let range = |lo: i64, hi: i64| Query::RangeI64 { field: N, lower: Bound::Included(lo), upper: Bound::Included(hi) };
    let nested = |q: Query, mode: NestedScoreMode, filter: bool| Query::Nested { filter, path: "c".into(), query: Box::new(q), score_mode: mode };
    let leaves = index.leaves();
    for (lo, hi) in [(0i64, 99i64), (10, 40), (97, 99), (50, 49), (-5, 3)] {
        let want: BTreeMap<String, usize> = live.iter().map(|(k, p)| (k.clone(), p.kids.iter().filter(|c| c.1 >= lo && c.1 <= hi).count())).filter(|(_, n)| *n > 0).collect();
        for (boost, mode) in [(1.0f32, NestedScoreMode::Avg), (1.0, NestedScoreMode::Max), (1.0, NestedScoreMode::Min), (1.0, NestedScoreMode::Sum), (1.0, NestedScoreMode::None), (2.5, NestedScoreMode::Avg), (2.5, NestedScoreMode::Sum)] {
            let child = if boost == 1.0 { range(lo, hi) } else { Query::Boost { query: Box::new(range(lo, hi)), boost } };
            let expect = |name: &str| -> f32 {
                let n = want[name];
                match mode {
                    NestedScoreMode::Sum => (0..n).map(|_| boost as f64).sum::<f64>() as f32,
                    NestedScoreMode::None => 0.0,
                    _ => boost,
                }
            };
            let q = nested(child.clone(), mode, false);
            let hits = run(index, &q, vec![]);
            assert_eq!(names(&hits), want.keys().cloned().collect::<BTreeSet<_>>(), "[{lo}, {hi}] {mode:?} x{boost}");
            for h in &hits {
                assert_eq!(h.1, expect(&h.0), "[{lo}, {hi}] {mode:?} x{boost} {}", h.0);
            }
            // Filter context: the same parents, score 0.
            let f = run(index, &nested(child.clone(), mode, true), vec![]);
            assert_eq!(names(&f), names(&hits));
            assert!(f.iter().all(|h| h.1 == 0.0));
            // A conjunction: the join verifies the other clause's parents.
            for &pick in &[0usize, 2] {
                let picked: Vec<&String> = live.keys().skip(pick).step_by(3).collect();
                let parents = Query::TermSet { field: NAME, terms: picked.iter().map(|n| n.as_bytes().to_vec()).collect(), boost: 1.0 };
                let conj = Query::Bool { must: vec![parents, q.clone()], should: vec![], must_not: vec![], filter: vec![], minimum_should_match: rustic_core::query::boolean::MinShouldMatch::Count(0) };
                let got = run(index, &conj, vec![]);
                let want_names: BTreeSet<String> = want.keys().filter(|n| picked.contains(n)).cloned().collect();
                assert_eq!(names(&got), want_names, "conjunction [{lo}, {hi}] {mode:?} pick {pick}");
                for h in &got {
                    assert_eq!(h.1, 1.0f32 + expect(&h.0));
                }
                // Bounded totals: the best page of the exhaustive ranking.
                for top in [&q, &conj] {
                    let full = Searcher::new(leaves.clone()).search(top, &SearchOptions { top_k: 100_000, ..Default::default() });
                    let lim = Searcher::new(leaves.clone()).search(top, &SearchOptions { top_k: 3, track_total_hits: TrackTotalHits::UpTo(5), ..Default::default() });
                    let page: Vec<(usize, u32, f32)> = full.hits.iter().take(3).map(|h| (h.leaf, h.doc, h.score)).collect();
                    assert_eq!(lim.hits.iter().map(|h| (h.leaf, h.doc, h.score)).collect::<Vec<_>>(), page, "[{lo}, {hi}] {mode:?} bounded page");
                    let n = full.hits.len() as u64;
                    let want_total = if n > 5 { (5, TotalHitsRelation::Gte) } else { (n, TotalHitsRelation::Eq) };
                    assert_eq!((lim.total_hits, lim.total_hits_relation), want_total, "[{lo}, {hi}] {mode:?} bounded total");
                }
            }
            // inner_hits: the matching children, in order, with offsets and the child score.
            let ctx = rustic_core::query::SearchContext::for_query(&q, &leaves, Default::default());
            for h in &hits {
                let ih = q.nested_inner_hits(&leaves[h.2], &ctx, h.3);
                let want_offsets: Vec<u32> = live[&h.0].kids.iter().enumerate().filter(|(_, c)| c.1 >= lo && c.1 <= hi).map(|(i, _)| i as u32).collect();
                assert_eq!(ih.iter().map(|x| x.offset).collect::<Vec<_>>(), want_offsets, "{}", h.0);
                assert!(ih.iter().all(|x| x.score == boost && leaves[h.2].nested_parent(x.doc) == Some(h.3)));
            }
        }
    }
}

fn max_query(color: &str) -> Query {
    Query::Nested { filter: false, path: "c".into(), query: Box::new(Query::term(COLOR, color)), score_mode: NestedScoreMode::Max }
}

fn find(index: &Index, name: &str) -> Option<DocAddress> {
    let leaves = index.leaves();
    for (i, l) in leaves.iter().enumerate() {
        let live = l.live_docs();
        if let Some(mut p) = l.postings(&Term::new_text(NAME, name)) {
            while let Some(d) = p.next() {
                if live(d) {
                    return Some(DocAddress::new(i as u32, d));
                }
            }
        }
    }
    None
}

#[test]
fn nested_blocks_match_oracle_through_writes_and_merges() {
    let mut r = Rng(42);
    let mut index = Index::new("n", schema(), IndexSettings::default());
    let mut live: BTreeMap<String, Parent> = BTreeMap::new();
    let mut next = 0;
    for round in 0..5 {
        // Batches through the multi-threaded writer and single adds.
        let batch: Vec<(String, Parent)> = (0..120)
            .map(|_| {
                next += 1;
                (format!("p{next}"), make_parent(&mut r))
            })
            .collect();
        let docs: Vec<Document> = batch.iter().map(|(n, p)| to_doc(n, p)).collect();
        let addrs = index.add_documents_parallel(&docs, 3).unwrap();
        for ((n, p), a) in batch.into_iter().zip(&addrs) {
            let leaves = index.leaves();
            assert_eq!(leaves[a.segment_id as usize].get_doc(a.doc_id).unwrap().unwrap().get(NAME).unwrap().as_text(), Some(n.as_str()));
            live.insert(n, p);
        }
        for _ in 0..20 {
            next += 1;
            let (n, p) = (format!("p{next}"), make_parent(&mut r));
            let a = index.add_document(&to_doc(&n, &p)).unwrap();
            assert!(a.doc_id >= p.kids.len() as u32);
            live.insert(n, p);
        }
        check_nested(&index, &live);
        // Deletes and updates (delete + re-add) remove whole blocks.
        let keys: Vec<String> = live.keys().cloned().collect();
        for _ in 0..25 {
            let k = &keys[r.below(keys.len() as u64) as usize];
            if !live.contains_key(k) {
                continue;
            }
            let a = find(&index, k).expect("live parent is findable");
            index.delete_doc(&a).unwrap();
            if r.below(2) == 0 {
                let p = make_parent(&mut r);
                index.add_document(&to_doc(k, &p)).unwrap();
                live.insert(k.clone(), p);
            } else {
                live.remove(k);
            }
        }
        check_nested(&index, &live);
        if round % 2 == 1 {
            index.refresh().unwrap();
            check_nested(&index, &live);
        }
        if round == 3 {
            index.force_merge().unwrap();
            assert_eq!(index.num_segments(), 1);
            check_nested(&index, &live);
        }
    }
    index.refresh().unwrap();
    index.force_merge().unwrap();
    check_nested(&index, &live);
    // Deleted blocks leave no orphaned children behind after the merge.
    let seg = &index.segments()[0];
    let n = seg.nested_info().unwrap();
    let kids: usize = live.values().map(|p| p.kids.len() + p.kids.iter().map(|k| k.2.len()).sum::<usize>()).sum();
    assert_eq!(seg.num_docs as usize, live.len() + kids);
    assert_eq!((0..seg.num_docs).filter(|&d| !n.is_root(d)).count(), kids);
}

#[test]
fn nested_negative_average_rounds_towards_positive_infinity() {
    let mut index = Index::new("negative", schema(), IndexSettings::default());
    let parent = Parent { kids: vec![("red".into(), -2, vec![]), ("red".into(), -1, vec![])] };
    index.add_document(&to_doc("p", &parent)).unwrap();
    for sealed in [false, true] {
        if sealed { index.refresh().unwrap(); index.force_merge().unwrap(); }
        for mode in [SortMode::Avg, SortMode::Median] {
            let sort = vec![SortField::Nested { path: "c".into(), filter: None,
                sort: Box::new(SortField::Field { field: N, kind: SortKind::I64,
                    descending: false, missing: SortMissing::Last, mode: Some(mode) }) }];
            let hits = run(&index, &Query::MatchAll, sort);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].4[0], SortValue::I64(-1));
        }
    }
}

#[test]
fn nested_blocks_persist_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = Rng(5);
    let mut live = BTreeMap::new();
    {
        let mut index = Index::create(dir.path(), "n", schema(), IndexSettings::default()).unwrap();
        for i in 0..50 {
            let p = make_parent(&mut r);
            index.add_document(&to_doc(&format!("p{i}"), &p)).unwrap();
            live.insert(format!("p{i}"), p);
        }
        index.refresh().unwrap();
        index.commit().unwrap();
    }
    let index = Index::open(dir.path()).unwrap();
    check_nested(&index, &live);
}

#[test]
fn geo_queries_and_sort_match_oracle() {
    let mut r = Rng(9);
    let mut index = Index::new("g", schema(), IndexSettings::default());
    // name -> quantized points
    let mut live: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for i in 0..6000 {
        let mut d = Document::new();
        let name = format!("g{i}");
        d.add_keyword(NAME, &name);
        let mut pts = Vec::new();
        for _ in 0..(i % 3) {
            // Cluster half the points around Europe for small-radius tests.
            let (lat, lon) = if r.below(2) == 0 { (r.unit() * 180.0 - 90.0, r.unit() * 360.0 - 180.0) } else { (45.0 + r.unit() * 10.0, r.unit() * 15.0) };
            d.add_geo_point(LOC, lat, lon);
            pts.push(quantize(lat, lon));
        }
        index.add_document(&d).unwrap();
        live.insert(name, pts);
        if i % 1700 == 1699 {
            index.refresh().unwrap();
        }
    }
    let check = |index: &Index, live: &BTreeMap<String, Vec<(f64, f64)>>, r: &mut Rng| {
        for q in 0..30 {
            let (query, pred): (Query, Box<dyn Fn(f64, f64) -> bool>) = match q % 3 {
                0 => {
                    let (lat, lon, m) = (40.0 + r.unit() * 20.0, r.unit() * 20.0, 1000.0 + r.unit() * 800_000.0);
                    (Query::GeoDistance { field: LOC, lat, lon, meters: m }, Box::new(move |a, b| haversine_meters(lat, lon, a, b) <= m))
                }
                1 => {
                    let (b, t) = (r.unit() * 180.0 - 90.0, r.unit() * 180.0 - 90.0);
                    let (top, bottom) = (t.max(b), t.min(b));
                    let (left, right) = (r.unit() * 360.0 - 180.0, r.unit() * 360.0 - 180.0);
                    let lon_ok = move |x: f64| if left <= right { x >= left && x <= right } else { x >= left || x <= right };
                    (Query::GeoBoundingBox { field: LOC, top, left, bottom, right }, Box::new(move |a, x| a >= bottom && a <= top && lon_ok(x)))
                }
                _ => {
                    let poly = vec![(44.0, 1.0), (56.0, 3.0), (52.0, 14.0), (47.0 + r.unit() * 3.0, 8.0), (45.0, 12.0)];
                    let p2 = poly.clone();
                    (Query::GeoPolygon { field: LOC, points: poly }, Box::new(move |a, b| point_in_polygon(&p2, a, b)))
                }
            };
            let want: BTreeSet<String> = live.iter().filter(|(_, pts)| pts.iter().any(|(a, b)| pred(*a, *b))).map(|(k, _)| k.clone()).collect();
            assert_eq!(names(&run(index, &query, vec![])), want, "{query:?}");
        }
        // Distance sort ascending; docs without points last (+inf).
        let (lat, lon) = (50.0, 7.0);
        let hits = run(index, &Query::MatchAll, vec![SortField::GeoDistance { field: LOC, lat, lon, descending: false, mode: None, plane: false }]);
        assert_eq!(hits.len(), live.len());
        let mut prev = f64::NEG_INFINITY;
        for h in &hits {
            let SortValue::F64(v) = h.4[0] else { panic!() };
            let want = live[&h.0].iter().map(|(a, b)| haversine_meters(lat, lon, *a, *b)).fold(f64::INFINITY, f64::min);
            assert!((v - want).abs() < 1e-6 || (v.is_infinite() && want.is_infinite()), "{v} vs {want}");
            assert!(v >= prev);
            prev = v;
        }
    };
    check(&index, &live, &mut r);
    // Deletes, then a merge rebuilds the geo index with renumbered docs.
    for i in (0..6000).step_by(7) {
        let name = format!("g{i}");
        let a = find(&index, &name).unwrap();
        index.delete_doc(&a).unwrap();
        live.remove(&name);
    }
    check(&index, &live, &mut r);
    index.refresh().unwrap();
    index.force_merge().unwrap();
    check(&index, &live, &mut r);
    let leaves = index.leaves();
    let some = live.iter().find(|(_, p)| p.len() == 2).unwrap();
    let a = find(&index, some.0).unwrap();
    assert_eq!(&leaves[a.segment_id as usize].geo_values(LOC, a.doc_id), some.1);
}
