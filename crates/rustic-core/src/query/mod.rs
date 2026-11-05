//! Query execution: doc-id set iteration, scoring, boolean composition,
//! and result collection.
//!
//! The core abstraction is [`DocSet`], a forward-only iterator over sorted
//! doc ids (Lucene's `DocIdSetIterator`). [`Scorer`] extends it with a
//! per-doc score. [`boolean`] composes scorers into AND/OR/NOT trees, and
//! [`collector`] gathers the top-k results. [`similarity`] holds the
//! scoring models, [`multi_term`] the multi-term rewrites, [`spans`] the
//! positional span queries, [`function_score`] score functions, and
//! [`explain`] score explanations.

pub mod bm25;
pub mod boolean;
pub mod bulk;
pub mod collector;
pub mod docset;
pub mod explain;
pub mod feature;
pub mod function_score;
pub mod geo_scorer;
pub mod multi_term;
pub mod numeric_dv;
pub mod query;
#[cfg(feature = "mutable-hnsw-poc")]
pub(crate) mod mutable_knn;
pub mod reader;
pub mod searcher;
pub mod scorer;
pub mod segment_scorer;
pub mod similarity;
mod sort_fast;
pub mod spans;
pub mod stats;

pub use bm25::{byte4_to_int, idf, int_to_byte4, Bm25Params, Bm25Weight};
pub use bulk::Bm25TermScorer;
pub use boolean::{
    union_of, BooleanScorer, Disjunction2, Exclusion, Intersection, IntersectionN,
    calculate_min_should_match, MinShouldMatch, MinShouldMatchUnion, RequiredOptionalScorer, Union,
};
pub use explain::Explanation;
pub use function_score::{BoostMode, DecayKind, FieldValueModifier, FunctionScoreScorer, Prepared as PreparedScoreFunction, ScoreFunction, ScoreMode};
pub use multi_term::{Fuzziness, MultiTermRewrite};
pub use geo_scorer::geo_pred_scorer;
pub use numeric_dv::{GeoDv, NumericDv};
pub use similarity::{
    AfterEffect, BasicModel, BooleanSim, Bm25, Classic, DfrSimilarity, Distribution, IbSimilarity, Lambda,
    LmDirichlet, LmJelinekMercer, Normalization, SimWeight, Similarity,
};
pub use spans::{IntervalsRule, SpanQuery, SpanScorer, Spans};
pub use collector::{collect_top_k, collect_top_k_live, collect_top_k_sorted, count, compare_field_docs, CountCollector, FieldDoc, ScoredDoc, SortSpec, SortValue, TopFieldCollector, TopKCollector};
pub use reader::{field_stats, term_stats, FieldStats, LeafReader, ShapeValues, TermStats};
pub use query::{hits_scorer, shared_bits_scorer, BoolBuilder, Bound, CustomQuery, InnerHitsCursor, Query, RangeDate, ScoredDocs, SearchContext};
pub use searcher::{
    Rescore, RescoreMode, SearchOptions, Searcher, SortField, SortKind, SortMissing, SortMode, TopDocs, TopHit, TotalHitsRelation, TrackTotalHits,
};
pub use docset::{AllDocSet, BitSetDocSet, DocSet, EmptyDocSet, VecDocSet, TERMINATED};
pub use scorer::{ConstScorer, Scorer, TermFreqSource, TermScorer};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn everything_is_send() {
        assert_send::<VecDocSet>();
        assert_send::<BitSetDocSet>();
        assert_send::<EmptyDocSet>();
        assert_send::<AllDocSet>();
        assert_send::<ConstScorer<VecDocSet>>();
        assert_send::<Bm25Weight>();
        assert_send::<Intersection<VecDocSet, BitSetDocSet>>();
        assert_send::<IntersectionN>();
        assert_send::<Union>();
        assert_send::<MinShouldMatchUnion>();
        assert_send::<Exclusion<ConstScorer<VecDocSet>, VecDocSet>>();
        assert_send::<RequiredOptionalScorer>();
        assert_send::<TopKCollector>();
        assert_send::<CountCollector>();
        assert_send::<ScoredDoc>();
        // Boxed composites are Send because DocSet: Send.
        assert_send::<Box<dyn DocSet>>();
        assert_send::<Box<dyn Scorer>>();
    }
}
