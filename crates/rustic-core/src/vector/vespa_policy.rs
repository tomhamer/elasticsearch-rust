//! Shared Vespa query policy; used by the native adapter and by ES kNN on
//! mutable-owned fields (`query/mutable_knn.rs`).
//!
//! Pinned Vespa 0521b00640ef934b359d2fc356f97d84499dd17d:
//! - searchcore/.../proton/matching/query.cpp:288–380: filter preparation.
//! - searchlib/.../queryeval/nearest_neighbor_blueprint.cpp:130–228: routing/budgets.
//! - searchlib/.../fef/indexproperties.cpp:356–513: defaults.
//! See docs/MUTABLE_HNSW_PLANNER_PLAN.md for links and ES compatibility limits.
//!
//! This module evaluates no predicates, constructs no bitsets, searches no graph,
//! and performs no exact-fill fallback. Its two-stage token prevents substituting
//! a measured filter ratio for the query estimate or silently changing policy
//! between preparation and execution. Lazy filters and distributed target-hit
//! resolution are outside this first native port. Input errors/checked overflow
//! are defensive Rust admission rules, not claims about Vespa's invalid inputs.
#![allow(dead_code)] // Non-default policy options remain pure-policy tests only.

use std::{error::Error, fmt};

/// The share of `num_candidates` an unfiltered graph search of a quantized
/// field explores (the ES adapter's beam width: `max(k, factor *
/// num_candidates, 100)`, see [`unfiltered_exploration`]). There one graph
/// per shard reaches its recall ceiling well before ES's per-segment search
/// does: at 0.7 dense_vector (int8_hnsw) recall stays at or above ES
/// 9.5.5's at every point the Rally track measures, and so_vector
/// (bbq_hnsw) 100/1000 goes 0.996 -> 0.992 (accepted). Raw (unquantized)
/// fields explore all of `num_candidates` and more (see
/// [`raw_exploration`]): on wiki_en_cohere_vector_int8 (byte, hnsw) one
/// graph is still far below ES's per-segment search there.
/// 1 (`RUSTIC_KNN_EXPLORATION_FACTOR=1`) explores exactly
/// `num_candidates`, as ES does.
pub const UNFILTERED_EXPLORATION_FACTOR: f64 = 0.7;

/// How much wider than `num_candidates` an unfiltered graph search of a
/// 1-bit (`bbq_hnsw`) field explores ([`binary_exploration`]). One-bit
/// codes steer a traversal coarsely, so one graph per shard with ES's
/// `num_candidates` stays far below ES's search of every segment's graph:
/// cohere 768-d `max_inner_product`, 1M documents, k 10, num_candidates 100,
/// oversample 3, recall@10 0.78 against ES 9.5.5's 0.856; exploring 2x
/// `num_candidates` 0.84, 3x 0.868, 4x 0.881 (p50/p99 1.3/1.6 ms at 1x,
/// 1.5/1.9 at 3x). 3x: above ES with a margin over run-to-run noise.
pub const BINARY_EXPLORATION_FACTOR: f64 = 3.0;

/// The widest beam [`binary_exploration`] widens to: at 1000 candidates
/// one graph is already at ES's recall (so_vector `bbq_hnsw` 100/1000:
/// 0.99), so larger requests explore exactly `num_candidates`.
pub const BINARY_EXPLORATION_CAP: usize = 1000;

/// [`BINARY_EXPLORATION_FACTOR`], or `RUSTIC_KNN_BINARY_EXPLORATION`
/// (at least 1; diagnostics, read once).
pub fn binary_exploration_factor() -> f64 {
    static FACTOR: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *FACTOR.get_or_init(|| std::env::var("RUSTIC_KNN_BINARY_EXPLORATION").ok().and_then(|v| v.parse().ok())
        .filter(|f: &f64| f.is_finite() && *f >= 1.0).unwrap_or(BINARY_EXPLORATION_FACTOR))
}

/// The beam width of an unfiltered graph search of a 1-bit field:
/// `max(k, num_candidates, min(factor * num_candidates, BINARY_EXPLORATION_CAP))`.
pub fn binary_exploration(k: usize, num_candidates: usize, factor: f64) -> usize {
    let widened = ((num_candidates as f64 * factor).ceil() as usize).min(BINARY_EXPLORATION_CAP);
    k.max(num_candidates).max(widened)
}

/// [`UNFILTERED_EXPLORATION_FACTOR`], or `RUSTIC_KNN_EXPLORATION_FACTOR`
/// (in `(0, 1]`; diagnostics, read once).
pub fn unfiltered_exploration_factor() -> f64 {
    static FACTOR: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *FACTOR.get_or_init(|| std::env::var("RUSTIC_KNN_EXPLORATION_FACTOR").ok().and_then(|v| v.parse().ok())
        .filter(|f: &f64| f.is_finite() && *f > 0.0 && *f <= 1.0).unwrap_or(UNFILTERED_EXPLORATION_FACTOR))
}

/// Candidates an unfiltered graph search always explores (when the request
/// asks for at least as many): a small beam is where one graph's recall is
/// still climbing (on the Rally corpus 10/100 at ef 70 fell below ES's
/// recall in one of two builds), and bounding it saves little.
pub const UNFILTERED_EXPLORATION_MIN: usize = 100;

/// The beam width of an unfiltered graph search of `k` hits with the
/// request's `num_candidates`: `max(k, ceil(factor * num_candidates),
/// UNFILTERED_EXPLORATION_MIN)`, never above `num_candidates` (when it is
/// at least `k`). Factor 1 is exactly `max(k, num_candidates)`.
pub fn unfiltered_exploration(k: usize, num_candidates: usize, factor: f64) -> usize {
    if factor >= 1.0 { return k.max(num_candidates); }
    let scaled = ((num_candidates as f64 * factor).ceil() as usize).max(UNFILTERED_EXPLORATION_MIN);
    k.max(scaled.min(num_candidates))
}

/// The beam width of an unfiltered graph search of a raw (unquantized)
/// field: `max(k, num_candidates, floor)`, `floor` = `factor * k` and at
/// least [`UNFILTERED_EXPLORATION_MIN`] (`factor` 0: no floor).
///
/// ES searches every segment's graph with `num_candidates`; one graph per
/// shard with the same beam loses the recall of the extra segments. On a
/// 2M-document slice of wiki_en_cohere_vector_int8 (ES 9.5.5: 18 segments)
/// ES recall@k was 0.915 / 0.954 / 0.971 at 10-20 / 10-50 / 10-100 and
/// 0.947 / 0.962 / 0.981 at 100-120 / 100-200 / 100-500; ES force merged to
/// one segment, 0.770 / 0.878 / 0.927 and 0.843 / 0.888 / 0.945, the one
/// graph's own level (0.748 / 0.878 / 0.935, 0.863 / 0.916 / 0.961 with
/// the full `num_candidates`). A floor of `2 k` (at least 100) brings every
/// point within 5% of ES's (10-20 and 10-50: 0.935, 100-120: 0.916).
pub fn raw_exploration(k: usize, num_candidates: usize, factor: f64) -> usize {
    let floor = if factor > 0.0 { ((k as f64 * factor).ceil() as usize).max(UNFILTERED_EXPLORATION_MIN) } else { 0 };
    k.max(num_candidates).max(floor)
}

/// [`raw_exploration`]'s floor factor: [`RAW_EXPLORATION_FLOOR`], or
/// `RUSTIC_KNN_RAW_FLOOR` (`0` disables; read once).
pub fn raw_exploration_floor() -> f64 {
    static FACTOR: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *FACTOR.get_or_init(|| std::env::var("RUSTIC_KNN_RAW_FLOOR").ok().and_then(|v| v.parse().ok())
        .filter(|f: &f64| f.is_finite() && *f >= 0.0).unwrap_or(RAW_EXPLORATION_FLOOR))
}

/// Default [`raw_exploration_floor`]: a raw field's unfiltered beam is at
/// least `2 k` (and 100) wide (`RUSTIC_KNN_RAW_FLOOR=0`: the request's).
pub const RAW_EXPLORATION_FLOOR: f64 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyConfig {
    pub exact_lower_limit: f64,
    pub postfilter_upper_limit: f64,
    pub filter_first_upper_limit: f64,
    pub target_hits_max_adjustment_factor: f64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            exact_lower_limit: exact_lower_limit(),
            postfilter_upper_limit: 1.0,
            filter_first_upper_limit: filter_first_upper_limit(),
            target_hits_max_adjustment_factor: 20.0,
        }
    }
}

/// Vespa's `approximate-threshold` (a rank-profile setting): estimated
/// filter hit ratios below it search exactly instead of the graph.
/// `RUSTIC_KNN_EXACT_LOWER` is a tuning seam.
fn exact_lower_limit() -> f64 {
    static LIMIT: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("RUSTIC_KNN_EXACT_LOWER").ok().and_then(|v| v.parse().ok())
        .filter(|v: &f64| (0.0..=1.0).contains(v)).unwrap_or(0.02))
}

/// Vespa's `filter-first-threshold` (a rank-profile setting): estimated
/// filter hit ratios below it take the filter-first (two-hop) traversal,
/// at or above it the plain one. 0.0 (filter-first off), Vespa's pinned
/// default. 0.15 had been chosen on so_vector (2M docs, its vector-correlated
/// 7.8% `tags: java` filter was faster filter-first, 2.03 against 2.79 ms),
/// but on random filters filter-first lost to the plain filtered beam across
/// the band (16M x 128-d int8_hnsw, 4 shards, k 10, num_candidates 100,
/// 64 clients): 3% recall@10 0.50 -> 0.93, 5% 0.59 -> 0.92 at 2,760 ->
/// 2,020 QPS, 10% 0.88 -> 0.91 at 620 -> 2,550 QPS (its third hop at
/// exploration 0.8 is what cost 10% 5.7 ms a query). A sparse exploration of
/// 0.3 (Vespa's newer default) only reached 0.68-0.75 at 2-5%.
/// `RUSTIC_KNN_FILTER_FIRST_UPPER` is a tuning seam (0.15: the old split).
fn filter_first_upper_limit() -> f64 {
    static LIMIT: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("RUSTIC_KNN_FILTER_FIRST_UPPER").ok().and_then(|v| v.parse().ok())
        .filter(|v: &f64| (0.0..=1.0).contains(v)).unwrap_or(0.0))
}

impl PolicyConfig {
    fn validate(self) -> Result<(), PolicyError> {
        for (name, ratio) in [
            ("exact_lower_limit", self.exact_lower_limit),
            ("postfilter_upper_limit", self.postfilter_upper_limit),
            ("filter_first_upper_limit", self.filter_first_upper_limit),
        ] {
            if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
                return Err(PolicyError::InvalidConfig(name));
            }
        }
        if self.exact_lower_limit > self.postfilter_upper_limit {
            return Err(PolicyError::InvalidConfig("lower exceeds upper"));
        }
        if !self.target_hits_max_adjustment_factor.is_finite()
            || self.target_hits_max_adjustment_factor <= 0.0
        {
            return Err(PolicyError::InvalidConfig("target adjustment factor"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparationInput {
    pub approximate: bool,
    pub has_index: bool,
    /// Whole-query estimate before GlobalFilter construction. Never inferred
    /// here from active graph nodes or the eventual exact bitset count.
    pub estimated_query_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    ApproximationDisabled,
    IndexUnavailable,
    EstimatedRatioBelowExactThreshold,
    ActualRatioBelowExactThreshold,
    ActiveRatioBelowFilterFirstThreshold,
    ActiveRatioAtOrAboveFilterFirstThreshold,
    EstimatedRatioAbovePostfilterThreshold,
    ConstructedFilterMatchedAll,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterPreparation {
    ExactWithoutGlobalFilter(Reason),
    BuildGlobalFilter,
    InactivePostFilter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterStatistics {
    /// Document-domain filter universe. This is NOT the policy denominator or
    /// graph node-ID limit. Pass it unchanged to the visited-work estimator.
    pub size: u32,
    /// Count of the prepared upper-bound filter; not necessarily the exact
    /// final conjunction count. Counting belongs outside this pure policy.
    pub count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedGlobalFilter {
    NotBuilt,
    /// Inactive has no valid size/count, matching Vespa's filter interface.
    Inactive,
    Active(FilterStatistics),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HitBudget {
    /// Already resolved per-content-node target; not ES num_candidates.
    pub target_hits: u32,
    pub explore_additional_hits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Exact,
    Plain,
    FilterFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterApplication {
    /// The outer query drives candidate documents for exact distance evaluation.
    ExactWithOuterQuery,
    ExactWithGlobalFilter,
    AnnPrefilter,
    /// The graph receives no predicate. Apply the outer query afterward; this
    /// can yield fewer hits and must NOT replace an ES knn.filter prefilter.
    AnnUnfilteredThenOuterQuery,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryPlan {
    pub route: Route,
    pub reason: Reason,
    pub filter_application: FilterApplication,
    pub requested_target_hits: u32,
    pub adjusted_target_hits: u32,
    /// None for exact; otherwise adjusted target plus additional exploration.
    /// Both are nonnegative, so this is also Vespa's max(k, explore_k).
    pub exploration_budget: Option<u32>,
    pub original_estimated_query_ratio: f64,
    pub effective_estimated_query_ratio: f64,
    /// Allocated attribute document-ID space, including cleared holes; never
    /// graph.active_nodes(). Zero is allowed only when no ratio is evaluated.
    pub attribute_document_count: u32,
    pub active_filter_ratio: Option<f64>,
    pub filter_statistics: Option<FilterStatistics>,
    /// Exact route on a quantized field that rescores: score codes, not f32
    /// rows (set by the caller; the policy itself leaves it false).
    pub exact_codes: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    InvalidConfig(&'static str),
    InvalidEstimatedRatio,
    ZeroTargetHits,
    UnexpectedFilterPreparation,
    FilterCountExceedsSize,
    ZeroAttributeDocumentCount,
    AdjustedTargetOutOfRange,
    ExplorationBudgetOverflow,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid native Vespa policy input: {self:?}")
    }
}
impl Error for PolicyError {}

#[derive(Clone, Copy, Debug)]
pub struct PreparedPolicy {
    config: PolicyConfig,
    estimated_ratio: f64,
    preparation: FilterPreparation,
}

pub fn choose_filter_preparation(
    input: PreparationInput,
    config: PolicyConfig,
) -> Result<PreparedPolicy, PolicyError> {
    config.validate()?;
    if !input.estimated_query_ratio.is_finite()
        || !(0.0..=1.0).contains(&input.estimated_query_ratio)
    {
        return Err(PolicyError::InvalidEstimatedRatio);
    }
    let preparation = if !input.approximate {
        FilterPreparation::ExactWithoutGlobalFilter(Reason::ApproximationDisabled)
    } else if !input.has_index {
        FilterPreparation::ExactWithoutGlobalFilter(Reason::IndexUnavailable)
    } else if input.estimated_query_ratio < config.exact_lower_limit {
        FilterPreparation::ExactWithoutGlobalFilter(Reason::EstimatedRatioBelowExactThreshold)
    } else if input.estimated_query_ratio <= config.postfilter_upper_limit {
        FilterPreparation::BuildGlobalFilter
    } else {
        FilterPreparation::InactivePostFilter
    };
    Ok(PreparedPolicy {
        config,
        estimated_ratio: input.estimated_query_ratio,
        preparation,
    })
}

impl PreparedPolicy {
    pub fn preparation(self) -> FilterPreparation {
        self.preparation
    }

    /// Complete the exact decision made by choose_filter_preparation. This
    /// consumes neither a predicate nor graph state: callers must obtain these
    /// statistics from the same prepared filter/read view they will execute.
    pub fn finish(
        self,
        budget: HitBudget,
        attribute_document_count: u32,
        filter: PreparedGlobalFilter,
    ) -> Result<QueryPlan, PolicyError> {
        if budget.target_hits == 0 {
            return Err(PolicyError::ZeroTargetHits);
        }
        let (route, reason, application, effective_ratio, actual_ratio, statistics) =
            match (self.preparation, filter) {
                (
                    FilterPreparation::ExactWithoutGlobalFilter(reason),
                    PreparedGlobalFilter::NotBuilt,
                ) => (
                    Route::Exact,
                    reason,
                    FilterApplication::ExactWithOuterQuery,
                    self.estimated_ratio,
                    None,
                    None,
                ),
                (FilterPreparation::BuildGlobalFilter, PreparedGlobalFilter::Active(stats)) => {
                    if stats.count > stats.size {
                        return Err(PolicyError::FilterCountExceedsSize);
                    }
                    if attribute_document_count == 0 {
                        return Err(PolicyError::ZeroAttributeDocumentCount);
                    }
                    // Do not clamp to stats.size or graph slots. The pinned
                    // denominator is the tensor attribute's allocated doc count.
                    let ratio = f64::from(stats.count) / f64::from(attribute_document_count);
                    let (route, reason, application) = if ratio < self.config.exact_lower_limit {
                        (
                            Route::Exact,
                            Reason::ActualRatioBelowExactThreshold,
                            FilterApplication::ExactWithGlobalFilter,
                        )
                    } else if ratio < self.config.filter_first_upper_limit {
                        (
                            Route::FilterFirst,
                            Reason::ActiveRatioBelowFilterFirstThreshold,
                            FilterApplication::AnnPrefilter,
                        )
                    } else {
                        (
                            Route::Plain,
                            Reason::ActiveRatioAtOrAboveFilterFirstThreshold,
                            FilterApplication::AnnPrefilter,
                        )
                    };
                    (
                        route,
                        reason,
                        application,
                        self.estimated_ratio,
                        Some(ratio),
                        Some(stats),
                    )
                }
                (FilterPreparation::BuildGlobalFilter, PreparedGlobalFilter::Inactive) => (
                    Route::Plain,
                    Reason::ConstructedFilterMatchedAll,
                    FilterApplication::AnnUnfilteredThenOuterQuery,
                    1.0, // Query::handle_global_filter normalization, not input mutation.
                    None,
                    None,
                ),
                (FilterPreparation::InactivePostFilter, PreparedGlobalFilter::Inactive) => (
                    Route::Plain,
                    Reason::EstimatedRatioAbovePostfilterThreshold,
                    FilterApplication::AnnUnfilteredThenOuterQuery,
                    self.estimated_ratio,
                    None,
                    None,
                ),
                _ => return Err(PolicyError::UnexpectedFilterPreparation),
            };
        let adjusted_target_hits = if application == FilterApplication::AnnUnfilteredThenOuterQuery
        {
            adjusted_target(
                budget.target_hits,
                effective_ratio,
                self.config.target_hits_max_adjustment_factor,
            )?
        } else {
            budget.target_hits
        };
        let exploration_budget = if route == Route::Exact {
            None // Unused additional exploration cannot overflow an exact request.
        } else {
            Some(
                adjusted_target_hits
                    .checked_add(budget.explore_additional_hits)
                    .ok_or(PolicyError::ExplorationBudgetOverflow)?,
            )
        };
        Ok(QueryPlan {
            route,
            reason,
            filter_application: application,
            requested_target_hits: budget.target_hits,
            adjusted_target_hits,
            exploration_budget,
            original_estimated_query_ratio: self.estimated_ratio,
            effective_estimated_query_ratio: effective_ratio,
            attribute_document_count,
            active_filter_ratio: actual_ratio,
            filter_statistics: statistics,
            exact_codes: false,
        })
    }
}

fn adjusted_target(target: u32, estimated_ratio: f64, cap: f64) -> Result<u32, PolicyError> {
    if estimated_ratio <= 0.0 {
        // Matches the native blueprint guard. Normal two-stage planning does
        // not enter post-filter adjustment for a zero estimate.
        return Ok(target);
    }
    let adjusted = (f64::from(target) / estimated_ratio)
        .min(f64::from(target) * cap)
        .trunc();
    if !adjusted.is_finite() || adjusted < 1.0 || adjusted > f64::from(u32::MAX) {
        return Err(PolicyError::AdjustedTargetOutOfRange);
    }
    Ok(adjusted as u32)
}

#[cfg(test)]
mod tests {
    /// A 1-bit field's unfiltered beam: `num_candidates` widened by the
    /// factor up to the cap, never below `k` or `num_candidates`.
    #[test]
    fn binary_exploration_widens_small_beams_up_to_the_cap() {
        assert_eq!(super::binary_exploration(30, 100, 4.0), 400);
        assert_eq!(super::binary_exploration(30, 300, 4.0), super::BINARY_EXPLORATION_CAP);
        assert_eq!(super::binary_exploration(30, 1000, 4.0), 1000);
        assert_eq!(super::binary_exploration(30, 5000, 4.0), 5000);
        assert_eq!(super::binary_exploration(2000, 100, 4.0), 2000);
        assert_eq!(super::binary_exploration(10, 100, 1.0), 100);
    }

    #[test]
    fn raw_exploration_is_at_least_num_candidates_and_its_floor() {
        use super::raw_exploration as e;
        for k in [1usize, 10, 100, 1000] {
            for nc in [1usize, k, 2 * k, 10 * k, 10_000] {
                // No floor: exactly ES's beam, max(k, num_candidates).
                assert_eq!(e(k, nc, 0.0), k.max(nc));
                let floored = e(k, nc, 2.0);
                assert_eq!(floored, k.max(nc).max((2 * k).max(100)));
            }
        }
        // The Rally points (k-num_candidates).
        assert_eq!([e(10, 20, 2.0), e(10, 50, 2.0), e(10, 100, 2.0)], [100, 100, 100]);
        assert_eq!([e(100, 120, 2.0), e(100, 200, 2.0), e(100, 500, 2.0)], [200, 200, 500]);
    }

    #[test]
    fn unfiltered_exploration_is_bounded_by_num_candidates_and_factor_one_is_es() {
        use super::unfiltered_exploration as e;
        for k in [1usize, 10, 100, 1000] {
            for nc in [1usize, k, k + 1, 2 * k, 10 * k, 1000, 10_000] {
                // Off (factor 1): exactly the old beam, max(k, num_candidates).
                assert_eq!(e(k, nc, 1.0), k.max(nc));
                let half = e(k, nc, 0.5);
                assert_eq!(half, k.max(nc.div_ceil(2).max(100).min(nc)));
                assert!(half >= k && half <= k.max(nc));
            }
        }
        assert_eq!(e(100, 1000, 0.5), 500);
        assert_eq!(e(100, 1000, 0.7), 700);
        assert_eq!(e(10, 100, 0.7), 100, "small beams are explored whole");
        assert_eq!(e(10, 50, 0.7), 50);
        assert_eq!(e(10, 120, 0.7), 100);
        assert_eq!(e(100, 300, 0.7), 210);
        assert_eq!(e(100, 100, 0.5), 100);
        assert_eq!(e(10, 10, 0.4), 10);
    }
    use super::*;

    fn prepared(estimate: f64, config: PolicyConfig) -> PreparedPolicy {
        choose_filter_preparation(
            PreparationInput {
                approximate: true,
                has_index: true,
                estimated_query_ratio: estimate,
            },
            config,
        )
        .unwrap()
    }

    fn budget() -> HitBudget {
        HitBudget {
            target_hits: 10,
            explore_additional_hits: 90,
        }
    }

    /// The defaults with a filter-first band (2% to 15%), to exercise its
    /// boundaries.
    fn filter_first_config() -> PolicyConfig {
        PolicyConfig { filter_first_upper_limit: 0.15, ..PolicyConfig::default() }
    }

    #[test]
    fn defaults_match_pinned_policy_without_rustic_absolute_scan_thresholds() {
        let c = PolicyConfig::default();
        assert_eq!(c.exact_lower_limit, 0.02);
        assert_eq!(c.postfilter_upper_limit, 1.0);
        assert_eq!(c.filter_first_upper_limit, 0.0);
        assert_eq!(c.target_hits_max_adjustment_factor, 20.0);
        let p = prepared(1.0, c)
            .finish(
                budget(),
                5,
                PreparedGlobalFilter::Active(FilterStatistics { size: 5, count: 5 }),
            )
            .unwrap();
        assert_eq!(p.route, Route::Plain, "no absolute small-graph exact rule");
    }

    #[test]
    fn unavailable_index_and_disabled_approximation_never_require_a_filter() {
        for (approximate, has_index, reason) in [
            (false, true, Reason::ApproximationDisabled),
            (false, false, Reason::ApproximationDisabled),
            (true, false, Reason::IndexUnavailable),
        ] {
            let p = choose_filter_preparation(
                PreparationInput {
                    approximate,
                    has_index,
                    estimated_query_ratio: 0.5,
                },
                PolicyConfig::default(),
            )
            .unwrap();
            assert_eq!(
                p.preparation(),
                FilterPreparation::ExactWithoutGlobalFilter(reason)
            );
            let plan = p
                .finish(
                    HitBudget {
                        target_hits: 10,
                        explore_additional_hits: u32::MAX,
                    },
                    0,
                    PreparedGlobalFilter::NotBuilt,
                )
                .unwrap();
            assert_eq!(plan.route, Route::Exact);
            assert_eq!(plan.reason, reason);
            assert_eq!(
                plan.filter_application,
                FilterApplication::ExactWithOuterQuery
            );
            assert_eq!(plan.exploration_budget, None);
        }
    }

    #[test]
    fn filter_preparation_keeps_pinned_strict_lower_and_inclusive_upper() {
        let c = PolicyConfig {
            postfilter_upper_limit: 0.3,
            ..PolicyConfig::default()
        };
        assert_eq!(
            prepared(0.019, c).preparation(),
            FilterPreparation::ExactWithoutGlobalFilter(Reason::EstimatedRatioBelowExactThreshold)
        );
        for estimate in [0.02, 0.1, 0.3] {
            assert_eq!(
                prepared(estimate, c).preparation(),
                FilterPreparation::BuildGlobalFilter
            );
        }
        assert_eq!(
            prepared(0.301, c).preparation(),
            FilterPreparation::InactivePostFilter
        );
        assert_eq!(
            prepared(1.0, PolicyConfig::default()).preparation(),
            FilterPreparation::BuildGlobalFilter
        );
    }

    #[test]
    fn measured_filter_ratio_selects_exact_filter_first_and_plain_at_boundaries() {
        let preparation = prepared(0.5, filter_first_config());
        for (count, route, reason) in [
            (0, Route::Exact, Reason::ActualRatioBelowExactThreshold),
            (19, Route::Exact, Reason::ActualRatioBelowExactThreshold),
            (
                20,
                Route::FilterFirst,
                Reason::ActiveRatioBelowFilterFirstThreshold,
            ),
            (
                149,
                Route::FilterFirst,
                Reason::ActiveRatioBelowFilterFirstThreshold,
            ),
            (
                150,
                Route::Plain,
                Reason::ActiveRatioAtOrAboveFilterFirstThreshold,
            ),
        ] {
            let stats = FilterStatistics { size: 2000, count };
            let plan = preparation
                .finish(budget(), 1000, PreparedGlobalFilter::Active(stats))
                .unwrap();
            assert_eq!(plan.route, route);
            assert_eq!(plan.reason, reason);
            assert_eq!(plan.active_filter_ratio, Some(f64::from(count) / 1000.0));
            assert_eq!(plan.filter_statistics, Some(stats));
            assert_eq!(
                plan.adjusted_target_hits, 10,
                "active filters never inflate k"
            );
            assert_eq!(
                plan.exploration_budget,
                if route == Route::Exact {
                    None
                } else {
                    Some(100)
                }
            );
        }
    }

    #[test]
    fn document_holes_and_filter_universe_do_not_become_graph_live_count() {
        let p = prepared(0.1, filter_first_config());
        // The same filter has 50 accepted docs in a 20K filter universe.
        // A 10K allocated attribute has holes; using 1K live nodes instead
        // would incorrectly choose filter-first rather than exact.
        let filter = PreparedGlobalFilter::Active(FilterStatistics {
            size: 20_000,
            count: 50,
        });
        let holes = p.finish(budget(), 10_000, filter).unwrap();
        let compact_doc_domain = p.finish(budget(), 1_000, filter).unwrap();
        assert_eq!(holes.route, Route::Exact);
        assert_eq!(compact_doc_domain.route, Route::FilterFirst);
        assert_eq!(holes.filter_statistics.unwrap().size, 20_000);
        let wider_filter = p
            .finish(
                budget(),
                100,
                PreparedGlobalFilter::Active(FilterStatistics {
                    size: 200,
                    count: 150,
                }),
            )
            .unwrap();
        assert_eq!(wider_filter.active_filter_ratio, Some(1.5));
        assert_eq!(wider_filter.route, Route::Plain);
        // A separate graph ID limit is intentionally absent from this API: it
        // must remain captured by the graph read view, not used as denominator.
    }

    #[test]
    fn an_inactive_constructed_filter_normalizes_estimate_but_not_provenance() {
        let p = prepared(0.1, PolicyConfig::default())
            .finish(budget(), 1000, PreparedGlobalFilter::Inactive)
            .unwrap();
        assert_eq!(p.reason, Reason::ConstructedFilterMatchedAll);
        assert_eq!(p.original_estimated_query_ratio, 0.1);
        assert_eq!(p.effective_estimated_query_ratio, 1.0);
        assert_eq!(p.adjusted_target_hits, 10);
        assert_eq!(p.exploration_budget, Some(100));
        assert_eq!(p.filter_statistics, None);
        assert_eq!(p.active_filter_ratio, None);
    }

    #[test]
    fn postfilter_budget_truncates_caps_and_preserves_filter_application() {
        let c = PolicyConfig {
            postfilter_upper_limit: 0.5,
            ..PolicyConfig::default()
        };
        let p = prepared(0.8, c)
            .finish(budget(), 1000, PreparedGlobalFilter::Inactive)
            .unwrap();
        assert_eq!(
            p.adjusted_target_hits, 12,
            "integer truncation, not rounding up"
        );
        assert_eq!(p.exploration_budget, Some(102));
        assert_eq!(
            p.filter_application,
            FilterApplication::AnnUnfilteredThenOuterQuery
        );
        let forced = PolicyConfig {
            exact_lower_limit: 0.0,
            postfilter_upper_limit: 0.0,
            ..c
        };
        let capped = prepared(0.001, forced)
            .finish(budget(), 1000, PreparedGlobalFilter::Inactive)
            .unwrap();
        assert_eq!(capped.adjusted_target_hits, 200);
        assert_eq!(capped.exploration_budget, Some(290));
        assert_eq!(adjusted_target(10, 0.0, 20.0), Ok(10));
    }

    #[test]
    fn phases_cannot_substitute_active_filters_or_skip_required_preparation() {
        let active = PreparedGlobalFilter::Active(FilterStatistics {
            size: 1000,
            count: 100,
        });
        let exact = prepared(0.001, PolicyConfig::default());
        let build = prepared(0.5, PolicyConfig::default());
        let post = prepared(
            0.8,
            PolicyConfig {
                postfilter_upper_limit: 0.5,
                ..PolicyConfig::default()
            },
        );
        for (p, wrong) in [
            (exact, active),
            (exact, PreparedGlobalFilter::Inactive),
            (build, PreparedGlobalFilter::NotBuilt),
            (post, active),
            (post, PreparedGlobalFilter::NotBuilt),
        ] {
            assert_eq!(
                p.finish(budget(), 1000, wrong),
                Err(PolicyError::UnexpectedFilterPreparation)
            );
        }
    }

    #[test]
    fn invalid_domains_and_nonfinite_policy_inputs_fail_explicitly() {
        let p = prepared(0.5, PolicyConfig::default());
        assert_eq!(
            p.finish(
                budget(),
                100,
                PreparedGlobalFilter::Active(FilterStatistics { size: 2, count: 3 })
            ),
            Err(PolicyError::FilterCountExceedsSize)
        );
        assert_eq!(
            p.finish(
                budget(),
                0,
                PreparedGlobalFilter::Active(FilterStatistics { size: 0, count: 0 })
            ),
            Err(PolicyError::ZeroAttributeDocumentCount)
        );
        assert_eq!(
            p.finish(
                HitBudget {
                    target_hits: 0,
                    explore_additional_hits: 100
                },
                100,
                PreparedGlobalFilter::Inactive
            ),
            Err(PolicyError::ZeroTargetHits)
        );
        for ratio in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            assert!(matches!(
                choose_filter_preparation(
                    PreparationInput {
                        approximate: true,
                        has_index: true,
                        estimated_query_ratio: ratio
                    },
                    PolicyConfig::default()
                ),
                Err(PolicyError::InvalidEstimatedRatio)
            ));
        }
        for config in [
            PolicyConfig {
                exact_lower_limit: f64::NAN,
                ..PolicyConfig::default()
            },
            PolicyConfig {
                postfilter_upper_limit: -0.1,
                ..PolicyConfig::default()
            },
            PolicyConfig {
                exact_lower_limit: 0.8,
                postfilter_upper_limit: 0.5,
                ..PolicyConfig::default()
            },
            PolicyConfig {
                filter_first_upper_limit: f64::INFINITY,
                ..PolicyConfig::default()
            },
            PolicyConfig {
                target_hits_max_adjustment_factor: 0.0,
                ..PolicyConfig::default()
            },
        ] {
            assert!(matches!(
                choose_filter_preparation(
                    PreparationInput {
                        approximate: true,
                        has_index: true,
                        estimated_query_ratio: 0.5
                    },
                    config
                ),
                Err(PolicyError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn adjusted_target_and_exploration_overflow_do_not_saturate_or_wrap() {
        let post = prepared(
            0.5,
            PolicyConfig {
                postfilter_upper_limit: 0.4,
                ..PolicyConfig::default()
            },
        );
        assert_eq!(
            post.finish(
                HitBudget {
                    target_hits: u32::MAX,
                    explore_additional_hits: 0
                },
                100,
                PreparedGlobalFilter::Inactive
            ),
            Err(PolicyError::AdjustedTargetOutOfRange)
        );
        let unfiltered = prepared(1.0, PolicyConfig::default());
        assert_eq!(
            unfiltered.finish(
                HitBudget {
                    target_hits: u32::MAX,
                    explore_additional_hits: 1
                },
                100,
                PreparedGlobalFilter::Inactive
            ),
            Err(PolicyError::ExplorationBudgetOverflow)
        );
        assert_eq!(
            unfiltered
                .finish(
                    HitBudget {
                        target_hits: u32::MAX,
                        explore_additional_hits: 0
                    },
                    100,
                    PreparedGlobalFilter::Inactive
                )
                .unwrap()
                .exploration_budget,
            Some(u32::MAX)
        );
    }

    #[test]
    fn target_budget_does_not_invent_a_short_hit_exact_fallback() {
        let plan = prepared(0.5, PolicyConfig::default())
            .finish(
                HitBudget {
                    target_hits: 100,
                    explore_additional_hits: 0,
                },
                5,
                PreparedGlobalFilter::Active(FilterStatistics { size: 5, count: 1 }),
            )
            .unwrap();
        assert_eq!(
            plan.route,
            Route::Plain,
            "ratio equals 20%, despite count<target"
        );
        assert_eq!(plan.filter_application, FilterApplication::AnnPrefilter);
        assert_eq!(plan.adjusted_target_hits, 100);
        assert_eq!(plan.exploration_budget, Some(100));
    }
}
