//! Opt-in native comparison of the production ordinal-filter planner policy.
//! Filter evaluation/translation is prepared outside clocks. This is not an
//! Elasticsearch, payload-index, quantization, PIT, or mixed-write benchmark.
use super::*;
use rustic_core::vector::{PlannerConfig, Quantization, hnsw::{FilterMode, VespaPlainParams}};

#[derive(Clone, Debug)]
pub(super) struct Settings {
    config: PlannerConfig,
    pool_threads: usize,
    exact_partitions: Option<usize>,
    vespa_defaults: bool,
    vespa_policy: vespa_policy::PolicyConfig,
}

impl Settings {
    pub(super) fn initialize(threads: usize) -> AnyResult<Self> {
        let pool_threads = rustic_core::query::searcher::in_search_pool(rayon::current_num_threads);
        if pool_threads != threads {
            return Err(format!("dedicated query pool has {pool_threads} threads, expected {threads}; set RAYON_NUM_THREADS={threads} before starting the planner POC").into());
        }
        Ok(Self {
            config: PlannerConfig {
                quantization: Quantization::None,
                ..Default::default()
            },
            pool_threads,
            exact_partitions: None,
            vespa_defaults: false,
            vespa_policy: vespa_policy::PolicyConfig::default(),
        })
    }

    pub(super) fn exact_diagnostic(threads: usize, partitions: usize) -> AnyResult<Self> {
        if partitions == 0 || partitions > threads {
            return Err("exact query threads must be in 1..=pool threads".into());
        }
        let mut settings = Self::initialize(threads)?;
        settings.exact_partitions = Some(partitions);
        Ok(settings)
    }

    pub(super) fn with_exact_lower_limit(mut self, limit: f64) -> AnyResult<Self> {
        if !limit.is_finite() || !(0.0..=1.0).contains(&limit) {
            return Err("vespa-exact-lower-limit must be finite and in [0,1]".into());
        }
        self.vespa_policy.exact_lower_limit = limit;
        Ok(self)
    }

    pub(super) fn manifest(&self) -> Value {
        json!({
            "policy": if self.vespa_defaults { if self.vespa_policy == vespa_policy::PolicyConfig::default() { "native_vespa_default_policy" } else { "native_vespa_configured_policy" } } else if self.exact_partitions.is_some() { "explicit_exact_static_docid_partition_diagnostic" } else { "production_thresholds_precomputed_live_ordinals_v1" },
            "vespa_defaults": self.vespa_defaults && self.vespa_policy == vespa_policy::PolicyConfig::default(),
            "vespa_exact_lower_limit": self.vespa_policy.exact_lower_limit,
            "vespa_policy_tuned": self.vespa_policy != vespa_policy::PolicyConfig::default(),
            "vespa_estimate_source": "fixture nominal selectivity, independently of prepared live count; not an ES estimator",
            "vespa_budget_mapping": "target_hits=k; explore_additional_hits=max(ef-k,0); no ES num_candidates mapping",
            "vespa_scope": "explicit recorded thresholds; bounded task-range exact workers; no thread-bundle admission, adaptive donation, outer-query execution or production integration",
            "exact_query_threads": self.exact_partitions,
            "exact_partition_scope": "Vespa static document-range partition model; shared Rayon scheduling, not Vespa thread-bundle admission or adaptive range donation",
            "full_scan_threshold": self.config.full_scan_threshold,
            "full_scan_threshold_kb": self.config.full_scan_threshold_kb,
            "full_scan_fraction": self.config.full_scan_fraction,
            "two_hop_fraction": self.config.two_hop_fraction,
            "query_pool_threads": self.pool_threads,
            "filter_representation": "precomputed live ordinal bitsets and accepted ordinal lists",
            "excluded_from_timing": "filter evaluation, owner-doc translation, bitset/count/list construction",
            "exact_parallel_min_accepted": 4096,
            "exact_chunk_rows": 1024,
            "vespa_exact_task_ranges_per_worker": 8,
            "vespa_exact_threshold": "shared query heap with ES score and docid tie semantics; angular scan does not early-exit",
            "vespa_exact_min_rows_per_worker": 1024,
            "exact_scheduler_constants": "Rustic policy settings, not claimed as Vespa defaults",
            "quantization": "none; original f32 cosine",
            "payload_overlay": "absent in this fixture",
            "baseline": "production exact/Plain/TwoHop policy plus short-hit exact fallback",
            "candidate": if self.vespa_defaults { "Vespa configured exact/Plain/filter-first dispatch without short-hit fallback" } else { "whole-graph exact or Plain with short-hit exact fallback; TwoHop not implemented" },
            "scope": "native ordinal-filter strategy comparison; not the full ES filter/planner path or sustainable mixed-load acceptance"
        })
    }

    pub(super) fn vespa_defaults(threads: usize, partitions: usize) -> AnyResult<Self> {
        let mut settings = Self::exact_diagnostic(threads, partitions)?;
        settings.vespa_defaults = true;
        Ok(settings)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Empty,
    Exact,
    Plain,
    TwoHop,
    FilterFirst,
}

#[derive(Clone, Copy)]
struct NativeVespaPolicy {
    token: vespa_policy::PreparedPolicy,
    global_filter: vespa_policy::PreparedGlobalFilter,
    document_count: u32,
}
impl NativeVespaPolicy {
    fn prepare(rows: &AcceptedRows, filter: Filter, config: vespa_policy::PolicyConfig) -> Self {
        use vespa_policy::*;
        let estimated_query_ratio = match filter { Filter::One => 0.01, Filter::All => 1.0, Filter::Ten => 0.1, Filter::Half => 0.5 };
        let token = choose_filter_preparation(PreparationInput {
            approximate: true, has_index: true, estimated_query_ratio,
        }, config).expect("validated native policy configuration");
        let stats = FilterStatistics { size: rows.filter_estimate.size, count: rows.filter_estimate.count };
        let global_filter = match token.preparation() {
            FilterPreparation::ExactWithoutGlobalFilter(_) => PreparedGlobalFilter::NotBuilt,
            FilterPreparation::BuildGlobalFilter if stats.count == stats.size => PreparedGlobalFilter::Inactive,
            FilterPreparation::BuildGlobalFilter => PreparedGlobalFilter::Active(stats),
            FilterPreparation::InactivePostFilter => unreachable!("default upper threshold is one"),
        };
        Self { token, global_filter, document_count: stats.size }
    }

    fn plan(self, k: usize, ef: usize) -> vespa_policy::QueryPlan {
        self.token.finish(vespa_policy::HitBudget {
            target_hits: u32::try_from(k).expect("validated native k"),
            explore_additional_hits: u32::try_from(ef.saturating_sub(k)).expect("validated native beam"),
        }, self.document_count, self.global_filter).expect("prepared native policy inputs")
    }

    fn route(self) -> Route {
        match self.plan(1, 1).route {
            vespa_policy::Route::Exact => Route::Exact,
            vespa_policy::Route::Plain => Route::Plain,
            vespa_policy::Route::FilterFirst => Route::FilterFirst,
        }
    }
}

fn route(
    config: &PlannerConfig,
    dim: usize,
    n: usize,
    matching: usize,
    has_graph: bool,
    filtered: bool,
) -> Route {
    if matching == 0 {
        return Route::Empty;
    }
    // Keep inequalities identical to vector/store.rs::plan_segment_search.
    // These fixture graphs have no payload overlay and always score f32.
    if !has_graph
        || matching <= config.full_scan_limit(dim as u32)
        || matching as f32 <= config.full_scan_fraction * n as f32
    {
        Route::Exact
    } else if filtered && (matching as f32) < config.two_hop_fraction * n as f32 {
        Route::TwoHop
    } else {
        Route::Plain
    }
}

#[derive(Clone)]
struct AcceptedRows {
    start: usize,
    end: usize,
    words: Vec<u64>,
    // Global corpus row IDs also index Corpus.norms, including churn extras.
    ordinals: Vec<u32>,
    filter_estimate: VespaFilterEstimate,
}
impl AcceptedRows {
    fn prepare(start: usize, end: usize, visible: &[bool], filter: Filter) -> Self {
        let mut words = vec![0; (end - start).div_ceil(64)];
        let mut ordinals = Vec::new();
        for key in start..end {
            if visible.get(key).copied().unwrap_or(false) && filter.accept(key as u64) {
                words[(key - start) / 64] |= 1u64 << ((key - start) % 64);
                ordinals.push(key as u32);
            }
        }
        let filter_estimate = VespaFilterEstimate {
            size: u32::try_from(end - start).expect("validated document-ID universe"),
            count: u32::try_from(ordinals.len()).expect("accepted rows within universe"),
        };
        Self {
            start,
            end,
            words,
            ordinals,
            filter_estimate,
        }
    }
    fn contains_local(&self, ordinal: u32) -> bool {
        let ordinal = ordinal as usize;
        self.words
            .get(ordinal / 64)
            .is_some_and(|w| (w >> (ordinal % 64)) & 1 != 0)
    }
    fn contains_key(&self, key: u64) -> bool {
        key >= self.start as u64
            && key < self.end as u64
            && self.contains_local((key - self.start as u64) as u32)
    }
    fn bytes(&self) -> usize {
        self.words.len() * 8 + self.ordinals.len() * 4
    }
}

pub(super) struct PreparedFilter {
    settings: Settings,
    parts: Vec<AcceptedRows>,
    tail: Option<AcceptedRows>,
    whole: Option<AcceptedRows>,
    routes: Vec<Route>,
    vespa: Option<NativeVespaPolicy>,
}
impl PreparedFilter {
    pub(super) fn prepare(
        engine: &Engine,
        corpus: &Corpus,
        visible: &[bool],
        filter: Filter,
        settings: &Settings,
    ) -> Self {
        match engine {
            Engine::Immutable { parts, tail, .. } => {
                let accepted: Vec<_> = parts
                    .iter()
                    .map(|part| AcceptedRows::prepare(part.start, part.end, visible, filter))
                    .collect();
                let routes = accepted
                    .iter()
                    .map(|rows| {
                        route(
                            &settings.config,
                            corpus.dims,
                            rows.end - rows.start,
                            rows.ordinals.len(),
                            true,
                            true,
                        )
                    })
                    .collect();
                Self {
                    settings: settings.clone(),
                    parts: accepted,
                    tail: Some(AcceptedRows::prepare(*tail, corpus.len(), visible, filter)),
                    whole: None,
                    routes,
                    vespa: None,
                }
            }
            Engine::Mutable { graph, .. } => {
                let whole = AcceptedRows::prepare(0, corpus.len(), visible, filter);
                let vespa = settings.vespa_defaults.then(|| NativeVespaPolicy::prepare(&whole, filter, settings.vespa_policy));
                let selected = route(
                    &settings.config,
                    corpus.dims,
                    graph.active_nodes() as usize,
                    whole.ordinals.len(),
                    true,
                    true,
                );
                Self {
                    settings: settings.clone(),
                    parts: Vec::new(),
                    tail: None,
                    whole: Some(whole),
                    routes: vec![if let Some(policy) = vespa { policy.route() } else if settings.exact_partitions.is_some() { Route::Exact } else { selected }],
                    vespa,
                }
            }
        }
    }

    pub(super) fn report(&self) -> Value {
        let description = |r: &AcceptedRows, route: Route| json!({"start":r.start,"end":r.end,"accepted":r.ordinals.len(),"route":format!("{route:?}")});
        let parts: Vec<_> = self
            .parts
            .iter()
            .zip(&self.routes)
            .map(|(rows, &r)| description(rows, r))
            .collect();
        json!({"configuration":self.settings.manifest(),"parts":parts,
            "tail":self.tail.as_ref().map(|r|description(r, if r.ordinals.is_empty(){Route::Empty}else{Route::Exact})),
            "whole_graph":self.whole.as_ref().map(|r|description(r,self.routes[0])),
            "vespa_decision":self.vespa.map(|p| { let d=p.plan(1,1); json!({"route":format!("{:?}",d.route),"reason":format!("{:?}",d.reason),"application":format!("{:?}",d.filter_application),"estimated_ratio":d.original_estimated_query_ratio,"actual_ratio":d.active_filter_ratio,"attribute_document_count":d.attribute_document_count,"budget_note":"route-only report; execution uses requested k and max(k,ef)"}) }),
            "prepared_bytes":self.parts.iter().map(AcceptedRows::bytes).sum::<usize>()+self.tail.as_ref().map_or(0,AcceptedRows::bytes)+self.whole.as_ref().map_or(0,AcceptedRows::bytes),
            "candidate_two_hop_missing":self.whole.is_some()&&self.routes[0]==Route::TwoHop})
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct QueryStats {
    pub exact_paths: u64,
    pub plain_graph_paths: u64,
    pub two_hop_paths: u64,
    pub filter_first_paths: u64,
    pub exact_fallbacks: u64,
    pub missing_two_hop_paths: u64,
    pub exact_rows_scored: u64,
}
impl QueryStats {
    fn add(&mut self, other: Self) {
        self.exact_paths += other.exact_paths;
        self.plain_graph_paths += other.plain_graph_paths;
        self.two_hop_paths += other.two_hop_paths;
        self.filter_first_paths += other.filter_first_paths;
        self.exact_fallbacks += other.exact_fallbacks;
        self.missing_two_hop_paths += other.missing_two_hop_paths;
        self.exact_rows_scored += other.exact_rows_scored;
    }
    pub(super) fn json(self) -> Value {
        json!({"exact_paths":self.exact_paths,"plain_graph_paths":self.plain_graph_paths,"two_hop_paths":self.two_hop_paths,"filter_first_paths":self.filter_first_paths,"exact_fallbacks":self.exact_fallbacks,"missing_two_hop_paths":self.missing_two_hop_paths,"exact_rows_scored":self.exact_rows_scored})
    }
    pub(super) fn total(values: impl Iterator<Item = Self>) -> Self {
        let mut total = Self::default();
        for value in values {
            total.add(value);
        }
        total
    }
}

pub(super) struct Outcome {
    pub hits: Vec<(u64, f32)>,
    pub vector_ops: u64,
    pub entry_fallback: u64,
    pub missing_source: u64,
    pub stats: QueryStats,
}
fn finish(mut hits: Vec<(u64, f32)>, k: usize) -> Vec<(u64, f32)> {
    hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    hits.truncate(k);
    hits
}

fn exact(corpus: &Corpus, query: &[f32], k: usize, rows: &AcceptedRows) -> Outcome {
    exact_partitioned(corpus, query, k, rows, None)
}

fn exact_partitioned(corpus: &Corpus, query: &[f32], k: usize, rows: &AcceptedRows, partitions: Option<usize>) -> Outcome {
    exact_partitioned_source(corpus, query, k, rows, partitions, None, false)
}

fn exact_partitioned_source(corpus: &Corpus, query: &[f32], k: usize, rows: &AcceptedRows, partitions: Option<usize>,
    owned: Option<(&MutableSourceLease<'_>, &[NodeHandle])>, task_scheduling: bool) -> Outcome {
    let scan = |ids: &[u32]| {
        rustic_core::vector::flat::poc_exact_top_k(
            query,
            Similarity::Cosine,
            k,
            ids.iter().map(|&key| {
                (
                    key,
                    if let Some((source, handles)) = owned {
                        source.get_node(handles[key as usize], key as u64).expect("prepared owned live row exists")
                    } else {
                        corpus.row(key as u64).expect("prepared live row exists")
                    },
                )
            }),
            Some(&corpus.norms),
        )
    };
    let chunks = if task_scheduling {
        use rustic_core::vector::exact_schedule::ExactTaskRanges;
        let threads = partitions.unwrap_or(1);
        let workers = ExactTaskRanges::workers(threads, rows.ordinals.len(), 1024);
        // More document ranges than workers lets sparse/skewed filters share
        // work without creating one Rayon task and one result heap per range.
        let ranges = ExactTaskRanges::new(rows.start..rows.end, workers.saturating_mul(8));
        let shared = rustic_core::vector::exact_shared::SharedExactTopK::new(k);
        let run = |_| {
            let mut operations = 0;
            while let Some(range) = ranges.claim() {
                let lo = rows.ordinals.partition_point(|&id| (id as usize) < range.start);
                let hi = rows.ordinals.partition_point(|&id| (id as usize) < range.end);
                operations += shared.scan(query, Similarity::Cosine, rows.ordinals[lo..hi].iter().map(|&key| {
                    let row = if let Some((source, handles)) = owned {
                        source.get_node(handles[key as usize], key as u64).expect("owned live row")
                    } else { corpus.row(key as u64).expect("fixture live row") };
                    (key, row)
                }));
            }
            operations
        };
        let operations = if workers == 1 { run(0) } else {
            rustic_core::query::searcher::in_search_pool(||
                (0..workers).into_par_iter().map(run).sum())
        };
        vec![(shared.finish(), operations)]
    } else if let Some(partitions) = partitions {
        // Partition document space, not accepted ordinals or HNSW nodes.
        // Empty/skewed ranges are deliberately retained in this static mode.
        let run = |part: usize| {
            let width = rows.end - rows.start;
            let start = rows.start + width / partitions * part + (width % partitions).min(part);
            let end = start + width / partitions + usize::from(part < width % partitions);
            let lo = rows.ordinals.partition_point(|&id| (id as usize) < start);
            let hi = rows.ordinals.partition_point(|&id| (id as usize) < end);
            scan(&rows.ordinals[lo..hi])
        };
        if partitions == 1 { vec![run(0)] } else {
            rustic_core::query::searcher::in_search_pool(||
                (0..partitions).into_par_iter().map(run).collect())
        }
    } else if rows.ordinals.len() < 4096 {
        vec![scan(&rows.ordinals)]
    } else {
        rustic_core::query::searcher::in_search_pool(|| {
            rows.ordinals.par_chunks(1024).map(scan).collect()
        })
    };
    let mut hits = Vec::new();
    let mut operations = 0;
    for (chunk, count) in chunks {
        hits.extend(chunk.into_iter().map(|(key, score)| (key as u64, score)));
        operations += count;
    }
    Outcome {
        hits: finish(hits, k),
        vector_ops: operations,
        entry_fallback: 0,
        missing_source: 0,
        stats: QueryStats {
            exact_paths: u64::from(!rows.ordinals.is_empty()),
            exact_rows_scored: operations,
            ..Default::default()
        },
    }
}

fn short_hits(k: usize, matching: usize, actual: usize) -> bool {
    actual < k.min(matching)
}

impl PreparedFilter {
    fn immutable_part(
        &self,
        part: &Part,
        rows: &AcceptedRows,
        selected: Route,
        corpus: &Corpus,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Outcome {
        if matches!(selected, Route::Empty | Route::Exact) {
            return exact(corpus, query, k, rows);
        }
        let source = RangeSource {
            corpus,
            start: part.start,
            end: part.end,
        };
        let mode = if selected == Route::TwoHop {
            FilterMode::TwoHop { memo: false }
        } else {
            FilterMode::Plain
        };
        let before = hnsw::poc_vector_ops();
        let hits = part.graph.search_mode(
            query,
            k,
            ef,
            &source,
            Similarity::Cosine,
            |ordinal| rows.contains_local(ordinal),
            mode,
        );
        let operations = hnsw::poc_vector_ops().wrapping_sub(before);
        let mut stats = QueryStats {
            plain_graph_paths: u64::from(selected == Route::Plain),
            two_hop_paths: u64::from(selected == Route::TwoHop),
            ..Default::default()
        };
        if short_hits(k, rows.ordinals.len(), hits.len()) {
            let mut fallback = exact(corpus, query, k, rows);
            stats.exact_fallbacks += 1;
            stats.add(fallback.stats);
            fallback.stats = stats;
            fallback.vector_ops += operations;
            return fallback;
        }
        Outcome {
            hits: hits
                .into_iter()
                .map(|(ordinal, score)| (part.start as u64 + ordinal as u64, score))
                .collect(),
            vector_ops: operations,
            entry_fallback: 0,
            missing_source: 0,
            stats,
        }
    }

    fn search(
        &self,
        engine: &Engine,
        corpus: &Corpus,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Outcome {
        match engine {
            Engine::Immutable { parts, .. } => {
                // VectorStore avoids a query-pool dispatch for one segment.
                // A physically present tail counts even if its filter is empty.
                if parts.len() == 1
                    && self
                        .tail
                        .as_ref()
                        .is_some_and(|tail| tail.start == tail.end)
                {
                    return self.immutable_part(
                        &parts[0],
                        &self.parts[0],
                        self.routes[0],
                        corpus,
                        query,
                        k,
                        ef,
                    );
                }
                let outcomes: Vec<_> = rustic_core::query::searcher::in_search_pool(|| {
                    (0..=parts.len())
                        .into_par_iter()
                        .map(|i| {
                            if i == parts.len() {
                                exact(corpus, query, k, self.tail.as_ref().unwrap())
                            } else {
                                self.immutable_part(
                                    &parts[i],
                                    &self.parts[i],
                                    self.routes[i],
                                    corpus,
                                    query,
                                    k,
                                    ef,
                                )
                            }
                        })
                        .collect()
                });
                let mut combined = Outcome {
                    hits: Vec::new(),
                    vector_ops: 0,
                    entry_fallback: 0,
                    missing_source: 0,
                    stats: QueryStats::default(),
                };
                for outcome in outcomes {
                    combined.hits.extend(outcome.hits);
                    combined.vector_ops += outcome.vector_ops;
                    combined.stats.add(outcome.stats);
                }
                combined.hits = finish(combined.hits, k);
                combined
            }
            Engine::Mutable { graph, owner, ordinals } => {
                let source = engine.source_read(corpus);
                let rows = self.whole.as_ref().unwrap();
                let selected = self.routes[0];
                if matches!(selected, Route::Empty | Route::Exact) {
                    return exact_partitioned_source(corpus, query, k, rows, self.settings.exact_partitions,
                        owner.as_ref().map(|_| (&source, ordinals.as_slice())), self.settings.vespa_defaults);
                }
                if let Some(policy) = self.vespa {
                    let decision = policy.plan(k, ef);
                    let width = decision.exploration_budget.expect("ANN budget") as usize;
                    let target = decision.adjusted_target_hits as usize;
                    let result = if selected == Route::FilterFirst {
                        graph.search_mode(query, target, width, &source, |key| rows.contains_key(key),
                            MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                                filter_estimate: Some(rows.filter_estimate), ..Default::default()
                            })).expect("prepared filter-first metadata")
                    } else if decision.filter_statistics.is_some() {
                        graph.search_mode(query, target, width, &source,
                            |key| rows.contains_key(key), MutableSearchMode::VespaPlain(VespaPlainParams {
                                filter_estimate: Some(rows.filter_estimate), ..Default::default()
                            })).expect("prepared plain metadata")
                    } else {
                        // With default upper=1, inactive means every document matched.
                        assert_eq!(rows.ordinals.len(), policy.document_count as usize);
                        graph.search_mode(query, target, width, &source, |_| true,
                            MutableSearchMode::VespaPlain(VespaPlainParams::default())).expect("unfiltered Plain parameters")
                    };
                    // Vespa has no short-hit exact-fill fallback here.
                    return Outcome { hits: result.hits, vector_ops: result.vector_ops,
                        entry_fallback: result.entry_fallback_scanned, missing_source: result.missing_source_lookups,
                        stats: QueryStats { plain_graph_paths: u64::from(selected == Route::Plain),
                            filter_first_paths: u64::from(selected == Route::FilterFirst), ..Default::default() } };
                }
                // No mutable TwoHop implementation yet. Record the missing
                // capability and run its existing Plain path, never downgrade
                // the immutable baseline's selective traversal to match it.
                let result = graph
                    .search_with_filter_estimate(
                        query, k, ef, &source,
                        |key| rows.contains_key(key),
                        rows.filter_estimate,
                    )
                    .expect("prepared document filter metadata");
                let stats = QueryStats {
                    plain_graph_paths: 1,
                    missing_two_hop_paths: u64::from(selected == Route::TwoHop),
                    ..Default::default()
                };
                let mut outcome = Outcome {
                    hits: result.hits,
                    vector_ops: result.vector_ops,
                    entry_fallback: result.entry_fallback_scanned,
                    missing_source: result.missing_source_lookups,
                    stats,
                };
                if short_hits(k, rows.ordinals.len(), outcome.hits.len()) {
                    let fallback = exact(corpus, query, k, rows);
                    outcome.hits = fallback.hits;
                    outcome.vector_ops += fallback.vector_ops;
                    outcome.stats.add(fallback.stats);
                    outcome.stats.exact_fallbacks += 1;
                }
                outcome
            }
        }
    }
}

pub(super) fn search(
    engine: &Engine,
    corpus: &Corpus,
    visible: &[bool],
    query: &[f32],
    k: usize,
    ef: usize,
    filter: Filter,
    plan: Option<&PreparedFilter>,
) -> Outcome {
    if let Some(plan) = plan {
        return plan.search(engine, corpus, query, k, ef);
    }
    let (hits, vector_ops, entry_fallback, missing_source) =
        engine.search(corpus, visible, query, k, ef, filter);
    Outcome {
        hits,
        vector_ops,
        entry_fallback,
        missing_source,
        stats: QueryStats::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vespa_runtime_dispatch_preserves_direct_search_and_exact_visibility() {
        let values: Vec<f32> = (0..500).flat_map(|i| [(i + 1) as f32, (i % 31 + 1) as f32]).collect();
        let mut mapping = memmap2::MmapMut::map_anon(values.len() * 4).unwrap();
        for (bytes, value) in mapping.chunks_exact_mut(4).zip(&values) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        let corpus = Corpus { data: mapping.make_read_only().unwrap(), dims: 2, base: 500,
            extras: Vec::new(), norms: values.chunks_exact(2).map(|v| v.iter().map(|x| x*x).sum()).collect() };
        let graph = MutableHnsw::new(2, Similarity::Cosine, HnswParams::default());
        for key in 0..500 { graph.insert(key, &corpus).unwrap(); }
        let engine = Engine::Mutable { graph: Arc::new(graph), owner: None, ordinals: Vec::new() };
        let settings = Settings { config: PlannerConfig::default(), pool_threads: 2,
            exact_partitions: Some(2), vespa_defaults: true, vespa_policy: vespa_policy::PolicyConfig::default() };
        let query = [2., 3.];
        let Engine::Mutable { graph, .. } = &engine else { unreachable!() };
        for filter in [Filter::All, Filter::Ten, Filter::Half] {
            let prepared = PreparedFilter::prepare(&engine, &corpus, &vec![true; 500], filter, &settings);
            let expected = match filter {
                Filter::One => unreachable!("exact route covered separately below"),
                Filter::All => graph.search(&query, 10, 100, &corpus, |_| true),
                Filter::Half => graph.search_with_filter_estimate(&query, 10, 100, &corpus,
                    |key| filter.accept(key), VespaFilterEstimate { size: 500, count: 250 }).unwrap(),
                Filter::Ten => graph.search_mode(&query, 10, 100, &corpus, |key| filter.accept(key),
                    MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
                        filter_estimate: Some(VespaFilterEstimate { size: 500, count: 50 }), ..Default::default()
                    })).unwrap(),
            };
            let actual = prepared.search(&engine, &corpus, &query, 10, 100);
            assert_eq!(actual.hits, expected.hits);
            assert_eq!(actual.vector_ops, expected.vector_ops);
            assert_eq!(actual.stats.exact_fallbacks, 0);
            assert_eq!(actual.stats.filter_first_paths, u64::from(matches!(filter, Filter::Ten)));
            assert_eq!(prepared.vespa.unwrap().plan(10, 100).exploration_budget, Some(100));
        }
        // Selectivity below Vespa's default lower threshold must exercise the
        // shared exact collector, including fewer accepted rows than requested k.
        let exact_prepared = PreparedFilter::prepare(&engine, &corpus, &vec![true; 500], Filter::One, &settings);
        assert_eq!(exact_prepared.routes, vec![Route::Exact]);
        let exact_actual = exact_prepared.search(&engine, &corpus, &query, 10, 100);
        assert_eq!(exact_actual.vector_ops, 5);
        assert_eq!(exact_actual.hits, exact(&corpus, &query, 10, exact_prepared.whole.as_ref().unwrap()).hits);
        assert!(settings.clone().with_exact_lower_limit(f64::NAN).is_err());
        let tuned = settings.clone().with_exact_lower_limit(0.11).unwrap();
        let tuned_prepared = PreparedFilter::prepare(&engine, &corpus, &vec![true; 500], Filter::Ten, &tuned);
        assert_eq!(tuned_prepared.routes, vec![Route::Exact]);
        assert!(tuned.manifest()["vespa_policy_tuned"].as_bool().unwrap());
        // Allocated document count, not live graph nodes, is the ratio denominator.
        let mut visible = vec![false; 500];
        visible[30] = true;
        let prepared = PreparedFilter::prepare(&engine, &corpus, &visible, Filter::Ten, &settings);
        assert_eq!(prepared.routes, vec![Route::Exact]);
        let actual = prepared.search(&engine, &corpus, &query, 10, 100);
        assert_eq!(actual.hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![30]);
        assert_eq!(actual.vector_ops, 1);
        assert_eq!(actual.stats.exact_fallbacks, 0);
        assert_eq!(actual.stats.exact_rows_scored, 1);
    }

    #[test]
    fn static_exact_partitions_preserve_filtered_top_k_with_holes_and_ties() {
        let values: Vec<f32> = (0..513).flat_map(|i| [(i % 17 + 1) as f32, (i % 7 + 1) as f32]).collect();
        let mut mapping = memmap2::MmapMut::map_anon(values.len() * 4).unwrap();
        for (bytes, value) in mapping.chunks_exact_mut(4).zip(&values) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        let corpus = Corpus { data: mapping.make_read_only().unwrap(), dims: 2, base: 513,
            extras: Vec::new(), norms: values.chunks_exact(2).map(|v| v.iter().map(|x| x*x).sum()).collect() };
        let mut visible = vec![true; 513];
        for i in (0..513).step_by(30) { visible[i] = false; }
        for filter in [Filter::Ten, Filter::All] {
            let rows = AcceptedRows::prepare(17, 511, &visible, filter);
            for k in [1, 10, 1000] {
                let reference = exact_partitioned(&corpus, &[2., 3.], k, &rows, Some(1));
                for threads in [2, 4, 8, 32, 512] {
                    let actual = exact_partitioned(&corpus, &[2., 3.], k, &rows, Some(threads));
                    assert_eq!(actual.hits, reference.hits, "threads={threads}, k={k}");
                    assert_eq!(actual.vector_ops, rows.ordinals.len() as u64);
                    let scheduled = exact_partitioned_source(&corpus, &[2., 3.], k, &rows, Some(threads), None, true);
                    assert_eq!(scheduled.hits, reference.hits, "task threads={threads}, k={k}");
                    assert_eq!(scheduled.vector_ops, rows.ordinals.len() as u64);
                }
            }
        }
        let empty = AcceptedRows::prepare(0, 513, &vec![false; 513], Filter::All);
        let result = exact_partitioned(&corpus, &[2., 3.], 10, &empty, Some(32));
        assert!(result.hits.is_empty());
        assert_eq!(result.vector_ops, 0);
    }

    #[test]
    fn planner_routes_match_production_boundaries_and_dimension_cap() {
        let config = PlannerConfig::default();
        assert_eq!(route(&config, 96, 100_000, 0, true, true), Route::Empty);
        for matching in [9_999, 10_000] {
            assert_eq!(
                route(&config, 96, 100_000, matching, true, true),
                Route::Exact
            );
        }
        assert_eq!(
            route(&config, 96, 100_000, 10_001, true, true),
            Route::TwoHop
        );
        assert_eq!(
            route(&config, 96, 100_000, 29_999, true, true),
            Route::TwoHop
        );
        assert_eq!(
            route(&config, 96, 100_000, 30_001, true, true),
            Route::Plain
        );
        // Preserve the production f32 arithmetic, including its rounding;
        // 0.3f32 * 1_000_000 is an exactly represented boundary.
        assert_eq!(
            route(&config, 96, 1_000_000, 299_999, true, true),
            Route::TwoHop
        );
        assert_eq!(
            route(&config, 96, 1_000_000, 300_000, true, true),
            Route::Plain
        );
        assert_eq!(
            route(&config, 96, 1_000_000, 50_000, true, true),
            Route::Exact
        );
        assert_eq!(
            route(&config, 96, 1_000_000, 50_001, true, true),
            Route::TwoHop
        );
        assert_eq!(
            route(&config, 1024, 100_000, 4_000, true, false),
            Route::Exact
        ); // Fraction still applies.
        assert_eq!(
            route(&config, 1024, 10_000, 2_501, true, true),
            Route::TwoHop
        );
        assert_eq!(route(&config, 128, 10_000, 2_501, true, true), Route::Exact);
        assert_eq!(
            route(&config, 96, 100_000, 100_000, false, false),
            Route::Exact
        );
        assert_eq!(
            route(&config, 96, 100_000, 20_000, true, false),
            Route::Plain
        );
    }
    #[test]
    fn planner_prepared_ordinals_preserve_counts_visibility_and_churn() {
        let mut visible = vec![true; 130];
        for key in [0, 10, 70] {
            visible[key] = false;
        }
        let rows = AcceptedRows::prepare(64, 130, &visible, Filter::Ten);
        assert_eq!(rows.ordinals, vec![80, 90, 100, 110, 120]);
        assert!(!rows.contains_key(70));
        assert!(rows.contains_local(80 - 64));
        assert!(!rows.contains_key(130));
        visible.push(true); // Replacement owns a new incarnation key.
        let next = AcceptedRows::prepare(64, 131, &visible, Filter::Ten);
        assert_eq!(next.ordinals, vec![80, 90, 100, 110, 120, 130]);
        assert!(!rows.contains_key(130)); // Previous prepared generation is immutable.
        assert_eq!(
            next.words
                .iter()
                .map(|w| w.count_ones() as usize)
                .sum::<usize>(),
            next.ordinals.len()
        );
    }
    #[test]
    fn planner_segmented_fixture_counts_select_exact_without_forcing_ann() {
        let config = PlannerConfig::default();
        for limit in [100_000usize, 1_000_000] {
            let visible = vec![true; limit];
            let graph_rows = limit - limit * 12 / 100;
            let step = graph_rows.div_ceil(12);
            for filter in [Filter::All, Filter::Ten] {
                let mut accepted = 0;
                let mut paths = 0;
                for start in (0..graph_rows).step_by(step) {
                    let rows = AcceptedRows::prepare(
                        start,
                        (start + step).min(graph_rows),
                        &visible,
                        filter,
                    );
                    accepted += rows.ordinals.len();
                    paths += 1;
                    let expected = if limit == 100_000 || matches!(filter, Filter::Ten) {
                        Route::Exact
                    } else {
                        Route::Plain
                    };
                    assert_eq!(
                        route(
                            &config,
                            96,
                            rows.end - rows.start,
                            rows.ordinals.len(),
                            true,
                            true
                        ),
                        expected
                    );
                }
                accepted += AcceptedRows::prepare(graph_rows, limit, &visible, filter)
                    .ordinals
                    .len();
                assert_eq!(paths, 12);
                assert_eq!(
                    accepted,
                    if matches!(filter, Filter::Ten) {
                        limit / 10
                    } else {
                        limit
                    }
                );
            }
            let candidate = AcceptedRows::prepare(0, limit, &visible, Filter::Ten);
            assert_eq!(
                route(&config, 96, limit, candidate.ordinals.len(), true, true),
                if limit == 100_000 {
                    Route::Exact
                } else {
                    Route::TwoHop
                }
            );
        }
    }
    #[test]
    fn planner_short_hit_fallback_uses_minimum_of_k_and_matching() {
        assert!(short_hits(10, 100, 9));
        assert!(!short_hits(10, 3, 3));
        assert!(short_hits(10, 3, 2));
        assert!(!short_hits(10, 0, 0));
        assert!(!short_hits(0, 100, 0));
    }
}
