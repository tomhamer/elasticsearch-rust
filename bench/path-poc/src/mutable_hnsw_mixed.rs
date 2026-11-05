//! Finite, open-loop insert/query experiment. This is a native CPU/queue gate,
//! not an ES refresh, historical-topology/PIT, or production-memory claim.
use super::*;
use rustic_core::vector::hnsw::{InsertReservation, PreparedMutableInsert};
use std::{
    collections::VecDeque,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{
            AtomicBool,
            Ordering::{Acquire, Release},
        },
        Arc, Condvar, Mutex, OnceLock,
    },
    time::Duration,
};

struct QueueState<T> {
    items: VecDeque<T>,
    closed: bool,
    peak: usize,
}
struct Queue<T> {
    state: Mutex<QueueState<T>>,
    changed: Condvar,
    capacity: usize,
}
impl<T> Queue<T> {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(QueueState {
                items: VecDeque::new(),
                closed: false,
                peak: 0,
            }),
            changed: Condvar::new(),
            capacity,
        }
    }
    fn push(&self, mut item: T, mark: impl FnOnce(&mut T)) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        while !state.closed && state.items.len() == self.capacity {
            state = self.changed.wait(state).unwrap();
        }
        if state.closed {
            return Err("pipeline queue closed before all arrivals were accepted".into());
        }
        mark(&mut item); // Actual admission time, after any backpressure wait.
        state.items.push_back(item);
        state.peak = state.peak.max(state.items.len());
        self.changed.notify_all();
        Ok(())
    }
    fn pop(&self) -> Option<T> {
        let mut state = self.state.lock().unwrap();
        while state.items.is_empty() && !state.closed {
            state = self.changed.wait(state).unwrap();
        }
        let item = state.items.pop_front();
        self.changed.notify_all();
        item
    }
    fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.changed.notify_all();
    }
    fn abort(&self) {
        let mut state = self.state.lock().unwrap();
        state.closed = true;
        state.items.clear();
        self.changed.notify_all();
    }
    fn peak(&self) -> usize {
        self.state.lock().unwrap().peak
    }
}

#[derive(Default)]
struct BudgetState {
    used: usize,
    peak: usize,
    cancelled: bool,
}
struct Budget {
    limit: usize,
    state: Mutex<BudgetState>,
    changed: Condvar,
}
struct Credit {
    budget: Arc<Budget>,
    bytes: usize,
}
impl Budget {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Credit, String> {
        if bytes > self.limit {
            return Err("one writer reservation exceeds pending-byte budget".into());
        }
        let mut state = self.state.lock().unwrap();
        while !state.cancelled && state.used > self.limit - bytes {
            state = self.changed.wait(state).unwrap();
        }
        if state.cancelled {
            return Err("writer pipeline cancelled".into());
        }
        state.used += bytes;
        state.peak = state.peak.max(state.used);
        Ok(Credit {
            budget: self.clone(),
            bytes,
        })
    }
    fn cancel(&self) {
        self.state.lock().unwrap().cancelled = true;
        self.changed.notify_all();
    }
}
impl Drop for Credit {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock().unwrap();
        state.used -= self.bytes;
        self.budget.changed.notify_all();
    }
}

struct Batch {
    generation: usize,
    first: usize,
    end: usize,
    first_due: f64,
    last_due: f64,
    reserve_started: f64,
    enqueued: f64,
    prepare_started: f64,
    prepare_ended: f64,
    prepared_enqueued: f64,
    reservations: Vec<InsertReservation>,
    prepared: Vec<PreparedMutableInsert>,
    prepared_bytes: usize,
    _credit: Credit,
}
struct QueryJob {
    id: usize,
    due: f64,
    enqueued: f64,
}
struct QuerySample {
    id: usize,
    query: usize,
    generation: usize,
    prefix: usize,
    due: f64,
    enqueued: f64,
    started: f64,
    ended: f64,
    hits: Vec<(u64, f32)>,
    ops: u64,
    fallback: u64,
    missing_source: u64,
}

fn wait_until(epoch: Instant, seconds: f64) {
    if let Some(remaining) =
        (epoch + Duration::from_secs_f64(seconds)).checked_duration_since(Instant::now())
    {
        std::thread::sleep(remaining);
    }
}
struct TrialControl<'a> {
    input: &'a Queue<Batch>,
    prepared: &'a Queue<Batch>,
    queries: &'a Queue<QueryJob>,
    budget: &'a Budget,
    failed: AtomicBool,
}
impl TrialControl<'_> {
    fn cancel(&self) {
        self.failed.store(true, Release);
        self.budget.cancel();
        self.input.abort();
        self.prepared.abort();
        self.queries.abort();
    }
}
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "non-string panic".into())
}
fn guarded<T>(
    control: &TrialControl<'_>,
    name: &str,
    body: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let result = catch_unwind(AssertUnwindSafe(body))
        .unwrap_or_else(|payload| Err(format!("{name} panicked: {}", panic_message(payload))));
    if result.is_err() {
        control.cancel();
    }
    result
}
fn start_worker(
    ready: &Barrier,
    start: &Barrier,
    epoch: &OnceLock<Instant>,
    control: &TrialControl<'_>,
    warmup: impl FnOnce(),
) -> Result<Instant, String> {
    // Even a failed warmup reaches both rendezvous points. Catching outside
    // the barrier would strand all other participants before cancellation.
    let warmed = catch_unwind(AssertUnwindSafe(warmup));
    if warmed.is_err() {
        control.cancel();
    }
    ready.wait();
    start.wait();
    if let Err(payload) = warmed {
        return Err(format!("warmup panicked: {}", panic_message(payload)));
    }
    if control.failed.load(Acquire) {
        return Err("trial cancelled during startup".into());
    }
    Ok(*epoch.get().unwrap())
}
fn distribution(mut values: Vec<f64>) -> Value {
    values.sort_unstable_by(f64::total_cmp);
    json!({"count":values.len(),"mean":if values.is_empty(){0.}else{values.iter().sum::<f64>()/values.len() as f64},"p50":percentile(&values,0.5),"p95":percentile(&values,0.95),"p99":percentile(&values,0.99),"max":values.last().copied().unwrap_or(0.)})
}

type Truth = Vec<Vec<Vec<(u64, f64)>>>;
// Insert-only generation g publishes exactly keys [0, ends[g]). Count that
// predicate once, before timing, rather than estimating from active graph rows.
// The existing acquire-loaded generation selects both prefix and this metadata.
fn generation_filter_estimates(ends: &[usize], filter: Filter) -> AnyResult<Vec<VespaFilterEstimate>> {
    let mut previous = 0usize;
    let mut count = 0usize;
    let mut estimates = Vec::with_capacity(ends.len());
    for &end in ends {
        if end < previous {
            return Err("mixed generation ends must not decrease".into());
        }
        count += (previous..end)
            .filter(|&key| filter.accept(key as u64))
            .count();
        estimates.push(VespaFilterEstimate {
            size: u32::try_from(end)?,
            count: u32::try_from(count)?,
        });
        previous = end;
    }
    Ok(estimates)
}

fn generation_truth(
    corpus: &Corpus,
    queries: &[Vec<f32>],
    ends: &[usize],
    k: usize,
    filter: Filter,
) -> Truth {
    // Insert-only top-k is incremental and exact: discarded old candidates can
    // never improve their rank after adding rows. Keep f64 scores throughout.
    let per_query: Vec<Vec<Vec<(u64, f64)>>> = queries
        .par_iter()
        .map(|query| {
            let mut heap = BinaryHeap::new();
            let mut previous = 0;
            let mut generations = Vec::with_capacity(ends.len());
            for &end in ends {
                for key in previous..end {
                    if filter.accept(key as u64) {
                        keep(
                            &mut heap,
                            Ranked {
                                key: key as u64,
                                score: exact_cosine(query, corpus.row(key as u64).unwrap()),
                            },
                            k,
                        );
                    }
                }
                let mut top: Vec<_> = heap.iter().map(|r| (r.key, r.score)).collect();
                top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                generations.push(top);
                previous = end;
            }
            generations
        })
        .collect();
    (0..ends.len())
        .map(|g| per_query.iter().map(|q| q[g].clone()).collect())
        .collect()
}
fn recall(
    corpus: &Corpus,
    query: &[f32],
    hits: &[(u64, f32)],
    exact: &[(u64, f64)],
    prefix: usize,
    filter: Filter,
) -> Result<f64, String> {
    if hits.len() != exact.len() {
        return Err(format!("incomplete hits {} != {}", hits.len(), exact.len()));
    }
    let mut seen = HashSet::new();
    let mut previous = f32::INFINITY;
    for &(key, score) in hits {
        if key >= prefix as u64
            || !filter.accept(key)
            || !seen.insert(key)
            || !score.is_finite()
            || score > previous
        {
            return Err("duplicate, invisible, rejected, nonfinite or unordered hit".into());
        }
        previous = score;
        let want = exact_cosine(query, corpus.row(key).ok_or("missing source row")?);
        if (score as f64 - want).abs() > 2e-5 {
            return Err(format!("score mismatch {score} != {want}"));
        }
    }
    Ok(if exact.is_empty() {
        1.
    } else {
        hits.iter()
            .filter(|h| exact.iter().any(|t| t.0 == h.0))
            .count() as f64
            / exact.len() as f64
    })
}

pub(super) fn run(
    args: &Args,
    engine: &Engine,
    corpus: &mut Corpus,
    queries: &[Vec<f32>],
    out: &mut Output,
) -> AnyResult<()> {
    let seconds = args.usize("mixed-seconds", 30)?;
    let write_rate = args.usize("mixed-write-rate", 500)?;
    let query_rate = args.usize("mixed-query-rate", 200)?;
    let clients = args.usize("mixed-clients", 8)?;
    let k = args.usize("mixed-k", 10)?;
    let ef = args.usize("mixed-ef", 400)?;
    let batch_size = args.usize("mixed-batch", 64)?;
    let queue_bytes = args.usize("mixed-queue-bytes", 33_554_432)?;
    let query_capacity = args.usize("mixed-query-queue", 64)?;
    let filter = Filter::parse(args.text("mixed-filter", "all"))?;
    if [
        seconds,
        write_rate,
        query_rate,
        clients,
        k,
        ef,
        batch_size,
        queue_bytes,
        query_capacity,
    ]
    .contains(&0)
    {
        return Err("mixed duration/rates/clients/k/ef/batch/queue bounds must be positive".into());
    }
    let total_writes = seconds
        .checked_mul(write_rate)
        .ok_or("write count overflow")?;
    let total_queries = seconds
        .checked_mul(query_rate)
        .ok_or("query count overflow")?;
    let initial = corpus.len();
    if initial
        .checked_add(total_writes)
        .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err("mixed source exceeds u32 baseline ordinal range".into());
    }
    // The proposal has <=64 levels. Budget raw row bytes plus conservatively
    // two degree capacities per level, Vec headers and proposal overhead. This
    // bounds estimated pending work, NOT worker scratch/graph/source memory.
    let m = args.usize("m", 16)?;
    let row_budget = m
        .checked_mul(16)
        .and_then(|n| n.checked_add(24))
        .and_then(|n| n.checked_mul(64))
        .and_then(|n| n.checked_add(128))
        .and_then(|n| corpus.dims.checked_mul(4).and_then(|r| n.checked_add(r)))
        .ok_or("queue byte estimate overflow")?;
    let largest_batch_bytes = batch_size
        .min(total_writes)
        .checked_mul(row_budget)
        .ok_or("batch byte estimate overflow")?;
    if queue_bytes < largest_batch_bytes {
        return Err(format!(
            "mixed queue budget must fit one batch ({largest_batch_bytes} estimated bytes)"
        )
        .into());
    }
    let memory_before = memory();
    let mut row_hash = Sha256::new();
    for i in 0..total_writes {
        let key = corpus.append_variant(i.wrapping_mul(13), 17);
        for &value in corpus.row(key).unwrap() {
            row_hash.update(value.to_le_bytes());
        }
    }
    let mut ends = vec![initial];
    for count in (0..total_writes).step_by(batch_size) {
        ends.push(initial + (count + batch_size).min(total_writes));
    }
    let filter_estimates = generation_filter_estimates(&ends, filter)?;
    let trace = json!({"schema":1,"kind":"insert_only_constant_rate_finite","initial_rows":initial,"insert_rows":total_writes,"query_requests":total_queries,"query_count":queries.len(),"seconds":seconds,"write_rows_per_second":write_rate,"query_requests_per_second":query_rate,"batch_rows":batch_size,"generation_ends":ends,"arrival_rule":"row/query i at (i+1)/rate; batch admitted when last row is due; query index=id%query_count","insert_rows_sha256":format!("{:x}",row_hash.finalize()),"query_sha256":hash(&serde_json::to_vec(queries)?)});
    let trace_hash = hash(&serde_json::to_vec(&trace)?);
    let mut config = json!({"trace_sha256":trace_hash,"seconds":seconds,"write_rate":write_rate,"query_rate":query_rate,"clients":clients,"k":k,"ef":ef,"filter":filter.name(),"batch_rows":batch_size,"queue_bytes":queue_bytes,"query_queue_requests":query_capacity,"row_budget_estimate":row_budget,"query_warmups_per_worker":queries.len().min(4),"source_rows_precreated":total_writes,"precreated_vector_payload_bytes":total_writes*corpus.dims*4,"precreated_norm_bytes":total_writes*4,"precreated_vec_header_bytes":total_writes*std::mem::size_of::<Vec<f32>>(),"memory_scope":"source rows/norms are precreated outside timing; all RSS includes them and exact-truth/output storage; NOT a production ingest-memory claim","queue_budget_scope":"conservative pending row+proposal estimate shared across preparation and commit; excludes graph, precreated source, truth, worker scratch, result records","visibility":"release-published completed-batch generation; each query captures acquire generation once; latest graph topology with prefix result visibility; NO historical topology/PIT guarantee","baseline_ingest":"new immutable source rows become exact-tail rows; no segment flush/graph rebuild modeled"});
    config["visited_filter_metadata"] = json!({
        "scope":"precomputed exact accepted counts for each completed-batch document-key prefix; selected with the existing generation read; independent of graph readiness/node-slot count",
        "excluded_from_timing":true,
        "generations":filter_estimates.iter().map(|e| json!({"size":e.size,"count":e.count})).collect::<Vec<_>>()
    });
    let mut provenance: Value = serde_json::from_slice(&fs::read(out.dir.join("run.json"))?)?;
    provenance["mixed"] = config.clone();
    provenance["semantics"] = json!("finite simultaneous insert/query gate; same offered trace and total completions; latest topology plus captured prefix visibility");
    provenance["not_covered"] = json!([
        "quantized scoring",
        "ES refresh or PIT",
        "updates/deletes during mixed load",
        "durable concurrent checkpoint/WAL",
        "production source ingestion allocations",
        "baseline tail flush/compaction"
    ]);
    out.save("run.json", &provenance)?;
    out.save("mixed-trace.json", &trace)?;
    out.emit(json!({"event":"mixed_configuration","config":config,"memory_before_precreated_rows":memory_before,"memory_after_precreated_rows":memory()}))?;
    let truth_start = Instant::now();
    let exact = generation_truth(corpus, queries, &ends, k, filter);
    out.save("mixed-truth.json",&json!({"accumulation":"f64 original stored f32","generation_ends":ends,"queries":queries,"hits":exact}))?;
    out.emit(json!({"event":"mixed_ground_truth","generations":ends.len(),"seconds":truth_start.elapsed().as_secs_f64(),"incremental_insert_only":true}))?;
    let corpus: &Corpus = corpus;
    let input = Queue::<Batch>::new(ends.len());
    let prepared = Queue::<Batch>::new(ends.len());
    let query_queue = Queue::<QueryJob>::new(query_capacity);
    let budget = Arc::new(Budget {
        limit: queue_bytes,
        state: Mutex::new(BudgetState::default()),
        changed: Condvar::new(),
    });
    let generation = AtomicUsize::new(0);
    let writes_done = AtomicUsize::new(0);
    let queries_done = AtomicUsize::new(0);
    let ready = Barrier::new(clients + 5);
    let start = Barrier::new(clients + 5);
    let epoch = OnceLock::new();
    let control = TrialControl {
        input: &input,
        prepared: &prepared,
        queries: &query_queue,
        budget: &budget,
        failed: AtomicBool::new(false),
    };
    let memory_before_timing = memory();
    let result = std::thread::scope(|scope| -> Result<_, String> {
        let write_producer = scope.spawn(|| {
            guarded(&control, "write producer", || {
                let epoch = start_worker(&ready, &start, &epoch, &control, || {})?;
                let result = (|| {
                    for g in 1..ends.len() {
                        let first = ends[g - 1];
                        let end = ends[g];
                        let first_due = (first - initial + 1) as f64 / write_rate as f64;
                        let last_due = (end - initial) as f64 / write_rate as f64;
                        wait_until(epoch, last_due);
                        let reserve_started = epoch.elapsed().as_secs_f64();
                        let credit = budget.reserve((end - first) * row_budget)?;
                        let reservations = if let Engine::Mutable { graph, .. } = engine {
                            (first..end)
                                .map(|key| graph.reserve_insert(key as u64))
                                .collect::<rustic_core::error::Result<Vec<_>>>()
                                .map_err(|e| e.to_string())?
                        } else {
                            Vec::new()
                        };
                        input.push(
                            Batch {
                                generation: g,
                                first,
                                end,
                                first_due,
                                last_due,
                                reserve_started,
                                enqueued: 0.,
                                prepare_started: 0.,
                                prepare_ended: 0.,
                                prepared_enqueued: 0.,
                                reservations,
                                prepared: Vec::new(),
                                prepared_bytes: 0,
                                _credit: credit,
                            },
                            |b| b.enqueued = epoch.elapsed().as_secs_f64(),
                        )?;
                    }
                    Ok(())
                })();
                input.close();
                result
            })
        });
        let preparer = scope.spawn(|| {
            guarded(&control, "preparer", || {
                let epoch = start_worker(&ready, &start, &epoch, &control, || {})?;
                let result = (|| {
                    while let Some(mut batch) = input.pop() {
                        batch.prepare_started = epoch.elapsed().as_secs_f64();
                        if let Engine::Mutable { graph, owner, .. } = engine {
                            if let Some(owner) = owner {
                                for reservation in &batch.reservations {
                                    owner.install(reservation, corpus.row(reservation.key()).ok_or("missing mixed insertion row")?)
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                            let source = MutableSourceLease::new(owner.as_deref(), corpus)?;
                            batch.prepared = std::mem::take(&mut batch.reservations)
                                .into_par_iter()
                                .map(|reservation| graph.prepare(reservation, &source))
                                .collect::<rustic_core::error::Result<Vec<_>>>()
                                .map_err(|e| e.to_string())?;
                        }
                        batch.prepared_bytes =
                            batch.prepared.iter().map(|p| p.estimated_bytes()).sum();
                        if batch.prepared_bytes + (batch.end - batch.first) * corpus.dims * 4
                            > batch._credit.bytes
                        {
                            return Err(
                                "proposal bytes exceeded conservative pending-work reservation"
                                    .into(),
                            );
                        }
                        batch.prepare_ended = epoch.elapsed().as_secs_f64();
                        prepared.push(batch, |b| {
                            b.prepared_enqueued = epoch.elapsed().as_secs_f64()
                        })?;
                    }
                    Ok(())
                })();
                if result.is_err() {
                    budget.cancel();
                    input.abort();
                    prepared.abort();
                }
                prepared.close();
                result
            })
        });
        let committer = scope.spawn(|| guarded(&control, "committer", || {
            let epoch = start_worker(&ready,&start,&epoch,&control, || {})?;
            let result = (|| {
                let mut samples = Vec::with_capacity(ends.len()-1);
                let mut expected = 1;
                while let Some(mut batch) = prepared.pop() {
                    if batch.generation != expected { return Err("out-of-order publication".into()); }
                    let commit_started = epoch.elapsed().as_secs_f64();
                    if let Engine::Mutable { graph, owner, .. } = engine {
                        {
                            let _commit = graph.commit_batch().map_err(|e| e.to_string())?;
                            let source = MutableSourceLease::new(owner.as_deref(), corpus)?;
                            for proposal in batch.prepared.drain(..) { graph.complete(proposal,&source).map_err(|e|e.to_string())?; }
                        }
                        if let Some(owner) = owner { owner.commit().map_err(|e| e.to_string())?; }
                    }
                    generation.store(batch.generation,Release);
                    writes_done.fetch_add(batch.end-batch.first,Relaxed);
                    let published = epoch.elapsed().as_secs_f64();
                    samples.push(json!({"generation":batch.generation,"first_key":batch.first,"end_key":batch.end,"rows":batch.end-batch.first,"first_row_due_seconds":batch.first_due,"last_row_due_seconds":batch.last_due,"reserve_started_seconds":batch.reserve_started,"enqueued_seconds":batch.enqueued,"prepare_started_seconds":batch.prepare_started,"prepare_ended_seconds":batch.prepare_ended,"prepared_enqueued_seconds":batch.prepared_enqueued,"commit_started_seconds":commit_started,"published_seconds":published,"estimated_pending_bytes":batch._credit.bytes,"prepared_proposal_bytes":batch.prepared_bytes}));
                    expected += 1;
                }
                Ok(samples)
            })();
            if result.is_err() { budget.cancel(); input.abort(); prepared.abort(); }
            result
        }));
        let query_producer = scope.spawn(|| {
            guarded(&control, "query producer", || {
                let epoch = start_worker(&ready, &start, &epoch, &control, || {})?;
                let result = (|| {
                    for id in 0..total_queries {
                        let due = (id + 1) as f64 / query_rate as f64;
                        wait_until(epoch, due);
                        query_queue.push(
                            QueryJob {
                                id,
                                due,
                                enqueued: 0.,
                            },
                            |q| q.enqueued = epoch.elapsed().as_secs_f64(),
                        )?;
                    }
                    Ok(())
                })();
                query_queue.close();
                result
            })
        });
        let workers: Vec<_> = (0..clients)
            .map(|_| {
                scope.spawn(|| {
                    guarded(&control, "query worker", || {
                        let epoch = start_worker(&ready, &start, &epoch, &control, || {
                            for query in queries.iter().take(4) {
                                std::hint::black_box(engine.search_accepted(
                                    corpus,
                                    initial,
                                    query,
                                    k,
                                    ef,
                                    filter,
                                    |_| true,
                                    Some(filter_estimates[0]),
                                ));
                            }
                        })?;
                        let mut samples = Vec::new();
                        while let Some(job) = query_queue.pop() {
                            let started = epoch.elapsed().as_secs_f64();
                            let g = generation.load(Acquire);
                            let prefix = ends[g];
                            let qi = job.id % queries.len();
                            let (hits, ops, fallback, missing_source) = engine.search_accepted(
                                corpus,
                                prefix,
                                &queries[qi],
                                k,
                                ef,
                                filter,
                                |_| true,
                                Some(filter_estimates[g]),
                            );
                            let ended = epoch.elapsed().as_secs_f64();
                            queries_done.fetch_add(1, Relaxed);
                            samples.push(QuerySample {
                                id: job.id,
                                query: qi,
                                generation: g,
                                prefix,
                                due: job.due,
                                enqueued: job.enqueued,
                                started,
                                ended,
                                hits,
                                ops,
                                fallback,
                                missing_source,
                            });
                        }
                        Ok(samples)
                    })
                })
            })
            .collect();
        ready.wait();
        epoch.set(Instant::now()).unwrap();
        let cpu_start = cpu_seconds();
        start.wait();
        if !control.failed.load(Acquire) {
            wait_until(*epoch.get().unwrap(), seconds as f64);
        }
        let offer_observed = epoch.get().unwrap().elapsed().as_secs_f64();
        let offer_cpu = cpu_seconds() - cpu_start;
        let offer_writes = writes_done.load(Relaxed);
        let offer_queries = queries_done.load(Relaxed);
        let offer_generation = generation.load(Acquire);
        // All finite arrivals drain. No timeout silently drops scheduled work.
        let wp = write_producer
            .join()
            .map_err(|_| "write producer panicked")?;
        let pp = preparer.join().map_err(|_| "preparer panicked")?;
        let commits = committer.join().map_err(|_| "committer panicked")?;
        let qp = query_producer
            .join()
            .map_err(|_| "query producer panicked")?;
        let mut samples = Vec::with_capacity(total_queries);
        for worker in workers {
            samples.extend(worker.join().map_err(|_| "query worker panicked")??);
        }
        let wall = epoch.get().unwrap().elapsed().as_secs_f64();
        let cpu = cpu_seconds() - cpu_start;
        wp?;
        pp?;
        qp?;
        Ok((
            samples,
            commits?,
            wall,
            cpu,
            offer_observed,
            offer_cpu,
            offer_writes,
            offer_queries,
            offer_generation,
        ))
    });
    let (
        mut samples,
        commits,
        wall,
        cpu,
        offer_observed,
        offer_cpu,
        offer_writes,
        offer_queries,
        offer_generation,
    ) = match result {
        Ok(result) => result,
        Err(error) => {
            out.emit(json!({"event":"mixed_failure","trace_sha256":trace_hash,"error":error,"offered_writes":total_writes,"completed_writes":writes_done.load(Relaxed),"offered_queries":total_queries,"completed_queries":queries_done.load(Relaxed),"last_published_generation":generation.load(Acquire),"all_arrivals_completed":false,"performance_accepted":false}))?;
            return Err(format!("mixed trial failed: {error}").into());
        }
    };
    samples.sort_unstable_by_key(|s| s.id);
    if samples.len() != total_queries
        || commits.len() != ends.len() - 1
        || writes_done.load(Relaxed) != total_writes
        || samples.iter().enumerate().any(|(i, s)| i != s.id)
        || generation.load(Acquire) != ends.len() - 1
    {
        return Err("mixed workload did not complete every planned arrival exactly once".into());
    }
    if budget.state.lock().unwrap().used != 0 {
        return Err("writer byte credits leaked".into());
    }
    out.save("mixed-writes.json", &json!(commits))?;
    let memory_after_timing = memory();
    let mut recalls = vec![Vec::new(); ends.len()];
    let mut records = Vec::with_capacity(samples.len());
    for s in &samples {
        if s.missing_source != 0 {
            return Err(format!(
                "missing source lookups in mixed query: {}",
                s.missing_source
            )
            .into());
        }
        if s.fallback != 0 {
            return Err(format!(
                "unexpected entry fallback in insert-only trial: {}",
                s.fallback
            )
            .into());
        }
        let r = recall(
            corpus,
            &queries[s.query],
            &s.hits,
            &exact[s.generation][s.query],
            s.prefix,
            filter,
        )?;
        recalls[s.generation].push(r);
        records.push(json!({"id":s.id,"query":s.query,"generation":s.generation,"visible_rows":s.prefix,"due_seconds":s.due,"enqueued_seconds":s.enqueued,"started_seconds":s.started,"ended_seconds":s.ended,"recall":r,"hits":s.hits,"vector_ops":s.ops,"entry_fallback_scanned":s.fallback,"missing_source_lookups":s.missing_source}));
    }
    out.save("mixed-queries.json", &json!(records))?;
    let audit_start = Instant::now();
    let mut audits = Vec::with_capacity(ends.len());
    for (g, &prefix) in ends.iter().enumerate() {
        let mut per_query = Vec::with_capacity(queries.len());
        for (qi, query) in queries.iter().enumerate() {
            let (hits, _, fallback, missing_source) = engine.search_accepted(
                corpus, prefix, query, k, ef, filter, |_| true,
                Some(filter_estimates[g]),
            );
            if missing_source != 0 {
                return Err("missing source lookups in post-drain audit".into());
            }
            if fallback != 0 {
                return Err("unexpected entry fallback in post-drain audit".into());
            }
            let r = recall(corpus, query, &hits, &exact[g][qi], prefix, filter)?;
            per_query.push(json!({"query":qi,"recall":r,"hits":hits}));
        }
        let mean = per_query
            .iter()
            .map(|q| q["recall"].as_f64().unwrap())
            .sum::<f64>()
            / queries.len() as f64;
        audits.push(json!({"generation":g,"visible_rows":prefix,"audit_mean_recall":mean,"audit_min_recall":per_query.iter().map(|q|q["recall"].as_f64().unwrap()).fold(1.,f64::min),"timed_queries":recalls[g].len(),"timed_mean_recall":if recalls[g].is_empty(){None}else{Some(recalls[g].iter().sum::<f64>()/recalls[g].len() as f64)},"audit_queries":per_query}));
    }
    out.save("mixed-generation-audit.json",&json!({"scope":"UNTIMED post-drain final topology plus historical visibility prefixes; not historical graph snapshot/PIT semantics","seconds":audit_start.elapsed().as_secs_f64(),"generations":audits}))?;
    let all_recalls: Vec<_> = recalls.iter().flatten().copied().collect();
    let pending_peak = budget.state.lock().unwrap().peak;
    let summary = json!({"event":"mixed_summary","trace_sha256":trace_hash,"all_arrivals_completed":true,"errors":0,"dropped_arrivals":0,"offered_writes":total_writes,"completed_writes":writes_done.load(Relaxed),"offered_queries":total_queries,"completed_queries":samples.len(),"published_generations":commits.len(),"planned_offer_seconds":seconds,"offer_observed_seconds":offer_observed,"offer_window":{"completed_writes":offer_writes,"completed_queries":offer_queries,"published_generation":offer_generation,"process_cpu_seconds":offer_cpu},"total_seconds_including_drain":wall,"drain_seconds":(wall-seconds as f64).max(0.),"process_cpu_seconds":cpu,"process_cpu_ms_per_query_including_writes":cpu*1000./total_queries as f64,"process_cpu_ms_per_write_including_queries":cpu*1000./total_writes as f64,"completed_queries_per_second_including_drain":total_queries as f64/wall,"completed_writes_per_second_including_drain":total_writes as f64/wall,"query_service_ms":distribution(samples.iter().map(|s|(s.ended-s.started)*1000.).collect()),"query_queue_ms":distribution(samples.iter().map(|s|(s.started-s.enqueued)*1000.).collect()),"query_admission_lateness_ms":distribution(samples.iter().map(|s|(s.enqueued-s.due)*1000.).collect()),"query_end_to_end_ms":distribution(samples.iter().map(|s|(s.ended-s.due)*1000.).collect()),"mean_query_visible_rows":samples.iter().map(|s|s.prefix as f64).sum::<f64>()/samples.len() as f64,"write_last_row_publish_lag_ms":distribution(commits.iter().map(|b|(b["published_seconds"].as_f64().unwrap()-b["last_row_due_seconds"].as_f64().unwrap())*1000.).collect()),"timed_mean_recall":all_recalls.iter().sum::<f64>()/all_recalls.len() as f64,"timed_min_recall":all_recalls.iter().copied().fold(1.,f64::min),"audit_generation_recalls":audits.iter().map(|g|g["audit_mean_recall"].as_f64().unwrap()).collect::<Vec<_>>(),"audit_min_generation_mean_recall":audits.iter().map(|g|g["audit_mean_recall"].as_f64().unwrap()).fold(1.,f64::min),"vector_ops":samples.iter().map(|s|s.ops).sum::<u64>(),"entry_fallback_scanned":samples.iter().map(|s|s.fallback).sum::<u64>(),"missing_source_lookups":samples.iter().map(|s|s.missing_source).sum::<u64>(),"pending_estimated_bytes_peak":pending_peak,"pending_estimated_bytes_limit":queue_bytes,"queue_depth_peaks":{"input_batches":input.peak(),"prepared_batches":prepared.peak(),"query_requests":query_queue.peak()},"graph_after":engine.stats(corpus),"memory_before_timing":memory_before_timing,"memory_after_timing":memory_after_timing,"limitations":["mixed process CPU includes both workloads; not per-operation attribution","precreated rows/truth/results included in RSS; not production memory","no baseline flush/rebuild/compaction","latest-topology prefix visibility only; no historical PIT or updates/deletes","fixed rate below saturation can make total wall ratios uninformative; inspect queue lag, CPU and drain"]});
    out.save("mixed-summary.json", &summary)?;
    out.emit(summary)
}

fn visibility_coverage(baseline: &[Value], candidate: &[Value]) -> Result<usize, String> {
    if baseline.is_empty() || baseline.len() != candidate.len() {
        return Err("mixed timed query coverage differs".into());
    }
    let mut deficits = 0;
    for (id, (b, c)) in baseline.iter().zip(candidate).enumerate() {
        if b["id"].as_u64() != Some(id as u64)
            || c["id"].as_u64() != Some(id as u64)
            || b["query"] != c["query"]
            || b["due_seconds"] != c["due_seconds"]
        {
            return Err("mixed query traces differ by arrival id/query/timestamp".into());
        }
        let baseline_rows = b["visible_rows"]
            .as_u64()
            .ok_or("missing baseline query visibility")?;
        let candidate_rows = c["visible_rows"]
            .as_u64()
            .ok_or("missing candidate query visibility")?;
        deficits += usize::from(candidate_rows < baseline_rows);
    }
    Ok(deficits)
}
fn matched_timed_recall(baseline: f64, candidate: f64, floor: f64) -> bool {
    baseline >= floor && candidate + 1e-12 >= floor.max(baseline)
}

pub(super) fn compare(runs: &[(Value, Vec<Value>)], floor: f64, out: &mut Output) -> AnyResult<()> {
    if runs.iter().any(|r| r.0["mixed"].is_null()) {
        return Err("cannot compare a mixed workload to a quiescent workload".into());
    }
    let candidate = runs
        .iter()
        .find(|r| r.0["mode"] == "mutable")
        .ok_or("mutable run missing")?;
    let summary = |events: &[Value]| -> AnyResult<Value> {
        let rows: Vec<_> = events
            .iter()
            .filter(|e| e["event"] == "mixed_summary")
            .collect();
        if rows.len() != 1 {
            return Err("each mixed run needs exactly one completed summary".into());
        }
        Ok(rows[0].clone())
    };
    let c = summary(&candidate.1)?;
    let load_queries = |run: &Value| -> AnyResult<Vec<Value>> {
        let root = run["comparison_input_dir"]
            .as_str()
            .ok_or("mixed comparison input directory missing")?;
        Ok(serde_json::from_slice(&fs::read(
            Path::new(root).join("mixed-queries.json"),
        )?)?)
    };
    let candidate_queries = load_queries(&candidate.0)?;
    let mut comparisons = Vec::new();
    for (run, events) in runs.iter().filter(|r| r.0["mode"] != "mutable") {
        for key in [
            "planner",
            "fixture_sha256",
            "query_sha256",
            "limit",
            "dims",
            "seed",
            "threads",
            "insert_batch",
            "queries",
            "cpu_affinity",
            "m",
            "ef_construction",
        ] {
            if run[key] != candidate.0[key] {
                return Err(
                    format!("incompatible mixed source/initial graph setting: {key}").into(),
                );
            }
        }
        // ef deliberately may differ: accept speed data only at equal-or-better
        // per-generation quality on the same f64 truth and visible prefixes.
        for key in [
            "trace_sha256",
            "seconds",
            "write_rate",
            "query_rate",
            "clients",
            "k",
            "filter",
            "batch_rows",
            "queue_bytes",
            "query_queue_requests",
            "row_budget_estimate",
        ] {
            if run["mixed"][key] != candidate.0["mixed"][key] {
                return Err(format!("incompatible mixed trace/config: {key}").into());
            }
        }
        let b = summary(events)?;
        let finished = [&b, &c].iter().all(|s| {
            s["all_arrivals_completed"] == true
                && s["errors"] == 0
                && s["dropped_arrivals"] == 0
                && s["offered_writes"] == s["completed_writes"]
                && s["offered_queries"] == s["completed_queries"]
        });
        if !finished
            || b["offered_writes"] != c["offered_writes"]
            || b["offered_queries"] != c["offered_queries"]
        {
            return Err("mixed comparison requires identical finite workloads fully completed without errors/drops".into());
        }
        let br = b["audit_generation_recalls"]
            .as_array()
            .ok_or("baseline generation audit missing")?;
        let cr = c["audit_generation_recalls"]
            .as_array()
            .ok_or("candidate generation audit missing")?;
        if br.len() != cr.len() || br.is_empty() {
            return Err("mixed generation audit coverage differs".into());
        }
        let quality = br.iter().zip(cr).all(|(b, c)| {
            b.as_f64().unwrap_or(0.) >= floor
                && c.as_f64().unwrap_or(0.) + 1e-12 >= floor.max(b.as_f64().unwrap_or(0.))
        }) && matched_timed_recall(
            b["timed_mean_recall"].as_f64().unwrap_or(0.),
            c["timed_mean_recall"].as_f64().unwrap_or(0.),
            floor,
        );
        let baseline_queries = load_queries(run)?;
        if baseline_queries.len() as u64 != b["completed_queries"].as_u64().unwrap_or(0)
            || candidate_queries.len() as u64 != c["completed_queries"].as_u64().unwrap_or(0)
        {
            return Err("mixed timed query artifact coverage disagrees with summary".into());
        }
        let visibility_deficits = visibility_coverage(&baseline_queries, &candidate_queries)?;
        // CPU/elapsed can also hide cheaper searches on a lagging prefix.
        // Average prefix size alone cannot establish comparable temporal work.
        let write_progress = c["offer_window"]["completed_writes"].as_u64().unwrap_or(0)
            >= b["offer_window"]["completed_writes"].as_u64().unwrap_or(0)
            && visibility_deficits == 0;
        comparisons.push(json!({"baseline":run["mode"],"fanout":run["fanout"],"tail_scheduling":run["tail_scheduling"],"trace_sha256":b["trace_sha256"],"same_finite_completions":true,"matched_quality":quality,"recall_floor":floor,"baseline_ef":run["mixed"]["ef"],"mutable_ef":candidate.0["mixed"]["ef"],"mutable_write_progress_at_least_baseline":write_progress,"accepted_mixed_speedup_comparison":quality&&write_progress,"query_visibility_deficits":visibility_deficits,"accepted_query_latency_speedup":quality&&write_progress,"finite_work_elapsed_speedup":if quality&&write_progress {Some(b["total_seconds_including_drain"].as_f64().unwrap()/c["total_seconds_including_drain"].as_f64().unwrap())}else{None},"finite_work_cpu_reduction_factor":if quality&&write_progress {Some(b["process_cpu_seconds"].as_f64().unwrap()/c["process_cpu_seconds"].as_f64().unwrap())}else{None},"query_p99_end_to_end_speedup":if quality&&write_progress {Some(b["query_end_to_end_ms"]["p99"].as_f64().unwrap()/c["query_end_to_end_ms"]["p99"].as_f64().unwrap())}else{None},"baseline_drain_seconds":b["drain_seconds"],"mutable_drain_seconds":c["drain_seconds"],"baseline_summary":b,"mutable_summary":c,"interpretation":"All speedup acceptance requires full equal finite completions, matched TIMED recall and post-drain audit recall, and no lower visible prefix for ANY corresponding query arrival; rejected ratios remain null. Post-drain audits use final topology, not PIT"}));
    }
    out.save("comparison.json",&json!({"kind":"finite_insert_only_mixed","recall_floor":floor,"comparisons":comparisons,"limitations":"not a saturation claim at one offered rate; sweep rates and inspect queue/admission lag and drain; no production-memory, historical-topology, durability or compaction claim"}))?;
    out.emit(json!({"event":"mixed_comparison","groups":comparisons.len(),"path":out.dir.join("comparison.json")}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_generation_filter_counts_match_each_captured_prefix() {
        let ends = [0, 1, 5, 9, 10, 11, 15, 20, 31];
        for filter in [Filter::All, Filter::Ten, Filter::Half] {
            let estimates = generation_filter_estimates(&ends, filter).unwrap();
            for (estimate, &end) in estimates.iter().zip(&ends) {
                assert_eq!(estimate.size as usize, end);
                assert_eq!(
                    estimate.count as usize,
                    (0..end).filter(|&key| filter.accept(key as u64)).count()
                );
            }
            assert_eq!(
                estimates[1].size, 1,
                "future prepared source rows cannot change an older generation"
            );
        }
        assert!(generation_filter_estimates(&[10, 9], Filter::Ten).is_err());
    }

    #[test]
    fn comparison_rejects_timed_recall_or_hidden_visibility_deficit() {
        assert!(!matched_timed_recall(0.99, 0.96, 0.95));
        assert!(matched_timed_recall(0.99, 0.995, 0.95));
        let records = |prefixes: &[usize]| {
            prefixes.iter().enumerate().map(|(id,prefix)|json!({"id":id,"query":id,"due_seconds":id as f64,"visible_rows":prefix})).collect::<Vec<_>>()
        };
        assert_eq!(
            visibility_coverage(&records(&[100, 200]), &records(&[150, 150])).unwrap(),
            1
        );
        assert_eq!(
            visibility_coverage(&records(&[100, 200]), &records(&[100, 201])).unwrap(),
            0
        );
    }
    #[test]
    fn warmup_and_worker_panics_cancel_without_stranding_startup_or_queues() {
        let input = Queue::new(1);
        let prepared = Queue::new(1);
        let queries = Queue::new(1);
        let budget = Arc::new(Budget {
            limit: 32,
            state: Mutex::new(BudgetState::default()),
            changed: Condvar::new(),
        });
        let control = TrialControl {
            input: &input,
            prepared: &prepared,
            queries: &queries,
            budget: &budget,
            failed: AtomicBool::new(false),
        };
        let ready = Barrier::new(3);
        let start = Barrier::new(3);
        let epoch = OnceLock::new();
        std::thread::scope(|scope| {
            let a = scope.spawn(|| {
                guarded(&control, "warmup test", || {
                    start_worker(&ready, &start, &epoch, &control, || {
                        panic!("expected warmup failure")
                    })
                })
            });
            let b = scope.spawn(|| {
                guarded(&control, "other worker", || {
                    start_worker(&ready, &start, &epoch, &control, || {})
                })
            });
            ready.wait();
            epoch.set(Instant::now()).unwrap();
            start.wait();
            assert!(a.join().unwrap().is_err());
            assert!(b.join().unwrap().is_err());
        });
        assert!(control.failed.load(Acquire));
        assert!(budget.reserve(1).is_err());
        assert!(queries.pop().is_none());
        let result: Result<(), String> = guarded(&control, "commit test", || {
            panic!("expected commit failure")
        });
        assert!(result.unwrap_err().contains("commit test panicked"));
    }
    #[test]
    fn queues_close_and_drain_without_dropping_arrivals() {
        let queue = Queue::new(1);
        std::thread::scope(|scope| {
            let producer = scope.spawn(|| {
                for i in 0..100 {
                    queue.push(i, |_| {}).unwrap();
                }
                queue.close();
            });
            let mut received = Vec::new();
            while let Some(i) = queue.pop() {
                received.push(i);
            }
            producer.join().unwrap();
            assert_eq!(received, (0..100).collect::<Vec<_>>());
        });
        assert_eq!(queue.peak(), 1);
        assert!(queue.push(101, |_| {}).is_err());
    }
    #[test]
    fn byte_credit_covers_both_pipeline_stages_until_dropped() {
        let budget = Arc::new(Budget {
            limit: 32,
            state: Mutex::new(BudgetState::default()),
            changed: Condvar::new(),
        });
        let input = Queue::new(1);
        let prepared = Queue::new(1);
        input.push(budget.reserve(24).unwrap(), |_| {}).unwrap();
        prepared.push(input.pop().unwrap(), |_| {}).unwrap();
        assert_eq!(budget.state.lock().unwrap().used, 24);
        drop(prepared.pop());
        assert_eq!(budget.state.lock().unwrap().used, 0);
        assert_eq!(budget.state.lock().unwrap().peak, 24);
        budget.cancel();
        assert!(budget.reserve(1).is_err());
    }
    #[test]
    fn incremental_insert_truth_matches_full_recomputation() {
        let file = tempfile::tempfile().unwrap();
        let values = [1.0f32, 0., 0., 1., -1., 0.];
        (&file)
            .write_all(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let data = unsafe { Mmap::map(&file).unwrap() };
        let mut corpus = Corpus {
            data,
            dims: 2,
            base: 3,
            extras: Vec::new(),
            norms: vec![1.; 3],
        };
        corpus.append_variant(0, 17);
        corpus.append_variant(1, 17);
        let queries = vec![vec![1., 0.], vec![0., 1.]];
        let ends = [3, 4, 5];
        for filter in [Filter::All, Filter::Half, Filter::Ten] {
            let incremental = generation_truth(&corpus, &queries, &ends, 2, filter);
            for (g, &end) in ends.iter().enumerate() {
                let visible: Vec<_> = (0..corpus.len()).map(|key| key < end).collect();
                assert_eq!(
                    incremental[g],
                    truth(&corpus, &visible, &queries, 2, filter)
                );
            }
        }
    }
}
