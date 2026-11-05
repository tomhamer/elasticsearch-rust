//! Search and aggregation results with replicas, on three node processes:
//! `split` has five shards with two replicas (every node holds a copy of
//! every shard), `ref` one shard and no replica, both the same seeded
//! documents (keyword, numeric, date, text, nested and multi-valued fields,
//! some missing). A battery of aggregations and searches runs on `split`
//! under each copy selection that both ES and this engine honour today: no
//! preference (copies rotate per request, so paged items mix copies) from
//! two coordinators, and every shard from one node's own copies
//! (`preference=_shards:0,1,2,3,4|_only_local` sent to that node). Every
//! selection must
//! - agree with the others: exactly, except the t-digest percentiles (ES's
//!   `TDigestState` depends on the order values arrive in, which differs
//!   between copies with different segments) and the HLL cardinalities,
//!   which keep cross_shard_correctness's tolerances;
//! - equal `ref` where ES is exact, within those tolerances where it is
//!   not. Shard-size-bounded terms are checked against ES's documented
//!   bound (`AbstractInternalTerms`: a returned count is at most the true
//!   one, at least it minus `doc_count_error_upper_bound`), and that bound
//!   itself is not compared with `ref` (one shard reports it differently).
//!
//! Each copy is also checked on its own (`_shards:k|_only_local` on each
//! node, the serving node read back from an explained hit's `_node`): the
//! same ids on every copy of a shard, the shards together exactly the
//! modelled docs.
//!
//! The battery runs again after index / update / delete batches, after the
//! node holding the most primaries is killed while a bulk is in flight
//! (every acked doc is there, once), and after that node restarts.
//!
//! `es_preferences_*` assert ES's `OperationRouting` / `Preference`
//! semantics (`_local`, `_only_local`, `_prefer_nodes`, `_only_nodes`,
//! session strings consistent across coordinators, unknown names 400).
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

const STEADY: usize = 3;
const FAILOVER: usize = 3;
const BREAKERS: usize = 3;
const DOCS: usize = 5000;
const BASE_MS: i64 = 1_700_000_000_000;
const DAY_MS: i64 = 86_400_000;
/// Float sums and moments are added in a different order per copy and per
/// shard split (ES's compensated sums keep them this close).
const FLOAT_TOL: f64 = 1e-9;
const WORDS: [&str; 16] = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"];

/// splitmix64: the same documents on every run and platform.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }

    /// 0..n, small values more often (the least of two draws).
    fn skewed(&mut self, n: u64) -> u64 {
        let a = self.below(n);
        a.min(self.below(n))
    }
}

fn uid(i: usize) -> String {
    format!("d{i:06}")
}

/// Document `i`: `cat` (skewed, 5% missing), `tag` (0-3 of 60, plus `hot`
/// on most `alpha` bodies), `n` (5% missing), `price` (double), `d` (3%
/// missing), `body` (text), `flag`, nested `items`.
fn doc(rng: &mut Rng, i: usize) -> Value {
    let len = 5 + rng.below(10);
    let words: Vec<&str> = (0..len).map(|_| WORDS[rng.skewed(WORDS.len() as u64) as usize]).collect();
    let n_tags = rng.below(4);
    let mut tags: Vec<String> = (0..n_tags).map(|_| format!("t{:02}", rng.below(60))).collect();
    if words.contains(&"alpha") && rng.chance(60) {
        tags.push("hot".into());
    }
    tags.sort();
    tags.dedup();
    let mut d = json!({"uid": uid(i), "price": rng.below(100_000) as f64 / 100.0, "body": words.join(" "), "flag": rng.chance(50)});
    if !rng.chance(5) {
        d["cat"] = json!(format!("c{:02}", rng.skewed(12)));
    }
    if !tags.is_empty() {
        d["tag"] = json!(tags);
    }
    if !rng.chance(5) {
        d["n"] = json!(rng.below(10_000));
    }
    if !rng.chance(3) {
        d["d"] = json!(BASE_MS + rng.below(90 * DAY_MS as u64) as i64);
    }
    let n_items = rng.below(4);
    let items: Vec<Value> = (0..n_items).map(|_| json!({"sku": format!("s{:02}", rng.below(40)), "qty": 1 + rng.below(20)})).collect();
    if !items.is_empty() {
        d["items"] = json!(items);
    }
    d
}

fn mappings() -> Value {
    json!({"properties": {"uid": {"type": "keyword"}, "cat": {"type": "keyword"}, "tag": {"type": "keyword"}, "n": {"type": "long"},
        "price": {"type": "double"}, "d": {"type": "date"}, "body": {"type": "text"}, "flag": {"type": "boolean"},
        "items": {"type": "nested", "properties": {"sku": {"type": "keyword"}, "qty": {"type": "integer"}}}}})
}

async fn create_split(pl: &Procs, via: usize) {
    let (st, v) = pl.call(via, Method::PUT, "/split", Some(json!({"settings": {"number_of_shards": 5, "number_of_replicas": 2}, "mappings": mappings()}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(via, "split", "green", 90).await;
}

/// `ref` lives on `keeper` (the node no test kills).
async fn create_ref(pl: &Procs, keeper: usize) {
    let (st, v) = pl.call(keeper, Method::PUT, "/ref", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0,
        "index.routing.allocation.require._name": format!("n{keeper}")}, "mappings": mappings()}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(keeper, "ref", "green", 90).await;
}

async fn bulk(pl: &Procs, via: usize, index: &str, lines: String) {
    let (st, v) = pl.ndjson(via, &format!("/{index}/_bulk"), lines).await;
    assert_eq!(st, 200, "{index}: {v}");
    assert_eq!(v["errors"], false, "{index}: {v}");
}

async fn refresh(pl: &Procs, via: usize) {
    let (st, v) = pl.call(via, Method::POST, "/split,ref/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
}

/// Index docs `ids` into both indices (and `model`), then refresh.
async fn load(pl: &Procs, via: usize, rng: &mut Rng, model: &mut BTreeMap<String, Value>, ids: std::ops::Range<usize>) {
    let all: Vec<usize> = ids.collect();
    for chunk in all.chunks(1000) {
        let mut lines = String::new();
        for &i in chunk {
            let d = doc(rng, i);
            lines.push_str(&json!({"index": {"_id": uid(i)}}).to_string());
            lines.push('\n');
            lines.push_str(&d.to_string());
            lines.push('\n');
            model.insert(uid(i), d);
        }
        bulk(pl, via, "split", lines.clone()).await;
        bulk(pl, via, "ref", lines).await;
    }
    refresh(pl, via).await;
}

/// One bulk body of new docs, partial updates (some unset `d`) and deletes,
/// each on a different id; applied to `model`.
fn write_batch(rng: &mut Rng, model: &mut BTreeMap<String, Value>, next: &mut usize) -> String {
    let existing: Vec<String> = model.keys().cloned().collect();
    let mut touched = BTreeSet::new();
    let mut pick = |rng: &mut Rng| loop {
        let id = existing[rng.below(existing.len() as u64) as usize].clone();
        if touched.insert(id.clone()) {
            return id;
        }
    };
    let mut lines = String::new();
    let mut push = |action: Value, source: Option<Value>| {
        lines.push_str(&action.to_string());
        lines.push('\n');
        if let Some(s) = source {
            lines.push_str(&s.to_string());
            lines.push('\n');
        }
    };
    for _ in 0..200 {
        let d = doc(rng, *next);
        push(json!({"index": {"_id": uid(*next)}}), Some(d.clone()));
        model.insert(uid(*next), d);
        *next += 1;
    }
    for _ in 0..200 {
        let id = pick(rng);
        let mut partial = json!({"n": rng.below(10_000), "cat": format!("c{:02}", rng.skewed(12)), "tag": [format!("t{:02}", rng.below(60))]});
        if rng.chance(10) {
            partial["d"] = Value::Null;
        }
        push(json!({"update": {"_id": id}}), Some(json!({"doc": partial})));
        for (k, v) in partial.as_object().unwrap() {
            model.get_mut(&id).unwrap()[k] = v.clone();
        }
    }
    for _ in 0..150 {
        let id = pick(rng);
        push(json!({"delete": {"_id": id}}), None);
        model.remove(&id);
    }
    lines
}

// ------------------------------------------------------------- the battery

#[derive(Clone, Copy, PartialEq, Debug)]
enum Check {
    /// Equal; floats within [`FLOAT_TOL`].
    Exact,
    /// Every number within this relative tolerance.
    Tol(f64),
    /// `top` (small `shard_size`) within ES's error bound of `all`.
    TermsBound,
    /// `significant_terms` `sig` next to the foreground `fg` terms: exact,
    /// or after deletes (background frequencies count deleted docs until
    /// merged, which differs per copy) counts only.
    Sig,
}

enum Kind {
    /// One search; the normalized response.
    Body(Value),
    /// A composite aggregation paged with `after` until no page is left.
    Composite { query: Value, sources: Value, size: u64 },
    /// `search_after` pages until an empty one.
    SearchAfter(Value),
    /// The same search at each `(from, size)`.
    Pages(Value, &'static [(u64, u64)]),
    /// A scroll to exhaustion: the ids, sorted.
    Scroll(Value),
    /// A point in time (opened with the preference) paged with
    /// `search_after` over the implicit `_shard_doc` tiebreaker.
    Pit(Value),
    Msearch(Vec<Value>),
    /// `_count` of a query.
    Count(Value),
}

struct Item {
    name: &'static str,
    kind: Kind,
    check: Check,
}

fn battery() -> Vec<Item> {
    let item = |name, kind, check| Item { name, kind, check };
    let aggs = |a: Value| Kind::Body(json!({"size": 0, "track_total_hits": true, "aggs": a}));
    let aggs_q = |q: Value, a: Value| Kind::Body(json!({"size": 0, "track_total_hits": true, "query": q, "aggs": a}));
    vec![
        // 61 tags: shard_size 100 sees them all, so the top 5 are exact.
        item("terms_size_below_term_count", aggs(json!({"t": {"terms": {"field": "tag", "size": 5, "shard_size": 100}}})), Check::Exact),
        item("terms_shard_size_bounded", aggs(json!({"top": {"terms": {"field": "tag", "size": 3, "shard_size": 3}}, "all": {"terms": {"field": "tag", "size": 100}}})), Check::TermsBound),
        item("terms_order_key", aggs(json!({"t": {"terms": {"field": "tag", "size": 7, "order": {"_key": "desc"}}}})), Check::Exact),
        item("terms_order_count_asc", aggs(json!({"t": {"terms": {"field": "cat", "size": 4, "shard_size": 50, "order": [{"_count": "asc"}, {"_key": "asc"}]}}})), Check::Exact),
        item("terms_order_sub_agg", aggs(json!({"t": {"terms": {"field": "cat", "size": 5, "order": [{"av": "desc"}, {"_key": "asc"}]}, "aggs": {"av": {"avg": {"field": "n"}}}}})), Check::Exact),
        item("terms_min_doc_count_0", aggs_q(json!({"bool": {"filter": [{"range": {"n": {"lt": 300}}}, {"term": {"flag": true}}]}}),
            json!({"t": {"terms": {"field": "tag", "size": 100, "min_doc_count": 0}}})), Check::Exact),
        item("terms_missing", aggs(json!({"t": {"terms": {"field": "cat", "size": 20, "missing": "none"}}})), Check::Exact),
        item("composite_paged", Kind::Composite { query: json!({"match_all": {}}), size: 40, sources: json!([
            {"cat": {"terms": {"field": "cat", "missing_bucket": true}}},
            {"week": {"date_histogram": {"field": "d", "calendar_interval": "week", "missing_bucket": true}}},
            {"flag": {"terms": {"field": "flag", "order": "desc"}}}]) }, Check::Exact),
        item("histogram_extended_bounds", aggs(json!({"h": {"histogram": {"field": "n", "interval": 750, "min_doc_count": 0, "extended_bounds": {"min": -1500, "max": 12000}},
            "aggs": {"p": {"stats": {"field": "price"}}}}})), Check::Exact),
        item("date_histogram_extended_bounds", aggs(json!({
            "dh": {"date_histogram": {"field": "d", "fixed_interval": "5d", "min_doc_count": 0, "extended_bounds": {"min": BASE_MS - 10 * DAY_MS, "max": BASE_MS + 100 * DAY_MS}}},
            "m": {"date_histogram": {"field": "d", "calendar_interval": "month", "time_zone": "+05:30"}}})), Check::Exact),
        item("range", aggs(json!({"r": {"range": {"field": "n", "ranges": [{"to": 100}, {"from": 100, "to": 2500}, {"from": 2500, "to": 7500}, {"from": 7500}]}, "aggs": {"s": {"sum": {"field": "n"}}}},
            "dr": {"date_range": {"field": "d", "ranges": [{"to": BASE_MS + 30 * DAY_MS}, {"from": BASE_MS + 30 * DAY_MS}]}}})), Check::Exact),
        item("filters", aggs(json!({"f": {"filters": {"other_bucket_key": "rest", "filters": {"c00": {"term": {"cat": "c00"}}, "alpha": {"match": {"body": "alpha"}}, "dear": {"range": {"price": {"gte": 500}}}}}}})), Check::Exact),
        item("nested_reverse_nested", aggs(json!({"items": {"nested": {"path": "items"}, "aggs": {"sku": {"terms": {"field": "items.sku", "size": 50},
            "aggs": {"qty": {"sum": {"field": "items.qty"}}, "back": {"reverse_nested": {}, "aggs": {"cats": {"terms": {"field": "cat", "size": 20}}}}}}}}})), Check::Exact),
        item("cardinality", aggs(json!({"uid": {"cardinality": {"field": "uid"}}, "tag": {"cardinality": {"field": "tag"}},
            "per_cat": {"terms": {"field": "cat", "size": 20}, "aggs": {"n": {"cardinality": {"field": "n", "precision_threshold": 100}}}}})), Check::Tol(0.05)),
        item("percentiles", aggs(json!({"td": {"percentiles": {"field": "n", "percents": [1, 5, 25, 50, 75, 95, 99]}},
            "hdr": {"percentiles": {"field": "n", "percents": [50, 99], "hdr": {"number_of_significant_value_digits": 3}}},
            "ranks": {"percentile_ranks": {"field": "n", "values": [2500, 5000]}}})), Check::Tol(0.02)),
        item("stats", aggs(json!({"n": {"stats": {"field": "n"}}, "price": {"extended_stats": {"field": "price"}}, "sum": {"sum": {"field": "price"}}, "avg": {"avg": {"field": "n"}},
            "tags": {"value_count": {"field": "tag"}}, "first": {"min": {"field": "d"}}, "last": {"max": {"field": "d"}}, "no_cat": {"missing": {"field": "cat"}}})), Check::Exact),
        item("top_hits_sorted", aggs(json!({"t": {"terms": {"field": "cat", "size": 20}, "aggs": {"top": {"top_hits": {"size": 3, "sort": [{"n": {"order": "desc"}}, {"uid": "asc"}], "_source": ["uid", "n", "cat"]}}}}})), Check::Exact),
        item("significant_terms", aggs_q(json!({"match": {"body": "alpha"}}),
            json!({"sig": {"significant_terms": {"field": "tag", "size": 5, "shard_size": 200}}, "fg": {"terms": {"field": "tag", "size": 200}}})), Check::Sig),
        item("multi_terms", aggs(json!({"mt": {"multi_terms": {"terms": [{"field": "cat"}, {"field": "flag"}], "size": 10, "shard_size": 100}}})), Check::Exact),
        item("pipeline_cumulative_sum", aggs(json!({"w": {"date_histogram": {"field": "d", "calendar_interval": "week", "min_doc_count": 0},
            "aggs": {"s": {"sum": {"field": "n"}}, "cs": {"cumulative_sum": {"buckets_path": "s"}}, "cc": {"cumulative_sum": {"buckets_path": "_count"}}}}})), Check::Exact),
        item("pipeline_bucket_sort", aggs(json!({"t": {"terms": {"field": "cat", "size": 20}, "aggs": {"s": {"sum": {"field": "n"}},
            "srt": {"bucket_sort": {"sort": [{"s": {"order": "desc"}}], "from": 1, "size": 4}}}}})), Check::Exact),
        // Many ties on `cat` and `flag`; `uid` makes the order total.
        item("search_after_ties", Kind::SearchAfter(json!({"size": 400, "query": {"range": {"price": {"gte": 50}}},
            "sort": [{"cat": {"order": "asc", "missing": "_last"}}, {"flag": "desc"}, {"uid": "asc"}], "_source": ["uid", "cat"]})), Check::Exact),
        item("deep_pages", Kind::Pages(json!({"query": {"match_all": {}}, "sort": [{"n": {"order": "desc", "missing": "_first"}}, {"uid": "desc"}], "_source": false, "track_total_hits": true}),
            &[(0, 50), (1000, 50), (2950, 100), (4900, 100)]), Check::Exact),
        item("collapse", Kind::Body(json!({"size": 20, "query": {"match_all": {}}, "collapse": {"field": "cat"}, "sort": [{"n": {"order": "desc"}}, {"uid": "asc"}],
            "_source": ["uid", "n"], "track_total_hits": true})), Check::Exact),
        item("track_total_hits_true", Kind::Body(json!({"size": 0, "track_total_hits": true, "query": {"range": {"n": {"gte": 2000}}}})), Check::Exact),
        item("track_total_hits_false", Kind::Body(json!({"size": 3, "track_total_hits": false, "sort": [{"uid": "asc"}], "_source": false, "query": {"term": {"flag": true}}})), Check::Exact),
        // ES: more matches than the threshold -> {value: threshold, relation: gte}.
        item("track_total_hits_number", Kind::Body(json!({"size": 0, "track_total_hits": 1000, "query": {"match_all": {}}})), Check::Exact),
        item("track_total_hits_number_above", Kind::Body(json!({"size": 0, "track_total_hits": 100_000, "query": {"match": {"body": "beta"}}})), Check::Exact),
        // Scores from a field only: the same on every copy and shard split.
        item("min_score", Kind::Body(json!({"size": 30, "min_score": 9000, "track_total_hits": true,
            "query": {"function_score": {"query": {"match_all": {}}, "field_value_factor": {"field": "n", "missing": 0}, "boost_mode": "replace"}},
            "sort": ["_score", {"uid": "asc"}], "_source": ["n"]})), Check::Exact),
        item("scroll_all", Kind::Scroll(json!({"size": 700, "sort": ["_doc"], "_source": false, "query": {"match_all": {}}})), Check::Exact),
        item("pit_search_after", Kind::Pit(json!({"size": 600, "sort": [{"n": {"order": "asc"}}], "_source": false, "query": {"range": {"price": {"lt": 800}}}})), Check::Exact),
        item("msearch", Kind::Msearch(vec![
            json!({"size": 0, "aggs": {"c": {"terms": {"field": "cat", "size": 20}}}}),
            json!({"size": 10, "sort": [{"uid": "desc"}], "_source": ["uid"]}),
            json!({"size": 0, "track_total_hits": true, "query": {"term": {"tag": "t01"}}})]), Check::Exact),
        item("count", Kind::Count(json!({"bool": {"filter": [{"term": {"cat": "c01"}}], "must_not": [{"exists": {"field": "d"}}]}})), Check::Exact),
        item("count_all", Kind::Count(json!({"match_all": {}})), Check::Exact),
        // query_then_fetch with a fetch phase that highlights.
        item("highlight", Kind::Body(json!({"size": 25, "query": {"match": {"body": "gamma delta"}}, "sort": [{"uid": "asc"}],
            "highlight": {"fields": {"body": {}}}, "_source": ["uid"], "track_total_hits": true})), Check::Exact),
    ]
}

/// A way of choosing copies: the coordinating node and the preference.
struct Sel {
    node: usize,
    pref: Option<String>,
}

impl Sel {
    fn label(&self) -> String {
        format!("n{} {}", self.node, self.pref.as_deref().unwrap_or("(none)"))
    }
}

/// The whole index from each live node's own copies, and no preference
/// (rotating copies) from two coordinators.
fn selections(live: &[usize]) -> Vec<Sel> {
    let mut v = vec![Sel { node: live[0], pref: None }, Sel { node: *live.last().unwrap(), pref: None }];
    v.extend(live.iter().map(|&n| Sel { node: n, pref: Some(ALL_LOCAL.into()) }));
    v
}

/// Every shard, from the receiving node's copies only.
const ALL_LOCAL: &str = "_shards:0,1,2,3,4|_only_local";

fn with_pref(path: &str, pref: &Option<String>) -> String {
    match pref {
        None => path.to_string(),
        // `|` percent-encoded: not every URI parser takes it raw.
        Some(p) => format!("{path}{}preference={}", if path.contains('?') { '&' } else { '?' }, p.replace('|', "%7C")),
    }
}

/// Every shard answered.
fn shards_ok(v: &Value) -> Result<(), String> {
    let s = &v["_shards"];
    if s["failed"] != 0 || s["successful"] != s["total"] || s["total"] == 0 {
        return Err(format!("_shards {s}"));
    }
    Ok(())
}

async fn search(pl: &Procs, node: usize, path: &str, body: Value) -> Result<Value, String> {
    let (st, v) = pl.call(node, Method::POST, path, Some(body.clone())).await;
    if st != 200 {
        return Err(format!("{path} {body}: {st} {v}"));
    }
    shards_ok(&v).map_err(|e| format!("{path} {body}: {e}"))?;
    Ok(v)
}

/// Keys that legitimately differ between runs, copies and the two indices.
const DROP: [&str; 10] = ["took", "timed_out", "_shards", "_index", "_shard", "_node", "_scroll_id", "pit_id", "num_reduce_phases", "terminated_early"];

fn normalize(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().filter(|(k, _)| !DROP.contains(&k.as_str())).map(|(k, v)| (k.clone(), normalize(v))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(normalize).collect()),
        _ => v.clone(),
    }
}

fn hits(v: &Value) -> Vec<Value> {
    v["hits"]["hits"].as_array().cloned().unwrap_or_default()
}

/// No `_id` twice.
fn unique_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(format!("duplicate id {id}"));
        }
    }
    Ok(())
}

/// What `kind` observes on `index` under `sel`.
async fn observe(pl: &Procs, index: &str, sel: &Sel, kind: &Kind) -> Result<Value, String> {
    let node = sel.node;
    let path = with_pref(&format!("/{index}/_search"), &sel.pref);
    match kind {
        Kind::Body(b) => Ok(normalize(&search(pl, node, &path, b.clone()).await?)),
        Kind::Count(q) => {
            let path = with_pref(&format!("/{index}/_count"), &sel.pref);
            let v = search(pl, node, &path, json!({"query": q})).await?;
            Ok(json!({"count": v["count"]}))
        }
        Kind::Composite { query, sources, size } => {
            let mut after: Option<Value> = None;
            let mut buckets = Vec::new();
            for _ in 0..1000 {
                let mut comp = json!({"size": size, "sources": sources});
                if let Some(a) = &after {
                    comp["after"] = a.clone();
                }
                let v = search(pl, node, &path, json!({"size": 0, "query": query, "aggs": {"c": {"composite": comp}}})).await?;
                let page = v["aggregations"]["c"]["buckets"].as_array().cloned().unwrap_or_default();
                if page.is_empty() {
                    break;
                }
                buckets.extend(page);
                match v["aggregations"]["c"].get("after_key") {
                    Some(a) => after = Some(a.clone()),
                    None => break,
                }
            }
            Ok(json!({"buckets": buckets}))
        }
        Kind::SearchAfter(b) => {
            let mut all: Vec<Value> = Vec::new();
            for _ in 0..1000 {
                let mut body = b.clone();
                if let Some(last) = all.last() {
                    body["search_after"] = last["sort"].clone();
                }
                let page = hits(&search(pl, node, &path, body).await?);
                if page.is_empty() {
                    break;
                }
                all.extend(page.iter().map(normalize));
            }
            unique_ids(all.iter().filter_map(|h| h["_id"].as_str()))?;
            Ok(json!({"hits": all}))
        }
        Kind::Pages(b, pages) => {
            let mut out = Vec::new();
            for (from, size) in pages.iter() {
                let mut body = b.clone();
                body["from"] = json!(from);
                body["size"] = json!(size);
                let v = search(pl, node, &path, body).await?;
                out.push(json!({"from": from, "total": v["hits"]["total"], "hits": normalize(&v["hits"]["hits"])}));
            }
            Ok(json!(out))
        }
        Kind::Scroll(b) => {
            let path = with_pref(&format!("/{index}/_search?scroll=1m"), &sel.pref);
            let v = search(pl, node, &path, b.clone()).await?;
            let total = v["hits"]["total"]["value"].as_u64().ok_or_else(|| format!("no total: {v}"))?;
            let mut id = v["_scroll_id"].clone();
            let mut ids: Vec<String> = hits(&v).iter().filter_map(|h| h["_id"].as_str().map(String::from)).collect();
            for _ in 0..1000 {
                let v = search(pl, node, "/_search/scroll", json!({"scroll": "1m", "scroll_id": id})).await?;
                let page = hits(&v);
                if page.is_empty() {
                    break;
                }
                ids.extend(page.iter().filter_map(|h| h["_id"].as_str().map(String::from)));
                id = v["_scroll_id"].clone();
            }
            let (st, v) = pl.call(node, Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": [id]}))).await;
            if st != 200 {
                return Err(format!("clear scroll: {st} {v}"));
            }
            unique_ids(ids.iter().map(String::as_str))?;
            if ids.len() as u64 != total {
                return Err(format!("scrolled {} of {total}", ids.len()));
            }
            ids.sort();
            Ok(json!({"total": total, "ids": ids}))
        }
        Kind::Pit(b) => {
            let open = with_pref(&format!("/{index}/_pit?keep_alive=1m"), &sel.pref);
            let (st, v) = pl.call(node, Method::POST, &open, None).await;
            if st != 200 {
                return Err(format!("open pit: {st} {v}"));
            }
            let mut pit = v["id"].clone();
            let (mut ids, mut keys) = (Vec::new(), Vec::new());
            let mut after: Option<Value> = None;
            for _ in 0..1000 {
                let mut body = b.clone();
                body["pit"] = json!({"id": pit, "keep_alive": "1m"});
                if let Some(a) = &after {
                    body["search_after"] = a.clone();
                }
                let v = search(pl, node, "/_search", body).await?;
                if let Some(p) = v.get("pit_id") {
                    pit = p.clone();
                }
                let page = hits(&v);
                let Some(last) = page.last() else { break };
                after = Some(last["sort"].clone());
                for h in &page {
                    ids.push(h["_id"].as_str().unwrap_or_default().to_string());
                    keys.push(h["sort"][0].clone());
                }
            }
            let (st, v) = pl.call(node, Method::DELETE, "/_pit", Some(json!({"id": pit}))).await;
            if st != 200 {
                return Err(format!("close pit: {st} {v}"));
            }
            unique_ids(ids.iter().map(String::as_str))?;
            ids.sort();
            // Ties are broken by `_shard_doc`, which differs per copy: the
            // sort keys in order and the ids as a set.
            Ok(json!({"ids": ids, "keys": keys}))
        }
        Kind::Msearch(bodies) => {
            let mut lines = String::new();
            for b in bodies {
                let mut header = json!({"index": index});
                if let Some(p) = &sel.pref {
                    header["preference"] = json!(p);
                }
                lines.push_str(&format!("{header}\n{b}\n"));
            }
            let (st, v) = pl.ndjson(node, "/_msearch", lines).await;
            if st != 200 {
                return Err(format!("msearch: {st} {v}"));
            }
            let responses = v["responses"].as_array().cloned().unwrap_or_default();
            if responses.len() != bodies.len() {
                return Err(format!("msearch answered {} of {}: {v}", responses.len(), bodies.len()));
            }
            for r in &responses {
                shards_ok(r).map_err(|e| format!("msearch: {e} {r}"))?;
            }
            Ok(normalize(&json!(responses)))
        }
    }
}

/// The first difference between `got` and `want` (numbers within `tol`,
/// relative, absolute below 1), if any.
fn diff(got: &Value, want: &Value, tol: f64, path: &str) -> Option<String> {
    match (got, want) {
        (Value::Object(a), Value::Object(b)) => {
            for k in a.keys().chain(b.keys()) {
                match (a.get(k), b.get(k)) {
                    (Some(x), Some(y)) => {
                        if let Some(d) = diff(x, y, tol, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    (x, y) => return Some(format!("{path}.{k}: {} vs {}", x.unwrap_or(&Value::Null), y.unwrap_or(&Value::Null))),
                }
            }
            None
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path}: {} vs {} entries", a.len(), b.len()));
            }
            a.iter().zip(b).enumerate().find_map(|(i, (x, y))| diff(x, y, tol, &format!("{path}[{i}]")))
        }
        (Value::Number(a), Value::Number(b)) => {
            let exact_ints = tol <= FLOAT_TOL && !a.is_f64() && !b.is_f64();
            let same = if exact_ints { a == b } else { (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= tol * b.as_f64().unwrap().abs().max(1.0) };
            (!same).then(|| format!("{path}: {a} vs {b}"))
        }
        _ => (got != want).then(|| format!("{path}: {got} vs {want}")),
    }
}

fn strip(v: &mut Value, key: &str) {
    match v {
        Value::Object(m) => {
            m.remove(key);
            m.values_mut().for_each(|v| strip(v, key));
        }
        Value::Array(a) => a.iter_mut().for_each(|v| strip(v, key)),
        _ => {}
    }
}

fn counts(buckets: &Value) -> BTreeMap<String, u64> {
    buckets.as_array().into_iter().flatten().filter_map(|b| Some((b["key"].as_str()?.to_string(), b["doc_count"].as_u64()?))).collect()
}

/// ES's bound on a terms aggregation cut by `shard_size`: each returned
/// count is at most the true one and at least it minus the agg's
/// `doc_count_error_upper_bound`.
fn terms_bound(got: &Value, want: &Value) -> Result<(), String> {
    let truth = counts(&want["aggregations"]["all"]["buckets"]);
    let top = &got["aggregations"]["top"];
    let err = top["doc_count_error_upper_bound"].as_u64().ok_or_else(|| format!("no doc_count_error_upper_bound: {top}"))?;
    let buckets = top["buckets"].as_array().cloned().unwrap_or_default();
    if buckets.len() != want["aggregations"]["top"]["buckets"].as_array().map_or(0, Vec::len) {
        return Err(format!("top: {} buckets: {top}", buckets.len()));
    }
    for b in &buckets {
        let (key, count) = (b["key"].as_str().unwrap_or_default(), b["doc_count"].as_u64().unwrap_or(0));
        let t = *truth.get(key).ok_or_else(|| format!("top has unknown {key}"))?;
        if count > t || t - count > err {
            return Err(format!("top {key}: {count}, true {t}, error bound {err}"));
        }
    }
    Ok(())
}

/// Counts only: each `sig` bucket's foreground count is `fg`'s count.
fn sig_counts(v: &mut Value) -> Result<(), String> {
    let fg = counts(&v["aggregations"]["fg"]["buckets"]);
    let sig = v["aggregations"]["sig"].as_object_mut().ok_or("no sig")?;
    for (key, count) in counts(&sig["buckets"]) {
        if fg.get(&key) != Some(&count) {
            return Err(format!("sig {key}: {count}, foreground {:?}", fg.get(&key)));
        }
    }
    sig.remove("buckets");
    sig.remove("bg_count");
    Ok(())
}

/// Compare one observation with another selection's (`vs_ref` false) or
/// with `ref`'s.
fn compare(item: &Item, got: &Value, want: &Value, vs_ref: bool, deletes: bool) -> Result<(), String> {
    let (mut got, mut want) = (got.clone(), want.clone());
    if item.check == Check::Sig && deletes {
        sig_counts(&mut got)?;
        sig_counts(&mut want)?;
    }
    if vs_ref {
        if item.check == Check::TermsBound {
            terms_bound(&got, &want)?;
            got["aggregations"].as_object_mut().map(|m| m.remove("top"));
            want["aggregations"].as_object_mut().map(|m| m.remove("top"));
        }
        // Only an upper bound in ES; one shard reports it differently.
        strip(&mut got, "doc_count_error_upper_bound");
        strip(&mut want, "doc_count_error_upper_bound");
    }
    let tol = match item.check {
        Check::Tol(t) => t,
        _ => FLOAT_TOL,
    };
    diff(&got, &want, tol, "").map_or(Ok(()), Err)
}

/// Run the battery on `ref` (from `keeper`) and on `split` under each of
/// `sels`; fails listing every mismatch. Returns `ref`'s observations.
async fn run_battery(pl: &Procs, phase: &str, keeper: usize, sels: &[Sel], deletes: bool) -> BTreeMap<&'static str, Value> {
    let started = Instant::now();
    let mut errs = Vec::new();
    let mut wants = BTreeMap::new();
    for item in battery() {
        let want = match observe(pl, "ref", &Sel { node: keeper, pref: None }, &item.kind).await {
            Ok(w) => w,
            Err(e) => {
                errs.push(format!("{} on ref: {e}", item.name));
                continue;
            }
        };
        let gots = futures::future::join_all(sels.iter().map(|s| observe(pl, "split", s, &item.kind))).await;
        let mut ok: Vec<(String, Value)> = Vec::new();
        for (s, got) in sels.iter().zip(gots) {
            match got {
                Ok(g) => ok.push((s.label(), g)),
                Err(e) => errs.push(format!("{} [{}]: {e}", item.name, s.label())),
            }
        }
        for (label, got) in &ok {
            if let Err(e) = compare(&item, got, &want, true, deletes) {
                errs.push(format!("{} [{label}] vs ref: {e}", item.name));
            }
        }
        for (label, got) in ok.iter().skip(1) {
            if let Err(e) = compare(&item, got, &ok[0].1, false, deletes) {
                errs.push(format!("{} [{label}] vs [{}]: {e}", item.name, ok[0].0));
            }
        }
        wants.insert(item.name, want);
    }
    assert!(errs.is_empty(), "{phase}: {} mismatches\n{}", errs.len(), errs.join("\n"));
    eprintln!("{phase}: battery agreed over {} selections in {:?}", sels.len(), started.elapsed());
    wants
}

/// `ref` itself holds exactly the modelled docs.
fn assert_model(wants: &BTreeMap<&'static str, Value>, model: &BTreeMap<String, Value>) {
    assert_eq!(wants["count_all"]["count"], model.len(), "ref count");
    let ids: Vec<&str> = wants["scroll_all"]["ids"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(ids.iter().copied().eq(model.keys().map(String::as_str)), "ref ids differ from the model");
}

/// Each copy on its own: `_shards:k|_only_local` sent to each live node is
/// one shard served by that node (an explained hit's `_node`), every copy
/// of a shard holds the same ids, and the shards together hold the model's
/// ids, once each.
async fn copy_matrix(pl: &Procs, phase: &str, model: &BTreeMap<String, Value>) {
    let live = pl.live();
    let ids = pl.node_ids(live[0]).await;
    let mut errs = Vec::new();
    let mut all: Vec<String> = Vec::new();
    for k in 0..5u64 {
        let path = format!("/split/_search?preference=_shards:{k}%7C_only_local");
        let mut first: Option<(usize, Vec<String>)> = None;
        for &n in &live {
            let v = match search(pl, n, &path, json!({"size": 10_000, "sort": [{"uid": "asc"}], "_source": false, "track_total_hits": true})).await {
                Ok(v) => v,
                Err(e) => {
                    errs.push(format!("shard {k} on n{n}: {e}"));
                    continue;
                }
            };
            if v["_shards"]["total"] != 1 {
                errs.push(format!("shard {k} on n{n}: _shards {}", v["_shards"]));
            }
            let got: Vec<String> = hits(&v).iter().filter_map(|h| h["_id"].as_str().map(String::from)).collect();
            if v["hits"]["total"]["value"] != got.len() {
                errs.push(format!("shard {k} on n{n}: total {} but {} hits", v["hits"]["total"], got.len()));
            }
            match search(pl, n, &path, json!({"size": 1, "explain": true, "query": {"match_all": {}}})).await {
                Ok(v) => {
                    if let Some(h) = hits(&v).first().filter(|h| h["_node"] != ids[&n].as_str()) {
                        errs.push(format!("shard {k} sent to n{n} was served by {}", h["_node"]));
                    }
                }
                Err(e) => errs.push(format!("shard {k} on n{n} explained: {e}")),
            }
            if let Some((m, f)) = &first {
                if *f != got {
                    let only_m: Vec<&String> = f.iter().filter(|id| !got.contains(id)).take(5).collect();
                    let only_n: Vec<&String> = got.iter().filter(|id| !f.contains(id)).take(5).collect();
                    errs.push(format!("shard {k}: copies on n{m} ({}) and n{n} ({}) differ: only n{m} {only_m:?}, only n{n} {only_n:?}", f.len(), got.len()));
                }
            }
            if first.is_none() {
                first = Some((n, got));
            }
        }
        all.extend(first.map(|(_, f)| f).unwrap_or_default());
    }
    if let Err(e) = unique_ids(all.iter().map(String::as_str)) {
        errs.push(format!("across shards: {e}"));
    }
    all.sort();
    if !all.iter().map(String::as_str).eq(model.keys().map(String::as_str)) {
        errs.push(format!("the shards hold {} ids, the model {}", all.len(), model.len()));
    }
    assert!(errs.is_empty(), "{phase} copies: {} mismatches\n{}", errs.len(), errs.join("\n"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replica_copies_agree_with_each_other_and_with_one_shard() {
    let pl = start_procs(STEADY, &[]).await;
    let keeper = pl.master(0).await;
    create_split(&pl, keeper).await;
    create_ref(&pl, keeper).await;
    let mut rng = Rng(0x5EED_0001);
    let mut model = BTreeMap::new();
    load(&pl, keeper, &mut rng, &mut model, 0..DOCS).await;
    pl.wait_health(keeper, "split", "green", 90).await;
    let sels = selections(&pl.live());
    let wants = run_battery(&pl, "steady", keeper, &sels, false).await;
    assert_model(&wants, &model);
    copy_matrix(&pl, "steady", &model).await;

    // Index / update / delete batches on both, then the battery again.
    let mut next = DOCS;
    for _ in 0..3 {
        let lines = write_batch(&mut rng, &mut model, &mut next);
        bulk(&pl, keeper, "split", lines.clone()).await;
        bulk(&pl, keeper, "ref", lines).await;
    }
    refresh(&pl, keeper).await;
    let wants = run_battery(&pl, "after writes", keeper, &sels, true).await;
    assert_model(&wants, &model);
    copy_matrix(&pl, "after writes", &model).await;

    // A flush with deletes outstanding, then a force merge expunging them:
    // the copies still agree, and (no deleted docs left in any statistics)
    // significant_terms is exact again.
    let (st, v) = pl.call(keeper, Method::POST, "/split,ref/_flush", None).await;
    assert_eq!(st, 200, "{v}");
    copy_matrix(&pl, "after flush", &model).await;
    let (st, v) = pl.call(keeper, Method::POST, "/split,ref/_forcemerge?max_num_segments=1", None).await;
    assert_eq!(st, 200, "{v}");
    refresh(&pl, keeper).await;
    let wants = run_battery(&pl, "after force merge", keeper, &sels, false).await;
    assert_model(&wants, &model);
    copy_matrix(&pl, "after force merge", &model).await;
}

/// Wait until every shard of `index` has a started primary on a node other
/// than `dead`.
async fn wait_promoted(pl: &Procs, via: usize, index: &str, shards: u64, dead: usize) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut rows = Vec::new();
    while Instant::now() < deadline {
        rows = pl.shards(via, index).await;
        let promoted = (0..shards).all(|k| rows.iter().any(|r| r["shard"] == k.to_string() && r["prirep"] == "p" && r["state"] == "STARTED" && r["node"] != format!("n{dead}")));
        if promoted {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("{index}: no promotion off n{dead}: {}", json!(rows));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn acked_writes_survive_killing_a_primary_holder_mid_bulk() {
    let mut pl = start_procs(FAILOVER, &[]).await;
    let master = pl.master(0).await;
    create_split(&pl, master).await;
    // The victim holds the most primaries; `ref` lives elsewhere.
    let rows = pl.shards(master, "split").await;
    let primaries = |i: usize| rows.iter().filter(|r| r["prirep"] == "p" && r["node"] == format!("n{i}")).count();
    let victim = (0..pl.len()).max_by_key(|i| (primaries(*i), *i != master)).unwrap();
    assert!(primaries(victim) > 0, "{}", json!(rows));
    let keeper = if victim != master { master } else { (victim + 1) % pl.len() };
    create_ref(&pl, keeper).await;
    let mut rng = Rng(0x5EED_0002);
    let mut model = BTreeMap::new();
    load(&pl, keeper, &mut rng, &mut model, 0..DOCS).await;

    // Eight bulks of new docs through `keeper`; the victim is killed as the
    // second one goes out.
    let docs: Vec<(String, Value)> = (DOCS..DOCS + 2000).map(|i| (uid(i), doc(&mut rng, i))).collect();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<usize>();
    let writer = {
        let (c, url, docs) = (pl.c.clone(), format!("http://127.0.0.1:{}/split/_bulk", pl.p.http[keeper]), docs.clone());
        tokio::spawn(async move {
            let mut status: BTreeMap<String, u64> = BTreeMap::new();
            for (b, chunk) in docs.chunks(250).enumerate() {
                let mut lines = String::new();
                for (id, d) in chunk {
                    lines.push_str(&format!("{}\n{d}\n", json!({"index": {"_id": id}})));
                }
                let r = c.post(&url).header("content-type", "application/x-ndjson").body(lines).timeout(Duration::from_secs(120)).send().await;
                if let Ok(r) = r {
                    let v: Value = r.json().await.unwrap_or(Value::Null);
                    for it in v["items"].as_array().into_iter().flatten() {
                        if let (Some(id), Some(st)) = (it["index"]["_id"].as_str(), it["index"]["status"].as_u64()) {
                            status.insert(id.to_string(), st);
                        }
                    }
                }
                let _ = tx.send(b);
            }
            status
        })
    };
    assert_eq!(rx.recv().await, Some(0));
    pl.kill(victim);
    let status = writer.await.unwrap();
    let acked: Vec<&str> = docs.iter().map(|(id, _)| id.as_str()).filter(|id| matches!(status.get(*id), Some(200 | 201))).collect();
    assert!(acked.len() >= 250, "the bulk before the kill was acked: {} acked", acked.len());

    pl.wait_nodes(keeper, pl.len() - 1, 90).await;
    pl.wait_health(keeper, "split", "yellow", 90).await;
    wait_promoted(&pl, keeper, "split", 5, victim).await;
    let (st, v) = pl.call(keeper, Method::POST, "/split/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    // Every acked doc is on every live copy, once.
    for &n in &pl.live() {
        for pref in ["_local", "_only_local"] {
            let path = format!("/split/_search?preference={pref}");
            let v = search(&pl, n, &path, json!({"size": 0, "track_total_hits": true, "query": {"ids": {"values": acked}}})).await.unwrap();
            assert_eq!(v["hits"]["total"]["value"], acked.len(), "n{n} {pref}: {v}");
        }
    }
    // The unacked ones again (idempotent: the same source), then `ref` gets
    // them all.
    let mut lines = String::new();
    for (id, d) in &docs {
        lines.push_str(&format!("{}\n{d}\n", json!({"index": {"_id": id}})));
        model.insert(id.clone(), d.clone());
    }
    let resend: String = docs.iter().filter(|(id, _)| !acked.contains(&id.as_str())).map(|(id, d)| format!("{}\n{d}\n", json!({"index": {"_id": id}}))).collect();
    if !resend.is_empty() {
        bulk(&pl, keeper, "split", resend).await;
    }
    bulk(&pl, keeper, "ref", lines).await;
    refresh(&pl, keeper).await;
    let sels = selections(&pl.live());
    // Re-sent docs leave deleted ones behind, per copy: significant_terms
    // background counts (doc freqs count deleted docs) legitimately differ.
    let wants = run_battery(&pl, "after failover", keeper, &sels, true).await;
    assert_model(&wants, &model);
    copy_matrix(&pl, "after failover", &model).await;

    // The victim comes back and recovers its copies.
    pl.spawn(victim, &[]).await;
    pl.wait_nodes(keeper, pl.len(), 90).await;
    pl.wait_health(keeper, "split", "green", 120).await;
    let sels = selections(&pl.live());
    let wants = run_battery(&pl, "after restart", keeper, &sels, true).await;
    assert_model(&wants, &model);
    copy_matrix(&pl, "after restart", &model).await;
}

const PREFS: usize = 3;
const SHAPES: usize = 3;

/// The serving node of every hit of an explained search.
fn served_by(v: &Value) -> BTreeSet<String> {
    hits(v).iter().map(|h| h["_node"].as_str().unwrap_or("(none)").to_string()).collect()
}

/// ES's `Preference` / `OperationRouting` semantics on diverged copies
/// (about a third of the docs updated or deleted, no force merge, so
/// copies differ in segments and deleted-doc statistics).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn es_preferences_choose_copies_as_es_does() {
    let pl = start_procs(PREFS, &[]).await;
    let keeper = pl.master(0).await;
    create_split(&pl, keeper).await;
    create_ref(&pl, keeper).await;
    let mut rng = Rng(0x5EED_0003);
    let mut model = BTreeMap::new();
    load(&pl, keeper, &mut rng, &mut model, 0..DOCS).await;
    let mut next = DOCS;
    for _ in 0..5 {
        let lines = write_batch(&mut rng, &mut model, &mut next);
        bulk(&pl, keeper, "split", lines.clone()).await;
        bulk(&pl, keeper, "ref", lines).await;
    }
    refresh(&pl, keeper).await;
    let ids = pl.node_ids(keeper).await;
    let live = pl.live();
    let q = json!({"size": 200, "explain": true, "_source": false, "query": {"match": {"body": "alpha beta"}}});
    let mut errs = Vec::new();
    for &y in &live {
        let z = live[(y + 1) % live.len()];
        // `_local` / `_only_local`: the receiving node's copies (it holds
        // one of every shard).
        for (pref, want) in [("_local".to_string(), y), ("_only_local".to_string(), y), (format!("_prefer_nodes:{}", ids[&z]), z), (format!("_only_nodes:{}", ids[&z]), z)] {
            match search(&pl, y, &format!("/split/_search?preference={pref}"), q.clone()).await {
                Ok(v) => {
                    let by = served_by(&v);
                    if by != BTreeSet::from([ids[&want].clone()]) || v["_shards"]["total"] != 5 {
                        errs.push(format!("n{y} {pref}: served by {by:?} (want n{want}), _shards {}", v["_shards"]));
                    }
                }
                Err(e) => errs.push(format!("n{y} {pref}: {e}")),
            }
        }
    }
    // A session string picks the same copy of each shard from every
    // coordinator: the same pages, scores included.
    let mut first: Option<(usize, Vec<Value>)> = None;
    for &n in &live {
        let mut pages = Vec::new();
        for from in [0, 30, 60, 90] {
            let body = json!({"from": from, "size": 30, "explain": true, "_source": false, "query": {"match": {"body": "alpha beta"}}});
            match search(&pl, n, "/split/_search?preference=sess1", body).await {
                Ok(v) => pages.extend(hits(&v).iter().map(|h| json!([h["_id"], h["_score"], h["_node"]]))),
                Err(e) => errs.push(format!("n{n} sess1: {e}")),
            }
        }
        if let Err(e) = unique_ids(pages.iter().filter_map(|p| p[0].as_str())) {
            errs.push(format!("n{n} sess1 pages: {e}"));
        }
        if let Some((m, f)) = &first {
            if let Some(d) = diff(&json!(pages), &json!(f), 0.0, "") {
                errs.push(format!("sess1 via n{n} vs via n{m}: {d}"));
            }
        }
        if first.is_none() {
            first = Some((n, pages));
        }
    }
    // `_search_shards` names the same first copy from every coordinator.
    let mut groups: Option<Value> = None;
    for &n in &live {
        let (st, v) = pl.call(n, Method::GET, "/split/_search_shards?preference=sess1", None).await;
        if st != 200 {
            errs.push(format!("n{n} _search_shards: {st} {v}"));
            continue;
        }
        let first_copies = json!(v["shards"].as_array().into_iter().flatten().map(|g| json!([g[0]["shard"], g[0]["node"]])).collect::<Vec<_>>());
        if let Some(g) = groups.as_ref().filter(|g| **g != first_copies) {
            errs.push(format!("_search_shards via n{n}: {first_copies} vs {g}"));
        }
        if groups.is_none() {
            groups = Some(first_copies);
        }
    }
    // Names ES 9 does not have, and nodes that do not exist.
    for p in ["_primary", "_replica", "_primary_first", "_replica_first", "_foo"] {
        for path in [format!("/split/_search?preference={p}"), format!("/split/_count?preference={p}")] {
            let (st, v) = pl.call(keeper, Method::POST, &path, Some(json!({"query": {"match_all": {}}}))).await;
            if st != 400 || v["error"]["type"] != "illegal_argument_exception" || v["error"]["reason"] != format!("no Preference for [{p}]") {
                errs.push(format!("{path}: {st} {v}"));
            }
        }
    }
    let (st, v) = pl.call(keeper, Method::POST, "/split/_search?preference=_only_nodes:nosuchnode", Some(json!({"size": 0}))).await;
    let reason = v["error"]["reason"].as_str().unwrap_or_default();
    if st != 400 || v["error"]["type"] != "illegal_argument_exception" || !reason.starts_with("no data nodes with criteria [nosuchnode] found for shard: [split][") {
        errs.push(format!("_only_nodes:nosuchnode: {st} {v}"));
    }
    // `_only_local` where the node lacks some shards: those count as failed
    // (no copy to search), the rest answer.
    let (st, v) = pl.call(keeper, Method::PUT, "/sparse", Some(json!({"settings": {"number_of_shards": 5, "number_of_replicas": 0}, "mappings": mappings()}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(keeper, "sparse", "green", 60).await;
    let rows = pl.shards(keeper, "sparse").await;
    for &n in &live {
        let local = rows.iter().filter(|r| r["node"] == format!("n{n}")).count();
        let (st, v) = pl.call(n, Method::POST, "/sparse/_search?preference=_only_local", Some(json!({"size": 0}))).await;
        let s = &v["_shards"];
        if st != 200 || s["total"] != 5 || s["successful"] != local || s["failed"] != 5 - local {
            errs.push(format!("sparse _only_local on n{n} ({local} local shards): {st} {v}"));
        }
    }
    assert!(errs.is_empty(), "ES preference semantics: {} mismatches\n{}", errs.len(), errs.join("\n"));
}

/// Routing partitions and ES's `_shards` header shapes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn routing_partitions_and_shard_headers_match_es() {
    let pl = start_procs(SHAPES, &[]).await;
    let keeper = pl.master(0).await;
    // `routing_partition_size` 2 on 5x2: a routed search covers two shards,
    // and finds exactly that routing's docs of the query.
    let (st, v) = pl.call(keeper, Method::PUT, "/parts", Some(json!({"settings": {"number_of_shards": 5, "number_of_replicas": 2, "index.routing_partition_size": 2},
        "mappings": {"_routing": {"required": true}, "properties": {"r": {"type": "keyword"}, "d": {"type": "date"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    pl.wait_health(keeper, "parts", "green", 90).await;
    let mut lines = String::new();
    for i in 0..2000 {
        let r = format!("r{}", i % 20);
        lines.push_str(&format!("{}\n{}\n", json!({"index": {"_id": format!("p{i}"), "routing": r}}), json!({"r": r, "d": BASE_MS + (i % 20) as i64 * DAY_MS + (i / 20) as i64 * 1000})));
    }
    bulk(&pl, keeper, "parts", lines).await;
    let (st, v) = pl.call(keeper, Method::POST, "/parts/_refresh", None).await;
    assert_eq!(st, 200, "{v}");
    for n in pl.live() {
        for r in ["r0", "r7", "r19"] {
            let v = search(&pl, n, &format!("/parts/_search?routing={r}"), json!({"size": 0, "track_total_hits": true, "query": {"term": {"r": r}}})).await.unwrap();
            assert_eq!(v["_shards"]["total"], 2, "n{n} routing {r}: {v}");
            assert_eq!(v["hits"]["total"]["value"], 100, "n{n} routing {r}: {v}");
        }
        // No failures: exactly ES's four counters, no `failures` key.
        let v = search(&pl, n, "/parts/_search", json!({"size": 0})).await.unwrap();
        let keys: Vec<&str> = v["_shards"].as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["total", "successful", "skipped", "failed"], "{v}");
    }
    // The can-match phase (`pre_filter_shard_size` 1) skips the shards whose
    // `d` range cannot match; skipped shards count as successful.
    let (lo, hi) = (BASE_MS + 3 * DAY_MS, BASE_MS + 3 * DAY_MS + 200_000);
    let mut can_match = 0;
    for k in 0..5 {
        let v = search(&pl, keeper, &format!("/parts/_search?preference=_shards:{k}"), json!({"size": 0, "aggs": {"lo": {"min": {"field": "d"}}, "hi": {"max": {"field": "d"}}}})).await.unwrap();
        let (min, max) = (v["aggregations"]["lo"]["value"].as_f64(), v["aggregations"]["hi"]["value"].as_f64());
        if let (Some(min), Some(max)) = (min, max) {
            can_match += usize::from(min <= hi as f64 && max >= lo as f64);
        }
    }
    let v = search(&pl, keeper, "/parts/_search?pre_filter_shard_size=1", json!({"size": 0, "track_total_hits": true, "query": {"range": {"d": {"gte": lo, "lte": hi}}}})).await.unwrap();
    assert_eq!(v["_shards"]["successful"], 5, "{v}");
    assert_eq!(v["_shards"]["skipped"], 5 - can_match, "{can_match} shards can match: {v}");
    assert_eq!(v["hits"]["total"]["value"], 100, "{v}");
    // `max_concurrent_shard_requests` must be positive.
    let (st, v) = pl.call(keeper, Method::POST, "/parts/_search?max_concurrent_shard_requests=0", Some(json!({"size": 0}))).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["type"], "illegal_argument_exception", "{v}");
    assert_eq!(v["error"]["reason"], "maxConcurrentShardRequests must be >= 1", "{v}");
}
