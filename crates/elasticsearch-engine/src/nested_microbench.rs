//! Microbench of every Rally `nested` task shape, with result checks:
//!
//! `cargo test -p elasticsearch-engine --release nested_microbench -- --ignored --nocapture`
//!
//! A synthetic StackOverflow-like index (the Rally `nested` mapping plus an
//! `answers.score` integer): questions with 1-3 Zipf-skewed tags out of the
//! track's 35, a creation date in 2008-2016 and 0-20 nested answers (skewed
//! low) each with a user, a date up to two years after the question and a
//! score. Indexed through the engine write path in three segments.
//!
//! Sizes: `NESTED_BENCH_QUESTIONS` (default 200,000 questions, about 1M
//! answers), `NESTED_BENCH_RUNS` (timed runs after one warm-up, best shown;
//! default 3), `NESTED_BENCH_QUERIES` (randomized requests per run, default
//! 35). The randomized tasks cycle through the track's 35 tags and 35 dates
//! (`queries.csv`) as Rally does; cycling more distinct dates than the
//! segment filter cache holds reproduces the Rally cache misses.
//!
//! Each row prints the mean ms per request, the result, and the ES 9.5.5
//! full-corpus p50 (11.2M questions / 29.4M docs there, so only the order of
//! magnitude compares). Every result is checked against a brute-force
//! evaluation over the generated questions: hit totals (exact and capped),
//! scores per `score_mode`, inner hits (totals, offsets, order, `_source`)
//! and the monthly `date_histogram` buckets. A second pass repeats the
//! checks after deleting every 97th question. The test fails after printing
//! the table if any check failed.

use crate::index::WriteReq;
use crate::node::Node;
use chrono::{Datelike, TimeZone, Utc};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Rally `nested/queries.csv`: (tag, date).
const RALLY: [(&str, &str); 35] = [
    ("java", "2012-04-08T21:15:33.873Z"),
    ("c#", "2012-01-02T13:27:55.631Z"),
    ("javascript", "2011-09-29T09:31:37.345Z"),
    ("php", "2012-04-10T03:16:00.727Z"),
    ("android", "2012-04-08T15:02:52.091Z"),
    ("jquery", "2012-04-03T08:39:17.337Z"),
    ("python", "2012-04-09T04:44:32.264Z"),
    ("html", "2012-04-09T05:11:21.702Z"),
    ("c++", "2012-06-01T22:18:55.219Z"),
    ("ios", "2012-04-09T02:15:44.330Z"),
    ("mysql", "2011-07-16T14:27:53.863Z"),
    ("css", "2012-04-09T17:48:32.247Z"),
    ("sql", "2010-09-27T15:38:50.503Z"),
    ("asp.net", "2012-04-09T06:06:10.029Z"),
    ("objective-c", "2012-04-08T22:03:57.592Z"),
    ("ruby-on-rails", "2012-04-08T19:35:07.412Z"),
    (".net", "2012-04-08T16:37:16.992Z"),
    ("iphone", "2012-04-10T12:45:17.230Z"),
    ("c", "2012-04-10T01:22:20.110Z"),
    ("arrays", "2012-04-09T15:13:45.728Z"),
    ("sql-server", "2012-04-10T07:36:56.023Z"),
    ("angularjs", "2012-04-10T12:26:25.743Z"),
    ("ruby", "2012-04-08T02:18:56.402Z"),
    ("json", "2012-04-08T07:11:47.490Z"),
    ("ajax", "2012-04-08T08:05:14.004Z"),
    ("regex", "2012-04-08T09:13:29.263Z"),
    ("xml", "2012-04-07T21:37:43.372Z"),
    ("asp.net-mvc", "2012-04-09T21:23:43.302Z"),
    ("r", "2012-03-07T11:59:36.114Z"),
    ("linux", "2012-04-09T12:51:50.530Z"),
    ("wpf", "2012-04-08T11:18:54.110Z"),
    ("django", "2012-04-13T02:18:51.407Z"),
    ("node.js", "2012-04-07T04:50:14.554Z"),
    ("database", "2012-04-10T15:53:29.825Z"),
    ("xcode", "2011-10-09T16:32:45.480Z"),
];

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f64 {
        self.next_u64() as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

struct Answer {
    user: u32,
    date: i64,
    score: i64,
}

struct Question {
    tags: Vec<usize>,
    date: i64,
    user: u32,
    answers: Vec<Answer>,
    deleted: bool,
}

fn ms(s: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis()
}

fn month_key(ms: i64) -> i64 {
    let d = Utc.timestamp_millis_opt(ms).unwrap();
    Utc.with_ymd_and_hms(d.year(), d.month(), 1, 0, 0, 0).unwrap().timestamp_millis()
}

fn generate(n: usize, seed: u64) -> Vec<Question> {
    let mut r = Lcg(seed);
    let lo = ms("2008-08-01T00:00:00.000Z");
    let hi = ms("2016-12-31T00:00:00.000Z");
    let two_years = 2 * 365 * 86_400_000i64;
    (0..n)
        .map(|_| {
            let ntags = 1 + r.below(3) as usize;
            let mut tags = Vec::with_capacity(ntags);
            while tags.len() < ntags {
                // Zipf-like: low indices (java, c#, ...) dominate.
                let t = ((r.unit() * r.unit()) * RALLY.len() as f64) as usize;
                if !tags.contains(&t) {
                    tags.push(t);
                }
            }
            let date = lo + (r.unit() * (hi - lo) as f64) as i64;
            let k = (r.unit().powi(3) * 21.0) as usize;
            let answers = (0..k.min(20))
                .map(|_| Answer { user: r.below(50_000) as u32, date: date + (r.unit() * two_years as f64) as i64, score: r.below(200) as i64 - 20 })
                .collect();
            Question { tags, date, user: r.below(50_000) as u32, answers, deleted: false }
        })
        .collect()
}

fn source(i: usize, q: &Question) -> Value {
    json!({
        "user": format!("u{}", q.user),
        "creationDate": q.date,
        "title": format!("question {i} about {}", RALLY[q.tags[0]].0),
        "qid": i.to_string(),
        "tag": q.tags.iter().map(|&t| RALLY[t].0).collect::<Vec<_>>(),
        "answer_count": q.answers.len(),
        "answers": q.answers.iter().map(|a| json!({"user": format!("u{}", a.user), "date": a.date, "score": a.score})).collect::<Vec<_>>(),
    })
}

struct Row {
    task: String,
    ms: f64,
    result: String,
    check: String,
    ok: bool,
    es: Option<f64>,
}

#[derive(Default)]
struct Table {
    rows: Vec<Row>,
}

impl Table {
    fn add(&mut self, task: &str, ms: f64, result: impl Into<String>, ok: bool, check: impl Into<String>, es: Option<f64>) {
        let row = Row { task: task.to_string(), ms, result: result.into(), check: check.into(), ok, es };
        eprintln!("  {:<52} {:>10.3} ms  {:<26} {}{}", row.task, row.ms, row.result, if row.ok { "ok " } else { "FAIL " }, row.check);
        self.rows.push(row);
    }

    fn print_and_assert(&self) {
        eprintln!("\n{:<52} {:>11} {:>10}  {:<28} check", "task", "ms/request", "ES p50", "result");
        for r in &self.rows {
            let es = r.es.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into());
            eprintln!("{:<52} {:>11.3} {:>10}  {:<28} {}{}", r.task, r.ms, es, r.result, if r.ok { "ok " } else { "FAIL " }, r.check);
        }
        let failed: Vec<&str> = self.rows.iter().filter(|r| !r.ok).map(|r| r.task.as_str()).collect();
        assert!(failed.is_empty(), "result checks failed: {failed:?}");
    }
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn search(svc: &Arc<crate::index::IndexService>, body: &Value) -> Value {
    let req = crate::search::parse_request(body, &std::collections::HashMap::new()).unwrap();
    let targets = vec![crate::search::Target::new(svc.clone())];
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    serde_json::from_str(&crate::search::execute(&targets, &req, &opts).unwrap().json).unwrap_or_else(|e| panic!("{body}: {e}"))
}

/// Run every body once to warm up, then `runs` timed passes over all of
/// them: the last pass's responses and the best pass's mean ms per request.
fn timed(svc: &Arc<crate::index::IndexService>, runs: usize, bodies: &[Value]) -> (Vec<Value>, f64) {
    let mut out: Vec<Value> = bodies.iter().map(|b| search(svc, b)).collect();
    let mut best = f64::INFINITY;
    for _ in 0..runs.max(1) {
        let t = Instant::now();
        out = bodies.iter().map(|b| search(svc, b)).collect();
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / bodies.len().max(1) as f64);
    }
    (out, best)
}

fn index_docs(svc: &Arc<crate::index::IndexService>, qs: &[Question]) -> f64 {
    let n = qs.len();
    let mut ms = 0.0;
    let mut batch = Vec::with_capacity(5_000);
    let flush_at = [n / 3, 2 * n / 3];
    for (i, q) in qs.iter().enumerate() {
        batch.push(WriteReq::Index { id: Some(i.to_string()), source: source(i, q).to_string(), routing: None, cond: Default::default() });
        if batch.len() == 5_000 || i + 1 == n || flush_at.contains(&i) {
            let t = Instant::now();
            for r in svc.write(std::mem::take(&mut batch), false) {
                r.unwrap();
            }
            if flush_at.contains(&i) {
                svc.flush().unwrap();
            }
            ms += t.elapsed().as_secs_f64() * 1e3;
        }
    }
    let t = Instant::now();
    svc.flush().unwrap();
    svc.refresh().unwrap();
    ms + t.elapsed().as_secs_f64() * 1e3
}

fn total(res: &Value) -> (u64, String) {
    let t = &res["hits"]["total"];
    (t["value"].as_u64().unwrap_or(u64::MAX), t["relation"].as_str().unwrap_or("?").to_string())
}

/// ES's default `track_total_hits` (10,000) rendering of `count`.
fn capped(count: u64) -> (u64, String) {
    if count > 10_000 {
        (10_000, "gte".into())
    } else {
        (count, "eq".into())
    }
}

fn hits(res: &Value) -> Vec<Value> {
    res["hits"]["hits"].as_array().cloned().unwrap_or_default()
}

fn hit_id(h: &Value) -> usize {
    h["_id"].as_str().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX)
}

fn score(h: &Value) -> f32 {
    h["_score"].as_f64().map(|x| x as f32).unwrap_or(f32::NAN)
}

fn nested_range(lte: &str) -> Value {
    json!({"range": {"answers.date": {"lte": lte}}})
}

/// Rally `NestedQueryParamSource` (plus optional extras on the `nested`).
fn nested_body(tag: &str, lte: &str, nested_extra: Value) -> Value {
    let mut nested = json!({"path": "answers", "query": nested_range(lte)});
    if let (Some(m), Some(x)) = (nested.as_object_mut(), nested_extra.as_object()) {
        for (k, v) in x {
            m.insert(k.clone(), v.clone());
        }
    }
    json!({"query": {"bool": {"must": [{"match": {"tag": tag}}, {"nested": nested}]}}})
}

fn with(mut body: Value, k: &str, v: Value) -> Value {
    body.as_object_mut().unwrap().insert(k.to_string(), v);
    body
}

/// The (tag, date) pairs of the randomized tasks: tags and dates drawn
/// independently, as Rally's param sources do.
fn pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n).map(|i| (i % RALLY.len(), (i * 11 + 3) % RALLY.len())).collect()
}

struct Ctx<'a> {
    svc: &'a Arc<crate::index::IndexService>,
    qs: &'a [Question],
    runs: usize,
    pairs: Vec<(usize, usize)>,
    /// The `match: tag` score of each tag (constant per tag: keyword).
    tag_score: Vec<f32>,
    suffix: &'static str,
}

impl Ctx<'_> {
    fn matching_answers(&self, i: usize, lim: i64) -> Vec<usize> {
        self.qs[i].answers.iter().enumerate().filter(|(_, a)| a.date <= lim).map(|(o, _)| o).collect()
    }
    fn matches(&self, i: usize, tag: usize, lim: i64) -> bool {
        let q = &self.qs[i];
        !q.deleted && q.tags.contains(&tag) && q.answers.iter().any(|a| a.date <= lim)
    }
    fn count(&self, tag: usize, lim: i64) -> u64 {
        (0..self.qs.len()).filter(|&i| self.matches(i, tag, lim)).count() as u64
    }
    fn name(&self, s: &str) -> String {
        format!("{s}{}", self.suffix)
    }
}

fn tag_scores(svc: &Arc<crate::index::IndexService>) -> Vec<f32> {
    RALLY.iter().map(|(t, _)| hits(&search(svc, &json!({"query": {"match": {"tag": t}}, "size": 1}))).first().map(score).unwrap_or(0.0)).collect()
}

/// `randomized-nested-queries` and the score-mode / context variants.
fn nested_queries(c: &Ctx, t: &mut Table) {
    let bodies: Vec<Value> = c.pairs.iter().map(|&(tg, dt)| nested_body(RALLY[tg].0, RALLY[dt].1, json!({}))).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let mut ok = true;
    let mut why = String::new();
    let mut sum = 0u64;
    for (k, &(tg, dt)) in c.pairs.iter().enumerate() {
        let lim = ms(RALLY[dt].1);
        let want = c.count(tg, lim);
        sum += want;
        let r = &res[k];
        let hs = hits(r);
        let expect = c.tag_score[tg] + 1.0f32;
        let good = total(r) == capped(want) && hs.len() as u64 == want.min(10) && hs.iter().all(|h| c.matches(hit_id(h), tg, lim) && score(h) == expect);
        // Exact totals through `track_total_hits: true` (untimed).
        let exact = total(&search(c.svc, &with(bodies[k].clone(), "track_total_hits", json!(true))));
        if !good || exact != (want, "eq".to_string()) {
            ok = false;
            why = format!("{} <= {}: got {:?} / exact {:?}, want {want}", RALLY[tg].0, RALLY[dt].1, total(r), exact);
        }
    }
    let check = if ok { "totals, hits, score = tag + 1 (avg)".to_string() } else { why };
    t.add(&c.name("randomized-nested-queries"), ms_, format!("{} matches/query", sum / c.pairs.len().max(1) as u64), ok, check, Some(9.99));

    // score_mode variants (one hit page per mode): avg / max / min score
    // the constant child score 1, sum the matching answers, none 0.
    for mode in ["avg", "max", "min", "sum", "none"] {
        let bodies: Vec<Value> = c.pairs.iter().take(8).map(|&(tg, dt)| with(nested_body(RALLY[tg].0, RALLY[dt].1, json!({"score_mode": mode})), "track_total_hits", json!(true))).collect();
        let (res, ms_) = timed(c.svc, c.runs, &bodies);
        let mut ok = true;
        let mut why = String::new();
        for (k, &(tg, dt)) in c.pairs.iter().take(8).enumerate() {
            let lim = ms(RALLY[dt].1);
            let want = c.count(tg, lim);
            let hs = hits(&res[k]);
            let expect = |i: usize| {
                let n = c.matching_answers(i, lim).len() as f32;
                c.tag_score[tg] + match mode { "sum" => n, "none" => 0.0, _ => 1.0 }
            };
            let mut good = total(&res[k]) == (want, "eq".to_string()) && hs.len() as u64 == want.min(10);
            good &= hs.iter().all(|h| c.matches(hit_id(h), tg, lim) && score(h) == expect(hit_id(h)));
            good &= hs.windows(2).all(|w| score(&w[0]) >= score(&w[1]));
            if mode == "sum" {
                // The best page holds the questions with the most matching answers.
                let best = (0..c.qs.len()).filter(|&i| c.matches(i, tg, lim)).map(|i| c.matching_answers(i, lim).len()).max().unwrap_or(0);
                good &= hs.first().is_none_or(|h| c.matching_answers(hit_id(h), lim).len() == best);
            }
            if !good {
                ok = false;
                why = format!("{} <= {}: total {:?} want {want}; first {:?}", RALLY[tg].0, RALLY[dt].1, total(&res[k]), hs.first().map(|h| (hit_id(h), score(h))));
            }
        }
        let check = if ok { format!("exact totals, per-hit {mode} scores, order") } else { why };
        t.add(&c.name(&format!("nested score_mode {mode} (track_total_hits)")), ms_, "", ok, check, None);
    }

    // Nested in filter context, nested alone, and a selective date window.
    let bodies: Vec<Value> = c.pairs.iter().take(8).map(|&(tg, dt)| json!({"query": {"bool": {"must": [{"match": {"tag": RALLY[tg].0}}], "filter": [{"nested": {"path": "answers", "query": nested_range(RALLY[dt].1)}}]}}})).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let ok = c.pairs.iter().take(8).enumerate().all(|(k, &(tg, dt))| {
        let lim = ms(RALLY[dt].1);
        total(&res[k]) == capped(c.count(tg, lim)) && hits(&res[k]).iter().all(|h| c.matches(hit_id(h), tg, lim) && score(h) == c.tag_score[tg])
    });
    t.add(&c.name("nested in bool.filter"), ms_, "", ok, "totals, hits, score = tag", None);

    let bodies: Vec<Value> = c.pairs.iter().take(8).map(|&(_, dt)| json!({"query": {"nested": {"path": "answers", "query": nested_range(RALLY[dt].1)}}, "track_total_hits": true})).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let ok = c.pairs.iter().take(8).enumerate().all(|(k, &(_, dt))| {
        let lim = ms(RALLY[dt].1);
        let want = c.qs.iter().filter(|q| !q.deleted && q.answers.iter().any(|a| a.date <= lim)).count() as u64;
        total(&res[k]) == (want, "eq".to_string()) && hits(&res[k]).iter().all(|h| c.qs[hit_id(h)].answers.iter().any(|a| a.date <= lim) && score(h) == 1.0)
    });
    t.add(&c.name("nested alone (track_total_hits)"), ms_, "", ok, "exact totals, score 1", None);

    let bodies: Vec<Value> = c.pairs.iter().take(8).map(|&(tg, dt)| {
        let hi = RALLY[dt].1;
        let lo = Utc.timestamp_millis_opt(ms(hi) - 3 * 86_400_000).unwrap().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        json!({"query": {"bool": {"must": [{"match": {"tag": RALLY[tg].0}}, {"nested": {"path": "answers", "query": {"range": {"answers.date": {"gte": lo, "lte": hi}}}}}]}}, "track_total_hits": true})
    }).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let ok = c.pairs.iter().take(8).enumerate().all(|(k, &(tg, dt))| {
        let hi = ms(RALLY[dt].1);
        let lo = hi - 3 * 86_400_000;
        let m = |i: usize| !c.qs[i].deleted && c.qs[i].tags.contains(&tg) && c.qs[i].answers.iter().any(|a| a.date >= lo && a.date <= hi);
        let want = (0..c.qs.len()).filter(|&i| m(i)).count() as u64;
        total(&res[k]) == (want, "eq".to_string()) && hits(&res[k]).iter().all(|h| m(hit_id(h)))
    });
    t.add(&c.name("nested 3-day window (selective)"), ms_, "", ok, "exact totals, hits", None);
}

/// `randomized-nested-queries-with-inner-hits_*`.
fn inner_hits(c: &Ctx, t: &mut Table, task: &str, size: usize, ih: usize, es: f64) {
    let bodies: Vec<Value> = c.pairs.iter().map(|&(tg, dt)| with(nested_body(RALLY[tg].0, RALLY[dt].1, json!({"inner_hits": {"size": ih}})), "size", json!(size))).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let mut ok = true;
    let mut why = String::new();
    let mut inner = 0usize;
    for (k, &(tg, dt)) in c.pairs.iter().enumerate() {
        let lim = ms(RALLY[dt].1);
        let want = c.count(tg, lim);
        let hs = hits(&res[k]);
        let mut good = total(&res[k]) == capped(want) && hs.len() as u64 == want.min(size as u64);
        for h in &hs {
            let i = hit_id(h);
            good &= c.matches(i, tg, lim);
            let m = c.matching_answers(i, lim);
            let ihits = &h["inner_hits"]["answers"]["hits"];
            let got = ihits["hits"].as_array().cloned().unwrap_or_default();
            inner += got.len();
            good &= ihits["total"]["value"].as_u64() == Some(m.len() as u64) && got.len() == m.len().min(ih);
            for (j, x) in got.iter().enumerate() {
                let off = x["_nested"]["offset"].as_u64().unwrap_or(u64::MAX) as usize;
                good &= x["_nested"]["field"] == json!("answers") && Some(&off) == m.get(j) && x["_id"] == h["_id"];
                good &= x["_source"]["date"].as_i64() == c.qs[i].answers.get(off).map(|a| a.date) && score(x) == 1.0;
            }
            if !good && why.is_empty() {
                why = format!("{} <= {}: question {i}: inner total {:?} want {}, offsets {:?} want {:?}", RALLY[tg].0, RALLY[dt].1, ihits["total"]["value"], m.len(), got.iter().map(|x| x["_nested"]["offset"].clone()).collect::<Vec<_>>(), m);
            }
        }
        if !good {
            ok = false;
            if why.is_empty() {
                why = format!("{} <= {}: total {:?} want {want}", RALLY[tg].0, RALLY[dt].1, total(&res[k]));
            }
        }
    }
    let check = if ok { "hits, inner totals/offsets/order/_source".to_string() } else { why };
    t.add(&c.name(task), ms_, format!("{} inner hits/query", inner / c.pairs.len().max(1)), ok, check, Some(es));
}

fn buckets(v: &Value) -> BTreeMap<i64, u64> {
    v.as_array().into_iter().flatten().filter(|b| b["doc_count"].as_u64().unwrap_or(0) > 0).map(|b| (b["key"].as_i64().unwrap_or(0), b["doc_count"].as_u64().unwrap_or(0))).collect()
}

fn brute_months(c: &Ctx, keep: impl Fn(&Question) -> bool) -> (BTreeMap<i64, u64>, u64) {
    let mut m = BTreeMap::new();
    let mut n = 0;
    for q in c.qs.iter().filter(|q| !q.deleted && keep(q)) {
        for a in &q.answers {
            *m.entry(month_key(a.date)).or_insert(0) += 1;
            n += 1;
        }
    }
    (m, n)
}

/// `nested-date-histo`, and the same aggregation under a tag query.
fn date_histo(c: &Ctx, t: &mut Table) {
    let agg = json!({"answers": {"nested": {"path": "answers"}, "aggs": {"date_histo": {"date_histogram": {"field": "answers.date", "calendar_interval": "month"}}}}});
    let body = json!({"size": 0, "aggs": agg});
    let (res, ms_) = timed(c.svc, c.runs, std::slice::from_ref(&body));
    let a = &res[0]["aggregations"]["answers"];
    let (want, n) = brute_months(c, |_| true);
    let got = buckets(&a["date_histo"]["buckets"]);
    let zero_ok = a["date_histo"]["buckets"].as_array().into_iter().flatten().all(|b| b["doc_count"].as_u64() != Some(0) || !want.contains_key(&b["key"].as_i64().unwrap_or(0)));
    let ok = got == want && a["doc_count"].as_u64() == Some(n) && zero_ok;
    t.add(&c.name("nested-date-histo"), ms_, format!("{} answers, {} months", n, got.len()), ok, "doc_count, monthly buckets = brute", Some(262.0));

    let bodies: Vec<Value> = (0..4).map(|tg| json!({"size": 0, "query": {"match": {"tag": RALLY[tg].0}}, "aggs": agg})).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let ok = (0..4).all(|tg| {
        let a = &res[tg]["aggregations"]["answers"];
        let (want, n) = brute_months(c, |q| q.tags.contains(&tg));
        buckets(&a["date_histo"]["buckets"]) == want && a["doc_count"].as_u64() == Some(n)
    });
    t.add(&c.name("nested-date-histo under match tag"), ms_, "", ok, "doc_count, monthly buckets = brute", None);
}

/// The non-nested tasks of the track (regression guards for the shared
/// collection paths).
fn other_tasks(c: &Ctx, t: &mut Table) {
    let bodies: Vec<Value> = c.pairs.iter().map(|&(tg, _)| json!({"query": {"match": {"tag": RALLY[tg].0}}})).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let count = |tg: usize| c.qs.iter().filter(|q| !q.deleted && q.tags.contains(&tg)).count() as u64;
    let ok = c.pairs.iter().enumerate().all(|(k, &(tg, _))| total(&res[k]) == capped(count(tg)) && hits(&res[k]).iter().all(|h| c.qs[hit_id(h)].tags.contains(&tg)));
    t.add(&c.name("randomized-term-queries"), ms_, "", ok, "totals, hits", Some(2.31));

    let bodies: Vec<Value> = c.pairs.iter().map(|&(tg, _)| json!({"query": {"match": {"tag": RALLY[tg].0}}, "sort": [{"answers.date": {"mode": "max", "order": "desc", "nested": {"path": "answers"}}}]})).collect();
    let (res, ms_) = timed(c.svc, c.runs, &bodies);
    let ok = c.pairs.iter().enumerate().all(|(k, &(tg, _))| {
        let best = c.qs.iter().filter(|q| !q.deleted && q.tags.contains(&tg)).filter_map(|q| q.answers.iter().map(|a| a.date).max()).max();
        let hs = hits(&res[k]);
        let first = hs.first().and_then(|h| h["sort"][0].as_i64());
        total(&res[k]) == capped(count(tg)) && first == best
    });
    t.add(&c.name("randomized-sorted-term-queries"), ms_, "", ok, "totals, first sort value = brute max", Some(63.1));

    let body = json!({"query": {"match_all": {}}});
    let (res, ms_) = timed(c.svc, c.runs, std::slice::from_ref(&body));
    let live = c.qs.iter().filter(|q| !q.deleted).count() as u64;
    t.add(&c.name("match-all"), ms_, "", total(&res[0]) == capped(live), "total", Some(1.81));
}

fn run_all(c: &Ctx, t: &mut Table) {
    nested_queries(c, t);
    inner_hits(c, t, "randomized-nested-queries-with-inner-hits_default", 10, 3, 10.2);
    inner_hits(c, t, "randomized-nested-queries-with-inner-hits_default_big_size", 100, 100, 21.0);
    date_histo(c, t);
    other_tasks(c, t);
}

#[test]
#[ignore]
fn nested_microbench() {
    let n = env("NESTED_BENCH_QUESTIONS", 200_000);
    let runs = env("NESTED_BENCH_RUNS", 3);
    let nq = env("NESTED_BENCH_QUERIES", 35);
    let mut qs = generate(n, 0x5eed_0f_5011);
    let answers: usize = qs.iter().map(|q| q.answers.len()).sum();
    eprintln!("nested microbench: {n} questions, {answers} answers ({} docs)", n + answers);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("sonested", &index_body(true)).unwrap();
    let svc = node.get_index("sonested").unwrap();
    let write_ms = index_docs(&svc, &qs);
    eprintln!("  indexed in {:.1} s ({:.0} questions/s)", write_ms / 1e3, n as f64 / (write_ms / 1e3));
    let mut t = Table::default();
    let tag_score = tag_scores(&svc);
    run_all(&Ctx { svc: &svc, qs: &qs, runs, pairs: pairs(nq), tag_score: tag_score.clone(), suffix: "" }, &mut t);

    // Deletes: every 97th question (its answers go with it).
    let dels: Vec<WriteReq> = (0..n).step_by(97).map(|i| WriteReq::Delete { id: i.to_string(), routing: None, cond: Default::default() }).collect();
    for r in svc.write(dels, false) {
        r.unwrap();
    }
    svc.refresh().unwrap();
    for i in (0..n).step_by(97) {
        qs[i].deleted = true;
    }
    let tag_score = tag_scores(&svc);
    run_all(&Ctx { svc: &svc, qs: &qs, runs: 1, pairs: pairs(nq.min(12)), tag_score, suffix: " (after deletes)" }, &mut t);

    // Merge cost: force merge to one segment, against the same sources
    // with `answers` a plain object (no child docs), to isolate what nested
    // blocks add (Rally `nested` merge time: rustic 418 s, ES 60 s).
    node.create_index("soflat", &index_body(false)).unwrap();
    let flat = node.get_index("soflat").unwrap();
    index_docs(&flat, &qs);
    let dels: Vec<WriteReq> = (0..n).step_by(97).map(|i| WriteReq::Delete { id: i.to_string(), routing: None, cond: Default::default() }).collect();
    for r in flat.write(dels, false) {
        r.unwrap();
    }
    flat.refresh().unwrap();
    for (label, index, docs) in [("force merge to 1 segment (nested answers)", &svc, n + answers), ("force merge to 1 segment (answers as object)", &flat, n)] {
        let t0 = Instant::now();
        index.force_merge(1, false).unwrap();
        index.refresh().unwrap();
        let ms_ = t0.elapsed().as_secs_f64() * 1e3;
        t.add(label, ms_, format!("{docs} docs, {:.0} ns/doc", ms_ * 1e6 / docs.max(1) as f64), true, "wall ms (not per request)", None);
    }
    let tag_score = tag_scores(&svc);
    run_all(&Ctx { svc: &svc, qs: &qs, runs: 1, pairs: pairs(nq.min(12)), tag_score, suffix: " (after merge)" }, &mut t);
    t.print_and_assert();
}

/// The Rally `nested` mapping (plus `answers.score`); `answers` a plain
/// object when not `nested`.
fn index_body(nested: bool) -> Value {
    let mut answers = json!({"properties": {"user": {"type": "keyword"}, "date": {"type": "date"}, "score": {"type": "integer"}}});
    if nested {
        answers["type"] = json!("nested");
    }
    json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"dynamic": "strict", "properties": {
            "user": {"type": "keyword"}, "creationDate": {"type": "date"}, "title": {"type": "text"},
            "qid": {"type": "keyword"}, "tag": {"type": "keyword"}, "answer_count": {"type": "integer"},
            "answers": answers
        }}
    })
}
