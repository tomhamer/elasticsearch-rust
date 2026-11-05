//! Process-local compiler telemetry, matching the process-local compilation cache.
//! History uses one counter per second, bounded to 24 hours per context.
use super::ScriptContext;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::{collections::{BTreeMap, VecDeque}, sync::OnceLock, time::Instant};

/// Per-second event counts, bounded to 24 hours.
#[derive(Default)]
struct History(VecDeque<(u64, u64)>);
impl History {
    fn prune(&mut self, now: u64) {
        while self.0.front().is_some_and(|(at, _)| now.saturating_sub(*at) >= 86_400) { self.0.pop_front(); }
    }
    fn record(&mut self, now: u64) {
        self.prune(now);
        if self.0.back().is_some_and(|(at, _)| *at == now) { self.0.back_mut().unwrap().1 += 1; } else { self.0.push_back((now, 1)); }
    }
    fn windows(&mut self, now: u64) -> [u64; 3] {
        self.prune(now);
        [300, 900, 86_400].map(|window| self.0.iter().filter(|(at, _)| now.saturating_sub(*at) < window).map(|(_, n)| n).sum())
    }
}
fn add(total: &mut [u64; 3], more: [u64; 3]) { for (t, m) in total.iter_mut().zip(more) { *t += m; } }
/// ES `TimeSeries`: written only when some window is non-zero.
fn put_history(out: &mut serde_json::Map<String, Value>, key: &str, windows: [u64; 3]) {
    if windows != [0; 3] { out.insert(key.into(), json!({"5m": windows[0], "15m": windows[1], "24h": windows[2]})); }
}

#[derive(Default)]
struct Counts {
    compilations: u64,
    evictions: u64,
    rejected: u64,
    history: History,
    eviction_history: History,
}
impl Counts {
    fn compiled(&mut self, now: u64) {
        self.compilations += 1;
        self.history.record(now);
    }
    fn evicted(&mut self, now: u64) {
        self.evictions += 1;
        self.eviction_history.record(now);
    }
}
#[derive(Default)]
struct Telemetry { contexts: BTreeMap<&'static str, Counts> }
impl Telemetry {
    /// ES `ScriptStats`. The general cache (the default) keeps one set of
    /// metrics and lists no contexts; the per-context caches list every
    /// context (`ScriptContextStats`, in name order), and the totals then
    /// carry no history.
    fn json(&mut self, now: u64, per_context: bool) -> Value {
        let (mut compilations, mut evictions, mut rejected) = (0, 0, 0);
        let (mut history, mut eviction_history) = ([0; 3], [0; 3]);
        let mut contexts = Vec::new();
        if per_context {
            for (name, _) in super::CONTEXT_NAMES { self.contexts.entry(*name).or_default(); }
        }
        for (name, counts) in self.contexts.iter_mut() {
            compilations += counts.compilations;
            evictions += counts.evictions;
            rejected += counts.rejected;
            let (h, e) = (counts.history.windows(now), counts.eviction_history.windows(now));
            add(&mut history, h);
            add(&mut eviction_history, e);
            if per_context {
                let mut context = serde_json::Map::new();
                context.insert("context".into(), json!(name));
                context.insert("compilations".into(), json!(counts.compilations));
                put_history(&mut context, "compilations_history", h);
                context.insert("cache_evictions".into(), json!(counts.evictions));
                put_history(&mut context, "cache_evictions_history", e);
                context.insert("compilation_limit_triggered".into(), json!(counts.rejected));
                contexts.push(Value::Object(context));
            }
        }
        let mut total = serde_json::Map::new();
        total.insert("compilations".into(), json!(compilations));
        total.insert("cache_evictions".into(), json!(evictions));
        total.insert("compilation_limit_triggered".into(), json!(rejected));
        if !per_context {
            put_history(&mut total, "compilations_history", history);
            put_history(&mut total, "cache_evictions_history", eviction_history);
        }
        total.insert("contexts".into(), Value::Array(contexts));
        Value::Object(total)
    }
}
fn state() -> &'static Mutex<Telemetry> {
    static STATE: OnceLock<Mutex<Telemetry>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(Telemetry::default()))
}
fn now() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs()
}
pub(super) fn compiled(context: ScriptContext) { state().lock().contexts.entry(context.name()).or_default().compiled(now()); }
pub(super) fn evicted(context: ScriptContext) { state().lock().contexts.entry(context.name()).or_default().evicted(now()); }
pub(super) fn rejected(context: ScriptContext) { state().lock().contexts.entry(context.name()).or_default().rejected += 1; }
pub fn stats_json() -> Value {
    let per_context = super::cache().lock().per_context();
    state().lock().json(now(), per_context)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn general_and_context_stats_have_the_es_shape_and_history_expires() {
        let mut telemetry = Telemetry::default();
        let score = telemetry.contexts.entry("score").or_default();
        score.compiled(0); score.compiled(0); score.compiled(600);
        score.evicted(600);
        telemetry.contexts.entry("filter").or_default().rejected = 2;
        let value = telemetry.json(600, false);
        assert_eq!(value["compilations"], 3);
        assert_eq!(value["cache_evictions"], 1);
        assert_eq!(value["compilation_limit_triggered"], 2);
        assert_eq!(value["compilations_history"], json!({"5m":1,"15m":3,"24h":3}));
        assert_eq!(value["cache_evictions_history"], json!({"5m":1,"15m":1,"24h":1}));
        assert_eq!(value["contexts"], json!([]));
        let keys: Vec<&String> = value.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["compilations", "cache_evictions", "compilation_limit_triggered", "compilations_history", "cache_evictions_history", "contexts"]);
        let value = telemetry.json(600, true);
        assert!(value.get("compilations_history").is_none());
        let contexts = value["contexts"].as_array().unwrap();
        assert_eq!(contexts.len(), super::super::CONTEXT_NAMES.len());
        let score = contexts.iter().find(|c| c["context"] == "score").unwrap();
        let keys: Vec<&String> = score.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["context", "compilations", "compilations_history", "cache_evictions", "cache_evictions_history", "compilation_limit_triggered"]);
        let idle = contexts.iter().find(|c| c["context"] == "update").unwrap();
        assert_eq!(idle, &json!({"context":"update","compilations":0,"cache_evictions":0,"compilation_limit_triggered":0}));
        let value = telemetry.json(87_000, false);
        assert_eq!(value["compilations"], 3);
        assert!(value.get("compilations_history").is_none() && value.get("cache_evictions_history").is_none());
    }
}
