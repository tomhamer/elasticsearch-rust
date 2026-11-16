//! Shard request cache (ES `index.requests.cache.enable`, default on):
//! responses of `size: 0` searches (hit counts, aggregations, suggestions)
//! are cached until the searched data changes. ES keys the cache by the
//! shard reader, so entries die at refresh; here documents are searchable
//! as soon as they are written, so the key carries a fingerprint of every
//! target's segments and memtables (see `Shard::data_fingerprint`), its
//! mapping and its settings/aliases.
//!
//! Like ES, `?request_cache=false` bypasses the cache, `?request_cache=true`
//! uses it even where the index setting disables it, and requests that are
//! not deterministic (`now`, scripts) or read other documents (terms
//! lookup, `more_like_this` docs, query rules, inference) are not cached.

use elasticsearch_engine::mapping::FieldLookup;
use elasticsearch_engine::search::Target;
use elasticsearch_engine::shard::Shard;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};

/// Total cached response bytes before the oldest entries are evicted.
const MAX_BYTES: usize = 64 << 20;
/// Responses larger than this are not cached.
const MAX_ENTRY: usize = 1 << 20;

pub static HITS: AtomicU64 = AtomicU64::new(0);
pub static MISSES: AtomicU64 = AtomicU64::new(0);
pub static EVICTIONS: AtomicU64 = AtomicU64::new(0);

struct Entry {
    json: Arc<str>,
    /// The indices (uuids) the response read: its bytes count toward each
    /// one's `memory_size_in_bytes`, its eviction toward their `evictions`.
    uuids: Vec<String>,
    /// The mappings the response was computed with (identity, not address:
    /// a replaced mapping fails the upgrade or the pointer comparison).
    lookups: Vec<Weak<FieldLookup>>,
}

#[derive(Default)]
struct Cache {
    map: HashMap<String, Entry>,
    order: VecDeque<String>,
    bytes: usize,
    /// Per index uuid (ES `RequestCacheStats` of its shards, summed).
    per_index: HashMap<String, IndexCounts>,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexCounts {
    pub memory_size_in_bytes: u64,
    pub evictions: u64,
    pub hit_count: u64,
    pub miss_count: u64,
}

impl Cache {
    fn counts(&mut self, uuid: &str) -> &mut IndexCounts {
        if !self.per_index.contains_key(uuid) {
            self.per_index.insert(uuid.to_string(), IndexCounts::default());
        }
        self.per_index.get_mut(uuid).expect("inserted above")
    }

    /// Account `e` (cached under `key`) leaving the cache.
    fn removed(&mut self, key: &str, e: &Entry, evicted: bool) {
        let size = (key.len() + e.json.len()) as u64;
        for u in &e.uuids {
            let c = self.counts(u);
            c.memory_size_in_bytes = c.memory_size_in_bytes.saturating_sub(size);
            if evicted { c.evictions += 1; }
        }
    }
}

/// The request cache statistics of index `uuid` (ES `request_cache` stats
/// section: its shards' figures summed).
pub fn index_stats(uuid: &str) -> IndexCounts {
    CACHE.lock().ok().and_then(|c| c.per_index.get(uuid).copied()).unwrap_or_default()
}

/// [`index_stats`] as the stats API renders it.
pub fn index_stats_json(uuid: &str) -> Value {
    let c = index_stats(uuid);
    serde_json::json!({"memory_size_in_bytes": c.memory_size_in_bytes, "evictions": c.evictions, "hit_count": c.hit_count, "miss_count": c.miss_count})
}

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(Default::default);

/// Request markers that make a search non-cacheable (substring test on the
/// serialized body: conservative, a false positive only skips the cache).
const UNCACHEABLE: &[&str] = &["now", "script", "random", "\"path\"", "\"like\"", "\"_id\"", "rule", "semantic", "sparse_vector", "text_expansion", "query_vector_builder", "indexed_shape", "inference"];

/// The cache key of a `size: 0` search, or `None` when it must not be
/// cached. `params` are the URL parameters, `body` the search body.
pub fn key(targets: &[Target], body: &Value, params: &HashMap<String, String>, size: usize) -> Option<(String, Vec<Weak<FieldLookup>>)> {
    if size != 0 || targets.is_empty() {
        return None;
    }
    let forced = match params.get("request_cache").map(String::as_str) {
        Some("false") => return None,
        Some("true") | Some("") => true,
        _ => false,
    };
    if params.contains_key("scroll") || body.get("pit").is_some() || body.get("profile").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let body_s = body.to_string();
    if UNCACHEABLE.iter().any(|m| body_s.contains(m)) {
        return None;
    }
    let mut key = String::with_capacity(body_s.len() + 128);
    let mut ps: Vec<(&String, &String)> = params.iter().filter(|(k, _)| !matches!(k.as_str(), "pretty" | "human" | "filter_path" | "error_trace" | "request_cache")).collect();
    if ps.iter().any(|(_, v)| v.contains("now")) {
        return None;
    }
    ps.sort();
    // Length-prefixed: a decoded `&`, `=` or newline inside a parameter value
    // (`q=foo&size=0` as one value) cannot read as another parameter map
    // (#566).
    for (k, v) in ps {
        key.push_str(&format!("{}:{k}{}:{v}", k.len(), v.len()));
    }
    key.push('\n');
    key.push_str(&body_s);
    let mut lookups = Vec::with_capacity(targets.len());
    for t in targets {
        if t.snapshot.is_some() {
            return None;
        }
        if !forced && t.svc.setting("index.requests.cache.enable").map(|v| v == "false").unwrap_or(false) {
            return None;
        }
        let lookup = t.svc.lookup();
        let fp: Vec<u64> = match &t.shard {
            Some(sh) => Shard::data_fingerprint(&sh.shard.reader_view()),
            None => t
                .svc
                .local_shards()
                .iter()
                .flat_map(|sh| {
                    let mut v = Shard::data_fingerprint(&sh.shard.reader_view());
                    if sh.id > 0 {
                        v.insert(0, u64::MAX - sh.id as u64);
                    }
                    v
                })
                .collect(),
        };
        use std::fmt::Write;
        let _ = write!(key, "\n{}|{:x}|{:x}|", t.svc.uuid, t.svc.meta_fingerprint(), Arc::as_ptr(&lookup) as usize);
        for x in fp {
            let _ = write!(key, "{x:x},");
        }
        // The alias filter is part of the query (ES rewrites it into the
        // shard request): `now` date math, scripts or lookups in it make the
        // request as uncacheable as they would in the body; a rolling-window
        // alias must never serve a count frozen at its first run.
        if let Some(f) = &t.alias_filter {
            let fs = f.to_string();
            if UNCACHEABLE.iter().any(|m| fs.contains(m)) {
                return None;
            }
            key.push_str(&fs);
        }
        if let Some(r) = &t.routing {
            let _ = write!(key, "|r={r}");
        }
        // A sparse index's open copies change as the cluster allocates them.
        if t.shard.is_none() && t.svc.is_sparse() {
            key.push_str("|open=");
            for sh in t.svc.local_shards() {
                let _ = write!(key, "{},", sh.id);
            }
        }
        lookups.push(Arc::downgrade(&lookup));
    }
    Some((key, lookups))
}

/// A cached response for `key` (with `took` reset to `took_ms`).
pub fn get(key: &str, targets: &[Target], took_ms: u128) -> Option<String> {
    let mut c = CACHE.lock().ok()?;
    let json = match c.map.get(key) {
        Some(e) if e.lookups.len() == targets.len() && e.lookups.iter().zip(targets).all(|(w, t)| w.upgrade().map(|l| Arc::ptr_eq(&l, &t.svc.lookup())).unwrap_or(false)) => Some(e.json.clone()),
        _ => None,
    };
    for t in targets {
        let counts = c.counts(&t.svc.uuid);
        if json.is_some() { counts.hit_count += 1 } else { counts.miss_count += 1 }
    }
    drop(c);
    let Some(json) = json else {
        MISSES.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    HITS.fetch_add(1, Ordering::Relaxed);
    Some(with_took(&json, took_ms))
}

/// Replace the leading `"took":N` of a search response.
fn with_took(json: &str, took_ms: u128) -> String {
    if let Some(rest) = json.strip_prefix("{\"took\":") {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        return format!("{{\"took\":{took_ms}{}", &rest[digits..]);
    }
    json.to_string()
}

/// Cache a successful, complete response.
/// Whether a search response completed on every shard: its own `_shards`
/// and `timed_out` read structurally (any failed count, any spelling), the
/// only responses the cache keeps.
fn complete_response(json: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct Shards {
        #[serde(default)]
        failed: serde_json::Value,
    }
    #[derive(serde::Deserialize)]
    struct Head {
        #[serde(default)]
        timed_out: bool,
        #[serde(rename = "_shards")]
        shards: Option<Shards>,
    }
    let Ok(h) = serde_json::from_str::<Head>(json) else { return false };
    let failed_none = h.shards.as_ref().is_some_and(|s| s.failed.as_f64() == Some(0.0) || s.failed.is_null());
    !h.timed_out && failed_none
}

pub fn put(key: String, lookups: Vec<Weak<FieldLookup>>, json: &str) {
    if json.len() > MAX_ENTRY || !json.starts_with("{\"took\":") || !complete_response(json) {
        return;
    }
    let Ok(mut c) = CACHE.lock() else { return };
    let size = key.len() + json.len();
    let uuids = key_uuids(&key);
    for u in &uuids {
        c.counts(u).memory_size_in_bytes += size as u64;
    }
    if let Some(old) = c.map.insert(key.clone(), Entry { json: Arc::from(json), uuids, lookups }) {
        c.bytes -= key.len() + old.json.len();
        c.removed(&key, &old, false);
    } else {
        c.order.push_back(key);
    }
    c.bytes += size;
    while c.bytes > MAX_BYTES {
        let Some(k) = c.order.pop_front() else { break };
        if let Some(e) = c.map.remove(&k) {
            c.bytes -= k.len() + e.json.len();
            c.removed(&k, &e, true);
            EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The index uuids a cache key names ([`key`] writes `\n{uuid}|` per target).
fn key_uuids(key: &str) -> Vec<String> {
    let mut out: Vec<String> = key.split('\n').skip(2).filter_map(|line| line.split('|').next()).filter(|u| !u.is_empty()).map(str::to_string).collect();
    out.sort();
    out.dedup();
    out
}

/// Drop every cached response that read one of the indices `uuids`
/// (`POST /_cache/clear?request`).
pub fn clear(uuids: &[String]) {
    let Ok(mut c) = CACHE.lock() else { return };
    let marks: Vec<String> = uuids.iter().map(|u| format!("\n{u}|")).collect();
    let doomed: Vec<String> = c.map.keys().filter(|k| marks.iter().any(|m| k.contains(m.as_str()))).cloned().collect();
    for k in doomed {
        if let Some(e) = c.map.remove(&k) {
            c.bytes -= k.len() + e.json.len();
            c.removed(&k, &e, false);
        }
    }
    let live: std::collections::VecDeque<String> = c.order.iter().filter(|k| c.map.contains_key(*k)).cloned().collect();
    c.order = live;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #172: a filtered alias with `now` date math is not cacheable (as a
    /// `now` in the body is not); a literal alias filter still is.
    #[test]
    fn dynamic_alias_filters_are_not_cached() {
        let dir = tempfile::tempdir().unwrap();
        let node = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        node.create_index("logs", &serde_json::json!({"mappings": {"properties": {"@timestamp": {"type": "date"}, "k": {"type": "keyword"}}}})).unwrap();
        let svc = node.get_index("logs").unwrap();
        let target = |filter: Option<Value>| Target { svc: svc.clone(), remote_index: None, alias_filter: filter, snapshot: None, access: None, shard: None, routing: None };
        let body = serde_json::json!({"size": 0, "aggs": {"n": {"value_count": {"field": "k"}}}});
        let params = HashMap::new();
        let rolling = serde_json::json!({"range": {"@timestamp": {"gte": "now-1d"}}});
        assert!(key(&[target(Some(rolling))], &body, &params, 0).is_none(), "a rolling-window alias is not cached");
        let lookup = serde_json::json!({"terms": {"k": {"index": "other", "id": "1", "path": "ks"}}});
        assert!(key(&[target(Some(lookup))], &body, &params, 0).is_none(), "a lookup in an alias filter is not cached");
        let fixed = serde_json::json!({"term": {"k": "a"}});
        let (k1, _) = key(&[target(Some(fixed.clone()))], &body, &params, 0).expect("a literal alias filter is cacheable");
        let (k2, _) = key(&[target(None)], &body, &params, 0).unwrap();
        assert_ne!(k1, k2, "the alias filter is part of the key");
    }

    /// #566: parameter maps whose values contain separators get distinct
    /// keys (`q=foo&size=0` as one value vs `q=foo`, `size=0`).
    #[test]
    fn separator_characters_in_parameters_do_not_collide() {
        let dir = tempfile::tempdir().unwrap();
        let node = elasticsearch_engine::node::Node::open(dir.path()).unwrap();
        node.create_index("k", &serde_json::json!({"mappings": {"properties": {"k": {"type": "keyword"}}}})).unwrap();
        let svc = node.get_index("k").unwrap();
        let t = Target { svc, remote_index: None, alias_filter: None, snapshot: None, access: None, shard: None, routing: None };
        let body = serde_json::json!({"size": 0});
        let p = |pairs: &[(&str, &str)]| pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect::<HashMap<_, _>>();
        let a = key(std::slice::from_ref(&t), &body, &p(&[("df", "k"), ("q", "foo&size=0")]), 0).unwrap().0;
        let b = key(std::slice::from_ref(&t), &body, &p(&[("df", "k"), ("q", "foo"), ("size", "0")]), 0).unwrap().0;
        assert_ne!(a, b);
        let c = key(std::slice::from_ref(&t), &body, &p(&[("a", "b=c")]), 0).unwrap().0;
        let d = key(std::slice::from_ref(&t), &body, &p(&[("a=b", "c")]), 0).unwrap().0;
        assert_ne!(c, d);
    }

    /// #372: any number of failed shards (or a timeout) keeps a response
    /// out of the cache; a complete one is cached.
    #[test]
    fn only_complete_responses_are_cached() {
        let body = |failed: u64, timed_out: bool| format!("{{\"took\":1,\"timed_out\":{timed_out},\"_shards\":{{\"total\":3,\"successful\":{},\"skipped\":0,\"failed\":{failed}}},\"hits\":{{\"total\":{{\"value\":0,\"relation\":\"eq\"}},\"max_score\":null,\"hits\":[]}}}}", 3 - failed);
        assert!(complete_response(&body(0, false)));
        for (failed, timed_out) in [(1, false), (2, false), (0, true)] {
            assert!(!complete_response(&body(failed, timed_out)), "{failed} {timed_out}");
            let key = format!("test-key-{failed}-{timed_out}");
            put(key.clone(), Vec::new(), &body(failed, timed_out));
            assert!(!CACHE.lock().unwrap().map.contains_key(&key));
        }
        put("test-key-ok".into(), Vec::new(), &body(0, false));
        assert!(CACHE.lock().unwrap().map.contains_key("test-key-ok"));
    }

    #[test]
    fn took_is_rewritten() {
        assert_eq!(with_took("{\"took\":123,\"timed_out\":false}", 0), "{\"took\":0,\"timed_out\":false}");
        assert_eq!(with_took("{\"x\":1}", 5), "{\"x\":1}");
    }
}
