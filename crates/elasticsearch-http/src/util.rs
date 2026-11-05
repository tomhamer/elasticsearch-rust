//! Shared helpers: `filter_path`, write responses, concurrency params,
//! byte sizes.

use crate::Req;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::shard::{VersionType, WriteCond, WriteResult};
use serde_json::{json, Map, Value};

fn seg_match(pat: &str, key: &str) -> bool {
    elasticsearch_engine::mapping::wildcard_match(pat, key)
}

fn apply_include(v: &Value, pats: &[Vec<String>]) -> Option<Value> {
    if pats.iter().any(|p| p.is_empty()) {
        return Some(v.clone());
    }
    match v {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, child) in m {
                let mut next: Vec<Vec<String>> = Vec::new();
                for p in pats {
                    if p[0] == "**" {
                        next.push(p.clone());
                        if p.len() > 1 && seg_match(&p[1], k) {
                            next.push(p[2..].to_vec());
                        }
                    } else if seg_match(&p[0], k) {
                        next.push(p[1..].to_vec());
                    }
                }
                if next.is_empty() {
                    continue;
                }
                if let Some(c) = apply_include(child, &next) {
                    out.insert(k.clone(), c);
                }
            }
            if out.is_empty() { None } else { Some(Value::Object(out)) }
        }
        Value::Array(a) => {
            let items: Vec<Value> = a.iter().filter_map(|x| apply_include(x, pats)).collect();
            if items.is_empty() { None } else { Some(Value::Array(items)) }
        }
        _ => None,
    }
}

fn apply_exclude(v: &mut Value, p: &[String]) {
    if p.is_empty() {
        return;
    }
    match v {
        Value::Object(m) => {
            let keys: Vec<String> = m.keys().cloned().collect();
            for k in keys {
                if p[0] == "**" {
                    if let Some(c) = m.get_mut(&k) {
                        apply_exclude(c, p);
                    }
                    if p.len() > 1 && seg_match(&p[1], &k) {
                        if p.len() == 2 {
                            m.remove(&k);
                        } else if let Some(c) = m.get_mut(&k) {
                            apply_exclude(c, &p[2..]);
                        }
                    }
                } else if seg_match(&p[0], &k) {
                    if p.len() == 1 {
                        m.remove(&k);
                    } else if let Some(c) = m.get_mut(&k) {
                        apply_exclude(c, &p[1..]);
                    }
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| apply_exclude(x, p)),
        _ => {}
    }
}

/// ES `?filter_path=a.b,hits.hits._id,**.key,-excluded`.
pub fn filter_path(v: &Value, spec: &str) -> Value {
    let mut inc: Vec<Vec<String>> = Vec::new();
    let mut exc: Vec<Vec<String>> = Vec::new();
    for p in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match p.strip_prefix('-') {
            Some(e) => exc.push(e.split('.').map(str::to_string).collect()),
            None => inc.push(p.split('.').map(str::to_string).collect()),
        }
    }
    let mut out = if inc.is_empty() { v.clone() } else { apply_include(v, &inc).unwrap_or_else(|| json!({})) };
    for e in &exc {
        apply_exclude(&mut out, e);
    }
    out
}

/// Concurrency / versioning parameters of a write.
pub fn write_cond(req: &Req) -> Result<WriteCond, EsError> {
    let mut c = WriteCond::default();
    let num = |k: &str| -> Result<Option<i64>, EsError> {
        match req.param(k) {
            None => Ok(None),
            Some(v) => v.parse::<i64>().map(Some).map_err(|_| EsError::illegal_argument(format!("Failed to parse long parameter [{k}] with value [{v}]"))),
        }
    };
    c.if_seq_no = num("if_seq_no")?;
    c.if_primary_term = num("if_primary_term")?.map(|x| x as u64);
    c.version = num("version")?.map(|x| x as u64);
    if let Some(vt) = req.param("version_type") {
        c.version_type = Some(VersionType::parse(vt)?);
    }
    match req.param("op_type") {
        None | Some("index") => {}
        Some("create") => c.create = true,
        Some(o) => return Err(EsError::illegal_argument(format!("opType must be 'create' or 'index', found: [{o}]"))),
    }
    Ok(c)
}

pub fn cond_from_meta(meta: &Map<String, Value>) -> Result<WriteCond, EsError> {
    let mut c = WriteCond::default();
    c.if_seq_no = meta.get("if_seq_no").and_then(Value::as_i64);
    c.if_primary_term = meta.get("if_primary_term").and_then(Value::as_u64);
    c.version = meta.get("version").and_then(Value::as_u64);
    if let Some(vt) = meta.get("version_type").and_then(Value::as_str) {
        c.version_type = Some(VersionType::parse(vt)?);
    }
    if meta.get("op_type").and_then(Value::as_str) == Some("create") {
        c.create = true;
    }
    Ok(c)
}

pub fn shards_json(svc: &IndexService) -> Value {
    let replicas = svc.setting_usize("index.number_of_replicas", 1);
    json!({"total": 1 + replicas, "successful": 1, "failed": 0})
}

/// The document write response body.
pub fn write_json(svc: &IndexService, r: &WriteResult, forced_refresh: bool) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("_index".into(), json!(svc.name));
    m.insert("_id".into(), json!(r.id));
    m.insert("_version".into(), json!(r.version));
    m.insert("result".into(), json!(r.result));
    if forced_refresh {
        m.insert("forced_refresh".into(), json!(true));
    }
    if r.result == "noop" {
        m.insert("_shards".into(), json!({"total": 0, "successful": 0, "failed": 0}));
    } else {
        m.insert("_shards".into(), shards_json(svc));
    }
    if svc.seq_no_disabled() {
        // `SequenceNumbers.UNASSIGNED_SEQ_NO` / `UNASSIGNED_PRIMARY_TERM`.
        m.insert("_seq_no".into(), json!(-2));
        m.insert("_primary_term".into(), json!(0));
    } else {
        m.insert("_seq_no".into(), json!(r.seq_no));
        m.insert("_primary_term".into(), json!(r.primary_term));
    }
    m
}

thread_local! {
    /// `refresh=wait_for` listeners registered by the request running on
    /// this thread ([`crate::blocking`] awaits them before responding).
    static DEFERRED_REFRESH: std::cell::RefCell<Option<Vec<tokio::sync::oneshot::Receiver<Result<bool, EsError>>>>> = const { std::cell::RefCell::new(None) };
}

/// Collect the refresh listeners `f` registers on this thread.
pub(crate) fn collect_deferred_refreshes<T>(f: impl FnOnce() -> T) -> (T, Vec<tokio::sync::oneshot::Receiver<Result<bool, EsError>>>) {
    let outer = DEFERRED_REFRESH.with(|d| d.borrow_mut().replace(Vec::new()));
    let out = f();
    let pending = DEFERRED_REFRESH.with(|d| std::mem::replace(&mut *d.borrow_mut(), outer)).unwrap_or_default();
    (out, pending)
}

/// Register a listener through `register` when the current request
/// collects them; whether it did (else the caller waits itself).
fn defer_refresh(register: impl FnOnce(elasticsearch_engine::shard::RefreshCallback)) -> bool {
    let collecting = DEFERRED_REFRESH.with(|d| d.borrow().is_some());
    if !collecting {
        return false;
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    DEFERRED_REFRESH.with(|d| if let Some(pending) = d.borrow_mut().as_mut() { pending.push(rx) });
    register(Box::new(move |result| { let _ = tx.send(result); }));
    true
}

/// `refresh` parameter: returns whether a forced refresh happened.
pub fn handle_refresh(param: Option<&str>, svc: &IndexService) -> Result<bool, EsError> {
    handle_refresh_doc(param, svc, None)
}

/// [`handle_refresh`] for a single-document write: only the shard holding
/// `doc` (`(_id, routing)`) is refreshed, as in ES.
pub fn handle_refresh_doc(param: Option<&str>, svc: &IndexService, doc: Option<(&str, Option<&str>)>) -> Result<bool, EsError> {
    let refresh = || match doc {
        Some((id, routing)) => svc.refresh_doc_shard(id, routing),
        None => svc.refresh(),
    };
    match param {
        None | Some("false") => Ok(false),
        Some("true") | Some("") => {
            refresh()?;
            Ok(true)
        }
        Some("wait_for") => {
            // ES parks the request until a refresh (the next scheduled one,
            // or any other) makes the write visible; past
            // `index.max_refresh_listeners` it forces one. With refreshes
            // disabled the request forces one.
            if svc.refresh_interval_ms() < 0 {
                refresh()?;
                return Ok(true);
            }
            // ES RefreshListeners: inside a request the write thread
            // returns at once and the response is sent once the refresh
            // completes the listener (`crate::blocking`).
            let deferred = defer_refresh(|done| match doc {
                Some((id, routing)) => svc.add_refresh_listener_doc(id, routing, done),
                None => svc.add_refresh_listener(done),
            });
            if deferred {
                return Ok(false);
            }
            match doc {
                Some((id, routing)) => svc.wait_for_refresh_doc(id, routing),
                None => svc.wait_for_refresh(),
            }
        }
        Some(o) => Err(EsError::illegal_argument(format!("Unknown value for refresh: [{o}]."))),
    }
}

/// ES byte-size formatting (`?bytes=` units or human `4.5kb`).
pub fn bytes_fmt(n: u64, unit: Option<&str>) -> String {
    match unit {
        Some("b") => n.to_string(),
        Some("kb") => (n / 1024).to_string(),
        Some("mb") => (n / (1024 * 1024)).to_string(),
        Some("gb") => (n / (1024 * 1024 * 1024)).to_string(),
        _ => {
            // `ByteSizeValue.toString()` truncates to one decimal.
            elasticsearch_engine::memory::byte_size(n)
        }
    }
}

pub fn human_bytes(n: u64) -> String {
    bytes_fmt(n, None)
}

/// ES `TimeValue.toString()`: largest unit with value >= 1, one decimal,
/// trailing `.0` dropped (`0s`, `250ms`, `1.5s`, `2m`, `1.2h`, `3d`).
pub fn human_time_nanos(nanos: u64) -> String {
    if nanos == 0 {
        return "0s".into();
    }
    let units: [(&str, u64); 7] = [
        ("d", 86_400_000_000_000),
        ("h", 3_600_000_000_000),
        ("m", 60_000_000_000),
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("micros", 1_000),
        ("nanos", 1),
    ];
    for (u, s) in units {
        if nanos >= s {
            let txt = format!("{:.1}", nanos as f64 / s as f64);
            let txt = txt.strip_suffix(".0").unwrap_or(&txt).to_string();
            return format!("{txt}{u}");
        }
    }
    "0s".into()
}

/// Whether `?human` output applies to this path: stats-style APIs whose
/// bodies never carry user documents.
pub fn human_applies(segs: &[String]) -> bool {
    segs.iter().any(|s| matches!(s.as_str(), "_stats" | "stats" | "_nodes" | "_cluster" | "_tasks" | "_recovery" | "_segments" | "_license" | "_xpack" | "_snapshot"))
}

/// `?human=true`: next to every `X_in_bytes` / `X_in_millis` / `X_in_nanos`
/// field insert a human-readable `X` (`"4.5kb"`, `"1.2s"`, or an ISO date
/// for timestamp-like millis fields), as ES's `XContentBuilder.humanReadable`.
pub fn humanize(v: &mut Value) {
    match v {
        Value::Object(m) => {
            let needs = m.iter().any(|(k, x)| x.is_number() && (k.ends_with("_in_bytes") || k.ends_with("_in_millis") || k.ends_with("_in_nanos")));
            for x in m.values_mut() {
                humanize(x);
            }
            if !needs {
                return;
            }
            let old = std::mem::take(m);
            let existing: std::collections::HashSet<String> = old.keys().cloned().collect();
            for (k, x) in old {
                if let Some(n) = x.as_i64() {
                    let human = if let Some(base) = k.strip_suffix("_in_bytes") {
                        Some((base, if n < 0 { n.to_string() } else { human_bytes(n as u64) }))
                    } else if let Some(base) = k.strip_suffix("_in_millis") {
                        let date_like = base.ends_with("start_time") || base.ends_with("_date") || base == "timestamp";
                        if date_like {
                            Some((base, elasticsearch_engine::date::DateFormatter::default().format(n, elasticsearch_engine::date::Tz::UTC)))
                        } else {
                            Some((base, if n < 0 { n.to_string() } else { human_time_nanos(n as u64 * 1_000_000) }))
                        }
                    } else if let Some(base) = k.strip_suffix("_in_nanos") {
                        Some((base, if n < 0 { n.to_string() } else { human_time_nanos(n as u64) }))
                    } else {
                        None
                    };
                    if let Some((base, h)) = human {
                        if !existing.contains(base) && !m.contains_key(base) {
                            m.insert(base.to_string(), Value::String(h));
                        }
                    }
                }
                m.insert(k, x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(humanize),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_paths() {
        let v = json!({"took": 1, "hits": {"total": {"value": 2}, "hits": [{"_id": "1", "_source": {"a": 1}}, {"_id": "2", "_source": {"a": 2}}]}});
        assert_eq!(filter_path(&v, "hits.hits._id"), json!({"hits": {"hits": [{"_id": "1"}, {"_id": "2"}]}}));
        assert_eq!(filter_path(&v, "took,hits.total.value"), json!({"took": 1, "hits": {"total": {"value": 2}}}));
        assert_eq!(filter_path(&v, "**._id"), json!({"hits": {"hits": [{"_id": "1"}, {"_id": "2"}]}}));
        assert_eq!(filter_path(&v, "-hits"), json!({"took": 1}));
        assert_eq!(filter_path(&v, "hits.*.value"), json!({"hits": {"total": {"value": 2}}}));
        assert_eq!(filter_path(&v, "nope"), json!({}));
    }

    #[test]
    fn human_fields() {
        assert_eq!(human_time_nanos(0), "0s");
        assert_eq!(human_time_nanos(250_000_000), "250ms");
        assert_eq!(human_time_nanos(1_500_000_000), "1.5s");
        assert_eq!(human_time_nanos(120_000_000_000), "2m");
        let mut v = json!({"store": {"size_in_bytes": 4608, "reserved_in_bytes": 0}, "indexing": {"index_total": 3, "index_time_in_millis": 1500}, "t": [{"running_time_in_nanos": 2000}]});
        humanize(&mut v);
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"store":{"size":"4.5kb","size_in_bytes":4608,"reserved":"0b","reserved_in_bytes":0},"indexing":{"index_total":3,"index_time":"1.5s","index_time_in_millis":1500},"t":[{"running_time":"2micros","running_time_in_nanos":2000}]}"#
        );
    }

    #[test]
    fn byte_sizes() {
        assert_eq!(human_bytes(225), "225b");
        assert_eq!(human_bytes(4608), "4.5kb");
        assert_eq!(human_bytes(1 << 20), "1mb");
        assert_eq!(bytes_fmt(4608, Some("b")), "4608");
    }
}

/// Physical memory `(total, free)` in bytes, if the OS tells us.
pub fn sys_memory() -> Option<(u64, u64)> {
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let get = |k: &str| {
            s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).map(|kb| kb * 1024)
        };
        let total = get("MemTotal:")?;
        return Some((total, get("MemAvailable:").or_else(|| get("MemFree:")).unwrap_or(0)));
    }
    // macOS / BSD.
    let out = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
    let total: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    let vm = std::process::Command::new("vm_stat").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let page: u64 = vm.split("page size of ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|p| p.parse().ok()).unwrap_or(4096);
    let pages = |k: &str| vm.lines().find(|l| l.starts_with(k)).and_then(|l| l.split(':').nth(1)).and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok()).unwrap_or(0);
    let free = (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")) * page;
    Some((total, free.min(total)))
}

/// Filesystem `(mount, total, available)` in bytes for `path` (POSIX `df -Pk`).
pub fn disk_usage(path: &std::path::Path) -> Option<(String, u64, u64)> {
    let out = std::process::Command::new("df").arg("-Pk").arg(path).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let line = text.lines().nth(1)?;
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() < 6 {
        return None;
    }
    let total = f[1].parse::<u64>().ok()? * 1024;
    let avail = f[3].parse::<u64>().ok()? * 1024;
    Some((f[5..].join(" "), total, avail))
}

/// Parameters every REST handler accepts (`BaseRestHandler.ALWAYS_CONSUMED`).
///
/// Superseded by the spec-driven prologue (`spec_validate`), which checks every
/// route against the parameters ES itself consumes. Kept only until the
/// remaining callers (`graph.rs`, `rollup.rs`) drop their duplicate checks.
const ALWAYS_CONSUMED: &[&str] = &["error_trace", "filter_path", "format", "human", "pretty", "source", "source_content_type"];

/// ES's `BaseRestHandler` check: any query parameter the handler does not consume is a 400
/// (`request [/s/_count] contains unrecognized parameter: [foo]`).
pub fn check_params(req: &Req, allowed: &[&str]) -> Result<(), EsError> {
    check_params_map(&req.path, &req.params, allowed)
}

pub(crate) fn check_params_map(path: &str, params: &std::collections::HashMap<String, String>, allowed: &[&str]) -> Result<(), EsError> {
    let mut bad: Vec<&String> = params.keys().filter(|k| !allowed.contains(&k.as_str()) && !ALWAYS_CONSUMED.contains(&k.as_str())).collect();
    if bad.is_empty() {
        return Ok(());
    }
    bad.sort();
    let list = bad.iter().map(|k| format!("[{k}]")).collect::<Vec<_>>().join(", ");
    Err(EsError::illegal_argument(format!("request [{}] contains unrecognized parameter{}: {list}", path, if bad.len() > 1 { "s" } else { "" })))
}
