//! Response post-processing shared by every cluster REST API: ES applies
//! `filter_path`, `pretty` and `human` to all JSON responses in its REST
//! layer, so the cluster APIs (which build their own bodies) must too.

use crate::front::{RestReq, RestResp};
use serde_json::{Map, Value};
use std::collections::HashMap;

fn flag(p: &HashMap<String, String>, k: &str) -> bool {
    matches!(p.get(k).map(String::as_str), Some("") | Some("true"))
}

fn is_json(r: &RestResp) -> bool {
    r.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("content-type") && v.contains("json"))
}

fn is_gzip(r: &RestResp) -> bool {
    r.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("content-encoding") && v.contains("gzip"))
}

/// Apply `human`, `filter_path` and `pretty` to a JSON response. Idempotent,
/// so responses the engine already processed are unaffected.
pub fn postprocess(req: &RestReq, resp: RestResp) -> RestResp {
    if !is_json(&resp) || is_gzip(&resp) || resp.body.is_empty() || req.method == "HEAD" {
        return resp;
    }
    let p = req.params();
    let human = flag(&p, "human");
    let pretty = flag(&p, "pretty");
    let fp = p.get("filter_path").filter(|s| !s.is_empty());
    if !human && !pretty && fp.is_none() {
        return resp;
    }
    render_json(resp, fp.map(String::as_str), pretty, human)
}

/// Apply `human`, then `filter_path`, then `pretty` to a JSON response, the
/// order ES's REST layer uses. Non-JSON, compressed and empty bodies pass
/// through (callers skip HEAD).
pub fn render_json(mut resp: RestResp, filter_path: Option<&str>, pretty: bool, human: bool) -> RestResp {
    if !is_json(&resp) || is_gzip(&resp) || resp.body.is_empty() {
        return resp;
    }
    let Some(mut v) = resp.body_json() else {
        if let Ok(raw)=std::str::from_utf8(&resp.body) {
            if let Ok(mut source)=elasticsearch_engine::source_json::Source::parse(raw) {
                if let Some(fp)=filter_path {source=source.select_structure(|v|elasticsearch_http::filter_path(v,fp));}
                resp.replace_body(if pretty {let mut s=serde_json::to_string_pretty(&source).unwrap();s.push('\n');s.into_bytes()}else{source.text().into_bytes()});
                resp.retain_headers(|(k,_)|!k.eq_ignore_ascii_case("content-length"));
            }
        }
        return resp;
    };
    if human {
        humanize(&mut v);
    }
    if let Some(fp) = filter_path {
        v = elasticsearch_http::filter_path(&v, fp);
    }
    resp.replace_body(if pretty {
        let mut s = serde_json::to_string_pretty(&v).unwrap_or_default();
        s.push('\n');
        s.into_bytes()
    } else {
        serde_json::to_vec(&v).unwrap_or_default()
    });
    resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
    resp
}

/// Remove response-shaping parameters from a URI before running it on an
/// engine whose output will be reshaped here.
pub fn strip_params(uri: &str, drop: &[&str]) -> String {
    let Some((path, q)) = uri.split_once('?') else { return uri.to_string() };
    let kept: Vec<&str> = q.split('&').filter(|kv| !kv.is_empty() && !drop.contains(&kv.split('=').next().unwrap_or(""))).collect();
    if kept.is_empty() { path.to_string() } else { format!("{path}?{}", kept.join("&")) }
}

pub fn iso_millis(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms).map(|d| d.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()).unwrap_or_default()
}

/// ES `TimeValue.toString()` for nanoseconds.
pub fn human_time_nanos(n: u64) -> String {
    const UNITS: [(u64, &str); 6] = [(86_400_000_000_000, "d"), (3_600_000_000_000, "h"), (60_000_000_000, "m"), (1_000_000_000, "s"), (1_000_000, "ms"), (1_000, "micros")];
    if n == 0 {
        return "0s".into();
    }
    for (div, u) in UNITS {
        if n >= div {
            let v = n as f64 / div as f64;
            let s = format!("{v:.1}");
            return format!("{}{u}", s.strip_suffix(".0").unwrap_or(&s));
        }
    }
    format!("{n}nanos")
}

/// `?human`: next to `X_in_bytes` / `X_in_millis` / `X_in_nanos` numbers
/// insert a readable `X` (size, duration, or ISO date for timestamps).
pub fn humanize(v: &mut Value) {
    match v {
        Value::Object(m) => {
            for x in m.values_mut() {
                humanize(x);
            }
            let needs = m.iter().any(|(k, x)| x.is_number() && (k.ends_with("_in_bytes") || k.ends_with("_in_millis") || k.ends_with("_in_nanos")));
            if !needs {
                return;
            }
            let old = std::mem::take(m);
            let existing: std::collections::HashSet<String> = old.keys().cloned().collect();
            let mut out = Map::new();
            for (k, x) in old {
                if let Some(n) = x.as_i64() {
                    let human = if let Some(base) = k.strip_suffix("_in_bytes") {
                        Some((base.to_string(), if n < 0 { n.to_string() } else { super::cat::human_bytes(n as u64) }))
                    } else if let Some(base) = k.strip_suffix("_in_millis") {
                        let date_like = base.ends_with("start_time") || base.ends_with("stop_time") || base.ends_with("_date") || base == "timestamp" || base.ends_with("_timestamp");
                        if date_like {
                            Some((base.to_string(), iso_millis(n)))
                        } else {
                            Some((base.to_string(), if n < 0 { n.to_string() } else { human_time_nanos(n as u64 * 1_000_000) }))
                        }
                    } else if let Some(base) = k.strip_suffix("_in_nanos") {
                        Some((base.to_string(), if n < 0 { n.to_string() } else { human_time_nanos(n as u64) }))
                    } else {
                        None
                    };
                    if let Some((base, h)) = human {
                        if !existing.contains(&base) && !out.contains_key(&base) {
                            out.insert(base, Value::String(h));
                        }
                    }
                }
                out.insert(k, x);
            }
            *m = out;
        }
        Value::Array(a) => a.iter_mut().for_each(humanize),
        _ => {}
    }
}

/// Lucene `LevenshteinDistance.getDistance` (1 - edits / max length).
fn similarity(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() || b.is_empty() {
        return if a == b { 1.0 } else { 0.0 };
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    1.0 - prev[b.len()] as f32 / a.len().max(b.len()) as f32
}

/// ES `BaseRestHandler.unrecognized`: `request [path] contains unrecognized
/// metric(s): [x] -> did you mean [y]?`.
pub fn unrecognized(path: &str, invalid: &[String], candidates: &[&str], kind: &str) -> String {
    let mut msg = format!("request [{path}] contains unrecognized {kind}{}: ", if invalid.len() > 1 { "s" } else { "" });
    let mut first = true;
    for bad in invalid {
        if !first {
            msg.push_str(", ");
        }
        first = false;
        msg.push_str(&format!("[{bad}]"));
        let mut scored: Vec<(f32, &str)> = candidates.iter().map(|c| (similarity(bad, c), *c)).filter(|(d, _)| *d > 0.5).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(b.1)));
        if !scored.is_empty() {
            let keys: Vec<&str> = scored.iter().map(|(_, k)| *k).collect();
            if keys.len() == 1 {
                msg.push_str(&format!(" -> did you mean [{}]?", keys[0]));
            } else {
                msg.push_str(&format!(" -> did you mean any of [{}]?", keys.join(", ")));
            }
        }
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_you_mean() {
        assert_eq!(unrecognized("/_nodes/stats/transprot", &["transprot".into()], &["transport", "jvm", "os"], "metric"), "request [/_nodes/stats/transprot] contains unrecognized metric: [transprot] -> did you mean [transport]?");
        assert_eq!(unrecognized("/x", &["zzz".into()], &["transport"], "metric"), "request [/x] contains unrecognized metric: [zzz]");
    }

    #[test]
    fn strip() {
        assert_eq!(strip_params("/a?filter_path=x&human=true&pretty", &["filter_path", "pretty"]), "/a?human=true");
        assert_eq!(strip_params("/a?pretty", &["pretty"]), "/a");
    }

    #[test]
    fn human() {
        let mut v = serde_json::json!({"start_time_in_millis": 0, "total_time_in_millis": 1500, "size_in_bytes": 2048});
        humanize(&mut v);
        assert_eq!(v["start_time"], "1970-01-01T00:00:00.000Z");
        assert_eq!(v["total_time"], "1.5s");
        assert_eq!(v["size"], "2kb");
        let before = v.clone();
        humanize(&mut v);
        assert_eq!(v, before);
    }
}
