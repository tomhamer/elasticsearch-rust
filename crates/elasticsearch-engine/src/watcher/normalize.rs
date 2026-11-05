//! Normalization of watch definitions to the form ES stores and echoes back
//! (defaults filled in, script shorthands expanded, Java `HashMap` ordering
//! for action maps).

use serde_json::{json, Map, Value};

/// Order of `keys` when iterated from a `java.util.HashMap` they were all
/// inserted into (default capacity, load factor 0.75).
pub fn java_order(keys: &[String]) -> Vec<String> {
    let n = keys.len();
    let mut cap = 16usize;
    while n > cap * 3 / 4 {
        cap *= 2;
    }
    let mut idx: Vec<(usize, usize, &String)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let mut h: i32 = 0;
            for c in k.encode_utf16() {
                h = h.wrapping_mul(31).wrapping_add(c as i32);
            }
            let hu = h as u32;
            let spread = hu ^ (hu >> 16);
            ((spread as usize) & (cap - 1), i, k)
        })
        .collect();
    idx.sort();
    idx.into_iter().map(|(_, _, k)| k.clone()).collect()
}

/// Re-key an object in Java `HashMap` iteration order.
pub fn java_ordered(m: &Map<String, Value>) -> Map<String, Value> {
    let keys: Vec<String> = m.keys().cloned().collect();
    let mut out = Map::new();
    for k in java_order(&keys) {
        if let Some(v) = m.get(&k) {
            out.insert(k, v.clone());
        }
    }
    out
}

/// `"return 1"` / `{"source": .., "lang": ..}` -> `{"source", "lang", "params"?}`.
pub fn script(v: &Value) -> Value {
    match v {
        Value::String(s) => json!({"source": s, "lang": "painless"}),
        Value::Object(o) => {
            let mut m = Map::new();
            if let Some(s) = o.get("source").or_else(|| o.get("inline")) {
                m.insert("source".into(), s.clone());
            }
            if let Some(i) = o.get("id") {
                m.insert("id".into(), i.clone());
            }
            if o.get("id").is_none() || o.get("lang").is_some() {
                m.insert("lang".into(), o.get("lang").cloned().unwrap_or(json!("painless")));
            }
            if let Some(p) = o.get("params") {
                m.insert("params".into(), p.clone());
            }
            Value::Object(m)
        }
        other => other.clone(),
    }
}

/// `IndicesOptions.fromXContent(parser, lenientExpandOpen())` followed by
/// `toXContent`: the full option set, or `None` when it equals the default
/// (ES then omits `indices_options`).
pub fn indices_options(io: &Value) -> Option<Value> {
    let o = io.as_object()?;
    let b = |k: &str, d: bool| match o.get(k) {
        Some(Value::Bool(x)) => *x,
        Some(Value::String(s)) => s == "true",
        _ => d,
    };
    let (mut open, mut closed, mut hidden) = (true, false, false);
    if let Some(ew) = o.get("expand_wildcards") {
        let states: Vec<String> = match ew {
            Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
            Value::String(s) => vec![s.clone()],
            _ => Vec::new(),
        };
        (open, closed, hidden) = (false, false, false);
        for st in states.iter().flat_map(|s| s.split(',')).map(str::trim) {
            match st {
                "open" => open = true,
                "closed" => closed = true,
                "hidden" => hidden = true,
                "all" => (open, closed, hidden) = (true, true, true),
                "none" => (open, closed, hidden) = (false, false, false),
                _ => {}
            }
        }
    }
    let ignore_unavailable = b("ignore_unavailable", true);
    let allow_no_indices = b("allow_no_indices", true);
    let ignore_throttled = b("ignore_throttled", false);
    if ignore_unavailable && allow_no_indices && !ignore_throttled && open && !closed && !hidden {
        return None;
    }
    let mut ew = Vec::new();
    for (on, n) in [(open, "open"), (closed, "closed"), (hidden, "hidden")] {
        if on {
            ew.push(n);
        }
    }
    Some(json!({"ignore_unavailable": ignore_unavailable, "expand_wildcards": ew, "allow_no_indices": allow_no_indices, "ignore_throttled": ignore_throttled}))
}

pub fn search_request(r: &Value) -> Value {
    let Some(o) = r.as_object() else { return r.clone() };
    let mut m = Map::new();
    m.insert("search_type".into(), o.get("search_type").cloned().unwrap_or(json!("query_then_fetch")));
    if let Some(i) = o.get("indices") {
        m.insert("indices".into(), match i {
            Value::String(s) => json!(s.split(',').collect::<Vec<_>>()),
            other => other.clone(),
        });
    }
    m.insert("rest_total_hits_as_int".into(), o.get("rest_total_hits_as_int").cloned().unwrap_or(json!(true)));
    if let Some(io) = o.get("indices_options").and_then(indices_options) {
        m.insert("indices_options".into(), io);
    }
    if let Some(b) = o.get("body") {
        m.insert("body".into(), b.clone());
    }
    if let Some(t) = o.get("template") {
        m.insert("template".into(), t.clone());
    }
    Value::Object(m)
}

/// Webhook / http input request template as ES echoes it.
pub fn http_request(r: &Value) -> Value {
    let Some(o) = r.as_object() else { return r.clone() };
    let mut scheme = o.get("scheme").and_then(Value::as_str).unwrap_or("http").to_string();
    let mut host = o.get("host").cloned();
    let mut port = o.get("port").cloned();
    let mut path = o.get("path").cloned();
    if let Some(Value::String(u)) = o.get("url") {
        if !u.contains("{{") {
            let (s, rest) = u.split_once("://").unwrap_or(("http", u));
            scheme = s.to_string();
            let (hp, p) = match rest.find('/') {
                Some(i) => (&rest[..i], &rest[i..]),
                None => (rest, "/"),
            };
            match hp.rsplit_once(':') {
                Some((h, pt)) if pt.parse::<u16>().is_ok() => {
                    host = Some(json!(h));
                    port = Some(json!(pt.parse::<u16>().unwrap()));
                }
                _ => host = Some(json!(hp)),
            }
            path = Some(json!(p));
        } else {
            return r.clone();
        }
    }
    let mut m = Map::new();
    m.insert("scheme".into(), json!(scheme));
    if let Some(h) = host {
        m.insert("host".into(), h);
    }
    m.insert("port".into(), port.unwrap_or(json!(if scheme == "https" { 443 } else { 80 })));
    m.insert("method".into(), json!(o.get("method").and_then(Value::as_str).unwrap_or("get").to_ascii_lowercase()));
    if let Some(p) = path {
        m.insert("path".into(), p);
    }
    m.insert("params".into(), o.get("params").cloned().unwrap_or(json!({})));
    m.insert("headers".into(), o.get("headers").cloned().unwrap_or(json!({})));
    for k in ["auth", "body", "connection_timeout", "read_timeout", "proxy"] {
        if let Some(v) = o.get(k) {
            m.insert(k.into(), v.clone());
        }
    }
    Value::Object(m)
}

/// `extract` is a Java `HashSet<String>`: iteration follows the hash order.
fn extract_order(v: &Value) -> Value {
    match v.as_array() {
        Some(a) if a.iter().all(Value::is_string) => {
            let keys: Vec<String> = a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
            json!(java_order(&keys))
        }
        _ => v.clone(),
    }
}

pub fn input(v: &Value) -> Value {
    let Some((k, d)) = v.as_object().and_then(|o| o.iter().next()) else { return v.clone() };
    match k.as_str() {
        "search" => {
            let mut m = Map::new();
            m.insert("request".into(), search_request(&d["request"]));
            for key in ["extract", "timeout"] {
                if let Some(x) = d.get(key) {
                    m.insert(key.into(), extract_order(x));
                }
            }
            json!({"search": m})
        }
        "http" => {
            let mut m = Map::new();
            m.insert("request".into(), http_request(&d["request"]));
            for key in ["response_content_type", "extract"] {
                if let Some(x) = d.get(key) {
                    m.insert(key.into(), extract_order(x));
                }
            }
            json!({"http": m})
        }
        "transform" => json!({"transform": transform(d)}),
        "chain" => {
            let inputs: Vec<Value> = d["inputs"].as_array().cloned().unwrap_or_default().iter().map(|sub| {
                let mut m = Map::new();
                if let Some(o) = sub.as_object() {
                    for (n, i) in o {
                        m.insert(n.clone(), input(i));
                    }
                }
                Value::Object(m)
            }).collect();
            json!({"chain": {"inputs": inputs}})
        }
        _ => v.clone(),
    }
}

pub fn condition(v: &Value) -> Value {
    match v.as_object().and_then(|o| o.iter().next()) {
        Some((k, d)) if k == "script" => json!({"script": script(d)}),
        _ => v.clone(),
    }
}

pub fn transform(v: &Value) -> Value {
    let Some((k, d)) = v.as_object().and_then(|o| o.iter().next()) else { return v.clone() };
    match k.as_str() {
        "script" => json!({"script": script(d)}),
        "search" => {
            let mut m = Map::new();
            m.insert("request".into(), search_request(&d["request"]));
            if let Some(t) = d.get("timeout") {
                m.insert("timeout".into(), t.clone());
            }
            json!({"search": m})
        }
        "chain" => json!({"chain": d.as_array().cloned().unwrap_or_default().iter().map(transform).collect::<Vec<_>>()}),
        _ => v.clone(),
    }
}

pub fn action(def: &Value) -> Value {
    let Some(o) = def.as_object() else { return def.clone() };
    let mut m = Map::new();
    for (k, v) in o {
        let nv = match k.as_str() {
            "logging" => {
                let mut l = Map::new();
                if let Some(c) = v.get("category") {
                    l.insert("category".into(), c.clone());
                }
                l.insert("level".into(), v.get("level").cloned().unwrap_or(json!("info")));
                if let Some(t) = v.get("text") {
                    l.insert("text".into(), t.clone());
                }
                Value::Object(l)
            }
            "webhook" => http_request(v),
            "condition" => condition(v),
            "transform" => transform(v),
            _ => v.clone(),
        };
        m.insert(k.clone(), nv);
    }
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_hashmap_order() {
        let k: Vec<String> = ["idx", "lg", "wh"].iter().map(|s| s.to_string()).collect();
        assert_eq!(java_order(&k), vec!["wh", "lg", "idx"]);
    }

    #[test]
    fn script_shorthand() {
        assert_eq!(script(&json!("return 1")), json!({"source": "return 1", "lang": "painless"}));
    }
}
