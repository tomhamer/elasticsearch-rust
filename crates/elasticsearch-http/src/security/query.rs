//! The small query DSL ES accepts for `_security/_query/{user,role,api_key}`
//! (bool, term(s), prefix, wildcard, range, exists, ids, match,
//! simple_query_string, match_all), evaluated over in-memory documents.

use elasticsearch_engine::error::EsError;
use elasticsearch_engine::security::privilege::glob;
use serde_json::Value;
use std::cmp::Ordering;

fn unsupported(kind: &str) -> EsError {
    EsError::illegal_argument(format!("Query type [{kind}] is not currently supported in this context"))
}

fn get<'a>(doc: &'a Value, path: &str) -> Vec<&'a Value> {
    let mut cur = vec![doc];
    for part in path.split('.') {
        let mut next = Vec::new();
        for c in cur {
            match c {
                Value::Object(o) => {
                    if let Some(v) = o.get(part) {
                        next.push(v);
                    }
                }
                Value::Array(a) => {
                    for x in a {
                        if let Some(v) = x.get(part) {
                            next.push(v);
                        }
                    }
                }
                _ => {}
            }
        }
        cur = next;
    }
    let mut out = Vec::new();
    for v in cur {
        match v {
            Value::Array(a) => out.extend(a.iter()),
            other => out.push(other),
        }
    }
    out
}

fn as_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn date_math(s: &str) -> Option<f64> {
    if let Ok(n) = s.parse::<f64>() {
        return Some(n);
    }
    if let Some(rest) = s.strip_prefix("now") {
        let now = chrono::Utc::now().timestamp_millis() as f64;
        if rest.is_empty() {
            return Some(now);
        }
        let rest = rest.split('/').next().unwrap_or("");
        let (sign, num) = if let Some(r) = rest.strip_prefix('-') { (-1.0, r) } else if let Some(r) = rest.strip_prefix('+') { (1.0, r) } else { return None };
        let ms = elasticsearch_engine::security::parse_time_ms(num, "range").ok()? as f64;
        return Some(now + sign * ms);
    }
    chrono::DateTime::parse_from_rfc3339(s).ok().map(|d| d.timestamp_millis() as f64).or_else(|| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok().map(|d| d.and_hms_opt(0, 0, 0).map(|t| t.and_utc().timestamp_millis() as f64).unwrap_or(0.0)))
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => date_math(s),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn eq(a: &Value, b: &Value, ci: bool) -> bool {
    match (a, b) {
        (Value::String(x), Value::String(y)) => if ci { x.eq_ignore_ascii_case(y) } else { x == y },
        (Value::Bool(x), Value::String(y)) | (Value::String(y), Value::Bool(x)) => y == &x.to_string(),
        (Value::Number(_), _) | (_, Value::Number(_)) => num(a).zip(num(b)).map(|(x, y)| x == y).unwrap_or(false),
        _ => a == b,
    }
}

/// `field` -> the document path to read (`map` renames user-facing fields).
pub fn matches(doc: &Value, q: &Value, map: &dyn Fn(&str) -> String) -> Result<bool, EsError> {
    let Some((kind, spec)) = q.as_object().and_then(|o| o.iter().next()) else { return Ok(true) };
    let field_spec = || -> Option<(String, &Value)> { spec.as_object().and_then(|o| o.iter().next()).map(|(f, v)| (map(f), v)) };
    Ok(match kind.as_str() {
        "match_all" => true,
        "match_none" => false,
        "bool" => {
            let list = |k: &str| -> Vec<&Value> {
                match &spec[k] {
                    Value::Array(a) => a.iter().collect(),
                    Value::Null => vec![],
                    o => vec![o],
                }
            };
            for c in list("must").into_iter().chain(list("filter")) {
                if !matches(doc, c, map)? {
                    return Ok(false);
                }
            }
            for c in list("must_not") {
                if matches(doc, c, map)? {
                    return Ok(false);
                }
            }
            let should = list("should");
            if !should.is_empty() {
                let mut hits = 0;
                for c in &should {
                    if matches(doc, c, map)? {
                        hits += 1;
                    }
                }
                let has_required = !list("must").is_empty() || !list("filter").is_empty();
                let msm = spec["minimum_should_match"].as_u64().or_else(|| spec["minimum_should_match"].as_str().and_then(|s| s.parse().ok())).unwrap_or(if has_required { 0 } else { 1 });
                if hits < msm as usize {
                    return Ok(false);
                }
            }
            true
        }
        "term" => {
            let Some((f, v)) = field_spec() else { return Ok(false) };
            let (val, ci) = match v {
                Value::Object(o) => (o.get("value").cloned().unwrap_or(Value::Null), o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false)),
                other => (other.clone(), false),
            };
            get(doc, &f).into_iter().any(|x| eq(x, &val, ci))
        }
        "terms" => {
            let Some((f, v)) = spec.as_object().and_then(|o| o.iter().find(|(k, _)| k.as_str() != "boost")).map(|(f, v)| (map(f), v)) else { return Ok(false) };
            let vals: Vec<Value> = v.as_array().cloned().unwrap_or_default();
            get(doc, &f).into_iter().any(|x| vals.iter().any(|y| eq(x, y, false)))
        }
        "ids" => {
            let vals: Vec<String> = spec["values"].as_array().map(|a| a.iter().map(as_str).collect()).unwrap_or_default();
            get(doc, &map("_id")).into_iter().any(|x| vals.contains(&as_str(x)))
        }
        "prefix" => {
            let Some((f, v)) = field_spec() else { return Ok(false) };
            let (val, ci) = match v {
                Value::Object(o) => (o.get("value").map(as_str).unwrap_or_default(), o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false)),
                other => (as_str(other), false),
            };
            get(doc, &f).into_iter().any(|x| {
                let s = as_str(x);
                if ci { s.to_lowercase().starts_with(&val.to_lowercase()) } else { s.starts_with(&val) }
            })
        }
        "wildcard" => {
            let Some((f, v)) = field_spec() else { return Ok(false) };
            let (val, ci) = match v {
                Value::Object(o) => (o.get("value").or_else(|| o.get("wildcard")).map(as_str).unwrap_or_default(), o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false)),
                other => (as_str(other), false),
            };
            get(doc, &f).into_iter().any(|x| {
                let s = as_str(x);
                if ci { glob(&val.to_lowercase(), &s.to_lowercase()) } else { glob(&val, &s) }
            })
        }
        "exists" => {
            let f = map(spec["field"].as_str().unwrap_or(""));
            get(doc, &f).into_iter().any(|x| !x.is_null())
        }
        "range" => {
            let Some((f, v)) = field_spec() else { return Ok(false) };
            let b = |k: &str| v.get(k).and_then(num);
            get(doc, &f).into_iter().filter_map(num).any(|x| {
                b("gt").map(|y| x > y).unwrap_or(true)
                    && b("gte").or_else(|| b("from")).map(|y| x >= y).unwrap_or(true)
                    && b("lt").map(|y| x < y).unwrap_or(true)
                    && b("lte").or_else(|| b("to")).map(|y| x <= y).unwrap_or(true)
            })
        }
        "match" | "match_phrase" | "match_bool_prefix" => {
            let Some((f, v)) = field_spec() else { return Ok(false) };
            let text = match v {
                Value::Object(o) => o.get("query").map(as_str).unwrap_or_default(),
                other => as_str(other),
            };
            let toks: Vec<String> = text.split_whitespace().map(|t| t.to_lowercase()).collect();
            get(doc, &f).into_iter().any(|x| {
                let s = as_str(x).to_lowercase();
                let words: Vec<&str> = s.split(|c: char| !c.is_alphanumeric()).collect();
                toks.iter().any(|t| words.contains(&t.as_str()) || s == *t)
            })
        }
        "simple_query_string" | "query_string" => {
            let text = spec["query"].as_str().unwrap_or("").to_lowercase();
            let fields: Vec<String> = spec["fields"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(|s| map(s)).collect()).unwrap_or_default();
            let toks: Vec<&str> = text.split_whitespace().collect();
            let vals: Vec<String> = if fields.is_empty() {
                let mut out = Vec::new();
                collect_strings(doc, &mut out);
                out
            } else {
                fields.iter().flat_map(|f| get(doc, f).into_iter().map(|v| as_str(v).to_lowercase())).collect()
            };
            toks.iter().any(|t| vals.iter().any(|v| if t.contains('*') { glob(t, v) } else { v.split(|c: char| !c.is_alphanumeric()).any(|w| w == *t) || v == t }))
        }
        other => return Err(unsupported(other)),
    })
}

fn collect_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.to_lowercase()),
        Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        Value::Object(o) => o.values().for_each(|x| collect_strings(x, out)),
        _ => {}
    }
}

fn cmp_vals(a: &Value, b: &Value) -> Ordering {
    match (num(a).filter(|_| a.is_number() || a.is_boolean()), num(b).filter(|_| b.is_number() || b.is_boolean())) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
        _ => match (a, b) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Greater,
            (_, Value::Null) => Ordering::Less,
            _ => as_str(a).cmp(&as_str(b)),
        },
    }
}

pub struct Page {
    pub total: usize,
    pub hits: Vec<(Value, Option<Vec<Value>>)>,
    pub aggregations: Option<Value>,
}

/// Group `docs` by the (first) value(s) of `field`, sorted by key.
fn group_by<'a>(docs: &[&'a Value], field: &str) -> Vec<(Value, Vec<&'a Value>)> {
    let mut groups: Vec<(Value, Vec<&'a Value>)> = Vec::new();
    for d in docs.iter().copied() {
        for v in get(d, field) {
            match groups.iter_mut().find(|(k, _)| k == v) {
                Some((_, g)) => {
                    if !g.iter().any(|x| std::ptr::eq(*x, d)) {
                        g.push(d)
                    }
                }
                None => groups.push((v.clone(), vec![d])),
            }
        }
    }
    groups.sort_by(|a, b| cmp_vals(&a.0, &b.0));
    groups
}

/// The small aggregation set ES allows over security documents: terms,
/// composite (terms sources), filter, filters, missing, value_count,
/// cardinality, with sub-aggregations.
pub fn aggregate(docs: &[&Value], spec: &Value, map: &dyn Fn(&str) -> String) -> Result<Value, EsError> {
    let mut out = serde_json::Map::new();
    let Some(spec) = spec.as_object() else { return Ok(Value::Object(out)) };
    for (name, def) in spec {
        let sub = if def["aggs"].is_null() { &def["aggregations"] } else { &def["aggs"] };
        let with_sub = |mut b: serde_json::Map<String, Value>, ds: &[&Value]| -> Result<Value, EsError> {
            b.insert("doc_count".into(), Value::from(ds.len()));
            if let Some(o) = aggregate(ds, sub, map)?.as_object() {
                b.extend(o.clone());
            }
            Ok(Value::Object(b))
        };
        let kind = def.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")).cloned()).unwrap_or_default();
        let body = &def[kind.as_str()];
        let res = match kind.as_str() {
            "terms" => {
                let field = map(body["field"].as_str().unwrap_or(""));
                let size = body["size"].as_u64().unwrap_or(10) as usize;
                let mut groups = group_by(docs, &field);
                groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| cmp_vals(&a.0, &b.0)));
                let mut buckets = Vec::new();
                for (k, ds) in groups.into_iter().take(size) {
                    let mut b = serde_json::Map::new();
                    b.insert("key".into(), k);
                    buckets.push(with_sub(b, &ds)?);
                }
                serde_json::json!({"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": buckets})
            }
            "composite" => {
                let size = body["size"].as_u64().unwrap_or(10) as usize;
                let mut srcs: Vec<(String, String)> = Vec::new();
                for s in body["sources"].as_array().into_iter().flatten() {
                    for (n, v) in s.as_object().into_iter().flatten() {
                        let f = v["terms"]["field"].as_str().ok_or_else(|| EsError::illegal_argument("composite sources other than [terms] are not supported"))?;
                        srcs.push((n.clone(), map(f)));
                    }
                }
                let mut groups: Vec<(Vec<Value>, Vec<&Value>)> = Vec::new();
                for d in docs.iter().copied() {
                    let mut combos: Vec<Vec<Value>> = vec![vec![]];
                    for (_, f) in &srcs {
                        let vals = get(d, f);
                        if vals.is_empty() {
                            combos.clear();
                            break;
                        }
                        combos = combos.into_iter().flat_map(|c| vals.iter().map(move |v| { let mut c = c.clone(); c.push((*v).clone()); c })).collect();
                    }
                    for c in combos {
                        match groups.iter_mut().find(|(k, _)| *k == c) {
                            Some((_, g)) => g.push(d),
                            None => groups.push((c, vec![d])),
                        }
                    }
                }
                groups.sort_by(|a, b| a.0.iter().zip(&b.0).map(|(x, y)| cmp_vals(x, y)).find(|c| *c != Ordering::Equal).unwrap_or(Ordering::Equal));
                if let Some(after) = body["after"].as_object() {
                    let ak: Vec<Value> = srcs.iter().map(|(n, _)| after.get(n).cloned().unwrap_or(Value::Null)).collect();
                    groups.retain(|(k, _)| k.iter().zip(&ak).map(|(x, y)| cmp_vals(x, y)).find(|c| *c != Ordering::Equal) == Some(Ordering::Greater));
                }
                groups.truncate(size);
                let key_obj = |k: &[Value]| -> Value { Value::Object(srcs.iter().zip(k).map(|((n, _), v)| (n.clone(), v.clone())).collect()) };
                let mut buckets = Vec::new();
                for (k, ds) in &groups {
                    let mut b = serde_json::Map::new();
                    b.insert("key".into(), key_obj(k));
                    buckets.push(with_sub(b, ds)?);
                }
                let mut r = serde_json::Map::new();
                if let Some((k, _)) = groups.last() {
                    r.insert("after_key".into(), key_obj(k));
                }
                r.insert("buckets".into(), Value::Array(buckets));
                Value::Object(r)
            }
            "filter" => {
                let mut ds = Vec::new();
                for d in docs {
                    if matches(d, body, map)? {
                        ds.push(*d);
                    }
                }
                with_sub(serde_json::Map::new(), &ds)?
            }
            "filters" => {
                let mut run_one = |q: &Value| -> Result<Value, EsError> {
                    let mut ds = Vec::new();
                    for d in docs {
                        if matches(d, q, map)? {
                            ds.push(*d);
                        }
                    }
                    with_sub(serde_json::Map::new(), &ds)
                };
                match &body["filters"] {
                    Value::Object(o) => {
                        let mut b = serde_json::Map::new();
                        for (k, q) in o {
                            b.insert(k.clone(), run_one(q)?);
                        }
                        serde_json::json!({"buckets": b})
                    }
                    Value::Array(a) => {
                        let mut b = Vec::new();
                        for q in a {
                            b.push(run_one(q)?);
                        }
                        serde_json::json!({"buckets": b})
                    }
                    _ => return Err(EsError::illegal_argument("[filters] requires [filters]")),
                }
            }
            "missing" => {
                let f = map(body["field"].as_str().unwrap_or(""));
                let ds: Vec<&Value> = docs.iter().copied().filter(|d| get(d, &f).is_empty()).collect();
                with_sub(serde_json::Map::new(), &ds)?
            }
            "value_count" => {
                let f = map(body["field"].as_str().unwrap_or(""));
                serde_json::json!({"value": docs.iter().map(|d| get(d, &f).len()).sum::<usize>()})
            }
            "cardinality" => {
                let f = map(body["field"].as_str().unwrap_or(""));
                serde_json::json!({"value": group_by(docs, &f).len()})
            }
            other => return Err(EsError::illegal_argument(format!("Aggregation type [{other}] is not currently supported in this context"))),
        };
        out.insert(name.clone(), res);
    }
    Ok(Value::Object(out))
}

/// Filter, sort and page `docs` per a `_query/*` request body.
pub fn run(docs: Vec<Value>, body: &Value, map: &dyn Fn(&str) -> String) -> Result<Page, EsError> {
    let mut hits = Vec::new();
    let q = &body["query"];
    for d in docs {
        if q.is_null() || matches(&d, q, map)? {
            hits.push(d);
        }
    }
    let total = hits.len();
    let aggregations = match (&body["aggs"], &body["aggregations"]) {
        (Value::Null, Value::Null) => None,
        (a, Value::Null) => Some(aggregate(&hits.iter().collect::<Vec<_>>(), a, map)?),
        (_, a) => Some(aggregate(&hits.iter().collect::<Vec<_>>(), a, map)?),
    };
    let mut sort: Vec<(String, bool)> = Vec::new();
    let sort_spec = match &body["sort"] {
        Value::Array(a) => a.clone(),
        Value::Null => vec![],
        o => vec![o.clone()],
    };
    for s in &sort_spec {
        match s {
            Value::String(f) => sort.push((f.clone(), false)),
            Value::Object(o) => {
                for (f, v) in o {
                    let desc = match v {
                        Value::String(d) => d == "desc",
                        Value::Object(m) => m.get("order").and_then(Value::as_str) == Some("desc"),
                        _ => false,
                    };
                    sort.push((f.clone(), desc));
                }
            }
            _ => {}
        }
    }
    let key = |d: &Value| -> Vec<Value> { sort.iter().map(|(f, _)| get(d, &map(f)).first().map(|v| (*v).clone()).unwrap_or(Value::Null)).collect() };
    if !sort.is_empty() {
        hits.sort_by(|a, b| {
            let (ka, kb) = (key(a), key(b));
            for (i, (_, desc)) in sort.iter().enumerate() {
                let c = cmp_vals(&ka[i], &kb[i]);
                if c != Ordering::Equal {
                    return if *desc { c.reverse() } else { c };
                }
            }
            Ordering::Equal
        });
    }
    if let Some(after) = body["search_after"].as_array() {
        hits.retain(|d| {
            let k = key(d);
            for (i, (_, desc)) in sort.iter().enumerate() {
                let c = cmp_vals(&k[i], after.get(i).unwrap_or(&Value::Null));
                if c != Ordering::Equal {
                    return if *desc { c == Ordering::Less } else { c == Ordering::Greater };
                }
            }
            false
        });
    }
    let from = body["from"].as_u64().unwrap_or(0) as usize;
    let size = body["size"].as_u64().unwrap_or(10) as usize;
    let page: Vec<(Value, Option<Vec<Value>>)> = hits
        .into_iter()
        .skip(from)
        .take(size)
        .map(|d| {
            let k = if sort.is_empty() { None } else { Some(key(&d)) };
            (d, k)
        })
        .collect();
    Ok(Page { total, hits: page, aggregations })
}
