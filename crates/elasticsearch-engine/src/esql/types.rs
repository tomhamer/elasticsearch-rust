//! Values of the structured ES|QL types: `aggregate_metric_double`,
//! `histogram`, `flattened` (carried as their JSON text).

use super::{fmt_double, Val};
use serde_json::{Map, Value};

/// An `aggregate_metric_double` value: any of its four sub-metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Amd {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub sum: Option<f64>,
    pub count: Option<i64>,
}

impl Amd {
    pub fn from_json(m: &Map<String, Value>) -> Amd {
        let f = |k: &str| m.get(k).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
        Amd { min: f("min"), max: f("max"), sum: f("sum"), count: f("value_count").map(|c| c as i64) }
    }

    /// Parse the JSON text of a value.
    pub fn parse(s: &str) -> Option<Amd> {
        match serde_json::from_str::<Value>(s).ok()? {
            Value::Object(m) => Some(Amd::from_json(&m)),
            _ => None,
        }
    }

    /// From one number (or the values of a multi-value).
    pub fn of_values(vals: &[f64]) -> Option<Amd> {
        if vals.is_empty() {
            return None;
        }
        Some(Amd {
            min: vals.iter().cloned().reduce(f64::min),
            max: vals.iter().cloned().reduce(f64::max),
            sum: Some(vals.iter().sum()),
            count: Some(vals.len() as i64),
        })
    }
}

/// `{"min":..,"max":..,"sum":..,"value_count":..}` with the present sub-metrics.
pub fn amd_json(a: &Amd) -> String {
    let mut parts = Vec::new();
    if let Some(v) = a.min {
        parts.push(format!("\"min\":{}", fmt_double(v)));
    }
    if let Some(v) = a.max {
        parts.push(format!("\"max\":{}", fmt_double(v)));
    }
    if let Some(v) = a.sum {
        parts.push(format!("\"sum\":{}", fmt_double(v)));
    }
    if let Some(v) = a.count {
        parts.push(format!("\"value_count\":{v}"));
    }
    format!("{{{}}}", parts.join(","))
}

/// The `aggregate_metric_double` value of a cell, if it is one.
pub fn amd_of(v: &Val) -> Option<Amd> {
    match v {
        Val::Str(s) => Amd::parse(s),
        _ => None,
    }
}

/// `{"values":[..],"counts":[..]}` of a histogram field value.
pub fn histogram_json(m: &Map<String, Value>) -> String {
    let nums = |k: &str| -> Vec<String> {
        m.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).map(|d| if k == "counts" { format!("{}", d as i64) } else { fmt_json_double(d) }).collect()).unwrap_or_default()
    };
    format!("{{\"values\":[{}],\"counts\":[{}]}}", nums("values").join(","), nums("counts").join(","))
}

/// A double as Jackson writes it (`0.1`, `1.0E7`).
fn fmt_json_double(d: f64) -> String {
    fmt_double(d)
}

/// A flattened field value: its leaves keyed by dotted path (sorted), all
/// values as strings.
pub fn flattened_json(v: &Value) -> String {
    // Every scalar leaf with its dotted path: object arrays and nested
    // arrays are walked through (a flattened field keys them all), and a
    // path repeated across array elements keeps every value, in order.
    let mut leaves: Vec<(String, Value)> = Vec::new();
    fn walk(prefix: &str, v: &Value, out: &mut Vec<(String, Value)>) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    walk(&p, x, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(prefix, x, out)),
            Value::Null => {}
            other => out.push((prefix.to_string(), leaf(other))),
        }
    }
    fn leaf(v: &Value) -> Value {
        match v {
            Value::String(s) => Value::String(s.clone()),
            other => Value::String(other.to_string()),
        }
    }
    walk("", v, &mut leaves);
    let mut grouped: std::collections::BTreeMap<String, Vec<Value>> = std::collections::BTreeMap::new();
    for (k, x) in leaves {
        grouped.entry(k).or_default().push(x);
    }
    let m: Map<String, Value> = grouped.into_iter().map(|(k, mut vs)| (k, if vs.len() == 1 { vs.pop().unwrap() } else { Value::Array(vs) })).collect();
    Value::Object(m).to_string()
}

/// A `_tsid` value: URL-safe base64 (no padding) of a hash of the series'
/// dimensions, like ES's binary time series ids.
pub fn tsid_encode(key: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut bytes = Vec::with_capacity(36);
    for seed in 0u64..4 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut h);
        key.hash(&mut h);
        bytes.extend_from_slice(&h.finish().to_be_bytes());
    }
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for k in 0..=chunk.len() {
            out.push(A[((b >> (18 - 6 * k)) & 63) as usize] as char);
        }
    }
    out
}

#[cfg(test)]
mod flattened_b7_tests {
    use super::*;
    use serde_json::json;

    /// Review finding 600: repeated leaf paths across object arrays keep
    /// every value and nested arrays are walked, so FIELD_EXTRACT sees all
    /// of them (sorted, deduplicated).
    #[test]
    fn flattened_reconstruction_keeps_repeated_and_nested_values() {
        let j = flattened_json(&json!({"items": [{"tag": "red"}, {"tag": "blue"}], "tags": [["red", "blue"]], "one": {"k": 1}}));
        let v: Value = serde_json::from_str(&j).unwrap();
        assert_eq!(v, json!({"items.tag": ["red", "blue"], "one.k": "1", "tags": ["red", "blue"]}));
        let got = super::super::func2::field_extract(&j, "items.tag").unwrap();
        assert_eq!(got, super::super::Val::Multi(vec![super::super::Val::Str("blue".into()), super::super::Val::Str("red".into())]));
        let got = super::super::func2::field_extract(&j, "tags").unwrap();
        assert_eq!(got, super::super::Val::Multi(vec![super::super::Val::Str("blue".into()), super::super::Val::Str("red".into())]));
    }
}
