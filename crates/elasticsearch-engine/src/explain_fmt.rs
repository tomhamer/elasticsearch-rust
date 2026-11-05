//! Rewrite core [`Explanation`] trees into the exact wording Lucene / ES
//! produce (`_explain`, `explain: true`): field names instead of core ids,
//! `weight(f:t in d) [PerFieldSimilarity], result of:`, BM25's
//! `score(freq=1.0), computed as boost * idf * tf from:` /
//! `idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:`, `max of:`
//! for dis_max without tie breaker, filter clauses, query boosts folded
//! into the term weight's `boost` leaf, and ES's function_score wording.

use crate::values;
use rustic_core::query::Explanation;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Java `Float.toString` (`1.0`, `0.5`, `1.0E-4`).
fn java_float(x: f32) -> String {
    let s = format!("{x}");
    if s.contains('.') || s.contains('e') || !x.is_finite() {
        s
    } else {
        format!("{s}.0")
    }
}

fn rename(d: &str, names: &HashMap<u16, String>) -> String {
    let mut out = String::with_capacity(d.len());
    let b = d.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if d[i..].starts_with("field") && (i == 0 || !(b[i - 1] as char).is_ascii_alphanumeric()) {
            let digits: String = d[i + 5..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() && d[i + 5 + digits.len()..].starts_with(':') {
                if let Some(n) = digits.parse::<u16>().ok().and_then(|id| names.get(&id)) {
                    out.push_str(n);
                    i += 5 + digits.len();
                    continue;
                }
            }
        }
        let ch = d[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn describe(d: &str) -> String {
    if d.starts_with("weight(") {
        if let Some(p) = d.rfind(" [") {
            if d.ends_with(']') {
                return format!("{} [PerFieldSimilarity], result of:", &d[..p]);
            }
        }
        return d.to_string();
    }
    if let Some(rest) = d.strip_prefix("score(freq=") {
        if let Some(close) = rest.find(')') {
            let freq = rest[..close].parse::<f32>().map(java_float).unwrap_or_else(|_| rest[..close].to_string());
            let tail = &rest[close..];
            let tail = if tail.ends_with("from:") { tail.to_string() } else { format!("{tail} from:") };
            return format!("score(freq={freq}{tail}");
        }
    }
    match d {
        "idf, computed as ln(1 + (N - n + 0.5) / (n + 0.5))" | "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5))" => {
            return "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:".to_string();
        }
        "N, total number of documents" => return "N, total number of documents with field".to_string(),
        "function score, boost_mode Multiply" => return "function score, product of:".to_string(),
        "function score, boost_mode Sum" => return "function score, sum of:".to_string(),
        "function score, boost_mode Avg" => return "function score, avg of:".to_string(),
        "function score, boost_mode Max" => return "function score, max of:".to_string(),
        "function score, boost_mode Min" => return "function score, min of:".to_string(),
        "function score, boost_mode Replace" => return "function score, score mode [replace]".to_string(),
        _ => {}
    }
    if let Some(rest) = d.strip_prefix("max plus ") {
        if let Some(t) = rest.strip_suffix(" times others of:") {
            let tie: f32 = t.parse().unwrap_or(0.0);
            return if tie == 0.0 { "max of:".to_string() } else { format!("max plus {} times others of:", java_float(tie)) };
        }
    }
    d.to_string()
}

/// Multiply the `boost` leaf of a `weight(...)` node's score by `b`.
fn fold_boost(mut w: Explanation, b: f32, value: f32) -> Explanation {
    w.value = value;
    for s in &mut w.details {
        if s.description.starts_with("score(") {
            s.value = value;
            for leaf in &mut s.details {
                if leaf.description == "boost" {
                    leaf.value *= b;
                }
            }
        }
    }
    w
}

/// The Lucene-worded copy of `e`.
pub fn to_lucene(e: &Explanation, names: &HashMap<u16, String>) -> Explanation {
    let details: Vec<Explanation> = e.details.iter().map(|c| to_lucene(c, names)).collect();
    let desc = rename(&e.description, names);
    // Boost on a term weight: Lucene folds it into the weight's boost.
    if desc == "product of:" && details.len() == 2 && details[1].description == "boost" && details[0].description.starts_with("weight(") {
        let b = details[1].value;
        let mut it = details.into_iter();
        return fold_boost(it.next().unwrap(), b, e.value);
    }
    if desc == "match on required clause, product of: (filter, score ignored)" {
        let mut d = vec![Explanation::leaf(0.0, "# clause")];
        d.extend(details);
        return Explanation::new(0.0, "match on required clause, product of:", d);
    }
    if desc.starts_with("functions combined with score_mode") {
        // ES: min(function value, maxBoost)
        let max_boost = desc.rsplit(' ').next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(f32::MAX);
        let inner = if details.len() == 1 {
            details.into_iter().next().unwrap()
        } else {
            let mode = desc.split_whitespace().nth(4).unwrap_or("Multiply").to_lowercase();
            Explanation::new(e.value, format!("function score, score mode [{mode}]"), details)
        };
        return Explanation::new(e.value, "min of:", vec![inner, Explanation::leaf(max_boost, "maxBoost")]);
    }
    Explanation::new(e.value, describe(&desc), details)
}

/// The JSON number of an explanation value: Lucene's `Explanation.getValue()` is a
/// `Float` rendered with its shortest round-trip digits (`0.7361701`, not the
/// widened double); document counts (`n`, `N`) are integers.
pub fn value_json(e: &Explanation) -> Value {
    let d = &e.description;
    if (d.starts_with("n, number of documents") || d.starts_with("N, total number of documents")) && e.value.fract() == 0.0 && e.value.is_finite() {
        return json!(e.value as i64);
    }
    if !e.value.is_finite() {
        return values::java_double(e.value as f64);
    }
    let shortest: f64 = format!("{}", e.value).parse().unwrap_or(e.value as f64);
    values::java_double(shortest)
}

/// Response JSON of an explanation tree (already Lucene-worded).
pub fn to_json(e: &Explanation) -> Value {
    json!({
        "value": value_json(e),
        "description": e.description,
        "details": e.details.iter().map(to_json).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_wording() {
        let names: HashMap<u16, String> = [(3u16, "title".to_string())].into_iter().collect();
        let score = Explanation::new(
            0.5,
            "score(freq=1), computed as boost * idf * tf",
            vec![
                Explanation::leaf(1.0, "boost"),
                Explanation::new(0.3, "idf, computed as ln(1 + (N - n + 0.5) / (n + 0.5))", vec![Explanation::leaf(1.0, "N, total number of documents")]),
            ],
        );
        let w = Explanation::new(0.5, "weight(field3:fox in 0) [BM25]", vec![score]);
        let boosted = Explanation::new(1.0, "product of:", vec![w, Explanation::leaf(2.0, "boost")]);
        let dm = Explanation::new(1.0, "max plus 0 times others of:", vec![boosted]);
        let out = to_lucene(&dm, &names);
        assert_eq!(out.description, "max of:");
        let w = &out.details[0];
        assert_eq!(w.description, "weight(title:fox in 0) [PerFieldSimilarity], result of:");
        assert_eq!(w.value, 1.0);
        assert_eq!(w.details[0].description, "score(freq=1.0), computed as boost * idf * tf from:");
        assert_eq!(w.details[0].details[0].value, 2.0);
        assert_eq!(w.details[0].details[1].description, "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:");
        assert_eq!(w.details[0].details[1].details[0].description, "N, total number of documents with field");
        assert_eq!(java_float(0.3), "0.3");
    }

    /// Captured from ES 8.19.22: `{"match": {"t": "hello"}}` explain on a 4-doc index.
    #[test]
    fn values_render_like_lucene_floats() {
        let boost = Explanation::leaf(2.2f32, "boost");
        assert_eq!(serde_json::to_string(&value_json(&boost)).unwrap(), "2.2");
        let idf = Explanation::leaf(0.6931472f32, "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:");
        assert_eq!(serde_json::to_string(&value_json(&idf)).unwrap(), "0.6931472");
        let tf = Explanation::leaf(0.48275858f32, "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:");
        assert_eq!(serde_json::to_string(&value_json(&tf)).unwrap(), "0.48275858");
        let k1 = Explanation::leaf(1.2f32, "k1, term saturation parameter");
        assert_eq!(serde_json::to_string(&value_json(&k1)).unwrap(), "1.2");
        // document counts are integers, the rest stay floats
        assert_eq!(serde_json::to_string(&value_json(&Explanation::leaf(2.0, "n, number of documents containing term"))).unwrap(), "2");
        assert_eq!(serde_json::to_string(&value_json(&Explanation::leaf(4.0, "N, total number of documents with field"))).unwrap(), "4");
        assert_eq!(serde_json::to_string(&value_json(&Explanation::leaf(3.0, "dl, length of field"))).unwrap(), "3.0");
    }
}
