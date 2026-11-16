//! Inference of `tree_ensemble` trained models (ES `TreeInferenceModel` /
//! `EnsembleInferenceModel`): a `tree` or an `ensemble` of trees with a
//! `weighted_sum`, `exponent`, `logistic_regression` or `weighted_mode`
//! output aggregator, for `regression` and `classification` targets.
//!
//! Features are read by name from the document; a missing or non-numeric
//! feature takes the node's `default_left` branch (ES `isMissing`). What
//! this evaluator does not implement (feature preprocessors, feature
//! importance) is refused with an error: an accepted model never answers
//! with an empty prediction (finding 157).

use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

fn bad(msg: impl Into<String>) -> EsError {
    EsError::new(400, "status_exception", msg)
}

fn floats(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(Value::as_f64).collect()
}

/// The feature value of `name` in `doc` (dotted names reach into objects);
/// NaN when missing or not numeric.
fn feature(doc: &Value, name: &str) -> f64 {
    let v = doc.get(name).or_else(|| name.split('.').try_fold(doc, |cur, p| cur.get(p)));
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(Value::Bool(b)) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => f64::NAN,
    }
}

/// The leaf value a tree gives `doc`.
fn leaf(tree: &Value, doc: &Value) -> Result<Vec<f64>> {
    let names: Vec<&str> = tree["feature_names"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let nodes = tree["tree_structure"].as_array().ok_or_else(|| bad("tree has no [tree_structure]"))?;
    let by_index = |i: usize| nodes.iter().find(|n| n["node_index"].as_u64() == Some(i as u64)).or_else(|| nodes.get(i));
    let mut node = by_index(0).ok_or_else(|| bad("tree has no root node"))?;
    for _ in 0..=nodes.len() {
        if let Some(v) = node.get("leaf_value") {
            return match v {
                Value::Number(n) => Ok(vec![n.as_f64().unwrap_or(0.0)]),
                other => floats(other).filter(|f| !f.is_empty()).ok_or_else(|| bad("tree leaf has no numeric [leaf_value]")),
            };
        }
        let split = node["split_feature"].as_u64().ok_or_else(|| bad("tree node has no [split_feature]"))? as usize;
        let name = names.get(split).ok_or_else(|| bad(format!("feature index [{split}] is out of bounds for the [feature_names] array")))?;
        let x = feature(doc, name);
        let threshold = node["threshold"].as_f64().unwrap_or(0.0);
        let left = if x.is_nan() {
            node["default_left"].as_bool().unwrap_or(false)
        } else {
            match node["decision_type"].as_str().unwrap_or("lte") {
                "lt" => x < threshold,
                "gt" => x > threshold,
                "gte" => x >= threshold,
                _ => x <= threshold,
            }
        };
        let child = node[if left { "left_child" } else { "right_child" }].as_u64().ok_or_else(|| bad("tree node has no child"))? as usize;
        node = by_index(child).ok_or_else(|| bad(format!("tree node [{child}] is missing")))?;
    }
    Err(bad("tree has a cycle"))
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| if x.is_finite() { (x - max).exp() } else { 0.0 }).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A classification tree's class probabilities from its leaf
/// (`TreeInferenceModel.classificationProbability`).
fn tree_probabilities(tree: &Value, value: &[f64]) -> Result<Vec<f64>> {
    if value.len() > 1 {
        return Ok(softmax(value));
    }
    let max_category = tree["tree_structure"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| n.get("leaf_value"))
        .filter_map(|v| v.as_f64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_f64)))
        .fold(0.0, f64::max);
    let class = value[0];
    if class < 0.0 || class.fract() != 0.0 || max_category.fract() != 0.0 {
        return Err(bad("classification tree leaf values must be whole class indices"));
    }
    let mut p = vec![0.0; max_category as usize + 1];
    p[class as usize] = 1.0;
    Ok(p)
}

/// The aggregator name and weights of an ensemble.
fn aggregator(e: &Value) -> (String, Option<Vec<f64>>) {
    match e.get("aggregate_output").and_then(Value::as_object).and_then(|o| o.iter().next()) {
        Some((name, cfg)) => (name.clone(), cfg.get("weights").and_then(floats)),
        None => ("weighted_sum".into(), None),
    }
}

/// Probabilities (classification) or the value (regression) of `model` for `doc`.
enum Raw {
    Value(f64),
    Probabilities(Vec<f64>),
}

fn evaluate(model: &Value, doc: &Value, classification: bool) -> Result<Raw> {
    if let Some(t) = model.get("tree") {
        let v = leaf(t, doc)?;
        return Ok(if classification { Raw::Probabilities(tree_probabilities(t, &v)?) } else { Raw::Value(v[0]) });
    }
    let e = model.get("ensemble").ok_or_else(|| bad("trained model is neither a [tree] nor an [ensemble]"))?;
    let trees = e["trained_models"].as_array().ok_or_else(|| bad("ensemble has no [trained_models]"))?;
    let mut values = Vec::with_capacity(trees.len());
    for m in trees {
        let t = m.get("tree").ok_or_else(|| bad("only trees are supported inside an ensemble"))?;
        values.push(leaf(t, doc)?);
    }
    let (agg, weights) = aggregator(e);
    if weights.as_ref().is_some_and(|w| w.len() != values.len()) {
        return Err(bad("values must be the same length as weights."));
    }
    let w = |j: usize| weights.as_ref().map_or(1.0, |w| w[j]);
    match agg.as_str() {
        "weighted_sum" if !classification => Ok(Raw::Value(values.iter().enumerate().map(|(j, v)| v[0] * w(j)).sum())),
        "exponent" if !classification => Ok(Raw::Value(values.iter().enumerate().map(|(j, v)| v[0] * w(j)).filter(|x| x.is_finite()).sum::<f64>().exp())),
        "logistic_regression" if classification => {
            let width = values[0].len();
            let mut sum = vec![0.0; width];
            for (j, v) in values.iter().enumerate() {
                if v.len() != width {
                    return Err(bad("value entries must have the same dimensions"));
                }
                for (i, x) in v.iter().enumerate() {
                    sum[i] += x * w(j);
                }
            }
            Ok(Raw::Probabilities(if width > 1 {
                softmax(&sum)
            } else {
                let p = sigmoid(sum[0]);
                vec![1.0 - p, p]
            }))
        }
        "weighted_mode" if classification => {
            let num_classes = e
                .get("aggregate_output")
                .and_then(|a| a.get("weighted_mode"))
                .and_then(|m| m.get("num_classes"))
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .or_else(|| e["classification_labels"].as_array().map(Vec::len))
                .unwrap_or(2);
            let mut freq = vec![f64::NEG_INFINITY; num_classes];
            for (j, v) in values.iter().enumerate() {
                let c = v[0];
                if v.len() != 1 || !c.is_finite() || c < 0.0 || c.fract() != 0.0 || c as usize >= num_classes {
                    return Err(bad("values must be whole, non-infinite, and positive"));
                }
                let f = &mut freq[c as usize];
                *f = if f.is_finite() { *f + w(j) } else { w(j) };
            }
            Ok(Raw::Probabilities(softmax(&freq)))
        }
        other => Err(bad(format!("output aggregator [{other}] is not supported for a [{}] ensemble", if classification { "classification" } else { "regression" }))),
    }
}

/// The inference result of a tree / ensemble `definition` (its
/// `trained_model` and `preprocessors`) for `doc` under the `kind`
/// (`regression` / `classification`) inference config `ic`.
pub fn infer(definition: &Value, kind: &str, ic: &Map<String, Value>, doc: &Value) -> Result<Value> {
    if definition.get("preprocessors").and_then(Value::as_array).is_some_and(|p| !p.is_empty()) {
        return Err(bad("trained model preprocessors are not supported for inference"));
    }
    if ic.get("num_top_feature_importance_values").and_then(Value::as_u64).unwrap_or(0) > 0 {
        return Err(bad("feature importance is not supported for inference"));
    }
    let model = definition.get("trained_model").ok_or_else(|| bad("trained model has no definition"))?;
    let target = model.get("tree").or_else(|| model.get("ensemble")).and_then(|m| m["target_type"].as_str()).unwrap_or("regression");
    let classification = match kind {
        "classification" => true,
        "regression" => false,
        other => return Err(bad(format!("Cannot infer using configuration for [{other}] when model target_type is [{target}]"))),
    };
    if classification != (target == "classification") {
        return Err(bad(format!("Cannot infer using configuration for [{kind}] when model target_type is [{target}]")));
    }
    let results_field = ic.get("results_field").and_then(Value::as_str).unwrap_or("predicted_value").to_string();
    match evaluate(model, doc, classification)? {
        Raw::Value(v) => Ok(json!({ results_field: v })),
        Raw::Probabilities(p) => {
            let holder = model.get("tree").or_else(|| model.get("ensemble")).unwrap_or(&Value::Null);
            let labels: Option<Vec<String>> = holder["classification_labels"].as_array().map(|a| a.iter().map(|l| l.as_str().map(str::to_string).unwrap_or_else(|| l.to_string())).collect());
            if labels.as_ref().is_some_and(|l| l.len() != p.len()) {
                return Err(EsError::new(
                    500,
                    "status_exception",
                    format!("model returned classification probabilities of size [{}] which is not equal to classification labels size [{}]", p.len(), labels.as_ref().map_or(0, Vec::len)),
                ));
            }
            let weights = holder.get("classification_weights").and_then(floats);
            let scores: Vec<f64> = match &weights {
                Some(w) if w.len() == p.len() => p.iter().zip(w).map(|(a, b)| a * b).collect(),
                _ => p.clone(),
            };
            let mut order: Vec<usize> = (0..p.len()).collect();
            // `Comparator.comparing(score).reversed()`: stable, best first.
            order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
            let label = |i: usize| labels.as_ref().map(|l| l[i].clone()).unwrap_or_else(|| i.to_string());
            let best = order[0];
            let mut out = Map::new();
            out.insert(results_field, json!(label(best)));
            let n = ic.get("num_top_classes").and_then(Value::as_i64).unwrap_or(0);
            if n != 0 {
                let count = if n < 0 { p.len() } else { (n as usize).min(p.len()) };
                let top: Vec<Value> = order.iter().take(count).map(|&i| json!({"class_name": label(i), "class_probability": p[i], "class_score": scores[i]})).collect();
                let field = ic.get("top_classes_results_field").and_then(Value::as_str).unwrap_or("top_classes").to_string();
                out.insert(field, Value::Array(top));
            }
            out.insert("prediction_probability".into(), json!(p[best]));
            out.insert("prediction_score".into(), json!(scores[best]));
            Ok(Value::Object(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stump(value: f64) -> Value {
        json!({"trained_model": {"tree": {"feature_names": [], "tree_structure": [{"node_index": 0, "leaf_value": value}], "target_type": "regression"}}})
    }

    #[test]
    fn constant_leaf_regression() {
        let r = infer(&stump(42.5), "regression", &Map::new(), &json!({"x": 1})).unwrap();
        assert_eq!(r, json!({"predicted_value": 42.5}));
        let ic = json!({"results_field": "out"}).as_object().unwrap().clone();
        assert_eq!(infer(&stump(-1.0), "regression", &ic, &json!({})).unwrap(), json!({"out": -1.0}));
    }

    fn split_tree(target: &str, left: Value, right: Value) -> Value {
        json!({"feature_names": ["f"], "target_type": target, "tree_structure": [
            {"node_index": 0, "split_feature": 0, "threshold": 0.5, "decision_type": "lte", "default_left": true, "left_child": 1, "right_child": 2},
            {"node_index": 1, "leaf_value": left},
            {"node_index": 2, "leaf_value": right}
        ]})
    }

    #[test]
    fn split_regression_and_ensembles() {
        let def = json!({"trained_model": {"tree": split_tree("regression", json!(1.0), json!(2.0))}});
        assert_eq!(infer(&def, "regression", &Map::new(), &json!({"f": 0.2})).unwrap()["predicted_value"], 1.0);
        assert_eq!(infer(&def, "regression", &Map::new(), &json!({"f": 0.9})).unwrap()["predicted_value"], 2.0);
        // Missing feature: default_left.
        assert_eq!(infer(&def, "regression", &Map::new(), &json!({})).unwrap()["predicted_value"], 1.0);
        let ens = json!({"trained_model": {"ensemble": {"feature_names": ["f"], "target_type": "regression", "aggregate_output": {"weighted_sum": {"weights": [0.5, 2.0]}}, "trained_models": [
            {"tree": split_tree("regression", json!(1.0), json!(2.0))},
            {"tree": split_tree("regression", json!(10.0), json!(20.0))}
        ]}}});
        assert_eq!(infer(&ens, "regression", &Map::new(), &json!({"f": 0.9})).unwrap()["predicted_value"], 41.0);
    }

    #[test]
    fn known_classifiers() {
        // Multi-value leaves: softmax over the leaf.
        let def = json!({"trained_model": {"tree": {
            "feature_names": ["f"], "target_type": "classification", "classification_labels": ["no", "yes"],
            "tree_structure": [
                {"node_index": 0, "split_feature": 0, "threshold": 0.5, "left_child": 1, "right_child": 2},
                {"node_index": 1, "leaf_value": [2.0, 0.0]},
                {"node_index": 2, "leaf_value": [0.0, 2.0]}
            ]}}});
        let ic = json!({"num_top_classes": 2}).as_object().unwrap().clone();
        let r = infer(&def, "classification", &ic, &json!({"f": 1.0})).unwrap();
        let p = 2f64.exp() / (2f64.exp() + 1.0);
        assert_eq!(r["predicted_value"], "yes");
        assert!((r["prediction_probability"].as_f64().unwrap() - p).abs() < 1e-12);
        assert_eq!(r["top_classes"][0]["class_name"], "yes");
        assert_eq!(r["top_classes"][1]["class_name"], "no");
        // Binary logistic regression ensemble: sigmoid of the weighted sum.
        let ens = json!({"trained_model": {"ensemble": {"feature_names": ["f"], "target_type": "classification", "classification_labels": ["a", "b"], "aggregate_output": {"logistic_regression": {"weights": [1.0, 1.0]}}, "trained_models": [
            {"tree": split_tree("regression", json!(-1.0), json!(1.0))},
            {"tree": split_tree("regression", json!(-0.5), json!(0.5))}
        ]}}});
        let r = infer(&ens, "classification", &Map::new(), &json!({"f": 0.0})).unwrap();
        assert_eq!(r["predicted_value"], "a");
        assert!((r["prediction_probability"].as_f64().unwrap() - (1.0 - sigmoid(-1.5))).abs() < 1e-12);
        assert!(r.get("top_classes").is_none());
        // Unsupported parts are refused, never answered empty.
        let pre = json!({"preprocessors": [{"one_hot_encoding": {}}], "trained_model": stump(1.0)["trained_model"].clone()});
        assert!(infer(&pre, "regression", &Map::new(), &json!({})).is_err());
        assert!(infer(&def, "regression", &Map::new(), &json!({})).is_err());
    }
}
