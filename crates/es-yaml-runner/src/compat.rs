//! `--compat 8`: run the previous major's YAML suites the way ES runs its
//! `yamlRestCompatTest` task against a 9.x node:
//!
//! * every `do` (except `cat.*` APIs) carries
//!   `Content-Type` / `Accept: application/vnd.elasticsearch+json;compatible-with=8`
//!   (`RestCompatTestTransformTask`'s `InjectHeaders`);
//! * the tests are rewritten with the transforms the ES projects list in
//!   their `build.gradle` (`yamlRestCompatTestTransform`: `skipTest`,
//!   `replaceValueInMatch`, `addAllowedWarningRegex`, ...), extracted into
//!   `compat/compat8.json` by `compat/gen_compat.py`;
//! * a cluster feature ES 9.5.5 does not know counts as present
//!   (`ESRestTestFeatureService`: "assumed, then removed in this version").

use crate::suite::TestFile;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// The `compatible-with` major every request carries (0 = native run).
pub static COMPAT: AtomicU8 = AtomicU8::new(0);

pub fn compat_major() -> Option<u8> {
    match COMPAT.load(Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// The `Content-Type` / `Accept` value of a compat run.
pub fn header_value(major: u8) -> String {
    format!("application/vnd.elasticsearch+json;compatible-with={major}")
}

/// Whether the compat headers go on a `do` of this API (ES skips `cat.*`).
pub fn injects_headers(api: &str) -> bool {
    !api.starts_with("cat.")
}

/// Cluster features ES 9.5.5 knows (`features-9.5.5.txt`).
fn known_features() -> &'static HashSet<&'static str> {
    static K: OnceLock<HashSet<&'static str>> = OnceLock::new();
    K.get_or_init(|| include_str!("../compat/features-9.5.5.txt").lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).collect())
}

/// Features actually enabled by the pinned official ES9.5.5 release fixture.
pub fn reference_features() -> &'static HashSet<String> {
    static FEATURES: OnceLock<HashSet<String>> = OnceLock::new();
    FEATURES.get_or_init(|| serde_json::from_str::<Vec<String>>(include_str!("../compat/enabled-features-9.5.5.json")).expect("reference features").into_iter().collect())
}

fn capability_key(entry: &Value) -> String {
    let strings = |key: &str| {
        let mut values: Vec<String> = match &entry[key] {
            Value::Array(a) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
            Value::String(s) => vec![s.clone()],
            _ => Vec::new(),
        };
        values.sort(); values.dedup(); values
    };
    serde_json::json!([entry["method"].as_str().unwrap_or("GET").to_uppercase(), entry["path"].as_str().unwrap_or("/"), strings("parameters"), strings("capabilities")]).to_string()
}

/// None means the gate has not been classified; it cannot silently exclude required coverage.
pub fn reference_capability(entry: &Value) -> Option<bool> {
    static CAPS: OnceLock<std::collections::HashMap<String, bool>> = OnceLock::new();
    let caps = CAPS.get_or_init(|| {
        let data: Value = serde_json::from_str(include_str!("../compat/capabilities-9.5.5.json")).expect("reference capabilities");
        data["entries"].as_array().expect("reference entries").iter().map(|e| (capability_key(&e["request"]), e["supported"].as_bool().expect("reference boolean"))).collect()
    });
    caps.get(&capability_key(entry)).copied()
}

/// Under `--compat`, a feature the current version does not know is assumed present.
pub fn assumed_feature(f: &str) -> bool {
    compat_major().is_some() && !known_features().contains(f)
}

/// Which ES projects' transforms apply to the suite being run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SuiteKind {
    Oss,
    XPack,
    Plugins,
}

enum Op {
    Skip { file: String, test: String, reason: String },
    /// `match` / `length` (`key`) entry `sub` gets `value`.
    ReplaceValue { key: &'static str, sub: String, value: Value, test: Option<String> },
    ReplaceKeyInMatch { sub: String, new_key: String, test: Option<String> },
    /// A `key: "<old>"` text entry becomes `key: <new>` (`replaceIsTrue`, `replaceValueTextByKeyValue`, ...).
    ReplaceText { key: String, old: String, new: Value, test: Option<String> },
    AllowedWarnings { regex: bool, list: Vec<String>, test: Option<String> },
}

pub struct Transforms {
    ops: Vec<Op>,
}

fn s(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(o) => o.to_string(),
    }
}

fn opt(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str).map(str::to_string)
}

impl Transforms {
    pub fn load(kind: SuiteKind) -> Transforms {
        let doc: Value = serde_json::from_str(include_str!("../compat/compat8.json")).expect("compat8.json");
        let suite = match kind {
            SuiteKind::Oss => "oss",
            SuiteKind::XPack => "main",
            SuiteKind::Plugins => "plug",
        };
        let mut ops = Vec::new();
        for project in doc["suites"][suite].as_array().into_iter().flatten().filter_map(Value::as_str) {
            for o in doc["projects"][project].as_array().into_iter().flatten() {
                let a = o["args"].as_array().cloned().unwrap_or_default();
                let op = match o["op"].as_str().unwrap_or("") {
                    "skipTest" => {
                        let full = s(a.first());
                        let parts: Vec<&str> = full.splitn(3, '/').collect();
                        if parts.len() < 3 {
                            continue;
                        }
                        Op::Skip { file: format!("{}/{}.yml", parts[0], parts[1]), test: parts[2].to_string(), reason: s(a.get(1)) }
                    }
                    "replaceValueInMatch" => Op::ReplaceValue { key: "match", sub: s(a.first()), value: a.get(1).cloned().unwrap_or(Value::Null), test: opt(a.get(2)) },
                    "replaceValueInLength" => Op::ReplaceValue { key: "length", sub: s(a.first()), value: a.get(1).cloned().unwrap_or(Value::Null), test: opt(a.get(2)) },
                    "replaceValueInCloseTo" => Op::ReplaceValue { key: "close_to", sub: s(a.first()), value: a.get(1).cloned().unwrap_or(Value::Null), test: opt(a.get(2)) },
                    "replaceKeyInMatch" => Op::ReplaceKeyInMatch { sub: s(a.first()), new_key: s(a.get(1)), test: opt(a.get(2)) },
                    "replaceIsTrue" => Op::ReplaceText { key: "is_true".into(), old: s(a.first()), new: a.get(1).cloned().unwrap_or(Value::Null), test: opt(a.get(2)) },
                    "replaceIsFalse" => Op::ReplaceText { key: "is_false".into(), old: s(a.first()), new: a.get(1).cloned().unwrap_or(Value::Null), test: opt(a.get(2)) },
                    "replaceValueTextByKeyValue" => Op::ReplaceText { key: s(a.first()), old: s(a.get(1)), new: a.get(2).cloned().unwrap_or(Value::Null), test: opt(a.get(3)) },
                    "addAllowedWarning" => Op::AllowedWarnings { regex: false, list: a.iter().map(|x| s(Some(x))).collect(), test: None },
                    "addAllowedWarningRegex" => Op::AllowedWarnings { regex: true, list: a.iter().map(|x| s(Some(x))).collect(), test: None },
                    "addAllowedWarningRegexForTest" => Op::AllowedWarnings { regex: true, list: vec![s(a.first())], test: opt(a.get(1)) },
                    other => {
                        eprintln!("compat: transform `{other}` not implemented (ignored)");
                        continue;
                    }
                };
                ops.push(op);
            }
        }
        Transforms { ops }
    }

    /// ES `skipTest("dir/file/test", reason)`: the reason when this test is skipped.
    pub fn skip_reason(&self, rel: &str, test: &str) -> Option<String> {
        self.ops.iter().find_map(|o| match o {
            Op::Skip { file, test: t, reason } if t == test && (rel == file || rel.ends_with(&format!("/{file}"))) => Some(format!("compat skip: {reason}")),
            _ => None,
        })
    }

    /// Rewrite the file's sections (`RestTestTransformer.transformRestTests`).
    pub fn apply(&self, file: &mut TestFile) {
        for op in &self.ops {
            if matches!(op, Op::Skip { .. }) {
                continue;
            }
            apply_section(op, "setup", &mut file.setup);
            apply_section(op, "teardown", &mut file.teardown);
            for (name, steps) in file.tests.iter_mut() {
                apply_section(op, name, steps);
            }
        }
    }
}

fn test_applies(test: &Option<String>, name: &str) -> bool {
    test.as_deref().is_none_or(|t| t == name)
}

fn apply_section(op: &Op, name: &str, steps: &mut [Value]) {
    let applies = match op {
        Op::Skip { .. } => false,
        Op::ReplaceValue { test, .. } | Op::ReplaceKeyInMatch { test, .. } | Op::ReplaceText { test, .. } | Op::AllowedWarnings { test, .. } => test_applies(test, name),
    };
    if !applies {
        return;
    }
    for step in steps.iter_mut() {
        traverse(op, step);
    }
}

fn key_to_find(op: &Op) -> &str {
    match op {
        Op::ReplaceValue { key, .. } => key,
        Op::ReplaceKeyInMatch { .. } => "match",
        Op::ReplaceText { key, .. } => key,
        Op::AllowedWarnings { .. } => "do",
        Op::Skip { .. } => "",
    }
}

/// `RestTestTransformer.traverseTest` for one transform.
fn traverse(op: &Op, node: &mut Value) {
    match node {
        Value::Array(a) => a.iter_mut().for_each(|v| traverse(op, v)),
        Value::Object(obj) => {
            let keys: Vec<String> = obj.keys().cloned().collect();
            let find = key_to_find(op);
            for k in keys {
                if k != find {
                    if let Some(v) = obj.get_mut(&k) {
                        traverse(op, v);
                    }
                    continue;
                }
                transform(op, obj, &k);
                // ES's `traverseTest` keeps descending into the matched key's
                // value (`do: {index: {index: ...}}` holds a nested `index`).
                if let Some(v) = obj.get_mut(&k) {
                    traverse(op, v);
                }
            }
        }
        _ => {}
    }
}

fn transform(op: &Op, parent: &mut Map<String, Value>, key: &str) {
    match op {
        Op::ReplaceValue { sub, value, .. } => {
            if let Some(Value::Object(m)) = parent.get_mut(key) {
                if m.contains_key(sub) {
                    m.remove(sub);
                    m.insert(sub.clone(), value.clone());
                }
            }
        }
        Op::ReplaceKeyInMatch { sub, new_key, .. } => {
            if let Some(Value::Object(m)) = parent.get_mut(key) {
                if let Some(v) = m.remove(sub) {
                    m.insert(new_key.clone(), v);
                }
            }
        }
        Op::ReplaceText { old, new, .. } => {
            if parent.get(key).and_then(Value::as_str) == Some(old.as_str()) {
                parent.insert(key.to_string(), new.clone());
            }
        }
        Op::AllowedWarnings { regex, list, .. } => {
            if let Some(Value::Object(d)) = parent.get_mut(key) {
                let field = if *regex { "allowed_warnings_regex" } else { "allowed_warnings" };
                let arr = d.entry(field.to_string()).or_insert_with(|| Value::Array(vec![]));
                if let Value::Array(a) = arr {
                    a.extend(list.iter().cloned().map(Value::String));
                }
            }
        }
        Op::Skip { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_load_and_apply() {
        let t = Transforms::load(SuiteKind::Oss);
        assert!(t.skip_reason("search/500_date_range.yml", "from, to, include_lower, include_upper deprecated").is_some());
        assert!(t.skip_reason("x/search/500_date_range.yml", "from, to, include_lower, include_upper deprecated").is_some());
        assert!(t.skip_reason("search/500_date_range.yml", "other").is_none());
        let x = Transforms::load(SuiteKind::XPack);
        let mut f = crate::suite::parse_file(
            "ml/a.yml",
            "---\n\"t\":\n  - do:\n      indices.get_mapping:\n        index: .ml-anomalies-shared\n  - is_true: \\.ml-anomalies-shared.mappings._meta.version\n",
        )
        .unwrap();
        x.apply(&mut f);
        let steps = &f.tests[0].1;
        assert_eq!(steps[0]["do"]["indices.get_mapping"]["index"], ".ml-anomalies-shared-000001");
        assert_eq!(steps[0]["do"]["allowed_warnings_regex"].as_array().map(Vec::len), Some(4));
        assert_eq!(steps[1]["is_true"], "\\.ml-anomalies-shared-000001.mappings._meta.version");
    }
}
