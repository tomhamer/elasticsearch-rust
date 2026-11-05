//! `flattened` field encoding: every leaf of the object becomes a root
//! value (`value`) and a keyed value (`key\0value`, ES `_keyed` sub-field),
//! where `key` is the dotted path inside the object.

use serde_json::Value;

pub const SEP: char = '\0';

/// `key\0value`.
pub fn keyed(key: &str, value: &str) -> String {
    let mut s = String::with_capacity(key.len() + 1 + value.len());
    s.push_str(key);
    s.push(SEP);
    s.push_str(value);
    s
}

/// Split a keyed value.
pub fn split(keyed: &str) -> Option<(&str, &str)> {
    keyed.split_once(SEP)
}

/// Prefix of every keyed value of `key`.
pub fn key_prefix(key: &str) -> String {
    let mut s = key.to_string();
    s.push(SEP);
    s
}

/// The leaves `(dotted key, value)` of a flattened object, depth-first in
/// document order. Arrays contribute each element under the same key,
/// nulls are skipped (or replaced by `null_value`). Fails with the depth
/// reached when it exceeds `depth_limit`.
pub fn flatten(v: &Value, depth_limit: usize, null_value: Option<&str>) -> Result<Vec<(String, String)>, usize> {
    let mut out = Vec::new();
    fn walk(v: &Value, key: &str, depth: usize, limit: usize, nv: Option<&str>, out: &mut Vec<(String, String)>) -> Result<(), usize> {
        match v {
            Value::Object(m) => {
                if depth + 1 > limit {
                    return Err(depth + 1);
                }
                for (k, x) in m {
                    let nk = if key.is_empty() { k.clone() } else { format!("{key}.{k}") };
                    walk(x, &nk, depth + 1, limit, nv, out)?;
                }
                Ok(())
            }
            Value::Array(a) => {
                for x in a {
                    walk(x, key, depth, limit, nv, out)?;
                }
                Ok(())
            }
            Value::Null => {
                if let Some(n) = nv {
                    out.push((key.to_string(), n.to_string()));
                }
                Ok(())
            }
            Value::String(s) => {
                out.push((key.to_string(), s.clone()));
                Ok(())
            }
            Value::Bool(b) => {
                out.push((key.to_string(), b.to_string()));
                Ok(())
            }
            Value::Number(n) => {
                out.push((key.to_string(), n.to_string()));
                Ok(())
            }
        }
    }
    walk(v, "", 0, depth_limit, null_value, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flatten_leaves() {
        let v = json!({"a": "x", "b": {"c": [1, "y", null], "d": true}});
        let l = flatten(&v, 20, None).unwrap();
        assert_eq!(l, vec![("a".into(), "x".into()), ("b.c".into(), "1".into()), ("b.c".into(), "y".into()), ("b.d".into(), "true".into())]);
        assert_eq!(flatten(&v, 1, None), Err(2));
        assert_eq!(keyed("b.c", "1"), "b.c\u{0}1");
        assert_eq!(split("b.c\u{0}1"), Some(("b.c", "1")));
    }
}
