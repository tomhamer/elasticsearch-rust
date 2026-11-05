//! `bucket_script` / `bucket_selector` scripts: Painless with the
//! `buckets_path` variables bound as `params.<name>`.

use crate::error::{EsError, Result};
use crate::script::{compile, ScriptContext, Val};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Run `src` with `vars` (plus non-numeric script `params`) as `params`.
pub fn eval_value(src: &str, vars: &HashMap<String, f64>, extra_params: Option<&Map<String, Value>>, ctx: ScriptContext) -> Result<Val> {
    let mut p = extra_params.cloned().unwrap_or_default();
    for (k, v) in vars {
        p.insert(k.clone(), serde_json::Number::from_f64(*v).map(Value::Number).unwrap_or(Value::Null));
    }
    let cs = compile(src, &Value::Object(p), ctx)?;
    let mut e = cs.executor(None, None);
    for (k, v) in vars {
        if !v.is_finite() {
            e.set_param(k, Val::Double(*v));
        }
    }
    e.run().map_err(|err| cs.error(&err))
}

/// Preserve raw request script parameters while binding bucket-path values.
pub fn eval_search_script(spec: &Value, vars: &HashMap<String, f64>, ctx: ScriptContext) -> Result<Val> {
    let cs = crate::script::compile_search_script(spec, ctx)?;
    let mut e = cs.executor(None, None);
    for (key, value) in vars { e.set_param(key, Val::Double(*value)); }
    e.run().map_err(|err| cs.error(&err))
}

/// Numeric result (`bucket_script`); booleans map to 1/0 (`bucket_selector`).
pub fn eval(src: &str, vars: &HashMap<String, f64>) -> Result<f64> {
    let v = eval_value(src, vars, None, ScriptContext::BucketScript)?;
    match v {
        Val::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        other => other.as_f64().ok_or_else(|| {
            EsError::new(400, "script_exception", "runtime error")
                .with("script", src.to_string())
                .with("lang", "painless")
                .caused_by(EsError::illegal_argument(format!("buckets_path must reference either a number value or a single value numeric metric aggregation, but [{}] contains a {}", src, other.type_name())))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates() {
        let mut v = HashMap::new();
        v.insert("a".to_string(), 10.0);
        v.insert("b".to_string(), 4.0);
        assert_eq!(eval("params.a / params.b * 100", &v).unwrap(), 250.0);
        assert_eq!(eval("params.a > 5 && params.b < 5", &v).unwrap(), 1.0);
        assert_eq!(eval("!(params.a > 5) || false", &v).unwrap(), 0.0);
        assert_eq!(eval("params['a'] % 3 + -1", &v).unwrap(), 0.0);
        assert_eq!(eval("Math.max(params.a, 20) - 2 * (1 + 1)", &v).unwrap(), 16.0);
        assert_eq!(eval("double r = params.a; if (r > 5) { return r * 2 } return 0", &v).unwrap(), 20.0);
        assert!(eval("params.a +", &v).is_err());
        assert!(eval("params.zzz", &v).is_err());
    }
}
