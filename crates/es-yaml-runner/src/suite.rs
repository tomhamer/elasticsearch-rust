//! Parsing of YAML test files into setup / teardown / test sections.

use serde::Deserialize;
use serde_json::{Map, Number, Value};

pub struct TestFile {
    /// Path relative to the suite root, e.g. `search/10_source_filtering.yml`.
    pub rel: String,
    /// API directory (first path component), e.g. `search`.
    pub dir: String,
    pub setup: Vec<Value>,
    pub teardown: Vec<Value>,
    pub tests: Vec<(String, Vec<Value>)>,
}

pub fn yaml_to_json(v: serde_yaml::Value) -> Value {
    use serde_yaml::Value as Y;
    match v {
        Y::Null => Value::Null,
        Y::Bool(b) => Value::Bool(b),
        Y::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                Value::Number(u.into())
            } else {
                let f = n.as_f64().unwrap_or(0.0);
                Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
            }
        }
        Y::String(s) => {
            // SnakeYAML parses Java double literals (`1.0d`, `2D`) as floats;
            // libyaml keeps them as strings. Strip the suffix so the value
            // indexes as a number, matching the ES test client. Only strip
            // when the prefix is a valid float literal (has a `.` or exponent)
            // so time values like `10d` (10 days) are preserved as strings.
            if let Some(stripped) = s.strip_suffix('d').or_else(|| s.strip_suffix('D')) {
                if !stripped.is_empty()
                    && stripped
                        .bytes()
                        .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'+' | b'-' | b'e' | b'E'))
                    && stripped.bytes().any(|b| b.is_ascii_digit())
                    && (stripped.contains('.') || stripped.contains('e') || stripped.contains('E'))
                {
                    if let Ok(f) = stripped.parse::<f64>() {
                        return Number::from_f64(f).map(Value::Number).unwrap_or(Value::String(s));
                    }
                }
            }
            Value::String(s)
        }
        Y::Sequence(s) => Value::Array(s.into_iter().map(yaml_to_json).collect()),
        Y::Mapping(m) => {
            let mut out = Map::new();
            for (k, v) in m {
                let key = match k {
                    Y::String(s) => s,
                    Y::Bool(b) => b.to_string(),
                    Y::Number(n) => n.to_string(),
                    Y::Null => "null".to_string(),
                    other => serde_yaml::to_string(&other).unwrap_or_default().trim().to_string(),
                };
                out.insert(key, yaml_to_json(v));
            }
            Value::Object(out)
        }
        Y::Tagged(t) => yaml_to_json(t.value),
    }
}

pub fn parse_file(rel: &str, text: &str) -> Result<TestFile, String> {
    // SnakeYAML (ES's parser) accepts a flow-mapping value glued to a plain
    // key (`{a.b:{"x":1}}`); libyaml does not, so space those out first.
    let fixed;
    let text = if serde_yaml::Deserializer::from_str(text).any(|d| serde_yaml::Value::deserialize(d).is_err()) {
        let re = regex::Regex::new(r"(^\s*|[{,]\s*)([A-Za-z0-9_.@$\-]+):([{\[])").expect("valid regex");
        // Integers beyond u64 (`18446744073709551616`, unsigned_long tests)
        // do not fit a YAML value here: send them as floats, the way a JSON
        // client without big-number support would.
        let big = regex::Regex::new(r#"(^|[^"'\w.])(\d{20,})\b"#).expect("valid regex");
        fixed = text
            .lines()
            .map(|l| {
                let l = re.replace_all(l, "$1$2: $3");
                big.replace_all(&l, |c: &regex::Captures| if c[2].parse::<u64>().is_ok() { c[0].to_string() } else { format!("{}{}.0", &c[1], &c[2]) }).into_owned()
            })
            .collect::<Vec<_>>()
            .join("\n");
        fixed.as_str()
    } else {
        text
    };
    let dir = rel.split('/').next().unwrap_or("").to_string();
    let mut f = TestFile { rel: rel.to_string(), dir, setup: vec![], teardown: vec![], tests: vec![] };
    for doc in serde_yaml::Deserializer::from_str(text) {
        let y = serde_yaml::Value::deserialize(doc).map_err(|e| format!("yaml parse error: {e}"))?;
        let v = yaml_to_json(y);
        let obj = match v {
            Value::Object(o) => o,
            Value::Null => continue,
            _ => return Err("yaml document is not a mapping".into()),
        };
        for (k, steps) in obj {
            let steps = match steps {
                Value::Array(a) => a,
                Value::Null => vec![],
                _ => return Err(format!("section [{k}] is not a list")),
            };
            match k.as_str() {
                "setup" => f.setup = steps,
                "teardown" => f.teardown = steps,
                _ => f.tests.push((k, steps)),
            }
        }
    }
    Ok(f)
}
