//! The ingest document: `_source` plus metadata (`_index`, `_id`, ...) in
//! one map (what scripts see as `ctx`), and the `_ingest` metadata map.
//! Field paths follow ES `IngestDocument`: dotted segments, numeric
//! segments index lists, `_source.` / `_ingest.` prefixes select the root.

use super::jtext::{java_tostring, mustache_string, url_encode, MediaType};
use crate::error::EsError;
use serde_json::{Map, Value};

/// Metadata keys stored next to the source fields.
pub const METADATA_KEYS: &[&str] = &["_index", "_id", "_routing", "_version", "_version_type", "_if_seq_no", "_if_primary_term", "_dynamic_templates"];

#[derive(Clone, Debug)]
pub struct IngestDocument {
    /// Source fields + metadata keys (an object).
    pub ctx: Value,
    /// `_ingest` metadata (an object).
    pub ingest: Value,
    /// Set by `reroute`: stop every enclosing pipeline and resolve the new index's pipelines.
    pub rerouted: bool,
}

/// Java class name of a value (ES error messages quote them).
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "java.lang.Boolean",
        Value::Number(n) => {
            if n.is_f64() {
                "java.lang.Double"
            } else if n.as_i64().map(|x| x >= i32::MIN as i64 && x <= i32::MAX as i64).unwrap_or(false) {
                "java.lang.Integer"
            } else {
                "java.lang.Long"
            }
        }
        Value::String(_) => "java.lang.String",
        Value::Array(_) => "java.util.ArrayList",
        Value::Object(_) => "java.util.HashMap",
    }
}

fn ia(msg: String) -> EsError {
    EsError::illegal_argument(msg)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Root {
    Ctx,
    Ingest,
}

fn split(path: &str) -> Result<(Root, Vec<&str>), EsError> {
    if path.is_empty() {
        return Err(ia("path cannot be null nor empty".into()));
    }
    let (root, rest) = if path == "_ingest" {
        (Root::Ingest, "")
    } else if let Some(r) = path.strip_prefix("_ingest.") {
        (Root::Ingest, r)
    } else if let Some(r) = path.strip_prefix("_source.") {
        (Root::Ctx, r)
    } else {
        (Root::Ctx, path)
    };
    let segs: Vec<&str> = if rest.is_empty() { Vec::new() } else { rest.split('.').collect() };
    if segs.iter().any(|s| s.is_empty()) {
        return Err(ia(format!("path [{path}] is not valid")));
    }
    Ok((root, segs))
}

fn index_of(seg: &str, path: &str, len: usize) -> Result<usize, EsError> {
    let i: usize = seg.parse().map_err(|_| ia(format!("[{seg}] is not an integer, cannot be used as an index as part of path [{path}]")))?;
    if i >= len {
        return Err(ia(format!("[{i}] is out of bounds for array with length [{len}] as part of path [{path}]")));
    }
    Ok(i)
}

impl IngestDocument {
    pub fn new(source: Map<String, Value>, index: &str, id: Option<&str>, routing: Option<&str>, timestamp: String) -> IngestDocument {
        let mut ctx = source;
        ctx.insert("_index".into(), Value::String(index.to_string()));
        if let Some(id) = id {
            ctx.insert("_id".into(), Value::String(id.to_string()));
        }
        if let Some(r) = routing {
            ctx.insert("_routing".into(), Value::String(r.to_string()));
        }
        let mut ingest = Map::new();
        ingest.insert("timestamp".into(), Value::String(timestamp));
        IngestDocument { ctx: Value::Object(ctx), ingest: Value::Object(ingest), rerouted: false }
    }

    pub fn ctx_map(&self) -> &Map<String, Value> {
        self.ctx.as_object().expect("ctx is an object")
    }

    pub fn ctx_map_mut(&mut self) -> &mut Map<String, Value> {
        if !self.ctx.is_object() {
            self.ctx = Value::Object(Map::new());
        }
        self.ctx.as_object_mut().unwrap()
    }

    pub fn ingest_map_mut(&mut self) -> &mut Map<String, Value> {
        if !self.ingest.is_object() {
            self.ingest = Value::Object(Map::new());
        }
        self.ingest.as_object_mut().unwrap()
    }

    pub fn meta_str(&self, k: &str) -> Option<String> {
        match self.ctx_map().get(k) {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            Some(v) => Some(v.to_string()),
        }
    }

    fn root(&self, r: Root) -> &Value {
        match r {
            Root::Ctx => &self.ctx,
            Root::Ingest => &self.ingest,
        }
    }

    fn root_mut(&mut self, r: Root) -> &mut Value {
        match r {
            Root::Ctx => &mut self.ctx,
            Root::Ingest => &mut self.ingest,
        }
    }

    /// Value at `path`; `Ok(None)` when absent, `Err` when the path walks
    /// through a scalar.
    pub fn get(&self, path: &str) -> Result<Option<&Value>, EsError> {
        Ok(self.lookup(path)?.ok())
    }

    /// `Ok(Err(segment))` names the missing segment.
    fn lookup(&self, path: &str) -> Result<Result<&Value, String>, EsError> {
        let (root, segs) = split(path)?;
        let mut cur = self.root(root);
        for s in segs {
            cur = match cur {
                Value::Object(m) => match m.get(s) {
                    Some(v) => v,
                    None => return Ok(Err(s.to_string())),
                },
                Value::Array(a) => {
                    let i: usize = s.parse().map_err(|_| ia(format!("[{s}] is not an integer, cannot be used as an index as part of path [{path}]")))?;
                    match a.get(i) {
                        Some(v) => v,
                        None => return Err(ia(format!("[{i}] is out of bounds for array with length [{}] as part of path [{path}]", a.len()))),
                    }
                }
                Value::Null => return Ok(Err(s.to_string())),
                other => return Err(ia(format!("cannot resolve [{s}] from object of type [{}] as part of path [{path}]", type_name(other)))),
            };
        }
        Ok(Ok(cur))
    }

    /// Value at `path` or ES's "not present" error.
    pub fn require(&self, path: &str) -> Result<&Value, EsError> {
        match self.lookup(path)? {
            Ok(v) => Ok(v),
            Err(seg) => Err(ia(format!("field [{seg}] not present as part of path [{path}]"))),
        }
    }

    pub fn has(&self, path: &str) -> bool {
        matches!(self.lookup(path), Ok(Ok(_)))
    }

    pub fn set(&mut self, path: &str, value: Value) -> Result<(), EsError> {
        self.set_inner(path, value, false, true)
    }

    pub fn append(&mut self, path: &str, value: Value, allow_duplicates: bool) -> Result<(), EsError> {
        self.set_inner(path, value, true, allow_duplicates)
    }

    fn set_inner(&mut self, path: &str, value: Value, append: bool, allow_dup: bool) -> Result<(), EsError> {
        let (root, segs) = split(path)?;
        if segs.is_empty() {
            if root == Root::Ingest && value.is_object() && !append {
                self.ingest = value;
                return Ok(());
            }
            return Err(ia(format!("cannot set [{path}]")));
        }
        let (leaf, parents) = segs.split_last().unwrap();
        if root == Root::Ctx && parents.is_empty() && !append && METADATA_KEYS.contains(leaf) {
            validate_meta(leaf, &value)?;
        }
        let mut cur = self.root_mut(root);
        for s in parents {
            if cur.is_null() {
                *cur = Value::Object(Map::new());
            }
            cur = match cur {
                Value::Object(m) => {
                    let e = m.entry(s.to_string()).or_insert(Value::Null);
                    if e.is_null() {
                        *e = Value::Object(Map::new());
                    }
                    e
                }
                Value::Array(a) => {
                    let i = index_of(s, path, a.len())?;
                    &mut a[i]
                }
                other => return Err(ia(format!("cannot set [{s}] with parent object of type [{}] as part of path [{path}]", type_name(other)))),
            };
        }
        match cur {
            Value::Object(m) => {
                if append {
                    match m.get_mut(*leaf) {
                        Some(existing) => append_values(existing, value, allow_dup),
                        None => {
                            let mut l = Value::Array(Vec::new());
                            append_values(&mut l, value, allow_dup);
                            m.insert(leaf.to_string(), l);
                        }
                    }
                } else {
                    m.insert(leaf.to_string(), value);
                }
                Ok(())
            }
            Value::Array(a) => {
                let i = index_of(leaf, path, a.len())?;
                if append {
                    append_values(&mut a[i], value, allow_dup);
                } else {
                    a[i] = value;
                }
                Ok(())
            }
            other => Err(ia(format!("cannot set [{leaf}] with parent object of type [{}] as part of path [{path}]", type_name(other)))),
        }
    }

    /// Remove `path`, returning the old value.
    pub fn remove(&mut self, path: &str) -> Result<Value, EsError> {
        let (root, segs) = split(path)?;
        let Some((leaf, parents)) = segs.split_last() else {
            return Err(ia(format!("cannot remove [{path}]")));
        };
        if root == Root::Ctx && parents.is_empty() && *leaf == "_version" {
            return Err(ia("_version cannot be removed".into()));
        }
        let mut cur = self.root_mut(root);
        for s in parents {
            cur = match cur {
                Value::Object(m) => match m.get_mut(*s) {
                    Some(v) => v,
                    None => return Err(ia(format!("field [{s}] not present as part of path [{path}]"))),
                },
                Value::Array(a) => {
                    let i = index_of(s, path, a.len())?;
                    &mut a[i]
                }
                Value::Null => return Err(ia(format!("field [{s}] not present as part of path [{path}]"))),
                other => return Err(ia(format!("cannot remove [{s}] from object of type [{}] as part of path [{path}]", type_name(other)))),
            };
        }
        match cur {
            Value::Object(m) => m.shift_remove(*leaf).ok_or_else(|| ia(format!("field [{leaf}] not present as part of path [{path}]"))),
            Value::Array(a) => {
                let i = index_of(leaf, path, a.len())?;
                Ok(a.remove(i))
            }
            Value::Null => Err(ia(format!("field [{leaf}] not present as part of path [{path}]"))),
            other => Err(ia(format!("cannot remove [{leaf}] from object of type [{}] as part of path [{path}]", type_name(other)))),
        }
    }

    /// Render a mustache template against the doc with ES's default
    /// (`application/json`) escaping.
    pub fn render(&self, tpl: &str) -> String {
        self.render_media(tpl, MediaType::Json)
    }

    /// Render with an explicit media type (`set`/`append` `media_type`).
    pub fn render_media(&self, tpl: &str, media: MediaType) -> String {
        if !tpl.contains("{{") {
            return tpl.to_string();
        }
        let toks = parse_template(tpl);
        let mut out = String::with_capacity(tpl.len());
        let mut stack: Vec<Value> = Vec::new();
        self.render_toks(&toks, &mut stack, media, &mut out);
        out
    }

    fn resolve(&self, name: &str, stack: &[Value]) -> Option<Value> {
        if name == "." || name == "this" {
            return stack.last().cloned();
        }
        let (first, rest) = name.split_once('.').unwrap_or((name, ""));
        for ctx in stack.iter().rev() {
            if let Value::Object(m) = ctx {
                if let Some(v) = m.get(first) {
                    return walk(v, rest);
                }
            }
        }
        self.get(name).ok().flatten().cloned()
    }

    fn render_toks(&self, toks: &[Tok], stack: &mut Vec<Value>, media: MediaType, out: &mut String) {
        for t in toks {
            match t {
                Tok::Text(s) => out.push_str(s),
                Tok::Var { name, raw } => {
                    if let Some(v) = self.resolve(name, stack) {
                        let s = mustache_string(&v);
                        out.push_str(&if *raw { s } else { media.escape(&s) });
                    }
                }
                Tok::Section { name, inverted, body } => {
                    // ES's built-in lambdas.
                    let fname = name.split_whitespace().next().unwrap_or("");
                    if matches!(fname, "toJson" | "join" | "url") && !*inverted {
                        let mut inner = String::new();
                        self.render_toks_raw(body, &mut inner);
                        let path = inner.trim();
                        let v = self.resolve(path, stack);
                        match fname {
                            "toJson" => out.push_str(&media.escape(&v.map(|v| v.to_string()).unwrap_or_default())),
                            "join" => {
                                let delim = name.split_once("delimiter=").map(|(_, d)| d.trim().trim_matches(|c| c == '\'' || c == '"').to_string()).unwrap_or_else(|| ",".into());
                                if let Some(Value::Array(a)) = v {
                                    out.push_str(&media.escape(&a.iter().map(java_tostring).collect::<Vec<_>>().join(&delim)));
                                }
                            }
                            _ => out.push_str(&url_encode(&path_or(&v, path))),
                        }
                        continue;
                    }
                    let v = self.resolve(name, stack);
                    let truthy = match &v {
                        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
                        Some(Value::Array(a)) => !a.is_empty(),
                        Some(Value::String(s)) => !s.is_empty(),
                        _ => true,
                    };
                    if *inverted {
                        if !truthy {
                            self.render_toks(body, stack, media, out);
                        }
                    } else if truthy {
                        match v {
                            Some(Value::Array(a)) => {
                                for item in a {
                                    stack.push(item);
                                    self.render_toks(body, stack, media, out);
                                    stack.pop();
                                }
                            }
                            Some(Value::Bool(true)) => self.render_toks(body, stack, media, out),
                            Some(other) => {
                                stack.push(other);
                                self.render_toks(body, stack, media, out);
                                stack.pop();
                            }
                            None => {}
                        }
                    }
                }
            }
        }
    }

    fn render_toks_raw(&self, toks: &[Tok], out: &mut String) {
        for t in toks {
            if let Tok::Text(s) = t {
                out.push_str(s);
            }
        }
    }

    /// Render templates in every string of a JSON value (keys too, like ES's `ValueSource`).
    pub fn render_value(&self, v: &Value) -> Value {
        self.render_value_media(v, MediaType::Json)
    }

    pub fn render_value_media(&self, v: &Value, media: MediaType) -> Value {
        match v {
            Value::String(s) if s.contains("{{") => Value::String(self.render_media(s, media)),
            Value::Array(a) => Value::Array(a.iter().map(|x| self.render_value_media(x, media)).collect()),
            Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (self.render_media(k, media), self.render_value_media(x, media))).collect()),
            other => other.clone(),
        }
    }

    /// Split into (source, metadata).
    pub fn into_parts(self) -> (Map<String, Value>, Map<String, Value>) {
        let mut src = match self.ctx {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        let mut meta = Map::new();
        for k in METADATA_KEYS {
            if let Some(v) = src.shift_remove(*k) {
                meta.insert(k.to_string(), v);
            }
        }
        (src, meta)
    }

    /// Source fields only (metadata keys excluded), as a JSON object.
    pub fn source_value(&self) -> Value {
        let mut m = Map::new();
        for (k, v) in self.ctx_map() {
            if !METADATA_KEYS.contains(&k.as_str()) {
                m.insert(k.clone(), v.clone());
            }
        }
        Value::Object(m)
    }
}

fn path_or(v: &Option<Value>, path: &str) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(other) => mustache_string(other),
        None => path.to_string(),
    }
}

fn walk(v: &Value, rest: &str) -> Option<Value> {
    if rest.is_empty() {
        return Some(v.clone());
    }
    let (seg, tail) = rest.split_once('.').unwrap_or((rest, ""));
    match v {
        Value::Object(m) => m.get(seg).and_then(|x| walk(x, tail)),
        Value::Array(a) => seg.parse::<usize>().ok().and_then(|i| a.get(i)).and_then(|x| walk(x, tail)),
        _ => None,
    }
}

enum Tok {
    Text(String),
    Var { name: String, raw: bool },
    Section { name: String, inverted: bool, body: Vec<Tok> },
}

fn parse_template(tpl: &str) -> Vec<Tok> {
    // Flat scan into (kind, text) then nest sections.
    enum Raw {
        Text(String),
        Var(String, bool),
        Open(String, bool),
        Close(String),
    }
    let mut raws: Vec<Raw> = Vec::new();
    let mut rest = tpl;
    while let Some(start) = rest.find("{{") {
        if start > 0 {
            raws.push(Raw::Text(rest[..start].to_string()));
        }
        let after = &rest[start..];
        let (open, close) = if after.starts_with("{{{") { (3, "}}}") } else { (2, "}}") };
        match after[open..].find(close) {
            Some(end) => {
                let inner = after[open..open + end].trim();
                if open == 3 {
                    raws.push(Raw::Var(inner.to_string(), true));
                } else if let Some(n) = inner.strip_prefix('#') {
                    raws.push(Raw::Open(n.trim().to_string(), false));
                } else if let Some(n) = inner.strip_prefix('^') {
                    raws.push(Raw::Open(n.trim().to_string(), true));
                } else if let Some(n) = inner.strip_prefix('/') {
                    raws.push(Raw::Close(n.trim().to_string()));
                } else if inner.starts_with('!') {
                } else if let Some(n) = inner.strip_prefix('&') {
                    raws.push(Raw::Var(n.trim().to_string(), true));
                } else {
                    raws.push(Raw::Var(inner.to_string(), false));
                }
                rest = &after[open + end + close.len()..];
            }
            None => {
                raws.push(Raw::Text(after.to_string()));
                rest = "";
            }
        }
    }
    if !rest.is_empty() {
        raws.push(Raw::Text(rest.to_string()));
    }
    // nest
    let mut stack: Vec<(String, bool, Vec<Tok>)> = vec![(String::new(), false, Vec::new())];
    for r in raws {
        match r {
            Raw::Text(s) => stack.last_mut().unwrap().2.push(Tok::Text(s)),
            Raw::Var(n, raw) => stack.last_mut().unwrap().2.push(Tok::Var { name: n, raw }),
            Raw::Open(n, inv) => stack.push((n, inv, Vec::new())),
            Raw::Close(_) => {
                if stack.len() > 1 {
                    let (name, inverted, body) = stack.pop().unwrap();
                    stack.last_mut().unwrap().2.push(Tok::Section { name, inverted, body });
                }
            }
        }
    }
    while stack.len() > 1 {
        let (name, inverted, body) = stack.pop().unwrap();
        stack.last_mut().unwrap().2.push(Tok::Section { name, inverted, body });
    }
    stack.pop().unwrap().2
}

/// Java-ish `toString` of a value for templates and string conversion.
pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => java_tostring(other),
    }
}

/// ES `Metadata` field validation for `_index`, `_id`, `_version`, ...
pub(super) fn validate_meta(key: &str, v: &Value) -> Result<(), EsError> {
    if v.is_null() {
        return if key == "_version" { Err(ia("_version cannot be null".into())) } else { Ok(()) };
    }
    let (ok, class) = match key {
        "_index" | "_id" | "_routing" => (v.is_string(), "java.lang.String"),
        "_version" | "_if_seq_no" | "_if_primary_term" => (v.is_number(), "java.lang.Number"),
        "_dynamic_templates" => (v.is_object(), "java.util.Map"),
        "_version_type" => {
            return match v.as_str() {
                Some("internal") | Some("external") | Some("external_gte") => Ok(()),
                _ => Err(ia(format!("_version_type must be a null or one of [internal, external, external_gte] but was [{}] with type [{}]", java_tostring(v), type_name(v)))),
            };
        }
        _ => return Ok(()),
    };
    if ok {
        Ok(())
    } else {
        Err(ia(format!("{key} [{}] is wrong type, expected assignable to [{class}], not [{}]", java_tostring(v), type_name(v))))
    }
}

fn append_values(existing: &mut Value, value: Value, allow_dup: bool) {
    if !existing.is_array() {
        let old = std::mem::take(existing);
        *existing = Value::Array(vec![old]);
    }
    let list = existing.as_array_mut().unwrap();
    let mut push = |x: Value| {
        if allow_dup || !list.contains(&x) {
            list.push(x);
        }
    };
    match value {
        Value::Array(a) => a.into_iter().for_each(&mut push),
        other => push(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc(v: Value) -> IngestDocument {
        IngestDocument::new(v.as_object().unwrap().clone(), "idx", Some("1"), None, "ts".into())
    }

    #[test]
    fn paths() {
        let mut d = doc(json!({"a": {"b": [1, {"c": 2}]}, "s": "x"}));
        assert_eq!(d.get("a.b.1.c").unwrap(), Some(&json!(2)));
        assert_eq!(d.get("_source.s").unwrap(), Some(&json!("x")));
        assert_eq!(d.get("_ingest.timestamp").unwrap(), Some(&json!("ts")));
        assert_eq!(d.get("_index").unwrap(), Some(&json!("idx")));
        assert!(d.get("nope.x").unwrap().is_none());
        assert_eq!(d.get("s.x").unwrap_err().reason, "cannot resolve [x] from object of type [java.lang.String] as part of path [s.x]");
        assert_eq!(d.require("a.z").unwrap_err().reason, "field [z] not present as part of path [a.z]");
        d.set("x.y.z", json!(1)).unwrap();
        assert_eq!(d.get("x").unwrap(), Some(&json!({"y": {"z": 1}})));
        d.append("tags", json!("a"), true).unwrap();
        d.append("tags", json!(["a", "b"]), false).unwrap();
        assert_eq!(d.get("tags").unwrap(), Some(&json!(["a", "b"])));
        d.append("s", json!("y"), true).unwrap();
        assert_eq!(d.get("s").unwrap(), Some(&json!(["x", "y"])));
        assert_eq!(d.remove("a.b.0").unwrap(), json!(1));
        assert_eq!(d.render("{{_index}}-{{{x.y.z}}}-{{missing}}"), "idx-1-");
        let (src, meta) = d.into_parts();
        assert_eq!(meta["_id"], json!("1"));
        assert!(src.get("_index").is_none());
    }
}
