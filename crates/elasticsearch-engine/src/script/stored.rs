//! Stored scripts (`PUT/GET/DELETE _scripts/{id}`), persisted per node in
//! `<data>/_state/scripts.json`.

use crate::error::{EsError, Result};
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StoredScript {
    pub lang: String,
    pub source: String,
    /// Original JSON string token when source contains unpaired UTF-16 units.
    pub raw_source: Option<String>,
    /// Mustache sources may be JSON objects: kept as given.
    pub source_json: Option<Value>,
    pub options: Option<Value>,
}

#[derive(Debug)]
pub struct StoredScripts {
    path: PathBuf,
    scripts: RwLock<BTreeMap<String, StoredScript>>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<StoredScripts>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<StoredScripts>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The stored scripts of the node whose data directory is `data`.
pub fn for_data_dir(data: &Path) -> Arc<StoredScripts> {
    let key = data.to_path_buf();
    let mut r = registry().lock();
    if let Some(s) = r.get(&key) {
        return s.clone();
    }
    let s = Arc::new(StoredScripts::load(data));
    r.insert(key, s.clone());
    s
}

/// The stored scripts of the node owning the index directory `index_dir`
/// (`<data>/indices/<uuid>`).
pub fn for_index_dir(index_dir: &Path) -> Option<Arc<StoredScripts>> {
    let data = index_dir.parent()?.parent()?;
    Some(for_data_dir(data))
}

impl StoredScripts {
    fn load(data: &Path) -> StoredScripts {
        let path = data.join("_state").join("scripts.json");
        let mut scripts = BTreeMap::new();
        if let Some(Value::Object(m)) = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
            for (id, v) in m {
                let lang = v["lang"].as_str().unwrap_or("painless").to_string();
                let source_json = v.get("source").filter(|s| !s.is_string()).cloned();
                let source = match &v["source"] {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                scripts.insert(id, StoredScript { lang, source, raw_source: v.get("raw_source").and_then(Value::as_str).map(str::to_owned), source_json, options: v.get("options").cloned() });
            }
        }
        StoredScripts { path, scripts: RwLock::new(scripts) }
    }

    fn persist(&self, m: &BTreeMap<String, StoredScript>) -> Result<()> {
        let mut o = Map::new();
        for (id, s) in m {
            let mut e = Map::new();
            e.insert("lang".into(), json!(s.lang));
            e.insert("source".into(), s.source_json.clone().unwrap_or_else(|| json!(s.source)));
            if let Some(raw) = &s.raw_source { e.insert("raw_source".into(), json!(raw)); }
            if let Some(op) = &s.options {
                e.insert("options".into(), op.clone());
            }
            o.insert(id.clone(), Value::Object(e));
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&Value::Object(o))?)?;
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    /// Internal cluster payload keeps raw JSON tokens as escaped strings.
    pub fn cluster_state(&self) -> Value {
        serde_json::to_value(&*self.scripts.read()).expect("stored script metadata")
    }

    pub fn restore_cluster_state(&self, state: &Value) -> Result<()> {
        let next: BTreeMap<String, StoredScript> = serde_json::from_value(state.clone())?;
        let mut current = self.scripts.write();
        if *current != next {
            self.persist(&next)?;
            *current = next;
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<StoredScript> {
        self.scripts.read().get(id).cloned()
    }

    pub fn all(&self) -> Vec<(String, StoredScript)> {
        self.scripts.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// `PUT _scripts/{id}[/{context}]` body.
    pub fn put(&self, id: &str, body: &Value, context: Option<&str>) -> Result<()> {
        self.put_source(id, body, None, context)
    }

    pub fn put_source(&self, id: &str, body: &Value, raw: Option<&crate::source_json::Source>, context: Option<&str>) -> Result<()> {
        let raw_source = raw.and_then(|r| r.get("script")).and_then(|r| r.get("source"))
            .map(|s| s.text()).filter(|s| s.starts_with('"') && serde_json::from_str::<String>(s).is_err());
        if id.is_empty() || id.contains('#') {
            return Err(EsError::illegal_argument(format!("Validation Failed: 1: id cannot contain '#' for stored script;")));
        }
        let o = body.as_object().ok_or_else(|| EsError::parsing("request body is required"))?;
        // `StoredScriptSource.parse`: the first field must be `script` holding an object.
        let script = match o.get("script") {
            Some(Value::Object(s)) => s,
            Some(other) => return Err(EsError::parsing(format!("unexpected token [{}], expected [{{, <source>]", crate::jsonpos::token_name(other))).at_key("script", None).with_lc()),
            None => {
                return Err(match o.iter().next() {
                    Some((k, _)) => EsError::parsing(format!("unexpected field [{k}], expected [script]")).at_key(k, None).cursor("key").with_lc(),
                    None => EsError::parsing("unexpected token [END_OBJECT, expected [script]").cursor("rootend").with_lc(),
                });
            }
        };
        let lang = script.get("lang").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("must specify lang for stored script"))?.to_string();
        let src = script.get("source").or(script.get("code")).ok_or_else(|| EsError::illegal_argument("must specify source for stored script"))?;
        let (source, source_json) = match src {
            Value::String(s) => (s.clone(), None),
            other if lang == "mustache" => (other.to_string(), Some(other.clone())),
            _ => return Err(EsError::illegal_argument("must specify source for stored script")),
        };
        if source.is_empty() {
            return Err(EsError::illegal_argument("source cannot be empty"));
        }
        match lang.as_str() {
            "painless" => {
                let ctx = match context {
                    Some(name) => super::ScriptContext::from_name(name).ok_or_else(|| EsError::illegal_argument(format!("Unknown context [{name}]")))?,
                    None => super::ScriptContext::PainlessTest,
                };
                if let Some(raw) = &raw_source {
                    super::source::compile_raw_literal_source(&crate::source_json::decode_string(raw), ctx)?;
                } else { super::program(&source, ctx)?; }
            }
            "mustache" => {}
            other => return Err(EsError::illegal_argument(format!("unable to put stored script with unsupported lang [{other}]"))),
        }
        let mut m = self.scripts.write();
        m.insert(id.to_string(), StoredScript { lang, source, raw_source, source_json, options: script.get("options").cloned() });
        self.persist(&m)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let mut m = self.scripts.write();
        if m.remove(id).is_none() {
            return Err(EsError::resource_not_found(format!("stored script [{id}] does not exist")));
        }
        self.persist(&m)
    }

    /// Lossless wire representation; internal raw_source metadata is never exposed.
    pub fn get_raw_json(&self, id: &str) -> (bool, String) {
        let Some(s) = self.get(id) else {
            return (false, json!({"_id":id,"found":false}).to_string());
        };
        let mut sc = Map::new();
        sc.insert("lang".into(), json!(s.lang));
        sc.insert("source".into(), s.source_json.unwrap_or_else(|| json!(s.source)));
        if let Some(options) = s.options { sc.insert("options".into(), options); }
        let mut body = crate::source_json::Source::from_value(&json!({"_id":id,"found":true,"script":sc}));
        if let Some(raw) = s.raw_source {
            if let crate::source_json::Source::Object(root) = &mut body {
                if let Some(crate::source_json::Source::Object(script)) = root.get_mut("script") {
                    if let Ok(source) = crate::source_json::Source::parse(&raw) { script.insert("source".into(), source); }
                }
            }
        }
        (true, body.text())
    }

    /// `GET _scripts/{id}` body (and whether it was found).
    pub fn get_json(&self, id: &str) -> (bool, Value) {
        match self.get(id) {
            Some(s) => {
                let mut sc = Map::new();
                sc.insert("lang".into(), json!(s.lang));
                sc.insert("source".into(), s.source_json.clone().unwrap_or_else(|| json!(s.source)));
                if let Some(o) = &s.options {
                    sc.insert("options".into(), o.clone());
                }
                (true, json!({ "_id": id, "found": true, "script": Value::Object(sc) }))
            }
            None => (false, json!({ "_id": id, "found": false })),
        }
    }
}

#[cfg(test)]
mod utf16_tests {
    use super::*;
    #[test]
    fn stored_literal_units_survive_reload_and_execution() {
        let dir = tempfile::tempdir().unwrap();
        let scripts = StoredScripts::load(dir.path());
        let raw = r#"{"script":{"lang":"painless","source":"ctx._source.marker = '\ud800'"}}"#;
        let (metadata, original) = crate::source_json::update_request(raw).unwrap();
        scripts.put_source("raw", &metadata, original.as_ref(), Some("update")).unwrap();
        let loaded = StoredScripts::load(dir.path());
        let peer_dir = tempfile::tempdir().unwrap();
        let peer = StoredScripts::load(peer_dir.path());
        peer.restore_cluster_state(&scripts.cluster_state()).unwrap();
        let peer_reloaded = StoredScripts::load(peer_dir.path());
        for scripts in [&scripts, &loaded, &peer, &peer_reloaded] {
            let (found, body) = scripts.get_raw_json("raw");
            assert!(found);
            assert!(body.contains(r#"\ud800"#), "{body}");
            assert!(!body.contains('�'), "{body}");
            let compiled = super::super::compile_json(&json!({"id":"raw"}), super::super::ScriptContext::Update, Some(scripts)).unwrap();
            let (_, source) = super::super::update::run_doc_script_source(&compiled, "{}", &super::super::update::DocMeta::default(), "index").unwrap();
            assert_eq!(source, r#"{"marker":"\ud800"}"#);
        }
        scripts.delete("raw").unwrap();
        peer.restore_cluster_state(&scripts.cluster_state()).unwrap();
        assert!(peer.get("raw").is_none());
        assert!(StoredScripts::load(peer_dir.path()).get("raw").is_none());
    }
}
