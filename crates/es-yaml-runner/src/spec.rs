//! Loader for the `rest-api-spec/api/*.json` endpoint descriptions.

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub struct ApiPath {
    pub path: String,
    pub methods: Vec<String>,
    pub parts: Vec<String>,
}

pub struct Api {
    pub paths: Vec<ApiPath>,
    pub params: HashSet<String>,
    pub bulk_body: bool,
}

pub struct Specs {
    pub apis: HashMap<String, Api>,
    pub common: HashSet<String>,
}

fn path_parts(p: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = p;
    while let Some(i) = rest.find('{') {
        let after = &rest[i + 1..];
        match after.find('}') {
            Some(j) => {
                out.push(after[..j].to_string());
                rest = &after[j + 1..];
            }
            None => break,
        }
    }
    out
}

impl Specs {
    pub fn load(dir: &Path) -> Result<Specs, String> {
        let mut apis = HashMap::new();
        let mut common = HashSet::new();
        let rd = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e} (run tests/es-yaml/fetch.sh)", dir.display()))?;
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let v: Value = serde_json::from_slice(&std::fs::read(&p).map_err(|e| e.to_string())?)
                .map_err(|e| format!("{}: {e}", p.display()))?;
            if p.file_name().and_then(|s| s.to_str()) == Some("_common.json") {
                if let Some(m) = v["params"].as_object() {
                    common.extend(m.keys().cloned());
                }
                continue;
            }
            let Some(obj) = v.as_object() else { continue };
            for (name, api) in obj {
                let mut paths = Vec::new();
                if let Some(ps) = api["url"]["paths"].as_array() {
                    for pp in ps {
                        let path = pp["path"].as_str().unwrap_or("").to_string();
                        let methods = pp["methods"].as_array().map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect()).unwrap_or_default();
                        let parts = path_parts(&path);
                        paths.push(ApiPath { path, methods, parts });
                    }
                }
                let params = api["params"].as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default();
                let bulk_body = api["body"]["serialize"].as_str() == Some("bulk");
                apis.insert(name.clone(), Api { paths, params, bulk_body });
            }
        }
        Ok(Specs { apis, common })
    }
}
