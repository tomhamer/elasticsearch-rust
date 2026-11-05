//! `user_agent` processor over the uap-core `regexes.yml` that ES's
//! `ingest-user-agent` module ships (Apache-2.0, copied verbatim into
//! `useragent_data/`), plus its `device_type_regexes.yml`.
//!
//! Custom regex files (`regex_file`) are `*.yml` files in
//! `$ES_PATH_CONF/ingest-user-agent` or any directory of
//! `ES_USER_AGENT_DIR` (`:`-separated). Java patterns the `regex` crate
//! cannot compile are skipped (none of ES's shipped patterns need
//! look-around or back-references).

use super::doc::{type_name, IngestDocument};
use super::{Cfg, PResult, PipelineContext, Processor};
use crate::error::{EsError, Result};
use parking_lot::Mutex;
use regex::{Regex, RegexSet};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

const DEFAULT_REGEXES: &str = include_str!("useragent_data/regexes.yml");
const DEVICE_TYPE_REGEXES: &str = include_str!("useragent_data/device_type_regexes.yml");
pub const DEFAULT_PARSER_NAME: &str = "_default_";

struct SubPattern {
    re: Regex,
    name: Option<String>,
    v: [Option<String>; 4],
}

#[derive(Clone, Debug, Default)]
pub struct VersionedName {
    pub name: String,
    pub major: Option<String>,
    pub minor: Option<String>,
    pub patch: Option<String>,
    pub build: Option<String>,
}

/// Patterns tried in order; a `RegexSet` finds the first match quickly.
struct PatternList {
    pats: Vec<SubPattern>,
    set: Option<RegexSet>,
}

impl PatternList {
    fn new(pats: Vec<SubPattern>) -> PatternList {
        let set = regex::RegexSetBuilder::new(pats.iter().map(|p| p.re.as_str())).size_limit(1 << 28).dfa_size_limit(1 << 26).build().ok();
        PatternList { pats, set }
    }

    fn first_match(&self, s: &str) -> Option<VersionedName> {
        match &self.set {
            Some(set) => {
                for i in set.matches(s).iter() {
                    if let Some(v) = self.pats[i].matches(s) {
                        return Some(v);
                    }
                }
                None
            }
            None => self.pats.iter().find_map(|p| p.matches(s)),
        }
    }
}

/// `$1` replaced (first occurrence) by group 1 when it participated.
fn replace_first_group(rep: &str, g1: Option<&str>, groups: usize) -> String {
    match g1 {
        Some(g) if rep.contains("$1") && groups >= 1 => rep.replacen("$1", g, 1),
        _ => rep.to_string(),
    }
}

impl SubPattern {
    /// ES `UserAgentSubpattern.match`.
    fn matches(&self, s: &str) -> Option<VersionedName> {
        let c = self.re.captures(s)?;
        let groups = c.len() - 1;
        let g = |i: usize| c.get(i).map(|m| m.as_str().to_string());
        let name = match &self.name {
            Some(r) => Some(replace_first_group(r, c.get(1).map(|m| m.as_str()), groups)),
            None if groups >= 1 => g(1),
            None => None,
        };
        let pick = |k: usize| match &self.v[k] {
            Some(r) => Some(r.clone()),
            None if groups > k + 1 => g(k + 2),
            None => None,
        };
        let name = name?;
        Some(VersionedName { name, major: pick(0), minor: pick(1), patch: pick(2), build: pick(3) })
    }
}

/// Translate a Java pattern into `regex` syntax where they differ.
fn java_to_rust(p: &str) -> String {
    let mut out = String::with_capacity(p.len());
    let mut chars = p.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some(n) if n.is_ascii_punctuation() && !"\\.+*?()|[]{}^$#&-~".contains(n) => out.push(n),
                Some(n) => {
                    out.push('\\');
                    out.push(n);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn compile(regex: &str, flag: Option<&str>) -> Option<Regex> {
    let src = java_to_rust(regex);
    let src = if flag == Some("i") { format!("(?i){src}") } else { src };
    regex::RegexBuilder::new(&src).size_limit(1 << 24).build().ok()
}

fn yaml_str(v: &serde_yaml::Value) -> Option<String> {
    match v {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn read_list(doc: &serde_yaml::Value, key: &str, name_key: &str, v_keys: [&str; 4]) -> Vec<SubPattern> {
    let mut out = Vec::new();
    let Some(serde_yaml::Value::Sequence(seq)) = doc.get(key) else { return out };
    for m in seq {
        let get = |k: &str| m.get(k).and_then(yaml_str);
        let Some(re) = get("regex") else { continue };
        let Some(re) = compile(&re, get("regex_flag").as_deref()) else {
            tracing::debug!("user_agent: skipping uncompilable pattern in [{key}]");
            continue;
        };
        out.push(SubPattern { re, name: get(name_key), v: [v_keys[0], v_keys[1], v_keys[2], v_keys[3]].map(|k| if k.is_empty() { None } else { get(k) }) });
    }
    out
}

/// ES `DeviceTypeParser`.
struct DeviceTypes {
    os: Vec<(Regex, String)>,
    browser: Vec<(Regex, String)>,
    device: Vec<(Regex, String)>,
    agent: Vec<(Regex, String)>,
}

impl DeviceTypes {
    fn load() -> DeviceTypes {
        let doc: serde_yaml::Value = serde_yaml::from_str(DEVICE_TYPE_REGEXES).unwrap_or(serde_yaml::Value::Null);
        let list = |k: &str| -> Vec<(Regex, String)> {
            match doc.get(k) {
                Some(serde_yaml::Value::Sequence(s)) => s
                    .iter()
                    .filter_map(|m| {
                        let re = compile(&m.get("regex").and_then(yaml_str)?, None)?;
                        Some((re, m.get("replacement").and_then(yaml_str)?))
                    })
                    .collect(),
                _ => Vec::new(),
            }
        };
        DeviceTypes { os: list("os_parsers"), browser: list("browser_parsers"), device: list("device_parsers"), agent: list("agent_string_parsers") }
    }

    fn find(list: &[(Regex, String)], s: &str) -> Option<String> {
        for (re, rep) in list {
            if let Some(c) = re.captures(s) {
                return Some(replace_first_group(rep, c.get(1).map(|m| m.as_str()), c.len() - 1));
            }
        }
        None
    }

    fn device_type(&self, ua: &str, agent: Option<&VersionedName>, os: Option<&VersionedName>, device: Option<&VersionedName>) -> String {
        if let Some(t) = Self::find(&self.agent, ua) {
            return t;
        }
        let mut found = Vec::new();
        if let Some(o) = os {
            found.extend(Self::find(&self.os, &o.name));
        }
        if let Some(a) = agent {
            found.extend(Self::find(&self.browser, &a.name));
        }
        if let Some(d) = device {
            found.extend(Self::find(&self.device, &d.name));
        }
        for t in ["Robot", "Tablet", "Phone", "Desktop"] {
            if found.iter().any(|x| x == t) {
                return t.to_string();
            }
        }
        "Other".into()
    }
}

/// One regex file.
pub struct UserAgentParser {
    ua: PatternList,
    os: PatternList,
    device: PatternList,
    cache: Mutex<lru::LruCache<String, Arc<Details>>>,
}

#[derive(Clone, Debug)]
pub struct Details {
    pub agent: Option<VersionedName>,
    pub os: Option<VersionedName>,
    pub device: Option<VersionedName>,
    pub device_type: Option<String>,
}

fn device_types() -> &'static DeviceTypes {
    static D: OnceLock<DeviceTypes> = OnceLock::new();
    D.get_or_init(DeviceTypes::load)
}

impl UserAgentParser {
    fn from_yaml(src: &str) -> std::result::Result<UserAgentParser, String> {
        let doc: serde_yaml::Value = serde_yaml::from_str(src).map_err(|_| "error parsing regular expression file".to_string())?;
        let ua = read_list(&doc, "user_agent_parsers", "family_replacement", ["v1_replacement", "v2_replacement", "v3_replacement", "v4_replacement"]);
        let os = read_list(&doc, "os_parsers", "os_replacement", ["os_v1_replacement", "os_v2_replacement", "os_v3_replacement", "os_v4_replacement"]);
        let device = read_list(&doc, "device_parsers", "device_replacement", ["", "", "", ""]);
        if ua.is_empty() && os.is_empty() && device.is_empty() {
            return Err("not a valid regular expression file".into());
        }
        Ok(UserAgentParser {
            ua: PatternList::new(ua),
            os: PatternList::new(os),
            device: PatternList::new(device),
            cache: Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(1000).unwrap())),
        })
    }

    pub fn parse(&self, s: &str, extract_device_type: bool) -> Arc<Details> {
        let key = format!("{}{s}", if extract_device_type { "1" } else { "0" });
        if let Some(d) = self.cache.lock().get(&key) {
            return d.clone();
        }
        let agent = self.ua.first_match(s);
        let os = self.os.first_match(s);
        let device = self.device.first_match(s);
        let device_type = if extract_device_type { Some(device_types().device_type(s, agent.as_ref(), os.as_ref(), device.as_ref())) } else { None };
        let d = Arc::new(Details { agent, os, device, device_type });
        self.cache.lock().put(key, d.clone());
        d
    }
}

fn custom_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(s) = std::env::var("ES_USER_AGENT_DIR") {
        v.extend(s.split(':').filter(|x| !x.is_empty()).map(PathBuf::from));
    }
    if let Ok(s) = std::env::var("ES_PATH_CONF") {
        v.push(PathBuf::from(s).join("ingest-user-agent"));
    }
    v
}

fn parsers() -> &'static Mutex<HashMap<String, Arc<UserAgentParser>>> {
    static P: OnceLock<Mutex<HashMap<String, Arc<UserAgentParser>>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The parser for `regex_file` (`_default_` = the shipped regexes).
pub fn parser(name: &str) -> std::result::Result<Option<Arc<UserAgentParser>>, String> {
    if let Some(p) = parsers().lock().get(name) {
        return Ok(Some(p.clone()));
    }
    let src = if name == DEFAULT_PARSER_NAME {
        DEFAULT_REGEXES.to_string()
    } else {
        if !name.ends_with(".yml") || name.contains('/') || name.contains('\\') {
            return Ok(None);
        }
        match custom_dirs().into_iter().map(|d| d.join(name)).find(|p| p.is_file()) {
            Some(p) => std::fs::read_to_string(p).map_err(|e| e.to_string())?,
            None => return Ok(None),
        }
    };
    let p = Arc::new(UserAgentParser::from_yaml(&src)?);
    parsers().lock().insert(name.to_string(), p.clone());
    Ok(Some(p))
}

/// ES `versionToString`.
fn version_string(v: &VersionedName) -> String {
    let mut s = String::new();
    let has = |x: &Option<String>| x.as_deref().map(|y| !y.is_empty()).unwrap_or(false);
    if has(&v.major) {
        s.push_str(v.major.as_deref().unwrap());
        if has(&v.minor) {
            s.push('.');
            s.push_str(v.minor.as_deref().unwrap());
            if has(&v.patch) {
                s.push('.');
                s.push_str(v.patch.as_deref().unwrap());
                if has(&v.build) {
                    s.push('.');
                    s.push_str(v.build.as_deref().unwrap());
                }
            }
        }
    }
    s
}

/// ES `UserAgentProcessor.Property`.
const PROPERTIES: &[&str] = &["NAME", "OS", "DEVICE", "ORIGINAL", "VERSION"];

pub struct UserAgentP {
    field: String,
    target: String,
    parser: Arc<UserAgentParser>,
    /// Indices into [`PROPERTIES`], ascending.
    properties: Vec<usize>,
    extract_device_type: bool,
    ignore_missing: bool,
}

impl Processor for UserAgentP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let ua = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None => return Err(doc.require(&self.field).err().unwrap().into()),
            Some(Value::Null) => return Err(EsError::illegal_argument(format!("field [{}] is null, cannot parse user-agent.", self.field)).into()),
            Some(Value::String(s)) => s.clone(),
            Some(other) => {
                return Err(EsError::illegal_argument(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(other))).into());
            }
        };
        let d = self.parser.parse(&ua, self.extract_device_type);
        let mut out = Map::new();
        for &p in &self.properties {
            match PROPERTIES[p] {
                "NAME" => {
                    out.insert("name".into(), Value::String(d.agent.as_ref().map(|a| a.name.clone()).unwrap_or_else(|| "Other".into())));
                }
                "VERSION" => {
                    if let Some(a) = d.agent.as_ref().filter(|a| a.major.is_some()) {
                        out.insert("version".into(), Value::String(version_string(a)));
                    }
                }
                "OS" => {
                    if let Some(o) = &d.os {
                        let mut m = Map::new();
                        m.insert("name".into(), Value::String(o.name.clone()));
                        if o.major.is_some() {
                            let v = version_string(o);
                            m.insert("version".into(), Value::String(v.clone()));
                            m.insert("full".into(), Value::String(format!("{} {v}", o.name)));
                        }
                        out.insert("os".into(), Value::Object(m));
                    }
                }
                "DEVICE" => {
                    let mut m = Map::new();
                    match &d.device {
                        Some(dv) => {
                            m.insert("name".into(), Value::String(dv.name.clone()));
                            if self.extract_device_type {
                                m.insert("type".into(), d.device_type.clone().map(Value::String).unwrap_or(Value::Null));
                            }
                        }
                        None => {
                            m.insert("name".into(), Value::String("Other".into()));
                            if self.extract_device_type {
                                m.insert("type".into(), Value::String(d.device_type.clone().unwrap_or_else(|| "Other".into())));
                            }
                        }
                    }
                    out.insert("device".into(), Value::Object(m));
                }
                _ => {
                    out.insert("original".into(), Value::String(ua.clone()));
                }
            }
        }
        doc.set(&self.target, Value::Object(out))?;
        Ok(())
    }
}

/// Build a `user_agent` processor (ES `UserAgentProcessor.Factory`).
pub fn build(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let target = cfg.str_or("target_field", "user_agent")?;
    let regex_file = cfg.str_or("regex_file", DEFAULT_PARSER_NAME)?;
    let names = cfg.opt_list_str("properties")?;
    let extract_device_type = cfg.bool_or("extract_device_type", false)?;
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    cfg.ignore(&["ecs"]);
    let parser = match parser(&regex_file) {
        Ok(Some(p)) => p,
        Ok(None) => return Err(cfg.err("regex_file", format!("regex file [{regex_file}] doesn't exist (has to exist at node startup)"))),
        Err(e) => return Err(EsError::new(400, "parse_exception", e)),
    };
    let properties = match names {
        None => (0..PROPERTIES.len()).collect(),
        Some(ns) => {
            let mut v = Vec::new();
            for n in ns {
                let up = n.to_uppercase();
                match PROPERTIES.iter().position(|p| *p == up) {
                    Some(i) => v.push(i),
                    None => return Err(cfg.err("properties", format!("illegal property value [{n}]. valid values are [{}]", PROPERTIES.join(", ")))),
                }
            }
            v.sort();
            v.dedup();
            v
        }
    };
    Ok(Box::new(UserAgentP { field, target, parser, properties, extract_device_type, ignore_missing }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_regexes() {
        let p = parser(DEFAULT_PARSER_NAME).unwrap().unwrap();
        let d = p.parse("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.149 Safari/537.36", true);
        let a = d.agent.as_ref().unwrap();
        assert_eq!(a.name, "Chrome");
        assert_eq!(version_string(a), "33.0.1750.149");
        let o = d.os.as_ref().unwrap();
        assert_eq!((o.name.as_str(), version_string(o).as_str()), ("Mac OS X", "10.9.2"));
        assert_eq!(d.device.as_ref().unwrap().name, "Mac");
        assert_eq!(d.device_type.as_deref(), Some("Desktop"));
        let d = p.parse("curl/7.64.1", false);
        assert_eq!(d.agent.as_ref().unwrap().name, "curl");
    }
}
