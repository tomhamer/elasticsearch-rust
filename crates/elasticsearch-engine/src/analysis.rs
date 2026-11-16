//! Per-index analysis registry: built-in analyzers plus the custom
//! analyzers, tokenizers, token filters, char filters and normalizers
//! declared under `index.analysis.*`.
//!
//! Core indexes `text` fields with a *built-in analyzer name* only
//! (`Field::analyzer`). For any other analyzer the engine analyzes the text
//! itself and hands core the tokens joined by spaces under the
//! `whitespace` analyzer (see [`IndexAnalyzer`]); `docs/ES_CORE_API_WANTS.md`
//! records the core hook that would remove that detour.

use crate::error::{EsError, Result};
use rustic_core::analysis::{Analyzer, CharFilter, Token, TokenFilter, Tokenizer};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

/// How `synonyms_set` filters are resolved while a registry is built.
///
/// ES loads a synonyms set from the `.synonyms` system index when a shard
/// recovers or the analyzers are reloaded (`IndexCreationContext.RELOAD_ANALYZERS`),
/// never while the create / update-settings request validates the analysis
/// settings (`CREATE_INDEX`: the filter is built empty).
#[derive(Clone, Debug, Default)]
pub struct AnalysisCtx {
    /// Data directory of the node whose synonyms sets apply (`None`: no set is known).
    pub data_dir: Option<PathBuf>,
    /// Load the sets (recovery / reload); `false` builds every `synonyms_set` filter empty.
    pub load_sets: bool,
}

/// Built-in analyzer names core resolves itself.
pub fn is_core_builtin(name: &str) -> bool {
    Analyzer::by_name(name).is_some()
}

/// How a text field is analyzed at index time.
#[derive(Clone)]
pub enum IndexAnalyzer {
    /// Core analyzes with this built-in name.
    Builtin(String),
    /// The engine pre-analyzes; core sees space-joined tokens.
    Custom(Arc<Analyzer>),
}

#[derive(Clone, Default)]
pub struct AnalysisRegistry {
    /// Accept analyzer / normalizer names without resolving them (re-reading
    /// definitions already validated against the index's own registry).
    pub(crate) lenient_names: bool,
    analyzers: HashMap<String, Arc<Analyzer>>,
    normalizers: HashMap<String, Arc<Analyzer>>,
    /// Custom analyzer names (need pre-analysis at index time).
    custom: HashSet<String>,
    /// Analyzers declared as a parameterless built-in type (`{"type": "english"}`).
    aliases: HashMap<String, String>,
    /// `index.mapping.ignore_above`: the default for keyword / flattened fields.
    /// `index.mapping.ignore_malformed`: the default for the field types
    /// that support `ignore_malformed`.
    pub default_ignore_malformed: bool,
    /// `index.mapping.coerce: false`: the default `coerce` of the fields
    /// that take one (`FieldMapper.COERCE_SETTING`, default true; stored
    /// negated so `Default` keeps coercion on).
    pub default_no_coerce: bool,
    /// `index.downsample.interval` of a downsampled index.
    pub downsample_interval: Option<String>,
    /// `index.mapping.dynamic_strings.auto_text: false` (the default in the
    /// columnar index modes): dynamic strings map to `keyword` only.
    pub dynamic_strings_keyword_only: bool,
    /// `index.version.created` (`None` while the index is being created:
    /// the current version).
    pub created: Option<crate::version::IndexVersion>,
    pub default_ignore_above: Option<usize>,
    /// `index.similarity.*`: the index's scoring similarities.
    pub sims: crate::similarity::SimConfig,
    /// `index.mode` (`standard` when unset): the `_index_mode` metadata field.
    pub index_mode: Option<String>,
    /// `index.use_time_series_doc_values_format` (default: the columnar
    /// index modes `time_series`, `logsdb`, `columnar`, `logsdb_columnar`).
    pub ts_doc_values_format: bool,
    /// `index.mapping.doc_values.multi_value` / `.nullability` (default true).
    pub dv_multi_value: bool,
    pub dv_nullability: bool,
    /// `index.mapping.index_disabled_by_default` (default: strict columnar).
    pub index_disabled_by_default: bool,
    /// `index.mapping.use_doc_values_skipper` (default: time series and
    /// logsdb / strict columnar indices of recent versions).
    pub use_dv_skipper: bool,
    /// The index sort fields (`index.sort.field`, else the mode's default).
    pub sort_fields: Vec<String>,
    /// Synthetic `_id` (`index.mapping.synthetic_id`, a time series index
    /// sorted by `_tsid`, `@timestamp`): sealed segments derive `_id` from
    /// columns ([`rustic_core::synthetic_id`]).
    pub synthetic_id: bool,
    /// `index.version.created`: the version-keyed mapping defaults (a new
    /// index, whose settings do not carry it yet, is the current version).
    pub created_version: crate::version::IndexVersion,
    /// `index.mapping.exclude_source_vectors` (the `sparse_vector` `store`
    /// default).
    pub exclude_source_vectors: bool,
    /// The index stores `ignore_malformed` values in sorted binary doc
    /// values (`IndexVersions.STORE_IGNORED_MALFORMED_IN_BINARY_DOC_VALUES`):
    /// synthetic source emits them in encoded-bytes order.
    pub malformed_sorted: bool,
}

impl std::fmt::Debug for AnalysisRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalysisRegistry").field("custom", &self.custom).finish()
    }
}

fn str_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        Some(Value::String(s)) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        _ => Vec::new(),
    }
}

fn num(v: &Value, key: &str, default: usize) -> usize {
    match v.get(key) {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(default as u64) as usize,
        Some(Value::String(s)) => s.parse().unwrap_or(default),
        _ => default,
    }
}

/// The node config directory analysis `*_path` settings resolve against
/// (ES `Environment.configDir()`: `$ES_PATH_CONF`, default `config`).
fn config_dir() -> PathBuf {
    PathBuf::from(std::env::var("ES_PATH_CONF").unwrap_or_else(|_| "config".into()))
}

/// ES `Analysis.getWordList`: the file named by `{key}_path` (resolved
/// against the config directory; one entry per line, blank lines and `#`
/// comments skipped) when set, else the inline `{key}` list; `None` when
/// neither is configured. A missing, unreadable or non-UTF-8 file fails
/// the analyzer instead of building an empty word list (finding 303).
fn word_list(def: &Value, key: &str) -> Result<Option<Vec<String>>> {
    if let Some(p) = def.get(format!("{key}_path")).and_then(Value::as_str) {
        let path = config_dir().join(p);
        let bytes = std::fs::read(&path).map_err(|_| EsError::illegal_argument(format!("IOException while reading {key}_path: {}", path.display())))?;
        let text = String::from_utf8(bytes)
            .map_err(|_| EsError::illegal_argument(format!("Unsupported character encoding detected while reading {key}_path: {} - files must be UTF-8 encoded", path.display())))?;
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        return Ok(Some(text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string).collect()));
    }
    Ok(def.get(key).map(|v| str_list(Some(v))))
}

impl AnalysisRegistry {
    /// A default registry accepting any analyzer / normalizer name.
    pub(crate) fn lenient() -> Self {
        AnalysisRegistry { lenient_names: true, ..AnalysisRegistry::default() }
    }

    /// Build from the nested `index.analysis` settings object (may be null);
    /// `synonyms_set` filters are built empty (see [`AnalysisCtx`]).
    pub fn from_settings(analysis: &Value) -> Result<Self> {
        Self::from_settings_in(analysis, &AnalysisCtx::default())
    }

    /// [`AnalysisRegistry::from_settings`] resolving `synonyms_set` filters per `ctx`.
    pub fn from_settings_in(analysis: &Value, ctx: &AnalysisCtx) -> Result<Self> {
        let mut reg = AnalysisRegistry::default();
        let empty = serde_json::Map::new();
        let section = |name: &str| analysis.get(name).and_then(Value::as_object).unwrap_or(&empty).clone();
        let char_filters = section("char_filter");
        let tokenizers = section("tokenizer");
        let filters = section("filter");

        let resolve_cf = |name: &str| -> Result<Box<dyn CharFilter>> {
            if let Some(def) = char_filters.get(name) {
                return build_char_filter(name, def);
            }
            global_char_filter(name).ok_or_else(|| EsError::illegal_argument(format!("failed to find global char_filter under [{name}]")))
        };
        let resolve_tok = |name: &str| -> Result<Box<dyn Tokenizer>> {
            if let Some(def) = tokenizers.get(name) {
                return build_tokenizer(name, def);
            }
            global_tokenizer(name).ok_or_else(|| EsError::illegal_argument(format!("failed to find global tokenizer under [{name}]")))
        };
        let resolve_tf_pre = |name: &str, pre: Option<&Analyzer>| -> Result<Box<dyn TokenFilter>> {
            if let Some(def) = filters.get(name) {
                return build_token_filter_in(name, def, pre, ctx);
            }
            Analyzer::token_filter_by_name(name).ok_or_else(|| EsError::illegal_argument(format!("failed to find global filter under [{name}]")))
        };
        let resolve_tf = |name: &str| resolve_tf_pre(name, None);

        for (name, def) in section("analyzer") {
            let ty = def.get("type").and_then(Value::as_str).unwrap_or(if def.get("tokenizer").is_some() { "custom" } else { "" });
            let analyzer = match ty {
                "custom" => {
                    let tok = def.get("tokenizer").and_then(Value::as_str).ok_or_else(|| {
                        EsError::illegal_argument(format!("analyzer [{name}] must specify either an analyzer type, or a tokenizer"))
                    })?;
                    let mut a = Analyzer::new(resolve_tok(tok)?);
                    for cf in str_list(def.get("char_filter")) {
                        a = a.with_char_filter(resolve_cf(&cf)?);
                    }
                    // Synonym rules are analyzed with the chain built so
                    // far, in which earlier synonym filters are the
                    // identity (ES `buildSynonymAnalyzer`, finding 423).
                    let mut rules = a.clone();
                    for tf in str_list(def.get("filter")) {
                        let f = resolve_tf_pre(&tf, Some(&rules))?;
                        if !filters.get(&tf).is_some_and(is_synonym_def) {
                            rules = rules.with_filter(f.clone_box());
                        }
                        a = a.with_filter(f);
                    }
                    a
                }
                "" => return Err(EsError::illegal_argument(format!("analyzer [{name}] must specify either an analyzer type, or a tokenizer"))),
                builtin => {
                    let base = Analyzer::by_name(builtin).ok_or_else(|| EsError::illegal_argument(format!("Unknown analyzer type [{builtin}] for [{name}]")))?;
                    match def.get("stopwords") {
                        Some(sw) if matches!(builtin, "standard" | "stop" | "simple" | "pattern") => {
                            let tok = if builtin == "standard" { "standard" } else { "letter" };
                            let stop = build_token_filter(&name, &serde_json::json!({"type": "stop", "stopwords": sw, "ignore_case": true}))?;
                            Analyzer::new(Analyzer::tokenizer_by_name(tok).unwrap())
                                .with_filter(Analyzer::token_filter_by_name("lowercase").unwrap())
                                .with_filter(stop)
                        }
                        // A language analyzer built from its configured stop
                        // words (inline, a named set, `_none_` or a file) and
                        // stem exclusions, which were accepted and ignored
                        // (ES `<Language>AnalyzerProvider`, SIDE 006).
                        _ if rustic_core::analysis::Analyzer::LANGUAGE_ANALYZERS.contains(&builtin)
                            && (def.get("stopwords").is_some() || def.get("stopwords_path").is_some() || def.get("stem_exclusion").is_some()) =>
                        {
                            let stop = if def.get("stopwords").is_some() || def.get("stopwords_path").is_some() {
                                let mut sdef = serde_json::json!({"type": "stop"});
                                for k in ["stopwords", "stopwords_path"] {
                                    if let Some(v) = def.get(k) {
                                        sdef[k] = v.clone();
                                    }
                                }
                                Some(build_token_filter(&name, &sdef)?)
                            } else {
                                None
                            };
                            let excl = str_list(def.get("stem_exclusion"));
                            Analyzer::language_with(builtin, stop, &excl).unwrap_or(base)
                        }
                        _ => base,
                    }
                }
            };
            let params = def.as_object().map(|m| m.keys().filter(|k| *k != "type").count()).unwrap_or(0);
            if params == 0 && ty != "custom" && is_core_builtin(ty) {
                reg.aliases.insert(name.clone(), ty.to_string());
            } else {
                reg.custom.insert(name.clone());
            }
            reg.analyzers.insert(name.clone(), Arc::new(analyzer));
        }
        // `pattern_text`'s `delimiter` analyzer (`DelimiterAnalyzer`).
        if let Some(tok) = build_tokenizer("delimiter", &serde_json::json!({"type": "char_group", "tokenize_on_chars": ["whitespace", "=", "?", ":", "[", "]", "{", "}", "\"", "\\", "'"]})).ok() {
            if let Some(lc) = Analyzer::token_filter_by_name("lowercase") {
                let a = Analyzer::new(tok).with_filter(lc);
                reg.custom.insert(crate::fieldtypes::pattern_text::DELIMITER_ANALYZER.to_string());
                reg.analyzers.insert(crate::fieldtypes::pattern_text::DELIMITER_ANALYZER.to_string(), Arc::new(a));
            }
        }
        for (name, def) in section("normalizer") {
            let mut a = Analyzer::new(Analyzer::tokenizer_by_name("keyword").unwrap());
            for cf in str_list(def.get("char_filter")) {
                a = a.with_char_filter(resolve_cf(&cf)?);
            }
            for tf in str_list(def.get("filter")) {
                a = a.with_filter(resolve_tf(&tf)?);
            }
            reg.normalizers.insert(name.clone(), Arc::new(a));
        }
        Ok(reg)
    }

    /// Resolve the built-in analyzer `name` as core's `legacy` variant
    /// (unless the index defines its own analyzer under that name).
    pub fn legacy_builtin(&mut self, name: &str, legacy: &str) {
        if self.analyzers.contains_key(name) {
            return;
        }
        if let Some(a) = Analyzer::by_name(legacy) {
            self.aliases.insert(name.to_string(), legacy.to_string());
            self.analyzers.insert(name.to_string(), Arc::new(a));
        }
    }

    /// Analyzer by name: custom first, then built-in.
    pub fn analyzer(&self, name: &str) -> Option<Arc<Analyzer>> {
        if let Some(a) = self.analyzers.get(name) {
            return Some(a.clone());
        }
        Analyzer::by_name(name).map(Arc::new)
    }

    pub fn has_analyzer(&self, name: &str) -> bool {
        self.analyzers.contains_key(name) || is_core_builtin(name) || matches!(name, "default" | "default_search" | "default_search_quote")
    }

    /// The `default` analyzer (custom override or `standard`).
    pub fn default_analyzer_name(&self) -> &str {
        if self.analyzers.contains_key("default") { "default" } else { "standard" }
    }

    pub fn default_search_analyzer_name(&self) -> &str {
        if self.analyzers.contains_key("default_search") { "default_search" } else { self.default_analyzer_name() }
    }

    /// Index-time handling for an analyzer name.
    pub fn index_analyzer(&self, name: &str) -> IndexAnalyzer {
        if let Some(b) = self.aliases.get(name) {
            IndexAnalyzer::Builtin(b.clone())
        } else if self.custom.contains(name) {
            IndexAnalyzer::Custom(self.analyzers[name].clone())
        } else if is_core_builtin(name) {
            IndexAnalyzer::Builtin(name.to_string())
        } else {
            IndexAnalyzer::Builtin("standard".into())
        }
    }

    /// The index's creation version (a new index: the current one).
    pub fn created_version(&self) -> crate::version::IndexVersion {
        self.created.unwrap_or(crate::version::IndexVersion::CURRENT)
    }

    /// A strict columnar index mode (`columnar`, `logsdb_columnar`).
    pub fn strict_columnar(&self) -> bool {
        matches!(self.index_mode.as_deref().map(str::to_ascii_lowercase).as_deref(), Some("columnar" | "logsdb_columnar"))
    }

    /// The index mode name as ES prints it (`columnar`).
    pub fn index_mode_name(&self) -> String {
        self.index_mode.as_deref().unwrap_or("standard").to_ascii_lowercase()
    }

    /// Whether `name` is an index-defined (custom) normalizer.
    pub fn is_custom_normalizer(&self, name: &str) -> bool {
        self.normalizers.contains_key(name)
    }

    pub fn normalizer(&self, name: &str) -> Option<Arc<Analyzer>> {
        if let Some(n) = self.normalizers.get(name) {
            return Some(n.clone());
        }
        match name {
            "lowercase" => Analyzer::normalizer(&["lowercase"]).map(Arc::new),
            _ => None,
        }
    }

    /// Build an ad-hoc analyzer for `_analyze` (tokenizer + filters + char
    /// filters given inline as names or definitions).
    pub fn adhoc(&self, tokenizer: &Value, filters: &[Value], char_filters: &[Value]) -> Result<Analyzer> {
        let tok = match tokenizer {
            Value::String(s) => global_tokenizer(s).ok_or_else(|| EsError::illegal_argument(format!("failed to find global tokenizer under [{s}]")))?,
            Value::Object(_) => build_tokenizer("_anonymous_tokenizer", tokenizer)?,
            _ => Analyzer::tokenizer_by_name("keyword").unwrap(),
        };
        let mut a = Analyzer::new(tok);
        for cf in char_filters {
            a = a.with_char_filter(match cf {
                Value::String(s) => global_char_filter(s).ok_or_else(|| EsError::illegal_argument(format!("failed to find global char_filter under [{s}]")))?,
                other => build_char_filter("_anonymous_charfilter", other)?,
            });
        }
        // Inline synonym rules are analyzed by the chain before them, as in
        // a configured analyzer (earlier synonym filters as the identity).
        let mut rules = a.clone();
        for f in filters {
            let built = match f {
                Value::String(s) => Analyzer::token_filter_by_name(s).ok_or_else(|| EsError::illegal_argument(format!("failed to find global filter under [{s}]")))?,
                other => build_token_filter_in("_anonymous_tokenfilter", other, Some(&rules), &AnalysisCtx::default())?,
            };
            if !is_synonym_def(f) {
                rules = rules.with_filter(built.clone_box());
            }
            a = a.with_filter(built);
        }
        Ok(a)
    }
}

/// Whether a filter definition is a synonym filter.
fn is_synonym_def(def: &Value) -> bool {
    matches!(def.get("type").and_then(Value::as_str), Some("synonym" | "synonym_graph"))
}

/// A tokenizer registered by name: core's built-ins plus the ML plugin's.
fn global_tokenizer(name: &str) -> Option<Box<dyn Tokenizer>> {
    Analyzer::tokenizer_by_name(name).or_else(|| crate::ml::mltok::tokenizer_by_name(name))
}

/// A char filter registered by name: core's built-ins plus the ML plugin's.
fn global_char_filter(name: &str) -> Option<Box<dyn CharFilter>> {
    Analyzer::char_filter_by_name(name).or_else(|| crate::ml::mltok::char_filter_by_name(name))
}

fn build_tokenizer(name: &str, def: &Value) -> Result<Box<dyn Tokenizer>> {
    let ty = def.get("type").and_then(Value::as_str).unwrap_or("");
    match ty {
        "ngram" | "nGram" | "edge_ngram" | "edgeNGram" => {
            let min = num(def, "min_gram", 1);
            let max = num(def, "max_gram", 2);
            let chars: Vec<String> = str_list(def.get("token_chars"));
            Ok(Box::new(NGramTok { min, max, edge: ty.starts_with('e'), token_chars: chars }))
        }
        "pattern" => {
            // ES `PatternTokenizerFactory`: `pattern`, Java `flags`
            // (`CASE_INSENSITIVE|COMMENTS`), and `group`: -1 splits on the
            // matches, N >= 0 emits capture group N of each match.
            let pat = def.get("pattern").and_then(Value::as_str).unwrap_or("\\W+");
            let re = regex::Regex::new(&format!("{}{pat}", java_regex_flags(def.get("flags").and_then(Value::as_str).unwrap_or(""))?))
                .map_err(|e| EsError::illegal_argument(format!("invalid pattern for tokenizer [{name}]: {e}")))?;
            let group = match def.get("group") {
                None | Some(Value::Null) => -1,
                Some(v) => v.as_i64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).ok_or_else(|| EsError::illegal_argument(format!("Failed to parse value [{v}] for setting [group]")))?,
            };
            if group >= re.captures_len() as i64 {
                return Err(EsError::illegal_argument(format!("invalid group [{group}] for tokenizer [{name}]: pattern [{pat}] has {} capturing groups", re.captures_len() - 1)));
            }
            Ok(Box::new(PatternTok { re, group: usize::try_from(group).ok() }))
        }
        "char_group" => {
            let chars = str_list(def.get("tokenize_on_chars"));
            Ok(Box::new(CharGroupTok { on: chars }))
        }
        other => global_tokenizer(other).ok_or_else(|| EsError::illegal_argument(format!("Unknown tokenizer type [{other}] for [{name}]"))),
    }
}

fn build_token_filter(name: &str, def: &Value) -> Result<Box<dyn TokenFilter>> {
    build_token_filter_in(name, def, None, &AnalysisCtx::default())
}

/// `pre`: the analyzer chain before this filter (used to analyze synonym rules).
fn build_token_filter_in(name: &str, def: &Value, pre: Option<&Analyzer>, ctx: &AnalysisCtx) -> Result<Box<dyn TokenFilter>> {
    let ty = def.get("type").and_then(Value::as_str).unwrap_or("");
    let flag = |k: &str| def.get(k).and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true")));
    match ty {
        "stop" => {
            let ignore_case = def.get("ignore_case").and_then(Value::as_bool).unwrap_or(false);
            match def.get("stopwords") {
                Some(Value::String(s)) if s.starts_with('_') => {
                    let lang = s.trim_matches('_');
                    if lang == "none" {
                        return Ok(Box::new(StopWords { words: Arc::new(HashSet::new()), ignore_case }));
                    }
                    Analyzer::token_filter_by_name(&format!("{lang}_stop")).ok_or_else(|| EsError::illegal_argument(format!("unknown stopword list [{s}]")))
                }
                None => match word_list(def, "stopwords")? {
                    Some(words) => Ok(Box::new(StopWords { words: Arc::new(words.into_iter().map(|w| if ignore_case { w.to_lowercase() } else { w }).collect()), ignore_case })),
                    None => Ok(Analyzer::token_filter_by_name("stop").unwrap()),
                },
                other => Ok(Box::new(StopWords { words: Arc::new(str_list(other).into_iter().map(|w| if ignore_case { w.to_lowercase() } else { w }).collect()), ignore_case })),
            }
        }
        "synonym" | "synonym_graph" => {
            // ES `SynonymTokenFilterFactory`: updateable synonyms are lenient
            // unless told otherwise (8.16+); index-time synonyms are strict.
            let updateable = flag("updateable").unwrap_or(false);
            let lenient = flag("lenient").unwrap_or(updateable);
            let expand = flag("expand").unwrap_or(true);
            let mut rules = str_list(def.get("synonyms"));
            let mut source = format!("['{name}' analyzer settings]");
            if def.get("synonyms").is_none() && def.get("synonyms_set").is_none_or(Value::is_null) {
                if let Some(file) = word_list(def, "synonyms")? {
                    rules = file;
                    source = format!("[{}]", def.get("synonyms_path").and_then(Value::as_str).unwrap_or_default());
                }
            }
            if let Some(spec) = def.get("synonyms_set").filter(|v| !v.is_null()) {
                if !updateable {
                    return Err(EsError::illegal_argument("Can't apply [synonyms_set]! Loading synonyms from index is supported only for search time synonyms!"));
                }
                // Synonyms API sets (`PUT _synonyms/{set}`), one or a list
                // (`synonyms.multiple_sets_per_filter`), loaded at recovery /
                // reload time; duplicates are ignored.
                let mut sets: Vec<String> = Vec::new();
                for n in str_list(Some(spec)) {
                    if !sets.contains(&n) {
                        sets.push(n);
                    }
                }
                if sets.len() > 100 {
                    return Err(EsError::illegal_argument("At most 100 synonym sets may be specified in [synonyms_set]"));
                }
                let shown = format!("[{}]", sets.join(", "));
                source = format!("[{shown} synonyms_sets in .synonyms index]");
                rules = Vec::new();
                if let (Some(data), true, false) = (&ctx.data_dir, ctx.load_sets, sets.is_empty()) {
                    // `SynonymsManagementAPIService.getSynonymSetRules`: missing
                    // sets fail the filter unless it is lenient (and all
                    // missing always fails, which a lenient filter tolerates
                    // with no rules).
                    let found: Vec<(String, Option<Vec<String>>)> = sets.iter().map(|n| (n.clone(), crate::search_ext::synonym_set_rules(data, n))).collect();
                    let missing: Vec<&str> = found.iter().filter(|(_, r)| r.is_none()).map(|(n, _)| n.as_str()).collect();
                    if !missing.is_empty() && !lenient {
                        let which = if missing.len() == sets.len() { shown.clone() } else { format!("[{}]", missing.join(", ")) };
                        return Err(EsError::resource_not_found(format!("synonyms sets {which} not found")));
                    }
                    if missing.len() < sets.len() {
                        rules = found.into_iter().filter_map(|(_, r)| r).flatten().collect();
                    }
                }
            }
            // `format`: `solr` (default) or `wordnet` (finding 422).
            let format = def.get("format").and_then(Value::as_str).unwrap_or("solr");
            let wordnet = match format {
                "solr" => false,
                "wordnet" => true,
                other => return Err(EsError::illegal_argument(format!("failed to build synonyms: unknown synonym format [{other}]"))),
            };
            Ok(Box::new(Synonyms::parse(&rules, expand, pre, lenient, &source, wordnet)?))
        }
        // Parameterized built-ins (finding 306): a named or inline
        // definition's options reach the filter instead of the type's
        // defaults.
        "limit" => Ok(Box::new(rustic_core::analysis::filter::LimitTokenCountFilter {
            max_token_count: num(def, "max_token_count", 1),
            consume_all: flag("consume_all_tokens").unwrap_or(false),
        })),
        "fingerprint" => {
            let sep = def.get("separator").and_then(Value::as_str).unwrap_or(" ");
            let mut chars = sep.chars();
            let separator = match (chars.next(), chars.next()) {
                (Some(c), None) => c,
                _ => return Err(EsError::illegal_argument(format!("Setting [separator] must be a single, non-null character. [{sep}] was provided."))),
            };
            Ok(Box::new(rustic_core::analysis::filter::FingerprintFilter { separator, max_output_size: num(def, "max_output_size", 255) }))
        }
        "lowercase" => match def.get("language").and_then(Value::as_str) {
            None => Ok(Analyzer::token_filter_by_name("lowercase").unwrap()),
            Some(lang) => match lang.to_lowercase().as_str() {
                l @ ("greek" | "irish" | "turkish") => Ok(Analyzer::token_filter_by_name(&format!("{l}_lowercase")).unwrap()),
                _ => Err(EsError::illegal_argument(format!("language [{lang}] not supported for lowercase"))),
            },
        },
        "elision" => {
            let articles = word_list(def, "articles")?.ok_or_else(|| EsError::illegal_argument("elision filter requires [articles] or [articles_path] setting"))?;
            let refs: Vec<&str> = articles.iter().map(String::as_str).collect();
            // ES `articles_case: true` matches case-insensitively; core's
            // flag means the opposite (match case-sensitively).
            Ok(Box::new(rustic_core::analysis::filter::ElisionFilter::new(&refs, !flag("articles_case").unwrap_or(false))))
        }
        "cjk_bigram" => {
            let ignored = str_list(def.get("ignored_scripts"));
            let refs: Vec<&str> = ignored.iter().map(String::as_str).collect();
            Ok(Box::new(rustic_core::analysis::filter::CjkBigramFilter::with_ignored_scripts(&refs, flag("output_unigrams").unwrap_or(false))))
        }
        "length" => Ok(Box::new(Length { min: num(def, "min", 0), max: num(def, "max", usize::MAX) })),
        "truncate" => Ok(Box::new(Truncate { len: num(def, "length", 10) })),
        "ngram" | "nGram" | "edge_ngram" | "edgeNGram" => Ok(Box::new(NGramFilter {
            min: num(def, "min_gram", 1),
            max: num(def, "max_gram", 2),
            edge: ty.starts_with('e'),
        })),
        "shingle" => {
            let min = num(def, "min_shingle_size", 2);
            let max = num(def, "max_shingle_size", 2);
            if min < 2 || max < min {
                return Err(EsError::illegal_argument(format!("failed to build filter [{name}]: [min_shingle_size] must be >= 2 and <= [max_shingle_size]")));
            }
            let flag = |k: &str, d: bool| def.get(k).and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or(d);
            let mut f = rustic_core::analysis::filter::ShingleFilter::new(min, max);
            f.output_unigrams = flag("output_unigrams", true);
            f.output_unigrams_if_no_shingles = flag("output_unigrams_if_no_shingles", false);
            if let Some(s) = def.get("token_separator").and_then(Value::as_str) {
                f.token_separator = s.to_string();
            }
            if let Some(s) = def.get("filler_token").and_then(Value::as_str) {
                f.filler_token = s.to_string();
            }
            Ok(Box::new(f))
        }
        "word_delimiter" | "word_delimiter_graph" => {
            let flag = |k: &str, d: bool| def.get(k).and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or(d);
            let mut f = rustic_core::analysis::filter::WordDelimiterFilter::default();
            f.generate_word_parts = flag("generate_word_parts", true);
            f.generate_number_parts = flag("generate_number_parts", true);
            f.catenate_words = flag("catenate_words", false);
            f.catenate_numbers = flag("catenate_numbers", false);
            f.catenate_all = flag("catenate_all", false);
            f.split_on_case_change = flag("split_on_case_change", true);
            f.split_on_numerics = flag("split_on_numerics", true);
            f.preserve_original = flag("preserve_original", false);
            f.stem_english_possessive = flag("stem_english_possessive", true);
            f.protected_words = str_list(def.get("protected_words")).into_iter().collect();
            f.graph = ty == "word_delimiter_graph";
            Ok(Box::new(f))
        }
        "keep" => {
            // ES `KeepWordFilterFactory`: exactly one word source.
            if def.get("keep_words").is_some() == def.get("keep_words_path").is_some() {
                return Err(EsError::illegal_argument("keep requires either `keep_words` or `keep_words_path` to be configured"));
            }
            let ignore_case = flag("keep_words_case").unwrap_or(false);
            let words = word_list(def, "keep_words")?.unwrap_or_default();
            Ok(Box::new(Keep { words: words.into_iter().map(|w| if ignore_case { w.to_lowercase() } else { w }).collect(), ignore_case }))
        }
        "pattern_replace" => {
            let pat = def.get("pattern").and_then(Value::as_str).unwrap_or("");
            let re = regex::Regex::new(pat).map_err(|e| EsError::illegal_argument(format!("invalid pattern for filter [{name}]: {e}")))?;
            Ok(Box::new(PatternReplaceTokens { re, replacement: def.get("replacement").and_then(Value::as_str).unwrap_or("").to_string() }))
        }
        "stemmer" | "snowball" => {
            use rustic_core::analysis::filter::{PersianStemFilter, SnowballFilter, StemmerFilter};
            let lang = def.get("language").or(def.get("name")).and_then(Value::as_str).unwrap_or(if ty == "snowball" { "English" } else { "english" });
            let lower = lang.to_lowercase();
            if ty == "stemmer" && lower == "german2" {
                crate::warnings::add("The 'german2' stemmer has been deprecated and folded into the 'german' Stemmer. Replace all usages of 'german2' with 'german'.");
            }
            let f: Option<Box<dyn TokenFilter>> = if ty == "snowball" {
                SnowballFilter::new(&lower).map(|f| Box::new(f) as Box<dyn TokenFilter>)
            } else if lower == "persian" {
                Some(Box::new(PersianStemFilter))
            } else {
                StemmerFilter::new(&lower).map(|f| Box::new(f) as Box<dyn TokenFilter>)
            };
            f.ok_or_else(|| {
                let mut c = lang.chars();
                let cap = c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default();
                EsError::illegal_argument(format!("Invalid stemmer class specified: {cap}"))
            })
        }
        // ES `KeywordMarkerTokenFilterFactory`: a word list (inline or a
        // file) or a pattern, not both; marked tokens are left alone by
        // later stemmers (P2 keyword_repeat / keyword_marker).
        "keyword_marker" => {
            let ignore_case = flag("ignore_case").unwrap_or(false);
            let pattern = def.get("keywords_pattern").and_then(Value::as_str);
            let has_words = def.get("keywords").is_some() || def.get("keywords_path").is_some();
            match (pattern, has_words) {
                (Some(_), true) => Err(EsError::illegal_argument("cannot specify both `keywords_pattern` and `keywords` or `keywords_path`")),
                (Some(p), false) => {
                    let re = regex_automata::meta::Regex::builder()
                        .syntax(regex_automata::util::syntax::Config::new().case_insensitive(ignore_case))
                        .build(&format!("^(?:{p})$"))
                        .map_err(|e| EsError::illegal_argument(format!("invalid keywords_pattern [{p}]: {e}")))?;
                    Ok(Box::new(rustic_core::analysis::filter::KeywordMarkerFilter::pattern(re)))
                }
                (None, true) => {
                    let words = word_list(def, "keywords")?.unwrap_or_default();
                    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
                    Ok(Box::new(rustic_core::analysis::filter::KeywordMarkerFilter::marker(&refs, ignore_case)))
                }
                (None, false) => Err(EsError::illegal_argument("keyword filter requires either `keywords`, `keywords_path`, or `keywords_pattern` to be configured")),
            }
        }
        other => Analyzer::token_filter_by_name(other).ok_or_else(|| EsError::illegal_argument(format!("Unknown filter type [{other}] for [{name}]"))),
    }
}

fn build_char_filter(name: &str, def: &Value) -> Result<Box<dyn CharFilter>> {
    let ty = def.get("type").and_then(Value::as_str).unwrap_or("");
    match ty {
        "mapping" => {
            let mut rules = Vec::new();
            let lines = word_list(def, "mappings")?.ok_or_else(|| EsError::illegal_argument("mapping requires either `mappings` or `mappings_path` to be configured"))?;
            // ES `MappingCharFilterFactory.parseRules`: every rule is
            // `lhs => rhs`; an invalid rule (or an empty left side) fails
            // the analyzer instead of being skipped (SIDE 017).
            for r in lines {
                let Some((a, b)) = r.split_once("=>") else {
                    return Err(EsError::illegal_argument(format!("Invalid Mapping Rule : [{r}]")));
                };
                let (lhs, rhs) = (unescape(a.trim())?, unescape(b.trim())?);
                if lhs.is_empty() {
                    return Err(EsError::illegal_argument(format!("Invalid Mapping Rule : [{r}]. Illegal mapping.")));
                }
                rules.push((lhs, rhs));
            }
            rules.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
            Ok(Box::new(MappingCf { rules }))
        }
        "pattern_replace" => {
            let pat = def.get("pattern").and_then(Value::as_str).unwrap_or("");
            let re = regex::Regex::new(pat).map_err(|e| EsError::illegal_argument(format!("invalid pattern for char_filter [{name}]: {e}")))?;
            Ok(Box::new(PatternReplaceCf { re, replacement: def.get("replacement").and_then(Value::as_str).unwrap_or("").replace("$", "$") }))
        }
        other => global_char_filter(other).ok_or_else(|| EsError::illegal_argument(format!("Unknown char_filter type [{other}] for [{name}]"))),
    }
}

/// ES `MappingCharFilterFactory.parseString`: `\\`, `\"`, `\'`, `\n`,
/// `\t`, `\r`, `\b`, `\f` and `\uXXXX` escapes.
fn unescape(s: &str) -> Result<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some(&e) = chars.get(i) else { return Err(EsError::illegal_argument(format!("Invalid escaped char in [{s}]"))) };
        i += 1;
        out.push(match e {
            '\\' => '\\',
            '"' => '"',
            '\'' => '\'',
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'b' => '\u{8}',
            'f' => '\u{c}',
            'u' => {
                let hex: String = chars.get(i..i + 4).map(|h| h.iter().collect()).ok_or_else(|| EsError::illegal_argument(format!("Invalid escaped char in [{s}]")))?;
                i += 4;
                let cp = u32::from_str_radix(&hex, 16).map_err(|_| EsError::illegal_argument(format!("Invalid escaped char in [{s}]")))?;
                char::from_u32(cp).ok_or_else(|| EsError::illegal_argument(format!("Invalid escaped char in [{s}]")))?
            }
            // Any other escaped char stands for itself.
            other => other,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Configurable components core does not expose with parameters.

#[derive(Clone)]
struct StopWords {
    words: Arc<HashSet<String>>,
    ignore_case: bool,
}
impl TokenFilter for StopWords {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .filter(|t| if self.ignore_case { !self.words.contains(&t.text.to_lowercase()) } else { !self.words.contains(&t.text) })
            .collect()
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }
}

/// Solr-format synonyms: `a, b, c` (equivalent; expand) and `a, b => c`,
/// multi-word on either side (`ny => new york`, `new york => ny`): matched
/// on token sequences and emitted as token sequences at consecutive
/// positions (later tokens shift), so a multi-word synonym is searchable
/// as its words and as a phrase.
#[derive(Clone)]
struct Synonyms {
    filter: rustic_core::analysis::filter::synonym::SynonymFilter,
}
impl Synonyms {
    /// Parse Solr-format rules, analyzing every word with `pre` (the chain
    /// before the filter). A word the chain drops (e.g. a stop word) is
    /// skipped when `lenient` (Lucene `ESSolrSynonymParser`), otherwise the
    /// build fails the way ES reports it: `failed to build synonyms from
    /// <source>` caused by `Invalid synonym rule at line N` caused by
    /// `term: <w> was completely eliminated by analyzer`.
    fn parse(rules: &[String], expand: bool, pre: Option<&Analyzer>, lenient: bool, source: &str, wordnet: bool) -> Result<Self> {
        let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
        let mut explicit: Vec<Vec<String>> = Vec::new();
        // One rule term: its words as the chain before the filter emits
        // them, exactly (finding 419: no extra case folding; a lowercase
        // filter in the chain makes rules case-insensitive). None: dropped.
        let norm = |w: &str| -> Result<Option<Vec<String>>> {
            let w = w.trim();
            if w.is_empty() {
                return Ok(None);
            }
            match pre {
                Some(a) => {
                    let toks: Vec<String> = a.analyze(w).into_iter().map(|t| t.text).collect();
                    if toks.is_empty() {
                        if lenient {
                            return Ok(None);
                        }
                        return Err(EsError::illegal_argument(format!("term: {w} was completely eliminated by analyzer")));
                    }
                    Ok(Some(toks))
                }
                None => Ok(Some(w.split_whitespace().map(str::to_string).collect())),
            }
        };
        let side = |words: &[String]| -> Result<Vec<Vec<String>>> {
            let mut out = Vec::new();
            for w in words {
                out.extend(norm(w)?);
            }
            Ok(out)
        };
        let fail = |line: usize, e: EsError| {
            EsError::illegal_argument(format!("failed to build synonyms from {source}"))
                .caused_by(EsError::new(400, "parse_exception", format!("Invalid synonym rule at line {line}")).caused_by(e))
        };
        // (line, left side or None for an equivalence group, words).
        let mut parsed: Vec<(usize, Option<Vec<String>>, Vec<String>)> = Vec::new();
        if wordnet {
            // Lucene `WordnetSynonymParser` (finding 422):
            // `s(<synset>,<n>,'<word>',...).`; consecutive lines of one
            // synset form one equivalence group.
            let mut cur: Option<(String, usize, Vec<String>)> = None;
            for (n, line) in rules.iter().flat_map(|r| r.lines()).enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let word = (|| {
                    let rest = line.strip_prefix("s(")?;
                    let id = rest.split(',').next()?.to_string();
                    let (a, b) = (line.find('\'')?, line.rfind('\'')?);
                    (b > a).then(|| (id, line[a + 1..b].replace("''", "'")))
                })();
                let Some((id, w)) = word else {
                    return Err(fail(n + 1, EsError::illegal_argument(format!("invalid WordNet synonym declaration: {line}"))));
                };
                match &mut cur {
                    Some((cid, _, words)) if *cid == id => words.push(w),
                    _ => {
                        if let Some((_, l, words)) = cur.take() {
                            parsed.push((l, None, words));
                        }
                        cur = Some((id, n + 1, vec![w]));
                    }
                }
            }
            if let Some((_, l, words)) = cur {
                parsed.push((l, None, words));
            }
        } else {
            for (line, r) in rules.iter().enumerate() {
                match r.split_once("=>") {
                    Some((lhs, rhs)) => parsed.push((line + 1, Some(lhs.split(',').map(str::to_string).collect()), rhs.split(',').map(str::to_string).collect())),
                    None => parsed.push((line + 1, None, r.split(',').map(str::to_string).collect())),
                }
            }
        }
        for (line, lhs, rhs) in parsed {
            let r = (|| -> Result<()> {
                match &lhs {
                    Some(lhs) => {
                        let outs = side(&rhs)?;
                        if outs.is_empty() {
                            return Ok(());
                        }
                        for l in side(lhs)? {
                            explicit.push(l.clone());
                            pairs.extend(outs.iter().map(|o| (l.clone(), o.clone())));
                        }
                    }
                    None => {
                        let words = side(&rhs)?;
                        if words.is_empty() {
                            return Ok(());
                        }
                        for w in &words {
                            if expand {
                                pairs.extend(words.iter().map(|o| (w.clone(), o.clone())));
                            } else {
                                pairs.push((w.clone(), words[0].clone()));
                            }
                        }
                    }
                }
                Ok(())
            })();
            r.map_err(|e| fail(line, e))?;
        }
        Ok(Synonyms { filter: rustic_core::analysis::filter::synonym::SynonymFilter::from_pairs(pairs, explicit) })
    }
}
impl TokenFilter for Synonyms {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        self.filter.filter(tokens)
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct Length {
    min: usize,
    max: usize,
}
impl TokenFilter for Length {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens.into_iter().filter(|t| { let n = t.text.chars().count(); n >= self.min && n <= self.max }).collect()
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct Truncate {
    len: usize,
}
impl TokenFilter for Truncate {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens.into_iter().map(|mut t| { t.text = t.text.chars().take(self.len).collect(); t }).collect()
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct Keep {
    words: HashSet<String>,
    ignore_case: bool,
}
impl TokenFilter for Keep {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens.into_iter().filter(|t| if self.ignore_case { self.words.contains(&t.text.to_lowercase()) } else { self.words.contains(&t.text) }).collect()
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct PatternReplaceTokens {
    re: regex::Regex,
    replacement: String,
}
impl TokenFilter for PatternReplaceTokens {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens.into_iter().map(|mut t| { t.text = self.re.replace_all(&t.text, self.replacement.as_str()).into_owned(); t }).collect()
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct NGramFilter {
    min: usize,
    max: usize,
    edge: bool,
}
impl TokenFilter for NGramFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        self.filter_bounded(tokens, usize::MAX).expect("unbounded")
    }
    fn filter_bounded(&self, tokens: Vec<Token>, budget: usize) -> Option<Vec<Token>> {
        let mut out = Vec::new();
        for t in tokens {
            let chars: Vec<char> = t.text.chars().collect();
            let starts = if self.edge { 0..1 } else { 0..chars.len() };
            for s in starts {
                for n in self.min..=self.max {
                    if s + n > chars.len() {
                        break;
                    }
                    if out.len() == budget {
                        return None;
                    }
                    out.push(Token { text: chars[s..s + n].iter().collect(), ..t.clone() });
                }
            }
        }
        Some(out)
    }
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }
}

fn char_class_ok(c: char, classes: &[String]) -> bool {
    if classes.is_empty() {
        return true;
    }
    classes.iter().any(|cl| match cl.as_str() {
        "letter" => c.is_alphabetic(),
        "digit" => c.is_numeric(),
        "whitespace" => c.is_whitespace(),
        "punctuation" => c.is_ascii_punctuation(),
        "symbol" => !c.is_alphanumeric() && !c.is_whitespace() && !c.is_ascii_punctuation(),
        _ => false,
    })
}

#[derive(Clone)]
struct NGramTok {
    min: usize,
    max: usize,
    edge: bool,
    token_chars: Vec<String>,
}
impl Tokenizer for NGramTok {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        self.tokenize_bounded(text, usize::MAX).expect("unbounded")
    }
    fn tokenize_bounded(&self, text: &str, budget: usize) -> Option<Vec<Token>> {
        // Split into runs of accepted chars, then n-gram each run.
        let mut out = Vec::new();
        let mut pos = 0u32;
        let mut runs: Vec<(usize, Vec<(usize, char)>)> = Vec::new();
        let mut cur: Vec<(usize, char)> = Vec::new();
        for (i, c) in text.char_indices() {
            if char_class_ok(c, &self.token_chars) {
                cur.push((i, c));
            } else if !cur.is_empty() {
                runs.push((0, std::mem::take(&mut cur)));
            }
        }
        if !cur.is_empty() {
            runs.push((0, cur));
        }
        for (_, run) in runs {
            let starts = if self.edge { 0..1 } else { 0..run.len() };
            for s in starts {
                for n in self.min..=self.max {
                    if s + n > run.len() {
                        break;
                    }
                    if out.len() == budget {
                        return None;
                    }
                    let from = run[s].0;
                    let last = run[s + n - 1];
                    let to = last.0 + last.1.len_utf8();
                    out.push(Token { text: text[from..to].to_string(), position: pos, offset_from: from as u32, offset_to: to as u32, keyword: false });
                    pos += 1;
                }
            }
        }
        Some(out)
    }
    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct PatternTok {
    re: regex::Regex,
    /// `None`: split mode (`group: -1`); `Some(n)`: emit capture group `n`.
    group: Option<usize>,
}
impl Tokenizer for PatternTok {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        let mut pos = 0;
        let mut push = |from: usize, to: usize, out: &mut Vec<Token>| {
            if to > from {
                out.push(Token { text: text[from..to].to_string(), position: pos, offset_from: from as u32, offset_to: to as u32, keyword: false });
                pos += 1;
            }
        };
        match self.group {
            // Lucene `PatternTokenizer` group mode: each match's group,
            // skipping groups that did not participate or matched empty.
            Some(g) => {
                for caps in self.re.captures_iter(text) {
                    if let Some(m) = caps.get(g) {
                        push(m.start(), m.end(), &mut out);
                    }
                }
            }
            None => {
                let mut last = 0;
                for m in self.re.find_iter(text) {
                    push(last, m.start(), &mut out);
                    last = m.end();
                }
                push(last, text.len(), &mut out);
            }
        }
        out
    }
    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

/// Inline-flag prefix for ES's Java regex `flags` string
/// (`Regex.flagsFromString`: `|`-separated `java.util.regex.Pattern` names).
fn java_regex_flags(flags: &str) -> Result<String> {
    let mut out = String::new();
    for f in flags.split('|').map(str::trim).filter(|f| !f.is_empty()) {
        let f = f.strip_prefix("Pattern.").unwrap_or(f);
        match f.to_ascii_uppercase().as_str() {
            "CASE_INSENSITIVE" => out.push('i'),
            "MULTILINE" => out.push('m'),
            "DOTALL" => out.push('s'),
            "COMMENTS" => out.push('x'),
            // Rust regexes are Unicode-aware already; CANON_EQ / LITERAL /
            // UNIX_LINES have no equivalent and are refused below.
            "UNICODE_CASE" | "UNICODE_CHARACTER_CLASS" => {}
            other => return Err(EsError::illegal_argument(format!("Unknown regex flag [{other}]"))),
        }
    }
    Ok(if out.is_empty() { out } else { format!("(?{out})") })
}

#[derive(Clone)]
struct CharGroupTok {
    on: Vec<String>,
}
impl Tokenizer for CharGroupTok {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let split = |c: char| {
            self.on.iter().any(|o| match o.as_str() {
                "whitespace" => c.is_whitespace(),
                "letter" => c.is_alphabetic(),
                "digit" => c.is_numeric(),
                "punctuation" => c.is_ascii_punctuation(),
                "symbol" => !c.is_alphanumeric() && !c.is_whitespace(),
                s => s.chars().count() == 1 && s.starts_with(c),
            })
        };
        let mut out = Vec::new();
        let mut start: Option<usize> = None;
        let mut pos = 0;
        for (i, c) in text.char_indices() {
            if split(c) {
                if let Some(s) = start.take() {
                    out.push(Token { text: text[s..i].to_string(), position: pos, offset_from: s as u32, offset_to: i as u32, keyword: false });
                    pos += 1;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            out.push(Token { text: text[s..].to_string(), position: pos, offset_from: s as u32, offset_to: text.len() as u32, keyword: false });
        }
        out
    }
    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct MappingCf {
    rules: Vec<(String, String)>,
}
impl CharFilter for MappingCf {
    fn filter(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        let b = text;
        'outer: while i < b.len() {
            for (from, to) in &self.rules {
                if !from.is_empty() && b[i..].starts_with(from.as_str()) {
                    out.push_str(to);
                    i += from.len();
                    continue 'outer;
                }
            }
            let c = b[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
        out
    }
    fn clone_box(&self) -> Box<dyn CharFilter> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct PatternReplaceCf {
    re: regex::Regex,
    replacement: String,
}
impl CharFilter for PatternReplaceCf {
    fn filter(&self, text: &str) -> String {
        self.re.replace_all(text, self.replacement.as_str()).into_owned()
    }
    fn clone_box(&self) -> Box<dyn CharFilter> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn texts(a: &Analyzer, s: &str) -> Vec<String> {
        a.analyze(s).into_iter().map(|t| t.text).collect()
    }

    /// Index `docs` (`text` field, `mapping` for it) under `settings` and
    /// return the ids `query` finds, sorted.
    fn hits_for(settings: serde_json::Value, mapping: serde_json::Value, docs: &[&str], query: serde_json::Value) -> Vec<String> {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("idx", &json!({"settings": settings, "mappings": {"properties": {"text": mapping}}})).unwrap();
        let w: Vec<_> = docs.iter().enumerate().map(|(i, d)| crate::index::WriteReq::Index { id: Some(i.to_string()), source: json!({"text": d}).to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(w, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let r = crate::ml::search(&node, "idx", &json!({"query": query, "size": 100})).unwrap();
        let mut ids: Vec<String> = r["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        ids
    }

    /// SIDE 004 / 005 / 011 / 012: indexed retrieval through the repaired
    /// analyzers (Thai words, Czech and Lithuanian inflections, separated CJK
    /// characters, block-level HTML).
    #[test]
    fn language_analyzers_retrieve_by_word() {
        let m = |a: &str| json!({"type": "text", "analyzer": a});
        assert_eq!(hits_for(json!({}), m("thai"), &["ภาษาไทย", "อื่น"], json!({"match": {"text": "ภาษา"}})), ["0"]);
        assert_eq!(hits_for(json!({}), m("czech"), &["Česká Republika", "jiné"], json!({"match": {"text": "české republiky"}})), ["0"]);
        assert_eq!(hits_for(json!({}), m("cjk"), &["中 国", "中国"], json!({"match": {"text": "中"}})), ["0"]);
        assert_eq!(hits_for(json!({}), m("cjk"), &["中 国", "中国"], json!({"match": {"text": "中国"}})), ["1"]);
        let html = json!({"analysis": {"analyzer": {"html": {"type": "custom", "char_filter": ["html_strip"], "tokenizer": "standard", "filter": ["lowercase"]}}}});
        assert_eq!(hits_for(html.clone(), m("html"), &["<p>alpha</p><p>beta</p>", "alphabeta"], json!({"match": {"text": "beta"}})), ["0"]);
        assert_eq!(hits_for(html, m("html"), &["alpha<br>beta", "x"], json!({"match": {"text": "alpha"}})), ["0"]);
    }

    /// SIDE 006: language analyzers apply configured stop words (`_none_`,
    /// a list, a file) and stem exclusions.
    #[test]
    fn language_analyzer_stopwords_and_exclusions() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("stop.txt");
        std::fs::write(&file, "# comment\nfoo\n").unwrap();
        let reg = AnalysisRegistry::from_settings(&json!({"analyzer": {
            "keep_all": {"type": "english", "stopwords": "_none_"},
            "listed": {"type": "english", "stopwords": ["foo"]},
            "filed": {"type": "english", "stopwords_path": file.to_string_lossy()},
            "excl": {"type": "english", "stem_exclusion": ["running"]},
            "fr": {"type": "french", "stopwords": "_none_"}
        }})).unwrap();
        assert_eq!(texts(&reg.analyzer("keep_all").unwrap(), "the running"), ["the", "run"]);
        assert_eq!(texts(&reg.analyzer("listed").unwrap(), "the foo running"), ["the", "run"]);
        assert_eq!(texts(&reg.analyzer("filed").unwrap(), "the foo bar"), ["the", "bar"]);
        assert_eq!(texts(&reg.analyzer("excl").unwrap(), "the running jumping"), ["running", "jump"]);
        assert_eq!(texts(&reg.analyzer("fr").unwrap(), "le chat"), ["le", "chat"]);
        assert_eq!(texts(&Analyzer::by_name("english").unwrap(), "the running"), ["run"]);
        let m = json!({"type": "text", "analyzer": "keep_all"});
        let settings = json!({"analysis": {"analyzer": {"keep_all": {"type": "english", "stopwords": "_none_"}}}});
        assert_eq!(hits_for(settings, m, &["The", "other"], json!({"match": {"text": "the"}})), ["0"]);
        assert!(AnalysisRegistry::from_settings(&json!({"analyzer": {"bad": {"type": "english", "stopwords_path": "/nonexistent/stop.txt"}}})).is_err());
    }

    /// SIDE 017: a `mappings_path` table is loaded (absolute or under the
    /// config dir), escapes decode, and an invalid rule fails.
    #[test]
    fn mapping_char_filter_tables_load() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("digits.txt");
        let rules: String = (0..10).map(|d| format!("\\u{:04X} => {d}\n", 0x0660 + d)).collect();
        std::fs::write(&file, rules).unwrap();
        let settings = json!({"analysis": {
            "char_filter": {"digits": {"type": "mapping", "mappings_path": file.to_string_lossy()}},
            "analyzer": {"ids": {"type": "custom", "char_filter": ["digits"], "tokenizer": "keyword"}}}});
        let reg = AnalysisRegistry::from_settings(&settings["analysis"]).unwrap();
        assert_eq!(texts(&reg.analyzer("ids").unwrap(), "٢٥٠١٥"), ["25015"]);
        assert_eq!(hits_for(settings, json!({"type": "text", "analyzer": "ids"}), &["٢٥٠١٥", "99"], json!({"match": {"text": "25015"}})), ["0"]);
        let inline = AnalysisRegistry::from_settings(&json!({
            "char_filter": {"m": {"type": "mapping", "mappings": ["\\u0041 => a", "\\\\ => slash", "\\t => tab"]}},
            "analyzer": {"a": {"type": "custom", "char_filter": ["m"], "tokenizer": "keyword"}}})).unwrap();
        assert_eq!(texts(&inline.analyzer("a").unwrap(), "A\\\t"), ["aslashtab"]);
        for bad in [json!(["no arrow here"]), json!([" => x"])] {
            let e = AnalysisRegistry::from_settings(&json!({"char_filter": {"m": {"type": "mapping", "mappings": bad}},
                "analyzer": {"a": {"type": "custom", "char_filter": ["m"], "tokenizer": "keyword"}}}));
            assert!(e.is_err(), "{bad}");
        }
        assert!(AnalysisRegistry::from_settings(&json!({"char_filter": {"m": {"type": "mapping", "mappings_path": "/nonexistent/map.txt"}},
            "analyzer": {"a": {"type": "custom", "char_filter": ["m"], "tokenizer": "keyword"}}})).is_err());
    }

    /// P2 keyword_repeat: the literal and the stemmed term are both kept
    /// (`keyword_repeat` marks the first copy a keyword); `keyword_marker`
    /// protects its words from a later stemmer.
    #[test]
    fn keyword_repeat_keeps_the_original() {
        let settings = json!({"analysis": {
            "filter": {"protect": {"type": "keyword_marker", "keywords": ["running"]}, "protect_re": {"type": "keyword_marker", "keywords_pattern": "jump.*"}},
            "analyzer": {
                "both": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "keyword_repeat", "porter_stem", "remove_duplicates"]},
                "marked": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "protect", "porter_stem"]},
                "marked_re": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "protect_re", "porter_stem"]}}}});
        let reg = AnalysisRegistry::from_settings(&settings["analysis"]).unwrap();
        let both = reg.analyzer("both").unwrap();
        assert_eq!(texts(&both, "running"), ["running", "run"]);
        assert_eq!(texts(&both, "run"), ["run"], "an unchanged word is deduplicated");
        assert_eq!(texts(&reg.analyzer("marked").unwrap(), "running jumping"), ["running", "jump"]);
        assert_eq!(texts(&reg.analyzer("marked_re").unwrap(), "running jumping"), ["run", "jumping"]);
        let m = json!({"type": "text", "analyzer": "both", "search_analyzer": "standard"});
        assert_eq!(hits_for(settings.clone(), m.clone(), &["running", "walked"], json!({"term": {"text": "running"}})), ["0"]);
        assert_eq!(hits_for(settings, m, &["running", "walked"], json!({"term": {"text": "run"}})), ["0"]);
        assert!(AnalysisRegistry::from_settings(&json!({"filter": {"k": {"type": "keyword_marker"}}, "analyzer": {"a": {"tokenizer": "standard", "filter": ["k"]}}})).is_err());
    }

    /// P2 word_delimiter_graph: with `preserve_original`, the original
    /// product code is an alternative path, so AND and phrase searches of
    /// the code find a document indexed with the plain standard analyzer.
    #[test]
    fn word_delimiter_graph_keeps_the_original_path() {
        let settings = json!({"analysis": {
            "filter": {"wdg": {"type": "word_delimiter_graph", "preserve_original": true}},
            "analyzer": {"codes": {"type": "custom", "tokenizer": "whitespace", "filter": ["wdg", "lowercase"]}}}});
        let m = json!({"type": "text", "analyzer": "standard", "search_analyzer": "codes"});
        let docs = ["PowerShot2000", "power shot 2000", "unrelated"];
        assert_eq!(hits_for(settings.clone(), m.clone(), &docs, json!({"match": {"text": {"query": "PowerShot2000", "operator": "and"}}})), ["0", "1"]);
        assert_eq!(hits_for(settings.clone(), m.clone(), &docs, json!({"match_phrase": {"text": "PowerShot2000"}})), ["0", "1"]);
        // Controls: an exact lowercase term and a word that does not split.
        assert_eq!(hits_for(settings.clone(), m.clone(), &docs, json!({"term": {"text": "powershot2000"}})), ["0"]);
        assert_eq!(hits_for(settings, m, &docs, json!({"match": {"text": {"query": "unrelated", "operator": "and"}}})), ["2"]);
    }

    /// Findings 419, 422, 423: synonym rules keep their case, the WordNet
    /// format groups declarations by synset, and a later synonym filter's
    /// rules are not rewritten by an earlier one.
    #[test]
    fn synonym_rule_fidelity() {
        let reg = AnalysisRegistry::from_settings(&json!({
            "analyzer": {
                "cased": {"tokenizer": "keyword", "filter": ["abbr"]},
                "wn": {"tokenizer": "keyword", "filter": ["wordnet"]},
                "chain": {"tokenizer": "keyword", "filter": ["first", "second"]},
                "lower_first": {"tokenizer": "keyword", "filter": ["lowercase", "abbr"]}
            },
            "filter": {
                "abbr": {"type": "synonym", "synonyms": ["US => USA"]},
                "wordnet": {"type": "synonym_graph", "format": "wordnet", "synonyms": [
                    "s(100000002,1,'car',n,1,0).", "s(100000002,2,'auto',n,1,0).", "s(100000003,1,'o''clock',n,1,0).", "s(100000003,2,'hour',n,1,0)."
                ]},
                "first": {"type": "synonym", "synonyms": ["car => auto"]},
                "second": {"type": "synonym", "synonyms": ["car => vehicle"]}
            }
        }))
        .unwrap();
        let cased = reg.analyzer("cased").unwrap();
        assert_eq!(texts(&cased, "US"), vec!["USA"]);
        assert_eq!(texts(&cased, "us"), vec!["us"]);
        // With a lowercase filter first, the rule itself is lowercased.
        assert_eq!(texts(&reg.analyzer("lower_first").unwrap(), "US"), vec!["usa"]);
        let wn = reg.analyzer("wn").unwrap();
        let mut auto = texts(&wn, "auto");
        auto.sort();
        assert_eq!(auto, vec!["auto", "car"]);
        let mut hour = texts(&wn, "hour");
        hour.sort();
        assert_eq!(hour, vec!["hour", "o'clock"]);
        let chain = reg.analyzer("chain").unwrap();
        assert_eq!(texts(&chain, "car"), vec!["auto"]);
        assert_eq!(texts(&chain, "auto"), vec!["auto"]);
        assert!(AnalysisRegistry::from_settings(&json!({"filter": {"s": {"type": "synonym", "format": "wordnet", "synonyms": ["car, auto"]}}, "analyzer": {"a": {"tokenizer": "keyword", "filter": ["s"]}}})).is_err());
        // Inline `_analyze` chains follow the same rules.
        let a = reg.adhoc(&json!("keyword"), &[json!({"type": "synonym", "synonyms": ["car => auto"]}), json!({"type": "synonym", "synonyms": ["car => vehicle"]})], &[]).unwrap();
        assert_eq!(texts(&a, "car"), vec!["auto"]);
    }

    #[test]
    fn custom_analyzer_with_configured_filters() {
        let reg = AnalysisRegistry::from_settings(&json!({
            "analyzer": {
                "my": {"type": "custom", "tokenizer": "standard", "char_filter": ["amp"], "filter": ["lowercase", "my_stop", "syn"]},
                "std_stop": {"type": "standard", "stopwords": ["the"]}
            },
            "char_filter": {"amp": {"type": "mapping", "mappings": ["& => and"]}},
            "filter": {
                "my_stop": {"type": "stop", "stopwords": ["a", "an"]},
                "syn": {"type": "synonym", "synonyms": ["quick, fast"]}
            },
            "normalizer": {"lc": {"type": "custom", "filter": ["lowercase"]}}
        }))
        .unwrap();
        let a = reg.analyzer("my").unwrap();
        assert_eq!(texts(&a, "A Quick fox & dog"), vec!["quick", "fast", "fox", "and", "dog"]);
        assert!(matches!(reg.index_analyzer("my"), IndexAnalyzer::Custom(_)));
        assert!(matches!(reg.index_analyzer("english"), IndexAnalyzer::Builtin(_)));
        assert_eq!(texts(&reg.analyzer("std_stop").unwrap(), "The Cat"), vec!["cat"]);
        assert_eq!(texts(&reg.normalizer("lc").unwrap(), "Hello World"), vec!["hello world"]);
        assert!(AnalysisRegistry::from_settings(&json!({"analyzer": {"x": {"tokenizer": "nope"}}})).is_err());
    }

    #[test]
    fn ngram_tokenizer() {
        let reg = AnalysisRegistry::from_settings(&json!({
            "analyzer": {"ac": {"tokenizer": "edge", "filter": ["lowercase"]}},
            "tokenizer": {"edge": {"type": "edge_ngram", "min_gram": 2, "max_gram": 3, "token_chars": ["letter"]}}
        }))
        .unwrap();
        assert_eq!(texts(&reg.analyzer("ac").unwrap(), "Quick Fox"), vec!["qu", "qui", "fo", "fox"]);
    }
}
