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

impl AnalysisRegistry {
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
                    for tf in str_list(def.get("filter")) {
                        // Synonym rules are analyzed with the chain built so far.
                        let f = resolve_tf_pre(&tf, Some(&a))?;
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
        for f in filters {
            a = a.with_filter(match f {
                Value::String(s) => Analyzer::token_filter_by_name(s).ok_or_else(|| EsError::illegal_argument(format!("failed to find global filter under [{s}]")))?,
                other => build_token_filter("_anonymous_tokenfilter", other)?,
            });
        }
        Ok(a)
    }
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
            let pat = def.get("pattern").and_then(Value::as_str).unwrap_or("\\W+");
            let re = regex::Regex::new(pat).map_err(|e| EsError::illegal_argument(format!("invalid pattern for tokenizer [{name}]: {e}")))?;
            Ok(Box::new(PatternTok { re }))
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
                None => Ok(Analyzer::token_filter_by_name("stop").unwrap()),
                other => Ok(Box::new(StopWords { words: Arc::new(str_list(other).into_iter().collect()), ignore_case })),
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
            Ok(Box::new(Synonyms::parse(&rules, expand, pre, lenient, &source)?))
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
            Ok(Box::new(f))
        }
        "keep" => Ok(Box::new(Keep { words: str_list(def.get("keep_words")).into_iter().collect() })),
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
        other => Analyzer::token_filter_by_name(other).ok_or_else(|| EsError::illegal_argument(format!("Unknown filter type [{other}] for [{name}]"))),
    }
}

fn build_char_filter(name: &str, def: &Value) -> Result<Box<dyn CharFilter>> {
    let ty = def.get("type").and_then(Value::as_str).unwrap_or("");
    match ty {
        "mapping" => {
            let mut rules = Vec::new();
            for r in str_list(def.get("mappings")) {
                if let Some((a, b)) = r.split_once("=>") {
                    rules.push((unescape(a.trim()), unescape(b.trim())));
                }
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

fn unescape(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t").replace("\\u0020", " ")
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

/// Solr-format synonyms: `a, b, c` (equivalent; expand) and `a, b => c`.
#[derive(Clone)]
struct Synonyms {
    map: Arc<HashMap<String, Vec<String>>>,
}
impl Synonyms {
    /// Parse Solr-format rules, analyzing every word with `pre` (the chain
    /// before the filter). A word the chain drops (e.g. a stop word) is
    /// skipped when `lenient` (Lucene `ESSolrSynonymParser`), otherwise the
    /// build fails the way ES reports it: `failed to build synonyms from
    /// <source>` caused by `Invalid synonym rule at line N` caused by
    /// `term: <w> was completely eliminated by analyzer`.
    fn parse(rules: &[String], expand: bool, pre: Option<&Analyzer>, lenient: bool, source: &str) -> Result<Self> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        let norm = |w: &str| -> Result<Vec<String>> {
            let w = w.trim();
            if w.is_empty() {
                return Ok(Vec::new());
            }
            match pre {
                Some(a) => {
                    let toks: Vec<String> = a.analyze(w).into_iter().map(|t| t.text.to_lowercase()).collect();
                    if toks.is_empty() {
                        if lenient {
                            return Ok(Vec::new());
                        }
                        return Err(EsError::illegal_argument(format!("term: {w} was completely eliminated by analyzer")));
                    }
                    Ok(vec![toks.join(" ")])
                }
                None => Ok(vec![w.to_lowercase()]),
            }
        };
        let side = |s: &str| -> Result<Vec<String>> {
            let mut out = Vec::new();
            for w in s.split(',') {
                out.extend(norm(w)?);
            }
            Ok(out)
        };
        for (line, r) in rules.iter().enumerate() {
            let parsed = (|| -> Result<()> {
                if let Some((lhs, rhs)) = r.split_once("=>") {
                    let outs = side(rhs)?;
                    if outs.is_empty() {
                        return Ok(());
                    }
                    for l in side(lhs)? {
                        map.entry(l).or_default().extend(outs.iter().cloned());
                    }
                } else {
                    let words = side(r)?;
                    if words.is_empty() {
                        return Ok(());
                    }
                    for w in &words {
                        let outs: Vec<String> = if expand { words.clone() } else { vec![words[0].clone()] };
                        map.entry(w.clone()).or_default().extend(outs);
                    }
                }
                Ok(())
            })();
            if let Err(e) = parsed {
                return Err(EsError::illegal_argument(format!("failed to build synonyms from {source}"))
                    .caused_by(EsError::new(400, "parse_exception", format!("Invalid synonym rule at line {}", line + 1)).caused_by(e)));
            }
        }
        for v in map.values_mut() {
            let mut seen = HashSet::new();
            v.retain(|x| seen.insert(x.clone()));
        }
        Ok(Synonyms { map: Arc::new(map) })
    }
}
impl TokenFilter for Synonyms {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::with_capacity(tokens.len());
        for t in tokens {
            match self.map.get(&t.text) {
                Some(alts) => {
                    for a in alts {
                        out.push(Token { text: a.clone(), ..t.clone() });
                    }
                }
                None => out.push(t),
            }
        }
        out
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
}
impl TokenFilter for Keep {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens.into_iter().filter(|t| self.words.contains(&t.text)).collect()
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
        let mut out = Vec::new();
        for t in tokens {
            let chars: Vec<char> = t.text.chars().collect();
            let starts: Vec<usize> = if self.edge { vec![0] } else { (0..chars.len()).collect() };
            for s in starts {
                for n in self.min..=self.max {
                    if s + n > chars.len() {
                        break;
                    }
                    out.push(Token { text: chars[s..s + n].iter().collect(), ..t.clone() });
                }
            }
        }
        out
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
            let starts: Vec<usize> = if self.edge { vec![0] } else { (0..run.len()).collect() };
            for s in starts {
                for n in self.min..=self.max {
                    if s + n > run.len() {
                        break;
                    }
                    let from = run[s].0;
                    let last = run[s + n - 1];
                    let to = last.0 + last.1.len_utf8();
                    out.push(Token { text: text[from..to].to_string(), position: pos, offset_from: from as u32, offset_to: to as u32 });
                    pos += 1;
                }
            }
        }
        out
    }
    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct PatternTok {
    re: regex::Regex,
}
impl Tokenizer for PatternTok {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        let mut last = 0;
        let mut pos = 0;
        let mut push = |from: usize, to: usize, out: &mut Vec<Token>| {
            if to > from {
                out.push(Token { text: text[from..to].to_string(), position: pos, offset_from: from as u32, offset_to: to as u32 });
                pos += 1;
            }
        };
        for m in self.re.find_iter(text) {
            push(last, m.start(), &mut out);
            last = m.end();
        }
        push(last, text.len(), &mut out);
        out
    }
    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
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
                    out.push(Token { text: text[s..i].to_string(), position: pos, offset_from: s as u32, offset_to: i as u32 });
                    pos += 1;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            out.push(Token { text: text[s..].to_string(), position: pos, offset_from: s as u32, offset_to: text.len() as u32 });
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
