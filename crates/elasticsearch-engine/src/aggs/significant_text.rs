//! Source-based significant text. Foreground terms are analyzed once per
//! matching document; background frequencies come from the inverted index.
//! Only shard candidates cross the wire, never the background vocabulary.

use super::{partial, Ctx, Docs, Key, Matcher};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup, ResolvedField};
use rustic_core::analysis::Token;
use rustic_core::types::Term;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};

mod order;

const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Unlike `search.max_buckets`, this bounds temporary term and duplicate
/// sequence state even when the requested result contains only ten buckets.
/// The same bytes are held on the request breaker.
struct Budget {
    used: usize,
    limit: usize,
    mem: crate::memory::RequestMemory,
}

impl Budget {
    fn new() -> Self {
        Self { used: 0, limit: std::env::var("RUSTIC_SIGNIFICANT_TEXT_MAX_BYTES").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MAX_BYTES), mem: Default::default() }
    }
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.mem.add(bytes as u64, "<agg [significant_text]>")?;
        let next = self.used.saturating_add(bytes);
        if next > self.limit {
            return Err(EsError::new(429, "circuit_breaking_exception", format!(
                "[significant_text] temporary aggregation state would use [{next}] bytes, exceeding limit [{}]", self.limit
            )).with("bytes_wanted", next).with("bytes_limit", self.limit).with("durability", "TRANSIENT"));
        }
        self.used = next;
        Ok(())
    }
    fn release(&mut self, bytes: usize) {
        self.used = self.used.saturating_sub(bytes);
        self.mem.release(bytes as u64);
    }
}

/// ES hashes tokens to one byte and remembers six-byte windows. A set is
/// sufficient because the non-debug filter only asks whether a window has
/// occurred before. Windows continue across source values, but not documents.
#[derive(Default)]
struct DuplicateSequences {
    seen: HashSet<[u8; 6]>,
    window: [u8; 6],
    len: usize,
}

impl DuplicateSequences {
    fn next_document(&mut self) { self.len = 0; }

    fn filter(&mut self, tokens: &[Token], budget: &mut Budget) -> Result<Vec<bool>> {
        let mut keep = vec![true; tokens.len()];
        for (i, t) in tokens.iter().enumerate() {
            self.window.rotate_left(1);
            self.window[5] = crate::tsdb::murmur3_128(t.text.as_bytes(), 0).0 as u8;
            self.len = (self.len + 1).min(6);
            if self.len < 6 { continue; }
            if self.seen.contains(&self.window) {
                keep[i.saturating_sub(5)..=i].fill(false);
            } else {
                // Includes hash-table slack; fail rather than silently
                // dropping old windows and changing duplicate semantics.
                budget.charge(32)?;
                self.seen.insert(self.window);
            }
        }
        Ok(keep)
    }
}

#[derive(Clone)]
struct Heuristic {
    name: String,
    negatives: bool,
    superset: bool,
    normalize_above: u64,
}

impl Heuristic {
    fn parse(o: &Map<String, Value>) -> Result<Self> {
        let mut h = Self { name: "jlh".into(), negatives: false, superset: true, normalize_above: 0 };
        let mut found = false;
        for name in ["jlh", "mutual_information", "chi_square", "gnd", "percentage", "p_value", "script_heuristic"] {
            let Some(cfg) = o.get(name) else { continue };
            if found { return Err(EsError::illegal_argument("Only one significance heuristic may be specified")); }
            found = true;
            if name == "script_heuristic" {
                return Err(EsError::illegal_argument("[script_heuristic] is not supported by significant_text yet"));
            }
            let p = cfg.as_object().ok_or_else(|| EsError::parsing(format!("[{name}] must be an object")))?;
            let allowed: &[&str] = match name {
                "mutual_information" | "chi_square" => &["include_negatives", "background_is_superset"],
                "gnd" => &["background_is_superset"],
                "p_value" => &["background_is_superset", "normalize_above"],
                _ => &[],
            };
            for key in p.keys() {
                if !allowed.contains(&key.as_str()) { return Err(EsError::parsing(format!("[{name}] unknown field [{key}]"))); }
            }
            let boolean = |key: &str, default| -> Result<bool> {
                p.get(key).map(|v| v.as_bool().ok_or_else(|| EsError::parsing(format!("[{name}][{key}] must be a boolean")))).unwrap_or(Ok(default))
            };
            h.name = name.into();
            h.negatives = boolean("include_negatives", false)?;
            h.superset = boolean("background_is_superset", true)?;
            h.normalize_above = minimum(p, "normalize_above", 0)?;
        }
        Ok(h)
    }
    fn score(&self, f: u64, b: u64, n: u64, total: u64) -> Result<f64> {
        if f > n || b > total {
            return Err(EsError::illegal_argument("significance frequency exceeds its document population"));
        }
        if matches!(self.name.as_str(), "gnd" | "mutual_information" | "chi_square" | "p_value") && self.superset
            && (f > b || n > total || b - f > total - n) {
            return Err(EsError::illegal_argument("background frequencies do not contain the foreground; set [background_is_superset] to false for disjoint sets"));
        }
        if self.name == "jlh" {
            if n == 0 || total == 0 { return Ok(0.0); }
            // ES assigns one occurrence when a foreground term is entirely
            // absent from a separately selected background.
            let foreground = f as f64 / n as f64;
            let background = b.max(1) as f64 / total as f64;
            return Ok((foreground - background).max(0.0) * (foreground / background));
        }
        if self.name == "gnd" && !self.superset {
            let (fx, fy, fxy, all) = ((b + f) as f64, n as f64, f as f64, (total + n) as f64);
            if fxy == 0.0 { return Ok(0.0); }
            if fx == fy && fx == fxy { return Ok(1.0); }
            return Ok((-((fx.ln().max(fy.ln()) - fxy.ln()) / (all.ln() - fx.ln().min(fy.ln())))).exp());
        }
        Ok(partial::sig_score(&self.name, self.negatives, self.superset, self.normalize_above, f, b, n, total))
    }
}

fn positive(o: &Map<String, Value>, key: &str, default: usize) -> Result<usize> {
    let Some(v) = o.get(key) else { return Ok(default) };
    let n = v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).filter(|n| *n > 0)
        .and_then(|n| usize::try_from(n).ok());
    n.ok_or_else(|| EsError::illegal_argument(format!("[{key}] must be greater than 0. Found [{v}]")))
}

fn minimum(o: &Map<String, Value>, key: &str, default: u64) -> Result<u64> {
    o.get(key).map(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .ok_or_else(|| EsError::illegal_argument(format!("[{key}] must be greater than or equal to 0. Found [{v}]")))).unwrap_or(Ok(default))
}

pub fn shard_size(o: &Map<String, Value>) -> Result<usize> {
    let size = positive(o, "size", 10)?;
    let suggested = (size.saturating_mul(3) / 2).saturating_add(10).saturating_mul(2);
    Ok(positive(o, "shard_size", suggested)?.max(size))
}

/// Used by the engine to keep each physical shard's sampler and candidate
/// selection independent, including a search over several one-shard indices.
pub fn present(spec: &Value) -> bool {
    spec.as_object().is_some_and(|m| m.values().any(|v| v.get("significant_text").is_some()
        || v.get("aggs").or_else(|| v.get("aggregations")).is_some_and(present)))
}

pub(super) fn validate_contexts(spec: &Value) -> Result<()> {
    for def in spec.as_object().into_iter().flat_map(|m| m.values()) {
        let Some(sub) = def.get("aggs").or_else(|| def.get("aggregations")) else { continue };
        if def.get("random_sampler").is_some() && present(sub) {
            // `significant_text` is rejected under `random_sampler`.
            return Err(EsError::illegal_argument("[significant_text] under [random_sampler] is not supported yet"));
        }
        validate_contexts(sub)?;
    }
    Ok(())
}

fn source_paths(lookup: &FieldLookup, name: &str) -> Vec<String> {
    // Match MappingLookup.sourcePaths: concrete multi-fields resolve to their
    // parent and copy_to destinations read all supplying fields. Explicit
    // aliases are literal source paths; the default caller passes the already
    // resolved concrete field name instead.
    let mut path = name;
    while lookup.fields.get(path).is_some_and(|f| f.is_multi_field) {
        let Some((parent, _)) = path.rsplit_once('.') else { break };
        path = parent;
    }
    let mut paths = vec![path.to_owned()];
    if let Some(sources) = lookup.copy_sources.get(path) { paths.extend(sources.iter().cloned()); }
    paths
}

/// Check the final rendered bucket tree against the configured request limit.
/// Walking the request avoids counting arbitrary `buckets` in metadata or
/// stored source. Intermediate candidates are governed by the memory budget,
/// not this output limit, since final size/score filtering has not run yet.
pub fn check_final_bucket_limit(spec: &Value, output: &Value, max: usize) -> Result<()> {
    fn walk(spec: &Value, output: &Value, count: &mut usize, max: usize) -> Result<()> {
        let Some(spec) = spec.as_object() else { return Ok(()) };
        for (name, def) in spec {
            let result = output.get(name).or_else(|| output.as_object().and_then(|o| o.iter()
                .find(|(key, _)| key.split_once('#').is_some_and(|(_, suffix)| suffix == name))
                .map(|(_, value)| value)));
            let Some(result) = result else { continue };
            let sub = def.get("aggs").or_else(|| def.get("aggregations"));
            match result.get("buckets") {
                Some(Value::Array(buckets)) => {
                    *count = count.saturating_add(buckets.len());
                    if *count > max { return Err(EsError::too_many_buckets(max)); }
                    if let Some(sub) = sub { for bucket in buckets { walk(sub, bucket, count, max)?; } }
                }
                Some(Value::Object(buckets)) => {
                    *count = count.saturating_add(buckets.len());
                    if *count > max { return Err(EsError::too_many_buckets(max)); }
                    if let Some(sub) = sub { for bucket in buckets.values() { walk(sub, bucket, count, max)?; } }
                }
                _ => { if let Some(sub) = sub { walk(sub, result, count, max)?; } }
            }
        }
        Ok(())
    }
    walk(spec, output, &mut 0, max)
}

/// Skip unrelated root values without allocating their JSON trees. Nested
/// objects and arrays are retained intact so source_values preserves its dotted
/// path and array semantics. Wildcards use the general source filter.
pub(crate) fn select_source(raw: &str, paths: &[String]) -> Result<Value> {
    use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor};
    if paths.iter().any(|p| p.contains('*') || p.contains('?')) {
        return Ok(serde_json::from_str(&crate::source_json::filter_raw(raw, paths, &[])?)?);
    }
    struct SourceKey<'a>(std::borrow::Cow<'a, str>);
    impl<'de> serde::Deserialize<'de> for SourceKey<'de> {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            struct KeyVisitor;
            impl<'de> Visitor<'de> for KeyVisitor {
                type Value = SourceKey<'de>;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a source key") }
                fn visit_borrowed_str<E: serde::de::Error>(self, s: &'de str) -> std::result::Result<Self::Value, E> { Ok(SourceKey(std::borrow::Cow::Borrowed(s))) }
                fn visit_str<E: serde::de::Error>(self, s: &str) -> std::result::Result<Self::Value, E> { Ok(SourceKey(std::borrow::Cow::Owned(s.to_owned()))) }
            }
            d.deserialize_str(KeyVisitor)
        }
    }
    struct Selected<'a>(&'a [String]);
    impl<'de> Visitor<'de> for Selected<'_> {
        type Value = Value;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a source object") }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
            let mut out = Map::new();
            while let Some(SourceKey(key)) = map.next_key::<SourceKey<'de>>()? {
                if self.0.iter().any(|p| p == key.as_ref() || p.strip_prefix(key.as_ref()).is_some_and(|suffix| suffix.starts_with('.'))) {
                    out.insert(key.into_owned(), map.next_value()?);
                } else { map.next_value::<IgnoredAny>()?; }
            }
            Ok(Value::Object(out))
        }
    }
    impl<'de> DeserializeSeed<'de> for Selected<'_> {
        type Value = Value;
        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> std::result::Result<Value, D::Error> { d.deserialize_map(self) }
    }
    let mut de = serde_json::Deserializer::from_str(raw);
    match Selected(paths).deserialize(&mut de).and_then(|out| { de.end()?; Ok(out) }) {
        Ok(out) => Ok(out),
        // An unrelated nested value can contain a lone UTF16 unit. Preserve
        // the general fetch path's raw filtering before parsing selected data.
        Err(_) => Ok(serde_json::from_str(&crate::source_json::filter_raw(raw, paths, &[])?)?),
    }
}

fn accepts_field(f: &ResolvedField) -> bool {
    matches!(f.ty, EsType::Text | EsType::MatchOnlyText | EsType::Keyword)
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub(super) fn significant_text(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        if sub.is_some_and(|s| !s.is_empty()) {
            return Err(EsError::illegal_argument(format!("Aggregator [{name}] of type [significant_text] cannot accept sub-aggregations")));
        }
        let field = super::field_name(o, name, "significant_text")?;
        let size = shard_size(o)?;
        let shard_min = minimum(o, "shard_min_doc_count", 0)?;
        minimum(o, "min_doc_count", 3)?;
        let heuristic = Heuristic::parse(o)?;
        for key in ["include", "exclude"] {
            if o.get(key).is_some_and(Value::is_object) {
                return Err(EsError::illegal_argument("Partition-based include/exclude is not supported by significant_text"));
            }
        }
        let include = Matcher::parse(o.get("include"))?;
        let exclude = Matcher::parse(o.get("exclude"))?;
        let duplicates = match o.get("filter_duplicate_text") {
            None => false,
            Some(Value::Bool(v)) => *v,
            Some(v) => return Err(EsError::parsing(format!("[filter_duplicate_text] must be a boolean, found [{v}]"))),
        };
        let source_fields: Option<Vec<&str>> = o.get("source_fields").map(|v| {
            v.as_array().ok_or_else(|| EsError::parsing("[source_fields] must be an array of strings"))?
                .iter().map(|v| v.as_str().ok_or_else(|| EsError::parsing("[source_fields] must be an array of strings"))).collect::<Result<Vec<_>>>()
        }).transpose()?;
        let mapped: Vec<_> = self.segs.iter().map(|s| s.lookup.get(field)).collect();
        for f in mapped.iter().flatten() {
            if !accepts_field(f) {
                return Err(EsError::illegal_argument(format!("Field [{}] has no analyzer, but SignificantText requires an analyzed field", f.name)));
            }
            if !f.indexed {
                return Err(EsError::illegal_argument(format!("Cannot search on field [{}] since it is not indexed.", f.name)));
            }
        }
        if mapped.iter().all(Option::is_none) {
            return Ok(json!({"doc_count": 0, "bg_count": 0, "buckets": []}));
        }
        let mut budget = Budget::new();
        let mut counts: HashMap<String, u64, ahash::RandomState> = HashMap::default();
        let mut spotter = DuplicateSequences::default();
        let mut subset = 0u64;
        let (mut values_fetched, mut chars_fetched, mut extract_ns, mut collect_ns, mut analyzed_count, mut extract_count) = (0u64,0u64,0u64,0u64,0u64,0u64);
        for (si, ds) in docs.iter().enumerate() {
            let Some(f) = mapped[si] else { continue };
            let analyzer = if f.ty == EsType::Keyword {
                f.normalizer.clone().unwrap_or_else(|| std::sync::Arc::new(rustic_core::analysis::Analyzer::keyword()))
            } else {
                // A search-only analyzer must never change the source tokens.
                // An absent index analyzer means the standard index analyzer,
                // even when search_analyzer is explicitly configured.
                match &f.index_analyzer {
                    Some(crate::analysis::IndexAnalyzer::Custom(a)) => a.clone(),
                    Some(crate::analysis::IndexAnalyzer::Builtin(n)) => std::sync::Arc::new(rustic_core::analysis::Analyzer::by_name(n).unwrap_or_else(rustic_core::analysis::Analyzer::standard)),
                    None => std::sync::Arc::new(rustic_core::analysis::Analyzer::standard()),
                }
            };
            let paths: Vec<String> = source_fields.as_ref().map(|fs| fs.iter().flat_map(|p| source_paths(self.segs[si].lookup, p)).collect())
                .unwrap_or_else(|| source_paths(self.segs[si].lookup, &f.name));
            let fetch_spec = json!({"_source": {"includes": paths}});
            let mut consume = |doc: rustic_core::types::DocId, source: Option<Value>| -> Result<()> {
                subset += 1;
                let weight = super::DOC_WEIGHTS.with(|weights| weights.borrow().as_ref()
                    .and_then(|segments| segments.get(si)).and_then(Option::as_ref)
                    .and_then(|values| values.get(doc as usize)).copied().unwrap_or(1)) as u64;
                spotter.next_document();
                let Some(source) = source.as_ref() else { return Ok(()); };
                let mut unique: HashSet<String, ahash::RandomState> = HashSet::default();
                let mut doc_bytes = 0usize;
                for path in &paths {
                    let extract_start = self.env.profile.then(std::time::Instant::now);
                    let values = crate::highlight::source_values(source, path);
                    if let Some(start) = extract_start { extract_count += 1; extract_ns += start.elapsed().as_nanos().max(1) as u64; }
                    for value in values {
                        let text = match value {
                            Value::String(s) => std::borrow::Cow::Borrowed(s.as_str()),
                            Value::Number(_) | Value::Bool(_) => std::borrow::Cow::Owned(value.to_string()),
                            _ => continue,
                        };
                        // Reserve analysis scratch before tokenization. Analyzer
                        // output is checked too, since synonym filters can expand it.
                        let scratch = text.len().saturating_mul(8).saturating_add(256);
                        budget.charge(scratch)?;
                        if self.env.profile { values_fetched += 1; chars_fetched += text.encode_utf16().count() as u64; }
                        let tokens = analyzer.analyze(&text);
                        let token_bytes = tokens.iter().map(|t| t.text.len().saturating_add(80)).sum::<usize>();
                        budget.charge(token_bytes)?;
                        let keep = if duplicates { Some(spotter.filter(&tokens, &mut budget)?) } else { None };
                        for (i, token) in tokens.into_iter().enumerate() {
                            if keep.as_ref().is_some_and(|k| !k[i]) { continue; }
                            if include.is_some() || exclude.is_some() {
                                let k = Key::Str(token.text.clone());
                                if include.as_ref().is_some_and(|m| !m.matches(&k)) || exclude.as_ref().is_some_and(|m| m.matches(&k)) { continue; }
                            }
                            if !unique.contains(&token.text) {
                                let bytes = token.text.len().saturating_add(96);
                                budget.charge(bytes)?;
                                doc_bytes += bytes;
                                unique.insert(token.text);
                            }
                        }
                        budget.release(token_bytes + scratch);
                    }
                }
                if self.env.profile { analyzed_count += unique.len() as u64; }
                for term in unique {
                    let collect_start = self.env.profile.then(std::time::Instant::now);
                    if let Some(n) = counts.get_mut(&term) { *n += weight; }
                    else {
                        budget.charge(term.len().saturating_add(128))?;
                        counts.insert(term, weight);
                    }
                    if let Some(start) = collect_start { collect_ns += start.elapsed().as_nanos().max(1) as u64; }
                }
                budget.release(doc_bytes);
                Ok(())
            };
            if let Some(fetch) = self.env.fetch_source {
                fetch(si, ds, &paths, &mut consume)?;
            } else {
                for &(doc, score) in ds {
                    let source = (self.env.fetch_hit)(si, doc, score, &fetch_spec)?.get("_source").cloned();
                    consume(doc, source)?;
                }
            }
        }
        // A filtered background must honor alias/DLS visibility. Unfiltered
        // ES statistics deliberately include deleted docs until a merge.
        let filtered = o.contains_key("background_filter") || self.env.background_filtered;
        let background = if filtered {
            let upper = self.segs.iter().map(|s| s.leaf.num_docs() as usize).fold(0usize, usize::saturating_add);
            // The initial visible set and the filtered set may coexist.
            budget.charge(upper.saturating_mul(24))?;
            let all: Docs = (0..self.segs.len()).map(|s| (self.env.all_docs)(s).into_iter().map(|d| (d, 0.0)).collect()).collect();
            Some(match o.get("background_filter") { Some(q) => self.filter_docs(q, &all)?, None => all })
        } else { None };
        let superset: u64 = match &background {
            Some(bg) => bg.iter().enumerate().filter(|(s, _)| mapped[*s].is_some()).map(|(_, ds)| ds.len() as u64).sum(),
            None => self.segs.iter().enumerate().filter(|(s, _)| mapped[*s].is_some()).map(|(_, s)| s.leaf.num_docs() as u64).sum(),
        };
        let total_buckets = counts.len();
        let mut counts: Vec<_> = counts.into_iter().filter(|(_, count)| *count >= shard_min).collect();
        counts.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut frequencies = vec![0u64; counts.len()];
        if background.is_none() {
            let texts: Vec<_> = counts.iter().map(|(term, _)| term.as_bytes()).collect();
            for (s, segment) in self.segs.iter().enumerate() {
                if let Some(id) = mapped[s].and_then(|f| f.core_id) {
                    segment.leaf.doc_freqs_sorted(id, &texts, |i, n| frequencies[i] += n as u64);
                }
            }
        }
        let mut scored = Vec::new();
        for ((term, count), mut bg_count) in counts.into_iter().zip(frequencies) {
            for (s, segment) in self.segs.iter().enumerate().filter(|_| background.is_some()) {
                let Some(f) = mapped[s] else { continue };
                let Some(id) = f.core_id else { continue };
                let t = Term::new_text(id, &term);
                let Some(mut postings) = segment.leaf.postings(&t) else { continue };
                let live = segment.leaf.undeleted_docs();
                while let Some(d) = postings.next() {
                    if !live(d) { continue; }
                    if background.as_ref().is_none_or(|bg| bg[s].binary_search_by_key(&d, |e| e.0).is_ok()) { bg_count += 1; }
                }
            }
            let score = heuristic.score(count, bg_count, subset, superset)?;
            scored.push((term, count, bg_count, score));
        }
        scored.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(size);
        let mut result = json!({"doc_count": subset, "bg_count": superset, "buckets": scored.into_iter().map(|(key, doc_count, bg_count, score)|
            json!({"key": key, "doc_count": doc_count, "score": super::fnum(score), "bg_count": bg_count})).collect::<Vec<_>>()});
        if self.env.profile {
            result["__significant_text_profile"] = json!({"values_fetched":values_fetched,"chars_fetched":chars_fetched,
                "extract_ns":extract_ns,"extract_count":extract_count,"collect_analyzed_ns":collect_ns,
                "collect_analyzed_count":analyzed_count,"total_buckets":total_buckets});
        }
        if partial::active() { Ok(result) }
        else {
            let result = reduce(o, &[&result], self.env.max_buckets, true)?;
            self.add_buckets(result["buckets"].as_array().map_or(0, Vec::len))?;
            Ok(result)
        }
    }
}

/// Shared by the in-process shard reducer and the node coordinator. A
/// non-final reduce keeps every shard candidate, including zero-score terms.
/// Background counts are summed only for shards that returned that candidate,
/// matching ES's deliberately approximate distributed significance counts.
pub fn reduce(conf: &Map<String, Value>, parts: &[&Value], max_buckets: usize, final_reduce: bool) -> Result<Value> {
    let heuristic = Heuristic::parse(conf)?;
    let size = positive(conf, "size", 10)?;
    let min = minimum(conf, "min_doc_count", 3)?;
    let subset = parts.iter().map(|p| p["doc_count"].as_u64().unwrap_or(0)).sum::<u64>();
    let superset = parts.iter().map(|p| p["bg_count"].as_u64().unwrap_or(0)).sum::<u64>();
    let mut counts: HashMap<String, (u64, u64), ahash::RandomState> = HashMap::default();
    let mut arrival = Vec::new();
    let mut budget = Budget::new();
    for part in parts {
        for b in part["buckets"].as_array().into_iter().flatten() {
            let Some(key) = b["key"].as_str() else { return Err(EsError::internal("significant_text shard returned a non-string key")) };
            if !counts.contains_key(key) { budget.charge(key.len().saturating_add(160))?; arrival.push(key.to_owned()); }
            let entry = counts.entry(key.into()).or_default();
            entry.0 += b["doc_count"].as_u64().unwrap_or(0);
            entry.1 += b["bg_count"].as_u64().unwrap_or(0);
        }
    }
    let mut buckets = Vec::new();
    for key in arrival {
        let (count, bg) = counts.remove(&key).expect("recorded key");
        let score = heuristic.score(count, bg, subset, superset)?;
        buckets.push((key, count, bg, score));
    }
    let initial = parts.iter().filter_map(|p| p["buckets"].as_array()).find(|b| !b.is_empty()).map_or(0, Vec::len);
    let selected = order::select(&buckets, initial, if final_reduce { size } else { buckets.len() },
        |b| !final_reduce || (b.3 > 0.0 && b.1 >= min));
    let buckets: Vec<_> = selected.into_iter().map(|i| buckets[i].clone()).collect();
    if final_reduce && buckets.len() > max_buckets { return Err(EsError::too_many_buckets(max_buckets)); }
    let mut result = json!({"doc_count": subset, "bg_count": superset, "buckets": buckets.into_iter().map(|(key, count, bg, score)|
        json!({"key": key, "doc_count": count, "score": super::fnum(score), "bg_count": bg})).collect::<Vec<_>>() });
    let mut debug = Map::new();
    for part in parts { if let Some(fields) = part["__significant_text_profile"].as_object() {
        for (key, value) in fields { let n = debug.get(key).and_then(Value::as_u64).unwrap_or(0); debug.insert(key.clone(),json!(n.saturating_add(value.as_u64().unwrap_or(0)))); }
    }}
    if !debug.is_empty() { result["__significant_text_profile"] = Value::Object(debug); }
    Ok(result)
}

#[cfg(test)]
mod tests;
