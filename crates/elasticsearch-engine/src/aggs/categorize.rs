//! `categorize_text`: groups text values into categories of similar
//! messages (ES ML `TokenListCategorizer`, a port of ml-cpp's token list
//! data categorizer).
//!
//! Tokens come from an `ml_standard`-style tokenizer (letters, digits,
//! `_ . -`; tokens starting with a digit are dropped). ES weights tokens
//! with an English part-of-speech dictionary; that dictionary is not
//! bundled here, so every alphabetic token of two or more letters counts as
//! a (non-verb) dictionary word — the same weighting scheme, without the
//! extra verb weight.

use super::super::{Ctx, Docs};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};

const MAX_TOKENS: usize = 100;
const EPSILON: f32 = 0.000001;
const MIN_DICTIONARY_LENGTH: usize = 2;
const CONSECUTIVE_FOR_EXTRA: usize = 3;
const EXTRA_VERB_WEIGHT: i32 = 5;
const EXTRA_OTHER_WEIGHT: i32 = 2;
const ADJACENCY_BOOST: i32 = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Tw {
    id: usize,
    w: i32,
}

/// `ml_standard` tokenization (after the `first_line_with_letters` char filter).
fn tokenize(text: &str) -> Vec<String> {
    let line = text.lines().find(|l| l.chars().any(char::is_alphabetic)).unwrap_or("");
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        let t = cur.trim_matches(|c: char| c == '.' || c == '-' || c == '_').to_string();
        cur.clear();
        if t.is_empty() {
            return;
        }
        let first = t.chars().next().unwrap();
        if first.is_ascii_digit() || !t.chars().any(char::is_alphabetic) {
            return;
        }
        out.push(t);
    };
    for c in line.chars() {
        if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    // Default categorization analyzer: day / month names and time zones are stop words.
    const STOP: &[&str] = &[
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "January", "February", "March", "April", "May",
        "June", "July", "August", "September", "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "GMT", "UTC",
    ];
    out.retain(|t| !STOP.contains(&t.as_str()));
    out.truncate(MAX_TOKENS);
    out
}

struct Weights {
    consecutive: usize,
}

impl Weights {
    fn weight(&mut self, term: &str) -> i32 {
        let dict = term.chars().count() >= MIN_DICTIONARY_LENGTH && term.chars().all(|c| c.is_ascii_alphabetic());
        if !dict {
            self.consecutive = 0;
            return 1;
        }
        self.consecutive += 1;
        let boost = if self.consecutive >= CONSECUTIVE_FOR_EXTRA { ADJACENCY_BOOST } else { 1 };
        1 + EXTRA_OTHER_WEIGHT * boost
    }
}

fn min_reweight(w: i32) -> i32 {
    if w <= ADJACENCY_BOOST {
        w
    } else {
        1 + (w - 1) / ADJACENCY_BOOST
    }
}

fn max_reweight(w: i32) -> i32 {
    let min_extra = EXTRA_VERB_WEIGHT.min(EXTRA_OTHER_WEIGHT);
    let max_extra = (EXTRA_VERB_WEIGHT + 1).max(EXTRA_OTHER_WEIGHT + 1);
    if w <= min_extra || w > max_extra {
        w
    } else {
        1 + (w - 1) * ADJACENCY_BOOST
    }
}

fn min_matching_weight(w: i32, t: f32) -> i32 {
    if w == 0 {
        0
    } else {
        (w as f32 * t + EPSILON).floor() as i32 + 1
    }
}

fn max_matching_weight(w: i32, t: f32) -> i32 {
    if w == 0 {
        0
    } else {
        (w as f32 / t - EPSILON).ceil() as i32 - 1
    }
}

/// Weighted edit distance (insert / delete cost the token weight, a
/// substitution the larger of the two weights).
fn edit_distance(a: &[Tw], b: &[Tw]) -> i32 {
    let mut prev: Vec<i32> = Vec::with_capacity(b.len() + 1);
    prev.push(0);
    for t in b {
        let l = *prev.last().unwrap();
        prev.push(l + t.w);
    }
    let mut cur = vec![0i32; b.len() + 1];
    for x in a {
        cur[0] = prev[0] + x.w;
        for (j, y) in b.iter().enumerate() {
            let sub = prev[j] + if x.id == y.id { 0 } else { x.w.max(y.w) };
            cur[j + 1] = sub.min(prev[j + 1] + x.w).min(cur[j] + y.w);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn similarity(a: &[Tw], aw: i32, b: &[Tw], bw: i32) -> f32 {
    let m = aw.max(bw);
    if m > 0 {
        1.0 - edit_distance(a, b) as f32 / m as f32
    } else {
        1.0
    }
}

struct Category {
    base: Vec<Tw>,
    base_weight: i32,
    base_len: usize,
    max_len: usize,
    /// Sorted by token id.
    common: Vec<Tw>,
    orig_unique_weight: i32,
    begin: usize,
    end: usize,
    matches: u64,
}

impl Category {
    fn common_weight(&self) -> i32 {
        self.common.iter().map(|t| t.w).sum()
    }
    fn has_common(&self, id: usize) -> bool {
        self.common.binary_search_by_key(&id, |t| t.id).is_ok()
    }
    fn missing_common_weight(&self, unique: &[Tw]) -> i32 {
        let present: i32 = self.common.iter().filter(|c| unique.binary_search_by_key(&c.id, |t| t.id).is_ok()).map(|c| c.w).sum();
        self.common_weight() - present
    }
    /// The ordered common tokens (the category key).
    fn key_tokens(&self) -> Vec<usize> {
        let mut seen = Vec::new();
        for t in &self.base[self.begin..self.end] {
            if self.has_common(t.id) && !seen.contains(&t.id) {
                seen.push(t.id);
            }
        }
        seen
    }
    fn max_matching_len(&self) -> usize {
        let ext = (self.max_len * 11 / 10).min((self.base_len as f64 * (self.common.len() as f64 / 1.5).max(2.0)) as usize);
        self.max_len.max(ext)
    }
    fn contains_in_order(&self, tokens: &[Tw]) -> bool {
        let key = self.key_tokens();
        let mut it = tokens.iter();
        key.iter().all(|k| it.any(|t| t.id == *k))
    }
    fn matches_search(&self, weight: i32, len: usize, unique: &[Tw], tokens: &[Tw]) -> bool {
        (self.base_weight == 0) == (weight == 0) && self.max_matching_len() >= len && self.missing_common_weight(unique) == 0 && self.contains_in_order(tokens)
    }
    fn add(&mut self, len: usize, unique: &[Tw], count: u64) {
        self.common.retain(|c| unique.binary_search_by_key(&c.id, |t| t.id).is_ok());
        while self.end > self.begin && !self.has_common(self.base[self.end - 1].id) {
            self.end -= 1;
        }
        while self.begin < self.end && !self.has_common(self.base[self.begin].id) {
            self.begin += 1;
        }
        self.max_len = self.max_len.max(len);
        self.matches += count;
    }
}

/// ES `TokenListCategorizer`: messages (on a shard) or shard categories (in
/// the coordinator reduce, `mergeWireCategory`) join the most similar
/// category, or start one.
struct Categorizer {
    lower: f32,
    upper: f32,
    vocab: HashMap<String, usize>,
    words: Vec<String>,
    cats: Vec<Category>,
    /// Indices into `cats`, most matches first.
    by_matches: Vec<usize>,
}

impl Categorizer {
    fn new(similarity_threshold: i64) -> Categorizer {
        let lower = similarity_threshold as f32 / 100.0;
        Categorizer { lower, upper: (1.0 + lower) / 2.0, vocab: HashMap::new(), words: Vec::new(), cats: Vec::new(), by_matches: Vec::new() }
    }

    fn id(&mut self, token: &str) -> usize {
        if let Some(id) = self.vocab.get(token) {
            return *id;
        }
        self.words.push(token.to_string());
        self.vocab.insert(token.to_string(), self.words.len() - 1);
        self.words.len() - 1
    }

    fn unique(tokens: &[Tw]) -> Vec<Tw> {
        let mut grouped: BTreeMap<usize, i32> = BTreeMap::new();
        for t in tokens {
            *grouped.entry(t.id).or_insert(0) += t.w;
        }
        grouped.into_iter().map(|(id, w)| Tw { id, w }).collect()
    }

    /// One message of `len` characters.
    fn add_text(&mut self, text: &str, filters: &[regex::Regex]) -> usize {
        let mut t = text.to_string();
        for f in filters {
            t = f.replace_all(&t, "").into_owned();
        }
        let mut wc = Weights { consecutive: 0 };
        let tokens: Vec<Tw> = tokenize(&t).iter().map(|tk| Tw { id: self.id(tk), w: wc.weight(tk) }).collect();
        let unique = Self::unique(&tokens);
        let len = text.chars().count();
        self.add(tokens, unique, len, len, None, 1)
    }

    /// Join (or start) a category: `tokens` / `unique` of a message, or a
    /// shard category's base / common tokens (`wire` carries the rest of it).
    fn add(&mut self, tokens: Vec<Tw>, unique: Vec<Tw>, base_len: usize, len: usize, wire: Option<(i32, usize, usize)>, count: u64) -> usize {
        let weight: i32 = tokens.iter().map(|t| t.w).sum();
        let min_rw: i32 = tokens.iter().map(|t| min_reweight(t.w)).sum();
        let max_rw: i32 = tokens.iter().map(|t| max_reweight(t.w)).sum();
        let mut min_w = min_matching_weight(min_rw, self.lower);
        let mut max_w = max_matching_weight(max_rw, self.lower);
        let mut best: Option<usize> = None;
        let mut best_sim = self.lower;
        let mut chosen: Option<usize> = None;
        for (pos, &ci) in self.by_matches.iter().enumerate() {
            let c = &self.cats[ci];
            let ms = c.matches_search(weight, len, &unique, &tokens);
            if !ms {
                if c.base_weight < min_w || c.base_weight > max_w {
                    continue;
                }
                let missing = c.missing_common_weight(&unique);
                if missing > 0 {
                    let prop = (c.common_weight() - missing) as f32 / c.orig_unique_weight as f32;
                    if prop < self.lower {
                        continue;
                    }
                }
            }
            let sim = similarity(&tokens, weight, &c.base, c.base_weight);
            if ms || sim > self.upper {
                chosen = Some(pos);
                break;
            }
            if sim > best_sim {
                best = Some(pos);
                best_sim = sim;
                min_w = min_matching_weight(min_rw, sim);
                max_w = max_matching_weight(max_rw, sim);
            }
        }
        match chosen.or(best) {
            Some(pos) => {
                let ci = self.by_matches[pos];
                self.cats[ci].add(len, &unique, count);
                // Keep `by_matches` ordered by matches (stable bubble up).
                let mut p = pos;
                while p > 0 && self.cats[self.by_matches[p - 1]].matches < self.cats[self.by_matches[p]].matches {
                    self.by_matches.swap(p - 1, p);
                    p -= 1;
                }
                ci
            }
            None => {
                let uw: i32 = unique.iter().map(|t| t.w).sum();
                let (orig_unique_weight, begin, end) = wire.unwrap_or((uw, 0, tokens.len()));
                self.cats.push(Category { base_weight: weight, end, base: tokens, base_len, max_len: len, common: unique, orig_unique_weight, begin, matches: count });
                self.by_matches.push(self.cats.len() - 1);
                self.cats.len() - 1
            }
        }
    }

    /// `(key, regex, max_matching_length)` of a category.
    fn render(&self, ci: usize) -> (String, String, usize) {
        let c = &self.cats[ci];
        let key: Vec<&str> = c.key_tokens().into_iter().map(|i| self.words[i].as_str()).collect();
        let regex = if key.is_empty() { ".*".to_string() } else { format!(".*?{}.*?", key.iter().map(|k| java_escape(k)).collect::<Vec<_>>().join(".+?")) };
        (key.join(" "), regex, c.max_matching_len())
    }

    /// A category as shard state (ES `SerializableTokenListCategory`).
    fn wire(&self, ci: usize) -> Value {
        let c = &self.cats[ci];
        let toks = |ts: &[Tw]| ts.iter().map(|t| json!([self.words[t.id], t.w])).collect::<Vec<_>>();
        json!({ "base": toks(&c.base), "common": toks(&c.common), "base_len": c.base_len, "max_len": c.max_len,
                "ouw": c.orig_unique_weight, "begin": c.begin, "end": c.end })
    }

    /// Merge one shard category of `count` documents.
    fn add_wire(&mut self, w: &Value, count: u64) -> Option<usize> {
        let mut toks = |k: &str| -> Option<Vec<Tw>> {
            w.get(k)?.as_array()?.iter().map(|t| Some(Tw { id: self.id(t.get(0)?.as_str()?), w: t.get(1)?.as_i64()? as i32 })).collect()
        };
        let base = toks("base")?;
        let mut common = toks("common")?;
        common.sort_by_key(|t| t.id);
        let num = |k: &str| w.get(k).and_then(Value::as_u64).map(|n| n as usize);
        let wire = (w.get("ouw")?.as_i64()? as i32, num("begin")?, num("end")?.min(base.len()));
        Some(self.add(base, common, num("base_len")?, num("max_len")?, Some(wire), count))
    }
}

/// The `categorize_text` options both the shard and the reduce read.
fn options(name: &str, o: &Map<String, Value>) -> Result<(i64, usize, u64)> {
    let thr = o.get("similarity_threshold").and_then(Value::as_i64).unwrap_or(70);
    if !(1..=100).contains(&thr) {
        return Err(EsError::illegal_argument(format!("[similarity_threshold] must be in the range [1, 100]. Found [{thr}] in [{name}]")));
    }
    let size = o.get("size").and_then(Value::as_i64).unwrap_or(10).max(1) as usize;
    let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(1);
    Ok((thr, size, min_doc_count))
}

/// The ordered categories with at least `min_doc_count` matches, at most `size`.
fn ranked(c: &Categorizer, size: usize, min_doc_count: u64) -> Vec<(usize, (String, String, usize))> {
    let mut out: Vec<(usize, (String, String, usize))> = (0..c.cats.len()).filter(|i| c.cats[*i].matches >= min_doc_count).map(|i| (i, c.render(i))).collect();
    out.sort_by(|a, b| c.cats[b.0].matches.cmp(&c.cats[a.0].matches).then_with(|| a.1 .0.cmp(&b.1 .0)));
    out.truncate(size);
    out
}

/// Coordinator reduce (ES `InternalCategorizationAggregation.reduce`): every
/// shard's categories (`__p` of each bucket) re-categorized in shard order.
/// Returns each final bucket's rendered fields and its `(shard, bucket)`
/// members; `None` when a bucket lacks its state (an older node).
pub fn reduce(name: &str, o: &Map<String, Value>, shards: &[Vec<&Value>]) -> Result<Option<Vec<(Map<String, Value>, Vec<(usize, usize)>)>>> {
    let (thr, size, min_doc_count) = options(name, o)?;
    let mut c = Categorizer::new(thr);
    let mut members: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (s, buckets) in shards.iter().enumerate() {
        for (b, bucket) in buckets.iter().enumerate() {
            let count = bucket.get("doc_count").and_then(Value::as_u64).unwrap_or(0);
            let Some(ci) = bucket.get("__p").and_then(|w| c.add_wire(w, count)) else { return Ok(None) };
            members.entry(ci).or_default().push((s, b));
        }
    }
    Ok(Some(ranked(&c, size, min_doc_count).into_iter().map(|(ci, (key, regex, mml))| {
        let mut m = Map::new();
        m.insert("doc_count".into(), json!(c.cats[ci].matches));
        m.insert("key".into(), json!(key));
        m.insert("regex".into(), json!(regex));
        m.insert("max_matching_length".into(), json!(mml));
        (m, members.remove(&ci).unwrap_or_default())
    }).collect()))
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub(super) fn categorize_text(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument(format!("Required [field] for [categorize_text] aggregation [{name}]")))?;
        let (thr, size, min_doc_count) = options(name, o)?;
        if o.contains_key("categorization_filters") && o.contains_key("categorization_analyzer") {
            return Err(EsError::illegal_argument("[categorization_filters] cannot be used with [categorization_analyzer]"));
        }
        if let Some(f) = self.segs.iter().find_map(|s| s.lookup.get(field)) {
            if !matches!(f.ty, EsType::Text | EsType::MatchOnlyText | EsType::Keyword | EsType::Wildcard | EsType::ConstantKeyword) {
                // ES's message swaps the field name and the field type class.
                let class = match f.declared {
                    t if t.is_numeric() => "NumberFieldType",
                    t if t.is_date() => "DateFieldType",
                    EsType::Boolean => "BooleanFieldType",
                    EsType::Ip => "IpFieldType",
                    EsType::GeoPoint => "GeoPointFieldType",
                    _ => "MappedFieldType",
                };
                let m = format!("categorize_text agg [{name}] only works on text and keyword fields. Cannot aggregate field type [{field}] via [{class}]");
                return Err(EsError::illegal_argument(m.clone()).caused_by(EsError::illegal_argument(m)));
            }
        }
        let filters: Vec<regex::Regex> = o
            .get("categorization_filters")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(regex::Regex::new).collect::<std::result::Result<Vec<_>, _>>())
            .transpose()
            .map_err(|e| EsError::illegal_argument(format!("categorization_filters: {e}")))?
            .unwrap_or_default();
        let mut c = Categorizer::new(thr);
        let mut cat_docs: Vec<Docs> = Vec::new();
        let spec = json!({ "_source": { "includes": [field] } });
        for (s, v) in docs.iter().enumerate() {
            if self.segs[s].lookup.get(field).is_none() {
                continue;
            }
            for &(d, sc) in v {
                let hit = (self.env.fetch_hit)(s, d, sc, &spec)?;
                let mut texts = Vec::new();
                collect_texts(hit.get("_source"), field, &mut texts);
                for text in texts {
                    let ci = c.add_text(&text, &filters);
                    if ci == cat_docs.len() {
                        cat_docs.push(vec![Vec::new(); docs.len()]);
                    }
                    cat_docs[ci][s].push((d, sc));
                }
            }
        }
        // A coordinator reduce follows: categories carry their token lists.
        let shard = crate::search::aggregation_shard_response();
        let out = ranked(&c, size, min_doc_count);
        self.add_buckets(out.len())?;
        let mut buckets = Vec::new();
        for (ci, (key, regex, mml)) in out {
            let mut d = std::mem::take(&mut cat_docs[ci]);
            for v in d.iter_mut() {
                v.sort_by_key(|x| x.0);
                v.dedup_by_key(|x| x.0);
            }
            let mut m = Map::new();
            m.insert("doc_count".into(), json!(c.cats[ci].matches));
            m.insert("key".into(), json!(key));
            m.insert("regex".into(), json!(regex));
            m.insert("max_matching_length".into(), json!(mml));
            m.extend(self.subs(sub, &d)?);
            if shard {
                m.insert("__p".into(), c.wire(ci));
            }
            buckets.push(m);
        }
        super::super::apply_parent_pipelines(sub, &mut buckets)?;
        Ok(json!({ "buckets": buckets }))
    }
}

/// Regex-quote a token the way ES's categorizer does (meta characters
/// only; `-` stays as is).
fn java_escape(t: &str) -> String {
    let mut out = String::new();
    for c in t.chars() {
        if "\\^$.|?*+()[]{}".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn collect_texts(src: Option<&Value>, path: &str, out: &mut Vec<String>) {
    let Some(src) = src else { return };
    match src {
        Value::Array(a) => a.iter().for_each(|x| collect_texts(Some(x), path, out)),
        Value::Object(m) => {
            if let Some(v) = m.get(path) {
                push_strings(v, out);
                return;
            }
            for (k, v) in m {
                if let Some(rest) = path.strip_prefix(k.as_str()).and_then(|r| r.strip_prefix('.')) {
                    collect_texts(Some(v), rest, out);
                }
            }
        }
        _ => {}
    }
}

fn push_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| push_strings(x, out)),
        Value::Number(n) => out.push(n.to_string()),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ml_standard_tokens() {
        assert_eq!(
            tokenize("Error: connection to host-1.example.com:9200 failed after 3 retries (timeout=30s) [id=0xdeadbeef]"),
            vec!["Error", "connection", "to", "host-1.example.com", "failed", "after", "retries", "timeout", "id"]
        );
        assert_eq!(tokenize("Node 2 stopping"), vec!["Node", "stopping"]);
    }
}
