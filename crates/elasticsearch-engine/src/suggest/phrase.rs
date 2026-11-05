//! The `phrase` suggester: ES's `NoisyChannelSpellChecker` over candidate
//! sets drawn by `direct_generator`s, scored by an n-gram language model
//! (`stupid_backoff` / `laplace` / `linear_interpolation` `WordScorer`s).

use super::distance::Dict;
use super::term::{suggest_similar, Mode, Spell};
use super::{field_analyzer, jfloat, tokens, SuggestTarget};
use crate::error::{EsError, Result};
use rustic_core::analysis::Analyzer;
use rustic_core::query::{SearchOptions, Searcher};
use serde_json::{json, Map, Value};
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Candidate {
    term: String,
    frequency: u64,
    distance: f64,
    score: f64,
    user_input: bool,
}

struct CandSet {
    original: Candidate,
    candidates: Vec<Candidate>,
}

/// `DirectCandidateGenerator`.
struct Generator<'a, 'b> {
    dict: Dict<'a, 'b>,
    spell: Spell,
    size: usize,
    use_ttf: bool,
    dict_size: u64,
    pre: Option<Arc<Analyzer>>,
    post: Option<Arc<Analyzer>>,
}

impl Generator<'_, '_> {
    fn frequency(&self, t: &str) -> u64 {
        let (df, ttf) = self.dict.stats(t);
        if self.use_ttf {
            ttf
        } else {
            df
        }
    }
    fn score(&self, freq: u64, channel: f64) -> f64 {
        channel * ((freq as f64 + 1.0) / (self.dict_size as f64 + 1.0))
    }
    fn create(&self, term: String, user_input: bool) -> Candidate {
        let f = self.frequency(&term);
        Candidate { score: self.score(f, 1.0), term, frequency: f, distance: 1.0, user_input }
    }
    fn threshold(tf: u64) -> f32 {
        if tf == 0 {
            return 0.0;
        }
        let t = tf as f64;
        (t * (t.log10() / 5f64.log10()) + 1.0).round().max(0.0) as f32
    }
    fn draw(&self, set: &mut CandSet) {
        let mut term = set.original.term.clone();
        if let Some(p) = &self.pre {
            if let Some(last) = p.analyze(&term).into_iter().last() {
                term = last.text;
            }
        }
        let thr = if self.spell.mode == Mode::Always { 0.0 } else { Self::threshold(set.original.frequency) };
        let words = suggest_similar(&self.dict, &term, self.size, &self.spell, Some(thr));
        for w in words {
            let c = Candidate { frequency: self.frequency(&w.term), distance: w.score as f64, score: self.score(w.freq, w.score as f64), term: w.term, user_input: false };
            match &self.post {
                None => add(set, c),
                Some(p) => {
                    for t in p.analyze(&c.term) {
                        let f = self.frequency(&t.text);
                        let nc = Candidate { term: t.text, frequency: f, distance: c.distance, score: self.score(c.frequency, c.distance), user_input: false };
                        add(set, nc);
                    }
                }
            }
        }
    }
}

fn add(set: &mut CandSet, c: Candidate) {
    if c.term == set.original.term {
        return;
    }
    if !set.candidates.iter().any(|x| x.term == c.term) {
        set.candidates.push(c);
    }
}

#[derive(Clone, Copy, Debug)]
enum Model {
    StupidBackoff(f64),
    Laplace(f64),
    Linear(f64, f64, f64),
}

/// `WordScorer` over the suggestion field.
struct Scorer<'a, 'b> {
    dict: Dict<'a, 'b>,
    use_ttf: bool,
    vocab: f64,
    num_terms: f64,
    real_word: f64,
    sep: String,
    model: Model,
}

impl Scorer<'_, '_> {
    fn freq(&self, t: &str) -> f64 {
        let (df, ttf) = self.dict.stats(t);
        (if self.use_ttf { ttf } else { df }) as f64
    }
    fn join(&self, parts: &[&str]) -> String {
        parts.join(&self.sep)
    }
    fn channel(&self, c: &Candidate) -> f64 {
        if c.distance == 1.0 {
            self.real_word
        } else {
            c.distance
        }
    }
    fn unigram(&self, w: &Candidate) -> f64 {
        match self.model {
            Model::Laplace(alpha) => (alpha + self.freq(&w.term)) / (self.vocab + alpha * self.num_terms),
            _ => (1.0 + self.freq(&w.term)) / (self.vocab + self.num_terms),
        }
    }
    fn bigram(&self, w: &Candidate, w1: &Candidate) -> f64 {
        let count = self.freq(&self.join(&[&w1.term, &w.term]));
        match self.model {
            Model::StupidBackoff(d) => {
                if count < 1.0 {
                    d * self.unigram(w)
                } else {
                    count / (w1.frequency as f64 + 0.00000000001)
                }
            }
            Model::Laplace(alpha) => (alpha + count) / (w1.frequency as f64 + alpha * self.num_terms),
            Model::Linear(_, bl, ul) => {
                if count < 1.0 {
                    ul * self.unigram(w)
                } else {
                    bl * (count / (0.5 + w1.frequency as f64)) + ul * self.unigram(w)
                }
            }
        }
    }
    fn trigram(&self, w: &Candidate, w1: &Candidate, w2: &Candidate) -> f64 {
        match self.model {
            Model::StupidBackoff(d) => {
                // ES `StupidBackoff.scoreTrigram`: look the (w1, w) bigram up first.
                let bi = self.freq(&self.join(&[&w1.term, &w.term]));
                if bi < 1.0 {
                    return d * self.unigram(w);
                }
                let tri = self.freq(&self.join(&[&w2.term, &w1.term, &w.term]));
                if tri < 1.0 {
                    return d * (bi / (w1.frequency as f64 + 0.00000000001));
                }
                tri / (bi + 0.00000000001)
            }
            // Laplace has no trigram model: the base `WordScorer` backs off to the bigram.
            Model::Laplace(_) => self.bigram(w, w1),
            Model::Linear(tl, _, _) => {
                let count = self.freq(&self.join(&[&w.term, &w1.term, &w2.term]));
                if count < 1.0 {
                    return self.bigram(w, w1);
                }
                let bi = self.freq(&self.join(&[&w.term, &w1.term]));
                tl * (count / (1.0 + bi)) + self.bigram(w, w1)
            }
        }
    }
    fn score(&self, path: &[Candidate], sets: &[CandSet], at: usize, gram: usize) -> f64 {
        let c = self.channel(&path[at]);
        let _ = &sets[at].original;
        let lm = if at == 0 || gram == 1 {
            self.unigram(&path[at])
        } else if at == 1 || gram == 2 {
            self.bigram(&path[at], &path[at - 1])
        } else {
            self.trigram(&path[at], &path[at - 1], &path[at - 2])
        };
        (c * lm).log10()
    }
}

struct Correction {
    score: f64,
    path: Vec<Candidate>,
}

/// `Correction.compareTo`: score, then (on ties) lexicographically smaller terms win.
fn better(a_score: f64, a: &[Candidate], b_score: f64, b: &[Candidate]) -> std::cmp::Ordering {
    if a_score == b_score {
        for (x, y) in a.iter().zip(b) {
            let c = x.term.cmp(&y.term);
            if c != std::cmp::Ordering::Equal {
                return c.reverse();
            }
        }
        return a.len().cmp(&b.len());
    }
    a_score.total_cmp(&b_score)
}

struct Search<'s, 'a, 'b> {
    scorer: &'s Scorer<'a, 'b>,
    gram: usize,
    max: usize,
    cutoff: f64,
    out: Vec<Correction>,
}

impl Search<'_, '_, '_> {
    fn update_top(&mut self, path: &[Candidate], log_score: f64) {
        let score = log_score.exp();
        if score <= self.cutoff {
            return;
        }
        if self.out.len() < self.max {
            self.out.push(Correction { score, path: path.to_vec() });
            return;
        }
        let (mi, _) = self
            .out
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| better(a.score, &a.path, b.score, &b.path))
            .expect("non-empty");
        if better(self.out[mi].score, &self.out[mi].path, score, path) == std::cmp::Ordering::Less {
            self.out[mi] = Correction { score, path: path.to_vec() };
        }
    }

    fn find(&mut self, sets: &[CandSet], path: &mut Vec<Candidate>, ord: usize, left: usize, path_score: f64) {
        let cur = &sets[ord];
        let last = ord == sets.len() - 1;
        path[ord] = cur.original.clone();
        let s0 = path_score + self.scorer.score(path, sets, ord, self.gram);
        if last {
            self.update_top(path, s0);
            if left > 0 {
                for c in &cur.candidates {
                    path[ord] = c.clone();
                    let s = path_score + self.scorer.score(path, sets, ord, self.gram);
                    self.update_top(path, s);
                }
            }
        } else if left > 0 {
            self.find(sets, path, ord + 1, left, s0);
            for c in &cur.candidates {
                path[ord] = c.clone();
                let s = path_score + self.scorer.score(path, sets, ord, self.gram);
                self.find(sets, path, ord + 1, left - 1, s);
            }
        } else {
            self.find(sets, path, ord + 1, 0, s0);
        }
    }
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn parse_model(o: &Map<String, Value>) -> Result<Model> {
    let Some(sm) = o.get("smoothing") else { return Ok(Model::StupidBackoff(0.4)) };
    let sm = sm.as_object().ok_or_else(|| EsError::parsing("[smoothing] must be an object"))?;
    let (k, v) = sm.iter().next().ok_or_else(|| EsError::illegal_argument("smoothing model is missing"))?;
    let g = |n: &str| v.get(n).and_then(num);
    Ok(match k.as_str() {
        "stupid_backoff" => Model::StupidBackoff(g("discount").unwrap_or(0.4)),
        "laplace" => Model::Laplace(g("alpha").unwrap_or(0.5)),
        "linear" => {
            let (t, b, u) = (g("trigram_lambda").unwrap_or(0.0), g("bigram_lambda").unwrap_or(0.0), g("unigram_lambda").unwrap_or(0.0));
            let sum = t + b + u;
            if (sum - 1.0).abs() > 0.001 {
                return Err(EsError::illegal_argument("linear smoothing lambdas must sum to 1"));
            }
            Model::Linear(t, b, u)
        }
        other => return Err(EsError::illegal_argument(format!("suggester[phrase] doesn't support object field [{other}]"))),
    })
}

/// Run one `phrase` suggestion; returns the entries array.
pub fn run(text: &str, o: &Map<String, Value>, targets: &[SuggestTarget]) -> Result<Value> {
    super::check_keys(
        "phrase",
        o,
        &[
            "field", "analyzer", "size", "shard_size", "text", "gram_size", "real_word_error_likelihood", "confidence", "separator", "max_errors", "direct_generator",
            "highlight", "collate", "smoothing", "force_unigrams", "token_limit",
        ],
    )?;
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("the required field option is missing"))?;
    let size = o.get("size").and_then(num).map(|x| x as usize).unwrap_or(5);
    if size == 0 {
        return Err(EsError::illegal_argument("size must be positive"));
    }
    let shard_size = o.get("shard_size").and_then(num).map(|x| x as usize).unwrap_or(size.max(5));
    let real_word = o.get("real_word_error_likelihood").and_then(num).unwrap_or(0.95);
    if !(0.0..=1.0).contains(&real_word) {
        return Err(EsError::illegal_argument("real_word_error_likelihood must be > 0.0"));
    }
    let confidence = o.get("confidence").and_then(num).unwrap_or(1.0);
    let max_errors = o.get("max_errors").and_then(num).unwrap_or(1.0);
    let separator = o.get("separator").and_then(Value::as_str).unwrap_or(" ").to_string();
    let require_unigram = o.get("force_unigrams").and_then(Value::as_bool).unwrap_or(true);
    let token_limit = o.get("token_limit").and_then(num).map(|x| x as usize).unwrap_or(10);
    let model = parse_model(o)?;
    let hl = o.get("highlight").and_then(Value::as_object).map(|h| {
        (h.get("pre_tag").and_then(Value::as_str).map(str::to_string), h.get("post_tag").and_then(Value::as_str).map(str::to_string))
    });
    let (pre_tag, post_tag) = match hl {
        Some((Some(a), Some(b))) => (Some(a), Some(b)),
        Some(_) => return Err(EsError::illegal_argument("Pre and post tag must both be null or both not be null.")),
        None => (None, None),
    };
    let collate = o.get("collate").and_then(Value::as_object).cloned();
    let entry_len = text.encode_utf16().count();
    // (phrase, highlighted, score, collate_match)
    let mut options: Vec<(String, Option<String>, f32, Option<bool>)> = Vec::new();
    for t in targets {
        let Some(f) = t.lookup.get(field) else {
            return Err(EsError::illegal_argument(format!("No mapping found for field [{field}]")));
        };
        let Some(fid) = f.core_id else { continue };
        let analyzer = field_analyzer(t, f, o.get("analyzer").and_then(Value::as_str))?;
        let gram = match o.get("gram_size").and_then(num) {
            Some(g) if g >= 1.0 => g as usize,
            Some(_) => return Err(EsError::illegal_argument("gram_size must be >= 1")),
            None => detect_shingle_size(&analyzer),
        };
        // Generators.
        let mut gdefs: Vec<Map<String, Value>> = match o.get("direct_generator") {
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_object().cloned()).collect(),
            Some(Value::Object(m)) => vec![m.clone()],
            _ => Vec::new(),
        };
        if gdefs.is_empty() {
            let mut m = Map::new();
            m.insert("field".into(), Value::String(field.to_string()));
            gdefs.push(m);
        }
        let mut gens = Vec::new();
        for g in &gdefs {
            let gf = g.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("[direct_generator] field is missing"))?;
            let gfield = t.lookup.get(gf).ok_or_else(|| EsError::illegal_argument(format!("No mapping found for field [{gf}]")))?;
            let Some(gid) = gfield.core_id else { continue };
            let dict = Dict { leaves: t.leaves, field: gid };
            let (ttf, df, _) = dict.field_totals();
            let use_ttf = ttf > 0;
            let an = |k: &str| -> Result<Option<Arc<Analyzer>>> {
                match g.get(k).and_then(Value::as_str) {
                    Some(n) => field_analyzer(t, gfield, Some(n)).map(Some),
                    None => Ok(None),
                }
            };
            gens.push(Generator {
                spell: Spell::parse(g)?,
                size: g.get("size").and_then(num).map(|x| x as usize).unwrap_or(5),
                use_ttf,
                dict_size: if use_ttf { ttf } else { df },
                pre: an("pre_filter")?,
                post: an("post_filter")?,
                dict,
            });
        }
        if gens.is_empty() {
            continue;
        }
        let sdict = Dict { leaves: t.leaves, field: fid };
        let (sttf, sdf, nterms) = sdict.field_totals();
        let s_use_ttf = sttf > 0;
        let vocab = if s_use_ttf { sttf as f64 } else { sdict.max_doc() as f64 };
        let _ = sdf;
        let num_terms = if vocab + nterms as f64 > 1.0 { nterms as f64 } else { 0.0 };
        let scorer = Scorer { dict: sdict, use_ttf: s_use_ttf, vocab, num_terms, real_word, sep: separator.clone(), model };
        // Candidate sets from the token stream.
        let mut sets: Vec<CandSet> = Vec::new();
        let mut any_unigram = false;
        let mut any_tokens = false;
        let mut last_pos: Option<u32> = None;
        for tok in tokens(&analyzer, text) {
            any_tokens = true;
            if require_unigram && tok.text.contains(' ') {
                continue;
            }
            any_unigram = true;
            if last_pos == Some(tok.position) && !sets.is_empty() {
                let c = gens[0].create(tok.text, true);
                let s = sets.last_mut().expect("set");
                add(s, c);
                continue;
            }
            last_pos = Some(tok.position);
            sets.push(CandSet { original: gens[0].create(tok.text, true), candidates: Vec::new() });
        }
        if require_unigram && !any_unigram && any_tokens {
            return Err(EsError::illegal_argument("At least one unigram is required but all tokens were ngrams"));
        }
        if sets.is_empty() || sets.len() >= token_limit {
            continue;
        }
        for set in sets.iter_mut() {
            for g in &gens {
                g.draw(set);
            }
            if set.candidates.len() > shard_size {
                set.candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| b.term.cmp(&a.term)));
                set.candidates.truncate(shard_size);
            }
        }
        let mut cutoff = f64::MIN_POSITIVE;
        if confidence > 0.0 {
            let path: Vec<Candidate> = sets.iter().map(|s| s.original.clone()).collect();
            let log: f64 = (0..sets.len()).map(|i| scorer.score(&path, &sets, i, gram)).sum();
            cutoff = log.exp() * confidence;
        }
        let missp = if max_errors >= 1.0 { max_errors as usize } else { (max_errors * sets.len() as f64).round() as usize };
        let mut search = Search { scorer: &scorer, gram, max: shard_size, cutoff, out: Vec::new() };
        let mut path: Vec<Candidate> = sets.iter().map(|s| s.original.clone()).collect();
        search.find(&sets, &mut path, 0, missp.max(1), 0.0);
        let mut corrections = search.out;
        corrections.sort_by(|a, b| better(b.score, &b.path, a.score, &a.path));
        for c in corrections {
            let phrase: String = c.path.iter().map(|x| x.term.as_str()).collect::<Vec<_>>().join(" ");
            let mut collate_match = None;
            if let Some(col) = &collate {
                let m = collate_matches(t, col, &phrase)?;
                let prune = col.get("prune").and_then(Value::as_bool).unwrap_or(false);
                if !m && !prune {
                    continue;
                }
                if prune {
                    collate_match = Some(m);
                }
            }
            let highlighted = match (&pre_tag, &post_tag) {
                (Some(a), Some(b)) => {
                    // Consecutive corrected terms share one tag pair (ES `Correction.join`).
                    let mut out = String::new();
                    let mut open = false;
                    for (i, x) in c.path.iter().enumerate() {
                        if i > 0 {
                            out.push(' ');
                        }
                        if !x.user_input && !open {
                            out.push_str(a);
                            open = true;
                        } else if x.user_input && open {
                            // close before the separator already written
                            out.truncate(out.len() - 1);
                            out.push_str(b);
                            out.push(' ');
                            open = false;
                        }
                        out.push_str(&x.term);
                    }
                    if open {
                        out.push_str(b);
                    }
                    Some(out)
                }
                _ => None,
            };
            let score = c.score as f32;
            match options.iter_mut().find(|o| o.0 == phrase) {
                Some(o) => o.2 = o.2.max(score),
                None => options.push((phrase, highlighted, score, collate_match)),
            }
        }
    }
    options.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    options.truncate(size);
    let opts: Vec<Value> = options
        .into_iter()
        .map(|(text, hl, score, cm)| {
            let mut m = Map::new();
            m.insert("text".into(), Value::String(text));
            if let Some(h) = hl {
                m.insert("highlighted".into(), Value::String(h));
            }
            m.insert("score".into(), jfloat(score));
            if let Some(c) = cm {
                m.insert("collate_match".into(), Value::Bool(c));
            }
            Value::Object(m)
        })
        .collect();
    Ok(json!([{"text": text, "offset": 0, "length": entry_len, "options": opts}]))
}

/// Max shingle size of an analyzer that emits shingles (probed), else 1.
fn detect_shingle_size(a: &Analyzer) -> usize {
    let toks = a.analyze("a b c d e f");
    toks.iter().map(|t| t.text.split(' ').count()).max().unwrap_or(1).max(1)
}

/// Does the rendered `collate` query match any document of the target?
fn collate_matches(t: &SuggestTarget, col: &Map<String, Value>, phrase: &str) -> Result<bool> {
    let q = col.get("query").ok_or_else(|| EsError::illegal_argument("collate query is missing"))?;
    let src = match q {
        Value::Object(m) if m.contains_key("source") => crate::template::source_text(&m["source"]),
        other => crate::template::source_text(other),
    };
    let mut params = col.get("params").and_then(Value::as_object).cloned().unwrap_or_default();
    params.insert("suggestion".into(), Value::String(phrase.to_string()));
    let qv = crate::template::render_to_json(&src, &Value::Object(params))?;
    let Some(svc) = t.svc else { return Ok(true) };
    let ctx = crate::query::dsl::QueryContext {
        deadline: None,
        lookup: t.lookup,
        index_name: &svc.name,
        index_uuid: &svc.uuid,
        aliases: Vec::new(),
        now: crate::date::now_millis(),
        leaves: t.leaves,
        default_fields: vec!["*".to_string()],
        terms_lookup: None,
        max_terms_count: 65_536,
        max_regex_length: svc.setting_usize("index.max_regex_length", 1000),
        max_result_window: 10_000,
        seq_no_disabled: false,
        allow_unmapped_fields: true,
        knn_prefilter: None,
        knn_prefilter_lookup: None,
        allow_expensive_queries: true,
        mappings: Some(&**svc),
    };
    let query = crate::query::dsl::parse_query(&qv, &ctx)?;
    let so = SearchOptions { top_k: 1, ..Default::default() };
    let top = Searcher::new(t.leaves.to_vec()).search(&query, &so);
    Ok(!top.hits.is_empty())
}
