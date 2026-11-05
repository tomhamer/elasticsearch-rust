//! The `term` suggester: Lucene's `DirectSpellChecker` per analyzed token.

use super::distance::{osa, Dict, Distance};
use super::{field_analyzer, jfloat, SuggestTarget};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Missing,
    Popular,
    Always,
}

/// `DirectSpellChecker` settings (shared with the phrase suggester's
/// `direct_generator`).
#[derive(Clone, Debug)]
pub struct Spell {
    pub accuracy: f32,
    pub max_edits: usize,
    pub max_inspections: usize,
    pub max_term_freq: f32,
    pub prefix_length: usize,
    pub min_word_length: usize,
    pub min_doc_freq: f32,
    pub distance: Distance,
    pub mode: Mode,
    pub sort_frequency: bool,
    pub lowercase: bool,
}

impl Default for Spell {
    fn default() -> Self {
        Spell {
            accuracy: 0.5,
            max_edits: 2,
            max_inspections: 5,
            max_term_freq: 0.01,
            prefix_length: 1,
            min_word_length: 4,
            min_doc_freq: 0.0,
            distance: Distance::Internal,
            mode: Mode::Missing,
            sort_frequency: false,
            lowercase: false,
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

impl Spell {
    /// Parse the spell-checker options of `o` (unknown keys are left to the caller).
    pub fn parse(o: &Map<String, Value>) -> Result<Spell> {
        let mut s = Spell::default();
        for (k, v) in o {
            let bad = || EsError::illegal_argument(format!("[{k}] has an invalid value"));
            match k.as_str() {
                "accuracy" => s.accuracy = num(v).ok_or_else(bad)? as f32,
                "max_edits" => {
                    let n = num(v).ok_or_else(bad)?;
                    if !(1.0..=2.0).contains(&n) {
                        return Err(EsError::illegal_argument(format!("Illegal max_edits value {n}")));
                    }
                    s.max_edits = n as usize;
                }
                "max_inspections" => s.max_inspections = num(v).ok_or_else(bad)? as usize,
                "max_term_freq" => s.max_term_freq = num(v).ok_or_else(bad)? as f32,
                "prefix_length" | "prefix_len" => s.prefix_length = num(v).ok_or_else(bad)? as usize,
                "min_word_length" | "min_word_len" => s.min_word_length = num(v).ok_or_else(bad)? as usize,
                "min_doc_freq" => s.min_doc_freq = num(v).ok_or_else(bad)? as f32,
                "lowercase_terms" => s.lowercase = v.as_bool().unwrap_or(false),
                "string_distance" => {
                    let n = v.as_str().unwrap_or("");
                    s.distance = Distance::parse(n).ok_or_else(|| EsError::illegal_argument(format!("Illegal distance option {n}")))?;
                }
                "suggest_mode" => {
                    s.mode = match v.as_str().unwrap_or("") {
                        "missing" => Mode::Missing,
                        "popular" => Mode::Popular,
                        "always" => Mode::Always,
                        other => return Err(EsError::illegal_argument(format!("Illegal suggest mode {other}"))),
                    }
                }
                "sort" => {
                    s.sort_frequency = match v.as_str().unwrap_or("") {
                        "score" => false,
                        "frequency" => true,
                        other => return Err(EsError::illegal_argument(format!("Illegal suggest sort {other}"))),
                    }
                }
                _ => {}
            }
        }
        Ok(s)
    }
}

/// One spell-checker suggestion (`SuggestWord`).
#[derive(Clone, Debug)]
pub struct Word {
    pub term: String,
    pub score: f32,
    /// Document frequency.
    pub freq: u64,
}

/// `DirectSpellChecker.suggestSimilar`: up to `num_sug` corrections of
/// `text`; `threshold` overrides `min_doc_freq` (phrase generators).
pub fn suggest_similar(dict: &Dict, text: &str, num_sug: usize, s: &Spell, threshold: Option<f32>) -> Vec<Word> {
    let text = if s.lowercase { text.to_lowercase() } else { text.to_string() };
    let qchars: Vec<char> = text.chars().collect();
    if s.min_word_length > 0 && qchars.len() < s.min_word_length {
        return Vec::new();
    }
    let (df0, _) = dict.stats(&text);
    let mut docfreq = df0;
    if s.mode == Mode::Missing && docfreq > 0 {
        return Vec::new();
    }
    let max_doc = dict.max_doc();
    if s.max_term_freq >= 1.0 {
        if docfreq as f32 > s.max_term_freq {
            return Vec::new();
        }
    } else if docfreq as f64 > (s.max_term_freq as f64 * max_doc as f64).ceil() {
        return Vec::new();
    }
    if s.mode != Mode::Popular {
        docfreq = 0;
    }
    let thr = threshold.unwrap_or(s.min_doc_freq);
    if thr >= 1.0 {
        docfreq = docfreq.max(thr as u64);
    } else if thr > 0.0 {
        docfreq = docfreq.max(((thr * dict.num_docs() as f32) as i64 - 1).max(0) as u64);
    }
    let inspections = num_sug * s.max_inspections;
    let mut found: HashMap<String, Word> = HashMap::new();
    let first = pass(dict, &text, &qchars, inspections, docfreq, 1, s);
    let more = s.max_edits > 1 && first.len() < inspections;
    for w in first {
        found.insert(w.term.clone(), w);
    }
    if more {
        for w in pass(dict, &text, &qchars, inspections, docfreq, s.max_edits, s) {
            found.entry(w.term.clone()).or_insert(w);
        }
    }
    let mut out: Vec<Word> = found.into_values().collect();
    sort_words(&mut out, s.sort_frequency);
    out.truncate(num_sug);
    out
}

pub fn sort_words(out: &mut [Word], by_freq: bool) {
    out.sort_by(|a, b| {
        let sc = b.score.total_cmp(&a.score);
        let fr = b.freq.cmp(&a.freq);
        let first = if by_freq { fr.then(sc) } else { sc.then(fr) };
        first.then_with(|| a.term.cmp(&b.term))
    });
}

/// One `FuzzyTermsEnum` pass with `ed` max edits.
fn pass(dict: &Dict, text: &str, q: &[char], inspections: usize, docfreq: u64, ed: usize, s: &Spell) -> Vec<Word> {
    let prefix_len = s.prefix_length.max(ed.saturating_sub(1)).min(q.len());
    let prefix: String = q[..prefix_len].iter().collect();
    let mut out: Vec<Word> = Vec::new();
    for (cand, (df, _)) in dict.prefixed(&prefix) {
        if cand == text {
            continue;
        }
        let c: Vec<char> = cand.chars().collect();
        if (c.len() as i64 - q.len() as i64).unsigned_abs() as usize > ed {
            continue;
        }
        let d = osa(&q[prefix_len..], &c[prefix_len.min(c.len())..]);
        if d > ed || d == 0 {
            continue;
        }
        if df <= docfreq {
            continue;
        }
        let score = match s.distance {
            Distance::Internal => 1.0 - d as f32 / (c.len().min(q.len()).max(1)) as f32,
            other => other.similarity(text, &cand),
        };
        if score < s.accuracy {
            continue;
        }
        out.push(Word { term: cand, score, freq: df });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.term.cmp(&b.term)));
    out.truncate(inspections);
    out
}

/// Run one `term` suggestion; returns the entries array.
pub fn run(text: &str, o: &Map<String, Value>, targets: &[SuggestTarget]) -> Result<Value> {
    super::check_keys(
        "term",
        o,
        &[
            "field", "analyzer", "size", "shard_size", "text", "suggest_mode", "accuracy", "sort", "string_distance", "max_edits", "max_inspections", "max_term_freq",
            "prefix_length", "prefix_len", "min_word_length", "min_word_len", "min_doc_freq",
        ],
    )?;
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("The required field option is missing"))?;
    let size = o.get("size").and_then(num).map(|x| x as usize).unwrap_or(5);
    if size == 0 {
        return Err(EsError::illegal_argument("size must be positive"));
    }
    let shard_size = o.get("shard_size").and_then(num).map(|x| x as usize).unwrap_or(size.max(5));
    let spell = Spell::parse(o)?;
    // token → (text, offset, length, options by term)
    let mut entries: Vec<(String, usize, usize, Vec<Word>)> = Vec::new();
    for t in targets {
        let Some(f) = t.lookup.get(field) else { continue };
        let Some(id) = f.core_id else { continue };
        let analyzer = field_analyzer(t, f, o.get("analyzer").and_then(Value::as_str))?;
        let dict = Dict { leaves: t.leaves, field: id };
        let toks = super::tokens(&analyzer, text);
        let fresh = entries.is_empty();
        for (i, tok) in toks.iter().enumerate() {
            let words = suggest_similar(&dict, &tok.text, shard_size, &spell, None);
            if fresh {
                entries.push((tok.text.clone(), tok.offset, tok.length, words));
            } else if let Some(e) = entries.get_mut(i) {
                for w in words {
                    match e.3.iter_mut().find(|x| x.term == w.term) {
                        Some(x) => {
                            x.freq += w.freq;
                            x.score = x.score.max(w.score);
                        }
                        None => e.3.push(w),
                    }
                }
            }
        }
    }
    if entries.is_empty() {
        // Field unmapped everywhere: entries without options (ES still analyzes).
        let a = rustic_core::analysis::Analyzer::standard();
        for tok in super::tokens(&a, text) {
            entries.push((tok.text, tok.offset, tok.length, Vec::new()));
        }
    }
    let out: Vec<Value> = entries
        .into_iter()
        .map(|(text, offset, length, mut words)| {
            sort_words(&mut words, spell.sort_frequency);
            words.truncate(size);
            let opts: Vec<Value> = words.iter().map(|w| json!({"text": w.term, "score": jfloat(w.score), "freq": w.freq})).collect();
            json!({"text": text, "offset": offset, "length": length, "options": opts})
        })
        .collect();
    Ok(Value::Array(out))
}
