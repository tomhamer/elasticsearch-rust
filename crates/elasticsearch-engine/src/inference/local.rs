//! CPU-only local models for the `elasticsearch` / `elser` services.
//!
//! Built-in, dependency-free CPU encoders. No model weights are bundled, so
//! the binary stays small and starts instantly; the encoders produce
//! vectors/weights with the *same shapes and model ids* as the hosted models,
//! so mappings, pipelines and clients work unchanged:
//!
//! * `.multilingual-e5-small` -> 384-d, L2-normalised, signed feature hashing
//!   of lowercase word unigrams, bigrams and character trigrams (lexically
//!   sensitive, deterministic; related texts land close together).
//! * `.elser_model_2` -> `{token: weight}` learned-sparse stand-in: BM25-style
//!   saturating term weights over the same word tokenisation, plus
//!   light-stemmed expansion tokens.
//!
//! For model-quality embeddings, point an inference endpoint at an external
//! service. [`BACKEND`] names the active implementation so it can be surfaced
//! in stats.

use serde_json::{json, Map, Value};

pub const BACKEND: &str = "hashing-fallback";
pub const E5_DIMS: usize = 384;

/// Lowercase word tokens (alphanumeric runs).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

const STOP: &[&str] = &["a", "an", "the", "of", "and", "or", "to", "in", "on", "is", "are", "was", "were", "for", "with", "as", "by", "at", "it", "this", "that", "be"];

fn fnv(s: &str, seed: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn stem(t: &str) -> String {
    for suf in ["ing", "edly", "ed", "es", "s", "ly"] {
        if t.len() > suf.len() + 3 && t.ends_with(suf) {
            return t[..t.len() - suf.len()].to_string();
        }
    }
    t.to_string()
}

fn add(v: &mut [f32], feature: &str, w: f32) {
    let h = fnv(feature, 0);
    let idx = (h % v.len() as u64) as usize;
    let sign = if (fnv(feature, 0x9e3779b97f4a7c15) & 1) == 0 { 1.0 } else { -1.0 };
    v[idx] += sign * w;
}

/// 384-d dense embedding (see module docs).
pub fn embed_dense(text: &str, prefix: Option<&str>) -> Vec<f32> {
    let mut v = vec![0f32; E5_DIMS];
    let toks: Vec<String> = tokenize(text).into_iter().map(|t| stem(&t)).collect();
    for (i, t) in toks.iter().enumerate() {
        let w = if STOP.contains(&t.as_str()) { 0.2 } else { 1.0 };
        add(&mut v, &format!("u:{t}"), w);
        if i + 1 < toks.len() {
            add(&mut v, &format!("b:{t} {}", toks[i + 1]), 0.5 * w);
        }
        let padded = format!("^{t}$");
        let chars: Vec<char> = padded.chars().collect();
        for win in chars.windows(3) {
            add(&mut v, &format!("c:{}", win.iter().collect::<String>()), 0.15 * w);
        }
    }
    let _ = prefix;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// ELSER-style sparse expansion `{token: weight}` (see module docs).
pub fn embed_sparse(text: &str) -> Map<String, Value> {
    let toks = tokenize(text);
    let mut tf: Vec<(String, f32)> = Vec::new();
    for t in &toks {
        if STOP.contains(&t.as_str()) || t.len() < 2 {
            continue;
        }
        for form in [t.clone(), stem(t)] {
            match tf.iter_mut().find(|(k, _)| *k == form) {
                Some(e) => e.1 += if form == *t { 1.0 } else { 0.5 },
                None => tf.push((form.clone(), if form == *t { 1.0 } else { 0.5 })),
            }
        }
    }
    let mut out = Map::new();
    // k1-saturated tf, scaled into ELSER's typical 0..3 weight range.
    for (t, f) in tf {
        let w = (f * 2.2) / (f + 1.2) * (1.0 + (t.len() as f32).min(10.0) / 20.0);
        out.insert(t, json!(((w * 1000.0).round() / 1000.0) as f64));
    }
    out
}

/// Rerank score: token overlap F-score (0..1).
pub fn rerank_score(query: &str, doc: &str) -> f64 {
    let q: Vec<String> = tokenize(query).into_iter().filter(|t| !STOP.contains(&t.as_str())).map(|t| stem(&t)).collect();
    let d: Vec<String> = tokenize(doc).into_iter().map(|t| stem(&t)).collect();
    if q.is_empty() || d.is_empty() {
        return 0.0;
    }
    let hit = q.iter().filter(|t| d.contains(t)).count() as f64;
    let recall = hit / q.len() as f64;
    let prec = d.iter().filter(|t| q.contains(t)).count() as f64 / d.len() as f64;
    if recall + prec == 0.0 {
        0.0
    } else {
        2.0 * recall * prec.max(0.05) / (recall + prec.max(0.05))
    }
}

// ---------------------------------------------------------------- chunking

/// Chunking settings (`word` / `sentence`).
#[derive(Clone, Debug, PartialEq)]
pub struct Chunking {
    pub strategy: String,
    pub max_chunk_size: usize,
    pub overlap: usize,
    pub sentence_overlap: usize,
}

impl Default for Chunking {
    fn default() -> Self {
        Chunking { strategy: "sentence".into(), max_chunk_size: 250, overlap: 0, sentence_overlap: 1 }
    }
}

impl Chunking {
    pub fn from_json(v: Option<&Value>) -> Chunking {
        let mut c = Chunking::default();
        if let Some(o) = v.and_then(Value::as_object) {
            if let Some(s) = o.get("strategy").and_then(Value::as_str) {
                c.strategy = s.to_string();
            }
            if let Some(n) = o.get("max_chunk_size").and_then(Value::as_u64) {
                c.max_chunk_size = n as usize;
            }
            if let Some(n) = o.get("overlap").and_then(Value::as_u64) {
                c.overlap = n as usize;
            }
            if let Some(n) = o.get("sentence_overlap").and_then(Value::as_u64) {
                c.sentence_overlap = n as usize;
            }
        }
        c
    }
}

/// `WordBoundaryChunker.chunkPositions`: windows of `size` words (ICU word
/// boundaries; words are the standard tokenizer's tokens) overlapping by
/// `overlap` words. A window ends at its last word's end and the next one
/// starts right after the end of its first non-overlapping word (so it
/// carries the separator in front of its first word).
pub fn word_chunks(text: &str, size: usize, overlap: usize) -> Vec<(usize, usize)> {
    use rustic_core::analysis::tokenizer::StandardTokenizer;
    use rustic_core::analysis::Tokenizer;
    let size = size.max(1);
    let less = size.saturating_sub(overlap);
    let mut out = Vec::new();
    let (mut in_chunk, mut next_start, mut start, mut since_mark) = (0usize, 0usize, 0usize, 0usize);
    for t in StandardTokenizer.tokenize(text) {
        let boundary = t.offset_to as usize;
        in_chunk += 1;
        since_mark += 1;
        if in_chunk >= size {
            out.push((start, boundary));
            in_chunk = overlap;
            if overlap == 0 {
                next_start = boundary;
            }
            start = next_start;
        }
        if since_mark == less {
            next_start = boundary;
            since_mark = 0;
        }
    }
    if in_chunk > overlap || out.is_empty() {
        out.push((start, text.len()));
    }
    out
}

/// Split `text` into `(start, end)` byte offsets of chunks.
pub fn chunk(text: &str, c: &Chunking) -> Vec<(usize, usize)> {
    if c.strategy == "none" {
        return vec![(0, text.len())];
    }
    if c.strategy == "word" {
        return word_chunks(text, c.max_chunk_size, c.overlap);
    }
    // word boundaries: (start, end) of whitespace-separated words
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                words.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        words.push((s, text.len()));
    }
    if words.is_empty() {
        return Vec::new();
    }
    let max = c.max_chunk_size.max(1);
    if c.strategy == "word" {
        let overlap = c.overlap.min(max.saturating_sub(1));
        let step = max - overlap;
        let mut out = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let end = (i + max).min(words.len());
            out.push((words[i].0, words[end - 1].1));
            if end == words.len() {
                break;
            }
            i += step;
        }
        return out;
    }
    // sentence strategy: sentences packed up to max words, with
    // sentence_overlap (0 or 1) sentences repeated at the start of the next.
    let mut sentences: Vec<(usize, usize, usize)> = Vec::new(); // start, end, word count
    let mut s_start = words[0].0;
    let mut count = 0;
    for (wi, &(ws, we)) in words.iter().enumerate() {
        count += 1;
        let w = &text[ws..we];
        let ends = w.ends_with('.') || w.ends_with('!') || w.ends_with('?');
        if ends || wi == words.len() - 1 {
            sentences.push((s_start, we, count));
            count = 0;
            if let Some(n) = words.get(wi + 1) {
                s_start = n.0;
            }
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < sentences.len() {
        let mut j = i;
        let mut n = 0;
        while j < sentences.len() && (n + sentences[j].2 <= max || j == i) {
            n += sentences[j].2;
            j += 1;
        }
        out.push((sentences[i].0, sentences[j - 1].1));
        if j >= sentences.len() {
            break;
        }
        i = if c.sentence_overlap > 0 && j - i > 1 { j - 1 } else { j };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dense_is_384_unit_and_related_texts_are_closer() {
        let a = embed_dense("the quick brown fox jumps over the lazy dog", None);
        assert_eq!(a.len(), 384);
        assert!((cos(&a, &a) - 1.0).abs() < 1e-4);
        let near = embed_dense("a quick brown fox jumping", None);
        let far = embed_dense("quarterly revenue forecast for cloud databases", None);
        assert!(cos(&a, &near) > cos(&a, &far));
    }

    #[test]
    fn sparse_has_expansion_tokens() {
        let m = embed_sparse("running dogs");
        assert!(m.contains_key("running") && m.contains_key("runn") && m.contains_key("dogs"));
    }

    #[test]
    fn word_chunking_with_overlap() {
        let text = "a b c d e f g h";
        let c = Chunking { strategy: "word".into(), max_chunk_size: 4, overlap: 2, sentence_overlap: 0 };
        let ch: Vec<&str> = chunk(text, &c).into_iter().map(|(s, e)| &text[s..e]).collect();
        assert_eq!(ch, vec!["a b c d", " c d e f", " e f g h"]);
    }

    #[test]
    fn sentence_chunking_packs_sentences() {
        let text = "One two. Three four. Five six.";
        let c = Chunking { strategy: "sentence".into(), max_chunk_size: 4, overlap: 0, sentence_overlap: 0 };
        let ch: Vec<&str> = chunk(text, &c).into_iter().map(|(s, e)| &text[s..e]).collect();
        assert_eq!(ch, vec!["One two. Three four.", "Five six."]);
    }

    #[test]
    fn rerank_prefers_overlap() {
        assert!(rerank_score("fox jumps", "the fox jumps high") > rerank_score("fox jumps", "database revenue"));
    }
}
