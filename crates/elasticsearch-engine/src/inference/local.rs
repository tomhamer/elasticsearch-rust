//! Built-in stand-ins for the `elasticsearch` / `elser` services' models.
//!
//! These are NOT the models they stand in for: dependency-free lexical CPU
//! encoders that produce output of the same *shape* (so mappings, pipelines
//! and clients work) with different semantics. They run only when the node
//! sets `xpack.inference.builtin_fallback: lexical` (see
//! [`super::backend`]); otherwise a model that cannot run fails like an
//! undeployed model in ES.
//!
//! * `.multilingual-e5-small` -> 384-d, L2-normalised, signed feature hashing
//!   of lowercase word unigrams, bigrams and character trigrams
//!   (`hashing-fallback`: lexical similarity only, no cross-lingual or
//!   semantic matching).
//! * `.elser_model_2` -> `{token: weight}`: BM25-style saturating term
//!   weights over the same word tokenisation plus light-stemmed tokens
//!   (`lexical-fallback`: no learned expansion).
//! * `.rerank-v1` -> token-overlap F-score (`lexical-fallback`).
//!
//! The chunkers here are used by every service.

use serde_json::{json, Map, Value};

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
    // Each window must start past the previous one: an overlap of the whole
    // window would emit every growing prefix of the text (finding 126).
    // Validated settings never get here ([`try_chunk`] refuses them).
    let overlap = overlap.min(size - 1);
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

/// [`chunk`], refusing settings that cannot make progress: ES's
/// `WordBoundaryChunker` rejects a word overlap above half the chunk size.
pub fn try_chunk(text: &str, c: &Chunking) -> crate::error::Result<Vec<(usize, usize)>> {
    if c.strategy == "word" && (c.max_chunk_size == 0 || c.overlap > c.max_chunk_size / 2) {
        return Err(crate::error::EsError::illegal_argument(format!(
            "Invalid chunking parameters, overlap [{}] must be < chunk size / 2 [{}]",
            c.overlap,
            c.max_chunk_size / 2
        )));
    }
    Ok(chunk(text, c))
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
    // sentence strategy (ES `SentenceBoundaryChunker`): sentences packed
    // up to max words, with sentence_overlap (0 or 1) sentences repeated
    // at the start of the next. Words are counted as the standard
    // tokenizer's tokens (each CJK ideograph is one, so unspaced Chinese
    // or Japanese is not "one word"), sentences end at `.`/`!`/`?` before
    // whitespace and at CJK full stops, and a sentence longer than the
    // limit is split on word boundaries (finding 359), so no chunk
    // exceeds `max_chunk_size` words.
    let sentences = sentence_spans(text, max);
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

/// `(start, end, words)` of each sentence of `text` (trimmed of
/// surrounding whitespace), sentences over `max` words split into word
/// windows of at most `max` words.
fn sentence_spans(text: &str, max: usize) -> Vec<(usize, usize, usize)> {
    use rustic_core::analysis::tokenizer::StandardTokenizer;
    use rustic_core::analysis::Tokenizer;
    let tokens = StandardTokenizer.tokenize(text);
    // Sentence boundaries: byte offsets just past each terminator.
    let mut bounds = Vec::new();
    let mut it = text.char_indices().peekable();
    while let Some((i, ch)) = it.next() {
        let end = i + ch.len_utf8();
        let ends = match ch {
            '。' | '！' | '？' | '｡' => true,
            '.' | '!' | '?' => it.peek().is_none_or(|(_, n)| n.is_whitespace()),
            _ => false,
        };
        if ends {
            bounds.push(end);
        }
    }
    bounds.push(text.len());
    let mut out = Vec::new();
    let mut from = 0usize;
    let mut ti = 0usize;
    for b in bounds {
        if b <= from {
            continue;
        }
        let raw = &text[from..b];
        let lead = raw.len() - raw.trim_start().len();
        let (s, e) = (from + lead, from + raw.trim_end().len());
        from = b;
        if s >= e {
            continue;
        }
        let first = ti;
        while ti < tokens.len() && (tokens[ti].offset_from as usize) < e {
            ti += 1;
        }
        let words = ti - first;
        if words <= max {
            out.push((s, e, words));
            continue;
        }
        // Too long for one chunk: word windows of the sentence (ES
        // `splitLongSentence`), each at most `max` words.
        let windows = word_chunks(&text[s..e], max, 0);
        for (ws, we) in windows {
            let (ws, we) = (s + ws, s + we);
            let n = tokens[first..ti].iter().filter(|t| (t.offset_from as usize) >= ws && (t.offset_from as usize) < we).count();
            let piece = &text[ws..we];
            let lead = piece.len() - piece.trim_start().len();
            out.push((ws + lead, ws + piece.trim_end().len(), n));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Finding 126: word chunking whose overlap is the whole window is
    /// refused (ES's `WordBoundaryChunker` check), and even when called
    /// directly the chunker moves forward with bounded output; a valid
    /// size 10 / overlap 5 window is unchanged.
    #[test]
    fn word_chunking_always_makes_progress() {
        let words: Vec<String> = (0..1_000).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let bad = Chunking { strategy: "word".into(), max_chunk_size: 10, overlap: 10, sentence_overlap: 0 };
        let e = try_chunk(&text, &bad).unwrap_err();
        assert_eq!(e.reason, "Invalid chunking parameters, overlap [10] must be < chunk size / 2 [5]");
        assert!(try_chunk(&text, &Chunking { overlap: 6, ..bad.clone() }).is_err());
        for overlap in [10usize, 50, 9] {
            let chunks = word_chunks(&text, 10, overlap);
            assert!(chunks.windows(2).all(|w| w[1].0 > w[0].0), "overlap {overlap}: starts must advance");
            let total: usize = chunks.iter().map(|(s, e)| e - s).sum();
            assert!(total <= 12 * text.len(), "overlap {overlap}: {total} bytes of chunks for {} bytes", text.len());
        }
        let ok = Chunking { strategy: "word".into(), max_chunk_size: 10, overlap: 5, sentence_overlap: 0 };
        let chunks: Vec<&str> = try_chunk(&text, &ok).unwrap().into_iter().map(|(s, e)| &text[s..e]).collect();
        assert_eq!(chunks[0], words[..10].join(" "));
        assert_eq!(chunks[1], format!(" {}", words[5..15].join(" ")));
        assert_eq!(chunks.len(), 199);
        assert_eq!(chunks, word_chunks(&text, 10, 5).into_iter().map(|(s, e)| &text[s..e]).collect::<Vec<_>>());
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

    /// Finding 359: oversized sentences and unspaced CJK text are split so
    /// that no chunk exceeds the word limit, and every word is covered.
    #[test]
    fn sentence_chunking_bounds_every_chunk() {
        use rustic_core::analysis::tokenizer::StandardTokenizer;
        use rustic_core::analysis::Tokenizer;
        let words = |s: &str| StandardTokenizer.tokenize(s).len();
        let c = Chunking::default();
        let check = |text: &str, min_chunks: usize| {
            let ch = chunk(text, &c);
            assert!(ch.len() >= min_chunks, "{} chunks", ch.len());
            for &(s, e) in &ch {
                assert!(words(&text[s..e]) <= 250, "chunk of {} words", words(&text[s..e]));
            }
            // Every word is inside some chunk.
            for t in StandardTokenizer.tokenize(text) {
                assert!(ch.iter().any(|&(s, e)| s <= t.offset_from as usize && t.offset_to as usize <= e), "word at {} uncovered", t.offset_from);
            }
        };
        // 600 words, no sentence punctuation.
        let prose: Vec<String> = (0..600).map(|i| format!("w{i}")).collect();
        check(&prose.join(" "), 3);
        // 900 ideographs without whitespace: one sentence, then with full stops.
        let cjk: String = "中文分词测试句子".chars().cycle().take(900).collect();
        check(&cjk, 4);
        let cjk_stops: String = (0..9).map(|_| format!("{}。", "日本語の文章です".chars().cycle().take(99).collect::<String>())).collect();
        check(&cjk_stops, 4);
        // Short ordinary sentences are unchanged.
        let text = "One two. Three four! Five?";
        let c3 = Chunking { strategy: "sentence".into(), max_chunk_size: 3, overlap: 0, sentence_overlap: 1 };
        let ch: Vec<&str> = chunk(text, &c3).into_iter().map(|(s, e)| &text[s..e]).collect();
        assert_eq!(ch, vec!["One two.", "Three four! Five?"]);
        // A decimal point is not a sentence end.
        let ch: Vec<&str> = chunk("Pi is 3.14 today. Yes.", &Chunking { strategy: "sentence".into(), max_chunk_size: 4, overlap: 0, sentence_overlap: 0 })
            .into_iter()
            .map(|(s, e)| &"Pi is 3.14 today. Yes."[s..e])
            .collect();
        assert_eq!(ch, vec!["Pi is 3.14 today.", "Yes."]);
    }

    #[test]
    fn rerank_prefers_overlap() {
        assert!(rerank_score("fox jumps", "the fox jumps high") > rerank_score("fox jumps", "database revenue"));
    }
}
