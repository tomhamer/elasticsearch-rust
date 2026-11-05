//! ES unified highlighter: Lucene `FieldHighlighter.highlightOffsetsEnums`
//! passage building over a break iterator, `PassageScorer` (BM25 with
//! k1 = 1.2, b = 0.75, pivot = 87), top-N passage selection, and ES's
//! `CustomPassageFormatter` / no-match summary.

use super::breaks::{word_boundaries, BreakIter, Scanner};
use super::matches::{term_freqs, Match};
use super::{encode, java_trim, u16s, Content, HighlightOpts};

const K1: f32 = 1.2;
const B: f32 = 0.75;
const PIVOT: f32 = 87.0;

fn weight(content_len: usize, total_term_freq: usize) -> f32 {
    let num_docs = 1.0 + content_len as f32 / PIVOT;
    (K1 + 1.0) * (1.0f64 + (num_docs as f64 + 0.5) / (total_term_freq as f64 + 0.5)).ln() as f32
}

fn tf(freq: usize, passage_len: usize) -> f32 {
    let norm = K1 * ((1.0 - B) + B * (passage_len as f32 / PIVOT));
    freq as f32 / (freq as f32 + norm)
}

fn norm(start: usize) -> f32 {
    1.0 + 1.0 / ((PIVOT as f64 + start as f64).ln() as f32)
}

#[derive(Clone, Debug, Default)]
struct Passage {
    start: usize,
    end: usize,
    score: f32,
    /// (start, end, term, freq in doc)
    matches: Vec<(usize, usize, String, usize)>,
}

fn score(p: &Passage, content_len: usize) -> f32 {
    // distinct terms in order of first appearance
    let mut terms: Vec<(&str, usize, usize)> = Vec::new(); // term, freq in passage, freq in doc
    for (_, _, t, f) in &p.matches {
        match terms.iter_mut().find(|x| x.0 == t.as_str()) {
            Some(x) => x.1 += 1,
            None => terms.push((t.as_str(), 1, *f)),
        }
    }
    let len = p.end - p.start;
    let mut s = 0f32;
    for (_, in_p, in_doc) in terms {
        s += tf(in_p, len) * weight(content_len, in_doc);
    }
    s * norm(p.start)
}

/// Lucene `FieldHighlighter.maybeAddPassage` over a bounded min-queue
/// ordered by (score, start).
fn add_passage(queue: &mut Vec<Passage>, mut p: Passage, max: usize, content_len: usize) {
    p.score = score(&p, content_len);
    let less = |a: &Passage, b: &Passage| a.score < b.score || (a.score == b.score && a.start < b.start);
    if queue.len() == max {
        let min = queue.iter().fold(None::<&Passage>, |m, x| match m {
            Some(y) if !less(x, y) => Some(y),
            _ => Some(x),
        });
        if let Some(m) = min {
            if p.score < m.score {
                return;
            }
        }
    }
    queue.push(p);
    if queue.len() > max {
        // poll the smallest
        let mut mi = 0;
        for i in 1..queue.len() {
            if less(&queue[i], &queue[mi]) {
                mi = i;
            }
        }
        queue.remove(mi);
    }
}

pub(crate) fn highlight(content: &Content, ms: &[Match], o: &HighlightOpts, tokenized: bool) -> Vec<String> {
    let nf = o.number_of_fragments;
    let (kind, max_passages) = if nf == 0 || !tokenized {
        (Scanner::Whole, if nf == 0 { usize::MAX - 1 } else { nf })
    } else {
        let k = match o.boundary_scanner.as_deref() {
            Some("word") => Scanner::Word,
            _ => {
                if o.fragment_size > 0 {
                    Scanner::BoundedSentence(o.fragment_size)
                } else {
                    Scanner::Sentence
                }
            }
        };
        (k, nf)
    };
    let text = &content.text;
    let content_len = text.len();
    let freqs = term_freqs(ms);
    let mut bi = BreakIter::new(text, &content.values, &kind);
    let mut queue: Vec<Passage> = Vec::new();
    let mut cur: Option<Passage> = None;
    let mut last_end = 0usize;
    for m in ms {
        let (start, end) = (m.start, m.end);
        if start < content_len && end > content_len {
            continue;
        }
        let new_passage = match &cur {
            None => true,
            Some(p) => start >= p.end,
        };
        if new_passage {
            if let Some(p) = cur.take() {
                add_passage(&mut queue, p, max_passages, content_len);
            }
            if start >= content_len {
                break;
            }
            let ps = bi.preceding(start + 1).max(last_end);
            last_end = bi.following(start);
            cur = Some(Passage { start: ps, end: last_end.min(content_len), ..Default::default() });
        }
        if let Some(p) = cur.as_mut() {
            p.matches.push((start, end, m.term.clone(), *freqs.get(&m.term).unwrap_or(&1)));
        }
    }
    if let Some(p) = cur.take() {
        add_passage(&mut queue, p, max_passages, content_len);
    }
    let mut passages = queue;
    passages.sort_by_key(|p| p.start);
    if passages.is_empty() {
        if o.no_match_size > 0 {
            if let Some(p) = no_match_passage(content, o.no_match_size) {
                let s = encode(&u16s(&text[p.0..p.1]), o.html_encode);
                let s = java_trim(&s).to_string();
                return if s.is_empty() { vec![] } else { vec![s] };
            }
        }
        return Vec::new();
    }
    let mut snippets: Vec<(String, f32)> = passages.iter().map(|p| (format(text, p, o), p.score)).filter(|(s, _)| !s.trim().is_empty()).collect();
    if o.order_score {
        snippets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    let n = if nf == 0 { snippets.len() } else { nf.min(snippets.len()) };
    snippets.into_iter().take(n).map(|x| x.0).collect()
}

/// ES `CustomPassageFormatter.format` for one passage.
fn format(text: &[u16], p: &Passage, o: &HighlightOpts) -> String {
    let mut sb = String::new();
    let mut pos = p.start;
    let mut ms = p.matches.clone();
    ms.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut i = 0;
    while i < ms.len() {
        let start = ms[i].0.max(pos);
        if start >= p.end {
            break;
        }
        sb.push_str(&encode(&u16s(&text[pos..start]), o.html_encode));
        let mut end = ms[i].1;
        while i + 1 < ms.len() && ms[i + 1].0 < end {
            i += 1;
            end = end.max(ms[i].1);
        }
        let end = end.min(p.end).max(start);
        sb.push_str(o.pre_tag(0));
        sb.push_str(&encode(&u16s(&text[start..end]), o.html_encode));
        sb.push_str(o.post_tag(0));
        pos = end;
        i += 1;
    }
    let tail_end = pos.max(p.end).min(text.len());
    sb.push_str(&encode(&u16s(&text[pos..tail_end]), o.html_encode));
    if sb.ends_with('\u{2029}') || sb.ends_with('\u{0}') {
        sb.pop();
    }
    // ES 8.14+: whole-field highlighting (`number_of_fragments: 0`) keeps surrounding whitespace.
    if o.number_of_fragments == 0 {
        return sb;
    }
    java_trim(&sb).to_string()
}

/// ES `CustomFieldHighlighter.getSummaryPassagesNoHighlight`.
fn no_match_passage(content: &Content, no_match_size: usize) -> Option<(usize, usize)> {
    let text = &content.text;
    let mut pos = 0;
    while pos < text.len() && text[pos] == super::MULTIVAL_SEP {
        pos += 1;
    }
    if pos >= text.len() {
        return None;
    }
    let mut end = text[pos..].iter().position(|&c| c == super::MULTIVAL_SEP).map(|i| i + pos).unwrap_or(text.len());
    if no_match_size + pos < end {
        let wb = word_boundaries(text);
        end = super::breaks::following(&wb, no_match_size + pos);
    }
    Some((pos, end))
}
