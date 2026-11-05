//! ES plain highlighter: Lucene `Highlighter.getBestTextFragments` with a
//! `QueryScorer` (fragment score = sum of the weights of the distinct query
//! terms in it) and `SimpleSpanFragmenter` (default; never splits a phrase
//! match) or `SimpleFragmenter` (`fragmenter: simple`); one value at a time.

use super::matches::{Match, Tok};
use super::{encode, u16s, Content, HighlightOpts};
use std::collections::{HashMap, HashSet};

struct Frag {
    start: usize,
    end: usize,
    score: f32,
    num: usize,
}

pub(crate) fn highlight(content: &Content, toks: &[Tok], ms: &[Match], o: &HighlightOpts) -> Vec<String> {
    // matched token offsets -> weight; positional spans per term text
    let mut hit: HashMap<(usize, usize), f32> = HashMap::new();
    let mut spans: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
    for m in ms {
        for p in &m.parts {
            let w = hit.entry(*p).or_insert(0.0);
            *w = w.max(m.weight);
        }
        if m.positional && m.parts.len() > 1 {
            let pos: Vec<u32> = toks.iter().filter(|t| m.parts.contains(&(t.start, t.end))).map(|t| t.pos).collect();
            if let (Some(a), Some(b)) = (pos.iter().min(), pos.iter().max()) {
                for t in &m.tokens {
                    spans.entry(t.clone()).or_default().push((*a, *b));
                }
            }
        }
    }
    let nf = o.number_of_fragments;
    let per_value = if nf == 0 { 1 } else { nf };
    let mut all: Vec<(String, f32)> = Vec::new();
    for &(vs, ve) in &content.values {
        let text = &content.text[vs..ve];
        let vt: Vec<&Tok> = toks.iter().filter(|t| t.start >= vs && t.end <= ve).collect();
        let frags = best_fragments(text, vs, &vt, &hit, &spans, o, per_value);
        all.extend(frags);
    }
    if o.order_score {
        all.sort_by(|a, b| ((b.1 - a.1).round() as i64).cmp(&0));
    }
    let out: Vec<String> = if nf == 0 && content.values.len() > 1 && !all.is_empty() {
        all.into_iter().map(|x| x.0).collect()
    } else {
        let n = all.len().min(per_value);
        all.into_iter().take(n).map(|x| x.0).collect()
    };
    if out.is_empty() && o.no_match_size > 0 {
        if let Some(&(vs, ve)) = content.values.first() {
            let text = &content.text[vs..ve];
            let vt: Vec<&Tok> = toks.iter().filter(|t| t.start >= vs && t.end <= ve).collect();
            let end = good_end(text.len(), &vt, vs, o.no_match_size);
            if end > 0 {
                return vec![u16s(&text[..end])];
            }
        }
    }
    out
}

/// ES `PlainHighlighter.findGoodEndForNoHighlightExcerpt`.
fn good_end(len: usize, toks: &[&Tok], base: usize, no_match: usize) -> usize {
    let mut end: isize = -1;
    for t in toks {
        let e = t.end - base;
        if e >= no_match {
            if e == no_match {
                end = no_match as isize;
            }
            return end.max(0) as usize;
        }
        end = e as isize;
    }
    len
}

#[allow(clippy::too_many_arguments)]
fn best_fragments(
    text: &[u16],
    base: usize,
    toks: &[&Tok],
    hit: &HashMap<(usize, usize), f32>,
    spans: &HashMap<String, Vec<(u32, u32)>>,
    o: &HighlightOpts,
    max_frags: usize,
) -> Vec<(String, f32)> {
    let mut new_text = String::new();
    let mut frags: Vec<Frag> = vec![Frag { start: 0, end: 0, score: 0.0, num: 0 }];
    let mut found: HashSet<&str> = HashSet::new();
    let mut total = 0f32;
    let pre = o.pre_tag(0).to_string();
    let post = o.post_tag(0).to_string();
    let html = o.html_encode;
    let frag_size = o.fragment_size;
    let text_size = text.len();
    let simple = o.fragmenter.as_deref() == Some("simple");
    let null_frag = o.number_of_fragments == 0;
    // fragmenter state
    let mut fpos: i64 = -1;
    let mut wait_for: i64 = -1;
    let mut cur_num_frags = 1usize;
    let mut prev_pos: Option<u32> = None;
    let mut last_end = 0usize;
    // token group
    let mut g_start = 0usize;
    let mut g_end = 0usize;
    let mut g_n = 0usize;
    let mut g_score = 0f32;
    let flush = |new_text: &mut String, g_start: usize, g_end: usize, g_score: f32, last_end: &mut usize| {
        let tok = u16s(&text[g_start..g_end]);
        let marked = if g_score > 0.0 { format!("{pre}{}{post}", encode(&tok, html)) } else { encode(&tok, html) };
        if g_start > *last_end {
            new_text.push_str(&encode(&u16s(&text[*last_end..g_start]), html));
        }
        new_text.push_str(&marked);
        *last_end = (*last_end).max(g_end);
    };
    for t in toks {
        let (s, e) = (t.start - base, t.end - base);
        if s > text_size || e > text_size {
            continue;
        }
        if g_n > 0 && s >= g_end {
            flush(&mut new_text, g_start, g_end, g_score, &mut last_end);
            g_n = 0;
            g_score = 0.0;
            // isNewFragment, with the current token's attributes
            let inc = match prev_pos {
                Some(p) => t.pos as i64 - p as i64,
                None => 1,
            };
            let is_new = if null_frag {
                false
            } else if simple {
                let n = e >= frag_size * cur_num_frags;
                if n {
                    cur_num_frags += 1;
                }
                n
            } else {
                fpos += inc;
                let mut blocked = false;
                if wait_for <= fpos {
                    wait_for = -1;
                } else if wait_for != -1 {
                    blocked = true;
                }
                if blocked {
                    false
                } else {
                    if let Some(sp) = spans.get(&t.text) {
                        for (a, b) in sp {
                            if *a as i64 == fpos {
                                wait_for = *b as i64 + 1;
                                break;
                            }
                        }
                    }
                    let n = e >= frag_size * cur_num_frags && text_size.saturating_sub(e) >= (frag_size >> 1);
                    if n {
                        cur_num_frags += 1;
                    }
                    n
                }
            };
            if is_new {
                let last = frags.last_mut().unwrap();
                last.score = total;
                last.end = new_text.len();
                frags.push(Frag { start: new_text.len(), end: new_text.len(), score: 0.0, num: frags.len() });
                found.clear();
                total = 0.0;
            }
        }
        prev_pos = Some(t.pos);
        // token score
        let ts = hit.get(&(t.start, t.end)).copied().unwrap_or(0.0);
        if ts > 0.0 && found.insert(t.text.as_str()) {
            total += ts;
        }
        if g_n == 0 {
            g_start = s;
            g_end = e;
        } else {
            g_start = g_start.min(s);
            g_end = g_end.max(e);
        }
        g_n += 1;
        g_score += ts;
    }
    frags.last_mut().unwrap().score = total;
    if g_n > 0 {
        flush(&mut new_text, g_start, g_end, g_score, &mut last_end);
    }
    if last_end < text_size {
        new_text.push_str(&encode(&u16s(&text[last_end..]), html));
    }
    frags.last_mut().unwrap().end = new_text.len();
    // top fragments: score desc, then fragment number asc
    frags.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(a.num.cmp(&b.num)));
    frags.truncate(max_frags);
    frags.into_iter().filter(|f| f.score > 0.0).map(|f| (new_text[f.start..f.end].to_string(), f.score)).collect()
}
