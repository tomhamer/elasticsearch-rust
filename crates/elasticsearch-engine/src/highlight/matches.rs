//! Query → match spans over a highlighted field's token stream(s). This is
//! the equivalent of Lucene's highlighting term/span extraction: term-ish
//! leaves match single tokens, phrases / spans / intervals match only where
//! the whole construct matches (positions respected), `must_not` clauses
//! never highlight. In weight-matches mode (ES unified default) a phrase or
//! span match is one span; otherwise its terms are separate matches.

use rustic_core::query::{Query, SpanQuery};
use rustic_core::types::Term;
use std::collections::HashMap;

/// One analyzed token of the content (offsets in UTF-16 units of the joined
/// field content).
#[derive(Clone, Debug)]
pub struct Tok {
    pub text: String,
    pub pos: u32,
    pub start: usize,
    pub end: usize,
}

/// One highlight match.
#[derive(Clone, Debug)]
pub struct Match {
    pub start: usize,
    pub end: usize,
    /// Identity of what matched (term text, or the phrase's terms).
    pub term: String,
    /// Sub-ranges (the terms of a phrase match; one entry otherwise).
    pub parts: Vec<(usize, usize)>,
    /// Token position of each part.
    pub part_pos: Vec<u32>,
    /// Index of the query leaf that produced it (fvh multi-tag `seqnum`).
    pub seq: usize,
    /// Query boost of the leaf.
    pub weight: f32,
    /// Produced by a positional (phrase / span) query.
    pub positional: bool,
    /// Term texts of the matched tokens (plain highlighter scoring).
    pub tokens: Vec<String>,
}

/// A token stream of the content analyzed for one core field; `field: None`
/// matches query leaves of any field (`require_field_match: false`).
pub struct Stream<'a> {
    pub field: Option<u16>,
    pub toks: &'a [Tok],
}

struct Cx<'a, 'b> {
    streams: &'b [Stream<'a>],
    weighted: bool,
    out: Vec<Match>,
    seq: usize,
}

/// Does `q` contain a node that disables weight-matches mode (kNN or a
/// nested / block-join query) anywhere, `must_not` included?
pub fn disables_weight_matches(q: &Query) -> bool {
    let mut found = false;
    q.visit(&mut |n| {
        if matches!(n, Query::Knn { .. } | Query::Nested { .. }) {
            found = true;
        }
    });
    found
}

/// Every match of `q` over the streams.
pub fn find(q: &Query, streams: &[Stream<'_>], weighted: bool) -> Vec<Match> {
    let mut cx = Cx { streams, weighted, out: Vec::new(), seq: 0 };
    walk(q, 1.0, &mut cx);
    let mut out = cx.out;
    out.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)).then(a.term.cmp(&b.term)));
    out.dedup_by(|b, a| a.start == b.start && a.end == b.end && a.term == b.term);
    out
}

fn toks_for<'a>(cx: &Cx<'a, '_>, field: u16) -> Vec<&'a [Tok]> {
    cx.streams.iter().filter(|s| s.field.map(|f| f == field).unwrap_or(true)).map(|s| s.toks).collect()
}

fn term_text(t: &Term) -> String {
    String::from_utf8_lossy(&t.text).into_owned()
}

fn push_tok(cx: &mut Cx, t: &Tok, weight: f32, seq: usize) {
    cx.out.push(Match { start: t.start, end: t.end, term: t.text.clone(), parts: vec![(t.start, t.end)], part_pos: vec![t.pos], seq, weight, positional: false, tokens: vec![t.text.clone()] });
}

fn single(cx: &mut Cx, field: u16, weight: f32, pred: &dyn Fn(&str) -> bool) {
    let seq = cx.seq;
    cx.seq += 1;
    for toks in toks_for(cx, field) {
        for t in toks {
            if pred(&t.text) {
                push_tok(cx, t, weight, seq);
            }
        }
    }
}

fn walk(q: &Query, boost: f32, cx: &mut Cx) {
    match q {
        Query::Term(t) => {
            let want = t.text.clone();
            single(cx, t.field, boost, &|s| s.as_bytes() == want.as_slice());
        }
        Query::TermSet { field, terms, boost: b } => {
            let set: std::collections::HashSet<Vec<u8>> = terms.iter().cloned().collect();
            single(cx, *field, boost * b, &|s| set.contains(s.as_bytes()));
        }
        Query::Phrase { terms, slop } => {
            if terms.is_empty() {
                return;
            }
            let slots: Vec<Vec<String>> = terms.iter().map(|t| vec![term_text(t)]).collect();
            phrase(cx, terms[0].field, &slots, *slop, boost);
        }
        Query::MultiPhrase { field, positions, slop } => {
            let slots: Vec<Vec<String>> = positions.iter().map(|p| p.iter().map(term_text).collect()).collect();
            if !slots.is_empty() {
                phrase(cx, *field, &slots, *slop, boost);
            }
        }
        Query::Prefix { field, prefix, case_insensitive, .. } => {
            let p = String::from_utf8_lossy(prefix).into_owned();
            let ci = *case_insensitive;
            single(cx, *field, boost, &|s| if ci { s.to_lowercase().starts_with(&p.to_lowercase()) } else { s.starts_with(p.as_str()) });
        }
        Query::Wildcard { field, pattern, case_insensitive, .. } => {
            if let Some(re) = wildcard_regex(pattern, *case_insensitive) {
                single(cx, *field, boost, &|s| re.is_match(s));
            }
        }
        Query::Regexp { field, pattern, case_insensitive, .. } => {
            // The query's own term matcher (intersections and complements included).
            use rustic_core::automaton::TermMatcher;
            if let Ok(m) = rustic_core::automaton::regexp::RegexpMatcher::new(pattern, rustic_core::automaton::regexp::RegexpFlags { case_insensitive: *case_insensitive }) {
                single(cx, *field, boost, &|s| m.matches(s.as_bytes()));
            }
        }
        Query::Fuzzy { field, text, fuzziness, prefix_length, transpositions, .. } => {
            let max = fuzzy_edits(fuzziness, text.chars().count());
            let text = text.clone();
            let pl = *prefix_length;
            let tr = *transpositions;
            single(cx, *field, boost, &|s| within_edits(&text, s, max, pl, tr));
        }
        Query::TermRange { field, lower, upper, .. } => {
            let (lo, hi) = (lower.clone(), upper.clone());
            single(cx, *field, boost, &|s| {
                let b = s.as_bytes();
                let lo_ok = match &lo {
                    rustic_core::query::Bound::Included(x) => b >= x.as_slice(),
                    rustic_core::query::Bound::Excluded(x) => b > x.as_slice(),
                    rustic_core::query::Bound::Unbounded => true,
                };
                let hi_ok = match &hi {
                    rustic_core::query::Bound::Included(x) => b <= x.as_slice(),
                    rustic_core::query::Bound::Excluded(x) => b < x.as_slice(),
                    rustic_core::query::Bound::Unbounded => true,
                };
                lo_ok && hi_ok
            });
        }
        Query::Span(sq) => span_matches(cx, sq, boost),
        Query::Intervals { field, rule } => span_matches(cx, &rule.to_span(*field), boost),
        Query::Bool { must, should, filter, .. } => {
            for c in must.iter().chain(should).chain(filter) {
                walk(c, boost, cx);
            }
        }
        Query::Boost { query, boost: b } => walk(query, boost * b, cx),
        Query::ConstScore { query, .. } => walk(query, boost, cx),
        Query::DisMax { queries, .. } => queries.iter().for_each(|c| walk(c, boost, cx)),
        Query::FunctionScore { query, .. } => walk(query, boost, cx),
        Query::Nested { query, .. } => walk(query, boost, cx),
        Query::Custom(c) => {
            for ch in c.children() {
                walk(ch, boost, cx);
            }
        }
        _ => {}
    }
}

pub fn wildcard_regex(pattern: &str, ci: bool) -> Option<regex::Regex> {
    let mut re = String::from(if ci { "(?si)^" } else { "(?s)^" });
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '\\' => {
                if let Some(n) = chars.next() {
                    re.push_str(&regex::escape(&n.to_string()));
                }
            }
            other => re.push_str(&regex::escape(&other.to_string())),
        }
    }
    re.push('$');
    regex::Regex::new(&re).ok()
}

fn fuzzy_edits(f: &rustic_core::query::Fuzziness, len: usize) -> usize {
    use rustic_core::query::Fuzziness;
    match f {
        Fuzziness::Edits(e) => (*e).min(2) as usize,
        Fuzziness::Auto => {
            if len < 3 {
                0
            } else if len < 6 {
                1
            } else {
                2
            }
        }
        Fuzziness::AutoLow(lo, hi) => {
            if len < *lo {
                0
            } else if len < *hi {
                1
            } else {
                2
            }
        }
    }
}

/// Damerau (optional) Levenshtein distance `<= max` with a shared prefix.
pub fn within_edits(a: &str, b: &str, max: usize, prefix: usize, transpositions: bool) -> bool {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().min(prefix) > b.len() || a[..a.len().min(prefix)] != b[..a.len().min(prefix).min(b.len())] {
        return false;
    }
    if (a.len() as isize - b.len() as isize).unsigned_abs() > max {
        return false;
    }
    if a == b {
        return true;
    }
    // Only the diagonal band `|i - j| <= max` can stay within `max` edits:
    // `O(len * max)` work and three rows of workspace, never a full matrix.
    let (n, m) = (a.len(), b.len());
    let inf = max + 1;
    let mut prev2: Vec<usize> = vec![inf; m + 1];
    let mut prev: Vec<usize> = (0..=m).map(|j| j.min(inf)).collect();
    let mut cur: Vec<usize> = vec![inf; m + 1];
    for i in 1..=n {
        let lo = i.saturating_sub(max).max(1);
        let hi = (i + max).min(m);
        cur[lo - 1] = if lo == 1 { i.min(inf) } else { inf };
        let mut row_min = cur[lo - 1];
        for j in lo..=hi {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut v = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(prev2[j - 2] + 1);
            }
            cur[j] = v.min(inf);
            row_min = row_min.min(cur[j]);
        }
        if hi < m {
            cur[hi + 1] = inf;
        }
        if row_min > max {
            return false;
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m] <= max
}

/// Phrase (or multi-phrase) matches with `slop` over one field's tokens.
fn phrase(cx: &mut Cx, field: u16, slots: &[Vec<String>], slop: u32, boost: f32) {
    let seq = cx.seq;
    cx.seq += 1;
    // An empty slot is a hole (a removed stop word): it matches nothing and
    // only keeps the later terms' distance (Lucene `MultiPhraseQuery`
    // positions), so the phrase's terms carry their own query offsets.
    let offsets: Vec<i64> = slots.iter().enumerate().filter(|(_, alts)| !alts.is_empty()).map(|(i, _)| i as i64).collect();
    let slots: Vec<&Vec<String>> = slots.iter().filter(|alts| !alts.is_empty()).collect();
    if slots.is_empty() {
        return;
    }
    for toks in toks_for(cx, field) {
        // candidate token indices per slot
        let cands: Vec<Vec<usize>> = slots.iter().map(|alts| (0..toks.len()).filter(|&i| alts.iter().any(|a| a == &toks[i].text)).collect()).collect();
        if cands.iter().any(|c| c.is_empty()) {
            continue;
        }
        let mut found: Vec<Vec<usize>> = Vec::new();
        // One work budget for the whole phrase over this field: the
        // candidate tree of a long repetitive text is otherwise exponential.
        let mut budget = PHRASE_WORK;
        for &first in &cands[0] {
            if budget == 0 {
                break;
            }
            let mut chosen = vec![first];
            if let Some(c) = best_combo(toks, &cands, &offsets, 1, &mut chosen, slop as i64, &mut budget) {
                if !found.contains(&c) {
                    found.push(c);
                }
            }
        }
        for combo in found {
            emit_positional(cx, toks, &combo, boost, seq);
        }
    }
}

/// Depth-first search for a combination whose spread
/// `max(pos_i - off_i) - min(pos_i - off_i)` is within `slop` (Lucene sloppy phrase).
/// Combinations [`best_combo`] may try for one phrase over one field's
/// tokens. ES's highlighter walks the phrase's matches in one pass; this
/// search stops at its budget, keeping the best combinations found.
pub(crate) const PHRASE_WORK: usize = 100_000;

fn best_combo(toks: &[Tok], cands: &[Vec<usize>], offsets: &[i64], k: usize, chosen: &mut Vec<usize>, slop: i64, budget: &mut usize) -> Option<Vec<usize>> {
    let spread = |ch: &[usize]| {
        let rel: Vec<i64> = ch.iter().enumerate().map(|(i, &t)| toks[t].pos as i64 - offsets[i]).collect();
        rel.iter().max().unwrap() - rel.iter().min().unwrap()
    };
    if k == cands.len() {
        return Some(chosen.clone());
    }
    let mut best: Option<(i64, Vec<usize>)> = None;
    let mut tried = 0;
    for &c in &cands[k] {
        if *budget == 0 {
            break;
        }
        *budget -= 1;
        if chosen.iter().any(|&x| toks[x].pos == toks[c].pos) {
            continue;
        }
        chosen.push(c);
        if spread(chosen) <= slop {
            tried += 1;
            if let Some(full) = best_combo(toks, cands, offsets, k + 1, chosen, slop, budget) {
                let s = spread(&full);
                if best.as_ref().map(|b| s < b.0).unwrap_or(true) {
                    best = Some((s, full));
                }
            }
        }
        chosen.pop();
        // A zero spread is the best any combination can do.
        if tried > 64 || best.as_ref().is_some_and(|b| b.0 == 0) {
            break;
        }
    }
    best.map(|b| b.1)
}

fn emit_positional(cx: &mut Cx, toks: &[Tok], idx: &[usize], boost: f32, seq: usize) {
    let mut trip: Vec<(usize, usize, u32)> = idx.iter().map(|&i| (toks[i].start, toks[i].end, toks[i].pos)).collect();
    trip.sort();
    let parts: Vec<(usize, usize)> = trip.iter().map(|x| (x.0, x.1)).collect();
    let part_pos: Vec<u32> = trip.iter().map(|x| x.2).collect();
    let texts: Vec<String> = idx.iter().map(|&i| toks[i].text.clone()).collect();
    if cx.weighted {
        let start = parts.iter().map(|p| p.0).min().unwrap_or(0);
        let end = parts.iter().map(|p| p.1).max().unwrap_or(0);
        cx.out.push(Match { start, end, term: texts.join(" "), parts, part_pos, seq, weight: boost, positional: true, tokens: texts });
    } else {
        for &i in idx {
            let t = &toks[i];
            cx.out.push(Match { start: t.start, end: t.end, term: t.text.clone(), parts: vec![(t.start, t.end)], part_pos: vec![t.pos], seq, weight: boost, positional: true, tokens: vec![t.text.clone()] });
        }
    }
}

/// A span: `[start_pos, end_pos)` and the token indices in it.
#[derive(Clone, Debug, PartialEq)]
struct Sp {
    start: u32,
    end: u32,
    toks: Vec<usize>,
}

fn span_field(sq: &SpanQuery) -> Option<u16> {
    match sq {
        SpanQuery::Term(t) => Some(t.field),
        SpanQuery::Prefix { field, .. } | SpanQuery::Wildcard { field, .. } => Some(*field),
        SpanQuery::Near { clauses, .. } => clauses.iter().find_map(span_field),
        SpanQuery::Or(cs) => cs.iter().find_map(span_field),
        SpanQuery::First { inner, .. } => span_field(inner),
        SpanQuery::Not { include, .. } => span_field(include),
    }
}

fn span_matches(cx: &mut Cx, sq: &SpanQuery, boost: f32) {
    let Some(field) = span_field(sq) else { return };
    let seq = cx.seq;
    cx.seq += 1;
    for toks in toks_for(cx, field) {
        let spans = eval_span(sq, toks);
        for s in spans {
            let mut idx = s.toks.clone();
            idx.sort();
            idx.dedup();
            emit_positional(cx, toks, &idx, boost, seq);
        }
    }
}

fn eval_span(sq: &SpanQuery, toks: &[Tok]) -> Vec<Sp> {
    let leaf = |pred: &dyn Fn(&str) -> bool| -> Vec<Sp> {
        toks.iter().enumerate().filter(|(_, t)| pred(&t.text)).map(|(i, t)| Sp { start: t.pos, end: t.pos + 1, toks: vec![i] }).collect()
    };
    match sq {
        SpanQuery::Term(t) => {
            let w = t.text.clone();
            leaf(&|s| s.as_bytes() == w.as_slice())
        }
        SpanQuery::Prefix { prefix, .. } => {
            let p = String::from_utf8_lossy(prefix).into_owned();
            leaf(&|s| s.starts_with(p.as_str()))
        }
        SpanQuery::Wildcard { pattern, .. } => match wildcard_regex(pattern, false) {
            Some(re) => leaf(&|s| re.is_match(s)),
            None => Vec::new(),
        },
        SpanQuery::Or(cs) => {
            let mut v: Vec<Sp> = cs.iter().flat_map(|c| eval_span(c, toks)).collect();
            v.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
            v.dedup();
            v
        }
        SpanQuery::First { inner, end } => eval_span(inner, toks).into_iter().filter(|s| s.end <= *end).collect(),
        SpanQuery::Not { include, exclude, pre, post } => {
            let ex = eval_span(exclude, toks);
            eval_span(include, toks)
                .into_iter()
                .filter(|s| !ex.iter().any(|e| (e.start as i64 - *pre as i64) < s.end as i64 && (e.end as i64 + *post as i64) > s.start as i64))
                .collect()
        }
        SpanQuery::Near { clauses, slop, in_order } => {
            let lists: Vec<Vec<Sp>> = clauses.iter().map(|c| eval_span(c, toks)).collect();
            if lists.is_empty() || lists.iter().any(|l| l.is_empty()) {
                return Vec::new();
            }
            let mut out = Vec::new();
            let mut chosen: Vec<usize> = Vec::new();
            near_dfs(&lists, 0, &mut chosen, *slop as i64, *in_order, &mut out, &mut 0);
            out.sort_by(|a: &Sp, b: &Sp| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
            out.dedup();
            out
        }
    }
}

fn near_dfs(lists: &[Vec<Sp>], k: usize, chosen: &mut Vec<usize>, slop: i64, ordered: bool, out: &mut Vec<Sp>, budget: &mut usize) {
    *budget += 1;
    if *budget > 20_000 {
        return;
    }
    if k == lists.len() {
        let spans: Vec<&Sp> = chosen.iter().enumerate().map(|(i, &c)| &lists[i][c]).collect();
        let start = spans.iter().map(|s| s.start).min().unwrap();
        let end = spans.iter().map(|s| s.end).max().unwrap();
        let len: i64 = spans.iter().map(|s| (s.end - s.start) as i64).sum();
        let gap = end as i64 - start as i64 - len;
        if gap <= slop {
            let mut toks: Vec<usize> = spans.iter().flat_map(|s| s.toks.iter().copied()).collect();
            toks.sort();
            out.push(Sp { start, end, toks });
        }
        return;
    }
    for (i, s) in lists[k].iter().enumerate() {
        if ordered && k > 0 {
            let prev = &lists[k - 1][chosen[k - 1]];
            if s.start < prev.end {
                continue;
            }
            let gap: i64 = (s.start - prev.end) as i64;
            if gap > slop {
                continue;
            }
        }
        if !ordered && chosen.iter().enumerate().any(|(j, &c)| {
            let o = &lists[j][c];
            o.start < s.end && s.start < o.end
        }) {
            continue;
        }
        chosen.push(i);
        near_dfs(lists, k + 1, chosen, slop, ordered, out, budget);
        chosen.pop();
    }
}

/// Term → number of matches of that term (Lucene `OffsetsEnum.freq()`).
pub fn term_freqs(ms: &[Match]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for x in ms {
        *m.entry(x.term.clone()).or_insert(0) += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full-matrix distance the banded one replaces (reference).
    fn full(a: &str, b: &str, transpositions: bool) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let (n, m) = (a.len(), b.len());
        let mut d = vec![vec![0usize; m + 1]; n + 1];
        for (i, row) in d.iter_mut().enumerate() {
            row[0] = i;
        }
        for j in 0..=m {
            d[0][j] = j;
        }
        for i in 1..=n {
            for j in 1..=m {
                let cost = usize::from(a[i - 1] != b[j - 1]);
                let mut v = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
                if transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                    v = v.min(d[i - 2][j - 2] + 1);
                }
                d[i][j] = v;
            }
        }
        d[n][m]
    }

    /// #324: the banded fuzzy check agrees with the full distance and
    /// decides 16,000-char terms in linear work and space.
    #[test]
    fn banded_fuzzy_matches_full_distance() {
        let mut seed = 5u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for _ in 0..3000 {
            let a: String = (0..next() % 8).map(|_| (b'a' + (next() % 3) as u8) as char).collect();
            let b: String = (0..next() % 8).map(|_| (b'a' + (next() % 3) as u8) as char).collect();
            for tr in [false, true] {
                for max in 0..3 {
                    assert_eq!(within_edits(&a, &b, max, 0, tr), full(&a, &b, tr) <= max, "{a:?} {b:?} {max} {tr}");
                }
            }
        }
        let long: String = (0..16_000).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let mut near: Vec<char> = long.chars().collect();
        near[15_000] = '#';
        let near: String = near.into_iter().collect();
        let started = std::time::Instant::now();
        assert!(within_edits(&long, &long, 1, 0, true));
        assert!(within_edits(&long, &near, 1, 0, true));
        assert!(!within_edits(&long, &near, 0, 0, true));
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}
