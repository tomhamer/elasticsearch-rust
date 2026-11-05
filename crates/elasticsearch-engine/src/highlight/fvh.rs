//! ES fast vector highlighter: Lucene `SimpleFragListBuilder` (margin =
//! `fragment_offset`, default 6) / `SingleFragListBuilder`
//! (`number_of_fragments: 0`), discrete multi-value fragments,
//! `SimpleFragmentsBuilder` / `ScoreOrderFragmentsBuilder`, and the
//! `SimpleBoundaryScanner` (`boundary_chars`, `boundary_max_scan`) or
//! break-iterator boundary scanners. Phrase matches keep one tag per term.

use super::breaks::{sentence_boundaries, word_boundaries};
use super::matches::Match;
use super::{encode, u16s, Content, HighlightOpts};
use crate::error::{EsError, Result};

#[derive(Clone, Debug)]
struct Info {
    start: usize,
    end: usize,
    toffs: Vec<(usize, usize)>,
    seq: usize,
    boost: f32,
}

#[derive(Clone, Debug)]
struct FragInfo {
    start: usize,
    end: usize,
    subs: Vec<Info>,
    boost: f32,
}

fn infos(ms: &[Match]) -> Vec<Info> {
    let mut out: Vec<Info> = Vec::new();
    for m in ms {
        // WeightedPhraseInfo: terms at consecutive positions share one tag
        let mut toffs: Vec<(usize, usize)> = Vec::new();
        let mut prev: Option<u32> = None;
        for (p, &pos) in m.parts.iter().zip(&m.part_pos) {
            match (toffs.last_mut(), prev) {
                (Some(last), Some(pp)) if pos == pp + 1 => last.1 = p.1,
                _ => toffs.push(*p),
            }
            prev = Some(pos);
        }
        out.push(Info { start: m.start, end: m.end, toffs, seq: m.seq, boost: m.weight });
    }
    out.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
    out
}

fn frag_list(infos: &[Info], frag_size: usize, margin: usize) -> Vec<FragInfo> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut start_offset = 0usize;
    let accept = |inf: &Info, match_len: usize| inf.toffs.len() <= 1 || match_len <= frag_size;
    while i < infos.len() {
        let pi = &infos[i];
        if pi.start < start_offset {
            i += 1;
            continue;
        }
        let mut wpil: Vec<Info> = Vec::new();
        let cur_start = pi.start;
        let mut cur_end = pi.end;
        let span_start = cur_start.saturating_sub(margin).max(start_offset);
        let span_end = cur_end.max(span_start + frag_size);
        if accept(pi, cur_end - cur_start) {
            wpil.push(pi.clone());
        }
        i += 1;
        while i < infos.len() {
            let n = &infos[i];
            if n.end <= span_end {
                cur_end = n.end;
                if accept(n, cur_end - cur_start) {
                    wpil.push(n.clone());
                }
                i += 1;
            } else {
                break;
            }
        }
        if wpil.is_empty() {
            continue;
        }
        let match_len = cur_end - cur_start;
        let new_margin = frag_size.saturating_sub(match_len) / 2;
        let ss = (cur_start as isize - new_margin as isize).max(start_offset as isize) as usize;
        let se = ss + match_len.max(frag_size);
        start_offset = se;
        let boost = wpil.iter().map(|x| x.boost).sum();
        out.push(FragInfo { start: ss, end: se, subs: wpil, boost });
    }
    out
}

/// Lucene `BaseFragmentsBuilder.discreteMultiValueHighlighting`.
fn discrete(frags: Vec<FragInfo>, content: &Content) -> Vec<FragInfo> {
    let mut per_value: Vec<Vec<FragInfo>> = vec![Vec::new(); content.values.len()];
    'outer: for fi in frags {
        let mut fi = fi;
        let mut field_end = 0usize;
        for (vi, (vs, ve)) in content.values.iter().enumerate() {
            let len = ve - vs;
            if len == 0 {
                field_end += 1;
                continue;
            }
            let field_start = field_end;
            field_end += len + 1;
            if fi.start >= field_start && fi.end >= field_start && fi.start <= field_end && fi.end <= field_end {
                per_value[vi].push(fi);
                continue 'outer;
            }
            if fi.subs.is_empty() {
                continue 'outer;
            }
            let first = fi.subs[0].toffs[0];
            if fi.start >= field_end || first.0 >= field_end {
                continue;
            }
            let frag_start = if fi.start > field_start && fi.start < field_end { fi.start } else { field_start };
            let frag_end = if fi.end > field_start && fi.end < field_end { fi.end } else { field_end };
            let mut subs = Vec::new();
            let mut boost = 0f32;
            for sub in fi.subs.iter_mut() {
                let mut taken = Vec::new();
                let mut keep = Vec::new();
                for t in sub.toffs.drain(..) {
                    if t.0 >= field_end {
                        keep.push(t);
                        continue;
                    }
                    let after = t.0 >= field_start;
                    let before = t.1 < field_end;
                    if after && before {
                        taken.push(t);
                    } else if after {
                        taken.push((t.0, field_end - 1));
                        keep.push((field_end, t.1));
                    } else if before {
                        taken.push((field_start, t.1));
                    } else {
                        taken.push((field_start, field_end - 1));
                        keep.push((field_end, t.1));
                    }
                }
                sub.toffs = keep;
                if !taken.is_empty() {
                    let mut s = sub.clone();
                    s.toffs = taken;
                    boost += s.boost;
                    subs.push(s);
                }
            }
            fi.subs.retain(|s| !s.toffs.is_empty());
            per_value[vi].push(FragInfo { start: frag_start, end: frag_end, subs, boost });
        }
    }
    let mut out: Vec<FragInfo> = per_value.into_iter().flatten().collect();
    out.sort_by_key(|f| f.start);
    out
}

struct Buffer<'a> {
    content: &'a Content,
    buf: Vec<u16>,
    next: usize,
}

impl<'a> Buffer<'a> {
    fn ensure(&mut self, end: usize) {
        while self.buf.len() < end && self.next < self.content.values.len() {
            let (s, e) = self.content.values[self.next];
            self.buf.extend_from_slice(&self.content.text[s..e]);
            self.buf.push(b' ' as u16);
            self.next += 1;
        }
    }
}

fn find_start(buf: &[u16], start: usize, o: &HighlightOpts) -> usize {
    match o.boundary_scanner.as_deref() {
        Some("sentence") | Some("word") => {
            if start > buf.len() || start < 1 {
                return start;
            }
            let b = if o.boundary_scanner.as_deref() == Some("word") { word_boundaries(&buf[..start]) } else { sentence_boundaries(&buf[..start]) };
            if b.len() >= 2 {
                b[b.len() - 2]
            } else {
                0
            }
        }
        _ => super::breaks::chars_find_start(buf, start, &o.boundary_chars, o.boundary_max_scan),
    }
}

fn find_end(buf: &[u16], start: usize, o: &HighlightOpts) -> usize {
    match o.boundary_scanner.as_deref() {
        Some("sentence") | Some("word") => {
            if start > buf.len() {
                return start;
            }
            let b = if o.boundary_scanner.as_deref() == Some("word") { word_boundaries(buf) } else { sentence_boundaries(buf) };
            super::breaks::following(&b, start)
        }
        _ => super::breaks::chars_find_end(buf, start, &o.boundary_chars, o.boundary_max_scan),
    }
}

fn make_fragment(b: &mut Buffer, fi: &FragInfo, o: &HighlightOpts) -> String {
    b.ensure(fi.end);
    let buf_len = b.buf.len().saturating_sub(1);
    let eo = if buf_len < fi.end { buf_len } else { find_end(&b.buf, fi.end, o) };
    let ms = find_start(&b.buf, fi.start, o).min(eo);
    let src = &b.buf[ms..eo];
    let mut out = String::new();
    let mut src_index = 0usize;
    for sub in &fi.subs {
        for t in &sub.toffs {
            if t.0 < ms || t.0 - ms < src_index || t.1 - ms > src.len() {
                continue;
            }
            let (a, z) = (t.0 - ms, t.1 - ms);
            out.push_str(&encode(&u16s(&src[src_index..a]), o.html_encode));
            out.push_str(o.pre_tag(sub.seq));
            out.push_str(&encode(&u16s(&src[a..z]), o.html_encode));
            out.push_str(o.post_tag(sub.seq));
            src_index = z;
        }
    }
    out.push_str(&encode(&u16s(&src[src_index..]), o.html_encode));
    out
}

pub(crate) fn highlight(content: &Content, ms: &[Match], o: &HighlightOpts) -> Result<Vec<String>> {
    let margin = o.fragment_offset.unwrap_or(6);
    let min_frag = (margin * 3).max(1);
    let nf = o.number_of_fragments;
    if nf != 0 && o.fragment_size < min_frag {
        return Err(EsError::illegal_argument(format!("fragCharSize({}) is too small. It must be {min_frag} or higher.", o.fragment_size)));
    }
    let infos = infos(ms);
    let mut frags = if nf == 0 {
        if infos.is_empty() {
            Vec::new()
        } else {
            let boost = infos.iter().map(|x| x.boost).sum();
            vec![FragInfo { start: 0, end: i32::MAX as usize, subs: infos, boost }]
        }
    } else {
        frag_list(&infos, o.fragment_size, margin)
    };
    if content.values.len() > 1 {
        frags = discrete(frags, content);
    }
    if o.order_score {
        frags.sort_by(|a, b| b.boost.partial_cmp(&a.boost).unwrap_or(std::cmp::Ordering::Equal).then(a.start.cmp(&b.start)));
    }
    let limit = if nf == 0 { usize::MAX } else { nf };
    let mut b = Buffer { content, buf: Vec::new(), next: 0 };
    let mut out: Vec<String> = Vec::new();
    for fi in frags.iter().take(limit) {
        out.push(make_fragment(&mut b, fi, o));
    }
    if out.is_empty() && o.no_match_size > 0 && !content.values.is_empty() {
        let fi = FragInfo { start: 0, end: o.no_match_size, subs: Vec::new(), boost: 0.0 };
        let mut b = Buffer { content, buf: Vec::new(), next: 0 };
        let s = make_fragment(&mut b, &fi, o);
        if !s.is_empty() {
            out.push(s);
        }
    }
    Ok(out)
}
