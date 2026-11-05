//! Determinization cost of `LIKE` / `RLIKE` patterns.
//!
//! ES|QL compiles both to Lucene automata and determinizes them with
//! Lucene's default work limit; patterns whose powerset construction costs
//! more fail with `Pattern was too complex to determinize`. This builds the
//! same kind of NFA (wildcards: `*`, `?`, literals; regexes: Lucene
//! `RegExp` core syntax) and runs the powerset construction with Lucene's
//! effort accounting (sum of NFA-state-set sizes, limit 10 x 10 000).

use std::collections::HashMap;

/// `Operations.DEFAULT_DETERMINIZE_WORK_LIMIT` x 10 (Lucene's effort budget).
const EFFORT_LIMIT: usize = 100_000;

#[derive(Clone, Debug)]
enum Pred {
    Any,
    /// Inclusive char ranges, negated when `.1`.
    Set(Vec<(char, char)>, bool),
}

impl Pred {
    fn matches(&self, c: char) -> bool {
        match self {
            Pred::Any => true,
            Pred::Set(r, neg) => r.iter().any(|(a, b)| *a <= c && c <= *b) != *neg,
        }
    }
}

#[derive(Default)]
struct Nfa {
    trans: Vec<Vec<(Pred, usize)>>,
    eps: Vec<Vec<usize>>,
}

impl Nfa {
    fn state(&mut self) -> usize {
        self.trans.push(Vec::new());
        self.eps.push(Vec::new());
        self.trans.len() - 1
    }
    fn edge(&mut self, a: usize, p: Pred, b: usize) {
        self.trans[a].push((p, b));
    }
    fn eps(&mut self, a: usize, b: usize) {
        self.eps[a].push(b);
    }
}

/// A Thompson fragment: (start, end).
type Frag = (usize, usize);

fn lit(n: &mut Nfa, p: Pred) -> Frag {
    let (a, b) = (n.state(), n.state());
    n.edge(a, p, b);
    (a, b)
}

fn empty(n: &mut Nfa) -> Frag {
    let a = n.state();
    (a, a)
}

fn concat(n: &mut Nfa, x: Frag, y: Frag) -> Frag {
    n.eps(x.1, y.0);
    (x.0, y.1)
}

fn star(n: &mut Nfa, x: Frag) -> Frag {
    let (a, b) = (n.state(), n.state());
    n.eps(a, x.0);
    n.eps(a, b);
    n.eps(x.1, x.0);
    n.eps(x.1, b);
    (a, b)
}

fn optional(n: &mut Nfa, x: Frag) -> Frag {
    let (a, b) = (n.state(), n.state());
    n.eps(a, x.0);
    n.eps(a, b);
    n.eps(x.1, b);
    (a, b)
}

fn alt(n: &mut Nfa, x: Frag, y: Frag) -> Frag {
    let (a, b) = (n.state(), n.state());
    n.eps(a, x.0);
    n.eps(a, y.0);
    n.eps(x.1, b);
    n.eps(y.1, b);
    (a, b)
}

/// Wildcard (`*`, `?`, `\` escapes) NFA.
fn wildcard(n: &mut Nfa, pat: &str) -> Frag {
    let mut f = empty(n);
    let mut it = pat.chars();
    while let Some(c) = it.next() {
        let g = match c {
            '*' => {
                let a = n.state();
                n.edge(a, Pred::Any, a);
                (a, a)
            }
            '?' => lit(n, Pred::Any),
            '\\' => {
                let e = it.next().unwrap_or('\\');
                lit(n, Pred::Set(vec![(e, e)], false))
            }
            c => lit(n, Pred::Set(vec![(c, c)], false)),
        };
        f = concat(n, f, g);
    }
    f
}

/// Regex NFA for Lucene `RegExp` core syntax; `None` when it doesn't parse.
struct Re<'a> {
    s: Vec<char>,
    i: usize,
    n: &'a mut Nfa,
}

impl Re<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }
    fn union(&mut self) -> Option<Frag> {
        let mut f = self.concat()?;
        while self.peek() == Some('|') {
            self.i += 1;
            let g = self.concat()?;
            f = alt(self.n, f, g);
        }
        Some(f)
    }
    fn concat(&mut self) -> Option<Frag> {
        let mut f = empty(self.n);
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let g = self.repeat()?;
            f = concat(self.n, f, g);
        }
        Some(f)
    }
    fn number(&mut self) -> Option<usize> {
        let st = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        self.s[st..self.i].iter().collect::<String>().parse().ok()
    }
    fn repeat(&mut self) -> Option<Frag> {
        let start = self.i;
        let mut f = self.atom()?;
        let end = self.i;
        loop {
            match self.peek() {
                Some('*') => {
                    self.i += 1;
                    f = star(self.n, f);
                }
                Some('+') => {
                    self.i += 1;
                    let again = self.rebuild(start, end)?;
                    let s = star(self.n, again);
                    f = concat(self.n, f, s);
                }
                Some('?') => {
                    self.i += 1;
                    f = optional(self.n, f);
                }
                Some('{') => {
                    self.i += 1;
                    let lo = self.number()?;
                    let hi = if self.peek() == Some(',') {
                        self.i += 1;
                        if self.peek() == Some('}') { None } else { Some(self.number()?) }
                    } else {
                        Some(lo)
                    };
                    if self.peek() != Some('}') {
                        return None;
                    }
                    self.i += 1;
                    if lo > 256 || hi.is_some_and(|h| h > 256 || h < lo) {
                        return None;
                    }
                    let after = self.i;
                    let mut g = empty(self.n);
                    for _ in 0..lo {
                        let x = self.rebuild(start, end)?;
                        g = concat(self.n, g, x);
                    }
                    match hi {
                        None => {
                            let x = self.rebuild(start, end)?;
                            let s = star(self.n, x);
                            g = concat(self.n, g, s);
                        }
                        Some(h) => {
                            for _ in lo..h {
                                let x = self.rebuild(start, end)?;
                                let o = optional(self.n, x);
                                g = concat(self.n, g, o);
                            }
                        }
                    }
                    self.i = after;
                    f = g;
                }
                _ => break,
            }
        }
        Some(f)
    }
    /// A fresh copy of the atom at `start..end`.
    fn rebuild(&mut self, start: usize, end: usize) -> Option<Frag> {
        let save = self.i;
        self.i = start;
        let f = self.atom()?;
        if self.i != end {
            return None;
        }
        self.i = save;
        Some(f)
    }
    fn class(&mut self) -> Option<Pred> {
        let mut neg = false;
        if self.peek() == Some('^') {
            neg = true;
            self.i += 1;
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let c = self.peek()?;
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;
            let c = if c == '\\' {
                let e = self.peek()?;
                self.i += 1;
                e
            } else {
                c
            };
            if self.peek() == Some('-') && self.s.get(self.i + 1).is_some_and(|x| *x != ']') {
                self.i += 1;
                let mut h = self.peek()?;
                self.i += 1;
                if h == '\\' {
                    h = self.peek()?;
                    self.i += 1;
                }
                ranges.push((c, h));
            } else {
                ranges.push((c, c));
            }
        }
        Some(Pred::Set(ranges, neg))
    }
    fn atom(&mut self) -> Option<Frag> {
        let c = self.peek()?;
        self.i += 1;
        Some(match c {
            '(' => {
                let f = if self.peek() == Some(')') { empty(self.n) } else { self.union()? };
                if self.peek() != Some(')') {
                    return None;
                }
                self.i += 1;
                f
            }
            '[' => {
                let p = self.class()?;
                lit(self.n, p)
            }
            '.' => lit(self.n, Pred::Any),
            '\\' => {
                let e = self.peek()?;
                self.i += 1;
                lit(self.n, Pred::Set(vec![(e, e)], false))
            }
            c => lit(self.n, Pred::Set(vec![(c, c)], false)),
        })
    }
}

fn closure(n: &Nfa, set: &mut Vec<usize>) {
    let mut stack = set.clone();
    let mut seen: std::collections::HashSet<usize> = set.iter().copied().collect();
    while let Some(s) = stack.pop() {
        for &t in &n.eps[s] {
            if seen.insert(t) {
                set.push(t);
                stack.push(t);
            }
        }
    }
    set.sort_unstable();
}

/// Does determinizing the NFA exceed Lucene's work limit?
fn exceeds(n: &Nfa, start: usize, accept: usize) -> bool {
    // Representative characters: one per interval between predicate boundaries.
    let mut bounds: Vec<u32> = vec![0];
    for ts in &n.trans {
        for (p, _) in ts {
            if let Pred::Set(r, _) = p {
                for (a, b) in r {
                    bounds.push(*a as u32);
                    bounds.push(*b as u32 + 1);
                }
            }
        }
    }
    bounds.sort_unstable();
    bounds.dedup();
    let reps: Vec<char> = bounds.iter().filter_map(|b| char::from_u32(*b)).collect();
    // Lucene's automata have no epsilon states: count only states with
    // character transitions (and the accept state).
    let weight = |set: &[usize]| set.iter().filter(|s| !n.trans[**s].is_empty() || **s == accept).count();
    let mut init = vec![start];
    closure(n, &mut init);
    let mut ids: HashMap<Vec<usize>, usize> = HashMap::new();
    ids.insert(init.clone(), 0);
    let mut work = vec![init];
    let mut effort = 0usize;
    while let Some(set) = work.pop() {
        effort += weight(&set);
        if effort >= EFFORT_LIMIT {
            return true;
        }
        for &c in &reps {
            let mut next: Vec<usize> = Vec::new();
            for &s in &set {
                for (p, t) in &n.trans[s] {
                    if p.matches(c) && !next.contains(t) {
                        next.push(*t);
                    }
                }
            }
            if next.is_empty() {
                continue;
            }
            closure(n, &mut next);
            if !ids.contains_key(&next) {
                ids.insert(next.clone(), ids.len());
                work.push(next);
            }
        }
    }
    false
}

/// `true` when a `LIKE` (`regex == false`) or `RLIKE` pattern is too complex to determinize.
pub fn too_complex(pattern: &str, regex: bool) -> bool {
    let mut n = Nfa::default();
    let f = if regex {
        let mut r = Re { s: pattern.chars().collect(), i: 0, n: &mut n };
        match r.union() {
            Some(f) if r.i == r.s.len() => f,
            _ => return false,
        }
    } else {
        wildcard(&mut n, pattern)
    };
    exceeds(&n, f.0, f.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complexity() {
        assert!(too_complex("*a?????????????", false));
        assert!(too_complex("*a????????????????", false));
        assert!(too_complex("(a|b)*a(a|b){13}", true));
        assert!(!too_complex("*foo*bar?", false));
        assert!(!too_complex("[a-z]+@[a-z]+\\.com", true));
        assert!(!too_complex("*a????", false));
    }
}
