//! Name analysis over a command pipeline: which column references never
//! resolve (`ResolveUnmapped` collects these for `unmapped_fields`).

use super::parse::{Cmd, Expr, Field};

/// Column references of an expression, in order (`name`, offset).
pub fn expr_refs(e: &Expr, out: &mut Vec<(String, usize)>) {
    match e {
        Expr::Col(n, p) => out.push((n.clone(), *p)),
        Expr::Bin(_, a, b) => {
            expr_refs(a, out);
            expr_refs(b, out);
        }
        Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => expr_refs(a, out),
        Expr::In(a, l, _) => {
            expr_refs(a, out);
            l.iter().for_each(|x| expr_refs(x, out));
        }
        Expr::Between(a, b, c, _) => {
            expr_refs(a, out);
            expr_refs(b, out);
            expr_refs(c, out);
        }
        Expr::Call(_, args, _) => args.iter().for_each(|x| expr_refs(x, out)),
        _ => {}
    }
}

/// Output names of a `DISSECT` pattern (`%{key}`, skipping `%{?key}` / `%{}`).
pub fn dissect_names(pattern: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = pattern;
    while let Some(s) = rest.find("%{") {
        let Some(e) = rest[s..].find('}') else { break };
        let raw = &rest[s + 2..s + e];
        rest = &rest[s + e + 1..];
        if raw.starts_with('?') || raw.starts_with('*') || raw.starts_with('&') || raw.is_empty() {
            continue;
        }
        let k = raw.trim_start_matches('+').split("->").next().unwrap_or("").split('/').next().unwrap_or("").to_string();
        if !k.is_empty() && !out.contains(&k) {
            out.push(k);
        }
    }
    out
}

/// Output names of a `GROK` pattern (`%{PATTERN:name[:type]}`).
pub fn grok_names(pattern: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = pattern;
    while let Some(s) = rest.find("%{") {
        let Some(e) = rest[s..].find('}') else { break };
        let raw = &rest[s + 2..s + e];
        rest = &rest[s + e + 1..];
        let mut parts = raw.splitn(3, ':');
        parts.next();
        if let Some(n) = parts.next() {
            if !n.is_empty() && !out.iter().any(|x| x == n) {
                out.push(n.to_string());
            }
        }
    }
    out
}

/// What the analysis needs from the outside world.
pub trait Env {
    /// Columns a `LOOKUP JOIN` index adds.
    fn lookup_columns(&self, index: &str) -> Vec<String>;
    /// Columns an `ENRICH` policy adds (when `WITH` lists none).
    fn enrich_columns(&self, policy: &str) -> Vec<String>;
}

fn put(scope: &mut Vec<String>, name: &str) {
    scope.retain(|n| n != name);
    scope.push(name.to_string());
}

fn matches(pat: &str, name: &str) -> bool {
    if pat.contains('*') {
        crate::mapping::wildcard_match(pat, name)
    } else {
        pat == name
    }
}

/// References that resolve against nothing in scope, in order of first
/// appearance (`name`, offset). `scope` starts as the source's columns.
pub fn unresolved(cmds: &[Cmd], scope: Vec<String>, env: &dyn Env) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    walk(cmds, scope, env, &mut out);
    out
}

fn note(scope: &[String], refs: Vec<(String, usize)>, out: &mut Vec<(String, usize)>) {
    for (n, p) in refs {
        if !scope.iter().any(|s| *s == n) && !out.iter().any(|(o, _)| *o == n) {
            out.push((n, p));
        }
    }
}

fn walk(cmds: &[Cmd], mut scope: Vec<String>, env: &dyn Env, out: &mut Vec<(String, usize)>) -> Vec<String> {
    let fields_refs = |fs: &[Field]| {
        let mut r = Vec::new();
        for f in fs {
            expr_refs(&f.expr, &mut r);
            if let Some(w) = &f.filter {
                expr_refs(w, &mut r);
            }
        }
        r
    };
    for c in cmds {
        match c {
            Cmd::From { .. } | Cmd::ShowInfo | Cmd::Promql(_) => {}
            Cmd::Row(fs) => {
                for f in fs {
                    put(&mut scope, &f.name);
                }
            }
            Cmd::Where(e, _) => {
                let mut r = Vec::new();
                expr_refs(e, &mut r);
                note(&scope, r, out);
            }
            Cmd::Eval(fs) => {
                for f in fs {
                    let mut r = Vec::new();
                    expr_refs(&f.expr, &mut r);
                    note(&scope, r, out);
                    put(&mut scope, &f.name);
                }
            }
            Cmd::Keep(pats, pos) => {
                let refs = pats.iter().filter(|p| !p.contains('*')).map(|p| (p.clone(), *pos)).collect();
                note(&scope, refs, out);
                let mut next: Vec<String> = Vec::new();
                for p in pats {
                    for n in scope.iter().filter(|n| matches(p, n)) {
                        if !p.contains('*') {
                            next.retain(|x| x != n);
                            next.push(n.clone());
                        } else if !next.contains(n) {
                            next.push(n.clone());
                        }
                    }
                    if !p.contains('*') && !scope.contains(p) {
                        next.push(p.clone());
                    }
                }
                scope = next;
            }
            Cmd::Drop(pats, pos) => {
                let refs = pats.iter().filter(|p| !p.contains('*')).map(|p| (p.clone(), *pos)).collect();
                note(&scope, refs, out);
                scope.retain(|n| !pats.iter().any(|p| matches(p, n)));
            }
            Cmd::Rename(pairs, pos) => {
                let refs = pairs.iter().map(|(o, _)| (o.clone(), *pos)).collect();
                note(&scope, refs, out);
                for (o, n) in pairs {
                    if scope.contains(o) {
                        if o != n {
                            scope.retain(|s| s != n);
                        }
                        if let Some(i) = scope.iter().position(|s| s == o) {
                            scope[i] = n.clone();
                        }
                    } else {
                        put(&mut scope, n);
                    }
                }
            }
            Cmd::Sort(keys) => {
                let mut r = Vec::new();
                keys.iter().for_each(|k| expr_refs(&k.expr, &mut r));
                note(&scope, r, out);
            }
            Cmd::Limit(_) | Cmd::Sample(_) => {}
            Cmd::Info(ts_info, _) => {
                scope = super::exec::INFO_COLUMNS.iter().map(|s| s.to_string()).collect();
                if *ts_info {
                    scope.push("dimensions".into());
                }
            }
            Cmd::LimitBy(_, keys) => {
                let mut r = Vec::new();
                keys.iter().for_each(|k| expr_refs(k, &mut r));
                note(&scope, r, out);
            }
            Cmd::SqlPivot { aggregate, column, values, pos } => {
                let mut refs = vec![(column.clone(), *pos)];
                expr_refs(aggregate, &mut refs);
                note(&scope, refs.clone(), out);
                scope.retain(|n| !refs.iter().any(|(r, _)| r == n));
                for (_, name) in values { put(&mut scope, name); }
            }
            Cmd::Stats { aggs, by, inline, .. } => {
                let mut r = fields_refs(aggs);
                r.extend(fields_refs(by));
                // Grouping names may be referenced by the aggregates (`x = g`).
                let by_names: Vec<String> = by.iter().map(|f| f.name.clone()).collect();
                r.retain(|(n, _)| !by_names.contains(n) || scope.contains(n));
                note(&scope, r, out);
                if *inline {
                    for f in aggs.iter().chain(by.iter()) {
                        put(&mut scope, &f.name);
                    }
                } else {
                    scope = aggs.iter().chain(by.iter()).map(|f| f.name.clone()).collect();
                }
            }
            Cmd::MvExpand(f, pos) => note(&scope, vec![(f.clone(), *pos)], out),
            Cmd::Dissect { field, pattern, .. } => {
                let mut r = Vec::new();
                expr_refs(field, &mut r);
                note(&scope, r, out);
                for n in dissect_names(pattern) {
                    put(&mut scope, &n);
                }
            }
            Cmd::Grok { field, pattern, .. } => {
                let mut r = Vec::new();
                expr_refs(field, &mut r);
                note(&scope, r, out);
                for n in grok_names(pattern) {
                    put(&mut scope, &n);
                }
            }
            Cmd::LookupJoin { index, on, on_pos, .. } => {
                note(&scope, on.iter().zip(on_pos).map(|(n, p)| (n.clone(), *p)).collect(), out);
                for n in env.lookup_columns(index) {
                    if !on.contains(&n) {
                        put(&mut scope, &n);
                    }
                }
            }
            Cmd::Enrich { policy, on, with, pos, .. } => {
                if let Some(o) = on {
                    note(&scope, vec![(o.clone(), *pos)], out);
                }
                let names: Vec<String> = if with.is_empty() { env.enrich_columns(policy) } else { with.iter().map(|(n, _)| n.clone()).collect() };
                for n in names {
                    put(&mut scope, &n);
                }
            }
            Cmd::Fork(branches) => {
                let mut union: Vec<String> = Vec::new();
                for b in branches {
                    for n in walk(b, scope.clone(), env, out) {
                        if !union.contains(&n) {
                            union.push(n);
                        }
                    }
                }
                union.retain(|n| n != "_fork");
                union.push("_fork".into());
                scope = union;
            }
        }
    }
    scope
}
