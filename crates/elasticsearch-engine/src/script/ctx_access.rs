//! Which parts of `ctx` a script can observe, for ingest scripts that run
//! on a document's source ([`super::source::partial_ctx`]).
//!
//! A `ctx` reached only through chains of constant-key accesses
//! (`ctx.a.b`, `ctx['a']['b']`, assignments to them) can be given to the
//! script with just those keys: a map the script only indexes by constant
//! keys behaves the same whatever other keys it has. Where a chain ends
//! (its value is used: passed, iterated, called, assigned elsewhere) the
//! node is observed as a whole and is converted in full. `metadata()`
//! reads ctx itself: such scripts get all of it.
use super::ast::{Expr, ExprKind, LambdaBody, ScriptAst, Stmt, StmtKind};
use std::collections::BTreeMap;

/// The `ctx` keys a script reaches by constant-key chains: `full` nodes
/// are observed whole, the others only through their `children`.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct CtxPaths {
    pub full: bool,
    pub children: BTreeMap<String, CtxPaths>,
}

impl CtxPaths {
    fn record(&mut self, path: &[String]) {
        let mut node = self;
        for key in path {
            if node.full {
                return;
            }
            node = node.children.entry(key.clone()).or_default();
        }
        node.full = true;
        node.children.clear();
    }
}

/// `None`: the script may observe `ctx` as a whole (or declares a `ctx` of
/// its own): convert all of it.
pub fn analyze(ast: &ScriptAst) -> Option<CtxPaths> {
    let mut walk = Walk { paths: CtxPaths::default(), shadowed: false };
    for f in &ast.functions {
        if f.params.iter().any(|(_, n)| n == "ctx") {
            return None;
        }
        walk.stmts(&f.body);
    }
    walk.stmts(&ast.body);
    (!walk.shadowed && !walk.paths.full).then_some(walk.paths)
}

struct Walk {
    paths: CtxPaths,
    shadowed: bool,
}

/// The constant keys of a chain rooted at `ctx` (`Some(vec![])`: `ctx`).
fn chain(e: &Expr) -> Option<Vec<String>> {
    match &e.kind {
        ExprKind::Var(n) if n == "ctx" => Some(Vec::new()),
        ExprKind::Field { obj, name, null_safe: false } => chain(obj).map(|mut p| { p.push(name.clone()); p }),
        ExprKind::Index { obj, idx } => match &idx.kind {
            ExprKind::Str(key) => chain(obj).map(|mut p| { p.push(key.clone()); p }),
            _ => None,
        },
        _ => None,
    }
}

impl Walk {
    fn declare(&mut self, name: &str) {
        if name == "ctx" {
            self.shadowed = true;
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Expr(e) | StmtKind::Throw(e) => self.expr(e),
            StmtKind::Decl { vars, .. } => {
                for (name, init, _) in vars {
                    self.declare(name);
                    if let Some(e) = init { self.expr(e); }
                }
            }
            StmtKind::If { cond, then, els } => {
                self.expr(cond);
                self.stmt(then);
                if let Some(e) = els { self.stmt(e); }
            }
            StmtKind::While { cond, body } | StmtKind::DoWhile { body, cond } => {
                self.expr(cond);
                self.stmt(body);
            }
            StmtKind::For { init, cond, update, body } => {
                if let Some(i) = init { self.stmt(i); }
                if let Some(c) = cond { self.expr(c); }
                for u in update { self.expr(u); }
                self.stmt(body);
            }
            StmtKind::ForEach { var, iter, body, .. } => {
                self.declare(var);
                self.expr(iter);
                self.stmt(body);
            }
            StmtKind::Block(b) => self.stmts(b),
            StmtKind::Return(e) => { if let Some(e) = e { self.expr(e); } }
            StmtKind::Try { body, catches } => {
                self.stmts(body);
                for (_, name, b) in catches {
                    self.declare(name);
                    self.stmts(b);
                }
            }
            StmtKind::Break | StmtKind::Continue | StmtKind::Empty => {}
        }
    }

    fn exprs(&mut self, es: &[Expr]) {
        for e in es {
            self.expr(e);
        }
    }

    fn expr(&mut self, e: &Expr) {
        if let Some(path) = chain(e) {
            self.paths.record(&path);
            return;
        }
        match &e.kind {
            ExprKind::Null | ExprKind::Bool(_) | ExprKind::Int(_) | ExprKind::Long(_) | ExprKind::Double(_) | ExprKind::Str(_)
            | ExprKind::Utf16(_) | ExprKind::Regex { .. } | ExprKind::Var(_) | ExprKind::MethodRef { .. } => {}
            ExprKind::RegexOp { text, pat, .. } => { self.expr(text); self.expr(pat); }
            ExprKind::ListLit(items) | ExprKind::NewArrayInit { items, .. } => self.exprs(items),
            ExprKind::MapLit(entries) => for (k, v) in entries { self.expr(k); self.expr(v); },
            // `metadata()` reads ctx's metadata keys itself.
            ExprKind::FnCall { name, args } if name == "metadata" => { self.paths.full = true; self.exprs(args) }
            ExprKind::New { args, .. } | ExprKind::FnCall { args, .. } => self.exprs(args),
            ExprKind::NewArray { dims, .. } => self.exprs(dims),
            ExprKind::Unary(_, a) | ExprKind::Cast { expr: a, .. } | ExprKind::InstanceOf { expr: a, .. } => self.expr(a),
            ExprKind::Binary(_, a, b) | ExprKind::And(a, b) | ExprKind::Or(a, b) | ExprKind::Elvis(a, b) => { self.expr(a); self.expr(b); }
            ExprKind::Ternary(a, b, c) => { self.expr(a); self.expr(b); self.expr(c); }
            ExprKind::Assign { target, value, .. } => { self.expr(target); self.expr(value); }
            ExprKind::IncDec { target, .. } => self.expr(target),
            ExprKind::Field { obj, .. } => self.expr(obj),
            ExprKind::Index { obj, idx } => { self.expr(obj); self.expr(idx); }
            ExprKind::Call { obj, args, .. } => { self.expr(obj); self.exprs(args); }
            ExprKind::Lambda { params, body } => {
                for p in params { self.declare(p); }
                match body.as_ref() {
                    LambdaBody::Expr(e) => self.expr(e),
                    LambdaBody::Block(b) => self.stmts(b),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(src: &str) -> Option<CtxPaths> {
        analyze(&super::super::parser::parse_with_literals(src, &Default::default()).unwrap().0)
    }

    fn keys(p: &CtxPaths) -> Vec<String> {
        fn go(p: &CtxPaths, prefix: &str, out: &mut Vec<String>) {
            for (k, c) in &p.children {
                let name = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                if c.full { out.push(name.clone()); }
                go(c, &name, out);
            }
        }
        let mut out = Vec::new();
        go(p, "", &mut out);
        out
    }

    #[test]
    fn constant_key_chains_are_the_observed_parts() {
        let p = paths("Random r = new Random(); ctx['kubernetes']['pod']['uid'] = 'value-' + r.nextLong();").unwrap();
        assert_eq!(keys(&p), ["kubernetes.pod.uid"]);
        let p = paths("ctx.a.b = ctx.c.size(); if (ctx.a?.x != null) { ctx.d += 1 }").unwrap();
        assert_eq!(keys(&p), ["a", "c", "d"], "a null-safe step observes its receiver");
        let p = paths("ctx.a.b.c = 1; ctx.a.b.d = ctx.a.e").unwrap();
        assert_eq!(keys(&p), ["a.b.c", "a.b.d", "a.e"]);
        let p = paths("ctx.a.b = 1; def x = ctx.a; x.c = 2").unwrap();
        assert_eq!(keys(&p), ["a"], "a chain's end used as a value is observed whole");
        let p = paths("for (def k : ctx.keySet()) {}");
        assert_eq!(p, None, "a method of ctx observes all of it");
        assert_eq!(paths("ctx[params.k] = 1"), None, "a dynamic key observes ctx");
        assert_eq!(paths("Debug.explain(ctx)"), None);
        assert_eq!(paths("ctx.a = metadata().index"), None, "metadata() reads ctx");
        assert_eq!(paths("def ctx = [:]; ctx.a = 1"), None, "a ctx of its own");
        assert_eq!(paths("[1].forEach(ctx -> ctx.a)"), None);
        let p = paths("[1, 2].forEach(x -> ctx.list.add(x))").unwrap();
        assert_eq!(keys(&p), ["list"], "lambdas count");
    }
}
