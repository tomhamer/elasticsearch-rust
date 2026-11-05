//! AST → closure tree. Variables are resolved to frame slots, `params.x`
//! and `doc['x']` with constant keys to binding tables, and common
//! `doc['x'].value` / `Math.*` shapes to non-allocating fast nodes.

use super::ast::*;
use super::error::{illegal_arg, npe, ScriptErr, SResult};
use super::parser::{parse, type_of_name};
use super::runtime::*;
use super::value::{Key, Val};
use super::ScriptContext;
use indexmap::IndexMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

mod num;
pub(crate) mod score_fns;
pub(crate) use num::Num;

type CResult<T> = Result<T, ScriptErr>;

/// Light static type of an expression (only what literals / declarations
/// tell us; everything else is `Def`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Def,
    Int,
    Long,
    Double,
    Bool,
    Str,
    Null,
    /// A list literal (`java.util.ArrayList`).
    List,
    /// A map literal (`java.util.HashMap`).
    Map,
    /// A statically typed reference of an unspecified class (`new X()`,
    /// typed locals, results of typed calls): calls on it are not `def`
    /// dispatched.
    Ref,
    /// A statically typed `java.math.BigDecimal` / `BigInteger` (fqn).
    Cls(&'static str),
}

impl St {
    fn of_ty(t: &Ty) -> St {
        match t {
            Ty::Int | Ty::Short | Ty::Byte | Ty::Char => St::Int,
            Ty::Long => St::Long,
            Ty::Double | Ty::Float => St::Double,
            Ty::Boolean => St::Bool,
            Ty::String => St::Str,
            Ty::Ref(n) => match n.as_str() {
                "BigDecimal" => St::Cls("java.math.BigDecimal"),
                "BigInteger" => St::Cls("java.math.BigInteger"),
                _ => St::Ref,
            },
            Ty::Array(_) => St::Ref,
            _ => St::Def,
        }
    }
    /// No static checks apply (`def`, or a reference we don't model).
    fn is_dyn(&self) -> bool {
        matches!(self, St::Def | St::Ref)
    }
    /// A receiver whose method calls are statically bound (not `def`).
    fn is_typed(&self) -> bool {
        matches!(self, St::Str | St::List | St::Map | St::Ref | St::Cls(_))
    }
    fn name(&self) -> &'static str {
        match self {
            St::Def => "def",
            St::Int => "int",
            St::Long => "long",
            St::Double => "double",
            St::Bool => "boolean",
            St::Str => "String",
            St::Null => "null",
            St::List => "java.util.ArrayList",
            St::Map => "java.util.HashMap",
            St::Ref => "def",
            St::Cls(c) => *c,
        }
    }
    fn is_num(&self) -> bool {
        matches!(self, St::Int | St::Long | St::Double)
    }
}

struct Compiler {
    ctx: ScriptContext,
    in_function: bool,
    scopes: Vec<Vec<(String, usize, Ty)>>,
    next_slot: usize,
    loop_depth: usize,
    fn_sigs: Vec<(String, Vec<Ty>, Ty)>,
    lambdas: Vec<Option<LambdaInfo>>,
    strings: Vec<String>,
    param_names: Vec<String>,
    doc_fields: Vec<String>,
    /// Per compiled node (by address): the source offset of the last
    /// bytecode debug-info write ES emits while evaluating it (where a
    /// runtime error in an enclosing statically bound call is reported),
    /// and its static type.
    dbg: std::collections::HashMap<usize, (Option<usize>, St)>,
    /// Inside an expression that already has a numeric fast path (its
    /// sub-expressions need none of their own).
    num_off: bool,
    /// The script reads `_score`.
    uses_score: bool,
    /// Class-binding state slots (`Rt::bindings`) allocated so far.
    nbindings: usize,
    /// Binding slot per call site (keyed by its argument list's address).
    binding_slots: std::collections::HashMap<usize, usize>,
}

pub fn compile_program(src: &str, ctx: ScriptContext) -> Result<Program, ScriptErr> {
    compile_program_literals(src, ctx, &std::collections::HashMap::new())
}

pub fn compile_program_literals(src: &str, ctx: ScriptContext, literals: &std::collections::HashMap<usize, Vec<u16>>) -> Result<Program, ScriptErr> {
    let (ast, stmt_starts) = super::parser::parse_with_literals(src, literals)?;
    let mut c = Compiler {
        ctx,
        in_function: false,
        scopes: vec![Vec::new()],
        next_slot: 0,
        loop_depth: 0,
        fn_sigs: ast.functions.iter().map(|f| (f.name.clone(), f.params.iter().map(|p| p.0.clone()).collect(), f.ret.clone())).collect(),
        lambdas: Vec::new(),
        strings: Vec::new(),
        param_names: Vec::new(),
        doc_fields: Vec::new(),
        dbg: std::collections::HashMap::new(),
        num_off: false,
        uses_score: false,
        nbindings: 0,
        binding_slots: std::collections::HashMap::new(),
    };
    for (i, f) in ast.functions.iter().enumerate() {
        if c.fn_sigs[..i].iter().any(|s| s.0 == f.name && s.1.len() == f.params.len()) {
            return Err(ScriptErr::compile(f.pos, format!("Illegal duplicate functions [{}/{}].", f.name, f.params.len())));
        }
    }
    let mut functions = Vec::new();
    for f in &ast.functions {
        c.in_function = true;
        let saved = (std::mem::replace(&mut c.scopes, vec![Vec::new()]), c.next_slot);
        c.next_slot = 0;
        for (t, n) in &f.params {
            c.declare(n, t.clone(), f.pos)?;
        }
        let body = c.block(&f.body, false)?;
        functions.push(FnInfo { name: f.name.clone(), params: f.params.iter().map(|p| p.0.clone()).collect(), ret: f.ret.clone(), nslots: c.next_slot, body });
        c.scopes = saved.0;
        c.next_slot = saved.1;
    }
    c.in_function = false;
    let (mut num_body, mut cond_body) = (None, None);
    let expr_body = match (ast.functions.is_empty(), ast.body.as_slice()) {
        (true, [Stmt { kind: StmtKind::Expr(e), .. }]) | (true, [Stmt { kind: StmtKind::Return(Some(e)), .. }]) => {
            let f = c.expr(e)?.0;
            num_body = c.num(e).map(num::NumFn::into_d);
            cond_body = c.num_cond(e);
            Some(f)
        }
        _ => None,
    };
    let result_pos = match ast.body.last() {
        Some(Stmt { kind: StmtKind::Expr(e), .. }) | Some(Stmt { kind: StmtKind::Return(Some(e)), .. }) => e.pos,
        Some(s) => s.pos,
        None => 0,
    };
    let body = c.block(&ast.body, true)?;
    let ctx_paths = super::ctx_access::analyze(&ast).map(std::sync::Arc::new);
    let lambdas = c.lambdas.into_iter().map(|l| l.expect("lambda compiled")).collect();
    Ok(Program {
        source: src.to_string(),
        stmt_starts,
        ctx,
        body,
        expr_body,
        num_body,
        cond_body,
        uses_score: c.uses_score,
        result_pos,
        nslots: c.next_slot.max(1),
        functions,
        lambdas,
        strings: c.strings,
        param_names: c.param_names,
        doc_fields: c.doc_fields,
        nbindings: c.nbindings,
        expression: None,
        ctx_paths,
    })
}

fn at(pos: usize) -> impl Fn(Box<ScriptErr>) -> Box<ScriptErr> {
    move |e| e.at(pos)
}

/// `IllegalArgumentException` → `illegal_argument_exception`.
pub fn snake_case(class: &str) -> String {
    let mut s = String::new();
    for (i, c) in class.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                s.push('_');
            }
            s.extend(c.to_lowercase());
        } else {
            s.push(c);
        }
    }
    s
}

const EXCEPTIONS: &[&str] = &[
    "Exception", "RuntimeException", "IllegalArgumentException", "IllegalStateException", "UnsupportedOperationException", "NullPointerException",
    "ArithmeticException", "ClassCastException", "NumberFormatException", "IndexOutOfBoundsException", "ArrayIndexOutOfBoundsException",
    "StringIndexOutOfBoundsException", "NoSuchElementException", "DateTimeParseException", "ConcurrentModificationException",
];

impl Compiler {
    fn lookup(&self, name: &str) -> Option<(usize, Ty)> {
        for s in self.scopes.iter().rev() {
            if let Some((_, slot, t)) = s.iter().rev().find(|(n, _, _)| n == name) {
                return Some((*slot, t.clone()));
            }
        }
        None
    }

    fn declare(&mut self, name: &str, ty: Ty, pos: usize) -> CResult<usize> {
        if self.scopes.iter().any(|s| s.iter().any(|(n, _, _)| n == name)) {
            return Err(ScriptErr::compile(pos, format!("variable [{name}] is already defined")));
        }
        if !self.in_function && matches!(name, "params" | "doc" | "_score" | "ctx") {
            return Err(ScriptErr::compile(pos, format!("variable [{name}] is already defined")));
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        self.scopes.last_mut().unwrap().push((name.to_string(), slot, ty));
        Ok(slot)
    }

    fn string_id(&mut self, s: &str) -> usize {
        match self.strings.iter().position(|x| x == s) {
            Some(i) => i,
            None => {
                self.strings.push(s.to_string());
                self.strings.len() - 1
            }
        }
    }

    fn param_id(&mut self, s: &str) -> usize {
        match self.param_names.iter().position(|x| x == s) {
            Some(i) => i,
            None => {
                self.param_names.push(s.to_string());
                self.param_names.len() - 1
            }
        }
    }

    fn doc_id(&mut self, s: &str) -> usize {
        match self.doc_fields.iter().position(|x| x == s) {
            Some(i) => i,
            None => {
                self.doc_fields.push(s.to_string());
                self.doc_fields.len() - 1
            }
        }
    }

    fn is_global(&self, e: &Expr, name: &str) -> bool {
        !self.in_function && matches!(&e.kind, ExprKind::Var(n) if n == name) && self.lookup(name).is_none()
    }

    fn static_class<'e>(&self, e: &'e Expr) -> Option<&'e str> {
        match &e.kind {
            ExprKind::Var(n)
                if self.lookup(n).is_none()
                    && STATIC_CLASSES.contains(&n.as_str())
                    && (n != "Json" || matches!(self.ctx, ScriptContext::Ingest | ScriptContext::Reindex | ScriptContext::Update | ScriptContext::Watcher | ScriptContext::WatcherCondition | ScriptContext::WatcherTemplate)) =>
            {
                Some(n.as_str())
            }
            _ => None,
        }
    }

    /// `doc['x']` / `doc.x` with a constant name → its binding slot.
    fn doc_slot(&mut self, e: &Expr) -> Option<usize> {
        match &e.kind {
            ExprKind::Index { obj, idx } if self.is_global(obj, "doc") => match &idx.kind {
                ExprKind::Str(s) => Some(self.doc_id(s)),
                _ => None,
            },
            ExprKind::Field { obj, name, null_safe: false } if self.is_global(obj, "doc") => Some(self.doc_id(name)),
            _ => None,
        }
    }

    /// `params.x` / `params['x']` with a constant key.
    fn param_key(&self, e: &Expr) -> Option<String> {
        match &e.kind {
            ExprKind::Index { obj, idx } if self.is_global(obj, "params") => match &idx.kind {
                ExprKind::Str(s) => Some(s.clone()),
                _ => None,
            },
            ExprKind::Field { obj, name, .. } if self.is_global(obj, "params") => Some(name.clone()),
            _ => None,
        }
    }

    // -----------------------------------------------------------------
    // Statements.

    fn block(&mut self, stmts: &[Stmt], implicit_return: bool) -> CResult<StmtFn> {
        self.scopes.push(Vec::new());
        let mut out: Vec<StmtFn> = Vec::with_capacity(stmts.len());
        for (i, s) in stmts.iter().enumerate() {
            let last = i + 1 == stmts.len();
            out.push(self.stmt(s, implicit_return && last)?);
        }
        self.scopes.pop();
        Ok(match out.len() {
            0 => Box::new(|_| Ok(Flow::Next)),
            1 => out.pop().unwrap(),
            _ => Box::new(move |rt| {
                for s in &out {
                    match s(rt)? {
                        Flow::Next => {}
                        f => return Ok(f),
                    }
                }
                Ok(Flow::Next)
            }),
        })
    }

    fn cond(&mut self, e: &Expr) -> CResult<Box<dyn Fn(&mut Rt) -> SResult<bool> + Send + Sync>> {
        let (f, st) = self.expr(e)?;
        if !st.is_dyn() && st != St::Bool {
            return Err(cc(e.pos, format!("Cannot cast from [{}] to [boolean].", jname_st(st))));
        }
        let pos = e.pos;
        // Numeric comparisons straight to `bool` (no `Val::Bool` round
        // trip); `None` falls back to the general node, whose own fast path
        // is side-effect free, so the re-evaluation is invisible.
        if !self.num_off && self.num_root(e) {
            if let Some(c) = self.num_cond(e) {
                return Ok(Box::new(move |rt| match c(rt) {
                    Some(b) => Ok(b),
                    None => f(rt)?.as_bool().map_err(at(pos)),
                }));
            }
        }
        // A `boolean` local read in place.
        if let ExprKind::Var(n) = &e.kind {
            if let Some((slot, Ty::Boolean)) = self.lookup(n) {
                return Ok(Box::new(move |rt| {
                    let cur = match rt.local(slot) {
                        Val::Bool(b) => Some(*b),
                        _ => None,
                    };
                    match cur {
                        Some(b) => Ok(b),
                        None => f(rt)?.as_bool().map_err(at(pos)),
                    }
                }));
            }
        }
        Ok(Box::new(move |rt| f(rt)?.as_bool().map_err(at(pos))))
    }

    fn stmt(&mut self, s: &Stmt, implicit_return: bool) -> CResult<StmtFn> {
        Ok(match &s.kind {
            StmtKind::Empty => Box::new(|_| Ok(Flow::Next)),
            StmtKind::Expr(e) => {
                let (f, _) = self.expr(e)?;
                if implicit_return {
                    Box::new(move |rt| Ok(Flow::Return(f(rt)?)))
                } else {
                    Box::new(move |rt| {
                        f(rt)?;
                        Ok(Flow::Next)
                    })
                }
            }
            StmtKind::Block(b) => self.block(b, implicit_return)?,
            StmtKind::Decl { ty, vars } => {
                let mut parts: Vec<(usize, Ty, Option<ExprFn>, usize)> = Vec::new();
                for (name, init, pos) in vars {
                    let init_f = match init {
                        Some(e) => {
                            let (f, st) = self.expr(e)?;
                            check_assign(ty, st, e.pos)?;
                            Some(f)
                        }
                        None => None,
                    };
                    let slot = self.declare(name, ty.clone(), *pos)?;
                    parts.push((slot, ty.clone(), init_f, *pos));
                }
                Box::new(move |rt| {
                    for (slot, ty, init, pos) in &parts {
                        let v = match init {
                            Some(f) => coerce(ty, f(rt)?).map_err(at(*pos))?,
                            None => default_of(ty),
                        };
                        rt.set_local(*slot, v);
                    }
                    Ok(Flow::Next)
                })
            }
            StmtKind::If { cond, then, els } => {
                let c = self.cond(cond)?;
                self.scopes.push(Vec::new());
                let t = self.stmt(then, implicit_return)?;
                self.scopes.pop();
                self.scopes.push(Vec::new());
                let e = match els {
                    Some(x) => Some(self.stmt(x, implicit_return)?),
                    None => None,
                };
                self.scopes.pop();
                Box::new(move |rt| {
                    if c(rt)? {
                        t(rt)
                    } else if let Some(e) = &e {
                        e(rt)
                    } else {
                        Ok(Flow::Next)
                    }
                })
            }
            StmtKind::While { cond, body } => {
                let c = self.cond(cond)?;
                self.loop_depth += 1;
                self.scopes.push(Vec::new());
                let b = self.stmt(body, false)?;
                self.scopes.pop();
                self.loop_depth -= 1;
                let pos = s.pos;
                Box::new(move |rt| {
                    while c(rt)? {
                        rt.tick().map_err(at(pos))?;
                        match b(rt)? {
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
                            _ => {}
                        }
                    }
                    Ok(Flow::Next)
                })
            }
            StmtKind::DoWhile { body, cond } => {
                self.loop_depth += 1;
                self.scopes.push(Vec::new());
                let b = self.stmt(body, false)?;
                self.scopes.pop();
                self.loop_depth -= 1;
                let c = self.cond(cond)?;
                let pos = s.pos;
                Box::new(move |rt| {
                    loop {
                        rt.tick().map_err(at(pos))?;
                        match b(rt)? {
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
                            _ => {}
                        }
                        if !c(rt)? {
                            break;
                        }
                    }
                    Ok(Flow::Next)
                })
            }
            StmtKind::For { init, cond, update, body } => {
                self.scopes.push(Vec::new());
                let i = match init {
                    Some(x) => Some(self.stmt(x, false)?),
                    None => None,
                };
                let c = match cond {
                    Some(x) => Some(self.cond(x)?),
                    None => None,
                };
                let u: Vec<ExprFn> = update.iter().map(|e| self.expr(e).map(|x| x.0)).collect::<CResult<_>>()?;
                self.loop_depth += 1;
                self.scopes.push(Vec::new());
                let b = self.stmt(body, false)?;
                self.scopes.pop();
                self.loop_depth -= 1;
                self.scopes.pop();
                let pos = s.pos;
                Box::new(move |rt| {
                    if let Some(i) = &i {
                        i(rt)?;
                    }
                    loop {
                        if let Some(c) = &c {
                            if !c(rt)? {
                                break;
                            }
                        }
                        rt.tick().map_err(at(pos))?;
                        match b(rt)? {
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
                            _ => {}
                        }
                        for f in &u {
                            f(rt)?;
                        }
                    }
                    Ok(Flow::Next)
                })
            }
            StmtKind::ForEach { ty, var, iter, body } => {
                let (it, _) = self.expr(iter)?;
                self.scopes.push(Vec::new());
                let slot = self.declare(var, ty.clone(), s.pos)?;
                self.loop_depth += 1;
                let b = self.stmt(body, false)?;
                self.loop_depth -= 1;
                self.scopes.pop();
                let ty = ty.clone();
                let pos = iter.pos;
                Box::new(move |rt| {
                    let coll = it(rt)?;
                    let _copy = super::allocation::Allocation::iterable(&coll).map_err(at(pos))?;
                    let items = iter_values(&coll).map_err(at(pos))?;
                    for v in items {
                        rt.set_local(slot, coerce(&ty, v).map_err(at(pos))?);
                        match b(rt)? {
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
                            _ => {}
                        }
                    }
                    Ok(Flow::Next)
                })
            }
            StmtKind::Return(e) => match e {
                Some(e) => {
                    let (f, _) = self.expr(e)?;
                    Box::new(move |rt| Ok(Flow::Return(f(rt)?)))
                }
                None => Box::new(|_| Ok(Flow::Return(Val::Null))),
            },
            StmtKind::Break => {
                if self.loop_depth == 0 {
                    return Err(ScriptErr::compile(s.pos, "invalid break statement"));
                }
                Box::new(|_| Ok(Flow::Break))
            }
            StmtKind::Continue => {
                if self.loop_depth == 0 {
                    return Err(ScriptErr::compile(s.pos, "invalid continue statement"));
                }
                Box::new(|_| Ok(Flow::Continue))
            }
            StmtKind::Throw(e) => {
                let (f, _) = self.expr(e)?;
                let pos = e.pos;
                Box::new(move |rt| {
                    let v = f(rt)?;
                    Err(match v {
                        Val::Exception(x) => Box::new(ScriptErr { kind: snake_case(&x.0), msg: x.1.clone(), pos: Some(pos), compile: false, frames: &[], cause: None, breaker: None }),
                        Val::Null => npe("cannot throw null").at(pos),
                        other => illegal_arg(format!("Cannot throw {}", other.type_name())).at(pos),
                    })
                })
            }
            StmtKind::Try { body, catches } => {
                let b = self.block(body, implicit_return)?;
                let mut cs = Vec::new();
                for (ty, var, blk) in catches {
                    self.scopes.push(Vec::new());
                    let slot = self.declare(var, Ty::Ref("Exception".into()), s.pos)?;
                    let f = self.block(blk, implicit_return)?;
                    self.scopes.pop();
                    cs.push((ty.clone(), snake_case(ty), slot, f));
                }
                Box::new(move |rt| match b(rt) {
                    Ok(f) => Ok(f),
                    Err(e) => {
                        if e.kind == "painless_error" || e.breaker.is_some() {
                            return Err(e);
                        }
                        for (ty, kind, slot, f) in &cs {
                            let all = matches!(ty.as_str(), "Exception" | "RuntimeException" | "Throwable");
                            if all || *kind == e.kind {
                                let class = class_of_kind(&e.kind);
                                rt.set_local(*slot, Val::Exception(Rc::new((class, e.msg.clone()))));
                                return f(rt);
                            }
                        }
                        Err(e)
                    }
                })
            }
        })
    }

    // -----------------------------------------------------------------
    // Expressions.

    fn exprs(&mut self, es: &[Expr]) -> CResult<Vec<ExprFn>> {
        es.iter().map(|e| self.expr(e).map(|x| x.0)).collect()
    }

    fn expr(&mut self, e: &Expr) -> CResult<(ExprFn, St)> {
        let r = self.expr_inner(e)?;
        let d = self.node_dbg(e, r.1);
        self.dbg.insert(e as *const Expr as usize, (d, r.1));
        Ok(r)
    }

    fn info(&self, e: &Expr) -> (Option<usize>, St) {
        self.dbg.get(&(e as *const Expr as usize)).copied().unwrap_or((None, St::Def))
    }

    /// Debug position contributed by a call / constructor argument: a `def`
    /// value is cast to the parameter type at the argument's location.
    fn arg_dbg(&self, a: &Expr) -> Option<usize> {
        let (d, st) = self.info(a);
        if st == St::Def && !matches!(a.kind, ExprKind::Lambda { .. } | ExprKind::MethodRef { .. } | ExprKind::Null) {
            Some(loc(a))
        } else {
            d
        }
    }

    fn args_dbg(&self, args: &[Expr]) -> Option<usize> {
        args.iter().filter_map(|a| self.arg_dbg(a)).last()
    }

    fn last_dbg(&self, es: &[&Expr]) -> Option<usize> {
        es.iter().filter_map(|x| self.info(x).0).last()
    }

    /// Where ES's bytecode last records a source position while evaluating
    /// `e` (see `dbg`).
    fn node_dbg(&self, e: &Expr, st: St) -> Option<usize> {
        let pos = e.pos;
        match &e.kind {
            ExprKind::New { args, .. } => self.args_dbg(args).or(Some(pos)),
            ExprKind::NewArray { dims, .. } => self.args_dbg(dims).or(Some(pos)),
            ExprKind::NewArrayInit { items, .. } => Some(items.last().map(loc).unwrap_or(pos)),
            ExprKind::Call { obj, args, .. } => {
                if self.static_class(obj).is_some() || self.info(obj).1.is_typed() {
                    self.args_dbg(args).or(Some(pos))
                } else {
                    // `def` call: the call records its `.`, then each argument
                    // its own positions (no cast at the call site).
                    self.last_dbg(&args.iter().collect::<Vec<_>>()).or(Some(pos))
                }
            }
            ExprKind::FnCall { args, .. } => self.last_dbg(&args.iter().collect::<Vec<_>>()).or(Some(pos)),
            ExprKind::Field { obj, name, .. } => match self.static_class(obj) {
                Some(c) => match static_field(c, name) {
                    Some(Val::Big(_) | Val::Time(_) | Val::Obj(_)) => Some(pos),
                    _ => None,
                },
                None => Some(pos),
            },
            ExprKind::Index { .. } | ExprKind::RegexOp { .. } => Some(pos),
            ExprKind::Binary(_, a, b) => {
                let ((da, sa), (db, sb)) = (self.info(a), self.info(b));
                if sa == St::Def || sb == St::Def { Some(loc(a)) } else { db.or(da) }
            }
            ExprKind::Unary(_, x) => {
                let (d, s) = self.info(x);
                if s == St::Def { Some(pos) } else { d }
            }
            ExprKind::And(a, b) | ExprKind::Or(a, b) | ExprKind::Elvis(a, b) => self.last_dbg(&[&**a, &**b]),
            ExprKind::Ternary(c, a, b) => self.last_dbg(&[&**c, &**a, &**b]),
            ExprKind::Cast { expr, .. } | ExprKind::InstanceOf { expr, .. } => self.info(expr).0,
            ExprKind::Assign { target, value, .. } => match target.kind {
                ExprKind::Field { .. } | ExprKind::Index { .. } => Some(pos),
                _ => self.info(value).0,
            },
            ExprKind::IncDec { target, .. } => self.info(target).0,
            ExprKind::ListLit(items) => self.last_dbg(&items.iter().collect::<Vec<_>>()),
            ExprKind::MapLit(pairs) => self.last_dbg(&pairs.iter().flat_map(|(k, v)| [k, v]).collect::<Vec<_>>()),
            _ => {
                let _ = st;
                None
            }
        }
    }

    /// Receiver of a list-style element read: `doc['x']` / `doc.x`,
    /// optionally under a cast that always succeeds on doc values.
    fn doc_list_recv(&mut self, r: &Expr) -> Option<usize> {
        if let ExprKind::Cast { ty, expr } = &r.kind {
            let listish = match ty {
                Ty::Def => true,
                Ty::Ref(n) => matches!(
                    n.as_str(),
                    "List" | "ArrayList" | "LinkedList" | "AbstractList" | "RandomAccess" | "Collection" | "Iterable" | "AbstractCollection" | "Object" | "ScriptDocValues"
                ),
                _ => false,
            };
            return if listish { self.doc_slot(expr) } else { None };
        }
        self.doc_slot(r)
    }

    /// Receiver of a geo point `lat` / `lon` read: `doc['x']`, or its first
    /// point (`.value`, `.getValue()`, `.get(0)`, `[0]`).
    fn doc_geo_recv(&mut self, r: &Expr) -> Option<usize> {
        match &r.kind {
            ExprKind::Field { obj, name, null_safe: false } if name == "value" => self.doc_slot(obj),
            ExprKind::Call { obj, name, args, null_safe: false } if name == "getValue" && args.is_empty() => self.doc_slot(obj),
            ExprKind::Call { obj, name, args, null_safe: false } if name == "get" && args.len() == 1 && matches!(args[0].kind, ExprKind::Int(0)) => self.doc_list_recv(obj),
            ExprKind::Index { obj, idx } if matches!(idx.kind, ExprKind::Int(0)) => self.doc_list_recv(obj),
            _ => self.doc_slot(r),
        }
    }

    /// Doc-value reads that can skip materializing the field's values
    /// (see [`DocFast`]).
    fn doc_fast(&mut self, e: &Expr) -> Option<DocFast> {
        match &e.kind {
            ExprKind::Call { obj, name, args, null_safe: false } => match (name.as_str(), args.as_slice()) {
                ("get", [a]) if matches!(a.kind, ExprKind::Int(0)) => self.doc_list_recv(obj).map(DocFast::First),
                ("getLat", []) => self.doc_geo_recv(obj).map(|s| DocFast::Geo(s, true)),
                ("getLon", []) => self.doc_geo_recv(obj).map(|s| DocFast::Geo(s, false)),
                _ => None,
            },
            ExprKind::Index { obj, idx } if matches!(idx.kind, ExprKind::Int(0)) => self.doc_list_recv(obj).map(DocFast::First),
            ExprKind::Field { obj, name, null_safe: false } if name == "lat" || name == "lon" => self.doc_geo_recv(obj).map(|s| DocFast::Geo(s, name == "lat")),
            _ => None,
        }
    }

    fn expr_inner(&mut self, e: &Expr) -> CResult<(ExprFn, St)> {
        if !self.num_off && self.num_root(e) {
            let nf = self.num(e).map(Ok).or_else(|| self.num_cond(e).map(Err));
            if let Some(nf) = nf {
                // The general node is always compiled: it reports compile
                // errors and evaluates every doc the fast path bails on.
                self.num_off = true;
                let r = self.expr_node(e);
                self.num_off = false;
                let (f, st) = r?;
                return Ok(match nf {
                    Ok(nf) => (nf.with_fallback(f), st),
                    Err(c) => (
                        Box::new(move |rt| match c(rt) {
                            Some(b) => Ok(Val::Bool(b)),
                            None => f(rt),
                        }),
                        st,
                    ),
                });
            }
        }
        let fast = self.doc_fast(e);
        // The general node is always compiled: it reports compile errors
        // and serves every doc the fast read cannot (missing values,
        // multi-valued fields, other field types, errors).
        let (f, st) = self.expr_node(e)?;
        Ok(match fast {
            None => (f, st),
            Some(DocFast::First(slot)) => (
                Box::new(move |rt| match rt.doc.fast_first(slot) {
                    Some(v) => Ok(v),
                    None => f(rt),
                }),
                st,
            ),
            Some(DocFast::Geo(slot, lat)) => (
                Box::new(move |rt| match rt.doc.fast_geo(slot, lat) {
                    Some(x) => Ok(Val::Double(x)),
                    None => f(rt),
                }),
                st,
            ),
        })
    }

    fn expr_node(&mut self, e: &Expr) -> CResult<(ExprFn, St)> {
        let pos = e.pos;
        Ok(match &e.kind {
            ExprKind::Null => (Box::new(|_| Ok(Val::Null)), St::Null),
            ExprKind::Bool(b) => {
                let b = *b;
                (Box::new(move |_| Ok(Val::Bool(b))), St::Bool)
            }
            ExprKind::Int(i) => {
                let i = *i;
                (Box::new(move |_| Ok(Val::Int(i))), St::Int)
            }
            ExprKind::Long(l) => {
                let l = *l;
                (Box::new(move |_| Ok(Val::Long(l))), St::Long)
            }
            ExprKind::Double(d) => {
                let d = *d;
                (Box::new(move |_| Ok(Val::Double(d))), St::Double)
            }
            ExprKind::Regex { pattern, flags, units } => {
                if !super::regex::regex_enabled() {
                    let mut e = ScriptErr::compile(pos, super::regex::DISABLED_MSG);
                    e.kind = "illegal_state_exception".into();
                    return Err(e);
                }
                let fl = super::regex::literal_flags(flags).map_err(|m| ScriptErr::compile(pos, m))?;
                let pd = std::sync::Arc::new(match units {Some(units)=>super::regex::compile_pattern_units(units,fl),None=>super::regex::compile_pattern(pattern,fl)}.map_err(|e| {
                    let mut c = ScriptErr::compile(pos + 1, e.msg.clone());
                    c.kind = e.kind.clone();
                    if let Some((_, desc)) = e.msg.rsplit_once(": ") {
                        // Java's PatternSyntaxException message: description, index, pattern, caret.
                        let idx = pattern.chars().count().saturating_sub(1);
                        c = c.with_cause("pattern_syntax_exception", format!("{desc} near index {idx}\n{pattern}\n{}^", " ".repeat(idx)));
                    }
                    c
                })?);
                (Box::new(move |_| Ok(Val::Pattern(pd.clone()))), St::Def)
            }
            ExprKind::RegexOp { find, text, pat } => {
                let (tf, _) = self.expr(text)?;
                let (pf, pst) = self.expr(pat)?;
                if pst == St::Str {
                    return Err(cc(pat.pos, "Cannot cast from [java.lang.String] to [java.util.regex.Pattern].".to_string()));
                }
                let find = *find;
                (Box::new(move |rt| super::regex::regex_op(find, &tf(rt)?, &pf(rt)?).map_err(at(pos))), St::Bool)
            }
            ExprKind::Utf16(units) => {
                let units=units.clone();
                (Box::new(move |_| Ok(Val::utf16(units.clone()))), St::Str)
            }
            ExprKind::Str(s) => {
                let id = self.string_id(s);
                (Box::new(move |rt| Ok(rt.consts[id].clone())), St::Str)
            }
            ExprKind::Var(name) => self.var(name, pos)?,
            ExprKind::ListLit(items) => {
                let fs = self.exprs(items)?;
                (Box::new(move |rt| {
                    let allocation = super::allocation::Allocation::values(fs.len())?;
                    let mut items = Vec::with_capacity(fs.len());
                    for f in &fs { items.push(f(rt)?); }
                    Ok(Val::new_list(items).allocated(allocation))
                }), St::List)
            }
            ExprKind::MapLit(pairs) => {
                let mut fs = Vec::new();
                for (k, v) in pairs {
                    fs.push((self.expr(k)?.0, self.expr(v)?.0));
                }
                (
                    Box::new(move |rt| {
                        let allocation = super::allocation::Allocation::entries(fs.len())?;
                        let mut m = IndexMap::with_capacity(fs.len());
                        for (k, v) in &fs {
                            let k = k(rt)?;
                            let v = v(rt)?;
                            m.insert(Key::from_val(&k), v);
                        }
                        Ok(Val::new_hash_map(m).allocated(allocation))
                    }),
                    St::Map,
                )
            }
            ExprKind::New { ty, args } => self.new_obj(ty, args, pos)?,
            ExprKind::NewArray { ty, dims } => {
                let ds = self.exprs(dims)?;
                let ty = ty.clone();
                (
                    Box::new(move |rt| {
                        let dims: Vec<i64> = ds.iter().map(|f| f(rt).and_then(|v| v.to_index())).collect::<SResult<_>>().map_err(at(pos))?;
                        fn build(ty: &Ty, dims: &[i64]) -> SResult<Val> {
                            let n = dims[0];
                            if n < 0 || n > MAX_LOOP as i64 {
                                return Err(ScriptErr::new("negative_array_size_exception", n.to_string()));
                            }
                            let allocation = super::allocation::Allocation::values(n as usize)?;
                            let mut items = Vec::with_capacity(n as usize);
                            for _ in 0..n { items.push(if dims.len() == 1 { default_of(ty) } else { build(ty, &dims[1..])? }); }
                            Ok(Val::new_array(ty.clone(), items).allocated(allocation))
                        }
                        build(&ty, &dims)
                    }),
                    St::Ref,
                )
            }
            ExprKind::NewArrayInit { ty, items } => {
                let fs = self.exprs(items)?;
                let elem = match ty {
                    Ty::Array(t) => (**t).clone(),
                    t => t.clone(),
                };
                (Box::new(move |rt| {
                    let allocation = super::allocation::Allocation::values(fs.len())?;
                    let mut items = Vec::with_capacity(fs.len());
                    for f in &fs { items.push(f(rt).and_then(|v| coerce(&elem, v)).map_err(at(pos))?); }
                    Ok(Val::new_array(elem.clone(), items).allocated(allocation))
                }), St::Ref)
            }
            ExprKind::Unary(op, x) => {
                let (f, st) = self.expr(x)?;
                let op = *op;
                if op == UnOp::Not && !st.is_dyn() && st != St::Bool {
                    return Err(cc(x.pos, format!("Cannot cast from [{}] to [boolean].", jname_st(st))));
                }
                let rst = match op {
                    UnOp::Not => St::Bool,
                    _ => st,
                };
                (Box::new(move |rt| unary(op, &f(rt)?).map_err(at(pos))), rst)
            }
            ExprKind::Binary(op, a, b) => self.binary(*op, a, b, pos)?,
            ExprKind::And(a, b) | ExprKind::Or(a, b) => {
                let is_and = matches!(e.kind, ExprKind::And(..));
                let x = self.cond(a)?;
                let y = self.cond(b)?;
                (Box::new(move |rt| Ok(Val::Bool(if is_and { x(rt)? && y(rt)? } else { x(rt)? || y(rt)? }))), St::Bool)
            }
            ExprKind::Ternary(c, a, b) => {
                let c = self.cond(c)?;
                let (x, sa) = self.expr(a)?;
                let (y, sb) = self.expr(b)?;
                let st = if sa == sb { sa } else { St::Def };
                (Box::new(move |rt| if c(rt)? { x(rt) } else { y(rt) }), st)
            }
            ExprKind::Elvis(a, b) => {
                let (x, _) = self.expr(a)?;
                let (y, _) = self.expr(b)?;
                (
                    Box::new(move |rt| {
                        let v = x(rt)?;
                        if v.is_null() { y(rt) } else { Ok(v) }
                    }),
                    St::Def,
                )
            }
            ExprKind::Assign { target, op, value } => self.assign(target, *op, value, pos)?,
            ExprKind::IncDec { target, inc, prefix } => {
                let one = Expr { kind: ExprKind::Int(1), pos };
                let op = if *inc { BinOp::Add } else { BinOp::Sub };
                let (f, st) = self.assign(target, Some(op), &one, pos)?;
                let general: ExprFn = if *prefix {
                    f
                } else {
                    // Postfix: value before the update.
                    let inv = if *inc { BinOp::Sub } else { BinOp::Add };
                    Box::new(move |rt| binary(inv, &f(rt)?, &Val::Int(1)).map_err(at(pos)))
                };
                // `i++` / `--i` on an `int` / `long` local: the wrapping
                // update in place (what `binary` + `cast` compute for it);
                // any other value in the slot takes the general node.
                let local = match &target.kind {
                    ExprKind::Var(n) => self.lookup(n),
                    _ => None,
                };
                let (inc, prefix) = (*inc, *prefix);
                match local {
                    Some((slot, Ty::Int)) => (
                        Box::new(move |rt| {
                            let cur = match rt.local(slot) {
                                Val::Int(i) => Some(*i),
                                _ => None,
                            };
                            match cur {
                                Some(old) => {
                                    let new = if inc { old.wrapping_add(1) } else { old.wrapping_sub(1) };
                                    rt.set_local(slot, Val::Int(new));
                                    Ok(Val::Int(if prefix { new } else { old }))
                                }
                                None => general(rt),
                            }
                        }),
                        st,
                    ),
                    Some((slot, Ty::Long)) => (
                        Box::new(move |rt| {
                            let cur = match rt.local(slot) {
                                Val::Long(l) => Some(*l),
                                _ => None,
                            };
                            match cur {
                                Some(old) => {
                                    let new = if inc { old.wrapping_add(1) } else { old.wrapping_sub(1) };
                                    rt.set_local(slot, Val::Long(new));
                                    Ok(Val::Long(if prefix { new } else { old }))
                                }
                                None => general(rt),
                            }
                        }),
                        st,
                    ),
                    _ => (general, st),
                }
            }
            ExprKind::Field { obj, name, null_safe } => {
                // `doc.x`: same as `doc['x']`.
                if let Some(slot) = self.doc_slot(e) {
                    return Ok((Box::new(move |rt| rt.doc.values(slot).map_err(at(pos))), St::Def));
                }
                self.field(obj, name, *null_safe, pos)?
            }
            ExprKind::Index { obj, idx } => {
                if let Some(slot) = self.doc_slot(e) {
                    return Ok((Box::new(move |rt| rt.doc.values(slot).map_err(at(pos))), St::Def));
                }
                if let Some(k) = self.param_key(e) {
                    return Ok(self.param_ref(&k));
                }
                let (o, _) = self.expr(obj)?;
                let (i, _) = self.expr(idx)?;
                (
                    Box::new(move |rt| {
                        let ov = o(rt)?;
                        let iv = i(rt)?;
                        index_get(rt, &ov, &iv).map_err(at(pos))
                    }),
                    St::Def,
                )
            }
            ExprKind::Call { obj, name, args, null_safe } => self.call(obj, name, args, *null_safe, pos)?,
            ExprKind::FnCall { name, args } => self.fn_call(name, args, pos)?,
            ExprKind::Cast { ty, expr } => {
                let (f, st) = self.expr(expr)?;
                if st == St::Null && matches!(ty, Ty::Int | Ty::Long | Ty::Double | Ty::Float | Ty::Boolean | Ty::Short | Ty::Byte | Ty::Char) {
                    return Err(ScriptErr::compile(expr.pos, format!("Cannot cast null to a primitive type [{}].", ty.name())));
                }
                if (st == St::Str && matches!(ty, Ty::Int | Ty::Long | Ty::Double | Ty::Float | Ty::Boolean | Ty::Short | Ty::Byte))
                    || (matches!(st, St::Int | St::Long | St::Double | St::Bool) && matches!(ty, Ty::String | Ty::Ref(_)))
                    || (st == St::Str && matches!(ty, Ty::Ref(_)) && !matches!(ty, Ty::Ref(n) if matches!(n.as_str(), "Object" | "CharSequence" | "Comparable")))
                {
                    return Err(cc(expr.pos, format!("Cannot cast from [{}] to [{}].", jname_st(st), fqn_ty(ty))));
                }
                let ty = ty.clone();
                let rst = St::of_ty(&ty);
                (Box::new(move |rt| cast(&ty, f(rt)?).map_err(at(pos))), rst)
            }
            ExprKind::InstanceOf { expr, ty } => {
                let (f, _) = self.expr(expr)?;
                let ty = ty.clone();
                (Box::new(move |rt| Ok(Val::Bool(instance_of(&f(rt)?, &ty)))), St::Bool)
            }
            ExprKind::Lambda { params, body } => {
                let id = self.lambda(params, body, pos)?;
                (Box::new(move |_| Ok(Val::Lambda(id))), St::Def)
            }
            ExprKind::MethodRef { class, method } => {
                if class == "this" && !self.fn_sigs.iter().any(|f| f.0 == *method) {
                    return Err(ScriptErr::compile(pos, format!("Unknown reference [this::{method}].")));
                }
                let r = (class.clone(), method.clone());
                (Box::new(move |_| Ok(Val::MethodRef(Rc::new(r.clone())))), St::Def)
            }
        })
    }

    fn var(&mut self, name: &str, pos: usize) -> CResult<(ExprFn, St)> {
        if let Some((slot, ty)) = self.lookup(name) {
            return Ok((Box::new(move |rt| Ok(rt.local(slot).fast_clone())), St::of_ty(&ty)));
        }
        // Painless user functions have their own scope. Context variables
        // must be passed explicitly; a parameter called ctx/params is local.
        if self.in_function {
            return Err(ScriptErr::compile(pos, format!("cannot resolve symbol [{name}]")));
        }
        Ok(match name {
            "params" => (Box::new(|rt| Ok(rt.params.clone())), St::Def),
            "_score" => {
                self.uses_score = true;
                (Box::new(|rt| Ok(Val::Double(rt.score))), St::Double)
            }
            "doc" => (Box::new(|_| Ok(Val::Doc)), St::Def),
            "ctx" => (Box::new(|rt| Ok(rt.ctx.clone())), St::Def),
            // `scripted_metric`: the phase's state (`ctx` slot of the executor).
            "state" if matches!(self.ctx, ScriptContext::MetricInit | ScriptContext::MetricMap | ScriptContext::MetricCombine) => (Box::new(|rt| Ok(rt.ctx.clone())), St::Def),
            "states" if self.ctx == ScriptContext::MetricReduce => (Box::new(|rt| Ok(rt.ctx.clone())), St::Def),
            "_value" => (Box::new(|rt| Ok(rt.value.clone())), St::Def),
            "_source" => (Box::new(|rt| rt.source()), St::Def),
            _ => {
                if type_of_name(name).is_some() || STATIC_CLASSES.contains(&name) {
                    return Err(ScriptErr::compile(pos, format!("cannot resolve symbol [{name}]")));
                }
                return Err(ScriptErr::compile(pos, format!("cannot resolve symbol [{name}]")));
            }
        })
    }

    fn param_ref(&mut self, key: &str) -> (ExprFn, St) {
        if key == "_source" {
            return (Box::new(|rt| rt.source()), St::Def);
        }
        let id = self.param_id(key);
        (Box::new(move |rt| Ok(rt.param_vals[id].fast_clone())), St::Def)
    }

    /// Operands that can be read without a nested closure call.
    fn leaf_of(&mut self, e: &Expr) -> Option<Leaf> {
        match &e.kind {
            ExprKind::Int(i) => Some(Leaf::Int(*i)),
            ExprKind::Long(l) => Some(Leaf::Long(*l)),
            ExprKind::Double(d) => Some(Leaf::Double(*d)),
            ExprKind::Var(n) => match self.lookup(n) {
                Some((slot, _)) => Some(Leaf::Local(slot)),
                None if !self.in_function && n == "_score" => Some(Leaf::Score),
                None => None,
            },
            ExprKind::Field { obj, name, null_safe: false } if name == "value" => self.doc_slot(obj).map(|s| Leaf::Doc(s, e.pos)),
            ExprKind::Field { .. } | ExprKind::Index { .. } => match self.param_key(e) {
                Some(k) if k != "_source" => Some(Leaf::Param(self.param_id(&k))),
                _ => None,
            },
            _ => None,
        }
    }

    fn operand(&mut self, e: &Expr) -> CResult<(Operand, St)> {
        let (f, st) = self.expr(e)?;
        Ok(match self.leaf_of(e) {
            Some(l) => (Operand::Leaf(l), st),
            None => (Operand::Expr(f), st),
        })
    }

    fn binary(&mut self, op: BinOp, a: &Expr, b: &Expr, _op_pos: usize) -> CResult<(ExprFn, St)> {
        // ES locates a binary expression (and any runtime error in it) at its
        // first token, i.e. the start of the left operand.
        let pos = a.pos;
        let (l, sa) = self.operand(a)?;
        let (r, sb) = self.operand(b)?;
        if matches!(op, BinOp::Div | BinOp::Rem) && matches!(b.kind, ExprKind::Int(0) | ExprKind::Long(0)) && matches!(a.kind, ExprKind::Int(_) | ExprKind::Long(_)) {
            // Constant folding in ES evaluates this at compile time.
            let mut e = ScriptErr::compile(pos, "/ by zero");
            e.kind = "arithmetic_exception".into();
            return Err(e);
        }
        // Light type checks on statically known operand types.
        let arith = matches!(op, BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge);
        if arith && (matches!(sa, St::Str | St::Bool) || matches!(sb, St::Str | St::Bool)) {
            let word = match op {
                BinOp::Lt => Some("less than"),
                BinOp::Le => Some("less than or equals"),
                BinOp::Gt => Some("greater than"),
                BinOp::Ge => Some("greater than or equals"),
                _ => None,
            };
            if let Some(w) = word {
                return Err(cc(a.pos, format!("cannot apply the {w} operator [{}] to the types [{}] and [{}]", op.symbol(), jname_st(sa), jname_st(sb))));
            }
            let (bad, bad_pos, other) = if matches!(sa, St::Str | St::Bool) { (sa, a.pos, sb) } else { (sb, b.pos, sa) };
            let target = match other {
                St::Long => "long",
                St::Double => "double",
                _ => "int",
            };
            return Err(cc(bad_pos, format!("Cannot cast from [{}] to [{}].", jname_st(bad), target)));
        }
        let st = match op {
            BinOp::Eq | BinOp::Ne | BinOp::RefEq | BinOp::RefNe | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => St::Bool,
            BinOp::Add if sa == St::Str || sb == St::Str => St::Str,
            _ if sa.is_num() && sb.is_num() => {
                if sa == St::Double || sb == St::Double {
                    St::Double
                } else if sa == St::Long || sb == St::Long {
                    St::Long
                } else {
                    St::Int
                }
            }
            _ => St::Def,
        };
        let f: ExprFn = match op {
            // Hot shapes without the generic dispatch.
            BinOp::Add => Box::new(move |rt| {
                let x = l.eval(rt)?;
                let y = r.eval(rt)?;
                fast_num(BinOp::Add, x, y).map_err(at(pos))
            }),
            BinOp::Mul => Box::new(move |rt| {
                let x = l.eval(rt)?;
                let y = r.eval(rt)?;
                fast_num(BinOp::Mul, x, y).map_err(at(pos))
            }),
            _ => Box::new(move |rt| {
                let x = l.eval(rt)?;
                let y = r.eval(rt)?;
                fast_num(op, x, y).map_err(at(pos))
            }),
        };
        Ok((f, st))
    }

    fn field(&mut self, obj: &Expr, name: &str, null_safe: bool, pos: usize) -> CResult<(ExprFn, St)> {
        if let Some(class) = self.static_class(obj) {
            return match static_field(class, name) {
                Some(Val::Big(_) | Val::Time(_) | Val::Obj(_)) => {
                    let (c, n) = (class.to_string(), name.to_string());
                    Ok((Box::new(move |_| Ok(static_field(&c, &n).unwrap_or(Val::Null))), St::Def))
                }
                Some(v) => Ok(const_fn(&v)),
                None => Err(ScriptErr::compile(pos, format!("cannot resolve symbol [{class}.{name}]"))),
            };
        }
        if let Some(slot) = self.doc_slot(obj) {
            match name {
                "value" => return Ok((Box::new(move |rt| rt.doc.value(slot).map_err(at(pos))), St::Def)),
                "empty" => return Ok((Box::new(move |rt| Ok(Val::Bool(rt.doc.size(slot).map_err(at(pos))? == 0))), St::Bool)),
                "length" => return Ok((Box::new(move |rt| Ok(Val::Int(rt.doc.size(slot).map_err(at(pos))? as i32))), St::Int)),
                "values" => {
                    return Ok((
                        Box::new(move |rt| match rt.doc.values(slot).map_err(at(pos))? {
                            Val::DocValues(d) => {
                                let allocation = super::allocation::Allocation::values(d.values.len()).map_err(at(pos))?;
                                Ok(Val::new_list(d.values.clone()).allocated(allocation))
                            },
                            v => Ok(v),
                        }),
                        St::Def,
                    ))
                }
                _ => {}
            }
        }
        if let Some(k) = self.param_key(&Expr { kind: ExprKind::Field { obj: Box::new(obj.clone()), name: name.to_string(), null_safe }, pos }) {
            return Ok(self.param_ref(&k));
        }
        let (o, ost) = self.expr(obj)?;
        if ost == St::Str {
            return Err(ScriptErr::compile(pos, format!("field [java.lang.String, {name}] not found")));
        }
        let name = name.to_string();
        Ok((
            Box::new(move |rt| {
                let ov = o(rt)?;
                if null_safe && ov.is_null() {
                    return Ok(Val::Null);
                }
                field_get(rt, &ov, &name).map_err(at(pos))
            }),
            St::Def,
        ))
    }

    fn call(&mut self, obj: &Expr, name: &str, args: &[Expr], null_safe: bool, pos: usize) -> CResult<(ExprFn, St)> {
        let n = args.len();
        if let Some(class) = self.static_class(obj) {
            let class = class.to_string();
            let mut fs = self.exprs(args)?;
            // Painless permits narrowing an in-range int literal at a char
            // parameter, but not an int carried by a dynamic variable.
            if class == "Character" {
                for i in 0..super::runtime::character_char_arguments(name,n) {
                    if let ExprKind::Int(value) = args[i].kind {
                        if !(0..=65535).contains(&value) {
                            return Err(cc(args[i].pos,"Cannot cast from [int] to [char].".into()));
                        }
                        fs[i]=Box::new(move |_| Ok(Val::Char(value as u16)));
                    }
                }
            }
            let name_s = name.to_string();
            if let Some(e) = wl_entry(&class) {
                if !e.1.contains(&format!("{name}/{n}").as_str()) {
                    return Err(ScriptErr::compile(pos, format!("static method [{}, {name}/{n}] not found", e.0)));
                }
            }
            if class == "String" && name == "format" {
                // `format(String, def[])` / `format(Locale, String, def[])`: Painless has no varargs.
                let sts: Vec<St> = args.iter().map(|a| self.static_type(a)).collect();
                if n == 3 && sts[0] != St::Def && sts[0] != St::Null {
                    return Err(cc(self.lit_pos(&args[0]), format!("Cannot cast from [{}] to [java.util.Locale].", jname_st(sts[0]))));
                }
                let last = &args[n - 1];
                if sts[n - 1] != St::Def && sts[n - 1] != St::Null && !matches!(last.kind, ExprKind::NewArray { .. } | ExprKind::NewArrayInit { .. }) {
                    return Err(cc(self.lit_pos(last), format!("Cannot cast from [{}] to [def[]].", jname_st(sts[n - 1]))));
                }
            }
            if (class == "Math" || class == "StrictMath") && n <= 2 {
                // Non-allocating numeric path.
                let st = match name {
                    "abs" | "max" | "min" => St::Def,
                    "round" => St::Long,
                    _ => St::Double,
                };
                if super::runtime::math(name, &vec![Val::Int(1); n]).ok().flatten().is_none() && name != "random" {
                    return Err(ScriptErr::compile(pos, format!("Unknown call [{name}] with [{n}] arguments on type [Math].")));
                }
                // Constant arguments fold at compile time (as javac / ES do
                // for constant expressions); errors stay at runtime.
                if name != "random" && n > 0 && args.iter().all(|a| matches!(a.kind, ExprKind::Int(_) | ExprKind::Long(_) | ExprKind::Double(_))) {
                    let lits: Vec<Val> = args
                        .iter()
                        .map(|a| match a.kind {
                            ExprKind::Int(i) => Val::Int(i),
                            ExprKind::Long(l) => Val::Long(l),
                            ExprKind::Double(d) => Val::Double(d),
                            _ => Val::Null,
                        })
                        .collect();
                    match math(&name_s, &lits) {
                        Ok(Some(Val::Double(d))) => return Ok((Box::new(move |_| Ok(Val::Double(d))), st)),
                        Ok(Some(Val::Long(l))) => return Ok((Box::new(move |_| Ok(Val::Long(l))), st)),
                        Ok(Some(Val::Int(i))) => return Ok((Box::new(move |_| Ok(Val::Int(i))), st)),
                        _ => {}
                    }
                }
                if n == 1 {
                    if let Some(op) = math_unary(name) {
                        let a = fs.into_iter().next().unwrap();
                        return Ok((
                            Box::new(move |rt| {
                                let x = a(rt)?;
                                let d = match &x {
                                    Val::Double(d) => *d,
                                    Val::Int(i) => *i as f64,
                                    Val::Long(l) => *l as f64,
                                    other => other.to_f64().map_err(at(pos))?,
                                };
                                Ok(Val::Double(op(d)))
                            }),
                            st,
                        ));
                    }
                }
                let f: ExprFn = match n {
                    0 => Box::new(move |_| math(&name_s, &[]).map(|v| v.unwrap_or(Val::Null)).map_err(at(pos))),
                    1 => {
                        let a = fs.into_iter().next().unwrap();
                        Box::new(move |rt| {
                            let x = a(rt)?;
                            math(&name_s, std::slice::from_ref(&x)).map(|v| v.unwrap_or(Val::Null)).map_err(at(pos))
                        })
                    }
                    _ => {
                        let mut it = fs.into_iter();
                        let (a, b) = (it.next().unwrap(), it.next().unwrap());
                        Box::new(move |rt| {
                            let x = a(rt)?;
                            let y = b(rt)?;
                            math(&name_s, &[x, y]).map(|v| v.unwrap_or(Val::Null)).map_err(at(pos))
                        })
                    }
                };
                return Ok((f, st));
            }
            // ES records the call's position, then each argument's: a
            // failure inside the method is reported at the last of these.
            let epos = self.args_dbg(args).unwrap_or(pos);
            let rst = if super::jbig::CLASSES.contains(&class.as_str()) || super::jtime::CLASSES.contains(&class.as_str()) { St::Ref } else { St::Def };
            return Ok((
                Box::new(move |rt| {
                    let a: Vec<Val> = fs.iter().map(|f| f(rt)).collect::<SResult<_>>()?;
                    match call_static(rt, &class, &name_s, a).map_err(at(epos))? {
                        Some(v) => Ok(v),
                        None => Err(illegal_arg(format!("Unknown call [{name_s}] with [{n}] arguments on type [{class}].")).at(pos)),
                    }
                }),
                rst,
            ));
        }
        if let Some(slot) = self.doc_slot(obj) {
            match (name, n) {
                ("size", 0) | ("getLength", 0) => return Ok((Box::new(move |rt| Ok(Val::Int(rt.doc.size(slot).map_err(at(pos))? as i32))), St::Int)),
                ("isEmpty", 0) => return Ok((Box::new(move |rt| Ok(Val::Bool(rt.doc.size(slot).map_err(at(pos))? == 0))), St::Bool)),
                ("getValue", 0) => return Ok((Box::new(move |rt| rt.doc.value(slot).map_err(at(pos))), St::Def)),
                _ => {}
            }
        }
        // ScriptDocValues element access like doc['x'].value.getMillis() works generically.
        let (o, ost) = self.expr(obj)?;
        if matches!(name, "replaceAll" | "replaceFirst") && n == 2 && matches!(args[0].kind, ExprKind::Regex { .. }) && matches!(args[1].kind, ExprKind::Str(_)) {
            return Err(cc(args[1].pos, "Cannot cast from [java.lang.String] to [java.util.function.Function].".to_string()));
        }
        if ost == St::Str {
            let key = format!("{name}/{n}");
            let found = ["String", "CharSequence", "Comparable", "Object"].iter().any(|c| wl_entry(c).is_some_and(|e| e.1.contains(&key.as_str())));
            if !found {
                return Err(ScriptErr::compile(pos, format!("member method [java.lang.String, {key}] not found")));
            }
        }
        if let St::Cls(c) = ost {
            let key = format!("{name}/{n}");
            let found = [c, "java.lang.Number", "java.lang.Comparable", "java.lang.Object"]
                .iter()
                .any(|c| super::whitelist::WHITELIST.iter().any(|e| e.0 == *c && e.1.contains(&key.as_str())));
            if !found {
                return Err(ScriptErr::compile(pos, format!("member method [{c}, {key}] not found")));
            }
        }
        let fs = self.exprs(args)?;
        // Statically bound calls report failures at the last position the
        // arguments recorded (else the call's `.`); `def` calls likewise, but
        // their arguments are not cast, so only the arguments' own positions count.
        let pos = if ost.is_typed() { self.args_dbg(args).unwrap_or(pos) } else { self.last_dbg(&args.iter().collect::<Vec<_>>()).unwrap_or(pos) };
        let rst = match ost {
            St::Cls(_) | St::Str => St::Ref,
            St::Ref | St::List | St::Map if !DEF_RETURNING.contains(&name) => St::Ref,
            _ => St::Def,
        };
        let name = name.to_string();
        if let Some(kind) = super::methods::StrFast::of(&name, fs.len()) {
            // Bound once here instead of dispatched by name per call; any
            // receiver/argument the fast form does not cover goes through
            // `call_method` with the values already evaluated (same order:
            // receiver, null-safe check, arguments).
            return Ok((
                Box::new(move |rt| {
                    let recv = o(rt)?;
                    if null_safe && recv.is_null() {
                        return Ok(Val::Null);
                    }
                    let a0 = match fs.first() {
                        Some(f) => Some(f(rt)?),
                        None => None,
                    };
                    let a1 = match fs.get(1) {
                        Some(f) => Some(f(rt)?),
                        None => None,
                    };
                    if let Some(v) = kind.eval(&recv, a0.as_ref(), a1.as_ref()) {
                        return Ok(v);
                    }
                    let a: Vec<Val> = a0.into_iter().chain(a1).collect();
                    super::methods::call_method(rt, recv, &name, a).map_err(at(pos))
                }),
                rst,
            ));
        }
        Ok((
            Box::new(move |rt| {
                let recv = o(rt)?;
                if null_safe && recv.is_null() {
                    return Ok(Val::Null);
                }
                let mut a = Vec::with_capacity(fs.len());
                for f in &fs {
                    a.push(f(rt)?);
                }
                super::methods::call_method(rt, recv, &name, a).map_err(at(pos))
            }),
            rst,
        ))
    }

    fn fn_call(&mut self, name: &str, args: &[Expr], pos: usize) -> CResult<(ExprFn, St)> {
        let fs = self.exprs(args)?;
        let n = fs.len();
        if let Some(idx) = self.fn_sigs.iter().position(|f| f.0 == name && f.1.len() == n) {
            let st = St::of_ty(&self.fn_sigs[idx].2);
            return Ok((
                Box::new(move |rt| {
                    let a: Vec<Val> = fs.iter().map(|f| f(rt)).collect::<SResult<_>>()?;
                    rt.invoke_fn(idx, a).map_err(at(pos))
                }),
                st,
            ));
        }
        if self.ctx == ScriptContext::Score {
            if let Some(kind) = score_fns::ScoreFn::of(name, n) {
                return Ok(self.score_fn(kind, args, fs, pos));
            }
        }
        if name == "metadata" && n == 0 && matches!(self.ctx, ScriptContext::Ingest | ScriptContext::IngestConditional | ScriptContext::PainlessTest) {
            return Ok((
                Box::new(move |rt| {
                    let mut m = IndexMap::new();
                    let ctx = rt.ctx.clone();
                    let get = |k: &str| match &ctx {
                        Val::Map(c) => c.borrow().get(&Key::str(k)).cloned().unwrap_or(Val::Null),
                        _ => Val::Null,
                    };
                    for (src, dst) in [("_index", "index"), ("_id", "id"), ("_routing", "routing"), ("_version", "version"), ("_version_type", "versionType"), ("_if_seq_no", "ifSeqNo"), ("_if_primary_term", "ifPrimaryTerm")] {
                        m.insert(Key::str(dst), get(src));
                    }
                    let now = match get("_ingest") {
                        Val::Map(i) => i.borrow().get(&Key::str("timestamp")).cloned().unwrap_or(Val::Null),
                        _ => Val::Null,
                    };
                    let now = match &now {
                        Val::Str(s) => chrono::DateTime::parse_from_rfc3339(s).map(|d| Val::Date(d.timestamp_millis())).unwrap_or(Val::Date(crate::date::now_millis())),
                        _ => Val::Date(crate::date::now_millis()),
                    };
                    m.insert(Key::str("now"), now);
                    Ok(Val::new_map(m))
                }),
                St::Def,
            ));
        }
        if name == "emit" && (n == 1 || n == 2) {
            return Ok((
                Box::new(move |rt| {
                    let v = fs[n - 1](rt)?;
                    if v.is_null() {
                        return Err(npe("Cannot emit a null value").at(pos));
                    }
                    if n == 2 {
                        let k = fs[0](rt)?;
                        let mut m = IndexMap::new();
                        m.insert(Key::from_val(&k), v);
                        rt.emitted.push(Val::new_map(m));
                    } else {
                        rt.emitted.push(v);
                    }
                    Ok(Val::Null)
                }),
                St::Def,
            ));
        }
        if name == "$" && n == 2 {
            // `$('field', default)`: the field's first value, or `default`
            // when the doc has none (Painless fields API shorthand).
            return Ok((
                Box::new(move |rt| {
                    let f = fs[0](rt)?;
                    let name = match &f {
                        Val::Str(s) => s.to_string(),
                        other => return Err(super::error::class_cast(format!("Cannot cast {} to java.lang.String", other.type_name())).at(pos)),
                    };
                    let dflt = fs[1](rt)?;
                    if !rt.doc.contains(&name) {
                        return Ok(dflt);
                    }
                    let slot = rt.doc.slot_of(&name);
                    let first = fields_api_values(rt, slot, &name).map_err(at(pos))?.into_iter().next();
                    Ok(first.unwrap_or(dflt))
                }),
                St::Def,
            ));
        }
        if name == "field" && n == 1 {
            // Fields API: `field('name')` is a per-document `Field` object.
            return Ok((
                Box::new(move |rt| {
                    let f = fs[0](rt)?;
                    let name = match &f {
                        Val::Str(s) => s.to_string(),
                        other => return Err(super::error::class_cast(format!("Cannot cast {} to java.lang.String", other.type_name())).at(pos)),
                    };
                    let slot = rt.doc.slot_of(&name);
                    let kind = rt.doc.fields[slot].kind;
                    let vals = if matches!(kind, super::doc::Kind::Unmapped) { Vec::new() } else { fields_api_values(rt, slot, &name).map_err(at(pos))? };
                    Ok(super::jmisc::new_field(name, kind, vals))
                }),
                St::Def,
            ));
        }
        if n == 2 && matches!(name, "maxSimDotProduct" | "maxSimInvHamming") {
            let fname = name.to_string();
            return Ok((
                Box::new(move |rt| {
                    let q = fs[0](rt)?;
                    let field = match fs[1](rt)? {
                        Val::Str(s) => s.to_string(),
                        other => return Err(super::error::class_cast(format!("Cannot cast {} to java.lang.String", other.type_name())).at(pos)),
                    };
                    let mut qvs: Vec<Vec<f64>> = Vec::new();
                    for v in iter_values(&q).map_err(at(pos))? {
                        // A hex-encoded byte vector.
                        if let Val::Str(h) = &v {
                            let h: &str = h;
                            if h.len() % 2 == 0 && h.bytes().all(|b| b.is_ascii_hexdigit()) {
                                qvs.push((0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap_or(0) as i8 as f64).collect());
                                continue;
                            }
                        }
                        let mut row = Vec::new();
                        for x in iter_values(&v).map_err(at(pos))? {
                            row.push(x.to_f64().map_err(at(pos))?);
                        }
                        qvs.push(row);
                    }
                    let slot = rt.doc.slot_of(&field);
                    let dv = rt.doc.values(slot).map_err(at(pos))?;
                    let first = match &dv {
                        Val::DocValues(d) => d.values.first().cloned(),
                        _ => None,
                    };
                    let Some(Val::Map(m)) = first else {
                        return Err(illegal_arg("A document doesn't have a value for a multi-vector field!").at(pos));
                    };
                    let (et, docs): (i32, Vec<Vec<f64>>) = {
                        let m = m.borrow();
                        let et = match m.get(&super::value::Key::str("element_type")) {
                            Some(Val::Int(i)) => *i,
                            _ => 0,
                        };
                        let mut docs = Vec::new();
                        if let Some(Val::List(l)) = m.get(&super::value::Key::str("vectors")) {
                            for a in l.borrow().iter() {
                                if let Val::Array { items, .. } = a {
                                    docs.push(items.borrow().iter().map(|x| x.to_f64().unwrap_or(0.0)).collect());
                                }
                            }
                        }
                        (et, docs)
                    };
                    let mut total = 0.0f64;
                    for qv in &qvs {
                        let mut best = f64::NEG_INFINITY;
                        for d in &docs {
                            let s = if fname == "maxSimInvHamming" {
                                if et == 0 {
                                    return Err(illegal_arg("hamming distance is only supported for byte or bit vectors").at(pos));
                                }
                                if qv.len() != d.len() {
                                    return Err(illegal_arg(format!("The query vector has a different number of dimensions [{}] than the document vectors [{}].", qv.len(), d.len())).at(pos));
                                }
                                let h: u32 = qv.iter().zip(d).map(|(a, b)| ((*a as i64 as u8) ^ (*b as i64 as u8)).count_ones()).sum();
                                let bits = (d.len() * 8) as f64;
                                ((bits - h as f64) / bits) as f32 as f64
                            } else if et == 2 && qv.len() == d.len() * 8 {
                                // Float query: the stored bits mask its components.
                                let mut acc = 0f32;
                                for (i, x) in qv.iter().enumerate() {
                                    let byte = d[i / 8] as i64 as u8;
                                    if byte & (0x80u8 >> (i % 8)) != 0 {
                                        acc += *x as f32;
                                    }
                                }
                                acc as f64
                            } else if et == 2 {
                                if qv.len() != d.len() {
                                    return Err(illegal_arg(format!("The query vector has a different number of dimensions [{}] than the document vectors [{}].", qv.len(), d.len() * 8)).at(pos));
                                }
                                qv.iter().zip(d).map(|(a, b)| ((*a as i64 as u8) & (*b as i64 as u8)).count_ones() as f64).sum()
                            } else {
                                if qv.len() != d.len() {
                                    return Err(illegal_arg(format!("The query vector has a different number of dimensions [{}] than the document vectors [{}].", qv.len(), d.len())).at(pos));
                                }
                                if et == 1 {
                                    qv.iter().zip(d).map(|(a, b)| (*a as i64) * (*b as i64)).sum::<i64>() as f64
                                } else {
                                    let q32: Vec<f32> = qv.iter().map(|x| *x as f32).collect();
                                    let d32: Vec<f32> = d.iter().map(|x| *x as f32).collect();
                                    rustic_core::vector::dot(&q32, &d32) as f64
                                }
                            };
                            best = best.max(s);
                        }
                        if best.is_finite() {
                            total += best;
                        }
                    }
                    Ok(Val::Double(total))
                }),
                St::Def,
            ));
        }
        if n == 2 && matches!(name, "cosineSimilarity" | "l2norm" | "l1norm" | "dotProduct" | "hamming" | "maxInnerProduct") {
            let fname = name.to_string();
            return Ok((
                Box::new(move |rt| {
                    let q = fs[0](rt)?;
                    let field = fs[1](rt)?;
                    let field = match &field {
                        Val::Str(s) => s.to_string(),
                        other => return Err(super::error::class_cast(format!("Cannot cast {} to java.lang.String", other.type_name())).at(pos)),
                    };
                    let mut qv: Vec<f64> = Vec::new();
                    for v in iter_values(&q).map_err(at(pos))? {
                        qv.push(v.to_f64().map_err(at(pos))?);
                    }
                    let dv = rt.doc.vector(&field).ok_or_else(|| {
                        illegal_arg(format!("A document doesn't have a value for a vector field!")).at(pos)
                    })?;
                    if dv.len() != qv.len() {
                        return Err(illegal_arg(format!(
                            "The query vector has a different number of dimensions [{}] than the document vectors [{}].",
                            qv.len(),
                            dv.len()
                        ))
                        .at(pos));
                    }
                    // Same f32 arithmetic as the kNN search (Lucene's
                    // VectorUtil), so script scores equal kNN scores.
                    use rustic_core::vector::{dot, l2_sq, norm};
                    let q32: Vec<f32> = qv.iter().map(|x| *x as f32).collect();
                    let d32: &[f32] = &dv;
                    let r = match fname.as_str() {
                        "cosineSimilarity" => {
                            let (qn, dn) = (norm(&q32), norm(d32));
                            if qn == 0.0 || dn == 0.0 {
                                return Err(illegal_arg("The vector has a magnitude of 0. Cannot compute cosine similarity.").at(pos));
                            }
                            (dot(&q32, d32) / (qn * dn)) as f64
                        }
                        "l2norm" => (l2_sq(&q32, d32) as f64).sqrt(),
                        "l1norm" => q32.iter().zip(d32.iter()).map(|(a, b)| (a - b).abs() as f64).sum(),
                        "hamming" => qv.iter().zip(dv.iter()).map(|(a, b)| ((*a as i64 as u8) ^ (*b as i64 as u8)).count_ones() as f64).sum(),
                        _ => dot(&q32, d32) as f64,
                    };
                    Ok(Val::Double(r))
                }),
                St::Def,
            ));
        }
        let _ = self.ctx;
        Err(ScriptErr::compile(pos, format!("Unknown call [{name}] with [{n}] arguments.")))
    }

    /// Where ES locates a literal argument: a negative number literal is a
    /// folded unary minus whose constant sits one past the `-`.
    fn lit_pos(&self, e: &Expr) -> usize {
        match &e.kind {
            ExprKind::Int(v) if *v < 0 => e.pos + 1,
            ExprKind::Long(v) if *v < 0 => e.pos + 1,
            ExprKind::Double(v) if *v < 0.0 => e.pos + 1,
            _ => e.pos,
        }
    }

    /// Static type of an expression when it is evident from the syntax alone.
    fn static_type(&self, e: &Expr) -> St {
        match &e.kind {
            ExprKind::Int(_) => St::Int,
            ExprKind::Long(_) => St::Long,
            ExprKind::Double(_) => St::Double,
            ExprKind::Bool(_) => St::Bool,
            ExprKind::Str(_) | ExprKind::Utf16(_) => St::Str,
            ExprKind::Null => St::Null,
            ExprKind::ListLit(_) => St::List,
            ExprKind::MapLit(_) => St::Map,
            ExprKind::Var(v) => self.lookup(v).map(|(_, t)| St::of_ty(&t)).unwrap_or(St::Def),
            _ => St::Def,
        }
    }

    fn new_obj(&mut self, ty: &str, args: &[Expr], new_pos: usize) -> CResult<(ExprFn, St)> {
        let pos = new_pos;
        let fs = self.exprs(args)?;
        let n = fs.len();
        if let Some(e) = wl_entry(ty) {
            if !e.2.contains(&(n as u8)) {
                return Err(ScriptErr::compile(pos, format!("constructor [{}, <init>/{n}] not found", e.0)));
            }
        }
        if matches!(ty, "BigDecimal" | "BigInteger") && n >= 1 {
            // `BigDecimal(String[, MathContext])`, `BigInteger(String[, int])`:
            // Painless resolves constructors by arity only.
            let params: [&str; 2] = if ty == "BigDecimal" { ["java.lang.String", "java.math.MathContext"] } else { ["java.lang.String", "int"] };
            for (i, a) in args.iter().enumerate().take(2) {
                let st = self.info(a).1;
                let bad = match (i, ty) {
                    (1, "BigInteger") => matches!(st, St::Double | St::Bool | St::Str | St::List | St::Map | St::Cls(_)),
                    _ => matches!(st, St::Int | St::Long | St::Double | St::Bool | St::List | St::Map | St::Cls(_)) || (i == 1 && st == St::Str),
                };
                if bad {
                    return Err(cc(self.lit_pos(a), format!("Cannot cast from [{}] to [{}].", jname_st(st), params[i])));
                }
            }
        }
        let rst = match ty {
            "BigDecimal" => St::Cls("java.math.BigDecimal"),
            "BigInteger" => St::Cls("java.math.BigInteger"),
            _ => St::Ref,
        };
        // ES records the `new`, then each argument: failures inside the
        // constructor are reported at the last of these.
        let pos = self.args_dbg(args).unwrap_or(new_pos);
        if matches!(ty, "TreeMap" | "TreeSet") && n == 1 {
            let st = self.static_type(&args[0]);
            if st != St::Def && st != St::Null {
                return Err(cc(self.lit_pos(&args[0]), format!("Cannot cast from [{}] to [java.util.Comparator].", jname_st(st))));
            }
        }
        let kind: u8 = match ty {
            "ArrayList" | "LinkedList" | "Vector" | "ArrayDeque" | "Stack" | "PriorityQueue" => 0,
            "HashMap" | "LinkedHashMap" | "TreeMap" => 1,
            "HashSet" | "LinkedHashSet" | "TreeSet" => 2,
            "StringBuilder" | "StringBuffer" => 3,
            "String" => 4,
            // java.util.Date: `new Date()` is now, `new Date(long)` an epoch-millis instant.
            "Date" if n <= 1 => 6,
            t if EXCEPTIONS.contains(&t) => 5,
            t if super::jbig::CLASSES.contains(&t) || super::jtime::CLASSES.contains(&t) || super::jmisc::CLASSES.contains(&t) => 7,
            other => return Err(ScriptErr::compile(new_pos, format!("cannot resolve symbol [{other}]"))),
        };
        let class = ty.to_string();
        Ok((
            Box::new(move |rt| {
                let a: Vec<Val> = fs.iter().map(|f| f(rt)).collect::<SResult<_>>()?;
                Ok(match kind {
                    0 => match a.first() {
                        Some(v) if !v.is_number() => {
                            let allocation = super::allocation::Allocation::iterable(v)?;
                            Val::new_list(iter_values(v).map_err(at(pos))?).allocated(allocation)
                        },
                        _ => Val::new_list(Vec::new()).allocated(super::allocation::Allocation::values(0).map_err(at(pos))?),
                    },
                    1 => super::methods::new_map_of(rt, &class, a.first()).map_err(at(pos))?,
                    2 => super::methods::new_set_of(rt, &class, a.first()).map_err(at(pos))?,
                    3 => {
                        if a.first().is_some_and(Val::is_null) {return Err(at(pos)(super::error::npe("sequence is null")));}
                        let _copy=a.first().map(super::allocation::Allocation::iterable).transpose().map_err(at(pos))?;
                        Val::Builder(Rc::new(RefCell::new(super::value::JBuilder::new(a.first().filter(|v| !v.is_number()).map(Val::string_units).unwrap_or_default(),!a.is_empty()).map_err(at(pos))?)))
                    },
                    4 => Val::string(a.first().map(Val::to_java_string).unwrap_or_default()),
                    6 => Val::Date(match a.first() {
                        Some(v) => v.to_i64().map_err(at(pos))?,
                        None => crate::date::now_millis(),
                    }),
                    7 => {
                        let c = class.as_str();
                        let r = if super::jbig::CLASSES.contains(&c) {
                            super::jbig::construct(&class, &a)
                        } else if super::jtime::CLASSES.contains(&c) {
                            super::jtime::construct(&class, &a)
                        } else {
                            super::jmisc::construct(&class, &a)
                        };
                        match r {
                            Some(r) => r.map_err(at(pos))?,
                            None => return Err(illegal_arg(format!("constructor [{class}, <init>/{n}] not found")).at(pos)),
                        }
                    }
                    _ => Val::Exception(Rc::new((class.clone(), if n > 0 { a[0].to_java_string() } else { String::new() }))),
                })
            }),
            rst,
        ))
    }

    fn lambda(&mut self, params: &[String], body: &LambdaBody, _pos: usize) -> CResult<u32> {
        let id = self.lambdas.len();
        self.lambdas.push(None);
        self.scopes.push(Vec::new());
        let mut slots = Vec::new();
        for p in params {
            // Lambda parameters may shadow nothing; reuse names across lambdas.
            let slot = self.next_slot;
            self.next_slot += 1;
            self.scopes.last_mut().unwrap().push((p.clone(), slot, Ty::Def));
            slots.push(slot);
        }
        let saved_loop = std::mem::replace(&mut self.loop_depth, 0);
        let body: ExprFn = match body {
            LambdaBody::Expr(e) => self.expr(e)?.0,
            LambdaBody::Block(b) => {
                let f = self.block(b, true)?;
                Box::new(move |rt| match f(rt)? {
                    Flow::Return(v) => Ok(v),
                    _ => Ok(Val::Null),
                })
            }
        };
        self.loop_depth = saved_loop;
        self.scopes.pop();
        self.lambdas[id] = Some(LambdaInfo { slots, body });
        Ok(id as u32)
    }

    fn assign(&mut self, target: &Expr, op: Option<BinOp>, value: &Expr, pos: usize) -> CResult<(ExprFn, St)> {
        let (vf, vst) = self.expr(value)?;
        match &target.kind {
            ExprKind::Var(name) => {
                let Some((slot, ty)) = self.lookup(name) else {
                    return Err(ScriptErr::compile(target.pos, format!("cannot resolve symbol [{name}]")));
                };
                if op.is_none() {
                    check_assign(&ty, vst, value.pos)?;
                }
                let st = St::of_ty(&ty);
                // `x = <numeric expr>` into a primitive local: store the
                // unboxed result (the conversions `coerce` applies: int/long
                // widen to double, int widens to long); `None` = general node.
                let typed = if op.is_none() && matches!(ty, Ty::Int | Ty::Long | Ty::Double) { self.num(value) } else { None };
                let target_ty = ty.clone();
                let general: ExprFn = Box::new(move |rt| {
                    let v = vf(rt)?;
                    let nv = match op {
                        None => coerce(&ty, v).map_err(at(pos))?,
                        Some(op) => {
                            let r = binary(op, rt.local(slot), &v).map_err(at(pos))?;
                            if matches!(ty, Ty::Def | Ty::Ref(_)) { r } else { cast(&ty, r).map_err(at(pos))? }
                        }
                    };
                    rt.set_local(slot, nv.fast_clone());
                    Ok(nv)
                });
                Ok(match (target_ty, typed) {
                    (Ty::Double, Some(nf)) => {
                        let d = nf.into_d();
                        (
                            Box::new(move |rt| match d(rt) {
                                Some(x) => {
                                    rt.set_local(slot, Val::Double(x));
                                    Ok(Val::Double(x))
                                }
                                None => general(rt),
                            }),
                            st,
                        )
                    }
                    (Ty::Long, Some(nf @ (num::NumFn::I(_) | num::NumFn::L(_)))) => match nf.into_l() {
                        Some(l) => (
                            Box::new(move |rt| match l(rt) {
                                Some(x) => {
                                    rt.set_local(slot, Val::Long(x));
                                    Ok(Val::Long(x))
                                }
                                None => general(rt),
                            }),
                            st,
                        ),
                        None => (general, st),
                    },
                    (Ty::Int, Some(num::NumFn::I(i))) => (
                        Box::new(move |rt| match i(rt) {
                            Some(x) => {
                                rt.set_local(slot, Val::Int(x));
                                Ok(Val::Int(x))
                            }
                            None => general(rt),
                        }),
                        st,
                    ),
                    _ => (general, st),
                })
            }
            ExprKind::Field { obj, name, .. } => {
                if self.static_class(obj).is_some() || self.doc_slot(obj).is_some() || self.is_global(obj, "doc") {
                    return Err(ScriptErr::compile(pos, format!("invalid assignment: cannot assign a value to [{name}]")));
                }
                let (o, _) = self.expr(obj)?;
                let name = name.to_string();
                Ok((
                    Box::new(move |rt| {
                        let ov = o(rt)?;
                        let v = vf(rt)?;
                        let nv = match op {
                            None => v,
                            Some(op) => {
                                let old = field_get(rt, &ov, &name).map_err(at(pos))?;
                                binary(op, &old, &v).map_err(at(pos))?
                            }
                        };
                        field_set(&ov, &name, nv.clone()).map_err(at(pos))?;
                        Ok(nv)
                    }),
                    St::Def,
                ))
            }
            ExprKind::Index { obj, idx } => {
                if self.is_global(obj, "doc") {
                    return Err(ScriptErr::compile(pos, "invalid assignment: doc values are read-only"));
                }
                let (o, _) = self.expr(obj)?;
                let (i, _) = self.expr(idx)?;
                Ok((
                    Box::new(move |rt| {
                        let ov = o(rt)?;
                        let iv = i(rt)?;
                        let v = vf(rt)?;
                        let nv = match op {
                            None => v,
                            Some(op) => {
                                let old = index_get(rt, &ov, &iv).map_err(at(pos))?;
                                binary(op, &old, &v).map_err(at(pos))?
                            }
                        };
                        index_set(&ov, &iv, nv.clone()).map_err(at(pos))?;
                        Ok(nv)
                    }),
                    St::Def,
                ))
            }
            _ => Err(ScriptErr::compile(pos, "invalid assignment: cannot assign a value to this expression")),
        }
    }
}

/// `Math` methods of one double argument returning a double (exactly
/// what [`math`] computes for them), resolved at compile time.
fn math_unary(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        "abs" => f64::abs,
        "sqrt" => f64::sqrt,
        "cbrt" => f64::cbrt,
        "log" => f64::ln,
        "log10" => f64::log10,
        "log1p" => f64::ln_1p,
        "exp" => f64::exp,
        "expm1" => f64::exp_m1,
        "floor" => f64::floor,
        "ceil" => f64::ceil,
        "sin" => f64::sin,
        "cos" => f64::cos,
        "tan" => f64::tan,
        "asin" => f64::asin,
        "acos" => f64::acos,
        "atan" => f64::atan,
        "sinh" => f64::sinh,
        "cosh" => f64::cosh,
        "tanh" => f64::tanh,
        "toRadians" => f64::to_radians,
        "toDegrees" => f64::to_degrees,
        _ => return None,
    })
}

/// A doc-value read compiled to a direct column access.
enum DocFast {
    /// The first value of a single-valued numeric field (`.get(0)`, `[0]`).
    First(usize),
    /// `lat` (true) / `lon` of a single-valued `geo_point`.
    Geo(usize, bool),
}

/// A leaf operand evaluated inline by its parent node.
enum Leaf {
    Int(i32),
    Long(i64),
    Double(f64),
    Local(usize),
    Score,
    Param(usize),
    /// `doc['x'].value` (slot, source position).
    Doc(usize, usize),
}

enum Operand {
    Leaf(Leaf),
    Expr(ExprFn),
}

impl Operand {
    #[inline(always)]
    fn eval(&self, rt: &mut Rt) -> SResult<Val> {
        match self {
            Operand::Expr(f) => f(rt),
            Operand::Leaf(l) => match l {
                Leaf::Doc(slot, pos) => rt.doc.value(*slot).map_err(at(*pos)),
                Leaf::Param(id) => Ok(rt.param_vals[*id].fast_clone()),
                Leaf::Score => Ok(Val::Double(rt.score)),
                Leaf::Double(d) => Ok(Val::Double(*d)),
                Leaf::Int(i) => Ok(Val::Int(*i)),
                Leaf::Long(v) => Ok(Val::Long(*v)),
                Leaf::Local(s) => Ok(rt.local(*s).fast_clone()),
            },
        }
    }
}

/// Binary op on owned operands: numeric fast paths skip the drop glue.
#[inline(always)]
fn fast_num(op: BinOp, x: Val, y: Val) -> SResult<Val> {
    let r = match (op, &x, &y) {
        (BinOp::Add, Val::Double(p), Val::Double(q)) => Val::Double(p + q),
        (BinOp::Sub, Val::Double(p), Val::Double(q)) => Val::Double(p - q),
        (BinOp::Mul, Val::Double(p), Val::Double(q)) => Val::Double(p * q),
        (BinOp::Div, Val::Double(p), Val::Double(q)) => Val::Double(p / q),
        (BinOp::Add, Val::Double(p), Val::Long(q)) | (BinOp::Add, Val::Long(q), Val::Double(p)) => Val::Double(p + *q as f64),
        (BinOp::Mul, Val::Double(p), Val::Long(q)) | (BinOp::Mul, Val::Long(q), Val::Double(p)) => Val::Double(p * *q as f64),
        (BinOp::Add, Val::Double(p), Val::Int(q)) | (BinOp::Add, Val::Int(q), Val::Double(p)) => Val::Double(p + *q as f64),
        (BinOp::Mul, Val::Double(p), Val::Int(q)) | (BinOp::Mul, Val::Int(q), Val::Double(p)) => Val::Double(p * *q as f64),
        (BinOp::Add, Val::Long(p), Val::Long(q)) => Val::Long(p.wrapping_add(*q)),
        (BinOp::Mul, Val::Long(p), Val::Long(q)) => Val::Long(p.wrapping_mul(*q)),
        (BinOp::Lt, Val::Double(p), Val::Double(q)) => Val::Bool(p < q),
        (BinOp::Gt, Val::Double(p), Val::Double(q)) => Val::Bool(p > q),
        _ => {
            let r = binary(op, &x, &y);
            return r;
        }
    };
    // Inline variants own nothing: skip the out-of-line drop glue.
    std::mem::forget(x);
    std::mem::forget(y);
    Ok(r)
}

fn check_assign(ty: &Ty, st: St, pos: usize) -> CResult<()> {
    if st == St::Null && matches!(ty, Ty::Int | Ty::Short | Ty::Byte | Ty::Long | Ty::Double | Ty::Float | Ty::Boolean | Ty::Char) {
        return Err(ScriptErr::compile(pos, format!("Cannot cast null to a primitive type [{}].", ty.name())));
    }
    let bad = match ty {
        Ty::Int | Ty::Short | Ty::Byte => matches!(st, St::Long | St::Double | St::Bool | St::Str),
        Ty::Long => matches!(st, St::Double | St::Bool | St::Str),
        Ty::Double | Ty::Float => matches!(st, St::Bool | St::Str),
        Ty::Boolean => matches!(st, St::Int | St::Long | St::Double | St::Str),
        Ty::String => matches!(st, St::Int | St::Long | St::Double | St::Bool),
        // Boxed / reference targets accept no primitive (only `def` boxes implicitly).
        Ty::Ref(_) => matches!(st, St::Int | St::Long | St::Double | St::Bool),
        _ => false,
    };
    if bad {
        return Err(cc(pos, format!("Cannot cast from [{}] to [{}].", jname_st(st), fqn_ty(ty))));
    }
    Ok(())
}

fn jname_st(st: St) -> &'static str {
    match st {
        St::Str => "java.lang.String",
        s => s.name(),
    }
}

/// Methods of typed receivers that Painless whitelists as returning `def`.
const DEF_RETURNING: &[&str] = &[
    "get", "remove", "put", "getOrDefault", "putIfAbsent", "computeIfAbsent", "computeIfPresent", "compute", "merge", "poll", "pollFirst", "pollLast",
    "peek", "peekFirst", "peekLast", "pop", "first", "last", "getFirst", "getLast", "firstKey", "lastKey", "floor", "ceiling", "higher", "lower",
    "floorKey", "ceilingKey", "higherKey", "lowerKey", "next", "orElse", "orElseGet", "orElseThrow", "getValue", "getKey", "setValue", "element",
    "removeFirst", "removeLast", "set", "min", "max", "reduce", "find", "findAll", "collect", "iterator", "stream", "apply", "accept", "test",
];

/// Source location ES gives an expression (a binary operation sits at its
/// first operand).
fn loc(e: &Expr) -> usize {
    match &e.kind {
        ExprKind::Binary(_, a, _) => loc(a),
        _ => e.pos,
    }
}

/// A compile error whose cause is a `ClassCastException`.
fn cc(pos: usize, msg: String) -> ScriptErr {
    let mut e = ScriptErr::compile(pos, msg);
    e.kind = "class_cast_exception".into();
    e
}

/// Whitelist entry of a class by simple name.
fn wl_entry(simple: &str) -> Option<&'static (&'static str, &'static [&'static str], &'static [u8], &'static [&'static str])> {
    let suffix = format!(".{simple}");
    // An inner class is written `Outer.Inner` (`ScriptDocValues.Geometry`)
    // and whitelisted as `Outer$Inner`.
    let inner = simple.contains('.').then(|| format!(".{}", simple.replace('.', "$")));
    super::whitelist::WHITELIST.iter().find(|e| e.0.ends_with(&suffix) || inner.as_ref().is_some_and(|i| e.0.ends_with(i.as_str())))
}

fn fqn_ty(t: &Ty) -> String {
    match t {
        Ty::String => "java.lang.String".into(),
        Ty::Ref(n) => wl_entry(n).map(|e| e.0.to_string()).unwrap_or_else(|| n.clone()),
        other => other.name(),
    }
}

fn default_of(t: &Ty) -> Val {
    match t {
        Ty::Int | Ty::Short | Ty::Byte => Val::Int(0),
        Ty::Long => Val::Long(0),
        Ty::Double | Ty::Float => Val::Double(0.0),
        Ty::Boolean => Val::Bool(false),
        Ty::Char => Val::Char(0),
        _ => Val::Null,
    }
}

/// Constant closure for a static-field value (numbers / enums only).
fn const_fn(v: &Val) -> (ExprFn, St) {
    match v {
        Val::Int(i) => {
            let i = *i;
            (Box::new(move |_| Ok(Val::Int(i))), St::Int)
        }
        Val::Long(l) => {
            let l = *l;
            (Box::new(move |_| Ok(Val::Long(l))), St::Long)
        }
        Val::Double(d) => {
            let d = *d;
            (Box::new(move |_| Ok(Val::Double(d))), St::Double)
        }
        Val::Bool(b) => {
            let b = *b;
            (Box::new(move |_| Ok(Val::Bool(b))), St::Bool)
        }
        Val::DayOfWeek(d) => {
            let d = *d;
            (Box::new(move |_| Ok(Val::DayOfWeek(d))), St::Def)
        }
        Val::Month(m) => {
            let m = *m;
            (Box::new(move |_| Ok(Val::Month(m))), St::Def)
        }
        Val::ChronoUnit(u) => {
            let u = *u;
            (Box::new(move |_| Ok(Val::ChronoUnit(u))), St::Def)
        }
        Val::Str(s) => {
            let s = s.to_string();
            (Box::new(move |_| Ok(Val::str(&s))), St::Str)
        }
        Val::List(_) => (Box::new(|_| Ok(Val::new_list(Vec::new()).allocated(super::allocation::Allocation::values(0)?))), St::Def),
        Val::Map(_) => (Box::new(|_| Ok(Val::new_map(IndexMap::new()).allocated(super::allocation::Allocation::entries(0)?))), St::Def),
        _ => (Box::new(|_| Ok(Val::Null)), St::Def),
    }
}

/// `illegal_argument_exception` → `IllegalArgumentException`.
pub fn class_of_kind(kind: &str) -> String {
    kind.split('_').map(|p| {
        let mut c = p.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    }).collect()
}

/// Compile `src` for `ctx` (no caching; see [`super::compile`]).
pub fn compile_arc(src: &str, ctx: ScriptContext) -> Result<Arc<Program>, ScriptErr> {
    compile_program(src, ctx).map(Arc::new)
}

/// Values of a field for the fields API (`field(..)`, `$(..)`): doc values,
/// or `_source` for `text` fields (ES reads those through a source fetcher).
fn fields_api_values(rt: &mut Rt, slot: usize, name: &str) -> SResult<Vec<Val>> {
    // Fields without doc values that the Fields API reads from `_source`
    // (e.g. `unsigned_long` with `doc_values: false`).
    if let Some(v) = rt.doc.field_api_values(slot)? {
        return Ok(v);
    }
    if rt.doc.fields[slot].kind == super::doc::Kind::Text {
        let mut cur = vec![rt.source()?];
        for part in name.split('.') {
            let mut next = Vec::new();
            for v in cur {
                match &v {
                    Val::Map(m) => {
                        if let Some(x) = m.borrow().get(&super::value::Key::str(part)) {
                            next.push(x.clone());
                        }
                    }
                    Val::List(l) => {
                        for x in l.borrow().iter() {
                            if let Val::Map(m) = x {
                                if let Some(y) = m.borrow().get(&super::value::Key::str(part)) {
                                    next.push(y.clone());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            cur = next;
        }
        let mut out = Vec::new();
        for v in cur {
            match v {
                Val::List(l) => out.extend(l.borrow().iter().filter(|x| !matches!(x, Val::Null)).cloned()),
                Val::Null => {}
                other => out.push(other),
            }
        }
        return Ok(out);
    }
    let vals = rt.doc.values(slot)?;
    iter_values(&vals)
}
