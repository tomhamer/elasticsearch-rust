//! Execution state and operator semantics of compiled scripts.

use super::ast::{BinOp, Ty, UnOp};
use super::doc::DocCtx;
use super::error::{class_cast, illegal_arg, npe, out_of_bounds, SResult, ScriptErr};
use super::value::{CollKind, Key, MapKind, Val, CHRONO_UNITS};
use super::ScriptContext;
use crate::mapping::{FieldLookup, SOURCE_FIELD};
use indexmap::IndexMap;
use rustic_core::query::LeafReader;
use rustic_core::types::DocId;
use std::rc::Rc;
use std::sync::Arc;

pub type ExprFn = Box<dyn Fn(&mut Rt) -> SResult<Val> + Send + Sync>;
pub type StmtFn = Box<dyn Fn(&mut Rt) -> SResult<Flow> + Send + Sync>;

pub enum Flow {
    Next,
    Break,
    Continue,
    Return(Val),
}

pub struct FnInfo {
    pub name: String,
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub nslots: usize,
    pub body: StmtFn,
}

pub struct LambdaInfo {
    pub slots: Vec<usize>,
    pub body: ExprFn,
}

/// A compiled script: closure tree plus the tables executors bind.
pub struct Program {
    pub source: String,
    pub stmt_starts: Vec<usize>,
    pub ctx: ScriptContext,
    pub body: StmtFn,
    /// Single-expression scripts: the expression itself (skips the
    /// statement layer per execution).
    pub expr_body: Option<ExprFn>,
    /// Unboxed numeric / comparison form of `expr_body` (see
    /// `compile::num`); `None` from a call = evaluate `expr_body`.
    pub num_body: Option<Box<dyn Fn(&mut Rt) -> Option<f64> + Send + Sync>>,
    pub cond_body: Option<Box<dyn Fn(&mut Rt) -> Option<bool> + Send + Sync>>,
    /// The script reads `_score` (callers may skip computing it otherwise).
    pub uses_score: bool,
    /// Source offset of the script's final expression (position of a
    /// result-cast failure, e.g. a non-boolean `if` condition).
    pub result_pos: usize,
    pub nslots: usize,
    pub functions: Vec<FnInfo>,
    pub lambdas: Vec<LambdaInfo>,
    /// String constants (converted once per executor).
    pub strings: Vec<String>,
    /// `params.x` names with constant keys.
    pub param_names: Vec<String>,
    /// `doc['x']` names with constant keys.
    pub doc_fields: Vec<String>,
    /// Number of class-binding state slots ([`Rt::bindings`]).
    pub nbindings: usize,
    /// `lang: expression` programs: their variables (linked per executor).
    pub expression: Option<Arc<super::expression::ExprMeta>>,
    /// The parts of `ctx` the script can observe ([`super::ctx_access`];
    /// `None`: all of it).
    pub ctx_paths: Option<Arc<super::ctx_access::CtxPaths>>,
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Program({:?})", self.source)
    }
}

/// Maximum loop iterations per execution (ES `script.painless.max_loop_counter`).
pub const MAX_LOOP: u64 = 1_000_000;

/// `ScriptErr::kind` of a script stopped by its search's deadline (ES
/// `ContextIndexSearcher.TimeExceededException`): the search reports
/// `timed_out` instead of failing.
pub const TIME_EXCEEDED: &str = "time_exceeded";

/// The deadline of a search (`timeout`) shared with the scripts it runs.
#[derive(Debug)]
pub struct Cancel {
    pub deadline: std::time::Instant,
    pub timed_out: std::sync::atomic::AtomicBool,
    /// The search task's cancellation: a cancelled task stops its scripts
    /// at the same amortised poll (the search then reports the
    /// cancellation, not a timeout).
    pub task: Option<Arc<rustic_core::cancel::Token>>,
}

impl Cancel {
    pub fn new(timeout: std::time::Duration) -> Arc<Cancel> {
        Arc::new(Cancel { deadline: std::time::Instant::now() + timeout, timed_out: std::sync::atomic::AtomicBool::new(false), task: rustic_core::cancel::current() })
    }

    pub fn expired(&self) -> bool {
        std::time::Instant::now() >= self.deadline || self.task.as_ref().is_some_and(|t| t.is_cancelled())
    }

    /// Whether work under this deadline should stop: a script already
    /// timed out (or its task was cancelled). One relaxed load (two with a
    /// task); for per-doc checks of script-valued work, not loops.
    #[inline]
    pub fn stopped(&self) -> bool {
        self.timed_out.load(std::sync::atomic::Ordering::Relaxed) || self.task.as_ref().is_some_and(|t| t.is_cancelled())
    }

    /// Record a script stopped by the deadline; `true` when `e` was one.
    pub fn absorb(this: &Option<Arc<Cancel>>, e: &ScriptErr) -> bool {
        match this {
            Some(c) if e.kind == TIME_EXCEEDED => {
                c.timed_out.store(true, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }
}

/// Per-executor runtime state.
pub struct Rt {
    pub locals: Vec<Val>,
    pub base: usize,
    pub prog: Arc<Program>,
    pub consts: Vec<Val>,
    pub params: Val,
    pub param_vals: Vec<Val>,
    pub score: f64,
    /// `_value` (aggregation value scripts).
    pub value: Val,
    pub doc: DocCtx,
    /// `ctx` (update / ingest).
    pub ctx: Val,
    /// `params._source` of the current doc (lazily loaded).
    pub source: Option<(DocId, Val)>,
    /// Explicit `_source` (painless_execute, update contexts).
    pub source_override: Option<Val>,
    pub emitted: Vec<Val>,
    pub loops: u64,
    /// Search-time cancellation check (`PainlessScript._getCancellationCheck`):
    /// when set, loops poll the request deadline instead of counting
    /// against `max_loop_counter`.
    pub cancel: Option<Arc<Cancel>>,
    /// The search task this script runs for (the thread's search token when
    /// the executor was made): its loops poll it every 1024 iterations,
    /// with or without a deadline, so a cancelled search stops its scripts.
    pub task: Option<Arc<rustic_core::cancel::Token>>,
    /// Address of a local at the start of the current native stack segment
    /// (set by `run`, reset when a deep call continues on a big stack).
    pub stack_base: usize,
    pub on_big_stack: bool,
    /// State of Painless class bindings (e.g. `decayGeoGauss`), created on
    /// a call site's first invocation from its constant arguments.
    pub bindings: Vec<Option<[f64; 4]>>,
    /// `lang: expression`: how each variable is read (filled on first run).
    pub expr_binds: Vec<super::expression::Bind>,
    pub expr_linked: bool,
}

/// Native stack a script may use before its calls continue on a dedicated
/// big-stack thread, and the budget (per thread) before `stack_overflow_error`.
const SOFT_STACK: usize = 512 * 1024;
const BIG_STACK: usize = 96 * 1024 * 1024;
const HARD_STACK: usize = 24 * 1024 * 1024;

#[inline(always)]
fn stack_pos() -> usize {
    let x = 0u8;
    std::hint::black_box(&x) as *const u8 as usize
}

/// Carries a value across the scoped big-stack thread. The spawning thread
/// blocks until the closure returns, so the (non-`Send`) values are only
/// ever touched by one thread at a time.
struct Unsend<T>(T);
unsafe impl<T> Send for Unsend<T> {}

impl Rt {
    #[inline]
    pub fn local(&self, slot: usize) -> &Val {
        &self.locals[self.base + slot]
    }
    #[inline]
    pub fn set_local(&mut self, slot: usize, v: Val) {
        let i = self.base + slot;
        self.locals[i] = v;
    }

    #[inline]
    pub fn tick(&mut self) -> SResult<()> {
        self.loops += 1;
        // `WriterConstants.CANCELLATION_POLL_INTERVAL`-style amortised poll
        // on the loop counter: the deadline (which covers the task) or, with
        // none, the search task's cancellation.
        if self.loops % 1024 == 0 {
            let stop = match &self.cancel {
                Some(c) => c.expired(),
                None => self.task.as_ref().is_some_and(|t| t.is_cancelled()),
            };
            if stop {
                return Err(ScriptErr::new(TIME_EXCEEDED, "Time exceeded"));
            }
        }
        if self.cancel.is_some() {
            return Ok(());
        }
        if self.loops >= MAX_LOOP {
            return Err(ScriptErr::new("painless_error", "The maximum number of statements that can be executed in a loop has been reached."));
        }
        Ok(())
    }

    /// `params._source` for the current doc.
    pub fn source(&mut self) -> SResult<Val> {
        if let Some(s) = &self.source_override {
            return Ok(s.clone());
        }
        let doc = self.doc.doc;
        if let Some((d, v)) = &self.source {
            if *d == doc {
                return Ok(v.clone());
            }
        }
        let Some(leaf) = self.doc.leaf else { return Ok(Val::Null) };
        self.source=None;
        super::allocation::Scope::collect_dead();
        let decode=super::allocation::Allocation::new();
        let mut retained=0u64;
        let stored=if decode.enabled() {
            leaf.get_doc_guarded(doc,|bytes| {
                decode.reserve(bytes)?;
                retained=retained.max(bytes);
                Ok::<_,Box<ScriptErr>>(())
            })?
        } else { leaf.get_doc(doc).map_err(|e| super::error::ScriptErr::breaker(crate::error::EsError::from(e)))? };
        if let Some(lookup)=&self.doc.lookup {
            let vector_bytes=lookup.fields.values().filter(|f|f.declared==crate::mapping::EsType::DenseVector).fold(0u64,|n,f|n.saturating_add((f.dims.unwrap_or(4096) as u64).saturating_mul(256)));
            decode.reserve(retained.saturating_add(vector_bytes))?;
        }
        let v = stored
            .and_then(|d| {
                d.get(SOURCE_FIELD).and_then(|v| v.as_text()).and_then(|s| match &self.doc.lookup {
                    // With the vectors the stored `_source` leaves to the vector store.
                    Some(lk) => crate::source_json::Source::parse(&crate::vector_source::load(s, lk, &leaf, doc)).ok(),
                    None => crate::source_json::Source::parse(crate::synthetic::residual_json(s)).ok(),
                })
            })
            .map(|j| super::source::from_source(&j))
            .unwrap_or(Val::Null);
        super::allocation::Scope::adopt(&v)?;
        self.source = Some((doc, v.clone()));
        Ok(v)
    }

    /// Call a lambda / method reference with `args`.
    pub fn call_fn(&mut self, f: &Val, args: &[Val]) -> SResult<Val> {
        match f {
            Val::Lambda(id) => {
                // SAFETY: as in `Executor::run` (`prog` is never replaced).
                let prog: &Program = unsafe { &*Arc::as_ptr(&self.prog) };
                let l = &prog.lambdas[*id as usize];
                if l.slots.len() != args.len() {
                    return Err(illegal_arg(format!("lambda expects {} arguments, got {}", l.slots.len(), args.len())));
                }
                for (s, a) in l.slots.iter().zip(args) {
                    self.set_local(*s, a.clone());
                }
                let saved_loops = std::mem::replace(&mut self.loops, 0);
                let r = (l.body)(self);
                self.loops = saved_loops;
                r
            }
            Val::MethodRef(r) => {
                let (class, method) = (&r.0, &r.1);
                if class == "this" {
                    return self.call_user_fn(method, args.to_vec());
                }
                // Static method of the class, else instance method on the first argument.
                if let Some(v) = call_static(self, class, method, args.to_vec())? {
                    return Ok(v);
                }
                let mut a = args.to_vec();
                if a.is_empty() {
                    return Err(illegal_arg(format!("cannot apply method reference [{class}::{method}]")));
                }
                let recv = a.remove(0);
                super::methods::call_method(self, recv, method, a)
            }
            Val::Obj(o) if matches!(**o, super::jmisc::Obj::Cmp(..)) => {
                if args.len() != 2 {
                    return Err(illegal_arg(format!("comparator expects 2 arguments, got {}", args.len())));
                }
                Ok(Val::Int(o.cmp_call(self, &args[0], &args[1])?.clamp(i32::MIN as i64, i32::MAX as i64) as i32))
            }
            Val::Null => Err(npe("cannot invoke a null function")),
            other => Err(class_cast(format!("Cannot cast {} to a functional interface", other.type_name()))),
        }
    }

    pub fn call_user_fn(&mut self, name: &str, args: Vec<Val>) -> SResult<Val> {
        let prog = self.prog.clone();
        let Some(idx) = prog.functions.iter().position(|f| f.name == name && f.params.len() == args.len()) else {
            return Err(illegal_arg(format!("Unknown call [{name}] with [{}] arguments.", args.len())));
        };
        self.invoke_fn(idx, args)
    }

    pub fn invoke_fn(&mut self, idx: usize, args: Vec<Val>) -> SResult<Val> {
        let used = self.stack_base.wrapping_sub(stack_pos());
        if used > SOFT_STACK && used < (1 << 40) {
            if !self.on_big_stack {
                let me = Unsend(self as *mut Rt);
                let args = Unsend(args);
                let allocations = Unsend(super::allocation::Scope::current());
                let r = std::thread::scope(|sc| {
                    std::thread::Builder::new()
                        .stack_size(BIG_STACK)
                        .spawn_scoped(sc, move || {
                            let allocations = allocations;
                            let _allocations = super::allocation::Scope::install(allocations.0);
                            let (me, args) = (me, args);
                            // SAFETY: the parent thread is blocked in `scope` until we return.
                            let rt: &mut Rt = unsafe { &mut *me.0 };
                            let old = rt.stack_base;
                            rt.stack_base = stack_pos();
                            rt.on_big_stack = true;
                            let r = rt.invoke_fn(idx, args.0);
                            rt.on_big_stack = false;
                            rt.stack_base = old;
                            Unsend(r)
                        })
                        .map(|h| h.join())
                });
                return match r {
                    Ok(Ok(Unsend(r))) => r,
                    _ => Err(ScriptErr::new("stack_overflow_error", "")),
                };
            }
            if used > HARD_STACK {
                return Err(ScriptErr::new("stack_overflow_error", ""));
            }
        }
        let prog = self.prog.clone();
        let f = &prog.functions[idx];
        let saved = self.base;
        let new_base = self.locals.len();
        self.locals.resize(new_base + f.nslots.max(args.len()), Val::Null);
        self.base = new_base;
        for (i, (a, t)) in args.into_iter().zip(&f.params).enumerate() {
            let v = coerce(t, a);
            match v {
                Ok(v) => self.locals[new_base + i] = v,
                Err(e) => {
                    self.locals.truncate(new_base);
                    self.base = saved;
                    return Err(e);
                }
            }
        }
        let saved_loops = std::mem::replace(&mut self.loops, 0);
        let r = (f.body)(self);
        self.loops = saved_loops;
        self.locals.truncate(new_base);
        self.base = saved;
        match r? {
            Flow::Return(v) => {
                if f.ret == Ty::Void {
                    Ok(Val::Null)
                } else {
                    coerce(&f.ret, v)
                }
            }
            _ => Ok(Val::Null),
        }
    }
}

/// A compiled script bound to its parameters, plus per-doc state. One per
/// thread / leaf; cheap to reuse across docs (no allocation per doc for
/// numeric expressions).
pub struct Executor {
    pub rt: Rt,
}

// SAFETY: an executor's reference-counted values are created by and only
// ever reachable from the executor itself (values leave it as JSON or plain
// numbers), so moving the whole executor to another thread is sound. It is
// never shared (`!Sync`).
unsafe impl Send for Executor {}

impl Executor {
    pub fn new(prog: Arc<Program>, params: &serde_json::Value, lookup: Option<Arc<FieldLookup>>, runtime: Option<Arc<super::RuntimeFields>>) -> Executor {
        let consts = prog.strings.iter().map(|s| Val::str(s)).collect();
        let params_val = match params {
            serde_json::Value::Object(_) => Val::from_json(params),
            _ => Val::new_map(IndexMap::new()),
        };
        let param_vals = prog
            .param_names
            .iter()
            .map(|n| match &params_val {
                Val::Map(m) => m.borrow().get(&Key::str(n)).cloned().unwrap_or(Val::Null),
                _ => Val::Null,
            })
            .collect();
        let doc = DocCtx::new(&prog.doc_fields, lookup, runtime);
        let nslots = prog.nslots;
        let nbindings = prog.nbindings;
        Executor {
            rt: Rt {
                locals: vec![Val::Null; nslots],
                base: 0,
                prog,
                consts,
                params: params_val,
                param_vals,
                score: 0.0,
                value: Val::Null,
                doc,
                ctx: Val::Null,
                source: None,
                source_override: None,
                emitted: Vec::new(),
                loops: 0,
                cancel: None,
                task: rustic_core::cancel::current(),
                stack_base: 0,
                on_big_stack: false,
                bindings: vec![None; nbindings],
                expr_binds: Vec::new(),
                expr_linked: false,
            },
        }
    }

    /// Bind doc-value readers of `leaf`.
    ///
    /// # Safety
    /// The executor keeps readers borrowed from `leaf` past this call: it
    /// must not be used (other than dropped or re-bound) once the borrow of
    /// the leaf's segment/memtable ends. Search scorers satisfy this since
    /// they never outlive the search that borrowed the leaf.
    pub unsafe fn bind_leaf(&mut self, leaf: &LeafReader<'_>) {
        // SAFETY: lifetime erasure; see the contract above.
        let leaf: LeafReader<'static> = unsafe { std::mem::transmute::<LeafReader<'_>, LeafReader<'static>>(*leaf) };
        self.rt.doc.bind(leaf);
        self.rt.source = None;
    }

    #[inline]
    pub fn set_doc(&mut self, doc: DocId) {
        if self.rt.doc.doc!=doc {self.rt.source=None;}
        self.rt.doc.doc = doc;
        super::allocation::Scope::collect_dead();
    }

    #[inline]
    pub fn set_score(&mut self, score: f64) {
        self.rt.score = score;
    }

    pub fn set_value(&mut self, v: Val) {
        self.rt.value = v;
    }

    pub fn set_ctx(&mut self, ctx: Val) {
        self.rt.ctx = ctx;
    }

    pub fn set_source(&mut self, src: Option<Val>) {
        self.rt.source_override = src;
    }

    /// Set `params[name]` (also for compiled constant-key accesses): for
    /// values JSON cannot carry (NaN / infinities).
    pub fn set_param(&mut self, name: &str, v: Val) {
        for (i, n) in self.rt.prog.param_names.iter().enumerate() {
            if n == name {
                self.rt.param_vals[i] = v.clone();
            }
        }
        if let Val::Map(m) = &self.rt.params {
            m.borrow_mut().insert(Key::str(name), v);
        }
    }

    pub fn params_mut(&mut self) -> &mut Val {
        &mut self.rt.params
    }

    /// Run the script; returns its value (`null` when it returns nothing).
    #[inline]
    pub fn run(&mut self) -> SResult<Val> {
        self.rt.loops = 0;
        if !self.rt.on_big_stack {
            self.rt.stack_base = stack_pos();
        }
        // SAFETY: `rt.prog` is never replaced, so the program outlives this
        // call; avoids an atomic refcount round trip per document.
        let prog: *const Program = Arc::as_ptr(&self.rt.prog);
        if let Some(e) = unsafe { &(*prog).expr_body } {
            return e(&mut self.rt);
        }
        match (unsafe { &(*prog).body })(&mut self.rt)? {
            Flow::Return(v) => Ok(v),
            _ => Ok(Val::Null),
        }
    }

    /// The script's value as a double when its numeric fast path applies.
    #[inline]
    pub fn run_f64(&mut self) -> Option<f64> {
        // SAFETY: as in `run`.
        let prog: *const Program = Arc::as_ptr(&self.rt.prog);
        unsafe { (*prog).num_body.as_ref() }.and_then(|f| f(&mut self.rt))
    }

    /// The script's boolean value when its comparison fast path applies.
    #[inline]
    pub fn run_cond(&mut self) -> Option<bool> {
        // SAFETY: as in `run`.
        let prog: *const Program = Arc::as_ptr(&self.rt.prog);
        unsafe { (*prog).cond_body.as_ref() }.and_then(|f| f(&mut self.rt))
    }

    /// Run and coerce the result to `boolean` (conditions); a failed cast
    /// is reported at the final expression.
    pub fn run_bool(&mut self) -> SResult<bool> {
        let v = self.run()?;
        let pos = self.rt.prog.result_pos;
        v.as_bool().map_err(|e| e.at(pos))
    }

    pub fn take_emitted(&mut self) -> Vec<Val> {
        std::mem::take(&mut self.rt.emitted)
    }

    pub fn program(&self) -> &Arc<Program> {
        &self.rt.prog
    }
}

// ---------------------------------------------------------------------------
// Operators.

fn op_err(op: BinOp, a: &Val, b: &Val) -> Box<ScriptErr> {
    if a.is_null() || b.is_null() {
        let side = if a.is_null() { "left" } else { "right" };
        let var = if op == BinOp::Add { format!("{side}Object") } else { side.to_string() };
        return npe(format!("Cannot invoke \"Object.getClass()\" because \"{var}\" is null"));
    }
    class_cast(format!("Cannot apply [{}] operation to types [{}] and [{}].", op.symbol(), a.type_name(), b.type_name()))
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NumRank {
    Int,
    Long,
    Double,
}

#[inline]
fn rank(v: &Val) -> Option<NumRank> {
    match v {
        Val::Int(_) | Val::Char(_) => Some(NumRank::Int),
        Val::Long(_) => Some(NumRank::Long),
        Val::Double(_) => Some(NumRank::Double),
        _ => None,
    }
}

#[inline]
fn as_i32(v: &Val) -> i32 {
    match v {
        Val::Int(i) => *i,
        Val::Char(c) => *c as i32,
        _ => 0,
    }
}

/// `x + y` of two well-formed strings as one `Rc<str>` (short results are
/// assembled on the stack: one heap allocation).
fn concat_str(x: &str, y: &str) -> Rc<str> {
    let n = x.len() + y.len();
    if n <= 64 {
        let mut buf = [0u8; 64];
        buf[..x.len()].copy_from_slice(x.as_bytes());
        buf[x.len()..n].copy_from_slice(y.as_bytes());
        if let Ok(s) = std::str::from_utf8(&buf[..n]) {
            return Rc::from(s);
        }
    }
    let mut s = String::with_capacity(n);
    s.push_str(x);
    s.push_str(y);
    Rc::from(s)
}

/// Binary arithmetic / comparison with Java numeric promotion.
#[inline]
pub fn binary(op: BinOp, a: &Val, b: &Val) -> SResult<Val> {
    // Fast paths.
    match (op, a, b) {
        (BinOp::Add, Val::Double(x), Val::Double(y)) => return Ok(Val::Double(x + y)),
        (BinOp::Sub, Val::Double(x), Val::Double(y)) => return Ok(Val::Double(x - y)),
        (BinOp::Mul, Val::Double(x), Val::Double(y)) => return Ok(Val::Double(x * y)),
        (BinOp::Div, Val::Double(x), Val::Double(y)) => return Ok(Val::Double(x / y)),
        (BinOp::Add, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x.wrapping_add(*y))),
        (BinOp::Sub, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x.wrapping_sub(*y))),
        (BinOp::Mul, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x.wrapping_mul(*y))),
        (BinOp::Lt, Val::Int(x), Val::Int(y)) => return Ok(Val::Bool(x < y)),
        (BinOp::Gt, Val::Int(x), Val::Int(y)) => return Ok(Val::Bool(x > y)),
        (BinOp::Le, Val::Int(x), Val::Int(y)) => return Ok(Val::Bool(x <= y)),
        (BinOp::Ge, Val::Int(x), Val::Int(y)) => return Ok(Val::Bool(x >= y)),
        // Same results as the general `NumRank::Int` / shift arms below.
        (BinOp::BitAnd, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x & y)),
        (BinOp::BitOr, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x | y)),
        (BinOp::BitXor, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x ^ y)),
        (BinOp::Shl, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x.wrapping_shl(((*y as i64) & 31) as u32))),
        (BinOp::Shr, Val::Int(x), Val::Int(y)) => return Ok(Val::Int(x.wrapping_shr(((*y as i64) & 31) as u32))),
        (BinOp::UShr, Val::Int(x), Val::Int(y)) => return Ok(Val::Int((*x as u32).wrapping_shr(((*y as i64) & 31) as u32) as i32)),
        _ => {}
    }
    match op {
        BinOp::Eq => return Ok(Val::Bool(a.equals(b))),
        BinOp::Ne => return Ok(Val::Bool(!a.equals(b))),
        BinOp::RefEq => return Ok(Val::Bool(a.ref_equals(b))),
        BinOp::RefNe => return Ok(Val::Bool(!a.ref_equals(b))),
        BinOp::Add => {
            // String + String without allocation accounting: the general
            // path below (UTF-16 round trip) yields exactly this `Val::Str`,
            // since two well-formed strings concatenate to a well-formed one.
            if let (Val::Str(x), Val::Str(y)) = (a, b) {
                if !super::allocation::Scope::active() {
                    return Ok(Val::Str(concat_str(x, y)));
                }
            }
            if matches!(a,Val::Utf16(_)) || matches!(b,Val::Utf16(_))
                || (matches!(a,Val::Char(_)) && matches!(b,Val::Str(_)|Val::Builder(_)))
                || (matches!(b,Val::Char(_)) && matches!(a,Val::Str(_)|Val::Builder(_)))
                || (matches!(a,Val::Builder(_)) && matches!(b,Val::Str(_)))
                || (matches!(b,Val::Builder(_)) && matches!(a,Val::Str(_))) {
                let _output=super::allocation::Allocation::scalar_text(&[a,b])?;
                let mut units=a.string_units();units.extend(b.string_units());
                let result=Val::utf16(units);
                super::allocation::Scope::adopt(&result)?;
                return Ok(result);
            }
            if matches!(a, Val::Str(_)) || matches!(b, Val::Str(_)) {
                // Containers can hold lone UTF16 units even when neither
                // outer operand is Val::Utf16. Preserve their Java text too.
                let _output=super::allocation::Allocation::scalar_text(&[a,b])?;
                let mut units = a.string_units();
                units.extend(b.string_units());
                let result=Val::utf16(units);
                super::allocation::Scope::adopt(&result)?;
                return Ok(result);
            }
        }
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
            if let (Val::Bool(x), Val::Bool(y)) = (a, b) {
                return Ok(Val::Bool(match op {
                    BinOp::BitAnd => *x & *y,
                    BinOp::BitOr => *x | *y,
                    _ => *x ^ *y,
                }));
            }
        }
        _ => {}
    }
    let (ra, rb) = match (rank(a), rank(b)) {
        (Some(x), Some(y)) => (x, y),
        _ => {
            // Date comparisons (ZonedDateTime/Instant are Comparable but not via `<`).
            return Err(op_err(op, a, b));
        }
    };
    match op {
        BinOp::Shl | BinOp::Shr | BinOp::UShr => {
            let n = b.as_i64().unwrap_or(0);
            if ra == NumRank::Double || rb == NumRank::Double {
                return Err(op_err(op, a, b));
            }
            return Ok(if ra == NumRank::Int {
                let x = as_i32(a);
                let n = (n & 31) as u32;
                Val::Int(match op {
                    BinOp::Shl => x.wrapping_shl(n),
                    BinOp::Shr => x.wrapping_shr(n),
                    _ => ((x as u32).wrapping_shr(n)) as i32,
                })
            } else {
                let x = a.as_i64().unwrap_or(0);
                let n = (n & 63) as u32;
                Val::Long(match op {
                    BinOp::Shl => x.wrapping_shl(n),
                    BinOp::Shr => x.wrapping_shr(n),
                    _ => ((x as u64).wrapping_shr(n)) as i64,
                })
            });
        }
        _ => {}
    }
    let r = ra.max(rb);
    match r {
        NumRank::Double => {
            let (x, y) = (a.as_f64().unwrap_or(0.0), b.as_f64().unwrap_or(0.0));
            Ok(match op {
                BinOp::Add => Val::Double(x + y),
                BinOp::Sub => Val::Double(x - y),
                BinOp::Mul => Val::Double(x * y),
                BinOp::Div => Val::Double(x / y),
                BinOp::Rem => Val::Double(x % y),
                BinOp::Lt => Val::Bool(x < y),
                BinOp::Le => Val::Bool(x <= y),
                BinOp::Gt => Val::Bool(x > y),
                BinOp::Ge => Val::Bool(x >= y),
                _ => return Err(op_err(op, a, b)),
            })
        }
        NumRank::Long => {
            let (x, y) = (a.as_i64().unwrap_or(0), b.as_i64().unwrap_or(0));
            Ok(match op {
                BinOp::Add => Val::Long(x.wrapping_add(y)),
                BinOp::Sub => Val::Long(x.wrapping_sub(y)),
                BinOp::Mul => Val::Long(x.wrapping_mul(y)),
                BinOp::Div => {
                    if y == 0 {
                        return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
                    }
                    Val::Long(x.wrapping_div(y))
                }
                BinOp::Rem => {
                    if y == 0 {
                        return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
                    }
                    Val::Long(x.wrapping_rem(y))
                }
                BinOp::Lt => Val::Bool(x < y),
                BinOp::Le => Val::Bool(x <= y),
                BinOp::Gt => Val::Bool(x > y),
                BinOp::Ge => Val::Bool(x >= y),
                BinOp::BitAnd => Val::Long(x & y),
                BinOp::BitOr => Val::Long(x | y),
                BinOp::BitXor => Val::Long(x ^ y),
                _ => return Err(op_err(op, a, b)),
            })
        }
        NumRank::Int => {
            let (x, y) = (as_i32(a), as_i32(b));
            Ok(match op {
                BinOp::Add => Val::Int(x.wrapping_add(y)),
                BinOp::Sub => Val::Int(x.wrapping_sub(y)),
                BinOp::Mul => Val::Int(x.wrapping_mul(y)),
                BinOp::Div => {
                    if y == 0 {
                        return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
                    }
                    Val::Int(x.wrapping_div(y))
                }
                BinOp::Rem => {
                    if y == 0 {
                        return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
                    }
                    Val::Int(x.wrapping_rem(y))
                }
                BinOp::Lt => Val::Bool(x < y),
                BinOp::Le => Val::Bool(x <= y),
                BinOp::Gt => Val::Bool(x > y),
                BinOp::Ge => Val::Bool(x >= y),
                BinOp::BitAnd => Val::Int(x & y),
                BinOp::BitOr => Val::Int(x | y),
                BinOp::BitXor => Val::Int(x ^ y),
                _ => return Err(op_err(op, a, b)),
            })
        }
    }
}

pub fn unary(op: UnOp, v: &Val) -> SResult<Val> {
    Ok(match (op, v) {
        (UnOp::Not, Val::Bool(b)) => Val::Bool(!b),
        (UnOp::Neg, Val::Int(i)) => Val::Int(i.wrapping_neg()),
        (UnOp::Neg, Val::Long(l)) => Val::Long(l.wrapping_neg()),
        (UnOp::Neg, Val::Double(d)) => Val::Double(-d),
        (UnOp::Neg, Val::Char(c)) => Val::Int(-(*c as i32)),
        (UnOp::Plus, v) if v.is_number() => match v {
            Val::Char(c) => Val::Int(*c as i32),
            v => v.clone(),
        },
        (UnOp::BitNot, Val::Int(i)) => Val::Int(!i),
        (UnOp::BitNot, Val::Long(l)) => Val::Long(!l),
        (_, Val::Null) => return Err(npe("Cannot invoke \"Object.getClass()\" because \"value\" is null")),
        (op, v) => {
            let s = match op {
                UnOp::Not => "!",
                UnOp::Neg => "-",
                UnOp::Plus => "+",
                UnOp::BitNot => "~",
            };
            return Err(class_cast(format!("Cannot apply [{s}] operation to type [{}].", v.type_name())));
        }
    })
}

fn prim_name(t: &Ty) -> String {
    t.name()
}

fn java_name(v: &Val) -> &'static str {
    match v {
        Val::Int(_) => "int",
        Val::Long(_) => "long",
        Val::Double(_) => "double",
        Val::Char(_) => "char",
        Val::Bool(_) => "boolean",
        other => other.type_name(),
    }
}

/// Implicit conversion on assignment to a declared type (widening only).
pub fn coerce(t: &Ty, v: Val) -> SResult<Val> {
    match t {
        Ty::Def | Ty::Void => Ok(v),
        Ty::Int | Ty::Short | Ty::Byte => match v {
            Val::Int(_) => Ok(v),
            Val::Char(c) => Ok(Val::Int(c as i32)),
            Val::Null => Err(npe(format!("cannot assign null to a primitive [{}]", prim_name(t)))),
            other => Err(class_cast(format!("Cannot cast from [{}] to [{}].", java_name(&other), prim_name(t)))),
        },
        Ty::Long => match v {
            Val::Long(_) => Ok(v),
            Val::Int(i) => Ok(Val::Long(i as i64)),
            Val::Char(c) => Ok(Val::Long(c as i64)),
            Val::Null => Err(npe("cannot assign null to a primitive [long]")),
            other => Err(class_cast(format!("Cannot cast from [{}] to [long].", java_name(&other)))),
        },
        Ty::Double | Ty::Float => match v {
            Val::Double(d) => Ok(Val::Double(if *t == Ty::Float { d as f32 as f64 } else { d })),
            Val::Int(i) => Ok(Val::Double(i as f64)),
            Val::Long(l) => Ok(Val::Double(l as f64)),
            Val::Char(c) => Ok(Val::Double(c as u32 as f64)),
            Val::Null => Err(npe(format!("cannot assign null to a primitive [{}]", prim_name(t)))),
            other => Err(class_cast(format!("Cannot cast from [{}] to [{}].", java_name(&other), prim_name(t)))),
        },
        Ty::Boolean => match v {
            Val::Bool(_) => Ok(v),
            other => Err(class_cast(format!("Cannot cast from [{}] to [boolean].", java_name(&other)))),
        },
        Ty::Char => match v {
            Val::Char(_) => Ok(v),
            Val::Str(_) | Val::Utf16(_) if v.string_units().len() == 1 => Ok(Val::Char(v.string_units()[0])),
            other => Err(class_cast(format!("Cannot cast from [{}] to [char].", java_name(&other)))),
        },
        Ty::String => match v {
            Val::Str(_) | Val::Utf16(_) | Val::Null => Ok(v),
            other => Err(class_cast(format!("Cannot cast {} to java.lang.String", other.type_name()))),
        },
        Ty::Ref(name) => check_ref(name, v),
        Ty::Array(_) => match v {
            Val::List(_) | Val::Array { .. } | Val::Null => Ok(v),
            other => Err(class_cast(format!("Cannot cast {} to {}", other.type_name(), t.name()))),
        },
    }
}

fn check_ref(name: &str, v: Val) -> SResult<Val> {
    let ext = match &v {
        Val::Time(_) | Val::Date(_) | Val::Instant(_) => super::jtime::instance_of(&v, name),
        Val::Big(_) => super::jbig::instance_of(&v, name),
        Val::Obj(_) => super::jmisc::instance_of(&v, name),
        _ => None,
    };
    let ok = match (name, &v) {
        _ if ext.is_some() => ext == Some(true),
        (_, Val::Null) => true,
        ("List" | "ArrayList" | "LinkedList" | "AbstractList" | "Deque" | "Queue" | "ArrayDeque" | "RandomAccess", Val::List(_) | Val::DocValues(_)) => true,
        ("Map" | "AbstractMap", Val::Map(_)) => true,
        ("HashMap", Val::Map(m)) => m.kind != MapKind::Tree,
        ("LinkedHashMap", Val::Map(m)) => matches!(m.kind, MapKind::Linked | MapKind::Json),
        ("TreeMap" | "SortedMap" | "NavigableMap", Val::Map(m)) => m.kind == MapKind::Tree,
        ("Set" | "AbstractSet", Val::Set(_)) => true,
        ("HashSet", Val::Set(l)) => l.kind != CollKind::TreeSet,
        ("LinkedHashSet", Val::Set(l)) => l.kind == CollKind::LinkedSet,
        ("TreeSet" | "SortedSet" | "NavigableSet", Val::Set(l)) => l.kind == CollKind::TreeSet,
        ("Collection" | "Iterable" | "AbstractCollection", Val::List(_) | Val::Set(_) | Val::DocValues(_) | Val::Array { .. }) => true,
        ("Integer", Val::Int(_)) | ("Long", Val::Long(_)) | ("Double", Val::Double(_)) | ("Boolean", Val::Bool(_)) | ("Character", Val::Char(_)) => true,
        ("Integer", Val::Char(c)) => return Ok(Val::Int(*c as i32)),
        ("Long", Val::Int(i)) => return Ok(Val::Long(*i as i64)),
        ("Double" | "Float", Val::Int(_) | Val::Long(_)) => return Ok(Val::Double(v.as_f64().unwrap_or(0.0))),
        ("Number", v) if v.is_number() => true,
        ("ZonedDateTime" | "JodaCompatibleZonedDateTime" | "Date", Val::Date(_)) => true,
        ("Instant", Val::Instant(_)) => true,
        ("DayOfWeek", Val::DayOfWeek(_)) | ("Month", Val::Month(_)) => true,
        ("Enum", Val::DayOfWeek(_) | Val::Month(_) | Val::ChronoUnit(_)) => true,
        ("Enum", Val::Time(_)) => ["ChronoUnit", "ChronoField", "TextStyle", "ResolverStyle", "FormatStyle", "IsoEra", "DayOfWeek", "Month"].iter().any(|c| super::jtime::instance_of(&v, c) == Some(true)),
        ("Map.Entry" | "Entry", Val::Entry(_)) => true,
        ("Optional", Val::Optional(_)) => true,
        ("Pattern", Val::Pattern(_)) | ("Matcher" | "MatchResult", Val::Matcher(_)) => true,
        ("StringBuilder" | "CharSequence", Val::Builder(_)) => true,
        ("CharSequence" | "Comparable", Val::Str(_) | Val::Utf16(_)) => true,
        ("Comparable", v) => v.is_number() || matches!(v, Val::Date(_) | Val::Instant(_)),
        ("Stream" | "IntStream" | "LongStream" | "DoubleStream", Val::Stream(_)) => true,
        ("Function" | "Predicate" | "BiFunction" | "Consumer" | "Supplier" | "Comparator", Val::Lambda(_) | Val::MethodRef(_)) => true,
        ("ScriptDocValues", Val::DocValues(_)) => true,
        // `ScriptDocValues.Geometry` (geo_point / geo_shape doc values, whose
        // values are point / shape maps); other inner classes by field kind
        // are not modelled and accept any doc values.
        ("ScriptDocValues.Geometry" | "ScriptDocValues.GeoPoints", Val::DocValues(d)) => d.values.first().is_none_or(|v| matches!(v, Val::Map(_))),
        (n, Val::DocValues(_)) if n.starts_with("ScriptDocValues.") => true,
        ("Exception" | "RuntimeException" | "IllegalArgumentException" | "IllegalStateException", Val::Exception(_)) => true,
        ("Object" | "GeoPoint" | "GeoBoundingBox", _) => true,
        _ => false,
    };
    if ok {
        Ok(v)
    } else {
        Err(class_cast(format!("Cannot cast {} to {}", v.type_name(), name)))
    }
}

fn f64_to_i64(d: f64) -> i64 {
    if d.is_nan() { 0 } else { d as i64 }
}

fn f64_to_i32(d: f64) -> i32 {
    if d.is_nan() { 0 } else { d as i32 }
}

/// Explicit cast `(T) v`.
pub fn cast(t: &Ty, v: Val) -> SResult<Val> {
    let num = |v: &Val| -> SResult<()> {
        if v.is_number() {
            Ok(())
        } else if v.is_null() {
            Err(npe(format!("cannot cast null to [{}]", t.name())))
        } else {
            Err(class_cast(format!("Cannot cast {} to {}", v.type_name(), t.name())))
        }
    };
    match t {
        Ty::Int => {
            num(&v)?;
            Ok(Val::Int(match v {
                Val::Double(d) => f64_to_i32(d),
                Val::Long(l) => l as i32,
                other => as_i32(&other),
            }))
        }
        Ty::Short => {
            num(&v)?;
            Ok(Val::Int(v.as_i64().unwrap_or(0) as i16 as i32))
        }
        Ty::Byte => {
            num(&v)?;
            Ok(Val::Int(v.as_i64().unwrap_or(0) as i8 as i32))
        }
        Ty::Long => {
            num(&v)?;
            Ok(Val::Long(match v {
                Val::Double(d) => f64_to_i64(d),
                other => other.as_i64().unwrap_or(0),
            }))
        }
        Ty::Double => {
            num(&v)?;
            Ok(Val::Double(v.as_f64().unwrap_or(0.0)))
        }
        Ty::Float => {
            num(&v)?;
            Ok(Val::Double(v.as_f64().unwrap_or(0.0) as f32 as f64))
        }
        Ty::Char => match v {
            Val::Str(_) | Val::Utf16(_) if v.string_units().len() == 1 => Ok(Val::Char(v.string_units()[0])),
            Val::Char(_) => Ok(v),
            ref n if n.is_number() => Ok(Val::Char(n.as_i64().unwrap_or(0) as u16)),
            other => Err(class_cast(format!("Cannot cast {} to char", other.type_name()))),
        },
        Ty::String => match v {
            Val::Char(c) => Ok(Val::utf16(vec![c])),
            other => coerce(t, other),
        },
        _ => coerce(t, v),
    }
}

pub fn instance_of(v: &Val, t: &Ty) -> bool {
    match t {
        Ty::Def => !v.is_null(),
        Ty::Int => matches!(v, Val::Int(_)),
        Ty::Long => matches!(v, Val::Long(_)),
        Ty::Double | Ty::Float => matches!(v, Val::Double(_)),
        Ty::Boolean => matches!(v, Val::Bool(_)),
        Ty::Char => matches!(v, Val::Char(_)),
        Ty::String => matches!(v, Val::Str(_) | Val::Utf16(_)),
        Ty::Ref(n) => !v.is_null() && check_ref(n, v.clone()).is_ok() && !(n == "Long" && matches!(v, Val::Int(_))) && !(n.starts_with("Double") && !matches!(v, Val::Double(_))),
        Ty::Array(_) => matches!(v, Val::List(_) | Val::Array { .. }),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Field / index access.

/// Elements of an iterable (for-each, addAll, ...).
pub fn iter_values(v: &Val) -> SResult<Vec<Val>> {
    Ok(match v {
        Val::List(l) | Val::Set(l) => l.borrow().clone(),
        Val::Array { items, .. } => items.borrow().clone(),
        Val::Map(m) => m.borrow().keys().map(Key::to_val).collect(),
        Val::DocValues(d) => d.values.clone(),
        Val::Stream(s) => s.values.clone(),
        Val::Str(s) => s.encode_utf16().map(Val::Char).collect(),
        Val::Obj(o) => match &**o {
            super::jmisc::Obj::Field { values, .. } => values.clone(),
            _ => return Err(class_cast(format!("Cannot cast {} to java.lang.Iterable", v.type_name()))),
        },
        Val::Null => return Err(npe("Cannot iterate over a null value")),
        other => return Err(class_cast(format!("Cannot cast {} to java.lang.Iterable", other.type_name()))),
    })
}

fn array_oob(mut e: Box<ScriptErr>) -> Box<ScriptErr> {
    if e.kind == "index_out_of_bounds_exception" {
        e.kind = "array_index_out_of_bounds_exception".into();
    }
    e
}

fn list_index(len: usize, i: i64) -> SResult<usize> {
    let j = if i < 0 { i + len as i64 } else { i };
    if j < 0 || j >= len as i64 {
        return Err(out_of_bounds(format!("Index {i} out of bounds for length {len}")));
    }
    Ok(j as usize)
}

pub fn index_get(rt: &mut Rt, obj: &Val, idx: &Val) -> SResult<Val> {
    match obj {
        Val::Map(m) => Ok(m.borrow().get(&Key::from_val(idx)).cloned().unwrap_or(Val::Null)),
        Val::List(l) => {
            let l = l.borrow();
            let i = list_index(l.len(), idx.to_index()?)?;
            Ok(l[i].clone())
        }
        Val::Array { items, .. } => {
            let l = items.borrow();
            let i = list_index(l.len(), idx.to_index()?).map_err(array_oob)?;
            Ok(l[i].clone())
        }
        Val::DocValues(d) => {
            let i = list_index(d.values.len(), idx.to_index()?)?;
            Ok(d.values[i].clone())
        }
        Val::Doc => {
            let name = match idx {
                Val::Str(s) => s.to_string(),
                other => other.to_java_string(),
            };
            let slot = rt.doc.slot_of(&name);
            rt.doc.values(slot)
        }
        Val::Null => Err(npe("cannot access method/field [normalizeIndex] from a null def reference")),
        other => Err(illegal_arg(format!("Attempting to address a non-array-like type [{}] as an array.", other.type_name()))),
    }
}

pub fn index_set(obj: &Val, idx: &Val, v: Val) -> SResult<()> {
    match obj {
        Val::Map(m) => {
            m.insert_checked(Key::from_val(idx), v)?;
            Ok(())
        }
        Val::List(l) => {
            l.register_mutation()?;
            let mut l = l.borrow_mut();
            let i = list_index(l.len(), idx.to_index()?)?;
            l[i] = v;
            Ok(())
        }
        Val::Array { items, .. } => {
            items.register_mutation()?;
            let mut l = items.borrow_mut();
            let i = list_index(l.len(), idx.to_index()?).map_err(array_oob)?;
            l[i] = v;
            Ok(())
        }
        Val::Null => Err(npe("cannot access method/field [normalizeIndex] from a null def reference")),
        other => Err(illegal_arg(format!("Attempting to address a non-array-like type [{}] as an array.", other.type_name()))),
    }
}

/// `obj.name` (map key, `length`, or a getter shortcut).
pub fn field_get(rt: &mut Rt, obj: &Val, name: &str) -> SResult<Val> {
    match obj {
        Val::Map(m) => Ok(m.borrow().get(&Key::str(name)).cloned().unwrap_or(Val::Null)),
        Val::List(l) if name == "length" => Ok(Val::Int(l.borrow().len() as i32)),
        Val::Array { items, .. } if name == "length" => Ok(Val::Int(items.borrow().len() as i32)),
        // `list.0` (DOTINTEGER): `list.get(0)`.
        Val::List(l) if !name.is_empty() && name.bytes().all(|c| c.is_ascii_digit()) => {
            let i: usize = name.parse().map_err(|_| illegal_arg(format!("invalid list index [{name}]")))?;
            let l = l.borrow();
            l.get(i).cloned().ok_or_else(|| out_of_bounds(format!("Index {i} out of bounds for length {}", l.len())))
        }
        Val::Null => Err(npe(format!("cannot access method/field [{name}] from a null def reference"))),
        Val::Doc => {
            let slot = rt.doc.slot_of(name);
            rt.doc.values(slot)
        }
        _ => {
            // Getter shortcut: x.foo -> x.getFoo() / x.isFoo().
            let mut cap = String::with_capacity(name.len() + 3);
            cap.push_str("get");
            let mut cs = name.chars();
            if let Some(f) = cs.next() {
                cap.extend(f.to_uppercase());
                cap.push_str(cs.as_str());
            }
            match super::methods::call_method(rt, obj.clone(), &cap, Vec::new()) {
                Ok(v) => Ok(v),
                Err(e) if e.kind == "illegal_argument_exception" && e.msg.starts_with("dynamic method") => {
                    let is = format!("is{}", &cap[3..]);
                    super::methods::call_method(rt, obj.clone(), &is, Vec::new()).map_err(|_| {
                        illegal_arg(format!("dynamic getter [{}, {name}] not found", obj.type_name()))
                    })
                }
                Err(e) => Err(e),
            }
        }
    }
}

pub fn field_set(obj: &Val, name: &str, v: Val) -> SResult<()> {
    match obj {
        Val::Map(m) => {
            m.insert_checked(Key::str(name), v)?;
            Ok(())
        }
        Val::Null => Err(npe(format!("cannot access method/field [{name}] from a null def reference"))),
        other => Err(illegal_arg(format!("dynamic setter [{}, {name}] not found", other.type_name()))),
    }
}

// ---------------------------------------------------------------------------
// Statics.

pub const STATIC_CLASSES: &[&str] = &[
    "Math", "StrictMath", "Integer", "Long", "Double", "Float", "Short", "Byte", "Boolean", "Character", "String", "Collections", "Arrays", "Objects", "Instant",
    "ZonedDateTime", "ZoneId", "ZoneOffset", "DayOfWeek", "Month", "ChronoUnit", "Collectors", "List", "Map", "Set", "Stream", "IntStream", "Optional", "Base64",
    "LocalDate", "DateTimeFormatter", "Duration", "System", "Debug", "Pattern", "LongStream", "DoubleStream", "Comparator", "Function", "Map.Entry",
    // java.time (jtime)
    "LocalDateTime", "LocalTime", "OffsetDateTime", "OffsetTime", "Period", "YearMonth", "Year", "MonthDay", "TemporalAdjusters", "TextStyle", "Locale",
    "ChronoField", "Clock", "FormatStyle", "ResolverStyle", "IsoChronology", "IsoFields",
    // java.math (jbig)
    "BigDecimal", "BigInteger", "MathContext", "RoundingMode",
    // jmisc
    "UUID", "StandardCharsets", "Normalizer", "TimeUnit",
    // org.elasticsearch.painless.api.Json (ingest / reindex / update / watcher)
    "Json",
];

fn arg_f(args: &[Val], i: usize) -> SResult<f64> {
    args.get(i).ok_or_else(|| illegal_arg("missing argument"))?.to_f64()
}

fn arg_str(args: &[Val], i: usize) -> SResult<String> {
    match args.get(i) {
        Some(Val::Str(s)) => Ok(s.to_string()),
        Some(Val::Null) | None => Err(npe("argument is null")),
        Some(v) => Ok(v.to_java_string()),
    }
}

fn parse_int(s: &str, radix: u32) -> SResult<i64> {
    i64::from_str_radix(s.trim_start_matches('+'), radix).map_err(|_| ScriptErr::new("number_format_exception", format!("For input string: \"{s}\"")))
}

/// JDK frames above the script frame for `Integer.parseInt` failures.
pub const PARSE_INT_FRAMES: &[&str] = &[
    "java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:67)",
    "java.base/java.lang.Integer.parseInt(Integer.java:529)",
    "java.base/java.lang.Integer.parseInt(Integer.java:626)",
];

pub fn static_field(class: &str, name: &str) -> Option<Val> {
    if super::jtime::CLASSES.contains(&class) {
        if let Some(v) = super::jtime::static_field(class, name) {
            return Some(v);
        }
    }
    if super::jbig::CLASSES.contains(&class) {
        return super::jbig::static_field(class, name);
    }
    Some(match (class, name) {
        ("Pattern", f) => Val::Int(super::regex::flag_const(f)?),
        ("Math" | "StrictMath", "PI") => Val::Double(std::f64::consts::PI),
        ("Math" | "StrictMath", "E") => Val::Double(std::f64::consts::E),
        ("Integer", "MAX_VALUE") => Val::Int(i32::MAX),
        ("Integer", "MIN_VALUE") => Val::Int(i32::MIN),
        ("Long", "MAX_VALUE") => Val::Long(i64::MAX),
        ("Long", "MIN_VALUE") => Val::Long(i64::MIN),
        ("Short", "MAX_VALUE") => Val::Int(i16::MAX as i32),
        ("Short", "MIN_VALUE") => Val::Int(i16::MIN as i32),
        ("Byte", "MAX_VALUE") => Val::Int(i8::MAX as i32),
        ("Byte", "MIN_VALUE") => Val::Int(i8::MIN as i32),
        ("Double", "MAX_VALUE") => Val::Double(f64::MAX),
        ("Double", "MIN_VALUE") => Val::Double(f64::from_bits(1)),
        ("Double", "POSITIVE_INFINITY") | ("Float", "POSITIVE_INFINITY") => Val::Double(f64::INFINITY),
        ("Double", "NEGATIVE_INFINITY") | ("Float", "NEGATIVE_INFINITY") => Val::Double(f64::NEG_INFINITY),
        ("Double", "NaN") | ("Float", "NaN") => Val::Double(f64::NAN),
        ("Float", "MAX_VALUE") => Val::Double(f32::MAX as f64),
        ("Float", "MIN_VALUE") => Val::Double(f32::from_bits(1) as f64),
        ("Boolean", "TRUE") => Val::Bool(true),
        ("Boolean", "FALSE") => Val::Bool(false),
        ("ZoneOffset", "UTC") => Val::str("Z"),
        ("Collections", "EMPTY_LIST") => Val::new_list(Vec::new()),
        ("Collections", "EMPTY_MAP") => Val::new_map(IndexMap::new()),
        ("DayOfWeek", d) => Val::DayOfWeek(super::value::DAYS.iter().position(|x| *x == d)? as u8 + 1),
        ("Month", m) => Val::Month(super::value::MONTHS.iter().position(|x| *x == m)? as u8 + 1),
        ("ChronoUnit", u) => Val::ChronoUnit(CHRONO_UNITS.iter().position(|x| *x == u)? as u8),
        ("DateTimeFormatter", f) => Val::str(f),
        _ => return None,
    })
}

fn to_millis(v: &Val) -> SResult<i64> {
    match v {
        Val::Date(m) | Val::Instant(m) => Ok(*m),
        other => Err(class_cast(format!("Cannot cast {} to java.time.temporal.Temporal", other.type_name()))),
    }
}

/// Character allowlist signatures that require char rather than code-point int.
pub(super) fn character_char_arguments(method: &str, arity: usize) -> usize {
    match (method,arity) {
        ("compare" | "isSurrogatePair" | "toCodePoint",2) => 2,
        ("hashCode" | "isHighSurrogate" | "isSurrogate" | "reverseBytes" | "toString" | "toUpperCase" | "toLowerCase" | "toTitleCase" | "valueOf",1) => 1,
        _ => 0,
    }
}

/// `Class.method(args)`; `Ok(None)` when the class has no such method.
pub fn call_static(rt: &mut Rt, class: &str, method: &str, args: Vec<Val>) -> SResult<Option<Val>> {
    if super::jtime::CLASSES.contains(&class) {
        if let Some(v) = super::jtime::call_static(rt, class, method, &args)? {
            return Ok(Some(v));
        }
    }
    if super::jbig::CLASSES.contains(&class) {
        return super::jbig::call_static(rt, class, method, &args);
    }
    if super::jmisc::CLASSES.contains(&class) {
        return super::jmisc::call_static(rt, class, method, &args);
    }
    let n = args.len();
    if class == "Character" {
        for arg in args.iter().take(character_char_arguments(method,n)) {
            if !matches!(arg,Val::Char(_)) {
                return Err(class_cast(format!("cannot implicitly cast def [{}] to char",arg.type_name())));
            }
        }
    }
    let v = match (class, method, n) {
        ("Math" | "StrictMath", _, _) => match math(method, &args)? {
            Some(v) => v,
            None => return Ok(None),
        },
        ("DateTimeFormatter", "ofPattern", 1 | 2) => Val::string(format!("{}{}", super::timefmt::PATTERN_PREFIX, args[0].to_java_string())),
        ("Matcher", "quoteReplacement", 1) => Val::utf16(super::regex::quote_replacement(&args[0].string_units())),
        ("Pattern", _, _) => match super::regex::pattern_static(method, &args)? {
            Some(v) => v,
            None => return Ok(None),
        },
        ("Integer" | "Long" | "Short" | "Byte", "parseInt" | "parseLong" | "parseShort" | "parseByte", 1) if args[0].is_null() => {
            return Err(ScriptErr::new("number_format_exception", "Cannot parse null string"));
        }
        ("Integer", "parseInt", 1) | ("Integer", "valueOf", 1) if matches!(args[0], Val::Str(_)) => {
            let s = arg_str(&args, 0)?;
            let v = parse_int(&s, 10).map_err(|e| e.frames(PARSE_INT_FRAMES))?;
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err(ScriptErr::new("number_format_exception", format!("For input string: \"{s}\"")).frames(PARSE_INT_FRAMES));
            }
            Val::Int(v as i32)
        }
        ("Integer", "parseInt", 2) => {
            let s = arg_str(&args, 0)?;
            Val::Int(parse_int(&s, args[1].to_i64()? as u32).map_err(|e| e.frames(PARSE_INT_FRAMES))? as i32)
        }
        ("Integer", "valueOf", 1) => cast(&Ty::Int, args[0].clone())?,
        ("String", "copyValueOf", 1 | 3) => {
            let Val::Array { elem: Ty::Char, items } = &args[0] else {
                return Err(if args[0].is_null() { npe("character array is null") } else { class_cast("Expected char[]") });
            };
            let chars = items.borrow();
            let offset = if n == 3 { args[1].to_i64()? } else { 0 };
            let count = if n == 3 { args[2].to_i64()? } else { chars.len() as i64 };
            if offset < 0 || count < 0 || offset > chars.len() as i64 || count > chars.len() as i64 - offset {
                return Err(illegal_arg("String character array range out of bounds"));
            }
            Val::utf16(chars[offset as usize..(offset + count) as usize].iter().map(|v| match v {
                Val::Char(c) => Ok(*c), _ => Err(class_cast("Expected char")),
            }).collect::<SResult<Vec<_>>>()?)
        }
        ("Integer", "toString", 1) | ("Long", "toString", 1) | ("Double", "toString", 1) | ("Float", "toString", 1) | ("Boolean", "toString", 1) | ("String", "valueOf", 1) | ("Objects", "toString", 1) => {
            Val::utf16(args[0].string_units())
        }
        ("Objects", "toString", 2) => {
            if args[0].is_null() {
                args[1].clone()
            } else {
                Val::utf16(args[0].string_units())
            }
        }
        ("Integer", "toHexString", 1) => Val::string(format!("{:x}", args[0].to_i64()? as i32 as u32)),
        ("Long", "toHexString", 1) => Val::string(format!("{:x}", args[0].to_i64()? as u64)),
        ("Integer", "toBinaryString", 1) => Val::string(format!("{:b}", args[0].to_i64()? as i32 as u32)),
        ("Integer" | "Long" | "Double" | "Float", "compare", 2) => Val::Int(args[0].compare(&args[1])? as i32),
        ("Integer", "max" | "min" | "sum", 2) => {
            let (a, b) = (args[0].to_i64()? as i32, args[1].to_i64()? as i32);
            Val::Int(match method { "max" => a.max(b), "min" => a.min(b), _ => a.wrapping_add(b) })
        }
        ("Long", "max" | "min" | "sum", 2) => {
            let (a, b) = (args[0].to_i64()?, args[1].to_i64()?);
            Val::Long(match method { "max" => a.max(b), "min" => a.min(b), _ => a.wrapping_add(b) })
        }
        ("Double" | "Float", "max" | "min" | "sum", 2) => {
            let (a, b) = (args[0].to_f64()?, args[1].to_f64()?);
            Val::Double(match method { "sum" => a + b, _ if a.is_nan() || b.is_nan() => f64::NAN, "max" => a.max(b), _ => a.min(b) })
        }
        ("Integer", "bitCount", 1) => Val::Int((args[0].to_i64()? as i32).count_ones() as i32),
        ("Long", "bitCount", 1) => Val::Int(args[0].to_i64()?.count_ones() as i32),
        ("Integer", "reverse", 1) => Val::Int((args[0].to_i64()? as i32).reverse_bits()),
        ("Long", "reverse", 1) => Val::Long(args[0].to_i64()?.reverse_bits()),
        ("Integer", "reverseBytes", 1) => Val::Int((args[0].to_i64()? as i32).swap_bytes()),
        ("Long", "reverseBytes", 1) => Val::Long(args[0].to_i64()?.swap_bytes()),
        ("Integer", "highestOneBit", 1) => {
            let v = args[0].to_i64()? as i32 as u32;
            Val::Int(if v == 0 { 0 } else { (1u32 << (31 - v.leading_zeros())) as i32 })
        }
        ("Long", "highestOneBit", 1) => {
            let v = args[0].to_i64()? as u64;
            Val::Long(if v == 0 { 0 } else { (1u64 << (63 - v.leading_zeros())) as i64 })
        }
        ("Integer", "lowestOneBit", 1) => {
            let v = args[0].to_i64()? as i32;
            Val::Int(v & v.wrapping_neg())
        }
        ("Long", "lowestOneBit", 1) => {
            let v = args[0].to_i64()?;
            Val::Long(v & v.wrapping_neg())
        }
        ("Integer", "numberOfLeadingZeros", 1) => Val::Int((args[0].to_i64()? as i32).leading_zeros() as i32),
        ("Long", "numberOfLeadingZeros", 1) => Val::Int(args[0].to_i64()?.leading_zeros() as i32),
        ("Integer", "numberOfTrailingZeros", 1) => Val::Int((args[0].to_i64()? as i32).trailing_zeros() as i32),
        ("Long", "numberOfTrailingZeros", 1) => Val::Int(args[0].to_i64()?.trailing_zeros() as i32),
        ("Integer", "rotateLeft", 2) => Val::Int((args[0].to_i64()? as i32).rotate_left((args[1].to_i64()? & 31) as u32)),
        ("Integer", "rotateRight", 2) => Val::Int((args[0].to_i64()? as i32).rotate_right((args[1].to_i64()? & 31) as u32)),
        ("Long", "rotateLeft", 2) => Val::Long(args[0].to_i64()?.rotate_left((args[1].to_i64()? & 63) as u32)),
        ("Long", "rotateRight", 2) => Val::Long(args[0].to_i64()?.rotate_right((args[1].to_i64()? & 63) as u32)),
        ("Integer", "toOctalString", 1) => Val::string(format!("{:o}", args[0].to_i64()? as i32 as u32)),
        ("Long", "toOctalString", 1) => Val::string(format!("{:o}", args[0].to_i64()? as u64)),
        ("Long", "toBinaryString", 1) => Val::string(format!("{:b}", args[0].to_i64()? as u64)),
        ("Integer", "toString", 2) | ("Long", "toString", 2) => {
            let (mut v, radix) = (args[0].to_i64()?, args[1].to_i64()?.clamp(2, 36) as i64);
            let neg = v < 0;
            let mut digits = Vec::new();
            if v == 0 {
                digits.push('0');
            }
            while v != 0 {
                digits.push(std::char::from_digit((v % radix).unsigned_abs() as u32, radix as u32).unwrap_or('0'));
                v /= radix;
            }
            if neg {
                digits.push('-');
            }
            Val::string(digits.iter().rev().collect::<String>())
        }
        ("Integer", "toUnsignedString", 1) => Val::string((args[0].to_i64()? as i32 as u32).to_string()),
        ("Long", "toUnsignedString", 1) => Val::string((args[0].to_i64()? as u64).to_string()),
        ("Integer", "toUnsignedLong", 1) => Val::Long(args[0].to_i64()? as i32 as u32 as i64),
        ("Integer", "divideUnsigned", 2) => {
            let b = args[1].to_i64()? as i32 as u32;
            if b == 0 {
                return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
            }
            Val::Int(((args[0].to_i64()? as i32 as u32) / b) as i32)
        }
        ("Integer", "remainderUnsigned", 2) => {
            let b = args[1].to_i64()? as i32 as u32;
            if b == 0 {
                return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
            }
            Val::Int(((args[0].to_i64()? as i32 as u32) % b) as i32)
        }
        ("Integer", "parseUnsignedInt", 1) => {
            let s = arg_str(&args, 0)?;
            Val::Int(s.parse::<u32>().map_err(|_| ScriptErr::new("number_format_exception", format!("For input string: \"{s}\"")))? as i32)
        }
        ("Integer", "decode", 1) | ("Long", "decode", 1) => {
            let s = arg_str(&args, 0)?;
            let (neg, t) = match s.strip_prefix('-') { Some(r) => (true, r.to_string()), None => (false, s.trim_start_matches('+').to_string()) };
            let (radix, digits) = if let Some(r) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).or_else(|| t.strip_prefix('#')) { (16, r.to_string()) } else if t.len() > 1 && t.starts_with('0') { (8, t[1..].to_string()) } else { (10, t.clone()) };
            let v = i64::from_str_radix(&digits, radix).map_err(|_| ScriptErr::new("number_format_exception", format!("For input string: \"{digits}\"")))?;
            let v = if neg { -v } else { v };
            if class == "Integer" { Val::Int(v as i32) } else { Val::Long(v) }
        }
        ("Integer", "hashCode", 1) => Val::Int(args[0].to_i64()? as i32),
        ("Long", "hashCode", 1) => Val::Int(Val::Long(args[0].to_i64()?).java_hash()),
        ("Double", "hashCode", 1) => Val::Int(Val::Double(args[0].to_f64()?).java_hash()),
        ("Boolean", "hashCode", 1) => Val::Int(if matches!(args[0], Val::Bool(true)) { 1231 } else { 1237 }),
        ("Boolean", "logicalAnd" | "logicalOr" | "logicalXor", 2) => {
            let (a, b) = (args[0].as_bool()?, args[1].as_bool()?);
            Val::Bool(match method { "logicalAnd" => a && b, "logicalOr" => a || b, _ => a ^ b })
        }
        ("Short", "parseShort", 1) => {
            let s = arg_str(&args, 0)?;
            let v = parse_int(&s, 10)?;
            if v < i16::MIN as i64 || v > i16::MAX as i64 {
                return Err(ScriptErr::new("number_format_exception", format!("Value out of range. Value:\"{s}\" Radix:10")));
            }
            Val::Int(v as i32)
        }
        ("Byte", "parseByte", 1) => {
            let s = arg_str(&args, 0)?;
            let v = parse_int(&s, 10)?;
            if v < i8::MIN as i64 || v > i8::MAX as i64 {
                return Err(ScriptErr::new("number_format_exception", format!("Value out of range. Value:\"{s}\" Radix:10")));
            }
            Val::Int(v as i32)
        }
        ("Character", "forDigit", 2) => match std::char::from_digit(args[0].to_i64()? as u32, args[1].to_i64()? as u32) {
            Some(c) => Val::Char(c as u16),
            None => Val::Char(0),
        },
        ("Character", "charCount", 1) => Val::Int(if args[0].to_i64()? >= 0x10000 { 2 } else { 1 }),
        ("Character", "toChars", 1 | 3) => {
            let cp=args[0].to_i64()?;
            if !(0..=0x10ffff).contains(&cp) {return Err(illegal_arg(format!("Not a valid Unicode code point: 0x{cp:X}")));}
            let units=if cp<0x10000 {vec![cp as u16]} else {vec![((cp-0x10000)/1024+0xd800) as u16,((cp-0x10000)%1024+0xdc00) as u16]};
            if n==1 {Val::new_array(super::ast::Ty::Char,units.into_iter().map(Val::Char).collect())} else {
                let Val::Array{elem: super::ast::Ty::Char,items}=&args[1] else {return Err(if args[1].is_null(){npe("destination is null")}else{class_cast("Expected char[]")});};
                let at=args[2].to_i64()?;
                let mut dest=items.borrow_mut();
                if at<0 || (at as usize).checked_add(units.len()).is_none_or(|end|end>dest.len()) {return Err(ScriptErr::new("index_out_of_bounds_exception","Invalid destination index"));}
                for (i,u) in units.iter().enumerate() {dest[at as usize+i]=Val::Char(*u);}
                Val::Int(units.len() as i32)
            }
        }
        ("Character", "toString", 1) => Val::utf16(vec![args[0].to_i64()? as u16]),
        ("Character", "hashCode", 1) => Val::Int(args[0].to_i64()? as u16 as i32),
        ("Character", "compare", 2) => Val::Int(args[0].to_i64()? as u16 as i32-args[1].to_i64()? as u16 as i32),
        ("Character", "reverseBytes", 1) => Val::Char((args[0].to_i64()? as u16).swap_bytes()),
        ("Character", "digit", 2) => Val::Int(match &args[0] {
            Val::Char(c) => char::from_u32(*c as u32).unwrap_or('\0'),
            other => char::from_u32(other.to_i64()? as u32).unwrap_or('\0'),
        }.to_digit(args[1].to_i64()? as u32).map(|d| d as i32).unwrap_or(-1)),
        ("Integer", "signum", 1) => Val::Int(args[0].to_i64()?.signum() as i32),
        ("Long", "parseLong", 1) | ("Long", "valueOf", 1) if matches!(args[0], Val::Str(_)) => Val::Long(parse_int(&arg_str(&args, 0)?, 10)?),
        ("Long", "parseLong", 2) => Val::Long(parse_int(&arg_str(&args, 0)?, args[1].to_i64()? as u32)?),
        ("Long", "parseUnsignedLong", 1 | 2) => {
            let s = arg_str(&args, 0)?;
            let radix = if args.len() == 2 { args[1].to_i64()? as u32 } else { 10 };
            let digits = s.strip_prefix('+').unwrap_or(&s);
            if s.starts_with('-') || digits.is_empty() {
                return Err(ScriptErr::new("number_format_exception", if s.starts_with('-') { format!("Illegal leading minus sign on unsigned string {s}.") } else { format!("For input string: \"{s}\"") }));
            }
            let v = u64::from_str_radix(digits, radix).map_err(|_| ScriptErr::new("number_format_exception", format!("String value {s} exceeds range of unsigned long.")))?;
            Val::Long(v as i64)
        }
        ("Long", "toUnsignedString", 1) => Val::string((args[0].to_i64()? as u64).to_string()),
        ("Long", "compareUnsigned", 2) => Val::Int(((args[0].to_i64()? as u64).cmp(&(args[1].to_i64()? as u64))) as i32),
        ("Long", "divideUnsigned", 2) => {
            let d = args[1].to_i64()? as u64;
            if d == 0 {
                return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
            }
            Val::Long(((args[0].to_i64()? as u64) / d) as i64)
        }
        ("Long", "remainderUnsigned", 2) => {
            let d = args[1].to_i64()? as u64;
            if d == 0 {
                return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
            }
            Val::Long(((args[0].to_i64()? as u64) % d) as i64)
        }
        ("Long", "valueOf", 1) => cast(&Ty::Long, args[0].clone())?,
        ("Double" | "Float", "parseDouble" | "parseFloat" | "valueOf", 1) => match &args[0] {
            Val::Str(s) => Val::Double(s.trim().parse::<f64>().map_err(|_| ScriptErr::new("number_format_exception", format!("For input string: \"{s}\"")))?),
            other => cast(&Ty::Double, other.clone())?,
        },
        ("Double" | "Float", "isNaN", 1) => Val::Bool(args[0].to_f64()?.is_nan()),
        ("Double" | "Float", "isInfinite", 1) => Val::Bool(args[0].to_f64()?.is_infinite()),
        ("Double" | "Float", "isFinite", 1) => Val::Bool(args[0].to_f64()?.is_finite()),
        ("Boolean", "parseBoolean", 1) => Val::Bool(matches!(&args[0], Val::Str(s) if s.eq_ignore_ascii_case("true"))),
        ("Boolean", "valueOf", 1) => match &args[0] {
            Val::Bool(b) => Val::Bool(*b),
            Val::Str(s) => Val::Bool(s.eq_ignore_ascii_case("true")),
            _ => Val::Bool(false),
        },
        ("Character", "codePointAt" | "codePointBefore", 2 | 3)
        | ("Character", "codePointCount", 3)
        | ("Character", "offsetByCodePoints", 3 | 5) => {
            let units = if n == 5 || (n == 3 && matches!(method,"codePointAt"|"codePointBefore")) {
                match &args[0] {
                    Val::Array{elem:Ty::Char,items}=>items.borrow().iter().map(|v|match v {Val::Char(c)=>Ok(*c),_=>Err(class_cast("Expected char"))}).collect::<SResult<Vec<_>>>()?,
                    Val::Null=>return Err(npe("character array is null")),
                    _=>return Err(class_cast("Expected char[]")),
                }
            } else {
                match &args[0] {
                    Val::Str(_)|Val::Utf16(_)|Val::Builder(_)=>args[0].string_units(),
                    Val::Null=>return Err(npe("character sequence is null")),
                    _=>return Err(class_cast("Expected CharSequence")),
                }
            };
            if method=="codePointAt" && n==3 {
                let limit=args[2].to_i64()?;
                if limit<0 || limit as usize>units.len() {return Err(out_of_bounds("Invalid character array limit"));}
                super::source::string_method(&args[0],&units[..limit as usize],method,&args[1..2])?
            } else if method=="codePointBefore" && n==3 {
                let index=args[1].to_i64()?;let start=args[2].to_i64()?;
                if start<0 || start as usize>units.len() || index<start || index as usize>units.len() {return Err(out_of_bounds("Invalid character array range"));}
                super::source::string_method(&args[0],&units[start as usize..],method,&[Val::Long(index-start)])?
            } else if method=="offsetByCodePoints" && n==5 {
                let start=args[1].to_i64()?;let count=args[2].to_i64()?;let index=args[3].to_i64()?;
                if start<0 || count<0 || start>units.len() as i64 || count>units.len() as i64-start || index<start || index>start+count {return Err(out_of_bounds("Invalid character array range"));}
                let relative=super::source::string_method(&args[0],&units[start as usize..(start+count) as usize],method,&[Val::Long(index-start),args[4].clone()])?.to_i64()?;
                Val::Int((start+relative) as i32)
            } else {super::source::string_method(&args[0],&units,method,&args[1..])?}
        }
        ("Character", "toUpperCase" | "toLowerCase" | "toTitleCase", 1) => {
            // Painless exposes only Character's char overloads, not Java's
            // similarly named int/code-point overloads.
            let cp = match &args[0] {
                Val::Char(c) => *c as i64,
                other => return Err(super::error::class_cast(format!("Cannot cast {} to char", other.type_name()))),
            };
            let table = match method {
                "toUpperCase" => super::simple_case::SIMPLE_CASE_UPPER,
                "toLowerCase" => super::simple_case::SIMPLE_CASE_LOWER,
                _ => super::simple_case::SIMPLE_CASE_TITLE,
            };
            let mapped = table.binary_search_by_key(&(cp as u32), |p| p.0).map(|i| table[i].1 as i64).unwrap_or(cp);
            Val::Char(mapped as u16)
        }
        ("Character", "isHighSurrogate" | "isLowSurrogate" | "isSurrogate", 1) => {
            let c = args[0].to_index()?;
            Val::Bool(match method {
                "isHighSurrogate" => (0xd800..=0xdbff).contains(&c),
                "isLowSurrogate" => (0xdc00..=0xdfff).contains(&c),
                _ => (0xd800..=0xdfff).contains(&c),
            })
        }
        ("Character", "isSurrogatePair", 2) => Val::Bool(
            (0xd800..=0xdbff).contains(&args[0].to_index()?) && (0xdc00..=0xdfff).contains(&args[1].to_index()?)
        ),
        ("Character", "isValidCodePoint" | "isBmpCodePoint" | "isSupplementaryCodePoint" | "charCount", 1) => {
            let cp = args[0].to_index()?;
            match method {
                "isValidCodePoint" => Val::Bool((0..=0x10ffff).contains(&cp)),
                "isBmpCodePoint" => Val::Bool((0..=0xffff).contains(&cp)),
                "isSupplementaryCodePoint" => Val::Bool((0x10000..=0x10ffff).contains(&cp)),
                _ => Val::Int(if cp >= 0x10000 { 2 } else { 1 }),
            }
        }
        ("Character", "highSurrogate" | "lowSurrogate", 1) => {
            let cp = args[0].to_index()? as i32;
            Val::Char(if method == "highSurrogate" {
                ((cp >> 10).wrapping_add(0xd800 - (0x10000 >> 10))) as u16
            } else { ((cp & 0x3ff) + 0xdc00) as u16 })
        }
        ("Character", "toCodePoint", 2) => {
            let hi=args[0].to_index()? as i32; let lo=args[1].to_index()? as i32;
            Val::Int((hi << 10).wrapping_add(lo).wrapping_add(0x10000 - (0xd800 << 10) - 0xdc00))
        }
        ("Character", m, 1) => {
            if let Val::Char(unit) = &args[0] {
                if m == "valueOf" { return Ok(Some(args[0].clone())); }
                if (0xd800..=0xdfff).contains(unit) {
                    return Ok(Some(match m {
                        "toUpperCase" | "toLowerCase" => args[0].clone(),
                        "getNumericValue" => Val::Int(-1),
                        "isDigit" | "isLetter" | "isLetterOrDigit" | "isWhitespace" | "isUpperCase" | "isLowerCase" | "isAlphabetic" => Val::Bool(false),
                        _ => return Ok(None),
                    }));
                }
            }
            let c = match &args[0] {
                Val::Char(c) => char::from_u32(*c as u32).unwrap_or('\0'),
                Val::Int(i) => char::from_u32(*i as u32).unwrap_or('\0'),
                Val::Str(s) if s.chars().count() == 1 => s.chars().next().unwrap(),
                other => return Err(class_cast(format!("Cannot cast {} to char", other.type_name()))),
            };
            match m {
                "isDigit" => Val::Bool(c.is_ascii_digit() || c.is_numeric()),
                "isLetter" => Val::Bool(c.is_alphabetic()),
                "isLetterOrDigit" => Val::Bool(c.is_alphanumeric()),
                "isWhitespace" => Val::Bool(c.is_whitespace()),
                "isUpperCase" => Val::Bool(c.is_uppercase()),
                "isLowerCase" => Val::Bool(c.is_lowercase()),
                "isAlphabetic" => Val::Bool(c.is_alphabetic()),
                "toUpperCase" => Val::Char(c.to_uppercase().next().unwrap_or(c) as u16),
                "toLowerCase" => Val::Char(c.to_lowercase().next().unwrap_or(c) as u16),
                "getNumericValue" => Val::Int(c.to_digit(36).map(|d| d as i32).unwrap_or(-1)),
                "valueOf" => Val::Char(c as u16),
                _ => return Ok(None),
            }
        }
        ("String", "join", _) if n >= 1 => {
            let sep = match &args[0] {
                Val::Str(_) | Val::Utf16(_) | Val::Builder(_) => args[0].string_units(),
                Val::Null => return Err(npe("delimiter is null")),
                _ => return Err(class_cast("Expected CharSequence delimiter")),
            };
            let values = if n == 2 { iter_values(&args[1])? } else { args[1..].to_vec() };
            let mut joined = Vec::new();
            for (i, value) in values.iter().enumerate() {
                if i > 0 { joined.extend_from_slice(&sep); }
                match value {
                    Val::Str(_) | Val::Utf16(_) | Val::Builder(_) | Val::Null => joined.extend(value.string_units()),
                    _ => return Err(class_cast("Expected CharSequence element")),
                }
            }
            Val::utf16(joined)
        }
        ("String", "format", 2 | 3) => {
            // `format(String, def[])` / `format(Locale, String, def[])`.
            let (fi, arr) = if n == 3 { (1, &args[2]) } else { (0, &args[1]) };
            if args[fi].is_null() { return Err(ScriptErr::new("null_pointer_exception", "format")); }
            let fmt = args[fi].string_units();
            let items = match arr {
                Val::Null => None,
                Val::Array { items, .. } => Some(items.borrow().clone()),
                Val::List(l) => Some(l.borrow().clone()),
                other => Some(vec![other.clone()]),
            };
            Val::utf16(java_format(&fmt, items.as_deref(), if n == 3 { &args[..1] } else { &[] })?)
        }
        ("String", "valueOf", 1) => Val::utf16(args[0].string_units()),
        ("Collections", "sort", 1) | ("Arrays", "sort", 1) => {
            super::methods::call_method(rt, args[0].clone(), "sort", vec![Val::Null])?;
            Val::Null
        }
        ("Collections", "sort", 2) => {
            super::methods::call_method(rt, args[0].clone(), "sort", vec![args[1].clone()])?;
            Val::Null
        }
        ("Collections", "reverse", 1) => {
            if let Val::List(l) = &args[0] {
                l.borrow_mut().reverse();
            }
            Val::Null
        }
        ("Collections", "max" | "min", 1) => {
            let _copy_vals = super::allocation::Allocation::iterable(&args[0])?;
            let vals = iter_values(&args[0])?;
            let mut best: Option<Val> = None;
            for v in vals {
                best = Some(match best {
                    None => v,
                    Some(b) => {
                        let o = v.compare(&b)?;
                        if (method == "max" && o.is_gt()) || (method == "min" && o.is_lt()) { v } else { b }
                    }
                });
            }
            best.ok_or_else(|| ScriptErr::new("no_such_element_exception", "collection is empty"))?
        }
        ("Collections", "frequency", 2) => {
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            Val::Int(iter_values(&args[0])?.iter().filter(|v| v.equals(&args[1])).count() as i32)
        },
        ("Collections", "emptyList", 0) | ("List", "of", 0) => {
            let allocation = super::allocation::Allocation::values(0)?;
            Val::new_list(Vec::new()).allocated(allocation)
        },
        ("Collections", "emptyMap", 0) | ("Map", "of", 0) => {
            let allocation = super::allocation::Allocation::entries(0)?;
            Val::new_map(IndexMap::new()).allocated(allocation)
        },
        ("Collections", "singletonList", 1) => {
            let allocation = super::allocation::Allocation::values(args.capacity())?;
            Val::List(Rc::new(super::value::JList::new(super::value::CollKind::List, args).immutable())).allocated(allocation)
        },
        ("Collections", "singleton", 1) => {
            let allocation = super::allocation::Allocation::values(args.capacity())?;
            Val::Set(Rc::new(super::value::JList::new(super::value::CollKind::LinkedSet, args).immutable())).allocated(allocation)
        },
        ("Collections", "singletonMap", 2) => {
            let allocation = super::allocation::Allocation::entries(1)?;
            let mut m = IndexMap::with_capacity(1);
            m.insert(Key::from_val(&args[0]), args[1].clone());
            Val::Map(Rc::new(super::value::JMap::new(super::value::MapKind::Json, m).immutable())).allocated(allocation)
        }
        ("Collections", "emptySet" | "emptyNavigableSet" | "emptySortedSet", 0) => {
            let allocation = super::allocation::Allocation::values(0)?;
            Val::Set(Rc::new(super::value::JList::new(super::value::CollKind::LinkedSet, Vec::new()).immutable())).allocated(allocation)
        },
        ("Collections", "nCopies", 2) => {
            let n = args[0].to_i64()?;
            if n < 0 {
                return Err(illegal_arg(format!("List length = {n}")));
            }
            let allocation = super::allocation::Allocation::values(n as usize)?;
            Val::List(Rc::new(super::value::JList::new(super::value::CollKind::List, vec![args[1].clone(); n as usize]).immutable())).allocated(allocation)
        }
        ("Collections", "reverseOrder", 0) => super::jmisc::reverse_order(None),
        ("Collections", "reverseOrder", 1) => super::jmisc::reverse_order(if args[0].is_null() { None } else { Some(args[0].clone()) }),
        ("Comparator", "comparing" | "comparingInt" | "comparingLong" | "comparingDouble", 1) => super::jmisc::comparing(args[0].clone(), None)?,
        ("Comparator", "comparing", 2) => super::jmisc::comparing(args[0].clone(), Some(args[1].clone()))?,
        ("Comparator", "naturalOrder", 0) => super::jmisc::natural_order(),
        ("Comparator", "reverseOrder", 0) => super::jmisc::reverse_order(None),
        ("Collections", "swap", 3) => {
            let Val::List(l) = &args[0] else { return Err(class_cast(format!("Cannot cast {} to java.util.List", args[0].type_name()))) };
            let (i, j) = (args[1].to_i64()?, args[2].to_i64()?);
            let len = l.len();
            for k in [i, j] {
                if k < 0 || k >= len as i64 {
                    return Err(out_of_bounds(format!("Index {k} out of bounds for length {len}")).frames(&[
                        "java.base/jdk.internal.util.Preconditions.outOfBounds(Preconditions.java:100)",
                        "java.base/jdk.internal.util.Preconditions.outOfBoundsCheckIndex(Preconditions.java:106)",
                        "java.base/jdk.internal.util.Preconditions.checkIndex(Preconditions.java:302)",
                        "java.base/java.util.Objects.checkIndex(Objects.java:365)",
                        "java.base/java.util.ArrayList.set(ArrayList.java:441)",
                        "java.base/java.util.Collections.swap(Collections.java:548)",
                    ]));
                }
            }
            l.borrow_mut().swap(i as usize, j as usize);
            Val::Null
        }
        ("Collections", "rotate", 2) => {
            if let Val::List(l) = &args[0] {
                let mut v = l.borrow_mut();
                let n = v.len();
                if n > 0 {
                    let d = args[1].to_i64()?.rem_euclid(n as i64) as usize;
                    v.rotate_right(d);
                }
            }
            Val::Null
        }
        ("Collections", "shuffle", 1 | 2) => {
            if let Val::List(l) = &args[0] {
                let mut v = l.borrow_mut();
                // Fisher-Yates with a simple LCG (ES output is random anyway).
                let mut seed: u64 = if n == 2 { args[1].to_i64().unwrap_or(0) as u64 } else { crate::date::now_millis() as u64 };
                for i in (1..v.len()).rev() {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let j = ((seed >> 33) % (i as u64 + 1)) as usize;
                    v.swap(i, j);
                }
            }
            Val::Null
        }
        ("Collections", "fill", 2) => {
            if let Val::List(l) = &args[0] {
                l.register_mutation()?;
                for x in l.borrow_mut().iter_mut() {
                    *x = args[1].clone();
                }
            }
            Val::Null
        }
        ("Collections", "addAll", _) if n >= 1 => {
            let mut changed = false;
            for a in &args[1..] {
                changed |= super::methods::call_method(rt, args[0].clone(), "add", vec![a.clone()])?.as_bool().unwrap_or(true);
            }
            Val::Bool(changed)
        }
        ("Collections", "disjoint", 2) => {
            let _copy_a = super::allocation::Allocation::iterable(&args[0])?;
            let a = iter_values(&args[0])?;
            let _copy_b = super::allocation::Allocation::iterable(&args[1])?;
            let b = iter_values(&args[1])?;
            Val::Bool(!a.iter().any(|x| b.iter().any(|y| x.equals(y))))
        }
        ("Collections", "binarySearch", 2 | 3) => {
            let _copy_v = super::allocation::Allocation::iterable(&args[0])?;
            let v = iter_values(&args[0])?;
            let cmp = if n == 3 { Some(args[2].clone()) } else { None };
            let (mut lo, mut hi) = (0i64, v.len() as i64 - 1);
            let mut found: Option<i64> = None;
            while lo <= hi {
                let mid = (lo + hi) >> 1;
                let o = super::methods::tree_cmp(rt, &cmp, &v[mid as usize], &args[1])?;
                match o {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid - 1,
                    std::cmp::Ordering::Equal => {
                        found = Some(mid);
                        break;
                    }
                }
            }
            Val::Int(found.unwrap_or(-(lo + 1)) as i32)
        }
        ("Collections", "copy", 2) => {
            let _copy_src = super::allocation::Allocation::iterable(&args[1])?;
            let src = iter_values(&args[1])?;
            let Val::List(l) = &args[0] else { return Err(class_cast(format!("Cannot cast {} to java.util.List", args[0].type_name()))) };
            if src.len() > l.len() {
                return Err(out_of_bounds("Source does not fit in dest"));
            }
            l.register_mutation()?;
            let mut d = l.borrow_mut();
            for (i, v) in src.into_iter().enumerate() {
                d[i] = v;
            }
            Val::Null
        }
        ("Collections", "indexOfSubList" | "lastIndexOfSubList", 2) => {
            let _copy_a = super::allocation::Allocation::iterable(&args[0])?;
            let a = iter_values(&args[0])?;
            let _copy_b = super::allocation::Allocation::iterable(&args[1])?;
            let b = iter_values(&args[1])?;
            let mut r = -1i64;
            if b.len() <= a.len() {
                for i in 0..=(a.len() - b.len()) {
                    if a[i..i + b.len()].iter().zip(b.iter()).all(|(x, y)| x.equals(y)) {
                        r = i as i64;
                        if method == "indexOfSubList" {
                            break;
                        }
                    }
                }
            }
            Val::Int(r as i32)
        }
        ("Collections", "replaceAll", 3) => {
            let mut changed = false;
            if let Val::List(l) = &args[0] {
                l.register_mutation()?;
                for x in l.borrow_mut().iter_mut() {
                    if x.equals(&args[1]) {
                        *x = args[2].clone();
                        changed = true;
                    }
                }
            }
            Val::Bool(changed)
        }
        ("Collections", "max" | "min", 2) => {
            let _copy_vals = super::allocation::Allocation::iterable(&args[0])?;
            let vals = iter_values(&args[0])?;
            let cmp = if args[1].is_null() { None } else { Some(args[1].clone()) };
            let mut best: Option<Val> = None;
            for v in vals {
                best = Some(match best {
                    None => v,
                    Some(b) => {
                        let o = super::methods::tree_cmp(rt, &cmp, &v, &b)?;
                        if (method == "max" && o.is_gt()) || (method == "min" && o.is_lt()) { v } else { b }
                    }
                });
            }
            best.ok_or_else(|| ScriptErr::new("no_such_element_exception", ""))?
        }
        ("Collections", "unmodifiableList" | "unmodifiableCollection", 1) => match &args[0] {
            Val::List(l) => {
                let allocation = super::allocation::Allocation::values(l.len())?;
                Val::List(Rc::new(super::value::JList::new(super::value::CollKind::List, l.borrow().clone()).immutable())).allocated(allocation)
            },
            Val::Set(l) => {
                let allocation = super::allocation::Allocation::values(l.len())?;
                Val::Set(Rc::new(super::value::JList::new(super::value::CollKind::LinkedSet, l.ordered().clone()).immutable())).allocated(allocation)
            },
            other => return Err(class_cast(format!("Cannot cast {} to java.util.List", other.type_name()))),
        },
        ("Collections", "unmodifiableSet" | "unmodifiableSortedSet" | "unmodifiableNavigableSet", 1) => match &args[0] {
            Val::Set(l) => {
                let allocation = super::allocation::Allocation::values(l.len())?;
                Val::Set(Rc::new(super::value::JList::new(super::value::CollKind::LinkedSet, l.ordered().clone()).immutable())).allocated(allocation)
            },
            other => return Err(class_cast(format!("Cannot cast {} to java.util.Set", other.type_name()))),
        },
        ("Collections", "unmodifiableMap" | "unmodifiableSortedMap" | "unmodifiableNavigableMap", 1) => match &args[0] {
            Val::Map(m) => {
                let allocation = super::allocation::Allocation::entries(m.borrow().len())?;
                Val::Map(Rc::new(super::value::JMap::new(super::value::MapKind::Json, m.ordered().clone()).immutable())).allocated(allocation)
            },
            other => return Err(class_cast(format!("Cannot cast {} to java.util.Map", other.type_name()))),
        },
        ("Arrays", "asList", _) | ("List", "of", _) => {
            let allocation = super::allocation::Allocation::values(args.capacity())?;
            Val::new_list(args).allocated(allocation)
        },
        ("Set", "of", _) => {
            let allocation = super::allocation::Allocation::values(args.len())?;
            let mut out: Vec<Val> = Vec::with_capacity(args.len());
            for a in args {
                if !out.iter().any(|x| x.equals(&a)) {
                    out.push(a);
                }
            }
            Val::Set(Rc::new(super::value::JList::new(super::value::CollKind::LinkedSet, out).immutable())).allocated(allocation)
        }
        ("Map", "of", _) if n % 2 == 0 => {
            let allocation = super::allocation::Allocation::entries(n / 2)?;
            let mut m = IndexMap::with_capacity(n / 2);
            for p in args.chunks(2) {
                m.insert(Key::from_val(&p[0]), p[1].clone());
            }
            Val::new_map(m).allocated(allocation)
        }
        ("Arrays", "deepToString", 1) => {
            if matches!(&args[0],Val::Array{elem:Ty::Boolean|Ty::Byte|Ty::Short|Ty::Char|Ty::Int|Ty::Long|Ty::Float|Ty::Double,..}) {
                return Err(class_cast("Expected Object[]"));
            }
            fn render(value:&Val, stack:&mut Vec<usize>)->Vec<u16> {
                if let Val::Array{items,..}=value {
                    let address=Rc::as_ptr(items) as usize;
                    if stack.contains(&address) {return "[...]".encode_utf16().collect();}
                    stack.push(address);let mut units=vec![91];
                    for (i,item) in items.borrow().iter().enumerate() {
                        if i>0 {units.extend([44,32]);}
                        units.extend(render(item,stack));
                    }
                    units.push(93);stack.pop();units
                } else {value.string_units()}
            }
            if !matches!(&args[0],Val::Array{..}|Val::Null) {return Err(class_cast("Expected Object[]"));}
            Val::utf16(render(&args[0],&mut Vec::new()))
        }
        ("Arrays", "toString", 1) => match &args[0] {
            Val::Null => Val::string("null".into()),
            Val::Array { items, .. } => {
                let mut units = vec![91];
                for (i, item) in items.borrow().iter().enumerate() {
                    if i > 0 { units.extend([44,32]); }
                    units.extend(item.string_units());
                }
                units.push(93);
                Val::utf16(units)
            }
            other => return Err(ScriptErr::new("class_cast_exception", format!("Expected array, got {}", other.type_name()))),
        },
        ("Objects", "equals", 2) => Val::Bool(args[0].equals(&args[1])),
        ("Objects", "isNull", 1) => Val::Bool(args[0].is_null()),
        ("Objects", "nonNull", 1) => Val::Bool(!args[0].is_null()),
        ("Objects", "requireNonNull", _) if n >= 1 => {
            if args[0].is_null() {
                return Err(npe(args.get(1).map(Val::to_java_string).unwrap_or_default()));
            }
            args[0].clone()
        }
        ("Objects", "hash", _) => Val::Int(args.iter().fold(1i32, |h, v| h.wrapping_mul(31).wrapping_add(v.java_hash()))),
        ("Objects", "hashCode", 1) => Val::Int(args[0].java_hash()),
        ("Instant", "ofEpochMilli", 1) => Val::Instant(args[0].to_i64()?),
        ("Instant", "ofEpochSecond", _) if n >= 1 => Val::Instant(args[0].to_i64()?.saturating_mul(1000) + args.get(1).and_then(Val::as_i64).unwrap_or(0) / 1_000_000),
        ("Instant", "parse", 1) => Val::Instant(parse_date_str(&arg_str(&args, 0)?)?),
        ("Instant", "now", 0) => Val::Instant(crate::date::now_millis()),
        ("ZonedDateTime", "parse", 1) => Val::Date(parse_date_str(&arg_str(&args, 0)?)?),
        ("ZonedDateTime", "ofInstant", 2) => Val::Date(to_millis(&args[0])?),
        ("ZonedDateTime", "now", _) => Val::Date(crate::date::now_millis()),
        ("ZoneId", "of", 1) => args[0].clone(),
        ("DayOfWeek", "of", 1) => Val::DayOfWeek(args[0].to_i64()?.clamp(1, 7) as u8),
        ("Month", "of", 1) => Val::Month(args[0].to_i64()?.clamp(1, 12) as u8),
        ("Json", "load", 1) => {
            let _input=super::allocation::Allocation::text_copy(&args[0])?;
            let text = arg_str(&args, 0)?;
            super::allocation::Scope::collect_dead();
            let _output=super::allocation::Allocation::new();
            _output.reserve((text.len() as u64).saturating_mul(256).saturating_add(256))?;
            let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| illegal_arg(format!("failed to parse JSON: {e}")))?;
            let result=Val::from_json(&v);
            super::allocation::Scope::adopt(&result)?;
            result
        }
        ("Json", "dump", 1 | 2) => {
            let _output=super::allocation::Allocation::text_copy(&args[0])?;
            let v = args[0].to_json();
            let pretty = args.get(1).map(|b| matches!(b, Val::Bool(true))).unwrap_or(false);
            let result=Val::string(if pretty { serde_json::to_string_pretty(&v) } else { serde_json::to_string(&v) }.map_err(|e| illegal_arg(e.to_string()))?);
            super::allocation::Scope::adopt(&result)?;
            result
        }
        ("Collectors", m, _) => Val::Obj(Rc::new(super::jmisc::Obj::Collector { kind: m.to_string(), args })),
        ("Stream", "of", _) => Val::stream_from_iter(args.into_iter().map(Ok))?,
        ("IntStream", "range", 2) | ("IntStream", "rangeClosed", 2) => {
            let a = args[0].to_i64()?;
            let b = args[1].to_i64()? + (method == "rangeClosed") as i64;
            if b - a > MAX_LOOP as i64 {
                return Err(illegal_arg("range too large"));
            }
            Val::stream_from_iter((a..b).map(|i| Ok(Val::Int(i as i32))))?
        }
        ("Optional", "of" | "ofNullable", 1) => Val::Optional(if args[0].is_null() { None } else { Some(Rc::new(args[0].clone())) }),
        ("Optional", "empty", 0) => Val::Optional(None),
        ("Debug", "explain", 1) => {
            return Err(ScriptErr::new("painless_explain_error", format!("{}", args[0].to_java_string())));
        }
        ("System", "currentTimeMillis", 0) => Val::Long(crate::date::now_millis()),
        ("System", "nanoTime", 0) => Val::Long(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0)),
        ("System", "getProperty", 1) => Val::Null,
        ("System", "getenv", 1) => Val::Null,
        _ => return Ok(None),
    };
    Ok(Some(v))
}

fn parse_date_str(s: &str) -> SResult<i64> {
    let f = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").map_err(|e| illegal_arg(e.reason))?;
    f.parse(s).map_err(|_| ScriptErr::new("date_time_parse_exception", format!("Text '{s}' could not be parsed")))
}

/// `Math.*` (also used by the compiler's no-allocation fast path).
pub fn math(method: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    let one = |f: fn(f64) -> f64| -> SResult<Val> { Ok(Val::Double(f(arg_f(args, 0)?))) };
    Ok(Some(match (method, n) {
        ("abs", 1) => {
            // Painless Math.abs always returns double (unlike Java's overloads).
            Val::Double(args[0].to_f64()?.abs())
        }
        ("max" | "min", 2) => {
            // Painless Math.max/min always return double (unlike Java's overloads).
            let (x, y) = (args[0].to_f64()?, args[1].to_f64()?);
            let max = method == "max";
            Val::Double(if x.is_nan() || y.is_nan() { f64::NAN } else if max { x.max(y) } else { x.min(y) })
        }
        ("pow", 2) => Val::Double(arg_f(args, 0)?.powf(arg_f(args, 1)?)),
        ("sqrt", 1) => one(f64::sqrt)?,
        ("cbrt", 1) => one(f64::cbrt)?,
        ("log", 1) => one(f64::ln)?,
        ("log10", 1) => one(f64::log10)?,
        ("log1p", 1) => one(f64::ln_1p)?,
        ("exp", 1) => one(f64::exp)?,
        ("expm1", 1) => one(f64::exp_m1)?,
        ("floor", 1) => one(f64::floor)?,
        ("ceil", 1) => one(f64::ceil)?,
        ("rint", 1) => one(|x| {
            let r = x.round();
            if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }
        })?,
        ("round", 1) => {
            let x = arg_f(args, 0)?;
            Val::Long(if x.is_nan() { 0 } else { (x + 0.5).floor() as i64 })
        }
        ("signum", 1) => one(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { x })?,
        ("sin", 1) => one(f64::sin)?,
        ("cos", 1) => one(f64::cos)?,
        ("tan", 1) => one(f64::tan)?,
        ("asin", 1) => one(f64::asin)?,
        ("acos", 1) => one(f64::acos)?,
        ("atan", 1) => one(f64::atan)?,
        ("sinh", 1) => one(f64::sinh)?,
        ("cosh", 1) => one(f64::cosh)?,
        ("tanh", 1) => one(f64::tanh)?,
        ("toRadians", 1) => one(f64::to_radians)?,
        ("toDegrees", 1) => one(f64::to_degrees)?,
        ("atan2", 2) => Val::Double(arg_f(args, 0)?.atan2(arg_f(args, 1)?)),
        ("hypot", 2) => Val::Double(arg_f(args, 0)?.hypot(arg_f(args, 1)?)),
        ("random", 0) => Val::Double(rand_f64()),
        ("floorDiv", 2) | ("floorMod", 2) => {
            let (a, b) = (args[0].to_i64()?, args[1].to_i64()?);
            if b == 0 {
                return Err(ScriptErr::new("arithmetic_exception", "/ by zero"));
            }
            // Java wraps MIN_VALUE / -1; Rust's ordinary division and
            // remainder panic there. Adjust truncation only when signs differ,
            // avoiding the overflowing `(remainder + divisor) % divisor` form.
            let q = a.wrapping_div(b);
            let rem = a.wrapping_rem(b);
            let adjust = rem != 0 && (a < 0) != (b < 0);
            let r = if method == "floorDiv" { if adjust { q - 1 } else { q } } else if adjust { rem + b } else { rem };
            if matches!((&args[0], &args[1]), (Val::Int(_), Val::Int(_))) { Val::Int(r as i32) } else { Val::Long(r) }
        }
        _ => return Ok(None),
    }))
}

fn rand_f64() -> f64 {
    use std::cell::Cell;
    thread_local!(static S: Cell<u64> = Cell::new(0x9E37_79B9_7F4A_7C15 ^ crate::date::now_millis() as u64));
    S.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 11) as f64 / (1u64 << 53) as f64
    })
}

// ---------------------------------------------------------------------------
// java.util.Formatter (`String.format`).

const FMT_TAIL_A: &str = "java.base/java.util.Formatter.format(Formatter.java:2698)";
const FMT_TAIL_B: &str = "java.base/java.lang.String.format(String.java:4830)";

macro_rules! fmt_frames {
    ($($f:literal),* $(,)?) => {
        &[$(concat!("java.base/java.util.Formatter", $f)),*, FMT_TAIL_A_LIT, FMT_TAIL_B_LIT]
    };
}
const FMT_TAIL_A_LIT: &str = FMT_TAIL_A;
const FMT_TAIL_B_LIT: &str = FMT_TAIL_B;

/// Frames of a failure while a specifier is being constructed (parse time).
macro_rules! spec_frames {
    ($($f:literal),* $(,)?) => {
        fmt_frames!($($f),*,
            "$FormatSpecifier.<init>(Formatter.java:3126)",
            "$FormatSpecifierParser.parse(Formatter.java:2915)",
            ".parse(Formatter.java:2814)",
            ".format(Formatter.java:2744)")
    };
}

const FR_UNKNOWN_CONV: &[&str] = fmt_frames!(
    "$FormatSpecifier.conversion(Formatter.java:3069)",
    "$FormatSpecifier.<init>(Formatter.java:3125)",
    "$FormatSpecifierParser.parse(Formatter.java:2915)",
    ".parse(Formatter.java:2814)",
    ".format(Formatter.java:2744)",
);
const FR_TRAILING_PCT: &[&str] = fmt_frames!(".parse(Formatter.java:2800)", ".format(Formatter.java:2744)");
const FR_MISSING_CONV: &[&str] = fmt_frames!(".parse(Formatter.java:2818)", ".format(Formatter.java:2744)");
const FR_DUP_FLAGS: &[&str] = fmt_frames!(
    "$Flags.parse(Formatter.java:4815)",
    "$FormatSpecifier.flags(Formatter.java:3035)",
    "$FormatSpecifier.<init>(Formatter.java:3111)",
    "$FormatSpecifierParser.parse(Formatter.java:2915)",
    ".parse(Formatter.java:2814)",
    ".format(Formatter.java:2744)",
);
const FR_BAD_INDEX: &[&str] = fmt_frames!(
    "$FormatSpecifier.index(Formatter.java:3022)",
    "$FormatSpecifier.<init>(Formatter.java:3108)",
    "$FormatSpecifierParser.parse(Formatter.java:2915)",
    ".parse(Formatter.java:2814)",
    ".format(Formatter.java:2744)",
);
const FR_INT_PREC: &[&str] = spec_frames!("$FormatSpecifier.checkInteger(Formatter.java:3398)", "$FormatSpecifier.check(Formatter.java:3137)");
const FR_INT_ALT: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkInteger(Formatter.java:3401)",
    "$FormatSpecifier.check(Formatter.java:3137)",
);
const FR_INT_GROUP: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkInteger(Formatter.java:3405)",
    "$FormatSpecifier.check(Formatter.java:3137)",
);
const FR_INT_WIDTH: &[&str] = spec_frames!("$FormatSpecifier.checkNumeric(Formatter.java:3436)", "$FormatSpecifier.checkInteger(Formatter.java:3396)", "$FormatSpecifier.check(Formatter.java:3137)");
const FR_INT_FLAGS: &[&str] = spec_frames!("$FormatSpecifier.checkNumeric(Formatter.java:3441)", "$FormatSpecifier.checkInteger(Formatter.java:3396)", "$FormatSpecifier.check(Formatter.java:3137)");
const FR_FLOAT_WIDTH: &[&str] = spec_frames!("$FormatSpecifier.checkNumeric(Formatter.java:3436)", "$FormatSpecifier.checkFloat(Formatter.java:3416)", "$FormatSpecifier.check(Formatter.java:3139)");
const FR_FLOAT_FLAGS: &[&str] = spec_frames!("$FormatSpecifier.checkNumeric(Formatter.java:3441)", "$FormatSpecifier.checkFloat(Formatter.java:3416)", "$FormatSpecifier.check(Formatter.java:3139)");
const FR_FLOAT_HEX: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkFloat(Formatter.java:3418)",
    "$FormatSpecifier.check(Formatter.java:3139)",
);
const FR_FLOAT_SCI: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkFloat(Formatter.java:3420)",
    "$FormatSpecifier.check(Formatter.java:3139)",
);
const FR_FLOAT_GEN: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkFloat(Formatter.java:3422)",
    "$FormatSpecifier.check(Formatter.java:3139)",
);
const FR_GEN_ALT: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkGeneral(Formatter.java:3366)",
    "$FormatSpecifier.check(Formatter.java:3133)",
);
const FR_GEN_WIDTH: &[&str] = spec_frames!("$FormatSpecifier.checkGeneral(Formatter.java:3370)", "$FormatSpecifier.check(Formatter.java:3133)");
const FR_GEN_FLAGS: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkGeneral(Formatter.java:3372)",
    "$FormatSpecifier.check(Formatter.java:3133)",
);
const FR_CHAR_PREC: &[&str] = spec_frames!("$FormatSpecifier.checkCharacter(Formatter.java:3387)", "$FormatSpecifier.check(Formatter.java:3135)");
const FR_CHAR_FLAGS: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkCharacter(Formatter.java:3389)",
    "$FormatSpecifier.check(Formatter.java:3135)",
);
const FR_CHAR_WIDTH: &[&str] = spec_frames!("$FormatSpecifier.checkCharacter(Formatter.java:3392)", "$FormatSpecifier.check(Formatter.java:3135)");
const FR_DT_PREC: &[&str] = spec_frames!("$FormatSpecifier.checkDateTime(Formatter.java:3375)", "$FormatSpecifier.check(Formatter.java:3131)");
const FR_DT_CONV: &[&str] = spec_frames!("$FormatSpecifier.checkDateTime(Formatter.java:3377)", "$FormatSpecifier.check(Formatter.java:3131)");
const FR_DT_FLAGS: &[&str] = spec_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.checkDateTime(Formatter.java:3379)",
    "$FormatSpecifier.check(Formatter.java:3131)",
);
const FR_DT_WIDTH: &[&str] = spec_frames!("$FormatSpecifier.checkDateTime(Formatter.java:3382)", "$FormatSpecifier.check(Formatter.java:3131)");
const FR_TEXT_PREC: &[&str] = spec_frames!("$FormatSpecifier.checkText(Formatter.java:3446)", "$FormatSpecifier.check(Formatter.java:3141)");
const FR_PCT_FLAGS: &[&str] = spec_frames!("$FormatSpecifier.checkText(Formatter.java:3451)", "$FormatSpecifier.check(Formatter.java:3141)");
const FR_PCT_WIDTH: &[&str] = spec_frames!("$FormatSpecifier.checkText(Formatter.java:3454)", "$FormatSpecifier.check(Formatter.java:3141)");
const FR_NL_WIDTH: &[&str] = spec_frames!("$FormatSpecifier.checkText(Formatter.java:3458)", "$FormatSpecifier.check(Formatter.java:3141)");
const FR_NL_FLAGS: &[&str] = spec_frames!("$FormatSpecifier.checkText(Formatter.java:3460)", "$FormatSpecifier.check(Formatter.java:3141)");
const FR_MISSING_PREV: &[&str] = fmt_frames!(".format(Formatter.java:2753)");
const FR_MISSING_ORD: &[&str] = fmt_frames!(".format(Formatter.java:2760)");
const FR_MISSING_IDX: &[&str] = fmt_frames!(".format(Formatter.java:2766)");
const FR_CONV_INT: &[&str] = fmt_frames!(
    "$FormatSpecifier.failConversion(Formatter.java:4662)",
    "$FormatSpecifier.printInteger(Formatter.java:3200)",
    "$FormatSpecifier.print(Formatter.java:3155)",
    ".format(Formatter.java:2761)",
);
const FR_CONV_FLOAT: &[&str] = fmt_frames!(
    "$FormatSpecifier.failConversion(Formatter.java:4662)",
    "$FormatSpecifier.printFloat(Formatter.java:3213)",
    "$FormatSpecifier.print(Formatter.java:3161)",
    ".format(Formatter.java:2761)",
);
const FR_CONV_CHAR: &[&str] = fmt_frames!(
    "$FormatSpecifier.failConversion(Formatter.java:4662)",
    "$FormatSpecifier.printCharacter(Formatter.java:3276)",
    "$FormatSpecifier.print(Formatter.java:3164)",
    ".format(Formatter.java:2761)",
);
const FR_CODE_POINT: &[&str] = fmt_frames!(
    "$FormatSpecifier.printCharacter(Formatter.java:3274)",
    "$FormatSpecifier.print(Formatter.java:3164)",
    ".format(Formatter.java:2761)",
);
const FR_CONV_DT: &[&str] = fmt_frames!(
    "$FormatSpecifier.failConversion(Formatter.java:4662)",
    "$FormatSpecifier.printDateTime(Formatter.java:3242)",
    "$FormatSpecifier.print(Formatter.java:3148)",
    ".format(Formatter.java:2761)",
);
const FR_STR_ALT: &[&str] = fmt_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.printString(Formatter.java:3288)",
    "$FormatSpecifier.print(Formatter.java:3170)",
    ".format(Formatter.java:2761)",
);
const FR_INT_PRINT_FLAGS: &[&str] = fmt_frames!(
    "$FormatSpecifier.failMismatch(Formatter.java:4658)",
    "$FormatSpecifier.checkBadFlags(Formatter.java:3410)",
    "$FormatSpecifier.print(Formatter.java:3530)",
    "$FormatSpecifier.print(Formatter.java:3496)",
    "$FormatSpecifier.printInteger(Formatter.java:3194)",
    "$FormatSpecifier.print(Formatter.java:3155)",
    ".format(Formatter.java:2761)",
);

// Flag bits, in `Formatter.Flags.toString` order.
const F_LEFT: u8 = 1;
const F_ALT: u8 = 2;
const F_PLUS: u8 = 4;
const F_SPACE: u8 = 8;
const F_ZERO: u8 = 16;
const F_GROUP: u8 = 32;
const F_PAREN: u8 = 64;
const F_PREV: u8 = 128;
const FLAG_CHARS: [(u8, char); 8] = [(F_LEFT, '-'), (F_ALT, '#'), (F_PLUS, '+'), (F_SPACE, ' '), (F_ZERO, '0'), (F_GROUP, ','), (F_PAREN, '('), (F_PREV, '<')];

fn flags_str(f: u8) -> String {
    FLAG_CHARS.iter().filter(|(b, _)| f & b != 0).map(|(_, c)| *c).collect()
}

#[derive(Clone, Copy, PartialEq)]
enum FIdx {
    Ordinary,
    Explicit(usize),
    Prev,
}

struct FSpec {
    idx: FIdx,
    flags: u8,
    width: Option<usize>,
    prec: Option<usize>,
    /// `t`/`T` date-time conversion (then `conv` is the suffix).
    dt: bool,
    upper: bool,
    /// Conversion, lower-cased.
    conv: char,
}

impl FSpec {
    /// `FormatSpecifier.toString()`.
    fn text(&self) -> String {
        let mut s = String::from("%");
        s.push_str(&flags_str(self.flags));
        if let FIdx::Explicit(i) = self.idx {
            s.push_str(&format!("{i}$"));
        }
        if let Some(w) = self.width {
            s.push_str(&w.to_string());
        }
        if let Some(p) = self.prec {
            s.push_str(&format!(".{p}"));
        }
        if self.dt {
            s.push(if self.upper { 'T' } else { 't' });
        }
        s.push(if self.upper { self.conv.to_ascii_uppercase() } else { self.conv });
        s
    }
    fn has(&self, f: u8) -> bool {
        self.flags & f != 0
    }
    fn conv_char(&self) -> char {
        if self.upper { self.conv.to_ascii_uppercase() } else { self.conv }
    }
}

enum FPiece {
    Lit(Vec<u16>),
    Spec(FSpec),
}

fn ferr(kind: &str, msg: impl Into<String>, frames: &'static [&'static str]) -> Box<ScriptErr> {
    ScriptErr::new(kind, msg).frames(frames)
}

fn mismatch(f: u8, c: char, frames: &'static [&'static str]) -> Box<ScriptErr> {
    ferr("format_flags_conversion_mismatch_exception", format!("Conversion = {c}, Flags = {}", flags_str(f)), frames)
}

/// `checkBadFlags`: the first of `bad` present fails.
fn bad_flags(sp: &FSpec, bad: &[u8], frames: &'static [&'static str]) -> SResult<()> {
    for b in bad {
        if sp.has(*b) {
            return Err(mismatch(*b, sp.conv_char(), frames));
        }
    }
    Ok(())
}

fn check_numeric(sp: &FSpec, w_frames: &'static [&'static str], f_frames: &'static [&'static str]) -> SResult<()> {
    if sp.width.is_none() && (sp.has(F_LEFT) || sp.has(F_ZERO)) {
        return Err(ferr("missing_format_width_exception", sp.text(), w_frames));
    }
    if (sp.has(F_PLUS) && sp.has(F_SPACE)) || (sp.has(F_LEFT) && sp.has(F_ZERO)) {
        return Err(ferr("illegal_format_flags_exception", format!("Flags = '{}'", flags_str(sp.flags)), f_frames));
    }
    Ok(())
}

const DT_CONVS: &str = "HIklMSLNpzZsQBbhAaCYyjmdeRTrDFc";

fn check_spec(sp: &FSpec) -> SResult<()> {
    let prec_err = |frames| ferr("illegal_format_precision_exception", sp.prec.unwrap_or(0).to_string(), frames);
    if sp.dt {
        if sp.prec.is_some() {
            return Err(prec_err(FR_DT_PREC));
        }
        if !DT_CONVS.contains(sp.conv) {
            return Err(ferr("unknown_format_conversion_exception", format!("Conversion = 't{}'", sp.conv), FR_DT_CONV));
        }
        bad_flags(sp, &[F_ALT, F_PLUS, F_SPACE, F_ZERO, F_GROUP, F_PAREN], FR_DT_FLAGS)?;
        if sp.width.is_none() && sp.has(F_LEFT) {
            return Err(ferr("missing_format_width_exception", sp.text(), FR_DT_WIDTH));
        }
        return Ok(());
    }
    match sp.conv {
        'b' | 'h' | 's' => {
            if sp.conv != 's' && sp.has(F_ALT) {
                return Err(mismatch(F_ALT, sp.conv_char(), FR_GEN_ALT));
            }
            if sp.width.is_none() && sp.has(F_LEFT) {
                return Err(ferr("missing_format_width_exception", sp.text(), FR_GEN_WIDTH));
            }
            bad_flags(sp, &[F_PLUS, F_SPACE, F_ZERO, F_GROUP, F_PAREN], FR_GEN_FLAGS)
        }
        'c' => {
            if sp.prec.is_some() {
                return Err(prec_err(FR_CHAR_PREC));
            }
            bad_flags(sp, &[F_ALT, F_PLUS, F_SPACE, F_ZERO, F_GROUP, F_PAREN], FR_CHAR_FLAGS)?;
            if sp.width.is_none() && sp.has(F_LEFT) {
                return Err(ferr("missing_format_width_exception", sp.text(), FR_CHAR_WIDTH));
            }
            Ok(())
        }
        'd' | 'o' | 'x' => {
            check_numeric(sp, FR_INT_WIDTH, FR_INT_FLAGS)?;
            if sp.prec.is_some() {
                return Err(prec_err(FR_INT_PREC));
            }
            if sp.conv == 'd' {
                bad_flags(sp, &[F_ALT], FR_INT_ALT)
            } else {
                bad_flags(sp, &[F_GROUP], FR_INT_GROUP)
            }
        }
        'e' | 'f' | 'g' | 'a' => {
            check_numeric(sp, FR_FLOAT_WIDTH, FR_FLOAT_FLAGS)?;
            match sp.conv {
                'a' => bad_flags(sp, &[F_PAREN, F_GROUP], FR_FLOAT_HEX),
                'e' => bad_flags(sp, &[F_GROUP], FR_FLOAT_SCI),
                'g' => bad_flags(sp, &[F_ALT], FR_FLOAT_GEN),
                _ => Ok(()),
            }
        }
        '%' => {
            if sp.prec.is_some() {
                return Err(prec_err(FR_TEXT_PREC));
            }
            if sp.flags & !F_LEFT != 0 {
                return Err(ferr("illegal_format_flags_exception", format!("Flags = '{}'", flags_str(sp.flags)), FR_PCT_FLAGS));
            }
            if sp.width.is_none() && sp.has(F_LEFT) {
                return Err(ferr("missing_format_width_exception", sp.text(), FR_PCT_WIDTH));
            }
            Ok(())
        }
        'n' => {
            if sp.prec.is_some() {
                return Err(prec_err(FR_TEXT_PREC));
            }
            if let Some(w) = sp.width {
                return Err(ferr("illegal_format_width_exception", w.to_string(), FR_NL_WIDTH));
            }
            if sp.flags != 0 {
                return Err(ferr("illegal_format_flags_exception", format!("Flags = '{}'", flags_str(sp.flags)), FR_NL_FLAGS));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_format(fmt: &[u16]) -> SResult<Vec<FPiece>> {
    // Format syntax is ASCII; literal spans retain the original UTF16 units.
    let cs: Vec<char> = fmt.iter().map(|&u| char::from_u32(u as u32).unwrap_or(char::REPLACEMENT_CHARACTER)).collect();
    let mut out = Vec::new();
    let mut lit = Vec::new();
    let mut i = 0;
    let digits = |j: &mut usize| -> Option<usize> {
        let st = *j;
        while *j < cs.len() && cs[*j].is_ascii_digit() {
            *j += 1;
        }
        if *j == st { None } else { Some(cs[st..*j].iter().collect::<String>().parse::<usize>().unwrap_or(usize::MAX)) }
    };
    while i < cs.len() {
        if cs[i] != '%' {
            lit.push(fmt[i]);
            i += 1;
            continue;
        }
        if i + 1 >= cs.len() {
            return Err(ferr("unknown_format_conversion_exception", "Conversion = '%'", FR_TRAILING_PCT));
        }
        if !lit.is_empty() {
            out.push(FPiece::Lit(std::mem::take(&mut lit)));
        }
        let first = cs[i + 1];
        let mut j = i + 1;
        // Argument index `n$` (digits not followed by `$` are a width / `0` flag).
        let mut idx = FIdx::Ordinary;
        {
            let mut k = j;
            if let Some(n) = digits(&mut k) {
                if k < cs.len() && cs[k] == '$' {
                    if n == 0 {
                        return Err(ferr("illegal_format_argument_index_exception", "Illegal format argument index = 0", FR_BAD_INDEX));
                    }
                    idx = FIdx::Explicit(n);
                    j = k + 1;
                }
            }
        }
        let mut flags = 0u8;
        while j < cs.len() {
            let Some((b, _)) = FLAG_CHARS.iter().find(|(_, c)| *c == cs[j]) else { break };
            if flags & b != 0 {
                return Err(ferr("duplicate_format_flags_exception", format!("Flags = '{}'", cs[j]), FR_DUP_FLAGS));
            }
            flags |= b;
            j += 1;
        }
        if flags & F_PREV != 0 {
            idx = FIdx::Prev;
        }
        let width = digits(&mut j);
        let mut prec = None;
        if j < cs.len() && cs[j] == '.' {
            let mut k = j + 1;
            match digits(&mut k) {
                Some(p) => {
                    prec = Some(p);
                    j = k;
                }
                None => return Err(ferr("unknown_format_conversion_exception", "Conversion = '.'", FR_MISSING_CONV)),
            }
        }
        if j >= cs.len() {
            return Err(ferr("unknown_format_conversion_exception", format!("Conversion = '{first}'"), FR_MISSING_CONV));
        }
        let c = cs[j];
        j += 1;
        let (dt, upper, conv) = if c == 't' || c == 'T' {
            if j >= cs.len() {
                return Err(ferr("unknown_format_conversion_exception", format!("Conversion = '{c}'"), FR_UNKNOWN_CONV));
            }
            let d = cs[j];
            j += 1;
            (true, c == 'T', d)
        } else {
            if !"bBhHsScCdoxXeEfgGaA%n".contains(c) {
                return Err(ferr("unknown_format_conversion_exception", format!("Conversion = '{c}'"), FR_UNKNOWN_CONV));
            }
            (false, c.is_ascii_uppercase(), c.to_ascii_lowercase())
        };
        // `Formatter.FormatSpecifier`: a width or precision past int range
        // is `Integer.MIN_VALUE` in the exception.
        if width.is_some_and(|w| w > i32::MAX as usize) {
            return Err(ferr("illegal_format_width_exception", i32::MIN.to_string(), &[]));
        }
        if prec.is_some_and(|p| p > i32::MAX as usize) {
            return Err(ferr("illegal_format_precision_exception", i32::MIN.to_string(), &[]));
        }
        let sp = FSpec { idx, flags, width, prec, dt, upper, conv };
        check_spec(&sp)?;
        // Padding (and numeric precision digits) are allocated; huge ones
        // fail as Java's OutOfMemoryError would instead of aborting.
        let padded = sp.width.unwrap_or(0).max(if matches!(conv, 'e' | 'f' | 'g' | 'a') && !dt { sp.prec.unwrap_or(0) } else { 0 });
        super::allocation::check_string_bytes(padded as u64 * 2)?;
        out.push(FPiece::Spec(sp));
        i = j;
    }
    if !lit.is_empty() {
        out.push(FPiece::Lit(lit));
    }
    Ok(out)
}

/// `Formatter.format(fmt, args)`; `args` is `None` for a null array.
pub fn java_format(fmt: &[u16], args: Option<&[Val]>, locale: &[Val]) -> SResult<Vec<u16>> {
    let pieces = parse_format(fmt)?;
    let mut out = Vec::new();
    let mut ordinary = 0usize;
    let mut last: Option<usize> = None;
    let len = args.map(|a| a.len()).unwrap_or(0);
    for p in pieces {
        let sp = match p {
            FPiece::Lit(s) => {
                out.extend(s);
                continue;
            }
            FPiece::Spec(sp) => sp,
        };
        if !sp.dt && (sp.conv == '%' || sp.conv == 'n') {
            let s = if sp.conv == 'n' { "\n".to_string() } else { "%".to_string() };
            out.extend(justify(&sp, s).encode_utf16());
            continue;
        }
        let ai = match sp.idx {
            FIdx::Prev => match last {
                Some(l) if args.is_none() || l < len => l,
                _ => return Err(ferr("missing_format_argument_exception", format!("Format specifier '{}'", sp.text()), FR_MISSING_PREV)),
            },
            FIdx::Ordinary => {
                let l = ordinary;
                ordinary += 1;
                if args.is_some() && l >= len {
                    return Err(ferr("missing_format_argument_exception", format!("Format specifier '{}'", sp.text()), FR_MISSING_ORD));
                }
                l
            }
            FIdx::Explicit(n) => {
                let l = n - 1;
                if args.is_some() && l >= len {
                    return Err(ferr("missing_format_argument_exception", format!("Format specifier '{}'", sp.text()), FR_MISSING_IDX));
                }
                l
            }
        };
        last = Some(ai);
        let arg = args.and_then(|a| a.get(ai)).cloned().unwrap_or(Val::Null);
        out.extend(format_one_units(&sp, &arg, locale)?);
    }
    Ok(out)
}

fn justify(sp: &FSpec, s: String) -> String {
    let n = s.chars().count();
    match sp.width {
        Some(w) if n < w => {
            let pad = " ".repeat(w - n);
            if sp.has(F_LEFT) { s + &pad } else { pad + &s }
        }
        _ => s,
    }
}

fn upper_if(sp: &FSpec, s: String) -> String {
    if sp.upper { s.to_uppercase() } else { s }
}

fn is_big(v: &Val, class: &str) -> bool {
    matches!(v, Val::Big(_)) && super::jbig::instance_of(v, class) == Some(true)
}

fn conv_err(sp: &FSpec, v: &Val, frames: &'static [&'static str]) -> Box<ScriptErr> {
    ferr("illegal_format_conversion_exception", format!("{} != {}", sp.conv_char(), v.type_name()), frames)
}

fn format_one_units(sp: &FSpec, v: &Val, locale: &[Val]) -> SResult<Vec<u16>> {
    if sp.dt || !matches!(sp.conv, 's' | 'c') {
        return Ok(format_one(sp, v)?.encode_utf16().collect());
    }
    let mut units = if sp.conv == 's' {
        if sp.has(F_ALT) { return Err(mismatch(F_ALT, 's', FR_STR_ALT)); }
        let mut units = v.string_units();
        if let Some(precision) = sp.prec { units.truncate(precision); }
        units
    } else {
        match v {
            Val::Null => "null".encode_utf16().collect(),
            Val::Char(c) => vec![*c],
            Val::Int(cp) if (0..=0x10ffff).contains(cp) => {
                if *cp <= 0xffff { vec![*cp as u16] }
                else { let n = (*cp as u32) - 0x10000; vec![0xd800 + (n >> 10) as u16, 0xdc00 + (n & 1023) as u16] }
            }
            Val::Int(cp) => return Err(ferr("illegal_format_code_point_exception", format!("Code point = 0x{:x}", *cp as u32), FR_CODE_POINT)),
            other => return Err(conv_err(sp, other, FR_CONV_CHAR)),
        }
    };
    if sp.upper { units = super::locale_case::map(&units, true, locale)?.string_units(); }
    if let Some(width) = sp.width {
        if width > units.len() {
            let padding = vec![32; width - units.len()];
            if sp.has(F_LEFT) { units.extend(padding); }
            else { let mut padded = padding; padded.extend(units); units = padded; }
        }
    }
    Ok(units)
}

fn format_one(sp: &FSpec, v: &Val) -> SResult<String> {
    if sp.dt {
        return format_dt(sp, v);
    }
    let s = match sp.conv {
        'b' => {
            let s = match v {
                Val::Null => "false".to_string(),
                Val::Bool(b) => b.to_string(),
                _ => "true".to_string(),
            };
            truncate(s, sp.prec)
        }
        'h' => {
            let s = if v.is_null() { "null".to_string() } else { format!("{:x}", v.java_hash() as u32) };
            truncate(s, sp.prec)
        }
        's' => {
            if sp.has(F_ALT) {
                return Err(mismatch(F_ALT, 's', FR_STR_ALT));
            }
            let s = if v.is_null() { "null".to_string() } else { v.to_java_string() };
            truncate(s, sp.prec)
        }
        'c' => match v {
            Val::Null => "null".to_string(),
            Val::Char(c) => String::from_utf16_lossy(&[*c]),
            Val::Int(i) => match char::from_u32(*i as u32) {
                Some(c) if *i >= 0 => c.to_string(),
                _ => return Err(ferr("illegal_format_code_point_exception", format!("Code point = 0x{:x}", *i as u32), FR_CODE_POINT)),
            },
            other => return Err(conv_err(sp, other, FR_CONV_CHAR)),
        },
        'd' | 'o' | 'x' => return format_int(sp, v),
        _ => return format_float(sp, v),
    };
    Ok(justify(sp, upper_if(sp, s)))
}

fn truncate(s: String, prec: Option<usize>) -> String {
    match prec {
        Some(p) if s.chars().count() > p => s.chars().take(p).collect(),
        _ => s,
    }
}

fn leading_sign(sp: &FSpec, neg: bool, sb: &mut String) {
    if !neg {
        if sp.has(F_PLUS) {
            sb.push('+');
        } else if sp.has(F_SPACE) {
            sb.push(' ');
        }
    } else if sp.has(F_PAREN) {
        sb.push('(');
    } else {
        sb.push('-');
    }
}

fn adjust_width(width: Option<usize>, sp: &FSpec, neg: bool) -> Option<usize> {
    match width {
        Some(w) if neg && sp.has(F_PAREN) => Some(w.saturating_sub(1)),
        w => w,
    }
}

/// `localizedMagnitude`: digits (optionally with one `.`), grouped when
/// `,` is set, zero-padded so `sb` reaches `width`.
fn magnitude(sp: &FSpec, sb: &mut String, value: &str, width: Option<usize>) {
    let begin = sb.len();
    let dot = value.find('.').unwrap_or(value.len());
    let int_part = &value[..dot];
    let n = int_part.len();
    for (k, ch) in int_part.chars().enumerate() {
        sb.push(ch);
        if sp.has(F_GROUP) && k + 1 < n && (n - 1 - k) % 3 == 0 {
            sb.push(',');
        }
    }
    sb.push_str(&value[dot..]);
    if let Some(w) = width {
        if sp.has(F_ZERO) {
            let have = sb.chars().count();
            if have < w {
                sb.insert_str(begin, &"0".repeat(w - have));
            }
        }
    }
}

fn format_int(sp: &FSpec, v: &Val) -> SResult<String> {
    // (negative, decimal magnitude, two's-complement bits) of the argument.
    let (neg, mag, bits): (bool, String, Option<(u64, u32)>) = match v {
        Val::Null => return Ok(justify(sp, "null".to_string())),
        Val::Int(i) => (*i < 0, (*i as i64).unsigned_abs().to_string(), Some((*i as u32 as u64, 32))),
        Val::Long(l) => (*l < 0, l.unsigned_abs().to_string(), Some((*l as u64, 64))),
        b if is_big(b, "BigInteger") => {
            let s = b.to_java_string();
            match s.strip_prefix('-') {
                Some(m) => (true, m.to_string(), None),
                None => (false, s, None),
            }
        }
        other => return Err(conv_err(sp, other, FR_CONV_INT)),
    };
    let mut sb = String::new();
    if sp.conv == 'd' {
        leading_sign(sp, neg, &mut sb);
        magnitude(sp, &mut sb, &mag, adjust_width(sp.width, sp, neg));
        if neg && sp.has(F_PAREN) {
            sb.push(')');
        }
        return Ok(justify(sp, sb));
    }
    let radix = if sp.conv == 'o' { 8 } else { 16 };
    let prefix = match (sp.has(F_ALT), radix) {
        (false, _) => "",
        (true, 8) => "0",
        (true, _) => if sp.upper { "0X" } else { "0x" },
    };
    match bits {
        Some((u, _)) => {
            bad_flags(sp, &[F_PAREN, F_SPACE, F_PLUS], FR_INT_PRINT_FLAGS)?;
            let digits = if radix == 8 { format!("{u:o}") } else { format!("{u:x}") };
            sb.push_str(prefix);
            if let (Some(w), true) = (sp.width, sp.has(F_ZERO)) {
                let len = digits.len() + prefix.len();
                if w > len {
                    sb.push_str(&"0".repeat(w - len));
                }
            }
            sb.push_str(&if sp.upper { digits.to_uppercase() } else { digits });
        }
        None => {
            leading_sign(sp, neg, &mut sb);
            let digits = dec_to_radix(&mag, radix);
            let mut len = digits.len() + sb.chars().count() + prefix.len();
            if neg && sp.has(F_PAREN) {
                len += 1;
            }
            sb.push_str(prefix);
            if let (Some(w), true) = (sp.width, sp.has(F_ZERO)) {
                if w > len {
                    sb.push_str(&"0".repeat(w - len));
                }
            }
            sb.push_str(&if sp.upper { digits.to_uppercase() } else { digits });
            if neg && sp.has(F_PAREN) {
                sb.push(')');
            }
        }
    }
    Ok(justify(sp, sb))
}

/// Non-negative decimal digit string → digits in `radix`.
fn dec_to_radix(dec: &str, radix: u32) -> String {
    let mut num: Vec<u32> = dec.bytes().filter(|b| b.is_ascii_digit()).map(|b| (b - b'0') as u32).collect();
    let mut out = Vec::new();
    while num.iter().any(|d| *d != 0) {
        let mut rem = 0u32;
        let mut next = Vec::with_capacity(num.len());
        for d in &num {
            let cur = rem * 10 + d;
            let q = cur / radix;
            rem = cur % radix;
            if !(next.is_empty() && q == 0) {
                next.push(q);
            }
        }
        out.push(std::char::from_digit(rem, radix).unwrap_or('0'));
        num = next;
    }
    if out.is_empty() {
        return "0".to_string();
    }
    out.iter().rev().collect()
}

/// Decimal digits `d1 d2 ...` (no leading zeros) and exponent `e` with
/// value = 0.d1d2... × 10^e; `None` for zero.
fn parse_decimal(s: &str) -> Option<(Vec<u8>, i64)> {
    let s = s.trim_start_matches(['-', '+']);
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i64>().unwrap_or(0)),
        None => (s, 0),
    };
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    let mut digits: Vec<u8> = ip.bytes().chain(fp.bytes()).map(|b| b - b'0').collect();
    let mut e = ip.len() as i64 + exp;
    let lead = digits.iter().take_while(|d| **d == 0).count();
    if lead == digits.len() {
        return None;
    }
    digits.drain(..lead);
    e -= lead as i64;
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }
    Some((digits, e))
}

/// `FormattedFloatingDecimal.applyPrecision`: round half-up keeping `keep`
/// digits; returns the new exponent.
fn apply_precision(digits: &mut Vec<u8>, e: i64, keep: i64) -> i64 {
    if keep < 0 || keep >= digits.len() as i64 {
        return e;
    }
    if keep == 0 {
        return if digits[0] >= 5 {
            *digits = vec![1];
            e + 1
        } else {
            *digits = vec![0];
            e
        };
    }
    let k = keep as usize;
    let up = digits[k] >= 5;
    digits.truncate(k);
    if up {
        let mut i = k;
        loop {
            if i == 0 {
                digits.insert(0, 1);
                digits.truncate(k);
                return e + 1;
            }
            i -= 1;
            if digits[i] == 9 {
                digits[i] = 0;
            } else {
                digits[i] += 1;
                break;
            }
        }
    }
    e
}

/// Plain decimal with exactly `prec` fraction digits.
fn fixed(digits: &[u8], e: i64, prec: usize) -> String {
    let d = |i: i64| -> char { if i >= 0 && (i as usize) < digits.len() { (b'0' + digits[i as usize]) as char } else { '0' } };
    let mut s = String::new();
    if e <= 0 {
        s.push('0');
    } else {
        for i in 0..e {
            s.push(d(i));
        }
    }
    if prec > 0 {
        s.push('.');
        for k in 0..prec as i64 {
            s.push(d(e + k));
        }
    }
    s
}

fn exp_str(x: i64) -> String {
    format!("{}{:02}", if x < 0 { '-' } else { '+' }, x.unsigned_abs())
}

fn format_float(sp: &FSpec, v: &Val) -> SResult<String> {
    let (neg, decimal, dval): (bool, Option<(Vec<u8>, i64)>, Option<f64>) = match v {
        Val::Null => return Ok(justify(sp, "null".to_string())),
        Val::Double(d) => {
            if d.is_nan() {
                return Ok(justify(sp, upper_if(sp, "NaN".to_string())));
            }
            let neg = d.is_sign_negative();
            if d.is_infinite() {
                let mut sb = String::new();
                leading_sign(sp, neg, &mut sb);
                sb.push_str("Infinity");
                if neg && sp.has(F_PAREN) {
                    sb.push(')');
                }
                return Ok(justify(sp, upper_if(sp, sb)));
            }
            let a = d.abs();
            (neg, if a == 0.0 { None } else { parse_decimal(&format!("{a:e}")) }, Some(a))
        }
        b if is_big(b, "BigDecimal") && sp.conv != 'a' => {
            let s = b.to_java_string();
            (s.starts_with('-'), parse_decimal(&s), None)
        }
        other => return Err(conv_err(sp, other, FR_CONV_FLOAT)),
    };
    let mut sb = String::new();
    leading_sign(sp, neg, &mut sb);
    let echar = if sp.upper { 'E' } else { 'e' };
    match sp.conv {
        'e' => {
            let prec = sp.prec.unwrap_or(6);
            let (mant, ex) = match decimal {
                None => (fixed(&[0], 1, prec), "+00".to_string()),
                Some((mut ds, e)) => {
                    let e = apply_precision(&mut ds, e, prec as i64 + 1);
                    (fixed(&ds, 1, prec), exp_str(e - 1))
                }
            };
            let mant = if sp.has(F_ALT) && prec == 0 { mant + "." } else { mant };
            let w = sp.width.map(|w| w.saturating_sub(ex.len() + 1));
            magnitude(sp, &mut sb, &mant, adjust_width(w, sp, neg));
            sb.push(echar);
            sb.push_str(&ex);
        }
        'f' => {
            let prec = sp.prec.unwrap_or(6);
            let mant = match decimal {
                None => fixed(&[0], 1, prec),
                Some((mut ds, e)) => {
                    let e2 = apply_precision(&mut ds, e, e + prec as i64);
                    fixed(&ds, e2, prec)
                }
            };
            let mant = if sp.has(F_ALT) && prec == 0 { mant + "." } else { mant };
            magnitude(sp, &mut sb, &mant, adjust_width(sp.width, sp, neg));
        }
        'g' => {
            let prec = match sp.prec {
                None => 6,
                Some(0) => 1,
                Some(p) => p,
            };
            match decimal {
                None => {
                    let mant = fixed(&[0], 1, prec - 1);
                    magnitude(sp, &mut sb, &mant, adjust_width(sp.width, sp, neg));
                }
                Some((mut ds, e)) => {
                    let e = apply_precision(&mut ds, e, prec as i64);
                    let x = e - 1;
                    if x < -4 || x >= prec as i64 {
                        let mant = fixed(&ds, 1, prec - 1);
                        let ex = exp_str(x);
                        let w = sp.width.map(|w| w.saturating_sub(ex.len() + 1));
                        magnitude(sp, &mut sb, &mant, adjust_width(w, sp, neg));
                        sb.push(echar);
                        sb.push_str(&ex);
                    } else {
                        let mant = fixed(&ds, e, (prec as i64 - (x + 1)).max(0) as usize);
                        magnitude(sp, &mut sb, &mant, adjust_width(sp.width, sp, neg));
                    }
                }
            }
        }
        _ => {
            // 'a': hexadecimal floating point.
            let d = dval.unwrap_or(0.0);
            let prec = match sp.prec {
                None => 0,
                Some(0) => 1,
                Some(p) => p,
            };
            let s = hex_double(d, prec);
            let pi = s.find('p').unwrap_or(s.len());
            sb.push_str(if sp.upper { "0X" } else { "0x" });
            if let (Some(w), true) = (sp.width, sp.has(F_ZERO)) {
                let lead = if sp.has(F_SPACE) || sp.has(F_PLUS) || neg { 3 } else { 2 };
                let n = w.saturating_sub(s.len() + lead);
                sb.push_str(&"0".repeat(n));
            }
            let mut va = s[..pi].to_string();
            if prec != 0 {
                let frac = va.split_once('.').map(|x| x.1.len()).unwrap_or(0);
                if frac < prec {
                    va.push_str(&"0".repeat(prec - frac));
                }
            }
            sb.push_str(&if sp.upper { va.to_uppercase() } else { va });
            sb.push(if sp.upper { 'P' } else { 'p' });
            sb.push_str(&s[pi + 1..]);
        }
    }
    if neg && sp.has(F_PAREN) {
        sb.push(')');
    }
    Ok(justify(sp, sb))
}

/// `Double.toHexString(d)` without the `0x` (rounded to `prec` hex digits
/// when 0 < prec < 13), for finite non-negative `d`.
fn hex_double(d: f64, prec: usize) -> String {
    if d == 0.0 {
        return "0.0p0".to_string();
    }
    let bits = d.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let mut man = bits & ((1u64 << 52) - 1);
    let strip = |m: u64| -> String {
        let s = format!("{m:013x}");
        let t = s.trim_end_matches('0');
        if t.is_empty() { "0".to_string() } else { t.to_string() }
    };
    if exp == 0 {
        return format!("0.{}p-1022", strip(man));
    }
    let mut e = exp - 1023;
    if prec > 0 && prec < 13 {
        let shift = 52 - 4 * prec as u32;
        let full = man | (1u64 << 52);
        let kept = full >> shift;
        let rem = full & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        let mut r = kept;
        if rem > half || (rem == half && kept & 1 == 1) {
            r += 1;
        }
        if r >> (53 - shift) != 0 {
            r >>= 1;
            e += 1;
        }
        man = (r << shift) & ((1u64 << 52) - 1);
    }
    format!("1.{}p{}", strip(man), e)
}

fn format_dt(sp: &FSpec, v: &Val) -> SResult<String> {
    use chrono::{Datelike, Timelike};
    let ms = match v {
        Val::Null => return Ok(justify(sp, upper_if(sp, "null".to_string()))),
        Val::Long(l) => *l,
        Val::Date(m) | Val::Instant(m) => *m,
        other => return Err(ferr("illegal_format_conversion_exception", format!("{} != {}", sp.conv, other.type_name()), FR_CONV_DT)),
    };
    let Some(t) = chrono::DateTime::from_timestamp_millis(ms) else {
        return Err(illegal_arg(format!("invalid epoch millis [{ms}]")));
    };
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    let h12 = |h: u32| if h % 12 == 0 { 12 } else { h % 12 };
    fn one(t: &chrono::DateTime<chrono::Utc>, c: char) -> String {
        use chrono::{Datelike, Timelike};
        const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
        const DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
        let h12 = |h: u32| if h % 12 == 0 { 12 } else { h % 12 };
        match c {
            'H' => format!("{:02}", t.hour()),
            'I' => format!("{:02}", h12(t.hour())),
            'M' => format!("{:02}", t.minute()),
            'S' => format!("{:02}", t.second()),
            'p' => (if t.hour() < 12 { "am" } else { "pm" }).to_string(),
            'Z' => "UTC".to_string(),
            'Y' => format!("{:04}", t.year()),
            'y' => format!("{:02}", t.year().rem_euclid(100)),
            'm' => format!("{:02}", t.month()),
            'd' => format!("{:02}", t.day()),
            'b' => MONTHS[t.month0() as usize][..3].to_string(),
            'a' => DAYS[t.weekday().num_days_from_monday() as usize][..3].to_string(),
            _ => String::new(),
        }
    }
    let s = match sp.conv {
        'k' => t.hour().to_string(),
        'l' => h12(t.hour()).to_string(),
        'L' => format!("{:03}", t.timestamp_subsec_millis()),
        'N' => format!("{:09}", t.timestamp_subsec_nanos()),
        'z' => "+0000".to_string(),
        's' => ms.div_euclid(1000).to_string(),
        'Q' => ms.to_string(),
        'B' => MONTHS[t.month0() as usize].to_string(),
        'h' => one(&t, 'b'),
        'A' => DAYS[t.weekday().num_days_from_monday() as usize].to_string(),
        'C' => format!("{:02}", t.year().div_euclid(100)),
        'j' => format!("{:03}", t.ordinal()),
        'e' => t.day().to_string(),
        'R' => format!("{}:{}", one(&t, 'H'), one(&t, 'M')),
        'T' => format!("{}:{}:{}", one(&t, 'H'), one(&t, 'M'), one(&t, 'S')),
        'r' => format!("{}:{}:{} {}", one(&t, 'I'), one(&t, 'M'), one(&t, 'S'), one(&t, 'p').to_uppercase()),
        'D' => format!("{}/{}/{}", one(&t, 'm'), one(&t, 'd'), one(&t, 'y')),
        'F' => format!("{}-{}-{}", one(&t, 'Y'), one(&t, 'm'), one(&t, 'd')),
        'c' => format!("{} {} {} {}:{}:{} {} {}", one(&t, 'a'), one(&t, 'b'), one(&t, 'd'), one(&t, 'H'), one(&t, 'M'), one(&t, 'S'), one(&t, 'Z'), one(&t, 'Y')),
        c => one(&t, c),
    };
    Ok(justify(sp, upper_if(sp, s)))
}
