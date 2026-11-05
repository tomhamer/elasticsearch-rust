//! Unboxed numeric fast path.
//!
//! Pure numeric expressions (literals, `_score`, locals, `params.x`,
//! single-valued doc-value reads, `+ - * / %`, unary minus, numeric casts,
//! `Math.*`, comparisons) are compiled a second time into closures that
//! return plain `i32` / `i64` / `f64` (or a small `Copy` [`Num`] when the
//! type is only known at run time), with no [`Val`] construction or drop
//! per node. Every such closure returns `None` instead of failing: missing
//! or multi-valued doc values, non-numeric runtime values, `/ by zero`.
//! The caller then evaluates the general closure of the same expression,
//! which yields exactly the value or error the script would have produced
//! anyway (all nodes accepted here are side-effect free, so re-evaluation
//! is invisible). Promotion and overflow follow [`binary`] / [`cast`]
//! exactly (Java binary numeric promotion, wrapping integer arithmetic,
//! saturating double → integer casts with `NaN` → 0).

use super::*;

/// A number whose type is only known at run time.
#[derive(Clone, Copy)]
pub(crate) enum Num {
    I(i32),
    L(i64),
    D(f64),
}

impl Num {
    #[inline(always)]
    fn of(v: &Val) -> Option<Num> {
        match v {
            Val::Int(i) => Some(Num::I(*i)),
            Val::Long(l) => Some(Num::L(*l)),
            Val::Double(d) => Some(Num::D(*d)),
            _ => None,
        }
    }
    #[inline(always)]
    fn f(self) -> f64 {
        match self {
            Num::I(i) => i as f64,
            Num::L(l) => l as f64,
            Num::D(d) => d,
        }
    }
    #[inline(always)]
    pub(super) fn val(self) -> Val {
        match self {
            Num::I(i) => Val::Int(i),
            Num::L(l) => Val::Long(l),
            Num::D(d) => Val::Double(d),
        }
    }
}

pub(super) type F<T> = Box<dyn Fn(&mut Rt) -> Option<T> + Send + Sync>;

/// A compiled numeric expression, by static result type.
pub(super) enum NumFn {
    I(F<i32>),
    L(F<i64>),
    D(F<f64>),
    /// Type decided per evaluation (`def` sources).
    N(F<Num>),
}

impl NumFn {
    fn into_n(self) -> F<Num> {
        match self {
            NumFn::I(f) => Box::new(move |rt| f(rt).map(Num::I)),
            NumFn::L(f) => Box::new(move |rt| f(rt).map(Num::L)),
            NumFn::D(f) => Box::new(move |rt| f(rt).map(Num::D)),
            NumFn::N(f) => f,
        }
    }
    pub(super) fn into_d(self) -> F<f64> {
        match self {
            NumFn::I(f) => Box::new(move |rt| f(rt).map(|i| i as f64)),
            NumFn::L(f) => Box::new(move |rt| f(rt).map(|l| l as f64)),
            NumFn::D(f) => f,
            NumFn::N(f) => Box::new(move |rt| f(rt).map(Num::f)),
        }
    }
    /// Widen an `int` / `long` expression to `long`.
    pub(super) fn into_l(self) -> Option<F<i64>> {
        Some(match self {
            NumFn::I(f) => Box::new(move |rt| f(rt).map(|i| i as i64)),
            NumFn::L(f) => f,
            _ => return None,
        })
    }
    fn is_d(&self) -> bool {
        matches!(self, NumFn::D(_))
    }
    fn is_n(&self) -> bool {
        matches!(self, NumFn::N(_))
    }

    /// The general-path [`ExprFn`] fed by this one, falling back to `slow`.
    pub(super) fn with_fallback(self, slow: ExprFn) -> ExprFn {
        match self {
            NumFn::I(f) => Box::new(move |rt| match f(rt) {
                Some(i) => Ok(Val::Int(i)),
                None => slow(rt),
            }),
            NumFn::L(f) => Box::new(move |rt| match f(rt) {
                Some(l) => Ok(Val::Long(l)),
                None => slow(rt),
            }),
            NumFn::D(f) => Box::new(move |rt| match f(rt) {
                Some(d) => Ok(Val::Double(d)),
                None => slow(rt),
            }),
            NumFn::N(f) => Box::new(move |rt| match f(rt) {
                Some(n) => Ok(n.val()),
                None => slow(rt),
            }),
        }
    }
}

/// `Math` / `StrictMath` methods of two doubles returning a double, as
/// [`math`] computes them.
fn math_binary(name: &str) -> Option<fn(f64, f64) -> f64> {
    Some(match name {
        "max" => |x, y| if x.is_nan() || y.is_nan() { f64::NAN } else { x.max(y) },
        "min" => |x, y| if x.is_nan() || y.is_nan() { f64::NAN } else { x.min(y) },
        "pow" => f64::powf,
        "atan2" => f64::atan2,
        "hypot" => f64::hypot,
        _ => return None,
    })
}

#[inline(always)]
fn int_op(op: BinOp, x: i32, y: i32) -> Option<i32> {
    Some(match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Div if y != 0 => x.wrapping_div(y),
        BinOp::Rem if y != 0 => x.wrapping_rem(y),
        _ => return None,
    })
}

#[inline(always)]
fn long_op(op: BinOp, x: i64, y: i64) -> Option<i64> {
    Some(match op {
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Div if y != 0 => x.wrapping_div(y),
        BinOp::Rem if y != 0 => x.wrapping_rem(y),
        _ => return None,
    })
}

#[inline(always)]
fn double_op(op: BinOp, x: f64, y: f64) -> f64 {
    match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => x / y,
        _ => x % y,
    }
}

#[inline(always)]
fn num_op(op: BinOp, x: Num, y: Num) -> Option<Num> {
    match (x, y) {
        (Num::D(_), _) | (_, Num::D(_)) => Some(Num::D(double_op(op, x.f(), y.f()))),
        (Num::I(a), Num::I(b)) => int_op(op, a, b).map(Num::I),
        (Num::L(a), Num::L(b)) => long_op(op, a, b).map(Num::L),
        (Num::I(a), Num::L(b)) => long_op(op, a as i64, b).map(Num::L),
        (Num::L(a), Num::I(b)) => long_op(op, a, b as i64).map(Num::L),
    }
}

#[inline(always)]
fn cmp_op(op: BinOp, o: Option<std::cmp::Ordering>) -> bool {
    use std::cmp::Ordering::*;
    match op {
        BinOp::Lt => o == Some(Less),
        BinOp::Le => matches!(o, Some(Less | Equal)),
        BinOp::Gt => o == Some(Greater),
        _ => matches!(o, Some(Greater | Equal)),
    }
}

#[inline(always)]
fn num_cmp(x: Num, y: Num) -> Option<std::cmp::Ordering> {
    match (x, y) {
        (Num::D(_), _) | (_, Num::D(_)) => x.f().partial_cmp(&y.f()),
        (Num::I(a), Num::I(b)) => Some(a.cmp(&b)),
        (Num::L(a), Num::L(b)) => Some(a.cmp(&b)),
        (Num::I(a), Num::L(b)) => Some((a as i64).cmp(&b)),
        (Num::L(a), Num::I(b)) => Some(a.cmp(&(b as i64))),
    }
}

impl Compiler {
    /// Whether `e` is worth a numeric fast path of its own (an operation,
    /// not a bare operand the general path already reads cheaply).
    pub(super) fn num_root(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Binary(op, ..) => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge),
            ExprKind::Cast { .. } | ExprKind::Unary(UnOp::Neg, _) => true,
            ExprKind::Call { obj, .. } => matches!(self.static_class(obj), Some("Math" | "StrictMath")),
            ExprKind::FnCall { name, args } => self.ctx == ScriptContext::Score && super::score_fns::ScoreFn::of(name, args.len()).is_some(),
            _ => false,
        }
    }

    /// Numeric comparison `a op b` (`<`, `<=`, `>`, `>=`), unboxed.
    pub(super) fn num_cond(&mut self, e: &Expr) -> Option<F<bool>> {
        let ExprKind::Binary(op, a, b) = &e.kind else { return None };
        let op = *op;
        if !matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            return None;
        }
        let (x, y) = (self.num(a)?, self.num(b)?);
        Some(match (x, y) {
            (x, y) if x.is_n() || y.is_n() => {
                let (x, y) = (x.into_n(), y.into_n());
                Box::new(move |rt| Some(cmp_op(op, num_cmp(x(rt)?, y(rt)?))))
            }
            // `int` vs `int`: the same order as the widened `long` comparison.
            (NumFn::I(x), NumFn::I(y)) => Box::new(move |rt| Some(cmp_op(op, x(rt)?.partial_cmp(&y(rt)?)))),
            (x, y) if x.is_d() || y.is_d() => {
                let (x, y) = (x.into_d(), y.into_d());
                Box::new(move |rt| Some(cmp_op(op, x(rt)?.partial_cmp(&y(rt)?))))
            }
            (x, y) => {
                let (x, y) = (x.into_l()?, y.into_l()?);
                Box::new(move |rt| Some(cmp_op(op, x(rt)?.partial_cmp(&y(rt)?))))
            }
        })
    }

    /// The numeric fast path of `e`, if every node of it is supported.
    pub(super) fn num(&mut self, e: &Expr) -> Option<NumFn> {
        if let Some(fast) = self.doc_fast(e) {
            return Some(match fast {
                DocFast::First(slot) => NumFn::N(Box::new(move |rt| rt.doc.fast_num(slot))),
                DocFast::Geo(slot, lat) => NumFn::D(Box::new(move |rt| rt.doc.fast_geo(slot, lat))),
            });
        }
        Some(match &e.kind {
            ExprKind::Int(i) => {
                let i = *i;
                NumFn::I(Box::new(move |_| Some(i)))
            }
            ExprKind::Long(l) => {
                let l = *l;
                NumFn::L(Box::new(move |_| Some(l)))
            }
            ExprKind::Double(d) => {
                let d = *d;
                NumFn::D(Box::new(move |_| Some(d)))
            }
            // A primitive local always holds its own variant (assignments
            // coerce); a typed read skips the run-time type switch and
            // still bails on anything else.
            ExprKind::Var(n) => match self.lookup(n) {
                Some((slot, Ty::Int)) => NumFn::I(Box::new(move |rt| match rt.local(slot) {
                    Val::Int(i) => Some(*i),
                    _ => None,
                })),
                Some((slot, Ty::Long)) => NumFn::L(Box::new(move |rt| match rt.local(slot) {
                    Val::Long(l) => Some(*l),
                    _ => None,
                })),
                Some((slot, Ty::Double | Ty::Float)) => NumFn::D(Box::new(move |rt| match rt.local(slot) {
                    Val::Double(d) => Some(*d),
                    _ => None,
                })),
                Some((slot, _)) => NumFn::N(Box::new(move |rt| Num::of(rt.local(slot)))),
                None if !self.in_function && n == "_score" => NumFn::D(Box::new(|rt| Some(rt.score))),
                None => return None,
            },
            // `doc['x'].value` / `doc.x.value`.
            ExprKind::Field { obj, name, null_safe: false } if name == "value" && self.doc_slot(obj).is_some() => {
                let slot = self.doc_slot(obj)?;
                NumFn::N(Box::new(move |rt| rt.doc.fast_num(slot)))
            }
            ExprKind::Call { obj, name, args, null_safe: false } if name == "getValue" && args.is_empty() && self.doc_slot(obj).is_some() => {
                let slot = self.doc_slot(obj)?;
                NumFn::N(Box::new(move |rt| rt.doc.fast_num(slot)))
            }
            ExprKind::Field { .. } | ExprKind::Index { .. } => match self.param_key(e) {
                Some(k) if k != "_source" => {
                    let id = self.param_id(&k);
                    NumFn::N(Box::new(move |rt| Num::of(&rt.param_vals[id])))
                }
                _ => return None,
            },
            ExprKind::Unary(UnOp::Neg, x) => match self.num(x)? {
                NumFn::I(f) => NumFn::I(Box::new(move |rt| f(rt).map(i32::wrapping_neg))),
                NumFn::L(f) => NumFn::L(Box::new(move |rt| f(rt).map(i64::wrapping_neg))),
                NumFn::D(f) => NumFn::D(Box::new(move |rt| f(rt).map(|d| -d))),
                NumFn::N(f) => NumFn::N(Box::new(move |rt| {
                    Some(match f(rt)? {
                        Num::I(i) => Num::I(i.wrapping_neg()),
                        Num::L(l) => Num::L(l.wrapping_neg()),
                        Num::D(d) => Num::D(-d),
                    })
                })),
            },
            ExprKind::Binary(op, a, b) if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem) => {
                let op = *op;
                let (x, y) = (self.num(a)?, self.num(b)?);
                if x.is_d() || y.is_d() {
                    // Anything with a double operand is a double operation.
                    let (x, y) = (x.into_d(), y.into_d());
                    match op {
                        BinOp::Add => NumFn::D(Box::new(move |rt| Some(x(rt)? + y(rt)?))),
                        BinOp::Sub => NumFn::D(Box::new(move |rt| Some(x(rt)? - y(rt)?))),
                        BinOp::Mul => NumFn::D(Box::new(move |rt| Some(x(rt)? * y(rt)?))),
                        BinOp::Div => NumFn::D(Box::new(move |rt| Some(x(rt)? / y(rt)?))),
                        _ => NumFn::D(Box::new(move |rt| Some(x(rt)? % y(rt)?))),
                    }
                } else if x.is_n() || y.is_n() {
                    let (x, y) = (x.into_n(), y.into_n());
                    NumFn::N(Box::new(move |rt| num_op(op, x(rt)?, y(rt)?)))
                } else {
                    match (x, y) {
                        (NumFn::I(x), NumFn::I(y)) => NumFn::I(Box::new(move |rt| int_op(op, x(rt)?, y(rt)?))),
                        (x, y) => {
                            let (x, y) = (x.into_l()?, y.into_l()?);
                            NumFn::L(Box::new(move |rt| long_op(op, x(rt)?, y(rt)?)))
                        }
                    }
                }
            }
            ExprKind::Cast { ty, expr } => {
                let x = self.num(expr)?;
                match ty {
                    Ty::Int => match x {
                        NumFn::I(f) => NumFn::I(f),
                        NumFn::L(f) => NumFn::I(Box::new(move |rt| f(rt).map(|l| l as i32))),
                        NumFn::D(f) => NumFn::I(Box::new(move |rt| f(rt).map(|d| if d.is_nan() { 0 } else { d as i32 }))),
                        NumFn::N(f) => NumFn::I(Box::new(move |rt| {
                            Some(match f(rt)? {
                                Num::I(i) => i,
                                Num::L(l) => l as i32,
                                Num::D(d) => {
                                    if d.is_nan() {
                                        0
                                    } else {
                                        d as i32
                                    }
                                }
                            })
                        })),
                    },
                    Ty::Long => match x {
                        NumFn::I(f) => NumFn::L(Box::new(move |rt| f(rt).map(|i| i as i64))),
                        NumFn::L(f) => NumFn::L(f),
                        NumFn::D(f) => NumFn::L(Box::new(move |rt| f(rt).map(|d| if d.is_nan() { 0 } else { d as i64 }))),
                        NumFn::N(f) => NumFn::L(Box::new(move |rt| {
                            Some(match f(rt)? {
                                Num::I(i) => i as i64,
                                Num::L(l) => l,
                                Num::D(d) => {
                                    if d.is_nan() {
                                        0
                                    } else {
                                        d as i64
                                    }
                                }
                            })
                        })),
                    },
                    Ty::Double => NumFn::D(x.into_d()),
                    Ty::Float => {
                        let f = x.into_d();
                        NumFn::D(Box::new(move |rt| f(rt).map(|d| d as f32 as f64)))
                    }
                    _ => return None,
                }
            }
            ExprKind::FnCall { name, args } if self.ctx == ScriptContext::Score && !self.fn_sigs.iter().any(|f| f.0 == *name && f.1.len() == args.len()) => {
                let kind = super::score_fns::ScoreFn::of(name, args.len())?;
                return self.score_fn_num(kind, args);
            }
            ExprKind::Call { obj, name, args, null_safe: false } if matches!(self.static_class(obj), Some("Math" | "StrictMath")) => match args.as_slice() {
                [a] => {
                    if name == "round" {
                        let f = self.num(a)?.into_d();
                        return Some(NumFn::L(Box::new(move |rt| f(rt).map(|x| if x.is_nan() { 0 } else { (x + 0.5).floor() as i64 }))));
                    }
                    let op = if name == "abs" { f64::abs } else { math_unary(name)? };
                    let f = self.num(a)?.into_d();
                    NumFn::D(Box::new(move |rt| f(rt).map(op)))
                }
                [a, b] => {
                    let op = math_binary(name)?;
                    let (x, y) = (self.num(a)?.into_d(), self.num(b)?.into_d());
                    NumFn::D(Box::new(move |rt| Some(op(x(rt)?, y(rt)?))))
                }
                _ => return None,
            },
            // `s.length()` of a local string (side-effect free receiver).
            ExprKind::Call { obj, name, args, .. } if name == "length" && args.is_empty() => match &obj.kind {
                ExprKind::Var(v) => {
                    let (slot, _) = self.lookup(v)?;
                    NumFn::I(Box::new(move |rt| crate::script::methods::string_length_fast(rt.local(slot))))
                }
                _ => return None,
            },
            _ => return None,
        })
    }
}
