//! Painless syntax tree.

/// A declared type. The compiler checks assignments against it, primitives
/// coerce on assignment, and runtime checks enforce reference types such as
/// `String`, `List` and `Map`.
#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Def,
    Void,
    Boolean,
    Byte,
    Short,
    Char,
    Int,
    Long,
    Float,
    Double,
    String,
    /// A whitelisted reference type by name (`List`, `Map`, `ArrayList`, ...).
    Ref(String),
    Array(Box<Ty>),
}

impl Ty {
    pub fn name(&self) -> String {
        match self {
            Ty::Def => "def".into(),
            Ty::Void => "void".into(),
            Ty::Boolean => "boolean".into(),
            Ty::Byte => "byte".into(),
            Ty::Short => "short".into(),
            Ty::Char => "char".into(),
            Ty::Int => "int".into(),
            Ty::Long => "long".into(),
            Ty::Float => "float".into(),
            Ty::Double => "double".into(),
            Ty::String => "String".into(),
            Ty::Ref(s) => s.clone(),
            Ty::Array(t) => format!("{}[]", t.name()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
    UShr,
    BitAnd,
    BitOr,
    BitXor,
    Eq,
    Ne,
    RefEq,
    RefNe,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::UShr => ">>>",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::RefEq => "===",
            BinOp::RefNe => "!==",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Plus,
    Not,
    BitNot,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub pos: usize,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Null,
    Bool(bool),
    Int(i32),
    Long(i64),
    Double(f64),
    Str(String),
    Utf16(Vec<u16>),
    /// `/pattern/flags`
    Regex { pattern: String, flags: String, units: Option<Vec<u16>> },
    /// `text =~ /re/` (find) / `text ==~ /re/` (matches)
    RegexOp { find: bool, text: Box<Expr>, pat: Box<Expr> },
    Var(String),
    ListLit(Vec<Expr>),
    MapLit(Vec<(Expr, Expr)>),
    New { ty: String, args: Vec<Expr> },
    NewArray { ty: Ty, dims: Vec<Expr> },
    NewArrayInit { ty: Ty, items: Vec<Expr> },
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Elvis(Box<Expr>, Box<Expr>),
    Assign { target: Box<Expr>, op: Option<BinOp>, value: Box<Expr> },
    IncDec { target: Box<Expr>, inc: bool, prefix: bool },
    Field { obj: Box<Expr>, name: String, null_safe: bool },
    Index { obj: Box<Expr>, idx: Box<Expr> },
    Call { obj: Box<Expr>, name: String, args: Vec<Expr>, null_safe: bool },
    /// A call of a bare name: a user function or a context builtin (`emit`).
    FnCall { name: String, args: Vec<Expr> },
    Cast { ty: Ty, expr: Box<Expr> },
    InstanceOf { expr: Box<Expr>, ty: Ty },
    Lambda { params: Vec<String>, body: Box<LambdaBody> },
    /// `Type::method` / `this::fn` reference.
    MethodRef { class: String, method: String },
}

#[derive(Clone, Debug)]
pub enum LambdaBody {
    Expr(Expr),
    Block(Vec<Stmt>),
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub pos: usize,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Expr(Expr),
    Decl { ty: Ty, vars: Vec<(String, Option<Expr>, usize)> },
    If { cond: Expr, then: Box<Stmt>, els: Option<Box<Stmt>> },
    While { cond: Expr, body: Box<Stmt> },
    DoWhile { body: Box<Stmt>, cond: Expr },
    For { init: Option<Box<Stmt>>, cond: Option<Expr>, update: Vec<Expr>, body: Box<Stmt> },
    ForEach { ty: Ty, var: String, iter: Expr, body: Box<Stmt> },
    Block(Vec<Stmt>),
    Return(Option<Expr>),
    Break,
    Continue,
    Throw(Expr),
    Try { body: Vec<Stmt>, catches: Vec<(String, String, Vec<Stmt>)> },
    Empty,
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: String,
    pub ret: Ty,
    pub params: Vec<(Ty, String)>,
    pub body: Vec<Stmt>,
    pub pos: usize,
}

#[derive(Clone, Debug)]
pub struct ScriptAst {
    pub functions: Vec<FnDecl>,
    pub body: Vec<Stmt>,
}
