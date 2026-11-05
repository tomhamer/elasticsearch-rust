//! Recursive-descent Painless parser.

use super::ast::*;
use super::error::ScriptErr;
use super::lexer::{lex, Tok, Token};

type PResult<T> = Result<T, ScriptErr>;

/// Reference type names usable in declarations and casts.
pub const REF_TYPES: &[&str] = &[
    "String", "Object", "Number", "Integer", "Long", "Double", "Float", "Boolean", "Character", "Byte", "Short", "List", "ArrayList", "LinkedList", "Map", "HashMap",
    "LinkedHashMap", "TreeMap", "Set", "HashSet", "LinkedHashSet", "TreeSet", "Collection", "Iterable", "Iterator", "ZonedDateTime", "Instant", "LocalDate", "LocalDateTime",
    "DayOfWeek", "Month", "Optional", "StringBuilder", "CharSequence", "Comparable", "Map.Entry", "Entry", "Function", "Predicate", "BiFunction", "Consumer", "Supplier",
    "Comparator", "Stream", "IntStream", "LongStream", "DoubleStream", "BigDecimal", "BigInteger", "Exception", "RuntimeException", "IllegalArgumentException",
    "IllegalStateException", "ScriptDocValues", "JodaCompatibleZonedDateTime", "GeoPoint", "GeoBoundingBox", "Date", "Pattern", "Matcher", "MatchResult",
    "MathContext", "RoundingMode", "Enum", "Random", "Locale",
];

fn prim(name: &str) -> Option<Ty> {
    Some(match name {
        "def" => Ty::Def,
        "void" => Ty::Void,
        "boolean" => Ty::Boolean,
        "byte" => Ty::Byte,
        "short" => Ty::Short,
        "char" => Ty::Char,
        "int" => Ty::Int,
        "long" => Ty::Long,
        "float" => Ty::Float,
        "double" => Ty::Double,
        _ => return None,
    })
}

pub fn type_of_name(name: &str) -> Option<Ty> {
    if let Some(t) = prim(name) {
        return Some(t);
    }
    if name == "String" {
        return Some(Ty::String);
    }
    if REF_TYPES.contains(&name) || super::jbig::CLASSES.contains(&name) || super::jtime::CLASSES.contains(&name) || super::jmisc::CLASSES.contains(&name) {
        return Some(Ty::Ref(name.to_string()));
    }
    None
}

/// The importable name of the type `parts` (joined by `.`) spells, if it
/// is one: a name [`type_of_name`] knows, or a whitelisted class written
/// as Painless spells it, either fully qualified (`java.util.Map.Entry`)
/// or by its class part (`Map.Entry`, `ScriptDocValues.Geometry`); inner
/// classes (`$` in the whitelist) are written with `.`.
pub fn qualified_type_name(parts: &[&str]) -> Option<String> {
    let joined = parts.join(".");
    if parts.len() == 1 {
        return type_of_name(&joined).map(|_| joined);
    }
    if type_of_name(&joined).is_some() {
        return Some(joined);
    }
    // Only a class part (capitalized) or a package (`java`, `org`, ...)
    // can start a type; `params.x` and `doc.y` never do.
    if !parts[0].starts_with(|c: char| c.is_ascii_uppercase()) && !matches!(parts[0], "java" | "org" | "javax") {
        return None;
    }
    super::whitelist::WHITELIST.iter().find_map(|(fqn, ..)| {
        let canonical = fqn.replace('$', ".");
        // The class part starts at the first capitalized segment.
        let start = canonical.split('.').position(|seg| seg.starts_with(|c: char| c.is_ascii_uppercase()))?;
        let short = canonical.split('.').skip(start).collect::<Vec<_>>().join(".");
        (joined == canonical || joined == short).then_some(short)
    })
}

const KEYWORDS: &[&str] = &[
    "if", "else", "while", "do", "for", "in", "continue", "break", "return", "new", "try", "catch", "throw", "this", "instanceof", "true", "false", "null",
];

pub struct Parser {
    literals: std::collections::HashMap<usize, Vec<u16>>,
    toks: Vec<Token>,
    i: usize,
    /// Statement start offsets (script_stack snippets).
    pub stmt_starts: Vec<usize>,
    /// Depth of enclosing call-argument lists: lambdas and method
    /// references are only valid there.
    in_args: usize,
    nest: crate::parse_depth::Nesting,
}

pub fn parse(src: &str) -> PResult<(ScriptAst, Vec<usize>)> {
    parse_with_literals(src, &std::collections::HashMap::new())
}

pub fn parse_with_literals(src: &str, literals: &std::collections::HashMap<usize, Vec<u16>>) -> PResult<(ScriptAst, Vec<usize>)> {
    let toks = lex(src)?;
    let mut p = Parser { toks, i: 0, stmt_starts: Vec::new(), in_args: 0, literals: literals.clone(), nest: crate::parse_depth::Nesting::new(512) };
    let mut functions = Vec::new();
    let mut body = Vec::new();
    // Functions come first.
    while p.is_fn_decl() {
        functions.push(p.fn_decl()?);
    }
    while !p.at_eof() {
        body.push(p.statement(true)?);
    }
    if body.is_empty() && functions.is_empty() {
        return Err(ScriptErr::compile(0, "cannot generate empty script"));
    }
    let mut starts = p.stmt_starts.clone();
    starts.sort_unstable();
    starts.dedup();
    Ok((ScriptAst { functions, body }, starts))
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.i].tok
    }
    fn peek_at(&self, k: usize) -> &Tok {
        let j = (self.i + k).min(self.toks.len() - 1);
        &self.toks[j].tok
    }
    fn pos(&self) -> usize {
        self.toks[self.i].pos
    }
    fn at_eof(&self) -> bool {
        matches!(self.peek(), Tok::Eof)
    }
    fn is_p(&self, p: &str) -> bool {
        matches!(self.peek(), Tok::P(x) if *x == p)
    }
    fn is_p_at(&self, k: usize, p: &str) -> bool {
        matches!(self.peek_at(k), Tok::P(x) if *x == p)
    }
    fn is_ident(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Ident(x) if x == s)
    }
    fn eat_p(&mut self, p: &str) -> bool {
        if self.is_p(p) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn eat_ident(&mut self, s: &str) -> bool {
        if self.is_ident(s) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn unexpected(&self) -> ScriptErr {
        let t = match self.peek() {
            Tok::Eof => "<EOF>".to_string(),
            Tok::Ident(s) => s.clone(),
            Tok::Int(v, _) => v.to_string(),
            Tok::Float(v) => v.to_string(),
            Tok::Str(s) => format!("'{s}'"),
            Tok::Regex(p, f) => format!("/{p}/{f}"),
            Tok::P(p) => p.to_string(),
        };
        if matches!(self.peek(), Tok::Eof) {
            return ScriptErr::compile(self.pos(), "unexpected end of script.");
        }
        ScriptErr::compile(self.pos(), format!("invalid sequence of tokens near ['{t}']."))
    }
    fn expect_p(&mut self, p: &str) -> PResult<()> {
        if self.eat_p(p) {
            Ok(())
        } else {
            Err(self.unexpected())
        }
    }
    fn ident(&mut self) -> PResult<String> {
        match self.peek().clone() {
            Tok::Ident(s) if !KEYWORDS.contains(&s.as_str()) => {
                self.i += 1;
                Ok(s)
            }
            _ => Err(self.unexpected()),
        }
    }

    /// Try to read a type at the cursor (without consuming on failure).
    /// Returns the type and the number of tokens it spans.
    fn type_at(&self, k: usize) -> Option<(Ty, usize)> {
        let (name, mut n) = self.dotted_type_at(k)?;
        let mut ty = type_of_name(&name).unwrap_or(Ty::Ref(name));
        while self.is_p_at(k + n, "[") && self.is_p_at(k + n + 1, "]") {
            ty = Ty::Array(Box::new(ty));
            n += 2;
        }
        Some((ty, n))
    }

    /// The longest `ID (. ID)*` run at `k` naming a type (Painless
    /// `type: ID (DOT DOTID)*`): a simple or imported dotted name
    /// (`Map.Entry`, `ScriptDocValues.Geometry`) or a fully qualified class
    /// (`java.util.Map.Entry`). Returns the importable name and the tokens
    /// it spans.
    fn dotted_type_at(&self, k: usize) -> Option<(String, usize)> {
        let mut parts: Vec<&str> = Vec::new();
        loop {
            match self.peek_at(k + parts.len() * 2) {
                Tok::Ident(s) => parts.push(s.as_str()),
                _ => break,
            }
            if !self.is_p_at(k + parts.len() * 2 - 1, ".") {
                break;
            }
        }
        // A trailing `.` read past the last identifier is not part of it.
        while !parts.is_empty() {
            if let Some(name) = qualified_type_name(&parts) {
                return Some((name, parts.len() * 2 - 1));
            }
            parts.pop();
        }
        None
    }

    fn read_type(&mut self) -> PResult<Ty> {
        match self.type_at(0) {
            Some((t, n)) => {
                self.i += n;
                Ok(t)
            }
            None => Err(self.unexpected()),
        }
    }

    fn is_fn_decl(&self) -> bool {
        if let Some((_, n)) = self.type_at(0) {
            if let Tok::Ident(name) = self.peek_at(n) {
                return !KEYWORDS.contains(&name.as_str()) && self.is_p_at(n + 1, "(");
            }
        }
        false
    }

    fn fn_decl(&mut self) -> PResult<FnDecl> {
        let pos = self.pos();
        let ret = self.read_type()?;
        let name = self.ident()?;
        self.expect_p("(")?;
        let mut params = Vec::new();
        if !self.is_p(")") {
            loop {
                let t = self.read_type()?;
                let n = self.ident()?;
                params.push((t, n));
                if !self.eat_p(",") {
                    break;
                }
            }
        }
        self.expect_p(")")?;
        let body = self.block()?;
        Ok(FnDecl { name, ret, params, body, pos })
    }

    fn block(&mut self) -> PResult<Vec<Stmt>> {
        self.expect_p("{")?;
        let mut out = Vec::new();
        while !self.is_p("}") {
            if self.at_eof() {
                return Err(self.unexpected());
            }
            out.push(self.statement(false)?);
        }
        self.i += 1;
        Ok(out)
    }

    /// End of a simple statement: `;`, or nothing before `}` / EOF.
    fn end_stmt(&mut self) -> PResult<()> {
        if self.eat_p(";") || self.is_p("}") || self.at_eof() {
            Ok(())
        } else {
            let t = match self.peek() {
                Tok::Ident(s) => s.clone(),
                Tok::Int(v, _) => v.to_string(),
                Tok::Float(v) => v.to_string(),
                Tok::Str(s) => format!("'{s}'"),
                Tok::Regex(p, f) => format!("/{p}/{f}"),
                Tok::P(p) => p.to_string(),
                Tok::Eof => "<EOF>".to_string(),
            };
            Err(ScriptErr::compile(self.pos(), format!("unexpected token ['{t}'] was expecting one of [{{<EOF>, ';'}}].")))
        }
    }

    /// Is a local declaration starting here (`Type name ...`)?
    fn is_decl(&self) -> bool {
        if let Some((_, n)) = self.type_at(0) {
            if let Tok::Ident(name) = self.peek_at(n) {
                if KEYWORDS.contains(&name.as_str()) && name != "in" {
                    return false;
                }
                return matches!(self.peek_at(n + 1), Tok::P("=") | Tok::P(";") | Tok::P(",") | Tok::P(":") | Tok::P("}") | Tok::Eof)
                    || matches!(self.peek_at(n + 1), Tok::Ident(s) if s == "in");
            }
        }
        false
    }

    fn statement(&mut self, top: bool) -> PResult<Stmt> {
        self.nested(true, |p| p.statement_unnested(top))
    }

    fn statement_unnested(&mut self, _top: bool) -> PResult<Stmt> {
        let pos = self.pos();
        self.stmt_starts.push(pos);
        let kind = match self.peek().clone() {
            Tok::P("{") => StmtKind::Block(self.block()?),
            Tok::P(";") => {
                self.i += 1;
                StmtKind::Empty
            }
            Tok::Ident(k) if k == "if" => {
                self.i += 1;
                self.expect_p("(")?;
                let cond = self.expr()?;
                self.expect_p(")")?;
                let then = Box::new(self.statement(false)?);
                let els = if self.eat_ident("else") { Some(Box::new(self.statement(false)?)) } else { None };
                StmtKind::If { cond, then, els }
            }
            Tok::Ident(k) if k == "while" => {
                self.i += 1;
                self.expect_p("(")?;
                let cond = self.expr()?;
                self.expect_p(")")?;
                let body = Box::new(self.statement(false)?);
                StmtKind::While { cond, body }
            }
            Tok::Ident(k) if k == "do" => {
                self.i += 1;
                let body = Box::new(self.statement(false)?);
                if !self.eat_ident("while") {
                    return Err(self.unexpected());
                }
                self.expect_p("(")?;
                let cond = self.expr()?;
                self.expect_p(")")?;
                self.end_stmt()?;
                StmtKind::DoWhile { body, cond }
            }
            Tok::Ident(k) if k == "for" => {
                self.i += 1;
                self.expect_p("(")?;
                // for-each: `Type x : e`, `x in e`, `Type x in e`.
                let each = {
                    if let Some((_, n)) = self.type_at(0) {
                        matches!(self.peek_at(n), Tok::Ident(_)) && (self.is_p_at(n + 1, ":") || matches!(self.peek_at(n + 1), Tok::Ident(s) if s == "in"))
                    } else {
                        matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek_at(1), Tok::Ident(s) if s == "in")
                    }
                };
                if each {
                    let ty = if self.type_at(0).is_some() && !matches!(self.peek_at(1), Tok::Ident(s) if s == "in") { self.read_type()? } else { Ty::Def };
                    let var = self.ident()?;
                    if !self.eat_p(":") && !self.eat_ident("in") {
                        return Err(self.unexpected());
                    }
                    let iter = self.expr()?;
                    self.expect_p(")")?;
                    let body = Box::new(self.statement(false)?);
                    StmtKind::ForEach { ty, var, iter, body }
                } else {
                    let init = if self.is_p(";") {
                        None
                    } else if self.is_decl() {
                        let p = self.pos();
                        Some(Box::new(Stmt { kind: self.decl_body()?, pos: p }))
                    } else {
                        let p = self.pos();
                        Some(Box::new(Stmt { kind: StmtKind::Expr(self.expr()?), pos: p }))
                    };
                    self.expect_p(";")?;
                    let cond = if self.is_p(";") { None } else { Some(self.expr()?) };
                    self.expect_p(";")?;
                    let mut update = Vec::new();
                    if !self.is_p(")") {
                        loop {
                            update.push(self.expr()?);
                            if !self.eat_p(",") {
                                break;
                            }
                        }
                    }
                    self.expect_p(")")?;
                    let body = Box::new(self.statement(false)?);
                    StmtKind::For { init, cond, update, body }
                }
            }
            Tok::Ident(k) if k == "return" => {
                self.i += 1;
                if self.is_p(";") || self.is_p("}") || self.at_eof() {
                    self.eat_p(";");
                    StmtKind::Return(None)
                } else {
                    let e = self.expr()?;
                    self.end_stmt()?;
                    StmtKind::Return(Some(e))
                }
            }
            Tok::Ident(k) if k == "break" => {
                self.i += 1;
                self.end_stmt()?;
                StmtKind::Break
            }
            Tok::Ident(k) if k == "continue" => {
                self.i += 1;
                self.end_stmt()?;
                StmtKind::Continue
            }
            Tok::Ident(k) if k == "throw" => {
                self.i += 1;
                let e = self.expr()?;
                self.end_stmt()?;
                StmtKind::Throw(e)
            }
            Tok::Ident(k) if k == "try" => {
                self.i += 1;
                let body = self.block()?;
                let mut catches = Vec::new();
                while self.eat_ident("catch") {
                    self.expect_p("(")?;
                    let ty = match self.peek().clone() {
                        Tok::Ident(s) => {
                            self.i += 1;
                            s
                        }
                        _ => return Err(self.unexpected()),
                    };
                    let var = self.ident()?;
                    self.expect_p(")")?;
                    let b = self.block()?;
                    catches.push((ty, var, b));
                }
                if catches.is_empty() {
                    return Err(self.unexpected());
                }
                StmtKind::Try { body, catches }
            }
            _ if self.is_decl() => {
                let d = self.decl_body()?;
                self.end_stmt()?;
                d
            }
            _ => {
                let e = self.expr()?;
                self.end_stmt()?;
                StmtKind::Expr(e)
            }
        };
        Ok(Stmt { kind, pos })
    }

    fn decl_body(&mut self) -> PResult<StmtKind> {
        let ty = self.read_type()?;
        let mut vars = Vec::new();
        loop {
            let p = self.pos();
            let name = self.ident()?;
            let init = if self.eat_p("=") { Some(self.expr()?) } else { None };
            vars.push((name, init, p));
            if !self.eat_p(",") {
                break;
            }
        }
        Ok(StmtKind::Decl { ty, vars })
    }

    // -----------------------------------------------------------------
    // Expressions.

    /// Runs `f` one nesting level deeper (see [`crate::parse_depth`]).
    fn nested<T>(&mut self, counted: bool, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        let levels = self.nest.snapshot();
        if !self.nest.enter(counted) {
            return Err(ScriptErr::compile(self.pos(), format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
        }
        let r = f(self);
        self.nest.restore(levels);
        r
    }

    pub fn expr(&mut self) -> PResult<Expr> {
        self.nested(true, Self::expr_unnested)
    }

    fn expr_unnested(&mut self) -> PResult<Expr> {
        self.assignment()
    }

    fn assignment(&mut self) -> PResult<Expr> {
        let lhs = self.conditional()?;
        let pos = self.pos();
        let op = match self.peek() {
            Tok::P("=") => Some(None),
            Tok::P("+=") => Some(Some(BinOp::Add)),
            Tok::P("-=") => Some(Some(BinOp::Sub)),
            Tok::P("*=") => Some(Some(BinOp::Mul)),
            Tok::P("/=") => Some(Some(BinOp::Div)),
            Tok::P("%=") => Some(Some(BinOp::Rem)),
            Tok::P("&=") => Some(Some(BinOp::BitAnd)),
            Tok::P("|=") => Some(Some(BinOp::BitOr)),
            Tok::P("^=") => Some(Some(BinOp::BitXor)),
            Tok::P("<<=") => Some(Some(BinOp::Shl)),
            Tok::P(">>=") => Some(Some(BinOp::Shr)),
            Tok::P(">>>=") => Some(Some(BinOp::UShr)),
            _ => None,
        };
        if let Some(op) = op {
            if !matches!(lhs.kind, ExprKind::Var(_) | ExprKind::Field { .. } | ExprKind::Index { .. }) {
                return Err(ScriptErr::compile(pos, "invalid assignment: cannot assign a value to this expression"));
            }
            self.i += 1;
            let value = self.assignment()?;
            return Ok(Expr { pos: lhs.pos, kind: ExprKind::Assign { target: Box::new(lhs), op, value: Box::new(value) } });
        }
        Ok(lhs)
    }

    fn conditional(&mut self) -> PResult<Expr> {
        let c = self.binary(0)?;
        if self.is_p("?") {
            let pos = self.pos();
            self.i += 1;
            let a = self.assignment()?;
            self.expect_p(":")?;
            let b = self.assignment()?;
            let _ = pos;
            return Ok(Expr { pos: c.pos, kind: ExprKind::Ternary(Box::new(c), Box::new(a), Box::new(b)) });
        }
        if self.is_p("?:") {
            self.i += 1;
            let b = self.assignment()?;
            return Ok(Expr { pos: c.pos, kind: ExprKind::Elvis(Box::new(c), Box::new(b)) });
        }
        Ok(c)
    }

    fn binop_prec(&self) -> Option<(u8, &'static str)> {
        let p = match self.peek() {
            Tok::P(p) => *p,
            Tok::Ident(s) if s == "instanceof" => return Some((7, "instanceof")),
            _ => return None,
        };
        let prec = match p {
            "||" => 1,
            "&&" => 2,
            "|" => 3,
            "^" => 4,
            "&" => 5,
            "==" | "!=" | "===" | "!==" => 6,
            "<" | "<=" | ">" | ">=" | "=~" | "==~" => 7,
            "<<" | ">>" | ">>>" => 8,
            "+" | "-" => 9,
            "*" | "/" | "%" => 10,
            _ => return None,
        };
        Some((prec, p))
    }

    fn binary(&mut self, min: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let Some((prec, op)) = self.binop_prec() else { break };
            if prec < min {
                break;
            }
            let pos = self.pos();
            if !self.nest.grow() {
                return Err(ScriptErr::compile(pos, format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
            }
            self.i += 1;
            if op == "instanceof" {
                let ty = self.read_type()?;
                lhs = Expr { pos, kind: ExprKind::InstanceOf { expr: Box::new(lhs), ty } };
                continue;
            }
            let rhs = self.binary(prec + 1)?;
            let kind = match op {
                "=~" | "==~" => ExprKind::RegexOp { find: op == "=~", text: Box::new(lhs), pat: Box::new(rhs) },
                "||" => ExprKind::Or(Box::new(lhs), Box::new(rhs)),
                "&&" => ExprKind::And(Box::new(lhs), Box::new(rhs)),
                _ => {
                    let b = match op {
                        "|" => BinOp::BitOr,
                        "^" => BinOp::BitXor,
                        "&" => BinOp::BitAnd,
                        "==" => BinOp::Eq,
                        "!=" => BinOp::Ne,
                        "===" => BinOp::RefEq,
                        "!==" => BinOp::RefNe,
                        "<" => BinOp::Lt,
                        "<=" => BinOp::Le,
                        ">" => BinOp::Gt,
                        ">=" => BinOp::Ge,
                        "<<" => BinOp::Shl,
                        ">>" => BinOp::Shr,
                        ">>>" => BinOp::UShr,
                        "+" => BinOp::Add,
                        "-" => BinOp::Sub,
                        "*" => BinOp::Mul,
                        "/" => BinOp::Div,
                        _ => BinOp::Rem,
                    };
                    ExprKind::Binary(b, Box::new(lhs), Box::new(rhs))
                }
            };
            lhs = Expr { pos, kind };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        self.nested(false, Self::unary_unnested)
    }

    fn unary_unnested(&mut self) -> PResult<Expr> {
        let pos = self.pos();
        match self.peek().clone() {
            Tok::P("!") => {
                self.i += 1;
                let e = self.unary()?;
                Ok(Expr { pos, kind: ExprKind::Unary(UnOp::Not, Box::new(e)) })
            }
            Tok::P("~") => {
                self.i += 1;
                let e = self.unary()?;
                Ok(Expr { pos, kind: ExprKind::Unary(UnOp::BitNot, Box::new(e)) })
            }
            Tok::P("-") => {
                self.i += 1;
                // Fold negative literals (so i64::MIN / i32::MIN parse).
                match self.peek().clone() {
                    Tok::Int(v, long) if !self.is_p_at(1, ".") && !self.is_p_at(1, "[") => {
                        self.i += 1;
                        let kind = int_kind(-(v as i128), long, pos)?;
                        Ok(Expr { pos, kind })
                    }
                    Tok::Float(v) if !self.is_p_at(1, ".") => {
                        self.i += 1;
                        Ok(Expr { pos, kind: ExprKind::Double(-v) })
                    }
                    _ => {
                        let e = self.unary()?;
                        Ok(Expr { pos, kind: ExprKind::Unary(UnOp::Neg, Box::new(e)) })
                    }
                }
            }
            Tok::P("+") => {
                self.i += 1;
                let e = self.unary()?;
                Ok(Expr { pos, kind: ExprKind::Unary(UnOp::Plus, Box::new(e)) })
            }
            Tok::P("++") | Tok::P("--") => {
                let inc = self.is_p("++");
                self.i += 1;
                let e = self.unary()?;
                Ok(Expr { pos, kind: ExprKind::IncDec { target: Box::new(e), inc, prefix: true } })
            }
            Tok::P("(") => {
                // Cast?
                if let Some((ty, n)) = self.type_at(1) {
                    if self.is_p_at(1 + n, ")") {
                        let is_prim = !matches!(ty, Ty::Ref(_) | Ty::String | Ty::Array(_) | Ty::Def);
                        let next = self.peek_at(2 + n).clone();
                        let starts_expr = match next {
                            Tok::Ident(ref s) => s != "instanceof",
                            Tok::Int(..) | Tok::Float(_) | Tok::Str(_) => true,
                            Tok::P("(") | Tok::P("!") | Tok::P("~") | Tok::P("[") => true,
                            Tok::P("-") | Tok::P("+") | Tok::P("++") | Tok::P("--") => is_prim,
                            _ => false,
                        };
                        if starts_expr {
                            self.i += 2 + n;
                            let e = self.unary()?;
                            return Ok(Expr { pos, kind: ExprKind::Cast { ty, expr: Box::new(e) } });
                        }
                    }
                }
                self.postfix()
            }
            _ => self.postfix(),
        }
    }

    fn args(&mut self) -> PResult<Vec<Expr>> {
        self.expect_p("(")?;
        let mut out = Vec::new();
        self.in_args += 1;
        if !self.is_p(")") {
            loop {
                match self.expr() {
                    Ok(e) => out.push(e),
                    Err(e) => {
                        self.in_args -= 1;
                        return Err(e);
                    }
                }
                if !self.eat_p(",") {
                    break;
                }
            }
        }
        self.in_args -= 1;
        self.expect_p(")")?;
        Ok(out)
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            let pos = self.pos();
            if self.is_p(".") || self.is_p("?.") {
                let null_safe = self.is_p("?.");
                self.i += 1;
                let name = match self.peek().clone() {
                    Tok::Ident(s) => {
                        self.i += 1;
                        s
                    }
                    _ => return Err(self.unexpected()),
                };
                if self.is_p("(") {
                    let args = self.args()?;
                    e = Expr { pos, kind: ExprKind::Call { obj: Box::new(e), name, args, null_safe } };
                } else {
                    e = Expr { pos, kind: ExprKind::Field { obj: Box::new(e), name, null_safe } };
                }
            } else if self.is_p("[") {
                self.i += 1;
                let idx = self.expr()?;
                self.expect_p("]")?;
                e = Expr { pos, kind: ExprKind::Index { obj: Box::new(e), idx: Box::new(idx) } };
            } else if self.is_p("++") || self.is_p("--") {
                let inc = self.is_p("++");
                self.i += 1;
                e = Expr { pos: e.pos, kind: ExprKind::IncDec { target: Box::new(e), inc, prefix: false } };
            } else {
                break;
            }
        }
        Ok(e)
    }

    /// `(a, b) ->` / `(def a) ->` / `() ->` lambda head at the cursor?
    fn paren_lambda_len(&self) -> Option<usize> {
        if !self.is_p("(") {
            return None;
        }
        let mut k = 1;
        loop {
            match self.peek_at(k) {
                Tok::P(")") => break,
                Tok::Ident(_) | Tok::P(",") => k += 1,
                _ => return None,
            }
        }
        if self.is_p_at(k + 1, "->") { Some(k + 1) } else { None }
    }

    fn lambda_body(&mut self) -> PResult<LambdaBody> {
        if self.is_p("{") {
            Ok(LambdaBody::Block(self.block()?))
        } else {
            Ok(LambdaBody::Expr(self.expr()?))
        }
    }

    fn primary(&mut self) -> PResult<Expr> {
        let pos = self.pos();
        let t = self.peek().clone();
        match t {
            Tok::Int(v, long) => {
                self.i += 1;
                Ok(Expr { pos, kind: int_kind(v as i128, long, pos)? })
            }
            Tok::Float(v) => {
                self.i += 1;
                Ok(Expr { pos, kind: ExprKind::Double(v) })
            }
            Tok::Str(s) => {
                self.i += 1;
                Ok(Expr { pos, kind: self.literals.get(&pos).map(|units|ExprKind::Utf16(units.clone())).unwrap_or(ExprKind::Str(s)) })
            }
            Tok::Regex(pattern, flags) => {
                self.i += 1;
                Ok(Expr { pos, kind: ExprKind::Regex { pattern, flags, units:self.literals.get(&pos).cloned() } })
            }
            Tok::P("(") => {
                if let Some(n) = self.paren_lambda_len().filter(|_| self.in_args > 0) {
                    let mut params = Vec::new();
                    let mut k = 1;
                    while k < n - 1 {
                        if let Tok::Ident(s) = self.peek_at(k) {
                            // `def x` / `int x`: the last ident before `,`/`)` is the name.
                            let next_is_ident = matches!(self.peek_at(k + 1), Tok::Ident(_));
                            if !next_is_ident {
                                params.push(s.clone());
                            }
                        }
                        k += 1;
                    }
                    self.i += n + 1;
                    let body = self.lambda_body()?;
                    return Ok(Expr { pos, kind: ExprKind::Lambda { params, body: Box::new(body) } });
                }
                self.i += 1;
                let e = self.expr()?;
                self.expect_p(")")?;
                Ok(e)
            }
            Tok::P("[") => {
                self.i += 1;
                if self.eat_p(":") {
                    self.expect_p("]")?;
                    return Ok(Expr { pos, kind: ExprKind::MapLit(Vec::new()) });
                }
                if self.eat_p("]") {
                    return Ok(Expr { pos, kind: ExprKind::ListLit(Vec::new()) });
                }
                let first = self.expr()?;
                if self.eat_p(":") {
                    let v = self.expr()?;
                    let mut pairs = vec![(first, v)];
                    while self.eat_p(",") {
                        let k = self.expr()?;
                        self.expect_p(":")?;
                        let v = self.expr()?;
                        pairs.push((k, v));
                    }
                    self.expect_p("]")?;
                    return Ok(Expr { pos, kind: ExprKind::MapLit(pairs) });
                }
                let mut items = vec![first];
                while self.eat_p(",") {
                    items.push(self.expr()?);
                }
                self.expect_p("]")?;
                Ok(Expr { pos, kind: ExprKind::ListLit(items) })
            }
            Tok::Ident(name) => {
                match name.as_str() {
                    "true" => {
                        self.i += 1;
                        return Ok(Expr { pos, kind: ExprKind::Bool(true) });
                    }
                    "false" => {
                        self.i += 1;
                        return Ok(Expr { pos, kind: ExprKind::Bool(false) });
                    }
                    "null" => {
                        self.i += 1;
                        return Ok(Expr { pos, kind: ExprKind::Null });
                    }
                    "new" => {
                        self.i += 1;
                        return self.new_expr(pos);
                    }
                    _ => {}
                }
                if KEYWORDS.contains(&name.as_str()) && name != "this" {
                    return Err(self.unexpected());
                }
                // x -> ...
                if self.is_p_at(1, "->") && self.in_args > 0 {
                    self.i += 2;
                    let body = self.lambda_body()?;
                    return Ok(Expr { pos, kind: ExprKind::Lambda { params: vec![name], body: Box::new(body) } });
                }
                // Type::method
                if self.is_p_at(1, "::") && self.in_args > 0 {
                    self.i += 2;
                    let method = match self.peek().clone() {
                        Tok::Ident(m) => {
                            self.i += 1;
                            m
                        }
                        _ => return Err(self.unexpected()),
                    };
                    return Ok(Expr { pos, kind: ExprKind::MethodRef { class: name, method } });
                }
                self.i += 1;
                if self.is_p("(") {
                    let args = self.args()?;
                    return Ok(Expr { pos, kind: ExprKind::FnCall { name, args } });
                }
                Ok(Expr { pos, kind: ExprKind::Var(name) })
            }
            _ => Err(self.unexpected()),
        }
    }

    fn new_expr(&mut self, pos: usize) -> PResult<Expr> {
        let name = match self.peek().clone() {
            // `new java.util.ArrayList()`, `new Outer.Inner(..)`: a dotted
            // type name, as its importable name.
            Tok::Ident(_) if self.is_p_at(1, ".") && self.dotted_type_at(0).is_some_and(|(_, n)| n > 1) => {
                let (name, n) = self.dotted_type_at(0).expect("checked");
                self.i += n;
                name
            }
            Tok::Ident(s) => {
                self.i += 1;
                s
            }
            _ => return Err(self.unexpected()),
        };
        if self.is_p("[") {
            let elem = type_of_name(&name).unwrap_or(Ty::Ref(name.clone()));
            // new T[] {..}  or  new T[n][m]
            if self.is_p_at(1, "]") {
                let mut ty = elem;
                while self.is_p("[") && self.is_p_at(1, "]") {
                    self.i += 2;
                    ty = Ty::Array(Box::new(ty));
                }
                self.expect_p("{")?;
                let mut items = Vec::new();
                if !self.is_p("}") {
                    loop {
                        items.push(self.expr()?);
                        if !self.eat_p(",") {
                            break;
                        }
                    }
                }
                self.expect_p("}")?;
                return Ok(Expr { pos, kind: ExprKind::NewArrayInit { ty, items } });
            }
            let mut dims = Vec::new();
            while self.eat_p("[") {
                dims.push(self.expr()?);
                self.expect_p("]")?;
            }
            return Ok(Expr { pos, kind: ExprKind::NewArray { ty: elem, dims } });
        }
        let args = self.args()?;
        Ok(Expr { pos, kind: ExprKind::New { ty: name, args } })
    }
}

fn int_kind(v: i128, long: bool, pos: usize) -> PResult<ExprKind> {
    if long {
        if v < i64::MIN as i128 || v > i64::MAX as i128 {
            return Err(ScriptErr::compile(pos, format!("Invalid long constant [{v}].")));
        }
        return Ok(ExprKind::Long(v as i64));
    }
    if v < i32::MIN as i128 || v > i32::MAX as i128 {
        return Err(ScriptErr::compile(pos, format!("Invalid int constant [{v}].")));
    }
    Ok(ExprKind::Int(v as i32))
}
