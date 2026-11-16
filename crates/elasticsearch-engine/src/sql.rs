//! Elasticsearch SQL (`POST /_sql`): a SELECT parser that compiles to the
//! ES|QL command pipeline ([`crate::esql`]) and to Query DSL for
//! `_sql/translate`.
//!
//! Supported: `SELECT [DISTINCT] items FROM index [WHERE] [GROUP BY]
//! [HAVING] [ORDER BY] [LIMIT n]`, `SELECT` without `FROM`, `SHOW TABLES`,
//! `SHOW COLUMNS FROM` / `DESCRIBE`, `SHOW FUNCTIONS`, `?` parameters.

use crate::error::{EsError, Result};
use crate::esql::exec::{self, Column, Table};
use crate::esql::lexer::{lex, Dialect, Tok};
use crate::esql::parse::{Cmd, Expr, Field, Params, Parser, SortKey};
use crate::esql::{func, Val};
use crate::node::Node;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct SqlQuery {
    pub cmds: Vec<Cmd>,
    /// Final column order (`None` = all).
    pub index: Option<String>,
    pub limit: Option<usize>,
}

pub enum Statement {
    Select(SqlQuery),
    ShowTables(Option<String>),
    ShowColumns(String),
    ShowFunctions,
    /// `SYS TABLES [CATALOG [LIKE] c] [[LIKE] t] [TYPE ...]` (JDBC metadata).
    SysTables { catalog: Option<String>, table: Option<String>, types: Option<Vec<String>> },
    /// `SYS COLUMNS [CATALOG c] [TABLE [LIKE] t] [LIKE col]`.
    SysColumns { table: Option<String>, column: Option<String> },
    /// `SYS TYPES [data_type]`.
    SysTypes(Option<i64>),
}

/// `e` with columns qualified by one of `quals` (`alias.col`) unqualified.
fn strip_qualifiers(e: &Expr, quals: &[String]) -> Expr {
    let s = |x: &Expr| Box::new(strip_qualifiers(x, quals));
    match e {
        Expr::Col(n, p) => {
            let bare = quals.iter().find_map(|q| n.strip_prefix(q.as_str()).and_then(|r| r.strip_prefix('.'))).filter(|r| !r.is_empty());
            Expr::Col(bare.unwrap_or(n).to_string(), *p)
        }
        Expr::Bin(op, a, b) => Expr::Bin(*op, s(a), s(b)),
        Expr::Neg(a) => Expr::Neg(s(a)),
        Expr::Not(a) => Expr::Not(s(a)),
        Expr::IsNull(a, n) => Expr::IsNull(s(a), *n),
        Expr::In(a, l, n) => Expr::In(s(a), l.iter().map(|x| strip_qualifiers(x, quals)).collect(), *n),
        Expr::Like(a, pats, n, r) => Expr::Like(s(a), pats.clone(), *n, *r),
        Expr::Call(name, args, p) => Expr::Call(name.clone(), args.iter().map(|x| strip_qualifiers(x, quals)).collect(), *p),
        Expr::Cast(a, t) => Expr::Cast(s(a), t.clone()),
        Expr::Between(a, b, c, n) => Expr::Between(s(a), s(b), s(c), *n),
        other => other.clone(),
    }
}

fn sql_err(full: &str, off: usize, msg: impl AsRef<str>) -> EsError {
    EsError::new(400, "parsing_exception", format!("{}: {}", crate::esql::pos_prefix(full, off), msg.as_ref()))
}

fn sql_params(v: Option<&Value>) -> Params {
    let mut p = Params::default();
    if let Some(Value::Array(a)) = v {
        for x in a {
            // A typed parameter (`{"type": "datetime", "value": ...}`, what
            // the JDBC driver sends) keeps its type: the parser converts the
            // value into a literal of that type (ES `visitParamLiteral`).
            let val = match x {
                Value::Object(m) if m.contains_key("value") => match m.get("type").and_then(Value::as_str) {
                    Some(t) => serde_json::json!({"value": m["value"].clone(), crate::esql::parse::SQL_PARAM_TYPE: t}),
                    None => m["value"].clone(),
                },
                o => o.clone(),
            };
            p.positional.push(val);
        }
    }
    p
}

const CLAUSES: &[&str] = &["WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "PIVOT"];

fn contains_agg(e: &Expr) -> bool {
    match e {
        Expr::Call(n, args, _) => func::AGGS.contains(&n.as_str()) || args.iter().any(contains_agg),
        Expr::Bin(_, a, b) => contains_agg(a) || contains_agg(b),
        Expr::Neg(a) | Expr::Not(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) => contains_agg(a),
        _ => false,
    }
}

/// Strip offsets so structurally-equal expressions compare equal.
fn norm(e: &Expr) -> Expr {
    match e {
        Expr::Col(n, _) => Expr::Col(n.clone(), 0),
        Expr::Call(n, a, _) => Expr::Call(n.clone(), a.iter().map(norm).collect(), 0),
        Expr::Bin(o, a, b) => Expr::Bin(*o, Box::new(norm(a)), Box::new(norm(b))),
        Expr::Neg(a) => Expr::Neg(Box::new(norm(a))),
        Expr::Not(a) => Expr::Not(Box::new(norm(a))),
        Expr::Cast(a, t) => Expr::Cast(Box::new(norm(a)), t.clone()),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(norm(a)), *n),
        Expr::Like(a, patterns, negated, regex) => Expr::Like(Box::new(norm(a)), patterns.clone(), *negated, *regex),
        Expr::In(a, l, n) => Expr::In(Box::new(norm(a)), l.iter().map(norm).collect(), *n),
        Expr::Between(a, b, c, n) => Expr::Between(Box::new(norm(a)), Box::new(norm(b)), Box::new(norm(c)), *n),
        other => other.clone(),
    }
}

/// Replace sub-expressions equal to one of `known` with column refs.
fn substitute(e: &Expr, known: &[(Expr, String)]) -> Expr {
    let ne = norm(e);
    if let Some((_, name)) = known.iter().find(|(k, _)| *k == ne) {
        return Expr::Col(name.clone(), 0);
    }
    match e {
        Expr::Call(n, a, p) => Expr::Call(n.clone(), a.iter().map(|x| substitute(x, known)).collect(), *p),
        Expr::Bin(o, a, b) => Expr::Bin(*o, Box::new(substitute(a, known)), Box::new(substitute(b, known))),
        Expr::Neg(a) => Expr::Neg(Box::new(substitute(a, known))),
        Expr::Not(a) => Expr::Not(Box::new(substitute(a, known))),
        Expr::Cast(a, t) => Expr::Cast(Box::new(substitute(a, known)), t.clone()),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(substitute(a, known)), *n),
        Expr::Like(a, patterns, negated, regex) => Expr::Like(Box::new(substitute(a, known)), patterns.clone(), *negated, *regex),
        Expr::In(a, l, n) => Expr::In(Box::new(substitute(a, known)), l.iter().map(|x| substitute(x, known)).collect(), *n),
        Expr::Between(a, b, c, n) => Expr::Between(Box::new(substitute(a, known)), Box::new(substitute(b, known)), Box::new(substitute(c, known)), *n),
        other => other.clone(),
    }
}

/// Resolve SELECT aliases in HAVING before lowering aggregate expressions.
/// Aggregate arguments remain in source-column scope, not SELECT-alias scope.
fn having_aliases(e: &Expr, items: &[Field]) -> Expr {
    match e {
        Expr::Col(n, _) => items.iter().find(|i| &i.name == n).map(|i| i.expr.clone()).unwrap_or_else(|| e.clone()),
        Expr::Call(n, _, _) if func::AGGS.contains(&n.as_str()) => e.clone(),
        Expr::Call(n, a, p) => Expr::Call(n.clone(), a.iter().map(|x| having_aliases(x, items)).collect(), *p),
        Expr::Bin(o, a, b) => Expr::Bin(*o, Box::new(having_aliases(a, items)), Box::new(having_aliases(b, items))),
        Expr::Neg(a) => Expr::Neg(Box::new(having_aliases(a, items))),
        Expr::Not(a) => Expr::Not(Box::new(having_aliases(a, items))),
        Expr::Cast(a, t) => Expr::Cast(Box::new(having_aliases(a, items)), t.clone()),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(having_aliases(a, items)), *n),
        Expr::Like(a, patterns, negated, regex) => Expr::Like(Box::new(having_aliases(a, items)), patterns.clone(), *negated, *regex),
        Expr::In(a, l, n) => Expr::In(Box::new(having_aliases(a, items)), l.iter().map(|x| having_aliases(x, items)).collect(), *n),
        Expr::Between(a, b, c, n) => Expr::Between(Box::new(having_aliases(a, items)), Box::new(having_aliases(b, items)), Box::new(having_aliases(c, items)), *n),
        other => other.clone(),
    }
}

/// Collect aggregate calls (normalised) in `e`.
fn agg_calls(e: &Expr, out: &mut Vec<Expr>) {
    match e {
        Expr::Call(n, a, _) if func::AGGS.contains(&n.as_str()) => {
            let ne = norm(e);
            if !out.contains(&ne) {
                out.push(ne);
            }
            let _ = a;
        }
        Expr::Call(_, a, _) => a.iter().for_each(|x| agg_calls(x, out)),
        Expr::Bin(_, a, b) => {
            agg_calls(a, out);
            agg_calls(b, out);
        }
        Expr::Neg(a) | Expr::Not(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) | Expr::Like(a, ..) => agg_calls(a, out),
        Expr::In(a, l, _) => {
            agg_calls(a, out);
            l.iter().for_each(|x| agg_calls(x, out));
        }
        Expr::Between(a, b, c, _) => {
            agg_calls(a, out);
            agg_calls(b, out);
            agg_calls(c, out);
        }
        _ => {}
    }
}

fn parse_pivot(p: &mut Parser<'_>) -> Result<Cmd> {
    let pos = p.pos();
    p.expect(&Tok::LParen, "(")?;
    let aggregate = p.expr()?;
    let mut calls = Vec::new();
    agg_calls(&aggregate, &mut calls);
    if calls.len() != 1 { return Err(EsError::new(400, "verification_exception", "PIVOT requires one aggregate expression")); }
    if !p.eat_kw("FOR") { return Err(p.unexpected("FOR")); }
    let column = p.qualified()?;
    if !p.eat_kw("IN") { return Err(p.unexpected("IN")); }
    p.expect(&Tok::LParen, "(")?;
    let mut values = Vec::new();
    loop {
        let value = match p.expr()? {
            Expr::Lit(v) if !matches!(v, Val::Null) => v,
            Expr::Neg(e) => match *e { Expr::Lit(Val::Int(n)) => Val::Int(-n), Expr::Lit(Val::Double(n)) => Val::Double(-n), _ => return Err(p.unexpected("literal pivot value")) },
            _ => return Err(p.unexpected("literal pivot value")),
        };
        let name = if p.eat_kw("AS") || matches!(p.peek(), Tok::Ident(_) | Tok::QIdent(_)) { p.ident()? } else {
            match &value { Val::Str(s) => s.clone(), _ => value.to_json().to_string() }
        };
        if values.iter().any(|(_, n)| n == &name) { return Err(EsError::new(400, "verification_exception", format!("Duplicate pivot column [{name}]"))); }
        values.push((value, name));
        if !p.eat(&Tok::Comma) { break; }
    }
    p.expect(&Tok::RParen, ")")?;
    p.expect(&Tok::RParen, ")")?;
    Ok(Cmd::SqlPivot { aggregate, column, values, pos })
}

pub fn parse(full: &str, params: &Params) -> Result<Statement> {
    crate::parse_depth::guarded(|| parse_unguarded(full, params))
}

fn parse_unguarded(full: &str, params: &Params) -> Result<Statement> {
    parse_with_counter(full, params, &std::cell::Cell::new(0))
}

fn parse_with_counter(full: &str, params: &Params, np: &std::cell::Cell<usize>) -> Result<Statement> {
    let trimmed = full.trim().trim_end_matches(';');
    let toks = lex(trimmed, 0, trimmed.len(), Dialect::Sql)?;
    // ES `SqlParser.invokeParser`: the depth pre-scan, before parsing.
    if crate::esql::lexer::prefix_depth_exceeds(&toks, crate::esql::lexer::QL_MAX_EXPRESSION_DEPTH) {
        return Err(EsError::new(400, "parsing_exception", format!("line -1:0: SQL statement exceeded the maximum expression depth allowed ({})", crate::esql::lexer::QL_MAX_EXPRESSION_DEPTH)));
    }
    let first = match &toks[0].tok {
        Tok::Ident(s) => s.to_ascii_uppercase(),
        _ => return Err(sql_err(trimmed, 0, "mismatched input expecting SELECT")),
    };
    let word = |i: usize| match toks.get(i).map(|t| &t.tok) {
        Some(Tok::Ident(s)) => s.to_ascii_uppercase(),
        _ => String::new(),
    };
    let rest_text = |i: usize| -> String { toks.get(i).map(|t| trimmed[t.start..].trim().to_string()).unwrap_or_default() };
    match first.as_str() {
        "SHOW" => {
            return Ok(match word(1).as_str() {
                "TABLES" => {
                    let pat = if word(2) == "LIKE" { toks.get(3).and_then(|t| if let Tok::Str(s) = &t.tok { Some(s.clone()) } else { None }) } else { None };
                    Statement::ShowTables(pat)
                }
                "COLUMNS" => {
                    let i = if word(2) == "FROM" || word(2) == "IN" { 3 } else { 2 };
                    Statement::ShowColumns(unquote(&rest_text(i)))
                }
                "FUNCTIONS" => Statement::ShowFunctions,
                _ => return Err(sql_err(trimmed, toks[1].start, "mismatched input expecting {TABLES, COLUMNS, FUNCTIONS}")),
            });
        }
        "DESCRIBE" | "DESC" => return Ok(Statement::ShowColumns(unquote(&rest_text(1)))),
        // ES SQL's system statements, which the JDBC driver's
        // `DatabaseMetaData` issues (`SysTables`, `SysColumns`, `SysTypes`).
        "SYS" => {
            let mut i = 2;
            // (The lexer ends the tokens with `Eof`.)
            let toks = &toks[..toks.iter().position(|t| matches!(t.tok, Tok::Eof)).unwrap_or(toks.len())];
            let lit = |i: usize| -> Option<String> {
                match toks.get(i).map(|t| &t.tok) {
                    Some(Tok::Str(v)) => Some(v.clone()),
                    Some(Tok::Ident(v)) | Some(Tok::QIdent(v)) => Some(v.clone()),
                    _ => None,
                }
            };
            // `[LIKE] value` at `i`: the value and the next position.
            let pattern = |i: usize| -> (Option<String>, usize) {
                if word(i) == "LIKE" { (lit(i + 1), i + 2) } else { (lit(i), i + 1) }
            };
            return Ok(match word(1).as_str() {
                "TABLES" => {
                    let (mut catalog, mut table, mut types) = (None, None, None);
                    while i < toks.len() {
                        match word(i).as_str() {
                            "CATALOG" => {
                                let (v, n) = pattern(i + 1);
                                catalog = v;
                                i = n;
                            }
                            "TYPE" => {
                                let mut ts = Vec::new();
                                i += 1;
                                while let Some(v) = lit(i) {
                                    ts.push(v);
                                    i += 1;
                                    if matches!(toks.get(i).map(|t| &t.tok), Some(Tok::Comma)) {
                                        i += 1;
                                    } else {
                                        break;
                                    }
                                }
                                types = Some(ts);
                            }
                            _ => {
                                let (v, n) = pattern(i);
                                if v.is_none() {
                                    return Err(sql_err(trimmed, toks[i].start, "mismatched input in SYS TABLES"));
                                }
                                table = v;
                                i = n;
                            }
                        }
                    }
                    Statement::SysTables { catalog, table, types }
                }
                "COLUMNS" => {
                    let (mut table, mut column) = (None, None);
                    while i < toks.len() {
                        match word(i).as_str() {
                            "CATALOG" => i = pattern(i + 1).1,
                            "TABLE" => {
                                let (v, n) = pattern(i + 1);
                                table = v;
                                i = n;
                            }
                            _ => {
                                let (v, n) = pattern(i);
                                if v.is_none() {
                                    return Err(sql_err(trimmed, toks[i].start, "mismatched input in SYS COLUMNS"));
                                }
                                column = v;
                                i = n;
                            }
                        }
                    }
                    Statement::SysColumns { table, column }
                }
                "TYPES" => {
                    let neg = matches!(toks.get(2).map(|t| &t.tok), Some(Tok::Op(o)) if o.to_string() == "-");
                    let n = toks.get(if neg { 3 } else { 2 }).and_then(|t| if let Tok::Int(n) = &t.tok { Some(*n) } else { None });
                    Statement::SysTypes(n.map(|n| if neg { -n } else { n }))
                }
                _ => return Err(sql_err(trimmed, toks.get(1).map(|t| t.start).unwrap_or(0), "mismatched input expecting {TABLES, COLUMNS, TYPES}")),
            });
        }
        "SELECT" => {}
        _ => return Err(sql_err(trimmed, 0, format!("mismatched input '{first}' expecting {{'(', 'DESCRIBE', 'SELECT', 'SHOW'}}"))),
    }
    let mut p = Parser::new(trimmed, 0, trimmed.len(), Dialect::Sql, params, np)?;
    p.i = 1;
    let distinct = p.eat_kw("DISTINCT");
    let mut top: Option<usize> = None;
    if p.eat_kw("TOP") {
        if let Tok::Int(n) = p.peek().clone() {
            p.i += 1;
            top = Some(n.max(0) as usize);
        }
    }
    // Select list.
    let mut star = false;
    let mut items: Vec<Field> = Vec::new();
    loop {
        if matches!(p.peek(), Tok::Op("*")) {
            p.i += 1;
            star = true;
        } else {
            let start = p.pos();
            let expr = p.expr()?;
            let text = p.text(start, p.i);
            let name = if p.eat_kw("AS") {
                p.ident()?
            } else if matches!(p.peek(), Tok::Ident(s) if !CLAUSES.iter().any(|c| s.eq_ignore_ascii_case(c)) && !s.eq_ignore_ascii_case("FROM")) || matches!(p.peek(), Tok::QIdent(_)) {
                p.ident()?
            } else {
                match &expr {
                    Expr::Col(n, _) => n.clone(),
                    _ => text,
                }
            };
            items.push(Field { name, expr, filter: None, pos: start });
        }
        if !p.eat(&Tok::Comma) {
            break;
        }
    }
    // FROM.
    let mut index: Option<String> = None;
    let mut input_commands = None;
    // A subquery's output columns (its final projection), for PIVOT's groups.
    let mut input_names: Option<Vec<String>> = None;
    // Table qualifiers columns may carry (`FROM emp AS e` → `e.emp_no`; the
    // table name itself without an alias).
    let mut quals: Vec<String> = Vec::new();
    if p.eat_kw("FROM") {
        if p.eat(&Tok::LParen) {
            let start = p.pos();
            let mut depth = 1;
            while !p.at_eof() {
                match p.peek() { Tok::LParen => depth += 1, Tok::RParen => { depth -= 1; if depth == 0 { break; } }, _ => {} }
                p.i += 1;
            }
            if depth != 0 { return Err(p.unexpected(")")); }
            let Statement::Select(inner) = parse_with_counter(&trimmed[start..p.pos()], params, np)? else {
                return Err(p.unexpected("SELECT"));
            };
            index = inner.index;
            if let Some(Cmd::Keep(names, _)) = inner.cmds.last() {
                if !names.iter().any(|n| n.contains('*')) { input_names = Some(names.clone()); }
            }
            input_commands = Some(inner.cmds);
            p.expect(&Tok::RParen, ")")?;
            if p.eat_kw("AS") || matches!(p.peek(), Tok::QIdent(_)) || matches!(p.peek(), Tok::Ident(n) if !CLAUSES.iter().any(|c| n.eq_ignore_ascii_case(c))) { quals.push(p.ident()?); }
        } else {
        let start = p.pos();
        let mut end = start;
        while !p.at_eof() && !CLAUSES.iter().any(|c| p.at_kw(c)) && !p.at_kw("AS") {
            end = p.toks[p.i].end;
            p.i += 1;
        }
        if p.eat_kw("AS") {
            quals.push(p.ident()?);
        }
        let raw = trimmed[start..end].trim();
        // `idx alias`
        let raw = if !raw.starts_with('"') && raw.contains(' ') {
            let mut words = raw.split_whitespace();
            let table = words.next().unwrap_or(raw);
            if let Some(alias) = words.next() {
                quals.push(alias.to_string());
            }
            table
        } else {
            raw
        };
        if raw.is_empty() {
            return Err(sql_err(trimmed, start, "mismatched input expecting a table name"));
        }
        index = Some(unquote(raw));
        if quals.is_empty() && !raw.contains(['*', ',']) {
            quals.push(unquote(raw));
        }
        }
    }
    let pivot = if p.eat_kw("PIVOT") { Some(parse_pivot(&mut p)?) } else { None };
    let where_ = if p.eat_kw("WHERE") { Some(p.expr()?) } else { None };
    let mut group: Vec<Expr> = Vec::new();
    if p.eat_kw("GROUP") {
        if !p.eat_kw("BY") {
            return Err(p.unexpected("BY"));
        }
        loop {
            group.push(p.expr()?);
            if !p.eat(&Tok::Comma) {
                break;
            }
        }
    }
    let having = if p.eat_kw("HAVING") { Some(having_aliases(&p.expr()?, &items)) } else { None };
    let mut order: Vec<SortKey> = Vec::new();
    if p.eat_kw("ORDER") {
        if !p.eat_kw("BY") {
            return Err(p.unexpected("BY"));
        }
        loop {
            let expr = p.expr()?;
            let desc = if p.eat_kw("DESC") {
                true
            } else {
                p.eat_kw("ASC");
                false
            };
            let nulls_first = if p.eat_kw("NULLS") {
                if p.eat_kw("FIRST") {
                    Some(true)
                } else {
                    p.eat_kw("LAST");
                    Some(false)
                }
            } else {
                // SQL default: nulls last for ASC, first for DESC? ES SQL: nulls last.
                Some(false)
            };
            order.push(SortKey { expr, desc, nulls_first });
            if !p.eat(&Tok::Comma) {
                break;
            }
        }
    }
    // Qualified columns (`e.emp_no`) name the table's own columns (ES SQL
    // resolves the qualifier against the FROM table or its alias).
    let (mut items, where_, group, having, order) = if quals.is_empty() {
        (items, where_, group, having, order)
    } else {
        for it in items.iter_mut() {
            let unnamed = matches!(&it.expr, Expr::Col(n, _) if *n == it.name);
            it.expr = strip_qualifiers(&it.expr, &quals);
            if let (true, Expr::Col(n, _)) = (unnamed, &it.expr) {
                it.name = n.clone();
            }
        }
        (
            items,
            where_.map(|e| strip_qualifiers(&e, &quals)),
            group.iter().map(|e| strip_qualifiers(e, &quals)).collect(),
            having.map(|e| strip_qualifiers(&e, &quals)),
            order.into_iter().map(|k| SortKey { expr: strip_qualifiers(&k.expr, &quals), ..k }).collect(),
        )
    };
    let mut limit = top;
    if p.eat_kw("LIMIT") {
        match p.peek().clone() {
            Tok::Int(n) => {
                p.i += 1;
                limit = Some(n.max(0) as usize);
            }
            _ => return Err(p.unexpected("INTEGER_VALUE")),
        }
    }
    if !p.at_eof() {
        return Err(p.unexpected("<EOF>"));
    }
    // ---- compile
    let mut cmds: Vec<Cmd> = Vec::new();
    if let Some(inner) = input_commands {
        cmds.extend(inner);
    } else {
    match &index {
        Some(ix) => cmds.push(Cmd::From { indices: vec![ix.clone()], metadata: vec![], pos: 0, ts: false, subqueries: vec![] }),
        None => cmds.push(Cmd::Row(vec![])),
    }
    }
    // ES runs PIVOT as a composite aggregation over (groups.., column): its
    // rows come in ascending group order (missing first).
    let mut pivot_groups: Vec<String> = Vec::new();
    if let (Some(Cmd::SqlPivot { aggregate, column, .. }), Some(names)) = (&pivot, &input_names) {
        let mut refs = Vec::new();
        crate::esql::analyze::expr_refs(aggregate, &mut refs);
        pivot_groups = names.iter().filter(|n| n.as_str() != column.as_str() && !refs.iter().any(|(r, _)| r.as_str() == n.as_str())).cloned().collect();
    }
    if let Some(pivot) = pivot { cmds.push(pivot); }
    if let Some(w) = where_ {
        if contains_agg(&w) {
            return Err(EsError::new(400, "verification_exception", "Found 1 problem\nCannot use WHERE filtering on aggregate function, use HAVING instead"));
        }
        cmds.push(Cmd::Where(w, 0));
    }
    let aggregated = !group.is_empty() || items.iter().any(|i| contains_agg(&i.expr)) || having.is_some() || distinct;
    let mut final_names: Vec<String> = items.iter().map(|i| i.name.clone()).collect();
    if aggregated {
        if star {
            return Err(EsError::new(400, "verification_exception", "Found 1 problem\nCannot use * with GROUP BY or aggregates"));
        }
        // Groups (GROUP BY alias resolves to the select expression).
        let mut by: Vec<Field> = Vec::new();
        let groups: Vec<Expr> = if distinct && group.is_empty() { items.iter().map(|i| i.expr.clone()).collect() } else { group.clone() };
        for (gi, g) in groups.iter().enumerate() {
            let resolved = match g {
                Expr::Col(n, _) => items.iter().find(|i| &i.name == n && !matches!(&i.expr, Expr::Col(m, _) if m == n)).map(|i| (i.expr.clone(), i.name.clone())),
                Expr::Lit(Val::Int(k)) if *k >= 1 && (*k as usize) <= items.len() => Some((items[*k as usize - 1].expr.clone(), items[*k as usize - 1].name.clone())),
                _ => None,
            };
            let (expr, name) = match resolved {
                Some(x) => x,
                None => {
                    let name = match g {
                        Expr::Col(n, _) => n.clone(),
                        _ => items.iter().find(|i| norm(&i.expr) == norm(g)).map(|i| i.name.clone()).unwrap_or_else(|| format!("__group{gi}")),
                    };
                    (g.clone(), name)
                }
            };
            by.push(Field { name, expr, filter: None, pos: 0 });
        }
        let mut known: Vec<(Expr, String)> = by.iter().map(|b| (norm(&b.expr), b.name.clone())).collect();
        // Aggregates needed by SELECT, HAVING, ORDER BY.
        let mut calls: Vec<Expr> = Vec::new();
        for i in &items {
            agg_calls(&i.expr, &mut calls);
        }
        if let Some(h) = &having {
            agg_calls(h, &mut calls);
        }
        for o in &order {
            agg_calls(&o.expr, &mut calls);
        }
        let mut aggs: Vec<Field> = Vec::new();
        for (k, c) in calls.iter().enumerate() {
            let name = format!("__agg{k}");
            aggs.push(Field { name: name.clone(), expr: c.clone(), filter: None, pos: 0 });
            known.push((c.clone(), name));
        }
        if aggs.is_empty() && by.is_empty() {
            return Err(EsError::new(400, "verification_exception", "Found 1 problem\nnothing to aggregate"));
        }
        cmds.push(Cmd::Stats { aggs, by: by.clone(), inline: false, pos: 0 });
        if let Some(h) = having {
            let h = substitute(&h, &known);
            cmds.push(Cmd::Where(h, 0));
        }
        // Projections over the aggregated table.
        let mut evals: Vec<Field> = Vec::new();
        for i in &items {
            let e = substitute(&i.expr, &known);
            if !matches!(&e, Expr::Col(n, _) if n == &i.name) {
                evals.push(Field { name: i.name.clone(), expr: e, filter: None, pos: 0 });
            }
        }
        // Order keys may reference select aliases or raw expressions.
        let mut sort_keys = Vec::new();
        for o in &order {
            let e = match &o.expr {
                Expr::Lit(Val::Int(k)) if *k >= 1 && (*k as usize) <= items.len() => Expr::Col(items[*k as usize - 1].name.clone(), 0),
                Expr::Col(n, _) if items.iter().any(|i| &i.name == n) => Expr::Col(n.clone(), 0),
                other => substitute(other, &known),
            };
            sort_keys.push(SortKey { expr: e, desc: o.desc, nulls_first: o.nulls_first });
        }
        // ES groups with a composite aggregation: groups come in ascending
        // key order (missing first), and an ORDER BY sorts them stably.
        group_order(&mut sort_keys, by.iter().map(|b| b.name.as_str()), &evals);
        projection_cmds(evals, false, &mut cmds);
        if !sort_keys.is_empty() {
            cmds.push(Cmd::Sort(sort_keys));
        }
    } else {
        let mut evals: Vec<Field> = Vec::new();
        for i in &items {
            if !matches!(&i.expr, Expr::Col(n, _) if n == &i.name) {
                evals.push(i.clone());
            }
        }
        let shadowed = evals.clone();
        projection_cmds(evals, star, &mut cmds);
        if !order.is_empty() || !pivot_groups.is_empty() {
            let mut keys: Vec<SortKey> = order
                .iter()
                .map(|o| SortKey {
                    expr: match &o.expr {
                        Expr::Lit(Val::Int(k)) if *k >= 1 && (*k as usize) <= items.len() => Expr::Col(items[*k as usize - 1].name.clone(), 0),
                        e => e.clone(),
                    },
                    desc: o.desc,
                    nulls_first: o.nulls_first,
                })
                .collect();
            group_order(&mut keys, pivot_groups.iter().map(String::as_str), &shadowed);
            cmds.push(Cmd::Sort(keys));
        }
    }
    if let Some(n) = limit {
        cmds.push(Cmd::Limit(n));
    }
    if star {
        final_names.clear();
    }
    if !final_names.is_empty() {
        cmds.push(Cmd::Keep(final_names, 0));
    }
    Ok(Statement::Select(SqlQuery { cmds, index, limit }))
}

/// The EVALs of a SELECT list. SQL evaluates every projection against the
/// input row, while an ES|QL EVAL field sees the columns the fields before it
/// assign: a projection reading a name an earlier projection assigns
/// (`SELECT a AS b, b AS a`) is evaluated first into a temporary column,
/// assigned from it after, and the temporaries are dropped (`drop`: no KEEP
/// follows to drop them, as for `SELECT *, ..`).
fn projection_cmds(evals: Vec<Field>, drop: bool, cmds: &mut Vec<Cmd>) {
    if evals.is_empty() {
        return;
    }
    let mut first: Vec<Field> = Vec::new();
    let mut second: Vec<Field> = Vec::with_capacity(evals.len());
    for (j, f) in evals.iter().enumerate() {
        let mut refs = Vec::new();
        crate::esql::analyze::expr_refs(&f.expr, &mut refs);
        if let Some(filter) = &f.filter {
            crate::esql::analyze::expr_refs(filter, &mut refs);
        }
        if refs.iter().any(|(r, _)| evals[..j].iter().any(|e| &e.name == r)) {
            let tmp = format!("__sql_projection{j}");
            first.push(Field { name: tmp.clone(), expr: f.expr.clone(), filter: f.filter.clone(), pos: f.pos });
            second.push(Field { name: f.name.clone(), expr: Expr::Col(tmp, f.pos), filter: None, pos: f.pos });
        } else {
            second.push(f.clone());
        }
    }
    let temps: Vec<String> = first.iter().map(|f| f.name.clone()).collect();
    if !first.is_empty() {
        cmds.push(Cmd::Eval(first));
    }
    cmds.push(Cmd::Eval(second));
    if drop && !temps.is_empty() {
        cmds.push(Cmd::Drop(temps, 0));
    }
}

/// Append each group key not already ordered on as an ascending,
/// missing-first tiebreaker (skipping names a projection redefines).
fn group_order<'a>(keys: &mut Vec<SortKey>, groups: impl Iterator<Item = &'a str>, evals: &[Field]) {
    for name in groups {
        if evals.iter().any(|f| f.name == name) || keys.iter().any(|k| matches!(&k.expr, Expr::Col(n, _) if n == name)) { continue; }
        keys.push(SortKey { expr: Expr::Col(name.to_string(), 0), desc: false, nulls_first: Some(true) });
    }
}

fn unquote(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2 && (t.starts_with('"') && t.ends_with('"') || t.starts_with('`') && t.ends_with('`')) {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

/// ES SQL type name for an ES|QL column type.
pub fn sql_type(t: &str) -> &str {
    match t {
        "date" | "date_nanos" => "datetime",
        "unsupported" => "unsupported",
        other => other,
    }
}

fn str_col(n: &str) -> Column {
    Column { name: n.to_string(), ty: "keyword".into() }
}

fn typed_col(n: &str, ty: &str) -> Column {
    Column { name: n.to_string(), ty: ty.into() }
}

/// The `SYS COLUMNS` description of an ES SQL column type (ES
/// `SqlDataTypes`): JDBC type name, `java.sql.Types` code, display size
/// (`COLUMN_SIZE`), byte size (`BUFFER_LENGTH`), decimal digits, radix,
/// SQL data type and datetime subcode.
struct JdbcType {
    name: &'static str,
    code: i64,
    display: i64,
    size: i64,
    digits: Option<i64>,
    radix: Option<i64>,
    sql_type: i64,
    sub: i64,
}

fn jdbc_type(ty: &str) -> JdbcType {
    const MAX: i64 = i32::MAX as i64;
    let t = |name: &'static str, code: i64, display: i64, size: i64, digits: Option<i64>, radix: Option<i64>, sql_type: i64, sub: i64| JdbcType { name, code, display, size, digits, radix, sql_type, sub };
    match ty {
        "keyword" | "constant_keyword" | "wildcard" => t("KEYWORD", 12, 32766, MAX, None, None, 12, 0),
        "text" | "match_only_text" => t("TEXT", 12, MAX, MAX, None, None, 12, 0),
        "ip" => t("IP", 12, 45, 45, None, None, 12, 0),
        "version" => t("VERSION", 12, MAX, MAX, None, None, 12, 0),
        "long" | "counter_long" => t("LONG", -5, 20, 8, None, Some(10), -5, 0),
        "unsigned_long" => t("UNSIGNED_LONG", 2, 20, 8, None, Some(10), 2, 0),
        "integer" | "counter_integer" => t("INTEGER", 4, 11, 4, None, Some(10), 4, 0),
        "short" => t("SHORT", 5, 6, 2, None, Some(10), 5, 0),
        "byte" => t("BYTE", -6, 5, 1, None, Some(10), -6, 0),
        "double" | "counter_double" => t("DOUBLE", 8, 25, 8, Some(15), Some(2), 8, 0),
        "scaled_float" => t("SCALED_FLOAT", 8, 25, 8, Some(15), Some(2), 8, 0),
        "float" => t("FLOAT", 7, 15, 4, Some(7), Some(2), 7, 0),
        "half_float" => t("HALF_FLOAT", 6, 25, 2, Some(3), Some(2), 6, 0),
        "boolean" => t("BOOLEAN", 16, 1, 1, None, None, 16, 0),
        "date" | "datetime" | "date_nanos" => t("DATETIME", 93, 34, 8, Some(9), None, 9, 3),
        "binary" => t("BINARY", -2, MAX, MAX, None, None, -2, 0),
        "geo_point" => t("GEO_POINT", 114, 58, 58, None, None, 114, 0),
        "geo_shape" => t("GEO_SHAPE", 114, MAX, MAX, None, None, 114, 0),
        "shape" => t("SHAPE", 114, MAX, MAX, None, None, 114, 0),
        _ => t("UNSUPPORTED", 1111, 0, 0, None, None, 1111, 0),
    }
}

/// `SYS TABLES`: the JDBC `getTables` / `getCatalogs` / `getTableTypes` rows.
fn sys_tables(node: &Node, authorize: Option<&exec::AuthorizeIndices>, catalog: Option<&str>, table: Option<&str>, types: Option<&[String]>) -> Result<Table> {
    let cols: Vec<Column> = ["TABLE_CAT", "TABLE_SCHEM", "TABLE_NAME", "TABLE_TYPE", "REMARKS", "TYPE_CAT", "TYPE_SCHEM", "TYPE_NAME", "SELF_REFERENCING_COL_NAME", "REF_GENERATION"].iter().map(|n| str_col(n)).collect();
    let null_row = |cat: Val, ty: Val| -> Vec<Val> {
        let mut r = vec![Val::Null; 10];
        r[0] = cat;
        r[3] = ty;
        r
    };
    let cluster = node.cluster_name.clone();
    // `getCatalogs`: CATALOG LIKE '%' LIKE '' (no table).
    if catalog == Some("%") && table == Some("") && types.is_none() {
        return Ok(Table { cols, rows: vec![null_row(Val::Str(cluster), Val::Null)] });
    }
    // `getTableTypes`: CATALOG LIKE '' LIKE '' TYPE '%'.
    if catalog == Some("") && table == Some("") && types.is_some_and(|t| t == ["%"]) {
        return Ok(Table { cols, rows: vec![null_row(Val::Null, Val::Str("TABLE".into())), null_row(Val::Null, Val::Str("VIEW".into()))] });
    }
    if catalog.is_some_and(|c| !c.is_empty() && !wildcard_match(c, &cluster)) {
        return Ok(Table { cols, rows: Vec::new() });
    }
    let want = |t: &str| types.is_none_or(|ts| ts.iter().any(|x| x == "%" || x.eq_ignore_ascii_case(t)));
    let visible: Option<std::collections::HashSet<String>> = match authorize {
        Some(a) => a.indices("*", &crate::node::ResolveOpts::from_params(&HashMap::new()), false)?.map(|r| r.names.into_iter().collect()),
        None => None,
    };
    let shown = |name: &str| visible.as_ref().is_none_or(|v| v.contains(name)) && table.is_none_or(|p| p.is_empty() || wildcard_match(p, name));
    let mut rows: Vec<Vec<Val>> = Vec::new();
    for svc in node.all_indices() {
        if svc.name.starts_with('.') {
            continue;
        }
        if want("TABLE") && shown(&svc.name) {
            let mut r = null_row(Val::Str(cluster.clone()), Val::Str("TABLE".into()));
            r[2] = Val::Str(svc.name.clone());
            r[4] = Val::Str(String::new());
            rows.push(r);
        }
        if want("VIEW") {
            for a in svc.aliases().keys().filter(|a| shown(a)) {
                let mut r = null_row(Val::Str(cluster.clone()), Val::Str("VIEW".into()));
                r[2] = Val::Str(a.clone());
                r[4] = Val::Str(String::new());
                rows.push(r);
            }
        }
    }
    rows.sort_by(|a, b| (a[3].to_text(), a[2].to_text()).cmp(&(b[3].to_text(), b[2].to_text())));
    rows.dedup();
    Ok(Table { cols, rows })
}

/// `SYS COLUMNS`: the JDBC `getColumns` rows of every visible table matching
/// `table`, columns matching `column`.
fn sys_columns(node: &Node, authorize: Option<&exec::AuthorizeIndices>, table: Option<&str>, column: Option<&str>) -> Result<Table> {
    let names = ["TABLE_CAT", "TABLE_SCHEM", "TABLE_NAME", "COLUMN_NAME", "DATA_TYPE", "TYPE_NAME", "COLUMN_SIZE", "BUFFER_LENGTH", "DECIMAL_DIGITS", "NUM_PREC_RADIX", "NULLABLE", "REMARKS", "COLUMN_DEF", "SQL_DATA_TYPE", "SQL_DATETIME_SUB", "CHAR_OCTET_LENGTH", "ORDINAL_POSITION", "IS_NULLABLE", "SCOPE_CATALOG", "SCOPE_SCHEMA", "SCOPE_TABLE", "SOURCE_DATA_TYPE", "IS_AUTOINCREMENT", "IS_GENERATEDCOLUMN"];
    let ints = ["DATA_TYPE", "COLUMN_SIZE", "BUFFER_LENGTH", "DECIMAL_DIGITS", "NUM_PREC_RADIX", "NULLABLE", "SQL_DATA_TYPE", "SQL_DATETIME_SUB", "CHAR_OCTET_LENGTH", "ORDINAL_POSITION"];
    let cols: Vec<Column> = names.iter().map(|n| if ints.contains(n) { typed_col(n, "integer") } else if *n == "SOURCE_DATA_TYPE" { typed_col(n, "short") } else { str_col(n) }).collect();
    let tables = sys_tables(node, authorize, None, Some(table.unwrap_or("%")), None)?;
    let mut rows = Vec::new();
    for t in &tables.rows {
        let tname = t[2].to_text();
        let Ok(columns) = exec::index_columns_as(node, &[tname.clone()], authorize) else { continue };
        // Ordinal positions follow the column names' order.
        let mut columns: Vec<_> = columns.into_iter().filter(|c| !(c.ty.starts_with("unsupported") || c.ty == "object" || c.ty == "nested")).collect();
        columns.sort_by(|a, b| a.name.cmp(&b.name));
        for (i, c) in columns.into_iter().enumerate() {
            if column.is_some_and(|p| !p.is_empty() && !wildcard_match(p, &c.name)) {
                continue;
            }
            let j = jdbc_type(&c.ty);
            rows.push(vec![
                Val::Str(node.cluster_name.clone()),
                Val::Null,
                Val::Str(tname.clone()),
                Val::Str(c.name.clone()),
                Val::Int(j.code),
                Val::Str(j.name.into()),
                Val::Int(j.display),
                Val::Int(j.size),
                j.digits.map(Val::Int).unwrap_or(Val::Null),
                j.radix.map(Val::Int).unwrap_or(Val::Null),
                Val::Int(1),
                Val::Null,
                Val::Null,
                Val::Int(j.sql_type),
                Val::Int(j.sub),
                if j.code == 12 { Val::Int(j.size) } else { Val::Null },
                Val::Int(i as i64 + 1),
                Val::Str("YES".into()),
                Val::Null,
                Val::Null,
                Val::Null,
                Val::Null,
                Val::Str("NO".into()),
                Val::Str("NO".into()),
            ]);
        }
    }
    Ok(Table { cols, rows })
}

/// ES SQL's data types as `SYS TYPES` lists them (ES 9.5.5): name,
/// JDBC type, precision, minimum/maximum scale, SQL data type, datetime
/// subcode, radix, in `DATA_TYPE` order.
const SQL_TYPES: &[(&str, i64, i64, Option<i64>, i64, i64, Option<i64>)] = &[
    ("BYTE", -6, 3, Some(0), -6, 0, Some(10)),
    ("LONG", -5, 19, Some(0), -5, 0, Some(10)),
    ("BINARY", -2, 2147483647, None, -2, 0, None),
    ("NULL", 0, 0, None, 0, 0, None),
    ("UNSIGNED_LONG", 2, 20, Some(0), 2, 0, Some(10)),
    ("INTEGER", 4, 10, Some(0), 4, 0, Some(10)),
    ("SHORT", 5, 5, Some(0), 5, 0, Some(10)),
    ("HALF_FLOAT", 6, 3, Some(3), 6, 0, Some(2)),
    ("FLOAT", 7, 7, Some(7), 7, 0, Some(2)),
    ("DOUBLE", 8, 15, Some(15), 8, 0, Some(2)),
    ("SCALED_FLOAT", 8, 15, Some(15), 8, 0, Some(2)),
    ("IP", 12, 45, None, 12, 0, None),
    ("KEYWORD", 12, 32766, None, 12, 0, None),
    ("TEXT", 12, 2147483647, None, 12, 0, None),
    ("VERSION", 12, 2147483647, None, 12, 0, None),
    ("BOOLEAN", 16, 1, None, 16, 0, None),
    ("DATE", 91, 29, None, 91, 1, None),
    ("TIME", 92, 24, Some(9), 92, 2, None),
    ("DATETIME", 93, 34, Some(9), 9, 3, None),
    ("INTERVAL_YEAR", 101, 7, None, 101, 0, None),
    ("INTERVAL_MONTH", 102, 7, None, 102, 0, None),
    ("INTERVAL_DAY", 103, 23, None, 103, 0, None),
    ("INTERVAL_HOUR", 104, 23, None, 104, 0, None),
    ("INTERVAL_MINUTE", 105, 23, None, 105, 0, None),
    ("INTERVAL_SECOND", 106, 23, None, 106, 0, None),
    ("INTERVAL_YEAR_TO_MONTH", 107, 7, None, 107, 0, None),
    ("INTERVAL_DAY_TO_HOUR", 108, 23, None, 108, 0, None),
    ("INTERVAL_DAY_TO_MINUTE", 109, 23, None, 109, 0, None),
    ("INTERVAL_DAY_TO_SECOND", 110, 23, None, 110, 0, None),
    ("INTERVAL_HOUR_TO_MINUTE", 111, 23, None, 111, 0, None),
    ("INTERVAL_HOUR_TO_SECOND", 112, 23, None, 112, 0, None),
    ("INTERVAL_MINUTE_TO_SECOND", 113, 23, None, 113, 0, None),
    ("GEO_POINT", 114, 58, None, 114, 0, None),
    ("GEO_SHAPE", 114, 2147483647, None, 114, 0, None),
    ("SHAPE", 114, 2147483647, None, 114, 0, None),
    ("UNSUPPORTED", 1111, 0, None, 1111, 0, None),
    ("NESTED", 2002, 0, None, 2002, 0, None),
    ("OBJECT", 2002, 0, None, 2002, 0, None),
];

/// `SYS TYPES [type]`: the JDBC `getTypeInfo` rows, as ES 9.5.5 answers.
fn sys_types(only: Option<i64>) -> Table {
    let names = ["TYPE_NAME", "DATA_TYPE", "PRECISION", "LITERAL_PREFIX", "LITERAL_SUFFIX", "CREATE_PARAMS", "NULLABLE", "CASE_SENSITIVE", "SEARCHABLE", "UNSIGNED_ATTRIBUTE", "FIXED_PREC_SCALE", "AUTO_INCREMENT", "LOCAL_TYPE_NAME", "MINIMUM_SCALE", "MAXIMUM_SCALE", "SQL_DATA_TYPE", "SQL_DATETIME_SUB", "NUM_PREC_RADIX", "INTERVAL_PRECISION"];
    let ty_of = |n: &str| match n {
        "DATA_TYPE" | "PRECISION" | "SQL_DATA_TYPE" | "SQL_DATETIME_SUB" | "NUM_PREC_RADIX" | "INTERVAL_PRECISION" => "integer",
        "NULLABLE" | "SEARCHABLE" | "MINIMUM_SCALE" | "MAXIMUM_SCALE" => "short",
        "CASE_SENSITIVE" | "UNSIGNED_ATTRIBUTE" | "FIXED_PREC_SCALE" | "AUTO_INCREMENT" => "boolean",
        _ => "keyword",
    };
    let cols: Vec<Column> = names.iter().map(|n| typed_col(n, ty_of(n))).collect();
    let signed = ["BYTE", "LONG", "INTEGER", "SHORT", "HALF_FLOAT", "FLOAT", "DOUBLE", "SCALED_FLOAT"];
    let rows = SQL_TYPES
        .iter()
        .filter(|t| only.is_none_or(|o| t.1 == o))
        .map(|&(name, code, precision, scale, sql_type, sub, radix)| {
            vec![
                Val::Str(name.into()),
                Val::Int(code),
                Val::Int(precision),
                Val::Str("'".into()),
                Val::Str("'".into()),
                Val::Null,
                Val::Int(2),
                Val::Bool(matches!(name, "KEYWORD" | "TEXT")),
                Val::Int(3),
                Val::Bool(!signed.contains(&name)),
                Val::Bool(false),
                Val::Bool(false),
                Val::Null,
                scale.map(Val::Int).unwrap_or(Val::Null),
                scale.map(Val::Int).unwrap_or(Val::Null),
                Val::Int(sql_type),
                Val::Int(sub),
                radix.map(Val::Int).unwrap_or(Val::Null),
                Val::Null,
            ]
        })
        .collect();
    Table { cols, rows }
}

fn wildcard_match(pat: &str, s: &str) -> bool {
    let re = format!("^{}$", regex::escape(pat).replace('%', ".*").replace('_', ".").replace("\\*", ".*"));
    regex::Regex::new(&re).map(|r| r.is_match(s)).unwrap_or(false)
}

/// Execute a SQL statement into a table.
/// SQL's row limit: none. ES SQL returns every row of a query (paged through
/// its cursor) up to its LIMIT, so the executor's implicit and maximum
/// limits only stand in for "unbounded"; memory is bounded by the request
/// breaker, not by a row count that would end a result early.
pub const SQL_ROWS: usize = i32::MAX as usize;

pub fn execute(node: &Node, query: &str, params_json: Option<&Value>, filter: Option<&Value>) -> Result<Table> {
    execute_with(node, query, params_json, filter, None, None, None)
}

/// Open a bounded unsorted SELECT/WHERE scan when every wanted shard has a local
/// readable copy. Deterministic row-local projections are evaluated per page;
/// aggregation, sorting and other expressions retain normal planning.
pub fn scan_with(node: &Node, query: &str, params_json: Option<&Value>, filter: Option<&Value>, restrict: Option<exec::Restrict>, authorize: Option<exec::AuthorizeIndices>, shards: Option<crate::esql::remote::SharedShardSource>) -> Result<Option<exec::SqlScan>> {
    let _predicates = exec::sql_predicates();
    // Runtime fields are computed by the search: the materialized path.
    if exec::sql_runtime().is_some() {
        return Ok(None);
    }
    let params = sql_params(params_json);
    let Statement::Select(q) = parse(query, &params)? else { return Ok(None) };
    let Some(index) = q.index else { return Ok(None) };
    fn page_expression(e: &Expr) -> bool {
        match e {
            Expr::Col(..) | Expr::Lit(_) => true,
            Expr::Cast(e, _) | Expr::Neg(e) => page_expression(e),
            Expr::Bin(_, a, b) => page_expression(a) && page_expression(b),
            Expr::Call(name, args, _) if matches!(name.as_str(),
                "LCASE" | "UCASE" | "LOWER" | "UPPER" | "TO_LOWER" | "TO_UPPER" |
                "SUBSTRING" | "SUBSTR" | "LENGTH" | "CHAR_LENGTH" | "CONCAT" |
                "CEIL" | "CEILING" | "FLOOR" | "ABS") => args.iter().all(page_expression),
            _ => false,
        }
    }
    if !q.cmds.iter().all(|c| match c {
        Cmd::From { .. } | Cmd::Where(..) | Cmd::Keep(..) | Cmd::Limit(_) => true,
        Cmd::Eval(fields) => fields.iter().all(|f| f.filter.is_none() && page_expression(&f.expr)),
        _ => false,
    })
        || q.cmds.iter().filter(|c| matches!(c, Cmd::From { .. })).count() != 1 { return Ok(None); }
    // The scan pushes every WHERE to the index against the source columns
    // and applies one (the outer) LIMIT: a WHERE after a projection or a
    // LIMIT (a flattened subquery's outer filter reads the inner query's
    // computed columns), or an inner LIMIT, keeps the stage order of the
    // materialized execution instead.
    let mut staged = false;
    let mut limits = 0usize;
    for c in &q.cmds {
        match c {
            Cmd::Where(..) if staged => return Ok(None),
            Cmd::Eval(_) | Cmd::Keep(..) => staged = true,
            Cmd::Limit(n) => {
                limits += 1;
                staged = true;
                if q.limit != Some(*n) { return Ok(None); }
            }
            _ => {}
        }
    }
    if limits != usize::from(q.limit.is_some()) { return Ok(None); }
    let mut ctx = exec::new_ctx(node, query, filter, SQL_ROWS, SQL_ROWS);
    ctx.restrict = restrict;
    ctx.authorize = authorize;
    ctx.shards = shards;
    let Some(scan) = exec::SqlScan::open(&ctx, &index, q.limit, &q.cmds)? else { return Ok(None) };
    // A row filter the index cannot answer (a function of a field) makes the
    // scan fetch, serialize and re-parse `_source` hits for every candidate.
    // When the candidates (or the LIMIT) bound what the result can retain,
    // the executor's single doc-values pass filters them far faster, with the
    // same rows in the same (`_doc`) order on a single shard.
    // (An explicit projection lets that pass read only the columns used.)
    // A direct scan (one local shard) already reads doc values in batches
    // and stops once a page is full, as ES's scrolled script query does.
    if scan.is_direct() { return Ok(Some(scan)); }
    let projected = matches!(q.cmds.last(), Some(Cmd::Keep(names, _)) if !names.iter().any(|n| n.contains('*')));
    if projected && scan.len() <= DOC_VALUE_FILTER_MAX_ROWS && has_row_filter(node, &index, &q.cmds) && single_shard(node, &index) {
        return Ok(None);
    }
    Ok(Some(scan))
}

/// Most rows a filtered SQL query may retain to take the materializing
/// doc-values pass instead of the streaming scan (bounds the retained rows:
/// only matching candidates are kept, at most `min(candidates, LIMIT)`).
const DOC_VALUE_FILTER_MAX_ROWS: usize = 500_000;

/// Whether a WHERE conjunct applies a function to a field and cannot be
/// pushed into the index (it is evaluated per row).
fn has_row_filter(node: &Node, index: &str, cmds: &[Cmd]) -> bool {
    fn conjuncts<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
        match e {
            Expr::Bin(crate::esql::parse::BinOp::And, a, b) => { conjuncts(a, out); conjuncts(b, out); }
            other => out.push(other),
        }
    }
    fn field_call(e: &Expr) -> bool {
        match e {
            Expr::Call(_, args, _) => args.iter().any(|a| matches!(a, Expr::Col(..)) || field_call(a)),
            Expr::Bin(_, a, b) => field_call(a) || field_call(b),
            Expr::Neg(a) | Expr::Not(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) | Expr::Like(a, ..) => field_call(a),
            Expr::In(a, l, _) => field_call(a) || l.iter().any(field_call),
            Expr::Between(a, b, c, _) => field_call(a) || field_call(b) || field_call(c),
            _ => false,
        }
    }
    let Ok(cols) = exec::index_columns(node, &[index.to_string()]) else { return false };
    let types: HashMap<String, String> = cols.into_iter().map(|c| (c.name, c.ty)).collect();
    cmds.iter().any(|c| match c {
        Cmd::Where(e, _) => {
            let mut cs = Vec::new();
            conjuncts(e, &mut cs);
            cs.into_iter().any(|c| field_call(c) && exec::to_dsl(c, &types, true).is_none())
        }
        _ => false,
    })
}

/// Whether `index` is read from exactly one shard (the scan's `_doc` order
/// interleaves shards; the doc-values pass reads them one after another).
fn single_shard(node: &Node, index: &str) -> bool {
    let mut opts = crate::node::ResolveOpts::from_params(&HashMap::new());
    opts.allow_no_indices = true;
    node.resolve(index, opts).and_then(|targets| crate::search::expand_targets(&targets)).is_ok_and(|targets| targets.len() == 1)
}

/// [`execute`] with the caller's DLS/FLS (`restrict`) and index privileges
/// (`authorize`) and, on a cluster, the shard source that reads every
/// wanted shard from a started copy on any node (as ES|QL does).
pub fn execute_with(node: &Node, query: &str, params_json: Option<&Value>, filter: Option<&Value>, restrict: Option<crate::esql::exec::Restrict>, authorize: Option<exec::AuthorizeIndices>, shards: Option<crate::esql::remote::SharedShardSource>) -> Result<Table> {
    let _predicates = exec::sql_predicates();
    let params = sql_params(params_json);
    let stmt = parse(query, &params)?;
    match stmt {
        Statement::Select(q) => {
            let trimmed = query.trim().trim_end_matches(';');
            let mut ctx = exec::new_ctx(node, trimmed, filter, SQL_ROWS, SQL_ROWS);
            ctx.restrict = restrict;
            ctx.authorize = authorize;
            ctx.shards = shards;
            let mut t = exec::run_cmds(&mut ctx, &q.cmds)?;
            // `SELECT *` hides unsupported fields and multi-fields
            // (`name.keyword` of a text `name`), as ES SQL does.
            if q.index.is_some() && q.cmds.iter().all(|c| !matches!(c, Cmd::Keep(..) | Cmd::Stats { .. })) {
                let names: Vec<String> = t.cols.iter().map(|c| c.name.clone()).collect();
                let keep: Vec<usize> = (0..t.cols.len())
                    .filter(|i| {
                        let n = &t.cols[*i].name;
                        !t.cols[*i].ty.starts_with("unsupported")
                            && !n.rsplit_once('.').map(|(parent, _)| names.contains(&parent.to_string())).unwrap_or(false)
                    })
                    .collect();
                // Move retained cells instead of duplicating the entire result
                // table just to remove multi-fields from SELECT *.
                t.project(&keep);
            }
            for column in &t.cols {
                if column.ty.starts_with("unsupported") {
                    return Err(exec::unsupported_err(&ctx, &column.name, trimmed.find(&column.name).unwrap_or(0), &column.ty));
                }
            }
            let mut floats = match &q.index {
                Some(index) => exec::sql_float_columns(&ctx, index)?,
                None => Default::default(),
            };
            for cmd in &q.cmds {
                match cmd {
                    Cmd::Eval(fields) => for field in fields {
                        let is_float = matches!(&field.expr, Expr::Col(name, _) if floats.contains(name));
                        floats.remove(&field.name);
                        if is_float { floats.insert(field.name.clone()); }
                    },
                    Cmd::Stats { .. } => floats.clear(),
                    _ => {}
                }
            }
            for c in t.cols.iter_mut() {
                c.ty = if c.ty == "double" && floats.contains(&c.name) {
                    "float".to_string()
                } else { sql_type(&c.ty).to_string() };
            }
            Ok(t)
        }
        Statement::ShowTables(pat) => {
            // Only the indices and aliases the caller may see.
            let visible: Option<std::collections::HashSet<String>> = match &authorize {
                Some(a) => a.indices("*", &crate::node::ResolveOpts::from_params(&HashMap::new()), false)?.map(|r| r.names.into_iter().collect()),
                None => None,
            };
            let shown = |name: &str| visible.as_ref().is_none_or(|v| v.contains(name));
            let mut rows = Vec::new();
            for svc in node.all_indices() {
                if svc.name.starts_with('.') {
                    continue;
                }
                if let Some(p) = &pat {
                    if !wildcard_match(p, &svc.name) {
                        continue;
                    }
                }
                if shown(&svc.name) {
                    rows.push(vec![Val::Str(node.cluster_name.clone()), Val::Str(svc.name.clone()), Val::Str("TABLE".into()), Val::Str("INDEX".into())]);
                }
                for a in svc.aliases().keys().filter(|a| shown(a)) {
                    rows.push(vec![Val::Str(node.cluster_name.clone()), Val::Str(a.clone()), Val::Str("VIEW".into()), Val::Str("ALIAS".into())]);
                }
            }
            rows.sort_by(|a, b| a[1].to_text().cmp(&b[1].to_text()));
            rows.dedup();
            Ok(Table {
                cols: ["catalog", "name", "type", "kind"].iter().map(|n| Column { name: n.to_string(), ty: "keyword".into() }).collect(),
                rows,
            })
        }
        Statement::ShowColumns(ix) => {
            // An index the caller may not see describes as no columns (ES:
            // its field caps resolve to no index).
            let hidden = match &authorize {
                Some(a) => a.indices(&ix, &crate::node::ResolveOpts::from_params(&HashMap::new()), false)?.is_some_and(|r| r.names.is_empty() || !r.unknown.is_empty()),
                None => false,
            };
            let cols = if hidden { Vec::new() } else { exec::index_columns_as(node, &[ix], authorize.as_ref())? };
            let rows = cols
                .iter()
                .map(|c| {
                    let jdbc = match c.ty.as_str() {
                        "keyword" | "text" | "ip" | "version" => "VARCHAR",
                        "long" | "unsigned_long" => "BIGINT",
                        "integer" => "INTEGER",
                        "double" => "DOUBLE",
                        "boolean" => "BOOLEAN",
                        "date" | "date_nanos" => "TIMESTAMP",
                        _ => "OTHER",
                    };
                    vec![Val::Str(c.name.clone()), Val::Str(jdbc.into()), Val::Str(sql_type(&c.ty).to_string())]
                })
                .collect();
            Ok(Table { cols: ["column", "type", "mapping"].iter().map(|n| Column { name: n.to_string(), ty: "keyword".into() }).collect(), rows })
        }
        Statement::SysTables { catalog, table, types } => sys_tables(node, authorize.as_ref(), catalog.as_deref(), table.as_deref(), types.as_deref()),
        Statement::SysColumns { table, column } => sys_columns(node, authorize.as_ref(), table.as_deref(), column.as_deref()),
        Statement::SysTypes(t) => Ok(sys_types(t)),
        Statement::ShowFunctions => {
            let mut rows = Vec::new();
            for f in func::AGGS {
                rows.push(vec![Val::Str(f.to_string()), Val::Str("AGGREGATE".into())]);
            }
            for f in func::SCALARS {
                rows.push(vec![Val::Str(f.to_string()), Val::Str("SCALAR".into())]);
            }
            Ok(Table { cols: ["name", "type"].iter().map(|n| Column { name: n.to_string(), ty: "keyword".into() }).collect(), rows })
        }
    }
}

/// `_sql/translate`: the Query DSL search request equivalent.
pub fn translate(node: &Node, query: &str, params_json: Option<&Value>, fetch_size: usize) -> Result<Value> {
    translate_as(node, query, params_json, fetch_size, None)
}

/// [`translate_as`] with the request's additional Query DSL `filter`
/// (ANDed with the translated WHERE, as the SQL query applies it).
pub fn translate_filtered(node: &Node, query: &str, params_json: Option<&Value>, filter: Option<&Value>, fetch_size: usize, authorize: Option<exec::AuthorizeIndices>) -> Result<Value> {
    let mut out = translate_as(node, query, params_json, fetch_size, authorize)?;
    if let Some(f) = filter.filter(|f| !f.is_null()) {
        match out.get_mut("query") {
            Some(q) => {
                if let Some(arr) = q["bool"]["filter"].as_array_mut() {
                    arr.push(f.clone());
                } else {
                    *q = json!({"bool": {"must": [q.clone()], "filter": [f.clone()], "adjust_pure_negative": true, "boost": 1.0}});
                }
            }
            None => {
                out["query"] = json!({"bool": {"filter": [f.clone()], "adjust_pure_negative": true, "boost": 1.0}});
            }
        }
    }
    Ok(out)
}

/// [`translate`] of the indices the caller may see (`authorize`).
pub fn translate_as(node: &Node, query: &str, params_json: Option<&Value>, fetch_size: usize, authorize: Option<exec::AuthorizeIndices>) -> Result<Value> {
    let _predicates = exec::sql_predicates();
    let params = sql_params(params_json);
    let Statement::Select(q) = parse(query, &params)? else {
        return Err(EsError::illegal_argument("only SELECT statements can be translated"));
    };
    if q.cmds.iter().any(|c| matches!(c, Cmd::SqlPivot { .. })) {
        return Err(EsError::illegal_argument("PIVOT translation to Query DSL is not supported"));
    }
    let index = q.index.clone().ok_or_else(|| EsError::illegal_argument("cannot translate a query without FROM"))?;
    let cols = exec::index_columns_as(node, &[index], authorize.as_ref())?;
    let mut types: HashMap<String, String> = cols.iter().map(|c| (c.name.clone(), c.ty.clone())).collect();
    let mut out = serde_json::Map::new();
    // Request runtime fields: part of the translated search (fields,
    // predicates and aggregations name them like mapped fields).
    if let Some(rt) = exec::sql_runtime() {
        for (n, d) in rt.as_object().into_iter().flatten() {
            types.insert(n.clone(), d["type"].as_str().unwrap_or("keyword").to_string());
        }
        out.insert("runtime_mappings".into(), rt);
    }
    // Computed columns of the query (an EVAL before the aggregation), by
    // name: a grouping or metric reading one is translated from its
    // expression.
    let mut evals: HashMap<String, Expr> = HashMap::new();
    let mut filters = Vec::new();
    let mut musts = Vec::new();
    let mut fields: Vec<Value> = Vec::new();
    let mut sort: Vec<Value> = Vec::new();
    let mut aggs: Option<Value> = None;
    for c in &q.cmds {
        match c {
            Cmd::Where(e, _) if aggs.is_none() => match exec::check_sql_condition(e, &types).and(Ok(exec::to_dsl(e, &types, true).map(|(q, s)| (exec::strip_single_value(q), s))))? {
                Some((dsl, true)) => musts.push(dsl),
                Some((dsl, false)) => filters.push(dsl),
                None => return Err(EsError::illegal_argument("WHERE condition cannot be translated to Query DSL")),
            },
            Cmd::Stats { aggs: a, by, .. } => {
                // Every group key and metric is translated, or the query is
                // refused: a translation never drops a grouping dimension or
                // a metric input (the search would answer other data).
                let untranslatable = |what: &str, name: &str| EsError::illegal_argument(format!("{what} [{name}] cannot be translated to Query DSL"));
                let mut sources = Vec::new();
                for b in by {
                    // A computed key is a script source (ES SQL translates
                    // grouping expressions to Painless scripts).
                    match inline_evals(&b.expr, &evals, &types) {
                        // `YEAR(date)` groups by a calendar year histogram
                        // (ES SQL's `GroupByDateHistogram`), in the session zone.
                        Expr::Call(f, args, _) if f == "YEAR" && matches!(args.as_slice(), [Expr::Col(c, _)] if types.get(c).is_some_and(|t| t.starts_with("date"))) => {
                            let Some(Expr::Col(c, _)) = args.first() else { unreachable!() };
                            sources.push(json!({ b.name.clone(): {"date_histogram": {"field": c, "missing_bucket": true, "order": "asc", "calendar_interval": "1y", "time_zone": zone_id(crate::esql::zone())}}}));
                        }
                        Expr::Col(n, _) if types.contains_key(&n) => {
                            let field = if types.get(&n).map(|t| t == "text").unwrap_or(false) { format!("{n}.keyword") } else { n.clone() };
                            sources.push(json!({ b.name.clone(): {"terms": {"field": field, "missing_bucket": true, "order": "asc"}}}));
                        }
                        e => {
                            let script = painless_script(&e, &types).ok_or_else(|| untranslatable("grouping", &b.name))?;
                            sources.push(json!({ b.name.clone(): {"terms": {"script": script, "missing_bucket": true, "order": "asc"}}}));
                        }
                    }
                }
                let mut metrics = serde_json::Map::new();
                for f in a {
                    if let Expr::Call(n, args, _) = &f.expr {
                        // The metric's value source: a field, or a script of
                        // its computed input.
                        let target: Option<Value> = match args.first().map(|a| inline_evals(a, &evals, &types)) {
                            None | Some(Expr::Star) => None,
                            Some(Expr::Lit(_)) if n == "COUNT" => None,
                            Some(Expr::Col(c, _)) if types.contains_key(&c) => Some(json!({"field": c})),
                            Some(e) => Some(json!({"script": painless_script(&e, &types).ok_or_else(|| untranslatable("aggregate", &f.name))?})),
                        };
                        // Numeric SUM / AVG / MIN / MAX are `stats` (ES SQL
                        // folds them into one stats aggregation).
                        let numeric = |t: &Value| t.get("script").is_some() || t["field"].as_str().and_then(|f| types.get(f)).is_some_and(|ty| matches!(ty.as_str(), "long" | "integer" | "short" | "byte" | "double" | "float" | "half_float" | "scaled_float" | "unsigned_long"));
                        let m = match (n.as_str(), target) {
                            ("AVG" | "SUM" | "MIN" | "MAX", Some(t)) if numeric(&t) => json!({"stats": t}),
                            ("COUNT", None) => continue,
                            ("COUNT", Some(t)) if t.get("field").is_some() => json!({"filter": {"exists": t}}),
                            ("COUNT", Some(t)) => json!({"value_count": t}),
                            ("COUNT_DISTINCT", Some(t)) => json!({"cardinality": t}),
                            ("AVG", Some(t)) => json!({"avg": t}),
                            ("SUM", Some(t)) => json!({"sum": t}),
                            ("MIN", Some(t)) => json!({"min": t}),
                            ("MAX", Some(t)) => json!({"max": t}),
                            ("MEDIAN", Some(t)) | ("PERCENTILE", Some(t)) => json!({"percentiles": t}),
                            _ => return Err(untranslatable("aggregate", &f.name)),
                        };
                        metrics.insert(f.name.clone(), m);
                    }
                }
                let agg = if sources.is_empty() {
                    Value::Object(metrics)
                } else {
                    let mut g = json!({"composite": {"size": fetch_size, "sources": sources}});
                    if !metrics.is_empty() {
                        g["aggregations"] = Value::Object(metrics);
                    }
                    json!({"groupby": g})
                };
                aggs = Some(agg);
            }
            Cmd::Eval(f) => {
                for x in f {
                    if let Expr::Col(n, _) = &x.expr {
                        fields.push(field_spec(n, &types));
                    } else if aggs.is_none() {
                        evals.insert(x.name.clone(), x.expr.clone());
                    }
                }
            }
            Cmd::Keep(names, _) if aggs.is_none() => {
                for n in names {
                    if types.contains_key(n) && !fields.iter().any(|f| f["field"] == json!(n)) {
                        fields.push(field_spec(n, &types));
                    }
                }
            }
            Cmd::Sort(keys) if aggs.is_none() => {
                for k in keys {
                    if let Expr::Col(n, _) = &k.expr {
                        let mut s = json!({"order": if k.desc { "desc" } else { "asc" }, "missing": if k.nulls_first == Some(true) { "_first" } else { "_last" }});
                        if let Some(t) = types.get(n) {
                            s["unmapped_type"] = json!(t);
                        }
                        sort.push(json!({ n.clone(): s }));
                    }
                }
            }
            _ => {}
        }
    }
    let has_aggs = aggs.is_some();
    out.insert("size".into(), json!(if has_aggs { 0 } else { q.limit.unwrap_or(fetch_size).min(fetch_size) }));
    if !musts.is_empty() || !filters.is_empty() {
        out.insert("query".into(), json!({"bool": {"must": musts, "filter": filters, "adjust_pure_negative": true, "boost": 1.0}}));
    }
    out.insert("_source".into(), json!(false));
    if !has_aggs {
        if fields.is_empty() {
            for c in &cols {
                if !c.name.rsplit_once('.').map(|(p, _)| types.contains_key(p)).unwrap_or(false) {
                    fields.push(field_spec(&c.name, &types));
                }
            }
        }
        out.insert("fields".into(), Value::Array(fields));
        if !sort.is_empty() {
            out.insert("sort".into(), Value::Array(sort));
        }
    } else {
        out.insert("aggregations".into(), aggs.unwrap());
    }
    out.insert("track_total_hits".into(), json!(-1));
    Ok(Value::Object(out))
}

/// A time zone as ES renders its `ZoneId` (`Z` for UTC).
fn zone_id(tz: crate::date::Tz) -> String {
    match tz {
        crate::date::Tz::Fixed(0) => "Z".into(),
        crate::date::Tz::Fixed(o) => format!("{}{:02}:{:02}", if o < 0 { '-' } else { '+' }, o.abs() / 3600, o.abs() % 3600 / 60),
        crate::date::Tz::Named(z) => z.name().to_string(),
    }
}

/// `e` with the computed columns it reads (`evals`, not index fields)
/// replaced by their expressions.
fn inline_evals(e: &Expr, evals: &HashMap<String, Expr>, types: &HashMap<String, String>) -> Expr {
    match e {
        Expr::Col(n, _) if !types.contains_key(n) => match evals.get(n) {
            Some(def) => inline_evals(def, evals, types),
            None => e.clone(),
        },
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(inline_evals(a, evals, types)), Box::new(inline_evals(b, evals, types))),
        Expr::Neg(a) => Expr::Neg(Box::new(inline_evals(a, evals, types))),
        Expr::Call(n, args, p) => Expr::Call(n.clone(), args.iter().map(|a| inline_evals(a, evals, types)).collect(), *p),
        other => other.clone(),
    }
}

/// The ES SQL script function (`InternalSqlScriptUtils` method) of a SQL
/// scalar, with literal arguments it adds after the SQL ones; `None`: a
/// function ES SQL does not translate into a script.
fn sql_script_fn(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    Some(match name {
        // string
        "ASCII" => ("ascii", &[]),
        "BIT_LENGTH" => ("bitLength", &[]),
        "CHAR" => ("character", &[]),
        "CHAR_LENGTH" | "CHARACTER_LENGTH" => ("charLength", &[]),
        "CONCAT" => ("concat", &[]),
        "INSERT" => ("insert", &[]),
        "LCASE" | "LOWER" | "TO_LOWER" => ("lcase", &[]),
        "LEFT" => ("left", &[]),
        "LENGTH" => ("length", &[]),
        "LOCATE" => ("locate", &[]),
        "LTRIM" => ("ltrim", &[]),
        "OCTET_LENGTH" | "BYTE_LENGTH" => ("octetLength", &[]),
        "POSITION" => ("position", &[]),
        "REPEAT" => ("repeat", &[]),
        "REPLACE" => ("replace", &[]),
        "RIGHT" => ("right", &[]),
        "RTRIM" => ("rtrim", &[]),
        "SPACE" => ("space", &[]),
        "STARTS_WITH" => ("startsWith", &[]),
        "SUBSTRING" | "SUBSTR" => ("substring", &[]),
        "TRIM" => ("trim", &[]),
        "UCASE" | "UPPER" | "TO_UPPER" => ("ucase", &[]),
        // math
        "ABS" => ("abs", &[]),
        "ACOS" => ("acos", &[]),
        "ASIN" => ("asin", &[]),
        "ATAN" => ("atan", &[]),
        "ATAN2" => ("atan2", &[]),
        "CBRT" => ("cbrt", &[]),
        "CEIL" | "CEILING" => ("ceil", &[]),
        "COS" => ("cos", &[]),
        "COSH" => ("cosh", &[]),
        "COT" => ("cot", &[]),
        "DEGREES" | "TO_DEGREES" => ("degrees", &[]),
        "E" => ("e", &[]),
        "EXP" => ("exp", &[]),
        "EXPM1" => ("expm1", &[]),
        "FLOOR" => ("floor", &[]),
        "LOG" => ("log", &[]),
        "LOG10" => ("log10", &[]),
        "MOD" => ("mod", &[]),
        "PI" => ("pi", &[]),
        "POWER" | "POW" => ("power", &[]),
        "RADIANS" | "TO_RADIANS" => ("radians", &[]),
        "RANDOM" | "RAND" => ("random", &[]),
        "ROUND" => ("round", &[]),
        "SIGN" | "SIGNUM" => ("sign", &[]),
        "SIN" => ("sin", &[]),
        "SINH" => ("sinh", &[]),
        "SQRT" => ("sqrt", &[]),
        "TAN" => ("tan", &[]),
        "TRUNCATE" | "TRUNC" => ("truncate", &[]),
        // date / time
        "YEAR" => ("dateTimeExtract", &["Z", "YEAR"]),
        "MONTH" | "MONTH_OF_YEAR" => ("dateTimeExtract", &["Z", "MONTH_OF_YEAR"]),
        "DAY" | "DAY_OF_MONTH" | "DAYOFMONTH" | "DOM" => ("dateTimeExtract", &["Z", "DAY_OF_MONTH"]),
        "DAY_OF_YEAR" | "DAYOFYEAR" | "DOY" => ("dateTimeExtract", &["Z", "DAY_OF_YEAR"]),
        "HOUR" | "HOUR_OF_DAY" => ("dateTimeExtract", &["Z", "HOUR_OF_DAY"]),
        "MINUTE" | "MINUTE_OF_HOUR" => ("dateTimeExtract", &["Z", "MINUTE_OF_HOUR"]),
        "MINUTE_OF_DAY" => ("dateTimeExtract", &["Z", "MINUTE_OF_DAY"]),
        "SECOND" | "SECOND_OF_MINUTE" => ("dateTimeExtract", &["Z", "SECOND_OF_MINUTE"]),
        "DAY_OF_WEEK" | "DAYOFWEEK" | "DOW" => ("dayOfWeek", &["Z"]),
        "ISO_DAY_OF_WEEK" | "ISODAYOFWEEK" | "ISODOW" | "IDOW" => ("dateTimeExtract", &["Z", "ISO_DAY_OF_WEEK"]),
        "WEEK" | "WEEK_OF_YEAR" => ("weekOfYear", &["Z"]),
        "ISO_WEEK_OF_YEAR" | "ISOWEEK" | "IW" => ("dateTimeExtract", &["Z", "ISO_WEEK_OF_YEAR"]),
        "QUARTER" => ("quarter", &["Z"]),
        "DAY_NAME" | "DAYNAME" => ("dayName", &["Z"]),
        "MONTH_NAME" | "MONTHNAME" => ("monthName", &["Z"]),
        "DATE_TRUNC" | "DATETRUNC" => ("dateTrunc", &["Z"]),
        "DATE_ADD" | "DATEADD" | "TIMESTAMPADD" | "TIMESTAMP_ADD" => ("dateAdd", &["Z"]),
        "DATE_DIFF" | "DATEDIFF" | "TIMESTAMPDIFF" | "TIMESTAMP_DIFF" => ("dateDiff", &["Z"]),
        "DATE_PART" | "DATEPART" => ("datePart", &["Z"]),
        "DATE_FORMAT" => ("dateFormat", &["Z"]),
        "DATETIME_FORMAT" => ("dateTimeFormat", &["Z"]),
        "DATE_PARSE" => ("dateParse", &["Z"]),
        "DATETIME_PARSE" => ("dateTimeParse", &["Z"]),
        // conditional
        "COALESCE" | "IFNULL" | "ISNULL" | "NVL" => ("coalesce", &[]),
        "GREATEST" => ("greatest", &[]),
        "LEAST" => ("least", &[]),
        "NULLIF" => ("nullif", &[]),
        "IIF" => ("iif", &[]),
        _ => return None,
    })
}

/// A Painless script (`{"source", "lang", "params"}`) computing `e` per
/// document the way ES SQL's script translation does: every operation is an
/// `InternalSqlScriptUtils` call (null in, null out), fields read through
/// `docValue`, literals passed as `params.vN`. `None`: an expression ES SQL
/// cannot translate (the translation is then refused).
fn painless_script(e: &Expr, types: &HashMap<String, String>) -> Option<Value> {
    use crate::esql::parse::BinOp;
    struct Gen<'t> {
        types: &'t HashMap<String, String>,
        params: serde_json::Map<String, Value>,
    }
    impl Gen<'_> {
        fn param(&mut self, v: Value) -> String {
            let k = format!("v{}", self.params.len());
            self.params.insert(k.clone(), v);
            format!("params.{k}")
        }
        fn call(&mut self, method: &str, args: Vec<String>) -> String {
            format!("InternalSqlScriptUtils.{method}({})", args.join(","))
        }
        fn walk(&mut self, e: &Expr) -> Option<String> {
            Some(match e {
                Expr::Col(n, _) => {
                    let t = self.types.get(n)?;
                    let f = if t == "text" { format!("{n}.keyword") } else { n.clone() };
                    let p = self.param(json!(f));
                    format!("InternalQlScriptUtils.docValue(doc,{p})")
                }
                Expr::Lit(v) => match v {
                    Val::Null => "null".to_string(),
                    other => {
                        let j = other.to_json();
                        self.param(j)
                    }
                },
                Expr::Neg(a) => {
                    let a = self.walk(a)?;
                    self.call("neg", vec![a])
                }
                Expr::Not(a) => {
                    let a = self.walk(a)?;
                    self.call("not", vec![a])
                }
                Expr::Bin(op, a, b) => {
                    let m = match op {
                        BinOp::Add => "add",
                        BinOp::Sub => "sub",
                        BinOp::Mul => "mul",
                        BinOp::Div => "div",
                        BinOp::Mod => "mod",
                        BinOp::Eq => "eq",
                        BinOp::Neq => "neq",
                        BinOp::Lt => "lt",
                        BinOp::Le => "lte",
                        BinOp::Gt => "gt",
                        BinOp::Ge => "gte",
                        BinOp::And => "and",
                        BinOp::Or => "or",
                    };
                    let (a, b) = (self.walk(a)?, self.walk(b)?);
                    self.call(m, vec![a, b])
                }
                Expr::IsNull(a, neg) => {
                    let a = self.walk(a)?;
                    self.call(if *neg { "isNotNull" } else { "isNull" }, vec![a])
                }
                Expr::In(a, list, neg) => {
                    let a = self.walk(a)?;
                    let items = list.iter().map(|x| self.walk(x)).collect::<Option<Vec<_>>>()?;
                    let c = self.call("in", vec![a, format!("[{}]", items.join(","))]);
                    if *neg { self.call("not", vec![c]) } else { c }
                }
                Expr::Between(a, lo, hi, neg) => {
                    let (x, l, h) = (self.walk(a)?, self.walk(lo)?, self.walk(hi)?);
                    let (ge, le) = (self.call("gte", vec![x.clone(), l]), self.call("lte", vec![x, h]));
                    let c = self.call("and", vec![ge, le]);
                    if *neg { self.call("not", vec![c]) } else { c }
                }
                Expr::Cast(a, t) => {
                    let a = self.walk(a)?;
                    let p = self.param(json!(t.to_ascii_uppercase()));
                    self.call("convert", vec![a, p])
                }
                Expr::Call(name, args, _) if name == "CASE" => {
                    let items = args.iter().map(|x| self.walk(x)).collect::<Option<Vec<_>>>()?;
                    self.call("caseFunction", vec![format!("[{}]", items.join(","))])
                }
                Expr::Call(name, args, _) => {
                    let (method, extra) = sql_script_fn(name)?;
                    let mut a = args.iter().map(|x| self.walk(x)).collect::<Option<Vec<_>>>()?;
                    for x in extra {
                        let p = self.param(json!(x));
                        a.push(p);
                    }
                    self.call(method, a)
                }
                _ => return None,
            })
        }
    }
    let mut g = Gen { types, params: serde_json::Map::new() };
    let source = g.walk(e)?;
    Some(json!({"source": source, "lang": "painless", "params": Value::Object(g.params)}))
}

fn field_spec(n: &str, types: &HashMap<String, String>) -> Value {
    match types.get(n).map(String::as_str) {
        Some("date") | Some("date_nanos") => json!({"field": n, "format": "strict_date_optional_time_nanos"}),
        _ => json!({ "field": n }),
    }
}

#[cfg(test)]
mod cursor_stage_tests {
    use super::*;
    use serde_json::json;

    fn node_with_rows(tmp: &std::path::Path) -> std::sync::Arc<Node> {
        let node = Node::open(tmp).unwrap();
        node.create_index("idx", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": {"properties": {"a": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("idx").unwrap();
        let reqs = [1, 2].iter().map(|a| crate::index::WriteReq::Index { id: Some(a.to_string()), source: json!({"a": a}).to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        node
    }

    fn rows(node: &Node, q: &str) -> Vec<String> {
        let mut out: Vec<String> = execute(node, q, None, None).unwrap().rows.iter().map(|r| r.iter().map(|v| v.to_text()).collect::<Vec<_>>().join(",")).collect();
        out.sort();
        out
    }

    fn scanned(node: &Node, q: &str) -> Option<Vec<String>> {
        let mut scan = scan_with(node, q, None, None, None, None, None).unwrap()?;
        let mut out: Vec<String> = scan.page(node, 1000).unwrap().iter().map(|r| r.as_array().map(|a| a.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")).unwrap_or_default()).collect();
        out.sort();
        Some(out)
    }

    /// Review finding 187: the cursor scan never moves a WHERE ahead of the
    /// projection it reads (a flattened subquery's outer filter on a
    /// computed column) and never drops an inner LIMIT: such queries keep
    /// the materialized stage order and give its result. A plain filtered
    /// scan still takes the cursor.
    #[test]
    fn cursor_scans_keep_subquery_stage_order() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node_with_rows(tmp.path());
        let q = "SELECT a FROM (SELECT a + 1 AS a FROM idx) WHERE a = 2";
        assert_eq!(scanned(&node, q), None, "the outer filter reads the computed column");
        assert_eq!(rows(&node, q), vec!["2"], "the row whose computed a is 2 (source a = 1)");
        let q = "SELECT a FROM (SELECT a FROM idx LIMIT 1)";
        assert_eq!(scanned(&node, q), None, "an inner LIMIT is not the scan's");
        assert_eq!(rows(&node, q).len(), 1);
        let q = "SELECT a FROM (SELECT a FROM idx LIMIT 1) LIMIT 5";
        assert_eq!(scanned(&node, q), None);
        assert_eq!(rows(&node, q).len(), 1);
        // Control: filter, then projection, then the outer LIMIT.
        let q = "SELECT a FROM idx WHERE a = 2 LIMIT 5";
        assert_eq!(scanned(&node, q), Some(vec!["2".to_string()]));
        assert_eq!(rows(&node, q), vec!["2"]);
    }
}

#[cfg(test)]
mod sql_semantics_tests {
    use super::*;
    use serde_json::json;

    fn node(tmp: &std::path::Path) -> std::sync::Arc<Node> {
        let node = Node::open(tmp).unwrap();
        node.create_index("ev", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": {"properties": {"d": {"type": "date"}, "g": {"type": "keyword"}, "v": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("ev").unwrap();
        let docs = [json!({"d": "2024-03-05T12:59:00Z", "g": "a", "v": 1}), json!({"d": "2025-01-01T00:00:00Z", "g": "b", "v": 2})];
        let reqs = docs.iter().map(|d| crate::index::WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        node
    }

    /// Review finding 364: SQL's date functions take ES SQL's signatures:
    /// `DATE_FORMAT(value, mysql_pattern)`, `DATE_PARSE(value, pattern)`,
    /// and `DATE_DIFF` counting unit boundaries (12:59 to 13:00 is 1 hour).
    #[test]
    fn sql_date_functions_use_sql_signatures() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        let t = execute(&node, "SELECT DATE_FORMAT(d, '%Y-%m-%d %H:%i') AS f FROM ev ORDER BY d", None, None).unwrap();
        assert_eq!(t.rows.iter().map(|r| r[0].to_text()).collect::<Vec<_>>(), vec!["2024-03-05 12:59", "2025-01-01 00:00"]);
        let t = execute(&node, "SELECT DATE_PARSE('05/03/2024', 'dd/MM/uuuu') AS p", None, None).unwrap();
        assert_eq!(t.rows[0][0], Val::Date(1_709_596_800_000));
        let h = |a: &str, b: &str| crate::esql::func::call(crate::esql::func::SQL_DATE_DIFF, &[Val::Str("hour".into()), Val::Date(crate::esql::func::parse_date_str(a).unwrap()), Val::Date(crate::esql::func::parse_date_str(b).unwrap())], &Default::default()).unwrap();
        assert_eq!(h("2024-03-05T12:59:00Z", "2024-03-05T13:00:00Z"), Val::Int(1));
        assert_eq!(h("2024-03-05T13:00:00Z", "2024-03-05T13:59:59Z"), Val::Int(0));
        assert_eq!(crate::esql::func::mysql_to_java_pattern("%d.%m.%Y at %H:%i"), "dd'.'MM'.'yyyy' at 'HH':'mm");
    }

    /// Review finding 368: a typed SQL parameter is a literal of its type
    /// (a JDBC timestamp is a datetime: `YEAR(?)` is its year).
    #[test]
    fn typed_parameters_keep_their_types() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        let params = json!([{"type": "datetime", "value": "2024-07-01T10:00:00Z"}]);
        let t = execute(&node, "SELECT YEAR(?) AS y", Some(&params), None).unwrap();
        assert_eq!(t.rows[0][0], Val::Int(2024));
        let t = execute(&node, "SELECT g FROM ev WHERE YEAR(d) = YEAR(?)", Some(&params), None).unwrap();
        assert_eq!(t.rows.len(), 1);
        let params = json!([{"type": "integer", "value": 2}]);
        let t = execute(&node, "SELECT g FROM ev WHERE v = ?", Some(&params), None).unwrap();
        assert_eq!(t.rows[0][0].to_text(), "b");
    }

    /// Review finding 370: translation never drops a computed grouping key
    /// or aggregate input: such queries are refused; plain fields translate.
    #[test]
    fn translation_refuses_what_it_cannot_express() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        // Expressions it has no script form for are refused, never dropped.
        assert!(translate(&node, "SELECT TO_BASE64(g) AS k, SUM(v) AS s FROM ev GROUP BY TO_BASE64(g)", None, 1000).is_err());
        assert!(translate(&node, "SELECT g, SUM(LENGTH(TO_BASE64(g))) AS s FROM ev GROUP BY g", None, 1000).is_err());
        let ok = translate(&node, "SELECT g, SUM(v) AS s, COUNT(*) AS c FROM ev GROUP BY g", None, 1000).unwrap();
        assert!(ok["aggregations"]["groupby"]["composite"]["sources"].is_array(), "{ok}");
    }
}

#[cfg(test)]
mod runtime_and_translate_tests {
    use super::*;
    use serde_json::json;

    fn node(tmp: &std::path::Path) -> std::sync::Arc<Node> {
        let node = Node::open(tmp).unwrap();
        node.create_index("rt", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": {"properties": {"d": {"type": "date"}, "g": {"type": "keyword"}, "v": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("rt").unwrap();
        let docs = [
            json!({"d": "2024-03-05T12:00:00Z", "g": "a", "v": 1}),
            json!({"d": "2024-07-01T00:00:00Z", "g": "a", "v": 2}),
            json!({"d": "2025-01-01T00:00:00Z", "g": "b", "v": 4}),
        ];
        let reqs = docs.iter().enumerate().map(|(i, d)| crate::index::WriteReq::Index { id: Some(i.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        node
    }

    fn search(node: &Node, body: &Value) -> Value {
        let req = crate::search::parse_request(body, &HashMap::new()).unwrap();
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(node.get_index("rt").unwrap())], &req, &opts).unwrap();
        serde_json::from_str(&out.json).unwrap()
    }

    /// The metric values of an aggregation bucket (or response level), in
    /// their translated (internal) names' order.
    /// (`funcs`: the SQL functions of the metrics in order; a `stats`
    /// metric answers SUM / AVG / MIN / MAX from its own field.)
    fn metrics(b: &Value, funcs: &[&str]) -> Vec<f64> {
        let mut m: Vec<(&String, &Value)> = b.as_object().unwrap().iter().filter(|(k, v)| *k != "key" && *k != "doc_count" && v.is_object()).collect();
        m.sort_by(|a, b| a.0.cmp(b.0));
        m.into_iter().zip(funcs).map(|((_, v), f)| {
            let key = match *f { "SUM" => "sum", "AVG" => "avg", "MIN" => "min", "MAX" => "max", _ => "value" };
            v.get(key).or_else(|| v.get("value")).and_then(Value::as_f64).unwrap_or(f64::NAN)
        }).collect()
    }

    fn rows(t: &Table) -> Vec<Vec<String>> {
        let mut r: Vec<Vec<String>> = t.rows.iter().map(|r| r.iter().map(|v| v.to_text()).collect()).collect();
        r.sort();
        r
    }

    /// Review finding 369: request `runtime_mappings` apply to SQL
    /// projections, predicates and aggregates (a same-name runtime field
    /// overrides the mapped one), matching the same runtime fields in a
    /// DSL search; translation carries them.
    #[test]
    fn sql_runtime_mappings_match_the_dsl() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        // `m10`: a new runtime field; `g`: a runtime override of a mapped field.
        let rt = json!({"m10": {"type": "long", "script": {"source": "emit(doc['d'].value.getMonthValue() * 10)"}}, "g": {"type": "keyword", "script": {"source": "emit('z' + doc['v'].value)"}}});
        let t = crate::esql::exec::with_sql_runtime(Some(&rt), || execute(&node, "SELECT g, m10, v FROM rt WHERE m10 > 10 ORDER BY m10", None, None)).unwrap();
        assert_eq!(rows(&t), vec![vec!["z1", "30", "1"], vec!["z2", "70", "2"]]);
        // The same runtime fields in a DSL search.
        let dsl = search(&node, &json!({"runtime_mappings": rt, "query": {"range": {"m10": {"gt": 10}}}, "fields": ["m10", "g"], "_source": false, "sort": [{"m10": "asc"}]}));
        let from_dsl: Vec<Vec<String>> = dsl["hits"]["hits"].as_array().unwrap().iter().map(|h| vec![h["fields"]["g"][0].as_str().unwrap_or_default().to_string(), h["fields"]["m10"][0].to_string()]).collect();
        assert_eq!(from_dsl, vec![vec!["z1", "30"], vec!["z2", "70"]]);
        let t = crate::esql::exec::with_sql_runtime(Some(&rt), || execute(&node, "SELECT g, SUM(m10) AS s FROM rt GROUP BY g", None, None)).unwrap();
        assert_eq!(rows(&t), vec![vec!["z1", "30"], vec!["z2", "70"], vec!["z4", "10"]]);
        // Without them: the mapped values.
        let t = execute(&node, "SELECT g, COUNT(*) AS c FROM rt GROUP BY g", None, None).unwrap();
        assert_eq!(rows(&t), vec![vec!["a", "2"], vec!["b", "1"]]);
        // Translation carries the runtime mappings and its search answers alike.
        let tr = crate::esql::exec::with_sql_runtime(Some(&rt), || translate(&node, "SELECT g, SUM(m10) AS s FROM rt GROUP BY g", None, 1000)).unwrap();
        assert_eq!(tr["runtime_mappings"], rt);
        let res = search(&node, &tr);
        let mut got: Vec<(String, f64)> = res["aggregations"]["groupby"]["buckets"].as_array().unwrap().iter().map(|b| (b["key"]["g"].as_str().unwrap_or_default().to_string(), metrics(b, &["SUM"])[0])).collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, vec![("z1".into(), 30.0), ("z2".into(), 70.0), ("z4".into(), 10.0)], "{res}");
    }

    /// Review finding 370: a computed grouping key and a computed metric
    /// input translate to Painless script sources; the translated search
    /// answers what the SQL query does.
    #[test]
    fn computed_keys_and_inputs_translate_to_scripts() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        let q = "SELECT YEAR(d) AS y, SUM(v * 2) AS s, MAX(v + 1) AS m FROM rt GROUP BY YEAR(d)";
        let sql = execute(&node, q, None, None).unwrap();
        let num = |v: &Val| v.to_text().parse::<f64>().unwrap_or(f64::NAN);
        let mut want: Vec<(i64, f64, f64)> = sql.rows.iter().map(|r| (num(&r[0]) as i64, num(&r[1]), num(&r[2]))).collect();
        want.sort_by_key(|w| w.0);
        assert_eq!(want, vec![(2024, 6.0, 3.0), (2025, 8.0, 5.0)], "{:?}", sql.rows);
        let tr = translate(&node, q, None, 1000).unwrap();
        // ES 9.5.5: YEAR groups by a 1y date_histogram; SUM / MAX are stats
        // over InternalSqlScriptUtils scripts reading InternalQlScriptUtils.docValue.
        let src = &tr["aggregations"]["groupby"]["composite"]["sources"][0]["y"]["date_histogram"];
        assert_eq!(src, &json!({"field": "d", "missing_bucket": true, "order": "asc", "calendar_interval": "1y", "time_zone": "Z"}), "{tr}");
        let m = tr["aggregations"]["groupby"]["aggregations"].as_object().unwrap();
        assert!(m.values().all(|a| a["stats"]["script"]["lang"] == "painless"), "{tr}");
        assert!(m.values().any(|a| a["stats"]["script"]["source"] == "InternalSqlScriptUtils.mul(InternalQlScriptUtils.docValue(doc,params.v0),params.v1)"), "{tr}");
        let res = search(&node, &tr);
        let mut got: Vec<(i64, f64, f64)> = res["aggregations"]["groupby"]["buckets"].as_array().unwrap().iter()
            .map(|b| (b["key"]["y"].as_i64().and_then(|ms| chrono::DateTime::from_timestamp_millis(ms)).map(|d| chrono::Datelike::year(&d) as i64).unwrap_or(-1), metrics(b, &["SUM", "MAX"])[0], metrics(b, &["SUM", "MAX"])[1])).collect();
        got.sort_by_key(|g| g.0);
        assert_eq!(got, want, "{res}");
        // A global computed metric.
        let tr = translate(&node, "SELECT SUM(v * 2) AS s FROM rt", None, 1000).unwrap();
        assert_eq!(metrics(&search(&node, &tr)["aggregations"], &["SUM"]), vec![14.0]);
    }

    fn fx_node(tmp: &std::path::Path) -> std::sync::Arc<Node> {
        let node = Node::open(tmp).unwrap();
        node.create_index("fx", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": {"properties": {"s": {"type": "keyword"}, "n": {"type": "long"}, "x": {"type": "double"}, "d": {"type": "date"}}}})).unwrap();
        let svc = node.get_index("fx").unwrap();
        let docs = [
            json!({"s": "abc", "n": 1, "x": 1.25, "d": "2024-03-05T12:34:56Z"}),
            json!({"s": "aXa", "n": 4, "x": -2.5, "d": "2024-12-29T00:00:00Z"}),
            json!({"s": "hello", "n": 9, "x": 3.75, "d": "2025-07-04T08:00:00Z"}),
            json!({}),
        ];
        let reqs = docs.iter().enumerate().map(|(i, d)| crate::index::WriteReq::Index { id: Some(i.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        node
    }

    fn fx_search(node: &Node, body: &Value) -> Value {
        let req = crate::search::parse_request(body, &HashMap::new()).unwrap();
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(node.get_index("fx").unwrap())], &req, &opts).unwrap();
        serde_json::from_str(&out.json).unwrap()
    }

    fn num_key(f: f64) -> String {
        if f == f.trunc() && f.abs() < 1e15 { format!("{}", f as i64) } else { format!("{:.9}", f) }
    }

    fn sql_key(v: &Val) -> String {
        match v {
            Val::Null => "null".into(),
            Val::Int(i) | Val::Long(i) => i.to_string(),
            Val::Double(d) => num_key(*d),
            other => other.to_text(),
        }
    }

    fn json_key(v: &Value) -> String {
        match v {
            Value::Null => "null".into(),
            Value::Number(n) => num_key(n.as_f64().unwrap()),
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// Review finding 370 (full): every family of ES SQL scalar ES
    /// translates into an `InternalSqlScriptUtils` script (string, math,
    /// date/time, conditional, conversion) groups the translated search
    /// exactly as the SQL query groups, null and missing values included;
    /// a function ES cannot script is refused.
    #[test]
    fn translated_scripts_answer_like_sql_across_function_families() {
        // ~70 distinct scripts compiled within seconds, beside every other
        // test of the process: an independent compilation budget (the
        // process-wide 150/5m one is shared, so the last expressions were
        // refused with 429 depending on what ran concurrently).
        let _compilation_budget = crate::script::rate_limit::fixture_budget("10000/1m");
        let tmp = tempfile::tempdir().unwrap();
        let node = fx_node(tmp.path());
        let exprs = [
            // string
            "CONCAT(s, '-')", "LENGTH(s)", "SUBSTRING(s, 2, 2)", "LEFT(s, 2)", "RIGHT(s, 2)", "REPLACE(s, 'a', 'Z')",
            "UCASE(s)", "LCASE(s)", "LOCATE('a', s)", "REPEAT(s, 2)", "ASCII(s)", "INSERT(s, 1, 1, 'Q')",
            "LTRIM(CONCAT('  ', s))", "CHAR(n + 64)", "OCTET_LENGTH(s)",
            // math
            "ROUND(x, 1)", "TRUNCATE(x, 0)", "POWER(n, 2)", "SQRT(n)", "ABS(x)", "SIGN(x)", "LOG10(n)", "EXP(n)",
            "FLOOR(x)", "CEILING(x)", "n % 2", "n * 2 + 1", "PI() * n", "DEGREES(x)", "COT(n)", "RANDOM(IFNULL(n, 0))", "MOD(n, 3)", "-x",
            // date / time
            "YEAR(d)", "MONTH(d)", "DAY(d)", "HOUR(d)", "MINUTE(d)", "DAY_OF_WEEK(d)", "ISO_DAY_OF_WEEK(d)", "DAY_OF_YEAR(d)",
            "WEEK(d)", "ISO_WEEK_OF_YEAR(d)", "QUARTER(d)", "DAY_NAME(d)", "MONTH_NAME(d)", "YEAR(DATE_TRUNC('decade', d))",
            "MONTH(DATE_TRUNC('quarter', d))", "DATE_DIFF('day', CAST('2024-01-01T00:00:00Z' AS DATETIME), d)",
            "DATE_PART('dayofyear', d)", "MONTH(DATE_ADD('month', 2, d))", "DATE_FORMAT(d, '%Y/%m')",
            // conditional
            "CASE WHEN n > 2 THEN 'big' ELSE 'small' END", "COALESCE(s, 'none')", "NULLIF(n, 4)", "GREATEST(n, 3)",
            "LEAST(x, 0)", "IIF(n > 3, 'y', 'n')", "IFNULL(n, -1)", "n IS NULL", "n IN (1, 9)", "NOT (n > 2)",
            // conversion
            "CAST(x AS INTEGER)", "CAST(n AS DOUBLE)", "CAST(n AS VARCHAR)", "CONVERT(x, LONG)",
        ];
        let mut failures = Vec::new();
        for e in exprs {
            let q = format!("SELECT {e} AS k, COUNT(*) AS c FROM fx GROUP BY {e}");
            let t = match execute(&node, &q, None, None) {
                Ok(t) => t,
                Err(err) => { failures.push(format!("sql {q}: {err}")); continue; }
            };
            let mut want: Vec<(String, i64)> = t.rows.iter().map(|r| (sql_key(&r[0]), r[1].as_i64().unwrap_or(-1))).collect();
            want.sort();
            let tr = match translate(&node, &q, None, 1000) {
                Ok(tr) => tr,
                Err(err) => { failures.push(format!("translate {q}: {err}")); continue; }
            };
            let histogram = tr["aggregations"]["groupby"]["composite"]["sources"][0]["k"].get("date_histogram").is_some();
            let src = &tr["aggregations"]["groupby"]["composite"]["sources"][0]["k"]["terms"]["script"]["source"];
            if !histogram && !src.as_str().is_some_and(|s| s.contains("InternalSqlScriptUtils.")) {
                failures.push(format!("no script {q}: {tr}"));
                continue;
            }
            let req = match crate::search::parse_request(&tr, &HashMap::new()) {
                Ok(r) => r,
                Err(err) => { failures.push(format!("parse {q}: {err}")); continue; }
            };
            let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
            let res: Value = match crate::search::execute(&[crate::search::Target::new(node.get_index("fx").unwrap())], &req, &opts) {
                Ok(out) => serde_json::from_str(&out.json).unwrap(),
                Err(err) => { failures.push(format!("search {q}: {err}\n{}", src)); continue; }
            };
            let mut got: Vec<(String, i64)> = res["aggregations"]["groupby"]["buckets"].as_array().map(|b| b.iter()
                .map(|b| {
                    let k = &b["key"]["k"];
                    // A YEAR histogram key is the year's first instant.
                    let k = match k.as_i64().filter(|_| histogram).and_then(chrono::DateTime::from_timestamp_millis) {
                        Some(d) => (chrono::Datelike::year(&d) as i64).to_string(),
                        None => json_key(k),
                    };
                    (k, b["doc_count"].as_i64().unwrap())
                }).collect()).unwrap_or_default();
            got.sort();
            if got != want {
                failures.push(format!("{q}: sql {want:?} != search {got:?}\n{}", src));
            }
        }
        assert!(failures.is_empty(), "{} of {} failed:\n{}", failures.len(), exprs.len(), failures.join("\n"));
        // Computed metric inputs across families.
        for (e, want) in [("LENGTH(s)", 11.0), ("ABS(x)", 7.5), ("DAY_OF_WEEK(d)", 3.0 + 1.0 + 6.0), ("IFNULL(n, 100)", 114.0)] {
            let q = format!("SELECT SUM({e}) AS m FROM fx");
            let sql = execute(&node, &q, None, None).unwrap();
            assert_eq!(sql.rows[0][0].as_f64(), Some(want), "{q}");
            let tr = translate(&node, &q, None, 1000).unwrap();
            assert_eq!(metrics(&fx_search(&node, &tr)["aggregations"], &["SUM"]), vec![want], "{q}: {tr}");
        }
        // A function ES SQL does not script is refused, as ES refuses it.
        let err = translate(&node, "SELECT TO_BASE64(s) AS k, COUNT(*) FROM fx GROUP BY TO_BASE64(s)", None, 1000).unwrap_err();
        assert!(err.reason.contains("cannot be translated"), "{err}");
    }
}

#[cfg(test)]
mod jdbc_and_alias_tests {
    use super::*;
    use serde_json::json;

    fn node(tmp: &std::path::Path) -> std::sync::Arc<Node> {
        let node = Node::open(tmp).unwrap();
        node.create_index("products", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": {"properties": {"sku": {"type": "keyword"}, "price": {"type": "long"}, "d": {"type": "date"}, "p": {"properties": {"sku": {"type": "keyword"}}}}}})).unwrap();
        let svc = node.get_index("products").unwrap();
        let docs = [json!({"sku": "A", "price": 5, "d": "2024-01-01T00:00:00Z", "p": {"sku": "inner-a"}}), json!({"sku": "B", "price": 7, "d": "2025-01-01T00:00:00Z", "p": {"sku": "inner-b"}})];
        let reqs = docs.iter().map(|d| crate::index::WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        node
    }

    fn texts(t: &Table) -> Vec<String> {
        t.rows.iter().map(|r| r.iter().map(|v| v.to_text()).collect::<Vec<_>>().join("|")).collect()
    }

    /// Review finding 424: table aliases qualify columns (projection,
    /// predicate, order); the table name qualifies too; the result column
    /// is the bare name.
    #[test]
    fn table_aliases_qualify_columns() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        let t = execute(&node, "SELECT pr.sku, pr.price FROM products AS pr WHERE pr.sku = 'A' ORDER BY pr.price", None, None).unwrap();
        assert_eq!(texts(&t), vec!["A|5"]);
        assert_eq!(t.cols[0].name, "sku");
        let t = execute(&node, "SELECT pr.sku FROM products pr ORDER BY pr.sku DESC", None, None).unwrap();
        assert_eq!(texts(&t), vec!["B", "A"]);
        let t = execute(&node, "SELECT products.price FROM products WHERE products.sku = 'B'", None, None).unwrap();
        assert_eq!(texts(&t), vec!["7"]);
        // A genuine object path still reads the object's field.
        let t = execute(&node, "SELECT p.sku FROM products ORDER BY p.sku", None, None).unwrap();
        assert_eq!(texts(&t), vec!["inner-a", "inner-b"]);
    }

    /// Review finding 399: the JDBC driver's system statements answer with
    /// their JDBC schemas: catalogs, table types, tables, columns, types.
    #[test]
    fn sys_statements_answer_jdbc_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        let t = execute(&node, "SYS TABLES CATALOG LIKE '%' LIKE ''", None, None).unwrap();
        assert_eq!(t.cols[0].name, "TABLE_CAT");
        assert_eq!(t.rows.len(), 1);
        let t = execute(&node, "SYS TABLES CATALOG LIKE '' LIKE '' TYPE '%'", None, None).unwrap();
        assert_eq!(t.rows.iter().map(|r| r[3].to_text()).collect::<Vec<_>>(), vec!["TABLE", "VIEW"]);
        let t = execute(&node, "SYS TABLES LIKE 'prod%' TYPE 'TABLE', 'VIEW'", None, None).unwrap();
        assert_eq!(t.rows.iter().map(|r| r[2].to_text()).collect::<Vec<_>>(), vec!["products"]);
        let t = execute(&node, "SYS COLUMNS TABLE LIKE 'products' LIKE 'pri%'", None, None).unwrap();
        assert_eq!(t.cols.len(), 24);
        assert_eq!(t.rows.len(), 1, "{:?}", t.rows);
        assert_eq!((t.rows[0][3].to_text(), t.rows[0][4].clone(), t.rows[0][5].to_text()), ("price".into(), Val::Int(-5), "LONG".into()));
        let t = execute(&node, "SYS TYPES", None, None).unwrap();
        assert!(t.rows.iter().any(|r| r[0].to_text() == "KEYWORD" && r[1] == Val::Int(12)), "{:?}", t.rows);
        let t = execute(&node, "SYS TYPES 4", None, None).unwrap();
        assert_eq!(t.rows.iter().map(|r| r[0].to_text()).collect::<Vec<_>>(), vec!["INTEGER"]);
    }

    /// Review finding 400: translation ANDs the request's additional filter
    /// with the translated WHERE (with and without one).
    #[test]
    fn translation_keeps_the_request_filter() {
        let tmp = tempfile::tempdir().unwrap();
        let node = node(tmp.path());
        let f = json!({"range": {"d": {"gte": "2025-01-01"}}});
        let tr = translate_filtered(&node, "SELECT sku FROM products WHERE price > 1", None, Some(&f), 1000, None).unwrap();
        assert!(tr["query"]["bool"]["filter"].as_array().unwrap().contains(&f), "{tr}");
        let tr = translate_filtered(&node, "SELECT SUM(price) AS s FROM products", None, Some(&f), 1000, None).unwrap();
        assert!(tr["query"]["bool"]["filter"].as_array().unwrap().contains(&f), "{tr}");
        // The filtered translation selects what the filtered SQL does.
        let sql = execute(&node, "SELECT sku FROM products", None, Some(&f)).unwrap();
        assert_eq!(texts(&sql), vec!["B"]);
        let tr = translate_filtered(&node, "SELECT sku FROM products", None, Some(&f), 1000, None).unwrap();
        let req = crate::search::parse_request(&tr, &HashMap::new()).unwrap();
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(node.get_index("products").unwrap())], &req, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        assert_eq!(v["hits"]["hits"].as_array().unwrap().len(), 1);
    }
}

/// `SYS` statements against ES 9.5.5's own answers (captured from a real
/// cluster; the catalog is that cluster's name).
#[cfg(test)]
mod es955_sys_tests {
    use super::*;
    use serde_json::json;

    const ES_COLUMNS: &str = r#"{"columns": [{"name": "TABLE_CAT", "type": "keyword"}, {"name": "TABLE_SCHEM", "type": "keyword"}, {"name": "TABLE_NAME", "type": "keyword"}, {"name": "COLUMN_NAME", "type": "keyword"}, {"name": "DATA_TYPE", "type": "integer"}, {"name": "TYPE_NAME", "type": "keyword"}, {"name": "COLUMN_SIZE", "type": "integer"}, {"name": "BUFFER_LENGTH", "type": "integer"}, {"name": "DECIMAL_DIGITS", "type": "integer"}, {"name": "NUM_PREC_RADIX", "type": "integer"}, {"name": "NULLABLE", "type": "integer"}, {"name": "REMARKS", "type": "keyword"}, {"name": "COLUMN_DEF", "type": "keyword"}, {"name": "SQL_DATA_TYPE", "type": "integer"}, {"name": "SQL_DATETIME_SUB", "type": "integer"}, {"name": "CHAR_OCTET_LENGTH", "type": "integer"}, {"name": "ORDINAL_POSITION", "type": "integer"}, {"name": "IS_NULLABLE", "type": "keyword"}, {"name": "SCOPE_CATALOG", "type": "keyword"}, {"name": "SCOPE_SCHEMA", "type": "keyword"}, {"name": "SCOPE_TABLE", "type": "keyword"}, {"name": "SOURCE_DATA_TYPE", "type": "short"}, {"name": "IS_AUTOINCREMENT", "type": "keyword"}, {"name": "IS_GENERATEDCOLUMN", "type": "keyword"}], "rows": [["rally-full", null, "sqlnull", "a", 4, "INTEGER", 11, 4, null, 10, 1, null, null, 4, 0, null, 1, "YES", null, null, null, null, "NO", "NO"], ["rally-full", null, "sqlnull", "b", 16, "BOOLEAN", 1, 1, null, null, 1, null, null, 16, 0, null, 2, "YES", null, null, null, null, "NO", "NO"], ["rally-full", null, "sqlnull", "k", 12, "KEYWORD", 32766, 2147483647, null, null, 1, null, null, 12, 0, 2147483647, 3, "YES", null, null, null, null, "NO", "NO"]]}"#;
    const ES_TYPES: &str = r#"{"columns": [{"name": "TYPE_NAME", "type": "keyword"}, {"name": "DATA_TYPE", "type": "integer"}, {"name": "PRECISION", "type": "integer"}, {"name": "LITERAL_PREFIX", "type": "keyword"}, {"name": "LITERAL_SUFFIX", "type": "keyword"}, {"name": "CREATE_PARAMS", "type": "keyword"}, {"name": "NULLABLE", "type": "short"}, {"name": "CASE_SENSITIVE", "type": "boolean"}, {"name": "SEARCHABLE", "type": "short"}, {"name": "UNSIGNED_ATTRIBUTE", "type": "boolean"}, {"name": "FIXED_PREC_SCALE", "type": "boolean"}, {"name": "AUTO_INCREMENT", "type": "boolean"}, {"name": "LOCAL_TYPE_NAME", "type": "keyword"}, {"name": "MINIMUM_SCALE", "type": "short"}, {"name": "MAXIMUM_SCALE", "type": "short"}, {"name": "SQL_DATA_TYPE", "type": "integer"}, {"name": "SQL_DATETIME_SUB", "type": "integer"}, {"name": "NUM_PREC_RADIX", "type": "integer"}, {"name": "INTERVAL_PRECISION", "type": "integer"}], "rows": [["BYTE", -6, 3, "'", "'", null, 2, false, 3, false, false, false, null, 0, 0, -6, 0, 10, null], ["LONG", -5, 19, "'", "'", null, 2, false, 3, false, false, false, null, 0, 0, -5, 0, 10, null], ["BINARY", -2, 2147483647, "'", "'", null, 2, false, 3, true, false, false, null, null, null, -2, 0, null, null], ["NULL", 0, 0, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 0, 0, null, null], ["UNSIGNED_LONG", 2, 20, "'", "'", null, 2, false, 3, true, false, false, null, 0, 0, 2, 0, 10, null], ["INTEGER", 4, 10, "'", "'", null, 2, false, 3, false, false, false, null, 0, 0, 4, 0, 10, null], ["SHORT", 5, 5, "'", "'", null, 2, false, 3, false, false, false, null, 0, 0, 5, 0, 10, null], ["HALF_FLOAT", 6, 3, "'", "'", null, 2, false, 3, false, false, false, null, 3, 3, 6, 0, 2, null], ["FLOAT", 7, 7, "'", "'", null, 2, false, 3, false, false, false, null, 7, 7, 7, 0, 2, null], ["DOUBLE", 8, 15, "'", "'", null, 2, false, 3, false, false, false, null, 15, 15, 8, 0, 2, null], ["SCALED_FLOAT", 8, 15, "'", "'", null, 2, false, 3, false, false, false, null, 15, 15, 8, 0, 2, null], ["IP", 12, 45, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 12, 0, null, null], ["KEYWORD", 12, 32766, "'", "'", null, 2, true, 3, true, false, false, null, null, null, 12, 0, null, null], ["TEXT", 12, 2147483647, "'", "'", null, 2, true, 3, true, false, false, null, null, null, 12, 0, null, null], ["VERSION", 12, 2147483647, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 12, 0, null, null], ["BOOLEAN", 16, 1, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 16, 0, null, null], ["DATE", 91, 29, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 91, 1, null, null], ["TIME", 92, 24, "'", "'", null, 2, false, 3, true, false, false, null, 9, 9, 92, 2, null, null], ["DATETIME", 93, 34, "'", "'", null, 2, false, 3, true, false, false, null, 9, 9, 9, 3, null, null], ["INTERVAL_YEAR", 101, 7, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 101, 0, null, null], ["INTERVAL_MONTH", 102, 7, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 102, 0, null, null], ["INTERVAL_DAY", 103, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 103, 0, null, null], ["INTERVAL_HOUR", 104, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 104, 0, null, null], ["INTERVAL_MINUTE", 105, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 105, 0, null, null], ["INTERVAL_SECOND", 106, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 106, 0, null, null], ["INTERVAL_YEAR_TO_MONTH", 107, 7, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 107, 0, null, null], ["INTERVAL_DAY_TO_HOUR", 108, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 108, 0, null, null], ["INTERVAL_DAY_TO_MINUTE", 109, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 109, 0, null, null], ["INTERVAL_DAY_TO_SECOND", 110, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 110, 0, null, null], ["INTERVAL_HOUR_TO_MINUTE", 111, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 111, 0, null, null], ["INTERVAL_HOUR_TO_SECOND", 112, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 112, 0, null, null], ["INTERVAL_MINUTE_TO_SECOND", 113, 23, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 113, 0, null, null], ["GEO_POINT", 114, 58, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 114, 0, null, null], ["GEO_SHAPE", 114, 2147483647, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 114, 0, null, null], ["SHAPE", 114, 2147483647, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 114, 0, null, null], ["UNSUPPORTED", 1111, 0, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 1111, 0, null, null], ["NESTED", 2002, 0, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 2002, 0, null, null], ["OBJECT", 2002, 0, "'", "'", null, 2, false, 3, true, false, false, null, null, null, 2002, 0, null, null]]}"#;
    const ES_TABLE_COLUMNS: &str = r#"[{"name": "TABLE_CAT", "type": "keyword"}, {"name": "TABLE_SCHEM", "type": "keyword"}, {"name": "TABLE_NAME", "type": "keyword"}, {"name": "TABLE_TYPE", "type": "keyword"}, {"name": "REMARKS", "type": "keyword"}, {"name": "TYPE_CAT", "type": "keyword"}, {"name": "TYPE_SCHEM", "type": "keyword"}, {"name": "TYPE_NAME", "type": "keyword"}, {"name": "SELF_REFERENCING_COL_NAME", "type": "keyword"}, {"name": "REF_GENERATION", "type": "keyword"}]"#;

    fn as_json(t: &Table) -> Value {
        json!({
            "columns": t.cols.iter().map(|c| json!({"name": c.name, "type": c.ty})).collect::<Vec<_>>(),
            "rows": t.rows.iter().map(|r| r.iter().map(Val::to_json).collect::<Vec<_>>()).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn sys_statements_answer_like_es() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("sqlnull", &json!({"mappings": {"properties": {"k": {"type": "keyword"}, "a": {"type": "integer"}, "b": {"type": "boolean"}}}})).unwrap();
        let cat = node.cluster_name.clone();
        let mut want: Value = serde_json::from_str(ES_COLUMNS).unwrap();
        for r in want["rows"].as_array_mut().unwrap() {
            r[0] = json!(cat);
        }
        let got = as_json(&execute(&node, "SYS COLUMNS TABLE LIKE 'sqlnull'", None, None).unwrap());
        assert_eq!(got, want);
        let want: Value = serde_json::from_str(ES_TYPES).unwrap();
        assert_eq!(as_json(&execute(&node, "SYS TYPES", None, None).unwrap()), want);
        let t = execute(&node, "SYS TABLES", None, None).unwrap();
        assert_eq!(as_json(&t)["columns"], serde_json::from_str::<Value>(ES_TABLE_COLUMNS).unwrap());
        assert_eq!(as_json(&t)["rows"], json!([[cat, null, "sqlnull", "TABLE", "", null, null, null, null, null]]));
    }
}

/// ES SQL week, day and quarter functions and DATE_ADD across zones,
/// against ES 9.5.5's answers.
#[cfg(test)]
mod es955_date_tests {
    use super::*;

    #[test]
    fn week_day_and_quarter_functions_answer_like_es() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // [WEEK_OF_YEAR, ISO_WEEK_OF_YEAR, DAY_OF_WEEK, ISO_DAY_OF_WEEK, DATE_PART('week'), QUARTER]
        let cases: [(&str, [i64; 6]); 4] = [
            ("2021-01-01T00:00:00Z", [1, 53, 6, 5, 1, 1]),
            ("2021-01-03T00:00:00Z", [2, 53, 1, 7, 2, 1]),
            ("2020-12-31T00:00:00Z", [53, 53, 5, 4, 53, 4]),
            ("2024-12-30T00:00:00Z", [53, 1, 2, 1, 53, 4]),
        ];
        for (d, want) in cases {
            let x = format!("CAST('{d}' AS DATETIME)");
            let q = format!("SELECT WEEK_OF_YEAR({x}), ISO_WEEK_OF_YEAR({x}), DAY_OF_WEEK({x}), ISO_DAY_OF_WEEK({x}), DATE_PART('week', {x}), QUARTER({x})");
            let t = execute(&node, &q, None, None).unwrap_or_else(|e| panic!("{q}: {e}"));
            let got: Vec<i64> = t.rows[0].iter().map(|v| v.as_i64().unwrap_or(-1)).collect();
            assert_eq!(got, want.to_vec(), "{d}");
        }
    }

    #[test]
    fn date_add_month_end_answers_like_es_in_each_zone() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let q = "SELECT DATE_ADD('month', 1, CAST('2021-01-31T00:00:00Z' AS DATETIME)) AS r";
        for (zone, want) in [("UTC", "2021-02-28T00:00:00.000Z"), ("Europe/Berlin", "2021-02-28T01:00:00.000+01:00")] {
            let _z = crate::esql::set_sql_zone(crate::date::Tz::parse_query_option(zone).unwrap());
            let t = execute(&node, q, None, None).unwrap();
            let Val::Date(ms) = t.rows[0][0] else { panic!("{:?}", t.rows) };
            assert_eq!(crate::esql::fmt_date(ms), want, "{zone}");
        }
    }
}

/// ES SQL CAST / CONVERT targets (SQL and ODBC `SQL_` names) and their
/// rounding, against ES 9.5.5.
#[cfg(test)]
mod es955_convert_tests {
    use super::*;

    #[test]
    fn cast_and_convert_resolve_sql_and_odbc_names_like_es() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // ES 9.5.5: [3, -2, 3, 3] (Java Math.round: half up).
        let t = execute(&node, "SELECT CAST(2.7 AS LONG), CAST(-2.5 AS INTEGER), CONVERT(2.5, SQL_BIGINT), CAST(2.5 AS SHORT)", None, None).unwrap();
        assert_eq!(t.rows[0].iter().map(|v| v.as_i64()).collect::<Vec<_>>(), vec![Some(3), Some(-2), Some(3), Some(3)]);
        for (q, want) in [
            ("SELECT CONVERT(2.5, SQL_INTEGER)", "3"),
            ("SELECT CONVERT(2.5, SQL_SMALLINT)", "3"),
            ("SELECT CONVERT(-2.7, SQL_TINYINT)", "-3"),
            ("SELECT CONVERT(7, SQL_DOUBLE)", "7.0"),
            ("SELECT CONVERT(7, SQL_FLOAT)", "7.0"),
            ("SELECT CONVERT(7, SQL_REAL)", "7.0"),
            ("SELECT CONVERT(42, SQL_VARCHAR)", "42"),
            ("SELECT CONVERT(42, SQL_CHAR)", "42"),
            ("SELECT CONVERT('true', SQL_BIT)", "true"),
            ("SELECT CONVERT(1, BOOLEAN)", "true"),
            ("SELECT CAST(1 AS BOOL)", "true"),
            ("SELECT CAST(1.5 AS FLOAT)", "1.5"),
            ("SELECT CAST(7 AS STRING)", "7"),
            ("SELECT CAST(7 AS VARCHAR)", "7"),
            ("SELECT CONVERT(7, SQL_WLONGVARCHAR)", "7"),
            ("SELECT CONVERT(2.5, SQL_NUMERIC)", "2.5"),
            ("SELECT CAST(2.5 AS BIGINT)", "3"),
            ("SELECT CAST(2.5 AS TINYINT)", "3"),
            ("SELECT CAST(2.5 AS SMALLINT)", "3"),
            ("SELECT CAST(-2.5 AS LONG)", "-2"),
            ("SELECT CONVERT('2021-01-31T00:00:00Z', SQL_TIMESTAMP) IS NOT NULL", "true"),
            ("SELECT CONVERT(2.5, BIGINT)", "3"),
        ] {
            let t = execute(&node, q, None, None).unwrap_or_else(|e| panic!("{q}: {e}"));
            assert_eq!(t.rows[0][0].to_text(), want, "{q}");
        }
        // ES 9.5.5's SqlDataTypes: FLOAT is double (the ODBC name
        // overrides the ES float type), VARCHAR is text, CHAR keyword.
        for (raw, want) in [("FLOAT", "double"), ("SQL_FLOAT", "double"), ("REAL", "float"), ("VARCHAR", "text"), ("SQL_VARCHAR", "text"),
                            ("CHAR", "keyword"), ("SQL_CHAR", "keyword"), ("STRING", "keyword"), ("TIMESTAMP", "datetime"), ("SQL_BIT", "boolean"),
                            ("SQL_UBIGINT", "unsigned_long"), ("SQL_INTERVAL_DAY_TO_SECOND", "interval_day_to_second"), ("HALF_FLOAT", "half_float")] {
            assert_eq!(crate::esql::parse::sql_cast_type(raw).as_deref(), Some(want), "{raw}");
        }
        // Names ES does not recognize: a SQL_ name not in ODBC_TO_ES, unknown names.
        for q in ["SELECT CONVERT(1, SQL_BOOLEAN)", "SELECT CAST(1 AS SQL_LONG)", "SELECT CAST(1 AS NOPE)"] {
            let e = execute(&node, q, None, None).err().unwrap_or_else(|| panic!("{q}"));
            assert!(e.kind == "parsing_exception" && e.reason.contains("Does not recognize type"), "{q}: {e}");
        }
    }

    use crate::esql::Val;

    fn one(node: &Node, q: &str) -> (Val, String) {
        let t = execute(node, q, None, None).unwrap_or_else(|e| panic!("{q}: {e}"));
        (t.rows[0][0].clone(), t.cols[0].ty.clone())
    }

    fn err(node: &Node, q: &str) -> EsError {
        execute(node, q, None, None).err().unwrap_or_else(|| panic!("{q}: no error"))
    }

    /// ES 9.5.5 SQL TIME (`SqlDataTypeConverter.conversionToTime` and from
    /// TIME): from datetime, strings, integers (`millis % day`), rationals,
    /// booleans; to strings, numbers (millis of the day) and booleans;
    /// never to datetime.
    #[test]
    fn time_conversions_follow_es() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (q, want) in [
            ("SELECT CAST(CAST('2020-05-06T12:34:56.789Z' AS DATETIME) AS TIME)", "12:34:56.789Z"),
            ("SELECT CAST('12:34:56' AS TIME)", "12:34:56.000Z"),
            ("SELECT CAST('12:34:56.7+01:00' AS TIME)", "12:34:56.700+01:00"),
            ("SELECT CAST(45296789 AS TIME)", "12:34:56.789Z"),
            ("SELECT CONVERT(86400001, SQL_TIME)", "00:00:00.001Z"),
            ("SELECT CAST(1.6 AS TIME)", "00:00:00.002Z"),
            ("SELECT CAST(TRUE AS TIME)", "00:00:00.001Z"),
        ] {
            let (v, ty) = one(&node, q);
            assert_eq!((v, ty.as_str()), (Val::Str(want.into()), "time"), "{q}");
        }
        assert_eq!(one(&node, "SELECT CAST(CAST(45296789 AS TIME) AS LONG)").0, Val::Long(45296789));
        assert_eq!(one(&node, "SELECT CAST(CAST(0 AS TIME) AS BOOLEAN)").0, Val::Bool(false));
        assert_eq!(one(&node, "SELECT CAST(CAST(45296789 AS TIME) AS STRING)").0, Val::Str("12:34:56.789Z".into()));
        let e = err(&node, "SELECT CAST(CAST(1 AS TIME) AS DATETIME)");
        assert!(e.reason.contains("Cannot cast [TIME] to [DATETIME]"), "{e}");
        let e = err(&node, "SELECT CAST('noon' AS TIME)");
        assert!(e.reason.contains("cannot cast [noon] to [time]"), "{e}");
    }

    /// ES 9.5.5 SQL intervals (`ExpressionBuilder.visitInterval`,
    /// `Intervals`): every literal form and type, ES's declaration and
    /// pattern errors, ISO values (JSON) and the SQL text form when cast to
    /// a string; any other conversion is `Cannot cast`.
    #[test]
    fn interval_types_follow_es() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (q, iso, ty) in [
            ("SELECT INTERVAL 1 YEAR", "P1Y", "interval_year"),
            ("SELECT INTERVAL 14 MONTHS", "P14M", "interval_month"),
            ("SELECT INTERVAL 3 DAY", "PT72H", "interval_day"),
            ("SELECT INTERVAL -2 HOURS", "PT-2H", "interval_hour"),
            ("SELECT INTERVAL 90 MINUTE", "PT1H30M", "interval_minute"),
            ("SELECT INTERVAL '1.5' SECOND", "PT1.5S", "interval_second"),
            ("SELECT INTERVAL '1-2' YEAR TO MONTH", "P1Y2M", "interval_year_to_month"),
            ("SELECT INTERVAL '1 2' DAY TO HOUR", "PT26H", "interval_day_to_hour"),
            ("SELECT INTERVAL '1 2:03' DAY TO MINUTE", "PT26H3M", "interval_day_to_minute"),
            ("SELECT INTERVAL '1 2:03:04.5' DAY TO SECOND", "PT26H3M4.5S", "interval_day_to_second"),
            ("SELECT INTERVAL '2:03' HOUR TO MINUTE", "PT2H3M", "interval_hour_to_minute"),
            ("SELECT INTERVAL '2:03:04' HOUR TO SECOND", "PT2H3M4S", "interval_hour_to_second"),
            ("SELECT INTERVAL '3:04.25' MINUTE TO SECOND", "PT3M4.25S", "interval_minute_to_second"),
            ("SELECT -INTERVAL '1-2' YEAR TO MONTH", "P-1Y-2M", "interval_year_to_month"),
            ("SELECT INTERVAL -'1 2' DAY TO HOUR", "PT-26H", "interval_day_to_hour"),
        ] {
            let (v, t) = one(&node, q);
            assert_eq!((v, t.as_str()), (Val::Str(iso.into()), ty), "{q}");
        }
        assert_eq!(one(&node, "SELECT CAST(INTERVAL '1-2' YEAR TO MONTH AS STRING)").0, Val::Str("+1-2".into()));
        assert_eq!(one(&node, "SELECT CAST(INTERVAL '1 2:03:04.5' DAY TO SECOND AS VARCHAR)").0, Val::Str("+1 02:03:04.5".into()));
        assert_eq!(one(&node, "SELECT CAST(INTERVAL 1 YEAR AS INTERVAL_YEAR)").0, Val::Str("P1Y".into()));
        for (q, msg) in [
            ("SELECT CAST('1' AS INTERVAL_YEAR)", "Cannot cast [KEYWORD] to [INTERVAL_YEAR]"),
            ("SELECT CAST(1 AS INTERVAL_DAY)", "Cannot cast [INTEGER] to [INTERVAL_DAY]"),
            ("SELECT CAST(INTERVAL 1 DAY AS LONG)", "Cannot cast [INTERVAL_DAY] to [LONG]"),
            ("SELECT CONVERT(INTERVAL 1 YEAR, SQL_INTERVAL_MONTH)", "Cannot cast [INTERVAL_YEAR] to [INTERVAL_MONTH]"),
            ("SELECT INTERVAL '1' YEAR TO DAY", "Invalid interval declaration; YEAR trailing unit required to be MONTH, received DAY"),
            ("SELECT INTERVAL '1' HOUR TO DAY", "Invalid interval declaration; trailing unit [DAY] needs to be smaller than leading unit[HOUR], expected one of [MINUTE, SECOND]"),
            ("SELECT INTERVAL 1 DAY TO HOUR", "trailing unit [HOUR] specified but the value is with numeric (single unit), use the string notation instead"),
            ("SELECT INTERVAL 1.5 DAY", "Fractional values are not supported for intervals"),
            ("SELECT INTERVAL '1-12' YEAR TO MONTH", "Invalid [INTERVAL YEAR TO MONTH] value [1-12]: [MONTH] unit has illegal value [12], expected a positive number up to [11]"),
            ("SELECT INTERVAL '1 2' DAY TO MINUTE", "Invalid [INTERVAL DAY TO MINUTE] value [1 2]: incorrect format, expecting [numeric] [numeric]:[numeric]"),
            ("SELECT INTERVAL '1x' DAY", "Invalid [INTERVAL DAY] value [1x]: unexpected trailing characters found [x]"),
            ("SELECT INTERVAL '1:2' DAY TO HOUR", "Invalid [INTERVAL DAY TO HOUR] value [1:2]: expected [ ] (at [1]) but found [:]"),
        ] {
            let e = err(&node, q);
            assert!(e.reason.contains(msg), "{q}: {e}");
        }
        // Datetime ± interval (`DateTimeArithmeticOperation`).
        let (v, ty) = one(&node, "SELECT CAST('2020-01-31T10:00:00Z' AS DATETIME) + INTERVAL 1 MONTH");
        assert!(ty == "date" || ty == "datetime", "{ty}");
        assert_eq!(v, Val::Date(crate::date::parse_date_math("2020-02-29T10:00:00Z", &crate::date::DateFormatter::new("strict_date_optional_time").unwrap(), crate::date::Tz::UTC, 0, false).unwrap()));
        let (v, _) = one(&node, "SELECT CAST('2020-01-31T10:00:00Z' AS DATETIME) - INTERVAL '1 02:00' DAY TO MINUTE");
        assert_eq!(v, Val::Date(crate::date::parse_date_math("2020-01-30T08:00:00Z", &crate::date::DateFormatter::new("strict_date_optional_time").unwrap(), crate::date::Tz::UTC, 0, false).unwrap()));
        // Text formats show the SQL form.
        let t = execute(&node, "SELECT INTERVAL '1-2' YEAR TO MONTH AS i", None, None).unwrap();
        let txt = crate::esql::format::text_table(&t, "csv", None, false, false).unwrap();
        assert_eq!(txt.trim(), "+1-2");
    }

}
