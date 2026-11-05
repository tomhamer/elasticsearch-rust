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
}

fn sql_err(full: &str, off: usize, msg: impl AsRef<str>) -> EsError {
    EsError::new(400, "parsing_exception", format!("{}: {}", crate::esql::pos_prefix(full, off), msg.as_ref()))
}

fn sql_params(v: Option<&Value>) -> Params {
    let mut p = Params::default();
    if let Some(Value::Array(a)) = v {
        for x in a {
            let val = match x {
                Value::Object(m) if m.contains_key("value") => m["value"].clone(),
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
    parse_with_counter(full, params, &std::cell::Cell::new(0))
}

fn parse_with_counter(full: &str, params: &Params, np: &std::cell::Cell<usize>) -> Result<Statement> {
    let trimmed = full.trim().trim_end_matches(';');
    let toks = lex(trimmed, 0, trimmed.len(), Dialect::Sql)?;
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
            if p.eat_kw("AS") || matches!(p.peek(), Tok::QIdent(_)) || matches!(p.peek(), Tok::Ident(n) if !CLAUSES.iter().any(|c| n.eq_ignore_ascii_case(c))) { p.ident()?; }
        } else {
        let start = p.pos();
        let mut end = start;
        while !p.at_eof() && !CLAUSES.iter().any(|c| p.at_kw(c)) && !p.at_kw("AS") {
            end = p.toks[p.i].end;
            p.i += 1;
        }
        if p.eat_kw("AS") {
            p.ident()?;
        }
        let raw = trimmed[start..end].trim();
        // `idx alias`
        let raw = if !raw.starts_with('"') && raw.contains(' ') { raw.split_whitespace().next().unwrap_or(raw) } else { raw };
        if raw.is_empty() {
            return Err(sql_err(trimmed, start, "mismatched input expecting a table name"));
        }
        index = Some(unquote(raw));
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
        if !evals.is_empty() {
            cmds.push(Cmd::Eval(evals));
        }
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
        if !evals.is_empty() {
            cmds.push(Cmd::Eval(evals));
        }
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

fn wildcard_match(pat: &str, s: &str) -> bool {
    let re = format!("^{}$", regex::escape(pat).replace('%', ".*").replace('_', ".").replace("\\*", ".*"));
    regex::Regex::new(&re).map(|r| r.is_match(s)).unwrap_or(false)
}

/// Execute a SQL statement into a table.
pub fn execute(node: &Node, query: &str, params_json: Option<&Value>, filter: Option<&Value>) -> Result<Table> {
    execute_with(node, query, params_json, filter, None, None)
}

/// Open a bounded unsorted SELECT/WHERE scan when every wanted shard has a local
/// readable copy. Deterministic row-local projections are evaluated per page;
/// aggregation, sorting and other expressions retain normal planning.
pub fn scan_with(node: &Node, query: &str, params_json: Option<&Value>, filter: Option<&Value>, restrict: Option<exec::Restrict>, shards: Option<crate::esql::remote::SharedShardSource>) -> Result<Option<exec::SqlScan>> {
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
    let mut ctx = exec::new_ctx(node, query, filter, 1_000_000, 1_000_000);
    ctx.restrict = restrict;
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

/// [`execute`] with the caller's DLS/FLS (`restrict`) and, on a cluster,
/// the shard source that reads every wanted shard from a started copy on
/// any node (as ES|QL does).
pub fn execute_with(node: &Node, query: &str, params_json: Option<&Value>, filter: Option<&Value>, restrict: Option<crate::esql::exec::Restrict>, shards: Option<crate::esql::remote::SharedShardSource>) -> Result<Table> {
    let params = sql_params(params_json);
    let stmt = parse(query, &params)?;
    match stmt {
        Statement::Select(q) => {
            let trimmed = query.trim().trim_end_matches(';');
            let mut ctx = exec::new_ctx(node, trimmed, filter, 1_000_000, 1_000_000);
            ctx.restrict = restrict;
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
                rows.push(vec![Val::Str(node.cluster_name.clone()), Val::Str(svc.name.clone()), Val::Str("TABLE".into()), Val::Str("INDEX".into())]);
                for a in svc.aliases().keys() {
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
            let cols = exec::index_columns(node, &[ix])?;
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
    let params = sql_params(params_json);
    let Statement::Select(q) = parse(query, &params)? else {
        return Err(EsError::illegal_argument("only SELECT statements can be translated"));
    };
    if q.cmds.iter().any(|c| matches!(c, Cmd::SqlPivot { .. })) {
        return Err(EsError::illegal_argument("PIVOT translation to Query DSL is not supported"));
    }
    let index = q.index.clone().ok_or_else(|| EsError::illegal_argument("cannot translate a query without FROM"))?;
    let cols = exec::index_columns(node, &[index])?;
    let types: HashMap<String, String> = cols.iter().map(|c| (c.name.clone(), c.ty.clone())).collect();
    let mut out = serde_json::Map::new();
    let mut filters = Vec::new();
    let mut musts = Vec::new();
    let mut fields: Vec<Value> = Vec::new();
    let mut sort: Vec<Value> = Vec::new();
    let mut aggs: Option<Value> = None;
    for c in &q.cmds {
        match c {
            Cmd::Where(e, _) if aggs.is_none() => match exec::to_dsl(e, &types, true) {
                Some((dsl, true)) => musts.push(dsl),
                Some((dsl, false)) => filters.push(dsl),
                None => return Err(EsError::illegal_argument("WHERE condition cannot be translated to Query DSL")),
            },
            Cmd::Stats { aggs: a, by, .. } => {
                let mut sources = Vec::new();
                for b in by {
                    if let Expr::Col(n, _) = &b.expr {
                        let field = if types.get(n).map(|t| t == "text").unwrap_or(false) { format!("{n}.keyword") } else { n.clone() };
                        sources.push(json!({ b.name.clone(): {"terms": {"field": field, "missing_bucket": true, "order": "asc"}}}));
                    }
                }
                let mut metrics = serde_json::Map::new();
                for f in a {
                    if let Expr::Call(n, args, _) = &f.expr {
                        let target = match args.first() {
                            Some(Expr::Col(c, _)) => Some(c.clone()),
                            _ => None,
                        };
                        let m = match (n.as_str(), target) {
                            ("COUNT", None) => continue,
                            ("COUNT", Some(t)) => json!({"filter": {"exists": {"field": t}}}),
                            ("COUNT_DISTINCT", Some(t)) => json!({"cardinality": {"field": t}}),
                            ("AVG", Some(t)) => json!({"avg": {"field": t}}),
                            ("SUM", Some(t)) => json!({"sum": {"field": t}}),
                            ("MIN", Some(t)) => json!({"min": {"field": t}}),
                            ("MAX", Some(t)) => json!({"max": {"field": t}}),
                            ("MEDIAN", Some(t)) | ("PERCENTILE", Some(t)) => json!({"percentiles": {"field": t}}),
                            _ => continue,
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

fn field_spec(n: &str, types: &HashMap<String, String>) -> Value {
    match types.get(n).map(String::as_str) {
        Some("date") | Some("date_nanos") => json!({"field": n, "format": "strict_date_optional_time_nanos"}),
        _ => json!({ "field": n }),
    }
}
