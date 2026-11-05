//! `kql` query (x-pack-kql): Kibana Query Language translated to the query
//! DSL, then parsed by the regular DSL parser.
//!
//! Grammar: `or` < `and` < `not` < primary. A primary is `( expr )`,
//! `field:value words`, `field:(value words)`, `field:{ nested expr }`,
//! `field >= value` (`>`, `>=`, `<`, `<=`) or bare free text. Consecutive
//! unquoted words form one value (`title:foo bar` is the value `foo bar`).

use super::dsl::QueryContext;
use crate::error::{EsError, Result};
use serde_json::{json, Value};

/// Escaped `*` inside a word (literal star, not a wildcard).
const LIT_STAR: char = '\u{0}';

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Colon,
    Op(&'static str),
    And,
    Or,
    Not,
    Word(String),
    Quoted(String),
}

fn lex(s: &str) -> Vec<(Tok, usize)> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out: Vec<(Tok, usize)> = Vec::new();
    while i < c.len() {
        let at = i;
        match c[i] {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '(' => { out.push((Tok::LParen, at)); i += 1 }
            ')' => { out.push((Tok::RParen, at)); i += 1 }
            '{' => { out.push((Tok::LBrace, at)); i += 1 }
            '}' => { out.push((Tok::RBrace, at)); i += 1 }
            ':' => { out.push((Tok::Colon, at)); i += 1 }
            '>' | '<' => {
                let eq = c.get(i + 1) == Some(&'=');
                out.push((Tok::Op(match (c[i], eq) { ('>', true) => ">=", ('>', false) => ">", ('<', true) => "<=", _ => "<" }), at));
                i += if eq { 2 } else { 1 };
            }
            '"' => {
                let mut v = String::new();
                i += 1;
                while i < c.len() && c[i] != '"' {
                    if c[i] == '\\' && i + 1 < c.len() { i += 1; }
                    v.push(c[i]);
                    i += 1;
                }
                i += 1;
                out.push((Tok::Quoted(v), at));
            }
            _ => {
                let mut v = String::new();
                while i < c.len() && !matches!(c[i], ' ' | '\t' | '\n' | '\r' | '(' | ')' | '{' | '}' | ':' | '<' | '>' | '"') {
                    if c[i] == '\\' && i + 1 < c.len() {
                        v.push(if c[i + 1] == '*' { LIT_STAR } else { c[i + 1] });
                        i += 2;
                    } else {
                        v.push(c[i]);
                        i += 1;
                    }
                }
                let t = match v.to_ascii_lowercase().as_str() { "and" => Tok::And, "or" => Tok::Or, "not" => Tok::Not, _ => Tok::Word(v) };
                out.push((t, at));
            }
        }
    }
    // A trailing operator keyword is plain text.
    if let Some((t, _)) = out.last_mut() {
        let w = match t { Tok::And => Some("and"), Tok::Or => Some("or"), Tok::Not => Some("not"), _ => None };
        if let Some(w) = w { *t = Tok::Word(w.to_string()); }
    }
    out
}

struct P<'a, 'b> {
    t: Vec<(Tok, usize)>,
    i: usize,
    q: &'a str,
    ctx: &'a QueryContext<'b>,
    case_insensitive: bool,
    time_zone: Option<String>,
    default_fields: Vec<String>,
    /// `field:{ ... }` context: the nested path prefix.
    prefix: Option<String>,
    nest: crate::parse_depth::Nesting,
}

impl P<'_, '_> {
    fn peek(&self) -> Option<&Tok> { self.t.get(self.i).map(|(t, _)| t) }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) { self.i += 1; true } else { false }
    }
    fn err<T>(&self, msg: &str) -> Result<T> {
        let col = self.t.get(self.i).map(|(_, p)| *p).unwrap_or(self.q.chars().count()) + 1;
        Err(EsError::query_shard(self.ctx.index_name, self.ctx.index_uuid, format!("Failed to parse KQL query [{}]", self.q))
            .caused_by(EsError::new(400, "kql_parsing_exception", format!("line 1:{col}: {msg}"))))
    }
    /// Text of the input ANTLR reports for "no viable alternative": the
    /// failed rule's tokens up to the offending one.
    fn rule_input(&self) -> String {
        if self.peek().is_none() && self.i >= 2 {
            if let (Some((Tok::Word(f), _)), Some((Tok::Colon, _))) = (self.t.get(self.i - 2), self.t.get(self.i - 1)) {
                return format!("{f}:");
            }
        }
        self.tok_text()
    }
    fn tok_text(&self) -> String {
        match self.peek() {
            Some(Tok::And) => "and".into(), Some(Tok::Or) => "or".into(), Some(Tok::Not) => "not".into(),
            Some(Tok::Word(w)) | Some(Tok::Quoted(w)) => w.clone(),
            Some(Tok::RParen) => ")".into(), Some(Tok::LParen) => "(".into(), Some(Tok::Colon) => ":".into(),
            Some(Tok::LBrace) => "{".into(), Some(Tok::RBrace) => "}".into(), Some(Tok::Op(o)) => (*o).into(),
            None => "<EOF>".into(),
        }
    }

    /// Runs `f` one nesting level deeper (see [`crate::parse_depth`]).
    fn nested<T>(&mut self, counted: bool, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let levels = self.nest.snapshot();
        if !self.nest.enter(counted) {
            return self.err(&format!("query nested more than {} levels deep", self.nest.max()));
        }
        let r = f(self);
        self.nest.restore(levels);
        r
    }

    fn or(&mut self) -> Result<Value> {
        self.nested(true, Self::or_unnested)
    }

    fn or_unnested(&mut self) -> Result<Value> {
        let mut parts = vec![self.and()?];
        while self.eat(&Tok::Or) { parts.push(self.and()?); }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { json!({"bool": {"should": parts, "minimum_should_match": 1}}) })
    }
    fn and(&mut self) -> Result<Value> {
        let mut parts = vec![self.not()?];
        while self.eat(&Tok::And) { parts.push(self.not()?); }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { json!({"bool": {"must": parts}}) })
    }
    fn not(&mut self) -> Result<Value> {
        self.nested(false, Self::not_unnested)
    }

    fn not_unnested(&mut self) -> Result<Value> {
        if self.eat(&Tok::Not) {
            let q = self.not()?;
            return Ok(json!({"bool": {"must_not": [q]}}));
        }
        self.primary()
    }

    /// Unquoted words joined by single spaces, or one quoted string.
    fn value(&mut self) -> Result<(String, bool)> {
        match self.peek().cloned() {
            Some(Tok::Quoted(s)) => { self.i += 1; Ok((s, true)) }
            Some(Tok::Word(_)) => {
                let mut parts = Vec::new();
                while let Some(Tok::Word(w)) = self.peek().cloned() { parts.push(w); self.i += 1; }
                Ok((parts.join(" "), false))
            }
            _ => { let t = self.rule_input(); self.err(&format!("no viable alternative at input '{t}'")) }
        }
    }

    fn primary(&mut self) -> Result<Value> {
        match self.peek().cloned() {
            Some(Tok::LParen) => {
                self.i += 1;
                let q = self.or()?;
                if !self.eat(&Tok::RParen) { let t = self.tok_text(); return self.err(&format!("missing ')' at '{t}'")); }
                Ok(q)
            }
            Some(Tok::Word(f)) | Some(Tok::Quoted(f)) => {
                let quoted_field = matches!(self.peek(), Some(Tok::Quoted(_)));
                match self.t.get(self.i + 1).map(|(t, _)| t.clone()) {
                    Some(Tok::Colon) => {
                        self.i += 2;
                        self.field_value(&f)
                    }
                    Some(Tok::Op(op)) => {
                        self.i += 2;
                        let (v, _) = self.value()?;
                        self.range(&f, op, &v)
                    }
                    _ => {
                        let _ = quoted_field;
                        let (v, quoted) = self.value()?;
                        self.free_text(&v, quoted)
                    }
                }
            }
            _ => { let t = self.tok_text(); self.err(&format!("no viable alternative at input '{t}'")) }
        }
    }

    fn field_value(&mut self, field: &str) -> Result<Value> {
        if self.eat(&Tok::LBrace) {
            let outer = self.prefix.clone();
            let path = match &outer { Some(p) => format!("{p}.{field}"), None => field.to_string() };
            self.prefix = Some(path.clone());
            let inner = self.or();
            self.prefix = outer;
            let inner = inner?;
            if !self.eat(&Tok::RBrace) { let t = self.tok_text(); return self.err(&format!("missing '}}' at '{t}'")); }
            return Ok(json!({"nested": {"path": path, "query": inner, "score_mode": "none"}}));
        }
        self.field_query_value(field)
    }

    /// `fieldQueryValue`: a literal, `NOT fieldQueryValue`, or a
    /// parenthesized `booleanFieldQueryValue` (ES 9: `field:(a AND b)`).
    fn field_query_value(&mut self, field: &str) -> Result<Value> {
        if self.eat(&Tok::Not) {
            let q = self.field_query_value(field)?;
            return Ok(json!({"bool": {"must_not": [q]}}));
        }
        if self.eat(&Tok::LParen) {
            let q = self.boolean_field_query_value(field)?;
            if !self.eat(&Tok::RParen) { let t = self.tok_text(); return self.err(&format!("missing ')' at '{t}'")); }
            return Ok(q);
        }
        let (v, quoted) = self.value()?;
        self.field_query(field, &v, quoted)
    }

    /// `booleanFieldQueryValue`: left-associative `AND` / `OR` of field values.
    fn boolean_field_query_value(&mut self, field: &str) -> Result<Value> {
        let mut acc = self.field_query_value(field)?;
        loop {
            let op = if self.eat(&Tok::And) {
                "must"
            } else if self.eat(&Tok::Or) {
                "should"
            } else {
                return Ok(acc);
            };
            let rhs = self.field_query_value(field)?;
            acc = if op == "must" { json!({"bool": {"must": [acc, rhs]}}) } else { json!({"bool": {"should": [acc, rhs], "minimum_should_match": 1}}) };
        }
    }

    fn full(&self, field: &str) -> String {
        match &self.prefix { Some(p) => format!("{p}.{field}"), None => field.to_string() }
    }

    /// Fields a (possibly wildcard) name expands to (ES `withFields`):
    /// searchable fields only, metadata fields ES ignores dropped.
    fn expand(&self, field: &str) -> Vec<String> {
        const IGNORED: &[&str] = &["_seq_no", "_index_mode", "_routing", "_ignored", "_nested_path", "_field_names"];
        let full = self.full(field);
        if matches!(full.as_str(), "_id" | "_index") {
            return vec![full];
        }
        let names: Vec<String> = if full.contains('*') { self.ctx.lookup.matching(&full) } else if self.ctx.lookup.get(&full).is_some() { vec![full] } else { Vec::new() };
        names
            .into_iter()
            .filter(|f| !IGNORED.contains(&f.as_str()))
            .filter(|f| self.ctx.lookup.get(f).is_some_and(|r| r.indexed || r.doc_values || matches!(r.declared, crate::mapping::EsType::ConstantKeyword)))
            .collect()
    }

    /// Wrap a leaf query on `f` in `nested` when it lives in a nested path
    /// (unless we are already inside that nested block).
    fn nest(&self, f: &str, q: Value) -> Value {
        let np = self.ctx.lookup.get(f).and_then(|r| r.nested_path.clone());
        match np {
            Some(p) if self.prefix.as_deref() != Some(p.as_str()) => json!({"nested": {"path": p, "query": q, "score_mode": "none"}}),
            _ => q,
        }
    }

    fn combine(&self, mut qs: Vec<Value>) -> Value {
        match qs.len() {
            0 => json!({"match_none": {}}),
            1 => qs.pop().unwrap(),
            _ => json!({"bool": {"should": qs, "minimum_should_match": 1}}),
        }
    }

    /// ES `KqlAstBuilder.visitFieldQuery`, per field.
    fn leaf(&self, f: &str, v: &str, quoted: bool) -> Value {
        use crate::mapping::EsType;
        if !quoted && v == "*" {
            return json!({"exists": {"field": f}});
        }
        let wildcard = !quoted && v.contains('*');
        let declared = self.ctx.lookup.get(f).map(|r| r.declared);
        if f == "_id" || f == "_index" {
            return json!({"term": {f: v.replace(LIT_STAR, "*")}});
        }
        if wildcard && declared == Some(EsType::Keyword) {
            return json!({"wildcard": {f: {"value": v.replace(LIT_STAR, "\\*"), "case_insensitive": self.case_insensitive}}});
        }
        if wildcard {
            return json!({"query_string": {"query": escape_lucene(v), "fields": [f]}});
        }
        let v = v.replace(LIT_STAR, "*");
        if declared == Some(EsType::Date) {
            let mut r = json!({"gte": v, "lte": v});
            if let Some(tz) = &self.time_zone {
                r["time_zone"] = json!(tz);
            }
            return json!({"range": {f: r}});
        }
        if declared == Some(EsType::Keyword) {
            return json!({"term": {f: {"value": v, "case_insensitive": self.case_insensitive}}});
        }
        // `KqlAstBuilder.fieldQueryApplier`: match queries are lenient (ES 9).
        if quoted { json!({"match_phrase": {f: v}}) } else { json!({"match": {f: {"query": v, "lenient": true}}}) }
    }

    fn field_query(&self, field: &str, v: &str, quoted: bool) -> Result<Value> {
        if self.full(field) == "*" && !quoted && v == "*" {
            return Ok(json!({"match_all": {}}));
        }
        let fs = self.expand(field);
        let qs = fs.iter().map(|f| self.nest(f, self.leaf(f, v, quoted))).collect();
        Ok(self.combine(qs))
    }

    fn range(&self, field: &str, op: &str, v: &str) -> Result<Value> {
        let fs = self.expand(field);
        let key = match op { ">" => "gt", ">=" => "gte", "<" => "lt", _ => "lte" };
        let qs = fs
            .iter()
            .map(|f| {
                let mut r = json!({ key: v.replace(LIT_STAR, "*") });
                if let Some(tz) = &self.time_zone { r["time_zone"] = json!(tz); }
                self.nest(f, json!({"range": {f: r}}))
            })
            .collect();
        Ok(self.combine(qs))
    }

    fn free_text(&self, v: &str, quoted: bool) -> Result<Value> {
        if !quoted && v == "*" {
            return Ok(json!({"match_all": {}}));
        }
        if !quoted && v.contains('*') {
            // ES: a wildcard field-less query is a `query_string`.
            return Ok(json!({"query_string": {"query": escape_lucene(v), "fields": self.default_fields}}));
        }
        let v = v.replace(LIT_STAR, "*");
        let mut mm = json!({"query": v, "fields": self.default_fields, "lenient": true});
        mm["type"] = json!(if quoted { "phrase" } else { "best_fields" });
        Ok(json!({"multi_match": mm}))
    }
}

/// Lucene `QueryParser.escape` of everything but the (unescaped) `*`
/// wildcards (ES `ParserUtils.escapeLuceneQueryString(text, true)`).
fn escape_lucene(v: &str) -> String {
    let mut out = String::with_capacity(v.len() + 8);
    for c in v.chars() {
        match c {
            '*' => out.push('*'),
            LIT_STAR => out.push_str("\\*"),
            '\\' | '+' | '-' | '!' | '(' | ')' | ':' | '^' | '[' | ']' | '"' | '{' | '}' | '~' | '?' | '|' | '&' | '/' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// `{"kql": {...}}` body → query DSL object.
pub fn rewrite(body: &Value, ctx: &QueryContext) -> Result<Value> {
    let o = body.as_object().ok_or_else(|| EsError::parsing("[kql] query malformed, no start_object after query name"))?;
    let q = o.get("query").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[kql] requires a [query]"))?;
    let default_fields = match o.get("default_field").and_then(Value::as_str) {
        Some(f) => vec![f.to_string()],
        None => ctx.default_fields.clone(),
    };
    let mut p = P {
        t: lex(q),
        i: 0,
        q,
        ctx,
        case_insensitive: o.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false),
        time_zone: o.get("time_zone").and_then(Value::as_str).map(str::to_string),
        default_fields,
        prefix: None,
        nest: crate::parse_depth::Nesting::new(256),
    };
    let dsl = if p.t.is_empty() {
        json!({"match_all": {}})
    } else {
        let v = p.or()?;
        if p.i != p.t.len() {
            let t = p.tok_text();
            return p.err(&format!("mismatched input '{t}' expecting {{<EOF>, 'and', 'or', UNQUOTED_LITERAL, WILDCARD}}"));
        }
        v
    };
    Ok(match o.get("boost") {
        Some(b) => json!({"bool": {"must": [dsl], "boost": b}}),
        None => dsl,
    })
}
