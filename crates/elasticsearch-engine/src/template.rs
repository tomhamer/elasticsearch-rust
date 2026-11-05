//! Mustache search templates (ES `lang: mustache`).
//!
//! A small Mustache renderer with Elasticsearch's extensions:
//! `{{var}}` (JSON-escaped, no quotes), `{{{var}}}` / `{{& var}}` (raw),
//! dotted names and list indices (`{{a.b.0}}`), sections `{{#x}}..{{/x}}`
//! (lists iterate, objects push context, truthy values render once),
//! inverted sections `{{^x}}..{{/x}}` (defaults), comments `{{! .. }}`,
//! delimiter changes `{{=<% %>=}}`, and the ES lambdas `{{#toJson}}p{{/toJson}}`,
//! `{{#join}}p{{/join}}`, `{{#join delimiter='|'}}p{{/join delimiter='|'}}`,
//! `{{#url}}..{{/url}}`.

use crate::error::{EsError, Result};
use serde_json::Value;

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    Var { name: String, escape: bool },
    Section { name: String, inverted: bool, children: Vec<Node> },
}

fn template_err(msg: impl Into<String>) -> EsError {
    EsError::new(400, "general_script_exception", "Failed to compile inline script [template] using lang [mustache]")
        .caused_by(EsError::new(400, "mustache_exception", msg))
}

fn parse(src: &str) -> Result<Vec<Node>> {
    let mut open = "{{".to_string();
    let mut close = "}}".to_string();
    // Stack of (section name, inverted, children).
    let mut stack: Vec<(String, bool, Vec<Node>)> = vec![(String::new(), false, Vec::new())];
    let mut rest = src;
    loop {
        let Some(p) = rest.find(open.as_str()) else {
            if !rest.is_empty() {
                stack.last_mut().unwrap().2.push(Node::Text(rest.to_string()));
            }
            break;
        };
        if p > 0 {
            stack.last_mut().unwrap().2.push(Node::Text(rest[..p].to_string()));
        }
        let after = &rest[p + open.len()..];
        // Triple mustache.
        if open == "{{" && after.starts_with('{') {
            let end = after.find("}}}").ok_or_else(|| template_err("Unclosed tag"))?;
            let name = after[1..end].trim().to_string();
            stack.last_mut().unwrap().2.push(Node::Var { name, escape: false });
            rest = &after[end + 3..];
            continue;
        }
        let end = after.find(close.as_str()).ok_or_else(|| template_err("Unclosed tag"))?;
        let tag = after[..end].trim();
        rest = &after[end + close.len()..];
        let (sigil, body) = match tag.chars().next() {
            Some(c @ ('#' | '^' | '/' | '!' | '&' | '=' | '>')) => (Some(c), tag[1..].trim()),
            _ => (None, tag),
        };
        match sigil {
            Some('!') | Some('>') => {}
            Some('=') => {
                let body = body.trim_end_matches('=').trim();
                let mut parts = body.split_whitespace();
                if let (Some(o), Some(c)) = (parts.next(), parts.next()) {
                    open = o.to_string();
                    close = c.to_string();
                }
            }
            Some('#') | Some('^') => stack.push((body.to_string(), sigil == Some('^'), Vec::new())),
            Some('/') => {
                let (name, inverted, children) = stack.pop().ok_or_else(|| template_err("Unopened section"))?;
                if stack.is_empty() || name.split_whitespace().next() != body.split_whitespace().next() {
                    return Err(template_err(format!("Mismatched start/end tags: {name} != {body}")));
                }
                stack.last_mut().unwrap().2.push(Node::Section { name, inverted, children });
            }
            Some('&') => stack.last_mut().unwrap().2.push(Node::Var { name: body.to_string(), escape: false }),
            _ => stack.last_mut().unwrap().2.push(Node::Var { name: body.to_string(), escape: true }),
        }
    }
    if stack.len() != 1 {
        return Err(template_err(format!("Section [{}] not closed", stack.last().unwrap().0)));
    }
    Ok(stack.pop().unwrap().2)
}

fn lookup<'a>(ctx: &[&'a Value], name: &str) -> Option<&'a Value> {
    if name == "." {
        return ctx.last().copied();
    }
    let mut parts = name.split('.');
    let first = parts.next()?;
    let mut cur = ctx.iter().rev().find_map(|c| child(c, first))?;
    for p in parts {
        cur = child(cur, p)?;
    }
    Some(cur)
}

fn child<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    match v {
        Value::Object(m) => m.get(k),
        Value::Array(a) => k.parse::<usize>().ok().and_then(|i| a.get(i)),
        _ => None,
    }
}

/// Java-ish `toString` of a parameter value.
fn to_text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(to_text).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, v)| format!("{k}={}", to_text(v))).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

fn json_escape(s: &str) -> String {
    let q = serde_json::to_string(s).unwrap_or_default();
    q[1..q.len() - 1].to_string()
}

pub fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'*' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Array(a) => !a.is_empty(),
        Value::String(s) => !s.is_empty(),
        _ => true,
    }
}

fn render_nodes(nodes: &[Node], ctx: &mut Vec<&Value>, out: &mut String) -> Result<()> {
    for n in nodes {
        match n {
            Node::Text(t) => out.push_str(t),
            Node::Var { name, escape } => {
                if let Some(v) = lookup(ctx, name) {
                    let t = to_text(v);
                    out.push_str(&if *escape { json_escape(&t) } else { t });
                }
            }
            Node::Section { name, inverted, children } => {
                let head = name.split_whitespace().next().unwrap_or("");
                if !*inverted && matches!(head, "toJson" | "join" | "url") {
                    let mut inner = String::new();
                    render_nodes(children, ctx, &mut inner)?;
                    match head {
                        "url" => out.push_str(&url_encode(&inner)),
                        "toJson" => {
                            let key = inner.trim();
                            let v = lookup(ctx, key).cloned().unwrap_or(Value::Null);
                            out.push_str(&serde_json::to_string(&v).unwrap_or_default());
                        }
                        _ => {
                            let delim = name
                                .split_once("delimiter=")
                                .map(|(_, d)| d.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
                                .unwrap_or_else(|| ",".to_string());
                            let key = inner.trim();
                            match lookup(ctx, key) {
                                Some(Value::Array(a)) => out.push_str(&a.iter().map(to_text).collect::<Vec<_>>().join(&delim)),
                                Some(v) => out.push_str(&to_text(v)),
                                None => {}
                            }
                        }
                    }
                    continue;
                }
                let v = lookup(ctx, head);
                let on = v.map(truthy).unwrap_or(false);
                if *inverted {
                    if !on {
                        render_nodes(children, ctx, out)?;
                    }
                    continue;
                }
                if !on {
                    continue;
                }
                let v = v.unwrap();
                match v {
                    Value::Array(a) => {
                        for item in a {
                            ctx.push(item);
                            let r = render_nodes(children, ctx, out);
                            ctx.pop();
                            r?;
                        }
                    }
                    Value::Object(_) => {
                        ctx.push(v);
                        let r = render_nodes(children, ctx, out);
                        ctx.pop();
                        r?;
                    }
                    _ => render_nodes(children, ctx, out)?,
                }
            }
        }
    }
    Ok(())
}

/// Compile `src` through the script cache in the `template` context, as ES
/// compiles a search template: a template not cached counts one compilation
/// against the compilation rate limit, and a syntax error is reported here.
pub fn admit(src: &str) -> Result<()> {
    crate::script::admit_template(src, || parse(src).map(drop))
}

/// Render a Mustache template with `params` as the root context.
pub fn render(src: &str, params: &Value) -> Result<String> {
    let nodes = parse(src)?;
    let mut out = String::new();
    let mut ctx: Vec<&Value> = vec![params];
    render_nodes(&nodes, &mut ctx, &mut out)?;
    Ok(out)
}

/// Template source as text: a string is used as is, an object is serialised.
pub fn source_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Render a search template to the request body it denotes.
pub fn render_to_json(src: &str, params: &Value) -> Result<Value> {
    let s = render(src, params)?;
    serde_json::from_str(&s).map_err(|e| {
        EsError::new(400, "x_content_parse_exception", format!("[{}:{}] Unexpected character in rendered template: {e}", e.line(), e.column()))
    })
}

/// Like [`render_to_json`], but tolerates trailing commas before `]` / `}`
/// in the rendered text. Search applications render their templates this
/// way (a list section `{{#l}}"{{.}}",{{/l}}` leaves a trailing comma).
pub fn render_to_json_lenient(src: &str, params: &Value) -> Result<Value> {
    let s = strip_trailing_commas(&render(src, params)?);
    serde_json::from_str(&s).map_err(|e| {
        EsError::new(400, "x_content_parse_exception", format!("[{}:{}] Unexpected character in rendered template: {e}", e.line(), e.column()))
    })
}

/// Drop every `,` that is followed (across whitespace) by `]` or `}` outside
/// of string literals.
fn strip_trailing_commas(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut in_str = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if in_str {
            if c == b'\\' && i + 1 < b.len() {
                out.push_str(&s[i..i + 2]);
                i += 2;
                continue;
            }
            if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == b',' {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < b.len() && (b[j] == b']' || b[j] == b'}') {
                i += 1;
                continue;
            }
        }
        // Copy the whole UTF-8 character starting here.
        let ch_len = s[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        out.push_str(&s[i..i + ch_len]);
        i += ch_len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn basics() {
        let p = json!({"q": "he said \"hi\"", "n": 5, "tags": ["a", "b"], "obj": {"x": 1}, "flag": false});
        assert_eq!(render("{{q}}", &p).unwrap(), "he said \\\"hi\\\"");
        assert_eq!(render("{{{q}}}", &p).unwrap(), "he said \"hi\"");
        assert_eq!(render("{{n}}-{{obj.x}}-{{tags.1}}", &p).unwrap(), "5-1-b");
        assert_eq!(render("{{#tags}}<{{.}}>{{/tags}}", &p).unwrap(), "<a><b>");
        assert_eq!(render("{{#flag}}yes{{/flag}}{{^flag}}no{{/flag}}", &p).unwrap(), "no");
        assert_eq!(render("{{size}}{{^size}}10{{/size}}", &p).unwrap(), "10");
        assert_eq!(render("{{#toJson}}tags{{/toJson}}", &p).unwrap(), "[\"a\",\"b\"]");
        assert_eq!(render("{{#join}}tags{{/join}}", &p).unwrap(), "a,b");
        assert_eq!(render("{{#join delimiter='||'}}tags{{/join delimiter='||'}}", &p).unwrap(), "a||b");
        assert_eq!(render("{{#url}}a b&c{{/url}}", &p).unwrap(), "a+b%26c");
        assert_eq!(render("{{! c }}{{=<% %>=}}<%n%>", &p).unwrap(), "5");
        assert!(render("{{#a}}x", &p).is_err());
    }
}
