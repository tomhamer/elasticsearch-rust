//! Java-compatible text helpers for ingest: `Double.toString`, `toString` of
//! collections (mustache renders values this way), JSON/URL escaping for
//! templates, `NumberFormatException` messages and `PatternSyntaxException`
//! messages (ES echoes them verbatim in processor config errors).

use serde_json::Value;

/// `Double.toString` (shortest repr digits, `1.0E10` style outside [1e-3, 1e7)).
pub fn java_double(d: f64) -> String {
    if d.is_nan() {
        return "NaN".into();
    }
    if d.is_infinite() {
        return if d > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if d == 0.0 {
        return if d.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = d.abs();
    if (1e-3..1e7).contains(&a) {
        let s = format!("{d}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        sci(&format!("{d:e}"))
    }
}

/// `Float.toString`.
pub fn java_float(f: f32) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = f.abs();
    if (1e-3..1e7).contains(&a) {
        let s = format!("{f}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        sci(&format!("{f:e}"))
    }
}

fn sci(s: &str) -> String {
    // Rust: "1.234e10" / "1e-5" -> Java "1.234E10" / "1.0E-5".
    let (m, e) = s.split_once('e').unwrap_or((s, "0"));
    let m = if m.contains('.') { m.to_string() } else { format!("{m}.0") };
    format!("{m}E{e}")
}

/// A JSON double as the exact `f32` Java would hold after `Float.parseFloat`.
pub fn f32_json(f: f32) -> Value {
    let s = format!("{f}");
    let d: f64 = s.parse().unwrap_or(f as f64);
    serde_json::Number::from_f64(d).map(Value::Number).unwrap_or(Value::Null)
}

/// Java `toString` of a JSON value as stored in a document map.
pub fn java_tostring(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if n.is_f64() { java_double(n.as_f64().unwrap_or(0.0)) } else { n.to_string() }
        }
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(java_tostring).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, x)| format!("{k}={}", java_tostring(x))).collect::<Vec<_>>().join(", ")),
    }
}

/// How a mustache `{{x}}` renders a value: null is empty, lists render as an
/// index map (`{0=a, 1=b}`), everything else as Java `toString`.
pub fn mustache_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Array(a) => format!("{{{}}}", a.iter().enumerate().map(|(i, x)| format!("{i}={}", java_tostring(x))).collect::<Vec<_>>().join(", ")),
        other => java_tostring(other),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaType {
    Json,
    Plain,
    Form,
}

impl MediaType {
    pub fn parse(s: &str) -> Option<MediaType> {
        match s {
            "application/json" => Some(MediaType::Json),
            "text/plain" => Some(MediaType::Plain),
            "application/x-www-form-urlencoded" => Some(MediaType::Form),
            _ => None,
        }
    }
    pub fn escape(self, s: &str) -> String {
        match self {
            MediaType::Json => json_escape(s),
            MediaType::Plain => s.to_string(),
            MediaType::Form => url_encode(s),
        }
    }
}

/// Jackson `JsonStringEncoder.quoteAsString`.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// `URLEncoder.encode(s, UTF_8)`.
pub fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'.' | b'-' | b'*' | b'_' => out.push(b as char),
            b' ' => out.push('+'),
            b => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Message of `NumberFormatException` for `Integer/Long.parseX(s)`.
pub fn nfe_int(s: &str) -> String {
    if s.is_empty() { "For input string: \"\"".to_string() } else { format!("For input string: \"{s}\"") }
}

/// Message of `NumberFormatException` for `Float/Double.parseX(s)`.
pub fn nfe_float(s: &str) -> String {
    let t = s.trim();
    if t.is_empty() {
        return "empty String".into();
    }
    // Java reports "multiple points" for a second '.' in a digit run.
    let dots = t.chars().filter(|c| *c == '.').count();
    if dots > 1 && t.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == '+') {
        return "multiple points".into();
    }
    format!("For input string: \"{s}\"")
}

/// Java `PatternSyntaxException.getMessage()` for the pattern mistakes users
/// make; `None` when the pattern looks syntactically valid to this scanner.
pub fn java_pattern_error(p: &str) -> Option<String> {
    let cs: Vec<char> = p.chars().collect();
    let n = cs.len();
    let msg = |desc: &str, idx: usize| -> String {
        let mut m = format!("{desc} near index {idx}\n{p}");
        if idx < n {
            m.push('\n');
            m.push_str(&" ".repeat(idx));
            m.push('^');
        }
        m
    };
    let mut depth: Vec<usize> = Vec::new();
    let mut i = 0usize;
    let mut prev_atom = false; // whether a quantifier may follow
    while i < n {
        let c = cs[i];
        match c {
            '\\' => {
                if i + 1 >= n {
                    return Some(msg("Unescaped trailing backslash", n));
                }
                if cs[i + 1] == 'Q' {
                    // quote until \E
                    let mut j = i + 2;
                    while j + 1 < n && !(cs[j] == '\\' && cs[j + 1] == 'E') {
                        j += 1;
                    }
                    i = (j + 2).min(n);
                    prev_atom = true;
                    continue;
                }
                if cs[i + 1] == 'c' && i + 2 >= n {
                    return Some(msg("Illegal control escape sequence", i + 1));
                }
                i += 2;
                prev_atom = true;
            }
            '[' => {
                // character class: find matching ']' honouring nesting and escapes
                let mut j = i + 1;
                if j < n && cs[j] == '^' {
                    j += 1;
                }
                let mut first = true;
                let mut lvl = 1;
                let mut closed = false;
                while j < n {
                    match cs[j] {
                        '\\' => j += 1,
                        '[' => lvl += 1,
                        ']' if !first || true => {
                            if first && j == i + 1 {
                                // "[]" - Java treats a leading ']' as literal only after '^'; keep scanning
                            }
                            lvl -= 1;
                            if lvl == 0 {
                                closed = true;
                                break;
                            }
                        }
                        '-' => {
                            // range check a-b
                            if j > i + 1 && j + 1 < n && cs[j + 1] != ']' && cs[j - 1] != '[' {
                                let (lo, hi) = (cs[j - 1], cs[j + 1]);
                                if hi != '\\' && hi != '[' && (lo as u32) > (hi as u32) {
                                    return Some(msg("Illegal character range", j + 1));
                                }
                            }
                        }
                        _ => {}
                    }
                    first = false;
                    j += 1;
                }
                if !closed {
                    if cs[n - 1] == '-' && n >= 2 && n - 1 > i + 1 {
                        return Some(msg("Illegal character range", n));
                    }
                    return Some(msg("Unclosed character class", if n == i + 1 { i } else { n - 1 }));
                }
                i = j + 1;
                prev_atom = true;
            }
            '(' => {
                depth.push(i);
                if i + 1 < n && cs[i + 1] == '?' {
                    if i + 2 >= n {
                        return Some(msg("Unknown inline modifier", i + 2));
                    }
                    match cs[i + 2] {
                        ':' | '=' | '!' | '>' => i += 3,
                        '<' => {
                            if i + 3 < n && (cs[i + 3] == '=' || cs[i + 3] == '!') {
                                i += 4;
                            } else {
                                // named group
                                let mut j = i + 3;
                                if j < n && !cs[j].is_ascii_alphabetic() {
                                    return Some(msg("capturing group name does not start with a Latin letter", j));
                                }
                                while j < n && cs[j].is_ascii_alphanumeric() {
                                    j += 1;
                                }
                                if j >= n || cs[j] != '>' {
                                    return Some(msg("named capturing group is missing trailing '>'", j));
                                }
                                i = j + 1;
                            }
                        }
                        'i' | 'm' | 's' | 'x' | 'u' | 'd' | 'U' | '-' => {
                            let mut j = i + 2;
                            while j < n && "imsxudU-".contains(cs[j]) {
                                j += 1;
                            }
                            if j >= n || (cs[j] != ')' && cs[j] != ':') {
                                return Some(msg("Unknown inline modifier", j));
                            }
                            if cs[j] == ')' {
                                depth.pop();
                            }
                            i = j + 1;
                        }
                        _ => return Some(msg("Unknown inline modifier", i + 2)),
                    }
                } else {
                    i += 1;
                }
                prev_atom = false;
            }
            ')' => {
                if depth.pop().is_none() {
                    return Some(if i == 0 { "Unmatched closing ')'\n".to_string() + p } else { msg("Unmatched closing ')'", i - 1) });
                }
                i += 1;
                prev_atom = true;
            }
            '*' | '+' | '?' => {
                if !prev_atom {
                    return Some(msg(&format!("Dangling meta character '{c}'"), i));
                }
                i += 1;
                // lazy / possessive suffix
                if i < n && (cs[i] == '?' || cs[i] == '+') {
                    i += 1;
                } else {
                    prev_atom = false;
                    // "a**" -> the second '*' is dangling
                    continue;
                }
                prev_atom = false;
            }
            '{' => {
                if !prev_atom {
                    return Some(msg("Illegal repetition", i + 1));
                }
                let mut j = i + 1;
                let mut digits = 0;
                while j < n && cs[j].is_ascii_digit() {
                    j += 1;
                    digits += 1;
                }
                if digits == 0 {
                    return Some(msg("Illegal repetition", i + 1));
                }
                let lo: u64 = cs[i + 1..j].iter().collect::<String>().parse().unwrap_or(0);
                let mut hi = Some(lo);
                if j < n && cs[j] == ',' {
                    j += 1;
                    let s = j;
                    while j < n && cs[j].is_ascii_digit() {
                        j += 1;
                    }
                    hi = if j > s { cs[s..j].iter().collect::<String>().parse().ok() } else { None };
                }
                if j >= n || cs[j] != '}' {
                    return Some(msg("Unclosed counted closure", j));
                }
                if let Some(h) = hi {
                    if h < lo {
                        return Some(msg("Illegal repetition range", j));
                    }
                }
                i = j + 1;
                if i < n && (cs[i] == '?' || cs[i] == '+') {
                    i += 1;
                }
                prev_atom = true;
            }
            '|' => {
                i += 1;
                prev_atom = false;
            }
            '^' | '$' => {
                i += 1;
                prev_atom = true;
            }
            _ => {
                i += 1;
                prev_atom = true;
            }
        }
    }
    if let Some(_open) = depth.pop() {
        return Some(msg("Unclosed group", n));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles() {
        assert_eq!(java_double(1.0), "1.0");
        assert_eq!(java_double(1.5), "1.5");
        assert_eq!(java_double(1.0e10), "1.0E10");
        assert_eq!(java_double(1.234e-5), "1.234E-5");
        assert_eq!(java_double(12345678.9), "1.23456789E7");
        assert_eq!(java_double(0.001), "0.001");
        assert_eq!(java_float(1e11f32), "1.0E11");
    }

    #[test]
    fn escapes() {
        assert_eq!(json_escape("a\"b\\c\n"), "a\\\"b\\\\c\\n");
        assert_eq!(url_encode("a b&c"), "a+b%26c");
    }

    #[test]
    fn rendering() {
        let v: Value = serde_json::json!({"a": 1, "b": [1]});
        assert_eq!(mustache_string(&v), "{a=1, b=[1]}");
        assert_eq!(mustache_string(&serde_json::json!([1, "x"])), "{0=1, 1=x}");
    }

    #[test]
    fn pattern_errors() {
        assert_eq!(java_pattern_error("(").unwrap(), "Unclosed group near index 1\n(");
        assert_eq!(java_pattern_error("*").unwrap(), "Dangling meta character '*' near index 0\n*\n^");
        assert_eq!(java_pattern_error("a)").unwrap(), "Unmatched closing ')' near index 0\na)\n^");
        assert_eq!(java_pattern_error(")").unwrap(), "Unmatched closing ')'\n)");
        assert_eq!(java_pattern_error("[abc").unwrap(), "Unclosed character class near index 3\n[abc\n   ^");
        assert_eq!(java_pattern_error("a{2").unwrap(), "Unclosed counted closure near index 3\na{2");
        assert_eq!(java_pattern_error("a{2,1}").unwrap(), "Illegal repetition range near index 5\na{2,1}\n     ^");
        assert_eq!(java_pattern_error("a**").unwrap(), "Dangling meta character '*' near index 2\na**\n  ^");
        assert!(java_pattern_error("a++b").is_none());
        assert!(java_pattern_error("(?<=a+)b").is_none());
    }
}
