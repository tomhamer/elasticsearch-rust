//! `pattern_text` (logsdb plugin, `PatternTextFieldMapper`): log messages
//! indexed like `match_only_text` with the `delimiter` analyzer, plus a
//! `<field>.template_id` keyword sub-field: the murmur3 hash of the message
//! with its digit-bearing tokens removed (`PatternTextValueProcessor`).

use base64::Engine as _;

/// The internal name of the `delimiter` analyzer in the registry.
pub const DELIMITER_ANALYZER: &str = "__pattern_text_delimiter";

/// `PatternTextValueProcessor.MAX_LOG_LEN_TO_STORE_AS_DOC_VALUE`.
const MAX_LEN: usize = 8 * 1024;

/// The index-time analyzer name of a `pattern_text` field's `analyzer`.
pub fn analyzer_name(param: Option<&str>) -> &'static str {
    match param {
        Some("standard") => "standard",
        _ => DELIMITER_ANALYZER,
    }
}

fn is_arg(token: &str) -> bool {
    token.chars().any(|c| c.is_ascii_digit() || c.is_numeric())
}

/// `PatternTextValueProcessor.splitInternal`: the template (args removed).
fn template(text: &[char]) -> String {
    let is_delim = |c: char| c.is_whitespace() || c == '[' || c == ']';
    let mut out = String::with_capacity(text.len());
    // Java `Pattern.split`: tokens between delimiters, trailing empty
    // tokens dropped.
    let mut tokens: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for (i, c) in text.iter().enumerate() {
        if is_delim(*c) {
            tokens.push((start, i));
            start = i + 1;
        }
    }
    tokens.push((start, text.len()));
    while tokens.len() > 1 && tokens.last().is_some_and(|(a, b)| a == b) {
        tokens.pop();
    }
    if tokens.len() == 1 && tokens[0].0 == tokens[0].1 && text.is_empty() {
        tokens.clear();
    }
    let mut idx = 0usize;
    for (a, b) in tokens {
        if a == b {
            if idx + 1 < text.len() {
                out.push(text[idx]);
                idx += 1;
            }
        } else {
            let tok: String = text[a..b].iter().collect();
            if !is_arg(&tok) {
                out.push_str(&tok);
            }
            idx += b - a;
            if idx < text.len() {
                out.push(text[idx]);
                idx += 1;
            }
        }
    }
    while idx < text.len() {
        out.push(text[idx]);
        idx += 1;
    }
    out
}

/// `PatternTextValueProcessor.templateId`.
pub fn template_id(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let t: String = if chars.len() > MAX_LEN {
        // Too long: the (truncated) text itself is the template.
        chars[..MAX_LEN].iter().collect()
    } else {
        template(&chars)
    };
    let (h1, _) = crate::tsdb::murmur3_128(t.as_bytes(), 0);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(h1.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates() {
        let t = |s: &str| template(&s.chars().collect::<Vec<_>>());
        assert_eq!(t("Found 5 errors for service [cheddar1]"), "Found  errors for service []");
        assert_eq!(t("some log message with no arg"), "some log message with no arg");
    }
}
