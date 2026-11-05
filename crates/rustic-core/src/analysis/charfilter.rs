//! Character filters: transform the raw text *before* tokenization.
//!
//! Mirrors Lucene's `CharFilter` family (`html_strip`, `mapping`).

use super::CharFilter;

/// Strips HTML/XML tags and decodes common character entities.
///
/// * Tags (`<b>`, `</div>`, `<br/>`, `<!-- comments -->`, `<!DOCTYPE>`,
///   `<?xml ... ?>`) are replaced by nothing, except that block-ish tags
///   are not specially handled: tag removal never inserts whitespace, as
///   in Lucene (`<p>a</p><p>b</p>` → `ab`). Use `escaped_tags` to keep
///   selected tags verbatim (ES `escaped_tags` option).
/// * `<script>` and `<style>` elements are removed together with their
///   content.
/// * Named entities `&amp; &lt; &gt; &quot; &apos; &nbsp;` (plus a handful
///   of other common ones) and numeric entities `&#NNN;` / `&#xHHH;` are
///   decoded. Unknown entities are left as-is.
#[derive(Clone, Debug, Default)]
pub struct HtmlStripCharFilter {
    /// Tag names (lowercase, without angle brackets) to leave untouched.
    pub escaped_tags: Vec<String>,
}

impl HtmlStripCharFilter {
    /// Create a filter that strips every tag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a filter that keeps the named tags (e.g. `["b", "i"]`) verbatim.
    pub fn with_escaped_tags(tags: &[&str]) -> Self {
        Self {
            escaped_tags: tags.iter().map(|t| t.to_lowercase()).collect(),
        }
    }

    fn decode_entity(name: &str) -> Option<String> {
        if let Some(num) = name.strip_prefix('#') {
            let code = if let Some(hex) = num.strip_prefix(['x', 'X']) {
                u32::from_str_radix(hex, 16).ok()?
            } else {
                num.parse::<u32>().ok()?
            };
            return char::from_u32(code).map(|c| c.to_string());
        }
        let s = match name {
            "amp" => "&",
            "lt" => "<",
            "gt" => ">",
            "quot" => "\"",
            "apos" => "'",
            "nbsp" => "\u{a0}",
            "copy" => "©",
            "reg" => "®",
            "trade" => "™",
            "hellip" => "…",
            "mdash" => "—",
            "ndash" => "–",
            "laquo" => "«",
            "raquo" => "»",
            "lsquo" => "‘",
            "rsquo" => "’",
            "ldquo" => "“",
            "rdquo" => "”",
            "euro" => "€",
            "pound" => "£",
            "yen" => "¥",
            "cent" => "¢",
            "deg" => "°",
            "times" => "×",
            "divide" => "÷",
            "middot" => "·",
            "bull" => "•",
            _ => return None,
        };
        Some(s.to_string())
    }

    /// Name of the tag starting at `rest` (which begins just after `<`),
    /// lowercased, without a leading `/`.
    fn tag_name(rest: &str) -> String {
        let rest = rest.strip_prefix('/').unwrap_or(rest);
        rest.chars()
            .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == ':')
            .collect::<String>()
            .to_lowercase()
    }
}

impl CharFilter for HtmlStripCharFilter {
    fn clone_box(&self) -> Box<dyn CharFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut i = 0usize;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            match bytes[i] {
                b'<' => {
                    let rest = &text[i + 1..];
                    // Comments.
                    if let Some(after) = rest.strip_prefix("!--") {
                        match after.find("-->") {
                            Some(end) => i += 1 + 3 + end + 3,
                            None => i = bytes.len(),
                        }
                        continue;
                    }
                    // <!DOCTYPE ...>, <?xml ... ?>, <![CDATA[ ... ]]>
                    if rest.starts_with("![CDATA[") {
                        let body = &rest["![CDATA[".len()..];
                        match body.find("]]>") {
                            Some(end) => {
                                out.push_str(&body[..end]);
                                i += 1 + "![CDATA[".len() + end + 3;
                            }
                            None => {
                                out.push_str(body);
                                i = bytes.len();
                            }
                        }
                        continue;
                    }
                    let name = Self::tag_name(rest);
                    let looks_like_tag = rest.starts_with('!')
                        || rest.starts_with('?')
                        || (!name.is_empty()
                            && rest
                                .chars()
                                .next()
                                .map(|c| c.is_alphabetic() || c == '/')
                                .unwrap_or(false));
                    if !looks_like_tag {
                        // A bare '<' (e.g. "a < b"): keep it.
                        out.push('<');
                        i += 1;
                        continue;
                    }
                    if self.escaped_tags.iter().any(|t| *t == name) {
                        // Copy the escaped tag verbatim.
                        match rest.find('>') {
                            Some(end) => {
                                out.push_str(&text[i..i + 1 + end + 1]);
                                i += 1 + end + 1;
                            }
                            None => {
                                out.push_str(&text[i..]);
                                i = bytes.len();
                            }
                        }
                        continue;
                    }
                    // Find the end of the tag, respecting quoted attributes.
                    let mut j = i + 1;
                    let mut quote: Option<u8> = None;
                    while j < bytes.len() {
                        let b = bytes[j];
                        match quote {
                            Some(q) if b == q => quote = None,
                            Some(_) => {}
                            None if b == b'"' || b == b'\'' => quote = Some(b),
                            None if b == b'>' => break,
                            None => {}
                        }
                        j += 1;
                    }
                    let is_close = rest.starts_with('/');
                    i = (j + 1).min(bytes.len());
                    // Drop script/style content.
                    if !is_close && (name == "script" || name == "style") {
                        let close = format!("</{name}");
                        match find_ci(&text[i..], &close) {
                            Some(end) => {
                                let after = i + end;
                                i = match text[after..].find('>') {
                                    Some(gt) => after + gt + 1,
                                    None => bytes.len(),
                                };
                            }
                            None => i = bytes.len(),
                        }
                    }
                }
                b'&' => {
                    let rest = &text[i + 1..];
                    let end = rest
                        .char_indices()
                        .take_while(|(k, c)| {
                            *k < 12 && (c.is_ascii_alphanumeric() || *c == '#')
                        })
                        .map(|(k, c)| k + c.len_utf8())
                        .last()
                        .unwrap_or(0);
                    if end > 0 && rest[end..].starts_with(';') {
                        if let Some(decoded) = Self::decode_entity(&rest[..end]) {
                            out.push_str(&decoded);
                            i += 1 + end + 1;
                            continue;
                        }
                    }
                    out.push('&');
                    i += 1;
                }
                _ => {
                    let c = text[i..].chars().next().unwrap();
                    out.push(c);
                    i += c.len_utf8();
                }
            }
        }
        out
    }
}

/// Case-insensitive ASCII substring search returning the byte offset in
/// `haystack`.
fn find_ci(haystack: &str, needle_lower: &str) -> Option<usize> {
    let n = needle_lower.len();
    if n == 0 || haystack.len() < n {
        return None;
    }
    let hb = haystack.as_bytes();
    let nb = needle_lower.as_bytes();
    (0..=hb.len() - n).find(|&k| {
        hb[k..k + n]
            .iter()
            .zip(nb)
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
    })
}

/// Replaces occurrences of configured strings with their replacements.
///
/// Equivalent to Lucene's `MappingCharFilter`: at each position the
/// *longest* matching key wins; matched text is not re-scanned, so
/// mappings never cascade. Keys may be multi-character; replacements may
/// be empty (deletion).
#[derive(Clone, Debug, Default)]
pub struct MappingCharFilter {
    /// `(from, to)` pairs, sorted by descending `from` length.
    mappings: Vec<(String, String)>,
}

impl MappingCharFilter {
    /// Build from `(from, to)` pairs.
    pub fn new(pairs: &[(&str, &str)]) -> Self {
        let mut mappings: Vec<(String, String)> = pairs
            .iter()
            .filter(|(from, _)| !from.is_empty())
            .map(|(from, to)| (from.to_string(), to.to_string()))
            .collect();
        mappings.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        Self { mappings }
    }

    /// Build from ES-style rule strings `"from => to"` (whitespace around
    /// `=>` is trimmed; ` `-style escapes are not interpreted).
    /// Rules without `=>` are ignored.
    pub fn from_rules(rules: &[&str]) -> Self {
        let pairs: Vec<(&str, &str)> = rules
            .iter()
            .filter_map(|r| {
                let (from, to) = r.split_once("=>")?;
                Some((from.trim(), to.trim()))
            })
            .collect();
        Self::new(&pairs)
    }

    /// The configured `(from, to)` pairs, longest `from` first.
    pub fn mappings(&self) -> &[(String, String)] {
        &self.mappings
    }
}

impl CharFilter for MappingCharFilter {
    fn clone_box(&self) -> Box<dyn CharFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, text: &str) -> String {
        if self.mappings.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        let mut i = 0usize;
        while i < text.len() {
            let rest = &text[i..];
            if let Some((from, to)) = self.mappings.iter().find(|(f, _)| rest.starts_with(f.as_str())) {
                out.push_str(to);
                i += from.len();
            } else {
                let c = rest.chars().next().unwrap();
                out.push(c);
                i += c.len_utf8();
            }
        }
        out
    }
}

/// Lucene `PersianCharFilter`: zero-width non-joiner (U+200C) becomes a
/// space so the tokenizer splits on it. Offsets are corrected back to the
/// original bytes (the ZWNJ is 3 bytes, the space 1).
#[derive(Clone, Debug, Default)]
pub struct PersianCharFilter;

impl CharFilter for PersianCharFilter {
    fn clone_box(&self) -> Box<dyn CharFilter> {
        Box::new(PersianCharFilter)
    }

    fn filter(&self, text: &str) -> String {
        if !text.contains('\u{200C}') {
            return text.to_string();
        }
        text.replace('\u{200C}', " ")
    }

    fn filter_mapped(&self, text: &str) -> (String, Vec<(u32, i64)>) {
        if !text.contains('\u{200C}') {
            return (text.to_string(), Vec::new());
        }
        let mut out = String::with_capacity(text.len());
        let mut map = Vec::new();
        let mut diff = 0i64;
        for c in text.chars() {
            if c == '\u{200C}' {
                out.push(' ');
                diff += 2;
                map.push((out.len() as u32, diff));
            } else {
                out.push(c);
            }
        }
        (out, map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::tokenizer::StandardTokenizer;
    use crate::analysis::Analyzer;

    // ---------- HtmlStripCharFilter ----------

    #[test]
    fn html_strip_basic_tags() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("<p>Hello <b>World</b></p>"), "Hello World");
    }

    #[test]
    fn html_strip_self_closing_and_attributes() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(
            f.filter(r#"a<br/>b<img src="x>y" alt='q>r'/>c"#),
            "abc"
        );
    }

    #[test]
    fn html_strip_comments_doctype_pi() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("<!DOCTYPE html><!-- c > --><?xml version='1'?>x"), "x");
    }

    #[test]
    fn html_strip_unterminated_comment() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("a<!-- never closed"), "a");
    }

    #[test]
    fn html_strip_script_and_style_content() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(
            f.filter("a<script>var x = '<b>';</script>b<STYLE>p{}</STYLE>c"),
            "abc"
        );
    }

    #[test]
    fn html_strip_named_entities() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(
            f.filter("&lt;a&gt; &amp; &quot;b&quot; &#39;c&#39; d&nbsp;e"),
            "<a> & \"b\" 'c' d\u{a0}e"
        );
    }

    #[test]
    fn html_strip_numeric_entities() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("&#72;&#105;&#x21;&#X263A;"), "Hi!☺");
    }

    #[test]
    fn html_strip_unknown_entity_kept() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("&bogus; & &#99999999; &amp"), "&bogus; & &#99999999; &amp");
    }

    #[test]
    fn html_strip_bare_less_than_kept() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("a < b and c<3"), "a < b and c<3");
    }

    #[test]
    fn html_strip_escaped_tags() {
        let f = HtmlStripCharFilter::with_escaped_tags(&["b"]);
        assert_eq!(f.filter("<p><b>bold</b> <i>it</i></p>"), "<b>bold</b> it");
    }

    #[test]
    fn html_strip_cdata() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("x<![CDATA[<raw> & stuff]]>y"), "x<raw> & stuffy");
    }

    #[test]
    fn html_strip_unicode_passthrough() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("<p>日本語 café</p>"), "日本語 café");
    }

    #[test]
    fn html_strip_empty() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter(""), "");
    }

    #[test]
    fn html_strip_unterminated_tag() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("abc<div class='x"), "abc");
    }

    // ---------- MappingCharFilter ----------

    #[test]
    fn mapping_basic() {
        let f = MappingCharFilter::new(&[("ph", "f"), ("qu", "k")]);
        assert_eq!(f.filter("phone quick"), "fone kick");
    }

    #[test]
    fn mapping_longest_match_first() {
        let f = MappingCharFilter::new(&[("a", "1"), ("aa", "2"), ("aaa", "3")]);
        assert_eq!(f.filter("aaaa"), "31");
    }

    #[test]
    fn mapping_no_cascade() {
        let f = MappingCharFilter::new(&[("a", "b"), ("b", "c")]);
        assert_eq!(f.filter("ab"), "bc");
    }

    #[test]
    fn mapping_deletion_and_unicode() {
        let f = MappingCharFilter::new(&[("-", ""), ("ß", "ss"), ("&", " and ")]);
        assert_eq!(f.filter("wi-fi straße A&B"), "wifi strasse A and B");
    }

    #[test]
    fn mapping_from_rules() {
        let f = MappingCharFilter::from_rules(&["٠ => 0", "١ => 1", "bad rule", ":) => _happy_"]);
        assert_eq!(f.mappings().len(), 3);
        assert_eq!(f.filter("١٠ :)"), "10 _happy_");
    }

    #[test]
    fn mapping_empty_is_identity() {
        let f = MappingCharFilter::new(&[]);
        assert_eq!(f.filter("unchanged"), "unchanged");
        let f = MappingCharFilter::new(&[("", "x")]);
        assert_eq!(f.filter("unchanged"), "unchanged");
    }

    // ---------- integration with Analyzer ----------

    #[test]
    fn analyzer_applies_char_filters_before_tokenizing() {
        let a = Analyzer::new(Box::new(StandardTokenizer))
            .with_char_filter(Box::new(HtmlStripCharFilter::new()))
            .with_char_filter(Box::new(MappingCharFilter::new(&[("-", " ")])));
        let toks = a.analyze("<b>wi-fi</b> &amp; <i>tea</i>");
        let texts: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["wi", "fi", "tea"]);
        // Offsets refer to the filtered text "wi fi & tea".
        assert_eq!((toks[2].offset_from, toks[2].offset_to), (8, 11));
    }

    #[test]
    fn char_filter_clone_box() {
        let f: Box<dyn CharFilter> = Box::new(MappingCharFilter::new(&[("a", "b")]));
        let g = f.clone_box();
        assert_eq!(g.filter("aa"), "bb");
    }
}
