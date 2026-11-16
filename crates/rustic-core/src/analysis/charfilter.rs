//! Character filters: transform the raw text *before* tokenization.
//!
//! Mirrors Lucene's `CharFilter` family (`html_strip`, `mapping`).

use super::CharFilter;

/// Strips HTML/XML tags and decodes character entities (Lucene
/// `HTMLStripCharFilter`).
///
/// * Inline element tags (`<b>`, `<span>`, `<a>`, `<img>`, ...) are
///   removed; every other tag (block-level: `<p>`, `<div>`, `<li>`,
///   `<br>`, ...) becomes a newline, so words in separate blocks stay
///   separate (`<p>a</p><p>b</p>` → `\na\n\nb\n`, SIDE 012). Comments,
///   `<!DOCTYPE ...>` and `<?...?>` are removed; `<script>` / `<style>`
///   elements, content included, become one newline. Use `escaped_tags` to
///   keep selected tags verbatim (ES `escaped_tags` option).
/// * Lucene's named entities (`HTML_ENTITIES`, `&nbsp;` being a space) and
///   numeric entities `&#NNN;` / `&#xHHH;` are decoded. Unknown entities
///   are left as-is.
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
        HTML_ENTITIES.binary_search_by(|(n, _)| (*n).cmp(name)).ok().map(|i| HTML_ENTITIES[i].1.to_string())
    }

    /// Lucene's inline elements: their tags are removed without a newline.
    fn is_inline(name: &str) -> bool {
        matches!(
            name,
            "a" | "b" | "i" | "q" | "s" | "u" | "abbr" | "acronym" | "basefont" | "bdo" | "big" | "cite" | "code" | "dfn" | "em"
                | "font" | "img" | "input" | "kbd" | "label" | "samp" | "select" | "small" | "span" | "strike" | "strong" | "sub"
                | "sup" | "textarea" | "tt" | "var"
        )
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
                    let unterminated = j >= bytes.len();
                    i = (j + 1).min(bytes.len());
                    // <!DOCTYPE ...> and <?...?> are removed.
                    let markup = rest.starts_with('!') || rest.starts_with('?');
                    // Drop script/style content: the element is one newline.
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
                        out.push('\n');
                        continue;
                    }
                    if !markup && !unterminated && !Self::is_inline(&name) {
                        out.push('\n');
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

/// Lucene `HTMLCharacterEntities`: every named entity `html_strip` decodes
/// (sorted by name; the upper-case variants Lucene accepts included).
const HTML_ENTITIES: &[(&str, char)] = &[
    ("AElig", '\u{c6}'),
    ("AMP", '\u{26}'),
    ("Aacute", '\u{c1}'),
    ("Acirc", '\u{c2}'),
    ("Agrave", '\u{c0}'),
    ("Alpha", '\u{391}'),
    ("Aring", '\u{c5}'),
    ("Atilde", '\u{c3}'),
    ("Auml", '\u{c4}'),
    ("Beta", '\u{392}'),
    ("COPY", '\u{a9}'),
    ("Ccedil", '\u{c7}'),
    ("Chi", '\u{3a7}'),
    ("Dagger", '\u{2021}'),
    ("Delta", '\u{394}'),
    ("ETH", '\u{d0}'),
    ("Eacute", '\u{c9}'),
    ("Ecirc", '\u{ca}'),
    ("Egrave", '\u{c8}'),
    ("Epsilon", '\u{395}'),
    ("Eta", '\u{397}'),
    ("Euml", '\u{cb}'),
    ("GT", '\u{3e}'),
    ("Gamma", '\u{393}'),
    ("Iacute", '\u{cd}'),
    ("Icirc", '\u{ce}'),
    ("Igrave", '\u{cc}'),
    ("Iota", '\u{399}'),
    ("Iuml", '\u{cf}'),
    ("Kappa", '\u{39a}'),
    ("LT", '\u{3c}'),
    ("Lambda", '\u{39b}'),
    ("Mu", '\u{39c}'),
    ("Ntilde", '\u{d1}'),
    ("Nu", '\u{39d}'),
    ("OElig", '\u{152}'),
    ("Oacute", '\u{d3}'),
    ("Ocirc", '\u{d4}'),
    ("Ograve", '\u{d2}'),
    ("Omega", '\u{3a9}'),
    ("Omicron", '\u{39f}'),
    ("Oslash", '\u{d8}'),
    ("Otilde", '\u{d5}'),
    ("Ouml", '\u{d6}'),
    ("Phi", '\u{3a6}'),
    ("Pi", '\u{3a0}'),
    ("Prime", '\u{2033}'),
    ("Psi", '\u{3a8}'),
    ("QUOT", '\u{22}'),
    ("REG", '\u{ae}'),
    ("Rho", '\u{3a1}'),
    ("Scaron", '\u{160}'),
    ("Sigma", '\u{3a3}'),
    ("THORN", '\u{de}'),
    ("Tau", '\u{3a4}'),
    ("Theta", '\u{398}'),
    ("Uacute", '\u{da}'),
    ("Ucirc", '\u{db}'),
    ("Ugrave", '\u{d9}'),
    ("Upsilon", '\u{3a5}'),
    ("Uuml", '\u{dc}'),
    ("Xi", '\u{39e}'),
    ("Yacute", '\u{dd}'),
    ("Yuml", '\u{178}'),
    ("Zeta", '\u{396}'),
    ("aacute", '\u{e1}'),
    ("acirc", '\u{e2}'),
    ("acute", '\u{b4}'),
    ("aelig", '\u{e6}'),
    ("agrave", '\u{e0}'),
    ("alefsym", '\u{2135}'),
    ("alpha", '\u{3b1}'),
    ("amp", '\u{26}'),
    ("and", '\u{2227}'),
    ("ang", '\u{2220}'),
    ("apos", '\u{27}'),
    ("aring", '\u{e5}'),
    ("asymp", '\u{2248}'),
    ("atilde", '\u{e3}'),
    ("auml", '\u{e4}'),
    ("bdquo", '\u{201e}'),
    ("beta", '\u{3b2}'),
    ("brvbar", '\u{a6}'),
    ("bull", '\u{2022}'),
    ("cap", '\u{2229}'),
    ("ccedil", '\u{e7}'),
    ("cedil", '\u{b8}'),
    ("cent", '\u{a2}'),
    ("chi", '\u{3c7}'),
    ("circ", '\u{2c6}'),
    ("clubs", '\u{2663}'),
    ("cong", '\u{2245}'),
    ("copy", '\u{a9}'),
    ("crarr", '\u{21b5}'),
    ("cup", '\u{222a}'),
    ("curren", '\u{a4}'),
    ("dArr", '\u{21d3}'),
    ("dagger", '\u{2020}'),
    ("darr", '\u{2193}'),
    ("deg", '\u{b0}'),
    ("delta", '\u{3b4}'),
    ("diams", '\u{2666}'),
    ("divide", '\u{f7}'),
    ("eacute", '\u{e9}'),
    ("ecirc", '\u{ea}'),
    ("egrave", '\u{e8}'),
    ("empty", '\u{2205}'),
    ("emsp", '\u{2003}'),
    ("ensp", '\u{2002}'),
    ("epsilon", '\u{3b5}'),
    ("equiv", '\u{2261}'),
    ("eta", '\u{3b7}'),
    ("eth", '\u{f0}'),
    ("euml", '\u{eb}'),
    ("euro", '\u{20ac}'),
    ("exist", '\u{2203}'),
    ("fnof", '\u{192}'),
    ("forall", '\u{2200}'),
    ("frac12", '\u{bd}'),
    ("frac14", '\u{bc}'),
    ("frac34", '\u{be}'),
    ("frasl", '\u{2044}'),
    ("gamma", '\u{3b3}'),
    ("ge", '\u{2265}'),
    ("gt", '\u{3e}'),
    ("hArr", '\u{21d4}'),
    ("harr", '\u{2194}'),
    ("hearts", '\u{2665}'),
    ("hellip", '\u{2026}'),
    ("iacute", '\u{ed}'),
    ("icirc", '\u{ee}'),
    ("iexcl", '\u{a1}'),
    ("igrave", '\u{ec}'),
    ("image", '\u{2111}'),
    ("infin", '\u{221e}'),
    ("int", '\u{222b}'),
    ("iota", '\u{3b9}'),
    ("iquest", '\u{bf}'),
    ("isin", '\u{2208}'),
    ("iuml", '\u{ef}'),
    ("kappa", '\u{3ba}'),
    ("lArr", '\u{21d0}'),
    ("lambda", '\u{3bb}'),
    ("lang", '\u{2329}'),
    ("laquo", '\u{ab}'),
    ("larr", '\u{2190}'),
    ("lceil", '\u{2308}'),
    ("ldquo", '\u{201c}'),
    ("le", '\u{2264}'),
    ("lfloor", '\u{230a}'),
    ("lowast", '\u{2217}'),
    ("loz", '\u{25ca}'),
    ("lrm", '\u{200e}'),
    ("lsaquo", '\u{2039}'),
    ("lsquo", '\u{2018}'),
    ("lt", '\u{3c}'),
    ("macr", '\u{af}'),
    ("mdash", '\u{2014}'),
    ("micro", '\u{b5}'),
    ("middot", '\u{b7}'),
    ("minus", '\u{2212}'),
    ("mu", '\u{3bc}'),
    ("nabla", '\u{2207}'),
    ("nbsp", '\u{20}'),
    ("ndash", '\u{2013}'),
    ("ne", '\u{2260}'),
    ("ni", '\u{220b}'),
    ("not", '\u{ac}'),
    ("notin", '\u{2209}'),
    ("nsub", '\u{2284}'),
    ("ntilde", '\u{f1}'),
    ("nu", '\u{3bd}'),
    ("oacute", '\u{f3}'),
    ("ocirc", '\u{f4}'),
    ("oelig", '\u{153}'),
    ("ograve", '\u{f2}'),
    ("oline", '\u{203e}'),
    ("omega", '\u{3c9}'),
    ("omicron", '\u{3bf}'),
    ("oplus", '\u{2295}'),
    ("or", '\u{2228}'),
    ("ordf", '\u{aa}'),
    ("ordm", '\u{ba}'),
    ("oslash", '\u{f8}'),
    ("otilde", '\u{f5}'),
    ("otimes", '\u{2297}'),
    ("ouml", '\u{f6}'),
    ("para", '\u{b6}'),
    ("part", '\u{2202}'),
    ("permil", '\u{2030}'),
    ("perp", '\u{22a5}'),
    ("phi", '\u{3c6}'),
    ("pi", '\u{3c0}'),
    ("piv", '\u{3d6}'),
    ("plusmn", '\u{b1}'),
    ("pound", '\u{a3}'),
    ("prime", '\u{2032}'),
    ("prod", '\u{220f}'),
    ("prop", '\u{221d}'),
    ("psi", '\u{3c8}'),
    ("quot", '\u{22}'),
    ("rArr", '\u{21d2}'),
    ("radic", '\u{221a}'),
    ("rang", '\u{232a}'),
    ("raquo", '\u{bb}'),
    ("rarr", '\u{2192}'),
    ("rceil", '\u{2309}'),
    ("rdquo", '\u{201d}'),
    ("real", '\u{211c}'),
    ("reg", '\u{ae}'),
    ("rfloor", '\u{230b}'),
    ("rho", '\u{3c1}'),
    ("rlm", '\u{200f}'),
    ("rsaquo", '\u{203a}'),
    ("rsquo", '\u{2019}'),
    ("sbquo", '\u{201a}'),
    ("scaron", '\u{161}'),
    ("sdot", '\u{22c5}'),
    ("sect", '\u{a7}'),
    ("shy", '\u{ad}'),
    ("sigma", '\u{3c3}'),
    ("sigmaf", '\u{3c2}'),
    ("sim", '\u{223c}'),
    ("spades", '\u{2660}'),
    ("sub", '\u{2282}'),
    ("sube", '\u{2286}'),
    ("sum", '\u{2211}'),
    ("sup", '\u{2283}'),
    ("sup1", '\u{b9}'),
    ("sup2", '\u{b2}'),
    ("sup3", '\u{b3}'),
    ("supe", '\u{2287}'),
    ("szlig", '\u{df}'),
    ("tau", '\u{3c4}'),
    ("there4", '\u{2234}'),
    ("theta", '\u{3b8}'),
    ("thetasym", '\u{3d1}'),
    ("thinsp", '\u{2009}'),
    ("thorn", '\u{fe}'),
    ("tilde", '\u{2dc}'),
    ("times", '\u{d7}'),
    ("trade", '\u{2122}'),
    ("uArr", '\u{21d1}'),
    ("uacute", '\u{fa}'),
    ("uarr", '\u{2191}'),
    ("ucirc", '\u{fb}'),
    ("ugrave", '\u{f9}'),
    ("uml", '\u{a8}'),
    ("upsih", '\u{3d2}'),
    ("upsilon", '\u{3c5}'),
    ("uuml", '\u{fc}'),
    ("weierp", '\u{2118}'),
    ("xi", '\u{3be}'),
    ("yacute", '\u{fd}'),
    ("yen", '\u{a5}'),
    ("yuml", '\u{ff}'),
    ("zeta", '\u{3b6}'),
    ("zwj", '\u{200d}'),
    ("zwnj", '\u{200c}'),
];

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
        assert_eq!(f.filter("<p>Hello <b>World</b></p>"), "\nHello World\n");
    }

    #[test]
    fn html_strip_self_closing_and_attributes() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(
            f.filter(r#"a<br/>b<img src="x>y" alt='q>r'/>c"#),
            "a\nbc"
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
            "a\nb\nc"
        );
    }

    #[test]
    fn html_strip_named_entities() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(
            f.filter("&lt;a&gt; &amp; &quot;b&quot; &#39;c&#39; d&nbsp;e"),
            "<a> & \"b\" 'c' d e"
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
        assert_eq!(f.filter("<p><b>bold</b> <i>it</i></p>"), "\n<b>bold</b> it\n");
    }

    #[test]
    fn html_strip_cdata() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("x<![CDATA[<raw> & stuff]]>y"), "x<raw> & stuffy");
    }

    #[test]
    fn html_strip_unicode_passthrough() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("<p>日本語 café</p>"), "\n日本語 café\n");
    }

    /// SIDE 012: block-level tags and `<br>` separate words (Lucene and
    /// the ES `html_strip` example); inline tags still join; entities from
    /// Lucene's table decode.
    #[test]
    fn html_strip_block_tags_separate_words() {
        let f = HtmlStripCharFilter::new();
        assert_eq!(f.filter("<p>I&apos;m so <b>happy</b>!</p>"), "\nI'm so happy!\n");
        assert_eq!(f.filter("<p>alpha</p><p>beta</p>"), "\nalpha\n\nbeta\n");
        assert_eq!(f.filter("alpha<br>beta"), "alpha\nbeta");
        assert_eq!(f.filter("<ul><li>one</li><li>two</li></ul>"), "\n\none\n\ntwo\n\n");
        assert_eq!(f.filter("al<b>pha</b><!-- x -->beta"), "alphabeta");
        assert_eq!(f.filter("caf&eacute; &Eacute;t&eacute; &AMP; &copy;"), "café Été & ©");
        let a = Analyzer::new(Box::new(StandardTokenizer)).with_char_filter(Box::new(HtmlStripCharFilter::new())).with_filter(Box::new(crate::analysis::filter::LowerCaseFilter));
        let words = |s: &str| a.analyze(s).into_iter().map(|t| t.text).collect::<Vec<_>>();
        assert_eq!(words("<p>alpha</p><p>beta</p>"), ["alpha", "beta"]);
        assert_eq!(words("alpha<br>beta<div>gamma</div>"), ["alpha", "beta", "gamma"]);
        assert_eq!(words("<span>al</span><span>pha</span>"), ["alpha"]);
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
