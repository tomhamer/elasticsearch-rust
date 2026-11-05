//! CJK bigram filter (Lucene `CJKBigramFilter`).

use crate::analysis::{Token, TokenFilter};

/// Which CJK scripts take part in bigramming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CjkScript {
    /// Han ideographs.
    Han,
    /// Japanese hiragana.
    Hiragana,
    /// Japanese katakana.
    Katakana,
    /// Korean hangul.
    Hangul,
}

/// Classify a single-character token.
pub fn cjk_script(c: char) -> Option<CjkScript> {
    match c {
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2EBEF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{3005}'
        | '\u{3007}' => Some(CjkScript::Han),
        '\u{3040}'..='\u{309F}' => Some(CjkScript::Hiragana),
        '\u{30A0}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{FF66}'..='\u{FF9F}' => {
            Some(CjkScript::Katakana)
        }
        '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}' => {
            Some(CjkScript::Hangul)
        }
        _ => None,
    }
}

/// Forms bigrams out of runs of consecutive single-character CJK tokens
/// (as emitted by the standard tokenizer, which gives each ideograph its
/// own token).
///
/// `你好世界` (4 tokens) → `你好`, `好世`, `世界`. A lone CJK character
/// (a run of length 1) is emitted as a unigram. Non-CJK tokens and
/// tokens of ignored scripts pass through unchanged. With
/// `output_unigrams`, every CJK character is also emitted on its own
/// (bigrams share the position of their first character, as in Lucene).
///
/// Bigram position = position of the first character; offsets span both
/// characters. A run is broken by a gap in positions (e.g. a removed
/// stop word) or by a non-CJK token.
#[derive(Clone, Debug)]
pub struct CjkBigramFilter {
    /// Scripts that are bigrammed (ES `ignored_scripts` is the complement).
    pub scripts: Vec<CjkScript>,
    /// Also emit the single characters (ES `output_unigrams`, default false).
    pub output_unigrams: bool,
}

impl Default for CjkBigramFilter {
    fn default() -> Self {
        Self {
            scripts: vec![CjkScript::Han, CjkScript::Hiragana, CjkScript::Katakana, CjkScript::Hangul],
            output_unigrams: false,
        }
    }
}

impl CjkBigramFilter {
    /// Build from ES `ignored_scripts` names (`"han"`, `"hiragana"`,
    /// `"katakana"`, `"hangul"`).
    pub fn with_ignored_scripts(ignored: &[&str], output_unigrams: bool) -> Self {
        let all = [
            ("han", CjkScript::Han),
            ("hiragana", CjkScript::Hiragana),
            ("katakana", CjkScript::Katakana),
            ("hangul", CjkScript::Hangul),
        ];
        Self {
            scripts: all
                .iter()
                .filter(|(n, _)| !ignored.iter().any(|i| i.eq_ignore_ascii_case(n)))
                .map(|(_, s)| *s)
                .collect(),
            output_unigrams,
        }
    }

    /// The standard tokenizer keeps Hangul and Katakana runs whole (UAX#29);
    /// split such a token into one token per character so it bigrams like
    /// a run of ideographs. Later positions shift to make room.
    fn explode_runs(&self, tokens: Vec<Token>) -> Vec<Token> {
        let needs = tokens.iter().any(|t| {
            let mut cs = t.text.chars();
            cs.next().is_some() && cs.next().is_some()
        });
        if !needs {
            return tokens;
        }
        let mut out = Vec::with_capacity(tokens.len() + 8);
        let mut shift = 0u32;
        for t in tokens {
            let script = t.text.chars().next().and_then(cjk_script);
            let run = t.text.chars().count() > 1
                && matches!(script, Some(CjkScript::Hangul | CjkScript::Katakana))
                && script.is_some_and(|s| self.scripts.contains(&s))
                && t.text.chars().all(|c| cjk_script(c) == script);
            if !run {
                out.push(Token { position: t.position + shift, ..t });
                continue;
            }
            let mut off = t.offset_from;
            for (k, c) in t.text.chars().enumerate() {
                let len = c.len_utf8() as u32;
                out.push(Token { text: c.to_string(), position: t.position + shift + k as u32, offset_from: off, offset_to: off + len });
                off += len;
            }
            shift += t.text.chars().count() as u32 - 1;
        }
        out
    }

    fn is_bigrammable(&self, t: &Token) -> bool {
        let mut it = t.text.chars();
        match (it.next(), it.next()) {
            (Some(c), None) => cjk_script(c).map(|s| self.scripts.contains(&s)).unwrap_or(false),
            _ => false,
        }
    }
}

impl TokenFilter for CjkBigramFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let tokens = self.explode_runs(tokens);
        let mut out = Vec::with_capacity(tokens.len());
        let n = tokens.len();
        let mut i = 0;
        while i < n {
            if !self.is_bigrammable(&tokens[i]) {
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            // Collect a run of consecutive bigrammable tokens.
            let start = i;
            let mut end = i + 1;
            while end < n
                && self.is_bigrammable(&tokens[end])
                && tokens[end].position == tokens[end - 1].position + 1
            {
                end += 1;
            }
            let run = &tokens[start..end];
            if run.len() == 1 {
                out.push(run[0].clone());
            } else {
                for (k, t) in run.iter().enumerate() {
                    if self.output_unigrams {
                        out.push(t.clone());
                    }
                    if k + 1 < run.len() {
                        let next = &run[k + 1];
                        out.push(Token {
                            text: format!("{}{}", t.text, next.text),
                            position: t.position,
                            offset_from: t.offset_from,
                            offset_to: next.offset_to,
                        });
                    }
                }
            }
            i = end;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::tokenizer::StandardTokenizer;
    use crate::analysis::Tokenizer;

    fn texts(t: &[Token]) -> Vec<&str> {
        t.iter().map(|t| t.text.as_str()).collect()
    }
    fn positions(t: &[Token]) -> Vec<u32> {
        t.iter().map(|t| t.position).collect()
    }

    #[test]
    fn bigrams_basic() {
        let toks = StandardTokenizer.tokenize("你好世界");
        let out = CjkBigramFilter::default().filter(toks);
        assert_eq!(texts(&out), vec!["你好", "好世", "世界"]);
        assert_eq!(positions(&out), vec![0, 1, 2]);
        assert_eq!((out[0].offset_from, out[0].offset_to), (0, 6));
        assert_eq!((out[2].offset_from, out[2].offset_to), (6, 12));
    }

    #[test]
    fn bigrams_with_unigrams() {
        let toks = StandardTokenizer.tokenize("日本語");
        let out = CjkBigramFilter { output_unigrams: true, ..Default::default() }.filter(toks);
        assert_eq!(texts(&out), vec!["日", "日本", "本", "本語", "語"]);
        assert_eq!(positions(&out), vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn lone_cjk_char_is_unigram() {
        let toks = StandardTokenizer.tokenize("a 中 b");
        let out = CjkBigramFilter::default().filter(toks);
        assert_eq!(texts(&out), vec!["a", "中", "b"]);
    }

    #[test]
    fn mixed_scripts_and_latin() {
        let toks = StandardTokenizer.tokenize("多くの学生 test カタカナ");
        let out = CjkBigramFilter::default().filter(toks);
        assert_eq!(
            texts(&out),
            vec!["多く", "くの", "の学", "学生", "test", "カタ", "タカ", "カナ"]
        );
    }

    #[test]
    fn ignored_script_passes_through() {
        let f = CjkBigramFilter::with_ignored_scripts(&["hiragana"], false);
        let toks = StandardTokenizer.tokenize("漢字かな");
        let out = f.filter(toks);
        assert_eq!(texts(&out), vec!["漢字", "か", "な"]);
    }

    #[test]
    fn position_gap_breaks_run() {
        let mut toks = StandardTokenizer.tokenize("你好世界");
        toks.remove(2); // simulate a removed token: 你(0) 好(1) 界(3)
        let out = CjkBigramFilter::default().filter(toks);
        assert_eq!(texts(&out), vec!["你好", "界"]);
    }

    #[test]
    fn hangul_bigrams() {
        let toks = StandardTokenizer.tokenize("한국어");
        let out = CjkBigramFilter::default().filter(toks);
        assert_eq!(texts(&out), vec!["한국", "국어"]);
    }

    #[test]
    fn empty() {
        assert!(CjkBigramFilter::default().filter(vec![]).is_empty());
    }
}
