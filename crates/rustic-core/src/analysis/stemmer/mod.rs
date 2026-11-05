//! Stemmers: Porter (1980), the Snowball family, Lucene's light/minimal
//! stemmers, and KStem.
//!
//! Each language module exposes `pub fn stem(word: &str) -> String`
//! operating on an already-lowercased word.

pub mod porter;

pub mod danish;
pub mod dutch;
pub mod english;
pub mod finnish;
pub mod french;
pub mod german;
pub mod hungarian;
pub mod italian;
pub mod kstem;
mod kstem_data;
pub mod light;
pub mod norwegian;
pub mod portuguese;
pub mod romanian;
pub mod russian;
pub mod spanish;
pub mod swedish;
pub mod turkish;

pub use porter::{porter_stem, snowball_porter_stem, StemFilter};

/// Full-vocabulary conformance tests against the official Snowball test data.
///
/// Source: `snowballstem/snowball-data` at commit `f08c4d63b9bc` (2023-11-09),
/// the data matching Snowball `34f3612e5e8c`, which is the Snowball revision
/// bundled by every released Lucene 10.x (and therefore Elasticsearch).
/// Each `testdata/snowball/<lang>.tsv.zst` is `voc.txt` and `output.txt`
/// pasted together (`word\tstem` per line), zstd-compressed; the complete
/// vocabularies are kept (no sampling).
#[cfg(test)]
pub(crate) mod vocab_tests {
    use std::path::Path;

    pub(crate) fn load(lang: &str) -> Vec<(String, String)> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/analysis/testdata/snowball")
            .join(format!("{lang}.tsv.zst"));
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let bytes = zstd::decode_all(&raw[..]).expect("zstd");
        let text = String::from_utf8(bytes).expect("utf-8");
        text.lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                let (w, s) = l.split_once('\t').expect("tab-separated pair");
                (w.to_string(), s.to_string())
            })
            .collect()
    }

    pub(crate) fn check(lang: &str, min_pairs: usize, stem: fn(&str) -> String) {
        let pairs = load(lang);
        assert!(pairs.len() >= min_pairs, "{lang}: only {} pairs", pairs.len());
        let mut failures = Vec::new();
        for (w, expected) in &pairs {
            let got = stem(w);
            if &got != expected {
                failures.push(format!("{w:?}: expected {expected:?}, got {got:?}"));
            }
        }
        if !failures.is_empty() {
            let pct = 100.0 * (pairs.len() - failures.len()) as f64 / pairs.len() as f64;
            panic!(
                "{lang}: {}/{} mismatches ({pct:.3}% pass). First 40:\n{}",
                failures.len(),
                pairs.len(),
                failures.iter().take(40).cloned().collect::<Vec<_>>().join("\n")
            );
        }
    }

    #[test]
    fn porter() {
        check("porter", 42_000, super::snowball_porter_stem);
    }

    /// Lucene's own `PorterStemmer` test data
    /// (`lucene/analysis/common/src/test/org/apache/lucene/analysis/en/porterTestData.zip`,
    /// all 23,531 pairs, no sampling) in `testdata/lucene/porter.tsv.zst`.
    #[test]
    fn lucene_porter() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/analysis/testdata/lucene/porter.tsv.zst");
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let text = String::from_utf8(zstd::decode_all(&raw[..]).expect("zstd")).expect("utf-8");
        let mut n = 0;
        let mut failures = Vec::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let (w, expected) = line.split_once('\t').expect("pair");
            n += 1;
            let got = super::porter_stem(w);
            if got != expected {
                failures.push(format!("{w:?}: expected {expected:?}, got {got:?}"));
            }
        }
        assert!(n >= 23_000, "only {n} pairs");
        assert!(failures.is_empty(), "{}/{n} mismatches:\n{}", failures.len(), failures[..failures.len().min(40)].join("\n"));
    }
    #[test]
    fn english() {
        check("english", 42_000, super::english::stem);
    }
    #[test]
    fn german() {
        check("german", 35_000, super::german::stem);
    }
    #[test]
    fn french() {
        check("french", 20_000, super::french::stem);
    }
    #[test]
    fn spanish() {
        check("spanish", 28_000, super::spanish::stem);
    }
    #[test]
    fn italian() {
        check("italian", 35_000, super::italian::stem);
    }
    #[test]
    fn portuguese() {
        check("portuguese", 32_000, super::portuguese::stem);
    }
    #[test]
    fn dutch() {
        check("dutch", 45_000, super::dutch::stem);
    }
    #[test]
    fn swedish() {
        check("swedish", 30_000, super::swedish::stem);
    }
    #[test]
    fn norwegian() {
        check("norwegian", 20_000, super::norwegian::stem);
    }
    #[test]
    fn danish() {
        check("danish", 23_000, super::danish::stem);
    }
    #[test]
    fn finnish() {
        check("finnish", 49_000, super::finnish::stem);
    }
    #[test]
    fn russian() {
        check("russian", 49_000, super::russian::stem);
    }
    #[test]
    fn hungarian() {
        check("hungarian", 30_000, super::hungarian::stem);
    }
    #[test]
    fn romanian() {
        check("romanian", 26_000, super::romanian::stem);
    }
    #[test]
    fn turkish() {
        check("turkish", 96_000, super::turkish::stem);
    }
}
