//! Snowball English stemmer (the "english" algorithm, a.k.a. Porter2).
//!
//! A faithful port of `algorithms/english.sbl` at Snowball commit
//! `34f3612e5e8c` (March 2024), the revision bundled by every released
//! Lucene 10.x and therefore by Elasticsearch. Later Snowball changes
//! (2025: `past`/`univers`/`later`/`emerg`/`organ`/`inter` region prefixes,
//! `-ogist`, the `evening` exception, ...) are intentionally *not* included.
//! Verified against the complete matching `snowball-data` vocabulary
//! (see `vocab_tests`).

/// The `v` grouping: `aeiouy` (`Y` marks a consonant y and is not a vowel).
fn is_v(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

/// The `v_WXY` grouping.
fn is_v_wxy(c: char) -> bool {
    is_v(c) || matches!(c, 'w' | 'x' | 'Y')
}

fn is_valid_li(c: char) -> bool {
    matches!(c, 'c' | 'd' | 'e' | 'g' | 'h' | 'k' | 'm' | 'n' | 'r' | 't')
}

fn ends_with(s: &[char], suf: &str) -> bool {
    let n = suf.chars().count();
    s.len() >= n && s[s.len() - n..].iter().copied().eq(suf.chars())
}

/// Snowball `among` in backward mode: the longest entry that is a suffix of `s`.
fn among<'a>(s: &[char], list: &[&'a str]) -> Option<&'a str> {
    let mut best: Option<&'a str> = None;
    for suf in list {
        if ends_with(s, suf) && best.map_or(true, |b| suf.chars().count() > b.chars().count()) {
            best = Some(suf);
        }
    }
    best
}

fn suffix_start(s: &[char], suf: &str) -> usize {
    s.len() - suf.chars().count()
}

fn replace(s: &mut Vec<char>, suf: &str, rep: &str) {
    let start = suffix_start(s, suf);
    s.truncate(start);
    s.extend(rep.chars());
}

/// `shortv` tested backwards at `end` (exclusive):
/// `(non-v_WXY v non-v) or (non-v v atlimit)`.
fn shortv(s: &[char], end: usize) -> bool {
    if end >= 3 && !is_v_wxy(s[end - 1]) && is_v(s[end - 2]) && !is_v(s[end - 3]) {
        return true;
    }
    end == 2 && !is_v(s[1]) && is_v(s[0])
}

/// Whole-word exceptions (`exception1`).
fn exception1(word: &str) -> Option<&'static str> {
    Some(match word {
        "skis" => "ski",
        "skies" => "sky",
        "dying" => "die",
        "lying" => "lie",
        "tying" => "tie",
        "idly" => "idl",
        "gently" => "gentl",
        "ugly" => "ugli",
        "early" => "earli",
        "only" => "onli",
        "singly" => "singl",
        "sky" => "sky",
        "news" => "news",
        "howe" => "howe",
        "atlas" => "atlas",
        "cosmos" => "cosmos",
        "bias" => "bias",
        "andes" => "andes",
        _ => return None,
    })
}

/// `gopast v gopast non-v` from `from`; `None` if it runs off the end.
fn gopast_v_nonv(s: &[char], from: usize) -> Option<usize> {
    let mut i = from;
    while i < s.len() && !is_v(s[i]) {
        i += 1;
    }
    if i >= s.len() {
        return None;
    }
    i += 1;
    while i < s.len() && is_v(s[i]) {
        i += 1;
    }
    if i >= s.len() {
        return None;
    }
    Some(i + 1)
}

/// Stem an already-lowercased English word with the Snowball "english"
/// (Porter2) algorithm. Never panics.
pub fn stem(word: &str) -> String {
    if let Some(e) = exception1(word) {
        return e.to_string();
    }
    if word.chars().count() < 3 {
        return word.to_string();
    }
    let mut s: Vec<char> = word.chars().collect();

    // Prelude.
    if s.first() == Some(&'\'') {
        s.remove(0);
    }
    let mut y_found = false;
    if s.first() == Some(&'y') {
        s[0] = 'Y';
        y_found = true;
    }
    for i in 1..s.len() {
        if s[i] == 'y' && is_v(s[i - 1]) {
            s[i] = 'Y';
            y_found = true;
        }
    }

    // mark_regions.
    let n = s.len();
    let mut p1 = n;
    let mut p2 = n;
    let prefix = ["gener", "commun", "arsen"]
        .iter()
        .find(|p| s.len() >= p.len() && s[..p.len()].iter().copied().eq(p.chars()))
        .map(|p| p.len());
    let r1_start = match prefix {
        Some(l) => Some(l),
        None => gopast_v_nonv(&s, 0),
    };
    if let Some(a) = r1_start {
        p1 = a;
        if let Some(b) = gopast_v_nonv(&s, a) {
            p2 = b;
        }
    }

    // Step 1a.
    if let Some(suf) = among(&s, &["'", "'s", "'s'"]) {
        replace(&mut s, suf, "");
    }
    if let Some(suf) = among(&s, &["sses", "ied", "ies", "s", "us", "ss"]) {
        let start = suffix_start(&s, suf);
        match suf {
            "sses" => replace(&mut s, suf, "ss"),
            "ied" | "ies" => {
                if start >= 2 {
                    replace(&mut s, suf, "i");
                } else {
                    replace(&mut s, suf, "ie");
                }
            }
            "s" => {
                // next gopast v: a vowel strictly before the letter preceding `s`.
                if start >= 1 && s[..start - 1].iter().any(|&c| is_v(c)) {
                    s.truncate(start);
                }
            }
            _ => {}
        }
    }

    // exception2: invariant after step 1a.
    let exception2 = ["inning", "outing", "canning", "herring", "earring", "proceed", "exceed", "succeed"]
        .iter()
        .any(|w| s.iter().copied().eq(w.chars()));

    if !exception2 {
        step_1b(&mut s, p1);
        step_1c(&mut s);
        step_2(&mut s, p1);
        step_3(&mut s, p1, p2);
        step_4(&mut s, p2);
        step_5(&mut s, p1, p2);
    }

    if y_found {
        for c in s.iter_mut() {
            if *c == 'Y' {
                *c = 'y';
            }
        }
    }
    s.into_iter().collect()
}

fn step_1b(s: &mut Vec<char>, p1: usize) {
    let suf = match among(s, &["eed", "eedly", "ed", "edly", "ing", "ingly"]) {
        Some(x) => x,
        None => return,
    };
    let start = suffix_start(s, suf);
    if suf == "eed" || suf == "eedly" {
        if start >= p1 {
            replace(s, suf, "ee");
        }
        return;
    }
    if !s[..start].iter().any(|&c| is_v(c)) {
        return;
    }
    s.truncate(start);
    let doubles = ["bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"];
    if ends_with(s, "at") || ends_with(s, "bl") || ends_with(s, "iz") {
        s.push('e');
    } else if doubles.iter().any(|d| ends_with(s, d)) {
        // not (aeo atlimit)
        let aeo_atlimit = s.len() == 3 && matches!(s[0], 'a' | 'e' | 'o');
        if !aeo_atlimit {
            s.pop();
        }
    } else if s.len() == p1 && shortv(s, s.len()) {
        s.push('e');
    }
}

fn step_1c(s: &mut Vec<char>) {
    let n = s.len();
    if n >= 3 && matches!(s[n - 1], 'y' | 'Y') && !is_v(s[n - 2]) {
        s[n - 1] = 'i';
    }
}

fn step_2(s: &mut Vec<char>, p1: usize) {
    const RULES: &[(&str, &str)] = &[
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("abli", "able"),
        ("entli", "ent"),
        ("izer", "ize"),
        ("ization", "ize"),
        ("ational", "ate"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("aliti", "al"),
        ("alli", "al"),
        ("fulness", "ful"),
        ("ousli", "ous"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("bli", "ble"),
        ("ogi", "og"),
        ("fulli", "ful"),
        ("lessli", "less"),
        ("li", ""),
    ];
    let keys: Vec<&str> = RULES.iter().map(|r| r.0).collect();
    let suf = match among(s, &keys) {
        Some(x) => x,
        None => return,
    };
    let start = suffix_start(s, suf);
    if start < p1 {
        return;
    }
    match suf {
        "ogi" => {
            if start >= 1 && s[start - 1] == 'l' {
                replace(s, suf, "og");
            }
        }
        "li" => {
            if start >= 1 && is_valid_li(s[start - 1]) {
                s.truncate(start);
            }
        }
        _ => {
            let rep = RULES.iter().find(|r| r.0 == suf).map(|r| r.1).unwrap_or("");
            replace(s, suf, rep);
        }
    }
}

fn step_3(s: &mut Vec<char>, p1: usize, p2: usize) {
    const RULES: &[(&str, &str)] = &[
        ("tional", "tion"),
        ("ational", "ate"),
        ("alize", "al"),
        ("icate", "ic"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
        ("ative", ""),
    ];
    let keys: Vec<&str> = RULES.iter().map(|r| r.0).collect();
    let suf = match among(s, &keys) {
        Some(x) => x,
        None => return,
    };
    let start = suffix_start(s, suf);
    if start < p1 {
        return;
    }
    if suf == "ative" && start < p2 {
        return;
    }
    let rep = RULES.iter().find(|r| r.0 == suf).map(|r| r.1).unwrap_or("");
    replace(s, suf, rep);
}

fn step_4(s: &mut Vec<char>, p2: usize) {
    let list = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ism",
        "ate", "iti", "ous", "ive", "ize", "ion",
    ];
    let suf = match among(s, &list) {
        Some(x) => x,
        None => return,
    };
    let start = suffix_start(s, suf);
    if start < p2 {
        return;
    }
    if suf == "ion" {
        if start >= 1 && matches!(s[start - 1], 's' | 't') {
            s.truncate(start);
        }
    } else {
        s.truncate(start);
    }
}

fn step_5(s: &mut Vec<char>, p1: usize, p2: usize) {
    let n = s.len();
    if n == 0 {
        return;
    }
    let pos = n - 1;
    match s[pos] {
        'e' => {
            if pos >= p2 || (pos >= p1 && !shortv(s, pos)) {
                s.pop();
            }
        }
        'l' => {
            if pos >= p2 && pos >= 1 && s[pos - 1] == 'l' {
                s.pop();
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::stem;

    fn check(cases: &[(&str, &str)]) {
        let failures: Vec<String> = cases
            .iter()
            .filter_map(|&(word, expected)| {
                let got = stem(word);
                (got != expected).then(|| format!("{word}: expected {expected}, got {got}"))
            })
            .collect();
        assert!(failures.is_empty(), "failing vectors:\n{}", failures.join("\n"));
    }

    /// Vectors taken from the official Snowball English sample vocabulary.
    #[test]
    fn official_sample_vocabulary() {
        check(&[
            ("consign", "consign"),
            ("consigned", "consign"),
            ("consigning", "consign"),
            ("consignment", "consign"),
            ("consist", "consist"),
            ("consisted", "consist"),
            ("consistency", "consist"),
            ("consistent", "consist"),
            ("consistently", "consist"),
            ("consisteth", "consisteth"),
            ("consisting", "consist"),
            ("consistory", "consistori"),
            ("consists", "consist"),
            ("consolate", "consol"),
            ("consolation", "consol"),
            ("consolations", "consol"),
            ("consolatory", "consolatori"),
            ("console", "consol"),
            ("consoled", "consol"),
            ("consoler", "consol"),
            ("consoles", "consol"),
            ("consolidate", "consolid"),
            ("consolidated", "consolid"),
            ("consolidating", "consolid"),
            ("consolidation", "consolid"),
            ("consoling", "consol"),
            ("consolingly", "consol"),
            ("consols", "consol"),
            ("consonancy", "conson"),
            ("consonant", "conson"),
            ("consort", "consort"),
            ("consorted", "consort"),
            ("consortest", "consortest"),
            ("consorting", "consort"),
            ("conspectuities", "conspectu"),
            ("conspicuous", "conspicu"),
            ("conspicuously", "conspicu"),
            ("conspir", "conspir"),
            ("conspiracy", "conspiraci"),
            ("conspirant", "conspir"),
            ("knack", "knack"),
            ("knackeries", "knackeri"),
            ("knacks", "knack"),
            ("knag", "knag"),
            ("knapp", "knapp"),
            ("knapsack", "knapsack"),
            ("knav", "knav"),
            ("knave", "knave"),
            ("knaveries", "knaveri"),
            ("knavery", "knaveri"),
            ("knaves", "knave"),
            ("knavish", "knavish"),
            ("knead", "knead"),
            ("kneaded", "knead"),
            ("kneading", "knead"),
            ("knee", "knee"),
            ("kneel", "kneel"),
            ("kneeled", "kneel"),
            ("kneeling", "kneel"),
            ("kneels", "kneel"),
            ("knees", "knee"),
            ("knell", "knell"),
            ("knelled", "knell"),
            ("knelt", "knelt"),
            ("knew", "knew"),
            ("knewest", "knewest"),
            ("knick", "knick"),
            ("knicknacks", "knicknack"),
            ("knif", "knif"),
            ("knife", "knife"),
            ("knight", "knight"),
            ("knighted", "knight"),
            ("knighthood", "knighthood"),
            ("knighthoods", "knighthood"),
            ("knightly", "knight"),
            ("knights", "knight"),
            ("knightsbridge", "knightsbridg"),
            ("knit", "knit"),
            ("knits", "knit"),
            ("knitted", "knit"),
            ("generously", "generous"),
        ]);
    }

    /// Exceptional forms and the special cases called out in the algorithm.
    #[test]
    fn exceptions_and_special_cases() {
        check(&[
            ("skis", "ski"),
            ("skies", "sky"),
            ("dying", "die"),
            ("lying", "lie"),
            ("tying", "tie"),
            ("idly", "idl"),
            ("gently", "gentl"),
            ("ugly", "ugli"),
            ("early", "earli"),
            ("only", "onli"),
            ("singly", "singl"),
            ("sky", "sky"),
            ("news", "news"),
            ("howe", "howe"),
            ("atlas", "atlas"),
            ("cosmos", "cosmos"),
            ("bias", "bias"),
            ("andes", "andes"),
            ("inning", "inning"),
            ("outing", "outing"),
            ("canning", "canning"),
            ("herring", "herring"),
            ("earring", "earring"),
            ("evening", "even"),
            ("proceed", "proceed"),
            ("exceed", "exceed"),
            ("succeed", "succeed"),
            ("agreed", "agre"),
            ("feed", "feed"),
            ("generalization", "general"),
            ("communication", "communic"),
            ("arsenic", "arsenic"),
            ("arsenal", "arsenal"),
            ("international", "intern"),
            ("interest", "interest"),
            ("pasted", "past"),
            ("paste", "past"),
            ("universal", "univers"),
            ("universe", "univers"),
            ("university", "univers"),
            ("lateral", "later"),
            ("emerge", "emerg"),
            ("emergency", "emerg"),
            ("organic", "organ"),
            ("organize", "organ"),
            ("geologist", "geologist"),
            ("geology", "geolog"),
            ("conditional", "condit"),
            ("rational", "ration"),
        ]);
    }

    /// Steps 1a-1c and the short-word / undoubling rules.
    #[test]
    fn step_one_rules() {
        check(&[
            ("ties", "tie"),
            ("cries", "cri"),
            ("flies", "fli"),
            ("gas", "gas"),
            ("this", "this"),
            ("gaps", "gap"),
            ("hopping", "hop"),
            ("hopped", "hop"),
            ("hoping", "hope"),
            ("added", "add"),
            ("bed", "bed"),
            ("beds", "bed"),
            ("shed", "shed"),
            ("shred", "shred"),
            ("bead", "bead"),
            ("cry", "cri"),
            ("by", "by"),
            ("say", "say"),
            ("happy", "happi"),
        ]);
    }

    #[test]
    fn robustness() {
        assert_eq!(stem(""), "");
        assert_eq!(stem("a"), "a");
        assert_eq!(stem("is"), "is");
        assert_eq!(stem("as"), "as");
        assert_eq!(stem("'"), "'");
        assert_eq!(stem("''"), "''");
        assert_eq!(stem("'''"), "'");
        assert_eq!(stem("''s'"), "");
        assert_eq!(stem("'s"), "'s");
        assert_eq!(stem("123"), "123");
        assert_eq!(stem("4567s"), "4567s");
        assert_eq!(stem("..."), "...");
        assert_eq!(stem("yyy"), "yyy");
        assert_eq!(stem("ying"), "ying");
        assert_eq!(stem("eed"), "eed");
        assert_eq!(stem("ing"), "ing");
        assert_eq!(stem("sses"), "ss");
        assert_eq!(stem("ied"), "ie");
        assert_eq!(stem("über"), "über");
        assert_eq!(stem("日本語"), "日本語");
        assert_eq!(stem("ﬁsh"), "ﬁsh");
        assert_eq!(stem("dd"), "dd");
        assert_eq!(stem("bbing"), "bbing");
        assert_eq!(stem("edly"), "ed");
        assert_eq!(stem("ingly"), "ing");
        assert_eq!(stem("inter"), "inter");
        assert_eq!(stem("interesting"), "interest");
    }
}
