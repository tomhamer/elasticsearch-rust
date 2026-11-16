//! `version` field encoding (ES `VersionEncoder`): semantic versions map to
//! strings whose byte order is semver precedence. Numeric components carry
//! a length prefix (so `1.10` sorts after `1.9`), a release sorts after its
//! pre-releases and before longer versions (`1.2.3-a` < `1.2.3` < `1.2.3.4`),
//! numeric pre-release identifiers sort before alphanumeric ones, and build
//! metadata is kept (`+build`, ASCII order, after the same version without
//! it), so the encoding is reversible (production review #444). Leading
//! zeros make a version invalid (`01.2`, `1.0.0-01`), as in ES. Strings that
//! are not valid versions sort after every valid one, in byte order.

fn enc_num(digits: &str, out: &mut String) {
    let d = digits.trim_start_matches('0');
    let d = if d.is_empty() { "0" } else { d };
    // Length prefix as a single printable char ('0' + len).
    out.push((b'0' + d.len().min(70) as u8) as char);
    out.push_str(d);
}

/// Is `s` a valid (relaxed semver) version: `N(.N)*(-pre)?(+build)?`.
pub fn is_valid(s: &str) -> bool {
    parse(s).is_some()
}

struct Parsed<'a> {
    main: Vec<&'a str>,
    pre: Option<Vec<&'a str>>,
    build: Option<&'a str>,
}

/// ES `VersionEncoder.LEGAL_VERSION_PATTERN` numbers: `0|[1-9]\d*`.
fn canonical_number(c: &str) -> bool {
    !c.is_empty() && c.chars().all(|c| c.is_ascii_digit()) && (c == "0" || !c.starts_with('0'))
}

fn parse(s: &str) -> Option<Parsed<'_>> {
    let core = s.split_once('+').map(|(c, b)| (c, Some(b))).unwrap_or((s, None));
    if let Some(b) = core.1 {
        if b.is_empty() || !b.split('.').all(|id| !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')) {
            return None;
        }
    }
    let (main, pre) = match core.0.split_once('-') {
        Some((m, p)) => (m, Some(p)),
        None => (core.0, None),
    };
    let main: Vec<&str> = main.split('.').collect();
    if main.is_empty() || !main.iter().all(|c| canonical_number(c)) {
        return None;
    }
    let pre = match pre {
        Some(p) => {
            let ids: Vec<&str> = p.split('.').collect();
            // A numeric identifier has no leading zero; others have a
            // letter or `-` (`\d*[a-zA-Z-][a-zA-Z0-9-]*`).
            if ids.iter().any(|id| {
                id.is_empty()
                    || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
                    || (id.chars().all(|c| c.is_ascii_digit()) && !canonical_number(id))
            }) {
                return None;
            }
            Some(ids)
        }
        None => None,
    };
    Some(Parsed { main, pre, build: core.1 })
}

/// Sortable encoding of a version string.
pub fn encode(s: &str) -> String {
    let Some(p) = parse(s) else {
        return format!("2{s}");
    };
    let mut out = String::from("1");
    for (i, c) in p.main.iter().enumerate() {
        if i > 0 {
            out.push('.');
        }
        enc_num(c, &mut out);
    }
    match p.pre {
        None => out.push('-'),
        Some(ids) => {
            out.push(',');
            for (i, id) in ids.iter().enumerate() {
                if i > 0 {
                    out.push('.');
                }
                if id.chars().all(|c| c.is_ascii_digit()) {
                    out.push('1');
                    enc_num(id, &mut out);
                } else {
                    out.push('2');
                    out.push_str(id);
                }
            }
        }
    }
    if let Some(b) = p.build {
        out.push('+');
        out.push_str(b);
    }
    out
}

/// The version string of an [`encode`]d value (the encoding is reversible).
pub fn decode(e: &str) -> String {
    if let Some(raw) = e.strip_prefix('2') {
        return raw.to_string();
    }
    let Some(body) = e.strip_prefix('1') else { return e.to_string() };
    let b = body.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    // Length-prefixed numbers: `<'0'+len><digits>`.
    let num = |i: &mut usize, out: &mut String| {
        if *i < b.len() {
            let n = (b[*i] - b'0') as usize;
            *i += 1;
            let end = (*i + n).min(b.len());
            out.push_str(&body[*i..end]);
            *i = end;
        }
    };
    while i < b.len() {
        match b[i] {
            b'.' => {
                out.push('.');
                i += 1;
            }
            b'-' => {
                if let Some(build) = body[i + 1..].strip_prefix('+') {
                    out.push('+');
                    out.push_str(build);
                }
                return out;
            }
            b',' => {
                i += 1;
                out.push('-');
                let mut first = true;
                while i < b.len() {
                    if !first {
                        if b[i] != b'.' {
                            break;
                        }
                        out.push('.');
                        i += 1;
                    }
                    first = false;
                    match b.get(i) {
                        Some(b'1') => {
                            i += 1;
                            num(&mut i, &mut out);
                        }
                        Some(b'2') => {
                            i += 1;
                            let end = body[i..].find(['.', '+']).map(|p| i + p).unwrap_or(b.len());
                            out.push_str(&body[i..end]);
                            i = end;
                        }
                        _ => break,
                    }
                }
                if let Some(build) = body[i.min(b.len())..].strip_prefix('+') {
                    out.push('+');
                    out.push_str(build);
                }
                return out;
            }
            _ => num(&mut i, &mut out),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence() {
        let ordered = [
            "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0",
            "1.0.0.1", "1.2", "1.2.0", "1.9.0", "1.10.0", "2.0.0", "11.0.0", "a-invalid", "zzz",
        ];
        for w in ordered.windows(2) {
            assert!(encode(w[0]) < encode(w[1]), "{} < {}", w[0], w[1]);
        }
        // Production review #444: build metadata is kept (ASCII order after
        // the bare version), leading zeros are invalid strings.
        assert!(encode("1.0.0") < encode("1.0.0+build.1"));
        assert!(encode("1.0.0+build.1") < encode("1.0.0+build.2"));
        assert!(encode("1.0.0+build.2") < encode("1.0.0.1"));
        assert!(encode("1.0.0-rc.1+b") < encode("1.0.0-rc.1.x"));
        assert!(!is_valid("01.2") && !is_valid("1.0.0-01"));
        assert!(encode("01.2") > encode("99.0"));
        for v in ["1.2.3+build.1", "1.2.3-rc.1+b5", "1.2.3-alpha+exp.sha.5114f85", "01.2", "1.0.0-01", "0.0.1"] {
            assert_eq!(decode(&encode(v)), v);
        }
        for v in ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-beta.11", "1.10.0", "11.0.0", "2.0.0-rc.1-x", "a-invalid", "1.el6"] {
            assert_eq!(decode(&encode(v)), v);
        }
        assert!(is_valid("1.2.3-rc.1+b5"));
        assert!(!is_valid("1..2"));
        assert!(!is_valid("v1.2"));
    }
}
