//! Certificate trust restrictions (`<ssl context>.trust_restrictions.path`
//! and `.trust_restrictions.x509_fields`), ES's `RestrictedTrustConfig` /
//! `RestrictedTrustManager`.
//!
//! After the normal chain verification, the peer's end-entity certificate
//! must carry a name that matches one of the patterns listed under
//! `trust.subject_name` in the restrictions file (YAML, e.g. ECK's
//! `trust.yml`). The names read are the `x509_fields`:
//! `subjectAltName.otherName.commonName` (the default: an `otherName` SAN
//! whose type-id is the CN OID 2.5.4.3, which ECK puts in every transport
//! certificate) and/or `subjectAltName.dnsName`. Patterns are ES
//! `Automatons` patterns: `*` and `?` wildcards; `/.../` regular expressions
//! are not supported and fail the configuration.
//!
//! The file is read when the SSL context is built. As in ES
//! (`RestrictedTrustConfig.getDependentFiles`), it is one of the context's
//! dependent files ([`crate::SslConfig::dependent_files`], the transport's
//! reloadable config), so editing it rebuilds the context with the new
//! names, as a rotated certificate does; a rebuilt context always carries
//! the restrictions.

use std::path::{Path, PathBuf};
use std::sync::Arc;

const CN_OID: &[u8] = &[0x55, 0x04, 0x03]; // 2.5.4.3
const SAN_OID: &[u8] = &[0x55, 0x1d, 0x11]; // 2.5.29.17

/// `trust_restrictions.x509_fields` values (ES `X509Field`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X509Field {
    SanOtherNameCommonName,
    SanDns,
}

impl X509Field {
    pub fn parse(value: &str, setting: &str) -> Result<X509Field, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "subjectaltname.othername.commonname" => Ok(X509Field::SanOtherNameCommonName),
            "subjectaltname.dnsname" => Ok(X509Field::SanDns),
            _ => Err(format!(
                "{value} is not a supported x509 field for trust restrictions. Recognised values are [subjectAltName.otherName.commonName, subjectAltName.dnsName] (setting [{setting}])"
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            X509Field::SanOtherNameCommonName => "subjectAltName.otherName.commonName",
            X509Field::SanDns => "subjectAltName.dnsName",
        }
    }
}

/// The parsed `trust_restrictions.*` settings of one SSL context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustRestrictionsConfig {
    pub path: PathBuf,
    pub fields: Vec<X509Field>,
}

impl TrustRestrictionsConfig {
    /// `<prefix>trust_restrictions.path` / `.x509_fields`; `None` without a
    /// path. Relative paths resolve against `conf_dir`.
    pub fn parse(prefix: &str, get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path) -> Result<Option<TrustRestrictionsConfig>, String> {
        let Some(path) = get(&format!("{prefix}trust_restrictions.path")).filter(|p| !p.is_empty()) else { return Ok(None) };
        let fields_key = format!("{prefix}trust_restrictions.x509_fields");
        let raw = get_list(&fields_key);
        let mut fields = Vec::new();
        for f in raw.iter().map(|f| f.trim()).filter(|f| !f.is_empty()) {
            let f = X509Field::parse(f, &fields_key)?;
            if !fields.contains(&f) {
                fields.push(f);
            }
        }
        if fields.is_empty() {
            fields.push(X509Field::SanOtherNameCommonName);
        }
        let path = PathBuf::from(path);
        let path = if path.is_absolute() { path } else { conf_dir.join(path) };
        Ok(Some(TrustRestrictionsConfig { path, fields }))
    }
}

/// Restrictions in force: the configuration and the trusted name patterns
/// read from the file when the context was built.
pub struct TrustRestrictions {
    config: TrustRestrictionsConfig,
    names: Vec<String>,
}

impl std::fmt::Debug for TrustRestrictions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "restrictedTrust=[{}]", self.config.path.display())
    }
}

impl TrustRestrictions {
    /// Read the restrictions file: an unreadable or invalid file fails the
    /// SSL configuration (at start, or the rebuild, which then keeps the
    /// previous context), as ES's `createTrustManager`.
    pub fn load(config: TrustRestrictionsConfig) -> Result<Arc<TrustRestrictions>, String> {
        let fail = |e: String| format!("failed to initialize TrustManager for restrictedTrust=[{}]: {e}", config.path.display());
        let content = std::fs::read(&config.path).map_err(|e| fail(e.to_string()))?;
        let names = parse_trusted_names(&String::from_utf8_lossy(&content)).map_err(fail)?;
        Ok(Arc::new(TrustRestrictions { config, names }))
    }

    /// The trusted name patterns.
    pub fn trusted_names(&self) -> &[String] {
        &self.names
    }

    /// ES `verifyTrust`: the end-entity certificate's names (per the
    /// configured fields) must match a trusted pattern.
    pub fn verify(&self, end_entity: &[u8]) -> Result<(), String> {
        let names = certificate_names(end_entity, &self.config.fields);
        let trusted = &self.names;
        if names.iter().any(|n| trusted.iter().any(|p| wildcard_match(p, n))) {
            return Ok(());
        }
        let fields: Vec<&str> = self.config.fields.iter().map(X509Field::name).collect();
        Err(format!("Certificate with fields [{}] with values {names:?} does not match the trusted names {trusted:?}", fields.join(", ")))
    }
}

/// `trust.subject_name` from the restrictions file (ES loads it as
/// settings: flat `trust.subject_name: [...]`, nested `trust:` /
/// `subject_name:`, flow or block lists, or a single value).
pub fn parse_trusted_names(text: &str) -> Result<Vec<String>, String> {
    let mut path: Vec<(usize, String)> = Vec::new();
    let mut names = Vec::new();
    let mut in_list = false;
    for raw in text.lines() {
        let line = strip_comment(raw);
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if let Some(item) = t.strip_prefix("- ").or_else(|| (t == "-").then_some("")) {
            if in_list {
                names.push(unquote(item.trim()));
                continue;
            }
            return Err(format!("unexpected list item [{t}]"));
        }
        let Some((k, v)) = split_key(t) else { return Err(format!("cannot parse line [{t}]")) };
        while path.last().is_some_and(|(i, _)| *i >= indent) {
            path.pop();
        }
        let full: String = path.iter().map(|(_, s)| s.as_str()).chain(std::iter::once(k)).collect::<Vec<_>>().join(".");
        in_list = false;
        let v = v.trim();
        if v.is_empty() {
            if full == "trust.subject_name" {
                in_list = true;
            }
            path.push((indent, k.to_string()));
            continue;
        }
        if full == "trust.subject_name" {
            if let Some(inner) = v.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
                names.extend(inner.split(',').map(|s| unquote(s.trim())).filter(|s| !s.is_empty()));
            } else {
                names.extend(unquote(v).split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
            }
        }
    }
    for n in &names {
        if n.starts_with('/') {
            return Err(format!("regular expression trust patterns are not supported: [{n}]"));
        }
    }
    Ok(names)
}

fn strip_comment(line: &str) -> &str {
    // A `#` starts a comment at the line start or after whitespace, outside quotes.
    let mut quote: Option<char> = None;
    let mut prev_space = true;
    for (i, c) in line.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '#' && prev_space => return &line[..i],
            None => {}
        }
        prev_space = c.is_whitespace();
    }
    line
}

fn split_key(t: &str) -> Option<(&str, &str)> {
    let i = t.find(':')?;
    let (k, rest) = (&t[..i], &t[i + 1..]);
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    Some((k.trim().trim_matches('"').trim_matches('\''), rest))
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\''))) {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// ES `Automatons` wildcard: `*` any run, `?` one character, `\` escapes.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    // Tokens: (literal char, is_wildcard_kind) with escapes resolved.
    #[derive(Clone, Copy, PartialEq)]
    enum Tok { Lit(char), Any, One }
    let mut toks = Vec::with_capacity(p.len());
    let mut i = 0;
    while i < p.len() {
        match p[i] {
            '\\' if i + 1 < p.len() => {
                toks.push(Tok::Lit(p[i + 1]));
                i += 2;
                continue;
            }
            '*' => toks.push(Tok::Any),
            '?' => toks.push(Tok::One),
            c => toks.push(Tok::Lit(c)),
        }
        i += 1;
    }
    let (mut ti, mut pi) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    while ti < t.len() {
        if pi < toks.len() && (toks[pi] == Tok::One || toks[pi] == Tok::Lit(t[ti])) {
            ti += 1;
            pi += 1;
        } else if pi < toks.len() && toks[pi] == Tok::Any {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < toks.len() && toks[pi] == Tok::Any {
        pi += 1;
    }
    pi == toks.len()
}

// ------------------------------------------------------------------ DER

/// One DER TLV: (tag, content, rest).
fn tlv(b: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, b) = b.split_first()?;
    let (&l0, mut b) = b.split_first()?;
    let len = if l0 < 0x80 {
        l0 as usize
    } else {
        let n = (l0 & 0x7f) as usize;
        if n == 0 || n > 4 || b.len() < n {
            return None;
        }
        let mut len = 0usize;
        for &x in &b[..n] {
            len = (len << 8) | x as usize;
        }
        b = &b[n..];
        len
    };
    if b.len() < len {
        return None;
    }
    Some((tag, &b[..len], &b[len..]))
}

/// The subjectAltName extension's GeneralNames content, if any.
fn san_extension(cert: &[u8]) -> Option<&[u8]> {
    let (_, certificate, _) = tlv(cert)?; // Certificate SEQUENCE
    let (_, tbs, _) = tlv(certificate)?; // tbsCertificate SEQUENCE
    let mut rest = tbs;
    while !rest.is_empty() {
        let (tag, content, next) = tlv(rest)?;
        rest = next;
        if tag == 0xa3 {
            // [3] EXPLICIT Extensions
            let (_, mut exts, _) = tlv(content)?;
            while !exts.is_empty() {
                let (_, ext, next) = tlv(exts)?;
                exts = next;
                let (_, oid, mut after) = tlv(ext)?;
                if oid != SAN_OID {
                    continue;
                }
                // Optional `critical BOOLEAN`, then the OCTET STRING.
                let (mut t, mut v, a) = tlv(after)?;
                if t == 0x01 {
                    after = a;
                    (t, v, _) = tlv(after)?;
                }
                if t != 0x04 {
                    return None;
                }
                let (_, names, _) = tlv(v)?; // GeneralNames SEQUENCE
                return Some(names);
            }
        }
    }
    None
}

/// The names of `fields` in a DER certificate (`readX509Certificate`).
pub fn certificate_names(cert: &[u8], fields: &[X509Field]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let Some(mut names) = san_extension(cert) else { return out };
    while let Some((tag, content, next)) = tlv(names) {
        names = next;
        match tag {
            // dNSName [2] IA5String (implicit)
            0x82 if fields.contains(&X509Field::SanDns) => {
                if let Ok(s) = std::str::from_utf8(content) {
                    out.push(s.to_string());
                }
            }
            // otherName [0] { type-id OID, value [0] EXPLICIT ANY }
            0xa0 if fields.contains(&X509Field::SanOtherNameCommonName) => {
                if let Some(cn) = other_name_cn(content) {
                    out.push(cn);
                }
            }
            _ => {}
        }
    }
    out.sort();
    out.dedup();
    out
}

fn other_name_cn(content: &[u8]) -> Option<String> {
    let (tag, oid, rest) = tlv(content)?;
    if tag != 0x06 || oid != CN_OID {
        return None;
    }
    let (_, value, _) = tlv(rest)?; // [0] EXPLICIT
    let (tag, mut inner, _) = tlv(value)?;
    // A constructed value wraps the string once more (`isConstructed`).
    let mut tag = tag;
    if tag & 0x20 != 0 {
        let (t, v, _) = tlv(inner)?;
        tag = t;
        inner = v;
    }
    match tag {
        // UTF8String, PrintableString, IA5String, T61String, VisibleString
        0x0c | 0x13 | 0x16 | 0x14 | 0x1a => std::str::from_utf8(inner).ok().map(str::to_string),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_names_file_forms() {
        assert_eq!(parse_trusted_names("trust.subject_name: [\"*.node.es.local\", b]\n").unwrap(), vec!["*.node.es.local", "b"]);
        assert_eq!(parse_trusted_names("trust:\n  subject_name:\n    - a.es.local\n    - \"b.es.local\"\n").unwrap(), vec!["a.es.local", "b.es.local"]);
        assert_eq!(parse_trusted_names("trust.subject_name:\n- x\n- y # comment\n").unwrap(), vec!["x", "y"]);
        assert_eq!(parse_trusted_names("trust.subject_name: one\n").unwrap(), vec!["one"]);
        assert!(parse_trusted_names("trust.subject_name: [\"/re.*/\"]\n").is_err());
        assert!(parse_trusted_names("# nothing\n").unwrap().is_empty());
    }

    #[test]
    fn wildcards() {
        assert!(wildcard_match("*.node.es.local", "es-0.node.es.local"));
        assert!(wildcard_match("es-?.node.*", "es-1.node.c.ns.es.local"));
        assert!(!wildcard_match("*.node.es.local", "es-0.node.other.local"));
        assert!(wildcard_match("exact", "exact"));
        assert!(!wildcard_match("exact", "exactly"));
        assert!(wildcard_match("a\\*b", "a*b"));
        assert!(!wildcard_match("a\\*b", "axb"));
    }

    #[test]
    fn restrictions_file_is_watched_with_the_context() {
        let get = |k: &str| -> Option<String> {
            match k {
                "x.trust_restrictions.path" => Some("trust.yml".into()),
                "x.certificate" => Some("n.crt".into()),
                "x.key" => Some("n.key".into()),
                "x.certificate_authorities" => Some("ca.crt".into()),
                _ => None,
            }
        };
        let list = |k: &str| -> Vec<String> { if k == "x.certificate_authorities" { vec!["ca.crt".into()] } else { vec![] } };
        let defaults = crate::Defaults { verification_mode: crate::VerificationMode::Certificate, client_authentication: crate::ClientAuth::Required };
        let c = crate::SslConfig::parse("x.", &get, &list, Path::new("/conf"), defaults).unwrap();
        let files = c.dependent_files();
        assert!(files.contains(&PathBuf::from("/conf/trust.yml")), "{files:?}");
        assert!(files.contains(&PathBuf::from("/conf/n.crt")) && files.contains(&PathBuf::from("/conf/ca.crt")));
    }

    #[test]
    fn fields_setting() {
        let get = |k: &str| -> Option<String> { (k == "x.trust_restrictions.path").then(|| "trust.yml".into()) };
        let none = |_: &str| -> Vec<String> { vec![] };
        let c = TrustRestrictionsConfig::parse("x.", &get, &none, Path::new("/conf")).unwrap().unwrap();
        assert_eq!(c.path, PathBuf::from("/conf/trust.yml"));
        assert_eq!(c.fields, vec![X509Field::SanOtherNameCommonName]);
        let both = |_: &str| -> Vec<String> { vec!["subjectAltName.dnsName".into(), "subjectAltName.otherName.commonName".into()] };
        assert_eq!(TrustRestrictionsConfig::parse("x.", &get, &both, Path::new("/conf")).unwrap().unwrap().fields, vec![X509Field::SanDns, X509Field::SanOtherNameCommonName]);
        let bad = |_: &str| -> Vec<String> { vec!["subject.cn".into()] };
        assert!(TrustRestrictionsConfig::parse("x.", &get, &bad, Path::new("/conf")).is_err());
    }
}
