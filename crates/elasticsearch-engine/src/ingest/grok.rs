//! Grok: named regex macros (`%{SYNTAX:SEMANTIC:TYPE}`) compiled to one
//! `regex::Regex`. The bank is the ES legacy (non-ECS) pattern set,
//! rewritten for the `regex` crate (no lookaround / atomic groups: those
//! only guard boundaries and are dropped or turned into plain groups).

use crate::error::EsError;
use regex::Regex;
use serde_json::{Map, Number, Value};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub const PATTERNS: &[(&str, &str)] = &[
    ("USERNAME", r"[a-zA-Z0-9._-]+"),
    ("USER", r"%{USERNAME}"),
    ("EMAILLOCALPART", r"[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~]{1,64}(?:\.[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~]{1,62}){0,63}"),
    ("EMAILADDRESS", r"%{EMAILLOCALPART}@%{HOSTNAME}"),
    ("INT", r"(?:[+-]?(?:[0-9]+))"),
    ("BASE10NUM", r"(?:[+-]?(?:(?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+)))"),
    ("NUMBER", r"(?:%{BASE10NUM})"),
    ("BASE16NUM", r"(?:[+-]?(?:0x)?(?:[0-9A-Fa-f]+))"),
    ("BASE16FLOAT", r"\b(?:[+-]?(?:0x)?(?:(?:[0-9A-Fa-f]+(?:\.[0-9A-Fa-f]*)?)|(?:\.[0-9A-Fa-f]+)))\b"),
    ("POSINT", r"\b(?:[1-9][0-9]*)\b"),
    ("NONNEGINT", r"\b(?:[0-9]+)\b"),
    ("WORD", r"\b\w+\b"),
    ("NOTSPACE", r"\S+"),
    ("SPACE", r"\s*"),
    ("DATA", r".*?"),
    ("GREEDYDATA", r".*"),
    ("QUOTEDSTRING", r#"(?:"(?:\\.|[^\\"])*"|'(?:\\.|[^\\'])*'|`(?:\\.|[^\\`])*`)"#),
    ("UUID", r"[A-Fa-f0-9]{8}-(?:[A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}"),
    ("URN", r"urn:[0-9A-Za-z][0-9A-Za-z-]{0,31}:(?:%[0-9a-fA-F]{2}|[0-9A-Za-z()+,.:=@;$_!*'/?#-])+"),
    ("MAC", r"(?:%{CISCOMAC}|%{WINDOWSMAC}|%{COMMONMAC})"),
    ("CISCOMAC", r"(?:(?:[A-Fa-f0-9]{4}\.){2}[A-Fa-f0-9]{4})"),
    ("WINDOWSMAC", r"(?:(?:[A-Fa-f0-9]{2}-){5}[A-Fa-f0-9]{2})"),
    ("COMMONMAC", r"(?:(?:[A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2})"),
    (
        "IPV6",
        r"(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,7}:|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}|(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}|(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}|(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}|[0-9A-Fa-f]{1,4}:(?::[0-9A-Fa-f]{1,4}){1,6}|::(?:[fF]{4}(?::0{1,4})?:)?%{IPV4}|(?:[0-9A-Fa-f]{1,4}:){1,4}:%{IPV4}|:(?:(?::[0-9A-Fa-f]{1,4}){1,7}|:)|[fF][eE]80:(?::[0-9A-Fa-f]{0,4}){0,4}%[0-9a-zA-Z]+)",
    ),
    ("IPV4", r"(?:(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2}))"),
    ("IP", r"(?:%{IPV6}|%{IPV4})"),
    ("HOSTNAME", r"\b(?:[0-9A-Za-z][0-9A-Za-z-]{0,62})(?:\.(?:[0-9A-Za-z][0-9A-Za-z-]{0,62}))*(?:\.?|\b)"),
    ("IPORHOST", r"(?:%{IP}|%{HOSTNAME})"),
    ("HOSTPORT", r"%{IPORHOST}:%{POSINT}"),
    ("PATH", r"(?:%{UNIXPATH}|%{WINPATH})"),
    ("UNIXPATH", r"(?:/[\w_%!$@:.,+~-]*)+"),
    ("TTY", r"(?:/dev/(?:pts|tty(?:[pq])?)(?:\w+)?/?(?:[0-9]+))"),
    ("WINPATH", r"(?:[A-Za-z]+:|\\)(?:\\[^\\?*]*)+"),
    ("URIPROTO", r"[A-Za-z](?:[A-Za-z0-9+\-.]+)+"),
    ("URIHOST", r"%{IPORHOST}(?::%{POSINT:port})?"),
    ("URIPATH", r"(?:/[A-Za-z0-9$.+!*'(){},~:;=@#%&_\-]*)+"),
    ("URIPARAM", r"\?[A-Za-z0-9$.+!*'|(){},~@#%&/=:;_?\-\[\]<>]*"),
    ("URIPATHPARAM", r"%{URIPATH}(?:%{URIPARAM})?"),
    ("URI", r"%{URIPROTO}://(?:%{USER}(?::[^@]*)?@)?(?:%{URIHOST})?(?:%{URIPATHPARAM})?"),
    ("MONTH", r"\b(?:[Jj]an(?:uary|uar)?|[Ff]eb(?:ruary|ruar)?|[Mm](?:a|ä)?r(?:ch|z)?|[Aa]pr(?:il)?|[Mm]a(?:y|i)?|[Jj]un(?:e|i)?|[Jj]ul(?:y|i)?|[Aa]ug(?:ust)?|[Ss]ep(?:tember)?|[Oo](?:c|k)?t(?:ober)?|[Nn]ov(?:ember)?|[Dd]e(?:c|z)(?:ember)?)\b"),
    ("MONTHNUM", r"(?:0?[1-9]|1[0-2])"),
    ("MONTHNUM2", r"(?:0[1-9]|1[0-2])"),
    ("MONTHDAY", r"(?:(?:0[1-9])|(?:[12][0-9])|(?:3[01])|[1-9])"),
    ("DAY", r"(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)"),
    ("YEAR", r"(?:\d\d){1,2}"),
    ("HOUR", r"(?:2[0123]|[01]?[0-9])"),
    ("MINUTE", r"(?:[0-5][0-9])"),
    ("SECOND", r"(?:(?:[0-5]?[0-9]|60)(?:[:.,][0-9]+)?)"),
    ("TIME", r"%{HOUR}:%{MINUTE}(?::%{SECOND})"),
    ("DATE_US", r"%{MONTHNUM}[/-]%{MONTHDAY}[/-]%{YEAR}"),
    ("DATE_EU", r"%{MONTHDAY}[./-]%{MONTHNUM}[./-]%{YEAR}"),
    ("ISO8601_TIMEZONE", r"(?:Z|[+-]%{HOUR}(?::?%{MINUTE}))"),
    ("ISO8601_SECOND", r"%{SECOND}"),
    ("TIMESTAMP_ISO8601", r"%{YEAR}-%{MONTHNUM}-%{MONTHDAY}[T ]%{HOUR}:?%{MINUTE}(?::?%{SECOND})?%{ISO8601_TIMEZONE}?"),
    ("DATE", r"%{DATE_US}|%{DATE_EU}"),
    ("DATESTAMP", r"%{DATE}[- ]%{TIME}"),
    ("TZ", r"(?:[APMCE][SD]T|UTC)"),
    ("DATESTAMP_RFC822", r"%{DAY} %{MONTH} %{MONTHDAY} %{YEAR} %{TIME} %{TZ}"),
    ("DATESTAMP_RFC2822", r"%{DAY}, %{MONTHDAY} %{MONTH} %{YEAR} %{TIME} %{ISO8601_TIMEZONE}"),
    ("DATESTAMP_OTHER", r"%{DAY} %{MONTH} %{MONTHDAY} %{TIME} %{TZ} %{YEAR}"),
    ("DATESTAMP_EVENTLOG", r"%{YEAR}%{MONTHNUM2}%{MONTHDAY}%{HOUR}%{MINUTE}%{SECOND}"),
    ("SYSLOGTIMESTAMP", r"%{MONTH} +%{MONTHDAY} %{TIME}"),
    ("PROG", r"[\x21-\x5a\x5c\x5e-\x7e]+"),
    ("SYSLOGPROG", r"%{PROG:program}(?:\[%{POSINT:pid}\])?"),
    ("SYSLOGHOST", r"%{IPORHOST}"),
    ("SYSLOGFACILITY", r"<%{NONNEGINT:facility}.%{NONNEGINT:priority}>"),
    ("HTTPDATE", r"%{MONTHDAY}/%{MONTH}/%{YEAR}:%{TIME} %{INT}"),
    ("QS", r"%{QUOTEDSTRING}"),
    ("SYSLOGBASE", r"%{SYSLOGTIMESTAMP:timestamp} (?:%{SYSLOGFACILITY} )?%{SYSLOGHOST:logsource} %{SYSLOGPROG}:"),
    ("LOGLEVEL", r"(?:[Aa]lert|ALERT|[Tt]race|TRACE|[Dd]ebug|DEBUG|[Nn]otice|NOTICE|[Ii]nfo?(?:rmation)?|INFO?(?:RMATION)?|[Ww]arn?(?:ing)?|WARN?(?:ING)?|[Ee]rr?(?:or)?|ERR?(?:OR)?|[Cc]rit?(?:ical)?|CRIT?(?:ICAL)?|[Ff]atal|FATAL|[Ss]evere|SEVERE|EMERG(?:ENCY)?|[Ee]merg(?:ency)?)"),
    ("HTTPDUSER", r"%{EMAILADDRESS}|%{USER}"),
    ("HTTPDERROR_DATE", r"%{DAY} %{MONTH} %{MONTHDAY} %{TIME} %{YEAR}"),
    (
        "COMMONAPACHELOG",
        r#"%{IPORHOST:clientip} %{HTTPDUSER:ident} %{USER:auth} \[%{HTTPDATE:timestamp}\] "(?:%{WORD:verb} %{NOTSPACE:request}(?: HTTP/%{NUMBER:httpversion})?|%{DATA:rawrequest})" %{NUMBER:response} (?:%{NUMBER:bytes}|-)"#,
    ),
    ("COMBINEDAPACHELOG", r"%{COMMONAPACHELOG} %{QS:referrer} %{QS:agent}"),
    (
        "HTTPD20_ERRORLOG",
        r"\[%{HTTPDERROR_DATE:timestamp}\] \[%{LOGLEVEL:loglevel}\] (?:\[client %{IPORHOST:clientip}\] ){0,1}%{GREEDYDATA:errormsg}",
    ),
    (
        "HTTPD24_ERRORLOG",
        r"\[%{HTTPDERROR_DATE:timestamp}\] \[%{WORD:module}:%{LOGLEVEL:loglevel}\] \[pid %{POSINT:pid}(?::tid %{NUMBER:tid})?\]( \(%{POSINT:proxy_errorcode}\)%{DATA:proxy_message}:)?( \[client %{IPORHOST:clientip}:%{POSINT:clientport}\])?( %{DATA:errorcode}:)? %{GREEDYDATA:message}",
    ),
    ("HTTPD_ERRORLOG", r"%{HTTPD20_ERRORLOG}|%{HTTPD24_ERRORLOG}"),
    ("JAVACLASS", r"(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*"),
    ("JAVAFILE", r"(?:[A-Za-z0-9_. -]+)"),
    ("JAVAMETHOD", r"(?:(?:<init>)|[a-zA-Z$_][a-zA-Z$_0-9]*)"),
    ("JAVASTACKTRACEPART", r"%{SPACE}at %{JAVACLASS:class}\.%{JAVAMETHOD:method}\(%{JAVAFILE:file}(?::%{NUMBER:line})?\)"),
    ("JAVATHREAD", r"(?:[A-Z]{2}-Processor[\d]+)"),
    ("JAVALOGMESSAGE", r"(.*)"),
    ("UNIXPATH_LEGACY", r"%{UNIXPATH}"),
    ("CRON_ACTION", r"[A-Z ]+"),
    ("CRONLOG", r"%{SYSLOGBASE} \(%{USER:user}\) %{CRON_ACTION:action} \(%{DATA:message}\)"),
    ("SYSLOGLINE", r"%{SYSLOGBASE}%{GREEDYDATA:message}"),
    ("SYSLOG5424PRI", r"<%{NONNEGINT:syslog5424_pri}>"),
    ("SYSLOG5424SD", r"\[%{DATA}\]+"),
    ("SYSLOG5424PRINTASCII", r"[!-~]+"),
    (
        "SYSLOG5424BASE",
        r"%{SYSLOG5424PRI}%{NONNEGINT:syslog5424_ver} +(?:%{TIMESTAMP_ISO8601:syslog5424_ts}|-) +(?:%{IPORHOST:syslog5424_host}|-) +(-|%{SYSLOG5424PRINTASCII:syslog5424_app}) +(-|%{SYSLOG5424PRINTASCII:syslog5424_proc}) +(-|%{SYSLOG5424PRINTASCII:syslog5424_msgid}) +(?:%{SYSLOG5424SD:syslog5424_sd}|-|)",
    ),
    ("SYSLOG5424LINE", r"%{SYSLOG5424BASE} +%{GREEDYDATA:syslog5424_msg}"),
    ("MAVEN_VERSION", r"(?:(\d+)\.)?(?:(\d+)\.)?(\*|\d+)(?:[.-](RELEASE|SNAPSHOT))?"),
];

const LEGACY_SRC: &str = include_str!("grok_legacy.txt");
const ECS_V1_SRC: &str = include_str!("grok_ecs_v1.txt");

fn parse_bank(src: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for line in src.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some((name, pat)) = l.split_once(char::is_whitespace) {
            m.insert(name.to_string(), pat.trim().to_string());
        }
    }
    m
}

/// ES's own pattern banks (`patterns/legacy`, `patterns/ecs-v1` of the grok library),
/// unmodified: they use lookaround, atomic groups and `\h`, so they compile with [`Grok::compile_es`].
pub fn bank_es(ecs_v1: bool) -> &'static BTreeMap<String, String> {
    static L: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    static E: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    if ecs_v1 {
        E.get_or_init(|| parse_bank(ECS_V1_SRC))
    } else {
        L.get_or_init(|| parse_bank(LEGACY_SRC))
    }
}

/// The built-in bank.
pub fn bank() -> &'static BTreeMap<String, String> {
    static B: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    B.get_or_init(|| PATTERNS.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CapType {
    Str,
    Int,
    Long,
    Float,
    Double,
    Boolean,
}

#[derive(Clone, Debug)]
pub struct Grok {
    /// Plain-regex engine (the lite bank); a never-matching placeholder when `fancy` is set.
    pub re: Regex,
    fancy: Option<fancy_regex::Regex>,
    /// group name (g0, g1, ...) → (field, type)
    caps: Vec<(String, String, CapType)>,
}

fn field_name(semantic: &str) -> String {
    // `[a][b]` → `a.b`
    if semantic.starts_with('[') {
        semantic.trim_start_matches('[').trim_end_matches(']').replace("][", ".")
    } else {
        semantic.to_string()
    }
}

impl Grok {
    pub fn compile(pattern: &str, defs: &BTreeMap<String, String>) -> Result<Grok, EsError> {
        let mut caps = Vec::new();
        let mut stack = Vec::new();
        let src = expand(pattern, defs, &mut caps, &mut stack, 0, false)?;
        let re = Regex::new(&src).map_err(|e| EsError::illegal_argument(format!("Invalid regex pattern found in: [{pattern}]. {e}")))?;
        Ok(Grok { re, fancy: None, caps })
    }

    /// Compile against ES's real banks (lookaround, atomic groups, `\h` supported).
    pub fn compile_es(pattern: &str, defs: &BTreeMap<String, String>) -> Result<Grok, EsError> {
        let mut caps = Vec::new();
        let mut stack = Vec::new();
        let src = expand(pattern, defs, &mut caps, &mut stack, 0, true)?;
        // Oniguruma allows stacked quantifiers (`.?*`); fancy-regex does not.
        let src = ascii_classes(&src.replace(".?*", "(?:.?)*"));
        let f = fancy_regex::Regex::new(&src).map_err(|e| EsError::illegal_argument(joni_message(&e.to_string())))?;
        Ok(Grok { re: Regex::new("a^").expect("placeholder"), fancy: Some(f), caps })
    }

    /// Captures as ES's grok processor produces them: typed values that fail to parse are
    /// `NumberFormatException`s and a semantic name used twice collects a list.
    pub fn captures_checked(&self, s: &str) -> Result<Option<Map<String, Value>>, EsError> {
        let found: Vec<Option<String>> = match &self.fancy {
            Some(f) => match f.captures(s) {
                Ok(Some(c)) => self.caps.iter().map(|(g, _, _)| c.name(g).map(|m| m.as_str().to_string())).collect(),
                _ => return Ok(None),
            },
            None => match self.re.captures(s) {
                Some(c) => self.caps.iter().map(|(g, _, _)| c.name(g).map(|m| m.as_str().to_string())).collect(),
                None => return Ok(None),
            },
        };
        let mut out = Map::new();
        for ((_, field, ty), v) in self.caps.iter().zip(found) {
            let Some(v) = v else { continue };
            let nfe = |msg: String| EsError::new(500, "number_format_exception", msg);
            let val = match ty {
                CapType::Str => Value::String(v),
                CapType::Int => Value::from(parse_java_int(&v, 32).map_err(nfe)?),
                CapType::Long => Value::from(parse_java_int(&v, 64).map_err(nfe)?),
                CapType::Float => {
                    let f: f32 = v.trim().parse().map_err(|_| nfe(super::jtext::nfe_float(&v)))?;
                    super::jtext::f32_json(f)
                }
                CapType::Double => {
                    let f: f64 = v.trim().parse().map_err(|_| nfe(super::jtext::nfe_float(&v)))?;
                    Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
                }
                CapType::Boolean => Value::Bool(v.eq_ignore_ascii_case("true")),
            };
            match out.get_mut(field) {
                Some(Value::Array(a)) => a.push(val),
                Some(existing) => {
                    let first = existing.take();
                    *existing = Value::Array(vec![first, val]);
                }
                None => {
                    out.insert(field.clone(), val);
                }
            }
        }
        Ok(Some(out))
    }

    /// Redaction spans: (start, end, label) per match, the label being the first
    /// participating capture's field (`REDACTED` when none).
    pub fn spans(&self, s: &str) -> Vec<(usize, usize, String)> {
        let mut out = Vec::new();
        if let Some(f) = &self.fancy {
            for c in f.captures_iter(s).flatten() {
                let whole = c.get(0).expect("group 0");
                let label = self.caps.iter().find(|(g, _, _)| c.name(g).is_some()).map(|(_, field, _)| field.clone()).unwrap_or_else(|| "REDACTED".into());
                out.push((whole.start(), whole.end(), label));
            }
        } else {
            let names: Vec<String> = self.re.capture_names().flatten().map(str::to_string).collect();
            for c in self.re.captures_iter(s) {
                let whole = c.get(0).expect("group 0");
                let label = names.iter().find(|n| c.name(n).is_some()).map(|n| self.field_of(n)).unwrap_or_else(|| "REDACTED".into());
                out.push((whole.start(), whole.end(), label));
            }
        }
        out
    }

    /// Field name a capture group (`g0`, ...) writes to.
    pub fn field_of(&self, group: &str) -> String {
        self.caps.iter().find(|c| c.0 == group).map(|c| c.1.clone()).unwrap_or_default()
    }

    /// Captures as a field map, or `None` when the pattern does not match.
    pub fn captures(&self, s: &str) -> Option<Map<String, Value>> {
        let found: Vec<Option<String>> = match &self.fancy {
            Some(f) => {
                let c = f.captures(s).ok()??;
                self.caps.iter().map(|(g, _, _)| c.name(g).map(|m| m.as_str().to_string())).collect()
            }
            None => {
                let c = self.re.captures(s)?;
                self.caps.iter().map(|(g, _, _)| c.name(g).map(|m| m.as_str().to_string())).collect()
            }
        };
        let mut out = Map::new();
        for ((_, field, ty), v) in self.caps.iter().zip(found) {
            if let Some(v) = v {
                if out.contains_key(field) {
                    continue;
                }
                let v = v.as_str();
                let val = match ty {
                    CapType::Str => Value::String(v.to_string()),
                    CapType::Int | CapType::Long => v.trim().parse::<i64>().map(Value::from).unwrap_or(Value::String(v.to_string())),
                    CapType::Float | CapType::Double => v.trim().parse::<f64>().ok().and_then(Number::from_f64).map(Value::Number).unwrap_or(Value::String(v.to_string())),
                    CapType::Boolean => Value::Bool(v.eq_ignore_ascii_case("true")),
                };
                out.insert(field.clone(), val);
            }
        }
        Some(out)
    }
}

fn parse_java_int(s: &str, bits: u32) -> Result<i64, String> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let bad = || super::jtext::nfe_int(s);
    if body.is_empty() || !body.chars().all(|c| c.is_ascii_digit()) {
        return Err(bad());
    }
    let v: i128 = body.parse().map_err(|_| bad())?;
    let v = if neg { -v } else { v };
    let (lo, hi) = if bits == 32 { (i32::MIN as i128, i32::MAX as i128) } else { (i64::MIN as i128, i64::MAX as i128) };
    if v < lo || v > hi {
        return Err(bad());
    }
    Ok(v as i64)
}

/// Joni (Oniguruma) wording for the regex errors people hit.
fn joni_message(fancy: &str) -> String {
    let tail = fancy.rsplit(": ").next().unwrap_or(fancy);
    let l = tail.to_lowercase();
    if l.contains("opening parenthesis") || l.contains("without closing") {
        "end pattern with unmatched parenthesis".into()
    } else if l.contains("unmatched close") || l.contains("unopened") || l.contains("closing parenthesis") {
        "unmatched close parenthesis".into()
    } else if l.contains("character class") {
        "premature end of char-class".into()
    } else if l.contains("repeat operator") {
        "target of repeat operator is invalid".into()
    } else {
        l
    }
}

/// Joni's `\w`, `\d`, `\s` are ASCII-only (while `\b` stays Unicode-aware).
fn ascii_classes(src: &str) -> String {
    let cs: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len() + 16);
    let mut in_class = 0usize;
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c == '\\' && i + 1 < cs.len() {
            let n = cs[i + 1];
            let (inside, outside) = match n {
                'w' => (Some("0-9A-Za-z_"), "[0-9A-Za-z_]"),
                'd' => (Some("0-9"), "[0-9]"),
                's' => (Some("\\t\\n\\x0B\\x0C\\r "), "[\\t\\n\\x0B\\x0C\\r ]"),
                'W' => (None, "[^0-9A-Za-z_]"),
                'D' => (None, "[^0-9]"),
                'S' => (None, "[^\\t\\n\\x0B\\x0C\\r ]"),
                _ => (None, ""),
            };
            if outside.is_empty() {
                out.push(c);
                out.push(n);
            } else if in_class > 0 {
                match inside {
                    Some(r) => out.push_str(r),
                    None => {
                        out.push(c);
                        out.push(n);
                    }
                }
            } else {
                out.push_str(outside);
            }
            i += 2;
            continue;
        }
        if c == '[' {
            in_class += 1;
        } else if c == ']' && in_class > 0 {
            in_class -= 1;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn expand(p: &str, defs: &BTreeMap<String, String>, caps: &mut Vec<(String, String, CapType)>, stack: &mut Vec<String>, depth: usize, native: bool) -> Result<String, EsError> {
    if depth > 64 {
        return Err(EsError::illegal_argument("circular reference in grok pattern"));
    }
    let mut out = String::with_capacity(p.len());
    let mut i = 0;
    let b = p.as_bytes();
    while i < p.len() {
        if p[i..].starts_with("%{") {
            let end = p[i..].find('}').map(|e| i + e).ok_or_else(|| EsError::illegal_argument(format!("Invalid grok pattern [{p}]")))?;
            let inner = &p[i + 2..end];
            let mut parts = inner.splitn(3, ':');
            let name = parts.next().unwrap_or("");
            let sem = parts.next();
            let ty = parts.next();
            let def = defs.get(name).ok_or_else(|| EsError::illegal_argument(format!("Unable to find pattern [{name}] in Grok's pattern dictionary")))?;
            if stack.iter().any(|s| s == name) {
                if native {
                    return Err(EsError::illegal_argument(format!("circular reference detected: {}->{name}", stack.join("->"))));
                }
                return Err(EsError::illegal_argument(format!("circular reference in pattern [{name}][{}]", stack.join("=>"))));
            }
            stack.push(name.to_string());
            let sub = expand(def, defs, caps, stack, depth + 1, native)?;
            stack.pop();
            match sem {
                Some(s) if !s.is_empty() => {
                    let g = format!("g{}", caps.len());
                    let t = match ty {
                        Some("int") => CapType::Int,
                        Some("long") => CapType::Long,
                        Some("float") => CapType::Float,
                        Some("double") => CapType::Double,
                        Some("boolean") => CapType::Boolean,
                        _ => CapType::Str,
                    };
                    caps.push((g.clone(), if native { s.to_string() } else { field_name(s) }, t));
                    out.push_str(&format!("(?P<{g}>{sub})"));
                }
                _ => {
                    out.push_str("(?:");
                    out.push_str(&sub);
                    out.push(')');
                }
            }
            i = end + 1;
            continue;
        }
        // Named groups written inline: (?<name>...) / (?P<name>...)
        if p[i..].starts_with("(?<") && !p[i..].starts_with("(?<=") && !p[i..].starts_with("(?<!") {
            if let Some(e) = p[i + 3..].find('>') {
                let name = &p[i + 3..i + 3 + e];
                let g = format!("g{}", caps.len());
                caps.push((g.clone(), field_name(name), CapType::Str));
                out.push_str(&format!("(?P<{g}>"));
                i = i + 3 + e + 1;
                continue;
            }
        }
        if p[i..].starts_with("(?P<") {
            if let Some(e) = p[i + 4..].find('>') {
                let name = &p[i + 4..i + 4 + e];
                let g = format!("g{}", caps.len());
                caps.push((g.clone(), field_name(name), CapType::Str));
                out.push_str(&format!("(?P<{g}>"));
                i = i + 4 + e + 1;
                continue;
            }
        }
        // Atomic groups → plain non-capturing groups (unless the engine supports them).
        if !native && p[i..].starts_with("(?>") {
            out.push_str("(?:");
            i += 3;
            continue;
        }
        if b[i] == b'\\' && i + 1 < p.len() {
            let ch = p[i + 1..].chars().next().unwrap();
            out.push('\\');
            out.push(ch);
            i += 1 + ch.len_utf8();
            continue;
        }
        let ch = p[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn apache_and_types() {
        let g = Grok::compile("%{COMBINEDAPACHELOG}", bank()).unwrap();
        let line = r#"127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326 "http://www.example.com/start.html" "Mozilla/4.08 [en] (Win98; I ;Nav)""#;
        let c = g.captures(line).unwrap();
        assert_eq!(c["clientip"], json!("127.0.0.1"));
        assert_eq!(c["verb"], json!("GET"));
        assert_eq!(c["timestamp"], json!("10/Oct/2000:13:55:36 -0700"));
        assert_eq!(c["bytes"], json!("2326"));
        assert_eq!(c["agent"], json!("\"Mozilla/4.08 [en] (Win98; I ;Nav)\""));
        let g = Grok::compile("%{IP:client} %{WORD:method} %{URIPATHPARAM:request} %{NUMBER:bytes:int} %{NUMBER:duration:float}", bank()).unwrap();
        let c = g.captures("55.3.244.1 GET /index.html 15824 0.043").unwrap();
        assert_eq!(c["bytes"], json!(15824));
        assert_eq!(c["duration"], json!(0.043));
        let g = Grok::compile("(?<id>[0-9]+) %{WORD:[a][b]}", bank()).unwrap();
        let c = g.captures("12 x").unwrap();
        assert_eq!(c["id"], json!("12"));
        assert_eq!(c["a.b"], json!("x"));
        assert!(Grok::compile("%{NOPE}", bank()).is_err());
    }

    #[test]
    fn es_banks_compile_and_match() {
        // every pattern of both banks compiles on the fancy engine
        for ecs in [false, true] {
            let bank = bank_es(ecs);
            for name in bank.keys() {
                // upstream typo (unclosed character class): ES cannot compile it either
                if bank[name].contains("[@metadata") {
                    continue;
                }
                Grok::compile_es(&format!("%{{{name}}}"), bank).unwrap_or_else(|e| panic!("{name} (ecs={ecs}): {}", e.reason));
            }
        }
        assert_eq!(bank_es(false).len(), 318);
        assert_eq!(bank_es(true).len(), 361);
        // lookbehind/lookahead in IPV4 keep digits from matching inside longer runs
        let g = Grok::compile_es("%{IPV4:ip}", bank_es(false)).unwrap();
        assert_eq!(g.captures("a 1.2.3.4 b").unwrap()["ip"], json!("1.2.3.4"));
        assert!(g.captures("1234.5.6.7").is_none());
        assert_eq!(g.captures("1234.5.6.7.8").unwrap()["ip"], json!("5.6.7.8"));
        let g = Grok::compile_es("%{COMBINEDAPACHELOG}", bank_es(true)).unwrap();
        let c = g.captures(r#"127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /a.gif HTTP/1.0" 200 2326 "http://x/" "Moz""#).unwrap();
        assert_eq!(c["source.address"], json!("127.0.0.1"));
        assert_eq!(c["http.request.method"], json!("GET"));
        let g = Grok::compile_es("%{RUUID:u}", bank_es(false)).unwrap();
        assert!(g.captures("0123456789abcdef0123456789abcdef").is_some());
    }
}
