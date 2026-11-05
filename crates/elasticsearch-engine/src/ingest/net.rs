//! Network-flavoured processors: `uri_parts`, `registered_domain`,
//! `network_direction`, `community_id`. Behaviour is checked against ES
//! 8.19.22 (Java `URI`/`URL` semantics, Apache `PublicSuffixMatcher`,
//! `InetAddresses`, `CommunityIdProcessor`).

use super::doc::{type_name, value_to_string, IngestDocument};
use super::jtext::java_tostring;
use super::processors::{ia, input, rf};
use super::{Cfg, PResult, PipelineContext, Processor};
use crate::error::{EsError, Result};
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::OnceLock;

// ------------------------------------------------------------ IP helpers

/// Guava/ES `InetAddresses.forString`: no brackets, no scope ids; an
/// IPv4-mapped IPv6 address becomes the IPv4 address (as `InetAddress` does).
pub fn inet_for_string(s: &str) -> std::result::Result<IpAddr, String> {
    let bad = || format!("'{s}' is not an IP string literal.");
    if s.is_empty() || s.contains('[') || s.contains('%') {
        return Err(bad());
    }
    match s.parse::<IpAddr>() {
        Ok(IpAddr::V6(v6)) => Ok(match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => IpAddr::V6(v6),
        }),
        Ok(ip) => Ok(ip),
        Err(_) => Err(bad()),
    }
}

fn is_link_local_v4(v: &Ipv4Addr) -> bool {
    v.octets()[0] == 169 && v.octets()[1] == 254
}

fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => v.is_private(),
        IpAddr::V6(v) => v.octets()[0] == 0xfd,
    }
}

fn is_link_local_multicast(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => v.octets()[0] == 224 && v.octets()[1] == 0 && v.octets()[2] == 0,
        IpAddr::V6(v) => v.octets()[0] == 0xff && (v.octets()[1] & 0x0f) == 2,
    }
}

fn is_interface_local_multicast(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(_) => false,
        IpAddr::V6(v) => v.octets()[0] == 0xff && (v.octets()[1] & 0x0f) == 1,
    }
}

fn is_link_local_unicast(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => is_link_local_v4(v),
        IpAddr::V6(v) => (v.segments()[0] & 0xffc0) == 0xfe80,
    }
}

fn is_broadcast(ip: &IpAddr) -> bool {
    matches!(ip, IpAddr::V4(v) if v.is_broadcast())
}

/// ES `NetworkDirectionProcessor.inNetwork`.
fn in_named_network(ip: &IpAddr, net: &str) -> Option<bool> {
    Some(match net {
        "loopback" => ip.is_loopback(),
        "unspecified" => ip.is_unspecified(),
        "multicast" => ip.is_multicast(),
        "link_local_unicast" => is_link_local_unicast(ip),
        "interface_local_multicast" => is_interface_local_multicast(ip),
        "link_local_multicast" => is_link_local_multicast(ip),
        "private" => is_private_ip(ip),
        "public" => !(ip.is_unspecified() || ip.is_loopback() || is_link_local_unicast(ip) || is_interface_local_multicast(ip) || is_link_local_multicast(ip) || is_private_ip(ip) || is_broadcast(ip)),
        "unicast" | "global_unicast" => !(ip.is_unspecified() || ip.is_loopback() || ip.is_multicast() || is_link_local_unicast(ip) || is_broadcast(ip)),
        _ => return None,
    })
}

fn in_cidr(ip: &IpAddr, net: &str) -> std::result::Result<bool, String> {
    let (addr, bits) = match net.split_once('/') {
        Some((a, b)) => (a, Some(b)),
        None => (net, None),
    };
    let n = inet_for_string(addr)?;
    let bits: Option<u32> = match bits {
        Some(b) => Some(b.parse().map_err(|_| format!("For input string: \"{b}\""))?),
        None => None,
    };
    Ok(match (ip, n) {
        (IpAddr::V4(a), IpAddr::V4(b)) => {
            let bits = bits.unwrap_or(32).min(32);
            let mask = if bits == 0 { 0 } else { u32::MAX << (32 - bits) };
            (u32::from(*a) & mask) == (u32::from(b) & mask)
        }
        (IpAddr::V6(a), IpAddr::V6(b)) => {
            let bits = bits.unwrap_or(128).min(128);
            let mask = if bits == 0 { 0 } else { u128::MAX << (128 - bits) };
            (u128::from(*a) & mask) == (u128::from(b) & mask)
        }
        _ => false,
    })
}

// ------------------------------------------------------------ java.net.URI / URL

#[derive(Debug, Default)]
pub struct JUri {
    pub scheme: Option<String>,
    pub host: Option<String>,
    pub port: i64,
    pub user_info: Option<String>,
    pub path: Option<String>,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

fn pct_decode(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(v) = s.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf16_lossy(&crate::script::decode_java_utf8(&out))
}

fn valid_escapes(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            if i + 2 >= b.len() {
                return false;
            }
            if !(b[i + 1].is_ascii_hexdigit() && b[i + 2].is_ascii_hexdigit()) {
                return false;
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    true
}

fn valid_hostname(h: &str) -> bool {
    let h = h.strip_suffix('.').unwrap_or(h);
    if h.is_empty() {
        return false;
    }
    let labels: Vec<&str> = h.split('.').collect();
    for (i, l) in labels.iter().enumerate() {
        let b = l.as_bytes();
        if b.is_empty() || !b[0].is_ascii_alphanumeric() || !b[b.len() - 1].is_ascii_alphanumeric() {
            return false;
        }
        if !b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'-') {
            return false;
        }
        if i == labels.len() - 1 && !b[0].is_ascii_alphabetic() {
            return false;
        }
    }
    true
}

fn valid_ipv4_literal(h: &str) -> bool {
    let p: Vec<&str> = h.split('.').collect();
    p.len() == 4 && p.iter().all(|x| !x.is_empty() && x.len() <= 3 && x.chars().all(|c| c.is_ascii_digit()) && x.parse::<u32>().map(|n| n <= 255).unwrap_or(false))
}

/// Server-based authority: `[userinfo@]host[:port]`.
pub(super) fn server_authority(a: &str) -> Option<(Option<String>, String, i64)> {
    let (ui, hp) = match a.find('@') {
        Some(i) => {
            let u = &a[..i];
            if !u.chars().all(|c| c.is_ascii_alphanumeric() || "-_.!~*'()%;:&=+$,".contains(c) || !c.is_ascii()) {
                return None;
            }
            (Some(pct_decode(u)), &a[i + 1..])
        }
        None => (None, a),
    };
    let (host, port) = if hp.starts_with('[') {
        let e = hp.find(']')?;
        let inner = &hp[1..e];
        if inner.is_empty() || !inner.chars().all(|c| c.is_ascii_hexdigit() || c == ':' || c == '.' || c == '%') {
            return None;
        }
        (hp[..=e].to_string(), &hp[e + 1..])
    } else {
        match hp.find(':') {
            Some(i) => (hp[..i].to_string(), &hp[i..]),
            None => (hp.to_string(), ""),
        }
    };
    if !host.starts_with('[') && !(valid_ipv4_literal(&host) || valid_hostname(&host)) {
        return None;
    }
    let port = if port.is_empty() {
        -1
    } else {
        let p = port.strip_prefix(':')?;
        if p.is_empty() {
            -1
        } else if p.chars().all(|c| c.is_ascii_digit()) {
            p.parse::<i64>().ok().filter(|n| *n <= i32::MAX as i64)?
        } else {
            return None;
        }
    };
    Some((ui, host, port))
}

/// `new URI(s)`; `Err` is a `URISyntaxException`.
pub fn java_uri(s: &str) -> std::result::Result<JUri, ()> {
    for c in s.chars() {
        if (c as u32) <= 0x20 || c as u32 == 0x7f || "\"<>\\^`{|}".contains(c) || (c as u32 >= 0x80 && c.is_whitespace()) {
            return Err(());
        }
    }
    if !valid_escapes(s) {
        return Err(());
    }
    let mut u = JUri { port: -1, ..Default::default() };
    let (main, frag) = match s.find('#') {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    if let Some(f) = frag {
        if f.contains('#') {
            return Err(());
        }
        u.fragment = Some(pct_decode(f));
    }
    // scheme
    let mut rest = main;
    if let Some(ci) = main.find(':') {
        let before = &main[..ci];
        if !before.contains(['/', '?']) {
            let ok = !before.is_empty() && before.chars().next().unwrap().is_ascii_alphabetic() && before.chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c));
            if !ok {
                return Err(());
            }
            u.scheme = Some(before.to_string());
            rest = &main[ci + 1..];
            if rest.is_empty() {
                return Err(());
            }
            if !rest.starts_with('/') {
                // opaque: scheme-specific part only
                return Ok(u);
            }
        }
    }
    let (hier, query) = match rest.find('?') {
        Some(i) => (&rest[..i], Some(&rest[i + 1..])),
        None => (rest, None),
    };
    if let Some(q) = query {
        u.query = Some(pct_decode(q));
    }
    let path;
    if let Some(after) = hier.strip_prefix("//") {
        let (auth, p) = match after.find('/') {
            Some(i) => (&after[..i], &after[i..]),
            None => (after, ""),
        };
        path = p;
        if !auth.is_empty() {
            match server_authority(auth) {
                Some((ui, host, port)) => {
                    u.user_info = ui;
                    u.host = Some(host);
                    u.port = port;
                }
                None => {
                    // registry-based authority
                    if !auth.chars().all(|c| c.is_ascii_alphanumeric() || "-_.!~*'()%$,;:@&=+".contains(c) || !c.is_ascii()) {
                        return Err(());
                    }
                }
            }
        }
    } else {
        path = hier;
    }
    u.path = Some(pct_decode(path));
    Ok(u)
}

/// `new URL(s)` for the protocols the JDK handles; `Err` = MalformedURLException.
fn java_url(s: &str) -> std::result::Result<JUri, ()> {
    let ci = s.find(':').ok_or(())?;
    let proto = &s[..ci];
    if !matches!(proto.to_ascii_lowercase().as_str(), "http" | "https" | "ftp" | "file" | "jar" | "jrt" | "mailto") {
        return Err(());
    }
    let mut u = JUri { port: -1, scheme: Some(proto.to_ascii_lowercase()), ..Default::default() };
    let mut rest = &s[ci + 1..];
    if let Some(i) = rest.find('#') {
        u.fragment = Some(rest[i + 1..].to_string());
        rest = &rest[..i];
    }
    if let Some(i) = rest.find('?') {
        u.query = Some(rest[i + 1..].to_string());
        rest = &rest[..i];
    }
    if let Some(after) = rest.strip_prefix("//") {
        let (auth, p) = match after.find('/') {
            Some(i) => (&after[..i], &after[i..]),
            None => (after, ""),
        };
        let hp = match auth.rfind('@') {
            Some(i) => {
                u.user_info = Some(auth[..i].to_string());
                &auth[i + 1..]
            }
            None => auth,
        };
        let (h, port) = if hp.starts_with('[') {
            match hp.find(']') {
                Some(e) => (&hp[..=e], hp[e + 1..].strip_prefix(':')),
                None => return Err(()),
            }
        } else {
            match hp.rfind(':') {
                Some(i) => (&hp[..i], Some(&hp[i + 1..])),
                None => (hp, None),
            }
        };
        if let Some(p) = port.filter(|p| !p.is_empty()) {
            u.port = p.parse::<i64>().map_err(|_| ())?;
        }
        u.host = Some(h.to_string());
        u.path = Some(p.to_string());
    } else {
        u.path = Some(rest.to_string());
        u.host = Some(String::new());
    }
    Ok(u)
}

fn opt(s: Option<String>) -> Value {
    s.map(Value::String).unwrap_or(Value::Null)
}

/// The `uri_parts` result map for `s`.
pub fn uri_parts_map(s: &str, keep_original: bool) -> Option<Map<String, Value>> {
    let (u, fallback) = match java_uri(s) {
        Ok(u) => (u, false),
        Err(()) => (java_url(s).ok()?, true),
    };
    let mut m = Map::new();
    m.insert("domain".into(), opt(u.host.clone()));
    if keep_original {
        m.insert("original".into(), Value::String(s.to_string()));
    }
    if let Some(f) = &u.fragment {
        m.insert("fragment".into(), Value::String(f.clone()));
    }
    if let Some(p) = &u.path {
        m.insert("path".into(), Value::String(p.clone()));
        if let Some(pi) = p.rfind('.') {
            if p.rfind('/').map(|sl| pi > sl).unwrap_or(true) {
                m.insert("extension".into(), Value::String(p[pi + 1..].to_string()));
            }
        }
    }
    m.insert("scheme".into(), opt(u.scheme.clone()));
    if u.port != -1 {
        m.insert("port".into(), Value::from(u.port));
    }
    if let Some(q) = &u.query {
        m.insert("query".into(), Value::String(q.clone()));
    }
    if let Some(ui) = &u.user_info {
        m.insert("user_info".into(), Value::String(ui.clone()));
        let (name, pass) = match ui.split_once(':') {
            Some((a, b)) => (a.to_string(), Some(b.to_string())),
            None => (ui.clone(), None),
        };
        m.insert("username".into(), Value::String(name));
        if let Some(p) = pass {
            m.insert("password".into(), Value::String(p));
        }
    }
    let _ = fallback;
    Some(m)
}

// ------------------------------------------------------------ uri_parts

pub struct UriPartsP {
    pub field: String,
    pub target: String,
    pub keep_original: bool,
    pub remove_if_successful: bool,
    pub ignore_missing: bool,
}

impl Processor for UriPartsP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot parse URI.")? else { return Ok(()) };
        let s = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        let m = uri_parts_map(&s, self.keep_original).ok_or_else(|| ia(format!("unable to parse URI [{s}]")))?;
        doc.set(&self.target, Value::Object(m))?;
        if self.remove_if_successful && self.field != self.target {
            doc.remove(&self.field)?;
        }
        Ok(())
    }
}

pub fn build_uri_parts(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let target = cfg.str_or("target_field", "url")?;
    let keep_original = cfg.bool_or("keep_original", true)?;
    let remove_if_successful = cfg.bool_or("remove_if_successful", false)?;
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    Ok(Box::new(UriPartsP { field, target, keep_original, remove_if_successful, ignore_missing }))
}

// ------------------------------------------------------------ registered_domain

struct Psl {
    icann: HashSet<&'static str>,
    private: HashSet<&'static str>,
    exc: HashSet<&'static str>,
}

fn psl() -> &'static Psl {
    static P: OnceLock<Psl> = OnceLock::new();
    P.get_or_init(|| {
        let text: &'static str = include_str!("psl.txt");
        let mut p = Psl { icann: HashSet::new(), private: HashSet::new(), exc: HashSet::new() };
        let mut section = 0u8; // 1 = ICANN, 2 = PRIVATE
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() {
                continue;
            }
            if let Some(c) = l.strip_prefix("//") {
                if c.contains("===BEGIN ICANN DOMAINS===") {
                    section = 1;
                } else if c.contains("===BEGIN PRIVATE DOMAINS===") {
                    section = 2;
                } else if c.contains("===END") {
                    section = 0;
                }
                continue;
            }
            if section == 0 {
                continue;
            }
            let l = l.strip_prefix('.').unwrap_or(l);
            if let Some(e) = l.strip_prefix('!') {
                p.exc.insert(e);
            } else if section == 1 {
                p.icann.insert(l);
            } else {
                p.private.insert(l);
            }
        }
        p
    })
}

/// Apache `PublicSuffixMatcher.getDomainRoot` as used by ES: ICANN rules give
/// "suffix + one label"; a private rule is itself the registered domain; no
/// matching rule yields `None`.
fn domain_root(fqdn: &str) -> Option<String> {
    let p = psl();
    if fqdn.starts_with('.') {
        return None;
    }
    let mut seg: Option<String> = Some(fqdn.to_lowercase());
    let mut domain_name: Option<String> = None;
    while let Some(s) = seg {
        if p.exc.contains(s.as_str()) {
            return Some(s);
        }
        if p.icann.contains(s.as_str()) {
            return domain_name;
        }
        if p.private.contains(s.as_str()) {
            return Some(s);
        }
        let nd = s.find('.');
        let next = nd.map(|i| s[i + 1..].to_string());
        if let Some(n) = &next {
            let w = format!("*.{n}");
            if p.icann.contains(w.as_str()) {
                return domain_name;
            }
            if p.private.contains(w.as_str()) {
                return Some(s);
            }
        }
        if nd.is_some() {
            domain_name = Some(s.clone());
        }
        seg = next;
    }
    None
}

fn is_public_suffix(d: &str) -> bool {
    let p = psl();
    let d = d.to_lowercase();
    if p.icann.contains(d.as_str()) || p.private.contains(d.as_str()) {
        return true;
    }
    if let Some(i) = d.find('.') {
        let w = format!("*.{}", &d[i + 1..]);
        return p.icann.contains(w.as_str()) || p.private.contains(w.as_str());
    }
    false
}

pub struct DomainInfo {
    pub domain: String,
    pub registered: Option<String>,
    pub etld: String,
    pub subdomain: Option<String>,
}

pub fn registered_domain(fqdn: &str) -> Option<DomainInfo> {
    if fqdn.trim().is_empty() {
        return None;
    }
    match domain_root(fqdn) {
        None => {
            let stripped = fqdn.strip_prefix('.').unwrap_or(fqdn);
            if is_public_suffix(stripped) {
                Some(DomainInfo { domain: fqdn.to_string(), registered: None, etld: fqdn.to_string(), subdomain: None })
            } else {
                None
            }
        }
        Some(root) => {
            let dot = root.find('.')?;
            let etld = root[dot + 1..].to_string();
            let sub = fqdn.rfind(&format!(".{root}")).map(|i| fqdn[..i].to_string());
            Some(DomainInfo { domain: fqdn.to_string(), registered: Some(root), etld, subdomain: sub })
        }
    }
}

pub struct RegisteredDomainP {
    pub field: String,
    pub target: String,
    pub ignore_missing: bool,
}

impl Processor for RegisteredDomainP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None => return Err(super::processors::missing(&self.field).into()),
            Some(Value::Null) => return Err(ia(format!("field [{}] is null, cannot process it.", self.field)).into()),
            Some(v) => v.clone(),
        };
        let fqdn = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        let Some(info) = registered_domain(&fqdn) else {
            if self.ignore_missing {
                return Ok(());
            }
            return Err(ia("unable to set domain information for document").into());
        };
        let p = |k: &str| if self.target.is_empty() { k.to_string() } else { format!("{}.{k}", self.target) };
        if !info.domain.is_empty() {
            doc.set(&p("domain"), Value::String(info.domain))?;
        }
        if let Some(r) = info.registered.filter(|r| !r.is_empty()) {
            doc.set(&p("registered_domain"), Value::String(r))?;
        }
        if !info.etld.is_empty() {
            doc.set(&p("top_level_domain"), Value::String(info.etld))?;
        }
        if let Some(s) = info.subdomain.filter(|s| !s.is_empty()) {
            doc.set(&p("subdomain"), Value::String(s))?;
        }
        Ok(())
    }
}

pub fn build_registered_domain(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let target = cfg.str_or("target_field", "")?;
    let ignore_missing = cfg.bool_or("ignore_missing", true)?;
    Ok(Box::new(RegisteredDomainP { field, target, ignore_missing }))
}

// ------------------------------------------------------------ network_direction

pub struct NetworkDirectionP {
    pub source: String,
    pub dest: String,
    pub target: String,
    pub networks: Option<Vec<String>>,
    pub networks_field: Option<String>,
    pub ignore_missing: bool,
}

impl NetworkDirectionP {
    fn is_internal(&self, nets: &[String], ip: &IpAddr) -> Result<bool> {
        for n in nets {
            let hit = match in_named_network(ip, n) {
                Some(b) => b,
                None => in_cidr(ip, n).map_err(ia)?,
            };
            if hit {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Processor for NetworkDirectionP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let get = |doc: &IngestDocument, f: &str| -> Result<Option<String>> {
            match doc.get(f)? {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                Some(other) => Err(ia(format!("field [{f}] of type [{}] cannot be cast to [java.lang.String]", type_name(other)))),
            }
        };
        let src = get(doc, &self.source)?;
        let dst = get(doc, &self.dest)?;
        let (Some(src), Some(dst)) = (src, dst) else {
            if self.ignore_missing {
                return Ok(());
            }
            let f = if doc.get(&self.source)?.map(|v| !v.is_null()).unwrap_or(false) { &self.dest } else { &self.source };
            return Err(ia(format!("field [{f}] not present as part of path [{f}]")).into());
        };
        let nets: Vec<String> = match (&self.networks, &self.networks_field) {
            (Some(n), _) => n.iter().map(|x| rf(doc, x)).collect(),
            (None, Some(f)) => match doc.get(f)? {
                Some(Value::Array(a)) => a.iter().map(value_to_string).collect(),
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Null) | None => {
                    if self.ignore_missing {
                        return Ok(());
                    }
                    return Err(ia(format!("field [{f}] not present as part of path [{f}]")).into());
                }
                Some(other) => return Err(ia(format!("field [{f}] of type [{}] cannot be cast to [java.util.List]", type_name(other))).into()),
            },
            _ => Vec::new(),
        };
        let s = inet_for_string(&src).map_err(ia)?;
        let d = inet_for_string(&dst).map_err(ia)?;
        let dir = match (self.is_internal(&nets, &s)?, self.is_internal(&nets, &d)?) {
            (true, true) => "internal",
            (true, false) => "outbound",
            (false, true) => "inbound",
            (false, false) => "external",
        };
        doc.set(&self.target, Value::String(dir.into()))?;
        Ok(())
    }
}

pub fn build_network_direction(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let source = cfg.str_or("source_ip", "source.ip")?;
    let dest = cfg.str_or("destination_ip", "destination.ip")?;
    let target = cfg.str_or("target_field", "network.direction")?;
    let networks = cfg.opt_list_str("internal_networks")?;
    let networks_field = cfg.opt_str("internal_networks_field")?;
    if networks.is_none() && networks_field.is_none() {
        return Err(cfg.err("internal_networks", "or [internal_networks_field] must be specified"));
    }
    let ignore_missing = cfg.bool_or("ignore_missing", true)?;
    Ok(Box::new(NetworkDirectionP { source, dest, target, networks, networks_field, ignore_missing }))
}

// ------------------------------------------------------------ community_id

pub struct CommunityIdP {
    pub src_ip: String,
    pub src_port: String,
    pub dst_ip: String,
    pub dst_port: String,
    pub iana: String,
    pub transport: String,
    pub icmp_type: String,
    pub icmp_code: String,
    pub target: String,
    pub seed: u16,
    pub ignore_missing: bool,
}

fn transport_number(v: &Value) -> std::result::Result<u8, String> {
    let from_number = |n: i64| -> std::result::Result<u8, String> {
        if !(0..=255).contains(&n) {
            Err(format!("invalid transport protocol number [{n}]"))
        } else {
            Ok(n as u8)
        }
    };
    match v {
        Value::Number(n) => from_number(n.as_i64().or(n.as_f64().map(|f| f as i64)).unwrap_or(-1)),
        Value::String(s) => match s.to_lowercase().as_str() {
            "icmp" => Ok(1),
            "igmp" => Ok(2),
            "tcp" => Ok(6),
            "udp" => Ok(17),
            "gre" => Ok(47),
            "ipv6-icmp" => Ok(58),
            "eigrp" => Ok(88),
            "ospf" => Ok(89),
            "pim" => Ok(103),
            "sctp" => Ok(132),
            _ => match s.parse::<i64>() {
                Ok(n) => from_number(n),
                Err(_) => Err(format!("could not parse transport type [{s}]")),
            },
        },
        other => Err(format!("transport type must be a number or string, got [{}]", type_name(other))),
    }
}

fn int_of(v: &Value, what: &str) -> std::result::Result<i64, String> {
    match v {
        Value::Number(n) => Ok(n.as_i64().or(n.as_f64().map(|f| f as i64)).unwrap_or(0)),
        Value::String(s) => s.parse::<i64>().map_err(|_| format!("unable to parse {what} [{s}]")),
        other => Err(format!("unable to parse {what} [{}]", java_tostring(other))),
    }
}

fn icmp_equivalent(t: i64, v6: bool) -> Option<i64> {
    if v6 {
        match t {
            128 => Some(129),
            129 => Some(128),
            133 => Some(134),
            134 => Some(133),
            136 => Some(135),
            135 => Some(136),
            130 => Some(131),
            139 => Some(140),
            140 => Some(139),
            144 => Some(145),
            145 => Some(144),
            _ => None,
        }
    } else {
        match t {
            8 => Some(0),
            0 => Some(8),
            13 => Some(14),
            14 => Some(13),
            15 => Some(16),
            10 => Some(9),
            9 => Some(10),
            17 => Some(18),
            18 => Some(17),
            _ => None,
        }
    }
}

impl CommunityIdP {
    /// Build the flow hash input; `Ok(None)` when the document lacks data.
    fn flow(&self, doc: &IngestDocument) -> Result<Option<String>> {
        use base64::Engine;
        let get = |f: &str| -> Result<Option<Value>> { Ok(doc.get(f)?.filter(|v| !v.is_null()).cloned()) };
        let sip = match get(&self.src_ip)? {
            Some(v) => value_to_string(&v),
            None => return Ok(None),
        };
        let dip = match get(&self.dst_ip)? {
            Some(v) => value_to_string(&v),
            None => return Ok(None),
        };
        let src = inet_for_string(&sip).map_err(ia)?;
        let dst = inet_for_string(&dip).map_err(ia)?;
        let tv = match get(&self.iana)? {
            Some(v) => v,
            None => match get(&self.transport)? {
                Some(v) => v,
                None => return Ok(None),
            },
        };
        let proto = transport_number(&tv).map_err(ia)?;
        let (mut sp, mut dp) = (0i64, 0i64);
        let mut icmp_oneway = false;
        let is_icmp = proto == 1 || proto == 58;
        match proto {
            6 | 17 | 132 => {
                let Some(a) = get(&self.src_port)? else { return Ok(None) };
                let a = int_of(&a, "source port").map_err(ia)?;
                if !(0..=65535).contains(&a) {
                    return Err(ia(format!("invalid source port [{}]", a)));
                }
                let Some(b) = get(&self.dst_port)? else { return Ok(None) };
                let b = int_of(&b, "destination port").map_err(ia)?;
                if !(0..=65535).contains(&b) {
                    return Err(ia(format!("invalid destination port [{}]", b)));
                }
                sp = a;
                dp = b;
            }
            1 | 58 => {
                let Some(t) = get(&self.icmp_type)? else { return Ok(None) };
                let t = int_of(&t, "icmp type").map_err(ia)?;
                let code = match get(&self.icmp_code)? {
                    Some(c) => int_of(&c, "icmp code").map_err(ia)?,
                    None => 0,
                };
                sp = t;
                match icmp_equivalent(t, proto == 58) {
                    Some(e) => dp = e,
                    None => {
                        dp = code;
                        icmp_oneway = true;
                    }
                }
            }
            _ => {}
        }
        let sb = match src {
            IpAddr::V4(v) => v.octets().to_vec(),
            IpAddr::V6(v) => v.octets().to_vec(),
        };
        let db = match dst {
            IpAddr::V4(v) => v.octets().to_vec(),
            IpAddr::V6(v) => v.octets().to_vec(),
        };
        // BigInteger(1, bytes) comparison: by numeric value, i.e. length then bytes.
        let cmp = if sb.len() != db.len() { sb.len().cmp(&db.len()) } else { sb.cmp(&db) };
        let ordered = cmp == std::cmp::Ordering::Less || (cmp == std::cmp::Ordering::Equal && sp < dp);
        let first = ordered || (is_icmp && icmp_oneway);
        let mut buf = self.seed.to_be_bytes().to_vec();
        if first {
            buf.extend_from_slice(&sb);
            buf.extend_from_slice(&db);
        } else {
            buf.extend_from_slice(&db);
            buf.extend_from_slice(&sb);
        }
        buf.extend_from_slice(&((proto as u16) << 8).to_be_bytes());
        if matches!(proto, 6 | 17 | 132 | 1 | 58) {
            let (a, b) = if first { (sp, dp) } else { (dp, sp) };
            buf.extend_from_slice(&(a as u16).to_be_bytes());
            buf.extend_from_slice(&(b as u16).to_be_bytes());
        }
        let h = super::util::sha1(&buf);
        Ok(Some(format!("1:{}", base64::engine::general_purpose::STANDARD.encode(h))))
    }
}

impl Processor for CommunityIdP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        match self.flow(doc)? {
            Some(id) => {
                doc.set(&self.target, Value::String(id))?;
                Ok(())
            }
            None => {
                if self.ignore_missing {
                    Ok(())
                } else {
                    Err(ia("unable to construct flow from document").into())
                }
            }
        }
    }
}

pub fn build_community_id(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let src_ip = cfg.str_or("source_ip", "source.ip")?;
    let src_port = cfg.str_or("source_port", "source.port")?;
    let dst_ip = cfg.str_or("destination_ip", "destination.ip")?;
    let dst_port = cfg.str_or("destination_port", "destination.port")?;
    let iana = cfg.str_or("iana_number", "network.iana_number")?;
    let transport = cfg.str_or("transport", "network.transport")?;
    let icmp_type = cfg.str_or("icmp_type", "icmp.type")?;
    let icmp_code = cfg.str_or("icmp_code", "icmp.code")?;
    let target = cfg.str_or("target_field", "network.community_id")?;
    let seed = cfg.opt_i64("seed")?.unwrap_or(0);
    if !(0..=65535).contains(&seed) {
        return Err(cfg.err("seed", "must be a value between 0 and 65535"));
    }
    let ignore_missing = cfg.bool_or("ignore_missing", true)?;
    Ok(Box::new(CommunityIdP { src_ip, src_port, dst_ip, dst_port, iana, transport, icmp_type, icmp_code, target, seed: seed as u16, ignore_missing }))
}

#[allow(dead_code)]
fn _unused(_: Ipv6Addr, _: EsError) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uris() {
        let m = uri_parts_map("https://user:pw@example.com:8443/a/b.php?x=1#frag", true).unwrap();
        assert_eq!(m["domain"], "example.com");
        assert_eq!(m["port"], 8443);
        assert_eq!(m["extension"], "php");
        assert_eq!(m["password"], "pw");
        let m = uri_parts_map("/rel/path?a=b", false).unwrap();
        assert!(m["scheme"].is_null() && m["domain"].is_null());
        assert_eq!(m["path"], "/rel/path");
        let m = uri_parts_map("mailto:a@b.com", false).unwrap();
        assert!(m.get("path").is_none());
        let m = uri_parts_map("http://h:abc/", false).unwrap();
        assert!(m["domain"].is_null());
        let m = uri_parts_map("http://h/a b", false).unwrap();
        assert_eq!(m["path"], "/a b");
        assert!(uri_parts_map("bad uri", false).is_none());
        let m = uri_parts_map("http://h.com/a.b/c", false).unwrap();
        assert!(m.get("extension").is_none());
        let m = uri_parts_map("http://[fe80::1]/", false).unwrap();
        assert_eq!(m["domain"], "[fe80::1]");
    }

    #[test]
    fn domains() {
        let d = registered_domain("www.google.co.uk").unwrap();
        assert_eq!((d.registered.as_deref(), d.etld.as_str(), d.subdomain.as_deref()), (Some("google.co.uk"), "co.uk", Some("www")));
        let d = registered_domain("foo.github.io").unwrap();
        assert_eq!((d.registered.as_deref(), d.etld.as_str()), (Some("github.io"), "io"));
        let d = registered_domain("co.uk").unwrap();
        assert!(d.registered.is_none() && d.etld == "co.uk");
        assert!(registered_domain("localhost").is_none());
        assert!(registered_domain("foo.zzz").is_none());
        let d = registered_domain("a.compute.amazonaws.com").unwrap();
        assert_eq!(d.registered.as_deref(), Some("a.compute.amazonaws.com"));
    }

    #[test]
    fn nets() {
        let ip: IpAddr = "fd00::1".parse().unwrap();
        assert!(in_named_network(&ip, "private").unwrap());
        let ip: IpAddr = "fc00::1".parse().unwrap();
        assert!(!in_named_network(&ip, "private").unwrap());
        assert!(inet_for_string("[::1]").is_err());
        assert!(matches!(inet_for_string("::ffff:8.8.8.8").unwrap(), IpAddr::V4(_)));
    }
}
