//! `geoip` / `ip_location` processors over MaxMind DB (`.mmdb`) files, as
//! ES's `ingest-geoip` module does it.
//!
//! Databases are read from disk (ES has no bundled databases either: they
//! are downloaded at runtime or dropped into `<config>/ingest-geoip`).
//! Searched, in order: every directory of `ES_GEOIP_DIR` (`:`-separated),
//! `$ES_PATH_CONF/ingest-geoip`, and `<data>/ingest-geoip` of the node
//! running the pipeline. Files are memory-loaded once and reloaded when
//! their mtime/size change. Without a database the processor tags the
//! document `_geoip_database_unavailable_<file>` like ES does before its
//! downloader has fetched the databases.

use super::doc::{type_name, IngestDocument};
use super::{Cfg, PResult, PipelineContext, Processor};
use crate::error::{EsError, Result};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime};

// ------------------------------------------------------------ MMDB reader

/// A decoded MaxMind DB value.
#[derive(Clone, Debug, PartialEq)]
pub enum MmVal {
    Str(String),
    F64(f64),
    F32(f32),
    Bytes(Vec<u8>),
    U64(u64),
    U128(u128),
    I32(i32),
    Bool(bool),
    Map(Vec<(String, MmVal)>),
    Arr(Vec<MmVal>),
}

impl MmVal {
    pub fn get(&self, k: &str) -> Option<&MmVal> {
        match self {
            MmVal::Map(m) => m.iter().find(|(x, _)| x == k).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn path(&self, p: &[&str]) -> Option<&MmVal> {
        let mut cur = self;
        for k in p {
            cur = cur.get(k)?;
        }
        Some(cur)
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MmVal::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MmVal::F64(f) => Some(*f),
            MmVal::F32(f) => Some(*f as f64),
            MmVal::U64(u) => Some(*u as f64),
            MmVal::I32(i) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MmVal::U64(u) => Some(*u),
            MmVal::I32(i) if *i >= 0 => Some(*i as u64),
            MmVal::U128(u) => Some(*u as u64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MmVal::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn to_json(&self) -> Value {
        match self {
            MmVal::Str(s) => json!(s),
            MmVal::F64(f) => json!(f),
            MmVal::F32(f) => json!(*f as f64),
            MmVal::Bytes(b) => json!(b),
            MmVal::U64(u) => json!(u),
            MmVal::U128(u) => json!(u.to_string()),
            MmVal::I32(i) => json!(i),
            MmVal::Bool(b) => json!(b),
            MmVal::Map(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()),
            MmVal::Arr(a) => Value::Array(a.iter().map(MmVal::to_json).collect()),
        }
    }
}

/// An in-memory MaxMind DB.
pub struct Mmdb {
    buf: Vec<u8>,
    node_count: u32,
    record_size: u32,
    ip_version: u32,
    tree_size: usize,
    pub database_type: String,
    pub build_epoch: u64,
    ipv4_start: u32,
    ipv4_depth: u32,
}

const META_MARKER: &[u8] = b"\xAB\xCD\xEFMaxMind.com";

fn bad(m: &str) -> EsError {
    EsError::new(500, "invalid_database_exception", m.to_string())
}

impl Mmdb {
    pub fn open(path: &Path) -> Result<Mmdb> {
        let buf = std::fs::read(path)?;
        Mmdb::from_bytes(buf)
    }

    pub fn from_bytes(buf: Vec<u8>) -> Result<Mmdb> {
        let start = buf.len().saturating_sub(128 * 1024);
        let pos = buf[start..]
            .windows(META_MARKER.len())
            .rposition(|w| w == META_MARKER)
            .map(|p| p + start + META_MARKER.len())
            .ok_or_else(|| bad("Could not find a MaxMind DB metadata marker in this file. Is this a valid MaxMind DB file?"))?;
        let (meta, _) = decode(&buf[pos..], 0, 0)?;
        let num = |k: &str| meta.get(k).and_then(MmVal::as_u64).ok_or_else(|| bad(&format!("missing metadata [{k}]")));
        let node_count = num("node_count")? as u32;
        let record_size = num("record_size")? as u32;
        let ip_version = num("ip_version")? as u32;
        if !matches!(record_size, 24 | 28 | 32) {
            return Err(bad(&format!("Unknown record size: {record_size}")));
        }
        let database_type = meta.get("database_type").and_then(MmVal::as_str).unwrap_or("").to_string();
        let build_epoch = meta.get("build_epoch").and_then(MmVal::as_u64).unwrap_or(0);
        let tree_size = (record_size as usize * 2 / 8) * node_count as usize;
        if tree_size + 16 > buf.len() {
            return Err(bad("The MaxMind DB file's search tree is corrupt"));
        }
        let mut db = Mmdb { buf, node_count, record_size, ip_version, tree_size, database_type, build_epoch, ipv4_start: 0, ipv4_depth: 0 };
        if ip_version == 6 {
            let mut node = 0u32;
            let mut depth = 0;
            while depth < 96 && node < node_count {
                node = db.record(node, 0);
                depth += 1;
            }
            db.ipv4_start = node;
            db.ipv4_depth = depth;
        }
        Ok(db)
    }

    fn record(&self, node: u32, bit: u32) -> u32 {
        let b = &self.buf;
        match self.record_size {
            24 => {
                let o = node as usize * 6 + bit as usize * 3;
                (b[o] as u32) << 16 | (b[o + 1] as u32) << 8 | b[o + 2] as u32
            }
            28 => {
                let o = node as usize * 7;
                if bit == 0 {
                    ((b[o + 3] as u32 & 0xF0) << 20) | (b[o] as u32) << 16 | (b[o + 1] as u32) << 8 | b[o + 2] as u32
                } else {
                    ((b[o + 3] as u32 & 0x0F) << 24) | (b[o + 4] as u32) << 16 | (b[o + 5] as u32) << 8 | b[o + 6] as u32
                }
            }
            _ => {
                let o = node as usize * 8 + bit as usize * 4;
                u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
            }
        }
    }

    /// The record for `ip` and the prefix length of its network.
    pub fn lookup(&self, ip: IpAddr) -> Result<Option<(MmVal, u32)>> {
        let (bytes, bits): (Vec<u8>, u32) = match ip {
            IpAddr::V4(v4) => (v4.octets().to_vec(), 32),
            IpAddr::V6(v6) => {
                if self.ip_version == 4 {
                    return Err(EsError::illegal_argument(format!(
                        "Error looking up {ip}. You attempted to look up an IPv6 address in an IPv4-only database."
                    )));
                }
                (v6.octets().to_vec(), 128)
            }
        };
        let (mut node, mut depth, offset_depth) = if bits == 32 && self.ip_version == 6 { (self.ipv4_start, 0u32, self.ipv4_depth) } else { (0, 0, 0) };
        while depth < bits && node < self.node_count {
            let bit = (bytes[(depth / 8) as usize] >> (7 - depth % 8)) & 1;
            node = self.record(node, bit as u32);
            depth += 1;
        }
        let _ = offset_depth;
        if node == self.node_count {
            return Ok(None);
        }
        if node < self.node_count {
            return Err(bad("Invalid node in search tree"));
        }
        let off = (node - self.node_count) as usize - 16;
        let data = &self.buf[self.tree_size + 16..];
        let (v, _) = decode(data, off, 0)?;
        Ok(Some((v, depth)))
    }
}

fn be(b: &[u8]) -> u64 {
    b.iter().fold(0u64, |a, x| a << 8 | *x as u64)
}

/// Decode the value at `pos` of a data section; returns it and the next position.
fn decode(d: &[u8], pos: usize, level: u32) -> Result<(MmVal, usize)> {
    if level > 64 {
        return Err(bad("MaxMind DB data nested too deeply"));
    }
    let get = |i: usize| d.get(i).copied().ok_or_else(|| bad("Unexpected end of MaxMind DB data section"));
    let slice = |a: usize, n: usize| d.get(a..a + n).ok_or_else(|| bad("Unexpected end of MaxMind DB data section"));
    let cb = get(pos)?;
    let mut p = pos + 1;
    let mut ty = cb >> 5;
    if ty == 1 {
        // pointer
        let ss = (cb >> 3) & 3;
        let vvv = (cb & 7) as usize;
        let (target, n) = match ss {
            0 => ((vvv << 8) | get(p)? as usize, 1),
            1 => (((vvv << 16) | be(slice(p, 2)?) as usize) + 2048, 2),
            2 => (((vvv << 24) | be(slice(p, 3)?) as usize) + 526_336, 3),
            _ => (be(slice(p, 4)?) as usize, 4),
        };
        let (v, _) = decode(d, target, level + 1)?;
        return Ok((v, p + n));
    }
    if ty == 0 {
        ty = 7 + get(p)?;
        p += 1;
    }
    let mut size = (cb & 0x1f) as usize;
    if size >= 29 {
        match size {
            29 => {
                size = 29 + get(p)? as usize;
                p += 1;
            }
            30 => {
                size = 285 + be(slice(p, 2)?) as usize;
                p += 2;
            }
            _ => {
                size = 65_821 + be(slice(p, 3)?) as usize;
                p += 3;
            }
        }
    }
    Ok(match ty {
        2 => (MmVal::Str(String::from_utf8_lossy(slice(p, size)?).into_owned()), p + size),
        3 => (MmVal::F64(f64::from_be_bytes(slice(p, 8)?.try_into().unwrap())), p + 8),
        4 => (MmVal::Bytes(slice(p, size)?.to_vec()), p + size),
        5 | 6 | 9 => (MmVal::U64(be(slice(p, size)?)), p + size),
        8 => {
            let raw = be(slice(p, size)?) as u32;
            (MmVal::I32(raw as i32), p + size)
        }
        10 => {
            let b = slice(p, size)?;
            (MmVal::U128(b.iter().fold(0u128, |a, x| a << 8 | *x as u128)), p + size)
        }
        7 => {
            let mut m = Vec::with_capacity(size);
            for _ in 0..size {
                let (k, np) = decode(d, p, level + 1)?;
                let (v, np) = decode(d, np, level + 1)?;
                p = np;
                let k = match k {
                    MmVal::Str(s) => s,
                    _ => return Err(bad("MaxMind DB map key is not a string")),
                };
                m.push((k, v));
            }
            (MmVal::Map(m), p)
        }
        11 => {
            let mut a = Vec::with_capacity(size);
            for _ in 0..size {
                let (v, np) = decode(d, p, level + 1)?;
                p = np;
                a.push(v);
            }
            (MmVal::Arr(a), p)
        }
        14 => (MmVal::Bool(size != 0), p),
        15 => (MmVal::F32(f32::from_be_bytes(slice(p, 4)?.try_into().unwrap())), p + 4),
        other => return Err(bad(&format!("Unknown MaxMind DB data type [{other}]"))),
    })
}

// ------------------------------------------------------------ database registry

struct Loaded {
    stamp: (Option<SystemTime>, u64),
    db: Arc<Mmdb>,
}

fn loaded() -> &'static Mutex<HashMap<PathBuf, Loaded>> {
    static L: OnceLock<Mutex<HashMap<PathBuf, Loaded>>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub hits_nanos: AtomicU64,
    pub misses_nanos: AtomicU64,
    pub evictions: AtomicU64,
}

const CACHE_SIZE: usize = 1000;

type CacheKey = (usize, String);

fn cache() -> &'static (Mutex<lru::LruCache<CacheKey, Option<(MmVal, u32)>>>, CacheStats) {
    static C: OnceLock<(Mutex<lru::LruCache<CacheKey, Option<(MmVal, u32)>>>, CacheStats)> = OnceLock::new();
    C.get_or_init(|| (Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(CACHE_SIZE).unwrap())), CacheStats::default()))
}

/// Directories searched for databases (`data` = the node's data dir).
pub fn search_dirs(data: Option<&Path>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(s) = std::env::var("ES_GEOIP_DIR") {
        v.extend(s.split(':').filter(|x| !x.is_empty()).map(PathBuf::from));
    }
    if let Ok(s) = std::env::var("ES_PATH_CONF") {
        v.push(Path::new(&s).join("ingest-geoip"));
    }
    if let Some(d) = data {
        v.push(d.join("ingest-geoip"));
    }
    v
}

/// The database `file` (e.g. `GeoLite2-City.mmdb`), when present.
pub fn database(file: &str, data: Option<&Path>) -> Option<Arc<Mmdb>> {
    if file.contains('/') || file.contains('\\') || file.contains("..") {
        return None;
    }
    for dir in search_dirs(data) {
        let p = dir.join(file);
        let Ok(md) = std::fs::metadata(&p) else { continue };
        if !md.is_file() {
            continue;
        }
        let stamp = (md.modified().ok(), md.len());
        let mut l = loaded().lock();
        if let Some(x) = l.get(&p) {
            if x.stamp == stamp {
                return Some(x.db.clone());
            }
        }
        match Mmdb::open(&p) {
            Ok(db) => {
                let db = Arc::new(db);
                l.insert(p, Loaded { stamp, db: db.clone() });
                return Some(db);
            }
            Err(e) => {
                tracing::warn!("failed to load geoip database [{}]: {e}", p.display());
                continue;
            }
        }
    }
    None
}

/// The `.mmdb` files available to a node: (name, dir, database type, build epoch).
pub fn available(data: Option<&Path>) -> Vec<(String, PathBuf, String, u64)> {
    let mut out: Vec<(String, PathBuf, String, u64)> = Vec::new();
    for dir in search_dirs(data) {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".mmdb")).collect();
        names.sort();
        for n in names {
            if out.iter().any(|x| x.0 == n) {
                continue;
            }
            if let Some(db) = database(&n, data) {
                out.push((n, dir.clone(), db.database_type.clone(), db.build_epoch));
            }
        }
    }
    out
}

/// `GET /_ingest/geoip/stats`.
pub fn stats_json(data: Option<&Path>, node_id: &str) -> Value {
    let (c, st) = cache();
    let count = c.lock().len();
    let dbs = available(data);
    let names: Vec<Value> = dbs.iter().map(|d| json!({"name": d.0})).collect();
    let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) / 1_000_000;
    let node = json!({
        "cache_stats": {
            "count": count,
            "hits": st.hits.load(Ordering::Relaxed),
            "misses": st.misses.load(Ordering::Relaxed),
            "evictions": st.evictions.load(Ordering::Relaxed),
            "hits_time_in_millis": ms(&st.hits_nanos),
            "misses_time_in_millis": ms(&st.misses_nanos)
        },
        "databases": names,
        "files_in_temp": [],
        "config_databases": dbs.iter().map(|d| json!(d.0)).collect::<Vec<_>>()
    });
    let mut nodes = Map::new();
    if !dbs.is_empty() {
        nodes.insert(node_id.to_string(), node);
    }
    json!({
        "stats": {
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_download_time": 0,
            "databases_count": dbs.len(),
            "skipped_updates": 0,
            "expired_databases": 0
        },
        "nodes": nodes
    })
}

/// Cached lookup.
fn cached_lookup(db: &Arc<Mmdb>, ip: IpAddr, ip_str: &str) -> Result<Option<(MmVal, u32)>> {
    let key = (Arc::as_ptr(db) as usize, ip_str.to_string());
    let (c, st) = cache();
    let t = Instant::now();
    if let Some(v) = c.lock().get(&key) {
        st.hits.fetch_add(1, Ordering::Relaxed);
        st.hits_nanos.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return Ok(v.clone());
    }
    let r = db.lookup(ip)?;
    let mut g = c.lock();
    if g.len() >= CACHE_SIZE && !g.contains(&key) {
        st.evictions.fetch_add(1, Ordering::Relaxed);
    }
    g.put(key, r.clone());
    st.misses.fetch_add(1, Ordering::Relaxed);
    st.misses_nanos.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(r)
}

// ------------------------------------------------------------ database kinds and properties

/// ES `Database.Property`, in enum order (also the output order here).
const PROPS: &[&str] = &[
    "IP",
    "COUNTRY_CONFIDENCE",
    "COUNTRY_IN_EUROPEAN_UNION",
    "COUNTRY_ISO_CODE",
    "COUNTRY_NAME",
    "CONTINENT_CODE",
    "CONTINENT_NAME",
    "REGION_ISO_CODE",
    "REGION_NAME",
    "CITY_CONFIDENCE",
    "CITY_NAME",
    "TIMEZONE",
    "LOCATION",
    "ASN",
    "ORGANIZATION_NAME",
    "NETWORK",
    "HOSTING_PROVIDER",
    "TOR_EXIT_NODE",
    "ANONYMOUS_VPN",
    "ANONYMOUS",
    "PUBLIC_PROXY",
    "RESIDENTIAL_PROXY",
    "DOMAIN",
    "ISP",
    "ISP_ORGANIZATION_NAME",
    "MOBILE_COUNTRY_CODE",
    "MOBILE_NETWORK_CODE",
    "CONNECTION_TYPE",
    "USER_TYPE",
    "TYPE",
    "POSTAL_CODE",
    "POSTAL_CONFIDENCE",
    "ACCURACY_RADIUS",
    "HOSTING",
    "TOR",
    "PROXY",
    "RELAY",
    "VPN",
    "SERVICE",
    "REGISTERED_COUNTRY_IN_EUROPEAN_UNION",
    "REGISTERED_COUNTRY_ISO_CODE",
    "REGISTERED_COUNTRY_NAME",
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    City,
    Country,
    Asn,
    AnonymousIp,
    ConnectionType,
    Domain,
    Enterprise,
    Isp,
}

impl Kind {
    fn from_type(t: &str) -> Option<Kind> {
        Some(if t.ends_with("-City") {
            Kind::City
        } else if t.ends_with("-Country") {
            Kind::Country
        } else if t.ends_with("-ASN") {
            Kind::Asn
        } else if t.ends_with("-Anonymous-IP") {
            Kind::AnonymousIp
        } else if t.ends_with("-Connection-Type") {
            Kind::ConnectionType
        } else if t.ends_with("-Domain") {
            Kind::Domain
        } else if t.ends_with("-Enterprise") {
            Kind::Enterprise
        } else if t.ends_with("-ISP") {
            Kind::Isp
        } else {
            return None;
        })
    }

    fn props(self) -> (&'static [&'static str], &'static [&'static str]) {
        const CITY_DEF: &[&str] = &["COUNTRY_ISO_CODE", "COUNTRY_NAME", "CONTINENT_NAME", "REGION_ISO_CODE", "REGION_NAME", "CITY_NAME", "LOCATION"];
        match self {
            Kind::City => (
                &[
                    "IP",
                    "COUNTRY_IN_EUROPEAN_UNION",
                    "COUNTRY_ISO_CODE",
                    "CONTINENT_CODE",
                    "COUNTRY_NAME",
                    "CONTINENT_NAME",
                    "REGION_ISO_CODE",
                    "REGION_NAME",
                    "CITY_NAME",
                    "TIMEZONE",
                    "LOCATION",
                    "POSTAL_CODE",
                    "ACCURACY_RADIUS",
                    "REGISTERED_COUNTRY_IN_EUROPEAN_UNION",
                    "REGISTERED_COUNTRY_ISO_CODE",
                    "REGISTERED_COUNTRY_NAME",
                ],
                CITY_DEF,
            ),
            Kind::Country => (
                &[
                    "IP",
                    "CONTINENT_CODE",
                    "CONTINENT_NAME",
                    "COUNTRY_NAME",
                    "COUNTRY_IN_EUROPEAN_UNION",
                    "COUNTRY_ISO_CODE",
                    "REGISTERED_COUNTRY_IN_EUROPEAN_UNION",
                    "REGISTERED_COUNTRY_ISO_CODE",
                    "REGISTERED_COUNTRY_NAME",
                ],
                &["CONTINENT_NAME", "COUNTRY_NAME", "COUNTRY_ISO_CODE"],
            ),
            Kind::Asn => (&["IP", "ASN", "ORGANIZATION_NAME", "NETWORK"], &["IP", "ASN", "ORGANIZATION_NAME", "NETWORK"]),
            Kind::AnonymousIp => (
                &["IP", "HOSTING_PROVIDER", "TOR_EXIT_NODE", "ANONYMOUS_VPN", "ANONYMOUS", "PUBLIC_PROXY", "RESIDENTIAL_PROXY"],
                &["HOSTING_PROVIDER", "TOR_EXIT_NODE", "ANONYMOUS_VPN", "ANONYMOUS", "PUBLIC_PROXY", "RESIDENTIAL_PROXY"],
            ),
            Kind::ConnectionType => (&["IP", "CONNECTION_TYPE"], &["CONNECTION_TYPE"]),
            Kind::Domain => (&["IP", "DOMAIN"], &["DOMAIN"]),
            Kind::Enterprise => (
                &[
                    "IP",
                    "COUNTRY_CONFIDENCE",
                    "COUNTRY_IN_EUROPEAN_UNION",
                    "COUNTRY_ISO_CODE",
                    "COUNTRY_NAME",
                    "CONTINENT_CODE",
                    "CONTINENT_NAME",
                    "REGION_ISO_CODE",
                    "REGION_NAME",
                    "CITY_CONFIDENCE",
                    "CITY_NAME",
                    "TIMEZONE",
                    "LOCATION",
                    "ASN",
                    "ORGANIZATION_NAME",
                    "NETWORK",
                    "HOSTING_PROVIDER",
                    "TOR_EXIT_NODE",
                    "ANONYMOUS_VPN",
                    "ANONYMOUS",
                    "PUBLIC_PROXY",
                    "RESIDENTIAL_PROXY",
                    "DOMAIN",
                    "ISP",
                    "ISP_ORGANIZATION_NAME",
                    "MOBILE_COUNTRY_CODE",
                    "MOBILE_NETWORK_CODE",
                    "USER_TYPE",
                    "CONNECTION_TYPE",
                    "POSTAL_CODE",
                    "POSTAL_CONFIDENCE",
                    "ACCURACY_RADIUS",
                    "REGISTERED_COUNTRY_IN_EUROPEAN_UNION",
                    "REGISTERED_COUNTRY_ISO_CODE",
                    "REGISTERED_COUNTRY_NAME",
                ],
                CITY_DEF,
            ),
            Kind::Isp => {
                const P: &[&str] = &["IP", "ASN", "ORGANIZATION_NAME", "NETWORK", "ISP", "ISP_ORGANIZATION_NAME", "MOBILE_COUNTRY_CODE", "MOBILE_NETWORK_CODE"];
                (P, P)
            }
        }
    }

    /// Parse `properties` (ES `Database.parseProperties`).
    fn parse(self, names: Option<&[String]>) -> std::result::Result<Vec<&'static str>, String> {
        let (valid, def) = self.props();
        let mut out: Vec<&'static str> = match names {
            None => def.to_vec(),
            Some(ns) => {
                let mut v = Vec::new();
                for n in ns {
                    let up = n.to_uppercase();
                    match valid.iter().find(|p| **p == up) {
                        Some(p) => v.push(*p),
                        None => {
                            let mut sorted: Vec<&str> = valid.to_vec();
                            sorted.sort_by_key(|p| PROPS.iter().position(|x| x == p));
                            return Err(format!("illegal property value [{n}]. valid values are [{}]", sorted.join(", ")));
                        }
                    }
                }
                v
            }
        };
        out.sort_by_key(|p| PROPS.iter().position(|x| x == p));
        out.dedup();
        Ok(out)
    }
}

fn en_name(v: &MmVal, rec: &str) -> Option<String> {
    v.path(&[rec, "names", "en"]).and_then(MmVal::as_str).map(str::to_string)
}

fn iso(v: &MmVal, rec: &str) -> Option<String> {
    v.path(&[rec, "iso_code"]).and_then(MmVal::as_str).map(str::to_string)
}

fn in_eu(v: &MmVal, rec: &str) -> Option<bool> {
    iso(v, rec)?;
    Some(v.path(&[rec, "is_in_european_union"]).and_then(MmVal::as_bool).unwrap_or(false))
}

/// Java `NetworkAddress.format` + `/prefix` of the network holding `ip`.
fn network(ip: IpAddr, prefix: u32) -> String {
    match ip {
        IpAddr::V4(v4) => {
            let p = prefix.min(32);
            let m = if p == 0 { 0 } else { u32::MAX << (32 - p) };
            format!("{}/{p}", std::net::Ipv4Addr::from(u32::from(v4) & m))
        }
        IpAddr::V6(v6) => {
            let p = prefix.min(128);
            let m = if p == 0 { 0 } else { u128::MAX << (128 - p) };
            format!("{}/{p}", std::net::Ipv6Addr::from(u128::from(v6) & m))
        }
    }
}

/// Java `NetworkAddress.format` (IPv4-mapped v6 shown as v4).
fn format_ip(ip: IpAddr) -> String {
    match ip {
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => v4.to_string(),
            None => v6.to_string(),
        },
        v4 => v4.to_string(),
    }
}

fn transform(kind: Kind, props: &[&str], rec: &MmVal, ip: IpAddr, prefix: u32) -> Map<String, Value> {
    let mut m = Map::new();
    let s = |p: &[&str]| rec.path(p).and_then(MmVal::as_str).map(|x| json!(x));
    let b = |k: &str| json!(rec.get(k).and_then(MmVal::as_bool).unwrap_or(false));
    let traits = |k: &str| rec.path(&["traits", k]);
    let is_asn_like = matches!(kind, Kind::Asn | Kind::Isp);
    for p in props {
        let (key, v): (&str, Option<Value>) = match *p {
            "IP" => ("ip", Some(json!(format_ip(ip)))),
            "COUNTRY_IN_EUROPEAN_UNION" => ("country_in_european_union", in_eu(rec, "country").map(|x| json!(x))),
            "COUNTRY_ISO_CODE" => ("country_iso_code", iso(rec, "country").map(|x| json!(x))),
            "COUNTRY_NAME" => ("country_name", en_name(rec, "country").map(|x| json!(x))),
            "COUNTRY_CONFIDENCE" => ("country_confidence", rec.path(&["country", "confidence"]).and_then(MmVal::as_u64).map(|x| json!(x))),
            "CONTINENT_CODE" => ("continent_code", s(&["continent", "code"])),
            "CONTINENT_NAME" => ("continent_name", en_name(rec, "continent").map(|x| json!(x))),
            "REGION_ISO_CODE" => {
                let sub = match rec.get("subdivisions") {
                    Some(MmVal::Arr(a)) => a.last(),
                    _ => None,
                };
                let v = match (iso(rec, "country"), sub.and_then(|x| x.get("iso_code")).and_then(MmVal::as_str)) {
                    (Some(c), Some(si)) => Some(json!(format!("{c}-{si}"))),
                    _ => None,
                };
                ("region_iso_code", v)
            }
            "REGION_NAME" => {
                let sub = match rec.get("subdivisions") {
                    Some(MmVal::Arr(a)) => a.last(),
                    _ => None,
                };
                ("region_name", sub.and_then(|x| x.path(&["names", "en"])).and_then(MmVal::as_str).map(|x| json!(x)))
            }
            "CITY_NAME" => ("city_name", en_name(rec, "city").map(|x| json!(x))),
            "CITY_CONFIDENCE" => ("city_confidence", rec.path(&["city", "confidence"]).and_then(MmVal::as_u64).map(|x| json!(x))),
            "TIMEZONE" => ("timezone", s(&["location", "time_zone"])),
            "LOCATION" => {
                let lat = rec.path(&["location", "latitude"]).and_then(MmVal::as_f64);
                let lon = rec.path(&["location", "longitude"]).and_then(MmVal::as_f64);
                ("location", match (lat, lon) {
                    (Some(a), Some(o)) => Some(json!({"lat": a, "lon": o})),
                    _ => None,
                })
            }
            "POSTAL_CODE" => ("postal_code", s(&["postal", "code"])),
            "POSTAL_CONFIDENCE" => ("postal_confidence", rec.path(&["postal", "confidence"]).and_then(MmVal::as_u64).map(|x| json!(x))),
            "ACCURACY_RADIUS" => ("accuracy_radius", rec.path(&["location", "accuracy_radius"]).and_then(MmVal::as_u64).map(|x| json!(x))),
            "REGISTERED_COUNTRY_IN_EUROPEAN_UNION" => ("registered_country_in_european_union", in_eu(rec, "registered_country").map(|x| json!(x))),
            "REGISTERED_COUNTRY_ISO_CODE" => ("registered_country_iso_code", iso(rec, "registered_country").map(|x| json!(x))),
            "REGISTERED_COUNTRY_NAME" => ("registered_country_name", en_name(rec, "registered_country").map(|x| json!(x))),
            "ASN" => (
                "asn",
                if is_asn_like { rec.get("autonomous_system_number") } else { traits("autonomous_system_number") }.and_then(MmVal::as_u64).map(|x| json!(x)),
            ),
            "ORGANIZATION_NAME" => (
                "organization_name",
                if is_asn_like { rec.get("autonomous_system_organization") } else { traits("autonomous_system_organization") }.and_then(MmVal::as_str).map(|x| json!(x)),
            ),
            "NETWORK" => ("network", Some(json!(network(ip, prefix)))),
            "HOSTING_PROVIDER" => ("hosting_provider", Some(if kind == Kind::Enterprise { json!(traits("is_hosting_provider").and_then(MmVal::as_bool).unwrap_or(false)) } else { b("is_hosting_provider") })),
            "TOR_EXIT_NODE" => ("tor_exit_node", Some(if kind == Kind::Enterprise { json!(traits("is_tor_exit_node").and_then(MmVal::as_bool).unwrap_or(false)) } else { b("is_tor_exit_node") })),
            "ANONYMOUS_VPN" => ("anonymous_vpn", Some(if kind == Kind::Enterprise { json!(traits("is_anonymous_vpn").and_then(MmVal::as_bool).unwrap_or(false)) } else { b("is_anonymous_vpn") })),
            "ANONYMOUS" => ("anonymous", Some(if kind == Kind::Enterprise { json!(traits("is_anonymous").and_then(MmVal::as_bool).unwrap_or(false)) } else { b("is_anonymous") })),
            "PUBLIC_PROXY" => ("public_proxy", Some(if kind == Kind::Enterprise { json!(traits("is_public_proxy").and_then(MmVal::as_bool).unwrap_or(false)) } else { b("is_public_proxy") })),
            "RESIDENTIAL_PROXY" => ("residential_proxy", Some(if kind == Kind::Enterprise { json!(traits("is_residential_proxy").and_then(MmVal::as_bool).unwrap_or(false)) } else { b("is_residential_proxy") })),
            "DOMAIN" => ("domain", if kind == Kind::Domain { s(&["domain"]) } else { s(&["traits", "domain"]) }),
            "ISP" => ("isp", if kind == Kind::Isp { s(&["isp"]) } else { s(&["traits", "isp"]) }),
            "ISP_ORGANIZATION_NAME" => ("isp_organization_name", if kind == Kind::Isp { s(&["organization"]) } else { s(&["traits", "organization"]) }),
            "MOBILE_COUNTRY_CODE" => ("mobile_country_code", if kind == Kind::Isp { s(&["mobile_country_code"]) } else { s(&["traits", "mobile_country_code"]) }),
            "MOBILE_NETWORK_CODE" => ("mobile_network_code", if kind == Kind::Isp { s(&["mobile_network_code"]) } else { s(&["traits", "mobile_network_code"]) }),
            "CONNECTION_TYPE" => ("connection_type", if kind == Kind::ConnectionType { s(&["connection_type"]) } else { s(&["traits", "connection_type"]) }),
            "USER_TYPE" => ("user_type", s(&["traits", "user_type"])),
            _ => continue,
        };
        if let Some(v) = v {
            m.insert(key.into(), v);
        }
    }
    m
}

// ------------------------------------------------------------ processor

pub struct GeoIpP {
    kind: String,
    field: String,
    target: String,
    database_file: String,
    properties: Option<Vec<String>>,
    ignore_missing: bool,
    first_only: bool,
}

fn ia(m: impl Into<String>) -> EsError {
    EsError::illegal_argument(m)
}

/// Data directory of the node whose pipeline store runs this processor.
fn data_dir(cx: &PipelineContext) -> Option<PathBuf> {
    let p = &cx.store.path;
    if p.as_os_str().is_empty() {
        return None;
    }
    p.parent().and_then(Path::parent).map(Path::to_path_buf)
}

impl GeoIpP {
    fn lookup(&self, db: &Arc<Mmdb>, kind: Kind, props: &[&str], s: &str) -> Result<Map<String, Value>> {
        let ip = super::util::parse_ip(s).ok_or_else(|| ia(format!("'{s}' is not an IP string literal.")))?;
        match cached_lookup(db, ip, s)? {
            None => Ok(Map::new()),
            Some((rec, prefix)) => Ok(transform(kind, props, &rec, ip, prefix)),
        }
    }
}

impl Processor for GeoIpP {
    fn process(&self, doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
        let data = data_dir(cx);
        let db = database(&self.database_file, data.as_deref());
        let tag = format!("_{}_database_unavailable_{}", self.kind, self.database_file);
        let Some(db) = db else {
            doc.append("tags", Value::String(tag), true)?;
            return Ok(());
        };
        let v = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None => return Err(doc.require(&self.field).err().unwrap().into()),
            Some(Value::Null) => return Err(ia(format!("field [{}] is null, cannot extract geoip information.", self.field)).into()),
            Some(v) => v.clone(),
        };
        let kind = Kind::from_type(&db.database_type)
            .ok_or_else(|| ia(format!("Unsupported database type [{}] for file [{}]", db.database_type, self.database_file)))?;
        let props = kind.parse(self.properties.as_deref()).map_err(ia)?;
        match v {
            Value::String(s) => {
                let data = self.lookup(&db, kind, &props, &s)?;
                if !data.is_empty() {
                    doc.set(&self.target, Value::Object(data))?;
                }
            }
            Value::Array(a) => {
                let mut matched = false;
                let mut list = Vec::with_capacity(a.len());
                for x in a {
                    let Value::String(s) = x else {
                        return Err(ia(format!("array in field [{}] should only contain strings", self.field)).into());
                    };
                    let data = self.lookup(&db, kind, &props, &s)?;
                    if data.is_empty() {
                        list.push(Value::Null);
                        continue;
                    }
                    if self.first_only {
                        doc.set(&self.target, Value::Object(data))?;
                        return Ok(());
                    }
                    matched = true;
                    list.push(Value::Object(data));
                }
                if matched {
                    doc.set(&self.target, Value::Array(list))?;
                }
            }
            other => {
                let _ = type_name(&other);
                return Err(ia(format!("field [{}] should contain only string or array of strings", self.field)).into());
            }
        }
        Ok(())
    }
}

/// Build a `geoip` / `ip_location` processor (ES `GeoIpProcessor.Factory`).
pub fn build(cfg: &mut Cfg, kind: &str) -> Result<Box<dyn Processor>> {
    let kind = if kind == "geo_ip" { "geoip" } else { kind };
    let field = cfg.req_str("field")?;
    let target = cfg.str_or("target_field", kind)?;
    let database_file = cfg.str_or("database_file", "GeoLite2-City.mmdb")?;
    let properties = cfg.opt_list_str("properties")?;
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    let first_only = cfg.bool_or("first_only", true)?;
    cfg.bool_or("download_database_on_pipeline_creation", true)?;
    cfg.ignore(&["fallback_to_default_databases"]);
    // Validate against the database when it can be found now (config dirs),
    // else against the type implied by the file name.
    let db_type = match database(&database_file, None) {
        Some(db) => Some(db.database_type.clone()),
        None => Some(database_file.trim_end_matches(".mmdb").to_string()),
    };
    if let Some(t) = db_type {
        let found = database(&database_file, None).is_some();
        match Kind::from_type(&t) {
            Some(k) => {
                if let Err(e) = k.parse(properties.as_deref()) {
                    return Err(cfg.err("properties", e));
                }
            }
            None if found => {
                return Err(cfg.err("database_file", format!("Unsupported database type [{t}] for file [{database_file}]")));
            }
            None => {}
        }
    }
    Ok(Box::new(GeoIpP { kind: kind.to_string(), field, target, database_file, properties, ignore_missing, first_only }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_and_format() {
        assert_eq!(network("89.160.20.128".parse().unwrap(), 17), "89.160.0.0/17");
        assert_eq!(format_ip("::ffff:1.2.3.4".parse().unwrap()), "1.2.3.4");
        assert!(Kind::City.parse(Some(&["nope".to_string()])).unwrap_err().starts_with("illegal property value [nope]. valid values are [IP, COUNTRY_IN_EUROPEAN_UNION, COUNTRY_ISO_CODE, COUNTRY_NAME"));
    }

    #[test]
    fn decode_basic_types() {
        // map{ "a": "xy", "b": uint16 300, "c": true }
        let d = [0xE3u8, 0x41, b'a', 0x42, b'x', b'y', 0x41, b'b', 0xA2, 0x01, 0x2C, 0x41, b'c', 0x01, 0x07];
        let (v, _) = decode(&d, 0, 0).unwrap();
        assert_eq!(v.get("a").and_then(MmVal::as_str), Some("xy"));
        assert_eq!(v.get("b").and_then(MmVal::as_u64), Some(300));
        assert_eq!(v.get("c").and_then(MmVal::as_bool), Some(true));
    }
}

// ------------------------------------------------------------ database configurations

/// `PUT/GET/DELETE /_ingest/{geoip,ip_location}/database/{id}`.
fn configs() -> &'static Mutex<std::collections::BTreeMap<(PathBuf, String), (Value, i64, i64)>> {
    static C: OnceLock<Mutex<std::collections::BTreeMap<(PathBuf, String), (Value, i64, i64)>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(std::collections::BTreeMap::new()))
}

const MAXMIND_NAMES: &[&str] = &["GeoIP2-Enterprise", "GeoIP2-ISP", "GeoIP2-Domain", "GeoIP2-Connection-Type", "GeoIP2-City", "GeoIP2-Anonymous-IP", "GeoIP2-Country"];
const IPINFO_NAMES: &[&str] = &["standard_asn", "standard_location", "standard_privacy", "country", "asn"];

/// 1-based line/column helpers over the raw request text (ES reports parser locations).
fn col_of(raw: &str, idx: usize) -> usize {
    raw[..idx.min(raw.len())].chars().count() + 1
}

/// For the top-level object of `raw`: `(key, index of the key's opening quote, value start, value end exclusive)`.
fn top_level_members(raw: &str) -> Vec<(String, usize, usize, usize)> {
    let b = raw.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() && b[i] != b'{' {
        i += 1;
    }
    i += 1;
    let skip_ws = |mut i: usize| {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        i
    };
    let value_end = |mut i: usize| -> usize {
        let mut depth = 0i32;
        let mut in_str = false;
        while i < b.len() {
            let c = b[i];
            if in_str {
                if c == b'\\' {
                    i += 1;
                } else if c == b'"' {
                    in_str = false;
                    if depth == 0 {
                        return i + 1;
                    }
                }
            } else {
                match c {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        if depth == 0 {
                            return i;
                        }
                        depth -= 1;
                        if depth == 0 {
                            return i + 1;
                        }
                    }
                    b',' if depth == 0 => return i,
                    _ => {}
                }
            }
            i += 1;
        }
        i
    };
    loop {
        i = skip_ws(i);
        if i >= b.len() || b[i] != b'"' {
            break;
        }
        let ks = i + 1;
        let mut j = ks;
        while j < b.len() && b[j] != b'"' {
            if b[j] == b'\\' {
                j += 1;
            }
            j += 1;
        }
        let key = raw[ks..j.min(raw.len())].to_string();
        let key_start = ks - 1;
        let after_key = j + 1;
        i = skip_ws(after_key);
        if i < b.len() && b[i] == b':' {
            i += 1;
        }
        i = skip_ws(i);
        let vs = i;
        let ve = value_end(i);
        out.push((key, key_start, vs, ve));
        i = skip_ws(ve);
        if i < b.len() && b[i] == b',' {
            i += 1;
        } else {
            break;
        }
    }
    out
}

pub fn put_config(data: &Path, id: &str, body: &Value, raw: &str, _ip_location: bool) -> Result<()> {
    let xc = |reason: String| EsError::new(400, "x_content_parse_exception", reason);
    let o = body.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "request body is required"))?;
    let members = top_level_members(raw);
    let known = ["name", "maxmind", "ipinfo", "local", "web"];
    for (k, key_start, _, _) in &members {
        if !known.contains(&k.as_str()) {
            return Err(xc(format!("[1:{}] [database] unknown field [{k}]", col_of(raw, *key_start))));
        }
    }
    // provider objects: maxmind needs account_id
    if let Some(mm) = o.get("maxmind") {
        let has = mm.get("account_id").map(|v| v.is_string()).unwrap_or(false);
        if !has {
            let end = members.iter().find(|m| m.0 == "maxmind").map(|m| m.3.saturating_sub(1)).unwrap_or(0);
            return Err(xc(format!("[1:{}] [database] failed to parse field [maxmind]", col_of(raw, end))).caused_by(ia("Required [account_id]")));
        }
    }
    if o.get("name").and_then(Value::as_str).is_none() {
        return Err(ia("Required [name]"));
    }
    let providers: Vec<&str> = ["maxmind", "ipinfo", "local", "web"].into_iter().filter(|k| o.contains_key(*k)).collect();
    if providers.len() != 1 {
        return Err(xc("Failed to build [database] after last required field arrived".into()).caused_by(ia(format!("Exactly one provider object must be specified, but [{}] were found", providers.len()))));
    }
    if matches!(providers[0], "local" | "web") {
        return Err(ia(format!("Database {id} is read only")));
    }
    let name = o["name"].as_str().unwrap_or("");
    let mut errs: Vec<String> = Vec::new();
    if id.len() > 255 {
        errs.push(format!("invalid database configuration id [{id}]: index name is too long, ({} > 255)", id.len()));
    }
    let names: &[&str] = if providers[0] == "maxmind" { MAXMIND_NAMES } else { IPINFO_NAMES };
    if !names.contains(&name) {
        errs.push(format!("invalid name [{name}]: must be a supported name ([[{}]])", names.join(", ")));
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let now = crate::date::now_millis();
    let mut c = configs().lock();
    let key = (data.to_path_buf(), id.to_string());
    if let Some(existing) = c.get(&key) {
        if &existing.0 == body {
            return Ok(()); // unchanged configurations keep their version
        }
    }
    let version = c.get(&key).map(|x| x.2 + 1).unwrap_or(1);
    c.insert(key, (body.clone(), now, version));
    Ok(())
}

pub fn get_configs(data: &Path, ids: Option<&str>) -> Result<Value> {
    let c = configs().lock();
    let render = |id: &str, v: &(Value, i64, i64)| json!({"id": id, "version": v.2, "modified_date_millis": v.1, "database": v.0});
    let pats: Vec<&str> = ids.map(|l| l.split(',').map(str::trim).filter(|s| !s.is_empty()).collect()).unwrap_or_default();
    let mut out = Vec::new();
    for ((d, id), v) in c.iter() {
        if d == data && (pats.is_empty() || pats.iter().any(|p| *p == "*" || crate::mapping::wildcard_match(p, id))) {
            out.push(render(id, v));
        }
    }
    Ok(json!({ "databases": out }))
}

pub fn delete_config(data: &Path, id: &str) -> Result<()> {
    match configs().lock().remove(&(data.to_path_buf(), id.to_string())) {
        Some(_) => Ok(()),
        None => Err(EsError::resource_not_found(format!("Database configuration not found: {id}"))),
    }
}
