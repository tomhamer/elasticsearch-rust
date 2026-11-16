//! GeoIP database downloads (ES `GeoIpDownloader`,
//! `EnterpriseGeoIpDownloader` and `DatabaseNodeService`; findings 446).
//!
//! Downloading and installing are separate, as in ES:
//!
//! - The downloader runs on the elected master (on a single node: the
//!   node). It fetches the default databases from
//!   `ingest.geoip.downloader.endpoint`
//!   (`?elastic_geoip_service_tos=agree`, `.tgz` archives checked against
//!   their MD5) and the databases configured through
//!   `_ingest/{geoip,ip_location}/database` (MaxMind accounts with the
//!   `ingest.geoip.downloader.maxmind.license_key` secure setting, IPinfo
//!   with `ingest.ip_location.downloader.ipinfo.token`, checked against
//!   their SHA-256). Each database is stored in the `.geoip_databases`
//!   system index as 1 MiB chunks, with a state document recording every
//!   database's checksum and chunk range. It runs when it is needed: with
//!   `ingest.geoip.downloader.eager.download`, a pipeline whose GeoIP / IP
//!   location processor downloads on creation (the default), or a
//!   configured database; and again every
//!   `ingest.geoip.downloader.poll.interval` (default 3d).
//! - Every node pulls the chunks of a database whose checksum differs from
//!   its installed copy, verifies and extracts it, and installs it in its
//!   `ingest-geoip` directory, where the processors load it.
//!
//! Pipelines installed by the stack's own registry (`_meta.managed`) do
//! not start the downloader.

use super::geoip::Mmdb;
use crate::error::EsError;
use crate::node::Node;
use base64::Engine as _;
use serde_json::{json, Value};
use sha2::Digest as _;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

pub const INDEX: &str = ".geoip_databases";
const DEFAULT_ENDPOINT: &str = "https://geoip.elastic.co/v1/database";
const MAXMIND_ENDPOINT: &str = "https://download.maxmind.com/geoip/databases";
const IPINFO_ENDPOINT: &str = "https://ipinfo.io/data";
/// The state document of `.geoip_databases`.
const STATE_ID: &str = "_geoip_downloader_state";
/// Chunk size of a stored database (ES: 1 MiB).
const CHUNK: usize = 1 << 20;
/// Largest database archive downloaded.
const MAX_ARCHIVE_BYTES: usize = 512 << 20;

/// The `.geoip_databases` index, on this node or through the cluster.
pub trait DbStore {
    /// Create the index (no-op when it exists).
    fn ensure_index(&self) -> Result<(), EsError>;
    /// Apply NDJSON bulk operations (refreshed).
    fn bulk(&self, ndjson: String) -> Result<(), EsError>;
    /// The `_source` of document `id`.
    fn get(&self, id: &str) -> Result<Option<Value>, EsError>;
    /// A search response; `None` when the index does not exist.
    fn search(&self, body: Value) -> Result<Option<Value>, EsError>;
}

fn index_body() -> Value {
    json!({
        "settings": {"index": {"hidden": true, "number_of_shards": 1, "auto_expand_replicas": "0-1"}},
        "mappings": {"dynamic": "strict", "properties": {
            "name": {"type": "keyword"},
            "chunk": {"type": "integer"},
            "data": {"type": "binary"},
            "digest": {"type": "keyword"},
            "state": {"type": "object", "enabled": false}
        }}
    })
}

/// [`DbStore`] on a single node's engine.
pub struct LocalStore<'a>(pub &'a Node);

impl DbStore for LocalStore<'_> {
    fn ensure_index(&self) -> Result<(), EsError> {
        if self.0.get_index(INDEX).is_some() {
            return Ok(());
        }
        match self.0.create_index(INDEX, &index_body()) {
            Ok(_) => Ok(()),
            Err(e) if e.kind == "resource_already_exists_exception" => Ok(()),
            Err(e) => Err(e),
        }
    }
    fn bulk(&self, ndjson: String) -> Result<(), EsError> {
        let svc = self.0.get_index(INDEX).ok_or_else(|| EsError::index_not_found(INDEX))?;
        let mut reqs = Vec::new();
        let mut lines = ndjson.lines().filter(|l| !l.trim().is_empty());
        while let Some(action) = lines.next() {
            let a: Value = serde_json::from_str(action)?;
            if let Some(id) = a.pointer("/delete/_id").and_then(Value::as_str) {
                reqs.push(crate::index::WriteReq::Delete { id: id.to_string(), routing: None, cond: Default::default() });
            } else {
                let id = a.pointer("/index/_id").and_then(Value::as_str).map(str::to_string);
                let source = lines.next().ok_or_else(|| EsError::illegal_argument("bulk action without a source"))?.to_string();
                reqs.push(crate::index::WriteReq::Index { id, source, routing: None, cond: Default::default() });
            }
        }
        for r in svc.write(reqs, true) {
            match r {
                Ok(_) => {}
                Err(e) if e.status == 404 => {}
                Err(e) => return Err(e),
            }
        }
        svc.refresh()
    }
    fn get(&self, id: &str) -> Result<Option<Value>, EsError> {
        let Some(svc) = self.0.get_index(INDEX) else { return Ok(None) };
        Ok(match svc.get(id)?.and_then(|g| g.source) {
            Some(s) => Some(serde_json::from_str(&s)?),
            None => None,
        })
    }
    fn search(&self, body: Value) -> Result<Option<Value>, EsError> {
        let Some(svc) = self.0.get_index(INDEX) else { return Ok(None) };
        let req = crate::search::parse_request(&body, &HashMap::new())?;
        let out = crate::search::execute(&[crate::search::Target::new(svc)], &req, &crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None })?;
        Ok(Some(serde_json::from_str(&out.json)?))
    }
}

/// [`DbStore`] through a cluster coordinator (`_system` calls).
pub struct RestStore(pub crate::watcher::RestCall);

impl DbStore for RestStore {
    fn ensure_index(&self) -> Result<(), EsError> {
        match (self.0)("PUT", &format!("/{INDEX}"), Some(index_body())) {
            Ok(_) => Ok(()),
            Err(e) if e.kind == "resource_already_exists_exception" => Ok(()),
            Err(e) => Err(e),
        }
    }
    fn bulk(&self, ndjson: String) -> Result<(), EsError> {
        let v = (self.0)("POST", &format!("/{INDEX}/_bulk?refresh=true"), Some(Value::String(ndjson)))?;
        if v["errors"] == true {
            let first = v["items"].as_array().into_iter().flatten().find_map(|i| i.as_object()?.values().next()?.get("error").filter(|e| e["type"] != "document_missing_exception").cloned());
            if let Some(e) = first {
                return Err(EsError::new(500, "exception", format!("writing [{INDEX}] failed: {e}")));
            }
        }
        Ok(())
    }
    fn get(&self, id: &str) -> Result<Option<Value>, EsError> {
        match (self.0)("GET", &format!("/{INDEX}/_doc/{id}"), None) {
            Ok(v) if v["found"] == true => Ok(Some(v["_source"].clone())),
            Ok(_) => Ok(None),
            Err(e) if e.status == 404 => Ok(None),
            Err(e) => Err(e),
        }
    }
    fn search(&self, body: Value) -> Result<Option<Value>, EsError> {
        match (self.0)("POST", &format!("/{INDEX}/_search"), Some(body)) {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.status == 404 => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// Where and how the downloader fetches.
pub struct Sources<'a> {
    /// The default-database endpoint (`None`: default databases are not
    /// downloaded).
    pub endpoint: Option<String>,
    pub maxmind_endpoint: String,
    pub ipinfo_endpoint: String,
    /// Configured databases `(id, database)`.
    pub configured: Vec<(String, Value)>,
    /// A secure setting.
    pub secret: &'a dyn Fn(&str) -> Option<String>,
}

/// The stored state: file name -> entry.
fn state(store: &dyn DbStore) -> Result<BTreeMap<String, Value>, EsError> {
    Ok(store.get(STATE_ID)?.and_then(|v| v["state"]["databases"].as_object().cloned()).map(|m| m.into_iter().collect()).unwrap_or_default())
}

fn put_state(store: &dyn DbStore, dbs: &BTreeMap<String, Value>) -> Result<(), EsError> {
    let doc = json!({"state": {"databases": dbs}});
    store.bulk(format!("{}\n{}\n", json!({"index": {"_id": STATE_ID}}), doc))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// One database to (maybe) download.
struct Wanted {
    file: String,
    digest: String,
    /// `md5` or `sha256`.
    algo: &'static str,
    /// `tgz` or `mmdb`.
    format: &'static str,
    url: String,
    headers: Vec<(String, String)>,
}

/// Download what changed into `store`; the number of databases stored.
pub fn run_downloader(store: &dyn DbStore, src: &Sources) -> Result<usize, String> {
    let client = crate::inference::providers::HttpClient::default();
    let get = |url: &str, headers: &[(String, String)], max: usize| -> Result<Vec<u8>, String> {
        let (status, body) = client.get_bytes(url, headers, Duration::from_secs(600), max, 3)?;
        if status != 200 {
            return Err(format!("[{}] returned status [{status}]", url.split('?').next().unwrap_or(url)));
        }
        Ok(body)
    };
    let mut wanted: Vec<Wanted> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    if let Some(endpoint) = &src.endpoint {
        let sep = if endpoint.contains('?') { '&' } else { '?' };
        match get(&format!("{endpoint}{sep}elastic_geoip_service_tos=agree"), &[], 16 << 20).and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| format!("invalid database overview: {e}"))) {
            Ok(overview) => {
                for entry in overview.as_array().into_iter().flatten() {
                    let (Some(name), Some(md5), Some(url)) = (entry["name"].as_str(), entry["md5_hash"].as_str(), entry["url"].as_str()) else { continue };
                    let Some(stem) = name.strip_suffix(".tgz").filter(|s| valid_name(s)) else { continue };
                    wanted.push(Wanted { file: format!("{stem}.mmdb"), digest: md5.to_ascii_lowercase(), algo: "md5", format: "tgz", url: url.to_string(), headers: Vec::new() });
                }
            }
            Err(e) => errors.push(e),
        }
    }
    for (id, db) in &src.configured {
        let Some(name) = db["name"].as_str().filter(|n| valid_name(n)) else { continue };
        if let Some(mm) = db.get("maxmind") {
            let account = mm["account_id"].as_str().unwrap_or("");
            let Some(key) = (src.secret)("ingest.geoip.downloader.maxmind.license_key") else {
                errors.push(format!("database [{id}]: no [ingest.geoip.downloader.maxmind.license_key] secure setting"));
                continue;
            };
            let auth = vec![("Authorization".to_string(), format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{account}:{key}"))))];
            let base = format!("{}/{name}/download?suffix=", src.maxmind_endpoint.trim_end_matches('/'));
            match get(&format!("{base}tar.gz.sha256"), &auth, 1 << 16) {
                Ok(b) => {
                    let text = String::from_utf8_lossy(&b);
                    let Some(sha) = text.split_whitespace().next().map(str::to_ascii_lowercase) else { continue };
                    wanted.push(Wanted { file: format!("{name}.mmdb"), digest: sha, algo: "sha256", format: "tgz", url: format!("{base}tar.gz"), headers: auth });
                }
                Err(e) => errors.push(format!("database [{id}]: {e}")),
            }
        } else if db.get("ipinfo").is_some() {
            let Some(token) = (src.secret)("ingest.ip_location.downloader.ipinfo.token") else {
                errors.push(format!("database [{id}]: no [ingest.ip_location.downloader.ipinfo.token] secure setting"));
                continue;
            };
            let base = format!("{}/{name}.mmdb", src.ipinfo_endpoint.trim_end_matches('/'));
            match get(&format!("{base}/checksums?token={token}"), &[], 1 << 16).and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string())) {
                Ok(v) => {
                    let Some(sha) = v["checksums"]["sha256"].as_str().map(str::to_ascii_lowercase) else {
                        errors.push(format!("database [{id}]: no sha256 checksum"));
                        continue;
                    };
                    wanted.push(Wanted { file: format!("{name}.mmdb"), digest: sha, algo: "sha256", format: "mmdb", url: format!("{base}?token={token}"), headers: Vec::new() });
                }
                Err(e) => errors.push(format!("database [{id}]: {e}")),
            }
        }
    }
    let mut dbs = state(store).map_err(|e| e.reason)?;
    let now = crate::date::now_millis();
    let mut stored = 0;
    for w in wanted {
        if dbs.get(&w.file).and_then(|e| e["digest"].as_str()) == Some(w.digest.as_str()) {
            if let Some(e) = dbs.get_mut(&w.file) {
                e["last_check"] = json!(now);
            }
            continue;
        }
        let bytes = match get(&w.url, &w.headers, MAX_ARCHIVE_BYTES) {
            Ok(b) => b,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };
        let got = digest(w.algo, &bytes);
        if got != w.digest {
            errors.push(format!("{} checksum mismatch for [{}], expected [{}] but was [{got}]", w.algo, w.file, w.digest));
            continue;
        }
        // Check the payload before publishing it to the nodes.
        if let Err(e) = extract(w.format, &w.file, &bytes) {
            errors.push(e);
            continue;
        }
        store.ensure_index().map_err(|e| e.reason)?;
        let chunks: Vec<&[u8]> = bytes.chunks(CHUNK).collect();
        for (n, group) in chunks.chunks(8).enumerate() {
            let mut body = String::new();
            for (k, c) in group.iter().enumerate() {
                let i = n * 8 + k;
                body.push_str(&json!({"index": {"_id": format!("{}_{i}_{}", w.file, &w.digest[..16.min(w.digest.len())])}}).to_string());
                body.push('\n');
                body.push_str(&json!({"name": w.file, "chunk": i, "digest": w.digest, "data": base64::engine::general_purpose::STANDARD.encode(c)}).to_string());
                body.push('\n');
            }
            store.bulk(body).map_err(|e| e.reason)?;
        }
        let old = dbs.insert(w.file.clone(), json!({"digest": w.digest, "algo": w.algo, "format": w.format, "first_chunk": 0, "last_chunk": chunks.len() as i64 - 1, "last_update": now, "last_check": now}));
        put_state(store, &dbs).map_err(|e| e.reason)?;
        stored += 1;
        // The previous generation's chunks.
        if let Some(old) = old.and_then(|o| o["digest"].as_str().map(str::to_string)) {
            let ids = chunk_ids(store, &w.file, &old).map_err(|e| e.reason)?;
            if !ids.is_empty() {
                store.bulk(ids.iter().map(|id| format!("{}\n", json!({"delete": {"_id": id}}))).collect()).map_err(|e| e.reason)?;
            }
        }
    }
    put_state(store, &dbs).map_err(|e| e.reason)?;
    if stored == 0 && !errors.is_empty() {
        return Err(errors.join("; "));
    }
    for e in errors {
        tracing::warn!("geoip download: {e}");
    }
    Ok(stored)
}

fn valid_name(s: &str) -> bool {
    !s.is_empty() && !s.contains(['/', '\\']) && !s.contains("..")
}

fn digest(algo: &str, bytes: &[u8]) -> String {
    match algo {
        "sha256" => hex(&sha2::Sha256::digest(bytes)),
        _ => hex(&md5_digest(bytes)),
    }
}

/// The `.mmdb` bytes of a downloaded payload.
fn extract(format: &str, file: &str, bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mmdb = match format {
        "mmdb" => bytes.to_vec(),
        _ => {
            let mut tar = Vec::new();
            flate2::read::GzDecoder::new(bytes).read_to_end(&mut tar).map_err(|e| format!("[{file}] archive is not gzip: {e}"))?;
            tar_member(&tar, file).ok_or_else(|| format!("archive does not contain [{file}]"))?
        }
    };
    Mmdb::from_bytes(mmdb.clone()).map_err(|e| format!("[{file}] is not a valid database: {}", e.reason))?;
    Ok(mmdb)
}

/// Ids of the chunks of `file` with `digest`.
fn chunk_ids(store: &dyn DbStore, file: &str, digest: &str) -> Result<Vec<String>, EsError> {
    let body = json!({"size": 10_000, "_source": false, "query": {"bool": {"filter": [{"term": {"name": file}}, {"term": {"digest": digest}}]}}});
    Ok(store.search(body)?.map(|v| v["hits"]["hits"].as_array().into_iter().flatten().filter_map(|h| h["_id"].as_str().map(str::to_string)).collect()).unwrap_or_default())
}

/// Install in `dir` every stored database whose checksum differs from the
/// installed copy; the number installed.
pub fn sync_node(store: &dyn DbStore, dir: &Path) -> Result<usize, String> {
    let dbs = state(store).map_err(|e| e.reason)?;
    let mut installed = 0;
    for (file, e) in dbs {
        let (Some(digest_want), Some(last)) = (e["digest"].as_str(), e["last_chunk"].as_i64()) else { continue };
        if !valid_name(&file) {
            continue;
        }
        let marker = dir.join(format!(".{file}.digest"));
        if std::fs::read_to_string(&marker).ok().as_deref() == Some(digest_want) && dir.join(&file).is_file() {
            continue;
        }
        let mut bytes = Vec::new();
        let mut after: Option<Value> = None;
        loop {
            let mut body = json!({"size": 8, "sort": [{"chunk": "asc"}], "query": {"bool": {"filter": [{"term": {"name": file}}, {"term": {"digest": digest_want}}]}}});
            if let Some(a) = &after {
                body["search_after"] = a.clone();
            }
            let Some(v) = store.search(body).map_err(|e| e.reason)? else { return Ok(installed) };
            let hits = v["hits"]["hits"].as_array().cloned().unwrap_or_default();
            for h in &hits {
                let data = h["_source"]["data"].as_str().unwrap_or("");
                bytes.extend(base64::engine::general_purpose::STANDARD.decode(data).map_err(|e| format!("chunk of [{file}]: {e}"))?);
            }
            match hits.last() {
                Some(h) if hits.len() == 8 => after = Some(h["sort"].clone()),
                _ => break,
            }
        }
        let algo = e["algo"].as_str().unwrap_or("md5");
        if digest(algo, &bytes) != digest_want {
            // Chunks still being replaced: the next pass picks it up.
            tracing::debug!("geoip database [{file}]: stored chunks ({} of {}) do not match the recorded checksum yet", bytes.len(), last + 1);
            continue;
        }
        let mmdb = extract(e["format"].as_str().unwrap_or("tgz"), &file, &bytes)?;
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let tmp = dir.join(format!(".{file}.tmp"));
        std::fs::write(&tmp, &mmdb).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, dir.join(&file)).map_err(|e| e.to_string())?;
        std::fs::write(&marker, digest_want).map_err(|e| e.to_string())?;
        installed += 1;
    }
    Ok(installed)
}

// ------------------------------------------------------------------ scheduling

/// A cluster schedules downloads itself ([`master_tick`] / [`sync_node`]
/// on every node): the single-node trigger does nothing.
static CLUSTERED: AtomicBool = AtomicBool::new(false);

pub fn set_clustered() {
    CLUSTERED.store(true, Ordering::SeqCst);
}

struct Entry {
    node: Weak<Node>,
    running: bool,
}

fn nodes() -> &'static Mutex<HashMap<PathBuf, Entry>> {
    static N: OnceLock<Mutex<HashMap<PathBuf, Entry>>> = OnceLock::new();
    N.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A dynamic cluster setting, else the node's own (`endpoint` and the
/// provider endpoints are node settings).
fn setting(node: &Node, key: &str) -> Option<String> {
    crate::ilm::cluster_setting(node, key).or_else(|| match node.node_settings.read().get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(other.to_string()),
    })
}

/// Make `node`'s settings available to downloads for its data directory;
/// with `ingest.geoip.downloader.eager.download`, start downloading.
pub fn register(node: &Arc<Node>) {
    nodes().lock().unwrap_or_else(|e| e.into_inner()).insert(node.data.clone(), Entry { node: Arc::downgrade(node), running: false });
    if setting(node, "ingest.geoip.downloader.eager.download").as_deref() == Some("true") {
        trigger(&node.data);
    }
}

/// Whether a pipeline definition has a GeoIP / IP location processor
/// asking for its database to be downloaded.
pub fn wants_download(pipeline: &Value) -> bool {
    fn walk(v: &Value) -> bool {
        match v {
            Value::Object(o) => o.iter().any(|(k, c)| {
                (matches!(k.as_str(), "geoip" | "ip_location") && c.get("download_database_on_pipeline_creation").and_then(|d| d.as_bool().or_else(|| d.as_str().map(|s| s != "false"))).unwrap_or(true))
                    || walk(c)
            }),
            Value::Array(a) => a.iter().any(walk),
            _ => false,
        }
    }
    pipeline.pointer("/_meta/managed").and_then(Value::as_bool) != Some(true) && walk(pipeline)
}

/// Whether the downloader has work on `node`: eager downloads, a GeoIP
/// pipeline, or a configured database.
pub fn needed(node: &Node) -> bool {
    setting(node, "ingest.geoip.downloader.eager.download").as_deref() == Some("true")
        || node.ingest().get(None).iter().any(|(_, p)| wants_download(p))
        || !super::geoip::configured(&node.data).is_empty()
}

fn poll_interval(node: &Node) -> Duration {
    setting(node, "ingest.geoip.downloader.poll.interval")
        .and_then(|s| crate::date::parse_time_value(&s))
        .map(|ms| Duration::from_millis(ms.max(60_000) as u64))
        .unwrap_or(Duration::from_secs(3 * 24 * 3600))
}

/// The downloader's sources on `node`.
pub fn sources<'a>(node: &Node, secret: &'a dyn Fn(&str) -> Option<String>) -> Sources<'a> {
    let enabled = setting(node, "ingest.geoip.downloader.enabled").as_deref() != Some("false");
    let default_needed = setting(node, "ingest.geoip.downloader.eager.download").as_deref() == Some("true") || node.ingest().get(None).iter().any(|(_, p)| wants_download(p));
    Sources {
        endpoint: (enabled && default_needed).then(|| setting(node, "ingest.geoip.downloader.endpoint").unwrap_or_else(|| DEFAULT_ENDPOINT.to_string())),
        maxmind_endpoint: setting(node, "ingest.geoip.downloader.maxmind.endpoint.default").unwrap_or_else(|| MAXMIND_ENDPOINT.to_string()),
        ipinfo_endpoint: setting(node, "ingest.ip_location.downloader.ipinfo.endpoint.default").unwrap_or_else(|| IPINFO_ENDPOINT.to_string()),
        configured: super::geoip::configured(&node.data),
        secret,
    }
}

fn last_runs() -> &'static Mutex<HashMap<PathBuf, Instant>> {
    static L: OnceLock<Mutex<HashMap<PathBuf, Instant>>> = OnceLock::new();
    L.get_or_init(Default::default)
}

/// The elected master's downloader pass (a cluster calls it periodically):
/// downloads when needed and the poll interval passed, or `force` (a new
/// pipeline / configuration).
pub fn master_tick(store: &dyn DbStore, node: &Node, secret: &dyn Fn(&str) -> Option<String>, force: bool) {
    if !needed(node) {
        return;
    }
    let due = last_runs().lock().unwrap_or_else(|e| e.into_inner()).get(&node.data).is_none_or(|t| t.elapsed() >= poll_interval(node));
    if !due && !force {
        return;
    }
    last_runs().lock().unwrap_or_else(|e| e.into_inner()).insert(node.data.clone(), Instant::now());
    match run_downloader(store, &sources(node, secret)) {
        Ok(n) if n > 0 => tracing::info!("stored [{n}] downloaded geoip databases in [{INDEX}]"),
        Ok(_) => {}
        Err(e) => tracing::warn!("geoip database download failed: {e}"),
    }
}

/// Run the downloader of the single node whose data directory is `data`
/// (on a cluster the cluster schedules it; a request asks for a pass).
pub fn trigger(data: &Path) {
    if CLUSTERED.load(Ordering::SeqCst) {
        FORCE.store(true, Ordering::SeqCst);
        return;
    }
    {
        let mut map = nodes().lock().unwrap_or_else(|e| e.into_inner());
        let Some(e) = map.get_mut(data) else { return };
        let Some(node) = e.node.upgrade() else { return };
        if e.running || setting(&node, "ingest.geoip.downloader.enabled").as_deref() == Some("false") && super::geoip::configured(data).is_empty() {
            return;
        }
        e.running = true;
    }
    let thread_data = data.to_path_buf();
    let spawned = std::thread::Builder::new().name("geoip-downloader".into()).spawn(move || {
        let data = thread_data;
        let no_secrets = |_: &str| -> Option<String> { None };
        let mut force = true;
        loop {
            if CLUSTERED.load(Ordering::SeqCst) {
                // The cluster took over scheduling.
                return;
            }
            let Some(node) = nodes().lock().unwrap_or_else(|e| e.into_inner()).get(&data).and_then(|e| e.node.upgrade()) else { return };
            let store = LocalStore(&node);
            master_tick(&store, &node, &no_secrets, force);
            force = false;
            if let Err(e) = sync_node(&store, &data.join("ingest-geoip")) {
                tracing::warn!("installing geoip databases failed: {e}");
            }
            drop(node);
            std::thread::sleep(Duration::from_secs(5));
        }
    });
    if spawned.is_err() {
        if let Some(e) = nodes().lock().unwrap_or_else(|e| e.into_inner()).get_mut(data) {
            e.running = false;
        }
    }
}

/// On a cluster: a pass was asked for since the last tick.
static FORCE: AtomicBool = AtomicBool::new(false);

/// Take the cluster's pending "download now" request.
pub fn take_force() -> bool {
    FORCE.swap(false, Ordering::SeqCst)
}

/// MD5 (RFC 1321) of `data`: the checksum the database overview carries.
pub fn md5_digest(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let (mut a0, mut b0, mut c0, mut d0) = (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_le_bytes());
    for chunk in msg.chunks(64) {
        let m: Vec<u32> = chunk.chunks(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    for (i, v) in [a0, b0, c0, d0].iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// The contents of the member of a (ustar) tar archive whose base name is
/// `file`.
fn tar_member(tar: &[u8], file: &str) -> Option<Vec<u8>> {
    let mut off = 0;
    while off + 512 <= tar.len() {
        let h = &tar[off..off + 512];
        if h.iter().all(|b| *b == 0) {
            return None;
        }
        let field = |a: usize, b: usize| {
            let f = &h[a..b];
            String::from_utf8_lossy(&f[..f.iter().position(|c| *c == 0).unwrap_or(f.len())]).to_string()
        };
        let mut name = field(0, 100);
        if &h[257..262] == b"ustar" {
            let prefix = field(345, 500);
            if !prefix.is_empty() {
                name = format!("{prefix}/{name}");
            }
        }
        let size = usize::from_str_radix(field(124, 136).trim(), 8).ok()?;
        let data = off + 512;
        if data + size > tar.len() {
            return None;
        }
        let kind = h[156];
        if (kind == b'0' || kind == 0) && name.rsplit('/').next() == Some(file) {
            return Some(tar[data..data + size].to_vec());
        }
        off = data + size.div_ceil(512) * 512;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn md5_known_vectors() {
        assert_eq!(hex(&md5_digest(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(&md5_digest(b"The quick brown fox jumps over the lazy dog")), "9e107d9d372bb6826bd81d3542a419d6");
        assert_eq!(hex(&md5_digest(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(&md5_digest(b"message digest")), "f96b697d7cb7938d525a2f31aaf161d0");
    }

    /// A one-member ustar archive.
    fn tar_of(path: &str, data: &[u8]) -> Vec<u8> {
        let mut h = vec![0u8; 512];
        h[..path.len()].copy_from_slice(path.as_bytes());
        h[100..107].copy_from_slice(b"0000644");
        h[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
        h[136..147].copy_from_slice(b"00000000000");
        h[156] = b'0';
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        h[148..156].copy_from_slice(b"        ");
        let sum: u32 = h.iter().map(|b| *b as u32).sum();
        h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
        let mut out = h;
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(512) * 512 + 1024, 0);
        out
    }

    fn tgz(path: &str, data: &[u8]) -> Vec<u8> {
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&tar_of(path, data)).unwrap();
        gz.finish().unwrap()
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!("{}/../elasticsearch-http/tests/data/geoip/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    /// Finding 446: the downloader stores default and configured
    /// (MaxMind, IPinfo) databases in `.geoip_databases` as chunks with a
    /// checksum state; a node installs them from there; unchanged
    /// databases are not downloaded again, and a new generation replaces
    /// the old chunks.
    #[test]
    fn databases_flow_through_the_system_index() {
        let city = fixture("GeoLite2-City.mmdb");
        let asn = fixture("GeoLite2-ASN.mmdb");
        let city_tgz = tgz("GeoLite2-City_20250101/GeoLite2-City.mmdb", &city);
        let ent_tgz = tgz("GeoIP2-City_20250101/GeoIP2-City.mmdb", &city);
        let ent_sha = hex(&sha2::Sha256::digest(&ent_tgz));
        let asn_sha = hex(&sha2::Sha256::digest(&asn));
        let overview = |url: &str| json!([{"name": "GeoLite2-City.tgz", "md5_hash": hex(&md5_digest(&city_tgz)), "url": url}]).to_string().into_bytes();
        let routes = vec![
            ("/maxmind/GeoIP2-City/download?suffix=tar.gz.sha256".to_string(), format!("{ent_sha}  GeoIP2-City_20250101.tar.gz\n").into_bytes()),
            ("/maxmind/GeoIP2-City/download?suffix=tar.gz".to_string(), ent_tgz.clone()),
            ("/ipinfo/asn.mmdb/checksums".to_string(), json!({"checksums": {"sha256": asn_sha}}).to_string().into_bytes()),
            ("/ipinfo/asn.mmdb?token=".to_string(), asn.clone()),
            ("/db/GeoLite2-City.tgz".to_string(), city_tgz.clone()),
            ("/v1/database".to_string(), Vec::new()),
        ];
        let (base, seen) = serve_with_overview(routes, overview);
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let store = LocalStore(&node);
        let secret = |k: &str| match k {
            "ingest.geoip.downloader.maxmind.license_key" => Some("lic".to_string()),
            "ingest.ip_location.downloader.ipinfo.token" => Some("tok".to_string()),
            _ => None,
        };
        let src = Sources {
            endpoint: Some(format!("{base}/v1/database")),
            maxmind_endpoint: format!("{base}/maxmind"),
            ipinfo_endpoint: format!("{base}/ipinfo"),
            configured: vec![
                ("ent".into(), json!({"name": "GeoIP2-City", "maxmind": {"account_id": "1234"}})),
                ("ipi".into(), json!({"name": "asn", "ipinfo": {}})),
            ],
            secret: &secret,
        };
        assert_eq!(run_downloader(&store, &src).unwrap(), 3);
        // MaxMind: basic auth with the account and the license key.
        let auth = format!("Basic {}", base64::engine::general_purpose::STANDARD.encode("1234:lic"));
        assert!(seen.lock().unwrap().iter().any(|h| h.contains("/maxmind/GeoIP2-City") && h.contains(&auth)), "{:?}", seen.lock().unwrap());
        let dir = tmp.path().join("installed");
        assert_eq!(sync_node(&store, &dir).unwrap(), 3);
        assert_eq!(std::fs::read(dir.join("GeoLite2-City.mmdb")).unwrap(), city);
        assert_eq!(std::fs::read(dir.join("GeoIP2-City.mmdb")).unwrap(), city);
        assert_eq!(std::fs::read(dir.join("asn.mmdb")).unwrap(), asn);
        // Nothing changed: no archive is fetched again, nothing reinstalled.
        let before = seen.lock().unwrap().len();
        assert_eq!(run_downloader(&store, &src).unwrap(), 0);
        let fetched: Vec<String> = seen.lock().unwrap()[before..].iter().map(|h| h.lines().next().unwrap_or("").to_string()).collect();
        assert!(fetched.iter().all(|l| l.contains("sha256") || l.contains("checksums") || l.contains("/v1/database")), "{fetched:?}");
        assert_eq!(sync_node(&store, &dir).unwrap(), 0);
        // The chunks are in the system index, with the state.
        let state = state(&store).unwrap();
        assert_eq!(state.len(), 3);
        assert_eq!(state["GeoLite2-City.mmdb"]["format"], "tgz");
        assert_eq!(chunk_ids(&store, "GeoLite2-City.mmdb", state["GeoLite2-City.mmdb"]["digest"].as_str().unwrap()).unwrap().len(), city_tgz.len().div_ceil(CHUNK));
    }

    /// A fake provider: `routes` maps a path prefix to a body (the
    /// `/v1/database` overview points at this server); every request head
    /// is recorded.
    fn serve_with_overview(mut routes: Vec<(String, Vec<u8>)>, overview: impl Fn(&str) -> Vec<u8>) -> (String, Arc<Mutex<Vec<String>>>) {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        for r in routes.iter_mut() {
            if r.0 == "/v1/database" {
                r.1 = overview(&format!("{base}/db/GeoLite2-City.tgz"));
            }
        }
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for c in l.incoming() {
                let Ok(mut c) = c else { continue };
                let mut buf = [0u8; 8192];
                let n = c.read(&mut buf).unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
                log.lock().unwrap().push(head.clone());
                let body = routes.iter().filter(|(p, _)| path.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|(_, b)| b.clone());
                let _ = match body {
                    Some(b) => write!(c, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len()).and_then(|_| c.write_all(&b)),
                    None => write!(c, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
                };
            }
        });
        (base, seen)
    }
}
