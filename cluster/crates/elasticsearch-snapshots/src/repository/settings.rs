//! Repository registration: metadata, node-level context (`path.repo`, S3
//! client settings, allowed URLs) and per-type settings validation.

use crate::blobstore::BlobStore;
use crate::blobstore::fs::{FsBlobStore, resolve_repo_location};
use crate::blobstore::s3::sigv4::Credentials;
use crate::blobstore::s3::{MIN_PART_SIZE, S3BlobStore, S3Config};
use crate::blobstore::url::{UrlBlobStore, file_url_path};
use crate::error::{Result, SnapshotError};
use crate::util::{parse_byte_size, simple_match, stringify_settings};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// What `PUT _snapshot/{name}` stores in the cluster state.
#[derive(Debug, Clone, PartialEq)]
pub struct RepositoryMetadata {
    pub name: String,
    pub repo_type: String,
    pub settings: Map<String, Value>,
}

impl RepositoryMetadata {
    pub fn new(name: impl Into<String>, repo_type: impl Into<String>, settings: Value) -> Self {
        let settings = match settings {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        Self { name: name.into(), repo_type: repo_type.into(), settings }
    }

    /// Parse a `PUT _snapshot/{name}` body: `{"type": "...", "settings": {...}}`.
    pub fn from_put_body(name: &str, body: &Value) -> Result<Self> {
        let obj = body.as_object().ok_or_else(|| SnapshotError::illegal_argument("request body is required"))?;
        let mut repo_type = None;
        let mut settings = Map::new();
        for (k, v) in obj {
            match k.as_str() {
                "type" => repo_type = v.as_str().map(|s| s.to_string()),
                "settings" => {
                    settings = v
                        .as_object()
                        .cloned()
                        .ok_or_else(|| SnapshotError::new("elasticsearch_parse_exception", 400, "malformed settings section, should include an inner object"))?
                }
                other => {
                    return Err(SnapshotError::new(
                        "elasticsearch_parse_exception",
                        400,
                        format!("Unknown parameter [{other}] for repository [{name}]"),
                    ));
                }
            }
        }
        let repo_type = repo_type.ok_or_else(|| SnapshotError::validation(&["type is missing".to_string()]))?;
        Ok(Self { name: name.to_string(), repo_type, settings })
    }

    /// `GET _snapshot/{name}` entry: `{"type": ..., "settings": {...string values}}`.
    pub fn to_json(&self) -> Value {
        json!({"type": self.repo_type, "settings": Value::Object(stringify_settings(&self.settings))})
    }

    /// Read a setting by flat (`a.b`) or nested key, stringified.
    pub fn setting(&self, key: &str) -> Option<String> {
        setting_str(&self.settings, key)
    }
}

pub(crate) fn setting_str(settings: &Map<String, Value>, key: &str) -> Option<String> {
    fn as_string(v: &Value) -> Option<String> {
        match v {
            Value::String(s) => Some(s.clone()),
            Value::Null => None,
            Value::Bool(b) => Some(b.to_string()),
            Value::Number(n) => Some(n.to_string()),
            other => Some(other.to_string()),
        }
    }
    if let Some(v) = settings.get(key) {
        return as_string(v);
    }
    // nested lookup
    let mut parts = key.split('.');
    let mut cur = settings.get(parts.next()?)?;
    for p in parts {
        cur = cur.as_object()?.get(p)?;
    }
    as_string(cur)
}

pub(crate) fn setting_bool(repo: &str, settings: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match setting_str(settings, key) {
        None => Ok(default),
        Some(s) => match s.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(SnapshotError::illegal_argument(format!(
                "Failed to parse value [{s}] as only [true] or [false] are allowed."
            ))
            .with_extra("repository", json!(repo))),
        },
    }
}

/// Named S3 client settings (`s3.client.<name>.*` in `elasticsearch.yml` / keystore).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct S3ClientSettings {
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub protocol: Option<String>,
    pub path_style_access: bool,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub session_token: Option<String>,
    pub max_retries: Option<u32>,
}

impl S3ClientSettings {
    /// Extract every `s3.client.<name>.<key>` from flat node settings.
    pub fn from_node_settings(flat: &BTreeMap<String, String>) -> BTreeMap<String, S3ClientSettings> {
        let mut out: BTreeMap<String, S3ClientSettings> = BTreeMap::new();
        for (k, v) in flat {
            let Some(rest) = k.strip_prefix("s3.client.") else { continue };
            let Some((name, key)) = rest.split_once('.') else { continue };
            let c = out.entry(name.to_string()).or_default();
            match key {
                "endpoint" => c.endpoint = Some(v.clone()),
                "region" => c.region = Some(v.clone()),
                "protocol" => c.protocol = Some(v.to_ascii_lowercase()),
                "path_style_access" => c.path_style_access = v == "true",
                "access_key" => c.access_key = Some(v.clone()),
                "secret_key" => c.secret_key = Some(v.clone()),
                "session_token" => c.session_token = Some(v.clone()),
                "max_retries" => c.max_retries = v.parse().ok(),
                _ => {}
            }
        }
        out
    }
}

/// Node-level inputs needed to instantiate repositories.
#[derive(Debug, Clone, Default)]
pub struct RepositoryContext {
    /// `path.repo`.
    pub path_repo: Vec<PathBuf>,
    /// `repositories.url.allowed_urls` (wildcards allowed).
    pub allowed_urls: Vec<String>,
    /// Named S3 clients; `default` is used when a repository names none.
    pub s3_clients: BTreeMap<String, S3ClientSettings>,
}

/// Settings common to all blob-store repositories.
#[derive(Debug, Clone, PartialEq)]
pub struct RepositorySettings {
    pub compress: bool,
    /// Max size of one data blob part; `None` = unlimited.
    pub chunk_size: Option<u64>,
    pub readonly: bool,
    pub max_snapshot_bytes_per_sec: Option<u64>,
    pub max_restore_bytes_per_sec: Option<u64>,
    /// Read buffer when streaming data blobs.
    pub io_buffer: u64,
    /// A `source` repository (`SourceOnlySnapshotRepository`) over its
    /// `delegate_type` store: indices without complete `_source` fail.
    pub source_only: bool,
}

impl Default for RepositorySettings {
    fn default() -> Self {
        Self {
            compress: true,
            chunk_size: None,
            readonly: false,
            max_snapshot_bytes_per_sec: Some(40 << 20),
            max_restore_bytes_per_sec: None,
            io_buffer: 1 << 20,
            source_only: false,
        }
    }
}

const INVALID_FILENAME_CHARS: &str = "[ , \", *, \\, <, |, ,, >, /, ?]";

fn valid_file_name(s: &str) -> bool {
    !s.chars().any(|c| matches!(c, '\\' | '/' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' | ','))
}

pub fn validate_repository_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(SnapshotError::repository_bad_request(name, "cannot be empty"));
    }
    if name.contains('#') {
        return Err(SnapshotError::repository_bad_request(name, "must not contain '#'"));
    }
    if !valid_file_name(name) {
        return Err(SnapshotError::repository_bad_request(
            name,
            format!("must not contain the following characters {INVALID_FILENAME_CHARS}"),
        ));
    }
    Ok(())
}

/// ES `Strings.validFileName` + snapshot name rules
/// (`InvalidSnapshotNameException`).
pub fn validate_snapshot_name(repo: &str, name: &str) -> Result<()> {
    let err = |d: &str| Err(SnapshotError::invalid_snapshot_name(repo, name, d));
    if name.is_empty() {
        return err("cannot be empty");
    }
    if name.contains(' ') {
        return err("must not contain whitespace");
    }
    if name.contains(',') {
        return err("must not contain ','");
    }
    if name.contains('#') {
        return err("must not contain '#'");
    }
    if name.starts_with('_') {
        return err("must not start with '_'");
    }
    if name.to_lowercase() != name {
        return err("must be lowercase");
    }
    if !valid_file_name(name) {
        return err(&format!("must not contain the following characters {INVALID_FILENAME_CHARS}"));
    }
    Ok(())
}

fn byte_setting(meta: &RepositoryMetadata, key: &str) -> Result<Option<Option<u64>>> {
    match meta.setting(key) {
        None => Ok(None),
        Some(v) => parse_byte_size(key, &v).map(Some),
    }
}

/// Parse common settings + build the blob store for a repository.
pub async fn build_store(meta: &RepositoryMetadata, ctx: &RepositoryContext) -> Result<(Arc<dyn BlobStore>, RepositorySettings)> {
    // `source`: the delegate repository type's store, snapshotting `_source` only.
    if meta.repo_type == "source" {
        let delegate = meta.setting("delegate_type").filter(|d| !d.is_empty())
            .ok_or_else(|| SnapshotError::illegal_argument("delegate_type must be set"))?;
        if delegate == "source" {
            return Err(SnapshotError::repository(&meta.name, "repository type [source] cannot delegate to itself"));
        }
        let inner = RepositoryMetadata { name: meta.name.clone(), repo_type: delegate, settings: meta.settings.clone() };
        let (store, mut rs) = Box::pin(build_store(&inner, ctx)).await?;
        rs.source_only = true;
        return Ok((store, rs));
    }
    validate_repository_name(&meta.name)?;
    let name = meta.name.as_str();
    let mut rs = RepositorySettings {
        compress: setting_bool(name, &meta.settings, "compress", true)?,
        readonly: setting_bool(name, &meta.settings, "readonly", false)?,
        ..Default::default()
    };
    if let Some(v) = byte_setting(meta, "max_snapshot_bytes_per_sec")? {
        rs.max_snapshot_bytes_per_sec = v.filter(|b| *b > 0);
    }
    if let Some(v) = byte_setting(meta, "max_restore_bytes_per_sec")? {
        rs.max_restore_bytes_per_sec = v.filter(|b| *b > 0);
    }
    if let Some(v) = byte_setting(meta, "chunk_size")? {
        if v == Some(0) {
            return Err(SnapshotError::illegal_argument(format!(
                "failed to parse value [0b] for setting [chunk_size], must be >= [1b]"
            )));
        }
        rs.chunk_size = v;
    }
    let store: Arc<dyn BlobStore> = match meta.repo_type.as_str() {
        "fs" => {
            let location = meta.setting("location").unwrap_or_default();
            if location.is_empty() {
                return Err(SnapshotError::repository(name, "missing location"));
            }
            let path = resolve_repo_location(name, &location, &ctx.path_repo)?;
            Arc::new(FsBlobStore::open(path, rs.readonly).await?)
        }
        "url" => {
            rs.readonly = true;
            let url = meta.setting("url").unwrap_or_default();
            if url.is_empty() {
                return Err(SnapshotError::repository(name, "missing url"));
            }
            let protocol = url.split(':').next().unwrap_or_default().to_ascii_lowercase();
            let supported = meta
                .setting("supported_protocols")
                .map(|s| s.split(',').map(|p| p.trim().to_ascii_lowercase()).collect::<Vec<_>>())
                .unwrap_or_else(|| vec!["http".into(), "https".into(), "ftp".into(), "file".into(), "jar".into()]);
            if !supported.contains(&protocol) || !matches!(protocol.as_str(), "http" | "https" | "file") {
                return Err(SnapshotError::repository(name, format!("unsupported url protocol [{protocol}] from URL [{url}]")));
            }
            if protocol == "file" {
                let p = file_url_path(&url).unwrap_or_default();
                if resolve_repo_location(name, &p.to_string_lossy(), &ctx.path_repo).is_err() {
                    return Err(SnapshotError::repository(
                        name,
                        format!("file url [{url}] doesn't match any of the locations specified by path.repo"),
                    ));
                }
            } else if !ctx.allowed_urls.iter().any(|p| simple_match(p, &url)) {
                return Err(SnapshotError::repository(
                    name,
                    format!("file url [{url}] doesn't match any of the locations specified by path.repo or repositories.url.allowed_urls"),
                ));
            }
            Arc::new(UrlBlobStore::open(&url).await?)
        }
        "s3" => {
            let bucket = meta.setting("bucket").unwrap_or_default();
            if bucket.is_empty() {
                return Err(SnapshotError::repository(name, "No bucket defined for s3 repository"));
            }
            let client_name = meta.setting("client").unwrap_or_else(|| "default".into());
            let client = match ctx.s3_clients.get(&client_name) {
                Some(c) => c.clone(),
                None if client_name == "default" => S3ClientSettings::default(),
                None => {
                    return Err(SnapshotError::illegal_argument(format!(
                        "Unknown s3 client name [{client_name}]. Existing client configs: {}",
                        ctx.s3_clients.keys().cloned().collect::<Vec<_>>().join(",")
                    )));
                }
            };
            let mut cfg = S3Config::new(bucket);
            cfg.base_path = meta.setting("base_path").unwrap_or_default();
            cfg.endpoint = meta.setting("endpoint").or(client.endpoint.clone());
            if let Some(r) = meta.setting("region").or(client.region.clone()) {
                cfg.region = r;
            }
            if let Some(p) = meta.setting("protocol").or(client.protocol.clone()) {
                cfg.protocol = p.to_ascii_lowercase();
            }
            cfg.path_style_access = setting_bool(name, &meta.settings, "path_style_access", client.path_style_access)?;
            // ES 9 (`S3ClientSettings.checkDeprecatedCredentials` +
            // `SecureSetting.insecureString`): repository-level
            // `access_key` / `secret_key` must come in pairs and are then
            // refused (`es.allow_insecure_settings` is off); credentials
            // come from the keystore's `s3.client.<name>.*` settings.
            match (meta.setting("access_key").is_some(), meta.setting("secret_key").is_some()) {
                (true, false) => {
                    return Err(SnapshotError::illegal_argument("Repository setting [access_key must be accompanied by setting [secret_key]"));
                }
                (false, true) => {
                    return Err(SnapshotError::illegal_argument("Repository setting [secret_key must be accompanied by setting [access_key]"));
                }
                (true, true) => {
                    return Err(SnapshotError::illegal_argument("Setting [access_key] is insecure, use the elasticsearch keystore instead"));
                }
                (false, false) => {}
            }
            let ak = client.access_key.clone();
            let sk = client.secret_key.clone();
            let st = client.session_token.clone();
            cfg.credentials = match (ak, sk) {
                (Some(a), Some(s)) => Some(Credentials { access_key: a, secret_key: s, session_token: st }),
                (None, None) => None,
                (Some(_), None) => {
                    return Err(SnapshotError::illegal_argument(
                        "Missing secret key for s3 client [".to_string() + &client_name + "]",
                    ));
                }
                (None, Some(_)) => {
                    return Err(SnapshotError::illegal_argument(
                        "Missing access key for s3 client [".to_string() + &client_name + "]",
                    ));
                }
            };
            cfg.server_side_encryption = setting_bool(name, &meta.settings, "server_side_encryption", false)?;
            if let Some(sc) = meta.setting("storage_class") {
                let up = sc.to_ascii_uppercase();
                if !matches!(
                    up.as_str(),
                    "STANDARD" | "REDUCED_REDUNDANCY" | "STANDARD_IA" | "ONEZONE_IA" | "INTELLIGENT_TIERING" | "GLACIER_IR"
                ) {
                    return Err(SnapshotError::new("blob_store_exception", 500, format!("`{sc}` is not a valid S3 Storage Class.")));
                }
                cfg.storage_class = Some(up);
            }
            if let Some(acl) = meta.setting("canned_acl") {
                if !matches!(
                    acl.as_str(),
                    "private"
                        | "public-read"
                        | "public-read-write"
                        | "authenticated-read"
                        | "log-delivery-write"
                        | "bucket-owner-read"
                        | "bucket-owner-full-control"
                ) {
                    return Err(SnapshotError::new("blob_store_exception", 500, format!("cannedACL is not valid: [{acl}]")));
                }
                cfg.canned_acl = Some(acl);
            }
            if let Some(b) = byte_setting(meta, "buffer_size")? {
                let b = b.unwrap_or(crate::blobstore::s3::DEFAULT_BUFFER_SIZE);
                if b < MIN_PART_SIZE {
                    return Err(SnapshotError::illegal_argument(format!(
                        "failed to parse value [{}] for setting [buffer_size], must be >= [5mb]",
                        meta.setting("buffer_size").unwrap_or_default()
                    )));
                }
                cfg.buffer_size = b;
            }
            if let Some(r) = client.max_retries {
                cfg.max_retries = r;
            }
            // S3 default chunk_size 5tb; minimum 5mb.
            match rs.chunk_size {
                None => rs.chunk_size = Some(5u64 << 40),
                Some(c) if c < MIN_PART_SIZE => {
                    return Err(SnapshotError::illegal_argument(format!(
                        "failed to parse value [{}] for setting [chunk_size], must be >= [5mb]",
                        meta.setting("chunk_size").unwrap_or_default()
                    )));
                }
                _ => {}
            }
            Arc::new(S3BlobStore::new(cfg)?)
        }
        other => {
            return Err(SnapshotError::repository(name, format!("repository type [{other}] does not exist")));
        }
    };
    Ok((store, rs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_name_rules() {
        assert!(validate_snapshot_name("r", "ok-name.1").is_ok());
        let e = validate_snapshot_name("r", "Bad").unwrap_err();
        assert_eq!(e.kind, "invalid_snapshot_name_exception");
        assert_eq!(e.reason, "[r:Bad] Invalid snapshot name [Bad], must be lowercase");
        assert!(validate_snapshot_name("r", "_x").unwrap_err().reason.ends_with("must not start with '_'"));
        assert!(validate_snapshot_name("r", "a b").unwrap_err().reason.ends_with("must not contain whitespace"));
        assert!(validate_snapshot_name("r", "a*b").unwrap_err().reason.contains("must not contain the following characters"));
    }

    #[test]
    fn s3_client_settings() {
        let mut flat = BTreeMap::new();
        flat.insert("s3.client.minio.endpoint".to_string(), "localhost:9000".to_string());
        flat.insert("s3.client.minio.protocol".to_string(), "HTTP".to_string());
        flat.insert("s3.client.minio.path_style_access".to_string(), "true".to_string());
        flat.insert("other".to_string(), "x".to_string());
        let c = S3ClientSettings::from_node_settings(&flat);
        let m = &c["minio"];
        assert_eq!(m.endpoint.as_deref(), Some("localhost:9000"));
        assert_eq!(m.protocol.as_deref(), Some("http"));
        assert!(m.path_style_access);
    }

    #[tokio::test]
    async fn insecure_repository_credentials_are_refused() {
        let ctx = RepositoryContext::default();
        for (settings, reason) in [
            (json!({"bucket": "b", "access_key": "a", "secret_key": "s"}), "Setting [access_key] is insecure, use the elasticsearch keystore instead"),
            (json!({"bucket": "b", "access_key": "a"}), "Repository setting [access_key must be accompanied by setting [secret_key]"),
            (json!({"bucket": "b", "secret_key": "s"}), "Repository setting [secret_key must be accompanied by setting [access_key]"),
        ] {
            let e = build_store(&RepositoryMetadata::new("r", "s3", settings), &ctx).await.err().unwrap();
            assert_eq!((e.kind, e.reason.as_str()), ("illegal_argument_exception", reason));
        }
        let mut clients = BTreeMap::new();
        clients.insert("default".to_string(), S3ClientSettings { secret_key: Some("s".into()), ..Default::default() });
        let ctx = RepositoryContext { s3_clients: clients, ..Default::default() };
        let e = build_store(&RepositoryMetadata::new("r", "s3", json!({"bucket": "b"})), &ctx).await.err().unwrap();
        assert_eq!(e.reason, "Missing access key for s3 client [default]");
    }

    #[tokio::test]
    async fn type_and_settings_errors() {
        let ctx = RepositoryContext::default();
        let e = build_store(&RepositoryMetadata::new("r", "source", json!({})), &ctx).await.err().unwrap();
        assert_eq!(e.reason, "delegate_type must be set");
        let e = build_store(&RepositoryMetadata::new("r", "nope", json!({})), &ctx).await.err().unwrap();
        assert_eq!(e.reason, "[r] repository type [nope] does not exist");
        let e = build_store(&RepositoryMetadata::new("r", "fs", json!({})), &ctx).await.err().unwrap();
        assert_eq!(e.reason, "[r] missing location");
        let e = build_store(&RepositoryMetadata::new("r", "s3", json!({})), &ctx).await.err().unwrap();
        assert_eq!(e.reason, "[r] No bucket defined for s3 repository");
        let e = build_store(&RepositoryMetadata::new("r", "url", json!({"url": "file:/etc"})), &ctx).await.err().unwrap();
        assert_eq!(e.reason, "[r] file url [file:/etc] doesn't match any of the locations specified by path.repo");
        let md = RepositoryMetadata::new("r", "s3", json!({"bucket": "b", "compress": true, "chunk_size": "1gb"}));
        let (store, rs) = build_store(&md, &ctx).await.unwrap();
        assert_eq!(store.kind(), "s3");
        assert_eq!(rs.chunk_size, Some(1 << 30));
        assert_eq!(md.to_json(), json!({"type":"s3","settings":{"bucket":"b","compress":"true","chunk_size":"1gb"}}));
        let e = RepositoryMetadata::from_put_body("r", &json!({"type":"fs","bogus":1})).unwrap_err();
        assert_eq!(e.reason, "Unknown parameter [bogus] for repository [r]");
    }
}
