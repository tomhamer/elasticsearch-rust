//! Read-only URL blob store (`type: url`): `http(s)://` or `file:` URLs.

use super::fs::FsBlobStore;
use super::{BlobStore, DeleteResult, WriteOptions, validate_key};
use crate::error::{Result, SnapshotError};
use async_trait::async_trait;
use bytes::Bytes;
use std::collections::BTreeMap;
use std::path::PathBuf;

enum Backend {
    File(FsBlobStore),
    Http { base: String, client: reqwest::Client },
}

pub struct UrlBlobStore {
    url: String,
    backend: Backend,
}

impl UrlBlobStore {
    /// `url` must already be validated against `path.repo` / allowed URLs.
    pub async fn open(url: &str) -> Result<Self> {
        let backend = if let Some(path) = file_url_path(url) {
            Backend::File(FsBlobStore::open(path, true).await?)
        } else if url.starts_with("http://") || url.starts_with("https://") {
            let mut base = url.to_string();
            if !base.ends_with('/') {
                base.push('/');
            }
            Backend::Http { base, client: reqwest::Client::new() }
        } else {
            return Err(SnapshotError::illegal_argument(format!("unsupported url protocol in [{url}]")));
        };
        Ok(Self { url: url.to_string(), backend })
    }

    fn unsupported() -> SnapshotError {
        SnapshotError::new("unsupported_operation_exception", 500, "URL repository doesn't support this operation")
    }

    async fn http_get(&self, key: &str, range: Option<(u64, u64)>) -> Result<Bytes> {
        let Backend::Http { base, client } = &self.backend else { unreachable!() };
        validate_key(key)?;
        let url = format!("{base}{}", crate::blobstore::s3::sigv4::uri_encode_path(key));
        let mut req = client.get(&url);
        if let Some((off, len)) = range {
            if len == 0 {
                return Ok(Bytes::new());
            }
            req = req.header("Range", format!("bytes={}-{}", off, off + len - 1));
        }
        let resp = req.send().await.map_err(|e| SnapshotError::io(format!("GET [{url}]"), e))?;
        let status = resp.status().as_u16();
        if status == 404 {
            return Err(SnapshotError::no_such_blob(key));
        }
        if status == 416 {
            return Ok(Bytes::new());
        }
        if !(200..300).contains(&status) {
            return Err(SnapshotError::io(format!("GET [{url}]"), format!("HTTP status {status}")));
        }
        let body = resp.bytes().await.map_err(|e| SnapshotError::io(format!("GET [{url}]"), e))?;
        // Server ignored the Range header: slice locally.
        if let (Some((off, len)), 200) = (range, status) {
            let start = (off as usize).min(body.len());
            let end = (start + len as usize).min(body.len());
            return Ok(body.slice(start..end));
        }
        Ok(body)
    }
}

/// `file:/a/b`, `file:///a/b` → `/a/b`.
pub fn file_url_path(url: &str) -> Option<PathBuf> {
    let rest = url.strip_prefix("file:")?;
    let rest = rest.strip_prefix("//").map(|r| r.strip_prefix("localhost").unwrap_or(r)).unwrap_or(rest);
    let decoded = percent_encoding::percent_decode_str(rest).decode_utf8_lossy().to_string();
    Some(PathBuf::from(decoded))
}

#[async_trait]
impl BlobStore for UrlBlobStore {
    fn kind(&self) -> &'static str {
        "url"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn describe(&self) -> String {
        self.url.clone()
    }

    async fn read(&self, key: &str) -> Result<Bytes> {
        match &self.backend {
            Backend::File(fs) => fs.read(key).await,
            Backend::Http { .. } => self.http_get(key, None).await,
        }
    }

    async fn read_admitted(&self, key: &str, admission: super::ReadAdmission) -> Result<Bytes> {
        match &self.backend {
            Backend::File(fs) => fs.read_admitted(key, admission).await,
            Backend::Http { base, client } => {
                validate_key(key)?;
                let url = format!("{base}{}", crate::blobstore::s3::sigv4::uri_encode_path(key));
                let response = client.get(&url).send().await.map_err(|e| SnapshotError::io("GET admitted blob", e))?;
                let status = response.status().as_u16();
                if status == 404 { return Err(SnapshotError::no_such_blob(key)); }
                if !(200..300).contains(&status) { return Err(SnapshotError::io("GET admitted blob",format!("HTTP status {status}"))); }
                super::admitted::response(response, admission).await
            }
        }
    }

    async fn read_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes> {
        match &self.backend {
            Backend::File(fs) => fs.read_range(key, offset, len).await,
            Backend::Http { .. } => self.http_get(key, Some((offset, len))).await,
        }
    }

    async fn write(&self, _key: &str, _data: Bytes, _opts: WriteOptions) -> Result<()> {
        Err(Self::unsupported())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        match &self.backend {
            Backend::File(fs) => fs.exists(key).await,
            Backend::Http { .. } => match self.http_get(key, Some((0, 1))).await {
                Ok(_) => Ok(true),
                Err(e) if e.is_not_found() => Ok(false),
                Err(e) => Err(e),
            },
        }
    }

    async fn list(&self, dir: &str, name_prefix: &str) -> Result<BTreeMap<String, u64>> {
        match &self.backend {
            Backend::File(fs) => fs.list(dir, name_prefix).await,
            Backend::Http { .. } => Err(Self::unsupported()),
        }
    }

    async fn list_children(&self, dir: &str) -> Result<Vec<String>> {
        match &self.backend {
            Backend::File(fs) => fs.list_children(dir).await,
            Backend::Http { .. } => Err(Self::unsupported()),
        }
    }

    async fn delete_batch(&self, _keys: &[String]) -> Result<()> {
        Err(Self::unsupported())
    }

    async fn delete_dir(&self, _dir: &str) -> Result<DeleteResult> {
        Err(Self::unsupported())
    }
}
