//! Read-only URL blob store (`type: url`): `http(s)://` or `file:` URLs.

use super::fs::FsBlobStore;
use super::{BlobStore, DeleteResult, WriteOptions, validate_key};
use crate::error::{Result, SnapshotError};
use async_trait::async_trait;
use bytes::Bytes;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// ES URL repository `http_max_retries` default.
const HTTP_MAX_RETRIES: u32 = 5;

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
            // ES URL repository defaults: `http_socket_timeout` 50s (here
            // the limit between response reads), `http_max_retries` 5.
            let client = reqwest::Client::builder()
                .connect_timeout(super::s3::CONNECT_TIMEOUT)
                .read_timeout(super::s3::DEFAULT_READ_TIMEOUT)
                .build()
                .map_err(|e| SnapshotError::io("building url repository client", e))?;
            Backend::Http { base, client }
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
        let mut attempt = 0u32;
        let (status, body) = loop {
            attempt += 1;
            let r = req.try_clone().expect("GET has no streaming body").send().await;
            let r = match r {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if !(200..300).contains(&status) {
                        Ok((status, Bytes::new()))
                    } else {
                        resp.bytes().await.map(|b| (status, b))
                    }
                }
                Err(e) => Err(e),
            };
            match r {
                Ok((s, _)) if s >= 500 && attempt <= HTTP_MAX_RETRIES => {}
                Ok(v) => break v,
                Err(e) if attempt <= HTTP_MAX_RETRIES => tracing::debug!("GET {url} failed, retrying: {e}"),
                Err(e) => return Err(SnapshotError::io(format!("GET [{url}]"), e)),
            }
            tokio::time::sleep(std::time::Duration::from_millis(100u64 << attempt.min(6))).await;
        };
        if status == 404 {
            return Err(SnapshotError::no_such_blob(key));
        }
        if status == 416 {
            return Ok(Bytes::new());
        }
        if !(200..300).contains(&status) {
            return Err(SnapshotError::io(format!("GET [{url}]"), format!("HTTP status {status}")));
        }
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
