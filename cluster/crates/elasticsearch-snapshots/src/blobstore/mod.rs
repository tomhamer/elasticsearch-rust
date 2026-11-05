//! Blob store abstraction (ES `BlobStore` / `BlobContainer`).
//!
//! A [`BlobStore`] is a flat key space with `/`-separated "directories"
//! (S3 semantics); [`BlobContainer`] is a store plus a path prefix and is what
//! the repository code works with.

mod admitted;
pub use admitted::ReadAdmission;
pub mod fs;
pub mod s3;
pub mod url;

use crate::error::{Result, SnapshotError};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WriteOptions {
    /// Fail with `file_already_exists_exception` if the blob exists.
    pub fail_if_exists: bool,
    /// Readers must never observe a partially written blob.
    pub atomic: bool,
}

impl WriteOptions {
    pub const OVERWRITE: Self = Self { fail_if_exists: false, atomic: false };
    pub const CREATE_NEW: Self = Self { fail_if_exists: true, atomic: false };
    pub const ATOMIC: Self = Self { fail_if_exists: false, atomic: true };
    pub const ATOMIC_CREATE_NEW: Self = Self { fail_if_exists: true, atomic: true };
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeleteResult {
    pub blobs_deleted: u64,
    pub bytes_deleted: u64,
}

impl DeleteResult {
    pub fn add(&mut self, o: DeleteResult) {
        self.blobs_deleted += o.blobs_deleted;
        self.bytes_deleted += o.bytes_deleted;
    }
}

pub type ByteStream = BoxStream<'static, Result<Bytes>>;

/// Key-based blob store. Keys are `/`-separated and never start with `/`.
/// "Directories" (`dir` arguments) are either empty (root) or end with `/`.
#[async_trait]
pub trait BlobStore: Send + Sync + 'static {
    /// `fs`, `s3`, `url`.
    fn kind(&self) -> &'static str;

    /// True for stores that cannot be written at all (`url`).
    fn is_read_only(&self) -> bool {
        false
    }

    /// Human-readable location (used in error messages).
    fn describe(&self) -> String;

    async fn read(&self, key: &str) -> Result<Bytes>;

    /// Collect an encoded body under caller admission. Implementations must not
    /// silently fall back to an unadmitted full-body read.
    async fn read_admitted(&self, _key: &str, _admission: ReadAdmission) -> Result<Bytes> {
        Err(SnapshotError::new("unsupported_operation_exception",500,"blob store does not support admitted reads"))
    }

    /// Read `len` bytes at `offset` (shorter if the blob ends earlier).
    async fn read_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes>;

    async fn write(&self, key: &str, data: Bytes, opts: WriteOptions) -> Result<()>;

    /// Write a blob of known length from a stream. Default buffers in memory.
    async fn write_stream(&self, key: &str, len: u64, mut stream: ByteStream, opts: WriteOptions) -> Result<()> {
        let mut buf = Vec::with_capacity(len as usize);
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk?);
        }
        self.write(key, Bytes::from(buf), opts).await
    }

    async fn exists(&self, key: &str) -> Result<bool>;

    /// Blobs directly inside `dir` whose name starts with `name_prefix`; name → size.
    async fn list(&self, dir: &str, name_prefix: &str) -> Result<BTreeMap<String, u64>>;

    /// Names of child "directories" directly inside `dir`.
    async fn list_children(&self, dir: &str) -> Result<Vec<String>>;

    /// Delete blobs; missing blobs are ignored.
    async fn delete_batch(&self, keys: &[String]) -> Result<()>;

    /// Recursively delete everything under `dir`.
    async fn delete_dir(&self, dir: &str) -> Result<DeleteResult>;
}

pub(crate) fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() || key.starts_with('/') || key.split('/').any(|seg| seg == ".." || seg == ".") {
        return Err(SnapshotError::illegal_argument(format!("invalid blob name [{key}]")));
    }
    Ok(())
}

/// A store plus a path (ES `BlobContainer`).
#[derive(Clone)]
pub struct BlobContainer {
    store: Arc<dyn BlobStore>,
    path: String,
}

impl std::fmt::Debug for BlobContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlobContainer({}:{})", self.store.kind(), self.path)
    }
}

impl BlobContainer {
    pub fn new(store: Arc<dyn BlobStore>, path: impl Into<String>) -> Self {
        let mut path = path.into();
        if !path.is_empty() && !path.ends_with('/') {
            path.push('/');
        }
        Self { store, path }
    }

    pub fn store(&self) -> &Arc<dyn BlobStore> {
        &self.store
    }

    /// Directory path, empty or ending with `/`.
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn child(&self, name: &str) -> BlobContainer {
        BlobContainer::new(self.store.clone(), format!("{}{}/", self.path, name.trim_matches('/')))
    }

    fn key(&self, name: &str) -> String {
        format!("{}{}", self.path, name)
    }

    pub async fn read_blob(&self, name: &str) -> Result<Bytes> {
        self.store.read(&self.key(name)).await
    }

    pub async fn read_blob_admitted(&self, name: &str, admission: ReadAdmission) -> Result<Bytes> {
        self.store.read_admitted(&self.key(name), admission).await
    }

    pub async fn read_blob_range(&self, name: &str, offset: u64, len: u64) -> Result<Bytes> {
        self.store.read_range(&self.key(name), offset, len).await
    }

    /// Stream a blob in `chunk`-sized range reads.
    pub fn read_blob_stream(&self, name: &str, total_len: u64, chunk: u64) -> ByteStream {
        let me = self.clone();
        let name = name.to_string();
        let chunk = chunk.max(1);
        futures::stream::unfold(0u64, move |off| {
            let me = me.clone();
            let name = name.clone();
            async move {
                if off >= total_len {
                    return None;
                }
                let len = chunk.min(total_len - off);
                let r = me.read_blob_range(&name, off, len).await;
                match r {
                    Ok(b) if b.is_empty() => Some((Err(SnapshotError::io(format!("reading [{name}]"), "unexpected end of blob")), total_len)),
                    Ok(b) => {
                        let n = b.len() as u64;
                        Some((Ok(b), off + n))
                    }
                    Err(e) => Some((Err(e), total_len)),
                }
            }
        })
        .boxed()
    }

    pub async fn write_blob(&self, name: &str, data: Bytes, fail_if_exists: bool) -> Result<()> {
        self.store
            .write(&self.key(name), data, WriteOptions { fail_if_exists, atomic: false })
            .await
    }

    pub async fn write_blob_atomic(&self, name: &str, data: Bytes, fail_if_exists: bool) -> Result<()> {
        self.store
            .write(&self.key(name), data, WriteOptions { fail_if_exists, atomic: true })
            .await
    }

    pub async fn write_blob_stream(&self, name: &str, len: u64, stream: ByteStream, fail_if_exists: bool) -> Result<()> {
        self.store
            .write_stream(&self.key(name), len, stream, WriteOptions { fail_if_exists, atomic: false })
            .await
    }

    pub async fn blob_exists(&self, name: &str) -> Result<bool> {
        self.store.exists(&self.key(name)).await
    }

    pub async fn list_blobs(&self) -> Result<BTreeMap<String, u64>> {
        self.store.list(&self.path, "").await
    }

    pub async fn list_blobs_by_prefix(&self, prefix: &str) -> Result<BTreeMap<String, u64>> {
        self.store.list(&self.path, prefix).await
    }

    pub async fn children(&self) -> Result<Vec<String>> {
        self.store.list_children(&self.path).await
    }

    pub async fn delete_blobs_ignoring_if_not_exists(&self, names: &[String]) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        let keys: Vec<String> = names.iter().map(|n| self.key(n)).collect();
        self.store.delete_batch(&keys).await
    }

    /// Recursively delete this container.
    pub async fn delete(&self) -> Result<DeleteResult> {
        self.store.delete_dir(&self.path).await
    }
}

/// Magic prefix of compressed metadata blobs.
pub const LZ4_MAGIC: &[u8; 4] = b"LZ4\0";

/// Encode a metadata blob (JSON), optionally LZ4-compressed (repository `compress` setting).
pub fn encode_metadata_blob(json: &[u8], compress: bool) -> Bytes {
    if compress {
        let mut out = Vec::with_capacity(json.len() / 2 + 8);
        out.extend_from_slice(LZ4_MAGIC);
        out.extend_from_slice(&lz4_flex::compress_prepend_size(json));
        Bytes::from(out)
    } else {
        Bytes::copy_from_slice(json)
    }
}

/// Decode a metadata blob written by [`encode_metadata_blob`] (compression auto-detected).
pub fn decode_metadata_blob(name: &str, data: &[u8]) -> Result<Vec<u8>> {
    if let Some(rest) = data.strip_prefix(LZ4_MAGIC.as_slice()) {
        lz4_flex::decompress_size_prepended(rest)
            .map_err(|e| SnapshotError::new("corrupt_state_exception", 500, format!("failed to decompress blob [{name}]: {e}")))
    } else {
        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_blob_roundtrip() {
        let json = br#"{"a":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}"#;
        let c = encode_metadata_blob(json, true);
        assert!(c.starts_with(LZ4_MAGIC));
        assert!(c.len() < json.len());
        assert_eq!(decode_metadata_blob("x", &c).unwrap(), json.to_vec());
        let p = encode_metadata_blob(json, false);
        assert_eq!(decode_metadata_blob("x", &p).unwrap(), json.to_vec());
    }
}
