//! Shared-filesystem blob store (`type: fs`).

use super::{BlobStore, ByteStream, DeleteResult, WriteOptions, validate_key};
use crate::error::{Result, SnapshotError};
use crate::util::random_base64_uuid;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

/// Lexical path normalisation (Java `Path.normalize()`).
pub fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// ES `Environment.resolveRepoDir`: resolve `location` against every
/// `path.repo` root and return the first that stays inside its root.
pub fn resolve_repo_location(repo: &str, location: &str, path_repo: &[PathBuf]) -> Result<PathBuf> {
    if location.trim().is_empty() {
        return Err(SnapshotError::repository(repo, "missing location"));
    }
    for root in path_repo {
        let nroot = normalize(root);
        let resolved = normalize(&nroot.join(location));
        if resolved.starts_with(&nroot) {
            return Ok(resolved);
        }
    }
    if path_repo.is_empty() {
        Err(SnapshotError::repository(
            repo,
            format!("location [{location}] doesn't match any of the locations specified by path.repo because this setting is empty"),
        ))
    } else {
        let list: Vec<String> = path_repo.iter().map(|p| p.display().to_string()).collect();
        Err(SnapshotError::repository(
            repo,
            format!(
                "location [{location}] doesn't match any of the locations specified by path.repo: [{}]",
                list.join(", ")
            ),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct FsBlobStore {
    root: PathBuf,
    buffer_size: usize,
}

impl FsBlobStore {
    /// Opens (creating if needed) a store rooted at `root`.
    pub async fn open(root: PathBuf, readonly: bool) -> Result<Self> {
        if !readonly {
            tokio::fs::create_dir_all(&root).await.map_err(|e| SnapshotError::io(format!("creating [{}]", root.display()), e))?;
        }
        Ok(Self { root, buffer_size: 128 * 1024 })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, key: &str) -> Result<PathBuf> {
        validate_key(key)?;
        Ok(self.root.join(key))
    }

    fn dir(&self, dir: &str) -> PathBuf {
        let d = dir.trim_end_matches('/');
        if d.is_empty() { self.root.clone() } else { self.root.join(d) }
    }

    fn not_found(key: &str, e: std::io::Error) -> SnapshotError {
        if e.kind() == std::io::ErrorKind::NotFound {
            SnapshotError::no_such_blob(key)
        } else {
            SnapshotError::io(format!("blob [{key}]"), e)
        }
    }

    async fn ensure_parent(p: &Path) -> Result<()> {
        if let Some(parent) = p.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| SnapshotError::io(format!("creating [{}]", parent.display()), e))?;
        }
        Ok(())
    }

    fn temp_path(p: &Path) -> PathBuf {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        p.with_file_name(format!("pending-{name}-{}", random_base64_uuid()))
    }

    /// Move a fully written temp file into place, honouring `fail_if_exists`.
    async fn publish(tmp: &Path, dest: &Path, key: &str, fail_if_exists: bool) -> Result<()> {
        let r = if fail_if_exists {
            // hard_link fails atomically if `dest` exists.
            match tokio::fs::hard_link(tmp, dest).await {
                Ok(()) => {
                    let _ = tokio::fs::remove_file(tmp).await;
                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(SnapshotError::blob_exists(key)),
                Err(e) => Err(SnapshotError::io(format!("blob [{key}]"), e)),
            }
        } else {
            tokio::fs::rename(tmp, dest).await.map_err(|e| SnapshotError::io(format!("blob [{key}]"), e))
        };
        if r.is_err() {
            let _ = tokio::fs::remove_file(tmp).await;
        }
        r?;
        if let Some(parent) = dest.parent() {
            if let Ok(d) = tokio::fs::File::open(parent).await {
                let _ = d.sync_all().await;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl BlobStore for FsBlobStore {
    fn kind(&self) -> &'static str {
        "fs"
    }

    fn describe(&self) -> String {
        self.root.display().to_string()
    }

    async fn read(&self, key: &str) -> Result<Bytes> {
        let p = self.path(key)?;
        tokio::fs::read(&p).await.map(Bytes::from).map_err(|e| Self::not_found(key, e))
    }

    async fn read_admitted(&self, key: &str, admission: super::ReadAdmission) -> Result<Bytes> {
        let mut file = tokio::fs::File::open(self.path(key)?).await.map_err(|e| Self::not_found(key,e))?;
        let mut out = super::admitted::Buffer::new(admission);
        let mut chunk = [0u8; 8192];
        loop {
            let n = file.read(&mut chunk).await?;
            if n == 0 { break; }
            out.extend(&chunk[..n])?;
        }
        Ok(out.finish())
    }

    async fn read_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes> {
        let p = self.path(key)?;
        let mut f = tokio::fs::File::open(&p).await.map_err(|e| Self::not_found(key, e))?;
        f.seek(std::io::SeekFrom::Start(offset)).await?;
        let mut buf = Vec::with_capacity(len.min(64 << 20) as usize);
        f.take(len).read_to_end(&mut buf).await?;
        Ok(Bytes::from(buf))
    }

    async fn write(&self, key: &str, data: Bytes, opts: WriteOptions) -> Result<()> {
        let stream = futures::stream::once(async move { Ok(data) }).boxed();
        self.write_stream(key, 0, stream, opts).await
    }

    async fn write_stream(&self, key: &str, _len: u64, mut stream: ByteStream, opts: WriteOptions) -> Result<()> {
        let dest = self.path(key)?;
        Self::ensure_parent(&dest).await?;
        if opts.atomic || opts.fail_if_exists {
            if opts.fail_if_exists && tokio::fs::try_exists(&dest).await.unwrap_or(false) {
                return Err(SnapshotError::blob_exists(key));
            }
            let tmp = Self::temp_path(&dest);
            let res: Result<()> = async {
                let f = tokio::fs::File::create(&tmp).await?;
                let mut w = tokio::io::BufWriter::with_capacity(self.buffer_size, f);
                while let Some(chunk) = stream.next().await {
                    w.write_all(&chunk?).await?;
                }
                w.flush().await?;
                w.get_mut().sync_all().await?;
                Ok(())
            }
            .await;
            if let Err(e) = res {
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(e);
            }
            Self::publish(&tmp, &dest, key, opts.fail_if_exists).await
        } else {
            let f = tokio::fs::File::create(&dest).await?;
            let mut w = tokio::io::BufWriter::with_capacity(self.buffer_size, f);
            while let Some(chunk) = stream.next().await {
                w.write_all(&chunk?).await?;
            }
            w.flush().await?;
            w.get_mut().sync_all().await?;
            Ok(())
        }
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let p = self.path(key)?;
        Ok(tokio::fs::metadata(&p).await.map(|m| m.is_file()).unwrap_or(false))
    }

    async fn list(&self, dir: &str, name_prefix: &str) -> Result<BTreeMap<String, u64>> {
        let d = self.dir(dir);
        let mut out = BTreeMap::new();
        let mut rd = match tokio::fs::read_dir(&d).await {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(SnapshotError::io(format!("listing [{}]", d.display()), e)),
        };
        while let Some(e) = rd.next_entry().await? {
            let md = match e.metadata().await {
                Ok(m) => m,
                Err(_) => continue,
            };
            if !md.is_file() {
                continue;
            }
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with(name_prefix) {
                out.insert(name, md.len());
            }
        }
        Ok(out)
    }

    async fn list_children(&self, dir: &str) -> Result<Vec<String>> {
        let d = self.dir(dir);
        let mut out = Vec::new();
        let mut rd = match tokio::fs::read_dir(&d).await {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(SnapshotError::io(format!("listing [{}]", d.display()), e)),
        };
        while let Some(e) = rd.next_entry().await? {
            if e.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
                out.push(e.file_name().to_string_lossy().to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<()> {
        for k in keys {
            let p = self.path(k)?;
            match tokio::fs::remove_file(&p).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(SnapshotError::io(format!("deleting [{k}]"), e)),
            }
        }
        Ok(())
    }

    async fn delete_dir(&self, dir: &str) -> Result<DeleteResult> {
        let d = self.dir(dir);
        let mut res = DeleteResult::default();
        let mut stack = vec![d.clone()];
        while let Some(cur) = stack.pop() {
            let mut rd = match tokio::fs::read_dir(&cur).await {
                Ok(rd) => rd,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(SnapshotError::io(format!("listing [{}]", cur.display()), e)),
            };
            while let Some(e) = rd.next_entry().await? {
                let md = e.metadata().await?;
                if md.is_dir() {
                    stack.push(e.path());
                } else {
                    res.blobs_deleted += 1;
                    res.bytes_deleted += md.len();
                }
            }
        }
        if d == self.root {
            // Never remove the root itself, only its content.
            let mut rd = tokio::fs::read_dir(&d).await?;
            while let Some(e) = rd.next_entry().await? {
                if e.file_type().await?.is_dir() {
                    tokio::fs::remove_dir_all(e.path()).await?;
                } else {
                    tokio::fs::remove_file(e.path()).await?;
                }
            }
        } else {
            match tokio::fs::remove_dir_all(&d).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(SnapshotError::io(format!("deleting [{}]", d.display()), e)),
            }
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_repo_validation() {
        let e = resolve_repo_location("r", "/tmp/x", &[]).unwrap_err();
        assert_eq!(e.kind, "repository_exception");
        assert_eq!(
            e.reason,
            "[r] location [/tmp/x] doesn't match any of the locations specified by path.repo because this setting is empty"
        );
        let roots = vec![PathBuf::from("/mnt/backups"), PathBuf::from("/data/repo")];
        let e = resolve_repo_location("r", "/tmp/x", &roots).unwrap_err();
        assert_eq!(
            e.reason,
            "[r] location [/tmp/x] doesn't match any of the locations specified by path.repo: [/mnt/backups, /data/repo]"
        );
        assert_eq!(resolve_repo_location("r", "/data/repo/a/../b", &roots).unwrap(), PathBuf::from("/data/repo/b"));
        assert_eq!(resolve_repo_location("r", "rel", &roots).unwrap(), PathBuf::from("/mnt/backups/rel"));
        assert!(resolve_repo_location("r", "../escape", &roots).is_err());
        let e = resolve_repo_location("r", "", &roots).unwrap_err();
        assert_eq!(e.reason, "[r] missing location");
    }

    #[tokio::test]
    async fn fs_store_ops() {
        let dir = tempfile::tempdir().unwrap();
        let s = FsBlobStore::open(dir.path().join("repo"), false).await.unwrap();
        s.write("a/b/c", Bytes::from_static(b"hello world"), WriteOptions::ATOMIC_CREATE_NEW).await.unwrap();
        let e = s.write("a/b/c", Bytes::from_static(b"x"), WriteOptions::ATOMIC_CREATE_NEW).await.unwrap_err();
        assert!(e.is_already_exists());
        let e = s.write("a/b/c", Bytes::from_static(b"x"), WriteOptions::CREATE_NEW).await.unwrap_err();
        assert!(e.is_already_exists());
        assert_eq!(&s.read("a/b/c").await.unwrap()[..], b"hello world");
        assert_eq!(&s.read_range("a/b/c", 6, 100).await.unwrap()[..], b"world");
        assert!(s.exists("a/b/c").await.unwrap());
        assert!(s.read("a/zz").await.unwrap_err().is_not_found());
        s.write("a/x1", Bytes::from_static(b"1"), WriteOptions::OVERWRITE).await.unwrap();
        s.write("a/y1", Bytes::from_static(b"22"), WriteOptions::OVERWRITE).await.unwrap();
        let l = s.list("a/", "x").await.unwrap();
        assert_eq!(l.into_iter().collect::<Vec<_>>(), vec![("x1".to_string(), 1)]);
        assert_eq!(s.list("a/", "").await.unwrap().len(), 2);
        assert_eq!(s.list_children("a/").await.unwrap(), vec!["b"]);
        s.delete_batch(&["a/x1".into(), "a/nope".into()]).await.unwrap();
        assert!(!s.exists("a/x1").await.unwrap());
        let r = s.delete_dir("a/").await.unwrap();
        assert_eq!(r, DeleteResult { blobs_deleted: 2, bytes_deleted: 13 });
        assert!(s.list_children("").await.unwrap().is_empty());
        assert!(s.read("../x").await.is_err());
    }
}
