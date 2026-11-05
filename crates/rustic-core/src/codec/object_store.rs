//! A [`Directory`] over an object store (sharding phase 1, stateless mode).
//!
//! [`ObjectStore`] is a minimal blocking blob API (get / ranged get / put /
//! delete / list / exists). [`ObjectStoreDirectory`] maps directory files
//! to objects `<prefix><name>`: every write is one `put` (object stores
//! replace objects atomically, which is the `Directory::write` contract),
//! reads fetch the whole object once and serve it from a cache, in memory
//! or (with [`ObjectStoreDirectory::with_cache_dir`]) as a memory-mapped
//! local file. Index files are immutable once written except `meta.json`
//! and `.del` sidecars, which this directory rewrites through its own
//! cache, so a single writer always reads its own writes. Other readers of
//! the same prefix call [`ObjectStoreDirectory::invalidate`] before reopening.
//!
//! Backends: [`LocalObjectStore`] (a directory tree; the test double, and
//! usable against any mounted bucket), and with the `s3` cargo feature
//! [`s3::S3ObjectStore`] (S3-compatible, AWS SigV4). The S3 backend has no
//! extra crate dependencies: signing is implemented here and HTTP goes
//! through the [`s3::HttpTransport`] trait (a plain-HTTP transport is
//! included for MinIO / in-VPC endpoints; plug an HTTPS client for AWS).

use super::directory::{Directory, FileSlice};
use crate::error::{Result, RusticError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A blocking object store.
pub trait ObjectStore: Send + Sync + fmt::Debug {
    /// The whole object; `RusticError::IndexNotFound` if absent.
    fn get(&self, key: &str) -> Result<Vec<u8>>;
    /// Bytes `range` of the object.
    fn get_range(&self, key: &str, range: Range<u64>) -> Result<Vec<u8>> {
        let all = self.get(key)?;
        let end = (range.end as usize).min(all.len());
        let start = (range.start as usize).min(end);
        Ok(all[start..end].to_vec())
    }
    /// Create or atomically replace the object.
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;
    /// Delete; absent objects are not an error.
    fn delete(&self, key: &str) -> Result<()>;
    /// Keys starting with `prefix`, sorted.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;
    fn list_admitted(&self, prefix: &str) -> Result<super::directory::AdmittedNames>;
    fn exists(&self, key: &str) -> Result<bool>;
}

fn not_found(key: &str) -> RusticError {
    RusticError::IndexNotFound(format!("object {key}"))
}

/// Objects as files under a root directory (`key` = relative path).
#[derive(Debug)]
pub struct LocalObjectStore {
    root: PathBuf,
}

impl LocalObjectStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        std::fs::create_dir_all(root.as_ref())?;
        Ok(Self { root: root.as_ref().to_path_buf() })
    }

    fn path(&self, key: &str) -> Result<PathBuf> {
        if key.split('/').any(|s| s == ".." || s == ".") || key.starts_with('/') {
            return Err(RusticError::InvalidQuery(format!("bad object key [{key}]")));
        }
        Ok(self.root.join(key))
    }

    fn walk(&self, dir: &Path, rel: &str, out: &mut Vec<String>) -> Result<()> {
        let rd = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for e in rd {
            let e = e?;
            let Some(name) = e.file_name().to_str().map(str::to_string) else { continue };
            let key = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
            if e.file_type()?.is_dir() {
                self.walk(&e.path(), &key, out)?;
            } else if !name.ends_with(".part~") {
                out.push(key);
            }
        }
        Ok(())
    }
}

impl ObjectStore for LocalObjectStore {
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        std::fs::read(self.path(key)?).map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { not_found(key) } else { e.into() })
    }

    fn get_range(&self, key: &str, range: Range<u64>) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(self.path(key)?).map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { not_found(key) } else { RusticError::from(e) })?;
        let len = f.metadata()?.len();
        let (start, end) = (range.start.min(len), range.end.min(len));
        f.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; (end - start) as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        use std::io::Write;
        let p = self.path(key)?;
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = p.with_file_name(format!("{}.part~", p.file_name().and_then(|n| n.to_str()).unwrap_or("obj")));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(data)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        match std::fs::remove_file(self.path(key)?) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn list_admitted(&self, prefix: &str) -> Result<super::directory::AdmittedNames> {
        fn walk(dir:&Path, rel:&str, prefix:&str, out:&mut super::directory::AdmittedNames)->Result<()> {
            let entries=match std::fs::read_dir(dir) { Ok(v)=>v,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>return Ok(()),Err(e)=>return Err(e.into()) };
            for entry in entries {
                let entry=entry?;let name=entry.file_name();let Some(name)=name.to_str() else {continue};
                let size=rel.len().saturating_add(name.len()).saturating_add(dir.as_os_str().as_encoded_bytes().len()).saturating_add(4);
                let _memory=crate::admission::shared().reserve((size as u64).saturating_mul(2),"object listing path producer")?;
                let key=if rel.is_empty(){name.to_owned()}else{format!("{rel}/{name}")};
                if entry.file_type()?.is_dir(){walk(&entry.path(),&key,prefix,out)?;}
                else if !name.ends_with(".part~") && key.starts_with(prefix){out.push(&key)?;}
            } Ok(())
        }
        let mut out=super::directory::AdmittedNames::new()?;walk(&self.root,"",prefix,&mut out)?;out.sort();Ok(out)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut out = Vec::new();
        self.walk(&self.root, "", &mut out)?;
        out.retain(|k| k.starts_with(prefix));
        out.sort();
        Ok(out)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.path(key)?.is_file())
    }
}

/// A [`Directory`] whose files are objects (see the module docs).
pub struct ObjectStoreDirectory {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    cache: RwLock<HashMap<String, FileSlice>>,
    cache_dir: Option<PathBuf>,
}

impl fmt::Debug for ObjectStoreDirectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectStoreDirectory({:?}, prefix={:?})", self.store, self.prefix)
    }
}

impl ObjectStoreDirectory {
    /// Files map to `<prefix><name>`; a non-empty prefix normally ends in `/`.
    pub fn new(store: Arc<dyn ObjectStore>, prefix: impl Into<String>) -> Self {
        Self { store, prefix: prefix.into(), cache: RwLock::new(HashMap::new()), cache_dir: None }
    }

    /// Keep fetched files as local files under `dir` (memory-mapped)
    /// instead of in memory.
    pub fn with_cache_dir(mut self, dir: impl AsRef<Path>) -> Result<Self> {
        std::fs::create_dir_all(dir.as_ref())?;
        self.cache_dir = Some(dir.as_ref().to_path_buf());
        Ok(self)
    }

    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    fn key(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    /// Drop cached contents (all files, or one).
    pub fn invalidate(&self, name: Option<&str>) {
        match name {
            Some(n) => {
                self.cache.write().remove(n);
            }
            None => self.cache.write().clear(),
        }
    }

    fn cached_slice(&self, name: &str, data: Vec<u8>) -> Result<FileSlice> {
        let Some(dir) = &self.cache_dir else { return Ok(FileSlice::from_vec(data)) };
        if data.is_empty() {
            return Ok(FileSlice::empty());
        }
        use std::io::Write;
        let path = dir.join(name);
        let tmp = dir.join(format!("{name}.tmp"));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&data)?;
        }
        std::fs::rename(&tmp, &path)?;
        let f = std::fs::File::open(&path)?;
        if !crate::codec::directory::mmap_allowed() {
            // `node.store.allow_mmap: false`: read into memory and charged
            // like any store file (the cache file stays for the next open).
            drop(data);
            return crate::codec::directory::read_unmapped(&f, name);
        }
        // SAFETY: cache files are written once (tmp + rename) and replaced,
        // never modified in place.
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        Ok(FileSlice::new(Arc::new(mmap)))
    }
}

impl Directory for ObjectStoreDirectory {
    fn open_read(&self, name: &str) -> Result<FileSlice> {
        if let Some(s) = self.cache.read().get(name) {
            return Ok(s.clone());
        }
        let data = self.store.get(&self.key(name)).map_err(|e| match e {
            RusticError::IndexNotFound(_) => RusticError::IndexNotFound(format!("file {name}")),
            e => e,
        })?;
        let slice = self.cached_slice(name, data)?;
        self.cache.write().insert(name.to_string(), slice.clone());
        Ok(slice)
    }

    fn write(&self, name: &str, data: &[u8]) -> Result<()> {
        self.store.put(&self.key(name), data)?;
        let slice = self.cached_slice(name, data.to_vec())?;
        self.cache.write().insert(name.to_string(), slice);
        Ok(())
    }

    fn exists(&self, name: &str) -> bool {
        self.cache.read().contains_key(name) || self.store.exists(&self.key(name)).unwrap_or(false)
    }

    fn delete(&self, name: &str) -> Result<()> {
        self.cache.write().remove(name);
        if let Some(d) = &self.cache_dir {
            let _ = std::fs::remove_file(d.join(name));
        }
        self.store.delete(&self.key(name))
    }

    fn list_admitted(&self, _temporary: bool) -> Result<super::directory::AdmittedNames> {
        let keys=self.store.list_admitted(&self.prefix)?;
        let mut out=super::directory::AdmittedNames::new()?;
        for key in keys.iter(){if let Some(name)=key.strip_prefix(&self.prefix){if !name.contains('/') && !name.ends_with(".tmp"){out.push(name)?;}}}
        out.sort();Ok(out)
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut v: Vec<String> = self
            .store
            .list(&self.prefix)?
            .into_iter()
            .filter_map(|k| k.strip_prefix(&self.prefix).map(str::to_string))
            .filter(|n| !n.contains('/') && !n.ends_with(".tmp"))
            .collect();
        v.sort();
        Ok(v)
    }
}

#[cfg(feature = "s3")]
pub mod s3;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn local_store_basics() {
        let tmp = tempfile::tempdir().unwrap();
        let s = LocalObjectStore::open(tmp.path()).unwrap();
        s.put("a/b/c", b"hello").unwrap();
        s.put("a/x", b"0123456789").unwrap();
        assert_eq!(s.get("a/b/c").unwrap(), b"hello");
        assert_eq!(s.get_range("a/x", 2..5).unwrap(), b"234");
        assert_eq!(s.list("a/").unwrap(), vec!["a/b/c".to_string(), "a/x".to_string()]);
        assert!(s.exists("a/x").unwrap());
        s.delete("a/x").unwrap();
        s.delete("a/x").unwrap();
        assert!(matches!(s.get("a/x"), Err(RusticError::IndexNotFound(_))));
        assert!(s.put("../escape", b"x").is_err());
    }

    #[test]
    fn index_on_object_store_directory() {
        let bucket = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalObjectStore::open(bucket.path()).unwrap());
        let dir = ObjectStoreDirectory::new(store.clone(), "indices/i1/");
        // Generic directory contract.
        dir.write("a", b"hello world").unwrap();
        assert_eq!(dir.open_read("a").unwrap().as_slice(), b"hello world");
        dir.write("a", b"replaced").unwrap();
        assert_eq!(dir.open_read("a").unwrap().as_slice(), b"replaced");
        assert_eq!(dir.list().unwrap(), vec!["a".to_string()]);
        dir.delete("a").unwrap();
        assert!(!dir.exists("a"));
        // A real index: build locally, transfer its commit into the bucket,
        // open it from a fresh object-store directory with a disk cache.
        let local = tempfile::tempdir().unwrap();
        let schema = vec![Field::new(0, "t", FieldType::Text)];
        let mut idx = crate::Index::create(local.path(), "o", schema, IndexSettings::default()).unwrap();
        for i in 0..30 {
            let mut d = Document::new();
            d.add_text(0, &format!("object store doc {i}"));
            idx.add_document(&d).unwrap();
        }
        idx.refresh().unwrap();
        idx.delete_doc(&DocAddress::new(0, 1)).unwrap();
        idx.commit().unwrap();
        let snap = idx.snapshot_commit().unwrap();
        snap.copy_to(&dir).unwrap();
        let fresh = ObjectStoreDirectory::new(store, "indices/i1/").with_cache_dir(cache.path()).unwrap();
        crate::transfer::verify(&fresh, snap.manifest()).unwrap();
        let opened = crate::Index::open_in(Arc::new(fresh)).unwrap();
        assert_eq!(opened.num_live_docs(), 29);
        assert_eq!(opened.search_term(&Term::new_text(0, "store"), 100).unwrap().total_hits, 29);
    }
}
