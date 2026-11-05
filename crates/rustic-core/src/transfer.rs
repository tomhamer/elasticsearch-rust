//! Export / import of an index's sealed segments (file-based moves,
//! peer recovery and segment replication).
//!
//! A commit point (`meta.json`) references immutable files: segment files
//! (`seg_XXXXXXXX.rstc`), their deletion sidecars (`seg_XXXXXXXX.del`,
//! rewritten at commit, absent when a segment has no deletes) and the
//! vector store's segment files (`vec_XXXXXXXX.rvec`: vectors and their
//! HNSW graphs, copied verbatim, never rebuilt by a transfer), the mutable
//! graph checkpoints (`.rmvc`) and the DiskBBQ index files (`ivf_*.rivc`,
//! `ivf_*.rivp`), opened in place by the receiver.
//!
//! [`snapshot_commit`] pins one commit: it reads `meta.json` and maps every
//! file it references, so the snapshot stays readable (and consistent)
//! even after later commits replace or delete those files. Its
//! [`CommitManifest`] lists every file with length and CRC-32.
//!
//! The receiving side writes files with [`import_commit`] (or
//! [`CommitSnapshot::copy_to`] for a local copy): each file is checked
//! against the manifest, files already present with the same checksum are
//! skipped (incremental segment replication), and `meta.json` is written
//! **last**, so a crash mid-transfer leaves the destination's previous
//! commit point intact. [`remove_unreferenced`] then deletes leftovers.

use crate::codec::directory::{Directory, FileIdentity, FileSlice};
use crate::codec::{crc32, deletes_file_name, segment_file_name, vector_segment_file_name};
use crate::error::{Result, RusticError};
use crate::index::Index;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use crate::admission::{self, Allocation};

/// The commit-point file name.
pub const COMMIT_FILE: &str = "meta.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileKind {
    CommitPoint,
    Segment,
    Deletes,
    VectorSegment,
    MutableVectorCheckpoint,
    /// A mutable graph's code log chunk (`mvq_*.rmvq`), referenced by its
    /// checkpoint.
    MutableVectorCodeLog,
    /// A DiskBBQ index file (`ivf_*.rivc` centroids, `ivf_*.rivp` postings).
    IvfIndex,
}

/// The kind of a vector-store file a commit's manifests reference (mutable
/// graph checkpoints, their code log chunks, DiskBBQ index files).
fn vector_file_kind(name: &str) -> FileKind {
    if crate::vector::store::is_code_log_file(name) {
        FileKind::MutableVectorCodeLog
    } else if crate::vector::ivf_field::is_ivf_file(name) {
        FileKind::IvfIndex
    } else {
        FileKind::MutableVectorCheckpoint
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitFile {
    pub name: String,
    pub len: u64,
    /// CRC-32 (IEEE) of the whole file.
    pub crc32: u32,
    pub kind: FileKind,
}

/// Every file of one commit point; `meta.json` is last.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitManifest {
    pub files: Vec<CommitFile>,
    /// Text segment ids, in commit order.
    pub segments: Vec<u32>,
    /// Vector segment ids.
    pub vector_segments: Vec<u32>,
}

impl CommitManifest {
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.len).sum()
    }

    pub fn get(&self, name: &str) -> Option<&CommitFile> {
        self.files.iter().find(|f| f.name == name)
    }
}

/// A pinned commit point. Clones share immutable data and its reservation.
#[derive(Clone)]
pub struct CommitSnapshot { inner: Arc<Allocation<SnapshotData>> }
struct SnapshotData { manifest: CommitManifest, data: HashMap<String, FileSlice> }

impl std::fmt::Debug for CommitSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CommitSnapshot({} files, {} bytes)", self.inner.manifest.files.len(), self.inner.manifest.total_bytes())
    }
}

#[derive(Deserialize)]
struct ReferencedSegment { id: u32 }
#[derive(Default, Deserialize)]
struct ReferencedVectors {
    #[serde(default)] segments: Vec<u32>,
    #[serde(default)] mutable_fields: Vec<crate::vector::store::MutableFieldManifest>,
    #[serde(default)] ivf_fields: Vec<crate::vector::ivf_field::IvfFieldManifest>,
}
#[derive(Deserialize)]
struct ReferencedMeta {
    segments: Vec<ReferencedSegment>,
    #[serde(default)] vector_store: ReferencedVectors,
}
struct References { segments: Vec<u32>, vectors: Vec<u32>, files: Vec<String> }

/// Decode only transfer fields, skipping schema/settings without a Value tree.
/// The returned names and ID arrays retain their own reservation.
fn referenced(meta: &[u8]) -> Result<Allocation<References>> {
    let _decode = admission::shared().reserve((meta.len() as u64).saturating_mul(16).saturating_add(4096), "transfer metadata decode")?;
    let decoded: ReferencedMeta = serde_json::from_slice(meta).map_err(|e| RusticError::Corrupted(format!("meta.json: {e}")))?;
    let count = decoded.vector_store.ivf_fields.iter().fold(decoded.vector_store.mutable_fields.iter().map(|m| 1 + m.code_files.len() as u64).sum::<u64>(),
        |n, m| n.saturating_add(1).saturating_add(m.batches.len() as u64));
    let bytes = count.saturating_mul(256).saturating_add((decoded.segments.len() as u64).saturating_mul(8))
        .saturating_add((decoded.vector_store.segments.capacity() as u64).saturating_mul(4)).saturating_add(4096);
    let memory = admission::shared().reserve(bytes, "transfer referenced files")?;
    let segments = decoded.segments.into_iter().map(|s| s.id).collect();
    let mut files: Vec<String> = decoded.vector_store.mutable_fields.iter().flat_map(|m| m.files()).collect();
    files.extend(decoded.vector_store.ivf_fields.iter().flat_map(|m| m.files()));
    Ok(Allocation::new(References { segments, vectors: decoded.vector_store.segments, files }, memory))
}

/// Pin the last commit point of `dir`. Callers must make sure no commit
/// runs concurrently *while this call executes* (a concurrent commit could
/// delete a merged-away file between reading `meta.json` and mapping it);
/// once it returns, the snapshot is independent of later commits.
pub fn snapshot_commit(dir: &dyn Directory) -> Result<CommitSnapshot> {
    let meta = dir.open_read(COMMIT_FILE)?;
    let references = referenced(meta.as_slice())?;
    let count = references.segments.len().saturating_mul(2).saturating_add(references.vectors.len()).saturating_add(references.files.len()).saturating_add(1);
    let memory = admission::shared().reserve((count as u64).saturating_mul(1024).saturating_add(4096), "transfer commit snapshot")?;
    let mut files = Vec::with_capacity(count);
    let mut data = HashMap::with_capacity(count);
    let mut add = |name: String, kind: FileKind, slice: FileSlice| {
        files.push(CommitFile { name: name.clone(), len: slice.len() as u64, crc32: crc32(slice.as_slice()), kind });
        data.insert(name, slice);
    };
    for &id in &references.segments {
        let name = segment_file_name(id);
        add(name.clone(), FileKind::Segment, dir.open_read(&name)?);
        let del = deletes_file_name(id);
        if dir.exists(&del) { add(del.clone(), FileKind::Deletes, dir.open_read(&del)?); }
    }
    for &id in &references.vectors {
        let name = vector_segment_file_name(id);
        add(name.clone(), FileKind::VectorSegment, dir.open_read(&name)?);
    }
    for name in &references.files { add(name.clone(), vector_file_kind(name), dir.open_read(name)?); }
    add(COMMIT_FILE.to_string(), FileKind::CommitPoint, meta);
    let manifest = CommitManifest { files, segments: references.segments.clone(), vector_segments: references.vectors.clone() };
    Ok(CommitSnapshot { inner: Arc::new(Allocation::new(SnapshotData { manifest, data }, memory)) })
}

/// One immutable pin owns its name/descriptor metadata across consumer clones.
pub type PinnedFile = Arc<Allocation<PinnedFileData>>;
pub struct PinnedFileData {
    pub name: String,
    pub kind: FileKind,
    pub data: FileSlice,
    pub identity: Option<FileIdentity>,
}
/// The pin registry also owns its vector storage until the caller is done.
pub type PinnedCommit = Allocation<Vec<PinnedFile>>;

/// Map every file the last commit point of `dir` references, `meta.json`
/// last, without reading them: the cheap part of [`snapshot_commit`], for
/// callers that hold a lock while pinning (a 30 GB graph checkpoint is
/// checksummed later, off the lock). Mapped files stay readable after a
/// later commit deletes them. Same concurrency contract as
/// [`snapshot_commit`].
pub fn pin_commit(dir: &dyn Directory) -> Result<PinnedCommit> {
    let pin = |name: &str, kind: FileKind| -> Result<PinnedFile> {
        let memory = admission::shared().reserve((name.len() as u64).saturating_add(512), "transfer file pin")?;
        let (data, identity) = dir.open_read_identified(&name)?;
        Ok(Arc::new(Allocation::new(PinnedFileData { name: name.to_owned(), kind, data, identity }, memory)))
    };
    let commit = pin(COMMIT_FILE, FileKind::CommitPoint)?;
    let references = referenced(commit.data.as_slice())?;
    let count = references.segments.len().saturating_mul(2).saturating_add(references.vectors.len()).saturating_add(references.files.len()).saturating_add(1);
    let memory = admission::shared().reserve((count as u64).saturating_mul(std::mem::size_of::<PinnedFile>() as u64), "transfer pin registry")?;
    let _formatted_names = admission::shared().reserve(256, "transfer formatted numeric filenames")?;
    let mut out = Vec::with_capacity(count);
    for &id in &references.segments {
        out.push(pin(&segment_file_name(id), FileKind::Segment)?);
        let del = deletes_file_name(id);
        if dir.exists(&del) { out.push(pin(&del, FileKind::Deletes)?); }
    }
    for &id in &references.vectors { out.push(pin(&vector_segment_file_name(id), FileKind::VectorSegment)?); }
    for name in &references.files { out.push(pin(name, vector_file_kind(name))?); }
    out.push(commit);
    Ok(Allocation::new(out, memory))
}

impl Index {
    /// [`snapshot_commit`] of this index's directory: the **last commit**
    /// (call [`Index::commit`] first to include recent changes). The caller
    /// must hold off commits for the duration of the call (the engine holds
    /// its shard write lock).
    pub fn snapshot_commit(&self) -> Result<CommitSnapshot> {
        snapshot_commit(&**self.directory())
    }

    /// [`pin_commit`] of this index's directory (the last commit).
    pub fn pin_commit(&self) -> Result<PinnedCommit> {
        pin_commit(&**self.directory())
    }
}

/// Statistics of a copy / import.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransferStats {
    pub files_copied: usize,
    pub bytes_copied: u64,
    pub files_reused: usize,
    pub bytes_reused: u64,
}

impl CommitSnapshot {
    pub fn manifest(&self) -> &CommitManifest {
        &self.inner.manifest
    }

    /// The pinned bytes of one file of the manifest.
    pub fn file(&self, name: &str) -> Option<&FileSlice> {
        self.inner.data.get(name)
    }

    /// Copy into `dst` (incremental: matching files are reused), commit
    /// point last.
    pub fn copy_to(&self, dst: &dyn Directory) -> Result<TransferStats> {
        import_commit(dst, &self.inner.manifest, |f, buffer| {
            let source = self.inner.data.get(&f.name).ok_or_else(|| RusticError::IndexNotFound(format!("file {}", f.name)))?;
            buffer.copy_from_slice(source.as_slice());
            Ok(())
        })
    }
}

fn check(f: &CommitFile, data: &[u8]) -> Result<()> {
    if data.len() as u64 != f.len {
        return Err(RusticError::Corrupted(format!("{}: length {} != expected {}", f.name, data.len(), f.len)));
    }
    let c = crc32(data);
    if c != f.crc32 {
        return Err(RusticError::Corrupted(format!("{}: checksum {c:08x} != expected {:08x}", f.name, f.crc32)));
    }
    Ok(())
}

/// Whether `dst` already holds `f` byte-for-byte.
fn present(dst: &dyn Directory, f: &CommitFile) -> bool {
    dst.exists(&f.name) && dst.open_read(&f.name).map(|d| check(f, d.as_slice()).is_ok()).unwrap_or(false)
}

/// Files of `manifest` that `dst` lacks (or holds with other contents).
pub fn missing_files(dst: &dyn Directory, manifest: &CommitManifest) -> Result<Allocation<Vec<CommitFile>>> {
    let bytes = manifest.files.iter().fold(256u64, |n, f| n.saturating_add(f.name.len() as u64).saturating_add(128));
    let memory = admission::shared().reserve(bytes, "missing transfer file list")?;
    let files = manifest.files.iter().filter(|f| f.kind == FileKind::CommitPoint || !present(dst, f)).cloned().collect();
    Ok(Allocation::new(files, memory))
}

/// Write a commit into `dst`, fetching each needed file's bytes with
/// `fetch` into an already admitted, exactly sized buffer and verifying it
/// against the manifest; the commit point goes
/// last. Files already present with the right checksum are not fetched.
pub fn import_commit(dst: &dyn Directory, manifest: &CommitManifest, mut fetch: impl FnMut(&CommitFile, &mut [u8]) -> Result<()>) -> Result<TransferStats> {
    let mut st = TransferStats::default();
    let mut commit = None;
    for f in &manifest.files {
        if f.kind == FileKind::CommitPoint {
            commit = Some(f);
            continue;
        }
        if present(dst, f) {
            st.files_reused += 1;
            st.bytes_reused += f.len;
            continue;
        }
        let _memory = admission::shared().reserve(f.len.saturating_mul(2), "transfer imported file buffer")?;
        let length = usize::try_from(f.len).map_err(|_| RusticError::Memory("transfer file length overflow".into()))?;
        let mut data = vec![0u8; length];
        fetch(f, &mut data)?;
        check(f, &data)?;
        dst.write(&f.name, &data)?;
        st.files_copied += 1;
        st.bytes_copied += f.len;
    }
    let f = commit.ok_or_else(|| RusticError::Corrupted("manifest without a commit point".into()))?;
    let _memory = admission::shared().reserve(f.len.saturating_mul(2), "transfer commit file buffer")?;
    let length = usize::try_from(f.len).map_err(|_| RusticError::Memory("transfer file length overflow".into()))?;
    let mut data = vec![0u8; length];
    fetch(f, &mut data)?;
    check(f, &data)?;
    dst.write(&f.name, &data)?;
    st.files_copied += 1;
    st.bytes_copied += f.len;
    Ok(st)
}

/// Check every file of `manifest` in `dir` (length + checksum).
pub fn verify(dir: &dyn Directory, manifest: &CommitManifest) -> Result<()> {
    for f in &manifest.files {
        let d = dir.open_read(&f.name)?;
        check(f, d.as_slice())?;
    }
    Ok(())
}

/// Delete index files (`seg_*`, `vec_*`, mutable graph checkpoints and
/// their code log chunks, DiskBBQ files) of `dir` that `manifest` does not
/// reference. Returns the deleted names.
pub fn remove_unreferenced(dir: &dyn Directory, manifest: &CommitManifest) -> Result<crate::codec::directory::AdmittedNames> {
    let _lookup_memory=admission::shared().reserve((manifest.files.len() as u64).saturating_mul(64).saturating_add(256), "transfer cleanup referenced file lookup")?;
    let keep:std::collections::HashSet<&str>=manifest.files.iter().map(|file|file.name.as_str()).collect();
    let mut out = crate::codec::directory::AdmittedNames::new()?;
    let names=dir.list_admitted(false)?;
    for name in names.iter() {
        let ours = (name.starts_with("seg_") && (name.ends_with(".rstc") || name.ends_with(".del"))) || (name.starts_with("vec_") && name.ends_with(".rvec")) || (name.starts_with("mutable_") && name.ends_with(".rmvc"))
            || crate::vector::store::is_code_log_file(name) || crate::vector::ivf_field::is_ivf_file(&name);
        if ours && !keep.contains(name.as_str()) {
            // Prepare returned ownership before mutating durable files.
            out.push(name)?;
            dir.delete(name)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::directory::{MmapDirectory, RamDirectory};
    use crate::types::*;

    #[cfg(feature = "mutable-hnsw-poc")]
    #[test]
    fn mutable_checkpoint_transfers_to_empty_directory_and_survives_new_commits() {
        use std::sync::Arc;
        use crate::vector::store::VectorFieldOptions;
        use crate::index::KnnRequest;
        let source = Arc::new(RamDirectory::default());
        let schema = vec![Field::new(1, "vector", FieldType::Vector { dim: 2 })
            .with_similarity("l2_norm").with_vector_options(Some(VectorFieldOptions::f32_hnsw()))];
        let mut index = Index::create_in(source.clone(), "transfer", schema, IndexSettings::default()).unwrap();
        index.enable_mutable_vectors().unwrap();
        for i in 0..30 {
            let mut doc = Document::new(); doc.add_vector(1, vec![i as f32, 1.]);
            index.add_document(&doc).unwrap();
        }
        index.commit().unwrap();
        index.refresh().unwrap();
        index.commit().unwrap();
        let snapshot = index.snapshot_commit().unwrap();
        assert!(!snapshot.inner.manifest.segments.is_empty(), "transfer fixture must seal text documents");
        assert_eq!(snapshot.inner.manifest.files.iter().filter(|f| f.kind == FileKind::MutableVectorCheckpoint).count(), 1);
        let query = KnnRequest::new(1, vec![0., 1.], 10);
        let expected: Vec<_> = index.knn_search(&query).unwrap().into_iter().map(|h| h.score).collect();
        index.commit().unwrap(); // old checkpoint can be unlinked; snapshot pins it.
        let destination = Arc::new(RamDirectory::default());
        snapshot.copy_to(&*destination).unwrap();
        verify(&*destination, snapshot.manifest()).unwrap();
        let restored = Index::open_in(destination.clone()).unwrap();
        assert_eq!(restored.knn_search(&query).unwrap().into_iter().map(|h| h.score).collect::<Vec<_>>(), expected);
        destination.write("mutable_ffffffffffffffff_0001.rmvc", b"obsolete").unwrap();
        // An orphan code log chunk (of a checkpoint no commit lists) is swept too.
        destination.write("mvq_ffffffffffffffff_00000000_00000001.rmvq", b"obsolete").unwrap();
        let mut removed: Vec<String> = remove_unreferenced(&*destination, snapshot.manifest()).unwrap().iter().cloned().collect();
        removed.sort();
        assert_eq!(removed, vec!["mutable_ffffffffffffffff_0001.rmvc".to_string(), "mvq_ffffffffffffffff_00000000_00000001.rmvq".to_string()]);
        verify(&*destination, snapshot.manifest()).unwrap();
        // Referenced code log chunks and DiskBBQ files carry their own kind.
        for f in &snapshot.manifest().files {
            match f.kind {
                FileKind::MutableVectorCodeLog => assert!(crate::vector::store::is_code_log_file(&f.name), "{}", f.name),
                FileKind::IvfIndex => assert!(crate::vector::ivf_field::is_ivf_file(&f.name), "{}", f.name),
                FileKind::MutableVectorCheckpoint => assert!(f.name.ends_with(".rmvc"), "{}", f.name),
                _ => {}
            }
        }
    }

    fn build(dir: &std::path::Path) -> Index {
        let schema = vec![Field::new(0, "title", FieldType::Text), Field::new(1, "n", FieldType::I64)];
        let mut idx = Index::create(dir, "t", schema, IndexSettings::default()).unwrap();
        for round in 0..3 {
            for i in 0..20 {
                let mut d = Document::new();
                d.add_text(0, &format!("hello world r{round} d{i}"));
                d.add_i64(1, i);
                idx.add_document(&d).unwrap();
            }
            idx.refresh().unwrap();
        }
        idx.delete_doc(&DocAddress::new(1, 3)).unwrap();
        idx.commit().unwrap();
        idx
    }

    #[test]
    fn snapshot_copy_open_roundtrip() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let mut idx = build(a.path());
        let snap = idx.snapshot_commit().unwrap();
        let m = snap.manifest().clone();
        assert_eq!(m.segments.len(), 3);
        assert_eq!(m.files.last().unwrap().kind, FileKind::CommitPoint);
        assert!(m.files.iter().any(|f| f.kind == FileKind::Deletes));
        // Later commits (merge deletes the old files) do not affect the snapshot.
        idx.force_merge().unwrap();
        idx.commit().unwrap();
        assert!(!a.path().join(segment_file_name(0)).exists());
        let dst = MmapDirectory::open(b.path()).unwrap();
        let st = snap.copy_to(&dst).unwrap();
        assert_eq!(st.files_copied, m.files.len());
        verify(&dst, &m).unwrap();
        let copy = Index::open(b.path()).unwrap();
        assert_eq!(copy.num_docs(), 60);
        assert_eq!(copy.num_live_docs(), 59);
        assert_eq!(copy.search_term(&Term::new_text(0, "hello"), 100).unwrap().total_hits, 59);
        // Incremental: a second copy reuses everything but the commit point.
        let st = snap.copy_to(&dst).unwrap();
        assert_eq!((st.files_copied, st.files_reused), (1, m.files.len() - 1));
        // The merged commit replaces it; leftovers are removed.
        let snap2 = idx.snapshot_commit().unwrap();
        snap2.copy_to(&dst).unwrap();
        let removed = remove_unreferenced(&dst, snap2.manifest()).unwrap();
        assert!(removed.contains(&segment_file_name(0)), "{removed:?}");
        let copy = Index::open(b.path()).unwrap();
        assert_eq!(copy.num_segments(), 1);
        assert_eq!(copy.num_live_docs(), 59);
    }

    #[test]
    fn corrupt_transfer_is_rejected_and_keeps_old_commit() {
        let a = tempfile::tempdir().unwrap();
        let idx = build(a.path());
        let snap = idx.snapshot_commit().unwrap();
        let dst = RamDirectory::new();
        let err = import_commit(&dst, snap.manifest(), |f, d| {
            d.copy_from_slice(snap.file(&f.name).unwrap().as_slice());
            if f.kind == FileKind::Segment {
                d[20] ^= 0xff;
            }
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(err, RusticError::Corrupted(_)), "{err}");
        assert!(!dst.exists(COMMIT_FILE), "commit point written only after every file verified");
        assert!(verify(&dst, snap.manifest()).is_err());
    }
}

#[cfg(test)]
mod admitted_metadata_tests {
    use super::*;
    use crate::codec::directory::RamDirectory;

    fn directory() -> RamDirectory {
        let dir = RamDirectory::new();
        dir.write(COMMIT_FILE, br#"{"segments":[],"vector_store":{"segments":[]}}"#).unwrap();
        dir
    }

    #[test]
    fn individual_pin_retains_its_owner_after_registry_drops() {
        let dir = directory();
        let owner = admission::Owner::new();
        let _scope = owner.enter();
        let pins = pin_commit(&dir).unwrap();
        let pin = pins[0].clone();
        drop(pins);
        assert!(owner.bytes().current > 0);
        assert_eq!(pin.name, COMMIT_FILE);
        drop(pin);
        assert_eq!(owner.bytes().current, 0);
    }

    #[test]
    fn snapshot_clones_share_admitted_metadata_until_last_drop() {
        let dir = directory();
        let owner = admission::Owner::new();
        let _scope = owner.enter();
        let first = snapshot_commit(&dir).unwrap();
        let bytes = owner.bytes().current;
        let second = first.clone();
        assert!(Arc::ptr_eq(&first.inner, &second.inner));
        assert_eq!(owner.bytes().current, bytes);
        drop(first);
        assert_eq!(owner.bytes().current, bytes);
        assert!(second.file(COMMIT_FILE).is_some());
        drop(second);
        assert_eq!(owner.bytes().current, 0);
    }
}
