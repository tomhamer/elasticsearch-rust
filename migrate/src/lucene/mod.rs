//! Read-only reader for the parts of a Lucene 9.x index (the format written
//! by Elasticsearch 8.x) that an upgrade needs: the latest commit, segment
//! infos, compound files, field infos, live docs (hard and soft deletes) and
//! stored fields. Search structures (postings, points, vectors, doc values
//! other than `__soft_deletes`) are never read: the new engine rebuilds them
//! from `_source`.

pub mod compress;
pub mod dv;
pub mod io;
pub mod stored;

use anyhow::{anyhow, bail, Context, Result};
use io::{expect_codec, read_header, read_index_header, In, FOOTER_LEN};
use memmap2::Mmap;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A whole file (or a slice of a compound file) in memory.
#[derive(Clone)]
pub struct Bytes {
    map: Arc<Mmap>,
    start: usize,
    len: usize,
}

impl Bytes {
    pub fn as_slice(&self) -> &[u8] {
        &self.map[self.start..self.start + self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

fn map_file(path: &Path) -> Result<Bytes> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = f.metadata()?.len() as usize;
    if len == 0 {
        bail!("{} is empty", path.display());
    }
    // SAFETY: Lucene never rewrites a committed file in place.
    let map = unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {}", path.display()))?;
    Ok(Bytes { map: Arc::new(map), start: 0, len })
}

/// Footer: magic (inverted), algorithm id, checksum. Only structure is checked;
/// CRC verification would double the read cost and the source ES verified it.
fn check_footer(b: &[u8], file: &str) -> Result<()> {
    if b.len() < FOOTER_LEN {
        bail!("{}: too short for a codec footer", file);
    }
    let mut f = In::at(b, b.len() - FOOTER_LEN);
    if f.be_u32()? != !io::CODEC_MAGIC {
        bail!("{}: bad codec footer (file truncated or not a Lucene file)", file);
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SegmentCommit {
    pub name: String,
    pub codec: String,
    pub del_gen: i64,
    pub del_count: i32,
    pub soft_del_count: i32,
    pub field_infos_gen: i64,
    pub dv_gen: i64,
    pub field_infos_files: Vec<String>,
    pub dv_update_files: BTreeMap<i32, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Commit {
    pub generation: u64,
    pub lucene_version: (u32, u32, u32),
    pub created_major: u32,
    pub segments: Vec<SegmentCommit>,
    pub user_data: BTreeMap<String, String>,
}

/// Picks the highest `segments_N` in `dir` (what Lucene opens by default).
pub fn latest_commit_file(dir: &Path) -> Result<Option<(u64, PathBuf)>> {
    let mut best: Option<(u64, PathBuf)> = None;
    for e in std::fs::read_dir(dir).with_context(|| format!("list {}", dir.display()))? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(gen) = name.strip_prefix("segments_") else { continue };
        let Ok(g) = io::parse_base36(gen) else { continue };
        if best.as_ref().map_or(true, |(b, _)| g > *b) {
            best = Some((g, e.path()));
        }
    }
    Ok(best)
}

pub fn read_commit(path: &Path) -> Result<Commit> {
    let file = path.display().to_string();
    let bytes = map_file(path)?;
    let b = bytes.as_slice();
    check_footer(b, &file)?;
    let mut inp = In::new(b);
    let h = read_index_header(&mut inp)?;
    expect_codec(&h, &file, &["segments"])?;
    if h.version < 10 {
        bail!("{}: segments format version {} is older than Lucene 9 (Elasticsearch 7.x); upgrade the cluster to 8.x first", file, h.version);
    }
    let generation = io::parse_base36(&h.suffix)?;
    let lucene_version = (inp.vint()?, inp.vint()?, inp.vint()?);
    let created_major = inp.vint()?;
    let _version = inp.be_i64()?;
    let _counter = inp.vlong()?;
    let n = inp.be_i32()?;
    if n < 0 {
        bail!("{}: negative segment count", file);
    }
    if n > 0 {
        let _min_seg_version = (inp.vint()?, inp.vint()?, inp.vint()?);
    }
    let mut segments = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let name = inp.string()?;
        inp.skip(16)?; // segment id, always present since Lucene 9
        let codec = inp.string()?;
        let del_gen = inp.be_i64()?;
        let del_count = inp.be_i32()?;
        let field_infos_gen = inp.be_i64()?;
        let dv_gen = inp.be_i64()?;
        let soft_del_count = inp.be_i32()?;
        if inp.u8()? == 1 {
            inp.skip(16)?;
        }
        let field_infos_files = inp.string_set()?;
        let n_dv = inp.be_i32()?;
        let mut dv_update_files = BTreeMap::new();
        for _ in 0..n_dv {
            let field = inp.be_i32()?;
            dv_update_files.insert(field, inp.string_set()?);
        }
        segments.push(SegmentCommit {
            name,
            codec,
            del_gen,
            del_count,
            soft_del_count,
            field_infos_gen,
            dv_gen,
            field_infos_files,
            dv_update_files,
        });
    }
    let user_data = inp.string_map()?;
    Ok(Commit { generation, lucene_version, created_major, segments, user_data })
}

#[derive(Debug, Clone)]
pub struct SegmentInfo {
    pub max_doc: u32,
    pub compound: bool,
    pub attributes: BTreeMap<String, String>,
}

/// Files of one segment: plain files in the directory plus, for compound
/// segments, the sub-files of `.cfs` keyed by their stripped name (".fdt").
pub struct SegmentFiles {
    dir: PathBuf,
    name: String,
    compound: HashMap<String, Bytes>,
}

impl SegmentFiles {
    /// `ext` is the part after the segment name, e.g. ".fdt" or "_Lucene90_0.dvm".
    pub fn open(&self, ext: &str) -> Result<Bytes> {
        if let Some(b) = self.compound.get(ext) {
            return Ok(b.clone());
        }
        map_file(&self.dir.join(format!("{}{}", self.name, ext)))
    }

    /// A per-commit file such as `_3v_1.fnm` or `_3v_2_Lucene90_0.dvm`; these
    /// are never inside the compound file.
    pub fn open_plain(&self, file: &str) -> Result<Bytes> {
        map_file(&self.dir.join(file))
    }

    pub fn find_ext(&self, ext: &str) -> Option<String> {
        self.compound.keys().find(|k| k.ends_with(ext)).cloned()
    }
}

pub fn read_segment_info(dir: &Path, name: &str) -> Result<SegmentInfo> {
    let path = dir.join(format!("{}.si", name));
    let file = path.display().to_string();
    let bytes = map_file(&path)?;
    let b = bytes.as_slice();
    check_footer(b, &file)?;
    let mut inp = In::new(b);
    let h = read_index_header(&mut inp)?;
    expect_codec(&h, &file, &["Lucene90SegmentInfo"])?;
    let version = (inp.le_i32()?, inp.le_i32()?, inp.le_i32()?);
    if inp.u8()? == 1 {
        let _min = (inp.le_i32()?, inp.le_i32()?, inp.le_i32()?);
    }
    let max_doc = inp.le_i32()?;
    if max_doc < 0 {
        bail!("{}: negative maxDoc", file);
    }
    let compound = inp.u8()? == 1;
    // hasBlocks was added in Lucene 9.10 without a format version bump
    if h.version >= 1 || (version.0, version.1) >= (9, 10) {
        let _has_blocks = inp.u8()?;
    }
    let _diagnostics = inp.string_map()?;
    let _files = inp.string_set()?;
    let attributes = inp.string_map()?;
    Ok(SegmentInfo { max_doc: max_doc as u32, compound, attributes })
}

pub fn open_segment_files(dir: &Path, name: &str, compound: bool) -> Result<SegmentFiles> {
    let mut files = SegmentFiles { dir: dir.to_path_buf(), name: name.to_string(), compound: HashMap::new() };
    if !compound {
        return Ok(files);
    }
    let cfe_path = dir.join(format!("{}.cfe", name));
    let cfe = map_file(&cfe_path)?;
    let file = cfe_path.display().to_string();
    check_footer(cfe.as_slice(), &file)?;
    let mut inp = In::new(cfe.as_slice());
    let h = read_index_header(&mut inp)?;
    expect_codec(&h, &file, &["Lucene90CompoundEntries"])?;
    let data = map_file(&dir.join(format!("{}.cfs", name)))?;
    let n = inp.vint()?;
    for _ in 0..n {
        let id = inp.string()?;
        let off = inp.le_i64()? as usize;
        let len = inp.le_i64()? as usize;
        if off + len > data.len {
            bail!("{}: entry {} [{}+{}] outside .cfs of {} bytes", file, id, off, len, data.len);
        }
        files.compound.insert(id, Bytes { map: data.map.clone(), start: off, len });
    }
    Ok(files)
}

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub number: u32,
    pub has_dv_skip_index: bool,
    pub attributes: BTreeMap<String, String>,
}

pub fn read_field_infos(b: &[u8], file: &str) -> Result<Vec<FieldInfo>> {
    check_footer(b, file)?;
    let mut inp = In::new(b);
    let h = read_index_header(&mut inp)?;
    expect_codec(&h, file, &["Lucene94FieldInfos"])?;
    let n = inp.vint()?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let name = inp.string()?;
        let number = inp.vint()?;
        let _bits = inp.u8()?;
        let _index_options = inp.u8()?;
        let _dv_type = inp.u8()?;
        let has_dv_skip_index = if h.version >= 2 { inp.u8()? != 0 } else { false };
        let _dv_gen = inp.le_i64()?;
        let attributes = inp.string_map()?;
        let point_dims = inp.vint()?;
        if point_dims != 0 {
            let _index_dims = inp.vint()?;
            let _num_bytes = inp.vint()?;
        }
        let _vector_dims = inp.vint()?;
        let _vector_encoding = inp.u8()?;
        let _vector_similarity = inp.u8()?;
        out.push(FieldInfo { name, number, has_dv_skip_index, attributes });
    }
    if inp.remaining() != FOOTER_LEN {
        bail!("{}: {} unexpected trailing bytes after field infos", file, inp.remaining() - FOOTER_LEN);
    }
    Ok(out)
}

/// One searchable segment as of a commit, with every non-live doc removed.
pub struct Segment {
    pub name: String,
    pub max_doc: u32,
    pub fields: Vec<FieldInfo>,
    pub live: Option<Vec<u64>>,
    pub hard_deleted: u32,
    pub soft_deleted: u32,
    pub files: SegmentFiles,
    pub info: SegmentInfo,
}

impl Segment {
    pub fn is_live(&self, doc: u32) -> bool {
        match &self.live {
            None => true,
            Some(bits) => bits[(doc >> 6) as usize] >> (doc & 63) & 1 == 1,
        }
    }

    pub fn live_count(&self) -> u32 {
        match &self.live {
            None => self.max_doc,
            Some(bits) => bits.iter().map(|w| w.count_ones()).sum(),
        }
    }

    pub fn field_name(&self, number: u32) -> Option<&str> {
        self.fields.iter().find(|f| f.number == number).map(|f| f.name.as_str())
    }
}

pub const SOFT_DELETES_FIELD: &str = "__soft_deletes";

pub fn open_segment(dir: &Path, sc: &SegmentCommit) -> Result<Segment> {
    if !sc.codec.starts_with("Lucene9") && !sc.codec.starts_with("Elasticsearch8") && !sc.codec.starts_with("Elasticsearch9") {
        // Unknown codecs are still attempted: ES wraps Lucene90 stored fields
        // under many codec names, and the stored-fields reader checks formats.
        eprintln!("warning: segment {} uses codec {}; attempting to read it anyway", sc.name, sc.codec);
    }
    let info = read_segment_info(dir, &sc.name)?;
    let files = open_segment_files(dir, &sc.name, info.compound)?;
    let fields = if sc.field_infos_gen > 0 {
        let fname = format!("{}_{}.fnm", sc.name, io::base36(sc.field_infos_gen as u64));
        let b = files.open_plain(&fname)?;
        read_field_infos(b.as_slice(), &fname)?
    } else {
        let b = files.open(".fnm")?;
        read_field_infos(b.as_slice(), &format!("{}.fnm", sc.name))?
    };
    let words = (info.max_doc as usize + 63) / 64;
    let mut live: Option<Vec<u64>> = None;
    let mut hard_deleted = 0;
    if sc.del_gen > 0 {
        let fname = format!("{}_{}.liv", sc.name, io::base36(sc.del_gen as u64));
        let bytes = files.open_plain(&fname)?;
        let b = bytes.as_slice();
        check_footer(b, &fname)?;
        let mut inp = In::new(b);
        let h = read_index_header(&mut inp)?;
        expect_codec(&h, &fname, &["Lucene90LiveDocs"])?;
        let mut bits = Vec::with_capacity(words);
        for _ in 0..words {
            bits.push(inp.le_u64()?);
        }
        let live_n: u32 = bits.iter().map(|w| w.count_ones()).sum();
        hard_deleted = info.max_doc - live_n;
        if hard_deleted != sc.del_count as u32 {
            bail!("{}: {} docs deleted but segments file says {}", fname, hard_deleted, sc.del_count);
        }
        live = Some(bits);
    }
    let mut soft_deleted = 0;
    if sc.soft_del_count > 0 {
        let fi = fields
            .iter()
            .find(|f| f.name == SOFT_DELETES_FIELD)
            .ok_or_else(|| anyhow!("segment {} has soft deletes but no {} field", sc.name, SOFT_DELETES_FIELD))?;
        let docs = dv::soft_deleted_docs(&files, sc, fi, &fields, info.max_doc)?;
        let bits = live.get_or_insert_with(|| {
            let mut v = vec![u64::MAX; words];
            if info.max_doc % 64 != 0 {
                v[words - 1] = (1u64 << (info.max_doc % 64)) - 1;
            }
            v
        });
        for d in docs {
            let w = &mut bits[(d >> 6) as usize];
            let m = 1u64 << (d & 63);
            if *w & m != 0 {
                *w &= !m;
                soft_deleted += 1;
            }
        }
        // Lucene's softDelCount excludes docs that are also hard-deleted.
        if soft_deleted != sc.soft_del_count as u32 {
            bail!(
                "segment {}: decoded {} soft-deleted docs but the commit records {}; refusing to guess which docs are live",
                sc.name,
                soft_deleted,
                sc.soft_del_count
            );
        }
    }
    Ok(Segment { name: sc.name.clone(), max_doc: info.max_doc, fields, live, hard_deleted, soft_deleted, files, info })
}

/// Reads a codec header from the start of `b`, for callers outside this module.
pub fn header_of(b: &[u8]) -> Result<io::Header> {
    read_header(&mut In::new(b))
}
