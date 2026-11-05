//! On-disk dense-vector formats.
//!
//! Vectors are **not** stored in text segments (see [`crate::vector::store`]
//! for the design). Two formats live here:
//!
//! 1. The **point map** — section 5 of a text segment file. Per vector
//!    field, a dense `doc -> PointId` column of `u64` (`u64::MAX` = the doc
//!    has no vector). A text-segment merge only rewrites this column with
//!    remapped doc ids; point ids never change.
//!
//!    ```text
//!    u32 num_fields
//!    per field: u32 field, u32 num_docs, [u64 point_id] x num_docs
//!    ```
//!
//! 2. The **vector segment file** (`vec_XXXXXXXX.rvec`): an immutable set
//!    of points of one field — point ids per ordinal, a dense flat `f32`
//!    column over ordinals `0..n`, and for *sealed* segments the HNSW graph
//!    over those ordinals (+ optional int8 or int4 copy). *Appendable* chunks (one
//!    per flush) have no graph and are searched exactly.
//!
//!    ```text
//!    "RVEC" u32 version
//!    u16 field, u8 similarity, u8 flags, u32 dim, u32 num_vectors
//!    4 x (u64 off, u64 len)      // point ids, flat, hnsw, quantized (file offsets)
//!    [sections, each 8-aligned]
//!    u32 crc32(everything before), "RVEC"
//!    ```
//!
//!    Section 3 holds *the* quantized copy: an [`Int8Vectors`](crate::vector::Int8Vectors)
//!    column when [`FLAG_INT8`] is set, an [`Int4Vectors`](crate::vector::Int4Vectors)
//!    column when [`FLAG_INT4`] is set (never both). The version, header
//!    size and 4-entry section table are frozen: a reader that only knows
//!    `FLAG_GRAPH` / `FLAG_INT8` opens an int4 or flat-sealed file, ignores
//!    the bits it does not know, and searches the `f32` column (see
//!    `old_reader_opens_int4_and_flat_files`).

use super::directory::FileSlice;
use super::crc32;
use crate::error::{Result, RusticError};
use crate::types::DocId;
use crate::vector::Similarity;

pub fn sim_tag(s: Similarity) -> u8 {
    match s {
        Similarity::Cosine => 0,
        Similarity::DotProduct => 1,
        Similarity::L2 => 2,
        Similarity::MaxInnerProduct => 3,
        Similarity::BitHamming => 4,
        Similarity::ByteCosine => 5,
        Similarity::ByteDotProduct => 6,
        Similarity::ByteL2 => 7,
        Similarity::ByteMaxInnerProduct => 8,
    }
}

pub fn sim_from_tag(t: u8) -> Result<Similarity> {
    Ok(match t {
        0 => Similarity::Cosine,
        1 => Similarity::DotProduct,
        2 => Similarity::L2,
        3 => Similarity::MaxInnerProduct,
        4 => Similarity::BitHamming,
        5 => Similarity::ByteCosine,
        6 => Similarity::ByteDotProduct,
        7 => Similarity::ByteL2,
        8 => Similarity::ByteMaxInnerProduct,
        _ => return Err(RusticError::Corrupted(format!("bad similarity tag {t}"))),
    })
}

/// Parse an ES `similarity` name.
pub fn similarity_by_name(name: &str) -> Option<Similarity> {
    Some(match name {
        "cosine" => Similarity::Cosine,
        "dot_product" => Similarity::DotProduct,
        "max_inner_product" => Similarity::MaxInnerProduct,
        "l2_norm" | "l2" => Similarity::L2,
        // Packed rows (`vector::packed`): ES `bit` / `byte` fields.
        "bit_hamming" => Similarity::BitHamming,
        "byte_cosine" => Similarity::ByteCosine,
        "byte_dot_product" => Similarity::ByteDotProduct,
        "byte_l2_norm" => Similarity::ByteL2,
        "byte_max_inner_product" => Similarity::ByteMaxInnerProduct,
        _ => return None,
    })
}

/// One field's vectors while a memtable is being built (brute-force
/// searchable until flush).
pub struct VectorFieldBuilder {
    pub similarity: Similarity,
    dim: u32,
    /// `(doc, vector)` in ascending doc order.
    rows: Vec<(DocId, Vec<f32>)>,
    /// The field's vector policy when the builder was created (decides
    /// how searches score these rows, see [`crate::vector::es_quant`]).
    pub options: Option<crate::vector::store::VectorFieldOptions>,
    /// ES-scoring codes fitted on the rows: `(key, rows fitted, codes, their
    /// node-wide accounting)`. Immutable between writes; an append
    /// invalidates both fit and codes.
    es_quant: parking_lot::Mutex<Option<EsCache>>,
}

type EsCache = ((u8, u32), usize, std::sync::Arc<crate::vector::es_quant::EsCodes>, crate::accounting::Charge);

impl VectorFieldBuilder {
    pub fn new(dim: u32, similarity: Similarity) -> Self {
        Self { similarity, dim, rows: Vec::new(), options: None, es_quant: parking_lot::Mutex::new(None) }
    }

    /// With the field's vector policy.
    pub fn with_options(mut self, options: Option<crate::vector::store::VectorFieldOptions>) -> Self {
        self.options = options;
        self
    }

    /// The scoring ES uses for these rows under `cfg` (`None`: exact).
    pub fn es_quant(&self, cfg: &crate::vector::store::PlannerConfig) -> Option<crate::vector::es_quant::EsQuant> {
        use crate::vector::quantize::Quantization;
        let opts = self.options?;
        let quant = match cfg.quantization {
            Quantization::Auto => opts.quantization,
            q => q,
        };
        crate::vector::es_quant::EsQuant::of(quant, &opts, cfg.oversample.unwrap_or(opts.oversample))
    }

    /// Brute-force k-NN scored as ES scores a quantized segment holding
    /// exactly these rows (every row, deleted ones included, fits the
    /// quantizer; see [`crate::vector::es_quant`]).
    pub fn knn_es(&self, spec: &crate::vector::es_quant::EsQuant, query: &[f32], k: usize, accept: impl Fn(DocId) -> bool) -> Vec<(DocId, f32)> {
        let key = spec.key();
        let coded = {
            let mut c = self.es_quant.lock();
            match &*c {
                Some((ck, n, coded, _)) if *ck == key && *n == self.rows.len() => coded.clone(),
                _ => {
                    let q = spec.fit(self.similarity, self.rows.len(), |i| self.rows[i].1.as_slice());
                    let coded = std::sync::Arc::new(crate::vector::es_quant::EsCodes::build(
                        q, self.similarity, self.rows.len(), self.dim as usize, |i| self.rows[i].1.as_slice()));
                    // Built after the fact (a closed memtable's charge does
                    // not cover it): accounted while cached.
                    let charge = crate::accounting::Charge::new(crate::accounting::Category::ReadCaches, coded.heap_bytes() as u64);
                    *c = Some((key, self.rows.len(), coded.clone(), charge));
                    coded
                }
            }
        };
        let scorer = crate::vector::es_quant::EsQuantScorer::new(coded.q.clone(), self.similarity, query);
        let mut scored: Vec<(DocId, f32)> = self.rows.iter().enumerate()
            .filter(|(_, (d, _))| accept(*d))
            .map(|(i, (d, _))| (*d, scorer.score_row(&coded, i))).collect();
        crate::vector::hnsw::add_vector_ops(scored.len() as u64);
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    /// Extra query-cache storage, separate from the raw rows already counted
    /// by SegmentWriter during indexing.
    pub fn cached_bytes(&self) -> usize {
        self.es_quant.lock().as_ref().map_or(0, |(_, _, codes, _)| codes.heap_bytes())
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn set(&mut self, doc: DocId, v: &[f32]) {
        assert_eq!(v.len(), self.dim as usize, "vector dim mismatch");
        debug_assert!(self.rows.last().map_or(true, |(d, _)| *d < doc));
        self.rows.push((doc, v.to_vec()));
        *self.es_quant.get_mut() = None;
    }

    pub fn get(&self, doc: DocId) -> Option<&[f32]> {
        self.rows
            .binary_search_by_key(&doc, |(d, _)| *d)
            .ok()
            .map(|i| self.rows[i].1.as_slice())
    }

    /// All `(doc, vector)` rows in doc order.
    pub fn rows(&self) -> &[(DocId, Vec<f32>)] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Brute-force k-NN over the in-memory rows.
    pub fn knn(&self, query: &[f32], k: usize, accept: impl Fn(DocId) -> bool) -> Vec<(DocId, f32)> {
        let mut scored: Vec<(DocId, f32)> = self
            .rows
            .iter()
            .filter(|(d, _)| accept(*d))
            .map(|(d, v)| (*d, self.similarity.score(query, v)))
            .collect();
        crate::vector::hnsw::add_vector_ops(scored.len() as u64);
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }
}

// ---------------------------------------------------------------------------
// Point map (text segment section).

/// Sentinel for "no vector".
pub const NO_POINT: u64 = u64::MAX;

fn rd_u32(data: &[u8], pos: usize) -> Result<u32> {
    data.get(pos..pos + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| RusticError::Corrupted("vector data truncated".into()))
}

fn rd_u64(data: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap())
}

/// Encode the per-field `doc -> PointId` columns ([`NO_POINT`] = none).
pub fn encode_point_map(fields: &[(u16, Vec<u64>)]) -> Vec<u8> {
    if fields.is_empty() {
        return Vec::new();
    }
    let total: usize = fields.iter().map(|(_, c)| 8 + c.len() * 8).sum();
    let mut out = Vec::with_capacity(4 + total);
    out.extend_from_slice(&(fields.len() as u32).to_le_bytes());
    for (field, col) in fields {
        out.extend_from_slice(&(*field as u32).to_le_bytes());
        out.extend_from_slice(&(col.len() as u32).to_le_bytes());
        for p in col {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }
    out
}

/// Reader over a text segment's point map.
pub struct PointMapReader<'a> {
    data: &'a [u8],
    /// `(field, byte offset of the column, num_docs)`.
    fields: std::borrow::Cow<'a, [(u16, usize, u32)]>,
}

impl<'a> PointMapReader<'a> {
    pub fn empty() -> PointMapReader<'static> {
        PointMapReader { data: &[], fields: std::borrow::Cow::Borrowed(&[]) }
    }

    pub fn open(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Ok(Self { data, fields: std::borrow::Cow::Borrowed(&[]) });
        }
        let n = rd_u32(data, 0)? as usize;
        let mut pos = 4;
        let mut fields = Vec::with_capacity(n);
        for _ in 0..n {
            let field = rd_u32(data, pos)? as u16;
            let num_docs = rd_u32(data, pos + 4)?;
            pos += 8;
            let end = pos + num_docs as usize * 8;
            if end > data.len() {
                return Err(RusticError::Corrupted("point map column out of range".into()));
            }
            fields.push((field, pos, num_docs));
            pos = end;
        }
        Ok(Self { data, fields: std::borrow::Cow::Owned(fields) })
    }

    /// Retain the validated offsets with their immutable segment bytes.
    pub(crate) fn into_directory(self) -> Vec<(u16, usize, u32)> {
        self.fields.into_owned()
    }

    /// The directory must have been parsed from these same immutable bytes.
    pub(crate) fn from_directory(data: &'a [u8], fields: &'a [(u16, usize, u32)]) -> Self {
        Self { data, fields: std::borrow::Cow::Borrowed(fields) }
    }

    /// Count vector columns without allocating a field listing.
    pub fn field_count(&self) -> usize { self.fields.len() }

    /// Vector fields present in this segment.
    pub fn fields(&self) -> Vec<u16> {
        self.fields.iter().map(|(f, _, _)| *f).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The point id of `doc`'s vector in `field`.
    pub fn get(&self, field: u16, doc: DocId) -> Option<u64> {
        let &(_, off, n) = self.fields.iter().find(|(f, _, _)| *f == field)?;
        if doc >= n {
            return None;
        }
        let p = rd_u64(self.data, off + doc as usize * 8);
        (p != NO_POINT).then_some(p)
    }

    /// Random-access view of `field`'s `doc -> PointId` column (the field
    /// lookup done once), or `None` when the segment has no such field.
    pub fn column(&self, field: u16) -> Option<PointColumn<'a>> {
        let &(_, off, n) = self.fields.iter().find(|(f, _, _)| *f == field)?;
        Some(PointColumn { data: &self.data[off..off + n as usize * 8], num_docs: n })
    }

    /// Every `(doc, point)` of `field`, ascending doc.
    pub fn iter(&self, field: u16) -> impl Iterator<Item = (DocId, u64)> + '_ {
        let entry = self.fields.iter().find(|(f, _, _)| *f == field).copied();
        let (off, n) = entry.map(|(_, o, n)| (o, n)).unwrap_or((0, 0));
        (0..n).filter_map(move |doc| {
            let p = rd_u64(self.data, off + doc as usize * 8);
            (p != NO_POINT).then_some((doc, p))
        })
    }
}

/// One field's `doc -> PointId` column of a text segment (see
/// [`PointMapReader::column`]).
#[derive(Clone, Copy)]
pub struct PointColumn<'a> {
    data: &'a [u8],
    num_docs: u32,
}

impl PointColumn<'_> {
    /// The point id of `doc`'s vector, if it has one.
    #[inline(always)]
    pub fn get(&self, doc: DocId) -> Option<u64> {
        if doc >= self.num_docs {
            return None;
        }
        let p = rd_u64(self.data, doc as usize * 8);
        (p != NO_POINT).then_some(p)
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// `Some(base)` when every doc `d` of the column has point `base + d`
    /// (the usual layout of a segment whose docs all carry a vector: points
    /// are assigned in doc order). One sequential pass over the column.
    pub fn identity_base(&self) -> Option<u64> {
        if self.num_docs == 0 {
            return None;
        }
        let base = rd_u64(self.data, 0);
        if base == NO_POINT {
            return None;
        }
        let ok = self.data.chunks_exact(8).enumerate().all(|(d, c)| u64::from_le_bytes(c.try_into().unwrap()) == base + d as u64);
        ok.then_some(base)
    }
}

// ---------------------------------------------------------------------------
// Vector segment file.

const RVEC_MAGIC: &[u8; 4] = b"RVEC";
/// Newest vector segment file format version. v2: `bit` / `byte` fields
/// store packed rows ([`crate::vector::packed`], similarity tags 4..=8)
/// instead of one `f32` per value. Only those fields' files are written at
/// v2 ([`rvec_version_for`]): `float` fields stay v1, which older nodes
/// still read.
pub const RVEC_VERSION: u32 = 2;

/// The format version a field of similarity `s` is written at: v2 for the
/// packed `bit` / `byte` rows, v1 otherwise.
pub fn rvec_version_for(s: Similarity) -> u32 {
    if sim_tag(s) >= 4 {
        2
    } else {
        1
    }
}
const RVEC_HEADER: usize = 8 + 12 + 4 * 16;
/// Flag bit: an int8 quantized copy is present (section 3).
pub const FLAG_INT8: u8 = 1;
/// Flag bit: the segment is sealed (has an HNSW graph).
pub const FLAG_GRAPH: u8 = 2;
/// Flag bit: a packed int4 quantized copy is present (section 3; exclusive
/// with [`FLAG_INT8`]). Readers predating it ignore the bit.
pub const FLAG_INT4: u8 = 4;
/// Flag bit: the segment is sealed *without* a graph (ES `*_flat`): it is
/// immutable and brute-force searched (over its quantized copy when it has
/// one). Readers predating it treat the file as an appendable chunk.
pub const FLAG_FLAT: u8 = 8;

/// The parts of a vector segment file, for encoding.
pub struct VectorSegmentParts<'a> {
    pub field: u16,
    pub similarity: Similarity,
    pub dim: u32,
    /// Point id of each ordinal.
    pub points: &'a [u64],
    /// Dense `f32` vectors, `points.len() * dim` values, ordinal order.
    pub vectors: &'a [f32],
    /// An encoded [`crate::vector::HnswGraph`] over ordinals (sealed only).
    pub hnsw: Option<&'a [u8]>,
    /// An encoded [`crate::vector::Int8Vectors`] column, if quantized.
    pub int8: Option<&'a [u8]>,
    /// An encoded [`crate::vector::Int4Vectors`] column, if int4-quantized
    /// (ignored when `int8` is also given).
    pub int4: Option<&'a [u8]>,
    /// Sealed without a graph ([`FLAG_FLAT`]).
    pub flat: bool,
}

fn pad8(out: &mut Vec<u8>) {
    while out.len() % 8 != 0 {
        out.push(0);
    }
}

/// Encode a vector segment file.
pub fn encode_vector_segment(p: &VectorSegmentParts<'_>) -> Vec<u8> {
    assert_eq!(p.vectors.len(), p.points.len() * p.dim as usize, "vector data size");
    let mut out = Vec::with_capacity(RVEC_HEADER + p.vectors.len() * 4 + p.points.len() * 8 + p.hnsw.map_or(0, |h| h.len()) + 64);
    out.extend_from_slice(RVEC_MAGIC);
    out.extend_from_slice(&rvec_version_for(p.similarity).to_le_bytes());
    out.extend_from_slice(&p.field.to_le_bytes());
    out.push(sim_tag(p.similarity));
    let quant = p.int8.or(p.int4).unwrap_or(&[]);
    let qflag = if p.int8.is_some() {
        FLAG_INT8
    } else if p.int4.is_some() {
        FLAG_INT4
    } else {
        0
    };
    out.push(qflag | if p.hnsw.is_some() { FLAG_GRAPH } else { 0 } | if p.flat && p.hnsw.is_none() { FLAG_FLAT } else { 0 });
    out.extend_from_slice(&p.dim.to_le_bytes());
    out.extend_from_slice(&(p.points.len() as u32).to_le_bytes());
    let table_at = out.len();
    out.resize(RVEC_HEADER, 0);
    let mut table = [(0u64, 0u64); 4];
    // Point ids.
    pad8(&mut out);
    table[0] = (out.len() as u64, (p.points.len() * 8) as u64);
    for id in p.points {
        out.extend_from_slice(&id.to_le_bytes());
    }
    // Flat f32 column.
    pad8(&mut out);
    table[1] = (out.len() as u64, (p.vectors.len() * 4) as u64);
    for f in p.vectors {
        out.extend_from_slice(&f.to_le_bytes());
    }
    for (i, section) in [p.hnsw.unwrap_or(&[]), quant].iter().enumerate() {
        pad8(&mut out);
        table[2 + i] = (out.len() as u64, section.len() as u64);
        out.extend_from_slice(section);
    }
    for (i, (off, len)) in table.iter().enumerate() {
        let at = table_at + i * 16;
        out[at..at + 8].copy_from_slice(&off.to_le_bytes());
        out[at + 8..at + 16].copy_from_slice(&len.to_le_bytes());
    }
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(RVEC_MAGIC);
    out
}

/// Write the canonical format without a second full vector/graph allocation.
/// MmapDirectory streams to an atomic temporary file. Directory backends using
/// the default create_output adapter still buffer until publication.
pub fn write_vector_segment_streamed(dir: &dyn super::directory::Directory, name: &str, p: &VectorSegmentParts<'_>) -> Result<()> {
    assert_eq!(p.vectors.len(), p.points.len() * p.dim as usize);
    let quant = p.int8.or(p.int4).unwrap_or(&[]);
    let qflag = if p.int8.is_some() { FLAG_INT8 } else if p.int4.is_some() { FLAG_INT4 } else { 0 };
    let mut header = Vec::with_capacity(RVEC_HEADER);
    header.extend_from_slice(RVEC_MAGIC);
    header.extend_from_slice(&rvec_version_for(p.similarity).to_le_bytes());
    header.extend_from_slice(&p.field.to_le_bytes());
    header.push(sim_tag(p.similarity));
    header.push(qflag | if p.hnsw.is_some() { FLAG_GRAPH } else { 0 } | if p.flat && p.hnsw.is_none() { FLAG_FLAT } else { 0 });
    header.extend_from_slice(&p.dim.to_le_bytes());
    header.extend_from_slice(&(p.points.len() as u32).to_le_bytes());
    let table_at = header.len(); header.resize(RVEC_HEADER, 0);
    let lengths = [p.points.len()*8, p.vectors.len()*4, p.hnsw.map_or(0, |x|x.len()), quant.len()];
    let mut offsets = [0usize;4]; let mut end=RVEC_HEADER;
    for i in 0..4 {
        end=(end+7)&!7; offsets[i]=end;
        let at=table_at+i*16;
        header[at..at+8].copy_from_slice(&(end as u64).to_le_bytes());
        header[at+8..at+16].copy_from_slice(&(lengths[i] as u64).to_le_bytes());
        end=end.checked_add(lengths[i]).expect("vector file length overflow");
    }
    let mut out=dir.create_output(name)?; let mut crc=0xFFFF_FFFF; let mut pos=0usize;
    let mut emit=|bytes:&[u8]|->Result<()> {out.write_all(bytes)?;crc=super::crc32_update(crc,bytes);pos+=bytes.len();Ok(())};
    emit(&header)?;
    let mut buffer=Vec::with_capacity(65536);
    for i in 0..4 {
        let pad=(offsets[i]-(if i==0 {RVEC_HEADER} else {offsets[i-1]+lengths[i-1]})) as usize;
        emit(&[0u8;8][..pad])?;
        match i {
            0=>for chunk in p.points.chunks(8192){buffer.clear();for x in chunk {buffer.extend_from_slice(&x.to_le_bytes())}emit(&buffer)?},
            1=>for chunk in p.vectors.chunks(16384){buffer.clear();for x in chunk {buffer.extend_from_slice(&x.to_le_bytes())}emit(&buffer)?},
            _=>for chunk in (if i==2 {p.hnsw.unwrap_or(&[])} else {quant}).chunks(65536){emit(chunk)?},
        }
    }
    drop(emit);
    assert_eq!(pos,end);
    out.write_all(&(!crc).to_le_bytes())?;out.write_all(RVEC_MAGIC)?;out.finish()
}

/// A parsed vector segment file (sections are sub-slices of the mapping).
pub struct VectorSegmentFile {
    pub field: u16,
    pub similarity: Similarity,
    pub dim: u32,
    pub num_vectors: u32,
    pub points: Vec<u64>,
    /// Raw little-endian `f32` bytes, `num_vectors * dim * 4`.
    pub vectors: FileSlice,
    pub hnsw: Option<FileSlice>,
    pub int8: Option<FileSlice>,
    /// Packed int4 column ([`FLAG_INT4`]).
    pub int4: Option<FileSlice>,
    /// Sealed without a graph ([`FLAG_FLAT`]).
    pub flat: bool,
}

/// Parse a vector segment file. `verify_crc` recomputes the checksum.
pub fn open_vector_segment(data: FileSlice, verify_crc: bool) -> Result<VectorSegmentFile> {
    let b = data.as_slice();
    if b.len() < RVEC_HEADER + 8 || &b[..4] != RVEC_MAGIC || &b[b.len() - 4..] != RVEC_MAGIC {
        return Err(RusticError::Corrupted("not a vector segment file".into()));
    }
    let version = rd_u32(b, 4)?;
    if !super::can_read_vector_segment_version(version) {
        return Err(RusticError::Unsupported(format!(
            "vector segment version {version} (this build reads {RVEC_VERSION})"
        )));
    }
    let body_end = b.len() - 8;
    if verify_crc && crc32(&b[..body_end]) != rd_u32(b, body_end)? {
        return Err(RusticError::Corrupted("vector segment checksum mismatch".into()));
    }
    let field = u16::from_le_bytes(b[8..10].try_into().unwrap());
    let similarity = sim_from_tag(b[10])?;
    if version < rvec_version_for(similarity) {
        return Err(RusticError::Corrupted(format!("vector segment version {version} with packed rows")));
    }
    let flags = b[11];
    let dim = rd_u32(b, 12)?;
    let num_vectors = rd_u32(b, 16)?;
    let mut sec = [(0usize, 0usize); 4];
    for (i, s) in sec.iter_mut().enumerate() {
        let at = 20 + i * 16;
        let off = rd_u64(b, at) as usize;
        let len = rd_u64(b, at + 8) as usize;
        if off.checked_add(len).is_none_or(|e| e > body_end) {
            return Err(RusticError::Corrupted("vector segment section out of range".into()));
        }
        *s = (off, len);
    }
    let n = num_vectors as usize;
    if sec[0].1 != n * 8 || sec[1].1 != n * dim as usize * 4 {
        return Err(RusticError::Corrupted("vector segment size mismatch".into()));
    }
    let points = (0..n).map(|i| rd_u64(b, sec[0].0 + i * 8)).collect();
    let slice = |(off, len): (usize, usize)| data.slice(off..off + len);
    Ok(VectorSegmentFile {
        field,
        similarity,
        dim,
        num_vectors,
        points,
        vectors: slice(sec[1]),
        hnsw: (flags & FLAG_GRAPH != 0).then(|| slice(sec[2])),
        int8: (flags & FLAG_INT8 != 0).then(|| slice(sec[3])),
        int4: (flags & FLAG_INT8 == 0 && flags & FLAG_INT4 != 0).then(|| slice(sec[3])),
        flat: flags & FLAG_GRAPH == 0 && flags & FLAG_FLAT != 0,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn memtable_quantized_cache_component_probe() {
        use crate::vector::es_quant::{EsQuant, EsQuantScorer};
        use std::hint::black_box;
        let mut builder = VectorFieldBuilder::new(96, Similarity::DotProduct);
        for doc in 0..10000 {
            let row: Vec<f32> = (0..96).map(|d| ((doc * 17 + d * 31) % 101) as f32 / 505.0 - 0.1).collect();
            builder.set(doc, &row);
        }
        let spec = EsQuant { bits: 7, osq: true, confidence_interval: None };
        let query: Vec<f32> = (0..96).map(|d| d as f32 / 960.0 - 0.05).collect();
        let q = spec.fit(builder.similarity, builder.rows.len(), |i| &builder.rows[i].1);
        let mut expected = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..16 {
            let mut scorer = EsQuantScorer::new(q.clone(), builder.similarity, &query);
            expected = builder.rows.iter().map(|(d,v)| (*d,scorer.score(v))).collect();
            expected.sort_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            expected.truncate(100);
            black_box(&expected);
        }
        let raw_ns = start.elapsed().as_nanos();
        assert_eq!(builder.knn_es(&spec, &query, 100, |_| true), expected);
        let start = std::time::Instant::now();
        for _ in 0..16 {
            assert_eq!(black_box(builder.knn_es(&spec, &query, 100, |_| true)), expected);
        }
        eprintln!("MEMTABLE_QUANT_CACHE_POC rows=10000 dims=96 repetitions=16 raw_ns={} cached_ns={}", raw_ns, start.elapsed().as_nanos());
    }

    #[test]
    fn memtable_quantized_cache_matches_raw_scores_after_append_and_policy_changes() {
        use crate::vector::es_quant::{EsQuant, EsQuantScorer};
        for sim in [Similarity::L2, Similarity::DotProduct, Similarity::Cosine, Similarity::MaxInnerProduct] {
            let mut builder = VectorFieldBuilder::new(16, sim);
            for generation in 0..3 {
                for i in 0..40 {
                    let doc = generation * 40 + i;
                    let row: Vec<f32> = (0..16).map(|d| ((doc * 17 + d * 31) % 101) as f32 / 101.0 - 0.5).collect();
                    builder.set(doc, &row);
                }
                assert_eq!(builder.cached_bytes(), 0);
                for osq in [false, true] { for bits in [4, 7] {
                    let spec = EsQuant { bits, osq, confidence_interval: None };
                    let query: Vec<f32> = (0..16).map(|d| d as f32 / 32.0 - 0.3).collect();
                    let q = spec.fit(sim, builder.rows.len(), |i| &builder.rows[i].1);
                    let mut scorer = EsQuantScorer::new(q, sim, &query);
                    let mut expected: Vec<_> = builder.rows.iter().filter(|(d, _)| d % 3 != 0)
                        .map(|(d, v)| (*d, scorer.score(v))).collect();
                    expected.sort_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    expected.truncate(13);
                    for _ in 0..2 {
                        assert_eq!(builder.knn_es(&spec, &query, 13, |d| d % 3 != 0), expected);
                        // Resident int4 rows are packed: require the code and
                        // correction storage, not the retired byte-per-cell layout.
                        let code_bytes = if bits == 4 { 8 } else { 16 };
                        let correction_bytes = if osq {
                            std::mem::size_of::<crate::vector::osq::OsqCorrections>()
                        } else { std::mem::size_of::<f32>() };
                        assert!(builder.cached_bytes() >= builder.rows.len() * (code_bytes + correction_bytes));
                        if bits == 4 && !osq {
                            assert!(builder.cached_bytes() < builder.rows.len() * (16 + correction_bytes),
                                "int4 cache must retain the packed representation");
                        }
                    }
                }}
            }
        }
    }

    #[test]
    fn streamed_vector_encoder_matches_canonical_bytes_and_crc() {
        use crate::codec::directory::{Directory,RamDirectory};
        for n in [0usize,1,3,17] { for kind in 0..6 {
            let points:Vec<u64>=(0..n as u64).map(|i|i*7).collect();
            let values:Vec<f32>=(0..n*3).map(|i|f32::from_bits([0,0x80000000,0x3f800000,0x7fc01234][i%4])).collect();
            let parts=VectorSegmentParts{field:9,similarity:Similarity::Cosine,dim:3,points:&points,vectors:&values,
                hnsw:if kind==1||kind==4 {Some(&[1,2,3,4,5])}else{None},
                int8:if kind==2||kind==4 {Some(&[9,8,7])}else{None},
                int4:if kind==3 {Some(&[6,5,4,3,2,1])}else{None},flat:kind==5};
            let expected=encode_vector_segment(&parts);let dir=RamDirectory::new();
            write_vector_segment_streamed(&dir,"test.rvec",&parts).unwrap();
            let actual=dir.open_read("test.rvec").unwrap();assert_eq!(actual.as_slice(),expected);
            assert_eq!(open_vector_segment(actual,true).unwrap().num_vectors,n as u32);
        }}
    }
    #[test]
    fn streamed_vector_encoder_crosses_buffers_on_disk_for_all_layouts() {
        use crate::codec::directory::{Directory, MmapDirectory};
        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::open(tmp.path()).unwrap();
        let n = 20003;
        let points: Vec<u64> = (0..n as u64).map(|i| i * 17).collect();
        let values: Vec<f32> = (0..n * 3).map(|i| f32::from_bits(
            [0, 0x80000000, 0x3f800000, 0x7fc01234, 0x7f800000][i % 5])).collect();
        let graph = vec![0x37; 65539];
        let quant = vec![0x91; 65541];
        for similarity in [Similarity::Cosine, Similarity::DotProduct, Similarity::L2, Similarity::MaxInnerProduct] {
            for layout in 0..8 {
                let parts = VectorSegmentParts { field: 511, similarity, dim: 3,
                    points: &points, vectors: &values,
                    hnsw: (layout & 1 != 0).then_some(graph.as_slice()),
                    int8: (layout & 2 != 0).then_some(quant.as_slice()),
                    int4: (layout & 4 != 0).then_some(quant.as_slice()), flat: true };
                let expected = encode_vector_segment(&parts);
                write_vector_segment_streamed(&dir, "test.rvec", &parts).unwrap();
                let actual = dir.open_read("test.rvec").unwrap();
                assert_eq!(actual.as_slice(), expected);
                assert_eq!(open_vector_segment(actual, true).unwrap().num_vectors, n as u32);
            }
        }
        assert_eq!(dir.list().unwrap(), vec!["test.rvec"]);
    }

    #[test]
    fn interrupted_stream_keeps_previous_vector_file_and_discards_partial_bytes() {
        use crate::codec::directory::{Directory, FileSlice, MmapDirectory, OutputFile};
        #[derive(Debug)]
        struct FailingDirectory(MmapDirectory);
        struct FailingOutput<'a> { inner: Box<dyn OutputFile + 'a>, bytes: usize }
        impl OutputFile for FailingOutput<'_> {
            fn write_all(&mut self, data: &[u8]) -> Result<()> {
                self.bytes += data.len();
                if self.bytes > 70000 {
                    return Err(std::io::Error::other("injected output failure").into());
                }
                self.inner.write_all(data)
            }
            fn finish(self: Box<Self>) -> Result<()> { self.inner.finish() }
        }
        impl Directory for FailingDirectory {
            fn open_read(&self, n: &str) -> Result<FileSlice> { self.0.open_read(n) }
            fn write(&self, n: &str, d: &[u8]) -> Result<()> { self.0.write(n, d) }
            fn exists(&self, n: &str) -> bool { self.0.exists(n) }
            fn delete(&self, n: &str) -> Result<()> { self.0.delete(n) }
            fn list(&self) -> Result<Vec<String>> { self.0.list() }
            fn list_admitted(&self, temporary: bool) -> crate::error::Result<crate::codec::directory::AdmittedNames> { self.0.list_admitted(temporary) }
            fn create_output<'a>(&'a self, n: &str) -> Result<Box<dyn OutputFile + 'a>> {
                Ok(Box::new(FailingOutput { inner: self.0.create_output(n)?, bytes: 0 }))
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let dir = FailingDirectory(MmapDirectory::open(tmp.path()).unwrap());
        let points: Vec<u64> = (0..20000).collect();
        let values = vec![0.25; 20000 * 3];
        let parts = VectorSegmentParts { field: 4, similarity: Similarity::L2, dim: 3,
            points: &points, vectors: &values, hnsw: None, int8: None, int4: None, flat: true };
        let original = encode_vector_segment(&parts);
        dir.write("test.rvec", &original).unwrap();
        assert!(write_vector_segment_streamed(&dir, "test.rvec", &parts).is_err());
        assert_eq!(dir.open_read("test.rvec").unwrap().as_slice(), original);
        assert_eq!(dir.list().unwrap(), vec!["test.rvec"]);
    }

    use super::*;

    #[test]
    fn point_map_roundtrip() {
        let col_a = vec![3, NO_POINT, 4, 1 << 40];
        let col_b = vec![NO_POINT; 4];
        let bytes = encode_point_map(&[(2, col_a), (9, col_b)]);
        let r = PointMapReader::open(&bytes).unwrap();
        assert_eq!(r.fields(), vec![2, 9]);
        assert_eq!(r.get(2, 0), Some(3));
        assert_eq!(r.get(2, 1), None);
        assert_eq!(r.get(2, 3), Some(1 << 40));
        assert_eq!(r.get(2, 4), None);
        assert_eq!(r.get(9, 0), None);
        assert_eq!(r.get(5, 0), None);
        assert_eq!(r.iter(2).collect::<Vec<_>>(), vec![(0, 3), (2, 4), (3, 1 << 40)]);
        let directory = r.into_directory();
        let cached = PointMapReader::from_directory(&bytes, &directory);
        assert!(matches!(cached.fields, std::borrow::Cow::Borrowed(_)));
        assert_eq!(cached.fields(), vec![2, 9]);
        assert_eq!(cached.iter(2).collect::<Vec<_>>(), vec![(0, 3), (2, 4), (3, 1 << 40)]);
        assert_eq!(cached.column(2).unwrap().get(3), Some(1 << 40));
        assert!(PointMapReader::open(&bytes[..bytes.len() - 1]).is_err());
        assert!(PointMapReader::open(&[]).unwrap().is_empty());
    }

    #[test]
    fn vector_segment_file_roundtrip() {
        let points = [7u64, 9];
        let vectors = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = encode_vector_segment(&VectorSegmentParts {
            field: 3,
            similarity: Similarity::L2,
            dim: 2,
            points: &points,
            vectors: &vectors,
            hnsw: Some(b"GRAPH"),
            int8: None,
            int4: None,
            flat: false,
        });
        let f = open_vector_segment(FileSlice::from_vec(bytes.clone()), true).unwrap();
        assert_eq!((f.field, f.similarity, f.dim, f.num_vectors), (3, Similarity::L2, 2, 2));
        assert_eq!(f.points, points);
        assert_eq!(f.vectors.len(), 16);
        assert_eq!(f.hnsw.unwrap().as_slice(), b"GRAPH");
        assert!(f.int8.is_none());
        assert!(f.int4.is_none());
        assert!(!f.flat);
        let mut bad = bytes.clone();
        bad[100] ^= 1;
        assert!(open_vector_segment(FileSlice::from_vec(bad), true).is_err());
        assert!(open_vector_segment(FileSlice::from_vec(b"nope".to_vec()), false).is_err());
    }

    /// A copy of the reader as it was before int4 / flat sealing existed
    /// (knows only `FLAG_GRAPH` and `FLAG_INT8`): what a node that has not
    /// been upgraded runs against files written by this build.
    fn open_with_old_reader(b: &[u8]) -> Result<(u8, Vec<u64>, Vec<f32>, Option<Vec<u8>>, Option<Vec<u8>>)> {
        if b.len() < RVEC_HEADER + 8 || &b[..4] != RVEC_MAGIC || &b[b.len() - 4..] != RVEC_MAGIC {
            return Err(RusticError::Corrupted("not a vector segment file".into()));
        }
        let version = rd_u32(b, 4)?;
        if version != 1 {
            return Err(RusticError::Unsupported(format!("vector segment version {version}")));
        }
        let body_end = b.len() - 8;
        if crc32(&b[..body_end]) != rd_u32(b, body_end)? {
            return Err(RusticError::Corrupted("checksum".into()));
        }
        let flags = b[11];
        let dim = rd_u32(b, 12)? as usize;
        let n = rd_u32(b, 16)? as usize;
        let mut sec = [(0usize, 0usize); 4];
        for (i, s) in sec.iter_mut().enumerate() {
            let at = 20 + i * 16;
            *s = (rd_u64(b, at) as usize, rd_u64(b, at + 8) as usize);
        }
        if sec[0].1 != n * 8 || sec[1].1 != n * dim * 4 {
            return Err(RusticError::Corrupted("size".into()));
        }
        let points = (0..n).map(|i| rd_u64(b, sec[0].0 + i * 8)).collect();
        let floats = b[sec[1].0..sec[1].0 + sec[1].1].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let hnsw = (flags & 2 != 0).then(|| b[sec[2].0..sec[2].0 + sec[2].1].to_vec());
        let int8 = (flags & 1 != 0).then(|| b[sec[3].0..sec[3].0 + sec[3].1].to_vec());
        Ok((flags, points, floats, hnsw, int8))
    }

    #[test]
    fn old_reader_opens_int4_and_flat_files() {
        let points = [1u64, 2, 3];
        let vectors = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let int4_blob = b"INT4-COLUMN-BYTES".to_vec();
        // int4 + graph.
        let bytes = encode_vector_segment(&VectorSegmentParts {
            field: 1,
            similarity: Similarity::Cosine,
            dim: 2,
            points: &points,
            vectors: &vectors,
            hnsw: Some(b"G"),
            int8: None,
            int4: Some(&int4_blob),
            flat: false,
        });
        let (flags, pts, floats, hnsw, int8) = open_with_old_reader(&bytes).unwrap();
        assert_eq!(flags, FLAG_GRAPH | FLAG_INT4);
        assert_eq!(pts, points);
        assert_eq!(floats, vectors);
        assert_eq!(hnsw.unwrap(), b"G");
        assert!(int8.is_none(), "an old node sees no int8 copy and searches f32");
        let f = open_vector_segment(FileSlice::from_vec(bytes), true).unwrap();
        assert_eq!(f.int4.unwrap().as_slice(), &int4_blob[..]);
        assert!(f.int8.is_none() && !f.flat);
        // int4 flat-sealed (no graph).
        let bytes = encode_vector_segment(&VectorSegmentParts {
            field: 1,
            similarity: Similarity::L2,
            dim: 2,
            points: &points,
            vectors: &vectors,
            hnsw: None,
            int8: None,
            int4: Some(&int4_blob),
            flat: true,
        });
        let (flags, _, floats, hnsw, int8) = open_with_old_reader(&bytes).unwrap();
        assert_eq!(flags, FLAG_INT4 | FLAG_FLAT);
        assert_eq!(floats, vectors);
        assert!(hnsw.is_none() && int8.is_none(), "an old node treats it as an appendable chunk");
        let f = open_vector_segment(FileSlice::from_vec(bytes), true).unwrap();
        assert!(f.flat && f.hnsw.is_none() && f.int4.is_some());
        // int8 files are byte-identical to the pre-int4 encoding: only the
        // two old flag bits are set and section 3 is the int8 column.
        let int8_blob = b"INT8-COLUMN".to_vec();
        let bytes = encode_vector_segment(&VectorSegmentParts {
            field: 1,
            similarity: Similarity::L2,
            dim: 2,
            points: &points,
            vectors: &vectors,
            hnsw: Some(b"G"),
            int8: Some(&int8_blob),
            int4: Some(&int4_blob),
            flat: true,
        });
        let (flags, _, _, _, int8) = open_with_old_reader(&bytes).unwrap();
        assert_eq!(flags, FLAG_GRAPH | FLAG_INT8);
        assert_eq!(int8.unwrap(), int8_blob);
        let f = open_vector_segment(FileSlice::from_vec(bytes), true).unwrap();
        assert!(f.int4.is_none() && !f.flat && f.int8.is_some());
        // Files from the old encoder (no new bits) still open.
        let f = open_vector_segment(FileSlice::from_vec(encode_vector_segment(&VectorSegmentParts {
            field: 1,
            similarity: Similarity::L2,
            dim: 2,
            points: &points,
            vectors: &vectors,
            hnsw: None,
            int8: None,
            int4: None,
            flat: false,
        })), true)
        .unwrap();
        assert!(f.hnsw.is_none() && f.int8.is_none() && f.int4.is_none() && !f.flat);
    }

    #[test]
    fn memtable_brute_force() {
        let mut b = VectorFieldBuilder::new(2, Similarity::L2);
        b.set(0, &[0.0, 0.0]);
        b.set(2, &[1.0, 1.0]);
        b.set(5, &[3.0, 3.0]);
        assert_eq!(b.get(2), Some(&[1.0, 1.0][..]));
        assert!(b.get(1).is_none());
        let r = b.knn(&[0.9, 0.9], 2, |_| true);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 0]);
        assert_eq!(b.knn(&[0.9, 0.9], 2, |d| d != 2)[0].0, 0);
    }
}
