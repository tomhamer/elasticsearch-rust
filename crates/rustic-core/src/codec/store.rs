//! Stored fields: LZ4-compressed chunks of serialized documents.
//!
//! ```text
//! chunk 0 | chunk 1 | ... | table: vint num_docs, vint num_chunks,
//!                            per chunk: vint off, vint len, vint docs
//!         | u64 table_offset
//! chunk = lz4(compress_prepend_size)( per doc: vint len, bincode(Document) )
//! ```
//!
//! From v18 a chunk whose size prefix has [`COMPACT_CHUNK`] set holds
//! [`encode_doc_compact`] docs instead of bincode ones. From v26 a chunk
//! whose size prefix has [`ZSTD_CHUNK`] set is zstd-compressed instead of
//! LZ4 (`index.codec: best_compression`, see [`StoredCompression`]).
//!
//! This mirrors Lucene's `Lucene90CompressingStoredFieldsFormat`: fetching
//! one doc decompresses one chunk, so random access is O(chunk) and bulk
//! sequential reads amortize to near-zero overhead. As in Lucene, a chunk is
//! bounded by bytes as well as docs: it is closed once it holds
//! [`CHUNK_DOCS`] docs or [`CHUNK_BYTES`] of serialized docs, whichever
//! comes first (see [`chunk_full`]). With a doc bound alone, fetching one
//! large doc (a dense vector in `_source` is ~20 KB of JSON) decompressed
//! 64 such docs, over a megabyte, per hit.
//!
//! When every chunk but the last holds [`CHUNK_DOCS`] docs (small docs), a
//! doc's chunk is `doc / CHUNK_DOCS`. Otherwise (byte-bounded chunks, or a
//! merge that copied its inputs' chunks verbatim, leaving short chunks at
//! the input boundaries and where deleted docs were dropped) the store is
//! "irregular" and a doc's chunk is found by binary search over per-chunk
//! doc starts.

use super::varint::{read_bytes, read_u64_le, read_vint, write_vint};
use crate::error::{Result, RusticError};
use crate::types::{DocId, Document};
use std::borrow::Cow;

/// Most docs in one chunk (Lucene `BEST_SPEED`: 1024). Small docs fill a
/// chunk to [`CHUNK_BYTES`] long before this.
pub const CHUNK_DOCS: usize = 1024;

/// Top bit of a chunk's LZ4 size prefix: its docs use the compact encoding
/// ([`encode_doc_compact`], format [`crate::codec::COMPACT_STORED_SINCE`]+)
/// instead of bincode. The flag travels with the chunk, so merges copy
/// chunks of either encoding verbatim.
pub const COMPACT_CHUNK: u8 = 0x80;

/// Second-highest bit of a chunk's size prefix (format
/// [`crate::codec::ZSTD_STORED_SINCE`]+): the chunk body is one zstd frame
/// instead of an LZ4 block. Like [`COMPACT_CHUNK`] the flag travels with
/// the chunk, so merges copy chunks of either compression verbatim. Chunks
/// of 1 GiB or more (never written: a chunk closes at its byte bound) stay
/// LZ4, so the two flags never collide with a size bit.
pub const ZSTD_CHUNK: u8 = 0x40;

/// Third-highest bit of a chunk's size prefix (format
/// [`crate::codec::UID_STORED_SINCE`]+): its compact docs may hold
/// [`crate::uid`]-encoded ids (value kind `COMPACT_UID`). A merge writing
/// below that format re-encodes such a chunk instead of copying it. Chunks
/// of 512 MiB or more are never written, so the flag never collides with a
/// size bit.
pub const UID_CHUNK: u8 = 0x20;

/// zstd level of [`StoredCompression::Best`] chunks (ES
/// `Zstd814StoredFieldsFormat.Mode.BEST_COMPRESSION`: level 3).
pub const BEST_ZSTD_LEVEL: i32 = 3;
/// Most docs in one [`StoredCompression::Best`] chunk (ES
/// `BEST_COMPRESSION`: 2048 docs).
pub const BEST_CHUNK_DOCS: usize = 2048;
/// A [`StoredCompression::Best`] chunk closes once its serialized docs
/// reach this many bytes (ES `BEST_COMPRESSION`: 240 KB zstd blocks) and
/// hold at least [`BEST_CHUNK_MIN_DOCS`] docs.
pub const BEST_CHUNK_BYTES: usize = 240 << 10;
/// Fewest docs a [`StoredCompression::Best`] chunk closes with before
/// [`BEST_CHUNK_MAX_BYTES`]: zstd finds the values documents of a stream
/// repeat a few dozen docs apart. Large docs (k8s_metrics' ~10 KB) get
/// chunks of up to 512 KB (`_source` 355 -> 263 B/doc), whose one-doc
/// fetch (~76 us) stays under that of a 240 KB chunk of ~1 KB docs
/// (big5, ~94 us); chunks of small docs are ES's, unchanged
/// (`best_chunk_fetch_microbench`).
pub const BEST_CHUNK_MIN_DOCS: usize = 48;
/// Largest [`StoredCompression::Best`] chunk, whatever its doc count.
pub const BEST_CHUNK_MAX_BYTES: usize = 512 << 10;

/// How a segment's stored-field chunks are framed and compressed: ES
/// `index.codec` (`default`: small LZ4 chunks for fast fetches;
/// `best_compression`: large zstd chunks, ES's `BEST_COMPRESSION` mode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StoredCompression {
    /// [`CHUNK_DOCS`] / [`CHUNK_BYTES`] chunks, LZ4.
    #[default]
    Fast,
    /// [`BEST_CHUNK_DOCS`] / [`BEST_CHUNK_BYTES`] (at least
    /// [`BEST_CHUNK_MIN_DOCS`], at most [`BEST_CHUNK_MAX_BYTES`]) chunks, zstd at
    /// [`BEST_ZSTD_LEVEL`] (from format [`crate::codec::ZSTD_STORED_SINCE`];
    /// below it, LZ4 chunks of the same size).
    Best,
}

impl StoredCompression {
    /// `index.codec` (`best_compression`; anything else is the default).
    pub fn from_codec(codec: Option<&str>) -> Self {
        match codec {
            Some(c) if c.eq_ignore_ascii_case("best_compression") => StoredCompression::Best,
            _ => StoredCompression::Fast,
        }
    }

    /// Whether a chunk holding `docs` docs in `raw_len` serialized bytes is
    /// full ([`chunk_full`] for [`StoredCompression::Fast`]).
    #[inline]
    pub fn chunk_full(self, docs: usize, raw_len: usize) -> bool {
        self.chunk_full_sized(docs, docs, raw_len)
    }

    /// [`Self::chunk_full`] with the count of docs over [`SIZED_DOC_BYTES`]
    /// ([`chunk_full_sized`]).
    #[inline]
    pub fn chunk_full_sized(self, docs: usize, sized: usize, raw_len: usize) -> bool {
        self.chunk_full_for(docs, sized, raw_len, true)
    }

    /// [`Self::chunk_full_sized`] of a segment whose fetches want small
    /// chunks of large docs (`small_chunks`: it holds vectors, see
    /// [`chunk_full_for`]).
    #[inline]
    pub fn chunk_full_for(self, docs: usize, sized: usize, raw_len: usize, small_chunks: bool) -> bool {
        match self {
            StoredCompression::Fast => chunk_full_for(docs, sized, raw_len, small_chunks),
            StoredCompression::Best => docs >= BEST_CHUNK_DOCS || raw_len >= BEST_CHUNK_MAX_BYTES || (raw_len >= BEST_CHUNK_BYTES && docs >= BEST_CHUNK_MIN_DOCS),
        }
    }

    /// Whether chunks written at segment format `version` are zstd.
    #[inline]
    pub fn zstd_at(self, version: u32) -> bool {
        self == StoredCompression::Best && version >= crate::codec::ZSTD_STORED_SINCE
    }
}

/// Whether a compressed chunk's docs are compact-encoded.
#[inline]
pub fn chunk_is_compact(chunk: &[u8]) -> bool {
    chunk.get(3).is_some_and(|b| b & COMPACT_CHUNK != 0)
}

/// Whether a compressed chunk is zstd ([`ZSTD_CHUNK`]).
#[inline]
pub fn chunk_is_zstd(chunk: &[u8]) -> bool {
    chunk.get(3).is_some_and(|b| b & ZSTD_CHUNK != 0)
}

/// Whether a compressed chunk may hold uid-encoded ids ([`UID_CHUNK`]).
#[inline]
pub fn chunk_is_uid(chunk: &[u8]) -> bool {
    chunk.get(3).is_some_and(|b| b & UID_CHUNK != 0)
}

/// The decompressed size of a compressed chunk (flags masked).
#[inline]
fn chunk_size(chunk: &[u8]) -> Option<usize> {
    let b = chunk.get(..4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3] & !(COMPACT_CHUNK | ZSTD_CHUNK | UID_CHUNK)]) as usize)
}

/// LZ4-compress `raw` with its size prefix, flagged compact when `compact`.
pub fn compress_chunk(raw: &[u8], compact: bool) -> Vec<u8> {
    let mut c = lz4_flex::compress_prepend_size(raw);
    if compact {
        c[3] |= COMPACT_CHUNK;
    }
    c
}

/// [`compress_chunk`], zstd-compressed at [`BEST_ZSTD_LEVEL`] and flagged
/// [`ZSTD_CHUNK`] when `zstd` (callers pass it only from format
/// [`crate::codec::ZSTD_STORED_SINCE`]). The size prefix is the same
/// little-endian u32 of the decompressed size.
pub fn compress_chunk_with(raw: &[u8], compact: bool, zstd: bool) -> Vec<u8> {
    compress_chunk_flagged(raw, compact, zstd, false)
}

/// [`compress_chunk_with`], flagged [`UID_CHUNK`] when `uid` (and
/// `compact`; callers pass it only from format
/// [`crate::codec::UID_STORED_SINCE`]).
pub fn compress_chunk_flagged(raw: &[u8], compact: bool, zstd: bool, uid: bool) -> Vec<u8> {
    let mut c = compress_chunk_body(raw, compact, zstd);
    if uid && compact && raw.len() < 1 << 29 {
        c[3] |= UID_CHUNK;
    }
    c
}

fn compress_chunk_body(raw: &[u8], compact: bool, zstd: bool) -> Vec<u8> {
    if !zstd || raw.len() >= 1 << 30 {
        return compress_chunk(raw, compact);
    }
    let body = match zstd::bulk::compress(raw, BEST_ZSTD_LEVEL) {
        Ok(b) => b,
        // Compressing a byte slice into a growable buffer cannot fail but
        // for allocation; LZ4 keeps the chunk readable either way.
        Err(_) => return compress_chunk(raw, compact),
    };
    let mut c = Vec::with_capacity(body.len() + 4);
    c.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    c.extend_from_slice(&body);
    c[3] |= ZSTD_CHUNK;
    if compact {
        c[3] |= COMPACT_CHUNK;
    }
    c
}

/// Decompress one stored chunk (any flags).
pub fn decompress_chunk_bytes(chunk: &[u8]) -> Result<Vec<u8>> {
    let size = chunk_size(chunk).ok_or_else(|| RusticError::Corrupted("lz4: missing size".into()))?;
    if chunk_is_zstd(chunk) {
        let out = zstd::bulk::decompress(&chunk[4..], size).map_err(|e| RusticError::Corrupted(format!("zstd: {e}")))?;
        if out.len() != size {
            return Err(RusticError::Corrupted("zstd: chunk size mismatch".into()));
        }
        return Ok(out);
    }
    lz4_flex::decompress(&chunk[4..], size).map_err(|e| RusticError::Corrupted(format!("lz4: {e}")))
}

const COMPACT_TEXT: u8 = 0;
const COMPACT_KEYWORD: u8 = 1;
const COMPACT_I64: u8 = 2;
const COMPACT_U64: u8 = 3;
const COMPACT_F64: u8 = 4;
const COMPACT_BOOL: u8 = 5;
const COMPACT_DATE: u8 = 6;
const COMPACT_GEO: u8 = 7;
const COMPACT_VECTOR: u8 = 8;
/// An id in its [`crate::uid`] binary form, read back as a keyword
/// (format [`crate::codec::UID_STORED_SINCE`]+, [`UID_CHUNK`] chunks).
const COMPACT_UID: u8 = 9;

/// Compact stored-doc encoding: `vint fields`, per field `vint id, u8
/// kind`, then the value: text and keyword `vint len, bytes`; i64 and date
/// zigzag vints; u64 a vint; f64 8 LE bytes; bool one byte; geo point two
/// f64; vector `vint n, n × f32 LE`. bincode spent 8 bytes on every
/// length, 4 on every variant tag and 8 on every integer.
pub fn encode_doc_compact<'f>(fields: impl ExactSizeIterator<Item = &'f (u16, crate::types::FieldValue)>, out: &mut Vec<u8>) {
    encode_doc_compact_uid(fields, None, out);
}

/// [`encode_doc_compact`], the keyword values of field `uid` (the index's
/// `_id`) in their [`crate::uid`] binary form (kind `COMPACT_UID`: 5 bytes
/// fewer an auto id, and bytes that compress). Only for
/// [`crate::codec::UID_STORED_SINCE`]+ chunks.
pub fn encode_doc_compact_uid<'f>(fields: impl ExactSizeIterator<Item = &'f (u16, crate::types::FieldValue)>, uid: Option<u16>, out: &mut Vec<u8>) {
    use crate::types::FieldValue as V;
    write_vint(fields.len() as u64, out);
    for (id, v) in fields {
        write_vint(*id as u64, out);
        match v {
            V::Keyword(t) if Some(*id) == uid => {
                out.push(COMPACT_UID);
                let b = crate::uid::encode(t);
                write_vint(b.len() as u64, out);
                out.extend_from_slice(&b);
            }
            V::Text(t) | V::Keyword(t) => {
                out.push(if matches!(v, V::Text(_)) { COMPACT_TEXT } else { COMPACT_KEYWORD });
                write_vint(t.len() as u64, out);
                out.extend_from_slice(t.as_bytes());
            }
            V::I64(x) | V::Date(x) => {
                out.push(if matches!(v, V::I64(_)) { COMPACT_I64 } else { COMPACT_DATE });
                write_vint(((x << 1) ^ (x >> 63)) as u64, out);
            }
            V::U64(x) => { out.push(COMPACT_U64); write_vint(*x, out); }
            V::F64(x) => { out.push(COMPACT_F64); out.extend_from_slice(&x.to_le_bytes()); }
            V::Bool(b) => { out.push(COMPACT_BOOL); out.push(*b as u8); }
            V::GeoPoint(a, b) => { out.push(COMPACT_GEO); out.extend_from_slice(&a.to_le_bytes()); out.extend_from_slice(&b.to_le_bytes()); }
            V::Vector(xs) => {
                out.push(COMPACT_VECTOR);
                write_vint(xs.len() as u64, out);
                for x in xs { out.extend_from_slice(&x.to_le_bytes()); }
            }
        }
    }
}

/// A compact doc (`bytes`, unframed) without its binary id field (kind
/// `COMPACT_UID`, [`encode_doc_compact_uid`]): `(field, id bytes)` and the
/// doc re-framed (`vint len, doc`) into `out`, the other fields' bytes as
/// they were. A doc without one is framed unchanged (`None`). For merges
/// moving ids into an id column ([`crate::codec::uid_column`]).
pub fn split_uid(bytes: &[u8], out: &mut Vec<u8>) -> Result<Option<(u16, Vec<u8>)>> {
    let bad = || RusticError::Corrupted("stored doc: compact encoding".into());
    let mut pos = 0usize;
    let n = read_vint(bytes, &mut pos)? as usize;
    let fields_at = pos;
    let mut found: Option<(u16, usize, usize, Vec<u8>)> = None;
    for (id, len) in compact_field_spans(bytes)? {
        let start = pos;
        pos += len as usize;
        if found.is_none() {
            let mut p = start;
            read_vint(bytes, &mut p)?;
            if bytes.get(p) == Some(&COMPACT_UID) {
                p += 1;
                let l = read_vint(bytes, &mut p)? as usize;
                let uid = bytes.get(p..p + l).ok_or_else(bad)?.to_vec();
                found = Some((id, start, pos, uid));
            }
        }
    }
    let Some((field, a, b, uid)) = found else {
        write_vint(bytes.len() as u64, out);
        out.extend_from_slice(bytes);
        return Ok(None);
    };
    let mut head = Vec::with_capacity(5);
    write_vint((n - 1) as u64, &mut head);
    let rest = (a - fields_at) + (bytes.len() - b);
    write_vint((head.len() + rest) as u64, out);
    out.extend_from_slice(&head);
    out.extend_from_slice(&bytes[fields_at..a]);
    out.extend_from_slice(&bytes[b..]);
    Ok(Some((field, uid)))
}

/// A compact doc (`bytes`, unframed) with binary id `uid` of `field` put
/// back as its last field, framed (`vint len, doc`) into `out`: the
/// inverse of [`split_uid`] up to field order (a merge writing below
/// [`crate::codec::UID_COLUMN_SINCE`] from an input with an id column).
pub fn join_uid(bytes: &[u8], field: u16, uid: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut pos = 0usize;
    let n = read_vint(bytes, &mut pos)? as usize;
    let mut doc = Vec::with_capacity(bytes.len() + uid.len() + 8);
    write_vint((n + 1) as u64, &mut doc);
    doc.extend_from_slice(&bytes[pos..]);
    write_vint(field as u64, &mut doc);
    doc.push(COMPACT_UID);
    write_vint(uid.len() as u64, &mut doc);
    doc.extend_from_slice(uid);
    write_vint(doc.len() as u64, out);
    out.extend_from_slice(&doc);
    Ok(())
}

/// The encoded bytes of each field value of a compact doc (`(field,
/// bytes)` in order; the leading field count is not attributed): exact,
/// binary ids included, without decoding the values.
pub fn compact_field_spans(bytes: &[u8]) -> Result<Vec<(u16, u64)>> {
    let bad = || RusticError::Corrupted("stored doc: compact encoding".into());
    let mut pos = 0usize;
    let n = read_vint(bytes, &mut pos)? as usize;
    let mut out = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        let start = pos;
        let id = u16::try_from(read_vint(bytes, &mut pos)?).map_err(|_| bad())?;
        let kind = *bytes.get(pos).ok_or_else(bad)?;
        pos += 1;
        match kind {
            COMPACT_TEXT | COMPACT_KEYWORD | COMPACT_UID => {
                let len = read_vint(bytes, &mut pos)? as usize;
                pos = pos.checked_add(len).ok_or_else(bad)?;
            }
            COMPACT_I64 | COMPACT_DATE | COMPACT_U64 => {
                read_vint(bytes, &mut pos)?;
            }
            COMPACT_F64 => pos += 8,
            COMPACT_BOOL => pos += 1,
            COMPACT_GEO => pos += 16,
            COMPACT_VECTOR => {
                let len = read_vint(bytes, &mut pos)? as usize;
                pos = pos.checked_add(len.checked_mul(4).ok_or_else(bad)?).ok_or_else(bad)?;
            }
            _ => return Err(bad()),
        }
        if pos > bytes.len() {
            return Err(bad());
        }
        out.push((id, (pos - start) as u64));
    }
    Ok(out)
}

/// Inverse of [`encode_doc_compact`].
pub fn decode_doc_compact(bytes: &[u8]) -> Result<Document> {
    use crate::types::FieldValue as V;
    let bad = || RusticError::Corrupted("stored doc: compact encoding".into());
    let mut pos = 0usize;
    let n = read_vint(bytes, &mut pos)? as usize;
    // Every field takes at least 3 bytes: a count past that is corrupt.
    if n > bytes.len() / 2 + 1 {
        return Err(bad());
    }
    let mut doc = Document::new();
    doc.fields.reserve_exact(n);
    let f64_at = |pos: &mut usize| -> Result<f64> { Ok(f64::from_le_bytes(read_bytes(bytes, pos, 8)?.try_into().unwrap())) };
    for _ in 0..n {
        let id = u16::try_from(read_vint(bytes, &mut pos)?).map_err(|_| bad())?;
        let kind = *read_bytes(bytes, &mut pos, 1)?.first().ok_or_else(bad)?;
        let v = match kind {
            COMPACT_TEXT | COMPACT_KEYWORD => {
                let len = read_vint(bytes, &mut pos)? as usize;
                let t = std::str::from_utf8(read_bytes(bytes, &mut pos, len)?).map_err(|_| bad())?.to_string();
                if kind == COMPACT_TEXT { V::Text(t) } else { V::Keyword(t) }
            }
            COMPACT_UID => {
                let len = read_vint(bytes, &mut pos)? as usize;
                V::Keyword(crate::uid::decode(read_bytes(bytes, &mut pos, len)?).ok_or_else(bad)?)
            }
            COMPACT_I64 | COMPACT_DATE => {
                let z = read_vint(bytes, &mut pos)?;
                let x = ((z >> 1) as i64) ^ -((z & 1) as i64);
                if kind == COMPACT_I64 { V::I64(x) } else { V::Date(x) }
            }
            COMPACT_U64 => V::U64(read_vint(bytes, &mut pos)?),
            COMPACT_F64 => V::F64(f64_at(&mut pos)?),
            COMPACT_BOOL => V::Bool(read_bytes(bytes, &mut pos, 1)?[0] != 0),
            COMPACT_GEO => { let a = f64_at(&mut pos)?; V::GeoPoint(a, f64_at(&mut pos)?) }
            COMPACT_VECTOR => {
                let len = read_vint(bytes, &mut pos)? as usize;
                let raw = read_bytes(bytes, &mut pos, len.checked_mul(4).ok_or_else(bad)?)?;
                V::Vector(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            }
            _ => return Err(bad()),
        };
        doc.fields.push((id, v));
    }
    if pos != bytes.len() {
        return Err(bad());
    }
    Ok(doc)
}

/// Append compact doc `bytes` to a plain chunk: `vint len, bincode(doc)`
/// (a merge writing below [`crate::codec::COMPACT_STORED_SINCE`], or one
/// mixing docs of several input chunks into a plain chunk).
pub fn append_compact_as_plain(bytes: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let doc = decode_doc_compact(bytes)?;
    let len = bincode::serialized_size(&doc).map_err(|e| RusticError::Corrupted(format!("stored doc: {e}")))?;
    write_vint(len, out);
    bincode::serialize_into(&mut *out, &doc).map_err(|e| RusticError::Corrupted(format!("stored doc: {e}")))
}

/// A decompressed compact chunk (`vint len, doc` each) with its binary ids
/// ([`encode_doc_compact_uid`]) written back as keywords: for a file
/// written below [`crate::codec::UID_STORED_SINCE`].
pub fn compact_chunk_without_uid(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 4);
    let mut pos = 0usize;
    let mut buf = Vec::new();
    while pos < raw.len() {
        let len = read_vint(raw, &mut pos)? as usize;
        let doc = decode_doc_compact(read_bytes(raw, &mut pos, len)?)?;
        buf.clear();
        encode_doc_compact(doc.fields.iter(), &mut buf);
        write_vint(buf.len() as u64, &mut out);
        out.extend_from_slice(&buf);
    }
    Ok(out)
}

/// Decode one stored doc of a chunk of either encoding.
pub fn decode_doc(bytes: &[u8], compact: bool) -> Result<Document> {
    if compact {
        decode_doc_compact(bytes)
    } else {
        bincode::deserialize(bytes).map_err(|e| RusticError::Corrupted(format!("stored doc: {e}")))
    }
}

/// A chunk is closed once its serialized docs reach this many bytes. A
/// fetch decompresses its chunk only through its own doc
/// ([`StoreReader::decompress_through`]), so this bounds the per-hit cost
/// (half a chunk on average, ~10 us at LZ4 speed). LZ4 finds the values a
/// stream of documents repeats within its window: 16 KB chunks (64 docs)
/// left http_logs' stored fields at 49 bytes a doc, 64 KB chunks 42, about
/// Lucene `BEST_SPEED`'s ratio (80 KB chunks in 8 KB sub-blocks behind a
/// preset dictionary measured 46: the dictionary is too small a window).
pub const CHUNK_BYTES: usize = 64 << 10;

/// A chunk of larger docs ([`LARGE_DOC_BYTES`] on average) closes at this
/// many bytes: the window gains them little (cohere's 440-byte documents:
/// 1% of the store from 16 KB to 64 KB chunks) while a fetch decodes half
/// a chunk per hit (a 100-hit kNN page: 3.1 ms with 16 KB chunks, 5.1 ms
/// with 64 KB; Rally nested's inner hits read their parents at c32m's
/// speed with 8 KB). Small docs fill [`CHUNK_BYTES`] (http_logs: 137 ->
/// 128 MB).
pub const LARGE_DOC_CHUNK_BYTES: usize = 8 << 10;
/// The average serialized size above which a chunk's docs are large.
pub const LARGE_DOC_BYTES: usize = 256;
/// Docs of at most this many serialized bytes (a nested child's empty
/// stored doc) are left out of the average: a nested segment's chunks are
/// as costly to fetch from as their parents' size makes them.
pub const SIZED_DOC_BYTES: usize = 16;

/// Whether a chunk holding `docs` docs in `raw_len` serialized bytes is
/// full, i.e. the next doc starts a new chunk.
#[inline]
pub fn chunk_full(docs: usize, raw_len: usize) -> bool {
    chunk_full_sized(docs, docs, raw_len)
}

/// [`chunk_full`] of a chunk `sized` of whose `docs` docs are over
/// [`SIZED_DOC_BYTES`]: the large-doc average is over those.
#[inline]
pub fn chunk_full_sized(docs: usize, sized: usize, raw_len: usize) -> bool {
    chunk_full_for(docs, sized, raw_len, true)
}

/// [`chunk_full_sized`] of a segment that does (`small_chunks`) or does not
/// hold vectors. Large-doc chunks close at [`LARGE_DOC_CHUNK_BYTES`] only
/// where a fetch reads them hit by hit: a vector index's kNN pages (cohere)
/// and a nested segment's parents behind empty child docs (inner hits).
/// Every other segment fills [`CHUNK_BYTES`]: its few-hundred-byte docs
/// (noaa ~290 bytes, geonames ~370) compress across the whole window,
/// 8 KB chunks left noaa's stored fields 20% larger (store +11%).
#[inline]
pub fn chunk_full_for(docs: usize, sized: usize, raw_len: usize, small_chunks: bool) -> bool {
    docs >= CHUNK_DOCS || raw_len >= CHUNK_BYTES
        || ((small_chunks || sized < docs) && raw_len >= LARGE_DOC_CHUNK_BYTES && raw_len > sized * LARGE_DOC_BYTES)
}

/// Start of each complete framed doc of decompressed chunk (prefix) `raw`.
pub fn chunk_doc_offsets(raw: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        let at = pos;
        let Ok(len) = read_vint(raw, &mut pos) else { break };
        let Some(end) = pos.checked_add(len as usize).filter(|&e| e <= raw.len()) else { break };
        out.push(at as u32);
        pos = end;
    }
    out
}

/// The framed doc starting at `at` (an entry of [`chunk_doc_offsets`]).
pub fn doc_bytes_at(raw: &[u8], at: usize) -> Result<&[u8]> {
    let mut pos = at;
    let len = read_vint(raw, &mut pos)? as usize;
    read_bytes(raw, &mut pos, len)
}

/// Serialize all stored documents of a segment.
pub fn encode_store(docs: &[Document]) -> Vec<u8> {
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    for doc in docs {
        let open = chunks.last().zip(counts.last()).is_some_and(|(c, &n)| !chunk_full(n as usize, c.len()));
        if !open {
            chunks.push(Vec::new());
            counts.push(0);
        }
        let raw = chunks.last_mut().expect("chunk opened above");
        let bytes = bincode::serialize(doc).expect("Document serializes");
        write_vint(bytes.len() as u64, raw);
        raw.extend_from_slice(&bytes);
        *counts.last_mut().expect("chunk opened above") += 1;
    }
    encode_store_chunks(&chunks, &counts)
}

/// Append the chunk table (`(off, len, docs)` per chunk) and its offset.
fn write_table(chunks: &[(usize, usize, usize)], num_docs: usize, out: &mut Vec<u8>) {
    let table_offset = out.len() as u64;
    write_vint(num_docs as u64, out);
    write_vint(chunks.len() as u64, out);
    for (off, len, docs) in chunks {
        write_vint(*off as u64, out);
        write_vint(*len as u64, out);
        write_vint(*docs as u64, out);
    }
    out.extend_from_slice(&table_offset.to_le_bytes());
}

/// The table of a store whose compressed chunks (`(len, docs)` each, laid
/// out back to back from offset 0) are written separately: appending these
/// bytes after the chunks yields a valid store section.
pub fn encode_store_table(chunks: &[(usize, usize)]) -> Vec<u8> {
    let mut off = 0usize;
    let mut table = Vec::with_capacity(chunks.len());
    let mut num_docs = 0usize;
    for &(len, docs) in chunks {
        table.push((off, len, docs));
        off += len;
        num_docs += docs;
    }
    let mut out = Vec::with_capacity(chunks.len() * 8 + 24);
    // `write_table` records `out.len()` as the table offset; account for
    // the chunk bytes that precede it.
    write_table(&table, num_docs, &mut out);
    let n = out.len();
    out[n - 8..].copy_from_slice(&(off as u64).to_le_bytes());
    out
}

/// [`encode_store`] from pre-framed chunks: `chunks[c]` is chunk `c`'s
/// uncompressed payload (per doc `vint len, bincode(Document)`) holding
/// `docs[c]` docs, framed by [`chunk_full`]. Byte-identical output; the
/// chunks are LZ4-compressed in parallel.
pub fn encode_store_chunks(chunks: &[Vec<u8>], docs: &[u32]) -> Vec<u8> {
    encode_store_chunks_flagged(chunks, docs, &[])
}

/// [`encode_store_chunks`] with chunk `c` compact-encoded when
/// `compact[c]` (missing entries: bincode).
pub fn encode_store_chunks_flagged(chunks: &[Vec<u8>], docs: &[u32], compact: &[bool]) -> Vec<u8> {
    encode_store_chunks_compressed(chunks, docs, compact, false)
}

/// [`encode_store_chunks_flagged`], every chunk zstd-compressed when `zstd`
/// ([`StoredCompression::zstd_at`]).
pub fn encode_store_chunks_compressed(chunks: &[Vec<u8>], docs: &[u32], compact: &[bool], zstd: bool) -> Vec<u8> {
    encode_store_chunks_uid(chunks, docs, compact, zstd, false)
}

/// [`encode_store_chunks_compressed`], every compact chunk flagged
/// [`UID_CHUNK`] when `uid` (format [`crate::codec::UID_STORED_SINCE`]+).
pub fn encode_store_chunks_uid(chunks: &[Vec<u8>], docs: &[u32], compact: &[bool], zstd: bool, uid: bool) -> Vec<u8> {
    use rayon::prelude::*;
    debug_assert_eq!(chunks.len(), docs.len());
    let flag = |i: usize| compact.get(i).copied().unwrap_or(false);
    // zstd chunks are ~15x larger than LZ4 ones: parallelize from two.
    let parallel = if zstd { chunks.len() > 1 } else { chunks.len() > 8 };
    let compressed: Vec<Vec<u8>> = if parallel {
        chunks.par_iter().enumerate().map(|(i, c)| compress_chunk_flagged(c, flag(i), zstd, uid)).collect()
    } else {
        chunks.iter().enumerate().map(|(i, c)| compress_chunk_flagged(c, flag(i), zstd, uid)).collect()
    };
    let mut out = Vec::with_capacity(compressed.iter().map(|c| c.len()).sum::<usize>() + chunks.len() * 8 + 32);
    let mut table = Vec::with_capacity(compressed.len());
    for (c, &n) in compressed.iter().zip(docs) {
        table.push((out.len(), c.len(), n as usize));
        out.extend_from_slice(c);
    }
    let num_docs = docs.iter().map(|&n| n as usize).sum();
    write_table(&table, num_docs, &mut out);
    out
}

/// The parsed chunk table of a store. Parsing walks every chunk entry, so
/// a reader that serves many lookups should parse it once and hand it to
/// [`StoreReader::with_table`].
#[derive(Clone, Debug)]
pub struct StoreTable {
    num_docs: u32,
    /// `(off, len)` of every compressed chunk.
    chunks: Vec<(usize, usize)>,
    /// First doc of every chunk, only for irregular stores (see the module
    /// docs); `None` when chunk `c` starts at `c * CHUNK_DOCS`.
    starts: Option<Vec<u32>>,
}

impl StoreTable {
    /// Bound table allocation using only borrowed header bytes.
    pub fn allocation_bound(data: &[u8]) -> Result<u64> {
        if data.len()<8 { return Err(RusticError::Corrupted("store too short".into())); }
        let mut end=data.len()-8;
        let mut pos=read_u64_le(data,&mut end)? as usize;
        read_vint(data,&mut pos)?;
        let chunks=read_vint(data,&mut pos)?;
        Ok(chunks.saturating_mul((std::mem::size_of::<(usize,usize)>()+2*std::mem::size_of::<u32>()) as u64).saturating_add(256))
    }
    /// Parse the table of the store section `data`.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(RusticError::Corrupted("store too short".into()));
        }
        let mut pos = data.len() - 8;
        let table_offset = read_u64_le(data, &mut pos)? as usize;
        let mut pos = table_offset;
        let num_docs = read_vint(data, &mut pos)? as u32;
        let num_chunks = read_vint(data, &mut pos)? as usize;
        let mut chunks = Vec::with_capacity(num_chunks);
        let mut starts = Vec::new();
        let mut regular = true;
        let mut next = 0u64;
        for c in 0..num_chunks {
            let off = read_vint(data, &mut pos)? as usize;
            let len = read_vint(data, &mut pos)? as usize;
            let docs = read_vint(data, &mut pos)?;
            if off + len > table_offset {
                return Err(RusticError::Corrupted("store chunk out of range".into()));
            }
            // The fixed `doc / CHUNK_DOCS` layout holds only while every
            // chunk but the last has exactly CHUNK_DOCS docs and the last
            // has at most that many. A best_compression chunk holds up to
            // BEST_CHUNK_DOCS: a store whose only (or last) chunk has more
            // than CHUNK_DOCS docs sent `locate` to chunk `doc / 64`, past
            // the table (Rally joins: "the len is 1 but the index is 1").
            if regular && docs != CHUNK_DOCS as u64 && (c + 1 != num_chunks || docs > CHUNK_DOCS as u64) {
                regular = false;
                // The table bound includes one exact starts allocation.
                // Do not grow a partially collected vector geometrically:
                // old/new overlap would exceed that preflight reservation.
                starts = Vec::with_capacity(num_chunks);
                starts.extend((0..c as u32).map(|i| i * CHUNK_DOCS as u32));
            }
            if !regular {
                starts.push(next as u32);
            }
            next += docs;
            chunks.push((off, len));
        }
        if next != num_docs as u64 {
            return Err(RusticError::Corrupted("store doc count mismatch".into()));
        }
        Ok(Self { num_docs, chunks, starts: (!regular).then_some(starts) })
    }
}

/// Decode an LZ4 block written by `lz4_flex::compress_prepend_size` (a
/// little-endian u32 of the decompressed size, then the block), stopping
/// after the first sequence for which `more(decoded so far)` is `None`;
/// `Some(n)`: ask again once `n` bytes are decoded.
fn lz4_prefix(chunk: &[u8], mut more: impl FnMut(&[u8]) -> Option<usize>) -> Result<Vec<u8>> {
    let bad = || RusticError::Corrupted("lz4: malformed block".into());
    let size = chunk_size(chunk).ok_or_else(bad)?;
    let src = &chunk[4..];
    // Slack past `size` for the fixed-width copies below (never part of
    // the result: `len` only counts decoded bytes).
    const SLACK: usize = 32;
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(size.checked_add(SLACK).ok_or_else(bad)?).map_err(|_| RusticError::Memory("lz4: output allocation".into()))?;
    let base = out.as_mut_ptr();
    let mut len = 0usize;
    // Lengths of 15 continue in 255-valued bytes.
    let ext = |i: &mut usize, mut n: usize| -> Result<usize> {
        loop {
            let b = *src.get(*i).ok_or_else(bad)?;
            *i += 1;
            n += b as usize;
            if b != 255 {
                return Ok(n);
            }
        }
    };
    let mut i = 0;
    let mut next = 0usize;
    let finish = |out: &mut Vec<u8>, len: usize| {
        // SAFETY: bytes `..len` of the allocation were written (literals
        // and matches are written strictly in order; fixed-width overcopies
        // only write past `len`, within the slack).
        unsafe { out.set_len(len) };
    };
    while i < src.len() {
        let token = src[i];
        i += 1;
        let mut lit = (token >> 4) as usize;
        if lit == 15 {
            lit = ext(&mut i, lit)?;
        }
        if i.checked_add(lit).is_none_or(|end| end > src.len()) || len + lit > size {
            return Err(bad());
        }
        // SAFETY: `src[i..i + lit]` is in bounds (checked) and `len + lit
        // <= size`; the 16-byte overcopy reads within `src` (checked) and
        // writes below `len + 16 <= size + SLACK`.
        unsafe {
            if lit <= 16 && i + 16 <= src.len() {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(i), base.add(len), 16);
            } else {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(i), base.add(len), lit);
            }
        }
        len += lit;
        i += lit;
        // The last sequence is literals only.
        if i == src.len() {
            break;
        }
        let off = u16::from_le_bytes(src.get(i..i + 2).ok_or_else(bad)?.try_into().unwrap()) as usize;
        i += 2;
        let mut ml = (token & 15) as usize;
        if ml == 15 {
            ml = ext(&mut i, ml)?;
        }
        ml += 4;
        if off == 0 || off > len || len + ml > size {
            return Err(bad());
        }
        let start = len - off;
        // SAFETY: `start < len` and `len + ml <= size`. With `off >= 8`
        // each 8-byte step reads bytes already written (its source ends at
        // most at the destination's start) and writes below `len + ml + 7
        // < size + SLACK`. Shorter offsets repeat the last `off` bytes one
        // byte at a time, as the format defines.
        unsafe {
            if off >= 8 {
                let mut k = 0;
                while k < ml {
                    std::ptr::copy_nonoverlapping(base.add(start + k), base.add(len + k), 8);
                    k += 8;
                }
            } else {
                for k in 0..ml {
                    *base.add(len + k) = *base.add(start + k);
                }
            }
        }
        len += ml;
        if len >= next {
            // SAFETY: bytes `..len` are written.
            let decoded = unsafe { std::slice::from_raw_parts(base, len) };
            match more(decoded) {
                None => { finish(&mut out, len); return Ok(out); }
                Some(n) => next = n,
            }
        }
    }
    if len != size {
        return Err(bad());
    }
    finish(&mut out, len);
    Ok(out)
}

/// Reader over an encoded store.
pub struct StoreReader<'a> {
    data: &'a [u8],
    table: Cow<'a, StoreTable>,
}

impl<'a> StoreReader<'a> {
    /// Open `data`, parsing its chunk table.
    pub fn open(data: &'a [u8]) -> Result<Self> {
        Ok(Self { data, table: Cow::Owned(StoreTable::parse(data)?) })
    }

    /// A reader over `data` whose table was parsed beforehand
    /// (`StoreTable::parse(data)`).
    pub fn with_table(data: &'a [u8], table: &'a StoreTable) -> Self {
        Self { data, table: Cow::Borrowed(table) }
    }

    /// Number of compressed chunks.
    pub fn num_chunks(&self) -> usize {
        self.table.chunks.len()
    }

    /// The compressed bytes of chunk `c` (as stored; copyable verbatim).
    pub fn raw_chunk(&self, c: usize) -> &'a [u8] {
        let (off, len) = self.table.chunks[c];
        &self.data[off..off + len]
    }

    /// First doc of chunk `c` and its doc count.
    pub fn chunk_docs(&self, c: usize) -> (DocId, u32) {
        let t = &*self.table;
        let start = match &t.starts {
            Some(s) => s[c],
            None => (c * CHUNK_DOCS) as u32,
        };
        let end = match &t.starts {
            Some(s) => s.get(c + 1).copied().unwrap_or(t.num_docs),
            None => (((c + 1) * CHUNK_DOCS) as u32).min(t.num_docs),
        };
        (start, end - start)
    }

    /// The LZ4 decoded allocation size, read before decompression allocates.
    pub fn decoded_chunk_bytes(&self,c:usize)->Option<usize> {
        chunk_size(self.raw_chunk(c))
    }

    /// Whether chunk `c`'s docs are compact-encoded ([`COMPACT_CHUNK`]).
    pub fn chunk_compact(&self, c: usize) -> bool {
        chunk_is_compact(self.raw_chunk(c))
    }

    /// Whether chunk `c` is zstd-compressed ([`ZSTD_CHUNK`]).
    pub fn chunk_zstd(&self, c: usize) -> bool {
        chunk_is_zstd(self.raw_chunk(c))
    }

    /// Whether any chunk is zstd-compressed (the segment was written with
    /// [`StoredCompression::Best`]).
    pub fn any_zstd(&self) -> bool {
        (0..self.num_chunks()).any(|c| self.chunk_zstd(c))
    }

    /// Decompress chunk `c` (per doc: `vint len, bincode(Document)`).
    pub fn decompress(&self, c: usize) -> Result<Vec<u8>> {
        self.decompress_chunk(c)
    }

    pub fn num_docs(&self) -> u32 {
        self.table.num_docs
    }

    fn decompress_chunk(&self, chunk: usize) -> Result<Vec<u8>> {
        decompress_chunk_bytes(self.raw_chunk(chunk))
    }

    /// The decompressed prefix of chunk `c` that holds its docs up to and
    /// including `within` (the whole chunk when they reach its end): a
    /// fetch does not decompress the docs after its own.
    pub fn decompress_through(&self, c: usize, within: usize) -> Result<Vec<u8>> {
        if within + 1 >= self.chunk_docs(c).1 as usize || self.chunk_zstd(c) {
            // The chunk's last doc: all of it, at lz4_flex's speed. A zstd
            // chunk is one frame, decompressed whole.
            return self.decompress_chunk(c);
        }
        let (mut pos, mut left) = (0usize, within + 1);
        lz4_prefix(self.raw_chunk(c), |out| {
            while left > 0 {
                let mut p = pos;
                // A length prefix is at most 10 bytes.
                let Ok(len) = read_vint(out, &mut p) else { return Some(pos + 10) };
                let end = p.saturating_add(len as usize);
                if end > out.len() {
                    return Some(end);
                }
                pos = end;
                left -= 1;
            }
            None
        })
    }

    /// The chunk holding `doc` and the doc's position in it.
    pub fn locate(&self, doc: DocId) -> Option<(usize, usize)> {
        if doc >= self.table.num_docs {
            return None;
        }
        Some(match &self.table.starts {
            None => (doc as usize / CHUNK_DOCS, doc as usize % CHUNK_DOCS),
            Some(s) => {
                let c = s.partition_point(|&st| st <= doc) - 1;
                (c, (doc - s[c]) as usize)
            }
        })
    }

    /// Document number `within` of a decompressed bincode chunk.
    pub fn doc_in_chunk(raw: &[u8], within: usize) -> Result<Document> {
        Self::doc_in_chunk_as(raw, within, false)
    }

    /// Document number `within` of a decompressed chunk of either encoding.
    pub fn doc_in_chunk_as(raw: &[u8], within: usize, compact: bool) -> Result<Document> {
        decode_doc(Self::doc_bytes_in_chunk(raw, within)?, compact)
    }

    /// Borrow a framed document before its decoder allocates owned fields.
    pub fn doc_bytes_in_chunk(raw:&[u8],within:usize)->Result<&[u8]> {
        let mut pos = 0;
        for _ in 0..within {
            let len = read_vint(raw, &mut pos)? as usize;
            pos += len;
        }
        let len = read_vint(raw, &mut pos)? as usize;
        read_bytes(raw,&mut pos,len)
    }

    /// Fetch one document.
    pub fn get(&self, doc: DocId) -> Result<Option<Document>> {
        let Some((chunk, within)) = self.locate(doc) else {
            return Ok(None);
        };
        let raw = self.decompress_through(chunk, within)?;
        Self::doc_in_chunk_as(&raw, within, self.chunk_compact(chunk)).map(Some)
    }

    /// Fetch the documents `docs` (ascending), decompressing each chunk
    /// once: `f(doc, document)` in order. Missing docs are skipped.
    pub fn get_many(&self, docs: &[DocId], mut f: impl FnMut(DocId, Document)) -> Result<()> {
        let mut cur: Option<(usize, Vec<u8>, usize, u32)> = None; // chunk, bytes, pos, next doc at pos
        for &doc in docs {
            if doc >= self.table.num_docs {
                continue;
            }
            let chunk = match &self.table.starts {
                None => doc as usize / CHUNK_DOCS,
                Some(s) => s.partition_point(|&st| st <= doc) - 1,
            };
            let (start, _) = self.chunk_docs(chunk);
            if cur.as_ref().is_none_or(|c| c.0 != chunk || c.3 > doc) {
                cur = Some((chunk, self.decompress_chunk(chunk)?, 0, start));
            }
            let (_, raw, pos, next) = cur.as_mut().unwrap();
            while *next < doc {
                let len = read_vint(raw, pos)? as usize;
                *pos += len;
                *next += 1;
            }
            let len = read_vint(raw, pos)? as usize;
            let bytes = read_bytes(raw, pos, len)?;
            *next += 1;
            let d = decode_doc(bytes, self.chunk_compact(chunk))?;
            f(doc, d);
        }
        Ok(())
    }

    /// Visit every document in order (used by merging).
    pub fn for_each(&self, mut f: impl FnMut(DocId, Document)) -> Result<()> {
        let mut doc_id: DocId = 0;
        for chunk in 0..self.num_chunks() {
            let raw = self.decompress_chunk(chunk)?;
            let compact = self.chunk_compact(chunk);
            let mut pos = 0;
            while pos < raw.len() {
                let len = read_vint(&raw, &mut pos)? as usize;
                let bytes = read_bytes(&raw, &mut pos, len)?;
                let doc = decode_doc(bytes, compact)?;
                f(doc_id, doc);
                doc_id += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs(n: usize) -> Vec<Document> {
        (0..n)
            .map(|i| {
                let mut d = Document::new();
                d.add_text(0, &format!("document number {i} with some repeated text text text"));
                d.add_i64(1, i as i64);
                d.add_bool(2, i % 2 == 0);
                d
            })
            .collect()
    }

    #[test]
    fn lz4_prefix_decodes_like_lz4_flex() {
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        let mut cases: Vec<Vec<u8>> = vec![Vec::new(), b"a".to_vec(), vec![0u8; 70_000], b"abcabcabcabc".repeat(900)];
        cases.push((0..50_000).map(|_| rnd()).collect());
        cases.push((0..60_000).map(|i| if i % 97 < 50 { b"xy"[i % 2] } else { rnd() % 4 }).collect());
        // Every short match offset (the byte-wise copies) and long literals.
        for period in 1..=17usize {
            let unit: Vec<u8> = (0..period).map(|_| rnd()).collect();
            let mut data: Vec<u8> = (0..3_000).map(|_| rnd()).collect();
            data.extend(unit.iter().cycle().take(5_000 + period));
            data.extend((0..37).map(|_| rnd()));
            cases.push(data);
        }
        cases.push(b"long article body words ".repeat(700));
        for data in cases {
            let packed = lz4_flex::compress_prepend_size(&data);
            assert_eq!(lz4_prefix(&packed, |_| Some(usize::MAX)).unwrap(), data);
            // Stopping early yields a prefix that covers what was asked.
            for want in [1usize, 100, 5000, 40_000] {
                let p = lz4_prefix(&packed, |out| (out.len() < want).then_some(want)).unwrap();
                assert!(data.starts_with(&p) && p.len() >= want.min(data.len()), "{} of {}", p.len(), data.len());
            }
        }
        assert!(lz4_prefix(&[5, 0, 0, 0, 0x50, b'a'], |_| Some(usize::MAX)).is_err());
    }

    /// A store whose one (or last) chunk holds more than CHUNK_DOCS docs
    /// (a best_compression chunk) locates every doc in it.
    #[test]
    fn large_last_chunk_is_not_regular() {
        let docs = docs(3 * CHUNK_DOCS + 5);
        let fixed_bytes = encode_store(&docs);
        let fixed = StoreReader::open(&fixed_bytes).unwrap();
        let mut raw = Vec::new();
        for c in 0..fixed.num_chunks() {
            raw.extend_from_slice(&fixed.decompress(c).unwrap());
        }
        for split in [vec![docs.len() as u32], vec![CHUNK_DOCS as u32, (docs.len() - CHUNK_DOCS) as u32]] {
            // Re-chunk the same docs: one chunk, or a regular one then a large last one.
            let mut chunks = Vec::new();
            let mut pos = 0usize;
            for &n in &split {
                let start = pos;
                for _ in 0..n {
                    let len = read_vint(&raw, &mut pos).unwrap() as usize;
                    pos += len;
                }
                chunks.push(raw[start..pos].to_vec());
            }
            let compact = vec![fixed.chunk_compact(0); split.len()];
            let bytes = encode_store_chunks_flagged(&chunks, &split, &compact);
            let r = StoreReader::open(&bytes).unwrap();
            for (d, want) in docs.iter().enumerate() {
                let got = r.get(d as DocId).unwrap().unwrap();
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "split {split:?} doc {d}");
            }
        }
    }

    #[test]
    fn chunk_doc_offsets_find_every_doc() {
        let mut docs = docs(300);
        for i in 0..40 {
            let mut d = Document::new();
            d.add_text(0, &format!("{i} {}", "body ".repeat(10 + i * 41)));
            docs.push(d);
        }
        let bytes = encode_store(&docs);
        let r = StoreReader::open(&bytes).unwrap();
        for c in 0..r.num_chunks() {
            let (_, n) = r.chunk_docs(c);
            for within in 0..n as usize {
                // Prefixes (partial decompression) hold the docs up to `within`.
                for raw in [r.decompress(c).unwrap(), r.decompress_through(c, within).unwrap()] {
                    let offsets = chunk_doc_offsets(&raw);
                    assert!(offsets.len() > within, "chunk {c} doc {within}");
                    assert_eq!(doc_bytes_at(&raw, offsets[within] as usize).unwrap(), StoreReader::doc_bytes_in_chunk(&raw, within).unwrap());
                }
            }
        }
    }

    #[test]
    fn decompress_through_covers_the_doc() {
        let mut docs = docs(200);
        // Byte-bounded chunks of a few large docs.
        for i in 0..40 {
            let mut d = Document::new();
            d.add_text(0, &format!("{i} {}", "long article body words ".repeat(50 + i * 37)));
            docs.push(d);
        }
        let bytes = encode_store(&docs);
        let r = StoreReader::open(&bytes).unwrap();
        for c in 0..r.num_chunks() {
            let full = r.decompress(c).unwrap();
            let (_, n) = r.chunk_docs(c);
            for within in 0..n as usize {
                let p = r.decompress_through(c, within).unwrap();
                assert!(full.starts_with(&p), "chunk {c}");
                let (a, b) = (StoreReader::doc_in_chunk(&p, within).unwrap(), StoreReader::doc_in_chunk(&full, within).unwrap());
                assert_eq!(format!("{a:?}"), format!("{b:?}"), "chunk {c} doc {within}");
            }
            assert_eq!(r.decompress_through(c, n as usize - 1).unwrap(), full);
        }
        for d in 0..docs.len() as u32 {
            let got = r.get(d).unwrap().unwrap();
            assert_eq!(got.get(0).unwrap().as_text(), docs[d as usize].get(0).unwrap().as_text(), "doc {d}");
        }
    }

    #[test]
    fn get_many_matches_get() {
        let docs = docs(300);
        let bytes = encode_store(&docs);
        let r = StoreReader::open(&bytes).unwrap();
        let want: Vec<DocId> = vec![0, 1, 2, 63, 64, 65, 100, 101, 190, 255, 299, 300, 5000];
        let mut got = Vec::new();
        r.get_many(&want, |d, doc| got.push((d, doc))).unwrap();
        assert_eq!(got.len(), 11);
        for (d, doc) in got {
            assert_eq!(format!("{doc:?}"), format!("{:?}", r.get(d).unwrap().unwrap()), "doc {d}");
        }
    }

    #[test]
    fn roundtrip() {
        let docs = docs(150);
        let bytes = encode_store(&docs);
        let r = StoreReader::open(&bytes).unwrap();
        assert_eq!(r.num_docs(), 150);
        for (i, d) in docs.iter().enumerate() {
            let got = r.get(i as DocId).unwrap().unwrap();
            assert_eq!(got.get(0).unwrap().as_text(), d.get(0).unwrap().as_text());
            assert_eq!(got.get(1).unwrap().as_i64(), Some(i as i64));
        }
        assert!(r.get(150).unwrap().is_none());
        let mut seen = 0;
        r.for_each(|id, d| {
            assert_eq!(d.get(1).unwrap().as_i64(), Some(id as i64));
            seen += 1;
        })
        .unwrap();
        assert_eq!(seen, 150);
    }

    /// The `_id` keyword in its binary form reads back as the same
    /// keyword, other fields (a keyword of another field holding an id
    /// shape included) unchanged; the chunk flag travels with the chunk and
    /// a downgrade re-encodes the ids as keywords.
    /// `split_uid` takes the binary id out of a compact doc (other fields'
    /// bytes unchanged) and `join_uid` puts it back as the last field: the
    /// doc decodes to the same fields; docs without one pass through.
    #[test]
    fn split_and_join_uid_round_trip() {
        use crate::types::FieldValue as V;
        let mut doc = every_kind();
        doc.fields.insert(1, (9, V::Keyword("AaGc2ZTwVrHG61YG-FIk".into())));
        let mut bytes = Vec::new();
        encode_doc_compact_uid(doc.fields.iter(), Some(9), &mut bytes);
        let mut framed = Vec::new();
        let (field, uid) = split_uid(&bytes, &mut framed).unwrap().unwrap();
        assert_eq!((field, crate::uid::decode(&uid).unwrap().as_str()), (9, "AaGc2ZTwVrHG61YG-FIk"));
        let stripped = doc_bytes_at(&framed, 0).unwrap().to_vec();
        let mut without = doc.clone();
        without.fields.remove(1);
        assert_eq!(format!("{:?}", decode_doc_compact(&stripped).unwrap().fields), format!("{:?}", without.fields));
        let mut back = Vec::new();
        join_uid(&stripped, field, &uid, &mut back).unwrap();
        let mut want = without.clone();
        want.fields.push((9, V::Keyword("AaGc2ZTwVrHG61YG-FIk".into())));
        assert_eq!(format!("{:?}", decode_doc_compact(doc_bytes_at(&back, 0).unwrap()).unwrap().fields), format!("{:?}", want.fields));
        let mut plain = Vec::new();
        encode_doc_compact(without.fields.iter(), &mut plain);
        let mut out = Vec::new();
        assert!(split_uid(&plain, &mut out).unwrap().is_none());
        assert_eq!(doc_bytes_at(&out, 0).unwrap(), plain.as_slice());
    }

    #[test]
    fn uid_values_round_trip_and_downgrade() {
        use crate::types::FieldValue as V;
        let ids = ["AaGbCX0F9q8YlnEvGqAa", "42", "my doc", "ab", "_-__", "日本"];
        let mut chunk = Vec::new();
        let mut plain = Vec::new();
        let docs: Vec<Document> = ids.iter().enumerate().map(|(i, id)| {
            let mut d = Document::new();
            d.fields.push((0, V::Keyword(id.to_string())));
            d.fields.push((1, V::Text(format!("{{\"n\":{i}}}"))));
            d.fields.push((3, V::Keyword(id.to_string())));
            d.fields.push((0, V::Keyword(format!("{id}x"))));
            d
        }).collect();
        for d in &docs {
            let mut one = Vec::new();
            encode_doc_compact_uid(d.fields.iter(), Some(0), &mut one);
            write_vint(one.len() as u64, &mut chunk);
            chunk.extend_from_slice(&one);
            let mut p = Vec::new();
            encode_doc_compact(d.fields.iter(), &mut p);
            write_vint(p.len() as u64, &mut plain);
            plain.extend_from_slice(&p);
            assert_eq!(format!("{:?}", decode_doc_compact(&one).unwrap().fields), format!("{:?}", d.fields));
        }
        // An auto id takes 15 bytes, not 20.
        let auto = [(0u16, V::Keyword(ids[0].to_string()))];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        encode_doc_compact_uid(auto.iter(), Some(0), &mut a);
        encode_doc_compact(auto.iter(), &mut b);
        assert_eq!(a.len() + 5, b.len());
        let c = compress_chunk_flagged(&chunk, true, false, true);
        assert!(chunk_is_uid(&c) && chunk_is_compact(&c) && !chunk_is_zstd(&c));
        assert_eq!(decompress_chunk_bytes(&c).unwrap(), chunk);
        assert!(!chunk_is_uid(&compress_chunk_flagged(&chunk, false, false, true)), "only compact chunks carry it");
        let store = encode_store_chunks_uid(&[chunk.clone()], &[docs.len() as u32], &[true], false, true);
        let r = StoreReader::open(&store).unwrap();
        for (i, d) in docs.iter().enumerate() {
            assert_eq!(format!("{:?}", r.get(i as DocId).unwrap().unwrap().fields), format!("{:?}", d.fields));
        }
        assert_eq!(compact_chunk_without_uid(&chunk).unwrap(), plain);
    }

    #[test]
    fn chunk_encoding_is_byte_identical() {
        for n in [0usize, 1, 63, 64, 65, 700] {
            let docs = docs(n);
            let mut chunks: Vec<Vec<u8>> = Vec::new();
            let mut counts: Vec<u32> = Vec::new();
            for d in docs.iter() {
                if chunks.last().zip(counts.last()).is_none_or(|(c, &k)| chunk_full(k as usize, c.len())) {
                    chunks.push(Vec::new());
                    counts.push(0);
                }
                let c = chunks.last_mut().unwrap();
                write_vint(bincode::serialized_size(d).unwrap(), c);
                bincode::serialize_into(&mut *c, d).unwrap();
                *counts.last_mut().unwrap() += 1;
            }
            assert_eq!(encode_store_chunks(&chunks, &counts), encode_store(&docs), "{n}");
        }
    }

    #[test]
    fn large_docs_are_chunked_by_bytes() {
        // ~20 KB docs (a 1024-dim vector as JSON in `_source`): a chunk must
        // close on CHUNK_BYTES long before CHUNK_DOCS.
        let docs: Vec<Document> = (0..300)
            .map(|i| {
                let mut d = Document::new();
                let vec: Vec<String> = (0..1024).map(|j| format!("{:.7}", ((i * 1024 + j) as f32).sin())).collect();
                d.add_text(0, &format!("{{\"v\":[{}]}}", vec.join(",")));
                d.add_i64(1, i as i64);
                d
            })
            .collect();
        let bytes = encode_store(&docs);
        let r = StoreReader::open(&bytes).unwrap();
        assert!(r.num_chunks() > 300 / CHUNK_DOCS * 4, "{} chunks", r.num_chunks());
        for c in 0..r.num_chunks() {
            let raw = r.decompress(c).unwrap().len();
            let one_doc = bincode::serialized_size(&docs[0]).unwrap() as usize + 3;
            assert!(raw < CHUNK_BYTES + one_doc * 2, "chunk {c}: {raw} bytes");
        }
        for (i, d) in docs.iter().enumerate() {
            assert_eq!(format!("{:?}", r.get(i as DocId).unwrap().unwrap()), format!("{d:?}"), "doc {i}");
        }
        // A reader over a pre-parsed table answers the same.
        let table = StoreTable::parse(&bytes).unwrap();
        let r2 = StoreReader::with_table(&bytes, &table);
        assert_eq!(r2.num_docs(), 300);
        assert_eq!(r2.get(257).unwrap().unwrap().get(1).unwrap().as_i64(), Some(257));
    }

    /// Chunks of docs averaging over LARGE_DOC_BYTES close at
    /// LARGE_DOC_CHUNK_BYTES (a fetch decodes less); smaller docs fill
    /// CHUNK_BYTES (a wider compression window): the rule as the writers
    /// apply it ([`chunk_full`]).
    #[test]
    fn chunk_size_follows_document_size() {
        let fill = |doc: usize| {
            let (mut docs, mut raw) = (0usize, 0usize);
            while !chunk_full(docs, raw) {
                docs += 1;
                raw += doc;
            }
            raw
        };
        assert!((LARGE_DOC_CHUNK_BYTES..LARGE_DOC_CHUNK_BYTES + 500).contains(&fill(440)), "{}", fill(440));
        assert!((LARGE_DOC_CHUNK_BYTES..LARGE_DOC_CHUNK_BYTES + 3000).contains(&fill(2500)));
        assert!((CHUNK_BYTES..CHUNK_BYTES + 200).contains(&fill(150)), "{}", fill(150));
        assert_eq!(fill(40), CHUNK_DOCS * 40);
        // A segment without vectors: noaa-sized (290-byte) and larger docs
        // fill CHUNK_BYTES for the wider compression window.
        let plain = |doc: usize| {
            let (mut docs, mut raw) = (0usize, 0usize);
            while !chunk_full_for(docs, docs, raw, false) {
                docs += 1;
                raw += doc;
            }
            raw
        };
        assert!((CHUNK_BYTES..CHUNK_BYTES + 300).contains(&plain(290)), "{}", plain(290));
        assert!((CHUNK_BYTES..CHUNK_BYTES + 2600).contains(&plain(2500)), "{}", plain(2500));
        // Without vectors, a nested segment's parents behind empty children
        // still close at LARGE_DOC_CHUNK_BYTES (inner hits read them).
        let (mut docs, mut sized, mut raw) = (0usize, 0usize, 0usize);
        while !chunk_full_for(docs, sized, raw, false) {
            if docs % 4 == 0 { raw += 330; sized += 1; } else { raw += 2; }
            docs += 1;
        }
        assert!((LARGE_DOC_CHUNK_BYTES..LARGE_DOC_CHUNK_BYTES + 400).contains(&raw), "{raw}");
        // Nested: each parent behind empty child docs (not counted in the
        // average) closes at LARGE_DOC_CHUNK_BYTES as its parents alone would.
        let (mut docs, mut sized, mut raw) = (0usize, 0usize, 0usize);
        while !chunk_full_sized(docs, sized, raw) {
            if docs % 4 == 0 { raw += 330; sized += 1; } else { raw += 2; }
            docs += 1;
        }
        assert!((LARGE_DOC_CHUNK_BYTES..LARGE_DOC_CHUNK_BYTES + 400).contains(&raw), "{raw}");
        // A chunk of small docs then large ones closes once their average passes.
        let (mut docs, mut raw) = (0usize, 0usize);
        while !chunk_full(docs, raw) {
            raw += if docs < 60 { 100 } else { 2000 };
            docs += 1;
        }
        assert!(raw >= LARGE_DOC_CHUNK_BYTES && raw < CHUNK_BYTES, "{raw}");
    }

    /// Chunks of any doc count (best_compression: up to BEST_CHUNK_DOCS),
    /// including a store whose only chunk, or whose last chunk after
    /// full-size ones, holds more than CHUNK_DOCS docs: every doc locates
    /// to its own chunk and reads back (get, get_many, chunk_docs, a
    /// pre-parsed table), and docs past the end are none.
    #[test]
    fn variable_size_chunks_locate_every_doc() {
        for counts in [vec![200u32], vec![64, 64, 150], vec![64, 10], vec![100, 64], vec![2048, 3, 700], vec![1], vec![64]] {
            let n: u32 = counts.iter().sum();
            let all = docs(n as usize);
            let mut chunks: Vec<Vec<u8>> = Vec::new();
            let mut i = 0usize;
            for &k in &counts {
                let mut c = Vec::new();
                for d in &all[i..i + k as usize] {
                    write_vint(bincode::serialized_size(d).unwrap(), &mut c);
                    bincode::serialize_into(&mut c, d).unwrap();
                }
                i += k as usize;
                chunks.push(c);
            }
            for zstd in [false, true] {
                let bytes = encode_store_chunks_compressed(&chunks, &counts, &[], zstd);
                let table = StoreTable::parse(&bytes).unwrap();
                for r in [StoreReader::open(&bytes).unwrap(), StoreReader::with_table(&bytes, &table)] {
                    assert_eq!(r.num_chunks(), counts.len());
                    let mut first = 0u32;
                    for (c, &k) in counts.iter().enumerate() {
                        assert_eq!(r.chunk_docs(c), (first, k), "{counts:?} chunk {c}");
                        first += k;
                    }
                    for doc in 0..n {
                        let (c, within) = r.locate(doc).unwrap_or_else(|| panic!("{counts:?} doc {doc}"));
                        assert!(c < r.num_chunks(), "{counts:?} doc {doc} -> chunk {c}");
                        let (start, k) = r.chunk_docs(c);
                        assert_eq!((start + within as u32, within < k as usize), (doc, true), "{counts:?} doc {doc}");
                        let got = r.get(doc).unwrap().unwrap();
                        assert_eq!(got.get(1).unwrap().as_i64(), Some(doc as i64), "{counts:?} doc {doc}");
                    }
                    assert!(r.locate(n).is_none() && r.get(n).unwrap().is_none());
                    let wanted: Vec<DocId> = (0..n).filter(|d| d % 7 != 3).chain([n, n + 5]).collect();
                    let mut seen = Vec::new();
                    r.get_many(&wanted, |d, doc| {
                        assert_eq!(doc.get(1).unwrap().as_i64(), Some(d as i64));
                        seen.push(d);
                    }).unwrap();
                    assert_eq!(seen, wanted.iter().copied().filter(|&d| d < n).collect::<Vec<_>>(), "{counts:?}");
                }
            }
        }
    }

    #[test]
    fn irregular_chunks_from_raw_copies() {
        // Copy the chunks of three stores verbatim (sizes not multiples of
        // CHUNK_DOCS) and re-index them with `encode_store_table`.
        let parts = [docs(70), docs(64), docs(5)];
        let encoded: Vec<Vec<u8>> = parts.iter().map(|d| encode_store(d)).collect();
        let mut body = Vec::new();
        let mut table = Vec::new();
        for e in &encoded {
            let r = StoreReader::open(e).unwrap();
            for c in 0..r.num_chunks() {
                body.extend_from_slice(r.raw_chunk(c));
                table.push((r.raw_chunk(c).len(), r.chunk_docs(c).1 as usize));
            }
        }
        body.extend_from_slice(&encode_store_table(&table));
        let r = StoreReader::open(&body).unwrap();
        assert_eq!(r.num_docs(), 139);
        assert_eq!(r.chunk_docs(1), (70, 64));
        let all: Vec<&Document> = parts.iter().flatten().collect();
        for (i, d) in all.iter().enumerate() {
            let got = r.get(i as DocId).unwrap().unwrap();
            assert_eq!(format!("{got:?}"), format!("{d:?}"), "doc {i}");
        }
        assert!(r.get(139).unwrap().is_none());
        let mut n = 0;
        r.for_each(|id, d| {
            assert_eq!(format!("{d:?}"), format!("{:?}", all[id as usize]));
            n += 1;
        })
        .unwrap();
        assert_eq!(n, 139);
    }

    fn every_kind() -> Document {
        use crate::types::FieldValue as V;
        let mut d = Document::new();
        d.fields = vec![
            (0, V::Text("héllo wörld".into())),
            (1, V::Keyword(String::new())),
            (2, V::I64(i64::MIN)),
            (2, V::I64(-1)),
            (3, V::U64(u64::MAX)),
            (4, V::F64(-0.25)),
            (5, V::Bool(true)),
            (6, V::Date(-86_400_000)),
            (7, V::GeoPoint(51.5, -0.12)),
            (8, V::Vector(vec![1.5, -2.0, 0.0])),
            (u16::MAX, V::I64(i64::MAX)),
        ];
        d
    }

    #[test]
    fn compact_docs_round_trip_and_are_smaller() {
        let d = every_kind();
        let mut buf = Vec::new();
        encode_doc_compact(d.fields.iter(), &mut buf);
        let back = decode_doc(&buf, true).unwrap();
        assert_eq!(format!("{:?}", back.fields), format!("{:?}", d.fields));
        assert!(buf.len() < bincode::serialize(&d).unwrap().len());
        for doc in docs(20) {
            let mut b = Vec::new();
            encode_doc_compact(doc.fields.iter(), &mut b);
            assert_eq!(format!("{:?}", decode_doc_compact(&b).unwrap().fields), format!("{:?}", doc.fields));
            assert!(b.len() + 25 < bincode::serialize(&doc).unwrap().len(), "{} bytes", b.len());
        }
        // Truncated, trailing or unknown-kind bytes are corruption, never a doc.
        for end in 0..buf.len() { assert!(decode_doc_compact(&buf[..end]).is_err(), "truncated at {end}"); }
        let mut trailing = buf.clone(); trailing.push(0);
        assert!(decode_doc_compact(&trailing).is_err());
        assert!(decode_doc_compact(&[1, 0, 9]).is_err());
    }

    #[test]
    fn flagged_and_plain_chunks_read_back_together() {
        let all = docs(150);
        // Chunk 0 and 2 bincode, chunk 1 compact (a store merged verbatim
        // from segments written before and after the format floor rose).
        let bounds = [(0usize, 64usize), (64, 128), (128, 150)];
        let mut chunks = Vec::new();
        let mut counts = Vec::new();
        for (i, &(a, b)) in bounds.iter().enumerate() {
            let mut raw = Vec::new();
            for d in &all[a..b] {
                let bytes = if i == 1 {
                    let mut v = Vec::new();
                    encode_doc_compact(d.fields.iter(), &mut v);
                    v
                } else {
                    bincode::serialize(d).unwrap()
                };
                write_vint(bytes.len() as u64, &mut raw);
                raw.extend_from_slice(&bytes);
            }
            chunks.push(raw);
            counts.push((b - a) as u32);
        }
        let flags = [false, true, false];
        let bytes = encode_store_chunks_flagged(&chunks, &counts, &flags);
        let r = StoreReader::open(&bytes).unwrap();
        assert_eq!(r.num_docs(), 150);
        for c in 0..3 {
            assert_eq!(r.chunk_compact(c), flags[c]);
            assert_eq!(chunk_is_compact(r.raw_chunk(c)), flags[c]);
            assert_eq!(r.decoded_chunk_bytes(c), Some(chunks[c].len()));
            assert_eq!(r.decompress(c).unwrap(), chunks[c]);
        }
        let want = |i: usize| format!("{:?}", all[i].fields);
        for i in 0..150 {
            assert_eq!(format!("{:?}", r.get(i as DocId).unwrap().unwrap().fields), want(i), "doc {i}");
        }
        let ids: Vec<DocId> = (0..150).step_by(7).collect();
        let mut seen = 0;
        r.get_many(&ids, |id, d| { assert_eq!(format!("{:?}", d.fields), want(id as usize)); seen += 1; }).unwrap();
        assert_eq!(seen, ids.len());
        let mut n = 0;
        r.for_each(|id, d| { assert_eq!(format!("{:?}", d.fields), want(id as usize)); n += 1; }).unwrap();
        assert_eq!(n, 150);
        // The flag never leaks into the size: a plain chunk is byte-identical
        // to lz4_flex's own framing.
        assert_eq!(compress_chunk(&chunks[0], false), lz4_flex::compress_prepend_size(&chunks[0]));
        assert_eq!(decompress_chunk_bytes(&compress_chunk(&chunks[1], true)).unwrap(), chunks[1]);
    }

    #[test]
    fn zstd_chunks_round_trip_and_compress_repetitive_docs() {
        // Metrics-shaped docs: a large block identical across docs, a few
        // per-doc values.
        let shared = format!("{{\"host\":{{\"ip\":[{}]}}}}", (0..200).map(|i| format!("\"10.{}.{}.1\"", i / 7, i % 251)).collect::<Vec<_>>().join(","));
        let all: Vec<Document> = (0..3000)
            .map(|i| {
                let mut d = Document::new();
                d.add_keyword(1, &format!("{{\"@timestamp\":{},\"v\":{},\"s\":{shared}}}", 1_700_000_000_000u64 + i * 10_000, i * 7919 % 1000));
                d.add_i64(2, i as i64);
                d
            })
            .collect();
        let frame = |mode: StoredCompression| {
            let (mut chunks, mut counts): (Vec<Vec<u8>>, Vec<u32>) = (Vec::new(), Vec::new());
            for d in &all {
                if chunks.is_empty() || mode.chunk_full(*counts.last().unwrap() as usize, chunks.last().unwrap().len()) {
                    chunks.push(Vec::new());
                    counts.push(0);
                }
                let mut one = Vec::new();
                encode_doc_compact(d.fields.iter(), &mut one);
                write_vint(one.len() as u64, chunks.last_mut().unwrap());
                chunks.last_mut().unwrap().extend_from_slice(&one);
                *counts.last_mut().unwrap() += 1;
            }
            let compact = vec![true; chunks.len()];
            (chunks, counts, compact)
        };
        let (fc, fn_, fk) = frame(StoredCompression::Fast);
        let fast = encode_store_chunks_flagged(&fc, &fn_, &fk);
        let (bc, bn, bk) = frame(StoredCompression::Best);
        assert!(bc.len() < fc.len(), "{} best chunks vs {} fast", bc.len(), fc.len());
        let best = encode_store_chunks_compressed(&bc, &bn, &bk, true);
        assert!(best.len() * 4 < fast.len(), "best {} vs fast {} bytes", best.len(), fast.len());
        let r = StoreReader::open(&best).unwrap();
        assert!(r.any_zstd() && (0..r.num_chunks()).all(|c| r.chunk_zstd(c) && r.chunk_compact(c)));
        assert!(!StoreReader::open(&fast).unwrap().any_zstd());
        let want = |i: usize| format!("{:?}", all[i].fields);
        for i in [0usize, 1, 63, 64, 2047, 2048, 2999] {
            assert_eq!(format!("{:?}", r.get(i as DocId).unwrap().unwrap().fields), want(i), "doc {i}");
        }
        let (c, within) = r.locate(100).unwrap();
        let raw = r.decompress_through(c, within).unwrap();
        assert_eq!(format!("{:?}", StoreReader::doc_in_chunk_as(&raw, within, true).unwrap().fields), want(100));
        let ids: Vec<DocId> = (0..3000).step_by(13).collect();
        let mut seen = 0;
        r.get_many(&ids, |id, d| { assert_eq!(format!("{:?}", d.fields), want(id as usize)); seen += 1; }).unwrap();
        assert_eq!(seen, ids.len());
        let mut n = 0;
        r.for_each(|id, d| { assert_eq!(format!("{:?}", d.fields), want(id as usize)); n += 1; }).unwrap();
        assert_eq!(n, 3000);
        // The size prefix carries both flags without touching the size.
        let z = compress_chunk_with(&bc[0], true, true);
        assert!(chunk_is_zstd(&z) && chunk_is_compact(&z));
        assert_eq!(chunk_size(&z), Some(bc[0].len()));
        assert_eq!(decompress_chunk_bytes(&z).unwrap(), bc[0]);
        assert_eq!(compress_chunk_with(&bc[0], false, false), compress_chunk(&bc[0], false));
        assert_eq!(StoredCompression::from_codec(Some("best_compression")), StoredCompression::Best);
        assert_eq!(StoredCompression::from_codec(Some("default")), StoredCompression::Fast);
        assert!(!StoredCompression::Best.zstd_at(crate::codec::ZSTD_STORED_SINCE - 1));
        assert!(StoredCompression::Best.zstd_at(crate::codec::ZSTD_STORED_SINCE) && !StoredCompression::Fast.zstd_at(crate::codec::VERSION));
    }

    /// Fetch cost of [`StoredCompression::Best`] chunk sizes:
    /// `cargo test -p rustic-core --release --lib best_chunk_fetch_microbench -- --ignored --nocapture`
    /// (`BEST_BENCH_DOCS`, default 300k big5-like ~1 KB log docs;
    /// `BEST_BENCH_CORPUS`: a file of JSON docs, one per line, instead).
    /// Prints stored bytes/doc and the time to fetch one random doc, ten
    /// random docs (a size-10 search) and ten consecutive docs, per chunk
    /// size.
    #[test]
    #[ignore]
    fn best_chunk_fetch_microbench() {
        use std::time::Instant;
        let n: usize = std::env::var("BEST_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(300_000);
        let mut seed = 7u64;
        let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); seed >> 33 };
        let sources: Vec<String> = match std::env::var("BEST_BENCH_CORPUS") {
            Ok(p) => std::fs::read_to_string(p).unwrap().lines().take(n).map(str::to_string).collect(),
            Err(_) => (0..n).map(|i| {
                let host = rnd() % 200;
                format!(
                    r#"{{"@timestamp":"2023-01-01T{:02}:{:02}:{:02}.{:03}Z","agent":{{"ephemeral_id":"{:016x}","id":"agent-{host}","name":"host-{host}","type":"filebeat","version":"8.8.0"}},"aws.cloudwatch":{{"ingestion_time":"2023-01-01T00:00:00Z","log_group":"/var/log/messages","log_stream":"stream-{}"}},"cloud":{{"region":"eu-central-1"}},"data_stream":{{"dataset":"generic","namespace":"default","type":"logs"}},"event":{{"dataset":"generic","id":"{:016x}","ingested":"2023-07-20T04:20:34.181Z"}},"host":{{"name":"host-{host}"}},"input":{{"type":"aws-cloudwatch"}},"log.file.path":"/var/log/messages/{}","message":"{} kernel: [{}] audit: type={} audit({}.{}:{}): pid={} uid=0 auid=4294967295 ses=4294967295 msg='op=PAM:session_open grantors=pam_unix acct=\"root\" exe=\"/usr/sbin/crond\" res=success'","metrics":{{"size":{},"tmin":{}}},"process":{{"name":"systemd"}},"tags":["preserve_original_event"]}}"#,
                    (i / 3600) % 24, (i / 60) % 60, i % 60, rnd() % 1000, rnd(), host % 7, rnd(), host, i, rnd() % 100_000, rnd() % 2000, rnd(), rnd() % 1000, rnd() % 10_000, rnd() % 65_536, rnd() % 4000, rnd() % 1000)
            }).collect(),
        };
        let n = sources.len();
        let all: Vec<Document> = sources.iter().map(|s| { let mut d = Document::new(); d.add_keyword(1, s); d }).collect();
        let picks: Vec<Vec<DocId>> = (0..300).map(|_| { let mut v: Vec<DocId> = (0..10).map(|_| (rnd() % n as u64) as DocId).collect(); v.sort_unstable(); v.dedup(); v }).collect();
        eprintln!("{n} docs, {:.0} B/doc JSON", sources.iter().map(|s| s.len()).sum::<usize>() as f64 / n as f64);
        // kb 0: the `StoredCompression::Best` rule.
        for kb in [16usize, 240, 512, 1024, 0] {
            let (mut chunks, mut counts): (Vec<Vec<u8>>, Vec<u32>) = (Vec::new(), Vec::new());
            for d in &all {
                let full = |c: &Vec<u8>, k: u32| if kb == 0 { StoredCompression::Best.chunk_full(k as usize, c.len()) } else { k as usize >= BEST_CHUNK_DOCS || c.len() >= kb << 10 };
                if chunks.is_empty() || full(chunks.last().unwrap(), *counts.last().unwrap()) {
                    chunks.push(Vec::new());
                    counts.push(0);
                }
                let mut one = Vec::new();
                encode_doc_compact(d.fields.iter(), &mut one);
                write_vint(one.len() as u64, chunks.last_mut().unwrap());
                chunks.last_mut().unwrap().extend_from_slice(&one);
                *counts.last_mut().unwrap() += 1;
            }
            let compact = vec![true; chunks.len()];
            let bytes = encode_store_chunks_compressed(&chunks, &counts, &compact, kb != 16);
            let r = StoreReader::open(&bytes).unwrap();
            let t = Instant::now();
            for p in &picks { r.get(p[0]).unwrap().unwrap(); }
            let one = t.elapsed().as_secs_f64() * 1e6 / picks.len() as f64;
            let t = Instant::now();
            for p in &picks { let mut k = 0; r.get_many(p, |_, _| k += 1).unwrap(); assert_eq!(k, p.len()); }
            let ten = t.elapsed().as_secs_f64() * 1e6 / picks.len() as f64;
            let t = Instant::now();
            for p in &picks { let s = p[0].min(n as u32 - 10); let ids: Vec<DocId> = (s..s + 10).collect(); r.get_many(&ids, |_, _| ()).unwrap(); }
            let seq = t.elapsed().as_secs_f64() * 1e6 / picks.len() as f64;
            eprintln!("  {:>8} chunks ({}): {:7.1} B/doc stored, {:6} chunks; fetch 1 random {:7.1} us, 10 random {:7.1} us, 10 consecutive {:6.1} us",
                if kb == 0 { "Best".to_string() } else { format!("{kb} KB") }, if kb != 16 { "zstd" } else { "lz4" }, bytes.len() as f64 / n as f64, chunks.len(), one, ten, seq);
        }
    }

    /// `cargo test -p rustic-core --release --lib store_chunk_layout_microbench -- --ignored --nocapture`
    /// (`STORE_BENCH_DOCS`, default 400k): noaa-shaped docs (`_source`
    /// JSON plus a 20-byte `_id`, compact-encoded) in single-block LZ4
    /// chunks of 16/32/64 KB, and in Lucene `BEST_SPEED`-style chunks
    /// (sub-blocks compressed behind a preset dictionary of the chunk's
    /// first bytes: a fetch decompresses the dictionary and the doc's
    /// sub-blocks only). Prints stored bytes/doc and the time to fetch and
    /// decode 10, 100 and 10,000 random docs (ascending ids, the last
    /// decoded block kept as the segment's chunk cache does), best of 5.
    #[test]
    #[ignore]
    fn store_chunk_layout_microbench() {
        let n: usize = std::env::var("STORE_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000);
        let mut seed = 42u64;
        let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); seed >> 33 };
        const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let docs: Vec<Vec<u8>> = (0..n).map(|i| {
            let st = rnd() % 5000;
            let (lat, lon) = (25.0 + (st % 97) as f64 * 0.25, -125.0 + (st % 233) as f64 * 0.25);
            let src = format!(
                r#"{{"date":"2016-{:02}-{:02}T00:00:00","TMAX":{:.1},"TMIN":{:.1},"PRCP":{:.1},"station":{{"elevation":{:.1},"name":"STATION NUMBER {st}","country":"United States","gsn_flag":"GSN","country_code":"US","location":{{"lat":{lat:.4},"lon":{lon:.4}}},"id":"USW000{st:05}"}}}}"#,
                1 + (i / 33_000) % 12, 1 + (i / 1_100) % 28, (rnd() % 400) as f64 / 10.0, (rnd() % 300) as f64 / 10.0 - 10.0, (rnd() % 90) as f64 / 10.0, (rnd() % 30_000) as f64 / 10.0);
            let id: String = (0..20).map(|_| ALPHA[(rnd() % 64) as usize] as char).collect();
            let mut d = Document::new();
            d.fields.push((0, crate::types::FieldValue::Text(src)));
            d.fields.push((1, crate::types::FieldValue::Keyword(id)));
            let mut b = Vec::new();
            encode_doc_compact(d.fields.iter(), &mut b);
            b
        }).collect();
        let raw_total: usize = docs.iter().map(|d| d.len() + 1).sum();
        // A chunk: first doc, per-doc ranges in the raw chunk, its dictionary
        // block and sub-blocks (raw range, compressed bytes).
        struct Chunk { first: usize, ranges: Vec<(usize, usize)>, dict: (usize, Vec<u8>), subs: Vec<(usize, usize, Vec<u8>)> }
        // (label, chunk bytes, sub-block bytes or 0 for one block, dictionary bytes)
        let layouts: [(&str, usize, usize, usize); 6] = [
            ("16K block (now)", 16 << 10, 0, 0),
            ("32K block", 32 << 10, 0, 0),
            ("64K block", 64 << 10, 0, 0),
            ("60K = 4K dict + 8K subs", 60 << 10, 8 << 10, 4 << 10),
            ("80K = 4K dict + 8K subs", 80 << 10, 8 << 10, 4 << 10),
            ("128K = 8K dict + 16K subs", 128 << 10, 16 << 10, 8 << 10),
        ];
        let fetch_sets: Vec<Vec<usize>> = [10usize, 100, 10_000].iter().map(|&k| {
            let mut ids: Vec<usize> = (0..k).map(|_| (rnd() as usize) % n).collect();
            ids.sort_unstable();
            ids
        }).collect();
        eprintln!("{n} docs, {:.1} raw B/doc", raw_total as f64 / n as f64);
        for (label, chunk_bytes, sub, dict) in layouts {
            let mut chunks: Vec<Chunk> = Vec::new();
            let mut i = 0;
            while i < n {
                let first = i;
                let mut raw = Vec::new();
                let mut ranges = Vec::new();
                while i < n && raw.len() < chunk_bytes && (sub > 0 || i - first < CHUNK_DOCS) {
                    write_vint(docs[i].len() as u64, &mut raw);
                    ranges.push((raw.len(), raw.len() + docs[i].len()));
                    raw.extend_from_slice(&docs[i]);
                    i += 1;
                }
                let c = if sub == 0 {
                    Chunk { first, ranges, dict: (raw.len(), lz4_flex::compress(&raw)), subs: Vec::new() }
                } else {
                    let d = dict.min(raw.len());
                    let subs = (d..raw.len()).step_by(sub).map(|a| {
                        let b = (a + sub).min(raw.len());
                        (a, b, lz4_flex::block::compress_with_dict(&raw[a..b], &raw[..d]))
                    }).collect();
                    Chunk { first, ranges, dict: (d, lz4_flex::compress(&raw[..d])), subs }
                };
                chunks.push(c);
            }
            // Payload plus a 4-byte size per block and ~6 table bytes per chunk.
            let stored: usize = chunks.iter().map(|c| c.dict.1.len() + 4 + c.subs.iter().map(|s| s.2.len() + 4).sum::<usize>() + 6).sum();
            let fetch = |ids: &[usize]| -> usize {
                let mut cache: Option<(usize, usize, Vec<u8>, usize)> = None; // chunk, block, bytes, raw start
                let mut dict_cache: Option<(usize, Vec<u8>)> = None;
                let mut total = 0usize;
                for &doc in ids {
                    let ci = chunks.partition_point(|c| c.first <= doc) - 1;
                    let c = &chunks[ci];
                    let (a, b) = c.ranges[doc - c.first];
                    let bytes: Vec<u8> = if c.subs.is_empty() {
                        if cache.as_ref().is_none_or(|k| k.0 != ci) {
                            cache = Some((ci, 0, lz4_flex::decompress(&c.dict.1, c.dict.0).unwrap(), 0));
                        }
                        cache.as_ref().unwrap().2[a..b].to_vec()
                    } else {
                        if dict_cache.as_ref().is_none_or(|k| k.0 != ci) {
                            dict_cache = Some((ci, lz4_flex::decompress(&c.dict.1, c.dict.0).unwrap()));
                        }
                        let dict = &dict_cache.as_ref().unwrap().1;
                        let mut out = Vec::with_capacity(b - a);
                        if a < dict.len() { out.extend_from_slice(&dict[a..b.min(dict.len())]); }
                        let first_sub = c.subs.partition_point(|s| s.1 <= a);
                        for (si, (sa, sb, comp)) in c.subs.iter().enumerate().skip(first_sub) {
                            if *sa >= b { break; }
                            if cache.as_ref().is_none_or(|k| k.0 != ci || k.1 != si + 1) {
                                cache = Some((ci, si + 1, lz4_flex::block::decompress_with_dict(comp, sb - sa, dict).unwrap(), *sa));
                            }
                            let (_, _, raw, start) = cache.as_ref().unwrap();
                            out.extend_from_slice(&raw[a.max(*sa) - start..b.min(*sb) - start]);
                        }
                        out
                    };
                    let d = decode_doc_compact(&bytes).unwrap();
                    total += d.fields.len();
                }
                total
            };
            let mut line = format!("{label:<28} {:>6.1} B/doc ({:.2}x raw)", stored as f64 / n as f64, stored as f64 / raw_total as f64);
            for ids in &fetch_sets {
                let best = (0..5).map(|_| { let t = std::time::Instant::now(); std::hint::black_box(fetch(ids)); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min);
                line += &format!("  {:>5}: {:>8.1} us ({:.2} us/doc)", ids.len(), best * 1e6, best * 1e6 / ids.len() as f64);
            }
            eprintln!("{line}");
        }
    }

    #[test]
    fn empty_store() {
        let bytes = encode_store(&[]);
        let r = StoreReader::open(&bytes).unwrap();
        assert_eq!(r.num_docs(), 0);
        assert!(r.get(0).unwrap().is_none());
    }

    #[test]
    fn compresses() {
        let docs = docs(640);
        let bytes = encode_store(&docs);
        let raw: usize = docs.iter().map(|d| bincode::serialize(d).unwrap().len()).sum();
        assert!(bytes.len() < raw / 2, "{} vs {}", bytes.len(), raw);
    }
}
