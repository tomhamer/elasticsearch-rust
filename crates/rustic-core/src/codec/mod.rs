//! On-disk segment format.
//!
//! One segment is one file (`seg_XXXXXXXX.rstc`) holding four sections,
//! each a self-describing blob, plus a fixed-size footer that locates them:
//!
//! ```text
//! "RSTC" u32 version
//! [terms][postings][fast fields][store][vectors][extra]   (each padded to 4 bytes)
//! footer (108 bytes): 6 × (u64 off, u64 len), u32 num_docs, u32 crc32, "RSTC"
//! ```
//!
//! Deletions live in a sibling `seg_XXXXXXXX.del` file (raw u64 words) so
//! that deleting never rewrites the (possibly multi-GB) segment file.
//!
//! The `[vectors]` section holds only the per-field `doc -> (vector
//! segment, ordinal)` map ([`vectors::VectorMapReader`]); the vectors and
//! HNSW graphs live in independent `vec_XXXXXXXX.rvec` vector segment files
//! (see [`crate::vector::store`]).
//!
//! The `[extra]` section is a small tagged container ([`encode_extra`]):
//! the nested-document path ordinals ([`crate::nested`]) and one geo index
//! per geo-point field ([`crate::geo`]). Segments without either leave it
//! empty.

pub mod bitpack;
pub mod directory;
pub mod fastfields;
pub mod object_store;
pub mod postings;
pub mod store;
pub mod terms;
pub mod uid_column;
pub mod varint;
pub mod vectors;

use crate::error::{Result, RusticError};
use directory::FileSlice;

pub const MAGIC: &[u8; 4] = b"RSTC";
/// v3: the vectors section became a vector map (vectors moved to `.rvec`).
/// v4: postings v4 (BitPacker4x blocks, fixed skip table with block-max
/// impacts, separate positions region) and 1-byte SmallFloat norms.
/// v5: fixed-width on-disk term block index (O(1) dictionary open) and
/// per-chunk doc counts in the store (merges copy chunks verbatim).
/// v6: sixth `[extra]` section (nested block ordinals, geo indexes).
/// v7: single-valued numeric fast-field columns carry a 1-D points index
/// (docs sorted by value) after the values; v6 files (no points index)
/// are still readable.
/// v8: sparse single-valued numeric columns append a presence bitmap after
/// the points index. v6/v7 files (no bitmap, such columns read as dense)
/// are still readable; see [`LEGACY_PRESENCE_BELOW`].
/// v9: the `[extra]` section carries per-field collection statistics
/// ([`EXTRA_FIELD_STATS`]: Lucene's `docCount` and `sumTotalTermFreq`) that
/// BM25 needs for `N` and `avgdl`. v6-v8 files have none and compute them on
/// first use; see [`LEGACY_FIELD_STATS_BELOW`].
/// v10: compact sparse numeric columns with bitmap rank lookup.
/// v11: single-valued keyword columns store ordinals 1-4 bytes wide, the
/// width in the top bits of the column's doc count; single-valued numeric
/// columns may store values as narrow offsets from a base and points-index
/// docs in 3 bytes (`PACKED` block-count flag). v10 columns read unchanged.
/// v12: postings lists with norms may carry a per-block impact staircase
/// (postings `FLAG_IMPACTS`, longer skip entries), and mutable vector
/// checkpoints may carry `RMVR` v3 rows. Lists and checkpoints without
/// them read unchanged; builds before v12 refuse v12 files
/// (`Unsupported`) instead of misreading them. Written once the cluster's
/// segment format floor reaches it, see [`write_version`].
/// v13: mutable vector graph checkpoints may be format 2 (`RMVC` v2,
/// `RMVR` v4 rows): the graph plus references to durable rows (f32 in the
/// vector store, codes in code log files the commit manifest lists).
/// Segment layouts are those of v12; builds before v13 refuse v13 files.
/// The same builds introduced DiskBBQ IVF files (`ivf_*.rivc`/`.rivp`) and
/// code log chunks (`mvq_*.rmvq`); no build reading less than v13 knows
/// them ([`VECTOR_FILES_SINCE`]). Other vector files are unchanged since
/// v11: payload segments (`vec_*.rvec`) and quantized code checkpoints
/// (`RQCB` v1) keep their layout.
/// v14: single-valued keyword columns may be sparse (on-disk fast-field
/// tag 11: a presence bitmap with ranks and the present docs' ordinals,
/// see [`fastfields`]), and numeric columns may store their values as
/// indexes into a sorted table of distinct values (`TABLE` packing), and
/// a `geo_shape` value that is one point may be encoded `p|POINT (x y)`. Written once the
/// cluster's floor reaches v14; builds before v14 refuse v14 files.
/// v15: single-valued numeric fields marked
/// [`crate::types::Field::column_stored`] are left out of the stored fields
/// and restored from their column on fetch (the `[extra]` entry
/// [`EXTRA_COLUMN_STORED`] lists them); a keyword field marked
/// [`crate::types::Field::column_from_postings`] (`_id`) has no column
/// ([`EXTRA_KEYWORD_FROM_POSTINGS`]).
/// v16: term dictionaries hold tagged offsets, single-doc lists inline
/// ([`INLINE_POSTINGS_SINCE`]), and geo indexes may be compact
/// ([`EXTRA_GEO2`]); builds before v16 refuse v16 files.
/// v17: compact term dictionary entries ([`COMPACT_TERMS_SINCE`]).
/// v18: stored chunks may hold compact-encoded docs (flag
/// [`store::COMPACT_CHUNK`], see [`COMPACT_STORED_SINCE`]).
/// v19: inline single-doc lists of positional fields carry their one
/// position ([`INLINE_POSITIONS_SINCE`]).
/// v20 - v22: earlier `geo_shape` layouts of unreleased builds (no longer
/// written or read).
/// v23: `geo_shape` fields hold ES's structures: a triangle tree and
/// per-doc shape doc values instead of a keyword column (tags 9 / 10; no
/// longer read).
/// v24: the same structures, the tree's nodes counting their triangles and
/// its leaf columns bit-packed, the doc values' headers in fixed 41-byte
/// slots (tags 11 / 12; no longer read).
/// v25: the doc values' header slots compacted to 33 bytes (the extent as
/// ES's type and two longitudes) ([`EXTRA_SHAPE_TREE`], [`EXTRA_SHAPE_DV`],
/// [`SHAPE_TREE_SINCE`]).
/// v26: stored chunks may be zstd-compressed (flag [`store::ZSTD_CHUNK`],
/// `index.codec: best_compression`, see [`ZSTD_STORED_SINCE`]), and
/// multi-valued keyword columns may store bit-packed offsets and ordinals
/// ([`PACKED_MULTI_KEYWORD_SINCE`]). Builds before v26 refuse v26 files.
/// v27: a norms column whose every doc has the same byte is stored as that
/// byte (fast-field tag 12, [`CONSTANT_NORMS_SINCE`]). Builds before v27
/// refuse v27 files.
/// v28: dense integer columns may pack each 256-doc block's values as
/// bit-packed offsets from the block's minimum, in units of the column's
/// GCD, and dense keyword columns each block's ordinals the same way (tag
/// 13), as Lucene's doc values pack per block ([`BLOCK_PACKED_SINCE`]):
/// a block of one value takes no bits. Builds before v28 refuse v28 files.
/// v29: a multi-valued keyword column of feature values (`sparse_vector` /
/// `rank_features` doc values) may store a token table, a weight table and
/// a packed (token, weight) pair per value (fast-field tag 14,
/// [`FEATURES_COLUMN_SINCE`]). Builds before v29 refuse v29 files.
/// v30: a dense float column of exact decimals (every value an integer over
/// one `10^e`, `e <= 9`) may store those integers packed per block like an
/// integer column (flag `DECIMAL` with the block packing,
/// [`DECIMAL_FLOAT_SINCE`]): `scaled_float` and fixed-precision gauges.
/// Builds before v30 refuse v30 files.
/// v31: compact stored chunks flagged `UID_CHUNK` may hold the index's
/// `_id` values in their binary form (`crate::uid`, value kind
/// `COMPACT_UID`), and `_id` terms are keyed by that form
/// ([`UID_STORED_SINCE`], [`UID_TERMS_SINCE`]). Builds before v31 refuse
/// v31 files. (Stored chunks of up to 1024 docs / 64 KB and 32-term
/// dictionary blocks need no version: earlier readers take any chunk or
/// block size.)
/// v32: term dictionary blocks open with a header byte: suffixes all in
/// `0-9a-f` take 4 bits a byte, all in URL-safe base64 6 bits, and a
/// block's entry lengths may take one byte ([`PACKED_TERMS_SINCE`],
/// `terms::TermDictWriter::packed`). Builds before v32 refuse v32 files.
/// v33: single-valued keyword columns' front-coded term tables may be
/// packed (`fastfields::FCP_MARK`: per-block header, suffixes of hex /
/// base64 bytes at 4 / 6 bits, one-byte lengths; [`PACKED_KEYWORD_TABLES_SINCE`]).
/// Builds before v33 refuse v33 files.
/// v34: a synthetic-source document's empty residual may be stored as two
/// bytes, its fields told back by its columns ([`EMPTY_RESIDUAL_SINCE`]),
/// and a numeric column of few distinct values may keep its points index
/// as a doc list per value ([`VALUE_LIST_POINTS_SINCE`]); a block-packed
/// integer column's blocks may hold deltas ([`DELTA_BLOCKS_SINCE`]) and a
/// term dictionary block of same-shape hex / base64 keys their integer
/// deltas ([`INT_TERMS_SINCE`]). Builds before v34 refuse v34 files.
/// v35: a merged segment may keep its documents' `_id`s in an id column
/// ([`uid_column`], [`EXTRA_UID_COLUMN`]) instead of their stored fields
/// ([`UID_COLUMN_SINCE`]). Builds before v35 refuse v35 files.
/// v36: a block-packed keyword column (tag 13) may hold docs without a
/// value: a flagged block's all-ones offset is a missing doc
/// ([`BLOCKED_MISSING_KEYWORD_SINCE`]; a TSDB keyword absent from other
/// metricsets' series no longer takes a byte or two a doc); a compact
/// sparse numeric column may give its present docs as runs and pack its
/// integer values per block of 256 present values
/// ([`COMPACT_RUNS_SINCE`]: a TSDB metric present on some series only).
/// Builds before v36 refuse v36 files.
/// v37: a boolean column may know which docs have a value (fast-field tag
/// 15, a 0 / 1 numeric column, [`BOOL_PRESENCE_SINCE`]): a doc without
/// the field no longer reads `false`, and a synthetic source whose
/// booleans come back from their column needs no stored field ids. A
/// block-packed integer block of 1 bit may hold any two values (its
/// minimum and maximum, [`TWO_VALUE_BLOCKS_SINCE`]), and a keyword column
/// of one term every doc has stores no ordinals
/// ([`CONSTANT_KEYWORD_COLUMN_SINCE`]). Builds before v37 refuse v37
/// files.
pub const VERSION: u32 = 37;
/// First version whose boolean columns may know their docs with a value.
pub const BOOL_PRESENCE_SINCE: u32 = 37;
/// First version whose 1-bit integer blocks hold their minimum or maximum.
pub const TWO_VALUE_BLOCKS_SINCE: u32 = 37;
/// First version whose keyword columns may be one term for every doc.
pub const CONSTANT_KEYWORD_COLUMN_SINCE: u32 = 37;
/// First version whose block-packed keyword columns may have missing docs.
pub const BLOCKED_MISSING_KEYWORD_SINCE: u32 = 36;
/// First version whose compact numeric columns may hold presence runs and
/// block-packed values.
pub const COMPACT_RUNS_SINCE: u32 = 36;
/// First version whose merges may move stored `_id`s into an id column.
pub const UID_COLUMN_SINCE: u32 = 35;
/// First version storing empty synthetic residuals as a marker.
pub const EMPTY_RESIDUAL_SINCE: u32 = 34;
/// First version whose low-cardinality numeric columns keep per-value doc
/// lists as their points index.
pub const VALUE_LIST_POINTS_SINCE: u32 = 34;
/// First version whose block-packed integer columns may store blocks as
/// deltas (index-sorted timestamps).
pub const DELTA_BLOCKS_SINCE: u32 = 34;
/// First version whose term dictionary blocks may key their terms as
/// integer deltas (same-length hex / base64 keys: k-ordered ids).
pub const INT_TERMS_SINCE: u32 = 34;
/// First version whose term dictionary blocks are packed.
pub const PACKED_TERMS_SINCE: u32 = 32;
/// First version whose keyword column term tables are packed.
pub const PACKED_KEYWORD_TABLES_SINCE: u32 = 33;
/// First version whose compact stored chunks may hold binary ids (see
/// [`store::UID_CHUNK`]).
pub const UID_STORED_SINCE: u32 = 31;
/// First version whose term dictionaries key the `_id` field's terms by
/// their binary form ([`crate::uid`]).
pub const UID_TERMS_SINCE: u32 = 31;
/// First version whose float columns may be stored as block-packed
/// decimals (see [`fastfields`]).
pub const DECIMAL_FLOAT_SINCE: u32 = 30;
/// First version whose single-valued keyword columns store front-coded term
/// tables (see [`fastfields`]).
pub const FRONT_CODED_TERMS_SINCE: u32 = 30;
/// First version whose feature-value columns may be stored as token and
/// weight tables with packed pairs (see [`fastfields`]).
pub const FEATURES_COLUMN_SINCE: u32 = 29;
/// First version whose dense integer and keyword columns may be packed per
/// block (see [`fastfields`]).
pub const BLOCK_PACKED_SINCE: u32 = 28;
/// First version whose norms columns may be constant (one byte for every
/// doc), as Lucene stores a field whose docs all have the same norm.
pub const CONSTANT_NORMS_SINCE: u32 = 27;
/// First version whose stored chunks may be zstd frames
/// ([`store::ZSTD_CHUNK`]): ES `Zstd814StoredFieldsFormat`'s
/// `BEST_COMPRESSION` mode for `index.codec: best_compression`.
pub const ZSTD_STORED_SINCE: u32 = 26;
/// First version whose multi-valued keyword columns may hold their doc
/// offsets and ordinals bit-packed (Lucene `SORTED_SET` doc values: ords
/// in `ceil(log2(terms))` bits, addresses monotonic-packed) instead of four
/// bytes each; see [`fastfields`].
pub const PACKED_MULTI_KEYWORD_SINCE: u32 = 26;
/// First version whose stored fields may leave out column-backed fields.
pub const COLUMN_STORED_SINCE: u32 = 15;
/// First version whose term dictionaries may hold a single-doc postings
/// list (doc freq 1, term freq 1, no positions: `_id`) in the entry itself
/// instead of the postings section: entries encode `offset << 1`, or
/// `doc << 1 | 1` for an inline list (Lucene's pulsed singletons).
pub const INLINE_POSTINGS_SINCE: u32 = 16;
/// First version whose geo indexes may use the compact layout
/// ([`EXTRA_GEO2`]: dense doc-order values, packed tree).
pub const GEO_COMPACT_SINCE: u32 = 16;
/// First version whose term dictionaries use compact entries
/// ([`terms::TermDictWriter::compact`]: no in-block copy of a block's first
/// key, inline docs and postings offsets as deltas within the block, an
/// inline list without its implied counts).
pub const COMPACT_TERMS_SINCE: u32 = 17;
/// First version whose stored chunks may use the compact doc encoding
/// ([`store::encode_doc_compact`]: varint lengths and integers, one-byte
/// kinds) instead of bincode.
pub const COMPACT_STORED_SINCE: u32 = 18;
/// First version whose term dictionaries keep single-doc lists of
/// positional (text) fields inline with their one position, as they keep
/// those without positions from [`INLINE_POSTINGS_SINCE`].
pub const INLINE_POSITIONS_SINCE: u32 = 19;
/// First version whose sealed segments may leave out the keyword column of
/// a field whose values follow from its postings (`_id`), listed by
/// [`EXTRA_KEYWORD_FROM_POSTINGS`].
pub const KEYWORD_FROM_POSTINGS_SINCE: u32 = 15;
/// First version whose keyword columns may be sparse.
pub const SPARSE_KEYWORD_SINCE: u32 = 14;
/// First version whose numeric columns may be table-packed.
pub const NUMERIC_TABLES_SINCE: u32 = 14;
/// First version whose segments hold `geo_shape` fields as ES does: the
/// triangle tree ([`crate::fieldtypes::shape_bkd`], [`EXTRA_SHAPE_TREE`])
/// and the shape doc values ([`crate::fieldtypes::shape_dv`],
/// [`EXTRA_SHAPE_DV`]). Shape data is written only at this version or
/// later (a lower cluster floor refuses `geo_shape` documents), and older
/// segments holding shape data are refused when opened.
pub const SHAPE_TREE_SINCE: u32 = 25;
/// First version whose postings may carry impact staircases.
pub const IMPACTS_SINCE: u32 = 12;
/// First version whose mutable vector checkpoints may carry `RMVR` v3 rows.
pub const RMVR3_SINCE: u32 = 12;
/// First version whose mutable vector checkpoints may be format 2 (code
/// log references, `RMVR` v4).
pub const RMVC2_SINCE: u32 = 13;
/// First segment format whose readers know DiskBBQ IVF files and mutable
/// graph code log chunks. IVF files are written whatever the cluster's floor
/// (a `bbq_disk` field has no other layout in this build), so a copy that
/// holds them is never shipped to a node reading less (file recovery checks
/// the target's read maximum, and an upgraded master's allocation keeps
/// `bbq_disk` copies off such nodes: the cluster crate's
/// `compatibility::index_required_reader`, pinned equal). Code log chunks are only referenced by format 2 graph
/// checkpoints ([`RMVC2_SINCE`]).
pub const VECTOR_FILES_SINCE: u32 = 13;
/// Oldest version this build can write (the layout of v11 is a subset).
pub const MIN_WRITE_VERSION: u32 = 11;
/// Version new files are written in before this process learns the
/// cluster's segment format floor ([`apply_cluster_segment_format`]): v11,
/// which every build of this format line reads (nodes that advertise no
/// read maximum count as v11 readers). A node starts from the floor its
/// local gateway persisted, so this only applies to a node that never
/// belonged to a cluster that negotiated a higher floor.
pub const DEFAULT_WRITE_VERSION: u32 = 11;

/// The cluster's negotiated segment format floor as this process last
/// applied it (0: none yet). The elected master raises the floor to the
/// lowest read maximum over every member once that exceeds it, and never
/// lowers it; a node below the floor cannot join, so every member reads
/// every format up to the floor.
static CLUSTER_FLOOR: FormatFloor = FormatFloor::new();

/// A segment format floor that only rises (0: none known yet).
pub struct FormatFloor(std::sync::atomic::AtomicU32);

impl FormatFloor {
    pub const fn new() -> Self {
        FormatFloor(std::sync::atomic::AtomicU32::new(0))
    }
    /// Raise to `floor` (never lowers; `0` is ignored).
    pub fn raise(&self, floor: u32) {
        if floor != 0 {
            self.0.fetch_max(floor, std::sync::atomic::Ordering::AcqRel);
        }
    }
    /// The floor, 0 when none is known.
    pub fn get(&self) -> u32 {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Default for FormatFloor {
    fn default() -> Self {
        Self::new()
    }
}

/// The on-disk format generation new files are written in (segment
/// headers, postings, mutable vector checkpoints): the cluster's segment
/// format floor ([`apply_cluster_segment_format`], else
/// [`DEFAULT_WRITE_VERSION`]) capped by this build's [`VERSION`] and by
/// `RUSTIC_SEGMENT_WRITE_VERSION` when set ([`write_version_cap`]).
///
/// It rises at runtime when the cluster's floor does. A writer reads it
/// once at the start of each file (a flush seal, a merge, a checkpoint)
/// and passes that value down, so a file's header and its contents (the
/// postings behind a segment header, the rows of a checkpoint) never
/// disagree even when the floor rises mid-write.
/// Tests may override it per thread ([`TEST_WRITE_VERSION`]).
pub fn write_version() -> u32 {
    #[cfg(test)]
    {
        let forced = TEST_WRITE_VERSION.with(|v| v.get());
        if forced != 0 {
            // As the setting is read: what this build writes, else the default.
            return if (MIN_WRITE_VERSION..=VERSION).contains(&forced) { forced } else { DEFAULT_WRITE_VERSION };
        }
    }
    effective_write_version(CLUSTER_FLOOR.get(), write_version_cap())
}

/// The format of a file holding `geo_shape` data: [`write_version`] once
/// the process knows a cluster floor (or a cap or test override is set);
/// without one (a standalone engine, whose files no older node reads) at
/// least [`SHAPE_TREE_SINCE`], the first format holding shapes. Below it,
/// shape data is refused.
pub fn shape_write_version() -> u32 {
    let v = write_version();
    #[cfg(test)]
    {
        if TEST_WRITE_VERSION.with(|t| t.get()) != 0 {
            return v;
        }
    }
    if CLUSTER_FLOOR.get() == 0 && write_version_cap().is_none() {
        v.max(SHAPE_TREE_SINCE)
    } else {
        v
    }
}

/// [`write_version`] for a cluster floor (`0`: unknown) and an optional
/// configured cap: never above the floor, this build's [`VERSION`] or the
/// cap, never below [`MIN_WRITE_VERSION`].
pub fn effective_write_version(floor: u32, cap: Option<u32>) -> u32 {
    let floor = if floor == 0 { DEFAULT_WRITE_VERSION } else { floor };
    floor.min(VERSION).min(cap.unwrap_or(VERSION)).max(MIN_WRITE_VERSION)
}

/// Record the cluster's segment format floor (from an applied cluster state
/// or, at startup, the committed floor the node last recorded). The process
/// value only rises: the cluster floor never lowers, and an older state
/// applied late must not undo a newer one. Returns the resulting
/// [`write_version`].
pub fn apply_cluster_segment_format(floor: u32) -> u32 {
    CLUSTER_FLOOR.raise(floor);
    write_version()
}

/// The cluster floor this process has applied, if any.
pub fn cluster_segment_format() -> Option<u32> {
    match CLUSTER_FLOOR.get() {
        0 => None,
        v => Some(v),
    }
}

/// `RUSTIC_SEGMENT_WRITE_VERSION`, read once per process: an upper bound on
/// [`write_version`] (testing, or holding a cluster at an older format in
/// an emergency), never a way to write above the cluster floor. A value
/// this build does not write caps at [`DEFAULT_WRITE_VERSION`] with a
/// warning (nodes refuse to start with one, see the node's writer setting
/// validation).
pub fn write_version_cap() -> Option<u32> {
    static V: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_write_version_cap(std::env::var("RUSTIC_SEGMENT_WRITE_VERSION").ok().as_deref()))
}

/// [`write_version_cap`]'s reading of `RUSTIC_SEGMENT_WRITE_VERSION`.
fn parse_write_version_cap(v: Option<&str>) -> Option<u32> {
    let raw = v?;
    match raw.trim().parse::<u32>() {
        Ok(v) if (MIN_WRITE_VERSION..=VERSION).contains(&v) => Some(v),
        _ => {
            tracing::warn!(
                "RUSTIC_SEGMENT_WRITE_VERSION={raw:?} is not a segment format this build writes ({MIN_WRITE_VERSION}..={VERSION}); capping writes at v{DEFAULT_WRITE_VERSION}"
            );
            Some(DEFAULT_WRITE_VERSION)
        }
    }
}


#[cfg(test)]
thread_local! {
    /// Test override of [`write_version`] on this thread (0: none). Work
    /// handed to a pool follows it ([`crate::on_pool`]); a seal's or a
    /// merge's parallel postings encoding uses the version its writer
    /// captured, whichever thread runs it.
    pub(crate) static TEST_WRITE_VERSION: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
/// Segments older than this version carry no per-field collection statistics
/// ([`EXTRA_FIELD_STATS`]); they are derived from the term dictionary,
/// norms and fast-field presence on first use.
pub const LEGACY_FIELD_STATS_BELOW: u32 = 9;
/// Segments older than this version have no fast-field presence bitmaps:
/// a doc without a value in a sparse numeric column reads as `0` there.
pub const LEGACY_PRESENCE_BELOW: u32 = 8;
/// Oldest segment version this build reads.
pub const MIN_READ_VERSION: u32 = 6;

/// Whether this build can read a segment file with the given format version.
///
/// Recovery should call this before opening a shard's segments so a node that
/// is too old for the data fails loudly here instead of deep inside segment
/// open (the same check runs in [`open_segment_file`]).
pub fn can_read_segment_version(version: u32) -> bool {
    (MIN_READ_VERSION..=VERSION).contains(&version)
}

/// Whether this build can read a vector segment file with the given format
/// version (see [`vectors::RVEC_VERSION`]).
pub fn can_read_vector_segment_version(version: u32) -> bool {
    (1..=vectors::RVEC_VERSION).contains(&version)
}
const NUM_SECTIONS: usize = 6;
const FOOTER_LEN: usize = NUM_SECTIONS * 16 + 4 + 4 + 4;

pub fn segment_file_name(id: u32) -> String {
    format!("seg_{id:08}.rstc")
}

pub fn deletes_file_name(id: u32) -> String {
    format!("seg_{id:08}.del")
}

/// File name of vector segment `id`.
pub fn vector_segment_file_name(id: u32) -> String {
    format!("vec_{id:08}.rvec")
}

/// Inverse of [`vector_segment_file_name`].
pub fn parse_vector_segment_file_name(name: &str) -> Option<u32> {
    name.strip_prefix("vec_")?.strip_suffix(".rvec")?.parse().ok()
}

/// CRC-32 (IEEE, reflected 0xEDB88320) over the body for corruption
/// detection. Uses the ARMv8 CRC32 instructions when available (the whole
/// segment body is checksummed on every flush, so a bitwise loop was ~20% of
/// indexing time), else slicing-by-8 tables. Same value either way.
pub fn crc32(data: &[u8]) -> u32 {
    !crc32_update(0xFFFF_FFFF, data)
}

/// Streaming form of [`crc32`]: start from `0xFFFF_FFFF`, feed the pieces
/// in order, invert at the end.
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("crc") {
            // SAFETY: the `crc` feature was detected at runtime.
            return unsafe { crc32_arm(crc, data) };
        }
    }
    crc32_sliced(crc, data)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32_arm(mut crc: u32, data: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32b, __crc32d};
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        crc = __crc32d(crc, u64::from_le_bytes(c.try_into().unwrap()));
    }
    for &b in chunks.remainder() {
        crc = __crc32b(crc, b);
    }
    crc
}

fn crc32_tables() -> &'static [[u32; 256]; 8] {
    static T: std::sync::OnceLock<Box<[[u32; 256]; 8]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = Box::new([[0u32; 256]; 8]);
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
            }
            t[0][i as usize] = c;
        }
        for i in 0..256 {
            for k in 1..8 {
                let prev = t[k - 1][i];
                t[k][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            }
        }
        t
    })
}

fn crc32_sliced(mut crc: u32, data: &[u8]) -> u32 {
    let t = crc32_tables();
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let lo = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) ^ crc;
        let hi = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
        crc = t[7][(lo & 0xFF) as usize]
            ^ t[6][((lo >> 8) & 0xFF) as usize]
            ^ t[5][((lo >> 16) & 0xFF) as usize]
            ^ t[4][(lo >> 24) as usize]
            ^ t[3][(hi & 0xFF) as usize]
            ^ t[2][((hi >> 8) & 0xFF) as usize]
            ^ t[1][((hi >> 16) & 0xFF) as usize]
            ^ t[0][(hi >> 24) as usize];
    }
    for &b in chunks.remainder() {
        crc = (crc >> 8) ^ t[0][((crc ^ b as u32) & 0xFF) as usize];
    }
    crc
}

/// The framing of a segment file around its sections: the file is
/// `header, terms, pad[0], postings, pad[1], fast, pad[2], store, pad[3],
/// vectors, footer` (see [`SegmentFileFrame::parts`]), so a writer can
/// stream the sections without first copying them into one buffer.
pub struct SegmentFileFrame {
    header: [u8; 8],
    /// Zero padding (0..=3 bytes) before sections 1..5.
    pads: [usize; NUM_SECTIONS - 1],
    footer: [u8; FOOTER_LEN],
}

const ZEROS: [u8; 4] = [0; 4];

impl SegmentFileFrame {
    /// Frame `sections` (terms, postings, fast, store, vectors, extra), computing
    /// the body checksum over them in place.
    pub fn new(sections: [&[u8]; NUM_SECTIONS], num_docs: u32) -> Self {
        Self::new_versioned(sections, num_docs, write_version())
    }

    /// [`SegmentFileFrame::new`] for a segment encoded as format `version`
    /// (the writer's snapshot of [`write_version`], so header and postings
    /// agree whichever threads encoded them).
    pub fn new_versioned(sections: [&[u8]; NUM_SECTIONS], num_docs: u32, version: u32) -> Self {
        Self::from_pieces_versioned(sections.each_ref().map(std::slice::from_ref), num_docs, version)
    }

    /// [`SegmentFileFrame::new`] for sections given as lists of pieces
    /// (each section is the concatenation of its pieces), so a merge can
    /// stream copied chunks straight from the input mappings.
    pub fn from_pieces(sections: [&[&[u8]]; NUM_SECTIONS], num_docs: u32) -> Self {
        Self::from_pieces_versioned(sections, num_docs, write_version())
    }

    fn from_pieces_versioned(sections: [&[&[u8]]; NUM_SECTIONS], num_docs: u32, version: u32) -> Self {
        let mut header = [0u8; 8];
        header[..4].copy_from_slice(MAGIC);
        header[4..].copy_from_slice(&version.to_le_bytes());
        let mut crc = crc32_update(0xFFFF_FFFF, &header);
        let mut pos = 8usize;
        let mut pads = [0usize; NUM_SECTIONS - 1];
        let mut offsets = [(0u64, 0u64); NUM_SECTIONS];
        for (i, pieces) in sections.iter().enumerate() {
            // 4-byte align every section so f32/u32 columns can be read in place.
            if i > 0 {
                let pad = (4 - pos % 4) % 4;
                pads[i - 1] = pad;
                crc = crc32_update(crc, &ZEROS[..pad]);
                pos += pad;
            }
            let len: usize = pieces.iter().map(|p| p.len()).sum();
            offsets[i] = (pos as u64, len as u64);
            for p in pieces.iter() {
                crc = crc32_update(crc, p);
            }
            pos += len;
        }
        let mut footer = [0u8; FOOTER_LEN];
        for (i, (off, len)) in offsets.iter().enumerate() {
            footer[i * 16..i * 16 + 8].copy_from_slice(&off.to_le_bytes());
            footer[i * 16 + 8..i * 16 + 16].copy_from_slice(&len.to_le_bytes());
        }
        let base = NUM_SECTIONS * 16;
        footer[base..base + 4].copy_from_slice(&num_docs.to_le_bytes());
        footer[base + 4..base + 8].copy_from_slice(&(!crc).to_le_bytes());
        footer[base + 8..].copy_from_slice(MAGIC);
        Self { header, pads, footer }
    }

    /// The file's pieces in order (concatenated, they are the file).
    pub fn parts<'a>(&'a self, sections: [&'a [u8]; NUM_SECTIONS]) -> Vec<&'a [u8]> {
        self.parts_from_pieces(sections.each_ref().map(std::slice::from_ref))
    }

    /// [`SegmentFileFrame::parts`] for sections given as pieces.
    pub fn parts_from_pieces<'a>(&'a self, sections: [&[&'a [u8]]; NUM_SECTIONS]) -> Vec<&'a [u8]> {
        let n: usize = sections.iter().map(|s| s.len()).sum();
        let mut v: Vec<&[u8]> = Vec::with_capacity(n + NUM_SECTIONS + 1);
        v.push(&self.header);
        for (i, s) in sections.into_iter().enumerate() {
            if i > 0 && self.pads[i - 1] > 0 {
                v.push(&ZEROS[..self.pads[i - 1]]);
            }
            v.extend_from_slice(s);
        }
        v.push(&self.footer);
        v
    }
}

/// Streams a segment file whose sections may be written in any order (the
/// footer records where each one landed), checksumming on the way: a merge
/// writes the large copied store while postings are still being encoded.
pub struct SegmentFileWriter<'a> {
    out: Box<dyn directory::OutputFile + 'a>,
    crc: u32,
    pos: u64,
    offsets: [(u64, u64); NUM_SECTIONS],
    open: Option<usize>,
}

/// Section indexes for [`SegmentFileWriter::begin_section`].
pub const SECTION_TERMS: usize = 0;
pub const SECTION_POSTINGS: usize = 1;
pub const SECTION_FAST: usize = 2;
pub const SECTION_STORE: usize = 3;
pub const SECTION_VECTORS: usize = 4;
pub const SECTION_EXTRA: usize = 5;

/// [extra] entry tag: nested path ordinals (field 0).
pub const EXTRA_NESTED: u8 = 1;
/// [extra] entry tag (v15+, empty payload): `field`'s stored value is its
/// i64 column's, restored on fetch when the stored doc lacks it.
pub const EXTRA_COLUMN_STORED: u8 = 6;
/// [extra] entry tag (v15+, empty payload): `field` has no keyword column;
/// its values follow from its postings and stored fields.
pub const EXTRA_KEYWORD_FROM_POSTINGS: u8 = 7;
/// [extra] entry tag (v16+): a geo index in the compact layout
/// ([`crate::geo::GeoIndex::encode_compact`]).
pub const EXTRA_GEO2: u8 = 8;
/// [extra] entry tag: geo index of a field.
pub const EXTRA_GEO: u8 = 2;
/// [extra] entry tag: collection statistics of an indexed text / keyword
/// field: `u64 doc_count, u64 sum_total_term_freq` (little endian).
pub const EXTRA_FIELD_STATS: u8 = 3;
/// [extra] entry tag of the retired `geo_shape` triangle index (v13 - v22):
/// a segment holding it is refused (reindex).
pub const EXTRA_SHAPE_LEGACY: u8 = 4;
/// [extra] entry tags of the v23 `geo_shape` tree and doc values: a
/// segment holding them is refused (reindex).
pub const EXTRA_SHAPE_TREE_V23: u8 = 9;
pub const EXTRA_SHAPE_DV_V23: u8 = 10;
/// [extra] entry tags of the v24 `geo_shape` tree and doc values: refused.
pub const EXTRA_SHAPE_TREE_V24: u8 = 11;
pub const EXTRA_SHAPE_DV_V24: u8 = 12;
/// [extra] entry tag (v25+): one part of a `geo_shape` field's triangle
/// tree ([`crate::fieldtypes::shape_bkd`]); a field may have several.
pub const EXTRA_SHAPE_TREE: u8 = 13;
/// [extra] entry tag (v25+): one part of a `geo_shape` field's doc values
/// ([`crate::fieldtypes::shape_dv::ShapeDvColumn`]).
pub const EXTRA_SHAPE_DV: u8 = 14;
/// `[extra]` entry of a synthetic id field (field = the id field; payload:
/// its `_tsid`, timestamp and routing-hash column ids, see
/// [`crate::synthetic_id`]): the segment holds no postings or stored values
/// of that field.
pub const EXTRA_SYNTHETIC_ID: u8 = 15;
/// [extra] entry tag (v35+): the documents' `_id`s of `field` in doc order
/// ([`uid_column`]); their stored documents hold no `_id`, which comes back
/// from this column on fetch.
pub const EXTRA_UID_COLUMN: u8 = 16;

/// Encode the `[extra]` section: `u32 n, n × (u8 tag, u16 field, u32 len,
/// bytes)`; empty when there are no entries.
pub fn encode_extra(entries: &[(u8, u16, Vec<u8>)]) -> Vec<u8> {
    if entries.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (tag, field, bytes) in entries {
        out.push(*tag);
        out.extend_from_slice(&field.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Inverse of [`encode_extra`] (borrowing the entry bytes).
pub fn decode_extra(data: &[u8]) -> Result<Vec<(u8, u16, &[u8])>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let err = || RusticError::Corrupted("extra section truncated".into());
    let n = u32::from_le_bytes(data.get(..4).ok_or_else(err)?.try_into().unwrap()) as usize;
    let mut pos = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let h = data.get(pos..pos + 7).ok_or_else(err)?;
        let tag = h[0];
        let field = u16::from_le_bytes([h[1], h[2]]);
        let len = u32::from_le_bytes([h[3], h[4], h[5], h[6]]) as usize;
        pos += 7;
        out.push((tag, field, data.get(pos..pos + len).ok_or_else(err)?));
        pos += len;
    }
    Ok(out)
}

impl<'a> SegmentFileWriter<'a> {
    pub fn new(out: Box<dyn directory::OutputFile + 'a>) -> Result<Self> {
        Self::with_version(out, write_version())
    }

    /// [`SegmentFileWriter::new`] for a segment encoded as format `version`.
    pub fn with_version(mut out: Box<dyn directory::OutputFile + 'a>, version: u32) -> Result<Self> {
        let mut header = [0u8; 8];
        header[..4].copy_from_slice(MAGIC);
        header[4..].copy_from_slice(&version.to_le_bytes());
        out.write_all(&header)?;
        Ok(Self { out, crc: crc32_update(0xFFFF_FFFF, &header), pos: 8, offsets: [(0, 0); NUM_SECTIONS], open: None })
    }

    fn put(&mut self, data: &[u8]) -> Result<()> {
        self.crc = crc32_update(self.crc, data);
        self.pos += data.len() as u64;
        self.out.write_all(data)
    }

    /// Start section `i` (4-byte aligned); it ends at the next
    /// `begin_section` or `finish`.
    pub fn begin_section(&mut self, i: usize) -> Result<()> {
        self.end_section();
        let pad = ((4 - self.pos % 4) % 4) as usize;
        self.put(&ZEROS[..pad])?;
        self.offsets[i].0 = self.pos;
        self.open = Some(i);
        Ok(())
    }

    /// Append bytes to the open section.
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        debug_assert!(self.open.is_some());
        self.put(data)
    }

    fn end_section(&mut self) {
        if let Some(i) = self.open.take() {
            self.offsets[i].1 = self.pos - self.offsets[i].0;
        }
    }

    /// Write the footer and publish the file. Sections never begun are empty.
    pub fn finish(mut self, num_docs: u32) -> Result<()> {
        self.end_section();
        let mut footer = [0u8; FOOTER_LEN];
        for (i, (off, len)) in self.offsets.iter().enumerate() {
            let off = if *len == 0 && *off == 0 { self.pos } else { *off };
            footer[i * 16..i * 16 + 8].copy_from_slice(&off.to_le_bytes());
            footer[i * 16 + 8..i * 16 + 16].copy_from_slice(&len.to_le_bytes());
        }
        let base = NUM_SECTIONS * 16;
        footer[base..base + 4].copy_from_slice(&num_docs.to_le_bytes());
        footer[base + 4..base + 8].copy_from_slice(&(!self.crc).to_le_bytes());
        footer[base + 8..].copy_from_slice(MAGIC);
        self.out.write_all(&footer)?;
        self.out.finish()
    }
}

/// Assemble a segment file from its encoded sections.
pub fn write_segment_file(
    terms: &[u8],
    postings: &[u8],
    fast: &[u8],
    store: &[u8],
    vectors: &[u8],
    num_docs: u32,
) -> Vec<u8> {
    write_segment_file_with_extra(terms, postings, fast, store, vectors, &[], num_docs)
}

/// [`write_segment_file`] with an `[extra]` section.
pub fn write_segment_file_with_extra(
    terms: &[u8],
    postings: &[u8],
    fast: &[u8],
    store: &[u8],
    vectors: &[u8],
    extra: &[u8],
    num_docs: u32,
) -> Vec<u8> {
    let sections = [terms, postings, fast, store, vectors, extra];
    let frame = SegmentFileFrame::new(sections, num_docs);
    frame.parts(sections).concat()
}

/// The sections of an opened segment file, as sub-slices of the mapping.
pub struct SegmentFile {
    pub terms: FileSlice,
    pub postings: FileSlice,
    pub fast: FileSlice,
    pub store: FileSlice,
    pub vectors: FileSlice,
    pub extra: FileSlice,
    pub num_docs: u32,
    /// The format version in the file header.
    pub version: u32,
}

/// Parse the footer and split a segment file into its sections.
///
/// `verify_crc` recomputes the checksum over the whole body; this is an
/// O(file) pass, so callers only do it on open/CheckIndex, not per query.
pub fn open_segment_file(data: FileSlice, verify_crc: bool) -> Result<SegmentFile> {
    let bytes = data.as_slice();
    if bytes.len() < 8 + FOOTER_LEN || &bytes[..4] != MAGIC {
        return Err(RusticError::Corrupted("not a segment file".into()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if !can_read_segment_version(version) {
        return Err(RusticError::Unsupported(format!(
            "segment version {version} (this build reads {MIN_READ_VERSION}..={VERSION})"
        )));
    }
    let footer = &bytes[bytes.len() - FOOTER_LEN..];
    if &footer[FOOTER_LEN - 4..] != MAGIC {
        return Err(RusticError::Corrupted("bad segment footer".into()));
    }
    let mut sections = [(0usize, 0usize); NUM_SECTIONS];
    for (i, s) in sections.iter_mut().enumerate() {
        let off = u64::from_le_bytes(footer[i * 16..i * 16 + 8].try_into().unwrap()) as usize;
        let len = u64::from_le_bytes(footer[i * 16 + 8..i * 16 + 16].try_into().unwrap()) as usize;
        if off + len > bytes.len() - FOOTER_LEN {
            return Err(RusticError::Corrupted("segment section out of range".into()));
        }
        *s = (off, len);
    }
    let base = NUM_SECTIONS * 16;
    let num_docs = u32::from_le_bytes(footer[base..base + 4].try_into().unwrap());
    let stored_crc = u32::from_le_bytes(footer[base + 4..base + 8].try_into().unwrap());
    if verify_crc && crc32(&bytes[..bytes.len() - FOOTER_LEN]) != stored_crc {
        return Err(RusticError::Corrupted("segment checksum mismatch".into()));
    }
    let slice = |(off, len): (usize, usize)| data.slice(off..off + len);
    Ok(SegmentFile {
        terms: slice(sections[0]),
        postings: slice(sections[1]),
        fast: slice(sections[2]),
        store: slice(sections[3]),
        vectors: slice(sections[4]),
        extra: slice(sections[5]),
        num_docs,
        version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_file_roundtrip() {
        let bytes = write_segment_file(b"TERMS", b"POSTINGS", b"", b"STORE", b"VEC", 42);
        let f = open_segment_file(FileSlice::from_vec(bytes.clone()), true).unwrap();
        assert_eq!(f.terms.as_slice(), b"TERMS");
        assert_eq!(f.postings.as_slice(), b"POSTINGS");
        assert!(f.fast.is_empty());
        assert_eq!(f.store.as_slice(), b"STORE");
        assert_eq!(f.vectors.as_slice(), b"VEC");
        assert_eq!(f.num_docs, 42);

        let mut corrupt = bytes.clone();
        corrupt[10] ^= 0xFF;
        assert!(open_segment_file(FileSlice::from_vec(corrupt.clone()), true).is_err());
        // Without verification the structural parse still succeeds.
        assert!(open_segment_file(FileSlice::from_vec(corrupt), false).is_ok());
        assert!(open_segment_file(FileSlice::from_vec(b"junk".to_vec()), false).is_err());
    }

    #[test]
    fn version_readability_checks() {
        for v in MIN_READ_VERSION..=VERSION {
            assert!(can_read_segment_version(v), "v{v}");
        }
        assert!(!can_read_segment_version(MIN_READ_VERSION - 1));
        assert!(!can_read_segment_version(VERSION + 1));
        assert!(MIN_WRITE_VERSION < IMPACTS_SINCE && IMPACTS_SINCE <= VERSION);
        assert!(can_read_segment_version(DEFAULT_WRITE_VERSION) && DEFAULT_WRITE_VERSION < IMPACTS_SINCE);
        // Unset, valid, and invalid settings (the last cap at v11).
        for (v, want) in [(None, None), (Some("11"), Some(11)), (Some(" 12 "), Some(12)), (Some("13"), Some(13)), (Some("14"), Some(14)), (Some("15"), Some(15)), (Some("16"), Some(16)), (Some("17"), Some(17)), (Some("18"), Some(18)), (Some("19"), Some(19)), (Some("20"), Some(20)), (Some("21"), Some(21)), (Some("22"), Some(22)), (Some("23"), Some(23)), (Some("24"), Some(24)), (Some("25"), Some(25)), (Some("26"), Some(26)), (Some("27"), Some(27)), (Some("28"), Some(28)), (Some("29"), Some(29)), (Some("30"), Some(30)), (Some("31"), Some(31)), (Some("32"), Some(32)), (Some("33"), Some(33)), (Some("34"), Some(34)), (Some("35"), Some(35)), (Some("36"), Some(36)), (Some("37"), Some(37)), (Some("38"), Some(11)), (Some("10"), Some(11)), (Some("v12"), Some(11)), (Some(""), Some(11))] {
            assert_eq!(parse_write_version_cap(v), want, "{v:?}");
        }
        // A test override reads the same way.
        for (v, want) in [(12, 12), (11, 11), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30), (31, 31), (32, 32), (33, 33), (34, 34), (35, 35), (36, 36), (37, 37), (38, DEFAULT_WRITE_VERSION), (10, DEFAULT_WRITE_VERSION)] {
            TEST_WRITE_VERSION.with(|t| t.set(v));
            assert_eq!(write_version(), want, "override {v}");
        }
        TEST_WRITE_VERSION.with(|t| t.set(0));
        assert!(can_read_vector_segment_version(vectors::RVEC_VERSION));
        assert!(!can_read_vector_segment_version(vectors::RVEC_VERSION + 1));
    }

    /// The written version follows the cluster floor, capped by this build
    /// and by the configured cap; an unknown floor writes the default.
    #[test]
    fn effective_write_version_follows_the_floor_and_the_cap() {
        assert_eq!(effective_write_version(0, None), DEFAULT_WRITE_VERSION);
        assert_eq!(effective_write_version(11, None), 11);
        assert_eq!(effective_write_version(12, None), 12);
        assert_eq!(effective_write_version(13, None), 13);
        // A floor above this build (never admitted) still writes what it can.
        assert_eq!(effective_write_version(VERSION + 1, None), VERSION);
        // The setting caps, it never raises above the floor.
        assert_eq!(effective_write_version(13, Some(12)), 12);
        assert_eq!(effective_write_version(11, Some(13)), 11);
        assert_eq!(effective_write_version(0, Some(13)), DEFAULT_WRITE_VERSION);
        assert_eq!(effective_write_version(13, Some(11)), 11);
    }

    /// The process floor only rises: a late older state cannot lower it.
    #[test]
    fn format_floor_never_lowers() {
        let floor = FormatFloor::new();
        assert_eq!(floor.get(), 0);
        floor.raise(0);
        assert_eq!(floor.get(), 0, "an absent floor changes nothing");
        floor.raise(12);
        assert_eq!(floor.get(), 12);
        floor.raise(11);
        assert_eq!(floor.get(), 12, "never lowers");
        floor.raise(13);
        assert_eq!(effective_write_version(floor.get(), None), 13);
        floor.raise(0);
        assert_eq!(floor.get(), 13);
    }

    #[test]
    fn crc_known_value() {
        // CRC-32/IEEE of "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(!crc32_sliced(0xFFFF_FFFF, b"123456789"), 0xCBF4_3926);
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 + 7) as u8).collect();
        for n in [0, 1, 7, 8, 9, 63, 1000] {
            assert_eq!(crc32(&data[..n]), !crc32_sliced(0xFFFF_FFFF, &data[..n]), "{n}");
            let split = crc32_update(crc32_update(0xFFFF_FFFF, &data[..n / 3]), &data[n / 3..n]);
            assert_eq!(!split, crc32(&data[..n]), "{n}");
        }
    }
}
