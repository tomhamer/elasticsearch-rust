//! On-disk term dictionary: a sorted-string table (SSTable).
//!
//! Terms are written in sorted order in blocks of [`BLOCK_TERMS`]. Within a
//! block each key is front-coded against the previous key (shared prefix
//! length + suffix). A fixed-width block index (first-key prefix, block
//! offset, first-key offset per block) lives at the end of the section and
//! is binary-searched in place: opening a dictionary reads a 32-byte
//! trailer and does no per-block work.
//!
//! Lookup: binary search the block index on the 8-byte key prefixes
//! (`num_terms / BLOCK_TERMS` entries, full first keys only on a prefix tie), then
//! decode at most one block sequentially. This is the layout Quickwit moved
//! to after finding it faster than an FST for point lookups on their
//! workloads; it is also simpler to merge.
//!
//! ```text
//! block 0 | block 1 | ... | first keys (concatenated)
//!   | index: (num_blocks + 1) × 16 bytes:
//!       u64 prefix       first 8 bytes of the block's first key, big-endian
//!                        packed, zero padded (order preserving)
//!       u32 block_off    from the section start
//!       u32 key_off      first key, from the first-keys start
//!     (the extra sentinel entry carries the end offsets)
//!   | trailer (32 bytes): u64 num_terms, u32 num_docs, u32 num_blocks,
//!       u64 keys_start, u64 index_start
//! ```
//! Dictionaries whose block data or first keys exceed 4 GiB use 24-byte
//! index entries with u64 offsets. The exact index length identifies the
//! width; the trailer and narrow representation remain unchanged.

use super::directory::FileSlice;
use super::varint::{read_bytes, read_vint, write_vint};
use crate::error::{Result, RusticError};
use crate::term_dictionary::{PostingsLocation, INLINE_POSTINGS};
use crate::types::Term;

/// Terms per block. Readers find a block by its index entry and decode it
/// to its end, so any size reads back. 32 (from 16) halves the per-block
/// index entry and first-key copy a term pays: `_id` terms (random
/// suffixes, one per doc) take 14.6 bytes instead of 16.0 on http_logs,
/// for one more 16-entry stretch decoded per lookup.
pub const BLOCK_TERMS: usize = 32;
const INDEX_ENTRY: usize = 16;
const WIDE_INDEX_ENTRY: usize = 24;
const TRAILER: usize = 32;

#[inline(always)]
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline(always)]
fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// The running deltas of a compact block's entries.
#[derive(Clone, Copy, Debug, Default)]
struct LocState {
    doc: u64,
    offset: u64,
}

#[inline]
fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap()) ^ u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if x != 0 {
            return i + (x.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Streaming writer. Keys must be added in strictly ascending order.
pub struct TermDictWriter {
    out: Vec<u8>,
    /// Per closed block: (block offset, first-key offset in `first_keys`).
    blocks: Vec<(usize, usize)>,
    first_keys: Vec<u8>,
    block_start: usize,
    prev_key: Vec<u8>,
    in_block: usize,
    num_terms: u64,
    /// Entries encode tagged offsets (inline single-doc lists, v15+).
    tagged: bool,
    /// Compact entries ([`TermDictWriter::compact`], v17+) and the
    /// running inline doc / offset they are deltas from (reset per block).
    compact: bool,
    /// Inline entries may carry a position ([`TermDictWriter::inline_positions`], v19+).
    positions: bool,
    prev_doc: u64,
    prev_offset: u64,
    /// Packed blocks ([`TermDictWriter::packed`], v32+): the open block's
    /// entries, written when it closes: suffix bytes, encoded locations,
    /// and per entry (shared, suffix start, suffix len, loc start, loc len).
    packed: bool,
    pend_suffix: Vec<u8>,
    pend_loc: Vec<u8>,
    pend: Vec<(u32, u32, u32, u32, u32)>,
    /// Block bytes already handed out by [`TermDictWriter::take_spill`]:
    /// `out` holds the section from this offset on.
    spilled: usize,
    /// Integer-keyed blocks ([`TermDictWriter::int_keys`], v34+): the open
    /// block's keys (ends in `pend_kend`), and its shape once it grows
    /// past [`BLOCK_TERMS`] as an integer-keyed block.
    int_keys: bool,
    pend_keys: Vec<u8>,
    pend_kend: Vec<u32>,
    ext: Option<IntShape>,
}

/// Packed block header ([`TermDictWriter::packed`]): bits 0-1 the suffix
/// alphabet, bit 2 one-byte entry lengths (`shared << 4 | suffix`).
pub(crate) const ALPHA_RAW: u8 = 0;
/// Suffix bytes all in `0-9a-f`: 4 bits a byte.
pub(crate) const ALPHA_HEX: u8 = 1;
/// Suffix bytes all in URL-safe base64 `A-Za-z0-9-_`: 6 bits a byte.
pub(crate) const ALPHA_B64: u8 = 2;
pub(crate) const NIBBLE_LENS: u8 = 4;
/// Integer-keyed block ([`TermDictWriter::int_keys`], v34+): every key has
/// the same length and shares a prefix with the block's first key; the rest
/// of a key, hex or URL-safe base64 characters, reads as an integer in the
/// characters' byte order, and an entry stores the delta from the previous
/// key's integer. The header byte is then followed by the prefix and key
/// lengths (one byte each).
pub(crate) const ALPHA_INT: u8 = 3;
/// [`ALPHA_INT`] header bit: base64 characters (else hex).
const INT_B64: u8 = 8;
/// Terms an integer-keyed block may hold: a delta is about a byte where
/// the suffix and its lengths took four, so the block's index entry and
/// first key copy dominate a 32-term block.
const INT_BLOCK_TERMS: usize = 128;
/// URL-safe base64 characters in byte order (an integer's digits sort as
/// the keys do).
const B64_ORDER: &[u8; 64] = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

#[inline(always)]
fn int_rank(b64: bool, c: u8) -> Option<u8> {
    if !b64 {
        return hex_code(c);
    }
    match c {
        b'-' => Some(0),
        b'0'..=b'9' => Some(c - b'0' + 1),
        b'A'..=b'Z' => Some(c - b'A' + 11),
        b'_' => Some(37),
        b'a'..=b'z' => Some(c - b'a' + 38),
        _ => None,
    }
}

fn write_vint128(mut v: u128, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline(always)]
fn read_vint128(bytes: &[u8], pos: &mut usize) -> Result<u128> {
    let (mut v, mut shift) = (0u128, 0u32);
    loop {
        let b = *bytes.get(*pos).ok_or_else(|| RusticError::Corrupted("truncated int key delta".into()))?;
        *pos += 1;
        if shift > 126 {
            return Err(RusticError::Corrupted("int key delta too long".into()));
        }
        v |= ((b & 0x7f) as u128) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

fn vint128_len(mut v: u128) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// The key shape of an [`ALPHA_INT`] block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IntShape {
    /// Bytes every key shares with the block's first key.
    p: usize,
    /// Every key's length.
    l: usize,
    b64: bool,
}

impl IntShape {
    #[inline(always)]
    fn bits(self) -> u32 {
        if self.b64 { 6 } else { 4 }
    }
    /// Most integer characters a key may have (the integer fits a u128).
    fn max_chars(b64: bool) -> usize {
        if b64 { 21 } else { 31 }
    }
    fn parse(bytes: &[u8]) -> Result<Self> {
        let bad = || RusticError::Corrupted("int key block header".into());
        let (&head, &p, &l) = match bytes {
            [h, p, l, ..] => (h, p, l),
            _ => return Err(bad()),
        };
        let shape = IntShape { p: p as usize, l: l as usize, b64: head & INT_B64 != 0 };
        if shape.p > shape.l || shape.l - shape.p > Self::max_chars(shape.b64) {
            return Err(bad());
        }
        Ok(shape)
    }
    /// The integer of `key` (of this shape's length, its characters in
    /// the alphabet).
    #[inline]
    fn value(self, key: &[u8]) -> Option<u128> {
        if key.len() != self.l {
            return None;
        }
        let mut v = 0u128;
        for &c in &key[self.p..] {
            v = (v << self.bits()) | int_rank(self.b64, c)? as u128;
        }
        Some(v)
    }
    /// Write the characters of `v` over `key[p..l]`.
    #[inline]
    fn write(self, mut v: u128, key: &mut [u8]) {
        let mask = (1u128 << self.bits()) - 1;
        for c in key[self.p..self.l].iter_mut().rev() {
            let r = (v & mask) as usize;
            *c = if self.b64 { B64_ORDER[r] } else { HEX_CHARS[r] };
            v >>= self.bits();
        }
    }
    /// The next key's integer: `cur + delta`, ascending and in range.
    #[inline(always)]
    fn step(self, cur: u128, delta: u128) -> Result<u128> {
        let bits = self.bits() as usize * (self.l - self.p);
        match cur.checked_add(delta) {
            Some(v) if delta > 0 && (bits >= 128 || v >> bits == 0) => Ok(v),
            _ => Err(RusticError::Corrupted("int key delta out of range".into())),
        }
    }
    /// Whether `key` fits a block of this shape whose first key is `first`.
    fn fits(self, first: &[u8], key: &[u8]) -> bool {
        key.len() == self.l && first.len() == self.l && key[..self.p] == first[..self.p] && key[self.p..].iter().all(|&c| int_rank(self.b64, c).is_some())
    }
}

#[inline]
pub(crate) fn hex_code(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

#[inline]
pub(crate) fn b64_code(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Bits a suffix byte takes under `alpha` (0: raw bytes).
#[inline]
pub(crate) fn alpha_bits(alpha: u8) -> u32 {
    match alpha {
        ALPHA_HEX => 4,
        ALPHA_B64 => 6,
        _ => 0,
    }
}

/// Bytes a packed suffix of `len` bytes takes under `alpha`.
#[inline]
pub(crate) fn packed_len(alpha: u8, len: usize) -> usize {
    match alpha_bits(alpha) {
        0 => len,
        b => (len * b as usize).div_ceil(8),
    }
}

/// The narrowest alphabet holding every byte of `bytes`.
pub(crate) fn choose_alpha(bytes: &[u8]) -> u8 {
    if bytes.iter().all(|&c| hex_code(c).is_some()) {
        ALPHA_HEX
    } else if bytes.iter().all(|&c| b64_code(c).is_some()) {
        ALPHA_B64
    } else {
        ALPHA_RAW
    }
}

/// Append `suffix` packed under `alpha` (LSB first; raw bytes for
/// [`ALPHA_RAW`]); its bytes must be in the alphabet.
pub(crate) fn pack_suffix(alpha: u8, suffix: &[u8], out: &mut Vec<u8>) {
    let bits = alpha_bits(alpha);
    if bits == 0 {
        out.extend_from_slice(suffix);
        return;
    }
    let (mut acc, mut have) = (0u32, 0u32);
    for &c in suffix {
        let v = if alpha == ALPHA_HEX { hex_code(c) } else { b64_code(c) }.unwrap_or(0) as u32;
        acc |= v << have;
        have += bits;
        while have >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            have -= 8;
        }
    }
    if have > 0 {
        out.push(acc as u8);
    }
}

/// Byte `i` of a suffix packed under `alpha` (a packed alphabet, LSB first).
#[inline]
pub(crate) fn packed_char(alpha: u8, packed: &[u8], i: usize) -> u8 {
    let bits = alpha_bits(alpha) as usize;
    let bit = i * bits;
    let word = packed.get(bit / 8).copied().unwrap_or(0) as u16 | (packed.get(bit / 8 + 1).copied().unwrap_or(0) as u16) << 8;
    let v = ((word >> (bit % 8)) & ((1 << bits) - 1)) as usize;
    if alpha == ALPHA_HEX { HEX_CHARS[v] } else { B64_CHARS[v] }
}

/// Decode a suffix of `len` bytes packed under `alpha` (LSB first) into `out`.
pub(crate) fn unpack_suffix(alpha: u8, packed: &[u8], len: usize, out: &mut Vec<u8>) {
    out.clear();
    let bits = alpha_bits(alpha);
    let mask = (1u32 << bits) - 1;
    let (mut acc, mut have, mut at) = (0u32, 0u32, 0usize);
    for _ in 0..len {
        while have < bits {
            acc |= (packed.get(at).copied().unwrap_or(0) as u32) << have;
            at += 1;
            have += 8;
        }
        let v = (acc & mask) as usize;
        acc >>= bits;
        have -= bits;
        out.push(if alpha == ALPHA_HEX { HEX_CHARS[v] } else { B64_CHARS[v] });
    }
}

impl Default for TermDictWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TermDictWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            blocks: Vec::new(),
            first_keys: Vec::new(),
            block_start: 0,
            prev_key: Vec::new(),
            in_block: 0,
            num_terms: 0,
            tagged: false,
            compact: false,
            positions: false,
            prev_doc: 0,
            prev_offset: 0,
            packed: false,
            pend_suffix: Vec::new(),
            pend_loc: Vec::new(),
            pend: Vec::new(),
            spilled: 0,
            int_keys: false,
            pend_keys: Vec::new(),
            pend_kend: Vec::new(),
            ext: None,
        }
    }

    /// Integer-keyed blocks (segment format [`crate::codec::INT_TERMS_SINCE`]+,
    /// with [`TermDictWriter::packed`]): a block whose keys have one length
    /// and differ only in trailing hex or base64 characters stores each key
    /// as the delta of those characters' integer from the previous key's
    /// ([`ALPHA_INT`]) when that is smaller, and grows to
    /// [`INT_BLOCK_TERMS`] terms while its keys keep the shape. k-ordered
    /// ids (logsdb auto ids) take about a byte a key this way.
    pub fn int_keys(mut self, on: bool) -> Self {
        self.int_keys = on && self.packed;
        self
    }

    /// Key `i` of the open block ([`TermDictWriter::int_keys`]).
    fn pend_key(&self, i: usize) -> &[u8] {
        let a = if i == 0 { 0 } else { self.pend_kend[i - 1] as usize };
        &self.pend_keys[a..self.pend_kend[i] as usize]
    }

    /// The open block's integer shape, if its keys have one.
    fn int_shape(&self) -> Option<IntShape> {
        let n = self.pend.len();
        if !self.int_keys || n < 2 {
            return None;
        }
        let l = self.pend_key(0).len();
        if l > 255 {
            return None;
        }
        let mut p = l;
        for i in 1..n {
            if self.pend_key(i).len() != l {
                return None;
            }
            p = p.min(self.pend[i].0 as usize);
        }
        let tail_ok = |b64: bool| (0..n).all(|i| self.pend_key(i)[p..].iter().all(|&c| int_rank(b64, c).is_some()));
        let b64 = if tail_ok(false) { false } else if tail_ok(true) { true } else { return None };
        (l - p <= IntShape::max_chars(b64)).then_some(IntShape { p, l, b64 })
    }

    /// Bytes the open block's keys take as an integer block of `shape`
    /// and as a packed block (locations aside, the same in both).
    fn int_sizes(&self, shape: IntShape) -> (usize, usize) {
        let mut int = 3;
        let mut prev = shape.value(self.pend_key(0)).unwrap_or(0);
        for i in 1..self.pend.len() {
            let v = shape.value(self.pend_key(i)).unwrap_or(prev);
            int += vint128_len(v.wrapping_sub(prev));
            prev = v;
        }
        let alpha = choose_alpha(&self.pend_suffix);
        let nibble = self.pend.iter().all(|&(sh, _, len, _, _)| sh < 16 && len < 16);
        let mut packed = 1;
        for &(shared, _, len, _, _) in &self.pend {
            packed += if nibble { 1 } else if (shared | len) < 0x80 { 2 } else { vint128_len(shared as u128) + vint128_len(len as u128) };
            packed += packed_len(alpha, len as usize);
        }
        (int, packed)
    }

    /// Packed blocks (segment format [`crate::codec::PACKED_TERMS_SINCE`]+,
    /// implies [`TermDictWriter::compact`]): a block opens with a header
    /// byte; suffixes all in `0-9a-f` take 4 bits a byte and suffixes all
    /// in URL-safe base64 6 bits (Lucene's block tree packs such suffixes
    /// too: hex and base64 ids are most high-cardinality keywords), and a
    /// block whose entries all share and add under 16 bytes writes each
    /// entry's two lengths in one byte. Readers open such a dictionary
    /// with [`TermDictReader::open_versioned`].
    pub fn packed(mut self, on: bool) -> Self {
        self.packed = on;
        if on { self = self.compact(true); }
        self
    }

    /// Write the open block's buffered entries ([`TermDictWriter::packed`]).
    fn flush_block(&mut self) {
        if self.pend.is_empty() {
            return;
        }
        let shape = self.ext.take().or_else(|| self.int_shape());
        if let Some(shape) = shape.filter(|&s| { let (int, packed) = self.int_sizes(s); int < packed }) {
            self.out.push(ALPHA_INT | if shape.b64 { INT_B64 } else { 0 });
            self.out.push(shape.p as u8);
            self.out.push(shape.l as u8);
            let mut prev = shape.value(self.pend_key(0)).expect("int key shape");
            for i in 0..self.pend.len() {
                if i > 0 {
                    let v = shape.value(self.pend_key(i)).expect("int key shape");
                    write_vint128(v - prev, &mut self.out);
                    prev = v;
                }
                let (_, _, _, l0, ll) = self.pend[i];
                self.out.extend_from_slice(&self.pend_loc[l0 as usize..(l0 + ll) as usize]);
            }
            self.pend.clear();
            self.pend_suffix.clear();
            self.pend_loc.clear();
            self.pend_keys.clear();
            self.pend_kend.clear();
            return;
        }
        self.pend_keys.clear();
        self.pend_kend.clear();
        let alpha = choose_alpha(&self.pend_suffix);
        let nibble = self.pend.iter().all(|&(sh, _, len, _, _)| sh < 16 && len < 16);
        self.out.push(alpha | if nibble { NIBBLE_LENS } else { 0 });
        for &(shared, s0, len, l0, ll) in &self.pend {
            let (shared, len) = (shared as usize, len as usize);
            if nibble {
                self.out.push(((shared << 4) | len) as u8);
            } else if (shared | len) < 0x80 {
                self.out.push(shared as u8);
                self.out.push(len as u8);
            } else {
                write_vint(shared as u64, &mut self.out);
                write_vint(len as u64, &mut self.out);
            }
            pack_suffix(alpha, &self.pend_suffix[s0 as usize..s0 as usize + len], &mut self.out);
            self.out.extend_from_slice(&self.pend_loc[l0 as usize..(l0 + ll) as usize]);
        }
        self.pend.clear();
        self.pend_suffix.clear();
        self.pend_loc.clear();
    }

    /// Inline lists may carry their one position (segment format
    /// [`crate::codec::INLINE_POSITIONS_SINCE`]+, implies
    /// [`TermDictWriter::compact`]): an inline entry is then
    /// `zigzag(doc delta) << 2 | 1`, or `| 3` followed by the position.
    pub fn inline_positions(mut self, on: bool) -> Self {
        self.positions = on;
        if on { self = self.compact(true); }
        self
    }

    /// Compact entries (segment format [`crate::codec::COMPACT_TERMS_SINCE`]+,
    /// implies [`TermDictWriter::tagged`]): a block's first entry holds no
    /// key bytes (its key is the block's first key); an inline single-doc
    /// list is `zigzag(doc - previous inline doc) << 1 | 1` alone (doc freq
    /// and total term freq 1 implied); another list is `(offset - previous
    /// offset) << 1`, then doc freq and total term freq. Both deltas start
    /// from 0 in each block. Readers open such a dictionary with
    /// [`TermDictReader::open_versioned`].
    pub fn compact(mut self, on: bool) -> Self {
        self.compact = on;
        self.tagged |= on;
        self
    }

    /// Encode tagged offsets: locations may be inline single-doc lists
    /// ([`INLINE_POSTINGS`]); readers open such a dictionary with
    /// [`TermDictReader::open_tagged`].
    pub fn tagged(mut self, on: bool) -> Self {
        self.tagged = on;
        self
    }

    /// A writer whose output buffer starts with `capacity` bytes reserved.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut w = Self::new();
        w.out.reserve(capacity);
        w
    }

    pub fn add(&mut self, key: &[u8], loc: &PostingsLocation) {
        debug_assert!(
            self.num_terms == 0 || self.in_block == 0 || self.prev_key.as_slice() < key,
            "terms must be added in ascending order"
        );
        if self.in_block >= BLOCK_TERMS {
            // An integer block grown past BLOCK_TERMS: close it before a
            // key of another shape.
            let fits = self.ext.is_some_and(|s| s.fits(self.pend_key(0), key));
            if !fits {
                self.flush_block();
                self.in_block = 0;
            }
        }
        let first = self.in_block == 0;
        if first {
            self.block_start = self.spilled + self.out.len();
            self.blocks.push((self.block_start, self.first_keys.len()));
            self.first_keys.extend_from_slice(key);
            self.prev_key.clear();
            self.prev_doc = 0;
            self.prev_offset = 0;
        }
        let shared = common_prefix(&self.prev_key, key);
        let suffix = key.len() - shared;
        // A compact block's first key is only in the first-keys region.
        let written = if self.compact && first { 0 } else { suffix };
        let loc_start = self.pend_loc.len();
        if self.packed {
            let s0 = self.pend_suffix.len();
            self.pend_suffix.extend_from_slice(&key[key.len() - written..]);
            self.pend.push((shared as u32, s0 as u32, written as u32, loc_start as u32, 0));
            if self.int_keys {
                self.pend_keys.extend_from_slice(key);
                self.pend_kend.push(self.pend_keys.len() as u32);
            }
        } else if (shared | written) < 0x80 {
            self.out.push(shared as u8);
            self.out.push(written as u8);
        } else {
            write_vint(shared as u64, &mut self.out);
            write_vint(written as u64, &mut self.out);
        }
        if !self.packed {
            self.out.extend_from_slice(&key[key.len() - written..]);
        }
        if self.compact {
            // Packed: the location goes to the open block's buffer.
            let out = if self.packed { &mut self.pend_loc } else { &mut self.out };
            if loc.offset & INLINE_POSTINGS != 0 {
                let doc = crate::term_dictionary::inline_doc(loc.offset) as u64;
                let delta = doc as i64 - self.prev_doc as i64;
                self.prev_doc = doc;
                if self.positions {
                    match crate::term_dictionary::inline_position(loc.offset) {
                        Some(p) => {
                            write_vint((zigzag(delta) << 2) | 3, out);
                            write_vint(p as u64, out);
                        }
                        None => write_vint((zigzag(delta) << 2) | 1, out),
                    }
                } else {
                    debug_assert!(crate::term_dictionary::inline_position(loc.offset).is_none(), "inline positions need format v19");
                    write_vint((zigzag(delta) << 1) | 1, out);
                }
            } else {
                let delta = loc.offset.wrapping_sub(self.prev_offset);
                debug_assert!(loc.offset >= self.prev_offset, "postings offsets ascend within a block");
                self.prev_offset = loc.offset;
                write_vint(delta << 1, out);
                write_vint(loc.doc_freq as u64, out);
                write_vint(loc.total_term_freq, out);
            }
            if self.packed {
                let n = (self.pend_loc.len() - loc_start) as u32;
                if let Some(e) = self.pend.last_mut() {
                    e.4 = n;
                }
            }
            self.prev_key.truncate(shared);
            self.prev_key.extend_from_slice(&key[shared..]);
            self.in_block += 1;
            self.num_terms += 1;
            if self.in_block == BLOCK_TERMS && self.ext.is_none() {
                // Keys an integer block holds smaller grow the block.
                self.ext = self.int_shape().filter(|&s| { let (int, packed) = self.int_sizes(s); int < packed });
            }
            let close = if self.ext.is_some() { self.in_block == INT_BLOCK_TERMS } else { self.in_block == BLOCK_TERMS };
            if close {
                if self.packed {
                    self.flush_block();
                }
                self.in_block = 0;
            }
            return;
        }
        let offset = if !self.tagged {
            debug_assert!(loc.offset & INLINE_POSTINGS == 0, "inline postings need a tagged dictionary");
            loc.offset
        } else if loc.offset & INLINE_POSTINGS != 0 {
            ((loc.offset & !INLINE_POSTINGS) << 1) | 1
        } else {
            loc.offset << 1
        };
        write_vint(offset, &mut self.out);
        write_vint(loc.doc_freq as u64, &mut self.out);
        write_vint(loc.total_term_freq, &mut self.out);

        self.prev_key.truncate(shared);
        self.prev_key.extend_from_slice(&key[shared..]);
        self.in_block += 1;
        self.num_terms += 1;
        if self.in_block == BLOCK_TERMS {
            self.in_block = 0;
        }
    }

    /// The closed blocks written so far, once they reach `threshold` bytes
    /// (between blocks only): the caller writes them out in order, ahead of
    /// what [`Self::finish`] returns (the section is their concatenation), so
    /// a merge's dictionary does not stay in memory until its end. Offsets
    /// in the index count the spilled bytes.
    pub fn take_spill(&mut self, threshold: usize) -> Option<Vec<u8>> {
        if self.in_block != 0 || !self.pend.is_empty() || self.out.len() < threshold {
            return None;
        }
        self.spilled += self.out.len();
        Some(std::mem::take(&mut self.out))
    }

    /// Bytes the writer holds now (buffers it keeps until [`Self::finish`]).
    pub fn retained_bytes(&self) -> usize {
        self.out.capacity() + self.first_keys.capacity() + self.blocks.capacity() * std::mem::size_of::<(usize, usize)>()
            + self.prev_key.capacity() + self.pend_suffix.capacity() + self.pend_loc.capacity() + self.pend.capacity() * 20
            + self.pend_keys.capacity() + self.pend_kend.capacity() * 4
    }

    /// The rest of the section: the blocks not yet spilled, the first keys,
    /// the index and the trailer (the whole section when nothing spilled).
    pub fn finish(mut self, num_docs: u32) -> Vec<u8> {
        self.flush_block();
        let wide = self.spilled + self.out.len() > u32::MAX as usize || self.first_keys.len() > u32::MAX as usize;
        self.finish_with_width(num_docs, wide)
    }

    fn finish_with_width(mut self, num_docs: u32, wide: bool) -> Vec<u8> {
        let blocks_end = self.spilled + self.out.len();
        let keys_start = self.spilled + self.out.len();
        self.out.extend_from_slice(&self.first_keys);
        let index_start = self.spilled + self.out.len();
        let entry_size = if wide { WIDE_INDEX_ENTRY } else { INDEX_ENTRY };
        self.out.reserve((self.blocks.len() + 1) * entry_size + TRAILER);
        for (i, &(off, key_off)) in self.blocks.iter().enumerate() {
            let key_end = self.blocks.get(i + 1).map_or(self.first_keys.len(), |b| b.1);
            let prefix = key_prefix(&self.first_keys[key_off..key_end]);
            self.out.extend_from_slice(&prefix.to_le_bytes());
            write_index_offsets(&mut self.out, off, key_off, wide);
        }
        // Sentinel: end offsets of the last block and its first key.
        self.out.extend_from_slice(&u64::MAX.to_le_bytes());
        write_index_offsets(&mut self.out, blocks_end, self.first_keys.len(), wide);
        self.out.extend_from_slice(&self.num_terms.to_le_bytes());
        self.out.extend_from_slice(&num_docs.to_le_bytes());
        self.out.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        self.out.extend_from_slice(&(keys_start as u64).to_le_bytes());
        self.out.extend_from_slice(&(index_start as u64).to_le_bytes());
        self.out
    }
}

fn write_index_offsets(out: &mut Vec<u8>, block: usize, key: usize, wide: bool) {
    if wide {
        out.extend_from_slice(&(block as u64).to_le_bytes());
        out.extend_from_slice(&(key as u64).to_le_bytes());
    } else {
        out.extend_from_slice(&(block as u32).to_le_bytes());
        out.extend_from_slice(&(key as u32).to_le_bytes());
    }
}

/// Read side of the SSTable. Opening it reads only the trailer; the block
/// index is binary-searched in the mapped bytes.
pub struct TermDictReader {
    /// Owns the mapping `ptr` points into.
    #[allow(dead_code)]
    data: FileSlice,
    /// `data.as_slice()` cached (the backing `Arc` lives in `data`).
    ptr: *const u8,
    len: usize,
    keys_start: usize,
    index_start: usize,
    num_blocks: usize,
    index_entry: usize,
    num_terms: u64,
    num_docs: u32,
    negative_filter: std::sync::OnceLock<Option<(u16, Vec<u64>)>>,
    /// Entries hold tagged offsets ([`TermDictWriter::tagged`]).
    tagged: bool,
    /// Compact entries ([`TermDictWriter::compact`]).
    compact: bool,
    /// Inline entries may carry a position ([`TermDictWriter::inline_positions`]).
    positions: bool,
    /// Blocks open with a header byte ([`TermDictWriter::packed`]).
    packed: bool,
}

pub(crate) fn negative_filter_address(bytes: &[u8], words: usize) -> (usize, u64) {
    negative_filter_slot(negative_filter_hash(bytes), words)
}

/// The hash a negative filter addresses `bytes` by (computed once per
/// lookup, then [`negative_filter_slot`] per filter).
pub fn negative_filter_hash(bytes: &[u8]) -> u64 {
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    let mut hash = bytes.len() as u64 ^ 0x9e3779b97f4a7c15;
    for part in bytes.chunks(8) {
        let mut word = [0u8; 8];
        word[..part.len()].copy_from_slice(part);
        hash = mix(hash ^ u64::from_le_bytes(word));
    }
    hash
}

/// Word and bit mask of `hash` in a filter of `words` (a power of two).
#[inline]
pub(crate) fn negative_filter_slot(hash: u64, words: usize) -> (usize, u64) {
    let mask = (0..5).fold(0u64, |mask, i| mask | (1 << ((hash >> (i * 6)) & 63)));
    ((hash >> 32) as usize & (words - 1), mask)
}

// SAFETY: `ptr`/`len` point into memory owned by `data` (an `Arc` of
// `Send + Sync` bytes), which lives exactly as long as the reader.
unsafe impl Send for TermDictReader {}
unsafe impl Sync for TermDictReader {}

impl TermDictReader {
    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        // SAFETY: see the `Send` impl.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    fn entry(&self, b: usize) -> &[u8] {
        let o = self.index_start + b * self.index_entry;
        &self.bytes()[o..o + self.index_entry]
    }

    #[inline(always)]
    fn prefix(&self, b: usize) -> u64 {
        u64::from_le_bytes(self.entry(b)[..8].try_into().unwrap())
    }

    #[inline(always)]
    fn offsets(&self, b: usize) -> (usize, usize) {
        let e = self.entry(b);
        if self.index_entry == WIDE_INDEX_ENTRY {
            (u64::from_le_bytes(e[8..16].try_into().unwrap()) as usize,
             u64::from_le_bytes(e[16..24].try_into().unwrap()) as usize)
        } else {
            (u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize,
             u32::from_le_bytes(e[12..16].try_into().unwrap()) as usize)
        }
    }

    /// First key of block `b`.
    #[inline]
    fn first_key(&self, b: usize) -> &[u8] {
        let (_, k0) = self.offsets(b);
        let (_, k1) = self.offsets(b + 1);
        &self.bytes()[self.keys_start + k0..self.keys_start + k1]
    }

    /// Encoded bytes of block `b`.
    #[inline]
    fn block_bytes(&self, b: usize) -> &[u8] {
        let (o0, _) = self.offsets(b);
        let (o1, _) = self.offsets(b + 1);
        &self.bytes()[o0..o1]
    }

    /// Block `b`'s header byte ([`TermDictWriter::packed`]; 0 otherwise)
    /// and where its first entry starts.
    #[inline]
    fn block_head(&self, bytes: &[u8]) -> (u8, usize) {
        match (self.packed, bytes.first().copied().unwrap_or(0)) {
            (false, _) => (0, 0),
            (true, h) if h & 3 == ALPHA_INT => (h, 3),
            (true, h) => (h, 1),
        }
    }

    /// An entry's shared prefix and suffix lengths at `pos` (block header
    /// `head`).
    #[inline(always)]
    fn read_lens(bytes: &[u8], pos: &mut usize, head: u8) -> Result<(usize, usize)> {
        if head & NIBBLE_LENS != 0 {
            let b = *bytes.get(*pos).ok_or_else(|| RusticError::Corrupted("term entry out of range".into()))?;
            *pos += 1;
            return Ok(((b >> 4) as usize, (b & 15) as usize));
        }
        match bytes.get(*pos..*pos + 2) {
            Some(&[s, l]) if (s | l) < 0x80 => {
                *pos += 2;
                Ok((s as usize, l as usize))
            }
            _ => {
                let s = read_vint(bytes, pos)? as usize;
                Ok((s, read_vint(bytes, pos)? as usize))
            }
        }
    }

    /// An entry's suffix of `len` bytes at `pos`: the stored bytes, or
    /// (packed alphabet) decoded into `scratch`.
    #[inline(always)]
    fn read_suffix<'b>(bytes: &'b [u8], pos: &mut usize, len: usize, head: u8, scratch: &'b mut Vec<u8>) -> Result<&'b [u8]> {
        let alpha = head & 3;
        let stored = read_bytes(bytes, pos, packed_len(alpha, len))?;
        if alpha == ALPHA_RAW {
            return Ok(stored);
        }
        unpack_suffix(alpha, stored, len, scratch);
        Ok(scratch.as_slice())
    }
}

/// Order-preserving 8-byte prefix of a key (`a < b` ⇒ `prefix(a) <= prefix(b)`,
/// and `prefix(a) < prefix(b)` ⇒ `a < b`).
#[inline]
fn key_prefix(k: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    let n = k.len().min(8);
    b[..n].copy_from_slice(&k[..n]);
    u64::from_be_bytes(b)
}

/// Skip the three vints of a term entry's location (offset, doc freq,
/// total term freq) without a branch per byte: find the third byte with a
/// clear continuation bit in a 16-byte window.
#[inline(always)]
fn skip_3_vints(bytes: &[u8], pos: &mut usize) -> Result<()> {
    if let Some(w) = bytes.get(*pos..*pos + 16) {
        let w = u128::from_le_bytes(w.try_into().unwrap());
        let mut ends = !w & 0x8080_8080_8080_8080_8080_8080_8080_8080u128;
        ends &= ends.wrapping_sub(1);
        ends &= ends.wrapping_sub(1);
        if ends != 0 {
            *pos += ends.trailing_zeros() as usize / 8 + 1;
            return Ok(());
        }
    }
    skip_vint(bytes, pos)?;
    skip_vint(bytes, pos)?;
    skip_vint(bytes, pos)
}

#[inline(always)]
fn skip_vint(bytes: &[u8], pos: &mut usize) -> Result<()> {
    loop {
        let Some(&b) = bytes.get(*pos) else {
            return Err(RusticError::Corrupted("truncated vint".into()));
        };
        *pos += 1;
        if b & 0x80 == 0 {
            return Ok(());
        }
    }
}

impl TermDictReader {
    pub fn open(data: FileSlice) -> Result<Self> {
        Self::open_tagged(data, false)
    }

    /// Open a dictionary written with tagged offsets or not (segment format
    /// [`crate::codec::INLINE_POSTINGS_SINCE`]+ writes them).
    pub fn open_tagged(data: FileSlice, tagged: bool) -> Result<Self> {
        Self::open_layout(data, tagged, false)
    }

    /// Open the dictionary of a segment of format `version` (tagged from
    /// [`crate::codec::INLINE_POSTINGS_SINCE`], compact from
    /// [`crate::codec::COMPACT_TERMS_SINCE`]).
    pub fn open_versioned(data: FileSlice, version: u32) -> Result<Self> {
        let mut r = Self::open_layout(data, version >= crate::codec::INLINE_POSTINGS_SINCE, version >= crate::codec::COMPACT_TERMS_SINCE)?;
        r.positions = version >= crate::codec::INLINE_POSITIONS_SINCE;
        r.packed = version >= crate::codec::PACKED_TERMS_SINCE;
        Ok(r)
    }

    fn open_layout(data: FileSlice, tagged: bool, compact: bool) -> Result<Self> {
        let bytes = data.as_slice();
        let corrupted = |m: &str| RusticError::Corrupted(format!("term dict: {m}"));
        if bytes.len() < TRAILER {
            return Err(corrupted("too short"));
        }
        let t = &bytes[bytes.len() - TRAILER..];
        let num_terms = u64::from_le_bytes(t[0..8].try_into().unwrap());
        let num_docs = u32::from_le_bytes(t[8..12].try_into().unwrap());
        let num_blocks = u32::from_le_bytes(t[12..16].try_into().unwrap()) as usize;
        let keys_start = u64::from_le_bytes(t[16..24].try_into().unwrap()) as usize;
        let index_start = u64::from_le_bytes(t[24..32].try_into().unwrap()) as usize;
        // The trailer fixes both the index start and entry count. Infer the
        // width from its exact length, retaining byte-for-byte legacy output
        // for dictionaries below 4 GiB and rejecting any other layout.
        let index_bytes = (bytes.len() - TRAILER).checked_sub(index_start)
            .ok_or_else(|| corrupted("bad index start"))?;
        let entries = num_blocks.checked_add(1).ok_or_else(|| corrupted("bad block count"))?;
        let index_entry = if entries.checked_mul(INDEX_ENTRY) == Some(index_bytes) {
            INDEX_ENTRY
        } else if entries.checked_mul(WIDE_INDEX_ENTRY) == Some(index_bytes) {
            WIDE_INDEX_ENTRY
        } else {
            return Err(corrupted("bad index layout"));
        };
        if keys_start > index_start { return Err(corrupted("bad key layout")); }
        let r = Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            data,
            keys_start,
            index_start,
            num_blocks,
            index_entry,
            num_terms,
            num_docs,
            negative_filter: std::sync::OnceLock::new(),
            tagged: tagged || compact,
            compact,
            positions: false,
            packed: false,
        };
        // O(1) sanity check of the sentinel; per-block offsets are checked
        // by slice indexing when a block is read.
        let (blocks_end, keys_len) = r.offsets(num_blocks);
        if blocks_end > keys_start || keys_start.checked_add(keys_len) != Some(index_start) {
            return Err(corrupted("index out of range"));
        }
        Ok(r)
    }

    /// Build an empty reader (segment with no terms).
    pub fn empty(num_docs: u32) -> Self {
        let bytes = TermDictWriter::new().finish(num_docs);
        Self::open(FileSlice::from_vec(bytes)).expect("empty dict is valid")
    }

    pub fn num_terms(&self) -> u64 {
        self.num_terms
    }

    /// The location of the entry at `pos` (after its key), advancing
    /// `pos`; `st` carries a compact block's deltas.
    #[inline(always)]
    fn read_loc(&self, bytes: &[u8], pos: &mut usize, st: &mut LocState) -> Result<PostingsLocation> {
        if !self.compact {
            return Ok(PostingsLocation {
                offset: self.offset(read_vint(bytes, pos)?),
                doc_freq: read_vint(bytes, pos)? as u32,
                total_term_freq: read_vint(bytes, pos)?,
            });
        }
        let x = read_vint(bytes, pos)?;
        if x & 1 == 1 {
            let (delta, position) = if self.positions {
                let p = if x & 2 != 0 { Some(read_vint(bytes, pos)?) } else { None };
                (unzigzag(x >> 2), p)
            } else {
                (unzigzag(x >> 1), None)
            };
            st.doc = (st.doc as i64).wrapping_add(delta) as u64;
            if st.doc > u32::MAX as u64 || position.is_some_and(|p| p > crate::term_dictionary::INLINE_MAX_POSITION as u64) {
                return Err(RusticError::Corrupted("inline doc out of range".into()));
            }
            let offset = crate::term_dictionary::inline_offset(st.doc as u32, position.map(|p| p as u32));
            return Ok(PostingsLocation { offset, doc_freq: 1, total_term_freq: 1 });
        }
        st.offset = st.offset.wrapping_add(x >> 1);
        Ok(PostingsLocation { offset: st.offset, doc_freq: read_vint(bytes, pos)? as u32, total_term_freq: read_vint(bytes, pos)? })
    }

    /// The key bytes of an entry whose stored suffix is `stored`: a compact
    /// block's first entry stores none (its key is the block's first key).
    #[inline(always)]
    fn entry_suffix<'b>(&'b self, b: usize, first: bool, stored: &'b [u8]) -> &'b [u8] {
        if self.compact && first { self.first_key(b) } else { stored }
    }

    /// A location's offset as stored: in a tagged dictionary, `offset << 1`
    /// or an inline list's `doc << 1 | 1` ([`INLINE_POSTINGS`]).
    #[inline(always)]
    fn offset(&self, raw: u64) -> u64 {
        if !self.tagged {
            raw
        } else if raw & 1 == 1 {
            INLINE_POSTINGS | (raw >> 1)
        } else {
            raw >> 1
        }
    }

    /// Walk every term with its exact encoded bytes, for disk-usage
    /// reporting: `each(key, entry_bytes, loc)` gets the entry's bytes in
    /// its block plus, for a block's first term, that block's copy of the
    /// key in the first-keys region and its index entry. Returns the bytes
    /// attributed to no term (the index sentinel entry and the trailer).
    pub fn for_each_term_bytes(&self, mut each: impl FnMut(&[u8], u64, &PostingsLocation)) -> Result<u64> {
        let mut key: Vec<u8> = Vec::new();
        let mut scratch = Vec::new();
        for b in 0..self.num_blocks {
            let bytes = self.block_bytes(b);
            let (head, mut pos) = self.block_head(bytes);
            let mut first = true;
            let mut st = LocState::default();
            key.clear();
            if head & 3 == ALPHA_INT && self.packed {
                let shape = IntShape::parse(bytes)?;
                key.extend_from_slice(self.first_key(b));
                let mut cur = shape.value(&key).ok_or_else(|| RusticError::Corrupted("int key block first key".into()))?;
                while pos < bytes.len() {
                    let start = pos;
                    if !first {
                        cur = shape.step(cur, read_vint128(bytes, &mut pos)?)?;
                        shape.write(cur, &mut key);
                    }
                    let loc = self.read_loc(bytes, &mut pos, &mut st)?;
                    let mut n = (pos - start) as u64;
                    if first {
                        n += (self.first_key(b).len() + self.index_entry + 3) as u64;
                        first = false;
                    }
                    each(&key, n, &loc);
                }
                continue;
            }
            while pos < bytes.len() {
                let start = pos;
                let (shared, suffix) = Self::read_lens(bytes, &mut pos, head)?;
                if shared > key.len() {
                    return Err(RusticError::Corrupted("bad shared prefix".into()));
                }
                key.truncate(shared);
                let stored = Self::read_suffix(bytes, &mut pos, suffix, head, &mut scratch)?;
                key.extend_from_slice(self.entry_suffix(b, first, stored));
                let loc = self.read_loc(bytes, &mut pos, &mut st)?;
                let mut n = (pos - start) as u64;
                if first {
                    // The block's header byte and first-key copy and index entry.
                    n += (self.first_key(b).len() + self.index_entry) as u64 + u64::from(self.packed);
                    first = false;
                }
                each(&key, n, &loc);
            }
        }
        Ok((self.index_entry + TRAILER) as u64)
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// The dictionary's encoded size in bytes.
    pub fn byte_len(&self) -> usize {
        self.len
    }

    /// Index of the last block whose first key is <= `key`.
    fn locate_block(&self, key: &[u8]) -> Option<usize> {
        let pk = key_prefix(key);
        // First block whose prefix is >= pk.
        let (mut lo, mut hi) = (0usize, self.num_blocks);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.prefix(mid) < pk {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        self.past_ties(lo, pk, key)
    }

    /// [`Self::locate_block`] for a key that sorts at or after the first key
    /// of block `from`. It gallops forward from `from`, so a run of ascending
    /// keys landing in nearby blocks costs a few probes per key instead of a
    /// full binary search.
    fn locate_block_after(&self, key: &[u8], from: usize) -> Option<usize> {
        let pk = key_prefix(key);
        let n = self.num_blocks;
        let (mut lo, mut hi, mut step) = (from, from, 1usize);
        while hi < n && self.prefix(hi) < pk {
            lo = hi + 1;
            hi = (hi + step).min(n);
            step *= 2;
        }
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.prefix(mid) < pk {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Blocks before `from` that tie on the prefix have first keys <= the
        // first key of `from` <= `key`, so the tie scan skips them anyway.
        self.past_ties(lo, pk, key)
    }

    /// From `lo`, the first block whose prefix is >= `pk`, step past the
    /// blocks whose first key is still <= `key` and return the last of them.
    fn past_ties(&self, lo: usize, pk: u64, key: &[u8]) -> Option<usize> {
        // Blocks from `lo` whose prefix ties with the key's need a full
        // compare of their first key.
        let mut idx = lo;
        while idx < self.num_blocks && self.prefix(idx) == pk && self.first_key(idx) <= key {
            idx += 1;
            if idx - lo > 8 {
                // Long tie run: binary search it.
                let (mut a, mut z) = (idx, self.num_blocks);
                while a < z {
                    let mid = (a + z) / 2;
                    if self.prefix(mid) == pk && self.first_key(mid) <= key {
                        a = mid + 1;
                    } else {
                        z = mid;
                    }
                }
                idx = a;
                break;
            }
        }
        idx.checked_sub(1)
    }

    /// Find `key` in block `b` without rebuilding keys: track the common
    /// prefix `m` of the previous (smaller) key with `key`; a front-coded
    /// entry sharing more than `m` bytes is still smaller, one sharing fewer
    /// is larger, and only `shared == m` needs a byte compare of the suffix.
    fn find_in_block(&self, b: usize, key: &[u8]) -> Result<Option<PostingsLocation>> {
        let bytes = self.block_bytes(b);
        let n = bytes.len();
        let (head, mut pos) = self.block_head(bytes);
        if head & 3 == ALPHA_INT && self.packed {
            // Every key of the block has its shape: compare integers.
            let shape = IntShape::parse(bytes)?;
            let fk = self.first_key(b);
            let mut cur = shape.value(fk).ok_or_else(|| RusticError::Corrupted("int key block first key".into()))?;
            if key.len() != shape.l || key[..shape.p] != fk[..shape.p] {
                return Ok(None);
            }
            let Some(want) = shape.value(key) else { return Ok(None) };
            let mut st = LocState::default();
            let mut first = true;
            while pos < n {
                if !std::mem::replace(&mut first, false) {
                    cur = shape.step(cur, read_vint128(bytes, &mut pos)?)?;
                }
                if cur > want {
                    return Ok(None);
                }
                let loc = self.read_loc(bytes, &mut pos, &mut st)?;
                if cur == want {
                    return Ok(Some(loc));
                }
            }
            return Ok(None);
        }
        let mut m = 0usize;
        let mut st = LocState::default();
        let mut first = true;
        let mut scratch = Vec::new();
        while pos < n {
            // Fast path: one-byte `shared` and `suffix_len` (keys < 128 bytes).
            let (shared, suffix_len) = Self::read_lens(bytes, &mut pos, head)?;
            let start = pos;
            pos += packed_len(head & 3, suffix_len);
            if pos > n {
                return Err(RusticError::Corrupted("term suffix out of range".into()));
            }
            if shared < m {
                return Ok(None);
            }
            let was_first = std::mem::replace(&mut first, false);
            if shared == m {
                let stored = if head & 3 == ALPHA_RAW {
                    &bytes[start..pos]
                } else {
                    unpack_suffix(head & 3, &bytes[start..pos], suffix_len, &mut scratch);
                    scratch.as_slice()
                };
                let suffix = self.entry_suffix(b, was_first, stored);
                let rest = &key[m..];
                let c = common_prefix(suffix, rest);
                if c == suffix.len() && c == rest.len() {
                    return Ok(Some(self.read_loc(bytes, &mut pos, &mut st)?));
                }
                if c == rest.len() || (c < suffix.len() && suffix[c] > rest[c]) {
                    return Ok(None);
                }
                m += c;
            }
            if self.compact {
                // The deltas need every entry's location.
                self.read_loc(bytes, &mut pos, &mut st)?;
            } else {
                skip_3_vints(bytes, &mut pos)?;
            }
        }
        Ok(None)
    }

    /// Decode block `b`, calling `f(key, loc)` per term until it returns false.
    fn for_each_in_block(
        &self,
        b: usize,
        mut f: impl FnMut(&[u8], PostingsLocation) -> bool,
    ) -> Result<()> {
        let bytes = self.block_bytes(b);
        let (head, mut pos) = self.block_head(bytes);
        let mut key = Vec::with_capacity(64);
        let mut st = LocState::default();
        let mut first = true;
        let mut scratch = Vec::new();
        if head & 3 == ALPHA_INT && self.packed {
            let shape = IntShape::parse(bytes)?;
            key.extend_from_slice(self.first_key(b));
            let mut cur = shape.value(&key).ok_or_else(|| RusticError::Corrupted("int key block first key".into()))?;
            while pos < bytes.len() {
                if !std::mem::replace(&mut first, false) {
                    cur = shape.step(cur, read_vint128(bytes, &mut pos)?)?;
                    shape.write(cur, &mut key);
                }
                let loc = self.read_loc(bytes, &mut pos, &mut st)?;
                if !f(&key, loc) {
                    break;
                }
            }
            return Ok(());
        }
        while pos < bytes.len() {
            let (shared, suffix_len) = Self::read_lens(bytes, &mut pos, head)?;
            if shared > key.len() {
                return Err(RusticError::Corrupted("bad shared prefix".into()));
            }
            key.truncate(shared);
            let stored = Self::read_suffix(bytes, &mut pos, suffix_len, head, &mut scratch)?;
            key.extend_from_slice(self.entry_suffix(b, std::mem::replace(&mut first, false), stored));
            let loc = self.read_loc(bytes, &mut pos, &mut st)?;
            if !f(&key, loc) {
                break;
            }
        }
        Ok(())
    }

    /// Exact lookup by raw key.
    pub fn lookup(&self, key: &[u8]) -> Result<Option<PostingsLocation>> {
        let Some(b) = self.locate_block(key) else {
            return Ok(None);
        };
        self.find_in_block(b, key)
    }

    /// Lookup by field and term text (the key is built on the stack).
    pub fn get_parts(&self, field: u16, text: &[u8]) -> Option<PostingsLocation> {
        let n = 2 + text.len();
        if n <= 64 {
            let mut buf = [0u8; 64];
            buf[..2].copy_from_slice(&field.to_le_bytes());
            buf[2..n].copy_from_slice(text);
            self.lookup(&buf[..n]).ok().flatten()
        } else {
            self.lookup(&Term::new(field, text.to_vec()).key()).ok().flatten()
        }
    }

    pub fn get(&self, term: &Term) -> Option<PostingsLocation> {
        self.get_parts(term.field, &term.text)
    }

    /// Look up the ascending `texts` of `field`, calling `f(i, loc)` for each
    /// `texts[i]` present. Each block search gallops from the previous key's
    /// block, so a large sorted batch (a `terms` query) avoids a full binary
    /// search per key.
    pub fn lookup_sorted(&self, field: u16, texts: &[&[u8]], mut f: impl FnMut(usize, PostingsLocation)) {
        debug_assert!(texts.windows(2).all(|w| w[0] <= w[1]), "texts must be ascending");
        let mut key = Vec::with_capacity(66);
        let mut block = None;
        let mut i = 0;
        while i < texts.len() {
            key.clear();
            key.extend_from_slice(&field.to_le_bytes());
            key.extend_from_slice(texts[i]);
            let b = match block {
                Some(from) => self.locate_block_after(&key, from),
                None => self.locate_block(&key),
            };
            let Some(b) = b else { i += 1; continue };
            block = Some(b);
            let mut end = i + 1;
            while end < texts.len() {
                key.truncate(2);
                key.extend_from_slice(texts[end]);
                if b + 1 < self.num_blocks && key.as_slice() >= self.first_key(b + 1) { break; }
                end += 1;
            }
            if end == i + 1 {
                key.truncate(2);
                key.extend_from_slice(texts[i]);
                if let Ok(Some(loc)) = self.find_in_block(b, &key) { f(i, loc); }
            } else {
                // Dense batches decode each shared block once instead of rescanning
                // its front-coded prefix independently for every requested term.
                let mut wanted = i;
                let prefix = field.to_le_bytes();
                let _ = self.for_each_in_block(b, |key, loc| {
                    if key.len() < 2 { return false; }
                    match key[..2].cmp(&prefix) {
                        std::cmp::Ordering::Less => return true,
                        std::cmp::Ordering::Greater => return false,
                        std::cmp::Ordering::Equal => {}
                    }
                    while wanted < end && texts[wanted] < &key[2..] { wanted += 1; }
                    while wanted < end && texts[wanted] == &key[2..] { f(wanted, loc); wanted += 1; }
                    wanted < end
                });
            }
            i = end;
        }
    }

    /// Optional negative-lookup accelerator for one near-unique field.
    /// Built only when explicitly requested by an append-heavy caller. Other
    /// fields and any decode failure conservatively retain the exact lookup.
    /// False positives are allowed; a negative never hides an indexed term.
    pub fn may_contain_unique_term(&self, term: &Term) -> bool {
        self.may_contain_unique_hash(term.field, negative_filter_hash(&term.text))
    }

    /// [`TermDictReader::may_contain_unique_term`] by the term's
    /// [`negative_filter_hash`] (hashed once for many filters).
    pub fn may_contain_unique_hash(&self, field: u16, hash: u64) -> bool {
        let filter = self.negative_filter.get_or_init(|| self.build_negative_filter(field).ok().map(|bits| (field, bits)));
        let Some((filter_field, bits)) = filter else { return true };
        if *filter_field != field { return true; }
        let (word, mask) = negative_filter_slot(hash, bits.len());
        bits[word] & mask == mask
    }

    fn build_negative_filter(&self, field: u16) -> Result<Vec<u64>> {
        // 12 bits/document, capped at 64 MiB per segment. Saturation only
        // increases false positives; it cannot affect correctness.
        let words = ((self.num_docs as u64 * 12).div_ceil(64).max(1).next_power_of_two()).min(8 * 1024 * 1024) as usize;
        let mut bits = vec![0u64; words];
        let lower = field.to_le_bytes();
        let upper = crate::types::field_key_end(field);
        let mut done = false;
        for b in self.locate_block(&lower).unwrap_or(0)..self.num_blocks {
            if done || upper.as_ref().is_some_and(|u| self.first_key(b) >= u.as_slice()) { break; }
            self.for_each_in_block(b, |key, _| {
                if key < lower.as_slice() { return true; }
                if !key.starts_with(&lower) { done = true; return false; }
                let (word, mask) = negative_filter_address(&key[2..], words);
                bits[word] |= mask;
                true
            })?;
        }
        Ok(bits)
    }

    pub fn contains(&self, term: &Term) -> bool {
        self.get(term).is_some()
    }

    pub fn doc_freq(&self, term: &Term) -> u32 {
        self.get(term).map(|l| l.doc_freq).unwrap_or(0)
    }

    pub fn total_term_freq(&self, term: &Term) -> u64 {
        self.get(term).map(|l| l.total_term_freq).unwrap_or(0)
    }

    /// BM25 IDF for a term in this segment.
    pub fn idf(&self, term: &Term) -> f32 {
        crate::term_dictionary::bm25_idf(self.doc_freq(term), self.num_docs)
    }

    /// All (key, location) pairs with `lower <= key < upper`.
    pub fn range(&self, lower: &[u8], upper: Option<&[u8]>) -> Result<Vec<(Vec<u8>, PostingsLocation)>> {
        let mut out = Vec::new();
        let start = self.locate_block(lower).unwrap_or(0);
        let mut done = false;
        for b in start..self.num_blocks {
            if done {
                break;
            }
            if let Some(u) = upper {
                if self.first_key(b) >= u {
                    break;
                }
            }
            self.for_each_in_block(b, |k, loc| {
                if k < lower {
                    return true;
                }
                if let Some(u) = upper {
                    if k >= u {
                        done = true;
                        return false;
                    }
                }
                out.push((k.to_vec(), loc));
                true
            })?;
        }
        Ok(out)
    }

    /// Terms of `field` with text in `[lower, upper)`; `None` = unbounded within the field.
    pub fn range_iter(
        &self,
        field: u16,
        lower: Option<&[u8]>,
        upper: Option<&[u8]>,
    ) -> Vec<(Vec<u8>, PostingsLocation)> {
        let prefix = field.to_le_bytes();
        let mut lo = prefix.to_vec();
        if let Some(l) = lower {
            lo.extend_from_slice(l);
        }
        let hi: Option<Vec<u8>> = match upper {
            Some(u) => {
                let mut h = prefix.to_vec();
                h.extend_from_slice(u);
                Some(h)
            }
            None => crate::types::Term::field_key_end(field),
        };
        self.range(&lo, hi.as_deref()).unwrap_or_default()
    }

    /// Every term in the dictionary, in order.
    pub fn iter_all(&self) -> Result<Vec<(Vec<u8>, PostingsLocation)>> {
        self.range(&[], None)
    }
}

impl TermDictReader {
    /// A streaming cursor over every term in order (no materialization).
    /// A cursor beginning at the block that may contain `lower`. Callers
    /// still skip preceding keys in that first block and stop at their upper
    /// bound, avoiding a materialized range or a scan of unrelated fields.
    pub fn cursor_from(&self,lower:&[u8])->TermCursor<'_> {
        let mut cursor=self.cursor();cursor.block=self.locate_block(lower).unwrap_or(0);cursor
    }
    pub fn cursor(&self) -> TermCursor<'_> {
        TermCursor {
            dict: self,
            block: 0,
            bytes: &[],
            pos: 0,
            key: Vec::with_capacity(64),
            loc: PostingsLocation { offset: 0, doc_freq: 0, total_term_freq: 0 },
            st: LocState::default(),
            head: 0,
            scratch: Vec::new(),
            start: 0,
            int: None,
        }
    }
}

/// Streaming in-order iterator over a [`TermDictReader`]: decodes one
/// block at a time into a reused key buffer.
pub struct TermCursor<'a> {
    dict: &'a TermDictReader,
    /// Next block to load.
    block: usize,
    bytes: &'a [u8],
    pos: usize,
    key: Vec<u8>,
    loc: PostingsLocation,
    /// The current block's running deltas (compact dictionaries).
    st: LocState,
    /// The current block's header ([`TermDictWriter::packed`]) and a
    /// packed suffix's decoded bytes.
    head: u8,
    scratch: Vec<u8>,
    /// Where the current block's first entry starts.
    start: usize,
    /// The current block's integer shape and key integer ([`ALPHA_INT`]).
    int: Option<(IntShape, u128)>,
}

impl<'a> TermCursor<'a> {
    /// Move to the next term; false at the end.
    pub fn advance(&mut self) -> Result<bool> {self.advance_guarded(|_|Ok(()))}
    /// Admit the decoded key before growing its reusable buffer.
    pub fn advance_guarded(&mut self,mut reserve:impl FnMut(u64)->Result<()>)->Result<bool> {
        if self.pos >= self.bytes.len() {
            if self.block >= self.dict.num_blocks {
                return Ok(false);
            }
            self.bytes = self.dict.block_bytes(self.block);
            self.block += 1;
            self.key.clear();
            self.st = LocState::default();
            if self.bytes.is_empty() {
                return Err(RusticError::Corrupted("empty term block".into()));
            }
            (self.head, self.pos) = self.dict.block_head(self.bytes);
            self.start = self.pos;
            self.int = None;
            if self.head & 3 == ALPHA_INT && self.dict.packed {
                let shape = IntShape::parse(self.bytes)?;
                let fk = self.dict.first_key(self.block - 1);
                let v = shape.value(fk).ok_or_else(|| RusticError::Corrupted("int key block first key".into()))?;
                self.int = Some((shape, v));
            }
        }
        let bytes = self.bytes;
        let first = self.pos == self.start;
        if let Some((shape, cur)) = self.int {
            if first {
                reserve(shape.l as u64 * 2 + 128)?;
                self.key.extend_from_slice(self.dict.first_key(self.block - 1));
            } else {
                let next = shape.step(cur, read_vint128(bytes, &mut self.pos)?)?;
                shape.write(next, &mut self.key);
                self.int = Some((shape, next));
            }
            self.loc = self.dict.read_loc(bytes, &mut self.pos, &mut self.st)?;
            return Ok(true);
        }
        let pos = &mut self.pos;
        let (shared, suffix_len) = TermDictReader::read_lens(bytes, pos, self.head)?;
        if shared > self.key.len() {
            return Err(RusticError::Corrupted("bad shared prefix".into()));
        }
        reserve((shared as u64).saturating_add(suffix_len as u64).saturating_mul(2).saturating_add(128))?;
        self.key.truncate(shared);
        let stored = TermDictReader::read_suffix(bytes, pos, suffix_len, self.head, &mut self.scratch)?;
        self.key.extend_from_slice(self.dict.entry_suffix(self.block - 1, first, stored));
        self.loc = self.dict.read_loc(bytes, pos, &mut self.st)?;
        Ok(true)
    }

    /// Current key (valid after `advance` returned true).
    #[inline]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Current location.
    #[inline]
    pub fn loc(&self) -> PostingsLocation {
        self.loc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(i: u64) -> PostingsLocation {
        PostingsLocation {
            offset: i * 10,
            doc_freq: i as u32,
            total_term_freq: i * 2,
        }
    }

    fn build(n: usize) -> TermDictReader {
        let mut w = TermDictWriter::new();
        for i in 0..n {
            w.add(Term::new_text(0, &format!("term_{i:06}")).key().as_slice(), &loc(i as u64));
        }
        TermDictReader::open(FileSlice::from_vec(w.finish(1000))).unwrap()
    }

    /// A dictionary whose closed blocks were handed out as they filled
    /// (`take_spill`, as a merge writes them to its scratch file) is,
    /// concatenated with the rest `finish` returns, byte-identical to one
    /// written in one piece, in every entry format (integer-keyed blocks of
    /// up to 128 terms included: they spill whole); spills come only between
    /// blocks, and the writer then holds less.
    #[test]
    fn spilled_dictionary_is_byte_identical() {
        let keys: Vec<Vec<u8>> = {
            let mut k: Vec<Vec<u8>> = (0..3000u32).map(|i| Term::new_text(0, &format!("text with value {}_{}", i % 37, i * 7919 % 100_003)).key()).collect();
            // Same-shape ids (integer-keyed blocks), runs broken by others.
            k.extend((0..2000u32).map(|i| Term::new_text(1, &format!("AaGc2ZTwVrHG6{:07x}", i * 3 + (i % 211 == 0) as u32)).key()));
            k.extend((0..40u32).map(|i| Term::new_text(1, &format!("AaGc2ZTwVrHG6{:07x}-x", i * 150)).key()));
            k.sort();
            k.dedup();
            k
        };
        let locs: Vec<PostingsLocation> = (0..keys.len()).map(|i| if i % 5 == 0 {
            PostingsLocation { offset: 100 + i as u64 * 40, doc_freq: 2, total_term_freq: 3 }
        } else {
            PostingsLocation { offset: INLINE_POSTINGS | (i as u64 * 13 % 100_000), doc_freq: 1, total_term_freq: 1 }
        }).collect();
        type Mode = fn(TermDictWriter) -> TermDictWriter;
        let modes: [(&str, Mode); 5] = [
            ("plain", |w| w),
            ("tagged", |w| w.tagged(true)),
            ("compact", |w| w.compact(true)),
            ("packed", |w| w.packed(true)),
            ("int", |w| w.packed(true).int_keys(true)),
        ];
        for (name, mode) in modes {
            let fixed = |i: usize| if name == "plain" { PostingsLocation { offset: 100 + i as u64 * 40, doc_freq: 2, total_term_freq: 3 } } else { locs[i] };
            let mut whole = mode(TermDictWriter::new());
            for (i, k) in keys.iter().enumerate() { whole.add(k, &fixed(i)); }
            let whole = whole.finish(5000);
            if name == "int" {
                let mut r = TermDictReader::open_layout(FileSlice::from_vec(whole.clone()), true, true).unwrap();
                r.packed = true;
                let ints = (0..r.num_blocks).filter(|&b| r.block_bytes(b)[0] & 3 == ALPHA_INT).count();
                assert!(ints > 3, "{ints} integer blocks");
                for (i, k) in keys.iter().enumerate() { assert_eq!(r.lookup(k).unwrap(), Some(locs[i])); }
            }
            for threshold in [1usize, 700, 64 << 10] {
                let mut w = mode(TermDictWriter::new());
                let mut spilled = Vec::new();
                let mut spills = 0;
                for (i, k) in keys.iter().enumerate() {
                    w.add(k, &fixed(i));
                    if let Some(blocks) = w.take_spill(threshold) {
                        spills += 1;
                        spilled.extend_from_slice(&blocks);
                    }
                }
                if threshold == 1 && name != "int" { assert!(spills >= keys.len() / BLOCK_TERMS - 1, "{name}: {spills} spills"); }
                if threshold == 1 && name == "int" { assert!(spills >= keys.len() / INT_BLOCK_TERMS, "{name}: {spills} spills"); }
                spilled.extend_from_slice(&w.finish(5000));
                assert_eq!(spilled, whole, "{name} threshold {threshold}");
            }
        }
    }

    /// Compact entries (v17) read exactly as tagged ones: lookups (hits and
    /// misses), sorted batches, cursors, ranges and per-term bytes; inline
    /// and listed locations mixed, ids sharing long prefixes; smaller.
    #[test]
    fn compact_entries_read_like_tagged_ones() {
        // 20-character time-ordered ids (as ES auto ids), a few listed terms.
        let keys: Vec<Vec<u8>> = (0..5000u32).map(|i| Term::new_text(0, &format!("AZ{:06}x{:04}Q{:06}", 100_000 + i * 3, i % 7919, i * 31 % 100_000)).key()).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        let mut off = 0u64;
        let locs: Vec<PostingsLocation> = sorted.iter().enumerate().map(|(i, _)| {
            if i % 9 == 4 {
                off += 37 + i as u64;
                PostingsLocation { offset: off, doc_freq: 3 + i as u32 % 5, total_term_freq: 9 }
            } else {
                PostingsLocation { offset: INLINE_POSTINGS | ((i as u64 * 7919) % 100_003), doc_freq: 1, total_term_freq: 1 }
            }
        }).collect();
        let build = |compact: bool| {
            let mut w = TermDictWriter::new().tagged(true).compact(compact);
            for (k, l) in sorted.iter().zip(&locs) { w.add(k, l); }
            w.finish(100_003)
        };
        let (tagged, compact) = (build(false), build(true));
        assert!(compact.len() * 10 < tagged.len() * 9, "compact {} vs tagged {}", compact.len(), tagged.len());
        let a = TermDictReader::open_tagged(FileSlice::from_vec(tagged), true).unwrap();
        let b = TermDictReader::open_layout(FileSlice::from_vec(compact), true, true).unwrap();
        for (k, l) in sorted.iter().zip(&locs) {
            assert_eq!(b.lookup(k).unwrap(), Some(*l));
            assert_eq!(a.lookup(k).unwrap(), Some(*l));
            let mut miss = k.clone();
            miss.push(b'!');
            assert_eq!(b.lookup(&miss).unwrap(), None);
        }
        let texts: Vec<&[u8]> = sorted.iter().step_by(3).map(|k| &k[2..]).collect();
        let (mut x, mut y) = (Vec::new(), Vec::new());
        a.lookup_sorted(0, &texts, |i, l| x.push((i, l)));
        b.lookup_sorted(0, &texts, |i, l| y.push((i, l)));
        assert_eq!(x, y);
        assert_eq!(x.len(), texts.len());
        let mut c = b.cursor();
        let mut seen = Vec::new();
        while c.advance().unwrap() { seen.push((c.key().to_vec(), c.loc())); }
        assert_eq!(seen, sorted.iter().cloned().zip(locs.iter().copied()).collect::<Vec<_>>());
        assert_eq!(a.range(&sorted[100], Some(&sorted[900])).unwrap(), b.range(&sorted[100], Some(&sorted[900])).unwrap());
        let mut n = 0;
        b.for_each_term_bytes(|k, _, l| { assert_eq!((k, *l), (sorted[n].as_slice(), locs[n])); n += 1; }).unwrap();
        assert_eq!(n, sorted.len());
    }

    /// Packed blocks (v32) read exactly as compact ones: lookups (hits and
    /// misses, including keys between and beyond packed suffixes), sorted
    /// batches, cursors (from any key), ranges and per-term bytes; hex,
    /// base64, mixed-alphabet, long and multi-field keys, inline and listed
    /// locations; hex and base64 ids take far fewer bytes.
    #[test]
    fn packed_blocks_read_like_compact_ones() {
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for i in 0..3000u64 {
            let h = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            keys.push(Term::new_text(3, &format!("{:016x}", h)).key());
            keys.push(Term::new_text(0, &format!("AaG{}", &format!("{:016x}{:016X}", h, h ^ 0xabcdef)[..17]).replace('9', "_").replace('8', "-")).key());
            keys.push(Term::new_text(5, &format!("{:08x}-{:04x}", h >> 32, h & 0xffff)).key());
            if i % 5 == 0 {
                keys.push(Term::new_text(7, &format!("GET /english/images/{}.gif HTTP/1.0 {}", i, "x".repeat((i % 40) as usize))).key());
            }
            if i % 50 == 0 {
                keys.push(Term::new_text(9, &format!("{}", i)).key());
                keys.push(Term::new_text(9, "").key());
            }
        }
        keys.sort();
        keys.dedup();
        let mut off = 0u64;
        let locs: Vec<PostingsLocation> = keys.iter().enumerate().map(|(i, _)| {
            if i % 11 == 3 {
                off += 41 + i as u64;
                PostingsLocation { offset: off, doc_freq: 2 + i as u32 % 5, total_term_freq: 9 }
            } else {
                PostingsLocation { offset: INLINE_POSTINGS | ((i as u64 * 7919) % 100_003), doc_freq: 1, total_term_freq: 1 }
            }
        }).collect();
        let build = |packed: bool, keys: &[Vec<u8>], locs: &[PostingsLocation]| {
            let mut w = TermDictWriter::new().tagged(true).compact(true).packed(packed);
            for (k, l) in keys.iter().zip(locs) { w.add(k, l); }
            w.finish(100_003)
        };
        let (compact, packed) = (build(false, &keys, &locs), build(true, &keys, &locs));
        assert!(packed.len() * 10 < compact.len() * 8, "packed {} vs compact {}", packed.len(), compact.len());
        let a = TermDictReader::open_layout(FileSlice::from_vec(compact), true, true).unwrap();
        let mut b = TermDictReader::open_layout(FileSlice::from_vec(packed), true, true).unwrap();
        b.packed = true;
        for (k, l) in keys.iter().zip(&locs) {
            assert_eq!(b.lookup(k).unwrap(), Some(*l), "{k:?}");
            for tweak in [b"!".as_slice(), b"0", b"a", b"_", b"~"] {
                let mut miss = k.clone();
                miss.extend_from_slice(tweak);
                assert_eq!(b.lookup(&miss).unwrap(), a.lookup(&miss).unwrap(), "{miss:?}");
            }
            let mut short = k.clone();
            if short.len() > 3 { short.pop(); assert_eq!(b.lookup(&short).unwrap(), a.lookup(&short).unwrap(), "{short:?}"); }
        }
        for field in [0u16, 3, 5, 7, 9] {
            let texts: Vec<&[u8]> = keys.iter().filter(|k| k[..2] == field.to_le_bytes()).step_by(3).map(|k| &k[2..]).collect();
            let (mut x, mut y) = (Vec::new(), Vec::new());
            a.lookup_sorted(field, &texts, |i, l| x.push((i, l)));
            b.lookup_sorted(field, &texts, |i, l| y.push((i, l)));
            assert_eq!(x, y);
            assert_eq!(x.len(), texts.len());
        }
        let mut c = b.cursor();
        let mut seen = Vec::new();
        while c.advance().unwrap() { seen.push((c.key().to_vec(), c.loc())); }
        assert_eq!(seen, keys.iter().cloned().zip(locs.iter().copied()).collect::<Vec<_>>());
        for at in [0usize, 1, 31, 32, 33, 777, keys.len() - 1] {
            let (mut ca, mut cb) = (a.cursor_from(&keys[at]), b.cursor_from(&keys[at]));
            for _ in 0..70 {
                assert_eq!(ca.advance().unwrap(), cb.advance().unwrap());
                assert_eq!((ca.key(), ca.loc()), (cb.key(), cb.loc()));
            }
        }
        for (lo, hi) in [(10usize, 900usize), (0, 40), (2000, keys.len() - 1)] {
            assert_eq!(a.range(&keys[lo], Some(&keys[hi])).unwrap(), b.range(&keys[lo], Some(&keys[hi])).unwrap());
        }
        let (mut n, mut total) = (0, 0u64);
        let rest = b.for_each_term_bytes(|k, bytes, l| { assert_eq!((k, *l), (keys[n].as_slice(), locs[n])); n += 1; total += bytes; }).unwrap();
        assert_eq!(n, keys.len());
        assert_eq!(total + rest, b.len as u64, "every byte attributed");
        // Unique hex and base64 ids alone: about 4 / 6 bits a suffix byte.
        for field in [3u16, 0] {
            let ks: Vec<Vec<u8>> = keys.iter().filter(|k| k[..2] == field.to_le_bytes()).cloned().collect();
            let ls: Vec<PostingsLocation> = (0..ks.len()).map(|i| PostingsLocation { offset: INLINE_POSTINGS | i as u64, doc_freq: 1, total_term_freq: 1 }).collect();
            let (c, p) = (build(false, &ks, &ls).len(), build(true, &ks, &ls).len());
            assert!(p * 10 < c * if field == 3 { 7 } else { 9 }, "field {field}: packed {p} vs compact {c}");
        }
    }

    /// v34 integer-keyed blocks read exactly like packed ones: lookups and
    /// misses (other lengths, prefixes, characters outside the alphabet,
    /// alphabet extremes), sorted batches, cursors from any key (block
    /// edges at 32 and 128 terms), ranges and per-term bytes, over
    /// k-ordered and random base64 ids, hex hashes, keys of mixed lengths
    /// and other fields; k-ordered ids take well under half the bytes.
    #[test]
    fn int_key_blocks_read_like_packed_ones() {
        use base64::Engine;
        let k_ordered = |seq: u64, ts: u64| {
            let mut b = [0u8; 15];
            b[..4].copy_from_slice(&[(ts >> 40) as u8, (ts >> 32) as u8, (ts >> 24) as u8, (ts >> 16) as u8]);
            b[4..10].copy_from_slice(&[0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f]);
            b[10..].copy_from_slice(&[(seq >> 16) as u8, (ts >> 8) as u8, (seq >> 8) as u8, ts as u8, seq as u8]);
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
        };
        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut ids: Vec<Vec<u8>> = Vec::new();
        for i in 0..6000u64 {
            let h = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let id = Term::new_text(0, &k_ordered(i + 77_000, 894_225_602_000 + i / 3)).key();
            ids.push(id.clone());
            keys.push(id);
            keys.push(Term::new_text(2, &base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(h.to_le_bytes())).key());
            keys.push(Term::new_text(3, &format!("{:016x}", h)).key());
            // One field mixing lengths and alphabets mid-run.
            let t = match i % 7 { 0 => format!("{:06x}", i), 1 => format!("{:07x}", i), 2 => format!("x{:05}", i), _ => format!("{:06x}", i * 3 + 1) };
            keys.push(Term::new_text(4, &t).key());
            if i % 97 == 0 {
                for edge in ["------", "______", "zzzzzz", "000000", "ffffff", "AAAAAA"] {
                    keys.push(Term::new_text(5, &format!("{edge}{i}")).key());
                }
            }
        }
        keys.sort();
        keys.dedup();
        let mut off = 0u64;
        let locs: Vec<PostingsLocation> = keys.iter().enumerate().map(|(i, _)| {
            if i % 13 == 5 {
                off += 17 + i as u64;
                PostingsLocation { offset: off, doc_freq: 2 + i as u32 % 3, total_term_freq: 7 }
            } else {
                PostingsLocation { offset: INLINE_POSTINGS | ((i as u64 * 31) % 200_003), doc_freq: 1, total_term_freq: 1 }
            }
        }).collect();
        let build = |int: bool, keys: &[Vec<u8>], locs: &[PostingsLocation]| {
            let mut w = TermDictWriter::new().tagged(true).compact(true).packed(true).int_keys(int);
            for (k, l) in keys.iter().zip(locs) { w.add(k, l); }
            w.finish(200_003)
        };
        let open = |bytes: Vec<u8>| {
            let mut r = TermDictReader::open_layout(FileSlice::from_vec(bytes), true, true).unwrap();
            r.packed = true;
            r.positions = false;
            r
        };
        let (a, b) = (open(build(false, &keys, &locs)), open(build(true, &keys, &locs)));
        assert!(b.len < a.len, "int {} vs packed {}", b.len, a.len);
        let ints = (0..b.num_blocks).filter(|&i| b.block_bytes(i)[0] & 3 == ALPHA_INT).count();
        assert!(ints > 0 && ints < b.num_blocks, "{ints} of {} blocks", b.num_blocks);
        for (k, l) in keys.iter().zip(&locs) {
            assert_eq!(b.lookup(k).unwrap(), Some(*l), "{k:?}");
            let mut misses = Vec::new();
            for tweak in [b"!".as_slice(), b"0", b"-", b"_", b"z", b"~"] {
                let mut m = k.clone();
                m.extend_from_slice(tweak);
                misses.push(m);
            }
            let mut m = k.clone();
            m.pop();
            misses.push(m);
            for c in [b'-', b'0', b'Z', b'_', b'z', b'!', b'f'] {
                let mut m = k.clone();
                *m.last_mut().unwrap() = c;
                misses.push(m);
                let mut m = k.clone();
                if m.len() > 4 { m[4] = c; misses.push(m); }
            }
            for m in &misses {
                assert_eq!(b.lookup(m).unwrap(), a.lookup(m).unwrap(), "{m:?}");
            }
        }
        for field in [0u16, 2, 3, 4, 5] {
            let texts: Vec<&[u8]> = keys.iter().filter(|k| k[..2] == field.to_le_bytes()).step_by(2).map(|k| &k[2..]).collect();
            let (mut x, mut y) = (Vec::new(), Vec::new());
            a.lookup_sorted(field, &texts, |i, l| x.push((i, l)));
            b.lookup_sorted(field, &texts, |i, l| y.push((i, l)));
            assert_eq!(x, y);
            assert_eq!(x.len(), texts.len());
        }
        let mut c = b.cursor();
        let mut seen = Vec::new();
        while c.advance().unwrap() { seen.push((c.key().to_vec(), c.loc())); }
        assert_eq!(seen, keys.iter().cloned().zip(locs.iter().copied()).collect::<Vec<_>>());
        for at in [0usize, 1, 31, 32, 33, 127, 128, 129, 255, 256, 5000, keys.len() - 1] {
            // Block edges differ: both skip to the key first.
            let (mut ca, mut cb) = (a.cursor_from(&keys[at]), b.cursor_from(&keys[at]));
            for c in [&mut ca, &mut cb] {
                while c.advance().unwrap() && c.key() < keys[at].as_slice() {}
                assert_eq!((c.key(), c.loc()), (keys[at].as_slice(), locs[at]));
            }
            for _ in 0..300 {
                assert_eq!(ca.advance().unwrap(), cb.advance().unwrap());
                assert_eq!((ca.key(), ca.loc()), (cb.key(), cb.loc()));
            }
        }
        for (lo, hi) in [(10usize, 900usize), (0, 40), (5000, 9000), (20_000, keys.len() - 1)] {
            assert_eq!(a.range(&keys[lo], Some(&keys[hi])).unwrap(), b.range(&keys[lo], Some(&keys[hi])).unwrap());
        }
        let (mut n, mut total) = (0, 0u64);
        let rest = b.for_each_term_bytes(|k, bytes, l| { assert_eq!((k, *l), (keys[n].as_slice(), locs[n])); n += 1; total += bytes; }).unwrap();
        assert_eq!(n, keys.len());
        assert_eq!(total + rest, b.len as u64, "every byte attributed");
        // k-ordered ids alone, docs in id order.
        ids.sort();
        let ls: Vec<PostingsLocation> = (0..ids.len()).map(|i| PostingsLocation { offset: INLINE_POSTINGS | i as u64, doc_freq: 1, total_term_freq: 1 }).collect();
        let (p, q) = (build(false, &ids, &ls).len(), build(true, &ids, &ls).len());
        assert!(q * 10 < p * 7, "k-ordered ids: int {q} vs packed {p}");
        let r = open(build(true, &ids, &ls));
        for (k, l) in ids.iter().zip(&ls) {
            assert_eq!(r.lookup(k).unwrap(), Some(*l));
        }
        // Random ids keep packed blocks (a delta is no smaller).
        let rnd: Vec<Vec<u8>> = { let mut v: Vec<Vec<u8>> = keys.iter().filter(|k| k[..2] == 2u16.to_le_bytes()).cloned().collect(); v.sort(); v };
        let lr: Vec<PostingsLocation> = (0..rnd.len()).map(|i| PostingsLocation { offset: INLINE_POSTINGS | i as u64, doc_freq: 1, total_term_freq: 1 }).collect();
        assert!(build(true, &rnd, &lr).len() <= build(false, &rnd, &lr).len());
    }

    #[test]
    #[ignore = "EC2 large-dictionary regression; allocates more than 4 GiB"]
    fn dictionary_above_four_gib_round_trips() {
        const COUNT: u64 = 70_000;
        let key = |n: u64| {
            let mut k = vec![b'x'; 65_536];
            k[..8].copy_from_slice(&n.to_be_bytes());
            k
        };
        let mut w = TermDictWriter::with_capacity(4_900_000_000);
        for n in 0..COUNT { w.add(&key(n), &loc(n)); }
        let bytes = w.finish(COUNT as u32);
        assert!(bytes.len() > u32::MAX as usize);
        let r = TermDictReader::open(FileSlice::from_vec(bytes)).unwrap();
        assert_eq!(r.index_entry, WIDE_INDEX_ENTRY);
        for n in [0, 15, 16, 65_535, 65_536, COUNT - 1] {
            assert_eq!(r.lookup(&key(n)).unwrap(), Some(loc(n)));
        }
        assert_eq!(r.lookup(&key(COUNT)).unwrap(), None);
    }

    #[test]
    fn wide_dictionary_matches_legacy_lookups_and_iteration() {
        for count in [0, 1, 16, 17, 257] {
            let make = |wide| {
                let mut w = TermDictWriter::new();
                for n in 0..count {
                    w.add(&Term::new_text(7, &format!("shared-prefix-{n:06}")).key(), &loc(n as u64));
                }
                TermDictReader::open(FileSlice::from_vec(w.finish_with_width(1000, wide))).unwrap()
            };
            let narrow = make(false);
            let wide = make(true);
            assert_eq!(narrow.iter_all().unwrap(), wide.iter_all().unwrap());
            for n in 0..count + 1 {
                let term = Term::new_text(7, &format!("shared-prefix-{n:06}"));
                assert_eq!(narrow.get(&term), wide.get(&term));
            }
            let terms = [b"shared-prefix-000000".as_slice(), b"shared-prefix-000016", b"shared-prefix-000256"];
            let mut a = Vec::new(); let mut b = Vec::new();
            narrow.lookup_sorted(7, &terms, |i, p| a.push((i, p)));
            wide.lookup_sorted(7, &terms, |i, p| b.push((i, p)));
            assert_eq!(a, b);
        }
        let mut bytes = Vec::new();
        write_index_offsets(&mut bytes, u32::MAX as usize + 123, u32::MAX as usize + 456, true);
        assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), u32::MAX as u64 + 123);
        assert_eq!(u64::from_le_bytes(bytes[8..].try_into().unwrap()), u32::MAX as u64 + 456);
    }

    #[test]
    fn lookup_sorted_matches_point_lookups() {
        // Two fields; long shared prefixes make runs of blocks that tie on
        // the 8-byte prefix.
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for f in [1u16, 2] {
            for i in 0..3000 {
                let text = if i % 3 == 0 { format!("same-prefix-{i:05}") } else { format!("t{:05}", i * 7) };
                keys.push(Term::new_text(f, &text).key());
            }
        }
        keys.sort();
        let mut w = TermDictWriter::new();
        for (i, k) in keys.iter().enumerate() {
            w.add(k, &loc(i as u64));
        }
        let d = TermDictReader::open(FileSlice::from_vec(w.finish(1000))).unwrap();
        let mut texts: Vec<Vec<u8>> = (0..9000).step_by(5).map(|i| format!("same-prefix-{i:05}").into_bytes()).collect();
        texts.extend((0..25000).step_by(3).map(|i| format!("t{i:05}").into_bytes()));
        texts.extend([b"".to_vec(), b"a".to_vec(), b"same".to_vec(), b"t00007".to_vec(), b"zzz".to_vec()]);
        texts.sort();
        let refs: Vec<&[u8]> = texts.iter().map(|t| t.as_slice()).collect();
        for f in [0u16, 1, 2, 3] {
            let mut got = Vec::new();
            d.lookup_sorted(f, &refs, |i, l| got.push((i, l.offset)));
            let want: Vec<(usize, u64)> = refs.iter().enumerate().filter_map(|(i, t)| d.get_parts(f, t).map(|l| (i, l.offset))).collect();
            assert_eq!(got, want, "field {f}");
            assert_eq!(!want.is_empty(), f == 1 || f == 2);
        }
    }

    #[test]
    fn unique_field_filter_never_hides_terms_across_field_boundaries() {
        let fields = [0, 1, 255, 256, 265, 65535];
        let mut terms = Vec::new();
        for field in fields {
            terms.push(Term::new_text(field, ""));
            for i in 0..2048 {
                terms.push(Term::new_text(field, &format!("common-prefix-{}-{i:06}", "x".repeat(i % 80))));
            }
        }
        let mut keys: Vec<_> = terms.iter().map(Term::key).collect();
        keys.sort();
        let mut writer = TermDictWriter::new();
        for (i, key) in keys.iter().enumerate() { writer.add(key, &loc(i as u64)); }
        let bytes = writer.finish(2049);
        for field in fields {
            let reader = TermDictReader::open(FileSlice::from_vec(bytes.clone())).unwrap();
            assert!(reader.negative_filter.get().is_none());
            for term in terms.iter().filter(|t| t.field == field) {
                assert!(reader.may_contain_unique_term(term), "{term:?}");
                assert!(reader.get(term).is_some());
            }
            // A cache built for one field never rejects a different field.
            assert!(terms.iter().all(|term| reader.may_contain_unique_term(term)));
            let rejected = (0..2048).filter(|i| !reader.may_contain_unique_term(&Term::new_text(field, &format!("absent-{i}")))).count();
            assert!(rejected > 1800, "negative lookup filter is ineffective: {rejected}");
        }
        let empty = TermDictReader::empty(0);
        assert!(!empty.may_contain_unique_term(&Term::new_text(65535, "missing")));
    }

    #[test]
    fn lookup_hits_and_misses() {
        let r = build(1000);
        assert_eq!(r.num_terms(), 1000);
        assert_eq!(r.num_docs(), 1000);
        for i in [0usize, 1, 31, 32, 33, 500, 998, 999] {
            let t = Term::new_text(0, &format!("term_{i:06}"));
            assert_eq!(r.get(&t), Some(loc(i as u64)), "term {i}");
        }
        assert!(r.get(&Term::new_text(0, "term_001000")).is_none());
        assert!(r.get(&Term::new_text(0, "aaa")).is_none());
        assert!(r.get(&Term::new_text(0, "zzz")).is_none());
        assert!(r.get(&Term::new_text(1, "term_000000")).is_none());
    }

    #[test]
    fn range_queries() {
        let r = build(100);
        let got = r.range_iter(0, Some(b"term_000010"), Some(b"term_000020"));
        assert_eq!(got.len(), 10);
        assert_eq!(got[0].0, Term::new_text(0, "term_000010").key());
        assert_eq!(r.range_iter(0, None, None).len(), 100);
        assert_eq!(r.range_iter(1, None, None).len(), 0);
        assert_eq!(r.range_iter(0, Some(b"term_000095"), None).len(), 5);
        assert_eq!(r.iter_all().unwrap().len(), 100);
    }

    #[test]
    fn lookup_matches_sorted_set_on_prefix_heavy_keys() {
        // Keys that are prefixes of each other, share long prefixes (ties in
        // the 8-byte block prefix) and straddle block boundaries.
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        let mut set = std::collections::BTreeSet::new();
        for _ in 0..3000 {
            let len = rnd() % 12;
            let k: Vec<u8> = (0..len).map(|_| b"aab\x00\xff"[rnd() % 5]).collect();
            set.insert(k);
        }
        let mut w = TermDictWriter::new();
        let keys: Vec<Vec<u8>> = set.iter().cloned().collect();
        for (i, k) in keys.iter().enumerate() {
            w.add(k, &loc(i as u64));
        }
        let r = TermDictReader::open(FileSlice::from_vec(w.finish(1))).unwrap();
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(r.lookup(k).unwrap(), Some(loc(i as u64)), "{k:?}");
        }
        for _ in 0..5000 {
            let len = rnd() % 13;
            let k: Vec<u8> = (0..len).map(|_| b"aabc\x00\xff\x01"[rnd() % 7]).collect();
            let want = keys.binary_search(&k).ok().map(|i| loc(i as u64));
            assert_eq!(r.lookup(&k).unwrap(), want, "{k:?}");
        }
    }

    #[test]
    fn cursor_streams_every_term() {
        let r = build(1000);
        let all = r.iter_all().unwrap();
        let mut c = r.cursor();
        let mut i = 0;
        while c.advance().unwrap() {
            assert_eq!(c.key(), all[i].0.as_slice());
            assert_eq!(c.loc(), all[i].1);
            i += 1;
        }
        assert_eq!(i, 1000);
        assert!(!TermDictReader::empty(1).cursor().advance().unwrap());
    }

    #[test]
    fn long_keys_and_prefix_ties() {
        // Keys longer than 127 bytes (multi-byte vints) sharing 8-byte prefixes.
        let mut keys: Vec<Vec<u8>> = (0..500u32)
            .map(|i| {
                let mut k = b"samepref".to_vec();
                k.extend(std::iter::repeat(b'x').take((i % 7) as usize * 40));
                k.extend_from_slice(format!("{i:05}").as_bytes());
                k
            })
            .collect();
        keys.sort();
        keys.dedup();
        let mut w = TermDictWriter::new();
        for (i, k) in keys.iter().enumerate() {
            w.add(k, &loc(i as u64));
        }
        let r = TermDictReader::open(FileSlice::from_vec(w.finish(1))).unwrap();
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(r.lookup(k).unwrap(), Some(loc(i as u64)));
        }
        assert_eq!(r.lookup(b"samepref").unwrap(), None);
        assert_eq!(r.lookup(b"samepreg").unwrap(), None);
        assert_eq!(r.iter_all().unwrap().len(), keys.len());
    }

    #[test]
    fn empty_dict() {
        let r = TermDictReader::empty(7);
        assert_eq!(r.num_terms(), 0);
        assert_eq!(r.num_docs(), 7);
        assert!(r.get(&Term::new_text(0, "x")).is_none());
        assert!(r.iter_all().unwrap().is_empty());
    }

    #[test]
    fn multiple_fields_sorted_by_key() {
        let mut keys: Vec<(Vec<u8>, u64)> = Vec::new();
        for (i, (f, t)) in [(0u16, "b"), (0, "a"), (1, "a"), (2, "zz"), (1, "c")].iter().enumerate() {
            keys.push((Term::new_text(*f, t).key(), i as u64));
        }
        keys.sort();
        let mut w = TermDictWriter::new();
        for (k, i) in &keys {
            w.add(k, &loc(*i));
        }
        let r = TermDictReader::open(FileSlice::from_vec(w.finish(5))).unwrap();
        assert_eq!(r.get(&Term::new_text(1, "c")).unwrap().offset, 40);
        assert_eq!(r.range_iter(1, None, None).len(), 2);
        assert_eq!(r.range_iter(0, None, None).len(), 2);
        assert_eq!(r.range_iter(2, None, None).len(), 1);
    }
}
