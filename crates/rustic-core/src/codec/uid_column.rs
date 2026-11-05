//! A segment's document `_id`s as a column (format
//! [`crate::codec::UID_COLUMN_SINCE`]+, [`crate::codec::EXTRA_UID_COLUMN`]):
//! ids in their [`crate::uid`] binary form, in doc order, in blocks of
//! [`BLOCK`] docs. Auto ids of an index come in time order, so consecutive
//! ids differ in their last bytes: read as big-endian integers (after a
//! byte permutation for ES's default auto-id layout, whose varying bytes
//! come first), the delta from the previous doc's id takes about a byte,
//! where the binary id in a stored document took about four after LZ4.
//!
//! Layout: `u8 order` ([`ORDER_IDENTITY`] or [`ORDER_AUTO_ID`]), `u32 docs`,
//! `u32 blocks`, `blocks + 1` u32 offsets (relative to the first block),
//! then the blocks. A block's entries, one a doc, are a vint each (128-bit):
//! `0` the doc has no id (a nested doc), odd `2 * len + 1` an id of `len`
//! bytes that follow, even `2 * (zigzag(delta) + 1)` an id of the previous
//! id's length (at most 16 bytes) whose integer is the previous id's plus
//! `delta`. A block's first id is always given whole.

use crate::error::{Result, RusticError};
use crate::types::DocId;

/// Docs a block holds: reading a doc's id decodes at most this many entries.
pub const BLOCK: usize = 256;
/// Ids read as they are.
pub const ORDER_IDENTITY: u8 = 0;
/// 15-byte ids read with ES's auto-id byte layout
/// (`TimeBasedUUIDGenerator`) permuted into the k-ordered one (timestamp
/// high bytes, node address, then the varying bytes).
pub const ORDER_AUTO_ID: u8 = 1;
/// The [`ORDER_AUTO_ID`] permutation: integer byte `i` is id byte `PERM[i]`.
const PERM: [usize; 15] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 12, 13, 14, 0];

fn zigzag(v: i128) -> u128 {
    ((v << 1) ^ (v >> 127)) as u128
}

fn unzigzag(v: u128) -> i128 {
    ((v >> 1) as i128) ^ -((v & 1) as i128)
}

fn write_vint128(mut v: u128, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn vint128_len(mut v: u128) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

#[inline]
fn read_vint128(b: &[u8], pos: &mut usize) -> Result<u128> {
    // Most entries are a byte or two (deltas of k-ordered ids, short
    // lengths): decoded in 64 bits, the 128-bit loop only past 63 bits.
    let start = *pos;
    let mut v = 0u64;
    let mut shift = 0u32;
    while shift < 63 {
        let Some(&x) = b.get(*pos) else { break };
        *pos += 1;
        v |= ((x & 0x7f) as u64) << shift;
        if x & 0x80 == 0 {
            return Ok(v as u128);
        }
        shift += 7;
    }
    *pos = start;
    read_vint128_wide(b, pos)
}

fn read_vint128_wide(b: &[u8], pos: &mut usize) -> Result<u128> {
    let (mut v, mut shift) = (0u128, 0u32);
    loop {
        let x = *b.get(*pos).ok_or_else(|| RusticError::Corrupted("uid column: truncated entry".into()))?;
        *pos += 1;
        if shift > 126 {
            return Err(RusticError::Corrupted("uid column: entry too long".into()));
        }
        v |= ((x & 0x7f) as u128) << shift;
        if x & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

/// The integer of `id` under `order` (ids of at most 16 bytes).
fn int_of(order: u8, id: &[u8]) -> u128 {
    let mut v = 0u128;
    if order == ORDER_AUTO_ID && id.len() == 15 {
        for &i in &PERM {
            v = (v << 8) | id[i] as u128;
        }
    } else {
        for &b in id {
            v = (v << 8) | b as u128;
        }
    }
    v
}

/// Write the `len`-byte id of integer `v` under `order` into `out`.
fn id_of(order: u8, v: u128, len: usize, out: &mut Vec<u8>) {
    out.clear();
    if order != ORDER_AUTO_ID && len <= 8 {
        // The common short id (numeric `_id`s): its bytes are `v`'s low
        // `len` bytes, big-endian.
        out.extend_from_slice(&(v as u64).to_be_bytes()[8 - len..]);
        return;
    }
    out.resize(len, 0);
    let byte = |k: usize| (v >> (8 * (len - 1 - k))) as u8;
    if order == ORDER_AUTO_ID && len == 15 {
        for (k, &i) in PERM.iter().enumerate() {
            out[i] = byte(k);
        }
    } else {
        for (k, b) in out.iter_mut().enumerate() {
            *b = byte(k);
        }
    }
}

/// The entries of `ids` under `order`, appended to `out` (with the block
/// offsets in `offsets`); returns the entries' size.
fn encode_with(order: u8, ids: &[Option<&[u8]>], offsets: &mut Vec<u32>, out: &mut Vec<u8>) -> usize {
    let start = out.len();
    offsets.clear();
    for block in ids.chunks(BLOCK) {
        offsets.push((out.len() - start) as u32);
        let mut prev: Option<&[u8]> = None;
        for id in block {
            match (id, prev) {
                (None, _) => out.push(0),
                (Some(id), Some(p)) if p.len() == id.len() && id.len() <= 16 => {
                    let delta = int_of(order, id).wrapping_sub(int_of(order, p)) as i128;
                    write_vint128((zigzag(delta) + 1) << 1, out);
                    prev = Some(id);
                }
                (Some(id), _) => {
                    write_vint128(((id.len() as u128) << 1) | 1, out);
                    out.extend_from_slice(id);
                    prev = Some(id);
                }
            }
        }
    }
    offsets.push((out.len() - start) as u32);
    out.len() - start
}

/// The size of the entries of `ids` under `order`.
fn size_with(order: u8, ids: &[Option<&[u8]>]) -> usize {
    let mut n = 0;
    for block in ids.chunks(BLOCK) {
        let mut prev: Option<&[u8]> = None;
        for id in block {
            n += match (id, prev) {
                (None, _) => 1,
                (Some(id), Some(p)) if p.len() == id.len() && id.len() <= 16 => {
                    let delta = int_of(order, id).wrapping_sub(int_of(order, p)) as i128;
                    vint128_len((zigzag(delta) + 1) << 1)
                }
                (Some(id), _) => vint128_len(((id.len() as u128) << 1) | 1) + id.len(),
            };
            if id.is_some() {
                prev = *id;
            }
        }
    }
    n
}

/// Ids gathered in doc order for [`encode`] without an allocation each:
/// their bytes back to back, per doc its end and whether it has one.
#[derive(Default)]
pub struct IdList {
    bytes: Vec<u8>,
    ends: Vec<u32>,
    has: Vec<bool>,
}

impl IdList {
    pub fn push(&mut self, id: Option<&[u8]>) {
        if let Some(id) = id {
            self.bytes.extend_from_slice(id);
        }
        self.ends.push(self.bytes.len() as u32);
        self.has.push(id.is_some());
    }
    pub fn len(&self) -> usize {
        self.ends.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }
    /// Whether any doc has an id.
    pub fn any(&self) -> bool {
        self.has.iter().any(|&h| h)
    }
    /// Bytes the list holds.
    pub fn heap_bytes(&self) -> usize {
        self.bytes.capacity() + self.ends.capacity() * 4 + self.has.capacity()
    }
    /// Doc `i`'s id.
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        if !*self.has.get(i)? {
            return None;
        }
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        Some(&self.bytes[start..self.ends[i] as usize])
    }
    /// Append `other`'s ids.
    pub fn extend(&mut self, other: &IdList) {
        for i in 0..other.len() {
            self.push(other.get(i));
        }
    }
    /// The ids in order (`None`: no id).
    pub fn refs(&self) -> Vec<Option<&[u8]>> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }
}

/// Reads ids of a column by doc, a decoded block at a time (merges read
/// their inputs' docs mostly in order).
pub struct Cursor<'a> {
    col: UidColumn<'a>,
    block: usize,
    ids: IdList,
}

impl<'a> Cursor<'a> {
    pub fn new(col: UidColumn<'a>) -> Self {
        Cursor { col, block: usize::MAX, ids: IdList::default() }
    }
    /// The decoded block (of the last [`Self::get`]), by position in it.
    pub fn into_block(self) -> IdList {
        self.ids
    }

    /// The id of `doc` (binary form).
    pub fn get(&mut self, doc: DocId) -> Result<Option<&[u8]>> {
        let b = doc as usize / BLOCK;
        if b != self.block {
            let mut ids = IdList::default();
            let (from, to) = ((b * BLOCK) as DocId, ((b + 1) * BLOCK).min(self.col.docs) as DocId);
            let mut next = from;
            self.col.for_range(from, to, |d, id| {
                while next < d {
                    ids.push(None);
                    next += 1;
                }
                ids.push(Some(id));
                next = d + 1;
            })?;
            while next < to {
                ids.push(None);
                next += 1;
            }
            self.ids = ids;
            self.block = b;
        }
        Ok(self.ids.get(doc as usize % BLOCK))
    }
}

/// The column of `ids` (binary form, doc order; `None`: no id), in the
/// order that takes fewer bytes.
pub fn encode(ids: &[Option<&[u8]>]) -> Vec<u8> {
    let auto = ids.iter().flatten().any(|id| id.len() == 15);
    let order = if auto && size_with(ORDER_AUTO_ID, ids) < size_with(ORDER_IDENTITY, ids) { ORDER_AUTO_ID } else { ORDER_IDENTITY };
    let mut body = Vec::with_capacity(ids.len() * 2);
    let mut offsets = Vec::new();
    encode_with(order, ids, &mut offsets, &mut body);
    let mut out = Vec::with_capacity(9 + offsets.len() * 4 + body.len());
    out.push(order);
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    out.extend_from_slice(&((offsets.len() - 1) as u32).to_le_bytes());
    for o in &offsets {
        out.extend_from_slice(&o.to_le_bytes());
    }
    out.extend_from_slice(&body);
    out
}

/// A read-only view of an encoded column.
#[derive(Clone, Copy, Debug)]
pub struct UidColumn<'a> {
    order: u8,
    docs: usize,
    offsets: &'a [u8],
    body: &'a [u8],
}

impl<'a> UidColumn<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let bad = || RusticError::Corrupted("uid column header".into());
        let order = *data.first().ok_or_else(bad)?;
        let u32_at = |at: usize| data.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).ok_or_else(bad);
        let docs = u32_at(1)?;
        let blocks = u32_at(5)?;
        if order > ORDER_AUTO_ID || blocks != docs.div_ceil(BLOCK) {
            return Err(bad());
        }
        let offsets = data.get(9..9 + (blocks + 1) * 4).ok_or_else(bad)?;
        let body = &data[9 + (blocks + 1) * 4..];
        let col = UidColumn { order, docs, offsets, body };
        if col.offset(blocks) != body.len() || (0..blocks).any(|b| col.offset(b) > col.offset(b + 1)) {
            return Err(bad());
        }
        Ok(col)
    }

    /// A view of a column [`Self::open`] checked before (the header only).
    pub fn open_unchecked(data: &'a [u8]) -> Option<Self> {
        let order = *data.first()?;
        let docs = u32::from_le_bytes(data.get(1..5)?.try_into().ok()?) as usize;
        let blocks = u32::from_le_bytes(data.get(5..9)?.try_into().ok()?) as usize;
        let offsets = data.get(9..9 + (blocks + 1) * 4)?;
        Some(UidColumn { order, docs, offsets, body: &data[9 + (blocks + 1) * 4..] })
    }

    /// Docs the column covers.
    pub fn num_docs(&self) -> usize {
        self.docs
    }

    fn offset(&self, b: usize) -> usize {
        u32::from_le_bytes(self.offsets[b * 4..b * 4 + 4].try_into().unwrap()) as usize
    }

    /// `f(doc, id)` for each doc of `[from, to)` (ids in binary form; docs
    /// without one skipped), decoding each block from its start once.
    pub fn for_range(&self, from: DocId, to: DocId, mut f: impl FnMut(DocId, &[u8])) -> Result<()> {
        let (from, to) = (from as usize, (to as usize).min(self.docs));
        let mut id: Vec<u8> = Vec::with_capacity(20);
        let mut b = from / BLOCK;
        while b * BLOCK < to {
            let end = self.offset(b + 1);
            let body = &self.body[..end];
            let mut pos = self.offset(b);
            // The current id: its integer and length (deltas step the
            // integer; its bytes are written only for docs asked for).
            let (mut cur, mut len, mut have, mut stale) = (0u128, 0usize, false, false);
            for d in b * BLOCK..((b + 1) * BLOCK).min(to) {
                let tag = read_vint128(body, &mut pos)?;
                if tag == 0 {
                    continue;
                }
                if tag & 1 == 1 {
                    len = (tag >> 1) as usize;
                    let bytes = body.get(pos..pos + len).ok_or_else(|| RusticError::Corrupted("uid column: id out of range".into()))?;
                    pos += len;
                    if d >= from {
                        id.clear();
                        id.extend_from_slice(bytes);
                        stale = false;
                    } else {
                        stale = true;
                    }
                    if len <= 16 {
                        cur = int_of(self.order, bytes);
                    }
                } else {
                    if !have || len > 16 {
                        return Err(RusticError::Corrupted("uid column: delta without an id".into()));
                    }
                    cur = cur.wrapping_add(unzigzag((tag >> 1) - 1) as u128);
                    stale = true;
                }
                have = true;
                if d >= from {
                    if stale {
                        id_of(self.order, cur, len, &mut id);
                        stale = false;
                    }
                    f(d as DocId, &id);
                }
            }
            b += 1;
        }
        Ok(())
    }

    /// The id of `doc` in binary form (`None`: none, or out of range).
    pub fn get(&self, doc: DocId) -> Option<Vec<u8>> {
        let mut out = None;
        self.for_range(doc, doc + 1, |_, id| out = Some(id.to_vec())).ok()?;
        out
    }

    /// The bytes the column takes.
    pub fn byte_len(&self) -> usize {
        9 + self.offsets.len() + self.body.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn roundtrip(ids: &[Option<Vec<u8>>]) -> Vec<u8> {
        let refs: Vec<Option<&[u8]>> = ids.iter().map(|x| x.as_deref()).collect();
        let bytes = encode(&refs);
        let col = UidColumn::open(&bytes).unwrap();
        assert_eq!(col.num_docs(), ids.len());
        for (d, want) in ids.iter().enumerate() {
            assert_eq!(col.get(d as DocId), *want, "doc {d}");
        }
        let mut seen = vec![None; ids.len()];
        col.for_range(0, ids.len() as DocId, |d, id| seen[d as usize] = Some(id.to_vec())).unwrap();
        assert_eq!(seen, ids);
        for (a, b) in [(3usize, 700usize), (255, 257), (0, 1)] {
            let mut got = Vec::new();
            col.for_range(a as DocId, b as DocId, |d, id| got.push((d, id.to_vec()))).unwrap();
            let want: Vec<(DocId, Vec<u8>)> = (a..b.min(ids.len())).filter_map(|d| ids[d].clone().map(|i| (d as DocId, i))).collect();
            assert_eq!(got, want);
        }
        bytes
    }

    fn auto_id(seq: u64, ts: u64, k_ordered: bool) -> Vec<u8> {
        let mut b = [0u8; 15];
        let addr = [0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f];
        if k_ordered {
            b[..4].copy_from_slice(&[(ts >> 40) as u8, (ts >> 32) as u8, (ts >> 24) as u8, (ts >> 16) as u8]);
            b[4..10].copy_from_slice(&addr);
            b[10..].copy_from_slice(&[(seq >> 16) as u8, (ts >> 8) as u8, (seq >> 8) as u8, ts as u8, seq as u8]);
        } else {
            b[0] = seq as u8;
            b[1] = (seq >> 16) as u8;
            b[2..6].copy_from_slice(&[(ts >> 16) as u8, (ts >> 24) as u8, (ts >> 32) as u8, (ts >> 40) as u8]);
            b[6..12].copy_from_slice(&addr);
            b[12..].copy_from_slice(&[(ts >> 8) as u8, (seq >> 8) as u8, ts as u8]);
        }
        // Through the base64 string, as `crate::uid::encode` reads it.
        crate::uid::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b))
    }

    /// Ids of every kind read back exactly, by doc and by range: k-ordered
    /// and ES default auto ids (each about a byte or two a doc, in the
    /// order that suits them), numeric and other ids, ids of other lengths
    /// within a run, docs without ids, partial blocks.
    #[test]
    fn ids_round_trip_and_auto_ids_take_a_byte_or_two() {
        let n = 5000u64;
        for k_ordered in [true, false] {
            let ids: Vec<Option<Vec<u8>>> = (0..n).map(|i| Some(auto_id(70_000 + i, 1_700_000_000_000 + i * 3, k_ordered))).collect();
            let bytes = roundtrip(&ids);
            assert!(bytes.len() * 2 < 5 * n as usize, "k_ordered {k_ordered}: {} bytes", bytes.len());
            assert_eq!(bytes[0], if k_ordered { ORDER_IDENTITY } else { ORDER_AUTO_ID });
        }
        let mut mixed: Vec<Option<Vec<u8>>> = Vec::new();
        for i in 0..3000u64 {
            mixed.push(match i % 13 {
                0 => None,
                1 => Some(crate::uid::encode(&format!("{}", i * 7))),
                2 => Some(crate::uid::encode(&format!("doc-{i}-é"))),
                3 => Some(crate::uid::encode(&"x".repeat(40 + (i % 5) as usize))),
                4 => Some(vec![0xfd; 16]),
                _ => Some(auto_id(i, 1_700_000_000_000 + i, true)),
            });
        }
        mixed.push(None);
        roundtrip(&mixed);
        roundtrip(&[]);
        roundtrip(&[None, None]);
        roundtrip(&[Some(vec![]), Some(vec![1]), Some(vec![0xff; 16]), Some(vec![0; 16]), Some(vec![0xff; 16])]);
    }

    #[test]
    fn corrupt_columns_are_refused_or_read_as_missing() {
        let ids: Vec<Option<Vec<u8>>> = (0..600u64).map(|i| Some(auto_id(i, 1_700_000_000_000 + i, true))).collect();
        let refs: Vec<Option<&[u8]>> = ids.iter().map(|x| x.as_deref()).collect();
        let bytes = encode(&refs);
        assert!(UidColumn::open(&bytes[..5]).is_err());
        let mut bad = bytes.clone();
        bad[0] = 9;
        assert!(UidColumn::open(&bad).is_err());
        assert!(UidColumn::open(&bytes[..bytes.len() - 1]).is_err());
        // Flipped entry bytes never panic.
        for at in (9..bytes.len()).step_by(7) {
            let mut b = bytes.clone();
            b[at] ^= 0x5a;
            if let Ok(c) = UidColumn::open(&b) {
                for d in 0..600 {
                    let _ = c.get(d);
                }
            }
        }
    }
}
