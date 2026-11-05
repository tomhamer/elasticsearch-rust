//! Synthetic document ids (ES `index.mapping.synthetic_id`, time series
//! indices): a document's `_id` is a function of its `_tsid`, `@timestamp`
//! and routing hash, so sealed segments keep neither `_id` postings nor a
//! stored `_id`. A term of the id field resolves through the `_tsid` column
//! (segments of such a core are sorted by `_tsid` ascending, then
//! `@timestamp` descending) and the timestamp column; a stored document gets
//! its `_id` back from the three columns. Memtables keep their in-memory id
//! postings (realtime lookups, the id filter); only segments drop them.
//!
//! The id layout belongs to the layer that builds ids: it installs a
//! [`Codec`] once ([`install`]).

use serde::{Deserialize, Serialize};

/// The columns an id field is derived from (core field ids).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticId {
    /// Keyword column holding the `_tsid` (sorted segment order's first key).
    pub tsid: u16,
    /// i64 column of the timestamp (second sort key, descending).
    pub timestamp: u16,
    /// i64 column of the routing hash.
    pub routing_hash: u16,
}

/// How ids map to and from their columns.
pub struct Codec {
    /// An id's `_tsid` column value and timestamp (`None`: not such an id).
    pub decode: fn(&[u8]) -> Option<(Vec<u8>, i64)>,
    /// The id of a document from its `_tsid` column value, timestamp and
    /// routing hash.
    pub encode: fn(&[u8], i64, i32) -> Option<String>,
}

static CODEC: std::sync::OnceLock<Codec> = std::sync::OnceLock::new();

/// Install the id codec (the first call wins; later ones are ignored).
pub fn install(codec: Codec) {
    let _ = CODEC.set(codec);
}

pub(crate) fn codec() -> Option<&'static Codec> {
    CODEC.get()
}

/// The `[extra]` payload of a segment's synthetic id field
/// ([`crate::codec::EXTRA_SYNTHETIC_ID`]): the three column ids.
pub(crate) fn encode_extra(s: &SyntheticId) -> Vec<u8> {
    let mut b = Vec::with_capacity(6);
    for f in [s.tsid, s.timestamp, s.routing_hash] {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b
}

pub(crate) fn decode_extra(b: &[u8]) -> Option<SyntheticId> {
    let f = |i: usize| Some(u16::from_le_bytes(b.get(i * 2..i * 2 + 2)?.try_into().ok()?));
    Some(SyntheticId { tsid: f(0)?, timestamp: f(1)?, routing_hash: f(2)? })
}
