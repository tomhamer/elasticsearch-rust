//! Encodings and query primitives behind Elasticsearch's specialised field
//! types (Lucene's `InetAddressPoint`, `RangeFieldMapper` encodings,
//! `FeatureField`, `CompletionPostingsFormat`, `MurmurHash3`, ...).
//!
//! Everything here is a pure function or a self-contained structure over
//! the core's existing primitives (keyword terms, multi-valued doc values,
//! stored fields); the Elasticsearch layer composes them into field types.
//! [`filter::FilterQuery`] is the generic "candidates + per-doc check"
//! query node used for doc-value verification (wildcard, ranges, shapes,
//! percolator, ...).

pub mod completion;
pub mod features;
pub mod filter;
pub mod flattened;
pub mod histogram;
pub mod ip;
pub mod murmur3;
pub mod ngram;
pub mod numeric;
pub mod range;
pub mod range_query;
pub mod geo2d;
pub mod shape;
pub mod shape_bkd;
pub mod shape_dv;
pub mod shape_field;
pub mod tessellator;
pub mod version;

/// Lower-case hex of `bytes`.
pub fn to_hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 15) as usize] as char);
    }
    s
}

/// Inverse of [`to_hex`].
pub fn from_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}
