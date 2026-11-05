//! Memo of fan-out merges. A coordinator's merge of shard search responses
//! (`fanout::merge_search_with_source`: parse each part, re-reduce the
//! aggregations, render) is a pure function of the request (URI query,
//! body, the bucket limit) and the shards' responses in shard order. A
//! repeated `size: 0` aggregation whose shards answer from their request
//! caches gets the same responses (only `took` differs) and so the same
//! merge: it is reused instead of being computed again (ES's coordinator
//! re-reduces, but from binary shard results, far cheaper than parsing and
//! re-rendering JSON). The key holds every input byte for byte, so a reuse
//! is exact; any change in any shard's answer is a different key.

use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

/// Total bytes of keys and merged responses held.
const MAX_BYTES: usize = 16 << 20;
/// Inputs (key bytes) above this are not memoized.
const MAX_KEY: usize = 1 << 20;

#[derive(Default)]
struct Memo {
    map: HashMap<Vec<u8>, Vec<u8>>,
    order: VecDeque<Vec<u8>>,
    bytes: usize,
}

static MEMO: LazyLock<Mutex<Memo>> = LazyLock::new(Default::default);

/// A shard response without its leading `"took":N` (the one field that
/// differs between two answers from a shard's request cache).
fn without_took(body: &[u8]) -> &[u8] {
    const PREFIX: &[u8] = b"{\"took\":";
    if let Some(rest) = body.strip_prefix(PREFIX) {
        let digits = rest.iter().take_while(|b| b.is_ascii_digit()).count();
        return &rest[digits..];
    }
    body
}

/// The memo key of a merge: the URI query, the body, the bucket limit and
/// each part (shard number, answering node, response without `took`), in
/// shard order. `None` when too large to keep.
pub(super) fn key(query: &str, body: &[u8], max_buckets: usize, parts: &mut Vec<(u32, &str, &[u8])>) -> Option<Vec<u8>> {
    parts.sort_by_key(|(k, n, _)| (*k, n.to_string()));
    let len = query.len() + body.len() + parts.iter().map(|(_, n, b)| n.len() + b.len() + 16).sum::<usize>() + 32;
    if len > MAX_KEY {
        return None;
    }
    let mut key = Vec::with_capacity(len);
    let mut put = |bytes: &[u8]| {
        key.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        key.extend_from_slice(bytes);
    };
    put(query.as_bytes());
    put(body);
    put(&(max_buckets as u64).to_le_bytes());
    for (k, n, b) in parts.iter() {
        put(&k.to_le_bytes());
        put(n.as_bytes());
        put(without_took(b));
    }
    Some(key)
}

/// The merged response memoized for `key`, with `took` set to `took_ms`.
pub(super) fn get(key: &[u8], took_ms: u64) -> Option<Vec<u8>> {
    let memo = MEMO.lock().ok()?;
    let merged = memo.map.get(key)?;
    Some(with_took(merged, took_ms))
}

/// Remember `merged` (a rendered response starting with `"took"`) for `key`.
pub(super) fn put(key: Vec<u8>, merged: &[u8]) {
    if !merged.starts_with(b"{\"took\":") {
        return;
    }
    let Ok(mut memo) = MEMO.lock() else { return };
    let size = key.len() + merged.len();
    if size > MAX_BYTES / 4 {
        return;
    }
    if let Some(old) = memo.map.insert(key.clone(), merged.to_vec()) {
        memo.bytes -= key.len() + old.len();
    } else {
        memo.order.push_back(key);
    }
    memo.bytes += size;
    while memo.bytes > MAX_BYTES {
        let Some(k) = memo.order.pop_front() else { break };
        if let Some(v) = memo.map.remove(&k) {
            memo.bytes -= k.len() + v.len();
        }
    }
}

fn with_took(json: &[u8], took_ms: u64) -> Vec<u8> {
    let rest = without_took(json);
    let mut out = Vec::with_capacity(rest.len() + 24);
    out.extend_from_slice(b"{\"took\":");
    out.extend_from_slice(took_ms.to_string().as_bytes());
    out.extend_from_slice(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn took_is_ignored_in_keys_and_rewritten_on_reuse() {
        let a = br#"{"took":3,"timed_out":false,"aggregations":{}}"#;
        let b = br#"{"took":17,"timed_out":false,"aggregations":{}}"#;
        let k1 = key("request_cache=true", b"{}", 100, &mut vec![(1, "n1", &a[..]), (0, "n1", &a[..])]).unwrap();
        let k2 = key("request_cache=true", b"{}", 100, &mut vec![(0, "n1", &b[..]), (1, "n1", &b[..])]).unwrap();
        assert_eq!(k1, k2, "same answers in any completion order, any took");
        let other = key("request_cache=true", b"{}", 100, &mut vec![(0, "n1", &br#"{"took":1,"timed_out":true}"#[..]), (1, "n1", &a[..])]).unwrap();
        assert_ne!(k1, other);
        put(k1.clone(), br#"{"took":9,"hits":{}}"#);
        assert_eq!(get(&k2, 4).unwrap(), br#"{"took":4,"hits":{}}"#.to_vec());
        assert!(get(&other, 4).is_none());
    }
}
