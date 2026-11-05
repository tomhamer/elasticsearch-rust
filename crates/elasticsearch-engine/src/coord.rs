//! The cluster coordinator a request is being served for.
//!
//! On a cluster every engine holds only the shard copies assigned to its
//! node, so engine work that reads or fills indices on its own (ML results
//! and state, datafeed and data frame analytics source searches) must go
//! through the cluster coordinator to reach every shard and to replicate.
//! The cluster front end hands the engine a REST dispatcher with each
//! request (`elasticsearch_http::ClusterRest`); the HTTP layer makes it this
//! thread's [`current`] coordinator while the request is served, and code
//! that continues the work on another thread captures it with [`current`]
//! and re-enters it with [`scoped`]. Without a coordinator (a single node)
//! the engine acts on its own indices.

use crate::error::Result;
use serde_json::Value;
use std::cell::RefCell;
use std::sync::Arc;

/// `(method, uri, body) -> response body`, or the response's error. A
/// `Value::String` body is sent as is (NDJSON for `_bulk`).
pub type Coordinator = Arc<dyn Fn(&str, &str, Option<Value>) -> Result<Value> + Send + Sync>;

thread_local! {
    static SCOPED: RefCell<Option<Coordinator>> = const { RefCell::new(None) };
}

/// Run `f` with `c` as this thread's coordinator.
pub fn scoped<R>(c: Option<Coordinator>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Coordinator>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            SCOPED.with(|s| *s.borrow_mut() = prev);
        }
    }
    let _restore = Restore(SCOPED.with(|s| std::mem::replace(&mut *s.borrow_mut(), c)));
    f()
}

/// The coordinator [`scoped`] set on this thread, if any.
pub fn current() -> Option<Coordinator> {
    SCOPED.with(|s| s.borrow().clone())
}

/// Percent-encode an index expression or id for a URI path segment
/// (keeps `,` and `*`, which index expressions use).
pub fn path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b',' | b'*') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_restores_previous() {
        assert!(current().is_none());
        let c: Coordinator = Arc::new(|_, _, _| Ok(Value::Null));
        scoped(Some(c), || {
            assert!(current().is_some());
            scoped(None, || assert!(current().is_none()));
            assert!(current().is_some());
        });
        assert!(current().is_none());
    }

    #[test]
    fn path_segment_encodes() {
        assert_eq!(path_segment(".ml-anomalies-a,b*"), ".ml-anomalies-a,b*");
        assert_eq!(path_segment("a b#c/d"), "a%20b%23c%2Fd");
    }
}
