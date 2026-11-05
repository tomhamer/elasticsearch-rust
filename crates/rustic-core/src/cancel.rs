//! Cancellation of a long open (an index whose graph checkpoint is caught
//! up, or rebuilt, from its committed vector rows): the caller runs the
//! open under [`with_cancel`], and the open's long loops call [`check`]
//! between batches. A cancelled open fails with [`RusticError::Cancelled`].
//!
//! Search task cancellation ([`Token`]) is separate: a search's token is
//! installed per thread with [`search_scope`] and polled by the search
//! loops at their existing check points.

use crate::error::{Result, RusticError};
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    static TOKEN: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Run `f` on this thread with `token` as its cancellation token (the
/// previous one is restored afterwards, also on a panic).
pub fn with_cancel<T>(token: Arc<AtomicBool>, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<Arc<AtomicBool>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            TOKEN.with(|t| *t.borrow_mut() = previous);
        }
    }
    let _restore = Restore(TOKEN.with(|t| t.borrow_mut().replace(token)));
    f()
}

/// `Err(Cancelled)` once this thread's token (if any) is set.
pub fn check() -> Result<()> {
    let cancelled = TOKEN.with(|t| t.borrow().as_ref().is_some_and(|token| token.load(Ordering::Acquire)));
    if cancelled {
        return Err(RusticError::Cancelled("the open was cancelled".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Search task cancellation
// ---------------------------------------------------------------------------

/// The cancellation of one search task (ES `CancellableTask`): set by
/// `_tasks/_cancel`, by a ban of its parent task, or when the client that
/// asked for it goes away. Long search loops poll it at the granularity of
/// their existing deadline checks (every 1024 docs, per segment, per
/// shard), never per document. A child task's token follows its parent's.
#[derive(Default)]
pub struct Token {
    flag: AtomicBool,
    parent: Option<Arc<Token>>,
    reason: std::sync::OnceLock<String>,
    /// Futures waiting for the cancellation ([`Token::cancelled`]).
    waiters: parking_lot::Mutex<Vec<std::task::Waker>>,
}

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Token").field("cancelled", &self.is_cancelled()).finish()
    }
}

impl Token {
    /// A token nothing has cancelled yet.
    pub fn new() -> Arc<Token> {
        Arc::new(Token::default())
    }

    /// A child task's token: cancelled with `parent` (or on its own).
    pub fn child_of(parent: Option<Arc<Token>>) -> Arc<Token> {
        Arc::new(Token { parent, ..Token::default() })
    }

    /// A token cancelled already (a child of a parent banned before it
    /// started).
    pub fn cancelled_with(reason: &str) -> Arc<Token> {
        let token = Token::new();
        token.cancel(reason);
        token
    }

    /// Whether this task (or an ancestor) was cancelled: one relaxed load
    /// per level.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed) || self.parent.as_ref().is_some_and(|p| p.is_cancelled())
    }

    /// Cancel the task (the first reason given is kept) and wake whatever
    /// waits for that.
    pub fn cancel(&self, reason: &str) {
        let _ = self.reason.set(reason.to_string());
        self.flag.store(true, Ordering::Release);
        let waiters = std::mem::take(&mut *self.waiters.lock());
        for waker in waiters {
            waker.wake();
        }
    }

    /// Why the task was cancelled (its own reason, else its nearest
    /// cancelled ancestor's); `None` while it runs.
    pub fn reason(&self) -> Option<String> {
        if self.flag.load(Ordering::Acquire) {
            return Some(self.reason.get().cloned().unwrap_or_else(|| "by user request".to_string()));
        }
        self.parent.as_ref().and_then(|p| p.reason())
    }

    /// Resolves once this task (or an ancestor) is cancelled: the waiter
    /// registers with every level of the chain.
    pub fn cancelled(&self) -> Cancelled<'_> {
        Cancelled(self)
    }
}

/// Future of [`Token::cancelled`].
pub struct Cancelled<'a>(&'a Token);

impl std::future::Future for Cancelled<'_> {
    type Output = ();
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        let token = self.0;
        if token.is_cancelled() {
            return std::task::Poll::Ready(());
        }
        let mut level = Some(token);
        while let Some(t) = level {
            let mut waiters = t.waiters.lock();
            if !waiters.iter().any(|w| w.will_wake(cx.waker())) {
                waiters.push(cx.waker().clone());
            }
            drop(waiters);
            level = t.parent.as_deref();
        }
        // `cancel` sets the flag before it takes the waiters: a cancel
        // racing the registration above is seen here.
        if token.is_cancelled() {
            return std::task::Poll::Ready(());
        }
        std::task::Poll::Pending
    }
}

/// When a per-document loop polls its timeout and its task's
/// cancellation: every 1024 docs, and more often while documents are slow
/// (a block of docs that took over 2ms shrinks the block eightfold, down to
/// 8 docs; fast blocks grow it back), so a cancellation reaches a slow
/// scorer within milliseconds. A fast loop pays one compare per doc and a
/// clock read per 1024 docs; no atomic per doc.
#[derive(Debug)]
pub struct Poll {
    next: u64,
    step: u64,
    last: Option<std::time::Instant>,
}

impl Default for Poll {
    fn default() -> Self {
        Poll::new()
    }
}

impl Poll {
    /// First due at doc 1 (as the loops' former `% 1024 == 1` check).
    pub fn new() -> Poll {
        Poll { next: 1, step: 1024, last: None }
    }

    /// Whether the loop at its `n`-th doc should poll now.
    #[inline]
    pub fn due(&mut self, n: u64) -> bool {
        if n < self.next {
            return false;
        }
        self.advance(n);
        true
    }

    #[inline(never)]
    fn advance(&mut self, n: u64) {
        let now = std::time::Instant::now();
        if let Some(last) = self.last {
            let took = now.duration_since(last);
            self.step = if took > std::time::Duration::from_millis(2) {
                (self.step / 8).max(8)
            } else if took < std::time::Duration::from_micros(500) {
                (self.step * 2).min(1024)
            } else {
                self.step
            };
        }
        self.last = Some(now);
        self.next = n.saturating_add(self.step);
    }
}

thread_local! {
    static SEARCH: RefCell<Option<Arc<Token>>> = const { RefCell::new(None) };
}

/// This thread's search task token (see [`search_scope`]).
pub fn current() -> Option<Arc<Token>> {
    SEARCH.with(|t| t.borrow().clone())
}

/// Whether this thread's search task was cancelled. A thread-local read
/// and an atomic load: for per-shard / per-segment / per-hit checks, not
/// per-document loops (those hold the token, see [`current`]).
pub fn is_cancelled() -> bool {
    SEARCH.with(|t| t.borrow().as_ref().is_some_and(|token| token.is_cancelled()))
}

/// The reason of this thread's cancelled search task.
pub fn reason() -> Option<String> {
    SEARCH.with(|t| t.borrow().as_ref().and_then(|token| token.reason()))
}

/// Restores the previous search token when dropped.
pub struct SearchScope {
    previous: Option<Arc<Token>>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for SearchScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SEARCH.with(|t| *t.borrow_mut() = previous);
    }
}

/// Make `token` this thread's search task token until the scope drops.
pub fn search_scope(token: Option<Arc<Token>>) -> SearchScope {
    SearchScope { previous: SEARCH.with(|t| t.replace(token)), _thread: std::marker::PhantomData }
}

/// Run `f` with `token` as this thread's search task token.
pub fn with_search<T>(token: Option<Arc<Token>>, f: impl FnOnce() -> T) -> T {
    let _scope = search_scope(token);
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_token_cancels_only_its_own_scope() {
        let token = Arc::new(AtomicBool::new(false));
        with_cancel(token.clone(), || {
            assert!(check().is_ok());
            token.store(true, Ordering::Release);
            assert!(matches!(check(), Err(RusticError::Cancelled(_))));
        });
        assert!(check().is_ok(), "restored after the scope");
    }

    #[test]
    fn a_search_token_follows_its_parent_and_scopes_restore() {
        let parent = Token::new();
        let child = Token::child_of(Some(parent.clone()));
        assert!(!child.is_cancelled() && child.reason().is_none());
        with_search(Some(child.clone()), || {
            assert!(!is_cancelled());
            parent.cancel("by user request");
            assert!(is_cancelled());
            assert_eq!(reason().as_deref(), Some("by user request"));
        });
        assert!(!is_cancelled(), "restored after the scope");
        let own = Token::cancelled_with("channel closed");
        assert_eq!(own.reason().as_deref(), Some("channel closed"));
        own.cancel("later");
        assert_eq!(own.reason().as_deref(), Some("channel closed"), "the first reason is kept");
    }

    #[test]
    fn a_cancel_wakes_its_waiter() {
        use std::future::Future;
        let token = Token::new();
        let waiter = token.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            // A minimal executor: park until woken.
            struct Unpark(std::thread::Thread);
            impl std::task::Wake for Unpark {
                fn wake(self: Arc<Self>) { self.0.unpark(); }
            }
            let waker = std::task::Waker::from(Arc::new(Unpark(std::thread::current())));
            let mut cx = std::task::Context::from_waker(&waker);
            let mut fut = std::pin::pin!(waiter.cancelled());
            tx.send(()).unwrap();
            while fut.as_mut().poll(&mut cx).is_pending() {
                std::thread::park();
            }
        });
        rx.recv().unwrap();
        token.cancel("by user request");
        handle.join().unwrap();
    }

    #[test]
    fn a_poll_is_due_every_1024_docs_and_more_often_for_slow_docs() {
        let mut poll = Poll::new();
        let due: Vec<u64> = (1..=3000u64).filter(|n| poll.due(*n)).collect();
        assert_eq!(&due[..3], &[1, 1025, 2049], "fast docs: every 1024");
        let mut slow = Poll::new();
        assert!(slow.due(1));
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(slow.due(1025));
        // The slow block shrank the next one to 128 docs.
        assert!(!slow.due(1152));
        assert!(slow.due(1153));
    }
}
