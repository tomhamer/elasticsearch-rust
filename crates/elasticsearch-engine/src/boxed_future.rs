//! Large request futures, boxed where they are created.
//!
//! A debug build gives every future an `async fn` awaits its own slot in the
//! awaiting poll frame (no stack-slot reuse), and every wrapper that takes a
//! future by value (panic guard, memory policy, task-local scopes) copies
//! it into its own frame. A route dispatch that awaits dozens of handler
//! futures then needs their combined size on the stack, and one request
//! crosses the cluster front, the engine router (security, authorization)
//! and the front again (plus the engine once more for node-level APIs):
//! in dev builds that overflowed the 2 MiB tokio worker stack on the first
//! request. [`boxed`] builds the future in its own short-lived call frame
//! and returns it on the heap, so the awaiting frame holds one pointer.
//! Release builds were never affected; they pay one allocation per boxed
//! hop.

use std::future::Future;
use std::pin::Pin;

/// A boxed, sendable future.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The future `make` returns, built in this frame and boxed.
pub fn boxed<'a, F, Fut>(make: F) -> BoxFuture<'a, Fut::Output>
where
    F: FnOnce() -> Fut,
    Fut: Future + Send + 'a,
{
    Box::pin(make())
}
