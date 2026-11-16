//! Bounds for the native recursive walks over script object graphs (Java
//! `hashCode`, `equals`, `toString` of collections). A script can build a
//! self-referencing collection, or a small graph whose walk is exponential
//! (a list holding itself twice), with a constant number of checked
//! instructions; the walk itself runs as native code that no fuel poll or
//! script stack guard sees. Java ends such a walk with a `StackOverflowError`
//! that Painless reports as a script failure; here every walk is bounded in
//! depth and in the containers it visits, and an exhausted walk is reported
//! as `stack_overflow_error` at the next script boundary ([`take`]): the
//! method call that started it, or the end of the run.

use super::error::ScriptErr;
use std::cell::Cell;

/// Nesting a walk may descend before it is cut off (well inside a worker's
/// native stack at a few hundred bytes per level).
pub const MAX_DEPTH: u32 = 1024;
/// Containers one top-level walk may visit (bounds exponential walks of
/// shared or cyclic graphs; flat collections count once).
pub const MAX_VISITS: u64 = 1 << 24;

/// Native stack one walk may use, whatever the build's frame sizes (a
/// worker has 2 MiB; debug frames are several times larger).
pub const MAX_STACK: usize = 512 * 1024;

#[derive(Clone, Copy, Default)]
struct State {
    depth: u32,
    visits: u64,
    exhausted: bool,
    /// Stack position when the outermost level was entered.
    base: usize,
}

thread_local! {
    static STATE: Cell<State> = const { Cell::new(State { depth: 0, visits: 0, exhausted: false, base: 0 }) };
}

#[inline(always)]
fn stack_pos() -> usize {
    let x = 0u8;
    std::hint::black_box(&x) as *const u8 as usize
}

/// One container level of a walk; `None` when the walk is over its bounds
/// (the caller returns a neutral value without descending).
pub struct Level(());

#[inline]
pub fn enter() -> Option<Level> {
    STATE.with(|s| {
        let mut st = s.get();
        if st.exhausted {
            return None;
        }
        let here = stack_pos();
        if st.depth == 0 {
            st.visits = 0;
            st.base = here;
        }
        st.visits += 1;
        let used = st.base.wrapping_sub(here);
        if st.depth >= MAX_DEPTH || st.visits > MAX_VISITS || (used > MAX_STACK && used < 1 << 40) {
            st.exhausted = true;
            s.set(st);
            return None;
        }
        st.depth += 1;
        s.set(st);
        Some(Level(()))
    })
}

impl Drop for Level {
    #[inline]
    fn drop(&mut self) {
        STATE.with(|s| {
            let mut st = s.get();
            st.depth = st.depth.saturating_sub(1);
            s.set(st);
        });
    }
}

/// Whether a walk on this thread was cut off since the last call (and
/// reset); the caller turns it into [`overflow`].
#[inline]
pub fn take() -> bool {
    STATE.with(|s| {
        let mut st = s.get();
        let hit = st.exhausted;
        if hit {
            st.exhausted = false;
            s.set(st);
        }
        hit
    })
}

/// The script failure of a cut-off walk (Java `StackOverflowError`).
pub fn overflow() -> Box<ScriptErr> {
    ScriptErr::new("stack_overflow_error", "")
}
