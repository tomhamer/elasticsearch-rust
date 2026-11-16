//! Nesting bound for the recursive-descent parsers: Painless, Lucene
//! expressions, ingest scripts, ES|QL / SQL, EQL and KQL (`query_string`
//! has its own group limit). Every nesting level recurses through several
//! parser frames, and the trees they build are walked recursively later
//! (translation, evaluation, drop), so unbounded nesting overflows the
//! request thread's stack, which aborts the node. A parser stops at `max`
//! levels, or once its nesting has used [`STACK_BUDGET`] bytes of stack,
//! whichever comes first (frame sizes differ between builds). Only the
//! expression / statement entry counts a level; the prefix / unary step a
//! level also passes through is checked against the stack budget only, so
//! `max` is the nesting depth users write (`(((x)))` is 3). Operator chains
//! (`1 + 1 + ... + 1`) are parsed in a loop but build a tree as deep as the
//! chain, which later walks recurse over: each operator a loop applies
//! [`Nesting::grow`]s one level, released when the enclosing nested call
//! returns. So `max` bounds the depth of the tree, like ES|QL's
//! `MAX_EXPRESSION_DEPTH`.

/// Stack each guarded parse runs on ([`guarded`]): reserved address space,
/// committed only as used. Request threads have 2 MiB and a parse may start
/// anywhere in it, so parsing never runs on them.
const PARSE_STACK: usize = 256 << 20;

/// Stack a parse may use for nesting: a backstop well inside
/// [`PARSE_STACK`], so the counted `max` (ES's limits) is what bounds
/// nesting in every build profile, whatever the caller's stack.
pub(crate) const STACK_BUDGET: usize = 192 << 20;

/// Stack a parse may use on a thread whose stack size is unknown (a 2 MiB
/// default thread; parses through [`guarded`] never run there).
const UNKNOWN_STACK_BUDGET: usize = 1 << 20;

/// Uncounted levels allowed per counted level (each nesting level passes
/// through a few prefix / unary steps; a bare chain of them is cut here).
const UNCOUNTED_PER_LEVEL: u32 = 8;

/// Stack a registered request thread must still have free for a guarded
/// parse to run on it directly ([`guarded`]).
const INLINE_MIN: usize = 4 << 20;
/// Stack kept free below an inline parse (for the walks of what it built).
const INLINE_RESERVE: usize = 2 << 20;

thread_local! {
    static ON_PARSE_STACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// A registered request thread's stack: (a position at its top, size).
    static THREAD_STACK: std::cell::Cell<Option<(usize, usize)>> = const { std::cell::Cell::new(None) };
    /// The stack budget the next outermost [`Nesting`] level takes: on a
    /// thread whose stack is unknown, a parse called directly (not through
    /// [`guarded`]) may use [`UNKNOWN_STACK_BUDGET`]; the parse stack and
    /// inline guarded parses set their own.
    static BUDGET: std::cell::Cell<usize> = const { std::cell::Cell::new(UNKNOWN_STACK_BUDGET) };
    /// A [`Nesting`] stopped on the stack budget (not on its count).
    static BUDGET_HIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// Tests: parse threads this thread started.
    static PARSE_THREADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Register the calling thread (call it first thing in the thread) as a
/// request thread with a `size`-byte stack: guarded parses then run on it
/// directly while it has room ([`guarded`]).
pub fn register_thread_stack(size: usize) {
    THREAD_STACK.with(|t| t.set(Some((stack_pos(), size))));
}

/// Run a recursive-descent parse (and the recursive build of what it
/// returns) with stack to spare, independent of the build's frame sizes:
/// on the calling thread when it is a registered request thread with at
/// least [`INLINE_MIN`] free (the common case: no thread, no allocation),
/// else, or when that inline run stops on its stack budget (deep input),
/// on a dedicated [`PARSE_STACK`] thread. Nested guarded parses run inline.
/// The caller blocks until it returns, so borrowed, non-`Send` inputs and
/// results only ever have one user.
pub(crate) fn guarded<T>(f: impl Fn() -> T) -> T {
    if ON_PARSE_STACK.with(|c| c.get()) {
        return f();
    }
    if let Some((top, size)) = THREAD_STACK.with(|t| t.get()) {
        let used = top.wrapping_sub(stack_pos());
        if used < size && size - used >= INLINE_MIN {
            #[cfg(test)]
            let budget = INLINE_BUDGET_FOR_TESTS.with(|b| b.get()).unwrap_or(size - used - INLINE_RESERVE);
            #[cfg(not(test))]
            let budget = size - used - INLINE_RESERVE;
            let previous = BUDGET.with(|b| b.replace(budget));
            let hit_before = BUDGET_HIT.with(|h| h.replace(false));
            let r = f();
            BUDGET.with(|b| b.set(previous));
            if !BUDGET_HIT.with(|h| h.replace(hit_before)) {
                return r;
            }
            // Deeper than this stack allows: parse again on the parse stack.
            drop(r);
        }
    }
    #[cfg(test)]
    PARSE_THREADS.with(|n| n.set(n.get() + 1));
    struct Unsend<T>(T);
    // SAFETY: the spawning thread blocks in `scope` until the closure
    // returns; values are never touched by two threads at once.
    unsafe impl<T> Send for Unsend<T> {}
    let f = Unsend(f);
    let outcome = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("es-parse".into())
            .stack_size(PARSE_STACK)
            .spawn_scoped(scope, move || {
                let f = f;
                ON_PARSE_STACK.with(|c| c.set(true));
                BUDGET.with(|b| b.set(STACK_BUDGET));
                Unsend((f.0)())
            })
            .map(|handle| handle.join())
    });
    match outcome {
        Ok(Ok(Unsend(v))) => v,
        Ok(Err(panic)) => std::panic::resume_unwind(panic),
        Err(error) => panic!("could not start a parse thread: {error}"),
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: the inline stack budget on this thread.
    static INLINE_BUDGET_FOR_TESTS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Nesting {
    /// Stack position when the outermost level was entered.
    base: usize,
    /// Stack the levels may use (the thread's [`BUDGET`] at the outermost level).
    budget: usize,
    depth: u32,
    /// Uncounted levels entered (they keep `base` while open).
    frames: u32,
    max: u32,
}

#[inline(always)]
fn stack_pos() -> usize {
    let x = 0u8;
    std::hint::black_box(&x) as *const u8 as usize
}

impl Nesting {
    pub(crate) const fn new(max: u32) -> Self {
        Self { base: 0, budget: UNKNOWN_STACK_BUDGET, depth: 0, frames: 0, max }
    }

    /// Enter one level (`counted`: a nesting level users see; otherwise
    /// only the stack budget applies); `false` (nothing entered) when
    /// nested too deep.
    #[inline]
    pub(crate) fn enter(&mut self, counted: bool) -> bool {
        let here = stack_pos();
        if self.depth == 0 && self.frames == 0 {
            self.base = here;
            self.budget = BUDGET.with(|b| b.get());
        }
        let used = self.base.wrapping_sub(here);
        if used > self.budget && used < 1 << 40 {
            BUDGET_HIT.with(|h| h.set(true));
            return false;
        }
        // (A huge `used` means the stack grew the other way: ignore it.)
        // Uncounted levels (prefix / unary steps: `-`, `!`, EQL `not`) still
        // build tree depth that later walks recurse over: they are counted
        // too, at `UNCOUNTED_PER_LEVEL` per level, so no nesting relies on
        // the stack budget alone.
        if (counted && self.depth >= self.max)
            || (!counted && self.frames >= self.max.saturating_mul(UNCOUNTED_PER_LEVEL))
        {
            return false;
        }
        if counted {
            self.depth += 1;
        } else {
            self.frames += 1;
        }
        true
    }

    /// The current levels, to [`Nesting::restore`] when a nested call
    /// returns (releasing what it entered and grew, on every path).
    #[inline]
    pub(crate) fn snapshot(&self) -> (u32, u32) {
        (self.depth, self.frames)
    }

    #[inline]
    pub(crate) fn restore(&mut self, (depth, frames): (u32, u32)) {
        self.depth = depth;
        self.frames = frames;
    }

    /// One more level of tree built by a loop (an operator applied to the
    /// expression so far); `false` when that is too deep.
    #[inline]
    pub(crate) fn grow(&mut self) -> bool {
        if self.depth >= self.max {
            return false;
        }
        self.depth += 1;
        true
    }

    pub(crate) fn max(&self) -> u32 {
        self.max
    }
}

#[cfg(test)]
mod tests {
    /// Runs `f` on a thread with the 2 MiB stack request threads have.
    fn on_request_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new().stack_size(2 << 20).spawn(f).unwrap().join().unwrap();
    }

    /// Deep-but-allowed inputs parse in every build profile: parses run on
    /// their own stack (`guarded`), so only the counted limits apply.
    const OPTIMIZED: bool = true;

    fn parens(open: &str, n: usize, inner: &str, close: &str) -> String {
        format!("{}{inner}{}", open.repeat(n), close.repeat(n))
    }

    #[test]
    fn painless_nesting_is_bounded() {
        on_request_stack(|| {
            // Parentheses: ~7 parser frames a level, so the stack budget
            // (not the 512-level count) is what bounds them.
            assert!(!OPTIMIZED || crate::script::parser::parse(&format!("return {};", parens("(", 200, "1", ")"))).is_ok());
            assert!(crate::script::parser::parse(&format!("return {};", vec!["1"; 500].join(" + "))).is_ok());
            for src in [format!("return {};", vec!["1"; 100_000].join(" + ")), format!("return {};", parens("(", 50_000, "1", ")")), format!("return {}1;", "-".repeat(50_000)),
                        format!("{}", parens("{", 50_000, "", "}")), format!("return {}true;", "!".repeat(50_000))] {
                let e = crate::script::parser::parse(&src).err().expect("too deep");
                assert!(format!("{e:?}").contains("too many nested"), "{e:?}");
            }
        });
    }

    #[test]
    fn esql_and_sql_nesting_is_bounded() {
        on_request_stack(|| {
            let params = crate::esql::parse::Params::default();
            for ok in [format!("ROW x = {}", parens("(", 390, "1", ")")), format!("ROW x = {}", vec!["1"; 390].join(" + "))] {
                assert!(!OPTIMIZED || crate::esql::parse::parse_query(&ok, &params).is_ok());
            }
            for q in [format!("ROW x = {}", vec!["1"; 100_000].join(" + ")), format!("ROW x = {}", parens("(", 50_000, "1", ")")), format!("ROW x = {}1", "-".repeat(50_000)),
                      format!("ROW x = {}true", "NOT ".repeat(50_000))] {
                let e = crate::esql::parse::parse_query(&q, &params).err().expect("too deep");
                assert!(e.reason.contains("maximum expression depth allowed (400)"), "{}", e.reason);
            }
            for q in [format!("SELECT {} FROM t", parens("(", 50_000, "1", ")")), format!("SELECT {} FROM t", vec!["1"; 100_000].join(" + "))] {
                let e = crate::sql::parse(&q, &params).err().expect("too deep");
                assert!(e.reason.contains("maximum expression depth"), "{}", e.reason);
            }
            // ES 9.5.5 `SqlParser`: parentheses plus a prefix run, 250.
            let e = crate::sql::parse(&format!("SELECT {} FROM t", parens("(", 251, "1", ")")), &params).err().expect("too deep");
            assert_eq!(e.reason, "line -1:0: SQL statement exceeded the maximum expression depth allowed (250)");
            assert!(crate::sql::parse(&format!("SELECT {} FROM t", parens("(", 240, "1", ")")), &params).is_ok());
            let e = crate::eql::parse(&format!("any where {}a == 1", "not ".repeat(251)), &crate::eql::Opts { case_sensitive: false }).err().expect("too deep");
            assert_eq!(e.reason, "line -1:0: EQL statement exceeded the maximum expression depth allowed (250)");
        });
    }

    /// A registered request thread with this much stack.
    fn on_registered_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || {
            super::register_thread_stack(64 << 20);
            f()
        }).unwrap().join().unwrap();
    }

    #[test]
    fn guarded_parses_run_inline_on_request_threads() {
        on_registered_stack(|| {
            let params = crate::esql::parse::Params::default();
            let threads = || super::PARSE_THREADS.with(|n| n.get());
            // Common and deep-but-allowed inputs: no parse thread at all.
            assert!(crate::esql::parse::parse_query("ROW x = 1", &params).is_ok());
            assert!(crate::esql::parse::parse_query(&format!("ROW x = {}", parens("(", 390, "1", ")")), &params).is_ok());
            assert!(crate::script::parser::parse("return 1 + 2;").is_ok());
            assert_eq!(threads(), 0, "a parse thread was started on a request thread with room");
            // Counted limits still apply inline (no retry).
            let e = crate::esql::parse::parse_query(&format!("ROW x = {}", parens("(", 50_000, "1", ")")), &params).err().expect("too deep");
            assert!(e.reason.contains("maximum expression depth allowed (400)"), "{}", e.reason);
            assert_eq!(threads(), 0);
            // An inline parse that runs out of its stack budget is retried
            // on the parse stack, and succeeds there.
            super::INLINE_BUDGET_FOR_TESTS.with(|b| b.set(Some(16 << 10)));
            assert!(crate::esql::parse::parse_query(&format!("ROW x = {}", parens("(", 300, "1", ")")), &params).is_ok());
            super::INLINE_BUDGET_FOR_TESTS.with(|b| b.set(None));
            assert_eq!(threads(), 1, "the budget-limited parse was not retried on the parse stack");
        });
        // An unregistered thread (or one without room) always uses the parse stack.
        on_request_stack(|| {
            assert!(crate::script::parser::parse("return 1;").is_ok());
            assert_eq!(super::PARSE_THREADS.with(|n| n.get()), 1);
        });
    }

    /// Microbench of the per-request costs this module and the body parsers
    /// add (run with `--ignored --nocapture`): a guarded parse inline vs on
    /// a parse thread, and the body / document depth pass on vs off.
    #[test]
    #[ignore]
    fn parse_path_microbench() {
        fn time(label: &str, n: u32, mut f: impl FnMut()) {
            let t = std::time::Instant::now();
            for _ in 0..n { f(); }
            eprintln!("bench {label}: {:.2} us/op", t.elapsed().as_secs_f64() * 1e6 / n as f64);
        }
        let body = br#"{"size":10,"query":{"bool":{"filter":{"geo_shape":{"location":{"shape":{"type":"polygon","coordinates":[[[-0.1,51.5],[-0.12,51.52],[-0.08,51.53],[-0.05,51.51],[-0.1,51.5]]]},"relation":"intersects"}}}}},"sort":[{"_doc":"asc"}]}"#;
        let doc = r#"{"name":"place","location":{"type":"Point","coordinates":[-0.1278,51.5074]},"tags":["a","b","c"],"pop":8900000,"meta":{"source":"x","updated":"2024-01-02T03:04:05Z"}}"#;
        on_registered_stack(move || {
            let params = crate::esql::parse::Params::default();
            time("esql parse, inline (request thread)", 20_000, || { crate::esql::parse::parse_query("FROM logs | WHERE x > 1 | LIMIT 10", &params).unwrap(); });
            time("body parse_plain", 200_000, || { std::hint::black_box(crate::jsonpos::parse_plain(body)); });
            time("body depth pass (c33b extra per body)", 200_000, || { std::hint::black_box(crate::jsonpos::nesting(body, 1000)); });
            time("document Source::parse", 200_000, || { std::hint::black_box(crate::source_json::Source::parse(doc).unwrap()); });
            time("document depth pass (c33b extra per document)", 200_000, || { std::hint::black_box(crate::jsonpos::nesting(doc.as_bytes(), 1000)); });
        });
        on_request_stack(|| {
            let params = crate::esql::parse::Params::default();
            time("esql parse, on a parse thread (c33b on every thread)", 5_000, || { crate::esql::parse::parse_query("FROM logs | WHERE x > 1 | LIMIT 10", &params).unwrap(); });
        });
    }

    #[test]
    fn json_nesting_follows_es_limit() {
        on_request_stack(|| {
            let deep = |n: usize| format!("{}1{}", "[".repeat(n), "]".repeat(n));
            // Up to ES's (Jackson's) 1000 levels: parsed, on any thread.
            let ok = deep(999);
            assert_eq!(crate::jsonpos::nesting(ok.as_bytes(), 1000), (999, None));
            assert!(crate::jsonpos::parse_deep(ok.as_bytes()).is_ok());
            assert!(crate::source_json::Source::parse(&format!("{{\"a\":{ok}}}")).is_ok());
            // Past it: refused before any recursive parse, with ES's message.
            let bad = deep(1001);
            let (depth, at) = crate::jsonpos::nesting(bad.as_bytes(), 1000);
            assert_eq!((depth, at), (1001, Some((1, 1001))));
            let e = crate::jsonpos::nesting_error(depth, 1, 1001);
            assert_eq!(e.reason, "[1:1001] Document nesting depth (1001) exceeds the maximum allowed (1000, from `StreamReadConstraints.getMaxNestingDepth()`)");
            assert!(crate::source_json::Source::parse(&bad).is_err());
            // Brackets inside strings do not count.
            assert_eq!(crate::jsonpos::nesting(br#"{"a":"[[[[[[["}"#, 1000), (1, None));
        });
    }

    #[test]
    fn eql_nesting_is_bounded() {
        on_request_stack(|| {
            let o = crate::eql::Opts { case_sensitive: false };
            assert!(crate::eql::parse(&format!("any where {}", parens("(", 100, "a == 1", ")")), &o).is_ok());
            for q in [format!("any where {}", parens("(", 50_000, "a == 1", ")")), format!("any where {}a == 1", "not ".repeat(50_000))] {
                let e = crate::eql::parse(&q, &o).err().expect("too deep");
                // ES 9.5.5 `EqlParser` (`MAX_EXPRESSION_DEPTH` 250).
                assert_eq!(e.reason, "line -1:0: EQL statement exceeded the maximum expression depth allowed (250)");
            }
        });
    }

    #[test]
    fn ingest_script_nesting_is_bounded() {
        on_request_stack(|| {
            assert!(crate::ingest::lite::LiteScript::compile(&format!("return {};", parens("(", 100, "1", ")"))).is_ok());
            assert!(crate::ingest::lite::LiteScript::compile(&format!("return {};", vec!["1"; 500].join(" + "))).is_ok());
            for src in [format!("return {};", parens("(", 50_000, "1", ")")), format!("return {};", vec!["1"; 100_000].join(" + ")),
                        format!("return {};", vec!["true"; 100_000].join(" && "))] {
                let e = crate::ingest::lite::LiteScript::compile(&src).err().expect("too deep");
                assert!(e.reason.contains("too many nested") || format!("{e:?}").contains("too many nested"), "{e:?}");
            }
        });
    }
}
