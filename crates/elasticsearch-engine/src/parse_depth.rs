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

/// Stack a parse may use for nesting (request threads have 2 MiB).
pub(crate) const STACK_BUDGET: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Nesting {
    /// Stack position when the outermost level was entered.
    base: usize,
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
        Self { base: 0, depth: 0, frames: 0, max }
    }

    /// Enter one level (`counted`: a nesting level users see; otherwise
    /// only the stack budget applies); `false` (nothing entered) when
    /// nested too deep.
    #[inline]
    pub(crate) fn enter(&mut self, counted: bool) -> bool {
        let here = stack_pos();
        if self.depth == 0 && self.frames == 0 {
            self.base = here;
        }
        let used = self.base.wrapping_sub(here);
        // (A huge `used` means the stack grew the other way: ignore it.)
        if (counted && self.depth >= self.max) || (used > STACK_BUDGET && used < 1 << 40) {
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

    /// Deep-but-allowed inputs must parse in optimized builds; debug frames
    /// are several times larger, so the stack budget cuts in earlier there.
    const OPTIMIZED: bool = !cfg!(debug_assertions);

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
        });
    }

    #[test]
    fn eql_nesting_is_bounded() {
        on_request_stack(|| {
            let o = crate::eql::Opts { case_sensitive: false };
            assert!(crate::eql::parse(&format!("any where {}", parens("(", 100, "a == 1", ")")), &o).is_ok());
            for q in [format!("any where {}", parens("(", 50_000, "a == 1", ")")), format!("any where {}a == 1", "not ".repeat(50_000))] {
                let e = crate::eql::parse(&q, &o).err().expect("too deep");
                assert!(e.reason.contains("nested more than"), "{}", e.reason);
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
