//! Automaton-based term matching for fuzzy, wildcard, regexp and prefix
//! queries, in the spirit of Lucene's `FuzzyQuery` / `WildcardQuery` /
//! `RegexpQuery` / `PrefixQuery` term enumeration.
//!
//! All matchers operate on the *term text* only (the 2-byte field prefix is
//! handled by the dictionary walk in [`expand`]).
//!
//! - [`levenshtein`]: edit-distance matching (Levenshtein / Damerau)
//! - [`wildcard`]: `*` / `?` patterns with `\` escapes
//! - [`regexp`]: whole-term regular expressions
//! - [`expand`]: bounded term-dictionary enumeration using a matcher

pub mod expand;
pub mod levenshtein;
pub mod regexp;
pub mod wildcard;

/// A byte-level matcher over term text (without the field prefix).
pub trait TermMatcher: Send + Sync {
    /// Returns true if `term` (the raw term bytes, no field prefix) matches.
    fn matches(&self, term: &[u8]) -> bool;

    /// A literal prefix every match must start with (lets the dictionary
    /// walk skip ahead). Empty if none.
    fn literal_prefix(&self) -> &[u8];
}
