//! Score explanations (Lucene's `Explanation`).

use std::fmt;

/// A tree describing how a score was computed.
#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    /// The value this node contributes.
    pub value: f32,
    /// Human-readable description of the computation.
    pub description: String,
    /// Sub-computations this value was derived from.
    pub details: Vec<Explanation>,
}

impl Explanation {
    /// A node with children.
    pub fn new(value: f32, description: impl Into<String>, details: Vec<Explanation>) -> Self {
        Self { value, description: description.into(), details }
    }

    /// A node without children.
    pub fn leaf(value: f32, description: impl Into<String>) -> Self {
        Self::new(value, description, Vec::new())
    }

    /// The canonical "did not match" explanation (value 0).
    pub fn no_match(description: impl Into<String>) -> Self {
        Self::leaf(0.0, format!("no match: {}", description.into()))
    }

    /// Number of nodes in the tree (this one included).
    pub fn node_count(&self) -> usize {
        1 + self.details.iter().map(|d| d.node_count()).sum::<usize>()
    }

    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        writeln!(f, "{:indent$}{} = {}", "", self.value, self.description, indent = depth * 2)?;
        for d in &self.details {
            d.fmt_indent(f, depth + 1)?;
        }
        Ok(())
    }
}

impl fmt::Display for Explanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}
