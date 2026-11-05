//! Nested documents / block join (Lucene `ToParentBlockJoinQuery`).
//!
//! A parent and its nested children are indexed as one contiguous block,
//! children first and the parent last (recursively: a child's own children
//! precede it). Every doc carries a *nested path* ordinal: `0` is the root
//! level (`""`, depth 0); other ordinals name a nested path and its depth.
//! The parent of a doc at depth `k` is therefore the next doc whose depth is
//! below `k` (Lucene's next-set-bit on the parent bitset), and a parent's
//! block is the run of deeper docs right before it.
//!
//! Segments without nested docs store nothing ([`NestedInfo`] is absent).

use crate::error::{Result, RusticError};
use crate::postings::BitSet;
use crate::types::DocId;

/// Per-leaf nested path table and per-doc path ordinals.
#[derive(Clone, Debug)]
pub struct NestedInfo {
    /// `(path, depth)` by ordinal; ordinal 0 is the root (`""`, 0).
    paths: Vec<(String, u16)>,
    /// Path ordinal of every doc.
    ords: Vec<u16>,
}

impl Default for NestedInfo {
    fn default() -> Self {
        Self { paths: vec![(String::new(), 0)], ords: Vec::new() }
    }
}

impl NestedInfo {
    /// Ordinal of `path` at `depth`, adding it if new.
    pub fn intern(&mut self, path: &str, depth: u16) -> u16 {
        if path.is_empty() && depth == 0 {
            return 0;
        }
        if let Some(i) = self.paths.iter().position(|(p, d)| p == path && *d == depth) {
            return i as u16;
        }
        self.paths.push((path.to_string(), depth));
        (self.paths.len() - 1) as u16
    }

    /// Record `doc`'s path ordinal (docs before it default to the root).
    pub fn set(&mut self, doc: DocId, ord: u16) {
        let d = doc as usize;
        if self.ords.len() <= d {
            self.ords.resize(d + 1, 0);
        }
        self.ords[d] = ord;
    }

    /// Grow to `num_docs` (new docs are roots).
    pub fn resize(&mut self, num_docs: u32) {
        if self.ords.len() < num_docs as usize {
            self.ords.resize(num_docs as usize, 0);
        }
    }

    /// Number of docs covered.
    pub fn num_docs(&self) -> u32 {
        self.ords.len() as u32
    }

    /// The `(path, depth)` table.
    pub fn paths(&self) -> &[(String, u16)] {
        &self.paths
    }

    /// Ordinal of a nested path (`None` if no doc here has it).
    pub fn path_ord(&self, path: &str) -> Option<u16> {
        self.paths.iter().skip(1).position(|(p, _)| p == path).map(|i| i as u16 + 1)
    }

    /// Path ordinal of `doc` (root for docs past the end).
    #[inline]
    pub fn ord(&self, doc: DocId) -> u16 {
        self.ords.get(doc as usize).copied().unwrap_or(0)
    }

    /// Nesting depth of `doc` (0 = root document).
    #[inline]
    pub fn depth(&self, doc: DocId) -> u16 {
        self.paths[self.ord(doc) as usize].1
    }

    /// Nested path of `doc` (`""` for root documents).
    pub fn path(&self, doc: DocId) -> &str {
        &self.paths[self.ord(doc) as usize].0
    }

    /// Whether `doc` is a root (top-level) document.
    #[inline]
    pub fn is_root(&self, doc: DocId) -> bool {
        self.ord(doc) == 0
    }

    /// Whether any doc is nested.
    pub fn has_nested(&self) -> bool {
        self.ords.iter().any(|&o| o != 0)
    }

    /// The doc whose block `doc` belongs to one level up (the next doc
    /// with a smaller depth); `None` for root docs.
    pub fn parent(&self, doc: DocId) -> Option<DocId> {
        let k = self.depth(doc);
        if k == 0 {
            return None;
        }
        (doc + 1..self.num_docs()).find(|&d| self.depth(d) < k)
    }

    /// The enclosing doc at nesting depth `depth` (the next doc whose
    /// depth is at most `depth`); `doc` itself when it is that shallow.
    pub fn ancestor_at(&self, doc: DocId, depth: u16) -> Option<DocId> {
        if self.depth(doc) <= depth {
            return Some(doc);
        }
        (doc + 1..self.num_docs()).find(|&d| self.depth(d) <= depth)
    }

    /// Depth of nested path ordinal `ord`.
    pub fn ord_depth(&self, ord: u16) -> u16 {
        self.paths.get(ord as usize).map(|p| p.1).unwrap_or(0)
    }

    /// The top-level document `doc` belongs to (itself for roots).
    pub fn root(&self, doc: DocId) -> DocId {
        if self.is_root(doc) {
            return doc;
        }
        (doc + 1..self.num_docs()).find(|&d| self.is_root(d)).unwrap_or(doc)
    }

    /// First doc of `doc`'s block (its earliest descendant, or itself).
    pub fn block_start(&self, doc: DocId) -> DocId {
        let k = self.depth(doc);
        let mut s = doc;
        while s > 0 && self.depth(s - 1) > k {
            s -= 1;
        }
        s
    }

    /// Visit the descendants of `parent` at nested path ordinal `ord`,
    /// ascending (no allocation).
    #[inline]
    pub fn for_each_child(&self, parent: DocId, ord: u16, mut f: impl FnMut(DocId)) {
        for d in self.block_start(parent)..parent {
            if self.ord(d) == ord {
                f(d);
            }
        }
    }

    /// Descendants of `parent` at nested path ordinal `ord`, ascending.
    pub fn children(&self, parent: DocId, ord: u16) -> Vec<DocId> {
        (self.block_start(parent)..parent).filter(|&d| self.ord(d) == ord).collect()
    }

    /// Iterator over descendants of `parent` at nested path ordinal `ord`,
    /// ascending, without allocating a `Vec`.
    pub fn children_iter(&self, parent: DocId, ord: u16) -> impl Iterator<Item = DocId> + '_ {
        (self.block_start(parent)..parent).filter(move |&d| self.ord(d) == ord)
    }

    /// Position of `child` among the same-path children of its parent (ES
    /// `_nested.offset`: the index in the parent's source array).
    pub fn offset(&self, child: DocId) -> u32 {
        let Some(p) = self.parent(child) else { return 0 };
        let o = self.ord(child);
        (self.block_start(p)..child).filter(|&d| self.ord(d) == o).count() as u32
    }

    /// Bitset of the non-root docs (`None` when there are none).
    pub fn child_bits(&self, num_docs: u32) -> Option<BitSet> {
        if !self.has_nested() {
            return None;
        }
        let mut b = BitSet::new(num_docs);
        for (d, &o) in self.ords.iter().enumerate() {
            if o != 0 && (d as u32) < num_docs {
                b.set(d as DocId);
            }
        }
        Some(b)
    }

    /// Serialize: `u32 paths, paths × (u32 len, bytes, u16 depth)`, `u32
    /// docs, docs × u16 ord`.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.paths.len() as u32).to_le_bytes());
        for (p, d) in &self.paths {
            out.extend_from_slice(&(p.len() as u32).to_le_bytes());
            out.extend_from_slice(p.as_bytes());
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&(self.ords.len() as u32).to_le_bytes());
        for o in &self.ords {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }

    /// Inverse of [`NestedInfo::encode`].
    pub fn decode(data: &[u8]) -> Result<Self> {
        let err = || RusticError::Corrupted("nested info truncated".into());
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let b = data.get(pos..pos + n).ok_or_else(err)?;
            pos += n;
            Ok(b)
        };
        let np = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let mut paths = Vec::with_capacity(np);
        for _ in 0..np {
            let len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            let p = String::from_utf8(take(len)?.to_vec()).map_err(|_| err())?;
            let d = u16::from_le_bytes(take(2)?.try_into().unwrap());
            paths.push((p, d));
        }
        let n = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let raw = take(n * 2)?;
        let ords = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if paths.is_empty() {
            return Err(err());
        }
        Ok(NestedInfo { paths, ords })
    }

    /// Concatenate several leaves' infos through a doc map (merge): `parts`
    /// yields, per input, its info (or `None` = all roots), its doc count
    /// and the new id of each old doc (`None` = dropped).
    pub fn merge<'a>(parts: impl Iterator<Item = (Option<&'a NestedInfo>, u32, Box<dyn Fn(DocId) -> Option<DocId> + 'a>)>, num_docs: u32) -> Option<NestedInfo> {
        let mut out = NestedInfo::default();
        let mut any = false;
        for (info, n, map) in parts {
            let Some(info) = info else { continue };
            if !info.has_nested() {
                continue;
            }
            let remap: Vec<u16> = info.paths.iter().map(|(p, d)| out.intern(p, *d)).collect();
            for old in 0..n {
                if let Some(new) = map(old) {
                    let o = remap[info.ord(old) as usize];
                    if o != 0 {
                        out.set(new, o);
                        any = true;
                    }
                }
            }
        }
        if !any {
            return None;
        }
        out.resize(num_docs);
        Some(out)
    }
}

/// How the scores of the matching children of one parent combine (ES
/// `nested.score_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NestedScoreMode {
    /// Mean of the child scores (ES default).
    #[default]
    Avg,
    /// Best child score.
    Max,
    /// Worst child score.
    Min,
    /// Sum of the child scores.
    Sum,
    /// Parents score 0.
    None,
}

impl NestedScoreMode {
    /// Parse an ES name (`avg`, `max`, `min`, `sum`, `none`).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "avg" => NestedScoreMode::Avg,
            "max" => NestedScoreMode::Max,
            "min" => NestedScoreMode::Min,
            "sum" | "total" => NestedScoreMode::Sum,
            "none" => NestedScoreMode::None,
            _ => return None,
        })
    }

    /// Combine `sum`/`min`/`max`/`count` of the child scores.
    pub fn combine(&self, sum: f32, min: f32, max: f32, count: u32) -> f32 {
        match self {
            NestedScoreMode::Avg => sum / count.max(1) as f32,
            NestedScoreMode::Max => max,
            NestedScoreMode::Min => min,
            NestedScoreMode::Sum => sum,
            NestedScoreMode::None => 0.0,
        }
    }
}

/// One inner hit: a matching child of a parent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestedHit {
    /// Leaf-local doc id of the child.
    pub doc: DocId,
    /// ES `_nested.offset` of the child.
    pub offset: u32,
    /// The child's score under the inner query.
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_navigation() {
        // root0: [a, a, b.c?]. Layout: 0:a 1:a 2:root | 3:a.x 4:a 5:root
        let mut n = NestedInfo::default();
        let a = n.intern("a", 1);
        let ax = n.intern("a.x", 2);
        for (d, o) in [(0, a), (1, a), (2, 0), (3, ax), (4, a), (5, 0)] {
            n.set(d, o);
        }
        assert_eq!(n.parent(0), Some(2));
        assert_eq!(n.parent(3), Some(4));
        assert_eq!(n.parent(4), Some(5));
        assert_eq!(n.root(3), 5);
        assert_eq!(n.children(2, a), vec![0, 1]);
        assert_eq!(n.children(5, a), vec![4]);
        assert_eq!(n.children(5, ax), vec![3]);
        assert_eq!(n.offset(1), 1);
        assert_eq!(n.offset(4), 0);
        assert_eq!(n.block_start(5), 3);
        let mut buf = Vec::new();
        n.encode(&mut buf);
        let m = NestedInfo::decode(&buf).unwrap();
        assert_eq!(m.children(5, m.path_ord("a.x").unwrap()), vec![3]);
        assert_eq!(m.child_bits(6).unwrap().count(), 4);
    }
}
