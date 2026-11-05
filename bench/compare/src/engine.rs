//! The engine abstraction the benchmark drives: build an index from the
//! corpus, open it, run one query (top-10 by score + exact total hit count).

use crate::common::{Doc, Query};
use std::path::Path;

/// Top-k size used for every query in every system.
pub const TOP_K: usize = 10;

/// Outcome of one query execution.
#[derive(Debug, Clone, Default)]
pub struct QueryResult {
    /// Exact number of matching (live) documents.
    pub total_hits: u64,
    /// Scores of the returned top-k hits, descending.
    pub top_scores: Vec<f32>,
}

/// An opened, searchable index.
pub trait Searcher {
    fn run(&self, q: &Query) -> QueryResult;
    fn num_segments(&self) -> usize;
}

/// Builder for an index of a given system.
pub trait Indexer {
    /// Called once per document, in corpus order.
    fn add(&mut self, doc: &Doc);
    /// Called every `refresh_every` docs (rustic: `refresh()`; tantivy: no-op
    /// by default, see `--tantivy-commit-every`).
    fn refresh(&mut self);
    /// Called once at the end; must make the index durable.
    fn finish(self: Box<Self>) -> FinishStats;
}

/// What `Indexer::finish` reports.
#[derive(Debug, Clone, Default)]
pub struct FinishStats {
    /// Final searchable segment count.
    pub segments: usize,
    /// Wall time of an optional force-merge into one segment (included in
    /// the indexing wall time as well).
    pub force_merge_s: Option<f64>,
}

/// Which system a subcommand targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum System {
    Rustic,
    Tantivy,
}

impl System {
    pub fn parse(s: &str) -> Self {
        match s {
            "rustic" => System::Rustic,
            "tantivy" => System::Tantivy,
            other => {
                eprintln!("unknown --system {other} (expected rustic|tantivy)");
                std::process::exit(2);
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            System::Rustic => "rustic",
            System::Tantivy => "tantivy",
        }
    }

    pub fn open(self, dir: &Path) -> Box<dyn Searcher> {
        match self {
            System::Rustic => Box::new(crate::rustic_engine::RusticSearcher::open(dir)),
            System::Tantivy => Box::new(crate::tantivy_engine::TantivySearcher::open(dir)),
        }
    }

    pub fn indexer(
        self,
        dir: &Path,
        threads: usize,
        heap_bytes: usize,
        commit_every: usize,
        force_merge: bool,
    ) -> Box<dyn Indexer> {
        match self {
            System::Rustic => Box::new(crate::rustic_engine::RusticIndexer::create(dir, force_merge)),
            System::Tantivy => {
                if force_merge {
                    eprintln!("[index] --force-merge is a no-op for tantivy (its 1 GB heap already yields one segment for this corpus)");
                }
                Box::new(crate::tantivy_engine::TantivyIndexer::create(dir, threads, heap_bytes, commit_every))
            }
        }
    }
}
