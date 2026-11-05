//! rustic-core adapter.
//!
//! Indexing: `Index::create` + `add_document` (or, with `--threads N > 1`,
//! `Index::add_documents_parallel` into N per-thread memtables) on a
//! dedicated indexing thread fed by the corpus parser + `refresh()` every N
//! docs (one segment file per refresh and memtable) + `commit()` at the
//! end.  See [`RusticIndexer`].  With
//! `--force-merge` the sealed segments are then merged into one with
//! `rustic_core::merge::merge_segments`; because `Index` has no API to swap
//! its segment list, the harness rewrites the `meta.json` commit point
//! itself and deletes the old segment files (see `bench/results/ISSUES.md`).
//!
//! Querying: each benchmark query becomes a `rustic_core::query::Query`
//! (`Term`, `Bool` must / should, `Phrase`, `RangeF64`) run through
//! `rustic_core::query::Searcher::search` over the index's segments — the
//! production path (one stats pass with index-wide BM25 statistics, as
//! Tantivy and Elasticsearch score; per-leaf bulk top-k with exact counts;
//! the merged k-th best score carried into later segments so they skip
//! non-competitive blocks; sequential for cheap queries, parallel per
//! segment for expensive ones).  BM25 k1 = 1.2, b = 0.75.

use crate::common::{Doc, Query};
use crate::engine::{FinishStats, Indexer, QueryResult, Searcher, TOP_K};
use rustic_core::query::query::{Bound, Query as RQuery};
use rustic_core::query::reader::LeafReader;
use rustic_core::query::searcher::{SearchOptions, Searcher as RSearcher};
use rustic_core::{Document, Field, FieldType, Index, IndexSettings, Term};
use std::path::{Path, PathBuf};

/// Field ids, fixed so the schema is identical across runs.
pub const F_ID: u16 = 0;
pub const F_TITLE: u16 = 1;
pub const F_BODY: u16 = 2;
pub const F_PRICE: u16 = 3;
pub const F_TS: u16 = 4;
pub const F_TAG: u16 = 5;

/// The benchmark schema. Numeric fields are fast-only (not inverted), the
/// keyword `tag` is inverted + fast, text fields use the `standard` analyzer
/// (lowercase + split on non-alphanumerics; apostrophes never occur in the
/// synthetic corpus).  rustic always stores every field.
pub fn schema() -> Vec<Field> {
    let mut id = Field::new(F_ID, "id", FieldType::I64);
    id.indexed = false;
    let title = Field::new(F_TITLE, "title", FieldType::Text);
    let body = Field::new(F_BODY, "body", FieldType::Text);
    let mut price = Field::new(F_PRICE, "price", FieldType::F64);
    price.indexed = false;
    let mut ts = Field::new(F_TS, "ts", FieldType::I64);
    ts.indexed = false;
    let tag = Field::new(F_TAG, "tag", FieldType::Keyword);
    vec![id, title, body, price, ts, tag]
}

pub fn field_id(name: &str) -> u16 {
    match name {
        "id" => F_ID,
        "title" => F_TITLE,
        "body" => F_BODY,
        "price" => F_PRICE,
        "ts" => F_TS,
        "tag" => F_TAG,
        other => panic!("unknown field {other}"),
    }
}

/// Indexing threads requested on the command line (`--threads N`, default
/// 1). `System::indexer` does not pass the value to rustic, so it is read
/// from the process arguments here.
fn threads_arg() -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1)
}

/// Docs per batch handed to the indexing thread.
const BATCH: usize = 1024;

enum Msg {
    Batch(Vec<Document>),
    Refresh,
}

/// The rustic indexer.
///
/// Default (pipelined, the same arrangement as Tantivy's `IndexWriter`):
/// the calling thread parses the corpus and builds `Document`s, a dedicated
/// indexing thread owns the `Index`. With `--threads 1` that thread adds
/// every doc to one memtable (single-threaded indexing); with `--threads N`
/// it calls `Index::add_documents_parallel(batch, N)`, so N per-thread
/// memtables (Lucene DWPT style) index concurrently and `refresh` flushes
/// them as N segments.
///
/// `RUSTIC_INLINE=1` indexes on the calling thread instead (parse and index
/// strictly sequential: one thread end to end, flush aside).
pub struct RusticIndexer {
    dir: PathBuf,
    force_merge: bool,
    /// Inline mode: the index itself.
    inline: Option<Index>,
    /// Pipelined mode.
    batch: Vec<Document>,
    tx: Option<std::sync::mpsc::SyncSender<Msg>>,
    worker: Option<std::thread::JoinHandle<Index>>,
}

impl RusticIndexer {
    pub fn create(dir: &Path, force_merge: bool) -> Self {
        std::fs::create_dir_all(dir).expect("create index dir");
        let index = Index::create(dir, "bench", schema(), IndexSettings::default()).expect("Index::create");
        let threads = threads_arg();
        let inline = std::env::var("RUSTIC_INLINE").is_ok_and(|v| v == "1");
        eprintln!(
            "[index]   rustic: {threads} indexing thread(s), {}",
            if inline { "inline (no parse/index pipeline)" } else { "pipelined (parser thread + indexing thread(s))" }
        );
        if inline {
            return Self { dir: dir.to_path_buf(), force_merge, inline: Some(index), batch: Vec::new(), tx: None, worker: None };
        }
        let (tx, rx) = std::sync::mpsc::sync_channel::<Msg>(4);
        let worker = std::thread::spawn(move || {
            let mut index = index;
            for msg in rx {
                match msg {
                    Msg::Batch(docs) => {
                        if threads == 1 {
                            for d in &docs {
                                index.add_document(d).expect("add_document");
                            }
                        } else {
                            index.add_documents_parallel(&docs, threads).expect("add_documents_parallel");
                        }
                    }
                    Msg::Refresh => index.refresh().expect("refresh"),
                }
            }
            index
        });
        Self {
            dir: dir.to_path_buf(),
            force_merge,
            inline: None,
            batch: Vec::with_capacity(BATCH),
            tx: Some(tx),
            worker: Some(worker),
        }
    }

    fn send_batch(&mut self) {
        if !self.batch.is_empty() {
            let b = std::mem::replace(&mut self.batch, Vec::with_capacity(BATCH));
            self.tx.as_ref().unwrap().send(Msg::Batch(b)).expect("indexing thread died");
        }
    }

    /// Merge every sealed segment into one and point the commit point at it.
    fn force_merge_all(&self, index: &Index) {
        use rustic_core::codec::{deletes_file_name, segment_file_name};
        let segments = index.segments().to_vec();
        if segments.len() < 2 {
            return;
        }
        let dir = index.directory().clone();
        let new_id = segments.iter().map(|s| s.id).max().unwrap_or(0) + 1;
        let merged = rustic_core::merge::merge_segments(&*dir, &segments, new_id).expect("merge_segments");
        // Rewrite meta.json: Index exposes no "replace segments" API.
        let meta_path = self.dir.join("meta.json");
        let mut meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).expect("read meta.json")).expect("parse meta.json");
        meta["segments"] = serde_json::json!([{"id": new_id, "num_docs": merged.num_docs}]);
        meta["next_segment_id"] = serde_json::json!(new_id + 1);
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).expect("write meta.json");
        for s in &segments {
            dir.delete(&segment_file_name(s.id)).expect("delete old segment");
            let del = deletes_file_name(s.id);
            if dir.exists(&del) {
                dir.delete(&del).expect("delete old .del");
            }
        }
    }
}

impl Indexer for RusticIndexer {
    fn add(&mut self, doc: &Doc) {
        let mut d = Document::new();
        d.fields.reserve_exact(6);
        d.add_i64(F_ID, doc.id);
        d.add_text(F_TITLE, &doc.title);
        d.add_text(F_BODY, &doc.body);
        d.add_f64(F_PRICE, doc.price);
        d.add_i64(F_TS, doc.ts);
        d.add_keyword(F_TAG, &doc.tag);
        match self.inline.as_mut() {
            Some(index) => {
                index.add_document(&d).expect("add_document");
            }
            None => {
                self.batch.push(d);
                if self.batch.len() >= BATCH {
                    self.send_batch();
                }
            }
        }
    }

    fn refresh(&mut self) {
        match self.inline.as_mut() {
            Some(index) => index.refresh().expect("refresh"),
            None => {
                self.send_batch();
                self.tx.as_ref().unwrap().send(Msg::Refresh).expect("indexing thread died");
            }
        }
    }

    fn finish(mut self: Box<Self>) -> FinishStats {
        let mut index = match self.inline.take() {
            Some(index) => index,
            None => {
                self.send_batch();
                drop(self.tx.take());
                self.worker.take().unwrap().join().expect("indexing thread panicked")
            }
        };
        index.refresh().expect("final refresh");
        index.commit().expect("commit");
        if self.force_merge && index.num_segments() > 1 {
            let t0 = std::time::Instant::now();
            self.force_merge_all(&index);
            let secs = t0.elapsed().as_secs_f64();
            eprintln!("[index]   force-merged into 1 segment in {secs:.1}s");
            return FinishStats {
                segments: 1,
                force_merge_s: Some(secs),
            };
        }
        FinishStats {
            segments: index.num_segments(),
            force_merge_s: None,
        }
    }
}

pub struct RusticSearcher {
    index: Index,
}

impl RusticSearcher {
    pub fn open(dir: &Path) -> Self {
        let index = Index::open(dir).expect("Index::open");
        Self { index }
    }

    /// The rustic query for a benchmark query.
    fn build(q: &Query) -> RQuery {
        let field = field_id(&q.field);
        match q.kind.as_str() {
            "term" => RQuery::term(field, q.term.as_deref().unwrap_or("")),
            "and" => RQuery::Bool {
                must: q.terms.iter().map(|t| RQuery::term(field, t)).collect(),
                should: Vec::new(),
                must_not: Vec::new(),
                filter: Vec::new(),
                minimum_should_match: Default::default(),
            },
            "or" => RQuery::Bool {
                must: Vec::new(),
                should: q.terms.iter().map(|t| RQuery::term(field, t)).collect(),
                must_not: Vec::new(),
                filter: Vec::new(),
                minimum_should_match: Default::default(),
            },
            "phrase" => RQuery::Phrase { terms: q.terms.iter().map(|t| Term::new_text(field, t)).collect(), slop: 0 },
            "range" => RQuery::RangeF64 {
                field,
                lower: Bound::Included(q.lower.unwrap_or(f64::MIN)),
                upper: Bound::Included(q.upper.unwrap_or(f64::MAX)),
            },
            other => panic!("unknown query kind {other}"),
        }
    }
}

impl Searcher for RusticSearcher {
    fn run(&self, q: &Query) -> QueryResult {
        let leaves: Vec<LeafReader<'_>> = self.index.segments().iter().map(|s| LeafReader::Segment(&**s)).collect();
        let opts = SearchOptions { top_k: TOP_K, ..Default::default() };
        let td = RSearcher::new(leaves).search(&Self::build(q), &opts);
        QueryResult {
            total_hits: td.total_hits,
            top_scores: td.hits.iter().map(|h| h.score).collect(),
        }
    }

    fn num_segments(&self) -> usize {
        self.index.num_segments()
    }
}
