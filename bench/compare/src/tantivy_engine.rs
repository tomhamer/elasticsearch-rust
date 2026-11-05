//! Tantivy 0.22 adapter.
//!
//! Schema: `title`/`body` are `TEXT | STORED` (tokenizer "default" =
//! simple split on non-alphanumerics + lowercase, positions indexed),
//! `price` f64 `FAST`, `ts` i64 `FAST`, `tag` `STRING | FAST`, `id` i64
//! `FAST | STORED`.  Stored fields are on so the on-disk size is comparable
//! to rustic, which always stores documents.
//!
//! Indexing uses `writer_with_num_threads(threads, heap)` (defaults: 1
//! thread, 1 GiB) and one `commit()` at the end (optionally every N docs with
//! `--tantivy-commit-every`).  Queries: `TermQuery`, `BooleanQuery`,
//! `PhraseQuery`, `RangeQuery` (fast-field path, since `price` is FAST) with
//! `(TopDocs::with_limit(10), Count)`.

use crate::common::{Doc, Query};
use crate::engine::{FinishStats, Indexer, QueryResult, Searcher, TOP_K};
use std::ops::Bound;
use std::path::Path;
use tantivy::collector::{Count, TopDocs};
use tantivy::query::{BooleanQuery, Occur, PhraseQuery, Query as TQuery, RangeQuery, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, FAST, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexWriter, ReloadPolicy, TantivyDocument, Term};

struct Fields {
    id: Field,
    title: Field,
    body: Field,
    price: Field,
    ts: Field,
    tag: Field,
}

fn build_schema() -> (Schema, Fields) {
    let mut b = Schema::builder();
    let id = b.add_i64_field("id", FAST | STORED);
    let title = b.add_text_field("title", TEXT | STORED);
    let body = b.add_text_field("body", TEXT | STORED);
    let price = b.add_f64_field("price", FAST);
    let ts = b.add_i64_field("ts", FAST);
    let tag = b.add_text_field("tag", STRING | FAST);
    (
        b.build(),
        Fields {
            id,
            title,
            body,
            price,
            ts,
            tag,
        },
    )
}

fn fields_of(schema: &Schema) -> Fields {
    Fields {
        id: schema.get_field("id").unwrap(),
        title: schema.get_field("title").unwrap(),
        body: schema.get_field("body").unwrap(),
        price: schema.get_field("price").unwrap(),
        ts: schema.get_field("ts").unwrap(),
        tag: schema.get_field("tag").unwrap(),
    }
}

pub struct TantivyIndexer {
    index: Index,
    writer: IndexWriter<TantivyDocument>,
    fields: Fields,
    commit_every: usize,
    since_commit: usize,
}

impl TantivyIndexer {
    pub fn create(dir: &Path, threads: usize, heap_bytes: usize, commit_every: usize) -> Self {
        std::fs::create_dir_all(dir).expect("create index dir");
        let (schema, fields) = build_schema();
        let index = Index::create_in_dir(dir, schema).expect("Index::create_in_dir");
        let writer = index
            .writer_with_num_threads::<TantivyDocument>(threads.max(1), heap_bytes)
            .expect("writer_with_num_threads");
        Self {
            index,
            writer,
            fields,
            commit_every,
            since_commit: 0,
        }
    }
}

impl Indexer for TantivyIndexer {
    fn add(&mut self, d: &Doc) {
        let f = &self.fields;
        self.writer
            .add_document(doc!(
                f.id => d.id,
                f.title => d.title.as_str(),
                f.body => d.body.as_str(),
                f.price => d.price,
                f.ts => d.ts,
                f.tag => d.tag.as_str(),
            ))
            .expect("add_document");
        self.since_commit += 1;
        if self.commit_every > 0 && self.since_commit >= self.commit_every {
            self.writer.commit().expect("commit");
            self.since_commit = 0;
        }
    }

    fn refresh(&mut self) {
        // Tantivy flushes segments on its own memory budget; nothing to do.
    }

    fn finish(mut self: Box<Self>) -> FinishStats {
        self.writer.commit().expect("commit");
        self.writer.wait_merging_threads().expect("wait_merging_threads");
        FinishStats {
            segments: self.index.searchable_segment_metas().map(|m| m.len()).unwrap_or(0),
            force_merge_s: None,
        }
    }
}

pub struct TantivySearcher {
    searcher: tantivy::Searcher,
    fields: Fields,
}

impl TantivySearcher {
    pub fn open(dir: &Path) -> Self {
        let index = Index::open_in_dir(dir).expect("Index::open_in_dir");
        let fields = fields_of(&index.schema());
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .expect("reader");
        Self {
            searcher: reader.searcher(),
            fields,
        }
    }

    fn field(&self, name: &str) -> Field {
        match name {
            "id" => self.fields.id,
            "title" => self.fields.title,
            "body" => self.fields.body,
            "price" => self.fields.price,
            "ts" => self.fields.ts,
            "tag" => self.fields.tag,
            other => panic!("unknown field {other}"),
        }
    }

    fn build(&self, q: &Query) -> Box<dyn TQuery> {
        let field = self.field(&q.field);
        let term_q = |t: &str| -> Box<dyn TQuery> {
            Box::new(TermQuery::new(
                Term::from_field_text(field, t),
                IndexRecordOption::WithFreqs,
            ))
        };
        match q.kind.as_str() {
            "term" => term_q(q.term.as_deref().unwrap_or("")),
            "and" => Box::new(BooleanQuery::new(
                q.terms.iter().map(|t| (Occur::Must, term_q(t))).collect(),
            )),
            "or" => Box::new(BooleanQuery::new(
                q.terms.iter().map(|t| (Occur::Should, term_q(t))).collect(),
            )),
            "phrase" => Box::new(PhraseQuery::new(
                q.terms.iter().map(|t| Term::from_field_text(field, t)).collect(),
            )),
            "range" => Box::new(RangeQuery::new_f64_bounds(
                q.field.clone(),
                Bound::Included(q.lower.unwrap_or(f64::MIN)),
                Bound::Included(q.upper.unwrap_or(f64::MAX)),
            )),
            other => panic!("unknown query kind {other}"),
        }
    }
}

impl Searcher for TantivySearcher {
    fn run(&self, q: &Query) -> QueryResult {
        let query = self.build(q);
        let (top, count) = self
            .searcher
            .search(&*query, &(TopDocs::with_limit(TOP_K), Count))
            .expect("search");
        QueryResult {
            total_hits: count as u64,
            top_scores: top.into_iter().map(|(s, _)| s).collect(),
        }
    }

    fn num_segments(&self) -> usize {
        self.searcher.segment_readers().len()
    }
}
