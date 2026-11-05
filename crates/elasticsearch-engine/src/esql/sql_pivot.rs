//! SQL PIVOT folds input batches into aggregate state instead of retaining
//! the source table. Memory depends on output groups, not input documents.
use super::*;
use crate::esql::partial::{Acc, AggKind};

const BATCH: usize = 1024;
/// Rows of one point-in-time page of a remote shard's PIVOT input: each
/// page is a search of the shard, so fewer, larger pages. (Tests page
/// their small shards in rows of a batch.)
#[cfg(not(test))]
const REMOTE_PAGE: usize = 8 * BATCH;
#[cfg(test)]
const REMOTE_PAGE: usize = BATCH;
/// Most documents one shard of a peer without point-in-time reads serves
/// in its single-fetch fallback.
const OLD_PEER_MAX_DOCS: usize = 100_000;

#[cfg(test)]
thread_local! {
    /// Test seam: a smaller [`OLD_PEER_MAX_DOCS`] on this thread.
    static OLD_PEER_MAX_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn old_peer_max_docs() -> usize {
    #[cfg(test)]
    if let Some(n) = OLD_PEER_MAX_OVERRIDE.with(|o| o.get()) {
        return n;
    }
    OLD_PEER_MAX_DOCS
}
/// Keep-alive of a remote shard's point in time between two pages.
const PIT_KEEP_ALIVE_MS: u64 = 60_000;
static NEXT_PIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn remote_rows_bound(schema:&IndexSchema,hits:&[Value])->u64 {
    let constants=schema.cols.iter().flat_map(|(_,column)|column.constants.values()).map(crate::memory::json_bytes).fold(0u64,u64::saturating_add);
    hits.iter().map(crate::memory::json_bytes).fold(0u64,u64::saturating_add).saturating_mul(128)
        .saturating_add((hits.len() as u64).saturating_mul(constants.saturating_mul(256).saturating_add((schema.cols.len() as u64).saturating_mul(128))))
}

/// Admit the whole next cartesian layer while its previous layer remains
/// alive. The returned reservation follows only this temporary key buffer.
fn expand_keys(keys:&[Vec<Val>],options:&[Val],max_buckets:usize)->Result<(Vec<Vec<Val>>,crate::memory::Reservation)> {
    let count=keys.len().saturating_mul(options.len());
    if count>max_buckets {return Err(EsError::new(429,"circuit_breaking_exception","SQL grouping expansion exceeds the bucket budget"));}
    let bytes=keys.iter().fold(0u64,|sum,key|options.iter().fold(sum,|sum,value|
        sum.saturating_add(crate::esql::exec::row_bytes(key).saturating_mul(2)).saturating_add((value.heap_bytes() as u64).saturating_mul(2)).saturating_add(128)));
    let memory=crate::memory::breakers().reserve_request(bytes,crate::esql::exec::ESQL_LABEL)?;
    let mut out=Vec::with_capacity(count);
    for key in keys {for value in options {
        let mut next=Vec::with_capacity(key.len()+1);next.extend(key.iter().cloned());next.push(value.clone());out.push(next);
    }}
    Ok((out,memory))
}

/// MIN/MAX may clone a wide scalar. Account that replacement before add,
/// then release the old retained value and unused speculative allowance.
fn add_acc_guarded(acc:&mut Acc,value:&Val,memory:&mut crate::memory::Reservation)->Result<()> {
    let previous=acc.retained_bytes() as u64;
    let bound=if matches!(acc,Acc::Min(_)|Acc::Max(_)){(value.heap_bytes() as u64).saturating_mul(2)}else{0};
    memory.grow(bound,crate::esql::exec::ESQL_LABEL)?;
    acc.add(value);
    let current=acc.retained_bytes() as u64;
    memory.shrink(previous.saturating_add(bound).saturating_sub(current));
    Ok(())
}

/// Whether doc values may stand in for `_source` on `target` (the
/// conditions of the direct SQL column read: no field security, stored
/// source without includes/excludes).
fn dv_allowed(target: &crate::search::Target) -> bool {
    target.field_access().is_none() && target.svc.source_stored_as_sent()
}

/// The doc-values field of a plain keyword (`keyword`) or float/double
/// (`numeric`) column on `target`, when its doc values hold exactly what
/// `_source` extraction would produce (same rules as the direct SQL read).
fn dv_field(column: &ColSrc, target: &crate::search::Target, lookup: &crate::mapping::FieldLookup, keyword: bool) -> Option<u16> {
    let index: &str = target.hit_index();
    if !column.constants.is_empty() || !column.per_index.is_empty() || column.ty == "unsupported" || column.ordered.iter().any(|i| i == index) { return None; }
    if !column.types.is_empty() && !index.is_empty() && !column.types.iter().any(|(_, indices)| indices.iter().any(|i| i == index)) { return None; }
    let field = lookup.fields.get(&column.path)?;
    let eligible = if keyword {
        column.ty == "keyword" && column.declared == "keyword" && field.declared == crate::mapping::EsType::Keyword
    } else {
        column.ty == "double" && matches!(column.declared.as_str(), "float" | "double") && field.declared.name() == column.declared && !field.ignore_malformed
    };
    if !eligible || !column.doc_values || column.template_id
        || !field.doc_values || field.null_value.is_some() || field.script.is_some()
        || field.normalizer.is_some() || column.normalizer.is_some()
        || field.ignore_above.is_some() || column.ignore_above.is_some() || field.is_multi_field
        || field.alias_path.is_some() || field.nested_path.is_some()
        || lookup.copy_sources.contains_key(&column.path) { return None; }
    field.core_id
}

/// Rows for `docs` like [`super::fast::sql_column_rows`], except that the
/// float/double columns marked `dv_only` are read from doc values alone.
///
/// ES SQL evaluates WHERE/ORDER BY scripts, GROUP BY sources and aggregate
/// inputs on doc values (`InternalSqlScriptUtils.docValue`, composite and
/// metric aggregations), never on `_source`; only projected values come
/// from the document. A doc without a doc value is therefore missing for
/// those columns, and the per-document `_source` fallback of the direct read
/// (which decodes whole stored documents for every sparse value, e.g. NOAA's
/// `TMIN`) is not needed for them.
pub(super) fn dv_column_rows(schema: &IndexSchema, wanted: &[bool], dv_only: &[bool], target: &crate::search::Target, lookup: &crate::mapping::FieldLookup,
    leaf: &rustic_core::query::LeafReader<'_>, docs: &[rustic_core::types::DocId]) -> Result<(Vec<Vec<Val>>, u64, crate::memory::Reservation)> {
    let reads: Vec<(usize, u16)> = if dv_only.iter().any(|d| *d) && dv_allowed(target) {
        schema.cols.iter().enumerate().filter(|(i, _)| dv_only[*i] && wanted[*i])
            .filter_map(|(i, (_, column))| dv_field(column, target, lookup, false).map(|field| (i, field))).collect()
    } else { Vec::new() };
    if reads.is_empty() { return super::fast::sql_column_rows(schema, wanted, target, lookup, leaf, docs); }
    let mut others = wanted.to_vec();
    for (column, _) in &reads { others[*column] = false; }
    let (mut rows, mut loaded, mut memory) = super::fast::sql_column_rows(schema, &others, target, lookup, leaf, docs)?;
    for &(column, field) in &reads {
        memory.grow(leaf.field_reader_allocation_bound(field), crate::esql::exec::ESQL_LABEL)?;
        // A single-valued column: one value or none per doc, no lists.
        if let rustic_core::query::LeafReader::Segment(segment) = leaf {
            if let Some(reader) = segment.fast_fields().get_f64(field) {
                memory.grow((docs.len() as u64).saturating_mul(64), crate::esql::exec::ESQL_LABEL)?;
                for (row, &doc) in rows.iter_mut().zip(docs) {
                    if let Some(x) = reader.get(doc) {
                        loaded += 1;
                        row[column] = Val::Double(x);
                    }
                }
                continue;
            }
        }
        let lists: Vec<Vec<f64>> = match leaf {
            rustic_core::query::LeafReader::Segment(segment) => {
                let ff = segment.fast_fields();
                if let Some(reader) = ff.get_f64(field) {
                    memory.grow((docs.len() as u64).saturating_mul(64), crate::esql::exec::ESQL_LABEL)?;
                    docs.iter().map(|&doc| reader.get(doc).into_iter().collect()).collect()
                } else if let Some(reader) = ff.get_multi_f64(field) {
                    for &doc in docs { memory.grow(leaf.field_values_allocation_bound(field, doc), crate::esql::exec::ESQL_LABEL)?; }
                    docs.iter().map(|&doc| reader.values_vec(doc)).collect()
                } else { vec![Vec::new(); docs.len()] }
            }
            rustic_core::query::LeafReader::Memtable(..) => {
                for &doc in docs { memory.grow(leaf.field_values_allocation_bound(field, doc), crate::esql::exec::ESQL_LABEL)?; }
                docs.iter().map(|&doc| leaf.f64_values(field, doc)).collect()
            }
        };
        for (row, list) in rows.iter_mut().zip(lists) {
            let mut values: Vec<Val> = list.into_iter().map(Val::Double).collect();
            values.sort_by(cmp_vals);
            let value = Val::from_list(values);
            if !value.is_null() { loaded += 1; }
            row[column] = value;
        }
    }
    Ok((rows, loaded, memory))
}

/// Whether `FROM`'s `load` reads every match for `rest` (no size/sort
/// pushdown): the shapes [`doc_value_rows`] may serve instead.
fn unbounded_load(rest: &[Cmd]) -> bool {
    for (k, cmd) in rest.iter().enumerate() {
        match cmd {
            Cmd::Limit(_) | Cmd::LimitBy(..) => return false,
            Cmd::Eval(_) | Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..) => {}
            Cmd::Sort(_) if k == 0 => return !matches!(rest.get(1), Some(Cmd::Limit(_))),
            _ => return true,
        }
    }
    false
}

/// Most candidate rows one batch of [`doc_value_rows`] reads.
const ROW_BATCH: usize = 1024;

/// SQL's table input (the matches of the pushed-down query with the
/// residual row filters applied), read in one pass over doc values: what
/// `FROM`'s batched `load` + residual `WHERE` produce, row for row and in
/// the same order, but filtering each batch as it is read (only matching
/// rows are retained), stopping at a row-local `LIMIT`, and reading
/// filter/sort/aggregate-only float columns without the `_source` fallback
/// ([`dv_column_rows`]). Covers none of `rest`.
fn doc_value_rows(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if schema.slots.is_some() || schema.targets.is_empty() || ctx.ts || ctx.esql { return Ok(None); }
    if residual.is_empty() && !unbounded_load(rest) { return Ok(None); }
    // Columns that can reach the output keep the projected-value read.
    let mut output: Vec<String> = Vec::new();
    for cmd in rest {
        match cmd {
            Cmd::Keep(names, _) => output.extend(names.iter().cloned()),
            Cmd::Eval(fields) => for field in fields {
                let mut refs = Vec::new();
                crate::esql::analyze::expr_refs(&field.expr, &mut refs);
                if let Some(filter) = &field.filter { crate::esql::analyze::expr_refs(filter, &mut refs); }
                output.extend(refs.into_iter().map(|(name, _)| name));
            },
            Cmd::Stats { by, .. } => for group in by {
                if let Expr::Col(name, _) = &group.expr { output.push(name.clone()); }
            },
            Cmd::Where(..) | Cmd::Sort(_) | Cmd::Limit(_) => {}
            _ => return Ok(None),
        }
    }
    let Some(needed) = needed_columns(ctx.full, schema) else { return Ok(None) };
    let wanted: Vec<bool> = schema.cols.iter().map(|(name, _)| needed.contains(name)).collect();
    let dv_only: Vec<bool> = schema.cols.iter().zip(&wanted).map(|((name, column), wanted)| *wanted && column.ty == "double" && !output.contains(name)).collect();
    if residual.is_empty() && !dv_only.iter().any(|d| *d) { return Ok(None); }
    // A row-local prefix ending in LIMIT n needs only the first n matches.
    let mut stop = None;
    if !residual.is_empty() {
        for cmd in rest {
            match cmd {
                Cmd::Eval(_) | Cmd::Keep(..) => continue,
                Cmd::Limit(n) => { stop = Some(*n); break; }
                _ => break,
            }
        }
    }
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&schema.targets)?;
    let filters: Vec<Cmd> = residual.iter().map(|(e, p)| Cmd::Where(e.clone(), *p)).collect();
    let mut table = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    let mut memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    if stop == Some(0) {
        memory.retain_for_request();
        return Ok(Some((table, 0)));
    }
    crate::search::for_each_match(&targets, query, ROW_BATCH, &mut |page| {
        let mut ids = page.doc_ids();
        loop {
            let docs: Vec<_> = ids.by_ref().take(ROW_BATCH).collect();
            if docs.is_empty() { break; }
            let (rows, loaded, _page_memory) = dv_column_rows(schema, &wanted, &dv_only, page.target, page.lookup, page.leaf, &docs)?;
            ctx.documents_found += rows.len() as u64;
            ctx.values_loaded += loaded;
            let mut batch = Table { cols: table.cols.clone(), rows };
            for filter in &filters { batch = apply(&mut *ctx, filter, batch)?; }
            memory.grow(super::fast::rows_bytes(&batch.rows), crate::esql::exec::ESQL_LABEL)?;
            table.rows.extend(batch.rows);
            if stop.is_some_and(|n| table.rows.len() >= n) { return Ok(false); }
        }
        Ok(true)
    })?;
    memory.retain_for_request();
    Ok(Some((table, 0)))
}

/// PIVOT aggregate states, keyed by grouping values with `WireVal`
/// equality (a value's type and bits): what the length-delimited JSON of
/// the wire values identifies, without encoding the common keys per row.
struct PivotStates {
    /// Compound and non-string keys: the JSON of their wire values.
    index: HashMap<String, usize>,
    /// One string grouping value (equal only to the same string).
    strs: HashMap<String, usize>,
    /// No grouping column: the one state.
    empty: Option<usize>,
    states: Vec<(Vec<Val>, Vec<(bool, Acc)>)>,
    key_bytes: usize,
    metric_bytes: usize,
    memory: crate::memory::Reservation,
    /// Pivot values (state cells per key).
    values: usize,
    kind: AggKind,
    max_buckets: usize,
}

impl PivotStates {
    /// A new key's state, admitted against the bucket, key-memory and
    /// request budgets (`encoded_len`: its JSON key length).
    fn admit(&mut self, key: Vec<Val>, encoded_len: usize) -> Result<usize> {
        self.key_bytes = self.key_bytes.saturating_add(encoded_len.saturating_mul(3));
        if (self.states.len()+1).saturating_mul(self.values.max(1)) > self.max_buckets || self.key_bytes.saturating_add(self.metric_bytes).saturating_add((self.states.len()+1).saturating_mul(self.values).saturating_mul(std::mem::size_of::<Acc>())) > 64*1024*1024 {
            return Err(EsError::new(429,"circuit_breaking_exception","PIVOT aggregate state exceeds the bucket or key-memory budget"));
        }
        self.memory.grow((encoded_len as u64).saturating_mul(3).saturating_add(crate::esql::exec::row_bytes(&key).saturating_mul(2)).saturating_add((self.values as u64).saturating_mul((std::mem::size_of::<Acc>() as u64+16)*2)).saturating_add(256),crate::esql::exec::ESQL_LABEL)?;
        let id = self.states.len();
        let kind = self.kind;
        self.states.push((key, (0..self.values).map(|_| (false, Acc::new(kind))).collect()));
        Ok(id)
    }

    fn empty_id(&mut self) -> Result<usize> {
        if let Some(id) = self.empty { return Ok(id); }
        // `[]`
        let id = self.admit(Vec::new(), 2)?;
        self.empty = Some(id);
        Ok(id)
    }

    fn str_id(&mut self, s: &str) -> Result<usize> {
        if let Some(&id) = self.strs.get(s) { return Ok(id); }
        // `[{"S":"…"}]`
        let id = self.admit(vec![Val::Str(s.to_string())], s.len().saturating_add(9))?;
        self.strs.insert(s.to_string(), id);
        Ok(id)
    }

    fn id_of(&mut self, key: Vec<Val>) -> Result<usize> {
        match key.as_slice() {
            [] => return self.empty_id(),
            [Val::Str(s)] => return self.str_id(s),
            _ => {}
        }
        // Length-delimited JSON avoids collisions between compound keys.
        let encoded = serde_json::to_string(&key.iter().map(crate::esql::partial::WireVal::from).collect::<Vec<_>>()).map_err(|e| EsError::internal(e.to_string()))?;
        if let Some(&id) = self.index.get(&encoded) { return Ok(id); }
        let id = self.admit(key, encoded.len())?;
        self.index.insert(encoded, id);
        Ok(id)
    }

    /// Fold one row's aggregate input into the selected cells of `id`.
    fn add(&mut self, id: usize, selected: &[bool], star: bool, value: &Val) -> Result<()> {
        let PivotStates { states, memory, metric_bytes, key_bytes, kind, .. } = self;
        // Only MIN/MAX retain variable-width values to account.
        let guarded = matches!(kind, AggKind::Min | AggKind::Max);
        for ((seen, acc), selected) in states[id].1.iter_mut().zip(selected) {
            if !selected { continue; }
            *seen=true;
            let previous_bytes = acc.retained_bytes();
            let mut add = |acc: &mut Acc, v: &Val| -> Result<()> { if guarded { add_acc_guarded(acc,v,&mut *memory) } else { acc.add(v); Ok(()) } };
            match (star, value) {
                (true, _) => add(acc,&Val::Null)?,
                (_, Val::Null) => {},
                (_, Val::Multi(v)) => for v in v.iter().filter(|v| !v.is_null()) { add(acc,v)?; },
                (_, v) => add(acc,v)?,
            }
            *metric_bytes = metric_bytes.saturating_sub(previous_bytes).saturating_add(acc.retained_bytes());
            if key_bytes.saturating_add(*metric_bytes) > 64*1024*1024 {
                return Err(EsError::new(429,"circuit_breaking_exception","PIVOT aggregate state exceeds the memory budget"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: fold every PIVOT row through the generic row path.
    pub(crate) static PIVOT_FAST_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: docs the PIVOT ordinal path folded on this thread.
    pub(crate) static PIVOT_FAST_DOCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn pivot_fast_enabled() -> bool {
    #[cfg(test)]
    if PIVOT_FAST_DISABLED.with(|d| d.get()) { return false; }
    true
}

/// The PIVOT ordinal path's columns (positions in the PIVOT input schema)
/// and the pivot literals, in pivot-value order.
struct PivotFast {
    pivot: usize,
    group: Option<usize>,
    arg: Option<usize>,
    tests: Vec<String>,
}

/// One leaf's single-valued keyword column: a segment's, by ordinal; a
/// memtable's (no ordinals), by term.
enum PivotKeyword<'a> {
    Ords(rustic_core::codec::fastfields::KeywordReader<'a>),
    Terms(&'a rustic_core::fast_fields::KeywordColumn),
}

impl PivotKeyword<'_> {
    fn num_terms(&self) -> usize {
        match self { PivotKeyword::Ords(r) => r.num_terms(), PivotKeyword::Terms(_) => 0 }
    }
}

/// One leaf's single-valued float column.
enum PivotFloat<'a> {
    Column(rustic_core::codec::fastfields::ColumnReader<'a, f64>),
    Memtable(&'a rustic_core::fast_fields::F64Column),
}

impl PivotFloat<'_> {
    fn get(&self, doc: rustic_core::types::DocId) -> Option<f64> {
        match self { PivotFloat::Column(c) => c.get(doc), PivotFloat::Memtable(c) => c.get(doc) }
    }
}

/// One leaf's single-valued doc-value columns of a [`PivotFast`] plan.
struct PivotReaders<'a> {
    pivot: PivotKeyword<'a>,
    group: Option<PivotKeyword<'a>>,
    arg: Option<PivotFloat<'a>>,
}

impl<'a> PivotReaders<'a> {
    /// The readers of `page`'s leaf, when its columns read exactly what
    /// the generic column read would (see [`dv_field`]): single-valued
    /// keyword columns and a single-valued float column, on a segment or a
    /// memtable (what a refresh publishes before it is sealed).
    fn open(plan: &PivotFast, schema: &IndexSchema, page: &crate::search::MatchPage<'a>) -> Option<Self> {
        if !dv_allowed(page.target) { return None; }
        let field = |c: usize, keyword: bool| dv_field(&schema.cols[c].1, page.target, page.lookup, keyword);
        match page.leaf {
            rustic_core::query::LeafReader::Segment(segment) => {
                let keyword = |c: usize| -> Option<PivotKeyword<'a>> {
                    let f = field(c, true)?;
                    if segment.keyword_from_postings(f) { return None; }
                    segment.keyword_reader(f).map(PivotKeyword::Ords)
                };
                let pivot = keyword(plan.pivot)?;
                let group = match plan.group { Some(c) => Some(keyword(c)?), None => None };
                let arg = match plan.arg {
                    Some(c) => Some(PivotFloat::Column(segment.fast_fields().get_f64(field(c, false)?)?)),
                    None => None,
                };
                Some(PivotReaders { pivot, group, arg })
            }
            rustic_core::query::LeafReader::Memtable(writer, ..) => {
                let ff = writer.fast_fields();
                let keyword = |c: usize| -> Option<PivotKeyword<'a>> { ff.get_keyword(field(c, true)?).map(PivotKeyword::Terms) };
                let pivot = keyword(plan.pivot)?;
                let group = match plan.group { Some(c) => Some(keyword(c)?), None => None };
                let arg = match plan.arg {
                    Some(c) => Some(PivotFloat::Memtable(ff.get_f64(field(c, false)?)?)),
                    None => None,
                };
                Some(PivotReaders { pivot, group, arg })
            }
        }
    }
}

/// Per leaf `(target, leaf)`: pivot ordinal -> selected-values mask, group
/// ordinal -> group state (both filled lazily), and their charge.
type PivotLeafTables = ((usize, usize), Vec<u64>, Vec<u32>, crate::memory::Reservation);

const MASK_UNSEEN: u64 = u64::MAX;
const MASK_INVALID: u64 = u64::MAX - 1;
const GROUP_UNSEEN: u32 = u32::MAX;
const GROUP_INVALID: u32 = u32::MAX - 1;
/// A valid UTF-8 group term whose group is created on first use.
const GROUP_PENDING: u32 = u32::MAX - 2;

/// One batch of the PIVOT ordinal path: exactly what reading `docs` with
/// [`dv_column_rows`] and folding them with `consume` does, in doc order,
/// but from ordinals and a float column. A doc without a (UTF-8) value in a
/// keyword column takes the generic read (`read`, which keeps its `_source`
/// fallback) and `consume`, in its place in doc order.
#[allow(clippy::too_many_arguments)]
fn pivot_fast_batch(ctx: &mut Ctx, fold: &mut PivotStates, plan: &PivotFast, readers: &PivotReaders<'_>, tables: &mut Option<PivotLeafTables>, leaf: (usize, usize), docs: &[rustic_core::types::DocId],
    read: &mut dyn FnMut(&mut Ctx, &mut PivotStates, &[rustic_core::types::DocId]) -> Result<Vec<Vec<Val>>>,
    consume: &dyn Fn(&mut Ctx, &mut PivotStates, Vec<Vec<Val>>) -> Result<()>) -> Result<()> {
    if tables.as_ref().is_none_or(|t| t.0 != leaf) {
        drop(tables.take());
        let pivot_terms = readers.pivot.num_terms();
        let group_terms = readers.group.as_ref().map_or(0, |g| g.num_terms());
        let memory = crate::memory::breakers().reserve_request((pivot_terms as u64).saturating_mul(8).saturating_add((group_terms as u64).saturating_mul(4)).saturating_add(128), crate::esql::exec::ESQL_LABEL)?;
        *tables = Some((leaf, vec![MASK_UNSEEN; pivot_terms], vec![GROUP_UNSEEN; group_terms], memory));
    }
    let Some((_, masks, group_ids, _)) = tables.as_mut() else { return Err(EsError::internal("PIVOT ordinal tables missing")) };
    // Pass 1: each doc's mask (0: no pivot value, skipped as the row path
    // skips it), or `None` when the doc needs the generic read.
    let mut plans: Vec<Option<u64>> = Vec::with_capacity(docs.len());
    let mut missing = Vec::new();
    for &doc in docs {
        let test = |term: &str| plan.tests.iter().enumerate().filter(|(_, t)| t.as_str() == term).fold(0u64, |m, (k, _)| m | 1 << k);
        let mask = match &readers.pivot {
            PivotKeyword::Ords(r) => {
                let ord = r.ord(doc);
                match masks.get_mut(ord as usize) {
                    None => MASK_INVALID,
                    Some(slot) => {
                        if *slot == MASK_UNSEEN {
                            *slot = match r.term(ord) { None => MASK_INVALID, Some(term) => test(&term) };
                        }
                        *slot
                    }
                }
            }
            PivotKeyword::Terms(c) => c.get(doc).map_or(MASK_INVALID, test),
        };
        let group_ok = mask == 0 || mask == MASK_INVALID || match &readers.group {
            None => true,
            Some(PivotKeyword::Ords(g)) => match group_ids.get_mut(g.ord(doc) as usize) {
                None => false,
                Some(state) => {
                    if *state == GROUP_UNSEEN { *state = if g.term(g.ord(doc)).is_some() { GROUP_PENDING } else { GROUP_INVALID }; }
                    *state != GROUP_INVALID
                }
            },
            Some(PivotKeyword::Terms(c)) => c.get(doc).is_some(),
        };
        if mask == MASK_INVALID || !group_ok {
            plans.push(None);
            missing.push(doc);
        } else {
            plans.push(Some(mask));
        }
    }
    let mut fallback = if missing.is_empty() { Vec::new().into_iter() } else { read(ctx, fold, &missing)?.into_iter() };
    // Pass 2, in doc order.
    let mut selected = vec![false; plan.tests.len()];
    let mut folded = 0usize;
    for (&doc, plan_mask) in docs.iter().zip(plans) {
        let Some(mask) = plan_mask else {
            let row = fallback.next().ok_or_else(|| EsError::internal("PIVOT fallback row missing"))?;
            consume(ctx, fold, vec![row])?;
            continue;
        };
        folded += 1;
        if mask == 0 { continue; }
        for (k, slot) in selected.iter_mut().enumerate() { *slot = mask >> k & 1 == 1; }
        let id = match &readers.group {
            None => fold.empty_id()?,
            Some(PivotKeyword::Terms(c)) => fold.str_id(c.get(doc).ok_or_else(|| EsError::internal("PIVOT group term missing"))?)?,
            Some(PivotKeyword::Ords(g)) => {
                let ord = g.ord(doc);
                let state = group_ids[ord as usize];
                if state < GROUP_PENDING { state as usize } else {
                    let term = g.term(ord).ok_or_else(|| EsError::internal("keyword ordinal without a term"))?;
                    let id = fold.str_id(&term)?;
                    group_ids[ord as usize] = u32::try_from(id).ok().filter(|id| *id < GROUP_PENDING).ok_or_else(|| EsError::internal("too many PIVOT groups"))?;
                    id
                }
            }
        };
        let value = match &readers.arg { Some(r) => r.get(doc).map(Val::Double).unwrap_or(Val::Null), None => Val::Null };
        fold.add(id, &selected, plan.arg.is_none(), &value)?;
    }
    ctx.documents_found += folded as u64;
    #[cfg(test)]
    PIVOT_FAST_DOCS.with(|n| n.set(n.get() + folded));
    Ok(())
}

pub(super) fn try_run(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    let prefix = rest.iter().take_while(|cmd| matches!(cmd, Cmd::Eval(_) | Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..) | Cmd::Where(..))).count();
    let Some(Cmd::SqlPivot { aggregate, column, values, pos }) = rest.get(prefix) else { return doc_value_rows(ctx, schema, query, residual, rest) };
    let Expr::Call(name, args, _) = aggregate else { return Ok(None) };
    let Some(mut kind) = AggKind::of(name) else { return Ok(None) };
    // Exact distinct sets need their own byte budget; leave that executor alone.
    if kind == AggKind::CountDistinct { return Ok(None); }
    let mut local: Vec<Cmd> = residual.iter().map(|(e,p)| Cmd::Where(e.clone(), *p)).collect();
    local.extend_from_slice(&rest[..prefix]);
    // An explicit leading projection defines the complete PIVOT input.
    // Compact the physical schema too: avoid allocating one null cell per
    // unrelated mapped field for every matching document.
    let compact;
    let schema = if let Some(Cmd::Keep(names, _)) = local.first().filter(|cmd| matches!(cmd, Cmd::Keep(names, _) if !names.iter().any(|name| name.contains('*')))) {
        compact = IndexSchema { cols: schema.cols.iter().filter(|(name, _)| names.contains(name)).cloned().collect(), targets: schema.targets.clone(), slots: schema.slots.clone(), projection: None };
        &compact
    } else { schema };
    let base = source_cols(schema, &[]);
    let mut empty = Table { cols: base.clone(), rows: Vec::new() };
    for cmd in &local { empty = apply(ctx, cmd, empty)?; }
    // Use the existing interpreter for validation and exact output metadata.
    let mut output = sql_pivot(ctx, empty.clone(), aggregate, column, values, *pos)?;
    let arg = match (kind, args.as_slice()) {
        (AggKind::Count, [Expr::Star]) => None,
        (_, [expr]) => Some(bind(ctx, expr, &empty.cols)?),
        _ => return Ok(None),
    };
    if kind == AggKind::Avg && arg.as_ref().and_then(|e| type_of(e, &empty.cols)).as_deref() == Some("integer") { kind = AggKind::AvgInt; }
    let mut refs = Vec::new();
    crate::esql::analyze::expr_refs(aggregate, &mut refs);
    let groups: Vec<usize> = empty.cols.iter().enumerate().filter(|(_,c)| c.name != *column && !refs.iter().any(|(n,_)| n == &c.name)).map(|(i,_)| i).collect();
    let filters: Vec<Expr> = values.iter().map(|(v,_)| bind(ctx, &Expr::Bin(BinOp::Eq, Box::new(Expr::Col(column.clone(), *pos)), Box::new(Expr::Lit(v.clone()))), &empty.cols)).collect::<Result<_>>()?;
    // Single string keys of a `column = 'literal'` keyword test need no
    // expression evaluation: (column index, literal) per pivot value.
    let keyword_tests: Vec<Option<(usize, String)>> = filters.iter().map(|e| match e {
        Expr::Bin(BinOp::Eq, a, b) => match (a.as_ref(), b.as_ref()) {
            (Expr::Idx(i), Expr::Lit(Val::Str(s))) if empty.cols.get(*i).is_some_and(|c| c.ty == "keyword") => Some((*i, s.clone())),
            _ => None,
        },
        _ => None,
    }).collect();
    let mut fold = PivotStates { index: HashMap::new(), strs: HashMap::new(), empty: None, states: Vec::new(), key_bytes: 0, metric_bytes: 0,
        memory: crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?, values: values.len(), kind,
        max_buckets: ctx.node.map(|n| n.max_buckets()).unwrap_or(65_536) as usize };
    let max_buckets = fold.max_buckets;
    let consume = |ctx: &mut Ctx, fold: &mut PivotStates, rows: Vec<Vec<Val>>| -> Result<()> {
        ctx.documents_found += rows.len() as u64;
        let mut table = Table { cols: base.clone(), rows };
        for cmd in &local { table = apply(ctx, cmd, table)?; }
        let mut selected = vec![false; filters.len()];
        for row in table.rows {
            let mut any = false;
            for ((slot, e), test) in selected.iter_mut().zip(&filters).zip(&keyword_tests) {
                *slot = match (test, test.as_ref().and_then(|(i, _)| row.get(*i))) {
                    (Some((_, literal)), Some(Val::Str(s))) => s == literal,
                    _ => matches!(eval(e, &row, &ctx.fctx)?, Val::Bool(true)),
                };
                any |= *slot;
            }
            if !any { continue; }
            // Scalar keys (no multi-valued grouping value) are the row's own
            // values: no cartesian expansion, no transient key buffers.
            if groups.iter().all(|&col| !matches!(row[col], Val::Multi(_))) {
                let value = match &arg { Some(e) => eval(e, &row, &ctx.fctx)?, None => Val::Null };
                let id = match groups.as_slice() {
                    [] => fold.empty_id()?,
                    [col] => match &row[*col] {
                        Val::Str(s) => fold.str_id(s)?,
                        v => fold.id_of(vec![v.clone()])?,
                    },
                    cols => fold.id_of(cols.iter().map(|&col| row[col].clone()).collect())?,
                };
                fold.add(id, &selected, arg.is_none(), &value)?;
                continue;
            }
            let mut key_memory=crate::memory::breakers().reserve_request(64,crate::esql::exec::ESQL_LABEL)?;
            let mut keys = vec![Vec::new()];
            for &col in &groups {
                let options = match &row[col] { Val::Multi(v) => v.as_slice(), v => std::slice::from_ref(v) };
                if keys.len().saturating_mul(options.len()) > max_buckets { return Err(EsError::new(429,"circuit_breaking_exception","PIVOT grouping expansion exceeds the bucket budget")); }
                let (next,next_memory)=expand_keys(&keys,options,max_buckets)?;
                keys=next;key_memory=next_memory;
            }
            let value = match &arg { Some(e) => eval(e, &row, &ctx.fctx)?, None => Val::Null };
            for key in keys {
                let encoded_memory=crate::memory::breakers().reserve_request(crate::esql::exec::row_bytes(&key).saturating_mul(32),crate::esql::exec::ESQL_LABEL)?;
                let id = fold.id_of(key)?;
                fold.add(id, &selected, arg.is_none(), &value)?;
                drop(encoded_memory);
            }
            drop(key_memory);
        }
        Ok(())
    };
    // Local targets are pinned for the entire fold, including concurrent refreshes.
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let slots = schema.slots.clone().unwrap_or_else(|| (0..schema.targets.len()).map(crate::esql::remote::Slot::Local).collect());
    let mut snapshots = HashMap::new();
    for slot in &slots {
        if let crate::esql::remote::Slot::Local(i) = slot {
            snapshots.insert(*i, node.snapshot(std::slice::from_ref(&schema.targets[*i]))?);
        }
    }
    // A leading explicit projection can avoid extracting unused wide fields.
    let wanted: Vec<bool> = match local.first() {
        Some(Cmd::Keep(names,_)) if !names.iter().any(|n| n.contains('*')) => schema.cols.iter().map(|(n,_)| names.contains(n)).collect(),
        _ => vec![true; schema.cols.len()],
    };
    // ES folds PIVOT into a composite aggregation whose metric reads doc
    // values: a float/double column only the aggregate reads skips the
    // per-document `_source` fallback for docs without a value.
    let plain_input = local.iter().all(|c| matches!(c, Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Where(..)));
    let dv_only: Vec<bool> = schema.cols.iter().zip(&wanted).map(|((name, c), wanted)|
        *wanted && plain_input && c.ty == "double" && name != column && refs.iter().any(|(r, _)| r == name)).collect();
    // ES runs this PIVOT as a composite aggregation over doc-value ordinals
    // with a doc-values metric: one keyword pivot column tested against
    // literals, at most one keyword grouping column and a doc-values-only
    // float argument (or `*`) read straight from the columns per doc.
    let fast: Option<PivotFast> = (|| {
        if !pivot_fast_enabled() || !local.iter().all(|c| matches!(c, Cmd::Keep(..) | Cmd::Drop(..))) || values.len() > 62 { return None; }
        let source = |i: usize| schema.cols.iter().position(|(n, _)| *n == empty.cols[i].name);
        let mut tests = Vec::with_capacity(keyword_tests.len());
        let mut pivot = None;
        for test in &keyword_tests {
            let (i, literal) = test.as_ref()?;
            if pivot.is_some_and(|p| p != *i) { return None; }
            pivot = Some(*i);
            tests.push(literal.clone());
        }
        let pivot = source(pivot?)?;
        if schema.cols[pivot].1.ty != "keyword" { return None; }
        let group = match groups.as_slice() {
            [] => None,
            [g] if empty.cols[*g].ty == "keyword" => Some(source(*g)?),
            _ => return None,
        };
        let arg = match &arg {
            None => None,
            Some(Expr::Idx(i)) => { let c = source(*i)?; if !dv_only[c] { return None; } Some(c) }
            Some(_) => return None,
        };
        Some(PivotFast { pivot, group, arg, tests })
    })();
    // With no renames/computed input, the pivot membership test is safe to
    // push into the index. Residual evaluation still enforces SQL semantics.
    // The ordinal path tests each candidate's pivot value from its ordinal
    // anyway: behind another filter (pivot_3pages' `elevation > 1600`) the
    // terms clause would only add a postings intersection over every doc of
    // its values (US is most of NOAA), so it is pushed only alone (it then
    // bounds the candidates) or for the row path and remote shards.
    let membership_pushdown = fast.is_none() || schema.slots.is_some() || *query == json!({"match_all": {}});
    let mut input_query = query.clone();
    if membership_pushdown && local.iter().all(|c| matches!(c,Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Where(..))) {
        let types = schema.cols.iter().map(|(n,c)| (n.clone(),c.ty.clone())).collect();
        let membership = Expr::In(Box::new(Expr::Col(column.clone(),*pos)),values.iter().map(|(v,_)| Expr::Lit(v.clone())).collect(),false);
        if let Some((q,false)) = to_dsl(&membership,&types,true) {
            input_query = json!({"bool":{"filter":[query,q]}});
        }
    }
    let query = &input_query;
    let mut leaf_tables: Option<PivotLeafTables> = None;
    for slot in slots {
        match slot {
            crate::esql::remote::Slot::Local(i) => {
                let targets = snapshots.remove(&i).ok_or_else(|| EsError::internal("missing PIVOT snapshot"))?;
                crate::search::for_each_match(&targets, query, BATCH, &mut |page| {
                    let readers = fast.as_ref().and_then(|plan| PivotReaders::open(plan, schema, page));
                    // Match-all pages may represent an entire leaf. Subdivide them too.
                    let mut ids = page.doc_ids();
                    loop {
                        let docs: Vec<_> = ids.by_ref().take(BATCH).collect();
                        if docs.is_empty() { break; }
                        if let (Some(plan), Some(readers)) = (fast.as_ref(), readers.as_ref()) {
                            pivot_fast_batch(ctx, &mut fold, plan, readers, &mut leaf_tables, (page.target_idx, page.leaf_idx), &docs, &mut |ctx, fold, missing| {
                                let (rows, loaded, _page_memory) = dv_column_rows(schema, &wanted, &dv_only, page.target, page.lookup, page.leaf, missing)?;
                                ctx.values_loaded += loaded;
                                let _ = fold;
                                Ok(rows)
                            }, &consume)?;
                            continue;
                        }
                        let (rows, loaded, _page_memory) = dv_column_rows(schema, &wanted, &dv_only, page.target, page.lookup, page.leaf, &docs)?;
                        ctx.values_loaded += loaded;
                        consume(ctx, &mut fold, rows)?;
                    }
                    Ok(true)
                })?;
            }
            crate::esql::remote::Slot::Remote(shard) => {
                let source = ctx.shards.clone().ok_or_else(|| EsError::internal("no shard source"))?;
                // Each page is a separate fetch. Like ES `search_after`, the
                // pages are consistent only against a point in time: the
                // shard's current view changes between them (seals and
                // merges move documents between leaves, updates and deletes
                // land), so the first page retains the copy's view on its
                // node and every page reads that view in `_shard_doc` order,
                // as the local slots read their snapshots.
                use crate::esql::remote::{FetchPit, FetchSpec, PitPhase};
                let id = format!("sql-pivot-{}-{}-{}", node.name, crate::date::now_millis(), NEXT_PIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
                let pit = |phase| Some(FetchPit { id: id.clone(), keep_alive_ms: PIT_KEEP_ALIVE_MS, phase });
                let mut opened = true;
                let mut pages = |ctx: &mut Ctx| -> Result<()> {
                    let mut body = search_body(schema, &[], &None, query, vec![json!("_shard_doc")]);
                    let mut phase = PitPhase::Open;
                    loop {
                        let open = phase == PitPhase::Open;
                        let spec = FetchSpec { shards:vec![shard.clone()], query:query.clone(), body:body.clone(), size:Some(REMOTE_PAGE), probes:vec![], agg:None, group_limit:None, pit:pit(phase), fast:None };
                        phase = PitPhase::Read;
                        let mut page_memory=crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?;
                        let mut results = source.fetch(&spec,&mut page_memory)?;
                        if results.len()!=1 { return Err(EsError::internal("PIVOT shard response missing")); }
                        if open && !results[0].pit {
                            // A node that does not know points in time served
                            // that page from its live view: read the shard in
                            // one fetch instead (single-phase, consistent).
                            opened = false;
                            // Bounded: one response holds at most this many.
                            let all = FetchSpec { shards:vec![shard.clone()], query:query.clone(), body:search_body(schema, &[], &None, query, Vec::new()), size:Some(old_peer_max_docs() + 1), probes:vec![], agg:None, group_limit:None, pit:None, fast:None };
                            let mut fallback_memory=crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?;
                            let mut results = source.fetch(&all,&mut fallback_memory)?;
                            if results.len()!=1 { return Err(EsError::internal("PIVOT shard response missing")); }
                            let hits = std::mem::take(&mut results[0].hits);
                            let max = old_peer_max_docs();
                            if hits.len() > max {
                                return Err(EsError::new(400, "illegal_argument_exception", format!(
                                    "PIVOT over [{}][{}] on a node without point-in-time reads: more than {max} documents match; upgrade the node or narrow the query", shard.index, shard.shard)));
                            }
                            fallback_memory.grow(remote_rows_bound(schema,&hits),crate::esql::exec::ESQL_LABEL)?;
                            let rows = hits.iter().map(|hit| schema.cols.iter().map(|(_,c)| extract(&hit["_source"], c, hit["_index"].as_str().unwrap_or(""))).collect()).collect();
                            return consume(ctx, &mut fold, rows);
                        }
                        let hits = std::mem::take(&mut results[0].hits);
                        if hits.is_empty() { return Ok(()); }
                        let after = hits.last().unwrap()["sort"].clone();
                        if !after.is_array() || body.get("search_after")==Some(&after) { return Err(EsError::internal("PIVOT shard cursor did not advance")); }
                        let done = hits.len()<REMOTE_PAGE;
                        page_memory.grow(remote_rows_bound(schema,&hits),crate::esql::exec::ESQL_LABEL)?;
                        let rows = hits.iter().map(|hit| schema.cols.iter().map(|(_,c)| extract(&hit["_source"], c, hit["_index"].as_str().unwrap_or(""))).collect()).collect();
                        consume(ctx, &mut fold, rows)?;
                        if done { return Ok(()); }
                        body["search_after"]=after;
                    }
                };
                let read = pages(ctx);
                // Released on every outcome (an error, a cancelled fold);
                // a close that does not arrive expires with the keep-alive.
                if opened {
                    let close = FetchSpec { shards:vec![shard.clone()], query:query.clone(), body:json!({}), size:Some(0), probes:vec![], agg:None, group_limit:None, pit:pit(PitPhase::Close), fast:None };
                    let mut close_memory=crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?;
                    if let Err(e) = source.fetch(&close,&mut close_memory) {
                        tracing::debug!("closing the PIVOT point in time [{id}] failed: {e}");
                    }
                }
                read?;
            }
        }
    }
    // Output conversion can transiently hold state plus wire and result values.
    let PivotStates { index, strs, states, memory: mut state_memory, .. } = fold;
    drop((index, strs));
    state_memory.grow(state_memory.bytes().saturating_mul(2),crate::esql::exec::ESQL_LABEL)?;
    for (mut key, cells) in states {
        for (seen,acc) in cells {
            let (value,overflow) = acc.state().finish(kind);
            if overflow { sum_overflow_warning(&ctx.fctx,*pos); }
            key.push(if seen { value } else { Val::Null });
        }
        output.rows.push(key);
    }
    let retained=super::fast::rows_bytes(&output.rows);
    if retained>state_memory.bytes(){state_memory.grow(retained-state_memory.bytes(),crate::esql::exec::ESQL_LABEL)?;}
    else{state_memory.shrink(state_memory.bytes()-retained);}
    state_memory.retain_for_request();
    Ok(Some((output,prefix+1)))
}

/// SQL COUNT(*) groups keyed like the generic `STATS` (`Val::key` of each
/// grouping value), a single string key looked up without encoding it
/// (`Val::key` of a string is `s` + the string: equal only to itself).
struct CountGroups {
    index: HashMap<Vec<String>, usize>,
    strs: HashMap<String, usize>,
    groups: Vec<(Vec<Val>, i64)>,
    bytes: usize,
    max_buckets: usize,
    memory: crate::memory::Reservation,
}

impl CountGroups {
    /// A new group, admitted against the bucket, key-memory and request
    /// budgets (`key_lens`: the summed `Val::key` lengths).
    fn admit(&mut self, key: Vec<Val>, key_lens: usize) -> Result<usize> {
        self.bytes = self.bytes.saturating_add(key_lens.saturating_mul(3)).saturating_add(128);
        if self.groups.len() >= self.max_buckets || self.bytes > 64*1024*1024 {
            return Err(EsError::new(429,"circuit_breaking_exception","SQL count grouping exceeds bucket or memory budget"));
        }
        self.memory.grow(256u64.saturating_add((key_lens as u64).saturating_mul(3)).saturating_add(crate::esql::exec::row_bytes(&key).saturating_mul(2)),crate::esql::exec::ESQL_LABEL)?;
        let id = self.groups.len();
        self.groups.push((key, 0));
        Ok(id)
    }

    fn str_id(&mut self, s: &str) -> Result<usize> {
        if let Some(&id) = self.strs.get(s) { return Ok(id); }
        let id = self.admit(vec![Val::Str(s.to_string())], s.len().saturating_add(1))?;
        self.strs.insert(s.to_string(), id);
        Ok(id)
    }

    fn id_of(&mut self, key: Vec<Val>) -> Result<usize> {
        if let [Val::Str(s)] = key.as_slice() { return self.str_id(s); }
        let encoded: Vec<String> = key.iter().map(Val::key).collect();
        if let Some(&id) = self.index.get(&encoded) { return Ok(id); }
        let lens = encoded.iter().map(String::len).sum();
        let id = self.admit(key, lens)?;
        self.index.insert(encoded, id);
        Ok(id)
    }

    /// Count one row in the group of every combination of its grouping
    /// values (`bound`; multi-values expand as the generic `STATS` does).
    fn count_row(&mut self, bound: &[Expr], row: &[Val], fctx: &FnCtx) -> Result<()> {
        if let [Expr::Idx(i)] = bound {
            if let Some(Val::Str(s)) = row.get(*i) {
                let id = self.str_id(s)?;
                self.groups[id].1 += 1;
                return Ok(());
            }
        }
        let max_buckets = self.max_buckets;
        let mut key_memory=crate::memory::breakers().reserve_request(64,crate::esql::exec::ESQL_LABEL)?;
        let mut keys = vec![Vec::new()];
        for expr in bound {
            let value = eval(expr, row, fctx)?;
            let options = match value { Val::Multi(values) => values, value => vec![value] };
            if keys.len().saturating_mul(options.len()) > max_buckets {
                return Err(EsError::new(429,"circuit_breaking_exception","SQL count grouping expansion exceeds bucket budget"));
            }
            let (next,next_memory)=expand_keys(&keys,&options,max_buckets)?;
            keys=next;key_memory=next_memory;
        }
        for key in keys {
            let encoded_memory=crate::memory::breakers().reserve_request(crate::esql::exec::row_bytes(&key).saturating_mul(32),crate::esql::exec::ESQL_LABEL)?;
            let id = self.id_of(key)?;
            self.groups[id].1 += 1;
            drop(encoded_memory);
        }
        drop(key_memory);
        Ok(())
    }
}

/// Ordinal → group of the current segment: not seen yet.
const ORD_UNSEEN: u32 = u32::MAX;
/// Ordinal → group of the current segment: its term is not UTF-8 (the
/// column read finds no value and reads `_source`).
const ORD_INVALID: u32 = u32::MAX - 1;

/// SQL COUNT(*) grouped by plain source columns needs only the grouping keys
/// and counters, never an input table or a list of member row IDs per group.
/// One keyword key counts each segment by doc-value ordinal (ES: a composite
/// `terms` source over ordinals), resolving each ordinal's term once.
pub(super) fn grouped_count(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, rest: &[Cmd]) -> Result<Option<Table>> {
    let Some(Cmd::Stats { aggs, by, inline: false, .. }) = rest.first() else { return Ok(None) };
    if schema.slots.is_some() || by.is_empty() || aggs.is_empty()
        || !aggs.iter().all(|a| a.filter.is_none() && matches!(&a.expr, Expr::Call(name, args, _) if name=="COUNT" && matches!(args.as_slice(), [Expr::Star])))
        || !by.iter().all(|g| g.filter.is_none() && matches!(&g.expr, Expr::Col(..))) {
        return Ok(None);
    }
    let empty = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    let mut output = apply(ctx, &rest[0], empty)?;
    let names: Vec<_> = by.iter().filter_map(|g| match &g.expr { Expr::Col(name, _) => Some(name), _ => None }).collect();
    let narrow = IndexSchema { cols: schema.cols.iter().filter(|(name, _)| names.contains(&name)).cloned().collect(), targets: schema.targets.clone(), slots: None, projection: None };
    let columns = source_cols(&narrow, &[]);
    let bound = by.iter().map(|g| bind(ctx, &g.expr, &columns)).collect::<Result<Vec<_>>>()?;
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&schema.targets)?;
    let mut counts = CountGroups { index: HashMap::new(), strs: HashMap::new(), groups: Vec::new(), bytes: 0,
        max_buckets: node.max_buckets() as usize, memory: crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)? };
    let wanted = vec![true; narrow.cols.len()];
    let ord_column = match bound.as_slice() { [Expr::Idx(i)] if *i < narrow.cols.len() => Some(*i), _ => None };
    // (target, leaf) of the cached ordinal → group table, and its charge.
    let mut leaf_groups: Option<((usize, usize), Vec<u32>, crate::memory::Reservation)> = None;
    crate::search::for_each_match(&targets, query, BATCH, &mut |page| {
        let (reader, decoded) = match (ord_column, page.leaf) {
            (Some(column), rustic_core::query::LeafReader::Segment(segment)) if dv_allowed(page.target) =>
                match dv_field(&narrow.cols[column].1, page.target, page.lookup, true).and_then(|field| segment.keyword_reader(field).map(|r| (field, r))) {
                    // A front-coded table's terms decoded once per segment,
                    // not per (leaf, ordinal) twice.
                    Some((field, reader)) => (Some(reader), segment.keyword_terms_decoded(field).ok().flatten()),
                    None => (None, None),
                },
            _ => (None, None),
        };
        // Read once per (leaf, ordinal) seen.
        let term_of = |reader: &rustic_core::codec::fastfields::KeywordReader<'_>, ord: u32| -> Option<String> {
            match decoded.as_ref().and_then(|terms| terms.get(ord as usize)) {
                Some(term) => Some(term.to_string()),
                None => reader.term(ord).map(|t| t.into_owned()),
            }
        };
        let mut ids = page.doc_ids();
        loop {
            let docs: Vec<_> = ids.by_ref().take(BATCH).collect();
            if docs.is_empty() { break; }
            ctx.documents_found += docs.len() as u64;
            let Some(reader) = &reader else {
                let (rows, loaded, _page_memory) = super::fast::sql_column_rows(&narrow, &wanted, page.target, page.lookup, page.leaf, &docs)?;
                ctx.values_loaded += loaded;
                for row in rows { counts.count_row(&bound, &row, &ctx.fctx)?; }
                continue;
            };
            let leaf_id = (page.target_idx, page.leaf_idx);
            if leaf_groups.as_ref().is_none_or(|(id, _, _)| *id != leaf_id) {
                drop(leaf_groups.take());
                let terms = reader.num_terms();
                let memory = crate::memory::breakers().reserve_request((terms as u64).saturating_mul(4).saturating_add(64), crate::esql::exec::ESQL_LABEL)?;
                leaf_groups = Some((leaf_id, vec![ORD_UNSEEN; terms], memory));
            }
            let Some((_, ord_groups, _)) = leaf_groups.as_mut() else { return Err(EsError::internal("ordinal groups missing")) };
            // A doc without a (UTF-8) doc value keeps the column read's
            // `_source` fallback, in its place in doc order.
            let mut missing = Vec::new();
            for &doc in &docs {
                let ord = reader.ord(doc);
                match ord_groups.get_mut(ord as usize) {
                    None => missing.push(doc),
                    Some(slot) => {
                        if *slot == ORD_UNSEEN && term_of(reader, ord).is_none() { *slot = ORD_INVALID; }
                        if *slot == ORD_INVALID { missing.push(doc); }
                    }
                }
            }
            let (fallback, fallback_memory) = if missing.is_empty() { (Vec::new(), None) } else {
                let (rows, loaded, memory) = super::fast::sql_column_rows(&narrow, &wanted, page.target, page.lookup, page.leaf, &missing)?;
                ctx.values_loaded += loaded;
                (rows, Some(memory))
            };
            let mut fallback = fallback.into_iter();
            for &doc in &docs {
                let ord = reader.ord(doc);
                match ord_groups.get_mut(ord as usize).filter(|slot| **slot != ORD_INVALID) {
                    Some(slot) => {
                        if *slot == ORD_UNSEEN {
                            let term = term_of(reader, ord).ok_or_else(|| EsError::internal("keyword ordinal without a term"))?;
                            let id = counts.str_id(&term)?;
                            *slot = u32::try_from(id).ok().filter(|id| *id < ORD_INVALID).ok_or_else(|| EsError::internal("too many SQL count groups"))?;
                        }
                        ctx.values_loaded += 1;
                        counts.groups[*slot as usize].1 += 1;
                    }
                    None => {
                        let row = fallback.next().ok_or_else(|| EsError::internal("SQL count fallback row missing"))?;
                        counts.count_row(&bound, &row, &ctx.fctx)?;
                    }
                }
            }
            drop(fallback_memory);
        }
        Ok(true)
    })?;
    drop(leaf_groups);
    let CountGroups { groups, index, strs, memory: mut state_memory, .. } = counts;
    state_memory.grow((groups.len() as u64).saturating_mul(64u64.saturating_add((aggs.len() as u64).saturating_mul(128))),crate::esql::exec::ESQL_LABEL)?;
    output.rows = groups.into_iter().map(|(key,count)| {
        let mut row=vec![Val::Long(count); aggs.len()];row.extend(key);row
    }).collect();
    drop((index, strs));
    let retained=super::fast::rows_bytes(&output.rows);
    if retained>state_memory.bytes(){state_memory.grow(retained-state_memory.bytes(),crate::esql::exec::ESQL_LABEL)?;}
    else{state_memory.shrink(state_memory.bytes()-retained);}
    state_memory.retain_for_request();
    Ok(Some(output))
}

// ------------------------------------------------------------ SQL folds

#[cfg(test)]
thread_local! {
    /// Tests: leave SQL aggregates and computed sorts to the generic
    /// executor (the reference the folds are compared against).
    pub(crate) static SQL_FOLDS_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: SQL aggregates and sorts the folds answered on this thread.
    pub(crate) static SQL_FOLDS_TAKEN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn sql_folds_enabled() -> bool {
    #[cfg(test)]
    if SQL_FOLDS_DISABLED.with(|d| d.get()) { return false; }
    true
}

fn fold_taken() {
    #[cfg(test)]
    SQL_FOLDS_TAKEN.with(|n| n.set(n.get() + 1));
}

/// Most distinct input values one [`MemoExpr`] remembers.
const MEMO_MAX_ENTRIES: usize = 1 << 16;

/// Scalar functions whose result depends on their arguments alone.
const MEMO_FUNCTIONS: &[&str] = &[
    "LCASE", "UCASE", "LOWER", "UPPER", "TO_LOWER", "TO_UPPER", "SUBSTRING", "SUBSTR", "LENGTH", "CHAR_LENGTH",
    "BIT_LENGTH", "OCTET_LENGTH", "CONCAT", "LEFT", "RIGHT", "LTRIM", "RTRIM", "TRIM", "REPLACE", "ASCII",
    "CEIL", "CEILING", "FLOOR", "ABS", "ROUND", "TRUNCATE", "SIGN", "SQRT", "MOD",
    "TO_LONG", "TO_INTEGER", "TO_DOUBLE", "TO_STRING", "TO_BOOLEAN",
];

/// The one input column of bound expression `e` when `e` is a pure function
/// of it (literals, operators and [`MEMO_FUNCTIONS`]): that column's value
/// then decides the result.
fn memo_column(e: &Expr) -> Option<usize> {
    fn walk(e: &Expr, col: &mut Option<usize>) -> bool {
        match e {
            Expr::Lit(_) => true,
            Expr::Idx(i) => {
                if let Some(c) = *col {
                    c == *i
                } else {
                    *col = Some(*i);
                    true
                }
            }
            Expr::Bin(_, a, b) => walk(a, col) && walk(b, col),
            Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => walk(a, col),
            Expr::In(a, list, _) => walk(a, col) && list.iter().all(|x| walk(x, col)),
            Expr::Between(a, b, c, _) => walk(a, col) && walk(b, col) && walk(c, col),
            Expr::Call(name, args, _) => MEMO_FUNCTIONS.contains(&name.as_str()) && args.iter().all(|x| walk(x, col)),
            _ => false,
        }
    }
    let mut col = None;
    if walk(e, &mut col) { col } else { None }
}

/// A bound row expression whose result is cached per distinct value of its
/// one input column ([`memo_column`]; a keyword's string, a scalar's type
/// and bits), else evaluated per row. Results are exactly `eval`'s: NOAA's
/// `LCASE(SUBSTRING(station.name, 0, 2))` runs once per station name and
/// `CEIL(TMIN) % 7` once per distinct temperature, not once per document.
pub(super) struct MemoExpr {
    expr: Expr,
    col: Option<usize>,
    strs: HashMap<String, Val, ahash::RandomState>,
    scalars: HashMap<(u8, u64), Val, ahash::RandomState>,
    memory: crate::memory::Reservation,
}

impl MemoExpr {
    fn new(expr: Expr) -> Result<Self> {
        // A bare column or literal is cheaper to read than to look up.
        let col = if matches!(expr, Expr::Idx(_) | Expr::Lit(_)) { None } else { memo_column(&expr) };
        Ok(MemoExpr { col, expr, strs: HashMap::default(), scalars: HashMap::default(),
            memory: crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)? })
    }

    fn eval(&mut self, row: &[Val], f: &FnCtx) -> Result<Val> {
        let Some(col) = self.col else { return eval(&self.expr, row, f) };
        let room = self.strs.len() + self.scalars.len() < MEMO_MAX_ENTRIES;
        let key = match row.get(col) {
            Some(Val::Str(s)) => {
                if let Some(v) = self.strs.get(s.as_str()) { return Ok(v.clone()); }
                let v = eval(&self.expr, row, f)?;
                if room {
                    self.memory.grow((s.len() + v.heap_bytes() + 128) as u64, crate::esql::exec::ESQL_LABEL)?;
                    self.strs.insert(s.clone(), v.clone());
                }
                return Ok(v);
            }
            Some(Val::Null) => (0u8, 0u64),
            Some(Val::Bool(b)) => (1, *b as u64),
            Some(Val::Int(i)) => (2, *i as u64),
            Some(Val::Long(i)) => (3, *i as u64),
            Some(Val::Double(d)) => (4, d.to_bits()),
            Some(Val::Date(d)) => (5, *d as u64),
            _ => return eval(&self.expr, row, f),
        };
        if let Some(v) = self.scalars.get(&key) { return Ok(v.clone()); }
        let v = eval(&self.expr, row, f)?;
        if room {
            self.memory.grow((v.heap_bytes() + 96) as u64, crate::esql::exec::ESQL_LABEL)?;
            self.scalars.insert(key, v.clone());
        }
        Ok(v)
    }
}

/// Bind `e` against `cols` behind a [`MemoExpr`].
pub(super) fn memo(ctx: &Ctx, e: &Expr, cols: &[Column]) -> Result<MemoExpr> {
    MemoExpr::new(bind(ctx, e, cols)?)
}

/// Whether every residual filter accepts `row` (`WHERE` keeps a row only
/// when its condition is `true`).
pub(super) fn accepts(filters: &mut [MemoExpr], row: &[Val], f: &FnCtx) -> Result<bool> {
    for filter in filters.iter_mut() {
        if filter.eval(row, f)? != Val::Bool(true) { return Ok(false); }
    }
    Ok(true)
}

/// The names of the columns `e` references.
fn column_refs(e: &Expr, out: &mut Vec<String>) {
    let mut refs = Vec::new();
    crate::esql::analyze::expr_refs(e, &mut refs);
    for (name, _) in refs {
        if !out.contains(&name) { out.push(name); }
    }
}

/// One aggregate of [`grouped_fold`], fed a group's values in row order and
/// reduced exactly as `compute_agg` reduces the same list.
#[derive(Clone)]
enum FoldAcc {
    CountStar(i64),
    Count(i64),
    /// `ints`: every value so far is an integer (then the sum is a checked
    /// long, else the compensated double sum of all of them).
    Sum { any: bool, ints: bool, isum: i64, overflow: bool, k: crate::esql::partial::Kahan },
    /// `int`: the argument is an `integer` (a checked long sum over the
    /// numeric count), else the compensated double sum over it.
    Avg { int: bool, n: i64, isum: i64, bad: bool, k: crate::esql::partial::Kahan },
    Min(Option<Val>),
    Max(Option<Val>),
}

impl FoldAcc {
    fn of(name: &str, star: bool, int: bool) -> Option<FoldAcc> {
        Some(match (name, star) {
            ("COUNT", true) => FoldAcc::CountStar(0),
            ("COUNT", false) => FoldAcc::Count(0),
            ("SUM", false) => FoldAcc::Sum { any: false, ints: true, isum: 0, overflow: false, k: Default::default() },
            ("AVG", false) => FoldAcc::Avg { int, n: 0, isum: 0, bad: false, k: Default::default() },
            ("MIN", false) => FoldAcc::Min(None),
            ("MAX", false) => FoldAcc::Max(None),
            _ => return None,
        })
    }

    /// Fold one row's argument (`None`: `COUNT(*)`): its non-null values.
    fn add(&mut self, value: Option<&Val>) {
        let Some(value) = value else {
            if let FoldAcc::CountStar(n) = self { *n += 1; }
            return;
        };
        for v in value.values() {
            match self {
                FoldAcc::CountStar(n) => *n += 1,
                FoldAcc::Count(n) => *n += 1,
                FoldAcc::Sum { any, ints, isum, overflow, k } => {
                    *any = true;
                    if *ints {
                        match v {
                            Val::Int(i) | Val::Long(i) => if !*overflow {
                                match isum.checked_add(*i) {
                                    Some(x) => *isum = x,
                                    None => *overflow = true,
                                }
                            },
                            _ => *ints = false,
                        }
                    }
                    if let Some(x) = v.as_f64() { k.add(x, 0.0); }
                }
                FoldAcc::Avg { n, isum, bad, k, .. } => {
                    if let Some(x) = v.as_f64() {
                        *n += 1;
                        k.add(x, 0.0);
                    }
                    if !*bad {
                        match v.as_i64().and_then(|i| isum.checked_add(i)) {
                            Some(x) => *isum = x,
                            None => *bad = true,
                        }
                    }
                }
                FoldAcc::Min(best) => if best.as_ref().is_none_or(|b| cmp_vals(v, b) == Ordering::Less) { *best = Some(v.clone()); },
                FoldAcc::Max(best) => if best.as_ref().is_none_or(|b| cmp_vals(v, b) != Ordering::Less) { *best = Some(v.clone()); },
            }
        }
    }

    fn finish(self, f: &FnCtx, pos: usize) -> Val {
        let compensated = |k: crate::esql::partial::Kahan| {
            let mut last = crate::esql::partial::Kahan::default();
            last.add(k.value, k.delta);
            last.value
        };
        match self {
            FoldAcc::CountStar(n) | FoldAcc::Count(n) => Val::Long(n),
            FoldAcc::Sum { any: false, .. } => Val::Null,
            FoldAcc::Sum { ints: true, overflow: true, .. } => {
                sum_overflow_warning(f, pos);
                Val::Null
            }
            FoldAcc::Sum { ints: true, isum, .. } => Val::Long(isum),
            FoldAcc::Sum { k, .. } => Val::Double(compensated(k)),
            FoldAcc::Avg { n: 0, .. } => Val::Null,
            FoldAcc::Avg { int: true, bad: true, .. } => {
                sum_overflow_warning(f, pos);
                Val::Null
            }
            FoldAcc::Avg { int: true, n, isum, .. } => Val::Double(isum as f64 / n as f64),
            FoldAcc::Avg { n, k, .. } => Val::Double(compensated(k) / n as f64),
            FoldAcc::Min(v) | FoldAcc::Max(v) => v.unwrap_or(Val::Null),
        }
    }
}

/// The groups of [`grouped_fold`] in first-seen order, keyed as the generic
/// `STATS` keys them (`Val::key` of each value): one string or integer key
/// is looked up without building that key.
struct FoldGroups {
    strs: HashMap<String, usize, ahash::RandomState>,
    ints: HashMap<i64, usize, ahash::RandomState>,
    index: HashMap<Vec<String>, usize, ahash::RandomState>,
    groups: Vec<(Vec<Val>, Vec<FoldAcc>)>,
    init: Vec<FoldAcc>,
    memory: crate::memory::Reservation,
}

impl FoldGroups {
    fn admit(&mut self, key: Vec<Val>) -> Result<usize> {
        self.memory.grow(crate::esql::exec::row_bytes(&key).saturating_mul(3).saturating_add(128).saturating_add((self.init.len() as u64).saturating_mul(64)), crate::esql::exec::ESQL_LABEL)?;
        self.groups.push((key, self.init.clone()));
        Ok(self.groups.len() - 1)
    }

    /// [`Self::id_of`] of the one-value key `[value]`, cloning `value` only
    /// for a new group.
    fn id_of_value(&mut self, value: &Val) -> Result<usize> {
        match value {
            Val::Str(s) => if let Some(&id) = self.strs.get(s.as_str()) { return Ok(id) },
            Val::Int(i) | Val::Long(i) => if let Some(&id) = self.ints.get(i) { return Ok(id) },
            Val::Double(d) if d.fract() == 0.0 && d.abs() < 9.0e15 => if let Some(&id) = self.ints.get(&(*d as i64)) { return Ok(id) },
            _ => {}
        }
        self.id_of(vec![value.clone()])
    }

    fn id_of(&mut self, key: Vec<Val>) -> Result<usize> {
        if let [value] = key.as_slice() {
            // `Val::key`: `s…` only for strings, `i…` for integers and
            // integral doubles below 9e15.
            if let Val::Str(s) = value {
                if let Some(&id) = self.strs.get(s.as_str()) { return Ok(id); }
                let s = s.clone();
                let id = self.admit(key)?;
                self.strs.insert(s, id);
                return Ok(id);
            }
            let int = match value {
                Val::Int(i) | Val::Long(i) => Some(*i),
                Val::Double(d) if d.fract() == 0.0 && d.abs() < 9.0e15 => Some(*d as i64),
                _ => None,
            };
            if let Some(i) = int {
                if let Some(&id) = self.ints.get(&i) { return Ok(id); }
                let id = self.admit(key)?;
                self.ints.insert(i, id);
                return Ok(id);
            }
        }
        let encoded: Vec<String> = key.iter().map(Val::key).collect();
        if let Some(&id) = self.index.get(&encoded) { return Ok(id); }
        let id = self.admit(key)?;
        self.index.insert(encoded, id);
        Ok(id)
    }
}

/// SQL `GROUP BY` / aggregates folded batch by batch into per-group states:
/// the `STATS` directly after `FROM` (and its residual `WHERE`s) with
/// `COUNT`, `SUM`, `AVG`, `MIN` and `MAX` of row expressions, grouped by row
/// expressions (ES: a composite aggregation with script sources and
/// metrics). Reads only the referenced columns, never retains input rows,
/// and evaluates single-column expressions once per distinct value. Groups
/// come out in first-seen order with the generic `STATS`'s keys, values
/// and types; multi-valued keys expand into every combination as it does.
pub(super) fn grouped_fold(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<Table>> {
    let Some(stats) = rest.first() else { return Ok(None) };
    let Cmd::Stats { aggs, by, inline: false, .. } = stats else { return Ok(None) };
    if !sql_folds_enabled() || schema.slots.is_some() || schema.targets.is_empty() || ctx.ts || ctx.esql || (aggs.is_empty() && by.is_empty()) { return Ok(None); }
    // No `needed_columns` gate: the fold reads only the columns its keys,
    // arguments and filters reference (and that set declines any selected
    // `COUNT(*)`, whose name reads as a KEEP wildcard).
    let mut names: Vec<&String> = aggs.iter().chain(by).map(|f| &f.name).collect();
    names.sort();
    names.dedup();
    if names.len() != aggs.len() + by.len() || aggs.iter().chain(by).any(|f| f.filter.is_some()) { return Ok(None); }
    let mut calls: Vec<(&str, Option<&Expr>, usize)> = Vec::with_capacity(aggs.len());
    for agg in aggs {
        let Expr::Call(name, args, pos) = &agg.expr else { return Ok(None) };
        match (name.as_str(), args.as_slice()) {
            ("COUNT", [Expr::Star]) => calls.push((name.as_str(), None, *pos)),
            ("COUNT" | "SUM" | "AVG" | "MIN" | "MAX", [arg]) if !matches!(arg, Expr::Star) => calls.push((name.as_str(), Some(arg), *pos)),
            _ => return Ok(None),
        }
    }
    // Every column the fold reads; all must be index columns.
    let mut refs: Vec<String> = Vec::new();
    for g in by { column_refs(&g.expr, &mut refs); }
    for (_, arg, _) in &calls { if let Some(arg) = arg { column_refs(arg, &mut refs); } }
    for (e, _) in residual { column_refs(e, &mut refs); }
    if !refs.iter().all(|r| schema.cols.iter().any(|(n, c)| n == r && c.ty != "unsupported" && c.ty != "aggregate_metric_double")) { return Ok(None); }
    // Validation and the output columns: the generic `STATS` over no rows.
    let mut empty = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    for (e, p) in residual { empty = apply(ctx, &Cmd::Where(e.clone(), *p), empty)?; }
    let mut output = apply(ctx, stats, empty)?;
    if output.cols.len() != aggs.len() + by.len() { return Ok(None); }
    let narrow = IndexSchema { cols: schema.cols.iter().filter(|(name, _)| refs.contains(name)).cloned().collect(), targets: schema.targets.clone(), slots: None, projection: None };
    let columns = source_cols(&narrow, &[]);
    let mut filters: Vec<MemoExpr> = residual.iter().map(|(e, _)| memo(ctx, e, &columns)).collect::<Result<_>>()?;
    let mut keys: Vec<MemoExpr> = by.iter().map(|g| memo(ctx, &g.expr, &columns)).collect::<Result<_>>()?;
    let mut args: Vec<Option<MemoExpr>> = Vec::with_capacity(calls.len());
    let mut init: Vec<FoldAcc> = Vec::with_capacity(calls.len());
    for (name, arg, _) in &calls {
        let bound = match arg { Some(arg) => Some(bind(ctx, arg, &columns)?), None => None };
        let int = bound.as_ref().and_then(|b| type_of(b, &columns)).as_deref() == Some("integer");
        let Some(acc) = FoldAcc::of(name, bound.is_none(), int) else { return Ok(None) };
        init.push(acc);
        args.push(match bound { Some(b) => Some(MemoExpr::new(b)?), None => None });
    }
    // As the doc-values pass reads them: float columns that never reach the
    // output (aggregate arguments, filters, computed keys) from doc values
    // alone; output columns keep the projected-value read.
    let mut output_names: Vec<String> = by.iter().filter_map(|g| match &g.expr { Expr::Col(n, _) => Some(n.clone()), _ => None }).collect();
    for cmd in rest {
        match cmd {
            Cmd::Keep(names, _) => output_names.extend(names.iter().cloned()),
            Cmd::Eval(fields) => for field in fields { column_refs(&field.expr, &mut output_names); },
            _ => {}
        }
    }
    let wanted = vec![true; narrow.cols.len()];
    let dv_only: Vec<bool> = narrow.cols.iter().map(|(name, c)| c.ty == "double" && !output_names.contains(name)).collect();
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&schema.targets)?;
    let mut groups = FoldGroups { strs: HashMap::default(), ints: HashMap::default(), index: HashMap::default(), groups: Vec::new(), init,
        memory: crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)? };
    // Without groups the one group exists even when nothing matches.
    if by.is_empty() { groups.id_of(Vec::new())?; }
    let mut values: Vec<Option<Val>> = Vec::with_capacity(args.len());
    if by.is_empty() && residual.is_empty() && args.iter().all(Option::is_none) {
        // Only COUNT(*): the number of matches, without reading them.
        let n = crate::search::par_count(&targets, query)? as i64;
        ctx.documents_found += n as u64;
        for acc in groups.groups[0].1.iter_mut() { *acc = FoldAcc::CountStar(n); }
    } else {
    crate::search::for_each_match(&targets, query, BATCH, &mut |page| {
        let mut ids = page.doc_ids();
        loop {
            let docs: Vec<_> = ids.by_ref().take(BATCH).collect();
            if docs.is_empty() { break; }
            let (rows, loaded, _page_memory) = dv_column_rows(&narrow, &wanted, &dv_only, page.target, page.lookup, page.leaf, &docs)?;
            ctx.documents_found += rows.len() as u64;
            ctx.values_loaded += loaded;
            for row in rows {
                if !accepts(&mut filters, &row, &ctx.fctx)? { continue; }
                // One grouping key: looked up by reference (a plain column is
                // not even cloned), no key vector per row.
                if let [single] = keys.as_mut_slice() {
                    let owned;
                    let value: &Val = match &single.expr {
                        Expr::Idx(i) if single.col.is_none() => &row[*i],
                        _ => { owned = single.eval(&row, &ctx.fctx)?; &owned }
                    };
                    if !matches!(value, Val::Multi(_)) {
                        values.clear();
                        for arg in args.iter_mut() {
                            values.push(match arg { Some(arg) => Some(arg.eval(&row, &ctx.fctx)?), None => None });
                        }
                        let id = groups.id_of_value(value)?;
                        for (acc, v) in groups.groups[id].1.iter_mut().zip(&values) { acc.add(v.as_ref()); }
                        continue;
                    }
                }
                let key: Vec<Val> = keys.iter_mut().map(|k| k.eval(&row, &ctx.fctx)).collect::<Result<_>>()?;
                values.clear();
                for arg in args.iter_mut() {
                    values.push(match arg { Some(arg) => Some(arg.eval(&row, &ctx.fctx)?), None => None });
                }
                let fold = |groups: &mut FoldGroups, key: Vec<Val>| -> Result<()> {
                    let id = groups.id_of(key)?;
                    for (acc, value) in groups.groups[id].1.iter_mut().zip(&values) { acc.add(value.as_ref()); }
                    Ok(())
                };
                if by.is_empty() {
                    // The one group (admitted up front).
                    for (acc, value) in groups.groups[0].1.iter_mut().zip(&values) { acc.add(value.as_ref()); }
                } else if key.iter().all(|v| !matches!(v, Val::Multi(_))) {
                    fold(&mut groups, key)?;
                } else {
                    // Every combination of the grouping values, in order.
                    let mut combos: Vec<Vec<Val>> = vec![Vec::new()];
                    for v in key {
                        let options = match v { Val::Multi(m) => m, v => vec![v] };
                        let mut next = Vec::with_capacity(combos.len() * options.len());
                        for c in &combos {
                            for o in &options {
                                let mut c2 = c.clone();
                                c2.push(o.clone());
                                next.push(c2);
                            }
                        }
                        combos = next;
                    }
                    groups.memory.grow(combos.iter().map(|c| crate::esql::exec::row_bytes(c)).fold(0u64, u64::saturating_add), crate::esql::exec::ESQL_LABEL)?;
                    for c in combos { fold(&mut groups, c)?; }
                }
            }
        }
        Ok(true)
    })?;
    }
    let FoldGroups { groups: states, strs, ints, index, memory: mut state_memory, .. } = groups;
    drop((strs, ints, index));
    let mut rows = Vec::with_capacity(states.len());
    for (key, accs) in states {
        let mut row: Vec<Val> = accs.into_iter().zip(&calls).map(|(acc, (_, _, pos))| acc.finish(&ctx.fctx, *pos)).collect();
        row.extend(key);
        rows.push(row);
    }
    // An aggregate whose type its argument does not tell takes its values'.
    for (k, col) in output.cols.iter_mut().enumerate().take(aggs.len()) {
        if col.ty == "null" {
            let vals: Vec<Val> = rows.iter().map(|r: &Vec<Val>| r[k].clone()).collect();
            col.ty = infer_type(None, &vals);
        }
    }
    output.rows = rows;
    let retained = super::fast::rows_bytes(&output.rows);
    if retained > state_memory.bytes() { state_memory.grow(retained - state_memory.bytes(), crate::esql::exec::ESQL_LABEL)?; }
    else { state_memory.shrink(state_memory.bytes() - retained); }
    state_memory.retain_for_request();
    fold_taken();
    Ok(Some(output))
}

/// Most distinct key values [`stable_single_key_order`] buckets.
const SORT_BUCKETS_MAX: usize = 4096;

/// The order of a stable sort of `values` by the one key `sort_keys[0]`
/// (positions into `values`): equal values keep their input order. A key of
/// few distinct values (`CEIL(TMIN) % 7 = 5`, a 2-letter prefix) is bucketed
/// in one pass and only the distinct values are sorted; values that compare
/// equal across buckets (`1` and `1.0`) are merged back into input order.
fn stable_single_key_order(sort_keys: &[SortKey], values: &[Val]) -> Vec<u32> {
    let cmp = |a: &Val, b: &Val| cmp_sort_keys(sort_keys, std::slice::from_ref(a), std::slice::from_ref(b));
    let mut strs: HashMap<&str, usize, ahash::RandomState> = HashMap::default();
    let mut scalars: HashMap<(u8, u64), usize, ahash::RandomState> = HashMap::default();
    let mut buckets: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut bucketed = true;
    for (i, v) in values.iter().enumerate() {
        let b = match v {
            Val::Str(s) => match strs.get(s.as_str()) {
                Some(&b) => b,
                None => { strs.insert(s.as_str(), buckets.len()); buckets.push((i, Vec::new())); buckets.len() - 1 }
            },
            other => {
                let key = match other {
                    Val::Null => (0u8, 0u64),
                    Val::Bool(b) => (1, *b as u64),
                    Val::Int(i) => (2, *i as u64),
                    Val::Long(i) => (3, *i as u64),
                    Val::Double(d) => (4, d.to_bits()),
                    Val::Date(d) => (5, *d as u64),
                    _ => { bucketed = false; break; }
                };
                match scalars.get(&key) {
                    Some(&b) => b,
                    None => { scalars.insert(key, buckets.len()); buckets.push((i, Vec::new())); buckets.len() - 1 }
                }
            }
        };
        if buckets.len() > SORT_BUCKETS_MAX { bucketed = false; break; }
        buckets[b].1.push(i as u32);
    }
    if !bucketed {
        let mut order: Vec<u32> = (0..values.len() as u32).collect();
        order.sort_by(|&a, &b| cmp(&values[a as usize], &values[b as usize]));
        return order;
    }
    // Distinct values in key order (stable: first-seen order among equals).
    let mut ids: Vec<usize> = (0..buckets.len()).collect();
    ids.sort_by(|&a, &b| cmp(&values[buckets[a].0], &values[buckets[b].0]));
    let mut order = Vec::with_capacity(values.len());
    let mut k = 0;
    while k < ids.len() {
        let mut end = k + 1;
        while end < ids.len() && cmp(&values[buckets[ids[k]].0], &values[buckets[ids[end]].0]) == std::cmp::Ordering::Equal { end += 1; }
        if end == k + 1 {
            order.extend_from_slice(&buckets[ids[k]].1);
        } else {
            let start = order.len();
            for &id in &ids[k..end] { order.extend_from_slice(&buckets[id].1); }
            order[start..].sort_unstable();
        }
        k = end;
    }
    order
}

/// SQL `ORDER BY` row expressions over the matches of `FROM` (and its
/// residual `WHERE`s), then `[LIMIT n] KEEP` of index columns: reads only
/// the referenced columns, evaluates single-column keys once per distinct
/// value, and retains just the sort keys and the projected values of each
/// match, instead of whole rows of every mapped field. Same stable order
/// (input in `_doc` order, as the generic `SORT`), values and columns.
pub(super) fn sorted_fold(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if !sql_folds_enabled() || schema.slots.is_some() || schema.targets.is_empty() || ctx.ts || ctx.esql { return Ok(None); }
    let Some(Cmd::Sort(sort_keys)) = rest.first() else { return Ok(None) };
    let (limit, keep_at) = match rest.get(1) { Some(Cmd::Limit(n)) => (Some(*n), 2), _ => (None, 1) };
    let Some(Cmd::Keep(names, _)) = rest.get(keep_at) else { return Ok(None) };
    // Plain column keys with nothing filtered locally: the index sorts them
    // (a top-n search) when a LIMIT follows.
    if residual.is_empty() && sort_keys.iter().all(|k| matches!(k.expr, Expr::Col(..))) { return Ok(None); }
    if names.iter().any(|n| n.contains('*')) || needed_columns(ctx.full, schema).is_none() { return Ok(None); }
    let covered = keep_at + 1;
    let mut refs: Vec<String> = Vec::new();
    for n in names { if !refs.contains(n) { refs.push(n.clone()); } }
    for k in sort_keys { column_refs(&k.expr, &mut refs); }
    for (e, _) in residual { column_refs(e, &mut refs); }
    if !refs.iter().all(|r| schema.cols.iter().any(|(n, c)| n == r && c.ty != "unsupported")) { return Ok(None); }
    // Validation and the output columns: the same commands over no rows.
    let mut empty = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    for (e, p) in residual { empty = apply(ctx, &Cmd::Where(e.clone(), *p), empty)?; }
    for cmd in &rest[..covered] { empty = apply(ctx, cmd, empty)?; }
    let out_cols = empty.cols;
    let narrow = IndexSchema { cols: schema.cols.iter().filter(|(name, _)| refs.contains(name)).cloned().collect(), targets: schema.targets.clone(), slots: None, projection: None };
    let columns = source_cols(&narrow, &[]);
    let Some(project) = out_cols.iter().map(|c| columns.iter().position(|n| n.name == c.name)).collect::<Option<Vec<usize>>>() else { return Ok(None) };
    let mut filters: Vec<MemoExpr> = residual.iter().map(|(e, _)| memo(ctx, e, &columns)).collect::<Result<_>>()?;
    let mut keys: Vec<(MemoExpr, bool, bool)> = Vec::with_capacity(sort_keys.len());
    for k in sort_keys {
        let bound = bind(ctx, &k.expr, &columns)?;
        let version = type_of(&bound, &columns).as_deref() == Some("version");
        keys.push((MemoExpr::new(bound)?, version, k.desc));
    }
    let wanted = vec![true; narrow.cols.len()];
    let dv_only: Vec<bool> = narrow.cols.iter().map(|(name, c)| c.ty == "double" && !names.contains(name)).collect();
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&schema.targets)?;
    let mut keyed: Vec<(Vec<Val>, Vec<Val>)> = Vec::new();
    // One key: its values (beside `outs`), sorted through their few distinct values.
    let single = keys.len() == 1;
    let mut single_keys: Vec<Val> = Vec::new();
    let mut outs: Vec<Vec<Val>> = Vec::new();
    let mut memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    crate::search::for_each_match(&targets, query, BATCH, &mut |page| {
        let mut ids = page.doc_ids();
        loop {
            let docs: Vec<_> = ids.by_ref().take(BATCH).collect();
            if docs.is_empty() { break; }
            let (rows, loaded, _page_memory) = dv_column_rows(&narrow, &wanted, &dv_only, page.target, page.lookup, page.leaf, &docs)?;
            ctx.documents_found += rows.len() as u64;
            ctx.values_loaded += loaded;
            let mut bytes = 0u64;
            for mut row in rows {
                if !accepts(&mut filters, &row, &ctx.fctx)? { continue; }
                if let [(key, version, desc)] = keys.as_mut_slice() {
                    let value = sort_key_value(key.eval(&row, &ctx.fctx)?, *version, *desc);
                    let out: Vec<Val> = project.iter().map(|&i| std::mem::replace(&mut row[i], Val::Null)).collect();
                    bytes = bytes.saturating_add(crate::esql::exec::row_bytes(std::slice::from_ref(&value))).saturating_add(crate::esql::exec::row_bytes(&out).saturating_mul(2));
                    single_keys.push(value);
                    outs.push(out);
                    continue;
                }
                let mut kv = Vec::with_capacity(keys.len());
                for (key, version, desc) in keys.iter_mut() {
                    kv.push(sort_key_value(key.eval(&row, &ctx.fctx)?, *version, *desc));
                }
                let out: Vec<Val> = project.iter().map(|&i| std::mem::replace(&mut row[i], Val::Null)).collect();
                bytes = bytes.saturating_add(crate::esql::exec::row_bytes(&kv)).saturating_add(crate::esql::exec::row_bytes(&out).saturating_mul(2));
                keyed.push((kv, out));
            }
            memory.grow(bytes, crate::esql::exec::ESQL_LABEL)?;
        }
        Ok(true)
    })?;
    let rows: Vec<Vec<Val>> = if single {
        let mut order = stable_single_key_order(sort_keys, &single_keys);
        if let Some(n) = limit { order.truncate(n); }
        order.into_iter().map(|i| std::mem::take(&mut outs[i as usize])).collect()
    } else {
        keyed.sort_by(|(a, _), (b, _)| cmp_sort_keys(sort_keys, a, b));
        if let Some(n) = limit { keyed.truncate(n); }
        keyed.into_iter().map(|(_, r)| r).collect()
    };
    let out = Table { cols: out_cols, rows };
    let retained = super::fast::rows_bytes(&out.rows);
    if retained > memory.bytes() { memory.grow(retained - memory.bytes(), crate::esql::exec::ESQL_LABEL)?; }
    else { memory.shrink(memory.bytes() - retained); }
    memory.retain_for_request();
    fold_taken();
    Ok(Some((out, covered)))
}

#[cfg(test)]
mod tests {
    use super::OLD_PEER_MAX_OVERRIDE;
    use crate::esql::remote::{FetchSpec, PitPhase, RemoteShard, ShardResult, ShardSource, Slot, Split};
    use crate::index::WriteReq;
    use crate::node::Node;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn pivot_key_expansion_checks_breaker_before_cloning() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        crate::memory::scoped(Some(crate::memory::RequestScope::new("pivot keys",breakers)),|| {
            let keys=vec![vec![crate::esql::Val::Str("key".repeat(1024))];8];
            let options=vec![crate::esql::Val::Str("value".repeat(1024));8];
            assert_eq!(super::expand_keys(&keys,&options,1000).err().unwrap().status,429);
            assert_eq!(breakers.request.reserved(),0);
            let (rows,memory)=super::expand_keys(&[vec![crate::esql::Val::Long(1)]],&[crate::esql::Val::Long(2)],1000).unwrap();
            assert_eq!(rows.len(),1);assert!(breakers.request.reserved()>0);
            drop(rows);drop(memory);assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn pivot_min_replacement_is_admitted_before_mutation() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        crate::memory::scoped(Some(crate::memory::RequestScope::new("pivot minimum",breakers)),|| {
            let mut memory=breakers.reserve_request(0,crate::esql::exec::ESQL_LABEL).unwrap();
            let mut acc=super::Acc::new(super::AggKind::Min);
            let value=crate::esql::Val::Str("x".repeat(64*1024));
            assert_eq!(super::add_acc_guarded(&mut acc,&value,&mut memory).unwrap_err().status,429);
            assert!(matches!(acc,super::Acc::Min(None)));
            assert_eq!(breakers.request.reserved(),0);
        });
    }

    const INDEX: &str = "pivot-pages";

    /// What changes on the shard right after its first page was served.
    #[derive(Clone, Copy)]
    enum Between {
        /// The published memtables are sealed last first, as parallel
        /// background seal jobs may install them: the leaves change order.
        Reorder,
        /// Documents already read are updated (new copies, new seq_nos) and
        /// documents not read yet are deleted, then refreshed.
        Writes,
    }

    /// Every shard read as a remote one, each fetch served here.
    struct Remote {
        node: Arc<Node>,
        between: Between,
        pages: AtomicUsize,
        closed: AtomicUsize,
    }

    impl ShardSource for Remote {
        fn split(&self, targets: Vec<crate::search::Target>) -> crate::error::Result<Split> {
            let targets = crate::search::expand_targets(&targets)?;
            let slots = targets.iter().map(|t| Slot::Remote(RemoteShard { index: t.svc.name.clone(), shard: t.shard.as_ref().unwrap().id, alias_filter: t.alias_filter.clone() })).collect();
            Ok(Split { targets, slots: Some(slots) })
        }

        fn fetch(&self, spec: &FetchSpec, _memory: &mut crate::memory::Reservation) -> crate::error::Result<Vec<ShardResult>> {
            let results = crate::esql::remote::serve(&self.node, spec, None, None)?;
            match spec.pit.as_ref().map(|p| p.phase) {
                Some(PitPhase::Close) => {
                    self.closed.fetch_add(1, Ordering::SeqCst);
                    return Ok(results);
                }
                None => panic!("remote PIVOT pages read through a point in time"),
                Some(_) => {}
            }
            if self.pages.fetch_add(1, Ordering::SeqCst) == 0 {
                let svc = self.node.get_index(INDEX).unwrap();
                match self.between {
                    Between::Reorder => {
                        let copy = svc.shard(0).unwrap();
                        let mut core = copy.shard.core.write();
                        let mut jobs = Vec::new();
                        while let Some(job) = core.begin_seal_part(false, 1).unwrap() {
                            jobs.push(job);
                        }
                        assert!(jobs.len() >= 2, "the published memtables are still unsealed");
                        while let Some(job) = jobs.pop() {
                            core.finish_seal(job.run()).unwrap();
                        }
                    }
                    Between::Writes => {
                        let mut writes: Vec<WriteReq> = (0..1500).map(doc).collect();
                        writes.extend((2900..3000).map(|i| WriteReq::Delete { id: i.to_string(), routing: None, cond: Default::default() }));
                        for result in svc.write(writes, false) { result.unwrap(); }
                        svc.refresh().unwrap();
                    }
                }
            }
            Ok(results)
        }
    }

    fn doc(i: usize) -> WriteReq {
        WriteReq::Index { id: Some(i.to_string()), source: json!({"g":format!("g{}", i % 3),"c":if i % 2 == 0 {"A"} else {"B"},"v":i}).to_string(), routing: None, cond: Default::default() }
    }

    /// One shard of 3000 docs in two published memtables (each over a page).
    fn node() -> (tempfile::TempDir, Arc<Node>) {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index(INDEX, &json!({"settings":{"number_of_shards":1,"number_of_replicas":0},
            "mappings":{"properties":{"g":{"type":"keyword"},"c":{"type":"keyword"},"v":{"type":"integer"}}}})).unwrap();
        let svc = node.get_index(INDEX).unwrap();
        for part in 0..2 {
            for result in svc.write((part * 1500..(part + 1) * 1500).map(doc).collect(), false) { result.unwrap(); }
            svc.refresh().unwrap();
        }
        (tmp, node)
    }

    /// The remote fold of each aggregate equals the local one taken before
    /// the fold started (the point in time of its first page).
    fn check(between: Between) {
        for agg in ["SUM(v)", "COUNT(v)"] {
            let (_tmp, node) = node();
            let query = format!("SELECT * FROM (SELECT g,c,v FROM \"{INDEX}\") PIVOT ({agg} FOR c IN ('A','B')) ORDER BY g");
            let local = crate::sql::execute(&node, &query, None, None).unwrap();
            let source = Arc::new(Remote { node: node.clone(), between, pages: AtomicUsize::new(0), closed: AtomicUsize::new(0) });
            let shared: crate::esql::remote::SharedShardSource = source.clone();
            let remote = crate::sql::execute_with(&node, &query, None, None, None, Some(shared)).unwrap();
            assert!(source.pages.load(Ordering::SeqCst) >= 3, "{agg}: the shard was read in several pages");
            assert_eq!(remote.rows, local.rows, "{agg}");
            assert_eq!(source.closed.load(Ordering::SeqCst), 1, "{agg}: the point in time is closed");
            assert!(node.fold_contexts.lock().is_empty(), "{agg}: nothing stays retained");
            assert!(node.pits.lock().is_empty(), "{agg}: no user-visible point in time");
        }
    }

    /// A peer that does not know points in time (an older node) ignores
    /// `pit`: the fold reads that shard in one fetch instead, never pages
    /// a live view, and sends no close.
    struct OldPeer {
        node: Arc<Node>,
        fetches: AtomicUsize,
    }

    impl ShardSource for OldPeer {
        fn split(&self, targets: Vec<crate::search::Target>) -> crate::error::Result<Split> {
            Remote { node: self.node.clone(), between: Between::Reorder, pages: AtomicUsize::new(1), closed: AtomicUsize::new(0) }.split(targets)
        }

        fn fetch(&self, spec: &FetchSpec, _memory: &mut crate::memory::Reservation) -> crate::error::Result<Vec<ShardResult>> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            assert!(!matches!(spec.pit.as_ref().map(|p| p.phase), Some(PitPhase::Read | PitPhase::Close)), "a read or close sent to a peer that opened nothing");
            let spec = FetchSpec { pit: None, ..spec.clone() };
            crate::esql::remote::serve(&self.node, &spec, None, None)
        }
    }

    #[test]
    fn a_peer_without_points_in_time_is_read_in_one_fetch() {
        let (_tmp, node) = node();
        let query = format!("SELECT * FROM (SELECT g,c,v FROM \"{INDEX}\") PIVOT (SUM(v) FOR c IN ('A','B')) ORDER BY g");
        let local = crate::sql::execute(&node, &query, None, None).unwrap();
        let source = Arc::new(OldPeer { node: node.clone(), fetches: AtomicUsize::new(0) });
        let shared: crate::esql::remote::SharedShardSource = source.clone();
        let remote = crate::sql::execute_with(&node, &query, None, None, None, Some(shared)).unwrap();
        assert_eq!(remote.rows, local.rows);
        assert_eq!(source.fetches.load(Ordering::SeqCst), 2, "the open page, then one full fetch");
    }

    /// M12: the old-peer single-fetch fallback is bounded: past the bound
    /// the query fails instead of building an unbounded response.
    #[test]
    fn the_old_peer_fallback_read_is_bounded() {
        let (_tmp, node) = node();
        let query = format!("SELECT * FROM (SELECT g,c,v FROM \"{INDEX}\") PIVOT (SUM(v) FOR c IN ('A','B')) ORDER BY g");
        let source = Arc::new(OldPeer { node: node.clone(), fetches: AtomicUsize::new(0) });
        let shared: crate::esql::remote::SharedShardSource = source.clone();
        OLD_PEER_MAX_OVERRIDE.with(|o| o.set(Some(1)));
        let result = crate::sql::execute_with(&node, &query, None, None, None, Some(shared));
        OLD_PEER_MAX_OVERRIDE.with(|o| o.set(None));
        let error = result.expect_err("an unbounded fallback read");
        assert_eq!(error.status, 400, "{error}");
        assert!(error.to_string().contains("point-in-time"), "{error}");
    }

    /// Fold points in time are capped per node.
    #[test]
    fn fold_contexts_are_capped() {
        let (_tmp, node) = node();
        let shard = RemoteShard { index: INDEX.into(), shard: 0, alias_filter: None };
        let open = |i: usize| FetchSpec { shards: vec![shard.clone()], query: json!({"match_all": {}}), body: json!({"_source": false}), size: Some(1), probes: vec![], agg: None, group_limit: None,
            pit: Some(crate::esql::remote::FetchPit { id: format!("cap-{i}"), keep_alive_ms: 60_000, phase: PitPhase::Open }), fast: None };
        for i in 0..crate::esql::remote::MAX_FOLD_CONTEXTS {
            crate::esql::remote::serve(&node, &open(i), None, None).unwrap();
        }
        let e = crate::esql::remote::serve(&node, &open(usize::MAX), None, None).unwrap_err();
        assert_eq!(e.status, 429);
        assert_eq!(node.fold_contexts.lock().len(), crate::esql::remote::MAX_FOLD_CONTEXTS);
        // Concurrent opens for the last free context: exactly one gets it.
        node.fold_contexts.lock().remove("cap-0");
        let barrier = std::sync::Barrier::new(8);
        let opened = std::thread::scope(|scope| {
            let (node, barrier, open) = (&node, &barrier, &open);
            let racers: Vec<_> = (0..8).map(|i| scope.spawn(move || {
                barrier.wait();
                crate::esql::remote::serve(node, &open(1_000 + i), None, None).is_ok()
            })).collect();
            racers.into_iter().map(|r| r.join().unwrap()).filter(|ok| *ok).count()
        });
        assert_eq!(opened, 1, "concurrent opens passed the cap");
        assert_eq!(node.fold_contexts.lock().len(), crate::esql::remote::MAX_FOLD_CONTEXTS);
    }

    #[test]
    fn remote_pages_read_every_document_once_while_leaves_reorder() {
        check(Between::Reorder);
    }

    #[test]
    fn remote_pages_ignore_updates_and_deletes_between_pages() {
        check(Between::Writes);
    }
}

/// The Rally `sql` track's shapes (NOAA-like sparse documents): the
/// doc-values pass, the PIVOT and grouped-COUNT folds and the implicit
/// composite ordering, against independent references.
#[cfg(test)]
mod rally_sql_tests {
    use crate::esql::Val;
    use crate::index::WriteReq;
    use crate::node::Node;
    use serde_json::{json, Value};
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    const NAMES: [&str; 5] = ["SAN JOSE", "SALT LAKE", "OTHER", "BETA", "Santa Fe"];

    /// (name, country, elevation, TMIN) of doc `n`: a name is missing every
    /// 13th doc, TMIN in 3 of 5 docs (exact in f32).
    fn doc(n: usize) -> (Option<&'static str>, &'static str, f64, Option<f64>) {
        let name = (n % 13 != 7).then(|| NAMES[n % NAMES.len()]);
        let country = ["AS", "CA", "US"][n % 3];
        let elevation = [3000.0, 2000.0, 1000.0][(n / 3) % 3];
        let tmin = (n % 5 >= 2).then(|| (n % 61) as f64 - 30.5);
        (name, country, elevation, tmin)
    }

    /// One index of `docs` docs over `shards` shards, in `segments` refreshes.
    fn weather(index: &str, docs: usize, shards: usize, segments: usize) -> (tempfile::TempDir, Arc<Node>) {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index(index, &json!({"settings":{"number_of_shards":shards,"number_of_replicas":0},"mappings":{"dynamic":"strict","properties":{
            "TMIN":{"type":"float"},
            "station":{"properties":{"name":{"type":"keyword"},"country_code":{"type":"keyword"},"elevation":{"type":"float"}}}
        }}})).unwrap();
        let svc = node.get_index(index).unwrap();
        let per = docs.div_ceil(segments.max(1));
        for part in 0..segments.max(1) {
            let writes: Vec<WriteReq> = (part * per..((part + 1) * per).min(docs)).map(|n| {
                let (name, country, elevation, tmin) = doc(n);
                let mut station = json!({"country_code": country, "elevation": elevation});
                if let Some(name) = name { station["name"] = json!(name); }
                let mut source = json!({"station": station});
                if let Some(t) = tmin { source["TMIN"] = json!(t); }
                WriteReq::Index { id: Some(n.to_string()), source: source.to_string(), routing: None, cond: Default::default() }
            }).collect();
            for result in svc.write(writes, false) { result.unwrap(); }
            svc.refresh().unwrap();
        }
        (tmp, node)
    }

    fn rows(node: &Node, query: &str) -> Vec<Vec<Val>> {
        crate::sql::execute(node, query, None, None).unwrap_or_else(|e| panic!("{query}: {e}")).rows
    }

    fn json_rows(rows: Vec<Vec<Val>>) -> Vec<Value> {
        rows.into_iter().map(|r| Value::Array(r.iter().map(Val::to_json).collect())).collect()
    }

    /// Every row of the streaming `_source` scan (the reference path).
    fn scanned(node: &Node, query: &str) -> Vec<Value> {
        paged(node, query, false, 97).into_iter().flat_map(|(rows, _)| rows).collect()
    }

    /// The pages of the scan the HTTP handler opens (`direct`: on the shard
    /// snapshot, else on the `_doc`-sorted search), `fetch` rows each, with
    /// the rows left after each page (`0`: no cursor).
    fn paged(node: &Node, query: &str, direct: bool, fetch: usize) -> Vec<(Vec<Value>, usize)> {
        let crate::sql::Statement::Select(q) = crate::sql::parse(query, &Default::default()).unwrap() else { panic!("{query}") };
        let ctx = crate::esql::exec::new_ctx(node, query, None, 1_000_000, 1_000_000);
        crate::esql::exec::SQL_SCAN_DIRECT_DISABLED.with(|d| d.set(!direct));
        let scan = crate::esql::exec::SqlScan::open(&ctx, q.index.as_deref().unwrap(), q.limit, &q.cmds);
        crate::esql::exec::SQL_SCAN_DIRECT_DISABLED.with(|d| d.set(false));
        let mut scan = scan.unwrap().expect("a scan");
        assert_eq!(scan.is_direct(), direct, "{query}");
        let mut out = Vec::new();
        loop {
            let page = scan.page(node, fetch).unwrap();
            let left = scan.len();
            let done = page.is_empty() || left == 0;
            out.push((page, left));
            if done { break; }
        }
        out
    }

    /// The executor's table for `query` (`folds`: with the SQL folds, else
    /// the generic `STATS`/`SORT` over materialized rows), or its error.
    fn executed(node: &Node, query: &str, folds: bool) -> Result<(Vec<(String, String)>, Vec<Vec<Val>>), String> {
        super::SQL_FOLDS_DISABLED.with(|d| d.set(!folds));
        let out = crate::sql::execute(node, query, None, None);
        super::SQL_FOLDS_DISABLED.with(|d| d.set(false));
        out.map(|t| (t.cols.into_iter().map(|c| (c.name, c.ty)).collect(), t.rows)).map_err(|e| e.to_string())
    }

    /// The folds return the generic executor's columns, types, rows and
    /// order (or its error) for each query, and answer every `folded` one.
    fn assert_folds_match(node: &Node, queries: &[&str], folded: &[&str]) {
        for query in queries {
            let taken = super::SQL_FOLDS_TAKEN.with(|n| n.get());
            let actual = executed(node, query, true);
            let took = super::SQL_FOLDS_TAKEN.with(|n| n.get()) > taken;
            let expected = executed(node, query, false);
            assert_eq!(actual, expected, "{query}");
            assert_eq!(took, folded.contains(query), "{query}: fold taken");
        }
    }

    const FILTERS: [&str; 4] = [
        "SELECT station.name FROM w WHERE LCASE(SUBSTRING(station.name, 0, 2)) = 'sa' AND station.elevation > 2800",
        "SELECT station.name FROM w WHERE LCASE(SUBSTRING(station.name, 0, 2)) = 'sa' AND station.elevation > 2800 LIMIT 10",
        "SELECT station.name FROM w WHERE CEIL(TMIN) % 7 = 5 AND station.elevation > 2800",
        "SELECT station.name, TMIN FROM w WHERE CEIL(TMIN) % 7 = 5 AND station.elevation > 2800",
    ];

    #[test]
    fn row_filters_read_from_doc_values_match_the_source_scan_in_order() {
        let (_tmp, node) = weather("w", 3000, 1, 3);
        for query in FILTERS {
            let expected = scanned(&node, query);
            assert!(!expected.is_empty(), "{query}");
            assert_eq!(json_rows(rows(&node, query)), expected, "{query}");
            // One shard: the HTTP entry point keeps the direct scan, which
            // stops reading once a page is full.
            assert!(crate::sql::scan_with(&node, query, None, None, None, None).unwrap().is_some_and(|scan| scan.is_direct()), "{query}");
            let direct: Vec<Value> = paged(&node, query, true, 97).into_iter().flat_map(|(rows, _)| rows).collect();
            assert_eq!(direct, expected, "{query}");
        }
        // No row filter: the scan keeps serving.
        assert!(crate::sql::scan_with(&node, "SELECT station.name FROM w WHERE station.elevation > 2800", None, None, None, None).unwrap().is_some());
    }

    #[test]
    fn row_filters_on_several_shards_keep_the_scan_and_the_same_rows() {
        let (_tmp, node) = weather("w", 3000, 2, 2);
        for query in FILTERS.iter().filter(|q| !q.contains("LIMIT")) {
            let mut expected: Vec<String> = scanned(&node, query).iter().map(Value::to_string).collect();
            let mut actual: Vec<String> = json_rows(rows(&node, query)).iter().map(Value::to_string).collect();
            expected.sort(); actual.sort();
            assert_eq!(actual, expected, "{query}");
            assert!(crate::sql::scan_with(&node, query, None, None, None, None).unwrap().is_some(), "{query}");
        }
    }

    #[test]
    fn script_sort_orders_by_the_doc_value_key_and_keeps_every_row() {
        let (_tmp, node) = weather("w", 3000, 1, 2);
        let out = rows(&node, "SELECT station.name FROM w WHERE station.elevation > 2800 ORDER BY CEIL(TMIN) % 7 = 5");
        let mut expected: Vec<(u8, usize)> = (0..3000).filter(|n| doc(*n).2 > 2800.0).map(|n| {
            // false < true < null (nulls last), ties in doc order.
            let key = match doc(n).3 { None => 2, Some(t) => u8::from(t.ceil() % 7.0 == 5.0) };
            (key, n)
        }).collect();
        expected.sort();
        let names: Vec<Val> = expected.iter().map(|(_, n)| doc(*n).0.map(|s| Val::Str(s.into())).unwrap_or(Val::Null)).collect();
        assert_eq!(out.into_iter().map(|mut r| r.remove(0)).collect::<Vec<_>>(), names);
    }

    #[test]
    fn script_aggregates_match_a_reference_in_group_order() {
        let (_tmp, node) = weather("w", 3000, 1, 2);
        // A group whose docs have no TMIN sums to NULL, as in ES SQL (the
        // names `n % 5 < 2` never carry a TMIN in this data).
        let mut sums: BTreeMap<Option<&str>, Option<f64>> = BTreeMap::new();
        for n in (0..3000).filter(|n| doc(*n).2 > 2800.0) {
            let (name, _, _, tmin) = doc(n);
            let sum = sums.entry(name).or_default();
            if let Some(t) = tmin { *sum = Some(sum.unwrap_or(0.0) + t.ceil() % 7.0); }
        }
        assert!(sums.values().any(Option::is_none) && sums.values().any(Option::is_some));
        let out = rows(&node, "SELECT station.name, SUM(CEIL(TMIN) % 7) FROM w WHERE station.elevation > 2800 GROUP BY 1");
        // Ascending group keys, missing first (BTreeMap: None < Some).
        assert_eq!(out.len(), sums.len());
        for (row, (name, sum)) in out.iter().zip(&sums) {
            assert_eq!(row[0], name.map(|s| Val::Str(s.into())).unwrap_or(Val::Null));
            match sum {
                None => assert_eq!(row[1], Val::Null, "{row:?}"),
                Some(sum) => assert!(row[1].as_f64().is_some_and(|v| (v - sum).abs() < 1e-9), "{row:?} vs {sum}"),
            }
        }
    }

    #[test]
    fn grouped_counts_by_ordinal_match_a_reference_across_segments_and_page_identically() {
        let (_tmp, node) = weather("w", 5000, 1, 4);
        let mut counts: BTreeMap<Option<&str>, i64> = BTreeMap::new();
        for n in (0..5000).filter(|n| doc(*n).2 > 1600.0) { *counts.entry(doc(n).0).or_default() += 1; }
        let expected: Vec<Vec<Val>> = counts.iter().map(|(k, c)| vec![k.map(|s| Val::Str(s.into())).unwrap_or(Val::Null), Val::Long(*c)]).collect();
        let query = "SELECT station.name, COUNT(*) FROM w WHERE station.elevation > 1600 GROUP BY station.name";
        let out = rows(&node, query);
        assert_eq!(out, expected);
        // ORDER BY the count: ties keep ascending key order (stable sort).
        let mut by_count = expected.clone();
        by_count.sort_by_key(|r| r[1].as_i64());
        assert_eq!(rows(&node, "SELECT station.name, COUNT(*) FROM w WHERE station.elevation > 1600 GROUP BY 1 ORDER BY 2"), by_count);
        // Pages of the cursor's rows are the unpaged result, in order.
        let mut pager = crate::search_ext::SqlRows::Table(out.clone().into_iter());
        let mut paged = Vec::new();
        while pager.len() > 0 { paged.extend(pager.page(&node, 3).unwrap()); }
        assert_eq!(paged, json_rows(out));
    }

    #[test]
    fn pivots_match_a_reference_in_group_order() {
        let (_tmp, node) = weather("w", 3000, 1, 2);
        let (mut sum, mut cnt) = (HashMap::<&str, f64>::new(), HashMap::<&str, f64>::new());
        for n in 0..3000 {
            let (_, country, _, tmin) = doc(n);
            if let Some(t) = tmin { *sum.entry(country).or_default() += t; *cnt.entry(country).or_default() += 1.0; }
        }
        let out = rows(&node, "SELECT * FROM (SELECT station.country_code, TMIN FROM w) PIVOT (AVG(TMIN) FOR station.country_code IN ('AS', 'CA'))");
        assert_eq!(out.len(), 1);
        for (cell, country) in out[0].iter().zip(["AS", "CA"]) {
            assert!((cell.as_f64().unwrap() - sum[country] / cnt[country]).abs() < 1e-9, "{country}: {cell:?}");
        }
        let mut counts: BTreeMap<Option<&str>, [i64; 2]> = BTreeMap::new();
        for n in (0..3000).filter(|n| doc(*n).2 > 1600.0) {
            let (name, country, _, _) = doc(n);
            if let Some(i) = ["US", "AS"].iter().position(|c| *c == country) { counts.entry(name).or_default()[i] += 1; }
        }
        let expected: Vec<Vec<Val>> = counts.iter().map(|(name, c)| {
            let cell = |v: i64| if v == 0 { Val::Null } else { Val::Long(v) };
            vec![name.map(|s| Val::Str(s.into())).unwrap_or(Val::Null), cell(c[0]), cell(c[1])]
        }).collect();
        let out = rows(&node, "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))");
        assert_eq!(out, expected);
    }

    #[test]
    fn direct_scans_page_the_rows_of_the_search_scan() {
        let (_tmp, node) = weather("w", 3000, 1, 3);
        // Deleted docs (of every segment) are skipped as the search skips them.
        let svc = node.get_index("w").unwrap();
        let deletes: Vec<WriteReq> = (0..3000).step_by(17).map(|n| WriteReq::Delete { id: n.to_string(), routing: None, cond: Default::default() }).collect();
        for result in svc.write(deletes, false) { result.unwrap(); }
        svc.refresh().unwrap();
        let unfiltered = [
            "SELECT station.name FROM w",
            "SELECT * FROM w",
            "SELECT LCASE(SUBSTRING(station.name, 0, 2)) FROM w",
            "SELECT station.name, TMIN FROM w WHERE station.elevation > 2800",
            "SELECT station.name FROM w LIMIT 1234",
            "SELECT station.name FROM w WHERE station.elevation > 99999",
        ];
        for query in unfiltered {
            for fetch in [1, 97, 1000, 1234, 5000] {
                // Same rows, same pages, and a cursor exactly when rows remain.
                assert_eq!(paged(&node, query, true, fetch), paged(&node, query, false, fetch), "{query} fetch {fetch}");
            }
        }
        for query in FILTERS {
            let expected = scanned(&node, query);
            for fetch in [1, 97, 1000, 5000] {
                let pages = paged(&node, query, true, fetch);
                assert!(pages.iter().all(|(rows, _)| rows.len() <= fetch), "{query}");
                assert_eq!(pages.into_iter().flat_map(|(rows, _)| rows).collect::<Vec<_>>(), expected, "{query} fetch {fetch}");
            }
        }
    }

    #[test]
    fn script_folds_match_the_generic_executor() {
        let (_tmp, node) = weather("w", 4000, 1, 3);
        let folded = [
            "SELECT LCASE(SUBSTRING(station.name, 0, 2)) FROM w WHERE station.elevation > 2800 GROUP BY 1",
            "SELECT CEIL(TMIN) % 7 FROM w WHERE station.elevation > 2800 GROUP BY 1",
            "SELECT station.name, SUM(LENGTH(SUBSTRING(station.name, 0, 5))) S FROM w WHERE station.elevation > 2800 GROUP BY 1",
            "SELECT station.name, SUM(LENGTH(SUBSTRING(station.name, 0, 5))) S FROM w WHERE station.elevation > 2800 GROUP BY 1 ORDER BY S",
            "SELECT station.name, SUM(LENGTH(SUBSTRING(station.name, 0, 5))) S FROM w WHERE station.elevation > 2800 GROUP BY 1 HAVING S > 1000",
            "SELECT station.name, SUM(CEIL(TMIN) % 7) FROM w WHERE station.elevation > 2800 GROUP BY 1",
            "SELECT station.name, SUM(CEIL(TMIN) % 7) FROM w WHERE station.elevation > 2800 GROUP BY 1 ORDER BY 2",
            "SELECT station.name, SUM(CEIL(TMIN) % 7) S FROM w WHERE station.elevation > 2800 GROUP BY 1 HAVING S > 0",
            "SELECT station.country_code, AVG(TMIN), MIN(TMIN), MAX(TMIN), COUNT(TMIN), COUNT(*), SUM(TMIN) FROM w GROUP BY station.country_code",
            "SELECT station.country_code, station.name, COUNT(*), MAX(station.name) FROM w GROUP BY 1, 2 ORDER BY 3 DESC, 1",
            "SELECT CEIL(TMIN) % 7 c, COUNT(*) FROM w WHERE LCASE(SUBSTRING(station.name, 0, 2)) = 'sa' GROUP BY c HAVING COUNT(*) > 3",
            "SELECT COUNT(*) FROM w WHERE station.elevation > 1600",
            "SELECT COUNT(*), SUM(TMIN), AVG(station.elevation) FROM w",
            "SELECT MAX(LENGTH(station.name)), MIN(station.name) FROM w WHERE station.elevation > 1600",
            "SELECT DISTINCT station.country_code FROM w",
            "SELECT station.name, SUM(TMIN) FROM w WHERE station.elevation > 99999 GROUP BY 1",
            "SELECT SUM(TMIN), COUNT(*), MIN(TMIN) FROM w WHERE station.elevation > 99999",
            "SELECT station.name FROM w WHERE station.elevation > 2800 ORDER BY LCASE(SUBSTRING(station.name, 0, 2))",
            "SELECT station.name FROM w WHERE station.elevation > 2800 ORDER BY CEIL(TMIN) % 7 = 5",
            "SELECT station.name, TMIN FROM w WHERE station.elevation > 1600 ORDER BY LCASE(station.name) DESC NULLS FIRST, CEIL(TMIN) LIMIT 100",
            "SELECT station.name FROM w WHERE CEIL(TMIN) % 7 = 5 ORDER BY station.name",
            "SELECT TMIN, station.name FROM w ORDER BY TMIN % 3 NULLS FIRST, station.name DESC LIMIT 7",
        ];
        // Left to the executor: plain sorted columns (a top-n search) and
        // grouped COUNT(*) by plain columns (the ordinal fold).
        let others = [
            "SELECT station.name FROM w ORDER BY station.name LIMIT 5",
            "SELECT station.name, COUNT(*) FROM w WHERE station.elevation > 1600 GROUP BY station.name",
        ];
        let all: Vec<&str> = folded.iter().chain(&others).copied().collect();
        assert_folds_match(&node, &all, &folded);
    }

    #[test]
    fn folds_keep_integer_sums_overflow_and_multi_valued_keys_of_the_generic_executor() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("m", &json!({"settings":{"number_of_shards":1,"number_of_replicas":0},"mappings":{"dynamic":"strict","properties":{
            "k":{"type":"keyword"},"n":{"type":"integer"},"l":{"type":"long"},"d":{"type":"double"},"tags":{"type":"keyword"}
        }}})).unwrap();
        let svc = node.get_index("m").unwrap();
        for part in 0..2 {
            let writes: Vec<WriteReq> = (part * 700..(part + 1) * 700).map(|i: usize| {
                let mut source = json!({"l": (i as i64 % 9) * 1_000_000_000_000_000i64});
                if i % 4 != 3 { source["k"] = json!(["a", "b", "c"][i % 3]); }
                if i % 5 != 0 { source["n"] = json!(i % 7); }
                source["d"] = if i % 3 == 0 { json!([i as f64 * 0.25, i as f64 * -0.5]) } else { json!(i as f64 * 0.1) };
                if i % 2 == 0 { source["tags"] = json!(["x", if i % 4 == 0 { "y" } else { "z" }]); } else if i % 3 == 1 { source["tags"] = json!("y"); }
                WriteReq::Index { id: Some(i.to_string()), source: source.to_string(), routing: None, cond: Default::default() }
            }).collect();
            for result in svc.write(writes, false) { result.unwrap(); }
            svc.refresh().unwrap();
        }
        let queries = [
            "SELECT k, SUM(n), AVG(n), MIN(n), MAX(n), COUNT(n), COUNT(*) FROM m GROUP BY k",
            "SELECT k, SUM(l), AVG(l) FROM m GROUP BY k",
            "SELECT SUM(l * 1000), COUNT(*) FROM m",
            "SELECT tags, COUNT(*), SUM(d) FROM m GROUP BY tags",
            "SELECT tags, k, AVG(d), MAX(d) FROM m GROUP BY tags, k",
            "SELECT n % 3, MAX(d), MIN(d), SUM(n + d) FROM m GROUP BY 1",
            "SELECT k FROM m ORDER BY LENGTH(k) DESC, n NULLS FIRST",
            "SELECT k, n FROM m WHERE n % 2 = 1 ORDER BY d LIMIT 50",
            "SELECT tags FROM m WHERE n > 2 ORDER BY UCASE(k)",
        ];
        for query in queries {
            assert_eq!(executed(&node, query, true), executed(&node, query, false), "{query}");
        }
    }

    #[test]
    fn pivot_ordinal_path_matches_the_row_path() {
        // Several segments, deletes, missing names (the row path's fallback
        // in place) and sparse TMIN.
        let (_tmp, node) = weather("w", 5000, 1, 4);
        let svc = node.get_index("w").unwrap();
        let deletes: Vec<WriteReq> = (0..5000).step_by(23).map(|n| WriteReq::Delete { id: n.to_string(), routing: None, cond: Default::default() }).collect();
        for result in svc.write(deletes, false) { result.unwrap(); }
        svc.refresh().unwrap();
        let queries = [
            "SELECT * FROM (SELECT station.country_code, TMIN FROM w) PIVOT (AVG(TMIN) FOR station.country_code IN ('AS', 'CA'))",
            "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))",
            "SELECT * FROM (SELECT station.name, station.country_code, TMIN FROM w) PIVOT (SUM(TMIN) FOR station.country_code IN ('US', 'CA', 'ZZ'))",
            "SELECT * FROM (SELECT station.name, station.country_code, TMIN FROM w WHERE station.elevation > 1600) PIVOT (MIN(TMIN) FOR station.country_code IN ('CA' AS a, 'CA' AS b, 'AS'))",
            "SELECT * FROM (SELECT station.country_code, TMIN FROM w) PIVOT (MAX(TMIN) FOR station.country_code IN ('US'))",
            "SELECT * FROM (SELECT station.name, station.country_code, TMIN FROM w) PIVOT (COUNT(TMIN) FOR station.country_code IN ('AS', 'US')) ORDER BY station.name DESC",
            // Behind a filter the pivot values are not pushed into the index:
            // every candidate is tested from its ordinal.
            "SELECT * FROM (SELECT station.name, station.country_code, TMIN FROM w WHERE station.elevation > 1600) PIVOT (AVG(TMIN) FOR station.country_code IN ('US', 'ZZ'))",
        ];
        let pivoted = |query: &str, fast: bool| {
            super::PIVOT_FAST_DISABLED.with(|d| d.set(!fast));
            let before = super::PIVOT_FAST_DOCS.with(|n| n.get());
            let out = crate::sql::execute(&node, query, None, None);
            let folded = super::PIVOT_FAST_DOCS.with(|n| n.get()) - before;
            super::PIVOT_FAST_DISABLED.with(|d| d.set(false));
            (out.map(|t| (t.cols.into_iter().map(|c| (c.name, c.ty)).collect::<Vec<_>>(), t.rows)).map_err(|e| e.to_string()), folded)
        };
        // Refreshed writes are memtable leaves (read by term); merged, they
        // are one segment (read by ordinal), as Rally's force-merged index.
        for merged in [false, true] {
            if merged {
                svc.force_merge(1, false).unwrap();
                svc.refresh().unwrap();
            }
            for query in queries {
                let (expected, none) = pivoted(query, false);
                let (actual, folded) = pivoted(query, true);
                assert!(expected.is_ok(), "{query}: {expected:?}");
                assert_eq!(actual, expected, "{query} (merged: {merged})");
                assert_eq!(none, 0, "{query}");
                assert!(folded > 0, "{query} (merged: {merged}): ordinal path not taken");
            }
        }
    }

    #[test]
    fn bucketed_single_key_order_is_the_stable_sort() {
        use crate::esql::parse::{Expr, SortKey};
        let mut values = vec![Val::Long(1), Val::Double(1.0), Val::Str("a".into()), Val::Null, Val::Long(0), Val::Double(1.0), Val::Long(1),
            Val::Bool(true), Val::Str("A".into()), Val::Double(-0.0), Val::Double(0.0), Val::Int(0), Val::Str("a".into()), Val::Null];
        for i in 0..300 { values.push(Val::Long(i % 5)); values.push(Val::Double((i % 3) as f64)); }
        for (desc, nulls_first) in [(false, None), (true, None), (false, Some(true)), (true, Some(false))] {
            let keys = [SortKey { expr: Expr::Col("k".into(), 0), desc, nulls_first }];
            let mut expected: Vec<u32> = (0..values.len() as u32).collect();
            expected.sort_by(|&a, &b| crate::esql::exec::cmp_sort_keys(&keys, std::slice::from_ref(&values[a as usize]), std::slice::from_ref(&values[b as usize])));
            assert_eq!(super::stable_single_key_order(&keys, &values), expected, "{desc} {nulls_first:?}");
            // Past the bucket budget, and with a multi-value: the plain stable sort.
            let many: Vec<Val> = (0..super::SORT_BUCKETS_MAX as i64 + 50).rev().map(Val::Long).chain([Val::Multi(vec![Val::Long(1), Val::Long(2)])]).collect();
            let mut expected: Vec<u32> = (0..many.len() as u32).collect();
            expected.sort_by(|&a, &b| crate::esql::exec::cmp_sort_keys(&keys, std::slice::from_ref(&many[a as usize]), std::slice::from_ref(&many[b as usize])));
            assert_eq!(super::stable_single_key_order(&keys, &many), expected);
        }
    }

    /// Timings of the Rally `sql` operations on a NOAA-like index
    /// (`RALLY_SQL_DOCS`, default 400k docs, one shard, one segment): each
    /// operation as the HTTP handler runs it (the scan when it keeps one,
    /// else the executor; Rally's pages of 1000 rows), with the paths of
    /// this branch disabled (`generic`: no direct scan, SQL folds or PIVOT
    /// ordinal path) and enabled (`current`). For a before/after of two
    /// revisions, run it on both:
    /// `cargo test -p elasticsearch-engine --release --lib rally_sql_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn rally_sql_bench() {
        let docs = std::env::var("RALLY_SQL_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000usize);
        let (_tmp, node) = weather("w", docs, 1, 1);
        let svc = node.get_index("w").unwrap();
        let _ = svc.force_merge(1, false);
        let over = |s: &str| format!("SELECT station.name, SUM({s}) S FROM w WHERE station.elevation > 2800 GROUP BY 1");
        let ops: Vec<(&str, String, usize)> = vec![
            ("select_field", "SELECT station.name FROM w".into(), 1),
            ("select_star", "SELECT * FROM w".into(), 1),
            ("select_scriptField", "SELECT LCASE(SUBSTRING(station.name, 0, 2)) FROM w".into(), 1),
            ("select_3pages", "SELECT station.name FROM w".into(), 3),
            ("select_stringScriptFilter", FILTERS[0].into(), 1),
            ("select_stringScriptFilterWithLimit", FILTERS[1].into(), 1),
            ("select_numericScriptFilter", FILTERS[2].into(), 1),
            ("select_stringScriptSort", "SELECT station.name FROM w WHERE station.elevation > 2800 ORDER BY LCASE(SUBSTRING(station.name, 0, 2))".into(), 1),
            ("select_numericScriptSort", "SELECT station.name FROM w WHERE station.elevation > 2800 ORDER BY CEIL(TMIN) % 7 = 5".into(), 1),
            ("aggregate_byStringScript", "SELECT LCASE(SUBSTRING(station.name, 0, 2)) FROM w WHERE station.elevation > 2800 GROUP BY 1".into(), 1),
            ("aggregate_byNumericScript", "SELECT CEIL(TMIN) % 7 FROM w WHERE station.elevation > 2800 GROUP BY 1".into(), 1),
            ("aggregate_overStringScript", over("LENGTH(SUBSTRING(station.name, 0, 5))"), 1),
            ("aggregate_overStringScript_sortByAggregate", over("LENGTH(SUBSTRING(station.name, 0, 5))") + " ORDER BY S", 1),
            ("aggregate_overStringScript_filterByAggregate", over("LENGTH(SUBSTRING(station.name, 0, 5))") + " HAVING S > 1000", 1),
            ("aggregate_overNumericScript", over("CEIL(TMIN) % 7"), 1),
            ("aggregate_overNumericScript_sortByAggregate", over("CEIL(TMIN) % 7") + " ORDER BY 2", 1),
            ("aggregate_overNumericScript_filterByAggregate", over("CEIL(TMIN) % 7") + " HAVING S > 0", 1),
            ("aggregate_3pages", "SELECT station.name, COUNT(*) FROM w WHERE station.elevation > 1600 GROUP BY station.name".into(), 3),
            ("aggregate_sortByAggregate_3pages", "SELECT station.name, COUNT(*) FROM w WHERE station.elevation > 1600 GROUP BY 1 ORDER BY 2".into(), 3),
            ("pivot", "SELECT * FROM (SELECT station.country_code, TMIN FROM w) PIVOT (AVG(TMIN) FOR station.country_code IN ('AS', 'CA'))".into(), 1),
            ("pivot_3pages", "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))".into(), 3),
            // Its first page alone: pages 2-3 read the materialized pivot
            // (the cursor never re-aggregates), so the two rows should match.
            ("pivot_3pages (first page only)", "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))".into(), 1),
            // The same pivot values behind a selective and an unselective filter.
            ("pivot_3pages (elevation > 2800)", "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 2800) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))".into(), 3),
            ("pivot_3pages (elevation > 0)", "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 0) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))".into(), 3),
        ];
        let set = |generic: bool| {
            crate::esql::exec::SQL_SCAN_DIRECT_DISABLED.with(|d| d.set(generic));
            super::SQL_FOLDS_DISABLED.with(|d| d.set(generic));
            super::PIVOT_FAST_DISABLED.with(|d| d.set(generic));
        };
        let run = |query: &str, pages: usize| -> Vec<Value> {
            match crate::sql::scan_with(&node, query, None, None, None, None).unwrap() {
                Some(mut scan) => (0..pages).flat_map(|_| scan.page(&node, 1000).unwrap()).collect(),
                None => json_rows(rows(&node, query).into_iter().take(pages * 1000).collect()),
            }
        };
        println!("{:<46} {:>12} {:>12}", "operation (median ms)", "generic", "current");
        for (name, query, pages) in &ops {
            let mut medians = Vec::new();
            let mut results = Vec::new();
            for generic in [true, false] {
                set(generic);
                let mut out = Vec::new();
                let mut times: Vec<f64> = (0..7).map(|_| {
                    let start = std::time::Instant::now();
                    out = std::hint::black_box(run(query, *pages));
                    start.elapsed().as_secs_f64() * 1000.0
                }).collect();
                times.sort_by(f64::total_cmp);
                medians.push(times[times.len() / 2]);
                results.push(out);
            }
            set(false);
            assert_eq!(results[0], results[1], "{name}: generic and current rows differ");
            println!("{name:<46} {:>12.2} {:>12.2}", medians[0], medians[1]);
        }
    }
}
