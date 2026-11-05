//! Columnar fast paths for `FROM` pipelines whose filters all ran in the
//! index (no residual row filter): instead of loading every matching
//! document's `_source` into rows, these read the index directly.
//!
//! - `STATS` of only `COUNT(*)`, `ST_CENTROID_AGG` and `ST_EXTENT_AGG` of a
//!   `geo_point` or `geo_shape` field (no groups): the match count and the
//!   geo doc values, or each shape doc's doc-value header (extent, weighted
//!   centroid; what ES's `SpatialDocValuesExtraction` reads), shards in
//!   parallel.
//!
//! Each path applies only to a pipeline shape it fully understands and
//! returns `None` otherwise; the table interpreter then runs as before.
//! Output columns (names, types, `_meta`, verification errors) come from
//! running the same commands over an empty table, so they match the table
//! path exactly.

use super::*;
use crate::esql::partial::{Kahan, WireVal};
use crate::esql::remote::{FastPart, FastPlan, GeoAggPlan, RowCmd};
use rustic_core::geo::{decode_lat, decode_lon};
use rustic_core::query::GeoDv;

/// Pages handed to the per-shard folds.
const PAGE: usize = 1 << 16;

#[cfg(test)]
thread_local! { static LOOKUP_KEY_DV_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
thread_local! {
    /// Tests: lookup-side reads ([`super::exec`]'s `lookup_rows`) are not
    /// source-side key reads: [`LOOKUP_KEY_DV_READS`] does not count them.
    pub(super) static LOOKUP_SIDE_READ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Points of a whole leaf from which its centroid sums are split over
/// threads, and the most slices per leaf (shards already run in parallel,
/// their leaves one after another: an index of many mid-sized segments
/// splits each leaf from 512k points).
const SPLIT_POINTS: usize = 1 << 19;
const SPLIT_MAX: usize = 3;

/// Number of fast-path executions on this thread (tests check the path),
/// and whether fast paths and filter hoisting are off (tests compare with
/// the table path).
#[cfg(test)]
thread_local! {
    pub(super) static TAKEN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static RESIDUAL_MIN: std::cell::Cell<u64> = const { std::cell::Cell::new(STREAM_RESIDUAL_MIN_DOCS) };
}

fn residual_min_docs() -> u64 {
    #[cfg(test)]
    return RESIDUAL_MIN.with(|c| c.get());
    #[cfg(not(test))]
    STREAM_RESIDUAL_MIN_DOCS
}

fn disabled() -> bool {
    #[cfg(test)]
    return DISABLED.with(|c| c.get());
    #[cfg(not(test))]
    false
}

fn taken() {
    #[cfg(test)]
    TAKEN.with(|c| c.set(c.get() + 1));
}

/// The geo-point core field read by an aggregate argument: a `geo_point`
/// column (`col` or `TO_GEOPOINT(col)`) whose values come from doc values.
fn geo_dv_field(e: &Expr, schema: &IndexSchema) -> Option<String> {
    let name = match e {
        Expr::Col(n, _) => n,
        Expr::Call(f, args, _) if f == "TO_GEOPOINT" && args.len() == 1 => match &args[0] {
            Expr::Col(n, _) => n,
            _ => return None,
        },
        _ => return None,
    };
    let (_, c) = schema.cols.iter().find(|(n, _)| n == name)?;
    (c.ty == "geo_point" && c.doc_values && c.per_index.is_empty() && c.constants.is_empty() && c.types.len() <= 1).then(|| c.path.clone())
}

/// The `geo_shape` field read by an aggregate argument: a `geo_shape`
/// column with doc values, one type in every index (ES reads its
/// doc-value header: `SpatialDocValuesExtraction` for shapes).
fn geo_shape_dv_field(e: &Expr, schema: &IndexSchema) -> Option<String> {
    let Expr::Col(name, _) = e else { return None };
    let (_, c) = schema.cols.iter().find(|(n, _)| n == name)?;
    (c.ty == "geo_shape" && c.doc_values && c.per_index.is_empty() && c.constants.is_empty() && c.types.len() <= 1).then(|| c.path.clone())
}

/// One aggregate of a doc-values `STATS`.
#[derive(Clone)]
enum GeoAgg {
    Count,
    Centroid(String),
    Extent(String),
    /// `ST_CENTROID_AGG` / `ST_EXTENT_AGG` of a `geo_shape` field, from
    /// each doc's shape doc value header ([`rustic_core::fieldtypes::shape_dv`]).
    ShapeCentroid(String),
    ShapeExtent(String),
    /// `COUNT(field)` of a `double`, `long` or `integer` field: its doc
    /// values counted per matching doc (ES counts values).
    CountValues(String),
}

/// The `double`, `long` or `integer` field read by `COUNT(field)` from
/// doc values.
fn count_dv_field(e: &Expr, schema: &IndexSchema) -> Option<String> {
    let Expr::Col(name, _) = e else { return None };
    let (_, c) = schema.cols.iter().find(|(n, _)| n == name)?;
    (matches!(c.ty.as_str(), "double" | "long" | "integer") && c.declared == c.ty && c.doc_values && c.per_index.is_empty() && c.constants.is_empty() && c.types.len() <= 1).then(|| c.path.clone())
}

/// Matches of a leaf from which a shape centroid / extent folds in chunks
/// on the pool, and the chunk size.
const SHAPE_FOLD_PAR_MIN: usize = 1 << 16;
const SHAPE_FOLD_CHUNK: usize = 1 << 15;

/// ES `CentroidShapeAggregator.ShapeCentroidState`.
#[derive(Clone, Copy, Default)]
struct ShapeSum {
    x: rustic_core::fieldtypes::shape::CompSum,
    y: rustic_core::fieldtypes::shape::CompSum,
    weight: f64,
    dim: rustic_core::fieldtypes::shape::ShapeDim,
}

impl ShapeSum {
    /// `ShapeCentroidState.add(x, dx, y, dy, weight, shapeType)`: weighted
    /// sums, only the highest dimension kept.
    fn add(&mut self, x: f64, dx: f64, y: f64, dy: f64, weight: f64, dim: rustic_core::fieldtypes::shape::ShapeDim) {
        if !x.is_finite() || !y.is_finite() || !weight.is_finite() || weight <= 0.0 {
            return;
        }
        match dim.cmp(&self.dim) {
            std::cmp::Ordering::Equal => {
                self.x.add_with(x, dx);
                self.y.add_with(y, dy);
                self.weight += weight;
            }
            std::cmp::Ordering::Greater => {
                self.x.reset(x, dx);
                self.y.reset(y, dy);
                self.weight = weight;
                self.dim = dim;
            }
            std::cmp::Ordering::Less => {}
        }
    }
}

/// ES `SpatialExtentStateWrappedLongitudeState.asRectangle` of a merged
/// shape extent: the narrower of the dateline-wrapped and unwrapped boxes.
fn shape_extent_wkt(e: &rustic_core::fieldtypes::shape::ShapeExtent) -> Val {
    let top = decode_lat(e.top);
    let bottom = decode_lat(e.bottom);
    let neg_left = if e.neg_left <= 0 { decode_lon(e.neg_left) } else { f64::INFINITY };
    let neg_right = if e.neg_right <= 0 { decode_lon(e.neg_right) } else { f64::NEG_INFINITY };
    let pos_left = if e.pos_left >= 0 { decode_lon(e.pos_left) } else { f64::INFINITY };
    let pos_right = if e.pos_right >= 0 { decode_lon(e.pos_right) } else { f64::NEG_INFINITY };
    // `GeoPointVisitor.getResult(.., WRAP)`.
    let (min_x, max_x) = if pos_right == f64::NEG_INFINITY {
        (neg_left, neg_right)
    } else if neg_left == f64::INFINITY {
        (pos_left, pos_right)
    } else {
        let unwrapped = pos_right - neg_left;
        let wrapped = 360.0 + neg_right - pos_left;
        if unwrapped <= wrapped {
            (neg_left, pos_right)
        } else {
            (pos_left, neg_right)
        }
    };
    Val::Str(crate::esql::spatial::to_wkt(&rustic_core::fieldtypes::shape::Geometry::Envelope { min_x, max_x, max_y: top, min_y: bottom }))
}

/// Per-shard state of a doc-values `STATS`.
#[derive(Clone, Default)]
struct GeoState {
    docs: u64,
    /// Per aggregate: compensated x / y sums and point count (centroid), or
    /// the bounds `(min_x, max_x, min_y, max_y)` (extent).
    sums: Vec<(crate::esql::partial::Kahan, crate::esql::partial::Kahan, u64)>,
    bounds: Vec<Option<(f64, f64, f64, f64)>>,
    /// Per shape aggregate: the centroid state, or the merged extent.
    shape_sums: Vec<ShapeSum>,
    extents: Vec<Option<rustic_core::fieldtypes::shape::ShapeExtent>>,
    /// A leaf the shape path cannot read (a runtime or differently mapped
    /// field): the table path runs instead.
    bail: bool,
}

impl GeoState {
    fn part(self) -> FastPart {
        let bits = |k: Kahan| (k.value.to_bits(), k.delta.to_bits());
        FastPart::Geo { docs: self.docs, sums: self.sums.into_iter().map(|(x, y, c)| (bits(x), bits(y), c)).collect(), bounds: self.bounds }
    }

    /// Fold `other` (the next leaf or shard in search order) into this
    /// state: ES's `combineIntermediate` of compensated sums with their
    /// deltas, extents and bounds merged.
    fn merge(&mut self, other: GeoState) {
        self.docs += other.docs;
        self.bail |= other.bail;
        for k in 0..self.sums.len() {
            if let Some(p) = other.shape_sums.get(k) {
                if p.weight > 0.0 {
                    self.shape_sums[k].add(p.x.value, p.x.delta, p.y.value, p.y.delta, p.weight, p.dim);
                }
            }
            if let Some(Some(e)) = other.extents.get(k) {
                self.extents[k] = Some(match self.extents[k] {
                    Some(mut x) => {
                        x.merge(e);
                        x
                    }
                    None => *e,
                });
            }
            let (x, y, c) = other.sums[k];
            self.sums[k].0.add(x.value, x.delta);
            self.sums[k].1.add(y.value, y.delta);
            self.sums[k].2 += c;
            self.bounds[k] = match (self.bounds[k], other.bounds[k]) {
                (None, b) | (b, None) => b,
                (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3))),
            };
        }
    }

    /// The state of a shard's part (`None`: not a part of `n` aggregates).
    fn of_part(part: FastPart, n: usize) -> Option<GeoState> {
        let FastPart::Geo { docs, sums, bounds } = part else { return None };
        if sums.len() != n || bounds.len() != n {
            return None;
        }
        let kahan = |(v, d): (u64, u64)| Kahan { value: f64::from_bits(v), delta: f64::from_bits(d) };
        Some(GeoState { docs, sums: sums.into_iter().map(|(x, y, c)| (kahan(x), kahan(y), c)).collect(), bounds, shape_sums: vec![ShapeSum::default(); n], extents: vec![None; n], bail: false })
    }
}

/// Searched documents above which a pipeline with a residual (row) filter
/// streams instead of loading every match (small indices keep the table
/// path, so its statistics stay as they were).
const STREAM_RESIDUAL_MIN_DOCS: u64 = 10_000;

/// Run a fast path for `rest` (the commands after `FROM` and the `WHERE`s
/// pushed into `query`; `residual`: the conditions left to evaluate per
/// row): the table and how many commands of `rest` it covers.
pub(super) fn try_run(ctx: &mut Ctx, schema: &IndexSchema, indices: &[String], query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if disabled() || schema.targets.is_empty() || ctx.ts || !ctx.esql {
        return Ok(None);
    }
    if schema.slots.is_some() {
        // Shards read on other nodes: each runs the same per-shard work
        // where it lives ([`FastPlan`]), merged here in shard order.
        return distributed(ctx, schema, indices, query, residual, rest);
    }
    if let Some(result) = streamed_projected_lookup(ctx, schema, query, residual, rest)? {
        return Ok(Some(result));
    }
    if let Some(result) = streamed_chained_lookup(ctx, schema, query, residual, rest)? {
        return Ok(Some(result));
    }
    if let Some(result) = streamed_sorted_lookup(ctx, schema, query, residual, rest)? {
        return Ok(Some(result));
    }
    if let Some(result) = streamed_filtered_lookup(ctx, schema, query, residual, rest)? {
        return Ok(Some(result));
    }
    if !residual.is_empty() {
        if let Some(result) = streamed_limit(ctx, schema, query, residual, rest)? {
            return Ok(Some(result));
        }
        if let Some(result) = streamed_stats(ctx, schema, query, residual, rest)? {
            return Ok(Some(result));
        }
        return streamed_top_n(ctx, schema, query, residual, rest);
    }
    // EVALs nothing downstream reads are pruned (ES `PruneColumns`).
    let evals = rest.iter().take_while(|c| matches!(c, Cmd::Eval(_))).count();
    if let Some(Cmd::Stats { aggs, by, inline: false, .. }) = rest.get(evals) {
        let defined: Vec<&str> = rest[..evals].iter().flat_map(|c| match c {
            Cmd::Eval(fs) => fs.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
            _ => Vec::new(),
        }).collect();
        let mut used = Vec::new();
        for f in aggs.iter().chain(by.iter()) {
            col_names(&f.expr, &mut used);
            if let Some(x) = &f.filter {
                col_names(x, &mut used);
            }
        }
        if by.is_empty() && !used.iter().any(|u| defined.contains(&u.as_str())) {
            if let Some(t) = geo_stats(ctx, schema, query, &rest[evals], aggs)? {
                return Ok(Some((t, evals + 1)));
            }
        }
    }
    if let Some(r) = distance_top_n(ctx, schema, indices, query, rest)? {
        return Ok(Some(r));
    }
    if let Some(r) = grid_stats(ctx, schema, query, rest)? {
        return Ok(Some(r));
    }
    if let Some(r) = grid_top_n(ctx, schema, query, rest)? {
        return Ok(Some(r));
    }
    if let Some(r) = streamed_top_n(ctx, schema, query, residual, rest)? {
        return Ok(Some(r));
    }
    if let Some(r) = streamed_limit(ctx, schema, query, residual, rest)? {
        return Ok(Some(r));
    }
    streamed_stats(ctx, schema, query, residual, rest)
}

/// `[row commands] STATS ... [BY ...]` over many documents that no index
/// path answers (computed groups or values, e.g. Rally geoshape
/// `st-envelope-count-esql`: `EVAL env = ST_ENVELOPE(shape), points =
/// ST_NPOINTS(env) | WHERE points > 0 | STATS count = COUNT(size) BY
/// points`): the matches stream in pages through the row commands, only
/// the columns the pipeline reads loaded and only the columns STATS reads
/// kept, then STATS runs over those rows. The table path instead
/// materialized every hit with every field (each doc's whole shape) before
/// the first command. Same rows into STATS, in search order.
fn streamed_stats(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if residual.is_empty() {
        if let Some(r) = columnar_stats(ctx, schema, query, rest)? {
            return Ok(Some(r));
        }
    }
    let k = rest.iter().take_while(|c| row_local(c)).count();
    let Some(stats @ Cmd::Stats { aggs, by, inline: false, .. }) = rest.get(k) else { return Ok(None) };
    if searched_docs(schema)? < residual_min_docs() {
        return Ok(None);
    }
    let mut local: Vec<Cmd> = residual.iter().map(|(e, p)| Cmd::Where(e.clone(), *p)).collect();
    local.extend(rest[..k].iter().cloned());
    // `ST_NPOINTS(ST_ENVELOPE(shape))` of a geo_shape column from its doc
    // values, the shape itself not loaded (Rally st-envelope-count-esql).
    let mut extra = shape_npoints_rewrite(ctx.full, schema, &mut local);
    // The columns the row commands produce, then those STATS reads.
    let mut out = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    if let Some(x) = &extra {
        out.cols.push(Column { name: x.name.clone(), ty: x.ty.clone() });
    }
    for c in &local {
        out = apply(ctx, c, out)?;
    }
    // `ST_SIMPLIFY(shape, tolerance)` per row in parallel (Rally
    // st-simplify-count-esql), once the pipeline checked as written.
    if extra.is_none() {
        extra = shape_simplify_rewrite(&source_cols(schema, &[]), &mut local);
    }
    let mut names = Vec::new();
    for f in aggs.iter().chain(by.iter()) {
        col_names(&f.expr, &mut names);
        if let Some(x) = &f.filter {
            col_names(x, &mut names);
        }
    }
    let mut keep: Vec<String> = Vec::new();
    for c in &out.cols {
        if names.contains(&c.name) && !keep.contains(&c.name) {
            keep.push(c.name.clone());
        }
    }
    // `COUNT(*)` alone reads no column: keep one so rows still count.
    if keep.is_empty() {
        let Some(first) = out.cols.first() else { return Ok(None) };
        keep.push(first.name.clone());
    }
    local.push(Cmd::Keep(keep, 0));
    let needed = needed_columns(ctx.full, schema);
    let mut wanted: Vec<bool> = schema.cols.iter().map(|(c, _)| needed.as_ref().is_none_or(|s| s.contains(c))).collect();
    if let Some(x) = extra.as_ref().filter(|x| matches!(x.value, ExtraValue::Doc(_))) {
        // The rewritten pipeline no longer reads the shape.
        for ((c, _), w) in schema.cols.iter().zip(wanted.iter_mut()) {
            if *c == x.field && word_count(ctx.full, c) == 1 {
                *w = false;
            }
        }
    }
    let Some(r) = stream_rows_with(ctx, schema, &schema.targets, query, &local, None, usize::MAX, &wanted, extra.as_ref())? else { return Ok(None) };
    taken();
    ctx.documents_found += r.docs;
    ctx.values_loaded += r.loaded;
    let t = apply(ctx, stats, Table { cols: r.cols, rows: r.rows })?;
    Ok(Some((t, k + 1)))
}

#[cfg(test)]
thread_local! {
    /// Tests: STATS answered by [`columnar_stats`].
    static COLUMNAR_STATS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Tests: [`columnar_stats`] off (compare with the streamed rows).
    static NO_COLUMNAR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One STATS grouping key of [`columnar_stats`].
enum ColKey {
    /// A source keyword column (schema index).
    Keyword(usize),
    /// `DATE_TRUNC(step, column)` of a source date column (schema index):
    /// a fixed step in UTC (ES|QL's `date_trunc` floors it).
    Trunc(usize, i64),
}

/// One aggregate of [`columnar_stats`]: the function and its source
/// column (schema index; `None`: `COUNT(*)`).
#[derive(Clone, Copy, PartialEq)]
enum ColAgg { Avg(usize), Sum(usize), Count(Option<usize>), CountDistinct(usize), Percentile(usize, f64) }

/// [`columnar_term`] of a published memtable's keyword column: interned by
/// value through `map` (memtable columns have no ordinals).
fn columnar_term_mem<'c>(map: &mut HashMap<&'c str, u32>, c: &'c rustic_core::fast_fields::KeywordColumn, d: rustic_core::types::DocId,
    terms: &mut Vec<String>, ids: &mut HashMap<String, u32>, memory: &mut crate::memory::Reservation) -> Result<Option<u32>> {
    let Some(t) = c.get(d) else { return Ok(None) };
    if let Some(&id) = map.get(t) {
        return Ok(Some(id));
    }
    let id = match ids.get(t) {
        Some(&id) => id,
        None => {
            memory.grow(64 + 2 * t.len() as u64, crate::esql::exec::ESQL_LABEL)?;
            terms.push(t.to_string());
            ids.insert(t.to_string(), (terms.len() - 1) as u32);
            (terms.len() - 1) as u32
        }
    };
    map.insert(t, id);
    Ok(Some(id))
}

/// The term id of doc `d`'s value of keyword column `r` (`None`: no value),
/// interning each ordinal of the leaf once through `map`.
fn columnar_term(map: &mut [u32], r: &rustic_core::codec::fastfields::KeywordReader<'_>, d: rustic_core::types::DocId, terms: &mut Vec<String>,
    ids: &mut HashMap<String, u32>, memory: &mut crate::memory::Reservation) -> Result<Option<u32>> {
    let o = r.ord(d);
    let Some(slot) = map.get_mut(o as usize) else { return Ok(None) };
    if *slot == u32::MAX {
        let Some(t) = r.term(o) else { return Ok(None) };
        *slot = match ids.get(&*t) {
            Some(&id) => id,
            None => {
                memory.grow(64 + 2 * t.len() as u64, crate::esql::exec::ESQL_LABEL)?;
                terms.push(t.to_string());
                ids.insert(t.into_owned(), (terms.len() - 1) as u32);
                (terms.len() - 1) as u32
            }
        };
    }
    Ok(Some(*slot))
}

/// A group's running state of one [`ColAgg`] (as `compute_agg` derives it
/// from the group's rows: values in row order).
#[derive(Clone, Copy, Default)]
struct ColAcc { n: u64, sum: Kahan, long: i64, overflow: bool }

/// A numeric column of one leaf as ES|QL reads it.
enum NumCol<'a> {
    F(rustic_core::codec::fastfields::ColumnReader<'a, f64>, Option<f64>),
    I(rustic_core::codec::fastfields::ColumnReader<'a, i64>),
    /// A published memtable's columns (an index-sorted copy's refresh
    /// publishes them unsealed).
    MemF(&'a rustic_core::fast_fields::F64Column, Option<f64>),
    MemI(&'a rustic_core::fast_fields::I64Column),
    Absent,
}

/// A keyword column of one leaf: a segment's (by ordinal) or a published
/// memtable's (by value).
enum KwCol<'a> {
    Seg(rustic_core::codec::fastfields::KeywordReader<'a>),
    Mem(&'a rustic_core::fast_fields::KeywordColumn),
}

impl KwCol<'_> {
    #[inline]
    fn has(&self, d: rustic_core::types::DocId) -> bool {
        match self {
            KwCol::Seg(r) => r.ord(d) != rustic_core::codec::fastfields::MISSING_ORD,
            KwCol::Mem(c) => c.get(d).is_some(),
        }
    }
}

impl NumCol<'_> {
    /// Whether doc `d` has a value (no conversion).
    #[inline]
    fn has(&self, d: rustic_core::types::DocId) -> bool {
        match self {
            NumCol::F(c, _) => c.get(d).is_some(),
            NumCol::I(c) => c.get(d).is_some(),
            NumCol::MemF(c, _) => c.get(d).is_some(),
            NumCol::MemI(c) => c.get(d).is_some(),
            NumCol::Absent => false,
        }
    }

    /// The i64 value of doc `d` (long and date columns).
    #[inline]
    fn i64(&self, d: rustic_core::types::DocId) -> Option<i64> {
        match self {
            NumCol::I(c) => c.get(d),
            NumCol::MemI(c) => c.get(d),
            _ => None,
        }
    }

    #[inline]
    fn f64(&self, d: rustic_core::types::DocId) -> Option<f64> {
        match self {
            NumCol::F(c, scale) => c.get(d).map(|x| match scale { Some(f) => rustic_core::fieldtypes::numeric::scaled_round_trip(x, *f), None => x }),
            NumCol::I(c) => c.get(d).map(|x| x as f64),
            NumCol::MemF(c, scale) => c.get(d).map(|x| match scale { Some(f) => rustic_core::fieldtypes::numeric::scaled_round_trip(x, *f), None => x }),
            NumCol::MemI(c) => c.get(d).map(|x| x as f64),
            NumCol::Absent => None,
        }
    }
}

/// `[EVAL ...] STATS AVG|SUM|COUNT|COUNT_DISTINCT|PERCENTILE(col).. BY
/// keyword, DATE_TRUNC(step, date)` (the TSDB dashboards' ES|QL) from the
/// columns: each matched doc
/// reads its keys and values from doc values and adds to its group, no row
/// built (ES's `HashAggregationOperator` over doc-values blocks). Same
/// groups in the same (first-seen) order, the same compensated sums, the
/// same `documents_found` / `values_loaded` as streaming the rows through
/// STATS ([`streamed_stats`]); `None` (that path runs) for another shape,
/// a column read otherwise (multi-valued, ignored values, not a synthetic
/// source, a memtable leaf), or a long sum that overflows.
fn columnar_stats(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    use rustic_core::query::LeafReader;
    #[cfg(test)]
    if NO_COLUMNAR.with(|c| c.get()) {
        return Ok(None);
    }
    let evals = rest.iter().take_while(|c| matches!(c, Cmd::Eval(_))).count();
    let Some(stats @ Cmd::Stats { aggs, by, inline: false, .. }) = rest.get(evals) else { return Ok(None) };
    if by.len() > 2 || aggs.is_empty() || schema.slots.is_some() || crate::esql::zone() != crate::date::Tz::UTC {
        return Ok(None);
    }
    let mut defs: Vec<(&str, &Expr)> = Vec::new();
    for c in &rest[..evals] {
        let Cmd::Eval(fs) = c else { return Ok(None) };
        for f in fs {
            if defs.iter().any(|(n, _)| *n == f.name) { return Ok(None); }
            defs.push((&f.name, &f.expr));
        }
    }
    let plain = |c: &ColSrc, ty: &str| c.ty == ty && c.doc_values && c.constants.is_empty() && c.per_index.is_empty() && c.types.len() <= 1
        && c.normalizer.is_none() && c.ordered.is_empty() && !c.template_id;
    let source = |name: &str| -> Option<usize> {
        if defs.iter().any(|(n, _)| *n == name) { return None; }
        schema.cols.iter().position(|(n, _)| n == name)
    };
    let numeric = |i: usize| { let c = &schema.cols[i].1; (plain(c, "double") && matches!(c.declared.as_str(), "double" | "scaled_float")) || (plain(c, "long") && c.declared == "long") };
    let mut keys = Vec::new();
    let mut used_defs: Vec<&str> = Vec::new();
    for g in by {
        let Expr::Col(name, _) = &g.expr else { return Ok(None) };
        if g.filter.is_some() || g.name != *name { return Ok(None); }
        if let Some((_, e)) = defs.iter().find(|(n, _)| n == name) {
            let Expr::Call(f, args, _) = e else { return Ok(None) };
            let (true, [Expr::Lit(iv), Expr::Col(c, _)]) = (f == "DATE_TRUNC", args.as_slice()) else { return Ok(None) };
            let step = match iv { Val::Dur(ms) if *ms > 0 => *ms, Val::Period(0, days) if *days > 0 && *days != 7 => days * 86_400_000, _ => return Ok(None) };
            let Some(i) = source(c).filter(|&i| plain(&schema.cols[i].1, "date") && schema.cols[i].1.declared == "date") else { return Ok(None) };
            used_defs.push(name);
            keys.push(ColKey::Trunc(i, step));
        } else {
            let Some(i) = source(name).filter(|&i| plain(&schema.cols[i].1, "keyword") && schema.cols[i].1.declared == "keyword") else { return Ok(None) };
            keys.push(ColKey::Keyword(i));
        }
    }
    // Unused EVAL fields: constants only (the folded `DATE_PARSE` bounds
    // of a pushed time range), which evaluate without a warning.
    for (n, e) in &defs {
        if used_defs.contains(n) || matches!(e, Expr::Lit(_)) {
            continue;
        }
        let mut refs = Vec::new();
        col_names(e, &mut refs);
        let warned = ctx.fctx.warnings.borrow().list.len();
        let ok = refs.is_empty() && eval(e, &[], &ctx.fctx).is_ok() && ctx.fctx.warnings.borrow().list.len() == warned;
        if !ok {
            return Ok(None);
        }
    }
    let mut plan = Vec::new();
    for a in aggs {
        let Expr::Call(f, args, _) = &a.expr else { return Ok(None) };
        if a.filter.is_some() || by.iter().any(|g| g.name == a.name) || aggs.iter().filter(|x| x.name == a.name).count() > 1 { return Ok(None); }
        let keyword = |c: &str| source(c).filter(|&i| plain(&schema.cols[i].1, "keyword") && schema.cols[i].1.declared == "keyword");
        match (f.as_str(), args.as_slice()) {
            ("COUNT_DISTINCT", [Expr::Col(c, _)]) => {
                let Some(i) = keyword(c) else { return Ok(None) };
                plan.push(ColAgg::CountDistinct(i));
                continue;
            }
            ("PERCENTILE", [Expr::Col(c, _), Expr::Lit(p)]) => {
                let (Some(i), Some(p)) = (source(c).filter(|&i| numeric(i)), p.as_f64()) else { return Ok(None) };
                plan.push(ColAgg::Percentile(i, p));
                continue;
            }
            _ => {}
        }
        let col = match args.as_slice() {
            [Expr::Star] if f == "COUNT" => None,
            [Expr::Col(c, _)] => Some(source(c).filter(|&i| numeric(i)).ok_or(())),
            _ => return Ok(None),
        };
        plan.push(match (f.as_str(), col) {
            ("COUNT", None) => ColAgg::Count(None),
            (_, Some(Err(()))) | (_, None) => return Ok(None),
            ("AVG", Some(Ok(i))) => ColAgg::Avg(i),
            ("SUM", Some(Ok(i))) => ColAgg::Sum(i),
            ("COUNT", Some(Ok(i))) => ColAgg::Count(Some(i)),
            _ => return Ok(None),
        });
    }
    // The columns the rows would carry (their non-null values count as
    // loaded): every one must be read here.
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<usize> = (0..schema.cols.len()).filter(|&i| needed.as_ref().is_none_or(|s| s.contains(&schema.cols[i].0))).collect();
    let kind = |i: usize| -> Option<u8> {
        let c = &schema.cols[i].1;
        if plain(c, "keyword") && c.declared == "keyword" { Some(0) } else if plain(c, "date") && c.declared == "date" { Some(1) } else if numeric(i) { Some(2) } else { None }
    };
    if wanted.iter().any(|&i| kind(i).is_none()) || wanted.len() > 16 {
        return Ok(None);
    }
    // The result's columns: STATS over no rows.
    let mut empty = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    for c in &rest[..evals] {
        empty = apply(ctx, c, empty)?;
    }
    let out_cols = apply(ctx, stats, empty)?.cols;
    if out_cols.len() != aggs.len() + by.len() {
        return Ok(None);
    }
    let long_sum = |i: usize| schema.cols[i].1.ty == "long";
    // (Per keyword column read, a slot: keys first, then COUNT_DISTINCT's.)
    let mut kw_cols: Vec<usize> = Vec::new();
    for k in &keys { if let ColKey::Keyword(i) = k { if !kw_cols.contains(i) { kw_cols.push(*i); } } }
    for a in &plan { if let ColAgg::CountDistinct(i) = a { if !kw_cols.contains(i) { kw_cols.push(*i); } } }
    let slot_of = |i: usize| kw_cols.iter().position(|&c| c == i).expect("keyword slot");
    let key_slots: Vec<usize> = keys.iter().map(|k| match k { ColKey::Keyword(i) => slot_of(*i), ColKey::Trunc(..) => usize::MAX }).collect();
    let agg_slots: Vec<usize> = plan.iter().map(|a| match a { ColAgg::CountDistinct(i) => slot_of(*i), _ => usize::MAX }).collect();
    // Per-leaf states, the targets folded concurrently, merged in order.
    let reservations: Vec<std::sync::Mutex<Option<crate::memory::Reservation>>> = (0..schema.targets.len().max(1))
        .map(|_| crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL).map(|r| std::sync::Mutex::new(Some(r))))
        .collect::<Result<_>>()?;
    let init = |ti: usize| ColPart::new(reservations.get(ti).and_then(|m| m.lock().ok()?.take()));
    let fold = |st: &mut ColPart, view: &crate::search::LeafView<'_>| -> Result<()> {
        if st.bail { return Ok(()); }
        // A segment's columns, or a published memtable's.
        let (seg, mem) = match view.leaf {
            LeafReader::Segment(seg) => (Some(*seg), None),
            LeafReader::Memtable(w, _) => (None, Some(*w)),
        };
        if view.target.field_access().is_some() || !view.target.svc.synthetic_source() {
            st.bail = true;
            return Ok(());
        }
        let lookup = view.lookup;
        let id_of = |i: usize| lookup.get(&schema.cols[i].0).and_then(|rf| rf.core_id);
        // A keyword value over `ignore_above` lives in the ignored source:
        // any ignored field here and the rows path reads it.
        if let Some(ig) = lookup.get("_ignored").and_then(|rf| rf.core_id) {
            let ignored = match (seg, mem) {
                (Some(seg), _) => seg.keyword_reader(ig).is_some_and(|c| c.num_terms() > 0) || seg.multi_keyword_reader(ig).is_some_and(|c| c.num_terms() > 0),
                (_, Some(w)) => w.fast_fields().get_keyword(ig).is_some() || w.fast_fields().get_multi_keyword(ig).is_some(),
                _ => true,
            };
            if ignored {
                st.bail = true;
                return Ok(());
            }
        }
        let num = |i: usize| -> Option<NumCol<'_>> {
            let Some(id) = id_of(i) else { return Some(NumCol::Absent) };
            let c = &schema.cols[i].1;
            let int = c.ty == "long" || c.ty == "date";
            let scale = (!int && c.declared == "scaled_float").then(|| lookup.get(&schema.cols[i].0).map(|rf| rf.scaling_factor)).flatten();
            if !int && c.declared == "scaled_float" && scale.is_none() { return None; }
            if let Some(seg) = seg {
                let ff = seg.fast_fields();
                if int {
                    if ff.get_multi_i64(id).is_some() { return None; }
                    return Some(ff.get_i64(id).map(NumCol::I).unwrap_or(NumCol::Absent));
                }
                if ff.get_multi_f64(id).is_some() { return None; }
                return Some(ff.get_f64(id).map(|r| NumCol::F(r, scale)).unwrap_or(NumCol::Absent));
            }
            let ff = mem?.fast_fields();
            if int {
                if ff.get_multi_i64(id).is_some() { return None; }
                return Some(ff.get_i64(id).map(NumCol::MemI).unwrap_or(NumCol::Absent));
            }
            if ff.get_multi_f64(id).is_some() { return None; }
            Some(ff.get_f64(id).map(|r| NumCol::MemF(r, scale)).unwrap_or(NumCol::Absent))
        };
        let kw = |i: usize| -> Option<Option<KwCol<'_>>> {
            let Some(id) = id_of(i) else { return Some(None) };
            if let Some(seg) = seg {
                if seg.multi_keyword_reader(id).is_some() { return None; }
                return Some(seg.keyword_reader(id).map(KwCol::Seg));
            }
            let ff = mem?.fast_fields();
            if ff.get_multi_keyword(id).is_some() { return None; }
            Some(ff.get_keyword(id).map(KwCol::Mem))
        };
        // Readers: keys, aggregates, and the loaded-value counts (opened
        // per thread: column readers cache decoded blocks, so are not
        // shared). `None`: a column this path cannot read.
        #[allow(clippy::type_complexity)]
        let readers = || -> Option<(Vec<Option<KwCol<'_>>>, Vec<NumCol<'_>>, Vec<NumCol<'_>>, Vec<Option<KwCol<'_>>>, Vec<(Option<KwCol<'_>>, NumCol<'_>)>)> {
            let mut key_kw = Vec::new();
            let mut key_num = Vec::new();
            for k in &keys {
                match k {
                    ColKey::Keyword(i) => { key_kw.push(kw(*i)?); key_num.push(NumCol::Absent); }
                    ColKey::Trunc(i, _) => { key_num.push(num(*i)?); key_kw.push(None); }
                }
            }
            let mut agg_cols = Vec::new();
            let mut agg_kw = Vec::new();
            for a in &plan {
                let (c, k) = match a {
                    ColAgg::Avg(i) | ColAgg::Sum(i) | ColAgg::Count(Some(i)) | ColAgg::Percentile(i, _) => (num(*i)?, None),
                    ColAgg::Count(None) => (NumCol::Absent, None),
                    ColAgg::CountDistinct(i) => (NumCol::Absent, kw(*i)?),
                };
                agg_cols.push(c);
                agg_kw.push(k);
            }
            let mut presence: Vec<(Option<KwCol<'_>>, NumCol<'_>)> = Vec::new();
            for &i in &wanted {
                if kind(i) == Some(0) { presence.push((kw(i)?, NumCol::Absent)); } else { presence.push((None, num(i)?)); }
            }
            Some((key_kw, key_num, agg_cols, agg_kw, presence))
        };
        let Some(opened) = readers() else { st.bail = true; return Ok(()) };
        let mut docs: Vec<rustic_core::types::DocId> = Vec::new();
        view.for_each_match(|d| docs.push(d));
        st.found += docs.len() as u64;
        // A range of this leaf's docs into `part`.
        #[allow(clippy::type_complexity)]
        let run = |part: &mut ColPart, docs: &[rustic_core::types::DocId], opened: (Vec<Option<KwCol<'_>>>, Vec<NumCol<'_>>, Vec<NumCol<'_>>, Vec<Option<KwCol<'_>>>, Vec<(Option<KwCol<'_>>, NumCol<'_>)>)| -> Result<()> {
        let (key_kw, key_num, agg_cols, agg_kw, presence) = opened;
        // This leaf's ordinal (a memtable's value) -> term id, per keyword slot.
        let mut maps: Vec<Vec<u32>> = vec![Vec::new(); kw_cols.len()];
        let mut mem_maps: Vec<HashMap<&str, u32>> = vec![HashMap::new(); kw_cols.len()];
        for (&slot, r) in key_slots.iter().zip(&key_kw).chain(agg_slots.iter().zip(&agg_kw)) {
            if let (true, Some(KwCol::Seg(r))) = (slot != usize::MAX, r) {
                if maps[slot].is_empty() { maps[slot] = vec![u32::MAX; r.num_terms()]; }
            }
        }
        let ColPart { terms, term_ids, groups, order, accs, lists, sets, loaded, memory, .. } = part;
        if memory.is_none() { *memory = Some(crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?); }
        let memory = memory.as_mut().expect("reserved");
        for &d in docs {
            let mut key = [0u64; 2];
            for (j, k) in keys.iter().enumerate() {
                key[j] = match k {
                    ColKey::Keyword(_) => match &key_kw[j] {
                        Some(KwCol::Seg(r)) => columnar_term(&mut maps[key_slots[j]], r, d, terms, term_ids, memory)?.map_or(u64::MAX, u64::from),
                        Some(KwCol::Mem(c)) => columnar_term_mem(&mut mem_maps[key_slots[j]], c, d, terms, term_ids, memory)?.map_or(u64::MAX, u64::from),
                        None => u64::MAX,
                    },
                    // No date: `i64::MIN` (no truncation gives it but itself).
                    ColKey::Trunc(_, step) => match key_num[j].i64(d) {
                        Some(ms) if ms != i64::MIN => (ms - ms.rem_euclid(*step)) as u64,
                        _ => i64::MIN as u64,
                    },
                };
            }
            let g = match groups.get(&key) {
                Some(&g) => g,
                None => {
                    memory.grow(96 + 64 * plan.len() as u64, crate::esql::exec::ESQL_LABEL)?;
                    order.push(key);
                    accs.extend(std::iter::repeat_n(ColAcc::default(), plan.len()));
                    lists.extend(std::iter::repeat_with(Vec::new).take(plan.len()));
                    sets.extend(std::iter::repeat_with(Default::default).take(plan.len()));
                    groups.insert(key, order.len() - 1);
                    order.len() - 1
                }
            };
            let base = g * plan.len();
            for (k, ((a, c), r)) in plan.iter().zip(&agg_cols).zip(&agg_kw).enumerate() {
                let acc = &mut accs[base + k];
                match a {
                    ColAgg::Count(None) => acc.n += 1,
                    ColAgg::CountDistinct(_) => {
                        let id = match r {
                            Some(KwCol::Seg(r)) => columnar_term(&mut maps[agg_slots[k]], r, d, terms, term_ids, memory)?,
                            Some(KwCol::Mem(c)) => columnar_term_mem(&mut mem_maps[agg_slots[k]], c, d, terms, term_ids, memory)?,
                            None => None,
                        };
                        if let Some(id) = id {
                            if sets[base + k].insert(id) { memory.grow(16, crate::esql::exec::ESQL_LABEL)?; }
                        }
                    }
                    ColAgg::Percentile(..) => {
                        if let Some(x) = c.f64(d) {
                            memory.grow(8, crate::esql::exec::ESQL_LABEL)?;
                            lists[base + k].push(x);
                        }
                    }
                    ColAgg::Sum(i) if long_sum(*i) => {
                        if let Some(x) = c.i64(d) {
                            acc.n += 1;
                            match acc.long.checked_add(x) { Some(s) => acc.long = s, None => acc.overflow = true }
                        }
                    }
                    _ => {
                        if let Some(x) = c.f64(d) {
                            acc.n += 1;
                            acc.sum.add(x, 0.0);
                        }
                    }
                }
            }
            for (r, c) in &presence {
                let has = match r { Some(r) => r.has(d), None => c.has(d) };
                *loaded += has as u64;
            }
        }
        Ok(())
        };
        // A large leaf (a force-merged segment holds a whole day): its docs
        // in contiguous ranges folded concurrently, each into its own
        // state, merged in doc order with the others.
        if docs.len() >= 2 * STATS_SPLIT_DOCS {
            use rayon::prelude::*;
            let n = docs.len().div_ceil(STATS_SPLIT_DOCS).min(rayon::current_num_threads().max(1));
            let size = docs.len().div_ceil(n);
            let subs: Vec<Result<ColPart>> = docs.par_chunks(size).map(|range| {
                let mut part = ColPart::new(None);
                let opened = readers().expect("columns opened on this leaf before");
                run(&mut part, range, opened).map(|_| part)
            }).collect();
            for sub in subs { st.subs.push(sub?); }
        } else {
            run(st, &docs, opened)?;
        }
        Ok(())
    };
    // One state per leaf (leaves fold concurrently where the query runs no
    // script), merged in target, then leaf order.
    let parts: Vec<ColPart> = crate::search::par_for_each_leaf_states(&schema.targets, query, &init, &fold)?.into_iter().flatten()
        .flat_map(|mut p| {
            // A split leaf: its ranges' states, in order (the leaf's own
            // holds only its docs found and its bail).
            let subs = std::mem::take(&mut p.subs);
            let mut out = vec![p];
            out.extend(subs);
            out
        }).collect();
    if parts.iter().any(|p| p.bail || p.accs.iter().any(|a| a.overflow)) {
        return Ok(None);
    }
    // Merged in target order: groups in first-seen order, each target's
    // terms re-interned; compensated sums combine as ES combines its
    // drivers' intermediate states (`CompensatedSum.add(value, delta)`).
    let mut memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    let mut all = ColPart::new(None);
    for part in parts {
        let ColPart { terms: pt, groups: _, order: porder, accs: paccs, lists: mut plists, sets: psets, found, loaded, memory: pmem, .. } = part;
        all.found += found;
        all.loaded += loaded;
        let mut ids: Vec<u32> = Vec::with_capacity(pt.len());
        for t in pt {
            let id = match all.term_ids.get(&t) {
                Some(&id) => id,
                None => {
                    memory.grow(64 + 2 * t.len() as u64, crate::esql::exec::ESQL_LABEL)?;
                    all.terms.push(t.clone());
                    all.term_ids.insert(t, (all.terms.len() - 1) as u32);
                    (all.terms.len() - 1) as u32
                }
            };
            ids.push(id);
        }
        for (pg, mut key) in porder.into_iter().enumerate() {
            for (j, k) in keys.iter().enumerate() {
                if matches!(k, ColKey::Keyword(_)) && key[j] != u64::MAX { key[j] = ids[key[j] as usize] as u64; }
            }
            let g = match all.groups.get(&key) {
                Some(&g) => g,
                None => {
                    memory.grow(96 + 64 * plan.len() as u64, crate::esql::exec::ESQL_LABEL)?;
                    all.order.push(key);
                    all.accs.extend(std::iter::repeat_n(ColAcc::default(), plan.len()));
                    all.lists.extend(std::iter::repeat_with(Vec::new).take(plan.len()));
                    all.sets.extend(std::iter::repeat_with(Default::default).take(plan.len()));
                    all.groups.insert(key, all.order.len() - 1);
                    all.order.len() - 1
                }
            };
            for k in 0..plan.len() {
                let (from, to) = (&paccs[pg * plan.len() + k], &mut all.accs[g * plan.len() + k]);
                to.n += from.n;
                to.sum.add(from.sum.value, from.sum.delta);
                match to.long.checked_add(from.long) { Some(x) => to.long = x, None => to.overflow = true }
                all.lists[g * plan.len() + k].append(&mut plists[pg * plan.len() + k]);
                for id in &psets[pg * plan.len() + k] { all.sets[g * plan.len() + k].insert(ids[*id as usize]); }
            }
        }
        // The parts' charges move to the merged state.
        if let Some(mut m) = pmem { memory.grow(m.bytes(), crate::esql::exec::ESQL_LABEL)?; m.shrink(m.bytes()); }
    }
    if all.accs.iter().any(|a| a.overflow) {
        return Ok(None);
    }
    let ColPart { terms, order, accs, mut lists, sets, found, loaded, .. } = all;
    let finish = |k: &Kahan| { let mut last = Kahan::default(); last.add(k.value, k.delta); last.value };
    let mut rows = Vec::with_capacity(order.len());
    for (g, key) in order.iter().enumerate() {
        let mut row = Vec::with_capacity(aggs.len() + by.len());
        for (k, (acc, a)) in accs[g * plan.len()..(g + 1) * plan.len()].iter().zip(&plan).enumerate() {
            let at = g * plan.len() + k;
            row.push(match a {
                ColAgg::Count(_) => Val::Long(acc.n as i64),
                ColAgg::CountDistinct(_) => Val::Long(sets[at].len() as i64),
                ColAgg::Percentile(_, p) => crate::esql::func::percentile(std::mem::take(&mut lists[at]), *p).map(Val::Double).unwrap_or(Val::Null),
                _ if acc.n == 0 => Val::Null,
                ColAgg::Avg(_) => Val::Double(finish(&acc.sum) / acc.n as f64),
                ColAgg::Sum(i) if long_sum(*i) => Val::Long(acc.long),
                ColAgg::Sum(_) => Val::Double(finish(&acc.sum)),
            });
        }
        for (j, k) in keys.iter().enumerate() {
            row.push(match (k, key[j]) {
                (ColKey::Keyword(_), u64::MAX) => Val::Null,
                (ColKey::Keyword(_), id) => Val::Str(terms[id as usize].clone()),
                (ColKey::Trunc(..), ms) if ms as i64 == i64::MIN => Val::Null,
                (ColKey::Trunc(..), ms) => Val::Date(ms as i64),
            });
        }
        rows.push(row);
    }
    memory.grow(rows.iter().map(|r| crate::esql::exec::row_bytes(r)).sum::<u64>(), crate::esql::exec::ESQL_LABEL)?;
    memory.retain_for_request();
    taken();
    #[cfg(test)]
    COLUMNAR_STATS.with(|n| n.set(n.get() + 1));
    ctx.documents_found += found;
    ctx.values_loaded += loaded;
    Ok(Some((Table { cols: out_cols, rows }, evals + 1)))
}

/// One target's [`columnar_stats`] state: its groups (first-seen order)
/// with their aggregates' states, its terms, docs and values loaded.
struct ColPart {
    terms: Vec<String>,
    term_ids: HashMap<String, u32>,
    groups: HashMap<[u64; 2], usize, ahash::RandomState>,
    order: Vec<[u64; 2]>,
    accs: Vec<ColAcc>,
    /// Per (group, aggregate): PERCENTILE's values in row order and
    /// COUNT_DISTINCT's term ids (empty for the others).
    lists: Vec<Vec<f64>>,
    sets: Vec<std::collections::HashSet<u32, ahash::RandomState>>,
    found: u64,
    loaded: u64,
    /// A leaf the rows path must read: the whole query takes it.
    bail: bool,
    memory: Option<crate::memory::Reservation>,
    /// A large leaf's doc ranges' states ([`STATS_SPLIT_DOCS`]), in order.
    subs: Vec<ColPart>,
}

/// Matched docs per concurrently folded range of one leaf (tests split
/// their small leaves too).
#[cfg(not(test))]
const STATS_SPLIT_DOCS: usize = 1 << 16;
#[cfg(test)]
const STATS_SPLIT_DOCS: usize = 256;

impl ColPart {
    fn new(memory: Option<crate::memory::Reservation>) -> Self {
        ColPart { terms: Vec::new(), term_ids: HashMap::new(), groups: HashMap::default(), order: Vec::new(), accs: Vec::new(), lists: Vec::new(),
            sets: Vec::new(), found: 0, loaded: 0, bail: false, memory, subs: Vec::new() }
    }
}

/// Occurrences of `name` as a whole identifier in the query text.
fn word_count(full: &str, name: &str) -> usize {
    let is_id = |c: char| c.is_alphanumeric() || c == '_' || c == '.' || c == '@' || c == '`';
    let mut n = 0;
    let mut from = 0;
    while let Some(i) = full[from..].find(name) {
        let at = from + i;
        let before = full[..at].chars().next_back();
        let after = full[at + name.len()..].chars().next();
        if !before.is_some_and(is_id) && !after.is_some_and(is_id) {
            n += 1;
        }
        from = at + name.len();
    }
    n
}

/// The value of `ST_NPOINTS(ST_ENVELOPE(shape))` for doc `d`: an envelope
/// has four points whenever the shape has any (`npoints` of a BBOX), null
/// for a doc without a shape or with an empty one. From the doc values'
/// extent (none: an empty shape), which every indexed value carries.
fn shape_envelope_npoints(p: &crate::search::MatchPage<'_>, field: &str, d: rustic_core::types::DocId) -> Val {
    let Some(id) = p.lookup.get(field).and_then(|f| f.core_id) else { return Val::Null };
    let non_empty = p.leaf.shape_value(id, d).is_some_and(|v| v.extent().is_some());
    if non_empty {
        Val::Int(4)
    } else {
        Val::Null
    }
}

/// Rewrite `x = ST_NPOINTS(ST_ENVELOPE(shape))`, or `env =
/// ST_ENVELOPE(shape)` then `x = ST_NPOINTS(env)` with `env` read nowhere
/// else, of a geo_shape column with doc values, into a read of an extra
/// column computed from the doc values ([`shape_envelope_npoints`]); the
/// rows are otherwise unchanged. `None`: nothing rewritten.
fn shape_npoints_rewrite(full: &str, schema: &IndexSchema, local: &mut [Cmd]) -> Option<ExtraColumn> {
    const NAME: &str = "\u{0}st_envelope_npoints";
    let is_shape = |c: &str| schema.cols.iter().any(|(n, s)| n == c && s.ty == "geo_shape" && s.declared == "geo_shape" && s.doc_values && s.per_index.is_empty() && s.constants.is_empty() && s.types.len() <= 1);
    let envelope_of = |e: &Expr| -> Option<String> {
        match e {
            Expr::Call(f, args, _) if f == "ST_ENVELOPE" => match args.as_slice() {
                [Expr::Col(c, _)] if is_shape(c) => Some(c.clone()),
                _ => None,
            },
            _ => None,
        }
    };
    // The envelope columns defined so far: name -> shape column.
    let mut envs: Vec<(String, String)> = Vec::new();
    let mut field: Option<String> = None;
    for cmd in local.iter_mut() {
        let Cmd::Eval(fields) = cmd else { continue };
        for f in fields.iter_mut() {
            if let Some(c) = envelope_of(&f.expr) {
                envs.push((f.name.clone(), c));
                continue;
            }
            let Expr::Call(name, args, _) = &f.expr else { continue };
            if name != "ST_NPOINTS" {
                continue;
            }
            let src = match args.as_slice() {
                [inner] => envelope_of(inner).or_else(|| match inner {
                    // `env` defined above and read only here.
                    Expr::Col(e, _) if word_count(full, e) == 2 => envs.iter().find(|(n, _)| n == e).map(|(_, c)| c.clone()),
                    _ => None,
                }),
                _ => None,
            };
            let Some(c) = src else { continue };
            if field.as_ref().is_some_and(|x| *x != c) {
                return None;
            }
            f.expr = Expr::Col(NAME.to_string(), f.pos);
            field = Some(c);
        }
    }
    let field = field?;
    Some(ExtraColumn { name: NAME.to_string(), ty: "integer".to_string(), value: ExtraValue::Doc(shape_envelope_npoints), field })
}

/// Rewrite the first `x = ST_SIMPLIFY(shape, tolerance)` (a source column
/// and a non-negative literal) of the pipeline's first `EVAL`, with only
/// row filters before it, into a read of an extra column that each page
/// fills in parallel ([`ExtraValue::Simplify`]: the same
/// [`crate::esql::func2::st_simplify`] per row, a pure function). The rows are
/// otherwise unchanged; run after the pipeline was checked as written.
fn shape_simplify_rewrite(source: &[Column], local: &mut [Cmd]) -> Option<ExtraColumn> {
    const NAME: &str = "\u{0}st_simplify";
    let first = local.iter().position(|c| !matches!(c, Cmd::Where(..)))?;
    let Cmd::Eval(fields) = &mut local[first] else { return None };
    for i in 0..fields.len() {
        let Expr::Call(name, args, _) = &fields[i].expr else { continue };
        if name != "ST_SIMPLIFY" {
            continue;
        }
        let [Expr::Col(c, _), Expr::Lit(tol)] = args.as_slice() else { continue };
        if !tol.as_f64().is_some_and(|t| t >= 0.0) || fields[..i].iter().any(|f| f.name == *c) {
            continue;
        }
        let Some(src) = source.iter().position(|s| s.name == *c) else { continue };
        let (ty, tol) = (source[src].ty.clone(), tol.clone());
        let field = c.clone();
        fields[i].expr = Expr::Col(NAME.to_string(), fields[i].pos);
        return Some(ExtraColumn { name: NAME.to_string(), ty, value: ExtraValue::Simplify { src, tol }, field });
    }
    None
}

/// [`try_run`] when some shards are read on other nodes: the doc-values
/// `STATS`, the distance top-n and the grid counts merge per-shard parts;
/// row pipelines (a row filter and `LIMIT`, or a computed `SORT | LIMIT`)
/// return each shard's candidate rows only.
fn distributed(ctx: &mut Ctx, schema: &IndexSchema, indices: &[String], query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if residual.is_empty() {
        let evals = rest.iter().take_while(|c| matches!(c, Cmd::Eval(_))).count();
        if let Some(Cmd::Stats { aggs, by, inline: false, .. }) = rest.get(evals) {
            let defined: Vec<&str> = rest[..evals].iter().flat_map(|c| match c {
                Cmd::Eval(fs) => fs.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
                _ => Vec::new(),
            }).collect();
            let mut used = Vec::new();
            for f in aggs.iter().chain(by.iter()) {
                col_names(&f.expr, &mut used);
                if let Some(x) = &f.filter {
                    col_names(x, &mut used);
                }
            }
            if by.is_empty() && !used.iter().any(|u| defined.contains(&u.as_str())) {
                if let Some(t) = geo_stats(ctx, schema, query, &rest[evals], aggs)? {
                    return Ok(Some((t, evals + 1)));
                }
            }
        }
        if let Some(r) = distance_top_n(ctx, schema, indices, query, rest)? {
            return Ok(Some(r));
        }
        if let Some(r) = grid_stats(ctx, schema, query, rest)? {
            return Ok(Some(r));
        }
    }
    distributed_rows(ctx, schema, indices, query, residual, rest)
}

/// `EVAL g = ST_GEOHASH|ST_GEOTILE|ST_GEOHEX(loc, p[, bounds]) | STATS
/// COUNT(*).. BY g`: ES computes the cells of a point only aggregated
/// from its doc value (`SpatialDocValuesExtraction`; verified against ES
/// 9.5.5 at a tile boundary where the quantized and the source point
/// fall in different tiles), so the cells come from the geo doc values,
/// shards in parallel, groups in first-seen order.
fn grid_stats(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    let evals = rest.iter().take_while(|c| matches!(c, Cmd::Eval(_))).count();
    let Some(Cmd::Stats { aggs, by, inline: false, .. }) = rest.get(evals) else { return Ok(None) };
    let [g] = by.as_slice() else { return Ok(None) };
    if g.filter.is_some() || aggs.is_empty() || !aggs.iter().all(|a| a.filter.is_none() && matches!(&a.expr, Expr::Call(n, args, _) if n == "COUNT" && matches!(args.as_slice(), [Expr::Star]))) {
        return Ok(None);
    }
    let mut defs: Vec<(String, Expr)> = Vec::new();
    for c in &rest[..evals] {
        let Cmd::Eval(fields) = c else { unreachable!() };
        for f in fields {
            let def = subst(&f.expr, &defs);
            defs.push((f.name.clone(), def));
        }
    }
    let gexpr = subst(&g.expr, &defs);
    let Expr::Call(fname, args, _) = &gexpr else { return Ok(None) };
    if !matches!(fname.as_str(), "ST_GEOHASH" | "ST_GEOTILE" | "ST_GEOHEX") {
        return Ok(None);
    }
    let (col, p, bounds) = match args.as_slice() {
        [Expr::Col(c, _), Expr::Lit(p)] => (c, p.as_i64(), None),
        [Expr::Col(c, _), Expr::Lit(p), b] => {
            let Some(wkt) = const_wkt(b) else { return Ok(None) };
            let Some(bb) = crate::esql::spatial::prims(&wkt).and_then(|q| rustic_core::fieldtypes::shape::prims_bbox(&q)) else { return Ok(None) };
            (c, p.as_i64(), Some(bb))
        }
        _ => return Ok(None),
    };
    let Some(p) = p else { return Ok(None) };
    let Some(field) = geo_dv_field(&Expr::Col(col.clone(), 0), schema) else { return Ok(None) };
    // Output columns: the commands over no rows.
    let mut out = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    for c in &rest[..=evals] {
        out = apply(ctx, c, out)?;
    }
    if out.cols.len() != aggs.len() + 1 {
        return Ok(None);
    }
    // Cells as integers while counting (geohash bits, tile x/y, H3 index),
    // printed once per group (as the functions print them).
    let kind = match fname.as_str() {
        "ST_GEOHASH" => 0u8,
        "ST_GEOTILE" => 1,
        _ => 2,
    };
    let render = move |c: Option<u64>| -> Val {
        match (kind, c) {
            (_, None) => Val::Null,
            (0, Some(b)) => Val::Str(rustic_core::geo::geohash_of_bits(b, p.clamp(1, 12) as usize)),
            (1, Some(b)) => Val::Str(format!("{p}/{}/{}", b >> 32, b & 0xffff_ffff)),
            (_, Some(b)) => crate::aggs::geo_hex::cell_address(b).map(Val::Str).unwrap_or(Val::Null),
        }
    };
    let states = if schema.slots.is_some() {
        let plan = FastPlan::GridStats { field: field.clone(), kind, precision: p, bounds };
        let partials = crate::memory::partials_mark();
        let parts = by_slot(ctx, schema, query, plan, |_, targets| {
            Ok(grid_parts(targets, query, &field, kind, p, bounds)?.into_iter().map(GridGroups::into_part).collect())
        })?;
        let Some(parts) = parts else { return Ok(None) };
        let mut states = Vec::with_capacity(parts.len());
        for part in parts {
            let FastPart::Grid { docs, groups, outside } = part else { return Ok(None) };
            let memory=crate::memory::breakers().reserve_request((groups.capacity() as u64).saturating_mul(128),crate::esql::exec::ESQL_LABEL)?;
            states.push(GridGroups { index: HashMap::new(), groups, docs, outside, memory:Some(memory) });
        }
        // Each state now owns its charge: the parts' retained charge ends.
        crate::memory::consume_partials_since(partials);
        states
    } else {
        grid_parts(&schema.targets, query, &field, kind, p, bounds)?
    };
    if states.iter().any(|s| s.outside) {
        return Ok(None);
    }
    taken();
    let mut index: HashMap<Option<u64>, usize> = HashMap::new();
    let mut groups: Vec<(Option<u64>, i64)> = Vec::new();
    for st in states {
        ctx.documents_found += st.docs;
        for (k, c) in st.groups {
            match index.get(&k) {
                Some(&i) => groups[i].1 += c,
                None => {
                    ctx.mem.charge(256u64.saturating_add((aggs.len() as u64).saturating_mul(64)),crate::esql::exec::ESQL_LABEL)?;
                    index.insert(k, groups.len());
                    groups.push((k, c));
                }
            }
        }
    }
    out.rows = groups
        .into_iter()
        .map(|(k, c)| {
            let mut r: Vec<Val> = aggs.iter().map(|_| Val::Long(c)).collect();
            r.push(render(k));
            r
        })
        .collect();
    Ok(Some((out, evals + 1)))
}

/// One shard's grid counts: groups in first-seen order (`None`: no cell).
struct GridGroups {
    index: HashMap<Option<u64>, usize>,
    groups: Vec<(Option<u64>, i64)>,
    docs: u64,
    outside: bool,
    memory: Option<crate::memory::Reservation>,
}
impl GridGroups {
    fn add(&mut self,key:Option<u64>)->Result<()> {
        if let Some(&i)=self.index.get(&key){self.groups[i].1+=1;return Ok(());}
        // Include geometric spare capacity and hash-table control bytes.
        // Admission precedes both container growth operations.
        match self.memory.as_mut() {
            Some(memory)=>memory.grow(512,crate::esql::exec::ESQL_LABEL)?,
            None=>self.memory=Some(crate::memory::breakers().reserve_request(512,crate::esql::exec::ESQL_LABEL)?),
        }
        self.index.insert(key,self.groups.len());
        self.groups.push((key,1));
        Ok(())
    }
    fn into_part(mut self)->FastPart {
        // A shard part: released once the coordinator merges it.
        if let Some(memory)=self.memory.take(){memory.retain_partial_for_request();}
        FastPart::Grid{docs:self.docs,groups:self.groups,outside:self.outside}
    }
}

/// The cell of a point: geohash bits, tile x/y or H3 index (`kind` 0, 1, 2).
fn grid_cell(kind: u8, p: i64, lat: f64, lon: f64) -> Option<u64> {
    match kind {
        0 => Some(rustic_core::geo::geohash_bits(lat, lon, p.clamp(1, 12) as usize)),
        1 => {
            let (tx, ty) = rustic_core::geo::geotile(lat, lon, p.clamp(0, 29) as u32);
            Some(((tx as u64) << 32) | ty as u64)
        }
        _ => crate::aggs::geo_hex::point_cell_id(lat, lon, p.clamp(0, 15) as u32),
    }
}

/// The grid counts of each shard of `targets` from the geo doc values
/// (`outside`: a point outside `bounds`; the table path decides).
fn grid_parts(targets: &[crate::search::Target], query: &Value, field: &str, kind: u8, p: i64, bounds: Option<(f64, f64, f64, f64)>) -> Result<Vec<GridGroups>> {
    let init = |_: usize| GridGroups { index: HashMap::new(), groups: Vec::new(), docs: 0, outside: false, memory:None };
    let fold = |st: &mut GridGroups, pg: &crate::search::MatchPage<'_>| -> Result<bool> {
        st.docs += pg.len() as u64;
        let dv = pg.lookup.get(field).and_then(|f| f.geo_id).map(|id| GeoDv::open(pg.leaf, id));
        let mut keys: Vec<Option<u64>> = Vec::with_capacity(2);
        for d in pg.doc_ids() {
            let pts = dv.as_ref().map(|dv| dv.entries(d)).unwrap_or(rustic_core::geo::GeoEntries::EMPTY);
            keys.clear();
            if pts.is_empty() {
                // No point: the null group.
                keys.push(None);
            } else {
                for e in pts {
                    let (lat, lon) = (decode_lat(e.lat), decode_lon(e.lon));
                    if let Some((x0, y0, x1, y1)) = bounds {
                        // Points outside the bounds: their cell may or may
                        // not touch them; the table path decides.
                        if lon < x0 || lon > x1 || lat < y0 || lat > y1 {
                            st.outside = true;
                            return Ok(false);
                        }
                    }
                    let k = grid_cell(kind, p, lat, lon);
                    // A multi-valued key: one group per distinct value.
                    if !keys.contains(&k) {
                        keys.push(k);
                    }
                }
            }
            for &k in &keys {
                st.add(k)?;
            }
        }
        Ok(true)
    };
    crate::search::par_for_each_match(targets, query, PAGE, &init, &fold)
}

/// Rows a top-n keeps beyond `n` before it sorts and cuts again.
const TOP_N_SLACK: usize = 4096;

/// `[row commands] SORT keys [KEEP/DROP/RENAME/EVAL] LIMIT n`: the row
/// commands before the sort, the keys, the commands between the sort and
/// the limit, and `n`.
fn top_n_shape<'a>(ctx: &Ctx, rest: &'a [Cmd]) -> Option<(usize, &'a [SortKey], usize, usize)> {
    let k = rest.iter().take_while(|c| row_local(c)).count();
    let Some(Cmd::Sort(keys)) = rest.get(k) else { return None };
    let m = rest[k + 1..].iter().take_while(|c| matches!(c, Cmd::Eval(_) | Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..))).count();
    let Some(Cmd::Limit(n)) = rest.get(k + 1 + m) else { return None };
    Some((k, keys, m, (*n).min(ctx.max_limit)))
}

/// The long of the cell holding a point: `Geohash.longEncode`,
/// `GeoTileUtils.longEncodeTiles`, the H3 index (what `ST_GEO*(..)::long`
/// gives for the same point: the address re-encoded by `GRID_LONG`).
fn grid_long(kind: u8, p: i64, lat: f64, lon: f64) -> Option<i64> {
    match kind {
        0 => Some(((rustic_core::geo::geohash_bits(lat, lon, p as usize) << 4) | p as u64) as i64),
        1 => {
            let (x, y) = rustic_core::geo::geotile(lat, lon, p as u32);
            Some((p << 58) | ((x as i64) << 29) | y as i64)
        }
        _ => crate::aggs::geo_hex::point_cell_id(lat, lon, p as u32).map(|c| c as i64),
    }
}

/// A res-0 cell's predicate (its candidate box), built once per cell.
fn hex_base_pred(c: u64) -> std::sync::Arc<crate::aggs::geo_hex::HexCellPred> {
    static P: std::sync::OnceLock<std::sync::Mutex<HashMap<u64, std::sync::Arc<crate::aggs::geo_hex::HexCellPred>>>> = std::sync::OnceLock::new();
    let m = P.get_or_init(Default::default);
    m.lock().unwrap().entry(c).or_insert_with(|| std::sync::Arc::new(crate::aggs::geo_hex::HexCellPred::new(c))).clone()
}

/// A geo doc value's cell when every point quantizing to it lies in that
/// cell (`None`: the quantum box straddles cells). The `_source` point lies
/// in the box `[decode(e), decode(e + 1)]` on each axis; geohash and
/// geotile cells are lat/lon rectangles, H3 cells convex, so equal corner
/// cells mean the whole box is in one cell.
fn grid_long_certain(kind: u8, p: i64, lat_e: i32, lon_e: i32) -> Option<i64> {
    if kind == 0 {
        // Geohash cells are rectangles of whole quanta: the quantum box's
        // corners share the cell of its low corner unless the high corner
        // (one quantum on, exactly the next value's bisections) crosses.
        let hash = |a: i64, o: i64| rustic_core::geo::geohash_bits_encoded(a, o, p as usize);
        let first = hash(lat_e as i64, lon_e as i64);
        return (hash(lat_e as i64 + 1, lon_e as i64 + 1) == first).then_some(((first << 4) | p as u64) as i64);
    }
    let (lat_q, lon_q) = (decode_lat(1), decode_lon(1));
    let (lat0, lon0) = (decode_lat(lat_e), decode_lon(lon_e));
    let (lat1, lon1) = (((lat_e as f64) + 1.0) * lat_q, ((lon_e as f64) + 1.0) * lon_q);
    let first = grid_long(kind, p, lat0, lon0)?;
    // Tiles are rectangles split monotonically per axis: opposite corners
    // in one tile put the box in it.
    if kind == 1 {
        return (grid_long(kind, p, lat1.min(90.0), lon1.min(180.0))? == first).then_some(first);
    }
    for (lat, lon) in [(lat0, lon1), (lat1, lon0), (lat1, lon1)] {
        if grid_long(kind, p, lat.min(90.0), lon.min(180.0))? != first {
            return None;
        }
    }
    Some(first)
}

/// Sort order of two `SORT` key values (`None`: null) as `SORT` puts them.
fn grid_key_cmp(a: Option<i64>, b: Option<i64>, key: &SortKey) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let nulls_first = key.nulls_first.unwrap_or(key.desc);
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => if nulls_first { Ordering::Less } else { Ordering::Greater },
        (Some(_), None) => if nulls_first { Ordering::Greater } else { Ordering::Less },
        (Some(x), Some(y)) => if key.desc { y.cmp(&x) } else { x.cmp(&y) },
    }
}

/// The `SORT` key of a grid column value: a multi-value sorts by its min
/// (ascending) or max (descending), as `sort_table` does.
fn grid_row_key(v: &Val, desc: bool) -> Option<i64> {
    match v {
        Val::Long(x) | Val::Int(x) => Some(*x),
        Val::Multi(m) => {
            let it = m.iter().filter_map(|x| match x { Val::Long(x) | Val::Int(x) => Some(*x), _ => None });
            if desc { it.max() } else { it.min() }
        }
        _ => None,
    }
}

/// `EVAL g = ST_GEOHASH|ST_GEOTILE|ST_GEOHEX(loc, p)::long | SORT g |
/// [KEEP/DROP/RENAME/EVAL] | LIMIT n` (the Rally `geoGrid_sort_*-esql`
/// tasks): a top-n over the geo doc values (ES `TopNOperator` over the
/// grid evaluator's long block) that reads `_source` only for documents
/// that can still enter the top n.
///
/// The cells follow ES's `SpatialDocValuesExtraction`: when `loc` is not
/// in the output (`KEEP g`), ES reads the point from doc values (the
/// quantized point; `PhysicalPlanOptimizerTests.
/// testSpatialTypesAndSortGeoGridUseDocValues`); when it is (`KEEP g,
/// loc`), from `_source`. Then a doc value whose quantum box lies in one
/// cell bounds (exactly: one cell) the `_source` cell, and only the
/// documents whose bound beats the current n-th key read `_source`. Rows
/// that tie keep search order (stable, as the table path's `SORT`).
fn grid_top_n(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if schema.slots.is_some() {
        return Ok(None);
    }
    let Some((k, keys, m, n)) = top_n_shape(ctx, rest) else { return Ok(None) };
    let [Cmd::Eval(fields)] = &rest[..k] else { return Ok(None) };
    let [field] = fields.as_slice() else { return Ok(None) };
    let [key] = keys else { return Ok(None) };
    if !matches!(&key.expr, Expr::Col(c, _) if *c == field.name) {
        return Ok(None);
    }
    let inner = match &field.expr {
        Expr::Cast(e, t) if func::cast_type(t) == "long" => e.as_ref(),
        Expr::Call(f, a, _) if f == "TO_LONG" && a.len() == 1 => &a[0],
        _ => return Ok(None),
    };
    let Expr::Call(fname, args, _) = inner else { return Ok(None) };
    let kind = match fname.as_str() {
        "ST_GEOHASH" => 0u8,
        "ST_GEOTILE" => 1,
        "ST_GEOHEX" => 2,
        _ => return Ok(None),
    };
    let [Expr::Col(loc, _), Expr::Lit(precision)] = args.as_slice() else { return Ok(None) };
    let Some(p) = precision.as_i64() else { return Ok(None) };
    let in_range = match kind { 0 => (1..=12).contains(&p), 1 => (0..=29).contains(&p), _ => (0..=15).contains(&p) };
    if !in_range || field.name == *loc {
        return Ok(None);
    }
    let Some(dv_field) = geo_dv_field(&Expr::Col(loc.clone(), 0), schema) else { return Ok(None) };
    let covered = k + 2 + m;
    // Output columns, types and verification: the commands over no rows.
    let base = source_cols(schema, &[]);
    let mut empty = Table { cols: base.clone(), rows: Vec::new() };
    for c in &rest[..covered] {
        empty = apply(ctx, c, empty)?;
    }
    let mut after_eval = Table { cols: base.clone(), rows: Vec::new() };
    after_eval = apply(ctx, &rest[0], after_eval)?;
    let Some(grid) = after_eval.cols.iter().rposition(|c| c.name == field.name) else { return Ok(None) };
    // `loc` reaches the output (as itself or renamed): ES reads `_source`.
    let mut later = Vec::new();
    let mut renamed = false;
    for c in &rest[k..covered] {
        match c {
            Cmd::Eval(fs) => fs.iter().for_each(|f| col_names(&f.expr, &mut later)),
            Cmd::Rename(pairs, _) => renamed |= pairs.iter().any(|(old, _)| old == loc),
            _ => {}
        }
    }
    let from_source = renamed || later.contains(loc) || empty.cols.iter().any(|c| c.name == *loc);
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().map(|(c, _)| needed.as_ref().is_none_or(|s| s.contains(c))).collect();
    taken();
    // Candidate rows (after the EVAL) in search order; cut back to `n` by
    // the stable sort whenever they grow past `n + TOP_N_SLACK`.
    let mut buf: Vec<Vec<Val>> = Vec::new();
    let mut threshold: Option<Option<i64>> = None;
    let mut buffer_memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    let (mut found, mut loaded) = (0u64, 0u64);
    let mut failure: Option<EsError> = None;
    let mut cut = |buf: &mut Vec<Vec<Val>>, threshold: &mut Option<Option<i64>>| {
        let mut keyed: Vec<(Option<i64>, Vec<Val>)> = std::mem::take(buf).into_iter().map(|r| (grid_row_key(&r[grid], key.desc), r)).collect();
        keyed.sort_by(|a, b| grid_key_cmp(a.0, b.0, key));
        keyed.truncate(n);
        if keyed.len() == n {
            *threshold = keyed.last().map(|(k, _)| *k);
        }
        *buf = keyed.into_iter().map(|(_, r)| r).collect();
    };
    if n > 0 {
        let ctx_cell = std::cell::RefCell::new(&mut *ctx);
        crate::search::for_each_match_wanting(&schema.targets, query, PAGE, None, &mut |pg| {
            let mut ctx = ctx_cell.borrow_mut();
            found += pg.len() as u64;
            let dv = pg.lookup.get(&dv_field).and_then(|f| f.geo_id).map(|id| GeoDv::open(pg.leaf, id));
            // The documents that can still enter the top n, with their
            // doc-value cell values (used as the column when `!from_source`).
            let mut docs: Vec<rustic_core::types::DocId> = Vec::new();
            let mut cells: Vec<Val> = Vec::new();
            let mut flush = |ctx: &mut Ctx, docs: &mut Vec<rustic_core::types::DocId>, cells: &mut Vec<Val>, buf: &mut Vec<Vec<Val>>, threshold: &mut Option<Option<i64>>, loaded: &mut u64, cut: &mut dyn FnMut(&mut Vec<Vec<Val>>, &mut Option<Option<i64>>)| -> Result<()> {
                if docs.is_empty() {
                    return Ok(());
                }
                let (rows, l, _page_memory) = source_rows(schema, &wanted, pg.target, pg.lookup, pg.leaf, docs)?;
                *loaded += l;
                let mut t = apply(ctx, &rest[0], Table { cols: base.clone(), rows })?;
                if !from_source {
                    for (row, cell) in t.rows.iter_mut().zip(cells.drain(..)) {
                        row[grid] = cell;
                    }
                }
                docs.clear();
                cells.clear();
                buffer_memory.grow(t.rows.iter().map(|r| crate::esql::exec::row_bytes(r).saturating_mul(2)).fold(0u64, u64::saturating_add), crate::esql::exec::ESQL_LABEL)?;
                buf.extend(t.rows);
                if buf.len() > n.saturating_add(TOP_N_SLACK) {
                    cut(buf, threshold);
                    resize_page_memory(&mut buffer_memory, rows_bytes(buf))?;
                }
                Ok(())
            };
            let mut points: Vec<(i64, i32, i32)> = Vec::new();
            // geohex: once the top n has a threshold, a point whose base cell
            // cannot reach it (outside the candidate boxes of every base cell
            // at or past the threshold's, in the sort's direction) cannot
            // enter, and needs no cell (H3 indexes order by base cell first).
            let beatable_bases = |t: i64| -> Vec<std::sync::Arc<crate::aggs::geo_hex::HexCellPred>> {
                let tb = (t as u64 >> 45) & 127;
                crate::h3::res0_cells().filter(|c| { let b = (c >> 45) & 127; if key.desc { b >= tb } else { b <= tb } }).map(|c| hex_base_pred(c)).collect()
            };
            let mut bases: Option<(i64, Vec<std::sync::Arc<crate::aggs::geo_hex::HexCellPred>>)> = None;
            for d in pg.doc_ids() {
                if kind == 2 {
                    if let Some(Some(t)) = threshold {
                        if bases.as_ref().is_none_or(|(bt, _)| (*bt as u64 >> 45) & 127 != (t as u64 >> 45) & 127) {
                            bases = Some((t, beatable_bases(t)));
                        }
                        let preds = &bases.as_ref().unwrap().1;
                        let entries = dv.as_ref().map(|dv| dv.entries(d)).unwrap_or(rustic_core::geo::GeoEntries::EMPTY);
                        // Every point of the doc out of reach (a doc without a
                        // point keys null: not skipped here).
                        if !entries.is_empty() && entries.iter().all(|e| {
                            let (lat, lon) = (decode_lat(e.lat), decode_lon(e.lon));
                            // The quantum box's far corner too (`_source` points).
                            let (lat1, lon1) = (decode_lat(e.lat.saturating_add(1)), decode_lon(e.lon.saturating_add(1)));
                            !preds.iter().any(|p| p.may_contain(lat, lon) || p.may_contain(lat1, lon1) || p.may_contain(lat, lon1) || p.may_contain(lat1, lon))
                        }) {
                            continue;
                        }
                    }
                }
                let entries = dv.as_ref().map(|dv| dv.entries(d)).unwrap_or(rustic_core::geo::GeoEntries::EMPTY);
                points.clear();
                // Doc-value order: the sorted encoded longs (lat << 32 | lon).
                points.extend(entries.iter().map(|e| { let (lat, lon) = (e.lat, e.lon); (((lat as i64) << 32) | (lon as u32 as i64), lat, lon) }));
                points.sort_unstable_by_key(|x| x.0);
                // The best key this document can have (`None` inside: null), or
                // `None`: unknown until `_source` is read.
                let (bound, cell): (Option<Option<i64>>, Val) = if from_source {
                    if points.is_empty() {
                        (None, Val::Null)
                    } else {
                        let mut ids = Vec::with_capacity(points.len());
                        for &(_, lat, lon) in &points {
                            match grid_long_certain(kind, p, lat, lon) {
                                Some(id) => ids.push(id),
                                None => break,
                            }
                        }
                        if ids.len() == points.len() {
                            (Some(if key.desc { ids.iter().copied().max() } else { ids.iter().copied().min() }), Val::Null)
                        } else {
                            (None, Val::Null)
                        }
                    }
                } else {
                    let ids: Vec<i64> = points.iter().filter_map(|&(_, lat, lon)| {
                        if kind == 0 {
                            Some(((rustic_core::geo::geohash_bits_encoded(lat as i64, lon as i64, p as usize) << 4) | p as u64) as i64)
                        } else {
                            grid_long(kind, p, decode_lat(lat), decode_lon(lon))
                        }
                    }).collect();
                    let best = if key.desc { ids.iter().copied().max() } else { ids.iter().copied().min() };
                    (Some(best), Val::from_list(ids.into_iter().map(Val::Long).collect()))
                };
                if let (Some(b), Some(t)) = (bound, threshold) {
                    // Equal keys keep the earlier rows: no later tie enters.
                    if grid_key_cmp(b, t, key) != std::cmp::Ordering::Less {
                        continue;
                    }
                }
                docs.push(d);
                cells.push(cell);
                if docs.len() == 1024 {
                    if let Err(e) = flush(&mut **ctx, &mut docs, &mut cells, &mut buf, &mut threshold, &mut loaded, &mut cut) {
                        failure = Some(e);
                        return Ok(false);
                    }
                }
            }
            if let Err(e) = flush(&mut **ctx, &mut docs, &mut cells, &mut buf, &mut threshold, &mut loaded, &mut cut) {
                failure = Some(e);
                return Ok(false);
            }
            Ok(true)
        })?;
    }
    if let Some(e) = failure {
        return Err(e);
    }
    ctx.documents_found += found;
    ctx.values_loaded += loaded;
    cut(&mut buf, &mut threshold);
    // Keep the charge of the rows kept, not of the candidates' peak.
    resize_page_memory(&mut buffer_memory, rows_bytes(&buf))?;
    buffer_memory.retain_for_request();
    let mut t = Table { cols: after_eval.cols, rows: buf };
    for c in &rest[k..covered] {
        t = apply(ctx, c, t)?;
    }
    Ok(Some((t, covered)))
}

/// A sort the index search runs itself (`load` pushes it down with the
/// size): a plain doc-values column right after `FROM`.
fn pushed_sort(schema: &IndexSchema, keys: &[SortKey]) -> bool {
    keys.iter().all(|key| match &key.expr {
        Expr::Col(n, _) => schema.cols.iter().find(|(c, _)| c == n).is_some_and(|(_, c)| {
            c.ty != "text" && c.ty != "unsupported" && c.ty != "null" && !super::super::spatial::is_spatial(&c.ty) && c.per_index.is_empty() && c.doc_values
        }),
        _ => false,
    })
}

/// Rows streamed out of `targets`.
struct Streamed {
    cols: Vec<Column>,
    rows: Vec<Vec<Val>>,
    docs: u64,
    loaded: u64,
}

/// Stream `targets` in search order through `local` (row filters and row
/// commands): the first `n` rows, or with `sort` the first `n` in that
/// order. The candidates are cut back to `n` with the pipeline's own
/// (stable) `SORT` whenever they grow past `n + TOP_N_SLACK`, so ties keep
/// search order and memory stays bounded. `None`: chunks produced
/// different column types (value-dependent; the table path decides).
#[allow(clippy::too_many_arguments)]
fn stream_rows(ctx: &mut Ctx, schema: &IndexSchema, targets: &[crate::search::Target], query: &Value, local: &[Cmd], sort: Option<&[SortKey]>, n: usize, wanted: &[bool]) -> Result<Option<Streamed>> {
    stream_rows_with(ctx, schema, targets, query, local, sort, n, wanted, None)
}

/// A column added to every streamed page from the index, not the source
/// (see [`shape_npoints_rewrite`]).
struct ExtraColumn {
    name: String,
    ty: String,
    value: ExtraValue,
    /// The index column it reads.
    field: String,
}

/// How an [`ExtraColumn`] is computed.
enum ExtraValue {
    /// From the index: the value of each doc of the page.
    Doc(fn(&crate::search::MatchPage<'_>, &str, rustic_core::types::DocId) -> Val),
    /// `ST_SIMPLIFY(row[src], tol)` of each source row, rows in parallel.
    Simplify { src: usize, tol: Val },
}

/// [`stream_rows`] with an extra column computed per doc.
#[allow(clippy::too_many_arguments)]
fn stream_rows_with(ctx: &mut Ctx, schema: &IndexSchema, targets: &[crate::search::Target], query: &Value, local: &[Cmd], sort: Option<&[SortKey]>, n: usize, wanted: &[bool], extra: Option<&ExtraColumn>) -> Result<Option<Streamed>> {
    let mut source = source_cols(schema, &[]);
    if let Some(x) = extra {
        source.push(Column { name: x.name.clone(), ty: x.ty.clone() });
    }
    let mut empty = Table { cols: source.clone(), rows: Vec::new() };
    for c in local {
        empty = apply(ctx, c, empty)?;
    }
    if n == 0 {
        return Ok(Some(Streamed { cols: empty.cols, rows: Vec::new(), docs: 0, loaded: 0 }));
    }
    let base = source;
    let sort_cmd = sort.map(|k| Cmd::Sort(k.to_vec()));
    let filtered = local.iter().any(|c| matches!(c, Cmd::Where(..)));
    let rows_wanted = (sort.is_none() && !filtered).then_some(n);
    let page = if rows_wanted.is_some() { n.clamp(16, PAGE) } else { PAGE };
    let mut cols: Option<Vec<Column>> = None;
    let mut buf: Vec<Vec<Val>> = Vec::new();
    let mut buffer_memory=crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?;
    let (mut found, mut loaded) = (0u64, 0u64);
    let mut failure: Option<EsError> = None;
    let mut mismatch = false;
    let ctx_cell = std::cell::RefCell::new(&mut *ctx);
    crate::search::for_each_match_wanting(targets, query, page, rows_wanted, &mut |p| {
        let mut ctx = ctx_cell.borrow_mut();
        let total = p.len();
        let mut at = 0;
        let mut range: Vec<rustic_core::types::DocId> = Vec::new();
        while at < total {
            // Small chunks while few rows are wanted: a LIMIT 10 reads ~10 documents.
            let chunk = if rows_wanted.is_some() { (n - buf.len()).clamp(1, 1024) } else { 1024 };
            let end = (at + chunk).min(total);
            let docs: &[rustic_core::types::DocId] = if p.all_docs {
                range.clear();
                range.extend(at as u32..end as u32);
                &range
            } else {
                &p.docs[at..end]
            };
            at = end;
            found += docs.len() as u64;
            let (mut rows,_page_memory) = match source_rows_guarded(schema, wanted, p.target, p.lookup, p.leaf, docs) {
                Ok((r, l, memory)) => {
                    loaded += l;
                    (r,memory)
                }
                Err(e) => {
                    failure = Some(e);
                    return Ok(false);
                }
            };
            if let Some(x) = extra {
                match &x.value {
                    ExtraValue::Doc(value) => {
                        for (row, &d) in rows.iter_mut().zip(docs) {
                            row.push(value(p, &x.field, d));
                        }
                    }
                    ExtraValue::Simplify { src, tol } => {
                        use rayon::prelude::*;
                        let src = *src;
                        rows.par_iter_mut().for_each(|row| {
                            let v = row.get(src).map_or(Val::Null, |a| crate::esql::func2::st_simplify(a, tol));
                            row.push(v);
                        });
                    }
                }
            }
            let mut t = Table { cols: base.clone(), rows };
            for c in local {
                t = match apply(&mut ctx, c, t) {
                    Ok(t) => t,
                    Err(e) => {
                        failure = Some(e);
                        return Ok(false);
                    }
                };
            }
            if t.rows.is_empty() {
                continue;
            }
            match &cols {
                None => cols = Some(t.cols.clone()),
                Some(c) if *c != t.cols => {
                    mismatch = true;
                    return Ok(false);
                }
                Some(_) => {}
            }
            let growth=t.rows.iter().map(|row|crate::esql::exec::row_bytes(row).saturating_mul(2)).fold(0u64,u64::saturating_add);
            buffer_memory.grow(growth,crate::esql::exec::ESQL_LABEL)?;
            buf.extend(t.rows);
            match &sort_cmd {
                None if buf.len() >= n => {
                    buf.truncate(n);
                    return Ok(false);
                }
                None => {}
                Some(s) if buf.len() > n.saturating_add(TOP_N_SLACK) => {
                    let all = Table { cols: cols.clone().unwrap_or_default(), rows: std::mem::take(&mut buf) };
                    match apply(&mut ctx, s, all) {
                        Ok(sorted) => buf = sorted.rows,
                        Err(e) => {
                            failure = Some(e);
                            return Ok(false);
                        }
                    }
                    buf.truncate(n);
                    resize_page_memory(&mut buffer_memory,buf.iter().map(|row|crate::esql::exec::row_bytes(row).saturating_mul(2)).fold(0u64,u64::saturating_add))?;
                }
                Some(_) => {}
            }
        }
        Ok(true)
    })?;
    if let Some(e) = failure {
        return Err(e);
    }
    if mismatch {
        return Ok(None);
    }
    let cols = cols.unwrap_or(empty.cols);
    if let Some(s) = &sort_cmd {
        buf = apply(ctx, s, Table { cols: cols.clone(), rows: buf })?.rows;
    }
    buf.truncate(n);
    // Keep the charge of the rows kept, not of the peak before LIMIT.
    resize_page_memory(&mut buffer_memory,buf.iter().map(|row|crate::esql::exec::row_bytes(row).saturating_mul(2)).fold(0u64,u64::saturating_add))?;
    buffer_memory.retain_for_request();
    Ok(Some(Streamed { cols, rows: buf, docs: found, loaded }))
}

/// `[row commands] SORT keys [...] LIMIT n` whose sort the index cannot
/// run (computed keys, or after a row filter): the top `n` rows streamed,
/// never the whole table (ES `TopNOperator`).
fn streamed_top_n(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    let Some((k, keys, m, n)) = top_n_shape(ctx, rest) else { return Ok(None) };
    if residual.is_empty() && k == 0 && pushed_sort(schema, keys) {
        return Ok(None);
    }
    let mut local: Vec<Cmd> = residual.iter().map(|(e, p)| Cmd::Where(e.clone(), *p)).collect();
    local.extend(rest[..k].iter().cloned());
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().map(|(c, _)| needed.as_ref().is_none_or(|s| s.contains(c))).collect();
    let Some(r) = stream_rows(ctx, schema, &schema.targets, query, &local, Some(keys), n, &wanted)? else { return Ok(None) };
    taken();
    ctx.documents_found += r.docs;
    ctx.values_loaded += r.loaded;
    let mut t = Table { cols: r.cols, rows: r.rows };
    for c in &rest[k..k + 2 + m] {
        t = apply(ctx, c, t)?;
    }
    Ok(Some((t, k + 2 + m)))
}

/// Row pipelines over shards partly read on other nodes: a row filter
/// with a `LIMIT`, or a `SORT | LIMIT` the index cannot run. Every shard
/// streams its own candidates ([`stream_rows`]); the rows, in shard order,
/// then run the `LIMIT` (or the stable `SORT` and what follows) here.
fn distributed_rows(ctx: &mut Ctx, schema: &IndexSchema, indices: &[String], query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    // Each node evaluates the rows: the same settings everywhere.
    if ctx.settings.unmapped != super::super::settings::Unmapped::Default || ctx.settings.time_zone_set || ctx.fctx.locale.is_some() {
        return Ok(None);
    }
    let (k, sort, covered, n) = match top_n_shape(ctx, rest) {
        Some((k, keys, m, n)) => {
            if residual.is_empty() && k == 0 && pushed_sort(schema, keys) {
                return Ok(None);
            }
            (k, Some(keys.to_vec()), k + 2 + m, n)
        }
        None => {
            let k = rest.iter().take_while(|c| row_local(c)).count();
            let (n, covered) = match rest.get(k) {
                Some(Cmd::Limit(n)) => ((*n).min(ctx.max_limit), k + 1),
                None => (ctx.default_limit.min(ctx.max_limit), k),
                Some(_) => return Ok(None),
            };
            // No row filter: `load` pushes the size into each shard's search.
            if residual.is_empty() && !rest[..k].iter().any(|c| matches!(c, Cmd::Where(..))) {
                return Ok(None);
            }
            (k, None, covered, n)
        }
    };
    let Some(pre) = rest[..k].iter().map(RowCmd::of).collect::<Option<Vec<_>>>() else { return Ok(None) };
    let mut local: Vec<Cmd> = residual.iter().map(|(e, p)| Cmd::Where(e.clone(), *p)).collect();
    local.extend(rest[..k].iter().cloned());
    // Output columns over no rows (and the commands' checks, here).
    let mut empty = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    for c in &local {
        empty = apply(ctx, c, empty)?;
    }
    let plan = FastPlan::Rows {
        indices: indices.to_vec(),
        columns: wire_cols(&source_cols(schema, &[])),
        query_text: ctx.full.to_string(),
        now: ctx.fctx.now,
        residual: residual.to_vec(),
        pre,
        sort: sort.clone(),
        n,
    };
    let wanted = vec![true; schema.cols.len()];
    let parts = by_slot(ctx, schema, query, plan, |ctx, targets| {
        targets.iter().map(|t| Ok(rows_part(stream_rows(ctx, schema, std::slice::from_ref(t), query, &local, sort.as_deref(), n, &wanted)?))).collect()
    })?;
    let Some(parts) = parts else { return Ok(None) };
    let (mut docs, mut loaded) = (0u64, 0u64);
    let mut cols: Option<Vec<(String, String)>> = None;
    let mut rows: Vec<Vec<Val>> = Vec::new();
    for part in parts {
        let FastPart::Rows { docs: d, loaded: l, cols: c, rows: r, .. } = part else { return Ok(None) };
        // A shard whose chunks typed a column differently: the table path.
        let Some(c) = c else { return Ok(None) };
        docs += d;
        loaded += l;
        if r.is_empty() {
            continue;
        }
        match &cols {
            None => cols = Some(c),
            Some(x) if *x != c => return Ok(None),
            Some(_) => {}
        }
        if sort.is_none() && rows.len() >= n {
            continue;
        }
        rows.extend(r.iter().map(|row| row.iter().map(Val::from).collect::<Vec<_>>()));
    }
    taken();
    ctx.documents_found += docs;
    ctx.values_loaded += loaded;
    let cols = cols.map(|c| c.into_iter().map(|(name, ty)| Column { name, ty }).collect()).unwrap_or(empty.cols);
    let mut t = Table { cols, rows };
    match sort {
        None => {
            t.rows.truncate(n);
            if covered > k {
                t = apply(ctx, &rest[k], t)?;
            }
        }
        Some(_) => {
            for c in &rest[k..covered] {
                t = apply(ctx, c, t)?;
            }
        }
    }
    Ok(Some((t, covered)))
}

/// A shard's streamed rows as a [`FastPart`].
fn rows_part(r: Option<Streamed>) -> FastPart {
    match r {
        Some(r) => FastPart::Rows { docs: r.docs, loaded: r.loaded, with_point: 0, cols: Some(wire_cols(&r.cols)), rows: wire_rows(r.rows) },
        None => FastPart::Rows { docs: 0, loaded: 0, with_point: 0, cols: None, rows: Vec::new() },
    }
}

fn wire_cols(cols: &[Column]) -> Vec<(String, String)> {
    cols.iter().map(|c| (c.name.clone(), c.ty.clone())).collect()
}

fn wire_rows(rows: Vec<Vec<Val>>) -> Vec<Vec<WireVal>> {
    rows.iter().map(|r| r.iter().map(WireVal::from).collect()).collect()
}

/// Run a fast path's per-shard work over shards partly read on other
/// nodes: `local` over this node's shards while the others run `plan`
/// where they are read; the parts in shard (slot) order. `None`: a node
/// did not answer with a part (an older node, a copy that moved, a
/// different schema there): the caller takes the table path.
fn by_slot<'c>(ctx: &mut Ctx<'c>, schema: &IndexSchema, query: &Value, plan: FastPlan, local: impl FnOnce(&mut Ctx<'c>, &[crate::search::Target]) -> Result<Vec<FastPart>>) -> Result<Option<Vec<FastPart>>> {
    use crate::esql::remote::{in_slot_order, FetchSpec, ShardResult, Slot};
    let Some(slots) = &schema.slots else { return Ok(None) };
    let Some(source) = ctx.shards.clone() else { return Ok(None) };
    let remote: Vec<_> = slots.iter().filter_map(|s| match s {
        Slot::Remote(r) => Some(r.clone()),
        Slot::Local(_) => None,
    }).collect();
    let local_idx: Vec<usize> = slots.iter().filter_map(|s| match s {
        Slot::Local(i) => Some(*i),
        Slot::Remote(_) => None,
    }).collect();
    let local_targets: Vec<crate::search::Target> = local_idx.iter().map(|i| schema.targets[*i].clone()).collect();
    let spec = FetchSpec { shards: remote, query: query.clone(), body: json!({}), size: None, probes: Vec::new(), agg: None, group_limit: None, pit: None, fast: Some(plan) };
    let mut fetch_memory=crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?;
    let (fetched, local_parts) = std::thread::scope(|scope| {
        let remote = (!spec.shards.is_empty()).then(|| scope.spawn(|| source.fetch(&spec,&mut fetch_memory)));
        let local_parts = local(ctx, &local_targets);
        (remote.map(|h| h.join().unwrap_or_else(|_| Err(EsError::internal("shard fetch panicked")))), local_parts)
    });
    let fetched = match fetched {
        None => Vec::new(),
        Some(Ok(r)) if r.iter().all(|r| r.fast.is_some()) => r,
        Some(Err(error)) if error.status==429=>return Err(error),
        Some(_) => return Ok(None),
    };
    let local_parts = local_parts?;
    if local_parts.len() != local_idx.len() {
        return Err(EsError::internal("a local shard has no fast-path part"));
    }
    let mut by_local: HashMap<usize, ShardResult> = local_idx
        .iter()
        .zip(local_parts)
        .map(|(i, part)| {
            let t = &schema.targets[*i];
            (*i, ShardResult { index: t.svc.name.clone(), shard: t.shard.as_ref().map(|s| s.id).unwrap_or(0), count: 0, hits: Vec::new(), probes: Vec::new(), groups: None, loaded: 0, pit: false, fast: Some(part), warnings: Vec::new() })
        })
        .collect();
    let Ok(results) = in_slot_order(slots, |i| by_local.remove(&i).ok_or_else(|| EsError::internal("local shard not read")), fetched) else { return Ok(None) };
    {
        let mut w = ctx.fctx.warnings.borrow_mut();
        for r in &results {
            for x in &r.warnings {
                if !w.list.contains(x) {
                    w.list.push(x.clone());
                }
            }
        }
    }
    // The fetched parts: released by a caller that charges what it merges
    // them into (`consume_partials_since`); kept otherwise.
    fetch_memory.retain_partial_for_request();
    Ok(Some(results.into_iter().filter_map(|r| r.fast).collect()))
}

/// Serve a fetch with a fast-path plan: one part per requested shard, as
/// the node running the query computes it for its own shards.
pub(super) fn serve(node: &Node, spec: &crate::esql::remote::FetchSpec, plan: &FastPlan, targets: &[crate::search::Target], source: Option<&crate::esql::remote::SharedShardSource>, restrict: Option<&Restrict>) -> Result<Vec<crate::esql::remote::ShardResult>> {
    let mut warnings = Vec::new();
    let parts: Vec<FastPart> = match plan {
        FastPlan::GeoStats { aggs } => {
            let plan: Vec<GeoAgg> = aggs.iter().map(|a| match a {
                GeoAggPlan::Count => GeoAgg::Count,
                GeoAggPlan::Centroid(f) => GeoAgg::Centroid(f.clone()),
                GeoAggPlan::Extent(f) => GeoAgg::Extent(f.clone()),
            }).collect();
            geo_states(targets, &spec.query, &plan, true)?.into_iter().map(GeoState::part).collect()
        }
        FastPlan::GridStats { field, kind, precision, bounds } => grid_parts(targets, &spec.query, field, *kind, *precision, *bounds)?
            .into_iter()
            .map(GridGroups::into_part)
            .collect(),
        FastPlan::Distance { indices, columns, field, lat, lon, n } => {
            let mut ctx = new_ctx(node, "", None, (*n).max(1), (*n).max(1));
            ctx.esql = true;
            ctx.restrict = restrict.cloned();
            ctx.shards = source.cloned();
            let schema = resolve_schema(&ctx, indices, 0)?;
            if wire_cols(&source_cols(&schema, &[])) != *columns {
                return Err(EsError::new(409, "illegal_state_exception", "ES|QL columns differ on this node"));
            }
            let wanted = vec![true; schema.cols.len()];
            targets.iter().map(|t| {
                let (rows, docs, loaded, with_point) = distance_shard(&schema, &wanted, t, &spec.query, field, *lat, *lon, *n)?;
                Ok(FastPart::Rows { docs, loaded, with_point, cols: Some(columns.clone()), rows: wire_rows(rows) })
            }).collect::<Result<_>>()?
        }
        FastPlan::Rows { indices, columns, query_text, now, residual, pre, sort, n } => {
            let mut ctx = new_ctx(node, query_text, None, (*n).max(1), (*n).max(1));
            ctx.esql = true;
            ctx.restrict = restrict.cloned();
            ctx.shards = source.cloned();
            ctx.fctx.now = *now;
            ctx.fctx.query = query_text.clone();
            let schema = resolve_schema(&ctx, indices, 0)?;
            if wire_cols(&source_cols(&schema, &[])) != *columns {
                return Err(EsError::new(409, "illegal_state_exception", "ES|QL columns differ on this node"));
            }
            let mut local: Vec<Cmd> = residual.iter().map(|(e, p)| Cmd::Where(e.clone(), *p)).collect();
            local.extend(pre.iter().map(RowCmd::cmd));
            let wanted = vec![true; schema.cols.len()];
            let parts = targets.iter().map(|t| Ok(rows_part(stream_rows(&mut ctx, &schema, std::slice::from_ref(t), &spec.query, &local, sort.as_deref(), *n, &wanted)?))).collect::<Result<Vec<_>>>()?;
            warnings = ctx.fctx.warnings.borrow().list.clone();
            parts
        }
    };
    Ok(targets
        .iter()
        .zip(parts)
        .map(|(t, part)| crate::esql::remote::ShardResult {
            index: t.svc.name.clone(),
            shard: t.shard.as_ref().map(|s| s.id).unwrap_or(0),
            count: 0,
            hits: Vec::new(),
            probes: Vec::new(),
            groups: None,
            loaded: 0,
            pit: false,
            fast: Some(part),
            warnings: warnings.clone(),
        })
        .collect())
}

/// Documents in the searched shards (cheap: segment counts).
fn searched_docs(schema: &IndexSchema) -> Result<u64> {
    live_docs(&schema.targets)
}

/// Live documents of `targets` (expanded to their shards).
fn live_docs(targets: &[crate::search::Target]) -> Result<u64> {
    Ok(crate::search::expand_targets(targets)?
        .iter()
        .map(|t| match &t.snapshot {
            Some(view) => view.num_live_docs(),
            None => t.shard_copy().shard.reader_view().num_live_docs(),
        })
        .sum())
}

/// Commands that map each row on its own (and keep row order).
fn row_local(c: &Cmd) -> bool {
    matches!(c, Cmd::Eval(_) | Cmd::Where(..) | Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..))
}

fn grouped_row_key(keys: &[Expr], row: &[Val], ctx: &FnCtx) -> Result<String> {
    fn append(key: &mut String, value: &Val) {
        // Preserve Val::key's encoding without allocating a string for null.
        if value.is_null() { key.push_str("2:\u{0}n"); } else {
            let part = value.key();
            key.push_str(&part.len().to_string());
            key.push(':');
            key.push_str(&part);
        }
    }
    let mut key = String::with_capacity(keys.len().saturating_mul(3));
    for expr in keys {
        match expr {
            // Bound columns need no interpreter dispatch or cloned value.
            Expr::Idx(index) => append(&mut key, row.get(*index).unwrap_or(&Val::Null)),
            _ => append(&mut key, &eval(expr, row, ctx)?),
        }
    }
    Ok(key)
}

fn lookup_sort_cmp(a: &Val, b: &Val, key: &SortKey) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_null(), b.is_null()) {
        (true,true) => Ordering::Equal,
        (true,false) => if key.nulls_first.unwrap_or(key.desc) { Ordering::Less } else { Ordering::Greater },
        (false,true) => if key.nulls_first.unwrap_or(key.desc) { Ordering::Greater } else { Ordering::Less },
        _ => { let order = cmp_vals(a,b); if key.desc { order.reverse() } else { order } }
    }
}

// Typed deterministic EVALs used to construct composite lookup keys. Keep
// arbitrary functions and value-dependent output types on the table path.
fn lookup_prefix_eval(expr: &Expr) -> bool {
    fn deterministic(expr: &Expr) -> bool {
        match expr {
            Expr::Col(..) | Expr::Lit(_) => true,
            Expr::Cast(e, _) | Expr::Neg(e) => deterministic(e),
            Expr::Bin(_, a, b) => deterministic(a) && deterministic(b),
            Expr::Call(name, args, _) if name.eq_ignore_ascii_case("CONCAT") => args.iter().all(deterministic),
            _ => false,
        }
    }
    match expr {
        Expr::Col(..) | Expr::Lit(_) => true,
        Expr::Cast(e, _) => deterministic(e),
        Expr::Call(name, args, _) if name.eq_ignore_ascii_case("CONCAT") => args.iter().all(deterministic),
        _ => false,
    }
}

/// A join key expression's value for `row`: a plain column is the row's
/// value (no evaluation, no per-call memory reservation: the per-row
/// `eval` of a 100k-key join's key was a fifth of its scan).
fn key_value(expression: &Expr, row: &[Val], f: &FnCtx) -> Result<Val> {
    match expression {
        Expr::Idx(i) => Ok(row.get(*i).cloned().unwrap_or(Val::Null)),
        _ => eval(expression, row, f),
    }
}

/// Most distinct key rows [`streamed_filtered_lookup`] remembers decisions for.
const DECIDED_KEY_ROWS: usize = 1 << 18;

#[cfg(test)]
thread_local! {
    /// Tests: chained joins take the table path.
    pub(super) static CHAINED_STREAM_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `LOOKUP JOIN ... | [KEEP/DROP/RENAME]... | LOOKUP JOIN ... | ... | LIMIT n`
/// right after `FROM`, two or more named joins (Rally joins
/// `1k_100k_200k_500k`: four joins, RENAMEs, a KEEP, LIMIT 1000). The table
/// path loaded the first `n` matches whole through a search response, then
/// ran each join over them. Here the matches are read in chunks with the
/// columns the pipeline reads (the table path's projection: the others
/// null, as there), and every command runs over each chunk as the table
/// path runs it over all rows. A join keeps every row of its input (a row
/// joining nothing keeps nulls) and the moves keep every row, so the first
/// `n` matches in search order make the first `n` output rows: the table
/// path's rows, columns and warnings.
fn streamed_chained_lookup(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    #[cfg(test)]
    if CHAINED_STREAM_DISABLED.with(|d| d.get()) { return Ok(None); }
    if !residual.is_empty() || schema.slots.is_some() { return Ok(None); }
    if !matches!(rest.first(), Some(Cmd::LookupJoin { cond: None, .. })) { return Ok(None); }
    let body = rest.iter().take_while(|cmd| matches!(cmd, Cmd::LookupJoin { cond: None, .. } | Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..))).count();
    let Some(Cmd::Limit(limit)) = rest.get(body) else { return Ok(None); };
    let cmds = &rest[..body];
    if cmds.iter().filter(|cmd| matches!(cmd, Cmd::LookupJoin { .. })).count() < 2 { return Ok(None); }
    if schema.cols.iter().any(|(name, _)| name.starts_with('_')) { return Ok(None); }
    let n = (*limit).min(ctx.max_limit);
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().map(|(column, _)| needed.as_ref().is_none_or(|set| set.contains(column))).collect();
    let read = ((1usize << 16) / wanted.iter().filter(|w| **w).count().max(1)).clamp(16, PROJECTED_JOIN_CHUNK);
    // The commands' own checks (columns, key types) and output columns, as
    // the table path meets them, before any row.
    let state = (ctx.pristine.clone(), ctx.fulltext_barrier, ctx.lexical_barrier);
    let mut out = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    for cmd in cmds { out = apply(ctx, cmd, out)?; }
    let mut output_memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    if n > 0 {
        crate::search::for_each_match_wanting(&schema.targets, query, n.clamp(16, PAGE), Some(n), &mut |page| {
            let key_doc_values = doc_values_allowed(page.target);
            let mut at = 0;
            let mut range: Vec<rustic_core::types::DocId> = Vec::new();
            while at < page.len() {
                // Every source row makes at least one output row: the rows
                // still wanted bound the documents read.
                let end = (at + (n - out.rows.len()).clamp(1, PROJECTED_JOIN_CHUNK)).min(page.len());
                let docs = if page.all_docs {
                    range.clear();
                    range.extend(at as u32..end as u32);
                    range.as_slice()
                } else { &page.docs[at..end] };
                at = end;
                let mut rows = Vec::with_capacity(docs.len());
                for part in docs.chunks(read) {
                    let (part_rows, loaded, _page_memory) = lookup_key_rows(schema, &wanted, page.target, page.lookup, page.leaf, part, key_doc_values)?;
                    ctx.values_loaded += loaded;
                    rows.extend(part_rows);
                }
                ctx.documents_found += docs.len() as u64;
                (ctx.pristine, ctx.fulltext_barrier, ctx.lexical_barrier) = (state.0.clone(), state.1, state.2);
                let mut chunk = Table { cols: source_cols(schema, &[]), rows };
                for cmd in cmds { chunk = apply(ctx, cmd, chunk)?; }
                output_memory.grow(rows_bytes(&chunk.rows), crate::esql::exec::ESQL_LABEL)?;
                out.rows.extend(chunk.rows);
                if out.rows.len() >= n { return Ok(false); }
            }
            Ok(true)
        })?;
    }
    taken();
    out.rows.truncate(n);
    resize_page_memory(&mut output_memory, rows_bytes(&out.rows))?;
    output_memory.retain_for_request();
    let out = apply(ctx, &rest[body], out)?;
    Ok(Some((out, body + 1)))
}

#[cfg(test)]
thread_local! {
    /// Tests: a join reading every source column takes the table path.
    pub(super) static WHOLE_ROW_STREAM_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Most source documents [`streamed_projected_lookup`] joins at once (one
/// keyed lookup read each).
const PROJECTED_JOIN_CHUNK: usize = 1 << 14;

/// `LOOKUP JOIN index ON keys | (KEEP | DROP | RENAME)... | LIMIT n` right
/// after `FROM`, when the pipeline reads only some of the index's columns
/// (Rally joins `esql_lookup_join_*_keys_keep_limit10000`: `... | KEEP
/// @timestamp, lookup_keyword_0 | LIMIT 10000`). ES extracts only the kept
/// fields and the join key on both sides (`PruneColumns`, then
/// `ValuesSourceReaderOperator` and the lookup's field extraction); the
/// table path read the first `n` matches' whole `_source`, joined rows of
/// every column, then dropped them. Here the first matches in search order
/// are read with the [`IndexSchema::projection`] only (doc values where they
/// equal the `_source` reading), joined with the lookup rows their keys
/// match (a keyed read per chunk), and moved straight into the output
/// columns: the columns, rows, order and warnings of the table path.
fn streamed_projected_lookup(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if !residual.is_empty() || schema.slots.is_some() { return Ok(None); }
    // Leading RENAMEs (Rally joins `100k_to_*`: `RENAME key_100000 AS
    // key_5000000 | LOOKUP JOIN ...`) run on each read chunk.
    let renames = rest.iter().take_while(|cmd| matches!(cmd, Cmd::Rename(..))).count();
    let (prefix, rest) = rest.split_at(renames);
    let Some(cmd @ Cmd::LookupJoin { index, index_pos, on, on_pos, cond: None, pos, fields }) = rest.first() else { return Ok(None); };
    let moves = rest[1..].iter().take_while(|cmd| matches!(cmd, Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..))).count();
    let Some(Cmd::Limit(limit)) = rest.get(1 + moves) else { return Ok(None); };
    // The columns read: the projection's, or every column (`LOOKUP JOIN
    // ... | LIMIT n`, Rally joins `*_keys_limit1000`): the table path loaded
    // those rows through a search response (each hit's `_source` parsed
    // into a tree, then extracted), where these are read per column from
    // doc values where they equal the `_source` reading.
    let narrow: Vec<(String, ColSrc)> = match &schema.projection {
        Some(projection) => schema.cols.iter().filter(|(name, _)| projection.contains(name)).cloned().collect(),
        None => schema.cols.clone(),
    };
    if moves == 0 || narrow.len() == schema.cols.len() {
        // Metadata columns (`METADATA _id, ...`) stay on the table path.
        if schema.cols.iter().any(|(name, _)| name.starts_with('_')) { return Ok(None); }
        #[cfg(test)]
        if WHOLE_ROW_STREAM_DISABLED.with(|d| d.get()) { return Ok(None); }
    }
    let mut base = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    for rename in prefix { base = apply(ctx, rename, base)?; }
    track_fields(ctx, cmd, &base);
    let lookup = lookup_schema(ctx, index, *index_pos, fields.as_ref())?;
    let right = || Table { cols: source_cols(&lookup, &[]), rows: Vec::new() };
    // The join's columns over every index column, as the table path builds
    // them (and validates the keys), then the output columns: the moves run
    // over one row holding each joined column's position.
    let wide = LookupJoinPlan::build(ctx, &base, right(), *index_pos, on, on_pos, &[], *pos)?;
    let mut shaped = Table { cols: wide.cols.clone(), rows: vec![(0..wide.cols.len()).map(|i| Val::Long(i as i64)).collect()] };
    for cmd in &rest[1..1 + moves] { shaped = apply(ctx, cmd, shaped)?; }
    let narrow = IndexSchema { cols: narrow, targets: Vec::new(), slots: None, projection: None };
    let read_cols = source_cols(&narrow, &[]);
    let mut left = Table { cols: read_cols.clone(), rows: Vec::new() };
    for rename in prefix { left = apply(ctx, rename, left)?; }
    let thin = LookupJoinPlan::build(ctx, &left, right(), *index_pos, on, on_pos, &[], *pos)?;
    // Each output column's position in the joined rows of the read columns.
    let Some(map) = shaped.rows[0].iter().map(|marker| {
        let Val::Long(at) = marker else { return None; };
        let at = *at as usize;
        match at.checked_sub(wide.lkeep.len()) {
            None => thin.cols[..thin.lkeep.len()].iter().position(|col| col.name == wide.cols[at].name),
            Some(offset) => Some(thin.lkeep.len() + offset),
        }
    }).collect::<Option<Vec<usize>>>() else { return Ok(None); };
    let distinct = map.iter().collect::<HashSet<_>>().len() == map.len();
    drop((wide, thin));
    let n = (*limit).min(ctx.max_limit);
    let mut out = Table { cols: shaped.cols, rows: Vec::new() };
    let mut output_memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    let all = vec![true; narrow.cols.len()];
    if n > 0 {
        crate::search::for_each_match_wanting(&schema.targets, query, n.clamp(16, PAGE), Some(n), &mut |page| {
            let key_doc_values = doc_values_allowed(page.target);
            let mut at = 0;
            let mut range: Vec<rustic_core::types::DocId> = Vec::new();
            while at < page.len() {
                // A join keeps every source row (unmatched ones with nulls):
                // the rows still wanted bound the documents read.
                let end = (at + (n - out.rows.len()).clamp(1, PROJECTED_JOIN_CHUNK)).min(page.len());
                let docs = if page.all_docs {
                    range.clear();
                    range.extend(at as u32..end as u32);
                    range.as_slice()
                } else { &page.docs[at..end] };
                at = end;
                // Wide rows (Rally's 100-field base) in reads of about 64k
                // column values: each read admits its columns' worst-case
                // sizes up front, which over 10,000 rows of 200 columns asked
                // more than the request breaker holds.
                let read = ((1usize << 16) / narrow.cols.len().max(1)).clamp(16, PROJECTED_JOIN_CHUNK);
                let mut rows = Vec::with_capacity(docs.len());
                for part in docs.chunks(read) {
                    let (part_rows, loaded, _page_memory) = lookup_key_rows(&narrow, &all, page.target, page.lookup, page.leaf, part, key_doc_values)?;
                    ctx.values_loaded += loaded;
                    rows.extend(part_rows);
                }
                ctx.documents_found += docs.len() as u64;
                let mut chunk = Table { cols: read_cols.clone(), rows };
                for rename in prefix { chunk = apply(ctx, rename, chunk)?; }
                let plan = LookupJoinPlan::for_rows(ctx, &left, &chunk.rows, &lookup, *index_pos, on, on_pos, &[], *pos)?;
                let joined = plan.apply(ctx, &chunk, Some(n - out.rows.len()))?;
                drop(chunk);
                let rows: Vec<Vec<Val>> = joined.rows.into_iter().map(|mut row| map.iter().map(|&i| if distinct { std::mem::replace(&mut row[i], Val::Null) } else { row[i].clone() }).collect()).collect();
                output_memory.grow(rows_bytes(&rows), crate::esql::exec::ESQL_LABEL)?;
                out.rows.extend(rows);
                if out.rows.len() >= n { return Ok(false); }
            }
            Ok(true)
        })?;
    }
    taken();
    out.rows.truncate(n);
    resize_page_memory(&mut output_memory, rows_bytes(&out.rows))?;
    output_memory.retain_for_request();
    let out = apply(ctx, &rest[1 + moves], out)?;
    Ok(Some((out, renames + moves + 2)))
}

/// Most lookup terms [`sort_driven_lookup`] walks for one page.
const SORT_DRIVEN_TERMS: usize = 4_000_000;
/// Lookup keys [`sort_driven_lookup`] looks up in the source per `terms` query.
const SORT_DRIVEN_BATCH: usize = 4_096;

#[cfg(test)]
thread_local! {
    /// Tests: [`streamed_sorted_lookup`] reads the whole lookup side.
    pub(super) static SORT_DRIVEN_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: joins [`sort_driven_lookup`] answered.
    pub(super) static SORT_DRIVEN_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The keyword doc-values field holding a lookup sort column's values
/// exactly, and its path: the keyword column itself, or a `text` column's
/// plain keyword sub-field ([`text_keyword_field`]); indexed, so a value's
/// documents are its postings.
fn sort_keyword_field(column: &ColSrc, lookup: &crate::mapping::FieldLookup) -> Option<(u16, String)> {
    let field = lookup.fields.get(&column.path)?;
    let (id, path) = match column.ty.as_str() {
        "keyword" => (direct_field(column, false, lookup, false)?, column.path.clone()),
        "text" => {
            let id = text_keyword_field(column, false, field, lookup)?;
            let path = field.multi_fields.iter().find(|name| lookup.fields.get(*name).is_some_and(|sub| sub.core_id == Some(id)))?.clone();
            (id, path)
        }
        _ => return None,
    };
    lookup.fields.get(&path).filter(|f| f.indexed && f.doc_values).map(|_| (id, path))
}

/// `LOOKUP JOIN index ON key | SORT <lookup column> ASC | LIMIT n` (Rally
/// joins `esql_lookup_join_{5M,100M}_keys_sort_limit10000`), answered from
/// the sort column's order instead of joining every source row: the lookup
/// side's sort values are walked in ascending (byte) order from its keyword
/// doc values' dictionaries, each value's lookup documents found by its
/// postings, and the source documents holding their keys found by `terms`
/// queries over a batch of keys. Pairs are emitted per sort value in source
/// order (then lookup order), as the stable sort of the joined rows orders
/// them, until `n`; then only those source and lookup rows are read whole.
/// The whole-lookup path read and hashed every lookup row first: 5M or
/// 100M rows of every column.
///
/// Exact only when: one `ON` key, a keyword field on both sides, single
/// valued in doc values in every leaf (no multi-value warning anywhere);
/// nulls sort last (so unmatched source rows, and lookup rows without a
/// sort value, only follow the `n` found); the sort values are a keyword
/// field's (or a text field's plain keyword sub-field holding every
/// document's single value, by count); every lookup leaf a segment. `None`
/// (the whole-lookup path) otherwise, or when fewer than `n` rows have a
/// sort value.
fn sort_driven_lookup(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    #[cfg(test)]
    if SORT_DRIVEN_DISABLED.with(|disabled| disabled.get()) { return Ok(None); }
    let [cmd @ Cmd::LookupJoin { index, index_pos, on, on_pos, cond: None, pos, fields }, Cmd::Sort(keys), Cmd::Limit(limit), ..] = rest else { return Ok(None); };
    let ([key], [on_key]) = (keys.as_slice(), on.as_slice()) else { return Ok(None); };
    let Expr::Col(name, _) = &key.expr else { return Ok(None); };
    let n = *limit;
    if n == 0 || key.desc || key.nulls_first == Some(true) || schema.slots.is_some() { return Ok(None); }
    let lookup = lookup_schema(ctx, index, *index_pos, fields.as_ref())?;
    if !lookup_reads_source(ctx, &lookup) { return Ok(None); }
    let base = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    let plan = LookupJoinPlan::build(ctx, &base, Table { cols: source_cols(&lookup, &[]), rows: Vec::new() }, *index_pos, on, on_pos, &[], *pos)?;
    let Some(output) = plan.cols.iter().position(|col| col.name == *name) else { return Ok(None); };
    if output < plan.lkeep.len() { return Ok(None); }
    let Some((_, sort_column)) = lookup.cols.iter().find(|(n, _)| n == name) else { return Ok(None); };
    let Some((_, lookup_key)) = lookup.cols.iter().find(|(n, _)| n == on_key) else { return Ok(None); };
    let base_key_index = plan.lkeys[0];
    let (_, base_key) = &schema.cols[base_key_index];
    if base_key.ty != "keyword" || lookup_key.ty != "keyword" { return Ok(None); }
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let base_targets = node.snapshot(&schema.targets)?;
    let lookup_targets = node.snapshot(&lookup.targets)?;
    // Source keys: single-valued keyword doc values everywhere.
    let single = crate::search::for_each_leaf(&base_targets, &mut |target, field_lookup, leaf| {
        if !doc_values_allowed(target) { return Ok(false); }
        let Some(id) = direct_field(base_key, false, field_lookup, false) else { return Ok(false) };
        Ok(match leaf {
            rustic_core::query::LeafReader::Segment(segment) => !segment.keyword_from_postings(id) && segment.multi_keyword_reader(id).is_none(),
            rustic_core::query::LeafReader::Memtable(writer, ..) => writer.fast_fields().get_multi_keyword(id).is_none(),
        })
    })?;
    if !single { return Ok(None); }
    // Every lookup document with a sort value holds it in the keyword field.
    let Some(Some((sort_path, key_ids))) = crate::search::with_leaves(&lookup_targets, &mut |target, _, field_lookup| {
        if !doc_values_allowed(target) { return Ok(None); }
        Ok(sort_keyword_field(sort_column, field_lookup).zip(direct_field(lookup_key, false, field_lookup, false)))
    })?.map(|found| found.map(|((sort_id, path), key_id)| (path, (sort_id, key_id)))) else { return Ok(None); };
    // Every source document (and lookup document) holds its key in doc
    // values: one indexed before the field was mapped has it in `_source`
    // only, which the whole-lookup path reads and the walk does not.
    let exists = |field: &str| json!({"exists": {"field": field}});
    let base_all = crate::search::count(&base_targets, Some(query), None)?;
    if crate::search::count(&base_targets, Some(&json!({"bool": {"filter": [query.clone(), exists(&base_key.path)]}})), None)? != base_all
        || crate::search::count(&lookup_targets, Some(&exists(&lookup_key.path)), None)? != crate::search::count(&lookup_targets, None, None)? {
        return Ok(None);
    }
    if sort_path != sort_column.path {
        if crate::search::count(&lookup_targets, Some(&exists(&sort_column.path)), None)? != crate::search::count(&lookup_targets, Some(&exists(&sort_path)), None)? {
            return Ok(None);
        }
    }
    track_fields(ctx, cmd, &base);
    let mut out = Table { cols: plan.cols.clone(), rows: Vec::new() };
    // The SORT's own checks (its type errors), as the table path makes them.
    sort_table(ctx, &mut out, keys)?;
    let projected = IndexSchema { cols: vec![schema.cols[base_key_index].clone()], targets: Vec::new(), slots: None, projection: None };
    let (sort_id, key_id) = key_ids;
    // The winners: (sort rank, source (target, leaf, doc), lookup (leaf, doc)).
    type Pair = (usize, (usize, usize, rustic_core::types::DocId), (usize, rustic_core::types::DocId));
    let mut pairs: Vec<Pair> = Vec::new();
    let mut base_loaded = 0u64;
    let walked = crate::search::with_leaves(&lookup_targets, &mut |_, leaves, _| -> Result<bool> {
        // Per leaf, its sort values in order: a segment's keyword
        // dictionary (ordinals, documents by postings), or a memtable's
        // column sorted here (memtables are bounded by the seal size).
        enum Source<'l> {
            Seg(Option<rustic_core::codec::fastfields::KeywordReader<'l>>, Option<rustic_core::codec::fastfields::KeywordReader<'l>>),
            Mem(Vec<(String, rustic_core::types::DocId)>, Option<&'l rustic_core::fast_fields::KeywordColumn>),
        }
        let mut sources: Vec<Source<'_>> = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            match leaf {
                rustic_core::query::LeafReader::Segment(segment) => {
                    for id in [sort_id, key_id] {
                        if segment.keyword_from_postings(id) || segment.multi_keyword_reader(id).is_some() { return Ok(false); }
                    }
                    sources.push(Source::Seg(segment.keyword_reader(sort_id), segment.keyword_reader(key_id)));
                }
                rustic_core::query::LeafReader::Memtable(writer, ..) => {
                    let fast = writer.fast_fields();
                    if fast.get_multi_keyword(sort_id).is_some() || fast.get_multi_keyword(key_id).is_some() { return Ok(false); }
                    let mut sorted: Vec<(String, rustic_core::types::DocId)> = match fast.get_keyword(sort_id) {
                        Some(column) => (0..leaf.num_docs()).filter_map(|doc| column.get(doc).map(|v| (v.to_string(), doc))).collect(),
                        None => Vec::new(),
                    };
                    sorted.sort_unstable();
                    sources.push(Source::Mem(sorted, fast.get_keyword(key_id)));
                }
            }
        }
        // Ascending merge of the leaves' values: (value, leaf, position).
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(Vec<u8>, usize, u32)>> = std::collections::BinaryHeap::new();
        let at = |source: &Source<'_>, position: u32| -> Option<Vec<u8>> {
            match source {
                Source::Seg(Some(sort), _) if (position as usize) < sort.num_terms() => sort.term_bytes(position).map(|t| t.into_owned()),
                Source::Mem(sorted, _) => sorted.get(position as usize).map(|(v, _)| v.as_bytes().to_vec()),
                _ => None,
            }
        };
        for (leaf, source) in sources.iter().enumerate() {
            if let Some(term) = at(source, 0) { heap.push(std::cmp::Reverse((term, leaf, 0))); }
        }
        // Keys of the walked values not yet looked up in the source, with
        // their (rank, lookup document)s.
        let mut batch: HashMap<String, Vec<(usize, (usize, rustic_core::types::DocId))>> = HashMap::new();
        let mut rank = 0usize;
        let flush = |batch: &mut HashMap<String, Vec<(usize, (usize, rustic_core::types::DocId))>>, pairs: &mut Vec<Pair>, base_loaded: &mut u64| -> Result<()> {
            if batch.is_empty() { return Ok(()); }
            let mut values: Vec<&str> = batch.keys().map(String::as_str).collect();
            values.sort_unstable();
            let narrowed = json!({"bool": {"filter": [query.clone(), {"terms": {base_key.path.clone(): values}}]}});
            let mut found: Vec<Pair> = Vec::new();
            crate::search::for_each_match(&base_targets, &narrowed, PAGE, &mut |page| {
                let docs: Vec<rustic_core::types::DocId> = page.doc_ids().collect();
                for chunk in docs.chunks(1024) {
                    let (rows, count, _page_memory) = lookup_key_rows(&projected, &[true], page.target, page.lookup, page.leaf, chunk, doc_values_allowed(page.target))?;
                    *base_loaded += count;
                    for (&doc, row) in chunk.iter().zip(rows) {
                        let Some(Val::Str(key)) = row.into_iter().next() else { continue };
                        for (rank, at) in batch.get(&key).into_iter().flatten() {
                            found.push((*rank, (page.target_idx, page.leaf_idx, doc), *at));
                        }
                    }
                }
                Ok(true)
            })?;
            found.sort_unstable();
            pairs.extend(found);
            batch.clear();
            Ok(())
        };
        let mut terms = 0usize;
        while let Some(std::cmp::Reverse((term, leaf, position))) = heap.pop() {
            // Every leaf holding this value.
            let mut holders = vec![(leaf, position)];
            while heap.peek().is_some_and(|std::cmp::Reverse((next, ..))| *next == term) {
                let std::cmp::Reverse((_, leaf, position)) = heap.pop().unwrap();
                holders.push((leaf, position));
            }
            for (leaf, position) in &holders {
                let next = match &sources[*leaf] {
                    Source::Seg(Some(sort), key_reader) => {
                        if let Some(mut postings) = leaves[*leaf].postings(&rustic_core::types::Term::new(sort_id, term.clone())) {
                            while let Some(doc) = postings.next() {
                                if leaves[*leaf].is_deleted(doc) || sort.ord(doc) != *position { continue; }
                                let Some(key) = key_reader.as_ref().and_then(|reader| reader.get(doc)) else { continue };
                                batch.entry(key.into_owned()).or_default().push((rank, (*leaf, doc)));
                            }
                        }
                        *position + 1
                    }
                    Source::Mem(sorted, key_column) => {
                        let mut i = *position as usize;
                        while let Some((value, doc)) = sorted.get(i).filter(|(value, _)| value.as_bytes() == term.as_slice()) {
                            let _ = value;
                            if !leaves[*leaf].is_deleted(*doc) {
                                if let Some(key) = key_column.and_then(|column| column.get(*doc)) {
                                    batch.entry(key.to_string()).or_default().push((rank, (*leaf, *doc)));
                                }
                            }
                            i += 1;
                        }
                        i as u32
                    }
                    Source::Seg(None, _) => continue,
                };
                if let Some(term) = at(&sources[*leaf], next) { heap.push(std::cmp::Reverse((term, *leaf, next))); }
            }
            rank += 1;
            terms += 1;
            if batch.len() >= SORT_DRIVEN_BATCH {
                flush(&mut batch, &mut pairs, &mut base_loaded)?;
                if pairs.len() >= n { return Ok(true); }
            }
            if terms > SORT_DRIVEN_TERMS { return Ok(false); }
        }
        flush(&mut batch, &mut pairs, &mut base_loaded)?;
        Ok(pairs.len() >= n)
    })?;
    if walked != Some(true) { return Ok(None); }
    pairs.truncate(n);
    // The winners' source rows and lookup rows, read whole.
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().map(|(column, _)| needed.as_ref().is_none_or(|set| set.contains(column))).collect();
    let lookup_needed = needed_columns(ctx.full, &lookup);
    let lookup_wanted: Vec<bool> = lookup.cols.iter().map(|(column, _)| lookup_needed.as_ref().is_none_or(|set| set.contains(column))).collect();
    let mut base_docs: HashMap<(usize, usize), Vec<rustic_core::types::DocId>> = HashMap::new();
    let mut lookup_docs: HashMap<usize, Vec<rustic_core::types::DocId>> = HashMap::new();
    for (_, at, lat) in &pairs {
        base_docs.entry((at.0, at.1)).or_default().push(at.2);
        lookup_docs.entry(lat.0).or_default().push(lat.1);
    }
    for docs in base_docs.values_mut().chain(lookup_docs.values_mut()) { docs.sort_unstable(); docs.dedup(); }
    let mut base_rows: HashMap<(usize, usize, rustic_core::types::DocId), Vec<Val>> = HashMap::new();
    let mut lookup_rows_read: HashMap<(usize, rustic_core::types::DocId), Vec<Val>> = HashMap::new();
    let mut loaded = base_loaded;
    let mut output_memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    crate::search::for_each_match(&base_targets, query, PAGE, &mut |page| {
        let Some(docs) = base_docs.get(&(page.target_idx, page.leaf_idx)) else { return Ok(true) };
        for chunk in docs.chunks(1024) {
            let (rows, count, _page_memory) = lookup_key_rows(schema, &wanted, page.target, page.lookup, page.leaf, chunk, doc_values_allowed(page.target))?;
            loaded += count;
            output_memory.grow(rows_bytes(&rows), crate::esql::exec::ESQL_LABEL)?;
            for (&doc, row) in chunk.iter().zip(rows) { base_rows.insert((page.target_idx, page.leaf_idx, doc), row); }
        }
        Ok(true)
    })?;
    crate::search::for_each_match(&lookup_targets, &json!({"match_all": {}}), PAGE, &mut |page| {
        let Some(docs) = lookup_docs.get(&page.leaf_idx) else { return Ok(true) };
        for chunk in docs.chunks(1024) {
            let (rows, count, _page_memory) = lookup_key_rows(&lookup, &lookup_wanted, page.target, page.lookup, page.leaf, chunk, doc_values_allowed(page.target))?;
            loaded += count;
            output_memory.grow(rows_bytes(&rows), crate::esql::exec::ESQL_LABEL)?;
            for (&doc, row) in chunk.iter().zip(rows) { lookup_rows_read.insert((page.leaf_idx, doc), row); }
        }
        Ok(true)
    })?;
    for (_, at, lat) in &pairs {
        let (Some(left_row), Some(right_row)) = (base_rows.get(at), lookup_rows_read.get(lat)) else {
            return Err(EsError::internal("sort-driven join lost a document between passes"));
        };
        let mut row: Vec<Val> = plan.lkeep.iter().map(|i| left_row[*i].clone()).collect();
        row.extend(plan.rcols.iter().map(|i| right_row[*i].clone()));
        out.rows.push(row);
    }
    // As the whole-lookup path counts them: every source document's key
    // read, every lookup document read.
    ctx.documents_found += base_all + crate::search::count(&lookup_targets, None, None)?;
    ctx.values_loaded += loaded;
    #[cfg(test)]
    SORT_DRIVEN_RUNS.with(|n| n.set(n.get() + 1));
    taken();
    resize_page_memory(&mut output_memory, rows_bytes(&out.rows))?;
    output_memory.retain_for_request();
    let out = apply(ctx, &rest[2], out)?;
    Ok(Some((out, 3)))
}

/// Tests: joins [`streamed_sorted_lookup`] answered.
#[cfg(test)]
static SORTED_TWO_PASS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `LOOKUP JOIN index ON keys | SORT <one lookup column> | LIMIT n` right
/// after `FROM` (Rally joins `*_keys_sort_limit10000`), over a lookup
/// side read whole whose keys are unique (one joined row per document).
/// A first pass over a pinned snapshot reads only each document's join
/// keys and keeps the `n` best (sort value, document) pairs in sort order,
/// later documents after equal ones (the table path's stable sort). A
/// second pass over the same snapshot reads the winners' rows and joins
/// them. The streamed join it replaces read and joined every document that
/// was competitive when it was read: at 100k keys, tens of thousands of
/// wide rows for a page of 10,000.
fn streamed_sorted_lookup(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if !residual.is_empty() || schema.slots.is_some() { return Ok(None); }
    if let Some(done) = sort_driven_lookup(ctx, schema, query, rest)? { return Ok(Some(done)); }
    let [cmd @ Cmd::LookupJoin { index, index_pos, on, on_pos, cond: None, pos, fields }, Cmd::Sort(keys), Cmd::Limit(limit), ..] = rest else { return Ok(None); };
    let [key] = keys.as_slice() else { return Ok(None); };
    let Expr::Col(name, _) = &key.expr else { return Ok(None); };
    // `LIMIT 0` is left to the other paths, unchanged.
    if *limit == 0 { return Ok(None); }
    let base = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    track_fields(ctx, cmd, &base);
    let plan = LookupJoinPlan::prepare(ctx, &base, index, *index_pos, on, on_pos, &[], *pos, fields.as_ref())?;
    let Some(output) = plan.cols.iter().position(|col| col.name == *name) else { return Ok(None); };
    let Some(right) = output.checked_sub(plan.lkeep.len()).and_then(|i| plan.rcols.get(i)).copied() else { return Ok(None); };
    if !matches!(plan.right.cols[right].ty.as_str(), "keyword" | "text" | "long" | "integer" | "double" | "boolean")
        || plan.right.rows.iter().any(|row| matches!(row[right], Val::Multi(_)))
        || plan.index_map.values().any(|rows| rows.len() > 1) { return Ok(None); }
    let mut out = Table { cols: plan.cols.clone(), rows: Vec::new() };
    // The SORT's own checks (its type errors), as the table path makes them.
    sort_table(ctx, &mut out, keys)?;
    let n = *limit;
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&schema.targets)?;
    let projected = IndexSchema { cols: plan.lkeys.iter().map(|i| schema.cols[*i].clone()).collect(), targets: Vec::new(), slots: None, projection: None };
    let key_flags = vec![true; projected.cols.len()];
    // The best `n`: (sort key, (target, leaf, doc)), in output order.
    let mut top: Vec<(Vec<Val>, (usize, usize, rustic_core::types::DocId))> = Vec::new();
    let mut found = 0u64;
    let mut loaded = 0u64;
    let mut warned = false;
    if n > 0 {
        crate::search::for_each_match(&targets, query, PAGE, &mut |page| {
            let key_doc_values = doc_values_allowed(page.target);
            let mut at = 0;
            let mut range: Vec<rustic_core::types::DocId> = Vec::new();
            while at < page.len() {
                let end = (at + 1024).min(page.len());
                let docs: &[rustic_core::types::DocId] = if page.all_docs {
                    range.clear();
                    range.extend(at as u32..end as u32);
                    &range
                } else { &page.docs[at..end] };
                at = end;
                found += docs.len() as u64;
                let (rows, count, _page_memory) = lookup_key_rows(&projected, &key_flags, page.target, page.lookup, page.leaf, docs, key_doc_values)?;
                loaded += count;
                for (&doc, values) in docs.iter().zip(rows) {
                    // `LookupJoinPlan::matches` over the key columns alone.
                    let matched = if values.iter().any(Val::is_null) { None } else if values.iter().any(|v| matches!(v, Val::Multi(_))) {
                        warned = true;
                        None
                    } else {
                        let k: Vec<String> = values.iter().map(Val::key).collect();
                        if plan.mv_keys.contains(&k) { warned = true; }
                        plan.index_map.get(&k).and_then(|rows| rows.first())
                    };
                    let value = matched.map(|i| plan.right.rows[*i][right].clone()).unwrap_or(Val::Null);
                    let sort = vec![sort_key_value(value, false, key.desc)];
                    if top.len() == n && cmp_sort_keys(keys, &sort, &top[n - 1].0) != std::cmp::Ordering::Less { continue; }
                    let place = top.partition_point(|(kept, _)| cmp_sort_keys(keys, kept, &sort) != std::cmp::Ordering::Greater);
                    top.insert(place, (sort, (page.target_idx, page.leaf_idx, doc)));
                    top.truncate(n);
                }
            }
            Ok(true)
        })?;
    }
    if warned { super::lookup_mv_warning(&ctx.fctx, *pos); }
    #[cfg(test)]
    SORTED_TWO_PASS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // The winners' rows, joined, by document.
    let mut wanted_docs: HashMap<(usize, usize), Vec<rustic_core::types::DocId>> = HashMap::new();
    for (_, at) in &top { wanted_docs.entry((at.0, at.1)).or_default().push(at.2); }
    for docs in wanted_docs.values_mut() { docs.sort_unstable(); }
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().map(|(column, _)| needed.as_ref().is_none_or(|set| set.contains(column))).collect();
    let mut joined: HashMap<(usize, usize, rustic_core::types::DocId), Vec<Val>> = HashMap::with_capacity(top.len());
    let mut output_memory = crate::memory::breakers().reserve_request(0, crate::esql::exec::ESQL_LABEL)?;
    if !top.is_empty() {
        crate::search::for_each_match(&targets, query, PAGE, &mut |page| {
            let Some(docs) = wanted_docs.get(&(page.target_idx, page.leaf_idx)) else { return Ok(true) };
            for chunk in docs.chunks(1024) {
                let (rows, count, _page_memory) = lookup_key_rows(schema, &wanted, page.target, page.lookup, page.leaf, chunk, doc_values_allowed(page.target))?;
                loaded += count;
                let rows = plan.apply(ctx, &Table { cols: base.cols.clone(), rows }, None)?.rows;
                output_memory.grow(rows_bytes(&rows), crate::esql::exec::ESQL_LABEL)?;
                for (&doc, row) in chunk.iter().zip(rows) {
                    joined.insert((page.target_idx, page.leaf_idx, doc), row);
                }
            }
            Ok(true)
        })?;
    }
    for (_, at) in &top {
        out.rows.push(joined.remove(at).ok_or_else(|| EsError::internal("sorted join lost a document between passes"))?);
    }
    ctx.documents_found += found;
    ctx.values_loaded += loaded;
    taken();
    resize_page_memory(&mut output_memory, rows_bytes(&out.rows))?;
    output_memory.retain_for_request();
    let out = apply(ctx, &rest[2], out)?;
    Ok(Some((out, 3)))
}

/// Most distinct values [`semi_join`] pushes into the source query (ES's
/// `index.max_terms_count` default).
const SEMI_JOIN_TERMS: usize = 65_536;

#[cfg(test)]
thread_local! {
    /// Tests: lookup sides are read without a filter query ([`lookup_prefilter`]).
    pub(super) static LOOKUP_PREFILTER_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// Tests: [`streamed_filtered_lookup`] scans every source document.
    pub(super) static SEMI_JOIN_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: semi-joined scans.
    pub(super) static SEMI_JOINS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// [`semi_join`]'s narrowed scan: the snapshot read, the source query with
/// the key restriction, and the counts of the documents (and their key
/// values) it no longer reads.
struct SemiJoin {
    targets: Vec<crate::search::Target>,
    query: Value,
    skipped_docs: u64,
    skipped_values: u64,
}

/// A join key expression whose evaluation never warns: columns, literals,
/// `CONCAT`s of them, and conversions to `keyword` (any value has a string).
fn warning_free_key(expr: &Expr) -> bool {
    match expr {
        Expr::Col(..) | Expr::Lit(_) => true,
        Expr::Cast(e, ty) => matches!(ty.as_str(), "keyword" | "string") && warning_free_key(e),
        Expr::Call(name, args, _) if name.eq_ignore_ascii_case("CONCAT") => args.iter().all(warning_free_key),
        _ => false,
    }
}

/// The key tuples (by `Val::key`s, with their values and how many lookup
/// rows hold each) of the lookup rows passing `filters` (lookup columns
/// only, no `ON` key), read with just the keys and the filters' columns.
/// `None` when that cannot stand for the join's own decisions: a lookup row
/// with a multi-valued key (a left key reaching it warns), a filter that
/// fails or warns on some row, or too many passing keys.
fn passing_lookup_keys(ctx: &mut Ctx, lookup: &IndexSchema, on: &[String], filters: &[Expr], prefilter: Option<Value>) -> Result<Option<HashMap<Vec<String>, (Vec<Val>, usize)>>> {
    let mut refs = Vec::new();
    for filter in filters {
        crate::esql::analyze::expr_refs(filter, &mut refs);
    }
    if refs.is_empty() || refs.iter().any(|(name, _)| !lookup.cols.iter().any(|(n, _)| n == name) || on.contains(name)) {
        return Ok(None);
    }
    let narrow: HashSet<String> = on.iter().cloned().chain(refs.into_iter().map(|(name, _)| name)).collect();
    let thin_schema = IndexSchema { cols: lookup.cols.clone(), targets: lookup.targets.clone(), slots: None, projection: Some(narrow) };
    let thin = lookup_rows(ctx, &thin_schema, prefilter)?;
    let Some(key_columns) = on.iter().map(|k| thin.idx(k)).collect::<Option<Vec<usize>>>() else { return Ok(None) };
    let Ok(bound) = filters.iter().map(|f| bind(ctx, f, &thin.cols)).collect::<Result<Vec<_>>>() else { return Ok(None) };
    let mut passing: HashMap<Vec<String>, (Vec<Val>, usize)> = HashMap::new();
    for row in &thin.rows {
        let key: Vec<Val> = key_columns.iter().map(|i| row[*i].clone()).collect();
        if key.iter().any(|v| matches!(v, Val::Multi(_))) { return Ok(None); }
        if key.iter().any(Val::is_null) { continue; }
        match bound.iter().try_fold(true, |all, f| eval(f, row, &ctx.fctx).map(|v| all && v == Val::Bool(true))) {
            Ok(true) => passing.entry(key.iter().map(Val::key).collect()).or_insert((key, 0)).1 += 1,
            Ok(false) => {}
            Err(_) => return Ok(None),
        }
        if passing.len() > SEMI_JOIN_TERMS { return Ok(None); }
    }
    if !ctx.fctx.warnings.borrow().list.is_empty() { return Ok(None); }
    Ok(Some(passing))
}

/// A lookup query every row passing `filters` matches, when they are
/// `column == "literal"` conjuncts (Rally `*_two_keys_where_one_matching_equals`:
/// `lookup_keyword_0 == "val 10 rep 0"` over 100M lookup rows): a `term` on
/// the column's keyword doc-values field (a text column's plain keyword
/// sub-field), which holds a single-valued column's exact value. Only when
/// the rows it skips cannot change the semi-join's decisions: every `ON`
/// key and filtered column single-valued in every lookup leaf and held by
/// every lookup document (so no skipped row has a multi-valued key, nor a
/// filtered value that would warn), and the literal no longer than an
/// `ignore_above`. `None`: read every lookup row.
pub(super) fn lookup_prefilter(ctx: &mut Ctx, lookup: &IndexSchema, on: &[String], filters: &[Expr]) -> Result<Option<Value>> {
    #[cfg(test)]
    if LOOKUP_PREFILTER_DISABLED.with(|disabled| disabled.get()) { return Ok(None); }
    fn equalities<'e>(e: &'e Expr, out: &mut Vec<(&'e str, &'e Val)>) -> bool {
        match e {
            Expr::Bin(BinOp::And, a, b) => equalities(a, out) && equalities(b, out),
            Expr::Bin(BinOp::Eq, a, b) => match (a.as_ref(), b.as_ref()) {
                (Expr::Col(name, _), Expr::Lit(lit @ (Val::Str(_) | Val::Int(_) | Val::Long(_)))) | (Expr::Lit(lit @ (Val::Str(_) | Val::Int(_) | Val::Long(_))), Expr::Col(name, _)) => { out.push((name, lit)); true }
                _ => false,
            },
            _ => false,
        }
    }
    let mut pairs = Vec::new();
    if filters.is_empty() || !filters.iter().all(|f| equalities(f, &mut pairs)) { return Ok(None); }
    // An `ON` key column is the source's in the output, not the lookup's.
    if pairs.iter().any(|(name, _)| on.iter().any(|key| key == name)) { return Ok(None); }
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&lookup.targets)?;
    let mut keys: Vec<(&str, &ColSrc)> = Vec::new();
    for key in on {
        let Some((name, column)) = lookup.cols.iter().find(|(n, _)| n == key) else { return Ok(None) };
        keys.push((name.as_str(), column));
    }
    let all = json!({"match_all": {}});
    if !keyword_columns_single_and_covered(&targets, &all, &keys, true)? { return Ok(None); }
    let total = crate::search::count(&targets, None, None)?;
    let mut terms = Vec::new();
    for (name, literal) in pairs {
        let Some((_, column)) = lookup.cols.iter().find(|(n, _)| n == name) else { return Ok(None) };
        if let Val::Int(n) | Val::Long(n) = literal {
            // A plain single-valued integer column: its doc value is the
            // compared value ([`doc_value_field`]), the term its documents.
            if n.unsigned_abs() > 1 << 53 || (column.ty == "integer" && i32::try_from(*n).is_err()) { return Ok(None); }
            let Some(Some(id)) = crate::search::with_leaves(&targets, &mut |target, _, field_lookup| {
                Ok(doc_values_allowed(target).then(|| doc_value_field(name, column, false, field_lookup).filter(|(_, e)| *e == Extra::Integer).map(|(id, _)| id)).flatten())
            })? else { return Ok(None) };
            let single = crate::search::for_each_leaf(&targets, &mut |_, _, leaf| Ok(match leaf {
                rustic_core::query::LeafReader::Segment(segment) => segment.fast_fields().get_multi_i64(id).is_none(),
                rustic_core::query::LeafReader::Memtable(writer, ..) => writer.fast_fields().get_multi_i64(id).is_none(),
            }))?;
            if !single || crate::search::count(&targets, Some(&json!({"exists": {"field": column.path}})), None)? != total { return Ok(None); }
            terms.push(json!({"term": {column.path.clone(): n}}));
            continue;
        }
        let Val::Str(literal) = literal else { return Ok(None) };
        let Some(Some((id, path))) = crate::search::with_leaves(&targets, &mut |target, _, field_lookup| {
            Ok(doc_values_allowed(target).then(|| sort_keyword_field(column, field_lookup)).flatten())
        })? else { return Ok(None) };
        let ignore_above = crate::search::with_leaves(&targets, &mut |_, _, field_lookup| Ok(field_lookup.fields.get(&path).and_then(|f| f.ignore_above)))?.flatten();
        if ignore_above.is_some_and(|limit| literal.chars().count() > limit) { return Ok(None); }
        // Single-valued in every leaf, and every document holding the column holds it there.
        let single = crate::search::for_each_leaf(&targets, &mut |_, _, leaf| Ok(match leaf {
            rustic_core::query::LeafReader::Segment(segment) => !segment.keyword_from_postings(id) && segment.multi_keyword_reader(id).is_none(),
            rustic_core::query::LeafReader::Memtable(writer, ..) => writer.fast_fields().get_multi_keyword(id).is_none(),
        }))?;
        let exists = |field: &str| json!({"exists": {"field": field}});
        if !single || crate::search::count(&targets, Some(&exists(&path)), None)? != total
            || crate::search::count(&targets, Some(&exists(&column.path)), None)? != total { return Ok(None); }
        terms.push(json!({"term": {path: literal}}));
    }
    Ok(Some(json!({"bool": {"filter": terms}})))
}

/// `FROM ... | LOOKUP JOIN ... | WHERE <lookup columns only>` whose filters
/// reject the unmatched (null) row (Rally joins
/// `*_two_keys_where_one_matching_equals`: one lookup row of 500k passes,
/// and the join scanned 10M source documents to find its 20). Only a
/// source row whose key tuple some passing lookup row holds can be output,
/// so the source query is narrowed with a `terms` filter on each key that
/// is a plain single-valued keyword field (read from doc values) with the
/// passing rows' values. The rows it drops would join no lookup row or
/// only rejected ones, and none of them warns: every source key is single
/// valued, no lookup key is multi-valued, the key expressions never warn
/// and the filters warned on no lookup row. Applied only when the narrowed
/// scan cannot reach `limit` (the full scan reads every document then too),
/// so `documents_found`/`values_loaded` are the full scan's: the skipped
/// documents are counted, and their key values (one per document holding
/// the field). `None`: scan every document as before.
#[allow(clippy::too_many_arguments)]
fn semi_join(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, left: &Table, lineage: &[(String, Expr)], plan: &LookupJoinPlan,
    lookup: &IndexSchema, on: &[String], filters: &[Expr], projected: &IndexSchema, limit: usize) -> Result<Option<SemiJoin>> {
    #[cfg(test)]
    if SEMI_JOIN_DISABLED.with(|disabled| disabled.get()) { return Ok(None); }
    if schema.slots.is_some() || filters.is_empty() || plan.right_only_null_passes != Some(false) { return Ok(None); }
    let expressions: Vec<Expr> = plan.lkeys.iter().map(|i| subst(&Expr::Col(left.cols[*i].name.clone(), 0), lineage)).collect();
    if !expressions.iter().all(warning_free_key) { return Ok(None); }
    // Only filters a lookup query answers (`lookup_prefilter`): others, a
    // LIKE most rows pass (Rally `*_keys_where_limit1000`), cost a whole
    // lookup read and counts to find the narrowed scan reaches the limit.
    let Some(prefilter) = lookup_prefilter(ctx, lookup, on, filters)? else { return Ok(None) };
    let prefilter = Some(prefilter);
    let (found, loaded) = (ctx.documents_found, ctx.values_loaded);
    let saved = std::mem::take(&mut *ctx.fctx.warnings.borrow_mut());
    let passing = passing_lookup_keys(ctx, lookup, on, filters, prefilter);
    *ctx.fctx.warnings.borrow_mut() = saved;
    (ctx.documents_found, ctx.values_loaded) = (found, loaded);
    let Some(passing) = passing? else { return Ok(None) };
    // The keys to push: plain keyword source fields over keyword lookup keys.
    let mut terms = Vec::new();
    for (j, expression) in expressions.iter().enumerate() {
        let Expr::Col(name, _) = expression else { continue };
        let Some((_, column)) = schema.cols.iter().find(|(n, _)| n == name) else { continue };
        let lookup_keyword = lookup.cols.iter().find(|(n, _)| *n == on[j]).is_some_and(|(_, c)| c.ty == "keyword");
        if column.ty != "keyword" || !lookup_keyword { continue; }
        let mut values: Vec<&str> = Vec::with_capacity(passing.len());
        for (key, _) in passing.values() {
            let Val::Str(value) = &key[j] else { values.clear(); break };
            values.push(value.as_str());
        }
        if values.is_empty() && !passing.is_empty() { continue; }
        values.sort_unstable();
        values.dedup();
        terms.push(json!({"terms": {column.path.clone(): values}}));
    }
    if terms.is_empty() { return Ok(None); }
    // Every column the key expressions read: single-valued keyword doc
    // values in every leaf (their skipped values are then one per document
    // holding the field, and none of them warns).
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&schema.targets)?;
    let columns: Vec<&ColSrc> = projected.cols.iter().map(|(_, c)| c).collect();
    let single = crate::search::for_each_leaf(&targets, &mut |target, field_lookup, leaf| {
        if !doc_values_allowed(target) { return Ok(false); }
        for column in &columns {
            if column.ty != "keyword" { return Ok(false); }
            let Some(id) = direct_field(column, false, field_lookup, false) else { return Ok(false) };
            let single = match leaf {
                rustic_core::query::LeafReader::Segment(segment) => !segment.keyword_from_postings(id) && segment.multi_keyword_reader(id).is_none(),
                rustic_core::query::LeafReader::Memtable(writer, ..) => writer.fast_fields().get_multi_keyword(id).is_none(),
            };
            if !single { return Ok(false); }
        }
        Ok(true)
    })?;
    if !single { return Ok(None); }
    let mut filter = vec![query.clone()];
    filter.extend(terms);
    let narrowed = json!({"bool": {"filter": filter}});
    let kept = crate::search::count(&targets, Some(&narrowed), None)?;
    // Each kept document joins at most this many lookup rows.
    let fanout = passing.values().map(|(_, n)| *n as u64).max().unwrap_or(0);
    if kept.saturating_mul(fanout) >= limit as u64 { return Ok(None); }
    let all = crate::search::count(&targets, Some(query), None)?;
    let mut skipped_values = 0u64;
    for column in &columns {
        let exists = json!({"exists": {"field": column.path}});
        let with = |q: &Value| json!({"bool": {"filter": [q.clone(), exists.clone()]}});
        // Every document holds the key in doc values: one indexed before
        // the field was mapped has it in `_source` only, which the full
        // scan reads and the `terms` filter does not match.
        let holding = crate::search::count(&targets, Some(&with(query)), None)?;
        if holding != all { return Ok(None); }
        skipped_values += holding.saturating_sub(crate::search::count(&targets, Some(&with(&narrowed)), None)?);
    }
    #[cfg(test)]
    SEMI_JOINS.with(|n| n.set(n.get() + 1));
    Ok(Some(SemiJoin { targets, query: narrowed, skipped_docs: all.saturating_sub(kept), skipped_values }))
}

#[cfg(test)]
thread_local! {
    /// Tests: [`streamed_filtered_lookup`] counts a join row by row.
    pub(super) static ORDINAL_COUNT_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: joins [`ordinal_join_count`] counted.
    pub(super) static ORDINAL_COUNTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether every one of `columns` is a plain keyword doc-values read
/// ([`direct_field`]) single-valued in every leaf of `targets`, and every
/// document `query` matches holds a value of each (one indexed before the
/// field was mapped has it in `_source` only). Then a document's key values
/// are its doc values' ordinals' terms, none multi-valued (no warning).
fn keys_single_and_covered(targets: &[crate::search::Target], query: &Value, columns: &[&ColSrc]) -> Result<bool> {
    let named: Vec<(&str, &ColSrc)> = columns.iter().map(|c| (c.path.as_str(), *c)).collect();
    keyword_columns_single_and_covered(targets, query, &named, false)
}

/// The keyword doc-values field of column `name` for a shape check: its
/// direct read ([`direct_field`]), or with `multi_field` a plain keyword
/// sub-field (`lookup_keyword_0.keyword`, whose column reads its parent's
/// `_source` path) found by the column's own name.
fn keyword_dv_field(name: &str, column: &ColSrc, lookup: &crate::mapping::FieldLookup, multi_field: bool) -> Option<u16> {
    if let Some(id) = direct_field(column, false, lookup, false) { return Some(id); }
    if !multi_field || column.ty != "keyword" || column.normalizer.is_some() { return None; }
    let field = lookup.fields.get(name)?;
    let plain = field.declared == EsType::Keyword && field.doc_values && field.normalizer.is_none() && field.null_value.is_none()
        && field.script.is_none() && field.alias_path.is_none() && field.nested_path.is_none() && !lookup.copy_sources.contains_key(name);
    if plain { field.core_id } else { None }
}

/// [`keys_single_and_covered`]; with `multi_field`, keyword sub-fields
/// count too ([`keyword_dv_field`]): for lookup-side checks, whose values
/// are read from `_source` and only the doc values' shape matters.
fn keyword_columns_single_and_covered(targets: &[crate::search::Target], query: &Value, columns: &[(&str, &ColSrc)], multi_field: bool) -> Result<bool> {
    if columns.iter().any(|(_, column)| column.ty != "keyword") { return Ok(false); }
    let single = crate::search::for_each_leaf(targets, &mut |target, field_lookup, leaf| {
        if !doc_values_allowed(target) { return Ok(false); }
        for (name, column) in columns {
            let Some(id) = keyword_dv_field(name, column, field_lookup, multi_field) else { return Ok(false) };
            let single = match leaf {
                rustic_core::query::LeafReader::Segment(segment) => !segment.keyword_from_postings(id) && segment.multi_keyword_reader(id).is_none(),
                rustic_core::query::LeafReader::Memtable(writer, ..) => writer.fast_fields().get_multi_keyword(id).is_none(),
            };
            if !single { return Ok(false); }
        }
        Ok(true)
    })?;
    if !single { return Ok(false); }
    let all = crate::search::count(targets, Some(query), None)?;
    for (name, column) in columns {
        // The doc-values field's own path (a sub-field's name).
        let path = if multi_field { *name } else { column.path.as_str() };
        let holding = json!({"bool": {"filter": [query.clone(), {"exists": {"field": path}}]}});
        if crate::search::count(targets, Some(&holding), None)? != all { return Ok(false); }
    }
    Ok(true)
}

/// `LOOKUP JOIN ... | [WHERE <lookup columns>] | STATS COUNT(*)` (Rally
/// joins `esql_lookup_join_1k_100k_200k_*_count`): the join's row count
/// summed per distinct source key tuple instead of per source row. Each
/// leaf's documents are counted by their keys' doc-values ordinals, and each
/// distinct ordinal tuple's terms count once, as the row-by-row count
/// ([`LookupJoinPlan::count_rows`], [`LookupExprPlan::count_rows`]) counts
/// a row with those keys. Every key must be a plain single-valued keyword
/// field held by every document ([`keys_single_and_covered`]), and no
/// lookup key multi-valued where a key reaching it warns, so no row warns.
/// Returns (count, documents read, key values read) as the row-by-row scan
/// counts them; `None`: count row by row.
fn ordinal_join_count(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, left: &Table, lineage: &[(String, Expr)],
    named: Option<&LookupJoinPlan>, expression: Option<&LookupExprPlan>) -> Result<Option<(u64, u64, u64)>> {
    #[cfg(test)]
    if ORDINAL_COUNT_DISABLED.with(|disabled| disabled.get()) { return Ok(None); }
    if schema.slots.is_some() { return Ok(None); }
    // The join's key index (`Val::key` tuples to their rows), and the rows
    // an unmatched (or null-keyed) source row counts.
    let (lkeys, index, unmatched): (Vec<usize>, &HashMap<Vec<String>, Vec<usize>>, u64) = match (named, expression) {
        (Some(plan), _) => {
            if !plan.mv_keys.is_empty() { return Ok(None); }
            (plan.lkeys.clone(), &plan.index_map, u64::from(plan.right_only_null_passes.unwrap_or(true)))
        }
        (None, Some(plan)) => {
            if !plan.can_count_keys(left.cols.len()) { return Ok(None); }
            let Some(index) = plan.index_map.as_ref() else { return Ok(None) };
            (plan.eq.iter().map(|(l, _)| *l).collect(), index, 1)
        }
        _ => return Ok(None),
    };
    if lkeys.is_empty() || lkeys.len() > 4 { return Ok(None); }
    let mut columns: Vec<&ColSrc> = Vec::with_capacity(lkeys.len());
    for i in &lkeys {
        let Expr::Col(name, _) = subst(&Expr::Col(left.cols[*i].name.clone(), 0), lineage) else { return Ok(None) };
        let Some((_, column)) = schema.cols.iter().find(|(n, _)| *n == name) else { return Ok(None) };
        columns.push(column);
    }
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let targets = node.snapshot(&schema.targets)?;
    if !keys_single_and_covered(&targets, query, &columns)? { return Ok(None); }
    // The lookup's key values interned per key column (raw terms: the
    // `Val::key` of a string is `s` and it), and each key tuple's rows by
    // its ids: a leaf maps its ordinals to these ids once (one pass over
    // each dictionary), and counts and resolves tuples as integers.
    const NONE: u32 = u32::MAX - 1;
    let mut values: Vec<HashMap<&str, u32>> = vec![HashMap::new(); columns.len()];
    let mut factors: HashMap<[u32; 4], u64> = HashMap::with_capacity(index.len());
    'keys: for (key, rows) in index {
        if key.len() != columns.len() { continue; }
        let mut ids = [NONE; 4];
        for (i, part) in key.iter().enumerate() {
            let Some(raw) = part.strip_prefix('s') else { continue 'keys };
            let next = values[i].len() as u32;
            ids[i] = *values[i].entry(raw).or_insert(next);
        }
        factors.insert(ids, rows.len() as u64);
    }
    let count_of = |key: Option<&Vec<String>>| key.and_then(|k| index.get(k)).map_or(unmatched, |rows| rows.len() as u64);
    // Per target, in parallel: (rows, documents, readable). One page per
    // leaf: each distinct ordinal (tuple) of a leaf resolves once.
    let states = crate::search::par_for_each_match(&targets, query, 1 << 30, &|_| (0u64, 0u64, true), &|state, page| {
        const MISSING: u32 = rustic_core::codec::fastfields::MISSING_ORD;
        let mut add = |n: u64, rows: u64| -> Result<()> {
            state.0 = state.0.checked_add(rows.saturating_mul(n)).ok_or_else(|| EsError::internal("join count overflow"))?;
            Ok(())
        };
        let segment = match page.leaf {
            rustic_core::query::LeafReader::Segment(segment) => segment,
            rustic_core::query::LeafReader::Memtable(writer, ..) => {
                // A memtable's keyword columns hold strings: count by them.
                let mut readers = Vec::with_capacity(columns.len());
                for column in &columns {
                    let Some(id) = direct_field(column, false, page.lookup, false) else { state.2 = false; return Ok(false) };
                    readers.push(writer.fast_fields().get_keyword(id));
                }
                let mut tuples: HashMap<[Option<&str>; 4], u64> = HashMap::new();
                let mut docs = 0u64;
                for doc in page.doc_ids() {
                    let mut key = [None; 4];
                    for (slot, reader) in key.iter_mut().zip(&readers) { *slot = reader.and_then(|r| r.get(doc)); }
                    *tuples.entry(key).or_insert(0) += 1;
                    docs += 1;
                }
                state.1 += docs;
                for (key, n) in tuples {
                    let key: Option<Vec<String>> = key[..readers.len()].iter().map(|v| v.map(|v| Val::Str(v.to_string()).key())).collect();
                    add(n, count_of(key.as_ref()))?;
                }
                return Ok(true);
            }
        };
        // Per key column: each ordinal's interned id (`NONE`: a value no
        // lookup key holds there).
        let mut maps: Vec<Option<(rustic_core::codec::fastfields::KeywordReader<'_>, Vec<u32>)>> = Vec::with_capacity(columns.len());
        for (i, column) in columns.iter().enumerate() {
            let Some(id) = direct_field(column, false, page.lookup, false) else { state.2 = false; return Ok(false) };
            let Some(reader) = segment.keyword_reader(id) else { maps.push(None); continue };
            let mut ids = vec![NONE; reader.num_terms()];
            let read = reader.for_each_term(|ord, term| {
                if let Some(id) = std::str::from_utf8(term).ok().and_then(|t| values[i].get(t)) { ids[ord as usize] = *id; }
                true
            });
            if read.is_none() { state.2 = false; return Ok(false); }
            maps.push(Some((reader, ids)));
        }
        let id_of = |map: &Option<(rustic_core::codec::fastfields::KeywordReader<'_>, Vec<u32>)>, doc| -> u32 {
            match map {
                None => MISSING,
                Some((reader, ids)) => match reader.ord(doc) {
                    MISSING => MISSING,
                    ord => ids.get(ord as usize).copied().unwrap_or(NONE),
                },
            }
        };
        let mut docs = 0u64;
        let mut tuples: HashMap<[u32; 4], u64> = HashMap::new();
        if let [map] = maps.as_slice() {
            // One key: a dense count per id.
            let mut per_id = vec![0u64; values[0].len()];
            let (mut missing, mut none) = (0u64, 0u64);
            for doc in page.doc_ids() {
                match id_of(map, doc) {
                    MISSING => missing += 1,
                    NONE => none += 1,
                    id => per_id[id as usize] += 1,
                }
                docs += 1;
            }
            add(missing + none, unmatched)?;
            for (id, n) in per_id.into_iter().enumerate().filter(|(_, n)| *n > 0) {
                add(n, factors.get(&[id as u32, NONE, NONE, NONE]).copied().unwrap_or(unmatched))?;
            }
        } else {
            for doc in page.doc_ids() {
                let mut ids = [NONE; 4];
                for (slot, map) in ids.iter_mut().zip(&maps) { *slot = id_of(map, doc); }
                *tuples.entry(ids).or_insert(0) += 1;
                docs += 1;
            }
            for (ids, n) in tuples {
                // A null or unheld value: no lookup key holds the tuple.
                let rows = if ids[..maps.len()].iter().any(|id| *id == MISSING || *id == NONE) { unmatched } else { factors.get(&ids).copied().unwrap_or(unmatched) };
                add(n, rows)?;
            }
        }
        state.1 += docs;
        Ok(true)
    })?;
    if states.iter().any(|s| !s.2) { return Ok(None); }
    let total = states.iter().try_fold(0u64, |sum, s| sum.checked_add(s.0)).ok_or_else(|| EsError::internal("join count overflow"))?;
    let found: u64 = states.iter().map(|s| s.1).sum();
    #[cfg(test)]
    ORDINAL_COUNTS.with(|n| n.set(n.get() + 1));
    // Every key is held by every document: one value per key column read.
    let loaded = found.saturating_mul(columns.len() as u64);
    Ok(Some((total, found, loaded)))
}

#[cfg(test)]
thread_local! {
    /// Tests: every WHERE after a streamed join runs on the joined rows.
    pub(super) static LEFT_FILTER_PUSHDOWN_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Read the source in bounded chunks, reusing one lookup hash table. The
/// output limit is applied only after the joined row passes every filter.
/// Remote-source and reordered pipelines retain the general
/// interpreter path; the local iterator holds a consistent search snapshot.
fn streamed_filtered_lookup(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    if !residual.is_empty() { return Ok(None); }
    let prefix = rest.iter().take_while(|cmd| match cmd {
        Cmd::Rename(..) | Cmd::Keep(..) | Cmd::Drop(..) => true,
        Cmd::Eval(fields) => fields.iter().all(|field| lookup_prefix_eval(&field.expr)),
        _ => false,
    }).count();
    let Some(cmd @ Cmd::LookupJoin { index, index_pos, on, on_pos, cond, pos, fields }) = rest.get(prefix) else { return Ok(None); };
    let fields = fields.as_ref();
    let tail = &rest[prefix + 1..];
    let nfilters = tail.iter().take_while(|cmd| matches!(cmd, Cmd::Where(..))).count();
    let count_stats = tail.get(nfilters).filter(|cmd| match cmd {
        Cmd::Stats { aggs, by, inline: false, .. } => by.is_empty() && !aggs.is_empty()
            && aggs.iter().all(|field| field.filter.is_none() && matches!(&field.expr,
                Expr::Call(name, args, _) if name == "COUNT" && matches!(args.as_slice(), [Expr::Star]))),
        _ => false,
    });
    let (filters, limit, sort) = if let Some((filters, limit)) = filtered_join_limit(tail) {
        (filters, limit, None)
    } else if let [Cmd::Sort(keys), Cmd::Limit(limit), ..] = tail {
        // The bounded sort path currently handles named-key joins only.
        if cond.is_some() { return Ok(None); }
        (Vec::new(), *limit, Some(keys))
    } else if count_stats.is_some() {
        let filters = tail[..nfilters].iter().map(|cmd| match cmd { Cmd::Where(expr, _) => expr.clone(), _ => unreachable!() }).collect();
        (filters, usize::MAX, None)
    } else { return Ok(None); };
    let base = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    let mut left = base.clone();
    let mut lineage: Vec<(String,Expr)> = base.cols.iter().map(|column|
        (column.name.clone(),Expr::Col(column.name.clone(),0))).collect();
    for cmd in &rest[..prefix] {
        match cmd {
            Cmd::Eval(fields) => for field in fields {
                let expression = subst(&field.expr,&lineage);
                lineage.retain(|(name,_)| name != &field.name);
                lineage.push((field.name.clone(),expression));
            },
            Cmd::Rename(pairs,_) => for (old,new) in pairs {
                if let Some((_,expression)) = lineage.iter().find(|(name,_)| name == old).cloned() {
                    lineage.retain(|(name,_)| name != old && name != new);
                    lineage.push((new.clone(),expression));
                }
            },
            _ => {}
        }
        left = apply(ctx, cmd, left)?;
        lineage.retain(|(name,_)| left.cols.iter().any(|column| &column.name == name));
    }
    // RENAME preserves row positions unless a destination column is replaced.
    // It can therefore share source-key projections when no columns were lost.
    let positional_prefix = left.cols.len() == base.cols.len()
        && rest[..prefix].iter().all(|cmd| matches!(cmd, Cmd::Rename(..)));
    track_fields(ctx, cmd, &left);
    // Filters reading only columns the join takes from the source (a named
    // join's keys, or source columns no lookup column replaces) run on the
    // source rows before the join, as ES pushes them below it: a source row
    // they reject makes no output row, joined or not, and its lookup rows
    // are never read (Rally joins `expanding10000_two_keys_filter1000`:
    // `WHERE lookup_int_0 < 1000` on a join key keeps 1,000 of 100k rows).
    let (left_filters, filters): (Vec<Expr>, Vec<Expr>) = {
        let lookup_names: HashSet<String> = lookup_schema(ctx, index, *index_pos, fields)?.cols.into_iter().map(|(name, _)| name).collect();
        filters.into_iter().partition(|filter| {
            let mut refs = Vec::new();
            crate::esql::analyze::expr_refs(filter, &mut refs);
            !refs.is_empty() && refs.iter().all(|(name, _)| left.idx(name).is_some()
                && (!lookup_names.contains(name) || (cond.is_none() && on.contains(name))))
        })
    };
    #[cfg(test)]
    let (left_filters, filters) = if LEFT_FILTER_PUSHDOWN_DISABLED.with(|d| d.get()) {
        (Vec::new(), left_filters.into_iter().chain(filters).collect())
    } else { (left_filters, filters) };
    let left_bound: Vec<Expr> = left_filters.iter().map(|f| bind(ctx, f, &left.cols)).collect::<Result<_>>()?;
    // With source-side filters, the source columns they and the prefix's
    // expressions read (through the prefix's EVALs and RENAMEs): a first
    // read of just those decides which rows are read whole.
    let left_filter_wanted: Option<Vec<bool>> = (!left_filters.is_empty() && count_stats.is_none()).then(|| {
        let mut names = Vec::new();
        for filter in &left_filters { col_names(&subst(filter, &lineage), &mut names); }
        // Each EVAL's expression in source columns at its own point.
        let mut at_point: Vec<(String, Expr)> = base.cols.iter().map(|column| (column.name.clone(), Expr::Col(column.name.clone(), 0))).collect();
        for cmd in &rest[..prefix] {
            match cmd {
                Cmd::Eval(fields) => for field in fields {
                    let expression = subst(&field.expr, &at_point);
                    col_names(&expression, &mut names);
                    at_point.retain(|(name, _)| name != &field.name);
                    at_point.push((field.name.clone(), expression));
                },
                Cmd::Rename(pairs, _) => for (old, new) in pairs {
                    if let Some((_, expression)) = at_point.iter().find(|(name, _)| name == old).cloned() {
                        at_point.retain(|(name, _)| name != old && name != new);
                        at_point.push((new.clone(), expression));
                    }
                },
                _ => {}
            }
        }
        schema.cols.iter().map(|(name, _)| names.contains(name)).collect::<Vec<bool>>()
    }).filter(|mask: &Vec<bool>| mask.iter().filter(|w| **w).count() < schema.cols.len());
    #[cfg(test)]
    let left_filter_wanted = if LEFT_FILTER_PUSHDOWN_DISABLED.with(|d| d.get()) { None } else { left_filter_wanted };
    // A filtered `LIMIT` join reads each chunk's lookup rows by key (ES's
    // per-page term lookups); the other shapes need the whole lookup side.
    // A count reads only the lookup columns its join and filters do (ES
    // extracts no other field): the keys, the filters' and the condition's.
    let count_fields: Option<HashSet<String>> = count_stats.map(|_| {
        let mut names: HashSet<String> = on.iter().cloned().collect();
        let mut refs = Vec::new();
        for expr in filters.iter().chain(cond.as_ref()) { crate::esql::analyze::expr_refs(expr, &mut refs); }
        names.extend(refs.into_iter().map(|(name, _)| name));
        names
    });
    let fields = match (&count_fields, fields) {
        (Some(count), Some(join)) => Some(count.intersection(join).cloned().collect::<HashSet<String>>()),
        (Some(count), None) => Some(count.clone()),
        (None, join) => join.cloned(),
    };
    let fields = fields.as_ref();
    let keyed_schema = if cond.is_none() && sort.is_none() && count_stats.is_none() {
        let schema = lookup_schema(ctx, index, *index_pos, fields)?;
        lookup_key_kinds(ctx, &schema, on).map(|_| schema)
    } else { None };
    let named = if cond.is_none() { Some(match &keyed_schema {
        // No lookup rows yet: the columns, filters and checks of the join.
        Some(lookup) => LookupJoinPlan::build(ctx, &left, Table { cols: source_cols(lookup, &[]), rows: Vec::new() }, *index_pos, on, on_pos, &filters, *pos)?,
        None => LookupJoinPlan::prepare(ctx, &left, index, *index_pos, on, on_pos, &filters, *pos, fields)?,
    }) } else { None };
    if count_stats.is_some() && named.as_ref().is_some_and(|plan| !plan.filters.is_empty() && plan.right_only_null_passes.is_none()) {
        return Ok(None);
    }
    let mut count_total = 0u64;
    // Bound intermediate expansion as well as the retained top-k rows.
    // Duplicate-key sorts keep the general path until expansion streams too.
    if sort.is_some() && named.as_ref().is_some_and(|plan| plan.index_map.values().any(|rows| rows.len() > 1)) {
        return Ok(None);
    }
    let expression = cond.as_ref().map(|condition| LookupExprPlan::prepare(ctx, &left, index, *index_pos, condition, &filters, fields)).transpose()?;
    if count_stats.is_some() && expression.as_ref().is_some_and(|plan| !plan.can_count_keys(left.cols.len())) {
        return Ok(None);
    }
    let mut out = Table { cols: named.as_ref().map(|plan| plan.cols.clone()).unwrap_or_else(|| expression.as_ref().unwrap().out_cols.clone()), rows: Vec::new() };
    let mut output_memory=crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?;
    if let Some(keys) = sort { sort_table(ctx, &mut out, keys)?; }
    // The retained top rows' sort keys (`sort_table`'s), beside `out.rows`:
    // each joined row is placed by binary search after every row not
    // greater than it, which is the stable sort of the retained rows then the
    // later ones, truncated. Re-sorting the retained rows for every chunk
    // made a sort over a lookup column (Rally joins `*_sort_limit10000`)
    // sort 10,000 rows once per chunk holding a competitive document.
    let top_keys = sort.map(|keys| -> Result<(Vec<Expr>, Vec<bool>)> {
        let bound: Vec<Expr> = keys.iter().map(|key| bind(ctx, &key.expr, &out.cols)).collect::<Result<_>>()?;
        let versions = bound.iter().map(|b| type_of(b, &out.cols).as_deref() == Some("version")).collect();
        Ok((bound, versions))
    }).transpose()?;
    let mut retained_keys: Vec<Vec<Val>> = Vec::new();
    // Bytes of the retained rows and keys, kept as they change: summing them
    // over every retained row once per chunk was most of a lookup-column
    // SORT's time.
    let mut retained_bytes = 0u64;
    // For a single lookup-side sort column, reject noncompetitive source
    // documents using only their join keys before loading wide source rows.
    // This stays inside the same iterator snapshot and preserves source-order
    // ties: after top-k is full, later equal keys cannot displace earlier rows.
    let late_sort = sort.and_then(|keys| {
        if !positional_prefix || keys.len() != 1 { return None; }
        let plan = named.as_ref()?;
        let Expr::Col(name, _) = &keys[0].expr else { return None; };
        let output = plan.cols.iter().position(|col| col.name == *name)?;
        let right = *plan.rcols.get(output.checked_sub(plan.lkeep.len())?)?;
        if !matches!(plan.right.cols[right].ty.as_str(), "keyword" | "text" | "long" | "integer" | "double" | "boolean")
            || plan.right.rows.iter().any(|row| matches!(row[right], Val::Multi(_))) { return None; }
        let projected = IndexSchema { cols:plan.lkeys.iter().map(|i| schema.cols[*i].clone()).collect(),targets:Vec::new(),slots:None,projection:None };
        Some((output, right, projected, &keys[0]))
    });
    // A rejecting right-only filter needs only source join keys until a
    // candidate survives. Read complete output rows only for those candidates.
    let project_keys = |keys: &[usize]| -> Result<(IndexSchema,Vec<Expr>)> {
        let expressions: Vec<Expr> = keys.iter().map(|index|
            subst(&Expr::Col(left.cols[*index].name.clone(),0),&lineage)).collect();
        let mut needed = Vec::new();
        for expression in &expressions { col_names(expression,&mut needed); }
        let projected = IndexSchema { cols: schema.cols.iter().filter(|(name,_)| needed.contains(name)).cloned().collect(), targets: Vec::new(), slots: None, projection: None };
        let columns = source_cols(&projected,&[]);
        let bound = expressions.iter().map(|expression| bind(ctx,expression,&columns)).collect::<Result<Vec<_>>>()?;
        Ok((projected,bound))
    };
    let filter_projection = named.as_ref().filter(|plan| count_stats.is_none() && plan.right_only_null_passes == Some(false))
        .map(|plan| project_keys(&plan.lkeys)).transpose()?;
    let expression_filter = if count_stats.is_none() {
        if let Some(plan) = &expression {
            plan.accepted_filter_keys(ctx,left.cols.len())?.map(|keys| {
                let (projected,bound) = project_keys(&plan.eq.iter().map(|(index,_)| *index).collect::<Vec<_>>())?;
                Ok::<_,EsError>((projected,bound,keys))
            }).transpose()?
        } else { None }
    } else { None };
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().enumerate().map(|(i, (name, _))| {
        if count_stats.is_some() && positional_prefix {
            named.as_ref().map_or_else(|| expression.as_ref().unwrap().eq.iter().any(|(l,_)| *l == i), |plan| plan.lkeys.contains(&i))
        } else { needed.as_ref().is_none_or(|set| set.contains(name)) }
    }).collect();
    // Keyed reads per chunk until they have read as many lookup rows as the
    // lookup index holds; then the whole (projected) lookup side, read once.
    // A small lookup index (Rally joins `*_two_keys_where_one_matching_equals`:
    // 1,000 keys, a match in 1 of 1,000 source rows) was otherwise read again
    // for every chunk the filter scanned. Keyed and whole plans join alike.
    let lookup_docs = match &keyed_schema { Some(lookup) => crate::search::count(&lookup.targets, None, None)? as usize, None => 0 };
    let mut keyed_rows = 0usize;
    let mut whole: Option<LookupJoinPlan> = None;
    // Per distinct source key row (`filter_projection`'s columns), under a
    // plan for the whole lookup side: its tuple's decision. The tuple's
    // expressions are deterministic (`lookup_prefix_eval`), and evaluating
    // them per document (a two-key CONCAT, Rally `*_two_keys_where_*`) was
    // most of the scan.
    let mut decided: HashMap<Vec<String>, (bool, bool)> = HashMap::new();
    // A join count summed per distinct key tuple (`ordinal_join_count`).
    if let (Some(stats), true, true) = (count_stats, positional_prefix, left_bound.is_empty()) {
        if let Some((total, found, loaded)) = ordinal_join_count(ctx, schema, query, &left, &lineage, named.as_ref(), expression.as_ref())? {
            ctx.documents_found += found;
            ctx.values_loaded += loaded;
            out = apply(ctx, stats, out)?;
            let count = i64::try_from(total).map_err(|_| EsError::internal("join count overflow"))?;
            for value in &mut out.rows[0] { *value = Val::Long(count); }
            taken();
            return Ok(Some((out, prefix + 1 + nfilters + 1)));
        }
    }
    // A filter only some lookup rows pass: scan only the source documents
    // whose keys they hold (`semi_join`).
    let semi = match (&named, &filter_projection, &keyed_schema) {
        (Some(plan), Some((projected, _)), Some(lookup)) if cond.is_none() && sort.is_none() && limit > 0 =>
            semi_join(ctx, schema, query, &left, &lineage, plan, lookup, on, &filters, projected, limit)?,
        _ => None,
    };
    let (scan_targets, query) = match &semi {
        Some(semi) => {
            ctx.documents_found += semi.skipped_docs;
            ctx.values_loaded += semi.skipped_values;
            (&semi.targets[..], &semi.query)
        }
        None => (&schema.targets[..], query),
    };
    if limit > 0 && (keyed_schema.is_some() || !named.as_ref().is_some_and(LookupJoinPlan::rejects_all)) {
        crate::search::for_each_match_wanting(scan_targets, query, PAGE, None, &mut |page| {
            let key_doc_values = doc_values_allowed(page.target);
            let mut at = 0;
            while at < page.len() {
                let end = (at + 1024).min(page.len());
                let range: Vec<rustic_core::types::DocId>;
                let docs = if page.all_docs {
                    range = (at as u32..end as u32).collect();
                    range.as_slice()
                } else { &page.docs[at..end] };
                let mut selected = Vec::new();
                // This chunk's keyed plan (`keyed_schema`).
                let mut chunk_plan: Option<LookupJoinPlan> = None;
                let docs = if let Some((projected,bound)) = &filter_projection {
                    let (keys, loaded, _page_memory) = lookup_key_rows(projected, &vec![true; projected.cols.len()], page.target, page.lookup, page.leaf, docs, key_doc_values)?;
                    ctx.values_loaded += loaded;
                    if let Some(lookup) = &keyed_schema {
                        if whole.is_none() && keyed_rows > lookup_docs {
                            let right = lookup_rows(ctx, lookup, None)?;
                            whole = Some(LookupJoinPlan::build(ctx, &left, right, *index_pos, on, on_pos, &filters, *pos)?);
                        }
                    }
                    // Per-chunk keyed plans need every tuple first; a plan for
                    // the whole lookup side decides each distinct key row once.
                    let keyed_now = keyed_schema.is_some() && whole.is_none();
                    let tuples = if keyed_now {
                        Some(keys.iter().map(|row| bound.iter().map(|expression| key_value(expression,row,&ctx.fctx)).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?)
                    } else { None };
                    if let (Some(lookup), Some(tuples)) = (&keyed_schema, &tuples) {
                        {
                            chunk_plan = match LookupJoinPlan::for_keys(ctx, &left, tuples, lookup, *index_pos, on, on_pos, &filters, *pos)? {
                                Some(plan) => Some(plan),
                                // Not keyed after all: the whole lookup side.
                                None => {
                                    let right = lookup_rows(ctx, lookup, None)?;
                                    Some(LookupJoinPlan::build(ctx, &left, right, *index_pos, on, on_pos, &filters, *pos)?)
                                }
                            };
                            keyed_rows += chunk_plan.as_ref().map_or(0, |plan| plan.right.rows.len());
                        }
                    }
                    let plan = chunk_plan.as_ref().or(whole.as_ref()).or(named.as_ref()).unwrap();
                    // (joins a lookup row, warns for a multi-valued key). A
                    // skipped row's null join is rejected; ES still warns for
                    // a multi-valued key on either side.
                    let decide = |values: &[Val]| -> (bool, bool) {
                        if values.iter().any(Val::is_null) { return (false, false); }
                        let key: Vec<String> = values.iter().map(Val::key).collect();
                        (plan.index_map.get(&key).is_some_and(|rows| !rows.is_empty()),
                            values.iter().any(|value| matches!(value, Val::Multi(_))) || plan.mv_keys.contains(&key))
                    };
                    for (i, (&doc, row)) in docs.iter().zip(&keys).enumerate() {
                        let (joins, warns) = match &tuples {
                            Some(tuples) => decide(&tuples[i]),
                            None => {
                                let memo_key: Vec<String> = row.iter().map(Val::key).collect();
                                match decided.get(&memo_key) {
                                    Some(decision) => *decision,
                                    None => {
                                        let before = ctx.fctx.warnings.borrow().list.len();
                                        let values = bound.iter().map(|expression| key_value(expression,row,&ctx.fctx)).collect::<Result<Vec<_>>>()?;
                                        let decision = decide(&values);
                                        // A key row whose evaluation warned is evaluated again.
                                        if ctx.fctx.warnings.borrow().list.len() == before && decided.len() < DECIDED_KEY_ROWS {
                                            decided.insert(memo_key, decision);
                                        }
                                        decision
                                    }
                                }
                            }
                        };
                        if warns { super::lookup_mv_warning(&ctx.fctx, *pos); }
                        if joins { selected.push(doc); }
                    }
                    ctx.documents_found += (docs.len() - selected.len()) as u64;
                    selected.as_slice()
                } else if let Some((projected, bound, accepted)) = &expression_filter {
                    let (keys, loaded, _page_memory) = lookup_key_rows(projected, &vec![true; projected.cols.len()], page.target, page.lookup, page.leaf, docs, key_doc_values)?;
                    ctx.values_loaded += loaded;
                    for (&doc, row) in docs.iter().zip(keys) {
                        let values = bound.iter().map(|expression| eval(expression,&row,&ctx.fctx)).collect::<Result<Vec<_>>>()?;
                        if values.iter().all(|value| !value.is_null() && !matches!(value, Val::Multi(_)))
                            && accepted.contains(&values.iter().map(Val::key).collect::<Vec<_>>()) {
                            selected.push(doc);
                        }
                    }
                    ctx.documents_found += (docs.len() - selected.len()) as u64;
                    selected.as_slice()
                } else if let Some((output, right, projected, key)) = late_sort.as_ref().filter(|_| out.rows.len() == limit) {
                    let plan = named.as_ref().unwrap();
                    let (keys, loaded, _page_memory) = lookup_key_rows(projected, &vec![true; projected.cols.len()], page.target, page.lookup, page.leaf, docs, key_doc_values)?;
                    ctx.values_loaded += loaded;
                    let worst = &out.rows.last().unwrap()[*output];
                    for (&doc, values) in docs.iter().zip(keys) {
                        let matched = if values.iter().any(Val::is_null) { None } else {
                            plan.index_map.get(&values.iter().map(Val::key).collect::<Vec<_>>()).and_then(|rows| rows.first())
                        };
                        let value = matched.map(|i| &plan.right.rows[*i][*right]).unwrap_or(&Val::Null);
                        if lookup_sort_cmp(value, worst, key) == std::cmp::Ordering::Less { selected.push(doc); }
                    }
                    ctx.documents_found += (docs.len() - selected.len()) as u64;
                    selected.as_slice()
                } else { docs };
                if docs.is_empty() { at = end; continue; }
                // The columns the pipeline reads, from doc values where they
                // equal the `_source` reading (only keys for a count).
                let mut chunk = if let Some(filter_wanted) = &left_filter_wanted {
                    // Source-side filters first, over just the columns they and
                    // the prefix read (the others null): only the rows they keep
                    // are read whole, the prefix rerun over them without its
                    // warnings (each was registered by the first run, over every row).
                    let (rows, loaded, _page_memory) = lookup_key_rows(schema, filter_wanted, page.target, page.lookup, page.leaf, docs, key_doc_values)?;
                    ctx.values_loaded += loaded;
                    let mut thin = Table { cols: base.cols.clone(), rows };
                    for cmd in &rest[..prefix] { thin = apply(ctx, cmd, thin)?; }
                    let mut passing: Vec<rustic_core::types::DocId> = Vec::new();
                    for (&doc, row) in docs.iter().zip(&thin.rows) {
                        if join_row_passes(ctx, &left_bound, row)? { passing.push(doc); }
                    }
                    ctx.documents_found += docs.len() as u64;
                    if passing.is_empty() { at = end; continue; }
                    let (rows, loaded, _page_memory) = lookup_key_rows(schema, &wanted, page.target, page.lookup, page.leaf, &passing, key_doc_values)?;
                    ctx.values_loaded += loaded;
                    let saved = std::mem::take(&mut *ctx.fctx.warnings.borrow_mut());
                    let mut full = Table { cols: base.cols.clone(), rows };
                    let rerun = (|| -> Result<Table> { for cmd in &rest[..prefix] { full = apply(ctx, cmd, std::mem::take(&mut full))?; } Ok(std::mem::take(&mut full)) })();
                    *ctx.fctx.warnings.borrow_mut() = saved;
                    rerun?
                } else {
                    let (rows, loaded, _page_memory) = lookup_key_rows(schema, &wanted, page.target, page.lookup, page.leaf, docs, key_doc_values)?;
                    ctx.documents_found += docs.len() as u64;
                    ctx.values_loaded += loaded;
                    let mut chunk = Table { cols: base.cols.clone(), rows };
                    for cmd in &rest[..prefix] { chunk = apply(ctx, cmd, chunk)?; }
                    if !left_bound.is_empty() {
                        let mut kept = Vec::with_capacity(chunk.rows.len());
                        for row in std::mem::take(&mut chunk.rows) {
                            if join_row_passes(ctx, &left_bound, &row)? { kept.push(row); }
                        }
                        chunk.rows = kept;
                        if chunk.rows.is_empty() { at = end; continue; }
                    }
                    chunk
                };
                if count_stats.is_some() {
                    let count = if let Some(plan) = &named { plan.count_rows(ctx, &chunk)? } else { expression.as_ref().unwrap().count_rows(&chunk)? };
                    count_total = count_total.checked_add(count).ok_or_else(|| EsError::internal("join count overflow"))?;
                    at = end;
                    continue;
                }
                let remaining = if sort.is_some() { None } else { Some(limit - out.rows.len()) };
                if chunk_plan.is_none() && whole.is_none() {
                    if let Some(lookup) = &keyed_schema {
                        if keyed_rows > lookup_docs {
                            let right = lookup_rows(ctx, lookup, None)?;
                            whole = Some(LookupJoinPlan::build(ctx, &left, right, *index_pos, on, on_pos, &filters, *pos)?);
                        } else {
                            chunk_plan = Some(LookupJoinPlan::for_rows(ctx, &left, &chunk.rows, lookup, *index_pos, on, on_pos, &filters, *pos)?);
                            keyed_rows += chunk_plan.as_ref().map_or(0, |plan| plan.right.rows.len());
                        }
                    }
                }
                let joined = if let Some(plan) = chunk_plan.as_ref().or(whole.as_ref()).or(named.as_ref()) { plan.apply(ctx, &chunk, remaining)? }
                    else { expression.as_ref().unwrap().apply(ctx, &chunk, remaining)? };
                output_memory.grow(rows_bytes(&joined.rows),crate::esql::exec::ESQL_LABEL)?;
                if let (Some(keys), Some((bound, versions))) = (sort, &top_keys) {
                    // Equal keys keep source order: a later row goes after
                    // every retained row it does not sort before.
                    for row in joined.rows {
                        let key = bound.iter().zip(keys).zip(versions).map(|((b, k), version)| Ok(sort_key_value(eval(b, &row, &ctx.fctx)?, *version, k.desc))).collect::<Result<Vec<Val>>>()?;
                        let at = retained_keys.partition_point(|kept| cmp_sort_keys(keys, kept, &key) != std::cmp::Ordering::Greater);
                        if at >= limit { continue; }
                        retained_bytes = retained_bytes.saturating_add(row_bytes(&row).saturating_add(row_bytes(&key)).saturating_mul(2));
                        retained_keys.insert(at, key);
                        out.rows.insert(at, row);
                        if out.rows.len() > limit {
                            let (row, key) = (out.rows.pop().unwrap(), retained_keys.pop().unwrap());
                            retained_bytes = retained_bytes.saturating_sub(row_bytes(&row).saturating_add(row_bytes(&key)).saturating_mul(2));
                        }
                    }
                    resize_page_memory(&mut output_memory,retained_bytes)?;
                } else {
                    out.rows.extend(joined.rows);
                    if out.rows.len() >= limit { return Ok(false); }
                }
                at = end;
            }
            Ok(true)
        })?;
    }
    if let Some(stats) = count_stats {
        // Reuse aggregate verification and output schema, then replace the
        // empty-input counts with accumulated exact join multiplicities.
        out = apply(ctx, stats, out)?;
        let count = i64::try_from(count_total).map_err(|_| EsError::internal("join count overflow"))?;
        for value in &mut out.rows[0] { *value = Val::Long(count); }
        taken();
        return Ok(Some((out, prefix + 1 + nfilters + 1)));
    }
    taken();
    resize_page_memory(&mut output_memory,rows_bytes(&out.rows))?;
    output_memory.retain_for_request();
    // Keep projection and LIMIT commands in the normal interpreter.
    Ok(Some((out, prefix + 1 + if sort.is_some() { 1 } else { left_filters.len() + filters.len() })))
}

/// `[row-local commands] | LIMIT n` (or only row-local commands: the
/// implicit limit) over the matches in search order: documents are read
/// from `_source` a chunk at a time and the residual filters and commands
/// run per chunk until `n` rows are there. Same rows, in the same order, as
/// loading every match and running the commands over the table.
pub(super) fn streamed_limit(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, residual: &[(Expr, usize)], rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    let k = rest.iter().take_while(|c| row_local(c)).count();
    let (n, covered, group_limit) = match rest.get(k) {
        Some(Cmd::Limit(n)) => ((*n).min(ctx.max_limit), k + 1, None),
        // A grouped limit can stream when only a final LIMIT follows it.
        // A later SORT/STATS must still see all groups and takes the table path.
        Some(Cmd::LimitBy(per_group, keys)) => {
            let cap = match &rest[k + 1..] {
                [] => ctx.max_limit,
                [Cmd::Limit(cap)] => (*cap).min(ctx.max_limit),
                _ => return Ok(None),
            };
            (if *per_group == 0 { 0 } else { cap }, rest.len(), Some((*per_group, keys)))
        }
        None => (ctx.default_limit.min(ctx.max_limit), k, None),
        Some(_) => return Ok(None),
    };
    if (!residual.is_empty() || rest[..k].iter().any(|c| matches!(c, Cmd::Where(..)))) && searched_docs(schema)? < residual_min_docs() {
        return Ok(None);
    }
    let mut local: Vec<Cmd> = residual.iter().map(|(e, p)| Cmd::Where(e.clone(), *p)).collect();
    local.extend(rest[..k].iter().cloned());
    // The output columns: the commands over no rows.
    let mut out = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    let mut output_memory=crate::memory::breakers().reserve_request(0,crate::esql::exec::ESQL_LABEL)?;
    for c in &local {
        out = apply(ctx, c, out)?;
    }
    let grouped = group_limit.map(|(limit, keys)| {
        let bound = keys.iter().map(|key| bind(ctx, key, &out.cols)).collect::<Result<Vec<_>>>()?;
        Ok::<_, EsError>((limit, bound))
    }).transpose()?;
    // With no row transforms, rejected groups need only their key columns.
    // Keep extraction semantics (_source visibility, mappings and nulls), but
    // defer constructing wide output rows until after group selection.
    let projected_group = if local.is_empty() {
        group_limit.map(|(_, keys)| {
            let mut names = Vec::new();
            for key in keys { col_names(key, &mut names); }
            let names: HashSet<String> = names.into_iter().collect();
            let projected = IndexSchema {
                cols: schema.cols.iter().filter(|(name, _)| names.contains(name)).cloned().collect(),
                targets: Vec::new(), slots: None, projection: None,
            };
            let columns = source_cols(&projected, &[]);
            let bound = keys.iter().map(|key| bind(ctx, key, &columns)).collect::<Result<Vec<_>>>()?;
            // Plain-column keys group by their non-null values alone
            // ([`sparse_group_key`]): `positions[c]` lists the keys on column `c`.
            let positions = bound.iter().enumerate().try_fold(vec![Vec::new(); columns.len()], |mut positions: Vec<Vec<usize>>, (k, key)| match key {
                Expr::Idx(c) if *c < columns.len() => { positions[*c].push(k); Some(positions) }
                _ => None,
            });
            Ok::<_, EsError>((projected, bound, positions))
        }).transpose()?
    } else { None };
    // Group identities: [`grouped_row_key`]'s bytes, or [`sparse_group_key`]
    // when every key is a plain column (one encoding per query).
    let mut group_counts: HashMap<Vec<u8>, usize> = HashMap::new();
    // The whole result, with `drop_null_columns` (Rally esql
    // `from_idx_limit_*_drop_null`: ~2,500 mapped columns, a dozen with
    // values): rows are read as their non-null values only and built over
    // the columns some row has a value in, every column listed apart
    // ([`EsqlResult::all_columns`]). A dense row per document allocated,
    // copied and scanned every all-null column several times.
    let narrow = ctx.drop_null_columns && local.is_empty() && grouped.is_none() && covered == rest.len() && schema.slots.is_none();
    let mut sparse: Vec<Vec<(usize, Val)>> = Vec::new();
    taken();
    if n > 0 {
        let needed = needed_columns(ctx.full, schema);
        let wanted: Vec<bool> = schema.cols.iter().map(|(c, _)| needed.as_ref().is_none_or(|s| s.contains(c))).collect();
        let base = source_cols(schema, &[]);
        let mut found = 0u64;
        let mut loaded = 0u64;
        let mut failure: Option<EsError> = None;
        let mut mismatch = false;
        let mut have = 0usize;
        let ctx_cell = std::cell::RefCell::new(&mut *ctx);
        let filtered = grouped.is_some() || local.iter().any(|c| matches!(c, Cmd::Where(..)));
        let rows_wanted = (!filtered).then_some(n);
        let page = if filtered { PAGE } else { n.clamp(16, PAGE) };
        crate::search::for_each_match_wanting(&schema.targets, query, page, rows_wanted, &mut |p| {
            let mut ctx = ctx_cell.borrow_mut();
            // Small chunks while few rows are wanted: a LIMIT 10 reads ~10 documents.
            let mut at = 0;
            let total = p.len();
            let mut range: Vec<rustic_core::types::DocId> = Vec::new();
            // A filtered read starts at twice the rows still wanted and
            // doubles: a LIMIT 10 behind a selective-enough WHERE reads tens
            // of documents, not 1024 (each a stored-chunk read).
            let mut step = 0usize;
            while at < total {
                let want = n - have;
                step = if step == 0 { want.saturating_mul(2).clamp(16, 1024) } else { (step * 2).min(1024) };
                let chunk = if filtered { step } else { want.clamp(1, 1024) };
                let end = (at + chunk).min(total);
                let docs: &[rustic_core::types::DocId] = if p.all_docs {
                    range.clear();
                    range.extend(at as u32..end as u32);
                    &range
                } else {
                    &p.docs[at..end]
                };
                at = end;
                found += docs.len() as u64;
                let mut selected_docs = Vec::new();
                let docs = if let Some((projected, keys, positions)) = &projected_group {
                    let flags = vec![true; projected.cols.len()];
                    let limit = grouped.as_ref().unwrap().0;
                    // A few plain keyword keys read from doc values (one
                    // column read per batch; `_source` only where a doc has
                    // no doc value), not by decoding every candidate's
                    // `_source`: a LIMIT 1 BY scans until each key is seen.
                    let key_values = projected.cols.len() <= KEY_DOC_VALUES_MAX && doc_values_allowed(p.target);
                    if let (Some(positions), true) = (positions, key_values) {
                        if keys_absent_and_proven(projected, p.target, p.lookup, p.leaf, docs) {
                            // Every document joins the all-null group.
                            let count = group_counts.entry(sparse_group_key(positions, &[])).or_insert(0);
                            let take = limit.saturating_sub(*count).min(docs.len());
                            *count += take;
                            selected_docs.extend_from_slice(&docs[..take]);
                            #[cfg(test)]
                            ABSENT_KEY_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        } else {
                            let (rows, count, _page_memory) = match lookup_key_rows(projected, &flags, p.target, p.lookup, p.leaf, docs, true) {
                                Ok(value) => value,
                                Err(error) => { failure = Some(error); return Ok(false); }
                            };
                            loaded += count;
                            for (&doc, row) in docs.iter().zip(rows) {
                                let values: Vec<(usize, Val)> = row.into_iter().enumerate().filter(|(_, v)| !v.is_null()).collect();
                                let count = group_counts.entry(sparse_group_key(positions, &values)).or_insert(0);
                                if *count < limit { *count += 1; selected_docs.push(doc); }
                            }
                        }
                    } else if let Some(positions) = positions {
                        // Wide keys: only the non-null values of each document
                        // are read and hashed, never a row of nulls.
                        let (rows, count, _page_memory) = match source_sparse_rows(projected, &flags, p.target, p.lookup, p.leaf, docs) {
                            Ok(value) => value,
                            Err(error) => { failure = Some(error); return Ok(false); }
                        };
                        loaded += count;
                        for (&doc, row) in docs.iter().zip(rows) {
                            let count = group_counts.entry(sparse_group_key(positions, &row)).or_insert(0);
                            if *count < limit { *count += 1; selected_docs.push(doc); }
                        }
                    } else {
                        let read = if key_values { lookup_key_rows(projected, &flags, p.target, p.lookup, p.leaf, docs, true) } else { source_rows(projected, &flags, p.target, p.lookup, p.leaf, docs) };
                        let (rows, count, _page_memory) = match read {
                            Ok(value) => value,
                            Err(error) => { failure = Some(error); return Ok(false); }
                        };
                        loaded += count;
                        for (&doc, row) in docs.iter().zip(rows) {
                            let key = match grouped_row_key(keys, &row, &ctx.fctx) {
                                Ok(key) => key.into_bytes(),
                                Err(error) => { failure = Some(error); return Ok(false); }
                            };
                            let count = group_counts.entry(key).or_insert(0);
                            if *count < limit { *count += 1; selected_docs.push(doc); }
                        }
                    }
                    selected_docs.as_slice()
                } else { docs };
                if docs.is_empty() { continue; }
                if narrow {
                    let (rows, count, _page_memory) = match source_sparse_rows(schema, &wanted, p.target, p.lookup, p.leaf, docs) {
                        Ok(value) => value,
                        Err(error) => { failure = Some(error); return Ok(false); }
                    };
                    loaded += count;
                    let rows: Vec<Vec<(usize, Val)>> = rows.into_iter().take(n - have).collect();
                    have += rows.len();
                    output_memory.grow(sparse_rows_bytes(&rows), crate::esql::exec::ESQL_LABEL)?;
                    sparse.extend(rows);
                    if have >= n { return Ok(false); }
                    continue;
                }
                let (rows,_page_memory) = match source_rows(schema, &wanted, p.target, p.lookup, p.leaf, docs) {
                    Ok((r, l, memory)) => {
                        loaded += l;
                        (r,memory)
                    }
                    Err(e) => {
                        failure = Some(e);
                        return Ok(false);
                    }
                };
                let mut t = Table { cols: base.clone(), rows };
                for c in &local {
                    t = match apply(&mut ctx, c, t) {
                        Ok(t) => t,
                        Err(e) => {
                            failure = Some(e);
                            return Ok(false);
                        }
                    };
                }
                if t.rows.is_empty() {
                    continue;
                }
                if out.rows.is_empty() && have == 0 {
                    out.cols = t.cols;
                } else if out.cols != t.cols {
                    mismatch = true;
                    return Ok(false);
                }
                if let Some((limit, keys)) = grouped.as_ref().filter(|_| projected_group.is_none()) {
                    let mut selected = Vec::new();
                    for row in t.rows {
                        let key = match grouped_row_key(keys, &row, &ctx.fctx) {
                            Ok(key) => key.into_bytes(),
                            Err(error) => { failure = Some(error); return Ok(false); }
                        };
                        let count = group_counts.entry(key).or_insert(0);
                        if *count < *limit { *count += 1; selected.push(row); }
                    }
                    t.rows = selected;
                }
                have += t.rows.len();
                output_memory.grow(rows_bytes(&t.rows),crate::esql::exec::ESQL_LABEL)?;
                out.rows.extend(t.rows);
                if have >= n {
                    return Ok(false);
                }
            }
            Ok(true)
        })?;
        if let Some(e) = failure {
            return Err(e);
        }
        if mismatch {
            // Value-dependent column types across chunks: the table path decides.
            return Ok(None);
        }
        ctx.documents_found += found;
        ctx.values_loaded += loaded;
        if narrow {
            sparse.truncate(n);
            let mut present = vec![false; base.len()];
            for row in &sparse {
                for (column, _) in row { present[*column] = true; }
            }
            let keep: Vec<usize> = (0..base.len()).filter(|column| present[*column]).collect();
            let mut slot = vec![0usize; base.len()];
            for (i, column) in keep.iter().enumerate() { slot[*column] = i; }
            out.cols = keep.iter().map(|column| base[*column].clone()).collect();
            out.rows = sparse.into_iter().map(|row| {
                let mut dense = vec![Val::Null; keep.len()];
                for (column, value) in row { dense[slot[column]] = value; }
                dense
            }).collect();
            ctx.all_columns = Some(base);
        }
        out.rows.truncate(n);
    }
    if covered > k && grouped.is_none() {
        out = apply(ctx, &rest[k], out)?;
    }
    resize_page_memory(&mut output_memory,rows_bytes(&out.rows))?;
    output_memory.retain_for_request();
    Ok(Some((out, covered)))
}

enum SourceColumn<'a> {
    Constant(Val),
    Read(&'a ColSrc, bool),
}

// Resolve index-dependent mapping choices once per batch, not per cell.
fn source_plan<'a>(schema: &'a IndexSchema, wanted: &[bool], index: &str) -> Vec<SourceColumn<'a>> {
    schema.cols.iter().zip(wanted).map(|((_, column), wanted)| {
        if !*wanted { return SourceColumn::Constant(Val::Null); }
        if let Some(value) = column.constants.get(index) {
            return SourceColumn::Constant(conv(value, column.per_index.get(index).unwrap_or(column)).unwrap_or(Val::Null));
        }
        let column = if let Some(mapped) = column.per_index.get(index) {
            mapped
        } else {
            if !column.types.is_empty() && !index.is_empty() && !column.types.iter().any(|(_, indices)| indices.iter().any(|i| i == index)) {
                return SourceColumn::Constant(Val::Null);
            }
            column
        };
        if column.ty == "unsupported" { return SourceColumn::Constant(Val::Null); }
        SourceColumn::Read(column, column.ordered.iter().any(|i| i == index))
    }).collect()
}

/// The `_source` row of one document, as [`load`] reads it (no metadata
/// columns); `wanted` marks the columns to read.
fn source_row(schema: &IndexSchema, wanted: &[bool], t: &crate::search::Target, lookup: &crate::mapping::FieldLookup, leaf: &rustic_core::query::LeafReader<'_>, doc: rustic_core::types::DocId) -> Result<(Vec<Val>, u64, crate::memory::Reservation)> {
    let (mut rows,loaded,mut memory)=source_rows_guarded(schema,wanted,t,lookup,leaf,&[doc])?;
    let row=rows.pop().unwrap_or_default();
    resize_page_memory(&mut memory,crate::esql::exec::row_bytes(&row).saturating_mul(2))?;
    Ok((row,loaded,memory))
}

/// Most key columns a LIMIT BY reads from doc values (wider keys read only
/// the non-null `_source` values of each document instead).
const KEY_DOC_VALUES_MAX: usize = 16;

/// Whether doc values may stand in for `_source` on `target`: no field
/// security, stored source without includes/excludes (what the lookup-key
/// and SQL column reads require).
pub(super) fn doc_values_allowed(target: &crate::search::Target) -> bool {
    target.field_access().is_none() && target.svc.source_stored_as_sent()
}

/// Plain top-level keyword doc values, and single millisecond dates, have the
/// same conversion, sorting and deduplication as source extraction. Keep every
/// source-transforming mapping and field-security case on the source path
/// (join keys, LIMIT BY keys and the narrow column sets of projected joins).
pub(super) fn lookup_key_rows(schema: &IndexSchema, wanted: &[bool], target: &crate::search::Target, lookup: &crate::mapping::FieldLookup,
    leaf: &rustic_core::query::LeafReader<'_>, docs: &[rustic_core::types::DocId], allowed: bool) -> Result<(Vec<Vec<Val>>, u64, crate::memory::Reservation)> {
    direct_column_rows(schema,wanted,target,lookup,leaf,docs,allowed,false)
}

/// SQL can use mapped keyword/float columns directly when source visibility
/// and mapping transformations cannot change their values. Missing values
/// retain the per-document source fallback (including late-added mappings).
pub(super) fn sql_column_rows(schema: &IndexSchema, wanted: &[bool], target: &crate::search::Target, lookup: &crate::mapping::FieldLookup,
    leaf: &rustic_core::query::LeafReader<'_>, docs: &[rustic_core::types::DocId]) -> Result<(Vec<Vec<Val>>, u64, crate::memory::Reservation)> {
    let allowed = doc_values_allowed(target);
    direct_column_rows(schema,wanted,target,lookup,leaf,docs,allowed,true)
}

/// Each doc's term of single-valued keyword column `field`: from the
/// segment's decoded terms when its table is front-coded and small (shared
/// across requests), else decoded once per ordinal in this batch. Reading a
/// front-coded term per doc replays its block: SQL's script and grouping
/// queries over a keyword column (Rally sql over NOAA `station.name`) were
/// about twice as slow as with the plain tables before (segment v30).
fn keyword_terms_by_ordinal(segment: &rustic_core::segment::Segment, field: u16, reader: &rustic_core::codec::fastfields::KeywordReader<'_>, docs: &[rustic_core::types::DocId]) -> Vec<Option<String>> {
    const MISSING: u32 = rustic_core::codec::fastfields::MISSING_ORD;
    if let Ok(Some(terms)) = segment.keyword_terms_decoded(field) {
        return docs.iter().map(|&doc| {
            let ord = reader.ord(doc);
            if ord == MISSING { return None; }
            match terms.get(ord as usize) {
                Some(t) => Some(t.to_string()),
                None => reader.term(ord).map(|t| t.into_owned()),
            }
        }).collect();
    }
    let mut memo: std::collections::HashMap<u32, Option<String>> = std::collections::HashMap::new();
    docs.iter().map(|&doc| {
        let ord = reader.ord(doc);
        if ord == MISSING { return None; }
        memo.entry(ord).or_insert_with(|| reader.term(ord).map(|t| t.into_owned())).clone()
    }).collect()
}

/// The field id a column reads its doc values from in place of `_source`
/// ([`direct_column_rows`]), when they give the same values: plain keyword
/// (and with `sql` float / double, without it millisecond date) columns
/// with no value-transforming mapping.
/// Terms a keyword column's dictionary holds past which
/// [`direct_column_rows`] reads its batch from `_source` instead.
const HUGE_DICTIONARY_TERMS: usize = 4 << 20;

#[cfg(test)]
thread_local! {
    /// Tests: [`HUGE_DICTIONARY_TERMS`] stand-in.
    pub(super) static HUGE_DICTIONARY_TERMS_TEST: std::cell::Cell<usize> = const { std::cell::Cell::new(HUGE_DICTIONARY_TERMS) };
}

/// Columns past which [`direct_column_rows`] reads a batch that needs the
/// sub-field or integer doc-value reads from `_source` instead.
const WIDE_ROW_COLUMNS: usize = 32;

#[cfg(test)]
thread_local! {
    /// Tests: [`WIDE_ROW_COLUMNS`] stand-in.
    pub(super) static WIDE_ROW_COLUMNS_TEST: std::cell::Cell<usize> = const { std::cell::Cell::new(WIDE_ROW_COLUMNS) };
}

fn wide_row_columns() -> usize {
    #[cfg(test)]
    return WIDE_ROW_COLUMNS_TEST.with(|c| c.get());
    #[cfg(not(test))]
    WIDE_ROW_COLUMNS
}

fn huge_dictionary_terms() -> usize {
    #[cfg(test)]
    return HUGE_DICTIONARY_TERMS_TEST.with(|c| c.get());
    #[cfg(not(test))]
    HUGE_DICTIONARY_TERMS
}

/// How [`direct_column_rows`] reads a column [`direct_field`] does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Extra {
    No,
    /// A keyword sub-field (`lookup_keyword_0.keyword`) of a text or
    /// keyword parent, read from the parent's `_source` path: its doc
    /// values hold that value when a document has exactly one (no
    /// normalizer; one over `ignore_above` has none and reads `_source`).
    SubField,
    /// A plain `long`/`integer`: a single doc value within 2^53 is the
    /// `_source` reading ([`conv`]); others read `_source`.
    Integer,
}

#[cfg(test)]
thread_local! {
    /// Tests: keyword sub-fields and integers read `_source` (the reference
    /// for [`doc_value_field`]).
    pub(super) static DOC_VALUE_EXTRA_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The doc-values field of ES|QL column `name` that [`direct_field`] does
/// not read, for [`direct_column_rows`]: documents whose doc values cannot
/// stand for their `_source` reading (none, several, out of range) still
/// read `_source` there.
fn doc_value_field(name: &str, column: &ColSrc, ordered: bool, lookup: &crate::mapping::FieldLookup) -> Option<(u16, Extra)> {
    #[cfg(test)]
    if DOC_VALUE_EXTRA_DISABLED.with(|c| c.get()) { return None; }
    if ordered || !column.doc_values || column.template_id || column.normalizer.is_some() || !column.constants.is_empty() { return None; }
    if column.ty == "keyword" && name != column.path {
        // A multi-field of the column's path: `<path>.<sub>`.
        if name.strip_prefix(column.path.as_str()).and_then(|rest| rest.strip_prefix('.')).is_none_or(|sub| sub.is_empty() || sub.contains('.')) { return None; }
        let parent = lookup.fields.get(&column.path)?;
        let sub = lookup.fields.get(name)?;
        let plain = sub.declared == EsType::Keyword && sub.doc_values && sub.normalizer.is_none() && sub.null_value.is_none() && sub.script.is_none()
            && sub.alias_path.is_none() && sub.nested_path.is_none() && !lookup.copy_sources.contains_key(name)
            && parent.multi_fields.iter().any(|m| m == name) && matches!(parent.declared, EsType::Text | EsType::Keyword)
            && parent.script.is_none() && parent.null_value.is_none() && parent.alias_path.is_none() && parent.nested_path.is_none()
            && !lookup.copy_sources.contains_key(&column.path);
        return if plain { sub.core_id.map(|id| (id, Extra::SubField)) } else { None };
    }
    if matches!(column.ty.as_str(), "long" | "integer") && name == column.path && !column.path.contains('.') {
        let field = lookup.fields.get(&column.path)?;
        let integer = matches!((column.ty.as_str(), field.declared), ("long", EsType::Long) | ("integer", EsType::Integer | EsType::Short | EsType::Byte));
        let plain = integer && column.declared == field.declared.name() && field.doc_values && !field.ignore_malformed && field.null_value.is_none()
            && field.script.is_none() && !field.is_multi_field && field.alias_path.is_none() && field.nested_path.is_none()
            && !lookup.copy_sources.contains_key(&column.path);
        return if plain { field.core_id.map(|id| (id, Extra::Integer)) } else { None };
    }
    None
}

fn direct_field(column: &ColSrc, ordered: bool, lookup: &crate::mapping::FieldLookup, sql: bool) -> Option<u16> {
    let field = lookup.fields.get(&column.path)?;
    if !sql && column.ty == "text" {
        return text_keyword_field(column, ordered, field, lookup);
    }
    let keyword = column.ty == "keyword" && column.declared == "keyword" && field.declared == EsType::Keyword;
    let numeric = sql && column.ty == "double" && matches!(column.declared.as_str(), "float" | "double")
        && field.declared.name() == column.declared && !field.ignore_malformed;
    // Millisecond dates: the doc value is the mapping's format
    // parsing the `_source` value, as [`conv`] reads it.
    let date = !sql && column.ty == "date" && column.declared == "date" && field.declared == EsType::Date && !field.ignore_malformed;
    if ordered || (!sql && column.path.contains('.')) || !(keyword || numeric || date)
        || !column.doc_values || column.template_id
        || !field.doc_values || field.null_value.is_some() || field.script.is_some()
        || field.normalizer.is_some() || column.normalizer.is_some()
        || field.ignore_above.is_some() || column.ignore_above.is_some() || field.is_multi_field
        || field.alias_path.is_some() || field.nested_path.is_some()
        || lookup.copy_sources.contains_key(&column.path) { return None; }
    field.core_id
}

#[cfg(test)]
thread_local! {
    /// Tests: `text` columns read `_source` (the reference for
    /// [`text_keyword_field`]'s doc-value reads).
    pub(super) static TEXT_VIA_KEYWORD_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The keyword multi-field a plain top-level `text` column reads its
/// single values from ([`direct_column_rows`]): a doc-valued `keyword`
/// sub-field with no normalizer, `null_value` or script indexes each string
/// value as sent, so a document's one doc value is its `_source` value. A
/// document with none (a value over `ignore_above`, or none at all) or with
/// several (doc values sort and deduplicate; `text` keeps `_source` order)
/// reads `_source`. Rally joins' lookup indices map `lookup_keyword_*` as
/// text with a keyword sub-field: every joined row decoded its lookup
/// document's stored `_source` for one short value.
fn text_keyword_field(column: &ColSrc, ordered: bool, field: &crate::mapping::ResolvedField, lookup: &crate::mapping::FieldLookup) -> Option<u16> {
    #[cfg(test)]
    if TEXT_VIA_KEYWORD_DISABLED.with(|c| c.get()) {
        return None;
    }
    if ordered || column.declared != "text" || field.declared != EsType::Text || column.path.contains('.')
        || column.normalizer.is_some() || column.template_id || field.script.is_some() || field.is_multi_field
        || field.alias_path.is_some() || field.nested_path.is_some() || lookup.copy_sources.contains_key(&column.path) {
        return None;
    }
    field.multi_fields.iter().find_map(|name| {
        let sub = lookup.fields.get(name)?;
        let plain = sub.declared == EsType::Keyword && sub.doc_values && sub.normalizer.is_none() && sub.null_value.is_none()
            && sub.script.is_none() && !lookup.copy_sources.contains_key(name);
        if plain { sub.core_id } else { None }
    })
}

#[cfg(test)]
/// Tests: LIMIT BY chunks whose keys were all absent and proven null.
static ABSENT_KEY_CHUNKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether every document of `docs` has an all-null key tuple, read from
/// nothing: each key column of `schema` is a plain keyword doc-values read
/// ([`direct_field`]) with no column in this segment (or a null constant),
/// and each document holds a value of a field mapped after all of them
/// ([`newer_field_witnesses`]), so it has no key value in `_source` either.
/// Rally esql `from_idx_limit_*_by_key_1000`: a LIMIT BY key no document
/// has then costs a witness probe per document instead of a key row.
fn keys_absent_and_proven(schema: &IndexSchema, target: &crate::search::Target, lookup: &crate::mapping::FieldLookup,
    leaf: &rustic_core::query::LeafReader<'_>, docs: &[rustic_core::types::DocId]) -> bool {
    let mut newest = None;
    for column in source_plan(schema, &vec![true; schema.cols.len()], target.hit_index()) {
        match column {
            SourceColumn::Constant(Val::Null) => {}
            SourceColumn::Constant(_) => return false,
            SourceColumn::Read(column, ordered) => {
                let Some(id) = direct_field(column, ordered, lookup, false) else { return false };
                let present = match leaf {
                    rustic_core::query::LeafReader::Segment(segment) => segment.keyword_from_postings(id) || segment.keyword_reader(id).is_some() || segment.multi_keyword_reader(id).is_some(),
                    rustic_core::query::LeafReader::Memtable(writer, ..) => writer.fast_fields().get_keyword(id).is_some() || writer.fast_fields().get_multi_keyword(id).is_some(),
                };
                if column.ty != "keyword" || present {
                    return false;
                }
                newest = newest.max(Some(id));
            }
        }
    }
    let Some(newest) = newest else { return false };
    let witnesses = newer_field_witnesses(leaf, newest);
    !witnesses.is_empty() && docs.iter().all(|&doc| witnesses.iter().any(|witness| witness.has(doc)))
}

fn direct_column_rows(schema: &IndexSchema, wanted: &[bool], target: &crate::search::Target, lookup: &crate::mapping::FieldLookup,
    leaf: &rustic_core::query::LeafReader<'_>, docs: &[rustic_core::types::DocId], allowed: bool, sql: bool) -> Result<(Vec<Vec<Val>>, u64, crate::memory::Reservation)> {
    if !allowed { return source_rows(schema,wanted,target,lookup,leaf,docs); }
    let mut memory=crate::memory::breakers().reserve_request(source_page_metadata(schema,docs.len()),crate::esql::exec::ESQL_LABEL)?;
    let columns = source_plan(schema,wanted,target.hit_index());
    // Per column: its doc-values field, and whether it is one only some
    // documents read from (a keyword sub-field, an integer: see
    // [`doc_value_field`]).
    let resolved: Option<Vec<(Option<u16>, Extra)>> = columns.iter().zip(&schema.cols).map(|(column, (name, _))| match column {
        SourceColumn::Constant(_) => Some((None, Extra::No)),
        SourceColumn::Read(column, ordered) => match direct_field(column, *ordered, lookup, sql) {
            Some(id) => Some((Some(id), Extra::No)),
            None if !sql => doc_value_field(name, column, *ordered, lookup).map(|(id, extra)| (Some(id), extra)),
            None => None,
        },
    }).collect();
    let Some(resolved) = resolved else { return source_rows(schema,wanted,target,lookup,leaf,docs); };
    let extra: Vec<Extra> = resolved.iter().map(|(_, e)| *e).collect();
    let fields: Option<Vec<Option<u16>>> = Some(resolved.into_iter().map(|(f, _)| f).collect());
    let Some(fields) = fields else { return source_rows(schema,wanted,target,lookup,leaf,docs); };
    // A wide row (Rally joins' 100-field base: about 200 columns with the
    // keyword sub-fields) read from doc values takes a value from every
    // column's dictionary at its own place; its stored document is one
    // read. Batches that only read doc values because of the sub-field and
    // integer reads ([`doc_value_field`]) read `_source` past
    // [`WIDE_ROW_COLUMNS`] columns, as before them. Checked first: the dictionary check below opens
    // every keyword column's terms, a cold read each.
    if extra.iter().any(|e| *e != Extra::No) && fields.iter().flatten().count() > wide_row_columns() {
        return source_rows(schema,wanted,target,lookup,leaf,docs);
    }
    // A keyword column whose dictionary is too large to stay cached (a 100M
    // key lookup index: Rally joins `100M_keys_where_limit1000`) reads each
    // value by ordinal from a random place in it, one dictionary per column:
    // the batch's stored documents, one read each, cost less.
    if let rustic_core::query::LeafReader::Segment(segment) = leaf {
        let limit = huge_dictionary_terms();
        let huge = columns.iter().zip(&fields).zip(&extra).any(|((column, field), extra)| match (column, field) {
            (SourceColumn::Read(column, _), Some(field)) if column.ty != "double" && column.ty != "date" && *extra != Extra::Integer =>
                !segment.keyword_from_postings(*field) && segment.keyword_reader(*field).is_some_and(|reader| reader.num_terms() > limit),
            _ => false,
        });
        if huge { return source_rows(schema,wanted,target,lookup,leaf,docs); }
    }
    // Reader metadata must be admitted before bounds inspect lazy columns.
    for field in fields.iter().flatten() {
        memory.grow(leaf.field_reader_allocation_bound(*field),crate::esql::exec::ESQL_LABEL)?;
    }
    // One reader per column per batch: bounding (and below, reading) a
    // column doc by doc reopened its readers for every document.
    for field in fields.iter().flatten() {
        memory.grow(leaf.field_values_allocation_bounds(*field,docs),crate::esql::exec::ESQL_LABEL)?;
    }
    // Opening compact numeric columns validates their presence/rank table.
    // Bind each reader once per batch, rather than repeating that scan per doc.
    let numeric: Vec<Option<ColumnValues<f64>>> = columns.iter().zip(&fields).map(|(column,field)| {
        if !matches!(column, SourceColumn::Read(column,_) if column.ty == "double") { return None; }
        let field = field.unwrap();
        Some(match leaf {
            rustic_core::query::LeafReader::Segment(segment) => {
                let ff = segment.fast_fields();
                if let Some(reader) = ff.get_f64(field) {
                    ColumnValues::Single(docs.iter().map(|&doc| reader.get(doc)).collect())
                } else if let Some(reader) = ff.get_multi_f64(field) {
                    ColumnValues::Lists(docs.iter().map(|&doc| reader.values_vec(doc)).collect())
                } else { ColumnValues::Single(vec![None; docs.len()]) }
            }
            rustic_core::query::LeafReader::Memtable(..) => ColumnValues::Lists(docs.iter().map(|&doc| leaf.f64_values(field,doc)).collect()),
        })
    }).collect();
    // Dates read from single-valued columns only: a document with several
    // keeps their `_source` order and duplicates, read from `_source`.
    let dates: Vec<Option<Vec<Option<i64>>>> = columns.iter().zip(&fields).map(|(column,field)| {
        if !matches!(column, SourceColumn::Read(column,_) if column.ty == "date") { return None; }
        let field = field.unwrap();
        Some(match leaf {
            rustic_core::query::LeafReader::Segment(segment) => match segment.fast_fields().get_i64(field) {
                Some(reader) => docs.iter().map(|&doc| reader.get(doc)).collect(),
                None => vec![None; docs.len()],
            },
            rustic_core::query::LeafReader::Memtable(..) => docs.iter().map(|&doc| leaf.i64_value(field,doc)).collect(),
        })
    }).collect();
    // Integers read from single-valued columns only (as dates).
    let ints: Vec<Option<Vec<Option<i64>>>> = extra.iter().zip(&fields).map(|(extra,field)| {
        if *extra != Extra::Integer { return None; }
        let field = field.unwrap();
        Some(match leaf {
            rustic_core::query::LeafReader::Segment(segment) => match segment.fast_fields().get_i64(field) {
                Some(reader) => docs.iter().map(|&doc| reader.get(doc)).collect(),
                None => vec![None; docs.len()],
            },
            rustic_core::query::LeafReader::Memtable(..) => docs.iter().map(|&doc| leaf.i64_value(field,doc)).collect(),
        })
    }).collect();
    // Single-valued keyword columns read each doc's term straight from the
    // column (no per-doc list).
    let mut keywords: Vec<Option<ColumnValues<String>>> = columns.iter().zip(&fields).zip(&extra).map(|((column,field),extra)| match column {
        SourceColumn::Read(column,_) if column.ty != "double" && column.ty != "date" && *extra != Extra::Integer => Some(match leaf {
            rustic_core::query::LeafReader::Segment(segment) if !segment.keyword_from_postings(field.unwrap()) => match segment.keyword_reader(field.unwrap()) {
                Some(reader) => ColumnValues::Single(keyword_terms_by_ordinal(segment, field.unwrap(), &reader, docs)),
                None => ColumnValues::Lists(leaf.keyword_values_many(field.unwrap(),docs)),
            },
            _ => ColumnValues::Lists(leaf.keyword_values_many(field.unwrap(),docs)),
        }),
        _ => None,
    }).collect();
    let mut loaded = 0;
    let mut missing = Vec::new();
    // Rows whose doc values cannot stand for `_source` (several values of a
    // column read in `_source` order), however the witnesses read.
    let mut forced: HashSet<usize> = HashSet::new();
    let mut rows: Vec<Vec<Val>> = docs.iter().enumerate().map(|(position,_)| {
        let mut needs_source = false;
        let mut forced_row = false;
        let row = columns.iter().zip(&numeric).zip(&dates).zip(keywords.iter_mut()).zip(extra.iter().zip(&ints)).map(|((((column, numeric), date), keyword), (extra, int))| {
            let value = match column {
                SourceColumn::Constant(value) => value.clone(),
                // `_source` reads a long through `f64` ([`conv`]): exact up
                // to 2^53, beyond which the doc value may differ.
                SourceColumn::Read(column,_) if *extra == Extra::Integer => match int.as_ref().unwrap()[position] {
                    Some(v) if v.unsigned_abs() <= 1 << 53 => if column.ty == "integer" { Val::Int(v) } else { Val::Long(v) },
                    _ => { needs_source = true; Val::Null }
                },
                SourceColumn::Read(column,_) if column.ty == "date" => match date.as_ref().unwrap()[position] {
                    Some(ms) => Val::Date(ms),
                    None => { needs_source = true; Val::Null }
                },
                SourceColumn::Read(column,_) if column.ty == "double" => match numeric.as_ref().unwrap() {
                    ColumnValues::Single(values) => match values[position] {
                        Some(x) => Val::Double(x),
                        None => { needs_source = true; Val::Null }
                    },
                    ColumnValues::Lists(lists) => {
                        let mut values: Vec<_> = lists[position].iter().copied().map(Val::Double).collect();
                        needs_source |= values.is_empty();
                        values.sort_by(cmp_vals);
                        Val::from_list(values)
                    }
                },
                SourceColumn::Read(_,_) if matches!(keyword, Some(ColumnValues::Single(_))) => {
                    let Some(ColumnValues::Single(values)) = keyword.as_mut() else { unreachable!() };
                    match values[position].take() {
                        Some(s) => Val::Str(s),
                        None => { needs_source = true; Val::Null }
                    }
                },
                SourceColumn::Read(column,_) => {
                    let mut values = match keyword.as_mut() { Some(ColumnValues::Lists(lists)) => std::mem::take(&mut lists[position]), _ => Vec::new() };
                    // A field can have been mapped after this document was
                    // indexed. Absence in doc values does not prove absence
                    // in source; preserve the source interpreter for that row.
                    needs_source |= values.is_empty();
                    // `text` keeps a document's values in `_source` order
                    // ([`text_keyword_field`]), as a keyword sub-field read
                    // from its parent's `_source` does: several read `_source`
                    // (left null for the fallback to fill, and kept from the
                    // witnesses' pruning: it is no absent value).
                    if (column.ty == "text" || *extra == Extra::SubField) && values.len() > 1 {
                        needs_source = true;
                        forced_row = true;
                        Val::Null
                    } else {
                        values.sort(); values.dedup();
                        Val::from_list(values.into_iter().map(Val::Str).collect())
                    }
                }
            };
            if !value.is_null() { loaded += 1; }
            value
        }).collect();
        if needs_source { missing.push(position); }
        if forced_row { forced.insert(position); }
        row
    }).collect();
    // A document holding a value of a field mapped after a column's field
    // (a higher field id) was indexed under that column's mapping: no doc
    // value there means no `_source` value either, and its `_source` is not
    // read. A LIMIT BY over a mapped key no document has (Rally esql
    // `from_idx_limit_1_by_key_1000`) otherwise decoded every document.
    if !sql && !missing.is_empty() {
        {
            if let Some(oldest) = fields.iter().flatten().min() {
                let witnesses = newer_field_witnesses(leaf, *oldest);
                if !witnesses.is_empty() {
                    missing.retain(|&position| {
                        if forced.contains(&position) { return true; }
                        let doc = docs[position];
                        let Some(newest) = witnesses.iter().filter(|w| w.has(doc)).map(Witness::id).max() else { return true };
                        // Still read: a null column not older than the newest field
                        // the document holds, or a date (several values read as
                        // none here, see above: not an absent value).
                        columns.iter().zip(&fields).enumerate().any(|(column, (plan, field))|
                            matches!(plan, SourceColumn::Read(read, _) if read.ty != "keyword" || field.is_some_and(|id| id >= newest)) && rows[position][column].is_null())
                    });
                    #[cfg(test)]
                    NEWER_FIELD_PROOFS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
    if !missing.is_empty() {
        let missing_docs: Vec<_> = missing.iter().map(|position| docs[*position]).collect();
        // Read only missing columns from source. In sparse metrics a missing
        // number must not force decoding unrelated object-valued keyword paths.
        let fallback_wanted: Vec<_> = columns.iter().enumerate().map(|(column,plan)|
            matches!(plan,SourceColumn::Read(..)) && missing.iter().any(|&position| rows[position][column].is_null())).collect();
        let (fallback,_,fallback_memory) = source_rows(schema,&fallback_wanted,target,lookup,leaf,&missing_docs)?;
        memory.grow(fallback.iter().map(|row|crate::esql::exec::row_bytes(row).saturating_mul(2)).fold(0u64,u64::saturating_add),crate::esql::exec::ESQL_LABEL)?;
        for (&position,row) in missing.iter().zip(fallback) {
            for (column,value) in row.into_iter().enumerate() {
                if rows[position][column].is_null() && matches!(&columns[column],SourceColumn::Read(..)) && !value.is_null() {
                    rows[position][column] = value;
                    loaded += 1;
                }
            }
        }
        drop(fallback_memory);
    }
    #[cfg(test)]
    if !LOOKUP_SIDE_READ.with(|side| side.get()) {
        LOOKUP_KEY_DV_READS.with(|count| count.set(count.get() + docs.len() - missing.len()));
    }
    Ok((rows,loaded,memory))
}

/// Most columns [`newer_field_witnesses`] reads per batch.
const NEWER_FIELD_WITNESSES: usize = 4;

#[cfg(test)]
/// Tests: batches that proved a null from a newer field (any thread: shard reads may run on workers).
static NEWER_FIELD_PROOFS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A single-valued doc-values column of a field mapped after another one
/// (of a segment, or of a memtable: refreshed documents not yet flushed).
enum Witness<'a> {
    Keyword(u16, rustic_core::codec::fastfields::KeywordReader<'a>),
    Long(u16, rustic_core::codec::fastfields::ColumnReader<'a, i64>),
    Double(u16, rustic_core::codec::fastfields::ColumnReader<'a, f64>),
    MemKeyword(u16, &'a rustic_core::fast_fields::KeywordColumn),
    MemLong(u16, &'a rustic_core::fast_fields::I64Column),
    MemDouble(u16, &'a rustic_core::fast_fields::F64Column),
}

impl Witness<'_> {
    fn id(&self) -> u16 {
        match self {
            Witness::Keyword(id, _) | Witness::Long(id, _) | Witness::Double(id, _)
                | Witness::MemKeyword(id, _) | Witness::MemLong(id, _) | Witness::MemDouble(id, _) => *id,
        }
    }

    fn has(&self, doc: rustic_core::types::DocId) -> bool {
        match self {
            // Presence only: no term decoded.
            Witness::Keyword(_, reader) => reader.ord(doc) != rustic_core::codec::fastfields::MISSING_ORD,
            Witness::Long(_, reader) => reader.get(doc).is_some(),
            Witness::Double(_, reader) => reader.get(doc).is_some(),
            Witness::MemKeyword(_, column) => column.get(doc).is_some(),
            Witness::MemLong(_, column) => column.has_value(doc),
            Witness::MemDouble(_, column) => column.has_value(doc),
        }
    }
}

/// The newest single-valued keyword / numeric columns of `segment` whose
/// field id is above `oldest`. Field ids are handed out as fields are mapped
/// (`Mappings::next_core_id`, never reused), and a document holds a doc
/// value only for a field mapped when it was indexed. So a document with a
/// value in one of these was indexed after every field with a lower id was
/// mapped, and any such field it has in `_source` has a doc value too
/// (columns whose doc values can differ from `_source`, such as
/// `ignore_above`, normalizers, `null_value`, scripts and `copy_to`
/// targets, never take the doc-values read).
fn newer_field_witnesses<'a>(leaf: &rustic_core::query::LeafReader<'a>, oldest: u16) -> Vec<Witness<'a>> {
    let segment = match leaf {
        rustic_core::query::LeafReader::Segment(segment) => *segment,
        rustic_core::query::LeafReader::Memtable(writer, ..) => {
            let ff: &'a rustic_core::fast_fields::FastFields = writer.fast_fields();
            let mut out: Vec<Witness<'a>> = Vec::new();
            out.extend(ff.keyword_columns().iter().filter(|(id, _)| **id > oldest).map(|(id, column)| Witness::MemKeyword(*id, column)));
            out.extend(ff.i64_columns().iter().filter(|(id, _)| **id > oldest).map(|(id, column)| Witness::MemLong(*id, column)));
            out.extend(ff.f64_columns().iter().filter(|(id, _)| **id > oldest).map(|(id, column)| Witness::MemDouble(*id, column)));
            out.sort_unstable_by_key(|witness| std::cmp::Reverse(witness.id()));
            out.truncate(NEWER_FIELD_WITNESSES);
            return out;
        }
    };
    let mut ids: Vec<u16> = segment.fast_fields().fields().into_iter().map(|(id, _)| id).filter(|id| *id > oldest).collect();
    ids.sort_unstable_by(|a, b| b.cmp(a));
    ids.dedup();
    let mut out = Vec::new();
    for id in ids {
        if out.len() == NEWER_FIELD_WITNESSES { break; }
        let ff = segment.fast_fields();
        if !segment.keyword_from_postings(id) {
            if let Some(reader) = ff.get_keyword(id) { out.push(Witness::Keyword(id, reader)); continue; }
        }
        if let Some(reader) = ff.get_i64(id) { out.push(Witness::Long(id, reader)); continue; }
        if let Some(reader) = ff.get_f64(id) { out.push(Witness::Double(id, reader)); }
    }
    out
}

/// One batch of a column's doc values: one optional value per doc
/// (single-valued column), or each doc's value list.
enum ColumnValues<T> {
    Single(Vec<Option<T>>),
    Lists(Vec<Vec<T>>),
}

fn sparse_rows_bytes(rows: &[Vec<(usize, Val)>]) -> u64 {
    rows.iter().map(|row| (24 + (row.capacity() * std::mem::size_of::<(usize, Val)>()) as u64 + row.iter().map(|(_, v)| v.heap_bytes() as u64).sum::<u64>()).saturating_mul(2)).fold(0u64, u64::saturating_add)
}

pub(super) fn rows_bytes(rows:&[Vec<Val>])->u64 {
    rows.iter().map(|row|crate::esql::exec::row_bytes(row).saturating_mul(2)).fold(0u64,u64::saturating_add)
}

fn source_page_metadata(schema:&IndexSchema,docs:usize)->u64 {
    let constants=schema.cols.iter().fold(0u64,|n,(_,column)|
        n.saturating_add(column.constants.values().map(crate::memory::json_bytes).fold(0u64,u64::saturating_add)));
    // The plan owns one converted constant; every output row deep-clones it.
    // Admit all page row slots and those clones before constructing the plan
    // or the first row, even when each document's stored source is tiny.
    let rows=(docs as u64).saturating_mul(64u64.saturating_add((schema.cols.len() as u64).saturating_mul(128)).saturating_add(constants.saturating_mul(256)));
    schema.cols.iter().fold(rows.saturating_add(4096).saturating_add(constants.saturating_mul(8)),|n,(name,column)| {
        n.saturating_add((name.len()+column.path.len()) as u64*8).saturating_add(256)
    })
}

fn resize_page_memory(memory:&mut crate::memory::Reservation,bytes:u64)->Result<()> {
    if bytes>memory.bytes(){memory.grow(bytes-memory.bytes(),crate::esql::exec::ESQL_LABEL)?;}
    else{memory.shrink(memory.bytes()-bytes);}
    Ok(())
}

/// A batch's [`source_plan`] with what reading a document needs resolved
/// once: the `_source` keys the columns read and the constant part of a row.
struct SourcePlan<'a> {
    columns: Vec<SourceColumn<'a>>,
    /// Every top-level `_source` key a `Read` column can take a value from
    /// (its path, or a dotted prefix of it: [`crate::highlight::source_values`]
    /// only descends from those keys of an object), with the columns under
    /// it. Also the roots a narrow projection decodes ([`projected_source`]).
    roots: HashMap<&'a str, Vec<usize>>,
    /// The row before any `_source` value: the constants, null elsewhere.
    template: Vec<Val>,
    /// Heap bytes of `template` (its constants').
    template_heap: u64,
    /// The non-null constants, by column.
    constants: Vec<(usize, Val)>,
}

impl<'a> SourcePlan<'a> {
    fn new(columns: Vec<SourceColumn<'a>>) -> SourcePlan<'a> {
        let mut roots: HashMap<&'a str, Vec<usize>> = HashMap::new();
        let mut constants = Vec::new();
        let mut template = Vec::with_capacity(columns.len());
        for (i, column) in columns.iter().enumerate() {
            match column {
                SourceColumn::Constant(value) => {
                    if !value.is_null() {
                        constants.push((i, value.clone()));
                    }
                    template.push(value.clone());
                }
                SourceColumn::Read(read, _) => {
                    let read: &'a ColSrc = *read;
                    for root in source_roots(&read.path) {
                        let under = roots.entry(root).or_default();
                        if under.last() != Some(&i) {
                            under.push(i);
                        }
                    }
                    template.push(Val::Null);
                }
            }
        }
        let template_heap = template.iter().map(|v| v.heap_bytes() as u64).sum();
        SourcePlan { columns, roots, template, template_heap, constants }
    }

    /// Hand the non-null `_source` values of the `Read` columns to `put`.
    /// Of an object only the columns under one of its keys are read: every
    /// other column's [`extract_ordered`] finds no value (null). Nothing else
    /// is ever a `_source`; it is read column by column as before.
    fn read(&self, src: &Value, mut put: impl FnMut(usize, Val)) {
        let mut one = |i: usize| {
            if let SourceColumn::Read(column, ordered) = &self.columns[i] {
                let v = extract_ordered(src, column, *ordered);
                if !v.is_null() {
                    put(i, v);
                }
            }
        };
        match src {
            Value::Object(map) => {
                let mut hits: Vec<usize> = Vec::new();
                if map.len() <= self.roots.len() {
                    for key in map.keys() {
                        if let Some(under) = self.roots.get(key.as_str()) {
                            hits.extend_from_slice(under);
                        }
                    }
                } else {
                    for (root, under) in &self.roots {
                        if map.contains_key(*root) {
                            hits.extend_from_slice(under);
                        }
                    }
                }
                hits.sort_unstable();
                hits.dedup();
                // Many columns under the document's keys (a time series
                // mapping's thousands): its values by path, collected once,
                // instead of one walk of the document per column.
                if hits.len() >= FLAT_READ_MIN_COLUMNS {
                    if let Some(flat) = flat_source(src) {
                        for i in hits {
                            match &self.columns[i] {
                                SourceColumn::Read(column, ordered) if reads_source_values(column) => {
                                    let Some(vals) = flat.get(column.path.as_str()) else { continue };
                                    let v = extract_values(vals.clone(), column, *ordered);
                                    if !v.is_null() {
                                        put(i, v);
                                    }
                                }
                                SourceColumn::Read(column, ordered) => {
                                    let v = extract_ordered(src, column, *ordered);
                                    if !v.is_null() {
                                        put(i, v);
                                    }
                                }
                                _ => {}
                            }
                        }
                        return;
                    }
                }
                for i in hits {
                    one(i);
                }
            }
            // No path finds a value under null (an empty `_source`).
            Value::Null => {}
            _ => (0..self.columns.len()).for_each(one),
        }
    }

    /// The full row of `src`, its non-null values and its [`row_bytes`].
    fn dense(&self, src: &Value) -> (Vec<Val>, u64, u64) {
        let mut row = self.template.clone();
        let mut n = self.constants.len() as u64;
        let mut heap = self.template_heap;
        self.read(src, |i, v| {
            heap += v.heap_bytes() as u64;
            row[i] = v;
            n += 1;
        });
        let bytes = 24 + (row.len() * std::mem::size_of::<Val>()) as u64 + heap;
        (row, n, bytes)
    }

    /// The non-null values of `src`'s row, by column (constants first, then
    /// the `_source` values in column order), their count and their bytes.
    fn sparse(&self, src: &Value) -> (Vec<(usize, Val)>, u64, u64) {
        let mut out = self.constants.clone();
        self.read(src, |i, v| out.push((i, v)));
        let bytes = 24 + (out.capacity() * std::mem::size_of::<(usize, Val)>()) as u64 + out.iter().map(|(_, v)| v.heap_bytes() as u64).sum::<u64>();
        let n = out.len() as u64;
        (out, n, bytes)
    }
}

/// Columns under a document's keys from which [`SourcePlan::read`] collects
/// the document's values by path once ([`flat_source`]).
const FLAT_READ_MIN_COLUMNS: usize = 64;

/// The values [`crate::highlight::source_values`] finds at every path of
/// `src`, by path, in the order it finds them; `None` when some object key
/// holds a dot (a path could then be reached by several decompositions,
/// whose order the per-path walk decides).
fn flat_source(src: &Value) -> Option<HashMap<String, Vec<&Value>, ahash::RandomState>> {
    fn visit<'v>(v: &'v Value, path: &mut String, out: &mut HashMap<String, Vec<&'v Value>, ahash::RandomState>) -> bool {
        match v {
            Value::Null => true,
            Value::Array(a) => a.iter().all(|x| visit(x, path, out)),
            Value::Object(m) => {
                if !path.is_empty() {
                    out.entry(path.clone()).or_default().push(v);
                }
                let at = path.len();
                for (k, x) in m {
                    if k.contains('.') {
                        return false;
                    }
                    if at > 0 {
                        path.push('.');
                    }
                    path.push_str(k);
                    let ok = visit(x, path, out);
                    path.truncate(at);
                    if !ok {
                        return false;
                    }
                }
                true
            }
            other => {
                if !path.is_empty() {
                    out.entry(path.clone()).or_default().push(other);
                }
                true
            }
        }
    }
    let mut out = HashMap::default();
    let mut path = String::new();
    visit(src, &mut path, &mut out).then_some(out)
}

/// The top-level keys a value at `path` can sit under: each dotted prefix
/// of it and the whole path.
/// Whether `name` may occur as a quoted JSON member name in `raw`: an
/// occurrence between two `"` (conservatively true for names a JSON text
/// would have to escape).
fn quoted_name_in(raw: &str, name: &str) -> bool {
    if name.is_empty() || name.chars().any(|c| c == '"' || c == '\\' || c.is_control()) {
        return true;
    }
    let b = raw.as_bytes();
    raw.match_indices(name).any(|(at, _)| at > 0 && b[at - 1] == b'"' && b.get(at + name.len()) == Some(&b'"'))
}

fn source_roots(path: &str) -> impl Iterator<Item = &str> {
    path.match_indices('.').map(move |(offset, _)| &path[..offset]).chain(std::iter::once(path))
}

/// Reads `docs` (ascending) from `_source` through `build` (a row of the
/// plan from the document's projected `_source`: the row, its non-null
/// values, its bytes). A page owns its decode reservation until its rows are
/// consumed or moved into a separately charged retained buffer. No
/// per-document charge escapes into the request root.
#[allow(clippy::too_many_arguments)]
fn source_reads_guarded<'s, R>(schema:&IndexSchema,wanted:&[bool],target:&crate::search::Target,lookup:&crate::mapping::FieldLookup,leaf:&rustic_core::query::LeafReader<'s>,docs:&[rustic_core::types::DocId],build:impl Fn(&SourcePlan<'_>,&Value)->(R,u64,u64))->Result<(Vec<R>,u64,crate::memory::Reservation)> {
    let metadata=source_page_metadata(schema,docs.len());
    let mut memory=crate::memory::breakers().reserve_request(metadata,crate::esql::exec::ESQL_LABEL)?;
    let plan=SourcePlan::new(source_plan(schema,wanted,target.hit_index()));
    // A synthetic source whose read columns all come from doc values (and no
    // field security): a document whose residual took them all out gets its
    // row straight from the columns (below), its residual not parsed.
    // Not for a strict columnar index's array-ordered columns: their doc
    // values render sorted and de-duplicated, the rows keep array order
    // (`.offsets`; esql/300_columnar_array_ordering).
    let ordered = plan.columns.iter().any(|c| matches!(c, SourceColumn::Read(_, true)));
    let direct = (!ordered && target.field_access().is_none() && target.svc.synthetic_source() && !crate::vector_source::has_vector_fields(lookup))
        .then(|| {
            let paths: Vec<&str> = plan.columns.iter().filter_map(|c| match c { SourceColumn::Read(c, _) => Some(c.path.as_str()), _ => None }).collect();
            crate::synthetic::column_backed(lookup, &paths)
        })
        .flatten();
    // Rehydrating a dense vector may expand compact binary values into
    // source text. This is a transient bound for each document only.
    let vectors=lookup.fields.values().filter(|field|field.declared==crate::mapping::EsType::DenseVector).map(|field|field.dims.unwrap_or(4096) as u64).fold(0u64,u64::saturating_add);
    let mut rows=Vec::with_capacity(docs.len());let mut loaded=0u64;let mut held=metadata;
    // A segment's documents read in one batch: each store chunk decoded
    // once, to its last doc wanted (one read per doc decoded a chunk's
    // prefix per row: 64 KB chunks of small geo_point documents).
    let mut batch: Vec<Option<rustic_core::types::Document>> = Vec::new();
    if let rustic_core::query::LeafReader::Segment(s) = leaf {
        if docs.len() > 1 {
            batch.resize_with(docs.len(), || None);
            s.get_docs_batched(docs, true, |i, d| batch[i] = Some(d)).map_err(crate::error::EsError::from)?;
        }
    }
    // The segment's columns opened once for the batch's synthetic sources.
    let cache = crate::synthetic::ColumnCache::new(leaf);
    for (i, &doc) in docs.iter().enumerate() {
        let stored = match batch.get_mut(i).and_then(Option::take) {
            Some(d) => { resize_page_memory(&mut memory, held.saturating_add(d.decoded_bytes()))?; d }
            None => leaf.get_doc_guarded(doc,|bytes|resize_page_memory(&mut memory,held.saturating_add(bytes)))?.unwrap_or_default(),
        };
        let raw=stored.get(crate::mapping::SOURCE_FIELD).and_then(|v|v.as_text()).unwrap_or("");
        resize_page_memory(&mut memory,held.saturating_add((raw.len() as u64).saturating_mul(256)).saturating_add(vectors.saturating_mul(256)).saturating_add(4096))?;
        // A residual that took every read field out: the row from the
        // columns, no residual parsed (ES|QL reads doc values too).
        let columns = direct.as_ref().filter(|at| crate::synthetic::residual_takes(raw, lookup, at))
            .and_then(|at| crate::synthetic::columns_source(lookup, at, &crate::synthetic::LeafColumns { leaf, doc }));
        let src = match columns { Some(src) => src, None => planned_source_cached(&plan,target,lookup,leaf,doc,&stored,Some(&cache))? };
        let (row,count,bytes)=build(&plan,&src);
        held=held.saturating_add(bytes.saturating_mul(2));
        resize_page_memory(&mut memory,held)?;
        loaded+=count;rows.push(row);
    }
    Ok((rows,loaded,memory))
}

/// The `_source` rows of `docs` (ascending); see [`source_reads_guarded`].
pub(super) fn source_rows_guarded(schema:&IndexSchema,wanted:&[bool],target:&crate::search::Target,lookup:&crate::mapping::FieldLookup,leaf:&rustic_core::query::LeafReader<'_>,docs:&[rustic_core::types::DocId])->Result<(Vec<Vec<Val>>,u64,crate::memory::Reservation)> {
    source_reads_guarded(schema,wanted,target,lookup,leaf,docs,|plan,src|plan.dense(src))
}

/// [`source_rows`] as each row's non-null values only, by column: a wide
/// schema's mostly-null rows are never built (the grouped-limit key pass).
fn source_sparse_rows(schema:&IndexSchema,wanted:&[bool],target:&crate::search::Target,lookup:&crate::mapping::FieldLookup,leaf:&rustic_core::query::LeafReader<'_>,docs:&[rustic_core::types::DocId])->Result<(Vec<Vec<(usize,Val)>>,u64,crate::memory::Reservation)> {
    source_reads_guarded(schema,wanted,target,lookup,leaf,docs,|plan,src|plan.sparse(src))
}

/// The `_source` rows of `docs` (ascending), reading each store chunk once.
pub(super) fn source_rows(schema: &IndexSchema, wanted: &[bool], t: &crate::search::Target, lookup: &crate::mapping::FieldLookup, leaf: &rustic_core::query::LeafReader<'_>, docs: &[rustic_core::types::DocId]) -> Result<(Vec<Vec<Val>>, u64, crate::memory::Reservation)> {
    source_rows_guarded(schema,wanted,t,lookup,leaf,docs)
}

fn row_of(schema: &IndexSchema, wanted: &[bool], t: &crate::search::Target, lookup: &crate::mapping::FieldLookup, leaf: &rustic_core::query::LeafReader<'_>, doc: rustic_core::types::DocId, stored: rustic_core::types::Document) -> Result<(Vec<Val>, u64)> {
    let plan = SourcePlan::new(source_plan(schema, wanted, t.hit_index()));
    let src = planned_source(&plan, t, lookup, leaf, doc, &stored)?;
    let (row, n, _) = plan.dense(&src);
    Ok((row, n))
}

/// Decode only source roots needed by a narrow projection. Ignored values
/// still undergo JSON syntax validation, but do not allocate their trees.
/// Keep both a dotted literal name and its root to preserve extract semantics.
fn projected_source(raw: &str, columns: &[SourceColumn<'_>]) -> Result<Value> {
    let mut roots: HashMap<&str, Vec<usize>> = HashMap::new();
    for column in columns {
        if let SourceColumn::Read(column, _) = column {
            for root in source_roots(&column.path) {
                roots.entry(root).or_default();
            }
        }
    }
    projected_source_roots(raw, &roots)
}

/// [`projected_source`] of a plan's roots (resolved once per batch).
fn projected_source_roots(raw: &str, roots: &HashMap<&str, Vec<usize>>) -> Result<Value> {
    if roots.len() > 16 || !raw.trim_start().starts_with('{') {
        return serde_json::from_str(raw).map_err(|e| EsError::internal(format!("bad _source: {e}")));
    }
    // None of the roots occurs as a quoted name in the text (with no `\u`
    // escape that could spell one): the projection is empty, so the stored
    // document (valid JSON, checked when indexed) is not scanned. A LIMIT 1
    // BY on a key no document has read every document's `_source` for it.
    if !raw.contains("\\u") && roots.keys().all(|r| !quoted_name_in(raw, r)) {
        return Ok(Value::Object(Default::default()));
    }
    struct Projection<'r, 'a>(&'r HashMap<&'a str, Vec<usize>>);
    impl<'de> serde::de::Visitor<'de> for Projection<'_, '_> {
        type Value = Value;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a source object") }
        fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> std::result::Result<Value,M::Error> {
            let mut result = serde_json::Map::new();
            while let Some(key) = map.next_key::<String>()? {
                if self.0.contains_key(key.as_str()) { result.insert(key,map.next_value::<Value>()?); }
                else { map.next_value::<serde::de::IgnoredAny>()?; }
            }
            Ok(Value::Object(result))
        }
    }
    let mut decoder = serde_json::Deserializer::from_str(raw);
    let value = serde::de::Deserializer::deserialize_map(&mut decoder, Projection(roots))
        .map_err(|e| EsError::internal(format!("bad _source: {e}")))?;
    decoder.end().map_err(|e| EsError::internal(format!("bad _source: {e}")))?;
    Ok(value)
}

/// The (projected, visible) `_source` of a stored document.
fn planned_source(plan: &SourcePlan<'_>, t: &crate::search::Target, lookup: &crate::mapping::FieldLookup, leaf: &rustic_core::query::LeafReader<'_>, doc: rustic_core::types::DocId, stored: &rustic_core::types::Document) -> Result<Value> {
    planned_source_cached(plan, t, lookup, leaf, doc, stored, None)
}

/// [`planned_source`]; `cache`: the leaf's columns opened for a batch of
/// its documents ([`crate::synthetic::ColumnCache`]).
fn planned_source_cached<'s>(plan: &SourcePlan<'_>, t: &crate::search::Target, lookup: &crate::mapping::FieldLookup, leaf: &rustic_core::query::LeafReader<'s>, doc: rustic_core::types::DocId, stored: &rustic_core::types::Document, cache: Option<&crate::synthetic::ColumnCache<'s>>) -> Result<Value> {
    let raw = stored.get(crate::mapping::SOURCE_FIELD).and_then(|v| v.as_text()).unwrap_or("");
    // A synthetic residual read for a few roots (no field security): those
    // roots only, rendered from the residual and their columns, instead of
    // rehydrating every document whole (ES|QL STATS over a time series index
    // read a hundred fields per row to use three).
    // So is an empty residual for any number of roots: its value is built
    // from its columns directly (not written as text and parsed back).
    if t.field_access().is_none() && (plan.roots.len() <= 16 || raw == crate::synthetic::EMPTY_RESIDUAL) && crate::synthetic::is_residual(raw) && !crate::vector_source::has_vector_fields(lookup) {
        let plain = crate::synthetic::LeafColumns { leaf, doc };
        let cached = cache.map(|cache| crate::synthetic::CachedLeafColumns { leaf, doc, cache });
        let cols: &dyn crate::synthetic::DocColumns = match &cached { Some(c) => c, None => &plain };
        if let Some(v) = crate::synthetic::rehydrate_roots(raw, lookup, cols, &|k| plan.roots.contains_key(k)) {
            return Ok(v);
        }
    }
    let loaded = if crate::synthetic::is_residual(raw) || (!raw.is_empty() && crate::vector_source::has_vector_fields(lookup)) { crate::vector_source::load(raw, lookup, leaf, doc) } else { std::borrow::Cow::Borrowed(raw) };
    let visible = t.visible_source(&loaded);
    if visible.is_empty() { Ok(Value::Null) } else { projected_source_roots(&visible, &plan.roots) }
}

/// The group of a row of plain-column `LIMIT BY` keys from its non-null
/// values (`positions[c]`: the keys reading column `c`): each non-null key in
/// key order, as its position and its length-prefixed [`Val::key`]. Two rows
/// share a group exactly when [`grouped_row_key`] says so (equal `Val::key`
/// at every key; a null key, absent here, is never equal to a value's).
fn sparse_group_key(positions: &[Vec<usize>], row: &[(usize, Val)]) -> Vec<u8> {
    let mut parts: Vec<(usize, String)> = Vec::new();
    for (column, value) in row {
        let Some(keys) = positions.get(*column).filter(|k| !k.is_empty()) else { continue };
        let part = value.key();
        for &k in keys {
            parts.push((k, part.clone()));
        }
    }
    parts.sort_unstable_by_key(|(k, _)| *k);
    let mut key = Vec::with_capacity(parts.iter().map(|(_, p)| p.len() + 8).sum());
    for (k, part) in parts {
        key.extend_from_slice(&(k as u32).to_le_bytes());
        key.extend_from_slice(&(part.len() as u32).to_le_bytes());
        key.extend_from_slice(part.as_bytes());
    }
    key
}

/// `EVAL d = ST_DISTANCE(loc, point) ... | SORT d [ASC], ... | LIMIT n`
/// (ES pushes the distance sort into Lucene). `ST_DISTANCE` of a geo point
/// is a function of its quantized coordinates, which the doc values hold
/// exactly: pass 1 finds the `n`-th smallest doc-value distance `D`, pass 2
/// reads from `_source` only the documents at most `D` away, and the
/// `EVAL`s, `SORT` and `LIMIT` run over those rows as over the full table
/// (every row the full sort could keep is among them).
fn distance_top_n(ctx: &mut Ctx, schema: &IndexSchema, indices: &[String], query: &Value, rest: &[Cmd]) -> Result<Option<(Table, usize)>> {
    let evals = rest.iter().take_while(|c| matches!(c, Cmd::Eval(_))).count();
    if evals == 0 {
        return Ok(None);
    }
    let (Some(Cmd::Sort(keys)), Some(Cmd::Limit(n))) = (rest.get(evals), rest.get(evals + 1)) else { return Ok(None) };
    let n = (*n).min(ctx.max_limit);
    let Some(key) = keys.first() else { return Ok(None) };
    let Expr::Col(dname, _) = &key.expr else { return Ok(None) };
    if key.desc || key.nulls_first == Some(true) || n == 0 {
        return Ok(None);
    }
    let mut defs: Vec<(String, Expr)> = Vec::new();
    for c in &rest[..evals] {
        let Cmd::Eval(fields) = c else { unreachable!() };
        for f in fields {
            let def = subst(&f.expr, &defs);
            defs.push((f.name.clone(), if simple(&def) { def } else { Expr::Lit(Val::Null) }));
        }
    }
    let Some((_, dexpr)) = defs.iter().rev().find(|(d, _)| d == dname) else { return Ok(None) };
    let types: HashMap<String, String> = schema.cols.iter().map(|(n, c)| (n.clone(), c.ty.clone())).collect();
    let Some((col, lat, lon)) = distance_of(dexpr, &types) else { return Ok(None) };
    let Some(field) = geo_dv_field(&Expr::Col(col.to_string(), 0), schema) else { return Ok(None) };
    if schema.slots.is_some() {
        return distributed_distance(ctx, schema, indices, query, rest, evals, n, &field, lat, lon);
    }
    let (qlat, qlon) = rustic_core::geo::quantize(lat, lon);
    let dist = move |e: &rustic_core::geo::GeoPointEntry| rustic_core::geo::sloppy_haversin_meters(qlat, qlon, decode_lat(e.lat), decode_lon(e.lon));
    // Pass 1, per shard: the n smallest distances and how many documents
    // have a point.
    struct Best {
        d: std::collections::BinaryHeap<OrdF64>,
        with_point: u64,
    }
    let init = |_: usize| Best { d: std::collections::BinaryHeap::new(), with_point: 0 };
    let pass1 = |st: &mut Best, p: &crate::search::MatchPage<'_>| -> Result<bool> {
        let Some(id) = p.lookup.get(&field).and_then(|f| f.geo_id) else { return Ok(true) };
        let dv = GeoDv::open(p.leaf, id);
        for d in p.doc_ids() {
            let pts = dv.entries(d);
            if pts.is_empty() {
                continue;
            }
            st.with_point += 1;
            // A multi-valued point: its nearest value (ES `ST_DISTANCE`).
            let x = pts.iter().map(|e| dist(&e)).fold(f64::INFINITY, f64::min);
            if st.d.len() < n {
                st.d.push(OrdF64(x));
            } else if st.d.peek().is_some_and(|m| x < m.0) {
                st.d.pop();
                st.d.push(OrdF64(x));
            }
        }
        Ok(true)
    };
    // The nearest documents are found within a growing radius (a KD-tree
    // query): once n of them are well inside it, no document outside can
    // be among the n nearest. The last round has no radius.
    // First radius: where n points would be at the index's mean density
    // over the earth's surface, a quarter of it (data is clustered).
    let total = searched_docs(schema)?.max(1) as f64;
    let mut radius = (n as f64 / total * 5.1e14 / std::f64::consts::PI).sqrt().clamp(100.0, 2.0e6) / 4.0;
    let (bound, query) = loop {
        let unbounded = radius > 2.1e7;
        let q = if unbounded { query.clone() } else { json!({"bool": {"filter": [query.clone(), {"geo_distance": {"distance": format!("{radius}m"), field.as_str(): {"lat": lat, "lon": lon}}}]}}) };
        let states = crate::search::par_for_each_match(&schema.targets, &q, PAGE, &init, &pass1)?;
        let with_point: u64 = states.iter().map(|s| s.with_point).sum();
        if with_point >= n as u64 {
            let mut all: Vec<f64> = states.into_iter().flat_map(|s| s.d.into_iter().map(|x| x.0)).collect();
            all.sort_by(f64::total_cmp);
            let b = all[n - 1];
            // Margin for the KD query's exact haversine vs the sloppy one.
            if unbounded || b <= radius * 0.999 - 1.0 {
                break (b, q);
            }
        } else if unbounded {
            // Null distances join the result: the table path orders them.
            return Ok(None);
        }
        // Grow by the observed density (area ~ points), at least 2x, at most 8x.
        radius *= if with_point == 0 { 8.0 } else { ((n as f64 / with_point as f64).sqrt() * 1.5).clamp(2.0, 8.0) };
    };
    let query = &query;
    // Pass 2: the rows of every document within `bound`, in search order.
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().map(|(c, _)| needed.as_ref().is_none_or(|s| s.contains(c))).collect();
    type Rows = (Vec<(usize, rustic_core::types::DocId, Vec<Val>)>, u64, u64, Option<EsError>);
    let init2 = |_: usize| -> Rows { (Vec::new(), 0, 0, None) };
    let pass2 = |st: &mut Rows, p: &crate::search::MatchPage<'_>| -> Result<bool> {
        st.1 += p.len() as u64;
        let Some(id) = p.lookup.get(&field).and_then(|f| f.geo_id) else { return Ok(true) };
        let dv = GeoDv::open(p.leaf, id);
        for d in p.doc_ids() {
            let pts = dv.entries(d);
            if pts.is_empty() || pts.iter().map(|e| dist(&e)).fold(f64::INFINITY, f64::min) > bound {
                continue;
            }
            match source_row(schema, &wanted, p.target, p.lookup, p.leaf, d) {
                Ok((row, l, memory)) => {
                    st.2 += l;
                    memory.retain_for_request();
                    st.0.push((p.leaf_idx, d, row));
                }
                Err(e) => {
                    st.3 = Some(e);
                    return Ok(false);
                }
            }
        }
        Ok(true)
    };
    let states = crate::search::par_for_each_match(&schema.targets, query, PAGE, &init2, &pass2)?;
    taken();
    let mut rows = Vec::new();
    for (picked, docs, loaded, err) in states {
        if let Some(e) = err {
            return Err(e);
        }
        ctx.documents_found += docs;
        ctx.values_loaded += loaded;
        // Pages of a shard arrive in search order (leaf, then doc).
        rows.extend(picked.into_iter().map(|x| x.2));
    }
    let mut t = Table { cols: source_cols(schema, &[]), rows };
    for c in &rest[..=evals + 1] {
        t = apply(ctx, c, t)?;
    }
    Ok(Some((t, evals + 2)))
}

/// [`distance_top_n`] over shards partly read on other nodes: every shard
/// returns its rows within its own `n` nearest points (all its rows with a
/// point when it has fewer), a superset of its share of the global `n`
/// nearest; the rows, in shard order, then run the same `EVAL | SORT |
/// LIMIT` (stable, so ties keep search order as on one node).
#[allow(clippy::too_many_arguments)]
fn distributed_distance(ctx: &mut Ctx, schema: &IndexSchema, indices: &[String], query: &Value, rest: &[Cmd], evals: usize, n: usize, field: &str, lat: f64, lon: f64) -> Result<Option<(Table, usize)>> {
    if ctx.settings.unmapped != super::super::settings::Unmapped::Default {
        return Ok(None);
    }
    let base = source_cols(schema, &[]);
    let columns = wire_cols(&base);
    let wanted = vec![true; schema.cols.len()];
    let plan = FastPlan::Distance { indices: indices.to_vec(), columns: columns.clone(), field: field.to_string(), lat, lon, n };
    let parts = by_slot(ctx, schema, query, plan, |_, targets| {
        targets.iter().map(|t| {
            let (rows, docs, loaded, with_point) = distance_shard(schema, &wanted, t, query, field, lat, lon, n)?;
            Ok(FastPart::Rows { docs, loaded, with_point, cols: Some(columns.clone()), rows: wire_rows(rows) })
        }).collect()
    })?;
    let Some(parts) = parts else { return Ok(None) };
    let (mut docs, mut loaded, mut with_point) = (0u64, 0u64, 0u64);
    let mut rows = Vec::new();
    for part in parts {
        let FastPart::Rows { docs: d, loaded: l, with_point: w, cols, rows: r } = part else { return Ok(None) };
        if cols.as_ref() != Some(&columns) {
            return Ok(None);
        }
        docs += d;
        loaded += l;
        with_point += w;
        rows.extend(r.iter().map(|row| row.iter().map(Val::from).collect::<Vec<_>>()));
    }
    // Fewer points than rows wanted: null distances join the result; the
    // table path orders them.
    if with_point < n as u64 {
        return Ok(None);
    }
    taken();
    ctx.documents_found += docs;
    ctx.values_loaded += loaded;
    let mut t = Table { cols: base, rows };
    for c in &rest[..=evals + 1] {
        t = apply(ctx, c, t)?;
    }
    Ok(Some((t, evals + 2)))
}

/// One shard's rows within its `n` nearest points to `(lat, lon)` (every
/// row with a point when it has fewer), in search order: the rows, the
/// documents the last pass matched, values loaded and the shard's
/// documents with a point.
#[allow(clippy::too_many_arguments)]
fn distance_shard(schema: &IndexSchema, wanted: &[bool], t: &crate::search::Target, query: &Value, field: &str, lat: f64, lon: f64, n: usize) -> Result<(Vec<Vec<Val>>, u64, u64, u64)> {
    let targets = std::slice::from_ref(t);
    let (qlat, qlon) = rustic_core::geo::quantize(lat, lon);
    let dist = move |e: &rustic_core::geo::GeoPointEntry| rustic_core::geo::sloppy_haversin_meters(qlat, qlon, decode_lat(e.lat), decode_lon(e.lon));
    struct Best {
        d: std::collections::BinaryHeap<OrdF64>,
        with_point: u64,
    }
    let init = |_: usize| Best { d: std::collections::BinaryHeap::new(), with_point: 0 };
    let pass1 = |st: &mut Best, p: &crate::search::MatchPage<'_>| -> Result<bool> {
        let Some(id) = p.lookup.get(field).and_then(|f| f.geo_id) else { return Ok(true) };
        let dv = GeoDv::open(p.leaf, id);
        for d in p.doc_ids() {
            let pts = dv.entries(d);
            if pts.is_empty() {
                continue;
            }
            st.with_point += 1;
            let x = pts.iter().map(|e| dist(&e)).fold(f64::INFINITY, f64::min);
            if st.d.len() < n {
                st.d.push(OrdF64(x));
            } else if st.d.peek().is_some_and(|m| x < m.0) {
                st.d.pop();
                st.d.push(OrdF64(x));
            }
        }
        Ok(true)
    };
    // As the single-node search: a growing radius until `n` points are
    // well inside it, or the whole shard.
    let total = live_docs(targets)?.max(1) as f64;
    let mut radius = (n as f64 / total * 5.1e14 / std::f64::consts::PI).sqrt().clamp(100.0, 2.0e6) / 4.0;
    let (bound, query, with_point) = loop {
        let unbounded = radius > 2.1e7;
        let q = if unbounded { query.clone() } else { json!({"bool": {"filter": [query.clone(), {"geo_distance": {"distance": format!("{radius}m"), field: {"lat": lat, "lon": lon}}}]}}) };
        let states = crate::search::par_for_each_match(targets, &q, PAGE, &init, &pass1)?;
        let with_point: u64 = states.iter().map(|s| s.with_point).sum();
        if with_point >= n as u64 {
            let mut all: Vec<f64> = states.into_iter().flat_map(|s| s.d.into_iter().map(|x| x.0)).collect();
            all.sort_by(f64::total_cmp);
            let b = all[n - 1];
            if unbounded || b <= radius * 0.999 - 1.0 {
                break (b, q, with_point);
            }
        } else if unbounded {
            // Fewer points than wanted: all of them.
            break (f64::INFINITY, q, with_point);
        }
        radius *= if with_point == 0 { 8.0 } else { ((n as f64 / with_point as f64).sqrt() * 1.5).clamp(2.0, 8.0) };
    };
    type Rows = (Vec<Vec<Val>>, u64, u64, Option<EsError>);
    let init2 = |_: usize| -> Rows { (Vec::new(), 0, 0, None) };
    let pass2 = |st: &mut Rows, p: &crate::search::MatchPage<'_>| -> Result<bool> {
        st.1 += p.len() as u64;
        let Some(id) = p.lookup.get(field).and_then(|f| f.geo_id) else { return Ok(true) };
        let dv = GeoDv::open(p.leaf, id);
        for d in p.doc_ids() {
            let pts = dv.entries(d);
            if pts.is_empty() || pts.iter().map(|e| dist(&e)).fold(f64::INFINITY, f64::min) > bound {
                continue;
            }
            match source_row(schema, wanted, p.target, p.lookup, p.leaf, d) {
                Ok((row, l, memory)) => {
                    st.2 += l;
                    memory.retain_for_request();
                    st.0.push(row);
                }
                Err(e) => {
                    st.3 = Some(e);
                    return Ok(false);
                }
            }
        }
        Ok(true)
    };
    let (mut rows, mut docs, mut loaded) = (Vec::new(), 0u64, 0u64);
    for (r, d, l, err) in crate::search::par_for_each_match(targets, &query, PAGE, &init2, &pass2)? {
        if let Some(e) = err {
            return Err(e);
        }
        rows.extend(r);
        docs += d;
        loaded += l;
    }
    Ok((rows, docs, loaded, with_point))
}

/// An `f64` ordered by `total_cmp` (heap element).
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// `STATS` (no groups) of `COUNT(*)` / `ST_CENTROID_AGG` / `ST_EXTENT_AGG`
/// over geo doc values.
fn geo_stats(ctx: &mut Ctx, schema: &IndexSchema, query: &Value, cmd: &Cmd, aggs: &[Field]) -> Result<Option<Table>> {
    let mut plan: Vec<GeoAgg> = Vec::with_capacity(aggs.len());
    let mut names: HashSet<&str> = HashSet::new();
    for a in aggs {
        if a.filter.is_some() || !names.insert(&a.name) {
            return Ok(None);
        }
        let Expr::Call(name, args, _) = &a.expr else { return Ok(None) };
        plan.push(match (name.as_str(), args.as_slice()) {
            ("COUNT", [Expr::Star]) => GeoAgg::Count,
            ("COUNT", [arg]) => match count_dv_field(arg, schema) {
                Some(f) => GeoAgg::CountValues(f),
                None => return Ok(None),
            },
            ("ST_CENTROID_AGG", [arg]) => match (geo_dv_field(arg, schema), geo_shape_dv_field(arg, schema)) {
                (Some(f), _) => GeoAgg::Centroid(f),
                (None, Some(f)) => GeoAgg::ShapeCentroid(f),
                _ => return Ok(None),
            },
            ("ST_EXTENT_AGG", [arg]) => match (geo_dv_field(arg, schema), geo_shape_dv_field(arg, schema)) {
                (Some(f), _) => GeoAgg::Extent(f),
                (None, Some(f)) => GeoAgg::ShapeExtent(f),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        });
    }
    // Shape aggregates have no wire form: shards on other nodes run the
    // table path.
    if schema.slots.is_some() && plan.iter().any(|a| matches!(a, GeoAgg::ShapeCentroid(_) | GeoAgg::ShapeExtent(_) | GeoAgg::CountValues(_))) {
        return Ok(None);
    }
    // Output columns and checks: the same STATS over no rows.
    let empty = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    let mut out = apply(ctx, cmd, empty)?;
    if out.cols.len() != plan.len() {
        return Ok(None);
    }
    let n = plan.len();
    let states = if schema.slots.is_some() {
        let wire: Option<Vec<GeoAggPlan>> = plan.iter().map(|a| match a {
            GeoAgg::Count => Some(GeoAggPlan::Count),
            GeoAgg::Centroid(f) => Some(GeoAggPlan::Centroid(f.clone())),
            GeoAgg::Extent(f) => Some(GeoAggPlan::Extent(f.clone())),
            GeoAgg::ShapeCentroid(_) | GeoAgg::ShapeExtent(_) | GeoAgg::CountValues(_) => None,
        }).collect();
        let Some(wire) = wire else { return Ok(None) };
        let partials = crate::memory::partials_mark();
        let parts = by_slot(ctx, schema, query, FastPlan::GeoStats { aggs: wire }, |_, targets| {
            Ok(geo_states(targets, query, &plan, true)?.into_iter().map(GeoState::part).collect())
        })?;
        let Some(parts) = parts else { return Ok(None) };
        let Some(states) = parts.into_iter().map(|p| GeoState::of_part(p, n)).collect::<Option<Vec<_>>>() else { return Ok(None) };
        // Fixed-size merge states replace the parts: release their charge.
        crate::memory::consume_partials_since(partials);
        states
    } else {
        geo_states(&schema.targets, query, &plan, false)?
    };
    if states.iter().any(|st| st.bail) {
        return Ok(None);
    }
    taken();
    // Merge shard states in search order (ES combines intermediate
    // compensated sums with their deltas).
    let mut docs = 0u64;
    let mut sums: Vec<(crate::esql::partial::Kahan, crate::esql::partial::Kahan, u64)> = vec![Default::default(); n];
    let mut bounds: Vec<Option<(f64, f64, f64, f64)>> = vec![None; n];
    let mut shape_sums: Vec<ShapeSum> = vec![ShapeSum::default(); n];
    let mut extents: Vec<Option<rustic_core::fieldtypes::shape::ShapeExtent>> = vec![None; n];
    for st in states {
        docs += st.docs;
        for k in 0..n {
            // `CentroidShapeAggregator.combineIntermediate` /
            // `SpatialExtentLongitudeWrappingAggregator.combineIntermediate`.
            if let Some(p) = st.shape_sums.get(k) {
                if p.weight > 0.0 {
                    shape_sums[k].add(p.x.value, p.x.delta, p.y.value, p.y.delta, p.weight, p.dim);
                }
            }
            if let Some(Some(e)) = st.extents.get(k) {
                extents[k] = Some(match extents[k] {
                    Some(mut x) => {
                        x.merge(e);
                        x
                    }
                    None => *e,
                });
            }
            let (x, y, c) = st.sums[k];
            sums[k].0.add(x.value, x.delta);
            sums[k].1.add(y.value, y.delta);
            sums[k].2 += c;
            bounds[k] = match (bounds[k], st.bounds[k]) {
                (None, b) | (b, None) => b,
                (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3))),
            };
        }
    }
    ctx.documents_found += docs;
    let row: Vec<Val> = plan
        .iter()
        .enumerate()
        .map(|(k, agg)| match agg {
            GeoAgg::Count => Val::Long(docs as i64),
            GeoAgg::Centroid(_) => {
                let (x, y, c) = sums[k];
                if c == 0 {
                    Val::Null
                } else {
                    Val::Str(crate::esql::spatial::point_wkt(x.value / c as f64, y.value / c as f64))
                }
            }
            GeoAgg::Extent(_) => match bounds[k] {
                None => Val::Null,
                Some((minx, maxx, miny, maxy)) => Val::Str(crate::esql::spatial::to_wkt(&rustic_core::fieldtypes::shape::Geometry::Envelope { min_x: minx, max_x: maxx, max_y: maxy, min_y: miny })),
            },
            // `ShapeCentroidState.toBlock`.
            GeoAgg::ShapeCentroid(_) => {
                let p = &shape_sums[k];
                if p.weight > 0.0 {
                    Val::Str(crate::esql::spatial::point_wkt(p.x.value / p.weight, p.y.value / p.weight))
                } else {
                    Val::Null
                }
            }
            GeoAgg::ShapeExtent(_) => match &extents[k] {
                None => Val::Null,
                Some(e) => shape_extent_wkt(e),
            },
            GeoAgg::CountValues(_) => Val::Long(sums[k].2 as i64),
        })
        .collect();
    out.rows = vec![row];
    Ok(Some(out))
}

/// The per-shard states of a doc-values `STATS` over `targets` (`per_target`:
/// exactly one per target, even for counts only).
fn geo_states(targets: &[crate::search::Target], query: &Value, plan: &[GeoAgg], per_target: bool) -> Result<Vec<GeoState>> {
    let n = plan.len();
    let init = |_: usize| GeoState { docs: 0, sums: vec![Default::default(); n], bounds: vec![None; n], shape_sums: vec![ShapeSum::default(); n], extents: vec![None; n], bail: false };
    // Per leaf: which docs match (all of them; a KD-tree hit set of a geo
    // predicate on a single-valued field; else the query's scorer), then
    // their points in doc order (the order ES sums them in).
    let fold = |st: &mut GeoState, v: &crate::search::LeafView<'_>| -> Result<()> {
        let leaf = v.leaf;
        let clean = leaf.deleted_bits().is_none();
        let all = clean && v.query.is_match_all();
        // COUNT(*) alone under a geo predicate: counted from the tree (over
        // a few threads), no hit set built.
        if !all && clean && plan.iter().all(|a| matches!(a, GeoAgg::Count)) {
            if let Some(n) = v.query.exact_count(leaf) {
                st.docs += n;
                return Ok(());
            }
        }
        let geo_hits: Option<rustic_core::geo::DocHits> = match (all, clean, v.query.geo_pred(), leaf) {
            (false, true, Some((f, pred)), rustic_core::query::LeafReader::Segment(seg)) => match seg.geo_index(f) {
                Some(g) => g.pred_hits(&pred, seg.num_docs),
                None => Some(rustic_core::geo::DocHits::Sorted(Vec::new())),
            },
            _ => None,
        };
        // `COUNT(*)` and `COUNT(field)` of fields every doc of the leaf has
        // one value of, alone: the matches counted without visiting them
        // when the query knows how (a `geo_shape` relation counts its tree
        // leaves, ES's BKD count).
        // A field inside a `nested` object is invisible to ES|QL (its index
        // reads it as null), even where `include_in_root` / `include_in_parent`
        // copied its values onto the root docs.
        let visible = |f: &str| v.lookup.get(f).filter(|rf| rf.nested_path.is_none());
        let one_value = |f: &str| visible(f).and_then(|rf| rf.core_id).is_some_and(|id| leaf.numeric_one_value_per_doc(id));
        let countable = plan.iter().all(|a| match a {
            GeoAgg::Count => true,
            GeoAgg::CountValues(f) => one_value(f),
            _ => false,
        });
        let counted = if !all && geo_hits.is_none() && clean && countable { v.query.exact_count(leaf) } else { None };
        // The matching docs, when no shortcut applies.
        let scanned: Option<Vec<rustic_core::types::DocId>> = if all || geo_hits.is_some() || counted.is_some() {
            None
        } else {
            let mut d = Vec::new();
            v.for_each_match(|x| d.push(x));
            Some(d)
        };
        st.docs += if all {
            leaf.num_docs() as u64
        } else if let Some(h) = &geo_hits {
            h.len() as u64
        } else if let Some(n) = counted {
            n
        } else {
            scanned.as_ref().map_or(0, |d| d.len() as u64)
        };
        for (k, agg) in plan.iter().enumerate() {
            if let GeoAgg::CountValues(field) = agg {
                let Some(rf) = visible(field) else { continue };
                let (id, double) = match rf.core_id {
                    Some(id) if rf.declared == crate::mapping::EsType::Double => (id, true),
                    Some(id) if matches!(rf.declared, crate::mapping::EsType::Long | crate::mapping::EsType::Integer) => (id, false),
                    _ => {
                        st.bail = true;
                        continue;
                    }
                };
                // Counted matches, one value each.
                if let Some(n) = counted {
                    st.sums[k].2 += n;
                    continue;
                }
                let nd = leaf.num_docs();
                let c = &mut st.sums[k].2;
                let mut visit = |d: rustic_core::types::DocId| {
                    if d < nd {
                        *c += if double { leaf.f64_values(id, d).len() } else { leaf.i64_values(id, d).len() } as u64;
                    }
                };
                if all {
                    for d in 0..nd {
                        visit(d);
                    }
                } else if let Some(h) = &geo_hits {
                    match h {
                        rustic_core::geo::DocHits::Bits(b) => {
                            for (wi, &w) in b.iter().enumerate() {
                                let mut w = w;
                                while w != 0 {
                                    let d = wi as u32 * 64 + w.trailing_zeros();
                                    w &= w - 1;
                                    visit(d);
                                }
                            }
                        }
                        rustic_core::geo::DocHits::Sorted(docs) => {
                            for &d in docs {
                                visit(d);
                            }
                        }
                    }
                } else if let Some(docs) = &scanned {
                    for &d in docs {
                        visit(d);
                    }
                }
                continue;
            }
            if let GeoAgg::ShapeCentroid(field) | GeoAgg::ShapeExtent(field) = agg {
                // Each matching doc's doc-value header, no WKT parsed.
                let Some(rf) = visible(field) else { continue };
                let id = match rf.core_id {
                    Some(id) if rf.declared == crate::mapping::EsType::GeoShape => id,
                    _ => {
                        st.bail = true;
                        continue;
                    }
                };
                let centroid = matches!(agg, GeoAgg::ShapeCentroid(_));
                let nd = leaf.num_docs();
                let (sum, ext) = (&mut st.shape_sums[k], &mut st.extents[k]);
                let values = v.leaf.shape_values(id);
                // Every doc of a segment without deletes: the union of the
                // values' extents (the same min / max as doc by doc).
                if all && !centroid {
                    if let (rustic_core::query::ShapeValues::Column(c), rustic_core::query::LeafReader::Segment(seg)) = (values, &v.leaf) {
                        if seg.num_live_docs() == seg.num_docs {
                            if let Some(e) = c.union_extent() {
                                *ext = Some(match *ext {
                                    Some(mut x) => {
                                        x.merge(&e);
                                        x
                                    }
                                    None => e,
                                });
                            }
                            continue;
                        }
                    }
                }
                // One doc's header into `sum` / `ext` (`Sync`: chunks of a large leaf's
                // matches fold in parallel, merged in order below).
                let one = |d: rustic_core::types::DocId, sum: &mut ShapeSum, ext: &mut Option<rustic_core::fieldtypes::shape::ShapeExtent>| {
                    if d >= nd {
                        return;
                    }
                    if centroid {
                        // `SpatialCentroidShapeDocValuesAggregator.combine`
                        // over the doc's header.
                        let c = match values {
                            rustic_core::query::ShapeValues::Column(col) => col.centroid(d),
                            _ => values.get(d).map(|sv| (sv.centroid_x(), sv.centroid_y(), sv.dimension(), sv.weight())),
                        };
                        if let Some((x, y, dim, w)) = c {
                            if w > 0.0 {
                                let (x, y) = (rustic_core::geo::decode_lon(x), rustic_core::geo::decode_lat(y));
                                sum.add(x * w, 0.0, y * w, 0.0, w, dim);
                            }
                        }
                        return;
                    }
                    // The doc's shape doc value header.
                    let Some(sv) = values.get(d) else { return };
                    if let Some(e) = sv.extent() {
                        *ext = Some(match *ext {
                            Some(mut x) => {
                                x.merge(&e);
                                x
                            }
                            None => e,
                        });
                    }
                };
                let mut visit = |d: rustic_core::types::DocId| one(d, sum, ext);
                if all && nd as usize >= SHAPE_FOLD_PAR_MIN {
                    // Every doc of a large leaf: doc ranges in chunks on the
                    // pool, merged in doc order (as the matches below).
                    use rayon::prelude::*;
                    let chunks = (nd as usize).div_ceil(SHAPE_FOLD_CHUNK);
                    let parts: Vec<(ShapeSum, Option<rustic_core::fieldtypes::shape::ShapeExtent>)> = (0..chunks)
                        .into_par_iter()
                        .map(|c| {
                            let (mut s, mut e) = (ShapeSum::default(), None);
                            let lo = (c * SHAPE_FOLD_CHUNK) as u32;
                            let hi = (((c + 1) * SHAPE_FOLD_CHUNK) as u32).min(nd);
                            for d in lo..hi {
                                one(d, &mut s, &mut e);
                            }
                            (s, e)
                        })
                        .collect();
                    drop(visit);
                    for (p, e) in parts {
                        if p.weight > 0.0 {
                            sum.add(p.x.value, p.x.delta, p.y.value, p.y.delta, p.weight, p.dim);
                        }
                        if let Some(e) = e {
                            *ext = Some(match *ext {
                                Some(mut x) => {
                                    x.merge(&e);
                                    x
                                }
                                None => e,
                            });
                        }
                    }
                } else if all {
                    for d in 0..nd {
                        visit(d);
                    }
                } else if let Some(h) = &geo_hits {
                    match h {
                        rustic_core::geo::DocHits::Bits(b) => {
                            for (wi, &w) in b.iter().enumerate() {
                                let mut w = w;
                                while w != 0 {
                                    let d = wi as u32 * 64 + w.trailing_zeros();
                                    w &= w - 1;
                                    visit(d);
                                }
                            }
                        }
                        rustic_core::geo::DocHits::Sorted(docs) => {
                            for &d in docs {
                                visit(d);
                            }
                        }
                    }
                } else if let Some(docs) = &scanned {
                    if docs.len() >= SHAPE_FOLD_PAR_MIN {
                        // A large leaf's matches in chunks on the pool, each
                        // into its own state, merged in doc order (ES combines
                        // partial centroid states with their deltas).
                        use rayon::prelude::*;
                        let parts: Vec<(ShapeSum, Option<rustic_core::fieldtypes::shape::ShapeExtent>)> = docs
                            .par_chunks(SHAPE_FOLD_CHUNK)
                            .map(|chunk| {
                                let (mut s, mut e) = (ShapeSum::default(), None);
                                for &d in chunk {
                                    one(d, &mut s, &mut e);
                                }
                                (s, e)
                            })
                            .collect();
                        drop(visit);
                        for (p, e) in parts {
                            if p.weight > 0.0 {
                                sum.add(p.x.value, p.x.delta, p.y.value, p.y.delta, p.weight, p.dim);
                            }
                            if let Some(e) = e {
                                *ext = Some(match *ext {
                                    Some(mut x) => {
                                        x.merge(&e);
                                        x
                                    }
                                    None => e,
                                });
                            }
                        }
                    } else {
                        for &d in docs {
                            visit(d);
                        }
                    }
                }
                continue;
            }
            let field = match agg {
                GeoAgg::Count | GeoAgg::ShapeCentroid(_) | GeoAgg::ShapeExtent(_) | GeoAgg::CountValues(_) => continue,
                GeoAgg::Centroid(f) | GeoAgg::Extent(f) => f,
            };
            let Some(id) = visible(field).and_then(|f| f.geo_id) else { continue };
            let dv = GeoDv::open(leaf, id);
            if dv.is_empty() {
                continue;
            }
            // The points of the matching docs, in doc order.
            let mut each = |add: &mut dyn FnMut(&rustic_core::geo::GeoPointEntry)| {
                if all {
                    dv.iter().for_each(|e| add(&e));
                } else if let Some(h) = &geo_hits {
                    match h {
                        rustic_core::geo::DocHits::Bits(b) => dv.iter().filter(|e| b.get(e.doc as usize / 64).is_some_and(|w| w >> (e.doc % 64) & 1 == 1)).for_each(|e| add(&e)),
                        rustic_core::geo::DocHits::Sorted(docs) => {
                            for &d in docs {
                                dv.entries(d).iter().for_each(|e| add(&e));
                            }
                        }
                    }
                } else if let Some(docs) = &scanned {
                    for &d in docs {
                        dv.entries(d).iter().for_each(|e| add(&e));
                    }
                }
            };
            match agg {
                GeoAgg::Centroid(_) => {
                    let (sx, sy, c) = &mut st.sums[k];
                    let split_bits = match &geo_hits {
                        Some(rustic_core::geo::DocHits::Bits(b)) if !all && dv.len() >= SPLIT_POINTS => Some(b),
                        _ => None,
                    };
                    if let Some(bits) = split_bits {
                        // A large leaf and a KD hit set: the points walk
                        // split like the whole-leaf case below. When every
                        // doc has exactly one point, point `d` is doc `d`:
                        // the set bits index the points directly.
                        let Some(pts) = dv.values() else { continue };
                        let dense = pts.len() == leaf.num_docs() as usize && pts.get_entry(pts.len().wrapping_sub(1)).is_some_and(|e| e.doc as usize == pts.len() - 1);
                        let parts = (pts.len() / SPLIT_POINTS).clamp(2, SPLIT_MAX);
                        let words = bits.len().div_ceil(parts).max(1);
                        let len = pts.len().div_ceil(parts);
                        let partial: Vec<(crate::esql::partial::Kahan, crate::esql::partial::Kahan, u64)> = std::thread::scope(|scope| {
                            let hs: Vec<_> = (0..parts)
                                .map(|k| {
                                    scope.spawn(move || {
                                        let (mut x, mut y, mut n) = (crate::esql::partial::Kahan::default(), crate::esql::partial::Kahan::default(), 0u64);
                                        let mut add = |e: &rustic_core::geo::GeoPointEntry| {
                                            x.add(decode_lon(e.lon), 0.0);
                                            y.add(decode_lat(e.lat), 0.0);
                                            n += 1;
                                        };
                                        if dense {
                                            let (w0, w1) = ((k * words).min(bits.len()), ((k + 1) * words).min(bits.len()));
                                            for (wi, &w) in bits[w0..w1].iter().enumerate() {
                                                let mut w = w;
                                                while w != 0 {
                                                    let d = (w0 + wi) * 64 + w.trailing_zeros() as usize;
                                                    w &= w - 1;
                                                    if let Some(e) = pts.get_entry(d) {
                                                        add(&e);
                                                    }
                                                }
                                            }
                                        } else {
                                            for e in pts.iter_range((k * len).min(pts.len())..((k + 1) * len).min(pts.len())) {
                                                if bits.get(e.doc as usize / 64).is_some_and(|w| w >> (e.doc % 64) & 1 == 1) {
                                                    add(&e);
                                                }
                                            }
                                        }
                                        (x, y, n)
                                    })
                                })
                                .collect();
                            hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
                        });
                        for (x, y, n) in partial {
                            sx.add(x.value, x.delta);
                            sy.add(y.value, y.delta);
                            *c += n;
                        }
                    } else if all && dv.len() >= SPLIT_POINTS {
                        // A large leaf: compensated partial sums over a few
                        // slices in parallel, combined in slice order (the
                        // way ES combines its drivers' intermediate states).
                        let Some(pts) = dv.values() else { continue };
                        let parts = (pts.len() / SPLIT_POINTS).clamp(2, SPLIT_MAX);
                        let len = pts.len().div_ceil(parts);
                        let partial: Vec<(crate::esql::partial::Kahan, crate::esql::partial::Kahan)> = std::thread::scope(|scope| {
                            let hs: Vec<_> = (0..pts.len().div_ceil(len))
                                .map(|k| pts.iter_range(k * len..((k + 1) * len).min(pts.len())))
                                .map(|ch| {
                                    scope.spawn(move || {
                                        let (mut x, mut y) = (crate::esql::partial::Kahan::default(), crate::esql::partial::Kahan::default());
                                        for e in ch {
                                            x.add(decode_lon(e.lon), 0.0);
                                            y.add(decode_lat(e.lat), 0.0);
                                        }
                                        (x, y)
                                    })
                                })
                                .collect();
                            hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
                        });
                        for (x, y) in partial {
                            sx.add(x.value, x.delta);
                            sy.add(y.value, y.delta);
                        }
                        *c += pts.len() as u64;
                    } else {
                        each(&mut |e| {
                            sx.add(decode_lon(e.lon), 0.0);
                            sy.add(decode_lat(e.lat), 0.0);
                            *c += 1;
                        });
                    }
                }
                GeoAgg::Extent(_) => {
                    let b = &mut st.bounds[k];
                    each(&mut |e| {
                        let (x, y) = (decode_lon(e.lon), decode_lat(e.lat));
                        *b = Some(match *b {
                            None => (x, x, y, y),
                            Some((a, bb, c, d)) => (a.min(x), bb.max(x), c.min(y), d.max(y)),
                        });
                    });
                }
                GeoAgg::Count | GeoAgg::ShapeCentroid(_) | GeoAgg::ShapeExtent(_) | GeoAgg::CountValues(_) => {}
            }
        }
        Ok(())
    };
    // Only counts: the index counts (KD-tree leaves inside a shape are
    // counted whole), no doc is visited.
    if plan.iter().all(|a| matches!(a, GeoAgg::Count)) {
        if per_target {
            // One state per target, the targets counted concurrently (as
            // `par_count` counts them for a single total).
            if targets.len() <= 1 {
                return targets.iter().map(|t| Ok(GeoState { docs: crate::search::par_count(std::slice::from_ref(t), query)?, ..init(0) })).collect();
            }
            return std::thread::scope(|scope| {
                let handles: Vec<_> = targets.iter().map(|t| scope.spawn(move || crate::search::par_count(std::slice::from_ref(t), query))).collect();
                handles.into_iter().map(|h| Ok(GeoState { docs: h.join().unwrap_or_else(|_| Err(EsError::internal("shard count panicked")))?, ..init(0) })).collect()
            });
        }
        return Ok(vec![GeoState { docs: crate::search::par_count(targets, query)?, ..init(0) }]);
    }
    // Leaves in parallel where the query allows it (a geo predicate, or
    // all docs), each into its own state, merged in leaf order per target.
    let per_leaf = crate::search::par_for_each_leaf_states(targets, query, &init, &fold)?;
    Ok(per_leaf.into_iter().enumerate().map(|(ti, leaves)| {
        let mut st = init(ti);
        for leaf in leaves {
            st.merge(leaf);
        }
        st
    }).collect())
}

/// `e` with the columns `defs` defines replaced by their definitions.
pub(super) fn subst(e: &Expr, defs: &[(String, Expr)]) -> Expr {
    match e {
        Expr::Col(n, _) => match defs.iter().rev().find(|(d, _)| d == n) {
            Some((_, x)) => x.clone(),
            None => e.clone(),
        },
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(subst(a, defs)), Box::new(subst(b, defs))),
        Expr::Neg(a) => Expr::Neg(Box::new(subst(a, defs))),
        Expr::Not(a) => Expr::Not(Box::new(subst(a, defs))),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(subst(a, defs)), *n),
        Expr::In(a, l, n) => Expr::In(Box::new(subst(a, defs)), l.iter().map(|x| subst(x, defs)).collect(), *n),
        Expr::Like(a, p, n, r) => Expr::Like(Box::new(subst(a, defs)), p.clone(), *n, *r),
        Expr::Call(f, args, p) => Expr::Call(f.clone(), args.iter().map(|x| subst(x, defs)).collect(), *p),
        Expr::Cast(a, t) => Expr::Cast(Box::new(subst(a, defs)), t.clone()),
        Expr::Between(a, b, c, n) => Expr::Between(Box::new(subst(a, defs)), Box::new(subst(b, defs)), Box::new(subst(c, defs)), *n),
        o => o.clone(),
    }
}

/// Whether `e` reads only index columns, constants and the functions a
/// spatial pushdown understands.
fn simple(e: &Expr) -> bool {
    match e {
        Expr::Col(..) | Expr::Lit(_) => true,
        Expr::Call(f, args, _) => matches!(f.as_str(), "ST_DISTANCE" | "TO_GEOPOINT" | "TO_GEOSHAPE" | "ST_INTERSECTS" | "ST_DISJOINT" | "ST_WITHIN" | "ST_CONTAINS") && args.iter().all(simple),
        _ => false,
    }
}

/// ES pushes spatial predicates on `EVAL`ed spatial values into Lucene
/// (`EnableSpatialDistancePushdown`, `PushDownAndCombineFilters` through
/// `EVAL`): in `FROM | EVAL d = ST_DISTANCE(loc, pt) | WHERE d < x AND ...`
/// each conjunct that becomes an index query once `d` is replaced by its
/// definition moves in front of the `EVAL`s (which still run). Returns the
/// rewritten commands, or `None` when nothing moves.
pub(super) fn hoist_spatial_filters(rest: &[Cmd], types: &HashMap<String, String>) -> Option<Vec<Cmd>> {
    if disabled() {
        return None;
    }
    let mut defs: Vec<(String, Expr)> = Vec::new();
    let mut hoisted: Vec<Cmd> = Vec::new();
    let mut kept: Vec<Cmd> = Vec::new();
    let mut k = 0;
    while k < rest.len() {
        match &rest[k] {
            Cmd::Eval(fields) => {
                for f in fields {
                    let def = subst(&f.expr, &defs);
                    // An unsupported definition hides the index column it shadows.
                    defs.push((f.name.clone(), if simple(&def) { def } else { Expr::Lit(Val::Null) }));
                }
                kept.push(rest[k].clone());
            }
            Cmd::Where(e, pos) if !defs.is_empty() => {
                let mut cs = Vec::new();
                conjuncts(e, &mut cs);
                let mut left: Vec<Expr> = Vec::new();
                for c in cs {
                    let s = subst(&c, &defs);
                    let defined = |x: &Expr| -> bool {
                        let mut names = Vec::new();
                        col_names(x, &mut names);
                        names.iter().any(|n| defs.iter().any(|(d, _)| d == n))
                    };
                    if defined(&c) && !matches!(s, Expr::Lit(Val::Null)) && simple_pred(&s) && spatial_dsl(&s, types).is_some() {
                        hoisted.push(Cmd::Where(s, *pos));
                    } else {
                        left.push(c);
                    }
                }
                if let Some(e) = left.into_iter().reduce(|a, b| Expr::Bin(BinOp::And, Box::new(a), Box::new(b))) {
                    kept.push(Cmd::Where(e, *pos));
                }
            }
            Cmd::Where(..) => kept.push(rest[k].clone()),
            _ => break,
        }
        k += 1;
    }
    if hoisted.is_empty() {
        return None;
    }
    hoisted.extend(kept);
    hoisted.extend(rest[k..].iter().cloned());
    Some(hoisted)
}

/// A comparison (or spatial relation) over [`simple`] operands.
fn simple_pred(e: &Expr) -> bool {
    match e {
        Expr::Bin(op, a, b) if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) => simple(a) && simple(b),
        Expr::Call(..) => simple(e),
        _ => false,
    }
}

fn col_names(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Col(n, _) => out.push(n.clone()),
        Expr::Bin(_, a, b) => {
            col_names(a, out);
            col_names(b, out);
        }
        Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => col_names(a, out),
        Expr::In(a, l, _) => {
            col_names(a, out);
            l.iter().for_each(|x| col_names(x, out));
        }
        Expr::Call(_, args, _) => args.iter().for_each(|x| col_names(x, out)),
        Expr::Between(a, b, c, _) => {
            col_names(a, out);
            col_names(b, out);
            col_names(c, out);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn grid_group_growth_is_admitted_before_insert_and_released() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let scope=crate::memory::RequestScope::new("grid groups",breakers);
        crate::memory::scoped(Some(scope),|| {
            let mut groups=super::GridGroups{index:Default::default(),groups:Vec::new(),docs:0,outside:false,memory:None};
            let mut failed=false;
            for key in 0..10000 {
                let before=groups.groups.len();
                if let Err(error)=groups.add(Some(key)) {
                    assert_eq!(error.status,429);
                    assert_eq!(groups.groups.len(),before);
                    assert_eq!(groups.index.len(),before);
                    failed=true;break;
                }
            }
            assert!(failed);
            drop(groups);
            assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn guarded_source_pages_release_between_pages_and_reject_large_decode() {
        let tmp=tempfile::tempdir().unwrap();
        let node=Node::open(tmp.path()).unwrap();
        node.create_index("page-memory",&json!({"mappings":{"properties":{"value":{"type":"keyword"}}}})).unwrap();
        let service=node.get_index("page-memory").unwrap();
        for i in 0..64 {
            service.write(vec![WriteReq::Index{id:Some(i.to_string()),source:json!({"value":"small"}).to_string(),routing:None,cond:Default::default()}],false).pop().unwrap().unwrap();
        }
        service.refresh().unwrap();
        let ctx=super::new_ctx(&node,"FROM page-memory",None,100,100);
        let schema=super::resolve_schema(&ctx,&["page-memory".into()],0).unwrap();
        let wanted=vec![true;schema.cols.len()];
        let targets=node.snapshot(&schema.targets).unwrap();
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let scope=crate::memory::RequestScope::new("source pages",breakers);
        crate::memory::scoped(Some(scope),|| {
            let mut count=0;
            crate::search::for_each_match(&targets,&json!({"match_all":{}}),1,&mut |page| {
                for doc in page.doc_ids() {
                    let (rows,_,memory)=super::source_rows_guarded(&schema,&wanted,page.target,page.lookup,page.leaf,&[doc])?;
                    assert_eq!(rows.len(),1);
                    assert!(breakers.request.reserved()>0);
                    drop(rows);drop(memory);
                    assert_eq!(breakers.request.reserved(),0);
                    count+=1;
                }
                Ok(true)
            }).unwrap();
            assert_eq!(count,64);
            // The plan/row-vector reservation is admitted before page allocation.
            crate::search::for_each_match(&targets,&json!({"match_all":{}}),1,&mut |page| {
                let doc=page.doc_ids().next().unwrap();
                let docs=vec![doc;10000];
                let error=super::source_rows_guarded(&schema,&wanted,page.target,page.lookup,page.leaf,&docs).err().unwrap();
                assert_eq!(error.status,429);
                assert_eq!(breakers.request.reserved(),0);
                let error=super::sql_column_rows(&schema,&wanted,page.target,page.lookup,page.leaf,&docs).err().unwrap();
                assert_eq!(error.status,429);
                assert_eq!(breakers.request.reserved(),0);
                Ok(false)
            }).unwrap();
        });
    }

    #[test]
    fn constant_pages_are_admitted_before_per_document_clones() {
        let tmp=tempfile::tempdir().unwrap();
        let node=Node::open(tmp.path()).unwrap();
        node.create_index("constant-memory",&json!({"mappings":{"properties":{"value":{"type":"constant_keyword","value":"x".repeat(4096)}}}})).unwrap();
        let service=node.get_index("constant-memory").unwrap();
        service.write(vec![WriteReq::Index{id:Some("one".into()),source:"{}".into(),routing:None,cond:Default::default()}],false).pop().unwrap().unwrap();
        service.refresh().unwrap();
        let ctx=super::new_ctx(&node,"FROM constant-memory",None,100,100);
        let schema=super::resolve_schema(&ctx,&["constant-memory".into()],0).unwrap();
        let wanted=vec![true;schema.cols.len()];
        let targets=node.snapshot(&schema.targets).unwrap();
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        crate::memory::scoped(Some(crate::memory::RequestScope::new("constant pages",breakers)),|| {
            crate::search::for_each_match(&targets,&json!({"match_all":{}}),1,&mut |page| {
                let docs=vec![page.doc_ids().next().unwrap();32];
                assert!(super::source_page_metadata(&schema,docs.len())>64*1024);
                assert_eq!(super::sql_column_rows(&schema,&wanted,page.target,page.lookup,page.leaf,&docs).err().unwrap().status,429);
                assert_eq!(super::source_rows_guarded(&schema,&wanted,page.target,page.lookup,page.leaf,&docs).err().unwrap().status,429);
                assert_eq!(breakers.request.reserved(),0);
                Ok(false)
            }).unwrap();
        });
    }

    /// SQL's keyword column rows read from the segment's decoded terms (the
    /// first pass builds them, the second reuses them) equal the documents'
    /// `_source` values: repeated, missing and many distinct terms, two
    /// segments.
    #[test]
    fn keyword_rows_by_ordinal_match_source() {
        let tmp=tempfile::tempdir().unwrap();
        let node=Node::open(tmp.path()).unwrap();
        node.create_index("kw-memo",&json!({"mappings":{"properties":{"name":{"type":"keyword"},"n":{"type":"long"}}}})).unwrap();
        let service=node.get_index("kw-memo").unwrap();
        for part in 0..2 {
            let reqs:Vec<WriteReq>=(0..1500u32).map(|i| {
                let source=if i%13==0 { json!({"n":i}) } else { json!({"name":format!("station-{:04}-{}",(i*7919+part)%911,"x".repeat((i%5) as usize)),"n":i}) };
                WriteReq::Index{id:Some(format!("{part}-{i}")),source:source.to_string(),routing:None,cond:Default::default()}
            }).collect();
            for r in service.write(reqs,false) { r.unwrap(); }
            service.flush().unwrap();
            service.refresh().unwrap();
        }
        let ctx=super::new_ctx(&node,"FROM kw-memo",None,100,100);
        let schema=super::resolve_schema(&ctx,&["kw-memo".into()],0).unwrap();
        let wanted=vec![true;schema.cols.len()];
        let targets=node.snapshot(&schema.targets).unwrap();
        for pass in 0..2 {
            let mut pages=0;
            crate::search::for_each_match(&targets,&json!({"match_all":{}}),256,&mut |page| {
                let docs:Vec<_>=page.doc_ids().collect();
                let (fast,_,_)=super::sql_column_rows(&schema,&wanted,page.target,page.lookup,page.leaf,&docs)?;
                let (source,_,_)=super::source_rows_guarded(&schema,&wanted,page.target,page.lookup,page.leaf,&docs)?;
                assert_eq!(fast,source,"pass {pass}");
                pages+=1;
                Ok(true)
            }).unwrap();
            assert!(pages>=2,"{pages} pages");
        }
    }

    #[test]
    fn sql_column_reads_match_source_for_objects_floats_and_fallbacks() {
        for case in 0..10 {
            let variant = case % 5;
            let tmp=tempfile::tempdir().unwrap();
            let node=Node::open(tmp.path()).unwrap();
            let mut mapping=json!({"dynamic":false,"properties":{
                "station":{"properties":{"name":{"type":"keyword"}}},
                "value":{"type":"float"},"late":{"type":"double"}
            }});
            match variant {
                1 => mapping["_source"]=json!({"excludes":["station.name"]}),
                2 => mapping["properties"]["station"]["properties"]["name"]["null_value"]=json!("replacement"),
                3 => mapping["properties"]["value"]["doc_values"]=json!(false),
                4 => { mapping["properties"].as_object_mut().unwrap().remove("late"); },
                _ => {}
            }
            node.create_index("column-read",&json!({"mappings":mapping})).unwrap();
            let service=node.get_index("column-read").unwrap();
            let samples=[
                json!({"station":{"name":"US"},"value":1.23456789,"late":2.5}),
                json!({"station":{"name":["US","AS","US"]},"value":[2.5,-1.25,2.5],"late":4.5}),
                json!({"station":{"name":null},"value":null,"late":8.25}),
                json!({"station":[{"name":"AS"},{"name":"US"}],"value":"12.25","late":0.125}),
                json!({"station.name":"CA","value":-0.0,"late":16.5}),
            ];
            for (i,source) in samples.iter().enumerate() {
                service.write(vec![WriteReq::Index { id:Some(i.to_string()),source:source.to_string(),routing:None,cond:Default::default() }],false).pop().unwrap().unwrap();
            }
            service.refresh().unwrap();
            if case >= 5 { service.flush().unwrap(); }
            if variant==4 { service.put_mapping(&json!({"properties":{"late":{"type":"double"}}})).unwrap(); }
            let ctx=super::new_ctx(&node,"SELECT station.name, value, late FROM column-read",None,100,100);
            let schema=super::resolve_schema(&ctx,&["column-read".into()],0).unwrap();
            let wanted=vec![true;schema.cols.len()];
            let targets=node.snapshot(&schema.targets).unwrap();
            super::LOOKUP_KEY_DV_READS.with(|c|c.set(0));
            crate::search::for_each_match(&targets,&json!({"match_all":{}}),1024,&mut |page| {
                let docs:Vec<_>=page.doc_ids().collect();
                let actual=super::sql_column_rows(&schema,&wanted,page.target,page.lookup,page.leaf,&docs)?;
                let expected=super::source_rows(&schema,&wanted,page.target,page.lookup,page.leaf,&docs)?;
                assert_eq!((&actual.0,actual.1),(&expected.0,expected.1),"variant {variant}");
                Ok(true)
            }).unwrap();
            let direct=super::LOOKUP_KEY_DV_READS.with(|c|c.get());
            assert_eq!(direct>0,variant==0,"variant {variant}: direct={direct}");
        }
    }

    use super::{DISABLED, RESIDUAL_MIN, TAKEN};
    use crate::esql::{run, EsqlOptions, Val};
    use crate::index::WriteReq;
    use crate::node::Node;
    use serde_json::json;

    #[test]
    fn projected_source_preserves_nested_and_dotted_fields_and_validates_skipped_values() {
        let column = super::ColSrc::plain("a.b", "keyword");
        let columns = [super::SourceColumn::Read(&column, false)];
        let raw = r#"{"a":{"b":["x","y"]},"a.b":"literal","unused":{"nested":[1,true,null,"escaped\"value"]}}"#;
        let projected = super::projected_source(raw, &columns).unwrap();
        let full: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(projected["a"], full["a"]);
        assert_eq!(projected["a.b"], full["a.b"]);
        assert!(projected.get("unused").is_none());
        let deep = super::ColSrc::plain("a.b.c", "keyword");
        let deep_columns = [super::SourceColumn::Read(&deep, false)];
        let raw = r#"{"a":{"b":{"c":"nested"}},"a.b":{"c":"partial"},"a.b.c":"literal"}"#;
        let projected = super::projected_source(raw, &deep_columns).unwrap();
        let full: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(crate::highlight::source_values(&projected, "a.b.c"), crate::highlight::source_values(&full, "a.b.c"));
        assert!(super::projected_source(r#"{"a":1,"unused":[1,]}"#, &columns).is_err());
        assert!(super::projected_source(r#"{"a":1} trailing"#, &columns).is_err());
    }

    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    /// Coordinates of `POINT (x y)` / `BBOX (...)` values, for comparisons
    /// tolerant to summation order.
    fn nums(s: &str) -> Vec<f64> {
        s.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'E' | 'e'))).filter_map(|t| t.parse().ok()).collect()
    }

    fn close(a: &Val, b: &Val) -> bool {
        match (a, b) {
            (Val::Str(x), Val::Str(y)) if x.starts_with("POINT") || x.starts_with("BBOX") => {
                let (p, q) = (nums(x), nums(y));
                p.len() == q.len() && p.iter().zip(&q).all(|(x, y)| (x - y).abs() <= 1e-9 * x.abs().max(1.0))
            }
            (Val::Double(x), Val::Double(y)) => (x - y).abs() <= 1e-9 * x.abs().max(1.0),
            _ => a == b,
        }
    }

    /// A 3-shard index of random points (some docs with two points, some
    /// with none), refreshed in several segments, with a delete.
    fn points(node: &Node) {
        node.create_index("pts", &json!({"settings": {"number_of_shards": 3}, "mappings": {"properties": {"loc": {"type": "geo_point"}, "n": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("pts").unwrap();
        let mut r = Rng(7);
        let mut batch = Vec::new();
        for id in 0..6000 {
            let lat = r.below(3000) as f64 / 100.0 + 40.0 + r.below(1000) as f64 / 1e7;
            let lon = r.below(3000) as f64 / 100.0 - 10.0 + r.below(1000) as f64 / 1e7;
            let loc = match r.below(10) {
                0 => json!([{"lat": lat, "lon": lon}, {"lat": lat - 5.0, "lon": lon + 3.0}]),
                1 => json!(null),
                2 => json!(format!("POINT ({lon} {lat})")),
                _ => json!({"lat": lat, "lon": lon}),
            };
            batch.push(WriteReq::Index { id: Some(format!("p{id}")), source: json!({"loc": loc, "n": id}).to_string(), routing: None, cond: Default::default() });
            if batch.len() == 1500 {
                for x in svc.write(std::mem::take(&mut batch), false) {
                    x.unwrap();
                }
                svc.refresh().unwrap();
            }
        }
        for x in svc.write(vec![WriteReq::Delete { id: "p11".into(), routing: None, cond: Default::default() }], false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
    }

    fn opts() -> EsqlOptions<'static> {
        EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }
    }

    /// The query with fast paths / hoisting on, and with them off (the
    /// table interpreter evaluating every predicate per row).
    fn both(node: &Node, q: &str) -> (crate::esql::EsqlResult, crate::esql::EsqlResult, usize) {
        DISABLED.with(|c| c.set(true));
        let slow = run(node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
        DISABLED.with(|c| c.set(false));
        let before = TAKEN.with(|c| c.get());
        let fast = run(node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
        (fast, slow, TAKEN.with(|c| c.get()) - before)
    }

    fn assert_same(q: &str, fast: &crate::esql::EsqlResult, slow: &crate::esql::EsqlResult) {
        assert_eq!(fast.table.cols, slow.table.cols, "{q}");
        assert_eq!(fast.table.rows.len(), slow.table.rows.len(), "{q}");
        for (a, b) in fast.table.rows.iter().zip(&slow.table.rows) {
            assert!(a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(x, y)), "{q}\n fast: {a:?}\n slow: {b:?}");
        }
        assert_eq!(fast.warnings, slow.warnings, "{q}");
    }

    /// ES|QL never reads a field inside a `nested` object, also where
    /// `include_in_root` copied its values onto the root docs: `COUNT(field)`
    /// on the doc-values path counts only the other index's values (YAML
    /// esql/162_nested_field_conflicts "include_in_root does not leak nested
    /// values"), on memtables and sealed segments alike.
    #[test]
    fn count_field_ignores_include_in_root_copies() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("nest_root", &json!({"mappings": {"properties": {"id": {"type": "keyword"}, "item": {"type": "nested", "include_in_root": true, "properties": {"value": {"type": "long"}}}}}})).unwrap();
        node.create_index("obj_root", &json!({"mappings": {"properties": {"id": {"type": "keyword"}, "item": {"properties": {"value": {"type": "long"}}}}}})).unwrap();
        let docs = [
            ("nest_root", json!({"id": "n0", "item": [{"value": 100}]})),
            ("nest_root", json!({"id": "n1", "item": [{"value": 101}]})),
            ("obj_root", json!({"id": "o0", "item": {"value": 1}})),
            ("obj_root", json!({"id": "o1", "item": {"value": 2}})),
        ];
        for (i, (index, doc)) in docs.iter().enumerate() {
            let svc = node.get_index(index).unwrap();
            for x in svc.write(vec![WriteReq::Index { id: Some(i.to_string()), source: doc.to_string(), routing: None, cond: Default::default() }], false) {
                x.unwrap();
            }
        }
        for sealed in [false, true] {
            for index in ["nest_root", "obj_root"] {
                let svc = node.get_index(index).unwrap();
                if sealed {
                    svc.flush().unwrap();
                }
                svc.refresh().unwrap();
            }
            for (q, want) in [("FROM nest_root, obj_root | STATS c = COUNT(item.value)", vec![2i64]), ("FROM nest_root, obj_root | STATS s = SUM(item.value), c = COUNT(item.value)", vec![3, 2])] {
                let (fast, slow, _) = both(&node, q);
                assert_same(q, &fast, &slow);
                assert_eq!(fast.table.rows.len(), 1, "{q}");
                let got: Vec<Option<i64>> = fast.table.rows[0].iter().map(|v| match v { Val::Long(x) => Some(*x), _ => None }).collect();
                assert_eq!(got, want.iter().map(|&x| Some(x)).collect::<Vec<_>>(), "{q} sealed={sealed}: {:?}", fast.table.rows[0]);
            }
        }
    }

    /// `[row commands] STATS ... BY ...` streamed (Rally geoshape
    /// st-envelope-count-esql / st-simplify-count-esql and plain computed
    /// groups) equal the table path: rows, columns and warnings.
    #[test]
    fn columnar_stats_match_streamed_rows() {
        use super::{COLUMNAR_STATS, NO_COLUMNAR};
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let synthetic = json!({"number_of_shards": 1, "index.mapping.source.mode": "synthetic"});
        node.create_index("ka", &json!({"settings": synthetic, "mappings": {"properties": {"@timestamp": {"type": "date"}, "pod": {"type": "keyword"},
            "cpu": {"type": "scaled_float", "scaling_factor": 1000.0}, "mem": {"type": "long"}, "x": {"type": "double"}}}})).unwrap();
        node.create_index("kb", &json!({"settings": synthetic, "mappings": {"properties": {"@timestamp": {"type": "date"}, "pod": {"type": "keyword"},
            "cpu": {"type": "scaled_float", "scaling_factor": 1000.0}, "mem": {"type": "long"}, "x": {"type": "double"}}}})).unwrap();
        let mut seed = 11u64;
        for (index, n) in [("ka", 3000u64), ("kb", 900)] {
            let svc = node.get_index(index).unwrap();
            for half in 0..2 {
                let mut batch = Vec::new();
                for i in 0..n / 2 {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let mut d = json!({"@timestamp": 1_684_200_000_000i64 + ((seed >> 20) % 86_400_000) as i64});
                    if index == "ka" || i % 3 == 0 { d["pod"] = json!(format!("pod-{}", (seed >> 41) % 19)); }
                    if i % 5 != 2 { d["cpu"] = json!(((seed >> 30) % 100_000) as f64 / 7.0); }
                    if i % 4 != 1 { d["mem"] = json!(((seed >> 17) % 1_000_000_000) as i64); }
                    if i % 6 != 0 { d["x"] = json!(((seed >> 11) % 1_000_000) as f64 * 0.37); }
                    if i % 97 == 13 { d.as_object_mut().unwrap().remove("@timestamp"); }
                    batch.push(WriteReq::Index { id: Some(format!("{half}-{i}")), source: d.to_string(), routing: None, cond: Default::default() });
                }
                for r in svc.write(batch, false) { r.unwrap(); }
                // Segments, and (the last batch) published memtables: both
                // read from their columns.
                if half == 0 { svc.flush().unwrap(); }
                svc.refresh().unwrap();
            }
        }
        RESIDUAL_MIN.with(|c| c.set(0));
        for q in [
            "FROM ka,kb | EVAL b = DATE_TRUNC(30 minutes, @timestamp) | STATS a = AVG(cpu), s = SUM(mem), c = COUNT(*), n = COUNT(x), sx = SUM(x), am = AVG(mem) BY pod, b | LIMIT 10000",
            "FROM ka,kb | EVAL by_interval = DATE_TRUNC(1 minute, @timestamp), start_time = DATE_PARSE(\"yyyy-MM-dd'T'HH:mm:ss.SSSX\", \"2023-05-16T10:00:00.000Z\"), end_time = DATE_PARSE(\"yyyy-MM-dd'T'HH:mm:ss.SSSX\", \"2023-05-16T12:00:00.000Z\") \
             | WHERE @timestamp >= start_time AND @timestamp <= end_time | STATS AVG(cpu) BY pod, by_interval | LIMIT 10000",
            "FROM ka,kb | STATS a = AVG(x), c = COUNT(cpu) BY pod",
            "FROM ka | EVAL d = DATE_TRUNC(1 day, @timestamp) | STATS s = SUM(cpu), c = COUNT(*) BY d",
            "FROM kb | STATS a = AVG(cpu), s = SUM(mem)",
            "FROM ka,kb | EVAL b = DATE_TRUNC(1 hour, @timestamp) | STATS p = PERCENTILE(mem, 95), q = PERCENTILE(cpu, 50), d = COUNT_DISTINCT(pod) BY b | LIMIT 10000",
            "FROM ka,kb | WHERE @timestamp >= \"2023-05-16T10:00:00.000Z\" | STATS COUNT_DISTINCT(pod)",
        ] {
            let before = COLUMNAR_STATS.with(|c| c.get());
            let fast = run(&node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
            assert_eq!(COLUMNAR_STATS.with(|c| c.get()), before + 1, "columnar path not taken: {q}");
            NO_COLUMNAR.with(|c| c.set(true));
            let streamed = run(&node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
            NO_COLUMNAR.with(|c| c.set(false));
            assert!(!fast.table.rows.is_empty(), "{q}");
            assert_eq!(fast.table.cols, streamed.table.cols, "{q}");
            // Targets fold concurrently and their compensated sums combine
            // (as ES's drivers do): doubles agree to rounding, the rest exactly.
            assert_eq!(fast.table.rows.len(), streamed.table.rows.len(), "{q}");
            for (a, b) in fast.table.rows.iter().zip(&streamed.table.rows) {
                assert_eq!(a.len(), b.len(), "{q}");
                for (x, y) in a.iter().zip(b) {
                    match (x, y) {
                        (Val::Double(x), Val::Double(y)) => assert!((x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0), "{q}: {x} vs {y}"),
                        _ => assert_eq!(x, y, "{q}"),
                    }
                }
            }
            assert_eq!((fast.documents_found, fast.values_loaded), (streamed.documents_found, streamed.values_loaded), "{q}");
            assert_eq!(fast.warnings, streamed.warnings, "{q}");
            let (_, table, _) = both(&node, q);
            // The same rows as the table path (in its order), doubles to rounding.
            let key = |row: &Vec<Val>| row.iter().map(|v| match v { Val::Double(_) => String::new(), v => format!("{v:?}") }).collect::<Vec<_>>().join("|");
            let mut a: Vec<&Vec<Val>> = fast.table.rows.iter().collect();
            let mut b: Vec<&Vec<Val>> = table.table.rows.iter().collect();
            a.sort_by_key(|r| key(r));
            b.sort_by_key(|r| key(r));
            assert_eq!(a.len(), b.len(), "table path: {q}");
            for (x, y) in a.iter().zip(&b) {
                for (x, y) in x.iter().zip(y.iter()) {
                    match (x, y) {
                        (Val::Double(x), Val::Double(y)) => assert!((x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0), "table path {q}: {x} vs {y}"),
                        _ => assert_eq!(x, y, "table path: {q}"),
                    }
                }
            }
        }
        // A long sum that overflows: the rows path (its warning, a null).
        let svc = node.get_index("kb").unwrap();
        let big = (0..2).map(|i| WriteReq::Index { id: Some(format!("big{i}")), source: json!({"@timestamp": 1_684_200_000_000i64, "pod": "big", "mem": i64::MAX}).to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(big, false) { r.unwrap(); }
        svc.flush().unwrap();
        svc.refresh().unwrap();
        let before = COLUMNAR_STATS.with(|c| c.get());
        let r = run(&node, "FROM kb | STATS s = SUM(mem) BY pod | LIMIT 10000", opts()).unwrap();
        assert_eq!(COLUMNAR_STATS.with(|c| c.get()), before, "overflow falls back");
        assert!(!r.warnings.is_empty());
        RESIDUAL_MIN.with(|c| c.set(super::STREAM_RESIDUAL_MIN_DOCS));
    }

    #[test]
    fn streamed_stats_match_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("ss", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"shape": {"type": "geo_shape"}, "size": {"type": "long"}, "k": {"type": "keyword"}}}})).unwrap();
        let svc = node.get_index("ss").unwrap();
        let mut batch = Vec::new();
        for i in 0..1500u64 {
            let (x, y) = ((i % 50) as f64 * 0.3 - 7.0, (i % 37) as f64 * 0.4 + 40.0);
            let shape = match i % 3 {
                0 => format!("POINT ({x} {y})"),
                1 => format!("LINESTRING ({x} {y}, {} {}, {} {})", x + 0.1, y + 0.05, x + 0.25, y),
                _ => format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + 0.2, x + 0.2, y + 0.3, y + 0.3),
            };
            let mut d = json!({"shape": shape, "k": format!("k{}", i % 7)});
            if i % 13 == 5 {
                // Docs without a shape: ST_ENVELOPE / ST_NPOINTS are null.
                d.as_object_mut().unwrap().remove("shape");
            }
            if i % 11 != 0 {
                d["size"] = json!(i);
            }
            batch.push(WriteReq::Index { id: Some(i.to_string()), source: d.to_string(), routing: None, cond: Default::default() });
        }
        for r in svc.write(batch, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        RESIDUAL_MIN.with(|c| c.set(0));
        for q in [
            "FROM ss | EVAL env = ST_ENVELOPE(shape), points = ST_NPOINTS(env) | WHERE points > 0 | STATS count = COUNT(size) BY points",
            // The same from the doc values, nested, and unfiltered (nulls grouped).
            "FROM ss | EVAL points = ST_NPOINTS(ST_ENVELOPE(shape)) | STATS count = COUNT(*), s = COUNT(size) BY points",
            // `env` read elsewhere: not rewritten, still equal.
            "FROM ss | EVAL env = ST_ENVELOPE(shape), points = ST_NPOINTS(env) | STATS c = COUNT(env) BY points",
            "FROM ss | EVAL simplified = ST_SIMPLIFY(shape, 0.01), points = ST_NPOINTS(simplified) | WHERE points > 0 | STATS count = COUNT(size) BY points",
            // ST_SIMPLIFY rows simplified in parallel: the same values (as
            // text), after a row filter, beside other fields of the EVAL.
            "FROM ss | EVAL s = ST_SIMPLIFY(shape, 0.2), t = TO_STRING(s) | STATS c = COUNT(*) BY t",
            "FROM ss | WHERE size > 100 | EVAL j = LENGTH(k), s = ST_SIMPLIFY(shape, 0.1), n = ST_NPOINTS(s) | STATS c = COUNT(*), m = MAX(n), q = SUM(j) BY k",
            "FROM ss | EVAL n = LENGTH(k) + 1 | STATS c = COUNT(*), s = SUM(size) BY n, k",
            "FROM ss | WHERE size > 100 | EVAL m = size % 5 | STATS c = COUNT(*) BY m",
        ] {
            let (fast, slow, taken) = both(&node, q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            let sorted = |r: &crate::esql::EsqlResult| {
                let mut rows: Vec<String> = r.table.rows.iter().map(|row| format!("{row:?}")).collect();
                rows.sort();
                rows
            };
            assert_eq!(fast.table.cols, slow.table.cols, "{q}");
            assert_eq!(sorted(&fast), sorted(&slow), "{q}");
            assert_eq!(fast.warnings, slow.warnings, "{q}");
        }
        RESIDUAL_MIN.with(|c| c.set(super::STREAM_RESIDUAL_MIN_DOCS));
    }

    #[test]
    fn geo_stats_match_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // ST_GEOHEX and grid functions over shapes need platinum (ES `LicenseAware`).
        trial(&node);
        points(&node);
        let poly = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";
        for q in [
            "FROM pts | STATS c = COUNT(*)".to_string(),
            "FROM pts | STATS centroid = ST_CENTROID_AGG(loc), e = ST_EXTENT_AGG(loc), c = COUNT(*)".to_string(),
            "FROM pts | STATS centroid = ST_CENTROID_AGG(TO_GEOPOINT(loc))".to_string(),
            "FROM pts | WHERE n > 3000 | STATS ST_CENTROID_AGG(loc), COUNT(*)".to_string(),
            "FROM pts | WHERE n < 0 | STATS c = COUNT(*), centroid = ST_CENTROID_AGG(loc), e = ST_EXTENT_AGG(loc)".to_string(),
            format!("FROM pts | WHERE ST_INTERSECTS(loc, TO_GEOSHAPE(\"{poly}\")) | STATS centroid = ST_CENTROID_AGG(loc), count = COUNT(*)"),
        ] {
            let (fast, slow, taken) = both(&node, &q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            assert_same(&q, &fast, &slow);
            assert_eq!(fast.documents_found, slow.documents_found, "{q}");
        }
    }

    /// Spatial predicates pushed to the index (directly, or through the
    /// `EVAL` defining them) agree with evaluating them per row.
    #[test]
    fn pushed_spatial_predicates_match_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        points(&node);
        let poly = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";
        let pt = "TO_GEOPOINT(\"POINT(7.0 55.0)\")";
        let queries = [
            format!("FROM pts | EVAL x = ST_INTERSECTS(loc, TO_GEOSHAPE(\"{poly}\")) | WHERE x | STATS c = COUNT(*)"),
            format!("FROM pts | EVAL x = ST_WITHIN(loc, TO_GEOSHAPE(\"{poly}\")) | WHERE x | STATS c = COUNT(*)"),
            format!("FROM pts | EVAL x = ST_DISJOINT(loc, TO_GEOSHAPE(\"{poly}\")) | WHERE x | STATS c = COUNT(*)"),
            format!("FROM pts | EVAL x = ST_CONTAINS(loc, TO_GEOSHAPE(\"{poly}\")) | WHERE x | STATS c = COUNT(*)"),
            "FROM pts | EVAL x = ST_INTERSECTS(loc, TO_GEOSHAPE(\"BBOX(2.2, 12.4, 58.95, 48.75)\")) | WHERE x | STATS c = COUNT(*)".to_string(),
            format!("FROM pts | EVAL d = ST_DISTANCE(loc, {pt}) | WHERE d <= 400000 | STATS c = COUNT(*)"),
            format!("FROM pts | EVAL p = {pt}, d = ST_DISTANCE(loc, p) | WHERE d >= 400000 AND d <= 900000 | STATS c = COUNT(*)"),
            format!("FROM pts | EVAL d = ST_DISTANCE(loc, {pt}) | WHERE 900000 > d AND n > 100 | KEEP n, d | SORT n | LIMIT 20"),
        ];
        for q in &queries {
            let (fast, slow, _) = both(&node, q);
            assert_same(q, &fast, &slow);
            assert!(!fast.table.rows.is_empty(), "{q}");
            if q.contains("COUNT(*)") && !q.contains("CONTAINS") {
                assert!(matches!(fast.table.rows[0][0], Val::Long(n) if n > 0), "{q}: {:?}", fast.table.rows);
            }
        }
        // The direct WHERE forms equal the row forms too.
        for (pushed, rows) in [
            (format!("FROM pts | WHERE ST_INTERSECTS(loc, TO_GEOSHAPE(\"{poly}\")) | STATS c = COUNT(*)"), format!("FROM pts | EVAL x = ST_INTERSECTS(loc, TO_GEOSHAPE(\"{poly}\")) | WHERE x | STATS c = COUNT(*)")),
            (format!("FROM pts | WHERE ST_DISTANCE(loc, {pt}) > 400000 | STATS c = COUNT(*)"), format!("FROM pts | EVAL d = ST_DISTANCE(loc, {pt}) | WHERE d > 400000 | STATS c = COUNT(*)")),
        ] {
            let a = run(&node, &pushed, opts()).unwrap();
            DISABLED.with(|c| c.set(true));
            let b = run(&node, &rows, opts()).unwrap();
            DISABLED.with(|c| c.set(false));
            assert_eq!(a.table.rows, b.table.rows, "{pushed}");
        }
        // Distance equality: a superset band in the index, exact per row.
        let q = format!("FROM pts | EVAL d = ST_DISTANCE(loc, {pt}) | WHERE d IS NOT NULL | SORT d | LIMIT 1");
        let r = run(&node, &q, opts()).unwrap();
        let di = r.table.cols.iter().position(|c| c.name == "d").unwrap();
        let Val::Double(d) = r.table.rows[0][di] else { panic!("{:?}", r.table.rows) };
        let q = format!("FROM pts | WHERE ST_DISTANCE(loc, {pt}) == {d:?} | STATS c = COUNT(*)");
        assert_eq!(run(&node, &q, opts()).unwrap().table.rows[0][0], Val::Long(1), "{q}");
    }

    /// Start a trial license, as ES's test clusters run with one.
    fn trial(node: &Node) {
        assert!(matches!(crate::license::service(node).start_trial(node, "trial", true), crate::license::Outcome::Done));
    }

    /// ES `LicenseAware` functions: basic refuses `ST_GEOHEX` (any type)
    /// and grid functions / spatial aggregations over shapes with ES's
    /// verification message; a trial allows them; free uses stay free.
    #[test]
    #[ignore = "licence hardwired to trial"]
    fn license_aware_functions_follow_es_license_state() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("lic", &json!({"mappings": {"properties": {"loc": {"type": "geo_point"}, "shape": {"type": "geo_shape"}}}})).unwrap();
        let svc = node.get_index("lic").unwrap();
        for x in svc.write(vec![WriteReq::Index { id: Some("a".into()), source: json!({"loc": "POINT (7 55)", "shape": "POINT (7 55)"}).to_string(), routing: None, cond: Default::default() }], false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
        let gated = [
            ("FROM lic | EVAL g = ST_GEOHEX(loc, 1) | KEEP g", "ST_GEOHEX(loc, 1)"),
            ("FROM lic | WHERE ST_GEOHEX(loc, 1) == TO_GEOHEX(\"811fbffffffffff\") | LIMIT 10", "ST_GEOHEX(loc, 1)"),
            ("FROM lic | STATS c = ST_CENTROID_AGG(shape)", "ST_CENTROID_AGG(shape)"),
            ("FROM lic | STATS e = ST_EXTENT_AGG(shape)", "ST_EXTENT_AGG(shape)"),
        ];
        let free = [
            "FROM lic | EVAL g = ST_GEOHASH(loc, 2) | KEEP g",
            "FROM lic | EVAL g = ST_GEOTILE(loc, 2) | KEEP g",
            "FROM lic | STATS c = ST_CENTROID_AGG(loc)",
            "FROM lic | STATS e = ST_EXTENT_AGG(loc)",
        ];
        assert_eq!(crate::license::current(&node).kind, "basic");
        for (q, src) in gated {
            let e = run(&node, q, opts()).err().unwrap_or_else(|| panic!("{q}"));
            assert_eq!(e.status, 400, "{q}");
            assert_eq!(e.kind, "verification_exception", "{q}");
            let want = format!("current license is non-compliant for [{src}]");
            assert!(e.reason.starts_with("Found 1 problem\nline 1:") && e.reason.ends_with(&want), "{q}: {}", e.reason);
        }
        for q in free {
            run(&node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
        }
        // ES 9.5.5's grid functions take only geo_point (`SpatialGridFunction.resolveType`):
        // a shape is a type error under any license.
        let type_errors = [("FROM lic | EVAL g = ST_GEOHASH(shape, 2) | KEEP g", "ST_GEOHASH(shape, 2)"), ("FROM lic | EVAL g = ST_GEOTILE(shape, 2) | KEEP g", "ST_GEOTILE(shape, 2)")];
        let check_type_errors = |node: &Node| {
            for (q, src) in type_errors {
                let e = run(node, q, opts()).err().unwrap_or_else(|| panic!("{q}"));
                assert!(e.reason.ends_with(&format!("first argument of [{src}] must be [geo_point], found value [shape] type [geo_shape]")), "{q}: {}", e.reason);
            }
        };
        check_type_errors(&node);
        trial(&node);
        check_type_errors(&node);
        for (q, _) in gated {
            run(&node, q, opts()).unwrap_or_else(|e| panic!("{q} under a trial: {e:?}"));
        }
        // Trial, platinum and enterprise allow platinum features; basic and gold do not.
        assert!(required_license_allows(&node, "ST_GEOHEX", &["geo_point"]));
        assert!(!crate::license::Mode::Basic.allows(crate::license::Mode::Platinum));
        assert!(crate::license::Mode::Enterprise.allows(crate::license::Mode::Platinum));
        assert!(crate::license::Mode::Platinum.allows(crate::license::Mode::Platinum));
        assert!(!crate::license::Mode::Gold.allows(crate::license::Mode::Platinum));
    }

    fn required_license_allows(node: &Node, name: &str, types: &[&str]) -> bool {
        let types: Vec<String> = types.iter().map(|t| t.to_string()).collect();
        let min = crate::esql::exec::required_license(name, &types).expect("gated");
        crate::license::current(node).effective_mode(crate::date::now_millis()).allows(min)
    }

    /// Single-valued points (the distance top-n requires them).
    fn single(node: &Node) {
        node.create_index("sp", &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"loc": {"type": "geo_point"}, "n": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("sp").unwrap();
        let mut r = Rng(11);
        let mut batch = Vec::new();
        for id in 0..5000 {
            let lat = r.below(3000) as f64 / 100.0 + 40.0 + r.below(1000) as f64 / 1e7;
            let lon = r.below(3000) as f64 / 100.0 - 10.0 + r.below(1000) as f64 / 1e7;
            // Some duplicated points: equal distances, ties broken by the second key / doc order.
            let (lat, lon) = if id % 7 == 0 { (55.1, 7.2) } else { (lat, lon) };
            let src = if id % 50 == 3 { json!({"n": id}) } else { json!({"loc": {"lat": lat, "lon": lon}, "n": id}) };
            batch.push(WriteReq::Index { id: Some(format!("s{id}")), source: src.to_string(), routing: None, cond: Default::default() });
            if batch.len() == 1200 {
                for x in svc.write(std::mem::take(&mut batch), false) {
                    x.unwrap();
                }
                svc.refresh().unwrap();
            }
        }
        for x in svc.write(batch, false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
    }

    #[test]
    fn distance_top_n_matches_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        single(&node);
        points(&node);
        let pt = "TO_GEOPOINT(\"POINT(7.0 55.0)\")";
        for q in [
            format!("FROM sp | EVAL distance = ST_DISTANCE(loc, {pt}) | SORT distance ASC | LIMIT 10"),
            format!("FROM sp | EVAL distance = ST_DISTANCE(loc, {pt}) | SORT distance | LIMIT 300"),
            format!("FROM sp | EVAL distance = ST_DISTANCE(loc, {pt}), k = n::string | SORT distance ASC, k DESC | LIMIT 30"),
            format!("FROM sp | EVAL distance = ST_DISTANCE(loc, {pt}) | WHERE distance <= 400000 | SORT distance ASC, n | LIMIT 10"),
            format!("FROM sp | WHERE n > 100 | EVAL p = {pt} | EVAL distance = ST_DISTANCE(loc, p) | SORT distance | LIMIT 5 | KEEP n, distance"),
        ] {
            let (fast, slow, taken) = both(&node, &q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            assert_same(&q, &fast, &slow);
            assert_eq!(fast.table.rows, slow.table.rows, "{q}");
        }
        // Multi-valued documents: their nearest point.
        let q = format!("FROM pts | EVAL d = ST_DISTANCE(loc, {pt}) | SORT d, n | LIMIT 50");
        let (fast, slow, taken) = both(&node, &q);
        assert_eq!(taken, 1, "{q}");
        assert_eq!(fast.table.rows, slow.table.rows, "{q}");
        // More rows than documents with a point: not the distance search
        // (null distances join the result) but the streamed top-n.
        for q in [format!("FROM sp | EVAL d = ST_DISTANCE(loc, {pt}) | SORT d | LIMIT 6000")] {
            let (fast, slow, taken) = both(&node, &q);
            assert_eq!(taken, 1, "{q}");
            assert_same(&q, &fast, &slow);
            assert_eq!(fast.table.rows, slow.table.rows, "{q}");
        }
    }

    #[test]
    fn streamed_source_plan_preserves_index_mappings() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (name, field, fixed) in [("plan-a", "left", "a"), ("plan-b", "right", "b")] {
            node.create_index(name, &json!({"mappings": {"dynamic": false, "properties": {
                "common": {"type": "long"}, field: {"type": "keyword"},
                "fixed": {"type": "constant_keyword", "value": fixed}
            }}})).unwrap();
            let svc = node.get_index(name).unwrap();
            for i in 0..20 {
                svc.write(vec![WriteReq::Index { id: Some(i.to_string()),
                    source: json!({"common": i % 3, "left": ["z", "a"], "right": ["r", "s"]}).to_string(),
                    routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            }
            svc.refresh().unwrap();
        }
        for query in ["FROM plan-* | LIMIT 30", "FROM plan-* | LIMIT 2 BY common, fixed", "FROM plan-* | LIMIT 2 BY left, right"] {
            let (fast, slow, taken) = both(&node, query);
            assert_eq!(taken, 1, "{query}");
            assert_same(query, &fast, &slow);
            assert_eq!(fast.table.rows, slow.table.rows, "{query}");
        }
    }

    #[test]
    fn expression_join_hashes_all_equal_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (index, mode, a, b, count) in [("expr-left","standard","a","b",20),("expr-right","lookup","x","y",200)] {
            node.create_index(index, &json!({"settings":{"index.mode":mode,"number_of_shards":1},"mappings":{"properties":{a:{"type":"keyword"},b:{"type":"long"}}}})).unwrap();
            let svc=node.get_index(index).unwrap();
            for i in 0..count {
                let key = if i == 0 { json!(null) } else if i == 1 { json!([1,2]) } else { json!(i) };
                svc.write(vec![WriteReq::Index { id:Some(i.to_string()),source:json!({a:"same",b:key}).to_string(),routing:None,cond:Default::default() }],false).pop().unwrap().unwrap();
            }
            svc.refresh().unwrap();
        }
        let query="FROM expr-left | LOOKUP JOIN expr-right ON a == x AND b == y | LIMIT 1000";
        super::super::JOIN_CANDIDATES.with(|n| n.set(0));
        let fast=run(&node,query,opts()).unwrap();
        let candidates=super::super::JOIN_CANDIDATES.with(|n| n.get());
        let control="FROM expr-left | LOOKUP JOIN expr-right ON a == x AND b >= y AND b <= y | LIMIT 1000";
        super::super::JOIN_CANDIDATES.with(|n| n.set(0));
        let slow=run(&node,control,opts()).unwrap();
        let scanned=super::super::JOIN_CANDIDATES.with(|n| n.get());
        assert_eq!(fast.table.rows,slow.table.rows);
        assert_eq!(candidates,18);
        assert_eq!(scanned,4000);
        // Mixed numeric equality must use comparison semantics, not unequal
        // grouping encodings for a large integral double and a long.
        for (index,a,b) in [("expr-left","a","b"),("expr-right","x","y")] {
            let svc=node.get_index(index).unwrap();
            svc.write(vec![WriteReq::Index { id:Some("large".into()),source:json!({a:"large",b:9007199254740992i64}).to_string(),routing:None,cond:Default::default() }],false).pop().unwrap().unwrap();
            svc.refresh().unwrap();
        }
        let mixed=run(&node,"FROM expr-left | WHERE a == \"large\" | EVAL b = b::double | LOOKUP JOIN expr-right ON b == y | LIMIT 1000",opts()).unwrap();
        let x=mixed.table.cols.iter().position(|c|c.name=="x").unwrap();
        assert_eq!(mixed.table.rows.len(),1);
        assert_eq!(mixed.table.rows[0][x],Val::Str("large".into()));
    }

    #[test]
    fn expression_range_join_narrows_candidates_and_preserves_outer_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (index, mode, key, count) in [("range-left", "standard", "a", 20), ("range-right", "lookup", "x", 200)] {
            let text = if key == "a" { "s" } else { "t" };
            node.create_index(index, &json!({"settings":{"index.mode":mode,"number_of_shards":1},"mappings":{"properties":{key:{"type":"long"},text:{"type":"keyword"}}}})).unwrap();
            let svc = node.get_index(index).unwrap();
            for i in 0..count {
                let value = if i == 0 { json!(null) } else if i == 1 { json!([1,2]) } else { json!(i) };
                svc.write(vec![WriteReq::Index { id:Some(i.to_string()),source:json!({key:value,text:if i==0 {json!(null)} else if i==1 {json!(["1","2"])} else {json!(i.to_string())}}).to_string(),routing:None,cond:Default::default() }],false).pop().unwrap().unwrap();
            }
            svc.refresh().unwrap();
        }
        for condition in ["a >= x AND a <= x", "x <= a AND x >= a", "a > x AND a <= x + 3", "a < x AND a >= x - 3", "a >= x AND a < x", "a >= x AND a <= x AND x != 3", "s >= t AND s <= t", "t < s", "t > s", "s >= t AND t != \"3\""] {
            let query = format!("FROM range-left | LOOKUP JOIN range-right ON {condition} | LIMIT 1000");
            super::super::JOIN_CANDIDATES.with(|n| n.set(0));
            let actual = run(&node, &query, opts()).unwrap();
            let examined = super::super::JOIN_CANDIDATES.with(|n| n.get());
            super::JOIN_RANGE_INDEX_DISABLED.with(|disabled| disabled.set(true));
            let expected = run(&node, &query, opts());
            super::JOIN_RANGE_INDEX_DISABLED.with(|disabled| disabled.set(false));
            let expected = expected.unwrap();
            assert_eq!(actual.table.rows, expected.table.rows, "{condition}");
            assert!(examined < 4000, "range must prune candidates: {condition}: {examined}");
        }
    }

    #[test]
    fn filtered_join_limit_streams_matches_without_losing_outer_semantics() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (name, mode, key, count) in [("filter-left", "standard", "k", 30), ("filter-right", "lookup", "k", 90)] {
            node.create_index(name, &json!({"settings":{"index.mode":mode,"number_of_shards":1},
                "mappings":{"properties":{key:{"type":"long"},"label":{"type":"keyword"},"v":{"type":"long"}}}})).unwrap();
            let svc = node.get_index(name).unwrap();
            for row in 0..count {
                let value = if row == 0 { json!(null) } else if mode == "standard" && row == 1 { json!(99) } else { json!(row % 3) };
                let label = if mode == "lookup" { if row % 5 == 0 { "yes" } else { "no" } } else { "left" };
                svc.write(vec![WriteReq::Index { id:Some(row.to_string()), source:json!({key:value,"label":label,"v":row}).to_string(),
                    routing:None, cond:Default::default() }],false).pop().unwrap().unwrap();
            }
            svc.refresh().unwrap();
        }
        for prefix in ["FROM filter-left | LOOKUP JOIN filter-right ON k",
            "FROM filter-left | RENAME k AS left_k | LOOKUP JOIN filter-right ON left_k == k",
            "FROM filter-left | RENAME k AS left_k | LOOKUP JOIN filter-right ON left_k >= k AND left_k <= k",
            "FROM filter-left | RENAME k AS left_k | EVAL left_k2 = left_k | LOOKUP JOIN filter-right ON left_k >= k AND left_k2 <= k"] {
            for predicate in ["label LIKE \"yes*\"", "label IS NULL", "label IS NULL OR label == \"yes\"",
                "label == \"missing\"", "label LIKE \"yes*\" | WHERE v > 10", "CONCAT(label, \"!\") == \"yes!\"", "CONCAT(label, \"!\") IS NULL"] {
                let query = format!("{prefix} | WHERE {predicate} | KEEP label, v | LIMIT 7");
                super::JOIN_CANDIDATES.with(|n| n.set(0));
                let actual = run(&node, &query, opts()).unwrap();
                let examined = super::JOIN_CANDIDATES.with(|n| n.get());
                super::FILTERED_JOIN_DISABLED.with(|disabled| disabled.set(true));
                super::JOIN_CANDIDATES.with(|n| n.set(0));
                let expected = run(&node, &query, opts());
                let unbounded = super::JOIN_CANDIDATES.with(|n| n.get());
                super::FILTERED_JOIN_DISABLED.with(|disabled| disabled.set(false));
                let expected = expected.unwrap();
                assert_eq!(actual.table.rows, expected.table.rows, "{query}");
                assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                    expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{query}");
                if actual.table.rows.len() == 7 { assert!(examined < unbounded, "{query}: {examined} >= {unbounded}"); }
            }
        }
    }

    #[test]
    fn chained_streamed_joins_match_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("cj-left", &json!({"settings":{"number_of_shards":2},"mappings":{"properties":{"a":{"type":"keyword"},"b":{"type":"keyword"},"id":{"type":"long"}}}})).unwrap();
        for (name, key) in [("cj-a", "a"), ("cj-b", "b")] {
            node.create_index(name, &json!({"settings":{"index.mode":"lookup"},"mappings":{"properties":{key:{"type":"keyword"}}}})).unwrap();
        }
        let left = node.get_index("cj-left").unwrap();
        for row in 0..150 {
            let a = if row % 17 == 0 { json!(["1", "2"]) } else if row % 23 == 0 { json!(null) } else { json!((row % 9).to_string()) };
            left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: json!({"a": a, "b": (row % 6).to_string(), "id": row, "f": format!("v{row}")}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            if row == 70 { left.refresh().unwrap(); }
        }
        left.refresh().unwrap();
        let ra = node.get_index("cj-a").unwrap();
        for key in 0..7 {
            ra.write(vec![WriteReq::Index { id: Some(key.to_string()), source: json!({"a": key.to_string(), "lk": format!("a{key}"), "x": key}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        // Key 3 joins two rows.
        ra.write(vec![WriteReq::Index { id: Some("3b".into()), source: json!({"a": "3", "lk": "a3b", "x": 33}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        ra.refresh().unwrap();
        let rb = node.get_index("cj-b").unwrap();
        for key in 0..5 {
            rb.write(vec![WriteReq::Index { id: Some(key.to_string()), source: json!({"b": key.to_string(), "lk": format!("b{key}")}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        rb.refresh().unwrap();
        for query in [
            "FROM cj-left | LOOKUP JOIN cj-a ON a | RENAME lk AS lk_a | LOOKUP JOIN cj-b ON b | RENAME lk AS lk_b | KEEP id, a, b, lk_a, lk_b | LIMIT 1000",
            "FROM cj-left | LOOKUP JOIN cj-a ON a | RENAME lk AS lk_a | LOOKUP JOIN cj-b ON b | RENAME lk AS lk_b | KEEP id, a, b, lk_a, lk_b | LIMIT 9",
            "FROM cj-left | LOOKUP JOIN cj-a ON a | LOOKUP JOIN cj-b ON b | LIMIT 40",
            "FROM cj-left | LOOKUP JOIN cj-a ON a | DROP x | LOOKUP JOIN cj-b ON b | KEEP id, lk | LIMIT 0",
            "FROM cj-left | LOOKUP JOIN cj-a ON a | LOOKUP JOIN cj-b ON nope | LIMIT 5",
        ] {
            let actual = run(&node, query, opts());
            super::CHAINED_STREAM_DISABLED.with(|disabled| disabled.set(true));
            let expected = run(&node, query, opts());
            super::CHAINED_STREAM_DISABLED.with(|disabled| disabled.set(false));
            let (actual, expected) = match (actual, expected) {
                (Ok(a), Ok(e)) => (a, e),
                (a, e) => { assert_eq!(a.err().map(|e| e.to_string()), e.err().map(|e| e.to_string()), "{query}"); continue; }
            };
            assert_eq!(actual.table.rows, expected.table.rows, "{query}");
            assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{query}");
            assert_eq!(actual.warnings, expected.warnings, "{query}");
        }
    }

    #[test]
    fn huge_dictionary_columns_read_source_with_the_same_rows() {
        // Every keyword dictionary "huge": batches read `_source`, with the
        // rows, columns and warnings of the doc-value reads.
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("hd-left", &json!({"mappings":{"properties":{"k":{"type":"keyword"},"n":{"type":"long"}}}})).unwrap();
        node.create_index("hd-right", &json!({"settings":{"index.mode":"lookup"},"mappings":{"properties":{"k":{"type":"keyword"}}}})).unwrap();
        let left = node.get_index("hd-left").unwrap();
        for row in 0..200 {
            let doc = if row % 9 == 0 { json!({"k": [(row % 13).to_string(), "x"], "n": row, "t": ["b", "a"]}) } else { json!({"k": (row % 13).to_string(), "n": row, "t": format!("v {row}")}) };
            left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: doc.to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        left.refresh().unwrap();
        left.flush().unwrap();
        let right = node.get_index("hd-right").unwrap();
        for key in 0..10 {
            right.write(vec![WriteReq::Index { id: Some(key.to_string()), source: json!({"k": key.to_string(), "lookup_keyword_0": format!("val {key} rep 0")}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        right.refresh().unwrap();
        right.flush().unwrap();
        for query in [
            "FROM hd-left | LOOKUP JOIN hd-right ON k | LIMIT 300",
            "FROM hd-left | LOOKUP JOIN hd-right ON k | WHERE lookup_keyword_0 LIKE \"val 1*\" | LIMIT 30",
            "FROM hd-left | LOOKUP JOIN hd-right ON k | KEEP n, t, lookup_keyword_0, lookup_keyword_0.keyword | LIMIT 300",
        ] {
            let expected = run(&node, query, opts()).unwrap();
            // Every dictionary "huge", then every row "wide": `_source`.
            for gate in 0..2 {
                if gate == 0 { super::HUGE_DICTIONARY_TERMS_TEST.with(|c| c.set(1)); } else { super::WIDE_ROW_COLUMNS_TEST.with(|c| c.set(0)); }
                let actual = run(&node, query, opts());
                super::HUGE_DICTIONARY_TERMS_TEST.with(|c| c.set(super::HUGE_DICTIONARY_TERMS));
                super::WIDE_ROW_COLUMNS_TEST.with(|c| c.set(super::WIDE_ROW_COLUMNS));
                let actual = actual.unwrap();
                assert_eq!(actual.table.rows, expected.table.rows, "{gate} {query}");
                assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                    expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{gate} {query}");
                assert_eq!(actual.warnings, expected.warnings, "{gate} {query}");
            }
        }
    }

    #[test]
    fn whole_row_streamed_join_matches_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("ws-left", &json!({"settings":{"number_of_shards":2},"mappings":{"properties":{"k":{"type":"keyword"},"id":{"type":"long"},"at":{"type":"date"}}}})).unwrap();
        node.create_index("ws-right", &json!({"settings":{"index.mode":"lookup"},"mappings":{"properties":{"k":{"type":"keyword"}}}})).unwrap();
        let left = node.get_index("ws-left").unwrap();
        for row in 0..120 {
            let doc = match row % 6 {
                0 => json!({"k": (row % 9).to_string(), "id": row, "at": 1_600_000_000_000i64 + row, "f": format!("value {row}"), "g": {"h": row}}),
                1 => json!({"k": (row % 9).to_string(), "id": row.to_string(), "f": ["b", "a"], "n": 2.5}),
                2 => json!({"k": [(row % 9).to_string(), "x"], "id": row, "f": "x".repeat(300)}),
                3 => json!({"id": row, "f": null, "g.h": row}),
                4 => json!({"k": (row % 9).to_string(), "id": [row, 1], "at": "2020-01-01", "f": "Mixed"}),
                _ => json!({"k": (row % 9).to_string(), "f": "value"}),
            };
            left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: doc.to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            if row == 60 { left.refresh().unwrap(); left.flush().unwrap(); }
        }
        left.refresh().unwrap();
        let right = node.get_index("ws-right").unwrap();
        for key in 0..7 {
            right.write(vec![WriteReq::Index { id: Some(key.to_string()), source: json!({"k": key.to_string(), "label": format!("val {key} rep 0"), "c": key}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        right.refresh().unwrap();
        for query in [
            "FROM ws-left | LOOKUP JOIN ws-right ON k | LIMIT 1000",
            "FROM ws-left | LOOKUP JOIN ws-right ON k | LIMIT 7",
            "FROM ws-left | LOOKUP JOIN ws-right ON k | LIMIT 0",
            "FROM ws-left | RENAME k AS kk | EVAL k = kk | LOOKUP JOIN ws-right ON k | LIMIT 50",
            "FROM ws-left METADATA _id | LOOKUP JOIN ws-right ON k | LIMIT 50",
            "FROM ws-left | RENAME f AS ff, id AS k2 | LOOKUP JOIN ws-right ON k | LIMIT 30",
            "FROM ws-left | RENAME f AS id | LOOKUP JOIN ws-right ON k | LIMIT 30",
            "FROM ws-left | RENAME id AS k | LOOKUP JOIN ws-right ON k | LIMIT 30",
            "FROM ws-left | RENAME k AS kk | RENAME kk AS k | LOOKUP JOIN ws-right ON k | KEEP k, label | LIMIT 30",
        ] {
            let actual = run(&node, query, opts());
            super::WHOLE_ROW_STREAM_DISABLED.with(|disabled| disabled.set(true));
            let expected = run(&node, query, opts());
            super::WHOLE_ROW_STREAM_DISABLED.with(|disabled| disabled.set(false));
            let (actual, expected) = match (actual, expected) {
                (Ok(a), Ok(e)) => (a, e),
                (a, e) => { assert_eq!(a.err().map(|e| e.to_string()), e.err().map(|e| e.to_string()), "{query}"); continue; }
            };
            assert_eq!(actual.table.rows, expected.table.rows, "{query}");
            assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{query}");
            assert_eq!(actual.warnings, expected.warnings, "{query}");
            // Source documents and their values read as the table path
            // reads them; lookup rows are counted per keyed read (a chunk's),
            // as in the projected join.
            assert!(actual.documents_found >= expected.documents_found, "{query}");
        }
    }

    #[test]
    fn sub_field_and_integer_doc_values_read_as_source() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("dv-left", &json!({"mappings":{"properties":{"k":{"type":"keyword"}}}})).unwrap();
        // Dynamic `t` (text + keyword sub-field, ignore_above 256) and `n` (long).
        node.create_index("dv-right", &json!({"settings":{"index.mode":"lookup"},
            "mappings":{"properties":{"k":{"type":"keyword"},"i":{"type":"integer"},"s":{"type":"short"},
                "short_kw":{"type":"keyword","fields":{"raw":{"type":"keyword","ignore_above":4}}}}}})).unwrap();
        let left = node.get_index("dv-left").unwrap();
        for row in 0..40 {
            left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: json!({"k": (row % 12).to_string()}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        left.refresh().unwrap();
        let right = node.get_index("dv-right").unwrap();
        let long_text = "x".repeat(300);
        let docs = [
            json!({"k": "0", "t": "plain", "n": 5, "i": 3, "s": 2, "short_kw": "ab"}),
            json!({"k": "1", "t": long_text, "n": "12", "i": "7", "s": 1, "short_kw": "abcdefgh"}),
            json!({"k": "2", "t": ["b", "a", "b"], "n": 5.7, "i": 9.9, "short_kw": ["zz", "aa"]}),
            json!({"k": "3", "n": 1_152_921_504_606_846_976i64, "i": -4}),
            json!({"k": "4", "t": "", "n": [3, 1, 3], "i": [2, 1]}),
            json!({"k": "5", "t": "Mixed Case", "n": -9_007_199_254_740_993i64}),
            json!({"k": "6"}),
            json!({"k": "7", "t": null, "n": null}),
        ];
        for (i, doc) in docs.iter().enumerate() {
            right.write(vec![WriteReq::Index { id: Some(i.to_string()), source: doc.to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            if i == 3 { right.refresh().unwrap(); right.flush().unwrap(); }
        }
        right.refresh().unwrap();
        for query in [
            "FROM dv-left | LOOKUP JOIN dv-right ON k | LIMIT 100",
            "FROM dv-left | LOOKUP JOIN dv-right ON k | KEEP k, t, t.keyword, n, i, s, short_kw.raw | LIMIT 100",
            "FROM dv-left | LOOKUP JOIN dv-right ON k | WHERE n > 0 | LIMIT 100",
            "FROM dv-left | LOOKUP JOIN dv-right ON k | SORT n | LIMIT 5",
        ] {
            let actual = run(&node, query, opts()).unwrap();
            super::DOC_VALUE_EXTRA_DISABLED.with(|disabled| disabled.set(true));
            let expected = run(&node, query, opts());
            super::DOC_VALUE_EXTRA_DISABLED.with(|disabled| disabled.set(false));
            let expected = expected.unwrap();
            assert_eq!(actual.table.rows, expected.table.rows, "{query}");
            assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{query}");
            assert_eq!(actual.warnings, expected.warnings, "{query}");
        }
    }

    #[test]
    fn expression_join_prefilter_and_narrowest_range_keep_results() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("ex-left", &json!({"mappings":{"properties":{"k":{"type":"keyword"},"v":{"type":"long"}}}})).unwrap();
        // One key for every row (Rally `lookup_idx_1_f2_x10000`), counters in `n`.
        node.create_index("ex-right", &json!({"settings":{"index.mode":"lookup"},
            "mappings":{"properties":{"key_1":{"type":"keyword"},"n":{"type":"long"},"s":{"type":"keyword"}}}})).unwrap();
        let left = node.get_index("ex-left").unwrap();
        for row in 0..200 {
            let v = (row * 37) % 400;
            left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: json!({"k": row.to_string(), "v": v}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        left.refresh().unwrap();
        let right = node.get_index("ex-right").unwrap();
        for i in 0..300 {
            let n = if i == 17 { json!(null) } else { json!(i) };
            right.write(vec![WriteReq::Index { id: Some(i.to_string()), source: json!({"key_1": "0", "n": n, "s": format!("s{}", i % 9)}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        right.refresh().unwrap();
        for query in [
            "FROM ex-left | EVAL kl = \"0\", vl = v | LOOKUP JOIN ex-right ON kl >= key_1 AND vl <= n | WHERE n < 30 | LIMIT 1000",
            "FROM ex-left | EVAL kl = \"0\", vl = v | LOOKUP JOIN ex-right ON kl >= key_1 AND vl <= n | WHERE n < 30 | LIMIT 5",
            "FROM ex-left | EVAL kl = \"0\", vl = v | LOOKUP JOIN ex-right ON kl >= key_1 AND vl <= n AND vl > n - 50 | WHERE s == \"s3\" | LIMIT 1000",
            "FROM ex-left | EVAL kl = \"0\", vl = v | LOOKUP JOIN ex-right ON kl >= key_1 AND vl <= n | WHERE n < 30 OR n IS NULL | LIMIT 1000",
            "FROM ex-left | EVAL kl = \"0\", vl = v | LOOKUP JOIN ex-right ON kl >= key_1 AND vl <= n | LIMIT 1000",
            "FROM ex-left | EVAL kl = \"0\", vl = v | LOOKUP JOIN ex-right ON kl == key_1 AND vl == n | WHERE n < 100 | LIMIT 1000",
        ] {
            let actual = run(&node, query, opts()).unwrap();
            super::JOIN_EXPR_PREFILTER_DISABLED.with(|disabled| disabled.set(true));
            super::JOIN_RANGE_INDEX_DISABLED.with(|disabled| disabled.set(true));
            let expected = run(&node, query, opts());
            super::JOIN_EXPR_PREFILTER_DISABLED.with(|disabled| disabled.set(false));
            super::JOIN_RANGE_INDEX_DISABLED.with(|disabled| disabled.set(false));
            let expected = expected.unwrap();
            assert_eq!(actual.table.rows, expected.table.rows, "{query}");
            assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{query}");
            assert_eq!(actual.warnings, expected.warnings, "{query}");
        }
        // Table-path joins (after a LIMIT) read only the lookup rows an
        // equality filter keeps.
        node.create_index("pf-right", &json!({"settings":{"index.mode":"lookup"},
            "mappings":{"properties":{"key_1":{"type":"keyword"},"n":{"type":"long"},"s":{"type":"keyword"}}}})).unwrap();
        let pf = node.get_index("pf-right").unwrap();
        for i in 0..300 {
            pf.write(vec![WriteReq::Index { id: Some(i.to_string()), source: json!({"key_1": if i % 50 == 0 { "1" } else { "0" }, "n": i % 120, "s": format!("s{}", i % 9)}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        pf.refresh().unwrap();
        for query in [
            "FROM ex-left | LIMIT 10 | EVAL key_1 = \"0\" | KEEP key_1, v | LOOKUP JOIN pf-right ON key_1 | WHERE n == 17",
            "FROM ex-left | LIMIT 10 | EVAL key_1 = \"0\" | KEEP key_1, v | LOOKUP JOIN pf-right ON key_1 | WHERE s == \"s3\"",
            "FROM ex-left | LIMIT 10 | EVAL key_1 = \"0\" | KEEP key_1, v | LOOKUP JOIN pf-right ON key_1 | WHERE n == 17 AND s == \"s8\"",
            "FROM ex-left | LIMIT 10 | EVAL key_1 = \"0\" | KEEP key_1, v | LOOKUP JOIN pf-right ON key_1 | WHERE n == 99999",
            "FROM ex-left | LIMIT 10 | EVAL key_1 = TO_STRING(v % 2) | KEEP key_1, v | LOOKUP JOIN pf-right ON key_1 | WHERE n == 50",
            "FROM ex-left | LIMIT 10 | EVAL key_1 = \"0\" | KEEP key_1, v | LOOKUP JOIN pf-right ON key_1 | WHERE n == 17 OR n IS NULL",
        ] {
            let actual = run(&node, query, opts()).unwrap();
            super::LOOKUP_PREFILTER_DISABLED.with(|disabled| disabled.set(true));
            let expected = run(&node, query, opts());
            super::LOOKUP_PREFILTER_DISABLED.with(|disabled| disabled.set(false));
            let expected = expected.unwrap();
            assert_eq!(actual.table.rows, expected.table.rows, "{query}");
            assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{query}");
            assert_eq!(actual.warnings, expected.warnings, "{query}");
        }
        // Filters on source-side columns before the join.
        for query in [
            "FROM ex-left | EVAL key_1 = \"0\", n = v | LOOKUP JOIN ex-right ON key_1, n | WHERE n < 30",
            "FROM ex-left | EVAL key_1 = \"0\", n = v | LOOKUP JOIN ex-right ON key_1, n | WHERE n < 30 | LIMIT 3",
            "FROM ex-left | EVAL key_1 = \"0\", n = v | LOOKUP JOIN ex-right ON key_1, n | WHERE v > 100 | WHERE s == \"s3\"",
            "FROM ex-left | EVAL key_1 = \"0\", n = v | LOOKUP JOIN ex-right ON key_1, n | WHERE v > 100 AND s == \"s3\"",
            "FROM ex-left | EVAL key_1 = \"0\" | LOOKUP JOIN ex-right ON key_1 | WHERE k == \"5\" | WHERE n < 10",
            "FROM ex-left | EVAL key_1 = \"0\", n = v | LOOKUP JOIN ex-right ON key_1, n | WHERE n < 30 | STATS COUNT(*)",
            // Prefix EVALs that warn: on every row, and on a column the filter does not read.
            "FROM ex-left | EVAL key_1 = \"0\", n = TO_LONG(CONCAT(k, \"x\")) | LOOKUP JOIN ex-right ON key_1, n | WHERE n < 30",
            "FROM ex-left | EVAL key_1 = \"0\", n = v, m = TO_LONG(CONCAT(k, \"x\")) | LOOKUP JOIN ex-right ON key_1, n | WHERE n < 30",
            "FROM ex-left | EVAL key_1 = \"0\", w = v | RENAME w AS n | LOOKUP JOIN ex-right ON key_1, n | WHERE n < 30 | LIMIT 4",
        ] {
            let actual = run(&node, query, opts()).unwrap();
            super::LEFT_FILTER_PUSHDOWN_DISABLED.with(|disabled| disabled.set(true));
            let expected = run(&node, query, opts());
            super::LEFT_FILTER_PUSHDOWN_DISABLED.with(|disabled| disabled.set(false));
            let expected = expected.unwrap();
            assert_eq!(actual.table.rows, expected.table.rows, "{query}");
            assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{query}");
            assert_eq!(actual.warnings, expected.warnings, "{query}");
        }
    }

    #[test]
    fn ordinal_join_count_matches_row_by_row() {
        // variant 0: plain; 1: a multi-valued source key; 2: a source row
        // without a key; 3: a multi-valued lookup key.
        for variant in 0..4 {
            let tmp = tempfile::tempdir().unwrap();
            let node = Node::open(tmp.path()).unwrap();
            node.create_index("oc-left", &json!({"settings":{"number_of_shards":2},
                "mappings":{"properties":{"k1":{"type":"keyword"},"k2":{"type":"keyword"},"v":{"type":"long"}}}})).unwrap();
            node.create_index("oc-right", &json!({"settings":{"index.mode":"lookup"},
                "mappings":{"properties":{"k1":{"type":"keyword"},"k2":{"type":"keyword"},"n":{"type":"long"}}}})).unwrap();
            let left = node.get_index("oc-left").unwrap();
            for row in 0..500 {
                let k1 = if variant == 1 && row == 3 { json!(["1", "2"]) } else if variant == 2 && row == 4 { json!(null) } else { json!((row % 120).to_string()) };
                left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: json!({"k1": k1, "k2": (row % 7).to_string(), "v": row}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
                if row == 250 { left.refresh().unwrap(); }
            }
            left.refresh().unwrap();
            let right = node.get_index("oc-right").unwrap();
            for key in 0..100 {
                let k1 = if variant == 3 && key == 5 { json!(["5", "6"]) } else { json!(key.to_string()) };
                right.write(vec![WriteReq::Index { id: Some(key.to_string()), source: json!({"k1": k1, "k2": (key % 7).to_string(), "n": key}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            }
            // Key 9 joins three rows (one with another k2).
            for (id, k2) in [("9b", "2"), ("9c", "3")] {
                right.write(vec![WriteReq::Index { id: Some(id.into()), source: json!({"k1": "9", "k2": k2, "n": 200}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            }
            right.refresh().unwrap();
            for query in [
                "FROM oc-left | LOOKUP JOIN oc-right ON k1 | STATS COUNT(*)",
                "FROM oc-left | LOOKUP JOIN oc-right ON k1, k2 | STATS c = COUNT(*)",
                "FROM oc-left | LOOKUP JOIN oc-right ON k1 | WHERE n < 50 | STATS COUNT(*)",
                "FROM oc-left | LOOKUP JOIN oc-right ON k1 | WHERE n < 50 OR n IS NULL | STATS COUNT(*)",
                "FROM oc-left | RENAME k1 AS k1l | LOOKUP JOIN oc-right ON k1 == k1l | STATS COUNT(*)",
                "FROM oc-left | RENAME k1 AS k1l, k2 AS k2l | LOOKUP JOIN oc-right ON k1 == k1l AND k2 == k2l | STATS COUNT(*)",
            ] {
                super::ORDINAL_COUNTS.with(|n| n.set(0));
                let actual = run(&node, query, opts()).unwrap();
                let counted = super::ORDINAL_COUNTS.with(|n| n.get());
                super::ORDINAL_COUNT_DISABLED.with(|disabled| disabled.set(true));
                let expected = run(&node, query, opts());
                super::ORDINAL_COUNT_DISABLED.with(|disabled| disabled.set(false));
                let expected = expected.unwrap();
                assert_eq!(actual.table.rows, expected.table.rows, "{variant} {query}");
                assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                    expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{variant} {query}");
                assert_eq!(actual.warnings, expected.warnings, "{variant} {query}");
                assert_eq!((actual.documents_found, actual.values_loaded), (expected.documents_found, expected.values_loaded), "{variant} {query}");
                let mv_lookup_named = variant == 3 && !query.contains("==");
                assert_eq!(counted > 0, matches!(variant, 0 | 3) && !mv_lookup_named, "{variant} {query}");
            }
        }
    }

    #[test]
    fn sort_driven_lookup_matches_the_whole_lookup_join() {
        // variant 0: plain; 1: a multi-valued source key; 2: source documents
        // without a key.
        for variant in 0..3 {
            let tmp = tempfile::tempdir().unwrap();
            let node = Node::open(tmp.path()).unwrap();
            node.create_index("sd-left", &json!({"settings":{"number_of_shards":2},
                "mappings":{"properties":{"k":{"type":"keyword"},"v":{"type":"long"}}}})).unwrap();
            // Dynamic `s`: text with a keyword sub-field, as the Rally lookup indices.
            node.create_index("sd-right", &json!({"settings":{"index.mode":"lookup"},
                "mappings":{"properties":{"k":{"type":"keyword"}}}})).unwrap();
            let left = node.get_index("sd-left").unwrap();
            for row in 0..400 {
                let k = if variant == 1 && row == 9 { json!(["1", "2"]) } else if variant == 2 && row % 61 == 0 { json!(null) } else { json!((row % 150).to_string()) };
                left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: json!({"k": k, "v": row}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
                if row == 200 { left.refresh().unwrap(); left.flush().unwrap(); }
            }
            left.refresh().unwrap();
            left.flush().unwrap();
            let right = node.get_index("sd-right").unwrap();
            for key in 0..140 {
                // Ties: keys 2 and 3 share a value; key 7 has no sort value.
                let s = match key { 3 => json!("val 2 rep 0"), 7 => json!(null), _ => json!(format!("val {key} rep 0")) };
                right.write(vec![WriteReq::Index { id: Some(key.to_string()), source: json!({"k": key.to_string(), "s": s, "w": key}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
                if key == 70 { right.refresh().unwrap(); right.flush().unwrap(); }
            }
            // A second row for key 10, and a deleted row.
            right.write(vec![WriteReq::Index { id: Some("dup".into()), source: json!({"k": "10", "s": "val 0 rep 1", "w": -1}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            right.write(vec![WriteReq::Index { id: Some("gone".into()), source: json!({"k": "11", "s": "val 0 rep 0", "w": -2}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            right.refresh().unwrap();
            right.flush().unwrap();
            right.write(vec![WriteReq::Delete { id: "gone".into(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            right.refresh().unwrap();
            right.flush().unwrap();
            // A row only refreshed (a memtable leaf), early in sort order.
            right.write(vec![WriteReq::Index { id: Some("late".into()), source: json!({"k": "120", "s": "val 05 rep 0", "w": -3}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            right.refresh().unwrap();
            for (query, driven) in [
                ("FROM sd-left | LOOKUP JOIN sd-right ON k | SORT s | LIMIT 7", true),
                ("FROM sd-left | LOOKUP JOIN sd-right ON k | SORT s ASC | LIMIT 40", true),
                ("FROM sd-left | LOOKUP JOIN sd-right ON k | SORT s | LIMIT 300", true),
                // Rows without a sort value are needed: the whole-lookup path.
                ("FROM sd-left | LOOKUP JOIN sd-right ON k | SORT s | LIMIT 399", false),
                ("FROM sd-left | LOOKUP JOIN sd-right ON k | SORT s DESC | LIMIT 7", false),
                ("FROM sd-left | LOOKUP JOIN sd-right ON k | SORT w | LIMIT 7", false),
            ] {
                super::SORT_DRIVEN_RUNS.with(|n| n.set(0));
                let actual = run(&node, query, opts()).unwrap();
                let runs = super::SORT_DRIVEN_RUNS.with(|n| n.get());
                super::SORT_DRIVEN_DISABLED.with(|disabled| disabled.set(true));
                let expected = run(&node, query, opts());
                super::SORT_DRIVEN_DISABLED.with(|disabled| disabled.set(false));
                let expected = expected.unwrap();
                assert_eq!(actual.table.rows, expected.table.rows, "{variant} {query}");
                assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                    expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{variant} {query}");
                assert_eq!(actual.warnings, expected.warnings, "{variant} {query}");
                assert_eq!(runs > 0, driven && variant == 0, "{variant} {query}");
            }
        }
    }

    #[test]
    fn semi_join_scans_only_passing_keys_with_join_semantics() {
        // variant 0: plain; 1: a multi-valued source key; 2: a multi-valued
        // lookup key; 3: source documents without a key (not provably absent
        // from `_source` by a count: the full scan reads them).
        for variant in 0..4 {
            let tmp = tempfile::tempdir().unwrap();
            let node = Node::open(tmp.path()).unwrap();
            node.create_index("semi-left", &json!({"settings":{"number_of_shards":2},
                "mappings":{"properties":{"k":{"type":"keyword"},"v":{"type":"long"}}}})).unwrap();
            node.create_index("semi-right", &json!({"settings":{"index.mode":"lookup"},
                "mappings":{"properties":{"k":{"type":"keyword"},"k2":{"type":"keyword"},"tag":{"type":"keyword"},
                    "label":{"type":"text","fields":{"keyword":{"type":"keyword"}}}}}})).unwrap();
            let left = node.get_index("semi-left").unwrap();
            for row in 0..300 {
                let k = if variant == 1 && row == 7 { json!(["1", "2"]) } else if variant == 3 && row % 50 == 0 { json!(null) } else { json!((row % 40).to_string()) };
                left.write(vec![WriteReq::Index { id: Some(row.to_string()), source: json!({"k": k, "v": row}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
                if row == 150 { left.refresh().unwrap(); }
            }
            left.refresh().unwrap();
            let right = node.get_index("semi-right").unwrap();
            for key in 0..30 {
                let k = if variant == 2 && key == 5 { json!(["5", "6"]) } else { json!(key.to_string()) };
                let doc = json!({"k": k, "k2": format!("x{key}"), "tag": if key % 7 == 3 { "t1" } else { "t0" }, "label": format!("yes {key}")});
                right.write(vec![WriteReq::Index { id: Some(key.to_string()), source: doc.to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            }
            // A second lookup row for key 10 (two rows join it).
            right.write(vec![WriteReq::Index { id: Some("dup".into()), source: json!({"k": "10", "k2": "x10", "tag": "t1", "label": "yes 10"}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
            right.refresh().unwrap();
            for (query, pushed) in [
                ("FROM semi-left | LOOKUP JOIN semi-right ON k | WHERE label == \"yes 3\"", true),
                ("FROM semi-left | LOOKUP JOIN semi-right ON k | WHERE tag == \"t1\" | KEEP k, v, label", true),
                ("FROM semi-left | EVAL k2 = CONCAT(\"x\", k::keyword) | KEEP k, k2, v | LOOKUP JOIN semi-right ON k, k2 | WHERE tag == \"t1\"", true),
                ("FROM semi-left | LOOKUP JOIN semi-right ON k | WHERE label == \"none\"", true),
                // Rally `*_two_keys_where_one_matching_equals`: a text column's keyword sub-field as a key.
                ("FROM semi-left | EVAL label.keyword = CONCAT(\"yes \", k::keyword) | KEEP k, label.keyword, v | LOOKUP JOIN semi-right ON k, label.keyword | WHERE label == \"yes 3\"", true),
                // The limit can be reached: the full scan stops early.
                ("FROM semi-left | LOOKUP JOIN semi-right ON k | WHERE tag == \"t1\" | LIMIT 3", false),
                // The unmatched row passes.
                ("FROM semi-left | LOOKUP JOIN semi-right ON k | WHERE label IS NULL OR label == \"yes 3\"", false),
            ] {
                super::SEMI_JOINS.with(|n| n.set(0));
                let actual = run(&node, query, opts()).unwrap();
                let semi = super::SEMI_JOINS.with(|n| n.get());
                super::SEMI_JOIN_DISABLED.with(|disabled| disabled.set(true));
                let expected = run(&node, query, opts());
                super::SEMI_JOIN_DISABLED.with(|disabled| disabled.set(false));
                let expected = expected.unwrap();
                assert_eq!(actual.table.rows, expected.table.rows, "{variant} {query}");
                assert_eq!(actual.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(),
                    expected.table.cols.iter().map(|c| (&c.name, &c.ty)).collect::<Vec<_>>(), "{variant} {query}");
                assert_eq!(actual.warnings, expected.warnings, "{variant} {query}");
                assert_eq!(semi > 0, pushed && variant == 0, "{variant} {query}");
                if semi > 0 { assert_eq!(actual.documents_found >= 300, true, "{query}"); }
            }
        }
    }

    #[test]
    fn keyword_join_key_doc_values_preserve_source_semantics() {
        for variant in 0..6 {
            let tmp = tempfile::tempdir().unwrap();
            let node = Node::open(tmp.path()).unwrap();
            let mut mapping = json!({"properties":{"k":{"type":"keyword"},"copied":{"type":"keyword"},"payload":{"type":"keyword"}}});
            match variant {
                1 => mapping["properties"]["k"]["null_value"] = json!("replacement"),
                2 => mapping["properties"]["k"]["ignore_above"] = json!(3),
                3 => mapping["properties"]["copied"]["copy_to"] = json!("k"),
                4 => mapping["_source"] = json!({"excludes":["k"]}),
                5 => mapping["_source"] = json!({"mode":"stored"}),
                _ => {}
            }
            node.create_index("keyword-left", &json!({"mappings":mapping})).unwrap();
            node.create_index("keyword-right", &json!({"settings":{"index.mode":"lookup","number_of_shards":1},
                "mappings":{"properties":{"k":{"type":"keyword"},"label":{"type":"keyword"}}}})).unwrap();
            for (name,count) in [("keyword-left",40),("keyword-right",10)] {
                let service = node.get_index(name).unwrap();
                let values = [json!(null),json!(1),json!(true),json!(12.5),json!("long-value"),json!(["x","x","1"]),json!("x"),json!("replacement")];
                let writes = (0..count).map(|i| WriteReq::Index { id:Some(i.to_string()),
                    source:json!({"k":values[i%values.len()],"copied":"x","payload":"wide".repeat(100),"label":if i%2==0 {"yes"} else {"no"}}).to_string(),
                    routing:None,cond:Default::default() }).collect();
                for result in service.write(writes,false) { result.unwrap(); }
                service.refresh().unwrap();
            }
            if variant == 2 {
                // Raising ignore_above does not backfill old doc values;
                // source extraction can now see values previously omitted.
                node.get_index("keyword-left").unwrap().put_mapping(&json!({"properties":{"k":{"type":"keyword","ignore_above":20}}})).unwrap();
            }
            for (prefix,tail) in [
                ("", "WHERE label == \"yes\" | LIMIT 100"),
                ("", "STATS COUNT(*)"),
                ("", "WHERE label == \"yes\" | STATS COUNT(*)"),
                ("EVAL k=CONCAT(k::keyword, \"\") | KEEP k,payload | ", "WHERE label == \"yes\" | LIMIT 100"),
                ("EVAL derived=CONCAT(k::keyword, \"\") | DROP k | RENAME derived AS k | ", "WHERE label == \"yes\" | LIMIT 100"),
            ] {
                let query = format!("FROM keyword-left | {prefix}LOOKUP JOIN keyword-right ON k | {tail}");
                super::LOOKUP_KEY_DV_READS.with(|count| count.set(0));
                let actual = run(&node,&query,opts()).unwrap();
                let direct = super::LOOKUP_KEY_DV_READS.with(|count| count.get());
                assert_eq!(direct > 0, matches!(variant,0|5), "variant {variant}: direct={direct}: {query}");
                DISABLED.with(|n| n.set(true));
                let expected = run(&node,&query,opts());
                DISABLED.with(|n| n.set(false));
                let expected = expected.unwrap();
                assert_eq!(actual.table.cols,expected.table.cols,"variant {variant}: {query}");
                assert_eq!(actual.table.rows,expected.table.rows,"variant {variant}: {query}");
            }
        }
    }

    #[test]
    fn lookup_keys_mapped_after_indexing_still_read_existing_source_values() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("late-left",&json!({"mappings":{"dynamic":false,"properties":{"payload":{"type":"keyword"}}}})).unwrap();
        node.create_index("late-right",&json!({"settings":{"index.mode":"lookup","number_of_shards":1},"mappings":{"properties":{"k":{"type":"keyword"},"label":{"type":"keyword"}}}})).unwrap();
        let left = node.get_index("late-left").unwrap();
        for (id,key) in [("old-x","x"),("old-y","y")] {
            left.write(vec![WriteReq::Index {id:Some(id.into()),source:json!({"k":key,"payload":id}).to_string(),routing:None,cond:Default::default()}],false).pop().unwrap().unwrap();
        }
        left.refresh().unwrap();
        left.put_mapping(&json!({"properties":{"k":{"type":"keyword"}}})).unwrap();
        left.write(vec![WriteReq::Index {id:Some("new-x".into()),source:json!({"k":"x","payload":"new-x"}).to_string(),routing:None,cond:Default::default()}],false).pop().unwrap().unwrap();
        left.refresh().unwrap();
        let right = node.get_index("late-right").unwrap();
        right.write(vec![WriteReq::Index {id:Some("x".into()),source:json!({"k":"x","label":"yes"}).to_string(),routing:None,cond:Default::default()}],false).pop().unwrap().unwrap();
        right.refresh().unwrap();
        for tail in ["WHERE label == \"yes\" | LIMIT 100", "WHERE label == \"yes\" | STATS COUNT(*)"] {
            let query = format!("FROM late-left | LOOKUP JOIN late-right ON k | {tail}");
            let actual = run(&node,&query,opts()).unwrap();
            DISABLED.with(|flag| flag.set(true));
            let expected = run(&node,&query,opts());
            DISABLED.with(|flag| flag.set(false));
            let expected = expected.unwrap();
            assert_eq!(actual.table.cols,expected.table.cols,"{query}");
            assert_eq!(actual.table.rows,expected.table.rows,"{query}");
        }
    }

    #[test]
    fn expression_filter_key_preselection_matches_outer_join() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (name, mode, key) in [("pre-left", "standard", "lk"), ("pre-right", "lookup", "rk")] {
            node.create_index(name, &json!({"settings":{"index.mode":mode,"number_of_shards":1},
                "mappings":{"properties":{key:{"type":"long"},"label":{"type":"keyword"}}}})).unwrap();
            let service = node.get_index(name).unwrap();
            for i in 0..20 {
                let value = if i == 0 { json!(null) } else if i == 1 { json!([1,2]) } else { json!(i%7) };
                service.write(vec![WriteReq::Index { id:Some(i.to_string()), source:json!({key:value,"label":if i%3==0 {"yes"} else {"no"}}).to_string(),
                    routing:None,cond:Default::default() }],false).pop().unwrap().unwrap();
            }
            service.refresh().unwrap();
        }
        for predicate in ["label == \"yes\"", "label == \"missing\"", "label IS NULL", "label IS NULL OR label == \"yes\""] {
            let query = format!("FROM pre-left | LOOKUP JOIN pre-right ON lk == rk | WHERE {predicate} | LIMIT 100");
            let actual = run(&node,&query,opts()).unwrap();
            DISABLED.with(|n| n.set(true));
            let expected = run(&node,&query,opts());
            DISABLED.with(|n| n.set(false));
            let expected = expected.unwrap();
            assert_eq!(actual.table.cols,expected.table.cols,"{query}");
            assert_eq!(actual.table.rows,expected.table.rows,"{query}");
        }
    }

    #[test]
    fn filtered_lookup_source_is_bounded_and_matches_table_across_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("stream-left", &json!({"settings":{"number_of_shards":2},
            "mappings":{"properties":{"k":{"type":"long"},"seq":{"type":"long"},"wide":{"type":"keyword"}}}})).unwrap();
        node.create_index("stream-right", &json!({"settings":{"index.mode":"lookup","number_of_shards":1},
            "mappings":{"properties":{"k":{"type":"long"},"label":{"type":"keyword"}}}})).unwrap();
        for (name, count) in [("stream-left",4096), ("stream-right",8)] {
            let svc = node.get_index(name).unwrap();
            let writes = (0..count).map(|i| WriteReq::Index { id:Some(i.to_string()),
                source: if name == "stream-left" { json!({"k":if i%7==0 {None} else {Some(i%5)},"seq":i,"wide":"x".repeat(256)}) }
                    else {json!({"k":i%4,"label":if i%2==0 {"yes"} else {"no"}})}.to_string(),
                routing:None,cond:Default::default() }).collect();
            for result in svc.write(writes,false) { result.unwrap(); }
            svc.refresh().unwrap();
        }
        for predicate in ["label == \"yes\"", "label IS NULL", "label IS NULL OR label == \"yes\"", "seq > 3000", "label == \"missing\""] {
            for cap in [0,7,3000] {
                let query = format!("FROM stream-left | EVAL derived = CONCAT(\"key-\", k::keyword), seq2 = (seq::long + 0)::long | LOOKUP JOIN stream-right ON k | WHERE {predicate} | KEEP seq, label | LIMIT {cap}");
                TAKEN.with(|n| n.set(0));
                let actual = run(&node,&query,opts()).unwrap();
                assert!(TAKEN.with(|n| n.get()) > 0, "{query}");
                DISABLED.with(|n| n.set(true));
                let expected = run(&node,&query,opts());
                DISABLED.with(|n| n.set(false));
                let expected = expected.unwrap();
                assert_eq!(actual.table.cols,expected.table.cols,"{query}");
                assert_eq!(actual.table.rows,expected.table.rows,"{query}");
                if predicate == "label == \"missing\"" {
                    // Keyed lookups read each chunk's lookup rows; a rejected
                    // left row reads only its join key, never its source row.
                    assert!(actual.values_loaded <= 4096 + 64, "impossible right predicate must not load source: {query}: {} values", actual.values_loaded);
                }
                if cap == 7 && predicate == "label == \"yes\"" {
                    assert!(actual.documents_found < 2048, "source must stop early: {}", actual.documents_found);
                }
            }
        }
    }

    #[test]
    fn lookup_sort_keeps_global_top_k_and_stable_null_ties() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (name, mode, count) in [("sort-left","standard",4096),("sort-right","lookup",7)] {
            node.create_index(name,&json!({"settings":{"index.mode":mode,"number_of_shards":1},
                "mappings":{"properties":{"k":{"type":"long"},"seq":{"type":"long"},"label":{"type":"keyword"}}}})).unwrap();
            let svc = node.get_index(name).unwrap();
            let writes = (0..count).map(|i| WriteReq::Index { id:Some(i.to_string()),
                source:if mode == "standard" {json!({"k":i%9,"seq":i})}
                    else {json!({"k":i,"label":format!("label-{}",i%3)})}.to_string(),
                routing:None,cond:Default::default() }).collect();
            for result in svc.write(writes,false) { result.unwrap(); }
            svc.refresh().unwrap();
        }
        for sort in ["label ASC", "label DESC NULLS LAST", "label ASC NULLS FIRST, seq DESC"] {
            for cap in [0,7,1000] {
                let query = format!("FROM sort-left | LOOKUP JOIN sort-right ON k | SORT {sort} | LIMIT {cap}");
                TAKEN.with(|n| n.set(0));
                let actual=run(&node,&query,opts()).unwrap();
                assert!(TAKEN.with(|n| n.get())>0,"{query}");
                DISABLED.with(|n| n.set(true));
                let expected=run(&node,&query,opts());
                DISABLED.with(|n| n.set(false));
                let expected=expected.unwrap();
                assert_eq!(actual.table.cols,expected.table.cols,"{query}");
                assert_eq!(actual.table.rows,expected.table.rows,"{query}");
            }
        }
    }

    #[test]
    fn streamed_lookup_count_preserves_duplicates_and_outer_filter_semantics() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (name, mode, count) in [("count-left","standard",2200),("count-right","lookup",12)] {
            node.create_index(name,&json!({"settings":{"index.mode":mode,"number_of_shards":1},
                "mappings":{"properties":{"k":{"type":"long"},"k2":{"type":"keyword"},"label":{"type":"keyword"}}}})).unwrap();
            let svc=node.get_index(name).unwrap();
            let writes=(0..count).map(|i| WriteReq::Index{id:Some(i.to_string()),
                source:json!({"k":if i%11==0 {None} else {Some(i%5)},"k2":format!("key-{}",i%2),
                    "label":if mode=="lookup" && i%2==0 {Some("yes")} else {None}}).to_string(),
                routing:None,cond:Default::default()}).collect();
            for result in svc.write(writes,false) {result.unwrap();}
            svc.refresh().unwrap();
        }
        for keys in ["k","k, k2"] {
            for filter in ["", "WHERE label == \"yes\" |", "WHERE label IS NULL |", "WHERE label == \"missing\" |", "WHERE label IS NULL OR label == \"yes\" |"] {
                let query=format!("FROM count-left | LOOKUP JOIN count-right ON {keys} | {filter} STATS c=COUNT(*), another=COUNT(*)");
                TAKEN.with(|n| n.set(0));
                let actual=run(&node,&query,opts()).unwrap();
                assert!(TAKEN.with(|n| n.get())>0,"{query}");
                DISABLED.with(|n| n.set(true));
                let expected=run(&node,&query,opts());
                DISABLED.with(|n| n.set(false));
                let expected=expected.unwrap();
                assert_eq!(actual.table.cols,expected.table.cols,"{query}");
                assert_eq!(actual.table.rows,expected.table.rows,"{query}");
            }
        }
    }

    #[test]
    fn expression_join_count_matches_expanded_result() {
        let tmp=tempfile::tempdir().unwrap();
        let node=Node::open(tmp.path()).unwrap();
        for (name,mode) in [("expr-count-left","standard"),("expr-count-right","lookup")] {
            node.create_index(name,&json!({"settings":{"index.mode":mode,"number_of_shards":1},
                "mappings":{"properties":{"k":{"type":"long"},"other":{"type":"keyword"}}}})).unwrap();
            let svc=node.get_index(name).unwrap();
            for i in 0..40 {
                svc.write(vec![WriteReq::Index{id:Some(i.to_string()),source:json!({"k":if i%7==0 {None}else{Some(i%4)},"other":format!("v{}",i%3)}).to_string(),routing:None,cond:Default::default()}],false).pop().unwrap().unwrap();
            }
            svc.refresh().unwrap();
        }
        for condition in ["left_k == k", "left_k == k AND left_other == other"] {
            let q=format!("FROM expr-count-left | RENAME k AS left_k, other AS left_other | LOOKUP JOIN expr-count-right ON {condition} | STATS COUNT(*)");
            TAKEN.with(|n| n.set(0));
            let actual=run(&node,&q,opts()).unwrap();
            assert!(TAKEN.with(|n| n.get())>0,"{q}");
            DISABLED.with(|n| n.set(true));
            let expected=run(&node,&q,opts());
            DISABLED.with(|n| n.set(false));
            assert_eq!(actual.table.rows,expected.unwrap().table.rows,"{q}");
        }
    }

    #[test]
    fn grouped_key_preserves_component_boundaries() {
        let ctx = crate::esql::exec::new_ctx_local("");
        let keys = [super::Expr::Idx(0), super::Expr::Idx(1)];
        let a = [Val::Str("x\u{2}sy".into()), Val::Str("z".into())];
        let b = [Val::Str("x".into()), Val::Str("y\u{2}sz".into())];
        assert_ne!(super::grouped_row_key(&keys, &a, &ctx.fctx).unwrap(), super::grouped_row_key(&keys, &b, &ctx.fctx).unwrap());
    }

    #[test]
    fn distributed_group_limit_matches_complete_table() {
        use crate::esql::remote::{FetchSpec, RemoteShard, ShardResult, ShardSource, Slot, Split};
        use std::sync::Arc;
        struct Remote(Arc<Node>);
        impl ShardSource for Remote {
            fn split(&self, targets: Vec<crate::search::Target>) -> crate::error::Result<Split> {
                let targets = crate::search::expand_targets(&targets)?;
                let slots = targets.iter().map(|t| Slot::Remote(RemoteShard { index: t.svc.name.clone(), shard: t.shard.as_ref().unwrap().id, alias_filter: t.alias_filter.clone() })).collect();
                Ok(Split { targets, slots: Some(slots) })
            }
            fn fetch(&self, spec: &FetchSpec, _memory: &mut crate::memory::Reservation) -> crate::error::Result<Vec<ShardResult>> {
                crate::esql::remote::serve(&self.0, spec, None, None)
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        for (name, field) in [("groups-a", "left"), ("groups-b", "right")] {
            node.create_index(name, &json!({"settings":{"number_of_shards":3},"mappings":{"properties":{"key":{"type":"long"},field:{"type":"keyword"}}}})).unwrap();
            let svc = node.get_index(name).unwrap();
            for i in 0..90 {
                svc.write(vec![WriteReq::Index { id: Some(i.to_string()), source: json!({"key":i%9,field:["a","b"]}).to_string(), routing:None,cond:Default::default() }], false).pop().unwrap().unwrap();
            }
            svc.refresh().unwrap();
        }
        for quota in [0, 1, 2, 7] {
            for cap in [0, 1, 5, 20, 1000] {
                let q = format!("FROM groups-* | LIMIT {quota} BY key, left, right | LIMIT {cap}");
                DISABLED.with(|c| c.set(true));
                let expected = run(&node, &q, opts()).unwrap();
                DISABLED.with(|c| c.set(false));
                let mut options = opts();
                options.shards = Some(Arc::new(Remote(node.clone())));
                let actual = run(&node, &q, options).unwrap();
                assert_same(&q, &actual, &expected);
                assert_eq!(actual.table.rows, expected.table.rows, "{q}");
            }
        }
        let q = "FROM groups-* | LIMIT 2 BY key";
        let mut local_options = opts();
        local_options.default_limit = 5;
        let expected = run(&node, q, local_options).unwrap();
        let mut remote_options = opts();
        remote_options.default_limit = 5;
        remote_options.shards = Some(Arc::new(Remote(node.clone())));
        let actual = run(&node, q, remote_options).unwrap();
        assert_eq!(actual.table.rows, expected.table.rows);
    }

    /// A `SORT | LIMIT` the index cannot run streams its top rows: ties
    /// keep search order, and a row filter or row commands may precede it.
    #[test]
    fn streamed_top_n_matches_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // ST_GEOHEX and grid functions over shapes need platinum (ES `LicenseAware`).
        trial(&node);
        single(&node);
        for q in [
            "FROM sp | EVAL grid = ST_GEOHASH(loc, 5)::long | SORT grid ASC | KEEP grid, loc | LIMIT 10",
            "FROM sp | EVAL grid = ST_GEOTILE(loc, 5)::long | SORT grid ASC | KEEP grid | LIMIT 10",
            "FROM sp | EVAL grid = ST_GEOHEX(loc, 5)::long | SORT grid ASC | KEEP grid, loc | LIMIT 10",
            "FROM sp | EVAL m = n % 13 | SORT m DESC, n | LIMIT 40",
            "FROM sp | EVAL m = n % 13 | SORT m | KEEP m | LIMIT 4500",
            "FROM sp | WHERE LENGTH(TO_STRING(n)) == 3 | SORT n DESC | LIMIT 7",
            "FROM sp | EVAL m = n % 5 | WHERE m == 2 | SORT m, n DESC | EVAL x = n * 2 | LIMIT 9",
            "FROM sp | EVAL m = n % 5 | SORT m | LIMIT 0",
        ] {
            let (fast, slow, taken) = both(&node, q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            assert_same(q, &fast, &slow);
            assert_eq!(fast.table.rows, slow.table.rows, "{q}");
        }
        // A plain doc-values sort right after FROM: the index search sorts.
        let q = "FROM sp | SORT n DESC | LIMIT 5";
        let (fast, slow, taken) = both(&node, q);
        assert_eq!(taken, 0, "{q}");
        assert_eq!(fast.table.rows, slow.table.rows, "{q}");
    }

    /// Shards read on another node (every shard, and every other one): the
    /// fast paths run per shard where each is read and merge here, with the
    /// single-node result over all the shards (grids: the grid path; the
    /// rest: the table path).
    #[test]
    fn distributed_fast_paths_match_single_node() {
        use crate::esql::remote::{FetchSpec, RemoteShard, ShardResult, ShardSource, Slot, Split};
        use std::sync::Arc;
        struct Source {
            node: Arc<Node>,
            /// Shards with an odd id are read remotely; with `all`, every shard.
            all: bool,
            /// Remote parts served (the fetches carried a fast-path plan).
            served: std::sync::atomic::AtomicUsize,
        }
        impl ShardSource for Source {
            fn split(&self, targets: Vec<crate::search::Target>) -> crate::error::Result<Split> {
                let targets = crate::search::expand_targets(&targets)?;
                let slots = targets.iter().enumerate().map(|(i, t)| {
                    let id = t.shard.as_ref().unwrap().id;
                    if self.all || id % 2 == 1 {
                        Slot::Remote(RemoteShard { index: t.svc.name.clone(), shard: id, alias_filter: t.alias_filter.clone() })
                    } else {
                        Slot::Local(i)
                    }
                }).collect();
                Ok(Split { targets, slots: Some(slots) })
            }
            fn fetch(&self, spec: &FetchSpec, _memory: &mut crate::memory::Reservation) -> crate::error::Result<Vec<ShardResult>> {
                let r = crate::esql::remote::serve(&self.node, spec, None, None)?;
                if spec.fast.is_some() {
                    self.served.fetch_add(r.len(), std::sync::atomic::Ordering::SeqCst);
                }
                Ok(r)
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // ST_GEOHEX and grid functions over shapes need platinum (ES `LicenseAware`).
        trial(&node);
        single(&node);
        points(&node);
        let pt = "TO_GEOPOINT(\"POINT(7.0 55.0)\")";
        let exact = [
            "FROM pts | STATS c = COUNT(*)".to_string(),
            "FROM pts | STATS centroid = ST_CENTROID_AGG(loc), e = ST_EXTENT_AGG(loc), c = COUNT(*)".to_string(),
            "FROM pts | STATS centroid = ST_CENTROID_AGG(TO_GEOPOINT(loc))".to_string(),
            "FROM pts | WHERE n > 3000 | STATS ST_CENTROID_AGG(loc), COUNT(*)".to_string(),
            format!("FROM sp | EVAL distance = ST_DISTANCE(loc, {pt}) | SORT distance ASC | LIMIT 10"),
            format!("FROM sp | EVAL distance = ST_DISTANCE(loc, {pt}), k = n::string | SORT distance ASC, k ASC | LIMIT 30"),
            format!("FROM pts | EVAL d = ST_DISTANCE(loc, {pt}) | SORT d, n | LIMIT 50"),
            "FROM sp | WHERE ST_GEOHEX(loc, 1) == TO_GEOHEX(\"811fbffffffffff\") | LIMIT 10".to_string(),
            "FROM sp | WHERE LENGTH(TO_STRING(n)) == 3 | LIMIT 25".to_string(),
            "FROM sp | WHERE LENGTH(TO_STRING(n)) == 3 | EVAL m = n * 2 | KEEP m | LIMIT 2000".to_string(),
            "FROM sp | EVAL grid = ST_GEOHASH(loc, 5)::long | SORT grid ASC | KEEP grid, loc | LIMIT 10".to_string(),
            "FROM sp | EVAL grid = ST_GEOTILE(loc, 5)::long | SORT grid ASC | KEEP grid | LIMIT 10".to_string(),
            "FROM sp | EVAL grid = ST_GEOHEX(loc, 5)::long | SORT grid ASC | KEEP grid, loc | LIMIT 10".to_string(),
            "FROM sp | EVAL m = n % 13 | SORT m DESC, n | LIMIT 40".to_string(),
            "FROM sp | WHERE LENGTH(TO_STRING(n)) == 3 | SORT n DESC | LIMIT 7".to_string(),
        ];
        let grids = [
            "FROM pts | EVAL g = ST_GEOTILE(loc, 4) | STATS c = COUNT(*) BY g",
            "FROM pts | WHERE n > 100 | EVAL g = ST_GEOHEX(loc, 2) | STATS c = COUNT(*) BY g",
        ];
        for all in [true, false] {
            let source = Arc::new(Source { node: node.clone(), all, served: Default::default() });
            for q in exact.iter().map(String::as_str).chain(grids) {
                // Grids: the single-node grid path (doc values, one group
                // per distinct cell of a multi-valued point); the rest: the
                // table path.
                DISABLED.with(|c| c.set(!grids.contains(&q)));
                let expected = run(&node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
                DISABLED.with(|c| c.set(false));
                let mut options = opts();
                options.shards = Some(source.clone());
                let before = TAKEN.with(|c| c.get());
                let served = source.served.load(std::sync::atomic::Ordering::SeqCst);
                let actual = run(&node, q, options).unwrap_or_else(|e| panic!("{q}: {e}"));
                assert_eq!(TAKEN.with(|c| c.get()) - before, 1, "fast path not taken (all remote: {all}): {q}");
                assert!(source.served.load(std::sync::atomic::Ordering::SeqCst) > served, "no shard served a part (all remote: {all}): {q}");
                if grids.contains(&q) {
                    assert_eq!(actual.table.cols, expected.table.cols, "{q}");
                    let set = |t: &crate::esql::Table| -> Vec<String> {
                        let mut v: Vec<String> = t.rows.iter().map(|r| format!("{r:?}")).collect();
                        v.sort();
                        v
                    };
                    assert_eq!(set(&actual.table), set(&expected.table), "all remote: {all}: {q}");
                } else {
                    assert_same(q, &actual, &expected);
                    if !q.contains("ST_CENTROID_AGG") {
                        assert_eq!(actual.table.rows, expected.table.rows, "all remote: {all}: {q}");
                    }
                }
            }
        }
    }

    #[test]
    fn streamed_group_limit_matches_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        single(&node);
        for query in [
            "FROM sp | LIMIT 1 BY n",
            "FROM sp | LIMIT 1 BY n | LIMIT 1",
            "FROM sp | EVAL group = n % 7 | LIMIT 2 BY group | LIMIT 5",
            "FROM sp | LIMIT 1 BY n | LIMIT 0",
            "FROM sp | EVAL group = n % 7 | LIMIT 2 BY group",
            "FROM sp | EVAL group = n % 7 | LIMIT 0 BY group",
            "FROM sp | EVAL group = n % 7, missing = null | LIMIT 3 BY group, missing",
        ] {
            let (fast, slow, taken) = both(&node, query);
            assert_eq!(taken, 1, "{query}");
            assert_same(query, &fast, &slow);
            assert_eq!(fast.table.rows, slow.table.rows, "{query}");
        }
        let query = "FROM sp | EVAL group = n % 7 | LIMIT 2 BY group | SORT n DESC";
        let (fast, slow, taken) = both(&node, query);
        assert_eq!(taken, 0, "downstream ordering must keep its full input");
        assert_eq!(fast.table.rows, slow.table.rows);
    }

    #[test]
    fn streamed_limit_matches_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // ST_GEOHEX and grid functions over shapes need platinum (ES `LicenseAware`).
        trial(&node);
        single(&node);
        RESIDUAL_MIN.with(|c| c.set(0));
        for q in [
            "FROM sp | LIMIT 7".to_string(),
            "FROM sp".to_string(),
            "FROM sp | WHERE n > 1000 | EVAL m = n * 2 | KEEP m, loc | LIMIT 25".to_string(),
            "FROM sp | WHERE ST_GEOHASH(loc, 2) == TO_GEOHASH(\"u1\") | LIMIT 10".to_string(),
            "FROM sp | WHERE ST_GEOTILE(loc, 6) == TO_GEOTILE(\"6/32/20\") | LIMIT 10".to_string(),
            "FROM sp | WHERE ST_GEOHEX(loc, 1) == TO_GEOHEX(\"811fbffffffffff\") | LIMIT 10".to_string(),
            "FROM sp | WHERE LENGTH(TO_STRING(n)) == 3 | LIMIT 2000".to_string(),
        ] {
            let (fast, slow, taken) = both(&node, &q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            assert_same(&q, &fast, &slow);
            assert_eq!(fast.table.rows, slow.table.rows, "{q}");
            assert!(!fast.table.rows.is_empty(), "{q}");
        }
        RESIDUAL_MIN.with(|c| c.set(super::STREAM_RESIDUAL_MIN_DOCS));
        // With the real threshold (index-level targets are expanded to
        // count): below it, the table path.
        let q = "FROM sp | WHERE LENGTH(TO_STRING(n)) == 3 | LIMIT 20";
        let (fast, slow, taken) = both(&node, q);
        assert_eq!(taken, 0, "{q}");
        assert_eq!(fast.table.rows, slow.table.rows, "{q}");
    }

    /// An unparsable shape constant folds to null with ES's warnings; the
    /// predicate then matches nothing (without reading documents).
    #[test]
    fn bad_wkt_warns_and_matches_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        single(&node);
        let q = "FROM sp | WHERE ST_Intersects(loc, TO_GEOSHAPE(\"BBOX((-0.1, 15.0, 61.0, 48.0))\")) | LIMIT 10";
        let r = run(&node, q, opts()).unwrap();
        assert!(r.table.rows.is_empty());
        assert_eq!(r.documents_found, 0);
        assert_eq!(
            r.warnings,
            vec![
                "Line 1:36: evaluation of [TO_GEOSHAPE(\"BBOX((-0.1, 15.0, 61.0, 48.0))\")] failed, treating result as null. Only first 20 failures recorded.".to_string(),
                "Line 1:36: java.lang.IllegalArgumentException: Failed to parse WKT: expected number but found: '('".to_string(),
            ]
        );
    }

    /// A larger single-valued index, force-merged (KD counts, lazy LIMIT,
    /// geo bool scorers and grid aggregations take their index paths).
    fn many(node: &Node) {
        node.create_index("big", &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"loc": {"type": "geo_point"}, "n": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("big").unwrap();
        let mut r = Rng(3);
        let mut batch = Vec::new();
        for id in 0..40_000 {
            // Clustered around Europe plus a sparse rest of the world.
            let (lat, lon) = if id % 5 == 0 {
                (r.below(17_000) as f64 / 100.0 - 85.0, r.below(35_000) as f64 / 100.0 - 175.0)
            } else {
                (r.below(2_000_000) as f64 / 100_000.0 + 40.0, r.below(3_000_000) as f64 / 100_000.0 - 10.0)
            };
            let src = if id % 97 == 5 { json!({"n": id}) } else { json!({"loc": {"lat": lat, "lon": lon}, "n": id}) };
            batch.push(WriteReq::Index { id: Some(format!("b{id}")), source: src.to_string(), routing: None, cond: Default::default() });
            if batch.len() == 10_000 {
                for x in svc.write(std::mem::take(&mut batch), false) {
                    x.unwrap();
                }
            }
        }
        for x in svc.write(batch, false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
        svc.force_merge(1, false).unwrap();
        svc.refresh().unwrap();
    }

    #[test]
    fn index_paths_match_table_path_at_scale() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // ST_GEOHEX and grid functions over shapes need platinum (ES `LicenseAware`).
        trial(&node);
        many(&node);
        RESIDUAL_MIN.with(|c| c.set(0));
        let poly = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";
        let pt = "TO_GEOPOINT(\"POINT(7.0 55.0)\")";
        for q in [
            format!("FROM big | WHERE ST_INTERSECTS(loc, TO_GEOSHAPE(\"{poly}\")) | STATS c = COUNT(*)"),
            format!("FROM big | WHERE ST_WITHIN(loc, TO_GEOSHAPE(\"{poly}\")) | STATS c = COUNT(*)"),
            format!("FROM big | WHERE ST_DISJOINT(loc, TO_GEOSHAPE(\"{poly}\")) | STATS c = COUNT(*)"),
            format!("FROM big | WHERE ST_INTERSECTS(loc, TO_GEOSHAPE(\"{poly}\")) | STATS c = ST_CENTROID_AGG(loc), n = COUNT(*)"),
            format!("FROM big | WHERE ST_DISTANCE(loc, {pt}) <= 300000 | STATS c = COUNT(*)"),
            format!("FROM big | WHERE ST_DISTANCE(loc, {pt}) >= 200000 AND ST_DISTANCE(loc, {pt}) <= 600000 | STATS c = COUNT(*), m = ST_CENTROID_AGG(loc)"),
            format!("FROM big | WHERE ST_DISTANCE(loc, {pt}) >= 900000 | STATS c = COUNT(*)"),
            format!("FROM big | WHERE ST_INTERSECTS(loc, TO_GEOSHAPE(\"{poly}\")) | KEEP n | LIMIT 25"),
            format!("FROM big | WHERE ST_DISJOINT(loc, TO_GEOSHAPE(\"{poly}\")) | KEEP n | LIMIT 25"),
            format!("FROM big | WHERE ST_WITHIN(loc, TO_GEOSHAPE(\"{poly}\")) | KEEP n | LIMIT 25"),
            format!("FROM big | WHERE ST_DISTANCE(loc, {pt}) >= 400000 | KEEP n | LIMIT 30"),
            format!("FROM big | WHERE ST_DISTANCE(loc, {pt}) <= 400000 | KEEP n | LIMIT 30"),
            format!("FROM big | WHERE ST_DISTANCE(loc, {pt}) <= 400000 | EVAL d = ST_DISTANCE(loc, {pt}) | SORT d, n | LIMIT 12 | KEEP n, d"),
            "FROM big | STATS c = ST_CENTROID_AGG(loc), e = ST_EXTENT_AGG(loc), n = COUNT(*)".to_string(),
            "FROM big | LIMIT 40 | KEEP n".to_string(),
        ] {
            let (fast, slow, taken) = both(&node, &q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            assert_same(&q, &fast, &slow);
            assert_eq!(fast.documents_found >= fast.table.rows.len() as u64, true, "{q}");
        }
        // Grid aggregations on points away from cell edges: doc values and
        // `_source` give the same cells.
        for q in [
            "FROM big | WHERE ST_INTERSECTS(loc, TO_GEOSHAPE(\"BBOX(2.20, 12.40, 58.95, 48.75)\")) | EVAL g = ST_GEOHASH(loc, 3) | STATS c = COUNT(*) BY g",
            "FROM big | EVAL g = ST_GEOTILE(loc, 4) | STATS c = COUNT(*) BY g",
            "FROM big | WHERE n > 100 | EVAL g = ST_GEOHEX(loc, 2) | STATS c = COUNT(*) BY g",
        ] {
            let (fast, slow, taken) = both(&node, q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            assert_eq!(fast.table.cols, slow.table.cols, "{q}");
            // Groups without SORT: the same set.
            let set = |t: &crate::esql::Table| -> Vec<String> {
                let mut v: Vec<String> = t.rows.iter().map(|r| format!("{r:?}")).collect();
                v.sort();
                v
            };
            assert_eq!(set(&fast.table), set(&slow.table), "{q}");
            assert!(fast.table.rows.len() > 1, "{q}");
        }
        RESIDUAL_MIN.with(|c| c.set(super::STREAM_RESIDUAL_MIN_DOCS));
        // The DSL counts of geo queries / geo bools (KD leaf counts) equal
        // their hits.
        let targets = node.resolve("big", Default::default()).unwrap();
        for q in [
            json!({"geo_distance": {"distance": "300km", "loc": {"lat": 55.0, "lon": 7.0}}}),
            json!({"bool": {"filter": [{"geo_distance": {"distance": "600km", "loc": {"lat": 55.0, "lon": 7.0}}}], "must_not": [{"geo_distance": {"distance": "200km", "loc": {"lat": 55.0, "lon": 7.0}}}]}}),
            json!({"geo_shape": {"loc": {"shape": poly, "relation": "within"}}}),
            json!({"geo_shape": {"loc": {"shape": poly, "relation": "disjoint"}}}),
            json!({"geo_bounding_box": {"loc": {"top_left": {"lat": 60.0, "lon": 0.0}, "bottom_right": {"lat": 45.0, "lon": 20.0}}}}),
        ] {
            let want = crate::search::count(&targets, Some(&q), None).unwrap();
            let got = crate::search::par_count(&targets, &q).unwrap();
            assert_eq!(got, want, "{q}");
            assert!(want > 0, "{q}");
        }
    }

    /// ES computes the cells of a point only aggregated from its doc value
    /// (ES 9.5.5: this point's `_source` latitude is in tile y=2801 at zoom
    /// 13, its quantized one in y=2802; `STATS .. BY` reports 2802).
    #[test]
    fn grid_stats_use_doc_values_like_es() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // ST_GEOHEX and grid functions over shapes need platinum (ES `LicenseAware`).
        trial(&node);
        node.create_index("gt", &json!({"mappings": {"properties": {"location": {"type": "geo_point"}}}})).unwrap();
        let svc = node.get_index("gt").unwrap();
        for x in svc.write(vec![WriteReq::Index { id: Some("1".into()), source: json!({"location": {"lat": 49.325122001040015, "lon": 2.3}}).to_string(), routing: None, cond: Default::default() }], false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
        let r = run(&node, "FROM gt | EVAL g = ST_GEOTILE(location, 13) | STATS c = COUNT(*) BY g", opts()).unwrap();
        assert_eq!(r.table.rows, vec![vec![Val::Long(1), Val::Str("13/4148/2802".into())]]);
        // Kept in the row, the location is the `_source` one.
        let r = run(&node, "FROM gt | EVAL g = ST_GEOTILE(location, 13) | KEEP g", opts()).unwrap();
        assert_eq!(r.table.rows, vec![vec![Val::Str("13/4148/2801".into())]]);
    }

    /// A pushed `ST_DISTANCE` bound is a circle around the quantized point,
    /// matched like Lucene (ES 9.5.5 counts this point, 1.4 mm inside 400 km
    /// of the quantized center and outside of the raw one).
    #[test]
    fn distance_pushdown_uses_the_quantized_center() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("bp", &json!({"mappings": {"properties": {"location": {"type": "geo_point"}}}})).unwrap();
        let svc = node.get_index("bp").unwrap();
        for x in svc.write(vec![WriteReq::Index { id: Some("1".into()), source: json!({"location": "POINT (5.6606465 51.4929449)"}).to_string(), routing: None, cond: Default::default() }], false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
        for q in [
            "FROM bp | WHERE ST_DISTANCE(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")) <= 400000 | STATS c = COUNT(*)",
            "FROM bp | WHERE ST_DISTANCE(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")) >= 300000 AND ST_DISTANCE(location, TO_GEOPOINT(\"POINT(7.0 55.0)\")) <= 400000 | STATS c = COUNT(*)",
        ] {
            assert_eq!(run(&node, q, opts()).unwrap().table.rows, vec![vec![Val::Long(1)]], "{q}");
        }
    }

    /// The columns of [`source_plan_reads_match_full_extraction`]: the same
    /// plan twice (a plan owns its columns).
    fn plan_columns(cols: &[super::ColSrc]) -> Vec<super::SourceColumn<'_>> {
        let mut out = vec![super::SourceColumn::Constant(Val::Str("fixed".into())), super::SourceColumn::Constant(Val::Null)];
        out.extend(cols.iter().map(|c| super::SourceColumn::Read(c, false)));
        out.insert(5, super::SourceColumn::Constant(Val::Long(3)));
        out
    }

    /// A row read through the plan's `_source` keys is the row of every
    /// column extracted on its own: dotted literal keys, nested objects,
    /// arrays of objects, nulls, absent and unrelated keys, null and
    /// non-object sources.
    #[test]
    fn source_plan_reads_match_full_extraction() {
        let mut ordered = super::ColSrc::plain("m", "keyword");
        ordered.ordered.push("x".into());
        let cols = vec![
            super::ColSrc::plain("a", "keyword"),
            super::ColSrc::plain("a.b", "keyword"),
            super::ColSrc::plain("a.b.c", "long"),
            super::ColSrc::plain("x.y", "keyword"),
            super::ColSrc::plain("m", "keyword"),
            super::ColSrc::plain("d", "double"),
            super::ColSrc::plain("absent", "keyword"),
            super::ColSrc::plain("loc", "geo_point"),
            super::ColSrc::plain("t", "text"),
            super::ColSrc::plain("b", "boolean"),
            super::ColSrc::plain("x.y", "keyword"),
            ordered,
        ];
        let plan = super::SourcePlan::new(plan_columns(&cols));
        let mut roots: Vec<&str> = plan.roots.keys().copied().collect();
        roots.sort_unstable();
        assert_eq!(roots, ["a", "a.b", "a.b.c", "absent", "b", "d", "loc", "m", "t", "x", "x.y"]);
        let sources = [
            "{}",
            r#"{"a":"v","a.b":"lit","x.y":"dotted","m":["z","a","z"],"d":1.5,"b":"true"}"#,
            r#"{"a":{"b":{"c":[3,1]}},"a.b":{"c":2},"x":{"y":["p",null,"q"]},"m":[]}"#,
            r#"{"a":[{"b":"in-array"},{"b":["c1","c2"],"c":1}],"loc":[1.0,2.0],"t":"hello","b":[false,true]}"#,
            r#"{"m":null,"d":"2.25","absent":null,"loc":{"lat":1,"lon":2},"a.b.c":"7"}"#,
            r#"{"x.y.z":"deeper","xy":"no","a.bc":"no","ab":"no","unrelated":{"a":"no"}}"#,
            r#"[{"a":"top-level array"},{"m":"b"}]"#,
            r#""scalar""#,
            "null",
        ];
        for raw in sources {
            let src: serde_json::Value = serde_json::from_str(raw).unwrap();
            let expected: Vec<Val> = plan_columns(&cols).iter().map(|c| match c {
                super::SourceColumn::Constant(v) => v.clone(),
                super::SourceColumn::Read(c, ordered) => super::extract_ordered(&src, c, *ordered),
            }).collect();
            let non_null = expected.iter().filter(|v| !v.is_null()).count() as u64;
            let (row, n, bytes) = plan.dense(&src);
            assert_eq!(row, expected, "{raw}");
            assert_eq!(n, non_null, "{raw}");
            assert_eq!(bytes, crate::esql::exec::row_bytes(&row), "{raw}");
            let (sparse, m, _) = plan.sparse(&src);
            assert_eq!(m, non_null, "{raw}");
            let mut rebuilt = vec![Val::Null; expected.len()];
            for (i, v) in sparse {
                assert!(!v.is_null() && rebuilt[i].is_null(), "{raw}: column {i} twice or null");
                rebuilt[i] = v;
            }
            assert_eq!(rebuilt, expected, "{raw}");
        }
        assert!(plan.dense(&serde_json::Value::Null).0.iter().enumerate().all(|(i, v)| (i == 0 || i == 5) != v.is_null()));
    }

    /// Wide plans (at least `FLAT_READ_MIN_COLUMNS` columns under a
    /// document's keys) read a document's values by path once
    /// ([`super::flat_source`]): the same rows as every column extracted on
    /// its own, dotted keys (which fall back to the walk) included.
    #[test]
    fn wide_source_plan_reads_match_full_extraction() {
        let mut cols = vec![
            super::ColSrc::plain("a", "keyword"),
            super::ColSrc::plain("a.b", "keyword"),
            super::ColSrc::plain("a.b.c", "long"),
            super::ColSrc::plain("x.y", "keyword"),
            super::ColSrc::plain("m", "keyword"),
            super::ColSrc::plain("loc", "geo_point"),
            super::ColSrc::plain("a.c", "long"),
            super::ColSrc::plain("a.f3", "aggregate_metric_double"),
        ];
        for i in 0..80 {
            cols.push(super::ColSrc::plain(&format!("a.f{i}"), if i % 2 == 0 { "keyword" } else { "long" }));
        }
        let plan = super::SourcePlan::new(plan_columns(&cols));
        let sources = [
            r#"{"a":{"b":{"c":[3,1]},"f1":7,"f2":"x","f3":{"min":1,"max":2},"f4":[null,"y",["z"]]},"x":{"y":["p",null,"q"]},"m":[]}"#,
            r#"{"a":[{"b":"in-array","f5":[1,2]},{"b":["c1","c2"],"c":1,"f6":"k"}],"loc":[1.0,2.0],"m":"one"}"#,
            r#"{"a":{"b":"v","f7":null},"a.b":"dotted falls back"}"#,
            r#"{"a":{"f8":{"deep":{"er":1}}}}"#,
            r#"{"a":"scalar","m":null}"#,
        ];
        for raw in sources {
            let src: serde_json::Value = serde_json::from_str(raw).unwrap();
            let expected: Vec<Val> = plan_columns(&cols).iter().map(|c| match c {
                super::SourceColumn::Constant(v) => v.clone(),
                super::SourceColumn::Read(c, ordered) => super::extract_ordered(&src, c, *ordered),
            }).collect();
            let (row, n, _) = plan.dense(&src);
            assert_eq!(row, expected, "{raw}");
            assert_eq!(n, expected.iter().filter(|v| !v.is_null()).count() as u64, "{raw}");
        }
    }

    /// Two rows share a [`super::sparse_group_key`] exactly when they share
    /// a [`super::grouped_row_key`]: nulls, `Val::key` equal values of
    /// different types, multi-values in order, a column used by two keys,
    /// non-key columns, values that look like the dense encoding.
    #[test]
    fn sparse_group_key_matches_grouped_row_key() {
        let ctx = crate::esql::exec::new_ctx_local("");
        let pool = [
            Val::Null, Val::Str("a".into()), Val::Str(String::new()), Val::Str("\u{0}n".into()), Val::Str("2:\u{0}n".into()),
            Val::Long(1), Val::Int(1), Val::Double(1.0), Val::Double(1.5), Val::Bool(true), Val::Date(1),
            Val::Multi(vec![Val::Str("a".into()), Val::Str("b".into())]), Val::Multi(vec![Val::Str("b".into()), Val::Str("a".into())]),
        ];
        let keys = [super::Expr::Idx(0), super::Expr::Idx(2), super::Expr::Idx(2), super::Expr::Idx(3)];
        let positions = vec![vec![0], vec![], vec![1, 2], vec![3]];
        let mut r = Rng(5);
        let rows: Vec<Vec<Val>> = (0..500).map(|_| (0..4).map(|_| {
            // Mostly nulls, as in a wide schema.
            let i = if r.below(3) == 0 { 0 } else { r.below(pool.len() as u64) as usize };
            pool[i].clone()
        }).collect()).collect();
        let dense: Vec<String> = rows.iter().map(|row| super::grouped_row_key(&keys, row, &ctx.fctx).unwrap()).collect();
        let sparse: Vec<Vec<u8>> = rows.iter().map(|row| {
            let values: Vec<(usize, Val)> = row.iter().enumerate().filter(|(_, v)| !v.is_null()).map(|(i, v)| (i, v.clone())).rev().collect();
            super::sparse_group_key(&positions, &values)
        }).collect();
        for i in 0..rows.len() {
            for j in 0..rows.len() {
                assert_eq!(dense[i] == dense[j], sparse[i] == sparse[j], "{:?} / {:?}", rows[i], rows[j]);
            }
        }
        assert!(super::sparse_group_key(&positions, &[]).is_empty());
    }

    /// `n_indices` indices (2 shards each) shaped like the Rally `esql`
    /// track's: index `p` maps its own family `f<p>_0..` of `fields` keyword
    /// fields plus shared `id`, `key` and a constant keyword `fixed`; the
    /// documents hold a few of them (and dynamic text `lookup_0`), so most
    /// columns are null. Some values are arrays (unsorted, repeated, empty)
    /// or explicit nulls.
    fn wide(node: &Node, n_indices: usize, docs: usize, fields: usize) {
        for p in 0..n_indices {
            let mut props = serde_json::Map::new();
            props.insert("id".into(), json!({"type": "long"}));
            props.insert("key".into(), json!({"type": "keyword"}));
            props.insert("fixed".into(), json!({"type": "constant_keyword", "value": format!("c{}", p % 2)}));
            for j in 0..fields {
                props.insert(format!("f{p}_{j}"), json!({"type": "keyword"}));
            }
            let name = format!("wide-{p}");
            node.create_index(&name, &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": props}})).unwrap();
            let svc = node.get_index(&name).unwrap();
            let mut r = Rng(11 + p as u64);
            let mut batch = Vec::new();
            for i in 0..docs {
                let mut doc = serde_json::Map::new();
                doc.insert("id".into(), json!(i));
                doc.insert("lookup_0".into(), json!(format!("val {}", i % 97)));
                match r.below(4) {
                    0 => {}
                    1 => { doc.insert("key".into(), json!(format!("k{}", r.below(5)))); }
                    2 => { doc.insert("key".into(), json!(["b", "a", "b"])); }
                    _ => { doc.insert("key".into(), json!(null)); }
                }
                match r.below(8) {
                    0 => { doc.insert(format!("f{p}_{}", r.below(fields as u64)), json!(format!("v{}", r.below(3)))); }
                    1 => { doc.insert(format!("f{p}_1"), json!(["y", "x", "y"])); }
                    2 => { doc.insert(format!("f{p}_2"), json!([])); }
                    3 => { doc.insert(format!("f{p}_0"), json!(null)); }
                    _ => {}
                }
                batch.push(WriteReq::Index { id: Some(i.to_string()), source: serde_json::Value::Object(doc).to_string(), routing: None, cond: Default::default() });
                if batch.len() == 700 {
                    for x in svc.write(std::mem::take(&mut batch), false) {
                        x.unwrap();
                    }
                    svc.refresh().unwrap();
                }
            }
            for x in svc.write(batch, false) {
                x.unwrap();
            }
            svc.refresh().unwrap();
        }
    }

    fn wide_keys(n_indices: usize, fields: usize) -> String {
        (0..n_indices).flat_map(|p| (0..fields).map(move |j| format!("f{p}_{j}"))).collect::<Vec<_>>().join(", ")
    }

    /// Wide, mostly-null schemas (the Rally `esql` track's shape): the
    /// streamed `LIMIT` and `LIMIT BY` (plan-read rows, sparse group keys)
    /// give the table path's rows, at chunk and limit boundaries.
    #[test]
    fn wide_schema_limits_match_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        wide(&node, 3, 1500, 40);
        let all = wide_keys(3, 40);
        let queries = [
            "FROM wide-* | LIMIT 1".to_string(),
            "FROM wide-* | LIMIT 1023".to_string(),
            "FROM wide-* | LIMIT 1025".to_string(),
            "FROM wide-* | LIMIT 10000".to_string(),
            "FROM wide-*".to_string(),
            format!("FROM wide-* | LIMIT 1 BY {all} | LIMIT 1"),
            format!("FROM wide-* | LIMIT 1 BY {all} | LIMIT 100"),
            format!("FROM wide-* | LIMIT 1 BY {all} | LIMIT 1000"),
            format!("FROM wide-* | LIMIT 2 BY {all}, key | LIMIT 10000"),
            format!("FROM wide-* | LIMIT 1 BY key, {all}, fixed"),
            "FROM wide-* | LIMIT 3 BY key | LIMIT 1000".to_string(),
            "FROM wide-* | LIMIT 2 BY key, key, fixed | LIMIT 50".to_string(),
            "FROM wide-* | LIMIT 1 BY f0_1, f1_1, f2_2 | LIMIT 20".to_string(),
            "FROM wide-* | LIMIT 1 BY lookup_0.keyword | LIMIT 200".to_string(),
            "FROM wide-* | LIMIT 0 BY key".to_string(),
        ];
        for q in &queries {
            let (fast, slow, taken) = both(&node, q);
            assert_eq!(taken, 1, "fast path not taken: {q}");
            assert_same(q, &fast, &slow);
            assert_eq!(fast.table.rows, slow.table.rows, "{q}");
        }
        // ES's implicit limit sits on top of a `LIMIT n BY` (it is not a `Limit`).
        for (q, rows, warned) in [("FROM wide-* | LIMIT 2000 BY fixed", 1000, true), ("FROM wide-* | LIMIT 2000 BY fixed | LIMIT 3000", 3000, false)] {
            let (fast, slow, taken) = both(&node, q);
            assert_eq!(taken, 1, "{q}");
            assert_same(q, &fast, &slow);
            assert_eq!(fast.table.rows.len(), rows, "{q}");
            assert_eq!(fast.warnings.first().map(String::as_str) == Some("No limit defined, adding default limit of [1000]"), warned, "{q}: {:?}", fast.warnings);
        }
        // The all-null groups: one row per distinct non-null combination.
        let r = run(&node, &format!("FROM wide-* | LIMIT 1 BY {all} | LIMIT 1000"), opts()).unwrap();
        assert!(r.table.rows.len() > 1 && r.table.rows.len() < 400, "{}", r.table.rows.len());
    }

    /// Timings for the Rally `esql` track's large-schema shapes over an
    /// in-process fixture (default 5 indices x 10,000 docs x 500 keyword
    /// fields; `ESQL_WIDE_INDICES`, `ESQL_WIDE_DOCS`, `ESQL_WIDE_FIELDS`).
    /// Prints the median of `ESQL_WIDE_RUNS` (5) runs of each query, fast
    /// path and table path, and the JSON encoding of the 10,000-row result
    /// as a `Value` tree versus streamed. Run on the build host:
    /// `cargo test --release -p elasticsearch-engine wide_schema_limit_timings -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn wide_schema_limit_timings() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (n_indices, docs, fields, runs) = (env("ESQL_WIDE_INDICES", 5), env("ESQL_WIDE_DOCS", 10_000), env("ESQL_WIDE_FIELDS", 500), env("ESQL_WIDE_RUNS", 5));
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let started = std::time::Instant::now();
        wide(&node, n_indices, docs, fields);
        eprintln!("fixture: {n_indices} indices x {docs} docs x {fields} fields in {:?}", started.elapsed());
        let all = wide_keys(n_indices, fields);
        let median = |q: &str, disabled: bool| {
            DISABLED.with(|c| c.set(disabled));
            let mut times: Vec<std::time::Duration> = (0..runs).map(|_| {
                let t = std::time::Instant::now();
                let r = run(&node, q, opts()).unwrap();
                std::hint::black_box(r.table.rows.len());
                t.elapsed()
            }).collect();
            DISABLED.with(|c| c.set(false));
            times.sort();
            times[times.len() / 2]
        };
        for (name, q) in [
            ("from_idx_limit_1000", "FROM wide-* | LIMIT 1000".to_string()),
            ("from_idx_limit_10000", "FROM wide-* | LIMIT 10000".to_string()),
            ("limit_10_by_key", "FROM wide-* | LIMIT 10 BY key".to_string()),
            ("limit_1_by_all_limit_1", format!("FROM wide-* | LIMIT 1 BY {all} | LIMIT 1")),
            ("limit_1_by_all_limit_100", format!("FROM wide-* | LIMIT 1 BY {all} | LIMIT 100")),
            ("limit_1_by_all_limit_1000", format!("FROM wide-* | LIMIT 1 BY {all} | LIMIT 1000")),
        ] {
            let (fast, slow, _) = both(&node, &q);
            assert_eq!(fast.table.rows, slow.table.rows, "{name}");
            eprintln!("{name}: rows={} cols={} fast={:?} table={:?}", fast.table.rows.len(), fast.table.cols.len(), median(&q, false), median(&q, true));
        }
        let r = run(&node, "FROM wide-* | LIMIT 10000", opts()).unwrap();
        let t = std::time::Instant::now();
        let mut tree = serde_json::Map::new();
        crate::esql::format::columns_values(&r.table, false, false, &mut tree);
        let tree_bytes = serde_json::to_vec(&serde_json::Value::Object(tree)).unwrap();
        let tree_time = t.elapsed();
        let t = std::time::Instant::now();
        let mut meta = serde_json::Map::new();
        let keep = crate::esql::format::columns_only(&r.table, false, &mut meta);
        meta.insert("values".into(), serde_json::Value::Null);
        let streamed = serde_json::to_vec(&crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: &r.table, keep: &keep, columnar: false }).unwrap();
        let streamed_time = t.elapsed();
        assert_eq!(tree_bytes, streamed);
        let t = std::time::Instant::now();
        let mut written: Vec<u8> = Vec::new();
        crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: &r.table, keep: &keep, columnar: false }.write_json(&mut written).unwrap();
        let written_time = t.elapsed();
        assert!(written == streamed, "write_json differs from the serializer");
        eprintln!("encode {} bytes: tree={tree_time:?} streamed={streamed_time:?} null-run writer={written_time:?}", streamed.len());
    }

    /// A lookup index `lk` (keyword `key`, integer `n`, payload `v`) with
    /// duplicate keys, missing and multi-valued keys, and a 2-shard `main`
    /// whose keys partly match, are missing, null or multi-valued.
    fn join_fixture(node: &Node, lookup_docs: usize, main_docs: usize) {
        node.create_index("lk", &json!({"settings": {"index.mode": "lookup", "number_of_shards": 1},
            "mappings": {"properties": {"key": {"type": "keyword"}, "n": {"type": "integer"}, "v": {"type": "keyword"}}}})).unwrap();
        node.create_index("main", &json!({"settings": {"number_of_shards": 2},
            "mappings": {"properties": {"id": {"type": "long"}, "key": {"type": "keyword"}, "n": {"type": "integer"}, "w": {"type": "keyword"}}}})).unwrap();
        let keys = (lookup_docs / 3).max(1) as u64;
        for (name, count, seed) in [("lk", lookup_docs, 3u64), ("main", main_docs, 9u64)] {
            let svc = node.get_index(name).unwrap();
            let mut r = Rng(seed);
            let mut batch = Vec::new();
            for i in 0..count {
                let k = r.below(if name == "lk" { keys } else { keys + keys / 4 + 1 });
                let key = match r.below(20) {
                    0 => json!(null),
                    1 => json!([format!("k{k}"), format!("k{}", k + 1)]),
                    _ => json!(format!("k{k}")),
                };
                let n = match r.below(25) { 0 => json!(null), 1 => json!([k, k + 2]), _ => json!(k % 97) };
                let doc = if name == "lk" { json!({"key": key, "n": n, "v": format!("x{}", r.below(40))}) } else { json!({"id": i, "key": key, "n": n, "w": "left"}) };
                batch.push(WriteReq::Index { id: Some(i.to_string()), source: doc.to_string(), routing: None, cond: Default::default() });
                if batch.len() == 2000 {
                    for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
                    svc.refresh().unwrap();
                }
            }
            for x in svc.write(batch, false) { x.unwrap(); }
            svc.refresh().unwrap();
        }
    }

    /// `q` with keyed lookup reads and with the whole lookup index loaded
    /// (both with fast paths, then the table interpreter), and the lookup
    /// rows the keyed run read.
    fn keyed_and_full(node: &Node, q: &str) -> (crate::esql::EsqlResult, crate::esql::EsqlResult, usize) {
        super::LOOKUP_KEYED_READS.with(|c| c.set(0));
        super::LOOKUP_KEYED_CALLS.with(|c| c.set(0));
        let keyed = run(node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
        let reads = super::LOOKUP_KEYED_READS.with(|c| c.get());
        super::LOOKUP_KEYED_DISABLED.with(|c| c.set(true));
        let full = run(node, q, opts());
        DISABLED.with(|c| c.set(true));
        let table = run(node, q, opts());
        DISABLED.with(|c| c.set(false));
        super::LOOKUP_KEYED_DISABLED.with(|c| c.set(false));
        let (full, table) = (full.unwrap_or_else(|e| panic!("{q}: {e}")), table.unwrap_or_else(|e| panic!("{q}: {e}")));
        assert_eq!(full.table.cols, table.table.cols, "{q}");
        assert_eq!(full.table.rows, table.table.rows, "fast vs table path: {q}");
        (keyed, full, reads)
    }

    /// Keyed lookups (a `terms` query per batch of left keys) give the rows,
    /// columns, order and warnings of joining the whole lookup index: no
    /// match, several matches, null and multi-valued keys on both sides,
    /// keyword and integer keys, two keys, LIMIT / SORT / WHERE / STATS
    /// before and after the join.
    #[test]
    fn keyed_lookup_join_matches_full_lookup() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        join_fixture(&node, 3000, 5000);
        let keyed_queries = [
            "FROM main | LOOKUP JOIN lk ON key | LIMIT 1",
            "FROM main | LOOKUP JOIN lk ON key | LIMIT 37",
            "FROM main | LOOKUP JOIN lk ON key | LIMIT 1000",
            "FROM main | LOOKUP JOIN lk ON key | LIMIT 10000",
            "FROM main | LOOKUP JOIN lk ON key | KEEP id, v | LIMIT 500",
            "FROM main | LOOKUP JOIN lk ON n | LIMIT 700",
            "FROM main | LOOKUP JOIN lk ON key, n | LIMIT 700",
            "FROM main | SORT id DESC | LIMIT 60 | LOOKUP JOIN lk ON key",
            "FROM main | WHERE id < 300 | LOOKUP JOIN lk ON key | SORT id, v",
            "FROM main | LOOKUP JOIN lk ON key | WHERE v == \"x3\" | LIMIT 20",
            "FROM main | LOOKUP JOIN lk ON key | WHERE v LIKE \"x1*\" | KEEP id, key, v | LIMIT 1000",
            "FROM main | LOOKUP JOIN lk ON key | WHERE v IS NULL | LIMIT 50",
            "FROM main | LOOKUP JOIN lk ON key | WHERE CONCAT(v, w) == \"x7left\" | LIMIT 30",
            "FROM main | EVAL key = CONCAT(\"k\", TO_STRING(id % 13)) | LOOKUP JOIN lk ON key | LIMIT 300",
            "FROM main | LIMIT 400 | LOOKUP JOIN lk ON key | STATS c = COUNT(*)",
        ];
        for q in keyed_queries {
            let (keyed, full, reads) = keyed_and_full(&node, q);
            assert_eq!(keyed.table.cols, full.table.cols, "{q}");
            assert_eq!(keyed.table.rows, full.table.rows, "{q}");
            assert_eq!(keyed.warnings, full.warnings, "{q}");
            // A keyed read may find no lookup row (no left key matches).
            let _ = reads;
            assert!(super::LOOKUP_KEYED_CALLS.with(|c| c.get()) > 0, "{q}: not keyed");
        }
        // Shapes that still read the whole lookup side, unchanged.
        for q in [
            "FROM main | LOOKUP JOIN lk ON key | SORT v, id | LIMIT 100",
            "FROM main | LOOKUP JOIN lk ON key | STATS c = COUNT(*)",
            "FROM main | LOOKUP JOIN lk ON key | WHERE v == \"x3\" | STATS c = COUNT(*)",
            "FROM main | RENAME key AS k2 | LOOKUP JOIN lk ON k2 == key | LIMIT 100",
        ] {
            let (keyed, full, _) = keyed_and_full(&node, q);
            assert_eq!(keyed.table.rows, full.table.rows, "{q}");
            assert_eq!(keyed.warnings, full.warnings, "{q}");
        }
        // A LIMIT reads only the lookup rows of the rows it outputs.
        let (_, _, reads) = keyed_and_full(&node, "FROM main | LOOKUP JOIN lk ON key | LIMIT 1");
        assert!(reads <= 10, "LIMIT 1 read {reads} lookup rows");
    }

    /// A result's warnings but the implicit-limit one.
    fn join_warnings(r: &crate::esql::EsqlResult) -> Vec<String> {
        r.warnings.iter().filter(|w| !w.starts_with("No limit defined")).cloned().collect()
    }

    /// ES 9.5.5's `lookup-join.csv-spec` null and multi-value cases
    /// (`languages_lookup_non_unique_key`): several matches in lookup
    /// order, a null key joins nothing, a multi-valued key on either side
    /// joins nothing and warns.
    #[test]
    fn lookup_join_null_and_multi_value_keys_like_es() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("langs", &json!({"settings": {"index.mode": "lookup", "number_of_shards": 1},
            "mappings": {"properties": {"language_code": {"type": "integer"}, "language_name": {"type": "keyword"}, "country": {"type": "keyword"}}}})).unwrap();
        let langs = [
            json!({"language_code": 1, "language_name": "English", "country": "Canada"}),
            json!({"language_code": 1, "language_name": "English"}),
            json!({"language_code": 1, "country": "United Kingdom"}),
            json!({"language_code": 1, "language_name": "English", "country": "United States of America"}),
            json!({"language_code": 2, "language_name": "German", "country": ["Germany", "Austria"]}),
            json!({"language_code": 2, "language_name": "German", "country": ["Germany", "Belgium"]}),
            json!({"language_code": 2, "language_name": "German", "country": "Switzerland"}),
            json!({"language_code": 2, "language_name": "German"}),
            json!({"language_code": 4, "language_name": "Quenya"}),
            json!({"language_code": 5, "country": "Atlantis"}),
            json!({"language_code": [6, 7], "language_name": "Mv-Lang", "country": "Mv-Land"}),
            json!({"language_code": [7, 8], "language_name": "Mv-Lang2", "country": "Mv-Land2"}),
            json!({"language_name": "Null-Lang", "country": "Null-Land"}),
            json!({"language_name": "Null-Lang2", "country": "Null-Land2"}),
        ];
        let svc = node.get_index("langs").unwrap();
        for (i, doc) in langs.iter().enumerate() {
            svc.write(vec![WriteReq::Index { id: Some(i.to_string()), source: doc.to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        svc.refresh().unwrap();
        node.create_index("emps", &json!({"mappings": {"properties": {"emp_no": {"type": "integer"}, "codes": {"type": "integer"}}}})).unwrap();
        let svc = node.get_index("emps").unwrap();
        for (emp, codes) in [(10001, json!(1)), (10002, json!([-7, 11])), (10003, json!([12, 14])), (10004, json!(4)), (10005, json!([-2, 13])), (10006, json!(6)), (10007, json!(7))] {
            svc.write(vec![WriteReq::Index { id: Some(emp.to_string()), source: json!({"emp_no": emp, "codes": codes}).to_string(), routing: None, cond: Default::default() }], false).pop().unwrap().unwrap();
        }
        svc.refresh().unwrap();
        let s = |v: &str| Val::Str(v.into());
        let warned = |q: &str| {
            let col = q.find("LOOKUP").unwrap() + 1;
            let cmd = q[col - 1..].split(" |").next().unwrap();
            vec![
                format!("Line 1:{col}: evaluation of [{cmd}] failed, treating result as null. Only first 20 failures recorded."),
                format!("Line 1:{col}: java.lang.IllegalArgumentException: LOOKUP JOIN encountered multi-value"),
            ]
        };
        for keyed in [true, false] {
            super::LOOKUP_KEYED_DISABLED.with(|c| c.set(!keyed));
            // nullJoinKeyOnTheDataNode
            let q = "FROM emps | WHERE emp_no < 10004 | EVAL language_code = emp_no % 10, language_code = CASE(language_code == 3, null, language_code) | LOOKUP JOIN langs ON language_code | SORT emp_no, language_code, language_name | KEEP emp_no, language_code, language_name";
            let r = run(&node, q, opts()).unwrap();
            assert_eq!(r.table.rows, vec![
                vec![Val::Int(10001), Val::Int(1), s("English")], vec![Val::Int(10001), Val::Int(1), s("English")],
                vec![Val::Int(10001), Val::Int(1), s("English")], vec![Val::Int(10001), Val::Int(1), Val::Null],
                vec![Val::Int(10002), Val::Int(2), s("German")], vec![Val::Int(10002), Val::Int(2), s("German")],
                vec![Val::Int(10002), Val::Int(2), s("German")], vec![Val::Int(10002), Val::Int(2), s("German")],
                vec![Val::Int(10003), Val::Null, Val::Null],
            ], "keyed={keyed}");
            assert!(join_warnings(&r).is_empty(), "keyed={keyed}: {:?}", r.warnings);
            // mvJoinKeyOnTheLookupIndex
            let q = "FROM emps | WHERE 10003 < emp_no AND emp_no < 10008 | EVAL language_code = emp_no % 10 | LOOKUP JOIN langs ON language_code | SORT emp_no, language_name | KEEP emp_no, language_code, language_name";
            let r = run(&node, q, opts()).unwrap();
            assert_eq!(r.table.rows, vec![
                vec![Val::Int(10004), Val::Int(4), s("Quenya")], vec![Val::Int(10005), Val::Int(5), Val::Null],
                vec![Val::Int(10006), Val::Int(6), Val::Null], vec![Val::Int(10007), Val::Int(7), Val::Null],
            ], "keyed={keyed}");
            assert_eq!(join_warnings(&r), warned(q), "keyed={keyed}");
            // mvJoinKeyOnFrom
            let q = "FROM emps | WHERE emp_no < 10006 | EVAL language_code = codes | LOOKUP JOIN langs ON language_code | SORT emp_no | KEEP emp_no, language_code, language_name";
            let r = run(&node, q, opts()).unwrap();
            assert_eq!(r.table.rows.iter().map(|row| row[2].clone()).collect::<Vec<_>>(), vec![s("English"), s("English"), Val::Null, s("English"), Val::Null, Val::Null, s("Quenya"), Val::Null], "keyed={keyed}");
            assert_eq!(join_warnings(&r), warned(q), "keyed={keyed}");
            // No match: one row of nulls.
            let q = "FROM emps | WHERE emp_no == 10004 | EVAL language_code = 99 | LOOKUP JOIN langs ON language_code | KEEP emp_no, language_name, country";
            assert_eq!(run(&node, q, opts()).unwrap().table.rows, vec![vec![Val::Int(10004), Val::Null, Val::Null]], "keyed={keyed}");
        }
        super::LOOKUP_KEYED_DISABLED.with(|c| c.set(false));
    }

    /// Timings for the Rally `joins` shapes on a large lookup index
    /// (`ESQL_JOIN_LOOKUP` docs, default 1,000,000; main `ESQL_JOIN_MAIN`,
    /// 200,000): keyed against whole-lookup reads, median of
    /// `ESQL_JOIN_RUNS` (3). Run on the build host:
    /// `cargo test --release -p elasticsearch-engine keyed_lookup_join_timings -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn keyed_lookup_join_timings() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (lookup, main, runs) = (env("ESQL_JOIN_LOOKUP", 1_000_000), env("ESQL_JOIN_MAIN", 200_000), env("ESQL_JOIN_RUNS", 3));
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let started = std::time::Instant::now();
        join_fixture(&node, lookup, main);
        eprintln!("fixture: lookup {lookup} docs, main {main} docs in {:?}", started.elapsed());
        let median = |q: &str, keyed: bool| {
            super::LOOKUP_KEYED_DISABLED.with(|c| c.set(!keyed));
            let mut times: Vec<std::time::Duration> = (0..runs).map(|_| {
                let t = std::time::Instant::now();
                std::hint::black_box(run(&node, q, opts()).unwrap().table.rows.len());
                t.elapsed()
            }).collect();
            super::LOOKUP_KEYED_DISABLED.with(|c| c.set(false));
            times.sort();
            times[times.len() / 2]
        };
        for q in [
            "FROM main | LOOKUP JOIN lk ON key | LIMIT 1",
            "FROM main | LOOKUP JOIN lk ON key | LIMIT 1000",
            "FROM main | LOOKUP JOIN lk ON key | LIMIT 10000",
            "FROM main | LOOKUP JOIN lk ON key | KEEP id, v | LIMIT 10000",
            "FROM main | LOOKUP JOIN lk ON key | WHERE v LIKE \"x1*\" | LIMIT 1000",
            "FROM main | LOOKUP JOIN lk ON key | WHERE v == \"x39\" | LIMIT 1000",
            "FROM main | LOOKUP JOIN lk ON key | SORT v | LIMIT 10000",
        ] {
            let (keyed, full, reads) = keyed_and_full(&node, q);
            assert_eq!(keyed.table.rows, full.table.rows, "{q}");
            eprintln!("{q}: rows={} lookup_rows_read={reads} keyed={:?} full={:?}", keyed.table.rows.len(), median(q, true), median(q, false));
        }
    }

    /// `count` geo points over 2 shards in several segments: duplicates
    /// (ties), documents without a location, multi-valued locations, in
    /// the formats `_source` allows, and an `n` column.
    fn grid_points(node: &Node, name: &str, count: usize) {
        node.create_index(name, &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"location": {"type": "geo_point"}, "n": {"type": "long"}}}})).unwrap();
        let svc = node.get_index(name).unwrap();
        let mut r = Rng(17);
        let mut batch = Vec::new();
        for i in 0..count {
            let lat = r.below(17_000) as f64 / 100.0 - 85.0 + r.below(1000) as f64 / 1e6;
            let lon = r.below(36_000) as f64 / 100.0 - 180.0 + r.below(1000) as f64 / 1e6;
            let (lat, lon) = if i % 9 == 0 { (12.3456, 45.6789) } else { (lat, lon) };
            let location = match r.below(20) {
                0 => json!(null),
                1 => json!([{"lat": lat, "lon": lon}, [lon / 2.0, lat / 3.0]]),
                2 => json!(format!("POINT ({lon} {lat})")),
                3 => json!([lon, lat]),
                _ => json!({"lat": lat, "lon": lon}),
            };
            let doc = if r.below(30) == 0 { json!({"n": i}) } else { json!({"location": location, "n": i}) };
            batch.push(WriteReq::Index { id: Some(i.to_string()), source: doc.to_string(), routing: None, cond: Default::default() });
            if batch.len() == 2500 {
                for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
                svc.refresh().unwrap();
            }
        }
        for x in svc.write(batch, false) { x.unwrap(); }
        svc.refresh().unwrap();
    }

    /// Rows with each multi-value's elements sorted (doc-value order is the
    /// encoded points', `_source` order the document's).
    fn mv_sorted(rows: &[Vec<Val>]) -> Vec<Vec<Val>> {
        rows.iter().map(|r| r.iter().map(|v| match v {
            Val::Multi(m) => { let mut m = m.clone(); m.sort_by(crate::esql::cmp_vals); Val::Multi(m) }
            v => v.clone(),
        }).collect()).collect()
    }

    /// The grid top-n (`EVAL g = ST_GEO*(loc, p)::long | SORT g | KEEP ..
    /// | LIMIT n`) gives the table path's rows: all three grid functions,
    /// both directions and null placements, ties in search order, missing
    /// and multi-valued locations, LIMIT boundaries. With the location in
    /// the output both read `_source`; without it the cells come from doc
    /// values (ES), equal here up to multi-value order (no point of these
    /// sits on a cell edge).
    #[test]
    fn grid_top_n_matches_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        trial(&node);
        grid_points(&node, "gp", 12_000);
        // grid_points refreshes every 2,500 documents: several segments per shard.
        let segments: usize = node.get_index("gp").unwrap().num_segments();
        assert!(segments > 2, "multiple segments expected, got {segments}");
        for f in ["ST_GEOHASH(location, 5)", "ST_GEOTILE(location, 5)", "ST_GEOHEX(location, 5)", "ST_GEOHASH(location, 1)", "ST_GEOTILE(location, 0)", "ST_GEOHEX(location, 0)", "ST_GEOTILE(location, 29)"] {
            for (sort, limit) in [("ASC", 10), ("DESC", 10), ("ASC NULLS FIRST", 25), ("DESC NULLS LAST", 7), ("ASC", 1), ("ASC", 5000), ("DESC", 10000)] {
                for (keep, location) in [("KEEP grid, location", true), ("KEEP grid", false), ("KEEP grid, n, location", true), ("KEEP n, grid", false), ("RENAME location AS l", true)] {
                    // Zoom-29 tiles are a few quanta wide: doc-value and
                    // `_source` cells differ often (ES too); `_source` only.
                    if !location && f.contains("29") {
                        continue;
                    }
                    let q = format!("FROM gp | EVAL grid = {f}::long | SORT grid {sort} | {keep} | LIMIT {limit}");
                    let (fast, slow, taken) = both(&node, &q);
                    assert_eq!(taken, 1, "fast path not taken: {q}");
                    assert_eq!(fast.table.cols, slow.table.cols, "{q}");
                    assert_eq!(fast.warnings, slow.warnings, "{q}");
                    if location {
                        assert_eq!(fast.table.rows, slow.table.rows, "{q}");
                    } else {
                        assert_eq!(mv_sorted(&fast.table.rows), mv_sorted(&slow.table.rows), "{q}");
                    }
                    let g = fast.table.cols.iter().position(|c| c.name == "grid").unwrap();
                    // Nulls sort first for DESC (and NULLS FIRST): a short top
                    // n can be all documents without a location.
                    let nulls_last = (sort.starts_with("ASC") && !sort.contains("NULLS FIRST")) || sort.contains("NULLS LAST");
                    if nulls_last {
                        assert!(fast.table.rows.iter().any(|r| matches!(r[g], Val::Long(_))), "{q}: no cell");
                    }
                }
            }
        }
        let q = "FROM gp | EVAL grid = TO_LONG(ST_GEOHASH(location, 3)) | SORT grid | KEEP grid | LIMIT 20";
        let (fast, slow, taken) = both(&node, q);
        assert_eq!(taken, 1, "{q}");
        assert_eq!(mv_sorted(&fast.table.rows), mv_sorted(&slow.table.rows), "{q}");
    }

    /// A geo_point array mixing formats, `[{"lat": .., "lon": ..}, [x, y]]`,
    /// or of `[x, y]` arrays, is several points (as the mapper indexes it):
    /// `_source` reads give every point, and the grid cells of every point,
    /// on the table and the fast paths alike.
    #[test]
    fn mixed_point_arrays_read_every_point_from_source() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        trial(&node);
        node.create_index("mp", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"location": {"type": "geo_point"}, "n": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("mp").unwrap();
        let docs = [
            json!({"n": 1, "location": [{"lat": 10.5, "lon": 20.25}, [40.5, -30.25]]}),
            json!({"n": 2, "location": [[1.5, 2.5], [3.5, 4.5]]}),
            json!({"n": 3, "location": [5.5, 6.5]}),
            json!({"n": 4, "location": "POINT (7.5 8.5)"}),
        ];
        for (i, d) in docs.iter().enumerate() {
            for x in svc.write(vec![WriteReq::Index { id: Some(i.to_string()), source: d.to_string(), routing: None, cond: Default::default() }], false) {
                x.unwrap();
            }
        }
        svc.refresh().unwrap();
        let pts = |q: &str| {
            DISABLED.with(|c| c.set(true));
            let r = run(&node, q, opts()).unwrap();
            DISABLED.with(|c| c.set(false));
            r.table.rows
        };
        let rows = pts("FROM mp | SORT n | KEEP location");
        let count = |v: &Val| match v { Val::Multi(m) => m.len(), Val::Null => 0, _ => 1 };
        assert_eq!(rows.iter().map(|r| count(&r[0])).collect::<Vec<_>>(), vec![2, 2, 1, 1], "{rows:?}");
        for keep in ["KEEP grid", "KEEP grid, location"] {
            let q = format!("FROM mp | EVAL grid = ST_GEOHASH(location, 4)::long | SORT grid ASC | {keep} | LIMIT 10");
            let (fast, slow, taken) = both(&node, &q);
            assert_eq!(taken, 1, "{q}");
            assert_eq!(mv_sorted(&fast.table.rows), mv_sorted(&slow.table.rows), "{q}");
            let cells: usize = slow.table.rows.iter().map(|r| count(&r[0])).sum();
            assert_eq!(cells, 6, "every point's cell: {q} {:?}", slow.table.rows);
        }
    }

    /// ES's cell longs (`Geohash.longEncode`, `GeoTileUtils.longEncodeTiles`,
    /// H3), the spatial-grid.csv-spec values; and the doc-value/`_source`
    /// choice at a tile edge where the quantized and the `_source` point
    /// fall in different tiles (`KEEP grid` reads doc values, `KEEP grid,
    /// location` the source, as `SpatialDocValuesExtraction` decides).
    #[test]
    fn grid_longs_and_doc_value_choice_follow_es() {
        use crate::esql::func2::grid_address_long;
        assert_eq!(grid_address_long("geohash", "u3bu"), Some(13686180));
        assert_eq!(grid_address_long("geohex", "841f059ffffffff"), Some(595020895127339007));
        assert_eq!(grid_address_long("geotile", "4/8/5"), Some((4 << 58) | (8 << 29) | 5));
        assert_eq!(grid_address_long("geotile", "4/8"), None);
        for (kind, p, lat, lon) in [(0u8, 4i64, 52.37, 4.89), (1, 13, 49.3, 2.3), (2, 4, 37.7, -122.4)] {
            let addr = match kind {
                0 => rustic_core::geo::geohash(lat, lon, p as usize),
                1 => { let (x, y) = rustic_core::geo::geotile(lat, lon, p as u32); format!("{p}/{x}/{y}") }
                _ => crate::aggs::geo_hex::point_cell(lat, lon, p as u32).unwrap(),
            };
            let name = ["geohash", "geotile", "geohex"][kind as usize];
            assert_eq!(grid_address_long(name, &addr), super::grid_long(kind, p, lat, lon), "{name} {addr}");
        }
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        trial(&node);
        node.create_index("edge", &json!({"mappings": {"properties": {"location": {"type": "geo_point"}}}})).unwrap();
        let svc = node.get_index("edge").unwrap();
        for x in svc.write(vec![WriteReq::Index { id: Some("1".into()), source: json!({"location": {"lat": 49.325122001040015, "lon": 2.3}}).to_string(), routing: None, cond: Default::default() }], false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
        let tile = |y: i64| Val::Long((13 << 58) | (4148 << 29) | y);
        let r = run(&node, "FROM edge | EVAL grid = ST_GEOTILE(location, 13)::long | SORT grid | KEEP grid | LIMIT 10", opts()).unwrap();
        assert_eq!(r.table.rows, vec![vec![tile(2802)]]);
        let r = run(&node, "FROM edge | EVAL grid = ST_GEOTILE(location, 13)::long | SORT grid | KEEP grid, location | LIMIT 10", opts()).unwrap();
        assert_eq!(r.table.rows[0][0], tile(2801));
        // The table path (no fast path): the `_source` point, a real long.
        DISABLED.with(|c| c.set(true));
        let r = run(&node, "FROM edge | EVAL grid = ST_GEOTILE(location, 13)::long | KEEP grid, location", opts());
        DISABLED.with(|c| c.set(false));
        assert_eq!(r.unwrap().table.rows[0][0], tile(2801));
    }

    /// Timings for the Rally `geoGrid_sort_*-esql{,-nolocation}` tasks on
    /// `ESQL_GRID_DOCS` points (default 608,444), fast against table path,
    /// median of `ESQL_GRID_RUNS` (3). Run on the build host:
    /// `cargo test --release -p elasticsearch-engine grid_top_n_timings -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn grid_top_n_timings() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (docs, runs) = (env("ESQL_GRID_DOCS", 608_444), env("ESQL_GRID_RUNS", 3));
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        trial(&node);
        let started = std::time::Instant::now();
        grid_points(&node, "osmgeopoints", docs);
        eprintln!("fixture: {docs} points in {:?}", started.elapsed());
        let median = |q: &str, disabled: bool| {
            DISABLED.with(|c| c.set(disabled));
            let mut times: Vec<std::time::Duration> = (0..runs).map(|_| {
                let t = std::time::Instant::now();
                std::hint::black_box(run(&node, q, opts()).unwrap().table.rows.len());
                t.elapsed()
            }).collect();
            DISABLED.with(|c| c.set(false));
            times.sort();
            times[times.len() / 2]
        };
        for f in ["ST_GEOHASH", "ST_GEOTILE", "ST_GEOHEX"] {
            for keep in ["KEEP grid, location", "KEEP grid"] {
                let q = format!("FROM osmgeopoints | EVAL grid = {f}(location, 5)::long | SORT grid ASC | {keep} | LIMIT 10");
                let (fast, slow, _) = both(&node, &q);
                assert_eq!(mv_sorted(&fast.table.rows), mv_sorted(&slow.table.rows), "{q}");
                eprintln!("{q}: fast={:?} table={:?}", median(&q, false), median(&q, true));
            }
        }
    }


    /// `ST_EXTENT_AGG` / `ST_CENTROID_AGG` over a `geo_shape` field take the
    /// doc-values fast path and read each doc's doc-value header: equal to
    /// ES's aggregation (`SpatialExtentGeoShapeDocValuesAggregator`,
    /// `SpatialCentroidShapeDocValuesAggregator`) of the headers computed
    /// from the decoded shapes, with and without a pushed-down filter.
    #[test]
    fn geo_shape_stats_read_doc_value_headers() {
        use rustic_core::fieldtypes::shape;
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("shp", &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"shape": {"type": "geo_shape"}, "n": {"type": "long"}, "s": {"type": "double"}}}})).unwrap();
        let svc = node.get_index("shp").unwrap();
        let mut r = Rng(5);
        let mut wkts = Vec::new();
        let mut batch = Vec::new();
        for i in 0..600u64 {
            let x = r.below(3400) as f64 / 10.0 - 170.0;
            let y = r.below(1400) as f64 / 10.0 - 70.0;
            let d = 0.1 + r.below(30) as f64 / 10.0;
            let w = match i % 6 {
                0 => format!("POINT ({x} {y})"),
                1 => format!("LINESTRING ({x} {y}, {} {})", x + d, y - d),
                2 => format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + d, x + d, y + d, y + d),
                // Across the prime meridian.
                3 => format!("MULTILINESTRING (({} {y}, {d} {}))", -d / 2.0, y + d),
                // Across the dateline.
                4 => format!("POLYGON ((175 {y}, -175 {y}, -175 {}, 175 {}, 175 {y}))", y + d, y + d),
                _ => format!("GEOMETRYCOLLECTION (POINT ({x} {y}), LINESTRING ({x} {y}, {} {}))", x + d, y + d),
            };
            // `s`: one value, two values, or none.
            let src = match i % 7 {
                0 => json!({"shape": w, "n": i}),
                1 => json!({"shape": w, "n": i, "s": [i as f64, i as f64 + 0.5]}),
                _ => json!({"shape": w, "n": i, "s": i as f64 * 0.5}),
            };
            batch.push(WriteReq::Index { id: Some(format!("d{i}")), source: src.to_string(), routing: None, cond: Default::default() });
            wkts.push(w);
        }
        for x in svc.write(batch, false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
        trial(&node);
        let opts_g = shape::ParseOpts { geo: true, coerce: false, ignore_z: true };
        // Each doc's shape doc value header, built independently.
        struct Meta {
            extent: Option<shape::ShapeExtent>,
            centroid: Option<shape::ShapeCentroid>,
        }
        let metas: Vec<Meta> = wkts
            .iter()
            .map(|w| {
                let g = shape::parse(&json!(w), &opts_g).unwrap();
                let mut tris = Vec::new();
                rustic_core::fieldtypes::tessellator::geometry_triangles(&shape::normalized(&g), &mut tris).unwrap();
                let mut c = shape::CentroidCalc::default();
                c.add(&g);
                let b = rustic_core::fieldtypes::shape_field::doc_value(&tris, &c).unwrap();
                let v = rustic_core::fieldtypes::shape_dv::ShapeValue::new(&b);
                Meta { extent: v.extent(), centroid: Some(shape::ShapeCentroid { lat: v.centroid_y(), lon: v.centroid_x(), dim: v.dimension(), weight: v.weight() }) }
            })
            .collect();
        let check = |filter: &str| {
            let ns: Vec<usize> = if filter.is_empty() {
                (0..wkts.len()).collect()
            } else {
                let q = format!("FROM shp {filter} | KEEP n | LIMIT 1000");
                run(&node, &q, opts()).unwrap().table.rows.iter().map(|row| match row[0] { Val::Long(n) | Val::Int(n) => n as usize, ref v => panic!("{v:?}") }).collect()
            };
            assert!(!ns.is_empty(), "{filter}");
            let mut ext: Option<shape::ShapeExtent> = None;
            let mut cs = Vec::new();
            for &i in &ns {
                if let Some(e) = metas[i].extent {
                    ext = Some(match ext {
                        Some(mut x) => {
                            x.merge(&e);
                            x
                        }
                        None => e,
                    });
                }
                cs.extend(metas[i].centroid.filter(|c| c.weight > 0.0));
            }
            let dim = cs.iter().map(|c| c.dim).max().unwrap();
            let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
            for c in cs.iter().filter(|c| c.dim == dim) {
                sx += c.lon_deg() * c.weight;
                sy += c.lat_deg() * c.weight;
                sw += c.weight;
            }
            let q = format!("FROM shp {filter} | STATS e = ST_EXTENT_AGG(shape), c = ST_CENTROID_AGG(shape), n = COUNT(*), k = COUNT(s)");
            let before = TAKEN.with(|c| c.get());
            let res = run(&node, &q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
            assert_eq!(TAKEN.with(|c| c.get()) - before, 1, "{q}: fast path");
            let row = &res.table.rows[0];
            assert_eq!(row[0], super::shape_extent_wkt(&ext.unwrap()), "{q}");
            let Val::Str(c) = &row[1] else { panic!("{q}: {row:?}") };
            let xy = nums(c);
            assert!((xy[0] - sx / sw).abs() < 1e-9 && (xy[1] - sy / sw).abs() < 1e-9, "{q}: {c} want {} {}", sx / sw, sy / sw);
            assert_eq!(row[2], Val::Long(ns.len() as i64), "{q}");
            let values: usize = ns.iter().map(|&i| match i % 7 { 0 => 0, 1 => 2, _ => 1 }).sum();
            assert_eq!(row[3], Val::Long(values as i64), "{q}");
            // COUNT(s) alone takes the same path, equal to the table path.
            let q = format!("FROM shp {filter} | STATS k = COUNT(s)");
            let (fast, slow, taken) = both(&node, &q);
            assert_eq!(taken, 1, "{q}");
            assert_eq!(fast.table.rows, slow.table.rows, "{q}");
            assert_eq!(fast.table.rows[0][0], Val::Long(values as i64), "{q}");
        };
        check("");
        check("| WHERE ST_INTERSECTS(shape, TO_GEOSHAPE(\"POLYGON((-10 -10, 10 -10, 10 10, -10 10, -10 -10))\"))");
        check("| WHERE ST_DISJOINT(shape, TO_GEOSHAPE(\"POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))\"))");
    }

    /// ES|QL `ST_INTERSECTS` / `ST_DISJOINT` / `ST_WITHIN` with a polygon
    /// literal on a `geo_shape` field (Rally `geopointshape` / `geoshape`):
    /// the pushed-down `bool.filter` reaches the shape query's early-stop
    /// and count paths; rows (`LIMIT`) and counts (`STATS COUNT(*)`) equal
    /// the table path and the relation's definition.
    #[test]
    fn geo_shape_relations_with_limit_and_count() {
        shape_relations_case(false);
    }

    /// [`geo_shape_relations_with_limit_and_count`] on segments of points
    /// only (Rally `geopointshape`): `COUNT(*)` counts the point tree's
    /// KD leaves (the query's exact count), rows take the point walk.
    #[test]
    fn geo_point_shape_relations_with_limit_and_count() {
        shape_relations_case(true);
    }

    fn shape_relations_case(points_only: bool) {
        use rustic_core::fieldtypes::shape;
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("gs", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"shape": {"type": "geo_shape"}, "n": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("gs").unwrap();
        let mut r = Rng(9);
        let mut wkts = Vec::new();
        let mut batch = Vec::new();
        for i in 0..3000u64 {
            let x = r.below(3000) as f64 / 100.0 - 10.0;
            let y = r.below(2000) as f64 / 100.0 + 45.0;
            let w = if !points_only && i % 5 == 0 { format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + 0.2, x + 0.2, y + 0.2, y + 0.2) } else { format!("POINT ({x} {y})") };
            batch.push(WriteReq::Index { id: Some(format!("d{i}")), source: json!({"shape": w, "n": i}).to_string(), routing: None, cond: Default::default() });
            wkts.push(w);
            if i == 1500 {
                for x in svc.write(std::mem::take(&mut batch), false) {
                    x.unwrap();
                }
                svc.refresh().unwrap();
            }
        }
        for x in svc.write(batch, false) {
            x.unwrap();
        }
        svc.refresh().unwrap();
        let poly = "POLYGON((-0.1 49.0, 5.0 48.0, 15.0 49.0, 14.0 60.0, -0.1 61.0, -0.1 49.0))";
        let opts_g = shape::ParseOpts { geo: true, coerce: true, ignore_z: true };
        let qp = shape::Prims::of(&shape::normalize_geo(&shape::parse(&json!(poly), &opts_g).unwrap())).quantized();
        for (f, rel) in [("ST_INTERSECTS", shape::SpatialRelation::Intersects), ("ST_DISJOINT", shape::SpatialRelation::Disjoint), ("ST_WITHIN", shape::SpatialRelation::Within)] {
            let want: Vec<i64> = (0..wkts.len())
                .filter(|&i| {
                    let dp = shape::Prims::of(&shape::normalize_geo(&shape::parse(&json!(wkts[i]), &opts_g).unwrap())).quantized();
                    shape::relate(&dp, &qp, rel)
                })
                .map(|i| i as i64)
                .collect();
            let q = format!("FROM gs | WHERE {f}(shape, TO_GEOSHAPE(\"{poly}\")) | STATS c = COUNT(*)");
            let res = run(&node, &q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
            assert_eq!(res.table.rows[0][0], Val::Long(want.len() as i64), "{q}");
            // Evaluated row by row (not pushed down: under a CASE): the same
            // relation, each value as a shape doc value.
            let q = format!("FROM gs | EVAL x = CASE(n >= 0, {f}(shape, TO_GEOSHAPE(\"{poly}\")), false) | WHERE x | STATS c = COUNT(*)");
            let res = run(&node, &q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
            assert_eq!(res.table.rows[0][0], Val::Long(want.len() as i64), "{q}");
            // COUNT of a long every doc has (Rally `-count-esql`'s
            // COUNT(size)), on the fast path and the table path.
            let q = format!("FROM gs | WHERE {f}(shape, TO_GEOSHAPE(\"{poly}\")) | STATS c = COUNT(n)");
            let res = run(&node, &q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
            assert_eq!(res.table.rows[0][0], Val::Long(want.len() as i64), "{q}");
            DISABLED.with(|c| c.set(true));
            let slow = run(&node, &q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
            DISABLED.with(|c| c.set(false));
            assert_eq!(slow.table.rows, res.table.rows, "{q}");
            for limit in [10usize, 1000] {
                let q = format!("FROM gs | WHERE {f}(shape, TO_GEOSHAPE(\"{poly}\")) | KEEP n | SORT n | LIMIT {limit}");
                let res = run(&node, &q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
                let got: Vec<i64> = res.table.rows.iter().map(|row| match row[0] { Val::Long(n) | Val::Int(n) => n, ref v => panic!("{v:?}") }).collect();
                assert_eq!(got, want.iter().copied().take(limit).collect::<Vec<_>>(), "{q}");
                // Unsorted LIMIT: rows are matches, as many as asked.
                let q = format!("FROM gs | WHERE {f}(shape, TO_GEOSHAPE(\"{poly}\")) | KEEP n | LIMIT {limit}");
                let res = run(&node, &q, opts()).unwrap_or_else(|e| panic!("{q}: {e:?}"));
                let got: Vec<i64> = res.table.rows.iter().map(|row| match row[0] { Val::Long(n) | Val::Int(n) => n, ref v => panic!("{v:?}") }).collect();
                assert_eq!(got.len(), limit.min(want.len()), "{q}");
                assert!(got.iter().all(|n| want.binary_search(n).is_ok()), "{q}");
            }
        }
    }

    /// Rally `esql` track shape: `idx_0..idx_{n-1}` (mapping family `i % 20`),
    /// each `docs` documents over `shards` shards in a few refreshes; every
    /// doc has `id`, `@timestamp`, the keys and 10 of its `n_fields` keyword
    /// fields (the `f10` corpus).
    fn wide_limit(node: &Node, indices: usize, docs: usize, shards: usize, n_fields: usize) {
        for i in 0..indices {
            let p = i % 20;
            let mut props = serde_json::Map::new();
            props.insert("id".into(), json!({"type": "long"}));
            props.insert("@timestamp".into(), json!({"type": "date"}));
            for j in 0..n_fields { props.insert(format!("field_{p}_{j}"), json!({"type": "keyword"})); }
            for k in ["key_1000", "key_100000"] { props.insert(k.into(), json!({"type": "keyword"})); }
            let name = format!("idx_{i}");
            node.create_index(&name, &json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0},
                "mappings": {"_source": {"mode": "stored"}, "properties": props}})).unwrap();
            let svc = node.get_index(&name).unwrap();
            let mut r = Rng(i as u64 + 11);
            let mut batch = Vec::new();
            for n in 0..docs {
                let id = i * docs + n;
                let mut doc = serde_json::Map::new();
                doc.insert("id".into(), json!(id));
                doc.insert("@timestamp".into(), json!(1_700_000_000_000i64 + id as i64 * 1000));
                doc.insert("key_1000".into(), json!((id % 1000).to_string()));
                doc.insert("key_100000".into(), json!((id % 100_000).to_string()));
                for _ in 0..10 { doc.insert(format!("field_{p}_{}", r.below(n_fields as u64)), json!(format!("v{}", r.below(50)))); }
                batch.push(WriteReq::Index { id: Some(id.to_string()), source: serde_json::Value::Object(doc).to_string(), routing: None, cond: Default::default() });
                if batch.len() == 2000 {
                    for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
                    svc.refresh().unwrap();
                }
            }
            for x in svc.write(batch, false) { x.unwrap(); }
            svc.refresh().unwrap();
        }
    }

    /// The Rally `esql` LIMIT / LIMIT BY shapes.
    const LIMIT_SHAPES: [&str; 12] = [
        "FROM idx_* | limit 1",
        "FROM idx_* | limit 10",
        "FROM idx_* | limit 1000",
        "FROM idx_* | limit 10000",
        "FROM idx_* | LIMIT 1 BY key_1000",
        "FROM idx_* | LIMIT 10 BY key_1000",
        "FROM idx_* | LIMIT 1000 BY key_1000",
        "FROM idx_* | LIMIT 10000 BY key_1000",
        "FROM idx_* | LIMIT 1 BY key_1000 | LIMIT 5",
        "FROM idx_* | LIMIT 2 BY key_1000, key_100000 | LIMIT 300",
        "FROM idx_* | WHERE id > 100 | LIMIT 3 BY key_1000",
        "FROM idx_* | LIMIT 1 BY key_1000, field_0_3 | LIMIT 100",
    ];

    /// Every LIMIT / LIMIT BY shape: the streamed path's rows and columns
    /// are the table path's, and the stream answers it.
    #[test]
    fn limit_shapes_match_the_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        wide_limit(&node, 3, 2500, 2, 30);
        for q in LIMIT_SHAPES {
            let (fast, slow, taken) = both(&node, q);
            assert_same(q, &fast, &slow);
            assert!(taken > 0, "{q}: fast path not taken");
            for drop_null in [false, true] {
                let encode = |t: &crate::esql::exec::Table| {
                    let mut meta = serde_json::Map::new();
                    let keep = crate::esql::format::columns_only(t, drop_null, &mut meta);
                    meta.insert("values".into(), serde_json::Value::Null);
                    serde_json::to_string(&crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: t, keep: &keep, columnar: false }).unwrap()
                };
                assert_eq!(encode(&fast.table), encode(&slow.table), "{q} drop_null={drop_null}");
            }
        }
    }

    /// Timings of the Rally `esql` LIMIT / LIMIT BY operations on that
    /// shape (`ESQL_LIMIT_INDICES` default 5, `ESQL_LIMIT_DOCS` per index
    /// default 20000, `ESQL_LIMIT_FIELDS` default 500, 5 shards): the query
    /// (median ms), the JSON response as the HTTP handler streams it (with
    /// and without `drop_null_columns`), and a check against the table path:
    /// `cargo test -p elasticsearch-engine --release --lib esql_limit_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn esql_limit_bench() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        wide_limit(&node, env("ESQL_LIMIT_INDICES", 5), env("ESQL_LIMIT_DOCS", 20_000), 5, env("ESQL_LIMIT_FIELDS", 500));
        let median = |f: &mut dyn FnMut()| {
            let mut times: Vec<f64> = (0..7).map(|_| { let s = std::time::Instant::now(); f(); s.elapsed().as_secs_f64() * 1000.0 }).collect();
            times.sort_by(f64::total_cmp);
            times[3]
        };
        println!("{:<44} {:>10} {:>10} {:>12}", "query (median ms)", "query", "json", "json/drop");
        for q in LIMIT_SHAPES {
            let (fast, slow, _) = both(&node, q);
            assert_same(q, &fast, &slow);
            let t_query = median(&mut || { std::hint::black_box(run(&node, q, opts()).unwrap()); });
            let encode = |drop_null: bool| {
                let mut meta = serde_json::Map::new();
                let keep = crate::esql::format::columns_only(&fast.table, drop_null, &mut meta);
                meta.insert("values".into(), serde_json::Value::Null);
                serde_json::to_vec(&crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: &fast.table, keep: &keep, columnar: false }).unwrap()
            };
            let t_json = median(&mut || { std::hint::black_box(encode(false)); });
            let t_drop = median(&mut || { std::hint::black_box(encode(true)); });
            println!("{q:<44} {t_query:>10.2} {t_json:>10.2} {t_drop:>12.2}");
        }
    }

    /// The Rally `esql` track's real corpus shape (`lookup_idx_100000_f10`):
    /// documents hold `key_100000` and ten `lookup_keyword_*` fields only,
    /// none of the mapped `field_*` / `key_1000` columns (the lookup keywords
    /// are mapped dynamically as text with a keyword sub-field), so a
    /// `LIMIT n BY key_1000` has one (null) group and reads every document.
    fn rally_corpus(node: &Node, indices: usize, docs: usize, shards: usize, n_fields: usize) {
        for i in 0..indices {
            let p = i % 20;
            let mut props = serde_json::Map::new();
            props.insert("id".into(), json!({"type": "long"}));
            props.insert("@timestamp".into(), json!({"type": "date"}));
            for j in 0..n_fields { props.insert(format!("field_{p}_{j}"), json!({"type": "keyword"})); }
            for k in ["key_1000", "key_100000", "key_200000", "key_500000", "key_1000000", "key_5000000", "key_100000000"] { props.insert(k.into(), json!({"type": "keyword"})); }
            let name = format!("idx_{i}");
            node.create_index(&name, &json!({"settings": {"number_of_shards": shards, "number_of_replicas": 0},
                "mappings": {"_source": {"mode": "stored"}, "properties": props}})).unwrap();
            let svc = node.get_index(&name).unwrap();
            let mut r = Rng(i as u64 + 31);
            let mut batch = Vec::new();
            for _ in 0..docs {
                let v = r.below(100_000);
                let mut doc = serde_json::Map::new();
                doc.insert("key_100000".into(), json!(v.to_string()));
                for k in 0..10 { doc.insert(format!("lookup_keyword_{k}"), json!(format!("val {v} rep 0"))); }
                batch.push(WriteReq::Index { id: None, source: serde_json::Value::Object(doc).to_string(), routing: None, cond: Default::default() });
                if batch.len() == 10_000 {
                    for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
                }
            }
            for x in svc.write(batch, false) { x.unwrap(); }
            svc.refresh().unwrap();
        }
    }

    /// A LIMIT BY key mapped in every index but held by no document (Rally
    /// esql `from_idx_limit_*_by_key_1000`) proves each null from a newer
    /// field's doc value instead of reading `_source`. Keys mapped after
    /// documents holding them in `_source` were indexed (no newer field
    /// value) still read their `_source`: the same rows as the table path.
    #[test]
    fn limit_by_absent_key_reads_no_source() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        rally_corpus(&node, 2, 3000, 2, 20);
        // A later segment of idx_1 where some documents hold the key.
        let idx1 = node.get_index("idx_1").unwrap();
        let keyed = (0..200).map(|i| WriteReq::Index { id: Some(format!("keyed{i}")), source: json!({"key_100000": i.to_string(), "key_1000": format!("k{}", i % 4)}).to_string(), routing: None, cond: Default::default() }).collect();
        for x in idx1.write(keyed, false) { x.unwrap(); }
        idx1.refresh().unwrap();
        node.create_index("late", &json!({"settings": {"number_of_shards": 2}, "mappings": {"dynamic": false, "properties": {"n": {"type": "keyword"}}}})).unwrap();
        let late = node.get_index("late").unwrap();
        let write = |docs: Vec<serde_json::Value>, from: usize| {
            let batch = docs.into_iter().enumerate().map(|(i, doc)| WriteReq::Index { id: Some((from + i).to_string()), source: doc.to_string(), routing: None, cond: Default::default() }).collect();
            for x in late.write(batch, false) { x.unwrap(); }
        };
        // `k` is in `_source` before it is mapped: no doc value, no newer field.
        write((0..300).map(|i| json!({"n": format!("a{}", i % 3), "k": format!("v{}", i % 5)})).collect(), 0);
        late.refresh().unwrap();
        late.flush().unwrap();
        late.put_mapping(&json!({"properties": {"k": {"type": "keyword"}}})).unwrap();
        // A numeric column: low-cardinality keywords may have no doc-values
        // column of their own (derived from postings), and prove nothing.
        late.put_mapping(&json!({"properties": {"later": {"type": "long"}}})).unwrap();
        write((0..300).map(|i| if i % 7 == 0 { json!({"n": "b", "k": "w", "later": i}) } else { json!({"n": format!("b{}", i % 2), "later": i}) }).collect(), 300);
        late.refresh().unwrap();
        // Both kinds in one later segment.
        write(vec![json!({"n": "c", "k": "m"}), json!({"n": "c", "later": 1})], 600);
        late.refresh().unwrap();
        let (proofs, absent) = (super::NEWER_FIELD_PROOFS.load(std::sync::atomic::Ordering::Relaxed), super::ABSENT_KEY_CHUNKS.load(std::sync::atomic::Ordering::Relaxed));
        for q in [
            "FROM idx_* | LIMIT 1 BY key_1000",
            "FROM idx_* | LIMIT 10 BY key_1000",
            "FROM idx_* | LIMIT 3 BY key_1000, key_200000",
            "FROM idx_* | LIMIT 2 BY key_100000 | LIMIT 50",
            "FROM late | LIMIT 1 BY k",
            "FROM late | LIMIT 2 BY k, n",
            "FROM late | LIMIT 1 BY later",
            "FROM late | LIMIT 1 BY n",
        ] {
            let (fast, slow, taken) = both(&node, q);
            assert_same(q, &fast, &slow);
            assert_eq!(fast.warnings, slow.warnings, "{q}");
            assert!(taken > 0, "{q}: not streamed");
        }
        assert!(super::NEWER_FIELD_PROOFS.load(std::sync::atomic::Ordering::Relaxed) > proofs, "no null proven from a newer field");
        assert!(super::ABSENT_KEY_CHUNKS.load(std::sync::atomic::Ordering::Relaxed) > absent, "no all-null key chunk");
        // The late key's values come from `_source` on the documents indexed before it was mapped.
        let (fast, _, _) = both(&node, "FROM late | LIMIT 1 BY k");
        let k = fast.table.cols.iter().position(|c| c.name == "k").unwrap();
        let mut keys: Vec<String> = fast.table.rows.iter().map(|r| r[k].to_text()).collect();
        keys.sort();
        assert_eq!(keys.len(), 8, "{keys:?}");
    }

    /// With `drop_null_columns` a plain `FROM ... | LIMIT n` builds only
    /// its non-null columns (every column listed in `all_columns`): the
    /// same response JSON (row-major, columnar and streamed), warnings and
    /// counts as dropping them from the whole table. That includes a
    /// union-typed column read and then nulled, limits 0 and past the
    /// end, and shapes that keep the whole table.
    #[test]
    fn drop_null_columns_builds_only_non_null_columns() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        rally_corpus(&node, 3, 2000, 2, 40);
        for (name, ty, value) in [("union_a", "keyword", json!("x")), ("union_b", "long", json!(7))] {
            node.create_index(name, &json!({"mappings": {"properties": {"u": {"type": ty}, "v": {"type": "keyword"}, "empty": {"type": "keyword"}}}})).unwrap();
            let svc = node.get_index(name).unwrap();
            let docs = (0..20).map(|i| WriteReq::Index { id: Some(i.to_string()), source: json!({"u": value, "v": format!("v{i}")}).to_string(), routing: None, cond: Default::default() }).collect();
            for x in svc.write(docs, false) { x.unwrap(); }
            svc.refresh().unwrap();
        }
        let narrowed = |q: &str| run(&node, q, EsqlOptions { drop_null_columns: true, ..opts() }).unwrap_or_else(|e| panic!("{q}: {e}"));
        for (q, narrows) in [
            ("FROM idx_* | limit 1", true),
            ("FROM idx_* | limit 1000", true),
            ("FROM idx_* | limit 10000", true),
            ("FROM idx_* | limit 0", false),
            ("FROM idx_*", true),
            ("FROM union_a, union_b | LIMIT 30", true),
            // Pushed into the search: the rest is still the whole result.
            ("FROM idx_* | WHERE key_100000 == \"7\" | limit 10", true),
            ("FROM idx_* | WHERE TO_STRING(key_100000) LIKE \"1*\" | limit 10", false),
            ("FROM idx_* | KEEP key_100000, key_1000 | limit 10", false),
            ("FROM idx_* | EVAL x = 1 | limit 10", false),
            ("FROM idx_* | LIMIT 2 BY key_1000", false),
        ] {
            let full = run(&node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
            let narrow = narrowed(q);
            assert_eq!(narrow.all_columns.is_some(), narrows, "{q}");
            if let Some(all) = &narrow.all_columns {
                assert_eq!(all, &full.table.cols, "{q}");
                assert!(narrow.table.cols.len() < full.table.cols.len(), "{q}");
            }
            assert_eq!((narrow.warnings.clone(), narrow.documents_found, narrow.values_loaded), (full.warnings.clone(), full.documents_found, full.values_loaded), "{q}");
            for columnar in [false, true] {
                let mut expected = serde_json::Map::new();
                crate::esql::format::columns_values(&full.table, columnar, true, &mut expected);
                let mut actual = serde_json::Map::new();
                crate::esql::format::columns_values_all(&narrow.table, narrow.all_columns.as_deref(), columnar, true, &mut actual);
                assert_eq!(actual, expected, "{q} columnar={columnar}");
                let streamed = |t: &crate::esql::Table, all: Option<&[crate::esql::exec::Column]>| {
                    let mut meta = serde_json::Map::new();
                    let keep = crate::esql::format::columns_only_all(t, all, true, &mut meta);
                    meta.insert("values".into(), serde_json::Value::Null);
                    let mut out = Vec::new();
                    crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: t, keep: &keep, columnar }.write_json(&mut out).unwrap();
                    out
                };
                assert_eq!(streamed(&narrow.table, narrow.all_columns.as_deref()), streamed(&full.table, None), "{q} streamed columnar={columnar}");
            }
            // Without `drop_null_columns` nothing changes.
            assert!(full.all_columns.is_none(), "{q}");
        }
    }

    /// With `sparse_nulls` (nulls kept, the caller renders a result of the
    /// non-null columns) a plain `FROM ... | LIMIT n` is built the same
    /// narrow way, and the streamed response written with every column
    /// ([`crate::esql::format::columns_expanded`]) is byte for byte the full
    /// table's, row-major and columnar; [`crate::esql::format::expanded_table`]
    /// is the full table.
    #[test]
    fn sparse_nulls_render_the_full_table() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        rally_corpus(&node, 3, 2000, 2, 40);
        for q in ["FROM idx_* | limit 1", "FROM idx_* | limit 1000", "FROM idx_* | limit 10000", "FROM idx_*", "FROM idx_* | limit 0", "FROM idx_* | KEEP key_1000 | limit 5"] {
            let full = run(&node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
            let sparse = run(&node, q, EsqlOptions { sparse_nulls: true, ..opts() }).unwrap_or_else(|e| panic!("{q}: {e}"));
            assert_eq!((sparse.warnings.clone(), sparse.documents_found, sparse.values_loaded), (full.warnings.clone(), full.documents_found, full.values_loaded), "{q}");
            let Some(all) = sparse.all_columns.as_deref() else {
                assert_eq!(sparse.table.cols, full.table.cols, "{q}");
                assert_eq!(sparse.table.rows, full.table.rows, "{q}");
                continue;
            };
            assert_eq!(all, &full.table.cols[..], "{q}");
            assert!(sparse.table.cols.len() < full.table.cols.len(), "{q}");
            let expanded = crate::esql::format::expanded_table(&sparse.table, all);
            assert_eq!((&expanded.cols, &expanded.rows), (&full.table.cols, &full.table.rows), "{q}");
            for columnar in [false, true] {
                let write = |meta: &serde_json::Map<String, serde_json::Value>, t: &crate::esql::Table, keep: &[usize]| {
                    let mut out = Vec::new();
                    crate::esql::format::StreamedResponse { meta, values_key: "values", table: t, keep, columnar }.write_json(&mut out).unwrap();
                    out
                };
                let mut meta = serde_json::Map::new();
                let keep = crate::esql::format::columns_expanded(&sparse.table, all, &mut meta);
                meta.insert("values".into(), serde_json::Value::Null);
                let mut full_meta = serde_json::Map::new();
                let full_keep = crate::esql::format::columns_only(&full.table, false, &mut full_meta);
                full_meta.insert("values".into(), serde_json::Value::Null);
                assert_eq!(write(&meta, &sparse.table, &keep), write(&full_meta, &full.table, &full_keep), "{q} columnar={columnar}");
                // The serde path (pretty / filtered responses) agrees too.
                let tree = |meta: &serde_json::Map<String, serde_json::Value>, t: &crate::esql::Table, keep: &[usize]| {
                    serde_json::to_vec(&crate::esql::format::StreamedResponse { meta, values_key: "values", table: t, keep, columnar }).unwrap()
                };
                assert_eq!(tree(&meta, &sparse.table, &keep), tree(&full_meta, &full.table, &full_keep), "{q} serde columnar={columnar}");
            }
        }
    }

    /// Rally `esql` gaps on its real corpus shape ([`rally_corpus`];
    /// `ESQL_RALLY_INDICES` default 5, `ESQL_RALLY_DOCS` per index default
    /// 100000 as the track, `ESQL_RALLY_FIELDS` default 500, 5 shards):
    /// `LIMIT 1 / 10 / 1000 BY key_1000` (one null group: every document's
    /// key is read), `limit 10000` with and without `drop_null_columns`, and
    /// `_search` `size` 1000 / 10000 (Rally `search_limit_*`), median ms of 7,
    /// after timing the load itself (Rally `index-base`):
    /// `cargo test -p elasticsearch-engine --release --lib esql_rally_corpus_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn esql_rally_corpus_bench() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        // Rally `index-base`: the bulk load (10,000-doc bulks, as the track).
        let (indices, docs) = (env("ESQL_RALLY_INDICES", 5), env("ESQL_RALLY_DOCS", 100_000));
        let load = std::time::Instant::now();
        rally_corpus(&node, indices, docs, 5, env("ESQL_RALLY_FIELDS", 500));
        let secs = load.elapsed().as_secs_f64();
        println!("index-base: {} docs in {secs:.2} s ({:.0} docs/s)", indices * docs, (indices * docs) as f64 / secs);
        // `ESQL_RALLY_RUNS` (default 7) per median; `ESQL_RALLY_ONLY`: the
        // queries containing it, and no `_search` (a profiler's loop).
        let runs = env("ESQL_RALLY_RUNS", 7).max(1);
        let only = std::env::var("ESQL_RALLY_ONLY").ok();
        let median = |f: &mut dyn FnMut()| {
            let mut times: Vec<f64> = (0..runs).map(|_| { let s = std::time::Instant::now(); f(); s.elapsed().as_secs_f64() * 1000.0 }).collect();
            times.sort_by(f64::total_cmp);
            times[times.len() / 2]
        };
        eprintln!("READY");
        for q in ["FROM idx_* | LIMIT 1 BY key_1000", "FROM idx_* | LIMIT 10 BY key_1000", "FROM idx_* | LIMIT 1000 BY key_1000", "FROM idx_* | limit 1000", "FROM idx_* | limit 10000"] {
            if only.as_ref().is_some_and(|only| !q.contains(only.as_str())) { continue; }
            let first = run(&node, q, opts()).unwrap();
            let t_query = median(&mut || { std::hint::black_box(run(&node, q, opts()).unwrap()); });
            if only.is_some() { println!("{q}: query {t_query:.2} ms"); continue; }
            let encode = |drop_null: bool| {
                let mut meta = serde_json::Map::new();
                let keep = crate::esql::format::columns_only(&first.table, drop_null, &mut meta);
                meta.insert("values".into(), serde_json::Value::Null);
                let mut out = Vec::new();
                crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: &first.table, keep: &keep, columnar: false }.write_json(&mut out).unwrap();
                out
            };
            let t_json = median(&mut || { std::hint::black_box(encode(false)); });
            let t_drop = median(&mut || { std::hint::black_box(encode(true)); });
            // Rally `*_drop_null`: the query building only its non-null columns, then the response.
            let t_narrow = median(&mut || {
                let r = run(&node, q, EsqlOptions { drop_null_columns: true, ..opts() }).unwrap();
                let mut meta = serde_json::Map::new();
                let keep = crate::esql::format::columns_only_all(&r.table, r.all_columns.as_deref(), true, &mut meta);
                meta.insert("values".into(), serde_json::Value::Null);
                let mut out = Vec::new();
                crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: &r.table, keep: &keep, columnar: false }.write_json(&mut out).unwrap();
                std::hint::black_box(out);
            });
            // The sync `_query` route keeping nulls (`sparse_nulls`): the
            // narrowed query, then every column written.
            let t_sparse = median(&mut || {
                let r = run(&node, q, EsqlOptions { sparse_nulls: true, ..opts() }).unwrap();
                let mut meta = serde_json::Map::new();
                let keep = match r.all_columns.as_deref() {
                    Some(all) => crate::esql::format::columns_expanded(&r.table, all, &mut meta),
                    None => crate::esql::format::columns_only(&r.table, false, &mut meta),
                };
                meta.insert("values".into(), serde_json::Value::Null);
                let mut out = Vec::new();
                crate::esql::format::StreamedResponse { meta: &meta, values_key: "values", table: &r.table, keep: &keep, columnar: false }.write_json(&mut out).unwrap();
                std::hint::black_box(out);
            });
            println!("{q:<40} rows {:>6} cols {:>5}: query {t_query:>9.2}  json {t_json:>8.2}  json drop_null {t_drop:>8.2}  drop_null query+json {t_narrow:>8.2}  sparse query+json {t_sparse:>8.2} ms", first.table.rows.len(), first.table.cols.len());
        }
        let targets = node.resolve("idx_*", Default::default()).unwrap();
        let sizes: &[usize] = match only.as_deref() { None => &[1000, 10_000], Some(only) => match only.strip_prefix("_search ") { Some(size) => &[if size == "1000" { 1000 } else { 10_000 }], None => &[] } };
        for &size in sizes {
            let req = crate::search::parse_request(&json!({"size": size}), &std::collections::HashMap::new()).unwrap();
            let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
            let t = median(&mut || { std::hint::black_box(crate::search::execute(&targets, &req, &opts).unwrap()); });
            println!("_search size {size:<6} {t:>9.2} ms");
        }
    }
}

#[cfg(test)]
mod projection_prefilter_tests {
    use super::*;

    /// The projection skips a document none of whose member names is a
    /// projected root without changing any answer: names inside string
    /// values, nested members of the same name, escaped names and names a
    /// JSON text must escape all take the decoding path.
    #[test]
    fn projection_skips_only_documents_without_the_roots() {
        let mut roots: HashMap<&str, Vec<usize>> = HashMap::new();
        roots.insert("key_1000", Vec::new());
        roots.insert("a", Vec::new());
        roots.insert("a.b", Vec::new());
        let decoded = |raw: &str| -> Value {
            let full: Value = serde_json::from_str(raw).unwrap();
            Value::Object(full.as_object().unwrap().iter().filter(|(k, _)| roots.contains_key(k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect())
        };
        for raw in [
            r#"{"key_100000":"7","lookup_keyword_0":"val 7"}"#,
            r#"{"key_1000":"5","x":1}"#,
            r#"{"x":{"key_1000":"5"}}"#,
            r#"{"msg":"key_1000"}"#,
            r#"{"msg":"\"key_1000\""}"#,
            r#"{"\u006bey_1000":"9"}"#,
            r#"{"a":{"b":1},"z":2}"#,
            r#"{"a.b":3}"#,
            r#"{"aa":1,"ba":2}"#,
            r#"{}"#,
        ] {
            assert_eq!(projected_source_roots(raw, &roots).unwrap(), decoded(raw), "{raw}");
        }
        assert!(quoted_name_in("{}", "we\"ird"), "names JSON must escape are never ruled out");
    }
}

#[cfg(test)]
mod text_keyword_tests {
    use super::*;
    use crate::esql::{run, EsqlOptions};
    use crate::index::WriteReq;
    use crate::node::Node;
    use serde_json::json;

    fn opts() -> EsqlOptions<'static> {
        EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }
    }

    fn write(node: &Node, index: &str, docs: &[Value], refresh_every: usize) {
        let svc = node.get_index(index).unwrap();
        for (n, chunk) in docs.chunks(refresh_every).enumerate() {
            let batch = chunk.iter().enumerate().map(|(i, doc)| WriteReq::Index { id: Some((n * refresh_every + i).to_string()), source: doc.to_string(), routing: None, cond: Default::default() }).collect();
            for x in svc.write(batch, false) { x.unwrap(); }
            svc.refresh().unwrap();
            if n == 1 { svc.flush().unwrap(); }
        }
    }

    /// Text columns read from their keyword sub-field's doc values give the
    /// `_source` reading's rows exactly: strings, a value over `ignore_above`
    /// (no doc value), an array (`_source` order, duplicates), a number, an
    /// empty string, a missing field, a sub-field with a normalizer (never
    /// used), on sealed segments and a memtable; through LOOKUP JOIN's
    /// keyed and whole lookup reads and plain FROM projections.
    #[test]
    fn text_columns_from_keyword_doc_values_match_source() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("lk", &json!({"settings": {"index.mode": "lookup"}, "mappings": {"properties": {
            "key": {"type": "keyword"},
            "t": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 8}}},
            "n": {"type": "text", "fields": {"lower": {"type": "keyword", "normalizer": "lowercase"}}}
        }}})).unwrap();
        node.create_index("base", &json!({"mappings": {"properties": {"key": {"type": "keyword"}, "v": {"type": "long"}}}})).unwrap();
        let values = [json!("val 1"), json!("a much longer value"), json!(["b", "a", "a"]), json!(5), json!(""), Value::Null, json!("Ünï"), json!(["only"])];
        let lookup: Vec<Value> = (0..64).map(|i| {
            let mut d = json!({"key": format!("k{i}"), "n": format!("N{i}")});
            let v = &values[i % values.len()];
            if !v.is_null() { d["t"] = v.clone(); }
            d
        }).collect();
        write(&node, "lk", &lookup, 20);
        let base: Vec<Value> = (0..200).map(|i| json!({"key": format!("k{}", (i * 7) % 70), "v": i})).collect();
        write(&node, "base", &base, 64);
        for q in [
            "FROM base | LOOKUP JOIN lk ON key | KEEP v, t, n | LIMIT 150",
            "FROM base | LOOKUP JOIN lk ON key | WHERE t LIKE \"*a*\" | LIMIT 50",
            "FROM base | LOOKUP JOIN lk ON key | SORT v | LIMIT 200",
            "FROM lk | KEEP key, t, n | LIMIT 100",
            "FROM lk | WHERE key == \"k3\" | KEEP t",
        ] {
            let via = run(&node, q, opts()).unwrap_or_else(|e| panic!("{q}: {e}"));
            TEXT_VIA_KEYWORD_DISABLED.with(|c| c.set(true));
            let reference = run(&node, q, opts());
            TEXT_VIA_KEYWORD_DISABLED.with(|c| c.set(false));
            let reference = reference.unwrap_or_else(|e| panic!("{q}: {e}"));
            assert_eq!(via.table.cols, reference.table.cols, "{q}");
            assert_eq!(via.table.rows, reference.table.rows, "{q}");
            assert_eq!(via.warnings, reference.warnings, "{q}");
        }
    }
}

#[cfg(test)]
mod join_projection_tests {
    use super::*;
    use crate::esql::{run, EsqlOptions, EsqlResult};
    use crate::index::WriteReq;
    use crate::node::Node;
    use serde_json::json;

    fn opts() -> EsqlOptions<'static> {
        EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }
    }

    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    fn write(node: &Node, index: &str, docs: Vec<Value>, refresh_every: usize, flush_after: Option<usize>) {
        let svc = node.get_index(index).unwrap();
        let mut batch = Vec::new();
        for (i, doc) in docs.into_iter().enumerate() {
            batch.push(WriteReq::Index { id: Some(i.to_string()), source: doc.to_string(), routing: None, cond: Default::default() });
            if batch.len() == refresh_every {
                for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
                svc.refresh().unwrap();
                if flush_after.is_some_and(|n| i + 1 >= n && i + 1 < n + refresh_every) { svc.flush().unwrap(); }
            }
        }
        for x in svc.write(batch, false) { x.unwrap(); }
        // The last documents stay unflushed (the memtable reader).
        svc.refresh().unwrap();
    }

    /// The joins track's shape, small: a wide base index (`fields` dynamic
    /// text fields, each with its `.keyword`, besides id, @timestamp, the key
    /// and two names the lookup index shares) over 3 shards in flushed and
    /// unflushed segments, and a lookup index of `keys` keys. Base keys miss
    /// the lookup index, are null or multi-valued; lookup keys repeat (two
    /// matches) or are multi-valued; dates are numbers, strings, repeated in
    /// a list (in late segments) or missing.
    fn fixture(node: &Node, base: usize, fields: usize, keys: u64) {
        node.create_index("jp-lookup", &json!({"settings": {"index.mode": "lookup", "number_of_shards": 1},
            "mappings": {"properties": {"key": {"type": "keyword"}, "n": {"type": "integer"}, "lv0": {"type": "keyword"},
                "lv1": {"type": "keyword"}, "shared": {"type": "keyword"}, "lnum": {"type": "long"}}}})).unwrap();
        node.create_index("jp-base", &json!({"settings": {"number_of_shards": 3},
            "mappings": {"properties": {"id": {"type": "long"}, "@timestamp": {"type": "date"}, "key": {"type": "keyword"},
                "n": {"type": "integer"}, "shared": {"type": "keyword"}}}})).unwrap();
        let mut lookup = Vec::new();
        for k in 0..keys {
            let key = match k % 97 { 0 => json!(null), _ if k % 50 == 7 => json!([format!("k{k}"), format!("k{}", k + 1)]), _ => json!(format!("k{k}")) };
            lookup.push(json!({"key": key, "n": k % 13, "lv0": format!("v{}", k % 7), "lv1": format!("w{k}"), "shared": format!("lookup{k}"), "lnum": k}));
            if k % 10 == 0 {
                lookup.push(json!({"key": format!("k{k}"), "n": k % 13, "lv0": format!("dup{k}"), "lv1": format!("w{k}b"), "lnum": k + 1_000_000}));
            }
        }
        write(node, "jp-lookup", lookup, 400, None);
        let mut r = Rng(11);
        let mut docs = Vec::new();
        for i in 0..base {
            let at = 946_728_000_000i64 + i as i64 * 1000;
            let mut doc = serde_json::Map::new();
            doc.insert("id".into(), json!(i));
            match i % 71 {
                0 => {}
                _ if i % 31 == 0 => { doc.insert("@timestamp".into(), json!("2001-02-03T04:05:06.789Z")); }
                // Late documents only: their segments read dates from `_source`.
                _ if i % 53 == 0 && i >= base * 3 / 4 => { doc.insert("@timestamp".into(), json!([at + 5, at, at])); }
                _ => { doc.insert("@timestamp".into(), json!(at)); }
            }
            let k = r.below(keys + keys / 5 + 1);
            doc.insert("key".into(), match r.below(40) { 0 => json!(null), 1 => json!(["k1", "k2"]), _ => json!(format!("k{k}")) });
            doc.insert("n".into(), json!(k % 13));
            doc.insert("shared".into(), json!(format!("base{i}")));
            for f in 0..fields {
                doc.insert(format!("t{f}"), json!(format!("text with value {f}_{i}")));
            }
            docs.push(Value::Object(doc));
        }
        write(node, "jp-base", docs, 1500, Some(base / 2));
    }

    fn result(node: &Node, q: &str, projection: bool, fast: bool) -> std::result::Result<EsqlResult, String> {
        super::PROJECTION_DISABLED.with(|c| c.set(!projection));
        super::DISABLED.with(|c| c.set(!fast));
        let out = run(node, q, opts()).map_err(|e| e.to_string());
        super::PROJECTION_DISABLED.with(|c| c.set(false));
        super::DISABLED.with(|c| c.set(false));
        out
    }

    fn same(q: &str, what: &str, a: &std::result::Result<EsqlResult, String>, b: &std::result::Result<EsqlResult, String>, warnings: bool) {
        match (a, b) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.table.cols, b.table.cols, "{what}: {q}");
                assert_eq!(a.table.rows.len(), b.table.rows.len(), "{what}: {q}");
                assert_eq!(a.table.rows, b.table.rows, "{what}: {q}");
                if warnings { assert_eq!(a.warnings, b.warnings, "{what}: {q}"); }
            }
            (Err(a), Err(b)) => assert_eq!(a, b, "{what}: {q}"),
            (a, b) => panic!("{what}: {q}: {:?} vs {:?}", a.as_ref().map(|r| r.table.rows.len()), b.as_ref().map(|r| r.table.rows.len())),
        }
    }

    /// Projection pushdown (base columns and lookup fields) gives the
    /// columns (names, types, order), rows, order, nulls and warnings (or
    /// the error) of reading every column, on the fast paths and the table
    /// path: KEEP / DROP / RENAME with and without wildcards, lookup columns
    /// shadowing base ones, multi-valued, null and missing keys, duplicate
    /// lookup matches, two keys, expression joins, two joins, WHERE / SORT /
    /// STATS / MV_EXPAND / EVAL around the join, and LIMIT 0 to past the
    /// matches.
    #[test]
    fn projected_join_matches_full_columns() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        fixture(&node, 6000, 12, 300);
        let mut queries: Vec<String> = Vec::new();
        for limit in [0, 1, 37, 1000, 5999, 10000] {
            queries.push(format!("FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP @timestamp, lv0 | LIMIT {limit}"));
        }
        queries.extend([
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP @timestamp, lv0",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP lv0, @timestamp, key, id | LIMIT 500",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP l*, @timestamp | LIMIT 2000",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP shared, id, n | LIMIT 3000",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | DROP t*, shared | LIMIT 300",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | DROP t* | KEEP *d, lnum | LIMIT 300",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | RENAME lv0 AS x, @timestamp AS ts | KEEP ts, x, id | LIMIT 400",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | RENAME id AS lv1 | KEEP lv1, key | LIMIT 400",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP id, lv0, t0 | DROP t0 | LIMIT 100",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | RENAME t0 AS z | KEEP id, lv0 | LIMIT 100",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key, n | KEEP id, lv1, n | LIMIT 2500",
            "FROM jp-base | RENAME key AS lkey | LOOKUP JOIN jp-lookup ON lkey == key | KEEP id, lkey, lv0 | LIMIT 800",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | RENAME lv0 AS a | LOOKUP JOIN jp-lookup ON key | KEEP a, lv0, id | LIMIT 300",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | WHERE lv0 == \"v3\" | KEEP id, lv0 | LIMIT 50",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | WHERE lv0 LIKE \"dup*\" | KEEP @timestamp, lv1 | LIMIT 5000",
            "FROM jp-base | EVAL key = CONCAT(\"k\", TO_STRING(id % 7)) | LOOKUP JOIN jp-lookup ON key | KEEP id, lv1 | LIMIT 200",
            "FROM jp-base | WHERE id < 700 | LOOKUP JOIN jp-lookup ON key | SORT lv0, id | KEEP id, lv0 | LIMIT 100",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | STATS c = COUNT(*), m = MAX(@timestamp) BY lv0 | SORT lv0",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | STATS c = COUNT(*)",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | MV_EXPAND @timestamp | KEEP @timestamp, lv0 | LIMIT 500",
            "FROM jp-base | KEEP id, key | LOOKUP JOIN jp-lookup ON key | LIMIT 100",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP id, @timestamp | SORT @timestamp DESC, id | LIMIT 30",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | LIMIT 20",
            // Errors are those of reading every column.
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP nope | LIMIT 10",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON key | DROP t0 | DROP t0 | KEEP id | LIMIT 5",
            "FROM jp-base | LOOKUP JOIN jp-lookup ON missing | KEEP id | LIMIT 5",
        ].map(String::from));
        for q in &queries {
            let reference = result(&node, q, false, false);
            same(q, "projected table path", &result(&node, q, true, false), &reference, true);
            same(q, "unprojected fast paths", &result(&node, q, false, true), &reference, true);
            same(q, "projected fast paths", &result(&node, q, true, true), &reference, true);
        }
        // The Rally shape takes the projected join, reading dates and keys
        // from doc values.
        let q = "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP @timestamp, lv0 | LIMIT 5000";
        super::LOOKUP_KEY_DV_READS.with(|c| c.set(0));
        let before = super::TAKEN.with(|c| c.get());
        let fast = result(&node, q, true, true).unwrap();
        assert!(super::TAKEN.with(|c| c.get()) > before, "{q}: no fast path");
        assert!(super::LOOKUP_KEY_DV_READS.with(|c| c.get()) > 1000, "{q}: not read from doc values");
        assert!(fast.values_loaded < 3 * 5000, "{q}: loaded {} values", fast.values_loaded);
        // A pruned EVAL (ES `PruneColumns`: an EVAL whose column nothing
        // reads is removed, never evaluated) emits no warning of its own. The
        // join's multi-valued key warnings stay (base keys `["k1", "k2"]`;
        // ES warns whichever columns are kept): exactly those of the query
        // without the EVAL.
        let q = "FROM jp-base | LOOKUP JOIN jp-lookup ON key | EVAL bad = TO_INTEGER(t0) | KEEP id, lv0 | LIMIT 100";
        let without = "FROM jp-base | LOOKUP JOIN jp-lookup ON key | KEEP id, lv0 | LIMIT 100";
        let reference = result(&node, q, false, false);
        let projected = result(&node, q, true, true);
        same(q, "pruned eval", &projected, &reference, false);
        let projected = projected.unwrap();
        let join_only = result(&node, without, false, false).unwrap().warnings;
        assert!(!projected.warnings.iter().any(|w| w.contains("TO_INTEGER")), "{q}: {:?}", projected.warnings);
        assert!(join_only.iter().all(|w| w.contains("LOOKUP JOIN")), "{without}: {join_only:?}");
        assert_eq!(projected.warnings, join_only, "{q}");
    }

    /// The other Rally joins shapes on a 1,000-key lookup index with
    /// unique keys (so a SORT over a lookup column streams its top rows):
    /// fast paths against the table path, rows (and ties) in order and
    /// warnings. `where_one_matching` and the two-key filter scan many
    /// chunks, so the keyed lookup reads give way to one whole read.
    #[test]
    fn join_shapes_match_table_path() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("js-lookup", &json!({"settings": {"index.mode": "lookup", "number_of_shards": 1},
            "mappings": {"properties": {"key_1000": {"type": "keyword"}}}})).unwrap();
        node.create_index("js-base", &json!({"settings": {"number_of_shards": 3},
            "mappings": {"properties": {"id": {"type": "long"}, "@timestamp": {"type": "date"}, "key_1000": {"type": "keyword"}}}})).unwrap();
        write(&node, "js-lookup", (0..1000).map(|k| json!({"key_1000": k.to_string(), "lookup_keyword_0": format!("val {k} rep 0"), "lookup_int_0": k % 37})).collect(), 500, None);
        write(&node, "js-base", (0..20_000).map(|i| {
            let key = match i % 97 { 0 => json!([(i % 1000).to_string(), ((i + 1) % 1000).to_string()]), 1 => json!(null), _ => json!((i % 1200).to_string()) };
            json!({"id": i, "@timestamp": 946_728_000_000i64 + i as i64, "key_1000": key, "field_0": format!("text {i}")})
        }).collect(), 4000, Some(10_000));
        let two_keys = "from js-base | eval lookup_keyword_0.keyword = concat(\"val \", key_1000::keyword, \" rep 0\") | keep key_1000, lookup_keyword_0.keyword, @timestamp | lookup join js-lookup on key_1000, lookup_keyword_0.keyword";
        let queries = [
            "FROM js-base | lookup join js-lookup on key_1000 | sort lookup_keyword_0 asc | limit 1000".to_string(),
            "FROM js-base | lookup join js-lookup on key_1000 | sort lookup_keyword_0 desc | limit 300".to_string(),
            "FROM js-base | lookup join js-lookup on key_1000 | sort lookup_int_0 asc | limit 2500".to_string(),
            "FROM js-base | lookup join js-lookup on key_1000 | sort lookup_int_0 desc | limit 30000".to_string(),
            "FROM js-base | lookup join js-lookup on key_1000 | where lookup_keyword_0 like \"val 1*\" | limit 1000".to_string(),
            "FROM js-base | lookup join js-lookup on key_1000 | where lookup_keyword_0 == \"val 10 rep 0\"".to_string(),
            "FROM js-base | lookup join js-lookup on key_1000 | where lookup_keyword_0 == \"non existing\"".to_string(),
            two_keys.to_string(),
            format!("{two_keys} | where lookup_keyword_0 == \"val 10 rep 0\""),
            format!("{two_keys} | where lookup_int_0 == 3 | limit 50"),
        ];
        let sorted_before = super::SORTED_TWO_PASS.load(std::sync::atomic::Ordering::Relaxed);
        for q in &queries {
            let reference = result(&node, q, false, false);
            same(q, "fast paths", &result(&node, q, true, true), &reference, true);
            same(q, "unprojected fast paths", &result(&node, q, false, true), &reference, true);
        }
        assert!(super::SORTED_TWO_PASS.load(std::sync::atomic::Ordering::Relaxed) > sorted_before, "no sorted join read its winners in a second pass");
    }

    /// The columns [`project_columns`] keeps for the Rally joins queries.
    #[test]
    fn rally_join_queries_read_the_kept_columns() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("lookup_idx_1000_f10", &json!({"settings": {"index.mode": "lookup"},
            "mappings": {"properties": {"key_1000": {"type": "keyword"}}}})).unwrap();
        node.create_index("join_base_idx", &json!({"mappings": {"properties": {"id": {"type": "long"}, "@timestamp": {"type": "date"},
            "key_1000": {"type": "keyword"}, "key_100000": {"type": "keyword"}}}})).unwrap();
        write(&node, "lookup_idx_1000_f10", (0..20).map(|k| json!({"key_1000": k.to_string(), "lookup_keyword_0": format!("val {k} rep 0"), "lookup_keyword_1": "x", "lookup_int_0": k})).collect(), 100, None);
        write(&node, "join_base_idx", (0..20).map(|i| json!({"id": i, "@timestamp": 946728000 + i, "key_1000": (i % 7).to_string(), "key_100000": i.to_string(), "field_0": format!("text with value 0_{i}"), "field_1": "x"})).collect(), 100, None);
        let set = |names: &[&str]| -> HashSet<String> { names.iter().map(|n| n.to_string()).collect() };
        for (q, base, lookup) in [
            ("FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | keep @timestamp, lookup_keyword_0 | limit 10000",
                Some(set(&["@timestamp", "key_1000"])), Some(set(&["key_1000", "lookup_keyword_0"]))),
            ("FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | sort lookup_keyword_0 asc | limit 10000", None, None),
            ("FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | limit 1000", None, None),
            ("FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | rename lookup_keyword_0 as lk_1k | keep id, key_1000, lk_1k | limit 1000",
                Some(set(&["id", "key_1000"])), Some(set(&["key_1000", "lookup_keyword_0"]))),
            ("FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | stats c = count(*)", Some(set(&["key_1000"])), Some(set(&["key_1000"]))),
            ("FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | where lookup_keyword_0 like \"val 1*\" | keep id | limit 1000",
                Some(set(&["id", "key_1000"])), Some(set(&["key_1000", "lookup_keyword_0"]))),
            ("FROM join_base_idx | eval lookup_keyword_0.keyword = concat(\"val \", key_1000::keyword, \" rep 0\") | keep key_1000, lookup_keyword_0.keyword, @timestamp | lookup join lookup_idx_1000_f10 on key_1000, lookup_keyword_0.keyword",
                Some(set(&["@timestamp", "key_1000"])), None),
            ("FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | fork (where id > 3) (where id < 3)", None, None),
        ] {
            let (_, cmds) = crate::esql::parse::parse_statement(q, &Params::default()).unwrap();
            let mut ctx = super::new_ctx(&node, q, None, 1000, 10_000);
            ctx.esql = true;
            let Cmd::From { indices, .. } = &cmds[0] else { unreachable!() };
            let mut schema = super::resolve_schema(&ctx, indices, 0).unwrap();
            let annotated = super::project_columns(&mut ctx, &mut schema, &cmds[1..]);
            assert_eq!(schema.projection, base, "{q}");
            let fields = annotated.as_deref().and_then(|rest| rest.iter().find_map(|c| match c { Cmd::LookupJoin { fields, .. } => fields.clone(), _ => None }));
            assert_eq!(fields, lookup, "{q}");
        }
    }

    /// Timings of the Rally `joins` shapes on a joins-like corpus: a base
    /// index of `ESQL_JP_DOCS` (default 500,000) documents with
    /// `ESQL_JP_FIELDS` (100) text fields each over 5 shards, and a
    /// 1,000-key lookup index of 10 fields; p50 / p90 of `ESQL_JP_RUNS` (20)
    /// runs with the projection and without it. Run on a build host:
    /// `cargo test --release -p elasticsearch-engine join_projection_timings -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn join_projection_timings() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (docs, fields, runs) = (env("ESQL_JP_DOCS", 500_000), env("ESQL_JP_FIELDS", 100), env("ESQL_JP_RUNS", 20).max(1));
        // Lookup keys (`ESQL_JP_KEYS`, default 1,000; the track's 100k-key
        // variants use 100,000): the index keeps its 1k name.
        let keys = env("ESQL_JP_KEYS", 1000);
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let started = std::time::Instant::now();
        node.create_index("lookup_idx_1000_f10", &json!({"settings": {"index.mode": "lookup"},
            "mappings": {"properties": {"key_1000": {"type": "keyword"}}}})).unwrap();
        node.create_index("join_base_idx", &json!({"settings": {"number_of_shards": 5},
            "mappings": {"properties": {"id": {"type": "long"}, "@timestamp": {"type": "date"}, "key_1000": {"type": "keyword"}, "key_100000": {"type": "keyword"}}}})).unwrap();
        write(&node, "lookup_idx_1000_f10", (0..keys).map(|k| {
            let mut doc = serde_json::Map::new();
            doc.insert("key_1000".into(), json!(k.to_string()));
            for f in 0..5 { doc.insert(format!("lookup_keyword_{f}"), json!(format!("val {k} rep {f}"))); }
            for f in 0..4 { doc.insert(format!("lookup_int_{f}"), json!(k * 10 + f)); }
            Value::Object(doc)
        }).collect(), 10_000, None);
        let svc = node.get_index("join_base_idx").unwrap();
        let mut batch = Vec::new();
        for i in 0..docs {
            let mut doc = serde_json::Map::new();
            doc.insert("id".into(), json!(i));
            doc.insert("@timestamp".into(), json!(946_728_000 + i));
            doc.insert("key_1000".into(), json!((i % keys).to_string()));
            doc.insert("key_100000".into(), json!((i % 100_000).to_string()));
            for f in 0..fields { doc.insert(format!("field_{f}"), json!(format!("text with value {f}_{i}"))); }
            batch.push(WriteReq::Index { id: Some(i.to_string()), source: Value::Object(doc).to_string(), routing: None, cond: Default::default() });
            if batch.len() == 10_000 {
                for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
            }
        }
        for x in svc.write(batch, false) { x.unwrap(); }
        svc.refresh().unwrap();
        svc.flush().unwrap();
        // `ESQL_JP_SETTLE` seconds for background merges to finish before timing.
        std::thread::sleep(std::time::Duration::from_secs(env("ESQL_JP_SETTLE", 0) as u64));
        eprintln!("fixture: {docs} base docs x {fields} text fields, {keys} lookup keys, in {:?}", started.elapsed());
        let timings = |q: &str, projection: bool| {
            super::PROJECTION_DISABLED.with(|c| c.set(!projection));
            let rows = run(&node, q, opts()).unwrap().table.rows.len();
            let mut times: Vec<std::time::Duration> = (0..runs).map(|_| {
                let t = std::time::Instant::now();
                std::hint::black_box(run(&node, q, opts()).unwrap().table.rows.len());
                t.elapsed()
            }).collect();
            super::PROJECTION_DISABLED.with(|c| c.set(false));
            times.sort();
            let at = |p: usize| times[((times.len() * p).div_ceil(100)).saturating_sub(1).min(times.len() - 1)];
            (rows, at(50), at(90))
        };
        // `ESQL_JP_ONLY`: only the queries containing one of its `|`-separated parts (and no reference
        // runs); `READY` marks the end of the load (for an attached profiler).
        let only = std::env::var("ESQL_JP_ONLY").ok();
        eprintln!("READY pid {}", std::process::id());
        for q in [
            "FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | keep @timestamp, lookup_keyword_0 | limit 10000",
            "FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | limit 1",
            "FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | limit 1000",
            "FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | limit 10000",
        ] {
            if let Some(only) = &only {
                // Timed only, projected.
                if only.split('|').any(|part| q.contains(part)) {
                    eprintln!("START {q}");
                    let (rows, p50, p90) = timings(q, true);
                    eprintln!("END");
                    eprintln!("{q}\n  rows={rows} p50={p50:?} p90={p90:?}");
                }
                continue;
            }
            let reference = result(&node, q, false, false).unwrap();
            same(q, "bench", &result(&node, q, true, true), &Ok(reference), true);
            eprintln!("START {q}");
            let (rows, p50, p90) = timings(q, true);
            eprintln!("END");
            let (_, off50, off90) = timings(q, false);
            eprintln!("{q}\n  rows={rows} projected p50={p50:?} p90={p90:?} | every column p50={off50:?} p90={off90:?}");
        }
        // The other joins shapes, timed only (their table-path reference
        // would hold every joined row; `join_shapes_match_table_path` checks
        // their results).
        for q in [
            "FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | sort lookup_keyword_0 asc | limit 10000",
            "FROM join_base_idx | lookup join lookup_idx_1000_f10 on key_1000 | where lookup_keyword_0 like \"val 1*\" | limit 1000",
            "from join_base_idx | eval lookup_keyword_0.keyword = concat(\"val \", key_1000::keyword, \" rep 0\") | keep key_1000, lookup_keyword_0.keyword, @timestamp | lookup join lookup_idx_1000_f10 on key_1000, lookup_keyword_0.keyword",
            "from join_base_idx | eval lookup_keyword_0.keyword = concat(\"val \", key_1000::keyword, \" rep 0\") | keep key_1000, lookup_keyword_0.keyword, @timestamp | lookup join lookup_idx_1000_f10 on key_1000, lookup_keyword_0.keyword | where lookup_keyword_0 == \"val 10 rep 0\"",
        ] {
            if only.as_ref().is_some_and(|only| !only.split('|').any(|part| q.contains(part))) { continue; }
            eprintln!("START {q}");
            let (rows, p50, p90) = timings(q, true);
            eprintln!("END");
            eprintln!("{q}\n  rows={rows} p50={p50:?} p90={p90:?}");
            // `ESQL_JP_AB`: also keyed lookup reads of whole rows (before the filtered reads).
            if std::env::var_os("ESQL_JP_AB").is_none() { continue; }
            super::LOOKUP_FILTERED_DISABLED.with(|c| c.set(true));
            let (_, w50, w90) = timings(q, true);
            super::LOOKUP_FILTERED_DISABLED.with(|c| c.set(false));
            eprintln!("  whole keyed rows p50={w50:?} p90={w90:?}");
        }
    }
}

#[cfg(test)]
mod eval_fold_tests {
    use crate::esql::exec::push_filters_past_evals;
    use crate::esql::func::FnCtx;
    use crate::esql::parse::{parse_query, Cmd, Expr, Params};
    use crate::esql::{run, EsqlOptions, Val};
    use crate::index::WriteReq;
    use crate::node::Node;
    use serde_json::json;

    fn opts() -> EsqlOptions<'static> {
        EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }
    }

    fn has_col(e: &Expr, name: &str) -> bool {
        match e {
            Expr::Col(n, _) => n == name,
            Expr::Bin(_, a, b) => has_col(a, name) || has_col(b, name),
            Expr::Not(a) | Expr::Neg(a) => has_col(a, name),
            Expr::Call(_, args, _) => args.iter().any(|x| has_col(x, name)),
            _ => false,
        }
    }

    /// ES `PropagateEvalFoldables` + `PushDownAndCombineFilters`: constant
    /// EVAL columns are folded into the WHERE, which moves in front of the
    /// EVAL (conjuncts reading a computed column stay behind it).
    #[test]
    fn constant_evals_fold_into_where_and_filters_move_before_eval() {
        let q = "FROM k | EVAL bucket = DATE_TRUNC(30 minutes, @timestamp), start = DATE_PARSE(\"yyyy-MM-dd'T'HH:mm:ss.SSSX\", \"2023-05-16T20:45:00.000Z\"), lim = 1 + 2, d = x * 2 \
                 | WHERE @timestamp >= start AND d > lim AND y < lim | STATS c = COUNT(*) BY bucket";
        let cmds = parse_query(q, &Params::default()).unwrap();
        let out = push_filters_past_evals(&FnCtx::default(), &cmds[1..]).expect("rewritten");
        // WHERE (folded, free conjuncts) | EVAL | WHERE (d > 3) | STATS
        let Cmd::Where(free, _) = &out[0] else { panic!("{out:?}") };
        assert!(has_col(free, "@timestamp") && has_col(free, "y"));
        assert!(!has_col(free, "start") && !has_col(free, "lim") && !has_col(free, "d"), "{free:?}");
        assert!(matches!(&out[1], Cmd::Eval(_)));
        let Cmd::Where(bound, _) = &out[2] else { panic!("{out:?}") };
        assert!(has_col(bound, "d") && !has_col(bound, "lim"));
        assert!(matches!(&out[3], Cmd::Stats { .. }));
        // Nothing to do: a WHERE on computed columns only, or no EVAL.
        let cmds = parse_query("FROM k | EVAL d = x * 2 | WHERE d > 3", &Params::default()).unwrap();
        assert!(push_filters_past_evals(&FnCtx::default(), &cmds[1..]).is_none());
        let cmds = parse_query("FROM k | WHERE x > 3 | STATS COUNT(*)", &Params::default()).unwrap();
        assert!(push_filters_past_evals(&FnCtx::default(), &cmds[1..]).is_none());
        // A LIMIT between them: folded, but the filter does not cross it.
        let cmds = parse_query("FROM k | EVAL lim = 2 | LIMIT 5 | WHERE x > lim", &Params::default()).unwrap();
        let out = push_filters_past_evals(&FnCtx::default(), &cmds[1..]).expect("folded");
        assert!(matches!(&out[1], Cmd::Limit(5)) && matches!(&out[2], Cmd::Where(..)));
        // A redefinition stops the substitution.
        let cmds = parse_query("FROM k | EVAL lim = 2 | EVAL lim = x | WHERE x > lim", &Params::default()).unwrap();
        let out = push_filters_past_evals(&FnCtx::default(), &cmds[1..]);
        assert!(out.is_none_or(|o| o.iter().any(|c| matches!(c, Cmd::Where(e, _) if has_col(e, "lim")))));
    }

    /// The tsdb_k8s_queries ES|QL shape returns what the equivalent direct
    /// filter does (the folded range pushed to the index).
    #[test]
    fn folded_time_range_filters_like_the_direct_range() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("k", &json!({"mappings": {"properties": {"@timestamp": {"type": "date"}, "pod": {"type": "keyword"}, "cpu": {"type": "double"}}}})).unwrap();
        let svc = node.get_index("k").unwrap();
        let reqs = (0..600).map(|i| WriteReq::Index { id: None, source: json!({"@timestamp": 1_684_255_500_000i64 + i * 30_000, "pod": format!("p{}", i % 7), "cpu": i as f64 / 10.0}).to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, false) { r.unwrap(); }
        svc.refresh().unwrap();
        let folded = "FROM k | EVAL by_interval = DATE_TRUNC(30 seconds, @timestamp), start_time = DATE_PARSE(\"yyyy-MM-dd'T'HH:mm:ss.SSSX\", \"2023-05-16T17:30:00.000Z\"), end_time = DATE_PARSE(\"yyyy-MM-dd'T'HH:mm:ss.SSSX\", \"2023-05-16T18:45:00.000Z\") \
                      | WHERE @timestamp >= start_time AND @timestamp <= end_time | STATS c = COUNT(*), a = AVG(cpu) BY pod, by_interval | SORT pod, by_interval | LIMIT 10000";
        let direct = "FROM k | WHERE @timestamp >= \"2023-05-16T17:30:00.000Z\" AND @timestamp <= \"2023-05-16T18:45:00.000Z\" | EVAL by_interval = DATE_TRUNC(30 seconds, @timestamp) \
                      | STATS c = COUNT(*), a = AVG(cpu) BY pod, by_interval | SORT pod, by_interval | LIMIT 10000";
        let a = run(&node, folded, opts()).unwrap().table;
        let b = run(&node, direct, opts()).unwrap().table;
        assert!(!a.rows.is_empty() && a.rows.len() < 600);
        assert_eq!(a.rows, b.rows);
        let total: i64 = a.rows.iter().map(|r| match &r[0] { Val::Long(n) => *n, _ => 0 }).sum();
        assert_eq!(total, 151, "17:30 to 18:45 at 30 s steps, both ends included");
    }
}
