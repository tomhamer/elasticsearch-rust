//! ES|QL execution over in-memory tables.

use super::func::{self, FnCtx, AGGS, FULL_TEXT};
use super::parse::{parse_query, BinOp, Cmd, Expr, Field, Params, SortKey};
use super::{cmp_vals, verification_err, Val};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use crate::node::{Node, ResolveOpts};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

#[path = "fast.rs"]
mod fast;
#[path = "sql_pivot.rs"]
mod sql_pivot_stream;

#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub name: String,
    pub ty: String,
}

#[derive(Clone, Debug, Default)]
pub struct Table {
    pub cols: Vec<Column>,
    pub rows: Vec<Vec<Val>>,
}

impl Table {
    fn idx(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|c| c.name == name)
    }

    /// Append (or replace, moving to the end) a column.
    fn put_col(&mut self, name: &str, ty: String, vals: Vec<Val>) {
        if let Some(i) = self.idx(name) {
            self.cols.remove(i);
            for r in self.rows.iter_mut() {
                r.remove(i);
            }
        }
        self.cols.push(Column { name: name.to_string(), ty });
        for (r, v) in self.rows.iter_mut().zip(vals) {
            r.push(v);
        }
    }

    pub(crate) fn project(&mut self, keep: &[usize]) {
        self.cols = keep.iter().map(|i| self.cols[*i].clone()).collect();
        for r in self.rows.iter_mut() {
            *r = keep.iter().map(|i| std::mem::replace(&mut r[*i], Val::Null)).collect();
        }
    }
}

pub struct EsqlOptions<'a> {
    pub params: Params,
    pub filter: Option<&'a Value>,
    /// Implicit LIMIT when the query has none.
    pub default_limit: usize,
    /// Cap on any LIMIT.
    pub max_limit: usize,
    /// `locale` (BCP 47 tag) for textual date formatting.
    pub locale: Option<String>,
    /// Query settings from the request body (`settings`, root aliases).
    pub body_settings: super::settings::BodySettings,
    /// The caller's document-/field-level security, stamped on every index read.
    pub restrict: Option<Restrict>,
    /// Cluster: where each wanted shard is read ([`super::remote`]).
    pub shards: Option<super::remote::SharedShardSource>,
    /// The response drops all-null columns (`drop_null_columns`): the result
    /// may hold only the non-null columns, every column listed in
    /// [`EsqlResult::all_columns`].
    pub drop_null_columns: bool,
    /// The caller renders a result holding only its non-null columns with
    /// every column listed in [`EsqlResult::all_columns`] as nulls
    /// ([`super::format::columns_expanded`]), when nulls are not dropped:
    /// a wide plain `FROM ... | LIMIT n` is then built over the columns
    /// with values only (no dense row of nulls per document).
    pub sparse_nulls: bool,
}

/// Stamps resolved targets with the caller's DLS/FLS (errors deny the read).
pub type Restrict = std::sync::Arc<dyn Fn(Vec<crate::search::Target>) -> Result<Vec<crate::search::Target>> + Send + Sync>;

impl Default for EsqlOptions<'_> {
    fn default() -> Self {
        EsqlOptions { params: Params::default(), filter: None, default_limit: 1000, max_limit: 10_000, locale: None, body_settings: Default::default(), restrict: None, shards: None, drop_null_columns: false, sparse_nulls: false }
    }
}

pub struct EsqlResult {
    pub table: Table,
    /// Response warnings (`Warning` headers).
    pub warnings: Vec<String>,
    /// Usage telemetry: the features (commands) and functions the query used.
    pub features: BTreeSet<&'static str>,
    pub functions: BTreeSet<&'static str>,
    pub took_ms: u64,
    pub documents_found: u64,
    pub values_loaded: u64,
    /// Rows the pipeline's operators emitted (all stages).
    pub rows_emitted: u64,
    /// The resolved query settings.
    pub settings: super::settings::Settings,
    /// `_meta` of output columns, by column name.
    pub col_meta: HashMap<String, Value>,
    /// With [`EsqlOptions::drop_null_columns`]: every column of the result,
    /// when `table` holds only those with a non-null value (the
    /// `all_columns` / `columns` of the response).
    pub all_columns: Option<Vec<Column>>,
}

pub struct Ctx<'a> {
    pub node: Option<&'a Node>,
    pub full: &'a str,
    pub filter: Option<&'a Value>,
    pub fctx: FnCtx,
    pub documents_found: u64,
    pub values_loaded: u64,
    pub default_limit: usize,
    pub max_limit: usize,
    /// Mapped types (and their indices) of the `unsupported` columns `FROM` loaded.
    pub unsupported: HashMap<String, Vec<(String, Vec<String>)>>,
    /// Index fields still holding their mapped values (full-text functions
    /// may only target these).
    pub pristine: HashSet<String>,
    /// The first command after which full-text functions may no longer run (`LIMIT`, `ROW`, ...).
    pub fulltext_barrier: Option<&'static str>,
    /// The nearest command `KQL`/`QSTR` cannot follow (anything but
    /// `WHERE`, `SORT`, `SAMPLE` and the source).
    pub lexical_barrier: Option<&'static str>,
    /// Resolved query settings (`SET`, request body).
    pub settings: super::settings::Settings,
    pub rows_emitted: u64,
    /// ES|QL (vs. the SQL layer reusing this engine): ES|QL function resolution applies.
    pub esql: bool,
    /// The source is `TS`: the first `STATS` aggregates per time series.
    pub ts: bool,
    pub ts_done: bool,
    /// Time series dimension columns of the `TS` source.
    pub ts_dims: Vec<String>,
    /// `_meta` of output columns (`column_metadata`), by column name.
    pub col_meta: std::cell::RefCell<HashMap<String, Value>>,
    /// Views being expanded (cycle detection), and whether any was read.
    pub view_stack: Vec<String>,
    pub used_views: bool,
    /// Unmapped names the next FROM must load for an enclosing union.
    pub extra_load: Vec<String>,
    /// The caller's DLS/FLS ([`EsqlOptions::restrict`]).
    pub restrict: Option<Restrict>,
    /// Cluster: where each wanted shard is read ([`EsqlOptions::shards`]).
    pub shards: Option<super::remote::SharedShardSource>,
    /// Request-breaker accounting of the query's pages (ES `BlockFactory`):
    /// the table held between commands, resized as it changes
    /// ([`account_table`]), released when the query ends or fails.
    pub mem: crate::memory::RequestMemory,
    /// Bytes of the table currently charged to `mem`.
    pub table_bytes: std::cell::Cell<u64>,
    /// The top-level pipeline's result may drop its all-null columns
    /// ([`EsqlOptions::drop_null_columns`]; off in sub-pipelines).
    pub drop_null_columns: bool,
    /// Every column of a result that dropped its all-null ones.
    pub all_columns: Option<Vec<Column>>,
}

/// The request-breaker label of ES|QL's memory (ES `BlockFactory`).
pub(crate) const ESQL_LABEL: &str = "<esql_block_factory>";

/// Heap bytes of one row.
pub(crate) fn row_bytes(r: &[Val]) -> u64 {
    24 + (r.len() * std::mem::size_of::<Val>()) as u64 + r.iter().map(|v| v.heap_bytes() as u64).sum::<u64>()
}

/// Visit every retained row: a sampled estimate can miss a single wide
/// expression result and release its handoff reservation too early.
fn table_bytes(t:&Table)->u64 {
    t.rows.iter().fold((t.rows.capacity() as u64).saturating_mul(std::mem::size_of::<Vec<Val>>() as u64),|sum,row|
        sum.saturating_add((row.capacity() as u64).saturating_mul(std::mem::size_of::<Val>() as u64)).saturating_add(row.iter().map(|v|v.heap_bytes() as u64).fold(0u64,u64::saturating_add)))
}

/// Charge the table the pipeline now holds in place of the previous one
/// (a trip fails the query with 429).
pub(crate) fn account_table(ctx: &Ctx, t: &Table) -> Result<()> {
    let est = table_bytes(t);
    let cur = ctx.table_bytes.get();
    if est > cur {
        ctx.mem.charge(est - cur, ESQL_LABEL)?;
    } else {
        ctx.mem.release(cur - est);
    }
    ctx.table_bytes.set(est);
    Ok(())
}

/// Parse and run an ES|QL query.
pub fn run(node: &Node, query: &str, opts: EsqlOptions) -> Result<EsqlResult> {
    let start = Instant::now();
    let (sets, cmds) = super::parse::parse_statement(query, &opts.params)?;
    let settings = super::settings::resolve(query, &sets, &opts.body_settings)?;
    let _zone = super::set_zone(settings.time_zone);
    let mut ctx = Ctx {
        node: Some(node),
        full: query,
        filter: opts.filter,
        fctx: FnCtx { query: query.to_string(), now: crate::date::now_millis(), locale: opts.locale.clone(), tz: settings.time_zone, ..Default::default() },
        documents_found: 0,
        values_loaded: 0,
        default_limit: opts.default_limit,
        max_limit: opts.max_limit,
        unsupported: HashMap::new(),
        pristine: HashSet::new(),
        fulltext_barrier: None,
        lexical_barrier: None,
        settings,
        rows_emitted: 0,
        esql: true,
        ts: false,
        ts_done: false,
        ts_dims: Vec::new(),
        col_meta: Default::default(),
        view_stack: Vec::new(),
        used_views: false,
        extra_load: Vec::new(),
        restrict: opts.restrict.clone(),
        shards: opts.shards.clone(),
        mem: Default::default(),
        table_bytes: Default::default(),
        drop_null_columns: false,
        all_columns: None,
    };
    ctx.drop_null_columns = (opts.drop_null_columns || opts.sparse_nulls) && super::approx::enabled(&ctx.settings.approximation).is_none();
    let mut table = run_cmds(&mut ctx, &cmds)?;
    let mut all_columns = ctx.all_columns.take();
    // Internal columns never reach the response.
    if table.cols.iter().any(|c| c.name.starts_with('\u{0}')) {
        let keep: Vec<usize> = (0..table.cols.len()).filter(|i| !table.cols[*i].name.starts_with('\u{0}')).collect();
        table.project(&keep);
    }
    if let Some(all) = &mut all_columns {
        all.retain(|c| !c.name.starts_with('\u{0}'));
    }
    for c in table.cols.iter_mut().chain(all_columns.iter_mut().flatten()) {
        if c.ty.starts_with(SUBQUERY_CONFLICT) {
            c.ty = "unsupported".into();
        }
    }
    // Unsupported (and unconverted union-type) columns always read as null.
    for (i, c) in table.cols.iter().enumerate() {
        if c.ty.starts_with("unsupported") {
            for r in table.rows.iter_mut() {
                r[i] = Val::Null;
            }
        }
    }
    // A table of the non-null columns keeps only those still non-null.
    if all_columns.is_some() {
        let keep = super::format::non_null_columns(&table);
        if keep.len() < table.cols.len() {
            table.project(&keep);
        }
    }
    let mut warnings = std::mem::take(&mut ctx.fctx.warnings.borrow_mut().list);
    // `AddImplicitLimit`: a query without LIMIT is capped (time-series aggregations silently).
    if !has_limit(&cmds) && !is_ts_aggregate(&cmds) {
        warnings.insert(0, format!("No limit defined, adding default limit of [{}]", ctx.default_limit));
    }
    let (mut features, functions) = super::telemetry::usage(&cmds);
    if ctx.used_views {
        features.insert("view");
    }
    let mut col_meta = if ctx.settings.column_metadata { std::mem::take(&mut *ctx.col_meta.borrow_mut()) } else { HashMap::new() };
    if let Some(conf) = super::approx::enabled(&ctx.settings.approximation) {
        super::approx::apply(query, &cmds, conf, &mut table, &mut col_meta, &mut warnings);
    }
    Ok(EsqlResult {
        table,
        warnings,
        features,
        functions,
        took_ms: start.elapsed().as_millis() as u64,
        documents_found: ctx.documents_found,
        values_loaded: ctx.values_loaded,
        rows_emitted: ctx.rows_emitted,
        settings: ctx.settings.clone(),
        col_meta,
        all_columns,
    })
}

/// Whether the plan has an explicit LIMIT (anywhere, FORK branches included).
/// `LIMIT n BY` is not one: ES's `AddImplicitLimit` looks for `Limit` nodes
/// only, so `FROM i | LIMIT 1 BY k` gets the default limit and its warning
/// (`AnalyzerTests.testImplicitDefaultLimitAfterLimitBy`).
fn has_limit(cmds: &[Cmd]) -> bool {
    cmds.iter().any(|c| match c {
        Cmd::Limit(_) => true,
        Cmd::Fork(bs) => bs.iter().any(|b| has_limit(b)),
        _ => false,
    })
}

/// Whether the plan aggregates a time series (`TS ... | STATS`), whose
/// implicit limit is not reported.
fn is_ts_aggregate(cmds: &[Cmd]) -> bool {
    matches!(cmds.first(), Some(Cmd::Promql(_))) || matches!(cmds.first(), Some(Cmd::From { ts: true, .. })) && cmds.iter().any(|c| matches!(c, Cmd::Stats { inline: false, .. }))
}

/// Run a parsed top-level pipeline (with the implicit/maximum limit).
pub fn run_cmds(ctx: &mut Ctx, cmds: &[Cmd]) -> Result<Table> {
    let limited = has_limit(cmds);
    let cap = match (limited, is_ts_aggregate(cmds)) {
        (true, false) => ctx.max_limit,
        (false, false) => ctx.default_limit,
        (true, true) => 10_000_000,
        (false, true) => 10_000,
    };
    // Expose the top-level output cap to safe row-preserving operators,
    // rather than materializing an expanding join before truncating it
    // (or a grouped limit's rows past the cap: ES's implicit `Limit` on top).
    let mut capped;
    let commands = if ctx.esql && !limited && cmds.iter().any(|c| matches!(c, Cmd::LookupJoin { .. } | Cmd::LimitBy(..))) {
        capped = cmds.to_vec();
        capped.push(Cmd::Limit(cap));
        capped.as_slice()
    } else { cmds };
    let mut t = exec_pipeline(ctx, commands, None)?;
    t.rows.truncate(cap);
    ctx.rows_emitted += t.rows.len() as u64;
    Ok(t)
}

// ------------------------------------------------------------------ binding

fn unknown_col(ctx: &Ctx, name: &str, pos: usize, cols: &[Column]) -> EsError {
    // KEEP/DROP/RENAME carry their keyword's position; ES points at the column itself.
    let off = ctx.full.get(pos..).and_then(|s| s.find(name)).unwrap_or(0);
    let names: Vec<&str> = cols.iter().filter(|c| !c.name.starts_with('\u{0}')).map(|c| c.name.as_str()).collect();
    verification_err(ctx.full, pos + off, format!("Unknown column [{name}]{}", super::did_you_mean(name, &names)))
}

/// `Cannot use field [..]` for an `unsupported` (or ambiguous union-type) column.
pub(crate) fn unsupported_err(ctx: &Ctx, name: &str, pos: usize, ty: &str) -> EsError {
    let listed: Vec<(String, Vec<String>)> = match ctx.unsupported.get(name) {
        Some(t) => t.clone(),
        None => ty.strip_prefix("unsupported:").map(|t| t.split(',').map(|x| (x.to_string(), vec![])).collect()).unwrap_or_default(),
    };
    if listed.len() >= 2 {
        let parts: Vec<String> = listed
            .iter()
            .map(|(t, ix)| {
                if ix.first().map(String::as_str) == Some(SOURCE_LEG) {
                    return format!("[{t}] due to loading from _source");
                }
                let shown = ix.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
                let more = ix.len().saturating_sub(3);
                match more {
                    0 => format!("[{t}] in [{shown}]"),
                    1 => format!("[{t}] in [{shown}] and [1] other index"),
                    m => format!("[{t}] in [{shown}] and [{m}] other indices"),
                }
            })
            .collect();
        return verification_err(ctx.full, pos, format!("Cannot use field [{name}] due to ambiguities being mapped as [{}] incompatible types: {}", listed.len(), parts.join(", ")));
    }
    let t = listed.first().map(|(t, _)| t.clone()).unwrap_or_else(|| "unsupported".into());
    verification_err(ctx.full, pos, format!("Cannot use field [{name}] with unsupported type [{t}]"))
}

/// The conversion function a `::type` cast stands for.
fn cast_fn(ty: &str) -> &'static str {
    match func::cast_type(ty) {
        "long" => "TO_LONG",
        "integer" => "TO_INTEGER",
        "double" => "TO_DOUBLE",
        "boolean" => "TO_BOOLEAN",
        "date" => "TO_DATETIME",
        "date_nanos" => "TO_DATE_NANOS",
        "ip" => "TO_IP",
        "version" => "TO_VERSION",
        "geo_point" => "TO_GEOPOINT",
        "cartesian_point" => "TO_CARTESIANPOINT",
        "geo_shape" => "TO_GEOSHAPE",
        "cartesian_shape" => "TO_CARTESIANSHAPE",
        "unsigned_long" => "TO_UNSIGNED_LONG",
        "aggregate_metric_double" => "TO_AGGREGATE_METRIC_DOUBLE",
        _ => "TO_STRING",
    }
}

/// Can conversion function `f` read values of ES|QL type `ty` (union-type resolution)?
fn converts_from(f: &str, ty: &str) -> bool {
    let ty = if ty == "datetime" { "date" } else { ty };
    let numeric = matches!(ty, "long" | "integer" | "double" | "unsigned_long" | "counter_long" | "counter_integer" | "counter_double");
    let stringy = matches!(ty, "keyword" | "text");
    match f {
        "TO_STRING" | "TO_STR" | "TO_KEYWORD" => matches!(ty, "keyword" | "text" | "boolean" | "date" | "date_nanos" | "ip" | "version" | "long" | "integer" | "double" | "unsigned_long" | "geo_point" | "cartesian_point" | "geo_shape" | "cartesian_shape"),
        "TO_TEXT" => stringy,
        "TO_IP" => stringy || ty == "ip",
        "TO_LONG" | "TO_INTEGER" | "TO_INT" | "TO_DOUBLE" | "TO_DBL" | "TO_UNSIGNED_LONG" | "TO_UL" | "TO_ULONG" => stringy || numeric || matches!(ty, "boolean" | "date" | "date_nanos"),
        "TO_DATETIME" | "TO_DT" | "TO_DATE_NANOS" => stringy || matches!(ty, "date" | "date_nanos" | "long" | "integer" | "double" | "unsigned_long"),
        "TO_BOOLEAN" | "TO_BOOL" => stringy || matches!(ty, "boolean" | "long" | "integer" | "double" | "unsigned_long"),
        "TO_VERSION" | "TO_VER" => stringy || ty == "version",
        "TO_GEOPOINT" => stringy || ty == "geo_point",
        "TO_CARTESIANPOINT" => stringy || ty == "cartesian_point",
        "TO_GEOSHAPE" => stringy || matches!(ty, "geo_point" | "geo_shape"),
        "TO_CARTESIANSHAPE" => stringy || matches!(ty, "cartesian_point" | "cartesian_shape"),
        "TO_AGGREGATE_METRIC_DOUBLE" | "TO_AGGREGATEMETRICDOUBLE" => numeric || ty == "aggregate_metric_double",
        _ => false,
    }
}

/// A union-type column converted by `f`: bound when every mapped type converts.
fn bind_converted(ctx: &Ctx, f: &str, arg: &Expr, cols: &[Column]) -> Option<Result<Expr>> {
    let Expr::Col(n, p) = arg else { return None };
    let i = cols.iter().rposition(|c| &c.name == n)?;
    let ty = &cols[i].ty;
    let types: Vec<String> = ty.strip_prefix("unsupported:")?.split(',').map(str::to_string).collect();
    if types.len() < 2 {
        return None;
    }
    Some(if types.iter().all(|t| converts_from(f, t)) { Ok(Expr::Idx(i)) } else { Err(unsupported_err(ctx, n, *p, ty)) })
}

pub fn bind(ctx: &Ctx, e: &Expr, cols: &[Column]) -> Result<Expr> {
    Ok(match e {
        Expr::Col(n, p) => match cols.iter().rposition(|c| &c.name == n) {
            Some(i) if cols[i].ty.starts_with(SUBQUERY_CONFLICT) => {
                let ts = cols[i].ty[SUBQUERY_CONFLICT.len()..].split(',').map(type_label).collect::<Vec<_>>().join(", ");
                return Err(verification_err(ctx.full, *p, format!("Column [{n}] has conflicting data types in subqueries: [{ts}]")));
            }
            Some(i) if cols[i].ty.starts_with("unsupported") => return Err(unsupported_err(ctx, n, *p, &cols[i].ty)),
            Some(i) => Expr::Idx(i),
            None => return Err(unknown_col(ctx, n, *p, cols)),
        },
        Expr::Call(name, args, p) if !ctx.esql && args.len() == 1 && func::is_conversion(name) && func::known(name) => match bind_converted(ctx, name, &args[0], cols) {
            Some(r) => Expr::Call(name.clone(), vec![r?], *p),
            None => Expr::Call(name.clone(), vec![bind(ctx, &args[0], cols)?], *p),
        },
        Expr::Cast(a, t) => match bind_converted(ctx, cast_fn(t), a, cols) {
            Some(r) => Expr::Cast(Box::new(r?), t.clone()),
            None => {
                let b = bind(ctx, a, cols)?;
                if ctx.esql {
                    check_cast(ctx, a, &b, t, cols)?;
                }
                let b = match func::cast_type(t) {
                    "long" => grid_to_long(&b, cols, expr_pos(a).unwrap_or(0)).unwrap_or(b),
                    _ => b,
                };
                Expr::Cast(Box::new(b), t.clone())
            }
        },
        Expr::Bin(op, a, b) => {
            let (ba, bb) = (bind(ctx, a, cols)?, bind(ctx, b, cols)?);
            // Counters only take part in expressions once cast (`::long`).
            if !matches!(op, BinOp::And | BinOp::Or) {
                for (x, bx) in [(a, &ba), (b, &bb)] {
                    if let Some(t) = type_of(bx, cols).filter(|t| t.starts_with("counter_")) {
                        let (name, p) = match x.as_ref() {
                            Expr::Col(n, p) => (n.clone(), *p),
                            _ => (String::new(), expr_pos(x).unwrap_or(0)),
                        };
                        return Err(verification_err(ctx.full, p, format!("argument of [{name}] must be [numeric except counter types], found value [{name}] type [{t}]")));
                    }
                }
            }
            Expr::Bin(*op, Box::new(ba), Box::new(bb))
        }
        Expr::Neg(a) => Expr::Neg(Box::new(bind(ctx, a, cols)?)),
        Expr::Not(a) => Expr::Not(Box::new(bind(ctx, a, cols)?)),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(bind(ctx, a, cols)?), *n),
        Expr::In(a, l, n) => Expr::In(Box::new(bind(ctx, a, cols)?), l.iter().map(|x| bind(ctx, x, cols)).collect::<Result<_>>()?, *n),
        Expr::Like(a, p, n, r) => Expr::Like(Box::new(bind(ctx, a, cols)?), p.clone(), *n, *r),
        Expr::Between(a, b, c, n) => Expr::Between(Box::new(bind(ctx, a, cols)?), Box::new(bind(ctx, b, cols)?), Box::new(bind(ctx, c, cols)?), *n),
        Expr::Call(name, args, p) => {
            if ctx.esql {
                return bind_call(ctx, name, args, *p, cols);
            }
            if !func::known(name) {
                return Err(verification_err(ctx.full, *p, format!("Unknown function [{}]", name.to_lowercase())));
            }
            if name == "CASE" {
                record_case_sources(ctx, args, *p, cols);
            }
            Expr::Call(name.clone(), args.iter().map(|x| bind(ctx, x, cols)).collect::<Result<_>>()?, *p)
        }
        other => other.clone(),
    })
}

/// `x::type` resolves like the conversion function it stands for.
fn check_cast(ctx: &Ctx, a: &Expr, b: &Expr, t: &str, cols: &[Column]) -> Result<()> {
    let f = cast_fn(t).to_ascii_lowercase();
    let Some(d) = super::typecheck::lookup(&f) else { return Ok(()) };
    let ty = type_of(b, cols).unwrap_or_else(|| "null".into());
    if ty.starts_with("unsupported") || super::typecheck::param_accepts_n(d, 0, &ty, Some(1)) {
        return Ok(());
    }
    let Some(start) = expr_pos(a) else { return Ok(()) };
    // Source text: the operand, `::` and the type name.
    let rest = &ctx.full[start..];
    let Some(cc) = rest.find("::") else { return Ok(()) };
    let after = &rest[cc + 2..];
    let tl = after.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').map(char::len_utf8).sum::<usize>();
    let src = &rest[..cc + 2 + tl];
    let arg = rest[..cc].trim();
    Err(verification_err(ctx.full, start, format!("argument of [{src}] must be [{}], found value [{arg}] type [{}]", super::typecheck::phrase(d, 0), type_label(&ty))))
}

/// A function's name as the query spells it.
fn typed_name(src: &str, pos: usize, name: &str) -> String {
    let w: String = src.get(pos..).unwrap_or("").chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    if w.eq_ignore_ascii_case(name) { w } else { name.to_ascii_lowercase() }
}

/// Functions internal to this engine (operators the parser lowers to calls).
fn internal_fn(name: &str) -> bool {
    matches!(name, "MATCH_OP")
}

/// The source text of the function call starting at `pos` (up to its
/// closing parenthesis).
pub fn call_src(src: &str, pos: usize) -> String {
    let args = call_arg_sources(src, pos);
    let b = src.as_bytes();
    let Some(open) = src.get(pos..).and_then(|r| r.find('(')).map(|o| pos + o) else { return src.get(pos..).unwrap_or("").to_string() };
    // The closing parenthesis follows the last argument.
    let mut i = match args.last() {
        Some((off, t)) => off + t.len(),
        None => open + 1,
    };
    while i < b.len() && b[i] != b')' {
        i += 1;
    }
    src[pos..(i + 1).min(src.len())].to_string()
}

/// A function's resolution: unknown functions, arity, string-literal
/// implicit casts and argument types, like ES's analyzer and verifier.
fn bind_call(ctx: &Ctx, name: &str, args: &[Expr], p: usize, cols: &[Column]) -> Result<Expr> {
    let lower = name.to_ascii_lowercase();
    // `TBUCKET(span)` buckets `@timestamp`.
    if name == "TBUCKET" && !args.is_empty() && args.len() != 2 {
        if !cols.iter().any(|c| c.name == "@timestamp") {
            let src = call_src(ctx.full, p);
            return Err(verification_err(ctx.full, p, format!("[{src}] requires the [@timestamp] field, which was either not present in the source index, or has been dropped or renamed")));
        }
        let mut a = vec![Expr::Col("@timestamp".into(), p)];
        a.extend(args.iter().cloned());
        let b = bind_call(ctx, "BUCKET", &a, p, cols)?;
        return Ok(match b {
            Expr::Call(_, a, p) => Expr::Call("BUCKET".into(), a, p),
            o => o,
        });
    }
    if super::ts::is_ts_fn(name) {
        let src = call_src(ctx.full, p);
        if !cols.iter().any(|c| c.name == "@timestamp") {
            return Err(verification_err(ctx.full, p, format!("[{src}] requires the [@timestamp] field, which was either not present in the source index, or has been dropped or renamed")));
        }
        return Err(verification_err(ctx.full, p, format!("time_series aggregate[{src}] can only be used with the TS command")));
    }
    let def = super::typecheck::lookup(&lower);
    let typed = typed_name(ctx.full, p, name);
    if def.is_none() && !internal_fn(name) {
        let names = super::typecheck::names();
        return Err(verification_err(ctx.full, p, format!("Unknown function [{typed}]{}", super::did_you_mean(&lower, &names))));
    }
    if let Some(d) = def {
        if let Some(m) = super::typecheck::arity_error(d, args.len()) {
            return Err(super::parsing_err(ctx.full, p, format!("error building [{typed}]: {m}")));
        }
    }
    if name == "CASE" {
        record_case_sources(ctx, args, p, cols);
    }
    let mut bargs: Vec<Expr> = Vec::with_capacity(args.len());
    for (i, a) in args.iter().enumerate() {
        // Conversion functions read union-typed fields natively.
        if i == 0 && args.len() == 1 && func::is_conversion(name) {
            if let Some(r) = bind_converted(ctx, name, a, cols) {
                bargs.push(r?);
                continue;
            }
        }
        bargs.push(bind(ctx, a, cols)?);
    }
    if let Some(d) = def {
        check_args(ctx, d, name, args, &mut bargs, cols, p)?;
    }
    // Geographic relations evaluate as ES does them (the left geometry as a
    // shape doc value, the right as the query's `Component2D`): marked here,
    // where the argument types are known.
    if matches!(name, "ST_INTERSECTS" | "ST_DISJOINT" | "ST_WITHIN" | "ST_CONTAINS") && bargs.len() == 2 && bargs.iter().any(|a| type_of(a, cols).is_some_and(|t| t == "geo_point" || t == "geo_shape")) {
        bargs.push(Expr::Lit(Val::Str(func::GEO_RELATION.into())));
    }
    if name == "TO_LONG" && bargs.len() == 1 {
        if let Some(cell) = grid_to_long(&bargs[0], cols, p) {
            return Ok(Expr::Cast(Box::new(cell), "long".into()));
        }
    }
    Ok(Expr::Call(name.to_string(), bargs, p))
}

/// A bound `geohash` / `geotile` / `geohex` value read as its long (ES
/// `ToLong` passes the cell's long through; cells are held as addresses
/// here): `None` for other types.
fn grid_to_long(b: &Expr, cols: &[Column], p: usize) -> Option<Expr> {
    match type_of(b, cols).as_deref() {
        Some(kind @ ("geohash" | "geotile" | "geohex")) => {
            Some(Expr::Call(super::func2::GRID_LONG.into(), vec![b.clone(), Expr::Lit(Val::Str(kind.into()))], p))
        }
        _ => None,
    }
}

/// `ImplicitCasting.castStringLiteral`: a string literal argument as the
/// parameter's target type.
fn cast_string_literal(s: &str, target: &str) -> std::result::Result<Expr, String> {
    let num = |s: &str| s.trim().parse::<f64>().ok().filter(|d| d.is_finite() || s.trim().eq_ignore_ascii_case("nan") || s.trim().to_ascii_lowercase().contains("inf"));
    Ok(match target {
        "double" => Expr::Lit(Val::Double(num(s).ok_or_else(|| format!("Cannot parse number [{s}]"))?)),
        "long" | "unsigned_long" => {
            let v = match s.trim().parse::<i64>() {
                Ok(i) => i,
                Err(_) => {
                    let d = num(s).ok_or_else(|| format!("Cannot parse number [{s}]"))?;
                    if d.fract() != 0.0 || d.abs() > 9.2e18 {
                        return Err(format!("Cannot parse number [{s}]"));
                    }
                    d as i64
                }
            };
            Expr::Lit(Val::Long(v))
        }
        "integer" => {
            let v = match s.trim().parse::<i64>() {
                Ok(i) => i,
                Err(_) => {
                    let d = num(s).ok_or_else(|| format!("Cannot parse number [{s}]"))?;
                    if d.fract() != 0.0 {
                        return Err(format!("Cannot parse number [{s}]"));
                    }
                    d as i64
                }
            };
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err(format!("[{s}] out of [integer] range"));
            }
            Expr::Lit(Val::Int(v))
        }
        "date" => Expr::Lit(Val::Date(func::parse_date_str(s).ok_or_else(|| format!("failed to parse date field [{s}] with format [strict_date_optional_time]"))?)),
        "date_period" | "time_duration" => match func::parse_temporal(s) {
            Some(v) => Expr::Lit(v),
            None => return Err(format!("Cannot parse [{s}] to {}", if target == "date_period" { "DATE_PERIOD" } else { "TIME_DURATION" })),
        },
        "boolean" => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Expr::Lit(Val::Bool(true)),
            "false" => Expr::Lit(Val::Bool(false)),
            _ => return Err(format!("cannot cast [{s}] to [boolean]")),
        },
        "ip" => {
            if func::ip_to_bytes(s).is_none() {
                return Err(format!("'{s}' is not an IP string literal."));
            }
            Expr::Cast(Box::new(Expr::Lit(Val::Str(s.to_string()))), "ip".into())
        }
        "version" | "geo_point" | "geo_shape" | "cartesian_point" | "cartesian_shape" => Expr::Cast(Box::new(Expr::Lit(Val::Str(s.to_string()))), target.to_string()),
        _ => Expr::Lit(Val::Str(s.to_string())),
    })
}

/// ES|QL `DataType` name for "Cannot convert string" messages.
fn cast_target_upper(t: &str) -> String {
    match t {
        "date" => "DATETIME".into(),
        t => t.to_ascii_uppercase(),
    }
}

/// Argument type resolution of a known function.
fn check_args(ctx: &Ctx, d: &super::typecheck::FnDef, name: &str, uargs: &[Expr], bargs: &mut [Expr], cols: &[Column], p: usize) -> Result<()> {
    use super::typecheck as tc;
    let srcs = call_arg_sources(ctx.full, p);
    let arg_src = |i: usize| -> String { srcs.get(i).map(|(_, t)| t.clone()).unwrap_or_default() };
    let arg_pos = |i: usize| -> usize { srcs.get(i).map(|(o, _)| *o).unwrap_or(p) };
    // String literals cast to what the parameter takes (scalars and groupings only).
    if matches!(d.kind.as_str(), "scalar" | "grouping") {
        for i in 0..bargs.len() {
            let Expr::Lit(Val::Str(s)) = &bargs[i] else { continue };
            if !matches!(uargs.get(i), Some(Expr::Lit(_))) {
                continue;
            }
            let Some(t) = tc::string_cast_target(d, i) else { continue };
            match cast_string_literal(s, t) {
                Ok(e) => bargs[i] = e,
                Err(m) => return Err(verification_err(ctx.full, arg_pos(i), format!("Cannot convert string [{s}] to [{}], error [{m}]", cast_target_upper(t)))),
            }
        }
    }
    let types: Vec<String> = bargs.iter().map(|b| type_of(b, cols).unwrap_or_else(|| "null".into())).collect();
    let src = call_src(ctx.full, p);
    let type_err = |i: usize, phrase: &str| {
        let ord = tc::ordinal(d, i);
        let ord = if ord.is_empty() { String::new() } else { format!("{ord} ") };
        verification_err(ctx.full, p, format!("{ord}argument of [{src}] must be [{phrase}], found value [{}] type [{}]", arg_src(i), type_label(&types[i])))
    };
    // Constant paths are parsed once, when the evaluator is built.
    if let ("JSON_EXTRACT" | "FIELD_EXTRACT", Some(Expr::Lit(Val::Str(path)))) = (name, bargs.get(1)) {
        if name == "FIELD_EXTRACT" {
            if let Some(m) = super::func2::field_extract_path_error(path) {
                return Err(verification_err(ctx.full, p, m.to_string()));
            }
        } else if let Err(m) = super::func2::parse_json_path(path) {
            return Err(EsError::illegal_argument(m));
        }
    }
    if matches!(name, "CLAMP" | "CLAMP_MIN" | "CLAMP_MAX") && types.first().map(String::as_str) == Some("null") {
        return Err(verification_err(ctx.full, p, "'field' must not be null in clamp()".to_string()));
    }
    // Parameters that must be constants.
    let constant_params: &[usize] = match name {
        "TOP_SNIPPETS" => &[1],
        "ROUND_TO" => &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        _ => &[],
    };
    for &i in constant_params {
        if let Some(b) = bargs.get(i) {
            if !foldable(b) {
                return Err(verification_err(ctx.full, arg_pos(i), format!("{} argument of [{src}] must be a constant, received [{}]", tc::ordinal_word(i), arg_src(i))));
            }
        }
    }
    match name {
        // Full-text functions resolve their own arguments (constant queries first).
        "MATCH" | "MATCH_OP" | "MATCH_PHRASE" | "MULTI_MATCH" | "QSTR" | "KQL" | "KNN" => return Ok(()),
        "CASE" => {
            let n = bargs.len();
            let mut i = 0;
            while i + 1 < n {
                if types[i] != "boolean" && types[i] != "null" {
                    return Err(verification_err(
                        ctx.full,
                        p,
                        format!("{} argument of [{src}] must be [boolean], found value [{}] type [{}]", tc::ordinal_word(i), arg_src(i), type_label(&types[i])),
                    ));
                }
                i += 2;
            }
            return Ok(());
        }
        "COALESCE" | "GREATEST" | "LEAST" | "MV_APPEND" => {
            let first = types.iter().find(|t| *t != "null").cloned();
            if let Some(f) = first {
                let fam = |t: &str| if t == "text" { "keyword".to_string() } else { t.to_string() };
                for i in 0..types.len() {
                    if types[i] != "null" && fam(&types[i]) != fam(&f) {
                        return Err(type_err(i, &type_label(&fam(&f))));
                    }
                }
            }
            if !matches!(name, "MV_APPEND") {
                if let Some(i) = (0..types.len()).find(|i| !types[*i].starts_with("unsupported") && !tc::param_accepts(d, *i, &types[*i])) {
                    return Err(type_err(i, &tc::phrase(d, i)));
                }
            }
            return Ok(());
        }
        _ => {}
    }
    for i in 0..bargs.len() {
        if types[i].starts_with("unsupported") || matches!(bargs[i], Expr::Map(_)) {
            continue;
        }
        // Pre-aggregated metrics feed these aggregations through their average.
        if i == 0 && types[i] == "aggregate_metric_double" && matches!(name, "STD_DEV" | "VARIANCE" | "MEDIAN" | "MEDIAN_ABSOLUTE_DEVIATION" | "PERCENTILE") {
            continue;
        }
        if !tc::param_accepts_n(d, i, &types[i], Some(bargs.len())) {
            return Err(type_err(i, &tc::phrase(d, i)));
        }
    }
    // `StSimplify.toEvaluator` / `getInputTolerance`: the tolerance is folded
    // when the evaluator is built; IllegalArgumentExceptions there fail the query.
    if name == "ST_SIMPLIFY" {
        if let Some(t) = bargs.get(1) {
            if !foldable(t) {
                return Err(EsError::illegal_argument("tolerance must be foldable".to_string()));
            }
            let value = match t {
                Expr::Lit(v) => v.as_f64(),
                Expr::Neg(inner) => match inner.as_ref() {
                    Expr::Lit(v) => v.as_f64().map(|x| -x),
                    _ => None,
                },
                _ => None,
            };
            if value.is_some_and(|x| x < 0.0) {
                return Err(EsError::illegal_argument("tolerance must not be negative".to_string()));
            }
        }
    }
    license_check(ctx, name, &types, &src, p)
}

/// The license an ES|QL function needs on its argument types (ES
/// `LicenseAware.licenseCheck`): `ST_GEOHEX` always needs platinum
/// (`StGeohex`); `ST_GEOHASH` / `ST_GEOTILE` (`SpatialGridFunction`) and the
/// spatial aggregations (`SpatialAggregateFunction`) only over shapes;
/// `CATEGORIZE` is the platinum machine-learning feature.
pub(crate) fn required_license(name: &str, types: &[String]) -> Option<crate::license::Mode> {
    use crate::license::Mode;
    let shape = types.first().is_some_and(|t| matches!(t.as_str(), "geo_shape" | "cartesian_shape"));
    match name {
        "ST_GEOHEX" | "CATEGORIZE" => Some(Mode::Platinum),
        "ST_GEOHASH" | "ST_GEOTILE" | "ST_CENTROID_AGG" | "ST_EXTENT_AGG" if shape => Some(Mode::Platinum),
        _ => None,
    }
}

/// ES `Verifier.licenseCheck`: `current license is non-compliant for
/// [<source>]` unless the cluster license allows the function
/// (`XPackLicenseState.isAllowedByLicense`: an active trial, or an active
/// license of at least the required mode; an expired one counts as basic).
fn license_check(ctx: &Ctx, name: &str, types: &[String], src: &str, p: usize) -> Result<()> {
    let (Some(node), true) = (ctx.node, ctx.esql) else { return Ok(()) };
    let Some(min) = required_license(name, types) else { return Ok(()) };
    if crate::license::current(node).effective_mode(crate::date::now_millis()).allows(min) {
        return Ok(());
    }
    Err(verification_err(ctx.full, p, format!("current license is non-compliant for [{src}]")))
}

/// `Line l:c: ` of a query offset, as compute-engine warnings print it.
fn warning_location(src: &str, off: usize) -> String {
    let off = off.min(src.len());
    let before = &src[..off];
    let line = before.matches('\n').count() + 1;
    let col = before.rfind('\n').map(|p| off - p).unwrap_or(off + 1);
    format!("Line {line}:{col}: ")
}

/// The source text (offset, text) of each argument of the call at `pos`:
/// the top-level comma-separated segments between its parentheses.
fn call_arg_sources(src: &str, pos: usize) -> Vec<(usize, String)> {
    let b = src.as_bytes();
    let Some(open) = src.get(pos..).and_then(|r| r.find('(')).map(|o| pos + o) else { return vec![] };
    let mut out = Vec::new();
    let (mut depth, mut start, mut i) = (0usize, open + 1, open + 1);
    let push = |a: usize, z: usize, out: &mut Vec<(usize, String)>| {
        let seg = &src[a..z];
        let lead = seg.len() - seg.trim_start().len();
        out.push((a + lead, seg.trim().to_string()));
    };
    while i < b.len() {
        match b[i] {
            b'"' | b'`' | b'\'' => {
                let q = b[i];
                // Triple-quoted strings.
                if q == b'"' && src[i..].starts_with("\"\"\"") {
                    i = src[i + 3..].find("\"\"\"").map(|e| i + 3 + e + 3).unwrap_or(b.len());
                    continue;
                }
                i += 1;
                while i < b.len() && b[i] != q {
                    if b[i] == b'\\' && q == b'"' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'(' | b'[' => depth += 1,
            b')' | b']' if depth > 0 => depth -= 1,
            b')' => {
                push(start, i, &mut out);
                return out;
            }
            b',' if depth == 0 => {
                push(start, i, &mut out);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// Remember where each `CASE` condition comes from, for the warnings a
/// multi-valued condition raises. A union-typed field converted in place
/// reports as the field itself (ES replaces the conversion by the field).
fn record_case_sources(ctx: &Ctx, args: &[Expr], pos: usize, cols: &[Column]) {
    let srcs = call_arg_sources(ctx.full, pos);
    let mut conds = Vec::new();
    let mut i = 0;
    while i + 1 < args.len() {
        let converted = match &args[i] {
            Expr::Cast(a, t) => bind_converted(ctx, cast_fn(t), a, cols).is_some_and(|r| r.is_ok()).then(|| a.as_ref()),
            Expr::Call(n, a, _) if a.len() == 1 && func::is_conversion(n) => bind_converted(ctx, n, &a[0], cols).is_some_and(|r| r.is_ok()).then(|| &a[0]),
            _ => None,
        };
        let (off, text) = match (converted, srcs.get(i)) {
            (Some(Expr::Col(n, p)), _) => (*p, n.clone()),
            (_, Some((o, t))) => (*o, t.clone()),
            _ => (expr_pos(&args[i]).unwrap_or(pos), String::new()),
        };
        conds.push((warning_location(ctx.full, off), text));
        i += 2;
    }
    ctx.fctx.case_conds.borrow_mut().insert(pos, conds);
}

/// Static type of a bound expression.
pub fn type_of(e: &Expr, cols: &[Column]) -> Option<String> {
    Some(match e {
        Expr::Idx(i) => cols[*i].ty.clone(),
        Expr::Lit(v) => v.type_name().to_string(),
        Expr::Bin(op, a, b) => match op {
            BinOp::And | BinOp::Or | BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => "boolean".into(),
            _ => {
                let (ta, tb) = (type_of(a, cols)?, type_of(b, cols)?);
                // Date +/- temporal amount is a date.
                let temporal = |t: &str| t == "date_period" || t == "time_duration";
                if (ta == "date" || ta == "date_nanos") && temporal(&tb) {
                    return Some(ta);
                }
                if (tb == "date" || tb == "date_nanos") && temporal(&ta) {
                    return Some(tb);
                }
                // A null operand takes the other operand's type.
                if ta == "null" {
                    return Some(tb);
                }
                if tb == "null" {
                    return Some(ta);
                }
                if ta.starts_with("date") && !tb.starts_with("date") {
                    return Some(ta);
                }
                if tb.starts_with("date") {
                    return Some(tb);
                }
                let rank = |t: &str| match t {
                    "integer" => 1,
                    "long" => 2,
                    "unsigned_long" => 3,
                    "double" => 4,
                    _ => 0,
                };
                if rank(&ta) == 0 || rank(&tb) == 0 {
                    return if ta == tb { Some(ta) } else { None };
                }
                if rank(&ta) >= rank(&tb) { ta } else { tb }
            }
        },
        Expr::Neg(a) => type_of(a, cols)?,
        Expr::Not(_) | Expr::IsNull(..) | Expr::In(..) | Expr::Like(..) | Expr::Between(..) => "boolean".into(),
        Expr::Cast(_, t) => func::cast_type(t).to_string(),
        Expr::Call(name, args, _) => {
            if FULL_TEXT.contains(&name.as_str()) {
                return Some("boolean".into());
            }
            let ats: Vec<String> = args.iter().map(|a| type_of(a, cols).unwrap_or_else(|| "null".into())).collect();
            func::return_type(name, &ats)?
        }
        _ => return None,
    })
}

fn infer_type(static_ty: Option<String>, vals: &[Val]) -> String {
    if let Some(t) = static_ty {
        if t != "null" {
            return t;
        }
    }
    for v in vals {
        if !v.is_null() {
            return v.type_name().to_string();
        }
    }
    "null".into()
}

// --------------------------------------------------------------- evaluation

fn like_match(s:&str,pat:&str,rlike:bool,f:&FnCtx)->Result<bool> {
    // A LIKE pattern of a literal with `*` only at its ends is an equality,
    // prefix, suffix or substring test: the regex it translates to answers
    // the same, after building its text (and a cache lookup) per row.
    if !rlike && !pat.contains(['?','\\']) {
        let core=pat.trim_matches('*');
        if !core.contains('*') {
            let (lead,trail)=(pat.starts_with('*'),pat.ends_with('*'));
            return Ok(if core.is_empty() { lead || trail || s.is_empty() } else { match (lead,trail) {
                (false,false)=>s==core,(false,true)=>s.starts_with(core),(true,false)=>s.ends_with(core),(true,true)=>s.contains(core),
            } });
        }
    }
    let _translation=crate::memory::breakers().reserve_request((pat.len() as u64).saturating_mul(16).saturating_add(256),ESQL_LABEL)?;
    let re=if rlike {format!("^(?:{pat})$")} else {
        let mut re=String::from("^");let mut chars=pat.chars();
        while let Some(ch)=chars.next() {
            match ch {
                '*'=>re.push_str("(?s:.*)"),
                '?'=>re.push_str("(?s:.)"),
                '\\'=>if let Some(n)=chars.next(){re.push_str(&regex::escape(&n.to_string()));},
                ch=>re.push_str(&regex::escape(&ch.to_string())),
            }
        }
        re.push('$');re
    };
    match f.regex_cache.borrow_mut().get(&re)? {Some(re)=>re.is_match(s),None=>Ok(false)}
}
fn like_any(s:&str,pats:&[String],rlike:bool,f:&FnCtx)->Result<bool> {
    for pattern in pats {if like_match(s,pattern,rlike,f)? {return Ok(true);}}
    Ok(false)
}

#[cfg(test)]
mod like_fast_tests {
    use super::*;

    /// The literal-with-end-stars shortcut answers as the translated regex
    /// does, for patterns and strings around its edges.
    #[test]
    fn like_shortcut_matches_regex() {
        let f = FnCtx::default();
        let pats = ["", "*", "**", "a", "a*", "*a", "*a*", "ab*", "**ab", "a*b", "a?", "a\\*", "val 1*", "*\n*", "é*"];
        let strs = ["", "a", "ab", "ba", "aab", "abb", "val 1", "val 10 rep 0", "xval 1", "\n", "a\nb", "é", "éa"];
        for p in pats {
            let mut re = String::from("^");
            let mut chars = p.chars();
            while let Some(ch) = chars.next() {
                match ch {
                    '*' => re.push_str("(?s:.*)"),
                    '?' => re.push_str("(?s:.)"),
                    '\\' => if let Some(n) = chars.next() { re.push_str(&regex::escape(&n.to_string())); },
                    ch => re.push_str(&regex::escape(&ch.to_string())),
                }
            }
            re.push('$');
            let re = regex::Regex::new(&re).unwrap();
            for s in strs {
                assert_eq!(like_match(s, p, false, &f).unwrap(), re.is_match(s), "{p:?} {s:?}");
            }
        }
    }
}

/// Own the expression's intermediate budget until its result has been
/// consumed. Callers retaining a column pass their output guard instead.
pub fn eval(e:&Expr,row:&[Val],f:&FnCtx)->Result<Val> {
    let mut memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
    eval_reserved(e,row,f,&mut memory)
}

fn eval_reserved(e:&Expr,row:&[Val],f:&FnCtx,memory:&mut crate::memory::Reservation)->Result<Val> {
    let before=memory.bytes();
    let value=eval_budget(e,row,f,memory)?;
    let retained=before.saturating_add((value.heap_bytes() as u64).saturating_mul(2)).saturating_add(64);
    if retained>memory.bytes(){memory.grow(retained-memory.bytes(),ESQL_LABEL)?;}
    else{memory.shrink(memory.bytes()-retained);}
    Ok(value)
}

/// Input-sized scalar operations have bounded copy/encoding expansion.
/// Operations whose output size is supplied by a numeric argument need a
/// separate bound: input bytes alone cannot admit REPEAT or SPACE safely.
fn scalar_text_bound(value:&Val)->Option<u64> {
    match value {
        Val::Null|Val::Multi(_)=>None,
        Val::Str(s)=>Some(s.len() as u64),
        // Numeric/date/period scalar formatting is bounded independently of
        // the supplied repeat count; align with func::s_arg coercion.
        _=>Some(128),
    }
}
fn expression_call_bytes(name:&str,args:&[Val])->u64 {
    let base=args.iter().fold(512u64,|sum,value|sum.saturating_add((value.heap_bytes() as u64).saturating_mul(32)).saturating_add(256));
    let extra=match name {
        "REPEAT"=>match (args.first().and_then(scalar_text_bound),args.get(1).and_then(Val::as_i64)) {
            (Some(len),Some(count)) if count>=0=>len.saturating_mul(count as u64).saturating_mul(2),_=>0,
        },
        "SPACE"=>args.first().and_then(Val::as_i64).filter(|n|*n>=0).map_or(0,|n|(n as u64).saturating_mul(2)),
        "REPLACE"=>match (args.first().and_then(scalar_text_bound),args.get(2).and_then(scalar_text_bound)) {
            (Some(len),Some(replacement))=>len.saturating_add(1).saturating_mul(replacement).saturating_mul(2),_=>0,
        },
        _=>0,
    };
    base.saturating_add(extra)
}

fn eval_budget(e: &Expr, row: &[Val], f: &FnCtx, memory:&mut crate::memory::Reservation) -> Result<Val> {
    Ok(match e {
        Expr::Lit(v) => { memory.grow((v.heap_bytes() as u64).saturating_mul(2).saturating_add(64),ESQL_LABEL)?;v.clone() },
        Expr::Idx(i) => {let value=row.get(*i);memory.grow(value.map_or(0,|v|v.heap_bytes() as u64).saturating_mul(2).saturating_add(64),ESQL_LABEL)?;value.cloned().unwrap_or(Val::Null)},
        Expr::Bin(op, a, b) => {
            let x = eval_budget(a, row, f, memory)?;
            if *op == BinOp::And && x == Val::Bool(false) {
                return Ok(Val::Bool(false));
            }
            if *op == BinOp::Or && x == Val::Bool(true) {
                return Ok(Val::Bool(true));
            }
            func::binop(*op, &x, &eval_budget(b, row, f, memory)?)
        }
        Expr::Neg(a) => match eval_budget(a, row, f, memory)? {
            Val::Int(i) => Val::Int(-i),
            Val::Long(i) => Val::Long(-i),
            Val::Double(d) => Val::Double(-d),
            Val::Dur(d) => Val::Dur(-d),
            Val::Period(m, d) => Val::Period(-m, -d),
            _ => Val::Null,
        },
        Expr::Not(a) => match eval_budget(a, row, f, memory)? {
            Val::Bool(b) => Val::Bool(!b),
            _ => Val::Null,
        },
        Expr::IsNull(a, not) => Val::Bool(eval_budget(a, row, f, memory)?.is_null() != *not),
        Expr::In(a, list, neg) => {
            let v = eval_budget(a, row, f, memory)?;
            if v.is_null() || matches!(v, Val::Multi(_)) {
                return Ok(Val::Null);
            }
            let mut saw_null = false;
            for x in list {
                let xv = eval_budget(x, row, f, memory)?;
                if xv.is_null() {
                    saw_null = true;
                    continue;
                }
                if func::compare(&v, &xv) == Some(Ordering::Equal) {
                    return Ok(Val::Bool(!*neg));
                }
            }
            if saw_null { Val::Null } else { Val::Bool(*neg) }
        }
        Expr::Like(a, pats, neg, rlike) => match eval_budget(a, row, f, memory)? {
            Val::Str(s) => Val::Bool(like_any(&s,pats,*rlike,f)? != *neg),
            Val::Null | Val::Multi(_) => Val::Null,
            other => Val::Bool(like_any(&other.to_text(),pats,*rlike,f)? != *neg),
        },
        Expr::Between(a, lo, hi, neg) => {
            let v = eval_budget(a, row, f, memory)?;
            let l = func::binop(BinOp::Ge, &v, &eval_budget(lo, row, f, memory)?);
            let h = func::binop(BinOp::Le, &v, &eval_budget(hi, row, f, memory)?);
            match func::binop(BinOp::And, &l, &h) {
                Val::Bool(b) => Val::Bool(b != *neg),
                o => o,
            }
        }
        Expr::Call(name, args, pos) => {
            if name == MATCH_RT {
                return match_rt_eval(args, row, f);
            }
            if FULL_TEXT.contains(&name.as_str()) || name == "SCORE" {
                return Err(EsError::new(
                    400,
                    "verification_exception",
                    format!("Found 1 problem\n[{name}] function is only supported in WHERE commands directly after FROM (it must be pushed down to the index)"),
                ));
            }
            if name == "CASE" {
                // Lazy: only the chosen branch is evaluated. A multi-valued
                // condition counts as false and warns.
                let mut i = 0;
                while i + 1 < args.len() {
                    match eval_budget(&args[i], row, f, memory)? {
                        Val::Bool(true) => return eval_budget(&args[i + 1], row, f, memory),
                        Val::Multi(_) => {
                            let conds = f.case_conds.borrow();
                            let (loc, text) = conds.get(pos).and_then(|c| c.get(i / 2)).cloned().unwrap_or_default();
                            f.warnings.borrow_mut().register(&loc, &format!("evaluation of [{text}] failed, treating result as false"), "java.lang.IllegalArgumentException: CASE expects a single-valued boolean");
                        }
                        _ => {}
                    }
                    i += 2;
                }
                return if args.len() % 2 == 1 { eval_budget(&args[args.len() - 1], row, f, memory) } else { Ok(Val::Null) };
            }
            memory.grow((args.len() as u64).saturating_mul(128),ESQL_LABEL)?;
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(match a {
                    // Function options reach the function as their JSON.
                    Expr::Map(m) => {
                        memory.grow(m.iter().fold(64u64,|sum,(key,value)|sum.saturating_add((key.len() as u64).saturating_add(value.heap_bytes() as u64).saturating_add(128))).saturating_mul(32),ESQL_LABEL)?;
                        Val::Str(Value::Object(m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()).to_string())
                    },
                    a => eval_budget(a, row, f, memory)?,
                });
            }
            memory.grow(expression_call_bytes(name,&vals),ESQL_LABEL)?;
            match func::call(name, &vals, f) {
                Ok(v) => v,
                Err(e) if e.kind == func::SRC_ERR => return Err(EsError::illegal_argument(e.reason.replace("{src}", &call_src(&f.query, *pos)))),
                Err(e) if e.kind == func::EVAL_WARN => {
                    let loc = warning_location(&f.query, *pos);
                    let src = call_src(&f.query, *pos);
                    f.warnings.borrow_mut().register(&loc, &format!("evaluation of [{src}] failed, treating result as null"), &e.reason);
                    Val::Null
                }
                Err(e) => return Err(e),
            }
        }
        Expr::Cast(a, t) => {let value=eval_budget(a,row,f,memory)?;memory.grow((value.heap_bytes() as u64).saturating_add(64).saturating_mul(32),ESQL_LABEL)?;func::cast(&value,t)?},
        Expr::Map(_) | Expr::Star => Val::Null,
        Expr::Col(n, _) => return Err(EsError::internal(format!("unbound column [{n}]"))),
        Expr::AggRef(_) => return Err(EsError::internal("unbound aggregate")),
    })
}

fn eval_col(ctx: &Ctx, t: &Table, f: &Field) -> Result<(String, Vec<Val>,crate::memory::Reservation)> {
    let b = bind(ctx, &f.expr, &t.cols)?;
    let st = type_of(&b, &t.cols);
    let slots=if t.idx(&f.name).is_none(){t.rows.iter().fold(0u64,|sum,row|sum.saturating_add((row.capacity().max(row.len()+1) as u64).saturating_mul(std::mem::size_of::<Val>() as u64).saturating_mul(2)))}else{0};
    let mut memory=crate::memory::breakers().reserve_request((t.rows.len() as u64).saturating_mul(128).saturating_add(slots),ESQL_LABEL)?;
    let mut vals = Vec::with_capacity(t.rows.len());
    for r in &t.rows {
        vals.push(eval_reserved(&b,r,&ctx.fctx,&mut memory)?);
    }
    Ok((infer_type(st, &vals), vals,memory))
}

// ----------------------------------------------------------------- pipeline

/// ES's `PropagateEvalFoldables` and the `EVAL` half of
/// `PushDownAndCombineFilters`, over the commands after `FROM`: a column an
/// `EVAL` defines as a constant (no column references: `start =
/// DATE_PARSE(fmt, "2023-05-16T20:45:00.000Z")`) is replaced by its value in
/// the `WHERE`s that follow it (through `SORT` / `LIMIT`, until anything
/// else, or an `EVAL` redefining it), and a `WHERE` that reads no column of
/// the `EVAL` just before it moves in front of it (the `EVAL` still runs, on
/// fewer rows). Filters then reach `FROM`'s index pushdown instead of
/// running per row after every `EVAL`. `None`: nothing changed.
pub(super) fn push_filters_past_evals(f: &FnCtx, cmds: &[Cmd]) -> Option<Vec<Cmd>> {
    let mut out: Vec<Cmd> = cmds.to_vec();
    let mut changed = false;
    for i in 0..out.len() {
        let Cmd::Eval(fields) = &out[i] else { continue };
        let mut consts: Vec<(String, Expr)> = Vec::new();
        for fd in fields {
            let e = fast::subst(&fd.expr, &consts);
            consts.retain(|(n, _)| *n != fd.name);
            if foldable(&e) && !matches!(&e, Expr::Map(_)) {
                if let Ok(v) = eval(&e, &[], f) {
                    consts.push((fd.name.clone(), Expr::Lit(v)));
                }
            }
        }
        if consts.is_empty() {
            continue;
        }
        for cmd in out.iter_mut().skip(i + 1) {
            match cmd {
                Cmd::Where(e, _) => {
                    let ne = fast::subst(e, &consts);
                    if ne != *e {
                        *e = ne;
                        changed = true;
                    }
                }
                Cmd::Sort(_) | Cmd::Limit(_) => {}
                Cmd::Eval(fs) => {
                    consts.retain(|(n, _)| !fs.iter().any(|fd| fd.name == *n));
                    if consts.is_empty() {
                        break;
                    }
                }
                _ => break,
            }
        }
    }
    fn reads(e: &Expr, names: &[&str]) -> bool {
        match e {
            Expr::Col(n, _) => names.contains(&n.as_str()),
            Expr::Idx(_) | Expr::AggRef(_) | Expr::Star => true,
            Expr::Lit(_) | Expr::Map(_) => false,
            Expr::Bin(_, a, b) => reads(a, names) || reads(b, names),
            Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => reads(a, names),
            Expr::In(a, l, _) => reads(a, names) || l.iter().any(|x| reads(x, names)),
            Expr::Between(a, b, c, _) => reads(a, names) || reads(b, names) || reads(c, names),
            Expr::Call(_, args, _) => args.iter().any(|x| reads(x, names)),
        }
    }
    fn conjuncts(e: &Expr, out: &mut Vec<Expr>) {
        match e {
            Expr::Bin(BinOp::And, a, b) => {
                conjuncts(a, out);
                conjuncts(b, out);
            }
            o => out.push(o.clone()),
        }
    }
    let and_all = |es: Vec<Expr>| es.into_iter().reduce(|a, b| Expr::Bin(BinOp::And, Box::new(a), Box::new(b)));
    loop {
        let mut moved = false;
        for j in 1..out.len() {
            let (Cmd::Eval(fields), Cmd::Where(e, pos)) = (&out[j - 1], &out[j]) else { continue };
            let defined: Vec<&str> = fields.iter().map(|fd| fd.name.as_str()).collect();
            // The conjuncts that read no column of the EVAL move; the
            // others stay after it.
            let mut parts = Vec::new();
            conjuncts(e, &mut parts);
            let (free, bound): (Vec<Expr>, Vec<Expr>) = parts.into_iter().partition(|c| !reads(c, &defined));
            let Some(free) = and_all(free) else { continue };
            let pos = *pos;
            let eval = out[j - 1].clone();
            let mut replacement = vec![Cmd::Where(free, pos), eval];
            if let Some(bound) = and_all(bound) {
                replacement.push(Cmd::Where(bound, pos));
            }
            out.splice(j - 1..=j, replacement);
            moved = true;
            changed = true;
            break;
        }
        if !moved {
            break;
        }
    }
    changed.then_some(out)
}

fn exec_pipeline(ctx: &mut Ctx, cmds: &[Cmd], input: Option<Table>) -> Result<Table> {
    if input.is_none() {
        if let Some(k) = cmds.iter().position(|c| matches!(c, Cmd::Info(..))) {
            return info_pipeline(ctx, cmds, k);
        }
    }
    let mut i = 0;
    let mut t = match input {
        Some(t) => t,
        None => {
            i = 1;
            match &cmds[0] {
                Cmd::From { indices, metadata, pos, ts, subqueries } => {
                    ctx.ts = *ts;
                    // Constant EVALs folded into the WHEREs after them, and
                    // those WHEREs moved in front of the EVALs: they reach
                    // the index pushdown below.
                    if subqueries.is_empty() {
                        if let Some(rewritten) = push_filters_past_evals(&ctx.fctx, &cmds[1..]) {
                            let mut all = vec![cmds[0].clone()];
                            all.extend(rewritten);
                            return exec_pipeline(ctx, &all, None);
                        }
                    }
                    // Views and subqueries in FROM: the union of their results and the indices'.
                    if let Some(t) = from_union(ctx, indices, metadata, *pos, subqueries, &cmds[1..])? {
                        return Ok(t);
                    }
                    // TS without SORT or STATS lists the newest points first.
                    let implicit_sort = *ts && !cmds[1..].iter().any(|c| matches!(c, Cmd::Sort(_) | Cmd::Stats { .. }));
                    if implicit_sort {
                        let nw = cmds[1..].iter().take_while(|c| matches!(c, Cmd::Where(..))).count();
                        let mut rest: Vec<Cmd> = cmds[1..1 + nw].to_vec();
                        rest.push(Cmd::Sort(vec![SortKey { expr: Expr::Col("@timestamp".into(), *pos), desc: true, nulls_first: None }]));
                        rest.extend(cmds[1 + nw..].iter().cloned());
                        let (mut t, consumed, hoisted) = from_with_pushdown(ctx, indices, metadata, *pos, &rest)?;
                        for c in &hoisted.as_deref().unwrap_or(&rest)[consumed..] {
                            t = apply(ctx, c, t)?;
                        }
                        return Ok(t);
                    }
                    let (t, consumed, hoisted) = from_with_pushdown(ctx, indices, metadata, *pos, &cmds[1..])?;
                    if let Some(rest) = hoisted {
                        // Filters moved in front of EVALs, or joins given their
                        // lookup columns: run what is left of the rewrite.
                        return exec_pipeline(ctx, &rest[consumed..], Some(t));
                    }
                    i += consumed;
                    t
                }
                Cmd::Row(fields) => {
                    ctx.fulltext_barrier = Some("ROW");
                    let mut t = Table { cols: vec![], rows: vec![vec![]] };
                    for f in fields {
                        let (ty, vals, memory) = eval_col(ctx, &t, f)?;
                        t.put_col(&f.name, ty, vals);
                        account_table(ctx,&t)?;
                        drop(memory);
                    }
                    // `unmapped_fields`: unresolved names read as null.
                    let scope: Vec<String> = t.cols.iter().map(|c| c.name.clone()).collect();
                    for n in unmapped_refs(ctx, &cmds[1..], scope) {
                        t.put_col(&n, "null".into(), vec![Val::Null]);
                    }
                    t
                }
                Cmd::Promql(q) => {
                    ctx.fulltext_barrier = Some("PROMQL");
                    super::promql::run(ctx, q)?
                }
                Cmd::ShowInfo => Table {
                    cols: vec![
                        Column { name: "version".into(), ty: "keyword".into() },
                        Column { name: "date".into(), ty: "keyword".into() },
                        Column { name: "hash".into(), ty: "keyword".into() },
                    ],
                    rows: vec![vec![Val::Str(crate::version::ES_VERSION.into()), Val::Str("2025-01-01T00:00:00Z".into()), Val::Str("rustic".into())]],
                },
                _ => return Err(EsError::parsing("query must start with a source command")),
            }
        }
    };
    while i < cmds.len() {
        account_table(ctx, &t)?;
        if let Cmd::LookupJoin { index, index_pos, cond, on, on_pos, pos, fields } = &cmds[i] {
            if let Some((filters, limit)) = filtered_join_limit(&cmds[i + 1..]) {
                track_fields(ctx, &cmds[i], &t);
                t = match cond {
                    Some(expr) => lookup_join_expr(ctx, t, index, *index_pos, expr, Some(limit), &filters, fields.as_ref())?,
                    None => lookup_join(ctx, t, index, *index_pos, on, on_pos, Some(limit), &filters, *pos, fields.as_ref())?,
                };
                // WHERE was evaluated once as rows were produced. LIMIT and
                // subsequent projection commands still execute normally.
                i += filters.len() + 1;
                continue;
            }
        }
        if let Some((counted, consumed)) = lookup_count(ctx, &t, &cmds[i], &cmds[i + 1..])? {
            t = counted;
            i += consumed + 1;
            continue;
        }
        t = apply_with_tail(ctx, &cmds[i], t, &cmds[i + 1..])?;
        i += 1;
    }
    Ok(t)
}

/// Full-text calls (name, args, offset) in an expression.
fn fulltext_calls<'e>(e: &'e Expr, out: &mut Vec<(&'e str, &'e [Expr], usize)>) {
    match e {
        Expr::Call(n, args, p) => {
            if FULL_TEXT.contains(&n.as_str()) {
                out.push((n.as_str(), args.as_slice(), *p));
            }
            args.iter().for_each(|a| fulltext_calls(a, out));
        }
        Expr::Bin(_, a, b) => {
            fulltext_calls(a, out);
            fulltext_calls(b, out);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) | Expr::In(a, ..) | Expr::Between(a, ..) => fulltext_calls(a, out),
        _ => {}
    }
}

/// Full-text functions outside `WHERE`: `... is only supported in WHERE and STATS commands`.
fn check_no_fulltext(ctx: &Ctx, fields: &[Field]) -> Result<()> {
    for f in fields {
        let mut calls = Vec::new();
        fulltext_calls(&f.expr, &mut calls);
        if let Some((n, _, p)) = calls.first() {
            return Err(verification_err(ctx.full, *p, format!("{} is only supported in WHERE and STATS commands", func::fulltext_label(n))));
        }
    }
    Ok(())
}

/// The error for a full-text `WHERE` that cannot be pushed to the index.
fn fulltext_where_err(ctx: &Ctx, e: &Expr, wpos: usize) -> EsError {
    let mut calls = Vec::new();
    fulltext_calls(e, &mut calls);
    for (n, args, _) in &calls {
        if matches!(*n, "MATCH" | "MATCH_OP" | "MATCH_PHRASE" | "TERM") {
            if let Some(Expr::Col(f, fp)) = args.first() {
                if !ctx.pristine.contains(f) {
                    // `MatchPhrase` keeps its class name as its function name.
                    let label = if *n == "MATCH_PHRASE" { "[MatchPhrase] function".to_string() } else { func::fulltext_label(n) };
                    return verification_err(ctx.full, *fp, format!("{label} cannot operate on [{f}], which is not a field from an index mapping"));
                }
            }
        }
    }
    let label = calls.first().map(|(n, _, _)| func::fulltext_label(n)).unwrap_or_default();
    if let (Some((n, _, _)), Some(b)) = (calls.iter().find(|(n, _, _)| matches!(*n, "KQL" | "QSTR")), ctx.lexical_barrier) {
        return verification_err(ctx.full, wpos, format!("{} cannot be used after {b}", func::fulltext_label(n)));
    }
    match ctx.fulltext_barrier {
        Some(b) => verification_err(ctx.full, wpos, format!("{label} cannot be used after {b}")),
        None => EsError::new(400, "verification_exception", format!("Found 1 problem\n{label} is only supported in WHERE commands directly after FROM (it must be pushed down to the index)")),
    }
}

/// Output columns of `METRICS_INFO` (`TS_INFO` adds `dimensions`).
pub const INFO_COLUMNS: &[&str] = &["metric_name", "data_stream", "unit", "metric_type", "field_type", "dimension_fields"];

/// `METRICS_INFO` / `TS_INFO` at `cmds[k]`: verified like ES, computed over
/// the documents reaching it (projections before it don't matter: ES reads
/// the documents themselves), then the rest of the pipeline runs on its rows.
fn info_pipeline(ctx: &mut Ctx, cmds: &[Cmd], k: usize) -> Result<Table> {
    let Cmd::Info(ts_info, pos) = &cmds[k] else { unreachable!() };
    let label = if *ts_info { "TS_INFO" } else { "METRICS_INFO" };
    let breaker = cmds[1..k].iter().rev().find_map(|c| match c {
        Cmd::Stats { .. } => Some("STATS"),
        Cmd::Limit(_) | Cmd::LimitBy(..) => Some("LIMIT"),
        Cmd::Sort(_) => Some("SORT"),
        _ => None,
    });
    if let Some(b) = breaker {
        return Err(verification_err(ctx.full, *pos, format!("{label} cannot be used after {b} command")));
    }
    let Some(Cmd::From { indices, metadata, pos: fpos, ts: true, subqueries }) = cmds.first() else {
        return Err(verification_err(ctx.full, *pos, format!("{label} can only be used with TS source command")));
    };
    let mut md = metadata.clone();
    if !md.iter().any(|m| m == "_index") {
        md.push("_index".into());
    }
    let mut prefix = vec![Cmd::From { indices: indices.clone(), metadata: md, pos: *fpos, ts: true, subqueries: subqueries.clone() }];
    prefix.extend(cmds[1..k].iter().filter(|c| !matches!(c, Cmd::Keep(..) | Cmd::Drop(..))).cloned());
    let docs = exec_pipeline(ctx, &prefix, None)?;
    let mut t = info_table(ctx, &docs, *ts_info);
    ctx.ts = false;
    ctx.ts_done = true;
    for c in &cmds[k + 1..] {
        t = apply(ctx, c, t)?;
    }
    Ok(t)
}

/// Iteration order of a `java.util.HashSet<String>` filled in `items` order.
fn java_hashset_order(items: &[String]) -> Vec<String> {
    let mut cap = 16usize;
    while items.len() > cap * 3 / 4 {
        cap *= 2;
    }
    let bucket = |s: &str| -> usize {
        let mut h: i32 = 0;
        for u in s.encode_utf16() {
            h = h.wrapping_mul(31).wrapping_add(u as i32);
        }
        let h = (h ^ ((h as u32) >> 16) as i32) as u32;
        (h as usize) & (cap - 1)
    };
    let mut v: Vec<(usize, usize, String)> = items.iter().enumerate().map(|(i, s)| (bucket(s), i, s.clone())).collect();
    v.sort();
    v.into_iter().map(|(_, _, s)| s).collect()
}

fn set_val(items: &[String]) -> Val {
    match items.len() {
        0 => Val::Null,
        1 => Val::Str(items[0].clone()),
        _ => Val::Multi(java_hashset_order(items).into_iter().map(Val::Str).collect()),
    }
}

fn push_unique(v: &mut Vec<String>, s: &str) {
    if !v.iter().any(|x| x == s) {
        v.push(s.to_string());
    }
}

/// `MetricsInfoOperator` / `TsInfoOperator` over the newest document of
/// each time series (per index).
fn info_table(ctx: &Ctx, docs: &Table, ts_info: bool) -> Table {
    struct Entry {
        name: String,
        ds: String,
        dims_json: String,
        units: Vec<String>,
        ftypes: Vec<String>,
        mtypes: Vec<String>,
        dims: Vec<String>,
    }
    let ii = docs.idx("_index");
    // Per index: (dimension fields, metric fields with (unit, field type, metric type)).
    let mut fields: HashMap<String, (Vec<String>, Vec<(String, Option<String>, String, String)>)> = HashMap::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut entries: Vec<Entry> = Vec::new();
    for r in &docs.rows {
        let Some(index) = ii.and_then(|i| match &r[i] {
            Val::Str(s) => Some(s.clone()),
            _ => None,
        }) else {
            continue;
        };
        let (dims, metrics) = fields.entry(index.clone()).or_insert_with(|| {
            let mut d = Vec::new();
            let mut m = Vec::new();
            if let Some(svc) = ctx.node.and_then(|n| n.get_index(&index)) {
                let lk = svc.lookup();
                let mut names: Vec<&String> = lk.fields.keys().collect();
                names.sort();
                for n in names {
                    let f = &lk.fields[n];
                    if f.time_series_dimension {
                        d.push(n.clone());
                    }
                    if let Some(mt) = &f.time_series_metric {
                        let unit = f.meta.as_ref().and_then(|m| m.get("unit")).and_then(Value::as_str).map(str::to_string);
                        m.push((n.clone(), unit, f.declared.name().to_string(), mt.clone()));
                    }
                }
            }
            (d, m)
        });
        let val_of = |n: &str| docs.idx(n).map(|i| &r[i]).filter(|v| !v.is_null());
        let mut dim_kv: BTreeMap<String, String> = BTreeMap::new();
        for d in dims.iter() {
            if let Some(v) = val_of(d) {
                dim_kv.insert(d.clone(), v.to_text());
            }
        }
        let dims_json = Value::Object(dim_kv.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect()).to_string();
        if !seen.insert((index.clone(), dims_json.clone())) {
            continue;
        }
        let ds = backing_data_stream(&index);
        let present: Vec<String> = dim_kv.keys().cloned().collect();
        for (name, unit, ft, mt) in metrics.iter() {
            if val_of(name).is_none() {
                continue;
            }
            let key_json = if ts_info { dims_json.clone() } else { String::new() };
            let pos = entries.iter().position(|e| e.name == *name && e.ds == ds && e.dims_json == key_json);
            let e = match pos {
                Some(p) => &mut entries[p],
                None => {
                    entries.push(Entry { name: name.clone(), ds: ds.clone(), dims_json: key_json, units: vec![], ftypes: vec![], mtypes: vec![], dims: vec![] });
                    entries.last_mut().unwrap()
                }
            };
            if let Some(u) = unit {
                push_unique(&mut e.units, u);
            }
            push_unique(&mut e.ftypes, ft);
            push_unique(&mut e.mtypes, mt);
            for d in &present {
                push_unique(&mut e.dims, d);
            }
        }
    }
    // Merge data streams with the same metric signature.
    struct Row {
        name: String,
        dims_json: String,
        ds: Vec<String>,
        units: Vec<String>,
        ftypes: Vec<String>,
        mtypes: Vec<String>,
        dims: Vec<String>,
    }
    let sorted = |v: &Vec<String>| {
        let mut x = v.clone();
        x.sort();
        x
    };
    let mut rows: Vec<Row> = Vec::new();
    for e in entries {
        let p = rows.iter().position(|r| r.name == e.name && r.dims_json == e.dims_json && sorted(&r.units) == sorted(&e.units) && sorted(&r.ftypes) == sorted(&e.ftypes) && sorted(&r.mtypes) == sorted(&e.mtypes));
        let r = match p {
            Some(p) => &mut rows[p],
            None => {
                rows.push(Row { name: e.name.clone(), dims_json: e.dims_json.clone(), ds: vec![], units: e.units.clone(), ftypes: e.ftypes.clone(), mtypes: e.mtypes.clone(), dims: vec![] });
                rows.last_mut().unwrap()
            }
        };
        push_unique(&mut r.ds, &e.ds);
        for d in &e.dims {
            push_unique(&mut r.dims, d);
        }
    }
    let mut cols: Vec<Column> = INFO_COLUMNS.iter().map(|n| Column { name: n.to_string(), ty: "keyword".into() }).collect();
    if ts_info {
        cols.push(Column { name: "dimensions".into(), ty: "keyword".into() });
    }
    let out = rows
        .into_iter()
        .map(|r| {
            let mut v = vec![Val::Str(r.name), set_val(&r.ds), set_val(&r.units), set_val(&r.mtypes), set_val(&r.ftypes), set_val(&r.dims)];
            if ts_info {
                v.push(Val::Str(r.dims_json));
            }
            v
        })
        .collect();
    Table { cols, rows: out }
}

/// `MetricsInfoOperator.resolveDataStreamName`: the data stream of a backing
/// index (`.ds-<name>-yyyy.MM.dd-NNNNNN`), else the index name.
fn backing_data_stream(index: &str) -> String {
    let base = index.rsplit_once(':').map(|(_, b)| b).unwrap_or(index);
    let core = match base.find(".ds-").or_else(|| base.find(".fs-")) {
        Some(i) if i == 0 || base[..i].ends_with('-') => &base[i + 4..],
        _ => return index.to_string(),
    };
    // Strip `-yyyy.MM.dd-NNNNNN`.
    let b = core.as_bytes();
    if b.len() > 18 {
        let tail = &core[core.len() - 18..];
        let tb = tail.as_bytes();
        let ok = tb[0] == b'-'
            && tb[1..5].iter().all(u8::is_ascii_digit)
            && tb[5] == b'.'
            && tb[6..8].iter().all(u8::is_ascii_digit)
            && tb[8] == b'.'
            && tb[9..11].iter().all(u8::is_ascii_digit)
            && tb[11] == b'-'
            && tb[12..18].iter().all(u8::is_ascii_digit);
        if ok {
            let name = &core[..core.len() - 18];
            return match index.rsplit_once(':') {
                Some((alias, _)) => format!("{alias}:{name}"),
                None => name.to_string(),
            };
        }
    }
    index.to_string()
}

/// Internal call evaluating `MATCH` on a column that is not an index field
/// (`Match.isRuntimeSearch`): `[field, query, kind]` where kind is `text`
/// (query = analyzed terms) or `value` (query = the converted value).
const MATCH_RT: &str = "\u{0}MATCH_RT";

/// Standard-analyzer terms of a text (`StandardAnalyzer`: standard
/// tokenizer, lower case, no stop words).
fn standard_terms(s: &str) -> Vec<String> {
    use rustic_core::analysis::tokenizer::StandardTokenizer;
    use rustic_core::analysis::Tokenizer;
    StandardTokenizer.tokenize(s).into_iter().map(|t| t.text.to_lowercase()).collect()
}

/// A `WHERE` whose full-text functions are all `MATCH`/`:` on columns that
/// are not index fields runs them per row; `None` = not such a condition.
fn runtime_match(ctx: &Ctx, e: &Expr, bound: &Expr, cols: &[Column], wpos: usize) -> Result<Option<Expr>> {
    let mut calls = Vec::new();
    fulltext_calls(e, &mut calls);
    let runtime = |(n, args, _): &(&str, &[Expr], usize)| matches!(*n, "MATCH" | "MATCH_OP") && matches!(args.first(), Some(Expr::Col(f, _)) if !ctx.pristine.contains(f));
    if calls.is_empty() || !calls.iter().all(runtime) || has_score(e) {
        return Ok(None);
    }
    let label = func::fulltext_label(calls[0].0);
    if let Some(b) = ctx.fulltext_barrier.filter(|b| !matches!(*b, "ROW" | "SAMPLE" | "PROMQL")) {
        return Err(verification_err(ctx.full, wpos, format!("{label} cannot be used after {b}")));
    }
    for (n, args, _) in &calls {
        if let (Some(Expr::Col(f, fp)), true) = (args.first(), args.len() > 2 && *n == "MATCH") {
            return Err(verification_err(ctx.full, *fp, format!("Options are not supported for [MATCH] function call on non-index-mapped field [{f}]")));
        }
    }
    let srcs = |p: usize| call_arg_sources(ctx.full, p);
    // Rewrite the bound calls; their order matches `calls`.
    let mut i = 0usize;
    let mut err: Option<EsError> = None;
    let out = rewrite_calls(bound, &mut |name, args| {
        if !matches!(name, "MATCH" | "MATCH_OP") {
            return None;
        }
        let pos = calls.get(i).map(|c| c.2).unwrap_or(0);
        i += 1;
        let field = args.first()?.clone();
        let ty = type_of(&field, cols).unwrap_or_else(|| "null".into());
        let q = match args.get(1) {
            Some(Expr::Lit(v)) => v.clone(),
            _ => Val::Null,
        };
        if ty == "text" {
            let terms = match &q {
                Val::Null => vec![],
                v => standard_terms(&v.to_text()),
            };
            if terms.is_empty() {
                return Some(Expr::Lit(Val::Bool(false)));
            }
            return Some(Expr::Call(MATCH_RT.into(), vec![field, Expr::Lit(Val::from_list(terms.into_iter().map(Val::Str).collect())), Expr::Lit(Val::Str("text".into()))], pos));
        }
        let conv = match (&q, ty.as_str()) {
            (Val::Null, _) | (_, "null") => None,
            (Val::Str(s), "integer" | "long" | "double" | "unsigned_long") => s.trim().parse::<f64>().ok().map(Val::Double),
            (Val::Str(s), "boolean") => match s.to_ascii_lowercase().as_str() {
                "true" => Some(Val::Bool(true)),
                "false" => Some(Val::Bool(false)),
                _ => None,
            },
            (Val::Str(s), t) if t.starts_with("date") => func::parse_date_str(s).map(Val::Date),
            (v, "keyword" | "ip" | "version") => Some(Val::Str(v.to_text())),
            (v @ (Val::Int(_) | Val::Long(_) | Val::Double(_)), "integer" | "long" | "double" | "unsigned_long") => Some(v.clone()),
            (v @ Val::Bool(_), "boolean") => Some(v.clone()),
            (v @ Val::Date(_), t) if t.starts_with("date") => Some(v.clone()),
            _ => None,
        };
        let Some(conv) = conv else {
            if !q.is_null() && ty != "null" && err.is_none() {
                let a = srcs(pos);
                let (qp, qs) = a.get(1).cloned().unwrap_or((pos, String::new()));
                let fs = a.first().map(|x| x.1.clone()).unwrap_or_default();
                err = Some(verification_err(ctx.full, qp, format!("[MATCH] query value [{qs}] does not match the type ([{ty}]) of non-index-mapped field [{fs}]")));
            }
            return Some(Expr::Lit(Val::Bool(false)));
        };
        Some(Expr::Call(MATCH_RT.into(), vec![field, Expr::Lit(conv), Expr::Lit(Val::Str("value".into()))], pos))
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok(Some(out))
}

fn has_score(e: &Expr) -> bool {
    match e {
        Expr::Call(n, args, _) => n == "SCORE" || args.iter().any(has_score),
        Expr::Bin(_, a, b) => has_score(a) || has_score(b),
        Expr::Not(a) | Expr::Neg(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) => has_score(a),
        _ => false,
    }
}

/// Replace calls (pre-order) where `f` returns a replacement.
fn rewrite_calls(e: &Expr, f: &mut dyn FnMut(&str, &[Expr]) -> Option<Expr>) -> Expr {
    match e {
        Expr::Call(n, args, p) => match f(n, args) {
            Some(r) => r,
            None => Expr::Call(n.clone(), args.iter().map(|a| rewrite_calls(a, f)).collect(), *p),
        },
        Expr::Bin(op, a, b) => {
            let a = rewrite_calls(a, f);
            Expr::Bin(*op, Box::new(a), Box::new(rewrite_calls(b, f)))
        }
        Expr::Not(a) => Expr::Not(Box::new(rewrite_calls(a, f))),
        o => o.clone(),
    }
}

/// `Match.process*` on one row: `false` (never null) unless a value matches.
fn match_rt_eval(args: &[Expr], row: &[Val], f: &FnCtx) -> Result<Val> {
    let v = eval(&args[0], row, f)?;
    let (Some(Expr::Lit(q)), Some(Expr::Lit(Val::Str(kind)))) = (args.get(1), args.get(2)) else { return Ok(Val::Bool(false)) };
    let vals = v.values();
    Ok(Val::Bool(if kind == "text" {
        let terms = q.values();
        vals.iter().any(|x| standard_terms(&x.to_text()).iter().any(|t| terms.iter().any(|q| matches!(**q, Val::Str(ref s) if s == t))))
    } else {
        vals.iter().any(|x| match (*x, q) {
            (Val::Str(a), Val::Str(b)) => a == b,
            (a, b) => func::compare(a, b) == Some(Ordering::Equal),
        })
    }))
}

/// Track which index fields commands overwrite or drop, and the first
/// command full-text functions cannot follow.
fn track_fields(ctx: &mut Ctx, cmd: &Cmd, t: &Table) {
    let lexical = match cmd {
        Cmd::Eval(_) => Some("EVAL"),
        Cmd::Keep(..) => Some("KEEP"),
        Cmd::Drop(..) => Some("DROP"),
        Cmd::Rename(..) => Some("RENAME"),
        Cmd::Limit(_) | Cmd::LimitBy(..) => Some("LIMIT"),
        Cmd::SqlPivot { .. } => Some("PIVOT"),
        Cmd::Stats { inline, .. } => Some(if *inline { "INLINE" } else { "STATS" }),
        Cmd::MvExpand(..) => Some("MV_EXPAND"),
        Cmd::Dissect { .. } => Some("DISSECT"),
        Cmd::Grok { .. } => Some("GROK"),
        Cmd::LookupJoin { .. } => Some("LOOKUP"),
        Cmd::Info(..) => Some("INFO"),
        Cmd::Enrich { .. } => Some("ENRICH"),
        Cmd::Fork(_) => Some("FORK"),
        _ => None,
    };
    if lexical.is_some() {
        ctx.lexical_barrier = lexical;
    }
    match cmd {
        Cmd::Eval(fields) => fields.iter().for_each(|f| {
            ctx.pristine.remove(&f.name);
        }),
        Cmd::Rename(pairs, _) => pairs.iter().for_each(|(a, b)| {
            ctx.pristine.remove(a);
            ctx.pristine.remove(b);
        }),
        Cmd::Drop(pats, _) => {
            for c in &t.cols {
                if pats.iter().any(|p| name_matches(p, &c.name)) {
                    ctx.pristine.remove(&c.name);
                }
            }
        }
        Cmd::Stats { .. } | Cmd::SqlPivot { .. } => {
            ctx.pristine.clear();
            ctx.fulltext_barrier.get_or_insert("STATS");
        }
        Cmd::Limit(_) => {
            ctx.fulltext_barrier.get_or_insert("LIMIT");
        }
        Cmd::Sample(_) => {
            ctx.fulltext_barrier.get_or_insert("SAMPLE");
        }
        _ => {}
    }
}

fn join_output_limit(tail: &[Cmd]) -> Option<usize> {
    for cmd in tail {
        match cmd {
            Cmd::Limit(n) => return Some(*n),
            Cmd::Eval(_) | Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..)
                | Cmd::Dissect { .. } | Cmd::Grok { .. } | Cmd::LookupJoin { .. } => {},
            _ => return None,
        }
    }
    None
}

// Fuse only scalar, side-effect-free WHERE expressions immediately after
// JOIN. Evaluate on the combined output schema (including shadowed columns),
// not on either input. Stop after enough surviving rows rather than building
// a potentially enormous Cartesian result before applying WHERE/LIMIT.
#[cfg(test)]
thread_local! { static FILTERED_JOIN_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

fn filtered_join_limit(tail: &[Cmd]) -> Option<(Vec<Expr>, usize)> {
    #[cfg(test)]
    if FILTERED_JOIN_DISABLED.with(|disabled| disabled.get()) { return None; }
    fn scalar(e: &Expr) -> bool {
        match e {
            Expr::Lit(_) | Expr::Col(..) => true,
            Expr::Bin(_, a, b) => scalar(a) && scalar(b),
            Expr::Not(a) | Expr::Neg(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => scalar(a),
            Expr::In(a, args, _) => scalar(a) && args.iter().all(scalar),
            Expr::Between(a, b, c, _) => scalar(a) && scalar(b) && scalar(c),
            // CONCAT is deterministic scalar evaluation on the combined row.
            // Do not admit full-text or nondeterministic calls here.
            Expr::Call(name, args, _) if name.eq_ignore_ascii_case("CONCAT") => args.iter().all(scalar),
            _ => false,
        }
    }
    let filters: Vec<_> = tail.iter().take_while(|c| matches!(c, Cmd::Where(e, _) if scalar(e)))
        .map(|c| match c { Cmd::Where(e, _) => e.clone(), _ => unreachable!() }).collect();
    if filters.is_empty() { return None; }
    let limit = join_output_limit(&tail[filters.len()..])?;
    Some((filters, limit))
}

fn join_row_passes(ctx: &Ctx, filters: &[Expr], row: &[Val]) -> Result<bool> {
    for filter in filters {
        if eval(filter, row, &ctx.fctx)? != Val::Bool(true) { return Ok(false); }
    }
    Ok(true)
}

fn apply_with_tail(ctx: &mut Ctx, cmd: &Cmd, t: Table, tail: &[Cmd]) -> Result<Table> {
    if let (Cmd::LookupJoin { index, index_pos, cond, on, on_pos, pos, fields }, Some(limit)) = (cmd, join_output_limit(tail)) {
        track_fields(ctx, cmd, &t);
        return match cond {
            Some(expr) => lookup_join_expr(ctx, t, index, *index_pos, expr, Some(limit), &[], fields.as_ref()),
            None => lookup_join(ctx, t, index, *index_pos, on, on_pos, Some(limit), &[], *pos, fields.as_ref()),
        };
    }
    apply(ctx, cmd, t)
}

fn apply(ctx: &mut Ctx, cmd: &Cmd, t: Table) -> Result<Table> {
    let out = apply_cmd(ctx, cmd, t)?;
    account_table(ctx, &out)?;
    Ok(out)
}

fn apply_cmd(ctx: &mut Ctx, cmd: &Cmd, mut t: Table) -> Result<Table> {
    if let Cmd::Eval(fields) | Cmd::Row(fields) = cmd {
        check_no_fulltext(ctx, fields)?;
    }
    if !matches!(cmd, Cmd::Where(..)) {
        track_fields(ctx, cmd, &t);
    }
    match cmd {
        Cmd::Where(e, wpos) => {
            let b = if has_fulltext(e) {
                let b = bind(ctx, e, &t.cols)?;
                match runtime_match(ctx, e, &b, &t.cols, *wpos)? {
                    Some(rt) => rt,
                    None => return Err(fulltext_where_err(ctx, e, *wpos)),
                }
            } else {
                bind(ctx, e, &t.cols)?
            };
            let mut out = Vec::with_capacity(t.rows.len());
            for r in t.rows.into_iter() {
                if eval(&b, &r, &ctx.fctx)? == Val::Bool(true) {
                    out.push(r);
                }
            }
            t.rows = out;
        }
        Cmd::Eval(fields) => {
            for f in fields {
                let (ty, vals, memory) = eval_col(ctx, &t, f)?;
                t.put_col(&f.name, ty, vals);
                account_table(ctx,&t)?;
                drop(memory);
            }
        }
        Cmd::Limit(n) => t.rows.truncate(*n),
        Cmd::LimitBy(n, keys) => {
            let bound: Vec<Expr> = keys.iter().map(|k| bind(ctx, k, &t.cols)).collect::<Result<_>>()?;
            let mut seen: HashMap<Vec<String>, usize> = HashMap::new();
            let mut out = Vec::with_capacity(t.rows.len());
            for r in t.rows.into_iter() {
                let key = bound.iter().map(|b| eval(b, &r, &ctx.fctx).map(|v| v.key())).collect::<Result<Vec<_>>>()?;
                let c = seen.entry(key).or_insert(0);
                if *c < *n {
                    *c += 1;
                    out.push(r);
                }
            }
            t.rows = out;
        }
        Cmd::Keep(pats, pos) => {
            let mut keep: Vec<usize> = Vec::new();
            for p in pats {
                let matched: Vec<usize> = (0..t.cols.len()).filter(|i| name_matches(p, &t.cols[*i].name)).collect();
                if matched.is_empty() {
                    return Err(if p.contains('*') {
                        verification_err(ctx.full, *pos, format!("No matches found for pattern [{p}]"))
                    } else {
                        unknown_col(ctx, p, *pos, &[])
                    });
                }
                for m in matched {
                    if !p.contains('*') {
                        keep.retain(|k| *k != m);
                        keep.push(m);
                    } else if !keep.contains(&m) {
                        keep.push(m);
                    }
                }
            }
            // Internal columns (the time series id) stay.
            let hidden: Vec<usize> = (0..t.cols.len()).filter(|i| t.cols[*i].name.starts_with('\u{0}') && !keep.contains(i)).collect();
            keep.extend(hidden);
            t.project(&keep);
        }
        Cmd::Drop(pats, pos) => {
            for p in pats {
                if !p.contains('*') && t.idx(p).is_none() {
                    return Err(unknown_col(ctx, p, *pos, &[]));
                }
            }
            let keep: Vec<usize> = (0..t.cols.len()).filter(|i| !pats.iter().any(|p| name_matches(p, &t.cols[*i].name))).collect();
            t.project(&keep);
        }
        Cmd::Rename(pairs, pos) => {
            for (old, new) in pairs {
                let i = t.idx(old).ok_or_else(|| unknown_col(ctx, old, *pos, &t.cols))?;
                if old == new {
                    continue;
                }
                if let Some(j) = t.idx(new) {
                    let keep: Vec<usize> = (0..t.cols.len()).filter(|k| *k != j).collect();
                    t.project(&keep);
                }
                let i = if t.idx(old).is_some() { t.idx(old).unwrap() } else { i };
                t.cols[i].name = new.clone();
            }
        }
        Cmd::Sort(keys) => sort_table(ctx, &mut t, keys)?,
        Cmd::SqlPivot { aggregate, column, values, pos } => t = sql_pivot(ctx, t, aggregate, column, values, *pos)?,
        Cmd::Stats { aggs, by, inline, pos } if ctx.ts && !ctx.ts_done && !*inline => {
            ctx.ts_done = true;
            t = super::ts::ts_stats(ctx, t, aggs, by, *pos)?;
        }
        Cmd::Stats { aggs, by, inline, pos } => t = stats(ctx, t, aggs, by, *inline, *pos)?,
        Cmd::MvExpand(f, pos) => {
            let i = t.idx(f).ok_or_else(|| unknown_col(ctx, f, *pos, &t.cols))?;
            let mut out = Vec::new();
            // The expanded rows are charged as they are built (the input's
            // own charge is reconciled once the table is replaced).
            let pages = crate::memory::RequestMemory::new();
            for r in t.rows.into_iter() {
                match &r[i] {
                    Val::Multi(m) => {
                        for v in m.clone() {
                            let mut nr = r.clone();
                            nr[i] = v;
                            pages.add(row_bytes(&nr), ESQL_LABEL)?;
                            out.push(nr);
                        }
                    }
                    _ => out.push(r),
                }
            }
            t.rows = out;
            if t.cols[i].ty == "text" {
                t.cols[i].ty = "text".into();
            }
        }
        Cmd::Dissect { field, pattern, sep, pos } => dissect(ctx, &mut t, field, pattern, sep, *pos)?,
        Cmd::Grok { field, pattern, pos } => grok(ctx, &mut t, field, pattern, *pos)?,
        Cmd::LookupJoin { index, index_pos, cond: Some(c), fields, .. } => t = lookup_join_expr(ctx, t, index, *index_pos, c, None, &[], fields.as_ref())?,
        Cmd::LookupJoin { index, index_pos, on, on_pos, pos, fields, .. } => t = lookup_join(ctx, t, index, *index_pos, on, on_pos, None, &[], *pos, fields.as_ref())?,
        Cmd::Enrich { policy, on, with, pos, .. } => t = enrich(ctx, t, policy, on.as_deref(), with, *pos)?,
        Cmd::Sample(p) => {
            let mut seed = (crate::date::now_millis() as u64) ^ 0x9E37_79B9_7F4A_7C15;
            t.rows.retain(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 11) as f64 / (1u64 << 53) as f64) < *p
            });
        }
        Cmd::Fork(branches) => {
            let mut outs = Vec::new();
            for (k, b) in branches.iter().enumerate() {
                let mut bt = exec_pipeline(ctx, b, Some(t.clone()))?;
                bt.rows.truncate(ctx.default_limit);
                let n = bt.rows.len();
                bt.put_col("_fork", "keyword".into(), vec![Val::Str(format!("fork{}", k + 1)); n]);
                outs.push(bt);
            }
            let mut cols: Vec<Column> = Vec::new();
            for o in &outs {
                for c in &o.cols {
                    if !cols.iter().any(|x| x.name == c.name) {
                        cols.push(c.clone());
                    }
                }
            }
            // `_fork` last.
            if let Some(p) = cols.iter().position(|c| c.name == "_fork") {
                let c = cols.remove(p);
                cols.push(c);
            }
            let mut rows = Vec::new();
            for o in outs {
                let map: Vec<Option<usize>> = cols.iter().map(|c| o.idx(&c.name)).collect();
                for r in o.rows {
                    rows.push(map.iter().map(|m| m.map(|i| r[i].clone()).unwrap_or(Val::Null)).collect());
                }
            }
            t = Table { cols, rows };
        }
        Cmd::From { .. } | Cmd::Row(_) | Cmd::ShowInfo | Cmd::Promql(_) => return Err(EsError::parsing("source commands must come first")),
        // Only reached inside FORK branches / subqueries.
        Cmd::Info(ts_info, pos) => {
            let label = if *ts_info { "TS_INFO" } else { "METRICS_INFO" };
            return Err(verification_err(ctx.full, *pos, format!("{label} can only be used with TS source command")));
        }
    }
    Ok(t)
}

fn sql_pivot(ctx: &mut Ctx, t: Table, aggregate: &Expr, column: &str, values: &[(Val, String)], pos: usize) -> Result<Table> {
    let mut refs = Vec::new();
    super::analyze::expr_refs(aggregate, &mut refs);
    let by: Vec<Field> = t.cols.iter().filter(|c| c.name != column && !refs.iter().any(|(n, _)| n == &c.name))
        .map(|c| Field { name: c.name.clone(), expr: Expr::Col(c.name.clone(), pos), filter: None, pos }).collect();
    if values.iter().any(|(_, n)| by.iter().any(|b| &b.name == n)) {
        return Err(verification_err(ctx.full, pos, "Pivot output conflicts with a grouping column"));
    }
    let t = apply(ctx, &Cmd::Where(Expr::In(Box::new(Expr::Col(column.into(), pos)), values.iter().map(|(v, _)| Expr::Lit(v.clone())).collect(), false), pos), t)?;
    let empty = t.rows.is_empty();
    let mut aggs = Vec::new();
    let mut outputs = Vec::new();
    let mut names: Vec<String> = by.iter().map(|b| b.name.clone()).collect();
    // Internal names cannot collide with either input or public output names.
    let mut prefix = "__sql_pivot_".to_string();
    while t.cols.iter().any(|c| c.name.starts_with(&prefix)) || values.iter().any(|(_, n)| n.starts_with(&prefix)) { prefix.push('_'); }
    for (i, (value, name)) in values.iter().enumerate() {
        let filter = Expr::Bin(BinOp::Eq, Box::new(Expr::Col(column.into(), pos)), Box::new(Expr::Lit(value.clone())));
        let metric = format!("{prefix}metric{i}");
        let count = format!("{prefix}count{i}");
        aggs.push(Field { name: metric.clone(), expr: aggregate.clone(), filter: Some(filter.clone()), pos });
        aggs.push(Field { name: count.clone(), expr: Expr::Call("COUNT".into(), vec![Expr::Star], pos), filter: Some(filter), pos });
        outputs.push(Field { name: name.clone(), expr: Expr::Call("CASE".into(), vec![
            Expr::Bin(BinOp::Eq, Box::new(Expr::Col(count, pos)), Box::new(Expr::Lit(Val::Int(0)))),
            Expr::Lit(Val::Null), Expr::Col(metric, pos)], pos), filter: None, pos });
        names.push(name.clone());
    }
    let mut result = stats(ctx, t, &aggs, &by, false, pos)?;
    result = apply(ctx, &Cmd::Eval(outputs), result)?;
    result = apply(ctx, &Cmd::Keep(names, pos), result)?;
    if empty { result.rows.clear(); }
    Ok(result)
}

// Column globs need no regex program or cache. Keep LIKE's Unicode '?',
// backslash escaping and trailing-backslash behavior without allocations.
fn column_glob(name:&str,pattern:&str)->bool {
    let mut text=name.chars();let mut pat=pattern.chars();let mut star=None;
    loop {
        let token=pat.next();
        let matched=match token {
            Some('*')=>{star=Some((text.clone(),pat.clone()));continue;},
            Some('?')=>text.next().is_some(),
            Some('\\')=>match pat.next(){Some(c)=>text.next()==Some(c),None=>text.clone().next().is_none()},
            Some(c)=>text.next()==Some(c),
            None=>{if text.clone().next().is_none(){return true;}false},
        };
        if matched {continue;}
        let Some((resume_text,resume_pattern))=star.as_mut() else {return false};
        if resume_text.next().is_none(){return false;}
        text=resume_text.clone();pat=resume_pattern.clone();
    }
}
fn name_matches(pat: &str, name: &str) -> bool {
    if !pat.contains('*') {
        return pat == name;
    }
    !name.starts_with('\u{0}') && column_glob(name,pat)
}

/// Query offset of the first column/function reference in an expression.
fn expr_pos(e: &Expr) -> Option<usize> {
    match e {
        Expr::Col(_, p) | Expr::Call(_, _, p) => Some(*p),
        Expr::Bin(_, a, b) => expr_pos(a).or_else(|| expr_pos(b)),
        Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) | Expr::In(a, ..) | Expr::Between(a, ..) => expr_pos(a),
        _ => None,
    }
}

fn sort_table(ctx: &Ctx, t: &mut Table, keys: &[SortKey]) -> Result<()> {
    let bound: Vec<Expr> = keys.iter().map(|k| bind(ctx, &k.expr, &t.cols)).collect::<Result<_>>()?;
    for (b, k) in bound.iter().zip(keys) {
        if let Some(ty) = type_of(b, &t.cols) {
            if super::spatial::is_spatial(&ty)
                || matches!(
                    ty.as_str(),
                    "_source" | "_tsid" | "counter_long" | "counter_double" | "counter_integer" | "aggregate_metric_double" | "histogram" | "flattened" | "dense_vector" | "tdigest" | "exponential_histogram" | "geohash" | "geotile" | "geohex"
                )
            {
                return Err(verification_err(ctx.full, expr_pos(&k.expr).unwrap_or(0), format!("cannot sort on {ty}")));
            }
        }
    }
    let versions: Vec<bool> = bound.iter().map(|b| type_of(b, &t.cols).as_deref() == Some("version")).collect();
    let mut keyed: Vec<(Vec<Val>, Vec<Val>)> = Vec::with_capacity(t.rows.len());
    // The sort keys sit beside the rows until the sort ends.
    let pages = crate::memory::RequestMemory::new();
    for r in std::mem::take(&mut t.rows) {
        let mut kv = Vec::with_capacity(bound.len());
        for (bi, (b, k)) in bound.iter().zip(keys).enumerate() {
            kv.push(sort_key_value(eval(b, &r, &ctx.fctx)?, versions[bi], k.desc));
        }
        pages.add(row_bytes(&kv), ESQL_LABEL)?;
        keyed.push((kv, r));
    }
    keyed.sort_by(|(a, _), (b, _)| cmp_sort_keys(keys, a, b));
    t.rows = keyed.into_iter().map(|(_, r)| r).collect();
    Ok(())
}

/// A row's evaluated sort key as `SORT` compares it: versions by semver
/// precedence (their sortable encoding), multi-values by their min (asc) /
/// max (desc).
pub(super) fn sort_key_value(v: Val, version: bool, desc: bool) -> Val {
    let v = if version {
        match v {
            Val::Str(s) => Val::Str(rustic_core::fieldtypes::version::encode(&s)),
            Val::Multi(m) => Val::Multi(m.into_iter().map(|x| match x {
                Val::Str(s) => Val::Str(rustic_core::fieldtypes::version::encode(&s)),
                o => o,
            }).collect()),
            o => o,
        }
    } else {
        v
    };
    match v {
        Val::Multi(m) => {
            let it = m.into_iter();
            (if desc { it.max_by(cmp_vals) } else { it.min_by(cmp_vals) }).unwrap_or(Val::Null)
        }
        v => v,
    }
}

/// `SORT`'s order of two rows' [`sort_key_value`]s (nulls placed per key).
pub(super) fn cmp_sort_keys(keys: &[SortKey], a: &[Val], b: &[Val]) -> Ordering {
    for (i, k) in keys.iter().enumerate() {
        let (x, y) = (&a[i], &b[i]);
        let nulls_first = k.nulls_first.unwrap_or(k.desc);
        let o = match (x.is_null(), y.is_null()) {
            (true, true) => Ordering::Equal,
            (true, false) => {
                if nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, true) => {
                if nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            _ => {
                let o = cmp_vals(x, y);
                if k.desc { o.reverse() } else { o }
            }
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

// -------------------------------------------------------------------- STATS

struct AggCall {
    name: String,
    args: Vec<Expr>,
    filter: Option<Expr>,
    /// Query offset of the call (warnings).
    pos: usize,
}

/// The warning of a long `SUM` that overflowed (the aggregate reads as null).
fn sum_overflow_warning(f: &FnCtx, pos: usize) {
    let loc = warning_location(&f.query, pos);
    let src = call_src(&f.query, pos);
    f.warnings.borrow_mut().register(&loc, &format!("evaluation of [{src}] failed, treating result as null"), "java.lang.ArithmeticException: long overflow");
}

fn extract_aggs(ctx: &Ctx, e: &Expr, cols: &[Column], filter: &Option<Expr>, calls: &mut Vec<AggCall>, inside: bool) -> Result<Expr> {
    Ok(match e {
        Expr::Call(name, args, p) if AGGS.contains(&name.as_str()) => {
            if inside {
                return Err(verification_err(ctx.full, *p, format!("The aggregation function [{}] cannot be nested in another aggregation", name.to_lowercase())));
            }
            let bargs: Vec<Expr> = args.iter().enumerate().map(|(ai, a)| match a {
                Expr::Star => Ok(Expr::Star),
                // A field mapped as `aggregate_metric_double` in some indices and as a
                // number in others aggregates as `aggregate_metric_double`.
                Expr::Col(n, _) if ai == 0 && amd_union(cols, n).is_some() => Ok(Expr::Call("TO_AGGREGATE_METRIC_DOUBLE".into(), vec![Expr::Idx(amd_union(cols, n).unwrap())], *p)),
                a => {
                    // Nested aggregates are rejected.
                    let mut dummy = Vec::new();
                    let x = extract_aggs(ctx, a, cols, &None, &mut dummy, true)?;
                    bind(ctx, &x, cols)
                }
            }).collect::<Result<_>>()?;
            if args.is_empty() && name != "COUNT" {
                return Err(verification_err(ctx.full, *p, format!("function [{}] requires an argument", name.to_lowercase())));
            }
            let mut bargs = bargs;
            if ctx.esql {
                check_agg(ctx, name, args, &mut bargs, cols, *p)?;
            }
            let f = match filter {
                Some(fx) => Some(bind(ctx, fx, cols)?),
                None => None,
            };
            calls.push(AggCall { name: name.clone(), args: if bargs.is_empty() { vec![Expr::Star] } else { bargs }, filter: f, pos: *p });
            Expr::AggRef(calls.len() - 1)
        }
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(extract_aggs(ctx, a, cols, filter, calls, inside)?), Box::new(extract_aggs(ctx, b, cols, filter, calls, inside)?)),
        Expr::Neg(a) => Expr::Neg(Box::new(extract_aggs(ctx, a, cols, filter, calls, inside)?)),
        Expr::Not(a) => Expr::Not(Box::new(extract_aggs(ctx, a, cols, filter, calls, inside)?)),
        Expr::Cast(a, t) => Expr::Cast(Box::new(extract_aggs(ctx, a, cols, filter, calls, inside)?), t.clone()),
        Expr::Call(n, args, p) => Expr::Call(n.clone(), args.iter().map(|a| extract_aggs(ctx, a, cols, filter, calls, inside)).collect::<Result<_>>()?, *p),
        other => other.clone(),
    })
}

/// The column index of a union-typed column whose types are
/// `aggregate_metric_double` and numbers.
fn amd_union(cols: &[Column], name: &str) -> Option<usize> {
    let i = cols.iter().rposition(|c| c.name == name)?;
    let types: Vec<&str> = cols[i].ty.strip_prefix("unsupported:")?.split(',').collect();
    (types.contains(&"aggregate_metric_double") && types.iter().all(|t| matches!(*t, "aggregate_metric_double" | "double" | "long" | "integer" | "float" | "half_float" | "scaled_float" | "short" | "byte"))).then_some(i)
}

/// [`amd_union`] for the time-series first pass.
pub fn amd_union_pub(cols: &[Column], name: &str) -> Option<usize> {
    amd_union(cols, name)
}

/// Whether an expression folds to a constant (no column references).
fn foldable(e: &Expr) -> bool {
    match e {
        Expr::Lit(_) | Expr::Map(_) => true,
        Expr::Col(..) | Expr::Idx(_) | Expr::AggRef(_) | Expr::Star => false,
        Expr::Bin(_, a, b) => foldable(a) && foldable(b),
        Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => foldable(a),
        Expr::In(a, l, _) => foldable(a) && l.iter().all(foldable),
        Expr::Between(a, b, c, _) => foldable(a) && foldable(b) && foldable(c),
        Expr::Call(n, args, _) => !AGGS.contains(&n.as_str()) && !FULL_TEXT.contains(&n.as_str()) && args.iter().all(foldable),
    }
}

/// Replace constant subexpressions (no column references) by their value,
/// as ES's constant folding does before translating to Lucene queries.
fn fold_consts(ctx: &Ctx, e: &Expr) -> Expr {
    let folded = |x: &Expr| -> Option<Expr> {
        if matches!(x, Expr::Lit(_) | Expr::Map(_)) || !foldable(x) {
            return None;
        }
        eval(x, &[], &ctx.fctx).ok().map(Expr::Lit)
    };
    if let Some(l) = folded(e) {
        if !matches!(e, Expr::Call(n, ..) if FULL_TEXT.contains(&n.as_str())) {
            return l;
        }
    }
    match e {
        Expr::Call(n, args, p) => Expr::Call(n.clone(), args.iter().map(|a| fold_consts(ctx, a)).collect(), *p),
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(fold_consts(ctx, a)), Box::new(fold_consts(ctx, b))),
        Expr::Not(a) => Expr::Not(Box::new(fold_consts(ctx, a))),
        Expr::Neg(a) => Expr::Neg(Box::new(fold_consts(ctx, a))),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(fold_consts(ctx, a)), *n),
        Expr::Cast(a, t) => Expr::Cast(Box::new(fold_consts(ctx, a)), t.clone()),
        Expr::In(a, l, n) => Expr::In(Box::new(fold_consts(ctx, a)), l.iter().map(|x| fold_consts(ctx, x)).collect(), *n),
        Expr::Between(a, b, c, n) => Expr::Between(Box::new(fold_consts(ctx, a)), Box::new(fold_consts(ctx, b)), Box::new(fold_consts(ctx, c)), *n),
        o => o.clone(),
    }
}

/// Full-text functions take constant queries: `Query must be a constant string in [...]`.
fn check_fulltext_queries(ctx: &Ctx, e: &Expr) -> Result<()> {
    let mut calls = Vec::new();
    fulltext_calls(e, &mut calls);
    for (n, args, p) in calls {
        let qi = match n {
            "MATCH" | "MATCH_OP" | "MATCH_PHRASE" | "TERM" => 1,
            "QSTR" | "KQL" | "MULTI_MATCH" => 0,
            _ => continue,
        };
        let Some(q) = args.get(qi) else { continue };
        if !foldable(q) {
            let srcs = call_arg_sources(ctx.full, p);
            let (qpos, qsrc) = if n == "MATCH_OP" {
                (expr_pos(q).unwrap_or(p), String::new())
            } else {
                srcs.get(qi).cloned().unwrap_or((p, String::new()))
            };
            let fsrc = if n == "MATCH_OP" { String::new() } else { call_src(ctx.full, p) };
            return Err(verification_err(ctx.full, qpos, format!("Query must be a constant string in [{fsrc}], found [{qsrc}]")));
        }
    }
    Ok(())
}

/// Aggregate function resolution: arity, argument types and the constant
/// arguments some aggregations require.
fn check_agg(ctx: &Ctx, name: &str, uargs: &[Expr], bargs: &mut [Expr], cols: &[Column], p: usize) -> Result<()> {
    let lower = name.to_ascii_lowercase();
    let Some(d) = super::typecheck::lookup(&lower) else { return Ok(()) };
    let n = if matches!(uargs.first(), Some(Expr::Star)) { uargs.len() } else { uargs.len() };
    if let Some(m) = super::typecheck::arity_error(d, n) {
        return Err(super::parsing_err(ctx.full, p, format!("error building [{}]: {m}", typed_name(ctx.full, p, name))));
    }
    check_args(ctx, d, name, uargs, bargs, cols, p)?;
    let srcs = call_arg_sources(ctx.full, p);
    let src = call_src(ctx.full, p);
    let arg_src = |i: usize| -> String { srcs.get(i).map(|(_, t)| t.clone()).unwrap_or_default() };
    let constant = |i: usize| -> Result<()> {
        match bargs.get(i) {
            Some(b) if !foldable(b) => Err(verification_err(ctx.full, p, format!("{} argument of [{src}] must be a constant, received [{}]", super::typecheck::ordinal_word(i), arg_src(i)))),
            _ => Ok(()),
        }
    };
    let fold = |i: usize| -> Option<Val> { bargs.get(i).and_then(|b| eval(b, &[], &ctx.fctx).ok()) };
    match name {
        "PERCENTILE" | "COUNT_DISTINCT" => constant(1)?,
        "TOP" | "SAMPLE" => {
            if let Some(b) = bargs.get(1) {
                if !foldable(b) {
                    return Err(verification_err(ctx.full, srcs.get(1).map(|x| x.0).unwrap_or(p), format!("Limit must be a constant integer in [{src}], found [{}]", arg_src(1))));
                }
                if let Some(v) = fold(1).and_then(|v| v.as_i64()) {
                    if v <= 0 {
                        return Err(verification_err(ctx.full, p, format!("Limit must be greater than 0 in [{src}], found [{v}]")));
                    }
                }
            }
            if name == "TOP" {
                if let Some(b) = bargs.get(2) {
                    if !foldable(b) {
                        return Err(verification_err(ctx.full, p, format!("Order must be a constant string in [{src}], found [{}]", arg_src(2))));
                    }
                    if let Some(Val::Str(o)) = fold(2) {
                        if !o.eq_ignore_ascii_case("asc") && !o.eq_ignore_ascii_case("desc") {
                            return Err(verification_err(ctx.full, p, format!("Invalid order value in [{src}], expected [ASC, DESC] but got [{o}]")));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn has_aggref(e: &Expr) -> bool {
    match e {
        Expr::AggRef(_) => true,
        Expr::Bin(_, a, b) => has_aggref(a) || has_aggref(b),
        Expr::Neg(a) | Expr::Not(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) => has_aggref(a),
        Expr::Call(_, args, _) => args.iter().any(has_aggref),
        _ => false,
    }
}

fn agg_type(call: &AggCall, cols: &[Column]) -> Option<String> {
    let at = call.args.first().and_then(|a| type_of(a, cols));
    Some(match call.name.as_str() {
        "COUNT" | "COUNT_DISTINCT" => "long".into(),
        "PRESENT" | "ABSENT" => "boolean".into(),
        "AVG" | "MEDIAN" | "MEDIAN_ABSOLUTE_DEVIATION" | "PERCENTILE" | "STD_DEV" | "STDDEV_POP" | "VARIANCE" | "VAR_POP" | "WEIGHTED_AVG" => "double".into(),
        "SUM" => match at.as_deref() {
            Some("integer") | Some("long") => "long".into(),
            _ => "double".into(),
        },
        "ST_CENTROID_AGG" => if at.as_deref().is_some_and(|t| t.starts_with("cartesian")) { "cartesian_point".into() } else { "geo_point".into() },
        "ST_EXTENT_AGG" => if at.as_deref().is_some_and(|t| t.starts_with("cartesian")) { "cartesian_shape".into() } else { "geo_shape".into() },
        "MIN" | "MAX" if at.as_deref() == Some("aggregate_metric_double") => "double".into(),
        _ => return at.map(|t| if t == "text" { "keyword".into() } else { t }),
    })
}

fn compute_agg(ctx: &Ctx, call: &AggCall, rows: &[&Vec<Val>], cols: &[Column]) -> Result<Val> {
    let f = &ctx.fctx;
    let rows: Vec<&Vec<Val>> = match &call.filter {
        Some(fx) => {
            let mut out = Vec::new();
            for r in rows {
                if eval(fx, r, f)? == Val::Bool(true) {
                    out.push(*r);
                }
            }
            out
        }
        None => rows.to_vec(),
    };
    let first = &call.args[0];
    if call.name == "COUNT" && matches!(first, Expr::Star) {
        return Ok(Val::Long(rows.len() as i64));
    }
    let constant = |k: usize| -> Result<Val> {
        match call.args.get(k) {
            Some(e) => eval(e, &[], f),
            None => Ok(Val::Null),
        }
    };
    if type_of(first, cols).as_deref() == Some("aggregate_metric_double") {
        // Aggregations over pre-aggregated values combine their sub-metrics.
        let mut amds = Vec::with_capacity(rows.len());
        for r in &rows {
            if let Some(a) = super::types::amd_of(&eval(first, r, f)?) {
                amds.push(a);
            }
        }
        let fold = |g: fn(&super::types::Amd) -> Option<f64>, h: fn(f64, f64) -> f64| amds.iter().filter_map(g).reduce(h);
        let sum = fold(|a| a.sum, |x, y| x + y);
        let count: Option<i64> = amds.iter().filter_map(|a| a.count).reduce(|x, y| x + y);
        return Ok(match call.name.as_str() {
            "MIN" => fold(|a| a.min, f64::min).map(Val::Double).unwrap_or(Val::Null),
            "MAX" => fold(|a| a.max, f64::max).map(Val::Double).unwrap_or(Val::Null),
            "SUM" => sum.map(Val::Double).unwrap_or(Val::Null),
            "COUNT" => Val::Long(count.unwrap_or(0)),
            "AVG" => match (sum, count) {
                (Some(s), Some(c)) if c > 0 => Val::Double(s / c as f64),
                _ => Val::Null,
            },
            "STD_DEV" | "VARIANCE" | "MEDIAN" | "MEDIAN_ABSOLUTE_DEVIATION" | "PERCENTILE" => {
                // The per-value average is the default metric.
                let avgs: Vec<f64> = amds.iter().filter_map(|a| match (a.sum, a.count) {
                    (Some(s), Some(c)) if c > 0 => Some(s / c as f64),
                    _ => None,
                }).collect();
                if avgs.is_empty() {
                    Val::Null
                } else {
                    match call.name.as_str() {
                        "MEDIAN" => func::percentile(avgs, 50.0).map(Val::Double).unwrap_or(Val::Null),
                        "PERCENTILE" => func::percentile(avgs, constant(1)?.as_f64().unwrap_or(50.0)).map(Val::Double).unwrap_or(Val::Null),
                        "MEDIAN_ABSOLUTE_DEVIATION" => func::mad(avgs).map(Val::Double).unwrap_or(Val::Null),
                        n => {
                            let m = avgs.iter().sum::<f64>() / avgs.len() as f64;
                            let var = avgs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / avgs.len() as f64;
                            Val::Double(if n == "VARIANCE" { var } else { var.sqrt() })
                        }
                    }
                }
            }
            _ => Val::Null,
        });
    }
    let mut vals: Vec<Val> = Vec::new();
    if call.name != "WEIGHTED_AVG" {
        for r in &rows {
            match eval(first, r, f)? {
                Val::Null => {}
                Val::Multi(m) => vals.extend(m.into_iter().filter(|x| !x.is_null())),
                v => vals.push(v),
            }
        }
    }
    let nums = |vals: &[Val]| -> Vec<f64> { vals.iter().filter_map(Val::as_f64).collect() };
    Ok(match call.name.as_str() {
        "COUNT" => Val::Long(vals.len() as i64),
        "PRESENT" => Val::Bool(!vals.is_empty()),
        "ABSENT" => Val::Bool(vals.is_empty()),
        "COUNT_DISTINCT" => Val::Long(vals.iter().map(Val::key).collect::<HashSet<_>>().len() as i64),
        "SUM" => {
            if vals.is_empty() {
                return Ok(Val::Null);
            }
            if vals.iter().all(|v| matches!(v, Val::Int(_) | Val::Long(_))) {
                let mut s: i64 = 0;
                for v in &vals {
                    match s.checked_add(v.as_i64().unwrap()) {
                        Some(x) => s = x,
                        None => {
                            sum_overflow_warning(f, call.pos);
                            return Ok(Val::Null);
                        }
                    }
                }
                Val::Long(s)
            } else {
                // ES|QL's compensated (Kahan) sum.
                Val::Double(super::partial::Kahan::sum(nums(&vals)))
            }
        }
        "AVG" => {
            let n = nums(&vals);
            if n.is_empty() {
                Val::Null
            } else if type_of(first, cols).as_deref() == Some("integer") {
                // `SUM(int)` (a long, `addExact`) divided by the count.
                let mut s: i64 = 0;
                for v in &vals {
                    match v.as_i64().and_then(|i| s.checked_add(i)) {
                        Some(x) => s = x,
                        None => {
                            sum_overflow_warning(f, call.pos);
                            return Ok(Val::Null);
                        }
                    }
                }
                Val::Double(s as f64 / n.len() as f64)
            } else {
                // Longs are summed as doubles (`SUM(TO_DOUBLE(x))`), compensated.
                Val::Double(super::partial::Kahan::sum(n.iter().copied()) / n.len() as f64)
            }
        }
        "MIN" => vals.into_iter().min_by(cmp_vals).unwrap_or(Val::Null),
        "MAX" => vals.into_iter().max_by(cmp_vals).unwrap_or(Val::Null),
        "MEDIAN" => func::percentile(nums(&vals), 50.0).map(Val::Double).unwrap_or(Val::Null),
        "PERCENTILE" => {
            let p = constant(1)?.as_f64().unwrap_or(50.0);
            func::percentile(nums(&vals), p).map(Val::Double).unwrap_or(Val::Null)
        }
        "MEDIAN_ABSOLUTE_DEVIATION" => func::mad(nums(&vals)).map(Val::Double).unwrap_or(Val::Null),
        "STD_DEV" | "STDDEV_POP" | "VARIANCE" | "VAR_POP" => {
            let n = nums(&vals);
            if n.is_empty() {
                return Ok(Val::Null);
            }
            let m = n.iter().sum::<f64>() / n.len() as f64;
            let var = n.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.len() as f64;
            Val::Double(if call.name.starts_with("VAR") { var } else { var.sqrt() })
        }
        "VALUES" => {
            let mut seen = HashSet::new();
            let mut out: Vec<Val> = vals.into_iter().filter(|v| seen.insert(v.key())).collect();
            out.sort_by(cmp_vals);
            Val::from_list(out)
        }
        "TOP" => {
            let n = constant(1)?.as_i64().unwrap_or(1).max(0) as usize;
            let desc = constant(2)?.as_str().map(|o| o.eq_ignore_ascii_case("desc")).unwrap_or(false);
            vals.sort_by(cmp_vals);
            if desc {
                vals.reverse();
            }
            vals.truncate(n);
            Val::from_list(vals)
        }
        "SAMPLE" => {
            let n = constant(1)?.as_i64().unwrap_or(1).max(0) as usize;
            vals.truncate(n);
            Val::from_list(vals)
        }
        "FIRST" | "LAST" => {
            let mut best: Option<(Val, Val)> = None;
            for r in &rows {
                let v = eval(first, r, f)?;
                let k = match call.args.get(1) {
                    Some(e) => eval(e, r, f)?,
                    None => Val::Null,
                };
                if v.is_null() || k.is_null() {
                    continue;
                }
                let better = match &best {
                    None => true,
                    Some((_, bk)) => {
                        let o = cmp_vals(&k, bk);
                        if call.name == "FIRST" { o == Ordering::Less } else { o == Ordering::Greater }
                    }
                };
                if better {
                    best = Some((v, k));
                }
            }
            best.map(|(v, _)| v).unwrap_or(Val::Null)
        }
        "WEIGHTED_AVG" => {
            let (mut sw, mut s) = (0.0, 0.0);
            for r in &rows {
                let v = eval(first, r, f)?;
                let w = eval(call.args.get(1).unwrap_or(&Expr::Lit(Val::Null)), r, f)?;
                if let (Some(v), Some(w)) = (v.as_f64(), w.as_f64()) {
                    if matches!(v, _) {
                        s += v * w;
                        sw += w;
                    }
                }
            }
            if sw == 0.0 { Val::Null } else { Val::Double(s / sw) }
        }
        "ST_CENTROID_AGG" | "ST_EXTENT_AGG" => {
            let geo = call.args.first().and_then(|a| type_of(a, cols)).as_deref() == Some("geo_point");
            let pts: Vec<(f64, f64)> = vals
                .iter()
                .filter_map(|v| v.as_str())
                .filter_map(super::spatial::point_xy)
                // Geo points aggregate from (quantized) doc values.
                .map(|(x, y)| if geo { (super::spatial::quantize_lon(x), super::spatial::quantize_lat(y)) } else { (x, y) })
                .collect();
            if pts.is_empty() {
                return Ok(Val::Null);
            }
            if call.name == "ST_CENTROID_AGG" {
                let n = pts.len() as f64;
                Val::Str(super::spatial::point_wkt(pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n))
            } else {
                let (minx, maxx) = pts.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
                let (miny, maxy) = pts.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
                Val::Str(super::spatial::to_wkt(&rustic_core::fieldtypes::shape::Geometry::Envelope { min_x: minx, max_x: maxx, max_y: maxy, min_y: miny }))
            }
        }
        other => return Err(EsError::new(400, "verification_exception", format!("Found 1 problem\nUnknown aggregate function [{}]", other.to_lowercase()))),
    })
}

fn stats(ctx: &Ctx, t: Table, aggs: &[Field], by: &[Field], inline: bool, pos: usize) -> Result<Table> {
    // Group expressions.
    let mut group_exprs = Vec::new();
    let mut group_cols = Vec::new();
    for g in by {
        if let Some(m) = bucket_column_meta(ctx, &g.expr, &t.cols) {
            ctx.col_meta.borrow_mut().insert(g.name.clone(), json!({"bucket": m}));
        }
        if let Expr::Call(n, _, p) = &g.expr {
            if AGGS.contains(&n.as_str()) {
                return Err(verification_err(ctx.full, *p, format!("grouping key [{}] cannot be an aggregate function", g.name)));
            }
        }
        let b = bind(ctx, &g.expr, &t.cols)?;
        group_cols.push(Column { name: g.name.clone(), ty: type_of(&b, &t.cols).unwrap_or_else(|| "null".into()) });
        group_exprs.push(b);
    }
    // Aggregates: outer expression over AggRefs and group columns.
    let mut calls: Vec<AggCall> = Vec::new();
    let mut outers = Vec::new();
    for a in aggs {
        let start = calls.len();
        let outer = extract_aggs(ctx, &a.expr, &t.cols, &a.filter, &mut calls, false)?;
        if !has_aggref(&outer) {
            // A plain group reference is allowed (`STATS g BY g` is not, but `x = g` style
            // references to grouping keys are).
            let refs_group = matches!(&outer, Expr::Col(n, _) if by.iter().any(|g| &g.name == n));
            if !refs_group {
                return Err(verification_err(ctx.full, a.pos, format!("expected an aggregate function but found [{}]", a.name)));
            }
        }
        let _ = start;
        outers.push(outer);
    }
    // Bind outer expressions against [agg results..., group values...].
    let mut virt: Vec<Column> = calls.iter().enumerate().map(|(k, c)| Column { name: format!("\u{0}agg{k}"), ty: agg_type(c, &t.cols).unwrap_or_else(|| "null".into()) }).collect();
    virt.extend(group_cols.iter().cloned());
    let nagg = calls.len();
    let bound_outers: Vec<Expr> = outers
        .iter()
        .map(|o| {
            let o = replace_aggrefs(o);
            bind(ctx, &o, &virt)
        })
        .collect::<Result<_>>()?;
    // Group rows (the groups' keys and member lists are charged as they
    // grow: ES's hash aggregation operator charges its blocks).
    let grouping = crate::memory::RequestMemory::new();
    let mut order: Vec<(Vec<Val>, Vec<usize>)> = Vec::new();
    let mut index: HashMap<Vec<String>, usize> = HashMap::new();
    let mut row_group: Vec<Option<usize>> = vec![None; t.rows.len()];
    if group_exprs.is_empty() {
        order.push((vec![], (0..t.rows.len()).collect()));
        for g in row_group.iter_mut() {
            *g = Some(0);
        }
    } else {
        for (ri, r) in t.rows.iter().enumerate() {
            let mut combos: Vec<Vec<Val>> = vec![vec![]];
            for g in &group_exprs {
                let v = eval(g, r, &ctx.fctx)?;
                let opts: Vec<Val> = match v {
                    Val::Multi(m) => m,
                    v => vec![v],
                };
                let mut next = Vec::new();
                for c in &combos {
                    for o in &opts {
                        let mut c2 = c.clone();
                        c2.push(o.clone());
                        next.push(c2);
                    }
                }
                combos = next;
            }
            for c in combos {
                let k: Vec<String> = c.iter().map(Val::key).collect();
                let gi = match index.get(&k) {
                    Some(gi) => *gi,
                    None => {
                        grouping.add(2 * row_bytes(&c) + k.iter().map(|s| 24 + s.capacity() as u64).sum::<u64>() + 64, ESQL_LABEL)?;
                        order.push((c.clone(), Vec::new()));
                        index.insert(k, order.len() - 1);
                        order.len() - 1
                    }
                };
                grouping.add(8, ESQL_LABEL)?;
                order[gi].1.push(ri);
                if row_group[ri].is_none() {
                    row_group[ri] = Some(gi);
                }
            }
        }
    }
    let _ = pos;
    // Evaluate.
    let mut out_rows: Vec<Vec<Val>> = Vec::with_capacity(order.len());
    for (key, members) in &order {
        let rows: Vec<&Vec<Val>> = members.iter().map(|i| &t.rows[*i]).collect();
        let mut vr: Vec<Val> = Vec::with_capacity(nagg + key.len());
        for c in &calls {
            vr.push(compute_agg(ctx, c, &rows, &t.cols)?);
        }
        vr.extend(key.iter().cloned());
        let mut row = Vec::with_capacity(bound_outers.len() + key.len());
        for o in &bound_outers {
            row.push(eval(o, &vr, &ctx.fctx)?);
        }
        row.extend(key.iter().cloned());
        out_rows.push(row);
    }
    let mut cols: Vec<Column> = Vec::new();
    for (k, a) in aggs.iter().enumerate() {
        let st = type_of(&bound_outers[k], &virt);
        let vals: Vec<Val> = out_rows.iter().map(|r| r[k].clone()).collect();
        cols.push(Column { name: a.name.clone(), ty: infer_type(st, &vals) });
    }
    if !inline {
        cols.extend(group_cols.iter().cloned());
        // Later duplicates shadow earlier ones.
        let mut keep: Vec<usize> = Vec::new();
        for i in 0..cols.len() {
            if let Some(p) = keep.iter().position(|k| cols[*k].name == cols[i].name) {
                keep.remove(p);
            }
            keep.push(i);
        }
        let mut res = Table { cols, rows: out_rows };
        res.project(&keep);
        return Ok(res);
    }
    // INLINE STATS: each input row gets its group's aggregates; the
    // aggregate and grouping columns move to the end.
    let mut res = t.clone();
    for (k, c) in cols.iter().enumerate() {
        let vals: Vec<Val> = row_group.iter().map(|g| g.map(|g| out_rows[g][k].clone()).unwrap_or(Val::Null)).collect();
        res.put_col(&c.name, c.ty.clone(), vals);
    }
    for g in &group_cols {
        if let Some(i) = res.idx(&g.name) {
            let vals: Vec<Val> = res.rows.iter().map(|r| r[i].clone()).collect();
            let ty = res.cols[i].ty.clone();
            res.put_col(&g.name, ty, vals);
        }
    }
    Ok(res)
}

/// `_meta.bucket` of a `BUCKET(..)` / `TBUCKET(..)` grouping.
pub fn bucket_column_meta(ctx: &Ctx, e: &Expr, cols: &[Column]) -> Option<Value> {
    let Expr::Call(n, args, _) = e else { return None };
    if n != "BUCKET" && n != "TBUCKET" {
        return None;
    }
    let mut a: Vec<Expr> = args.clone();
    if n == "TBUCKET" && a.len() != 2 {
        a.insert(0, Expr::Col("@timestamp".into(), 0));
    }
    let date = match &a[0] {
        Expr::Col(c, _) => cols.iter().rfind(|x| &x.name == c).map(|x| x.ty.starts_with("date")).unwrap_or(false),
        _ => false,
    };
    let mut vals = vec![Val::Null];
    for x in &a[1..] {
        vals.push(eval(&bind(ctx, x, &[]).ok()?, &[], &ctx.fctx).ok()?);
    }
    func::bucket_meta(&vals, date)
}

/// One branch of a `FROM` union.
enum Branch {
    Indices(Vec<String>),
    View(String),
    Sub(Vec<Cmd>),
}

/// `FROM` naming views (exactly or by wildcard) or `(subqueries)`: the
/// union of each branch's result, then the rest of the pipeline. `None`
/// when FROM reads only indices.
fn from_union(ctx: &mut Ctx, indices: &[String], metadata: &[String], pos: usize, subqueries: &[Vec<Cmd>], rest: &[Cmd]) -> Result<Option<Table>> {
    let names = ctx.node.map(|n| super::views::store(n).names()).unwrap_or_default();
    let mut views: Vec<String> = Vec::new();
    let mut others: Vec<String> = Vec::new();
    for p in indices {
        if names.contains(p) {
            if !views.contains(p) {
                views.push(p.clone());
            }
        } else if p.contains('*') && !names.is_empty() {
            let matched: Vec<&String> = names.iter().filter(|n| crate::mapping::wildcard_match(p, n)).collect();
            for m in &matched {
                if !views.contains(m) {
                    views.push((*m).clone());
                }
            }
            // The pattern may also match indices.
            let resolves = ctx.node.is_some_and(|node| {
                let mut o = ResolveOpts::from_params(&HashMap::new());
                o.allow_no_indices = true;
                node.resolve(p, o).map(|t| !t.is_empty()).unwrap_or(false)
            });
            if resolves || matched.is_empty() {
                others.push(p.clone());
            }
        } else {
            others.push(p.clone());
        }
    }
    if views.is_empty() && subqueries.is_empty() {
        return Ok(None);
    }
    if !views.is_empty() {
        ctx.used_views = true;
    }
    // Loads an enclosing union broadcast here go to every branch too.
    let inherited = std::mem::take(&mut ctx.extra_load);
    let mut branches: Vec<Branch> = Vec::new();
    if !others.is_empty() {
        branches.push(Branch::Indices(others));
    }
    branches.extend(views.into_iter().map(Branch::View));
    branches.extend(subqueries.iter().cloned().map(Branch::Sub));
    let run_branches = |ctx: &mut Ctx, extra: &[String]| -> Result<Vec<Table>> {
        let mut parts = Vec::new();
        for b in &branches {
            ctx.extra_load = inherited.clone();
            ctx.extra_load.extend(extra.iter().filter(|e| !inherited.contains(e)).cloned());
            let t = match b {
                Branch::Indices(ix) => {
                    let saved = (ctx.ts, ctx.ts_done);
                    let r = from_with_pushdown(ctx, ix, metadata, pos, &[]).map(|(t, _, _)| t);
                    ctx.ts = saved.0;
                    ctx.ts_done = saved.1;
                    r?
                }
                Branch::View(v) => {
                    if let Some(start) = ctx.view_stack.iter().position(|x| x == v) {
                        let chain = ctx.view_stack[start..].join(" -> ");
                        return Err(EsError::new(400, "verification_exception", format!("circular view reference '{v}': {chain}")));
                    }
                    let q = ctx.node.map(|n| super::views::store(n).get(v).unwrap_or_default()).unwrap_or_default();
                    let (_, cmds) = super::parse::parse_statement(&q, &Params::default())?;
                    ctx.view_stack.push(v.clone());
                    let r = run_sub_pipeline(ctx, &cmds);
                    ctx.view_stack.pop();
                    r?
                }
                Branch::Sub(cmds) => run_sub_pipeline(ctx, cmds)?,
            };
            ctx.extra_load.clear();
            parts.push(t);
        }
        Ok(parts)
    };
    let mut parts = run_branches(ctx, &[])?;
    // `unmapped_fields`: names the outer pipeline references that no branch
    // surfaces are loaded in every branch.
    if ctx.settings.unmapped != super::settings::Unmapped::Default {
        let mut scope: Vec<String> = Vec::new();
        for t in &parts {
            for c in &t.cols {
                if !scope.contains(&c.name) {
                    scope.push(c.name.clone());
                }
            }
        }
        let extra = unmapped_refs(ctx, rest, scope);
        if !extra.is_empty() {
            parts = run_branches(ctx, &extra)?;
        }
    }
    // Union by column name; differing types conflict.
    let mut cols: Vec<Column> = Vec::new();
    for t in &parts {
        for c in &t.cols {
            match cols.iter_mut().find(|x| x.name == c.name) {
                None => cols.push(c.clone()),
                Some(x) if x.ty == "null" => x.ty = c.ty.clone(),
                Some(x) if c.ty != "null" && x.ty != c.ty && !x.ty.starts_with(SUBQUERY_CONFLICT) => {
                    let mut ts = vec![x.ty.clone(), c.ty.clone()];
                    ts.sort();
                    x.ty = format!("{SUBQUERY_CONFLICT}{}", ts.join(","));
                }
                Some(x) if x.ty.starts_with(SUBQUERY_CONFLICT) && c.ty != "null" => {
                    let mut ts: Vec<String> = x.ty[SUBQUERY_CONFLICT.len()..].split(',').map(str::to_string).collect();
                    if !ts.contains(&c.ty) {
                        ts.push(c.ty.clone());
                        ts.sort();
                        x.ty = format!("{SUBQUERY_CONFLICT}{}", ts.join(","));
                    }
                }
                _ => {}
            }
        }
    }
    let mut rows = Vec::new();
    for t in parts {
        let map: Vec<Option<usize>> = cols.iter().map(|c| t.idx(&c.name)).collect();
        for r in t.rows {
            rows.push(map.iter().map(|m| m.map(|i| r[i].clone()).unwrap_or(Val::Null)).collect());
        }
    }
    let mut t = Table { cols, rows };
    for c in rest {
        t = apply(ctx, c, t)?;
    }
    Ok(Some(t))
}

/// Type prefix of a union column whose branches disagree on its type.
pub const SUBQUERY_CONFLICT: &str = "\u{0}conflict:";

/// Run a nested source pipeline (`PROMQL`'s compiled plan) with its own
/// time-series state.
pub fn run_sub_pipeline(ctx: &mut Ctx, cmds: &[Cmd]) -> Result<Table> {
    let saved = (ctx.ts, ctx.ts_done, std::mem::take(&mut ctx.ts_dims), std::mem::replace(&mut ctx.drop_null_columns, false));
    ctx.ts = false;
    ctx.ts_done = false;
    let r = exec_pipeline(ctx, cmds, None);
    ctx.ts = saved.0;
    ctx.ts_done = saved.1;
    ctx.ts_dims = saved.2;
    ctx.drop_null_columns = saved.3;
    r
}

/// `STATS` without INLINE (the time-series second pass).
pub fn stats_pub(ctx: &Ctx, t: Table, aggs: &[Field], by: &[Field], pos: usize) -> Result<Table> {
    stats(ctx, t, aggs, by, false, pos)
}

fn replace_aggrefs(e: &Expr) -> Expr {
    match e {
        Expr::AggRef(k) => Expr::Col(format!("\u{0}agg{k}"), 0),
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(replace_aggrefs(a)), Box::new(replace_aggrefs(b))),
        Expr::Neg(a) => Expr::Neg(Box::new(replace_aggrefs(a))),
        Expr::Not(a) => Expr::Not(Box::new(replace_aggrefs(a))),
        Expr::Cast(a, t) => Expr::Cast(Box::new(replace_aggrefs(a)), t.clone()),
        Expr::Call(n, args, p) => Expr::Call(n.clone(), args.iter().map(replace_aggrefs).collect(), *p),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(replace_aggrefs(a)), *n),
        other => other.clone(),
    }
}

// ------------------------------------------------------------ DISSECT/GROK

struct DissectKey {
    name: String,
    append: bool,
    skip: bool,
    pad: bool,
    delim: String,
}

fn dissect(ctx: &Ctx, t: &mut Table, field: &Expr, pattern: &str, sep: &str, pos: usize) -> Result<()> {
    // Parse pattern.
    let mut prefix = String::new();
    let mut keys: Vec<DissectKey> = Vec::new();
    let mut rest = pattern;
    while let Some(p) = rest.find("%{") {
        let lit = &rest[..p];
        match keys.last_mut() {
            Some(k) => k.delim.push_str(lit),
            None => prefix.push_str(lit),
        }
        let end = rest[p..].find('}').ok_or_else(|| verification_err(ctx.full, pos, format!("Invalid pattern [{pattern}] for dissect: Unable to find match for dissect pattern")))? + p;
        let mut k = &rest[p + 2..end];
        let mut dk = DissectKey { name: String::new(), append: false, skip: false, pad: false, delim: String::new() };
        if let Some(x) = k.strip_suffix("->") {
            dk.pad = true;
            k = x;
        }
        if let Some(x) = k.strip_prefix('+') {
            dk.append = true;
            k = x;
        } else if let Some(x) = k.strip_prefix('?') {
            dk.skip = true;
            k = x;
        } else if k.starts_with('*') || k.starts_with('&') {
            return Err(verification_err(ctx.full, pos, "Reference keys not supported in dissect patterns"));
        }
        let k = k.split('/').next().unwrap_or("");
        if k.is_empty() {
            dk.skip = true;
        }
        dk.name = k.to_string();
        keys.push(dk);
        rest = &rest[end + 1..];
    }
    match keys.last_mut() {
        Some(k) => k.delim.push_str(rest),
        None => return Err(verification_err(ctx.full, pos, format!("Invalid pattern [{pattern}] for dissect: Unable to find any keys or delimiters."))),
    }
    let mut names: Vec<String> = Vec::new();
    for k in &keys {
        if !k.skip && !names.contains(&k.name) {
            names.push(k.name.clone());
        }
    }
    let b = bind(ctx, field, &t.cols)?;
    let mut cols: Vec<Vec<Val>> = vec![Vec::with_capacity(t.rows.len()); names.len()];
    for r in &t.rows {
        let v = eval(&b, r, &ctx.fctx)?;
        let res = match v {
            Val::Str(s) => dissect_one(&s, &prefix, &keys, sep),
            _ => None,
        };
        for (i, n) in names.iter().enumerate() {
            cols[i].push(res.as_ref().and_then(|m| m.get(n).cloned()).map(Val::Str).unwrap_or(Val::Null));
        }
    }
    for (n, vals) in names.iter().zip(cols) {
        t.put_col(n, "keyword".into(), vals);
    }
    Ok(())
}

fn dissect_one(s: &str, prefix: &str, keys: &[DissectKey], sep: &str) -> Option<BTreeMap<String, String>> {
    let mut rest = s.strip_prefix(prefix)?;
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for (i, k) in keys.iter().enumerate() {
        let val;
        if k.delim.is_empty() {
            if i + 1 == keys.len() {
                val = rest;
                rest = "";
            } else {
                val = "";
            }
        } else {
            let p = rest.find(&k.delim)?;
            val = &rest[..p];
            rest = &rest[p + k.delim.len()..];
            if k.pad {
                while rest.starts_with(&k.delim) {
                    rest = &rest[k.delim.len()..];
                }
            }
        }
        if k.skip {
            continue;
        }
        if k.append {
            if let Some(e) = out.get_mut(&k.name) {
                e.push_str(sep);
                e.push_str(val);
                continue;
            }
        }
        out.insert(k.name.clone(), val.to_string());
    }
    Some(out)
}

fn grok_base(name: &str) -> Option<&'static str> {
    Some(match name {
        "USERNAME" | "USER" => r"[a-zA-Z0-9._-]+",
        "EMAILLOCALPART" => r"[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~]+(?:\.[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~]+)*",
        "EMAILADDRESS" => "%{EMAILLOCALPART}@%{HOSTNAME}",
        "INT" => r"(?:[+-]?(?:[0-9]+))",
        "BASE10NUM" | "NUMBER" => r"(?:[+-]?(?:(?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+)))",
        "BASE16NUM" => r"(?:0[xX]?[0-9a-fA-F]+)",
        "POSINT" => r"\b(?:[1-9][0-9]*)\b",
        "NONNEGINT" => r"\b(?:[0-9]+)\b",
        "WORD" => r"\b\w+\b",
        "NOTSPACE" => r"\S+",
        "SPACE" => r"\s*",
        "DATA" => r".*?",
        "GREEDYDATA" => r".*",
        "QUOTEDSTRING" | "QS" => r#"(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)"#,
        "UUID" => r"[A-Fa-f0-9]{8}-(?:[A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}",
        "MAC" => r"(?:(?:[A-Fa-f0-9]{2}[:-]){5}[A-Fa-f0-9]{2}|(?:[A-Fa-f0-9]{4}\.){2}[A-Fa-f0-9]{4})",
        "IPV4" => r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])",
        "IPV6" => r"(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}",
        "IP" => "(?:%{IPV6}|%{IPV4})",
        "HOSTNAME" => r"\b(?:[0-9A-Za-z][0-9A-Za-z-]{0,62})(?:\.(?:[0-9A-Za-z][0-9A-Za-z-]{0,62}))*(?:\.?|\b)",
        "IPORHOST" | "HOST" => "(?:%{IP}|%{HOSTNAME})",
        "HOSTPORT" => "%{IPORHOST}:%{POSINT}",
        "PATH" | "UNIXPATH" => r"(?:/[\w_%!$@:.,+~-]*)+",
        "URIPROTO" => r"[A-Za-z](?:[A-Za-z0-9+\-.]+)+",
        "URIPATH" => r"(?:/[A-Za-z0-9$.+!*'(){},~:;=@#%&_\-]*)+",
        "URIPARAM" => r"\?[A-Za-z0-9$.+!*'|(){},~@#%&/=:;_?\-\[\]<>]*",
        "URIPATHPARAM" => "%{URIPATH}(?:%{URIPARAM})?",
        "URIHOST" => "%{IPORHOST}(?::%{POSINT})?",
        "URI" => r"%{URIPROTO}://(?:%{USER}(?::[^@]*)?@)?(?:%{URIHOST})?(?:%{URIPATHPARAM})?",
        "MONTH" => r"\b(?:[Jj]an(?:uary|uar)?|[Ff]eb(?:ruary|ruar)?|[Mm](?:a|ä)?r(?:ch|z)?|[Aa]pr(?:il)?|[Mm]a(?:y|i)?|[Jj]un(?:e|i)?|[Jj]ul(?:y|i)?|[Aa]ug(?:ust)?|[Ss]ep(?:tember)?|[Oo](?:c|k)?t(?:ober)?|[Nn]ov(?:ember)?|[Dd]e(?:c|z)(?:ember)?)\b",
        "MONTHNUM" => r"(?:0?[1-9]|1[0-2])",
        "MONTHDAY" => r"(?:(?:0[1-9])|(?:[12][0-9])|(?:3[01])|[1-9])",
        "DAY" => r"(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)",
        "YEAR" => r"(?:\d\d){1,2}",
        "HOUR" => r"(?:2[0123]|[01]?[0-9])",
        "MINUTE" => r"(?:[0-5][0-9])",
        "SECOND" => r"(?:(?:[0-5]?[0-9]|60)(?:[:.,][0-9]+)?)",
        "TIME" => "%{HOUR}:%{MINUTE}(?::%{SECOND})",
        "ISO8601_TIMEZONE" => "(?:Z|[+-]%{HOUR}(?::?%{MINUTE}))",
        "TIMESTAMP_ISO8601" => "%{YEAR}-%{MONTHNUM}-%{MONTHDAY}[T ]%{HOUR}:?%{MINUTE}(?::?%{SECOND})?%{ISO8601_TIMEZONE}?",
        "DATE_US" => "%{MONTHNUM}[/-]%{MONTHDAY}[/-]%{YEAR}",
        "DATE_EU" => r"%{MONTHDAY}[./-]%{MONTHNUM}[./-]%{YEAR}",
        "DATE" => "%{DATE_US}|%{DATE_EU}",
        "HTTPDATE" => "%{MONTHDAY}/%{MONTH}/%{YEAR}:%{TIME} %{INT}",
        "LOGLEVEL" => r"(?:[Aa]lert|ALERT|[Tt]race|TRACE|[Dd]ebug|DEBUG|[Nn]otice|NOTICE|[Ii]nfo?(?:rmation)?|INFO?(?:RMATION)?|[Ww]arn?(?:ing)?|WARN?(?:ING)?|[Ee]rr?(?:or)?|ERR?(?:OR)?|[Cc]rit?(?:ical)?|CRIT?(?:ICAL)?|[Ff]atal|FATAL|[Ss]evere|SEVERE|EMERG(?:ENCY)?|[Ee]merg(?:ency)?)",
        _ => return None,
    })
}

fn grok_push(out:&mut String,text:&str,memory:&mut crate::memory::Reservation)->Result<()> {
    memory.grow((text.len() as u64).saturating_mul(2).saturating_add(64),ESQL_LABEL)?;
    out.push_str(text);Ok(())
}
fn grok_compile(ctx: &Ctx, pattern: &str, pos: usize, caps: &mut Vec<(String, String)>, depth: usize,memory:&mut crate::memory::Reservation) -> Result<String> {
    if depth > 20 {
        return Err(verification_err(ctx.full, pos, "grok pattern recursion too deep"));
    }
    let mut out = String::new();
    let mut rest = pattern;
    while let Some(p) = rest.find("%{") {
        grok_push(&mut out,&rest[..p],memory)?;
        let end = rest[p..].find('}').ok_or_else(|| verification_err(ctx.full, pos, format!("Invalid GROK pattern [{pattern}]")))? + p;
        let spec = &rest[p + 2..end];
        let mut parts = spec.splitn(3, ':');
        let pname = parts.next().unwrap_or("");
        let field = parts.next();
        let ty = parts.next().unwrap_or("keyword");
        let base = grok_base(pname).ok_or_else(|| verification_err(ctx.full, pos, format!("Invalid GROK pattern [{pattern}]: Unable to find pattern [{pname}] in Grok's pattern dictionary")))?;
        let inner = grok_compile(ctx, base, pos, &mut Vec::new(), depth + 1,memory)?;
        match field {
            Some(f) => {
                memory.grow((f.len() as u64).saturating_add(ty.len() as u64).saturating_mul(2).saturating_add(256),ESQL_LABEL)?;
                let g = format!("g{}", caps.len());
                caps.push((f.to_string(), ty.to_string()));
                for piece in ["(?P<",g.as_str(),">",inner.as_str(),")"] {grok_push(&mut out,piece,memory)?;}
            }
            None => {for piece in ["(?:",inner.as_str(),")"] {grok_push(&mut out,piece,memory)?;}},
        }
        rest = &rest[end + 1..];
    }
    grok_push(&mut out,rest,memory)?;
    Ok(out)
}

fn grok(ctx: &Ctx, t: &mut Table, field: &Expr, pattern: &str, pos: usize) -> Result<()> {
    let mut memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
    let mut caps = Vec::new();
    let re_src = grok_compile(ctx, pattern, pos, &mut caps, 0,&mut memory)?;
    let re = super::regex_memory::Program::compile_detailed(&re_src)?.map_err(|e| verification_err(ctx.full, pos, format!("Invalid GROK pattern [{pattern}]: {e}")))?;
    let b = bind(ctx, field, &t.cols)?;
    let mut names: Vec<(String, String)> = Vec::new();
    for (n, ty) in &caps {
        if !names.iter().any(|(x, _)| x == n) {
            memory.grow((n.len() as u64).saturating_add(ty.len() as u64).saturating_mul(2).saturating_add(256),ESQL_LABEL)?;
            names.push((n.clone(), ty.clone()));
        }
    }
    memory.grow((names.len() as u64).saturating_mul(t.rows.len() as u64).saturating_mul(128).saturating_add((names.len() as u64).saturating_mul(128)),ESQL_LABEL)?;
    let mut cols: Vec<Vec<Val>> = vec![Vec::with_capacity(t.rows.len()); names.len()];
    for r in &t.rows {
        let mut input_memory=crate::memory::breakers().reserve_request((caps.len() as u64).saturating_mul(128),ESQL_LABEL)?;
        let v = eval_reserved(&b, r, &ctx.fctx,&mut input_memory)?;
        let m = match &v {
            Val::Str(s) => Some(re.captures(s)?),
            _ => None,
        };
        for (i, (n, ty)) in names.iter().enumerate() {
            let mut vals = Vec::new();
            if let Some(m) = &m {
                for (k, (cn, _)) in caps.iter().enumerate() {
                    if cn == n {
                        if let Some(s) = m.named(match &v{Val::Str(s)=>s,_=>""},&format!("g{k}")) {
                            memory.grow((s.len() as u64).saturating_mul(2).saturating_add(128),ESQL_LABEL)?;
                            vals.push(match ty.as_str() {
                                "int" => s.parse::<i64>().map(Val::Int).unwrap_or(Val::Null),
                                "long" => s.parse::<i64>().map(Val::Long).unwrap_or(Val::Null),
                                "float" | "double" => s.parse::<f64>().map(Val::Double).unwrap_or(Val::Null),
                                "boolean" => Val::Bool(s.eq_ignore_ascii_case("true")),
                                _ => Val::Str(s.to_string()),
                            });
                        }
                    }
                }
            }
            cols[i].push(Val::from_list(vals));
        }
    }
    for ((n, ty), vals) in names.iter().zip(cols) {
        let t2 = match ty.as_str() {
            "int" => "integer",
            "long" => "long",
            "float" | "double" => "double",
            "boolean" => "boolean",
            _ => "keyword",
        };
        t.put_col(n, t2.into(), vals);
    }
    account_table(ctx,t)?;
    Ok(())
}

// ----------------------------------------------------------------- loading

pub fn esql_type(ty: EsType) -> Option<&'static str> {
    Some(match ty.name() {
        "text" | "match_only_text" | "pattern_text" | "annotated_text" | "search_as_you_type" | "semantic_text" => "text",
        "keyword" | "constant_keyword" | "wildcard" => "keyword",
        "long" => "long",
        "integer" | "short" | "byte" | "token_count" => "integer",
        "double" | "float" | "half_float" | "scaled_float" => "double",
        "unsigned_long" => "unsigned_long",
        "boolean" => "boolean",
        "date" => "date",
        "date_nanos" => "date_nanos",
        "ip" => "ip",
        "version" => "version",
        "geo_point" => "geo_point",
        "point" => "cartesian_point",
        "geo_shape" => "geo_shape",
        "shape" => "cartesian_shape",
        "aggregate_metric_double" => "aggregate_metric_double",
        "histogram" => "histogram",
        "flattened" => "flattened",
        "dense_vector" => "dense_vector",
        "tdigest" => "tdigest",
        "exponential_histogram" => "exponential_histogram",
        "alias" | "object" | "nested" => return None,
        _ => "unsupported",
    })
}

/// ES|QL type of a mapped field (its declared type wins over the type it
/// behaves like, e.g. `version`, `token_count`).
fn field_esql_type(f: &crate::mapping::ResolvedField) -> Option<&'static str> {
    esql_type(f.declared).or_else(|| esql_type(f.ty))
}

#[derive(Clone)]
struct ColSrc {
    path: String,
    ty: String,
    /// The mapped type (`half_float`, `float`, `scaled_float`, ... round like doc values).
    declared: String,
    date_fmt: Option<crate::date::DateFormatter>,
    ignore_above: Option<usize>,
    normalizer: Option<std::sync::Arc<rustic_core::analysis::Analyzer>>,
    /// `constant_keyword` values by index name.
    constants: HashMap<String, Value>,
    scaling_factor: f64,
    /// Values come from doc values (sorted, keywords deduplicated) rather than `_source`.
    doc_values: bool,
    /// Per mapped type (ES|QL type, or the mapping type when unsupported): the indices.
    types: Vec<(String, Vec<String>)>,
    /// How each index's documents read this column (union types read natively).
    per_index: HashMap<String, ColSrc>,
    /// A time series dimension.
    dimension: bool,
    /// Strict columnar indices (`logsdb_columnar`, `columnar`): their doc
    /// values keep each document's array order (`.offsets`).
    ordered: Vec<String>,
    /// `pattern_text`'s `template_id` sub-field: computed from the message.
    template_id: bool,
}

impl ColSrc {
    fn plain(path: &str, ty: &str) -> ColSrc {
        ColSrc { path: path.to_string(), ty: ty.to_string(), declared: ty.to_string(), date_fmt: None, ignore_above: None, normalizer: None, constants: HashMap::new(), scaling_factor: 1.0, doc_values: true, types: Vec::new(), per_index: HashMap::new(), dimension: false, ordered: Vec::new(), template_id: false }
    }

    fn of(path: String, ty: &str, f: &crate::mapping::ResolvedField) -> ColSrc {
        ColSrc {
            path,
            ty: ty.to_string(),
            declared: f.declared.name().to_string(),
            date_fmt: if ty.starts_with("date") { Some(f.date_format.clone()) } else { None },
            ignore_above: f.ignore_above,
            normalizer: f.normalizer.clone(),
            constants: HashMap::new(),
            scaling_factor: f.scaling_factor,
            doc_values: f.doc_values,
            types: Vec::new(),
            per_index: HashMap::new(),
            dimension: f.time_series_dimension,
            ordered: Vec::new(),
            template_id: false,
        }
    }

    /// The column type: `unsupported` columns carry their mapped types
    /// (`unsupported:keyword,long`) for errors and `original_types`.
    fn col_type(&self) -> String {
        if self.ty != "unsupported" || self.types.is_empty() {
            return self.ty.clone();
        }
        format!("unsupported:{}", self.types.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join(","))
    }

    fn copy(&self) -> ColSrc {
        ColSrc {
            path: self.path.clone(),
            ty: self.ty.clone(),
            declared: self.declared.clone(),
            date_fmt: self.date_fmt.clone(),
            ignore_above: self.ignore_above,
            normalizer: self.normalizer.clone(),
            constants: self.constants.clone(),
            scaling_factor: self.scaling_factor,
            doc_values: self.doc_values,
            types: self.types.clone(),
            per_index: HashMap::new(),
            dimension: self.dimension,
            ordered: self.ordered.clone(),
            template_id: self.template_id,
        }
    }
}

/// A double as the field's doc values store it.
fn round_stored(d: f64, c: &ColSrc) -> f64 {
    match c.declared.as_str() {
        "float" => d as f32 as f64,
        "half_float" => crate::values::to_half_float(d),
        "scaled_float" if c.scaling_factor > 0.0 => (d * c.scaling_factor).round() / c.scaling_factor,
        _ => d,
    }
}

fn conv(v: &Value, c: &ColSrc) -> Option<Val> {
    Some(match c.ty.as_str() {
        "unsigned_long" => match v {
            Value::Number(n) => match n.as_i64() {
                Some(i) => Val::Long(i),
                // Beyond i64: kept as an (exactly printed) double.
                None => Val::Double(n.as_f64()?),
            },
            Value::String(s) => match s.trim().parse::<i64>() {
                Ok(i) => Val::Long(i),
                Err(_) => Val::Double(s.trim().parse::<f64>().ok()?),
            },
            _ => return None,
        },
        "long" | "counter_long" => match v {
            Value::Number(n) => Val::Long(n.as_i64().or_else(|| n.as_f64().map(|f| f as i64))?),
            Value::String(s) => Val::Long(s.trim().parse::<f64>().ok()? as i64),
            _ => return None,
        },
        "integer" | "counter_integer" => match v {
            Value::Number(n) => Val::Int(n.as_i64().or_else(|| n.as_f64().map(|f| f as i64))?),
            Value::String(s) => Val::Int(s.trim().parse::<f64>().ok()? as i64),
            _ => return None,
        },
        "double" | "counter_double" => match v {
            Value::Number(n) => Val::Double(round_stored(n.as_f64()?, c)),
            Value::String(s) => Val::Double(round_stored(s.trim().parse().ok()?, c)),
            _ => return None,
        },
        "keyword" => {
            let s = match v {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => return None,
            };
            if let Some(ia) = c.ignore_above {
                if s.chars().count() > ia {
                    return None;
                }
            }
            let s = match &c.normalizer {
                Some(n) => n.analyze(&s).into_iter().map(|t| t.text).collect::<Vec<_>>().join(""),
                None => s,
            };
            Val::Str(s)
        }
        "text" | "ip" | "version" => match v {
            Value::String(s) => Val::Str(s.clone()),
            Value::Number(n) => Val::Str(n.to_string()),
            Value::Bool(b) => Val::Str(b.to_string()),
            _ => return None,
        },
        "boolean" => match v {
            Value::Bool(b) => Val::Bool(*b),
            Value::String(s) if s == "true" => Val::Bool(true),
            Value::String(s) if s == "false" || s.is_empty() => Val::Bool(false),
            _ => return None,
        },
        "date" | "date_nanos" => match v {
            Value::String(s) => match &c.date_fmt {
                Some(f) => Val::Date(f.parse(s).ok()?),
                None => Val::Date(func::parse_date_str(s)?),
            },
            Value::Number(n) => match &c.date_fmt {
                Some(f) if f.accepts_numbers() => Val::Date(f.parse_number(n.as_f64()?).ok()?),
                _ => Val::Date(n.as_f64()? as i64),
            },
            _ => return None,
        },
        t if super::spatial::is_spatial(t) => Val::Str(super::spatial::from_source(v, t)?),
        "aggregate_metric_double" => match v {
            Value::Object(m) => Val::Str(super::types::amd_json(&super::types::Amd::from_json(m))),
            _ => return None,
        },
        "histogram" => match v {
            Value::Object(m) => Val::Str(super::types::histogram_json(m)),
            _ => return None,
        },
        "flattened" => match v {
            Value::Object(_) => Val::Str(super::types::flattened_json(v)),
            Value::Null => return None,
            other => Val::Str(super::types::flattened_json(other)),
        },
        "dense_vector" => match v {
            Value::Number(n) => Val::Double(round_stored(n.as_f64()?, c)),
            _ => return None,
        },
        _ => return None,
    })
}

fn extract(src: &Value, c: &ColSrc, index: &str) -> Val {
    if let Some(cv) = c.constants.get(index) {
        return conv(cv, c.per_index.get(index).unwrap_or(c)).unwrap_or(Val::Null);
    }
    // Union types: each index's documents read with that index's mapping.
    if let Some(pc) = c.per_index.get(index) {
        return if pc.ty == "unsupported" { Val::Null } else { extract_ordered(src, pc, pc.ordered.iter().any(|i| i == index)) };
    }
    if c.ty == "unsupported" {
        return Val::Null;
    }
    // Documents of indices that don't map the field have no value for it.
    if !c.types.is_empty() && !index.is_empty() && !c.types.iter().any(|(_, ix)| ix.iter().any(|i| i == index)) {
        return Val::Null;
    }
    extract_ordered(src, c, c.ordered.iter().any(|i| i == index))
}

fn extract_one(src: &Value, c: &ColSrc) -> Val {
    extract_ordered(src, c, false)
}

/// `in_order`: the document's array order survives (strict columnar doc values).
fn extract_ordered(src: &Value, c: &ColSrc, in_order: bool) -> Val {
    // Points: an array of two (or three) numbers is one `[x, y]` point, at
    // the top level or inside an array of points, as the geo_point mapper
    // parses it (`[{"lat": .., "lon": ..}, [x, y]]` is two points).
    let vals = if c.ty == "geo_point" || c.ty == "cartesian_point" { point_source_values(src, &c.path) } else { crate::highlight::source_values(src, &c.path) };
    extract_values(vals, c, in_order)
}

/// Whether [`extract_ordered`] reads column `c` with
/// [`crate::highlight::source_values`] (every type but points).
fn reads_source_values(c: &ColSrc) -> bool {
    c.ty != "geo_point" && c.ty != "cartesian_point"
}

/// [`extract_ordered`] of the values `source_values` found for `c`.
fn extract_values(vals: Vec<&Value>, c: &ColSrc, in_order: bool) -> Val {
    // Whole-value types read the field's `_source` value as one.
    if matches!(c.ty.as_str(), "aggregate_metric_double" | "histogram" | "flattened") {
        return vals.first().and_then(|v| conv(v, c)).unwrap_or(Val::Null);
    }
    let mut out: Vec<Val> = vals.iter().filter_map(|v| conv(v, c)).collect();
    if c.template_id {
        out = out.into_iter().map(|v| Val::Str(crate::fieldtypes::pattern_text::template_id(&v.to_text()))).collect();
    }
    // Doc values come back sorted (and de-duplicated for keyword-like
    // fields); `_source`-loaded fields keep the document order.
    // Text reads its doc values only when they are enabled explicitly.
    let sortable = !matches!(c.ty.as_str(), "geo_point" | "cartesian_point" | "geo_shape" | "cartesian_shape" | "dense_vector") && (c.ty != "text" || matches!(c.declared.as_str(), "text" | "match_only_text"));
    if out.len() > 1 && c.doc_values && sortable && !in_order {
        if c.ty == "version" {
            out.sort_by_cached_key(|v| rustic_core::fieldtypes::version::encode(&v.to_text()));
        } else {
            out.sort_by(cmp_vals);
        }
        if matches!(c.ty.as_str(), "keyword" | "ip" | "version") {
            out.dedup();
        }
    }
    Val::from_list(out)
}

/// The `_source` values of a point field at `path`: like
/// `highlight::source_values`, but an array of 2 or 3 numbers stays one
/// value (a `[x, y]` point) wherever it appears among the values.
fn point_source_values<'v>(src: &'v Value, path: &str) -> Vec<&'v Value> {
    fn leaf<'v>(v: &'v Value, out: &mut Vec<&'v Value>) {
        match v {
            Value::Array(a) if crate::values::is_point_array(a) => out.push(v),
            Value::Array(a) => a.iter().for_each(|x| leaf(x, out)),
            Value::Null => {}
            other => out.push(other),
        }
    }
    fn walk<'v>(v: &'v Value, parts: &[&str], out: &mut Vec<&'v Value>) {
        if parts.is_empty() {
            leaf(v, out);
            return;
        }
        match v {
            Value::Object(m) => {
                // Dotted keys at any level, as `source_values`.
                for i in (1..=parts.len()).rev() {
                    if let Some(c) = m.get(&parts[..i].join(".")) {
                        walk(c, &parts[i..], out);
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, parts, out)),
            _ => {}
        }
    }
    let parts: Vec<&str> = path.split('.').collect();
    let mut out = Vec::new();
    walk(src, &parts, &mut out);
    out
}

/// Mapping metadata fields (never ES|QL columns; `METADATA` loads some).
fn is_metadata_field(name: &str) -> bool {
    matches!(
        name,
        "_id" | "_index" | "_routing" | "_source" | "_seq_no" | "_primary_term" | "_version" | "_ignored" | "_ignored_source" | "_field_names" | "_tier" | "_doc_count"
            | "_data_stream_timestamp" | "_tsid" | "_ts_routing_hash" | "_nested_path" | "_feature" | "_index_mode" | "_inference_fields" | "_size" | "_recovery_source"
            | "_recovery_source_size" | "_score" | "_type" | "_uid" | "_parent" | "_timestamp" | "_ttl" | "_all"
    )
}

/// Marker "index" of the `_source` leg of an `unmapped_fields=load` type conflict.
const SOURCE_LEG: &str = "\u{0}source";

/// ES|QL `DataType.typeName()` of a type, as conflict messages and
/// `original_types` print it (`date` is `datetime`).
pub fn type_label(ty: &str) -> String {
    match ty {
        "date" => "datetime".into(),
        t => t.to_string(),
    }
}

struct IndexSchema {
    cols: Vec<(String, ColSrc)>,
    targets: Vec<crate::search::Target>,
    /// Cluster: the wanted shards in search order when some are read on
    /// other nodes (`None`: `targets` are all read here).
    slots: Option<Vec<super::remote::Slot>>,
    /// The columns the pipeline can read ([`project_columns`], ES's field
    /// extraction after `PruneColumns`): the others load as null, and are
    /// never read from doc values or `_source`. `None`: every column.
    projection: Option<HashSet<String>>,
}

fn resolve_schema(ctx: &Ctx, indices: &[String], pos: usize) -> Result<IndexSchema> {
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let expr = indices.join(",");
    let mut opts = ResolveOpts::from_params(&HashMap::new());
    opts.allow_no_indices = true;
    // FROM accepts cross-cluster expressions: plan the remote parts.
    let targets = match crate::ccr::remote::resolve_local_part(node, &expr, opts).map(|(targets, _)| targets) {
        Ok(t) => t,
        // A list with a missing concrete index is the plain index_not_found 404; a single pattern is a verification error.
        Err(e) if e.kind == "index_not_found_exception" && indices.len() == 1 => {
            return Err(verification_err(ctx.full, pos, format!("Unknown index [{expr}]")));
        }
        Err(e) => return Err(e),
    };
    if targets.is_empty() {
        return Err(verification_err(ctx.full, pos, format!("Unknown index [{expr}]")));
    }
    let (targets, slots) = match &ctx.shards {
        Some(source) => {
            let split = source.split(targets)?;
            (split.targets, split.slots)
        }
        None => (targets, None),
    };
    // Document- and field-level security: every read honours the target's access.
    let targets = match &ctx.restrict {
        Some(r) => r(targets)?,
        None => targets,
    };
    let mut map: BTreeMap<String, ColSrc> = BTreeMap::new();
    // Names that are objects (or flattened sub-paths) in some index.
    let mut objects: HashSet<String> = HashSet::new();
    for t in &targets {
        let lk = t.svc.lookup();
        for (name, f) in &lk.fields {
            if matches!(f.ty, EsType::Object | EsType::Nested) {
                objects.insert(name.clone());
            }
        }
        objects.extend(lk.objects.keys().cloned());
    }
    for t in &targets {
        let lk = t.svc.lookup();
        let tsdb = t.svc.setting("index.mode").is_some_and(|m| m.eq_ignore_ascii_case("time_series"));
        let mut parent: HashMap<String, String> = HashMap::new();
        for (name, f) in &lk.fields {
            for mf in &f.multi_fields {
                parent.insert(mf.clone(), name.clone());
            }
        }
        let fls = t.field_access();
        let strict = t.svc.setting("index.mode").is_some_and(|m| matches!(m.to_ascii_lowercase().as_str(), "columnar" | "logsdb_columnar"));
        for (name, f) in &lk.fields {
            if is_metadata_field(name) {
                continue;
            }
            if fls.is_some_and(|fa| !fa.allows(name)) {
                continue;
            }
            // Fields inside `nested` objects are invisible to ES|QL.
            if f.nested_path.is_some() || f.ty == EsType::Nested {
                continue;
            }
            // Aliases read their target.
            let (f, target) = if f.ty == EsType::Alias {
                match lk.get(name) {
                    Some(t) if t.ty != EsType::Alias => (t.as_ref(), t.name.clone()),
                    _ => continue,
                }
            } else {
                (f.as_ref(), name.clone())
            };
            let Some(ty) = field_esql_type(f) else { continue };
            // Counter metrics of time-series indices are their own types.
            let ty = match (ty, f.time_series_metric.as_deref()) {
                ("long", Some("counter")) if tsdb => "counter_long",
                ("integer", Some("counter")) if tsdb => "counter_integer",
                ("double", Some("counter")) if tsdb => "counter_double",
                (t, _) => t,
            };
            // `pattern_text`'s `template_id` reads the message.
            let pattern_parent = target.strip_suffix(".template_id").filter(|p| lk.fields.get(*p).is_some_and(|pf| pf.declared == EsType::PatternText)).map(str::to_string);
            let path = pattern_parent.clone().or_else(|| parent.get(&target).cloned()).unwrap_or(target);
            let idx = t.svc.name.clone();
            let label = type_label(if ty == "unsupported" { f.declared.name() } else { ty });
            let mut this = ColSrc::of(path.clone(), ty, f);
            this.template_id = pattern_parent.is_some();
            if strict {
                this.ordered.push(idx.clone());
            }
            let c = map.entry(name.clone()).or_insert_with(|| ColSrc::of(path, ty, f));
            c.template_id |= this.template_id;
            if strict && !c.ordered.contains(&idx) {
                c.ordered.push(idx.clone());
            }
            if c.ty != ty {
                c.ty = "unsupported".into();
            }
            if let Some(cv) = &f.constant_value {
                c.constants.insert(idx.clone(), cv.clone());
            }
            match c.types.iter_mut().find(|(t, _)| *t == label) {
                Some((_, ix)) => ix.push(idx.clone()),
                None => c.types.push((label, vec![idx.clone()])),
            }
            c.per_index.insert(idx, this);
        }
    }
    // A field in one index that is an object in another cannot be used.
    for (name, c) in map.iter_mut() {
        if objects.contains(name) && !c.types.is_empty() {
            c.ty = "unsupported".into();
            if !c.types.iter().any(|(t, _)| t == "object") {
                c.types.push(("object".into(), vec![]));
            }
        }
    }
    let load = ctx.settings.unmapped == super::settings::Unmapped::Load;
    let all_idx: Vec<String> = {
        let mut v: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
        v.sort();
        v.dedup();
        v
    };
    for (name, c) in map.iter_mut() {
        c.types.sort();
        for (_, ix) in c.types.iter_mut() {
            ix.sort();
            ix.dedup();
        }
        // `unmapped_fields=load`: indices that don't map the field read it from `_source`
        // as a keyword. A keyword column takes those values; any other single type
        // keeps nulls there; several mapped types conflict with the `_source` leg too.
        let missing: Vec<String> = if load { all_idx.iter().filter(|i| !c.types.iter().any(|(_, ix)| ix.contains(i))).cloned().collect() } else { vec![] };
        let mut keep_per_index = false;
        if !missing.is_empty() {
            if c.types.len() == 1 && c.ty == "keyword" {
                for i in &missing {
                    let mut src = ColSrc::plain(name, "keyword");
                    src.doc_values = false;
                    c.per_index.insert(i.clone(), src);
                }
                keep_per_index = true;
            } else if c.types.len() >= 2 {
                c.ty = "unsupported".into();
                c.types.insert(0, ("keyword".into(), vec![SOURCE_LEG.into()]));
            }
        }
        // `date` and `date_nanos` union to `date_nanos` (9.x implicit casting).
        let labels: Vec<&str> = c.types.iter().map(|(t, _)| t.as_str()).collect();
        if labels == ["date_nanos", "datetime"] {
            c.ty = "date_nanos".into();
            keep_per_index = true;
        }
        // Single-mapping columns read through the merged source.
        if c.types.len() < 2 && c.ty != "unsupported" && !keep_per_index {
            c.per_index.clear();
        }
    }
    Ok(IndexSchema { cols: map.into_iter().collect(), targets, slots, projection: None })
}

/// Index columns a query can read, or `None` when any column may reach the
/// result: the schema's [`IndexSchema::projection`] (the pipeline's own
/// analysis), narrowed by what the query text references
/// ([`text_needed_columns`]). Both are supersets of what the query reads,
/// and so is their intersection.
fn needed_columns(full: &str, schema: &IndexSchema) -> Option<std::collections::HashSet<String>> {
    match (&schema.projection, text_needed_columns(full, schema)) {
        (Some(projection), Some(text)) => Some(text.into_iter().filter(|name| projection.contains(name)).collect()),
        (Some(projection), None) => Some(projection.clone()),
        (None, text) => text,
    }
}

/// Index columns a query text can read, or `None` when any column may reach
/// the result. Only queries with a plain `STATS` reduce the table to what they
/// reference; the match is a superset (any identifier-like token naming the
/// column, a dotted prefix or extension of it, or a wildcard matching it).
fn text_needed_columns(full: &str, schema: &IndexSchema) -> Option<std::collections::HashSet<String>> {
    // SQL's explicit projection and expression dependencies identify every
    // source column the interpreter can read, including filters and sort keys
    // that do not appear in the output. Keep schema columns for validation,
    // but do not fetch/convert unrelated values from wide documents.
    if !full.contains('?') {
        if let Ok(crate::sql::Statement::Select(plan)) = crate::sql::parse(full, &Params::default()) {
            if !matches!(plan.cmds.last(), Some(Cmd::Keep(..))) { return None; }
            fn col_names(expr: &Expr, names: &mut Vec<String>) {
                let mut refs = Vec::new();
                super::analyze::expr_refs(expr, &mut refs);
                names.extend(refs.into_iter().map(|(name, _)| name));
            }
            let mut refs = Vec::new();
            for cmd in &plan.cmds {
                match cmd {
                    Cmd::From { .. } | Cmd::Limit(_) => {},
                    Cmd::Where(expr, _) => col_names(expr, &mut refs),
                    Cmd::Eval(fields) => for field in fields {
                        col_names(&field.expr, &mut refs);
                        if let Some(expr) = &field.filter { col_names(expr, &mut refs); }
                    },
                    Cmd::Stats { aggs, by, .. } => for field in aggs.iter().chain(by) {
                        col_names(&field.expr, &mut refs);
                        if let Some(expr) = &field.filter { col_names(expr, &mut refs); }
                    },
                    Cmd::Sort(keys) => for key in keys { col_names(&key.expr, &mut refs); },
                    Cmd::Keep(names, _) => {
                        if names.iter().any(|n| n.contains('*')) { return None; }
                        refs.extend(names.iter().cloned());
                    },
                    _ => return None,
                }
            }
            return Some(schema.cols.iter().filter(|(name, _)| refs.contains(name)).map(|(name, _)| name.clone()).collect());
        }
    }
    let up = full.to_ascii_uppercase();
    // Parameters may name fields (`?f`), so the query text doesn't list them all.
    if !up.contains("STATS") || up.contains("INLINE") || up.contains("FORK") || up.contains("LOOKUP") || up.contains("ENRICH") || full.contains('?') {
        return None;
    }
    let toks: Vec<&str> = full.split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '.' | '@' | '*' | '-'))).filter(|t| !t.is_empty() && *t != "*").collect();
    let mut need = std::collections::HashSet::new();
    for (name, _) in &schema.cols {
        let hit = toks.iter().any(|t| {
            *t == name
                || name.strip_prefix(*t).is_some_and(|r| r.starts_with('.'))
                || t.strip_prefix(name.as_str()).is_some_and(|r| r.starts_with('.'))
                || (t.contains('*') && crate::mapping::wildcard_match(t, name))
        });
        if hit {
            need.insert(name.clone());
        }
    }
    Some(need)
}

/// Load documents from `indices` into a table.
/// Prefix of the hidden boolean columns holding per-document full-text matches.
const FT_PROBE: &str = "\u{0}ft";

/// Replace each full-text call of `e` by a hidden column that `load` fills
/// with whether the document matches the call's query. Sets `failed` when a
/// call cannot be expressed as a query.
fn probe_fulltext(e: &Expr, types: &HashMap<String, String>, probes: &mut Vec<Value>, failed: &mut bool) -> Expr {
    macro_rules! rec {
        ($x:expr) => {
            probe_fulltext($x, types, probes, failed)
        };
    }
    match e {
        Expr::Call(n, _, p) if FULL_TEXT.contains(&n.as_str()) => match to_dsl(e, types, true) {
            Some((q, _)) => {
                probes.push(q);
                Expr::Col(format!("{FT_PROBE}{}", probes.len() - 1), *p)
            }
            None => {
                *failed = true;
                e.clone()
            }
        },
        Expr::Call(n, _, _) if n == "SCORE" => {
            *failed = true;
            e.clone()
        }
        Expr::Call(n, args, p) => Expr::Call(n.clone(), args.iter().map(|a| rec!(a)).collect(), *p),
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(rec!(a)), Box::new(rec!(b))),
        Expr::Not(a) => Expr::Not(Box::new(rec!(a))),
        Expr::Neg(a) => Expr::Neg(Box::new(rec!(a))),
        Expr::IsNull(a, n) => Expr::IsNull(Box::new(rec!(a)), *n),
        Expr::Cast(a, t) => Expr::Cast(Box::new(rec!(a)), t.clone()),
        o => o.clone(),
    }
}

/// The search `FROM` runs (without `size`): the pushed-down query and sort,
/// reading the `_source` paths of the needed columns.
fn search_body(schema: &IndexSchema, metadata: &[String], needed: &Option<HashSet<String>>, query: &Value, sort: Vec<Value>) -> Value {
    let source = match needed {
        Some(n) if n.is_empty() && !metadata.iter().any(|m| m == "_source") => json!(false),
        // Columns read their source path (aliases and multi-fields read another field).
        Some(n) if !metadata.iter().any(|m| m == "_source") => {
            let mut paths: Vec<String> = schema.cols.iter().filter(|(name, _)| n.contains(name)).flat_map(|(_, c)| std::iter::once(c.path.clone()).chain(c.per_index.values().map(|p| p.path.clone()))).collect();
            paths.sort();
            paths.dedup();
            json!(paths)
        }
        _ => json!(true),
    };
    let mut body = json!({"query": query.clone(), "_source": source, "track_total_hits": false});
    if !sort.is_empty() {
        body["sort"] = Value::Array(sort);
    }
    if metadata.iter().any(|m| m == "_score") {
        body["track_scores"] = json!(true);
    }
    if metadata.iter().any(|m| m == "_version") {
        body["version"] = json!(true);
    }
    body
}

/// The index and metadata columns `FROM` loads.
fn source_cols(schema: &IndexSchema, metadata: &[String]) -> Vec<Column> {
    let mut cols: Vec<Column> = schema.cols.iter().map(|(n, c)| Column { name: n.clone(), ty: c.col_type() }).collect();
    for m in metadata {
        let ty = match m.as_str() {
            "_score" => "double",
            "_version" => "long",
            "_source" => "_source",
            "_tsid" => "_tsid",
            _ => "keyword",
        };
        cols.push(Column { name: m.clone(), ty: ty.into() });
    }
    cols
}

/// Column types partial `STATS` states can read.
const PARTIAL_TYPES: &[&str] = &["integer", "long", "double", "keyword", "date", "date_nanos", "boolean", "ip", "version"];

/// The partial aggregation plan of a `STATS` directly after `FROM` (and the
/// `WHERE`s pushed into its search), when every aggregate is decomposable
/// ([`super::partial`]): plain columns, no filters, no expressions.
fn agg_plan(ctx: &Ctx, schema: &IndexSchema, indices: &[String], cmd: &Cmd) -> Option<super::partial::AggPlan> {
    use super::partial::{AggKind, AggPlan, AggSpec};
    let Cmd::Stats { aggs, by, inline: false, .. } = cmd else { return None };
    if !ctx.esql || ctx.ts {
        return None;
    }
    let mut types: Vec<(String, String)> = Vec::new();
    let mut read = |n: &str, numeric: bool| -> Option<()> {
        let (_, c) = schema.cols.iter().find(|(c, _)| c == n)?;
        let ty = c.col_type();
        if !PARTIAL_TYPES.contains(&ty.as_str()) || !c.per_index.is_empty() || (numeric && !matches!(ty.as_str(), "integer" | "long" | "double")) {
            return None;
        }
        if !types.iter().any(|(t, _)| t == n) {
            types.push((n.to_string(), ty));
        }
        Some(())
    };
    let mut names: HashSet<&str> = HashSet::new();
    let mut plan_by = Vec::new();
    for g in by {
        match &g.expr {
            Expr::Col(n, _) if *n == g.name && g.filter.is_none() && names.insert(&g.name) => {
                read(n, false)?;
                plan_by.push(n.clone());
            }
            _ => return None,
        }
    }
    let mut plan_aggs = Vec::new();
    for a in aggs {
        if a.filter.is_some() || !names.insert(&a.name) {
            return None;
        }
        let Expr::Call(name, args, _) = &a.expr else { return None };
        let kind = AggKind::of(name)?;
        let col = match (kind, args.as_slice()) {
            (AggKind::Count, [Expr::Star]) => None,
            (AggKind::CountDistinct, [Expr::Col(c, _)] | [Expr::Col(c, _), Expr::Lit(Val::Int(_))]) => Some(c.clone()),
            (_, [Expr::Col(c, _)]) => Some(c.clone()),
            _ => return None,
        };
        if let Some(c) = &col {
            read(c, matches!(kind, AggKind::Sum | AggKind::Avg))?;
        }
        // `AVG` of an integer column is a long sum over the count.
        let integer = col.as_ref().is_some_and(|c| schema.cols.iter().any(|(n, src)| n == c && src.col_type() == "integer"));
        let kind = if kind == AggKind::Avg && integer { AggKind::AvgInt } else { kind };
        plan_aggs.push(AggSpec { kind, col });
    }
    Some(AggPlan { indices: indices.to_vec(), unmapped_load: ctx.settings.unmapped == super::settings::Unmapped::Load, by: plan_by, aggs: plan_aggs, types })
}

/// One shard's partial `STATS` states: its matching documents, read for
/// the plan's columns (`srcs`, in `plan.types` order), grouped.
fn shard_partial(node: &Node, t: &crate::search::Target, query: &Value, body: &Value, plan: &super::partial::AggPlan, srcs: &[&ColSrc]) -> Result<super::remote::ShardResult> {
    let targets = std::slice::from_ref(t);
    let count = crate::search::count(targets, Some(query), None)?;
    // Only `COUNT(*)`s without groups: the match count is the whole state
    // (no document is read).
    if plan.by.is_empty() && plan.aggs.iter().all(|a| a.kind == super::partial::AggKind::Count && a.col.is_none()) {
        let states = plan.aggs.iter().map(|_| super::partial::AggState::Count(count as i64)).collect();
        return Ok(super::remote::ShardResult {
            index: t.svc.name.clone(),
            shard: t.shard.as_ref().map(|s| s.id).unwrap_or(0),
            count,
            hits: Vec::new(),
            probes: Vec::new(),
            groups: Some(vec![super::partial::PartialGroup { key: Vec::new(), states }]),
            loaded: 0,
            pit: false,
            fast: None,
            warnings: Vec::new(),
        });
    }
    let mut rows = Vec::with_capacity(count as usize);
    let mut loaded = 0u64;
    if count > 0 {
        let mut b = body.clone();
        b["size"] = json!(count);
        for h in super::remote::search_hits(node, targets, &b)? {
            let index = h["_index"].as_str().unwrap_or("");
            let row: Vec<Val> = srcs.iter().map(|c| extract(&h["_source"], c, index)).collect();
            loaded += row.iter().filter(|v| !v.is_null()).count() as u64;
            rows.push(row);
        }
    }
    let pos = |n: &str| plan.types.iter().position(|(c, _)| c == n).unwrap_or(0);
    let by: Vec<usize> = plan.by.iter().map(|b| pos(b)).collect();
    let aggs: Vec<(super::partial::AggKind, Option<usize>)> = plan.aggs.iter().map(|a| (a.kind, a.col.as_deref().map(pos))).collect();
    Ok(super::remote::ShardResult {
        index: t.svc.name.clone(),
        shard: t.shard.as_ref().map(|s| s.id).unwrap_or(0),
        count,
        hits: Vec::new(),
        probes: Vec::new(),
        groups: Some(super::partial::partial_groups(&rows, &by, &aggs)),
        loaded,
        pit: false,
        fast: None,
        warnings: Vec::new(),
    })
}

/// The `_source` paths of the plan's columns, read the way `load` does.
fn plan_srcs<'s>(schema: &'s IndexSchema, plan: &super::partial::AggPlan) -> Option<Vec<&'s ColSrc>> {
    plan.types.iter().map(|(n, ty)| schema.cols.iter().find(|(c, src)| c == n && src.col_type() == *ty).map(|(_, c)| c)).collect()
}

/// Execute a bounded grouped limit using the existing snapshot-backed local
/// stream. Only plain column keys and a final limit use this wire plan.
pub(super) fn serve_group_limit(node: &Node, spec: &super::remote::FetchSpec, plan: &super::remote::GroupLimitPlan, targets: &[crate::search::Target], source: Option<&super::remote::SharedShardSource>, restrict: Option<&Restrict>) -> Result<Vec<super::remote::ShardResult>> {
    let mut ctx = new_ctx(node, "", None, plan.cap, plan.cap);
    ctx.esql = true;
    ctx.restrict = restrict.cloned();
    ctx.shards = source.cloned();
    let mut schema = resolve_schema(&ctx, &plan.indices, 0)?;
    let columns: Vec<_> = source_cols(&schema, &[]).into_iter().map(|c| (c.name, c.ty)).collect();
    if columns != plan.columns {
        return Err(EsError::new(409, "illegal_state_exception", "ES|QL grouped-limit columns differ on this node"));
    }
    let commands = [Cmd::LimitBy(plan.per_group, plan.keys.iter().map(|k| Expr::Col(k.clone(), 0)).collect()), Cmd::Limit(plan.cap)];
    schema.slots = None;
    let mut results = Vec::with_capacity(targets.len());
    for target in targets {
        schema.targets = vec![target.clone()];
        let (table, _) = fast::streamed_limit(&mut ctx, &schema, &spec.query, &[], &commands)?
            .ok_or_else(|| EsError::internal("grouped-limit stream unavailable"))?;
        let hits = table.rows.iter().map(|row| serde_json::to_value(row.iter().map(super::partial::WireVal::from).collect::<Vec<_>>())
            .map_err(|e| EsError::internal(e.to_string()))).collect::<Result<Vec<_>>>()?;
        results.push(super::remote::ShardResult { index: target.svc.name.clone(), shard: target.shard.as_ref().map(|s| s.id).unwrap_or(0), count: table.rows.len() as u64, hits, probes: vec![], groups: None, loaded: 0, pit: false, fast: None, warnings: Vec::new() });
    }
    Ok(results)
}

/// For source-order LIMIT n BY keys | LIMIT N, each shard need return only
/// its first N locally retained rows. Prior shards can consume at most R
/// quota slots when R rows have been retained globally, leaving at least
/// N-R eligible rows from the next shard. Fetch in slot order and stop as
/// soon as N rows survive; never materialize the complete source table.
fn distributed_group_limit(ctx: &mut Ctx, schema: &IndexSchema, indices: &[String], query: &Value, rest: &[Cmd]) -> Result<Option<Table>> {
    use super::remote::{FetchSpec, GroupLimitPlan, Slot};
    let (per_group, keys, cap) = match rest {
        [Cmd::LimitBy(per_group, keys), Cmd::Limit(cap)] => (*per_group, keys, *cap),
        [Cmd::LimitBy(per_group, keys)] => (*per_group, keys, ctx.max_limit),
        _ => return Ok(None),
    };
    let Some(slots) = &schema.slots else { return Ok(None) };
    let Some(source) = ctx.shards.clone() else { return Ok(None) };
    if ctx.settings.unmapped != super::settings::Unmapped::Default { return Ok(None); }
    let names = keys.iter().map(|e| match e { Expr::Col(n, _) => Some(n.clone()), _ => None }).collect::<Option<Vec<_>>>();
    let Some(names) = names else { return Ok(None) };
    let mut out = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    // Binding validates keys even for a zero-row request.
    let bound = keys.iter().map(|e| bind(ctx, e, &out.cols)).collect::<Result<Vec<_>>>()?;
    let cap = cap.min(ctx.max_limit);
    if cap == 0 || per_group == 0 { return Ok(Some(out)); }
    let plan = GroupLimitPlan { indices: indices.to_vec(), columns: out.cols.iter().map(|c| (c.name.clone(), c.ty.clone())).collect(), keys: names, per_group, cap };
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let mut output_memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
    let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
    for slot in slots {
        let mut fetch_memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
        let mut spec = FetchSpec { shards: vec![], query: query.clone(), body: json!({}), size: None, probes: vec![], agg: None, group_limit: Some(plan.clone()), pit: None, fast: None };
        let results = match slot {
            Slot::Local(i) => serve_group_limit(node, &spec, &plan, std::slice::from_ref(&schema.targets[*i]), Some(&source), ctx.restrict.as_ref())?,
            Slot::Remote(shard) => {
                spec.shards.push(shard.clone());
                let fetched = source.fetch(&spec,&mut fetch_memory)?;
                super::remote::in_slot_order(std::slice::from_ref(slot), |_| Err(EsError::internal("unexpected local grouped limit")), fetched)?
            }
        };
        for result in results {
            for hit in result.hits {
                let wire: Vec<super::partial::WireVal> = serde_json::from_value(hit).map_err(|e| EsError::internal(format!("invalid grouped-limit row: {e}")))?;
                let row: Vec<Val> = wire.iter().map(Val::from).collect();
                if row.len() != out.cols.len() { return Err(EsError::internal("grouped-limit row width differs")); }
                let key = bound.iter().map(|e| eval(e, &row, &ctx.fctx).map(|v| v.key())).collect::<Result<Vec<_>>>()?;
                let count = counts.entry(key).or_insert(0);
                if *count < per_group {
                    *count += 1;
                    output_memory.grow(row_bytes(&row).saturating_mul(2),ESQL_LABEL)?;
                    out.rows.push(row);
                    if out.rows.len() == cap { output_memory.retain_for_request();return Ok(Some(out)); }
                }
            }
        }
    }
    output_memory.retain_for_request();
    Ok(Some(out))
}

/// A data node's part of a columnar fast path over its copies of `targets`
/// (a fetch with a [`super::remote::FastPlan`]).
pub(super) fn serve_fast(node: &Node, spec: &super::remote::FetchSpec, plan: &super::remote::FastPlan, targets: &[crate::search::Target], source: Option<&super::remote::SharedShardSource>, restrict: Option<&Restrict>) -> Result<Vec<super::remote::ShardResult>> {
    fast::serve(node, spec, plan, targets, source, restrict)
}

/// A data node's partial `STATS` over its copies of `targets` (a fetch with
/// an aggregation plan): the schema is resolved here the same way.
pub(super) fn serve_partial(node: &Node, spec: &super::remote::FetchSpec, plan: &super::partial::AggPlan, targets: &[crate::search::Target], source: Option<&super::remote::SharedShardSource>, restrict: Option<&Restrict>) -> Result<Vec<super::remote::ShardResult>> {
    let mut ctx = new_ctx(node, "", None, 1000, 10_000);
    ctx.esql = true;
    ctx.restrict = restrict.cloned();
    ctx.shards = source.cloned();
    if plan.unmapped_load {
        ctx.settings.unmapped = super::settings::Unmapped::Load;
    }
    let schema = resolve_schema(&ctx, &plan.indices, 0)?;
    let srcs = plan_srcs(&schema, plan).ok_or_else(|| EsError::new(409, "illegal_state_exception", "ES|QL partial aggregation columns differ on this node"))?;
    shard_partials(node, targets.iter().collect(), &spec.query, &spec.body, plan, &srcs)
}

/// [`shard_partial`] of each target, one thread per shard; results in order.
fn shard_partials(node: &Node, targets: Vec<&crate::search::Target>, query: &Value, body: &Value, plan: &super::partial::AggPlan, srcs: &[&ColSrc]) -> Result<Vec<super::remote::ShardResult>> {
    if targets.len() <= 1 {
        return targets.into_iter().map(|t| shard_partial(node, t, query, body, plan, srcs)).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = targets.into_iter().map(|t| scope.spawn(move || shard_partial(node, t, query, body, plan, srcs))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(EsError::internal("partial aggregation panicked")))).collect()
    })
}

/// `STATS` over shards partly read on other nodes, from per-shard partial
/// states (`None`: a node could not aggregate; the caller loads rows).
fn distributed_stats(ctx: &mut Ctx, schema: &IndexSchema, metadata: &[String], query: &Value, slots: &[super::remote::Slot], plan: super::partial::AggPlan, cmd: &Cmd) -> Result<Option<Table>> {
    use super::remote::{in_slot_order, FetchSpec, Slot};
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let Some(shards) = ctx.shards.clone() else { return Ok(None) };
    let Some(srcs) = plan_srcs(schema, &plan) else { return Ok(None) };
    let needed: HashSet<String> = plan.types.iter().map(|(n, _)| n.clone()).collect();
    let body = search_body(schema, &[], &Some(needed), query, Vec::new());
    let remote: Vec<_> = slots.iter().filter_map(|s| match s { Slot::Remote(r) => Some(r.clone()), Slot::Local(_) => None }).collect();
    // Remote shards aggregate on their nodes while the local ones run here.
    let local: Vec<usize> = slots.iter().filter_map(|s| match s { Slot::Local(i) => Some(*i), Slot::Remote(_) => None }).collect();
    let spec = FetchSpec { shards: remote, query: query.clone(), body: body.clone(), size: None, probes: Vec::new(), agg: Some(plan.clone()), group_limit: None, pit: None, fast: None };
    let mut fetch_memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
    let (fetched, local_results) = std::thread::scope(|scope| {
        let remote = (!spec.shards.is_empty()).then(|| scope.spawn(|| shards.fetch(&spec,&mut fetch_memory)));
        let local_results = shard_partials(node, local.iter().map(|i| &schema.targets[*i]).collect(), query, &body, &plan, &srcs);
        (remote.map(|h| h.join().unwrap_or_else(|_| Err(EsError::internal("shard fetch panicked")))), local_results)
    });
    let fetched = match fetched {
        None => Vec::new(),
        Some(Ok(r)) if r.iter().all(|r| r.groups.is_some()) => r,
        Some(Err(error)) if error.status==429=>return Err(error),
        Some(_) => return Ok(None),
    };
    let mut local_results: HashMap<usize, super::remote::ShardResult> = local.into_iter().zip(local_results?).collect();
    let Ok(results) = in_slot_order(slots, |i| local_results.remove(&i).ok_or_else(|| EsError::internal("local shard not aggregated")), fetched) else { return Ok(None) };
    ctx.documents_found += results.iter().map(|r| r.count).sum::<u64>();
    ctx.values_loaded += results.iter().map(|r| r.loaded).sum::<u64>();
    let kinds: Vec<super::partial::AggKind> = plan.aggs.iter().map(|a| a.kind).collect();
    fetch_memory.grow(fetch_memory.bytes(),ESQL_LABEL)?;
    let merged = super::partial::merge_groups(results.into_iter().map(|r| r.groups.unwrap_or_default()).collect(), !plan.by.is_empty(), &kinds);
    // Output columns (names, types, checks): the `STATS` over no rows.
    let empty = Table { cols: source_cols(schema, metadata), rows: Vec::new() };
    let mut out = apply(ctx, cmd, empty)?;
    let Cmd::Stats { aggs, .. } = cmd else { return Ok(None) };
    let mut rows = Vec::with_capacity(merged.len());
    for g in &merged {
        let mut row = Vec::with_capacity(g.states.len() + g.key.len());
        for (k, st) in g.states.iter().enumerate() {
            let (v, overflowed) = st.finish(kinds[k]);
            if overflowed {
                if let Expr::Call(_, _, p) = &aggs[k].expr {
                    sum_overflow_warning(&ctx.fctx, *p);
                }
            }
            row.push(v);
        }
        row.extend(g.key.iter().map(Val::from));
        rows.push(row);
    }
    out.rows = rows;
    let output_memory=crate::memory::breakers().reserve_request(fast::rows_bytes(&out.rows),ESQL_LABEL)?;
    output_memory.retain_for_request();
    Ok(Some(out))
}

fn load(ctx: &mut Ctx, schema: &IndexSchema, metadata: &[String], query: Value, sort: Vec<Value>, size: Option<usize>, probes: &[Value]) -> Result<Table> {
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let mut cols = source_cols(schema, metadata);
    // SQL aggregates and computed sorts consume all matching rows. Read local
    // source in batches so each compressed store chunk is decoded once, and
    // avoid building/serializing/parsing a full search-hit JSON response.
    // Keep sorted/limited searches and remote slots on their existing planner.
    if !ctx.esql && !ctx.ts && metadata.is_empty() && probes.is_empty()
        && schema.slots.is_none() && sort.is_empty() && size.is_none()
    {
        if let Some(needed) = needed_columns(ctx.full, schema) {
            let wanted: Vec<bool> = schema.cols.iter().map(|(name, _)| needed.contains(name)).collect();
            let targets = node.snapshot(&schema.targets)?;
            let mut table = Table { cols, rows: Vec::new() };
            let mut memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
            crate::search::for_each_match(&targets, &query, 1024, &mut |page| {
                let mut ids = page.doc_ids();
                loop {
                    let docs: Vec<_> = ids.by_ref().take(1024).collect();
                    if docs.is_empty() { break; }
                    let (rows, loaded, _page_memory) = fast::sql_column_rows(schema, &wanted, page.target, page.lookup, page.leaf, &docs)?;
                    ctx.documents_found += rows.len() as u64;
                    ctx.values_loaded += loaded;
                    memory.grow(fast::rows_bytes(&rows),ESQL_LABEL)?;
                    table.rows.extend(rows);
                }
                Ok(true)
            })?;
            memory.retain_for_request();
            return Ok(table);
        }
    }
    // Index modes (`METADATA _index_mode`) and TS dimensions (`METADATA _tsid`).
    let modes: HashMap<String, String> = schema.targets.iter().map(|t| (t.svc.name.clone(), t.svc.setting("index.mode").map(|m| m.to_ascii_lowercase()).unwrap_or_else(|| "standard".into()))).collect();
    let tsid_dims: Vec<&(String, ColSrc)> = if metadata.iter().any(|m| m == "_tsid") { schema.cols.iter().filter(|(_, c)| c.dimension).collect() } else { vec![] };
    for k in 0..probes.len() {
        cols.push(Column { name: format!("{FT_PROBE}{k}"), ty: "boolean".into() });
    }
    // `TS`: each row's time series (its dimension values).
    let dims: Vec<&(String, ColSrc)> = if ctx.ts { schema.cols.iter().filter(|(_, c)| c.dimension).collect() } else { vec![] };
    if ctx.ts {
        cols.push(Column { name: super::ts::TSID.into(), ty: "keyword".into() });
    }
    if schema.targets.is_empty() {
        return Ok(Table { cols, rows: vec![] });
    }
    let needed;
    let hits: Vec<Value>;
    let probe_sets: Vec<HashSet<(String, String)>>;
    let mut fetch_memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
    if let Some(slots) = &schema.slots {
        // Some shards are read on other nodes: each shard runs the same
        // search wherever it is read, merged as one search over them all.
        use super::remote::{in_slot_order, merge, shard_result, FetchSpec, Slot};
        needed = if ctx.ts { None } else { needed_columns(ctx.full, schema) };
        let shards = ctx.shards.clone().ok_or_else(|| EsError::internal("no shard source"))?;
        let sorted = !sort.is_empty();
        let body = search_body(schema, metadata, &needed, &query, sort);
        let remote: Vec<_> = slots.iter().filter_map(|s| match s { Slot::Remote(r) => Some(r.clone()), Slot::Local(_) => None }).collect();
        let fetched = if remote.is_empty() { Vec::new() } else { shards.fetch(&FetchSpec { shards: remote, query: query.clone(), body: body.clone(), size, probes: probes.to_vec(), agg: None, group_limit: None, pit: None, fast: None },&mut fetch_memory)? };
        let results = in_slot_order(slots, |i| shard_result(node, &schema.targets[i], &query, &body, size, probes), fetched)?;
        let merged = merge(results, sorted, size, probes.len());
        hits = merged.hits;
        probe_sets = merged.probes;
    } else {
        let count = crate::search::count(&schema.targets, Some(&query), None)? as usize;
        let total = match size {
            Some(n) => n.min(count),
            None => count,
        };
        if total > 0 {
            needed = if ctx.ts { None } else { needed_columns(ctx.full, schema) };
            let mut body = search_body(schema, metadata, &needed, &query, sort);
            body["size"] = json!(total);
            hits = super::remote::search_hits(node, &schema.targets, &body)?;
            // Documents matching each full-text probe, as (index, id).
            probe_sets = probes.iter().map(|pq| super::remote::probe_set(node, &schema.targets, pq, &query).map(|ids| ids.into_iter().collect())).collect::<Result<_>>()?;
        } else {
            needed = None;
            hits = Vec::new();
            probe_sets = Vec::new();
        }
    }
    let mut rows = Vec::new();
    // The rows as they are built (the table is charged as a whole once
    // the pipeline holds it).
    let mut loading = crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
    let mut retained=0u64;
    {
        ctx.documents_found += hits.len() as u64;
        for h in hits {
            let scratch=crate::memory::json_bytes(&h).saturating_mul(256).saturating_add((cols.len() as u64).saturating_mul(128)).saturating_add(schema.cols.iter().flat_map(|(_,column)|column.constants.values()).map(crate::memory::json_bytes).fold(0u64,u64::saturating_add).saturating_mul(256));
            loading.grow(scratch,ESQL_LABEL)?;
            let src = &h["_source"];
            let mut row: Vec<Val> = Vec::with_capacity(cols.len());
            for (n, c) in &schema.cols {
                if needed.as_ref().is_some_and(|set| !set.contains(n)) {
                    row.push(Val::Null);
                    continue;
                }
                let v = extract(src, c, h["_index"].as_str().unwrap_or(""));
                if !v.is_null() {
                    ctx.values_loaded += 1;
                }
                row.push(v);
            }
            for m in metadata {
                row.push(match m.as_str() {
                    "_score" => h["_score"].as_f64().map(Val::Double).unwrap_or(Val::Double(0.0)),
                    "_version" => h["_version"].as_i64().map(Val::Long).unwrap_or(Val::Long(1)),
                    "_source" => {
                        if src.is_null() {
                            Val::Null
                        } else {
                            Val::Str(src.to_string())
                        }
                    }
                    "_ignored" => match &h["_ignored"] {
                        Value::Array(a) => Val::from_list(a.iter().filter_map(|x| x.as_str().map(|s| Val::Str(s.to_string()))).collect()),
                        _ => Val::Null,
                    },
                    "_index_mode" => Val::Str(modes.get(h["_index"].as_str().unwrap_or("")).cloned().unwrap_or_else(|| "standard".into())),
                    "_tsid" => {
                        let ix = h["_index"].as_str().unwrap_or("");
                        if modes.get(ix).map(String::as_str) != Some("time_series") {
                            Val::Null
                        } else {
                            let mut key = String::new();
                            for (n, c) in &tsid_dims {
                                let v = extract(src, c, ix);
                                if !v.is_null() {
                                    key.push_str(n);
                                    key.push('=');
                                    key.push_str(&v.to_text());
                                    key.push('\u{1}');
                                }
                            }
                            Val::Str(super::types::tsid_encode(&key))
                        }
                    }
                    other => h[other].as_str().map(|s| Val::Str(s.to_string())).unwrap_or(Val::Null),
                });
            }
            if !probe_sets.is_empty() {
                let key = (h["_index"].as_str().unwrap_or("").to_string(), h["_id"].as_str().unwrap_or("").to_string());
                row.extend(probe_sets.iter().map(|set| Val::Bool(set.contains(&key))));
            }
            if ctx.ts {
                let mut m = serde_json::Map::new();
                for (n, c) in &dims {
                    let v = extract(src, c, h["_index"].as_str().unwrap_or(""));
                    if !v.is_null() {
                        m.insert(n.clone(), v.to_json());
                    }
                }
                row.push(Val::Str(Value::Object(m).to_string()));
            }
            retained=retained.saturating_add(row_bytes(&row).saturating_mul(2));
            if retained>loading.bytes(){loading.grow(retained-loading.bytes(),ESQL_LABEL)?;}
            else{loading.shrink(loading.bytes()-retained);}
            rows.push(row);
        }
    }
    loading.retain_for_request();
    Ok(Table { cols, rows })
}

fn conjuncts(e: &Expr, out: &mut Vec<Expr>) {
    match e {
        Expr::Bin(BinOp::And, a, b) => {
            conjuncts(a, out);
            conjuncts(b, out);
        }
        o => out.push(o.clone()),
    }
}

fn has_fulltext(e: &Expr) -> bool {
    match e {
        Expr::Call(n, args, _) => FULL_TEXT.contains(&n.as_str()) || n == "SCORE" || args.iter().any(has_fulltext),
        Expr::Bin(_, a, b) => has_fulltext(a) || has_fulltext(b),
        Expr::Not(a) | Expr::Neg(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) => has_fulltext(a),
        _ => false,
    }
}

fn lit_json(v: &Val, ty: &str) -> Option<Value> {
    match (v, ty) {
        (Val::Null | Val::Multi(_) | Val::Dur(_) | Val::Period(..), _) => None,
        (Val::Date(d), t) if t.starts_with("date") => Some(json!(d)),
        (Val::Str(s), t) if t.starts_with("date") => func::parse_date_str(s).map(|d| json!(d)),
        (_, t) if t.starts_with("date") => None,
        (Val::Int(_) | Val::Long(_) | Val::Double(_), "long" | "integer" | "double" | "unsigned_long") => Some(v.to_json()),
        (Val::Str(_), "long" | "integer" | "double" | "unsigned_long") => None,
        (Val::Bool(_), "boolean") => Some(v.to_json()),
        (Val::Str(s), "keyword" | "ip" | "version") => Some(json!(s)),
        _ => None,
    }
}

fn field_of<'e>(e: &'e Expr, types: &HashMap<String, String>) -> Option<(&'e str, String)> {
    match e {
        Expr::Col(n, _) => types.get(n).map(|t| (n.as_str(), t.clone())),
        _ => None,
    }
}

fn exists_and(f: &str, q: Value) -> Value {
    json!({"bool": {"filter": [{"exists": {"field": f}}], "must_not": [q]}})
}

/// A constant spatial argument as WKT: a string literal, or `TO_GEOSHAPE` /
/// `TO_GEOPOINT` of one.
fn const_wkt(e: &Expr) -> Option<String> {
    match e {
        Expr::Lit(Val::Str(s)) => super::spatial::parse_wkt(s, "geo_shape"),
        Expr::Call(f, args, _) if (f == "TO_GEOSHAPE" || f == "TO_GEOPOINT") && args.len() == 1 => const_wkt(&args[0]),
        _ => None,
    }
}

/// `ST_DISTANCE(field, point)` (either order) of a `geo_point` field and a
/// constant point: the field and the point's `(lat, lon)`.
fn distance_of<'e>(e: &'e Expr, types: &HashMap<String, String>) -> Option<(&'e str, f64, f64)> {
    let Expr::Call(f, args, _) = e else { return None };
    if f != "ST_DISTANCE" || args.len() != 2 {
        return None;
    }
    let (field, other) = match (field_of(&args[0], types), field_of(&args[1], types)) {
        (Some(x), None) => (x, &args[1]),
        (None, Some(x)) => (x, &args[0]),
        _ => return None,
    };
    if field.1 != "geo_point" {
        return None;
    }
    let (x, y) = super::spatial::point_xy(&const_wkt(other)?)?;
    // ES|QL's spatial constants are quantized like doc values: a pushed
    // `ST_DISTANCE` bound is a circle around the quantized point (ES 9.5.5
    // counts a point 1.4 mm inside 400 km of the quantized center but
    // outside of the raw one).
    let (lat, lon) = rustic_core::geo::quantize(y, x);
    Some((field.0, lat, lon))
}

fn geo_distance_q(f: &str, lat: f64, lon: f64, meters: f64) -> Value {
    json!({"geo_distance": {"distance": format!("{meters}m"), f: {"lat": lat, "lon": lon}}})
}

/// A query matching a superset of the docs where `e` holds, for conditions
/// [`to_dsl`] cannot translate exactly: the condition still runs on the
/// loaded rows, the query only narrows what is loaded. (`ST_DISTANCE(..) ==
/// d`: a 1m band around the distance.)
pub fn superset_dsl(e: &Expr, types: &HashMap<String, String>) -> Option<Value> {
    let Expr::Bin(BinOp::Eq, a, b) = e else { return None };
    if let Some(q) = grid_cell_dsl(a, b, types).or_else(|| grid_cell_dsl(b, a, types)) {
        return Some(q);
    }
    let ((f, lat, lon), d) = match (distance_of(a, types), distance_of(b, types), a.as_ref(), b.as_ref()) {
        (Some(x), _, _, Expr::Lit(v)) | (None, Some(x), Expr::Lit(v), _) => (x, v.as_f64()?),
        _ => return None,
    };
    if !d.is_finite() || d < 0.0 {
        return None;
    }
    let outer = geo_distance_q(f, lat, lon, d + 1.0);
    Some(if d > 1.0 { json!({"bool": {"filter": [outer], "must_not": [geo_distance_q(f, lat, lon, d - 1.0)]}}) } else { outer })
}

/// `ST_GEOHASH(field, p) == "cell"` / `ST_GEOTILE(field, z) == "z/x/y"`:
/// the cell's bounding box (padded; boundaries included), a superset of
/// the points in the cell.
fn grid_cell_dsl(call: &Expr, cell: &Expr, types: &HashMap<String, String>) -> Option<Value> {
    let (Expr::Call(f, args, _), Expr::Lit(Val::Str(c))) = (call, cell) else { return None };
    let [field, Expr::Lit(p)] = args.as_slice() else { return None };
    let (name, ty) = field_of(field, types)?;
    if ty != "geo_point" {
        return None;
    }
    let p = p.as_i64()?;
    let pad = 1e-9;
    let (top, left, bottom, right) = match f.as_str() {
        "ST_GEOHASH" if (1..=12).contains(&p) && c.len() == p as usize => rustic_core::geo::geohash_bbox(c)?,
        "ST_GEOTILE" if (0..=29).contains(&p) => {
            let parts: Vec<u32> = c.split('/').map(|x| x.parse().ok()).collect::<Option<_>>()?;
            let [z, x, y] = parts.as_slice() else { return None };
            if *z as i64 != p || *x >= 1u32 << z || *y >= 1u32 << z {
                return None;
            }
            let (t, l, b, r) = rustic_core::geo::geotile_bbox(*z, *x, *y);
            // Latitudes beyond the Mercator range fall in the edge rows.
            (if *y == 0 { 90.0 } else { t }, l, if *y + 1 == 1u32 << z { -90.0 } else { b }, r)
        }
        _ => return None,
    };
    Some(json!({"geo_bounding_box": { name: {"top_left": {"lat": (top + pad).min(90.0), "lon": (left - pad).max(-180.0)}, "bottom_right": {"lat": (bottom - pad).max(-90.0), "lon": (right + pad).min(180.0)}}}}))
}

/// Spatial predicates on indexed geo fields, run in the index (ES pushes
/// them to Lucene): `ST_INTERSECTS` & co as `geo_shape`, `ST_DISTANCE`
/// bounds as `geo_distance`.
fn spatial_dsl(e: &Expr, types: &HashMap<String, String>) -> Option<Value> {
    // A null constant argument (an unparsable shape folds to null, with a
    // warning) makes the relation null: no document matches.
    if let Expr::Call(name, args, _) = e {
        if matches!(name.as_str(), "ST_INTERSECTS" | "ST_DISJOINT" | "ST_WITHIN" | "ST_CONTAINS") && args.len() == 2 && args.iter().any(|a| matches!(a, Expr::Lit(Val::Null))) {
            return Some(json!({"match_none": {}}));
        }
    }
    if let Expr::Call(name, args, _) = e {
        if matches!(name.as_str(), "ST_INTERSECTS" | "ST_DISJOINT" | "ST_WITHIN" | "ST_CONTAINS") && args.len() == 2 {
            let (f, ty, wkt, swapped) = match (field_of(&args[0], types), const_wkt(&args[1])) {
                (Some((f, t)), Some(w)) => (f, t, w, false),
                _ => match (const_wkt(&args[0]), field_of(&args[1], types)) {
                    (Some(w), Some((f, t))) => (f, t, w, true),
                    _ => return None,
                },
            };
            if ty != "geo_point" && ty != "geo_shape" {
                return None;
            }
            let rel = match (name.as_str(), swapped) {
                ("ST_INTERSECTS", _) => "intersects",
                ("ST_DISJOINT", _) => "disjoint",
                ("ST_WITHIN", false) | ("ST_CONTAINS", true) => "within",
                _ => "contains",
            };
            return Some(json!({"geo_shape": { f: {"shape": wkt, "relation": rel}}}));
        }
    }
    if let Expr::Bin(op, a, b) = e {
        if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            let found = match (distance_of(a, types), distance_of(b, types), a.as_ref(), b.as_ref()) {
                (Some(x), _, _, Expr::Lit(v)) => v.as_f64().map(|d| (x, d, *op)),
                (None, Some(x), Expr::Lit(v), _) => v.as_f64().map(|d| {
                    let flipped = match op {
                        BinOp::Lt => BinOp::Gt,
                        BinOp::Le => BinOp::Ge,
                        BinOp::Gt => BinOp::Lt,
                        _ => BinOp::Le,
                    };
                    (x, d, flipped)
                }),
                _ => None,
            };
            if let Some(((f, lat, lon), d, op)) = found {
                if !d.is_finite() {
                    return None;
                }
                if d < 0.0 {
                    // Distances are never negative.
                    return Some(if matches!(op, BinOp::Lt | BinOp::Le) { json!({"match_none": {}}) } else { json!({"exists": {"field": f}}) });
                }
                let within = geo_distance_q(f, lat, lon, d);
                // `>=`: has a point (the whole-world box, same as `exists`
                // for a geo point) and none within: one geo field, so the
                // index answers it in one KD pass.
                let world = json!({"geo_bounding_box": { f: {"top_left": {"lat": 90.0, "lon": -180.0}, "bottom_right": {"lat": -90.0, "lon": 180.0}}}});
                return Some(if matches!(op, BinOp::Lt | BinOp::Le) { within } else { json!({"bool": {"filter": [world], "must_not": [within]}}) });
            }
        }
    }
    None
}

/// Translate a WHERE expression on raw index fields to Query DSL:
/// `(query, scoring)`.
pub fn to_dsl(e: &Expr, types: &HashMap<String, String>, params_ok: bool) -> Option<(Value, bool)> {
    let _ = params_ok;
    if let Some(q) = spatial_dsl(e, types) {
        return Some((q, false));
    }
    match e {
        Expr::Lit(Val::Bool(true)) => Some((json!({"match_all": {}}), false)),
        Expr::Lit(Val::Bool(false)) | Expr::Lit(Val::Null) => Some((json!({"match_none": {}}), false)),
        Expr::Bin(BinOp::And, a, b) => {
            let (qa, sa) = to_dsl(a, types, true)?;
            let (qb, sb) = to_dsl(b, types, true)?;
            let mut must = Vec::new();
            let mut filter = Vec::new();
            for (q, s) in [(qa, sa), (qb, sb)] {
                if s { must.push(q) } else { filter.push(q) }
            }
            Some((json!({"bool": {"must": must, "filter": filter}}), sa || sb))
        }
        Expr::Bin(BinOp::Or, a, b) => {
            let (qa, sa) = to_dsl(a, types, true)?;
            let (qb, sb) = to_dsl(b, types, true)?;
            Some((json!({"bool": {"should": [qa, qb], "minimum_should_match": 1}}), sa || sb))
        }
        Expr::Not(a) => {
            let (q, _) = to_dsl(a, types, true)?;
            Some((json!({"bool": {"must_not": [q]}}), false))
        }
        Expr::Bin(op, a, b) if matches!(op, BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) => {
            let (f, ty, lit, op) = match (field_of(a, types), field_of(b, types), a.as_ref(), b.as_ref()) {
                (Some((f, t)), None, _, Expr::Lit(v)) => (f, t, v, *op),
                (None, Some((f, t)), Expr::Lit(v), _) => (
                    f,
                    t,
                    v,
                    match op {
                        BinOp::Lt => BinOp::Gt,
                        BinOp::Le => BinOp::Ge,
                        BinOp::Gt => BinOp::Lt,
                        BinOp::Ge => BinOp::Le,
                        o => *o,
                    },
                ),
                _ => return None,
            };
            if lit.is_null() {
                return Some((json!({"match_none": {}}), false));
            }
            let v = lit_json(lit, &ty)?;
            let date = ty.starts_with("date");
            let range = |k: &str| {
                let mut r = json!({ k: v.clone() });
                if date {
                    r["format"] = json!("epoch_millis");
                }
                json!({"range": { f: r }})
            };
            let eq = if date { json!({"range": { f: {"gte": v.clone(), "lte": v.clone(), "format": "epoch_millis"}}}) } else { json!({"term": { f: v.clone() }}) };
            Some((
                match op {
                    BinOp::Eq => eq,
                    BinOp::Neq => exists_and(f, eq),
                    BinOp::Lt => range("lt"),
                    BinOp::Le => range("lte"),
                    BinOp::Gt => range("gt"),
                    _ => range("gte"),
                },
                false,
            ))
        }
        Expr::IsNull(a, not) => {
            let (f, _) = field_of(a, types)?;
            let ex = json!({"exists": {"field": f}});
            Some((if *not { ex } else { json!({"bool": {"must_not": [ex]}}) }, false))
        }
        Expr::In(a, list, neg) => {
            let (f, ty) = field_of(a, types)?;
            let mut vals = Vec::new();
            for x in list {
                match x {
                    Expr::Lit(Val::Null) => {}
                    Expr::Lit(v) => vals.push(lit_json(v, &ty)?),
                    _ => return None,
                }
            }
            if ty.starts_with("date") {
                return None;
            }
            let q = json!({"terms": { f: vals }});
            Some((if *neg { exists_and(f, q) } else { q }, false))
        }
        Expr::Like(a, pats, neg, rlike) => {
            let (f, ty) = field_of(a, types)?;
            if ty != "keyword" {
                return None;
            }
            let qs: Vec<Value> = pats
                .iter()
                .map(|p| if *rlike { json!({"regexp": { f: {"value": p}}}) } else { json!({"wildcard": { f: {"value": p}}}) })
                .collect();
            let q = if qs.len() == 1 { qs[0].clone() } else { json!({"bool": {"should": qs, "minimum_should_match": 1}}) };
            Some((if *neg { exists_and(f, q) } else { q }, false))
        }
        Expr::Between(a, lo, hi, neg) => {
            let (f, ty) = field_of(a, types)?;
            let (Expr::Lit(l), Expr::Lit(h)) = (lo.as_ref(), hi.as_ref()) else { return None };
            let mut r = json!({"gte": lit_json(l, &ty)?, "lte": lit_json(h, &ty)?});
            if ty.starts_with("date") {
                r["format"] = json!("epoch_millis");
            }
            let q = json!({"range": { f: r }});
            Some((if *neg { exists_and(f, q) } else { q }, false))
        }
        Expr::Call(name, args, _) => {
            let opts = |k: usize| -> serde_json::Map<String, Value> {
                match args.get(k) {
                    Some(Expr::Map(m)) => m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect(),
                    _ => serde_json::Map::new(),
                }
            };
            let lit_s = |k: usize| -> Option<Value> {
                match args.get(k) {
                    Some(Expr::Lit(Val::Null)) | None => None,
                    Some(Expr::Lit(v)) => Some(v.to_json()),
                    _ => None,
                }
            };
            match name.as_str() {
                "MATCH" | "MATCH_OP" | "MATCH_PHRASE" => {
                    let f = match args.first()? {
                        Expr::Col(n, _) => n.clone(),
                        _ => return None,
                    };
                    let mut o = opts(2);
                    o.insert("query".into(), lit_s(1)?);
                    let k = if name == "MATCH_PHRASE" { "match_phrase" } else { "match" };
                    Some((json!({ k: { f: o }}), true))
                }
                "MULTI_MATCH" => {
                    let q = lit_s(0)?;
                    let mut fields = Vec::new();
                    for a in &args[1..] {
                        match a {
                            Expr::Col(n, _) => fields.push(json!(n)),
                            Expr::Map(_) => {}
                            _ => return None,
                        }
                    }
                    let mut o = match args.last() {
                        Some(Expr::Map(m)) => m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect(),
                        _ => serde_json::Map::new(),
                    };
                    o.insert("query".into(), q);
                    o.insert("fields".into(), Value::Array(fields));
                    Some((json!({"multi_match": o}), true))
                }
                "QSTR" | "KQL" => {
                    let q = lit_s(0)?;
                    let mut o = opts(1);
                    let qs = q.as_str()?.to_string();
                    let qs = if name == "KQL" { kql_to_qs(&qs) } else { qs };
                    o.insert("query".into(), json!(qs));
                    Some((json!({"query_string": o}), true))
                }
                "TERM" => {
                    let f = match args.first()? {
                        Expr::Col(n, _) => n.clone(),
                        _ => return None,
                    };
                    Some((json!({"term": { f: lit_s(1)? }}), true))
                }
                "STARTS_WITH" => {
                    let (f, ty) = field_of(args.first()?, types)?;
                    if ty != "keyword" {
                        return None;
                    }
                    Some((json!({"prefix": { f: lit_s(1)? }}), false))
                }
                "CIDR_MATCH" => {
                    let (f, ty) = field_of(args.first()?, types)?;
                    if ty != "ip" {
                        return None;
                    }
                    let mut vals = Vec::new();
                    for a in &args[1..] {
                        match a {
                            Expr::Lit(Val::Str(s)) => vals.push(json!(s)),
                            Expr::Lit(Val::Multi(m)) => vals.extend(m.iter().map(Val::to_json)),
                            _ => return None,
                        }
                    }
                    Some((json!({"terms": { f: vals }}), false))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn kql_to_qs(q: &str) -> String {
    q.split_whitespace()
        .map(|w| match w {
            "and" => "AND",
            "or" => "OR",
            "not" => "NOT",
            o => o,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// FROM + leading WHERE/SORT/LIMIT pushdown. Returns the table and how many
/// following commands were consumed.
/// The outside world of the name analysis: lookup indices and enrich policies.
struct AnalyzeEnv<'c, 'a> {
    ctx: &'c Ctx<'a>,
}

impl super::analyze::Env for AnalyzeEnv<'_, '_> {
    fn lookup_columns(&self, index: &str) -> Vec<String> {
        resolve_schema(self.ctx, &[index.to_string()], 0).map(|s| s.cols.into_iter().map(|(n, _)| n).collect()).unwrap_or_default()
    }
    fn enrich_columns(&self, policy: &str) -> Vec<String> {
        self.ctx.node.and_then(|n| crate::enrich::store(n).get(policy).map(|p| p.enrich_fields.clone())).unwrap_or_default()
    }
}

/// `ResolveUnmapped`: under `unmapped_fields=nullify|load`, the names the
/// rest of the pipeline references but nothing defines.
fn unmapped_refs(ctx: &Ctx, rest: &[Cmd], scope: Vec<String>) -> Vec<String> {
    if ctx.settings.unmapped == super::settings::Unmapped::Default {
        return vec![];
    }
    super::analyze::unresolved(rest, scope, &AnalyzeEnv { ctx }).into_iter().map(|(n, _)| n).collect()
}

/// Offsets of the items of a `KEEP`/`DROP` at `kw_pos`.
fn list_item_positions(full: &str, kw_pos: usize) -> Vec<usize> {
    let after = full[kw_pos..].find(|c: char| c.is_whitespace()).map(|i| kw_pos + i).unwrap_or(full.len());
    let end = super::lexer::split_top(full, after, full.len(), b'|', super::lexer::Dialect::Esql)[0].1;
    super::lexer::split_top(full, after, end, b',', super::lexer::Dialect::Esql)
        .into_iter()
        .map(|(a, b)| {
            let seg = &full[a..b];
            a + (seg.len() - seg.trim_start().len())
        })
        .collect()
}

/// `unmapped_fields=load` cannot read sub-paths of `flattened` fields:
/// `Loading subfield [x.y] when parent [x] is of flattened field type ...`.
fn check_flattened_subfields(ctx: &Ctx, schema: &IndexSchema, rest: &[Cmd], scope: Vec<String>, from_pos: usize) -> Result<()> {
    let flattened: Vec<&String> = schema.cols.iter().filter(|(_, c)| c.ty == "flattened" || c.types.iter().any(|(t, _)| t == "flattened")).map(|(n, _)| n).collect();
    if flattened.is_empty() {
        return Ok(());
    }
    let parent_of = |n: &str| -> Option<String> { flattened.iter().filter(|f| n.starts_with(&format!("{f}."))).max_by_key(|f| f.len()).map(|f| f.to_string()) };
    let mut problems: Vec<(usize, String)> = Vec::new();
    let msg = |n: &str, p: &str| format!("Loading subfield [{n}] when parent [{p}] is of flattened field type is not supported with unmapped_fields=\"load\"");
    // Partially mapped fields under a flattened parent would load `_source` there.
    for (n, c) in &schema.cols {
        if let Some(p) = parent_of(n) {
            if c.ty != "flattened" && !c.types.iter().any(|(t, _)| t == "flattened") {
                problems.push((from_pos, msg(n, &p)));
            }
        }
    }
    let listed: Vec<&String> = rest.iter().flat_map(|c| match c {
        Cmd::Keep(items, _) | Cmd::Drop(items, _) => items.iter().collect::<Vec<_>>(),
        _ => vec![],
    }).collect();
    for (n, p) in super::analyze::unresolved(rest, scope.clone(), &AnalyzeEnv { ctx }) {
        if listed.contains(&&n) {
            continue;
        }
        if let Some(parent) = parent_of(&n) {
            let off = ctx.full.get(p..).and_then(|s| s.find(n.as_str())).map(|o| p + o).unwrap_or(p);
            if !problems.iter().any(|(_, m)| *m == msg(&n, &parent)) {
                problems.push((off, msg(&n, &parent)));
            }
        }
    }
    // Patterns that only name sub-paths of a flattened field.
    for c in rest {
        if let Cmd::Keep(items, kp) | Cmd::Drop(items, kp) = c {
            let poss = list_item_positions(ctx.full, *kp);
            for (k, it) in items.iter().enumerate() {
                if scope.iter().any(|s| name_matches(it, s)) {
                    continue;
                }
                if let Some(parent) = parent_of(it) {
                    problems.push((poss.get(k).copied().unwrap_or(*kp), msg(it, &parent)));
                }
            }
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    problems.sort_by_key(|p| p.0);
    problems.dedup();
    let n = problems.len();
    let lines: Vec<String> = problems.iter().map(|(o, m)| format!("{}: {m}", super::pos_prefix(ctx.full, *o))).collect();
    Err(EsError::new(400, "verification_exception", format!("Found {n} problem{}\n{}", if n == 1 { "" } else { "s" }, lines.join("\n"))))
}

#[cfg(test)]
thread_local! {
    /// Tests: read every column (the reference of [`project_columns`]).
    pub(super) static PROJECTION_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn projection_enabled() -> bool {
    #[cfg(test)]
    if PROJECTION_DISABLED.with(|c| c.get()) {
        return false;
    }
    true
}

/// ES `PruneColumns` and field-extraction pushdown for `FROM` followed by
/// `rest`: sets `schema.projection` to the index columns the pipeline can
/// read, and returns `rest` with each `LOOKUP JOIN`'s `fields` set to the
/// lookup columns it can read (`None` when there is no join to annotate).
///
/// A column is read when an expression, a join key, a grouping or
/// `MV_EXPAND` reads it, or when it can reach the output. Each column
/// carries the index and lookup columns its values come from; KEEP, DROP,
/// RENAME, EVAL, STATS and joins move those sets along by name. Every
/// approximation keeps more columns, never fewer (a name another column may
/// shadow keeps both sources), so a column left out is never observed: its
/// nulls are dropped, renamed away or overwritten before anything reads it.
/// The table keeps every column (names, types and order), so validation and
/// errors are unchanged. A command the analysis does not follow (FORK,
/// ENRICH, INLINE STATS, ...) leaves every column read.
fn project_columns(ctx: &mut Ctx, schema: &mut IndexSchema, rest: &[Cmd]) -> Option<Vec<Cmd>> {
    // Only these commands remove a column (or replace its values).
    if !rest.iter().any(|cmd| matches!(cmd, Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..) | Cmd::Eval(_) | Cmd::Stats { .. })) {
        return None;
    }
    // Sources of a column: (0, index column), or (k, column of the k-th
    // join's lookup index).
    type Sources = HashSet<(usize, String)>;
    // Expressions whose column references are all `Col`s by name.
    fn by_name(e: &Expr) -> bool {
        match e {
            Expr::Idx(_) | Expr::AggRef(_) => false,
            // Query strings name fields inside a literal.
            Expr::Call(name, args, _) => !matches!(name.to_ascii_uppercase().as_str(), "QSTR" | "KQL") && args.iter().all(by_name),
            Expr::Bin(_, a, b) => by_name(a) && by_name(b),
            Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => by_name(a),
            Expr::In(a, list, _) => by_name(a) && list.iter().all(by_name),
            Expr::Between(a, b, c, _) => by_name(a) && by_name(b) && by_name(c),
            Expr::Lit(_) | Expr::Col(..) | Expr::Star | Expr::Map(_) => true,
        }
    }
    // The sources of the columns `e` reads (`None`: not analysable).
    fn sources(e: &Expr, cols: &HashMap<String, Sources>) -> Option<Sources> {
        if !by_name(e) {
            return None;
        }
        let mut refs = Vec::new();
        super::analyze::expr_refs(e, &mut refs);
        Some(refs.iter().filter_map(|(name, _)| cols.get(name)).flatten().cloned().collect())
    }
    let mut cols: HashMap<String, Sources> = schema.cols.iter().map(|(name, _)| (name.clone(), HashSet::from([(0, name.clone())]))).collect();
    let mut read: Sources = HashSet::new();
    // The joins (join k is `joins[k - 1]`): positions in `rest`, lookup columns.
    let mut joins: Vec<(usize, usize)> = Vec::new();
    for (at, cmd) in rest.iter().enumerate() {
        match cmd {
            Cmd::Where(e, _) => read.extend(sources(e, &cols)?),
            Cmd::Sort(keys) => {
                for key in keys {
                    read.extend(sources(&key.expr, &cols)?);
                }
            }
            Cmd::LimitBy(_, keys) => {
                for key in keys {
                    read.extend(sources(key, &cols)?);
                }
            }
            Cmd::Limit(_) | Cmd::Sample(_) => {}
            Cmd::MvExpand(name, _) => read.extend(cols.get(name).cloned().unwrap_or_default()),
            Cmd::Eval(fields) => {
                for field in fields {
                    let mut from = sources(&field.expr, &cols)?;
                    if let Some(filter) = &field.filter {
                        from.extend(sources(filter, &cols)?);
                    }
                    cols.insert(field.name.clone(), from);
                }
            }
            Cmd::Keep(patterns, _) => cols.retain(|name, _| name.starts_with('\u{0}') || patterns.iter().any(|p| name_matches(p, name))),
            Cmd::Drop(patterns, _) => cols.retain(|name, _| !patterns.iter().any(|p| name_matches(p, name))),
            Cmd::Rename(pairs, _) => {
                for (old, new) in pairs {
                    if let Some(from) = cols.remove(old) {
                        cols.entry(new.clone()).or_default().extend(from);
                    }
                }
            }
            Cmd::Dissect { field, pattern, .. } => {
                read.extend(sources(field, &cols)?);
                for name in super::analyze::dissect_names(pattern) {
                    cols.entry(name).or_default();
                }
            }
            // GROK's columns hold values of the field read here.
            Cmd::Grok { field, .. } => read.extend(sources(field, &cols)?),
            Cmd::Stats { aggs, by, inline: false, .. } => {
                for field in aggs.iter().chain(by) {
                    read.extend(sources(&field.expr, &cols)?);
                    if let Some(filter) = &field.filter {
                        read.extend(sources(filter, &cols)?);
                    }
                }
                cols = aggs.iter().chain(by).map(|field| (field.name.clone(), HashSet::new())).collect();
            }
            Cmd::LookupJoin { index, index_pos, on, cond, .. } => {
                // An invalid join fails when it runs, as without the analysis.
                let lookup = lookup_schema(ctx, index, *index_pos, None).ok()?;
                let k = joins.len() + 1;
                joins.push((at, lookup.cols.len()));
                // Keys (and an expression's references) are read on both
                // sides; a name resolves to whichever side has it.
                let mut keys: Vec<String> = on.clone();
                if let Some(cond) = cond {
                    if !by_name(cond) {
                        return None;
                    }
                    let mut refs = Vec::new();
                    super::analyze::expr_refs(cond, &mut refs);
                    keys.extend(refs.into_iter().map(|(name, _)| name));
                }
                for key in &keys {
                    read.extend(cols.get(key).cloned().unwrap_or_default());
                    if lookup.cols.iter().any(|(name, _)| name == key) {
                        read.insert((k, key.clone()));
                    }
                }
                // Lookup columns shadow left ones of the same name (kept as a
                // source too); a named join keeps the left key column.
                for (name, _) in &lookup.cols {
                    if cond.is_none() && on.contains(name) {
                        continue;
                    }
                    cols.entry(name.clone()).or_default().insert((k, name.clone()));
                }
            }
            _ => return None,
        }
    }
    read.extend(cols.into_values().flatten());
    // Every column read: no projection (`_source` is read whole, unfiltered).
    let of = |k: usize, all: usize| -> Option<HashSet<String>> {
        let names: HashSet<String> = read.iter().filter(|(j, _)| *j == k).map(|(_, name)| name.clone()).collect();
        (names.len() < all).then_some(names)
    };
    schema.projection = of(0, schema.cols.len());
    let fields: Vec<(usize, HashSet<String>)> = joins.into_iter().enumerate().filter_map(|(k, (at, all))| Some((at, of(k + 1, all)?))).collect();
    if fields.is_empty() {
        return None;
    }
    let mut out = rest.to_vec();
    for (at, names) in fields {
        if let Cmd::LookupJoin { fields, .. } = &mut out[at] {
            *fields = Some(names);
        }
    }
    Some(out)
}

/// `FROM` with the `WHERE`s (and size / sort) it can push into the index
/// search: the table, how many commands of the returned rewrite (`Some`:
/// filters moved in front of `EVAL`s or joins given their lookup `fields`,
/// else `rest` itself) it covered. Only the columns the pipeline reads are
/// loaded ([`project_columns`]).
fn from_with_pushdown(ctx: &mut Ctx, indices: &[String], metadata: &[String], pos: usize, rest: &[Cmd]) -> Result<(Table, usize, Option<Vec<Cmd>>)> {
    let mut schema = resolve_schema(ctx, indices, pos)?;
    if ctx.ts {
        ctx.ts_dims = schema.cols.iter().filter(|(_, c)| c.dimension || c.ty == "flattened").map(|(n, _)| n.clone()).collect();
    }
    {
        let mut scope: Vec<String> = schema.cols.iter().map(|(n, _)| n.clone()).collect();
        scope.extend(metadata.iter().cloned());
        let load = ctx.settings.unmapped == super::settings::Unmapped::Load;
        let extra = std::mem::take(&mut ctx.extra_load);
        if load {
            check_flattened_subfields(ctx, &schema, rest, scope.clone(), pos)?;
        }
        let mut refs = unmapped_refs(ctx, rest, scope);
        for n in extra {
            if !refs.contains(&n) && !schema.cols.iter().any(|(c, _)| *c == n) {
                refs.push(n);
            }
        }
        for n in refs {
            // Loaded from `_source` as keyword, or a null-typed column.
            let mut c = ColSrc::plain(&n, if load { "keyword" } else { "null" });
            c.doc_values = false;
            schema.cols.push((n, c));
        }
    }
    for (n, c) in &schema.cols {
        if c.ty == "unsupported" {
            ctx.unsupported.insert(n.clone(), c.types.clone());
        }
        ctx.pristine.insert(n.clone());
    }
    let mut types: HashMap<String, String> = schema.cols.iter().map(|(n, c)| (n.clone(), c.ty.clone())).collect();
    // Keyword metadata fields filter in Lucene too (`_ignored` matches any of its values).
    for m in metadata {
        if matches!(m.as_str(), "_index" | "_id" | "_ignored") {
            types.insert(m.clone(), "keyword".into());
        }
    }
    let hoisted = if ctx.ts || !ctx.esql { None } else { fast::hoist_spatial_filters(rest, &types) };
    // The columns the pipeline reads (ES `PruneColumns` and field
    // extraction): its joins come back with their lookup columns set.
    let hoisted = if ctx.ts || !ctx.esql || !projection_enabled() { hoisted } else { project_columns(ctx, &mut schema, hoisted.as_deref().unwrap_or(rest)).or(hoisted) };
    let rest: &[Cmd] = hoisted.as_deref().unwrap_or(rest);
    let mut must: Vec<Value> = Vec::new();
    let mut filter: Vec<Value> = Vec::new();
    if let Some(f) = ctx.filter {
        filter.push(f.clone());
    }
    let mut residual: Vec<(Expr, usize)> = Vec::new();
    let mut ft_probes: Vec<Value> = Vec::new();
    let mut consumed = 0;
    let all_cols: Vec<Column> = {
        let mut c: Vec<Column> = schema.cols.iter().map(|(n, c)| Column { name: n.clone(), ty: c.col_type() }).collect();
        for m in metadata {
            c.push(Column { name: m.clone(), ty: "keyword".into() });
        }
        c
    };
    for cmd in rest {
        let Cmd::Where(e, wpos) = cmd else { break };
        // Validate columns / functions before pushing down.
        bind(ctx, e, &all_cols)?;
        check_fulltext_queries(ctx, e)?;
        let e = &fold_consts(ctx, e);
        let mut cs = Vec::new();
        conjuncts(e, &mut cs);
        for c in cs {
            match to_dsl(&c, &types, true) {
                Some((q, true)) => must.push(q),
                Some((q, false)) => filter.push(q),
                None => {
                    if let Some(q) = superset_dsl(&c, &types) {
                        filter.push(q);
                    }
                    // Full-text functions combined with conditions Lucene cannot
                    // evaluate run in the compute engine: each one becomes a
                    // per-document "matches" column.
                    let c = if has_fulltext(&c) {
                        let mut failed = false;
                        let c = probe_fulltext(&c, &types, &mut ft_probes, &mut failed);
                        if failed {
                            return Err(verification_err(ctx.full, *wpos, "Invalid condition: full-text functions can only be combined with conditions that can be pushed down to Lucene"));
                        }
                        c
                    } else {
                        c
                    };
                    residual.push((c, *wpos));
                }
            }
        }
        consumed += 1;
    }
    // Scored queries order hits by score, not by shard: partial aggregation
    // (which merges in shard order) needs constant scores.
    let scoring = !must.is_empty();
    let query = if must.is_empty() && filter.is_empty() { json!({"match_all": {}}) } else { json!({"bool": {"must": must, "filter": filter}}) };
    // Size / sort pushdown.
    let mut size: Option<usize> = None;
    let mut sort: Vec<Value> = Vec::new();
    if residual.is_empty() {
        let tail = &rest[consumed..];
        let mut hint: Option<usize> = Some(ctx.default_limit);
        let has_limit = tail.iter().any(|c| matches!(c, Cmd::Limit(_)));
        if !has_limit {
            hint = Some(ctx.default_limit);
        }
        let mut k = 0;
        while k < tail.len() {
            match &tail[k] {
                Cmd::Limit(n) => {
                    hint = Some(*n);
                    break;
                }
                // A positive per-group quota always retains the first row.
                // A following global LIMIT 1 therefore needs just that row,
                // including when shards are remote. Never cross a WHERE,
                // SORT, or aggregate to apply this bound.
                Cmd::LimitBy(n, _) if *n > 0 && join_output_limit(&tail[k + 1..]) == Some(1) => {
                    hint = Some(1);
                    break;
                }
                Cmd::Eval(_) | Cmd::Keep(..) | Cmd::Drop(..) | Cmd::Rename(..) | Cmd::Dissect { .. } | Cmd::Grok { .. } | Cmd::LookupJoin { .. } => {}
                Cmd::Sort(keys) if k == 0 => {
                    let pushable = keys.iter().all(|key| match &key.expr {
                        Expr::Col(n, _) => {
                            !metadata.contains(n)
                                && types.get(n).map(|t| t != "text" && t != "unsupported" && t != "null" && !super::spatial::is_spatial(t)).unwrap_or(false)
                                && schema.cols.iter().find(|(c, _)| c == n).is_some_and(|(_, c)| c.per_index.is_empty() && c.doc_values)
                        }
                        _ => false,
                    });
                    match (pushable, tail.get(1)) {
                        (true, Some(Cmd::Limit(n))) => {
                            for key in keys {
                                if let Expr::Col(n, _) = &key.expr {
                                    let nulls_first = key.nulls_first.unwrap_or(key.desc);
                                    let ut = match types.get(n).map(String::as_str) {
                                        Some("integer") => "integer",
                                        Some("long") | Some("counter_long") => "long",
                                        Some("double") | Some("counter_double") => "double",
                                        Some("unsigned_long") => "unsigned_long",
                                        Some("date") => "date",
                                        Some("date_nanos") => "date_nanos",
                                        Some("boolean") => "boolean",
                                        Some("ip") => "ip",
                                        Some("version") => "version",
                                        _ => "keyword",
                                    };
                                    sort.push(json!({ n.clone(): {"order": if key.desc { "desc" } else { "asc" }, "missing": if nulls_first { "_first" } else { "_last" }, "unmapped_type": ut}}));
                                }
                            }
                            hint = Some(*n);
                        }
                        _ => hint = None,
                    }
                    break;
                }
                _ => {
                    hint = None;
                    break;
                }
            }
            k += 1;
        }
        size = hint.map(|h| h.min(ctx.max_limit));
    }
    if !ctx.esql && metadata.is_empty() && ft_probes.is_empty() && !scoring {
        if residual.is_empty() {
            if let Some(table) = sql_pivot_stream::grouped_count(ctx, &schema, &query, &rest[consumed..])? {
                return Ok((table, consumed + 1, hoisted));
            }
        }
        // Aggregates and computed sorts fold the referenced columns batch by
        // batch instead of materializing a row of every mapped field per match.
        if let Some(table) = sql_pivot_stream::grouped_fold(ctx, &schema, &query, &residual, &rest[consumed..])? {
            return Ok((table, consumed + 1, hoisted));
        }
        if let Some((table, covered)) = sql_pivot_stream::sorted_fold(ctx, &schema, &query, &residual, &rest[consumed..])? {
            return Ok((table, consumed + covered, hoisted));
        }
        if let Some((table, covered)) = sql_pivot_stream::try_run(ctx, &schema, &query, &residual, &rest[consumed..])? {
            return Ok((table, consumed + covered, hoisted));
        }
    }
    // Pipelines the index answers directly (doc values, counts).
    if ft_probes.is_empty() && !scoring && metadata.is_empty() {
        if let Some((t, n)) = fast::try_run(ctx, &schema, indices, &query, &residual, &rest[consumed..])? {
            return Ok((t, consumed + n, hoisted));
        }
    }
    if metadata.is_empty() && residual.is_empty() && ft_probes.is_empty() && !scoring && !ctx.ts && ctx.esql {
        if let Some(table) = distributed_group_limit(ctx, &schema, indices, &query, &rest[consumed..])? {
            return Ok((table, rest.len(), hoisted));
        }
    }
    // A decomposable STATS aggregates per shard (in parallel, on the node
    // holding each shard) and merges partial states instead of loading rows.
    if residual.is_empty() && ft_probes.is_empty() && !scoring && ctx.shards.is_some() {
        if let Some(cmd) = rest.get(consumed) {
            // Every shard here: one slot per shard copy target.
            let all_local = || -> Option<Vec<super::remote::Slot>> {
                schema.targets.iter().all(|t| t.shard.is_some() && t.snapshot.is_none()).then(|| (0..schema.targets.len()).map(super::remote::Slot::Local).collect())
            };
            let slots = schema.slots.clone().or_else(all_local);
            if let (Some(slots), Some(plan)) = (slots, agg_plan(ctx, &schema, indices, cmd)) {
                if let Some(t) = distributed_stats(ctx, &schema, metadata, &query, &slots, plan, cmd)? {
                    return Ok((t, consumed + 1, hoisted));
                }
            }
        }
    }
    let mut t = load(ctx, &schema, metadata, query, sort, size, &ft_probes)?;
    for (e, wpos) in residual {
        t = apply(ctx, &Cmd::Where(e, wpos), t)?;
    }
    if !ft_probes.is_empty() {
        let keep: Vec<usize> = (0..t.cols.len()).filter(|i| !t.cols[*i].name.starts_with(FT_PROBE)).collect();
        t.project(&keep);
    }
    Ok((t, consumed, hoisted))
}

/// The lookup index of a `LOOKUP JOIN`, loaded.
fn lookup_right(ctx: &mut Ctx, index: &str, index_pos: usize, fields: Option<&HashSet<String>>) -> Result<Table> {
    let schema = lookup_schema(ctx, index, index_pos, fields)?;
    lookup_rows(ctx, &schema, None)
}

/// Lookup-side reads past this many distinct values of one key column load
/// the whole lookup index instead of a `terms` query.
const LOOKUP_KEYED_MAX_TERMS: usize = 65_536;

#[cfg(test)]
thread_local! {
    /// Tests: always load the whole lookup index (the keyed read's reference).
    pub(super) static LOOKUP_KEYED_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: keyed lookup reads always read whole rows (the reference of [`lookup_keyed_filtered`]).
    pub(super) static LOOKUP_FILTERED_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: lookup rows read through a keyed `terms` query.
    pub(super) static LOOKUP_KEYED_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Tests: lookup sides read by key (a `terms` query, or nothing when no
    /// left key can match).
    pub(super) static LOOKUP_KEYED_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether the lookup side reads its stored `_source` directly (all shards
/// here, stored source) rather than through a search response.
fn lookup_reads_source(ctx: &Ctx, schema: &IndexSchema) -> bool {
    ctx.esql && schema.slots.is_none() && !schema.targets.is_empty() && schema.targets.iter().all(|t| {
        t.svc.source_stored_as_sent()
    })
}

/// The key columns a keyed lookup can query exactly: plain indexed
/// `keyword`, `long` and `integer` fields (no normalizer, constant, union
/// type or computed value), whose `terms` query matches every lookup
/// document holding the value. `None`: load the whole lookup index.
fn lookup_key_kinds(ctx: &Ctx, schema: &IndexSchema, on: &[String]) -> Option<Vec<bool>> {
    #[cfg(test)]
    if LOOKUP_KEYED_DISABLED.with(|c| c.get()) {
        return None;
    }
    if !lookup_reads_source(ctx, schema) {
        return None;
    }
    let lookup = schema.targets[0].svc.lookup();
    on.iter().map(|name| {
        let (_, c) = schema.cols.iter().find(|(n, _)| n == name)?;
        let field = lookup.fields.get(name)?;
        let keyword = c.ty == "keyword" && c.declared == "keyword";
        let integral = matches!(c.ty.as_str(), "long" | "integer") && c.declared == c.ty;
        let plain = c.per_index.is_empty() && c.constants.is_empty() && c.normalizer.is_none() && !c.template_id && c.ordered.is_empty()
            && field.indexed && field.normalizer.is_none() && field.script.is_none() && field.alias_path.is_none() && field.nested_path.is_none();
        (plain && (keyword || integral)).then_some(keyword)
    }).collect()
}

/// The lookup rows a set of left key tuples (`keys`, in `ON` order) can
/// join: a `terms` query per key column on the lookup index (ES's
/// `LookupFromIndexService` term queries), superset of the rows whose keys
/// equal a tuple's ([`LookupJoinPlan`] then matches exactly). Null and
/// multi-valued left keys never match (`QueryList.onlySingleValues`), nor
/// values of another type. `None`: not keyed (see [`lookup_key_kinds`]).
fn lookup_keyed(ctx: &mut Ctx, schema: &IndexSchema, on: &[String], keys: &[Vec<Val>]) -> Result<Option<Table>> {
    lookup_keyed_with(ctx, schema, on, keys, None)
}

/// [`lookup_keyed`] reading only the rows `extra` (a lookup query) matches.
fn lookup_keyed_with(ctx: &mut Ctx, schema: &IndexSchema, on: &[String], keys: &[Vec<Val>], extra: Option<&Value>) -> Result<Option<Table>> {
    let Some(kinds) = lookup_key_kinds(ctx, schema, on) else { return Ok(None) };
    let mut strings: Vec<HashSet<&str>> = vec![HashSet::new(); on.len()];
    let mut numbers: Vec<HashSet<i64>> = vec![HashSet::new(); on.len()];
    'rows: for tuple in keys {
        if tuple.len() != on.len() {
            return Ok(None);
        }
        for (k, value) in tuple.iter().enumerate() {
            // A value whose `Val::key` no lookup value of the column can have.
            let fits = match (kinds[k], value) {
                (true, Val::Str(_)) => true,
                (false, Val::Int(_) | Val::Long(_)) => true,
                (false, Val::Double(d)) => d.fract() == 0.0 && d.abs() < 9.0e15,
                _ => false,
            };
            if !fits {
                continue 'rows;
            }
        }
        for (k, value) in tuple.iter().enumerate() {
            match value {
                Val::Str(s) => { strings[k].insert(s.as_str()); }
                Val::Int(i) | Val::Long(i) => { numbers[k].insert(*i); }
                Val::Double(d) => { numbers[k].insert(*d as i64); }
                _ => {}
            }
        }
        if strings.iter().zip(&numbers).any(|(s, n)| s.len() + n.len() > LOOKUP_KEYED_MAX_TERMS) {
            return Ok(None);
        }
    }
    let mut terms = Vec::with_capacity(on.len());
    for (k, name) in on.iter().enumerate() {
        let values: Vec<Value> = if kinds[k] {
            let mut v: Vec<&str> = strings[k].iter().copied().collect();
            v.sort_unstable();
            v.into_iter().map(|s| Value::String(s.to_string())).collect()
        } else {
            let mut v: Vec<i64> = numbers[k].iter().copied().collect();
            v.sort_unstable();
            v.into_iter().map(Value::from).collect()
        };
        if values.is_empty() {
            // No left row can match: the lookup side reads nothing.
            #[cfg(test)]
            LOOKUP_KEYED_CALLS.with(|c| c.set(c.get() + 1));
            return Ok(Some(Table { cols: source_cols(schema, &[]), rows: Vec::new() }));
        }
        terms.push(json!({"terms": {name.clone(): values}}));
    }
    terms.extend(extra.cloned());
    let query = if terms.len() == 1 { terms.pop().unwrap() } else { json!({"bool": {"filter": terms}}) };
    let table = lookup_rows(ctx, schema, Some(query))?;
    #[cfg(test)]
    {
        LOOKUP_KEYED_READS.with(|c| c.set(c.get() + table.rows.len()));
        LOOKUP_KEYED_CALLS.with(|c| c.set(c.get() + 1));
    }
    Ok(Some(table))
}

/// [`lookup_keyed`] for a join whose `filters` read only lookup columns
/// (none an `ON` key): a first read takes the keys and the filters' columns
/// of the keyed rows, and only the rows of keys some passing row holds are
/// read whole. Every other row keeps just those columns: the filters reject
/// it (single-valued key) or it only warns (multi-valued key), so its other
/// columns never reach the output. Each row is in the result once, a key's
/// rows in index order, so the join plan built over it ([`LookupJoinPlan::build`])
/// matches, rejects and warns as over the plain keyed read. Rally joins
/// `*_keys_where_*` over 100k keys: a chunk's 1,024 keyed rows were decoded
/// whole for the ~11% the filter keeps.
fn lookup_keyed_filtered(ctx: &mut Ctx, schema: &IndexSchema, on: &[String], keys: &[Vec<Val>], filters: &[Expr]) -> Result<Option<Table>> {
    #[cfg(test)]
    if LOOKUP_FILTERED_DISABLED.with(|c| c.get()) {
        return lookup_keyed(ctx, schema, on, keys);
    }
    let mut refs = Vec::new();
    for filter in filters {
        super::analyze::expr_refs(filter, &mut refs);
    }
    let lookup_col = |name: &str| schema.cols.iter().any(|(n, _)| n == name);
    if filters.is_empty() || refs.iter().any(|(name, _)| !lookup_col(name) || on.contains(name)) {
        return lookup_keyed(ctx, schema, on, keys);
    }
    let narrow: HashSet<String> = on.iter().cloned().chain(refs.into_iter().map(|(name, _)| name)).collect();
    let wide = schema.projection.as_ref().map_or(schema.cols.len(), HashSet::len);
    if narrow.len() >= wide {
        return lookup_keyed(ctx, schema, on, keys);
    }
    let thin_schema = IndexSchema { cols: schema.cols.clone(), targets: schema.targets.clone(), slots: None, projection: Some(narrow) };
    let Some(thin) = lookup_keyed(ctx, &thin_schema, on, keys)? else { return Ok(None) };
    let Some(key_columns) = on.iter().map(|k| thin.idx(k)).collect::<Option<Vec<usize>>>() else { return lookup_keyed(ctx, schema, on, keys) };
    let Ok(bound) = filters.iter().map(|f| bind(ctx, f, &thin.cols)).collect::<Result<Vec<_>>>() else { return lookup_keyed(ctx, schema, on, keys) };
    // The filters run again (with their warnings) when the plan is built:
    // here they only pick the rows to read whole.
    let saved = std::mem::take(&mut *ctx.fctx.warnings.borrow_mut());
    let mut passing: Vec<Vec<Val>> = Vec::new();
    let mut passing_keys: HashSet<Vec<String>> = HashSet::new();
    let mut failed = false;
    for row in &thin.rows {
        let key: Vec<Val> = key_columns.iter().map(|i| row[*i].clone()).collect();
        if key.iter().any(|v| v.is_null() || matches!(v, Val::Multi(_))) { continue; }
        match bound.iter().try_fold(true, |all, f| eval(f, row, &ctx.fctx).map(|v| all && v == Val::Bool(true))) {
            Ok(true) => {
                if passing_keys.insert(key.iter().map(Val::key).collect()) { passing.push(key); }
            }
            Ok(false) => {}
            Err(_) => { failed = true; break; }
        }
    }
    *ctx.fctx.warnings.borrow_mut() = saved;
    if failed {
        return lookup_keyed(ctx, schema, on, keys);
    }
    let mut rows: Vec<Vec<Val>> = Vec::new();
    if !passing.is_empty() {
        let Some(full) = lookup_keyed(ctx, schema, on, &passing)? else { return lookup_keyed(ctx, schema, on, keys) };
        // Multi-valued key rows come from the first read (once).
        rows.extend(full.rows.into_iter().filter(|row| key_columns.iter().all(|i| !matches!(row[*i], Val::Multi(_)))));
    }
    rows.extend(thin.rows.into_iter().filter(|row| {
        let multi = key_columns.iter().any(|i| matches!(row[*i], Val::Multi(_)));
        multi || !passing_keys.contains(&key_columns.iter().map(|i| row[*i].key()).collect::<Vec<_>>())
    }));
    Ok(Some(Table { cols: thin.cols, rows }))
}

/// The lookup rows matching `query` (all of them for `None`), in index
/// order. Local stored sources are read in batches (each store chunk
/// decoded once), not through a search response holding every hit.
fn lookup_rows(ctx: &mut Ctx, schema: &IndexSchema, query: Option<Value>) -> Result<Table> {
    let query = query.unwrap_or_else(|| json!({"match_all": {}}));
    if !lookup_reads_source(ctx, schema) {
        return load(ctx, schema, &[], query, vec![], None, &[]);
    }
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    let needed = needed_columns(ctx.full, schema);
    let wanted: Vec<bool> = schema.cols.iter().map(|(name, _)| needed.as_ref().is_none_or(|set| set.contains(name))).collect();
    let targets = node.snapshot(&schema.targets)?;
    let mut table = Table { cols: source_cols(schema, &[]), rows: Vec::new() };
    let mut memory = crate::memory::breakers().reserve_request(0, ESQL_LABEL)?;
    crate::search::for_each_match(&targets, &query, 1024, &mut |page| {
        let mut ids = page.doc_ids();
        loop {
            let docs: Vec<_> = ids.by_ref().take(1024).collect();
            if docs.is_empty() { break; }
            // Doc values where they give the `_source` values (keyword
            // keys, text columns' keyword sub-fields), else `_source`.
            #[cfg(test)]
            fast::LOOKUP_SIDE_READ.with(|side| side.set(true));
            let read = fast::lookup_key_rows(schema, &wanted, page.target, page.lookup, page.leaf, &docs, fast::doc_values_allowed(page.target));
            #[cfg(test)]
            fast::LOOKUP_SIDE_READ.with(|side| side.set(false));
            let (rows, loaded, _page_memory) = read?;
            ctx.documents_found += rows.len() as u64;
            ctx.values_loaded += loaded;
            memory.grow(fast::rows_bytes(&rows), ESQL_LABEL)?;
            table.rows.extend(rows);
        }
        Ok(true)
    })?;
    // The caller's plan charges the rows it keeps ([`lookup_plan_charge`]):
    // retaining this read's charge for the request as well counted every
    // keyed read of a streamed join (one per source chunk) until the
    // request ended, though each chunk's rows are dropped after it.
    drop(memory);
    Ok(table)
}

/// The text of the command at `pos` (to its pipe), for warnings.
fn command_src(src: &str, pos: usize) -> String {
    let rest = src.get(pos..).unwrap_or("");
    let mut quote = false;
    let b = rest.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\\' if quote => i += 1,
            b'"' => quote = !quote,
            b'|' if !quote => break,
            _ => {}
        }
        i += 1;
    }
    rest[..i.min(rest.len())].trim_end().to_string()
}

/// ES's warning for a multi-valued join key (`LOOKUP JOIN encountered
/// multi-value`, the row joins nothing), at the command's position.
fn lookup_mv_warning(f: &FnCtx, pos: usize) {
    let loc = warning_location(&f.query, pos);
    let src = command_src(&f.query, pos);
    f.warnings.borrow_mut().register(&loc, &format!("evaluation of [{src}] failed, treating result as null"), "java.lang.IllegalArgumentException: LOOKUP JOIN encountered multi-value");
}

/// The lookup index of a `LOOKUP JOIN`, validated (one index in `lookup` mode),
/// reading only the join's `fields` (the command's; `None`: every column).
fn lookup_schema(ctx: &mut Ctx, index: &str, index_pos: usize, fields: Option<&HashSet<String>>) -> Result<IndexSchema> {
    let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
    // The right side must be exactly one index in `lookup` mode.
    let mut opts = ResolveOpts::from_params(&HashMap::new());
    opts.allow_no_indices = true;
    let targets = match node.resolve(index, opts) {
        Ok(t) if !t.is_empty() => t,
        _ => return Err(verification_err(ctx.full, index_pos, format!("Unknown index [{index}]"))),
    };
    let mut names: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
    names.sort();
    names.dedup();
    if names.len() > 1 {
        return Err(verification_err(ctx.full, index_pos, format!("Lookup Join requires a single lookup mode index; [{index}] resolves to multiple indices")));
    }
    let mode = targets[0].svc.setting("index.mode").map(|m| m.to_ascii_lowercase()).unwrap_or_else(|| "standard".into());
    if mode != "lookup" {
        return Err(verification_err(ctx.full, index_pos, format!("Lookup Join requires a single lookup mode index; [{index}] resolves to [{}] in [{mode}] mode", names[0])));
    }
    let mut schema = resolve_schema(ctx, &[index.to_string()], index_pos)?;
    schema.projection = fields.cloned();
    Ok(schema)
}

/// Offset of the query's source command (`line 1:1` for a plain `FROM`).
fn source_cmd_pos(full: &str) -> usize {
    let up = full.to_ascii_uppercase();
    ["FROM", "TS", "ROW"].iter().filter_map(|k| up.find(k)).min().unwrap_or(0)
}

#[cfg(test)]
std::thread_local! { static JOIN_ROW_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; static JOIN_CANDIDATES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
thread_local! { static JOIN_RANGE_INDEX_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
thread_local! {
    /// Tests: an expression join keeps the lookup rows its filters reject.
    static JOIN_EXPR_PREFILTER_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `LOOKUP JOIN index ON <expression>`: each left row joins the lookup rows
/// satisfying the condition (column references resolve to the side that
/// has them); output = left columns not shadowed + all lookup columns.
struct LookupExprPlan {
    right: Table,
    bound: Expr,
    eq: Vec<(usize, usize)>,
    index_map: Option<HashMap<Vec<String>, Vec<usize>>>,
    ranges: Vec<(usize, usize, BinOp)>,
    /// Per range conjunct's lookup column, the rows with a single value
    /// there, in value order.
    ordered: Vec<(usize, Vec<usize>)>,
    lkeep: Vec<usize>,
    out_cols: Vec<Column>,
    filters: Vec<Expr>,
    all: Vec<usize>,
    /// The lookup side and its key index, on the request breaker for the
    /// life of the join (the fast path keeps a plan for the whole query).
    _charge: crate::memory::Reservation,
}

/// Reserve key serialization/index rows before building the lookup map.
/// Repeated ON/equality columns can duplicate one wide value many times.
/// The request-breaker charge of a join plan over `right`: the lookup rows,
/// a copy of each row's key values (at most the row's own bytes) with a
/// string header per key column, and the key index entry. (Each row's bytes
/// were multiplied by `16 * key columns + 4`: a 500k-row lookup side of
/// ~1 KB rows with two keys asked ~18 GB, and Rally joins
/// `esql_lookup_join_500k_two_keys_where_one_matching_equals` tripped the
/// request breaker.)
fn lookup_plan_charge(right:&Table,key_columns:usize)->Result<crate::memory::Reservation> {
    let per_key=(key_columns as u64).saturating_mul(48);
    let bytes=right.rows.iter().fold(4096u64,|sum,row|sum.saturating_add(row_bytes(row).saturating_mul(2)).saturating_add(per_key).saturating_add(128));
    crate::memory::breakers().reserve_request(bytes,ESQL_LABEL)
}

fn join_row_scratch(left:&[Val],right:Option<&[Val]>,output_columns:usize)->Result<crate::memory::Reservation> {
    let bytes=row_bytes(left).saturating_add(right.map_or(0,row_bytes)).saturating_mul(8).saturating_add((output_columns as u64).saturating_mul(128)).saturating_add(256);
    crate::memory::breakers().reserve_request(bytes,ESQL_LABEL)
}

fn retain_join_output(mut memory:crate::memory::Reservation,cols:&[Column])->Result<()> {
    memory.grow(cols.iter().fold(64u64,|sum,col|sum.saturating_add((col.name.len()+col.ty.len()) as u64*2+128)),ESQL_LABEL)?;
    memory.retain_for_request();Ok(())
}

impl LookupExprPlan {
    fn prepare(ctx: &mut Ctx, left: &Table, index: &str, index_pos: usize, cond: &Expr, filters: &[Expr], fields: Option<&HashSet<String>>) -> Result<Self> {
        let mut right = lookup_right(ctx, index, index_pos, fields)?;
        const L: &str = "\u{1}l:";
        const R: &str = "\u{1}r:";
        // Resolve references.
        fn walk(e: &Expr, f: &mut dyn FnMut(&str, usize) -> Result<String>) -> Result<Expr> {
            Ok(match e {
                Expr::Col(n, p) => Expr::Col(f(n, *p)?, *p),
                Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(walk(a, f)?), Box::new(walk(b, f)?)),
                Expr::Not(a) => Expr::Not(Box::new(walk(a, f)?)),
                Expr::Neg(a) => Expr::Neg(Box::new(walk(a, f)?)),
                Expr::IsNull(a, n) => Expr::IsNull(Box::new(walk(a, f)?), *n),
                Expr::Cast(a, t) => Expr::Cast(Box::new(walk(a, f)?), t.clone()),
                Expr::In(a, l, n) => Expr::In(Box::new(walk(a, f)?), l.iter().map(|x| walk(x, f)).collect::<Result<_>>()?, *n),
                Expr::Like(a, p, n, r) => Expr::Like(Box::new(walk(a, f)?), p.clone(), *n, *r),
                Expr::Between(a, b, c, n) => Expr::Between(Box::new(walk(a, f)?), Box::new(walk(b, f)?), Box::new(walk(c, f)?), *n),
                Expr::Call(n, a, p) => Expr::Call(n.clone(), a.iter().map(|x| walk(x, f)).collect::<Result<_>>()?, *p),
                o => o.clone(),
            })
        }
        let full = ctx.full;
        let src_pos = source_cmd_pos(full);
        let cond = walk(cond, &mut |n, p| {
            match (left.idx(n).is_some(), right.idx(n).is_some()) {
                (true, true) => {
                    let (a, b) = (super::pos_prefix(full, src_pos), super::pos_prefix(full, index_pos));
                    Err(EsError::new(400, "verification_exception", format!("Found ambiguous reference to [{n}]; matches any of [{} [{n}], {} [{n}]]", a, b)))
                }
                (true, false) => Ok(format!("{L}{n}")),
                (false, true) => Ok(format!("{R}{n}")),
                (false, false) => Err(verification_err(full, p, format!("Unknown column [{n}]"))),
            }
        })?;
        let mut cols: Vec<Column> = left.cols.iter().map(|c| Column { name: format!("{L}{}", c.name), ty: c.ty.clone() }).collect();
        cols.extend(right.cols.iter().map(|c| Column { name: format!("{R}{}", c.name), ty: c.ty.clone() }));
        let bound = bind(ctx, &cond, &cols)?;
        // Hash every compatible cross-side equality. Constant leading keys
        // must not turn a selective composite equijoin into a nested-loop scan.
        let nl = left.cols.len();
        let mut conj = Vec::new();
        fn conjuncts<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
            match e {
                Expr::Bin(BinOp::And, a, b) => {
                    conjuncts(a, out);
                    conjuncts(b, out);
                }
                o => out.push(o),
            }
        }
        conjuncts(&bound, &mut conj);
        let lkeep: Vec<usize> = (0..nl).filter(|i| right.idx(&left.cols[*i].name).is_none()).collect();
        let mut out_cols: Vec<Column> = lkeep.iter().map(|i| left.cols[*i].clone()).collect();
        out_cols.extend(right.cols.iter().cloned());
        let filters = filters.iter().map(|e| bind(ctx, e, &out_cols)).collect::<Result<Vec<_>>>()?;
        // Filters over lookup columns only that reject the unmatched (null)
        // row: a lookup row they reject never makes an output row (a source
        // row whose matches are all rejected outputs nothing, as unmatched
        // it outputs its rejected null row), so it is dropped before the
        // join (Rally joins `*_expanding_*_filter1000`: 1,000 of 10,000
        // rows pass). Only with no multi-valued lookup value and no warning
        // from the filters (the join evaluates neither on dropped rows).
        if !filters.is_empty() && filters.iter().all(|expr| lookup_right_only(expr, lkeep.len()))
            && !join_row_passes(ctx, &filters, &vec![Val::Null; out_cols.len()])?
            && !right.rows.iter().any(|row| row.iter().any(|v| matches!(v, Val::Multi(_)))) {
            let saved = std::mem::take(&mut *ctx.fctx.warnings.borrow_mut());
            let mut keep = Vec::with_capacity(right.rows.len());
            let mut ok = true;
            for row in &right.rows {
                let mut combined = vec![Val::Null; lkeep.len()];
                combined.extend(row.iter().cloned());
                match join_row_passes(ctx, &filters, &combined) {
                    Ok(passes) => keep.push(passes),
                    Err(_) => { ok = false; break; }
                }
            }
            ok &= ctx.fctx.warnings.borrow().list.is_empty();
            *ctx.fctx.warnings.borrow_mut() = saved;
            #[cfg(test)]
            { ok &= !JOIN_EXPR_PREFILTER_DISABLED.with(|disabled| disabled.get()); }
            if ok {
                let mut at = keep.into_iter();
                right.rows.retain(|_| at.next().unwrap_or(true));
            }
        }
        fn hash_type(ty: &str) -> Option<&str> {
            match ty {
                "integer" | "long" => Some("integer"),
                "keyword" | "text" => Some("string"),
                "double" | "datetime" | "boolean" => Some(ty),
                _ => None,
            }
        }
        let eq: Vec<(usize, usize)> = conj.iter().filter_map(|c| match c {
            Expr::Bin(BinOp::Eq, a, b) => match (a.as_ref(), b.as_ref()) {
                (Expr::Idx(x), Expr::Idx(y)) if *x < nl && *y >= nl => Some((*x, *y - nl)),
                (Expr::Idx(x), Expr::Idx(y)) if *y < nl && *x >= nl => Some((*y, *x - nl)),
                _ => None,
            },
            _ => None,
        }).filter(|(li, ri)| {
            let ty = hash_type(&left.cols[*li].ty);
            ty.is_some() && ty == hash_type(&right.cols[*ri].ty)
        }).collect();
        let _charge=lookup_plan_charge(&right,eq.len())?;
        let index_map: Option<HashMap<Vec<String>, Vec<usize>>> = (!eq.is_empty()).then(|| {
            let mut m: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
            for (k, r) in right.rows.iter().enumerate() {
                if eq.iter().all(|(_, ri)| !r[*ri].is_null() && !matches!(r[*ri], Val::Multi(_))) {
                    m.entry(eq.iter().map(|(_, ri)| r[*ri].key()).collect()).or_default().push(k);
                }
            }
            m
        });
        // An inequality-only join otherwise tests every lookup row per source
        // row. Index one right-side column, narrow its interval using every
        // compatible conjunct, then evaluate the original predicate as usual.
        let ranges: Vec<(usize, usize, BinOp)> = conj.iter().filter_map(|c| match c {
            Expr::Bin(op, a, b) if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) => {
                match (a.as_ref(), b.as_ref()) {
                    (Expr::Idx(l), Expr::Idx(r)) if *l < nl && *r >= nl => Some((*l, *r - nl, *op)),
                    (Expr::Idx(r), Expr::Idx(l)) if *l < nl && *r >= nl => Some((*l, *r - nl, match op {
                        BinOp::Lt => BinOp::Gt, BinOp::Le => BinOp::Ge,
                        BinOp::Gt => BinOp::Lt, _ => BinOp::Le,
                    })),
                    _ => None,
                }
            }
            _ => None,
        }).filter(|(l, r, _)| {
            let ty = hash_type(&left.cols[*l].ty);
            matches!(ty, Some("string" | "integer" | "datetime")) && ty == hash_type(&right.cols[*r].ty)
        }).collect();
        #[cfg(test)]
        let use_range_index = !JOIN_RANGE_INDEX_DISABLED.with(|disabled| disabled.get());
        #[cfg(not(test))]
        let use_range_index = true;
        // Every range conjunct's lookup column indexed: a source row's
        // candidates are the narrowest of their intervals (Rally joins
        // `*_gte_lte_*`: the first conjunct, on a constant key, kept every
        // lookup row).
        let ordered: Vec<(usize, Vec<usize>)> = if use_range_index && index_map.is_none() {
            let mut columns: Vec<usize> = ranges.iter().map(|&(_, ri, _)| ri).collect();
            columns.dedup();
            columns.sort_unstable();
            columns.dedup();
            columns.into_iter().map(|ri| {
                let mut ids: Vec<_> = right.rows.iter().enumerate().filter_map(|(i, row)|
                    (!row[ri].is_null() && !matches!(row[ri], Val::Multi(_))).then_some(i)).collect();
                ids.sort_by(|&a, &b| func::compare(&right.rows[a][ri], &right.rows[b][ri]).unwrap().then(a.cmp(&b)));
                (ri, ids)
            }).collect()
        } else { Vec::new() };
        let all: Vec<usize> = (0..right.rows.len()).collect();
        Ok(Self { right, bound, eq, index_map, ranges, ordered, lkeep, out_cols, filters, all, _charge })
    }

    fn equality_keys_cover_condition(&self, left_columns: usize) -> bool {
        fn covered(expr: &Expr, eq: &[(usize, usize)], nl: usize) -> bool {
            match expr {
                Expr::Bin(BinOp::And, a, b) => covered(a, eq, nl) && covered(b, eq, nl),
                Expr::Bin(BinOp::Eq, a, b) => match (a.as_ref(), b.as_ref()) {
                    (Expr::Idx(a), Expr::Idx(b)) => eq.iter().any(|(l,r)| (*a == *l && *b == nl + r) || (*b == *l && *a == nl + r)),
                    _ => false,
                },
                _ => false,
            }
        }
        self.index_map.is_some() && covered(&self.bound, &self.eq, left_columns)
    }

    fn can_count_keys(&self, left_columns: usize) -> bool {
        self.filters.is_empty() && self.equality_keys_cover_condition(left_columns)
    }

    /// Only reject source rows when equality keys completely cover the join
    /// condition and the post-join predicate rejects null outer rows.
    fn accepted_filter_keys(&self, ctx: &Ctx, left_columns: usize) -> Result<Option<HashSet<Vec<String>>>> {
        if self.filters.is_empty() || !self.equality_keys_cover_condition(left_columns)
            || !self.filters.iter().all(|expr| lookup_right_only(expr, self.lkeep.len())) {
            return Ok(None);
        }
        if join_row_passes(ctx, &self.filters, &vec![Val::Null; self.out_cols.len()])? { return Ok(None); }
        let mut keys = HashSet::new();
        for (key, matches) in self.index_map.as_ref().unwrap() {
            for &index in matches {
                let mut row = vec![Val::Null; self.lkeep.len()];
                row.extend(self.right.rows[index].iter().cloned());
                if join_row_passes(ctx, &self.filters, &row)? { keys.insert(key.clone()); break; }
            }
        }
        Ok(Some(keys))
    }

    fn count_rows(&self, left: &Table) -> Result<u64> {
        let mut total = 0u64;
        let index = self.index_map.as_ref().ok_or_else(|| EsError::internal("missing join count index"))?;
        for row in &left.rows {
            let matches = if self.eq.iter().any(|(i,_)| row[*i].is_null() || matches!(row[*i],Val::Multi(_))) { None } else {
                let key: Vec<String> = self.eq.iter().map(|(i,_)| row[*i].key()).collect();
                index.get(&key)
            };
            total = total.checked_add(matches.map_or(1, |rows| rows.len() as u64))
                .ok_or_else(|| EsError::internal("join count overflow"))?;
        }
        Ok(total)
    }

    fn apply(&self, ctx: &Ctx, left: &Table, limit: Option<usize>) -> Result<Table> {
        let Self { right, bound, eq, index_map, ranges, ordered, lkeep, out_cols, filters, all, .. } = self;
        let mut rows = Vec::new();
        // Rows are charged as they are built (ES BlockFactory charges each
        // block): a join that multiplies its input trips before it is all
        // in memory.
        let mut pages = crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
        'left: for l in &left.rows {
            if limit.is_some_and(|n| rows.len() >= n) { break; }
            let _probe_memory=crate::memory::breakers().reserve_request(row_bytes(l).saturating_mul(16).saturating_add((right.rows.len() as u64).saturating_mul(16)),ESQL_LABEL)?;
            let mut interval_candidates = Vec::new();
            if !ordered.is_empty() {
                // Each indexed column's interval; the narrowest is searched
                // (the condition is evaluated on every candidate, so any of
                // them holds every match).
                let mut best: Option<&[usize]> = None;
                for (ri, ids) in ordered {
                    let (mut lo, mut hi) = (0, ids.len());
                    for &(li, r, op) in ranges {
                        if r != *ri { continue; }
                        let value = &l[li];
                        if value.is_null() || matches!(value, Val::Multi(_)) { hi = 0; break; }
                        let less = || ids.partition_point(|&i| func::compare(&right.rows[i][*ri], value) == Some(std::cmp::Ordering::Less));
                        let le = || ids.partition_point(|&i| func::compare(&right.rows[i][*ri], value) != Some(std::cmp::Ordering::Greater));
                        match op {
                            BinOp::Lt => lo = lo.max(le()),
                            BinOp::Le => lo = lo.max(less()),
                            BinOp::Gt => hi = hi.min(less()),
                            BinOp::Ge => hi = hi.min(le()),
                            _ => unreachable!(),
                        }
                    }
                    let slice = if lo < hi { &ids[lo..hi] } else { &ids[0..0] };
                    if best.is_none_or(|b| slice.len() < b.len()) { best = Some(slice); }
                }
                interval_candidates.extend_from_slice(best.unwrap_or(&[]));
                // Keep lookup input order and therefore existing LIMIT semantics.
                interval_candidates.sort_unstable();
            }
            let cands: &[usize] = match &index_map {
                Some(m) if eq.iter().all(|(li, _)| !l[*li].is_null() && !matches!(l[*li], Val::Multi(_))) => {
                    m.get(&eq.iter().map(|(li, _)| l[*li].key()).collect::<Vec<_>>()).map(Vec::as_slice).unwrap_or(&[])
                }
                Some(_) => &[],
                None if !ordered.is_empty() => &interval_candidates,
                None => &all,
            };
            let mut matched = false;
            for &k in cands {
                #[cfg(test)]
                JOIN_CANDIDATES.with(|n| n.set(n.get() + 1));
                let _scratch=join_row_scratch(l,Some(&right.rows[k]),out_cols.len())?;
                let mut combined:Vec<Val>=Vec::with_capacity(left.cols.len()+right.cols.len());
                combined.extend(l.iter().cloned());
                combined.extend(right.rows[k].iter().cloned());
                if eval(&bound, &combined, &ctx.fctx)? == Val::Bool(true) {
                    matched = true;
                    let mut row: Vec<Val> = lkeep.iter().map(|i| l[*i].clone()).collect();
                    row.extend(right.rows[k].iter().cloned());
                    if join_row_passes(ctx, &filters, &row)? { pages.grow(row_bytes(&row).saturating_mul(2), ESQL_LABEL)?; rows.push(row); }
                    if limit.is_some_and(|n| rows.len() >= n) { break 'left; }
                }
            }
            if !matched {
                let _scratch=join_row_scratch(l,None,out_cols.len())?;
                let mut row: Vec<Val> = lkeep.iter().map(|i| l[*i].clone()).collect();
                row.extend(right.cols.iter().map(|_| Val::Null));
                if join_row_passes(ctx, &filters, &row)? { pages.grow(row_bytes(&row).saturating_mul(2), ESQL_LABEL)?; rows.push(row); }
            }
        }
        retain_join_output(pages,out_cols)?;
        Ok(Table { cols: out_cols.clone(), rows })
    }
}

#[allow(clippy::too_many_arguments)]
fn lookup_join_expr(ctx: &mut Ctx, left: Table, index: &str, index_pos: usize, cond: &Expr, limit: Option<usize>, filters: &[Expr], fields: Option<&HashSet<String>>) -> Result<Table> {
    LookupExprPlan::prepare(ctx, &left, index, index_pos, cond, filters, fields)?.apply(ctx, &left, limit)
}

/// COUNT(*) after an equijoin does not need the expanded rows. Evaluate
/// deterministic right-only predicates once per lookup row and accumulate
/// matching multiplicities by key. A key with matches rejected by WHERE
/// contributes zero; an unmatched left row still contributes its null row.
fn lookup_count(ctx: &mut Ctx, left: &Table, command: &Cmd, tail: &[Cmd]) -> Result<Option<(Table, usize)>> {
    let Cmd::LookupJoin { index, index_pos, on, on_pos, cond: None, pos, fields } = command else { return Ok(None) };
    let nfilters = tail.iter().take_while(|c| matches!(c, Cmd::Where(..))).count();
    let Some(stats @ Cmd::Stats { aggs, by, inline: false, .. }) = tail.get(nfilters) else { return Ok(None) };
    if !by.is_empty() || aggs.is_empty() || !aggs.iter().all(|f| f.filter.is_none() && matches!(&f.expr, Expr::Call(name, args, _) if name == "COUNT" && matches!(args.as_slice(), [Expr::Star]))) {
        return Ok(None);
    }
    // Only the lookup rows the left keys can match (a keyed read).
    let schema = lookup_schema(ctx, index, *index_pos, fields.as_ref())?;
    let keyed = match on.iter().map(|k| left.idx(k)).collect::<Option<Vec<usize>>>() {
        Some(lk) => {
            let keys: Vec<Vec<Val>> = left.rows.iter().map(|r| lk.iter().map(|i| r[*i].clone()).collect()).collect();
            lookup_keyed(ctx, &schema, on, &keys)?
        }
        None => None,
    };
    let right = match keyed { Some(right) => right, None => lookup_rows(ctx, &schema, None)? };
    // The lookup table (charged row by row while it loaded) and the key
    // counts are held on the request breaker while the counts are made and
    // used: charged as soon as the load returns.
    let held = crate::memory::RequestMemory::new();
    held.charge(table_bytes(&right), ESQL_LABEL)?;
    let rcols: Vec<usize> = (0..right.cols.len()).filter(|i| !on.contains(&right.cols[*i].name)).collect();
    // Deliberately narrow: no random functions, full-text calls, or left
    // references may be moved out of the per-left-row evaluation.
    fn supported(e: &Expr, right: &Table, cols: &[usize]) -> bool {
        match e {
            Expr::Lit(_) => true,
            Expr::Col(name, _) => cols.iter().any(|i| right.cols[*i].name == *name),
            Expr::Bin(_, a, b) => supported(a, right, cols) && supported(b, right, cols),
            Expr::Not(a) | Expr::Neg(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) => supported(a, right, cols),
            _ => false,
        }
    }
    if tail[..nfilters].iter().any(|c| !matches!(c, Cmd::Where(e, _) if supported(e, &right, &rcols))) { return Ok(None); }
    let mut lkeys = Vec::new();
    let mut rkeys = Vec::new();
    for (i, key) in on.iter().enumerate() {
        let pos = on_pos.get(i).copied().unwrap_or(*index_pos);
        lkeys.push(left.idx(key).ok_or_else(|| verification_err(ctx.full, pos, format!("Unknown column [{key}] in left side of join")))?);
        let ri = right.idx(key).ok_or_else(|| verification_err(ctx.full, pos, format!("Unknown column [{key}] in right side of join")))?;
        if matches!(right.cols[ri].ty.as_str(), "text" | "flattened") { return Ok(None); }
        rkeys.push(ri);
    }
    let predicates = tail[..nfilters].iter().map(|c| match c { Cmd::Where(e, _) => bind(ctx, e, &right.cols), _ => unreachable!() }).collect::<Result<Vec<_>>>()?;
    let passes = |row: &[Val]| -> Result<bool> {
        for predicate in &predicates { if eval(predicate, row, &ctx.fctx)? != Val::Bool(true) { return Ok(false); } }
        Ok(true)
    };
    let null_passes = passes(&vec![Val::Null; right.cols.len()])?;
    let mut counts: HashMap<Vec<String>, u64> = HashMap::new();
    // Multi-valued lookup keys never match (ES), and warn when reached.
    let mut mv_keys: HashSet<Vec<String>> = HashSet::new();
    for row in &right.rows {
        if rkeys.iter().any(|i| matches!(row[*i], Val::Multi(_))) {
            mv_keys.extend(key_combinations(&rkeys.iter().map(|i| &row[*i]).collect::<Vec<_>>()));
            continue;
        }
        let key: Vec<String> = rkeys.iter().map(|i| row[*i].key()).collect();
        if !counts.contains_key(&key) {
            held.add(64 + key.iter().map(|s| 24 + s.capacity() as u64).sum::<u64>(), ESQL_LABEL)?;
        }
        let count = counts.entry(key).or_insert(0);
        if passes(row)? { *count += 1; }
    }
    let mut total = 0u64;
    for row in &left.rows {
        let count = if lkeys.iter().any(|i| row[*i].is_null()) { None }
            else if lkeys.iter().any(|i| matches!(row[*i], Val::Multi(_))) { lookup_mv_warning(&ctx.fctx, *pos); None }
            else {
                let key: Vec<String> = lkeys.iter().map(|i| row[*i].key()).collect();
                if mv_keys.contains(&key) { lookup_mv_warning(&ctx.fctx, *pos); }
                counts.get(&key).copied()
            };
        total += count.unwrap_or(u64::from(null_passes));
    }
    let mut cols: Vec<Column> = left.cols.iter().filter(|c| !rcols.iter().any(|i| right.cols[*i].name == c.name)).cloned().collect();
    cols.extend(rcols.iter().map(|i| right.cols[*i].clone()));
    track_fields(ctx, command, left);
    let mut out = apply(ctx, stats, Table { cols, rows: vec![] })?;
    for value in &mut out.rows[0] { *value = Val::Long(i64::try_from(total).map_err(|_| EsError::internal("join count overflow"))?); }
    Ok(Some((out, nfilters + 1)))
}

// Only deterministic scalar expressions admitted by filtered_join_limit.
// Bound positions account for lookup columns shadowing source columns.
fn lookup_right_only(expr: &Expr, first_right: usize) -> bool {
    match expr {
        Expr::Lit(_) => true,
        Expr::Idx(i) => *i >= first_right,
        Expr::Bin(_, a, b) => lookup_right_only(a, first_right) && lookup_right_only(b, first_right),
        Expr::Not(a) | Expr::Neg(a) | Expr::IsNull(a, _) | Expr::Cast(a, _) | Expr::Like(a, ..) => lookup_right_only(a, first_right),
        Expr::In(a, args, _) => lookup_right_only(a, first_right) && args.iter().all(|e| lookup_right_only(e, first_right)),
        Expr::Between(a, b, c, _) => [a, b, c].iter().all(|e| lookup_right_only(e, first_right)),
        Expr::Call(name, args, _) if name.eq_ignore_ascii_case("CONCAT") => args.iter().all(|e| lookup_right_only(e, first_right)),
        _ => false,
    }
}

struct LookupJoinPlan {
    right: Table,
    lkeys: Vec<usize>,
    lkeep: Vec<usize>,
    rcols: Vec<usize>,
    cols: Vec<Column>,
    index_map: HashMap<Vec<String>, Vec<usize>>,
    /// Key tuples (`Val::key`s) a multi-valued lookup row holds: matching
    /// one warns (ES), the row itself never joins.
    mv_keys: HashSet<Vec<String>>,
    /// The command's offset (warnings).
    pos: usize,
    filters: Vec<Expr>,
    right_only_null_passes: Option<bool>,
    /// The lookup side and its key index, on the request breaker for the
    /// life of the join.
    _charge: crate::memory::Reservation,
}

/// Every key tuple a lookup row with multi-valued keys holds (one value
/// per key), capped: a huge product only loses a warning, never a match.
fn key_combinations(values: &[&Val]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = vec![Vec::new()];
    for v in values {
        let parts: Vec<String> = match v {
            Val::Multi(m) => m.iter().map(Val::key).collect(),
            v => vec![v.key()],
        };
        let mut next = Vec::with_capacity((out.len() * parts.len()).min(4096));
        'all: for prefix in &out {
            for p in &parts {
                let mut k = prefix.clone();
                k.push(p.clone());
                next.push(k);
                if next.len() >= 4096 { break 'all; }
            }
        }
        out = next;
    }
    out
}

impl LookupJoinPlan {
    /// The plan over the whole lookup index (any left rows; the streamed
    /// paths build it once, before reading the left side).
    #[allow(clippy::too_many_arguments)]
    fn prepare(ctx: &mut Ctx, left: &Table, index: &str, index_pos: usize, on: &[String], on_pos: &[usize], filters: &[Expr], pos: usize, fields: Option<&HashSet<String>>) -> Result<Self> {
        let right = lookup_right(ctx, index, index_pos, fields)?;
        Self::build(ctx, left, right, index_pos, on, on_pos, filters, pos)
    }

    /// The plan for the left rows `rows` (with `left`'s columns): keyed
    /// when every `ON` column is on the left and the lookup keys allow it.
    #[allow(clippy::too_many_arguments)]
    fn for_rows(ctx: &mut Ctx, left: &Table, rows: &[Vec<Val>], schema: &IndexSchema, index_pos: usize, on: &[String], on_pos: &[usize], filters: &[Expr], pos: usize) -> Result<Self> {
        let keyed = match on.iter().map(|k| left.idx(k)).collect::<Option<Vec<usize>>>() {
            Some(lkeys) => {
                let keys: Vec<Vec<Val>> = rows.iter().map(|r| lkeys.iter().map(|i| r[*i].clone()).collect()).collect();
                lookup_keyed(ctx, schema, on, &keys)?
            }
            // A missing key column: the full plan reports it, as before.
            None => None,
        };
        let right = match keyed {
            Some(right) => right,
            None => lookup_rows(ctx, schema, None)?,
        };
        Self::build(ctx, left, right, index_pos, on, on_pos, filters, pos)
    }

    /// [`Self::for_rows`] over only the lookup rows `extra` matches.
    #[allow(clippy::too_many_arguments)]
    fn for_rows_with(ctx: &mut Ctx, left: &Table, rows: &[Vec<Val>], schema: &IndexSchema, index_pos: usize, on: &[String], on_pos: &[usize], filters: &[Expr], pos: usize, extra: &Value) -> Result<Self> {
        let keyed = match on.iter().map(|k| left.idx(k)).collect::<Option<Vec<usize>>>() {
            Some(lkeys) => {
                let keys: Vec<Vec<Val>> = rows.iter().map(|r| lkeys.iter().map(|i| r[*i].clone()).collect()).collect();
                lookup_keyed_with(ctx, schema, on, &keys, Some(extra))?
            }
            None => None,
        };
        let right = match keyed {
            Some(right) => right,
            None => lookup_rows(ctx, schema, Some(extra.clone()))?,
        };
        Self::build(ctx, left, right, index_pos, on, on_pos, filters, pos)
    }

    /// The plan for the key tuples `keys` (`ON` order; the left side's
    /// columns from `left`): `None` when a keyed read does not apply.
    #[allow(clippy::too_many_arguments)]
    fn for_keys(ctx: &mut Ctx, left: &Table, keys: &[Vec<Val>], schema: &IndexSchema, index_pos: usize, on: &[String], on_pos: &[usize], filters: &[Expr], pos: usize) -> Result<Option<Self>> {
        match lookup_keyed_filtered(ctx, schema, on, keys, filters)? {
            Some(right) => Self::build(ctx, left, right, index_pos, on, on_pos, filters, pos).map(Some),
            None => Ok(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build(ctx: &mut Ctx, left: &Table, right: Table, index_pos: usize, on: &[String], on_pos: &[usize], filters: &[Expr], pos: usize) -> Result<Self> {
        let mut lkeys = Vec::new();
        let mut rkeys = Vec::new();
        for (n, k) in on.iter().enumerate() {
            let kp = on_pos.get(n).copied().unwrap_or(index_pos);
            lkeys.push(left.idx(k).ok_or_else(|| verification_err(ctx.full, kp, format!("Unknown column [{k}] in left side of join")))?);
            let ri = right.idx(k).ok_or_else(|| verification_err(ctx.full, kp, format!("Unknown column [{k}] in right side of join")))?;
            if matches!(right.cols[ri].ty.as_str(), "text" | "flattened") {
                return Err(verification_err(ctx.full, kp, format!("JOIN with right field [{k}] of type [{}] is not supported", right.cols[ri].ty.to_ascii_uppercase())));
            }
            rkeys.push(ri);
        }
        let rcols: Vec<usize> = (0..right.cols.len()).filter(|i| !rkeys.contains(i)).collect();
        // Keep key components separate: keyword values may contain any separator.
        let _charge=lookup_plan_charge(&right,rkeys.len())?;
        let mut index_map: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        // ES matches single-valued lookup keys only (`SingleValueMatchQuery`
        // around each term query): a multi-valued one joins nothing and warns
        // when a left key reaches it.
        let mut mv_keys: HashSet<Vec<String>> = HashSet::new();
        for (ri, r) in right.rows.iter().enumerate() {
            if rkeys.iter().any(|i| matches!(r[*i], Val::Multi(_))) {
                mv_keys.extend(key_combinations(&rkeys.iter().map(|i| &r[*i]).collect::<Vec<_>>()));
                continue;
            }
            let k: Vec<String> = rkeys.iter().map(|i| r[*i].key()).collect();
            index_map.entry(k).or_default().push(ri);
        }
        let lkeep: Vec<usize> = (0..left.cols.len()).filter(|i| !rcols.iter().any(|r| right.cols[*r].name == left.cols[*i].name)).collect();
        let mut cols: Vec<Column> = lkeep.iter().map(|i| left.cols[*i].clone()).collect();
        cols.extend(rcols.iter().map(|i| right.cols[*i].clone()));
        let filters = filters.iter().map(|e| bind(ctx, e, &cols)).collect::<Result<Vec<_>>>()?;
        // A right-only predicate is invariant across source rows. Keep keys
        // whose original matches all failed as empty lists: those rows are
        // matched-but-rejected, not unmatched outer-join rows.
        let right_only_null_passes = if !filters.is_empty() && filters.iter().all(|expr| lookup_right_only(expr, lkeep.len())) {
            let mut accepted = Vec::with_capacity(right.rows.len());
            for row in &right.rows {
                let mut combined = vec![Val::Null; lkeep.len()];
                combined.extend(rcols.iter().map(|i| row[*i].clone()));
                accepted.push(join_row_passes(ctx, &filters, &combined)?);
            }
            for matches in index_map.values_mut() { matches.retain(|i| accepted[*i]); }
            Some(join_row_passes(ctx, &filters, &vec![Val::Null; cols.len()])?)
        } else { None };
        Ok(Self { right, lkeys, lkeep, rcols, cols, index_map, mv_keys, pos, filters, right_only_null_passes, _charge })
    }

    /// The lookup rows `l` joins (`None`: none, the null row), with ES's
    /// multi-value warnings: a multi-valued left key matches nothing
    /// (`QueryList.onlySingleValues`), nor does a multi-valued lookup key.
    fn matches(&self, ctx: &Ctx, l: &[Val]) -> Option<&Vec<usize>> {
        if self.lkeys.iter().any(|i| l[*i].is_null()) {
            return None;
        }
        if self.lkeys.iter().any(|i| matches!(l[*i], Val::Multi(_))) {
            lookup_mv_warning(&ctx.fctx, self.pos);
            return None;
        }
        let k: Vec<String> = self.lkeys.iter().map(|i| l[*i].key()).collect();
        if !self.mv_keys.is_empty() && self.mv_keys.contains(&k) {
            lookup_mv_warning(&ctx.fctx, self.pos);
        }
        self.index_map.get(&k)
    }

    fn rejects_all(&self) -> bool {
        self.right_only_null_passes == Some(false) && self.index_map.values().all(Vec::is_empty)
    }

    fn count_rows(&self, ctx: &Ctx, left: &Table) -> Result<u64> {
        let mut total = 0u64;
        for row in &left.rows {
            let matches = self.matches(ctx, row);
            let count = matches.map(|rows| rows.len() as u64)
                .unwrap_or(u64::from(self.right_only_null_passes.unwrap_or(true)));
            total = total.checked_add(count).ok_or_else(|| EsError::internal("join count overflow"))?;
        }
        Ok(total)
    }

    fn apply(&self, ctx: &Ctx, left: &Table, limit: Option<usize>) -> Result<Table> {
        let Self { right, lkeys, lkeep, rcols, cols, filters, right_only_null_passes, .. } = self;
        let mut rows = Vec::new();
        // Rows are charged as they are built (ES BlockFactory charges each
        // block): a join that multiplies its input trips before it is all
        // in memory.
        let mut pages = crate::memory::breakers().reserve_request(0,ESQL_LABEL)?;
        'left: for l in &left.rows {
            if limit.is_some_and(|n| rows.len() >= n) { break; }
            let _probe_memory=crate::memory::breakers().reserve_request(row_bytes(l).saturating_mul((lkeys.len() as u64).saturating_mul(16).saturating_add(1)),ESQL_LABEL)?;
            match self.matches(ctx, l) {
                Some(ms) => {
                    for m in ms {
                        #[cfg(test)]
                        JOIN_CANDIDATES.with(|n| n.set(n.get() + 1));
                        let _scratch=join_row_scratch(l,Some(&right.rows[*m]),cols.len())?;
                        let mut row: Vec<Val> = lkeep.iter().map(|i| l[*i].clone()).collect();
                        row.extend(rcols.iter().map(|i| right.rows[*m][*i].clone()));
                        if right_only_null_passes.is_some() || join_row_passes(ctx, &filters, &row)? { pages.grow(row_bytes(&row).saturating_mul(2), ESQL_LABEL)?; rows.push(row); }
                        if limit.is_some_and(|n| rows.len() >= n) { break 'left; }
                    }
                }
                None => {
                    if *right_only_null_passes == Some(false) { continue; }
                    let _scratch=join_row_scratch(l,None,cols.len())?;
                    #[cfg(test)]
                    JOIN_ROW_COPIES.with(|n|n.set(n.get()+1));
                    let mut row: Vec<Val> = lkeep.iter().map(|i| l[*i].clone()).collect();
                    row.extend(rcols.iter().map(|_| Val::Null));
                    if join_row_passes(ctx, &filters, &row)? { pages.grow(row_bytes(&row).saturating_mul(2), ESQL_LABEL)?; rows.push(row); }
                    if limit.is_some_and(|n| rows.len() >= n) { break 'left; }
                }
            }
        }
        retain_join_output(pages,cols)?;
        Ok(Table { cols: cols.clone(), rows })
    }
}

#[allow(clippy::too_many_arguments)]
fn lookup_join(ctx: &mut Ctx, left: Table, index: &str, index_pos: usize, on: &[String], on_pos: &[usize], limit: Option<usize>, filters: &[Expr], pos: usize, fields: Option<&HashSet<String>>) -> Result<Table> {
    // Only the left rows that can be output need their lookup rows.
    let rows = &left.rows[..limit.unwrap_or(usize::MAX).min(left.rows.len())];
    let rows = if filters.is_empty() { rows } else { &left.rows[..] };
    let schema = lookup_schema(ctx, index, index_pos, fields)?;
    // Filters a lookup query answers (`fast::lookup_prefilter`: equalities
    // of lookup columns, which reject the unmatched row): only the rows they
    // keep are read (Rally joins `limit10_expanding10000_filter1`: 1 of
    // 10,000 rows of one key). A key whose rows all fail joins nothing either
    // way: its rejected matches, or its unmatched null row the filters reject.
    let prefilter = if filters.is_empty() { None } else { fast::lookup_prefilter(ctx, &schema, on, filters)? };
    let plan = match &prefilter {
        Some(extra) => LookupJoinPlan::for_rows_with(ctx, &left, rows, &schema, index_pos, on, on_pos, filters, pos, extra)?,
        None => LookupJoinPlan::for_rows(ctx, &left, rows, &schema, index_pos, on, on_pos, filters, pos)?,
    };
    plan.apply(ctx, &left, limit)
}

/// `ENRICH`: join each row with the policy's enrich index on the match
/// column; enrich columns are appended (multi-valued over several matches).
fn enrich(ctx: &mut Ctx, mut t: Table, policy: &str, on: Option<&str>, with: &[(String, String)], pos: usize) -> Result<Table> {
    let node = ctx.node.ok_or_else(|| verification_err(ctx.full, pos, format!("cannot find enrich policy [{policy}]")))?;
    let Some(p) = crate::enrich::store(node).get(policy).filter(|_| crate::enrich::has_enrich_index(node, policy)) else {
        return Err(verification_err(ctx.full, pos, format!("cannot find enrich policy [{policy}]")));
    };
    let match_col = on.map(str::to_string).unwrap_or_else(|| p.match_field.clone());
    let mi = t.idx(&match_col).ok_or_else(|| unknown_col(ctx, &match_col, pos, &t.cols))?;
    let fields: Vec<(String, String)> = if with.is_empty() { p.enrich_fields.iter().map(|f| (f.clone(), f.clone())).collect() } else { with.to_vec() };
    for (_, src) in &fields {
        if !p.enrich_fields.contains(src) {
            return Err(verification_err(ctx.full, pos, format!("Enrich field [{src}] not found in enrich policy [{policy}]")));
        }
    }
    // Range policies only take inputs of the range's type (or strings, parsed per value).
    if p.kind == "range" {
        let range_ty = {
            let mut opts = ResolveOpts::from_params(&HashMap::new());
            opts.allow_no_indices = true;
            node.resolve(&p.indices.join(","), opts).ok().and_then(|ts| ts.iter().find_map(|t| t.svc.lookup().get(&p.match_field).map(|f| f.declared.name().to_string())))
        };
        let input_ty = t.cols[mi].ty.clone();
        let range = match range_ty.as_deref() {
            Some("integer_range") => Some(("INTEGER", &["integer", "long"][..])),
            Some("long_range") => Some(("LONG", &["integer", "long"][..])),
            Some("double_range") => Some(("DOUBLE", &["double", "integer", "long"][..])),
            Some("float_range") => Some(("DOUBLE", &["double", "integer", "long"][..])),
            Some("date_range") => Some(("DATETIME", &["date", "date_nanos"][..])),
            Some("ip_range") => Some(("IP", &["ip"][..])),
            _ => None,
        };
        if let Some((rt, ok)) = range {
            if !ok.contains(&input_ty.as_str()) && !matches!(input_ty.as_str(), "keyword" | "text" | "null") {
                let it = if input_ty == "date" { "DATETIME".to_string() } else { input_ty.to_ascii_uppercase() };
                return Err(verification_err(ctx.full, pos, format!("ENRICH range and input types are incompatible. range[{rt}], input[{it}]")));
            }
        }
    }
    let schema = resolve_schema(ctx, &p.indices, pos).ok();
    let col_src = |name: &str| -> ColSrc {
        schema
            .as_ref()
            .and_then(|s| s.cols.iter().find(|(n, _)| n == name).map(|(_, c)| c.copy()))
            .unwrap_or_else(|| ColSrc::plain(name, "keyword"))
    };
    let srcs: Vec<ColSrc> = fields
        .iter()
        .map(|(_, f)| {
            let mut c = col_src(f);
            // The enrich index maps a geo_match field as a shape.
            if p.kind == "geo_match" && *f == p.match_field {
                c = ColSrc::plain(&c.path, "geo_shape");
                c.doc_values = false;
            }
            c
        })
        .collect();
    // The enrich columns are charged as they are built: their slots before
    // they are allocated, each value's heap as it is made.
    let pages = crate::memory::RequestMemory::new();
    pages.charge((fields.len() * t.rows.len() * std::mem::size_of::<Val>()) as u64, ESQL_LABEL)?;
    let mut out: Vec<Vec<Val>> = vec![Vec::with_capacity(t.rows.len()); fields.len()];
    if p.kind == "match" {
        let docs = crate::enrich::lookup(node, policy, serde_json::json!({"match_all": {}}), 10_000)?;
        let mcs = col_src(&p.match_field);
        let mut by_key: HashMap<String, Vec<usize>> = HashMap::new();
        // Inputs match like a term query on the match field: by their string form.
        for (di, d) in docs.iter().enumerate() {
            for v in extract(d, &mcs, "").values() {
                by_key.entry(v.to_text()).or_default().push(di);
            }
        }
        for r in &t.rows {
            let mut hit: Vec<usize> = Vec::new();
            for v in r[mi].values() {
                if let Some(ds) = by_key.get(&v.to_text()) {
                    hit.extend(ds.iter().filter(|d| !hit.contains(d)).copied().collect::<Vec<_>>());
                }
            }
            for (k, c) in srcs.iter().enumerate() {
                let v = Val::from_list(hit.iter().flat_map(|d| extract(&docs[*d], c, "").values().into_iter().cloned().collect::<Vec<_>>()).collect());
                pages.add(v.heap_bytes() as u64, ESQL_LABEL)?;
                out[k].push(v);
            }
        }
    } else {
        for r in &t.rows {
            let v = &r[mi];
            let docs = if v.is_null() {
                std::sync::Arc::new(Vec::new())
            } else {
                let jv = match v {
                    Val::Multi(m) => Value::Array(m.iter().map(Val::to_json).collect()),
                    Val::Date(d) => json!(d),
                    other => other.to_json(),
                };
                // Values that don't parse as the match field's type match nothing (ES warns).
                crate::enrich::lookup_query(&p.kind, &p.match_field, &jv, "intersects")
                    .and_then(|q| crate::enrich::lookup(node, policy, q, 128))
                    .unwrap_or_else(|_| std::sync::Arc::new(Vec::new()))
            };
            for (k, c) in srcs.iter().enumerate() {
                let v = Val::from_list(docs.iter().flat_map(|d| extract(d, c, "").values().into_iter().cloned().collect::<Vec<_>>()).collect());
                pages.add(v.heap_bytes() as u64, ESQL_LABEL)?;
                out[k].push(v);
            }
        }
    }
    for ((name, _), (vals, c)) in fields.iter().zip(out.into_iter().zip(srcs.iter())) {
        t.put_col(name, c.ty.clone(), vals);
    }
    Ok(t)
}

/// Run commands over an existing table (no node access; used by tests/SQL).
pub fn run_on_table(full: &str, cmds: &[Cmd], t: Table) -> Result<Table> {
    let mut ctx = Ctx { node: None, full, filter: None, fctx: FnCtx { now: crate::date::now_millis(), ..Default::default() }, documents_found: 0, values_loaded: 0, default_limit: 1000, max_limit: 10_000, unsupported: HashMap::new(), pristine: HashSet::new(), fulltext_barrier: None, lexical_barrier: None, settings: Default::default(), rows_emitted: 0, esql: false, ts: false, ts_done: false, ts_dims: Vec::new(), col_meta: Default::default(), view_stack: Vec::new(), used_views: false, extra_load: Vec::new(), restrict: None, shards: None, mem: Default::default(), table_bytes: Default::default(), drop_null_columns: false, all_columns: None };
    exec_pipeline(&mut ctx, cmds, Some(t))
}

/// New context for embedding (SQL).
pub fn new_ctx<'a>(node: &'a Node, full: &'a str, filter: Option<&'a Value>, default_limit: usize, max_limit: usize) -> Ctx<'a> {
    Ctx { node: Some(node), full, filter, fctx: FnCtx { now: crate::date::now_millis(), ..Default::default() }, documents_found: 0, values_loaded: 0, default_limit, max_limit, unsupported: HashMap::new(), pristine: HashSet::new(), fulltext_barrier: None, lexical_barrier: None, settings: Default::default(), rows_emitted: 0, esql: false, ts: false, ts_done: false, ts_dims: Vec::new(), col_meta: Default::default(), view_stack: Vec::new(), used_views: false, extra_load: Vec::new(), restrict: None, shards: None, mem: Default::default(), table_bytes: Default::default(), drop_null_columns: false, all_columns: None }
}

/// An unsorted SQL scan pins segment snapshots and retains search cursors plus
/// at most one candidate batch of filtered rows. Projections and WHERE expressions
/// run incrementally, including eligible computed projections; sorting and remote-only shard slots
/// retain the existing executor.
pub struct SqlScan {
    schema: IndexSchema,
    request: crate::search::SearchRequest,
    cursors: Option<Vec<crate::search::Cursor>>,
    remaining: usize,
    pub columns: Vec<Value>,
    output_remaining: usize,
    commands: Vec<Cmd>,
    query_text: String,
    wanted: Vec<bool>,
    now: i64,
    pending: std::collections::VecDeque<Value>,
    /// One local shard: pages read its pinned snapshot directly.
    direct: Option<DirectScan>,
}

/// The direct read of a [`SqlScan`] over one local shard: candidate docs come
/// straight from the snapshot (`_doc` order, the order of the `_doc`-sorted
/// search it replaces) and their columns from doc values / stored source in
/// batches, without a search response per page.
struct DirectScan {
    query: Value,
    /// The columns pages read (those the plan references; all for `*`):
    /// pages carry no null cell per unrelated mapped field.
    narrow: IndexSchema,
    /// Where the next page resumes: (target, leaf, doc).
    pos: (usize, usize, rustic_core::types::DocId),
    /// Filter-only float columns, read from doc values alone (as ES SQL
    /// evaluates script filters, and as the materializing pass reads them).
    dv_only: Vec<bool>,
}

/// Candidate docs a direct SQL scan reads per batch.
const SQL_SCAN_BATCH: usize = 1024;

#[cfg(test)]
thread_local! {
    /// Tests: open SQL scans on the `_doc`-sorted search (the reference).
    pub(crate) static SQL_SCAN_DIRECT_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn sql_scan_direct_enabled() -> bool {
    #[cfg(test)]
    if SQL_SCAN_DIRECT_DISABLED.with(|d| d.get()) { return false; }
    true
}

impl SqlScan {
    pub(crate) fn open(ctx: &Ctx, index: &str, limit: Option<usize>, commands: &[Cmd]) -> Result<Option<Self>> {
        let node = ctx.node.ok_or_else(|| EsError::internal("no node"))?;
        let mut schema = resolve_schema(ctx, &[index.to_string()], 0)?;
        if schema.slots.is_some() { return Ok(None); }
        schema.targets = node.snapshot(&schema.targets)?;
        let names: HashSet<String> = schema.cols.iter().map(|(n, _)| n.clone()).collect();
        schema.cols.retain(|(name, source)| source.ty != "unsupported"
            && !name.rsplit_once('.').is_some_and(|(parent, _)| names.contains(parent)));
        let mut conditions = ctx.filter.iter().map(|q| (*q).clone()).collect::<Vec<_>>();
        let types = schema.cols.iter().map(|(n,c)| (n.clone(), c.ty.clone())).collect();
        let mut local = Vec::new();
        for cmd in commands {
            match cmd {
                Cmd::From { .. } | Cmd::Limit(_) => {},
                Cmd::Where(expr, pos) => {
                    if has_fulltext(expr) { return Ok(None); }
                    bind(ctx, expr, &source_cols(&schema, &[]))?;
                    let mut conjuncts_ = Vec::new();
                    conjuncts(expr, &mut conjuncts_);
                    for e in conjuncts_ {
                        if let Some((q, _)) = to_dsl(&e, &types, true) { conditions.push(q); }
                        else { local.push(Cmd::Where(e, *pos)); }
                    }
                },
                other => local.push(other.clone()),
            }
        }
        // A clause-less bool is match_all (ES rewrites it so): the count is
        // then the snapshot's live docs, not a visit of every doc.
        let query = if conditions.is_empty() { json!({"match_all":{}}) } else { json!({"bool":{"filter":conditions}}) };
        let needed = needed_columns(ctx.full, &schema);
        let wanted: Vec<bool> = schema.cols.iter().map(|(name, _)| needed.as_ref().is_none_or(|names| names.contains(name))).collect();
        // One local shard: read pages from its snapshot directly (the same
        // `_doc` order as the search below). Several shards keep the
        // `_doc`-sorted search, whose order interleaves them.
        let direct = (sql_scan_direct_enabled() && schema.targets.len() == 1 && schema.targets[0].snapshot.is_some()).then(|| {
            // Columns that can reach the output keep the projected-value read.
            let mut output: Vec<String> = Vec::new();
            for cmd in &local {
                match cmd {
                    Cmd::Keep(names, _) => output.extend(names.iter().cloned()),
                    Cmd::Eval(fields) => for field in fields {
                        let mut refs = Vec::new();
                        super::analyze::expr_refs(&field.expr, &mut refs);
                        if let Some(filter) = &field.filter { super::analyze::expr_refs(filter, &mut refs); }
                        output.extend(refs.into_iter().map(|(name, _)| name));
                    },
                    _ => {}
                }
            }
            let narrow = IndexSchema { cols: schema.cols.iter().zip(&wanted).filter(|(_, w)| **w).map(|(c, _)| c.clone()).collect(), targets: schema.targets.clone(), slots: None, projection: None };
            let dv_only = narrow.cols.iter().map(|(name, column)|
                needed.is_some() && column.ty == "double" && !output.contains(name)).collect();
            DirectScan { query: query.clone(), narrow, pos: (0, 0, 0), dv_only }
        });
        // Count without collecting hits (the top-hits collector visited every
        // match only to count it: ~0.3 s over 33M docs before the first row).
        // A direct scan with a residual filter does not count at all: like
        // ES, a full page keeps a cursor, and the scan learns it is done when
        // its candidates run out.
        let filtered = local.iter().any(|c| matches!(c, Cmd::Where(..)));
        let remaining = if direct.is_some() && filtered { usize::MAX } else { crate::search::par_count(&schema.targets, &query)? as usize };
        let body = search_body(&schema, &[], &needed, &query, vec![json!("_doc")]);
        let request = crate::search::parse_request(&body, &HashMap::new())?;
        let mut empty = Table { cols: source_cols(&schema, &[]), rows: Vec::new() };
        let mut validation = new_ctx(node, ctx.full, None, 1_000_000, 1_000_000);
        let mut floats: HashSet<String> = schema.cols.iter().filter(|(_, source)|
            source.ty == "double" && source.declared == "float" && source.per_index.values().all(|src| src.declared == "float"))
            .map(|(name, _)| name.clone()).collect();
        for cmd in &local {
            if let Cmd::Eval(fields) = cmd {
                for field in fields {
                    let is_float = matches!(&field.expr, Expr::Col(name, _) if floats.contains(name));
                    floats.remove(&field.name);
                    if is_float { floats.insert(field.name.clone()); }
                }
            }
            empty = apply(&mut validation, cmd, empty)?;
        }
        let columns = empty.cols.iter().map(|col| {
            let float = col.ty == "double" && floats.contains(&col.name);
            json!({"name":col.name,"type":if float {"float"} else {crate::sql::sql_type(&col.ty)}})
        }).collect();
        Ok(Some(Self { schema, request, cursors: None, remaining, columns,
            output_remaining: limit.unwrap_or(usize::MAX), commands: local, query_text: ctx.full.to_string(), wanted, now: ctx.fctx.now, pending: Default::default(), direct }))
    }

    /// Upper bound on remaining output; residual filters may exhaust earlier.
    pub fn len(&self) -> usize { self.remaining.saturating_add(self.pending.len()).min(self.output_remaining) }

    /// Whether pages read one local shard's snapshot directly (no search
    /// response per page): residual filters then cost only the candidates
    /// read until a page is full.
    pub fn is_direct(&self) -> bool { self.direct.is_some() }

    /// [`SqlScan::page`] over one local shard: read candidate batches from
    /// the saved position until the page is full, filtering each batch as it
    /// is read; surplus rows of the last batch wait in `pending`.
    fn page_direct(&mut self, node: &Node, fetch: usize) -> Result<Vec<Value>> {
        let want = fetch.min(self.output_remaining);
        let mut rows: Vec<Value> = Vec::new();
        while rows.len() < want {
            let Some(row) = self.pending.pop_front() else { break };
            rows.push(row);
        }
        if rows.len() < want && self.remaining > 0 {
            let mut ctx = new_ctx(node, &self.query_text, None, 1_000_000, 1_000_000);
            ctx.fctx.now = self.now;
            let filtered = self.commands.iter().any(|c| matches!(c, Cmd::Where(..)));
            // Unfiltered pages read exactly the rows they return.
            let batch = if filtered { SQL_SCAN_BATCH } else { (want - rows.len()).min(SQL_SCAN_BATCH) };
            let commands = &self.commands;
            let direct = self.direct.as_mut().ok_or_else(|| EsError::internal("SQL scan is not direct"))?;
            let (pending, remaining) = (&mut self.pending, &mut self.remaining);
            let mut read_any = false;
            let schema = &direct.narrow;
            let wanted = vec![true; schema.cols.len()];
            let cols = source_cols(schema, &[]);
            // The leading row filters, each evaluated once per distinct
            // value of its one input column (a station name, a temperature).
            let lead = commands.iter().take_while(|c| matches!(c, Cmd::Where(..))).count();
            let mut filters = commands[..lead].iter().map(|c| match c {
                Cmd::Where(e, _) => sql_pivot_stream::memo(&ctx, e, &cols),
                _ => unreachable!(),
            }).collect::<Result<Vec<_>>>()?;
            let dv_only = &direct.dv_only;
            let pos = &mut direct.pos;
            crate::search::for_each_match_from(&schema.targets, &direct.query, batch, *pos, &mut |page| {
                let docs = page.docs;
                let Some(&last) = docs.last() else { return Ok(true) };
                read_any = true;
                let (batch_rows, _loaded, _page_memory) = sql_pivot_stream::dv_column_rows(schema, &wanted, dv_only, page.target, page.lookup, page.leaf, docs)?;
                *pos = (page.target_idx, page.leaf_idx, last + 1);
                *remaining = remaining.saturating_sub(docs.len());
                let mut kept = Vec::with_capacity(batch_rows.len());
                for row in batch_rows {
                    if sql_pivot_stream::accepts(&mut filters, &row, &ctx.fctx)? { kept.push(row); }
                }
                let mut table = Table { cols: cols.clone(), rows: kept };
                for cmd in &commands[lead..] { table = apply(&mut ctx, cmd, table)?; }
                for row in table.rows {
                    let row = Value::Array(row.into_iter().map(|v| v.to_json()).collect());
                    if rows.len() < want { rows.push(row); } else { pending.push_back(row); }
                }
                Ok(rows.len() < want && *remaining > 0)
            })?;
            // The snapshot's count bounds its matches: nothing left to read.
            if !read_any || rows.len() < want { self.remaining = 0; }
        }
        self.output_remaining = self.output_remaining.saturating_sub(rows.len());
        Ok(rows)
    }

    pub fn page(&mut self, node: &Node, fetch: usize) -> Result<Vec<Value>> {
        if self.direct.is_some() { return self.page_direct(node, fetch); }
        let want = fetch.min(self.output_remaining);
        let mut rows = Vec::new();
        let mut ctx = new_ctx(node, &self.query_text, None, 1_000_000, 1_000_000);
        ctx.fctx.now = self.now;
        while rows.len() < want {
            while rows.len() < want {
                let Some(row) = self.pending.pop_front() else { break };
                rows.push(row);
            }
            if rows.len() == want || self.remaining == 0 { break; }
            // A selective filter must not degenerate into one search per
            // candidate when just one output row remains to fill the page.
            let batch = if self.commands.iter().any(|c| matches!(c, Cmd::Where(..))) { 1024 } else { want - rows.len() };
            self.request.size = batch.min(self.remaining).min(1024);
            let out = crate::search::execute(&self.schema.targets, &self.request, &crate::search::ExecOpts {
                terms_lookup: None, cursors: self.cursors.as_deref(), max_buckets: node.max_buckets(),
                scroll_id: Some(String::new()), pit_id: None,
            })?;
            let mut result: Value = serde_json::from_str(&out.json).map_err(|e| EsError::internal(format!("bad SQL scan output: {e}")))?;
            if result["_shards"]["failed"].as_u64().unwrap_or(0) != 0 || result["timed_out"] == true {
                return Err(EsError::internal("SQL scan search did not complete successfully"));
            }
            let hits = match result["hits"]["hits"].take() { Value::Array(hits) => hits, _ => Vec::new() };
            if hits.is_empty() { return Err(EsError::internal("SQL scan exhausted before its snapshot count")); }
            self.remaining = self.remaining.saturating_sub(hits.len());
            self.cursors = Some(out.cursors);
            let mut table = Table { cols: source_cols(&self.schema, &[]), rows: hits.into_iter().map(|hit| {
                self.schema.cols.iter().zip(&self.wanted).map(|((_, source), wanted)|
                    if *wanted { extract(&hit["_source"], source, hit["_index"].as_str().unwrap_or("")) } else { Val::Null }).collect()
            }).collect() };
            for cmd in &self.commands { table = apply(&mut ctx, cmd, table)?; }
            self.pending.extend(table.rows.into_iter().map(|r| Value::Array(r.into_iter().map(|v| v.to_json()).collect())));
        }
        self.output_remaining = self.output_remaining.saturating_sub(rows.len());
        Ok(rows)
    }
}

/// SQL preserves a mapped float's metadata even though execution widens it to
/// double. Only homogeneous float mappings qualify; computed expressions and
/// mixed-index types must retain the executor's inferred type.
pub(crate) fn sql_float_columns(ctx: &Ctx, index: &str) -> Result<HashSet<String>> {
    let schema = resolve_schema(ctx, &[index.to_string()], 0)?;
    Ok(schema.cols.iter().filter(|(_, c)| c.ty == "double" && c.declared == "float"
        && c.per_index.values().all(|src| src.declared == "float"))
        .map(|(name, _)| name.clone()).collect())
}

/// Columns (name, ES|QL type) of the indices matching `indices`.
pub fn index_columns(node: &Node, indices: &[String]) -> Result<Vec<Column>> {
    let ctx = new_ctx(node, "", None, 1000, 10_000);
    let schema = resolve_schema(&ctx, indices, 0)?;
    Ok(schema.cols.iter().map(|(n, c)| Column { name: n.clone(), ty: c.col_type() }).collect())
}

/// A node-less ES|QL context for `query` (tests: warnings stay readable).
#[cfg(test)]
pub fn new_ctx_local(query: &str) -> Ctx<'_> {
    Ctx { node: None, full: query, filter: None, fctx: FnCtx { query: query.to_string(), now: crate::date::now_millis(), ..Default::default() }, documents_found: 0, values_loaded: 0, default_limit: 1000, max_limit: 10_000, unsupported: HashMap::new(), pristine: HashSet::new(), fulltext_barrier: None, lexical_barrier: None, settings: Default::default(), rows_emitted: 0, esql: true, ts: false, ts_done: false, ts_dims: Vec::new(), col_meta: Default::default(), view_stack: Vec::new(), used_views: false, extra_load: Vec::new(), restrict: None, shards: None, mem: Default::default(), table_bytes: Default::default(), drop_null_columns: false, all_columns: None }
}

/// Run a node-less query (`ROW ...`, `SHOW INFO`).
pub fn run_local(query: &str, params: &Params) -> Result<Table> {
    let cmds = parse_query(query, params)?;
    let mut ctx = Ctx { node: None, full: query, filter: None, fctx: FnCtx { now: crate::date::now_millis(), ..Default::default() }, documents_found: 0, values_loaded: 0, default_limit: 1000, max_limit: 10_000, unsupported: HashMap::new(), pristine: HashSet::new(), fulltext_barrier: None, lexical_barrier: None, settings: Default::default(), rows_emitted: 0, esql: true, ts: false, ts_done: false, ts_dims: Vec::new(), col_meta: Default::default(), view_stack: Vec::new(), used_views: false, extra_load: Vec::new(), restrict: None, shards: None, mem: Default::default(), table_bytes: Default::default(), drop_null_columns: false, all_columns: None };
    run_cmds(&mut ctx, &cmds)
}

#[cfg(test)]
mod join_memory_tests {
    use super::*;

    fn breakers()->&'static crate::memory::Breakers {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });breakers
    }

    #[test]
    fn lookup_key_index_is_admitted_before_construction() {
        let breakers=breakers();
        crate::memory::scoped(Some(crate::memory::RequestScope::new("lookup keys",breakers)),|| {
            let right=Table{cols:vec![Column{name:"key".into(),ty:"keyword".into()}],rows:vec![vec![Val::Str("x".repeat(4096))];8]};
            assert_eq!(lookup_plan_charge(&right,4).err().unwrap().status,429);
            assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn unmatched_lookup_admits_wide_null_row_before_cloning() {
        let tmp=tempfile::tempdir().unwrap();let node=Node::open(tmp.path()).unwrap();
        let ctx=new_ctx(&node,"FROM left | LOOKUP JOIN right ON key",None,100,100);
        let left=Table{cols:vec![Column{name:"key".into(),ty:"long".into()}],rows:vec![vec![Val::Long(1)]]};
        let right_cols:Vec<_>=(0..10000).map(|i|Column{name:format!("r{i}"),ty:"keyword".into()}).collect();
        let mut cols=left.cols.clone();cols.extend(right_cols.iter().cloned());
        let breakers=breakers();
        crate::memory::scoped(Some(crate::memory::RequestScope::new("wide unmatched lookup",breakers)),|| {
            let plan=LookupJoinPlan{right:Table{cols:right_cols,rows:Vec::new()},lkeys:vec![0],lkeep:vec![0],rcols:(0..10000).collect(),cols,index_map:HashMap::new(),mv_keys:HashSet::new(),pos:0,filters:Vec::new(),right_only_null_passes:None,_charge:breakers.reserve_request(0,ESQL_LABEL).unwrap()};
            JOIN_ROW_COPIES.with(|n|n.set(0));
            assert_eq!(plan.apply(&ctx,&left,None).err().unwrap().status,429);
            assert_eq!(JOIN_ROW_COPIES.with(|n|n.get()),0);
            assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn expression_numeric_expansion_is_admitted_before_function_call() {
        let breakers=breakers();
        crate::memory::scoped(Some(crate::memory::RequestScope::new("expression expansion",breakers)),|| {
            let ctx=new_ctx_local("ROW x=SPACE(10000000)");
            let space=Expr::Call("SPACE".into(),vec![Expr::Lit(Val::Long(10000000))],0);
            assert_eq!(eval(&space,&[],&ctx.fctx).unwrap_err().status,429);
            let repeat=Expr::Call("REPEAT".into(),vec![Expr::Lit(Val::Str("x".into())),Expr::Lit(Val::Long(10000000))],0);
            assert_eq!(eval(&repeat,&[],&ctx.fctx).unwrap_err().status,429);
            let coerced=Expr::Call("REPEAT".into(),vec![Expr::Lit(Val::Long(1)),Expr::Lit(Val::Long(10000000))],0);
            assert_eq!(eval(&coerced,&[],&ctx.fctx).unwrap_err().status,429);
            let replace=Expr::Call("REPLACE".into(),vec![Expr::Lit(Val::Str("x".repeat(1000))),Expr::Lit(Val::Str("".into())),Expr::Lit(Val::Long(i64::MAX))],0);
            assert_eq!(eval(&replace,&[],&ctx.fctx).unwrap_err().status,429);
            assert_eq!(breakers.request.reserved(),0);
            let small=Expr::Call("LENGTH".into(),vec![Expr::Lit(Val::Str("small".into()))],0);
            for _ in 0..1000 { assert!(eval(&small,&[],&ctx.fctx).is_ok());assert_eq!(breakers.request.reserved(),0); }
        });
    }

    #[test]
    fn regex_expansion_breaker_is_not_an_invalid_pattern_result() {
        let breakers=breakers();
        crate::memory::scoped(Some(crate::memory::RequestScope::new("regex expansion",breakers)),|| {
            let ctx=new_ctx_local("ROW x=1");
            let rlike=Expr::Like(Box::new(Expr::Lit(Val::Str("a".into()))),vec!["a{100000}".into()],false,true);
            assert_eq!(eval(&rlike,&[],&ctx.fctx).unwrap_err().status,429);
            let replace=Expr::Call("REPLACE".into(),vec![Expr::Lit(Val::Str("a".into())),Expr::Lit(Val::Str("a{100000}".into())),Expr::Lit(Val::Str("b".into()))],0);
            assert_eq!(eval(&replace,&[],&ctx.fctx).unwrap_err().status,429);
            drop(ctx);assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn grok_expansion_is_admitted_before_regex_or_capture_materialization() {
        let breakers=breakers();
        crate::memory::scoped(Some(crate::memory::RequestScope::new("grok expansion",breakers)),|| {
            let ctx=new_ctx_local("ROW x=1");
            let mut table=Table{cols:vec![],rows:vec![vec![]]};
            assert_eq!(grok(&ctx,&mut table,&Expr::Lit(Val::Str("a".into())),"a{100000}",0).unwrap_err().status,429);
            assert!(table.cols.is_empty());assert_eq!(breakers.request.reserved(),0);
            let mut memory=crate::memory::breakers().reserve_request(0,ESQL_LABEL).unwrap();
            let pattern="%{WORD:x}".repeat(4096);
            assert_eq!(grok_compile(&ctx,&pattern,0,&mut Vec::new(),0,&mut memory).unwrap_err().status,429);
            drop(memory);assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn column_globs_preserve_unicode_and_escaping_without_regex_cache() {
        assert!(column_glob("a💡z","a?*"));
        assert!(column_glob("a*z","a\\**"));
        assert!(!column_glob("ab","a?z*"));
        assert!(column_glob("abc","a*c"));
        assert!(column_glob("abc","abc\\"));
    }

    #[test]
    fn table_handoff_counts_unsampled_wide_row() {
        let mut rows=vec![vec![Val::Long(1)];128];
        rows[127]=vec![Val::Str("x".repeat(100000))];
        let table=Table{cols:vec![],rows};
        assert!(table_bytes(&table)>=100000);
    }

    #[test]
    fn rejected_lookup_candidates_release_scratch_between_rows() {
        let breakers=breakers();
        crate::memory::scoped(Some(crate::memory::RequestScope::new("lookup scratch",breakers)),|| {
            let left=vec![Val::Str("left".into())];let right=vec![Val::Str("right".into())];
            for _ in 0..1024 {
                let memory=join_row_scratch(&left,Some(&right),2).unwrap();
                let combined:Vec<_>=left.iter().chain(&right).cloned().collect();
                assert!(breakers.request.reserved()>0);
                drop(combined);drop(memory);assert_eq!(breakers.request.reserved(),0);
            }
            let wide=vec![Val::Str("x".repeat(64*1024))];
            assert_eq!(join_row_scratch(&left,Some(&wide),2).err().unwrap().status,429);
            assert_eq!(breakers.request.reserved(),0);
        });
    }
}
