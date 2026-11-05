//! Synthetic `_source` (`index.mapping.source.mode: synthetic`).
//!
//! ES stores no `_source` for such an index and rebuilds it at fetch time
//! from doc values, stored fields and `_ignored_source`. The result is a
//! *canonical* document: keys sorted, dotted names expanded, each mapped leaf
//! rendered from what the field type indexed (keyword/ip/version values
//! sorted and de-duplicated, numerics sorted, floats rounded to their
//! precision, dates printed with the field's format, ...), object arrays
//! merged into one object of arrays, and anything the mapper could not
//! represent (`ignore_above` / `ignore_malformed` values, unmapped fields of
//! `dynamic: false` objects, disabled objects, `synthetic_source_keep`
//! fields, field types without synthetic support) kept verbatim.
//!
//! This module computes that canonical form at index time, from the incoming
//! `_source` and the mapping the document was parsed with, so every reader of
//! the stored `_source` (get, mget, search, update, reindex, scripts,
//! highlighting) sees exactly what ES would reconstruct, without touching the
//! fetch paths. The per-field rendering mirrors the doc-values layout ES
//! reads back (see the comments on each type).

use crate::mapping::{EsType, FieldLookup, FieldMapping, Mapper, Mappings, ObjectMapping, ResolvedField};
use rustic_core::fieldtypes::{flattened, numeric, range as rng, version};
use serde_json::{json, Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};

/// `synthetic_source_keep` (per field / object, or the index-level default).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Keep {
    None,
    Arrays,
    All,
}

impl Keep {
    pub fn parse(s: &str) -> Option<Keep> {
        match s {
            "none" => Some(Keep::None),
            "arrays" => Some(Keep::Arrays),
            "all" => Some(Keep::All),
            _ => None,
        }
    }

    /// `index.mapping.synthetic_source_keep` of an index (default `arrays`
    /// in `logsdb` mode, else `none`; `Mapper.SYNTHETIC_SOURCE_KEEP_INDEX_SETTING`).
    pub fn of_index(settings: &crate::settings::FlatSettings) -> Keep {
        match crate::settings::get_str(settings, "index.mapping.synthetic_source_keep") {
            Some(k) => Keep::parse(k).unwrap_or(Keep::None),
            // `columnar_stored`: arrays come back as ingested.
            // `columnar_stored`, and columnar modes (doc values keep array
            // order: `mapper.columnar.maintain_array_order`): arrays come
            // back as ingested.
            None if crate::settings::get_str(settings, "index.mapping.source.mode").is_some_and(|m| m.eq_ignore_ascii_case("columnar_stored"))
                || crate::settings::get_str(settings, "index.mode").is_some_and(|m| m.eq_ignore_ascii_case("columnar") || m.eq_ignore_ascii_case("logsdb_columnar")) =>
            {
                Keep::Arrays
            }
            None if crate::settings::get_str(settings, "index.mode").map(|m| m.eq_ignore_ascii_case("logsdb")).unwrap_or(false) => Keep::Arrays,
            None => Keep::None,
        }
    }
}

/// The `_source` mode of an index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SourceMode {
    Stored,
    Synthetic,
    Disabled,
}

impl std::fmt::Display for SourceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SourceMode::Stored => "stored",
            SourceMode::Synthetic => "synthetic",
            SourceMode::Disabled => "disabled",
        })
    }
}

/// Effective source mode: `index.mapping.source.mode`, else the mapping's
/// `_source.mode` / `_source.enabled`, else the index mode's default
/// (`logsdb` and `time_series` indices are synthetic).
pub fn source_mode(setting: Option<&str>, m: &Mappings, index_mode: Option<&str>) -> SourceMode {
    let logsdb = matches!(index_mode.map(str::to_ascii_lowercase).as_deref(), Some("logsdb" | "time_series" | "columnar" | "logsdb_columnar"));
    let mapped = m.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str);
    match setting.or(mapped).map(str::to_ascii_lowercase).as_deref() {
        // `columnar_stored`: served like synthetic source (flattened,
        // mapped fields only).
        Some("synthetic" | "columnar_stored") => SourceMode::Synthetic,
        Some("disabled") => SourceMode::Disabled,
        Some("stored") => {
            if m.source_enabled() {
                SourceMode::Stored
            } else {
                SourceMode::Disabled
            }
        }
        _ => {
            if !m.source_enabled() {
                SourceMode::Disabled
            } else if logsdb {
                SourceMode::Synthetic
            } else {
                SourceMode::Stored
            }
        }
    }
}

/// `force_synthetic_source`: the synthetic form of a stored `_source` of
/// `svc` (`None` when it cannot be parsed or rendered).
pub fn force(svc: &crate::index::IndexService, raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;
    let meta = svc.meta();
    let lk = svc.lookup();
    let keep = Keep::of_index(&meta.settings);
    synthesize(&v, &meta.mappings, &lk, keep).ok().map(|(v, _)| v.to_string())
}

/// `LogsdbIndexModeSettingsProvider`: a new index that would use synthetic
/// source (`index.mapping.source.mode: synthetic`, a `logsdb` /
/// `time_series` index without that setting, or a `_source.mode: synthetic`
/// mapping) gets `index.mapping.source.mode: STORED` (`COLUMNAR_STORED` in a
/// strict columnar mode) when the license does not include synthetic source:
/// enterprise / trial do; gold / platinum only for time series indices and
/// `profiling-*` / `.profiling-*` / `metrics-apm.*` data, with a license
/// started by [`crate::license::SYNTHETIC_SOURCE_LEGACY_CUTOFF`]; basic and
/// standard never (`LogsdbLicenseService.fallbackToStoredSource`). Applies
/// only on a node that started its license service.
///
/// Never triggers: licensing is hardwired to enterprise/trial (see
/// [`crate::license::current`] and AGENTS.md "Licensing: always
/// enterprise").
pub fn license_fallback(node: &crate::node::Node, index: &str, settings: &mut crate::settings::FlatSettings, mappings: &Value) {
    use crate::license::Mode;
    if !crate::license::started(node) {
        return;
    }
    let mode = crate::settings::get_str(settings, "index.mode").map(str::to_ascii_lowercase);
    let explicit = crate::settings::get_str(settings, "index.mapping.source.mode").map(str::to_ascii_lowercase);
    let usage = match explicit.as_deref() {
        Some(m) => m == "synthetic",
        None => matches!(mode.as_deref(), Some("logsdb" | "time_series")),
    } || mappings.get("_source").and_then(|s| s.get("mode")).and_then(Value::as_str).is_some_and(|m| m.eq_ignore_ascii_case("synthetic"));
    if !usage {
        return;
    }
    let license = crate::license::current(node);
    let effective = license.effective_mode(crate::date::now_millis());
    if effective.allows(Mode::Enterprise) {
        return;
    }
    // The backing index of data stream `ds` is `.ds-<ds>-<date>-<generation>`.
    let ds = index.strip_prefix(".ds-").unwrap_or("");
    let legacy = mode.as_deref() == Some("time_series") || ds.starts_with("profiling-") || index.starts_with(".profiling-") || ds.starts_with("metrics-apm.");
    if legacy && license.start_date_in_millis <= crate::license::SYNTHETIC_SOURCE_LEGACY_CUTOFF && effective.allows(Mode::Gold) {
        return;
    }
    let columnar = matches!(mode.as_deref(), Some("columnar" | "logsdb_columnar"));
    // ES logs this at debug; a warning here makes a benchmark that meant to
    // run synthetic source (a trial started too late) visible in the log.
    tracing::warn!(index, license = effective.name(), "creating index with stored source: synthetic source needs an enterprise or trial license");
    settings.insert("index.mapping.source.mode".into(), Value::String(if columnar { "COLUMNAR_STORED" } else { "STORED" }.into()));
}

/// Create-time validation of the `_source` configuration.
pub fn validate_create(settings: &crate::settings::FlatSettings, m: &Mappings) -> crate::error::Result<()> {
    use crate::error::EsError;
    let setting = crate::settings::get_str(settings, "index.mapping.source.mode");
    if let Some(v) = setting {
        if !matches!(v.to_ascii_lowercase().as_str(), "stored" | "synthetic" | "disabled" | "columnar_stored") {
            return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.index.mapper.SourceFieldMapper.Mode.{}", v.to_ascii_uppercase())));
        }
        // `IndexMode.supportedSourceModes`.
        let im = crate::settings::get_str(settings, "index.mode").map(str::to_ascii_lowercase).unwrap_or_else(|| "standard".into());
        let columnar = matches!(im.as_str(), "columnar" | "logsdb_columnar");
        let supported: &[&str] = if columnar { &["SYNTHETIC", "COLUMNAR_STORED"] } else { &["DISABLED", "STORED", "SYNTHETIC"] };
        let up = v.to_ascii_uppercase();
        if !supported.contains(&up.as_str()) {
            return Err(EsError::illegal_argument(format!("unsupported source mode [{up}] for index mode [{im}]; supported values: [{}]", supported.join(", "))));
        }
    }
    let src = m.source.as_ref();
    let has = |k: &str| src.and_then(|s| s.get(k)).is_some();
    if has("mode") && has("enabled") {
        return Err(EsError::mapper_parsing("Failed to parse mapping: Cannot set both [mode] and [enabled] parameters").caused_by(EsError::mapper_parsing("Cannot set both [mode] and [enabled] parameters")));
    }
    let index_mode = crate::settings::get_str(settings, "index.mode");
    let mode = source_mode(setting, m, index_mode);
    let wrap = |msg: &str| EsError::mapper_parsing(format!("Failed to parse mapping: {msg}")).caused_by(EsError::illegal_argument(msg));
    if let (Some(im @ ("logsdb" | "time_series" | "columnar" | "logsdb_columnar")), SourceMode::Disabled) = (index_mode, mode) {
        return Err(wrap(&format!("_source can not be disabled in index using [{im}] index mode")));
    }
    // The setting wins over the mapping's `_source.mode` (and the index-mode
    // default): the effective mode is stored in the mapping so
    // `get_mapping` reports it.
    // `index.recovery.use_synthetic_source` may only be enabled when the
    // effective source mode is synthetic.
    // Columnar index modes rebuild recovery source synthetically.
    if let Some(im) = index_mode.map(str::to_ascii_lowercase).filter(|m| matches!(m.as_str(), "columnar" | "logsdb_columnar")) {
        if crate::settings::get_str(settings, "index.recovery.use_synthetic_source").is_some_and(|s| s.eq_ignore_ascii_case("false")) {
            return Err(EsError::illegal_argument(format!("The setting [index.recovery.use_synthetic_source] must not be false when [index.mode] is set to [{}].", im.to_ascii_uppercase())));
        }
    }
    if mode != SourceMode::Synthetic {
        let on = crate::settings::get_str(settings, "index.recovery.use_synthetic_source").map(|s| s == "true" || s.eq_ignore_ascii_case("true")).unwrap_or(false);
        if on {
            return Err(EsError::illegal_argument("Setting [index.recovery.use_synthetic_source] can only be enabled when the source mode is synthetic"));
        }
    }
    let filtered = ["includes", "excludes"].iter().any(|k| src.and_then(|s| s.get(*k)).is_some_and(|v| v.as_array().map(|a| !a.is_empty()).unwrap_or(true)));
    if mode == SourceMode::Synthetic && filtered {
        return Err(wrap("filtering the stored _source is incompatible with synthetic source"));
    }
    Ok(())
}

/// Canonical synthetic form of `src` under `m` / `lk`. The second element of
/// the result holds the `_ignored_source` entries (fields the synthetic
/// source could not represent), one base64 value per field in document
/// order, or `None` when there are none.
thread_local! {
    /// Tests: force the general walk (compare with [`Synth::plain_items`]).
    pub(crate) static FORCE_GENERAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: documents whose stored forms came from [`plain_forms`].
    pub(crate) static PLAIN_FORMS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// [`synthesize`] then [`store_forms`] in one walk for a plain document
/// (see [`Synth::plain_items`]: mapped keys only, no arrays): the canonical
/// source text and the residual are written while each leaf is
/// canonicalized, no canonical `Value` built and walked again. `None`:
/// not plain, or a leaf the general path handles (an ignored source entry,
/// an error), or no residuals here; the caller runs the two passes. Same
/// bytes as they write (tests compare every document).
pub fn plain_forms(src: &Value, m: &Mappings, lk: &FieldLookup, idx_keep: Keep, doc: &rustic_core::types::Document) -> Option<(String, Option<String>)> {
    let Value::Object(o) = src else { return None };
    if m.root.enabled == Some(false) || lk.may_have_vector_fields || !residuals_enabled() || FORCE_GENERAL.with(|f| f.get()) {
        return None;
    }
    let s = Synth { m, lk, idx_keep, now: crate::date::now_millis(), err: std::cell::RefCell::new(None), ignored: std::cell::RefCell::new(Vec::new()) };
    let cols = column_fields(lk);
    let empty = Mappings::default();
    let mut w = ResidualWalk { s: residual_synth(&empty, lk), cols: &cols, doc: IndexedDoc::new(doc), taken: Vec::new() };
    let mut full = Vec::with_capacity(4096);
    let mut res = Vec::with_capacity(1024);
    s.plain_forms(&m.root, &mut String::with_capacity(128), o, Some(&cols.1), &mut w, &mut full, &mut res)?;
    if s.err.borrow().is_some() || !s.ignored.borrow().is_empty() {
        return None;
    }
    let full = String::from_utf8(full).expect("serde_json writes UTF-8");
    PLAIN_FORMS.with(|n| n.set(n.get() + 1));
    if w.taken.is_empty() {
        return Some((full, None));
    }
    w.taken.sort_unstable();
    let mut out = String::with_capacity(res.len() + 8 * w.taken.len() + 2);
    out.push(RESIDUAL_MARK);
    write_ids(&mut out, &w.taken);
    out.push(RESIDUAL_MARK);
    out.push_str(std::str::from_utf8(&res).expect("serde_json writes UTF-8"));
    Some((full, Some(out)))
}

pub fn synthesize(src: &Value, m: &Mappings, lk: &FieldLookup, idx_keep: Keep) -> crate::error::Result<(Value, Option<Vec<IgnoredEntry>>)> {
    let Value::Object(o) = src else { return Ok((src.clone(), None)) };
    if m.root.enabled == Some(false) {
        return Ok((src.clone(), None));
    }
    let s = Synth { m, lk, idx_keep, now: crate::date::now_millis(), err: std::cell::RefCell::new(None), ignored: std::cell::RefCell::new(Vec::new()) };
    // Plain documents (mapped keys only, no arrays) walk the source in
    // place ([`Synth::plain_items`]); anything else takes the general walk
    // over an owned copy.
    let fast = if FORCE_GENERAL.with(|f| f.get()) { None } else { s.plain_items(&m.root, &mut String::with_capacity(128), o) };
    let out = Value::Object(match fast {
        Some(map) => map,
        None => {
            // A bailed plain walk recorded nothing it would not record again.
            s.ignored.borrow_mut().clear();
            *s.err.borrow_mut() = None;
            let entries: Vec<(String, Value)> = o.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            items_to_map(s.object_items(&m.root, "", entries, false))
        }
    });
    let ignored = s.ignored_source(src);
    match s.err.into_inner() {
        Some(e) => Err(e),
        None => Ok((out, ignored)),
    }
}

// ---------------------------------------------------------------------------
// Stored residual: synthetic source without the column-backed fields.
//
// ES stores no `_source` for a synthetic-source index: it rebuilds it at
// fetch time from doc values (and stored fields / `_ignored_source`). Here
// the canonical form is still computed at index time (it is what every
// reader must see, and the translog keeps it), but what is *stored* is the
// residual: that form without the leaves its doc-value columns rebuild
// exactly (keywords, ips, numbers, dates, booleans). A fetch renders those
// leaves from the columns with the same per-field rendering and puts them
// back in place. A document whose rebuilt form would differ in any byte is
// stored whole (index-time check), so the output is always the canonical
// form.

/// First char of a stored `_source` holding a residual:
/// `MARK ids MARK residual-json`, `ids` the comma-separated core ids of the
/// column-backed fields.
pub const RESIDUAL_MARK: char = '\u{1}';

/// A stored `_source` that is an empty residual whose column-backed fields
/// are exactly the ones whose columns hold a value for the document
/// (segment format [`rustic_core::codec::EMPTY_RESIDUAL_SINCE`]+): `MARK
/// \u{3}`, two bytes, where the ids list cost a logsdb document a dozen.
/// Readers rebuild the ids from the columns ([`present_ids`]).
pub const EMPTY_RESIDUAL: &str = "\u{1}\u{3}";

/// Whether column field `(kind, rf)` has a value for the document `cols`
/// reads (constant keywords and booleans never count: their presence is not
/// in the column).
fn column_has_value(rf: &ResolvedField, kind: ColKind, cols: &dyn DocColumns) -> bool {
    let id = rf.core_id.unwrap_or(u16::MAX);
    match kind {
        ColKind::Constant => false,
        ColKind::Keyword | ColKind::Ip | ColKind::I64 | ColKind::U64 | ColKind::F64 => cols.present(id),
        // A boolean column before v37 reads `false` for a doc without a
        // value: its presence is unknown, so a doc using one keeps its ids.
        ColKind::Bool => cols.bool_present(id).unwrap_or(false),
        ColKind::TextFromKeyword(child) => cols.present(child),
    }
}

/// The ids list of the column-backed fields that have a value for the
/// document `cols` reads ([`EMPTY_RESIDUAL`]'s ids).
pub fn present_ids(lk: &FieldLookup, cols: &dyn DocColumns) -> String {
    let cf = column_fields(lk);
    // Metadata fields (`_tsid`, added after parsing) are never in a source.
    let ids: Vec<u16> = match cols.present_fields() {
        // The document's own fields, each to the column fields it makes
        // present (a time series mapping has thousands of column fields; a
        // document a hundred-odd values).
        Some(fields) => {
            let mut ids: Vec<u16> = Vec::with_capacity(fields.len());
            for f in fields {
                let at = cf.2.partition_point(|&(p, _)| p < f);
                ids.extend(cf.2[at..].iter().take_while(|&&(p, _)| p == f)
                    .filter(|&&(_, i)| cf.0[i].2 != ColKind::Bool || column_has_value(&cf.0[i].1, ColKind::Bool, cols))
                    .map(|&(_, i)| cf.0[i].0));
            }
            ids.sort_unstable();
            ids.dedup();
            ids
        }
        None => cf.0.iter().filter(|(_, rf, kind, _)| !rf.name.starts_with('_') && column_has_value(rf, *kind, cols)).map(|(id, ..)| *id).collect(),
    };
    // Tests: the document's own fields give exactly the per-field answers.
    #[cfg(test)]
    if cols.present_fields().is_some() {
        let slow: Vec<u16> = cf.0.iter().filter(|(_, rf, kind, _)| !rf.name.starts_with('_') && column_has_value(rf, *kind, cols)).map(|(id, ..)| *id).collect();
        assert_eq!(ids, slow);
    }
    let mut out = String::new();
    write_ids(&mut out, &ids);
    out
}

/// `residual` as [`EMPTY_RESIDUAL`] when it holds nothing but its ids and
/// those are exactly the column-backed fields with a value in `doc` (then
/// the columns tell them back); else `residual` unchanged.
pub fn compact_residual(residual: String, lk: &FieldLookup, doc: &rustic_core::types::Document) -> String {
    let empty = matches!(residual_parts(&residual), Some((ids, None, "{}")) if {
        let present = present_ids(lk, &IndexedDoc::new(doc));
        let ids: Vec<u16> = parse_ids(ids).collect();
        let mut canon = String::new();
        write_ids(&mut canon, &ids);
        canon == present
    });
    if empty { EMPTY_RESIDUAL.to_string() } else { residual }
}

/// The source an [`EMPTY_RESIDUAL`] stands for: every column-backed field
/// that has a value (rendered once: a field renders `None` when its column
/// holds none), in canonical order; only the roots `root` accepts.
fn rehydrate_empty(lk: &FieldLookup, cols: &dyn DocColumns, root: &dyn Fn(&str) -> bool) -> String {
    with_empty_vals(lk, cols, root, |cf, vals, _| {
        let mut out = Vec::with_capacity(256);
        write_merged(&mut out, None, &cf.1, &cf.0, vals);
        String::from_utf8(out).unwrap_or_else(|_| "{}".to_string())
    })
}

/// [`rehydrate_empty`] as a JSON value, built directly (not written and
/// parsed back: ES|QL reads every row's source as a value).
pub fn rehydrate_empty_value(lk: &FieldLookup, cols: &dyn DocColumns, root: &dyn Fn(&str) -> bool) -> Value {
    fn build(node: &ColumnTrie, vals: &mut [Option<Value>]) -> Map<String, Value> {
        let mut m = Map::new();
        for (k, c) in &node.children {
            if !c.any(vals) {
                continue;
            }
            // As `write_merged`: a leaf's value, else the object below it.
            let v = match c.leaf.filter(|&i| vals[i].is_some()) {
                Some(i) => vals[i].take().unwrap_or(Value::Null),
                None => Value::Object(build(c, vals)),
            };
            m.insert(k.clone(), v);
        }
        m
    }
    with_empty_vals(lk, cols, root, |cf, vals, _| Value::Object(build(&cf.1, vals)))
}

thread_local! {
    /// [`with_empty_vals`]' rendered values, one slot per column field of
    /// the mapping (thousands): kept between documents, only the slots a
    /// document filled cleared after it.
    static EMPTY_VALS: std::cell::RefCell<Vec<Option<Value>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The rendered column values of an [`EMPTY_RESIDUAL`] document (every
/// column-backed field with a value, under the roots `root` accepts), by
/// [`ColumnFields`] entry, handed to `f` with the entries filled.
fn with_empty_vals<R>(lk: &FieldLookup, cols: &dyn DocColumns, root: &dyn Fn(&str) -> bool, f: impl FnOnce(&ColumnFields, &mut [Option<Value>], &[usize]) -> R) -> R {
    let cf = column_fields(lk);
    let empty = Mappings::default();
    let s = residual_synth(&empty, lk);
    // A boolean renders only from a column that knows the doc has a value.
    let eligible = |(_, rf, kind, keys): &(u16, std::sync::Arc<ResolvedField>, ColKind, Vec<String>)| {
        let kind_ok = match kind {
            ColKind::Constant => false,
            ColKind::Bool => cols.bool_present(rf.core_id.unwrap_or(u16::MAX)) == Some(true),
            _ => true,
        };
        kind_ok && !rf.name.starts_with('_') && keys.first().is_some_and(|k| root(k))
    };
    // A nested call (a renderer rehydrating another document) gets its own.
    let mut vals = EMPTY_VALS.with(|v| v.try_borrow_mut().map(|mut v| std::mem::take(&mut *v)).unwrap_or_default());
    // Every slot is None between documents (each filled one is reset).
    if vals.len() != cf.0.len() {
        vals.clear();
        vals.resize(cf.0.len(), None);
    }
    let mut filled: Vec<usize> = Vec::new();
    // Only the entries whose column the document may have a value in
    // (a segment's hundred-odd columns of a mapping's thousands): the
    // others render nothing, so they are not read.
    match cols.column_fields() {
        Some(fields) => {
            for fld in fields {
                let at = cf.2.partition_point(|&(p, _)| p < fld);
                for &(_, i) in cf.2[at..].iter().take_while(|&&(p, _)| p == fld) {
                    let e = &cf.0[i];
                    if vals[i].is_none() && eligible(e) {
                        vals[i] = render_column(&s, &e.1, e.2, cols);
                        if vals[i].is_some() {
                            filled.push(i);
                        }
                    }
                }
            }
        }
        None => {
            for (i, e) in cf.0.iter().enumerate() {
                if eligible(e) {
                    vals[i] = render_column(&s, &e.1, e.2, cols);
                    if vals[i].is_some() {
                        filled.push(i);
                    }
                }
            }
        }
    }
    let out = f(&cf, &mut vals, &filled);
    for i in filled {
        vals[i] = None;
    }
    EMPTY_VALS.with(|v| {
        if let Ok(mut v) = v.try_borrow_mut() {
            *v = vals;
        }
    });
    out
}

/// [`residual_parts`] of a stored residual, an [`EMPTY_RESIDUAL`]'s ids
/// rebuilt from `cols` into `buf`.
fn residual_parts_cols<'a>(raw: &'a str, lk: &FieldLookup, cols: &dyn DocColumns, buf: &'a mut String) -> Option<(&'a str, Option<&'a str>, &'a str)> {
    if raw == EMPTY_RESIDUAL {
        *buf = present_ids(lk, cols);
        return Some((buf.as_str(), None, "{}"));
    }
    residual_parts(raw)
}

/// A residual's ascending core ids, runs of consecutive ids as `a-b`
/// (`9,11-17,19`): a time series document lists a hundred-odd mostly
/// consecutive ids, which took most of its stored bytes one by one.
fn write_ids(out: &mut String, ids: &[u16]) {
    let mut i = 0;
    while i < ids.len() {
        let mut j = i;
        while j + 1 < ids.len() && ids[j + 1] == ids[j] + 1 {
            j += 1;
        }
        if i > 0 {
            out.push(',');
        }
        out.push_str(&ids[i].to_string());
        if j > i {
            out.push('-');
            out.push_str(&ids[j].to_string());
        }
        i = j + 1;
    }
}

/// The ids [`write_ids`] wrote (plain comma lists too).
fn parse_ids(ids: &str) -> impl Iterator<Item = u16> + '_ {
    ids.split(',').flat_map(|part| {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.parse::<u16>().ok(), b.parse::<u16>().ok()),
            None => (part.parse::<u16>().ok(), part.parse::<u16>().ok()),
        };
        match (a, b) {
            (Some(a), Some(b)) if a <= b => a..=b,
            // (An empty range.)
            _ => 1..=0,
        }
    })
}

/// Whether a stored `_source` is a synthetic residual.
#[inline]
pub fn is_residual(raw: &str) -> bool {
    raw.starts_with(RESIDUAL_MARK)
}

/// The residual JSON of a residual `_source` (the fields not rebuilt from
/// columns), or `raw` itself.
pub fn residual_json(raw: &str) -> &str {
    if !is_residual(raw) {
        return raw;
    }
    if raw == EMPTY_RESIDUAL {
        return "{}";
    }
    residual_parts(raw).map_or("", |(_, _, json)| json)
}

/// Opens a residual's `_ignored_source` spec, between its ids and its JSON
/// (`MARK ids MARK SPEC spec-json SPEC json`): the entries the residual's
/// own string leaves stand for, which are then not stored a second time
/// (see [`store_forms`]).
pub const IGNORED_SPEC_MARK: char = '\u{2}';

/// (ids, ignored-source spec, residual JSON) of a residual `_source`.
pub fn residual_parts(raw: &str) -> Option<(&str, Option<&str>, &str)> {
    let body = raw.strip_prefix(RESIDUAL_MARK)?;
    let (ids, rest) = body.split_once(RESIDUAL_MARK)?;
    match rest.strip_prefix(IGNORED_SPEC_MARK) {
        Some(r) => {
            let (spec, json) = r.split_once(IGNORED_SPEC_MARK)?;
            Some((ids, Some(spec), json))
        }
        None => Some((ids, None, rest)),
    }
}

/// `_ignored_source` entries that a residual's string leaves stand for:
/// each entry's name, parent offset and leaf, when every entry of the
/// document is a single string (a `match_only_text` message, an unmapped
/// string) that the residual keeps whole at that field's path. `None`: the
/// entries are stored as they are.
fn foldable_ignored(synth: &Value, lk: &FieldLookup, entries: &[IgnoredEntry], taken: &[u16]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let mut spec = Vec::with_capacity(entries.len());
    for e in entries {
        let s = e.value.strip_prefix(b"S")?;
        let keys = leaf_keys(lk, &e.name);
        let Some(Value::String(leaf)) = get_at(synth, &keys) else { return None };
        if leaf.as_bytes() != s {
            return None;
        }
        // The leaf must stay in the residual (not rebuilt from a column).
        if lk.fields.get(&e.name).and_then(|f| f.core_id).is_some_and(|id| taken.contains(&id)) {
            return None;
        }
        spec.push(json!([e.name, e.parent]));
    }
    Some(Value::Array(spec).to_string())
}

/// `stored` (a whole stored synthetic `_source`, no column rebuilt) as a
/// residual that folds the document's `_ignored_source` entries into its
/// own string leaves (taken from `ignored`, which is then `None`), so an
/// unmapped string is stored once, not also as an `_ignored_source` value;
/// `stored` itself when they do not fold.
pub fn fold_ignored(stored: String, synth: &Value, lk: &FieldLookup, ignored: &mut Option<Vec<IgnoredEntry>>) -> String {
    if !residuals_enabled() {
        return stored;
    }
    let Some(spec) = ignored.as_deref().and_then(|e| foldable_ignored(synth, lk, e, &[])) else { return stored };
    *ignored = None;
    let mut out = String::with_capacity(stored.len() + spec.len() + 4);
    out.push(RESIDUAL_MARK);
    out.push(RESIDUAL_MARK);
    out.push(IGNORED_SPEC_MARK);
    out.push_str(&spec);
    out.push(IGNORED_SPEC_MARK);
    out.push_str(&stored);
    out
}

/// The stored `_ignored_source` values of a document whose residual folded
/// them ([`foldable_ignored`]), rebuilt from `full`, its rehydrated
/// source. `None` when `raw` (the stored `_source`) folded nothing.
pub fn folded_ignored_source(raw: &str, full: &str, lk: &FieldLookup, format: IgnoredSourceFormat) -> Option<Vec<String>> {
    let (_, spec, _) = residual_parts(raw)?;
    let spec: Value = serde_json::from_str(spec?).ok()?;
    let full: Value = serde_json::from_str(full).ok()?;
    let mut entries = Vec::new();
    for e in spec.as_array()? {
        let name = e.get(0)?.as_str()?.to_string();
        let parent = e.get(1)?.as_u64()? as usize;
        let leaf = get_at(&full, &leaf_keys(lk, &name))?.as_str()?;
        let mut value = Vec::with_capacity(leaf.len() + 1);
        value.push(b'S');
        value.extend_from_slice(leaf.as_bytes());
        entries.push(IgnoredEntry { name, parent, value });
    }
    Some(encode_ignored_source(&entries, format))
}

/// Whether synthetic source stores residuals (`RUSTIC_SYNTHETIC_RESIDUAL=0`
/// stores the whole canonical form, as before; for comparisons).
fn residuals_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RUSTIC_SYNTHETIC_RESIDUAL").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false")))
}

/// One document's doc-value columns.
pub trait DocColumns {
    /// Keyword (and ip term) values, sorted and de-duplicated.
    fn keywords(&self, field: u16) -> Vec<String>;
    fn i64s(&self, field: u16) -> Vec<i64>;
    fn f64s(&self, field: u16) -> Vec<f64>;
    fn u64s(&self, field: u16) -> Vec<u64>;
    fn bools(&self, field: u16) -> Vec<bool>;
    /// Whether the column of `field` holds a value for the document (not
    /// meaningful for booleans, whose column reads `false` for a missing
    /// value).
    fn present(&self, field: u16) -> bool {
        !self.keywords(field).is_empty() || !self.i64s(field).is_empty() || !self.u64s(field).is_empty() || !self.f64s(field).is_empty()
    }
    /// Every field [`Self::present`] holds, ascending, when the document
    /// can list them (`None`: ask per field).
    fn present_fields(&self) -> Option<Vec<u16>> {
        None
    }
    /// The fields that may have a value (a superset of
    /// [`Self::present_fields`], cheaper to list: a segment's columns),
    /// ascending; `None`: any field may.
    fn column_fields(&self) -> Option<Vec<u16>> {
        self.present_fields()
    }
    /// Whether boolean `field` has a value for the document, when its
    /// column can tell (`None`: a column before
    /// [`rustic_core::codec::BOOL_PRESENCE_SINCE`]).
    fn bool_present(&self, field: u16) -> Option<bool> {
        let _ = field;
        None
    }
}

/// The columns of a parsed (or pending) document: the values its fields
/// add to the columns, read the way the columns return them.
impl DocColumns for rustic_core::types::Document {
    fn keywords(&self, field: u16) -> Vec<String> {
        let mut v: Vec<String> = self.fields.iter().filter(|(f, _)| *f == field).filter_map(|(_, v)| match v {
            rustic_core::types::FieldValue::Keyword(s) => Some(s.clone()),
            _ => None,
        }).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    fn i64s(&self, field: u16) -> Vec<i64> {
        self.fields.iter().filter(|(f, _)| *f == field).filter_map(|(_, v)| match v {
            rustic_core::types::FieldValue::I64(x) | rustic_core::types::FieldValue::Date(x) => Some(*x),
            _ => None,
        }).collect()
    }
    fn f64s(&self, field: u16) -> Vec<f64> {
        self.fields.iter().filter(|(f, _)| *f == field).filter_map(|(_, v)| match v {
            rustic_core::types::FieldValue::F64(x) => Some(*x),
            _ => None,
        }).collect()
    }
    fn u64s(&self, field: u16) -> Vec<u64> {
        self.fields.iter().filter(|(f, _)| *f == field).filter_map(|(_, v)| match v {
            rustic_core::types::FieldValue::U64(x) => Some(*x),
            _ => None,
        }).collect()
    }
    fn bools(&self, field: u16) -> Vec<bool> {
        let v: Vec<bool> = self.fields.iter().filter(|(f, _)| *f == field).filter_map(|(_, v)| match v {
            rustic_core::types::FieldValue::Bool(b) => Some(*b),
            _ => None,
        }).collect();
        // A boolean column holds one value per doc: more are not rebuilt.
        if v.len() == 1 { v } else { Vec::new() }
    }
    /// The parsed document knows its booleans (a pending one served by a
    /// realtime GET).
    fn bool_present(&self, field: u16) -> Option<bool> {
        Some(self.fields.iter().any(|(f, v)| *f == field && matches!(v, rustic_core::types::FieldValue::Bool(_))))
    }
}

/// The columns of document `doc` of a leaf (segment or memtable).
pub struct LeafColumns<'a, 'b> {
    pub leaf: &'a rustic_core::query::reader::LeafReader<'b>,
    pub doc: rustic_core::types::DocId,
}

impl DocColumns for LeafColumns<'_, '_> {
    fn keywords(&self, field: u16) -> Vec<String> {
        let mut v = self.leaf.keyword_values(field, self.doc);
        v.sort_unstable();
        v.dedup();
        v
    }
    fn i64s(&self, field: u16) -> Vec<i64> {
        self.leaf.i64_values(field, self.doc)
    }
    fn f64s(&self, field: u16) -> Vec<f64> {
        self.leaf.f64_values(field, self.doc)
    }
    fn u64s(&self, field: u16) -> Vec<u64> {
        self.leaf.u64_values(field, self.doc)
    }
    fn bools(&self, field: u16) -> Vec<bool> {
        self.leaf.bool_value(field, self.doc).into_iter().collect()
    }
    fn present(&self, field: u16) -> bool {
        self.leaf.has_doc_value(field, self.doc)
    }
    /// The segment's columns with a value for the doc: a field without a
    /// column here has none (the mapping's other thousands are not asked).
    fn present_fields(&self) -> Option<Vec<u16>> {
        let ids = self.leaf.column_field_ids()?;
        Some(ids.into_iter().filter(|&f| self.leaf.has_doc_value(f, self.doc)).collect())
    }
    fn column_fields(&self) -> Option<Vec<u16>> {
        self.leaf.column_field_ids()
    }
    fn bool_present(&self, field: u16) -> Option<bool> {
        self.leaf.bool_presence(field, self.doc)
    }
}

/// One field's opened column on a segment ([`CachedLeafColumns`]): the
/// common single-valued kinds; anything else reads through the leaf.
enum CachedColumn<'s> {
    I64(rustic_core::codec::fastfields::ColumnReader<'s, i64>),
    U64(rustic_core::codec::fastfields::ColumnReader<'s, u64>),
    F64(rustic_core::codec::fastfields::ColumnReader<'s, f64>),
    Kw(rustic_core::codec::fastfields::KeywordReader<'s>),
    Bool(rustic_core::codec::fastfields::BoolReader<'s>),
    Leaf,
}

/// A segment's columns opened once for a batch of its documents (ES|QL
/// reads every row's synthetic source: a hundred-odd columns a document,
/// each found in the directory and its header parsed per document
/// before). Answers exactly as [`LeafColumns`] does.
pub struct ColumnCache<'s> {
    segment: Option<&'s rustic_core::segment::Segment>,
    columns: std::cell::RefCell<std::collections::HashMap<u16, std::rc::Rc<CachedColumn<'s>>, ahash::RandomState>>,
}

impl<'s> ColumnCache<'s> {
    /// For `leaf` (a memtable reads through the leaf).
    pub fn new(leaf: &rustic_core::query::reader::LeafReader<'s>) -> Self {
        let segment = match *leaf {
            rustic_core::query::reader::LeafReader::Segment(s) => Some(s),
            _ => None,
        };
        ColumnCache { segment, columns: Default::default() }
    }

    fn column(&self, field: u16) -> std::rc::Rc<CachedColumn<'s>> {
        if let Some(c) = self.columns.borrow().get(&field) {
            return c.clone();
        }
        let c = std::rc::Rc::new(match self.segment {
            None => CachedColumn::Leaf,
            // The order `LeafReader`'s value reads try them in.
            Some(s) if s.keyword_from_postings(field) => CachedColumn::Leaf,
            Some(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_i64(field) {
                    CachedColumn::I64(c)
                } else if let Some(c) = ff.get_u64(field) {
                    CachedColumn::U64(c)
                } else if let Some(c) = ff.get_f64(field) {
                    CachedColumn::F64(c)
                } else if let Some(c) = s.keyword_reader(field) {
                    CachedColumn::Kw(c)
                } else if let Some(c) = ff.get_bool(field) {
                    CachedColumn::Bool(c)
                } else {
                    CachedColumn::Leaf
                }
            }
        });
        self.columns.borrow_mut().insert(field, c.clone());
        c
    }
}

/// [`LeafColumns`] reading through a [`ColumnCache`].
pub struct CachedLeafColumns<'a, 'b, 'c> {
    pub leaf: &'a rustic_core::query::reader::LeafReader<'b>,
    pub doc: rustic_core::types::DocId,
    pub cache: &'c ColumnCache<'b>,
}

impl CachedLeafColumns<'_, '_, '_> {
    fn plain(&self) -> LeafColumns<'_, '_> {
        LeafColumns { leaf: self.leaf, doc: self.doc }
    }
}

impl DocColumns for CachedLeafColumns<'_, '_, '_> {
    fn keywords(&self, field: u16) -> Vec<String> {
        match &*self.cache.column(field) {
            // A single-valued column: at most one value, so sorted.
            CachedColumn::Kw(c) => c.get(self.doc).map(|v| v.into_owned()).into_iter().collect(),
            _ => self.plain().keywords(field),
        }
    }
    fn i64s(&self, field: u16) -> Vec<i64> {
        match &*self.cache.column(field) {
            CachedColumn::I64(c) => c.get(self.doc).into_iter().collect(),
            _ => self.plain().i64s(field),
        }
    }
    fn f64s(&self, field: u16) -> Vec<f64> {
        match &*self.cache.column(field) {
            CachedColumn::F64(c) => c.get(self.doc).into_iter().collect(),
            _ => self.plain().f64s(field),
        }
    }
    fn u64s(&self, field: u16) -> Vec<u64> {
        match &*self.cache.column(field) {
            CachedColumn::U64(c) => c.get(self.doc).into_iter().collect(),
            _ => self.plain().u64s(field),
        }
    }
    fn bools(&self, field: u16) -> Vec<bool> {
        match &*self.cache.column(field) {
            CachedColumn::Bool(c) => if c.has_value(self.doc) { vec![c.get(self.doc)] } else { Vec::new() },
            _ => self.plain().bools(field),
        }
    }
    fn present(&self, field: u16) -> bool {
        match &*self.cache.column(field) {
            CachedColumn::I64(c) => c.has_value(self.doc),
            CachedColumn::U64(c) => c.has_value(self.doc),
            CachedColumn::F64(c) => c.has_value(self.doc),
            CachedColumn::Kw(c) => c.ord(self.doc) != rustic_core::codec::fastfields::MISSING_ORD,
            CachedColumn::Bool(c) => c.has_value(self.doc),
            CachedColumn::Leaf => self.plain().present(field),
        }
    }
    fn present_fields(&self) -> Option<Vec<u16>> {
        let ids = self.leaf.column_field_ids()?;
        Some(ids.into_iter().filter(|&f| self.present(f)).collect())
    }
    fn column_fields(&self) -> Option<Vec<u16>> {
        self.leaf.column_field_ids()
    }
    fn bool_present(&self, field: u16) -> Option<bool> {
        match &*self.cache.column(field) {
            CachedColumn::Bool(c) => c.presence_known().then(|| c.has_value(self.doc)),
            _ => self.plain().bool_present(field),
        }
    }
}

/// How a column-backed field's values come back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ColKind {
    /// A `constant_keyword` with its mapped value: rebuilt from the mapping
    /// (ES synthetic source does the same); no column is read.
    Constant,
    Keyword,
    Ip,
    I64,
    U64,
    F64,
    Bool,
    /// A `text` / `match_only_text` value rebuilt from the column of its
    /// keyword multi-field (this core id), as ES rebuilds it: only where
    /// that renders the value exactly (within `ignore_above`, no
    /// normalizer); otherwise the residual keeps the value.
    TextFromKeyword(u16),
}

/// Whether `rf`'s synthetic value can be rebuilt from its column: a
/// top-level (not nested) doc-values field that no `copy_to` feeds from
/// and that is not a multi-field, of a type whose column holds the values.
fn col_kind(rf: &ResolvedField) -> Option<ColKind> {
    if rf.declared == EsType::ConstantKeyword && rf.core_id.is_some() && rf.nested_path.is_none() && !rf.is_multi_field
        && rf.copy_to.is_empty() && rf.constant_value.as_ref().is_some_and(Value::is_string) {
        return Some(ColKind::Constant);
    }
    if rf.core_id.is_none() || !rf.doc_values || !rf.copy_to.is_empty() || rf.is_multi_field || rf.nested_path.is_some() || rf.alias_path.is_some() || rf.ty != rf.declared {
        return None;
    }
    Some(match rf.declared {
        EsType::Keyword => ColKind::Keyword,
        EsType::Ip => ColKind::Ip,
        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Date => ColKind::I64,
        EsType::UnsignedLong => ColKind::U64,
        EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => ColKind::F64,
        EsType::Boolean => ColKind::Bool,
        _ => return None,
    })
}

/// The source values `rf`'s column holds for a document.
fn column_raws(rf: &ResolvedField, kind: ColKind, cols: &dyn DocColumns) -> Vec<Value> {
    let id = rf.core_id.unwrap_or(u16::MAX);
    match kind {
        ColKind::Constant => rf.constant_value.iter().cloned().collect(),
        ColKind::Keyword => cols.keywords(id).into_iter().map(Value::String).collect(),
        ColKind::Ip => cols.keywords(id).into_iter().map(|t| Value::String(crate::values::ip_display(&t))).collect(),
        ColKind::I64 => cols.i64s(id).into_iter().map(|x| Value::Number(x.into())).collect(),
        ColKind::U64 => cols.u64s(id).into_iter().map(|x| Value::Number(x.into())).collect(),
        ColKind::F64 => cols.f64s(id).into_iter().filter_map(|x| Number::from_f64(x).map(Value::Number)).collect(),
        ColKind::Bool => cols.bools(id).into_iter().map(Value::Bool).collect(),
        ColKind::TextFromKeyword(child) => cols.keywords(child).into_iter().map(Value::String).collect(),
    }
}

/// `rf`'s synthetic value rendered from its column (`None`: no value).
fn render_column(s: &Synth<'_>, rf: &ResolvedField, kind: ColKind, cols: &dyn DocColumns) -> Option<Value> {
    // Keywords and ips come back sorted and de-duplicated (keyword bytes,
    // ip address bytes), which is their synthetic form: rendered directly
    // (the index-time check compares the result with the canonical form).
    match kind {
        ColKind::Constant => return rf.constant_value.clone(),
        // The keyword column's values (sorted, de-duplicated) as the text.
        ColKind::TextFromKeyword(child) => return combine(cols.keywords(child).into_iter().map(Value::String).collect()),
        // Dates: the column holds epoch milliseconds, rendered in the
        // field's format as the canonical form renders a parsed date
        // (re-parsing the number would read it in the format's unit:
        // `epoch_second` multiplied it by 1000).
        ColKind::I64 if rf.declared == EsType::Date => {
            let mut ms = cols.i64s(rf.core_id.unwrap_or(u16::MAX));
            ms.sort_unstable();
            return combine(ms.into_iter().map(|ms| Value::String(rf.date_format.format(ms, crate::date::Tz::UTC))).collect());
        }
        ColKind::Keyword if rf.normalizer.is_none() => {
            return combine(cols.keywords(rf.core_id.unwrap_or(u16::MAX)).into_iter().map(Value::String).collect());
        }
        ColKind::Ip => {
            let terms = cols.keywords(rf.core_id.unwrap_or(u16::MAX));
            return combine(terms.iter().map(|t| Value::String(match crate::values::term_to_ip(t) {
                Some(b) => crate::values::ip_display_bytes(b),
                None => t.clone(),
            })).collect());
        }
        _ => {}
    }
    let raws = column_raws(rf, kind, cols);
    if raws.is_empty() {
        return None;
    }
    s.canon(rf, &raws, true).and_then(combine)
}

/// The object keys of field `name` in the synthetic source: the mapped
/// object path splits (a `subobjects: false` object keeps the rest of the
/// name as one key).
fn leaf_keys(lk: &FieldLookup, name: &str) -> Vec<String> {
    // Under `subobjects: false` the dotted rest is no object path, so it
    // stays one key.
    let mut keys = Vec::new();
    let mut start = 0;
    for (i, _) in name.match_indices('.') {
        let prefix = &name[..i];
        if lk.objects.contains_key(prefix) && !lk.fields.contains_key(prefix) {
            keys.push(name[start..i].to_string());
            start = i + 1;
        }
    }
    keys.push(name[start..].to_string());
    keys
}

/// Remove the value at `keys` from `v` (dropping objects it empties).
#[cfg(test)]
fn take_at(v: &mut Value, keys: &[String]) -> Option<Value> {
    let Value::Object(m) = v else { return None };
    if keys.len() == 1 {
        return m.shift_remove(&keys[0]);
    }
    let child = m.get_mut(&keys[0])?;
    let out = take_at(child, &keys[1..]);
    if child.as_object().is_some_and(|c| c.is_empty()) {
        m.shift_remove(&keys[0]);
    }
    out
}

/// The value at `keys` in `v`.
fn get_at<'v>(v: &'v Value, keys: &[String]) -> Option<&'v Value> {
    let mut cur = v;
    for k in keys {
        cur = cur.as_object()?.get(k)?;
    }
    Some(cur)
}

/// Put `val` at `keys` into `v`, each key at its sorted position (the
/// synthetic source's objects are sorted by key), creating objects.
fn put_at(v: &mut Value, keys: &[String], val: Value) -> bool {
    let Value::Object(m) = v else { return false };
    let key = &keys[0];
    if keys.len() == 1 {
        if m.contains_key(key) {
            return false;
        }
        let at = m.keys().position(|k| k.as_str() > key.as_str()).unwrap_or(m.len());
        m.shift_insert(at, key.clone(), val);
        return true;
    }
    if !m.contains_key(key) {
        let at = m.keys().position(|k| k.as_str() > key.as_str()).unwrap_or(m.len());
        m.shift_insert(at, key.clone(), Value::Object(Map::new()));
    }
    put_at(m.get_mut(key).expect("inserted above"), &keys[1..], val)
}

/// The column-backed fields of a mapping: (core id, field, kind, object
/// keys in the synthetic source), by core id. Cached on the lookup.
/// Also the trie of those keys, which the index-time walk follows.
#[derive(Debug)]
pub struct ColumnFields(Vec<(u16, std::sync::Arc<ResolvedField>, ColKind, Vec<String>)>, ColumnTrie, Vec<(u16, usize)>);

/// One object level of the column-backed fields' key paths.
#[derive(Debug, Default)]
struct ColumnTrie {
    /// The [`ColumnFields`] entry whose path ends here.
    leaf: Option<usize>,
    /// By key, ascending (the residual's object order).
    children: Vec<(String, ColumnTrie)>,
    /// The entries whose paths end at or below this node, ascending: a
    /// document renders a hundred-odd of a mapping's thousands of column
    /// fields, so a subtree is asked by its leaves, not walked.
    leaves: Vec<u32>,
}

impl ColumnTrie {
    /// The child at key `k`.
    fn child(&self, k: &str) -> Option<&ColumnTrie> {
        self.children.binary_search_by(|(c, _)| c.as_str().cmp(k)).ok().map(|i| &self.children[i].1)
    }

    /// The child at key `k`, added when missing (keys stay sorted).
    fn child_mut(&mut self, k: &str) -> &mut ColumnTrie {
        let i = match self.children.binary_search_by(|(c, _)| c.as_str().cmp(k)) {
            Ok(i) => i,
            Err(i) => {
                self.children.insert(i, (k.to_string(), ColumnTrie::default()));
                i
            }
        };
        &mut self.children[i].1
    }

    fn finish(&mut self) {
        let mut leaves: Vec<u32> = self.leaf.map(|i| i as u32).into_iter().collect();
        for (_, c) in &mut self.children {
            c.finish();
            leaves.extend_from_slice(&c.leaves);
        }
        leaves.sort_unstable();
        self.leaves = leaves;
    }

    /// Whether some column below this node rendered a value.
    fn any(&self, vals: &[Option<Value>]) -> bool {
        self.leaves.iter().any(|&i| vals[i as usize].is_some())
    }
}

/// The keyword multi-field of text field `rf` whose column can rebuild its
/// values ([`ColKind::TextFromKeyword`]).
fn text_keyword_child(lk: &FieldLookup, rf: &ResolvedField) -> Option<u16> {
    if !matches!(rf.declared, EsType::Text | EsType::MatchOnlyText) || rf.ty != rf.declared || rf.core_id.is_none() || rf.is_multi_field
        || !rf.copy_to.is_empty() || rf.nested_path.is_some() || rf.alias_path.is_some() || rf.stored {
        return None;
    }
    rf.multi_fields.iter().filter_map(|name| lk.fields.get(name)).find(|c| {
        c.declared == EsType::Keyword && c.ty == EsType::Keyword && c.doc_values && c.normalizer.is_none() && c.core_id.is_some()
    }).and_then(|c| c.core_id)
}

fn column_fields(lk: &FieldLookup) -> std::sync::Arc<ColumnFields> {
    lk.synthetic_columns
        .get_or_init(|| {
            let mut out: Vec<_> = lk.fields.values().filter_map(|rf| {
                let kind = col_kind(rf).or_else(|| text_keyword_child(lk, rf).map(ColKind::TextFromKeyword))?;
                Some((rf.core_id?, rf.clone(), kind, leaf_keys(lk, &rf.name)))
            }).collect();
            out.sort_by_key(|(id, ..)| *id);
            out.dedup_by_key(|(id, ..)| *id);
            let mut trie = ColumnTrie::default();
            for (i, (.., keys)) in out.iter().enumerate() {
                let mut node = &mut trie;
                for k in keys {
                    node = node.child_mut(k);
                }
                node.leaf.get_or_insert(i);
            }
            trie.finish();
            // `(field whose presence column_has_value reads, entry)`, sorted:
            // what `present_ids` asks of a document that lists its fields.
            let mut probes: Vec<(u16, usize)> = out.iter().enumerate().filter(|(_, (_, rf, ..))| !rf.name.starts_with('_')).filter_map(|(i, (_, rf, kind, _))| match kind {
                ColKind::Keyword | ColKind::Ip | ColKind::I64 | ColKind::U64 | ColKind::F64 => Some((rf.core_id.unwrap_or(u16::MAX), i)),
                ColKind::TextFromKeyword(child) => Some((*child, i)),
                // Asked again per document (only a column that knows its
                // presence counts it).
                ColKind::Bool => Some((rf.core_id.unwrap_or(u16::MAX), i)),
                ColKind::Constant => None,
            }).collect();
            probes.sort_unstable();
            std::sync::Arc::new(ColumnFields(out, trie, probes))
        })
        .clone()
}

/// A parsed document's fields by field id (stable within a field), for
/// the per-field column reads of the index-time residual check.
struct IndexedDoc<'d> {
    doc: &'d rustic_core::types::Document,
    order: Vec<(u16, u32)>,
}

impl<'d> IndexedDoc<'d> {
    fn new(doc: &'d rustic_core::types::Document) -> Self {
        let mut order: Vec<(u16, u32)> = doc.fields.iter().enumerate().map(|(i, (f, _))| (*f, i as u32)).collect();
        order.sort_unstable();
        IndexedDoc { doc, order }
    }

    fn values(&self, field: u16) -> impl Iterator<Item = &'d rustic_core::types::FieldValue> + '_ {
        let lo = self.order.partition_point(|(f, _)| *f < field);
        let doc = self.doc;
        self.order[lo..].iter().take_while(move |(f, _)| *f == field).map(move |(_, i)| &doc.fields[*i as usize].1)
    }

    /// Whether the keyword column of `field` renders exactly `want`
    /// ([`render_column`] for a keyword without normalizer, without
    /// building the values).
    fn keyword_renders(&self, field: u16, want: &Value) -> bool {
        let mut vals: Vec<&str> = self.values(field).filter_map(|v| match v {
            rustic_core::types::FieldValue::Keyword(s) => Some(s.as_str()),
            _ => None,
        }).collect();
        vals.sort_unstable();
        vals.dedup();
        match want {
            Value::String(w) => vals.len() == 1 && vals[0] == w,
            Value::Array(a) => a.len() > 1 && a.len() == vals.len() && a.iter().zip(&vals).all(|(x, v)| x.as_str() == Some(*v)),
            _ => false,
        }
    }
}

/// Whether boolean columns written now know their docs with a value
/// ([`rustic_core::codec::BOOL_PRESENCE_SINCE`]).
fn bool_columns_tell() -> bool {
    rustic_core::codec::write_version() >= rustic_core::codec::BOOL_PRESENCE_SINCE
}

/// Same values as the [`rustic_core::types::Document`] implementation.
impl DocColumns for IndexedDoc<'_> {
    fn keywords(&self, field: u16) -> Vec<String> {
        let mut v: Vec<String> = self.values(field).filter_map(|v| match v {
            rustic_core::types::FieldValue::Keyword(s) => Some(s.clone()),
            _ => None,
        }).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    fn i64s(&self, field: u16) -> Vec<i64> {
        self.values(field).filter_map(|v| match v {
            rustic_core::types::FieldValue::I64(x) | rustic_core::types::FieldValue::Date(x) => Some(*x),
            _ => None,
        }).collect()
    }
    fn f64s(&self, field: u16) -> Vec<f64> {
        self.values(field).filter_map(|v| match v {
            rustic_core::types::FieldValue::F64(x) => Some(*x),
            _ => None,
        }).collect()
    }
    fn u64s(&self, field: u16) -> Vec<u64> {
        self.values(field).filter_map(|v| match v {
            rustic_core::types::FieldValue::U64(x) => Some(*x),
            _ => None,
        }).collect()
    }
    fn bools(&self, field: u16) -> Vec<bool> {
        let v: Vec<bool> = self.values(field).filter_map(|v| match v {
            rustic_core::types::FieldValue::Bool(b) => Some(*b),
            _ => None,
        }).collect();
        if v.len() == 1 { v } else { Vec::new() }
    }
    /// A value any of the four lists would hold, without building them
    /// (every column-backed field of the mapping is asked, per document).
    fn present(&self, field: u16) -> bool {
        use rustic_core::types::FieldValue as F;
        let present = self.values(field).any(|v| matches!(v, F::Keyword(_) | F::I64(_) | F::Date(_) | F::U64(_) | F::F64(_)));
        #[cfg(test)]
        assert_eq!(present, !self.keywords(field).is_empty() || !self.i64s(field).is_empty() || !self.u64s(field).is_empty() || !self.f64s(field).is_empty());
        present
    }
    fn present_fields(&self) -> Option<Vec<u16>> {
        use rustic_core::types::FieldValue as F;
        let bools = bool_columns_tell();
        let mut out: Vec<u16> = self.order.iter()
            .filter(|&&(_, i)| matches!(self.doc.fields[i as usize].1, F::Keyword(_) | F::I64(_) | F::Date(_) | F::U64(_) | F::F64(_))
                || (bools && matches!(self.doc.fields[i as usize].1, F::Bool(_))))
            .map(|&(f, _)| f).collect();
        out.dedup();
        Some(out)
    }
    /// The document's boolean values reach a column that knows its docs
    /// with a value only when segments are written at v37 or later (the
    /// format floor only rises, so its seal and merges write them so).
    fn bool_present(&self, field: u16) -> Option<bool> {
        bool_columns_tell().then(|| self.values(field).any(|v| matches!(v, rustic_core::types::FieldValue::Bool(_))))
    }
}

fn residual_synth<'a>(m: &'a Mappings, lk: &'a FieldLookup) -> Synth<'a> {
    Synth { m, lk, idx_keep: Keep::None, now: crate::date::now_millis(), err: std::cell::RefCell::new(None), ignored: std::cell::RefCell::new(Vec::new()) }
}

#[cfg(test)]
/// Whether every object on the path to `keys` in `v` has strictly
/// ascending keys: then [`put_at`] puts a removed leaf (and any object its
/// removal emptied) back at exactly its position.
fn path_sorted(v: &Value, keys: &[String], memo: &mut Vec<(usize, bool)>) -> bool {
    let mut cur = v;
    for k in keys {
        let Some(m) = cur.as_object() else { return false };
        let addr = m as *const Map<String, Value> as usize;
        let sorted = match memo.iter().find(|(a, _)| *a == addr) {
            Some((_, s)) => *s,
            None => {
                let s = m.keys().zip(m.keys().skip(1)).all(|(a, b)| a.as_str() < b.as_str());
                memo.push((addr, s));
                s
            }
        };
        if !sorted {
            return false;
        }
        match m.get(k) {
            Some(next) => cur = next,
            None => return false,
        }
    }
    true
}

/// The two stored forms of canonical synthetic source `synth` of a
/// document whose indexed fields are `doc`, from one walk: `synth`
/// serialized (the translog's source), and the residual to store: `synth`
/// without the leaves its columns render identically (each in an object
/// sorted by key, so a fetch puts it back at the same position; objects
/// the removals empty are dropped), with their core ids. The residual is
/// `None` when no field is rebuilt from columns (the document then stores
/// `synth` whole). Same result as the reference `residual_for` (tests).
///
/// `ignored`: the document's `_ignored_source` entries. When the residual
/// keeps every one of them whole as a string leaf (big5's
/// `match_only_text` message), it records them in its header instead and
/// they are dropped from `ignored` (stored once, not twice; readers rebuild
/// them with [`folded_ignored_source`]).
pub fn store_forms(synth: &Value, lk: &FieldLookup, doc: &rustic_core::types::Document, ignored: &mut Option<Vec<IgnoredEntry>>) -> (String, Option<String>) {
    let Value::Object(root) = synth else { return (synth.to_string(), None) };
    if lk.may_have_vector_fields || !residuals_enabled() {
        return (synth.to_string(), None);
    }
    let cols = column_fields(lk);
    let empty = Mappings::default();
    let mut w = ResidualWalk { s: residual_synth(&empty, lk), cols: &cols, doc: IndexedDoc::new(doc), taken: Vec::new() };
    let mut full = Vec::with_capacity(4096);
    let mut res = Vec::with_capacity(1024);
    w.object(root, &cols.1, true, &mut full, &mut res);
    let full = String::from_utf8(full).expect("serde_json writes UTF-8");
    if w.taken.is_empty() {
        return (full, None);
    }
    w.taken.sort_unstable();
    let mut out = String::with_capacity(res.len() + 8 * w.taken.len() + 2);
    out.push(RESIDUAL_MARK);
    write_ids(&mut out, &w.taken);
    out.push(RESIDUAL_MARK);
    if let Some(spec) = ignored.as_deref().and_then(|e| foldable_ignored(synth, lk, e, &w.taken)) {
        out.push(IGNORED_SPEC_MARK);
        out.push_str(&spec);
        out.push(IGNORED_SPEC_MARK);
        *ignored = None;
    }
    out.push_str(std::str::from_utf8(&res).expect("serde_json writes UTF-8"));
    (full, Some(out))
}

struct ResidualWalk<'a> {
    s: Synth<'a>,
    cols: &'a ColumnFields,
    doc: IndexedDoc<'a>,
    /// Core ids of the leaves left out of the residual.
    taken: Vec<u16>,
}

impl ResidualWalk<'_> {
    /// Write object `m` (at trie `node`; `sorted`: every enclosing object's
    /// keys strictly ascend) to `full`, and to `res` without the leaves its
    /// columns rebuild. Returns whether removals left the `res` object
    /// empty (its parent then drops it).
    fn object(&mut self, m: &Map<String, Value>, node: &ColumnTrie, sorted: bool, full: &mut Vec<u8>, res: &mut Vec<u8>) -> bool {
        let sorted = sorted && m.keys().zip(m.keys().skip(1)).all(|(a, b)| a.as_str() < b.as_str());
        full.push(b'{');
        res.push(b'{');
        let mut kept = 0usize;
        let mut removed = false;
        for (i, (k, v)) in m.iter().enumerate() {
            if i > 0 {
                full.push(b',');
            }
            let key_at = full.len();
            serde_json::to_writer(&mut *full, k.as_str()).expect("writing to a Vec");
            full.push(b':');
            let mark = res.len();
            if kept > 0 {
                res.push(b',');
            }
            res.extend_from_slice(&full[key_at..]);
            let child = node.child(k.as_str());
            match (child, v) {
                (Some(c), Value::Object(cm)) if !c.children.is_empty() => {
                    if self.object(cm, c, sorted, full, res) {
                        res.truncate(mark);
                        removed = true;
                    } else {
                        kept += 1;
                    }
                    continue;
                }
                (Some(ColumnTrie { leaf: Some(at), .. }), _) if sorted && self.rebuilds(*at, v) => {
                    serde_json::to_writer(&mut *full, v).expect("writing to a Vec");
                    res.truncate(mark);
                    removed = true;
                    self.taken.push(self.cols.0[*at].0);
                    continue;
                }
                _ => {}
            }
            let at = full.len();
            serde_json::to_writer(&mut *full, v).expect("writing to a Vec");
            res.extend_from_slice(&full[at..]);
            kept += 1;
        }
        full.push(b'}');
        res.push(b'}');
        removed && kept == 0
    }

    /// Whether column field `at` renders exactly `want`.
    fn rebuilds(&self, at: usize, want: &Value) -> bool {
        let (id, rf, kind, _) = &self.cols.0[at];
        // Values too large for a column term are never rebuilt.
        if matches!(kind, ColKind::Keyword) && want.as_str().is_some_and(|x| x.len() > 8191) {
            return false;
        }
        if matches!(kind, ColKind::Keyword) && rf.normalizer.is_none() {
            return self.doc.keyword_renders(*id, want);
        }
        // Leaves are scalars or arrays of scalars: `==` compares them
        // exactly (numbers by representation).
        render_column(&self.s, rf, *kind, &self.doc).is_some_and(|got| got == *want)
    }
}

/// The residual to store for canonical synthetic source `synth` of a
/// document whose indexed fields are `doc`: `synth` without the leaves its
/// columns render identically (each in an object sorted by key, so a fetch
/// puts it back at the same position), with their core ids. `None` when no
/// field is rebuilt from columns (the document then stores `synth` whole).
/// Reference for [`store_forms`].
#[cfg(test)]
pub fn residual_for(mut synth: Value, lk: &FieldLookup, doc: &rustic_core::types::Document) -> Option<String> {
    if lk.may_have_vector_fields || !residuals_enabled() {
        return None;
    }
    let empty = Mappings::default();
    let s = residual_synth(&empty, lk);
    let cols = column_fields(lk);
    let mut taken: Vec<(u16, &[String])> = Vec::new();
    {
        let mut memo: Vec<(usize, bool)> = Vec::new();
        for (id, rf, kind, keys) in cols.0.iter() {
            let Some(want) = get_at(&synth, keys) else { continue };
            // Values too large for a column term are never rebuilt.
            if matches!(kind, ColKind::Keyword) && want.as_str().is_some_and(|x| x.len() > 8191) {
                continue;
            }
            let Some(got) = render_column(&s, rf, *kind, doc) else { continue };
            // Leaves are scalars or arrays of scalars: `==` compares them
            // exactly (numbers by representation).
            if got != *want || !path_sorted(&synth, keys, &mut memo) {
                continue;
            }
            taken.push((*id, keys.as_slice()));
        }
    }
    if taken.is_empty() {
        return None;
    }
    for (_, keys) in &taken {
        take_at(&mut synth, keys);
    }
    let mut out = String::with_capacity(64);
    out.push(RESIDUAL_MARK);
    let mut ids: Vec<u16> = taken.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    write_ids(&mut out, &ids);
    out.push(RESIDUAL_MARK);
    out.push_str(&synth.to_string());
    Some(out)
}

/// The canonical synthetic source of a stored residual `raw`, its
/// column-backed fields rendered from `cols` (`None`: `raw` is not a
/// residual or is malformed).
pub fn rehydrate(raw: &str, lk: &FieldLookup, cols: &dyn DocColumns) -> Option<String> {
    if raw == EMPTY_RESIDUAL {
        return Some(rehydrate_empty(lk, cols, &|_| true));
    }
    let mut buf = String::new();
    let (ids, _, json) = residual_parts_cols(raw, lk, cols, &mut buf)?;
    let v: Value = serde_json::from_str(json).ok()?;
    let cf = column_fields(lk);
    let empty = Mappings::default();
    let s = residual_synth(&empty, lk);
    // Each wanted column's rendered value, by its index in `cf.0`.
    let mut vals: Vec<Option<Value>> = vec![None; cf.0.len()];
    let mut missing = false;
    for id in parse_ids(ids) {
        match cf.0.binary_search_by_key(&id, |(i, ..)| *i) {
            Ok(at) => {
                if vals[at].is_none() {
                    let (_, rf, kind, _) = &cf.0[at];
                    vals[at] = render_column(&s, rf, *kind, cols);
                }
            }
            Err(_) => missing = true,
        }
    }
    // A lookup whose cache predates a field it gained since (never for the
    // lookups the write path builds), or an object the merge cannot place
    // as `put_at` would: the general path.
    if missing || !merge_ok(&v, &cf.1, &vals) {
        return rehydrate_put(ids, v, lk, cols);
    }
    let mut out = Vec::with_capacity(json.len() + 512);
    let Value::Object(root) = &v else { return Some(v.to_string()) };
    write_merged(&mut out, Some(root), &cf.1, &cf.0, &vals);
    String::from_utf8(out).ok()
}

/// Whether the stored source `raw` is a residual whose columns render
/// every column-backed field `at` ([`column_backed`]): each one's id taken
/// out of it (a field the columns would not render exactly, such as
/// duplicate booleans or a keyword over `ignore_above`, stays in the
/// residual, and a document stored whole takes none), no ignored source
/// folded into it.
pub fn residual_takes(raw: &str, lk: &FieldLookup, at: &[usize]) -> bool {
    let Some((ids, None, _)) = residual_parts(raw) else { return false };
    let cf = column_fields(lk);
    let taken: Vec<u16> = parse_ids(ids).collect();
    at.iter().all(|&i| cf.0.get(i).is_some_and(|(id, ..)| taken.contains(id)))
}

/// The column-backed fields among `paths` (a synthetic source renders each
/// from its column), with their source keys, or `None` when some path is
/// not one (it needs the stored residual).
pub fn column_backed<'l>(lk: &'l FieldLookup, paths: &[&str]) -> Option<Vec<usize>> {
    let cf = column_fields(lk);
    paths.iter().map(|p| cf.0.iter().position(|(_, rf, _, _)| rf.name == *p)).collect()
}

/// A synthetic source holding only the column-backed fields `at`
/// ([`column_backed`]) of the document `cols` reads, rendered from their
/// columns (fields without a value left out).
pub fn columns_source(lk: &FieldLookup, at: &[usize], cols: &dyn DocColumns) -> Option<Value> {
    let cf = column_fields(lk);
    let empty = Mappings::default();
    let s = residual_synth(&empty, lk);
    let mut v = Value::Object(Map::new());
    for &i in at {
        let (_, rf, kind, keys) = cf.0.get(i)?;
        if let Some(val) = render_column(&s, rf, *kind, cols) {
            if !put_at(&mut v, keys, val) {
                return None;
            }
        }
    }
    Some(v)
}

/// [`rehydrate`] of only the top-level keys `root` accepts, as a value: the
/// residual's keys among them and the column-backed fields under them (a
/// projection reading a few fields of every document renders those, not
/// the whole document). `None` (the caller rehydrates the whole source):
/// not a residual, `_ignored_source` folded into it, or a column unknown to
/// this lookup.
pub fn rehydrate_roots(raw: &str, lk: &FieldLookup, cols: &dyn DocColumns, root: &dyn Fn(&str) -> bool) -> Option<Value> {
    if raw == EMPTY_RESIDUAL {
        return Some(rehydrate_empty_value(lk, cols, root));
    }
    let mut buf = String::new();
    let (ids, spec, json) = residual_parts_cols(raw, lk, cols, &mut buf)?;
    if spec.is_some() {
        return None;
    }
    let mut v: Value = serde_json::from_str(json).ok()?;
    let Value::Object(m) = &mut v else { return None };
    m.retain(|k, _| root(k));
    let cf = column_fields(lk);
    let empty = Mappings::default();
    let s = residual_synth(&empty, lk);
    for id in parse_ids(ids) {
        let at = cf.0.binary_search_by_key(&id, |(i, ..)| *i).ok()?;
        let (_, rf, kind, keys) = &cf.0[at];
        if !keys.first().is_some_and(|k| root(k)) {
            continue;
        }
        if let Some(val) = render_column(&s, rf, *kind, cols) {
            if !put_at(&mut v, keys, val) {
                return None;
            }
        }
    }
    Some(v)
}

/// Whether every residual object a rendered column goes into has strictly
/// ascending keys (true of every residual the write path stores): merging
/// in key order then places each column as [`put_at`] would.
fn merge_ok(v: &Value, node: &ColumnTrie, vals: &[Option<Value>]) -> bool {
    let Value::Object(m) = v else { return true };
    if !m.keys().zip(m.keys().skip(1)).all(|(a, b)| a.as_str() < b.as_str()) {
        return !node.any(vals);
    }
    for (k, x) in m {
        if let (Some(c), Value::Object(_)) = (node.child(k.as_str()), x) {
            if !merge_ok(x, c, vals) {
                return false;
            }
        }
    }
    true
}

/// Write object `obj` (residual; `None`: a new object) merged with the
/// rendered columns under `node`, keys ascending.
fn write_merged(out: &mut Vec<u8>, obj: Option<&Map<String, Value>>, node: &ColumnTrie, cols: &[(u16, std::sync::Arc<ResolvedField>, ColKind, Vec<String>)], vals: &[Option<Value>]) {
    let write = |out: &mut Vec<u8>, v: &Value| serde_json::to_writer(&mut *out, v).expect("writing to a Vec");
    let key = |out: &mut Vec<u8>, k: &str| {
        serde_json::to_writer(&mut *out, k).expect("writing to a Vec");
        out.push(b':');
    };
    // A trie child's contribution: a leaf value, or a new object.
    let leaf_of = |c: &ColumnTrie| c.leaf.and_then(|i| vals[i].as_ref());
    out.push(b'{');
    let mut first = true;
    let mut sep = |out: &mut Vec<u8>| {
        if !first {
            out.push(b',');
        }
        first = false;
    };
    let mut res = obj.into_iter().flat_map(|m| m.iter()).peekable();
    let mut tri = node.children.iter().map(|(k, c)| (k, c)).filter(|(_, c)| c.any(vals)).peekable();
    loop {
        let take_trie = match (res.peek(), tri.peek()) {
            (None, None) => break,
            (Some(_), None) => false,
            (None, Some(_)) => true,
            (Some((rk, _)), Some((tk, _))) => tk.as_str() < rk.as_str(),
        };
        if take_trie {
            let (k, c) = tri.next().expect("peeked");
            sep(out);
            key(out, k);
            match leaf_of(c) {
                Some(v) => write(out, v),
                None => write_merged(out, None, c, cols, vals),
            }
            continue;
        }
        let (rk, rv) = res.next().expect("peeked");
        sep(out);
        key(out, rk);
        // The same key in the trie: an object the columns go into; a leaf
        // the residual already holds stays (as `put_at` leaves it).
        match (tri.peek(), rv) {
            (Some((tk, c)), Value::Object(m)) if tk.as_str() == rk.as_str() => {
                let c = *c;
                tri.next();
                write_merged(out, Some(m), c, cols, vals);
            }
            (Some((tk, _)), _) if tk.as_str() == rk.as_str() => {
                tri.next();
                write(out, rv);
            }
            _ => write(out, rv),
        }
    }
    out.push(b'}');
}

/// [`rehydrate`] by one [`put_at`] per column (the general path).
fn rehydrate_put(ids: &str, mut v: Value, lk: &FieldLookup, cols: &dyn DocColumns) -> Option<String> {
    let empty = Mappings::default();
    let s = residual_synth(&empty, lk);
    let mut wanted: std::collections::HashSet<u16> = parse_ids(ids).collect();
    for (id, rf, kind, keys) in column_fields(lk).0.iter() {
        if !wanted.remove(id) {
            continue;
        }
        if let Some(val) = render_column(&s, rf, *kind, cols) {
            put_at(&mut v, keys, val);
        }
    }
    for id in wanted {
        if let Some(rf) = lk.fields.values().find(|f| f.core_id == Some(id)) {
            if let Some(val) = col_kind(rf).and_then(|k| render_column(&s, rf, k, cols)) {
                put_at(&mut v, &leaf_keys(lk, &rf.name), val);
            }
        }
    }
    Some(v.to_string())
}

thread_local! {
    /// The indexing request's XContent type for `_ignored_source` objects
    /// and arrays: (XContentDataHelper type byte, encoder). JSON when unset.
    static XCONTENT: std::cell::Cell<Option<(u8, fn(&Value) -> Vec<u8>)>> = const { std::cell::Cell::new(None) };
}

/// Restores the previous [`XCONTENT`] on drop.
pub struct XContentGuard(Option<(u8, fn(&Value) -> Vec<u8>)>);

impl Drop for XContentGuard {
    fn drop(&mut self) {
        XCONTENT.with(|x| x.set(self.0));
    }
}

/// Encode `_ignored_source` objects and arrays of documents indexed on this
/// thread as the request's XContent type (`y` YAML, `c` CBOR, `s` Smile),
/// as ES copies the parser's structure into a builder of the same type.
pub fn ignored_source_xcontent(tag: u8, enc: fn(&Value) -> Vec<u8>) -> XContentGuard {
    XContentGuard(XCONTENT.with(|x| x.replace(Some((tag, enc)))))
}

#[allow(dead_code)]
struct Synth<'a> {
    m: &'a Mappings,
    lk: &'a FieldLookup,
    idx_keep: Keep,
    now: i64,
    /// First error met while rendering (reported by [`synthesize`]).
    err: std::cell::RefCell<Option<crate::error::EsError>>,
    /// Fields the synthetic source could not represent: (full name, parent
    /// offset, value) — rendered into the `_ignored_source` entries.
    ignored: std::cell::RefCell<Vec<(String, usize, Value)>>,
}

impl<'a> Synth<'a> {
    /// Record a field the synthetic source dropped (skipping empty objects).
    /// `parent`: the length of the enclosing object's path plus the dot
    /// (0 at the root), ES `NameValue.parentOffset`.
    fn record_ignored(&self, name: &str, parent: usize, v: &Value) {
        if matches!(v, Value::Object(m) if m.is_empty()) {
            return;
        }
        self.ignored.borrow_mut().push((name.to_string(), parent, v.clone()));
    }

    /// The `_ignored_source` entries in document order (values encoded as
    /// XContentDataHelper), or `None` when nothing was dropped.
    fn ignored_source(&self, src: &Value) -> Option<Vec<IgnoredEntry>> {
        let mut entries = self.ignored.borrow_mut();
        if entries.is_empty() {
            return None;
        }
        if entries.len() > 1 {
            let mut order = std::collections::HashMap::new();
            doc_order(src, "", &mut order);
            entries.sort_by_key(|(name, _, _)| order.get(name.as_str()).copied().unwrap_or(usize::MAX));
        }
        let xc = XCONTENT.with(|x| x.get());
        Some(
            entries
                .iter()
                .map(|(name, parent, v)| {
                    let mut value = Vec::new();
                    ignored_value(&mut value, v, xc);
                    IgnoredEntry { name: name.clone(), parent: *parent, value }
                })
                .collect(),
        )
    }
}

/// A field the synthetic source could not represent: its full name, parent
/// offset and XContentDataHelper-encoded value.
#[derive(Clone, Debug)]
pub struct IgnoredEntry {
    pub name: String,
    pub parent: usize,
    pub value: Vec<u8>,
}

/// How an index stores `_ignored_source` (`IgnoredSourceFieldMapper.ignoredSourceFormat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IgnoredSourceFormat {
    /// One `SingularIgnoredSourceEncoding` value per entry, in document
    /// order (legacy stored fields, and binary doc values from
    /// `IGNORED_SOURCE_AS_DOC_VALUES_NO_FF` with the time series doc values format).
    Singular,
    /// `CoalescedIgnoredSourceEncoding`: one value per field name (from
    /// `IGNORED_SOURCE_COALESCED_ENTRIES`).
    Coalesced,
}

impl IgnoredSourceFormat {
    pub fn of(created: crate::version::IndexVersion, ts_doc_values_format: bool) -> Self {
        use crate::version::IndexVersion;
        if created.on_or_after(IndexVersion::IGNORED_SOURCE_AS_DOC_VALUES_NO_FF) && ts_doc_values_format {
            IgnoredSourceFormat::Singular
        } else if created.on_or_after(IndexVersion::IGNORED_SOURCE_COALESCED_ENTRIES) {
            IgnoredSourceFormat::Coalesced
        } else {
            IgnoredSourceFormat::Singular
        }
    }
}

/// The base64 `_ignored_source` values of `entries` in `format`.
pub fn encode_ignored_source(entries: &[IgnoredEntry], format: IgnoredSourceFormat) -> Vec<String> {
    match format {
        IgnoredSourceFormat::Singular => entries
            .iter()
            .map(|e| {
                let mut buf = Vec::with_capacity(4 + e.name.len() + e.value.len());
                let header = e.name.len() as u32 + ((e.parent as u32) << 16);
                buf.extend_from_slice(&header.to_le_bytes());
                buf.extend_from_slice(e.name.as_bytes());
                buf.extend_from_slice(&e.value);
                base64_encode(&buf)
            })
            .collect(),
        IgnoredSourceFormat::Coalesced => {
            // Grouped by name through `HashMap.computeIfAbsent`: iteration
            // goes bucket by bucket, and within a bucket newest first
            // (`computeIfAbsent` links a new node at the bin's head; a
            // resize keeps the relative order).
            let mut groups: Vec<(&str, Vec<&IgnoredEntry>)> = Vec::new();
            let mut cap = 16usize;
            for e in entries {
                if groups.len() > cap * 3 / 4 {
                    cap *= 2;
                }
                match groups.iter_mut().find(|(n, _)| *n == e.name) {
                    Some((_, g)) => g.push(e),
                    None => groups.push((&e.name, vec![e])),
                }
            }
            let mut keyed: Vec<(usize, std::cmp::Reverse<usize>, &(&str, Vec<&IgnoredEntry>))> = groups.iter().enumerate().map(|(i, g)| (java_bucket(g.0, cap), std::cmp::Reverse(i), g)).collect();
            keyed.sort_by_key(|(b, i, _)| (*b, *i));
            keyed
                .into_iter()
                .map(|(_, _, (name, vals))| {
                    let mut buf = Vec::new();
                    write_vint(&mut buf, vals.len() as u32);
                    write_vint(&mut buf, name.encode_utf16().count() as u32);
                    buf.extend_from_slice(name.as_bytes());
                    for v in vals {
                        write_vint(&mut buf, v.parent as u32);
                        write_vint(&mut buf, v.value.len() as u32);
                        buf.extend_from_slice(&v.value);
                    }
                    base64_encode(&buf)
                })
                .collect()
        }
    }
}

fn write_vint(buf: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        buf.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

/// The bucket of string key `k` in a Java `HashMap` of table size `cap`.
fn java_bucket(k: &str, cap: usize) -> usize {
    let mut h: i32 = 0;
    for c in k.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(c as i32);
    }
    let h = h ^ ((h as u32) >> 16) as i32;
    (h as u32 as usize) & (cap - 1)
}

/// First position of every path (objects and leaves) in `v`, in document order.
fn doc_order(v: &Value, path: &str, out: &mut std::collections::HashMap<String, usize>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                let full = join(path, k);
                let n = out.len();
                out.entry(full.clone()).or_insert(n);
                doc_order(x, &full, out);
            }
        }
        Value::Array(a) => {
            for x in a {
                doc_order(x, path, out);
            }
        }
        _ => {}
    }
}

/// XContentDataHelper value encoding: a type byte, then the content.
fn ignored_value(buf: &mut Vec<u8>, v: &Value, xc: Option<(u8, fn(&Value) -> Vec<u8>)>) {
    match v {
        Value::Object(_) | Value::Array(_) => match xc {
            Some((tag, enc)) => {
                buf.push(tag);
                buf.extend_from_slice(&enc(v));
            }
            None => {
                buf.push(b'j');
                buf.extend_from_slice(v.to_string().as_bytes());
            }
        },
        Value::String(s) => {
            buf.push(b'S');
            buf.extend_from_slice(s.as_bytes());
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                match i32::try_from(i) {
                    Ok(i) => {
                        buf.push(b'I');
                        buf.extend_from_slice(&i.to_le_bytes());
                    }
                    Err(_) => {
                        buf.push(b'L');
                        buf.extend_from_slice(&i.to_le_bytes());
                    }
                }
            } else if let (Some(u), true) = (n.as_u64(), n.is_u64()) {
                // BigInteger: the two's-complement big-endian magnitude.
                buf.push(b'i');
                let mut b = u.to_be_bytes().to_vec();
                b.insert(0, 0);
                while b.len() > 1 && b[0] == 0 && b[1] & 0x80 == 0 {
                    b.remove(0);
                }
                buf.extend_from_slice(&b);
            } else {
                buf.push(b'D');
                buf.extend_from_slice(&n.as_f64().unwrap_or(0.0).to_le_bytes());
            }
        }
        Value::Bool(b) => buf.push(if *b { b't' } else { b'f' }),
        Value::Null => buf.push(b'n'),
    }
}

/// Minimal base64 (standard alphabet, padding) for the `_ignored_source` blob.
fn base64_encode(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

/// `(sort name, output key, value)`.
type Item = (String, String, Value);

fn items_to_map(mut items: Vec<Item>) -> Map<String, Value> {
    items.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Map::new();
    for (_, k, v) in items {
        out.insert(k, v);
    }
    out
}

fn join(path: &str, name: &str) -> String {
    if path.is_empty() {
        name.to_string()
    } else {
        format!("{path}.{name}")
    }
}

/// Values kept verbatim (ES `_ignored_source`): arrays are flattened one
/// level into the list, nulls dropped, a lone value stays scalar.
fn merge_raw(vals: Vec<Value>) -> Option<Value> {
    if vals.len() == 1 {
        return match vals.into_iter().next().unwrap() {
            Value::Null => None,
            Value::Array(a) => Some(Value::Array(a.into_iter().filter(|x| !x.is_null()).collect())),
            x => Some(x),
        };
    }
    let mut out = Vec::new();
    for v in vals {
        match v {
            Value::Null => {}
            Value::Array(a) => out.extend(a.into_iter().filter(|x| !x.is_null())),
            x => out.push(x),
        }
    }
    match out.len() {
        0 => None,
        1 => out.pop(),
        _ => Some(Value::Array(out)),
    }
}

/// Every non-array element of `vals` (arrays flattened recursively).
/// `aggregate_metric_double`: valid objects re-rendered from their metrics
/// (mapping order), then the values the field did not index
/// (`ignore_malformed`) verbatim, arrays flattened.
fn aggregate_metric_like(rf: &ResolvedField, raws: &[Value], sorted_malformed: bool) -> Vec<Value> {
    let metrics: Vec<String> = rf
        .ext
        .as_ref()
        .and_then(|x| x.params.get("metrics").and_then(Value::as_array).map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect()))
        .unwrap_or_default();
    let valid = |o: &Map<String, Value>| -> bool {
        if o.keys().any(|k| !metrics.contains(k)) || metrics.iter().any(|m| !o.contains_key(m)) {
            return false;
        }
        for m in &metrics {
            match o[m].as_f64() {
                Some(n) if n.is_finite() => {
                    if m == "value_count" && (n < 0.0 || n.fract() != 0.0) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        match (o.get("min").and_then(Value::as_f64), o.get("max").and_then(Value::as_f64)) {
            (Some(a), Some(b)) => a <= b,
            _ => true,
        }
    };
    let mut good = Vec::new();
    let mut ign = Vec::new();
    for v in flatten(raws) {
        match v {
            Value::Null => {}
            Value::Object(o) if valid(o) => {
                let mut m = Map::new();
                for k in &metrics {
                    let n = o[k].as_f64().unwrap_or(0.0);
                    m.insert(k.clone(), if k == "value_count" { Value::from(n as i64) } else { f64_value(n) });
                }
                good.push(Value::Object(m));
            }
            other => ign.push(other.clone()),
        }
    }
    // Malformed copies in sorted binary doc values: JSON byte order.
    if sorted_malformed {
        ign.sort_by_cached_key(|v| v.to_string());
    }
    good.into_iter().chain(ign).collect()
}

fn flatten(vals: &[Value]) -> Vec<&Value> {
    fn walk<'v>(v: &'v Value, out: &mut Vec<&'v Value>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            x => out.push(x),
        }
    }
    let mut out = Vec::new();
    vals.iter().for_each(|v| walk(v, &mut out));
    out
}

/// The values of a field: arrays flattened, except where an array *is* the
/// value (`geo_point` `[lon, lat]`, `dense_vector`).
fn items_for<'v>(rf: &ResolvedField, raws: &'v [Value]) -> Vec<&'v Value> {
    match rf.declared {
        EsType::DenseVector => raws.iter().collect(),
        EsType::GeoPoint => {
            fn collect<'v>(v: &'v Value, out: &mut Vec<&'v Value>) {
                match v {
                    Value::Array(a) if crate::values::is_point_array(a) => out.push(v),
                    Value::Array(a) => a.iter().for_each(|x| collect(x, out)),
                    x => out.push(x),
                }
            }
            let mut out = Vec::new();
            raws.iter().for_each(|v| collect(v, &mut out));
            out
        }
        _ => flatten(raws),
    }
}

/// A scalar combined result: nothing, one value, or an array.
fn combine(mut vals: Vec<Value>) -> Option<Value> {
    match vals.len() {
        0 => None,
        1 => vals.pop(),
        _ => Some(Value::Array(vals)),
    }
}

/// Shortest decimal of an f32 as a JSON number (ES prints `float` values
/// with Java's `Float.toString`).
fn f32_value(x: f32) -> Value {
    if !x.is_finite() {
        return Value::Null;
    }
    Number::from_f64(x.to_string().parse::<f64>().unwrap_or(x as f64)).map(Value::Number).unwrap_or(Value::Null)
}

fn f64_value(x: f64) -> Value {
    Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}

enum Resolved {
    Child(String, Option<String>),
    Unmapped,
}

impl<'a> Synth<'a> {
    fn resolve(&self, om: &ObjectMapping, key: &str) -> Resolved {
        if om.properties.contains_key(key) {
            return Resolved::Child(key.to_string(), None);
        }
        if !om.no_subobjects() {
            if let Some((h, r)) = key.split_once('.') {
                if !h.is_empty() && !r.is_empty() && om.properties.contains_key(h) {
                    return Resolved::Child(h.to_string(), Some(r.to_string()));
                }
            }
        }
        Resolved::Unmapped
    }

    fn keep_of(&self, params: &Map<String, Value>) -> Keep {
        params.get("synthetic_source_keep").and_then(Value::as_str).and_then(Keep::parse).unwrap_or(self.idx_keep)
    }

    /// [`Synth::object_items`] of a plain object, read in place: every key
    /// a mapped child of `om` (no dotted or unmapped keys), no arrays, and
    /// child objects plain too (enabled, not nested, not kept verbatim).
    /// Each leaf goes through [`Synth::leaf`] with the same single raw
    /// value, in the same order, so the items are the general walk's.
    /// `None` (nothing kept): not plain, the caller takes the general walk.
    /// A time series document (a hundred mapped scalar leaves) spent most of
    /// its parse in the general walk's copies and grouping.
    fn plain_items(&self, om: &ObjectMapping, path: &mut String, map: &Map<String, Value>) -> Option<Map<String, Value>> {
        if om.no_subobjects() || om.columnar {
            return None;
        }
        let mut keys: Vec<(&String, &Value)> = Vec::with_capacity(map.len());
        for (k, v) in map {
            if v.is_array() || !om.properties.contains_key(k) {
                return None;
            }
            keys.push((k, v));
        }
        // The general walk groups by name in a BTreeMap and orders the
        // items by full path: within one object, the names' order.
        keys.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut out = Map::with_capacity(keys.len());
        let base = path.len();
        for (name, v) in keys {
            if base > 0 {
                path.push('.');
            }
            path.push_str(name);
            let value = match om.properties.get(name.as_str())? {
                Mapper::Field(fm) => self.leaf(fm, path, std::slice::from_ref(v), false),
                Mapper::Object(child) => {
                    let Value::Object(inner) = v else { return None };
                    if child.enabled == Some(false) || child.nested || self.keep_of(&child.params) == Keep::All {
                        return None;
                    }
                    let map = self.plain_items(child, path, inner)?;
                    (!map.is_empty()).then_some(Value::Object(map))
                }
            };
            path.truncate(base);
            if let Some(value) = value {
                out.insert(name.clone(), value);
            }
        }
        Some(out)
    }

    /// [`Synth::plain_items`] written straight to the stored forms, as
    /// [`ResidualWalk::object`] writes the canonical object it builds: the
    /// canonical text to `full`, and to `res` without the leaves their
    /// columns rebuild (trie `node`; `None`: none below). Returns the
    /// object's item count (no items: the caller drops its key, as the
    /// canonical form omits an empty object) and whether removals emptied
    /// its residual object. `None`: not plain.
    #[allow(clippy::too_many_arguments)]
    fn plain_forms(&self, om: &ObjectMapping, path: &mut String, map: &Map<String, Value>, node: Option<&ColumnTrie>, w: &mut ResidualWalk<'_>,
        full: &mut Vec<u8>, res: &mut Vec<u8>) -> Option<(usize, bool)> {
        if om.no_subobjects() || om.columnar {
            return None;
        }
        // Each key's mapper, looked up once.
        let mut keys: Vec<(&String, &Value, &Mapper)> = Vec::with_capacity(map.len());
        for (k, v) in map {
            if v.is_array() {
                return None;
            }
            let Some(mapper) = om.properties.get(k.as_str()) else { return None };
            keys.push((k, v, mapper));
        }
        keys.sort_unstable_by(|a, b| a.0.cmp(b.0));
        full.push(b'{');
        res.push(b'{');
        let (mut items, mut kept, mut removed) = (0usize, 0usize, false);
        let base = path.len();
        for (name, v, mapper) in keys {
            if base > 0 {
                path.push('.');
            }
            path.push_str(name);
            let (full_mark, res_mark) = (full.len(), res.len());
            if items > 0 {
                full.push(b',');
            }
            let key_at = full.len();
            serde_json::to_writer(&mut *full, name.as_str()).expect("writing to a Vec");
            full.push(b':');
            if kept > 0 {
                res.push(b',');
            }
            res.extend_from_slice(&full[key_at..]);
            let child = node.and_then(|n| n.child(name.as_str()));
            match mapper {
                Mapper::Field(fm) => match self.leaf(fm, path, std::slice::from_ref(v), false) {
                    None => {
                        full.truncate(full_mark);
                        res.truncate(res_mark);
                    }
                    // An object value (a leaf the residual walk would
                    // descend into): the general path.
                    Some(Value::Object(_)) if child.is_some_and(|c| !c.children.is_empty()) => return None,
                    Some(value) => {
                        items += 1;
                        match child {
                            Some(ColumnTrie { leaf: Some(at), .. }) if w.rebuilds(*at, &value) => {
                                serde_json::to_writer(&mut *full, &value).expect("writing to a Vec");
                                res.truncate(res_mark);
                                removed = true;
                                w.taken.push(w.cols.0[*at].0);
                            }
                            _ => {
                                let at = full.len();
                                serde_json::to_writer(&mut *full, &value).expect("writing to a Vec");
                                res.extend_from_slice(&full[at..]);
                                kept += 1;
                            }
                        }
                    }
                },
                Mapper::Object(sub) => {
                    let Value::Object(inner) = v else { return None };
                    if sub.enabled == Some(false) || sub.nested || self.keep_of(&sub.params) == Keep::All {
                        return None;
                    }
                    // The residual walk descends where the trie continues,
                    // and copies the object whole elsewhere (no removals).
                    let below = child.filter(|c| !c.children.is_empty());
                    if child.is_some_and(|c| c.children.is_empty() && c.leaf.is_some()) {
                        return None;
                    }
                    let (n, dropped) = self.plain_forms(sub, path, inner, below, w, full, res)?;
                    if n == 0 {
                        full.truncate(full_mark);
                        res.truncate(res_mark);
                    } else if dropped {
                        items += 1;
                        res.truncate(res_mark);
                        removed = true;
                    } else {
                        items += 1;
                        kept += 1;
                    }
                }
            }
            path.truncate(base);
        }
        full.push(b'}');
        res.push(b'}');
        Some((items, removed && kept == 0))
    }

    /// Output items of the object `om` (full path `path`) built from `entries`
    /// (source `key -> value` pairs, in document order).
    fn object_items(&self, om: &ObjectMapping, path: &str, entries: Vec<(String, Value)>, in_arr: bool) -> Vec<Item> {
        let entries = if om.no_subobjects() { flatten_dotted(om, entries) } else { entries };
        let mut mapped: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        let mut unmapped: Vec<(String, Vec<Value>)> = Vec::new();
        let mut add_unmapped = |k: String, v: Value| match unmapped.iter_mut().find(|(n, _)| *n == k) {
            Some((_, vs)) => vs.push(v),
            None => unmapped.push((k, vec![v])),
        };
        for (k, v) in entries {
            match self.resolve(om, &k) {
                Resolved::Child(name, None) => mapped.entry(name).or_default().push(v),
                Resolved::Child(name, Some(rest)) => {
                    if matches!(om.properties.get(&name), Some(Mapper::Object(_))) {
                        let mut m = Map::new();
                        m.insert(rest, v);
                        mapped.entry(name).or_default().push(Value::Object(m));
                    } else {
                        add_unmapped(k, v);
                    }
                }
                Resolved::Unmapped => add_unmapped(k, v),
            }
        }
        let mut items: Vec<Item> = Vec::new();
        for (name, vals) in mapped {
            let full = join(path, &name);
            let v = match om.properties.get(&name) {
                Some(Mapper::Field(fm)) => self.leaf(fm, &full, &vals, in_arr),
                Some(Mapper::Object(child)) => self.child_object(child, &full, vals, in_arr),
                None => None,
            };
            if let Some(v) = v {
                items.push((full, name, v));
            }
        }
        // Unmapped keys: a dotted key is expanded one level (`a.b.c` -> `a: {"b.c": v}`).
        let mut grouped: Vec<(String, Map<String, Value>)> = Vec::new();
        // A columnar root drops unmapped (`dynamic: false`, disabled) fields
        // instead of keeping them as ignored source.
        if om.columnar {
            unmapped.clear();
        }
        for (k, vals) in unmapped {
            let Some(v) = merge_raw(vals) else { continue };
            self.record_ignored(&join(path, &k), if path.is_empty() { 0 } else { path.len() + 1 }, &v);
            match k.split_once('.') {
                Some((h, r)) if !h.is_empty() && !r.is_empty() && !om.no_subobjects() => match grouped.iter_mut().find(|(n, _)| n == h) {
                    Some((_, m)) => {
                        m.insert(r.to_string(), v);
                    }
                    None => {
                        let mut m = Map::new();
                        m.insert(r.to_string(), v);
                        grouped.push((h.to_string(), m));
                    }
                },
                _ => items.push((join(path, &k), k, v)),
            }
        }
        for (h, m) in grouped {
            items.push((join(path, &h), h, Value::Object(m)));
        }
        items
    }

    fn child_object(&self, child: &ObjectMapping, full: &str, vals: Vec<Value>, in_arr: bool) -> Option<Value> {
        if child.enabled == Some(false) {
            let v = merge_raw(vals);
            if let Some(v) = &v {
                self.record_ignored(full, full.rfind('.').map_or(0, |i| i + 1), v);
            }
            return v;
        }
        // Inside an array (this value's own, or an enclosing one) the
        // `arrays` mode keeps the source as written.
        let arr_here = in_arr || vals.iter().any(Value::is_array);
        let keep = self.keep_of(&child.params);
        // Columnar modes keep leaf array order only: objects are rebuilt.
        if keep == Keep::All || (keep == Keep::Arrays && arr_here && !self.m.root.columnar) {
            return merge_raw(vals);
        }
        // The objects are consumed (moved, not cloned): each level of a
        // document was copied again before.
        fn take_objects(v: Value, out: &mut Vec<Map<String, Value>>) {
            match v {
                Value::Array(a) => a.into_iter().for_each(|x| take_objects(x, out)),
                Value::Object(m) => out.push(m),
                _ => {}
            }
        }
        let mut objs: Vec<Map<String, Value>> = Vec::new();
        vals.into_iter().for_each(|v| take_objects(v, &mut objs));
        if child.nested {
            let mut out: Vec<Value> = objs.into_iter().map(|m| Value::Object(items_to_map(self.object_items(child, full, m.into_iter().collect(), arr_here)))).collect();
            return match out.len() {
                0 => None,
                1 => out.pop(),
                _ => Some(Value::Array(out)),
            };
        }
        let mut entries = Vec::new();
        for m in objs {
            entries.extend(m);
        }
        let map = items_to_map(self.object_items(child, full, entries, arr_here));
        if map.is_empty() {
            None
        } else {
            Some(Value::Object(map))
        }
    }

    /// One mapped leaf field.
    /// (`raws` borrowed: the plain walk passes the source's own value.)
    fn leaf(&self, fm: &FieldMapping, full: &str, raws: &[Value], in_arr: bool) -> Option<Value> {
        let keep = self.keep_of(&fm.params);
        // Kept source (ignored source) is the value exactly as written.
        let verbatim = |raws: &[Value]| -> Option<Value> {
            match raws {
                [v @ Value::Array(_)] => Some(v.clone()),
                _ => None,
            }
        };
        if keep == Keep::All {
            return verbatim(raws).or_else(|| merge_raw(raws.to_vec()));
        }
        let arr_src = raws.iter().any(Value::is_array);
        let Some(rf) = self.lk.fields.get(full) else { return merge_raw(raws.to_vec()) };
        // Fields feeding `copy_to` keep their source as written.
        if !rf.copy_to.is_empty() {
            if flatten(&raws).iter().any(|v| v.is_object()) && !matches!(rf.declared, EsType::GeoPoint | EsType::Flattened) {
                let mut e = self.err.borrow_mut();
                if e.is_none() {
                    *e = Some(crate::error::EsError::document_parsing(format!(
                        "[1:1] failed to parse field [{full}] of type [{}]: Copy-to currently only works for value-type fields, not object fields",
                        rf.declared.name()
                    )));
                }
            }
            return self.fallback(&rf, &raws);
        }
        // `arrays` mode (inside an array) and fields only kept as stored
        // values: document order, no sorting or de-duplication.
        let ordered = (keep == Keep::Arrays && (in_arr || arr_src)) || (!rf.doc_values && rf.stored);
        match self.canon(&rf, &raws, !ordered) {
            // Types without native array (offset) support keep the array
            // source as written (ignored source); the others rebuild it.
            Some(vals) if ordered && keep == Keep::Arrays && arr_src => {
                if rf.declared == EsType::CountedKeyword {
                    verbatim(&raws).or(Some(Value::Array(vals)))
                } else {
                    Some(Value::Array(vals))
                }
            }
            Some(vals) => combine(vals),
            None => self.fallback(&rf, &raws),
        }
    }

    /// Values of a field type with no synthetic support (or one the mapping
    /// configures out of it): kept as given (text-like fields as strings).
    fn fallback(&self, rf: &ResolvedField, raws: &[Value]) -> Option<Value> {
        let textual = matches!(rf.declared, EsType::Text | EsType::MatchOnlyText | EsType::AnnotatedText | EsType::SearchAsYouType);
        if textual {
            let vals: Vec<Value> = flatten(raws)
                .into_iter()
                .filter(|v| !v.is_null())
                .map(|v| match v {
                    Value::Number(n) => Value::String(n.to_string()),
                    Value::Bool(b) => Value::String(b.to_string()),
                    x => x.clone(),
                })
                .collect();
            let v = combine(vals);
            if let Some(v) = &v {
                self.record_ignored(&rf.name, rf.name.rfind('.').map_or(0, |i| i + 1), v);
            }
            return v;
        }
        let v = merge_raw(raws.to_vec());
        if let Some(v) = &v {
            self.record_ignored(&rf.name, rf.name.rfind('.').map_or(0, |i| i + 1), v);
        }
        v
    }

    /// Malformed values (`ignore_malformed`) in output order: document order
    /// for indices created before `STORE_IGNORED_MALFORMED_IN_BINARY_DOC_VALUES`,
    /// else sorted by their `XContentDataHelper` encoding (type byte, then
    /// the little-endian / UTF-8 value bytes), duplicates kept.
    fn malformed_order(&self, mut ign: Vec<Value>) -> Vec<Value> {
        if ign.len() > 1 && self.lk.registry.malformed_sorted {
            ign.sort_by_cached_key(encoded_value);
        }
        ign
    }

    /// `Some(values in output order)`, or `None` when the field falls back
    /// to verbatim values.
    fn canon(&self, rf: &ResolvedField, raws: &[Value], sorted: bool) -> Option<Vec<Value>> {
        if let Some(out) = Self::canon_single(rf, raws) {
            #[cfg(test)]
            assert_eq!(out, self.canon_general(rf, raws, sorted), "single-value canon of {}", rf.name);
            return out;
        }
        self.canon_general(rf, raws, sorted)
    }

    /// [`Synth::canon`] of one value (a time series document's leaves),
    /// without its intermediate lists: what the general branch gives for a
    /// single non-null, non-array value of a long- or float-family field, a
    /// plain keyword, a boolean, a date or an ip (checked against it in
    /// tests). `None`: another case (a malformed value among them).
    #[inline]
    fn canon_single(rf: &ResolvedField, raws: &[Value]) -> Option<Option<Vec<Value>>> {
        let [v] = raws else { return None };
        if v.is_array() || v.is_null() {
            return None;
        }
        match rf.declared {
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => {
                if !rf.doc_values && !rf.stored {
                    return Some(None);
                }
                let (min, max): (i64, i64) = match rf.ty {
                    EsType::Byte => (i8::MIN as i64, i8::MAX as i64),
                    EsType::Short => (i16::MIN as i64, i16::MAX as i64),
                    EsType::Integer => (i32::MIN as i64, i32::MAX as i64),
                    _ => (i64::MIN, i64::MAX),
                };
                Some(Some(vec![match crate::document::parse_integer(v, rf.coerce, min, max, rf.ty) {
                    Ok(n) => Value::from(n),
                    Err(_) => v.clone(),
                }]))
            }
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => {
                if !rf.doc_values && !rf.stored {
                    return Some(None);
                }
                let r = match v {
                    Value::Number(n) => n.as_f64(),
                    Value::String(s) if rf.coerce => s.trim().parse::<f64>().ok(),
                    _ => None,
                };
                Some(Some(vec![match r {
                    Some(x) if x.is_finite() => {
                        let x = match rf.ty {
                            EsType::Float => crate::values::to_float(x),
                            EsType::HalfFloat => crate::values::to_half_float(x),
                            EsType::ScaledFloat => numeric::scaled_round_trip(x, rf.scaling_factor),
                            _ => x,
                        };
                        match rf.ty {
                            EsType::Float | EsType::HalfFloat => f32_value(x as f32),
                            _ => f64_value(x),
                        }
                    }
                    _ => v.clone(),
                }]))
            }
            // A plain keyword's one value, as a string (what `keyword_like`
            // keeps of it, under or over `ignore_above` alike).
            EsType::Keyword if rf.normalizer.is_none() => {
                if !rf.doc_values && !rf.stored {
                    return Some(None);
                }
                Some(Some(vec![match v {
                    Value::String(s) => Value::String(s.clone()),
                    Value::Number(n) => Value::String(n.to_string()),
                    Value::Bool(b) => Value::String(b.to_string()),
                    _ => return None,
                }]))
            }
            EsType::Boolean => {
                if !rf.doc_values && !rf.stored {
                    return Some(None);
                }
                let b = match v {
                    Value::Bool(b) => *b,
                    Value::String(s) if s == "true" => true,
                    Value::String(s) if s == "false" || s.is_empty() => false,
                    _ => return None,
                };
                Some(Some(vec![Value::Bool(b)]))
            }
            EsType::Date => {
                if !rf.doc_values && !rf.stored {
                    return Some(None);
                }
                let ms = match v {
                    Value::String(s) => rf.date_format.parse(s).ok()?,
                    Value::Number(n) => rf.date_format.parse_number(n.as_f64().unwrap_or(0.0)).ok()?,
                    _ => return None,
                };
                Some(Some(vec![Value::String(rf.date_format.format(ms, crate::date::Tz::UTC))]))
            }
            EsType::Ip => {
                if !rf.doc_values && !rf.stored {
                    return Some(None);
                }
                let b = v.as_str().and_then(crate::values::parse_ip)?;
                Some(Some(vec![Value::String(crate::values::ip_display_bytes(b))]))
            }
            _ => None,
        }
    }

    fn canon_general(&self, rf: &ResolvedField, raws: &[Value], sorted: bool) -> Option<Vec<Value>> {
        let items = items_for(rf, raws);
        // `null` entries become the field's `null_value`, or vanish.
        let with_nulls = |items: &[&Value]| -> Vec<Value> {
            items
                .iter()
                .filter_map(|v| match v {
                    Value::Null => rf.null_value.clone(),
                    x => Some((*x).clone()),
                })
                .collect()
        };
        match rf.declared {
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Version => {
                if rf.declared == EsType::Keyword && ((!rf.doc_values && !rf.stored) || (rf.normalizer.is_some() && !rf.normalizer_skip_store)) {
                    return None;
                }
                if rf.normalizer.is_some() && !rf.stored {
                    return Some(self.keyword_like_norm(rf, &with_nulls(&items), rf.ignore_above, sorted));
                }
                Some(self.keyword_like(rf, &with_nulls(&items), rf.ignore_above, rf.declared == EsType::Version, sorted))
            }
            // Native: every value (duplicates included), sorted.
            EsType::CountedKeyword => {
                let mut vals: Vec<String> = items.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
                if sorted {
                    vals.sort();
                }
                Some(vals.into_iter().map(Value::String).collect())
            }
            // `semantic`: its original inputs in document order
            // (`SemanticOriginalValueEncoder`).
            EsType::Semantic => Some(flatten(raws).into_iter().filter(|v| !v.is_null()).map(crate::inference::semantic::original_value).collect()),
            EsType::Text | EsType::MatchOnlyText => {
                if rf.stored {
                    let vals = with_nulls(&items)
                        .into_iter()
                        .filter_map(|v| match v {
                            Value::String(s) => Some(Value::String(s)),
                            Value::Number(n) => Some(Value::String(n.to_string())),
                            Value::Bool(b) => Some(Value::String(b.to_string())),
                            _ => None,
                        })
                        .collect();
                    return Some(vals);
                }
                let sub = rf.multi_fields.iter().filter_map(|n| self.lk.fields.get(n)).find(|s| s.declared == EsType::Keyword && s.doc_values && (s.normalizer.is_none() || s.normalizer_skip_store))?;
                let vals = with_nulls(&items);
                if sorted && self.lk.registry.created_version().on_or_after(crate::version::IndexVersion::KEYWORD_MULTI_FIELDS_NOT_STORED_WHEN_IGNORED) {
                    // The text field keeps the values its keyword multi-field
                    // ignored in its own fallback field, loaded first
                    // (`TextFieldMapper.syntheticFieldLoader`).
                    let too_long = |v: &Value| v.as_str().is_some_and(|s| sub.ignore_above.is_some_and(|ia| s.chars().count() > ia));
                    let (ign, kept): (Vec<Value>, Vec<Value>) = vals.into_iter().partition(|v| too_long(v));
                    let mut out = ign;
                    out.extend(self.keyword_like_norm(sub, &kept, sub.ignore_above, sorted));
                    return Some(out);
                }
                Some(self.keyword_like_norm(sub, &vals, sub.ignore_above, sorted))
            }
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let (min, max): (i64, i64) = match rf.ty {
                    EsType::Byte => (i8::MIN as i64, i8::MAX as i64),
                    EsType::Short => (i16::MIN as i64, i16::MAX as i64),
                    EsType::Integer => (i32::MIN as i64, i32::MAX as i64),
                    _ => (i64::MIN, i64::MAX),
                };
                let mut dv: Vec<i64> = Vec::new();
                let mut ign: Vec<Value> = Vec::new();
                for v in with_nulls(&items) {
                    match crate::document::parse_integer(&v, rf.coerce, min, max, rf.ty) {
                        Ok(n) => dv.push(n),
                        Err(_) => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_unstable();
                }
                Some(dv.into_iter().map(Value::from).chain(self.malformed_order(ign)).collect())
            }
            EsType::UnsignedLong => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let mut dv: Vec<u64> = Vec::new();
                let mut ign = Vec::new();
                for v in with_nulls(&items) {
                    let r = match &v {
                        Value::Number(n) => n.as_u64(),
                        Value::String(s) if rf.coerce => s.trim().parse::<u64>().ok(),
                        _ => None,
                    };
                    match r {
                        Some(n) => dv.push(n),
                        None => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_unstable();
                }
                Some(dv.into_iter().map(Value::from).chain(self.malformed_order(ign)).collect())
            }
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let mut dv: Vec<f64> = Vec::new();
                let mut ign = Vec::new();
                for v in with_nulls(&items) {
                    let r = match &v {
                        Value::Number(n) => n.as_f64(),
                        Value::String(s) if rf.coerce => s.trim().parse::<f64>().ok(),
                        _ => None,
                    };
                    match r {
                        Some(x) if x.is_finite() => dv.push(match rf.ty {
                            EsType::Float => crate::values::to_float(x),
                            EsType::HalfFloat => crate::values::to_half_float(x),
                            EsType::ScaledFloat => numeric::scaled_round_trip(x, rf.scaling_factor),
                            _ => x,
                        }),
                        _ => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_by(|a, b| a.total_cmp(b));
                }
                let ty = rf.ty;
                Some(
                    dv.into_iter()
                        .map(|x| match ty {
                            EsType::Float | EsType::HalfFloat => f32_value(x as f32),
                            _ => f64_value(x),
                        })
                        .chain(self.malformed_order(ign))
                        .collect(),
                )
            }
            EsType::Boolean => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let mut dv: Vec<bool> = Vec::new();
                let mut ign = Vec::new();
                for v in with_nulls(&items) {
                    let b = match &v {
                        Value::Bool(b) => Some(*b),
                        Value::String(s) => match s.as_str() {
                            "true" => Some(true),
                            "false" | "" => Some(false),
                            _ => None,
                        },
                        _ => None,
                    };
                    match b {
                        Some(b) => dv.push(b),
                        None => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_unstable();
                }
                Some(dv.into_iter().map(Value::Bool).chain(self.malformed_order(ign)).collect())
            }
            EsType::Date => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let mut dv: Vec<i64> = Vec::new();
                let mut ign = Vec::new();
                for v in with_nulls(&items) {
                    let r = match &v {
                        Value::String(s) => rf.date_format.parse(s).ok(),
                        Value::Number(n) => rf.date_format.parse_number(n.as_f64().unwrap_or(0.0)).ok(),
                        _ => None,
                    };
                    match r {
                        Some(ms) => dv.push(ms),
                        None => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_unstable();
                }
                Some(dv.into_iter().map(|ms| Value::String(rf.date_format.format(ms, crate::date::Tz::UTC))).chain(self.malformed_order(ign)).collect())
            }
            EsType::DateNanos => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let mut dv: Vec<i64> = Vec::new();
                let mut ign = Vec::new();
                for v in with_nulls(&items) {
                    match crate::fieldtypes::index::parse_nanos(rf, &v) {
                        Ok(n) => dv.push(n),
                        Err(_) => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_unstable();
                }
                Some(dv.into_iter().map(|n| Value::String(numeric::format_nanos(n))).chain(self.malformed_order(ign)).collect())
            }
            EsType::Ip => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let mut dv: Vec<[u8; 16]> = Vec::new();
                let mut ign = Vec::new();
                for v in with_nulls(&items) {
                    match v.as_str().and_then(crate::values::parse_ip) {
                        Some(b) => dv.push(b),
                        None => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_unstable();
                    dv.dedup();
                }
                Some(dv.into_iter().map(|b| Value::String(crate::values::ip_display_bytes(b))).chain(self.malformed_order(ign)).collect())
            }
            EsType::GeoPoint => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let mut pts: Vec<(i32, i32)> = Vec::new();
                let mut ign = Vec::new();
                let vals = items;
                for v in vals {
                    if v.is_null() {
                        continue;
                    }
                    match crate::values::parse_geo_point(v) {
                        Some((lat, lon)) if (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) => pts.push((encode_lat(lat), encode_lon(lon))),
                        _ => ign.push(v.clone()),
                    }
                }
                if sorted {
                    pts.sort_by_key(|&(a, o)| (a, o as u32));
                }
                Some(
                    pts.into_iter()
                        .map(|(a, o)| {
                            let mut m = Map::new();
                            m.insert("lat".into(), f64_value(decode_lat(a)));
                            m.insert("lon".into(), f64_value(decode_lon(o)));
                            Value::Object(m)
                        })
                        .chain(self.malformed_order(ign))
                        .collect(),
                )
            }
            EsType::IntegerRange | EsType::LongRange | EsType::FloatRange | EsType::DoubleRange | EsType::DateRange | EsType::IpRange => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let kind = rng::RangeKind::parse(rf.declared.name())?;
                let mut dv: Vec<(u128, u128)> = Vec::new();
                let mut ign = Vec::new();
                for v in with_nulls(&items) {
                    match crate::fieldtypes::index::parse_range_value(rf, kind, &v, self.now) {
                        Ok(r) => dv.push(r),
                        Err(_) => ign.push(v),
                    }
                }
                if sorted {
                    dv.sort_unstable();
                    dv.dedup();
                }
                Some(dv.into_iter().map(|(lo, hi)| range_value(rf, kind, lo, hi)).chain(self.malformed_order(ign)).collect())
            }
            EsType::Flattened => {
                if !rf.doc_values && !rf.stored {
                    return None;
                }
                let depth = rf.ext.as_ref().and_then(|x| x.u64("depth_limit")).unwrap_or(20) as usize;
                let nv = rf.null_value.as_ref().and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Null => None,
                    o => Some(o.to_string()),
                });
                let mut keyed: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
                for v in items.iter() {
                    if !v.is_object() {
                        continue;
                    }
                    let leaves = flattened::flatten(v, depth, nv.as_deref()).ok()?;
                    for (k, val) in leaves {
                        keyed.entry(k).or_default().insert(val);
                    }
                }
                // Mapped sub-fields come back under their dotted names.
                let mut mapped: Vec<(String, Value)> = Vec::new();
                if let Some(props) = rf.ext.as_ref().and_then(|e| e.params.get("properties")).and_then(Value::as_object) {
                    for (n, spec) in props {
                        keyed.remove(n);
                        let numeric = spec.get("type").and_then(Value::as_str).and_then(EsType::parse).is_some_and(|t| t.is_numeric());
                        let mut vals: Vec<Value> = items
                            .iter()
                            .flat_map(|it| crate::highlight::source_values(it, n).into_iter().cloned().collect::<Vec<_>>())
                            .filter(|v| !v.is_null())
                            .map(|v| match (&v, numeric) {
                                (Value::String(x), true) => x.parse::<f64>().ok().map(|f| if f.fract() == 0.0 { json!(f as i64) } else { json!(f) }).unwrap_or(v),
                                (Value::Number(_), false) => Value::String(v.to_string()),
                                _ => v,
                            })
                            .collect();
                        match vals.len() {
                            0 => {}
                            1 => mapped.push((n.clone(), vals.pop().unwrap())),
                            _ => mapped.push((n.clone(), Value::Array(vals))),
                        }
                    }
                }
                if keyed.is_empty() && mapped.is_empty() {
                    return Some(Vec::new());
                }
                let mut tree = if keyed.is_empty() { json!({}) } else { flattened_tree(keyed) };
                if let Some(o) = tree.as_object_mut() {
                    for (n, v) in mapped {
                        o.insert(n, v);
                    }
                }
                Some(vec![tree])
            }
            EsType::DenseVector => {
                let et = rf.ext.as_ref().and_then(|e| e.params.get("element_type")).and_then(Value::as_str).unwrap_or("float").to_string();
                let mut out = Vec::new();
                for v in raws {
                    let Value::Array(a) = v else {
                        out.push(v.clone());
                        continue;
                    };
                    let arr: Vec<Value> = a
                        .iter()
                        .map(|x| match (et.as_str(), x.as_f64()) {
                            // Java `Float.toString` (round-half-even on
                            // shortest-digit ties: 0.369140625 is
                            // `0.36914062`), as ES renders vector values.
                            ("float", Some(f)) => crate::values::java_float(f as f32),
                            _ => x.clone(),
                        })
                        .collect();
                    out.push(Value::Array(arr));
                }
                // A vector is one value (an array); never merged into a list.
                Some(out)
            }
            EsType::Histogram | EsType::TDigest => Some(self.histogram_like(rf, &raws)),
            // The doc value re-serialized (`ExponentialHistogramXContent`);
            // malformed values verbatim.
            EsType::ExponentialHistogram => {
                let (mut good, mut bad) = (Vec::new(), Vec::new());
                for v in flatten(&raws).into_iter().filter(|v| !v.is_null()) {
                    match crate::fieldtypes::exphist::parse(&rf.name, v, rf.coerce) {
                        Ok(h) => good.push(crate::fieldtypes::exphist::render(&h)),
                        Err(_) => bad.push(v.clone()),
                    }
                }
                if self.lk.registry.malformed_sorted {
                    bad.sort_by_cached_key(|v| v.to_string());
                }
                Some(good.into_iter().chain(bad).collect())
            }
            EsType::AggregateMetricDouble => Some(aggregate_metric_like(rf, &raws, self.lk.registry.malformed_sorted)),
            _ => None,
        }
    }

    /// `histogram`: reconstruct `{values, counts}` from the source, dropping
    /// zero-count entries (the doc value only stores non-zero pairs). Valid
    /// histogram objects are re-emitted (extra keys preserved); values the
    /// field did not index (`ignore_malformed`) are kept verbatim.
    fn histogram_like(&self, rf: &ResolvedField, raws: &[Value]) -> Vec<Value> {
        let mut valid: Vec<Value> = Vec::new();
        let mut ign: Vec<Value> = Vec::new();
        for v in flatten(raws) {
            if v.is_null() {
                continue;
            }
            if rf.declared == EsType::TDigest {
                // `TDigestSyntheticFieldLoader`: summary, then the non-empty
                // centroids.
                match crate::fieldtypes::index::parse_tdigest(v) {
                    Ok(t) => {
                        let mut m = Map::new();
                        if let Some((mn, mx, sum)) = t.summary() {
                            m.insert("min".into(), f64_value(mn));
                            m.insert("max".into(), f64_value(mx));
                            m.insert("sum".into(), f64_value(sum));
                        }
                        let kept: Vec<(f64, i64)> = t.centroids.iter().zip(&t.counts).filter(|(_, c)| **c != 0).map(|(x, c)| (*x, *c)).collect();
                        m.insert("centroids".into(), Value::Array(kept.iter().map(|(x, _)| f64_value(*x)).collect()));
                        m.insert("counts".into(), Value::Array(kept.iter().map(|(_, c)| Value::from(*c)).collect()));
                        valid.push(Value::Object(m));
                    }
                    Err(_) => ign.push(v.clone()),
                }
                continue;
            }
            match crate::fieldtypes::index::parse_histogram(rf.declared, v) {
                Ok((vs, cs)) => {
                    let mut m = Map::new();
                    let vals: Vec<Value> = vs.iter().zip(cs.iter()).filter(|(_, c)| **c != 0).map(|(x, _)| f64_value(*x)).collect();
                    let cnts: Vec<Value> = vs.iter().zip(cs.iter()).filter(|(_, c)| **c != 0).map(|(_, c)| Value::from(*c)).collect();
                    m.insert("values".into(), Value::Array(vals));
                    m.insert("counts".into(), Value::Array(cnts));
                    // Preserve any extra keys the source carried (e.g. "hello").
                    if let Some(o) = v.as_object() {
                        for (k, val) in o {
                            if k != "values" && k != "counts" {
                                m.insert(k.clone(), val.clone());
                            }
                        }
                    }
                    valid.push(Value::Object(m));
                }
                Err(_) => ign.push(v.clone()),
            }
        }
        // Indices storing malformed values in sorted binary doc values emit
        // them ordered by their JSON encoding
        // (`IgnoreMalformedStoredValues.encodeBuilderAsJson`); older ones
        // in document order.
        if self.lk.registry.malformed_sorted {
            let mut keyed: Vec<(String, Value)> = ign.into_iter().map(|v| (v.to_string(), v)).collect();
            keyed.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            ign = keyed.into_iter().map(|(_, v)| v).collect();
        }
        // Malformed values are kept verbatim; arrays are flattened (ES's
        // MalformedValuesLayer re-emits the raw tokens).
        let ign: Vec<Value> = flatten(&ign).into_iter().filter(|x| !x.is_null()).cloned().collect();
        // Valid histograms first (doc-value order: descending by first value),
        // then the values the field did not index, in document order.
        valid.sort_by(|a, b| {
            let fa = a.get("values").and_then(Value::as_array).and_then(|x| x.first()).and_then(Value::as_f64).unwrap_or(f64::MIN);
            let fb = b.get("values").and_then(Value::as_array).and_then(|x| x.first()).and_then(Value::as_f64).unwrap_or(f64::MIN);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        valid.into_iter().chain(ign).collect()
    }

    /// keyword / wildcard / version / constant_keyword: sorted unique values,
    /// then values the field did not index (`ignore_above`) in document order.
    /// [`Self::keyword_like`] for a keyword whose doc values hold the
    /// normalized value: values within `ignore_above` come back normalized,
    /// the ignored ones verbatim.
    fn keyword_like_norm(&self, rf: &ResolvedField, vals: &[Value], ignore_above: Option<usize>, sorted: bool) -> Vec<Value> {
        let vals: Vec<Value> = vals
            .iter()
            .map(|v| match v {
                Value::String(s) if !ignore_above.is_some_and(|ia| s.chars().count() > ia) => normalized(rf, std::slice::from_ref(v)).pop().unwrap_or_else(|| v.clone()),
                other => other.clone(),
            })
            .collect();
        self.keyword_like(rf, &vals, ignore_above, false, sorted)
    }

    /// Whether ignored keyword values live in binary doc values.
    fn ignored_keywords_in_binary_dv(&self) -> bool {
        use crate::version::IndexVersion;
        let created = self.lk.registry.created_version();
        if created.on_or_after(IndexVersion::STORE_IGNORED_WILDCARD_FIELDS_IN_BINARY_DOC_VALUES) {
            self.lk.registry.ts_doc_values_format
        } else {
            created.on_or_after(IndexVersion::STORE_IGNORED_KEYWORDS_IN_BINARY_DOC_VALUES)
        }
    }

    fn keyword_like(&self, rf: &ResolvedField, vals: &[Value], ignore_above: Option<usize>, is_version: bool, sorted: bool) -> Vec<Value> {
        let mut dv: Vec<String> = Vec::new();
        let mut ign: Vec<Value> = Vec::new();
        let mut ign_above: Vec<String> = Vec::new();
        for v in vals {
            let s = match v {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => {
                    ign.push(v.clone());
                    continue;
                }
            };
            if rf.declared != EsType::ConstantKeyword && ignore_above.is_some_and(|ia| s.chars().count() > ia) {
                ign_above.push(s);
                continue;
            }
            dv.push(s);
        }
        // Values over `ignore_above` kept in binary doc values come back
        // sorted (`KeywordFieldMapper.storeIgnoredFieldsInBinaryDocValues`).
        if sorted && rf.declared == EsType::Keyword && ign_above.len() > 1 && self.ignored_keywords_in_binary_dv() {
            ign_above.sort();
        }
        let ign: Vec<Value> = ign_above.into_iter().map(Value::String).chain(ign).collect();
        if sorted {
            if is_version {
                dv.sort_by_key(|s| version::encode(s));
            } else {
                dv.sort();
            }
            dv.dedup();
        }
        dv.into_iter().map(Value::String).chain(ign).collect()
    }
}

/// `XContentDataHelper.encodeToken` of a JSON value (the sort key of
/// values kept in binary doc values).
fn encoded_value(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    match v {
        Value::String(s) => {
            out.push(b'S');
            out.extend_from_slice(s.as_bytes());
        }
        Value::Bool(true) => out.push(b't'),
        Value::Bool(false) => out.push(b'f'),
        Value::Null => out.push(b'n'),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if let Ok(i) = i32::try_from(i) {
                    out.push(b'I');
                    out.extend_from_slice(&i.to_le_bytes());
                } else {
                    out.push(b'L');
                    out.extend_from_slice(&i.to_le_bytes());
                }
            } else if let Some(u) = n.as_u64() {
                out.push(b'i');
                out.extend_from_slice(u.to_string().as_bytes());
            } else {
                out.push(b'D');
                out.extend_from_slice(&n.as_f64().unwrap_or(0.0).to_le_bytes());
            }
        }
        other => {
            out.push(b'j');
            out.extend_from_slice(other.to_string().as_bytes());
        }
    }
    out
}

/// String values through the keyword field's normalizer (what its doc
/// values hold); other values unchanged.
fn normalized(rf: &ResolvedField, vals: &[Value]) -> Vec<Value> {
    let Some(n) = &rf.normalizer else { return vals.to_vec() };
    vals.iter()
        .map(|v| match v {
            Value::String(s) => Value::String(n.analyze(s).into_iter().map(|t| t.text).collect::<Vec<_>>().join("")),
            other => other.clone(),
        })
        .collect()
}

/// `subobjects: false`: nested JSON objects are flattened into dotted names.
fn flatten_dotted(om: &ObjectMapping, entries: Vec<(String, Value)>) -> Vec<(String, Value)> {
    fn walk(om: &ObjectMapping, prefix: &str, v: Value, out: &mut Vec<(String, Value)>) {
        match v {
            Value::Object(m) if !om.properties.contains_key(prefix) => {
                for (k, x) in m {
                    walk(om, &format!("{prefix}.{k}"), x, out);
                }
            }
            x => out.push((prefix.to_string(), x)),
        }
    }
    let mut out = Vec::new();
    for (k, v) in entries {
        walk(om, &k, v, &mut out);
    }
    out
}

enum Tree {
    Leaf(Value),
    Node(BTreeMap<String, Tree>),
}

fn tree_value(t: Tree) -> Value {
    match t {
        Tree::Leaf(v) => v,
        Tree::Node(m) => Value::Object(m.into_iter().map(|(k, v)| (k, tree_value(v))).collect()),
    }
}

/// `flattened`: `a.b.c -> v` keyed values back into nested objects; a key
/// that collides with an existing leaf keeps its remaining dots.
fn flattened_tree(keyed: BTreeMap<String, BTreeSet<String>>) -> Value {
    let mut root: BTreeMap<String, Tree> = BTreeMap::new();
    for (key, dv) in keyed {
        let vals: Vec<Value> = dv.into_iter().map(Value::String).collect();
        let Some(v) = combine(vals) else { continue };
        let segs: Vec<&str> = key.split('.').collect();
        let mut cur = &mut root;
        let mut i = 0;
        loop {
            let last = i + 1 == segs.len();
            if last {
                cur.insert(segs[i].to_string(), Tree::Leaf(v));
                break;
            }
            if matches!(cur.get(segs[i]), Some(Tree::Leaf(_))) {
                cur.insert(segs[i..].join("."), Tree::Leaf(v));
                break;
            }
            let e = cur.entry(segs[i].to_string()).or_insert_with(|| Tree::Node(BTreeMap::new()));
            match e {
                Tree::Node(m) => cur = m,
                Tree::Leaf(_) => unreachable!(),
            }
            i += 1;
        }
    }
    tree_value(Tree::Node(root))
}

fn range_value(rf: &ResolvedField, kind: rng::RangeKind, lo: u128, hi: u128) -> Value {
    use rng::RangeKind as K;
    let render = |k: u128| -> Value {
        match kind {
            K::Integer | K::Long => Value::from(rng::i64_of_key(k)),
            K::Float => f32_value(rng::f64_of_key(k) as f32),
            K::Double => f64_value(rng::f64_of_key(k)),
            K::Date => Value::String(rf.date_format.format(rng::i64_of_key(k), crate::date::Tz::UTC)),
            K::Ip => Value::String(rustic_core::fieldtypes::ip::display(rng::ip_of_key(k))),
        }
    };
    let mut m = Map::new();
    m.insert("gte".into(), if lo == kind.min_key() { Value::Null } else { render(lo) });
    m.insert("lte".into(), if hi == kind.max_key() { Value::Null } else { render(hi) });
    Value::Object(m)
}

// Lucene `GeoEncodingUtils`: 32-bit quantisation of latitude / longitude.
const LAT_DECODE: f64 = 180.0 / 4294967296.0;
const LON_DECODE: f64 = 360.0 / 4294967296.0;

fn encode_lat(lat: f64) -> i32 {
    let lat = if lat == 90.0 { f64::from_bits(lat.to_bits() - 1) } else { lat };
    (lat / LAT_DECODE).floor() as i32
}

fn encode_lon(lon: f64) -> i32 {
    let lon = if lon == 180.0 { f64::from_bits(lon.to_bits() - 1) } else { lon };
    (lon / LON_DECODE).floor() as i32
}

fn decode_lat(e: i32) -> f64 {
    e as f64 * LAT_DECODE
}

fn decode_lon(e: i32) -> f64 {
    e as f64 * LON_DECODE
}

#[cfg(test)]
mod tests {
    /// `_ignored_source` values as ES 9.5.5 stores them (bytes from the
    /// reference node): singular per entry, and coalesced per field name.
    #[test]
    fn ignored_source_encodings_match_es() {
        fn base64_decode(s: &str) -> Vec<u8> {
            const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut bits = 0u32;
            let mut n = 0;
            let mut out = Vec::new();
            for c in s.bytes().filter(|&c| c != b'=') {
                bits = (bits << 6) | A.iter().position(|&a| a == c).unwrap() as u32;
                n += 6;
                if n >= 8 {
                    n -= 8;
                    out.push((bits >> n) as u8);
                }
            }
            out
        }
        let mut v = Vec::new();
        ignored_value(&mut v, &serde_json::json!(1), None);
        let value = IgnoredEntry { name: "value".into(), parent: 0, value: v };
        let mut v = Vec::new();
        ignored_value(&mut v, &serde_json::json!("f5t7-66gt"), None);
        let id = IgnoredEntry { name: "id".into(), parent: 0, value: v };
        assert_eq!(encode_ignored_source(&[value, id], IgnoredSourceFormat::Singular), vec!["BQAAAHZhbHVlSQEAAAA=", "AgAAAGlkU2Y1dDctNjZndA=="]);
        let mut v = vec![b'y'];
        v.extend_from_slice(b"---\nname: \"foo\"\nvalue: 10\n");
        let obj = IgnoredEntry { name: "object".into(), parent: 0, value: v };
        assert_eq!(encode_ignored_source(&[obj], IgnoredSourceFormat::Coalesced), vec!["AQZvYmplY3QAG3ktLS0KbmFtZTogImZvbyIKdmFsdWU6IDEwCg=="]);
        let mut v = Vec::new();
        ignored_value(&mut v, &serde_json::json!(1.5), None);
        assert_eq!(v, [&[b'D'][..], &1.5f64.to_le_bytes()].concat());
        // Java `HashMap` order (reference node): by bucket, newest first within one.
        let e = |n: &str| IgnoredEntry { name: n.into(), parent: 0, value: vec![b't'] };
        let names = |v: Vec<String>| -> Vec<String> {
            v.iter()
                .map(|b| {
                    let raw = base64_decode(b);
                    String::from_utf8(raw[2..2 + raw[1] as usize].to_vec()).unwrap()
                })
                .collect()
        };
        let doc = ["zz", "aaaaa", "big", "t", "arr", "obj", "m"].map(e);
        assert_eq!(names(encode_ignored_source(&doc, IgnoredSourceFormat::Coalesced)), ["arr", "zz", "big", "t", "aaaaa", "obj", "m"]);
        assert_eq!(names(encode_ignored_source(&[e("arr"), e("zz")], IgnoredSourceFormat::Coalesced)), ["zz", "arr"]);
    }

    use super::*;
    use crate::analysis::AnalysisRegistry;
    use serde_json::json;

    fn syn(mapping: Value, src: Value) -> Value {
        let reg = AnalysisRegistry::default();
        let mut m = Mappings::parse(&mapping, &reg).unwrap();
        m.assign_core_ids();
        let lk = FieldLookup::build(&m, &reg);
        // Apply dynamic mapping like the write path does.
        let (_, upd) = crate::document::parse_with_dynamic(&src, "1", &m, &lk, 1000).unwrap();
        let (m, lk) = match upd {
            Some((m2, lk2)) => (m2, lk2),
            None => (m, lk),
        };
        synthesize(&src, &m, &lk, Keep::None).unwrap().0
    }

    #[test]
    fn geo_quantisation_matches_lucene() {
        assert_eq!(decode_lat(encode_lat(41.12)), 41.1199999647215);
        assert_eq!(decode_lon(encode_lon(-71.34)), -71.34000004269183);
        assert_eq!(decode_lat(encode_lat(2.5)), 2.4999999767169356);
    }

    #[test]
    fn keywords_sorted_deduped_and_ignored_kept() {
        let out = syn(
            json!({"properties": {"k": {"type": "keyword", "ignore_above": 3}, "o": {"properties": {"b": {"type": "keyword"}, "a": {"type": "keyword"}}}}}),
            json!({"k": ["zzzzz", "ab", "abcd", "aa", "ab"], "o": [{"b": "x", "a": ["2", "1"]}, {"a": "3"}], "n": null}),
        );
        assert_eq!(out, json!({"k": ["aa", "ab", "zzzzz", "abcd"], "o": {"a": ["1", "2", "3"], "b": "x"}}));
        // Canonical form is a fixed point.
        assert_eq!(syn(json!({"properties": {"k": {"type": "keyword", "ignore_above": 3}}}), out.clone())["k"], out["k"]);
    }

    #[test]
    fn numbers_dates_booleans() {
        let out = syn(
            json!({"properties": {"l": {"type": "long"}, "f": {"type": "float"}, "hf": {"type": "half_float"}, "sf": {"type": "scaled_float", "scaling_factor": 100},
                "d": {"type": "double"}, "b": {"type": "boolean"}, "t": {"type": "date"}, "ip": {"type": "ip"}}}),
            json!({"l": [3, 1, 3], "f": 1.1, "hf": 1.1, "sf": 1.234, "d": 100, "b": [true, false, "true", ""], "t": ["2020-01-01", "2019-05-05T10:00:00Z"], "ip": ["::ffff:1.2.3.4", "10.0.0.1", "::1", "10.0.0.1"]}),
        );
        assert_eq!(
            out,
            json!({"b": [false, false, true, true], "d": 100.0, "f": 1.1, "hf": 1.0996094, "ip": ["::1", "1.2.3.4", "10.0.0.1"], "l": [1, 3, 3], "sf": 1.23,
                "t": ["2019-05-05T10:00:00.000Z", "2020-01-01T00:00:00.000Z"]})
        );
    }

    #[test]
    fn unmapped_and_disabled_kept_verbatim() {
        let out = syn(
            json!({"properties": {"o": {"type": "object", "enabled": false}, "p": {"properties": {"x": {"type": "keyword"}}, "dynamic": false}}}),
            json!({"o": {"z": [1, 2, {"b": 1}], "a": null, "to.a.b": 1}, "p": [{"x": "b", "y": 1, "w.v": 3}, {"x": "a", "y": [2, 3]}]}),
        );
        assert_eq!(out, json!({"o": {"z": [1, 2, {"b": 1}], "a": null, "to.a.b": 1}, "p": {"w": {"v": 3}, "x": ["a", "b"], "y": [1, 2, 3]}}));
    }

    #[test]
    fn nested_and_flattened_and_ranges() {
        let out = syn(
            json!({"properties": {"n": {"type": "nested", "properties": {"b": {"type": "long"}}}, "f": {"type": "flattened"}, "r": {"type": "integer_range"}}}),
            json!({"n": [{"b": [2, 1]}, {"b": 5}], "f": {"b": 1, "a": [3, "x", true], "c": {"d": "q"}}, "r": [{"gt": 1, "lt": 5}, {"gte": 1}]}),
        );
        assert_eq!(out, json!({"f": {"a": ["3", "true", "x"], "b": "1", "c": {"d": "q"}}, "n": [{"b": [1, 2]}, {"b": 5}], "r": [{"gte": 1, "lte": null}, {"gte": 2, "lte": 4}]}));
    }

    #[test]
    fn keep_modes() {
        let reg = AnalysisRegistry::default();
        let mapping = json!({"properties": {"a": {"type": "keyword", "synthetic_source_keep": "all"}, "b": {"type": "keyword", "synthetic_source_keep": "arrays"}, "o": {"properties": {"x": {"type": "keyword"}}}}});
        let mut m = Mappings::parse(&mapping, &reg).unwrap();
        m.assign_core_ids();
        let lk = FieldLookup::build(&m, &reg);
        let src = json!({"a": ["z", "a", "z"], "b": ["z", "a"], "o": [{"x": "b"}, {"x": "a"}]});
        assert_eq!(synthesize(&src, &m, &lk, Keep::None).unwrap().0, json!({"a": ["z", "a", "z"], "b": ["z", "a"], "o": {"x": ["a", "b"]}}));
        assert_eq!(synthesize(&src, &m, &lk, Keep::Arrays).unwrap().0, json!({"a": ["z", "a", "z"], "b": ["z", "a"], "o": [{"x": "b"}, {"x": "a"}]}));
        assert_eq!(synthesize(&json!({"b": "s"}), &m, &lk, Keep::None).unwrap().0, json!({"b": "s"}));
    }
}

#[cfg(test)]
mod residual_id_tests {
    #[test]
    fn residual_ids_write_runs_and_parse_back() {
        for ids in [vec![], vec![9u16], vec![9, 11, 12, 13, 14, 19, 20, 25], vec![1, 2, 3, 4, 5], vec![7, 9, 11, 65535]] {
            let mut s = String::new();
            super::write_ids(&mut s, &ids);
            assert_eq!(super::parse_ids(&s).collect::<Vec<_>>(), ids, "{s}");
        }
        let mut s = String::new();
        super::write_ids(&mut s, &[9, 11, 12, 13, 14, 15, 16, 17, 19]);
        assert_eq!(s, "9,11-17,19");
        // The earlier comma lists read the same.
        assert_eq!(super::parse_ids("3,5,6").collect::<Vec<_>>(), vec![3, 5, 6]);
    }
}
