//! `index.sort.*` validation (any index mode) and the `logsdb` index mode rules.

use crate::error::{EsError, Result};
use crate::mapping::FieldLookup;
use crate::settings::FlatSettings;
use serde_json::Value;

fn list(s: &FlatSettings, key: &str) -> Vec<String> {
    match s.get(key) {
        Some(Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect(),
        Some(Value::String(v)) => v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        Some(Value::Null) | None => vec![],
        Some(other) => vec![other.to_string()],
    }
}

fn show(v: &[String]) -> String {
    format!("[{}]", v.join(", "))
}

pub fn is_logsdb(s: &FlatSettings) -> bool {
    s.get("index.mode").and_then(Value::as_str) == Some("logsdb")
}

/// `index.logsdb.route_on_sort_fields: true` lets a logsdb index carry an `index.routing_path`.
pub fn routes_on_sort_fields(s: &FlatSettings) -> bool {
    is_logsdb(s) && (matches!(s.get("index.logsdb.route_on_sort_fields"), Some(Value::Bool(true))) || s.get("index.logsdb.route_on_sort_fields").and_then(Value::as_str) == Some("true"))
}

/// The index's sort fields: `index.sort.field`, else the index mode's
/// default sort (`IndexSortConfigDefaults`).
pub fn effective_sort_fields(s: &FlatSettings) -> Vec<String> {
    if s.get("index.sort.field").is_some_and(|v| !v.is_null()) {
        return list(s, "index.sort.field");
    }
    let truthy = |k: &str| s.get(k).is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true"));
    match s.get("index.mode").and_then(Value::as_str).map(str::to_ascii_lowercase).as_deref() {
        Some("time_series") => vec!["_tsid".into(), "@timestamp".into()],
        Some("logsdb" | "logsdb_columnar") => {
            let mut v = Vec::new();
            if truthy("index.logsdb.sort_on_host_name") {
                v.push("host.name".to_string());
            }
            if truthy("index.logsdb.sort_on_message_template") {
                v.push("message.template_id".to_string());
            }
            v.push("@timestamp".into());
            v
        }
        _ => Vec::new(),
    }
}

/// `DocumentMapper.validate`: a runtime field may not shadow an index sort
/// field.
pub fn check_runtime_shadows_sort(s: &FlatSettings, runtime: Option<&Value>) -> Result<()> {
    let Some(rt) = runtime.and_then(Value::as_object).filter(|o| !o.is_empty()) else { return Ok(()) };
    let sort = effective_sort_fields(s);
    for (name, def) in rt {
        // A composite runtime field's sub-fields count too.
        let mut names = vec![name.clone()];
        if let Some(sub) = def.get("fields").and_then(Value::as_object) {
            names.extend(sub.keys().map(|k| format!("{name}.{k}")));
        }
        if let Some(n) = names.iter().find(|n| sort.contains(n)) {
            return Err(EsError::mapper_parsing(format!("runtime field [{n}] shadows an index sort field, which would prevent shards from being allocated")));
        }
    }
    Ok(())
}

/// Sort fields must exist, live outside `nested` objects and line up with the order / mode / missing lists.
pub fn validate_sort(s: &FlatSettings, lookup: &FieldLookup) -> Result<()> {
    // `IndexSortConfig.validateSortSettings`: order / mode / missing need
    // an explicit `index.sort.field`.
    if s.get("index.sort.field").is_none_or(Value::is_null) {
        for key in ["index.sort.order", "index.sort.mode", "index.sort.missing"] {
            if s.get(key).is_some_and(|v| !v.is_null()) {
                return Err(EsError::illegal_argument(format!("setting [{key}] requires [index.sort.field] to be configured")));
            }
        }
    }
    let fields = list(s, "index.sort.field");
    for (key, what) in [("index.sort.order", "order"), ("index.sort.mode", "mode"), ("index.sort.missing", "missing")] {
        let mut vals = list(s, key);
        if vals.is_empty() {
            continue;
        }
        if what == "mode" {
            vals = vals.into_iter().map(|v| v.to_uppercase()).collect();
        }
        if vals.len() != fields.len() {
            return Err(EsError::illegal_argument(format!("index.sort.fields:{} index.sort.{what}:{}, size mismatch", show(&fields), show(&vals))));
        }
    }
    for f in &fields {
        match lookup.get(f) {
            Some(rf) => {
                if let Some(np) = &rf.nested_path {
                    return Err(EsError::illegal_argument(format!("cannot apply index sort to field [{f}] under nested object [{np}]")));
                }
            }
            None => match lookup.objects.get(f.as_str()) {
                Some(o) if o.nested => return Err(EsError::illegal_argument(format!("cannot apply index sort to field [{f}] under nested object [{f}]"))),
                _ => return Err(EsError::illegal_argument(format!("unknown index sort field:[{f}]"))),
            },
        }
    }
    Ok(())
}

/// The sort an index mode applies without `index.sort.field`
/// (`IndexSortConfigDefaults`): fields, orders, modes and missing values.
/// `time_series` sorts on `_tsid` ascending then `@timestamp` descending;
/// `logsdb` on `@timestamp` descending, after `host.name` /
/// `message.template_id` when enabled (or `host.name, @timestamp` ascending
/// for indices created before optional host sorting); other modes have none.
pub type SortSpec = (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>, Vec<&'static str>);

/// See [`SortSpec`].
pub fn mode_default_sort(s: &FlatSettings) -> SortSpec {
    use crate::version::IndexVersion;
    let mode = s.get("index.mode").and_then(Value::as_str).map(str::to_ascii_lowercase).unwrap_or_else(|| "standard".into());
    let created = IndexVersion::from_setting(s.get("index.version.created"));
    let truthy = |k: &str| s.get(k).is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true"));
    match mode.as_str() {
        "time_series" => (vec!["_tsid", "@timestamp"], vec!["asc", "desc"], vec!["min", "max"], vec!["_last", "_last"]),
        "logsdb" | "logsdb_columnar" => {
            if created.on_or_after(IndexVersion::LOGSB_OPTIONAL_SORTING_ON_HOST_NAME) || created.between(IndexVersion::LOGSB_OPTIONAL_SORTING_ON_HOST_NAME_BACKPORT, IndexVersion::UPGRADE_TO_LUCENE_10_0_0) {
                match (truthy("index.logsdb.sort_on_host_name"), truthy("index.logsdb.sort_on_message_template")) {
                    (true, true) => (vec!["host.name", "message.template_id", "@timestamp"], vec!["asc", "asc", "desc"], vec!["min", "min", "max"], vec!["_last", "_last", "_last"]),
                    (true, false) => (vec!["host.name", "@timestamp"], vec!["asc", "desc"], vec!["min", "max"], vec!["_last", "_last"]),
                    (false, true) => (vec!["message.template_id", "@timestamp"], vec!["asc", "desc"], vec!["min", "max"], vec!["_last", "_last"]),
                    (false, false) => (vec!["@timestamp"], vec!["desc"], vec!["max"], vec!["_last"]),
                }
            } else {
                (vec!["host.name", "@timestamp"], vec!["asc", "asc"], vec!["min", "min"], vec!["_last", "_last"])
            }
        }
        _ => (vec![], vec![], vec![], vec![]),
    }
}

/// Whether `time_series` / `logsdb` indices without `index.sort.field`
/// physically sort their segments by the mode's default sort, as ES does.
/// On by default (`RUSTIC_MODE_DEFAULT_SORT=false` turns it off): sorted
/// segments compress far better and let timestamp sorts and filtered
/// queries skip blocks. Ingest of big5 at Rally's shape (1 client,
/// 500-doc bulks, 5M docs) runs 22.7k docs/s sorted against ES 9.5.5's
/// 20.4k; only a refresh per bulk (k8s_metrics shaped, 8 clients, which
/// is not a time_series / logsdb track) made sorted refreshes costly.
pub fn mode_default_sort_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RUSTIC_MODE_DEFAULT_SORT").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false")))
}

/// The core index sort: explicit `index.sort.*` settings, else (with
/// [`mode_default_sort_enabled`]) the index mode's default sort
/// ([`mode_default_sort`]: a `time_series` index's segments hold their docs
/// by `_tsid`, then `@timestamp` descending, as ES's do). `None` without
/// either, or when a field cannot be resolved to a sortable column yet.
/// Defaults of explicit settings as ES `IndexSortConfig`: ascending, `min`
/// for ascending / `max` for descending sorts, missing values last.
pub fn core_sort(s: &FlatSettings, lookup: &FieldLookup) -> Option<rustic_core::index_sort::IndexSort> {
    core_sort_with(s, lookup, mode_default_sort_enabled())
}

/// [`core_sort`]; `mode_defaults`: apply the index mode's default sort.
pub fn core_sort_with(s: &FlatSettings, lookup: &FieldLookup, mode_defaults: bool) -> Option<rustic_core::index_sort::IndexSort> {
    use crate::mapping::EsType;
    use rustic_core::index_sort::{IndexSort, IndexSortField, IndexSortKind};
    let (fields, order, mode, missing) = if s.get("index.sort.field").is_some_and(|v| !v.is_null()) {
        (list(s, "index.sort.field"), list(s, "index.sort.order"), list(s, "index.sort.mode"), list(s, "index.sort.missing"))
    } else if !mode_defaults {
        return None;
    } else {
        let owned = |v: Vec<&str>| v.into_iter().map(str::to_string).collect::<Vec<_>>();
        let (f, o, m, mi) = mode_default_sort(s);
        (owned(f), owned(o), owned(m), owned(mi))
    };
    if fields.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(fields.len());
    for (i, name) in fields.iter().enumerate() {
        let f = lookup.get(name)?;
        let field = f.core_id?;
        let kind = match f.ty {
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Date | EsType::DateNanos | EsType::Boolean => IndexSortKind::Long,
            EsType::UnsignedLong => IndexSortKind::UnsignedLong,
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => IndexSortKind::Double,
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Ip => IndexSortKind::Keyword,
            _ => return None,
        };
        let reverse = order.get(i).is_some_and(|o| o.eq_ignore_ascii_case("desc"));
        let max_mode = match mode.get(i) {
            Some(m) => m.eq_ignore_ascii_case("max"),
            None => reverse,
        };
        let missing_first = missing.get(i).is_some_and(|m| m == "_first");
        out.push(IndexSortField { field, kind, reverse, max_mode, missing_first });
    }
    Some(IndexSort::new(out))
}
