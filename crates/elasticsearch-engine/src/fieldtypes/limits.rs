//! Mapping limits (`index.mapping.*.limit`): total fields, depth, nested
//! fields / objects, field name length, dimension fields, and
//! `total_fields.ignore_dynamic_beyond_limit`.

use crate::error::{EsError, Result};
use crate::mapping::{Mapper, Mappings};
use std::collections::BTreeMap;

/// Effective limits of an index.
#[derive(Clone, Debug)]
pub struct Limits {
    pub total_fields: usize,
    /// Index query cap used while validating stored percolator queries.
    pub max_regex_length: usize,
    pub depth: usize,
    pub nested_fields: usize,
    pub nested_objects: usize,
    pub field_name_length: usize,
    pub dimension_fields: usize,
    pub ignore_dynamic_beyond_limit: bool,
    /// `index.mapping.field_name_length.ignore_dynamic_beyond_limit`.
    pub ignore_dynamic_beyond_field_name_length: bool,
    /// `index.mapping.array_objects.limit`: objects inside arrays per document.
    pub array_objects: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { total_fields: 1000, max_regex_length: 1000, depth: 20, nested_fields: 50, nested_objects: 10_000, field_name_length: usize::MAX, dimension_fields: 32_768, ignore_dynamic_beyond_limit: false, ignore_dynamic_beyond_field_name_length: false, array_objects: 50_000 }
    }
}

impl Limits {
    /// From flat index settings (`get(key)` returns the setting's string).
    pub fn from_settings(get: impl Fn(&str) -> Option<String>) -> Limits {
        let mut d = Limits::default();
        let created = crate::version::IndexVersion::from_setting(get("index.version.created").map(serde_json::Value::String).as_ref());
        d.nested_fields = crate::settings::nested_fields_limit_default(created);
        let n = |k: &str, def: usize| get(k).and_then(|s| s.parse::<usize>().ok()).unwrap_or(def);
        Limits {
            total_fields: n("index.mapping.total_fields.limit", d.total_fields),
            max_regex_length: n("index.max_regex_length", d.max_regex_length),
            depth: n("index.mapping.depth.limit", d.depth),
            nested_fields: n("index.mapping.nested_fields.limit", d.nested_fields),
            nested_objects: n("index.mapping.nested_objects.limit", d.nested_objects),
            field_name_length: n("index.mapping.field_name_length.limit", d.field_name_length),
            dimension_fields: n("index.mapping.dimension_fields.limit", d.dimension_fields),
            // logsdb indices drop dynamic fields past the limit unless told otherwise.
            ignore_dynamic_beyond_limit: get("index.mapping.total_fields.ignore_dynamic_beyond_limit").map(|s| s == "true").unwrap_or_else(|| get("index.mode").as_deref() == Some("logsdb")),
            ignore_dynamic_beyond_field_name_length: get("index.mapping.field_name_length.ignore_dynamic_beyond_limit").as_deref() == Some("true"),
            array_objects: n("index.mapping.array_objects.limit", d.array_objects),
        }
    }
}

struct Stats {
    depth: usize,
    nested: usize,
    dimensions: usize,
    longest: Option<String>,
    /// The object holding the first field deeper than the depth limit.
    too_deep: Option<String>,
}

fn walk(props: &BTreeMap<String, Mapper>, depth: usize, s: &mut Stats, limit_len: usize, limit_depth: usize, prefix: &str) {
    for (k, m) in props {
        if depth + 1 > limit_depth && s.too_deep.is_none() && !prefix.is_empty() {
            s.too_deep = Some(prefix.to_string());
        }
        if k.chars().count() > limit_len && s.longest.is_none() {
            s.longest = Some(k.clone());
        }
        match m {
            Mapper::Field(f) => {
                s.depth = s.depth.max(depth + 1);
                if f.bool_param("time_series_dimension", false) {
                    s.dimensions += 1;
                }
            }
            Mapper::Object(o) => {
                s.depth = s.depth.max(depth + 1);
                if o.nested {
                    s.nested += 1;
                }
                walk(&o.properties, depth + 1, s, limit_len, limit_depth, &if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") });
            }
        }
    }
}

/// Check a mapping against the limits (put mapping / dynamic updates).
pub fn check(m: &Mappings, l: &Limits) -> Result<()> {
    let total = m.total_fields();
    if total > l.total_fields {
        return Err(EsError::illegal_argument(format!("Limit of total fields [{}] has been exceeded", l.total_fields)));
    }
    let mut s = Stats { depth: 0, nested: 0, dimensions: 0, longest: None, too_deep: None };
    walk(&m.root.properties, 0, &mut s, l.field_name_length, l.depth, "");
    if s.depth > l.depth {
        return Err(EsError::illegal_argument(match &s.too_deep {
            Some(p) => format!("Limit of mapping depth [{}] has been exceeded due to object field [{p}]", l.depth),
            None => format!("Limit of mapping depth [{}] has been exceeded due to object field", l.depth),
        }));
    }
    if s.nested > l.nested_fields {
        return Err(EsError::illegal_argument(format!(
            "Limit of nested fields [{}] has been exceeded",
            l.nested_fields
        )));
    }
    if s.dimensions > l.dimension_fields {
        return Err(EsError::illegal_argument(format!("Limit of total dimension fields [{}] has been exceeded", l.dimension_fields)));
    }
    if let Some(n) = s.longest {
        return Err(EsError::illegal_argument(format!("Field name [{n}] is longer than the limit of [{}] characters", l.field_name_length)));
    }
    Ok(())
}

/// Number of nested objects a document would create (for
/// `index.mapping.nested_objects.limit`).
pub fn check_nested_objects(doc: &rustic_core::types::Document, l: &Limits) -> Result<()> {
    fn count(d: &rustic_core::types::Document) -> usize {
        d.nested.iter().map(|(_, c)| 1 + count(c)).sum()
    }
    let n = count(doc);
    if n > l.nested_objects {
        return Err(EsError::document_parsing(format!(
            "The number of nested documents has exceeded the allowed limit of [{}]. This limit can be set by changing the [index.mapping.nested_objects.limit] index level setting.",
            l.nested_objects
        )));
    }
    Ok(())
}
