//! `_field_caps`: per-field type / searchable / aggregatable across the
//! target indices, with `fields` wildcards, `include_unmapped`, `filters`
//! (`+metadata,-metadata,-nested,-multifield,-parent`) and `types`.

use crate::error::{EsError, Result};
use crate::mapping::{wildcard_match, EsType};
use crate::script::{RuntimeFields, RtType};
use crate::search::Target;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// (name, type, searchable, aggregatable) of the metadata fields every index has.
const METADATA: &[(&str, &str, bool, bool)] = &[
    ("_data_stream_timestamp", "_data_stream_timestamp", false, false),
    ("_doc_count", "integer", false, false),
    ("_feature", "_feature", false, false),
    ("_field_names", "_field_names", true, false),
    ("_id", "_id", true, false),
    ("_ignored", "_ignored", true, true),
    ("_ignored_source", "_ignored_source", false, false),
    ("_index", "_index", true, true),
    ("_index_mode", "_index_mode", true, true),
    ("_inference_fields", "_inference_fields", false, false),
    ("_nested_path", "_nested_path", true, false),
    ("_routing", "_routing", true, false),
    ("_seq_no", "_seq_no", true, true),
    ("_source", "_source", false, false),
    ("_tier", "keyword", true, true),
    ("_version", "_version", false, true),
];

/// The `type` `_field_caps` reports for a mapped type.
fn caps_type(t: EsType) -> String {
    match t {
        // `SemanticTextFieldType.familyTypeName()`.
        EsType::MatchOnlyText | EsType::PatternText | EsType::SemanticText => "text",
        EsType::ConstantKeyword | EsType::Wildcard => "keyword",
        EsType::TokenCount => "integer",
        o => o.name(),
    }
    .to_string()
}

#[derive(Clone)]
struct Cap {
    ty: String,
    searchable: bool,
    aggregatable: bool,
    metadata: bool,
    meta: Option<Value>,
    /// `true` for a real (indexed) field, `false` for a runtime field.
    real: bool,
    /// `time_series_dimension: true`.
    ts_dimension: bool,
    /// `time_series_metric` value.
    ts_metric: Option<String>,
}

pub struct FieldCapsOpts<'a> {
    pub fields: &'a [String],
    pub include_unmapped: bool,
    pub filters: &'a [String],
    pub types: &'a [String],
    /// Request-level `runtime_mappings`.
    pub runtime_mappings: Option<&'a Value>,
    /// `include_empty_fields` (default `true`): when `false`, fields with no
    /// indexed values in the target indices are omitted.
    pub include_empty_fields: bool,
}

fn index_caps(t: &Target, opts: &FieldCapsOpts) -> BTreeMap<String, Cap> {
    // Field-level security: ungranted fields are absent from the (restricted) lookup.
    let lk = t.lookup();
    let fls = t.field_access();
    let only_meta = opts.filters.iter().any(|f| f == "+metadata");
    let no_meta = opts.filters.iter().any(|f| f == "-metadata");
    let no_nested = opts.filters.iter().any(|f| f == "-nested");
    let no_multi = opts.filters.iter().any(|f| f == "-multifield");
    let no_parent = opts.filters.iter().any(|f| f == "-parent");
    let is_ts = t.svc.setting("index.mode").as_deref() == Some("time_series");
    let mut out = BTreeMap::new();
    if !no_meta {
        let routing_dv = t.svc.meta().mappings.routing_doc_values();
        for (n, ty, s, a) in METADATA {
            let a = *a || (*n == "_routing" && routing_dv);
            out.insert(n.to_string(), Cap { ty: ty.to_string(), searchable: *s, aggregatable: a, metadata: true, meta: None, real: true, ts_dimension: false, ts_metric: None });
        }
    }
    if only_meta {
        return out;
    }
    for (name, rf) in &lk.fields {
        if name == crate::tsdb::ROUTING_HASH_FIELD && lk.get(crate::tsdb::TSID_FIELD).is_some() {
            continue;
        }
        if name == crate::tsdb::TSID_FIELD {
            if !no_meta {
                out.insert(name.clone(), Cap { ty: "_tsid".to_string(), searchable: false, aggregatable: true, metadata: true, meta: None, real: true, ts_dimension: false, ts_metric: None });
            }
            continue;
        }
        if no_nested && rf.nested_path.is_some() {
            continue;
        }
        if no_multi && rf.is_multi_field {
            continue;
        }
        let (ty, rf2) = match (rf.ty, &rf.alias_path) {
            (EsType::Alias, Some(p)) => match lk.fields.get(p) {
                Some(target) => (target.ty, target.clone()),
                None => continue,
            },
            _ => (rf.ty, rf.clone()),
        };
        let aggregatable = match (ty, rf2.declared) {
            (EsType::GeoPoint, _) => rf2.doc_values,
            (_, EsType::GeoShape | EsType::Shape) => rf2.doc_values,
            (_, EsType::Binary) => rf2.doc_values,
            _ => rf2.is_aggregatable(),
        };
        let sayt_sub = rf2.declared == EsType::SearchAsYouType && rf2.is_multi_field && ["._2gram", "._3gram", "._4gram", "._index_prefix"].iter().any(|s| name.ends_with(s));
        let searchable = match (ty, rf2.declared) {
            (EsType::GeoPoint, _) => true,
            (_, EsType::Histogram | EsType::TDigest | EsType::ExponentialHistogram | EsType::Percolator | EsType::Binary) => false,
            // counted_keyword: its SORTED_SET doc values are always present,
            // so it is searchable even with `index: false` (ES 9.x).
            (_, EsType::CountedKeyword) => true,
            _ if sayt_sub => true,
            _ => rf2.is_searchable(),
        };
        let cap_ty = if sayt_sub && name.ends_with("._index_prefix") { "prefix".to_string() } else { caps_type(rf2.declared) };
        if let (EsType::Join, Some(rel)) = (rf2.declared, rf2.ext.as_ref().and_then(|e| e.params.get("relations")).and_then(Value::as_object)) {
            for parent in rel.keys() {
                out.insert(format!("{name}#{parent}"), Cap { ty: "parent".into(), searchable: true, aggregatable: true, metadata: false, meta: None, real: true, ts_dimension: false, ts_metric: None });
            }
        }
        if rf2.declared == EsType::CountedKeyword && rf.alias_path.is_none() {
            // The hidden `<field>_count` binary doc-values field holding the counts.
            out.insert(format!("{name}_count"), Cap { ty: "binary".into(), searchable: false, aggregatable: true, metadata: false, meta: None, real: true, ts_dimension: false, ts_metric: None });
        }
        if rf2.declared == EsType::Percolator {
            for (sub, ty, s, a) in [("extracted_terms", "keyword", true, false), ("extraction_result", "keyword", true, false), ("minimum_should_match_field", "integer", true, true), ("query_builder_field", "binary", false, true), ("range_field", "ip_range", true, false)] {
                out.insert(format!("{name}.{sub}"), Cap { ty: ty.into(), searchable: s, aggregatable: a, metadata: false, meta: None, real: true, ts_dimension: false, ts_metric: None });
            }
        }
        out.insert(name.clone(), Cap { ty: cap_ty, searchable, aggregatable, metadata: false, meta: rf2.meta.clone(), real: true, ts_dimension: rf2.time_series_dimension && is_ts, ts_metric: rf2.time_series_metric.clone().filter(|_| is_ts) });
    }
    // Runtime fields: the mapping's `runtime` section overlaid with the
    // request's `runtime_mappings`.
    let mut rt = RuntimeFields::default();
    if let Some(m) = t.svc.meta().mappings.runtime.as_ref() {
        if let Err(e) = rt.parse_into(m) {
            eprintln!("field_caps: failed to parse runtime mapping: {e}");
        }
    }
    if let Some(r) = opts.runtime_mappings {
        if let Err(e) = rt.parse_into(r) {
            eprintln!("field_caps: failed to parse runtime_mappings: {e}");
        }
    }
    for rf in &rt.fields {
        if out.contains_key(&rf.name) || fls.is_some_and(|fa| !fa.allows(&rf.name)) {
            continue;
        }
        let (searchable, aggregatable) = match rf.ty {
            RtType::Long | RtType::Double | RtType::Boolean | RtType::Date | RtType::Ip => (true, true),
            RtType::Keyword => (true, false),
            RtType::Lookup => (false, false),
            RtType::GeoShape => (true, true),
        };
        out.insert(rf.name.clone(), Cap { ty: rf.ty.name().to_string(), searchable, aggregatable, metadata: false, meta: None, real: false, ts_dimension: false, ts_metric: None });
    }
    // Parents of the leaf fields: every dotted prefix that is not itself a
    // field (empty / disabled objects have no leaves and so are not listed).
    let mut prefixes: BTreeSet<String> = BTreeSet::new();
    for name in lk.fields.keys() {
        if name == crate::tsdb::TSID_FIELD || out.get(name).is_some_and(|c| c.metadata) {
            continue;
        }
        let parts: Vec<&str> = name.split('.').collect();
        for n in 1..parts.len() {
            prefixes.insert(parts[..n].join("."));
        }
    }
    for path in prefixes {
        if lk.fields.contains_key(&path) || out.contains_key(&path) {
            continue;
        }
        let nested = lk.objects.get(&path).is_some_and(|i| i.nested);
        if no_nested && nested {
            continue;
        }
        // `-parent` drops object/nested parents but keeps their children.
        if no_parent {
            continue;
        }
        // An object is visible only when some granted field lives below it.
        if fls.is_some() {
            let below = format!("{path}.");
            if !lk.fields.keys().any(|f| f.starts_with(&below)) {
                continue;
            }
        }
        out.insert(path.clone(), Cap { ty: if nested { "nested" } else { "object" }.into(), searchable: false, aggregatable: false, metadata: false, meta: None, real: true, ts_dimension: false, ts_metric: None });
    }
    out
}

/// Build the `_field_caps` response for already resolved targets.
pub fn field_caps(targets: &[Target], opts: &FieldCapsOpts) -> Result<Value> {
    if opts.fields.is_empty() {
        return Err(EsError::action_validation(&["no fields specified"]));
    }
    let pats: Vec<&str> = opts.fields.iter().flat_map(|f| f.split(',')).map(str::trim).filter(|f| !f.is_empty()).collect();
    let mut index_names: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
    index_names.sort();
    index_names.dedup();
    // field → type → [(index, cap)]
    let mut by_field: BTreeMap<String, BTreeMap<String, Vec<(String, Cap)>>> = BTreeMap::new();
    let mut present: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for t in targets {
        for (name, cap) in index_caps(t, opts) {
            if !pats.iter().any(|p| wildcard_match(p, &name)) {
                continue;
            }
            if !opts.types.is_empty() && !opts.types.iter().any(|x| *x == cap.ty) {
                continue;
            }
            if !opts.include_empty_fields && cap.real && !cap.metadata && !matches!(cap.ty.as_str(), "object" | "nested") {
                let q = json!({"exists": {"field": name}});
                if crate::search::count(std::slice::from_ref(t), Some(&q), None).unwrap_or(1) == 0 {
                    continue;
                }
            }
            present.entry(name.clone()).or_default().insert(t.svc.name.clone());
            by_field.entry(name).or_default().entry(cap.ty.clone()).or_default().push((t.svc.name.clone(), cap));
        }
    }
    // `FieldCapabilities.isInference` (ES 9): whether the field is an
    // inference (`semantic_text`) field of the index.
    let lookups: HashMap<String, std::sync::Arc<crate::mapping::FieldLookup>> = targets.iter().map(|t| (t.svc.name.clone(), t.svc.lookup())).collect();
    let is_inference = |index: &str, field: &str| lookups.get(index).and_then(|l| l.fields.get(field)).is_some_and(|f| crate::inference::semantic::is_inference_field(f));
    let mut fields = Map::new();
    for (name, types) in by_field {
        let mut unmapped: Vec<String> = Vec::new();
        if opts.include_unmapped {
            let have = present.get(&name).cloned().unwrap_or_default();
            unmapped = index_names.iter().filter(|i| !have.contains(*i)).cloned().collect();
        }
        let multi = types.len() > 1 || !unmapped.is_empty();
        let mut tm = Map::new();
        for (ty, entries) in types {
            let mut m = Map::new();
            let all_s = entries.iter().all(|(_, c)| c.searchable);
            let all_a = entries.iter().all(|(_, c)| c.aggregatable);
            m.insert("type".into(), json!(ty));
            m.insert("metadata_field".into(), json!(entries[0].1.metadata));
            m.insert("searchable".into(), json!(all_s));
            m.insert("aggregatable".into(), json!(all_a));
            let inf: Vec<bool> = entries.iter().map(|(i, _)| is_inference(i, &name)).collect();
            m.insert("inference".into(), json!(inf.iter().all(|x| *x)));
            if multi || entries.iter().any(|(_, c)| !c.real) {
                let mut ix: Vec<&String> = entries.iter().map(|(i, _)| i).collect();
                ix.sort();
                m.insert("indices".into(), json!(ix));
            }
            if !all_s && entries.iter().any(|(_, c)| c.searchable) {
                let mut ix: Vec<&String> = entries.iter().filter(|(_, c)| !c.searchable).map(|(i, _)| i).collect();
                ix.sort();
                m.insert("non_searchable_indices".into(), json!(ix));
            }
            if !all_a && entries.iter().any(|(_, c)| c.aggregatable) {
                let mut ix: Vec<&String> = entries.iter().filter(|(_, c)| !c.aggregatable).map(|(i, _)| i).collect();
                ix.sort();
                m.insert("non_aggregatable_indices".into(), json!(ix));
            }
            if inf.iter().any(|x| *x) && !inf.iter().all(|x| *x) {
                let mut ix: Vec<&String> = entries.iter().zip(&inf).filter(|(_, x)| !**x).map(|((i, _), _)| i).collect();
                ix.sort();
                m.insert("non_inference_indices".into(), json!(ix));
            }
            // Time-series attributes: present only when at least one index
            // declares the field as a dimension / metric.
            if entries.iter().any(|(_, c)| c.ts_dimension) {
                m.insert("time_series_dimension".into(), json!(entries.iter().all(|(_, c)| c.ts_dimension)));
                if !entries.iter().all(|(_, c)| c.ts_dimension) {
                    let mut ix: Vec<&String> = entries.iter().filter(|(_, c)| !c.ts_dimension).map(|(i, _)| i).collect();
                    ix.sort();
                    m.insert("non_dimension_indices".into(), json!(ix));
                }
            }
            let metrics: BTreeSet<&str> = entries.iter().filter_map(|(_, c)| c.ts_metric.as_deref()).collect();
            if !metrics.is_empty() {
                // Metric present on every index and identical: reported; any
                // disagreement (including a missing metric) is a conflict.
                let uniform = metrics.len() == 1 && entries.iter().all(|(_, c)| c.ts_metric.is_some());
                if uniform {
                    m.insert("time_series_metric".into(), json!(metrics.iter().next().unwrap()));
                } else {
                    let mut ix: Vec<&String> = entries.iter().map(|(i, _)| i).collect();
                    ix.sort();
                    m.insert("metric_conflicts_indices".into(), json!(ix));
                }
            }
            let mut meta: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
            for (_, c) in &entries {
                if let Some(Value::Object(mo)) = &c.meta {
                    for (k, v) in mo {
                        meta.entry(k.clone()).or_default().insert(v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()));
                    }
                }
            }
            if !meta.is_empty() {
                m.insert("meta".into(), json!(meta));
            }
            tm.insert(ty, Value::Object(m));
        }
        if !unmapped.is_empty() {
            tm.insert("unmapped".into(), json!({"type": "unmapped", "metadata_field": false, "searchable": false, "aggregatable": false, "indices": unmapped}));
        }
        fields.insert(name, Value::Object(tm));
    }
    Ok(json!({ "indices": index_names, "fields": fields }))
}
