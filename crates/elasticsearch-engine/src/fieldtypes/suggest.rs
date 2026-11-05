//! The completion suggester (`suggest.<name>.completion`) over
//! `completion` fields: per segment, the field's doc values are loaded
//! into an FST ([`CompletionIndex`]) and looked up by prefix (optionally
//! fuzzy), filtered / boosted by contexts.

use crate::error::{EsError, Result};
use crate::mapping::{EsType, ResolvedField, SOURCE_FIELD};
use crate::values;
use rustic_core::fieldtypes::completion::{analyze, top_suggestions, CompletionEntry, CompletionIndex};
use serde_json::{json, Map, Value};

const SEP: char = '\u{1f}';
const CSEP: char = '\u{1e}';

/// A context mapping of a completion field.
#[derive(Clone, Debug)]
pub struct ContextDef {
    pub name: String,
    pub geo: bool,
    pub precision: usize,
    pub path: Option<String>,
}

pub fn context_defs(f: &ResolvedField) -> Vec<ContextDef> {
    let Some(c) = f.ext.as_ref().and_then(|e| e.params.get("contexts")) else { return Vec::new() };
    let arr = match c {
        Value::Array(a) => a.clone(),
        o @ Value::Object(_) => vec![o.clone()],
        _ => vec![],
    };
    arr.iter()
        .filter_map(|d| {
            Some(ContextDef {
                name: d.get("name")?.as_str()?.to_string(),
                geo: d.get("type").and_then(Value::as_str).map(|t| t.eq_ignore_ascii_case("geo")).unwrap_or(false),
                precision: d.get("precision").map(parse_precision).unwrap_or(6),
                path: d.get("path").and_then(Value::as_str).map(str::to_string),
            })
        })
        .collect()
}

/// Indexed values of one context entry (categories, or geohashes).
pub fn context_values(def: &ContextDef, v: &Value) -> std::result::Result<Vec<String>, EsError> {
    if def.geo {
        let pts: Vec<&Value> = match v {
            Value::Array(a) if !(a.len() == 2 && a.iter().all(Value::is_number)) => a.iter().collect(),
            x => vec![x],
        };
        let mut out = Vec::new();
        for p in pts {
            let (lat, lon) = values::parse_geo_point(p).ok_or_else(|| EsError::illegal_argument(format!("failed to parse geo context [{}]", def.name)))?;
            out.push(values::geohash_encode(lat, lon, def.precision));
        }
        return Ok(out);
    }
    Ok(match v {
        Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect(),
        Value::String(s) => vec![s.clone()],
        Value::Null => vec![],
        other => vec![other.to_string()],
    })
}

/// Doc value of one input: `input SEP weight (SEP name CSEP value)*`.
pub fn encode_entry(input: &str, weight: u64, ctxs: &[(String, String)]) -> String {
    let mut s = format!("{input}{SEP}{weight}");
    for (n, v) in ctxs {
        s.push(SEP);
        s.push_str(n);
        s.push(CSEP);
        s.push_str(v);
    }
    s
}

pub fn decode_entry(s: &str) -> Option<(String, u64, Vec<(String, String)>)> {
    let mut it = s.split(SEP);
    let input = it.next()?.to_string();
    let weight = it.next()?.parse().ok()?;
    let ctxs = it.filter_map(|c| c.split_once(CSEP).map(|(a, b)| (a.to_string(), b.to_string()))).collect();
    Some((input, weight, ctxs))
}

pub use crate::suggest::SuggestTarget;

/// Run a `suggest` section (see [`crate::suggest::run`]).
pub fn run(spec: &Value, targets: &[SuggestTarget]) -> Result<Value> {
    crate::suggest::run(spec, targets, false)
}

/// Geohash length of a context `precision` (a level, or a distance like
/// `5km`): ES `GeoUtils.geoHashLevelsForPrecision` rounded to a cell at
/// least as large as the distance.
pub fn parse_precision(v: &Value) -> usize {
    let meters = match v {
        Value::Number(n) => return n.as_u64().unwrap_or(6).clamp(1, 12) as usize,
        Value::String(s) => match s.parse::<u64>() {
            Ok(n) => return (n as usize).clamp(1, 12),
            Err(_) => match crate::values::parse_distance(&Value::String(s.clone())) {
                Some(m) => m,
                None => return 6,
            },
        },
        _ => return 6,
    };
    const EQ: f64 = 40_075_016.685_578_49;
    const POLAR: f64 = std::f64::consts::PI * 6_356_752.314_245;
    if meters <= 0.0 {
        return 12;
    }
    let ratio = 1.0 + POLAR / EQ;
    let width = ((meters * meters) / (ratio * ratio)).sqrt();
    let part = (EQ / width).ceil().round() as u64;
    let level = 63 - part.max(1).leading_zeros() as i64;
    let mut l = if part <= (1u64 << level) { (level + 1) / 2 } else { (level + 2) / 2 } as usize;
    let cell = |l: usize| -> f64 {
        let w = EQ / (1u64 << (((l + 1) / 2) * 3 + (l / 2) * 2)) as f64;
        let h = POLAR / (1u64 << (((l + 1) / 2) * 2 + (l / 2) * 3)) as f64;
        (w * w + h * h).sqrt()
    };
    l = l.clamp(1, 12);
    if cell(l) < meters {
        l = (l - 1).max(1);
    }
    l
}

/// The 8 neighbouring cells of a geohash (same length).
pub fn geohash_neighbors(h: &str) -> Vec<String> {
    let Some((lat, lon)) = values::geohash_decode(h) else { return Vec::new() };
    let n = h.len();
    let lon_bits = (5 * n + 1) / 2;
    let lat_bits = 5 * n / 2;
    let dw = 360.0 / (1u64 << lon_bits) as f64;
    let dh = 180.0 / (1u64 << lat_bits) as f64;
    let mut out = Vec::new();
    for dy in [-1.0, 0.0, 1.0] {
        for dx in [-1.0, 0.0, 1.0] {
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            let la = lat + dy * dh;
            if !(-90.0..=90.0).contains(&la) {
                continue;
            }
            let mut lo = lon + dx * dw;
            if lo > 180.0 {
                lo -= 360.0;
            }
            if lo < -180.0 {
                lo += 360.0;
            }
            out.push(values::geohash_encode(la, lo, n));
        }
    }
    out
}

pub fn completion(text: &str, regex: bool, c: &Map<String, Value>, targets: &[SuggestTarget], source: &crate::search::SourceFilter) -> Result<Value> {
    let field = c.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("the required field option [field] is missing"))?;
    let size = c.get("size").and_then(Value::as_u64).unwrap_or(5) as usize;
    if size == 0 {
        return Err(EsError::illegal_argument("size must be positive"));
    }
    let skip_dup = c.get("skip_duplicates").and_then(Value::as_bool).unwrap_or(false);
    let fuzzy = match c.get("fuzzy") {
        None | Some(Value::Bool(false)) | Some(Value::Null) => None,
        Some(fz) => {
            let edits = match fz.get("fuzziness") {
                Some(Value::Number(n)) => n.as_u64().unwrap_or(1).min(2) as u8,
                Some(Value::String(s)) if s.to_ascii_lowercase().starts_with("auto") => {
                    let (lo, hi) = s
                        .split_once(':')
                        .and_then(|(_, r)| r.split_once(','))
                        .and_then(|(a, b)| Some((a.trim().parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?)))
                        .unwrap_or((3, 6));
                    let n = text.chars().count();
                    if n < lo {
                        0
                    } else if n < hi {
                        1
                    } else {
                        2
                    }
                }
                Some(Value::String(s)) => s.parse::<u8>().unwrap_or(1).min(2),
                _ => 1,
            };
            let prefix_len = fz.get("prefix_length").and_then(Value::as_u64).unwrap_or(1) as usize;
            let min_len = fz.get("min_length").and_then(Value::as_u64).unwrap_or(3) as usize;
            let transpositions = fz.get("transpositions").and_then(Value::as_bool).unwrap_or(true);
            if text.chars().count() < min_len || edits == 0 {
                None
            } else {
                Some((edits, prefix_len, transpositions))
            }
        }
    };
    let regex_re = if regex {
        Some(regex::Regex::new(&format!("^(?:{text})")).map_err(|e| EsError::illegal_argument(format!("invalid regex [{text}]: {e}")))?)
    } else {
        None
    };
    let ctx_filter = c.get("contexts").and_then(Value::as_object).cloned();
    let mut ctx_mappings = 0usize;
    // (score, text, index, target, leaf, doc, matched contexts)
    #[allow(clippy::type_complexity)]
    let mut all: Vec<(f32, String, String, usize, usize, rustic_core::types::DocId, Vec<(String, String)>)> = Vec::new();
    for (ti, t) in targets.iter().enumerate() {
        let Some(f) = t.lookup.get(field) else { continue };
        if f.declared != EsType::Completion {
            return Err(EsError::illegal_argument(format!("Field [{field}] is not a completion suggest field")));
        }
        let Some(id) = f.core_id else { continue };
        let defs = context_defs(f);
        ctx_mappings = ctx_mappings.max(defs.len());
        let preserve = f.ext.as_ref().map(|e| e.bool("preserve_separators", true)).unwrap_or(true);
        let max_len = f.ext.as_ref().and_then(|e| e.u64("max_input_length")).unwrap_or(50) as usize;
        let wanted = query_contexts(&defs, ctx_filter.as_ref())?;
        if !defs.is_empty() && wanted.iter().all(|(_, v)| v.is_empty()) {
            return Err(EsError::illegal_argument("Missing mandatory contexts in context query"));
        }
        for (li, leaf) in t.leaves.iter().enumerate() {
            let live = leaf.live_docs();
            let mut entries = Vec::new();
            for d in 0..leaf.num_docs() {
                if !live(d) {
                    continue;
                }
                for enc in leaf.keyword_values(id, d) {
                    if let Some((input, weight, contexts)) = decode_entry(&enc) {
                        entries.push(CompletionEntry { doc: d, input, weight, contexts });
                    }
                }
            }
            if entries.is_empty() {
                continue;
            }
            let idx = CompletionIndex::build(entries, |s| analyze(s, preserve, max_len));
            let mut hits = match &regex_re {
                Some(re) => (0..idx.len())
                    .filter(|&i| {
                        let e = idx.entry(i);
                        re.is_match(&e.input) || re.is_match(&e.input.to_lowercase())
                    })
                    .map(|i| (i, 1.0f32))
                    .collect(),
                None => idx.lookup(&analyze(text, preserve, max_len), fuzzy),
            };
            // Matched contexts per doc (ES groups the contexts of every hit path of a doc).
            let mut matched: std::collections::HashMap<rustic_core::types::DocId, Vec<(String, String)>> = std::collections::HashMap::new();
            if !defs.is_empty() {
                hits = hits
                    .into_iter()
                    .filter_map(|(i, s)| {
                        let e = idx.entry(i);
                        let mut boost: Option<f32> = None;
                        let mut hit_ctx: Vec<(String, String)> = Vec::new();
                        for (n, vals) in &wanted {
                            for (val, b, prefix) in vals {
                                for (cn, cv) in e.contexts.iter() {
                                    if cn == n && if *prefix { cv.starts_with(val.as_str()) } else { cv == val } {
                                        boost = Some(boost.map_or(*b, |x: f32| x.max(*b)));
                                        if !hit_ctx.contains(&(cn.clone(), cv.clone())) {
                                            hit_ctx.push((cn.clone(), cv.clone()));
                                        }
                                    }
                                }
                            }
                        }
                        boost.map(|b| {
                            let m = matched.entry(e.doc).or_default();
                            for c in hit_ctx {
                                if !m.contains(&c) {
                                    m.push(c);
                                }
                            }
                            (i, s * b)
                        })
                    })
                    .collect();
            }
            for (sug, score) in top_suggestions(&idx, &hits, size, false) {
                let cx = matched.get(&sug.doc).cloned().unwrap_or_default();
                all.push((score, sug.input, t.index.to_string(), ti, li, sug.doc, cx));
            }
        }
    }
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(&b.1)));
    let mut seen = std::collections::HashSet::new();
    let mut options = Vec::new();
    for (score, input, index, ti, li, doc, cx) in all {
        if skip_dup && !seen.insert(input.clone()) {
            continue;
        }
        let leaf = &targets[ti].leaves[li];
        let stored = leaf.get_doc(doc)?;
        let id = leaf.keyword_value(crate::mapping::ID_FIELD, doc).unwrap_or_default();
        // The fetch phase's `_source`: vectors left out where the index
        // excludes them (`index.mapping.exclude_source_vectors`).
        let lk = targets[ti].lookup;
        let source_json = stored.as_ref().and_then(|d| d.get(SOURCE_FIELD)).and_then(|v| v.as_text()).and_then(|s| {
            let rebuilt = crate::synthetic::is_residual(s).then(|| crate::vector_source::load(s, lk, leaf, doc).into_owned());
            let s = rebuilt.as_deref().unwrap_or(s);
            let stripped = if lk.registry.exclude_source_vectors { crate::vector_source::strip_source(s, lk, &|_| false) } else { None };
            serde_json::from_str::<Value>(stripped.as_deref().unwrap_or(s)).ok()
        });
        let mut m = Map::new();
        m.insert("text".into(), json!(input));
        m.insert("_index".into(), json!(index));
        m.insert("_id".into(), json!(id));
        m.insert("_score".into(), values::java_double(score as f64));
        match (source_json, source) {
            (Some(s), crate::search::SourceFilter::All) => {
                m.insert("_source".into(), s);
            }
            (Some(s), crate::search::SourceFilter::Filter { includes, excludes }) => {
                m.insert("_source".into(), crate::search::filter_source(&s, includes, excludes));
            }
            _ => {}
        }
        if !cx.is_empty() {
            m.insert("contexts".into(), java_context_map(&cx, ctx_mappings));
        }
        options.push(Value::Object(m));
        if options.len() >= size {
            break;
        }
    }
    Ok(json!({ "text": text, "offset": 0, "length": text.encode_utf16().count(), "options": options }))
}

/// Query contexts: name → [(value, boost, prefix)].
#[allow(clippy::type_complexity)]
fn query_contexts(defs: &[ContextDef], filter: Option<&Map<String, Value>>) -> Result<Vec<(String, Vec<(String, f32, bool)>)>> {
    let mut wanted = Vec::new();
    let Some(cf) = filter else { return Ok(wanted) };
    for (n, v) in cf {
        let def = defs.iter().find(|d| &d.name == n).ok_or_else(|| {
            EsError::illegal_argument(format!("Unknown context name [{n}], must be one of {:?}", defs.iter().map(|d| d.name.clone()).collect::<Vec<_>>()))
        })?;
        let items: Vec<Value> = match v {
            Value::Array(a) if !(def.geo && a.len() == 2 && a.iter().all(Value::is_number)) => a.clone(),
            x => vec![x.clone()],
        };
        let mut vals = Vec::new();
        for it in items {
            let obj = it.as_object().filter(|o| o.contains_key("context"));
            let boost = obj.and_then(|o| o.get("boost")).and_then(Value::as_f64).unwrap_or(1.0) as f32;
            if def.geo {
                let pv = obj.map(|o| &o["context"]).unwrap_or(&it);
                let (lat, lon) = values::parse_geo_point(pv).ok_or_else(|| EsError::illegal_argument("failed to parse geo context"))?;
                let qprec = obj.and_then(|o| o.get("precision")).map(parse_precision).unwrap_or(12);
                let neighbours: Vec<usize> = match obj.and_then(|o| o.get("neighbours")) {
                    Some(Value::Array(a)) => a.iter().map(parse_precision).collect(),
                    Some(x) => vec![parse_precision(x)],
                    None => vec![],
                };
                let gh = values::geohash_encode(lat, lon, def.precision.min(qprec));
                let mut locs: Vec<String> = vec![gh.clone()];
                if neighbours.is_empty() && gh.len() == def.precision {
                    locs.extend(geohash_neighbors(&gh));
                } else {
                    for np in neighbours.into_iter().filter(|&p| p < gh.len()) {
                        let t = gh[..np].to_string();
                        locs.extend(geohash_neighbors(&t));
                        locs.push(t);
                    }
                }
                locs.sort();
                locs.dedup();
                for l in locs {
                    let pre = l.len() < def.precision;
                    vals.push((l, boost, pre));
                }
            } else {
                let (val, prefix) = match obj {
                    Some(o) => (
                        o["context"].as_str().map(str::to_string).unwrap_or_else(|| o["context"].to_string()),
                        o.get("prefix").and_then(Value::as_bool).unwrap_or(false),
                    ),
                    None => (it.as_str().map(str::to_string).unwrap_or_else(|| it.to_string()), false),
                };
                vals.push((val, boost, prefix));
            }
        }
        wanted.push((n.clone(), vals));
    }
    Ok(wanted)
}

/// Java `String.hashCode()`.
fn java_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// Order `items` like iterating a `java.util.HashMap` with `cap` buckets
/// (bucket index, then insertion order within a bucket).
pub(crate) fn java_order<T>(items: Vec<(String, T)>, cap: usize) -> Vec<(String, T)> {
    let mut v: Vec<(usize, usize, (String, T))> = items
        .into_iter()
        .enumerate()
        .map(|(i, it)| {
            let h = java_hash(&it.0) as u32;
            (((h ^ (h >> 16)) as usize) & (cap - 1), i, it)
        })
        .collect();
    v.sort_by_key(|x| (x.0, x.1));
    v.into_iter().map(|x| x.2).collect()
}

/// The `contexts` map of a completion option: `Maps.newMapWithExpectedSize(n)`
/// (a `HashMap`) of name to a `HashSet` of values, in Java iteration order.
fn java_context_map(cx: &[(String, String)], mappings: usize) -> Value {
    let cap = java_expected_capacity(mappings);
    let mut by_name: Vec<(String, Vec<(String, ())>)> = Vec::new();
    for (n, v) in cx {
        match by_name.iter_mut().find(|(k, _)| k == n) {
            Some((_, vs)) => vs.push((v.clone(), ())),
            None => by_name.push((n.clone(), vec![(v.clone(), ())])),
        }
    }
    let mut out = Map::new();
    for (n, vs) in java_order(by_name, cap) {
        let vals: Vec<Value> = java_order(vs, 16).into_iter().map(|(v, _)| Value::String(v)).collect();
        out.insert(n, Value::Array(vals));
    }
    Value::Object(out)
}

/// Table size of `Maps.newMapWithExpectedSize(n)` (a `HashMap`).
pub(crate) fn java_expected_capacity(n: usize) -> usize {
    ((n as f32 / 0.75 + 1.0) as usize).max(1).next_power_of_two()
}

#[cfg(test)]
mod java_order_tests {
    use super::*;

    /// Orders captured from ES 8.19.22 (`_termvectors` field order, completion `contexts`).
    #[test]
    fn hashmap_iteration_order() {
        let order = |names: &[&str]| -> Vec<String> {
            let mut v: Vec<&str> = names.to_vec();
            v.sort();
            let cap = java_expected_capacity(v.len());
            java_order(v.into_iter().map(|n| (n.to_string(), ())).collect(), cap).into_iter().map(|x| x.0).collect()
        };
        assert_eq!(order(&["title", "plain"]), ["title", "plain"]);
        assert_eq!(order(&["title", "plain", "body", "tag"]), ["tag", "body", "title", "plain"]);
        assert_eq!(order(&["alpha", "beta", "gamma", "delta", "zeta", "omega"]), ["zeta", "alpha", "delta", "beta", "gamma", "omega"]);
        assert_eq!(order(&["cat", "loc"]), ["loc", "cat"]);
    }
}
