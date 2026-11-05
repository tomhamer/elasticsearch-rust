//! ES 9 `index.mapping.exclude_source_vectors` (on by default for indices
//! created by 9.2+): vector embeddings are not kept in `_source` as sent.
//!
//! ES strips the indexed `dense_vector`, stored `sparse_vector` and
//! `rank_vectors` values from the stored `_source` and patches them back
//! from the vector storage when a source is loaded with vectors
//! (`SourceLoader.SyntheticVectors`): the value comes back in its
//! *indexed* form (Java `float`s, cosine vectors denormalized, bfloat16
//! rounded, base64 / hex decoded, sparse weights at feature precision) and
//! at the end of the object that held it. Searches, gets and mgets then
//! leave every vector field out of `_source` unless the request asks for
//! them (`_source.exclude_vectors: false`, `_source_exclude_vectors=false`)
//! or names them in `_source.includes`.
//!
//! Here the stored `_source` is already the patched one: [`rehydrate`]
//! rewrites each vector value into its indexed form and moves it to the end
//! of its object when the document is indexed, so updates, reindex, the
//! translog and every reader see exactly what ES's patched source holds,
//! and [`strip`] removes the vector fields on reads that exclude them.

use crate::mapping::{EsType, FieldLookup, ResolvedField};
use crate::version::IndexVersion;
use serde_json::{Map, Value};

/// The `index.mapping.exclude_source_vectors` value of an index (its
/// default follows the creation version).
pub fn excludes_source_vectors(get: impl Fn(&str) -> Option<String>) -> bool {
    match get("index.mapping.exclude_source_vectors") {
        Some(v) => v == "true",
        None => {
            let created = get("index.version.created").map(Value::String);
            IndexVersion::from_setting(created.as_ref()).on_or_after(IndexVersion::EXCLUDE_SOURCE_VECTORS_DEFAULT)
        }
    }
}

/// [`excludes_source_vectors`] over a flat settings map.
pub fn excludes_in(settings: &crate::settings::FlatSettings) -> bool {
    excludes_source_vectors(|k| settings.get(k).map(|v| match v {
        Value::String(s) => s.clone(),
        o => o.to_string(),
    }))
}

/// Field types ES treats as vector embeddings (`isVectorEmbedding`): left
/// out of a `_source` that excludes vectors.
fn is_vector_embedding(f: &ResolvedField) -> bool {
    matches!(f.declared, EsType::DenseVector | EsType::SparseVector | EsType::RankVectors)
}

/// Whether ES keeps `f` out of the stored `_source` and patches it back
/// (`syntheticVectorsLoader() != null`): indexed `dense_vector`s, stored
/// `sparse_vector`s (stored by default in such an index) and `rank_vectors`.
fn is_synthetic_vector(f: &ResolvedField) -> bool {
    let p = |k: &str| f.ext.as_ref().and_then(|e| e.params.get(k));
    match f.declared {
        EsType::DenseVector => !matches!(p("index"), Some(Value::Bool(false))) && !matches!(p("index").and_then(Value::as_str), Some("false")),
        // `store` is a common mapping parameter, already resolved onto the
        // field; it is not retained in the extended-type parameter map.
        EsType::SparseVector => f.stored,
        EsType::RankVectors => true,
        _ => false,
    }
}

/// The value ES patches back for `f` from what the document held, or
/// `None` when nothing is patched (null / unparseable).
fn indexed_form(f: &ResolvedField, v: &Value) -> Option<Value> {
    match f.declared {
        EsType::DenseVector => {
            if v.is_null() {
                return None;
            }
            let (row, dims) = crate::vectors::doc_vector_with_dims(f, v).ok()?;
            Some(dense_value(f, row, dims))
        }
        EsType::SparseVector => {
            // Terms in index order, weights at `FeatureField` precision.
            let mut toks: Vec<(String, f32)> = Vec::new();
            let objs: Vec<&Value> = match v {
                Value::Array(a) => a.iter().collect(),
                Value::Null => return None,
                x => vec![x],
            };
            for o in objs {
                for (t, w) in o.as_object()? {
                    let w = w.as_f64()? as f32;
                    toks.push((t.clone(), rustic_core::fieldtypes::features::quantize(w)));
                }
            }
            toks.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            toks.dedup_by(|a, b| a.0 == b.0);
            Some(Value::Object(toks.into_iter().map(|(t, w)| (t, crate::values::java_float(w))).collect::<Map<String, Value>>()))
        }
        EsType::RankVectors => {
            let a = v.as_array()?;
            let et = f.ext.as_ref().and_then(|e| e.params.get("element_type")).and_then(Value::as_str).unwrap_or("float");
            let byte = et == "byte" || et == "bit";
            let num = |x: f64| -> Value {
                match et {
                    "byte" | "bit" => Value::from(x as i64),
                    "bfloat16" => crate::values::java_float(crate::vectors::bf16_round(x as f32)),
                    _ => crate::values::java_float(x as f32),
                }
            };
            Some(Value::Array(
                a.iter()
                    .map(|row| match row {
                        Value::Array(r) => Value::Array(r.iter().map(|x| num(x.as_f64().unwrap_or(0.0))).collect()),
                        // A hex-encoded byte vector comes back as its bytes.
                        Value::String(h) if byte && h.len() % 2 == 0 && h.bytes().all(|b| b.is_ascii_hexdigit()) => {
                            Value::Array((0..h.len()).step_by(2).map(|i| Value::from(u8::from_str_radix(&h[i..i + 2], 16).unwrap_or(0) as i8 as i64)).collect())
                        }
                        other => other.clone(),
                    })
                    .collect(),
            ))
        }
        _ => None,
    }
}

/// The `dense_vector` value ES patches back for `f`, from the vector as
/// indexed (what [`crate::vectors::doc_vector`] produced, which is also
/// what the vector store holds).
fn dense_value(f: &ResolvedField, vec: Vec<f32>, dims: usize) -> Value {
    let et = crate::vectors::element_type(f);
    Value::Array(match et {
        // Packed rows: one signed byte per value (`byte`) / per 8 bits (`bit`).
        crate::vectors::ElementType::Byte | crate::vectors::ElementType::Bit => match crate::vectors::Packing::for_type(et, dims) {
            Some(p) => p.bytes(&vec).iter().map(|&b| Value::from(b as i8 as i64)).collect(),
            None => Vec::new(),
        },
        crate::vectors::ElementType::BFloat16 => vec.into_iter().map(crate::values::java_float).collect(),
        crate::vectors::ElementType::Float => {
            // Cosine vectors are stored unit length (with their magnitude)
            // unless already within 1e-3 of it, and denormalized on read.
            let mut sq = 0f32;
            for x in &vec {
                sq += x * x;
            }
            if f.similarity == "cosine" && (sq - 1.0).abs() >= 1e-3 && sq > 0.0 {
                let len = (sq as f64).sqrt() as f32;
                vec.into_iter().map(|x| crate::values::java_float((x / len) * len)).collect()
            } else {
                vec.into_iter().map(crate::values::java_float).collect()
            }
        }
    })
}

/// The synthetic vector fields of a mapping, by full path.
fn synthetic_fields(lk: &FieldLookup) -> Vec<&ResolvedField> {
    if !lk.may_have_vector_fields { return Vec::new(); }
    let mut v: Vec<&ResolvedField> = lk.fields.values().map(|f| f.as_ref()).filter(|f| is_synthetic_vector(f)).collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// Whether the mapping has any field [`rehydrate`] would rewrite.
pub fn has_synthetic_vectors(lk: &FieldLookup) -> bool {
    if !lk.may_have_vector_fields { return false; }
    lk.fields.values().any(|f| is_synthetic_vector(f))
}

/// The `_source` to store for `src` in an index that excludes source
/// vectors: every synthetic vector value in its indexed form, moved to the
/// end of its object (empty parents kept); `None` when nothing changes.
pub fn rehydrate(src: &Value, lk: &FieldLookup) -> Option<Value> {
    let fields = synthetic_fields(lk);
    if fields.is_empty() {
        return None;
    }
    let mut out = src.clone();
    let mut changed = false;
    walk(&mut out, "", &fields, &mut changed);
    changed.then_some(out)
}

fn walk(v: &mut Value, prefix: &str, fields: &[&ResolvedField], changed: &mut bool) {
    match v {
        Value::Array(a) => a.iter_mut().for_each(|x| walk(x, prefix, fields, changed)),
        Value::Object(m) => {
            let mut moved: Vec<(String, String, Option<Value>)> = Vec::new();
            for (k, c) in m.iter_mut() {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                if let Some(f) = fields.iter().find(|f| f.name == path) {
                    moved.push((path, k.clone(), indexed_form(f, c)));
                } else if c.is_object() || c.is_array() {
                    walk(c, &path, fields, changed);
                }
            }
            if moved.is_empty() {
                return;
            }
            *changed = true;
            moved.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, k, _) in &moved {
                m.shift_remove(k);
            }
            for (_, k, val) in moved {
                if let Some(val) = val {
                    m.insert(k, val);
                }
            }
        }
        _ => {}
    }
}

/// Whether a read of `_source` leaves the vectors out: the request's
/// explicit choice, else the index setting.
pub fn exclude_for(explicit: Option<bool>, index_excludes: bool) -> bool {
    explicit.unwrap_or(index_excludes)
}

/// Remove every vector embedding field from `src` (inside objects, arrays
/// and nested objects), except those `keep` (explicitly included by the
/// request's `_source.includes`).
pub fn strip(src: &mut Value, lk: &FieldLookup, keep: &dyn Fn(&str) -> bool) {
    let names = strip_names(lk, keep);
    if names.is_empty() {
        return;
    }
    strip_walk(src, "", &names);
}

/// The vector embedding fields a read leaves out (all but those `keep`).
fn strip_names<'l>(lk: &'l FieldLookup, keep: &dyn Fn(&str) -> bool) -> Vec<&'l str> {
    if !lk.may_have_vector_fields { return Vec::new(); }
    lk.fields.values().filter(|f| is_vector_embedding(f) && !keep(&f.name)).map(|f| f.name.as_str()).collect()
}

/// Removes the `names` fields under `v` (at path `prefix`); whether any was.
fn strip_walk(v: &mut Value, prefix: &str, names: &[&str]) -> bool {
    let mut removed = false;
    match v {
        Value::Array(a) => a.iter_mut().for_each(|x| removed |= strip_walk(x, prefix, names)),
        Value::Object(m) => {
            let keys: Vec<String> = m.keys().cloned().collect();
            for k in keys {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                if names.contains(&path.as_str()) {
                    m.shift_remove(&k);
                    removed = true;
                } else if is_parent_of(&path, names) {
                    if let Some(c) = m.get_mut(&k) {
                        removed |= strip_walk(c, &path, names);
                    }
                }
            }
        }
        _ => {}
    }
    removed
}

/// Whether some of `names` lies under `path`.
fn is_parent_of(path: &str, names: &[&str]) -> bool {
    names.iter().any(|n| n.len() > path.len() && n.starts_with(path) && n.as_bytes()[path.len()] == b'.')
}

/// The top-level members of a JSON object, values left unparsed.
struct RawMembers<'a>(Vec<(std::borrow::Cow<'a, str>, &'a serde_json::value::RawValue)>);

impl<'de> serde::Deserialize<'de> for RawMembers<'de> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = RawMembers<'de>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some((k, v)) = a.next_entry::<std::borrow::Cow<'de, str>, &'de serde_json::value::RawValue>()? {
                    out.push((k, v));
                }
                Ok(RawMembers(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// [`strip`] over a raw `_source`, without parsing the members that hold
/// no vector field (the stored text of the others is kept byte for byte,
/// as ES returns its stored source). `None` when nothing is removed.
fn strip_raw(raw: &str, names: &[&str]) -> Option<String> {
    let members: RawMembers = serde_json::from_str(raw).ok()?;
    let mut out = String::with_capacity(raw.len());
    let mut removed = false;
    out.push('{');
    for (k, v) in &members.0 {
        let text: std::borrow::Cow<'_, str> = if names.contains(&k.as_ref()) {
            removed = true;
            continue;
        } else if is_parent_of(k, names) {
            let mut val: Value = serde_json::from_str(v.get()).ok()?;
            if strip_walk(&mut val, k, names) {
                removed = true;
                val.to_string().into()
            } else {
                v.get().into()
            }
        } else {
            v.get().into()
        };
        if out.len() > 1 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(k.as_ref()).ok()?);
        out.push(':');
        out.push_str(&text);
    }
    out.push('}');
    removed.then_some(out)
}

/// A stored `_source` with every vector embedding field left out, but
/// those `keep`; `None` when it is returned unchanged.
pub fn strip_source(raw: &str, lk: &FieldLookup, keep: &dyn Fn(&str) -> bool) -> Option<String> {
    let names = strip_names(lk, keep);
    if names.is_empty() || raw.is_empty() {
        return None;
    }
    strip_raw(raw, &names)
}

/// Whether the mapping has any vector embedding field (nothing to strip
/// otherwise).
pub fn has_vector_fields(lk: &FieldLookup) -> bool {
    if !lk.may_have_vector_fields { return false; }
    lk.fields.values().any(|f| is_vector_embedding(f))
}

/// ES `SourceFilter.isExplicitlyIncluded`: a full include pattern match.
pub fn explicitly_included(includes: &[String], path: &str) -> bool {
    includes.iter().any(|p| crate::mapping::wildcard_match(p, path))
}

/// A stored `_source` (raw JSON) as a read returns it: vectors left out
/// when the read excludes them (`explicit` the request's
/// `exclude_vectors`, `includes` its `_source.includes`). `None` when it is
/// returned unchanged.
pub fn for_read(raw: &str, lk: &FieldLookup, index_excludes: bool, explicit: Option<bool>, includes: &[String]) -> Option<String> {
    if !exclude_for(explicit, index_excludes) || !has_vector_fields(lk) || raw.is_empty() {
        return None;
    }
    strip_source(raw, lk, &|p| explicitly_included(includes, p))
}


// ---------------------------------------------------------------------------
// Vectors kept out of the stored `_source`.
//
// ES stores the source without its synthetic vectors and patches them back
// from the vector storage on load. Here the stored `_source` of an index
// that excludes source vectors is the patched source (above) with each
// `dense_vector` that the vector store holds replaced by `null`: a value
// the patched source never has (a null vector is dropped from it), so the
// marker keeps the field's place (end of its object, dotted or not) while
// the stored text shrinks to the document without its embeddings. Loads
// put the vector store's value back in its indexed form ([`load`]); reads
// that leave vectors out drop the marked keys like any vector field.
// The translog keeps the full patched source.
// ---------------------------------------------------------------------------

/// Whether `f`'s value is taken out of the stored source and served from
/// the index: an indexed `dense_vector` (from the vector store) or a stored
/// `sparse_vector` (from its `token\0weight` doc values) outside nested
/// objects (one value per document, held by the root document).
pub(crate) fn is_store_backed(f: &ResolvedField) -> bool {
    matches!(f.declared, EsType::DenseVector | EsType::SparseVector) && f.core_id.is_some() && f.nested_path.is_none() && is_synthetic_vector(f)
}

/// The `sparse_vector` value ES patches back for core field `id` from the
/// document's feature doc values (`cols`): its tokens in index order with
/// their `FeatureField`-precision weights, as [`indexed_form`] renders the
/// value sent. `None` without a (decodable, single-weighted) feature.
fn sparse_value(cols: &dyn crate::synthetic::DocColumns, id: u16) -> Option<Value> {
    let encoded = cols.keywords(id);
    if encoded.is_empty() {
        return None;
    }
    let mut out = Map::with_capacity(encoded.len());
    for e in &encoded {
        let (t, w) = rustic_core::fieldtypes::features::decode(e)?;
        // A token indexed with two weights: not what one value renders.
        if out.insert(t.to_string(), crate::values::java_float(w)).is_some() {
            return None;
        }
    }
    Some(Value::Object(out))
}

/// The value a store-backed field `f` is patched back with: the vector
/// store's row (`vector`) or the feature doc values (`cols`).
fn store_value(f: &ResolvedField, vector: &dyn Fn(u16) -> Option<Vec<f32>>, cols: &dyn crate::synthetic::DocColumns) -> Option<Value> {
    let id = f.core_id?;
    match f.declared {
        EsType::SparseVector => sparse_value(cols, id),
        _ => Some(dense_value(f, vector(id)?, f.dims.unwrap_or(0))),
    }
}

/// Whether a `fields` request (`pats`) can return a value of a vector the
/// stored `_source` leaves to the vector store: some pattern matches a
/// store-backed field or an object holding one. Otherwise the vectors need
/// not be loaded for it: what `fields` reads from `_source` leaves their
/// marked keys alone (a 1024-dim vector rendered and parsed back per hit
/// for `fields: ["docid"]`, the Rally recall request, cost more than the
/// search).
pub fn fields_load_vectors(lk: &FieldLookup, pats: &[(String, Option<String>)]) -> bool {
    if pats.is_empty() {
        return false;
    }
    store_backed(lk).iter().any(|f| {
        let name = f.name.as_str();
        let parents = name.match_indices('.').map(|(i, _)| &name[..i]);
        std::iter::once(name).chain(parents).any(|path| pats.iter().any(|(p, _)| crate::mapping::wildcard_match(p, path)))
    })
}

/// The store-backed vector fields of a mapping.
fn store_backed(lk: &FieldLookup) -> Vec<&ResolvedField> {
    if !lk.registry.exclude_source_vectors || !lk.may_have_vector_fields {
        return Vec::new();
    }
    lk.fields.values().map(|f| f.as_ref()).filter(|f| is_store_backed(f)).collect()
}

/// The `_source` to store for `patched` (the output of [`rehydrate`]) of
/// the parsed document `doc`: every store-backed vector replaced by the
/// `null` marker when the index gives it back as is (a dense vector the
/// document indexed; a sparse vector its feature doc values rebuild
/// exactly). `None` when none is.
pub fn stored_form(patched: &Value, lk: &FieldLookup, doc: &rustic_core::types::Document) -> Option<String> {
    stored_form_marked(patched, lk, doc).map(|(s, _)| s)
}

/// [`stored_form`] and the paths of the sparse vectors it marked (their
/// values then come back from the doc values, so the stored-fields block
/// need not keep them).
pub fn stored_form_marked(patched: &Value, lk: &FieldLookup, doc: &rustic_core::types::Document) -> Option<(String, Vec<String>)> {
    let indexed = |id: u16| doc.fields.iter().any(|(f, v)| *f == id && matches!(v, rustic_core::types::FieldValue::Vector(_)));
    let fields: Vec<&ResolvedField> = store_backed(lk)
        .into_iter()
        .filter(|f| f.core_id.is_some_and(|id| f.declared == EsType::SparseVector || indexed(id)))
        .collect();
    if fields.is_empty() {
        return None;
    }
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    let sparse_marked = std::cell::RefCell::new(Vec::new());
    let accept = |path: &str, v: &Value| -> bool {
        match fields.iter().find(|f| f.name == path) {
            Some(f) if f.declared == EsType::SparseVector => {
                let ok = f.core_id.and_then(|id| sparse_value(doc, id)).as_ref() == Some(v);
                if ok {
                    sparse_marked.borrow_mut().push(path.to_string());
                }
                ok
            }
            Some(_) => true,
            None => false,
        }
    };
    let mut out = patched.clone();
    let marked = mark_walk(&mut out, "", &names, &accept);
    marked.then(|| (out.to_string(), sparse_marked.into_inner()))
}

/// The `_source` to store for a synthetic-source document whose canonical
/// synthetic source is `synth` (stored whole: no doc-value residual):
/// every indexed `dense_vector` the document holds replaced by the `null`
/// marker [`load`] patches back from the vector store, so the vector is
/// kept once, in the vector store, instead of also as JSON text in the
/// stored fields. As in ES (no `_source` is stored for synthetic source;
/// vectors come back from their vector values), a vector is read back in
/// its indexed form: Java floats, a cosine vector denormalized. `None` when
/// nothing is marked.
pub fn synthetic_stored_form(synth: &Value, lk: &FieldLookup, doc: &rustic_core::types::Document) -> Option<String> {
    let indexed = |id: u16| doc.fields.iter().any(|(f, v)| *f == id && matches!(v, rustic_core::types::FieldValue::Vector(_)));
    let fields: Vec<&ResolvedField> = store_backed(lk)
        .into_iter()
        .filter(|f| f.declared == EsType::DenseVector && f.core_id.is_some_and(indexed))
        .collect();
    if fields.is_empty() {
        return None;
    }
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    let mut out = synth.clone();
    mark_walk(&mut out, "", &names, &|path, _| fields.iter().any(|f| f.name == path)).then(|| out.to_string())
}

fn mark_walk(v: &mut Value, prefix: &str, names: &[&str], accept: &dyn Fn(&str, &Value) -> bool) -> bool {
    let mut marked = false;
    match v {
        Value::Array(a) => a.iter_mut().for_each(|x| marked |= mark_walk(x, prefix, names, accept)),
        Value::Object(m) => {
            for (k, c) in m.iter_mut() {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                if names.contains(&path.as_str()) {
                    if !c.is_null() && accept(&path, c) {
                        *c = Value::Null;
                        marked = true;
                    }
                } else if is_parent_of(&path, names) {
                    marked |= mark_walk(c, &path, names, accept);
                }
            }
        }
        _ => {}
    }
    marked
}

/// A stored `_source` with the store-backed vectors put back (`vector`:
/// the document's vector of a core field; `cols`: its doc values). `None`
/// when unchanged.
pub fn rehydrate_stored(raw: &str, lk: &FieldLookup, vector: &dyn Fn(u16) -> Option<Vec<f32>>, cols: &dyn crate::synthetic::DocColumns) -> Option<String> {
    // Every marker is a `null`.
    if raw.len() < 4 || !raw.contains("null") {
        return None;
    }
    let fields = store_backed(lk);
    if fields.is_empty() {
        return None;
    }
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    let value_of = |path: &str| -> Option<Value> {
        let f = fields.iter().find(|f| f.name == path)?;
        store_value(f, vector, cols)
    };
    let members: RawMembers = serde_json::from_str(raw).ok()?;
    let mut out = String::with_capacity(raw.len() + 1024);
    let mut changed = false;
    out.push('{');
    for (k, v) in &members.0 {
        let text: std::borrow::Cow<'_, str> = if names.contains(&k.as_ref()) && v.get() == "null" {
            changed = true;
            match value_of(k) {
                Some(val) => val.to_string().into(),
                // Nothing in the store: the key goes, as ES leaves it out.
                None => continue,
            }
        } else if is_parent_of(k, &names) {
            let mut val: Value = serde_json::from_str(v.get()).ok()?;
            if fill_walk(&mut val, k, &names, &value_of) {
                changed = true;
                val.to_string().into()
            } else {
                v.get().into()
            }
        } else {
            v.get().into()
        };
        if out.len() > 1 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(k.as_ref()).ok()?);
        out.push(':');
        out.push_str(&text);
    }
    out.push('}');
    changed.then_some(out)
}

fn fill_walk(v: &mut Value, prefix: &str, names: &[&str], value_of: &dyn Fn(&str) -> Option<Value>) -> bool {
    let mut changed = false;
    match v {
        Value::Array(a) => a.iter_mut().for_each(|x| changed |= fill_walk(x, prefix, names, value_of)),
        Value::Object(m) => {
            let keys: Vec<String> = m.keys().cloned().collect();
            for k in keys {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                if names.contains(&path.as_str()) {
                    if m.get(&k).is_some_and(Value::is_null) {
                        changed = true;
                        match value_of(&path) {
                            Some(val) => {
                                m.insert(k, val);
                            }
                            None => {
                                m.shift_remove(&k);
                            }
                        }
                    }
                } else if is_parent_of(&path, names) {
                    if let Some(c) = m.get_mut(&k) {
                        changed |= fill_walk(c, &path, names, value_of);
                    }
                }
            }
        }
        _ => {}
    }
    changed
}

/// The `_source` of document `doc` of `leaf` as loaded with its vectors
/// (`raw` its stored `_source`).
pub fn load<'a>(raw: &'a str, lk: &FieldLookup, leaf: &rustic_core::query::reader::LeafReader<'_>, doc: rustic_core::types::DocId) -> std::borrow::Cow<'a, str> {
    // A synthetic-source residual: its column-backed fields come back first.
    if crate::synthetic::is_residual(raw) {
        let full = crate::synthetic::rehydrate(raw, lk, &crate::synthetic::LeafColumns { leaf, doc }).unwrap_or_default();
        return match rehydrate_stored(&full, lk, &|fid| leaf.vector(fid, doc), &crate::synthetic::LeafColumns { leaf, doc }) {
            Some(s) => s.into(),
            None => full.into(),
        };
    }
    match rehydrate_stored(raw, lk, &|fid| leaf.vector(fid, doc), &crate::synthetic::LeafColumns { leaf, doc }) {
        Some(s) => s.into(),
        None => raw.into(),
    }
}


/// Bound the transient JSON trees, vector rows and growing output before source
/// reconstruction. This is intentionally conservative: each `null` token can
/// expand one vector, including repeated markers inside arrays. The callback
/// receives a total peak bound, not an incremental charge.
pub fn load_guarded<'a, E>(raw: &'a str, lk: &FieldLookup,
    leaf: &rustic_core::query::reader::LeafReader<'_>, doc: rustic_core::types::DocId,
    mut before: impl FnMut(u64) -> std::result::Result<(), E>) -> std::result::Result<std::borrow::Cow<'a, str>, E> {
    if crate::synthetic::is_residual(raw) {
        // The rebuilt fields are bounded by the leaf's columns; the residual
        // by its stored size.
        before((raw.len() as u64).saturating_mul(64).saturating_add(64 << 10))?;
        return Ok(load(raw, lk, leaf, doc));
    }
    if !lk.registry.exclude_source_vectors || !lk.may_have_vector_fields || !raw.contains("null") {
        return Ok(raw.into());
    }
    let mut max_dims = 0u64;
    let mut fields_bytes = 0u64;
    let mut sparse = false;
    for field in lk.fields.values().filter(|field| is_store_backed(field)) {
        if field.declared == EsType::SparseVector {
            // A feature vector: bounded below by its doc values (charged
            // per marker below).
            sparse = true;
            fields_bytes = fields_bytes.saturating_add(field.name.len() as u64).saturating_add(128);
            continue;
        }
        // Valid indexed dense-vector mappings have known dimensions, capped
        // at 4096. Unknown dimensions have no stored vector but budget the cap.
        max_dims = max_dims.max(field.dims.unwrap_or(4096) as u64);
        fields_bytes = fields_bytes.saturating_add(field.name.len() as u64).saturating_add(128);
    }
    if max_dims == 0 && !sparse { return Ok(raw.into()); }
    let markers = raw.as_bytes().windows(4).filter(|bytes| *bytes == b"null").count() as u64;
    let per_marker = max_dims.saturating_mul(128).max(if sparse { 1 << 20 } else { 0 });
    let peak = (raw.len() as u64).saturating_mul(512)
        .saturating_add(markers.saturating_mul(per_marker))
        .saturating_add(fields_bytes.saturating_mul(4)).saturating_add(4096);
    before(peak)?;
    Ok(load(raw, lk, leaf, doc))
}


/// A filtered `_source` (`_source.includes` / `excludes`) as ES renders it:
/// when the load patched in a vector that the filter keeps, ES filters the
/// patched map (`SourceFilter.filterMap` -> `XContentMapValues.filter`),
/// which builds Java `HashMap`s, so every object comes out in `HashMap`
/// order; otherwise it filters the stored bytes, in source order.
pub fn finish_filtered(out: Value, lk: &FieldLookup) -> Value {
    if !lk.may_have_vector_fields { return out; }
    let names: Vec<&str> = lk.fields.values().filter(|f| is_synthetic_vector(f)).map(|f| f.name.as_str()).collect();
    if !names.is_empty() && holds_any(&out, "", &names) {
        java_ordered(out, "", &names)
    } else {
        out
    }
}

/// [`crate::search::filter_source`] then [`finish_filtered`].
pub fn filter_loaded(src: &Value, includes: &[String], excludes: &[String], lk: &FieldLookup) -> Value {
    let out = crate::search::filter_source(src, includes, excludes);
    // An empty filter (`SourceFilter.empty`) leaves the source as loaded.
    if includes.is_empty() && excludes.is_empty() {
        return out;
    }
    finish_filtered(out, lk)
}

fn holds_any(v: &Value, prefix: &str, names: &[&str]) -> bool {
    match v {
        Value::Array(a) => a.iter().any(|x| holds_any(x, prefix, names)),
        Value::Object(m) => m.iter().any(|(k, c)| {
            let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            (names.contains(&path.as_str()) && !c.is_null()) || (is_parent_of(&path, names) && holds_any(c, &path, names))
        }),
        _ => false,
    }
}

/// `v` with every object's keys in Java `HashMap` iteration order.
/// The vector values themselves (`names`, at their paths) keep their order:
/// ES patches in a sparse vector's tokens in index order.
fn java_ordered(v: Value, prefix: &str, names: &[&str]) -> Value {
    match v {
        Value::Array(a) => Value::Array(a.into_iter().map(|x| java_ordered(x, prefix, names)).collect()),
        Value::Object(m) => {
            let order: Vec<String> = crate::jsonpos::java_hashmap_order(&m).into_iter().map(|(k, _)| k.clone()).collect();
            let mut m = m;
            let mut out = Map::with_capacity(order.len());
            for k in order {
                if let Some(x) = m.shift_remove(&k) {
                    let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    let x = if names.contains(&path.as_str()) { x } else { java_ordered(x, &path, names) };
                    out.insert(k, x);
                }
            }
            Value::Object(out)
        }
        x => x,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn excludes_by_version() {
        assert!(excludes_source_vectors(|k| (k == "index.version.created").then(|| IndexVersion::CURRENT.to_string())));
        assert!(!excludes_source_vectors(|k| (k == "index.version.created").then(|| IndexVersion::LATEST_8X.to_string())));
        assert!(!excludes_source_vectors(|k| (k == "index.mapping.exclude_source_vectors").then(|| "false".to_string())));
        let _ = json!({});
    }

    #[test]
    fn strip_raw_keeps_other_members_verbatim() {
        let names = ["emb", "obj.v", "n.v"];
        let raw = r#"{"a" : 1.50,"emb":[1.0,2.0],"obj":{"x":1,"v":[3.0]},"n":[{"v":[1.0]},{"y":2}],"s":"q\"x"}"#;
        assert_eq!(strip_raw(raw, &names).unwrap(), r#"{"a":1.50,"obj":{"x":1},"n":[{},{"y":2}],"s":"q\"x"}"#);
        // Dotted top-level key naming the field.
        assert_eq!(strip_raw(r#"{"obj.v":[1.0],"b":true}"#, &names).unwrap(), r#"{"b":true}"#);
        // Nothing to remove: unchanged.
        assert_eq!(strip_raw(r#"{"a":1,"obj":{"x":1}}"#, &names), None);
        let mut v: Value = serde_json::from_str(raw).unwrap();
        strip_walk(&mut v, "", &names);
        assert_eq!(serde_json::from_str::<Value>(&strip_raw(raw, &names).unwrap()).unwrap(), v);
    }
}
