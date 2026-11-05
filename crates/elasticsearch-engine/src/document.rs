//! `_source` JSON → core [`Document`], driven by the mapping.
//!
//! One pass over the parsed source emits a core value per mapped leaf
//! (multi-fields, `copy_to`, `null_value`, `ignore_above`, `normalizer`,
//! numeric coercion and range checks, date formats, ip, geo_point,
//! dense_vector). Unmapped fields under `dynamic: true` are collected as
//! mapping additions instead; the caller merges them and re-runs the pass.

use crate::analysis::IndexAnalyzer;
use crate::error::{EsError, Result};
use crate::mapping::{Dynamic, EsType, FieldLookup, Mapper, Mappings, ResolvedField};
use crate::values;
use rustic_core::types::{Document, FieldValue};
use serde_json::{Map, Value};
use std::collections::HashSet;

/// Outcome of parsing one document.
pub enum ParseOutcome {
    Doc(ParsedDoc),
    /// Dynamic mapping additions needed before the document can be indexed.
    NeedsMapping(Vec<(String, Mapper)>),
}

#[derive(Debug)]
pub struct ParsedDoc {
    pub doc: Document,
    /// Fields skipped by `ignore_above` / `ignore_malformed` (`_ignored`).
    pub ignored: Vec<String>,
    /// The `_source` to store when it differs from the one received: the
    /// canonical form of a synthetic-source index (see [`crate::synthetic`]).
    pub source: Option<String>,
}

struct Ctx<'a> {
    id: &'a str,
    max_regex_length: usize,
    mappings: &'a Mappings,
    lookup: &'a FieldLookup,
    doc: Document,
    ignored: Vec<String>,
    pending: Vec<(String, Mapper)>,
    pending_paths: HashSet<String>,
    /// Emitting a `copy_to` target (copy_to is not followed recursively).
    copying: bool,
    /// Dynamic fields dropped by `ignore_dynamic_beyond_limit` (`_ignored`).
    skip: Option<&'a HashSet<String>>,
    /// Indexing the values an index-time script emitted.
    scripting: bool,
    /// Objects parsed inside arrays so far (`index.mapping.array_objects.limit`).
    array_objects: usize,
    /// `doc_values.multi_value: false` fields that got a value.
    single_seen: HashSet<String>,
    /// `doc_values.nullability: false` fields that got a value (current
    /// Lucene document).
    satisfied: HashSet<String>,
}

/// A value as ES prints it in error previews (Java `Map` / `List` `toString`).
pub fn preview(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, x)| format!("{k}={}", preview(x))).collect::<Vec<_>>().join(", ")),
        Value::Array(a) => format!("[{}]", a.iter().map(preview).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

/// Bulk `dynamic_template_params`: field path -> param name -> value.
pub type TemplateParams = std::collections::HashMap<String, std::collections::HashMap<String, String>>;

/// Remove dotted `path` from `v` (nested objects or dotted keys, inside arrays too).
fn remove_path(v: &mut Value, path: &str) {
    match v {
        Value::Array(a) => a.iter_mut().for_each(|x| remove_path(x, path)),
        Value::Object(o) => {
            if o.remove(path).is_some() {
                return;
            }
            let mut idx = 0;
            while let Some(i) = path[idx..].find('.') {
                let (head, tail) = (&path[..idx + i], &path[idx + i + 1..]);
                if let Some(child) = o.get_mut(head) {
                    remove_path(child, tail);
                    if child.as_object().is_some_and(|c| c.is_empty()) {
                        o.remove(head);
                    }
                    return;
                }
                idx += i + 1;
            }
        }
        _ => {}
    }
}

/// Marks a pending dynamic addition as a runtime field (its definition).
const RUNTIME_MARK: &str = "\0runtime";

thread_local! {
    /// Bulk `dynamic_templates` of the document being written on this
    /// thread: field path -> dynamic template name.
    static REQUESTED_TEMPLATES: std::cell::RefCell<Option<std::collections::HashMap<String, String>>> = const { std::cell::RefCell::new(None) };
    /// Bulk `dynamic_template_params` of the document being written.
    static TEMPLATE_PARAMS: std::cell::RefCell<Option<TemplateParams>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` (a write of one document) with the bulk item's
/// `dynamic_templates` (path -> template name) in effect.
pub fn with_dynamic_templates<R>(templates: std::collections::HashMap<String, String>, f: impl FnOnce() -> R) -> R {
    let prev = REQUESTED_TEMPLATES.with(|t| t.replace(Some(templates)));
    let r = f();
    REQUESTED_TEMPLATES.with(|t| *t.borrow_mut() = prev);
    r
}

/// Run `f` with the bulk item's `dynamic_template_params` (field path ->
/// `{{param}}` values substituted into the dynamic template's mapping).
pub fn with_dynamic_template_params<R>(params: TemplateParams, f: impl FnOnce() -> R) -> R {
    let prev = TEMPLATE_PARAMS.with(|t| t.replace(Some(params)));
    let r = f();
    TEMPLATE_PARAMS.with(|t| *t.borrow_mut() = prev);
    r
}

/// The `dynamic_template_params` for the field at `path` (empty if none).
pub fn template_params(path: &str) -> std::collections::HashMap<String, String> {
    TEMPLATE_PARAMS.with(|t| t.borrow().as_ref().and_then(|m| m.get(path).cloned()).unwrap_or_default())
}

fn requested_template(path: &str) -> Option<String> {
    REQUESTED_TEMPLATES.with(|t| t.borrow().as_ref().and_then(|m| m.get(path).cloned()))
}

/// Parse `source` for indexing.
pub fn parse_document(source: &Value, id: &str, mappings: &Mappings, lookup: &FieldLookup) -> Result<ParseOutcome> {
    parse_document_skip(source, id, mappings, lookup, None)
}

/// [`parse_document`] treating the dynamic fields in `skip` as ignored.
pub fn parse_document_skip(source: &Value, id: &str, mappings: &Mappings, lookup: &FieldLookup, skip: Option<&HashSet<String>>) -> Result<ParseOutcome> {
    parse_document_skip_with_regex_limit(source, id, mappings, lookup, skip, 1000)
}

fn parse_document_skip_with_regex_limit(source: &Value, id: &str, mappings: &Mappings, lookup: &FieldLookup, skip: Option<&HashSet<String>>, max_regex_length: usize) -> Result<ParseOutcome> {
    let obj = source.as_object().ok_or_else(|| EsError::document_parsing("[1:1] failed to parse: Root document must be an object"))?;
    let mut ctx = Ctx { id, max_regex_length, mappings, lookup, doc: Document::new(), ignored: Vec::new(), pending: Vec::new(), pending_paths: HashSet::new(), copying: false, skip, scripting: false, array_objects: 0, single_seen: HashSet::new(), satisfied: HashSet::new() };
    if lookup.root_enabled {
        let root_dyn = lookup.root_dynamic.unwrap_or(Dynamic::True);
        // `_inference_fields`: semantic_text inference results the document carries.
        crate::inference::semantic::with_supplied(obj, || -> Result<()> {
            ctx.walk_object(obj, "", root_dyn)?;
            ctx.emit_inference()
        })?;
    }
    // ES enforces the `nullability=false` fields of the mapping the
    // document is parsed against (not the dynamic fields it introduces,
    // which carry a value), and a document that fails it contributes no
    // dynamic mapping update: check before asking for new fields, so a
    // rejected document never grows the mapping (`other` in
    // `{"other": "x"}` missing a required `kw` would otherwise become a
    // required field of its own under `index.mapping.doc_values.nullability:
    // false`).
    ctx.enforce_required(None)?;
    if !ctx.pending.is_empty() {
        return Ok(ParseOutcome::NeedsMapping(ctx.pending));
    }
    if lookup.has_index_scripts && !IN_INDEX_SCRIPT.with(|c| c.get()) {
        ctx.run_index_scripts(source)?;
    }
    crate::fieldtypes::index::emit_size(source, lookup, &mut ctx.doc);
    crate::suggest::index::apply_contexts(source, lookup, &mut ctx.doc)?;
    // Shape doc values are written here, in parallel across a bulk, not
    // under the indexing lock.
    if !ctx.doc.shape_triangles.is_empty() {
        ctx.doc.encode_shape_values();
    }
    // A bulk parses all of its documents before any is staged: return each
    // one's unclaimed admission chunk now, not when it is staged.
    ctx.doc.settle_allocation();
    Ok(ParseOutcome::Doc(ParsedDoc { doc: ctx.doc, ignored: ctx.ignored, source: None }))
}

/// Pending-mapping marker: the index needs its `_doc_count` column.
const DOC_COUNT_MARK: &str = "\u{0}doc_count";

thread_local! {
    /// `index.mapping.array_objects.limit` of the parse in progress.
    static ARRAY_OBJECTS_LIMIT: std::cell::Cell<usize> = const { std::cell::Cell::new(50_000) };
    /// Parsing the one-document index an index-time script reads `doc[...]` from.
    static IN_INDEX_SCRIPT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Compiled index-time scripts, by field and context, with their spec.
    static INDEX_SCRIPTS: std::cell::RefCell<std::collections::HashMap<(String, &'static str), (Value, std::sync::Arc<crate::script::CompiledScript>)>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

impl<'a> Ctx<'a> {
    fn scalar_text(&mut self,value:&Value)->Result<Option<String>> {
        let display:&dyn std::fmt::Display=match value {Value::String(v)=>v,Value::Number(v)=>v,Value::Bool(v)=>v,_=>return Ok(None)};
        struct Count(usize);
        impl std::fmt::Write for Count {fn write_str(&mut self,s:&str)->std::fmt::Result{self.0=self.0.checked_add(s.len()).ok_or(std::fmt::Error)?;Ok(())}}
        use std::fmt::Write;
        let mut count=Count(0);write!(&mut count,"{display}").map_err(|_|EsError::new(429,"circuit_breaking_exception","scalar text size overflow"))?;
        self.doc.admit_allocation(count.0)?;
        let mut out=String::new();out.try_reserve_exact(count.0).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
        write!(&mut out,"{display}").map_err(|e|EsError::internal(e.to_string()))?;
        Ok(Some(out))
    }

    // Emit leaves directly: no unbounded intermediate Vec<String> and no
    // duplicate root payload. Each scalar/keyed output is admitted first.
    fn flattened_leaves(&mut self, value: &Value, root: Option<u16>, keyed: Option<(u16, &str)>) -> Result<()> {
        match value {
            Value::Object(map) => { for value in map.values() { self.flattened_leaves(value, root, keyed)?; } }
            Value::Array(values) => { for value in values { self.flattened_leaves(value, root, keyed)?; } }
            Value::Null => {}
            _ => {
                let Some(text) = self.scalar_text(value)? else { return Ok(()); };
                if let Some((id, key)) = keyed {
                    let bytes = key.len().checked_add(1).and_then(|n| n.checked_add(text.len()))
                        .ok_or_else(|| EsError::new(429, "circuit_breaking_exception", "flattened output overflow"))?;
                    self.doc.admit_allocation(bytes)?;
                    let mut output = String::new();
                    output.try_reserve_exact(bytes).map_err(|e| EsError::new(429, "circuit_breaking_exception", e.to_string()))?;
                    output.push_str(key); output.push('\0'); output.push_str(&text);
                    // Preserve the root-before-keyed field order.
                    if let Some(root) = root { self.doc.try_push_field((root, FieldValue::Keyword(text)))?; }
                    self.doc.try_push_field((id, FieldValue::Keyword(output)))?;
                } else if let Some(root) = root {
                    self.doc.try_push_field((root, FieldValue::Keyword(text)))?;
                }
            }
        }
        Ok(())
    }

    /// `DocumentParserContext.incrementAndCheckObjectArrayElementLimit`:
    /// one object element parsed inside an array.
    fn count_array_object(&mut self) -> Result<()> {
        self.array_objects += 1;
        let limit = ARRAY_OBJECTS_LIMIT.with(|c| c.get());
        if self.array_objects > limit {
            return Err(EsError::document_parsing(format!(
                "[1:1] The total number of objects across all arrays in the document has exceeded the allowed limit of [{limit}]. This limit can be set by changing the [index.mapping.array_objects.limit] index level setting."
            )));
        }
        Ok(())
    }

    /// `_doc_count` (`DocCountFieldMapper.parseCreateField`): a positive
    /// integer, kept as a doc value bucket aggregations count by.
    fn doc_count(&mut self, v: &Value) -> Result<()> {
        let n = match v {
            Value::Null => return Ok(()),
            Value::Array(_) => return Err(EsError::document_parsing("[1:1] Arrays are not allowed for field [_doc_count].")),
            Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
            Value::String(s) => s.trim().parse::<i64>().ok(),
            _ => None,
        };
        let Some(n) = n else {
            return Err(EsError::document_parsing(format!("[1:1] Field [_doc_count] must be a positive integer. Value [{}] is not allowed.", preview(v))));
        };
        if n <= 0 {
            return Err(EsError::document_parsing(format!("[1:1] Field [_doc_count] must be a positive integer. Value [{n}] is not allowed.")));
        }
        match self.lookup.doc_count_id {
            Some(id) => self.doc.try_push_field((id, FieldValue::I64(n)))?,
            None => {
                if self.pending_paths.insert(crate::mapping::DOC_COUNT_FIELD_NAME.to_string()) {
                    let mut fm = crate::mapping::FieldMapping::new(EsType::Long);
                    fm.params.insert(DOC_COUNT_MARK.into(), Value::Bool(true));
                    self.pending.push((crate::mapping::DOC_COUNT_FIELD_NAME.to_string(), Mapper::Field(fm)));
                }
            }
        }
        Ok(())
    }

    /// Fields with an index-time `script` (`FieldMapper` `executeScript`):
    /// run each against the document (`params._source`, and `doc[...]` of
    /// a one-document index of it) and index the values it emits.
    fn run_index_scripts(&mut self, source: &Value) -> Result<()> {
        let mut fields: Vec<ResolvedField> = self.lookup.fields.values().filter(|f| f.script.is_some()).map(|f| (**f).clone()).collect();
        fields.sort_by(|a, b| a.name.cmp(&b.name));
        let lookup = std::sync::Arc::new(self.lookup.clone());
        IN_INDEX_SCRIPT.with(|c| c.set(true));
        let mem = crate::script::execute_api::build_memory_index_sealed(self.lookup, source, false);
        IN_INDEX_SCRIPT.with(|c| c.set(false));
        let mem = mem?;
        let leaves = mem.leaves();
        let src_val = crate::script::Val::from_json(source);
        for f in fields {
            let spec = f.script.clone().unwrap_or(Value::Null);
            let context = match f.ty { crate::mapping::EsType::Double => crate::script::ScriptContext::RuntimeDouble, _ => crate::script::ScriptContext::RuntimeLong };
            // Cached per field name and context, checked against the spec:
            // not keyed by the spec's JSON text, which cost a serialization
            // per field of every document.
            let cached = INDEX_SCRIPTS.with(|c| c.borrow().get(&(f.name.clone(), context.name())).filter(|(s, _)| *s == spec).map(|(_, cs)| cs.clone()));
            let cs = match cached {
                Some(cs) => cs,
                None => {
                    let cs = std::sync::Arc::new(crate::script::compile_json(&spec, context, None)?);
                    INDEX_SCRIPTS.with(|c| c.borrow_mut().insert((f.name.clone(), context.name()), (spec.clone(), cs.clone())));
                    cs
                }
            };
            let mut exec = cs.executor(Some(lookup.clone()), None);
            exec.rt.doc.source_json = Some(source.clone());
            exec.set_source(Some(src_val.clone()));
            if let Some(leaf) = leaves.first() {
                // SAFETY: `mem` outlives `exec`, dropped at the end of this iteration.
                unsafe { exec.bind_leaf(leaf) };
                exec.set_doc(0);
            }
            let context = match f.declared {
                EsType::Keyword => "keyword_field",
                EsType::Boolean => "boolean_field",
                EsType::Date | EsType::DateNanos => "date_field",
                EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => "double_field",
                EsType::Ip => "ip_field",
                EsType::GeoPoint => "geo_point_field",
                _ => "long_field",
            };
            let out = exec.run().map_err(|e| cs.error(&e)).and_then(|_| crate::script::execute_api::emitted_result(context, exec.take_emitted()).map_err(|e| cs.error(&e)));
            let vals = match out {
                Ok(v) => v,
                Err(e) if f.script_error_continue => {
                    let _ = e;
                    self.ignored.push(f.name.clone());
                    continue;
                }
                Err(e) => {
                    return Err(EsError::document_parsing(format!("[1:1] Error executing script on field [{}]", f.name)).caused_by(e));
                }
            };
            self.scripting = true;
            let r = vals.as_array().cloned().unwrap_or_default().iter().try_for_each(|v| self.emit_field(&f, v));
            self.scripting = false;
            r?;
        }
        Ok(())
    }

    fn walk_object(&mut self, obj: &Map<String, Value>, path: &str, dynamic: Dynamic) -> Result<()> {
        for (k, v) in obj {
            if k.is_empty() || k.trim().is_empty() {
                return Err(EsError::document_parsing("[1:1] failed to parse: field name cannot be an empty string")
                    .caused_by(EsError::illegal_argument("field name cannot be an empty string")));
            }
            if k.starts_with('_') && path.is_empty() && matches!(k.as_str(), "_id" | "_index" | "_routing" | "_source" | "_version" | "_seq_no" | "_primary_term" | "_ignored" | "_field_names") {
                return Err(EsError::document_parsing(format!(
                    "[1:1] Field [{k}] is a metadata field and cannot be added inside a document. Use the index API request parameters."
                )));
            }
            if path.is_empty() && crate::inference::semantic::is_inference_key(k) {
                continue;
            }
            if path.is_empty() && k == crate::mapping::DOC_COUNT_FIELD_NAME {
                self.doc_count(v)?;
                continue;
            }
            // A root key is its own full path: no per-field copy of it.
            let joined;
            let full: &str = if path.is_empty() { k.as_str() } else {
                // Built directly: `format!` costs several times the copy
                // for every nested key of every document.
                let mut s = String::with_capacity(path.len() + 1 + k.len());
                s.push_str(path);
                s.push('.');
                s.push_str(k);
                joined = s;
                &joined
            };
            let leaf = k.rsplit('.').next().unwrap_or(k);
            self.value(full, leaf, path, v, dynamic)?;
        }
        Ok(())
    }

    fn value(&mut self, full: &str, name: &str, parent: &str, v: &Value, dynamic: Dynamic) -> Result<()> {
        // Columnar root: `prefix_properties` decide (`enabled: false`
        // prefixes drop their subtree; the deepest `dynamic` applies).
        let mut dynamic = dynamic;
        if !self.mappings.root.prefix.is_empty() {
            let mut best: Option<(usize, Dynamic)> = None;
            for (p, pp) in &self.mappings.root.prefix {
                let under = full.len() > p.len() && full.starts_with(p.as_str()) && full.as_bytes()[p.len()] == b'.';
                if pp.enabled == Some(false) && (under || full == p) {
                    return Ok(());
                }
                if let (true, Some(d)) = (under, pp.dynamic) {
                    if best.is_none_or(|(l, _)| p.len() > l) {
                        best = Some((p.len(), d));
                    }
                }
            }
            if let Some((_, d)) = best {
                dynamic = d;
            }
        }
        if self.lookup.fields.get(full).is_some_and(|f| f.ty == EsType::Alias) && !self.lookup.passthrough_alias.contains(full) {
            return Err(EsError::document_parsing(format!("[1:1] Cannot write to a field alias [{full}].")));
        }
        // `self.lookup` is a `&'a` borrow independent of `self`: emit from it
        // directly instead of cloning the field's Arc (two atomic RMWs per
        // field of every document).
        let lookup: &'a FieldLookup = self.lookup;
        if let Some(f) = lookup.get(full) {
            return self.emit_field(f, v);
        }
        if let Some(info) = self.lookup.objects.get(full).copied() {
            if !info.enabled {
                return Ok(());
            }
            if info.nested {
                return self.nested_value(full, name, v, info);
            }
            return match v {
                Value::Object(m) => self.walk_object(m, full, info.dynamic),
                Value::Array(a) => {
                    for x in a {
                        if x.is_object() {
                            self.count_array_object()?;
                        }
                        self.value(full, name, parent, x, dynamic)?;
                    }
                    Ok(())
                }
                Value::Null => Ok(()),
                other => Err(EsError::document_parsing(format!(
                    "[1:1] object mapping for [{full}] tried to parse field [{name}] as object, but found a concrete value"
                ))
                .with("preview", preview(other))),
            };
        }
        // Unmapped. A dotted name walks through its parent objects: the
        // deepest mapped one decides (`dynamic`, `enabled`).
        let mut dynamic = dynamic;
        let rel = if parent.is_empty() { full } else { full.strip_prefix(parent).map(|r| r.trim_start_matches('.')).unwrap_or(full) };
        if rel.contains('.') {
            let mut end = rel.len();
            while let Some(i) = rel[..end].rfind('.') {
                end = i;
                let p = if parent.is_empty() { rel[..end].to_string() } else { format!("{parent}.{}", &rel[..end]) };
                if let Some(info) = self.lookup.objects.get(&p) {
                    if !info.enabled {
                        return Ok(());
                    }
                    dynamic = info.dynamic;
                    break;
                }
            }
        }
        match dynamic {
            // Under `subobjects: false` the dotted leaves below an unmapped object may still be mapped.
            Dynamic::False => match v {
                Value::Object(o) if self.mappings.no_subobjects_at(parent) => self.walk_object(o, full, dynamic),
                _ => Ok(()),
            },
            Dynamic::Runtime => self.dynamic(full, name, v, dynamic),
            Dynamic::Strict => {
                if v.is_null() {
                    return Ok(());
                }
                Err(EsError::strict_dynamic(name, parent))
            }
            Dynamic::True => self.dynamic(full, name, v, dynamic),
        }
    }

    /// A `nested` object (or array of them): every object becomes a child
    /// document of the current one (indexed as one block with it, children
    /// first); its fields are not indexed into the parent unless
    /// `include_in_parent` / `include_in_root`.
    fn nested_value(&mut self, full: &str, name: &str, v: &Value, info: crate::mapping::ObjectInfo) -> Result<()> {
        match v {
            Value::Array(a) => {
                for x in a {
                    if x.is_object() {
                        self.count_array_object()?;
                    }
                    self.nested_value(full, name, x, info)?;
                }
                Ok(())
            }
            Value::Object(m) => {
                let parent = std::mem::take(&mut self.doc);
                let parent_satisfied = std::mem::take(&mut self.satisfied);
                let r = self.walk_object(m, full, info.dynamic).and_then(|_| self.enforce_required(Some(full)));
                self.satisfied = parent_satisfied;
                let child = std::mem::replace(&mut self.doc, parent);
                r?;
                if info.include_in_parent || info.include_in_root {
                    self.doc.try_extend_fields_from(&child)?;
                }
                let mut child = child;
                crate::stored::attach_nested(&mut child, m, full, self.lookup);
                child.settle_allocation();
                self.doc.try_add_nested(full, child)?;
                Ok(())
            }
            Value::Null => Ok(()),
            other => Err(EsError::document_parsing(format!(
                "[1:1] object mapping for [{full}] tried to parse field [{name}] as object, but found a concrete value"
            ))
            .with("preview", preview(other))),
        }
    }

    fn dynamic(&mut self, full: &str, name: &str, v: &Value, dynamic: Dynamic) -> Result<()> {
        if self.skip.map(|s| s.contains(full)).unwrap_or(false) {
            // A dotted name would have created its parent objects first: ES ignores the
            // outermost object that is not mapped yet (`host` for `host.name`).
            let mut name = full.to_string();
            let mut prefix = String::new();
            for part in full.split('.') {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(part);
                if self.lookup.get(&prefix).is_none() && !self.lookup.objects.contains_key(&prefix) {
                    name = prefix.clone();
                    break;
                }
            }
            if !self.ignored.iter().any(|x| *x == name) {
                self.ignored.push(name);
            }
            return Ok(());
        }
        let kind = match v {
            Value::Null => return Ok(()),
            Value::Array(a) => {
                if a.len() >= 128 && a.len() <= 4096 && a.iter().all(|x| x.is_f64()) {
                    "dense_vector"
                } else {
                    match a.iter().find(|x| !x.is_null()) {
                        Some(first) => {
                            if first.is_array() {
                                // nested arrays flatten
                                for x in a {
                                    self.dynamic(full, name, x, dynamic)?;
                                }
                                return Ok(());
                            }
                            return {
                                for x in a {
                                    if x.is_object() {
                                        self.count_array_object()?;
                                    }
                                    if !x.is_null() {
                                        self.dynamic(full, name, x, dynamic)?;
                                    }
                                }
                                Ok(())
                            };
                        }
                        None => return Ok(()),
                    }
                }
            }
            Value::Object(_) => "object",
            Value::Bool(_) => "boolean",
            Value::Number(n) => {
                if n.is_f64() {
                    "double"
                } else {
                    "long"
                }
            }
            Value::String(s) => {
                if self.mappings.date_detection() && crate::date::may_be_date(s) && crate::date::detect_date(s, &self.mappings.dynamic_date_formatters()) {
                    "date"
                } else if self.mappings.numeric_detection() && s.trim().parse::<i64>().is_ok() {
                    "long"
                } else if self.mappings.numeric_detection() && s.trim().parse::<f64>().is_ok() {
                    "double"
                } else {
                    "string"
                }
            }
        };
        // Under `subobjects: false` a dynamic object is not mapped: its leaves get dotted names.
        if kind == "object" {
            let parent = full.strip_suffix(name).unwrap_or("").trim_end_matches('.');
            // ...unless a requested template maps the object to a field
            // (`aggregate_metric_double`, `histogram`, ...).
            let to_field = match requested_template(full) {
                Some(req) => matches!(self.mappings.dynamic_mapper_named(name, full, "object", Some(&req), &self.lookup.registry)?, Some(Mapper::Field(_))),
                None => self.mappings.no_subobjects_at(parent) && matches!(self.mappings.dynamic_mapper(name, full, "object", &self.lookup.registry)?, Some(Mapper::Field(_))),
            };
            if !to_field && self.mappings.no_subobjects_at(parent) {
                if let Value::Object(o) = v {
                    return self.walk_object(o, full, dynamic);
                }
            }
        }
        // A dotted name creates its parent objects first: a requested
        // (bulk `dynamic_templates`) template for an unmapped parent applies
        // to it, and a leaf mapping there cannot take the object value.
        if full.contains('.') {
            let parts: Vec<&str> = full.split('.').collect();
            for i in 1..parts.len() {
                let p = parts[..i].join(".");
                if self.lookup.get(&p).is_some() || self.lookup.objects.contains_key(&p) || self.pending_paths.contains(&p) {
                    continue;
                }
                let Some(req) = requested_template(&p) else { continue };
                if let Some(Mapper::Field(fm)) = self.mappings.dynamic_mapper_named(parts[i - 1], &p, "object", Some(&req), &self.lookup.registry)? {
                    let mut inner = Value::Null;
                    for k in parts[i..].iter().rev() {
                        let mut o = Map::new();
                        o.insert(k.to_string(), if inner.is_null() { v.clone() } else { inner });
                        inner = Value::Object(o);
                    }
                    return Err(EsError::document_parsing(format!(
                        "[1:1] failed to parse field [{p}] of type [{}] in document with id '{}'. Preview of field's value: '{}'",
                        fm.ty.name(),
                        self.id,
                        preview(&inner)
                    )));
                }
            }
        }
        // Runtime fields: `dynamic: runtime`, or a matching `runtime` template.
        if kind != "object" && kind != "dense_vector" {
            if self.mappings.has_runtime(full) || self.pending_paths.contains(full) {
                return Ok(());
            }
            let requested = requested_template(full);
            // The first matching template decides: a `runtime` one (or none,
            // under `dynamic: runtime`) adds a runtime field, a `mapping` one a
            // concrete field.
            if let Some(def) = self.mappings.dynamic_runtime(name, full, kind, requested.as_deref(), dynamic == Dynamic::Runtime) {
                let mut fm = crate::mapping::FieldMapping::new(EsType::Keyword);
                fm.params.insert(RUNTIME_MARK.into(), def);
                self.pending_paths.insert(full.to_string());
                self.pending.push((full.to_string(), Mapper::Field(fm)));
                return Ok(());
            }
        } else if kind == "object" && dynamic == Dynamic::Runtime {
            // No object mappers under `dynamic: runtime`: the leaves get dotted runtime names.
            if let Value::Object(o) = v {
                return self.walk_object(o, full, dynamic);
            }
        }
        if !self.pending_paths.contains(full) {
            let requested = requested_template(full);
            if let Some(mut m) = self.mappings.dynamic_mapper_named(name, full, kind, requested.as_deref(), &self.lookup.registry)? {
                // A date detected through custom `dynamic_date_formats` keeps the matching format.
                // The built-in `yyyy/MM/dd [HH:mm:ss]` default is kept too (`strict_date_optional_time` is the mapper default).
                if let (Mapper::Field(fm), Value::String(sv)) = (&mut m, v) {
                    if fm.ty == EsType::Date && !fm.params.contains_key("format") {
                        let default_fmts = vec!["strict_date_optional_time".to_string(), "yyyy/MM/dd HH:mm:ss||yyyy/MM/dd".to_string()];
                        let fmts = self.mappings.dynamic_date_formats.as_ref().unwrap_or(&default_fmts);
                        if let Some(f) = fmts.iter().find(|f| crate::date::DateFormatter::new(f).is_ok_and(|d| d.parse(sv).is_ok())) {
                            if self.mappings.dynamic_date_formats.is_some() {
                                fm.params.insert("format".into(), Value::String(f.clone()));
                            } else if f != "strict_date_optional_time" {
                                // ES reports the built-in slash format with `epoch_millis` and an empty locale.
                                fm.params.insert("format".into(), Value::String(format!("{f}||epoch_millis")));
                                fm.params.insert("locale".into(), Value::String(String::new()));
                            }
                        }
                    }
                }
                // Leaves of a pass-through object with `time_series_dimension` are dimensions.
                if let Mapper::Field(fm) = &mut m {
                    // The lookup lists every pass-through object of this
                    // mapping: without one, no mapping walk per new field.
                    if !fm.params.contains_key("time_series_dimension") && !self.lookup.passthrough.is_empty() && self.mappings.under_passthrough_dimension(full) {
                        // Dimension strings are keywords (text cannot be a dimension).
                        if fm.ty == EsType::Text {
                            *fm = crate::mapping::FieldMapping::new(EsType::Keyword);
                        }
                        fm.params.insert("time_series_dimension".into(), Value::Bool(true));
                    }
                }
                let is_obj = matches!(m, Mapper::Object(_));
                // A templated object may set its own `dynamic` for its children.
                let child_dyn = match &m {
                    Mapper::Object(om) => om.dynamic.unwrap_or(dynamic),
                    _ => dynamic,
                };
                self.pending_paths.insert(full.to_string());
                self.pending.push((full.to_string(), m));
                if is_obj {
                    if let Value::Object(o) = v {
                        return self.walk_object(o, full, child_dyn);
                    }
                }
            }
        } else if let Value::Object(o) = v {
            return self.walk_object(o, full, dynamic);
        }
        Ok(())
    }

    /// Field types parsed by [`crate::fieldtypes::index`].
    fn emit_ext(&mut self, f: &ResolvedField, v: &Value) -> Result<()> {
        let scalars = {
            let mut e = crate::fieldtypes::index::Emit {
                id: self.id,
                max_regex_length: self.max_regex_length,                lookup: self.lookup,
                doc: &mut self.doc,
                ignored: &mut self.ignored,
                pending: &mut self.pending,
                pending_paths: &mut self.pending_paths,
                now: chrono::Utc::now().timestamp_millis(),
            };
            crate::fieldtypes::index::emit(f, v, &mut e)?
        };
        if f.declared == EsType::Completion && !f.multi_fields.is_empty() {
            // Completion sub-fields get the whole entry, others each input.
            for sub in &f.multi_fields {
                let Some(sf) = self.lookup.fields.get(sub).cloned() else { continue };
                if sf.declared == EsType::Completion {
                    self.emit_field(&sf, v)?;
                } else {
                    for i in crate::suggest::index::completion_inputs(v) {
                        self.emit_scalar(&sf, &Value::String(i))?;
                    }
                }
            }
        }
        for x in scalars {
            self.after(f, &x)?;
        }
        Ok(())
    }

    /// Record `name` in `_field_names` (once per doc): the field has a value
    /// but leaves no doc-values / postings presence signal.
    fn mark_present(&mut self, name: &str) -> Result<()> {
        let already = self.doc.fields.iter().any(|(id, v)| *id == crate::mapping::FIELD_NAMES_FIELD && v.as_text() == Some(name));
        if !already {
            self.doc.try_add_keyword(crate::mapping::FIELD_NAMES_FIELD,name)?;
        }
        Ok(())
    }

    /// The flattened part of a value with mapped sub-fields.
    fn emit_flattened_rest(&mut self, f: &ResolvedField, v: &Value) -> Result<()> {
        self.emit_ext(f, v)
    }

    fn emit_field(&mut self, f: &ResolvedField, v: &Value) -> Result<()> {
        // `flattened` mapped sub-fields: their values index into the
        // sub-fields; the rest stays flattened.
        if f.declared == EsType::Flattened && f.flattened_key().is_none() {
            if let Some(props) = f.ext.as_ref().and_then(|e| e.params.get("properties")).and_then(Value::as_object).filter(|p| !p.is_empty()) {
                let names: Vec<String> = props.keys().cloned().collect();
                let mut rest = v.clone();
                for n in &names {
                    let vals: Vec<Value> = match v {
                        Value::Object(_) => crate::highlight::source_values(v, n).into_iter().cloned().collect(),
                        Value::Array(a) => a.iter().flat_map(|x| crate::highlight::source_values(x, n).into_iter().cloned().collect::<Vec<_>>()).collect(),
                        _ => Vec::new(),
                    };
                    remove_path(&mut rest, n);
                    if let Some(sf) = self.lookup.fields.get(&format!("{}.{n}", f.name)).cloned() {
                        for x in vals {
                            self.emit_field(&sf, &x)?;
                        }
                    }
                }
                let empty = matches!(&rest, Value::Object(o) if o.is_empty());
                if empty {
                    return Ok(());
                }
                return self.emit_flattened_rest(f, &rest);
            }
        }
        if f.declared == EsType::PatternText {
            if let Value::Array(a) = v {
                if a.iter().filter(|x| !x.is_null()).count() > 1 {
                    let cause = EsError::illegal_argument(format!("Multiple values are not allowed for field [{}].", f.name));
                    return Err(EsError::document_parsing(format!(
                        "[1:1] failed to parse field [{}] of type [pattern_text] in document with id '{}'. Preview of field's value: '{}'",
                        f.name,
                        self.id,
                        preview(v)
                    ))
                    .caused_by(cause));
                }
            }
        }
        if f.script.is_some() && !IN_INDEX_SCRIPT.with(|c| c.get()) && has_value(v) && !self.scripting {
            return Err(EsError::document_parsing(format!("[1:1] failed to parse field [{}] of type [{}] in document with id '{}'. Preview of field's value: '{}'", f.name, f.ty.name(), self.id, preview(v)))
                .caused_by(EsError::illegal_argument("Cannot index data directly into a field with a [script] parameter")));
        }
        if f.declared.is_semantic() && has_value(v) {
            self.learn_model_settings(f, v);
        }
        if crate::fieldtypes::index::handles(f) {
            if ((f.ty == EsType::Binary && !f.doc_values) || f.ty == EsType::SparseVector) && has_value(v) {
                self.mark_present(&f.name.clone())?;
            }
            return self.emit_ext(f, v);
        }
        // ES NumberFieldMapper treats an exactly empty string as null when
        // coercion is enabled (including null_value substitution). Do this
        // before presence/single-value accounting; whitespace is still invalid.
        let v = if f.coerce && v.as_str() == Some("") && matches!(f.ty,
            EsType::Byte | EsType::Short | EsType::Integer | EsType::Long |
            EsType::HalfFloat | EsType::Float | EsType::Double)
        { &Value::Null } else { v };
        match (f.ty, v) {
            (EsType::DenseVector, Value::Array(_)) | (EsType::DenseVector, Value::String(_)) => return self.emit_vector(f, v),
            (EsType::GeoPoint, Value::Array(a)) if values::is_point_array(a) => return self.emit_scalar(f, v),
            (EsType::Flattened, Value::Object(_)) => return self.emit_scalar(f, v),
            (_, Value::Array(a)) => {
                for x in a {
                    self.emit_field(f, x)?;
                }
                return Ok(());
            }
            (_, Value::Null) => {
                if let Some(nv) = f.null_value.clone() {
                    return self.emit_scalar(f, &nv);
                }
                return Ok(());
            }
            (t, Value::Object(_)) if t != EsType::GeoPoint && t != EsType::Flattened => {
                return Err(EsError::document_parsing(format!(
                    "[1:1] failed to parse field [{}] of type [{}] in document with id '{}'. Preview of field's value: '{}'",
                    f.name,
                    f.ty.name(),
                    self.id,
                    preview(v)
                ))
                .caused_by(EsError::illegal_argument(format!("Can't get text on a {} at 1:1", json_type_name(v)))));
            }
            _ => {}
        }
        self.emit_scalar(f, v)
    }

    /// Index the inference results of the document's `semantic_text`
    /// fields (supplied `_inference_fields` or computed for the document):
    /// one blob per field, validated against its mapping.
    fn emit_inference(&mut self) -> Result<()> {
        for (name, sup) in crate::inference::semantic::supplied_all() {
            let Some(f) = self.lookup.fields.get(&name).cloned() else { continue };
            if !f.declared.is_semantic() {
                continue;
            }
            let Some(aux) = f.aux_id("inference") else { continue };
            if self.doc.fields.iter().any(|(id, _)| *id == aux) {
                continue;
            }
            let params = f.ext.as_ref().map(|x| (*x.params).clone()).unwrap_or_default();
            let id = params.get("inference_id").and_then(Value::as_str).unwrap_or(crate::inference::semantic::DEFAULT_INFERENCE_ID).to_string();
            let mapped = params.get("model_settings").cloned().filter(|x| !x.is_null());
            let (blob, _) = crate::inference::semantic::supplied_blob(&f.name, self.id, &id, mapped.as_ref(), &sup)?;
            self.doc.try_push_field((aux, FieldValue::Keyword(blob)))?;
            self.learn_model_settings(&f, &Value::Null);
        }
        Ok(())
    }

    /// A `semantic_text` field's `model_settings` are added to its mapping
    /// by the first document with a value (`SemanticTextFieldMapper.addDynamicUpdate`):
    /// the document's reported settings, else the endpoint's.
    fn learn_model_settings(&mut self, f: &ResolvedField, v: &Value) {
        let params = f.ext.as_ref().map(|x| (*x.params).clone()).unwrap_or_default();
        if params.get("model_settings").is_some_and(|m| !m.is_null()) || self.pending_paths.contains(&f.name) {
            return;
        }
        let id = params.get("inference_id").and_then(Value::as_str).unwrap_or(crate::inference::semantic::DEFAULT_INFERENCE_ID).to_string();
        let reported = if crate::inference::semantic::is_legacy_value(v) { Some(v.clone()) } else { crate::inference::semantic::supplied(&f.name) };
        let settings = match reported {
            Some(sup) => crate::inference::semantic::reported_settings(&sup),
            None => crate::inference::current().and_then(|svc| svc.get_endpoint(&id)).and_then(|ep| crate::inference::model_settings(&ep)),
        };
        let Some(settings) = settings else { return };
        let mut fm = crate::mapping::FieldMapping::new(f.declared);
        fm.params = params;
        fm.params.insert("model_settings".into(), settings);
        self.pending_paths.insert(f.name.clone());
        if f.is_multi_field {
            // A multi-field is updated through its parent field.
            if let Some((parent, sub)) = f.name.rsplit_once('.') {
                let pty = self.lookup.fields.get(parent).map(|p| p.declared).unwrap_or(EsType::Text);
                let mut pm = crate::mapping::FieldMapping::new(pty);
                pm.fields.insert(sub.to_string(), fm);
                self.pending.push((parent.to_string(), Mapper::Field(pm)));
                return;
            }
        }
        self.pending.push((f.name.clone(), Mapper::Field(fm)));
    }

    fn malformed(&mut self, f: &ResolvedField, v: &Value, cause: EsError) -> Result<()> {
        if f.ignore_malformed {
            self.ignored.push(f.name.clone());
            // A malformed value is still copied to the `copy_to` targets.
            return self.copy_out(f, v);
        }
        Err(EsError::document_parsing(format!(
            "[1:1] failed to parse field [{}] of type [{}] in document with id '{}'. Preview of field's value: '{}'",
            f.name,
            f.ty.name(),
            self.id,
            preview(v)
        ))
        .caused_by(cause))
    }

    fn emit_vector(&mut self, f: &ResolvedField, v: &Value) -> Result<()> {
        // element types (float / byte / bit, hex strings): `crate::vectors`.
        self.doc.admit_allocation(crate::vectors::document_vector_storage(f,v)?)?;
        // `dims`: the vector's ES dimension count (`vec` is the core row,
        // packed for `byte` / `bit`).
        let (vec, dims) = match crate::vectors::doc_vector_with_dims(f, v) {
            Ok(x) => x,
            Err(cause) => return self.malformed(f, v, cause),
        };
        match f.dims {
            None => {
                // Learn dims from the first document (ES does the same); the
                // update carries the field's other parameters so it merges
                // (and is validated against `index_options`) cleanly.
                if !self.pending_paths.contains(&f.name) {
                    let mut fm = crate::mapping::FieldMapping::new(EsType::DenseVector);
                    if let Some(ext) = &f.ext {
                        fm.params = (*ext.params).clone();
                    }
                    fm.params.insert("dims".into(), Value::from(dims));
                    // The learnt dims must be valid for the field's
                    // `index_options` (bbq needs >= 64, int4 even dims, ...).
                    crate::vectors::normalize_mapping(&f.name, &mut fm.params, crate::vectors::VecMappingCtx { warn: false, ..crate::vectors::VecMappingCtx::of(&self.lookup.registry) })
                        .map_err(|e| EsError::document_parsing(format!("[1:1] failed to parse: {}", e.reason)).caused_by(e))?;
                    self.pending_paths.insert(f.name.clone());
                    self.pending.push((f.name.clone(), Mapper::Field(fm)));
                }
                Ok(())
            }
            Some(d) if d < dims => Err(EsError::document_parsing(format!(
                "[1:1] failed to parse: The [dense_vector] field [{}] in doc [document with id '{}'] has more dimensions than defined in the mapping [{}]",
                f.name, self.id, d
            ))),
            Some(d) if d != dims => Err(EsError::document_parsing(format!(
                "[1:1] failed to parse: The [dense_vector] field [{}] in doc [document with id '{}'] has a different number of dimensions [{}] than defined in the mapping [{}]",
                f.name,
                self.id,
                dims,
                d
            ))),
            Some(_) => {
                if let Some(tol) = crate::vectors::unit_length_tolerance(f) {
                    // ES `Element.isUnitVector` on the `float`-accumulated
                    // squared magnitude.
                    let mut sq = 0f32;
                    for x in &vec {
                        sq += x * x;
                    }
                    if (sq - 1.0).abs() >= tol {
                        let preview: Vec<String> = vec.iter().take(5).map(|x| crate::values::java_float(*x).to_string()).collect();
                        return Err(EsError::document_parsing(format!(
                            "[1:1] failed to parse: The [dot_product] similarity can only be used with unit-length vectors. Preview of invalid vector: [{}{}]",
                            preview.join(", "),
                            if vec.len() >= 5 { ", ..." } else { "" }
                        )));
                    }
                }
                if let Some(id) = f.core_id {
                    self.doc.try_push_field((id, FieldValue::Vector(vec)))?;
                }
                // `exists` finds vector fields through `_field_names`.
                self.mark_present(&f.name.clone())?;
                Ok(())
            }
        }
    }

    /// `DocumentParserContext.enforceRequiredFields` for the document of
    /// nested scope `scope` (`None` = root).
    fn enforce_required(&self, scope: Option<&str>) -> Result<()> {
        if self.lookup.required.is_empty() {
            return Ok(());
        }
        let missing: Vec<&str> = self.lookup.required.iter().filter(|(n, s)| s.as_deref() == scope && !self.satisfied.contains(n)).map(|(n, _)| n.as_str()).collect();
        if missing.is_empty() {
            return Ok(());
        }
        let msg = format!("Field(s) [{}] are configured with [nullability=false] but no value was provided", missing.join(", "));
        Err(EsError::document_parsing(format!("[1:1] failed to parse: {msg}")).caused_by(EsError::illegal_argument(msg)))
    }

    fn emit_scalar(&mut self, f: &ResolvedField, v: &Value) -> Result<()> {
        if (f.single_value || f.required) && !v.is_null() {
            if f.single_value && !self.single_seen.insert(f.name.clone()) {
                let msg = format!("Field [{}] is configured with [multi_value=false] but encountered multiple values in the same document", f.name);
                return Err(EsError::document_parsing(format!("[1:1] failed to parse: {msg}")).caused_by(EsError::illegal_argument(msg)));
            }
            if f.required {
                self.satisfied.insert(f.name.clone());
            }
        }
        // `counted_keyword` only takes strings.
        if f.declared == EsType::CountedKeyword && !v.is_string() && !v.is_null() {
            return self.malformed(f, v, EsError::illegal_argument(format!("Encountered unexpected token [{}].", json_type_name(v))));
        }
        if crate::fieldtypes::index::handles(f) {
            return self.emit_ext(f, v);
        }
        let unmapped_ids = f.core_id.is_none() && f.lat_id.is_none() && f.geo_id.is_none();
        // Text / keyword record `_field_names` only when they omit norms
        // (`TextFieldMapper` / `KeywordFieldMapper.parseCreateField`).
        let unindexed = f.core_id.is_some() && !f.indexed && !f.doc_values && !matches!(f.ty, EsType::DenseVector) && !(matches!(f.ty, EsType::Text | EsType::Keyword) && f.norms);
        if (unmapped_ids || unindexed) && !matches!(f.ty, EsType::DenseVector) && !v.is_null() {
            // Neither postings nor doc values (`index: false` text, ...).
            self.mark_present(&f.name.clone())?;
        }
        match f.ty {
            EsType::Text | EsType::MatchOnlyText => {
                let Some(s)=self.scalar_text(v)? else{return Ok(())};
                if let Some(dv) = f.aux_id("dv") {
                    self.doc.try_add_keyword(dv,&s)?;
                }
                // `pattern_text`: the message's template id.
                if f.declared == EsType::PatternText {
                    if let Some(tid) = self.lookup.fields.get(&format!("{}.template_id", f.name)).and_then(|t| t.core_id) {
                        self.doc.try_push_field((tid, FieldValue::Keyword(crate::fieldtypes::pattern_text::template_id(&s))))?;
                    }
                }
                if f.core_id.is_some() && s.trim().is_empty() {
                    // No tokens, so no postings or norms: only `_field_names`.
                    self.mark_present(&f.name.clone())?;
                }
                if let Some(id) = f.core_id {
                    let text = match &f.index_analyzer {
                        Some(IndexAnalyzer::Custom(a)) => {
                            // Pre-analyzed: exact positions (stacked tokens, holes) and
                            // spaces inside tokens survive (`_pretokenized` analyzer).
                            let toks = a.analyze(&s);
                            // Token production is a separate analyzer boundary;
                            // pre-admit this encoded output before allocating it.
                            use std::fmt::Write;
                            struct Count(usize);
                            impl std::fmt::Write for Count {fn write_str(&mut self,s:&str)->std::fmt::Result{self.0=self.0.checked_add(s.len()).ok_or(std::fmt::Error)?;Ok(())}}
                            let mut count=Count(0);
                            for (i,t) in toks.iter().enumerate(){
                                write!(&mut count,"{}",t.position).map_err(|_|EsError::new(429,"circuit_breaking_exception","analyzed text overflow"))?;
                                count.0=count.0.checked_add(usize::from(i>0)+1).and_then(|n|n.checked_add(t.text.len())).ok_or_else(||EsError::new(429,"circuit_breaking_exception","analyzed text overflow"))?;
                            }
                            self.doc.admit_allocation(count.0)?;
                            let mut out=String::new();out.try_reserve_exact(count.0).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
                            for (i, t) in toks.iter().enumerate() {
                                if i > 0 {
                                    out.push('\u{2}');
                                }
                                write!(&mut out,"{}",t.position).map_err(|e|EsError::internal(e.to_string()))?;
                                out.push('\u{1}');
                                out.extend(t.text.chars().filter(|c| *c != '\u{1}' && *c != '\u{2}'));
                            }
                            out
                        }
                        _ => s,
                    };
                    self.doc.try_push_field((id, FieldValue::Text(text)))?;
                }
                self.after(f, v)
            }
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard => {
                let Some(s)=self.scalar_text(v)? else{return Ok(())};
                if let Some(ia) = f.ignore_above {
                    if s.chars().count() > ia {
                        if !self.ignored.contains(&f.name) {
                            self.ignored.push(f.name.clone());
                        }
                        return self.after(f, v);
                    }
                }
                let s = match &f.normalizer {
                    Some(n) => {
                        let tokens=n.analyze(&s);
                        let bytes=tokens.iter().try_fold(0usize,|n,t|n.checked_add(t.text.len())).ok_or_else(||EsError::new(429,"circuit_breaking_exception","normalized text overflow"))?;
                        self.doc.admit_allocation(bytes)?;
                        let mut out=String::new();out.try_reserve_exact(bytes).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
                        for token in tokens {out.push_str(&token.text);}
                        out
                    },
                    None => s,
                };
                if let Some(id) = f.core_id {
                    self.doc.try_push_field((id, FieldValue::Keyword(s)))?;
                }
                self.after(f, v)
            }
            EsType::Flattened => {
                // A dotted source key under a flattened field emits both its
                // root value and the keyed sub-column, in original leaf order.
                if let Some(key) = f.ext.as_ref().and_then(|e| e.flattened_key.as_deref()) {
                    let root = f.name.strip_suffix(key).and_then(|r| r.strip_suffix('.'))
                        .and_then(|r| self.lookup.fields.get(r)).and_then(|r| r.core_id);
                    return self.flattened_leaves(v, root, f.core_id.map(|id| (id, key)));
                }
                self.flattened_leaves(v, f.core_id, None)
            }
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => {
                let (min, max): (i64, i64) = match f.ty {
                    EsType::Byte => (i8::MIN as i64, i8::MAX as i64),
                    EsType::Short => (i16::MIN as i64, i16::MAX as i64),
                    EsType::Integer => (i32::MIN as i64, i32::MAX as i64),
                    _ => (i64::MIN, i64::MAX),
                };
                match parse_integer(v, f.coerce, min, max, f.ty) {
                    Ok(n) => {
                        if let Some(id) = f.core_id {
                            self.doc.try_push_field((id, FieldValue::I64(n)))?;
                        }
                        self.after(f, v)
                    }
                    Err(e) => self.malformed(f, v, e),
                }
            }
            EsType::UnsignedLong => {
                // An empty string is a null (ES `UnsignedLongFieldMapper`).
                if matches!(v, Value::String(s) if s.is_empty()) {
                    return match f.null_value.clone() {
                        Some(nv) => self.emit_scalar(f, &nv),
                        None => Ok(()),
                    };
                }
                let r = match v {
                    Value::Number(n) => n.as_u64().ok_or(()),
                    Value::String(s) if f.coerce => s.trim().parse::<u64>().map_err(|_| ()),
                    _ => Err(()),
                };
                match r {
                    Ok(n) => {
                        if let Some(id) = f.core_id {
                            self.doc.try_push_field((id, FieldValue::U64(n)))?;
                        }
                        self.after(f, v)
                    }
                    Err(_) => {
                        let s = preview(v);
                        let numeric = matches!(v, Value::Number(_)) || s.trim().parse::<f64>().is_ok();
                        let msg = if numeric { format!("Value [{s}] is out of range for unsigned long") } else { format!("Value \"{s}\" is not a valid unsigned long") };
                        self.malformed(f, v, EsError::illegal_argument(msg))
                    }
                }
            }
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => {
                let r = match v {
                    Value::Number(n) => n.as_f64().ok_or_else(|| EsError::illegal_argument("bad number")),
                    Value::String(s) if f.coerce => s.trim().parse::<f64>().map_err(|_| EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))),
                    Value::String(s) => Err(EsError::illegal_argument(format!("For input string: \"{s}\""))),
                    _ => Err(EsError::illegal_argument(format!("Current token ({}) not numeric, can not use numeric value accessors", json_type_name(v)))),
                };
                match r {
                    Ok(x) if !x.is_finite() => self.malformed(f, v, EsError::illegal_argument(format!("[{}] supports only finite values, but got [{x}]", f.ty.name()))),
                    Ok(x) => {
                        let x = match f.ty {
                            EsType::Float => values::to_float(x),
                            EsType::HalfFloat => values::to_half_float(x),
                            EsType::ScaledFloat => rustic_core::fieldtypes::numeric::scaled_round_trip(x, f.scaling_factor),
                            _ => x,
                        };
                        if let Some(id) = f.core_id {
                            self.doc.try_push_field((id, FieldValue::F64(x)))?;
                        }
                        self.after(f, v)
                    }
                    Err(e) => self.malformed(f, v, e),
                }
            }
            EsType::Boolean => {
                let b = match v {
                    Value::Bool(b) => Some(*b),
                    Value::String(s) => match s.as_str() {
                        "true" => Some(true),
                        "false" | "" => Some(false),
                        _ => None,
                    },
                    _ => None,
                };
                match b {
                    Some(b) => {
                        if let Some(id) = f.core_id {
                            self.doc.try_push_field((id, FieldValue::Bool(b)))?;
                        }
                        self.after(f, v)
                    }
                    None => self.malformed(
                        f,
                        v,
                        EsError::illegal_argument(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", preview(v))),
                    ),
                }
            }
            EsType::Date | EsType::DateNanos => {
                let r = match v {
                    Value::String(s) => f.date_format.parse(s),
                    Value::Number(n) => f.date_format.parse_number(n.as_f64().unwrap_or(0.0)),
                    _ => Err(EsError::illegal_argument("bad date")),
                };
                match r {
                    Ok(ms) => {
                        if let Some(id) = f.core_id {
                            self.doc.try_push_field((id, FieldValue::Date(ms)))?;
                        }
                        self.after(f, v)
                    }
                    Err(_) => self.malformed(
                        f,
                        v,
                        EsError::illegal_argument(format!("failed to parse date field [{}] with format [{}]", preview(v), f.date_format.spec)),
                    ),
                }
            }
            EsType::Ip => {
                let s = match v {
                    Value::String(s) => s.as_str(),
                    _ => "",
                };
                match values::parse_ip(s) {
                    Some(b) => {
                        if let Some(id) = f.core_id {
                            self.doc.try_push_field((id, FieldValue::Keyword(values::ip_to_term(&b))))?;
                        }
                        self.after(f, v)
                    }
                    None => self.malformed(f, v, EsError::illegal_argument(format!("'{}' is not an IP string literal.", preview(v)))),
                }
            }
            EsType::GeoPoint => match values::parse_geo_point(v) {
                Some((lat, lon)) => {
                    if let (Some(a), Some(b)) = (f.lat_id, f.lon_id) {
                        self.doc.try_push_field((a, FieldValue::F64(lat)))?;
                        self.doc.try_push_field((b, FieldValue::F64(lon)))?;
                    }
                    if let Some(g) = f.geo_id {
                        self.doc.try_push_field((g, FieldValue::GeoPoint(lat, lon)))?;
                    }
                    if !f.multi_fields.is_empty() {
                        // Multi-fields of a geo_point see its geohash.
                        return self.after(f, &Value::String(values::geohash_encode(lat, lon, 12)));
                    }
                    Ok(())
                }
                None => self.malformed(f, v, EsError::illegal_argument("failed to parse geo_point")),
            },
            EsType::DenseVector => self.malformed(f, v, EsError::illegal_argument("dense_vector value must be an array")),
            _ => Ok(()),
        }
    }

    /// Multi-fields and `copy_to` for a value of `f`.
    fn after(&mut self, f: &ResolvedField, v: &Value) -> Result<()> {
        let lookup: &'a FieldLookup = self.lookup;
        for sub in &f.multi_fields {
            if let Some(sf) = lookup.fields.get(sub) {
                self.emit_scalar(sf, v)?;
            }
        }
        self.copy_out(f, v)
    }

    /// `copy_to` targets of `f` receive `v` (not recursively: a copied value
    /// does not follow the target's own `copy_to`).
    fn copy_out(&mut self, f: &ResolvedField, v: &Value) -> Result<()> {
        if self.copying {
            return Ok(());
        }
        let lookup: &'a FieldLookup = self.lookup;
        for target in &f.copy_to {
            self.copying = true;
            let r = match lookup.get(target) {
                Some(tf) => self.emit_scalar(tf, v),
                None => {
                    let leaf = target.rsplit('.').next().unwrap_or(target).to_string();
                    self.dynamic(target, &leaf, v, Dynamic::True)
                }
            };
            self.copying = false;
            r?;
        }
        Ok(())
    }
}

/// A non-null value (or an array holding one).
fn has_value(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Array(a) => a.iter().any(has_value),
        _ => true,
    }
}

pub(crate) fn parse_integer(v: &Value, coerce: bool, min: i64, max: i64, ty: EsType) -> Result<i64> {
    let out_of_range = |s: String| EsError::illegal_argument(format!("Value [{s}] is out of range for {}", match ty {
        EsType::Byte => "a byte",
        EsType::Short => "a short",
        EsType::Integer => "an integer",
        _ => "a long",
    }));
    let n: i64 = match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(f) = n.as_f64() {
                if f.fract() != 0.0 && !coerce {
                    return Err(EsError::illegal_argument(format!("Value [{f}] has a decimal part")));
                }
                if f > max as f64 || f < min as f64 {
                    return Err(out_of_range(f.to_string()));
                }
                f.trunc() as i64
            } else {
                return Err(out_of_range(n.to_string()));
            }
        }
        Value::String(s) => {
            if !coerce {
                let tn = match ty {
                    EsType::Byte => "Byte",
                    EsType::Short => "Short",
                    EsType::Integer => "Integer",
                    _ => "Long",
                };
                return Err(EsError::illegal_argument(format!("{tn} value passed as String")));
            }
            let t = s.trim();
            if let Ok(i) = t.parse::<i64>() {
                i
            } else if let Ok(f) = t.parse::<f64>() {
                if f > max as f64 || f < min as f64 {
                    return Err(out_of_range(t.to_string()));
                }
                f.trunc() as i64
            } else {
                let kind = if ty == EsType::Long { "illegal_argument_exception" } else { "number_format_exception" };
                return Err(EsError::new(400, kind, format!("For input string: \"{s}\"")));
            }
        }
        Value::Bool(_) => return Err(EsError::illegal_argument("Current token (VALUE_TRUE) not numeric, can not use numeric value accessors")),
        _ => return Err(EsError::illegal_argument("not a number")),
    };
    if n < min || n > max {
        return Err(out_of_range(n.to_string()));
    }
    Ok(n)
}

/// Parse with dynamic-mapping resolution: applies additions to a copy of
/// `mappings` and retries. Returns the parsed doc and, if the mapping
/// changed, the new mapping.
pub fn parse_with_dynamic(
    source: &Value,
    id: &str,
    mappings: &Mappings,
    lookup: &FieldLookup,
    total_fields_limit: usize,
) -> Result<(ParsedDoc, Option<(Mappings, FieldLookup)>)> {
    let limits = crate::fieldtypes::limits::Limits { total_fields: total_fields_limit, ..Default::default() };
    parse_with_limits(source, id, mappings, lookup, &limits)
}

/// The document parsed against `mappings` when it introduces no new
/// field, else `None` (without growing the mapping or building a lookup:
/// [`parse_with_limits`] does that when the caller goes on). Errors as
/// [`parse_with_limits`]'s first round.
pub fn parse_unless_new_fields(
    source: &Value,
    id: &str,
    mappings: &Mappings,
    lookup: &FieldLookup,
    limits: &crate::fieldtypes::limits::Limits,
) -> Result<Option<ParsedDoc>> {
    ARRAY_OBJECTS_LIMIT.with(|c| c.set(limits.array_objects));
    struct ResetLimit;
    impl Drop for ResetLimit {
        fn drop(&mut self) {
            ARRAY_OBJECTS_LIMIT.with(|c| c.set(50_000));
        }
    }
    let _reset = ResetLimit;
    let skip: HashSet<String> = HashSet::new();
    match parse_document_skip_with_regex_limit(source, id, mappings, lookup, Some(&skip), limits.max_regex_length)? {
        ParseOutcome::Doc(d) => {
            crate::fieldtypes::limits::check_nested_objects(&d.doc, limits)?;
            Ok(Some(d))
        }
        ParseOutcome::NeedsMapping(_) => Ok(None),
    }
}

/// [`parse_with_dynamic`] under all the `index.mapping.*` limits
/// (`ignore_dynamic_beyond_limit` drops new dynamic fields past the total
/// fields limit into `_ignored` instead of failing).
pub fn parse_with_limits(
    source: &Value,
    id: &str,
    mappings: &Mappings,
    lookup: &FieldLookup,
    limits: &crate::fieldtypes::limits::Limits,
) -> Result<(ParsedDoc, Option<(Mappings, FieldLookup)>)> {
    let total_fields_limit = limits.total_fields;
    ARRAY_OBJECTS_LIMIT.with(|c| c.set(limits.array_objects));
    struct ResetLimit;
    impl Drop for ResetLimit {
        fn drop(&mut self) {
            ARRAY_OBJECTS_LIMIT.with(|c| c.set(50_000));
        }
    }
    let _reset = ResetLimit;
    let mut skip: HashSet<String> = HashSet::new();
    let mut current: Option<(Mappings, FieldLookup)> = None;
    for _ in 0..64 {
        let (m, lk) = match &current {
            Some((m, lk)) => (m, lk),
            None => (mappings, lookup),
        };
        match parse_document_skip_with_regex_limit(source, id, m, lk, Some(&skip), limits.max_regex_length)? {
            ParseOutcome::Doc(d) => {
                crate::fieldtypes::limits::check_nested_objects(&d.doc, limits)?;
                return Ok((d, current));
            }
            ParseOutcome::NeedsMapping(adds) => {
                let mut next = m.clone();
                apply_new_fields(&mut next, adds, limits, &lookup.registry, &mut skip);
                next.assign_core_ids();
                // Each count walks the whole mapping (thousands of fields
                // for metricbeat): once per side.
                let (next_total, prev_total) = (next.total_fields(), m.total_fields());
                if next_total > total_fields_limit {
                    return Err(EsError::illegal_argument(format!("Limit of total fields [{total_fields_limit}] has been exceeded while adding new fields [{}]", next_total - prev_total)));
                }
                if next_total != prev_total {
                    crate::fieldtypes::limits::check(&next, limits)?;
                }
                let lk = FieldLookup::build(&next, &lookup.registry);
                current = Some((next, lk));
            }
        }
    }
    Err(EsError::internal("dynamic mapping did not converge"))
}

/// Grow `next` by one parse round's new fields `adds` as
/// [`parse_with_limits`] does: `_doc_count` and dynamic runtime fields go to
/// their sections; fields the `ignore_dynamic_beyond_*` limits drop go to
/// `skip`; an existing field keeps its mapping (the first document to
/// introduce a field decides its type, as in ES).
fn apply_new_fields(next: &mut Mappings, adds: Vec<(String, Mapper)>, limits: &crate::fieldtypes::limits::Limits, registry: &crate::analysis::AnalysisRegistry, skip: &mut HashSet<String>) {
    for (path, mapper) in adds {
        // A dynamic runtime field goes to the mapping's `runtime` section.
        if let Mapper::Field(fm) = &mapper {
            if fm.params.contains_key(DOC_COUNT_MARK) {
                next.enable_doc_count();
                continue;
            }
            if let Some(def) = fm.params.get(RUNTIME_MARK) {
                let mut rt = next.runtime.as_ref().and_then(Value::as_object).cloned().unwrap_or_default();
                rt.insert(path.clone(), def.clone());
                next.runtime = Some(Value::Object(rt));
                continue;
            }
        }
        // `field_name_length.ignore_dynamic_beyond_limit`: a new
        // dynamic field whose leaf name is too long is ignored.
        if limits.ignore_dynamic_beyond_field_name_length && path.rsplit('.').next().map(|l| l.chars().count() > limits.field_name_length).unwrap_or(false) {
            skip.insert(path);
            continue;
        }
        if limits.ignore_dynamic_beyond_limit {
            let mut trial = next.clone();
            trial.insert_at_with(&path, mapper, Some(registry));
            if trial.total_fields() > limits.total_fields {
                skip.insert(path);
            } else {
                *next = trial;
            }
        } else {
            next.insert_at_with(&path, mapper, Some(registry));
        }
    }
}

/// The mapping grown by every new field `sources` introduce (in order: the
/// first to introduce a field decides its type), each source parsed once
/// against `mappings` / `lookup` and no lookup built: one mapping update
/// for a whole bulk instead of one (and a full lookup rebuild) per source
/// that meets a new field. `None` when no source introduces a field, or
/// the grown mapping breaks a limit (the sources then fail, or drop their
/// fields, one by one through [`parse_with_limits`]). Sources needing a
/// second round (a field whose mapping depends on an earlier new one) get
/// it when they are parsed against the result.
pub fn grow_mapping_for<'a>(
    sources: impl IntoIterator<Item = (&'a Value, &'a str)>,
    mappings: &Mappings,
    lookup: &FieldLookup,
    limits: &crate::fieldtypes::limits::Limits,
) -> Option<Mappings> {
    let mut next: Option<Mappings> = None;
    for (source, id) in sources {
        ARRAY_OBJECTS_LIMIT.with(|c| c.set(limits.array_objects));
        let outcome = parse_document_skip_with_regex_limit(source, id, mappings, lookup, None, limits.max_regex_length);
        ARRAY_OBJECTS_LIMIT.with(|c| c.set(50_000));
        let Ok(ParseOutcome::NeedsMapping(adds)) = outcome else { continue };
        let grown = next.get_or_insert_with(|| mappings.clone());
        let mut skip = HashSet::new();
        apply_new_fields(grown, adds, limits, &lookup.registry, &mut skip);
    }
    let mut grown = next?;
    grown.assign_core_ids();
    let total = grown.total_fields();
    if total > limits.total_fields || (total != mappings.total_fields() && crate::fieldtypes::limits::check(&grown, limits).is_err()) {
        return None;
    }
    Some(grown)
}

#[allow(dead_code)]
fn parse_with_dynamic_legacy(
    source: &Value,
    id: &str,
    mappings: &Mappings,
    lookup: &FieldLookup,
    total_fields_limit: usize,
) -> Result<(ParsedDoc, Option<(Mappings, FieldLookup)>)> {
    let mut current: Option<(Mappings, FieldLookup)> = None;
    for _ in 0..32 {
        let (m, lk) = match &current {
            Some((m, lk)) => (m, lk),
            None => (mappings, lookup),
        };
        match parse_document(source, id, m, lk)? {
            ParseOutcome::Doc(d) => return Ok((d, current)),
            ParseOutcome::NeedsMapping(adds) => {
                let mut next = m.clone();
                for (path, mapper) in adds {
                    next.insert_at_with(&path, mapper, Some(&lookup.registry));
                }
                next.assign_core_ids();
                if next.total_fields() > total_fields_limit {
                    return Err(EsError::illegal_argument(format!("Limit of total fields [{total_fields_limit}] has been exceeded while adding new fields [{}]", next.total_fields() - m.total_fields())));
                }
                let lk = FieldLookup::build(&next, &lookup.registry);
                current = Some((next, lk));
            }
        }
    }
    Err(EsError::internal("dynamic mapping did not converge"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisRegistry;
    use serde_json::json;

    /// ES `41_doc_values_nullability.yml` "index.mapping.doc_values.
    /// nullability=false requires all fields to carry a value": a document
    /// missing the required `kw` fails before its new `other` field is
    /// mapped, so it grows no mapping, and the next document, with `kw`
    /// only, is accepted.
    #[test]
    fn rejected_required_field_document_grows_no_mapping() {
        let mut reg = AnalysisRegistry::default();
        reg.index_mode = Some("logsdb_columnar".into());
        reg.dv_nullability = false;
        let mut m = Mappings::parse(&json!({"properties": {"@timestamp": {"type": "date"}, "kw": {"type": "keyword"}}}), &reg).unwrap();
        m.assign_core_ids();
        let lk = FieldLookup::build(&m, &reg);
        assert_eq!(lk.required.iter().map(|(n, _)| n.as_str()).collect::<HashSet<_>>(), HashSet::from(["@timestamp", "kw"]));
        let limits = crate::fieldtypes::limits::Limits { total_fields: 1000, ..Default::default() };
        let rejected = json!({"@timestamp": "2024-01-01T00:00:00Z", "other": "x"});
        let e = parse_with_limits(&rejected, "1", &m, &lk, &limits).err().unwrap();
        assert!(e.reason.contains("Field(s) [kw] are configured with [nullability=false] but no value was provided"), "{}", e.reason);
        assert!(grow_mapping_for([(&rejected, "1")], &m, &lk, &limits).is_none(), "a rejected document proposes no new field");
        assert!(parse_unless_new_fields(&rejected, "1", &m, &lk, &limits).is_err());
        let accepted = json!({"@timestamp": "2024-01-01T00:00:00Z", "kw": "a"});
        let (_, upd) = parse_with_limits(&accepted, "2", &m, &lk, &limits).unwrap();
        assert!(upd.is_none());
        // A document that brings the required fields and a new one grows
        // the mapping by it.
        let grows = json!({"@timestamp": "2024-01-01T00:00:00Z", "kw": "a", "other": "x"});
        let (_, upd) = parse_with_limits(&grows, "3", &m, &lk, &limits).unwrap();
        assert!(upd.unwrap().1.get("other").is_some());
    }

    fn setup(m: Value) -> (Mappings, FieldLookup) {
        let reg = AnalysisRegistry::default();
        let mut mp = Mappings::parse(&m, &reg).unwrap();
        mp.assign_core_ids();
        let lk = FieldLookup::build(&mp, &reg);
        (mp, lk)
    }

    fn values_of(doc: &Document, id: u16) -> Vec<FieldValue> {
        doc.get_all(id).cloned().collect()
    }

    #[test]
    fn flattened_stream_preserves_leaf_and_root_key_order() {
        let (mappings, lookup) = setup(json!({}));
        let mut ctx = Ctx { id: "1", max_regex_length: 1000, mappings: &mappings, lookup: &lookup,
            doc: Document::new(), ignored: Vec::new(), pending: Vec::new(), pending_paths: HashSet::new(),
            copying: false, skip: None, scripting: false, array_objects: 0, single_seen: HashSet::new(), satisfied: HashSet::new() };
        ctx.flattened_leaves(&json!(["abc", {"nested": [12, true, null]}]), Some(1), Some((2, "key"))).unwrap();
        let actual: Vec<_> = ctx.doc.fields.iter().map(|(id, value)| (*id, value.as_text().unwrap())).collect();
        assert_eq!(actual, vec![(1,"abc"),(2,"key\0abc"),(1,"12"),(2,"key\012"),(1,"true"),(2,"key\0true")]);
    }

    #[test]
    fn field_names_marks_fields_without_a_presence_signal() {
        let (m, lk) = setup(json!({"properties": {
            "bin": {"type": "binary", "doc_values": false},
            "bin_dv": {"type": "binary", "doc_values": true},
            "empty": {"type": "text"}, "full": {"type": "text"},
            "unindexed": {"type": "text", "index": false},
            "num": {"type": "long"}
        }}));
        let src = json!({"bin": "YWJj", "bin_dv": "YWJj", "empty": "", "full": "hi", "unindexed": "x", "num": 1});
        let (pd, _) = parse_with_dynamic(&src, "1", &m, &lk, 1000).unwrap();
        let mut names: Vec<String> = values_of(&pd.doc, crate::mapping::FIELD_NAMES_FIELD).iter().filter_map(|v| v.as_text().map(str::to_string)).collect();
        names.sort();
        // `unindexed` text keeps norms, so it records no marker
        // (`TextFieldMapper.parseCreateField` only when norms are omitted).
        assert_eq!(names, vec!["bin", "empty"]);
        // Absent / null values leave no marker.
        let (pd, _) = parse_with_dynamic(&json!({"bin": null}), "2", &m, &lk, 1000).unwrap();
        assert!(values_of(&pd.doc, crate::mapping::FIELD_NAMES_FIELD).is_empty());
        // `_field_names` is a reserved document key.
        assert!(parse_with_dynamic(&json!({"_field_names": "x"}), "3", &m, &lk, 1000).is_err());
    }

    #[test]
    fn numeric_empty_strings_follow_null_coercion_without_hiding_malformed_values() {
        for ty in ["byte", "short", "integer", "long", "half_float", "float", "double"] {
            let (m, lk) = setup(json!({"properties": {"n": {"type": ty}}}));
            let id = lk.get("n").unwrap().core_id.unwrap();
            let (parsed, _) = parse_with_dynamic(&json!({"n": ["", null, "5"]}), "empty-pmid", &m, &lk, 1000).unwrap();
            assert_eq!(values_of(&parsed.doc, id).len(), 1, "{ty}");
            assert!(parsed.ignored.is_empty(), "Empty coerced values are null, not malformed: {ty}");
            let error = parse_with_dynamic(&json!({"n": " "}), "space", &m, &lk, 1000).err().unwrap();
            assert_eq!(error.caused_by.as_ref().unwrap().kind,
                if ty == "long" { "illegal_argument_exception" } else { "number_format_exception" }, "{ty}");
            let (strict, strict_lk) = setup(json!({"properties": {"n": {"type": ty, "coerce": false}}}));
            assert!(parse_with_dynamic(&json!({"n": ""}), "strict", &strict, &strict_lk, 1000).is_err(), "{ty}");
            let (defaults, defaults_lk) = setup(json!({"properties": {"n": {"type": ty, "null_value": 7, "ignore_malformed": true}}}));
            let (parsed, _) = parse_with_dynamic(&json!({"n": ["", null, "bad"]}), "defaults", &defaults, &defaults_lk, 1000).unwrap();
            let values = values_of(&parsed.doc, defaults_lk.get("n").unwrap().core_id.unwrap());
            assert_eq!(values.len(), 2, "{ty}");
            for value in values {
                if matches!(ty, "byte" | "short" | "integer" | "long") {
                    assert!(matches!(value, FieldValue::I64(7)), "{ty}: {value:?}");
                } else {
                    assert!(matches!(value, FieldValue::F64(v) if v == 7.0), "{ty}: {value:?}");
                }
            }
            assert_eq!(parsed.ignored, vec!["n".to_string()], "{ty}");
        }
    }

    #[test]
    fn typed_conversion() {
        let (m, lk) = setup(json!({"properties": {
            "t": {"type": "text", "fields": {"k": {"type": "keyword"}}},
            "k": {"type": "keyword", "ignore_above": 3, "null_value": "NULL"},
            "n": {"type": "integer"}, "f": {"type": "float"}, "sf": {"type": "scaled_float", "scaling_factor": 10},
            "b": {"type": "boolean"}, "d": {"type": "date"}, "ip": {"type": "ip"}, "g": {"type": "geo_point"},
            "v": {"type": "dense_vector", "dims": 2}, "obj": {"properties": {"x": {"type": "long"}}},
            "c": {"type": "keyword", "copy_to": "all"}, "all": {"type": "text"}
        }}));
        let src = json!({"t": "Hello World", "k": ["ab", "abcd", null], "n": "42", "f": 1.1, "sf": 1.26, "b": "false",
            "d": "2020-01-01T00:00:00Z", "ip": "10.0.0.1", "g": [1.0, 2.0], "v": [0.5, 0.5], "obj": {"x": 7}, "c": "copied"});
        let (pd, upd) = parse_with_dynamic(&src, "1", &m, &lk, 1000).unwrap();
        assert!(upd.is_none());
        let id = |p: &str| m.core_ids[p];
        assert_eq!(values_of(&pd.doc, id("t")).len(), 1);
        assert!(matches!(&values_of(&pd.doc, id("t.k"))[0], FieldValue::Keyword(s) if s == "Hello World"));
        let ks: Vec<String> = values_of(&pd.doc, id("k")).iter().map(|v| v.as_text().unwrap().to_string()).collect();
        assert_eq!(ks, vec!["ab".to_string()]);
        assert_eq!(pd.ignored, vec!["k".to_string()]);
        assert!(matches!(values_of(&pd.doc, id("n"))[0], FieldValue::I64(42)));
        assert!(matches!(values_of(&pd.doc, id("f"))[0], FieldValue::F64(x) if x == 1.100000023841858));
        assert!(matches!(values_of(&pd.doc, id("sf"))[0], FieldValue::F64(x) if (x - 1.3).abs() < 1e-9));
        assert!(matches!(values_of(&pd.doc, id("b"))[0], FieldValue::Bool(false)));
        assert!(matches!(values_of(&pd.doc, id("d"))[0], FieldValue::Date(1577836800000)));
        assert!(matches!(&values_of(&pd.doc, id("ip"))[0], FieldValue::Keyword(s) if *s == values::ipv4(10, 0, 0, 1)));
        assert!(matches!(values_of(&pd.doc, id("g#geo"))[0], FieldValue::GeoPoint(x, _) if x == 2.0));
        assert!(!m.core_ids.contains_key("g#lat"));
        assert!(matches!(&values_of(&pd.doc, id("v"))[0], FieldValue::Vector(v) if v.len() == 2));
        assert!(matches!(values_of(&pd.doc, id("obj.x"))[0], FieldValue::I64(7)));
        assert!(matches!(&values_of(&pd.doc, id("all"))[0], FieldValue::Text(s) if s == "copied"));
    }

    #[test]
    fn errors_and_ignore_malformed() {
        let (m, lk) = setup(json!({"properties": {"n": {"type": "byte"}, "m": {"type": "long", "ignore_malformed": true}, "o": {"properties": {"x": {"type": "long"}}}}}));
        let e = parse_with_dynamic(&json!({"n": 300}), "1", &m, &lk, 1000).err().unwrap();
        assert_eq!(e.kind, "document_parsing_exception");
        assert!(e.reason.contains("failed to parse field [n] of type [byte] in document with id '1'. Preview of field's value: '300'"), "{}", e.reason);
        assert_eq!(e.caused_by.unwrap().reason, "Value [300] is out of range for a byte");
        let (pd, _) = parse_with_dynamic(&json!({"m": "abc"}), "1", &m, &lk, 1000).unwrap();
        assert_eq!(pd.ignored, vec!["m"]);
        let e = parse_with_dynamic(&json!({"o": 5}), "1", &m, &lk, 1000).err().unwrap();
        assert!(e.reason.contains("object mapping for [o] tried to parse field [o] as object"));
        let e = parse_with_dynamic(&json!({"n": {"a": 1}}), "1", &m, &lk, 1000).err().unwrap();
        assert_eq!(e.kind, "document_parsing_exception");
    }

    #[test]
    fn dynamic_mapping_rules() {
        let (m, lk) = setup(json!({}));
        let src = json!({"title": "hello", "count": 5, "ratio": 0.5, "ok": true, "when": "2021-02-03", "user": {"name": "x", "tags": ["a", "b"]}, "nothing": null, "list": [1, 2]});
        let (pd, upd) = parse_with_dynamic(&src, "1", &m, &lk, 1000).unwrap();
        let (m2, lk2) = upd.unwrap();
        let j = m2.to_json();
        assert_eq!(j["properties"]["title"], json!({"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}));
        assert_eq!(j["properties"]["count"], json!({"type": "long"}));
        assert_eq!(j["properties"]["ratio"], json!({"type": "float"}));
        assert_eq!(j["properties"]["ok"], json!({"type": "boolean"}));
        assert_eq!(j["properties"]["when"], json!({"type": "date"}));
        assert_eq!(j["properties"]["user"]["properties"]["tags"]["type"], json!("text"));
        assert_eq!(j["properties"]["list"], json!({"type": "long"}));
        assert!(j["properties"].get("nothing").is_none());
        assert!(lk2.get("user.name.keyword").is_some());
        assert_eq!(pd.doc.get_all(m2.core_ids["list"]).count(), 2);
    }

    #[test]
    fn strict_and_false() {
        let (m, lk) = setup(json!({"dynamic": "strict", "properties": {"a": {"type": "keyword"}, "o": {"dynamic": "false", "properties": {}}}}));
        let e = parse_with_dynamic(&json!({"b": 1}), "1", &m, &lk, 1000).err().unwrap();
        assert_eq!(e.kind, "strict_dynamic_mapping_exception");
        assert_eq!(e.reason, "[1:1] mapping set to strict, dynamic introduction of [b] within [_doc] is not allowed");
        let (_, upd) = parse_with_dynamic(&json!({"o": {"z": 1}}), "1", &m, &lk, 1000).unwrap();
        assert!(upd.is_none());
    }

    #[test]
    fn dense_vector_dims_learnt_and_checked() {
        let (m, lk) = setup(json!({"properties": {"v": {"type": "dense_vector"}}}));
        let (_, upd) = parse_with_dynamic(&json!({"v": [1.0, 2.0, 3.0]}), "1", &m, &lk, 1000).unwrap();
        let (m2, lk2) = upd.unwrap();
        assert_eq!(m2.to_json()["properties"]["v"]["dims"], json!(3));
        let e = parse_with_dynamic(&json!({"v": [1.0]}), "2", &m2, &lk2, 1000).err().unwrap();
        assert!(e.reason.contains("different number of dimensions [1] than defined in the mapping [3]"));
    }

    #[test]
    fn total_fields_limit() {
        let (m, lk) = setup(json!({}));
        let e = parse_with_dynamic(&json!({"a": 1, "b": 2, "c": 3}), "1", &m, &lk, 2).err().unwrap();
        assert!(e.reason.starts_with("Limit of total fields [2] has been exceeded"));
    }

    fn noaa_mapping() -> (Mappings, FieldLookup) {
        setup(json!({"properties": {
            "date": {"type": "date"},
            "TAVG": {"type": "float"}, "TMAX": {"type": "float"}, "TMIN": {"type": "float"}, "PRCP": {"type": "float"}, "SNWD": {"type": "float"},
            "station": {"properties": {
                "id": {"type": "keyword"}, "name": {"type": "keyword"}, "country": {"type": "keyword"}, "country_code": {"type": "keyword"},
                "gsn_flag": {"type": "keyword"}, "wmo_id": {"type": "keyword"}, "elevation": {"type": "float"}, "location": {"type": "geo_point"}
            }}
        }}))
    }

    /// A Rally noaa-shaped document (numeric, date and keyword leaves under
    /// a root object and one nested object level).
    fn noaa_doc(i: usize) -> Value {
        json!({"date": format!("2016-01-{:02}T00:00:00", i % 28 + 1), "TAVG": 22.9 + i as f64, "TMAX": 30.1, "TMIN": 15.5, "PRCP": 0.0, "SNWD": i as f64,
            "station": {"elevation": 34.0, "name": format!("SHARJAH INTER. AIRP {i}"), "country": "United Arab Emirates", "gsn_flag": "GSN",
                "location": {"lat": 25.333, "lon": 55.517}, "country_code": "AE", "wmo_id": "41196", "id": format!("AE{i:09}")}})
    }

    fn parsed_doc(src: &Value, m: &Mappings, lk: &FieldLookup) -> Document {
        match parse_document(src, "1", m, lk).unwrap() {
            ParseOutcome::Doc(d) => d.doc,
            ParseOutcome::NeedsMapping(_) => panic!("every leaf is mapped"),
        }
    }

    /// Chunked producer admission changes only how often the budget is
    /// visited: the indexed fields equal those of exact (one admission per
    /// producer) accounting, and a parsed document keeps no spare credit.
    #[test]
    fn chunked_admission_indexes_the_same_document() {
        let (m, lk) = noaa_mapping();
        for i in 0..4 {
            let src = noaa_doc(i);
            let chunked = parsed_doc(&src, &m, &lk);
            assert_eq!(chunked.unclaimed_allocation(), 0, "settled at the end of the parse");
            assert!(chunked.fields.len() >= 13, "{:?}", chunked.fields);
            // Under a plan smaller than one chunk every admission is exact.
            let lease = rustic_core::admission::Lease::reserve(rustic_core::admission::shared(), rustic_core::types::ALLOCATION_CHUNK / 2, "exact admission").unwrap();
            let exact = {
                let _plan = lease.enter();
                parsed_doc(&src, &m, &lk)
            };
            assert_eq!(format!("{:?}", chunked.fields), format!("{:?}", exact.fields));
        }
    }

    /// A document whose producers exceed the budget is still refused with
    /// the circuit-breaking error, before the oversized value is allocated,
    /// and leaves nothing charged.
    #[test]
    fn oversized_document_is_still_refused_by_the_breaker() {
        let (m, lk) = setup(json!({"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}));
        let plan: u64 = 8 << 10;
        let lease = rustic_core::admission::Lease::reserve(rustic_core::admission::shared(), plan, "small plan").unwrap();
        let _plan = lease.enter();
        assert!(matches!(parse_document(&json!({"k": "fits", "n": 1}), "1", &m, &lk), Ok(ParseOutcome::Doc(_))));
        let error = parse_document(&json!({"k": "x".repeat(64 << 10), "n": 1}), "2", &m, &lk).err().expect("refused");
        assert_eq!(error.status, 429);
        assert_eq!(error.kind, "circuit_breaking_exception");
        assert_eq!(lease.available(), plan, "nothing stays charged after either document");
    }

    /// `cargo test -p elasticsearch-engine --release --lib noaa_parse_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn noaa_parse_microbench() {
        let (m, lk) = noaa_mapping();
        let sources: Vec<String> = (0..10_000).map(|i| noaa_doc(i).to_string()).collect();
        let rounds = 20;
        let started = std::time::Instant::now();
        let mut fields = 0usize;
        for _ in 0..rounds {
            for s in &sources {
                let v = crate::jsonpos::parse_plain(s.as_bytes()).expect("plain JSON");
                fields += std::hint::black_box(parsed_doc(&v, &m, &lk)).fields.len();
            }
        }
        let n = sources.len() * rounds;
        let secs = started.elapsed().as_secs_f64();
        eprintln!("noaa parse_plain + parse_document: {n} docs, {fields} fields, {:.0} docs/s/thread, {:.0} ns/doc", n as f64 / secs, secs * 1e9 / n as f64);
    }
}
