//! Cluster-wide engine metadata beyond indices and templates.
//!
//! ILM policies, ingest pipelines, synonym sets, query rulesets and search
//! applications are stored by the engine outside its index/template state.
//! Like every metadata API they execute on the elected master's engine;
//! [`extract`] copies the result into metadata customs so it is published,
//! and [`apply`] makes every other node's engine match. Without this a
//! resource exists only on the master and a read served by another node
//! answers 404.

use elasticsearch_cluster::state::Metadata;
use elasticsearch_engine::node::Node as Engine;
use elasticsearch_engine::search_ext;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// `{"policies": {name: {"policy", "version"}}, "running": bool}`.
pub const ILM_CUSTOM: &str = "index_lifecycle";
/// `{"pipeline": [{"id", "config"}]}` (the ES `IngestMetadata` shape).
pub const INGEST_CUSTOM: &str = "ingest";
/// `{set: [[rule id, synonyms], ...]}`. ES keeps these in a system index.
pub const SYNONYMS_CUSTOM: &str = "_synonyms";
/// `{ruleset: [rule, ...]}`.
pub const RULESETS_CUSTOM: &str = "_query_rulesets";
/// `{name: definition}`.
pub const SEARCH_APPS_CUSTOM: &str = "_search_applications";
/// `{name: {"<type>": policy}}` (ES `EnrichMetadata`).
pub const ENRICH_CUSTOM: &str = "_enrich_policies";
/// `{"endpoints": [...], "secrets": {...}}` (ES `.inference` system index).
pub const INFERENCE_CUSTOM: &str = "_inference_endpoints";
pub const STORED_SCRIPTS_CUSTOM: &str = "_stored_scripts";
pub const ESQL_VIEWS_CUSTOM: &str = "_esql_views";
/// Internal persisted license state, including whether the trial was used.
pub const LICENSE_CUSTOM: &str = "_license_state";
/// Native authentication state, hidden from the public metadata API.
pub const SECURITY_CUSTOM: &str = "_security_state";

/// The customs [`apply`] / [`apply_changed`] write into an engine. Any other
/// custom (data streams, persistent tasks, ...) is never held back for an
/// engine API in flight: there is nothing of it in the engine to overwrite.
pub const ENGINE_APPLIED_CUSTOMS: &[&str] = &[
    "repositories", SECURITY_CUSTOM, LICENSE_CUSTOM, "snapshot_lifecycle", ILM_CUSTOM, INGEST_CUSTOM,
    SYNONYMS_CUSTOM, RULESETS_CUSTOM, SEARCH_APPS_CUSTOM,
];

/// The customs [`extract_only`] can read back from an engine.
pub fn is_extractable_custom(key: &str) -> bool {
    matches!(key, STORED_SCRIPTS_CUSTOM | SECURITY_CUSTOM | LICENSE_CUSTOM | ILM_CUSTOM | INGEST_CUSTOM | SYNONYMS_CUSTOM
        | RULESETS_CUSTOM | SEARCH_APPS_CUSTOM | ENRICH_CUSTOM | INFERENCE_CUSTOM | ESQL_VIEWS_CUSTOM)
}

/// Whether `engine` already holds custom `key` exactly as `md` has it (an
/// engine API changed it there and its own publication carries that form).
pub fn engine_holds_custom(engine: &Engine, md: &Metadata, key: &str) -> bool {
    if !is_extractable_custom(key) {
        return false;
    }
    let mut own = Metadata::default();
    extract_only(engine, &mut own, Some(key));
    same_custom(key, own.customs.get(key), md.customs.get(key))
}

/// Whether two forms of custom `key` say the same thing: pipelines are
/// compared by id (a kept job appends a new one where the engine lists it
/// in id order). Anything else compares exactly.
pub fn same_custom(key: &str, a: Option<&Value>, b: Option<&Value>) -> bool {
    if a == b {
        return true;
    }
    if key == INGEST_CUSTOM {
        let by_id = |v: Option<&Value>| -> Option<BTreeMap<String, Value>> {
            v.map_or(Some(BTreeMap::new()), |c| c["pipeline"].as_array().map(|a| a.iter().filter_map(|p| Some((p["id"].as_str()?.to_string(), p["config"].clone()))).collect()))
        };
        return by_id(a).is_some_and(|a| Some(a) == by_id(b));
    }
    false
}

/// Customs that have no counterpart under `metadata` in ES's
/// `GET _cluster/state`.
pub fn is_internal_custom(key: &str) -> bool {
    matches!(key, SYNONYMS_CUSTOM | RULESETS_CUSTOM | SEARCH_APPS_CUSTOM | LICENSE_CUSTOM | SECURITY_CUSTOM | ENRICH_CUSTOM | INFERENCE_CUSTOM | ESQL_VIEWS_CUSTOM | STORED_SCRIPTS_CUSTOM)
}

fn set_custom(md: &mut Metadata, key: &str, v: Value, empty: bool) {
    if empty {
        md.customs.remove(key);
    } else if md.customs.get(key) != Some(&v) {
        md.customs.insert(key.into(), v);
    }
}

/// Copy the master engine's extra metadata into `md.customs`.
pub fn extract(engine: &Engine, md: &mut Metadata) {
    extract_only(engine, md, None);
}

/// [`extract`] of one custom family (`Some(key)`), or of all (`None`).
pub fn extract_only(engine: &Engine, md: &mut Metadata, only: Option<&str>) {
    let want = |key: &str| only.is_none_or(|o| o == key);
    if want(STORED_SCRIPTS_CUSTOM) {
        set_custom(md, STORED_SCRIPTS_CUSTOM, elasticsearch_engine::script::stored::for_data_dir(&engine.data).cluster_state(), false);
    }
    if want(SECURITY_CUSTOM) {
        set_custom(md, SECURITY_CUSTOM, engine.security.cluster_state(), false);
    }
    if want(LICENSE_CUSTOM) {
        set_custom(md, LICENSE_CUSTOM, elasticsearch_engine::license::service(engine).cluster_state(), false);
    }
    if want(ILM_CUSTOM) {
        let st = engine.ilm.st_snapshot();
        let policies: Map<String, Value> = st.policies.iter().map(|(k, p)| (k.clone(), json!({"policy": p.policy, "version": p.version}))).collect();
        // The requested direction (`running`) and the mode: STOPPING becomes
        // STOPPED when the elected master finds no index inside an unsafe
        // action, and publishes it (every node then reports the same mode).
        let running = st.mode == "RUNNING";
        set_custom(md, ILM_CUSTOM, json!({"policies": policies, "running": running, "mode": st.mode}), false);
    }
    if want(INGEST_CUSTOM) {
        let pipelines: Vec<Value> = engine.ingest().get(None).into_iter().map(|(id, config)| json!({"id": id, "config": config})).collect();
        let empty = pipelines.is_empty();
        set_custom(md, INGEST_CUSTOM, json!({"pipeline": pipelines}), empty);
    }
    let ext = search_ext::ext(engine);
    if want(SYNONYMS_CUSTOM) {
        let syn: Map<String, Value> = ext.synonyms.read().iter().map(|(k, rules)| (k.clone(), json!(rules))).collect();
        let empty = syn.is_empty();
        set_custom(md, SYNONYMS_CUSTOM, Value::Object(syn), empty);
    }
    if want(RULESETS_CUSTOM) {
        let rules: Map<String, Value> = ext.rulesets.read().iter().map(|(k, v)| (k.clone(), Value::Array(v.clone()))).collect();
        let empty = rules.is_empty();
        set_custom(md, RULESETS_CUSTOM, Value::Object(rules), empty);
    }
    if want(SEARCH_APPS_CUSTOM) {
        let apps: Map<String, Value> = ext.search_apps.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let empty = apps.is_empty();
        set_custom(md, SEARCH_APPS_CUSTOM, Value::Object(apps), empty);
    }
    // Kept when empty, so deleting the last policy reaches every node.
    if want(ENRICH_CUSTOM) {
        set_custom(md, ENRICH_CUSTOM, elasticsearch_engine::enrich::store(engine).cluster_state(), false);
    }
    // Kept when empty, so deleting the last endpoint reaches every node.
    if want(INFERENCE_CUSTOM) {
        set_custom(md, INFERENCE_CUSTOM, elasticsearch_engine::inference::service_for(&engine.data).cluster_state(), false);
    }
    // Kept when empty, so deleting the last view reaches every node.
    if want(ESQL_VIEWS_CUSTOM) {
        set_custom(md, ESQL_VIEWS_CUSTOM, elasticsearch_engine::esql::views::store(engine).cluster_state(), false);
    }
}

/// The custom family an engine API under `first_segment` changes, when it
/// changes exactly one (its kept job then snapshots only that family).
pub fn custom_family_of_api(first_segment: &str) -> Option<&'static str> {
    match first_segment {
        "_ingest" => Some(INGEST_CUSTOM),
        "_security" => Some(SECURITY_CUSTOM),
        "_ilm" => Some(ILM_CUSTOM),
        "_license" => Some(LICENSE_CUSTOM),
        "_scripts" => Some(STORED_SCRIPTS_CUSTOM),
        "_synonyms" => Some(SYNONYMS_CUSTOM),
        "_query_rules" => Some(RULESETS_CUSTOM),
        "_application" => Some(SEARCH_APPS_CUSTOM),
        "_enrich" => Some(ENRICH_CUSTOM),
        "_inference" => Some(INFERENCE_CUSTOM),
        // Only `_query/view` mutates (`POST /_query` is a read and never
        // takes a kept job).
        "_query" => Some(ESQL_VIEWS_CUSTOM),
        _ => None,
    }
}

/// Make a non-master node's ES|QL views the published ones (a query that
/// reads a view can run on, or be forwarded to, any node).
pub fn apply_stored_scripts(engine: &Engine, md: &Metadata) -> Option<String> {
    let scripts = md.customs.get(STORED_SCRIPTS_CUSTOM)?;
    elasticsearch_engine::script::stored::for_data_dir(&engine.data).restore_cluster_state(scripts).err().map(|e| e.reason)
}

pub fn apply_esql_views(engine: &Engine, md: &Metadata) -> Option<String> {
    // Empty view catalogs are omitted by extract_only; absence must clear
    // a previous catalog on followers and the validation copy.
    let views = md.customs.get(ESQL_VIEWS_CUSTOM).unwrap_or(&Value::Null);
    elasticsearch_engine::esql::views::store(engine).restore_cluster_state(views).err().map(|e| e.reason)
}

/// Make the local engine's extra metadata match `md`. Returns
/// `(resource, error)` pairs for anything that could not be applied.
pub fn apply(engine: &Engine, md: &Metadata) -> Vec<(String, String)> {
    let mut errors = Vec::new();
    engine.snapshots.restore_cluster_registrations(
        md.customs.get("repositories").and_then(Value::as_object)
            .map(|entries| entries.iter().map(|(name, value)| (name.clone(), value.clone())).collect())
            .unwrap_or_default(),
    );
    if let Some(security) = md.customs.get(SECURITY_CUSTOM) {
        if let Err(error) = engine.security.restore_cluster_state(security) {
            errors.push(("native security state".into(), error.reason));
        }
    }
    if let Some(license) = md.customs.get(LICENSE_CUSTOM) {
        if let Err(error) = elasticsearch_engine::license::service(engine).restore_cluster_state(license) {
            errors.push(("license".into(), error.reason));
        }
    }
    let slm = crate::snapshot_api::slm::SlmModel::from_value(md.customs.get("snapshot_lifecycle"));
    engine.ilm.restore_cluster_slm_health(slm.mode.as_str().to_string(), slm.policies.len());
    apply_ilm(engine, None, md, &mut errors);
    apply_ingest(engine, None, md, &mut errors);
    apply_ext(engine, None, md, &mut errors);
    errors
}

/// [`apply`] of what changed from `prev` to `md`, entry by entry: a family
/// whose custom did not change is not touched, and within pipelines and
/// lifecycle policies only the entries that changed are written or
/// removed. An engine API's change not yet published (a pipeline PUT whose
/// kept job is still to run) is so never wiped by an unrelated commit.
pub fn apply_changed(engine: &Engine, prev: &Metadata, md: &Metadata) -> Vec<(String, String)> {
    let changed = |k: &str| prev.customs.get(k) != md.customs.get(k);
    let mut errors = Vec::new();
    if changed("repositories") {
        engine.snapshots.restore_cluster_registrations(
            md.customs.get("repositories").and_then(Value::as_object)
                .map(|entries| entries.iter().map(|(name, value)| (name.clone(), value.clone())).collect())
                .unwrap_or_default(),
        );
    }
    if changed(SECURITY_CUSTOM) {
        if let Some(security) = md.customs.get(SECURITY_CUSTOM) {
            if let Err(error) = engine.security.restore_cluster_state(security) {
                errors.push(("native security state".into(), error.reason));
            }
        }
    }
    if changed(LICENSE_CUSTOM) {
        if let Some(license) = md.customs.get(LICENSE_CUSTOM) {
            if let Err(error) = elasticsearch_engine::license::service(engine).restore_cluster_state(license) {
                errors.push(("license".into(), error.reason));
            }
        }
    }
    if changed("snapshot_lifecycle") {
        let slm = crate::snapshot_api::slm::SlmModel::from_value(md.customs.get("snapshot_lifecycle"));
        engine.ilm.restore_cluster_slm_health(slm.mode.as_str().to_string(), slm.policies.len());
    }
    if changed(ILM_CUSTOM) {
        apply_ilm(engine, Some(prev), md, &mut errors);
    }
    if changed(INGEST_CUSTOM) {
        apply_ingest(engine, Some(prev), md, &mut errors);
    }
    if [SYNONYMS_CUSTOM, RULESETS_CUSTOM, SEARCH_APPS_CUSTOM].iter().any(|k| changed(k)) {
        apply_ext(engine, Some(prev), md, &mut errors);
    }
    errors
}

/// An engine change to custom `key` (`before` -> `after`, both as engine 0
/// held them) laid onto the committed value `base`, as ES's master runs
/// each task against the latest state: only the entries that changed are
/// written or removed, at any depth of the custom's objects (users, roles,
/// scripts, synonym sets, lifecycle policies, data streams, ...;
/// pipelines by id). A committed entry engine 0 does not hold yet (it
/// failed to apply there) is so never reverted, and two concurrent changes
/// of different entries both land. `None`: the custom is absent.
pub fn merge_custom_change(key: &str, base: Option<&Value>, before: Option<&Value>, after: Option<&Value>) -> Option<Value> {
    if key == INGEST_CUSTOM {
        let entries = |v: Option<&Value>| -> Vec<(String, Value)> {
            v.and_then(|c| c["pipeline"].as_array()).map(|a| a.iter().filter_map(|p| Some((p["id"].as_str()?.to_string(), p["config"].clone()))).collect()).unwrap_or_default()
        };
        let (b, a) = (entries(before).into_iter().collect::<BTreeMap<_, _>>(), entries(after).into_iter().collect::<BTreeMap<_, _>>());
        let mut out = entries(base);
        for id in b.keys().chain(a.keys()).cloned().collect::<BTreeSet<String>>() {
            if b.get(&id) == a.get(&id) {
                continue;
            }
            match (out.iter().position(|(k, _)| k == &id), a.get(&id)) {
                // In place: the array keeps its order (no spurious change).
                (Some(i), Some(config)) => out[i].1 = config.clone(),
                (Some(i), None) => {
                    out.remove(i);
                }
                (None, Some(config)) => out.push((id, config.clone())),
                (None, None) => {}
            }
        }
        return (!out.is_empty()).then(|| json!({"pipeline": out.into_iter().map(|(id, config)| json!({"id": id, "config": config})).collect::<Vec<_>>()}));
    }
    let object = |v: Option<&Value>| v.and_then(Value::as_object).cloned().unwrap_or_default();
    // Absent is empty (an emptied custom is removed): never "revert to absent".
    let (mut out, b, a) = (object(base), object(before), object(after));
    let changed: Vec<String> = b.keys().chain(a.keys()).filter(|k| b.get(*k) != a.get(*k)).cloned().collect::<BTreeSet<String>>().into_iter().collect();
    if changed.is_empty() {
        return base.cloned();
    }
    match key {
        // One license: replaced whole (never fields of two under one signature).
        LICENSE_CUSTOM => return after.cloned(),
        SECURITY_CUSTOM => {
            for field in &changed {
                if field == "native_revision" {
                    continue;
                }
                // Users, roles, mappings, keys, ... replaced whole per entry;
                // application privileges and service tokens one level down.
                let depth = if matches!(field.as_str(), "app_privileges" | "service_tokens") { 2 } else { 1 };
                let merged = merge_entries(out.get(field), b.get(field), a.get(field), depth);
                set_or_remove(&mut out, field, merged);
            }
            // A revision above every node's (nodes skip a revision they saw).
            let rev = |m: &Map<String, Value>| m.get("native_revision").and_then(Value::as_u64).unwrap_or(0);
            out.insert("native_revision".into(), json!(rev(&out).max(rev(&a)) + 1));
        }
        ILM_CUSTOM => {
            for field in &changed {
                if field == "policies" {
                    let (bp, ap) = (object(b.get(field)), object(a.get(field)));
                    let mut policies = object(out.get(field));
                    for name in bp.keys().chain(ap.keys()).filter(|k| bp.get(*k) != ap.get(*k)).cloned().collect::<BTreeSet<String>>() {
                        match ap.get(&name) {
                            Some(entry) => {
                                // Replaced whole, at a version above the committed one.
                                let mut entry = entry.clone();
                                let v = |e: Option<&Value>| e.and_then(|e| e["version"].as_u64()).unwrap_or(0);
                                entry["version"] = json!(v(Some(&entry)).max(v(policies.get(&name)) + 1));
                                policies.insert(name, entry);
                            }
                            None => {
                                policies.remove(&name);
                            }
                        }
                    }
                    out.insert(field.clone(), Value::Object(policies));
                } else {
                    set_or_remove(&mut out, field, a.get(field).cloned());
                }
            }
        }
        crate::mirror::DATA_STREAMS_CUSTOM => {
            for name in &changed {
                let merged = match (out.get(name), b.get(name), a.get(name)) {
                    // The committed stream plus the job's own changes.
                    (Some(committed), before, Some(ours)) => Some(merge_stream(committed, before, ours)),
                    // Deleted by the job.
                    (Some(_), Some(_), None) => None,
                    // Created by the job.
                    (None, None, Some(ours)) => Some(ours.clone()),
                    // Deleted in the committed state meanwhile: stays deleted.
                    (None, Some(_), _) => None,
                    (committed, _, _) => committed.cloned(),
                };
                set_or_remove(&mut out, name, merged);
            }
        }
        // Inference endpoints: an array of endpoints, by `inference_id`, and
        // their secrets, by id.
        INFERENCE_CUSTOM => {
            for field in &changed {
                if field == "endpoints" {
                    let by_id = |v: Option<&Value>| -> Vec<(String, Value)> {
                        v.and_then(Value::as_array).map(|a| a.iter().filter_map(|e| Some((e["inference_id"].as_str()?.to_string(), e.clone()))).collect()).unwrap_or_default()
                    };
                    let (bm, am): (BTreeMap<String, Value>, BTreeMap<String, Value>) = (by_id(b.get(field)).into_iter().collect(), by_id(a.get(field)).into_iter().collect());
                    let mut list = by_id(out.get(field));
                    for id in bm.keys().chain(am.keys()).filter(|k| bm.get(*k) != am.get(*k)).cloned().collect::<BTreeSet<String>>() {
                        match (list.iter().position(|(k, _)| k == &id), am.get(&id)) {
                            (Some(i), Some(e)) => list[i].1 = e.clone(),
                            (Some(i), None) => {
                                list.remove(i);
                            }
                            (None, Some(e)) => list.push((id, e.clone())),
                            (None, None) => {}
                        }
                    }
                    out.insert(field.clone(), Value::Array(list.into_iter().map(|(_, e)| e).collect()));
                } else {
                    // `secrets` (by id) and anything else: entries whole.
                    let merged = merge_entries(out.get(field), b.get(field), a.get(field), 1);
                    set_or_remove(&mut out, field, merged);
                }
            }
        }
        // Scripts, synonym sets, rulesets, search applications, enrich
        // policies, ...: entries replaced whole.
        _ => {
            for name in &changed {
                set_or_remove(&mut out, name, a.get(name).cloned());
            }
        }
    }
    (!out.is_empty() || after.is_some()).then_some(Value::Object(out))
}

fn set_or_remove(out: &mut Map<String, Value>, key: &str, v: Option<Value>) {
    match v {
        Some(v) => {
            out.insert(key.to_string(), v);
        }
        None => {
            out.remove(key);
        }
    }
}

/// Three-way merge down to `depth` levels of objects; below that an
/// entry is replaced whole (a PUT replaces a role, a script, ...).
fn merge_entries(base: Option<&Value>, before: Option<&Value>, after: Option<&Value>, depth: usize) -> Option<Value> {
    if before == after {
        return base.cloned();
    }
    if depth == 0 {
        return after.cloned();
    }
    let object = |v: Option<&Value>| v.and_then(Value::as_object).cloned().unwrap_or_default();
    let (mut out, b, a) = (object(base), object(before), object(after));
    for k in b.keys().chain(a.keys()).cloned().collect::<BTreeSet<String>>() {
        if b.get(&k) != a.get(&k) {
            let merged = merge_entries(out.get(&k), b.get(&k), a.get(&k), depth - 1);
            set_or_remove(&mut out, &k, merged);
        }
    }
    Some(Value::Object(out))
}

/// A data stream the job changed (`before` -> `ours`) laid onto its
/// committed form: the committed stream, plus only the job's own changes.
/// Backing (and failure) indices the job removed are removed; ones it added
/// go where the job put them (in front: an added backing index; at the end:
/// a rollover, which also raises the generation); other fields changed by
/// the job are the job's, the rest the committed ones.
fn merge_stream(committed: &Value, before: Option<&Value>, ours: &Value) -> Value {
    let mut out = committed.clone();
    let none = json!({});
    let before = before.unwrap_or(&none);
    for list in ["indices", "failure_indices"] {
        let names = |v: &Value| -> Vec<String> { v[list].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
        let (was, now) = (names(before), names(ours));
        if was == now {
            continue;
        }
        let mut all = names(committed);
        all.retain(|n| now.contains(n) || !was.contains(n));
        for (i, n) in now.iter().enumerate() {
            if was.contains(n) || all.contains(n) {
                continue;
            }
            // Anchor additions before the next surviving entry from the
            // job's ordered list. This preserves front insertion order and
            // mid-stream replacements without making them the write index.
            // A true rollover has no following anchor and remains appended.
            let next = now[i + 1..].iter().find_map(|next| all.iter().position(|existing| existing == next));
            all.insert(next.unwrap_or(all.len()), n.clone());
        }
        out[list] = json!(all);
    }
    let fields: BTreeSet<String> = before.as_object().into_iter().chain(ours.as_object()).flat_map(|m| m.keys().cloned()).collect();
    for f in fields {
        if matches!(f.as_str(), "indices" | "failure_indices" | "generation") || before.get(&f) == ours.get(&f) {
            continue;
        }
        match ours.get(&f) {
            Some(v) => out[f.as_str()] = v.clone(),
            None => {
                if let Some(o) = out.as_object_mut() {
                    o.remove(&f);
                }
            }
        }
    }
    let g = |v: &Value| v["generation"].as_u64().unwrap_or(0);
    // The engine has already applied ES generation rules for add/remove,
    // replacements and both backing/failure-store rollovers. Carry its delta
    // rather than guessing from where newly named indices were inserted.
    out["generation"] = json!(g(committed).saturating_add(g(ours).saturating_sub(g(before))));
    out
}

/// The customs an [`apply`] error name belongs to (empty: unknown).
pub fn custom_of_error(name: &str) -> &'static [&'static str] {
    if name.starts_with("pipeline ") {
        &[INGEST_CUSTOM]
    } else if name.starts_with("ilm ") {
        &[ILM_CUSTOM]
    } else if name == "native security state" {
        &[SECURITY_CUSTOM]
    } else if name == "license" {
        &[LICENSE_CUSTOM]
    } else if name == "_ext_state" {
        // Synonyms, rulesets and search applications persist together.
        &[SYNONYMS_CUSTOM, RULESETS_CUSTOM, SEARCH_APPS_CUSTOM]
    } else {
        &[]
    }
}

/// [`apply`] for the master service's scratch engine (`crate::scratch`):
/// the resources metadata requests validate against (license, ILM
/// policies, pipelines, synonym sets, rulesets, search applications), not
/// native security (the scratch engine shares the live security service)
/// or repository storage. Repository registrations are copied only as
/// validation metadata; the scratch engine must never open or mutate their files.
pub fn apply_for_validation(engine: &Engine, md: &Metadata) -> Vec<(String, String)> {
    let mut errors = Vec::new();
    if let Some(error)=apply_esql_views(engine,md) {errors.push((ESQL_VIEWS_CUSTOM.into(),error));}
    engine.snapshots.restore_cluster_registrations(
        md.customs.get("repositories").and_then(Value::as_object)
            .map(|entries| entries.iter().map(|(name, value)| (name.clone(), value.clone())).collect())
            .unwrap_or_default(),
    );
    if let Some(license) = md.customs.get(LICENSE_CUSTOM) {
        if let Err(error) = elasticsearch_engine::license::service(engine).restore_cluster_state(license) {
            errors.push(("license".into(), error.reason));
        }
    }
    apply_ilm(engine, None, md, &mut errors);
    apply_ingest(engine, None, md, &mut errors);
    apply_ext(engine, None, md, &mut errors);
    errors
}

/// Make an engine's inference endpoints the published ones: every engine
/// but the master's own (which owns them, like enrich policies) resolves
/// semantic_text fields and inference queries for the shards it holds.
pub fn apply_inference(engine: &Engine, md: &Metadata) -> Option<String> {
    let endpoints = md.customs.get(INFERENCE_CUSTOM)?;
    elasticsearch_engine::inference::service_for(&engine.data).restore_cluster_state(endpoints).err().map(|e| e.reason)
}

/// Make a non-master node's enrich policies the published ones. The master
/// owns them: a state built by another metadata task can carry an older
/// copy, which must not undo a put or delete the master just made.
pub fn apply_enrich(engine: &Engine, md: &Metadata) -> Option<String> {
    let enrich = md.customs.get(ENRICH_CUSTOM)?;
    elasticsearch_engine::enrich::store(engine).restore_cluster_state(enrich).err().map(|e| e.reason)
}

fn apply_ilm(engine: &Engine, prev: Option<&Metadata>, md: &Metadata, errors: &mut Vec<(String, String)>) {
    // A metadata without the custom predates this module (or comes from a
    // restore): leave the local policies alone.
    let Some(c) = md.customs.get(ILM_CUSTOM) else { return };
    let want = c["policies"].as_object().cloned().unwrap_or_default();
    // With `prev`: only the policies that changed from it.
    let before = prev.map(|p| p.customs.get(ILM_CUSTOM).and_then(|c| c["policies"].as_object().cloned()).unwrap_or_default());
    let ilm = &engine.ilm;
    let local = ilm.st_snapshot();
    for name in local.policies.keys() {
        let removed = match &before {
            Some(b) => b.contains_key(name) && !want.contains_key(name),
            None => !want.contains_key(name),
        };
        if removed {
            let _ = ilm.delete_policy(name);
        }
    }
    for (name, w) in &want {
        if before.as_ref().is_some_and(|b| b.get(name) == Some(w)) {
            continue;
        }
        let version = w["version"].as_u64().unwrap_or(1).max(1);
        let policy = &w["policy"];
        let mut cur = match local.policies.get(name) {
            Some(p) if p.version == version && &p.policy == policy => continue,
            Some(p) if p.version < version => p.version,
            Some(_) => {
                let _ = ilm.delete_policy(name);
                0
            }
            None => 0,
        };
        // The engine only exposes "store with version + 1"; step up to the
        // published version (one step in the usual case of a single put).
        if version - cur > 256 {
            cur = version - 1;
        }
        while cur < version {
            if let Err(e) = ilm.replace_policy(name, policy.clone()) {
                errors.push((format!("ilm policy {name}"), e.reason.clone()));
                break;
            }
            cur += 1;
        }
    }
    let running = c["running"].as_bool().unwrap_or(true);
    if running != (local.mode == "RUNNING") {
        if let Err(e) = ilm.set_mode(running) {
            errors.push(("ilm mode".into(), e.reason.clone()));
        }
    }
    // STOPPED as the master decided it (`ClusterNode::settle_lifecycle_stop`).
    if !running && c["mode"].as_str() == Some("STOPPED") {
        if let Err(e) = ilm.set_stopped() {
            errors.push(("ilm mode".into(), e.reason.clone()));
        }
    }
}

fn pipelines_of(md: &Metadata) -> BTreeMap<String, Value> {
    md.customs
        .get(INGEST_CUSTOM)
        .and_then(|c| c["pipeline"].as_array())
        .map(|a| a.iter().filter_map(|p| Some((p["id"].as_str()?.to_string(), p["config"].clone()))).collect())
        .unwrap_or_default()
}

fn apply_ingest(engine: &Engine, prev: Option<&Metadata>, md: &Metadata, errors: &mut Vec<(String, String)>) {
    let all_want = pipelines_of(md);
    let want = all_want.clone();
    // With `prev`: only the pipelines that changed from it (one this node
    // added outside the cluster state, not yet published, stays).
    let before = prev.map(pipelines_of);
    let store = engine.ingest();
    let have: BTreeMap<String, Value> = store.get(None).into_iter().collect();
    let want: BTreeMap<String, Value> = match &before {
        Some(b) => want.into_iter().filter(|(id, config)| b.get(id) != Some(config)).collect(),
        None => want,
    };
    for id in have.keys() {
        let removed = match &before {
            Some(b) => b.contains_key(id) && !all_want.contains_key(id),
            None => !want.contains_key(id),
        };
        // `delete` takes a wildcard expression; ids with `*` cannot exist.
        if removed && !id.contains('*') {
            if let Err(e) = store.delete(id) {
                errors.push((format!("pipeline {id}"), e.reason.clone()));
            }
        }
    }
    for (id, config) in want {
        if have.get(&id) != Some(&config) {
            if let Err(e) = store.restore_from_cluster_state(&id, config) {
                errors.push((format!("pipeline {id}"), e.reason.clone()));
            }
        }
    }
}

fn apply_ext(engine: &Engine, prev: Option<&Metadata>, md: &Metadata, errors: &mut Vec<(String, String)>) {
    let ext = search_ext::ext(engine);
    let mut dirty = false;
    // An absent family is empty (these customs are dropped when empty).
    let _ = prev;
    let leave = |_: &str| false;

    let want: BTreeMap<String, Vec<(String, String)>> =
        md.customs.get(SYNONYMS_CUSTOM).and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    let mut changed: BTreeSet<String> = BTreeSet::new();
    if !leave(SYNONYMS_CUSTOM) {
        let mut g = ext.synonyms.write();
        if *g != want {
            for k in g.keys().chain(want.keys()) {
                if g.get(k) != want.get(k) {
                    changed.insert(k.clone());
                }
            }
            *g = want;
            dirty = true;
        }
    }

    let want: BTreeMap<String, Vec<Value>> = md.customs.get(RULESETS_CUSTOM).and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    if !leave(RULESETS_CUSTOM) {
        let mut g = ext.rulesets.write();
        if *g != want {
            *g = want;
            dirty = true;
        }
    }
    let want: BTreeMap<String, Value> = md.customs.get(SEARCH_APPS_CUSTOM).and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    if !leave(SEARCH_APPS_CUSTOM) {
        let mut g = ext.search_apps.write();
        if *g != want {
            *g = want;
            dirty = true;
        }
    }

    if dirty {
        if let Err(e) = ext.persist() {
            errors.push(("_ext_state".into(), e.reason.clone()));
        }
    }
    // Feed the analysis chain: analyzers built from now on (index opens,
    // recoveries) use the new rules. Existing analyzers reload only when the
    // update asked for it (`refresh=true`, which the coordinator broadcasts
    // to every node) or on `_reload_search_analyzers`, as in ES: a
    // `refresh=false` update must not change search results yet.
    for set in changed {
        ext.publish_synonyms(&set);
    }
}

/// Delete this node's copy of `index` without changing data stream
/// membership.
///
/// `Engine::delete_index` treats the call as a user delete: it refuses the
/// write index of a data stream and removes any other backing index from its
/// stream. Wiping a local shard copy before peer recovery is neither: the
/// index stays in the cluster and in its stream. Use the recovery deletion
/// operation so the stream is never temporarily absent from concurrent reads.
pub fn delete_local_copy(engine: &Engine, index: &str) -> Result<(), String> {
    engine.delete_local_index_copy(index).map_err(|error| error.reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Combined reviews: an engine change laid onto the committed custom
    /// never reverts committed entries engine 0 lacks, replaces entries
    /// whole, raises revisions, and keeps committed data stream changes.
    #[test]
    fn engine_changes_merge_onto_the_committed_customs() {
        // Security: committed user `s` engine 0 never applied; a PUT of `u`.
        let base = json!({"native_revision": 10, "users": {"s": {"roles": ["a"]}}, "roles": {"r": {"run_as": ["x"], "cluster": ["all"]}}});
        let before = json!({"native_revision": 9, "users": {}, "roles": {"r": {"run_as": ["x"], "cluster": ["all"]}}});
        let after = json!({"native_revision": 10, "users": {"u": {"roles": ["b"]}}, "roles": {"r": {"cluster": ["all"]}}});
        let m = merge_custom_change(SECURITY_CUSTOM, Some(&base), Some(&before), Some(&after)).unwrap();
        assert!(m["users"]["s"].is_object() && m["users"]["u"].is_object(), "{m}");
        assert!(m["roles"]["r"].get("run_as").is_none(), "a replaced role keeps no stale field: {m}");
        assert_eq!(m["native_revision"], json!(11), "above every node's revision");
        // An absent committed custom is empty, not a reason to revert.
        let m = merge_custom_change(SYNONYMS_CUSTOM, Some(&json!({"t": [], "s": []})), None, Some(&json!({"s": []}))).unwrap();
        assert!(m.get("t").is_some(), "{m}");
        // Data streams: a rollover committed meanwhile survives a job that
        // changed only the stream's `_meta`.
        let committed = json!({"s": {"indices": [".ds-s-000001", ".ds-s-000002"], "generation": 2, "meta": {"v": 1}}});
        let was = json!({"s": {"indices": [".ds-s-000001"], "generation": 1, "meta": {"v": 1}}});
        let ours = json!({"s": {"indices": [".ds-s-000001"], "generation": 1, "meta": {"v": 2}}});
        let m = merge_custom_change(crate::mirror::DATA_STREAMS_CUSTOM, Some(&committed), Some(&was), Some(&ours)).unwrap();
        assert_eq!(m["s"]["indices"], json!([".ds-s-000001", ".ds-s-000002"]), "{m}");
        assert_eq!((m["s"]["generation"].clone(), m["s"]["meta"]["v"].clone()), (json!(2), json!(2)));
        // A stream deleted in the committed state is not resurrected.
        let m = merge_custom_change(crate::mirror::DATA_STREAMS_CUSTOM, Some(&json!({})), Some(&was), Some(&ours));
        assert!(m.is_none_or(|m| m.get("s").is_none()));
        // Inference endpoints by id.
        let base = json!({"endpoints": [{"inference_id": "x", "v": 1}], "secrets": {}});
        let before = json!({"endpoints": [], "secrets": {}});
        let after = json!({"endpoints": [{"inference_id": "y", "v": 1}], "secrets": {"y": {"k": 1}}});
        let m = merge_custom_change(INFERENCE_CUSTOM, Some(&base), Some(&before), Some(&after)).unwrap();
        assert_eq!(m["endpoints"].as_array().map(Vec::len), Some(2), "{m}");
        assert!(m["secrets"]["y"].is_object());
    }

    #[test]
    fn stream_merge_preserves_positions_and_generation_deltas() {
        let before = json!({"indices":["a","b","c"],"failure_indices":["f1"],"generation":3});
        let committed = json!({"indices":["a","b","c","concurrent-write"],"failure_indices":["f1"],"generation":4});
        let cases = [
            (json!({"indices":["a","replacement","c"],"failure_indices":["f1"],"generation":4}), json!(["a","replacement","c","concurrent-write"]), 5),
            (json!({"indices":["front1","front2","a","b","c"],"failure_indices":["f1"],"generation":5}), json!(["front1","front2","a","b","c","concurrent-write"]), 6),
            (json!({"indices":["a","c"],"failure_indices":["f1"],"generation":4}), json!(["a","c","concurrent-write"]), 5),
            (json!({"indices":["a","b","c"],"failure_indices":["f1","f2"],"generation":4}), json!(["a","b","c","concurrent-write"]), 5),
        ];
        for (ours, expected, generation) in cases {
            let merged = merge_stream(&committed, Some(&before), &ours);
            assert_eq!(merged["indices"], expected);
            assert_eq!(merged["failure_indices"], ours["failure_indices"]);
            assert_eq!(merged["generation"], json!(generation));
        }
    }

    #[test]
    fn recovery_deletion_preserves_backing_and_failure_membership() {
        let directory = tempfile::tempdir().unwrap();
        let engine = Engine::open(directory.path()).unwrap();
        engine.create_index(".ds-metrics-000001", &json!({})).unwrap();
        engine.create_index(".fs-metrics-000002", &json!({})).unwrap();
        let stream = json!({"name":"metrics","generation":2,"template":"metrics",
            "indices":[".ds-metrics-000001"],"failure_indices":[".fs-metrics-000002"]});
        engine.data_streams.write().insert("metrics".into(), serde_json::from_value(stream.clone()).unwrap());
        delete_local_copy(&engine, ".ds-metrics-000001").unwrap();
        delete_local_copy(&engine, ".fs-metrics-000002").unwrap();
        let preserved = engine.data_stream("metrics").unwrap();
        assert_eq!(preserved.generation, 2);
        assert_eq!(preserved.indices, vec![".ds-metrics-000001"]);
        assert_eq!(preserved.failure_indices, vec![".fs-metrics-000002"]);
        assert!(!preserved.failure_needs_rollover());
    }

    #[test]
    fn repository_and_slm_health_follow_cluster_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let mut metadata = Metadata::default();
        metadata.customs.insert("repositories".into(), json!({"repo":{"type":"fs","settings":{"location":"/cluster/repo"}}}));
        metadata.customs.insert("snapshot_lifecycle".into(), json!({"operation_mode":"STOPPED", "policies":{"daily":{"policy":{"name":"daily", "schedule":"0 0 0 * * ?", "repository":"repo"}}}}));
        assert!(apply(&engine, &metadata).is_empty());
        assert_eq!(engine.snapshots.registered_readonly("repo"), Some(false));
        let health = elasticsearch_engine::ilm::health_indicator(&engine, "slm");
        assert_eq!(health["status"], "yellow");
        assert_eq!(health["details"]["policies"], 1);
        metadata.customs.remove("repositories");
        metadata.customs.remove("snapshot_lifecycle");
        assert!(apply(&engine, &metadata).is_empty());
        assert_eq!(engine.snapshots.registered_readonly("repo"), None);
        assert_eq!(elasticsearch_engine::ilm::health_indicator(&engine, "slm")["details"]["policies"], 0);
    }

    #[test]
    #[ignore = "licence hardwired to trial"]
    fn license_updates_follow_authoritative_metadata() {
        let master_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let master = Engine::open(master_dir.path()).unwrap();
        let peer = Engine::open(peer_dir.path()).unwrap();
        let authority = elasticsearch_engine::license::service(&master);
        authority.start_trial(&master, "trial", true);
        let mut metadata = Metadata::default();
        extract(&master, &mut metadata);
        assert!(is_internal_custom(LICENSE_CUSTOM));
        assert!(is_internal_custom(SECURITY_CUSTOM));
        assert!(apply(&peer, &metadata).is_empty());
        assert_eq!(elasticsearch_engine::license::current(&master), elasticsearch_engine::license::current(&peer));
        authority.start_basic(&master, true);
        extract(&master, &mut metadata);
        assert!(apply(&peer, &metadata).is_empty());
        assert_eq!(elasticsearch_engine::license::current(&peer).mode(), elasticsearch_engine::license::Mode::Basic);
        assert!(!elasticsearch_engine::license::service(&peer).eligible_for_trial());
    }
}
