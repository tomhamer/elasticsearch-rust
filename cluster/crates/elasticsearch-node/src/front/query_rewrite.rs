//! Materialize queries that read another document (terms lookup, percolate
//! by id, `indexed_shape`, `more_like_this` items by id) once, through the
//! ordinary document route, as ES's coordinator rewrite does: a shard can
//! only read the documents it holds.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone)]
struct Lookup {
    pointer: String,
    index: String,
    id: String,
    path: String,
    routing: Option<String>,
}

/// A query object (at `pointer`) whose documents are fetched and inlined.
#[derive(Clone)]
enum DocQuery {
    /// `percolate` with `index` / `id`: the document becomes `document`.
    Percolate { index: String, id: String, routing: Option<String> },
    /// `geo_shape` / `shape` `indexed_shape` of `field`: the value at `path`.
    Shape { kind: String, field: String, index: String, id: String, path: String, routing: Option<String> },
    /// `more_like_this` `like` / `unlike` items naming `_index` + `_id`.
    MoreLikeThis { items: Vec<(String, usize, String, String, Option<String>)> },
}

fn escape(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

/// An object or array: what [`collect`] and [`collect_docs`] descend into.
fn is_container(value: &Value) -> bool {
    matches!(value, Value::Object(_) | Value::Array(_))
}

fn collect_docs(value: &Value, pointer: &str, out: &mut Vec<(String, DocQuery)>) {
    match value {
        Value::Object(object) => {
            if let Some(p) = object.get("percolate").and_then(Value::as_object) {
                if let (Some(index), Some(id)) = (p.get("index").and_then(Value::as_str), p.get("id").and_then(Value::as_str)) {
                    out.push((format!("{pointer}/percolate"), DocQuery::Percolate { index: index.into(), id: id.into(), routing: p.get("routing").and_then(Value::as_str).map(str::to_string) }));
                }
            }
            for kind in ["geo_shape", "shape"] {
                for (field, spec) in object.get(kind).and_then(Value::as_object).into_iter().flatten() {
                    let Some(shape) = spec.get("indexed_shape").and_then(Value::as_object) else { continue };
                    if let (Some(index), Some(id)) = (shape.get("index").and_then(Value::as_str), shape.get("id").and_then(Value::as_str)) {
                        out.push((format!("{pointer}/{kind}/{}", escape(field)), DocQuery::Shape {
                            kind: kind.into(), field: field.clone(), index: index.into(), id: id.into(),
                            path: shape.get("path").and_then(Value::as_str).unwrap_or("shape").into(),
                            routing: shape.get("routing").and_then(Value::as_str).map(str::to_string),
                        }));
                    }
                }
            }
            if let Some(mlt) = object.get("more_like_this").and_then(Value::as_object) {
                let mut items = Vec::new();
                for key in ["like", "unlike"] {
                    let list: Vec<&Value> = match mlt.get(key) {
                        Some(Value::Array(a)) => a.iter().collect(),
                        Some(v) => vec![v],
                        None => Vec::new(),
                    };
                    for (i, item) in list.into_iter().enumerate() {
                        if let (Some(index), Some(id), None) = (item.get("_index").and_then(Value::as_str), item.get("_id").and_then(Value::as_str), item.get("doc")) {
                            items.push((key.to_string(), i, index.to_string(), id.to_string(), item.get("routing").and_then(Value::as_str).map(str::to_string)));
                        }
                    }
                }
                if !items.is_empty() {
                    out.push((pointer.to_string(), DocQuery::MoreLikeThis { items }));
                }
            }
            // Scalars hold no lookups: no pointer for them (a kNN query
            // vector is hundreds of numbers).
            for (key, child) in object.iter().filter(|(_, c)| is_container(c)) {
                collect_docs(child, &format!("{pointer}/{}", escape(key)), out);
            }
        }
        Value::Array(array) => {
            for (index, child) in array.iter().enumerate().filter(|(_, c)| is_container(c)) {
                collect_docs(child, &format!("{pointer}/{index}"), out);
            }
        }
        _ => {}
    }
}

/// Inline the fetched documents (`fetch(index, id, routing)`: `_index` and
/// `_source` of the GET) into one query.
fn apply_doc(body: &mut Value, pointer: &str, q: &DocQuery, fetch: &dyn Fn(&str, &str, &Option<String>) -> Value) {
    let Some(slot) = body.pointer_mut(pointer) else { return };
    match q {
        DocQuery::Percolate { index, id, routing } => {
            let Some(o) = slot.as_object_mut() else { return };
            for k in ["index", "id", "routing", "preference", "version"] {
                o.shift_remove(k);
            }
            o.insert("document".into(), fetch(index, id, routing)["_source"].clone());
        }
        DocQuery::Shape { kind, field, index, id, path, routing } => {
            let _ = (kind, field);
            let source = fetch(index, id, routing)["_source"].clone();
            let shape = elasticsearch_engine::highlight::source_values(&source, path).into_iter().next().cloned().unwrap_or(Value::Null);
            if let Some(o) = slot.as_object_mut() {
                o.shift_remove("indexed_shape");
                o.insert("shape".into(), shape);
            }
        }
        DocQuery::MoreLikeThis { items } => {
            let mut exclude = Vec::new();
            let Some(mlt) = slot.get_mut("more_like_this").and_then(Value::as_object_mut) else { return };
            let include = mlt.get("include").and_then(Value::as_bool).unwrap_or(false);
            let query_fields = mlt.get("fields").cloned();
            for (key, i, index, id, routing) in items {
                // A missing document contributes no terms (ES skips it).
                let got = fetch(index, id, routing);
                let concrete = got["_index"].as_str().unwrap_or(index).to_string();
                let source = got["_source"].clone();
                let mut doc = json!({"_index": concrete, "doc": if source.is_object() { source } else { json!({}) }});
                let slot = match mlt.get_mut(key.as_str()) {
                    Some(Value::Array(a)) => a.get_mut(*i),
                    Some(v) => Some(v),
                    None => None,
                };
                let Some(slot) = slot else { continue };
                // An `_id` item reads the query's `fields` (or its own); an
                // artificial doc would read all of its fields.
                if let Some(f) = slot.get("fields").cloned().or_else(|| query_fields.clone()) {
                    doc["fields"] = f;
                }
                *slot = doc;
                if key == "like" && !include {
                    exclude.push(json!({"bool": {"filter": [{"term": {"_index": concrete}}, {"ids": {"values": [id]}}]}}));
                }
            }
            // ES leaves the liked documents out of the hits unless `include`.
            if !exclude.is_empty() {
                let inner = std::mem::take(slot);
                *slot = json!({"bool": {"must": [inner], "must_not": exclude}});
            }
        }
    }
}

fn collect(value: &Value, pointer: &str, lookups: &mut Vec<Lookup>) {
    match value {
        Value::Object(object) => {
            if let Some(terms) = object.get("terms").and_then(Value::as_object) {
                for (field, spec) in terms {
                    if let (Some(index), Some(id), Some(path)) = (spec["index"].as_str(), spec["id"].as_str(), spec["path"].as_str()) {
                        lookups.push(Lookup {
                            pointer: format!("{pointer}/terms/{}", field.replace('~', "~0").replace('/', "~1")),
                            index: index.into(), id: id.into(), path: path.into(),
                            routing: spec["routing"].as_str().map(str::to_string),
                        });
                    }
                }
            }
            for (key, child) in object.iter().filter(|(_, c)| is_container(c)) {
                collect(child, &format!("{pointer}/{}", key.replace('~', "~0").replace('/', "~1")), lookups);
            }
        }
        Value::Array(array) => {
            for (index, child) in array.iter().enumerate().filter(|(_, c)| is_container(c)) {
                collect(child, &format!("{pointer}/{index}"), lookups);
            }
        }
        _ => {}
    }
}

/// Lookup vector builders use a coordinator search with doc values, as ES
/// does, rather than trying to read another index from each target shard.
fn collect_vectors(value: &Value, pointer: &str, out: &mut Vec<Lookup>) {
    // Only visit actual search DSL positions. Script params, runtime mappings,
    // term values and source-like literal objects are opaque user data.
    for key in ["query", "post_filter"] {
        if let Some(query) = value.get(key) { collect_vector_query(query, &format!("{pointer}/{key}"), out); }
    }
    if let Some(knn) = value.get("knn") { collect_vector_knn(knn, &format!("{pointer}/knn"), out); }
    if let Some(rescore)=value.get("rescore") { collect_vector_rescore(rescore,&format!("{pointer}/rescore"),out); }
    if let Some(retriever)=value.get("retriever") { collect_vector_retriever(retriever,&format!("{pointer}/retriever"),out); }
    for key in ["aggs","aggregations"] { if let Some(aggs)=value.get(key) { collect_vector_aggs(aggs,&format!("{pointer}/{key}"),out); } }
    if let Some(subs)=value.get("sub_searches").and_then(Value::as_array) { for(i,sub) in subs.iter().enumerate() { if let Some(query)=sub.get("query") {collect_vector_query(query,&format!("{pointer}/sub_searches/{i}/query"),out);} } }
}

fn collect_vector_rescore(value:&Value,pointer:&str,out:&mut Vec<Lookup>) {
    if let Some(items)=value.as_array() {for(i,item) in items.iter().enumerate(){collect_vector_rescore(item,&format!("{pointer}/{i}"),out);}return;}
    if let Some(query)=value.get("query").and_then(|q|q.get("rescore_query")){collect_vector_query(query,&format!("{pointer}/query/rescore_query"),out);}
}

fn collect_vector_aggs(value:&Value,pointer:&str,out:&mut Vec<Lookup>) {
    let Some(aggs)=value.as_object() else{return};
    for(name,spec) in aggs {
        let path=format!("{pointer}/{}",escape(name));
        if let Some(query)=spec.get("filter"){collect_vector_query(query,&format!("{path}/filter"),out);}
        if let Some(filters)=spec.get("filters").and_then(|f|f.get("filters")) {
            let path=format!("{path}/filters/filters");
            if let Some(named)=filters.as_object(){for(name,q) in named{collect_vector_query(q,&format!("{path}/{}",escape(name)),out);}}
            else {collect_vector_query(filters,&path,out);}
        }
        for kind in ["significant_terms","significant_text"] {if let Some(q)=spec.get(kind).and_then(|s|s.get("background_filter")){collect_vector_query(q,&format!("{path}/{kind}/background_filter"),out);}}
        for key in ["aggs","aggregations"]{if let Some(sub)=spec.get(key){collect_vector_aggs(sub,&format!("{path}/{key}"),out);}}
    }
}

fn collect_vector_retriever(value:&Value,pointer:&str,out:&mut Vec<Lookup>) {
    let Some(object)=value.as_object() else{return};
    for(kind,spec) in object {
        if !matches!(kind.as_str(),"standard"|"knn"|"rrf"|"linear"|"rescorer"|"rule"|"pinned"|"text_similarity_reranker"|"random_reranker"|"diversify"){continue;}
        let path=format!("{pointer}/{}",escape(kind));
        if kind!="knn" {if let Some(q)=spec.get("filter"){collect_vector_query(q,&format!("{path}/filter"),out);}}
        if kind=="standard" {if let Some(q)=spec.get("query"){collect_vector_query(q,&format!("{path}/query"),out);}}
        if kind=="knn" {collect_vector_knn(spec,&path,out);}
        if kind=="rescorer" {if let Some(r)=spec.get("rescore"){collect_vector_rescore(r,&format!("{path}/rescore"),out);}}
        if let Some(child)=spec.get("retriever"){collect_vector_retriever(child,&format!("{path}/retriever"),out);}
        if let Some(children)=spec.get("retrievers").and_then(Value::as_array){for(i,child) in children.iter().enumerate(){
            let path=format!("{path}/retrievers/{i}");
            if let Some(r)=child.get("retriever"){collect_vector_retriever(r,&format!("{path}/retriever"),out);}else{collect_vector_retriever(child,&path,out);}
        }}
    }
}

fn collect_vector_query(value: &Value, pointer: &str, out: &mut Vec<Lookup>) {
    if let Some(items) = value.as_array() {
        for (i, query) in items.iter().enumerate() { collect_vector_query(query, &format!("{pointer}/{i}"), out); }
        return;
    }
    let Some(object) = value.as_object() else { return };
    for (kind, spec) in object {
        let path = format!("{pointer}/{}",escape(kind));
        let keys: &[&str] = match kind.as_str() {
            "knn" => { collect_vector_knn(spec,&path,out); continue; }
            "bool" => &["must","should","filter","must_not"],
            "constant_score" => &["filter"],
            "boosting" => &["positive","negative"],
            "dis_max" => &["queries"],
            "nested" | "has_child" | "has_parent" | "script_score" | "function_score" => &["query"],
            "pinned" => &["organic"],
            _ => &[],
        };
        for key in keys { if let Some(query)=spec.get(*key) { collect_vector_query(query,&format!("{path}/{key}"),out); } }
        if kind=="function_score" {if let Some(functions)=spec.get("functions").and_then(Value::as_array){for(i,f) in functions.iter().enumerate(){if let Some(q)=f.get("filter"){collect_vector_query(q,&format!("{path}/functions/{i}/filter"),out);}}}}
    }
}

fn collect_vector_knn(value: &Value, pointer: &str, out: &mut Vec<Lookup>) {
    if let Some(items)=value.as_array() {
        for (i, knn) in items.iter().enumerate() { collect_vector_knn(knn,&format!("{pointer}/{i}"),out); }
        return;
    }
    let Some(object)=value.as_object() else {return};
    if let Some(filter)=object.get("filter"){collect_vector_query(filter,&format!("{pointer}/filter"),out);}
    let Some(builder)=object.get("query_vector_builder").and_then(Value::as_object) else {return};
    if builder.len()!=1 || object.contains_key("query_vector") {return;}
    let Some(spec)=builder.get("lookup").and_then(Value::as_object) else {return};
    if !spec.keys().all(|k|matches!(k.as_str(),"index"|"id"|"path"|"routing")) || !spec.get("routing").is_none_or(Value::is_string) {return;}
    if let (Some(index),Some(id),Some(path))=(spec.get("index").and_then(Value::as_str),spec.get("id").and_then(Value::as_str),spec.get("path").and_then(Value::as_str)) {
        out.push(Lookup {pointer:pointer.into(),index:index.into(),id:id.into(),path:path.into(),routing:spec.get("routing").and_then(Value::as_str).map(str::to_string)});
    }
}

fn lookup_vector_value(lookup: &Lookup, response: &Value) -> Result<Value, RestResp> {
    let hit = response["hits"]["hits"].as_array().and_then(|a| a.first()).ok_or_else(|| RestResp::error(404, "status_exception", &format!("Document with id [{}] not found in index [{}] for lookup query vector builder.", lookup.id, lookup.index)))?;
    let value = hit["fields"][&lookup.path].as_array().and_then(|a| a.first()).filter(|v| !v.is_null()).ok_or_else(|| RestResp::error(404, "status_exception", &format!("Field [{}] not found in document with id [{}] for lookup query vector builder.", lookup.path, lookup.id)))?;
    if !value.as_array().is_some_and(|values| values.iter().all(Value::is_number)) {
        let class = if value.is_string() { "java.lang.String" } else if value.is_boolean() { "java.lang.Boolean" } else { "java.lang.Long" };
        return Err(RestResp::error(400, "illegal_argument_exception", &format!("Field [{}] in document with id [{}] is of unsupported type [{class}] for lookup query vector builder.", lookup.path, lookup.id)));
    }
    Ok(value.clone())
}

/// Whether [`rewrite`] has anything to do for `body` (else it answers
/// `Ok(None)` without a request): checked on the borrowed body, so a search
/// with nothing to look up is not copied.
pub(super) fn needed(body: &Value) -> bool {
    let mut vectors = Vec::new();
    collect_vectors(body, "", &mut vectors);
    if !vectors.is_empty() {
        return true;
    }
    let mut lookups = Vec::new();
    collect(body, "", &mut lookups);
    if !lookups.is_empty() {
        return true;
    }
    let mut docs = Vec::new();
    collect_docs(body, "", &mut docs);
    !docs.is_empty()
}

pub(super) fn rewrite<'a>(node: &'a Arc<ClusterNode>, request: &'a RestReq, body: Value) -> futures::future::BoxFuture<'a, Result<Option<Value>, RestResp>> {
    Box::pin(rewrite_inner(node, request, body))
}

async fn rewrite_inner(node: &Arc<ClusterNode>, request: &RestReq, mut body: Value) -> Result<Option<Value>, RestResp> {
    let mut vectors = Vec::new();
    collect_vectors(&body, "", &mut vectors);
    for lookup in &vectors {
        let mut search = request.clone();
        search.set_method("POST").map_err(|e|RestResp::json(e.status,&e.to_json()))?;
        let mut uri = format!("/{}/_search?preference=_local", encode_component(&lookup.index));
        if let Some(routing) = &lookup.routing { uri.push_str(&format!("&routing={}", encode_component(routing))); }
        search.set_uri(&uri).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
        search.set_json(&json!({"query":{"ids":{"values":[lookup.id]}},"_source":false,"stored_fields":"_none_","docvalue_fields":[lookup.path],"size":1})).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
        search.retain_headers(|(name,_)|!name.eq_ignore_ascii_case(EXEC_LOCAL)&&!name.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER)).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
        let response = Box::pin(dispatch(node, search)).await;
        if !response.is_success() { return Err(response); }
        let vector = lookup_vector_value(lookup, &response.body_json().unwrap_or(Value::Null))?;
        if let Some(slot) = body.pointer_mut(&lookup.pointer).and_then(Value::as_object_mut) {
            slot.remove("query_vector_builder");
            slot.insert("query_vector".into(), vector);
        }
    }
    let mut lookups = Vec::new();
    collect(&body, "", &mut lookups);
    let mut docs = Vec::new();
    collect_docs(&body, "", &mut docs);
    if lookups.is_empty() && docs.is_empty() { return Ok((!vectors.is_empty()).then_some(body)); }
    let mut sources: BTreeMap<(String, String, Option<String>), Value> = BTreeMap::new();
    let mut wanted: Vec<(String, String, Option<String>)> = lookups.iter().map(|l| (l.index.clone(), l.id.clone(), l.routing.clone())).collect();
    for (_, q) in &docs {
        match q {
            DocQuery::Percolate { index, id, routing } | DocQuery::Shape { index, id, routing, .. } => wanted.push((index.clone(), id.clone(), routing.clone())),
            DocQuery::MoreLikeThis { items } => wanted.extend(items.iter().map(|(_, _, index, id, routing)| (index.clone(), id.clone(), routing.clone()))),
        }
    }
    // ES fails these when the document is missing (MLT items and terms
    // lookups just contribute nothing).
    let required: BTreeMap<(String, String, Option<String>), String> = docs.iter().filter_map(|(_, q)| match q {
        DocQuery::Percolate { index, id, routing } => Some(((index.clone(), id.clone(), routing.clone()), format!("indexed document [{index}/{id}] couldn't be found"))),
        DocQuery::Shape { index, id, routing, .. } => Some(((index.clone(), id.clone(), routing.clone()), format!("Shape with ID [{id}] not found"))),
        DocQuery::MoreLikeThis { .. } => None,
    }).collect();
    for key in wanted {
        if !sources.contains_key(&key) {
            let lookup = Lookup { pointer: String::new(), index: key.0.clone(), id: key.1.clone(), path: String::new(), routing: key.2.clone() };
            let mut get = request.clone();
            get.set_method("GET").map_err(|e|RestResp::json(e.status,&e.to_json()))?;
            get.set_body(&[]).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
            let mut uri = format!("/{}/_doc/{}", encode_component(&lookup.index), encode_component(&lookup.id));
            if let Some(routing) = &lookup.routing {
                uri.push_str(&format!("?routing={}", encode_component(routing)));
            }
            get.set_uri(&uri).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
            get.retain_headers(|(name,_)|!name.eq_ignore_ascii_case(EXEC_LOCAL)&&!name.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER)).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
            let response = Box::pin(dispatch(node, get)).await;
            let value = response.body_json().unwrap_or(Value::Null);
            if !response.is_success() && !(response.status == 404 && value["found"] == false) {
                return Err(response);
            }
            if let (Some(why), false) = (required.get(&key), value["found"] == true) {
                return Err(RestResp::error(404, "resource_not_found_exception", why));
            }
            // The concrete index too (the item may name an alias).
            sources.insert(key.clone(), json!({"_index": value["_index"], "_source": value["_source"]}));
        }
    }
    for lookup in lookups {
        let key = (lookup.index.clone(), lookup.id.clone(), lookup.routing.clone());
        let source = &sources[&key]["_source"];
        let values = elasticsearch_engine::highlight::source_values(source, &lookup.path).into_iter().cloned().collect();
        if let Some(slot) = body.pointer_mut(&lookup.pointer) { *slot = Value::Array(values); }
    }
    // Deepest first, so a wrapped `more_like_this` cannot move another's pointer.
    docs.sort_by_key(|(p, _)| std::cmp::Reverse(p.matches('/').count()));
    let fetch = |index: &str, id: &str, routing: &Option<String>| sources.get(&(index.to_string(), id.to_string(), routing.clone())).cloned().unwrap_or(Value::Null);
    for (pointer, q) in &docs {
        apply_doc(&mut body, pointer, q, &fetch);
    }
    Ok(Some(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_lookup_visits_supported_query_containers_without_parameters() {
        let knn=json!({"knn":{"field":"v","query_vector_builder":{"lookup":{"index":"i","id":"1","path":"v"}}}});
        let request=json!({"query":{"function_score":{"functions":[{"filter":knn,"script_score":{"script":{"params":{"query":knn}}}}]}},
            "aggs":{"a":{"filter":knn,"aggs":{"b":{"filters":{"filters":{"named":knn}}}}}},
            "rescore":[{"query":{"rescore_query":knn}}],
            "retriever":{"rrf":{"retrievers":[{"standard":{"query":knn}},{"retriever":{"knn":knn["knn"]}}]}},
            "knn":{"field":"v","query_vector":[1.0],"filter":knn}});
        let mut found=Vec::new();collect_vectors(&request,"",&mut found);
        assert_eq!(found.len(),7);
        assert!(found.iter().all(|l|!l.pointer.contains("params")));
    }

    #[test]
    fn vector_lookup_uses_decoded_doc_values_and_preserves_errors() {
        let request = json!({"knn":[{"field":"v","query_vector_builder":{"lookup":{"index":"other","id":"1","path":"v","routing":"r"}}}]});
        let mut lookups = Vec::new(); collect_vectors(&request, "", &mut lookups);
        assert_eq!(lookups.len(),1); assert_eq!(lookups[0].pointer,"/knn/0");
        assert_eq!(lookups[0].routing.as_deref(),Some("r"));
        let lookup = &lookups[0];
        assert_eq!(lookup_vector_value(lookup,&json!({"hits":{"hits":[{"fields":{"v":[[0.5,-1.0,2.0]]}}]}})).unwrap(),json!([0.5,-1.0,2.0]));
        assert_eq!(lookup_vector_value(lookup,&json!({"hits":{"hits":[]}})).unwrap_err().status,404);
        assert_eq!(lookup_vector_value(lookup,&json!({"hits":{"hits":[{"fields":{}}]}})).unwrap_err().status,404);
        assert_eq!(lookup_vector_value(lookup,&json!({"hits":{"hits":[{"fields":{"v":["text"]}}]}})).unwrap_err().status,400);
        let literal=json!({"query_vector_builder":{"lookup":{"index":"missing","id":"1","path":"v"}}});
        let opaque=json!({"query":{"script":{"script":{"source":"return true","params":{"knn":literal}}}},"runtime_mappings":{"x":{"type":"long","script":{"params":{"knn":literal}}}},"_source":literal});
        let mut hidden=Vec::new();collect_vectors(&opaque,"",&mut hidden);assert!(hidden.is_empty());
        let nested=json!({"query":{"bool":{"filter":[{"knn":{"field":"v","query_vector_builder":{"lookup":{"index":"i","id":"1","path":"v"}}}}]}}});
        collect_vectors(&nested,"",&mut hidden);assert_eq!(hidden.len(),1);assert_eq!(hidden[0].pointer,"/query/bool/filter/0/knn");
        let mut invalid=Vec::new(); collect_vectors(&json!({"knn":{"query_vector_builder":{"lookup":{"index":"i","id":"1","path":"v","unknown":true}}}}),"",&mut invalid);
        assert!(invalid.is_empty(),"invalid builders remain for strict parsing");
        for routing in [Value::Null,json!(7),json!(true),json!({"x":1})] {
            collect_vectors(&json!({"knn":{"query_vector_builder":{"lookup":{"index":"i","id":"1","path":"v","routing":routing}}}}),"",&mut invalid);
            assert!(invalid.is_empty(),"non-string routing remains for strict parsing");
        }
    }

    #[test]
    fn collects_nested_lookup_queries_without_rewriting_literal_terms() {
        let value = json!({"query":{"bool":{"filter":[
            {"terms":{"user/name":{"index":"lookup","id":"one","path":"followers", "routing":"team"}}},
            {"terms":{"user":["one","two"]}}
        ]}}});
        let mut found = Vec::new();
        collect(&value, "", &mut found);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].routing.as_deref(), Some("team"));
        assert!(value.pointer(&found[0].pointer).unwrap().is_object());
    }

    #[test]
    fn document_lookups_are_inlined() {
        let mut body = json!({"query": {"bool": {"must": [
            {"percolate": {"field": "q", "index": "docs", "id": "1"}},
            {"geo_shape": {"area": {"indexed_shape": {"index": "shapes", "id": "s", "path": "geo.outline"}, "relation": "within"}}},
            {"more_like_this": {"fields": ["body"], "like": [{"_index": "docs", "_id": "1"}, "free text", {"_index": "docs", "_id": "missing"}], "min_term_freq": 1}}
        ]}}});
        let mut docs = Vec::new();
        collect_docs(&body, "", &mut docs);
        assert_eq!(docs.len(), 3);
        docs.sort_by_key(|(p, _)| std::cmp::Reverse(p.matches('/').count()));
        let fetch = |index: &str, id: &str, _: &Option<String>| match (index, id) {
            ("docs", "1") => json!({"_index": "docs-000001", "_source": {"body": "hello world"}}),
            ("shapes", "s") => json!({"_index": "shapes", "_source": {"geo": {"outline": {"type": "point", "coordinates": [1, 2]}}}}),
            _ => json!({"_index": null, "_source": null}),
        };
        for (pointer, q) in &docs {
            apply_doc(&mut body, pointer, q, &fetch);
        }
        let must = &body["query"]["bool"]["must"];
        assert_eq!(must[0]["percolate"], json!({"field": "q", "document": {"body": "hello world"}}));
        assert_eq!(must[1]["geo_shape"]["area"], json!({"relation": "within", "shape": {"type": "point", "coordinates": [1, 2]}}));
        let mlt = &must[2]["bool"]["must"][0]["more_like_this"];
        assert_eq!(mlt["like"][0], json!({"_index": "docs-000001", "doc": {"body": "hello world"}, "fields": ["body"]}));
        assert_eq!(must[2]["bool"]["must_not"][0]["bool"]["filter"][0], json!({"term": {"_index": "docs-000001"}}));
        assert_eq!(mlt["like"][1], "free text");
        assert_eq!(mlt["like"][2], json!({"_index": "docs", "doc": {}, "fields": ["body"]}));
        assert_eq!(must[2]["bool"]["must_not"].as_array().unwrap().len(), 2);
    }
}
