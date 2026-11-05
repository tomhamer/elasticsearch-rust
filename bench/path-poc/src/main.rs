//! Restricted single-shard search reproduction. See README for production mapping.
use axum::{
    Router,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    routing::post,
};
use rustic_core::query::query::Bound;
use rustic_core::query::{
    LeafReader, Query, SearchOptions, Searcher, SortField, SortKind, SortValue, TrackTotalHits,
};
use rustic_core::{Document, Field, FieldType, Index, IndexSettings, Term};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader},
    sync::{Arc, RwLock},
    time::Instant,
};

mod http_bench;
mod in_process;
mod index_poc;
mod knn_path;
mod load_bench;
mod fsync_poc;
mod http_overhead_poc;
mod http_write_poc;
mod write_path_poc;
mod nested_datehist_poc;
mod nested_poc;
mod percolate_poc;
mod sort_poc;
mod store_poc;

const ID: u16 = 0;
const SOURCE: u16 = 6;
fn field(name: &str) -> Result<u16, String> {
    match name {
        "title" => Ok(1),
        "body" => Ok(2),
        "price" => Ok(3),
        "ts" => Ok(4),
        "tag" => Ok(5),
        _ => Err(format!("unsupported field {name}")),
    }
}
fn schema(vector: Option<&str>) -> Vec<Field> {
    let mut fields: Vec<Field> = [
        ("_id", FieldType::Keyword),
        ("title", FieldType::Text),
        ("body", FieldType::Text),
        ("price", FieldType::F64),
        ("ts", FieldType::I64),
        ("tag", FieldType::Keyword),
        ("_source", FieldType::Keyword),
    ]
    .into_iter()
    .enumerate()
    .map(|(id, (name, ty))| {
        let mut f = Field::new(id as u16, name, ty);
        f.stored = id == ID as usize || id == SOURCE as usize;
        if id == 3 || id == 4 || id == SOURCE as usize {
            f.indexed = false;
        }
        if id == SOURCE as usize {
            f.fast = false;
        }
        f
    })
    .collect();
    if let Some(kind) = vector {
        let mut f = Field::new(7, "v", FieldType::Vector { dim: 128 });
        f.stored = false;
        f.vector_options = Some(if kind == "int8" {
            rustic_core::vector::store::VectorFieldOptions::quantized_hnsw(
                rustic_core::vector::Quantization::Int8,
            )
            .with_osq(true)
            .with_confidence_interval(Some(1.0))
            .with_hnsw(16, 100)
        } else {
            rustic_core::vector::store::VectorFieldOptions::f32_hnsw().with_hnsw(16, 100)
        });
        fields.push(f);
    }
    fields
}
fn parse_query(v: &Value) -> Result<Query, String> {
    let o = v.as_object().ok_or("query must be an object")?;
    if o.len() != 1 {
        return Err("query must have one clause".into());
    }
    let (kind, def) = o.iter().next().unwrap();
    match kind.as_str() {
        "match_all" => Ok(Query::MatchAll),
        "term" => {
            let d = def.as_object().ok_or("bad term")?;
            if d.len() != 1 {
                return Err("one term field required".into());
            }
            let (name, val) = d.iter().next().unwrap();
            let text = val
                .as_str()
                .or_else(|| val.get("value").and_then(Value::as_str))
                .ok_or("string term required")?;
            Ok(Query::term(field(name)?, text))
        }
        "bool" => {
            let obj = def.as_object().ok_or("bool must be an object")?;
            if obj
                .keys()
                .any(|k| !["must", "should", "must_not", "filter"].contains(&k.as_str()))
            {
                return Err("unsupported bool option".into());
            }
            let clauses = |name: &str| -> Result<Vec<Query>, String> {
                match def.get(name) {
                    None => Ok(vec![]),
                    Some(Value::Array(a)) => a.iter().map(parse_query).collect(),
                    Some(q) => Ok(vec![parse_query(q)?]),
                }
            };
            if def.get("minimum_should_match").is_some() {
                return Err("minimum_should_match unsupported".into());
            }
            Ok(Query::Bool {
                must: clauses("must")?,
                should: clauses("should")?,
                must_not: clauses("must_not")?,
                filter: clauses("filter")?,
                minimum_should_match: Default::default(),
            })
        }
        "match_phrase" => {
            let d = def.as_object().ok_or("bad phrase")?;
            if d.len() != 1 {
                return Err("one phrase field required".into());
            }
            let (name, val) = d.iter().next().ok_or("empty phrase")?;
            let text = val.as_str().ok_or("string phrase required")?;
            let id = field(name)?;
            Ok(Query::Phrase {
                terms: text
                    .split_whitespace()
                    .map(|t| Term::new_text(id, t))
                    .collect(),
                slop: 0,
            })
        }
        "range" => {
            let d = def.as_object().ok_or("bad range")?;
            if d.len() != 1 {
                return Err("one range field required".into());
            }
            let (name, b) = d.iter().next().ok_or("empty range")?;
            if b.as_object()
                .ok_or("range bounds must be an object")?
                .keys()
                .any(|k| !["gte", "lte"].contains(&k.as_str()))
            {
                return Err("only inclusive gte/lte bounds supported".into());
            }
            if name != "price" {
                return Err("only price ranges supported".into());
            }
            Ok(Query::RangeF64 {
                field: field(name)?,
                lower: Bound::Included(b.get("gte").and_then(Value::as_f64).ok_or("gte required")?),
                upper: Bound::Included(b.get("lte").and_then(Value::as_f64).ok_or("lte required")?),
            })
        }
        _ => Err(format!("unsupported query {kind}")),
    }
}
fn request_query(req: &Value) -> Result<Query, String> {
    if let Some(knn) = req.get("knn") {
        if req.get("query").is_some() {
            return Err("hybrid unsupported".into());
        }
        let o = knn.as_object().ok_or("knn object required")?;
        if o.keys().any(|k| {
            ![
                "field",
                "query_vector",
                "k",
                "num_candidates",
                "filter",
                "rescore_vector",
            ]
            .contains(&k.as_str())
        }) || knn["field"] != "v"
        {
            return Err("unsupported knn option".into());
        }
        let vector: Vec<f32> = knn["query_vector"]
            .as_array()
            .ok_or("vector required")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .filter(|n| n.is_finite())
                    .map(|n| n as f32)
                    .ok_or("invalid vector")
            })
            .collect::<Result<_, _>>()?;
        if vector.len() != 128 {
            return Err("128 dimensions required".into());
        }
        let k = knn["k"].as_u64().ok_or("k required")? as usize;
        let nc = knn["num_candidates"]
            .as_u64()
            .ok_or("num_candidates required")? as usize;
        if k == 0 || nc < k || nc > 10000 {
            return Err("invalid candidate count".into());
        }
        Ok(Query::Knn {
            field: 7,
            vector,
            k,
            num_candidates: nc,
            filter: knn
                .get("filter")
                .map(parse_query)
                .transpose()?
                .map(Box::new),
            similarity_threshold: None,
            boost: 1.0,
            oversample: knn
                .get("rescore_vector")
                .and_then(|r| r.get("oversample"))
                .and_then(Value::as_f64)
                .map(|n| n as f32),
            visit_percentage: knn.get("visit_percentage").and_then(Value::as_f64).map(|n| n as f32),
        })
    } else {
        parse_query(req.get("query").unwrap_or(&json!({"match_all": {}})))
    }
}
fn index_info(index: &Index) -> Value {
    json!({"documents":index.num_docs(),"document_segments":index.num_segments(),"vectors":index.schema.iter().filter(|f|matches!(f.field_type,FieldType::Vector{..})).map(|f|json!({"field":f.name,"field_id":f.id,"segments":index.vector_store().segments(f.id).iter().map(|s|json!({"id":s.id,"vectors":s.len(),"sealed":s.is_sealed(),"graph":s.has_graph(),"flat":s.is_flat()})).collect::<Vec<_>>()})).collect::<Vec<_>>()})
}
struct App {
    index: RwLock<Index>,
    inline: bool,
    in_place: bool,
}
fn request_sort(req: &Value) -> Result<Vec<SortField>, String> {
    let Some(values) = req.get("sort") else {
        return Ok(Vec::new());
    };
    values
        .as_array()
        .ok_or("sort array required")?
        .iter()
        .map(|value| {
            let map = value.as_object().ok_or("one-field sort object required")?;
            if map.len() != 1 {
                return Err("one-field sort object required".into());
            }
            let (name, direction) = map.iter().next().unwrap();
            let kind = match name.as_str() {
                "price" => SortKind::F64,
                "ts" => SortKind::I64,
                _ => return Err("POC numeric sorts support price and ts".into()),
            };
            let sort = SortField::field(field(name)?, kind);
            match direction.as_str() {
                Some("asc") => Ok(sort),
                Some("desc") => Ok(sort.desc()),
                _ => Err("asc/desc string required".into()),
            }
        })
        .collect()
}
fn execute(app: &App, mode: &str, bytes: &[u8]) -> Result<String, String> {
    if ![
        "baseline",
        "naive",
        "candidate",
        "core",
        "mapped",
        "mapped-candidate",
        "mapped-topk",
        "mapped-collect-topk",
        "native-kernel",
        "native-index",
    ]
    .contains(&mode)
    {
        return Err("unknown mode".into());
    }
    let start = Instant::now();
    let req: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    for key in req.as_object().ok_or("object required")?.keys() {
        if ![
            "query",
            "knn",
            "size",
            "_source",
            "track_total_hits",
            "request_cache",
            "sort",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unsupported request option {key}"));
        }
    }
    let source = match req.get("_source") {
        None | Some(Value::Bool(true)) => true,
        Some(Value::Bool(false)) => false,
        _ => return Err("boolean _source required".into()),
    };
    let size = req.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
    if size > 10_000 || req.get("size").is_some_and(|s| s.as_u64().is_none()) {
        return Err("size must be an integer from 0 to 10000".into());
    }
    let q = request_query(&req)?;
    let sort = request_sort(&req)?;
    if !sort.is_empty()
        && (req.get("knn").is_some() || mode.starts_with("mapped") || mode.starts_with("native-"))
    {
        return Err(
            "numeric sorts require a text query in a baseline, naive, candidate or core mode"
                .into(),
        );
    }
    let parse_us = start.elapsed().as_secs_f64() * 1e6;
    let index = app.index.read().unwrap();
    let leaves: Vec<LeafReader<'_>> = index.leaves();
    let tracking = match req.get("track_total_hits") {
        Some(Value::Bool(true)) => TrackTotalHits::True,
        Some(Value::Bool(false)) => TrackTotalHits::False,
        Some(Value::Number(n)) => TrackTotalHits::UpTo(n.as_u64().ok_or("bad hit threshold")?),
        None => TrackTotalHits::UpTo(10_000),
        _ => return Err("bad hit tracking".into()),
    };
    let query_start = Instant::now();
    let options = SearchOptions {
        top_k: size,
        track_total_hits: tracking,
        sort,
        ..Default::default()
    };
    let (td, knn_phases) = if mode == "native-index" {
        let Query::Knn {
            field,
            vector,
            k,
            num_candidates,
            filter,
            boost,
            similarity_threshold,
            oversample,
            visit_percentage,
        } = &q
        else {
            return Err("native index requires KNN".into());
        };
        let request = rustic_core::index::KnnRequest {
            field: *field,
            vector: vector.clone(),
            k: *k,
            num_candidates: *num_candidates,
            filter: filter.as_deref().cloned(),
            boost: *boost,
            similarity_threshold: *similarity_threshold,
        };
        let config = rustic_core::vector::store::PlannerConfig {
            oversample: *oversample,
            visit_ratio: visit_percentage.map(|p| p / 100.0),
            ..Default::default()
        };
        let (hits, stats) = index.knn_search_with(&request, &config).unwrap();
        let hits = hits
            .into_iter()
            .map(|hit| rustic_core::query::TopHit::new(hit.leaf, hit.doc, hit.score, vec![]))
            .collect::<Vec<_>>();
        (
            rustic_core::query::TopDocs {
                total_hits: hits.len() as u64,
                max_score: hits.first().map_or(0.0, |h| h.score),
                hits,
                ..Default::default()
            },
            json!({"vector_ops":stats.vector_ops,"exact_segments":stats.exact_segments,"graph_segments":stats.graph_segments}),
        )
    } else if mode == "native-kernel" {
        let Query::Knn {
            field,
            vector,
            k,
            num_candidates,
            filter: None,
            boost: 1.0,
            oversample: None,
            similarity_threshold: None,
            visit_percentage: _,
        } = &q
        else {
            return Err("native kernel requires unfiltered, unboosted float KNN".into());
        };
        let segments = index.vector_store().segments(*field);
        if segments.len() != 1
            || !segments[0].has_graph()
            || index.vector_field_options(*field).quantization
                != rustic_core::vector::Quantization::None
        {
            return Err("native kernel requires one float HNSW graph".into());
        }
        let segment = &segments[0];
        // Controlled immutable fixture only: no hidden points or deletions.
        if segment.len() as u64 != index.num_docs() as u64 {
            return Err("native kernel requires one vector per live document".into());
        }
        let hits = segment
            .search_graph(vector, *k, *num_candidates, |_| true)
            .into_iter()
            .map(|(ordinal, score)| {
                let (segment_id, doc) = index
                    .vector_store()
                    .owner(segment.points()[ordinal as usize])
                    .unwrap();
                let leaf = leaves
                    .iter()
                    .position(|leaf| matches!(leaf, LeafReader::Segment(s) if s.id == segment_id))
                    .unwrap();
                rustic_core::query::TopHit::new(leaf, doc, score, vec![])
            })
            .collect::<Vec<_>>();
        (
            rustic_core::query::TopDocs {
                total_hits: hits.len() as u64,
                max_score: hits.first().map_or(0.0, |h| h.score),
                hits,
                ..Default::default()
            },
            json!({}),
        )
    } else if mode == "mapped-collect-topk" {
        knn_path::search_collect_topk(&leaves, &q, &options)
    } else if mode.starts_with("mapped") {
        knn_path::search(
            &leaves,
            &q,
            &options,
            mode == "mapped-topk"
                && match &q {
                    Query::Knn { field, boost, .. } => {
                        *boost == 1.0
                            && index.vector_field_options(*field).quantization
                                == rustic_core::vector::Quantization::None
                    }
                    _ => false,
                },
        )
    } else {
        (
            Searcher::new(leaves.clone()).search(&q, &options),
            json!({}),
        )
    };
    let query_us = query_start.elapsed().as_secs_f64() * 1e6;
    let fetch_start = Instant::now();
    let mut hits = Vec::new();
    if mode != "core" {
        for h in &td.hits {
            let leaf = &leaves[h.leaf];
            // Baseline mirrors search::fetch_hit: always get_doc, even _source:false.
            // Candidate is valid only for this restricted schema (no routing/ignored metadata).
            let stored = if (mode == "candidate"
                || mode == "naive"
                || mode == "mapped-candidate"
                || mode == "mapped-topk"
                || mode == "mapped-collect-topk"
                || mode == "native-kernel"
                || mode == "native-index")
                && !source
            {
                None
            } else {
                Some(leaf.get_doc(h.doc).ok_or("missing stored doc")?)
            };
            let id = match &stored {
                Some(d) => d
                    .get(ID)
                    .and_then(|v| v.as_text())
                    .ok_or("missing id")?
                    .to_string(),
                None => leaf
                    .keyword_value(ID, h.doc)
                    .ok_or("missing id doc value")?,
            };
            let raw = stored
                .as_ref()
                .and_then(|d| d.get(SOURCE))
                .and_then(|v| v.as_text())
                .unwrap_or("");
            // Same hit string allocation and raw _source append as ES fetch_hit.
            let mut hit = String::with_capacity(raw.len() + 128);
            let score = if options.sort.is_empty() {
                h.score.to_string()
            } else {
                "null".into()
            };
            hit.push_str(&format!(
                "{{\"_index\":\"path-poc\",\"_id\":{},\"_score\":{score}",
                serde_json::to_string(&id).unwrap()
            ));
            if !options.sort.is_empty() {
                let values: Vec<Value> = h
                    .sort
                    .iter()
                    .map(|v| match v {
                        SortValue::I64(v) => json!(v),
                        SortValue::F64(v) => json!(v),
                        SortValue::Str(v) => json!(v),
                        SortValue::Missing => Value::Null,
                    })
                    .collect();
                hit.push_str(&format!(
                    ",\"sort\":{}",
                    serde_json::to_string(&values).unwrap()
                ));
            }
            if source {
                hit.push_str(",\"_source\":");
                hit.push_str(raw);
            }
            hit.push('}');
            hits.push(hit);
        }
    }
    let fetch_us = fetch_start.elapsed().as_secs_f64() * 1e6;
    let max_score = if options.sort.is_empty() {
        json!(td.max_score)
    } else {
        Value::Null
    };
    let envelope = json!({"took": start.elapsed().as_millis(), "timed_out": false, "_shards": {"total":1,"successful":1,"skipped":0,"failed":0}, "hits": {"total":{"value":td.total_hits,"relation": if format!("{:?}", td.total_hits_relation) == "Gte" {"gte"} else {"eq"}}, "max_score": max_score, "hits": []}, "_poc": {"parse_us":parse_us,"query_us":query_us,"fetch_us":fetch_us,"mode":mode,"knn_phases":knn_phases}}).to_string();
    // Production assembles raw hit strings; do not parse and serialize _source again.
    Ok(envelope.replace("\"hits\":[]", &format!("\"hits\":[{}]", hits.join(","))))
}
async fn search(
    State(app): State<Arc<App>>,
    Path(mode): Path<String>,
    bytes: Bytes,
) -> Result<([(axum::http::HeaderName, &'static str); 1], String), (StatusCode, String)> {
    let result = if app.in_place {
        tokio::task::block_in_place(|| execute(&app, &mode, &bytes))
    } else if app.inline {
        execute(&app, &mode, &bytes)
    } else {
        tokio::task::spawn_blocking(move || execute(&app, &mode, &bytes))
            .await
            .unwrap()
    };
    result
        .map(|s| ([(axum::http::header::CONTENT_TYPE, "application/json")], s))
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}
async fn msearch(
    State(app): State<Arc<App>>,
    Path(mode): Path<String>,
    bytes: Bytes,
) -> Result<([(axum::http::HeaderName, &'static str); 1], String), (StatusCode, String)> {
    tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() % 2 != 0 || !text.ends_with('\n') {
            return Err("msearch requires header/body pairs and a final newline".into());
        }
        let mut responses = Vec::with_capacity(lines.len() / 2);
        for pair in lines.chunks_exact(2) {
            let header: Value = serde_json::from_str(pair[0]).map_err(|e| e.to_string())?;
            if header
                .as_object()
                .is_none_or(|o| o.keys().any(|k| k != "request_cache"))
            {
                return Err("only request_cache is supported in msearch headers".into());
            }
            let mut response = execute(&app, &mode, pair[1].as_bytes())?;
            response.pop();
            response.push_str(",\"status\":200}");
            responses.push(response);
        }
        Ok::<_, String>(format!(
            "{{\"took\":{},\"responses\":[{}]}}",
            start.elapsed().as_millis(),
            responses.join(",")
        ))
    })
    .await
    .unwrap()
    .map(|s| ([(axum::http::header::CONTENT_TYPE, "application/json")], s))
    .map_err(|e| (StatusCode::BAD_REQUEST, e))
}
fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let mut builder =
        if mode == "bench-http" || mode == "bench-load" || mode == "serve-inline-single" {
            tokio::runtime::Builder::new_current_thread()
        } else {
            tokio::runtime::Builder::new_multi_thread()
        };
    builder.enable_all().build().unwrap().block_on(async_main());
}
async fn async_main() {
    // Same process-wide scoring initialization as engine::shard::Shard::new/open.
    rustic_core::query::bm25::set_legacy_k1_boost(true);
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("bench-load") => {
            load_bench::run(&args).await;
        }
        Some("bench-http") => {
            http_bench::run(&args).await;
        }
        Some("bench") => {
            in_process::run(&args);
        }
        Some("requests") => {
            in_process::requests(&args);
        }
        Some("inspect") => {
            let index = Index::open(std::path::Path::new(&args[2])).unwrap();
            println!("{}", index_info(&index));
        }
        Some("inspect-vectors") => {
            let index = Index::open(std::path::Path::new(&args[2])).unwrap();
            let vectors: Vec<_> = index.schema.iter().filter(|f| matches!(f.field_type, FieldType::Vector { .. }))
                .flat_map(|f| index.vector_store().segments(f.id).into_iter().map(move |s| {
                    json!({"field":f.name,"segment":s.id,"vectors":s.len(),
                        "encoded_graph_bytes":s.graph().map(|g| g.encode().len())})
                })).collect();
            println!("{}", json!({"documents":index.num_docs(),"vectors":vectors}));
        }
        Some("seal") => {
            let mut index = Index::open(std::path::Path::new(&args[2])).unwrap();
            let report = index.force_merge_vectors(1, false).unwrap();
            index.force_merge().unwrap();
            index.commit().unwrap();
            println!("report={report:?}; {}", index_info(&index));
        }
        Some("truth") => {
            in_process::truth(&args);
        }
        Some("percolate-index") => {
            percolate_poc::index(&args);
        }
        Some("percolate-bench") => {
            percolate_poc::bench(&args);
        }
        Some("percolate-debug") => {
            percolate_poc::debug_misses(&args);
        }
        Some("nested-bench") => {
            nested_poc::run();
        }
        Some("nested-datehist-bench") => {
            nested_datehist_poc::run();
        }
        Some("store-bench") => {
            store_poc::run();
        }
        Some("http-write-bench") => {
            http_write_poc::run();
        }
        Some("http-overhead-bench") => {
            http_overhead_poc::run();
        }
        Some("write-path-bench") => {
            write_path_poc::run();
        }
        Some("fsync-bench") => {
            fsync_poc::run();
        }
        Some("sort-bench") => {
            sort_poc::run();
        }
        Some("index-bench") => {
            index_poc::run(&args[1..]);
        }
        Some("index") | Some("index-auto") => {
            let mut index = Index::create(
                std::path::Path::new(&args[3]),
                "path-poc",
                schema(args.get(4).map(String::as_str)),
                IndexSettings::default(),
            )
            .unwrap();
            if let Some(threads) = args.get(5) {
                index.optimizer_policy.hnsw.threads = threads
                    .parse()
                    .expect("integer graph build thread count required");
            }
            for line in BufReader::new(std::fs::File::open(&args[2]).unwrap()).lines() {
                let raw = line.unwrap();
                let v: Value = serde_json::from_str(&raw).unwrap();
                let mut d = Document::new();
                d.add_keyword(ID, &v["id"].to_string());
                d.add_text(1, v["title"].as_str().unwrap());
                d.add_text(2, v["body"].as_str().unwrap());
                d.add_f64(3, v["price"].as_f64().unwrap());
                d.add_i64(4, v["ts"].as_i64().unwrap());
                d.add_keyword(5, v["tag"].as_str().unwrap());
                if let Some(v) = v.get("v") {
                    d.add_vector(
                        7,
                        v.as_array()
                            .unwrap()
                            .iter()
                            .map(|n| n.as_f64().unwrap() as f32)
                            .collect(),
                    );
                }
                d.add_keyword(SOURCE, &raw);
                index.add_document(&d).unwrap();
            }
            index.refresh().unwrap();
            index.commit().unwrap();
            // ES Shard::force_merge seals/merges vectors separately from documents.
            // A document merge alone leaves appendable vectors on the exact-scan path.
            if args[1] == "index-auto" {
                // Same normal optimizer job used by Node::optimize_vectors_all;
                // leave document segments intact and obey default thresholds.
                let report = index.vector_optimizer().run().unwrap();
                eprintln!("automatic optimizer: {report:?}");
            } else {
                index.force_merge_vectors(1, false).unwrap();
                index.force_merge().unwrap();
            }
            index.commit().unwrap();
            println!("{}", index_info(&index));
        }
        Some("serve")
        | Some("serve-inline")
        | Some("serve-inline-single")
        | Some("serve-in-place") => {
            let index = Index::open(std::path::Path::new(&args[2])).unwrap();
            let app = Arc::new(App {
                index: RwLock::new(index),
                inline: args[1] != "serve",
                in_place: args[1] == "serve-in-place",
            });
            let router = Router::new()
                .route("/{mode}/_search", post(search))
                .route("/{mode}/_msearch", post(msearch))
                .with_state(app);
            let listener = tokio::net::TcpListener::bind(
                args.get(3).map(String::as_str).unwrap_or("127.0.0.1:19212"),
            )
            .await
            .unwrap();
            axum::serve(listener, router).await.unwrap();
        }
        _ => panic!("usage: es-path-poc index CORPUS INDEX [vectors|int8] | serve INDEX [ADDR]"),
    }
}
