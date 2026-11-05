//! HTTP tests for search extras: highlighting (unified / fvh / plain),
//! suggesters (term / phrase / completion), term vectors, and the smaller
//! search features (cross_fields blending, geo decay, collapse inner_hits,
//! `_explain` descriptions).

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

async fn start(dir: &std::path::Path) -> (String, Client) {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    (format!("http://{addr}"), Client::new())
}

async fn call(c: &Client, m: Method, url: String, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(m, url);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, serde_json::from_str(&t).unwrap_or(Value::String(t)))
}

/// Create `index` with `body` and index `docs` (id = position) with refresh.
#[allow(dead_code)]
async fn setup(c: &Client, base: &str, index: &str, body: Value, docs: &[Value]) {
    let (st, v) = call(c, Method::PUT, format!("{base}/{index}"), Some(body)).await;
    assert_eq!(st, 200, "{v}");
    for (i, d) in docs.iter().enumerate() {
        let (st, v) = call(c, Method::PUT, format!("{base}/{index}/_doc/{}?refresh=true", i + 1), Some(d.clone())).await;
        assert!(st == 200 || st == 201, "{v}");
    }
}

// ---- tests (each feature area appends its own #[tokio::test] fns below) ----

// ---- term vectors ----

#[tokio::test]
async fn termvectors_stored_artificial_and_multi() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    setup(
        &c,
        &base,
        "tv",
        json!({"mappings": {"properties": {"text": {"type": "text", "term_vector": "with_positions_offsets"}, "kw": {"type": "keyword"}}}}),
        &[json!({"text": "The quick brown fox is brown.", "kw": "Foo"}), json!({"text": "brown dog"})],
    )
    .await;
    let (st, v) = call(&c, Method::GET, format!("{base}/tv/_termvectors/1?term_statistics=true"), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_id"], "1");
    assert_eq!(v["found"], true);
    let t = &v["term_vectors"]["text"];
    assert_eq!(t["field_statistics"], json!({"sum_doc_freq": 7, "doc_count": 2, "sum_ttf": 8}));
    assert_eq!(t["terms"]["brown"]["doc_freq"], 2);
    assert_eq!(t["terms"]["brown"]["ttf"], 3);
    assert_eq!(t["terms"]["brown"]["term_freq"], 2);
    assert_eq!(t["terms"]["brown"]["tokens"][1], json!({"position": 5, "start_offset": 23, "end_offset": 28}));
    // keyword has no stored term vectors: only returned when asked for
    assert!(v["term_vectors"].get("kw").is_none());
    let (_, v) = call(&c, Method::POST, format!("{base}/tv/_termvectors/1"), Some(json!({"fields": ["kw"], "positions": false}))).await;
    assert_eq!(v["term_vectors"]["kw"]["terms"]["Foo"]["tokens"][0], json!({"start_offset": 0, "end_offset": 3}));
    // artificial doc + per-field analyzer + filter
    let (st, v) = call(
        &c,
        Method::POST,
        format!("{base}/tv/_termvectors"),
        Some(json!({"doc": {"text": "brown brown brown cat"}, "per_field_analyzer": {"text": "whitespace"}, "filter": {"max_num_terms": 1}})),
    )
    .await;
    assert_eq!(st, 200, "{v}");
    assert!(v.get("_id").is_none());
    let terms = v["term_vectors"]["text"]["terms"].as_object().unwrap();
    assert_eq!(terms.len(), 1);
    assert!(terms["brown"]["score"].is_number());
    // missing doc
    let (_, v) = call(&c, Method::GET, format!("{base}/tv/_termvectors/9"), None).await;
    assert_eq!(v["found"], false);
    // mtermvectors with per-item errors
    let (st, v) = call(
        &c,
        Method::POST,
        format!("{base}/_mtermvectors?term_statistics=true"),
        Some(json!({"docs": [{"_index": "tv", "_id": "1"}, {"_index": "nope", "_id": "1"}]})),
    )
    .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["docs"][0]["term_vectors"]["text"]["terms"]["brown"]["ttf"], 3);
    assert_eq!(v["docs"][1]["error"]["type"], "index_not_found_exception");
    let (_, v) = call(&c, Method::POST, format!("{base}/tv/_mtermvectors"), Some(json!({"ids": ["1", "2"]}))).await;
    assert_eq!(v["docs"][1]["term_vectors"]["text"]["terms"]["dog"]["term_freq"], 1);
    let (st, _) = call(&c, Method::POST, format!("{base}/_mtermvectors"), Some(json!({"docs": [{"_index": "tv", "_id": "1", "versionType": "external"}]}))).await;
    assert_eq!(st, 400);
}

#[tokio::test]
async fn collapse_fields_inner_hits_and_validation() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let docs = [
        json!({"g": 1, "s": 10}),
        json!({"g": 1, "s": 6}),
        json!({"g": 1, "s": 24}),
        json!({"g": 25, "s": 10}),
        json!({"g": 25, "s": 5}),
        json!({"g": 3, "s": 36}),
    ];
    setup(&c, &base, "col", json!({"mappings": {"properties": {"g": {"type": "integer"}, "s": {"type": "integer"}}}}), &docs).await;
    let body = json!({"collapse": {"field": "g", "max_concurrent_group_searches": 4, "inner_hits": [{"name": "asc", "size": 2, "sort": [{"s": "asc"}]}, {"name": "top", "size": 1}]}, "sort": [{"s": "desc"}]});
    let (st, v) = call(&c, Method::POST, format!("{base}/col/_search?rest_total_hits_as_int=true"), Some(body)).await;
    assert_eq!(st, 200, "{v}");
    let hits = v["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0]["fields"]["g"], json!([3]));
    assert_eq!(hits[1]["fields"]["g"], json!([1]));
    assert_eq!(hits[1]["inner_hits"]["asc"]["hits"]["total"], 3);
    assert_eq!(hits[1]["inner_hits"]["asc"]["hits"]["hits"][0]["_id"], "2");
    assert_eq!(hits[1]["inner_hits"]["asc"]["hits"]["hits"][1]["_id"], "1");
    assert_eq!(hits[2]["inner_hits"]["top"]["hits"]["hits"].as_array().unwrap().len(), 1);
    // missing inner_hits name
    let (st, _) = call(&c, Method::POST, format!("{base}/col/_search"), Some(json!({"collapse": {"field": "g", "inner_hits": {"size": 1}}}))).await;
    assert_eq!(st, 400);
    // scroll + collapse
    let (st, v) = call(&c, Method::POST, format!("{base}/col/_search?scroll=1m"), Some(json!({"collapse": {"field": "g"}}))).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("cannot use `collapse` in a scroll context"), "{v}");
    // search_after must sort on the collapse field
    let (st, v) = call(&c, Method::POST, format!("{base}/col/_search"), Some(json!({"collapse": {"field": "g"}, "search_after": [6], "sort": [{"s": "desc"}]}))).await;
    assert_eq!(st, 400, "{v}");
    // nested collapse inside inner_hits must not have inner_hits
    let (st, v) = call(&c, Method::POST, format!("{base}/col/_search"), Some(json!({"collapse": {"field": "g", "inner_hits": {"name": "x", "collapse": {"field": "s", "inner_hits": {}}}}}))).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "parse_exception", "{v}");
}

#[tokio::test]
async fn cross_fields_blends_document_frequencies() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    // "smith" is common in last_name but rare in first_name: with blended
    // IDF the first_name match must not outrank the last_name match.
    let mut docs = vec![json!({"first_name": "smith", "last_name": "jones"})];
    for _ in 0..5 {
        docs.push(json!({"first_name": "john", "last_name": "smith"}));
    }
    setup(&c, &base, "people", json!({"mappings": {"properties": {"first_name": {"type": "text"}, "last_name": {"type": "text"}}}}), &docs).await;
    let q = json!({"query": {"multi_match": {"query": "smith", "type": "cross_fields", "fields": ["first_name", "last_name"]}}});
    let (st, v) = call(&c, Method::POST, format!("{base}/people/_search"), Some(q)).await;
    assert_eq!(st, 200, "{v}");
    let hits = v["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 6);
    let s_first = hits.iter().find(|h| h["_id"] == "1").unwrap()["_score"].as_f64().unwrap();
    let s_last = hits.iter().find(|h| h["_id"] == "2").unwrap()["_score"].as_f64().unwrap();
    // Blended df: max(1, 5) on both fields, +1 bias on the rarer field
    // (Lucene BlendedTermQuery), so the rare first_name match scores lower
    // instead of ~6x higher (unblended idf of df=1 vs df=5).
    assert!(s_first > 0.0 && s_first < s_last, "{s_first} vs {s_last}");
    // operator and: every term must match in some field
    let q = json!({"query": {"multi_match": {"query": "john smith", "type": "cross_fields", "operator": "and", "fields": ["first_name", "last_name"]}}});
    let (_, v) = call(&c, Method::POST, format!("{base}/people/_search"), Some(q)).await;
    assert_eq!(v["hits"]["total"]["value"], 5);
    let q = json!({"query": {"multi_match": {"query": "smith", "type": "cross_fields", "fuzziness": 1, "fields": ["first_name"]}}});
    let (st, _) = call(&c, Method::POST, format!("{base}/people/_search"), Some(q)).await;
    assert_eq!(st, 400);
}

#[tokio::test]
async fn function_score_geo_decay() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let docs = [json!({"loc": {"lat": 0.0, "lon": 0.0}}), json!({"loc": {"lat": 0.0, "lon": 0.09}}), json!({"name": "nowhere"})];
    setup(&c, &base, "geo", json!({"mappings": {"properties": {"loc": {"type": "geo_point"}, "name": {"type": "keyword"}}}}), &docs).await;
    // doc 2 is ~10km away: gauss with scale 10km, decay 0.5 -> ~0.5
    let q = json!({"query": {"function_score": {"query": {"match_all": {}}, "gauss": {"loc": {"origin": "0,0", "scale": "10km", "decay": 0.5}}, "boost_mode": "replace"}}});
    let (st, v) = call(&c, Method::POST, format!("{base}/geo/_search"), Some(q)).await;
    assert_eq!(st, 200, "{v}");
    let score = |id: &str| v["hits"]["hits"].as_array().unwrap().iter().find(|h| h["_id"] == id).unwrap()["_score"].as_f64().unwrap();
    assert!((score("1") - 1.0).abs() < 1e-6);
    assert!((score("2") - 0.5).abs() < 0.01, "{}", score("2"));
    assert!((score("3") - 1.0).abs() < 1e-6, "missing point scores 1");
    // linear with offset: within offset -> 1
    let q = json!({"query": {"function_score": {"functions": [{"linear": {"loc": {"origin": {"lat": 0, "lon": 0}, "scale": "5km", "offset": "20km"}}, "weight": 2}], "boost_mode": "replace"}}});
    let (_, v) = call(&c, Method::POST, format!("{base}/geo/_search"), Some(q)).await;
    let s2 = v["hits"]["hits"].as_array().unwrap().iter().find(|h| h["_id"] == "2").unwrap()["_score"].as_f64().unwrap();
    assert!((s2 - 2.0).abs() < 1e-6, "{s2}");
}

#[tokio::test]
async fn explain_uses_lucene_wording() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    setup(&c, &base, "ex", json!({"mappings": {"properties": {"t": {"type": "text"}}}}), &[json!({"t": "quick fox"}), json!({"t": "lazy dog"})]).await;
    let (st, v) = call(&c, Method::POST, format!("{base}/ex/_explain/1"), Some(json!({"query": {"match": {"t": {"query": "fox", "boost": 2}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let e = &v["explanation"];
    assert_eq!(e["description"], "weight(t:fox in 0) [PerFieldSimilarity], result of:", "{v}");
    let score = &e["details"][0];
    assert_eq!(score["description"], "score(freq=1.0), computed as boost * idf * tf from:");
    assert_eq!(score["details"][0]["description"], "boost");
    // Lucene folds BM25's (k1 + 1) = 2.2 into the boost: 2 * 2.2 = 4.4 (as ES 8.19 reports).
    assert_eq!(score["details"][0]["value"], 4.4);
    assert_eq!(score["details"][1]["description"], "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:");
    let (_, v) = call(&c, Method::POST, format!("{base}/ex/_explain/1"), Some(json!({"query": {"dis_max": {"queries": [{"match": {"t": "fox"}}, {"match": {"t": "quick"}}]}}}))).await;
    assert_eq!(v["explanation"]["description"], "max of:", "{v}");
}

#[tokio::test]
async fn term_and_phrase_suggesters() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    setup(
        &c,
        &base,
        "sg",
        json!({"mappings": {"properties": {"body": {"type": "text"}}}}),
        &[json!({"body": "Amsterdam meetup"}), json!({"body": "nobel prize"}), json!({"body": "nobel prize winners"}), json!({"body": "noble warriors"})],
    )
    .await;
    let (st, v) = call(&c, Method::POST, format!("{base}/sg/_search?typed_keys=true"), Some(json!({"size": 0, "suggest": {
        "text": "The Amsterdma meetpu",
        "t": {"term": {"field": "body"}}
    }}))).await;
    assert_eq!(st, 200, "{v}");
    let e = &v["suggest"]["term#t"];
    assert_eq!(e[0], json!({"text": "the", "offset": 0, "length": 3, "options": []}));
    assert_eq!(e[1]["text"], "amsterdma");
    assert_eq!(e[1]["offset"], 4);
    assert_eq!(e[1]["length"], 9);
    assert_eq!(e[1]["options"][0], json!({"text": "amsterdam", "score": 0.8888889, "freq": 1}));
    assert_eq!(e[2]["options"][0], json!({"text": "meetup", "score": 0.8333333, "freq": 1}));
    // suggest_mode missing: a term in the index gets nothing; always: neighbours.
    let (_, v) = call(&c, Method::POST, format!("{base}/sg/_search"), Some(json!({"suggest": {
        "a": {"text": "noble", "term": {"field": "body"}},
        "b": {"text": "noble", "term": {"field": "body", "suggest_mode": "always", "sort": "frequency"}}
    }}))).await;
    assert_eq!(v["suggest"]["a"][0]["options"], json!([]));
    assert_eq!(v["suggest"]["b"][0]["options"][0]["text"], "nobel", "{v}");
    assert_eq!(v["suggest"]["b"][0]["options"][0]["freq"], 2);

    let (st, v) = call(&c, Method::POST, format!("{base}/sg/_search"), Some(json!({"size": 0, "suggest": {
        "p": {"text": "nobl prize", "phrase": {"field": "body", "highlight": {"pre_tag": "<em>", "post_tag": "</em>"}}}
    }}))).await;
    assert_eq!(st, 200, "{v}");
    let p = &v["suggest"]["p"][0];
    assert_eq!(p["text"], "nobl prize");
    assert_eq!(p["offset"], 0);
    assert_eq!(p["length"], 10);
    assert_eq!(p["options"][0]["text"], "nobel prize", "{v}");
    assert_eq!(p["options"][0]["highlighted"], "<em>nobel</em> prize");
    assert_eq!(p["options"][1]["text"], "noble prize");
    assert!(p["options"][0]["score"].as_f64().unwrap() > p["options"][1]["score"].as_f64().unwrap());
    // collate with prune: "noble prize" matches no document as a phrase.
    let (_, v) = call(&c, Method::POST, format!("{base}/sg/_search"), Some(json!({"size": 0, "suggest": {
        "p": {"text": "nobl prize", "phrase": {"field": "body", "collate": {"query": {"source": {"match_phrase": {"body": "{{suggestion}}"}}}, "prune": true}}}
    }}))).await;
    let o = &v["suggest"]["p"][0]["options"];
    assert_eq!(o[0]["collate_match"], true, "{v}");
    assert_eq!(o[1]["collate_match"], false);
    let (_, v) = call(&c, Method::POST, format!("{base}/sg/_search"), Some(json!({"size": 0, "suggest": {
        "p": {"text": "nobl prize", "phrase": {"field": "body", "collate": {"query": {"source": {"match_phrase": {"body": "{{suggestion}}"}}}}}}
    }}))).await;
    assert_eq!(v["suggest"]["p"][0]["options"].as_array().unwrap().len(), 1, "{v}");
}

#[tokio::test]
async fn completion_contexts_regex_and_multi_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    let (st, v) = call(&c, Method::PUT, format!("{base}/cs"), Some(json!({"mappings": {"properties": {
        "color": {"type": "keyword"},
        "loc": {"type": "geo_point"},
        "s": {"type": "completion", "contexts": [
            {"name": "color", "type": "category", "path": "color"},
            {"name": "loc", "type": "geo", "precision": "5km", "path": "loc"}
        ], "fields": {"raw": {"type": "keyword"}}},
        "plain": {"type": "completion"}
    }}}))).await;
    assert_eq!(st, 200, "{v}");
    for (i, d) in [
        json!({"s": "Marriot in Amsterdam", "color": "red", "loc": {"lat": 52.22, "lon": 4.53}, "plain": "bar"}),
        json!({"s": "Marriot in Berlin", "color": "blue", "loc": {"lat": 53.31, "lon": 13.24}, "plain": "baz"}),
    ]
    .iter()
    .enumerate()
    {
        let (st, v) = call(&c, Method::PUT, format!("{base}/cs/_doc/{i}?refresh=true"), Some(d.clone())).await;
        assert!(st == 201 || st == 200, "{v}");
    }
    let (_, v) = call(&c, Method::POST, format!("{base}/cs/_search"), Some(json!({"suggest": {"r": {"prefix": "mar", "completion": {"field": "s", "contexts": {"loc": {"lat": 52.2263, "lon": 4.543}}}}}}))).await;
    let o = &v["suggest"]["r"][0]["options"];
    assert_eq!(o.as_array().map(Vec::len), Some(1), "{v}");
    assert_eq!(o[0]["text"], "Marriot in Amsterdam");
    let (_, v) = call(&c, Method::POST, format!("{base}/cs/_search"), Some(json!({"suggest": {"r": {"prefix": "mar", "completion": {"field": "s", "contexts": {"color": [{"context": "blue", "boost": 3}]}}}}}))).await;
    assert_eq!(v["suggest"]["r"][0]["options"][0]["text"], "Marriot in Berlin", "{v}");
    assert_eq!(v["suggest"]["r"][0]["options"][0]["_score"], 3.0);
    let (st, v) = call(&c, Method::POST, format!("{base}/cs/_search"), Some(json!({"suggest": {"r": {"prefix": "mar", "completion": {"field": "s"}}}}))).await;
    assert_eq!(st, 400);
    assert!(v.to_string().contains("Missing mandatory contexts in context query"), "{v}");
    let (st, v) = call(&c, Method::PUT, format!("{base}/cs/_doc/9"), Some(json!({"s": "no context"}))).await;
    assert_eq!(st, 400);
    assert!(v.to_string().contains("Contexts are mandatory in context enabled completion field [s]"), "{v}");
    // keyword sub-field of a completion field
    let (_, v) = call(&c, Method::POST, format!("{base}/cs/_search"), Some(json!({"query": {"term": {"s.raw": "Marriot in Berlin"}}}))).await;
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");
    // regex + fuzzy
    let (_, v) = call(&c, Method::POST, format!("{base}/cs/_search"), Some(json!({"suggest": {"r": {"regex": "b.z", "completion": {"field": "plain"}}}}))).await;
    assert_eq!(v["suggest"]["r"][0]["options"][0]["text"], "baz", "{v}");
    assert_eq!(v["suggest"]["r"][0]["options"].as_array().unwrap().len(), 1);
    let (_, v) = call(&c, Method::POST, format!("{base}/cs/_search"), Some(json!({"suggest": {"r": {"prefix": "bzz", "completion": {"field": "plain", "fuzzy": {"fuzziness": 1}}}}}))).await;
    assert_eq!(v["suggest"]["r"][0]["options"][0]["text"], "baz", "{v}");
}

// ---- highlighting ----

async fn hl(c: &Client, base: &str, index: &str, body: Value) -> (u16, Value) {
    call(c, Method::POST, format!("{base}/{index}/_search"), Some(body)).await
}

#[tokio::test]
async fn highlight_unified_plain_fvh() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    setup(
        &c,
        &base,
        "hl",
        json!({"settings": {"index.highlight.max_analyzed_offset": 200}, "mappings": {"properties": {
            "text": {"type": "text", "fields": {"tv": {"type": "text", "term_vector": "with_positions_offsets"}}},
            "multi": {"type": "text"},
            "long": {"type": "text"},
            "kw": {"type": "keyword", "ignore_above": 5}
        }}}),
        &[json!({
            "text": "The quick brown fox is brown.",
            "multi": ["a fox", "nothing here", "fox b"],
            "long": "First sentence here. The fox jumps over < things. Last one.",
            "kw": ["abc", "abcdefgh"]
        })],
    )
    .await;
    // unified (default): every term occurrence
    let (st, v) = hl(&c, &base, "hl", json!({"query": {"match": {"text": "brown"}}, "highlight": {"fields": {"text": {}}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["highlight"]["text"][0], "The quick <em>brown</em> fox is <em>brown</em>.");
    // weight matches: a phrase is one tag; plain tags its terms, only where the phrase matched
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"match_phrase": {"text": "quick brown"}}, "highlight": {"fields": {"text": {}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["text"][0], "The <em>quick brown</em> fox is brown.");
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"match_phrase": {"text": "quick brown"}}, "highlight": {"type": "plain", "fields": {"text": {}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["text"][0], "The <em>quick</em> <em>brown</em> fox is brown.");
    // sentence passages + html encoder
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"match": {"long": "fox"}}, "highlight": {"encoder": "html", "fields": {"long": {}}}})).await;
    // ES's sentence break iterator does not split at ". L" here; the passage runs to the end.
    assert_eq!(v["hits"]["hits"][0]["highlight"]["long"][0], "The <em>fox</em> jumps over &lt; things. Last one.");
    // number_of_fragments 0: whole values with matches
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"match": {"multi": "fox"}}, "highlight": {"fields": {"multi": {"number_of_fragments": 0}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["multi"], json!(["a <em>fox</em>", "<em>fox</em> b"]));
    // no_match_size extends to the next word boundary
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"match": {"text": "fox"}}, "highlight": {"fields": {"long": {"no_match_size": 8}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["long"][0], "First sentence");
    // keyword values over ignore_above are not highlighted
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"prefix": {"kw": "ab"}}, "highlight": {"fields": {"kw": {}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["kw"], json!(["<em>abc</em>"]));
    // fvh needs term vectors
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"match": {"text.tv": "fox"}}, "highlight": {"type": "fvh", "fields": {"text.tv": {}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["text.tv"][0], "The quick brown <em>fox</em> is brown.");
    let (st, v) = hl(&c, &base, "hl", json!({"query": {"match": {"text": "fox"}}, "highlight": {"type": "fvh", "fields": {"text": {}}}})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["root_cause"][0]["type"], "illegal_argument_exception");
    // highlight_query
    let (_, v) = hl(&c, &base, "hl", json!({"query": {"match_all": {}}, "highlight": {"fields": {"text": {"highlight_query": {"prefix": {"text": "qu"}}}}}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["text"][0], "The <em>quick</em> brown fox is brown.");
    // validation
    let (st, v) = hl(&c, &base, "hl", json!({"query": {"match_all": {}}, "highlight": {"number_of_fragments": -1, "fields": {"text": {}}}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["root_cause"][0]["type"], "x_content_parse_exception");
    assert_eq!(v["error"]["caused_by"]["reason"], "[number_of_fragments] must be between [0] and [10000]");
}

#[tokio::test]
async fn highlight_max_analyzed_offset() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, c) = start(tmp.path()).await;
    setup(
        &c,
        &base,
        "mo",
        json!({"settings": {"index.highlight.max_analyzed_offset": 30}, "mappings": {"properties": {"f": {"type": "text"}, "g": {"type": "text", "index_options": "offsets"}}}}),
        &[json!({"f": "The quick brown fox went to the forest and saw another fox.", "g": "The quick brown fox went to the forest and saw another fox."})],
    )
    .await;
    let (st, v) = hl(&c, &base, "mo", json!({"query": {"match": {"f": "fox"}}, "highlight": {"fields": {"f": {}}}})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["root_cause"][0]["type"], "illegal_argument_exception");
    let (_, v) = hl(&c, &base, "mo", json!({"query": {"match": {"f": "fox"}}, "highlight": {"fields": {"f": {}}, "max_analyzed_offset": 20}})).await;
    assert_eq!(v["hits"]["hits"][0]["highlight"]["f"][0], "The quick brown <em>fox</em> went to the forest and saw another fox.");
    // offsets in the postings: no re-analysis, no limit
    let (st, v) = hl(&c, &base, "mo", json!({"query": {"match": {"g": "fox"}}, "highlight": {"fields": {"g": {}}}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["highlight"]["g"][0], "The quick brown <em>fox</em> went to the forest and saw another <em>fox</em>.");
}
