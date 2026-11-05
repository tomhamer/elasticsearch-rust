//! Java JSON permits unpaired UTF-16 escapes in source documents.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unpaired_source_surrogates_replicate_and_remain_searchable() {
    let cl = start(&[]).await;
    let ports = cl.p;
    let (status, v) = cl.call(0, Method::PUT, "/surrogates", Some(json!({
        "settings":{"number_of_shards":2,"number_of_replicas":1},
        "mappings":{"properties":{"body":{"type":"text"}}}
    }))).await;
    assert_eq!(status, 200, "{v}");
    let raw = r#"{"body":"before \ud83d after \ude02 pair \ud83d\ude02"}"#;
    let bulk = format!("{{\"index\":{{\"_id\":\"one\"}}}}\n{raw}\n");
    let v: Value = cl.c.post(format!("http://127.0.0.1:{}/surrogates/_bulk?refresh=true",ports.http[0]))
        .header("content-type","application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["errors"],false,"{v}");
    let (_, health) = cl.call(0,Method::GET,"/_cluster/health/surrogates?wait_for_status=green&timeout=60s",None).await;
    assert_eq!(health["status"],"green","{health}");
    for node in 0..2 {
        let response=cl.c.post(format!("http://127.0.0.1:{}/surrogates/_search",ports.http[node]))
            .json(&json!({"_source":false,"script_fields":{"loaded":{"script":{"source":"params._source.body"}}}})).send().await.unwrap();
        let status=response.status();let text=response.text().await.unwrap();assert_eq!(status,200,"{text}");
        let parsed=elasticsearch_engine::source_json::Source::parse(&text).unwrap();
        let elasticsearch_engine::source_json::Source::Array(hits)=parsed.get("hits").unwrap().get("hits").unwrap()else{panic!("{text}")};
        let elasticsearch_engine::source_json::Source::Array(loaded)=hits[0].get("fields").unwrap().get("loaded").unwrap()else{panic!("{text}")};
        assert!(loaded[0].equivalent(elasticsearch_engine::source_json::Source::parse(raw).unwrap().get("body").unwrap()),"{text}");
    }
    for node in 0..2 {
        let query=r#"{"query":{"script":{"script":{"source":"(int)params.s.charAt(0)==55296","params":{"s":"\ud800"}}}}}"#;
        let body=format!("{{}}\n{query}\n{{}}\n{}\n",r#"{"size":1,"script_fields":{"raw":{"script":{"source":"params.s","params":{"s":"\udfff"}}}}}"#);
        let response=cl.c.post(format!("http://127.0.0.1:{}/surrogates/_msearch",ports.http[node])).header("content-type","application/x-ndjson").body(body).send().await.unwrap();
        let status=response.status();let text=response.text().await.unwrap();assert_eq!(status,200,"{text}");
        let parsed=elasticsearch_engine::source_json::Source::parse(&text).unwrap();
        let elasticsearch_engine::source_json::Source::Array(responses)=parsed.get("responses").unwrap()else{panic!("{text}")};
        assert_eq!(responses.len(),2,"{text}");
        assert_eq!(responses[0].get("hits").unwrap().get("total").unwrap().get("value").unwrap().text(),"1","{text}");
        assert!(text.contains(r#"\udfff"#),"{text}");assert!(!text.contains("unreadable response"),"{text}");
        let response=cl.c.post(format!("http://127.0.0.1:{}/surrogates/_count",ports.http[node])).header("content-type","application/json").body(query).send().await.unwrap();
        let status=response.status();let text=response.text().await.unwrap();assert_eq!(status,200,"{text}");assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["count"],1,"{text}");
    }
    for node in 0..2 {
        let response = cl.c.get(format!("http://127.0.0.1:{}/surrogates/_doc/one",ports.http[node])).send().await.unwrap();
        assert_eq!(response.status(),200);
        let text = response.text().await.unwrap();
        assert!(text.contains(raw),"{text}");
        let (status, v) = cl.call(node,Method::POST,"/surrogates/_search",Some(json!({"_source":false,"query":{"match":{"body":"before"}}}))).await;
        assert_eq!(status,200,"{v}");
        assert_eq!(v["hits"]["total"]["value"],1,"{v}");
    }

    for node in 0..2 {
        for path in ["/surrogates/_doc/one?_source_includes=body", "/surrogates/_source/one?_source_includes=body"] {
            let r=cl.c.get(format!("http://127.0.0.1:{}{path}",ports.http[node])).send().await.unwrap();
            assert_eq!(r.status(),200);
            assert!(r.text().await.unwrap().contains(raw));
        }
        let r=cl.c.post(format!("http://127.0.0.1:{}/surrogates/_search",ports.http[node])).json(&json!({"_source":["body"],"query":{"match_all":{}}})).send().await.unwrap();
        assert_eq!(r.status(),200);let text=r.text().await.unwrap();assert!(text.contains(raw),"{text}");
    }
    let r=cl.c.post(format!("http://127.0.0.1:{}/surrogates/_update/one?refresh=true",ports.http[0])).json(&json!({"doc":{"other":2},"_source":true})).send().await.unwrap();
    assert_eq!(r.status(),200);
    let text=r.text().await.unwrap();assert!(text.contains(r#""body":"before \ud83d after \ude02 pair \ud83d\ude02""#),"{text}");
    for node in 0..2 {
        let r=cl.c.get(format!("http://127.0.0.1:{}/surrogates/_doc/one",ports.http[node])).send().await.unwrap();
        let text=r.text().await.unwrap();assert!(text.contains(r#""other":2"#));assert!(text.contains(r#"\ud83d"#));
    }
    let (status,v)=cl.call(0,Method::POST,"/surrogates/_update/one",Some(json!({"doc":{"other":2}}))).await;
    assert_eq!(status,200,"{v}");assert_eq!(v["result"],"noop");
    let script="ctx._source.copy = ctx._source.body; ctx._source.len = ctx._source.body.length(); ctx._source.appended = ctx._source.body + ' suffix'; ctx._source.other = 3";
    let r=cl.c.post(format!("http://127.0.0.1:{}/surrogates/_update/one?refresh=true",ports.http[0])).json(&json!({"script":{"source":script},"_source":true})).send().await.unwrap();
    let status=r.status();let text=r.text().await.unwrap();assert_eq!(status,200,"{text}");assert!(text.contains(r#"\ud83d"#),"{text}");assert!(!text.contains('�'),"{text}");
    let (status,v)=cl.call(0,Method::POST,"/_reindex?refresh=true",Some(json!({"source":{"index":"surrogates"},"dest":{"index":"copied"}}))).await;
    assert_eq!(status,200,"{v}");assert_eq!(v["created"],1,"{v}");
    let r=cl.c.get(format!("http://127.0.0.1:{}/copied/_doc/one",ports.http[1])).send().await.unwrap();
    let text=r.text().await.unwrap();assert!(text.contains(r#"\ud83d"#),"{text}");assert!(!text.contains('�'),"{text}");

    let body=r#"{"doc":{"new":"\ud800"},"_source":true}"#;
    let r=cl.c.post(format!("http://127.0.0.1:{}/surrogates/_update/one?refresh=true",ports.http[0])).header("content-type","application/json").body(body).send().await.unwrap();
    let status=r.status();let text=r.text().await.unwrap();assert_eq!(status,200,"{text}");assert!(text.contains(r#"\ud800"#),"{text}");
    let bulk=r#"{"update":{"_index":"surrogates","_id":"one"}}
{"doc":{"new":"\udfff"},"_source":true}
"#;
    let r=cl.c.post(format!("http://127.0.0.1:{}/_bulk?refresh=true",ports.http[0])).header("content-type","application/x-ndjson").body(bulk).send().await.unwrap();
    let status=r.status();let text=r.text().await.unwrap();assert_eq!(status,200,"{text}");assert!(text.contains(r#""errors":false"#),"{text}");assert!(text.contains(r#"\udfff"#),"{text}");

    // The coordinator must retain raw script parameters while enforcing one
    // document budget across both primary shards (and their replicas).
    let (status, v) = cl.call(0, Method::PUT, "/rawlimits", Some(json!({
        "settings":{"number_of_shards":2,"number_of_replicas":1}
    }))).await;
    assert_eq!(status, 200, "{v}");
    for id in 0..12 {
        let (status,v)=cl.call(0,Method::PUT,&format!("/rawlimits/_doc/{id}"),Some(json!({"n":id}))).await;
        assert_eq!(status,201,"{v}");
    }
    cl.call(0,Method::POST,"/rawlimits/_refresh",None).await;
    let raw = r#"{"max_docs":3,"script":{"source":"ctx._source.marker = params.value","params":{"value":"\ud800"}}}"#;
    let response=cl.c.post(format!("http://127.0.0.1:{}/rawlimits/_update_by_query?refresh=true",ports.http[0]))
        .header("content-type","application/json").body(raw).send().await.unwrap();
    assert_eq!(response.status(),200);
    let v:Value=response.json().await.unwrap();
    assert_eq!(v["updated"],3,"{v}");
    assert_eq!(v["total"],3,"{v}");
    for node in 0..2 {
        let mut marked=0;
        for id in 0..12 {
            let response=cl.c.get(format!("http://127.0.0.1:{}/rawlimits/_doc/{id}",ports.http[node])).send().await.unwrap();
            assert_eq!(response.status(),200);
            let text=response.text().await.unwrap();
            assert!(!text.contains('�'),"{text}");
            if text.contains(r#""marker":"\ud800""#) {marked+=1;}
        }
        assert_eq!(marked,3,"node {node}: exactly three raw sources must be changed");
    }

    let stored = r#"{"script":{"lang":"painless","source":"ctx._source.stored = '\udfff'"}}"#;
    let response=cl.c.put(format!("http://127.0.0.1:{}/_scripts/rawunits",ports.http[0]))
        .header("content-type","application/json").body(stored).send().await.unwrap();
    let status=response.status();let text=response.text().await.unwrap();assert_eq!(status,200,"{text}");
    let (status,v)=cl.call(1,Method::POST,"/rawlimits/_update/11?refresh=true",Some(json!({"script":{"id":"rawunits"}}))).await;
    assert_eq!(status,200,"{v}");
    for node in 0..2 {
        let response=cl.c.get(format!("http://127.0.0.1:{}/rawlimits/_doc/11",ports.http[node])).send().await.unwrap();
        assert_eq!(response.status(),200);
        let text=response.text().await.unwrap();
        assert!(text.contains(r#""stored":"\udfff""#),"{text}");
        assert!(!text.contains('�'),"{text}");
    }

    for node in 0..2 {
        let response=cl.c.post(format!("http://127.0.0.1:{}/rawlimits/_search",ports.http[node]))
            .json(&json!({"size":12,"runtime_mappings":{"runtime_half":{"type":"keyword","script":{"source":"emit(params.s.substring(1,2))","params":{"s":"😀"}}}},"fields":["runtime_half"],"script_fields":{"half":{"script":{"source":"params.s.substring(0,1)","params":{"s":"😀"}}}}}))
            .send().await.unwrap();
        let status=response.status(); let raw=response.text().await.unwrap();
        assert_eq!(status,200,"{raw}");
        assert_eq!(raw.matches(r#""half":["\ud83d"]"#).count(),12,"{raw}");
        assert_eq!(raw.matches(r#""runtime_half":["�"]"#).count(),12,"{raw}");
    }

    for node in 0..2 {
        let body=r#"{"size":12,"runtime_mappings":{"raw_runtime":{"type":"keyword","script":{"source":"emit(params.s)","params":{"s":"\ud800"}}}},"fields":["raw_runtime"],"script_fields":{"param":{"script":{"source":"params.s","params":{"s":"\udfff"}}},"literal":{"script":{"source":"'\ud800'"}}}}"#;
        let response=cl.c.post(format!("http://127.0.0.1:{}/rawlimits/_search",ports.http[node]))
            .header("content-type","application/json").body(body).send().await.unwrap();
        let status=response.status();let raw=response.text().await.unwrap();
        assert_eq!(status,200,"{raw}");
        assert_eq!(raw.matches(r#""param":["\udfff"]"#).count(),12,"{raw}");
        assert_eq!(raw.matches(r#""literal":["\ud800"]"#).count(),12,"{raw}");
        assert_eq!(raw.matches(r#""raw_runtime":["�"]"#).count(),12,"{raw}");
    }

    // Numeric script sorts expose exact UTF16 units without a UTF8 sort-key
    // boundary, and exercise coordinator pagination and both shard owners.
    for node in 0..2 {
        for (script, expected) in [
            (r#"{"source":"(int)params.s.charAt(0)","params":{"s":"\ud800"}}"#,55296),
            (r#"{"source":"(int)'\udfff'.charAt(0)"}"#,57343),
        ] {
            let body=format!(r#"{{"size":12,"sort":[{{"_script":{{"type":"number","script":{script},"order":"asc"}}}}]}}"#);
            let response=cl.c.post(format!("http://127.0.0.1:{}/rawlimits/_search",ports.http[node]))
                .header("content-type","application/json").body(body).send().await.unwrap();
            let status=response.status();let text=response.text().await.unwrap();
            assert_eq!(status,200,"{text}");
            // Existing raw document sources are independently lossless.
            let parsed=elasticsearch_engine::source_json::Source::parse(&text).unwrap();
            let hits=parsed.get("hits").unwrap().get("hits").unwrap();
            let elasticsearch_engine::source_json::Source::Array(hits)=hits else {panic!("missing hits")};
            assert_eq!(hits.len(),12,"{text}");
            for hit in hits {
                let sort:Value=serde_json::from_str(&hit.get("sort").unwrap().text()).unwrap();
                assert_eq!(sort[0].as_f64(),Some(expected as f64),"{text}");
            }
        }
    }

    for node in 0..2 {
        for (body, aggregation) in [
            (r#"{"_source":false,"size":12,"query":{"script":{"script":{"source":"(int)params.s.charAt(0)==55296","params":{"s":"\ud800"}}}}}"#,false),
            (r#"{"_source":false,"size":12,"query":{"script_score":{"query":{"match_all":{}},"script":{"source":"(int)params.s.charAt(0)","params":{"s":"\ud800"}}}}}"#,false),
            (r#"{"_source":false,"size":12,"query":{"function_score":{"query":{"match_all":{}},"script_score":{"script":{"source":"(int)'\ud800'.charAt(0)"}}}}}"#,false),
            (r#"{"_source":false,"size":12,"aggs":{"units":{"max":{"script":{"source":"(int)params.s.charAt(0)","params":{"s":"\ud800"}}}}}}"#,true),
            (r#"{"_source":false,"size":12,"rescore":{"window_size":12,"query":{"query_weight":0,"rescore_query_weight":1,"rescore_query":{"script_score":{"query":{"match_all":{}},"script":{"source":"(int)params.s.charAt(0)","params":{"s":"\ud800"}}}}}}}"#,false),
        ] {
            let response=cl.c.post(format!("http://127.0.0.1:{}/rawlimits/_search",ports.http[node]))
                .header("content-type","application/json").body(body).send().await.unwrap();
            let status=response.status();let text=response.text().await.unwrap();
            assert_eq!(status,200,"{body}: {text}");
            let parsed:Value=serde_json::from_str(&text).unwrap();
            assert_eq!(parsed["hits"]["hits"].as_array().unwrap().len(),12,"{text}");
            if aggregation {assert_eq!(parsed["aggregations"]["units"]["value"].as_f64(),Some(55296.0),"{text}");}
        }
    }

    for node in 0..2 {
        for (body,path) in [
            (r#"{"size":0,"aggs":{"units":{"scripted_metric":{"params":{"s":"\ud800"},"init_script":"state.n=0","map_script":"state.n += 1","combine_script":"state.n","reduce_script":"(int)params.s.charAt(0)"}}}}"#, "/aggregations/units/value"),
            (r#"{"size":0,"aggs":{"by_n":{"histogram":{"field":"n","interval":20},"aggs":{"units":{"bucket_script":{"buckets_path":{"n":"_count"},"script":{"source":"(int)params.s.charAt(0)","params":{"s":"\ud800"}}}}}}}}"#, "/aggregations/by_n/buckets/0/units/value"),
        ] {
            let response=cl.c.post(format!("http://127.0.0.1:{}/rawlimits/_search",ports.http[node])).header("content-type","application/json").body(body).send().await.unwrap();
            let status=response.status();let text=response.text().await.unwrap();assert_eq!(status,200,"{text}");let parsed:Value=serde_json::from_str(&text).unwrap();assert_eq!(parsed.pointer(path).and_then(Value::as_f64),Some(55296.0),"{text}");
        }
    }

    for node in 0..2 {
        let body=r#"{"size":0,"aggs":{"raw":{"scripted_metric":{"params":{"s":"\ud800"},"init_script":"state.s=params.s","map_script":"state.n=1","combine_script":"state.s","reduce_script":"states[0]"}}}}"#;
        let response=cl.c.post(format!("http://127.0.0.1:{}/rawlimits/_search",ports.http[node])).header("content-type","application/json").body(body).send().await.unwrap();
        let status=response.status();let text=response.text().await.unwrap();assert_eq!(status,200,"{text}");
        let parsed=elasticsearch_engine::source_json::Source::parse(&text).unwrap();
        let value=parsed.get("aggregations").unwrap().get("raw").unwrap().get("value").unwrap();
        assert!(value.equivalent(&elasticsearch_engine::source_json::Source::parse(r#""\ud800""#).unwrap()),"{text}");
        assert!(!text.contains("utf16_metric_value"),"{text}");
    }

    let (status,value)=cl.call(0,Method::PUT,"/_ingest/pipeline/lossless-ingest",Some(json!({"processors":[{"script":{"source":"ctx.generated=(char)55296; ctx.copy=ctx.body"}},{"rename":{"field":"copy","target_field":"renamed"}}]}))).await;
    assert_eq!(status,200,"{value}");
    for node in 0..2 {
        let raw=r#"{"body":"a\udfff"}"#;
        let response=cl.c.put(format!("http://127.0.0.1:{}/surrogates/_doc/ingest-{node}?pipeline=lossless-ingest&refresh=true",ports.http[node])).header("content-type","application/json").body(raw).send().await.unwrap();
        let status=response.status();let body=response.text().await.unwrap();assert_eq!(status,201,"{body}");
        for peer in 0..2 {
            let response=cl.c.get(format!("http://127.0.0.1:{}/surrogates/_doc/ingest-{node}",ports.http[peer])).send().await.unwrap();
            let body=response.text().await.unwrap();let parsed=elasticsearch_engine::source_json::Source::parse(&body).unwrap();
            let source=parsed.get("_source").unwrap();
            assert!(source.get("generated").unwrap().equivalent(&elasticsearch_engine::source_json::Source::parse(r#""\ud800""#).unwrap()),"{body}");
            assert!(source.get("renamed").unwrap().equivalent(&elasticsearch_engine::source_json::Source::parse(r#""a\udfff""#).unwrap()),"{body}");
        }
    }

}
