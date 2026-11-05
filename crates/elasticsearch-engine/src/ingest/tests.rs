use super::*;
use serde_json::json;

fn run(pipeline: Value, src: Value) -> std::result::Result<IngestDocument, String> {
    let store = IngestService::in_memory();
    let p = compile_pipeline("p", &pipeline).map_err(|e| e.reason)?;
    let mut doc = IngestDocument::new(src.as_object().unwrap().clone(), "idx", Some("1"), None, "2024-01-01T00:00:00Z".into());
    let cx = PipelineContext::new(&store);
    match run_pipeline(&p, &mut doc, &cx) {
        Ok(()) | Err(ProcessorError::Terminated) => Ok(doc),
        Err(ProcessorError::Dropped) => Err("dropped".into()),
        Err(ProcessorError::Failed(e, _)) => Err(e.reason),
    }
}

fn src(d: &IngestDocument) -> Value {
    d.source_value()
}

#[test]
fn set_remove_rename_append() {
    let d = run(
        json!({"processors": [
            {"set": {"field": "a.b", "value": "{{x}}-{{_index}}"}},
            {"set": {"field": "x", "value": "nope", "override": false}},
            {"rename": {"field": "y", "target_field": "z"}},
            {"remove": {"field": ["w", "missing"], "ignore_missing": true}},
            {"append": {"field": "tags", "value": ["t1", "t2"]}},
            {"set": {"field": "copy", "copy_from": "a"}}
        ]}),
        json!({"x": "v", "y": 1, "w": 2}),
    )
    .unwrap();
    assert_eq!(src(&d), json!({"x": "v", "a": {"b": "v-idx"}, "z": 1, "tags": ["t1", "t2"], "copy": {"b": "v-idx"}}));
    let e = run(json!({"processors": [{"rename": {"field": "nope", "target_field": "x"}}]}), json!({})).unwrap_err();
    assert_eq!(e, "field [nope] doesn't exist");
    let e = run(json!({"processors": [{"remove": {"field": "nope"}}]}), json!({})).unwrap_err();
    assert_eq!(e, "field [nope] not present as part of path [nope]");
}

#[test]
fn config_errors() {
    let e = compile_pipeline("p", &json!({"processors": [{"set": {"value": 1}}]})).err().unwrap();
    assert_eq!(e.kind, "parse_exception");
    assert_eq!(e.reason, "[field] required property is missing");
    let e = compile_pipeline("p", &json!({"processors": [{"nope": {}}]})).err().unwrap();
    assert_eq!(e.reason, "No processor type exists with name [nope]");
    let e = compile_pipeline("p", &json!({"processors": [{"set": {"field": "a", "value": 1, "bogus": 1}}]})).err().unwrap();
    assert_eq!(e.reason, "processor [set] doesn't support one or more provided configuration parameters [bogus]");
    let e = compile_pipeline("p", &json!({"processors": [], "invalid_field": {}})).err().unwrap();
    assert_eq!(e.kind, "parse_exception");
    let e = compile_pipeline("p", &json!({"description": "x"})).err().unwrap();
    assert_eq!(e.reason, "[processors] required property is missing");
}

#[test]
fn failures_and_handlers() {
    // processor-level on_failure sees the failure metadata
    let d = run(
        json!({"processors": [{"rename": {"field": "nope", "target_field": "x", "tag": "t1",
            "on_failure": [{"set": {"field": "err", "value": "{{_ingest.on_failure_message}}|{{_ingest.on_failure_processor_type}}|{{_ingest.on_failure_processor_tag}}"}}]}},
            {"set": {"field": "after", "value": true}}]}),
        json!({}),
    )
    .unwrap();
    assert_eq!(src(&d), json!({"err": "field [nope] doesn't exist|rename|t1", "after": true}));
    assert!(d.ingest.get("on_failure_message").is_none());
    // ignore_failure
    let d = run(json!({"processors": [{"fail": {"message": "x", "ignore_failure": true}}, {"set": {"field": "ok", "value": 1}}]}), json!({})).unwrap();
    assert_eq!(src(&d), json!({"ok": 1}));
    // pipeline-level on_failure stops the pipeline
    let d = run(
        json!({"processors": [{"fail": {"message": "boom {{a}}"}}, {"set": {"field": "never", "value": 1}}],
               "on_failure": [{"set": {"field": "msg", "value": "{{_ingest.on_failure_message}}"}}]}),
        json!({"a": 1}),
    )
    .unwrap();
    assert_eq!(src(&d), json!({"a": 1, "msg": "boom 1"}));
    assert_eq!(run(json!({"processors": [{"drop": {}}]}), json!({})).unwrap_err(), "dropped");
    // conditions
    let d = run(json!({"processors": [{"set": {"field": "x", "value": 1, "if": "ctx.a == 'y'"}}, {"set": {"field": "z", "value": 2, "if": "ctx.a?.length() > 0"}}]}), json!({"a": "b"})).unwrap();
    assert_eq!(src(&d), json!({"a": "b", "z": 2}));
    let d = run(json!({"processors": [{"set": {"field": "x", "value": 1}}, {"terminate": {}}, {"set": {"field": "y", "value": 1}}]}), json!({})).unwrap();
    assert_eq!(src(&d), json!({"x": 1}));
}

#[test]
fn transforms() {
    let d = run(
        json!({"processors": [
            {"lowercase": {"field": "a"}}, {"uppercase": {"field": "b"}}, {"trim": {"field": "c"}},
            {"split": {"field": "d", "separator": ","}}, {"join": {"field": "e", "separator": "-"}},
            {"convert": {"field": "f", "type": "integer"}}, {"convert": {"field": "g", "type": "boolean"}},
            {"convert": {"field": "h", "type": "double"}}, {"convert": {"field": "i", "type": "string"}},
            {"json": {"field": "j", "target_field": "jj"}},
            {"bytes": {"field": "k"}}, {"urldecode": {"field": "l"}}, {"html_strip": {"field": "m"}},
            {"gsub": {"field": "n", "pattern": "(\\d+)", "replacement": "<$1>"}},
            {"sort": {"field": "o", "order": "desc"}},
            {"dot_expander": {"field": "p.q"}}
        ]}),
        json!({"a": "AbC", "b": ["x", "y"], "c": "  t ", "d": "1,2,,", "e": [1, "a"], "f": "42", "g": "TRUE", "h": "1.5", "i": 3.0,
               "j": "{\"x\": [1]}", "k": "2kb", "l": "a%20b", "m": "<b>bold</b>", "n": "a1b22", "o": [1, 3, 2], "p.q": 7}),
    )
    .unwrap();
    let s = src(&d);
    assert_eq!(s["a"], json!("abc"));
    assert_eq!(s["b"], json!(["X", "Y"]));
    assert_eq!(s["c"], json!("t"));
    assert_eq!(s["d"], json!(["1", "2"]));
    assert_eq!(s["e"], json!("1-a"));
    assert_eq!(s["f"], json!(42));
    assert_eq!(s["g"], json!(true));
    assert_eq!(s["h"], json!(1.5));
    assert_eq!(s["i"], json!("3.0"));
    assert_eq!(s["jj"], json!({"x": [1]}));
    assert_eq!(s["k"], json!(2048));
    assert_eq!(s["l"], json!("a b"));
    assert_eq!(s["m"], json!("bold"));
    assert_eq!(s["n"], json!("a<1>b<22>"));
    assert_eq!(s["o"], json!([3, 2, 1]));
    assert_eq!(s["p"], json!({"q": 7}));
    let e = run(json!({"processors": [{"convert": {"field": "f", "type": "integer"}}]}), json!({"f": "x"})).unwrap_err();
    assert_eq!(e, "unable to convert [x] to integer");
}

#[test]
fn date_grok_dissect_kv_csv() {
    let d = run(
        json!({"processors": [
            {"date": {"field": "t", "formats": ["dd/MMM/yyyy:HH:mm:ss Z"]}},
            {"date": {"field": "u", "target_field": "u2", "formats": ["UNIX_MS"], "timezone": "Europe/Amsterdam"}},
            {"grok": {"field": "msg", "patterns": ["%{IP:client} %{WORD:method} %{NUMBER:bytes:int}"]}},
            {"dissect": {"field": "line", "pattern": "%{a} %{b} [%{c}]"}},
            {"kv": {"field": "kvs", "field_split": " ", "value_split": "=", "target_field": "kv"}},
            {"csv": {"field": "row", "target_fields": ["c1", "c2", "c3"]}}
        ]}),
        json!({"t": "10/Oct/2000:13:55:36 -0700", "u": "0", "msg": "1.2.3.4 GET 15", "line": "x y [z w]", "kvs": "a=1 b=2", "row": "p,\"q,r\",s"}),
    )
    .unwrap();
    let s = src(&d);
    assert_eq!(s["@timestamp"], json!("2000-10-10T20:55:36.000Z"));
    assert_eq!(s["u2"], json!("1970-01-01T01:00:00.000+01:00"));
    assert_eq!((s["client"].clone(), s["method"].clone(), s["bytes"].clone()), (json!("1.2.3.4"), json!("GET"), json!(15)));
    assert_eq!((s["a"].clone(), s["b"].clone(), s["c"].clone()), (json!("x"), json!("y"), json!("z w")));
    assert_eq!(s["kv"], json!({"a": "1", "b": "2"}));
    assert_eq!((s["c1"].clone(), s["c2"].clone(), s["c3"].clone()), (json!("p"), json!("q,r"), json!("s")));
    let e = run(json!({"processors": [{"grok": {"field": "m", "patterns": ["%{NUMBER:n}"]}}]}), json!({"m": "abc"})).unwrap_err();
    assert_eq!(e, "Provided Grok expressions do not match field value: [abc]");
    let e = run(json!({"processors": [{"date": {"field": "t", "formats": ["ISO8601"]}}]}), json!({"t": "nope"})).unwrap_err();
    assert_eq!(e, "unable to parse date [nope]");
}

#[test]
fn foreach_script_pipeline_and_more() {
    let d = run(
        json!({"processors": [
            {"foreach": {"field": "list", "processor": {"uppercase": {"field": "_ingest._value"}}}},
            {"foreach": {"field": "objs", "processor": {"set": {"field": "_ingest._value.seen", "value": true}}}},
            {"script": {"source": "ctx.total = 0; for (def x : ctx.nums) { ctx.total += x * params.f }", "params": {"f": 2}}},
            {"fingerprint": {"fields": ["list"], "method": "SHA-256"}},
            {"user_agent": {"field": "ua"}},
            {"uri_parts": {"field": "u"}},
            {"geoip": {"field": "ip"}}
        ]}),
        json!({"list": ["a", "b"], "objs": [{"x": 1}], "nums": [1, 2], "ua": "curl/7.64.1", "u": "http://example.com/a.html", "ip": "8.8.8.8"}),
    )
    .unwrap();
    let s = src(&d);
    assert_eq!(s["list"], json!(["A", "B"]));
    assert_eq!(s["objs"], json!([{"x": 1, "seen": true}]));
    assert_eq!(s["total"], json!(6));
    assert!(s["fingerprint"].as_str().unwrap().len() == 44);
    assert_eq!(s["user_agent"]["name"], json!("curl"));
    assert_eq!(s["user_agent"]["original"], json!("curl/7.64.1"));
    assert_eq!(s["url"]["domain"], json!("example.com"));
    // No GeoIP database available: ES tags the document.
    assert_eq!(s["tags"], json!(["_geoip_database_unavailable_GeoLite2-City.mmdb"]));
    assert!(d.ingest.get("_value").map(|v| v.is_null()).unwrap_or(true));
    // nested pipelines + cycle detection
    let store = IngestService::in_memory();
    store.put("inner", json!({"processors": [{"set": {"field": "inner", "value": "{{_ingest.pipeline}}"}}]}), None).unwrap();
    store.put("outer", json!({"processors": [{"pipeline": {"name": "inner"}}, {"set": {"field": "outer", "value": "{{_ingest.pipeline}}"}}]}), None).unwrap();
    store.put("loop", json!({"processors": [{"pipeline": {"name": "loop"}}]}), None).unwrap();
    let cx = PipelineContext::new(&store);
    let mut doc = IngestDocument::new(Map::new(), "i", None, None, "ts".into());
    run_pipeline(&store.pipeline("outer").unwrap(), &mut doc, &cx).unwrap();
    assert_eq!(doc.source_value(), json!({"inner": "inner", "outer": "outer"}));
    let mut doc = IngestDocument::new(Map::new(), "i", None, None, "ts".into());
    match run_pipeline(&store.pipeline("loop").unwrap(), &mut doc, &cx) {
        Err(ProcessorError::Failed(e, _)) => assert_eq!(e.reason, "Cycle detected for pipeline: loop"),
        _ => panic!("expected cycle"),
    }
    // if_version
    assert!(store.put("inner", json!({"processors": []}), Some(3)).is_err());
}

#[test]
fn pipeline_definition_validation() {
    let ok = json!({"set": {"field": "a", "value": 1}});
    let e = compile_pipeline("p", &json!({"processors": [ok.clone()], "deprecated": "x"})).err().unwrap();
    assert_eq!(e.reason, "[deprecated] property isn't a boolean, but of type [java.lang.String]");
    let e = compile_pipeline("p", &json!({"processors": [ok.clone()], "description": 5})).err().unwrap();
    assert_eq!(e.reason, "[description] property isn't a string, but of type [java.lang.Integer]");
    let e = compile_pipeline("p", &json!({"processors": [ok.clone()], "version": 1.5})).err().unwrap();
    assert_eq!(e.reason, "[version] property cannot be converted to an int [1.5]");
    let e = compile_pipeline("p", &json!({"processors": [ok.clone()], "on_failure": []})).err().unwrap();
    assert_eq!(e.reason, "pipeline [p] cannot have an empty on_failure option defined");
    let e = compile_pipeline("p", &json!([1, 2])).err().unwrap();
    assert_eq!(e.reason, "[processors] required property is missing");
    assert!(compile_pipeline("p", &json!({"processors": [], "version": "5", "deprecated": null, "_meta": null})).is_ok());
}

#[test]
fn store_semantics() {
    let st = IngestService::in_memory();
    let body = json!({"processors": []});
    st.put("d1", body.clone(), None).unwrap();
    st.put("d2", body.clone(), None).unwrap();
    // DELETE does not split commas; `_all` is not special for GET
    assert_eq!(st.delete("d1,d2").unwrap_err().reason, "pipeline [d1,d2] is missing");
    assert_eq!(st.delete("zz*").unwrap_err().reason, "pipeline [zz*] is missing");
    assert!(st.get(Some("_all")).is_empty());
    assert_eq!(st.get(Some("d1,nope")).len(), 1);
    assert_eq!(st.delete("d*").unwrap().len(), 2);
}

#[test]
fn reroute_flag_halts_enclosing_pipelines() {
    let d = run(
        json!({"processors": [
            {"reroute": {"destination": "other"}},
            {"set": {"field": "after", "value": true}}
        ]}),
        json!({"x": 1}),
    )
    .unwrap();
    assert!(d.rerouted);
    assert_eq!(d.meta_str("_index").as_deref(), Some("other"));
    assert!(d.source_value().get("after").is_none());
}

#[test]
fn reroute_static_value_rules() {
    let mk = |f: &str, v: &str| compile_pipeline("p", &json!({"processors": [{"reroute": {f: v}}]}));
    assert!(mk("dataset", "apm.transaction.1m").is_ok());
    assert!(mk("namespace", "a-b").is_ok());
    assert!(mk("dataset", "a-b").is_err());
    assert!(mk("type", "Metrics").is_err());
    assert!(mk("dataset", "{{ds}}").is_ok());
    assert!(mk("dataset", &"a".repeat(101)).is_err());
    assert_eq!(mk("dataset", "a b").err().unwrap().reason, "[dataset] 'a b' contains disallowed characters");
}

#[test]
fn cluster_pipeline_restore_preserves_managed_dates() {
    let source = IngestService::in_memory();
    source.put("p", json!({"processors":[{"set":{"field":"restored","value":true}}]}), None).unwrap();
    let config = source.get(Some("p"))[0].1.clone();
    let target = IngestService::in_memory();
    assert!(target.put("p", config.clone(), None).is_err());
    target.restore_from_cluster_state("p", config.clone()).unwrap();
    assert_eq!(target.get(Some("p"))[0].1, config);
    assert!(target.restore_from_cluster_state("bad", json!({"processors":[{"missing_processor":{}}]})).is_err());
}

#[test]
fn inference_results_field_keeps_sparse_tokens_in_mapped_field() {
    let dir = tempfile::tempdir().unwrap();
    let service = std::sync::Arc::new(crate::inference::InferenceService::open(dir.path()));
    crate::inference::scoped(Some(service), || {
        let pipeline = json!({"processors":[{"inference":{
            "model_id":".elser_model_2", "target_field":"ml",
            "field_map":{"body":"text_field"},
            "inference_config":{"text_expansion":{"results_field":"tokens"}}
        }}]});
        let output = src(&run(pipeline, json!({"body":"search engine performance"})).unwrap());
        assert!(output["ml"]["tokens"].is_object(), "{output}");
        assert!(output["ml"].get("predicted_value").is_none());
        assert_eq!(output["ml"]["model_id"], ".elser_model_2");
        let default = src(&run(json!({"processors":[{"inference":{
            "model_id":".elser_model_2", "target_field":"ml", "field_map":{"body":"text_field"}
        }}]}), json!({"body":"search engine performance"})).unwrap());
        assert_eq!(output["ml"]["tokens"], default["ml"]["predicted_value"]);
        // Many feature names are values of one mapped rank_features field,
        // never thousands of dynamic mapping entries.
        let mut output = output;
        output["ml"]["tokens"] = Value::Object((0..1200).map(|i|
            (format!("token{i}"), json!(1.0))).collect());
        let registry = crate::analysis::AnalysisRegistry::default();
        let mut mapping = crate::mapping::Mappings::parse(&json!({"properties":{
            "body":{"type":"text"}, "ml":{"properties":{
                "model_id":{"type":"keyword"}, "tokens":{"type":"rank_features"}
            }}
        }}), &registry).unwrap();
        mapping.assign_core_ids();
        let lookup = crate::mapping::FieldLookup::build(&mapping, &registry);
        let (_, update) = crate::document::parse_with_dynamic(&output, "1", &mapping, &lookup, 1000).unwrap();
        assert!(update.is_none());
    });
}

#[test]
fn ingest_script_source_preserves_raw_parameters_literals_and_output() {
    use crate::source_json::Source;
    for raw in [
        r#"{"source":"ctx.raw=params.s; ctx.half=params.pair.substring(0,1)","params":{"s":"\ud800","pair":"😀"}}"#,
        r#"{"source":"ctx.raw='\ud800'; ctx.half='😀'.substring(0,1)"}"#,
    ] {
        let script=super::Script::from_source(&Source::parse(raw).unwrap()).unwrap();
        script.validate(crate::script::ScriptContext::Ingest).unwrap();
        let mut doc=Source::parse(r#"{"old":"\udfff","original":"safe"}"#).unwrap();
        script.execute_source(&mut doc).unwrap();
        let expected=Source::parse(r#"{"old":"\udfff","original":"safe","raw":"\ud800","half":"\ud83d"}"#).unwrap();
        assert!(doc.equivalent(&expected),"{}",doc.text());
    }
    let script=super::Script::from_source(&Source::parse(r#"{"source":"ctx[params.key]='bad'","params":{"key":"\ud800"}}"#).unwrap()).unwrap();
    let mut doc=Source::parse(r#"{"original":"safe"}"#).unwrap();
    let before=doc.text();
    assert!(script.execute_source(&mut doc).is_err());
    assert_eq!(doc.text(),before);
}

#[test]
fn pipeline_source_compiler_preserves_nested_script_inputs() {
    use crate::source_json::Source;
    let raw=Source::parse(r#"{"processors":[
      {"script":{"source":"ctx.a=(int)params.s.charAt(0)","params":{"s":"\ud800"},"if":{"source":"params.s.length()==1","params":{"s":"\udfff"}}}},
      {"foreach":{"field":"items","processor":{"script":{"source":"ctx.b=(int)'\udfff'.charAt(0)"}}}},
      {"fail":{"message":"expected","on_failure":[{"script":{"source":"ctx.c=(int)params.s.charAt(0)","params":{"s":"\ud801"}}}]}}
    ]}"#).unwrap();
    let pipeline=compile_pipeline_source("raw",&raw).unwrap();
    let store=IngestService::in_memory();
    let cx=PipelineContext::new(&store);
    let mut doc=IngestDocument::new(json!({"items":[1]}).as_object().unwrap().clone(),"idx",Some("1"),None,"now".into());
    run_pipeline(&pipeline,&mut doc,&cx).unwrap();
    assert_eq!(doc.ctx["a"],55296);
    assert_eq!(doc.ctx["b"],57343);
    assert_eq!(doc.ctx["c"],55297);
    let invalid=Source::parse(r#"{"processors":[{"script":{"source":"true","params":{"s":"\ud800"},"unknown":1}}]}"#).unwrap();
    assert!(compile_pipeline_source("invalid",&invalid).is_err());
}

#[test]
fn lossless_ingest_self_reference_matches_source_generation_error() {
    use crate::source_json::Source;
    let script=super::Script::from_source(&Source::parse(r#"{"source":"ctx.loop=ctx; ctx._index='changed'"}"#).unwrap()).unwrap();
    let mut context=Source::parse(r#"{"_index":"original","raw":"\ud800"}"#).unwrap();
    let before=context.text();
    let error=script.execute_source(&mut context).unwrap_err();
    assert!(error.to_string().contains("Failed to generate the source document for ingest pipeline"),"{error}");
    assert_eq!(context.text(),before);
}
