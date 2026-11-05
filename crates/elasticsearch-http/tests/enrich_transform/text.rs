//! geoip / user_agent / attachment processors, `_ingest/geoip/*`, and
//! `_text_structure`. The `.mmdb` files under `tests/data/geoip` are
//! MaxMind's test databases (CC BY-SA 4.0) as ES ships them in its
//! `geoip-fixture`.

#[allow(unused_imports)]
use super::{start, Es};
use reqwest::Method;
use serde_json::json;

fn with_geoip_dbs(dir: &std::path::Path) {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/geoip");
    let dst = dir.join("ingest-geoip");
    std::fs::create_dir_all(&dst).unwrap();
    for f in ["GeoLite2-City.mmdb", "GeoLite2-ASN.mmdb"] {
        std::fs::copy(src.join(f), dst.join(f)).unwrap();
    }
}

#[tokio::test]
async fn geoip_processor_with_databases() {
    let dir = tempfile::tempdir().unwrap();
    with_geoip_dbs(dir.path());
    let es = start(dir.path()).await;
    let (st, v) = es.put("/_ingest/pipeline/g", json!({"processors": [{"geoip": {"field": "ip"}}]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/t/_doc/1?pipeline=g", json!({"ip": "89.160.20.128"})).await;
    assert_eq!(st, 201, "{v}");
    let s = es.source("t", "1").await;
    let g = &s["geoip"];
    assert_eq!(g.as_object().unwrap().len(), 7, "{g}");
    assert_eq!(g["city_name"], json!("Linköping"));
    assert_eq!(g["country_iso_code"], json!("SE"));
    assert_eq!(g["region_iso_code"], json!("SE-E"));
    assert_eq!(g["region_name"], json!("Östergötland County"));
    assert_eq!(g["continent_name"], json!("Europe"));
    assert_eq!(g["location"], json!({"lat": 58.4167, "lon": 15.6167}));

    // lists, first_only=false: misses become null
    let (st, _) = es.put("/_ingest/pipeline/g2", json!({"processors": [{"geoip": {"field": "ip", "first_only": false, "properties": ["ip", "timezone"]}}]})).await;
    assert_eq!(st, 200);
    es.put("/t/_doc/2?pipeline=g2", json!({"ip": ["89.160.20.128", "127.0.0.1"]})).await;
    let s = es.source("t", "2").await;
    assert_eq!(s["geoip"], json!([{"ip": "89.160.20.128", "timezone": "Europe/Stockholm"}, null]));

    // ASN database
    es.put("/_ingest/pipeline/a", json!({"processors": [{"geoip": {"field": "ip", "database_file": "GeoLite2-ASN.mmdb"}}]})).await;
    es.put("/t/_doc/3?pipeline=a", json!({"ip": "89.160.20.128"})).await;
    let s = es.source("t", "3").await;
    assert_eq!(s["geoip"], json!({"ip": "89.160.20.128", "asn": 29518, "organization_name": "Bredband2 AB", "network": "89.160.0.0/17"}));

    // missing database: tagged like ES before its downloader fetched it
    es.put("/_ingest/pipeline/c", json!({"processors": [{"geoip": {"field": "ip", "database_file": "GeoLite2-Country.mmdb"}}]})).await;
    es.put("/t/_doc/4?pipeline=c", json!({"ip": "89.160.20.128"})).await;
    let s = es.source("t", "4").await;
    assert_eq!(s["tags"], json!(["_geoip_database_unavailable_GeoLite2-Country.mmdb"]));

    // invalid property
    let (st, v) = es.put("/_ingest/pipeline/bad", json!({"processors": [{"geoip": {"field": "ip", "properties": ["asn"]}}]})).await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().starts_with("[properties] illegal property value [asn]. valid values are [IP, COUNTRY_IN_EUROPEAN_UNION"), "{v}");

    // stats
    let (st, v) = es.get("/_ingest/geoip/stats").await;
    assert_eq!(st, 200);
    assert_eq!(v["stats"]["databases_count"], json!(2));
    let node = v["nodes"].as_object().unwrap().values().next().unwrap().clone();
    assert!(node["cache_stats"]["misses"].as_u64().unwrap() >= 1, "{node}");
    assert_eq!(node["databases"], json!([{"name": "GeoLite2-ASN.mmdb"}, {"name": "GeoLite2-City.mmdb"}]));

    // database configuration CRUD
    let (st, _) = es.put("/_ingest/geoip/database/my_db", json!({"name": "GeoIP2-City", "maxmind": {"account_id": "1234"}})).await;
    assert_eq!(st, 200);
    let (_, v) = es.get("/_ingest/geoip/database/my_db").await;
    assert_eq!(v["databases"][0]["database"]["maxmind"]["account_id"], json!("1234"));
    let (st, _) = es.delete("/_ingest/geoip/database/my_db").await;
    assert_eq!(st, 200);
    // ES 8.19 answers a GET of a missing configuration with 200 and no databases.
    let (st, v) = es.get("/_ingest/geoip/database/my_db").await;
    assert_eq!((st, v), (200, json!({"databases": []})));
}

#[tokio::test]
async fn user_agent_and_attachment() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    let ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.149 Safari/537.36";
    let (_, v) = es
        .post(
            "/_ingest/pipeline/_simulate",
            json!({"pipeline": {"processors": [{"user_agent": {"field": "ua", "extract_device_type": true}}]}, "docs": [{"_source": {"ua": ua}}]}),
        )
        .await;
    let u = &v["docs"][0]["doc"]["_source"]["user_agent"];
    assert_eq!(u["name"], json!("Chrome"), "{v}");
    assert_eq!(u["version"], json!("33.0.1750.149"));
    assert_eq!(u["os"], json!({"name": "Mac OS X", "version": "10.9.2", "full": "Mac OS X 10.9.2"}));
    assert_eq!(u["device"], json!({"name": "Mac", "type": "Desktop"}));
    assert_eq!(u["original"], json!(ua));
    let (st, v) = es.put("/_ingest/pipeline/x", json!({"processors": [{"user_agent": {"field": "ua", "regex_file": "nope.yml"}}]})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], json!("[regex_file] regex file [nope.yml] doesn't exist (has to exist at node startup)"));

    // attachment: plain text through the base64 decoder
    let (_, v) = es
        .post(
            "/_ingest/pipeline/_simulate",
            json!({"pipeline": {"processors": [{"attachment": {"field": "f", "indexed_chars": 30, "remove_binary": true}}]},
                   "docs": [{"_source": {"f": "VGhpcyBpcyBhbiBlbmdsaXNoIHRleHQgdG8gdGVzdCBpZiB0aGUgcGlwZWxpbmUgd29ya3M="}}]}),
        )
        .await;
    let s = &v["docs"][0]["doc"]["_source"];
    assert_eq!(s["attachment"], json!({"content": "This is an english text to tes", "content_type": "text/plain; charset=ISO-8859-1", "content_length": 30, "language": "en"}), "{v}");
    assert!(s.get("f").is_none());
    let (st, v) = es.put("/_ingest/pipeline/y", json!({"processors": [{"attachment": {"field": "f", "properties": ["nope"]}}]})).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("illegal field option [nope]. valid values are [CONTENT, TITLE"), "{v}");
}

#[tokio::test]
async fn text_structure_apis() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    // NDJSON
    let nd = "{\"airline\": \"AAL\", \"responsetime\": 132.2046, \"time\": 1403481600}\n{\"airline\": \"JZA\", \"responsetime\": 990.4628, \"time\": 1403481700}\n{\"airline\": \"AAL\", \"responsetime\": 134.2046, \"time\": 1403481800}\n";
    let (st, body) = es.text(Method::POST, "/_text_structure/find_structure?explain=true", Some(nd.to_string())).await;
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["format"], json!("ndjson"));
    assert_eq!(v["has_byte_order_marker"], json!(false));
    assert_eq!(v["timestamp_field"], json!("time"));
    assert_eq!(v["mappings"]["properties"]["@timestamp"], json!({"type": "date"}));
    assert_eq!(v["ingest_pipeline"]["processors"][0]["date"], json!({"field": "time", "formats": ["UNIX"]}));
    assert_eq!(v["field_stats"]["time"]["earliest"], json!("1403481600"));
    assert!(v["explanation"].is_array());

    // CSV with a header row
    let csv = "name,count,time\nfoo,1,2020-01-01T00:00:00Z\nbar,2,2020-01-02T00:00:00Z\nbaz,3,2020-01-03T00:00:00Z\n";
    let (st, body) = es.text(Method::POST, "/_text_structure/find_structure", Some(csv.to_string())).await;
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["format"], json!("delimited"));
    assert_eq!(v["column_names"], json!(["name", "count", "time"]));
    assert_eq!(v["exclude_lines_pattern"], json!("^\"?name\"?,\"?count\"?,\"?time\"?"));
    assert_eq!(v["ingest_pipeline"]["processors"][0]["csv"]["target_fields"], json!(["name", "count", "time"]));
    assert_eq!(v["mappings"]["properties"]["count"], json!({"type": "long"}));

    // log lines: last message is dropped (it may be incomplete)
    let log = "2019-05-16 16:56:14 INFO line 1 first\n  continued\n2019-05-16 16:56:15 WARN line 2 second\n2019-05-16 16:56:16 INFO line 3 third\n";
    let (st, body) = es.text(Method::POST, "/_text_structure/find_structure", Some(log.to_string())).await;
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["format"], json!("semi_structured_text"));
    assert_eq!(v["num_messages_analyzed"], json!(2));
    assert_eq!(v["multiline_start_pattern"], json!("^\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}"));
    assert_eq!(v["grok_pattern"], json!("%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:loglevel} .*? %{INT:field} .*"));
    assert_eq!(v["need_client_timezone"], json!(true));

    // messages
    let (st, v) = es
        .post("/_text_structure/find_message_structure", json!({"messages": ["2019-05-16 16:56:14 line 1 xyz", "2019-05-16 16:56:14 line 2 xyz", "2019-05-16 16:56:14 line 3 xyz"]}))
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["grok_pattern"], json!("%{TIMESTAMP_ISO8601:timestamp} .*? %{INT:field} .*"));

    // field structure from an index
    for (i, t) in ["{\"a\": 1}", "{\"a\": 2}", "{\"a\": 3}"].iter().enumerate() {
        es.put(&format!("/idx/_doc/{i}?refresh=true"), json!({ "message": t })).await;
    }
    let (st, v) = es.get("/_text_structure/find_field_structure?index=idx&field=message").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["format"], json!("ndjson"));
    assert_eq!(v["mappings"]["properties"]["a"], json!({"type": "long"}));

    // errors
    let (st, v) = es.text(Method::POST, "/_text_structure/find_structure?charset=wrong", Some(nd.to_string())).await;
    assert_eq!(st, 400, "{v}");
    let (st, _) = es.text(Method::POST, "/_text_structure/find_structure", Some("one line\n".to_string())).await;
    assert_eq!(st, 400);

    // test_grok_pattern
    let (st, v) = es
        .post("/_text_structure/test_grok_pattern", json!({"grok_pattern": "%{WORD}.*%{WORD:first_name} %{WORD:last_name}!", "text": ["Hello   Dave Roberts!", "no"]}))
        .await;
    assert_eq!(st, 200);
    assert_eq!(
        v,
        json!({"matches": [
            {"matched": true, "fields": {"first_name": [{"match": "Dave", "offset": 8, "length": 4}], "last_name": [{"match": "Roberts", "offset": 13, "length": 7}]}},
            {"matched": false}
        ]})
    );
}
