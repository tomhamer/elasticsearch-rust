#[allow(unused_imports)]
use super::{start, Es};
use serde_json::{json, Value};

async fn setup_source(es: &Es) {
    let (st, v) = es.put("/users", json!({"mappings": {"properties": {"email": {"type": "keyword"}, "name": {"type": "keyword"}, "zip": {"type": "integer"}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es
        .bulk(
            "/users/_bulk?refresh=true",
            &[
                json!({"index": {"_id": "1"}}),
                json!({"email": "a@x.io", "name": "Ann", "zip": 1000}),
                json!({"index": {"_id": "2"}}),
                json!({"email": "b@x.io", "name": "Bob", "zip": 2000}),
            ],
        )
        .await;
    assert_eq!(st, 200, "{v}");
}

#[tokio::test]
async fn enrich_policy_crud_execute_and_processor() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    setup_source(&es).await;
    let policy = json!({"match": {"indices": "users", "match_field": "email", "enrich_fields": ["name", "zip"]}});
    let (st, v) = es.put("/_enrich/policy/users-policy", policy.clone()).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_enrich/policy/users-policy", policy.clone()).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "resource_already_exists_exception");
    let (st, v) = es.put("/_enrich/policy/Bad", policy.clone()).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("must be lowercase"), "{v}");
    let (st, v) = es.put("/_enrich/policy/p2", json!({"foo": {"indices": "users", "match_field": "email", "enrich_fields": ["name"]}})).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("unsupported policy type [foo]"), "{v}");

    let (_, v) = es.get("/_enrich/policy/users-policy").await;
    assert_eq!(v["policies"][0]["config"]["match"]["name"], "users-policy");
    assert_eq!(v["policies"][0]["config"]["match"]["indices"], json!(["users"]));
    let (_, v) = es.get("/_enrich/policy").await;
    assert_eq!(v["policies"].as_array().unwrap().len(), 1);

    // Pipeline creation needs the enrich index.
    let pipe = json!({"processors": [{"enrich": {"policy_name": "users-policy", "field": "email", "target_field": "user"}}]});
    let (st, v) = es.put("/_ingest/pipeline/p", pipe.clone()).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("no enrich index exists for policy with name [users-policy]"), "{v}");

    let (st, v) = es.put_empty("/_enrich/policy/users-policy/_execute").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["status"]["phase"], "COMPLETE");
    let (st, v) = es.get("/.enrich-users-policy/_search").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 2);
    // Executing again swaps the alias and drops the old index.
    let (st, _) = es.post_empty("/_enrich/policy/users-policy/_execute").await;
    assert_eq!(st, 200);
    let (_, v) = es.get("/_alias/.enrich-users-policy?expand_wildcards=all").await;
    assert_eq!(v.as_object().unwrap().len(), 1, "{v}");

    let (st, v) = es.put("/_ingest/pipeline/p", pipe).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es
        .bulk(
            "/events/_bulk?pipeline=p&refresh=true",
            &[json!({"index": {"_id": "1"}}), json!({"email": "a@x.io"}), json!({"index": {"_id": "2"}}), json!({"email": "zz@x.io"}), json!({"index": {"_id": "3"}}), json!({"other": 1})],
        )
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["items"][2]["index"]["status"], 400, "{v}");
    assert_eq!(es.source("events", "1").await, json!({"email": "a@x.io", "user": {"email": "a@x.io", "name": "Ann", "zip": 1000}}));
    assert_eq!(es.source("events", "2").await, json!({"email": "zz@x.io"}));

    // max_matches > 1 → array; override=false keeps an existing target.
    let (_, v) = es
        .post(
            "/_ingest/pipeline/_simulate",
            json!({"pipeline": {"processors": [
                {"enrich": {"policy_name": "users-policy", "field": "emails", "target_field": "u", "max_matches": 5}},
                {"enrich": {"policy_name": "users-policy", "field": "emails", "target_field": "keep", "override": false, "max_matches": 2}},
                {"enrich": {"policy_name": "users-policy", "field": "missing", "target_field": "m", "ignore_missing": true}}
            ]}, "docs": [{"_source": {"emails": ["a@x.io", "b@x.io"], "keep": 1}}]}),
        )
        .await;
    let src = &v["docs"][0]["doc"]["_source"];
    assert_eq!(src["u"].as_array().map(|a| a.len()), Some(2), "{v}");
    assert_eq!(src["keep"], 1);
    let (_, v) = es.post("/_ingest/pipeline/_simulate", json!({"pipeline": {"processors": [{"enrich": {"policy_name": "users-policy", "field": "e", "target_field": "u", "max_matches": 200}}]}, "docs": [{"_source": {}}]})).await;
    assert!(v["error"]["reason"].as_str().unwrap_or("").contains("should be between 1 and 128"), "{v}");

    // ES|QL ENRICH.
    let (st, v) = es.post("/_query", json!({"query": "FROM events | KEEP email | ENRICH users-policy ON email WITH who = name | SORT email"})).await;
    assert_eq!(st, 200, "{v}");
    let cols: Vec<&str> = v["columns"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(cols, vec!["email", "who"]);
    assert_eq!(v["values"], json!([["a@x.io", "Ann"], ["zz@x.io", null]]));
    let (st, v) = es.post("/_query", json!({"query": "FROM events | KEEP email | ENRICH _any:users-policy | SORT email | LIMIT 1"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["values"], json!([["a@x.io", "Ann", 1000]]));
    let (st, v) = es.post("/_query", json!({"query": "FROM events | ENRICH nope"})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "verification_exception", "{v}");
    assert_eq!(v["error"]["reason"], "Found 1 problem\nline 1:22: cannot find enrich policy [nope]");

    // Stats, then delete (refused while a pipeline uses the policy).
    let (_, v) = es.get("/_enrich/_stats").await;
    assert_eq!(v["executing_policies"], json!([]));
    assert!(v["coordinator_stats"][0]["executed_searches_total"].as_u64().unwrap() > 0, "{v}");
    assert!(v["cache_stats"][0]["size_in_bytes"].is_number());
    let (st, v) = es.delete("/_enrich/policy/users-policy").await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "Could not delete policy [users-policy] because a pipeline is referencing it [p]");
    es.delete("/_ingest/pipeline/p").await;
    let (st, _) = es.delete("/_enrich/policy/users-policy").await;
    assert_eq!(st, 200);
    let (st, v) = es.delete("/_enrich/policy/users-policy").await;
    assert_eq!(st, 404);
    assert_eq!(v["error"]["reason"], "policy [users-policy] not found");
    let (_, v) = es.get("/_cat/indices/.enrich-*?expand_wildcards=all&format=json").await;
    assert_eq!(v, json!([]));
}

#[tokio::test]
async fn enrich_range_policy_and_async_execute() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    let (st, v) = es.put("/nets", json!({"mappings": {"properties": {"r": {"type": "integer_range"}, "label": {"type": "keyword"}}}})).await;
    assert_eq!(st, 200, "{v}");
    es.bulk(
        "/nets/_bulk?refresh=true",
        &[json!({"index": {}}), json!({"r": {"gte": 0, "lt": 10}, "label": "low"}), json!({"index": {}}), json!({"r": {"gte": 10, "lt": 100}, "label": "high"})],
    )
    .await;
    let (st, v) = es.put("/_enrich/policy/ranges", json!({"range": {"indices": ["nets"], "match_field": "r", "enrich_fields": ["label"]}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post_empty("/_enrich/policy/ranges/_execute?wait_for_completion=false").await;
    assert_eq!(st, 200, "{v}");
    let task = v["task"].as_str().unwrap().to_string();
    let (st, v) = es.get(&format!("/_tasks/{task}?wait_for_completion=true")).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/_ingest/pipeline/_simulate", json!({"pipeline": {"processors": [{"enrich": {"policy_name": "ranges", "field": "n", "target_field": "t"}}]}, "docs": [{"_source": {"n": 42}}]})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["docs"][0]["doc"]["_source"]["t"]["label"], "high", "{v}");
    let (st, v) = es.put_empty("/_enrich/policy/missing/_execute").await;
    assert_eq!(st, 404, "{v}");
    let _: Value = v;
}
