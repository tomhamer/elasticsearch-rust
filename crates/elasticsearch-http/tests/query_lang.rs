//! HTTP tests for ES|QL, SQL, EQL, search templates, rank_eval, search
//! applications, query rules and the synonyms API.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

struct Es {
    base: String,
    c: Client,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    start_node(node).await
}

async fn start_node(node: std::sync::Arc<Node>) -> Es {
    // Like ES's own test clusters: a self-generated trial license, so the
    // license-gated APIs exercised here are served (ES 403s them on basic).
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("trial"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new() }
}

impl Es {
    async fn call(&self, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
        (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
    }
    async fn raw(&self, m: Method, p: &str, body: &str, ct: &str) -> (u16, String, String) {
        let resp = self.c.request(m, format!("{}{}", self.base, p)).header("content-type", ct).body(body.to_string()).send().await.unwrap();
        let st = resp.status().as_u16();
        let ctype = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        (st, ctype, resp.text().await.unwrap())
    }
    async fn get(&self, p: &str) -> (u16, Value) {
        self.call(Method::GET, p, None).await
    }
    async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::POST, p, Some(b)).await
    }
    async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::PUT, p, Some(b)).await
    }
    async fn delete(&self, p: &str) -> (u16, Value) {
        self.call(Method::DELETE, p, None).await
    }
    async fn bulk(&self, lines: &[Value]) {
        let mut s = String::new();
        for l in lines {
            s.push_str(&l.to_string());
            s.push('\n');
        }
        let r = self.c.post(format!("{}/_bulk?refresh=true", self.base)).header("content-type", "application/x-ndjson").body(s).send().await.unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["errors"], false, "{v}");
    }
}

/// `emp`: 6 employees.
async fn seed(es: &Es) {
    let (st, v) = es
        .put(
            "/emp",
            json!({"mappings": {"properties": {
                "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "dept": {"type": "keyword"},
                "salary": {"type": "long"},
                "height": {"type": "double"},
                "hired": {"type": "date"},
                "active": {"type": "boolean"}
            }}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    let rows = [
        ("1", "Alice Smith", "eng", 100, 1.70, "2020-01-15T00:00:00Z", true),
        ("2", "Bob Jones", "eng", 80, 1.80, "2021-06-01T00:00:00Z", true),
        ("3", "Carol White", "sales", 60, 1.65, "2019-03-10T00:00:00Z", false),
        ("4", "Dan Brown", "sales", 70, 1.90, "2022-11-20T00:00:00Z", true),
        ("5", "Eve Black", "hr", 50, 1.60, "2023-02-01T00:00:00Z", true),
        ("6", "Frank Green", "eng", 120, 1.75, "2018-07-07T00:00:00Z", false),
    ];
    let mut lines = Vec::new();
    for (id, n, d, s, h, t, a) in rows {
        lines.push(json!({"index": {"_index": "emp", "_id": id}}));
        lines.push(json!({"name": n, "dept": d, "salary": s, "height": h, "hired": t, "active": a}));
    }
    es.bulk(&lines).await;
}

#[tokio::test]
async fn esql_basic_pipeline() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    // FROM | WHERE | SORT | KEEP | LIMIT
    let (st, v) = es.post("/_query", json!({"query": "FROM emp | WHERE salary > 65 AND dept == \"eng\" | SORT salary DESC | KEEP name, salary | LIMIT 2"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["columns"], json!([{"name": "name", "type": "text"}, {"name": "salary", "type": "long"}]));
    assert_eq!(v["values"], json!([["Frank Green", 120], ["Alice Smith", 100]]));
    assert_eq!(v["is_partial"], false);
    assert!(v["took"].is_number());
    // STATS ... BY
    let (st, v) = es.post("/_query", json!({"query": "FROM emp | STATS avg_salary = AVG(salary), n = COUNT(*), max_h = MAX(height) BY dept | SORT dept"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["columns"], json!([{"name": "avg_salary", "type": "double"}, {"name": "n", "type": "long"}, {"name": "max_h", "type": "double"}, {"name": "dept", "type": "keyword"}]));
    assert_eq!(v["values"], json!([[100.0, 3, 1.8, "eng"], [50.0, 1, 1.6, "hr"], [65.0, 2, 1.9, "sales"]]));
    // EVAL, dates, columnar, metadata
    let (st, v) = es
        .post("/_query", json!({"query": "FROM emp METADATA _id | WHERE hired >= \"2021-01-01\" | EVAL y = DATE_EXTRACT(\"year\", hired) | SORT hired | KEEP _id, y, hired", "columnar": true}))
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["values"], json!([["2", "4", "5"], [2021, 2022, 2023], ["2021-06-01T00:00:00.000Z", "2022-11-20T00:00:00.000Z", "2023-02-01T00:00:00.000Z"]]));
    // MATCH full-text pushdown + _score
    let (st, v) = es.post("/_query", json!({"query": "FROM emp METADATA _score | WHERE MATCH(name, \"alice\") | KEEP name, _score"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["values"].as_array().unwrap().len(), 1);
    assert!(v["values"][0][1].as_f64().unwrap() > 0.0);
    // params
    let (st, v) = es.post("/_query", json!({"query": "FROM emp | WHERE salary >= ? AND dept == ?d | STATS c = COUNT(*)", "params": [70, {"d": "eng"}]})).await;
    assert_eq!(st, 400, "ES9 rejects mixed named and unnamed parameters: {v}");
    let (st, v) = es.post("/_query", json!({"query": "FROM emp | WHERE salary >= ?salary AND dept == ?d | STATS c = COUNT(*)", "params": [{"salary":70}, {"d":"eng"}]})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["values"], json!([[3]]));
    // csv / txt formats
    let (st, ct, body) = es.raw(Method::POST, "/_query?format=csv", r#"{"query": "FROM emp | SORT salary | KEEP dept, salary | LIMIT 2"}"#, "application/json").await;
    assert_eq!(st, 200);
    assert!(ct.starts_with("text/csv"), "{ct}");
    assert_eq!(body, "dept,salary\r\nhr,50\r\nsales,60\r\n");
    let (st, _, body) = es.raw(Method::POST, "/_query?format=txt", r#"{"query": "ROW a = 1, b = \"x\""}"#, "application/json").await;
    assert_eq!(st, 200);
    assert_eq!(body, "       a       |       b       \n---------------+---------------\n1              |x              \n");
    // errors
    let (st, v) = es.post("/_query", json!({"query": "FROM emp | WHERE nope > 1"})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["type"], "verification_exception");
    assert!(v["error"]["reason"].as_str().unwrap().contains("Unknown column [nope]"));
    let (st, v) = es.post("/_query", json!({"query": "FROM missing_idx"})).await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("Unknown index [missing_idx]"));
    // async
    let (st, v) = es.post("/_query/async", json!({"query": "FROM emp | STATS c = COUNT(*)", "keep_on_completion": true})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["is_running"], false);
    let id = v["id"].as_str().unwrap().to_string();
    let (st, v) = es.get(&format!("/_query/async/{id}")).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["values"], json!([[6]]));
    let (st, _) = es.delete(&format!("/_query/async/{id}")).await;
    assert_eq!(st, 200);
    let (st, _) = es.get(&format!("/_query/async/{id}")).await;
    assert_eq!(st, 404);
}

#[tokio::test]
async fn sql_select() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es.post("/_sql", json!({"query": "SELECT dept, salary FROM emp WHERE salary > 60 ORDER BY salary DESC LIMIT 3"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["columns"], json!([{"name": "dept", "type": "keyword"}, {"name": "salary", "type": "long"}]));
    assert_eq!(v["rows"], json!([["eng", 120], ["eng", 100], ["eng", 80]]));
    let (st, v) = es.post("/_sql", json!({"query": "SELECT dept, COUNT(*) AS c, AVG(salary) FROM emp GROUP BY dept HAVING COUNT(*) > 1 ORDER BY dept"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["columns"], json!([{"name": "dept", "type": "keyword"}, {"name": "c", "type": "long"}, {"name": "AVG(salary)", "type": "double"}]));
    assert_eq!(v["rows"], json!([["eng", 3, 100.0], ["sales", 2, 65.0]]));
    // paging with cursor
    let (st, v) = es.post("/_sql", json!({"query": "SELECT salary FROM emp ORDER BY salary", "fetch_size": 4})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["rows"].as_array().unwrap().len(), 4);
    let cursor = v["cursor"].as_str().unwrap().to_string();
    let (st, v) = es.post("/_sql", json!({"cursor": cursor})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["rows"], json!([[100], [120]]));
    assert!(v.get("cursor").is_none());
    // txt format
    let (st, _, body) = es.raw(Method::POST, "/_sql?format=txt", r#"{"query": "SELECT dept FROM emp WHERE salary = 50"}"#, "application/json").await;
    assert_eq!(st, 200);
    assert_eq!(body, "     dept      \n---------------\nhr             \n");
    // translate
    let (st, v) = es.post("/_sql/translate", json!({"query": "SELECT dept FROM emp WHERE salary > 60 ORDER BY salary"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["size"], 1000);
    assert_eq!(v["_source"], false);
    assert_eq!(v["query"]["bool"]["filter"][0], json!({"range": {"salary": {"gt": 60}}}));
    assert_eq!(v["fields"], json!([{"field": "dept"}]));
    // SHOW TABLES / errors
    let (st, v) = es.post("/_sql", json!({"query": "SHOW TABLES"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["rows"][0][1], "emp");
    let (st, v) = es.post("/_sql", json!({"query": "SELEC x FROM emp"})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["type"], "parsing_exception");
}

#[tokio::test]
async fn sql_star_projection_preserves_paged_values() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (status, first) = es.post("/_sql", json!({"query": "SELECT * FROM emp ORDER BY salary", "fetch_size": 2})).await;
    assert_eq!(status, 200, "{first}");
    let columns = first["columns"].as_array().unwrap();
    assert!(!columns.iter().any(|c| c["name"] == "name.keyword"));
    let name = columns.iter().position(|c| c["name"] == "name").unwrap();
    let salary = columns.iter().position(|c| c["name"] == "salary").unwrap();
    let mut rows = first["rows"].as_array().unwrap().clone();
    assert_eq!(rows.len(), 2);
    let mut cursor = first["cursor"].clone();
    for page in 0..2 {
        assert!(cursor.is_string());
        let (status, next) = es.post("/_sql", json!({"cursor": cursor})).await;
        assert_eq!(status, 200, "{next}");
        assert_eq!(next["rows"].as_array().unwrap().len(), 2);
        rows.extend(next["rows"].as_array().unwrap().iter().cloned());
        cursor = next["cursor"].clone();
        if page == 1 { assert!(cursor.is_null()); }
    }
    let projected: Vec<Value> = rows.iter().map(|r| json!([r[name], r[salary]])).collect();
    assert_eq!(projected, vec![json!(["Eve Black", 50]), json!(["Carol White", 60]),
        json!(["Dan Brown", 70]), json!(["Bob Jones", 80]),
        json!(["Alice Smith", 100]), json!(["Frank Green", 120])]);
}

#[tokio::test]
async fn sql_plain_star_streams_snapshot_across_shards_and_honors_limit() {
    use elasticsearch_engine::search_ext::{self, SqlRows};
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let es = start_node(node.clone()).await;
    let (status, body) = es.put("/scan", json!({"settings":{"number_of_shards":2},
        "mappings":{"properties":{"value":{"type":"long"}}}})).await;
    assert_eq!(status,200,"{body}");
    let mut lines = Vec::new();
    for i in 0..12 { lines.push(json!({"index":{"_index":"scan","_id":i.to_string()}})); lines.push(json!({"value":i})); }
    es.bulk(&lines).await;
    let (status, first) = es.post("/_sql",json!({"query":"SELECT * FROM scan","fetch_size":3})).await;
    assert_eq!(status,200,"{first}");
    let state = search_ext::ext(&node);
    {
        let cursors = state.sql_cursors.lock();
        let stored = cursors.get(first["cursor"].as_str().unwrap()).unwrap();
        assert!(matches!(&stored.rows, SqlRows::Scan(_)));
        assert_eq!(stored.rows.len(),9);
    }
    es.bulk(&[json!({"index":{"_index":"scan","_id":"later"}}),json!({"value":999})]).await;
    let mut rows = first["rows"].as_array().unwrap().clone();
    let mut cursor = first["cursor"].clone();
    while cursor.is_string() {
        let (status, next) = es.post("/_sql",json!({"cursor":cursor})).await;
        assert_eq!(status,200,"{next}");
        rows.extend(next["rows"].as_array().unwrap().iter().cloned());
        cursor = next["cursor"].clone();
        assert!(rows.len()<=12,"snapshot must exclude subsequently inserted rows");
    }
    let mut values: Vec<i64> = rows.iter().map(|r|r[0].as_i64().unwrap()).collect();
    values.sort_unstable();assert_eq!(values,(0..12).collect::<Vec<_>>());
    assert!(state.sql_cursors.lock().is_empty());
    let (status, first) = es.post("/_sql",json!({"query":"SELECT * FROM scan LIMIT 7","fetch_size":4})).await;
    assert_eq!(status,200,"{first}");assert_eq!(first["rows"].as_array().unwrap().len(),4);
    let (status, next) = es.post("/_sql",json!({"cursor":first["cursor"]})).await;
    assert_eq!(status,200,"{next}");assert_eq!(next["rows"].as_array().unwrap().len(),3);
    assert!(next["cursor"].is_null());
}

#[tokio::test]
async fn sql_mapped_float_metadata_survives_alias_but_not_computation() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (status, body) = es.put("/floats", json!({"mappings":{"properties":{
        "f":{"type":"float"}, "d":{"type":"double"}
    }}})).await;
    assert_eq!(status, 200, "{body}");
    es.bulk(&[json!({"index":{"_index":"floats"}}),json!({"f":1.5,"d":2.5})]).await;
    let (status, body) = es.post("/_sql", json!({"query":"SELECT f, f AS alias, f + d AS computed FROM floats"})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["columns"],json!([{"name":"f","type":"float"},{"name":"alias","type":"float"},{"name":"computed","type":"double"}]));
    assert_eq!(body["rows"],json!([[1.5,1.5,4.0]]));
    let (status, body) = es.post("/_sql",json!({"query":"SELECT f + d AS f FROM floats"})).await;
    assert_eq!(status,200,"{body}");
    assert_eq!(body["columns"][0]["type"],"double");
}

#[tokio::test]
async fn sql_star_omits_unsupported_range_columns_across_pages() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (status, body) = es.put("/weather", json!({"mappings":{"properties":{
        "name":{"type":"keyword"}, "range":{"type":"double_range"},
        "value":{"type":"double"}
    }}})).await;
    assert_eq!(status, 200, "{body}");
    es.bulk(&[json!({"index":{"_index":"weather","_id":"1"}}),
        json!({"name":"first","range":{"gte":1,"lte":2},"value":1.5}),
        json!({"index":{"_index":"weather","_id":"2"}}),
        json!({"name":"second","range":{"gte":3,"lte":4},"value":3.5})]).await;
    let (status, first) = es.post("/_sql", json!({"query":"SELECT * FROM weather ORDER BY value", "fetch_size":1})).await;
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["columns"], json!([{"name":"name","type":"keyword"},{"name":"value","type":"double"}]));
    assert_eq!(first["rows"], json!([["first",1.5]]));
    let (status, second) = es.post("/_sql", json!({"cursor":first["cursor"]})).await;
    assert_eq!(status, 200, "{second}");
    assert_eq!(second["rows"], json!([["second",3.5]]));
    assert!(second["cursor"].is_null());
    // Explicit access must still report the unsupported type, not silently omit it.
    let (status, body) = es.post("/_sql", json!({"query":"SELECT range FROM weather"})).await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn sql_cursor_deadline_renewal_expiry_and_close() {
    use elasticsearch_engine::search_ext;
    use std::time::{Duration, Instant};
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let es = start_node(node.clone()).await;
    seed(&es).await;
    let (status, first) = es.post("/_sql", json!({"query": "SELECT * FROM emp ORDER BY salary", "fetch_size": 1, "page_timeout": "1m"})).await;
    assert_eq!(status, 200, "{first}");
    let state = search_ext::ext(&node);
    let first_id = first["cursor"].as_str().unwrap();
    {
        let mut cursors = state.sql_cursors.lock();
        let cursor = cursors.get_mut(first_id).unwrap();
        assert_eq!(cursor.rows.len(), 5);
        cursor.expires = Instant::now() + Duration::from_secs(10);
    }
    let (status, next) = es.post("/_sql", json!({"cursor": first_id, "page_timeout": "1m"})).await;
    assert_eq!(status, 200, "{next}");
    let next_id = next["cursor"].as_str().unwrap();
    {
        let mut cursors = state.sql_cursors.lock();
        assert!(!cursors.contains_key(first_id));
        let cursor = cursors.get_mut(next_id).unwrap();
        assert_eq!(cursor.rows.len(), 4);
        assert!(cursor.expires > Instant::now() + Duration::from_secs(30));
        // Deterministic expiry, without depending on a scheduler or sleep.
        cursor.expires = Instant::now() - Duration::from_secs(1);
    }
    let (status, expired) = es.post("/_sql", json!({"cursor": next_id})).await;
    assert_eq!(status, 404, "{expired}");
    assert_eq!(expired["error"]["type"], "search_phase_execution_exception", "{expired}");
    assert!(state.sql_cursors.lock().is_empty());
    let (status, first) = es.post("/_sql", json!({"query": "SELECT * FROM emp", "fetch_size": 1})).await;
    assert_eq!(status, 200, "{first}");
    let (status, closed) = es.post("/_sql/close", json!({"cursor": first["cursor"]})).await;
    assert_eq!(status, 200, "{closed}");
    assert_eq!(closed["succeeded"], true);
    assert!(state.sql_cursors.lock().is_empty());
}

#[tokio::test]
async fn sql_page_timeout_parser_matches_es9_types() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    for timeout in [Value::Null, json!(100), json!(true), json!([]), json!({}), json!("invalid"), json!("-2ms"), json!("1.5s"), json!("2")] {
        let (status, value) = es.post("/_sql", json!({"query": "SELECT 1", "page_timeout": timeout})).await;
        assert_eq!(status, 400, "{value}");
        assert_eq!(value["error"]["type"], "x_content_parse_exception", "{value}");
    }
    for timeout in ["0", "-1", "0ms", "-1ms", "1m"] {
        let (status, value) = es.post("/_sql", json!({"query": "SELECT 1", "page_timeout": timeout})).await;
        assert_eq!(status, 200, "{value}");
    }
}

#[tokio::test]
async fn eql_event_and_sequence() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es
        .put("/ev", json!({"mappings": {"properties": {"@timestamp": {"type": "date"}, "event": {"properties": {"category": {"type": "keyword"}}}, "process": {"properties": {"name": {"type": "keyword"}, "pid": {"type": "long"}}}, "user": {"type": "keyword"}}}}))
        .await;
    assert_eq!(st, 200, "{v}");
    let evs = [
        ("1", "2024-01-01T00:00:01Z", "process", "cmd.exe", 1, "alice"),
        ("2", "2024-01-01T00:00:02Z", "file", "cmd.exe", 1, "alice"),
        ("3", "2024-01-01T00:00:03Z", "process", "powershell.exe", 2, "bob"),
        ("4", "2024-01-01T00:00:04Z", "network", "cmd.exe", 1, "alice"),
        ("5", "2024-01-01T00:00:05Z", "process", "explorer.exe", 3, "bob"),
    ];
    let mut lines = Vec::new();
    for (id, ts, cat, name, pid, user) in evs {
        lines.push(json!({"index": {"_index": "ev", "_id": id}}));
        lines.push(json!({"@timestamp": ts, "event": {"category": cat}, "process": {"name": name, "pid": pid}, "user": user}));
    }
    es.bulk(&lines).await;
    let (st, v) = es.post("/ev/_eql/search", json!({"query": "process where process.name == \"cmd.exe\""})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["is_partial"], false);
    assert_eq!(v["is_running"], false);
    assert_eq!(v["timed_out"], false);
    assert_eq!(v["hits"]["total"], json!({"value": 1, "relation": "eq"}));
    assert_eq!(v["hits"]["events"][0]["_id"], "1");
    assert_eq!(v["hits"]["events"][0]["_index"], "ev");
    let (st, v) = es.post("/ev/_eql/search", json!({"query": "any where user == \"bob\" and process.pid > 2", "size": 5})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["events"].as_array().unwrap().iter().map(|e| e["_id"].clone()).collect::<Vec<_>>(), vec![json!("5")]);
    let (st, v) = es.post("/ev/_eql/search", json!({"query": "process where process.name : \"*.EXE\" and not process.name in (\"cmd.exe\")"})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["events"].as_array().unwrap().iter().map(|e| e["_id"].clone()).collect::<Vec<_>>(), vec![json!("3"), json!("5")]);
    // sequence by
    let (st, v) = es.post("/ev/_eql/search", json!({"query": "sequence by user [process where true] [network where true]"})).await;
    assert_eq!(st, 200, "{v}");
    let seqs = v["hits"]["sequences"].as_array().unwrap();
    assert_eq!(seqs.len(), 1, "{v}");
    assert_eq!(seqs[0]["join_keys"], json!(["alice"]));
    assert_eq!(seqs[0]["events"].as_array().unwrap().iter().map(|e| e["_id"].clone()).collect::<Vec<_>>(), vec![json!("1"), json!("4")]);
    // errors
    let (st, v) = es.post("/ev/_eql/search", json!({"query": "process where"})).await;
    assert_eq!(st, 400, "{v}");
}

#[tokio::test]
async fn eql_boolean_function_comparisons() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/predicates", json!({"mappings": {"properties": {
        "@timestamp": {"type": "date"}, "name": {"type": "keyword"},
        "prefix": {"type": "keyword"}
    }}})).await;
    assert_eq!(st, 200, "{v}");
    let mut lines = Vec::new();
    for (id, name) in [("1", json!("ssh")), ("2", json!("bash")), ("3", Value::Null)] {
        lines.push(json!({"index": {"_index": "predicates", "_id": id}}));
        lines.push(json!({"@timestamp": format!("2024-01-01T00:00:0{id}Z"), "name": name, "prefix": "ss"}));
    }
    es.bulk(&lines).await;
    for function in ["startsWith(name, \"ss\")", "startsWith(name, prefix)",
                     "startsWith(concat(name, \" \"), prefix)", "startsWith(substring(name, 0, 2), \"ss\")",
                     "endsWith(name, \"sh\")", "stringContains(name, \"s\")"] {
        let positive = if function.starts_with("startsWith") { vec![json!("1")] } else { vec![json!("1"), json!("2")] };
        let mut negative = if function.starts_with("startsWith") { vec![json!("2")] } else { vec![] };
        // A function of a field and literals is a query: ES negates it as
        // `bool.must_not`, which also matches the doc without `name` (3);
        // over an expression or with a field argument it is a null-safe
        // script, which does not.
        if ["startsWith(name, \"ss\")", "endsWith(name, \"sh\")", "stringContains(name, \"s\")"].contains(&function) {
            negative.push(json!("3"));
        }
        for (suffix, expected) in [("== true", positive.clone()), ("!= false", positive),
                                   ("== false", negative.clone()), ("!= true", negative),
                                   ("== null", vec![json!("3")]), ("!= null", vec![json!("1"), json!("2")])] {
            let query = format!("any where {function} {suffix}");
            let (st, v) = es.post("/predicates/_eql/search", json!({"query": query})).await;
            assert_eq!(st, 200, "{query}: {v}");
            let ids: Vec<_> = v["hits"]["events"].as_array().unwrap().iter().map(|e| e["_id"].clone()).collect();
            assert_eq!(ids, expected, "{query}: {v}");
        }
    }
    let (st, v) = es.post("/predicates/_eql/search", json!({"query":
        "sequence with maxspan=30m [any where startsWith(name, \"ss\") == true] [any where startsWith(name, \"ss\") == false] | tail 100 | head 50"
    })).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["sequences"].as_array().unwrap().len(), 1, "{v}");
}

#[tokio::test]
async fn eql_optional_join_keys() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/optional-keys", json!({"mappings": {"properties": {
        "@timestamp": {"type": "date"}, "source": {"properties": {"ip": {"type": "ip"}}}
    }}})).await;
    assert_eq!(st, 200, "{v}");
    let mut lines = Vec::new();
    for id in 1..=4 {
        lines.push(json!({"index": {"_index": "optional-keys", "_id": id.to_string()}}));
        let mut doc = json!({"@timestamp": format!("2024-01-01T00:00:0{id}Z")});
        if id > 2 { doc["source"] = json!({"ip": "10.0.0.1"}); }
        lines.push(doc);
    }
    es.bulk(&lines).await;
    for kind in ["sequence", "sample"] {
        for (key, count) in [("?source.ip", 2), ("source.ip", 1)] {
            let query = format!("{kind} by {key} [any where true] [any where true]");
            let (st, v) = es.post("/optional-keys/_eql/search", json!({"query": query})).await;
            assert_eq!(st, 200, "{query}: {v}");
            let seqs = v["hits"]["sequences"].as_array().unwrap();
            assert_eq!(seqs.len(), count, "{query}: {v}");
            assert_eq!(seqs.iter().any(|s| s["join_keys"] == json!([null])), key.starts_with('?'), "{v}");
        }
    }
}

#[tokio::test]
async fn eql_ordered_pipe_windows() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/pipe-events", json!({"mappings": {"properties": {
        "@timestamp": {"type": "date"}, "group": {"type": "long"}, "stage": {"type": "long"}
    }}})).await;
    assert_eq!(st, 200, "{v}");
    let mut lines = Vec::new();
    for group in 1..=8 {
        for stage in 1..=2 {
            lines.push(json!({"index": {"_index": "pipe-events", "_id": format!("{group}-{stage}")}}));
            lines.push(json!({"@timestamp": 1704067200000i64 + group * 1000 + stage, "group": group, "stage": stage}));
        }
    }
    es.bulk(&lines).await;
    for mode in 0..3 {
        let sequence = mode != 0;
        let base = match mode {
            0 => "any where stage == 1",
            1 => "sequence by group [any where stage == 1] [any where stage == 2]",
            _ => "sequence by group [any where true] [any where true]",
        };
        for (pipes, size, expected) in [
            (" | head 6 | tail 2", 8, vec![5, 6]),
            (" | tail 6 | head 2", 8, vec![3, 4]),
            (" | head 6 | tail 4 | head 2", 8, vec![3, 4]),
            (" | tail 6 | head 4 | tail 2", 8, vec![5, 6]),
            (" | head 6 | tail 2", 4, vec![3, 4]),
            (" | tail 6 | head 2", 4, vec![5, 6]),
            (" | head 0 | tail 2", 8, vec![]),
            (" | tail 20 | head 2", 20, vec![1, 2]),
        ] {
            for position in ["head", "tail"] {
                let query = format!("{base}{pipes}");
                let (st, v) = es.post("/pipe-events/_eql/search", json!({"query": query, "size": size, "result_position": position})).await;
                assert_eq!(st, 200, "{query}: {v}");
                let actual: Vec<i64> = if sequence {
                    v["hits"]["sequences"].as_array().unwrap().iter().map(|s| s["join_keys"][0].as_i64().unwrap()).collect()
                } else {
                    v["hits"]["events"].as_array().unwrap().iter().map(|e| e["_source"]["group"].as_i64().unwrap()).collect()
                };
                assert_eq!(actual, expected, "{query}, size={size}, position={position}: {v}");
            }
        }
    }
}

#[tokio::test]
async fn search_templates() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es
        .post(
            "/_render/template",
            json!({"source": "{\"query\": {\"terms\": {\"dept\": {{#toJson}}depts{{/toJson}} }}, \"size\": {{size}}{{^size}}10{{/size}} }", "params": {"depts": ["eng", "hr"]}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["template_output"], json!({"query": {"terms": {"dept": ["eng", "hr"]}}, "size": 10}));
    // Object source + inline search.
    let (st, v) = es.post("/emp/_search/template", json!({"source": {"query": {"match": {"dept": "{{d}}"}}}, "params": {"d": "sales"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 2);
    // Stored template.
    let (st, v) = es.put("/_scripts/by_dept", json!({"script": {"lang": "mustache", "source": {"query": {"term": {"dept": "{{dept}}"}}}}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post("/emp/_search/template", json!({"id": "by_dept", "params": {"dept": "eng"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 3);
    let (st, v) = es.post("/_render/template/by_dept", json!({"params": {"dept": "hr"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["template_output"], json!({"query": {"term": {"dept": "hr"}}}));
    // msearch template
    let body = format!(
        "{}\n{}\n{}\n{}\n",
        json!({"index": "emp"}),
        json!({"id": "by_dept", "params": {"dept": "eng"}}),
        json!({}),
        json!({"source": "{\"query\":{\"term\":{\"dept\":\"{{d}}\"}}}", "params": {"d": "hr"}})
    );
    let (st, _, text) = es.raw(Method::POST, "/emp/_msearch/template", &body, "application/x-ndjson").await;
    assert_eq!(st, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["responses"][0]["hits"]["total"]["value"], 3);
    assert_eq!(v["responses"][1]["hits"]["total"]["value"], 1);
    assert_eq!(v["responses"][1]["status"], 200);
    // Missing stored template.
    let (st, v) = es.post("/emp/_search/template", json!({"id": "nope"})).await;
    assert_eq!(st, 404, "{v}");
}

#[tokio::test]
async fn rank_eval_metrics() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let req = |metric: Value| {
        json!({
            "requests": [{
                "id": "eng_query",
                "request": {"query": {"term": {"dept": "eng"}}, "sort": [{"salary": "desc"}]},
                "ratings": [{"_index": "emp", "_id": "6", "rating": 1}, {"_index": "emp", "_id": "2", "rating": 1}, {"_index": "emp", "_id": "3", "rating": 1}]
            }],
            "metric": metric
        })
    };
    let (st, v) = es.post("/emp/_rank_eval", req(json!({"precision": {"k": 3, "relevant_rating_threshold": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    // hits: 6 (rated 1), 1 (unrated), 2 (rated 1) → 2/3
    let p = v["metric_score"].as_f64().unwrap();
    assert!((p - 2.0 / 3.0).abs() < 1e-9, "{v}");
    let d = &v["details"]["eng_query"];
    assert_eq!(d["unrated_docs"], json!([{"_index": "emp", "_id": "1"}]));
    assert_eq!(d["metric_details"]["precision"], json!({"relevant_docs_retrieved": 2, "docs_retrieved": 3}));
    assert_eq!(v["failures"], json!({}));
    let (_, v) = es.post("/emp/_rank_eval", req(json!({"recall": {"k": 3}}))).await;
    assert!((v["metric_score"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-9, "{v}");
    let (_, v) = es.post("/emp/_rank_eval", req(json!({"mean_reciprocal_rank": {"k": 3}}))).await;
    assert_eq!(v["metric_score"], 1.0);
    assert_eq!(v["details"]["eng_query"]["metric_details"]["mean_reciprocal_rank"]["first_relevant"], 1);
    let (_, v) = es.post("/emp/_rank_eval", req(json!({"dcg": {"k": 3, "normalize": true}}))).await;
    let ndcg = v["metric_score"].as_f64().unwrap();
    assert!(ndcg > 0.0 && ndcg < 1.0, "{v}");
    let (_, v) = es.post("/emp/_rank_eval", req(json!({"expected_reciprocal_rank": {"k": 3, "maximum_relevance": 1}}))).await;
    assert!(v["metric_score"].as_f64().unwrap() > 0.0, "{v}");
}

#[tokio::test]
async fn synonyms_api_and_reload() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, v) = es.put("/_synonyms/my-set", json!({"synonyms_set": [{"id": "r1", "synonyms": "hello, hi"}, {"synonyms": "bye => goodbye"}]})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(v["result"], "created");
    assert_eq!(v["reload_analyzers_details"]["_shards"]["total"], 0);
    let (st, v) = es.get("/_synonyms/my-set").await;
    assert_eq!(st, 200);
    assert_eq!(v["count"], 2);
    assert_eq!(v["synonyms_set"].as_array().unwrap().iter().find(|r| r["id"] == "r1").unwrap()["synonyms"], "hello, hi");
    let (st, v) = es.get("/_synonyms").await;
    assert_eq!(st, 200);
    assert_eq!(v, json!({"count": 1, "results": [{"synonyms_set": "my-set", "count": 2}]}));
    // Index using the set as a search analyzer.
    let (st, v) = es
        .put(
            "/syn",
            json!({"settings": {"analysis": {"filter": {"syn": {"type": "synonym_graph", "synonyms_set": "my-set", "updateable": true}}, "analyzer": {"my_an": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "syn"]}}}},
                   "mappings": {"properties": {"f": {"type": "text", "search_analyzer": "my_an"}}}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    es.bulk(&[json!({"index": {"_index": "syn", "_id": "1"}}), json!({"f": "hello"}), json!({"index": {"_index": "syn", "_id": "2"}}), json!({"f": "salute"})]).await;
    let (_, v) = es.post("/syn/_search", json!({"query": {"match": {"f": "hi"}}})).await;
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");
    // Add a rule: reloads the index's analyzers.
    let (st, v) = es.put("/_synonyms/my-set/r3", json!({"synonyms": "greetings, salute"})).await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(v["result"], "created");
    assert_eq!(v["reload_analyzers_details"]["reload_details"][0]["index"], "syn");
    assert_eq!(v["reload_analyzers_details"]["reload_details"][0]["reloaded_analyzers"], json!(["my_an"]));
    let (_, v) = es.post("/syn/_search", json!({"query": {"match": {"f": "greetings"}}})).await;
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");
    let (st, v) = es.get("/_synonyms/my-set/r3").await;
    assert_eq!(st, 200);
    assert_eq!(v, json!({"id": "r3", "synonyms": "greetings, salute"}));
    // Validation, in-use deletion, rule deletion.
    let (st, _) = es.put("/_synonyms/bad", json!({"synonyms_set": [{"synonyms": "a => b => c"}]})).await;
    assert_eq!(st, 400);
    let (st, v) = es.delete("/_synonyms/my-set").await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("cannot be deleted as it is used in the following indices: syn"));
    let (st, v) = es.delete("/_synonyms/my-set/r3").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["result"], "deleted");
    let (st, _) = es.get("/_synonyms/missing").await;
    assert_eq!(st, 404);
}

#[tokio::test]
async fn query_rules_and_search_application() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    seed(&es).await;
    let (st, v) = es
        .put(
            "/_query_rules/rs1",
            json!({"rules": [
                {"rule_id": "pin_eve", "type": "pinned", "criteria": [{"type": "exact", "metadata": "q", "values": ["eng"]}], "actions": {"ids": ["5"]}},
                {"rule_id": "hide_bob", "type": "exclude", "criteria": [{"type": "prefix", "metadata": "q", "values": ["en"]}], "actions": {"docs": [{"_index": "emp", "_id": "2"}]}}
            ]}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["result"], "created");
    let (st, v) = es.post("/_query_rules/rs1/_test", json!({"match_criteria": {"q": "eng"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["total_matched_rules"], 2);
    let (st, v) = es
        .post("/emp/_search", json!({"query": {"rule": {"organic": {"term": {"dept": "eng"}}, "ruleset_ids": ["rs1"], "match_criteria": {"q": "eng"}}}}))
        .await;
    assert_eq!(st, 200, "{v}");
    let ids: Vec<Value> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].clone()).collect();
    assert_eq!(ids[0], "5", "{v}");
    assert!(!ids.contains(&json!("2")), "{v}");
    assert_eq!(ids.len(), 3, "{v}");
    let (_, v) = es.get("/_query_rules").await;
    assert_eq!(v["count"], 1);
    assert_eq!(v["results"][0]["rule_total_count"], 2);
    let (st, v) = es.get("/_query_rules/rs1/_rule/pin_eve").await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["type"], "pinned");
    let (st, _) = es.delete("/_query_rules/rs1/_rule/pin_eve").await;
    assert_eq!(st, 200);
    let (st, _) = es.delete("/_query_rules/rs1").await;
    assert_eq!(st, 200);
    let (st, _) = es.get("/_query_rules/rs1").await;
    assert_eq!(st, 404);

    // Search application.
    let (st, v) = es
        .put(
            "/_application/search_application/app1",
            json!({"indices": ["emp"], "template": {"script": {"source": "{\"query\": {\"term\": {\"dept\": \"{{dept}}\"}}, \"size\": {{size}}}", "params": {"dept": "eng", "size": 10}}}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["result"], "created");
    let (st, v) = es.post("/_application/search_application/app1/_search", json!({"params": {"dept": "hr"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 1);
    let (st, v) = es.post("/_application/search_application/app1/_render_query", json!({})).await;
    assert_eq!(st, 200, "{v}");
    // Rendered the way ES re-serialises the parsed query (`term` in object form).
    assert_eq!(v, json!({"query": {"term": {"dept": {"value": "eng"}}}, "size": 10}));
    let (_, v) = es.get("/_application/search_application").await;
    assert_eq!(v["count"], 1);
    assert_eq!(v["results"][0]["name"], "app1");
    let (st, _) = es.delete("/_application/search_application/app1").await;
    assert_eq!(st, 200);
    let (st, _) = es.get("/_application/search_application/app1").await;
    assert_eq!(st, 404);

    // Native feature reset removes the system stores and their catalog state,
    // while preserving the ordinary indices used by applications.
    assert_eq!(es.put("/_query_rules/reset-rules", json!({"rules": [{"rule_id":"pin", "type":"pinned", "criteria":[{"type":"exact", "metadata":"q", "values":["eng"]}], "actions":{"ids":["1"]}}]})).await.0, 200);
    assert_eq!(es.put("/_application/search_application/reset-app", json!({"indices":["emp"]})).await.0, 200);
    for index in [".search-app-1", ".query-rules-1", ".connector-secrets-1"] {
        assert_eq!(es.put(&format!("/{index}"), json!({"settings":{"index.hidden":true}})).await.0, 200);
    }
    let (status, reset) = es.call(Method::POST, "/_features/_reset", None).await;
    assert_eq!(status, 200, "{reset}");
    assert!(reset["features"].as_array().unwrap().iter().all(|feature| feature["status"] == "SUCCESS"), "{reset}");
    assert_eq!(es.get("/_query_rules").await.1["count"], 0);
    assert_eq!(es.get("/_application/search_application").await.1["count"], 0);
    assert_eq!(es.get("/emp/_count").await.1["count"], 6);
    for index in [".search-app-1", ".query-rules-1", ".connector-secrets-1"] {
        assert_eq!(es.get(&format!("/{index}")).await.0, 404);
    }
}

#[tokio::test]
async fn sql_having_rally_script_aggregate_alias() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    let (status, body) = es.put("/weather", json!({"mappings":{"properties":{
        "station":{"properties":{"name":{"type":"keyword"},"elevation":{"type":"integer"}}},
        "TMIN":{"type":"integer"}
    }}})).await;
    assert_eq!(status, 200, "{body}");
    let mut docs = Vec::new();
    for i in 0..302 {
        docs.push(json!({"index":{"_index":"weather","_id":i.to_string()}}));
        docs.push(json!({"station":{"name":if i < 301 {"alpha"} else {"beta"},"elevation":3000},"TMIN":2}));
    }
    es.bulk(&docs).await;
    let (status, body) = es.post("/_sql", json!({"query":
        "SELECT station.name, SUM(LENGTH(SUBSTRING(station.name, 0, 5))) S FROM weather WHERE station.elevation > 2800 GROUP BY 1 HAVING S > 1000"
    })).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["columns"][1]["name"], "S");
    assert_eq!(body["rows"], json!([["alpha",1505]]));
}

#[tokio::test]
async fn sql_pivot_rally_average_and_paged_counts() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    let (status, body) = es.put("/weather", json!({"mappings":{"properties":{
        "station":{"properties":{"name":{"type":"keyword"},"country_code":{"type":"keyword"},"elevation":{"type":"integer"}}},
        "TMIN":{"type":"integer"}
    }}})).await;
    assert_eq!(status, 200, "{body}");
    let mut docs = Vec::new();
    for n in 0..9 {
        for (country, value) in [("US", 100), ("US", 200), ("AS", 6), ("CA", 10)] {
            docs.push(json!({"index":{"_index":"weather"}}));
            docs.push(json!({"station":{"name":format!("station{n}"),"country_code":country,"elevation":3000},"TMIN":value}));
        }
    }
    es.bulk(&docs).await;
    let (status, body) = es.post("/_sql", json!({"query":
        "SELECT * FROM (SELECT station.country_code, TMIN FROM weather) PIVOT (AVG(TMIN) FOR station.country_code IN ('AS', 'CA'))"
    })).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["columns"].as_array().unwrap().iter().map(|c|c["name"].as_str().unwrap()).collect::<Vec<_>>(), vec!["AS","CA"]);
    assert_eq!(body["rows"], json!([[6.0,10.0]]));
    let query = "SELECT * FROM (SELECT station.name, station.country_code FROM weather WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS')) ORDER BY station.name";
    let (status, mut body) = es.post("/_sql", json!({"query":query,"fetch_size":3})).await;
    assert_eq!(status, 200, "{body}");
    let mut rows = Vec::new();
    loop {
        rows.extend(body["rows"].as_array().unwrap().iter().cloned());
        let Some(cursor) = body.get("cursor").cloned() else {break};
        let (status, next) = es.post("/_sql",json!({"cursor":cursor})).await;
        assert_eq!(status,200,"{next}");body=next;
    }
    assert_eq!(rows,(0..9).map(|n|json!([format!("station{n}"),2,1])).collect::<Vec<_>>());
    let (status, body) = es.post("/_sql",json!({"query":"SELECT country, missing FROM (SELECT station.country_code AS code, TMIN FROM weather) AS src PIVOT (AVG(TMIN) FOR code IN ('CA' AS country, 'ZZ' AS missing))"})).await;
    assert_eq!(status,200,"{body}");assert_eq!(body["rows"],json!([[10.0,null]]));
    let (status, body) = es.post("/_sql",json!({"query":"SELECT * FROM (SELECT station.country_code FROM weather) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'ZZ'))"})).await;
    assert_eq!(status,200,"{body}");assert_eq!(body["rows"],json!([[18,null]]));
    let (status, body) = es.post("/_sql",json!({"query":"SELECT * FROM (SELECT TMIN FROM weather) PIVOT (COUNT(*) FOR TMIN IN (6 AS six, 10 AS ten))"})).await;
    assert_eq!(status,200,"{body}");assert_eq!(body["rows"],json!([[9,9]]));
    for query in [
        "SELECT * FROM (SELECT station.country_code FROM weather) PIVOT (COUNT(*) FOR station.country_code IN ('ZZ'))",
        "SELECT * FROM (SELECT station.country_code, TMIN FROM weather WHERE TMIN < 0) PIVOT (AVG(TMIN) FOR station.country_code IN ('US'))"
    ] {
        let (status, body) = es.post("/_sql",json!({"query":query})).await;
        assert_eq!(status,200,"{body}");assert_eq!(body["rows"],json!([]));
    }
    let (status, _) = es.post("/_sql",json!({"query":"SELECT * FROM weather PIVOT (TMIN FOR station.country_code IN ('US'))"})).await;
    assert_eq!(status,400);
}

#[tokio::test]
async fn sql_string_filter_streams_candidates_and_preserves_cursor() {
    use elasticsearch_engine::search_ext::{self, SqlRows};
    let tmp=tempfile::tempdir().unwrap();
    let node=Node::open(tmp.path()).unwrap();
    let es=start_node(node.clone()).await;
    let (status,body)=es.put("/weather-filter",json!({"settings":{"number_of_shards":2},"mappings":{"properties":{
        "station":{"properties":{"name":{"type":"keyword"},"elevation":{"type":"double"}}}
    }}})).await;
    assert_eq!(status,200,"{body}");
    let mut lines=Vec::new();
    for i in 0..300 {
        lines.push(json!({"index":{"_index":"weather-filter","_id":i.to_string()}}));
        lines.push(json!({"station":{"name":if i%3==0 {"SANTA"}else{"OTHER"},"elevation":if i%2==0 {3000}else{1000}}}));
    }
    es.bulk(&lines).await;
    let query="SELECT station.name FROM \"weather-filter\" WHERE LCASE(SUBSTRING(station.name, 0, 2)) = 'sa' AND station.elevation > 2800";
    let (status,first)=es.post("/_sql",json!({"query":query,"fetch_size":7})).await;
    assert_eq!(status,200,"{first}");
    assert_eq!(first["rows"].as_array().unwrap().len(),7);
    let state=search_ext::ext(&node);
    assert!(matches!(&state.sql_cursors.lock().get(first["cursor"].as_str().unwrap()).unwrap().rows,SqlRows::Scan(_)));
    // The cursor must keep the original snapshot and evaluate all later pages.
    es.bulk(&[json!({"index":{"_index":"weather-filter","_id":"later"}}),json!({"station":{"name":"SANTA","elevation":5000}})]).await;
    let mut body=first;
    let mut count=0;
    loop {
        for row in body["rows"].as_array().unwrap() { assert_eq!(row,&json!(["SANTA"])); count+=1; }
        let Some(cursor)=body.get("cursor").cloned() else {break};
        let (status,next)=es.post("/_sql",json!({"cursor":cursor})).await;
        assert_eq!(status,200,"{next}");body=next;
        assert!(count<=50);
    }
    assert_eq!(count,50);
    let (status,limited)=es.post("/_sql",json!({"query":format!("{query} LIMIT 10"),"fetch_size":20})).await;
    assert_eq!(status,200,"{limited}");assert_eq!(limited["rows"].as_array().unwrap().len(),10);
    assert!(limited.get("cursor").is_none());
    let (status,none)=es.post("/_sql",json!({"query":query.replace("'sa'","'zz'"),"fetch_size":7})).await;
    assert_eq!(status,200,"{none}");assert_eq!(none["rows"],json!([]));assert!(none.get("cursor").is_none());
}

#[tokio::test]
async fn sql_pivot_streams_multiple_batches_and_shards() {
    let dir=tempfile::tempdir().unwrap();
    let es=start(dir.path()).await;
    let (status, body)=es.put("/pivot-batches",json!({"settings":{"number_of_shards":2},"mappings":{"properties":{
        "g":{"type":"keyword"},"country":{"type":"keyword"},"v":{"type":"double"},"unused":{"type":"keyword","index":false}
    }}})).await;
    assert_eq!(status,200,"{body}");
    let mut lines=Vec::new();
    for i in 0..4097 {
        lines.push(json!({"index":{"_index":"pivot-batches","_id":i.to_string()}}));
        lines.push(json!({"g":"one","country":if i%3==0 {"A"}else{"B"},"v":if i%7==0 {serde_json::Value::Null}else{json!(i as f64)},"unused":"x".repeat(2048)}));
        if lines.len()==512 { es.bulk(&lines).await; lines.clear(); }
    }
    if !lines.is_empty() { es.bulk(&lines).await; }
    for aggregate in ["AVG(v)","SUM(v)","MIN(v)","MAX(v)","COUNT(v)","COUNT(*)"] {
        let query=format!("SELECT * FROM (SELECT g, country, v FROM \"pivot-batches\") PIVOT ({aggregate} FOR country IN ('A', 'B', 'missing')) ORDER BY g");
        // COUNT(*) implicitly groups by v too; omit v for that shape.
        let query=if aggregate=="COUNT(*)" { query.replace("g, country, v","g, country") } else {query};
        let (status, body)=es.post("/_sql",json!({"query":query,"fetch_size":1})).await;
        assert_eq!(status,200,"{aggregate}: {body}");
        let mut expected=vec![json!("one")];
        for a in [true,false] {
            let numbers:Vec<f64>=(0..4097).filter(|i| (*i%3==0)==a && (aggregate=="COUNT(*)" || *i%7!=0)).map(|i|i as f64).collect();
            let v=match aggregate {
                "AVG(v)" => json!(numbers.iter().sum::<f64>()/numbers.len() as f64),
                "SUM(v)" => json!(numbers.iter().sum::<f64>()),
                "MIN(v)" => json!(numbers.first().unwrap()),
                "MAX(v)" => json!(numbers.last().unwrap()),
                _ => json!(numbers.len()),
            };
            expected.push(v);
        }
        expected.push(serde_json::Value::Null);
        assert_eq!(body["rows"],json!([expected]),"{aggregate}: {body}");
    }
    let (status, body)=es.put("/_cluster/settings",json!({"transient":{"search.max_buckets":2}})).await;
    assert_eq!(status,200,"{body}");
    let query="SELECT * FROM (SELECT g,country,v FROM \"pivot-batches\") PIVOT (AVG(v) FOR country IN ('A','B','missing'))";
    let (status, body)=es.post("/_sql",json!({"query":query})).await;
    assert_eq!(status,429,"{body}");
    assert_eq!(body["error"]["type"],"circuit_breaking_exception");
    let (status, body)=es.get("/_cluster/health").await;
    assert_eq!(status,200,"{body}");

}

#[tokio::test]
async fn query_rules_dynamic_limit() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let rule = |id: usize| json!({"rule_id": id.to_string(), "type": "pinned",
        "criteria": [{"type": "exact", "metadata": "q", "values": ["test"]}],
        "actions": {"ids": ["1"]}});
    let rules: Vec<_> = (0..1000).map(rule).collect();
    let (st, _) = es.put("/_query_rules/large", json!({"rules": rules})).await;
    assert_eq!(st, 400);
    let key = "xpack.applications.rules.max_rules_per_ruleset";
    for invalid in [json!(0), json!(1001), json!("abc"), json!([])] {
        let (st, v) = es.put("/_cluster/settings", json!({"persistent": {key: invalid}})).await;
        assert_eq!(st, 400, "{v}");
    }
    let (st, v) = es.put("/_cluster/settings", json!({"persistent": {key: "1000"}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_query_rules/large", json!({"rules": rules})).await;
    assert_eq!(st, 200, "{v}");
    let (st, _) = es.put("/_query_rules/large/_rule/1000", rule(1000)).await;
    assert_eq!(st, 400);
    let (st, v) = es.put("/_query_rules/large/_rule/0", rule(0)).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_cluster/settings", json!({"transient": {key: 2}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_query_rules/small", json!({"rules": [rule(0), rule(1)]})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_query_rules/small/_rule/2", rule(2)).await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("[2]"), "{v}");
    let (st, v) = es.put("/_cluster/settings", json!({"transient": {key: null}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_query_rules/small/_rule/2", rule(2)).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_cluster/settings", json!({"persistent": {key: null}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, _) = es.put("/_query_rules/large", json!({"rules": rules})).await;
    assert_eq!(st, 400);
}

#[tokio::test]
async fn esql_composite_keys_preserve_field_boundaries() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    for (index, mode) in [("key_left", "standard"), ("key_right", "lookup")] {
        let (status, body) = es.put(&format!("/{index}"), json!({
            "settings": {"index.mode": mode, "number_of_shards": 1},
            "mappings": {"properties": {
                "a": {"type":"keyword"}, "b": {"type":"keyword"}, "label": {"type":"keyword"}
            }}
        })).await;
        assert_eq!(status, 200, "{body}");
    }
    // These tuples produce the same concatenated key if a raw separator is used.
    let first = json!({"a":"x\u{2}sy", "b":"z"});
    let second = json!({"a":"x", "b":"y\u{2}sz"});
    es.bulk(&[
        json!({"index":{"_index":"key_left","_id":"1"}}), first.clone(),
        json!({"index":{"_index":"key_left","_id":"2"}}), second.clone(),
        json!({"index":{"_index":"key_right","_id":"1"}}), json!({"a":first["a"],"b":first["b"],"label":"first"}),
        json!({"index":{"_index":"key_right","_id":"2"}}), json!({"a":second["a"],"b":second["b"],"label":"second"}),
    ]).await;
    let (status, body) = es.post("/_query", json!({"query":
        "FROM key_left | LOOKUP JOIN key_right ON a, b | STATS n = COUNT(*) BY label | SORT label"
    })).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["values"], json!([[1,"first"],[1,"second"]]));
    let (status, body) = es.post("/_query", json!({"query":
        "FROM key_left | STATS n = COUNT(*) BY a, b | SORT a"
    })).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["values"], json!([[1,second["a"],second["b"]],[1,first["a"],first["b"]]]));
    let (status, body) = es.post("/_query", json!({"query":
        "FROM key_left | LIMIT 1 BY a, b | STATS n = COUNT(*)"
    })).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["values"], json!([[2]]));
    for (pipeline, count) in [
        ("LIMIT 1 BY a, b | LIMIT 1 | STATS n = COUNT(*)", 1),
        ("LIMIT 0 BY a, b | LIMIT 1 | STATS n = COUNT(*)", 0),
        ("LIMIT 1 BY a, b | WHERE a == \"x\" | LIMIT 1 | STATS n = COUNT(*)", 1),
        ("LIMIT 1 BY a, b | STATS n = COUNT(*) | LIMIT 1", 2),
    ] {
        let (status, body) = es.post("/_query", json!({"query":format!("FROM key_left | {pipeline}")})).await;
        assert_eq!(status, 200, "{pipeline}: {body}");
        assert_eq!(body["values"], json!([[count]]), "{pipeline}: {body}");
    }
}

#[tokio::test]
async fn esql_join_limits_do_not_cross_filters_or_aggregates() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    for (index, mode) in [("bounded_left", "standard"), ("bounded_right", "lookup")] {
        let (status, body) = es.put(&format!("/{index}"), json!({
            "settings":{"index.mode":mode,"number_of_shards":1},
            "mappings":{"properties":{"k":{"type":"keyword"},"ordinal":{"type":"integer"}}}
        })).await;
        assert_eq!(status,200,"{body}");
    }
    let mut docs = Vec::new();
    for index in ["bounded_left", "bounded_right"] {
        for n in 0..120 {
            docs.push(json!({"index":{"_index":index,"_id":n.to_string()}}));
            docs.push(json!({"k":"same","ordinal":n}));
        }
    }
    es.bulk(&docs).await;
    for (pipeline, expected) in [
        ("LOOKUP JOIN bounded_right ON k | LIMIT 1 | STATS n = COUNT(*)", 1),
        ("LOOKUP JOIN bounded_right ON k | STATS n = COUNT(*)", 14400),
        ("LOOKUP JOIN bounded_right ON k | WHERE ordinal == 119 | STATS n = COUNT(*)", 120),
        ("LOOKUP JOIN bounded_right ON k | WHERE ordinal == 119 | LIMIT 1 | STATS n = COUNT(*)", 1),
        ("RENAME k AS left_k | LOOKUP JOIN bounded_right ON left_k == k | LIMIT 1 | STATS n = COUNT(*)", 1),
    ] {
        let (status,body)=es.post("/_query",json!({"query":format!("FROM bounded_left | {pipeline}")})).await;
        assert_eq!(status,200,"{pipeline}: {body}");
        assert_eq!(body["values"],json!([[expected]]),"{pipeline}: {body}");
    }
    let (status,body)=es.post("/_query",json!({"query":"FROM bounded_left | LOOKUP JOIN bounded_right ON k"})).await;
    assert_eq!(status,200,"{body}");
    assert_eq!(body["values"].as_array().unwrap().len(),1000);
    es.bulk(&[
        json!({"index":{"_index":"bounded_left","_id":"missing"}}), json!({"k":"missing","ordinal":-1}),
        json!({"index":{"_index":"bounded_left","_id":"null"}}), json!({"ordinal":-2}),
    ]).await;
    for (filter, expected) in [("",14402),("WHERE ordinal < 0 |",0),("WHERE ordinal IS NULL |",2),("WHERE ordinal == 119 |",120)] {
        let q=format!("FROM bounded_left | LOOKUP JOIN bounded_right ON k | {filter} STATS n = COUNT(*)");
        let (status,body)=es.post("/_query",json!({"query":q})).await;
        assert_eq!(status,200,"{body}");
        assert_eq!(body["values"],json!([[expected]]),"{q}: {body}");
    }
}

#[tokio::test]
async fn sql_computed_projection_pages_without_materializing() {
    use elasticsearch_engine::search_ext::{self, SqlRows};
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let es = start_node(node.clone()).await;
    let (status, body) = es.put("/computed-pages", json!({"settings":{"number_of_shards":2},"mappings":{"properties":{
        "name":{"type":"keyword"}, "n":{"type":"integer"}, "f":{"type":"float"}
    }}})).await;
    assert_eq!(status, 200, "{body}");
    let mut docs = Vec::new();
    for n in 0..2051 {
        docs.push(json!({"index":{"_index":"computed-pages","_id":n.to_string()}}));
        docs.push(json!({"n":n,"name":if n%5==0 {Value::Null}else{json!("SANTA")},"f":1.5}));
    }
    es.bulk(&docs).await;
    for projection in ["n, LCASE(SUBSTRING(name, 0, 2)) AS prefix, f AS copied", "n, CEIL(f) AS f, n + 1 AS next"] {
        let query = format!("SELECT {projection} FROM \"computed-pages\" WHERE n >= 0");
        let reference = elasticsearch_engine::sql::execute(&node, &query, None, None).unwrap();
        let expected_columns: Vec<Value> = reference.cols.iter().map(|c| json!({"name":c.name,"type":c.ty})).collect();
        let mut expected: Vec<Value> = reference.rows.into_iter().map(|r|json!(r.into_iter().map(|v|v.to_json()).collect::<Vec<_>>())).collect();
        let (status, mut body) = es.post("/_sql", json!({"query":query,"fetch_size":137})).await;
        assert_eq!(status,200,"{body}");
        assert_eq!(body["columns"],json!(expected_columns));
        let state = search_ext::ext(&node);
        assert!(matches!(&state.sql_cursors.lock().get(body["cursor"].as_str().unwrap()).unwrap().rows, SqlRows::Scan(_)));
        let mut actual = Vec::new();
        loop {
            actual.extend(body["rows"].as_array().unwrap().iter().cloned());
            let Some(cursor) = body.get("cursor").cloned() else { break };
            let (status, next) = es.post("/_sql",json!({"cursor":cursor})).await;
            assert_eq!(status,200,"{next}"); body=next;
            assert!(actual.len()<=2051);
        }
        actual.sort_by_key(|r|r[0].as_i64()); expected.sort_by_key(|r|r[0].as_i64());
        assert_eq!(actual,expected);
        assert!(state.sql_cursors.lock().is_empty());
    }
    let (status, grouped) = es.post("/_sql", json!({"query":
        "SELECT n % 2 AS parity, COUNT(*) AS count, SUM(n) AS total FROM \"computed-pages\" GROUP BY n % 2 ORDER BY parity"
    })).await;
    assert_eq!(status,200,"{grouped}");
    assert_eq!(grouped["rows"],json!([[0,1026,1051650],[1,1025,1050625]]));
    let (status, body) = es.post("/_sql",json!({"query":"SELECT LCASE(name) AS lowered FROM \"computed-pages\" WHERE n > 3000","fetch_size":3})).await;
    assert_eq!(status,200,"{body}"); assert_eq!(body["rows"],json!([]));
    assert_eq!(body["columns"],json!([{"name":"lowered","type":"keyword"}]));
}

#[tokio::test]
async fn sql_narrow_sources_keep_filter_sort_and_aggregate_dependencies() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (status, body) = es.put("/narrow-sql",json!({"mappings":{"properties":{
        "station":{"properties":{"name":{"type":"keyword"},"elevation":{"type":"integer"}}},
        "n":{"type":"integer"},"unused":{"type":"keyword"}
    }}})).await;
    assert_eq!(status,200,"{body}");
    es.bulk(&[
        json!({"index":{"_index":"narrow-sql"}}),json!({"station":{"name":"B","elevation":3000},"n":3,"unused":"wide"}),
        json!({"index":{"_index":"narrow-sql"}}),json!({"station":{"name":"A","elevation":3000},"n":7,"unused":"wide"}),
        json!({"index":{"_index":"narrow-sql"}}),json!({"station":{"name":"A","elevation":1000},"n":100,"unused":"wide"})
    ]).await;
    for (query, expected) in [
        ("SELECT station.name FROM \"narrow-sql\" WHERE station.elevation > 2000 ORDER BY n DESC",json!([["A"],["B"]])),
        ("SELECT LCASE(station.name) AS name, SUM(n + 1) AS total FROM \"narrow-sql\" WHERE station.elevation > 2000 GROUP BY station.name HAVING total > 4 ORDER BY total",json!([["a",8]])),
        ("SELECT COUNT(*) AS count FROM \"narrow-sql\" WHERE station.elevation > 2000",json!([[2]]))
    ] {
        let (status, body) = es.post("/_sql",json!({"query":query})).await;
        assert_eq!(status,200,"{body}"); assert_eq!(body["rows"],expected,"{query}");
    }
    let (status, body) = es.post("/_sql",json!({"query":"SELECT * FROM \"narrow-sql\" LIMIT 1"})).await;
    assert_eq!(status,200,"{body}");
    assert!(body["columns"].as_array().unwrap().iter().any(|c|c["name"]=="unused"));
}

#[tokio::test]
async fn sql_grouped_count_streams_multivalued_null_keys_and_pages() {
    let tmp=tempfile::tempdir().unwrap();
    let es=start(tmp.path()).await;
    let (status,body)=es.put("/count-stream",json!({"settings":{"number_of_shards":2},"mappings":{"properties":{
        "k":{"type":"keyword"},"n":{"type":"integer"},"unused":{"type":"keyword"}
    }}})).await;
    assert_eq!(status,200,"{body}");
    let mut docs=Vec::new();
    for n in 0..2051 {
        docs.push(json!({"index":{"_index":"count-stream","_id":n.to_string()}}));
        docs.push(json!({"n":n,"k":if n%3==0 {json!(["a","b"])}else if n%3==1 {json!("a")}else{Value::Null},"unused":"wide"}));
    }
    es.bulk(&docs).await;
    let query="SELECT k, COUNT(*) AS c, COUNT(*) AS other FROM \"count-stream\" GROUP BY k HAVING c > 0 ORDER BY k NULLS LAST";
    let (status,mut body)=es.post("/_sql",json!({"query":query,"fetch_size":1})).await;
    assert_eq!(status,200,"{body}");
    let mut rows=Vec::new();
    loop {
        rows.extend(body["rows"].as_array().unwrap().iter().cloned());
        let Some(cursor)=body.get("cursor").cloned() else {break};
        let (status,next)=es.post("/_sql",json!({"cursor":cursor})).await;
        assert_eq!(status,200,"{next}");body=next;
    }
    assert_eq!(rows,vec![json!(["a",1368,1368]),json!(["b",684,684]),json!([null,683,683])]);
    let (status,body)=es.post("/_sql",json!({"query":"SELECT k, COUNT(*) FROM \"count-stream\" WHERE n < 0 GROUP BY k"})).await;
    assert_eq!(status,200,"{body}");assert_eq!(body["rows"],json!([]));
    let (status,body)=es.post("/_sql",json!({"query":"SELECT k, n, COUNT(*) AS c FROM \"count-stream\" WHERE n < 3 GROUP BY k, n ORDER BY n, k NULLS LAST"})).await;
    assert_eq!(status,200,"{body}");
    assert_eq!(body["rows"],json!([["a",0,1],["b",0,1],["a",1,1],[null,2,1]]));
}
