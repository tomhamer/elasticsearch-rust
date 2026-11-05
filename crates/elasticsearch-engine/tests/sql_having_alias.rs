use elasticsearch_engine::{index::WriteReq, node::Node, sql};
use serde_json::json;

#[test]
fn having_select_aliases_execute_with_source_and_aggregate_scopes() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    node.create_index("weather", &json!({"mappings":{"properties":{
        "station":{"properties":{"name":{"type":"keyword"},"elevation":{"type":"integer"}}},
        "TMIN":{"type":"integer"},"S":{"type":"integer"}
    }}})).unwrap();
    let index = node.get_index("weather").unwrap();
    for (id, name, value) in [("1", "alpha", 2), ("2", "alpha", 4), ("3", "beta", 1)] {
        index.write(vec![WriteReq::Index { id: Some(id.into()),
            source: json!({"station":{"name":name,"elevation":3000},"TMIN":value,"S":1}).to_string(),
            routing: None, cond: Default::default() }], false).remove(0).unwrap();
    }
    index.refresh().unwrap();
    let cases = [
        ("SELECT station.name, SUM(LENGTH(SUBSTRING(station.name, 0, 5))) S FROM weather WHERE station.elevation > 2800 GROUP BY 1 HAVING S > 5", 1),
        ("SELECT station.name, SUM(CEIL(TMIN) % 7) S FROM weather WHERE station.elevation > 2800 GROUP BY 1 HAVING S > 0", 2),
        ("SELECT station.name, SUM(TMIN) + 1 AS total FROM weather GROUP BY 1 HAVING ABS(total) BETWEEN 6 AND 8", 1),
        ("SELECT station.name, SUM(TMIN) AS total FROM weather GROUP BY 1 HAVING total IN (6) AND total IS NOT NULL", 1),
        ("SELECT station.name AS label, SUM(TMIN) AS total FROM weather GROUP BY 1 HAVING label = 'alpha'", 1),
        ("SELECT station.name AS label, SUM(TMIN) AS total FROM weather GROUP BY 1 HAVING label LIKE 'al%'", 1),
        ("SELECT SUM(TMIN) AS total FROM weather HAVING total > 6", 1),
        ("SELECT SUM(TMIN) AS total FROM weather HAVING total > 7", 0),
        ("SELECT SUM(TMIN) AS S FROM weather HAVING SUM(S) = 3 AND S = 7", 1),
        ("SELECT SUM(TMIN) AS total FROM weather HAVING SUM(TMIN) = 7", 1),
    ];
    for (query, rows) in cases {
        let result = sql::execute(&node, query, None, None).unwrap_or_else(|e| panic!("{query}: {e}"));
        assert_eq!(result.rows.len(), rows, "{query}: {:?}", result.rows);
    }
    let err = sql::execute(&node, "SELECT SUM(TMIN) AS total FROM weather HAVING missing_alias > 0", None, None).err().unwrap();
    assert_eq!(err.kind, "verification_exception");
}
