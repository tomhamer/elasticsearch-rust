use elasticsearch_engine::{mapping::reserved_core_fields, shard::{Shard, WriteOp}};
use rustic_core::types::{Document, Field, FieldType};

#[test]
fn compressed_primary_history_pages_across_flush_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let mut schema = reserved_core_fields();
    schema.push(Field::new(8, "title", FieldType::Text));
    let shard = Shard::create(tmp.path(), "test", schema.clone()).unwrap();
    shard.add_retention_lease("reader", 0, "test").unwrap();
    for batch in 0..2 {
        let ops = (batch * 200..(batch + 1) * 200).map(|i| {
            let mut doc = Document::new();
            doc.add_text(8, "repeated title for compression");
            WriteOp::Index {
                id: format!("d{i}"), source: "{\"title\":\"repeated title for compression\"}".into(),
                doc, ignored: vec![], routing: None, cond: Default::default(), auto_id: false,
            }
        }).collect();
        assert!(shard.apply("test", "uuid", ops, true).iter().all(Result::is_ok));
        shard.flush().unwrap();
    }
    drop(shard);
    let (shard, replay) = Shard::open(tmp.path(), schema).unwrap();
    assert!(replay.is_empty());
    let expected: Vec<_> = shard.changes_range(0, 399, true).unwrap().collect();
    assert_eq!(expected.len(), 400);
    let mut cursor = None;
    let mut actual = Vec::new();
    while actual.len() < expected.len() {
        // A byte budget must stop within a compressed frame too.
        let page = shard.changes_from_cursor_sized(
            &mut cursor, actual.len() as i64, 399, 7, 3, |_| 1,
        ).unwrap();
        assert!(!page.is_empty() && page.len() <= 3);
        actual.extend(page);
    }
    assert_eq!(actual, expected);
    assert!(shard.changes_from_cursor(&mut cursor, 400, 399, 7).unwrap().is_empty());
}
