//! Small EC2 CPU diagnostic using real documents; not Rally or ingestion timing.
use elasticsearch_engine::{analysis::AnalysisRegistry, document, mapping::{FieldLookup, Mappings}, settings::FlatSettings, tsdb};
use serde_json::{json, Value};
use std::{fs, hint::black_box, time::Instant};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 4, "index-definition.json source.jsonl repeats");
    let definition: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let definition = definition.as_object().unwrap().values().next().unwrap();
    let settings: FlatSettings = serde_json::from_value(definition["settings"].clone()).unwrap();
    let reg = AnalysisRegistry::default();
    let mut mapping = Mappings::parse(&definition["mappings"], &reg).unwrap();
    tsdb::prepare_mappings(&settings, &mut mapping, &reg).unwrap();
    mapping.assign_core_ids();
    let lookup = FieldLookup::build(&mapping, &reg);
    let docs: Vec<Value> = fs::read_to_string(&args[2]).unwrap().lines().map(|s| elasticsearch_engine::jsonpos::parse_source_value(s).unwrap()).collect();
    assert!(!docs.is_empty());
    let repeats: usize = args[3].parse().unwrap();
    for doc in &docs {
        let original = tsdb::build_ids(&mapping, &settings, doc).unwrap();
        let prepared = lookup.tsdb_fields.build_ids(&mapping, &settings, doc).unwrap();
        assert_eq!((original.id, original.tsid), (prepared.id, prepared.tsid));
        assert_eq!(tsdb::check_position_fields(&mapping, doc).is_ok(), lookup.tsdb_fields.check_positions(doc).is_ok());
    }
    let mut times = Vec::new();
    for phase in ["ids", "prepared_ids", "positions", "prepared_positions", "mapper"] {
        let start = Instant::now();
        for _ in 0..repeats {
            for doc in &docs {
                match phase {
                    "ids" => { black_box(tsdb::build_ids(&mapping, &settings, black_box(doc)).unwrap()); }
                    "prepared_ids" => { black_box(lookup.tsdb_fields.build_ids(&mapping, &settings, black_box(doc)).unwrap()); }
                    "prepared_positions" => { lookup.tsdb_fields.check_positions(black_box(doc)).unwrap(); }
                    "positions" => { tsdb::check_position_fields(&mapping, black_box(doc)).unwrap(); }
                    _ => { black_box(document::parse_document(black_box(doc), "diagnostic", &mapping, &lookup).unwrap()); }
                }
            }
        }
        times.push(json!({"phase":phase,"ns":start.elapsed().as_nanos()}));
    }
    println!("{}",json!({"scope":"isolated CPU diagnostic, no indexing or HTTP","documents":docs.len(),"repeats":repeats,"phases":times}));
}
