//! Build a disposable payload-edge overlay from the fixture's actual metadata.
//! Usage: INDEX CORPUS SEGMENT_ID OUTPUT_GRAPH; never edits INDEX.
use rustic_core::{Index, vector::{hnsw::{HnswBuilder, HnswParams, VectorSource}, segment::VectorSegment}};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::{BufRead, BufReader}, time::Instant};
struct Subset<'a> { segment: &'a VectorSegment, members: Vec<bool> }
impl VectorSource for Subset<'_> {
    fn dim(&self) -> u32 { self.segment.dim }
    fn num_docs(&self) -> u32 { self.segment.len() }
    fn get(&self, doc: u32) -> Option<&[f32]> {
        self.members.get(doc as usize).copied().unwrap_or(false).then(|| self.segment.vector(doc))
    }
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(16);
    let ef: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(100);
    let index = Index::open(std::path::Path::new(&args[1])).unwrap();
    let segment = VectorSegment::open(index.directory().as_ref(), args[3].parse().unwrap()).unwrap();
    let leaves = index.leaves();
    assert_eq!(leaves.len(), 1, "fixture requires one document leaf");
    let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut count = 0;
    for (ord, line) in BufReader::new(std::fs::File::open(&args[2]).unwrap()).lines().enumerate() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        assert_eq!(leaves[0].keyword_value(0, ord as u32).unwrap(), v["id"].to_string(), "fixture ordinal/ID mismatch");
        let row: Vec<f32> = v["v"].as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect();
        assert_eq!(row, segment.vector(ord as u32), "fixture ordinal/vector mismatch");
        groups.entry(v["tag"].as_str().unwrap().to_string()).or_default().push(ord as u32);
        count += 1;
    }
    assert_eq!(count, segment.len());
    let base = segment.graph().expect("sealed graph required");
    let start = Instant::now();
    let partitions: Vec<_> = groups.values().map(|ords| {
        let mut members = vec![false; count as usize];
        for &ord in ords { members[ord as usize] = true; }
        let subset = Subset { segment: &segment, members };
        HnswBuilder::build(segment.dim, &subset, count, segment.similarity,
            HnswParams { m, ef_construction: ef, threads: 8, ..Default::default() })
    }).collect();
    let mut manifest = Vec::new();
    for ((value, _), (id, graph)) in groups.iter().zip(partitions.iter().enumerate()) {
        let path = format!("{}.part-{id}.hnsw", args[4]);
        std::fs::write(&path, graph.encode_upper_navigation()).unwrap();
        manifest.push(json!({"value":value,"graph":path}));
    }
    std::fs::write(format!("{}.parts.json", args[4]), serde_json::to_vec(&manifest).unwrap()).unwrap();
    let overlay = base.with_payload_edges(&partitions);
    let encoded = overlay.encode();
    std::fs::write(&args[4], &encoded).unwrap();
    println!("{}", json!({"documents":count,"values":groups.len(),"payload_m":m,"payload_ef_construction":ef,"build_seconds":start.elapsed().as_secs_f64(),
        "base_bytes":base.encode().len(),"overlay_bytes":encoded.len(),"base_mean_degree":base.mean_degree(),
        "overlay_mean_degree":overlay.mean_degree(),"fixture_ids_and_vectors_verified":true}));
}
