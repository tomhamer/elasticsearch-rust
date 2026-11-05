//! Read-only existing-segment POC; isolates graph/column cost from ES recovery.
//! Output is library diagnostic evidence, not an ES-vs-Qdrant API benchmark.
use rustic_core::{codec::directory::MmapDirectory, vector::{segment::VectorSegment, Quantization}};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
fn process() -> Value {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<_> = stat.rsplit_once(") ").map(|(_, s)| s.split_whitespace().collect()).unwrap_or_default();
    json!({"rss":status.lines().find(|l|l.starts_with("VmRSS:")),"hwm":status.lines().find(|l|l.starts_with("VmHWM:")),"minor_faults":fields.get(7),"major_faults":fields.get(9)})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 { return Err("usage: rustic-vector-diagnose CORE_DIRECTORY queries.json [rounds=10] [only_segment_id] [ordinal_stride=1]".into()) }
    let root = Path::new(&args[1]);
    // MmapDirectory can create directories, so reject missing inputs first.
    if !root.is_dir() { return Err("input core directory missing".into()) }
    let meta: Value = serde_json::from_slice(&std::fs::read(root.join("meta.json"))?)?;
    let queries: Vec<Vec<f32>> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let rounds = args.get(3).map(|x|x.parse::<usize>()).transpose()?.unwrap_or(10);
    let only_segment = args.get(4).map(|x|x.parse::<u32>()).transpose()?;
    let stride = args.get(5).map(|x|x.parse::<u32>()).transpose()?.unwrap_or(1).max(1);
    let directory = MmapDirectory::open(root)?;
    let mut segments = Vec::new();
    for id in meta["vector_store"]["segments"].as_array().ok_or("missing vector segment list")? {
        let id = id.as_u64().ok_or("invalid segment id")? as u32;
        if only_segment.is_some_and(|wanted| wanted != id) {continue}
        let segment = VectorSegment::open(&directory,id)?;
        println!("{}",json!({"event":"segment_open","id":id,"rows":segment.len(),"dim":segment.dim,"graph":segment.has_graph(),"process":process()}));
        segments.push(segment);
    }
    let begun=Instant::now();
    for segment in &segments {segment.warm(Quantization::None);}
    println!("{}",json!({"event":"warm","elapsed_ms":begun.elapsed().as_secs_f64()*1000.,"resident_column_bytes":segments.iter().map(|s|s.resident_bytes()).sum::<usize>(),"process":process()}));
    for round in 0..rounds { for (qi,q) in queries.iter().enumerate() {
        let begun=Instant::now(); let before=process(); let mut candidates=Vec::new();
        for segment in &segments {
            let t=Instant::now();
            let hits=if segment.has_graph() {segment.search_graph(q,10,500,|o|o % stride == 0)} else {segment.search_exact(q,10,(0..segment.len()).filter(|o|o % stride == 0))};
            println!("{}",json!({"event":"segment_query","round":round,"query":qi,"segment":segment.id,"rows":segment.len(),"graph":segment.has_graph(),"elapsed_ms":t.elapsed().as_secs_f64()*1000.}));
            candidates.extend(hits.into_iter().map(|(ord,score)|(segment.id,ord,score)));
        }
        candidates.sort_by(|a,b|b.2.total_cmp(&a.2));candidates.truncate(10);
        println!("{}",json!({"event":"query","round":round,"query":qi,"elapsed_ms":begun.elapsed().as_secs_f64()*1000.,"before":before,"after":process(),"hits":candidates}));
    }}
    Ok(())
}
