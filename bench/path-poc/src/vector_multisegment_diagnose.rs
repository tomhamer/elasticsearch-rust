//! Read-only retained-index KNN diagnosis. No HTTP/Rally performance claims.
//! Opens existing committed cores without recovery, optimizer, or writes.
use rustic_core::{Index, index::KnnRequest, vector::store::PlannerConfig};
use serde_json::json;
use std::{path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        return Err("usage: vector-multisegment-diagnose core-directories.json queries.ndjson field [rounds=12]".into());
    }
    let paths: Vec<String> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let queries: Vec<Vec<f32>> = std::fs::read_to_string(&args[2])?.lines()
        .take(20).map(serde_json::from_str).collect::<Result<_, _>>()?;
    let rounds = args.get(4).map(|x| x.parse::<usize>()).transpose()?.unwrap_or(12);
    if paths.is_empty() || queries.is_empty() || rounds < 3 { return Err("nonempty inputs and at least three rounds required".into()); }
    let mut indexes = Vec::new();
    for path in &paths {
        let root = Path::new(path);
        if !root.is_dir() || !root.join("meta.json").is_file() { return Err("existing committed core required".into()); }
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("meta.json"))?)?;
        let started = Instant::now();
        let index = Index::open(root)?;
        let field = index.field_id(&args[3]).ok_or("missing vector field")?;
        println!("{}", json!({"event":"open","path":path,"elapsed_ms":started.elapsed().as_secs_f64()*1000.,
            "live_docs":index.num_live_docs(),"vector_segments":meta["vector_store"]["segments"],
            "segment_details":index.vector_store().segments(field).iter().map(|s| json!({"points":s.len(),"sealed":s.is_sealed(),"graph":s.has_graph()})).collect::<Vec<_>>()}));
        indexes.push((index, field));
    }
    for round in 0..rounds {
        for (copy, (index, field)) in indexes.iter().enumerate() {
            for (k, candidates) in [(10, 100), (100, 1000)] {
                for (qi, query) in queries.iter().enumerate() {
                    let mut request = KnnRequest::new(*field, query.clone(), k);
                    request.num_candidates = candidates;
                    let started = Instant::now();
                    let (hits, stats) = index.knn_search_with(&request, &PlannerConfig::default()).unwrap();
                    let millis = started.elapsed().as_secs_f64()*1000.;
                    if hits.len() != k { return Err(format!("incomplete hits: {} != {k}", hits.len()).into()); }
                    println!("{}", json!({"event":"query","round":round,"warmup":round<2,"copy":copy,
                        "query":qi,"k":k,"num_candidates":candidates,"elapsed_ms":millis,
                        "graph_segments":stats.graph_segments,"exact_segments":stats.exact_segments,
                        "fallbacks":stats.fallbacks,"vector_ops":stats.vector_ops,
                        "hits":hits.iter().map(|h|json!([h.leaf,h.doc,h.score])).collect::<Vec<_>>()}));
                }
            }
        }
    }
    Ok(())
}
