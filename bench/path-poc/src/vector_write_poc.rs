//! Fresh process per mode; measures serialization allocation, not HNSW build.
use rustic_core::{codec::{directory::{Directory,MmapDirectory},vectors::{VectorSegmentParts,encode_vector_segment,write_vector_segment_streamed,open_vector_segment}},vector::Similarity};
use serde_json::json;
use std::{path::Path,time::Instant};
fn memory()->String{std::fs::read_to_string("/proc/self/status").unwrap_or_default().lines().filter(|s|s.starts_with("VmRSS:")||s.starts_with("VmHWM:")||s.starts_with("RssAnon:")).collect::<Vec<_>>().join("; ")}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let a:Vec<_>=std::env::args().collect();if a.len()!=4{return Err("usage: rustic-vector-write-poc baseline|stream OUTPUT_DIRECTORY ROWS".into())}
 let root=Path::new(&a[2]);if root.exists(){return Err("output directory exists".into())}
 let n:usize=a[3].parse()?;let dim=128;let points:Vec<_>=(0..n as u64).collect();let vectors:Vec<f32>=(0..n*dim).map(|i|((i*17%1000)as f32-500.)/500.).collect();
 let graph=vec![0x51u8;n*64];let quant=vec![0x33u8;n*128];
 // Non-empty graph/quant byte sections test encoding, not graph decodability.
 let parts=VectorSegmentParts{field:7,similarity:Similarity::Cosine,dim:dim as u32,points:&points,vectors:&vectors,hnsw:Some(&graph),int8:Some(&quant),int4:None,flat:false};
 let dir=MmapDirectory::open(root)?;let before=memory();let t=Instant::now();
 if a[1]=="stream"{write_vector_segment_streamed(&dir,"result.rvec",&parts)?}else if a[1]=="baseline"{let bytes=encode_vector_segment(&parts);dir.write("result.rvec",&bytes)?}else{return Err("invalid mode".into())}
 let elapsed=t.elapsed().as_secs_f64()*1000.;let after=memory();
 // Release input buffers before round-trip verification so mapped output
 // does not artificially double the verification phase working set.
 drop(parts);drop(vectors);drop(graph);drop(quant);drop(points);
 let parsed=open_vector_segment(dir.open_read("result.rvec")?,true)?;assert_eq!(parsed.num_vectors,n as u32);assert!(parsed.points.iter().enumerate().all(|(i,&p)|p==i as u64));
 println!("{}",json!({"mode":a[1],"rows":n,"elapsed_ms":elapsed,"before":before,"after":after,"file_bytes":dir.open_read("result.rvec")?.len(),"crc_verified":true}));
 Ok(())
}
