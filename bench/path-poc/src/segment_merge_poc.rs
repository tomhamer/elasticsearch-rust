//! EC2-only merge diagnostic over retained real segment files. No ES server.
use rustic_core::{codec::directory::MmapDirectory, merge::merge_segments, segment::Segment};
use std::{path::PathBuf, sync::Arc, time::Instant};

fn main() {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_ansi(false).with_writer(std::io::stderr).init();
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().expect("output directory"));
    std::fs::create_dir_all(&out).unwrap();
    let segments: Vec<_> = args.map(|path| {
        let path = PathBuf::from(path);
        let id = path.file_stem().unwrap().to_str().unwrap().strip_prefix("seg_").unwrap().parse().unwrap();
        let dir = MmapDirectory::open(path.parent().unwrap()).unwrap();
        Arc::new(Segment::open(&dir, id).unwrap())
    }).collect();
    assert!(segments.len() >= 2);
    let documents: u32 = segments.iter().map(|s| s.num_docs).sum();
    for pass in 0..3 {
        let dir_path = out.join(format!("pass-{pass}"));
        std::fs::create_dir_all(&dir_path).unwrap();
        let dir = MmapDirectory::open(&dir_path).unwrap();
        let start = Instant::now();
        let merged = merge_segments(&dir, &segments, 900000).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(merged.num_docs, documents);
        println!("{{\"pass\":{pass},\"documents\":{documents},\"merge_ms\":{ms}}}");
        drop(merged);
        drop(dir);
        // Keep one complete output for baseline/candidate byte comparison.
        if pass > 0 { std::fs::remove_dir_all(dir_path).unwrap(); }
    }
}
