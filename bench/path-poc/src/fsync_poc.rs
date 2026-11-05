//! POC: measure the per-request translog fsync cost that dominates the
//! Metricbeat indexing gap (Rust 4,252 docs/s vs ES9 46,553 docs/s).
//!
//! The in-process write path is only 53.6 µs/doc (see http_write_poc.rs),
//! but Rally shows 15,882 µs/doc — a 296× gap. The `sync_mode` function in
//! docs.rs defaults to `true` (request durability), so every single-doc
//! index triggers a translog fsync via `group_sync`. This POC measures the
//! fsync cost on the same gp3 volume Rally uses.
//!
//! Run: `es-path-poc fsync-bench`

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::time::Instant;

const N_OPS: usize = 1000;
const BUF_SIZE: usize = 1024; // ~1 KB per translog record (metricbeat doc)

pub fn run() {
    let dir = std::env::temp_dir().join("fsync-poc");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("translog-test");
    let _ = std::fs::remove_file(&path);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap();

    let buf = vec![0u8; BUF_SIZE];

    // Warmup: 10 writes + fsyncs to settle the page cache.
    for _ in 0..10 {
        file.write_all(&buf).unwrap();
        file.sync_data().unwrap();
    }

    // Measure: N writes, each followed by an fsync (request durability).
    let mut t_write = 0.0f64;
    let mut t_fsync = 0.0f64;
    for _ in 0..N_OPS {
        let t0 = Instant::now();
        file.write_all(&buf).unwrap();
        t_write += t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        file.sync_data().unwrap();
        t_fsync += t1.elapsed().as_secs_f64();
    }

    let write_us = t_write * 1_000_000.0 / N_OPS as f64;
    let fsync_us = t_fsync * 1_000_000.0 / N_OPS as f64;
    let total_us = write_us + fsync_us;

    eprintln!("\n=== Translog fsync cost (gp3 volume) ===");
    eprintln!("Ops: {N_OPS}, buf: {BUF_SIZE} bytes");
    eprintln!("write_all:  {write_us:8.2} us/op");
    eprintln!("sync_data:  {fsync_us:8.2} us/op");
    eprintln!("total:      {total_us:8.2} us/op");
    eprintln!();
    eprintln!("Rally: Rust 15,882 us/doc vs ES9 514 us/doc (30.9x gap)");
    eprintln!("In-process POC: 53.6 us/doc (no fsync)");
    eprintln!("If fsync dominates: {:.2} us/doc explains the gap", fsync_us);
    eprintln!();
    eprintln!("ES9 uses group commit: one fsync per batch of ~500 docs,");
    eprintln!("so per-doc fsync cost = {:.2} us (amortized).", fsync_us / 500.0);
    eprintln!("Our per-doc fsync cost = {:.2} us (no amortization).", fsync_us);
    eprintln!("Ratio: {:.1}x", fsync_us / (fsync_us / 500.0));

    let _ = std::fs::remove_file(&path);
}
