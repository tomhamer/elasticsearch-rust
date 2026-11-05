//! Rally-like `_bulk` load driver (std only): N clients, each sending
//! `bulk_size`-doc bulks (`{"index":{"_index":IDX}}` + source line, the body
//! Rally's `bulk` operation sends for a corpus without ids) over a keep-alive
//! HTTP/1.1 connection, until the corpus (or `max_docs`) is exhausted.
//!
//! cargo run --release -p elasticsearch-node --example bulk_load -- \
//!   127.0.0.1:9200 documents.json geonames [bulk_size=5000] [clients=8] [max_docs=0] [warmup_secs=0]
//!
//! Prints throughput (docs/s, after warmup) and service-time percentiles.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Source {
    reader: BufReader<std::fs::File>,
    sent: u64,
    max: u64,
}

fn next_body(src: &Mutex<Source>, index: &str, bulk: usize) -> Option<(Vec<u8>, usize)> {
    let mut s = src.lock().unwrap();
    let mut body = Vec::with_capacity(bulk * 400);
    let action = format!("{{\"index\":{{\"_index\":\"{index}\"}}}}\n");
    let mut n = 0;
    let mut line = String::new();
    while n < bulk && (s.max == 0 || s.sent < s.max) {
        line.clear();
        if s.reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        body.extend_from_slice(action.as_bytes());
        body.extend_from_slice(line.trim_end().as_bytes());
        body.push(b'\n');
        n += 1;
        s.sent += 1;
    }
    (n > 0).then_some((body, n))
}

fn read_response(r: &mut BufReader<TcpStream>) -> std::io::Result<(u16, Vec<u8>)> {
    let mut status_line = String::new();
    r.read_line(&mut status_line)?;
    let status: u16 = status_line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut len: Option<usize> = None;
    let mut chunked = false;
    loop {
        let mut h = String::new();
        r.read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().ok();
        } else if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            let mut sz = String::new();
            r.read_line(&mut sz)?;
            let n = usize::from_str_radix(sz.trim(), 16).unwrap_or(0);
            let mut chunk = vec![0; n + 2];
            r.read_exact(&mut chunk)?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..n]);
        }
    } else if let Some(n) = len {
        body.resize(n, 0);
        r.read_exact(&mut body)?;
    }
    Ok((status, body))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: bulk_load HOST:PORT FILE INDEX [bulk_size] [clients] [max_docs] [warmup_secs]");
        std::process::exit(2);
    }
    let host = a[1].clone();
    let index = a[3].clone();
    let bulk: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let clients: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let max: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
    let warmup = Duration::from_secs(a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0));
    let f = std::fs::File::open(&a[2]).expect("corpus");
    let src = Arc::new(Mutex::new(Source { reader: BufReader::with_capacity(1 << 20, f), sent: 0, max }));
    let start = Instant::now();
    // (completion time since start, docs, service time)
    let samples: Arc<Mutex<Vec<(Duration, usize, Duration)>>> = Arc::new(Mutex::new(Vec::new()));
    let errors = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let handles: Vec<_> = (0..clients)
        .map(|_| {
            let (src, samples, host, index, errors) = (src.clone(), samples.clone(), host.clone(), index.clone(), errors.clone());
            std::thread::spawn(move || {
                let conn = TcpStream::connect(&host).expect("connect");
                conn.set_nodelay(true).ok();
                let mut w = conn.try_clone().unwrap();
                let mut r = BufReader::with_capacity(1 << 20, conn);
                while let Some((body, n)) = next_body(&src, &index, bulk) {
                    let t0 = Instant::now();
                    let head = format!("POST /_bulk HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\n\r\n", body.len());
                    w.write_all(head.as_bytes()).unwrap();
                    w.write_all(&body).unwrap();
                    let (status, resp) = read_response(&mut r).expect("response");
                    let dt = t0.elapsed();
                    // Rally checks the top-level `errors` flag.
                    let head = &resp[..resp.len().min(64)];
                    if status != 200 || head.windows(13).any(|x| x == b"\"errors\":true") {
                        errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if errors.load(std::sync::atomic::Ordering::Relaxed) <= 3 {
                            eprintln!("error response {status}: {}", String::from_utf8_lossy(&resp[..resp.len().min(600)]));
                        }
                    }
                    samples.lock().unwrap().push((start.elapsed(), n, dt));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let total = start.elapsed();
    let s = samples.lock().unwrap();
    let all_docs: usize = s.iter().map(|x| x.1).sum();
    let measured: Vec<&(Duration, usize, Duration)> = s.iter().filter(|x| x.0 >= warmup).collect();
    let mdocs: usize = measured.iter().map(|x| x.1).sum();
    let mspan = total.saturating_sub(warmup).as_secs_f64().max(1e-9);
    let mut lat: Vec<f64> = measured.iter().map(|x| x.2.as_secs_f64() * 1000.0).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| if lat.is_empty() { 0.0 } else { lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)] };
    println!(
        "docs={all_docs} elapsed={:.1}s overall={:.0} docs/s | measured docs={mdocs} tput={:.0} docs/s | service ms p50={:.1} p90={:.1} p99={:.1} max={:.1} | error_responses={}",
        total.as_secs_f64(),
        all_docs as f64 / total.as_secs_f64(),
        mdocs as f64 / mspan,
        pct(0.5),
        pct(0.9),
        pct(0.99),
        pct(1.0),
        errors.load(std::sync::atomic::Ordering::Relaxed)
    );
}
