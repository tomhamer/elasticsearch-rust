//! Short concurrent HTTP load probe; systems run separately on the same CPUs.
use super::*;

pub(super) async fn run(args: &[String]) {
    let entries: Vec<Value> = BufReader::new(std::fs::File::open(&args[2]).unwrap())
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    let targets: Vec<_> = args[3]
        .split(',')
        .map(|target| target.split_once('=').unwrap())
        .collect();
    let clients: Vec<usize> = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("1,4,8")
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    let seconds: u64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(8);
    let client = reqwest::Client::builder()
        .no_gzip()
        .tcp_nodelay(true)
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();
    let mut results = Vec::new();
    for (name, url) in targets {
        assert!(url.starts_with("http://127.0.0.1:"));
        let prepared = Arc::new(
            entries
                .iter()
                .map(|entry| {
                    client
                        .post(url)
                        .header("Content-Type", "application/json")
                        .body(serde_json::to_vec(&entry["request"]).unwrap())
                        .build()
                        .unwrap()
                })
                .collect::<Vec<_>>(),
        );
        // Untimed warmup includes every query, not just the first query vector.
        for _ in 0..3 {
            for request in prepared.iter() {
                let response = client.execute(request.try_clone().unwrap()).await.unwrap();
                assert!(response.status().is_success());
                let _ = response.bytes().await.unwrap();
            }
        }
        for &concurrency in &clients {
            assert!(concurrency > 0);
            let start = Instant::now();
            let until = start + std::time::Duration::from_secs(seconds);
            let mut workers = Vec::new();
            for worker in 0..concurrency {
                let client = client.clone();
                let requests = prepared.clone();
                workers.push(tokio::spawn(async move {
                    let mut seed = 42u64 + worker as u64 * 104729;
                    let mut timings = Vec::new();
                    let mut errors = 0;
                    while Instant::now() < until {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let index = ((seed >> 32) as usize) % requests.len();
                        let request = requests[index].try_clone().unwrap();
                        let t = Instant::now();
                        let response = client.execute(request).await.unwrap();
                        let status = response.status();
                        let bytes = response.bytes().await.unwrap();
                        let value: Value = serde_json::from_slice(&bytes).unwrap();
                        timings.push(t.elapsed().as_secs_f64() * 1e6);
                        if !status.is_success()
                            || value["_shards"]["failed"] != 0
                            || value["timed_out"] == true
                        {
                            errors += 1;
                        }
                    }
                    (timings, errors)
                }));
            }
            let mut timings = Vec::new();
            let mut errors = 0;
            for worker in workers {
                let (part, failures) = worker.await.unwrap();
                timings.extend(part);
                errors += failures;
            }
            let wall = start.elapsed().as_secs_f64();
            timings.sort_by(f64::total_cmp);
            let n = timings.len();
            assert!(n > 0);
            let result = json!({"target":name,"clients":concurrency,"requests":n,"errors":errors,
                "wall_seconds":wall,"queries_per_second":n as f64/wall,
                "p50_us":timings[n/2],"p95_us":timings[(n as f64*0.95).ceil() as usize-1]});
            println!("{result}");
            results.push(result);
            std::fs::write(&args[4],json!({"scope":"separate systems; persistent pre-encoded requests; mixed queries; response decoding included",
                "requests":args[2],"results":results}).to_string()).unwrap();
            assert_eq!(errors, 0);
        }
    }
}
