//! `elasticsearch-faults`: Jepsen-style fault-injection suite for the
//! cluster-aware `elasticsearch` binary. Linux only (iptables, cgroup v2,
//! passwordless sudo). See `cluster/docs/FAULT_RESULTS.md`.
//!
//! ```text
//! elasticsearch-faults run [--scenario all|a,b,..] [--seeds N | --seed S] [--mode short|long]
//!                          [--nodes 5] [--port-base 9600] [--bin PATH] [--out DIR]
//!                          [--duration SECS] [--workers N] [--op-timeout SECS] [--netem]
//!                          [--registers N] [--keep-data] [--repeat-until-fail] [--fast-fd]
//! elasticsearch-faults list
//! elasticsearch-faults cleanup [--port-base 9600]
//! ```
//!
//! `--out` is a retained parent directory. Each invocation prints its absolute
//! artifact directory before starting nodes and writes `invocation.json` plus
//! `summary.json` / `summary.md` there. Repeated seeds get fresh scenario
//! directories; neither previous summaries nor failed node logs are overwritten.

mod api;
mod artifacts;
mod checker;
mod cluster;
mod history;
mod monitor;
mod nemesis;
mod net;
mod run;
mod workload;

use run::{Config, RunResult};
use std::path::PathBuf;
use std::time::Duration;

fn usage() -> ! {
    eprintln!("{}", include_str!("main.rs").lines().skip(5).take(8).map(|l| l.trim_start_matches("//! ")).collect::<Vec<_>>().join("\n"));
    std::process::exit(2);
}

fn cleanup(port_base: u16) {
    net::teardown(port_base);
    cluster::cleanup_processes(port_base);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first().cloned() else { usage() };
    let mut opts: std::collections::BTreeMap<String, String> = Default::default();
    let mut i = 1;
    while i < args.len() {
        let a = args[i].trim_start_matches("--").to_string();
        let is_flag = matches!(a.as_str(), "netem" | "keep-data" | "repeat-until-fail" | "fast-fd");
        if is_flag {
            opts.insert(a, "true".into());
            i += 1;
        } else {
            let Some(val) = args.get(i + 1) else { usage() };
            opts.insert(a, val.clone());
            i += 2;
        }
    }
    let port_base: u16 = opts.get("port-base").map(|s| s.parse().expect("port-base")).unwrap_or(9600);
    match cmd.as_str() {
        "list" => {
            for (n, d) in nemesis::SCENARIOS {
                println!("{n:<22} {d}");
            }
        }
        "cleanup" => {
            cleanup(port_base);
            println!("cleaned up port base {port_base}");
        }
        "run" => {
            let short = opts.get("mode").map(|m| m != "long").unwrap_or(true);
            let nodes: usize = opts.get("nodes").map(|s| s.parse().expect("nodes")).unwrap_or(5);
            assert!((3..=15).contains(&nodes), "--nodes must be 3..=15");
            let bin = opts.get("bin").map(PathBuf::from).unwrap_or_else(|| {
                let me = std::env::current_exe().expect("exe");
                me.parent().expect("dir").join("elasticsearch")
            });
            if !bin.exists() {
                eprintln!("node binary {} not found (--bin)", bin.display());
                std::process::exit(2);
            }
            let mut cfg = Config {
                nodes,
                port_base,
                bin,
                out: opts.get("out").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("/tmp/es-faults-{port_base}"))),
                short,
                dur: Duration::from_secs(opts.get("duration").map(|s| s.parse().expect("duration")).unwrap_or(if short { 25 } else { 300 })),
                workers: opts.get("workers").map(|s| s.parse().expect("workers")).unwrap_or(if short { 24 } else { 24 }),
                op_timeout: Duration::from_secs(opts.get("op-timeout").map(|s| s.parse().expect("op-timeout")).unwrap_or(if short { 10 } else { 30 })),
                netem: opts.contains_key("netem"),
                registers: opts.get("registers").map(|s| s.parse().expect("registers")).unwrap_or(5),
                keep_data: opts.contains_key("keep-data"),
                fast_fd: opts.contains_key("fast-fd"),
            };
            let scen: Vec<String> = match opts.get("scenario").map(String::as_str) {
                None | Some("all") => nemesis::SCENARIOS.iter().map(|(n, _)| n.to_string()).collect(),
                Some(list) => list.split(',').map(str::to_string).collect(),
            };
            for s in &scen {
                if !nemesis::SCENARIOS.iter().any(|(n, _)| n == s) {
                    eprintln!("unknown scenario {s}; see `list`");
                    std::process::exit(2);
                }
            }
            let seeds: Vec<u64> = match (opts.get("seed"), opts.get("seeds")) {
                (Some(s), _) => s.split(',').map(|x| x.parse().expect("seed")).collect(),
                (None, Some(n)) => (1..=n.parse::<u64>().expect("seeds")).collect(),
                (None, None) => vec![1],
            };
            let until_fail = opts.contains_key("repeat-until-fail");
            let requested_out = cfg.out.clone();
            cfg.out = artifacts::invocation_dir(&requested_out).expect("unique invocation directory");
            std::fs::write(cfg.out.join("invocation.json"), serde_json::to_vec_pretty(&serde_json::json!({
                "requested_out": requested_out, "artifacts": cfg.out,
                "node_binary": cfg.bin, "pid": std::process::id(),
                "scenarios": scen, "seeds": seeds,
                "retention": "Prior invocations are never deleted; collect before explicit cleanup."
            })).unwrap()).expect("invocation provenance");
            // Print before starting nodes, so even an interrupted invocation is discoverable.
            println!("artifacts: {}", cfg.out.display());
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("runtime");
            let code = rt.block_on(async move {
                // Clean up on Ctrl-C / SIGTERM.
                tokio::spawn(async move {
                    #[cfg(unix)]
                    {
                        use tokio::signal::unix::{signal, SignalKind};
                        let mut term = signal(SignalKind::terminate()).expect("sigterm");
                        tokio::select! {
                            _ = term.recv() => {}
                            _ = tokio::signal::ctrl_c() => {}
                        }
                    }
                    #[cfg(not(unix))]
                    let _ = tokio::signal::ctrl_c().await;
                    eprintln!("interrupted: cleaning up");
                    cleanup(port_base);
                    std::process::exit(130);
                });
                cleanup(port_base);
                let mut results: Vec<RunResult> = Vec::new();
                let mut round = 0u64;
                'outer: loop {
                    for seed in &seeds {
                        for s in &scen {
                            let seed = seed + round * 1000;
                            let r = run::run_one(&cfg, s, seed).await;
                            let failed = !r.ok();
                            results.push(r);
                            let _ = std::fs::write(cfg.out.join("summary.md"), run::summary_md(&results, &cfg));
                            let _ = std::fs::write(cfg.out.join("summary.json"), serde_json::to_string_pretty(&results).unwrap_or_default());
                            if failed && until_fail {
                                break 'outer;
                            }
                        }
                    }
                    round += 1;
                    if !until_fail {
                        break;
                    }
                }
                cleanup(port_base);
                println!("\n{}", run::summary_md(&results, &cfg));
                println!("artifacts: {}", cfg.out.display());
                if results.iter().all(|r| r.ok()) { 0 } else { 1 }
            });
            std::process::exit(code);
        }
        _ => usage(),
    }
}
