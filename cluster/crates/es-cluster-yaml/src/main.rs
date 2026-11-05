//! `es-cluster-yaml`: runs Elasticsearch's YAML REST test suite
//! (`tests/es-yaml/{test,api}`) against an EXTERNAL, possibly multi-node
//! cluster, the way ES's `qa/smoke-test-multinode` does: requests
//! round-robin over every node's HTTP address and the cluster is wiped
//! between tests (`ESRestTestCase.wipeCluster`).
//!
//! The executor (do/match/stash/catch/skip semantics) is es-yaml-runner's own
//! `exec.rs`/`spec.rs`/`suite.rs`, pulled in by build.rs, so both runners
//! always agree on test semantics.
//!
//! ```text
//! cargo run -p es-cluster-yaml -- --url http://127.0.0.1:9400,http://127.0.0.1:9401 \
//!     [--filter <dir-or-path>]... [--test <name substr>] [--baseline single.json]
//!     [--report [docs/ES_YAML_CLUSTER_COMPAT.md]] [--json out.json] [-v]
//! ```

#[allow(dead_code, clippy::all)]
mod spec {
    include!(concat!(env!("OUT_DIR"), "/spec.rs"));
}
#[allow(dead_code, clippy::all)]
mod suite {
    include!(concat!(env!("OUT_DIR"), "/suite.rs"));
}
#[allow(dead_code, clippy::all)]
mod exec {
    include!(concat!(env!("OUT_DIR"), "/exec.rs"));
}
mod report;
mod wipe;

// The shared executor uses native/compat header and feature semantics.
// Include the same module so its relative embedded reference data stays valid.
#[allow(dead_code)]
#[path = "../../../../crates/es-yaml-runner/src/compat.rs"]
mod compat;

use exec::Outcome;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct TestResult {
    pub file: String,
    pub dir: String,
    pub name: String,
    pub outcome: Outcome,
}

struct Args {
    suite: PathBuf,
    /// Directory under `suite` holding the tests: `test` (default) or
    /// `xpack-test` (`--xpack`).
    tests_dir: String,
    urls: Vec<String>,
    filters: Vec<String>,
    test: Option<String>,
    report: Option<PathBuf>,
    json: Option<PathBuf>,
    baseline: Option<PathBuf>,
    timeout: u64,
    wipe: bool,
    verbose: bool,
}

fn parse_args() -> Args {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut a = Args {
        suite: root.join("tests/es-yaml"),
        tests_dir: "test".into(),
        urls: vec![],
        filters: vec![],
        test: None,
        report: None,
        json: None,
        baseline: None,
        timeout: 180,
        wipe: true,
        verbose: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> Option<String> {
            *i += 1;
            argv.get(*i).cloned()
        };
        match argv[i].as_str() {
            "--suite" => a.suite = PathBuf::from(next(&mut i).expect("--suite <dir>")),
            "--url" | "-u" => a.urls.extend(next(&mut i).expect("--url <url[,url]>").split(',').map(|s| s.trim().trim_end_matches('/').to_string()).filter(|s| !s.is_empty())),
            "--filter" | "-f" => a.filters.push(next(&mut i).expect("--filter <dir>")),
            "--test" | "-t" => a.test = next(&mut i),
            "--json" => a.json = next(&mut i).map(PathBuf::from),
            "--baseline" => a.baseline = next(&mut i).map(PathBuf::from),
            "--timeout" => a.timeout = next(&mut i).and_then(|s| s.parse().ok()).expect("--timeout <secs>"),
            "--no-wipe" => a.wipe = false,
            "--tests-dir" => a.tests_dir = next(&mut i).expect("--tests-dir <name under --suite>"),
            "--xpack" => a.tests_dir = "xpack-test".into(),
            "--security" => exec::SECURITY.store(true, std::sync::atomic::Ordering::Relaxed),
            "--export-fixture" => {
                let module = next(&mut i).expect("--export-fixture <module>");
                println!("{}", serde_json::to_string_pretty(&exec::external_fixture(&module)).unwrap());
                std::process::exit(0);
            },
            "--reference-url" => exec::set_reference_base(next(&mut i).expect("--reference-url <official ES URL>").trim_end_matches('/').to_string()),
            "--module-security-fixtures" => exec::MODULE_SECURITY_FIXTURES.store(true, std::sync::atomic::Ordering::Relaxed),
            "--no-xpack" => exec::NO_XPACK.store(true, std::sync::atomic::Ordering::Relaxed),
            "--report" => {
                if argv.get(i + 1).map(|s| !s.starts_with('-')).unwrap_or(false) {
                    a.report = next(&mut i).map(PathBuf::from);
                } else {
                    a.report = Some(root.join("cluster/docs/ES_YAML_CLUSTER_COMPAT.md"));
                }
            }
            "-v" | "--verbose" => a.verbose = true,
            "-h" | "--help" => {
                eprintln!("usage: es-cluster-yaml --url <http://host:port[,...]> [--filter <dir|path>]... [--test <substr>] [--baseline single-node.json] [--reference-url official-es-url] [--report [file]] [--json file] [--timeout secs] [--no-wipe] [--security] [--module-security-fixtures] [--no-xpack] [--xpack | --tests-dir <name>] [-v]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if a.urls.is_empty() {
        a.urls.push("http://127.0.0.1:9400".into());
    }
    a
}

fn collect_files(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_files(&p, root, out);
        } else if matches!(p.extension().and_then(|s| s.to_str()), Some("yml") | Some("yaml")) {
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
}

/// Same filter semantics as es-yaml-runner, plus a trailing `*` on the
/// directory (`cluster.*`).
fn matches_filter(rel: &str, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    let dir = rel.split('/').next().unwrap_or("");
    filters.iter().any(|f| {
        let f = f.trim_end_matches('/');
        if let Some(prefix) = f.strip_suffix('*') {
            if !f.contains('/') {
                return dir.starts_with(prefix);
            }
        }
        dir == f || rel.starts_with(&format!("{f}/")) || rel == f || (f.contains('/') && rel.contains(f))
    })
}

fn main() {
    let args = parse_args();
    let specs = match spec::Specs::load(&args.suite.join("api")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let test_root = args.suite.join(&args.tests_dir);
    let mut files = Vec::new();
    collect_files(&test_root, &test_root, &mut files);
    files.retain(|(rel, _)| matches_filter(rel, &args.filters));
    exec::set_bases(args.urls.clone());
    let base = args.urls[0].clone();
    eprintln!("running {} files against {}", files.len(), args.urls.join(", "));

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().worker_threads(4).build().expect("runtime");
    let started = std::time::Instant::now();
    let wiper = wipe::Wiper::new(&base);
    let (results, topology) = rt.block_on(async {
        if let Err(e) = wiper.wait_ready(args.urls.len()).await {
            eprintln!("{e}");
            std::process::exit(1);
        }
        let topology = reqwest::get(format!("{base}/_cat/nodes?h=name,node.role,master,version&s=name")).await.ok();
        let topology = match topology {
            Some(r) => r.text().await.unwrap_or_default(),
            None => String::new(),
        };
        let mut all = Vec::new();
        if args.wipe {
            let problems = wiper.wipe().await;
            if !problems.is_empty() {
                for problem in &problems { eprintln!("initial wipe: {problem}"); }
                all.push(TestResult { file: "<fixture>".into(), dir: "<runner>".into(), name: "initial cleanup".into(),
                    outcome: Outcome::Fail { msg: problems.join("; "), category: "runner: cleanup error".into() } });
            }
        }
        for (rel, path) in files {
            let dir = rel.split('/').next().unwrap_or("").to_string();
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => {
                    all.push(TestResult { file: rel, dir, name: "<file>".into(), outcome: Outcome::Fail { msg: e.to_string(), category: "runner: read error".into() } });
                    continue;
                }
            };
            let file = match suite::parse_file(&rel, &text) {
                Ok(f) => f,
                Err(e) => {
                    all.push(TestResult { file: rel, dir, name: "<file>".into(), outcome: Outcome::Fail { msg: e, category: "runner: yaml parse error".into() } });
                    continue;
                }
            };
            for (name, steps) in &file.tests {
                if let Some(t) = &args.test {
                    if !name.contains(t.as_str()) {
                        continue;
                    }
                }
                let mut outcome = exec::run_test(&specs, &file, steps, &base, Duration::from_secs(args.timeout)).await;
                let ran = !matches!(outcome, Outcome::Skip(_));
                if ran && args.wipe {
                    let problems = wiper.wipe().await;
                    if !problems.is_empty() {
                        for problem in &problems { eprintln!("  wipe after [{} :: {}]: {problem}", file.rel, name); }
                        let cleanup = format!("cleanup failed: {}", problems.join("; "));
                        match &mut outcome {
                            Outcome::Pass => outcome = Outcome::Fail { msg: cleanup, category: "runner: cleanup error".into() },
                            Outcome::Fail { msg, .. } => { msg.push_str("; "); msg.push_str(&cleanup); }
                            Outcome::Skip(_) => {}
                        }
                    }
                }
                match &outcome {
                    Outcome::Pass => eprintln!("PASS {} :: {}", file.rel, name),
                    Outcome::Fail { msg, .. } => eprintln!("FAIL {} :: {}{}", file.rel, name, if args.verbose { format!("\n     {msg}") } else { String::new() }),
                    Outcome::Skip(_) => {}
                }
                all.push(TestResult { file: file.rel.clone(), dir: file.dir.clone(), name: name.clone(), outcome });
            }
            // Keep incomplete-run evidence separate from the final report.
            if let Some(path) = &args.json {
                report::write_json(&path.with_extension("partial.json"), &all);
            }
        }
        (all, topology)
    });
    rt.shutdown_timeout(Duration::from_secs(2));
    let mut results = results;
    results.sort_by(|a, b| (&a.file, &a.name).cmp(&(&b.file, &b.name)));

    let (mut pass, mut fail, mut skip) = (0, 0, 0);
    for r in &results {
        match &r.outcome {
            Outcome::Pass => pass += 1,
            Outcome::Fail { .. } => fail += 1,
            Outcome::Skip(_) => skip += 1,
        }
    }
    let ran = pass + fail;
    println!(
        "tests: {} | pass {pass} | fail {fail} | skip {skip} | assertion pass rate {:.1}% of executed | selected cases passed {:.1}% | {:.1}s",
        results.len(),
        100.0 * pass as f64 / ran.max(1) as f64,
        100.0 * pass as f64 / results.len().max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    println!("Scope: selected YAML cases. A pass rate is not full ES9 parity; failures and excluded/skipped cases must be reviewed.");
    if let Some(p) = &args.json {
        report::write_json(p, &results);
    }
    if let Some(p) = &args.report {
        let version = std::fs::read_to_string(args.suite.join("VERSION")).unwrap_or_else(|_| "unknown".into());
        let baseline = args.baseline.as_deref().map(report::load_baseline).unwrap_or_default();
        report::write_markdown(p, &results, &baseline, version.trim(), &args.filters, &args.urls, &topology);
        eprintln!("report written to {}", p.display());
    }
    // Persist every report before signalling failure to standalone callers.
    // Skips remain explicit in the report and are not counted as failures.
    if results.is_empty() {
        eprintln!("No YAML cases selected; check suite, tests directory and filters.");
        std::process::exit(2);
    }
    if fail > 0 {
        std::process::exit(1);
    }
}
