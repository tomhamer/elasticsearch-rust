//! `es-yaml-runner`: runs Elasticsearch's YAML REST test suite
//! (`tests/es-yaml/{test,api}`, fetched by `tests/es-yaml/fetch.sh`) against
//! fresh in-process `elasticsearch-http` servers and reports compatibility.
//!
//! ```text
//! cargo run -p es-yaml-runner -- [--filter <dir-or-path>]... [--test <name substr>]
//!     [--jobs N] [--report [docs/ES_YAML_COMPAT.md]] [--json out.json] [-v]
//!     [--security]   # + x-pack security suites (tests/es-yaml/xpack-test) on a security-enabled node
//!     [--xpack]      # + x-pack security suites on the default (security disabled) node
//!     [--no-xpack]   # the suite's cluster has no x-pack (ES's OSS rest-api-spec suite)
//!     [--compat 8]   # ES's yamlRestCompatTest: the previous major's suite, every request
//!                    # `compatible-with=8`, ES's compat transforms + skips (see compat.rs)
//! ```

mod compat;
mod exec;
mod report;
mod spec;
mod suite;

use exec::Outcome;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct TestResult {
    pub file: String,
    pub dir: String,
    pub name: String,
    pub outcome: Outcome,
}

struct Args {
    suite: PathBuf,
    filters: Vec<String>,
    test: Option<String>,
    jobs: usize,
    report: Option<PathBuf>,
    json: Option<PathBuf>,
    verbose: bool,
    /// Include the x-pack security suites (`tests/es-yaml/xpack-test`).
    xpack: bool,
}

fn parse_args() -> Args {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut a = Args {
        suite: root.join("tests/es-yaml"),
        filters: vec![],
        test: None,
        jobs: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8),
        report: None,
        json: None,
        verbose: false,
        xpack: false,
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
            "--filter" | "-f" => a.filters.push(next(&mut i).expect("--filter <dir>")),
            "--test" | "-t" => a.test = next(&mut i),
            "--jobs" | "-j" => a.jobs = next(&mut i).and_then(|s| s.parse().ok()).expect("--jobs N"),
            "--json" => a.json = next(&mut i).map(PathBuf::from),
            "--report" => {
                if argv.get(i + 1).map(|s| !s.starts_with('-')).unwrap_or(false) {
                    a.report = next(&mut i).map(PathBuf::from);
                } else {
                    a.report = Some(root.join("docs/ES_YAML_COMPAT.md"));
                }
            }
            "-v" | "--verbose" => a.verbose = true,
            // x-pack security suites against a security-enabled node (ES's XPackRestIT setup)
            "--security" => {
                a.xpack = true;
                // ES's x-pack YAML cluster names its single node `yamlRestTest-0`.
                // SAFETY: runs during arg parsing, before any threads are spawned.
        unsafe { std::env::set_var("HOSTNAME", "yamlRestTest-0") };
                exec::SECURITY.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // x-pack security suites against the default (security disabled) node
            "--xpack" => a.xpack = true,
            "--no-xpack" => exec::NO_XPACK.store(true, std::sync::atomic::Ordering::Relaxed),
            "--compat" => {
                let v = next(&mut i).and_then(|s| s.parse::<u8>().ok()).expect("--compat <major>");
                if v != 8 {
                    eprintln!("--compat: only 8 (the previous major of {}) is supported", elasticsearch_engine::version::ES_VERSION);
                    std::process::exit(2);
                }
                compat::COMPAT.store(v, std::sync::atomic::Ordering::Relaxed);
            }
            "-h" | "--help" => {
                eprintln!("usage: es-yaml-runner [--filter <dir|path substr>]... [--test <name substr>] [--jobs N] [--report [file]] [--json file] [-v] [--security | --xpack] [--no-xpack] [--compat 8]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
        i += 1;
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
        } else if p.extension().and_then(|s| s.to_str()) == Some("yml") || p.extension().and_then(|s| s.to_str()) == Some("yaml") {
            let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
}

/// Suites whose ES test class skips itself on this hardware (a JUnit
/// assumption, before any test runs).
fn assumption_skip(rel: &str) -> Option<String> {
    // `GPUClientYamlTestSuiteIT` / `GPUClientMultiNodeYamlTestSuiteIT`:
    // `assumeTrue(GPUSupport.isSupported())`; this node has no GPU.
    (rel.starts_with("gpu/gpu/supported/") || rel.contains("/gpu/supported/")).then(|| "GPU suite: assumes a supported GPU".to_string())
}

fn matches_filter(rel: &str, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    let dir = rel.split('/').next().unwrap_or("");
    filters.iter().any(|f| {
        let f = f.trim_end_matches('/');
        dir == f || rel.starts_with(&format!("{f}/")) || rel == f || (f.contains('/') && rel.contains(f))
    })
}

fn main() {
    // ES's YAML test clusters (InferenceRestIT, RRF/Linear rank ITs) install the
    // `inference-service-test` plugin; register its mock inference services.
    // SAFETY: runs before any threads are spawned.
    unsafe { std::env::set_var("ES_INFERENCE_TEST_SERVICES", "1") };
    let mut args = parse_args();
    let specs = match spec::Specs::load(&args.suite.join("api")) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let test_root = args.suite.join("test");
    let mut files = Vec::new();
    collect_files(&test_root, &test_root, &mut files);
    if std::env::var("ES_YAML_SECURITY").map(|v| v == "1" || v == "true").unwrap_or(false) {
        args.xpack = true;
        // SAFETY: runs during arg parsing, before any threads are spawned.
        unsafe { std::env::set_var("HOSTNAME", "yamlRestTest-0") };
        exec::SECURITY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if args.xpack {
        exec::DEFAULT_DISTRIBUTION.store(true, std::sync::atomic::Ordering::Relaxed);
        let xroot = args.suite.join("xpack-test");
        collect_files(&xroot, &xroot, &mut files);
    }
    files.retain(|(rel, _)| matches_filter(rel, &args.filters));
    if let Ok(ex) = std::env::var("ES_YAML_EXCLUDE") {
        files.retain(|(rel, _)| !ex.split(',').any(|e| !e.is_empty() && rel.contains(e)));
    }
    let transforms = compat::compat_major().map(|_| {
        let kind = if exec::NO_XPACK.load(std::sync::atomic::Ordering::Relaxed) {
            compat::SuiteKind::Oss
        } else if args.xpack {
            compat::SuiteKind::XPack
        } else {
            compat::SuiteKind::Plugins
        };
        Arc::new(compat::Transforms::load(kind))
    });
    eprintln!("running {} files with {} jobs{}", files.len(), args.jobs, if transforms.is_some() { " (compatible-with=8)" } else { "" });

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().worker_threads(args.jobs.max(2)).build().expect("runtime");
    let started = std::time::Instant::now();
    let results: Vec<TestResult> = rt.block_on(async {
        let sem = Arc::new(tokio::sync::Semaphore::new(args.jobs.max(1)));
        let mut set = tokio::task::JoinSet::new();
        for (rel, path) in files {
            let sem = sem.clone();
            let specs = specs.clone();
            let test_filter = args.test.clone();
            let transforms = transforms.clone();
            set.spawn(async move {
                let _permit = sem.acquire_owned().await.expect("semaphore");
                let dir = rel.split('/').next().unwrap_or("").to_string();
                let text = match std::fs::read_to_string(&path) {
                    Ok(t) => t,
                    Err(e) => return vec![TestResult { file: rel, dir, name: "<file>".into(), outcome: Outcome::Fail { msg: e.to_string(), category: "runner: read error".into() } }],
                };
                let file = match suite::parse_file(&rel, &text) {
                    Ok(mut f) => {
                        if let Some(t) = &transforms {
                            t.apply(&mut f);
                        }
                        f
                    }
                    Err(e) => return vec![TestResult { file: rel, dir, name: "<file>".into(), outcome: Outcome::Fail { msg: e, category: "runner: yaml parse error".into() } }],
                };
                let mut out = Vec::new();
                for (name, steps) in &file.tests {
                    if let Some(t) = &test_filter {
                        if !name.contains(t.as_str()) {
                            continue;
                        }
                    }
                    let outcome = match assumption_skip(&file.rel).or_else(|| transforms.as_ref().and_then(|t| t.skip_reason(&file.rel, name))) {
                        Some(reason) => exec::Outcome::Skip(reason),
                        None => exec::run_test(&specs, &file, steps).await,
                    };
                    out.push(TestResult { file: file.rel.clone(), dir: file.dir.clone(), name: name.clone(), outcome });
                }
                out
            });
        }
        let mut all = Vec::new();
        while let Some(r) = set.join_next().await {
            match r {
                Ok(v) => all.extend(v),
                Err(e) => eprintln!("file task panicked: {e}"),
            }
        }
        all
    });
    rt.shutdown_timeout(std::time::Duration::from_secs(2));
    let mut results = results;
    results.sort_by(|a, b| (&a.file, &a.name).cmp(&(&b.file, &b.name)));

    let (mut pass, mut fail, mut skip) = (0, 0, 0);
    for r in &results {
        match &r.outcome {
            Outcome::Pass => pass += 1,
            Outcome::Fail { msg, .. } => {
                fail += 1;
                if args.verbose {
                    println!("FAIL {} :: {}\n     {}", r.file, r.name, msg);
                }
            }
            Outcome::Skip(_) => skip += 1,
        }
    }
    let ran = pass + fail;
    println!(
        "tests: {} | pass {pass} | fail {fail} | skip {skip} | assertion pass rate {:.1}% of executed | selected cases passed {:.1}%  | {:.1}s",
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
        report::write_markdown(p, &results, version.trim(), &args.filters);
        eprintln!("report written to {}", p.display());
    }
}
