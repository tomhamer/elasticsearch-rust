//! Shared types: corpus/query records, latency statistics, machine info and
//! the JSON result envelope that every subcommand (and the Lucene harness in
//! `bench/lucene`) writes.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// One corpus document as produced by `bench/corpus/gen.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct Doc {
    pub id: i64,
    pub title: String,
    pub body: String,
    pub price: f64,
    pub ts: i64,
    pub tag: String,
}

/// One query as produced by `bench/corpus/queries.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct Query {
    pub id: u32,
    pub class: String,
    pub field: String,
    pub kind: String,
    #[serde(default)]
    pub term: Option<String>,
    #[serde(default)]
    pub terms: Vec<String>,
    #[serde(default)]
    pub lower: Option<f64>,
    #[serde(default)]
    pub upper: Option<f64>,
}

impl Query {
    /// All terms of the query (single-term queries carry `term`).
    pub fn all_terms(&self) -> Vec<&str> {
        match &self.term {
            Some(t) => vec![t.as_str()],
            None => self.terms.iter().map(|s| s.as_str()).collect(),
        }
    }

    /// Human-readable form for mismatch reports.
    pub fn describe(&self) -> String {
        match self.kind.as_str() {
            "range" => format!(
                "{}:[{} TO {}]",
                self.field,
                self.lower.unwrap_or(f64::MIN),
                self.upper.unwrap_or(f64::MAX)
            ),
            "phrase" => format!("{}:\"{}\"", self.field, self.terms.join(" ")),
            "and" => format!("{}:({})", self.field, self.terms.join(" AND ")),
            "or" => format!("{}:({})", self.field, self.terms.join(" OR ")),
            _ => format!("{}:{}", self.field, self.all_terms().join(" ")),
        }
    }
}

/// Stream the JSONL corpus, calling `f` for each document.
pub fn for_each_doc(path: &Path, mut f: impl FnMut(Doc)) -> std::io::Result<usize> {
    let file = fs::File::open(path)?;
    let reader = BufReader::with_capacity(1 << 20, file);
    let mut n = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let doc: Doc = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("bad corpus line {}: {e}", n + 1));
        f(doc);
        n += 1;
    }
    Ok(n)
}

/// Load all queries.
pub fn load_queries(path: &Path) -> std::io::Result<Vec<Query>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).expect("bad query line"));
    }
    Ok(out)
}

/// Total size in bytes of all regular files under `dir` (recursive).
pub fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(md) = entry.metadata() {
                total += md.len();
            }
        }
    }
    total
}

/// Nearest-rank percentile (`p` in 0..=100) of a *sorted* slice.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Summary statistics of a set of latencies, in microseconds.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyStats {
    pub n: usize,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub mean_us: f64,
    pub min_us: f64,
    pub max_us: f64,
}

impl LatencyStats {
    pub fn from_durations(ds: &[Duration]) -> Self {
        let mut v: Vec<f64> = ds.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            return Self::default();
        }
        Self {
            n: v.len(),
            p50_us: percentile_sorted(&v, 50.0),
            p90_us: percentile_sorted(&v, 90.0),
            p99_us: percentile_sorted(&v, 99.0),
            mean_us: v.iter().sum::<f64>() / v.len() as f64,
            min_us: v[0],
            max_us: v[v.len() - 1],
        }
    }
}

/// Machine description captured with every result.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MachineInfo {
    pub cpu: String,
    pub logical_cores: u32,
    pub physical_cores: u32,
    pub ram_bytes: u64,
    pub os: String,
    pub arch: String,
    /// Free-form host label from `$BENCH_MACHINE` (e.g. an EC2 instance type).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
}

/// `key : value` lookup in a /proc-style file (Linux).
fn proc_field(path: &str, key: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    text.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim() == key).then(|| v.trim().to_string())
    })
}

fn sysctl(name: &str) -> String {
    Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

impl MachineInfo {
    pub fn capture() -> Self {
        let os = if cfg!(target_os = "macos") {
            let name = Command::new("sw_vers")
                .arg("-productName")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|_| "macOS".into());
            let ver = Command::new("sw_vers")
                .arg("-productVersion")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_default();
            format!("{name} {ver}")
        } else {
            std::env::consts::OS.to_string()
        };
        let label = std::env::var("BENCH_MACHINE").unwrap_or_default();
        if cfg!(target_os = "linux") {
            let logical = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(0);
            // physical = distinct (physical id, core id) pairs
            let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            let mut cores = std::collections::BTreeSet::new();
            let mut phys = String::new();
            for l in cpuinfo.lines() {
                if let Some((k, v)) = l.split_once(':') {
                    match k.trim() {
                        "physical id" => phys = v.trim().to_string(),
                        "core id" => {
                            cores.insert((phys.clone(), v.trim().to_string()));
                        }
                        _ => {}
                    }
                }
            }
            let mem_kb: u64 = proc_field("/proc/meminfo", "MemTotal")
                .and_then(|v| v.split_whitespace().next().and_then(|x| x.parse().ok()))
                .unwrap_or(0);
            let os = proc_field("/etc/os-release", "PRETTY_NAME")
                .or_else(|| {
                    fs::read_to_string("/etc/os-release").ok()?.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string()))
                })
                .unwrap_or_else(|| "Linux".into());
            return Self {
                cpu: proc_field("/proc/cpuinfo", "model name").unwrap_or_default(),
                logical_cores: logical,
                physical_cores: cores.len() as u32,
                ram_bytes: mem_kb * 1024,
                os,
                arch: std::env::consts::ARCH.to_string(),
                label,
            };
        }
        Self {
            cpu: sysctl("machdep.cpu.brand_string"),
            logical_cores: sysctl("hw.ncpu").parse().unwrap_or(0),
            physical_cores: sysctl("hw.physicalcpu").parse().unwrap_or(0),
            ram_bytes: sysctl("hw.memsize").parse().unwrap_or(0),
            os,
            arch: std::env::consts::ARCH.to_string(),
            label,
        }
    }
}

/// Result envelope. `metrics` is subcommand-specific (see the README for the
/// exact shape of each); the Lucene harness writes the same envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFile {
    pub system: String,
    pub subcommand: String,
    pub timestamp_unix: u64,
    pub machine: MachineInfo,
    pub params: serde_json::Value,
    pub metrics: serde_json::Value,
}

impl ResultFile {
    pub fn new(system: &str, subcommand: &str, params: serde_json::Value, metrics: serde_json::Value) -> Self {
        Self {
            system: system.to_string(),
            subcommand: subcommand.to_string(),
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            machine: MachineInfo::capture(),
            params,
            metrics,
        }
    }

    /// Write to `results_dir/<system>-<subcommand>.json`.
    pub fn write(&self, results_dir: &Path) -> std::io::Result<PathBuf> {
        fs::create_dir_all(results_dir)?;
        let path = results_dir.join(format!("{}-{}.json", self.system, self.subcommand));
        fs::write(&path, serde_json::to_string_pretty(self).unwrap())?;
        Ok(path)
    }
}

/// Minimal `--flag value` argument parser (no external deps).
pub struct Args {
    pub subcommand: String,
    pairs: Vec<(String, String)>,
}

impl Args {
    pub fn parse() -> Self {
        let mut it = std::env::args().skip(1);
        let subcommand = it.next().unwrap_or_default();
        let mut pairs = Vec::new();
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            let a = &rest[i];
            if let Some(name) = a.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    pairs.push((k.to_string(), v.to_string()));
                } else if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                    pairs.push((name.to_string(), rest[i + 1].clone()));
                    i += 1;
                } else {
                    pairs.push((name.to_string(), "true".to_string()));
                }
            } else {
                // Positional values (used by `report --inputs a.json b.json ...`).
                pairs.push(("_".to_string(), a.clone()));
            }
            i += 1;
        }
        Self { subcommand, pairs }
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.pairs.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    pub fn require(&self, name: &str) -> String {
        self.get(name)
            .unwrap_or_else(|| {
                eprintln!("missing required argument --{name}");
                std::process::exit(2);
            })
            .to_string()
    }

    pub fn get_or(&self, name: &str, default: &str) -> String {
        self.get(name).unwrap_or(default).to_string()
    }

    pub fn get_usize(&self, name: &str, default: usize) -> usize {
        self.get(name).map(|v| v.parse().expect("integer argument")).unwrap_or(default)
    }

    /// All values given for a repeated/multi-valued flag plus positionals
    /// (e.g. `--inputs results/*.json` after shell expansion).
    pub fn get_all(&self, name: &str) -> Vec<String> {
        self.pairs
            .iter()
            .filter(|(k, _)| k == name || k == "_")
            .map(|(_, v)| v.clone())
            .collect()
    }
}
