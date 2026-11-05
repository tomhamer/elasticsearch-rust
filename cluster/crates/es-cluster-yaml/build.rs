//! Copies the single-node runner's executor (`crates/es-yaml-runner/src/
//! {exec,spec,suite}.rs`) into `OUT_DIR` with minimal textual patches so this
//! crate stays in sync with it automatically:
//!
//! * `//!` inner doc comments are stripped (not allowed in `include!`d files);
//! * `exec.rs`: the engine import and the in-process `run_test` are removed and
//!   `src/exec_ext.rs` (the external-cluster `run_test`) is appended inside the
//!   same module, so it can use the private `Ctx` API unchanged;
//! * request URLs go through `pick_base` (per-request round robin over the
//!   cluster's HTTP addresses, like ES's REST test client). This patch is
//!   optional: if their code changes shape, the runner falls back to one URL.

use std::path::PathBuf;

fn strip_inner_docs(s: &str) -> String {
    s.lines().filter(|l| !l.trim_start().starts_with("//!")).map(|l| format!("{l}\n")).collect()
}

/// Remove `pub async fn run_test(...) { ... }` (and its doc comment).
fn remove_fn(src: &str, sig: &str) -> String {
    let Some(start) = src.find(sig) else { panic!("es-cluster-yaml build: `{sig}` not found in es-yaml-runner exec.rs") };
    // Back up over the preceding `///` lines.
    let mut cut = start;
    let before = &src[..start];
    let mut lines: Vec<&str> = before.split_inclusive('\n').collect();
    while let Some(l) = lines.last() {
        if l.trim_start().starts_with("///") || l.trim().is_empty() && lines.len() > 1 && lines[lines.len() - 2].trim_start().starts_with("///") {
            cut -= l.len();
            lines.pop();
        } else {
            break;
        }
    }
    let open = start + src[start..].find('{').expect("fn body");
    let mut depth = 0i32;
    let mut end = open;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = open + i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    format!("{}{}", &src[..cut], &src[end..])
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let runner = manifest.join("../../../crates/es-yaml-runner/src");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for f in ["exec.rs", "spec.rs", "suite.rs"] {
        println!("cargo:rerun-if-changed={}", runner.join(f).display());
    }
    println!("cargo:rerun-if-changed=src/exec_ext.rs");
    let read = |f: &str| std::fs::read_to_string(runner.join(f)).unwrap_or_else(|e| panic!("reading {}: {e}", runner.join(f).display()));
    let write = |f: &str, s: String| std::fs::write(out.join(f), s).expect("write OUT_DIR");

    write("spec.rs", strip_inner_docs(&read("spec.rs")));
    write("suite.rs", strip_inner_docs(&read("suite.rs")));

    let mut exec = strip_inner_docs(&read("exec.rs"));
    // Top-level imports only: function-local `use`s (indented) must stay.
    exec = exec.lines().filter(|l| !l.starts_with("use elasticsearch_engine")).map(|l| format!("{l}\n")).collect();
    exec = remove_fn(&exec, "pub async fn run_test(");
    // Optional: per-request round robin over the cluster's nodes.
    let url_pat = "format!(\"{}{}\", self.base, path)";
    if exec.contains(url_pat) {
        exec = exec.replacen(url_pat, "format!(\"{}{}\", pick_base(&self.base), path)", 1);
        exec = exec.replace(".trim_start_matches(&self.base)", ".trim_start_matches(base_of(&built.url))");
    } else {
        println!("cargo:warning=es-cluster-yaml: URL round-robin patch did not apply; all requests go to the first --url");
    }
    let ext = manifest.join("src/exec_ext.rs");
    exec.push_str(&format!("\ninclude!({:?});\n", ext.display().to_string()));
    write("exec.rs", exec);
}
