//! Retained cluster fixtures. Collection, not a later test, owns cleanup.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn safe_tag(tag: &str) -> String {
    tag.chars().take(100).map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn run_dir() -> &'static Path {
    static RUN: OnceLock<PathBuf> = OnceLock::new();
    RUN.get_or_init(|| {
        let root = std::env::var_os("RUSTIC_TEST_ARTIFACT_DIR").map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("rustic-two-node-tests"));
        std::fs::create_dir_all(&root).expect("test artifact root");
        let exe = std::env::current_exe().expect("test executable");
        let name = safe_tag(&exe.file_name().unwrap().to_string_lossy());
        let dir = tempfile::Builder::new().prefix(&format!("{}-{name}-", std::process::id()))
            .disable_cleanup(true).tempdir_in(root).expect("unique test run directory");
        std::fs::write(dir.path().join("run.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "pid": std::process::id(), "executable": exe,
            "retention": "Collect results and stop all nodes before explicit cleanup; absence of a live PID is not success."
        })).unwrap()).expect("run provenance");
        dir.path().to_path_buf()
    })
}

/// Fresh storage, even for repeated tags or reused process IDs. The original
/// fixture is the evidence: no copy and no sweep of older runs is made.
pub fn retained_dir(tag: &str) -> tempfile::TempDir {
    let dir = tempfile::Builder::new().prefix(&format!("{}-", safe_tag(tag)))
        .disable_cleanup(true).tempdir_in(run_dir()).expect("unique retained fixture");
    std::fs::write(dir.path().join("fixture.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "tag": tag, "test_thread": std::thread::current().name(), "pid": std::process::id()
    })).unwrap()).expect("fixture provenance");
    eprintln!("retained test fixture: {}", dir.path().display());
    dir
}

pub fn scratch_dir(tag: &str) -> PathBuf {
    retained_dir(tag).path().to_path_buf()
}
