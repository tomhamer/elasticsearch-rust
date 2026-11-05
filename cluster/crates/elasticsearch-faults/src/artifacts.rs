//! Unique, retained output directories; rerunning a seed never erases evidence.
use std::io;
use std::path::{Path, PathBuf};

pub fn unique_dir(root: &Path, stem: &str) -> io::Result<PathBuf> {
    std::fs::create_dir_all(root)?;
    let root = std::fs::canonicalize(root)?;
    for attempt in 0u64.. {
        let dir = root.join(format!("{stem}-{attempt}"));
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    unreachable!("exhausted output directory identifiers")
}

/// The supplied --out remains a discoverable parent containing every invocation.
/// Each child has its own summaries, invocation metadata and scenario artifacts.
pub fn invocation_dir(root: &Path) -> io::Result<PathBuf> {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map_err(io::Error::other)?.as_nanos();
    unique_dir(root, &format!("run-{stamp}-{}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_invocations_and_seeds_preserve_failed_artifacts() {
        let root = invocation_dir(&std::env::temp_dir().join("rustic-faults-artifact-tests")).unwrap();
        let first = invocation_dir(&root).unwrap();
        std::fs::write(first.join("summary.json"), "old failed summary").unwrap();
        let failed = unique_dir(&first, "pause-n3-s1").unwrap();
        std::fs::write(failed.join("node.log"), "original failure").unwrap();
        let retry = unique_dir(&first, "pause-n3-s1").unwrap();
        let second = invocation_dir(&root).unwrap();
        assert_ne!(first, second);
        assert_ne!(failed, retry);
        assert_eq!(std::fs::read_to_string(first.join("summary.json")).unwrap(), "old failed summary");
        assert_eq!(std::fs::read_to_string(failed.join("node.log")).unwrap(), "original failure");
        assert!(second.is_absolute());
        // This test owns all descendants, and no node process was started.
        std::fs::remove_dir_all(root).unwrap();
    }
}
