//! Filesystem-only harness contract; execute with the integration suite on EC2.
#[path = "common/evidence.rs"]
mod evidence;

#[test]
fn repeated_fixture_names_and_unwinds_preserve_prior_evidence() {
    let first = evidence::scratch_dir("repeated-fixture");
    std::fs::write(first.join("failed-node.log"), "original failure").unwrap();
    let second = evidence::scratch_dir("repeated-fixture");
    assert_ne!(first, second);
    assert_eq!(std::fs::read_to_string(first.join("failed-node.log")).unwrap(), "original failure");
    let failed = evidence::retained_dir("panicking-fixture");
    let path = failed.path().to_path_buf();
    std::fs::write(path.join("node.log"), "before panic").unwrap();
    assert!(std::panic::catch_unwind(move || { let _owned = failed; panic!("simulated test assertion"); }).is_err());
    assert_eq!(std::fs::read_to_string(path.join("node.log")).unwrap(), "before panic");
    // Only this completed filesystem contract owns these fixtures; no nodes exist.
    for dir in [first, second, path] { std::fs::remove_dir_all(dir).unwrap(); }
}
