//! Per-request response warnings (ES `HeaderWarning` / deprecation logger)
//! and the test modules a node runs with.
//!
//! Warnings raised while a request executes on this thread are collected
//! here; the HTTP layer drains them into `Warning` headers.

use std::cell::RefCell;

thread_local! {
    static WARNINGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static TEST_MODULES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Record a warning for the current request (deduplicated, in order).
pub fn add(msg: impl Into<String>) {
    let msg = msg.into();
    WARNINGS.with(|w| {
        let mut w = w.borrow_mut();
        if !w.contains(&msg) {
            w.push(msg);
        }
    });
}

/// Drain the warnings recorded on this thread.
pub fn take() -> Vec<String> {
    WARNINGS.with(|w| std::mem::take(&mut *w.borrow_mut()))
}

/// Node setting listing the test modules a node runs with (ES test
/// clusters' `clusterModules`, e.g. `x-pack-test-deprecated-query`).
pub const TEST_MODULES_SETTING: &str = "node.test_modules";

/// Restores the previous test-module list when dropped.
pub struct TestModulesGuard(Vec<String>);

impl Drop for TestModulesGuard {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.0);
        TEST_MODULES.with(|t| *t.borrow_mut() = prev);
    }
}

/// Make `modules` the installed test modules on this thread.
pub fn enter_test_modules(modules: Vec<String>) -> TestModulesGuard {
    TestModulesGuard(TEST_MODULES.with(|t| t.replace(modules)))
}

pub fn has_test_module(name: &str) -> bool {
    TEST_MODULES.with(|t| t.borrow().iter().any(|m| m == name))
}
