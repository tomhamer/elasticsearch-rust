//! Script resource containment: shared execution fuel with cancellation /
//! deadline polling on every loop kind, function entry and long built-ins
//! (production review #15), and breaker admission of collection
//! allocation in every script context (#16).

use super::compile::compile_arc;
use super::runtime::{Cancel, TIME_EXCEEDED};
use super::*;
use crate::memory::Breakers;
use serde_json::json;
use std::time::{Duration, Instant};

fn executor(src: &str, ctx: ScriptContext) -> Executor {
    let prog = compile_arc(src, ctx).unwrap_or_else(|e| panic!("compile {src:?}: {} at {:?}", e.msg, e.pos));
    Executor::new(prog, &json!({}), None, None)
}

/// Run `src` under a `deadline` (the search `timeout` path) on its own
/// worker thread; the worker must come back with `time_exceeded` promptly.
fn assert_stopped_by_deadline(src: &str, ctx: ScriptContext, deadline: Duration) {
    let src = src.to_string();
    let started = Instant::now();
    let worker = std::thread::spawn(move || {
        let mut e = executor(&src, ctx);
        e.rt.cancel = Some(Cancel::new(deadline));
        e.run().map(|v| v.to_json()).map_err(|e| (e.kind.clone(), e.msg.clone()))
    });
    // The worker is freed (joined) shortly after the deadline, not after
    // the billions of steps the script asks for.
    let result = worker.join().unwrap();
    let took = started.elapsed();
    let (kind, msg) = result.expect_err("script must be stopped");
    assert_eq!(kind, TIME_EXCEEDED, "{msg}");
    assert!(took < deadline + Duration::from_millis(750), "worker held for {took:?}");
}

#[test]
fn deadline_cancels_nested_enhanced_for_loops() {
    // 1000^3 enhanced-for iterations; only the build loop is an ordinary loop.
    let src = "List l = new ArrayList(); for (int i = 0; i < 1000; i++) { l.add(i); } long n = 0; \
               for (def a : l) { for (def b : l) { for (def c : l) { n += a + b + c; } } } return n;";
    assert_stopped_by_deadline(src, ScriptContext::Score, Duration::from_millis(50));
    // Arrays and map key sets iterate the same way.
    let src = "int[] a = new int[2000]; Map m = new HashMap(); for (int i = 0; i < 2000; i++) { m.put(i, i); } long n = 0; \
               for (int x : a) { for (def k : m.keySet()) { for (int y : a) { n += x + y; } } } return n;";
    assert_stopped_by_deadline(src, ScriptContext::Filter, Duration::from_millis(50));
}

#[test]
fn deadline_cancels_branching_recursion_without_loops() {
    let src = "long f(int n) { if (n < 2) { return n; } return f(n - 1) + f(n - 2); } return f(64);";
    assert_stopped_by_deadline(src, ScriptContext::Score, Duration::from_millis(50));
    // Lambdas called by built-ins (no script loop at all).
    let src = "List l = new ArrayList(); for (int i = 0; i < 3000; i++) { l.add(i); } \
               return l.stream().mapToLong(a -> l.stream().mapToLong(b -> l.stream().mapToLong(c -> c).sum()).sum()).sum();";
    assert_stopped_by_deadline(src, ScriptContext::Score, Duration::from_millis(50));
}

#[test]
fn deadline_is_not_catchable() {
    let src = "while (true) { try { List l = Collections.nCopies(5000, 1); for (def x : l) { } } catch (Exception e) { } }";
    assert_stopped_by_deadline(src, ScriptContext::Score, Duration::from_millis(50));
}

#[test]
fn cancelled_task_stops_enhanced_loops_without_a_deadline() {
    let token = rustic_core::cancel::Token::new();
    let src = "List l = new ArrayList(); for (int i = 0; i < 2000; i++) { l.add(i); } long n = 0; \
               for (def a : l) { for (def b : l) { for (def c : l) { n += 1; } } } return n;";
    let mut e = executor(src, ScriptContext::Score);
    e.rt.task = Some(token.clone());
    // Cancelled up front: a timed canceller races `max_loop_counter` on fast
    // hosts. The first fuel poll (every POLL_INTERVAL units) must stop it.
    token.cancel("test");
    let started = Instant::now();
    let err = e.run().unwrap_err();
    assert_eq!(err.kind, TIME_EXCEEDED);
    assert!(started.elapsed() < Duration::from_millis(800), "{:?}", started.elapsed());
}

#[test]
fn loop_limit_matches_es_and_cpu_containment_is_the_deadline() {
    let limit = "The maximum number of statements that can be executed in a loop has been reached.";
    // Enhanced loops count against `max_loop_counter` (no deadline: update / ingest).
    let err = executor("List l = Collections.nCopies(1100000, 1); long n = 0; for (def x : l) { n += x; } return n;", ScriptContext::PainlessTest).run().unwrap_err();
    assert_eq!((err.kind.as_str(), err.msg.as_str()), ("painless_error", limit));
    // ES: a counter per function invocation. Callees do not add to their
    // caller's count, function entries are not loop iterations.
    assert_eq!(executor("void g() { for (int i = 0; i < 600000; i++) { } } g(); g(); return 1;", ScriptContext::PainlessTest).run().unwrap().to_json(), json!(1));
    assert_eq!(executor("int g() { int s = 0; for (int i = 0; i < 3; i++) { s += i; } return s; } long t = 0; for (int i = 0; i < 400000; i++) { t += g(); } return t;", ScriptContext::PainlessTest).run().unwrap().to_json(), json!(1_200_000));
    assert_eq!(executor("List l = [1, 2, 3]; long t = 0; for (int i = 0; i < 400000; i++) { l.forEach(x -> { }); t++; } return t;", ScriptContext::PainlessTest).run().unwrap().to_json(), json!(400_000));
    // Loop-free recursion is no loop: fib(30) (2.7M calls) runs as in ES.
    let fib = "long f(int n) { if (n < 2) { return n; } return f(n - 1) + f(n - 2); }";
    assert_eq!(executor(&format!("{fib} return f(30);"), ScriptContext::Update).run().unwrap().to_json(), json!(832040));
    // The limit still applies inside a callee's own loop.
    let err = executor("void g() { while (true) { } } g(); return 1;", ScriptContext::PainlessTest).run().unwrap_err();
    assert_eq!(err.msg, limit);
    // Runaway recursion is stopped by the deadline instead (fib(64) never
    // finishes): see `deadline_cancels_branching_recursion_without_loops`.
    let mut e = executor(&format!("{fib} return f(20);"), ScriptContext::PainlessTest);
    for _ in 0..3 {
        assert_eq!(e.run().unwrap().to_json(), json!(6765));
    }
}

// ---------------------------------------------------------------- #16

fn small_breaker(limit: &str) -> (rustic_core::admission::PrivateShared, &'static Breakers) {
    let budget = rustic_core::admission::private_shared(256 << 20);
    let breakers = Breakers::private_for_tests(1 << 30);
    let limit = limit.to_string();
    breakers.apply_settings(&move |key| match key {
        "indices.breaker.request.limit" => Some(limit.clone()),
        "indices.breaker.total.limit" => Some("-1".into()),
        _ => None,
    });
    (budget, breakers)
}

fn run_on(breakers: &'static Breakers, src: &str, ctx: ScriptContext) -> std::result::Result<serde_json::Value, Box<ScriptErr>> {
    let request = crate::memory::RequestScope::new("script containment", breakers);
    crate::memory::scoped(Some(request), || {
        let mut e = executor(src, ctx);
        let r = e.run().map(|v| v.to_json());
        drop(e);
        r
    })
}

#[test]
fn collections_are_admitted_in_every_script_context() {
    let (_budget, breakers) = small_breaker("1mb");
    let grow = "List l = new ArrayList(); for (int i = 0; i < 200000; i++) { l.add(i); }";
    let map = "Map m = new HashMap(); for (int i = 0; i < 200000; i++) { m.put(i, i); }";
    for ctx in [ScriptContext::Filter, ScriptContext::Field, ScriptContext::Score, ScriptContext::NumberSort, ScriptContext::Update,
        ScriptContext::Ingest, ScriptContext::RuntimeLong, ScriptContext::Aggs, ScriptContext::MetricMap, ScriptContext::MetricInit,
        ScriptContext::Watcher, ScriptContext::PainlessTest] {
        for body in [grow, map] {
            let src = format!("{body} return 1;");
            let err = run_on(breakers, &src, ctx).expect_err(&format!("{ctx:?}: {src}"));
            assert_eq!(err.breaker.as_ref().map(|b| b.status), Some(429), "{ctx:?}: {} {}", err.kind, err.msg);
            assert_eq!(breakers.request.reserved(), 0, "{ctx:?}: reservation leaked");
        }
        // Within the limit the script runs and gives every byte back.
        assert_eq!(run_on(breakers, "List l = new ArrayList(); for (int i = 0; i < 1000; i++) { l.add(i); } return l.size();", ctx).unwrap(), json!(1000), "{ctx:?}");
        assert_eq!(breakers.request.reserved(), 0, "{ctx:?}");
    }
}

#[test]
fn ncopies_and_nested_arrays_are_rejected_before_allocation() {
    let (_budget, breakers) = small_breaker("1mb");
    for (src, kind) in [
        // 2^31-1 copies: a Java lazy view; materialised here only if admitted.
        ("return Collections.nCopies(2147483647, 'x').size();", None),
        ("return Collections.nCopies(100000000, 1).size();", None),
        // Each dimension is in range; their product is not affordable.
        ("int[][] a = new int[40000][40000]; return a.length;", None),
        // Beyond Java's largest array in total: OutOfMemoryError, before any work.
        ("int[][][] a = new int[100000][100000][100000]; return a.length;", Some("out_of_memory_error")),
        ("long[][] a = new long[1000000][1000000]; return a.length;", Some("out_of_memory_error")),
    ] {
        let started = Instant::now();
        let err = run_on(breakers, src, ScriptContext::Score).expect_err(src);
        match kind {
            Some(kind) => assert_eq!(err.kind, kind, "{src}: {}", err.msg),
            None => assert_eq!(err.breaker.as_ref().map(|b| b.status), Some(429), "{src}: {} {}", err.kind, err.msg),
        }
        // Rejected up front: no partial fill of hundreds of megabytes first.
        assert!(started.elapsed() < Duration::from_millis(250), "{src}: {:?}", started.elapsed());
        assert_eq!(breakers.request.reserved(), 0, "{src}");
    }
    // Affordable ones still work.
    assert_eq!(run_on(breakers, "return Collections.nCopies(1000, 'x').size();", ScriptContext::Score).unwrap(), json!(1000));
    assert_eq!(run_on(breakers, "int[][] a = new int[100][100]; a[99][99] = 7; return a[99][99];", ScriptContext::Score).unwrap(), json!(7));
    assert_eq!(super::compile::array_cells(&[10, 10, 10]).unwrap(), 1110);
    assert!(super::compile::array_cells(&[1_000_000, 1_000_000, 1_000_000]).is_err());
}

#[test]
fn doubling_string_concatenation_is_admitted() {
    let (_budget, breakers) = small_breaker("4mb");
    let err = run_on(breakers, "String s = 'x'; for (int i = 0; i < 40; i++) { s = s + s; } return s.length();", ScriptContext::Score).unwrap_err();
    assert_eq!(err.kind, "out_of_memory_error", "{}", err.msg);
    assert_eq!(breakers.request.reserved(), 0);
}

#[test]
fn self_referencing_collections_release_their_reservations() {
    let (_budget, breakers) = small_breaker("16mb");
    for _ in 0..50 {
        let v = run_on(breakers, "List l = new ArrayList(); for (int i = 0; i < 1000; i++) { l.add(i); } l.add(l); Map m = new HashMap(); m.put('self', m); return 1;", ScriptContext::Update).unwrap();
        assert_eq!(v, json!(1));
        assert_eq!(breakers.request.reserved(), 0, "cyclic collection kept its reservation past the executor");
    }
}

/// An executor's collection allocations draw from one pooled breaker
/// reservation (not a reservation and release per allocation): many runs
/// of a filter script that builds small collections hold a bounded amount
/// between runs, a run needing more than the limit still trips, and the
/// executor gives everything back when it drops.
#[test]
fn collection_accounting_is_pooled_per_executor() {
    let (_budget, breakers) = small_breaker("1mb");
    let request = crate::memory::RequestScope::new("pooled accounting", breakers);
    crate::memory::scoped(Some(request), || {
        let mut e = executor("List parts = ['a', ' ', 'b']; StringBuilder sb = new StringBuilder(); for (def p : parts) { sb.append(p); } return sb.toString().startsWith('a');", ScriptContext::Filter);
        for _ in 0..10_000 {
            assert_eq!(e.run().unwrap().to_json(), json!(true));
        }
        let held = breakers.request.reserved();
        assert!(held > 0 && held <= 256 * 1024, "pool held {held} bytes between runs");
        drop(e);
    });
    assert_eq!(breakers.request.reserved(), 0, "the pool outlived its executor");
    // A run needing more than the limit still trips, and releases.
    let err = run_on(breakers, "List l = new ArrayList(); for (int i = 0; i < 200000; i++) { l.add(i); } return l.size();", ScriptContext::Filter).unwrap_err();
    assert_eq!(err.breaker.as_ref().map(|b| b.status), Some(429), "{} {}", err.kind, err.msg);
    assert_eq!(breakers.request.reserved(), 0);
}
