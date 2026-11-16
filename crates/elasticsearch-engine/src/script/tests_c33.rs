//! Closure tests for the c33 scripting review findings (601, 606-609,
//! 615-618, 623, 624): native walks and callbacks are contained, lambdas
//! keep their own frame, dependent runtime fields share the deadline, and
//! float / parsing / locale / stream / regex / OffsetTime semantics follow
//! Painless on the JDK.

use super::compile::compile_arc;
use super::*;
use serde_json::{json, Value};

fn exec(src: &str) -> std::result::Result<Value, (String, String)> {
    let prog = compile_arc(src, ScriptContext::PainlessTest).map_err(|e| (e.kind.clone(), e.msg.clone()))?;
    let mut e = Executor::new(prog, &json!({}), None, None);
    e.run().map(|v| v.to_json()).map_err(|e| (e.kind.clone(), e.msg.clone()))
}

fn ok(src: &str) -> Value {
    exec(src).unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"))
}

fn err_kind(src: &str) -> String {
    match exec(src) {
        Ok(v) => panic!("{src:?} should fail, returned {v}"),
        Err((kind, _)) => kind,
    }
}

// ---- 601: native object-graph walks are bounded -------------------------

#[test]
fn cyclic_collection_walks_fail_as_a_script_error() {
    assert_eq!(err_kind("List l = new ArrayList(); l.add(l); return l.hashCode();"), "stack_overflow_error");
    assert_eq!(err_kind("Map m = new HashMap(); m.put('k', m); return m.hashCode();"), "stack_overflow_error");
    // An indirect cycle: hashing, equality and a set insertion.
    assert_eq!(err_kind("List a = new ArrayList(); List b = new ArrayList(); a.add(b); b.add(a); return a.hashCode();"), "stack_overflow_error");
    assert_eq!(err_kind("List a = new ArrayList(); List b = new ArrayList(); a.add(a); b.add(b); return a.equals(b);"), "stack_overflow_error");
    // A cyclic map key is hashed on insertion.
    assert_eq!(err_kind("List l = new ArrayList(); l.add(l); Map m = new HashMap(); m.put(l, 1); return m.size();"), "stack_overflow_error");
    // Not catchable (a Java Error).
    assert_eq!(err_kind("List l = new ArrayList(); l.add(l); try { return l.hashCode(); } catch (Exception e) { return -1; }"), "stack_overflow_error");
}

#[test]
fn exponential_shared_graphs_are_cut_off_promptly() {
    let started = std::time::Instant::now();
    // 60 levels, each holding the level below twice: 2^60 visits unbounded.
    let src = "List l = new ArrayList(); for (int i = 0; i < 60; i++) { List n = new ArrayList(); n.add(l); n.add(l); l = n; } return l.hashCode();";
    assert_eq!(err_kind(src), "stack_overflow_error");
    assert!(started.elapsed() < std::time::Duration::from_secs(30), "walk not bounded: {:?}", started.elapsed());
}

#[test]
fn direct_self_references_print_as_java_does() {
    assert_eq!(ok("List l = new ArrayList(); l.add(1); l.add(l); return l.toString();"), json!("[1, (this Collection)]"));
    assert_eq!(ok("Map m = new HashMap(); m.put('a', m); return m.toString();"), json!("{a=(this Map)}"));
    // Ordinary nesting is unaffected.
    assert_eq!(ok("return [[1, 2], [3]].hashCode() == [[1, 2], [3]].hashCode();"), json!(true));
}

// ---- 606: optional-section nesting is bounded ---------------------------

#[test]
fn deeply_nested_optional_sections_are_rejected() {
    let deep = format!("{}yyyy{}", "[".repeat(5000), "]".repeat(5000));
    let src = format!("return DateTimeFormatter.ofPattern('{deep}').format(LocalDate.of(2024, 1, 2));");
    assert_eq!(err_kind(&src), "illegal_argument_exception");
    // Unclosed sections too (closed implicitly).
    let src = format!("return DateTimeFormatter.ofPattern('{}yyyy').format(LocalDate.of(2024, 1, 2));", "[".repeat(5000));
    assert_eq!(err_kind(&src), "illegal_argument_exception");
    let fine = format!("{}yyyy{}", "[".repeat(100), "]".repeat(100));
    assert_eq!(ok(&format!("return DateTimeFormatter.ofPattern('{fine}').format(LocalDate.of(2024, 1, 2));")), json!("2024"));
}

// ---- 607: recursive callbacks are stack-guarded --------------------------

#[test]
fn recursive_lambdas_end_in_a_stack_overflow_error() {
    // No deadline: only the stack guard can stop it.
    // (Painless lambdas are call arguments: stored through a method call.)
    let src = "Map h = new HashMap(); h.put('f', n -> h.get('f').apply(n + 1)); return h.get('f').apply(0);";
    let worker = std::thread::Builder::new().stack_size(2 << 20).spawn(move || err_kind(src)).unwrap();
    assert_eq!(worker.join().expect("the worker thread must survive"), "stack_overflow_error");
    // Mutual recursion through two callbacks.
    let src = "Map h = new HashMap(); h.put('a', n -> h.get('b').apply(n)); h.put('b', n -> h.get('a').apply(n)); return h.get('a').apply(0);";
    let worker = std::thread::Builder::new().stack_size(2 << 20).spawn(move || err_kind(src)).unwrap();
    assert_eq!(worker.join().expect("the worker thread must survive"), "stack_overflow_error");
}

// ---- 608: lambdas run in their own frame --------------------------------

#[test]
fn a_lambda_passed_to_a_function_reads_its_own_captures() {
    // The review's witness, with the lambda reaching the helper through a
    // collection (Painless lambdas are call arguments).
    let src = "int useCallback(int replacement, def callback) { return callback.apply(2); } \
               int bias = 100; List cbs = new ArrayList(); cbs.add(x -> x + bias); return useCallback(7, cbs.get(0));";
    assert_eq!(ok(src), json!(102));
    // A smaller callee frame must not be indexed past its end.
    let src = "int f(def cb) { return cb.apply(1); } int a = 1; int b = 2; int c = 3; List cbs = new ArrayList(); cbs.add(x -> x + a + b + c); return f(cbs.get(0));";
    assert_eq!(ok(src), json!(7));
    // A closure made inside a function keeps the function's values after it returns.
    let src = "def adder(int n, List out) { out.add(x -> x + n); return null; } List fs = new ArrayList(); adder(5, fs); adder(9, fs); \
               return fs.get(0).apply(1) * 100 + fs.get(1).apply(1);";
    assert_eq!(ok(src), json!(610));
    // Nested callbacks and reentrant calls.
    let src = "int k = 10; List fs = new ArrayList(); fs.add(x -> [1, 2, 3].stream().mapToInt(y -> y * k + x).sum()); def outer = fs.get(0); return outer.apply(1) + outer.apply(2);";
    assert_eq!(ok(src), json!(129));
    // Captures are values at evaluation: a later reassignment is not seen.
    let src = "int bias = 1; List fs = new ArrayList(); fs.add(x -> x + bias); bias = 50; return fs.get(0).apply(1);";
    assert_eq!(ok(src), json!(2));
}

// ---- 609: dependent runtime fields share the outer execution ------------

#[test]
fn dependent_runtime_field_failures_keep_their_identity() {
    use super::error::ScriptErr;
    let breaker = ScriptErr::breaker(crate::error::EsError::new(429, "circuit_breaking_exception", "data too large"));
    let mapped = super::doc::dependent_failure("f", breaker);
    assert!(mapped.breaker.is_some(), "a breaker trip became a catchable exception");
    for kind in [super::runtime::TIME_EXCEEDED, "painless_error", "stack_overflow_error"] {
        assert_eq!(super::doc::dependent_failure("f", ScriptErr::new(kind, "x")).kind, kind);
    }
    let ordinary = super::doc::dependent_failure("f", ScriptErr::new("null_pointer_exception", "boom"));
    assert_eq!(ordinary.kind, "illegal_argument_exception");
    assert!(ordinary.msg.contains("[f]"));
}

#[test]
fn runs_hand_their_deadline_to_dependent_runtime_fields() {
    let mut fields = RuntimeFields::default();
    fields.parse_into(&json!({"dep": {"type": "long", "script": {"source": "emit(1)"}}})).unwrap();
    let prog = compile_arc("return 1;", ScriptContext::Score).unwrap();
    let mut e = Executor::new(prog, &json!({}), None, Some(std::sync::Arc::new(fields)));
    assert!(!e.rt.doc.rt_execs.is_empty());
    e.rt.cancel = Some(super::runtime::Cancel::new(std::time::Duration::from_secs(60)));
    e.run().unwrap();
    let (outer, inner) = (e.rt.cancel.as_ref().unwrap(), e.rt.doc.cancel.as_ref().expect("the doc context lost the deadline"));
    assert!(std::sync::Arc::ptr_eq(outer, inner));
}

// ---- 615: float semantics ------------------------------------------------

#[test]
fn float_arithmetic_rounds_to_single_precision() {
    assert_eq!(ok("float base = 16777216; return (base + 1) - base;"), json!(0.0));
    assert_eq!(ok("float amount = 16777217; return (long) amount;"), json!(16777216));
    assert_eq!(ok("return (long) 16777217f;"), json!(16777216));
    assert_eq!(ok("float f = 16777216; return f == 16777217;"), json!(true));
    assert_eq!(ok("float f = 0.1f; double d = f; return d;"), json!(0.10000000149011612));
    assert_eq!(ok("float f = 1; f += 16777216; return (long) f;"), json!(16777216));
    assert_eq!(ok("return (float) 16777217L == 16777216f;"), json!(true));
    // Double arithmetic is unchanged.
    assert_eq!(ok("double base = 16777216; return (base + 1) - base;"), json!(1.0));
    // double to float needs an explicit cast.
    assert_eq!(err_kind("float f = 1.5; return f;"), "class_cast_exception");
    assert_eq!(ok("return Float.parseFloat('16777217');"), json!(16777216.0));
}

// ---- 616: signed parsing never wraps -------------------------------------

#[test]
fn radix_parsing_and_decode_reject_out_of_range_values() {
    assert_eq!(err_kind("return Integer.parseInt('2147483648', 10);"), "number_format_exception");
    assert_eq!(err_kind("return Integer.parseInt('-2147483649', 10);"), "number_format_exception");
    assert_eq!(ok("return Integer.parseInt('-2147483648', 10);"), json!(-2147483648i64));
    assert_eq!(ok("return Integer.parseInt('7fffffff', 16);"), json!(2147483647));
    assert_eq!(err_kind("return Integer.decode('0x100000000');"), "number_format_exception");
    assert_eq!(ok("return Integer.decode('-0x80000000');"), json!(-2147483648i64));
    assert_eq!(ok("return Integer.decode('010');"), json!(8));
    assert_eq!(ok("return Long.decode('-0x8000000000000000');"), json!(i64::MIN));
    // A bad radix is a NumberFormatException, not a panic.
    assert_eq!(err_kind("return Integer.parseInt('7', 99);"), "number_format_exception");
    assert_eq!(err_kind("return Long.parseLong('7', 1);"), "number_format_exception");
    assert_eq!(err_kind("return Integer.parseInt('+-5', 10);"), "number_format_exception");
}

// ---- 617: localized formatters follow their locale ------------------------

#[test]
fn localized_formatters_use_the_locale_pattern() {
    assert_eq!(ok("return DateTimeFormatter.ofLocalizedTime(FormatStyle.SHORT).withLocale(Locale.UK).format(LocalTime.of(13, 45));"), json!("13:45"));
    assert_eq!(ok("return DateTimeFormatter.ofLocalizedDate(FormatStyle.SHORT).withLocale(Locale.UK).format(LocalDate.of(2024, 3, 5));"), json!("05/03/2024"));
    assert_eq!(ok("return DateTimeFormatter.ofLocalizedDate(FormatStyle.SHORT).withLocale(Locale.GERMANY).format(LocalDate.of(2024, 3, 5));"), json!("05.03.24"));
    assert_eq!(ok("return LocalTime.parse('13:45', DateTimeFormatter.ofLocalizedTime(FormatStyle.SHORT).withLocale(Locale.UK)).toString();"), json!("13:45"));
    // The US default is unchanged.
    assert_eq!(ok("return DateTimeFormatter.ofLocalizedDate(FormatStyle.SHORT).withLocale(Locale.US).format(LocalDate.of(2024, 3, 5));"), json!("3/5/24"));
}

// ---- 618: streams are lazy -----------------------------------------------

#[test]
fn stream_callbacks_run_only_for_consumed_elements() {
    assert_eq!(ok("List seen = new ArrayList(); [1, 2, 3].stream().peek(x -> seen.add(x)); return seen.size();"), json!(0));
    assert_eq!(ok("List seen = new ArrayList(); def r = [1, 2, 3].stream().peek(x -> seen.add(x)).limit(1).collect(Collectors.toList()); return [seen, r];"), json!([[1], [1]]));
    assert_eq!(ok("List seen = new ArrayList(); def f = [1, 2, 3].stream().map(x -> { seen.add(x); return x * 2; }).filter(x -> x > 2).findFirst().get(); return [seen, f];"), json!([[1, 2], 4]));
    // An exception from an element never consumed does not fail the pipeline.
    assert_eq!(ok("return [1, 0].stream().map(x -> 10 / x).findFirst().get();"), json!(10));
    // Sized count skips map / peek callbacks (JDK behaviour).
    assert_eq!(ok("List seen = new ArrayList(); long c = [1, 2, 3].stream().peek(x -> seen.add(x)).count(); return [c, seen.size()];"), json!([3, 0]));
    assert_eq!(ok("return [1, 2, 3, 4].stream().filter(x -> x % 2 == 0).count();"), json!(2));
    // Order of callbacks is per element, and sorted is a barrier.
    assert_eq!(ok("List log = new ArrayList(); [2, 1].stream().peek(x -> log.add('p' + x)).forEach(x -> log.add('f' + x)); return log;"), json!(["p2", "f2", "p1", "f1"]));
    assert_eq!(ok("List log = new ArrayList(); [2, 1].stream().peek(x -> log.add('p' + x)).sorted().forEach(x -> log.add('f' + x)); return log;"), json!(["p2", "p1", "f1", "f2"]));
    assert_eq!(ok("return [[1, 2], [3, 4]].stream().flatMap(l -> l.stream()).limit(3).collect(Collectors.toList());"), json!([1, 2, 3]));
    assert_eq!(ok("return [3, 1, 3, 2].stream().distinct().sorted().skip(1).collect(Collectors.toList());"), json!([2, 3]));
    assert_eq!(ok("return [1, 2, 3].stream().anyMatch(x -> x == 2);"), json!(true));
}

// ---- 623: Java regex defaults --------------------------------------------

#[test]
fn regex_classes_and_case_folding_default_to_ascii() {
    assert_eq!(ok("return '\u{0663}' ==~ /\\d+/;"), json!(false));
    assert_eq!(ok("return '\u{0663}' ==~ /\\d+/U;"), json!(true));
    assert_eq!(ok("return '\u{212A}' ==~ /k/i;"), json!(false));
    assert_eq!(ok("return '\u{212A}' ==~ /k/iu;"), json!(true));
    assert_eq!(ok("return 'K' ==~ /k/i;"), json!(true));
    assert_eq!(ok("return 'B' ==~ /[a-c]/i;"), json!(true));
    assert_eq!(ok("return '\u{00E9}' ==~ /\\w/;"), json!(false));
    assert_eq!(ok("return 'ab_9' ==~ /\\w+/;"), json!(true));
    assert_eq!(ok("return /(?<word>[a-z]+)/.matcher('hello').matches();"), json!(true));
}

#[test]
fn canonical_equivalence_matches_composed_and_decomposed_forms() {
    let (composed, decomposed) = ("\u{e9}", "e\u{301}");
    // Java `Pattern.CANON_EQ`: either form matches the other.
    assert_eq!(ok(&format!("return '{composed}' ==~ /{decomposed}/c;")), json!(true));
    assert_eq!(ok(&format!("return '{decomposed}' ==~ /{composed}/c;")), json!(true));
    assert_eq!(ok(&format!("return '{composed}' ==~ /{composed}/c;")), json!(true));
    // Without the flag they are different text.
    assert_eq!(ok(&format!("return '{composed}' ==~ /{decomposed}/;")), json!(false));
    // Offsets and groups are those of the original input.
    assert_eq!(ok(&format!("Matcher m = /{composed}/c.matcher('caf{decomposed}!'); m.find(); return [m.start(), m.end(), m.group()];")), json!([3, 5, decomposed]));
    assert_eq!(ok(&format!("Matcher m = /{decomposed}/c.matcher('caf{composed}!'); m.find(); return [m.start(), m.end(), m.group()];")), json!([3, 4, composed]));
    assert_eq!(ok(&format!("Matcher m = /(c)(a)(f)({composed})/c.matcher('caf{decomposed}'); m.find(); return [m.group(4), m.start(4), m.end(4)];")), json!([decomposed, 3, 5]));
    assert_eq!(ok(&format!("return /{composed}/c.matcher('caf{decomposed} caf{composed}').replaceAll('E');")), json!("cafE cafE"));
    // As in Java (the pattern, not the input, is rewritten): a lone `e`
    // matches the `e` of the decomposed form, never the composed one.
    assert_eq!(ok(&format!("return /e/c.matcher('{decomposed}').find();")), json!(true));
    assert_eq!(ok(&format!("return /e/c.matcher('{composed}').find();")), json!(false));
}

#[test]
fn canonical_equivalence_covers_classes_properties_and_quotes_like_java() {
    let (composed, decomposed) = ("\u{e9}", "e\u{301}");
    // `NFCCharProperty`: a class or property matches a spelling whose NFC
    // is one code point it accepts, as one unit.
    assert_eq!(ok(&format!("return '{decomposed}' ==~ /[{composed}]/c;")), json!(true));
    assert_eq!(ok(&format!("return '{decomposed}' ==~ /[{composed}]/;")), json!(false));
    assert_eq!(ok(&format!("return '{decomposed}' ==~ /[\u{e0}-\u{fa}]/c;")), json!(true));
    assert_eq!(ok(&format!("return '{decomposed}' ==~ /[^x]/c;")), json!(true));
    assert_eq!(ok(&format!("return '{decomposed}' ==~ /\\p{{L}}/c;")), json!(true));
    assert_eq!(ok(&format!("return '{decomposed}' ==~ /\\p{{L}}/;")), json!(false));
    assert_eq!(ok(&format!("return 'E\u{301}' ==~ /[{composed}]/ciu;")), json!(true));
    // The class text itself is composed (`normalizeClazz`).
    assert_eq!(ok(&format!("return '{composed}' ==~ /[{decomposed}]/c;")), json!(true));
    // Hangul: conjoining jamo compose to a syllable in the class.
    assert_eq!(ok("return '\u{1112}\u{1161}\u{11AB}' ==~ /[\u{AC00}-\u{D7A3}]/c;"), json!(true));
    assert_eq!(ok("return '\u{D558}\u{11AB}' ==~ /[\u{AC00}-\u{D7A3}]/c;"), json!(true));
    // Offsets are the input's.
    assert_eq!(ok(&format!("Matcher m = /[{composed}]/c.matcher('caf{decomposed}!'); m.find(); return [m.start(), m.end()];")), json!([3, 5]));
    // Literal text: every equivalent mark order and partial composition
    // (`produceEquivalentAlternation`), e.g. U+1EC7 as U+00EA U+0323.
    for input in ["\u{1EC7}", "e\u{323}\u{302}", "e\u{302}\u{323}", "\u{EA}\u{323}", "\u{1EB9}\u{302}"] {
        assert_eq!(ok(&format!("return '{input}' ==~ /\u{1EC7}/c;")), json!(true), "{input:?}");
    }
    // A singleton's own spelling and both canonical forms (U+212B).
    for input in ["\u{212B}", "\u{C5}", "A\u{30A}"] {
        assert_eq!(ok(&format!("return '{input}' ==~ /\u{212B}/c;")), json!(true), "{input:?}");
    }
    // `\Q...\E`: Java's rewrite does not know quotes, so a quoted composite
    // becomes its alternation text, quoted: literal `(?:é|é)`.
    assert_eq!(ok(&format!("return '{composed}' ==~ /\\Q{composed}\\E/c;")), json!(false));
    assert_eq!(ok(&format!("return '(?:{composed}|{decomposed})' ==~ /\\Q{composed}\\E/c;")), json!(true));
    assert_eq!(ok("return 'a.b' ==~ /\\Qa.b\\E/c;"), json!(true));
    assert_eq!(ok("return 'axb' ==~ /\\Qa.b\\E/c;"), json!(false));
    // ASCII patterns are unchanged.
    assert_eq!(ok("return 'abc' ==~ /[a-c]+/c;"), json!(true));
}

// ---- 624: OffsetTime durations are not wrapped into a day ----------------

#[test]
fn offset_time_durations_cross_midnight_correctly() {
    assert_eq!(ok("return ChronoUnit.HOURS.between(OffsetTime.parse('01:00Z'), OffsetTime.parse('01:00+02:00'));"), json!(-2));
    assert_eq!(ok("return ChronoUnit.HOURS.between(OffsetTime.parse('01:00+02:00'), OffsetTime.parse('01:00Z'));"), json!(2));
    assert_eq!(ok("return Duration.between(OffsetTime.parse('01:00Z'), OffsetTime.parse('01:00+02:00')).toString();"), json!("PT-2H"));
    assert_eq!(ok("return ChronoUnit.MINUTES.between(OffsetTime.parse('23:30Z'), OffsetTime.parse('00:30Z'));"), json!(-1380));
}

// ---- quadratic collection paths ------------------------------------------

#[test]
fn bulk_collection_operations_are_linear() {
    // 50k x 50k: billions of comparisons quadratically, milliseconds with an index.
    let started = std::time::Instant::now();
    let src = "List a = new ArrayList(); List b = new ArrayList(); for (int i = 0; i < 50000; i++) { a.add(i); b.add(i * 2); } \
               a.retainAll(b); List c = new ArrayList(a); c.removeAll(b); return [a.size(), c.size(), a.containsAll(b.subList(0, 100))];";
    assert_eq!(ok(src), json!([25000, 0, true]));
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "bulk operations took {:?}", started.elapsed());
    // Java equals semantics: an Integer is not a Long.
    assert_eq!(ok("List a = new ArrayList(); a.add(1L); a.removeAll([1]); return a.size();"), json!(1));
}

#[test]
fn hash_sets_are_constant_time() {
    // n insertions, n/3 lookups and removals: ~n^2/2 comparisons as a
    // scanned vector (4.5e10 at 300k, 1.8e9 at 60k), linear hashed. Debug
    // builds interpret several times slower, so they use the smaller n.
    let n: i64 = if cfg!(debug_assertions) { 60_000 } else { 300_000 };
    let started = std::time::Instant::now();
    let src = format!("Set s = new HashSet(); for (int i = 0; i < {n}; i++) {{ s.add(i); }} int hits = 0; \
               for (int i = 0; i < {n}; i += 3) {{ if (s.contains(i)) {{ hits++; }} s.remove(i); }} return [s.size(), hits, s.contains(1), s.contains(3)];");
    assert_eq!(ok(&src), json!([n - n / 3, n / 3, true, false]));
    assert!(started.elapsed() < std::time::Duration::from_secs(20), "set operations took {:?}", started.elapsed());
    // Java equals / hashCode: an Integer is not a Long; duplicates ignored;
    // LinkedHashSet keeps insertion order through removals.
    assert_eq!(ok("Set s = new HashSet(); s.add(1); s.add(1L); s.add(1); return s.size();"), json!(2));
    assert_eq!(ok("Set s = new LinkedHashSet(); s.add('c'); s.add('a'); s.add('b'); s.remove('a'); s.add('d'); return new ArrayList(s);"), json!(["c", "b", "d"]));
    assert_eq!(ok("Set s = new HashSet(); s.add([1, 2]); return s.contains([1, 2]);"), json!(true));
}

#[test]
fn deep_container_chains_do_not_exhaust_the_native_stack() {
    // 100k levels built by plain loops (inner-first and outer-first), on a
    // worker with an ordinary 2 MiB stack: serializing, hashing and dropping
    // them (at teardown, or mid-run when the last reference goes) must all
    // stay shallow.
    let worker = std::thread::Builder::new().stack_size(2 << 20).spawn(|| {
        let inner_first = "List l = new ArrayList(); for (int i = 0; i < 100000; i++) { List n = new ArrayList(); n.add(l); l = n; } return l;";
        let prog = compile_arc(inner_first, ScriptContext::PainlessTest).unwrap();
        let mut e = Executor::new(prog, &json!({}), None, None);
        let v = e.run().unwrap();
        let rejected = v.try_to_json().is_err();
        drop(v);
        drop(e);
        let hashed = err_kind("List l = new ArrayList(); for (int i = 0; i < 100000; i++) { List n = new ArrayList(); n.add(l); l = n; } return l.hashCode();");
        let dropped = ok("List root = new ArrayList(); List cur = root; for (int i = 0; i < 100000; i++) { List n = new ArrayList(); cur.add(n); cur = n; } root = null; cur = null; return 1;");
        let maps = ok("Map m = new HashMap(); for (int i = 0; i < 100000; i++) { Map n = new HashMap(); n.put('k', m); m = n; } m = null; return 2;");
        (rejected, hashed, dropped, maps)
    }).unwrap();
    let (rejected, hashed, dropped, maps) = worker.join().expect("the worker thread must survive");
    assert!(rejected, "a 100k-deep result serialized");
    assert_eq!(hashed, "stack_overflow_error");
    assert_eq!(dropped, json!(1));
    assert_eq!(maps, json!(2));
}
