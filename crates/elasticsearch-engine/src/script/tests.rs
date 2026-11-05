//! Painless snippet table: `(source, params, expected JSON result)`.

use super::*;
use serde_json::{json, Value};

fn run_with(src: &str, params: Value) -> std::result::Result<Value, EsError> {
    let cs = compile(src, &params, ScriptContext::PainlessTest)?;
    let mut e = cs.executor(None, None);
    e.run().map(|v| v.to_json()).map_err(|err| cs.error(&err))
}

#[allow(dead_code)]
fn run(src: &str) -> Value {
    match run_with(src, json!({"x": 10, "y": 2.5, "name": "Bob", "list": [3, 1, 2], "m": {"a": 1, "b": {"c": "deep"}}, "factor": 1.5})) {
        Ok(v) => v,
        Err(e) => panic!("script {src:?} failed: {}", e.to_json()),
    }
}

#[test]
fn snippet_table() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let cases: Vec<(&str, Value)> = vec![
        // literals & arithmetic
        ("1 + 2", json!(3)),
        ("5 / 2", json!(2)),
        ("5 / 2.0", json!(2.5)),
        ("5 % 3", json!(2)),
        ("-7 / 2", json!(-3)),
        ("-7 % 3", json!(-1)),
        ("2147483647 + 1", json!(-2147483648i64)),
        ("2147483647L + 1", json!(2147483648i64)),
        ("1.5f * 2", json!(3.0)),
        ("0x1F", json!(31)),
        ("1e3", json!(1000.0)),
        ("(1 + 2) * 3", json!(9)),
        ("1 << 4 | 1", json!(17)),
        ("-16 >> 2", json!(-4)),
        ("-16 >>> 28", json!(15)),
        ("~5", json!(-6)),
        ("7 & 3 ^ 1", json!(2)),
        ("10 - 2 - 3", json!(5)),
        ("2 * 3 + 4 * 5", json!(26)),
        ("1.0 / 0", json!(f64::INFINITY.to_string().replace("inf", "Infinity"))),
        // comparison / logic
        ("1 < 2 && 2 <= 2", json!(true)),
        ("1 > 2 || !(3 >= 4)", json!(true)),
        ("1 == 1L", json!(true)),
        ("1 == 1.0", json!(true)),
        ("'a' == 'a'", json!(true)),
        ("'a' != 'b'", json!(true)),
        ("null == null", json!(true)),
        ("def a = [1,2]; def b = [1,2]; a == b", json!(true)),
        ("def a = [1,2]; def b = [1,2]; a === b", json!(false)),
        ("true ? 'y' : 'n'", json!("y")),
        ("null ?: 'dflt'", json!("dflt")),
        ("'v' ?: 'dflt'", json!("v")),
        ("def m = null; m?.foo", Value::Null),
        ("def s = null; s?.length()", Value::Null),
        // variables & types
        ("int i = 5; i += 2; i", json!(7)),
        ("int i = 5; i /= 2; i", json!(2)),
        ("long l = 5; l * 3", json!(15)),
        ("double d = 1; d", json!(1.0)),
        ("int i = 3; i++; i", json!(4)),
        ("int i = 3; int j = i++; j", json!(3)),
        ("int i = 3; ++i", json!(4)),
        ("int i = 3; i--; --i", json!(1)),
        ("def x = 1; x = 'str'; x", json!("str")),
        ("int a = 1, b = 2; a + b", json!(3)),
        ("(int) 3.9", json!(3)),
        ("(long) -3.9", json!(-3)),
        ("(double) 3", json!(3.0)),
        ("(int) 'a'.charAt(0)", json!(97)),
        ("(char) 98", json!("b")),
        ("String s = 'x'; s instanceof String", json!(true)),
        ("def v = 5; v instanceof Integer", json!(true)),
        ("def v = 5L; v instanceof Integer", json!(false)),
        // strings
        ("'abc' + 1", json!("abc1")),
        ("'v' + 1.0", json!("v1.0")),
        ("'v' + 0.1", json!("v0.1")),
        ("'v' + null", json!("vnull")),
        ("1 + 2 + 'x'", json!("3x")),
        ("'hello'.length()", json!(5)),
        ("'hello'.substring(1, 3)", json!("el")),
        ("'hello'.substring(2)", json!("llo")),
        ("'Hello'.toLowerCase()", json!("hello")),
        ("'Hello'.toUpperCase()", json!("HELLO")),
        ("'hello'.indexOf('l')", json!(2)),
        ("'hello'.lastIndexOf('l')", json!(3)),
        ("'hello'.indexOf('z')", json!(-1)),
        ("'a.b.c'.splitOnToken('.')", json!(["a", "b", "c"])),
        ("'  x  '.trim()", json!("x")),
        ("'hello'.contains('ell')", json!(true)),
        ("'hello'.startsWith('he') && 'hello'.endsWith('lo')", json!(true)),
        ("'a-b'.replace('-', '+')", json!("a+b")),
        ("''.isEmpty()", json!(true)),
        ("'abc'.charAt(1)", json!("b")),
        ("'b'.compareTo('a')", json!(1)),
        ("'AbC'.equalsIgnoreCase('abc')", json!(true)),
        ("String.valueOf(5)", json!("5")),
        ("Integer.toString(42)", json!("42")),
        ("String.join('-', ['a','b'])", json!("a-b")),
        // numbers & statics
        ("Integer.parseInt('42') + 1", json!(43)),
        ("Long.parseLong('9999999999')", json!(9999999999i64)),
        ("Double.parseDouble('1.5') * 2", json!(3.0)),
        ("Math.max(3, 7)", json!(7.0)),
        ("Math.min(3L, 7)", json!(3.0)),
        ("Math.max(1.5, 2)", json!(2.0)),
        ("Math.abs(-4)", json!(4.0)),
        ("Math.pow(2, 10)", json!(1024.0)),
        ("Math.sqrt(16)", json!(4.0)),
        ("Math.log10(1000)", json!(3.0)),
        ("Math.floor(2.7)", json!(2.0)),
        ("Math.ceil(2.1)", json!(3.0)),
        ("Math.round(2.5)", json!(3)),
        ("Math.round(-2.5)", json!(-2)),
        ("Math.PI > 3.14", json!(true)),
        ("Integer.MAX_VALUE", json!(2147483647)),
        ("Long.MIN_VALUE", json!(i64::MIN)),
        ("Double.isNaN(0.0 / 0)", json!(true)),
        ("5.intValue()", json!(5)),
        ("def d = 7.9; d.intValue()", json!(7)),
        ("Boolean.parseBoolean('TRUE')", json!(true)),
        // lists
        ("[1, 2, 3].size()", json!(3)),
        ("def l = [1, 2]; l.add(3); l", json!([1, 2, 3])),
        ("def l = new ArrayList(); l.add('a'); l.add(0, 'b'); l", json!(["b", "a"])),
        ("def l = [5, 6, 7]; l[1]", json!(6)),
        ("def l = [5, 6, 7]; l[-1]", json!(7)),
        ("def l = [5, 6, 7]; l[0] = 9; l", json!([9, 6, 7])),
        ("def l = [5, 6, 7]; l.remove(0); l", json!([6, 7])),
        ("def l = ['a', 'b']; l.remove('a'); l", json!(["b"])),
        ("[1, 2, 3].contains(2)", json!(true)),
        ("[1, 2, 3].indexOf(3)", json!(2)),
        ("[].isEmpty()", json!(true)),
        ("def l = [3, 1, 2]; l.sort(null); l", json!([1, 2, 3])),
        ("def l = [3, 1, 2]; Collections.sort(l); l", json!([1, 2, 3])),
        ("def l = [3, 1, 2]; l.sort((a, b) -> b - a); l", json!([3, 2, 1])),
        ("def l = [1, 2, 3, 4]; l.removeIf(x -> x % 2 == 0); l", json!([1, 3])),
        ("[1, 2, 3].stream().filter(x -> x > 1).count()", json!(2)),
        ("[1, 2, 3].stream().map(x -> x * 2).collect(Collectors.toList())", json!([2, 4, 6])),
        ("[1, 2, 3].stream().mapToInt(x -> x).sum()", json!(6)),
        ("[1.5, 2.5].stream().mapToDouble(x -> x).sum()", json!(4.0)),
        ("[4, 2].stream().mapToInt(x -> x).average().getAsDouble()", json!(3.0)),
        ("[4, 9, 2].stream().max(Integer::compare).get()", json!(9)),
        ("['b', 'a'].stream().sorted().collect(Collectors.joining(','))", json!("a,b")),
        ("[1, 2, 3].stream().anyMatch(x -> x > 2)", json!(true)),
        ("[1, 2, 3].stream().allMatch(x -> x > 2)", json!(false)),
        ("[1, 1, 2].stream().distinct().count()", json!(2)),
        ("[1, 2, 3].stream().reduce(0, (a, b) -> a + b)", json!(6)),
        ("[1, 2, 3].sum()", json!(6)),
        ("['a', 'b'].join('|')", json!("a|b")),
        ("[1, 2, 3].findAll(x -> x > 1)", json!([2, 3])),
        ("[1, 2, 3].collect(x -> x * 10)", json!([10, 20, 30])),
        ("[1, 2, 3].any(x -> x == 2)", json!(true)),
        ("new int[3]", json!([0, 0, 0])),
        ("int[] a = new int[] {1, 2}; a.length", json!(2)),
        ("String[] a = new String[2]; a[0] = 'x'; a", json!(["x", null])),
        ("def l = [1,2,3,4]; l.subList(1, 3)", json!([2, 3])),
        ("def s = new HashSet(); s.add(1); s.add(1); s.add(2); s.size()", json!(2)),
        // maps
        ("def m = [:]; m.put('a', 1); m", json!({"a": 1})),
        ("def m = ['a': 1, 'b': 2]; m.b", json!(2)),
        ("def m = ['a': 1]; m['a'] += 5; m.a", json!(6)),
        ("def m = ['a': 1]; m.c = 3; m", json!({"a": 1, "c": 3})),
        ("def m = ['a': 1]; m.containsKey('a') && !m.containsKey('z')", json!(true)),
        ("def m = ['a': 1]; m.getOrDefault('z', 9)", json!(9)),
        ("def m = ['a': 1, 'b': 2]; m.remove('a'); m", json!({"b": 2})),
        ("def m = ['a': 1, 'b': 2]; def t = 0; for (def k : m.keySet()) { t += m[k]; } t", json!(3)),
        ("def m = ['a': 1, 'b': 2]; def s = ''; for (def e : m.entrySet()) { s += e.getKey() + e.getValue(); } s", json!("a1b2")),
        ("def m = ['a': 1]; m.values()", json!([1])),
        ("def m = new HashMap(); m.computeIfAbsent('k', k -> []).add(1); m", json!({"k": [1]})),
        ("def m = ['x': 1]; m.merge('x', 5, (a, b) -> a + b); m.x", json!(6)),
        ("def m = [:]; m.size()", json!(0)),
        ("def m = ['a': ['b': 3]]; m.a.b", json!(3)),
        // control flow
        ("int t = 0; for (int i = 0; i < 5; i++) { t += i; } t", json!(10)),
        ("int t = 0; for (def x : [1, 2, 3]) { t += x } return t;", json!(6)),
        ("int t = 0; for (x in [1, 2, 3]) { t += x } t", json!(6)),
        ("int i = 0; while (i < 10) { i += 3; } i", json!(12)),
        ("int i = 0; do { i++; } while (i < 3); i", json!(3)),
        ("int t = 0; for (int i = 0; i < 10; i++) { if (i == 5) break; if (i % 2 == 0) continue; t += i; } t", json!(4)),
        ("if (1 > 2) { return 'a'; } else if (2 > 1) { return 'b'; } else { return 'c'; }", json!("b")),
        ("if (false) { 1 } else { 2 }", json!(2)),
        ("def x = 5; if (x > 3) return 'big'; return 'small';", json!("big")),
        ("return;", Value::Null),
        // functions & lambdas
        ("int sq(int x) { return x * x; } sq(4)", json!(16)),
        ("int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); } fib(10)", json!(55)),
        ("[1, 2, 3].stream().map(String::valueOf).collect(Collectors.toList())", json!(["1", "2", "3"])),
        ("['a', 'bb'].stream().mapToInt(String::length).sum()", json!(3)),
        // params
        ("params.x * 2", json!(20)),
        ("params['x'] + params.y", json!(12.5)),
        ("params.name + '!'", json!("Bob!")),
        ("params.list.size()", json!(3)),
        ("params.m.b.c", json!("deep")),
        ("params.missing == null", json!(true)),
        ("params.containsKey('x')", json!(true)),
        ("params.get('name').length()", json!(3)),
        // dates
        ("ZonedDateTime.parse('2020-03-05T10:11:12Z').getYear()", json!(2020)),
        ("ZonedDateTime.parse('2020-03-05T10:11:12Z').getDayOfWeek()", json!("THURSDAY")),
        ("ZonedDateTime.parse('2020-03-05T10:11:12Z').getDayOfWeek().getValue()", json!(4)),
        ("ZonedDateTime.parse('2020-03-05T10:11:12Z').getDayOfWeek() == DayOfWeek.THURSDAY", json!(true)),
        ("ZonedDateTime.parse('2020-03-05T10:11:12Z').toInstant().toEpochMilli()", json!(1583403072000i64)),
        ("ZonedDateTime.parse('2020-03-05T10:11:12Z').plusDays(1).getDayOfMonth()", json!(6)),
        ("ZonedDateTime.parse('2020-03-05T10:11:12Z').monthValue", json!(3)),
        ("ChronoUnit.DAYS.between(ZonedDateTime.parse('2020-01-01T00:00:00Z'), ZonedDateTime.parse('2020-01-31T00:00:00Z'))", json!(30)),
        ("Instant.ofEpochMilli(0)", json!("1970-01-01T00:00:00Z")),
        ("ZonedDateTime.parse('2020-03-05T10:11:00Z')", json!("2020-03-05T10:11Z")),
        // misc
        ("def sb = new StringBuilder(); sb.append('a').append(1); sb.toString()", json!("a1")),
        ("try { throw new IllegalArgumentException('boom'); } catch (IllegalArgumentException e) { return e.getMessage(); }", json!("boom")),
        ("try { [].get(3); } catch (Exception e) { return 'caught'; }", json!("caught")),
        ("// comment\n/* block */ 1", json!(1)),
        ("def x = 1; x == 1 ? 'one' : 'other'", json!("one")),
        ("Objects.equals(null, null)", json!(true)),
        ("Optional.ofNullable(null).orElse(3)", json!(3)),
    ];
    let mut failures = Vec::new();
    for (src, want) in &cases {
        match run_with(src, json!({"x": 10, "y": 2.5, "name": "Bob", "list": [3, 1, 2], "m": {"a": 1, "b": {"c": "deep"}}})) {
            Ok(got) => {
                let ok = got == *want || (got.is_number() && want.is_number() && got.as_f64() == want.as_f64() && got.is_f64() == want.is_f64());
                if !ok {
                    failures.push(format!("{src:?}: got {got}, want {want}"));
                }
            }
            Err(e) => failures.push(format!("{src:?}: error {}", e.to_json())),
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// Painless `type: ID (DOT DOTID)*`: inner-class and fully qualified
/// type names in declarations, casts, `instanceof`, `new` and arrays (Rally
/// geoshape's `ScriptDocValues.Geometry geometry = doc["shape"]`).
#[test]
fn dotted_type_names() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (src, want) in [
        ("ScriptDocValues.Geometry g = null; return g == null", json!(true)),
        ("ScriptDocValues.Geometry g; g = null; return 1", json!(1)),
        ("ScriptDocValues.Geometry[] a = new ScriptDocValues.Geometry[2]; return a.length", json!(2)),
        ("java.util.List l = new java.util.ArrayList(); l.add(4); return l.size()", json!(1)),
        ("java.util.Map m = new java.util.HashMap(); m.put('k', 2); return m.get('k')", json!(2)),
        ("Map m = ['a': 1]; for (Map.Entry e : m.entrySet()) { return e.getKey() } return null", json!("a")),
        ("Map m = ['a': 1]; for (java.util.Map.Entry e : m.entrySet()) { return e instanceof Map.Entry } return null", json!(true)),
        ("Map m = ['a': 7]; def e = m.entrySet().iterator().next(); return ((Map.Entry) e).getValue()", json!(7)),
        ("def x = [1]; return x instanceof java.util.List", json!(true)),
        ("def x = 'a'; return x instanceof ScriptDocValues.Geometry", json!(false)),
        // Not types: field and parameter access keep working.
        ("return params.x + params.m.b.c.length()", json!(14)),
        // Painless whitelists only `double max(double,double)` (java.lang.txt), so ints widen.
        ("return Math.max(1, 2)", json!(2.0)),
    ] {
        match run_with(src, json!({"x": 10, "m": {"b": {"c": "deep"}}})) {
            Ok(v) => assert_eq!(v, want, "{src}"),
            Err(e) => panic!("{src}: {}", e.to_json()),
        }
    }
    // An unknown dotted name is still not a type.
    let e = run_with("Foo.Bar b = null; return 1", json!({})).expect_err("unknown type");
    assert_eq!(e.reason, "compile error");
}

#[test]
fn compile_errors() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (src, needle) in [
        ("1 +", "unexpected end of script"),
        ("undefinedVar + 1", "cannot resolve symbol [undefinedVar]"),
        ("int x = 'str';", "Cannot cast from [java.lang.String] to [int]"),
        ("int x = 1; int x = 2;", "already defined"),
        ("break;", "invalid break"),
        ("foo(1)", "Unknown call [foo] with [1] arguments"),
        ("if (1) { 2 }", "Cannot cast from [int] to [boolean]"),
        ("'a' - 1", "Cannot cast from [java.lang.String] to [int]"),
        ("new Foo()", "cannot resolve symbol [Foo]"),
        ("Math.nope(1)", "static method [java.lang.Math, nope/1] not found"),
    ] {
        let e = run_with(src, json!({})).expect_err(src);
        assert_eq!(e.kind, "script_exception", "{src}");
        assert_eq!(e.reason, "compile error", "{src}");
        let cause = e.caused_by.as_ref().unwrap();
        assert!(cause.reason.contains(needle), "{src}: {}", cause.reason);
        let j = e.to_json();
        assert_eq!(j["error"]["lang"], "painless");
        assert!(j["error"]["script_stack"].is_array());
        assert!(j["error"]["position"]["offset"].is_number(), "{src}: {j}");
    }
}

#[test]
fn runtime_errors() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (src, kind, needle) in [
        ("def x = null; x.length()", "null_pointer_exception", ""),
        ("[1, 2].get(5)", "index_out_of_bounds_exception", "out of bounds"),
        // `1 / 0` is constant-folded into a compile error by ES; keep the divisor dynamic.
        ("def z = 0; 1 / z", "arithmetic_exception", "/ by zero"),
        ("Integer.parseInt('x1')", "number_format_exception", "For input string"),
        ("def x = 'a'; x.foo()", "illegal_argument_exception", "dynamic method [java.lang.String, foo/0] not found"),
        ("throw new IllegalStateException('bad')", "illegal_state_exception", "bad"),
        ("while (true) {}", "painless_error", "maximum number of statements"),
        ("def x = 'a'; if (x) { 1 }", "class_cast_exception", "cannot cast def [java.lang.String] to boolean"),
    ] {
        let e = run_with(src, json!({})).expect_err(src);
        assert_eq!(e.reason, "runtime error", "{src}");
        let c = e.caused_by.as_ref().unwrap();
        assert_eq!(c.kind, kind, "{src}: {}", c.reason);
        assert!(c.reason.contains(needle), "{src}: {}", c.reason);
        let j = e.to_json();
        let stack = j["error"]["script_stack"].as_array().unwrap();
        // JDK frames (as ES prints them) may precede the script snippet + pointer.
        assert!(stack.len() >= 2, "{src}: {j}");
        assert!(stack.iter().take(stack.len() - 2).all(|f| f.as_str().unwrap().starts_with("java.base/")), "{src}: {j}");
        assert!(stack[stack.len() - 1].as_str().unwrap().ends_with("^---- HERE"));
    }
}

#[test]
fn error_json_shape() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let e = run_with("int x = 1;\nx.foo();", json!({})).unwrap_err();
    let j = e.to_json();
    assert_eq!(j["status"], 400);
    let err = &j["error"];
    assert_eq!(err["root_cause"][0]["type"], "script_exception");
    assert_eq!(err["type"], "script_exception");
    assert_eq!(err["reason"], "runtime error");
    assert_eq!(err["script"], "int x = 1;\nx.foo();");
    assert_eq!(err["lang"], "painless");
    assert_eq!(err["caused_by"]["type"], "illegal_argument_exception");
    assert_eq!(err["script_stack"][0], "x.foo();");
    assert_eq!(err["script_stack"][1], " ^---- HERE");
    assert_eq!(err["position"], json!({"offset": 12, "start": 11, "end": 19}));
}

#[test]
fn cache_reuses_programs() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let a = program("params.q + 1", ScriptContext::Score).unwrap();
    let b = program("params.q + 1", ScriptContext::Score).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    let c1 = compile("params.q + 1", &json!({"q": 1}), ScriptContext::Score).unwrap();
    let c2 = compile("params.q + 1", &json!({"q": 41}), ScriptContext::Score).unwrap();
    assert!(Arc::ptr_eq(&c1.program, &c2.program));
    assert_eq!(c2.executor(None, None).run().unwrap().to_json(), json!(42));
}

#[test]
fn ctx_scripts() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let cs = compile("ctx._source.count += params.n; if (ctx._source.count > 5) { ctx.op = 'delete' }", &json!({"n": 3}), ScriptContext::Update).unwrap();
    let meta = update::DocMeta { index: "i".into(), id: Some("1".into()), ..Default::default() };
    let out = update::run_doc_script(&cs, json!({"count": 1, "tags": ["a"]}), &meta, "index").unwrap();
    assert_eq!(out.op, update::DocOp::Index);
    assert_eq!(out.source, json!({"count": 4, "tags": ["a"]}));
    let out = update::run_doc_script(&cs, json!({"count": 4}), &meta, "index").unwrap();
    assert_eq!(out.op, update::DocOp::Delete);
    let cs = compile("ctx._source.tags.add(params.t); ctx._source.remove('gone'); ctx._id = 'new'", &json!({"t": "b"}), ScriptContext::Reindex).unwrap();
    let out = update::run_doc_script(&cs, json!({"tags": ["a"], "gone": 1}), &meta, "index").unwrap();
    assert_eq!(out.source, json!({"tags": ["a", "b"]}));
    assert_eq!(out.meta.id.as_deref(), Some("new"));
    let cs = compile("ctx.op = 'bogus'", &json!({}), ScriptContext::Update).unwrap();
    assert!(update::run_doc_script(&cs, json!({}), &meta, "index").is_err());
}

#[test]
fn user_function_context_names_are_local() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let script = compile(
        "boolean is_user_trusted(def ctx, def users) { return users.contains(ctx.user); } ctx.trusted = is_user_trusted(ctx, params.users);",
        &json!({"users":["alice"]}), ScriptContext::Ingest,
    ).unwrap();
    let result = execute_ctx(&script, json!({"user":"alice"}).as_object().unwrap().clone()).unwrap();
    assert_eq!(result["trusted"], true);
    let result = execute_ctx(&script, json!({"user":"bob"}).as_object().unwrap().clone()).unwrap();
    assert_eq!(result["trusted"], false);
    assert_eq!(run("def f(def params, def doc, double _score) { return params.x + doc.x + _score; } f(['x':1], ['x':2], 3)"), json!(6.0));
    assert_eq!(run("int f() { int ctx = 4; return ctx; } f()"), json!(4));
    for source in [
        "def f() { return params.x; } f()",
        "def f() { return ctx; } f()",
        "def f(def ctx, def ctx) { return ctx; } f(1,2)",
        "def ctx = 1; ctx",
    ] {
        assert!(compile(source, &json!({}), ScriptContext::Ingest).is_err(), "{source}");
    }
}

#[test]
fn jmisc_library() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (src, want) in [
        ("StringJoiner j = new StringJoiner(', ', '[', ']'); j.add('a').add('b'); j.toString() + '|' + j.length()", json!("[a, b]|6")),
        ("StringJoiner j = new StringJoiner('-'); j.setEmptyValue('EMPTY'); j.toString()", json!("EMPTY")),
        ("StringJoiner a = new StringJoiner(','); a.add('x'); StringJoiner b = new StringJoiner('+'); b.add('y').add('z'); a.merge(b).toString()", json!("x,y+z")),
        ("UUID u = UUID.fromString('123e4567-e89b-12d3-a456-426614174000'); u.toString() + '|' + u.version() + '|' + u.variant()", json!("123e4567-e89b-12d3-a456-426614174000|1|2")),
        ("UUID.fromString('123e4567-e89b-12d3-a456-426614174000').getMostSignificantBits()", json!(1314564453825188563i64)),
        ("UUID.fromString('00000000-0000-0001-0000-000000000002').compareTo(UUID.fromString('00000000-0000-0001-0000-000000000001'))", json!(1)),
        ("new UUID(1L, 2L).toString()", json!("00000000-0000-0001-0000-000000000002")),
        ("UUID.randomUUID().toString().length()", json!(36)),
        ("def l = [3, 1, 2]; l.sort(Collections.reverseOrder()); l", json!([3, 2, 1])),
        ("def l = ['bb', 'a', 'ccc']; l.sort(Comparator.comparing(s -> s.length()).reversed()); l", json!(["ccc", "bb", "a"])),
        ("def it = [1, 2, 3].iterator(); def out = []; while (it.hasNext()) { out.add(it.next() * 10) } out", json!([10, 20, 30])),
        ("def m = MathContext.DECIMAL32; m instanceof MathContext", json!(true)),
        ("[RoundingMode.UP instanceof Enum, DayOfWeek.MONDAY instanceof Enum, MathContext.DECIMAL32 instanceof Enum]", json!([true, true, false])),
    ] {
        assert_eq!(run(src), want, "{src}");
    }
    let e = run_with("UUID.fromString('zz')", json!({})).unwrap_err();
    assert_eq!(e.caused_by.as_ref().unwrap().reason, "Invalid UUID string: zz");
    let e = run_with("def it = [1].iterator(); it.next(); it.next()", json!({})).unwrap_err();
    assert_eq!(e.caused_by.as_ref().unwrap().kind, "no_such_element_exception");
}

#[test]
fn string_format() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (src, want) in [
        ("String.format('%05d|%-4s|%x', new def[] {42, 'a', 255})", "00042|a   |ff"),
        ("String.format('%.1f %.0f %.2f %.20f', new def[] {0.05, 2.5, 1.005, 0.1})", "0.1 3 1.01 0.10000000000000000000"),
        ("String.format('%g|%g|%g|%G|%.3g|%g', new def[] {0.00012345, 123456789.0, 1.0, 1e-5, 0.0001, 0.0})", "0.000123450|1.23457e+08|1.00000|1.00000E-05|0.000100|0.00000"),
        ("String.format('%#x|%#o|%x|%X', new def[] {255, 8, -1, -1L})", "0xff|010|ffffffff|FFFFFFFFFFFFFFFF"),
        ("String.format('%,.2f|%(,.2f|% d|%(d|%+.1e|%e|%E', new def[] {1234567.891, -1234.5, 5, -5, 12345.678, 0.0, -1e100})", "1,234,567.89|(1,234.50)| 5|(5)|+1.2e+04|0.000000e+00|-1.000000E+100"),
        ("String.format('%f %f %f %e', new def[] {Double.NaN, Double.POSITIVE_INFINITY, -1.0/0, Double.NaN})", "NaN Infinity -Infinity NaN"),
        ("String.format('%b %B %b %b %.2b %5b', new def[] {null, 'x', false, 0, true, true})", "false TRUE false true tr  true"),
        ("String.format('%h %H %h %h', new def[] {'hi', 'hi', null, 17})", "d01 D01 null 11"),
        ("String.format('%.2s|%5.1s|%-5S|', new def[] {'abc', 'xyz', 'q'})", "ab|    x|Q    |"),
        ("String.format('%d %x %o %,d', new def[] {new BigInteger('-123456789012345678901234567890'), new BigInteger('-255'), new BigInteger('8'), new BigInteger('1234567')})", "-123456789012345678901234567890 -ff 10 1,234,567"),
        ("String.format('%.3f %e %g %s', new def[] {new BigDecimal('1.23456'), new BigDecimal('12345.678'), new BigDecimal('0.0001'), new BigDecimal('1E+3')})", "1.235 1.234568e+04 0.000100000 1E+3"),
        ("String.format('%a %A %a %a %.2a', new def[] {1.0, -0.5, 0.0, Double.MIN_VALUE, 3.14159})", "0x1.0p0 -0X1.0P-1 0x0.0p0 0x0.0000000000001p-1022 0x1.92p1"),
        ("String.format('%,d %,d %,10d %,010d', new def[] {-1234567, 123, 1234, -1234})", "-1,234,567 123      1,234 -00001,234"),
        ("String.format('%5%|%-5%|', new def[] {})", "    %|%    |"),
        ("String.format('%-+5d|%+05d|%+(d|% (d|%(5d', new def[] {3, 3, -3, 3, -3})", "+3   |+0003|(3)| 3|  (3)"),
        ("String.format('%#.0e|%#.0f|%+.2f|% .2f', new def[] {3.0, 3.0, 1.0, 1.0})", "3.e+00|3.|+1.00| 1.00"),
        ("String.format('%010.3e|%-12.2e|', new def[] {-12.5, 12.5})", "-1.250e+01|1.25e+01    |"),
        ("String.format('%#X|%08x|%#010x|%(x', new def[] {255, 255, 255, new BigInteger('-255')})", "0XFF|000000ff|0x000000ff|(ff)"),
        ("String.format('%,.3f|%.3f|%.1f', new def[] {0.0005, -0.0005, -0.04})", "0.001|-0.001|-0.0"),
        ("String.format('%1$s %s %s %2$s %<s', new def[] {'a', 'b'})", "a a b b b"),
        ("String.format('%.0e|%.1e|%e|%e', new def[] {9.5, 9.95, 1.0E-300, 123.0})", "1e+01|1.0e+01|1.000000e-300|1.230000e+02"),
        ("String.format('%.3g|%.3g|%g|%g', new def[] {99950.0, 0.000099995, 1e-4, 999999.5})", "1.00e+05|0.000100|0.000100000|1.00000e+06"),
        ("String.format('%tF %tT %TB', new def[] {1583402472123L, 1583402472123L, 1L})", "2020-03-05 10:01:12 JANUARY"),
        ("String.format('%s', (def[]) null)", "null"),
    ] {
        assert_eq!(run(src), json!(want), "{src}");
    }
    for (src, kind, reason) in [
        ("String.format('%d', new def[] {1.5})", "illegal_format_conversion_exception", "d != java.lang.Double"),
        ("String.format('%q', new def[] {1})", "unknown_format_conversion_exception", "Conversion = 'q'"),
        ("String.format('%', new def[] {1})", "unknown_format_conversion_exception", "Conversion = '%'"),
        ("String.format('%5', new def[] {1})", "unknown_format_conversion_exception", "Conversion = '5'"),
        ("String.format('%.2d', new def[] {1})", "illegal_format_precision_exception", "2"),
        ("String.format('%#d', new def[] {1})", "format_flags_conversion_mismatch_exception", "Conversion = d, Flags = #"),
        ("String.format('%-d', new def[] {1})", "missing_format_width_exception", "%-d"),
        ("String.format('%-05d', new def[] {1})", "illegal_format_flags_exception", "Flags = '-0'"),
        ("String.format('%--5d', new def[] {1})", "duplicate_format_flags_exception", "Flags = '-'"),
        ("String.format('%<s', new def[] {'a'})", "missing_format_argument_exception", "Format specifier '%<s'"),
        ("String.format('%3$s', new def[] {'a'})", "missing_format_argument_exception", "Format specifier '%3$s'"),
        ("String.format('%s %s', new def[] {'a'})", "missing_format_argument_exception", "Format specifier '%s'"),
        ("String.format('%c', new def[] {-5})", "illegal_format_code_point_exception", "Code point = 0xfffffffb"),
        ("String.format('%c', new def[] {'ab'})", "illegal_format_conversion_exception", "c != java.lang.String"),
        ("String.format('%tq', new def[] {1L})", "unknown_format_conversion_exception", "Conversion = 'tq'"),
        ("String.format('%5n', new def[] {})", "illegal_format_width_exception", "5"),
        ("String.format('%0$s', new def[] {'a'})", "illegal_format_argument_index_exception", "Illegal format argument index = 0"),
    ] {
        let e = run_with(src, json!({})).expect_err(src);
        let c = e.caused_by.as_ref().unwrap();
        assert_eq!((c.kind.as_str(), c.reason.as_str()), (kind, reason), "{src}");
    }
}

/// Runtime errors in statically bound calls / constructors are reported at
/// the last position ES's bytecode records (the call's `.`, then each
/// argument), `def` calls at their `.`.
#[test]
fn runtime_error_positions() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (src, offset) in [
        ("new BigDecimal('1').divide(new BigDecimal('0'))", 27),
        ("new BigDecimal('x')", 0),
        ("def y = 1; new BigDecimal('x')", 11),
        ("def s = 'x'; new BigDecimal(s)", 28),
        ("def x = new BigDecimal('1'); x.divide(new BigDecimal('0'))", 38),
        ("new BigDecimal('1').divide(BigDecimal.ZERO)", 37),
        ("BigDecimal a = new BigDecimal('1'); BigDecimal b = BigDecimal.ZERO; a.divide(b)", 69),
        ("BigDecimal.valueOf(1).divide(BigDecimal.valueOf(0))", 39),
        ("String s = 'abc'; s.substring(Integer.parseInt('5'))", 37),
        ("def x = 6; 'abc'.substring(1 + x)", 27),
        ("new BigDecimal('1.5').setScale(0, RoundingMode.UNNECESSARY)", 46),
        ("new BigDecimal('1').add(new BigDecimal('2')).divide(new BigDecimal('0').add(new BigDecimal('0')))", 76),
        ("String.format('%d', new def[] {1.5})", 31),
        ("String.format('%-%|%5%|%n', new def[] {})", 28),
        ("def l = [1]; l.get(5)", 14),
    ] {
        let e = run_with(src, json!({})).expect_err(src);
        assert_eq!(e.to_json()["error"]["position"]["offset"], json!(offset), "{src}");
    }
    for (src, offset, reason) in [
        ("new BigDecimal('1').divide(new BigDecimal('0'), 2, RoundingMode.UNNECESSARY)", 19, "member method [java.math.BigDecimal, divide/3] not found"),
        ("new BigDecimal(1, new MathContext(2)).divide(new BigDecimal(0))", 15, "Cannot cast from [int] to [java.lang.String]."),
        ("new BigDecimal(0.1)", 15, "Cannot cast from [double] to [java.lang.String]."),
    ] {
        let e = run_with(src, json!({})).expect_err(src);
        assert_eq!(e.reason, "compile error", "{src}");
        assert_eq!(e.caused_by.as_ref().unwrap().reason, reason, "{src}");
        assert_eq!(e.to_json()["error"]["position"]["offset"], json!(offset), "{src}");
    }
}

/// The unboxed numeric path (`compile::num`) agrees with the general one,
/// including where it must bail (strings, `/ by zero`).
#[test]
fn numeric_fast_path() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let cases: Vec<(&str, Value)> = vec![
        ("params.x + params.y", json!(12.5)),
        ("params.x * 3", json!(30)),
        ("params.x * 3000000000L", json!(30000000000i64)),
        ("2147483647 + params.x", json!(-2147483639)),
        ("params.x % 3", json!(1)),
        ("-params.x", json!(-10)),
        ("-params.x / 4", json!(-2)),
        ("(int) (0.0 / 0.0) + params.x", json!(10)),
        ("(int) (1e20 * params.y)", json!(2147483647)),
        ("(long) params.y", json!(2)),
        ("(float) 0.1 * params.x", json!(1.0000000149011612)),
        ("Math.abs(-params.x)", json!(10.0)),
        ("Math.round(params.y)", json!(3)),
        ("Math.max(params.x, 3)", json!(10.0)),
        ("Math.log(params.x) / params.x", json!(10f64.ln() / 10.0)),
        ("params.x > params.y", json!(true)),
        ("params.x <= 9", json!(false)),
        ("params.name + params.x", json!("Bob10")),
        ("Math.sqrt(16)", json!(4.0)),
        ("params.factor * params.list.size()", json!(4.5)),
    ];
    for (src, want) in cases {
        assert_eq!(run(src), want, "{src}");
    }
    let err = run_with("params.x / (params.x - 10)", json!({"x": 10})).unwrap_err().to_json().to_string();
    assert!(err.contains("by zero"), "{err}");
    let err = run_with("(int) params.s + 1", json!({"s": "a"})).unwrap_err().to_json().to_string();
    assert!(err.contains("class_cast_exception"), "{err}");
}


#[test]
fn random_typed_declaration_seeded_sequence_and_aliasing() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert_eq!(run("Random r = new Random(1); return [r.nextLong(), r.nextLong()];"), json!([-4964420948893066024i64, 7564655870752979346i64]));
    assert_eq!(run("Random r = new Random(1); def alias = r; long a = alias.nextLong(); r.setSeed(1); return [a == r.nextLong(), r.nextLong()];"), json!([true, 7564655870752979346i64]));
    assert_eq!(run("Random r = new Random(1); return r.nextInt(10);"), json!(5));
    assert!(run_with("Random r = new Random(1); return r.nextInt(0);", json!({})).is_err());
    let value = run("Random r = new Random(); return 'value-' + r.nextLong();");
    assert!(value.as_str().unwrap().strip_prefix("value-").unwrap().parse::<i64>().is_ok());
}

#[test]
fn substring_preserves_split_utf16_pairs() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("params.s.substring(1, 2)", vec![0xd83d]),
        ("params.s.substring(2, 3)", vec![0xde00]),
        ("params.s.substring(3)", vec![0x42]),
        ("params.s.substring(1, 2) + params.s.substring(2, 3)", vec![0xd83d, 0xde00]),
    ] {
        let cs = compile(script, &json!({"s":"A😀B"}), ScriptContext::PainlessTest).unwrap();
        let mut exec = cs.executor(None, None);
        let value = exec.run().unwrap();
        assert_eq!(value.string_units(), expected, "{script}");
        let raw = super::source::to_source(&value).unwrap().text();
        assert_eq!(crate::source_json::decode_string(&raw), expected, "{script}");
    }
}

#[test]
fn string_search_uses_utf16_offsets() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let params=json!({"s":"A😀BA😀"});
    for (script, expected) in [
        ("params.s.indexOf('B')",json!(3)),
        ("params.s.indexOf('😀', 2)",json!(5)),
        ("params.s.lastIndexOf('😀')",json!(5)),
        ("params.s.lastIndexOf('😀', 4)",json!(1)),
        ("params.s.lastIndexOf('A', -1)",json!(-1)),
        ("params.s.indexOf('', 99)",json!(7)),
        ("params.s.lastIndexOf('', 99)",json!(7)),
        ("params.s.startsWith('B', 3)",json!(true)),
        ("params.s.startsWith('A', -1)",json!(false)),
    ] {assert_eq!(run_with(script,params.clone()).unwrap(),expected,"{script}");}
}

#[test]
fn string_builder_retains_utf16_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("def b=new StringBuilder(params.s.substring(1,2)); b.append(params.s.substring(2,3)); b.toString()",vec![0xd83d,0xde00]),
        ("def b=new StringBuilder(params.s); b.reverse(); b.toString()",vec![66,0xd83d,0xde00,65]),
        ("def b=new StringBuilder(params.s); b.setLength(2); b.toString()",vec![65,0xd83d]),
        ("def b=new StringBuilder('A'); b.setLength(3); b.toString()",vec![65,0,0]),
        ("def b=new StringBuilder(params.s); b.insert(2,'x'); b.toString()",vec![65,0xd83d,120,0xde00,66]),
        ("def b=new StringBuilder(params.s.substring(1,2)); ''+b",vec![0xd83d]),
        ("def b=new StringBuilder(params.s); b.deleteCharAt(2); b.toString()",vec![65,0xd83d,66]),
        ("def b=new StringBuilder(params.s); b.delete(2,99); b.toString()",vec![65,0xd83d]),
        ("def b=new StringBuilder(params.s); b.replace(2,3,'x'); b.toString()",vec![65,0xd83d,120,66]),
        ("def b=new StringBuilder(params.s); b.setCharAt(2,(char) 65); b.toString()",vec![65,0xd83d,65,66]),
        ("def b=new StringBuilder(); b.appendCodePoint(128512).appendCodePoint(55296); b.toString()",vec![0xd83d,0xde00,0xd800]),
        ("def b=new StringBuilder(params.s); b.substring(1,2)",vec![0xd83d]),
    ] {
        let cs=compile(script,&json!({"s":"A😀B"}),ScriptContext::PainlessTest).unwrap();
        let mut exec=cs.executor(None,None);let value=exec.run().unwrap();
        assert_eq!(value.string_units(),expected,"{script}");
        assert_eq!(crate::source_json::decode_string(&super::source::to_source(&value).unwrap().text()),expected);
    }
}

#[test]
fn string_builder_utf16_bounds_and_positions() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("def b=new StringBuilder(params.s); b.codePointAt(1)", json!(128512)),
        ("def b=new StringBuilder(params.s); b.codePointBefore(3)", json!(128512)),
        ("def b=new StringBuilder(params.s); b.codePointCount(0,4)", json!(3)),
        ("def b=new StringBuilder(params.s); b.offsetByCodePoints(0,2)", json!(3)),
        ("def b=new StringBuilder(params.s); b.indexOf('B')", json!(3)),
    ] { assert_eq!(run_with(script,json!({"s":"A😀B"})).unwrap(),expected,"{script}"); }
    for script in [
        "def b=new StringBuilder('a'); b.setLength(-1)",
        "def b=new StringBuilder('a'); b.deleteCharAt(1)",
        "def b=new StringBuilder('a'); b.delete(1,0)",
        "def b=new StringBuilder(); b.appendCodePoint(1114112)",
    ] { assert!(run_with(script,json!({})).is_err(),"{script}"); }
}

#[test]
fn raw_update_params_preserve_units_and_cached_parameter_access() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let (body, raw)=crate::source_json::update_request(r#"{"script":{"source":"ctx._source.direct = params.value; ctx._source.dynamic = params[params.key]; ctx._source.nested = params.nested;","params":{"value":"\ud800","key":"value","nested":["\udfff",{"v":"\ud83d\ude00"}]}}}"#).unwrap();
    let mut script=compile_json(&body["script"],ScriptContext::Update,None).unwrap();
    script.source_params=raw.as_ref().unwrap().get("script").unwrap().get("params").cloned().map(std::sync::Arc::new);
    for _ in 0..2 {
        let (_,out)=execute_ctx_source(&script,serde_json::Map::new(),"{}").unwrap();
        let actual=crate::source_json::Source::parse(&out).unwrap();
        let expected=crate::source_json::Source::parse(r#"{"direct":"\ud800","dynamic":"\ud800","nested":["\udfff",{"v":"\ud83d\ude00"}]}"#).unwrap();
        assert!(actual.equivalent(&expected));
    }
}

#[test]
fn character_values_retain_utf16_code_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("(int) params.s.charAt(1)", 55357),
        ("(int) params.s.charAt(2)", 56832),
        ("(int) params.s.toCharArray()[1]", 55357),
        ("(int) Character.toChars(128512)[1]", 56832),
        ("(int) (char) 65535", 65535),
    ] { assert_eq!(run_with(script,json!({"s":"A😀B"})).unwrap(),json!(expected),"{script}"); }
    for script in ["params.s.charAt(1)","(String) params.s.charAt(1)","'' + params.s.charAt(1)","new StringBuilder().append(params.s.charAt(1)).toString()"] {
        let cs=compile(script,&json!({"s":"A😀B"}),ScriptContext::PainlessTest).unwrap();
        let v=cs.executor(None,None).run().unwrap();
        assert_eq!(v.string_units(),vec![0xd83d],"{script}");
        assert_eq!(crate::source_json::decode_string(&super::source::to_source(&v).unwrap().text()),vec![0xd83d]);
    }
}

#[test]
fn utf16_transformations_preserve_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    use super::value::Val;
    let units=[0x20,0xd800,0x41,0xdc00,0x20];
    let recv=Val::utf16(units.to_vec());
    for (name,args,expected) in [
        ("trim",vec![],vec![0xd800,0x41,0xdc00]),
        ("toLowerCase",vec![],vec![0x20,0xd800,0x61,0xdc00,0x20]),
        ("stripLeading",vec![],vec![0xd800,0x41,0xdc00,0x20]),
        ("replace",vec![Val::utf16(vec![0xd800]),Val::utf16(vec![0xdc01])],vec![0x20,0xdc01,0x41,0xdc00,0x20]),
    ] {let v=super::source::string_method(&recv,&units,name,&args).unwrap();assert_eq!(v.string_units(),expected);}
    for (script,expected) in [
        ("params.s.codePointAt(1)",128512),
        ("params.s.codePointBefore(3)",128512),
        ("params.s.codePointAt(2)",56832),
        ("params.s.codePointCount(0,4)",3),
        ("params.s.codePointCount(0,2)",2),
        ("params.s.offsetByCodePoints(0,2)",3),
        ("params.s.offsetByCodePoints(3,-1)",1),
    ] {assert_eq!(run_with(script,json!({"s":"A😀B"})).unwrap(),json!(expected),"{script}");}
}

#[test]
fn raw_literal_source_and_invalid_field_names() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let units:Vec<u16>="return 'before ".encode_utf16().chain([0xd800]).chain(" after';".encode_utf16()).collect();
    let program=super::source::compile_raw_literal_source(&units,ScriptContext::PainlessTest).unwrap();
    let cs=super::CompiledScript {program,params:std::sync::Arc::new(json!({})),source_params:None};
    let value=cs.executor(None,None).run().unwrap();
    let expected:Vec<u16>="before ".encode_utf16().chain([0xd800]).chain(" after".encode_utf16()).collect();
    assert_eq!(value.string_units(),expected);
    assert!(compile(r"return '\ud800';",&json!({}),ScriptContext::PainlessTest).is_err());
    let map=super::value::Val::new_map([(super::value::Key::from_val(&super::value::Val::utf16(vec![0xd800])),super::value::Val::Int(1))].into_iter().collect());
    assert!(super::source::to_source(&map).is_err());
}

#[test]
fn utf16_string_arguments_do_not_stringify_null_or_numbers() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    use super::value::Val;
    for units in [vec![0xd800],vec![65]] {
        let recv=Val::utf16(units.clone());
        for method in ["concat","contains","startsWith","endsWith","indexOf","lastIndexOf","compareTo"] {
            for arg in [Val::Null,Val::Int(42)] {assert!(super::source::string_method(&recv,&units,method,&[arg]).is_err(),"{method}");}
        }
        assert!(super::source::string_method(&recv,&units,"replace",&[Val::Null,Val::str("x")]).is_err());
    }
    assert_eq!(super::source::string_method(&Val::str(""),&[],"repeat",&[Val::Int(i32::MAX)]).unwrap().string_units(),Vec::<u16>::new());
    assert_eq!(run_with("params.s.contains(params.s.substring(1,2))",json!({"s":"A😀B"})).unwrap(),json!(true));
}

#[test]
fn nested_collection_stringification_preserves_utf16() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for script in [
        "def x=[params.s.substring(0,1)]; x.toString()",
        "def x=[params.s.substring(0,1)]; ''+x",
        "def x=[params.s.substring(0,1)]; new StringBuilder().append(x).toString()",
    ] {
        let cs=compile(script,&json!({"s":"😀"}),ScriptContext::PainlessTest).unwrap();
        let value=cs.executor(None,None).run().unwrap();
        assert_eq!(value.string_units(),vec![91,0xd83d,93],"{script}");
    }
    let value=Val::new_map([(super::value::Key::str("key"),Val::new_list(vec![Val::utf16(vec![0xdfff])]))].into_iter().collect());
    assert_eq!(value.string_units(),vec![123,107,101,121,61,91,0xdfff,93,125]);
}

#[test]
fn string_static_conversions_retain_utf16() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("String.valueOf(params.s.substring(0,1))", vec![0xd83d]),
        ("String.valueOf(params.s.charAt(1))", vec![0xde00]),
        ("String.valueOf([params.s.substring(0,1)])", vec![91,0xd83d,93]),
        ("String.join(params.s.substring(0,1), ['a', 'b'])", vec![97,0xd83d,98]),
        ("String.join('-', [params.s.substring(0,1), null])", vec![0xd83d,45,110,117,108,108]),
    ] {
        let cs=compile(script,&json!({"s":"😀"}),ScriptContext::PainlessTest).unwrap();
        let value=cs.executor(None,None).run().unwrap();
        assert_eq!(value.string_units(),expected,"{script}");
    }
}

#[test]
fn character_surrogate_helpers_match_java_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    // ES 9.5.5 java.lang.txt excludes isLowSurrogate, despite the Java API.
    assert!(run_with("Character.isLowSurrogate((char)57343)",json!({})).is_err());
    for (script, expected) in [
        ("Character.isHighSurrogate((char)55296)", json!(true)),
        ("Character.isHighSurrogate((char)56320)", json!(false)),
        ("Character.isSurrogate((char)55295)", json!(false)),
        ("Character.isSurrogatePair((char)55357,(char)56832)", json!(true)),
        ("Character.isSurrogatePair((char)56832,(char)55357)", json!(false)),
        ("Character.toCodePoint((char)55357,(char)56832)", json!(128512)),
        ("(int)Character.highSurrogate(128512)", json!(55357)),
        ("(int)Character.lowSurrogate(128512)", json!(56832)),
        ("Character.charCount(65535)", json!(1)),
        ("Character.charCount(65536)", json!(2)),
        ("Character.isValidCodePoint(1114112)", json!(false)),
        ("Character.isValidCodePoint(55296)", json!(true)),
        ("Character.isBmpCodePoint(-1)", json!(false)),
        ("Character.isSupplementaryCodePoint(1114111)", json!(true)),
    ] { assert_eq!(run_with(script,json!({})).unwrap(),expected,"{script}"); }
}

#[test]
fn character_arrays_keep_surrogate_codepoints() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (source,expected) in [
        ("(int)Character.toChars(55296)[0]",json!(55296)),
        ("Character.toString((char)55296).length()",json!(1)),
        ("Character.hashCode((char)57343)",json!(57343)),
        ("Character.compare((char)55296,(char)55297)",json!(-1)),
        ("(int)Character.reverseBytes((char)55296)",json!(216)),
        ("char[] a=new char[4]; int n=Character.toChars(128512,a,1); [n,(int)a[1],(int)a[2]]",json!([2,55357,56832])),
    ] {assert_eq!(run_with(source,json!({})).unwrap(),expected,"{source}");}
    assert!(run_with("Character.toChars(1114112)",json!({})).is_err());
}

#[test]
fn utf16_character_copy_and_literal_split() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (source,expected) in [
        ("char[] a=new char[4]; params.s.getChars(1,3,a,0); [(int)a[0],(int)a[1]]",json!([55357,56832])),
        ("params.s.contentEquals(new StringBuilder(params.s))",json!(true)),
        ("params.s.splitOnToken(params.s.substring(1,2))[1].charAt(0) == (char)56832",json!(true)),
        ("Character.valueOf((char)55357).charValue() == (char)55357",json!(true)),
        ("Character.valueOf((char)55357).compareTo(Character.valueOf((char)56832))",json!(-1475)),
    ] {assert_eq!(run_with(source,json!({"s":"A😀B"})).unwrap(),expected,"{source}");}
}

#[test]
fn utf16_char_array_copy_and_objects_fallback() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("char[] a = new char[] {(char)65,(char)55357,(char)56832,(char)66}; String.copyValueOf(a).length()", json!(4)),
        ("char[] a = new char[] {(char)65,(char)55357,(char)56832,(char)66}; (int)String.copyValueOf(a,1,1).charAt(0)", json!(55357)),
        ("char[] a = new char[] {(char)57343}; String.copyValueOf(a,1,0).length()", json!(0)),
        ("(int)Objects.toString(params.s.substring(0,1), 'fallback').charAt(0)", json!(55357)),
    ] { assert_eq!(run_with(script,json!({"s":"😀"})).unwrap(),expected,"{script}"); }
    assert!(run_with("String.copyValueOf(new char[1], -1, 1)",json!({})).is_err());
    assert!(run_with("String.copyValueOf(new char[1], 0, 2)",json!({})).is_err());
}

#[test]
fn string_joiner_preserves_utf16_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("def h=params.s.substring(0,1); def j=new StringJoiner(h); j.add('a').add('b'); [(int)j.toString().charAt(1),j.length()]",json!([55357,3])),
        ("def h=params.s.substring(0,1); def j=new StringJoiner(','); j.add(h); def k=new StringJoiner(':'); k.add(h).add('x'); j.merge(k); [(int)j.toString().charAt(0),(int)j.toString().charAt(2),j.length()]",json!([55357,55357,5])),
        ("def j=new StringJoiner(','); j.setEmptyValue(params.s.substring(1)); (int)j.toString().charAt(0)",json!(56832)),
    ] {assert_eq!(run_with(script,json!({"s":"😀"})).unwrap(),expected,"{script}");}
    assert!(run_with("new StringJoiner(null)",json!({})).is_err());
    assert!(run_with(r#"new StringJoiner(",").merge(null)"#,json!({})).is_err());
}

#[test]
fn utf16_case_comparisons_preserve_surrogates_and_java_simple_case() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script,expected) in [
        ("params.s.substring(0,1).equalsIgnoreCase(params.s.substring(0,1))",json!(true)),
        ("params.s.substring(0,1).equalsIgnoreCase(params.s.substring(1,2))",json!(false)),
        ("'İ'.equalsIgnoreCase('i')",json!(true)),
        ("'😀'.compareToIgnoreCase(params.s.substring(0,1))",json!(1)),
        ("'ß'.equalsIgnoreCase('SS')",json!(false)),
        ("'𐐀'.equalsIgnoreCase('𐐨')",json!(true)),
        ("params.s.regionMatches(1,params.s.substring(1,2),0,1)",json!(true)),
        ("params.s.regionMatches(true,2,'a',0,1)",json!(true)),
        ("'ab'.regionMatches(3,'x',0,-1)",json!(true)),
        ("'ab'.regionMatches(-1,'x',0,0)",json!(false)),
        ("params.s.substring(0,1).compareToIgnoreCase(params.s.substring(1,2))",json!(-1475)),
    ] {assert_eq!(run_with(script,json!({"s":"😀A"})).unwrap(),expected,"{script}");}
}

#[test]
fn character_case_mapping_matches_painless_char_overloads() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script,expected) in [
        ("(int)Character.toUpperCase((char)223)",json!(223)),
        ("(int)Character.toLowerCase((char)304)",json!(105)),
        ("(int)Character.toUpperCase((char)55296)",json!(55296)),
        ("(int)Character.toTitleCase((char)454)",json!(453)),
    ] {assert_eq!(run_with(script,json!({})).unwrap(),expected,"{script}");}
}

#[test]
fn character_case_disallows_unexposed_integer_overloads() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for script in ["Character.toUpperCase(66600)","Character.toLowerCase(66560)","Character.toTitleCase(params.n)"] {
        assert!(run_with(script,json!({"n":454})).is_err(),"{script}");
    }
}

#[test]
fn collection_and_stream_join_preserve_utf16_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script, expected) in [
        ("[params.s.substring(0,1),null].join(params.s.substring(1,2))",vec![0xd83d,0xde00,110,117,108,108]),
        ("['a','b'].stream().collect(Collectors.joining(params.s.substring(0,1),params.s.substring(1,2),'!'))",vec![0xde00,97,0xd83d,98,33]),
        ("[params.s.substring(0,1),params.s.substring(1,2)].stream().collect(Collectors.joining())",vec![0xd83d,0xde00]),
    ] {
        let cs=compile(script,&json!({"s":"😀"}),ScriptContext::PainlessTest).unwrap();
        assert_eq!(cs.executor(None,None).run().unwrap().string_units(),expected,"{script}");
    }
}

#[test]
fn character_allowlist_narrows_literals_but_not_dynamic_values() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script,expected) in [
        ("Character.toUpperCase(97)",json!("A")),
        ("Character.toString(97)",json!("a")),
        ("Character.isHighSurrogate(55296)",json!(true)),
        ("Character.toCodePoint(55357,56832)",json!(128512)),
    ] {assert_eq!(run_with(script,json!({})).unwrap(),expected,"{script}");}
    for script in ["Character.toUpperCase(66600)","def x=97; Character.toUpperCase(x)","def x='a'; Character.toUpperCase(x)","Character.isHighSurrogate(params.n)"] {
        assert!(run_with(script,json!({"n":55296})).is_err(),"{script}");
    }
}

#[test]
fn locale_casing_preserves_lone_units_and_java_special_cases() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script,input,expected) in [
        ("params.s.toLowerCase(new Locale('tr'))",vec![0xd800,73,775,0xdfff],vec![0xd800,105,0xdfff]),
        ("params.s.toUpperCase(Locale.forLanguageTag('az'))",vec![0xd800,105,0xdfff],vec![0xd800,304,0xdfff]),
        ("params.s.toLowerCase(new Locale('lt'))",vec![0xd800,73,769,0xdfff],vec![0xd800,105,775,769,0xdfff]),
        ("params.s.toUpperCase(new Locale('lt'))",vec![105,775,0xdfff],vec![73,0xdfff]),
        ("params.s.toLowerCase(Locale.ROOT)",vec![65,931,769,0xd800],vec![97,962,769,0xd800]),
        ("params.s.toUpperCase(Locale.ENGLISH)",vec![0xd800,223,0xdfff],vec![0xd800,83,83,0xdfff]),
    ] {
        let cs=compile(script,&json!({}),ScriptContext::PainlessTest).unwrap();
        let mut executor=cs.executor(None,None);
        executor.set_param("s",Val::utf16(input));
        assert_eq!(executor.run().unwrap_or_else(|e|panic!("{script}: {e:?}")).string_units(),expected,"{script}");
    }
    assert_eq!(run_with("DateTimeFormatter.ofPattern('MMM', Locale.ENGLISH).format(LocalDate.of(2024,1,1))",json!({})).unwrap(),json!("Jan"));
    assert!(run_with("'x'.toUpperCase(null)",json!({})).is_err());
    assert!(run_with("'x'.toLowerCase('tr')",json!({})).is_err());
    let cs=compile("new Locale(params.s).getLanguage()",&json!({}),ScriptContext::PainlessTest).unwrap();
    let mut executor=cs.executor(None,None);
    executor.set_param("s",Val::utf16(vec![0xd800,65]));
    assert_eq!(executor.run().unwrap().string_units(),vec![0xd800,97]);
}

#[test]
fn character_sequence_streams_arrays_and_bounded_codepoints() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (script,expected) in [
        ("params.s.chars().sum()",json!(112254)),
        ("params.s.codePoints().sum()",json!(128577)),
        ("new StringBuilder(params.s).chars().sum()",json!(112254)),
        ("String.copyValueOf(params.s.toCharArray())",json!("A😀")),
        ("Character.codePointAt(params.s,1)",json!(128512)),
        ("Character.codePointAt(params.s.toCharArray(),1,2)",json!(55357)),
        ("Character.codePointBefore(params.s.toCharArray(),3,2)",json!(56832)),
        ("Character.codePointCount(params.s,0,3)",json!(2)),
        ("Character.offsetByCodePoints(params.s.toCharArray(),1,2,1,1)",json!(3)),
        ("Character.offsetByCodePoints(params.s,3,-1)",json!(1)),
    ] {assert_eq!(run_with(script,json!({"s":"A😀"})).unwrap(),expected,"{script}");}
    for script in ["Character.codePointAt(params.s.toCharArray(),1,1)","Character.codePointBefore(params.s.toCharArray(),2,2)","Character.offsetByCodePoints(params.s.toCharArray(),1,2,1,2)"] {
        assert!(run_with(script,json!({"s":"A😀"})).is_err(),"{script}");
    }
}

#[test]
fn floor_math_matches_java_boundaries_without_panicking() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for (a,b,div,rem) in [
        (i64::MIN,-1,i64::MIN,0),
        (i64::MAX,i64::MAX,1,0),
        (i64::MAX-1,i64::MAX,0,i64::MAX-1),
        (i64::MIN,i64::MAX,-2,i64::MAX-1),
        (i64::MAX,i64::MIN,-1,-1),
        (-1,i64::MIN,0,-1),
        (1,-2,-1,-1),(-1,2,-1,1),(-1,-2,0,-1),
    ] {
        for (method,expected) in [("floorDiv",div),("floorMod",rem)] {
            let args=[Val::Long(a),Val::Long(b)];
            assert_eq!(super::runtime::math(method,&args).unwrap().unwrap().to_i64().unwrap(),expected,"{method}({a},{b})");
        }
    }
    for (method,expected) in [("floorDiv",i32::MIN),("floorMod",0)] {
        let result=super::runtime::math(method,&[Val::Int(i32::MIN),Val::Int(-1)]).unwrap().unwrap();
        assert!(matches!(result,Val::Int(n) if n==expected));
        let error=super::runtime::math(method,&[Val::Long(1),Val::Long(0)]).unwrap_err();
        assert_eq!(error.kind,"arithmetic_exception");
    }
    assert!(run_with("Math.floorDiv(Long.MIN_VALUE,-1L)",json!({})).is_err());
    assert!(run_with("Math.floorMod(Long.MIN_VALUE,-1L)",json!({})).is_err());
}

#[test]
fn formatter_preserves_utf16_literals_arguments_precision_and_width() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let fmt = |s: &str, values: &[Val]| super::runtime::java_format(&s.encode_utf16().collect::<Vec<_>>(), Some(values), &[]).unwrap();
    assert_eq!(fmt("[%3.1s]", &[Val::string("😀".into())]), vec![91,32,32,0xd83d,93]);
    assert_eq!(fmt("%c:%c", &[Val::Char(0xd800), Val::Int(0xdfff)]), vec![0xd800,58,0xdfff]);
    assert_eq!(fmt("%2$s/%1$s/%<s", &[Val::utf16(vec![0xd800]), Val::utf16(vec![0xdfff])]), vec![0xdfff,47,0xd800,47,0xd800]);
    assert_eq!(fmt("%S", &[Val::utf16(vec![0xd800,0x61,0xdf])]), vec![0xd800,65,83,83]);
    assert_eq!(super::runtime::java_format(&[0xd800,37,115,0xdfff], Some(&[Val::utf16(vec![0xdc00])]), &[]).unwrap(), vec![0xd800,0xdc00,0xdfff]);
    assert!(super::runtime::java_format(&[37,99], Some(&[Val::Int(0x110000)]), &[]).is_err());
}

#[test]
fn arrays_to_string_keeps_surrogate_characters() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert_eq!(run_with("char[] a = new char[2]; a[0]=(char)55296; a[1]=(char)57343; def s=Arrays.deepToString(new def[] {a}); return [(int)s.charAt(2),(int)s.charAt(5),s.length()];",json!({})).unwrap(),json!([55296,57343,8]));
}

#[test]
fn painless_dynamic_array_clone_is_not_exposed() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert!(run_with("String.copyValueOf(params.s.toCharArray().clone())",json!({"s":"A😀"})).is_err());
}

#[test]
fn utf16_explicit_utf8_encoding_uses_java_replacement() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let value=Val::utf16(vec![0xd800,65,0xd83d,0xde00,0xdfff]);
    let units=value.string_units();
    let actual=super::source::string_method(&value,&units,"encodeBase64",&[]).unwrap();
    assert_eq!(actual.to_java_string(),"P0Hwn5iAPw==");
    let expected=Val::str("?A😀?");
    for method in ["sha1","sha256","sha512"] {
        assert_eq!(super::source::string_method(&value,&units,method,&[]).unwrap().to_java_string(),super::source::string_method(&expected,&expected.string_units(),method,&[]).unwrap().to_java_string());
    }
    assert!(super::source::string_method(&value,&units,"decodeBase64",&[]).is_err());
}

#[test]
fn java_utf8_decode_groups_malformed_sequences() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    use base64::Engine;
    for (bytes,expected) in [
        (vec![0xed,0xa0,0x80],vec![0xfffd]),
        (vec![0xed,0xa0,65],vec![0xfffd,65]),
        (vec![0xf0,0x90,0x80],vec![0xfffd]),
        (vec![0xe0,0x80,0x80],vec![0xfffd,0xfffd,0xfffd]),
        (vec![0xf4,0x90,0x80,0x80],vec![0xfffd,0xfffd,0xfffd,0xfffd]),
        (vec![0xf0,0x9f,0x98,0x80],vec![0xd83d,0xde00]),
    ] {
        let value=Val::string(base64::engine::general_purpose::STANDARD.encode(bytes));
        assert_eq!(super::source::string_method(&value,&value.string_units(),"decodeBase64",&[]).unwrap().string_units(),expected);
    }
}

#[test]
fn deep_array_string_preserves_cycles_and_painless_allowlist() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert_eq!(run_with("def[] a=new def[1]; a[0]=a; return Arrays.deepToString(a)",json!({})).unwrap(),json!("[[...]]"));
    assert!(run_with("Arrays.toString(new char[] {(char)65})",json!({})).is_err());
}

#[test]
fn builder_ranges_and_capacity_follow_java_code_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert_eq!(run_with("def b=new StringBuilder(); def a=b.capacity(); b.append('12345678901234567'); def c=b.capacity(); b.setLength(0); return [a,c,b.capacity()]",json!({})).unwrap(),json!([16,34,34]));
    assert_eq!(run_with("def b=new StringBuilder('😀'); b.append('😀',0,1); b.append(b,2,3); char[] a=new char[2]; b.getChars(2,4,a,0); return [b.capacity(),(int)a[0],(int)a[1]]",json!({})).unwrap(),json!([18,55357,55357]));
}

#[test]
fn request_script_slots_preserve_params_without_reinterpreting_user_numbers() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let mut environment=ScriptEnv::new(None,None);
    environment.raw_search_scripts=vec![crate::source_json::Source::parse(r#"{"source":"(int)params.s.charAt(0)","params":{"s":"\ud800"}}"#).unwrap(),crate::source_json::Source::from_value(&json!(0))];
    with_env(environment,|| {
        let script=compile_search_script(&json!(0),ScriptContext::Field).unwrap();
        assert_eq!(script.executor(None,None).run().unwrap().to_i64().unwrap(),55296);
        assert!(compile_search_script(&json!(1),ScriptContext::Field).is_err());
        assert!(compile_json(&json!(0),ScriptContext::Field,None).is_err());
        assert!(crate::search::parse_request(&json!({"query":{"script":{"script":0}}}),&Default::default()).is_ok());
        assert!(crate::search::parse_request(&json!({"query":{"script":{"script":1}}}),&Default::default()).is_err());
        for body in [json!({"query":{"script_score":{"query":{"match_all":{}},"script":0}}}),json!({"aggs":{"units":{"max":{"script":0}}}}),json!({"rescore":{"query":{"rescore_query":{"script_score":{"query":{"match_all":{}},"script":0}}}}})] {
            assert!(crate::search::parse_request(&body,&Default::default()).is_ok(),"{body}");
        }
    });
}

#[test]
fn regex_groups_offsets_and_replacement_use_utf16() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let raw=crate::source_json::Source::parse(r#"{"source":"def m=/(.)/.matcher(params.s); def out=[]; while(m.find()) {out.add(m.start()); out.add(m.end()); out.add((int)m.group(1).charAt(0));} return out","params":{"s":"\ud800A\ud83d\ude00\udfff"}}"#).unwrap();
    let wrapper=crate::source_json::Source::Object(std::collections::BTreeMap::from([("script".into(),raw.clone())]));
    let (metadata,_)=crate::source_json::update_request(&wrapper.text()).unwrap();
    let cs=compile_json_source(&metadata["script"],Some(&raw),ScriptContext::PainlessTest,None).unwrap();
    assert_eq!(cs.executor(None,None).run().unwrap().to_json(),json!([0,1,55296,1,2,65,2,4,55357,4,5,57343]));
    assert_eq!(run_with("def m=/./.matcher('😀'); m.find(1); return [m.start(),m.end(),(int)m.group().charAt(0)]",json!({})).unwrap(),json!([1,2,56832]));
}

#[test]
fn raw_regex_literals_preserve_unpaired_pattern_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for expression in [r"/[\ud800]/.matcher(params.s).matches()",r"/\Q\ud800\E/.matcher(params.s).matches()"] {
        let text=format!(r#"{{"source":"{}","params":{{"s":"\ud800"}}}}"#,expression.replace('\\',"\\\\").replace("\\\\ud800","\\ud800"));
        let raw=crate::source_json::Source::parse(&text).unwrap();
        let wrapper=crate::source_json::Source::Object(std::collections::BTreeMap::from([("script".into(),raw.clone())]));
        let (metadata,_)=crate::source_json::update_request(&wrapper.text()).unwrap();
        let cs=compile_json_source(&metadata["script"],Some(&raw),ScriptContext::PainlessTest,None).unwrap();
        assert_eq!(cs.executor(None,None).run().unwrap().to_json(),json!(true),"{text}");
    }
}

#[test]
fn matcher_regions_and_named_replacements_keep_code_unit_offsets() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert_eq!(run_with("def m=/(.)/.matcher('😀x'); m.region(1,2); return [m.matches(),m.start(),m.end(),(int)m.group(1).charAt(0),m.regionStart(),m.regionEnd()]",json!({})).unwrap(),json!([true,1,2,56832,1,2]));
    assert_eq!(run_with("def m=/(?<unit>.)/.matcher(params.s); m.find(); return [(int)m.namedGroup('unit').charAt(0),m.start(1),m.end(1)]",json!({"s":"😀"})).unwrap(),json!([55357,0,2]));
}

#[test]
fn return_regex_literal_is_not_tokenized_as_division() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    for source in [r"return /[^a]/.matcher(params.s).matches()",r"return /[\p{Cs}&&[^A]]/.matcher(params.s).matches()",r"return /\Qx\E/.matcher('x').matches()"] {
        let script=super::compile_json(&serde_json::json!({"source":source,"params":{"s":"z"}}),super::ScriptContext::PainlessTest,None).unwrap();
        assert!(script.executor(None,None).run().is_ok());
    }
}

#[test]
fn matcher_replacement_resets_region_and_updates_subsequent_offsets() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert_eq!(run_with("def m=/a/.matcher('a😀a'); m.region(1,3); def out=m.replaceFirst('x'); def start=m.start(); m.find(); return [out,start,m.start(),m.end(),m.regionStart(),m.regionEnd()]",json!({})).unwrap(),json!(["x😀a",0,3,4,0,4]));
    assert_eq!(run_with("def m=/a/.matcher('aba'); m.region(1,2); def out=m.replaceAll('x'); return [out,m.regionStart(),m.regionEnd(),m.find()]",json!({})).unwrap(),json!(["xbx",0,3,false]));
    assert!(run_with("def m=/a/.matcher('aba'); m.replaceAll('x'); return m.group()",json!({})).is_err());
}

#[test]
fn regex_dot_uses_java_line_terminators_and_scoped_dotall() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    assert_eq!(run_with("return /./.matcher(params.s).find()",json!({"s":"\u{2028}"})).unwrap(),json!(false));
    assert_eq!(run_with("return /./s.matcher(params.s).find()",json!({"s":"\u{2028}"})).unwrap(),json!(true));
    assert_eq!(run_with("return /(?s:.)(?-s:.)/.matcher(params.s).matches()",json!({"s":"\u{2028}a"})).unwrap(),json!(true));
    assert_eq!(run_with("return /(?s:.)(?-s:.)/.matcher(params.s).matches()",json!({"s":"a\u{2028}"})).unwrap(),json!(false));
}

#[test]
fn raw_utf16_programs_share_cache_without_conflating_lone_units() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let units = |unit| vec![39, unit, 39];
    let a = super::source::compile_raw_literal_source(&units(0xd800), ScriptContext::PainlessTest).unwrap();
    let same = super::source::compile_raw_literal_source(&units(0xd800), ScriptContext::PainlessTest).unwrap();
    let different = super::source::compile_raw_literal_source(&units(0xd801), ScriptContext::PainlessTest).unwrap();
    assert!(Arc::ptr_eq(&a, &same));
    assert!(!Arc::ptr_eq(&a, &different));
}

#[test]
fn huge_string_allocations_are_script_errors_not_aborts() {
    let _compilation_budget = rate_limit::fixture_budget("10000/1m");
    let cause = |src: &str| {
        let e = run_with(src, json!({})).unwrap_err();
        assert_eq!(e.status, 400, "{src}: {e:?}");
        assert_eq!(e.kind, "script_exception", "{src}: {e:?}");
        e.to_json()["error"]["caused_by"].clone()
    };
    for src in ["return 'x'.repeat(Integer.MAX_VALUE)", "def s = 'x'; return s.repeat(Integer.MAX_VALUE)"] {
        let c = cause(src);
        assert!(c["reason"].as_str().is_some_and(|r| r.contains("repeat/1] not found")), "{src}: {c}");
    }
    assert_eq!(cause("return String.format('%2000000000s', new def[] {'a'})")["type"], "out_of_memory_error");
    assert_eq!(cause("return String.format('%.2000000000f', new def[] {1.0})")["type"], "out_of_memory_error");
    let c = cause("return String.format('%99999999999s', new def[] {'a'})");
    assert_eq!((c["type"].as_str(), c["reason"].as_str()), (Some("illegal_format_width_exception"), Some("-2147483648")));
    assert_eq!(run("return String.format('%5s', new def[] {'a'})"), json!("    a"));
}
