//! Function signatures and argument type resolution, from ES 9.5.5's
//! generated function definitions (`functions.json`: every function's
//! signatures, whether it is snapshot-only, and the expected-type phrase
//! each argument's type error prints, as a release build reports it).

use serde_json::Value;
use std::collections::HashMap;
use std::sync::LazyLock;

pub struct Sig {
    /// (type, optional) per parameter.
    pub params: Vec<(String, bool)>,
    pub variadic: bool,
    pub ret: String,
}

pub struct FnDef {
    pub name: String,
    /// `scalar`, `agg`, `grouping`, `time_series_agg`, `operator`.
    pub kind: String,
    pub snapshot_only: bool,
    pub sigs: Vec<Sig>,
    /// Per parameter: (ordinal word, expected-type phrase) of its type error.
    pub phrases: Vec<Option<(String, String)>>,
}

struct Table {
    fns: HashMap<String, FnDef>,
    aliases: HashMap<String, String>,
}

static TABLE: LazyLock<Table> = LazyLock::new(|| {
    let v: Value = serde_json::from_str(include_str!("functions.json")).expect("functions.json");
    let mut fns = HashMap::new();
    for (name, d) in v["fns"].as_object().into_iter().flatten() {
        let sigs = d["s"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|s| Sig {
                params: s[0].as_array().into_iter().flatten().map(|p| (p[0].as_str().unwrap_or("").to_string(), p[1].as_bool().unwrap_or(false))).collect(),
                variadic: s[1].as_bool().unwrap_or(false),
                ret: s[2].as_str().unwrap_or("").to_string(),
            })
            .collect();
        let phrases = d["ph"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| p.as_array().map(|a| (a[0].as_str().unwrap_or("").to_string(), a[1].as_str().unwrap_or("").to_string())))
            .collect();
        fns.insert(
            name.clone(),
            FnDef { name: name.clone(), kind: d["k"].as_str().unwrap_or("").to_string(), snapshot_only: d["snap"].as_bool().unwrap_or(false), sigs, phrases },
        );
    }
    let aliases = v["aliases"].as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect();
    Table { fns, aliases }
});

/// The definition of a function (by any of its names, case-insensitive)
/// registered in a release build.
pub fn lookup(name: &str) -> Option<&'static FnDef> {
    let n = name.to_ascii_lowercase();
    let t = &*TABLE;
    let n = t.aliases.get(&n).cloned().unwrap_or(n);
    t.fns.get(&n).filter(|d| !d.snapshot_only && d.kind != "operator")
}

/// Names of all release functions (for "did you mean").
pub fn names() -> Vec<&'static str> {
    TABLE.fns.values().filter(|d| !d.snapshot_only && d.kind != "operator").map(|d| d.name.as_str()).collect()
}

/// An operator's definition (`add`, `like`, ...).
pub fn operator(name: &str) -> Option<&'static FnDef> {
    TABLE.fns.get(name)
}

const NUM: [&str; 7] = ["zero", "one", "two", "three", "four", "five", "six"];

/// `FunctionArgumentValidation`: the arity error of `n` arguments, if any.
pub fn arity_error(d: &FnDef, n: usize) -> Option<std::borrow::Cow<'static, str>> {
    let max = d.sigs.iter().map(|s| s.params.len()).max().unwrap_or(0);
    let min = d.sigs.iter().map(|s| s.params.iter().filter(|p| !p.1).count()).min().unwrap_or(0);
    let variadic = d.sigs.iter().any(|s| s.variadic);
    if variadic {
        return if min <= 1 {
            (n == 0).then_some("expects at least one argument".into())
        } else {
            (n < 2).then_some("expects at least two arguments".into())
        };
    }
    if n >= min && n <= max {
        return None;
    }
    Some(match (max, min) {
        (0, _) => "expects no arguments".into(),
        (1, _) => "expects exactly one argument".into(),
        (2, 1) | (2, 0) => "expects one or two arguments".into(),
        (2, _) => "expects exactly two arguments".into(),
        (3, 0) | (3, 1) => "expects one, two or three arguments".into(),
        (3, 2) => "expects two or three arguments".into(),
        (3, _) => "expects exactly three arguments".into(),
        (4, 3) => "expects three or four arguments".into(),
        (4, 2) => "expects minimum two, maximum four arguments".into(),
        (4, _) => "expects exactly four arguments".into(),
        (m, lo) if lo < m => format!("expects between {} and {} arguments", NUM[lo.min(6)], NUM[m.min(6)]).into(),
        (m, _) => format!("expects exactly {} arguments", NUM[m.min(6)]).into(),
    })
}

/// The declared types of parameter `i` (a variadic tail repeats its last
/// parameter).
pub fn param_types(d: &FnDef, i: usize) -> Vec<&str> {
    param_types_n(d, i, None)
}

/// [`param_types`] over the signatures that take `n` arguments (all when
/// `None` or when none does).
pub fn param_types_n(d: &FnDef, i: usize, n: Option<usize>) -> Vec<&str> {
    let takes = |s: &Sig, n: usize| {
        let req = s.params.iter().filter(|p| !p.1).count();
        n >= req && (s.variadic || n <= s.params.len())
    };
    let any = n.is_some_and(|n| d.sigs.iter().any(|s| takes(s, n)));
    let mut out: Vec<&str> = Vec::new();
    for s in &d.sigs {
        if any && !takes(s, n.unwrap()) {
            continue;
        }
        let p = match s.params.get(i) {
            Some(p) => Some(p),
            None if s.variadic => s.params.last(),
            None => None,
        };
        if let Some((t, _)) = p {
            if !out.contains(&t.as_str()) {
                out.push(t.as_str());
            }
        }
    }
    out
}

/// Whether an ES|QL type is accepted where `declared` is.
fn accepts(declared: &str, ty: &str) -> bool {
    declared == ty || (declared == "date" && ty == "datetime")
}

/// `DATA_TYPE_CASTING_PRIORITY`: the type a string literal argument is
/// implicitly cast to (`None` when the parameter takes strings or nothing castable).
pub fn string_cast_target(d: &FnDef, i: usize) -> Option<&'static str> {
    const PRIORITY: &[&str] = &[
        "date", "date_period", "time_duration", "double", "long", "integer", "ip", "version", "geo_point", "geo_shape", "cartesian_point", "cartesian_shape", "geohash", "geohex", "geotile", "boolean",
        "unsigned_long", "dense_vector",
    ];
    let types = param_types(d, i);
    if types.is_empty() || types.iter().any(|t| matches!(*t, "keyword" | "text")) {
        return None;
    }
    PRIORITY.iter().find(|p| types.contains(p)).copied()
}

/// The ordinal word of argument `i` in type errors (`""` when the function
/// takes one parameter).
pub fn ordinal(d: &FnDef, i: usize) -> String {
    if let Some(Some((o, _))) = d.phrases.get(i) {
        return o.clone();
    }
    let max = d.sigs.iter().map(|s| s.params.len()).max().unwrap_or(0);
    if max <= 1 && !d.sigs.iter().any(|s| s.variadic) {
        return String::new();
    }
    ordinal_word(i).to_string()
}

pub fn ordinal_word(i: usize) -> &'static str {
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"].get(i).copied().unwrap_or("")
}

/// The expected-type phrase of parameter `i`.
pub fn phrase(d: &FnDef, i: usize) -> String {
    if let Some(Some((_, p))) = d.phrases.get(i) {
        return p.clone();
    }
    describe(&param_types(d, i))
}

/// Describe a set of types the way `TypeResolutions` does (`string`,
/// `numeric`, `a, b or c`).
pub fn describe(types: &[&str]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let has = |t: &str| types.contains(&t);
    let numeric = has("integer") && has("long") && has("double");
    let string = has("keyword") && has("text");
    for t in types {
        match *t {
            "keyword" | "text" if string => {
                if !parts.iter().any(|p| p == "string") {
                    parts.push("string".into());
                }
            }
            "integer" | "long" | "double" | "unsigned_long" if numeric => {
                if !parts.iter().any(|p| p == "numeric") {
                    parts.push("numeric".into());
                }
            }
            "date" => parts.push("datetime".into()),
            other => parts.push(other.to_string()),
        }
    }
    match parts.len() {
        0 => String::new(),
        1 => parts.remove(0),
        _ => {
            let last = parts.pop().unwrap();
            format!("{} or {last}", parts.join(", "))
        }
    }
}

/// Whether `ty` (an ES|QL column type) fits parameter `i`.
pub fn param_accepts(d: &FnDef, i: usize, ty: &str) -> bool {
    param_accepts_n(d, i, ty, None)
}

/// [`param_accepts`] for a call with `n` arguments.
pub fn param_accepts_n(d: &FnDef, i: usize, ty: &str, n: Option<usize>) -> bool {
    if ty == "null" {
        return true;
    }
    let types = param_types_n(d, i, n);
    if types.is_empty() {
        return true;
    }
    types.iter().any(|t| accepts(t, ty))
}

/// The return type for argument types `args` (`null` arguments match
/// anything); the first matching signature wins.
pub fn return_type(d: &FnDef, args: &[String]) -> Option<String> {
    let fits = |s: &Sig| -> bool {
        let req = s.params.iter().filter(|p| !p.1).count();
        if args.len() < req || (!s.variadic && args.len() > s.params.len()) {
            return false;
        }
        args.iter().enumerate().all(|(i, a)| {
            let p = s.params.get(i).or_else(|| if s.variadic { s.params.last() } else { None });
            match p {
                Some((t, _)) => a == "null" || accepts(t, a),
                None => false,
            }
        })
    };
    // Prefer a signature with exactly as many parameters as arguments.
    d.sigs.iter().filter(|s| s.params.len() == args.len()).find(|s| fits(s)).or_else(|| d.sigs.iter().find(|s| fits(s))).map(|s| if s.ret == "date" { "date".into() } else { s.ret.clone() })
}
