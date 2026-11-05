//! ES|QL query settings (`QuerySettings`): the in-query `SET name = value;`
//! statements and the request body `settings` object (plus its legacy
//! top-level aliases). Precedence: default < request body < `SET`.

use super::parse::Expr;
use super::{parsing_err, Val};
use crate::date::Tz;
use crate::error::{EsError, Result};
use serde_json::Value;

/// `unmapped_fields`: how references to fields missing from the mapping resolve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Unmapped {
    /// Unknown columns fail the query.
    #[default]
    Default,
    /// Unknown columns read as `null` (type `null`).
    Nullify,
    /// Unknown columns load from `_source` as `keyword`.
    Load,
}

impl Unmapped {
    pub fn name(&self) -> &'static str {
        match self {
            Unmapped::Default => "default",
            Unmapped::Nullify => "nullify",
            Unmapped::Load => "load",
        }
    }
}

/// Resolved query settings.
#[derive(Clone, Debug, Default)]
pub struct Settings {
    /// `time_zone` (UTC when unset).
    pub time_zone: Tz,
    /// Whether a `time_zone` was supplied at all.
    pub time_zone_set: bool,
    pub unmapped: Unmapped,
    pub column_metadata: bool,
    /// `approximation` (`true`/`false` or a settings map).
    pub approximation: Option<Value>,
}

/// One `SET name = value;` of a query.
#[derive(Clone, Debug)]
pub struct QuerySetting {
    pub name: String,
    pub value: Expr,
    /// Offset of the `SET` keyword.
    pub pos: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Keyword,
    Boolean,
    /// A boolean or a map (`approximation`).
    Object,
}

struct Def {
    name: &'static str,
    kind: Kind,
    /// Settable under the request body `settings` object.
    body: bool,
    /// Also accepted as a top-level request body field.
    root_alias: bool,
}

/// `QuerySettings.ALL`.
const DEFS: &[Def] = &[
    Def { name: "approximation", kind: Kind::Object, body: true, root_alias: true },
    Def { name: "column_metadata", kind: Kind::Boolean, body: true, root_alias: false },
    Def { name: "project_routing", kind: Kind::Keyword, body: true, root_alias: true },
    Def { name: "time_zone", kind: Kind::Keyword, body: true, root_alias: true },
    Def { name: "unmapped_fields", kind: Kind::Keyword, body: false, root_alias: false },
];

fn def(name: &str) -> Option<&'static Def> {
    DEFS.iter().find(|d| d.name == name)
}

/// The top-level request body fields that alias a setting.
pub const ROOT_ALIASES: &[&str] = &["approximation", "project_routing", "time_zone"];

/// A typed setting value.
#[derive(Clone, Debug, PartialEq)]
enum Typed {
    Zone(Tz),
    Unmapped(Unmapped),
    Bool(bool),
    Json(Value),
    Routing(String),
}

/// Java `ZoneId.of(..).normalized()`.
pub fn parse_zone(s: &str) -> std::result::Result<Tz, String> {
    let bad = || format!("Invalid time zone [{s}]");
    if s.is_empty() || s.trim() != s {
        return Err(bad());
    }
    Tz::parse(s).map_err(|_| bad())
}

fn parse_unmapped(s: &str) -> std::result::Result<Unmapped, String> {
    match s.to_ascii_uppercase().as_str() {
        "DEFAULT" => Ok(Unmapped::Default),
        "NULLIFY" => Ok(Unmapped::Nullify),
        "LOAD" => Ok(Unmapped::Load),
        _ => Err(format!("Invalid unmapped_fields resolution [{s}], must be one of [DEFAULT, NULLIFY, LOAD]")),
    }
}

fn parse_bool(s: &str) -> std::result::Result<bool, String> {
    match s {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("Failed to parse value [{s}] as only [true] or [false] are allowed.")),
    }
}

/// `readFromString`: parse a keyword-typed value.
fn read_string(d: &Def, s: &str) -> std::result::Result<Typed, String> {
    Ok(match d.name {
        "time_zone" => Typed::Zone(parse_zone(s)?),
        "unmapped_fields" => Typed::Unmapped(parse_unmapped(s)?),
        "column_metadata" => Typed::Bool(parse_bool(s)?),
        "project_routing" => Typed::Routing(s.to_string()),
        _ => Typed::Json(Value::String(s.to_string())),
    })
}

/// The setting's validator, run on user-supplied values.
fn validate_value(d: &Def, v: &Typed) -> Option<String> {
    match (d.name, v) {
        // Serverless cross-project search only.
        ("project_routing", _) => Some("cross-project search not enabled".into()),
        ("approximation", Typed::Json(j)) => super::approx::validate(j),
        _ => None,
    }
}

/// ES|QL type name of a `SET` value expression.
fn expr_type(e: &Expr) -> &'static str {
    match e {
        Expr::Lit(Val::Str(_)) => "KEYWORD",
        Expr::Lit(Val::Bool(_)) => "BOOLEAN",
        Expr::Lit(Val::Int(_)) => "INTEGER",
        Expr::Lit(Val::Long(_)) => "LONG",
        Expr::Lit(Val::Double(_)) => "DOUBLE",
        Expr::Lit(Val::Null) => "NULL",
        Expr::Map(_) => "MAP",
        _ => "UNSUPPORTED",
    }
}

/// `QuerySettings.validate` + `readFromExpression` for one `SET`.
fn read_set(full: &str, s: &QuerySetting) -> Result<(&'static Def, Typed)> {
    let d = def(&s.name).ok_or_else(|| parsing_err(full, s.pos, format!("Unknown setting [{}]", s.name)))?;
    let want = match d.kind {
        Kind::Keyword => Some("KEYWORD"),
        Kind::Boolean => Some("BOOLEAN"),
        Kind::Object => None,
    };
    if let Some(w) = want {
        if expr_type(&s.value) != w {
            return Err(parsing_err(full, s.pos, format!("Setting [{}] must be of type {w}", s.name)));
        }
    }
    let err = |m: String| parsing_err(full, s.pos, format!("Error validating setting [{}]: {m}", s.name));
    let v = match (&s.value, d.kind) {
        (Expr::Lit(Val::Str(x)), Kind::Keyword) => read_string(d, x).map_err(err)?,
        (Expr::Lit(Val::Bool(b)), _) => {
            if d.kind == Kind::Boolean {
                Typed::Bool(*b)
            } else {
                Typed::Json(Value::Bool(*b))
            }
        }
        (Expr::Map(m), Kind::Object) => Typed::Json(Value::Object(m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())),
        (other, _) => return Err(err(format!("unsupported value [{}]", expr_type(other).to_ascii_lowercase()))),
    };
    if let Some(m) = validate_value(d, &v) {
        return Err(err(m));
    }
    Ok((d, v))
}

/// Settings supplied in the request body (`settings.{}` and root aliases),
/// already validated; `SET`s are folded in by [`resolve`].
#[derive(Clone, Debug, Default)]
pub struct BodySettings {
    values: Vec<(&'static str, Typed)>,
}

impl BodySettings {
    /// Parse the request body: `settings` (an object) and the root aliases.
    /// `loc(path, value)` gives the `[line:col] ` of a key (or its value).
    pub fn from_body(body: &Value, loc: &dyn Fn(&[&str], bool) -> String) -> Result<BodySettings> {
        let mut out = BodySettings::default();
        let mut nested: Vec<(&'static str, Typed)> = Vec::new();
        if let Some(s) = body.get("settings") {
            let failed = |e: EsError| EsError::x_content_parse(format!("{}[esql/query] failed to parse field [settings]", loc(&["settings"], false))).caused_by(e);
            let Some(obj) = s.as_object() else {
                return Err(failed(EsError::x_content_parse(format!("{}[settings] must be an object", loc(&["settings"], true)))));
            };
            for (k, v) in obj {
                let Some(d) = def(k) else {
                    return Err(failed(EsError::x_content_parse(format!("{}Unknown setting [{k}] under [settings]", loc(&["settings", k], false)))));
                };
                if !d.body {
                    return Err(failed(EsError::x_content_parse(format!("{}Setting [{k}] is not exposed as a request body parameter", loc(&["settings", k], false)))));
                }
                let t = read_json(d, v).map_err(|m| failed(EsError::x_content_parse(format!("{}Failed to parse value for setting [{k}]: {m}", loc(&["settings", k], true)))))?;
                nested.push((d.name, t));
            }
        }
        for name in ROOT_ALIASES {
            let Some(v) = body.get(*name) else { continue };
            let d = def(name).unwrap();
            let t = read_json(d, v).map_err(|m| EsError::x_content_parse(format!("{}[esql/query] failed to parse field [{name}]", loc(&[name], false))).caused_by(EsError::illegal_argument(m)))?;
            if let Some((_, other)) = nested.iter().find(|(n, _)| n == name) {
                if *other != t {
                    return Err(EsError::illegal_argument(format!(
                        "Setting [{name}] has conflicting values at the top level of the request body and under [settings]; specify it in only one place, or with the same value in both."
                    )));
                }
                continue;
            }
            out.values.push((d.name, t));
        }
        out.values.extend(nested);
        Ok(out)
    }
}

/// `readFromJson`.
fn read_json(d: &Def, v: &Value) -> std::result::Result<Typed, String> {
    match d.kind {
        Kind::Object => match v {
            Value::Bool(_) | Value::Object(_) | Value::Null => Ok(Typed::Json(v.clone())),
            Value::String(s) => parse_bool(s).map(|b| Typed::Json(Value::Bool(b))),
            other => Err(format!("unsupported value [{other}]")),
        },
        Kind::Boolean => match v {
            Value::Bool(b) => Ok(Typed::Bool(*b)),
            Value::String(s) => parse_bool(s).map(Typed::Bool),
            other => parse_bool(&other.to_string()).map(Typed::Bool),
        },
        Kind::Keyword => match v {
            Value::String(s) => read_string(d, s),
            Value::Number(n) => read_string(d, &n.to_string()),
            Value::Bool(b) => read_string(d, &b.to_string()),
            other => Err(format!("unsupported value [{other}]")),
        },
    }
}

fn apply(out: &mut Settings, name: &str, v: Typed) {
    match (name, v) {
        ("time_zone", Typed::Zone(z)) => {
            out.time_zone = z;
            out.time_zone_set = true;
        }
        ("unmapped_fields", Typed::Unmapped(u)) => out.unmapped = u,
        ("column_metadata", Typed::Bool(b)) => out.column_metadata = b,
        ("approximation", Typed::Json(j)) => out.approximation = if j.is_null() { None } else { Some(j) },
        _ => {}
    }
}

/// Validate the query's `SET`s (parsing errors, in order) and fold
/// default < body < `SET`.
pub fn resolve(full: &str, sets: &[QuerySetting], body: &BodySettings) -> Result<Settings> {
    let mut typed = Vec::with_capacity(sets.len());
    for s in sets {
        typed.push(read_set(full, s)?);
    }
    let mut out = Settings::default();
    for (name, v) in &body.values {
        let d = def(name).unwrap();
        if let Some(m) = validate_value(d, v) {
            return Err(EsError::new(400, "verification_exception", format!("Error validating setting [{name}]: {m}")));
        }
        apply(&mut out, name, v.clone());
    }
    for (d, v) in typed {
        apply(&mut out, d.name, v);
    }
    Ok(out)
}
