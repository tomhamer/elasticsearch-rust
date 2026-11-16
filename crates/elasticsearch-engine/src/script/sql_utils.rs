//! `InternalSqlScriptUtils`: the static helpers ES SQL's translated scripts
//! call (`_sql/translate` script sources). Each one evaluates exactly as the
//! SQL query evaluates the function (the shared ES|QL/SQL scalar
//! implementation), with SQL's null-in, null-out semantics.

use super::error::{illegal_arg, SResult};
use super::runtime::Rt;
use super::value::Val;
use crate::esql::parse::BinOp;
use crate::esql::Val as Q;

/// A script value as a SQL value.
fn to_q(v: &Val) -> Q {
    match v {
        Val::Null => Q::Null,
        Val::Bool(b) => Q::Bool(*b),
        Val::Int(i) => Q::Int(*i as i64),
        Val::Long(l) => Q::Long(*l),
        Val::Double(d) => Q::Double(*d),
        Val::Date(ms) | Val::Instant(ms) => Q::Date(*ms),
        Val::Set(l) => Q::from_list(l.ordered().iter().map(to_q).collect()),
        Val::List(l) => Q::from_list(l.ordered().iter().map(to_q).collect()),
        Val::DocValues(d) => Q::from_list(d.values.iter().map(to_q).collect()),
        other => Q::Str(other.to_java_string()),
    }
}

/// The members of a script list argument, nulls kept in place (a `List`
/// of conditions and results, or of `IN` values).
fn items_of(v: Option<&Val>) -> Vec<Q> {
    match v {
        Some(Val::List(l)) => l.ordered().iter().map(to_q).collect(),
        Some(Val::Null) | None => Vec::new(),
        Some(one) => vec![to_q(one)],
    }
}

/// A SQL value as a script value.
fn from_q(v: Q) -> Val {
    match v {
        Q::Null => Val::Null,
        Q::Bool(b) => Val::Bool(b),
        Q::Int(i) => i32::try_from(i).map(Val::Int).unwrap_or(Val::Long(i)),
        Q::Long(l) => Val::Long(l),
        Q::Double(d) => Val::Double(d),
        Q::Date(ms) => Val::Date(ms),
        Q::Nanos(n) => Val::Date(n.div_euclid(crate::esql::NANOS_PER_MILLI)),
        Q::ULong(u) => i64::try_from(u).map(Val::Long).unwrap_or(Val::Double(u as f64)),
        Q::Multi(m) => Val::new_list(m.into_iter().map(from_q).collect()),
        Q::Str(s) => Val::str(&s),
        other => Val::str(&other.to_text()),
    }
}

fn sql(name: &str, args: &[Q]) -> SResult<Val> {
    crate::esql::func::call(name, args, &crate::esql::func::FnCtx::default()).map(from_q).map_err(|e| illegal_arg(e.reason))
}

fn truth(v: &Q) -> Option<bool> {
    match v {
        Q::Bool(b) => Some(*b),
        _ => None,
    }
}

/// `InternalSqlScriptUtils.method(args)`; `None`: no such method.
pub fn call(rt: &mut Rt, method: &str, args: Vec<Val>) -> SResult<Option<Val>> {
    if method == "docValue" {
        let name = args.get(1).map(Val::to_java_string).unwrap_or_default();
        let slot = rt.doc.slot_of(&name);
        return Ok(Some(match rt.doc.values(slot)? {
            Val::DocValues(d) => d.values.first().cloned().unwrap_or(Val::Null),
            other => other,
        }));
    }
    let q: Vec<Q> = args.iter().map(to_q).collect();
    let a = |i: usize| q.get(i).cloned().unwrap_or(Q::Null);
    let bin = |op: BinOp| from_q(crate::esql::func::binop(op, &a(0), &a(1)));
    Ok(Some(match method {
        "add" => bin(BinOp::Add),
        "sub" => bin(BinOp::Sub),
        "mul" => bin(BinOp::Mul),
        "div" => bin(BinOp::Div),
        "mod" => bin(BinOp::Mod),
        "eq" => bin(BinOp::Eq),
        "neq" => bin(BinOp::Neq),
        "lt" => bin(BinOp::Lt),
        "lte" => bin(BinOp::Le),
        "gt" => bin(BinOp::Gt),
        "gte" => bin(BinOp::Ge),
        "and" => bin(BinOp::And),
        "or" => bin(BinOp::Or),
        "neg" => from_q(crate::esql::func::binop(BinOp::Mul, &a(0), &Q::Long(-1))),
        "not" => truth(&a(0)).map(|b| Val::Bool(!b)).unwrap_or(Val::Null),
        "isNull" => Val::Bool(a(0).is_null()),
        "isNotNull" => Val::Bool(!a(0).is_null()),
        "in" => {
            let x = a(0);
            if x.is_null() {
                Val::Null
            } else {
                let items = items_of(args.get(1));
                let mut null = false;
                let mut hit = false;
                for i in &items {
                    match truth(&crate::esql::func::binop(BinOp::Eq, &x, i)) {
                        Some(true) => hit = true,
                        Some(false) => {}
                        None => null = true,
                    }
                }
                if hit { Val::Bool(true) } else if null { Val::Null } else { Val::Bool(false) }
            }
        }
        "convert" => {
            let ty = args.get(1).map(Val::to_java_string).unwrap_or_default().to_ascii_lowercase();
            from_q(crate::esql::func::cast(&a(0), &ty).map_err(|e| illegal_arg(e.reason))?)
        }
        "caseFunction" => {
            let items = items_of(args.first());
            let mut i = 0;
            let mut out = Val::Null;
            while i + 1 < items.len() {
                if truth(&items[i]) == Some(true) {
                    out = from_q(items[i + 1].clone());
                    i = usize::MAX - 1;
                    break;
                }
                i += 2;
            }
            if i != usize::MAX - 1 && items.len() % 2 == 1 {
                out = from_q(items[items.len() - 1].clone());
            }
            out
        }
        "dateTimeExtract" => {
            let part = args.get(2).map(Val::to_java_string).unwrap_or_default();
            let f = match part.as_str() {
                "YEAR" => "YEAR",
                "MONTH_OF_YEAR" => "MONTH",
                "DAY_OF_MONTH" => "DAY_OF_MONTH",
                "HOUR_OF_DAY" => "HOUR",
                "MINUTE_OF_HOUR" => "MINUTE",
                "SECOND_OF_MINUTE" => "SECOND",
                "DAY_OF_YEAR" => "DAY_OF_YEAR",
                "MINUTE_OF_DAY" => "MINUTE_OF_DAY",
                "ISO_DAY_OF_WEEK" => "ISO_DAY_OF_WEEK",
                "ISO_WEEK_OF_YEAR" => "ISO_WEEK_OF_YEAR",
                other => return Err(illegal_arg(format!("unsupported date part [{other}]"))),
            };
            sql(f, &[a(0)])?
        }
        "dayOfWeek" => sql("DAY_OF_WEEK", &[a(0)])?,
        "weekOfYear" => sql("WEEK_OF_YEAR", &[a(0)])?,
        "quarter" => sql("QUARTER", &[a(0)])?,
        "dayName" => sql("DAY_NAME", &[a(0)])?,
        "monthName" => sql("MONTH_NAME", &[a(0)])?,
        "dateAdd" => sql("DATE_ADD", &[a(0), a(1), a(2)])?,
        "datePart" => sql("DATE_PART", &[a(0), a(1)])?,
        "dateTrunc" => sql("DATE_TRUNC", &[a(0), a(1)])?,
        "dateDiff" => sql(crate::esql::func::SQL_DATE_DIFF, &[a(0), a(1), a(2)])?,
        "dateFormat" => match a(1) {
            Q::Str(p) => sql("DATE_FORMAT", &[Q::Str(crate::esql::func::mysql_to_java_pattern(&p)), a(0)])?,
            _ => Val::Null,
        },
        "dateTimeFormat" => sql("DATE_FORMAT", &[a(1), a(0)])?,
        "dateParse" | "dateTimeParse" => sql("DATE_PARSE", &[a(1), a(0)])?,
        "coalesce" => sql("COALESCE", &q)?,
        "greatest" => sql(crate::esql::func::SQL_GREATEST, &q)?,
        "least" => sql(crate::esql::func::SQL_LEAST, &q)?,
        // ES SQL's LOCATE(pattern, source[, start]), literal REPLACE,
        // NULL-as-'' CONCAT, trailing-blank-free LENGTH.
        "locate" => {
            let mut args = vec![a(1), a(0)];
            args.extend(q.iter().skip(2).cloned());
            sql("LOCATE", &args)?
        }
        "replace" => sql(crate::esql::func::SQL_REPLACE, &q)?,
        "concat" => sql(crate::esql::func::SQL_CONCAT, &q)?,
        "length" => sql(crate::esql::func::SQL_LENGTH, &q)?,
        "nullif" => sql("NULLIF", &q)?,
        "iif" => sql("IIF", &q)?,
        // String and math scalars: the SQL function of the same meaning.
        other => {
            let name = match other {
                "ascii" => "ASCII",
                "bitLength" => "BIT_LENGTH",
                "charLength" => "CHAR_LENGTH",
                "concat" => "CONCAT",
                "insert" => "INSERT",
                "lcase" => "LCASE",
                "left" => "LEFT",
                "length" => "LENGTH",
                "locate" => "LOCATE",
                "ltrim" => "LTRIM",
                "octetLength" => "OCTET_LENGTH",
                "position" => "POSITION",
                "repeat" => "REPEAT",
                "replace" => "REPLACE",
                "right" => "RIGHT",
                "rtrim" => "RTRIM",
                "space" => "SPACE",
                "startsWith" => "STARTS_WITH",
                "substring" => "SUBSTRING",
                "trim" => "TRIM",
                "ucase" => "UCASE",
                "abs" => "ABS",
                "acos" => "ACOS",
                "asin" => "ASIN",
                "atan" => "ATAN",
                "atan2" => "ATAN2",
                "cbrt" => "CBRT",
                "ceil" => "CEIL",
                "cos" => "COS",
                "cosh" => "COSH",
                "degrees" => "DEGREES",
                "e" => "E",
                "exp" => "EXP",
                "floor" => "FLOOR",
                "log" => "LOG",
                "log10" => "LOG10",
                "pi" => "PI",
                "power" => "POWER",
                "radians" => "RADIANS",
                "round" => "ROUND",
                "sign" => "SIGN",
                "sin" => "SIN",
                "sinh" => "SINH",
                "sqrt" => "SQRT",
                "tan" => "TAN",
                "truncate" => "TRUNCATE",
                "cot" => "COT",
                "expm1" => "EXPM1",
                "character" => "CHAR",
                "random" => "RANDOM",
                _ => return Ok(None),
            };
            sql(name, &q)?
        }
    }))
}
