//! Instance methods of the whitelisted types (String, List, Map, Set,
//! numbers, dates, streams, doc values, ...).

use super::error::{class_cast, illegal_arg, illegal_state, npe, out_of_bounds, unsupported, SResult, ScriptErr};
use super::runtime::{binary, call_static, iter_values, Rt};
use super::value::{java_double_string, CollKind, JList, JMap, Key, MapKind, Val, CHRONO_UNITS, DAYS, MONTHS};
use super::ast::BinOp;
use chrono::{Datelike, Timelike};
use indexmap::IndexMap;
use std::rc::Rc;

/// A `GeoPoint` doc value (`{lat, lon}`).
fn is_geo_point(v: &Val) -> bool {
    match v {
        Val::Map(m) => {
            let m = m.borrow();
            m.len() == 2 && m.contains_key(&Key::str("lat")) && m.contains_key(&Key::str("lon"))
        }
        _ => false,
    }
}

/// `GeoPoint` accessors and distance functions (meters).
fn geo_point_method(p: &Val, name: &str, args: &[Val]) -> SResult<Val> {
    let Val::Map(m) = p else { return Err(illegal_arg("not a geo point")) };
    let (lat, lon) = {
        let m = m.borrow();
        (m.get(&Key::str("lat")).and_then(Val::as_f64).unwrap_or(0.0), m.get(&Key::str("lon")).and_then(Val::as_f64).unwrap_or(0.0))
    };
    Ok(match (name, args.len()) {
        ("getLat", 0) => Val::Double(lat),
        ("getLon", 0) => Val::Double(lon),
        ("getGeohash", 0) => Val::string(crate::values::geohash_encode(lat, lon, 12)),
        ("arcDistance", 2) => Val::Double(rustic_core::geo::haversine_meters(lat, lon, args[0].to_f64()?, args[1].to_f64()?)),
        ("planeDistance", 2) => Val::Double(rustic_core::geo::plane_meters(lat, lon, args[0].to_f64()?, args[1].to_f64()?)),
        ("geohashDistance", 1) => {
            let h = args[0].to_java_string();
            let (la, lo) = crate::values::geohash_decode(&h).ok_or_else(|| illegal_arg(format!("unsupported symbol [{h}] in geohash [{h}]")))?;
            Val::Double(rustic_core::geo::haversine_meters(lat, lon, la, lo))
        }
        _ => return Err(illegal_arg(format!("member method [org.elasticsearch.common.geo.GeoPoint, {name}/{}] not found", args.len()))),
    })
}

/// A `geo_shape` doc value (see `doc::geo_shape_value`).
fn is_geo_shape(v: &Val) -> bool {
    matches!(v, Val::Map(m) if m.borrow().contains_key(&Key::str(super::doc::GEO_SHAPE_KEY)))
}

/// `ScriptDocValues.Geometry` methods of `geo_point` / `geo_shape` doc values.
fn geo_values_method(values: &[Val], name: &str) -> SResult<Val> {
    let first = values.first().ok_or_else(|| illegal_state(super::doc::MISSING_VALUE))?;
    let get = |v: &Val, k: &str| match v {
        Val::Map(m) => m.borrow().get(&Key::str(k)).cloned().unwrap_or(Val::Null),
        _ => Val::Null,
    };
    if is_geo_shape(first) {
        return Ok(match name {
            "getCentroid" => get(first, "centroid"),
            "getBoundingBox" => get(first, "bbox"),
            "getLabelPosition" => super::doc::geo_shape_label(first),
            "getDimensionalType" => get(first, "dim"),
            "getMercatorWidth" => get(first, "mw"),
            "getMercatorHeight" => get(first, "mh"),
            _ => Val::Null,
        });
    }
    // geo_point values: centroid of the points, their bounding box.
    let pts: Vec<(f64, f64)> = values.iter().map(|v| (get(v, "lat").as_f64().unwrap_or(0.0), get(v, "lon").as_f64().unwrap_or(0.0))).collect();
    let n = pts.len() as f64;
    let (top, bottom) = pts.iter().fold((f64::NEG_INFINITY, f64::INFINITY), |a, p| (a.0.max(p.0), a.1.min(p.0)));
    let (left, right) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    let centroid = || {
        let mut m = IndexMap::new();
        m.insert(Key::str("lat"), Val::Double(pts.iter().map(|p| p.0).sum::<f64>() / n));
        m.insert(Key::str("lon"), Val::Double(pts.iter().map(|p| p.1).sum::<f64>() / n));
        Val::new_map(m)
    };
    use crate::aggs::geo_shape::{lat_mercator, lon_mercator};
    Ok(match name {
        "getCentroid" => centroid(),
        "getLabelPosition" => if pts.len() == 1 { first.clone() } else { centroid() },
        "getBoundingBox" => super::doc::geo_bbox_val(top, left, bottom, right),
        "getDimensionalType" => Val::Int(0),
        "getMercatorWidth" => Val::Double(lon_mercator(right) - lon_mercator(left)),
        "getMercatorHeight" => Val::Double(lat_mercator(top) - lat_mercator(bottom)),
        _ => Val::Null,
    })
}

/// A `rank_vectors` doc value ([`super::doc::rank_vectors_value`]).
pub(crate) fn is_rank_vectors(v: &Val) -> bool {
    matches!(v, Val::Map(m) if m.borrow().contains_key(&Key::str(super::doc::RANK_VECTORS_KEY)))
}

fn not_found(recv: &Val, name: &str, n: usize) -> Box<ScriptErr> {
    illegal_arg(format!("dynamic method [{}, {name}/{n}] not found", recv.type_name()))
}

/// JDK frames ES prints above the script frame for `ArrayList.get` out of bounds.
const ARRAYLIST_GET_FRAMES: &[&str] = &[
    "java.base/jdk.internal.util.Preconditions.outOfBounds(Preconditions.java:100)",
    "java.base/jdk.internal.util.Preconditions.outOfBoundsCheckIndex(Preconditions.java:106)",
    "java.base/jdk.internal.util.Preconditions.checkIndex(Preconditions.java:302)",
    "java.base/java.util.Objects.checkIndex(Objects.java:365)",
    "java.base/java.util.ArrayList.get(ArrayList.java:428)",
];
const STRING_CHARAT_FRAMES: &[&str] = &[
    "java.base/jdk.internal.util.Preconditions$1.apply(Preconditions.java:55)",
    "java.base/jdk.internal.util.Preconditions$1.apply(Preconditions.java:52)",
    "java.base/jdk.internal.util.Preconditions$4.apply(Preconditions.java:213)",
    "java.base/jdk.internal.util.Preconditions$4.apply(Preconditions.java:210)",
    "java.base/jdk.internal.util.Preconditions.outOfBounds(Preconditions.java:98)",
    "java.base/jdk.internal.util.Preconditions.outOfBoundsCheckIndex(Preconditions.java:106)",
    "java.base/jdk.internal.util.Preconditions.checkIndex(Preconditions.java:302)",
    "java.base/java.lang.String.checkIndex(String.java:5279)",
    "java.base/java.lang.StringLatin1.charAt(StringLatin1.java:58)",
    "java.base/java.lang.String.charAt(String.java:1843)",
];
const STRING_SUBSTRING_FRAMES: &[&str] = &[
    "java.base/jdk.internal.util.Preconditions$1.apply(Preconditions.java:55)",
    "java.base/jdk.internal.util.Preconditions$1.apply(Preconditions.java:52)",
    "java.base/jdk.internal.util.Preconditions$4.apply(Preconditions.java:213)",
    "java.base/jdk.internal.util.Preconditions$4.apply(Preconditions.java:210)",
    "java.base/jdk.internal.util.Preconditions.outOfBounds(Preconditions.java:98)",
    "java.base/jdk.internal.util.Preconditions.outOfBoundsCheckFromToIndex(Preconditions.java:112)",
    "java.base/jdk.internal.util.Preconditions.checkFromToIndex(Preconditions.java:349)",
    "java.base/java.lang.String.checkBoundsBeginEnd(String.java:5312)",
    "java.base/java.lang.String.substring(String.java:3274)",
];
const UNMODIFIABLE_ADD_FRAMES: &[&str] = &["java.base/java.util.Collections$UnmodifiableCollection.add(Collections.java:1095)"];

/// ES resolves `def` method calls against the allowlist of the receiver's
/// runtime class (and its supertypes); a method we happen to implement but
/// ES does not allow must still fail the same way.
fn dyn_check(recv: &Val, name: &str, n: usize) -> SResult<()> {
    let supers: &[&str] = match recv {
        Val::Map(m) => match m.kind {
            MapKind::Tree => &["java.util.TreeMap", "java.util.AbstractMap", "java.util.NavigableMap", "java.util.SortedMap", "java.util.Map", "java.lang.Object"],
            MapKind::Linked => &["java.util.LinkedHashMap", "java.util.HashMap", "java.util.AbstractMap", "java.util.Map", "java.lang.Object"],
            _ => &["java.util.HashMap", "java.util.AbstractMap", "java.util.Map", "java.lang.Object"],
        },
        Val::Set(l) => match l.kind {
            CollKind::TreeSet => &["java.util.TreeSet", "java.util.AbstractSet", "java.util.NavigableSet", "java.util.SortedSet", "java.util.Set", "java.util.AbstractCollection", "java.util.Collection", "java.lang.Iterable", "java.lang.Object"],
            CollKind::LinkedSet => &["java.util.LinkedHashSet", "java.util.HashSet", "java.util.AbstractSet", "java.util.Set", "java.util.AbstractCollection", "java.util.Collection", "java.lang.Iterable", "java.lang.Object"],
            _ => &["java.util.HashSet", "java.util.AbstractSet", "java.util.Set", "java.util.AbstractCollection", "java.util.Collection", "java.lang.Iterable", "java.lang.Object"],
        },
        _ => return Ok(()),
    };
    let key = format!("{name}/{n}");
    if super::whitelist::WHITELIST.iter().any(|e| supers.contains(&e.0) && e.1.contains(&key.as_str())) {
        Ok(())
    } else {
        Err(not_found(recv, name, n))
    }
}

fn arg(args: &[Val], i: usize) -> SResult<&Val> {
    args.get(i).ok_or_else(|| illegal_arg("missing argument"))
}

fn str_arg(args: &[Val], i: usize) -> SResult<String> {
    match arg(args, i)? {
        Val::Str(s) => Ok(s.to_string()),
        Val::Char(c) => Ok(String::from_utf16_lossy(&[*c])),
        Val::Null => Err(npe("argument is null")),
        other => Err(class_cast(format!("Cannot cast {} to java.lang.String", other.type_name()))),
    }
}

fn int_arg(args: &[Val], i: usize) -> SResult<i64> {
    arg(args, i)?.to_index()
}

/// Char offset → byte offset (Java strings index UTF-16 units; we index chars).
fn char_slice(s: &str, a: i64, b: i64) -> SResult<String> {
    let n = s.chars().count() as i64;
    if a < 0 || b > n || a > b {
        return Err(ScriptErr::new("string_index_out_of_bounds_exception", format!("Range [{a}, {b}) out of bounds for length {n}")).frames(STRING_SUBSTRING_FRAMES));
    }
    Ok(s.chars().skip(a as usize).take((b - a) as usize).collect())
}

fn find_chars(hay: &str, needle: &str, from: usize, last: bool) -> i64 {
    let hc: Vec<char> = hay.chars().collect();
    let nc: Vec<char> = needle.chars().collect();
    if nc.is_empty() {
        return if last { hc.len() as i64 } else { from.min(hc.len()) as i64 };
    }
    if nc.len() > hc.len() {
        return -1;
    }
    let range: Box<dyn Iterator<Item = usize>> = if last { Box::new((0..=(hc.len() - nc.len()).min(from)).rev()) } else { Box::new(from..=hc.len() - nc.len()) };
    for i in range {
        if hc[i..i + nc.len()] == nc[..] {
            return i as i64;
        }
    }
    -1
}

fn new_set(v: Vec<Val>) -> Val {
    Val::new_set(CollKind::LinkedSet, v)
}

fn sort_vals(rt: &mut Rt, items: &mut [Val], cmp: Option<&Val>) -> SResult<()> {
    let mut err = None;
    match cmp {
        Some(f) if !f.is_null() => {
            let f = f.clone();
            // Insertion-stable merge sort through the comparator.
            let mut v: Vec<Val> = items.to_vec();
            merge_sort(&mut v, &mut |a, b| {
                if err.is_some() {
                    return std::cmp::Ordering::Equal;
                }
                match rt.call_fn(&f, &[a.clone(), b.clone()]).and_then(|r| r.to_i64()) {
                    Ok(x) => x.cmp(&0),
                    Err(e) => {
                        err = Some(e);
                        std::cmp::Ordering::Equal
                    }
                }
            });
            items.clone_from_slice(&v);
        }
        _ => items.sort_by(|a, b| match a.compare(b) {
            Ok(o) => o,
            Err(e) => {
                if err.is_none() {
                    err = Some(e);
                }
                std::cmp::Ordering::Equal
            }
        }),
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn merge_sort(v: &mut Vec<Val>, cmp: &mut dyn FnMut(&Val, &Val) -> std::cmp::Ordering) {
    if v.len() <= 1 {
        return;
    }
    let mid = v.len() / 2;
    let mut right = v.split_off(mid);
    merge_sort(v, cmp);
    merge_sort(&mut right, cmp);
    let left = std::mem::take(v);
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if cmp(&right[j], &left[i]).is_lt() {
            v.push(right[j].clone());
            j += 1;
        } else {
            v.push(left[i].clone());
            i += 1;
        }
    }
    v.extend_from_slice(&left[i..]);
    v.extend_from_slice(&right[j..]);
}

fn sum_vals(vals: &[Val]) -> SResult<Val> {
    let mut acc = Val::Int(0);
    let all_int = vals.iter().all(|v| matches!(v, Val::Int(_)));
    if !all_int {
        acc = if vals.iter().all(|v| matches!(v, Val::Int(_) | Val::Long(_))) { Val::Long(0) } else { Val::Double(0.0) };
    }
    for v in vals {
        acc = binary(BinOp::Add, &acc, v)?;
    }
    Ok(acc)
}

fn opt(v: Option<Val>) -> Val {
    Val::Optional(v.map(Rc::new))
}

fn date_parts(ms: i64) -> chrono::NaiveDateTime {
    crate::date::from_millis(ms)
}

fn add_months(ms: i64, months: i64) -> i64 {
    crate::date::add_units(ms, crate::date::CalUnit::Month, months, crate::date::Tz::UTC)
}

fn chrono_between(unit: u8, a: i64, b: i64) -> i64 {
    let d = b - a;
    match CHRONO_UNITS[unit as usize] {
        "MILLIS" => d,
        "SECONDS" => d / 1000,
        "MINUTES" => d / 60_000,
        "HOURS" => d / 3_600_000,
        "HALF_DAYS" => d / 43_200_000,
        "DAYS" => d / 86_400_000,
        "WEEKS" => d / 604_800_000,
        "MONTHS" | "YEARS" => {
            let (x, y) = (date_parts(a), date_parts(b));
            let mut months = (y.year() as i64 - x.year() as i64) * 12 + (y.month() as i64 - x.month() as i64);
            // Incomplete last month.
            let xr = (x.day(), x.num_seconds_from_midnight(), x.nanosecond());
            let yr = (y.day(), y.num_seconds_from_midnight(), y.nanosecond());
            if months > 0 && yr < xr {
                months -= 1;
            } else if months < 0 && yr > xr {
                months += 1;
            }
            if CHRONO_UNITS[unit as usize] == "YEARS" { months / 12 } else { months }
        }
        _ => d,
    }
}

fn truncate_ms(ms: i64, unit: u8) -> i64 {
    let step = match CHRONO_UNITS[unit as usize] {
        "MILLIS" => 1,
        "SECONDS" => 1000,
        "MINUTES" => 60_000,
        "HOURS" => 3_600_000,
        "HALF_DAYS" => 43_200_000,
        "DAYS" => 86_400_000,
        _ => return ms,
    };
    ms.div_euclid(step) * step
}

/// Date-like (ZonedDateTime / Instant) methods.
fn date_method(recv: &Val, ms: i64, zoned: bool, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let wrap = |m: i64| if zoned { Val::Date(m) } else { Val::Instant(m) };
    let dt = date_parts(ms);
    let n = args.len();
    let num = |i: usize| -> SResult<i64> { arg(args, i)?.to_i64() };
    Ok(Some(match (name, n) {
        ("getMillis" | "toEpochMilli" | "getTime", 0) => Val::Long(ms),
        ("toEpochSecond" | "getEpochSecond", 0) => Val::Long(ms.div_euclid(1000)),
        ("toInstant", 0) => Val::Instant(ms),
        ("atZone", 1) => Val::Date(ms),
        ("withZoneSameInstant" | "withZoneSameLocal", 1) => {
            let z = arg(args, 0)?.to_java_string();
            if matches!(z.as_str(), "Z" | "UTC" | "GMT" | "+00:00") {
                Val::Date(ms)
            } else {
                return Err(unsupported(format!("time zone [{z}] is not supported in scripts (UTC only)")));
            }
        }
        ("getYear", 0) if zoned => Val::Int(dt.year()),
        ("getMonthValue", 0) if zoned => Val::Int(dt.month() as i32),
        ("getMonth", 0) if zoned => Val::Month(dt.month() as u8),
        ("getMonthOfYear", 0) if zoned => Val::Int(dt.month() as i32),
        ("getDayOfMonth", 0) if zoned => Val::Int(dt.day() as i32),
        ("getDayOfYear", 0) if zoned => Val::Int(dt.ordinal() as i32),
        ("getDayOfWeek" | "getDayOfWeekEnum", 0) if zoned => Val::DayOfWeek(dt.weekday().number_from_monday() as u8),
        ("getHour" | "getHourOfDay", 0) if zoned => Val::Int(dt.hour() as i32),
        ("getMinute" | "getMinuteOfHour", 0) if zoned => Val::Int(dt.minute() as i32),
        ("getSecond" | "getSecondOfMinute", 0) if zoned => Val::Int(dt.second() as i32),
        ("getNano", 0) => Val::Int(dt.nanosecond() as i32),
        ("getMillisOfSecond", 0) => Val::Int((dt.nanosecond() / 1_000_000) as i32),
        ("getZone" | "getOffset", 0) if zoned => Val::str("Z"),
        ("plusMillis", 1) => wrap(ms + num(0)?),
        ("minusMillis", 1) => wrap(ms - num(0)?),
        ("plusSeconds", 1) => wrap(ms + num(0)? * 1000),
        ("minusSeconds", 1) => wrap(ms - num(0)? * 1000),
        ("plusMinutes", 1) => wrap(ms + num(0)? * 60_000),
        ("minusMinutes", 1) => wrap(ms - num(0)? * 60_000),
        ("plusHours", 1) => wrap(ms + num(0)? * 3_600_000),
        ("minusHours", 1) => wrap(ms - num(0)? * 3_600_000),
        ("plusDays", 1) => wrap(ms + num(0)? * 86_400_000),
        ("minusDays", 1) => wrap(ms - num(0)? * 86_400_000),
        ("plusWeeks", 1) => wrap(ms + num(0)? * 604_800_000),
        ("minusWeeks", 1) => wrap(ms - num(0)? * 604_800_000),
        ("plusMonths", 1) => wrap(add_months(ms, num(0)?)),
        ("minusMonths", 1) => wrap(add_months(ms, -num(0)?)),
        ("plusYears", 1) => wrap(add_months(ms, num(0)? * 12)),
        ("minusYears", 1) => wrap(add_months(ms, -num(0)? * 12)),
        ("plus" | "minus", 2) => match arg(args, 1)? {
            Val::ChronoUnit(u) => {
                let k = num(0)? * if name == "minus" { -1 } else { 1 };
                match CHRONO_UNITS[*u as usize] {
                    "MONTHS" => wrap(add_months(ms, k)),
                    "YEARS" => wrap(add_months(ms, k * 12)),
                    "DECADES" => wrap(add_months(ms, k * 120)),
                    unit => {
                        let step = match unit {
                            "NANOS" | "MICROS" => return Err(illegal_arg(format!("Unsupported unit: {unit}"))),
                            "MILLIS" => 1,
                            "SECONDS" => 1000,
                            "MINUTES" => 60_000,
                            "HOURS" => 3_600_000,
                            "HALF_DAYS" => 43_200_000,
                            "DAYS" => 86_400_000,
                            "WEEKS" => 604_800_000,
                            _ => return Err(illegal_arg(format!("Unsupported unit: {unit}"))),
                        };
                        wrap(ms + k * step)
                    }
                }
            }
            other => return Err(class_cast(format!("Cannot cast {} to java.time.temporal.TemporalUnit", other.type_name()))),
        },
        ("isBefore", 1) => Val::Bool(ms < date_ms(arg(args, 0)?)?),
        ("isAfter", 1) => Val::Bool(ms > date_ms(arg(args, 0)?)?),
        ("isEqual", 1) => Val::Bool(ms == date_ms(arg(args, 0)?)?),
        ("compareTo", 1) => Val::Int(ms.cmp(&date_ms(arg(args, 0)?)?) as i32),
        ("equals", 1) => Val::Bool(recv.equals(arg(args, 0)?)),
        ("truncatedTo", 1) => match arg(args, 0)? {
            Val::ChronoUnit(u) => wrap(truncate_ms(ms, *u)),
            other => return Err(class_cast(format!("Cannot cast {} to java.time.temporal.TemporalUnit", other.type_name()))),
        },
        ("until", 2) => match arg(args, 1)? {
            Val::ChronoUnit(u) => Val::Long(chrono_between(*u, ms, date_ms(arg(args, 0)?)?)),
            _ => return Err(illegal_arg("until expects a ChronoUnit")),
        },
        ("format", 1) => {
            let f = arg(args, 0)?.to_java_string();
            if super::timefmt::is_formatter(&f) {
                Val::string(super::timefmt::format(&f, ms).map_err(|e| illegal_arg(e))?)
            } else {
                let fmt = crate::date::DateFormatter::new(&f).map_err(|e| illegal_arg(e.reason))?;
                Val::string(fmt.format(ms, crate::date::Tz::UTC))
            }
        }
        ("withNano" | "withSecond" | "withMinute" | "withHour" | "withDayOfMonth" | "withDayOfYear" | "withMonth" | "withYear", 1) if zoned => {
            use chrono::{Datelike, Timelike};
            let v = num(0)?;
            let bad = |what: &str, lo: i64, hi: i64| ScriptErr::new("date_time_exception", format!("Invalid value for {what} (valid values {lo} - {hi}): {v}"));
            let u = v.try_into().ok();
            let r = match name {
                "withNano" => u.and_then(|x: u32| dt.with_nanosecond(x)).ok_or_else(|| bad("NanoOfSecond", 0, 999_999_999)),
                "withSecond" => u.and_then(|x: u32| dt.with_second(x)).ok_or_else(|| bad("SecondOfMinute", 0, 59)),
                "withMinute" => u.and_then(|x: u32| dt.with_minute(x)).ok_or_else(|| bad("MinuteOfHour", 0, 59)),
                "withHour" => u.and_then(|x: u32| dt.with_hour(x)).ok_or_else(|| bad("HourOfDay", 0, 23)),
                "withDayOfMonth" => u.and_then(|x: u32| dt.with_day(x)).ok_or_else(|| bad("DayOfMonth", 1, 31)),
                "withDayOfYear" => u.and_then(|x: u32| dt.with_ordinal(x)).ok_or_else(|| bad("DayOfYear", 1, 366)),
                "withMonth" => u.and_then(|x: u32| dt.with_month(x)).ok_or_else(|| bad("MonthOfYear", 1, 12)),
                _ => i32::try_from(v).ok().and_then(|x| dt.with_year(x)).ok_or_else(|| bad("Year", -999_999_999, 999_999_999)),
            }?;
            Val::Date(r.and_utc().timestamp_millis())
        }
        ("toLocalDate" | "toLocalDateTime" | "toOffsetDateTime", 0) => Val::Date(ms),
        ("toString", 0) => Val::utf16(recv.string_units()),
        ("hashCode", 0) => Val::Int((ms ^ (ms >> 32)) as i32),
        _ => return Ok(None),
    }))
}

fn date_ms(v: &Val) -> SResult<i64> {
    match v {
        Val::Date(m) | Val::Instant(m) => Ok(*m),
        other => Err(class_cast(format!("Cannot cast {} to java.time.chrono.ChronoZonedDateTime", other.type_name()))),
    }
}

fn collector(rt: &mut Rt, items: Vec<Val>, c: &Val) -> SResult<Val> {
    let (kind, cargs): (String, Vec<Val>) = match c {
        Val::Obj(o) => match &**o {
            super::jmisc::Obj::Collector { kind, args } => (kind.clone(), args.clone()),
            _ => return Err(class_cast(format!("Cannot cast {} to java.util.stream.Collector", c.type_name()))),
        },
        Val::MethodRef(r) if r.0 == "Collectors" => {
            let (k, a) = r.1.split_once(':').unwrap_or((&r.1, ""));
            (k.to_string(), if a.is_empty() { Vec::new() } else { vec![Val::str(a)] })
        }
        Val::Null => return Err(npe("collector is null")),
        other => return Err(class_cast(format!("Cannot cast {} to java.util.stream.Collector", other.type_name()))),
    };
    let a = |i: usize| cargs.get(i).cloned().unwrap_or(Val::Null);
    let na = cargs.len();
    Ok(match kind.as_str() {
        "toList" | "toUnmodifiableList" => {
            let allocation = super::allocation::Allocation::values(items.capacity())?;
            Val::new_list(items).allocated(allocation)
        },
        "toCollection" => {
            let coll = rt.call_fn(&a(0), &[])?;
            for v in items {
                call_method(rt, coll.clone(), "add", vec![v])?;
            }
            coll
        }
        "toSet" | "toUnmodifiableSet" => {
            let s = Val::new_set(CollKind::HashSet, Vec::new());
            if let Val::Set(l) = &s {
                for v in items {
                    set_add(rt, l, v)?;
                }
            }
            s
        }
        "joining" => {
            let sep = cargs.first().map(Val::string_units).unwrap_or_default();
            let mut out = cargs.get(1).map(Val::string_units).unwrap_or_default();
            for (i, value) in items.iter().enumerate() {
                if i != 0 { out.extend_from_slice(&sep); }
                out.extend(value.string_units());
            }
            if let Some(suffix) = cargs.get(2) { out.extend(suffix.string_units()); }
            Val::utf16(out)
        }
        "counting" => Val::Long(items.len() as i64),
        "summingInt" | "summingLong" | "summingDouble" | "averagingInt" | "averagingLong" | "averagingDouble" => {
            let f = a(0);
            let n = items.len();
            let (mut si, mut sd) = (0i64, 0f64);
            for v in items {
                let r = rt.call_fn(&f, &[v])?;
                if r.is_null() {
                    return Err(npe("Cannot invoke \"java.lang.Number.longValue()\" because the return value of the mapper is null"));
                }
                si = si.wrapping_add(r.to_i64()?);
                sd += r.to_f64()?;
            }
            match kind.as_str() {
                "summingInt" => Val::Int(si as i32),
                "summingLong" => Val::Long(si),
                "summingDouble" => Val::Double(sd),
                "averagingDouble" => Val::Double(if n == 0 { 0.0 } else { sd / n as f64 }),
                _ => Val::Double(if n == 0 { 0.0 } else { si as f64 / n as f64 }),
            }
        }
        "minBy" | "maxBy" => {
            let cmp = Some(a(0));
            let mut best: Option<Val> = None;
            for v in items {
                best = Some(match best {
                    None => v,
                    Some(b) => {
                        let o = tree_cmp(rt, &cmp, &v, &b)?;
                        if (kind == "maxBy" && o.is_gt()) || (kind == "minBy" && o.is_lt()) { v } else { b }
                    }
                });
            }
            Val::Optional(best.map(Rc::new))
        }
        "mapping" => {
            let f = a(0);
            let _mapped = super::allocation::Allocation::values(items.len())?;
            let mut mapped = Vec::with_capacity(items.len());
            for v in items {
                mapped.push(rt.call_fn(&f, &[v])?);
            }
            collector(rt, mapped, &a(1))?
        }
        "filtering" => {
            let f = a(0);
            let _kept = super::allocation::Allocation::values(items.len())?;
            let mut kept = Vec::with_capacity(items.len());
            for v in items {
                if rt.call_fn(&f, &[v.clone()])?.as_bool()? {
                    kept.push(v);
                }
            }
            collector(rt, kept, &a(1))?
        }
        "collectingAndThen" => {
            let r = collector(rt, items, &a(0))?;
            rt.call_fn(&a(1), &[r])?
        }
        "groupingBy" => {
            let f = a(0);
            let downstream = if na >= 2 { Some(a(na - 1)) } else { None };
            let _groups = super::allocation::Allocation::entries(items.len())?;
            // Each group Vec has at least four slots and grows geometrically;
            // the total capacity is bounded by four times the input length.
            let _group_values = super::allocation::Allocation::values(items.len().saturating_mul(4))?;
            let mut groups: IndexMap<Key, Vec<Val>> = IndexMap::with_capacity(items.len());
            for v in items {
                let k = rt.call_fn(&f, &[v.clone()])?;
                if k.is_null() {
                    return Err(npe("element cannot be mapped to a null key"));
                }
                groups.entry(Key::from_val(&k)).or_default().push(v);
            }
            let allocation = super::allocation::Allocation::entries(groups.len())?;
            let mut m = IndexMap::with_capacity(groups.len());
            for (k, vs) in groups {
                let val = match &downstream {
                    Some(d) => collector(rt, vs, d)?,
                    None => {
                        let allocation = super::allocation::Allocation::values(vs.capacity())?;
                        Val::new_list(vs).allocated(allocation)
                    },
                };
                m.insert(k, val);
            }
            if na == 3 {
                let target = rt.call_fn(&a(1), &[])?;
                for (k, v) in m {
                    call_method(rt, target.clone(), "put", vec![k.to_val(), v])?;
                }
                target
            } else {
                Val::new_hash_map(m).allocated(allocation)
            }
        }
        "partitioningBy" => {
            let f = a(0);
            let downstream = if na >= 2 { Some(a(1)) } else { None };
            let _partition = super::allocation::Allocation::values(items.len().saturating_mul(2))?;
            let (mut no, mut yes) = (Vec::with_capacity(items.len()), Vec::with_capacity(items.len()));
            for v in items {
                if rt.call_fn(&f, &[v.clone()])?.as_bool()? {
                    yes.push(v);
                } else {
                    no.push(v);
                }
            }
            let fin = |rt: &mut Rt, vs: Vec<Val>| -> SResult<Val> {
                match &downstream {
                    Some(d) => collector(rt, vs, d),
                    None => {
                        let allocation = super::allocation::Allocation::values(vs.capacity())?;
                        Ok(Val::new_list(vs).allocated(allocation))
                    },
                }
            };
            let allocation = super::allocation::Allocation::entries(2)?;
            let mut m = IndexMap::with_capacity(2);
            m.insert(Key::Bool(false), fin(rt, no)?);
            m.insert(Key::Bool(true), fin(rt, yes)?);
            Val::new_map(m).allocated(allocation)
        }
        "toMap" | "toUnmodifiableMap" => {
            let (kf, vf) = (a(0), a(1));
            let merge = if na >= 3 { Some(a(2)) } else { None };
            let allocation = super::allocation::Allocation::entries(items.len())?;
            let mut m: IndexMap<Key, Val> = IndexMap::with_capacity(items.len());
            for v in items {
                let k = rt.call_fn(&kf, &[v.clone()])?;
                let val = rt.call_fn(&vf, &[v])?;
                if val.is_null() {
                    return Err(npe("Cannot invoke \"Object.getClass()\" because \"value\" is null"));
                }
                let key = Key::from_val(&k);
                if let Some(old) = m.get(&key).cloned() {
                    match &merge {
                        Some(mf) => {
                            let nv = rt.call_fn(mf, &[old, val])?;
                            m.insert(key, nv);
                        }
                        None => return Err(illegal_state(format!("Duplicate key {} (attempted merging values {} and {})", k.to_java_string(), old.to_java_string(), val.to_java_string()))),
                    }
                } else {
                    m.insert(key, val);
                }
            }
            if na == 4 {
                let target = rt.call_fn(&a(3), &[])?;
                for (k, v) in m {
                    call_method(rt, target.clone(), "put", vec![k.to_val(), v])?;
                }
                target
            } else {
                Val::new_hash_map(m).allocated(allocation)
            }
        }
        "reducing" => match na {
            1 => {
                let op = a(0);
                let mut acc: Option<Val> = None;
                for v in items {
                    acc = Some(match acc {
                        None => v,
                        Some(x) => rt.call_fn(&op, &[x, v])?,
                    });
                }
                Val::Optional(acc.map(Rc::new))
            }
            2 => {
                let op = a(1);
                let mut acc = a(0);
                for v in items {
                    acc = rt.call_fn(&op, &[acc, v])?;
                }
                acc
            }
            _ => {
                let (mapper, op) = (a(1), a(2));
                let mut acc = a(0);
                for v in items {
                    let mv = rt.call_fn(&mapper, &[v])?;
                    acc = rt.call_fn(&op, &[acc, mv])?;
                }
                acc
            }
        },
        other => return Err(unsupported(format!("Collectors.{other} is not supported"))),
    })
}

/// Stream operations (eager over a snapshot).
fn stream_method(rt: &mut Rt, items: &Rc<super::value::JStream>, name: &str, args: Vec<Val>) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("filter", 1) => {
            let allocation = super::allocation::Allocation::new();
            let mut out = Vec::new();
            for v in items.iter() {
                if rt.call_fn(&args[0], std::slice::from_ref(v))?.as_bool()? { allocation.push(&mut out, v.clone())?; }
            }
            Val::stream_from_values(out, allocation)
        }
        ("map" | "mapToObj", 1) => Val::stream_from_iter(items.iter().map(|v| rt.call_fn(&args[0], std::slice::from_ref(v))))?,
        ("mapToInt", 1) => Val::stream_from_iter(items.iter().map(|v| rt.call_fn(&args[0], std::slice::from_ref(v)).and_then(|r| super::runtime::cast(&super::ast::Ty::Int, r))))?,
        ("mapToLong", 1) => Val::stream_from_iter(items.iter().map(|v| rt.call_fn(&args[0], std::slice::from_ref(v)).and_then(|r| super::runtime::cast(&super::ast::Ty::Long, r))))?,
        ("mapToDouble", 1) => Val::stream_from_iter(items.iter().map(|v| rt.call_fn(&args[0], std::slice::from_ref(v)).and_then(|r| super::runtime::cast(&super::ast::Ty::Double, r))))?,
        ("asDoubleStream", 0) => Val::stream_from_iter(items.iter().map(|v| v.to_f64().map(Val::Double)))?,
        ("asLongStream", 0) => Val::stream_from_iter(items.iter().map(|v| v.to_i64().map(Val::Long)))?,
        ("boxed" | "sequential" | "parallel" | "stream", 0) => Val::Stream(items.clone()),
        ("flatMap", 1) => {
            let allocation = super::allocation::Allocation::new();
            let mut out = Vec::new();
            for v in items.iter() {
                let r = rt.call_fn(&args[0], std::slice::from_ref(v))?;
                let _copy = super::allocation::Allocation::iterable(&r)?;
                for value in iter_values(&r)? { allocation.push(&mut out, value)?; }
            }
            Val::stream_from_values(out, allocation)
        }
        ("sorted", 0 | 1) => {
            let allocation = super::allocation::Allocation::values(items.len())?;
            let mut v = items.values.clone();
            sort_vals(rt, &mut v, args.first())?;
            Val::stream_from_values(v, allocation)
        }
        ("distinct", 0) => {
            let allocation = super::allocation::Allocation::new();
            let mut out: Vec<Val> = Vec::new();
            for v in items.iter() {
                if !out.iter().any(|x| x.equals(v)) { allocation.push(&mut out, v.clone())?; }
            }
            Val::stream_from_values(out, allocation)
        }
        ("limit", 1) => Val::stream_from_iter(items.iter().take(args[0].to_i64()?.max(0) as usize).cloned().map(Ok))?,
        ("skip", 1) => Val::stream_from_iter(items.iter().skip(args[0].to_i64()?.max(0) as usize).cloned().map(Ok))?,
        ("peek", 1) => {
            for v in items.iter() {
                rt.call_fn(&args[0], std::slice::from_ref(v))?;
            }
            Val::Stream(items.clone())
        }
        ("count", 0) => Val::Long(items.len() as i64),
        ("sum", 0) => sum_vals(items)?,
        ("average", 0) => {
            if items.is_empty() {
                opt(None)
            } else {
                let s: f64 = items.iter().map(|v| v.to_f64()).sum::<SResult<f64>>()?;
                opt(Some(Val::Double(s / items.len() as f64)))
            }
        }
        ("max" | "min", 0 | 1) => {
            let mut best: Option<Val> = None;
            for v in items.iter() {
                best = Some(match best {
                    None => v.clone(),
                    Some(b) => {
                        let o = match args.first() {
                            Some(f) => rt.call_fn(f, &[v.clone(), b.clone()])?.to_i64()?.cmp(&0),
                            None => v.compare(&b)?,
                        };
                        if (name == "max" && o.is_gt()) || (name == "min" && o.is_lt()) { v.clone() } else { b }
                    }
                });
            }
            opt(best)
        }
        ("anyMatch", 1) => {
            for v in items.iter() {
                if rt.call_fn(&args[0], std::slice::from_ref(v))?.as_bool()? {
                    return Ok(Some(Val::Bool(true)));
                }
            }
            Val::Bool(false)
        }
        ("allMatch", 1) | ("noneMatch", 1) => {
            let want = name == "allMatch";
            for v in items.iter() {
                if rt.call_fn(&args[0], std::slice::from_ref(v))?.as_bool()? != want {
                    return Ok(Some(Val::Bool(false)));
                }
            }
            Val::Bool(true)
        }
        ("findFirst" | "findAny", 0) => opt(items.first().cloned()),
        ("forEach", 1) => {
            for v in items.iter() {
                rt.call_fn(&args[0], std::slice::from_ref(v))?;
            }
            Val::Null
        }
        ("reduce", 2) => {
            let mut acc = args[0].clone();
            for v in items.iter() {
                acc = rt.call_fn(&args[1], &[acc, v.clone()])?;
            }
            acc
        }
        ("reduce", 1) => {
            let mut acc: Option<Val> = None;
            for v in items.iter() {
                acc = Some(match acc {
                    None => v.clone(),
                    Some(a) => rt.call_fn(&args[0], &[a, v.clone()])?,
                });
            }
            opt(acc)
        }
        ("toList" | "toArray", 0) => {
            let allocation = super::allocation::Allocation::values(items.len())?;
            Val::new_list(items.values.clone()).allocated(allocation)
        },
        ("collect", 1) => {
            let _copy = super::allocation::Allocation::values(items.len())?;
            collector(rt, items.values.clone(), &args[0])?
        },
        ("iterator", 0) => {
            let allocation = super::allocation::Allocation::values(items.len())?;
            Val::new_list(items.values.clone()).allocated(allocation)
        },
        _ => return Ok(None),
    }))
}

/// Methods shared by List / Set / doc values (read-only collection ops).
fn collection_method(rt: &mut Rt, recv: &Val, items: Vec<Val>, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("size", 0) | ("getLength", 0) => Val::Int(items.len() as i32),
        ("isEmpty", 0) => Val::Bool(items.is_empty()),
        ("contains", 1) => Val::Bool(items.iter().any(|v| v.equals(&args[0]))),
        ("containsAll", 1) => {
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            let other = iter_values(&args[0])?;
            Val::Bool(other.iter().all(|o| items.iter().any(|v| v.equals(o))))
        }
        ("stream", 0) => {
            let allocation = super::allocation::Allocation::values(items.capacity())?;
            Val::stream_from_values(items, allocation)
        },
        ("forEach", 1) => {
            for v in items {
                rt.call_fn(&args[0], &[v])?;
            }
            Val::Null
        }
        ("toString", 0) => Val::utf16(recv.string_units()),
        ("equals", 1) => Val::Bool(recv.equals(&args[0])),
        ("hashCode", 0) => Val::Int(items.iter().fold(1i32, |h, v| h.wrapping_mul(31).wrapping_add(v.java_hash()))),
        ("toArray", 0) | ("iterator", 0) => {
            let allocation = super::allocation::Allocation::values(items.capacity())?;
            Val::new_list(items).allocated(allocation)
        },
        // Painless collection augmentations.
        ("any", 1) => {
            for v in items {
                if rt.call_fn(&args[0], &[v])?.as_bool()? {
                    return Ok(Some(Val::Bool(true)));
                }
            }
            Val::Bool(false)
        }
        ("every", 1) => {
            for v in items {
                if !rt.call_fn(&args[0], &[v])?.as_bool()? {
                    return Ok(Some(Val::Bool(false)));
                }
            }
            Val::Bool(true)
        }
        ("find", 1) => {
            for v in items {
                if rt.call_fn(&args[0], std::slice::from_ref(&v))?.as_bool()? {
                    return Ok(Some(v));
                }
            }
            Val::Null
        }
        ("findAll", 1) => {
            let allocation = super::allocation::Allocation::values(items.len())?;
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                if rt.call_fn(&args[0], std::slice::from_ref(&v))?.as_bool()? {
                    out.push(v);
                }
            }
            Val::new_list(out).allocated(allocation)
        }
        ("collect", 1) => {
            let allocation = super::allocation::Allocation::values(items.len())?;
            let mut out = Vec::with_capacity(items.len());
            for value in items { out.push(rt.call_fn(&args[0], &[value])?); }
            Val::new_list(out).allocated(allocation)
        },
        ("sum", 0) => sum_vals(&items)?,
        ("sum", 1) => {
            let _allocation = super::allocation::Allocation::values(items.len())?;
            let mut mapped = Vec::with_capacity(items.len());
            for value in items { mapped.push(rt.call_fn(&args[0], &[value])?); }
            sum_vals(&mapped)?
        }
        ("join", 1) => {
            let separator = match &args[0] {
                Val::Str(_) | Val::Utf16(_) => args[0].string_units(),
                _ => str_arg(args, 0)?.encode_utf16().collect(),
            };
            let mut out = Vec::new();
            for (i, value) in items.iter().enumerate() {
                if i != 0 { out.extend_from_slice(&separator); }
                out.extend(value.string_units());
            }
            Val::utf16(out)
        },
        ("findResult", 1) => {
            for v in items {
                let r = rt.call_fn(&args[0], &[v])?;
                if !r.is_null() {
                    return Ok(Some(r));
                }
            }
            Val::Null
        }
        ("groupBy", 1) => {
            let allocation = super::allocation::Allocation::entries(items.len())?;
            let mut m: IndexMap<Key, Val> = IndexMap::with_capacity(items.len());
            for v in items {
                let k = Key::from_val(&rt.call_fn(&args[0], std::slice::from_ref(&v))?);
                match m.get(&k) {
                    Some(Val::List(l)) => { l.reserve_growth(1)?; l.borrow_mut().push(v); },
                    _ => {
                        let allocation = super::allocation::Allocation::values(1)?;
                        m.insert(k, Val::new_list(vec![v]).allocated(allocation));
                    }
                }
            }
            Val::new_map(m).allocated(allocation)
        }
        _ => return Ok(None),
    }))
}

/// `String` methods whose call sites the compiler binds by name and arity
/// once (see `Compiler::call`), instead of [`call_method`]'s by-name
/// dispatch per invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrFast {
    Length,
    IsEmpty,
    CharAt,
    Substring1,
    Substring2,
}

impl StrFast {
    pub fn of(name: &str, n: usize) -> Option<StrFast> {
        Some(match (name, n) {
            ("length", 0) => StrFast::Length,
            ("isEmpty", 0) => StrFast::IsEmpty,
            ("charAt", 1) => StrFast::CharAt,
            ("substring", 1) => StrFast::Substring1,
            ("substring", 2) => StrFast::Substring2,
            _ => return None,
        })
    }

    /// The call's value when the receiver is a `Val::Str`, the arguments are
    /// `int`s, allocation accounting is off and the call succeeds; then
    /// [`call_method`] computes exactly this value. Anything else (other
    /// receivers / argument types, non-ASCII text for the index-taking
    /// methods, out-of-range indices) is `None`: the caller runs
    /// [`call_method`] with the same, already evaluated, values, which
    /// produces the result or error the general path always did.
    #[inline]
    pub fn eval(self, recv: &Val, a0: Option<&Val>, a1: Option<&Val>) -> Option<Val> {
        let Val::Str(s) = recv else { return None };
        if super::allocation::Scope::active() {
            return None;
        }
        match self {
            StrFast::Length => Some(Val::Int(utf16_len(s) as i32)),
            StrFast::IsEmpty => Some(Val::Bool(s.is_empty())),
            StrFast::CharAt => {
                let Some(Val::Int(i)) = a0 else { return None };
                let i = *i;
                if i < 0 || !s.is_ascii() || i as usize >= s.len() {
                    return None;
                }
                Some(Val::Char(s.as_bytes()[i as usize] as u16))
            }
            StrFast::Substring1 => {
                let Some(Val::Int(a)) = a0 else { return None };
                ascii_substring(s, *a as i64, s.len() as i64)
            }
            StrFast::Substring2 => {
                let (Some(Val::Int(a)), Some(Val::Int(b))) = (a0, a1) else { return None };
                ascii_substring(s, *a as i64, *b as i64)
            }
        }
    }
}

/// `String.length()`: UTF-16 code units.
#[inline]
fn utf16_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.encode_utf16().count() }
}

/// `s.substring(a, b)` of an ASCII string (byte = UTF-16 unit offsets) when
/// in range.
#[inline]
fn ascii_substring(s: &str, a: i64, b: i64) -> Option<Val> {
    if a < 0 || a > b || b > s.len() as i64 || !s.is_ascii() {
        return None;
    }
    Some(Val::str(&s[a as usize..b as usize]))
}

/// `s.length()` of a local holding a `Val::Str` (numeric fast path of
/// `Compiler::num`); `None` = evaluate the general call.
#[inline]
pub fn string_length_fast(v: &Val) -> Option<i32> {
    match v {
        Val::Str(s) if !super::allocation::Scope::active() => Some(utf16_len(s) as i32),
        _ => None,
    }
}

pub fn call_method(rt: &mut Rt, recv: Val, name: &str, args: Vec<Val>) -> SResult<Val> {
    let _output=if name=="toString" && args.is_empty() {Some(super::allocation::Allocation::scalar_text(&[&recv])?)} else {None};
    let result=call_method_inner(rt,recv,name,args)?;
    if matches!(result,Val::Str(_)|Val::Utf16(_)) {super::allocation::Scope::adopt(&result)?;}
    Ok(result)
}

fn call_method_inner(rt: &mut Rt, recv: Val, name: &str, args: Vec<Val>) -> SResult<Val> {
    let n = args.len();
    match &recv {
        Val::Time(_) | Val::Date(_) | Val::Instant(_) | Val::DayOfWeek(_) | Val::Month(_) | Val::ChronoUnit(_) => {
            if let Some(v) = super::jtime::call_method(rt, &recv, name, &args)? {
                return Ok(v);
            }
            if matches!(recv, Val::Time(_)) {
                return Err(not_found(&recv, name, n));
            }
        }
        Val::Big(_) => return super::jbig::call_method(rt, &recv, name, &args)?.ok_or_else(|| not_found(&recv, name, n)),
        Val::Obj(_) => return super::jmisc::call_method(rt, &recv, name, &args)?.ok_or_else(|| not_found(&recv, name, n)),
        _ => {}
    }
    match &recv {
        Val::Time(_) | Val::Big(_) | Val::Obj(_) => Err(not_found(&recv, name, n)),
        Val::Null => Err(npe(format!("cannot access method/field [{name}] from a null def reference"))),
        Val::Str(s) if name == "format" && n == 1 && super::timefmt::is_formatter(s) && matches!(args[0], Val::Date(_) | Val::Instant(_)) => {
            let ms = match &args[0] { Val::Date(m) | Val::Instant(m) => *m, _ => 0 };
            Ok(Val::string(super::timefmt::format(s, ms).map_err(illegal_arg)?))
        }
        Val::Str(_) | Val::Utf16(_) if matches!(name, "replaceAll" | "replaceFirst") && matches!(args.first(), Some(Val::Pattern(_))) => {
            let Val::Pattern(p) = &args[0] else { unreachable!() };
            let _input_copy=super::allocation::Allocation::text_copy(&recv)?;
            if let Val::Matcher(m) = super::regex::new_matcher_units(p, &recv.string_units())? {
                if let Some(v) = super::regex::matcher_method(rt, &m, name, &args[1..])? {
                    return Ok(v);
                }
            }
            Err(not_found(&recv, name, n))
        }
        Val::Pattern(p) => super::regex::pattern_method(p, name, &args)?.ok_or_else(|| not_found(&recv, name, n)),
        Val::Matcher(m) => {
            let m = m.clone();
            super::regex::matcher_method(rt, &m, name, &args)?.ok_or_else(|| not_found(&recv, name, n))
        }
        // Painless does not whitelist `String.repeat` (java.lang.txt), so a
        // def receiver cannot reach it either (and cannot ask for a huge string).
        Val::Utf16(_) | Val::Str(_) if name == "repeat" => Err(not_found(&recv, name, n)),
        Val::Utf16(s) => super::source::string_method(&recv, s, name, &args),
        Val::Str(s) => string_method(&recv, s, name, &args).and_then(|r| r.ok_or_else(|| not_found(&recv, name, n))),
        Val::List(l) => {
            let l = l.clone();
            if l.immutable && matches!(name, "add" | "addAll" | "remove" | "removeAll" | "retainAll" | "removeIf" | "replaceAll" | "set" | "sort" | "clear") {
                return Err(ScriptErr::new("unsupported_operation_exception", "").frames(if name == "add" && n == 1 { UNMODIFIABLE_ADD_FRAMES } else { &[] }));
            }
            if let Some(v) = list_method(rt, &l, name, &args)? {
                return Ok(v);
            }
            let _copy_allocation = super::allocation::Allocation::values(l.len())?;
            let items = l.borrow().clone();
            collection_method(rt, &recv, items, name, &args)?.ok_or_else(|| not_found(&recv, name, n))
        }
        // Painless does not expose array Object.clone through dynamic dispatch.
        Val::Array { .. } => Err(not_found(&recv, name, n)),
        Val::Set(l) => {
            let l = l.clone();
            dyn_check(&recv, name, n)?;
            if let Some(v) = set_method(rt, &recv, &l, name, &args)? {
                return Ok(v);
            }
            let _copy_allocation = super::allocation::Allocation::values(l.len())?;
            let items = l.ordered().clone();
            collection_method(rt, &recv, items, name, &args)?.ok_or_else(|| not_found(&recv, name, n))
        }
        Val::Map(_) if is_geo_point(&recv) && matches!(name, "getLat" | "getLon" | "getGeohash" | "arcDistance" | "planeDistance" | "geohashDistance") => geo_point_method(&recv, name, &args),
        Val::Map(m) if n == 0 && matches!(name, "topLeft" | "bottomRight") && m.borrow().len() == 2 && m.borrow().contains_key(&Key::str("top_left")) => {
            // `GeoBoundingBox`.
            Ok(m.borrow().get(&Key::str(if name == "topLeft" { "top_left" } else { "bottom_right" })).cloned().unwrap_or(Val::Null))
        }
        Val::Map(m) => {
            let m = m.clone();
            dyn_check(&recv, name, n)?;
            map_method(rt, &recv, &m, name, &args)?.ok_or_else(|| not_found(&recv, name, n))
        }
        Val::DocValues(d) => {
            let r = match (name, n) {
                ("getValue", 0) => d.values.first().cloned().ok_or_else(|| illegal_state(super::doc::MISSING_VALUE))?,
                ("getValues", 0) => {
                    let allocation = super::allocation::Allocation::values(d.values.len())?;
                    Val::new_list(d.values.clone()).allocated(allocation)
                },
                ("get", 1) => {
                    let i = args[0].to_index()?;
                    d.values.get(i as usize).cloned().ok_or_else(|| out_of_bounds(format!("Index {i} out of bounds for length {}", d.values.len())))?
                }
                ("isEmpty", 0) | ("getEmpty", 0) => Val::Bool(d.values.is_empty()),
                ("getMagnitudes" | "getVectorValues", 0) if d.values.first().is_some_and(is_rank_vectors) => {
                    let first = d.values.first().cloned().unwrap_or(Val::Null);
                    let key = if name == "getMagnitudes" { "magnitudes" } else { "vectors" };
                    let list = super::runtime::field_get(rt, &first, key)?;
                    if name == "getMagnitudes" {
                        list
                    } else {
                        let items = match &list {
                            Val::List(l) => l.borrow().clone(),
                            _ => Vec::new(),
                        };
                        super::jmisc::new_iter(items, None)?
                    }
                }
                ("getMagnitudes" | "getVectorValues", 0) if d.values.is_empty() => return Err(illegal_state(super::doc::MISSING_VALUE)),
                ("getCentroid" | "getBoundingBox" | "getLabelPosition" | "getDimensionalType" | "getMercatorWidth" | "getMercatorHeight", 0)
                    if d.values.first().is_some_and(|v| is_geo_shape(v) || is_geo_point(v)) || d.values.is_empty() =>
                {
                    geo_values_method(&d.values, name)?
                }
                ("getLat", 0) | ("getLon", 0) => {
                    let first = d.values.first().cloned().ok_or_else(|| illegal_state(super::doc::MISSING_VALUE))?;
                    let key = if name == "getLat" { "lat" } else { "lon" };
                    super::runtime::field_get(rt, &first, key)?
                }
                (
                    "arcDistance" | "planeDistance" | "geohashDistance" | "arcDistanceWithDefault" | "planeDistanceWithDefault" | "geohashDistanceWithDefault" | "getGeohash",
                    _,
                ) if d.values.first().is_some_and(is_geo_point) || d.values.is_empty() => {
                    let with_default = name.ends_with("WithDefault");
                    match d.values.first() {
                        Some(p) => geo_point_method(p, name.trim_end_matches("WithDefault"), &args[..args.len() - with_default as usize])?,
                        None if with_default => args.last().cloned().unwrap_or(Val::Null),
                        None => return Err(illegal_state(super::doc::MISSING_VALUE)),
                    }
                }
                _ => {
                    let _copy_allocation = super::allocation::Allocation::values(d.values.len())?;
                    return collection_method(rt, &recv, d.values.clone(), name, &args)?.ok_or_else(|| not_found(&recv, name, n));
                }
            };
            Ok(r)
        }
        Val::Stream(items) => {
            let items = items.clone();
            stream_method(rt, &items, name, args)?.ok_or_else(|| not_found(&recv, name, n))
        }
        Val::Int(_) | Val::Long(_) | Val::Double(_) | Val::Char(_) => number_method(&recv, name, &args)?.ok_or_else(|| not_found(&recv, name, n)),
        Val::Bool(b) => match (name, n) {
            ("booleanValue", 0) => Ok(Val::Bool(*b)),
            ("toString", 0) => Ok(Val::string(b.to_string())),
            ("equals", 1) => Ok(Val::Bool(recv.equals(&args[0]))),
            ("compareTo", 1) => Ok(Val::Int(b.cmp(&args[0].as_bool()?) as i32)),
            ("hashCode", 0) => Ok(Val::Int(recv.java_hash())),
            _ => Err(not_found(&recv, name, n)),
        },
        Val::Date(ms) => date_method(&recv, *ms, true, name, &args)?.ok_or_else(|| not_found(&recv, name, n)),
        Val::Instant(ms) => date_method(&recv, *ms, false, name, &args)?.ok_or_else(|| not_found(&recv, name, n)),
        Val::DayOfWeek(d) | Val::Month(d) => {
            let is_day = matches!(recv, Val::DayOfWeek(_));
            match (name, n) {
                ("getValue" | "ordinal", 0) => Ok(Val::Int(*d as i32 - if name == "ordinal" { 1 } else { 0 })),
                ("name" | "toString", 0) => Ok(Val::str(if is_day { DAYS[*d as usize - 1] } else { MONTHS[*d as usize - 1] })),
                ("equals", 1) => Ok(Val::Bool(recv.equals(&args[0]))),
                ("compareTo", 1) => Ok(Val::Int(recv.compare(&args[0])? as i32)),
                ("plus", 1) | ("minus", 1) => {
                    let k = args[0].to_i64()? * if name == "minus" { -1 } else { 1 };
                    let m = if is_day { 7 } else { 12 };
                    let v = ((*d as i64 - 1 + k).rem_euclid(m) + 1) as u8;
                    Ok(if is_day { Val::DayOfWeek(v) } else { Val::Month(v) })
                }
                ("getDisplayName", _) => {
                    let full = if is_day { DAYS[*d as usize - 1] } else { MONTHS[*d as usize - 1] };
                    let mut s = full[..1].to_string();
                    s.push_str(&full[1..].to_lowercase());
                    Ok(Val::string(s))
                }
                _ => Err(not_found(&recv, name, n)),
            }
        }
        Val::ChronoUnit(u) => match (name, n) {
            ("between", 2) => Ok(Val::Long(chrono_between(*u, date_ms(&args[0])?, date_ms(&args[1])?))),
            ("toString" | "name", 0) => Ok(Val::utf16(recv.string_units())),
            _ => Err(not_found(&recv, name, n)),
        },
        Val::Optional(o) => {
            let o = o.clone();
            match (name, n) {
                ("isPresent", 0) => Ok(Val::Bool(o.is_some())),
                ("isEmpty", 0) => Ok(Val::Bool(o.is_none())),
                ("get" | "getAsDouble" | "getAsInt" | "getAsLong" | "orElseThrow", 0) => {
                    o.map(|v| (*v).clone()).ok_or_else(|| ScriptErr::new("no_such_element_exception", "No value present"))
                }
                ("orElse", 1) => Ok(o.map(|v| (*v).clone()).unwrap_or_else(|| args[0].clone())),
                ("orElseGet", 1) => match o {
                    Some(v) => Ok((*v).clone()),
                    None => rt.call_fn(&args[0], &[]),
                },
                ("ifPresent", 1) => {
                    if let Some(v) = o {
                        rt.call_fn(&args[0], &[(*v).clone()])?;
                    }
                    Ok(Val::Null)
                }
                ("map", 1) => match o {
                    Some(v) => {
                        let r = rt.call_fn(&args[0], &[(*v).clone()])?;
                        Ok(opt(if r.is_null() { None } else { Some(r) }))
                    }
                    None => Ok(opt(None)),
                },
                ("filter", 1) => match o {
                    Some(v) => {
                        let keep = rt.call_fn(&args[0], &[(*v).clone()])?.as_bool()?;
                        Ok(opt(if keep { Some((*v).clone()) } else { None }))
                    }
                    None => Ok(opt(None)),
                },
                ("toString", 0) => Ok(Val::utf16(recv.string_units())),
                _ => Err(not_found(&recv, name, n)),
            }
        }
        Val::Entry(e) => match (name, n) {
            ("getKey", 0) => Ok(e.0.clone()),
            ("getValue", 0) => Ok(e.1.clone()),
            ("toString", 0) => Ok(Val::utf16(recv.string_units())),
            ("equals", 1) => Ok(Val::Bool(recv.equals(&args[0]))),
            _ => Err(not_found(&recv, name, n)),
        },
        Val::Builder(b) => {
            let b = b.clone();
            let _argument_copies: Vec<_> = args.iter().map(super::allocation::Allocation::text_copy).collect::<SResult<_>>()?;
            let result = match (name, n) {
                ("capacity",0)=>Ok(Val::Int(b.borrow().capacity as i32)),
                ("getChars",4)=>super::source::string_method(&recv,&b.borrow(),name,&args),
                ("append",3)=>{
                    let text=match &args[0] {Val::Null=>"null".encode_utf16().collect(),Val::Str(_)|Val::Utf16(_)|Val::Builder(_)=>args[0].string_units(),_=>return Err(class_cast("Expected CharSequence"))};
                    let start=args[1].to_i64()?;let end=args[2].to_i64()?;
                    if start<0 || end<start || end as usize>text.len() {return Err(ScriptErr::new("index_out_of_bounds_exception","Invalid CharSequence range"));}
                    let mut builder=b.borrow_mut();
                    let length=builder.len().saturating_add(end as usize-start as usize);
                    builder.reserve_length(length)?;
                    builder.extend_from_slice(&text[start as usize..end as usize]);
                    Ok(recv.clone())
                }
                ("append", 1) => {
                    let units=args[0].string_units();
                    let mut builder=b.borrow_mut();
                    let length=builder.len().saturating_add(units.len());
                    builder.reserve_length(length)?;
                    builder.extend(units);
                    Ok(recv.clone())
                }
                ("toString", 0) => {
                    Ok(Val::utf16(b.borrow().to_vec()))
                },
                ("length", 0) => Ok(Val::Int(b.borrow().len() as i32)),
                ("reverse", 0) => {
                    let mut units=b.borrow_mut();
                    units.reverse();
                    // Java reverses code units, then restores valid pairs.
                    let mut i=0;
                    while i+1<units.len() {
                        if (0xdc00..=0xdfff).contains(&units[i]) && (0xd800..=0xdbff).contains(&units[i+1]) {
                            units.swap(i,i+1); i+=2;
                        } else {i+=1;}
                    }
                    Ok(recv.clone())
                }
                ("setLength", 1) => {
                    let k = args[0].to_index()?;
                    if k < 0 { return Err(ScriptErr::new("string_index_out_of_bounds_exception", format!("length {k}"))); }
                    let mut builder=b.borrow_mut();
                    builder.reserve_length(k as usize)?;
                    builder.resize(k as usize, 0);
                    Ok(Val::Null)
                }
                ("chars" | "codePoints", 0)
                | ("charAt" | "codePointAt" | "codePointBefore", 1)
                | ("codePointCount" | "offsetByCodePoints" | "subSequence", 2)
                | ("substring" | "indexOf" | "lastIndexOf", 1 | 2) => {
                    super::source::string_method(&recv, &b.borrow(), name, &args)
                }
                ("appendCodePoint", 1) => {
                    let cp = args[0].to_index()?;
                    if !(0..=0x10ffff).contains(&cp) { return Err(illegal_arg(format!("Not a valid Unicode code point: {cp}"))); }
                    let mut units = b.borrow_mut();
                    let length=units.len().saturating_add(if cp<=0xffff {1}else{2});
                    units.reserve_length(length)?;
                    if cp <= 0xffff { units.push(cp as u16); }
                    else { let cp = cp as u32 - 0x10000; units.extend([0xd800 | (cp >> 10) as u16, 0xdc00 | (cp & 0x3ff) as u16]); }
                    Ok(recv.clone())
                }
                ("setCharAt", 2) | ("deleteCharAt", 1) => {
                    let i = args[0].to_index()?;
                    let mut units = b.borrow_mut();
                    if i < 0 || i as usize >= units.len() { return Err(ScriptErr::new("string_index_out_of_bounds_exception", format!("index {i}"))); }
                    if name == "deleteCharAt" { units.remove(i as usize); Ok(recv.clone()) }
                    else {
                        let Val::Char(c) = &args[1] else { return Err(class_cast("Cannot cast argument to char")); };
                        units[i as usize] = *c;
                        Ok(Val::Null)
                    }
                }
                ("delete", 2) | ("replace", 3) => {
                    let start = args[0].to_index()?;
                    let end = args[1].to_index()?;
                    let replacement = if name == "replace" {
                        if matches!(args[2], Val::Null) { return Err(npe("replacement is null")); }
                        args[2].string_units()
                    } else { Vec::new() };
                    let mut units = b.borrow_mut();
                    if start < 0 || start as usize > units.len() || end < start {
                        return Err(ScriptErr::new("string_index_out_of_bounds_exception", format!("start {start}, end {end}")));
                    }
                    let end = (end as usize).min(units.len());
                    let length=units.len().saturating_sub(end-start as usize).saturating_add(replacement.len());
                    units.reserve_length(length)?;
                    units.splice(start as usize..end, replacement);
                    Ok(recv.clone())
                }
                ("insert", 2) => {
                    let k = args[0].to_index()? as usize;
                    let ins=args[1].string_units();
                    let mut units=b.borrow_mut();
                    if k>units.len() {return Err(ScriptErr::new("string_index_out_of_bounds_exception",format!("offset {k}")));}
                    let length=units.len().saturating_add(ins.len());
                    units.reserve_length(length)?;
                    units.splice(k..k,ins);
                    Ok(recv.clone())
                }
                _ => Err(not_found(&recv, name, n)),
            };
            b.borrow_mut().update_capacity();
            result
        }
        Val::Exception(e) => match (name, n) {
            ("getMessage" | "getLocalizedMessage", 0) => Ok(if e.1.is_empty() { Val::Null } else { Val::string(e.1.clone()) }),
            ("toString", 0) => Ok(Val::utf16(recv.string_units())),
            _ => Err(not_found(&recv, name, n)),
        },
        Val::Lambda(_) | Val::MethodRef(_) => match name {
            "apply" | "test" | "accept" | "get" | "applyAsInt" | "applyAsLong" | "applyAsDouble" | "compare" | "run" => rt.call_fn(&recv, &args),
            _ => Err(not_found(&recv, name, n)),
        },
        Val::Doc => match (name, n) {
            ("containsKey", 1) => {
                let k = str_arg(&args, 0)?;
                Ok(Val::Bool(rt.doc.contains(&k)))
            }
            ("get", 1) => {
                let k = str_arg(&args, 0)?;
                let slot = rt.doc.slot_of(&k);
                rt.doc.values(slot)
            }
            _ => Err(not_found(&recv, name, n)),
        },
    }
}

fn number_method(recv: &Val, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("charValue",0) if matches!(recv,Val::Char(_)) => recv.clone(),
        ("compareTo",1) if matches!(recv,Val::Char(_)) && matches!(args[0],Val::Char(_)) => Val::Int(recv.to_i64()? as i32-args[0].to_i64()? as i32),
        ("intValue", 0) => super::runtime::cast(&super::ast::Ty::Int, recv.clone())?,
        ("longValue", 0) => super::runtime::cast(&super::ast::Ty::Long, recv.clone())?,
        ("doubleValue", 0) => Val::Double(recv.to_f64()?),
        ("floatValue", 0) => Val::Double(recv.to_f64()? as f32 as f64),
        ("shortValue", 0) => super::runtime::cast(&super::ast::Ty::Short, recv.clone())?,
        ("byteValue", 0) => super::runtime::cast(&super::ast::Ty::Byte, recv.clone())?,
        ("toString", 0) => Val::utf16(recv.string_units()),
        ("equals", 1) => Val::Bool(std::mem::discriminant(recv) == std::mem::discriminant(&args[0]) && recv.equals(&args[0])),
        ("compareTo", 1) => Val::Int(recv.compare(&args[0])? as i32),
        ("hashCode", 0) => Val::Int(recv.java_hash()),
        ("isNaN", 0) => Val::Bool(recv.to_f64()?.is_nan()),
        ("isInfinite", 0) => Val::Bool(recv.to_f64()?.is_infinite()),
        _ => return Ok(None),
    }))
}

fn string_method(recv: &Val, s: &Rc<str>, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let _receiver_copy=super::allocation::Allocation::iterable(recv)?;
    let n = args.len();
    if matches!((name,n),("chars"|"codePoints"|"toCharArray"|"decodeBase64",0)) {
        return super::source::string_method(recv,&s.encode_utf16().collect::<Vec<_>>(),name,args).map(Some);
    }
    if matches!((name,n),("toLowerCase"|"toUpperCase",0|1)) {
        return super::source::string_method(recv,&s.encode_utf16().collect::<Vec<_>>(),name,args).map(Some);
    }
    if matches!((name,n), ("equalsIgnoreCase"|"compareToIgnoreCase",1)|("regionMatches",4|5)) {
        return super::source::string_method(recv, &s.encode_utf16().collect::<Vec<_>>(), name, args).map(Some);
    }
    if matches!((name,n), ("contentEquals",1)|("getChars",4)|("splitOnToken",1|2)) {
        return super::source::string_method(recv, &s.encode_utf16().collect::<Vec<_>>(), name, args).map(Some);
    }
    if matches!((name,n), ("indexOf" | "lastIndexOf", 1 | 2) | ("startsWith", 2)) {
        return super::source::string_method(recv, &s.encode_utf16().collect::<Vec<_>>(), name, args).map(Some);
    }
    if matches!((name,n), ("codePointAt"|"codePointBefore"|"compareTo"|"concat"|"contains"|"startsWith"|"endsWith",1)|("codePointCount"|"offsetByCodePoints"|"replace",2)|("trim"|"strip"|"stripLeading"|"stripTrailing"|"isBlank",0)) {
        return super::source::string_method(recv, &s.encode_utf16().collect::<Vec<_>>(), name, args).map(Some);
    }
    // Java substring offsets address UTF-16 code units, including positions
    // inside a surrogate pair. Keep the existing BMP/error path unchanged.
    if (s.chars().any(|c| c.len_utf16() == 2) || _receiver_copy.enabled())
        && matches!((name, n), ("substring", 1 | 2) | ("subSequence", 2))
    {
        return super::source::string_method(recv, &s.encode_utf16().collect::<Vec<_>>(), name, args).map(Some);
    }
    let len = || s.chars().count() as i64;
    Ok(Some(match (name, n) {
        ("length", 0) => Val::Int(s.encode_utf16().count() as i32),
        ("isEmpty", 0) => Val::Bool(s.is_empty()),
        ("isBlank", 0) => Val::Bool(s.trim().is_empty()),
        ("charAt", 1) => {
            let i = int_arg(args, 0)?;
            s.encode_utf16().nth(i.max(0) as usize).filter(|_| i >= 0).map(Val::Char).ok_or_else(|| ScriptErr::new("string_index_out_of_bounds_exception", format!("Index {i} out of bounds for length {}", len())).frames(STRING_CHARAT_FRAMES))?
        }
        ("codePointAt", 1) => {
            let i = int_arg(args, 0)?;
            Val::Int(s.encode_utf16().nth(i.max(0) as usize).ok_or_else(|| ScriptErr::new("string_index_out_of_bounds_exception", format!("index {i}")))? as i32)
        }
        ("substring", 1) => Val::string(char_slice(s, int_arg(args, 0)?, len())?),
        ("substring", 2) => Val::string(char_slice(s, int_arg(args, 0)?, int_arg(args, 1)?)?),
        ("indexOf", 1 | 2) => {
            let from = if n == 2 { int_arg(args, 1)?.max(0) as usize } else { 0 };
            Val::Int(find_chars(s, &str_arg(args, 0)?, from, false) as i32)
        }
        ("lastIndexOf", 1 | 2) => {
            let from = if n == 2 { int_arg(args, 1)?.max(0) as usize } else { usize::MAX };
            Val::Int(find_chars(s, &str_arg(args, 0)?, from, true) as i32)
        }
        ("contains", 1) => Val::Bool(s.contains(str_arg(args, 0)?.as_str())),
        ("startsWith", 1) => Val::Bool(s.starts_with(str_arg(args, 0)?.as_str())),
        ("startsWith", 2) => {
            let off = int_arg(args, 1)?.max(0) as usize;
            let rest: String = s.chars().skip(off).collect();
            Val::Bool(rest.starts_with(str_arg(args, 0)?.as_str()))
        }
        ("endsWith", 1) => Val::Bool(s.ends_with(str_arg(args, 0)?.as_str())),
        ("toLowerCase", 0 | 1) => Val::string(s.to_lowercase()),
        ("toUpperCase", 0 | 1) => Val::string(s.to_uppercase()),
        ("trim", 0) => Val::str(s.trim_matches(|c: char| c <= ' ')),
        ("strip", 0) => Val::str(s.trim()),
        ("stripLeading", 0) => Val::str(s.trim_start()),
        ("stripTrailing", 0) => Val::str(s.trim_end()),
        ("replace", 2) => Val::string(s.replace(str_arg(args, 0)?.as_str(), str_arg(args, 1)?.as_str())),
        ("replaceAll", 2) | ("replaceFirst", 2) => {
            let re = regex::Regex::new(&str_arg(args, 0)?).map_err(|e| ScriptErr::new("pattern_syntax_exception", e.to_string()))?;
            let rep = str_arg(args, 1)?.replace('$', "$$").replace("$$1", "${1}");
            Val::string(if name == "replaceAll" { re.replace_all(s, rep.as_str()).into_owned() } else { re.replace(s, rep.as_str()).into_owned() })
        }
        ("matches", 1) => {
            let re = regex::Regex::new(&format!("^(?:{})$", str_arg(args, 0)?)).map_err(|e| ScriptErr::new("pattern_syntax_exception", e.to_string()))?;
            Val::Bool(re.is_match(s))
        }
        ("split", 1 | 2) => {
            let allocation = super::allocation::Allocation::values(s.len().saturating_add(1).saturating_mul(4))?;
            let pat = str_arg(args, 0)?;
            let limit = if n == 2 { int_arg(args, 1)? } else { 0 };
            let mut parts: Vec<String> = if pat.len() == 1 && !".$|()[{^?*+\\".contains(pat.as_str()) {
                if limit > 0 { s.splitn(limit as usize, pat.as_str()).map(str::to_string).collect() } else { s.split(pat.as_str()).map(str::to_string).collect() }
            } else {
                let re = regex::Regex::new(&pat).map_err(|e| ScriptErr::new("pattern_syntax_exception", e.to_string()))?;
                if limit > 0 { re.splitn(s, limit as usize).map(str::to_string).collect() } else { re.split(s).map(str::to_string).collect() }
            };
            if limit == 0 {
                while parts.last().is_some_and(|p| p.is_empty()) && parts.len() > 0 {
                    parts.pop();
                }
            }
            if parts.is_empty() && !s.is_empty() {
                parts.push(String::new());
            }
            Val::new_list(parts.into_iter().map(Val::string).collect()).allocated(allocation)
        }
        ("splitOnToken", 1 | 2) => {
            let allocation = super::allocation::Allocation::values(s.len().saturating_add(1).saturating_mul(4))?;
            let tok = str_arg(args, 0)?;
            if tok.is_empty() {
                return Err(illegal_arg("Cannot splitOnToken with an empty token"));
            }
            let parts: Vec<Val> = if n == 2 {
                let lim = int_arg(args, 1)?;
                if lim > 0 { s.splitn(lim as usize, tok.as_str()).map(Val::str).collect() } else { s.split(tok.as_str()).map(Val::str).collect() }
            } else {
                s.split(tok.as_str()).map(Val::str).collect()
            };
            Val::new_list(parts).allocated(allocation)
        }
        ("equals", 1) => Val::Bool(matches!(&args[0], Val::Str(o) if o == s)),
        ("equalsIgnoreCase", 1) => Val::Bool(matches!(&args[0], Val::Str(o) if o.to_lowercase() == s.to_lowercase())),
        ("compareTo", 1) => {
            let o = str_arg(args, 0)?;
            Val::Int(java_str_cmp(s, &o))
        }
        ("compareToIgnoreCase", 1) => {
            let o = str_arg(args, 0)?;
            Val::Int(java_str_cmp(&s.to_lowercase(), &o.to_lowercase()))
        }
        ("concat", 1) => Val::string(format!("{s}{}", str_arg(args, 0)?)),
        ("hashCode", 0) => Val::Int(recv.java_hash()),
        ("toString" | "intern", 0) => recv.clone(),
        ("chars", 0) => Val::stream_from_iter(s.chars().map(|c| Ok(Val::Int(c as i32))))?,
        ("toCharArray", 0) => {
            let allocation = super::allocation::Allocation::values(s.len())?;
            let mut values = Vec::with_capacity(s.len());
            values.extend(s.encode_utf16().map(Val::Char));
            Val::new_list(values).allocated(allocation)
        },
        ("encodeBase64", 0) => {
            use base64::Engine;
            Val::string(base64::engine::general_purpose::STANDARD.encode(s.as_bytes()))
        }
        ("sha1", 0) => Val::string(hex(&sha1(s.as_bytes()))),
        ("sha256", 0) => {
            use sha2::Digest;
            Val::string(hex(&sha2::Sha256::digest(s.as_bytes())))
        }
        ("sha512", 0) => {
            use sha2::Digest;
            Val::string(hex(&sha2::Sha512::digest(s.as_bytes())))
        }
        ("decodeBase64", 0) => {
            use base64::Engine;
            let b = base64::engine::general_purpose::STANDARD.decode(s.as_bytes()).map_err(|e| illegal_arg(e.to_string()))?;
            Val::string(String::from_utf8_lossy(&b).into_owned())
        }
        _ => return Ok(None),
    }))
}

fn java_str_cmp(a: &str, b: &str) -> i32 {
    let (x, y): (Vec<u16>, Vec<u16>) = (a.encode_utf16().collect(), b.encode_utf16().collect());
    for (p, q) in x.iter().zip(&y) {
        if p != q {
            return *p as i32 - *q as i32;
        }
    }
    x.len() as i32 - y.len() as i32
}

fn list_method(rt: &mut Rt, l: &super::value::List, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    let len = l.borrow().len();
    let idx = |i: i64, len: usize, incl: bool| -> SResult<usize> {
        let ok = if incl { i >= 0 && i as usize <= len } else { i >= 0 && (i as usize) < len };
        if !ok {
            return Err(out_of_bounds(format!("Index {i} out of bounds for length {len}")));
        }
        Ok(i as usize)
    };
    let first_or = |miss: Option<Val>| -> SResult<Val> {
        match l.borrow().first() {
            Some(v) => Ok(v.clone()),
            None => miss.ok_or_else(|| ScriptErr::new("no_such_element_exception", "")),
        }
    };
    let last_or = |miss: Option<Val>| -> SResult<Val> {
        match l.borrow().last() {
            Some(v) => Ok(v.clone()),
            None => miss.ok_or_else(|| ScriptErr::new("no_such_element_exception", "")),
        }
    };
    Ok(Some(match (name, n) {
        ("get", 1) => {
            let i = int_arg(args, 0)?;
            if i < 0 || i as usize >= len {
                return Err(out_of_bounds(format!("Index {i} out of bounds for length {len}")).frames(ARRAYLIST_GET_FRAMES));
            }
            l.borrow()[i as usize].clone()
        }
        ("add", 1) => {
            l.reserve_growth(1)?;
            l.borrow_mut().push(args[0].clone());
            Val::Bool(true)
        }
        // Deque / Queue API (`ArrayDeque`, `LinkedList` share the list representation).
        ("push" | "addFirst", 1) => {
            l.reserve_growth(1)?;
            l.borrow_mut().insert(0, args[0].clone());
            Val::Null
        }
        ("offerFirst", 1) => {
            l.reserve_growth(1)?;
            l.borrow_mut().insert(0, args[0].clone());
            Val::Bool(true)
        }
        ("addLast", 1) => {
            l.reserve_growth(1)?;
            l.borrow_mut().push(args[0].clone());
            Val::Null
        }
        ("offer" | "offerLast", 1) => {
            l.reserve_growth(1)?;
            l.borrow_mut().push(args[0].clone());
            Val::Bool(true)
        }
        ("pop" | "removeFirst" | "remove" | "element" | "getFirst", 0) if len == 0 => return Err(ScriptErr::new("no_such_element_exception", "")),
        ("pop" | "removeFirst" | "remove", 0) => l.borrow_mut().remove(0),
        ("poll" | "pollFirst", 0) => {
            if len == 0 {
                Val::Null
            } else {
                l.borrow_mut().remove(0)
            }
        }
        ("removeLast", 0) => l.borrow_mut().pop().ok_or_else(|| ScriptErr::new("no_such_element_exception", ""))?,
        ("pollLast", 0) => l.borrow_mut().pop().unwrap_or(Val::Null),
        ("peek" | "peekFirst", 0) => first_or(Some(Val::Null))?,
        ("element" | "getFirst", 0) => first_or(None)?,
        ("peekLast", 0) => last_or(Some(Val::Null))?,
        ("getLast", 0) => last_or(None)?,
        ("descendingIterator", 0) => super::jmisc::new_iter(l.borrow().iter().rev().cloned(), None)?,
        ("iterator" | "listIterator", 0) => super::jmisc::new_iter(l.borrow().iter().cloned(), Some(l.clone()))?,
        ("add", 2) => {
            let i = idx(int_arg(args, 0)?, len, true)?;
            l.reserve_growth(1)?;
            l.borrow_mut().insert(i, args[1].clone());
            Val::Null
        }
        ("addAll", 1) => {
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            let vals = iter_values(&args[0])?;
            let changed = !vals.is_empty();
            l.reserve_growth(vals.len())?;
            l.borrow_mut().extend(vals);
            Val::Bool(changed)
        }
        ("set", 2) => {
            let i = idx(int_arg(args, 0)?, len, false)?;
            l.register_mutation()?;
            std::mem::replace(&mut l.borrow_mut()[i], args[1].clone())
        }
        ("remove", 1) => match &args[0] {
            Val::Int(i) => {
                let i = idx(*i as i64, len, false)?;
                l.borrow_mut().remove(i)
            }
            v => {
                let pos = l.borrow().iter().position(|x| x.equals(v));
                match pos {
                    Some(p) => {
                        l.borrow_mut().remove(p);
                        Val::Bool(true)
                    }
                    None => Val::Bool(false),
                }
            }
        },
        ("removeAll", 1) => {
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            let other = iter_values(&args[0])?;
            let mut b = l.borrow_mut();
            let before = b.len();
            b.retain(|x| !other.iter().any(|o| o.equals(x)));
            Val::Bool(b.len() != before)
        }
        ("retainAll", 1) => {
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            let other = iter_values(&args[0])?;
            let mut b = l.borrow_mut();
            let before = b.len();
            b.retain(|x| other.iter().any(|o| o.equals(x)));
            Val::Bool(b.len() != before)
        }
        ("indexOf", 1) => Val::Int(l.borrow().iter().position(|x| x.equals(&args[0])).map(|p| p as i32).unwrap_or(-1)),
        ("lastIndexOf", 1) => Val::Int(l.borrow().iter().rposition(|x| x.equals(&args[0])).map(|p| p as i32).unwrap_or(-1)),
        ("clear", 0) => {
            l.borrow_mut().clear();
            Val::Null
        }
        ("sort", 0 | 1) => {
            let _copy = super::allocation::Allocation::values(l.len().saturating_mul(2))?;
            let mut items = l.borrow().clone();
            sort_vals(rt, &mut items, args.first())?;
            *l.borrow_mut() = items;
            Val::Null
        }
        ("removeIf", 1) => {
            let _copy = super::allocation::Allocation::values(l.len().saturating_mul(2))?;
            let items = l.borrow().clone();
            let mut keep = Vec::with_capacity(items.len());
            for v in items {
                if !rt.call_fn(&args[0], std::slice::from_ref(&v))?.as_bool()? {
                    keep.push(v);
                }
            }
            let changed = keep.len() != l.borrow().len();
            *l.borrow_mut() = keep;
            Val::Bool(changed)
        }
        ("replaceAll", 1) => {
            l.register_mutation()?;
            let _copies = super::allocation::Allocation::values(l.len().saturating_mul(2))?;
            let items = l.borrow().clone();
            let mut out = Vec::with_capacity(items.len());
            for value in items {
                let value = rt.call_fn(&args[0], &[value])?;
                out.push(value);
            }
            *l.borrow_mut() = out;
            Val::Null
        }
        ("subList", 2) => {
            let a = idx(int_arg(args, 0)?, len, true)?;
            let b = idx(int_arg(args, 1)?, len, true)?;
            if a > b {
                return Err(illegal_arg(format!("fromIndex({a}) > toIndex({b})")));
            }
            let allocation = super::allocation::Allocation::values(b - a)?;
            Val::new_list(l.borrow()[a..b].to_vec()).allocated(allocation)
        }
        ("getFirst", 0) => l.borrow().first().cloned().ok_or_else(|| ScriptErr::new("no_such_element_exception", "list is empty"))?,
        ("getLast", 0) => l.borrow().last().cloned().ok_or_else(|| ScriptErr::new("no_such_element_exception", "list is empty"))?,
        _ => return Ok(None),
    }))
}

fn map_method(rt: &mut Rt, recv: &Val, m: &super::value::MapRef, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    if m.immutable && matches!(name, "put" | "putAll" | "remove" | "clear" | "putIfAbsent" | "compute" | "computeIfAbsent" | "computeIfPresent" | "merge" | "replace" | "replaceAll") {
        return Err(ScriptErr::new("unsupported_operation_exception", ""));
    }
    if m.kind == MapKind::Tree {
        if let Some(v) = tree_map_method(rt, m, name, args)? {
            return Ok(Some(v));
        }
    }
    let key = |i: usize| -> SResult<Key> { Ok(Key::from_val(arg(args, i)?)) };
    Ok(Some(match (name, n) {
        ("get", 1) => m.borrow().get(&key(0)?).cloned().unwrap_or(Val::Null),
        ("getOrDefault", 2) => m.borrow().get(&key(0)?).cloned().unwrap_or_else(|| args[1].clone()),
        ("put", 2) => m.insert_checked(key(0)?, args[1].clone())?.unwrap_or(Val::Null),
        ("putIfAbsent", 2) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned();
            match cur {
                Some(v) if !v.is_null() => v,
                _ => {
                    m.insert_checked(k, args[1].clone())?;
                    Val::Null
                }
            }
        }
        ("putAll", 1) => {
            if let Val::Map(o) = &args[0] {
                let _copy = super::allocation::Allocation::entries(o.borrow().len())?;
                let items: Vec<(Key, Val)> = o.borrow().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                for (k, v) in items {
                    m.insert_checked(k, v)?;
                }
            }
            Val::Null
        }
        ("containsKey", 1) => Val::Bool(m.borrow().contains_key(&key(0)?)),
        ("containsValue", 1) => Val::Bool(m.borrow().values().any(|v| v.equals(&args[0]))),
        ("remove", 1) => m.borrow_mut().shift_remove(&key(0)?).unwrap_or(Val::Null),
        ("keySet" | "navigableKeySet", 0) => {
            let allocation = super::allocation::Allocation::values(m.borrow().len())?;
            new_set(m.ordered().keys().map(Key::to_val).collect()).allocated(allocation)
        },
        ("values", 0) => {
            let allocation = super::allocation::Allocation::values(m.borrow().len())?;
            Val::new_list(m.ordered().values().cloned().collect()).allocated(allocation)
        },
        ("entrySet", 0) => {
            let allocation = super::allocation::Allocation::entries(m.borrow().len())?;
            new_set(m.ordered().iter().map(|(k, v)| Val::Entry(Rc::new((k.to_val(), v.clone())))).collect()).allocated(allocation)
        },
        ("size", 0) => Val::Int(m.borrow().len() as i32),
        ("isEmpty", 0) => Val::Bool(m.borrow().is_empty()),
        ("clear", 0) => {
            m.borrow_mut().clear();
            Val::Null
        }
        ("forEach", 1) => {
            let items: Vec<(Val, Val)> = m.ordered().iter().map(|(k, v)| (k.to_val(), v.clone())).collect();
            for (k, v) in items {
                rt.call_fn(&args[0], &[k, v])?;
            }
            Val::Null
        }
        ("computeIfAbsent", 2) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned();
            match cur {
                Some(v) if !v.is_null() => v,
                _ => {
                    let v = rt.call_fn(&args[1], &[args[0].clone()])?;
                    if !v.is_null() {
                        m.insert_checked(k, v.clone())?;
                    }
                    v
                }
            }
        }
        ("computeIfPresent", 2) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned();
            match cur {
                Some(v) if !v.is_null() => {
                    let nv = rt.call_fn(&args[1], &[args[0].clone(), v])?;
                    if nv.is_null() {
                        m.borrow_mut().shift_remove(&k);
                    } else {
                        m.insert_checked(k, nv.clone())?;
                    }
                    nv
                }
                _ => Val::Null,
            }
        }
        ("compute", 2) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned().unwrap_or(Val::Null);
            let nv = rt.call_fn(&args[1], &[args[0].clone(), cur])?;
            if nv.is_null() {
                m.borrow_mut().shift_remove(&k);
            } else {
                m.insert_checked(k, nv.clone())?;
            }
            nv
        }
        ("merge", 3) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned();
            let nv = match cur {
                Some(v) if !v.is_null() => rt.call_fn(&args[2], &[v, args[1].clone()])?,
                _ => args[1].clone(),
            };
            if nv.is_null() {
                m.borrow_mut().shift_remove(&k);
            } else {
                m.insert_checked(k, nv.clone())?;
            }
            nv
        }
        ("replace", 2) => {
            let k = key(0)?;
            let has = m.borrow().contains_key(&k);
            if has { m.insert_checked(k, args[1].clone())?.unwrap_or(Val::Null) } else { Val::Null }
        }
        ("equals", 1) => Val::Bool(recv.equals(&args[0])),
        ("toString", 0) => Val::utf16(recv.string_units()),
        ("hashCode", 0) => Val::Int(recv.java_hash()),
        ("getByPath", 1 | 2) => {
            let path = str_arg(args, 0)?;
            let mut cur = recv.clone();
            for p in path.split('.') {
                cur = match &cur {
                    Val::Map(mm) => mm.borrow().get(&Key::str(p)).cloned().unwrap_or(Val::Null),
                    Val::List(ll) => p.parse::<usize>().ok().and_then(|i| ll.borrow().get(i).cloned()).unwrap_or(Val::Null),
                    _ => Val::Null,
                };
                if cur.is_null() {
                    break;
                }
            }
            if cur.is_null() && n == 2 { args[1].clone() } else { cur }
        }
        ("collect", 1) => {
            let _copy = super::allocation::Allocation::values(m.borrow().len().saturating_mul(2))?;
            let allocation = super::allocation::Allocation::values(m.borrow().len())?;
            let items: Vec<(Val, Val)> = m.ordered().iter().map(|(k, v)| (k.to_val(), v.clone())).collect();
            let mut out = Vec::with_capacity(items.len());
            for (k,v) in items { out.push(rt.call_fn(&args[0], &[k,v])?); }
            Val::new_list(out).allocated(allocation)
        }
        ("count", 1) => {
            let _copy = super::allocation::Allocation::values(m.borrow().len().saturating_mul(2))?;
            let items: Vec<(Val, Val)> = m.ordered().iter().map(|(k, v)| (k.to_val(), v.clone())).collect();
            let mut c = 0;
            for (k, v) in items {
                if rt.call_fn(&args[0], &[k, v])?.as_bool()? {
                    c += 1;
                }
            }
            Val::Int(c)
        }
        _ => {
            // Static-looking calls on map values fall back to `call_static` (none).
            let _ = call_static;
            let _ = java_double_string;
            return Ok(None);
        }
    }))
}

// ---------------------------------------------------------------------------
// Java collection classes: HashMap/LinkedHashMap/TreeMap, HashSet/
// LinkedHashSet/TreeSet construction and the NavigableMap/NavigableSet API.

fn is_comparator(v: &Val) -> bool {
    matches!(v, Val::Lambda(_) | Val::MethodRef(_)) || matches!(v, Val::Obj(o) if matches!(**o, super::jmisc::Obj::Cmp(_, _)))
}

/// `compare(a, b)` under a tree's comparator (natural order when `None`).
pub fn tree_cmp(rt: &mut Rt, cmp: &Option<Val>, a: &Val, b: &Val) -> SResult<std::cmp::Ordering> {
    match cmp {
        Some(f) => Ok(rt.call_fn(f, &[a.clone(), b.clone()])?.to_i64()?.cmp(&0)),
        None => {
            if a.is_null() || b.is_null() {
                return Err(npe("Cannot invoke \"java.lang.Comparable.compareTo(Object)\" because \"k1\" is null"));
            }
            a.compare(b)
        }
    }
}

/// Binary search of `k` among sorted `keys`: `Ok(i)` found, `Err(i)` insertion point.
fn tree_search(rt: &mut Rt, cmp: &Option<Val>, keys: &[Val], k: &Val) -> SResult<Result<usize, usize>> {
    let (mut lo, mut hi) = (0usize, keys.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        match tree_cmp(rt, cmp, &keys[mid], k)? {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Ok(Ok(mid)),
        }
    }
    Ok(Err(lo))
}

fn tree_keys(m: &JMap) -> Vec<Val> {
    m.ordered().keys().map(Key::to_val).collect()
}

/// `TreeMap.put`: comparator-aware; returns the previous value.
pub fn tree_put(rt: &mut Rt, m: &JMap, k: Val, v: Val) -> SResult<Val> {
    m.register_mutation()?;
    let _copy = super::allocation::Allocation::values(m.borrow().len())?;
    let keys = tree_keys(m);
    match tree_search(rt, &m.cmp, &keys, &k)? {
        Ok(i) => {
            let mut b = m.borrow_mut();
            let (_, slot) = b.get_index_mut(i).expect("index in range");
            Ok(std::mem::replace(slot, v))
        }
        Err(i) => {
            let key = Key::from_val(&k);
            m.reserve_entry(&key)?;
            m.borrow_mut().shift_insert(i, key, v);
            Ok(Val::Null)
        }
    }
}

/// Re-sort a comparator `TreeMap` after generic mutations.
fn tree_resort(rt: &mut Rt, m: &JMap) -> SResult<()> {
    if m.cmp.is_none() {
        return Ok(());
    }
    let _scratch = super::allocation::Allocation::entries(m.borrow().len().saturating_mul(4))?;
    let entries: Vec<(Key, Val)> = m.borrow().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let mut keys: Vec<Val> = entries.iter().map(|(k, _)| k.to_val()).collect();
    sort_vals(rt, &mut keys, m.cmp.as_ref())?;
    let mut out = IndexMap::with_capacity(entries.len());
    let mut all: IndexMap<Key, Val> = entries.into_iter().collect();
    for k in keys {
        let kk = Key::from_val(&k);
        if let Some(v) = all.shift_remove(&kk) {
            out.insert(kk, v);
        }
    }
    *m.borrow_mut() = out;
    Ok(())
}

fn tree_map_of(src: &JMap, entries: Vec<(Key, Val)>) -> SResult<Val> {
    let allocation = super::allocation::Allocation::entries(entries.len())?;
    Ok(Val::Map(Rc::new(JMap::new(MapKind::Tree, entries.into_iter().collect()).with_cmp(src.cmp.clone()))).allocated(allocation))
}

fn entry_of(k: &Key, v: &Val) -> Val {
    Val::Entry(Rc::new((k.to_val(), v.clone())))
}

fn no_such(msg: &str) -> Box<ScriptErr> {
    ScriptErr::new("no_such_element_exception", msg)
}

/// NavigableMap methods of a `TreeMap` (and comparator-aware writes).
fn tree_map_method(rt: &mut Rt, m: &Rc<JMap>, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    let _scratch = super::allocation::Allocation::entries(m.borrow().len().saturating_mul(4))?;
    let entries = || -> Vec<(Key, Val)> { m.ordered().iter().map(|(k, v)| (k.clone(), v.clone())).collect() };
    // Index of the boundary key: (found, insertion point) under the comparator.
    let mut locate = |rt: &mut Rt, k: &Val| -> SResult<(Option<usize>, usize)> {
        let keys = tree_keys(m);
        Ok(match tree_search(rt, &m.cmp, &keys, k)? {
            Ok(i) => (Some(i), i),
            Err(i) => (None, i),
        })
    };
    let pick = |e: &[(Key, Val)], i: Option<usize>, key_only: bool| -> Val {
        match i.and_then(|i| e.get(i)) {
            Some((k, v)) => {
                if key_only {
                    k.to_val()
                } else {
                    entry_of(k, v)
                }
            }
            None => Val::Null,
        }
    };
    Ok(Some(match (name, n) {
        ("put", 2) => tree_put(rt, m, args[0].clone(), args[1].clone())?,
        ("putAll", 1) => {
            if let Val::Map(o) = &args[0] {
                let items: Vec<(Val, Val)> = o.ordered().iter().map(|(k, v)| (k.to_val(), v.clone())).collect();
                for (k, v) in items {
                    tree_put(rt, m, k, v)?;
                }
            }
            Val::Null
        }
        ("get" | "containsKey" | "remove", 1) if m.cmp.is_some() => {
            let (found, _) = locate(rt, &args[0])?;
            let e = entries();
            match (name, found) {
                ("containsKey", f) => Val::Bool(f.is_some()),
                (_, None) => Val::Null,
                ("get", Some(i)) => e[i].1.clone(),
                (_, Some(i)) => m.borrow_mut().shift_remove_index(i).map(|x| x.1).unwrap_or(Val::Null),
            }
        }
        ("firstKey" | "lastKey", 0) => {
            let e = entries();
            let x = if name == "firstKey" { e.first() } else { e.last() };
            x.map(|(k, _)| k.to_val()).ok_or_else(|| no_such(""))?
        }
        ("firstEntry" | "lastEntry", 0) => {
            let e = entries();
            let x = if name == "firstEntry" { e.first() } else { e.last() };
            x.map(|(k, v)| entry_of(k, v)).unwrap_or(Val::Null)
        }
        ("pollFirstEntry" | "pollLastEntry", 0) => {
            let e = entries();
            if e.is_empty() {
                Val::Null
            } else {
                let i = if name == "pollFirstEntry" { 0 } else { e.len() - 1 };
                m.borrow_mut().shift_remove_index(i);
                entry_of(&e[i].0, &e[i].1)
            }
        }
        ("ceilingKey" | "ceilingEntry" | "floorKey" | "floorEntry" | "higherKey" | "higherEntry" | "lowerKey" | "lowerEntry", 1) => {
            let (found, ins) = locate(rt, &args[0])?;
            let e = entries();
            let i = match (&name[..name.len() - if name.ends_with("Key") { 3 } else { 5 }], found) {
                ("ceiling", Some(i)) | ("floor", Some(i)) => Some(i),
                ("ceiling", None) | ("higher", None) => Some(ins),
                ("higher", Some(i)) => Some(i + 1),
                ("floor", None) | ("lower", _) => ins.checked_sub(1),
                _ => None,
            };
            pick(&e, i, name.ends_with("Key"))
        }
        ("headMap", 1 | 2) | ("tailMap", 1 | 2) => {
            let incl = if n == 2 { args[1].as_bool()? } else { name == "tailMap" };
            let (found, ins) = locate(rt, &args[0])?;
            let e = entries();
            let cut = match found {
                Some(i) => {
                    if (name == "headMap") == incl {
                        i + 1
                    } else {
                        i
                    }
                }
                None => ins,
            };
            let part = if name == "headMap" { e[..cut].to_vec() } else { e[cut..].to_vec() };
            tree_map_of(m, part)?
        }
        ("subMap", 2 | 4) => {
            let (lo, lo_incl, hi, hi_incl) = if n == 2 { (&args[0], true, &args[1], false) } else { (&args[0], args[1].as_bool()?, &args[2], args[3].as_bool()?) };
            if tree_cmp(rt, &m.cmp, lo, hi)?.is_gt() {
                return Err(illegal_arg("fromKey > toKey"));
            }
            let (lf, li) = locate(rt, lo)?;
            let (hf, hi_i) = locate(rt, hi)?;
            let e = entries();
            let a = match lf {
                Some(i) if !lo_incl => i + 1,
                Some(i) => i,
                None => li,
            };
            let b = match hf {
                Some(i) if hi_incl => i + 1,
                Some(i) => i,
                None => hi_i,
            };
            tree_map_of(m, if a < b { e[a..b].to_vec() } else { Vec::new() })?
        }
        ("descendingMap", 0) => {
            let mut e = entries();
            e.reverse();
            let allocation = super::allocation::Allocation::entries(e.len())?;
            Val::Map(Rc::new(JMap::new(MapKind::Linked, e.into_iter().collect()))).allocated(allocation)
        }
        ("descendingKeySet", 0) => {
            let allocation = super::allocation::Allocation::values(m.borrow().len())?;
            new_set(entries().into_iter().rev().map(|(k, _)| k.to_val()).collect()).allocated(allocation)
        },
        ("comparator", 0) => m.cmp.clone().unwrap_or(Val::Null),
        ("clone", 0) => tree_map_of(m, entries())?,
        _ => {
            if m.cmp.is_some() && matches!(name, "putIfAbsent" | "compute" | "computeIfAbsent" | "merge") {
                let recv = Val::Map(m.clone());
                let mut tmp = m.clone();
                let r = map_method_untree(rt, &recv, &mut tmp, name, args)?;
                tree_resort(rt, m)?;
                return Ok(r);
            }
            return Ok(None);
        }
    }))
}

fn map_method_untree(rt: &mut Rt, recv: &Val, m: &mut Rc<JMap>, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    // Generic map semantics with the tree hook bypassed (the caller re-sorts).
    let n = args.len();
    let key = |i: usize| -> SResult<Key> { Ok(Key::from_val(arg(args, i)?)) };
    let _ = recv;
    Ok(Some(match (name, n) {
        ("putIfAbsent", 2) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned();
            match cur {
                Some(v) if !v.is_null() => v,
                _ => {
                    m.insert_checked(k, args[1].clone())?;
                    Val::Null
                }
            }
        }
        ("computeIfAbsent", 2) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned();
            match cur {
                Some(v) if !v.is_null() => v,
                _ => {
                    let v = rt.call_fn(&args[1], &[args[0].clone()])?;
                    if !v.is_null() {
                        m.insert_checked(k, v.clone())?;
                    }
                    v
                }
            }
        }
        ("compute", 2) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned().unwrap_or(Val::Null);
            let nv = rt.call_fn(&args[1], &[args[0].clone(), cur])?;
            if nv.is_null() {
                m.borrow_mut().shift_remove(&k);
            } else {
                m.insert_checked(k, nv.clone())?;
            }
            nv
        }
        ("merge", 3) => {
            let k = key(0)?;
            let cur = m.borrow().get(&k).cloned();
            let nv = match cur {
                Some(v) if !v.is_null() => rt.call_fn(&args[2], &[v, args[1].clone()])?,
                _ => args[1].clone(),
            };
            if nv.is_null() {
                m.borrow_mut().shift_remove(&k);
            } else {
                m.insert_checked(k, nv.clone())?;
            }
            nv
        }
        _ => return Ok(None),
    }))
}

/// Set `add` honoring the set kind; returns whether the set changed.
pub fn set_add(rt: &mut Rt, l: &JList, v: Val) -> SResult<bool> {
    if l.kind == CollKind::TreeSet {
        let items = l.ordered().clone();
        if l.cmp.is_none() && v.is_null() {
            return Err(npe("Cannot invoke \"java.lang.Comparable.compareTo(Object)\" because \"k1\" is null"));
        }
        // Java TreeMap.put compares the first key with itself (type check).
        if items.is_empty() {
            tree_cmp(rt, &l.cmp, &v, &v)?;
        }
        return match tree_search(rt, &l.cmp, &items, &v)? {
            Ok(_) => Ok(false),
            Err(i) => {
                l.reserve_growth(1)?;
                l.borrow_mut().insert(i, v);
                Ok(true)
            }
        };
    }
    if l.borrow().iter().any(|x| x.equals(&v)) {
        return Ok(false);
    }
    l.reserve_growth(1)?;
    l.borrow_mut().push(v);
    Ok(true)
}

fn tree_set_of(src: &JList, items: Vec<Val>) -> SResult<Val> {
    let allocation = super::allocation::Allocation::values(items.capacity())?;
    Ok(Val::Set(Rc::new(JList::new(CollKind::TreeSet, items).with_cmp(src.cmp.clone()))).allocated(allocation))
}

fn set_method(rt: &mut Rt, recv: &Val, l: &Rc<JList>, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    if l.immutable && matches!(name, "add" | "addAll" | "remove" | "removeAll" | "retainAll" | "removeIf" | "clear" | "pollFirst" | "pollLast") {
        return Err(ScriptErr::new("unsupported_operation_exception", ""));
    }
    let tree = l.kind == CollKind::TreeSet;
    let _scratch = super::allocation::Allocation::values(l.len().saturating_mul(3))?;
    let items = || l.ordered().clone();
    Ok(Some(match (name, n) {
        ("add", 1) => Val::Bool(set_add(rt, l, args[0].clone())?),
        ("addAll", 1) => {
            let mut changed = false;
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            for v in iter_values(&args[0])? {
                changed |= set_add(rt, l, v)?;
            }
            Val::Bool(changed)
        }
        ("contains", 1) if tree && l.cmp.is_some() => {
            let it = items();
            Val::Bool(tree_search(rt, &l.cmp, &it, &args[0])?.is_ok())
        }
        ("remove", 1) => {
            let mut b = l.borrow_mut();
            match b.iter().position(|x| x.equals(&args[0])) {
                Some(p) => {
                    b.remove(p);
                    Val::Bool(true)
                }
                None => Val::Bool(false),
            }
        }
        ("removeAll", 1) => {
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            let other = iter_values(&args[0])?;
            let mut b = l.borrow_mut();
            let before = b.len();
            b.retain(|x| !other.iter().any(|o| o.equals(x)));
            Val::Bool(b.len() != before)
        }
        ("clear", 0) => {
            l.borrow_mut().clear();
            Val::Null
        }
        ("removeIf", 1) => {
            let all = items();
            let mut keep = Vec::with_capacity(all.len());
            for v in all {
                if !rt.call_fn(&args[0], std::slice::from_ref(&v))?.as_bool()? {
                    keep.push(v);
                }
            }
            let changed = keep.len() != l.len();
            *l.borrow_mut() = keep;
            Val::Bool(changed)
        }
        ("retainAll", 1) => {
            let _copy = super::allocation::Allocation::iterable(&args[0])?;
            let other = iter_values(&args[0])?;
            let mut b = l.borrow_mut();
            let before = b.len();
            b.retain(|x| other.iter().any(|o| o.equals(x)));
            Val::Bool(b.len() != before)
        }
        ("iterator", 0) => super::jmisc::new_iter(l.ordered().iter().cloned(), Some(l.clone()))?,
        ("clone", 0) => {
            let allocation = super::allocation::Allocation::values(l.len())?;
            Val::Set(Rc::new(JList::new(l.kind, items()).with_cmp(l.cmp.clone()))).allocated(allocation)
        },
        ("first" | "last" | "getFirst" | "getLast", 0) if tree => {
            let it = items();
            let x = if name.ends_with("irst") { it.first() } else { it.last() };
            x.cloned().ok_or_else(|| no_such(""))?
        }
        ("pollFirst" | "pollLast", 0) if tree => {
            let it = items();
            if it.is_empty() {
                Val::Null
            } else {
                let i = if name == "pollFirst" { 0 } else { it.len() - 1 };
                l.borrow_mut().remove(i)
            }
        }
        ("ceiling" | "floor" | "higher" | "lower", 1) if tree => {
            let it = items();
            let i = match (name, tree_search(rt, &l.cmp, &it, &args[0])?) {
                ("ceiling" | "floor", Ok(i)) => Some(i),
                ("ceiling" | "higher", Err(i)) => Some(i),
                ("higher", Ok(i)) => Some(i + 1),
                (_, Err(i)) | (_, Ok(i)) => i.checked_sub(1),
            };
            i.and_then(|i| it.get(i).cloned()).unwrap_or(Val::Null)
        }
        ("headSet" | "tailSet", 1 | 2) if tree => {
            let incl = if n == 2 { args[1].as_bool()? } else { name == "tailSet" };
            let it = items();
            let cut = match tree_search(rt, &l.cmp, &it, &args[0])? {
                Ok(i) => {
                    if (name == "headSet") == incl {
                        i + 1
                    } else {
                        i
                    }
                }
                Err(i) => i,
            };
            tree_set_of(l, if name == "headSet" { it[..cut].to_vec() } else { it[cut..].to_vec() })?
        }
        ("subSet", 2 | 4) if tree => {
            let (lo, lo_incl, hi, hi_incl) = if n == 2 { (&args[0], true, &args[1], false) } else { (&args[0], args[1].as_bool()?, &args[2], args[3].as_bool()?) };
            if tree_cmp(rt, &l.cmp, lo, hi)?.is_gt() {
                return Err(illegal_arg("fromKey > toKey"));
            }
            let it = items();
            let a = match tree_search(rt, &l.cmp, &it, lo)? {
                Ok(i) if !lo_incl => i + 1,
                Ok(i) | Err(i) => i,
            };
            let b = match tree_search(rt, &l.cmp, &it, hi)? {
                Ok(i) if hi_incl => i + 1,
                Ok(i) | Err(i) => i,
            };
            tree_set_of(l, if a < b { it[a..b].to_vec() } else { Vec::new() })?
        }
        ("descendingSet", 0) if tree => {
            let allocation = super::allocation::Allocation::values(l.len())?;
            new_set(items().into_iter().rev().collect()).allocated(allocation)
        },
        ("descendingIterator", 0) if tree => super::jmisc::new_iter(l.ordered().iter().rev().cloned(), None)?,
        ("comparator", 0) if tree => l.cmp.clone().unwrap_or(Val::Null),
        _ => {
            let _ = recv;
            return Ok(None);
        }
    }))
}

/// `new HashMap(..)` / `new LinkedHashMap(..)` / `new TreeMap(..)`.
pub fn new_map_of(rt: &mut Rt, class: &str, arg0: Option<&Val>) -> SResult<Val> {
    let kind = match class {
        "TreeMap" => MapKind::Tree,
        "LinkedHashMap" => MapKind::Linked,
        _ => MapKind::Hash,
    };
    let allocation = super::allocation::Allocation::entries(match arg0 { Some(Val::Map(m)) => m.borrow().len(), _ => 0 })?;
    let (src, cmp, cap) = match arg0 {
        Some(Val::Map(o)) => {
            let entries: IndexMap<Key, Val> = o.ordered().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            let cmp = if kind == MapKind::Tree && o.kind == MapKind::Tree { o.cmp.clone() } else { None };
            let cap = super::value::copy_capacity(entries.len(), false);
            (entries, cmp, cap)
        }
        Some(c) if is_comparator(c) => (IndexMap::new(), Some(c.clone()), 16),
        Some(Val::Int(c)) if *c < 0 => return Err(illegal_arg(format!("Illegal initial capacity: {c}"))),
        Some(v) if v.is_number() => (IndexMap::new(), None, super::value::table_size_for(v.to_i64()?.max(1) as usize)),
        Some(Val::Null) if kind == MapKind::Tree => (IndexMap::new(), None, 16),
        Some(Val::Null) => return Err(npe("Cannot invoke \"java.util.Map.size()\" because \"m\" is null")),
        _ => (IndexMap::new(), None, 16),
    };
    if kind == MapKind::Tree {
        let empty = super::allocation::Allocation::entries(0)?;
        let Val::Map(m) = Val::Map(Rc::new(JMap::new(MapKind::Tree, IndexMap::new()).with_cmp(cmp))).allocated(empty) else { unreachable!() };
        for (k, v) in src {
            tree_put(rt, &m, k.to_val(), v)?;
        }
        return Ok(Val::Map(m));
    }
    Ok(Val::Map(Rc::new(JMap::new(kind, src).with_cap(cap))).allocated(allocation))
}

/// `new HashSet(..)` / `new LinkedHashSet(..)` / `new TreeSet(..)`.
pub fn new_set_of(rt: &mut Rt, class: &str, arg0: Option<&Val>) -> SResult<Val> {
    let kind = match class {
        "TreeSet" => CollKind::TreeSet,
        "LinkedHashSet" => CollKind::LinkedSet,
        _ => CollKind::HashSet,
    };
    let _copy = arg0.map(super::allocation::Allocation::iterable).transpose()?;
    let (vals, cmp, cap) = match arg0 {
        Some(c) if is_comparator(c) => (Vec::new(), Some(c.clone()), 16),
        Some(Val::Set(o)) if kind == CollKind::TreeSet && o.kind == CollKind::TreeSet => (o.ordered().clone(), o.cmp.clone(), 16),
        Some(v) if v.is_number() => (Vec::new(), None, super::value::table_size_for(v.to_i64()?.max(1) as usize)),
        Some(Val::Null) => return Err(npe("Cannot invoke \"java.util.Collection.size()\" because \"c\" is null")),
        Some(v) => {
            let vals = iter_values(v)?;
            let n = vals.len();
            (vals, None, super::value::copy_capacity(n, true))
        }
        None => (Vec::new(), None, 16),
    };
    let empty = super::allocation::Allocation::values(0)?;
    let Val::Set(l) = Val::Set(Rc::new(JList::new(kind, Vec::new()).with_cmp(cmp).with_cap(cap))).allocated(empty) else { unreachable!() };
    for v in vals {
        set_add(rt, &l, v)?;
    }
    Ok(Val::Set(l))
}

pub(super) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// SHA-1 (`String.sha1()` augmentation).
pub(super) fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod sha_tests {
    #[test]
    fn sha1_known() {
        assert_eq!(super::hex(&super::sha1(b"foo")), "0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33");
        assert_eq!(super::hex(&super::sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }
}
