//! `java.time` values for Painless: `LocalDate`, `LocalDateTime`,
//! `LocalTime`, `OffsetDateTime`, `OffsetTime`, `ZonedDateTime` (any zone),
//! nanosecond `Instant`, `Duration`, `Period`, `ZoneId`/`ZoneOffset`,
//! `YearMonth`, `Year`, `MonthDay`, `DateTimeFormatter`, `TemporalAdjusters`,
//! `TextStyle`, `Locale`, `ChronoField`/`IsoFields`, `Clock`, `ZoneRules`.
//!
//! UTC `ZonedDateTime`s with millisecond precision stay `Val::Date(ms)` and
//! millisecond `Instant`s `Val::Instant(ms)` (doc values produce those);
//! everything else is a `Val::Time`. The arithmetic is a port of the
//! OpenJDK algorithms (proleptic years in ±999999999, epoch-day based) so
//! results and error messages match Java's.

use super::ast::Ty;
use super::error::{class_cast, illegal_arg, npe, unsupported, SResult, ScriptErr};
use super::runtime::Rt;
use super::value::{Val, CHRONO_UNITS, DAYS, MONTHS};
use chrono::{Offset as _, TimeZone as _};
use chrono_tz::{OffsetComponents as _, OffsetName as _};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::rc::Rc;

/// Static classes handled here (consulted before the built-in statics).
pub const CLASSES: &[&str] = &[
    "LocalDate", "LocalDateTime", "LocalTime", "OffsetDateTime", "OffsetTime", "ZonedDateTime", "Instant", "Duration", "Period", "ZoneId", "ZoneOffset", "YearMonth",
    "Year", "MonthDay", "DateTimeFormatter", "TemporalAdjusters", "TextStyle", "Locale", "ChronoField", "ChronoUnit", "DayOfWeek", "Month", "Clock", "FormatStyle",
    "ResolverStyle", "IsoChronology", "IsoFields", "TemporalAdjuster", "ZoneRules", "DateTimeParseException", "DateTimeException",
];

const NPS: i64 = 1_000_000_000;
const NPD: i64 = 86_400 * NPS;
const SPD: i64 = 86_400;
const MIN_YEAR: i64 = -999_999_999;
const MAX_YEAR: i64 = 999_999_999;
const DAYS_0000_TO_1970: i64 = 719_528;
const INSTANT_MIN_SEC: i64 = -31_557_014_167_219_200;
const INSTANT_MAX_SEC: i64 = 31_556_889_864_403_199;

// ---------------------------------------------------------------------------
// Errors

// JDK stack frames ES prints above the script frame (`script_stack`). Each
// error carries the frames of its Java throw site; the dispatch layer
// appends the public-API call chain (`fr`).

type Fr = &'static str;

macro_rules! jf {
    ($cls:literal, $m:literal, $file:literal, $l:literal) => {
        concat!("java.base/", $cls, ".", $m, "(", $file, ".java:", $l, ")")
    };
}
macro_rules! jt {
    ($cls:literal, $m:literal, $l:literal) => {
        concat!("java.base/java.time.", $cls, ".", $m, "(", $cls, ".java:", $l, ")")
    };
}

const VR_CHECK: Fr = jf!("java.time.temporal.ValueRange", "checkValidValue", "ValueRange", 319);
const CF_CHECK: Fr = jf!("java.time.temporal.ChronoField", "checkValidValue", "ChronoField", 721);
const VR_CHECK_INT: Fr = jf!("java.time.temporal.ValueRange", "checkValidIntValue", "ValueRange", 338);
const CF_CHECK_INT: Fr = jf!("java.time.temporal.ChronoField", "checkValidIntValue", "ChronoField", 740);
const REQ_NON_NULL: Fr = jf!("java.util.Objects", "requireNonNull", "Objects", 246);
const MATH_ADD: Fr = jf!("java.lang.Math", "addExact", "Math", 935);
const MATH_MUL: Fr = jf!("java.lang.Math", "multiplyExact", "Math", 1036);
const MATH_MUL2: Fr = jf!("java.lang.Math", "multiplyExact", "Math", 1012);
const MATH_MUL_INT: Fr = jf!("java.lang.Math", "multiplyExact", "Math", 996);
const MATH_TO_INT: Fr = jf!("java.lang.Math", "toIntExact", "Math", 1375);
const ENUM_VALUEOF: Fr = jf!("java.lang.Enum", "valueOf", "Enum", 293);
const ENUM_VALUEOF_NULL: Fr = jf!("java.lang.Enum", "valueOf", "Enum", 291);
const TA_RANGE: Fr = jf!("java.time.temporal.TemporalAccessor", "range", "TemporalAccessor", 174);
const TA_GET: Fr = jf!("java.time.temporal.TemporalAccessor", "get", "TemporalAccessor", 218);
const DTF_PARSE_RESOLVED: Fr = jf!("java.time.format.DateTimeFormatter", "parseResolved0", "DateTimeFormatter", 2131);
const DTF_PARSE_UNPARSED: Fr = jf!("java.time.format.DateTimeFormatter", "parseResolved0", "DateTimeFormatter", 2134);
const DTF_CREATE_ERROR: Fr = jf!("java.time.format.DateTimeFormatter", "createError", "DateTimeFormatter", 2102);
const DTF_PARSE_Q: Fr = jf!("java.time.format.DateTimeFormatter", "parse", "DateTimeFormatter", 2033);
const DTF_PARSE_Q_ERR: Fr = jf!("java.time.format.DateTimeFormatter", "parse", "DateTimeFormatter", 2037);
const DTF_PARSE_1: Fr = jf!("java.time.format.DateTimeFormatter", "parse", "DateTimeFormatter", 1959);
const DTF_PARSE_1_ERR: Fr = jf!("java.time.format.DateTimeFormatter", "parse", "DateTimeFormatter", 1963);
const DTFB_APPEND_PATTERN: Fr = jf!("java.time.format.DateTimeFormatterBuilder", "appendPattern", "DateTimeFormatterBuilder", 1910);
const DTFB_PARSE_PATTERN_FIELD: Fr = jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 1942);
const DTF_OF_PATTERN_1: Fr = jf!("java.time.format.DateTimeFormatter", "ofPattern", "DateTimeFormatter", 569);
const DTF_OF_PATTERN_2: Fr = jf!("java.time.format.DateTimeFormatter", "ofPattern", "DateTimeFormatter", 593);
const DTF_FORMAT_TAIL: [Fr; 3] = [
    jf!("java.time.format.DateTimeFormatterBuilder$CompositePrinterParser", "format", "DateTimeFormatterBuilder", 2549),
    jf!("java.time.format.DateTimeFormatter", "formatTo", "DateTimeFormatter", 1928),
    jf!("java.time.format.DateTimeFormatter", "format", "DateTimeFormatter", 1902),
];
const DTPC_GET_VALUE: Fr = jf!("java.time.format.DateTimePrintContext", "getValue", "DateTimePrintContext", 373);
const DTPC_GET_ZONE: Fr = jf!("java.time.format.DateTimePrintContext", "getValue", "DateTimePrintContext", 353);
const DTPC_WRAPPED: Fr = jf!("java.time.format.DateTimePrintContext$1", "getLong", "DateTimePrintContext", 282);
const CU_BETWEEN: Fr = jf!("java.time.temporal.ChronoUnit", "between", "ChronoUnit", 271);
const CU_ADD_TO: Fr = jf!("java.time.temporal.ChronoUnit", "addTo", "ChronoUnit", 265);
const DUR_BETWEEN: Fr = jt!("Duration", "between", 520);
const DUR_ADD_TO: Fr = jt!("Duration", "addTo", 1151);
const DUR_SUB_FROM: Fr = jt!("Duration", "subtractFrom", 1186);
const PER_ADD_TO_DAYS: Fr = jt!("Period", "addTo", 909);
const PER_ADD_TO_MONTHS: Fr = jt!("Period", "addTo", 905);
const LD_CREATE_479: Fr = jt!("LocalDate", "create", 479);
const LD_CREATE_477: Fr = jt!("LocalDate", "create", 477);
const LD_OF_292: Fr = jt!("LocalDate", "of", 292);
const LD_OF_293: Fr = jt!("LocalDate", "of", 293);
const LD_OF_294: Fr = jt!("LocalDate", "of", 294);
const LD_OF_295: Fr = jt!("LocalDate", "of", 295);
const LD_OF_YEAR_DAY_312: Fr = jt!("LocalDate", "ofYearDay", 312);
const LD_OF_YEAR_DAY_313: Fr = jt!("LocalDate", "ofYearDay", 313);
const LD_OF_YEAR_DAY_316: Fr = jt!("LocalDate", "ofYearDay", 316);
const LD_OF_EPOCH_DAY: Fr = jt!("LocalDate", "ofEpochDay", 365);
const LD_PLUS_DAYS_1394: Fr = jt!("LocalDate", "plusDays", 1394);
const LD_PLUS_DAYS_1395: Fr = jt!("LocalDate", "plusDays", 1395);
const LD_PLUS_1273: Fr = jt!("LocalDate", "plus", 1273);
const LD_UNTIL_1656: Fr = jt!("LocalDate", "until", 1656);
const LD_FROM_423: Fr = jt!("LocalDate", "from", 423);
const LD_FROM_419: Fr = jt!("LocalDate", "from", 419);
const LD_GET0_718: Fr = jt!("LocalDate", "get0", 718);
const LD_GET0_710: Fr = jt!("LocalDate", "get0", 710);
const LD_GET0_714: Fr = jt!("LocalDate", "get0", 714);
const LD_GETLONG_698: Fr = jt!("LocalDate", "getLong", 698);
const LD_GET_661: Fr = jt!("LocalDate", "get", 661);
const LD_WITH_1067: Fr = jt!("LocalDate", "with", 1067);
const LD_WITH_1052: Fr = jt!("LocalDate", "with", 1052);
const LD_WITH_1070: Fr = jt!("LocalDate", "with", 1070);
const LT_OF_308: Fr = jt!("LocalTime", "of", 308);
const LT_OF_312: Fr = jt!("LocalTime", "of", 312);
const LT_OF_329: Fr = jt!("LocalTime", "of", 329);
const LT_OF_333: Fr = jt!("LocalTime", "of", 333);
const LT_OF_334: Fr = jt!("LocalTime", "of", 334);
const LT_OF_351: Fr = jt!("LocalTime", "of", 351);
const LT_OF_352: Fr = jt!("LocalTime", "of", 352);
const LT_OF_353: Fr = jt!("LocalTime", "of", 353);
const LT_OF_354: Fr = jt!("LocalTime", "of", 354);
const LT_PLUS_1087: Fr = jt!("LocalTime", "plus", 1087);
const LT_PLUS_127: Fr = jt!("LocalTime", "plus", 127);
const LT_UNTIL_1429: Fr = jt!("LocalTime", "until", 1429);
const LT_FROM_443: Fr = jt!("LocalTime", "from", 443);
const LT_GET0_712: Fr = jt!("LocalTime", "get0", 712);
const LT_GET0_698: Fr = jt!("LocalTime", "get0", 698);
const LT_GETLONG_690: Fr = jt!("LocalTime", "getLong", 690);
const LT_GET_653: Fr = jt!("LocalTime", "get", 653);
const LT_RANGE_620: Fr = jt!("LocalTime", "range", 620);
const LT_WITH_886: Fr = jt!("LocalTime", "with", 886);
const LT_WITH_869: Fr = jt!("LocalTime", "with", 869);
const LT_WITH_127: Fr = jt!("LocalTime", "with", 127);
const LT_WITH_781: Fr = jt!("LocalTime", "with", 781);
const LT_TRUNC_988: Fr = jt!("LocalTime", "truncatedTo", 988);
const LDT_OF_315: Fr = jt!("LocalDateTime", "of", 315);
const LDT_OF_316: Fr = jt!("LocalDateTime", "of", 316);
const LDT_OF_340: Fr = jt!("LocalDateTime", "of", 340);
const LDT_OF_341: Fr = jt!("LocalDateTime", "of", 341);
const LDT_OF_365: Fr = jt!("LocalDateTime", "of", 365);
const LDT_OF_366: Fr = jt!("LocalDateTime", "of", 366);
const LDT_PLUS_1190: Fr = jt!("LocalDateTime", "plus", 1190);
const LDT_UNTIL_1731: Fr = jt!("LocalDateTime", "until", 1731);
const LDT_FROM_465: Fr = jt!("LocalDateTime", "from", 465);
const LDT_GETLONG_720: Fr = jt!("LocalDateTime", "getLong", 720);
const LDT_GET_689: Fr = jt!("LocalDateTime", "get", 689);
const LDT_WITH_969: Fr = jt!("LocalDateTime", "with", 969);
const LDT_WITH_967: Fr = jt!("LocalDateTime", "with", 967);
const LDT_TRUNC_1120: Fr = jt!("LocalDateTime", "truncatedTo", 1120);
const ODT_UNTIL_1666: Fr = jt!("OffsetDateTime", "until", 1666);
const ODT_FROM_375: Fr = jt!("OffsetDateTime", "from", 375);
const ZDT_PLUS_1604: Fr = jt!("ZonedDateTime", "plus", 1604);
const ZDT_UNTIL_2140: Fr = jt!("ZonedDateTime", "until", 2140);
const ZDT_FROM_570: Fr = jt!("ZonedDateTime", "from", 570);
const ZDT_WITH_1311: Fr = jt!("ZonedDateTime", "with", 1311);
const ZDT_GET_823: Fr = jt!("ZonedDateTime", "get", 823);
const ZDT_GET_820: Fr = jt!("ZonedDateTime", "get", 820);
const ZDT_OF_INSTANT_412: Fr = jt!("ZonedDateTime", "ofInstant", 412);
const ZDT_OF_LOCAL_372: Fr = jt!("ZonedDateTime", "ofLocal", 372);
const ZDT_OF_297: Fr = jt!("ZonedDateTime", "of", 297);
const INST_CREATE: Fr = jt!("Instant", "create", 414);
const INST_PLUS_SECONDS: Fr = jt!("Instant", "plusSeconds", 909);
const INST_PLUS_887: Fr = jt!("Instant", "plus", 887);
const INST_PLUS_882: Fr = jt!("Instant", "plus", 882);
const INST_UNTIL_1185: Fr = jt!("Instant", "until", 1185);
const INST_FROM_381: Fr = jt!("Instant", "from", 381);
const INST_FROM_374: Fr = jt!("Instant", "from", 374);
const INST_GETLONG_603: Fr = jt!("Instant", "getLong", 603);
const INST_GET_566: Fr = jt!("Instant", "get", 566);
const INST_WITH_719: Fr = jt!("Instant", "with", 719);
const INST_WITH_707: Fr = jt!("Instant", "with", 707);
const INST_TRUNC_754: Fr = jt!("Instant", "truncatedTo", 754);
const YM_FROM_283: Fr = jt!("YearMonth", "from", 283);
const YM_UNTIL_1075: Fr = jt!("YearMonth", "until", 1075);
const YM_PLUS_833: Fr = jt!("YearMonth", "plus", 833);
const YM_GETLONG_513: Fr = jt!("YearMonth", "getLong", 513);
const YM_WITH_711: Fr = jt!("YearMonth", "with", 711);
const Y_FROM_264: Fr = jt!("Year", "from", 264);
const Y_UNTIL_927: Fr = jt!("Year", "until", 927);
const Y_PLUS_720: Fr = jt!("Year", "plus", 720);
const Y_GETLONG_507: Fr = jt!("Year", "getLong", 507);
const Y_WITH_631: Fr = jt!("Year", "with", 631);
const MD_FROM_298: Fr = jt!("MonthDay", "from", 298);
const MD_GETLONG_470: Fr = jt!("MonthDay", "getLong", 470);
const MD_OF_239: Fr = jt!("MonthDay", "of", 239);
const MD_OF_242: Fr = jt!("MonthDay", "of", 242);
const MD_OF_264: Fr = jt!("MonthDay", "of", 264);
const DOW_FROM_195: Fr = jt!("DayOfWeek", "from", 195);
const DOW_OF_165: Fr = jt!("DayOfWeek", "of", 165);
const DOW_GETLONG_348: Fr = jt!("DayOfWeek", "getLong", 348);
const DOW_RANGE_285: Fr = jt!("DayOfWeek", "range", 285);
const DOW_GET_318: Fr = jt!("DayOfWeek", "get", 318);
const MON_FROM_222: Fr = jt!("Month", "from", 222);
const MON_OF_187: Fr = jt!("Month", "of", 187);
const MON_GETLONG_375: Fr = jt!("Month", "getLong", 375);
const MON_RANGE_312: Fr = jt!("Month", "range", 312);
const MON_GET_345: Fr = jt!("Month", "get", 345);
const ZID_FROM_466: Fr = jt!("ZoneId", "from", 466);
const ZID_OF_359: Fr = jt!("ZoneId", "of", 359);
const ZID_OF_405: Fr = jt!("ZoneId", "of", 405);
const ZID_OF_411: Fr = jt!("ZoneId", "of", 411);
const ZOFF_FROM_367: Fr = jt!("ZoneOffset", "from", 367);
const ZOFF_OF_240: Fr = jt!("ZoneOffset", "of", 240);
const ZOFF_OF_260: Fr = jt!("ZoneOffset", "of", 260);
const ZOFF_OF_264: Fr = jt!("ZoneOffset", "of", 264);
const ZOFF_OF_269: Fr = jt!("ZoneOffset", "of", 269);
const ZOFF_PARSE_NUMBER: Fr = jt!("ZoneOffset", "parseNumber", 288);
const ZOFF_VALIDATE_383: Fr = jt!("ZoneOffset", "validate", 383);
const ZOFF_VALIDATE_388: Fr = jt!("ZoneOffset", "validate", 388);
const ZOFF_VALIDATE_398: Fr = jt!("ZoneOffset", "validate", 398);
const ZOFF_OF_HMS_336: Fr = jt!("ZoneOffset", "ofHoursMinutesSeconds", 336);
const PARSED_GETLONG: Fr = jf!("java.time.format.Parsed", "getLong", "Parsed", 215);
const OT_GETLONG_538: Fr = jt!("OffsetTime", "getLong", 538);
const DUR_PLUS_878: Fr = jt!("Duration", "plus", 878);
const DUR_PLUS_SECONDS_835: Fr = jt!("Duration", "plusSeconds", 835);
const DUR_CREATE_1087: Fr = jt!("Duration", "create", 1087);
const DUR_PLUS_764: Fr = jt!("Duration", "plus", 764);

/// Intern a frame list so it can live in `ScriptErr::frames`.
fn intern_frames(parts: &[Fr]) -> &'static [Fr] {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<Vec<Fr>, &'static [Fr]>>> = OnceLock::new();
    let mut c = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|p| p.into_inner());
    if let Some(s) = c.get(parts) {
        return s;
    }
    let leaked: &'static [Fr] = Box::leak(parts.to_vec().into_boxed_slice());
    c.insert(parts.to_vec(), leaked);
    leaked
}

/// Append outer (caller) frames to an error.
fn fr(e: Box<ScriptErr>, outer: &[Fr]) -> Box<ScriptErr> {
    if outer.is_empty() {
        return e;
    }
    let mut v: Vec<Fr> = e.frames.to_vec();
    v.extend_from_slice(outer);
    e.frames(intern_frames(&v))
}
/// Replace an error's frames.
fn frs(e: Box<ScriptErr>, frames: &[Fr]) -> Box<ScriptErr> {
    e.frames(intern_frames(frames))
}
/// `Objects.requireNonNull(x, name)` failure.
fn npe_req(name: &str, outer: &[Fr]) -> Box<ScriptErr> {
    let mut v = vec![REQ_NON_NULL];
    v.extend_from_slice(outer);
    frs(npe(name), &v)
}

fn dte(m: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("date_time_exception", m)
}
fn utte(m: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("unsupported_temporal_type_exception", m)
}
fn dtpe(m: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("date_time_parse_exception", m)
}
fn arith(m: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("arithmetic_exception", m)
}
fn overflow() -> Box<ScriptErr> {
    frs(arith("long overflow"), &[MATH_ADD])
}
fn overflow_mul() -> Box<ScriptErr> {
    frs(arith("long overflow"), &[MATH_MUL, MATH_MUL2])
}
fn int_overflow() -> Box<ScriptErr> {
    frs(arith("integer overflow"), &[MATH_TO_INT])
}
/// Painless `def` argument cast failure.
fn cast_to(v: &Val, to: &str) -> Box<ScriptErr> {
    class_cast(format!("Cannot cast from [{}] to [{}].", v.def_name(), to))
}
fn unsupported_unit(u: Unit) -> Box<ScriptErr> {
    utte(format!("Unsupported unit: {}", unit_name(u)))
}
fn unsupported_field(f: F) -> Box<ScriptErr> {
    utte(format!("Unsupported field: {}", f.name()))
}
fn unable(what: &str, v: &Val) -> Box<ScriptErr> {
    let e = dte(format!("Unable to obtain {what} from TemporalAccessor: {} of type {}", v.to_java_string(), v.type_name()));
    let f = match what {
        "LocalDate" => LD_FROM_423,
        "LocalTime" => LT_FROM_443,
        "LocalDateTime" => LDT_FROM_465,
        "Instant" => INST_FROM_381,
        "ZonedDateTime" => ZDT_FROM_570,
        "OffsetDateTime" => ODT_FROM_375,
        "YearMonth" => YM_FROM_283,
        "Year" => Y_FROM_264,
        "MonthDay" => MD_FROM_298,
        "DayOfWeek" => DOW_FROM_195,
        "Month" => MON_FROM_222,
        "ZoneId" => ZID_FROM_466,
        "ZoneOffset" => ZOFF_FROM_367,
        _ => return e,
    };
    frs(e, &[f])
}
/// `X.from(null)`.
fn unable_null(what: &str) -> Box<ScriptErr> {
    match what {
        "LocalDate" => npe_req("temporal", &[LD_FROM_419]),
        "Instant" => npe_req("temporal", &[INST_FROM_374]),
        "Year" => npe_req("temporal", &[jt!("Year", "from", 256)]),
        _ => npe("temporal"),
    }
}

// ---------------------------------------------------------------------------
// Argument helpers

fn at(args: &[Val], i: usize) -> Val {
    args.get(i).cloned().unwrap_or(Val::Null)
}
fn long_arg(args: &[Val], i: usize) -> SResult<i64> {
    match at(args, i) {
        Val::Int(v) => Ok(v as i64),
        Val::Long(v) => Ok(v),
        Val::Char(c) => Ok(c as i64),
        Val::Null => Err(illegal_arg("Cannot cast null to a primitive type [long].")),
        other => Err(cast_to(&other, "long")),
    }
}
fn int_arg(args: &[Val], i: usize) -> SResult<i64> {
    match at(args, i) {
        Val::Int(v) => Ok(v as i64),
        Val::Char(c) => Ok(c as i64),
        Val::Null => Err(illegal_arg("Cannot cast null to a primitive type [int].")),
        other => Err(cast_to(&other, "int")),
    }
}
fn bool_arg(args: &[Val], i: usize) -> SResult<bool> {
    match at(args, i) {
        Val::Bool(b) => Ok(b),
        Val::Null => Err(illegal_arg("Cannot cast null to a primitive type [boolean].")),
        other => Err(cast_to(&other, "boolean")),
    }
}
fn str_arg(args: &[Val], i: usize, pname: &str) -> SResult<String> {
    match at(args, i) {
        Val::Str(s) => Ok(s.to_string()),
        Val::Null => Err(npe(pname)),
        other => Err(cast_to(&other, "java.lang.String")),
    }
}
fn zone_arg(v: &Val, pname: &str) -> SResult<Zone> {
    match v {
        Val::Time(t) => match &**t {
            JTime::Zone(z) => Ok(z.clone()),
            _ => Err(cast_to(v, "java.time.ZoneId")),
        },
        Val::Null => Err(npe(pname)),
        _ => Err(cast_to(v, "java.time.ZoneId")),
    }
}
fn offset_arg(v: &Val, pname: &str) -> SResult<i32> {
    match v {
        Val::Time(t) => match &**t {
            JTime::Zone(Zone::Offset(o)) => Ok(*o),
            _ => Err(cast_to(v, "java.time.ZoneOffset")),
        },
        Val::Null => Err(npe(pname)),
        _ => Err(cast_to(v, "java.time.ZoneOffset")),
    }
}
fn fmt_arg(v: &Val, pname: &str) -> SResult<Rc<Fmt>> {
    match v {
        Val::Time(t) => match &**t {
            JTime::Formatter(f) => Ok(f.clone()),
            _ => Err(cast_to(v, "java.time.format.DateTimeFormatter")),
        },
        // Formatter values from the legacy string-based path.
        Val::Str(s) if super::timefmt::is_formatter(s) => match s.strip_prefix(super::timefmt::PATTERN_PREFIX) {
            Some(p) => Ok(Rc::new(Fmt::of_pattern(p, Locale::en())?)),
            None => iso_formatter(s).ok_or_else(|| cast_to(v, "java.time.format.DateTimeFormatter")),
        },
        Val::Null => Err(npe(pname)),
        _ => Err(cast_to(v, "java.time.format.DateTimeFormatter")),
    }
}
fn unit_arg(v: &Val) -> SResult<Unit> {
    match v {
        Val::ChronoUnit(u) => Ok(*u),
        Val::Time(t) => match &**t {
            JTime::Unit(u) => Ok(*u),
            _ => Err(cast_to(v, "java.time.temporal.TemporalUnit")),
        },
        Val::Null => Err(npe("Cannot invoke \"java.time.temporal.TemporalUnit.addTo(java.time.temporal.Temporal, long)\" because \"unit\" is null")),
        _ => Err(cast_to(v, "java.time.temporal.TemporalUnit")),
    }
}
fn field_arg(v: &Val) -> SResult<F> {
    match v {
        Val::Time(t) => match &**t {
            JTime::Field(f) => Ok(*f),
            _ => Err(cast_to(v, "java.time.temporal.TemporalField")),
        },
        Val::Null => Err(npe("field")),
        _ => Err(cast_to(v, "java.time.temporal.TemporalField")),
    }
}
fn locale_arg(v: &Val) -> SResult<Locale> {
    match v {
        Val::Time(t) => match &**t {
            JTime::Locale(l) => Ok(l.clone()),
            _ => Err(cast_to(v, "java.util.Locale")),
        },
        Val::Null => Err(npe("locale")),
        _ => Err(cast_to(v, "java.util.Locale")),
    }
}
fn style_arg(v: &Val) -> SResult<u8> {
    match v {
        Val::Time(t) => match &**t {
            JTime::TextStyle(s) => Ok(*s),
            _ => Err(cast_to(v, "java.time.format.TextStyle")),
        },
        Val::Null => Err(npe("style")),
        _ => Err(cast_to(v, "java.time.format.TextStyle")),
    }
}
fn dow_arg(v: &Val) -> SResult<u8> {
    match v {
        Val::DayOfWeek(d) => Ok(*d),
        Val::Null => Err(npe("dayOfWeek")),
        _ => Err(cast_to(v, "java.time.DayOfWeek")),
    }
}
fn ldate_arg(v: &Val) -> SResult<LDate> {
    match v {
        Val::Time(t) => match &**t {
            JTime::LocalDate(d) => Ok(*d),
            _ => Err(cast_to(v, "java.time.LocalDate")),
        },
        Val::Null => Err(npe("date")),
        _ => Err(cast_to(v, "java.time.LocalDate")),
    }
}
fn ltime_arg(v: &Val) -> SResult<i64> {
    match v {
        Val::Time(t) => match &**t {
            JTime::LocalTime(n) => Ok(*n),
            _ => Err(cast_to(v, "java.time.LocalTime")),
        },
        Val::Null => Err(npe("time")),
        _ => Err(cast_to(v, "java.time.LocalTime")),
    }
}
fn ldt_arg(v: &Val) -> SResult<LDT> {
    match v {
        Val::Time(t) => match &**t {
            JTime::LocalDateTime(d) => Ok(*d),
            _ => Err(cast_to(v, "java.time.LocalDateTime")),
        },
        Val::Null => Err(npe("localDateTime")),
        _ => Err(cast_to(v, "java.time.LocalDateTime")),
    }
}
fn instant_arg(v: &Val) -> SResult<(i64, u32)> {
    match v {
        Val::Instant(ms) => Ok(ms_to_inst(*ms)),
        Val::Time(t) => match &**t {
            JTime::Instant(s, n) => Ok((*s, *n)),
            _ => Err(cast_to(v, "java.time.Instant")),
        },
        Val::Null => Err(npe("instant")),
        _ => Err(cast_to(v, "java.time.Instant")),
    }
}
fn duration_arg(v: &Val) -> SResult<(i64, u32)> {
    match v {
        Val::Time(t) => match &**t {
            JTime::Duration(s, n) => Ok((*s, *n)),
            _ => Err(cast_to(v, "java.time.Duration")),
        },
        Val::Null => Err(npe("duration")),
        _ => Err(cast_to(v, "java.time.Duration")),
    }
}

fn ms_to_inst(ms: i64) -> (i64, u32) {
    (ms.div_euclid(1000), (ms.rem_euclid(1000) * 1_000_000) as u32)
}

fn ok(v: Val) -> SResult<Option<Val>> {
    Ok(Some(v))
}
fn t(j: JTime) -> Val {
    Val::Time(Rc::new(j))
}

// ---------------------------------------------------------------------------
// Civil date / time arithmetic (OpenJDK algorithms)

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LDate {
    y: i32,
    m: u8,
    d: u8,
}

/// Local date-time: date plus nano-of-day.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LDT {
    date: LDate,
    nod: i64,
}

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}
fn month_len(m: u8, leap: bool) -> u8 {
    match m {
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}
fn first_day_of_year(m: u8, leap: bool) -> i64 {
    let l = if leap { 1 } else { 0 };
    match m {
        1 => 1,
        2 => 32,
        3 => 60 + l,
        4 => 91 + l,
        5 => 121 + l,
        6 => 152 + l,
        7 => 182 + l,
        8 => 213 + l,
        9 => 244 + l,
        10 => 274 + l,
        11 => 305 + l,
        _ => 335 + l,
    }
}

impl LDate {
    fn of(y: i64, m: i64, d: i64) -> SResult<LDate> {
        F::Year.check(y).map_err(|e| fr(e, &[LD_OF_292]))?;
        F::MonthOfYear.check(m).map_err(|e| fr(e, &[LD_OF_293]))?;
        F::DayOfMonth.check(d).map_err(|e| fr(e, &[LD_OF_294]))?;
        LDate::create(y as i32, m as u8, d as u8).map_err(|e| fr(e, &[LD_OF_295]))
    }
    fn create(y: i32, m: u8, d: u8) -> SResult<LDate> {
        if d > 28 && d > month_len(m, is_leap(y as i64)) {
            if d == 29 {
                return Err(frs(dte(format!("Invalid date 'February 29' as '{y}' is not a leap year")), &[LD_CREATE_477]));
            }
            return Err(frs(dte(format!("Invalid date '{} {}'", MONTHS[m as usize - 1], d)), &[LD_CREATE_479]));
        }
        Ok(LDate { y, m, d })
    }
    fn of_year_day(y: i64, doy: i64) -> SResult<LDate> {
        F::Year.check(y).map_err(|e| fr(e, &[LD_OF_YEAR_DAY_312]))?;
        F::DayOfYear.check(doy).map_err(|e| fr(e, &[LD_OF_YEAR_DAY_313]))?;
        let leap = is_leap(y);
        if doy == 366 && !leap {
            return Err(frs(dte(format!("Invalid date 'DayOfYear 366' as '{y}' is not a leap year")), &[LD_OF_YEAR_DAY_316]));
        }
        let mut m = ((doy - 1) / 31 + 1) as u8;
        let month_end = first_day_of_year(m, leap) + month_len(m, leap) as i64 - 1;
        if doy > month_end {
            m += 1;
        }
        let dom = doy - first_day_of_year(m, leap) + 1;
        Ok(LDate { y: y as i32, m, d: dom as u8 })
    }
    fn resolve_prev(y: i32, m: u8, d: u8) -> LDate {
        LDate { y, m, d: d.min(month_len(m, is_leap(y as i64))) }
    }
    fn from_epoch_day(ed: i64) -> SResult<LDate> {
        F::EpochDay.check(ed).map_err(|e| fr(e, &[LD_OF_EPOCH_DAY]))?;
        let (y, m, d) = civil_from_days(ed);
        Ok(LDate { y: y as i32, m, d })
    }
    fn epoch_day(&self) -> i64 {
        days_from_civil(self.y as i64, self.m, self.d)
    }
}

/// Epoch day of a proleptic date (OpenJDK `LocalDate.toEpochDay`, any year).
fn days_from_civil(y: i64, m: u8, d: u8) -> i64 {
    let m = m as i64;
    let mut total = 365 * y;
    if y >= 0 {
        total += (y + 3) / 4 - (y + 99) / 100 + (y + 399) / 400;
    } else {
        total -= y / -4 - y / -100 + y / -400;
    }
    total += (367 * m - 362) / 12;
    total += d as i64 - 1;
    if m > 2 {
        total -= 1;
        if !is_leap(y) {
            total -= 1;
        }
    }
    total - DAYS_0000_TO_1970
}

/// Proleptic (year, month, day) of an epoch day (OpenJDK `LocalDate.ofEpochDay`).
fn civil_from_days(ed: i64) -> (i64, u8, u8) {
    {
        let mut zero_day = ed + DAYS_0000_TO_1970 - 60;
        let mut adjust = 0;
        if zero_day < 0 {
            let adjust_cycles = (zero_day + 1) / 146_097 - 1;
            adjust = adjust_cycles * 400;
            zero_day += -adjust_cycles * 146_097;
        }
        let mut year_est = (400 * zero_day + 591) / 146_097;
        let mut doy_est = zero_day - (365 * year_est + year_est / 4 - year_est / 100 + year_est / 400);
        if doy_est < 0 {
            year_est -= 1;
            doy_est = zero_day - (365 * year_est + year_est / 4 - year_est / 100 + year_est / 400);
        }
        year_est += adjust;
        let march_doy0 = doy_est;
        let march_month0 = (march_doy0 * 5 + 2) / 153;
        let month = (march_month0 + 2) % 12 + 1;
        let dom = march_doy0 - (march_month0 * 306 + 5) / 10 + 1;
        year_est += march_month0 / 10;
        (year_est, month as u8, dom as u8)
    }
}

impl LDate {
    fn leap(&self) -> bool {
        is_leap(self.y as i64)
    }
    fn length_of_month(&self) -> u8 {
        month_len(self.m, self.leap())
    }
    fn length_of_year(&self) -> i64 {
        if self.leap() {
            366
        } else {
            365
        }
    }
    fn day_of_year(&self) -> i64 {
        first_day_of_year(self.m, self.leap()) + self.d as i64 - 1
    }
    fn day_of_week(&self) -> u8 {
        ((self.epoch_day() + 3).rem_euclid(7) + 1) as u8
    }
    fn proleptic_month(&self) -> i64 {
        self.y as i64 * 12 + self.m as i64 - 1
    }
    fn plus_days(&self, n: i64) -> SResult<LDate> {
        if n == 0 {
            return Ok(*self);
        }
        let ed = self.epoch_day().checked_add(n).ok_or_else(|| fr(overflow(), &[LD_PLUS_DAYS_1394]))?;
        LDate::from_epoch_day(ed).map_err(|e| fr(e, &[LD_PLUS_DAYS_1395]))
    }
    fn plus_weeks(&self, n: i64) -> SResult<LDate> {
        self.plus_days(n.checked_mul(7).ok_or_else(|| fr(overflow_mul(), &[jt!("LocalDate", "plusWeeks", 1355)]))?)
    }
    fn plus_months(&self, n: i64) -> SResult<LDate> {
        if n == 0 {
            return Ok(*self);
        }
        let calc = self.proleptic_month().checked_add(n).ok_or_else(overflow)?;
        let ny = F::Year.check_int(calc.div_euclid(12)).map_err(|e| fr(e, &[jt!("LocalDate", "plusMonths", 1334)]))?;
        let nm = (calc.rem_euclid(12) + 1) as u8;
        Ok(LDate::resolve_prev(ny as i32, nm, self.d))
    }
    fn plus_years(&self, n: i64) -> SResult<LDate> {
        if n == 0 {
            return Ok(*self);
        }
        let ny = F::Year.check_int((self.y as i64).saturating_add(n)).map_err(|e| fr(e, &[jt!("LocalDate", "plusYears", 1304)]))?;
        Ok(LDate::resolve_prev(ny as i32, self.m, self.d))
    }
    fn with_year(&self, y: i64) -> SResult<LDate> {
        F::Year.check(y).map_err(|e| fr(e, &[jt!("LocalDate", "withYear", 1089)]))?;
        Ok(LDate::resolve_prev(y as i32, self.m, self.d))
    }
    fn with_month(&self, m: i64) -> SResult<LDate> {
        F::MonthOfYear.check(m).map_err(|e| fr(e, &[jt!("LocalDate", "withMonth", 1108)]))?;
        Ok(LDate::resolve_prev(self.y, m as u8, self.d))
    }
    fn with_day(&self, d: i64) -> SResult<LDate> {
        if d == self.d as i64 {
            return Ok(*self);
        }
        LDate::of(self.y as i64, self.m as i64, d).map_err(|e| fr(e, &[jt!("LocalDate", "withDayOfMonth", 1128)]))
    }
    fn with_day_of_year(&self, doy: i64) -> SResult<LDate> {
        if doy == self.day_of_year() {
            return Ok(*self);
        }
        LDate::of_year_day(self.y as i64, doy).map_err(|e| fr(e, &[jt!("LocalDate", "withDayOfYear", 1147)]))
    }
    /// `LocalDate.until(end, MONTHS)`.
    fn months_until(&self, end: &LDate) -> i64 {
        let p1 = self.proleptic_month() * 32 + self.d as i64;
        let p2 = end.proleptic_month() * 32 + end.d as i64;
        (p2 - p1) / 32
    }
    fn until_unit(&self, end: &LDate, unit: Unit) -> SResult<i64> {
        Ok(match unit {
            U_DAYS => end.epoch_day() - self.epoch_day(),
            U_WEEKS => (end.epoch_day() - self.epoch_day()) / 7,
            U_MONTHS => self.months_until(end),
            U_YEARS => self.months_until(end) / 12,
            U_DECADES => self.months_until(end) / 120,
            U_CENTURIES => self.months_until(end) / 1200,
            U_MILLENNIA => self.months_until(end) / 12000,
            U_ERAS => era_of(end.y) - era_of(self.y),
            U_QUARTER_YEARS => self.months_until(end) / 3,
            U_WEEK_BASED_YEARS => iso_week_based_year(end) - iso_week_based_year(self),
            u => return Err(frs(unsupported_unit(u), &[LD_UNTIL_1656])),
        })
    }
    /// `LocalDate.until(end)` as a `Period`.
    fn period_until(&self, end: &LDate) -> SResult<(i32, i32, i32)> {
        let mut total_months = end.proleptic_month() - self.proleptic_month();
        let mut days = end.d as i64 - self.d as i64;
        if total_months > 0 && days < 0 {
            total_months -= 1;
            let calc = self.plus_months(total_months)?;
            days = end.epoch_day() - calc.epoch_day();
        } else if total_months < 0 && days > 0 {
            total_months += 1;
            days -= end.length_of_month() as i64;
        }
        let years = total_months / 12;
        let months = total_months % 12;
        Ok((i32::try_from(years).map_err(|_| arith("integer overflow"))?, months as i32, days as i32))
    }
    fn plus_unit(&self, n: i64, unit: Unit) -> SResult<LDate> {
        match unit {
            U_DAYS => self.plus_days(n),
            U_WEEKS => self.plus_weeks(n),
            U_MONTHS => self.plus_months(n),
            U_YEARS => self.plus_years(n),
            U_DECADES => self.plus_years(n.checked_mul(10).ok_or_else(overflow)?),
            U_CENTURIES => self.plus_years(n.checked_mul(100).ok_or_else(overflow)?),
            U_MILLENNIA => self.plus_years(n.checked_mul(1000).ok_or_else(overflow)?),
            U_ERAS => {
                let v = era_of(self.y).checked_add(n).ok_or_else(overflow)?;
                self.with_field(F::Era, v)
            }
            U_QUARTER_YEARS => self.plus_months(n.checked_mul(3).ok_or_else(overflow)?),
            U_WEEK_BASED_YEARS => {
                let wby = iso_week_based_year(self) + n;
                let week = iso_week_of_wby(self);
                let dow = self.day_of_week() as i64;
                // Java IsoFields.WEEK_BASED_YEARS.addTo
                let base = LDate::of(wby, 1, 4)?;
                let max = iso_week_range_max(wby);
                let week = week.min(max);
                let d = base.plus_days((week - 1) * 7 + (dow - base.day_of_week() as i64))?;
                Ok(d)
            }
            u => Err(frs(unsupported_unit(u), &[LD_PLUS_1273])),
        }
    }
    fn with_field(&self, f: F, v: i64) -> SResult<LDate> {
        let chk = |f: F, v: i64| f.check(v).map_err(|e| fr(e, &[LD_WITH_1052]));
        Ok(match f {
            F::DayOfWeek => {
                chk(f, v)?;
                self.plus_days(v - self.day_of_week() as i64)?
            }
            F::AlignedDayOfWeekInMonth => {
                chk(f, v)?;
                self.plus_days(v - self.get(F::AlignedDayOfWeekInMonth).unwrap_or(0))?
            }
            F::AlignedDayOfWeekInYear => {
                chk(f, v)?;
                self.plus_days(v - self.get(F::AlignedDayOfWeekInYear).unwrap_or(0))?
            }
            F::DayOfMonth => {
                chk(f, v)?;
                self.with_day(v)?
            }
            F::DayOfYear => {
                chk(f, v)?;
                self.with_day_of_year(v)?
            }
            F::EpochDay => LDate::from_epoch_day(v)?,
            F::AlignedWeekOfMonth => {
                chk(f, v)?;
                self.plus_weeks(v - self.get(F::AlignedWeekOfMonth).unwrap_or(0))?
            }
            F::AlignedWeekOfYear => {
                chk(f, v)?;
                self.plus_weeks(v - self.get(F::AlignedWeekOfYear).unwrap_or(0))?
            }
            F::MonthOfYear => {
                chk(f, v)?;
                self.with_month(v)?
            }
            F::ProlepticMonth => {
                chk(f, v)?;
                self.plus_months(v - self.proleptic_month())?
            }
            F::YearOfEra => {
                chk(f, v)?;
                self.with_year(if self.y >= 1 { v } else { 1 - v })?
            }
            F::Year => {
                chk(f, v)?;
                self.with_year(v)?
            }
            F::Era => {
                chk(f, v)?;
                if era_of(self.y) == v {
                    *self
                } else {
                    self.with_year(1 - self.y as i64)?
                }
            }
            F::QuarterOfYear => {
                f.check(v).map_err(|e| frs(e, &[VR_CHECK, jf!("java.time.temporal.IsoFields$Field$2", "adjustInto", "IsoFields", 421), LD_WITH_1070]))?;
                self.plus_months((v - self.get(F::QuarterOfYear).unwrap_or(0)) * 3)?
            }
            F::DayOfQuarter => {
                // Java IsoFields.DAY_OF_QUARTER.adjustInto (lenient within range)
                let cur = self.get(F::DayOfQuarter).unwrap_or(0);
                let range_max = quarter_len(self);
                if v < 1 || v > range_max {
                    return Err(frs(dte(format!("Invalid value for DayOfQuarter (valid values 1 - 90/92): {v}")), &[VR_CHECK, jf!("java.time.temporal.IsoFields$Field$1", "adjustInto", "IsoFields", 345), LD_WITH_1070]));
                }
                self.plus_days(v - cur)?
            }
            F::WeekOfWeekBasedYear => {
                f.check(v).map_err(|e| frs(e, &[VR_CHECK, jf!("java.time.temporal.IsoFields$Field$3", "adjustInto", "IsoFields", 475), LD_WITH_1070]))?;
                self.plus_weeks(v - iso_week_of_wby(self))?
            }
            F::WeekBasedYear => {
                chk(f, v)?;
                self.plus_unit(v - iso_week_based_year(self), U_WEEK_BASED_YEARS)?
            }
            _ => return Err(frs(unsupported_field(f), &[LD_WITH_1067])),
        })
    }
    fn get(&self, f: F) -> Option<i64> {
        Some(match f {
            F::DayOfWeek => self.day_of_week() as i64,
            F::AlignedDayOfWeekInMonth => ((self.d as i64 - 1) % 7) + 1,
            F::AlignedDayOfWeekInYear => ((self.day_of_year() - 1) % 7) + 1,
            F::DayOfMonth => self.d as i64,
            F::DayOfYear => self.day_of_year(),
            F::EpochDay => self.epoch_day(),
            F::AlignedWeekOfMonth => ((self.d as i64 - 1) / 7) + 1,
            F::AlignedWeekOfYear => ((self.day_of_year() - 1) / 7) + 1,
            F::MonthOfYear => self.m as i64,
            F::ProlepticMonth => self.proleptic_month(),
            F::YearOfEra => {
                if self.y >= 1 {
                    self.y as i64
                } else {
                    1 - self.y as i64
                }
            }
            F::Year => self.y as i64,
            F::Era => era_of(self.y),
            F::QuarterOfYear => ((self.m as i64 - 1) / 3) + 1,
            F::DayOfQuarter => {
                let q0 = (self.m as i64 - 1) / 3;
                let start = first_day_of_year((q0 * 3 + 1) as u8, self.leap());
                self.day_of_year() - start + 1
            }
            F::WeekOfWeekBasedYear => iso_week_of_wby(self),
            F::WeekBasedYear => iso_week_based_year(self),
            _ => return None,
        })
    }
    fn write(&self, out: &mut String) {
        push_year(out, self.y as i64);
        let _ = write!(out, "-{:02}-{:02}", self.m, self.d);
    }
    fn hash(&self) -> i32 {
        let y = self.y;
        (y & 0xFFFF_F800u32 as i32) ^ ((y << 11).wrapping_add((self.m as i32) << 6).wrapping_add(self.d as i32))
    }
}

fn era_of(y: i32) -> i64 {
    if y >= 1 {
        1
    } else {
        0
    }
}
fn quarter_len(d: &LDate) -> i64 {
    match (d.m - 1) / 3 {
        0 => {
            if d.leap() {
                91
            } else {
                90
            }
        }
        1 => 91,
        _ => 92,
    }
}

/// ISO week-based-year of a date (`IsoFields.WEEK_BASED_YEAR`).
fn iso_week_based_year(d: &LDate) -> i64 {
    let mut year = d.y as i64;
    let doy = d.day_of_year();
    let dow0 = d.day_of_week() as i64 - 1;
    if doy <= 3 {
        if doy - dow0 < -2 {
            year -= 1;
        }
    } else if doy >= 363 {
        let doy = doy - 363 - if d.leap() { 1 } else { 0 };
        if doy - dow0 >= 0 {
            year += 1;
        }
    }
    year
}
fn iso_week_range_max(wby: i64) -> i64 {
    match LDate::of(wby, 1, 4) {
        Ok(jan4) => {
            let jan1 = LDate { y: jan4.y, m: 1, d: 1 };
            let dow = jan1.day_of_week();
            if dow == 4 || (dow == 3 && jan1.leap()) {
                53
            } else {
                52
            }
        }
        Err(_) => 52,
    }
}
/// ISO week-of-week-based-year (`IsoFields.WEEK_OF_WEEK_BASED_YEAR`).
fn iso_week_of_wby(d: &LDate) -> i64 {
    let dow0 = d.day_of_week() as i64 - 1;
    let doy0 = d.day_of_year() - 1;
    let doy_thu0 = doy0 + (3 - dow0);
    let aligned_week = doy_thu0 / 7;
    let first_thu_doy0 = doy_thu0 - aligned_week * 7;
    let mut first_mon_doy0 = first_thu_doy0 - 3;
    if first_mon_doy0 < -3 {
        first_mon_doy0 += 7;
    }
    if doy0 < first_mon_doy0 {
        return iso_week_range_max(d.y as i64 - 1);
    }
    let mut week = (doy0 - first_mon_doy0) / 7 + 1;
    if week == 53 && !(first_mon_doy0 == -3 || (first_mon_doy0 == -2 && d.leap())) {
        week = 1;
    }
    week
}

/// `LocalDate.toString` year rendering (`0000`, `-0001`, `+10000`).
fn push_year(out: &mut String, y: i64) {
    if y.abs() < 1000 {
        if y < 0 {
            out.push('-');
        }
        let _ = write!(out, "{:04}", y.abs());
    } else {
        if y > 9999 {
            out.push('+');
        }
        let _ = write!(out, "{y}");
    }
}

/// `LocalTime.toString` (`10:11`, `10:11:12.005`).
fn push_time(out: &mut String, nod: i64) {
    let h = nod / (3600 * NPS);
    let m = (nod / (60 * NPS)) % 60;
    let s = (nod / NPS) % 60;
    let n = nod % NPS;
    let _ = write!(out, "{h:02}:{m:02}");
    if s > 0 || n > 0 {
        let _ = write!(out, ":{s:02}");
        if n > 0 {
            push_fraction(out, n);
        }
    }
}
/// Nanos as `.SSS`, `.SSSSSS` or `.SSSSSSSSS` (Java's grouping).
fn push_fraction(out: &mut String, n: i64) {
    if n % 1_000_000 == 0 {
        let _ = write!(out, ".{:03}", n / 1_000_000);
    } else if n % 1000 == 0 {
        let _ = write!(out, ".{:06}", n / 1000);
    } else {
        let _ = write!(out, ".{:09}", n);
    }
}

fn time_of(h: i64, m: i64, s: i64, n: i64) -> SResult<i64> {
    time_of_at(h, m, s, n, &[LT_OF_351, LT_OF_352, LT_OF_353, LT_OF_354])
}
/// `LocalTime.of` with the JDK line of each range check (hour, minute, second, nano).
fn time_of_at(h: i64, m: i64, s: i64, n: i64, lines: &[Fr; 4]) -> SResult<i64> {
    F::HourOfDay.check(h).map_err(|e| fr(e, &[lines[0]]))?;
    F::MinuteOfHour.check(m).map_err(|e| fr(e, &[lines[1]]))?;
    F::SecondOfMinute.check(s).map_err(|e| fr(e, &[lines[2]]))?;
    F::NanoOfSecond.check(n).map_err(|e| fr(e, &[lines[3]]))?;
    Ok(h * 3600 * NPS + m * 60 * NPS + s * NPS + n)
}
const LT_OF_2: [Fr; 4] = [LT_OF_308, LT_OF_312, LT_OF_312, LT_OF_312];
const LT_OF_3: [Fr; 4] = [LT_OF_329, LT_OF_333, LT_OF_334, LT_OF_334];
const LT_OF_4: [Fr; 4] = [LT_OF_351, LT_OF_352, LT_OF_353, LT_OF_354];
fn time_hash(nod: i64) -> i32 {
    (nod ^ (nod >> 32)) as i32
}
fn time_get(nod: i64, f: F) -> Option<i64> {
    let h = nod / (3600 * NPS);
    let m = (nod / (60 * NPS)) % 60;
    let s = (nod / NPS) % 60;
    let n = nod % NPS;
    Some(match f {
        F::NanoOfSecond => n,
        F::NanoOfDay => nod,
        F::MicroOfSecond => n / 1000,
        F::MicroOfDay => nod / 1000,
        F::MilliOfSecond => n / 1_000_000,
        F::MilliOfDay => nod / 1_000_000,
        F::SecondOfMinute => s,
        F::SecondOfDay => nod / NPS,
        F::MinuteOfHour => m,
        F::MinuteOfDay => h * 60 + m,
        F::HourOfAmPm => h % 12,
        F::ClockHourOfAmPm => {
            let ham = h % 12;
            if ham % 12 == 0 {
                12
            } else {
                ham
            }
        }
        F::HourOfDay => h,
        F::ClockHourOfDay => {
            if h == 0 {
                24
            } else {
                h
            }
        }
        F::AmPmOfDay => h / 12,
        _ => return None,
    })
}
fn time_with(nod: i64, f: F, v: i64) -> SResult<i64> {
    if time_get(nod, f).is_none() {
        return Err(frs(unsupported_field(f), &[LT_WITH_886]));
    }
    f.check(v).map_err(|e| fr(e, &[LT_WITH_869]))?;
    let h = nod / (3600 * NPS);
    let m = (nod / (60 * NPS)) % 60;
    let s = (nod / NPS) % 60;
    let n = nod % NPS;
    Ok(match f {
        F::NanoOfSecond => nod - n + v,
        F::NanoOfDay => v,
        F::MicroOfSecond => nod - n + v * 1000,
        F::MicroOfDay => v * 1000,
        F::MilliOfSecond => nod - n + v * 1_000_000,
        F::MilliOfDay => v * 1_000_000,
        F::SecondOfMinute => nod + (v - s) * NPS,
        F::SecondOfDay => nod + (v - nod / NPS) * NPS,
        F::MinuteOfHour => nod + (v - m) * 60 * NPS,
        F::MinuteOfDay => nod + (v - (h * 60 + m)) * 60 * NPS,
        F::HourOfAmPm => nod + (v - h % 12) * 3600 * NPS,
        F::ClockHourOfAmPm => nod + ((if v == 12 { 0 } else { v }) - h % 12) * 3600 * NPS,
        F::HourOfDay => nod + (v - h) * 3600 * NPS,
        F::ClockHourOfDay => nod + ((if v == 24 { 0 } else { v }) - h) * 3600 * NPS,
        F::AmPmOfDay => nod + (v - h / 12) * 12 * 3600 * NPS,
        _ => return Err(frs(unsupported_field(f), &[LT_WITH_886])),
    })
}
/// `LocalTime.plusNanos` (wraps around midnight).
fn time_plus(nod: i64, nanos: i128) -> i64 {
    ((nod as i128 + nanos).rem_euclid(NPD as i128)) as i64
}
fn time_plus_unit(nod: i64, n: i64, unit: Unit) -> SResult<i64> {
    Ok(match unit {
        U_NANOS => time_plus(nod, n as i128),
        U_MICROS => time_plus(nod, (n as i128 % 86_400_000_000) * 1000),
        U_MILLIS => time_plus(nod, (n as i128 % 86_400_000) * 1_000_000),
        U_SECONDS => time_plus(nod, (n as i128 % 86_400) * NPS as i128),
        U_MINUTES => time_plus(nod, (n as i128 % 1440) * 60 * NPS as i128),
        U_HOURS => time_plus(nod, (n as i128 % 24) * 3600 * NPS as i128),
        U_HALF_DAYS => time_plus(nod, (n as i128 % 2) * 12 * 3600 * NPS as i128),
        u => return Err(frs(unsupported_unit(u), &[LT_PLUS_1087])),
    })
}
fn time_until(a: i64, b: i64, unit: Unit) -> SResult<i64> {
    let d = b - a;
    Ok(match unit {
        U_NANOS => d,
        U_MICROS => d / 1000,
        U_MILLIS => d / 1_000_000,
        U_SECONDS => d / NPS,
        U_MINUTES => d / (60 * NPS),
        U_HOURS => d / (3600 * NPS),
        U_HALF_DAYS => d / (12 * 3600 * NPS),
        u => return Err(frs(unsupported_unit(u), &[LT_UNTIL_1429])),
    })
}
/// `truncatedTo` unit validation; returns the unit's length in nanos.
fn truncation_nanos(unit: Unit) -> SResult<i64> {
    if unit == U_NANOS {
        return Ok(1);
    }
    let (secs, nanos) = unit_duration(unit);
    if secs > SPD {
        return Err(frs(utte("Unit is too large to be used for truncation"), &[LT_TRUNC_988]));
    }
    let dur = secs * NPS + nanos as i64;
    if NPD % dur != 0 {
        return Err(frs(utte("Unit must divide into a standard day without remainder"), &[LT_TRUNC_988]));
    }
    Ok(dur)
}
fn time_truncate(nod: i64, unit: Unit) -> SResult<i64> {
    let dur = truncation_nanos(unit)?;
    Ok((nod / dur) * dur)
}

impl LDT {
    fn of(date: LDate, nod: i64) -> LDT {
        LDT { date, nod }
    }
    fn epoch_second(&self, off: i32) -> i64 {
        self.date.epoch_day() * SPD + self.nod / NPS - off as i64
    }
    fn nanos(&self) -> u32 {
        (self.nod % NPS) as u32
    }
    fn from_epoch(secs: i64, nanos: u32, off: i32) -> SResult<LDT> {
        let local = secs.checked_add(off as i64).ok_or_else(overflow)?;
        let date = LDate::from_epoch_day(local.div_euclid(SPD))?;
        Ok(LDT { date, nod: local.rem_euclid(SPD) * NPS + nanos as i64 })
    }
    /// Add a signed amount of `unit_nanos`-sized units (Java `plusWithOverflow`).
    fn plus_nanos(&self, n: i64, unit_nanos: i64) -> SResult<LDT> {
        if n == 0 {
            return Ok(*self);
        }
        let total = self.nod as i128 + n as i128 * unit_nanos as i128;
        let days = total.div_euclid(NPD as i128);
        let nod = total.rem_euclid(NPD as i128) as i64;
        let days = i64::try_from(days).map_err(|_| overflow())?;
        Ok(LDT { date: self.date.plus_days(days)?, nod })
    }
    fn plus_unit(&self, n: i64, unit: Unit) -> SResult<LDT> {
        Ok(match unit {
            U_NANOS => self.plus_nanos(n, 1)?,
            U_MICROS => self.plus_nanos(n, 1000)?,
            U_MILLIS => self.plus_nanos(n, 1_000_000)?,
            U_SECONDS => self.plus_nanos(n, NPS)?,
            U_MINUTES => self.plus_nanos(n, 60 * NPS)?,
            U_HOURS => self.plus_nanos(n, 3600 * NPS)?,
            U_HALF_DAYS => self.plus_nanos(n, 12 * 3600 * NPS)?,
            u => LDT { date: self.date.plus_unit(n, u).map_err(|e| fr(e, &[LDT_PLUS_1190]))?, nod: self.nod },
        })
    }
    fn until(&self, end: &LDT, unit: Unit) -> SResult<i64> {
        if unit_is_time_based(unit) {
            let mut amount = end.date.epoch_day() - self.date.epoch_day();
            if amount == 0 {
                return time_until(self.nod, end.nod, unit);
            }
            let mut time_part = end.nod as i128 - self.nod as i128;
            if amount > 0 {
                amount -= 1;
                time_part += NPD as i128;
            } else {
                amount += 1;
                time_part -= NPD as i128;
            }
            let (mul, div): (i128, i128) = match unit {
                U_NANOS => (NPD as i128, 1),
                U_MICROS => (86_400_000_000, 1000),
                U_MILLIS => (86_400_000, 1_000_000),
                U_SECONDS => (SPD as i128, NPS as i128),
                U_MINUTES => (1440, 60 * NPS as i128),
                U_HOURS => (24, 3600 * NPS as i128),
                _ => (2, 12 * 3600 * NPS as i128),
            };
            let r = amount as i128 * mul + time_part / div;
            return i64::try_from(r).map_err(|_| overflow());
        }
        let mut end_date = end.date;
        if end_date > self.date && end.nod < self.nod {
            end_date = end_date.plus_days(-1)?;
        } else if end_date < self.date && end.nod > self.nod {
            end_date = end_date.plus_days(1)?;
        }
        self.date.until_unit(&end_date, unit).map_err(|e| fr(e, &[LDT_UNTIL_1731]))
    }
    fn truncated(&self, unit: Unit) -> SResult<LDT> {
        Ok(LDT { date: self.date, nod: time_truncate(self.nod, unit).map_err(|e| fr(e, &[LDT_TRUNC_1120]))? })
    }
    fn with_field(&self, f: F, v: i64) -> SResult<LDT> {
        if f.is_time_based() {
            Ok(LDT { date: self.date, nod: time_with(self.nod, f, v).map_err(|e| fr(e, &[LDT_WITH_967]))? })
        } else {
            Ok(LDT { date: self.date.with_field(f, v).map_err(|e| fr(e, &[LDT_WITH_969]))?, nod: self.nod })
        }
    }
    fn write(&self, out: &mut String) {
        self.date.write(out);
        out.push('T');
        push_time(out, self.nod);
    }
    fn hash(&self) -> i32 {
        self.date.hash() ^ time_hash(self.nod)
    }
}

// ---------------------------------------------------------------------------
// Instants and durations

fn inst_create(secs: i64, nano_adj: i64) -> SResult<(i64, u32)> {
    let s = secs.checked_add(nano_adj.div_euclid(NPS)).ok_or_else(overflow)?;
    if !(INSTANT_MIN_SEC..=INSTANT_MAX_SEC).contains(&s) {
        return Err(frs(dte("Instant exceeds minimum or maximum instant"), &[INST_CREATE]));
    }
    Ok((s, nano_adj.rem_euclid(NPS) as u32))
}
fn inst_plus(i: (i64, u32), secs: i64, nanos: i64) -> SResult<(i64, u32)> {
    if secs == 0 && nanos == 0 {
        return Ok(i);
    }
    let es = i.0.checked_add(secs).ok_or_else(overflow)?;
    let es = es.checked_add(nanos / NPS).ok_or_else(overflow)?;
    let nos = i.1 as i64 + nanos % NPS;
    inst_create(es, nos).map_err(|e| fr(e, &[INST_PLUS_SECONDS]))
}
fn inst_plus_unit(i: (i64, u32), n: i64, unit: Unit) -> SResult<(i64, u32)> {
    match unit {
        U_NANOS => inst_plus(i, 0, n),
        U_MICROS => inst_plus(i, n / 1_000_000, (n % 1_000_000) * 1000),
        U_MILLIS => inst_plus(i, n / 1000, (n % 1000) * 1_000_000),
        U_SECONDS => inst_plus(i, n, 0),
        U_MINUTES => inst_plus(i, n.checked_mul(60).ok_or_else(overflow)?, 0),
        U_HOURS => inst_plus(i, n.checked_mul(3600).ok_or_else(overflow)?, 0),
        U_HALF_DAYS => inst_plus(i, n.checked_mul(43_200).ok_or_else(overflow)?, 0),
        U_DAYS => inst_plus(i, n.checked_mul(SPD).ok_or_else(overflow)?, 0),
        u => Err(frs(unsupported_unit(u), &[INST_PLUS_887])),
    }
    .map_err(|e| if e.frames.len() == 2 && e.frames[1] == INST_PLUS_SECONDS { fr(e, &[INST_PLUS_882]) } else { e })
}
fn inst_until(a: (i64, u32), b: (i64, u32), unit: Unit) -> SResult<i64> {
    let nanos_until = || -> SResult<i64> {
        let secs = b.0.checked_sub(a.0).ok_or_else(overflow)?;
        let total = secs.checked_mul(NPS).ok_or_else(overflow)?;
        total.checked_add(b.1 as i64 - a.1 as i64).ok_or_else(overflow)
    };
    let secs_until = || -> i64 {
        let mut secs = b.0 - a.0;
        let nanos = b.1 as i64 - a.1 as i64;
        if secs > 0 && nanos < 0 {
            secs -= 1;
        } else if secs < 0 && nanos > 0 {
            secs += 1;
        }
        secs
    };
    Ok(match unit {
        U_NANOS => nanos_until()?,
        U_MICROS => nanos_until()? / 1000,
        U_MILLIS => {
            let ms = b.0.checked_sub(a.0).ok_or_else(overflow)?.checked_mul(1000).ok_or_else(overflow)?;
            ms + (b.1 as i64 / 1_000_000 - a.1 as i64 / 1_000_000)
        }
        U_SECONDS => secs_until(),
        U_MINUTES => secs_until() / 60,
        U_HOURS => secs_until() / 3600,
        U_HALF_DAYS => secs_until() / 43_200,
        U_DAYS => secs_until() / SPD,
        u => return Err(frs(unsupported_unit(u), &[INST_UNTIL_1185])),
    })
}
fn inst_truncate(i: (i64, u32), unit: Unit) -> SResult<(i64, u32)> {
    if unit == U_NANOS {
        return Ok(i);
    }
    let dur = truncation_nanos(unit).map_err(|e| frs(e, &[INST_TRUNC_754]))?;
    let nod = i.0.rem_euclid(SPD) * NPS + i.1 as i64;
    let result = (nod / dur) * dur;
    inst_plus(i, 0, result - nod)
}
fn inst_write(out: &mut String, i: (i64, u32)) {
    let (y, mo, d) = civil_from_days(i.0.div_euclid(SPD));
    let sod = i.0.rem_euclid(SPD);
    push_year(out, y);
    let _ = write!(out, "-{mo:02}-{d:02}T{:02}:{:02}:{:02}", sod / 3600, (sod / 60) % 60, sod % 60);
    if i.1 != 0 {
        push_fraction(out, i.1 as i64);
    }
    out.push('Z');
}
fn inst_hash(i: (i64, u32)) -> i32 {
    ((i.0 ^ (i.0 >> 32)) as i32).wrapping_add(51i32.wrapping_mul(i.1 as i32))
}
/// Epoch seconds printable with a 4-digit year (`0000`..`9999`).
fn four_digit_year(secs: i64) -> bool {
    (-62_167_219_200..=253_402_300_799).contains(&secs)
}
fn inst_to_val(i: (i64, u32)) -> Val {
    if i.1 % 1_000_000 == 0 && four_digit_year(i.0) {
        if let Some(ms) = i.0.checked_mul(1000).and_then(|m| m.checked_add(i.1 as i64 / 1_000_000)) {
            return Val::Instant(ms);
        }
    }
    t(JTime::Instant(i.0, i.1))
}

/// `Duration.ofSeconds(secs, nanoAdj)`.
fn dur_create(secs: i64, nano_adj: i64) -> SResult<(i64, u32)> {
    let s = secs.checked_add(nano_adj.div_euclid(NPS)).ok_or_else(overflow)?;
    Ok((s, nano_adj.rem_euclid(NPS) as u32))
}
fn dur_plus(d: (i64, u32), secs: i64, nanos: i64) -> SResult<(i64, u32)> {
    if secs == 0 && nanos == 0 {
        return Ok(d);
    }
    let es = d.0.checked_add(secs).ok_or_else(|| fr(overflow(), &[DUR_PLUS_878]))?;
    let es = es.checked_add(nanos / NPS).ok_or_else(|| fr(overflow(), &[DUR_PLUS_878]))?;
    dur_create(es, d.1 as i64 + nanos % NPS).map_err(|e| fr(e, &[DUR_PLUS_878]))
}
fn dur_total_nanos(d: (i64, u32)) -> i128 {
    d.0 as i128 * NPS as i128 + d.1 as i128
}
fn dur_from_nanos(n: i128) -> SResult<(i64, u32)> {
    let secs = n.div_euclid(NPS as i128);
    let nanos = n.rem_euclid(NPS as i128);
    let secs = i64::try_from(secs).map_err(|_| frs(arith(format!("Exceeds capacity of Duration: {n}")), &[DUR_CREATE_1087]))?;
    Ok((secs, nanos as u32))
}
fn dur_plus_unit(d: (i64, u32), n: i64, unit: Unit) -> SResult<(i64, u32)> {
    if unit == U_DAYS {
        return dur_plus(d, n.checked_mul(SPD).ok_or_else(overflow)?, 0);
    }
    if unit_is_estimated(unit) {
        return Err(frs(utte("Unit must not have an estimated duration"), &[DUR_PLUS_764]));
    }
    if n == 0 {
        return Ok(d);
    }
    let (us, un) = unit_duration(unit);
    if un == 0 {
        return dur_plus(d, n.checked_mul(us).ok_or_else(overflow)?, 0).map_err(|e| fr(e, &[DUR_PLUS_SECONDS_835, jt!("Duration", "plus", 774)]));
    }
    let total = (us as i128 * NPS as i128 + un as i128) * n as i128;
    let add = dur_from_nanos(total)?;
    dur_plus(d, add.0, add.1 as i64)
}
fn dur_write(out: &mut String, d: (i64, u32)) {
    let (seconds, nanos) = (d.0, d.1 as i64);
    if seconds == 0 && nanos == 0 {
        out.push_str("PT0S");
        return;
    }
    let eff = if seconds < 0 && nanos > 0 { seconds + 1 } else { seconds };
    let hours = eff / 3600;
    let minutes = (eff % 3600) / 60;
    let secs = eff % 60;
    out.push_str("PT");
    if hours != 0 {
        let _ = write!(out, "{hours}H");
    }
    if minutes != 0 {
        let _ = write!(out, "{minutes}M");
    }
    if secs == 0 && nanos == 0 && out.len() > 2 {
        return;
    }
    if seconds < 0 && nanos > 0 {
        if secs == 0 {
            out.push_str("-0");
        } else {
            let _ = write!(out, "{secs}");
        }
    } else {
        let _ = write!(out, "{secs}");
    }
    if nanos > 0 {
        let frac = if seconds < 0 { 2 * NPS - nanos } else { nanos + NPS };
        let s = frac.to_string();
        let digits = s[1..].trim_end_matches('0');
        out.push('.');
        out.push_str(digits);
    }
    out.push('S');
}
fn dur_hash(d: (i64, u32)) -> i32 {
    ((d.0 ^ (d.0 >> 32)) as i32).wrapping_add(51i32.wrapping_mul(d.1 as i32))
}

fn period_write(out: &mut String, p: (i32, i32, i32)) {
    if p == (0, 0, 0) {
        out.push_str("P0D");
        return;
    }
    out.push('P');
    if p.0 != 0 {
        let _ = write!(out, "{}Y", p.0);
    }
    if p.1 != 0 {
        let _ = write!(out, "{}M", p.1);
    }
    if p.2 != 0 {
        let _ = write!(out, "{}D", p.2);
    }
}
fn period_total_months(p: (i32, i32, i32)) -> i64 {
    p.0 as i64 * 12 + p.1 as i64
}

// ---------------------------------------------------------------------------
// Zones

#[derive(Clone, Debug)]
pub enum Zone {
    /// `ZoneOffset` (total seconds).
    Offset(i32),
    /// Fixed-offset `ZoneRegion`: `UTC`, `GMT`, `UT`, `UTC+02:00`, ...
    Fixed(String, i32),
    /// IANA region.
    Region(chrono_tz::Tz),
}

fn offset_id(secs: i32) -> String {
    if secs == 0 {
        return "Z".into();
    }
    let a = secs.abs();
    let mut s = format!("{}{:02}:{:02}", if secs < 0 { '-' } else { '+' }, a / 3600, (a % 3600) / 60);
    if a % 60 != 0 {
        let _ = write!(s, ":{:02}", a % 60);
    }
    s
}

fn parse_offset_number(id: &str, pos: usize, colon: bool) -> SResult<i32> {
    let b = id.as_bytes();
    if colon && b.get(pos - 1) != Some(&b':') {
        return Err(frs(dte(format!("Invalid ID for ZoneOffset, colon not found when expected: {id}")), &[jt!("ZoneOffset", "parseNumber", 285), ZOFF_OF_240]));
    }
    let c1 = b.get(pos).copied().unwrap_or(0);
    let c2 = b.get(pos + 1).copied().unwrap_or(0);
    if !c1.is_ascii_digit() || !c2.is_ascii_digit() {
        return Err(frs(dte(format!("Invalid ID for ZoneOffset, non numeric characters found: {id}")), &[ZOFF_PARSE_NUMBER, ZOFF_OF_240]));
    }
    Ok(((c1 - b'0') as i32) * 10 + (c2 - b'0') as i32)
}

/// `ZoneOffset.of(id)`.
fn parse_offset_id(id: &str) -> SResult<i32> {
    if id == "Z" {
        return Ok(0);
    }
    let owned;
    let id2: &str = if id.chars().count() == 2 {
        let mut c = id.chars();
        let (a, b) = (c.next().unwrap_or(' '), c.next().unwrap_or(' '));
        owned = format!("{a}0{b}");
        &owned
    } else {
        id
    };
    let (h, m, s) = match id2.len() {
        3 => (parse_offset_number(id2, 1, false)?, 0, 0),
        5 => (parse_offset_number(id2, 1, false)?, parse_offset_number(id2, 3, false)?, 0),
        6 => (parse_offset_number(id2, 1, false)?, parse_offset_number(id2, 4, true)?, 0),
        7 => (parse_offset_number(id2, 1, false)?, parse_offset_number(id2, 3, false)?, parse_offset_number(id2, 5, false)?),
        9 => (parse_offset_number(id2, 1, false)?, parse_offset_number(id2, 4, true)?, parse_offset_number(id2, 7, true)?),
        _ => return Err(frs(dte(format!("Invalid ID for ZoneOffset, invalid format: {id}")), &[ZOFF_OF_260])),
    };
    match id2.as_bytes()[0] {
        b'+' => offset_of_hms(h, m, s).map_err(|e| fr(e, &[ZOFF_OF_269])),
        b'-' => offset_of_hms(-h, -m, -s).map_err(|e| fr(e, &[ZOFF_OF_269])),
        _ => Err(frs(dte(format!("Invalid ID for ZoneOffset, plus/minus not found when expected: {id}")), &[ZOFF_OF_264])),
    }
}

fn offset_of_hms(h: i32, m: i32, s: i32) -> SResult<i32> {
    let v = |line: Fr, msg: String| frs(dte(msg), &[line, ZOFF_OF_HMS_336]);
    if !(-18..=18).contains(&h) {
        return Err(v(ZOFF_VALIDATE_383, format!("Zone offset hours not in valid range: value {h} is not in the range -18 to 18")));
    }
    if h > 0 {
        if m < 0 || s < 0 {
            return Err(v(ZOFF_VALIDATE_388, "Zone offset minutes and seconds must be positive because hours is positive".into()));
        }
    } else if h < 0 {
        if m > 0 || s > 0 {
            return Err(v(jt!("ZoneOffset", "validate", 391), "Zone offset minutes and seconds must be negative because hours is negative".into()));
        }
    } else if (m > 0 && s < 0) || (m < 0 && s > 0) {
        return Err(v(jt!("ZoneOffset", "validate", 394), "Zone offset minutes and seconds must have the same sign".into()));
    }
    if !(-59..=59).contains(&m) {
        return Err(v(ZOFF_VALIDATE_398, format!("Zone offset minutes not in valid range: value {m} is not in the range -59 to 59")));
    }
    if !(-59..=59).contains(&s) {
        return Err(v(jt!("ZoneOffset", "validate", 401), format!("Zone offset seconds not in valid range: value {s} is not in the range -59 to 59")));
    }
    if h.abs() == 18 && (m != 0 || s != 0) {
        return Err(v(jt!("ZoneOffset", "validate", 404), "Zone offset not in valid range: -18:00 to +18:00".into()));
    }
    Ok(h * 3600 + m * 60 + s)
}
fn offset_of_total(secs: i64) -> SResult<i32> {
    if secs.abs() > 18 * 3600 {
        return Err(frs(dte("Zone offset not in valid range: -18:00 to +18:00"), &[jt!("ZoneOffset", "ofTotalSeconds", 434)]));
    }
    Ok(secs as i32)
}

const TZ_CLAMP: i64 = 8_000_000_000_000;
fn tz_naive(secs: i64) -> chrono::NaiveDateTime {
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs.clamp(-TZ_CLAMP, TZ_CLAMP), 0).map(|d| d.naive_utc()).unwrap_or_default()
}

impl Zone {
    /// `ZoneId.of(id)`.
    fn parse(id: &str) -> SResult<Zone> {
        if id == "Z" {
            return Ok(Zone::Offset(0));
        }
        if id.chars().count() <= 1 || id.starts_with('+') || id.starts_with('-') {
            return Ok(Zone::Offset(parse_offset_id(id).map_err(|e| fr(e, &[ZID_OF_405, ZID_OF_359]))?));
        }
        if matches!(id, "UTC" | "GMT" | "UT") {
            return Ok(Zone::Fixed(id.to_string(), 0));
        }
        for prefix in ["UTC+", "UTC-", "GMT+", "GMT-", "UT+", "UT-"] {
            if id.starts_with(prefix) {
                let off = parse_offset_id(&id[prefix.len() - 1..])
                    .map_err(|_| frs(dte(format!("Invalid ID for offset-based ZoneId: {id}")), &[jt!("ZoneId", "ofWithPrefix", 437), jt!("ZoneId", "of", 407), ZID_OF_359]))?;
                return Ok(Zone::of_offset(&prefix[..prefix.len() - 1], off));
            }
        }
        Zone::region(id).map_err(|e| fr(e, &[ZID_OF_411, ZID_OF_359]))
    }
    fn of_offset(prefix: &str, off: i32) -> Zone {
        if off == 0 {
            if prefix.is_empty() {
                return Zone::Offset(0);
            }
            return Zone::Fixed(prefix.to_string(), 0);
        }
        if prefix.is_empty() {
            return Zone::Offset(off);
        }
        Zone::Fixed(format!("{prefix}{}", offset_id(off)), off)
    }
    fn region(id: &str) -> SResult<Zone> {
        let valid = id.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '~' | '/' | '.' | '_' | '+' | '-'));
        if !valid {
            return Err(frs(dte(format!("Invalid ID for region-based ZoneId, invalid format: {id}")), &[jt!("ZoneRegion", "checkName", 152), jt!("ZoneRegion", "ofId", 117)]));
        }
        match id.parse::<chrono_tz::Tz>() {
            Ok(tz) if tz.name() == id => Ok(Zone::Region(tz)),
            _ => Err(frs(
                ScriptErr::new("zone_rules_exception", format!("Unknown time-zone ID: {id}")),
                &[jf!("java.time.zone.ZoneRulesProvider", "getProvider", "ZoneRulesProvider", 272), jf!("java.time.zone.ZoneRulesProvider", "getRules", "ZoneRulesProvider", 227), jt!("ZoneRegion", "ofId", 121)],
            )),
        }
    }
    fn id(&self) -> String {
        match self {
            Zone::Offset(o) => offset_id(*o),
            Zone::Fixed(id, _) => id.clone(),
            Zone::Region(tz) => tz.name().to_string(),
        }
    }
    fn is_offset(&self) -> bool {
        matches!(self, Zone::Offset(_))
    }
    fn fixed_offset(&self) -> Option<i32> {
        match self {
            Zone::Offset(o) | Zone::Fixed(_, o) => Some(*o),
            Zone::Region(_) => None,
        }
    }
    /// `ZoneId.normalized()`.
    fn normalized(&self) -> Zone {
        match self {
            Zone::Fixed(_, o) => Zone::Offset(*o),
            z => z.clone(),
        }
    }
    fn same(&self, o: &Zone) -> bool {
        match (self, o) {
            (Zone::Offset(a), Zone::Offset(b)) => a == b,
            (Zone::Offset(_), _) | (_, Zone::Offset(_)) => false,
            (a, b) => a.id() == b.id(),
        }
    }
    fn hash(&self) -> i32 {
        match self {
            Zone::Offset(o) => *o,
            z => str_hash(&z.id()),
        }
    }
    fn offset_at(&self, secs: i64) -> i32 {
        match self {
            Zone::Offset(o) | Zone::Fixed(_, o) => *o,
            Zone::Region(tz) => tz.offset_from_utc_datetime(&tz_naive(secs)).fix().local_minus_utc(),
        }
    }
    fn standard_offset_at(&self, secs: i64) -> i32 {
        match self {
            Zone::Offset(o) | Zone::Fixed(_, o) => *o,
            Zone::Region(tz) => tz.offset_from_utc_datetime(&tz_naive(secs)).base_utc_offset().num_seconds() as i32,
        }
    }
    fn is_dst_at(&self, secs: i64) -> bool {
        match self {
            Zone::Region(tz) => tz.offset_from_utc_datetime(&tz_naive(secs)).dst_offset().num_seconds() != 0,
            _ => false,
        }
    }
    /// Zone abbreviation at an instant (`EST`, `CEST`, ...).
    fn abbrev_at(&self, secs: i64) -> String {
        match self {
            Zone::Offset(o) => offset_id(*o),
            Zone::Fixed(id, _) => id.clone(),
            Zone::Region(tz) => {
                let off = tz.offset_from_utc_datetime(&tz_naive(secs));
                match off.abbreviation() {
                    Some(a) if !a.starts_with(['+', '-']) => a.to_string(),
                    _ => format!("GMT{}", loc_offset(off.fix().local_minus_utc(), true)),
                }
            }
        }
    }
    /// Valid offsets for a local date-time: `(offsets, gap)` where `gap`
    /// holds `(before, after)` when the local time falls in a gap.
    fn valid_offsets(&self, local_secs: i64) -> (Vec<i32>, Option<(i32, i32)>) {
        match self {
            Zone::Offset(o) | Zone::Fixed(_, o) => (vec![*o], None),
            Zone::Region(tz) => match tz.from_local_datetime(&tz_naive(local_secs)) {
                chrono::LocalResult::Single(d) => (vec![d.offset().fix().local_minus_utc()], None),
                chrono::LocalResult::Ambiguous(a, b) => (vec![a.offset().fix().local_minus_utc(), b.offset().fix().local_minus_utc()], None),
                chrono::LocalResult::None => {
                    let before = self.offset_at(local_secs - SPD);
                    let after = self.offset_at(local_secs + SPD);
                    (Vec::new(), Some((before, after)))
                }
            },
        }
    }
    /// `ZonedDateTime.ofLocal`: resolve a local date-time to an offset.
    fn resolve_local(&self, ldt: LDT, preferred: Option<i32>) -> SResult<(LDT, i32)> {
        let local_secs = ldt.epoch_second(0);
        let (offs, gap) = self.valid_offsets(local_secs);
        if offs.len() == 1 {
            return Ok((ldt, offs[0]));
        }
        if let Some((before, after)) = gap {
            let shifted = ldt.plus_nanos((after - before) as i64, NPS)?;
            return Ok((shifted, after));
        }
        if let Some(p) = preferred {
            if offs.contains(&p) {
                return Ok((ldt, p));
            }
        }
        Ok((ldt, offs[0]))
    }
    fn display_name(&self, full: bool) -> String {
        match self {
            Zone::Offset(o) => offset_id(*o),
            Zone::Fixed(id, _) => {
                if full {
                    match id.as_str() {
                        "UTC" => "Coordinated Universal Time".into(),
                        "GMT" => "Greenwich Mean Time".into(),
                        _ => id.clone(),
                    }
                } else {
                    id.clone()
                }
            }
            Zone::Region(tz) => {
                let id = tz.name();
                if let Some((_, long, short)) = ZONE_GENERIC.iter().find(|z| z.0 == id) {
                    return if full { long.to_string() } else { short.to_string() };
                }
                if full {
                    let city = id.rsplit('/').next().unwrap_or(id).replace('_', " ");
                    format!("{city} Time")
                } else {
                    let off = self.offset_at(crate::date::now_millis() / 1000);
                    let abbr = self.abbrev_at(crate::date::now_millis() / 1000);
                    if abbr.starts_with("GMT") || abbr.starts_with(['+', '-']) {
                        format!("GMT{}", loc_offset(off, true))
                    } else {
                        abbr
                    }
                }
            }
        }
    }
}

/// Generic zone display names (`ZoneId.getDisplayName`, pattern `v`).
const ZONE_GENERIC: &[(&str, &str, &str)] = &[
    ("America/New_York", "Eastern Time", "ET"),
    ("America/Chicago", "Central Time", "CT"),
    ("America/Denver", "Mountain Time", "MT"),
    ("America/Phoenix", "Mountain Time", "MT"),
    ("America/Los_Angeles", "Pacific Time", "PT"),
    ("America/Anchorage", "Alaska Time", "AKT"),
    ("Pacific/Honolulu", "Hawaii-Aleutian Time", "HST"),
    ("America/Toronto", "Eastern Time", "ET"),
    ("America/Vancouver", "Pacific Time", "PT"),
    ("Europe/London", "London Time", "GMT"),
    ("Europe/Paris", "Central European Time", "CET"),
    ("Europe/Berlin", "Central European Time", "CET"),
    ("Europe/Madrid", "Central European Time", "CET"),
    ("Europe/Rome", "Central European Time", "CET"),
    ("Europe/Amsterdam", "Central European Time", "CET"),
    ("Europe/Moscow", "Moscow Time", "GMT+03:00"),
    ("Asia/Tokyo", "Japan Time", "JST"),
    ("Asia/Shanghai", "China Time", "CST"),
    ("Asia/Kolkata", "Kolkata Time", "IST"),
    ("Australia/Sydney", "Australian Eastern Time", "AET"),
    ("Australia/Melbourne", "Australian Eastern Time", "AET"),
];

/// Full zone names by abbreviation (pattern `zzzz`).
const ZONE_LONG: &[(&str, &str)] = &[
    ("EST", "Eastern Standard Time"),
    ("EDT", "Eastern Daylight Time"),
    ("CST", "Central Standard Time"),
    ("CDT", "Central Daylight Time"),
    ("MST", "Mountain Standard Time"),
    ("MDT", "Mountain Daylight Time"),
    ("PST", "Pacific Standard Time"),
    ("PDT", "Pacific Daylight Time"),
    ("AKST", "Alaska Standard Time"),
    ("AKDT", "Alaska Daylight Time"),
    ("HST", "Hawaii-Aleutian Standard Time"),
    ("GMT", "Greenwich Mean Time"),
    ("BST", "British Summer Time"),
    ("IST", "India Standard Time"),
    ("CET", "Central European Standard Time"),
    ("CEST", "Central European Summer Time"),
    ("EET", "Eastern European Standard Time"),
    ("EEST", "Eastern European Summer Time"),
    ("WET", "Western European Standard Time"),
    ("WEST", "Western European Summer Time"),
    ("MSK", "Moscow Standard Time"),
    ("JST", "Japan Standard Time"),
    ("KST", "Korean Standard Time"),
    ("AEST", "Australian Eastern Standard Time"),
    ("AEDT", "Australian Eastern Daylight Time"),
    ("ACST", "Australian Central Standard Time"),
    ("ACDT", "Australian Central Daylight Time"),
    ("AWST", "Australian Western Standard Time"),
    ("NZST", "New Zealand Standard Time"),
    ("NZDT", "New Zealand Daylight Time"),
    ("SAST", "South Africa Standard Time"),
    ("HKT", "Hong Kong Standard Time"),
    ("WIB", "Western Indonesia Time"),
    ("PKT", "Pakistan Standard Time"),
    ("UTC", "Coordinated Universal Time"),
];
/// Zones whose `zzzz` name isn't derivable from the abbreviation.
const ZONE_LONG_BY_ID: &[(&str, &str, &str)] = &[
    ("Asia/Shanghai", "China Standard Time", "China Standard Time"),
    ("Asia/Dubai", "Gulf Standard Time", "Gulf Standard Time"),
    ("Asia/Kolkata", "India Standard Time", "India Standard Time"),
    ("Asia/Singapore", "Singapore Standard Time", "Singapore Standard Time"),
    ("Asia/Riyadh", "Arabian Standard Time", "Arabian Standard Time"),
    ("America/Sao_Paulo", "Brasilia Standard Time", "Brasilia Summer Time"),
    ("America/Argentina/Buenos_Aires", "Argentina Standard Time", "Argentina Summer Time"),
    ("Europe/Istanbul", "GMT+03:00", "GMT+03:00"),
];

fn zone_text(z: &Zone, secs: i64, full: bool) -> String {
    match z {
        Zone::Offset(o) => offset_id(*o),
        Zone::Fixed(id, _) => {
            if full {
                z.display_name(true)
            } else {
                id.clone()
            }
        }
        Zone::Region(tz) => {
            let abbr = z.abbrev_at(secs);
            if !full {
                return abbr;
            }
            if let Some((_, std, dst)) = ZONE_LONG_BY_ID.iter().find(|e| e.0 == tz.name()) {
                return if z.is_dst_at(secs) { dst.to_string() } else { std.to_string() };
            }
            ZONE_LONG.iter().find(|e| e.0 == abbr).map(|e| e.1.to_string()).unwrap_or(abbr)
        }
    }
}

/// Localized offset (`GMT+5:30` short, `GMT+05:30` full) without the `GMT`.
fn loc_offset(secs: i32, full: bool) -> String {
    if secs == 0 {
        return String::new();
    }
    let a = secs.abs();
    let sign = if secs < 0 { '-' } else { '+' };
    let (h, m, s) = (a / 3600, (a % 3600) / 60, a % 60);
    if full {
        let mut out = format!("{sign}{h:02}:{m:02}");
        if s != 0 {
            let _ = write!(out, ":{s:02}");
        }
        out
    } else {
        let mut out = format!("{sign}{h}");
        if m != 0 || s != 0 {
            let _ = write!(out, ":{m:02}");
            if s != 0 {
                let _ = write!(out, ":{s:02}");
            }
        }
        out
    }
}

fn str_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// `ZonedDateTime`.
#[derive(Clone, Debug)]
pub struct Zoned {
    ldt: LDT,
    off: i32,
    zone: Zone,
}

impl Zoned {
    fn of_instant(i: (i64, u32), zone: Zone) -> SResult<Zoned> {
        let off = zone.offset_at(i.0);
        Ok(Zoned { ldt: LDT::from_epoch(i.0, i.1, off)?, off, zone })
    }
    fn of_local(ldt: LDT, zone: Zone, preferred: Option<i32>) -> SResult<Zoned> {
        let (ldt, off) = zone.resolve_local(ldt, preferred)?;
        Ok(Zoned { ldt, off, zone })
    }
    fn of_strict(ldt: LDT, off: i32, zone: Zone) -> SResult<Zoned> {
        let (offs, gap) = zone.valid_offsets(ldt.epoch_second(0));
        if offs.contains(&off) {
            return Ok(Zoned { ldt, off, zone });
        }
        if gap.is_some() {
            let mut s = String::new();
            ldt.write(&mut s);
            return Err(frs(dte(format!("LocalDateTime '{s}' does not exist in zone '{}' due to a gap in the local time-line, typically caused by daylight savings", zone.id())), &[jt!("ZonedDateTime", "ofStrict", 489)]));
        }
        let mut s = String::new();
        ldt.write(&mut s);
        Err(frs(dte(format!("ZoneOffset '{}' is not valid for LocalDateTime '{s}' in zone '{}'", offset_id(off), zone.id())), &[jt!("ZonedDateTime", "ofStrict", 493)]))
    }
    fn instant(&self) -> (i64, u32) {
        (self.ldt.epoch_second(self.off), self.ldt.nanos())
    }
    fn epoch_second(&self) -> i64 {
        self.ldt.epoch_second(self.off)
    }
    fn with_ldt(&self, ldt: LDT) -> SResult<Zoned> {
        Zoned::of_local(ldt, self.zone.clone(), Some(self.off))
    }
    fn with_instant(&self, i: (i64, u32)) -> SResult<Zoned> {
        Zoned::of_instant(i, self.zone.clone())
    }
    fn write(&self, out: &mut String) {
        self.ldt.write(out);
        out.push_str(&offset_id(self.off));
        if !self.zone.is_offset() {
            out.push('[');
            out.push_str(&self.zone.id());
            out.push(']');
        }
    }
    fn hash(&self) -> i32 {
        self.ldt.hash() ^ self.off ^ self.zone.hash().rotate_left(3)
    }
    fn to_val(self) -> Val {
        if let Zone::Offset(0) = self.zone {
            if self.ldt.nod % 1_000_000 == 0 && (0..=9999).contains(&self.ldt.date.y) {
                let secs = self.ldt.epoch_second(0);
                if let Some(ms) = secs.checked_mul(1000).and_then(|m| m.checked_add(self.ldt.nanos() as i64 / 1_000_000)) {
                    return Val::Date(ms);
                }
            }
        }
        t(JTime::Zoned(self))
    }
    fn from_ms(ms: i64) -> Zoned {
        let i = ms_to_inst(ms);
        Zoned { ldt: LDT::from_epoch(i.0, i.1, 0).unwrap_or(LDT { date: LDate { y: 1970, m: 1, d: 1 }, nod: 0 }), off: 0, zone: Zone::Offset(0) }
    }
}

// ---------------------------------------------------------------------------
// Units and fields

type Unit = u8;
const U_NANOS: Unit = 0;
const U_MICROS: Unit = 1;
const U_MILLIS: Unit = 2;
const U_SECONDS: Unit = 3;
const U_MINUTES: Unit = 4;
const U_HOURS: Unit = 5;
const U_HALF_DAYS: Unit = 6;
const U_DAYS: Unit = 7;
const U_WEEKS: Unit = 8;
const U_MONTHS: Unit = 9;
const U_YEARS: Unit = 10;
const U_DECADES: Unit = 11;
const U_CENTURIES: Unit = 12;
const U_MILLENNIA: Unit = 13;
const U_ERAS: Unit = 14;
const U_FOREVER: Unit = 15;
const U_QUARTER_YEARS: Unit = 16;
const U_WEEK_BASED_YEARS: Unit = 17;

fn unit_name(u: Unit) -> String {
    match u {
        U_QUARTER_YEARS => "QuarterYears".into(),
        U_WEEK_BASED_YEARS => "WeekBasedYears".into(),
        _ => {
            let n = CHRONO_UNITS[(u as usize).min(15)];
            let mut out = String::new();
            for part in n.split('_') {
                let mut c = part.chars();
                if let Some(f) = c.next() {
                    out.push(f);
                    out.push_str(&c.as_str().to_lowercase());
                }
            }
            out
        }
    }
}
/// `ChronoUnit.getDuration()` as (seconds, nanos).
fn unit_duration(u: Unit) -> (i64, u32) {
    match u {
        U_NANOS => (0, 1),
        U_MICROS => (0, 1000),
        U_MILLIS => (0, 1_000_000),
        U_SECONDS => (1, 0),
        U_MINUTES => (60, 0),
        U_HOURS => (3600, 0),
        U_HALF_DAYS => (43_200, 0),
        U_DAYS => (SPD, 0),
        U_WEEKS => (7 * SPD, 0),
        U_MONTHS => (31_556_952 / 12, 0),
        U_YEARS => (31_556_952, 0),
        U_DECADES => (315_569_520, 0),
        U_CENTURIES => (3_155_695_200, 0),
        U_MILLENNIA => (31_556_952_000, 0),
        U_ERAS => (31_556_952_000_000_000, 0),
        U_FOREVER => (i64::MAX, 999_999_999),
        U_QUARTER_YEARS => (31_556_952 / 4, 0),
        _ => (31_556_952, 0),
    }
}
fn unit_is_time_based(u: Unit) -> bool {
    u < U_DAYS
}
fn unit_is_date_based(u: Unit) -> bool {
    (U_DAYS..=U_ERAS).contains(&u) || u >= U_QUARTER_YEARS
}
fn unit_is_estimated(u: Unit) -> bool {
    u >= U_DAYS
}
fn unit_val(u: Unit) -> Val {
    if u < 16 {
        Val::ChronoUnit(u)
    } else {
        t(JTime::Unit(u))
    }
}

/// `ChronoField` (ordinal order), `IsoFields` and localized `WeekFields`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum F {
    NanoOfSecond,
    NanoOfDay,
    MicroOfSecond,
    MicroOfDay,
    MilliOfSecond,
    MilliOfDay,
    SecondOfMinute,
    SecondOfDay,
    MinuteOfHour,
    MinuteOfDay,
    HourOfAmPm,
    ClockHourOfAmPm,
    HourOfDay,
    ClockHourOfDay,
    AmPmOfDay,
    DayOfWeek,
    AlignedDayOfWeekInMonth,
    AlignedDayOfWeekInYear,
    DayOfMonth,
    DayOfYear,
    EpochDay,
    AlignedWeekOfMonth,
    AlignedWeekOfYear,
    MonthOfYear,
    ProlepticMonth,
    YearOfEra,
    Year,
    Era,
    InstantSeconds,
    OffsetSeconds,
    // IsoFields
    DayOfQuarter,
    QuarterOfYear,
    WeekOfWeekBasedYear,
    WeekBasedYear,
    // Localized WeekFields (pattern letters e/c, w, W, Y)
    LocDayOfWeek,
    LocWeekOfMonth,
    LocWeekOfYear,
    LocWeekBasedYear,
}

const CHRONO_FIELDS: [F; 30] = [
    F::NanoOfSecond,
    F::NanoOfDay,
    F::MicroOfSecond,
    F::MicroOfDay,
    F::MilliOfSecond,
    F::MilliOfDay,
    F::SecondOfMinute,
    F::SecondOfDay,
    F::MinuteOfHour,
    F::MinuteOfDay,
    F::HourOfAmPm,
    F::ClockHourOfAmPm,
    F::HourOfDay,
    F::ClockHourOfDay,
    F::AmPmOfDay,
    F::DayOfWeek,
    F::AlignedDayOfWeekInMonth,
    F::AlignedDayOfWeekInYear,
    F::DayOfMonth,
    F::DayOfYear,
    F::EpochDay,
    F::AlignedWeekOfMonth,
    F::AlignedWeekOfYear,
    F::MonthOfYear,
    F::ProlepticMonth,
    F::YearOfEra,
    F::Year,
    F::Era,
    F::InstantSeconds,
    F::OffsetSeconds,
];

impl F {
    fn name(&self) -> &'static str {
        match self {
            F::NanoOfSecond => "NanoOfSecond",
            F::NanoOfDay => "NanoOfDay",
            F::MicroOfSecond => "MicroOfSecond",
            F::MicroOfDay => "MicroOfDay",
            F::MilliOfSecond => "MilliOfSecond",
            F::MilliOfDay => "MilliOfDay",
            F::SecondOfMinute => "SecondOfMinute",
            F::SecondOfDay => "SecondOfDay",
            F::MinuteOfHour => "MinuteOfHour",
            F::MinuteOfDay => "MinuteOfDay",
            F::HourOfAmPm => "HourOfAmPm",
            F::ClockHourOfAmPm => "ClockHourOfAmPm",
            F::HourOfDay => "HourOfDay",
            F::ClockHourOfDay => "ClockHourOfDay",
            F::AmPmOfDay => "AmPmOfDay",
            F::DayOfWeek => "DayOfWeek",
            F::AlignedDayOfWeekInMonth => "AlignedDayOfWeekInMonth",
            F::AlignedDayOfWeekInYear => "AlignedDayOfWeekInYear",
            F::DayOfMonth => "DayOfMonth",
            F::DayOfYear => "DayOfYear",
            F::EpochDay => "EpochDay",
            F::AlignedWeekOfMonth => "AlignedWeekOfMonth",
            F::AlignedWeekOfYear => "AlignedWeekOfYear",
            F::MonthOfYear => "MonthOfYear",
            F::ProlepticMonth => "ProlepticMonth",
            F::YearOfEra => "YearOfEra",
            F::Year => "Year",
            F::Era => "Era",
            F::InstantSeconds => "InstantSeconds",
            F::OffsetSeconds => "OffsetSeconds",
            F::DayOfQuarter => "DayOfQuarter",
            F::QuarterOfYear => "QuarterOfYear",
            F::WeekOfWeekBasedYear => "WeekOfWeekBasedYear",
            F::WeekBasedYear => "WeekBasedYear",
            F::LocDayOfWeek => "DayOfWeek",
            F::LocWeekOfMonth => "WeekOfMonth",
            F::LocWeekOfYear => "WeekOfYear",
            F::LocWeekBasedYear => "WeekBasedYear",
        }
    }
    /// Enum constant name (`DAY_OF_MONTH`).
    fn enum_name(&self) -> String {
        let mut out = String::new();
        for (i, c) in self.name().chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_uppercase());
        }
        out
    }
    fn from_enum_name(n: &str) -> Option<F> {
        CHRONO_FIELDS.iter().copied().find(|f| f.enum_name() == n)
    }
    /// `range()` as (min, maxSmallest, maxLargest).
    fn range(&self) -> (i64, i64, i64) {
        match self {
            F::NanoOfSecond => (0, 999_999_999, 999_999_999),
            F::NanoOfDay => (0, NPD - 1, NPD - 1),
            F::MicroOfSecond => (0, 999_999, 999_999),
            F::MicroOfDay => (0, 86_400_000_000 - 1, 86_400_000_000 - 1),
            F::MilliOfSecond => (0, 999, 999),
            F::MilliOfDay => (0, 86_400_000 - 1, 86_400_000 - 1),
            F::SecondOfMinute => (0, 59, 59),
            F::SecondOfDay => (0, SPD - 1, SPD - 1),
            F::MinuteOfHour => (0, 59, 59),
            F::MinuteOfDay => (0, 1439, 1439),
            F::HourOfAmPm => (0, 11, 11),
            F::ClockHourOfAmPm => (1, 12, 12),
            F::HourOfDay => (0, 23, 23),
            F::ClockHourOfDay => (1, 24, 24),
            F::AmPmOfDay => (0, 1, 1),
            F::DayOfWeek | F::AlignedDayOfWeekInMonth | F::AlignedDayOfWeekInYear | F::LocDayOfWeek => (1, 7, 7),
            F::DayOfMonth => (1, 28, 31),
            F::DayOfYear => (1, 365, 366),
            F::EpochDay => (-365_243_219_162, 365_241_780_471, 365_241_780_471),
            F::AlignedWeekOfMonth => (1, 4, 5),
            F::AlignedWeekOfYear => (1, 53, 53),
            F::MonthOfYear => (1, 12, 12),
            F::ProlepticMonth => (MIN_YEAR * 12, MAX_YEAR * 12 + 11, MAX_YEAR * 12 + 11),
            F::YearOfEra => (1, MAX_YEAR, MAX_YEAR + 1),
            F::Year | F::WeekBasedYear | F::LocWeekBasedYear => (MIN_YEAR, MAX_YEAR, MAX_YEAR),
            F::Era => (0, 1, 1),
            F::InstantSeconds => (i64::MIN, i64::MAX, i64::MAX),
            F::OffsetSeconds => (-64_800, 64_800, 64_800),
            F::DayOfQuarter => (1, 90, 92),
            F::QuarterOfYear => (1, 4, 4),
            F::WeekOfWeekBasedYear => (1, 52, 53),
            F::LocWeekOfMonth => (0, 4, 6),
            F::LocWeekOfYear => (0, 52, 54),
        }
    }
    fn range_str(&self) -> String {
        let (min, ms, ml) = self.range();
        range_str(min, min, ms, ml)
    }
    /// `checkValidValue`.
    fn check(&self, v: i64) -> SResult<i64> {
        let (min, _, max) = self.range();
        if v < min || v > max {
            return Err(frs(dte(format!("Invalid value for {} (valid values {}): {v}", self.name(), self.range_str())), &[VR_CHECK, CF_CHECK]));
        }
        Ok(v)
    }
    /// `checkValidIntValue`.
    fn check_int(&self, v: i64) -> SResult<i64> {
        let (min, _, max) = self.range();
        if v < min || v > max {
            return Err(frs(dte(format!("Invalid value for {} (valid values {}): {v}", self.name(), self.range_str())), &[VR_CHECK_INT, CF_CHECK_INT]));
        }
        Ok(v)
    }
    fn is_int_range(&self) -> bool {
        let (min, _, max) = self.range();
        min >= i32::MIN as i64 && max <= i32::MAX as i64
    }
    fn is_time_based(&self) -> bool {
        *self <= F::AmPmOfDay
    }
    fn is_date_based(&self) -> bool {
        (*self >= F::DayOfWeek && *self <= F::Era) || *self >= F::DayOfQuarter
    }
    fn is_chrono(&self) -> bool {
        *self <= F::OffsetSeconds
    }
    fn base_unit(&self) -> Unit {
        match self {
            F::NanoOfSecond | F::NanoOfDay => U_NANOS,
            F::MicroOfSecond | F::MicroOfDay => U_MICROS,
            F::MilliOfSecond | F::MilliOfDay => U_MILLIS,
            F::SecondOfMinute | F::SecondOfDay | F::InstantSeconds | F::OffsetSeconds => U_SECONDS,
            F::MinuteOfHour | F::MinuteOfDay => U_MINUTES,
            F::HourOfAmPm | F::ClockHourOfAmPm | F::HourOfDay | F::ClockHourOfDay => U_HOURS,
            F::AmPmOfDay => U_HALF_DAYS,
            F::DayOfWeek | F::AlignedDayOfWeekInMonth | F::AlignedDayOfWeekInYear | F::DayOfMonth | F::DayOfYear | F::EpochDay | F::DayOfQuarter | F::LocDayOfWeek => U_DAYS,
            F::AlignedWeekOfMonth | F::AlignedWeekOfYear | F::WeekOfWeekBasedYear | F::LocWeekOfMonth | F::LocWeekOfYear => U_WEEKS,
            F::MonthOfYear | F::ProlepticMonth => U_MONTHS,
            F::YearOfEra | F::Year | F::LocWeekBasedYear => U_YEARS,
            F::Era => U_ERAS,
            F::QuarterOfYear => U_QUARTER_YEARS,
            F::WeekBasedYear => U_WEEK_BASED_YEARS,
        }
    }
    fn range_unit(&self) -> Unit {
        match self {
            F::NanoOfSecond | F::MicroOfSecond | F::MilliOfSecond => U_SECONDS,
            F::SecondOfMinute => U_MINUTES,
            F::MinuteOfHour => U_HOURS,
            F::HourOfAmPm | F::ClockHourOfAmPm => U_HALF_DAYS,
            F::NanoOfDay | F::MicroOfDay | F::MilliOfDay | F::SecondOfDay | F::MinuteOfDay | F::HourOfDay | F::ClockHourOfDay | F::AmPmOfDay => U_DAYS,
            F::DayOfWeek | F::AlignedDayOfWeekInMonth | F::AlignedDayOfWeekInYear | F::LocDayOfWeek => U_WEEKS,
            F::DayOfMonth | F::AlignedWeekOfMonth | F::LocWeekOfMonth => U_MONTHS,
            F::DayOfYear | F::AlignedWeekOfYear | F::MonthOfYear | F::LocWeekOfYear => U_YEARS,
            F::DayOfQuarter => U_QUARTER_YEARS,
            F::QuarterOfYear => U_YEARS,
            F::WeekOfWeekBasedYear => U_WEEK_BASED_YEARS,
            F::YearOfEra => U_ERAS,
            F::EpochDay | F::ProlepticMonth | F::Year | F::Era | F::InstantSeconds | F::OffsetSeconds | F::WeekBasedYear | F::LocWeekBasedYear => U_FOREVER,
        }
    }
}

fn range_str(min_s: i64, min_l: i64, max_s: i64, max_l: i64) -> String {
    let mut s = min_s.to_string();
    if min_s != min_l {
        let _ = write!(s, "/{min_l}");
    }
    let _ = write!(s, " - {max_s}");
    if max_s != max_l {
        let _ = write!(s, "/{max_l}");
    }
    s
}

// ---------------------------------------------------------------------------
// Locale

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Locale {
    lang: String,
    country: String,
    variant: String,
}

impl Locale {
    pub(super) fn language(&self) -> &str { &self.lang }
    fn new(lang: &str, country: &str) -> Locale {
        Locale { lang: lang.to_lowercase(), country: country.to_uppercase(), variant: String::new() }
    }
    fn en() -> Locale {
        Locale::new("en", "")
    }
    fn root() -> Locale {
        Locale::new("", "")
    }
    fn to_string(&self) -> String {
        let mut s = self.lang.clone();
        if !self.country.is_empty() || !self.variant.is_empty() {
            s.push('_');
            s.push_str(&self.country);
        }
        if !self.variant.is_empty() {
            s.push('_');
            s.push_str(&self.variant);
        }
        s
    }
    fn tag(&self) -> String {
        if self.lang.is_empty() && self.country.is_empty() {
            return "und".into();
        }
        let mut s = if self.lang.is_empty() { "und".to_string() } else { self.lang.clone() };
        if !self.country.is_empty() {
            s.push('-');
            s.push_str(&self.country);
        }
        if !self.variant.is_empty() {
            s.push('-');
            s.push_str(&self.variant);
        }
        s
    }
    fn for_tag(tag: &str) -> Locale {
        let mut parts = tag.split(['-', '_']);
        let lang = parts.next().unwrap_or("");
        let mut l = Locale::new(if lang == "und" { "" } else { lang }, "");
        for p in parts {
            if p.len() == 2 && p.chars().all(|c| c.is_ascii_alphabetic()) && l.country.is_empty() {
                l.country = p.to_uppercase();
            } else if p.len() == 3 && p.chars().all(|c| c.is_ascii_digit()) && l.country.is_empty() {
                l.country = p.to_string();
            } else if !p.is_empty() && p.len() >= 4 {
                l.variant = p.to_string();
            }
        }
        l
    }
    fn constant(name: &str) -> Option<Locale> {
        Some(match name {
            "ROOT" => Locale::root(),
            "ENGLISH" => Locale::new("en", ""),
            "FRENCH" => Locale::new("fr", ""),
            "GERMAN" => Locale::new("de", ""),
            "ITALIAN" => Locale::new("it", ""),
            "JAPANESE" => Locale::new("ja", ""),
            "KOREAN" => Locale::new("ko", ""),
            "CHINESE" => Locale::new("zh", ""),
            "SIMPLIFIED_CHINESE" => Locale::new("zh", "CN"),
            "TRADITIONAL_CHINESE" => Locale::new("zh", "TW"),
            "FRANCE" => Locale::new("fr", "FR"),
            "GERMANY" => Locale::new("de", "DE"),
            "ITALY" => Locale::new("it", "IT"),
            "JAPAN" => Locale::new("ja", "JP"),
            "KOREA" => Locale::new("ko", "KR"),
            "UK" => Locale::new("en", "GB"),
            "US" => Locale::new("en", "US"),
            "CANADA" => Locale::new("en", "CA"),
            "CANADA_FRENCH" => Locale::new("fr", "CA"),
            "CHINA" | "PRC" => Locale::new("zh", "CN"),
            "TAIWAN" => Locale::new("zh", "TW"),
            _ => return None,
        })
    }
    fn hash(&self) -> i32 {
        str_hash(&self.to_string())
    }
    fn display_name(&self) -> String {
        let lang = match self.lang.as_str() {
            "en" => "English",
            "fr" => "French",
            "de" => "German",
            "it" => "Italian",
            "ja" => "Japanese",
            "ko" => "Korean",
            "zh" => "Chinese",
            "es" => "Spanish",
            "pt" => "Portuguese",
            "nl" => "Dutch",
            "ru" => "Russian",
            "" => "",
            other => other,
        };
        let country = match self.country.as_str() {
            "US" => "United States",
            "GB" => "United Kingdom",
            "FR" => "France",
            "DE" => "Germany",
            "IT" => "Italy",
            "JP" => "Japan",
            "KR" => "South Korea",
            "CN" => "China",
            "TW" => "Taiwan",
            "CA" => "Canada",
            "ES" => "Spain",
            "AU" => "Australia",
            "" => "",
            other => other,
        };
        if country.is_empty() {
            lang.to_string()
        } else if lang.is_empty() {
            country.to_string()
        } else {
            format!("{lang} ({country})")
        }
    }
    /// `WeekFields.of(locale)` as (first day 1=Mon..7=Sun, minimal days).
    fn week_fields(&self) -> (u8, u8) {
        match self.country.as_str() {
            "" => {
                if self.lang.is_empty() || self.lang == "en" {
                    (7, 1)
                } else {
                    (1, 4)
                }
            }
            "US" | "CA" | "JP" | "BR" | "MX" | "AU" | "IN" | "CN" | "TW" | "KR" | "ZA" | "IL" | "PH" | "HK" | "SA" | "AR" | "CO" | "PE" => (7, 1),
            _ => (1, 4),
        }
    }
}

// ---------------------------------------------------------------------------
// Values

/// `TemporalAdjusters` instances.
#[derive(Clone, Debug)]
pub enum Adj {
    FirstDayOfMonth,
    LastDayOfMonth,
    FirstDayOfNextMonth,
    FirstDayOfYear,
    LastDayOfYear,
    FirstDayOfNextYear,
    DayOfWeekInMonth(i64, u8),
    Next(u8),
    NextOrSame(u8),
    Previous(u8),
    PreviousOrSame(u8),
    /// `ofDateAdjuster(lambda)`.
    DateAdjuster(Val),
}

/// `TemporalAccessor` produced by `DateTimeFormatter.parse`.
#[derive(Clone, Debug)]
pub struct Parsed {
    fields: Vec<(F, i64)>,
    zone: Option<Zone>,
    date: Option<LDate>,
    time: Option<i64>,
}

#[derive(Debug)]
pub enum JTime {
    LocalDate(LDate),
    LocalTime(i64),
    LocalDateTime(LDT),
    OffsetDateTime(LDT, i32),
    OffsetTime(i64, i32),
    Zoned(Zoned),
    Instant(i64, u32),
    Duration(i64, u32),
    Period(i32, i32, i32),
    Zone(Zone),
    YearMonth(i32, u8),
    Year(i32),
    MonthDay(u8, u8),
    Formatter(Rc<Fmt>),
    Adjuster(Adj),
    TextStyle(u8),
    Locale(Locale),
    Field(F),
    /// `IsoFields.QUARTER_YEARS` / `WEEK_BASED_YEARS`.
    Unit(Unit),
    ResolverStyle(u8),
    FormatStyle(u8),
    Clock(i64, u32, Zone),
    OffsetClock(Rc<JTime>, i64, u32),
    ZoneRules(Zone),
    Chronology,
    Era(u8),
    /// `ValueRange`: (minSmallest, minLargest, maxSmallest, maxLargest).
    Range(i64, i64, i64, i64),
    Parsed(Parsed),
    /// `TemporalQueries.*`.
    Query(&'static str),
    /// `ZoneOffsetTransition` (epoch second, before, after).
    Transition(i64, i32, i32),
}

const TEXT_STYLES: [&str; 6] = ["FULL", "FULL_STANDALONE", "SHORT", "SHORT_STANDALONE", "NARROW", "NARROW_STANDALONE"];
const RESOLVER_STYLES: [&str; 3] = ["STRICT", "SMART", "LENIENT"];
const FORMAT_STYLES: [&str; 4] = ["FULL", "LONG", "MEDIUM", "SHORT"];

impl JTime {
    /// Borrowed upper bound for Java text rendering, without constructing the
    /// formatter description or recursively formatting offset-clock objects.
    pub(crate) fn text_bound(&self) -> u64 {
        fn zone(z:&Zone)->u64 { match z {Zone::Fixed(s,_)=>s.len() as u64,_=>128} }
        fn elements(els:&[El])->u64 {
            els.iter().fold(0u64,|n,e|n.saturating_add(match e {
                El::Lit(s)=>(s.len() as u64).saturating_mul(2).saturating_add(2),
                El::Opt(v)|El::Group(v)|El::Localized(_,_,v)=>elements(v).saturating_add(256),
                _=>256,
            }))
        }
        512u64.saturating_add(match self {
            JTime::Formatter(f)=>elements(&f.els),
            JTime::Locale(l)=>(l.lang.len() as u64).saturating_add(l.country.len() as u64).saturating_add(l.variant.len() as u64),
            JTime::Zone(z)|JTime::ZoneRules(z)|JTime::Clock(_,_,z)=>zone(z),
            JTime::Zoned(z)=>zone(&z.zone),
            JTime::OffsetClock(base,_,_)=>base.text_bound(),
            JTime::Parsed(p)=>(p.fields.len() as u64).saturating_mul(128).saturating_add(p.zone.as_ref().map(zone).unwrap_or(0)),
            _=>0,
        })
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            JTime::LocalDate(_) => "java.time.LocalDate",
            JTime::LocalTime(_) => "java.time.LocalTime",
            JTime::LocalDateTime(_) => "java.time.LocalDateTime",
            JTime::OffsetDateTime(..) => "java.time.OffsetDateTime",
            JTime::OffsetTime(..) => "java.time.OffsetTime",
            JTime::Zoned(_) => "java.time.ZonedDateTime",
            JTime::Instant(..) => "java.time.Instant",
            JTime::Duration(..) => "java.time.Duration",
            JTime::Period(..) => "java.time.Period",
            JTime::Zone(Zone::Offset(_)) => "java.time.ZoneOffset",
            JTime::Zone(_) => "java.time.ZoneId",
            JTime::YearMonth(..) => "java.time.YearMonth",
            JTime::Year(_) => "java.time.Year",
            JTime::MonthDay(..) => "java.time.MonthDay",
            JTime::Formatter(_) => "java.time.format.DateTimeFormatter",
            JTime::Adjuster(_) => "java.time.temporal.TemporalAdjuster",
            JTime::TextStyle(_) => "java.time.format.TextStyle",
            JTime::Locale(_) => "java.util.Locale",
            JTime::Field(f) => {
                if f.is_chrono() {
                    "java.time.temporal.ChronoField"
                } else {
                    "java.time.temporal.TemporalField"
                }
            }
            JTime::Unit(_) => "java.time.temporal.TemporalUnit",
            JTime::ResolverStyle(_) => "java.time.format.ResolverStyle",
            JTime::FormatStyle(_) => "java.time.format.FormatStyle",
            JTime::Clock(..) | JTime::OffsetClock(..) => "java.time.Clock",
            JTime::ZoneRules(_) => "java.time.zone.ZoneRules",
            JTime::Chronology => "java.time.chrono.IsoChronology",
            JTime::Era(_) => "java.time.chrono.IsoEra",
            JTime::Range(..) => "java.time.temporal.ValueRange",
            JTime::Parsed(_) => "java.time.format.Parsed",
            JTime::Query(_) => "java.time.temporal.TemporalQuery",
            JTime::Transition(..) => "java.time.zone.ZoneOffsetTransition",
        }
    }

    pub fn write_java(&self, out: &mut String) {
        match self {
            JTime::LocalDate(d) => d.write(out),
            JTime::LocalTime(n) => push_time(out, *n),
            JTime::LocalDateTime(d) => d.write(out),
            JTime::OffsetDateTime(d, o) => {
                d.write(out);
                out.push_str(&offset_id(*o));
            }
            JTime::OffsetTime(n, o) => {
                push_time(out, *n);
                out.push_str(&offset_id(*o));
            }
            JTime::Zoned(z) => z.write(out),
            JTime::Instant(s, n) => inst_write(out, (*s, *n)),
            JTime::Duration(s, n) => dur_write(out, (*s, *n)),
            JTime::Period(y, m, d) => period_write(out, (*y, *m, *d)),
            JTime::Zone(z) => out.push_str(&z.id()),
            JTime::YearMonth(y, m) => {
                push_year(out, *y as i64);
                let _ = write!(out, "-{m:02}");
            }
            JTime::Year(y) => {
                let _ = write!(out, "{y}");
            }
            JTime::MonthDay(m, d) => {
                let _ = write!(out, "--{m:02}-{d:02}");
            }
            JTime::Formatter(f) => out.push_str(&f.describe()),
            JTime::Adjuster(_) => out.push_str("java.time.temporal.TemporalAdjusters$$Lambda@0"),
            JTime::TextStyle(s) => out.push_str(TEXT_STYLES[*s as usize]),
            JTime::Locale(l) => out.push_str(&l.to_string()),
            JTime::Field(f) => out.push_str(f.name()),
            JTime::Unit(u) => out.push_str(&unit_name(*u)),
            JTime::ResolverStyle(s) => out.push_str(RESOLVER_STYLES[*s as usize]),
            JTime::FormatStyle(s) => out.push_str(FORMAT_STYLES[*s as usize]),
            JTime::Clock(s, n, z) => {
                out.push_str("FixedClock[");
                inst_write(out, (*s, *n));
                let _ = write!(out, ",{}]", z.id());
            }
            JTime::OffsetClock(base, s, n) => {
                out.push_str("OffsetClock[");
                base.write_java(out);
                out.push(',');
                dur_write(out, (*s, *n));
                out.push(']');
            }
            JTime::ZoneRules(z) => {
                let now = crate::date::now_millis() / 1000;
                let _ = write!(out, "ZoneRules[currentStandardOffset={}]", offset_id(z.standard_offset_at(now)));
            }
            JTime::Chronology => out.push_str("ISO"),
            JTime::Era(e) => out.push_str(if *e == 1 { "CE" } else { "BCE" }),
            JTime::Range(a, b, c, d) => out.push_str(&range_str(*a, *b, *c, *d)),
            JTime::Parsed(p) => p.write(out),
            JTime::Query(q) => out.push_str(q),
            JTime::Transition(secs, before, after) => {
                let ldt = LDT::from_epoch(*secs, 0, *before).unwrap_or(LDT { date: LDate { y: 1970, m: 1, d: 1 }, nod: 0 });
                let mut s = String::new();
                ldt.write(&mut s);
                let _ = write!(out, "Transition[{} at {s}{} to {}]", if after > before { "Gap" } else { "Overlap" }, offset_id(*before), offset_id(*after));
            }
        }
    }

    pub fn hash(&self) -> i32 {
        match self {
            JTime::LocalDate(d) => d.hash(),
            JTime::LocalTime(n) => time_hash(*n),
            JTime::LocalDateTime(d) => d.hash(),
            JTime::OffsetDateTime(d, o) => d.hash() ^ o,
            JTime::OffsetTime(n, o) => time_hash(*n) ^ o,
            JTime::Zoned(z) => z.hash(),
            JTime::Instant(s, n) => inst_hash((*s, *n)),
            JTime::Duration(s, n) => dur_hash((*s, *n)),
            JTime::Period(y, m, d) => y.wrapping_add(m.rotate_left(8)).wrapping_add(d.rotate_left(16)),
            JTime::Zone(z) => z.hash(),
            JTime::YearMonth(y, m) => y ^ ((*m as i32) << 27),
            JTime::Year(y) => *y,
            JTime::MonthDay(m, d) => ((*m as i32) << 6) + *d as i32,
            JTime::Locale(l) => l.hash(),
            JTime::Range(a, b, c, d) => {
                let h = a + (b << 16) + (b >> 48) + (c << 32) + (c >> 32) + (d << 48) + (d >> 16);
                (h ^ (h >> 32)) as i32
            }
            JTime::Clock(s, n, z) => inst_hash((*s, *n)) ^ z.hash(),
            other => str_hash(&{
                let mut s = String::new();
                other.write_java(&mut s);
                s
            }),
        }
    }
}

impl Parsed {
    fn write(&self, out: &mut String) {
        out.push('{');
        for (i, (f, v)) in self.fields.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{}={v}", f.name());
        }
        out.push_str("},ISO");
        if let Some(z) = &self.zone {
            let _ = write!(out, ",{}", z.id());
        }
        if self.date.is_some() || self.time.is_some() {
            out.push_str(" resolved to ");
            if let Some(d) = &self.date {
                d.write(out);
            }
            if let Some(t) = self.time {
                if self.date.is_some() {
                    out.push('T');
                }
                push_time(out, t);
            }
        }
    }
    fn field(&self, f: F) -> Option<i64> {
        if let Some((_, v)) = self.fields.iter().find(|(g, _)| *g == f) {
            return Some(*v);
        }
        if let Some(d) = &self.date {
            if let Some(v) = d.get(f) {
                return Some(v);
            }
        }
        if let Some(t) = self.time {
            if let Some(v) = time_get(t, f) {
                return Some(v);
            }
        }
        None
    }
    fn offset(&self) -> Option<i32> {
        self.field(F::OffsetSeconds).map(|v| v as i32)
    }
}

/// Convert a value to a temporal view (`Val::Date`/`Val::Instant` included).
fn as_jtime(v: &Val) -> Option<JTime> {
    match v {
        Val::Date(ms) => Some(JTime::Zoned(Zoned::from_ms(*ms))),
        Val::Instant(ms) => {
            let i = ms_to_inst(*ms);
            Some(JTime::Instant(i.0, i.1))
        }
        Val::Time(t) => Some(match &**t {
            JTime::LocalDate(d) => JTime::LocalDate(*d),
            JTime::LocalTime(n) => JTime::LocalTime(*n),
            JTime::LocalDateTime(d) => JTime::LocalDateTime(*d),
            JTime::OffsetDateTime(d, o) => JTime::OffsetDateTime(*d, *o),
            JTime::OffsetTime(n, o) => JTime::OffsetTime(*n, *o),
            JTime::Zoned(z) => JTime::Zoned(z.clone()),
            JTime::Instant(s, n) => JTime::Instant(*s, *n),
            JTime::Duration(s, n) => JTime::Duration(*s, *n),
            JTime::Period(y, m, d) => JTime::Period(*y, *m, *d),
            JTime::Zone(z) => JTime::Zone(z.clone()),
            JTime::YearMonth(y, m) => JTime::YearMonth(*y, *m),
            JTime::Year(y) => JTime::Year(*y),
            JTime::MonthDay(m, d) => JTime::MonthDay(*m, *d),
            JTime::Parsed(p) => JTime::Parsed(p.clone()),
            _ => return None,
        }),
        _ => None,
    }
}

/// `equals` where at least one side is a `Val::Time` (or Date vs Instant).
pub fn equals(a: &Val, b: &Val) -> bool {
    let (Some(x), Some(y)) = (as_jtime(a), as_jtime(b)) else {
        return match (a, b) {
            (Val::Time(x), Val::Time(y)) => match (&**x, &**y) {
                (JTime::Formatter(f), JTime::Formatter(g)) => Rc::ptr_eq(f, g) || f.describe() == g.describe(),
                (JTime::TextStyle(f), JTime::TextStyle(g)) => f == g,
                (JTime::Locale(f), JTime::Locale(g)) => f == g,
                (JTime::Field(f), JTime::Field(g)) => f == g,
                (JTime::Unit(f), JTime::Unit(g)) => f == g,
                (JTime::ResolverStyle(f), JTime::ResolverStyle(g)) => f == g,
                (JTime::FormatStyle(f), JTime::FormatStyle(g)) => f == g,
                (JTime::Clock(s, n, z), JTime::Clock(s2, n2, z2)) => s == s2 && n == n2 && z.same(z2),
                (JTime::ZoneRules(z), JTime::ZoneRules(z2)) => z.id() == z2.id(),
                (JTime::Chronology, JTime::Chronology) => true,
                (JTime::Era(e), JTime::Era(f)) => e == f,
                (JTime::Range(a, b, c, d), JTime::Range(e, f, g, h)) => (a, b, c, d) == (e, f, g, h),
                (JTime::Query(q), JTime::Query(r)) => q == r,
                (JTime::Transition(a, b, c), JTime::Transition(d, e, f)) => (a, b, c) == (d, e, f),
                _ => false,
            },
            _ => false,
        };
    };
    match (&x, &y) {
        (JTime::LocalDate(a), JTime::LocalDate(b)) => a == b,
        (JTime::LocalTime(a), JTime::LocalTime(b)) => a == b,
        (JTime::LocalDateTime(a), JTime::LocalDateTime(b)) => a == b,
        (JTime::OffsetDateTime(a, o), JTime::OffsetDateTime(b, p)) => a == b && o == p,
        (JTime::OffsetTime(a, o), JTime::OffsetTime(b, p)) => a == b && o == p,
        (JTime::Zoned(a), JTime::Zoned(b)) => a.ldt == b.ldt && a.off == b.off && a.zone.same(&b.zone),
        (JTime::Instant(s, n), JTime::Instant(s2, n2)) => s == s2 && n == n2,
        (JTime::Duration(s, n), JTime::Duration(s2, n2)) => s == s2 && n == n2,
        (JTime::Period(a, b, c), JTime::Period(d, e, f)) => (a, b, c) == (d, e, f),
        (JTime::Zone(a), JTime::Zone(b)) => a.same(b),
        (JTime::YearMonth(a, b), JTime::YearMonth(c, d)) => (a, b) == (c, d),
        (JTime::Year(a), JTime::Year(b)) => a == b,
        (JTime::MonthDay(a, b), JTime::MonthDay(c, d)) => (a, b) == (c, d),
        _ => false,
    }
}

fn sign(v: i64) -> Ordering {
    v.cmp(&0)
}

/// Java `compareTo` result (raw int) for two temporal values of the same class.
fn compare_to(a: &Val, b: &Val) -> SResult<i64> {
    let Some(x) = as_jtime(a) else { return Err(class_cast(format!("Cannot cast {} to java.lang.Comparable", a.type_name()))) };
    let target = match &x {
        JTime::LocalDate(_) => "java.time.chrono.ChronoLocalDate",
        JTime::LocalTime(_) => "java.time.LocalTime",
        JTime::LocalDateTime(_) => "java.time.chrono.ChronoLocalDateTime",
        JTime::OffsetDateTime(..) => "java.time.OffsetDateTime",
        JTime::OffsetTime(..) => "java.time.OffsetTime",
        JTime::Zoned(_) => "java.time.chrono.ChronoZonedDateTime",
        JTime::Instant(..) => "java.time.Instant",
        JTime::Duration(..) => "java.time.Duration",
        JTime::Zone(Zone::Offset(_)) => "java.time.ZoneOffset",
        JTime::YearMonth(..) => "java.time.YearMonth",
        JTime::Year(_) => "java.time.Year",
        JTime::MonthDay(..) => "java.time.MonthDay",
        _ => return Err(class_cast(format!("Cannot cast {} to java.lang.Comparable", a.type_name()))),
    };
    if b.is_null() {
        return Err(match T::of(a) {
            Some(tv) => npe(null_other_msg(&tv, true)),
            None => match x {
                JTime::Duration(..) => npe("Cannot read field \"seconds\" because \"otherDuration\" is null"),
                JTime::Zone(_) => npe("Cannot read field \"totalSeconds\" because \"other\" is null"),
                _ => npe("Cannot compare with null"),
            },
        });
    }
    let Some(y) = as_jtime(b) else { return Err(cast_to(b, target)) };
    let ldt_cmp = |a: &LDT, b: &LDT| -> i64 {
        let c = (a.date.y as i64 - b.date.y as i64, a.date.m as i64 - b.date.m as i64, a.date.d as i64 - b.date.d as i64);
        if c.0 != 0 {
            return c.0;
        }
        if c.1 != 0 {
            return c.1;
        }
        if c.2 != 0 {
            return c.2;
        }
        a.nod.cmp(&b.nod) as i64
    };
    Ok(match (&x, &y) {
        (JTime::LocalDate(a), JTime::LocalDate(b)) => {
            let c = a.y as i64 - b.y as i64;
            if c != 0 {
                c
            } else if a.m != b.m {
                a.m as i64 - b.m as i64
            } else {
                a.d as i64 - b.d as i64
            }
        }
        (JTime::LocalTime(a), JTime::LocalTime(b)) => a.cmp(b) as i64,
        (JTime::LocalDateTime(a), JTime::LocalDateTime(b)) => ldt_cmp(a, b),
        (JTime::OffsetDateTime(a, o), JTime::OffsetDateTime(b, p)) => {
            if o == p {
                ldt_cmp(a, b)
            } else {
                let c = a.epoch_second(*o).cmp(&b.epoch_second(*p)) as i64;
                if c != 0 {
                    c
                } else if a.nanos() != b.nanos() {
                    a.nanos() as i64 - b.nanos() as i64
                } else {
                    ldt_cmp(a, b)
                }
            }
        }
        (JTime::OffsetTime(a, o), JTime::OffsetTime(b, p)) => {
            if o == p {
                a.cmp(b) as i64
            } else {
                let c = (a - *o as i64 * NPS).cmp(&(b - *p as i64 * NPS)) as i64;
                if c != 0 {
                    c
                } else {
                    a.cmp(b) as i64
                }
            }
        }
        (JTime::Zoned(a), JTime::Zoned(b)) => {
            let c = a.epoch_second().cmp(&b.epoch_second()) as i64;
            if c != 0 {
                c
            } else if a.ldt.nanos() != b.ldt.nanos() {
                a.ldt.nanos() as i64 - b.ldt.nanos() as i64
            } else {
                let c = ldt_cmp(&a.ldt, &b.ldt);
                if c != 0 {
                    c
                } else {
                    java_str_cmp(&a.zone.id(), &b.zone.id())
                }
            }
        }
        (JTime::Instant(s, n), JTime::Instant(s2, n2)) => {
            let c = s.cmp(s2) as i64;
            if c != 0 {
                c
            } else {
                *n as i64 - *n2 as i64
            }
        }
        (JTime::Duration(s, n), JTime::Duration(s2, n2)) => {
            let c = s.cmp(s2) as i64;
            if c != 0 {
                c
            } else {
                *n as i64 - *n2 as i64
            }
        }
        (JTime::Zone(Zone::Offset(a)), JTime::Zone(Zone::Offset(b))) => *b as i64 - *a as i64,
        (JTime::YearMonth(a, b), JTime::YearMonth(c, d)) => {
            if a != c {
                *a as i64 - *c as i64
            } else {
                *b as i64 - *d as i64
            }
        }
        (JTime::Year(a), JTime::Year(b)) => *a as i64 - *b as i64,
        (JTime::MonthDay(a, b), JTime::MonthDay(c, d)) => {
            if a != c {
                *a as i64 - *c as i64
            } else {
                *b as i64 - *d as i64
            }
        }
        _ => return Err(cast_to(b, target)),
    })
}

fn java_str_cmp(a: &str, b: &str) -> i64 {
    let (x, y): (Vec<u16>, Vec<u16>) = (a.encode_utf16().collect(), b.encode_utf16().collect());
    for (c, d) in x.iter().zip(y.iter()) {
        if c != d {
            return *c as i64 - *d as i64;
        }
    }
    x.len() as i64 - y.len() as i64
}

/// `compareTo` where at least one side is a `Val::Time` (or Date vs Instant).
pub fn compare(a: &Val, b: &Val) -> SResult<Ordering> {
    compare_to(a, b).map(sign)
}

// ---------------------------------------------------------------------------
// Temporal accessor view (field lookups for formatting / get / getLong)

#[derive(Clone, Debug, Default)]
struct Acc {
    date: Option<LDate>,
    time: Option<i64>,
    off: Option<i32>,
    zone: Option<Zone>,
    inst: Option<(i64, u32)>,
    /// Explicit fields (partials such as `YearMonth`, `Parsed`).
    extra: Vec<(F, i64)>,
    /// Only `extra` fields are supported (no date/time components).
    partial: bool,
    desc: String,
    /// Receiver class (for JDK frames).
    kind: &'static str,
}

/// `getLong` frames of a receiver class for an unsupported field.
fn getlong_chain(kind: &str, time_based: bool) -> Vec<Fr> {
    match kind {
        "LocalDate" => vec![LD_GET0_718, LD_GETLONG_698],
        "LocalTime" => vec![LT_GET0_712, LT_GETLONG_690],
        "LocalDateTime" => {
            if time_based {
                vec![LT_GET0_712, LT_GETLONG_690, LDT_GETLONG_720]
            } else {
                vec![LD_GET0_718, LD_GETLONG_698, LDT_GETLONG_720]
            }
        }
        "OffsetTime" => vec![LT_GET0_712, LT_GETLONG_690, OT_GETLONG_538],
        "Instant" => vec![INST_GETLONG_603],
        "YearMonth" => vec![YM_GETLONG_513],
        "Year" => vec![Y_GETLONG_507],
        "MonthDay" => vec![MD_GETLONG_470],
        "DayOfWeek" => vec![DOW_GETLONG_348],
        "Month" => vec![MON_GETLONG_375],
        "Parsed" => vec![PARSED_GETLONG],
        _ => Vec::new(),
    }
}

/// Innermost frames of `get(field)` for an unsupported field.
fn get_chain_inner(kind: &str, f: F) -> Vec<Fr> {
    match kind {
        "LocalDate" => vec![LD_GET0_718, LD_GET_661],
        "LocalTime" => vec![LT_GET0_712, LT_GET_653],
        "LocalDateTime" | "ZonedDateTime" | "OffsetDateTime" => {
            if f.is_time_based() {
                vec![LT_GET0_712, LT_GET_653]
            } else {
                vec![LD_GET0_718, LD_GET_661]
            }
        }
        "YearMonth" => vec![TA_RANGE, jt!("YearMonth", "range", 446)],
        "Year" => vec![TA_RANGE, jt!("Year", "range", 443)],
        "MonthDay" => vec![TA_RANGE, jt!("MonthDay", "range", 406)],
        _ => vec![],
    }
}
/// Outer `get` frames of composite / partial receivers.
fn get_chain_tail(kind: &str, mut inner: Vec<Fr>) -> Vec<Fr> {
    match kind {
        "LocalDateTime" => inner.push(LDT_GET_689),
        "ZonedDateTime" => {
            inner.push(LDT_GET_689);
            inner.push(ZDT_GET_823);
        }
        "YearMonth" => inner.push(jt!("YearMonth", "get", 478)),
        "Year" => inner.push(jt!("Year", "get", 474)),
        "MonthDay" => inner.push(jt!("MonthDay", "get", 437)),
        _ => {}
    }
    inner
}

impl Acc {
    fn of(v: &Val) -> Option<Acc> {
        let j = as_jtime(v)?;
        let kind = match &j {
            JTime::LocalDate(_) => "LocalDate",
            JTime::LocalTime(_) => "LocalTime",
            JTime::LocalDateTime(_) => "LocalDateTime",
            JTime::OffsetDateTime(..) => "OffsetDateTime",
            JTime::OffsetTime(..) => "OffsetTime",
            JTime::Zoned(_) => "ZonedDateTime",
            JTime::Instant(..) => "Instant",
            JTime::YearMonth(..) => "YearMonth",
            JTime::Year(_) => "Year",
            JTime::MonthDay(..) => "MonthDay",
            JTime::Parsed(_) => "Parsed",
            _ => "",
        };
        let mut a = Acc { desc: v.to_java_string(), kind, ..Default::default() };
        match j {
            JTime::LocalDate(d) => a.date = Some(d),
            JTime::LocalTime(n) => a.time = Some(n),
            JTime::LocalDateTime(d) => {
                a.date = Some(d.date);
                a.time = Some(d.nod);
            }
            JTime::OffsetDateTime(d, o) => {
                a.date = Some(d.date);
                a.time = Some(d.nod);
                a.off = Some(o);
                a.inst = Some((d.epoch_second(o), d.nanos()));
            }
            JTime::OffsetTime(n, o) => {
                a.time = Some(n);
                a.off = Some(o);
            }
            JTime::Zoned(z) => {
                a.date = Some(z.ldt.date);
                a.time = Some(z.ldt.nod);
                a.off = Some(z.off);
                a.inst = Some(z.instant());
                a.zone = Some(z.zone);
            }
            JTime::Instant(s, n) => a.inst = Some((s, n)),
            JTime::YearMonth(y, m) => {
                a.partial = true;
                let yoe = if y >= 1 { y as i64 } else { 1 - y as i64 };
                a.extra = vec![(F::Year, y as i64), (F::YearOfEra, yoe), (F::Era, era_of(y)), (F::MonthOfYear, m as i64), (F::ProlepticMonth, y as i64 * 12 + m as i64 - 1)];
            }
            JTime::Year(y) => {
                a.partial = true;
                let yoe = if y >= 1 { y as i64 } else { 1 - y as i64 };
                a.extra = vec![(F::Year, y as i64), (F::YearOfEra, yoe), (F::Era, era_of(y))];
            }
            JTime::MonthDay(m, d) => {
                a.partial = true;
                a.extra = vec![(F::MonthOfYear, m as i64), (F::DayOfMonth, d as i64)];
            }
            JTime::Parsed(p) => {
                a.date = p.date;
                a.time = p.time;
                a.zone = p.zone.clone();
                a.off = p.offset();
                a.extra = p.fields.clone();
                if a.date.is_none() && a.time.is_none() {
                    a.partial = true;
                }
            }
            _ => return None,
        }
        if let Val::DayOfWeek(d) = v {
            a.partial = true;
            a.extra = vec![(F::DayOfWeek, *d as i64)];
        }
        if let Val::Month(m) = v {
            a.partial = true;
            a.extra = vec![(F::MonthOfYear, *m as i64)];
        }
        Some(a)
    }
    fn of_any(v: &Val) -> Option<Acc> {
        match v {
            Val::DayOfWeek(d) => Some(Acc { partial: true, extra: vec![(F::DayOfWeek, *d as i64)], desc: v.to_java_string(), kind: "DayOfWeek", ..Default::default() }),
            Val::Month(m) => Some(Acc { partial: true, extra: vec![(F::MonthOfYear, *m as i64)], desc: v.to_java_string(), kind: "Month", ..Default::default() }),
            _ => Acc::of(v),
        }
    }
    fn get(&self, f: F, locale: &Locale) -> Option<i64> {
        if let Some((_, v)) = self.extra.iter().find(|(g, _)| *g == f) {
            return Some(*v);
        }
        if self.partial && self.date.is_none() && self.time.is_none() {
            return match f {
                F::OffsetSeconds => self.off.map(|o| o as i64),
                F::InstantSeconds => self.inst.map(|i| i.0),
                _ => None,
            };
        }
        match f {
            F::InstantSeconds => self.inst.map(|i| i.0),
            F::OffsetSeconds => self.off.map(|o| o as i64),
            F::NanoOfSecond | F::MicroOfSecond | F::MilliOfSecond if self.time.is_none() => {
                let n = self.inst?.1 as i64;
                Some(match f {
                    F::NanoOfSecond => n,
                    F::MicroOfSecond => n / 1000,
                    _ => n / 1_000_000,
                })
            }
            F::LocDayOfWeek | F::LocWeekOfMonth | F::LocWeekOfYear | F::LocWeekBasedYear => {
                let d = self.date?;
                let (sow, min) = locale.week_fields();
                Some(match f {
                    F::LocDayOfWeek => loc_dow(&d, sow),
                    F::LocWeekOfMonth => loc_week_of_month(&d, sow, min),
                    F::LocWeekOfYear => loc_week_of_wby(&d, sow, min),
                    _ => loc_week_based_year(&d, sow, min),
                })
            }
            f if f.is_time_based() => self.time.and_then(|t| time_get(t, f)),
            f => self.date.and_then(|d| d.get(f)),
        }
    }
    fn supports(&self, f: F, locale: &Locale) -> bool {
        self.get(f, locale).is_some()
    }
}

fn loc_dow(d: &LDate, sow: u8) -> i64 {
    (d.day_of_week() as i64 - sow as i64).rem_euclid(7) + 1
}
fn start_of_week_offset(day: i64, dow: i64, min_days: u8) -> i64 {
    let week_start = (day - dow).rem_euclid(7);
    let mut offset = -week_start;
    if week_start + 1 > min_days as i64 {
        offset = 7 - week_start;
    }
    offset
}
fn compute_week(offset: i64, day: i64) -> i64 {
    (7 + offset + (day - 1)) / 7
}
fn loc_week_of_month(d: &LDate, sow: u8, min: u8) -> i64 {
    let dow = loc_dow(d, sow);
    let dom = d.d as i64;
    compute_week(start_of_week_offset(dom, dow, min), dom)
}
fn loc_week_of_wby(d: &LDate, sow: u8, min: u8) -> i64 {
    let dow = loc_dow(d, sow);
    let doy = d.day_of_year();
    let offset = start_of_week_offset(doy, dow, min);
    let mut week = compute_week(offset, doy);
    if week == 0 {
        if let Ok(prev) = d.plus_days(-doy) {
            return loc_week_of_wby(&prev, sow, min);
        }
    } else if week > 50 {
        let new_year_week = compute_week(offset, d.length_of_year() + min as i64);
        if week >= new_year_week {
            week = week - new_year_week + 1;
        }
    }
    week
}
fn loc_week_based_year(d: &LDate, sow: u8, min: u8) -> i64 {
    let dow = loc_dow(d, sow);
    let doy = d.day_of_year();
    let offset = start_of_week_offset(doy, dow, min);
    let week = compute_week(offset, doy);
    let year = d.y as i64;
    if week == 0 {
        return year - 1;
    }
    let new_year_week = compute_week(offset, d.length_of_year() + min as i64);
    if week >= new_year_week {
        return year + 1;
    }
    year
}

// ---------------------------------------------------------------------------
// Localized names

const EN_MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const EN_DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const FR_MONTHS: [&str; 12] = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"];
const FR_MONTHS_S: [&str; 12] = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."];
const FR_DAYS: [&str; 7] = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"];
const FR_DAYS_S: [&str; 7] = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."];
const DE_MONTHS: [&str; 12] = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"];
const DE_MONTHS_S: [&str; 12] = ["Jan.", "Feb.", "März", "Apr.", "Mai", "Juni", "Juli", "Aug.", "Sept.", "Okt.", "Nov.", "Dez."];
const DE_DAYS: [&str; 7] = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"];
const DE_DAYS_S: [&str; 7] = ["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."];
const ES_MONTHS: [&str; 12] = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"];
const ES_MONTHS_S: [&str; 12] = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sept", "oct", "nov", "dic"];
const ES_DAYS: [&str; 7] = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"];
const ES_DAYS_S: [&str; 7] = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"];
const IT_MONTHS: [&str; 12] = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"];
const IT_MONTHS_S: [&str; 12] = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"];
const IT_DAYS: [&str; 7] = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"];
const IT_DAYS_S: [&str; 7] = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"];

/// Style: 0 FULL, 1 FULL_STANDALONE, 2 SHORT, 3 SHORT_STANDALONE, 4 NARROW, 5 NARROW_STANDALONE.
fn month_name(m: u8, style: u8, locale: &Locale) -> String {
    let i = (m as usize).clamp(1, 12) - 1;
    let (full, short) = match locale.lang.as_str() {
        "fr" => (FR_MONTHS[i], FR_MONTHS_S[i]),
        "de" => (DE_MONTHS[i], DE_MONTHS_S[i]),
        "es" => (ES_MONTHS[i], ES_MONTHS_S[i]),
        "it" => (IT_MONTHS[i], IT_MONTHS_S[i]),
        _ => (EN_MONTHS[i], if locale.country == "GB" && m == 9 { "Sept" } else { &EN_MONTHS[i][..3] }),
    };
    match style {
        0 | 1 => full.to_string(),
        2 | 3 => short.to_string(),
        _ => full.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_default(),
    }
}
fn day_name(d: u8, style: u8, locale: &Locale) -> String {
    let i = (d as usize).clamp(1, 7) - 1;
    let (full, short) = match locale.lang.as_str() {
        "fr" => (FR_DAYS[i], FR_DAYS_S[i]),
        "de" => (DE_DAYS[i], DE_DAYS_S[i]),
        "es" => (ES_DAYS[i], ES_DAYS_S[i]),
        "it" => (IT_DAYS[i], IT_DAYS_S[i]),
        _ => (EN_DAYS[i], &EN_DAYS[i][..3]),
    };
    match style {
        0 | 1 => full.to_string(),
        2 | 3 => short.to_string(),
        _ => full.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_default(),
    }
}
/// Display names for a text field: `(value, text)` pairs for the style.
fn text_names(f: F, style: u8, locale: &Locale) -> Vec<(i64, String)> {
    match f {
        F::MonthOfYear => (1..=12).map(|m| (m as i64, month_name(m, style, locale))).collect(),
        F::DayOfWeek | F::LocDayOfWeek => (1..=7).map(|d| (d as i64, day_name(d, style, locale))).collect(),
        F::AmPmOfDay => {
            if style >= 4 {
                vec![(0, "a".into()), (1, "p".into())]
            } else {
                vec![(0, "AM".into()), (1, "PM".into())]
            }
        }
        F::Era => match style {
            0 | 1 => vec![(0, "Before Christ".into()), (1, "Anno Domini".into())],
            2 | 3 => vec![(0, "BC".into()), (1, "AD".into())],
            _ => vec![(0, "B".into()), (1, "A".into())],
        },
        F::QuarterOfYear => (1..=4)
            .map(|q| {
                let s = match style {
                    0 | 1 => format!("{}{} quarter", q, ["st", "nd", "rd", "th"][q as usize - 1]),
                    2 | 3 => format!("Q{q}"),
                    _ => q.to_string(),
                };
                (q as i64, s)
            })
            .collect(),
        _ => Vec::new(),
    }
}
fn day_period(nod: i64) -> &'static str {
    let h = nod / (3600 * NPS);
    if nod == 0 {
        "midnight"
    } else if nod == 12 * 3600 * NPS {
        "noon"
    } else if h >= 21 {
        "at night"
    } else if h >= 18 {
        "in the evening"
    } else if h >= 12 {
        "in the afternoon"
    } else {
        "in the morning"
    }
}

// ---------------------------------------------------------------------------
// DateTimeFormatter

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Normal,
    Always,
    #[allow(dead_code)]
    Never,
    NotNegative,
    ExceedsPad,
}
impl Sign {
    fn name(&self) -> &'static str {
        match self {
            Sign::Normal => "NORMAL",
            Sign::Always => "ALWAYS",
            Sign::Never => "NEVER",
            Sign::NotNegative => "NOT_NEGATIVE",
            Sign::ExceedsPad => "EXCEEDS_PAD",
        }
    }
    /// Java `SignStyle.parse`.
    fn allows(&self, positive: bool, strict: bool) -> bool {
        match self {
            Sign::Normal => !positive || !strict,
            Sign::Always | Sign::ExceedsPad => true,
            _ => !strict,
        }
    }
}

#[derive(Clone, Debug)]
enum El {
    Lit(String),
    /// `sub`: adjacent-value reserved width (-1 = fixed width).
    Num { f: F, min: u8, max: u8, sign: Sign, sub: i8 },
    Reduced { f: F, width: u8, base: i64, sub: i8 },
    Text { f: F, style: u8 },
    /// RFC-1123 style English short names (`Text(Field)`).
    TextMap(F),
    Frac { min: u8, max: u8, point: bool },
    Offset { pat: &'static str, no_off: &'static str },
    LocOffset(bool),
    ZoneId { region_only: bool },
    ZoneText(bool),
    GenericZone(bool),
    Instant,
    DayPeriod,
    Opt(Vec<El>),
    Group(Vec<El>),
    CaseSensitive(bool),
    Strict(bool),
    Localized(Option<u8>, Option<u8>, Vec<El>),
}

#[derive(Debug)]
pub struct Fmt {
    els: Vec<El>,
    locale: Locale,
    zone: Option<Zone>,
    /// 0 STRICT, 1 SMART, 2 LENIENT.
    resolver: u8,
}

fn describe_els(els: &[El], out: &mut String) {
    for el in els {
        match el {
            El::Lit(s) => {
                if s == "'" {
                    out.push_str("''");
                } else {
                    let _ = write!(out, "'{}'", s.replace('\'', "''"));
                }
            }
            El::Num { f, min, max, sign, .. } => {
                if *min == 1 && *max == 19 && *sign == Sign::Normal {
                    let _ = write!(out, "Value({})", f.name());
                } else if min == max && *sign == Sign::NotNegative {
                    let _ = write!(out, "Value({},{min})", f.name());
                } else {
                    let _ = write!(out, "Value({},{min},{max},{})", f.name(), sign.name());
                }
            }
            El::Reduced { f, width, base, .. } => {
                let _ = write!(out, "ReducedValue({},{width},{width},{base}-01-01)", f.name());
            }
            El::Text { f, style } => {
                if *style == 0 {
                    let _ = write!(out, "Text({})", f.name());
                } else {
                    let _ = write!(out, "Text({},{})", f.name(), TEXT_STYLES[*style as usize]);
                }
            }
            El::TextMap(f) => {
                let _ = write!(out, "Text({})", f.name());
            }
            El::Frac { min, max, point } => {
                let _ = write!(out, "Fraction(NanoOfSecond,{min},{max}{})", if *point { ",DecimalPoint" } else { "" });
            }
            El::Offset { pat, no_off } => {
                let _ = write!(out, "Offset({pat},'{no_off}')");
            }
            El::LocOffset(full) => {
                let _ = write!(out, "LocalizedOffset({})", if *full { "FULL" } else { "SHORT" });
            }
            El::ZoneId { region_only } => out.push_str(if *region_only { "ZoneRegionId()" } else { "ZoneId()" }),
            El::ZoneText(full) => {
                let _ = write!(out, "ZoneText({})", if *full { "FULL" } else { "SHORT" });
            }
            El::GenericZone(full) => {
                let _ = write!(out, "ZoneText({})", if *full { "FULL" } else { "SHORT" });
            }
            El::Instant => out.push_str("Instant()"),
            El::DayPeriod => out.push_str("DayPeriod(SHORT)"),
            El::Opt(inner) => {
                out.push('[');
                describe_els(inner, out);
                out.push(']');
            }
            El::Group(inner) => {
                out.push('(');
                describe_els(inner, out);
                out.push(')');
            }
            El::CaseSensitive(b) => {
                let _ = write!(out, "ParseCaseSensitive({b})");
            }
            El::Strict(b) => {
                let _ = write!(out, "ParseStrict({b})");
            }
            El::Localized(d, t, _) => {
                let _ = write!(out, "Localized({},{})", d.map(|s| FORMAT_STYLES[s as usize]).unwrap_or(""), t.map(|s| FORMAT_STYLES[s as usize]).unwrap_or(""));
            }
        }
    }
}

/// Pattern compiler state (one per optional-section nesting level).
struct Builder {
    els: Vec<El>,
    active: Option<usize>,
}
impl Builder {
    fn push(&mut self, el: El) {
        self.els.push(el);
        self.active = None;
    }
    /// Java `appendValue(NumberPrinterParser)` adjacency rules.
    fn push_value(&mut self, el: El) {
        let (min, max, sign) = match &el {
            El::Num { min, max, sign, .. } => (*min, *max, *sign),
            El::Reduced { width, .. } => (*width, *width, Sign::NotNegative),
            _ => (0, 0, Sign::Normal),
        };
        match self.active {
            Some(i) => {
                let fixed_new = min == max && sign == Sign::NotNegative;
                if fixed_new {
                    if let Some(El::Num { sub, .. } | El::Reduced { sub, .. }) = self.els.get_mut(i) {
                        if *sub >= 0 {
                            *sub = (*sub as i32 + max as i32).min(127) as i8;
                        }
                    }
                    let mut el = el;
                    if let El::Num { sub, .. } | El::Reduced { sub, .. } = &mut el {
                        *sub = -1;
                    }
                    self.els.push(el);
                } else {
                    if let Some(El::Num { sub, .. } | El::Reduced { sub, .. }) = self.els.get_mut(i) {
                        *sub = -1;
                    }
                    self.els.push(el);
                    self.active = Some(self.els.len() - 1);
                }
            }
            None => {
                self.els.push(el);
                self.active = Some(self.els.len() - 1);
            }
        }
    }
}

fn num(f: F, min: u8, max: u8, sign: Sign) -> El {
    El::Num { f, min, max, sign, sub: 0 }
}
fn fixed(f: F, w: u8) -> El {
    El::Num { f, min: w, max: w, sign: Sign::NotNegative, sub: 0 }
}

impl Fmt {
    fn new(els: Vec<El>) -> Fmt {
        Fmt { els, locale: Locale::en(), zone: None, resolver: 1 }
    }
    fn describe(&self) -> String {
        let mut s = String::new();
        describe_els(&self.els, &mut s);
        s
    }

    /// `DateTimeFormatter.ofPattern`.
    fn of_pattern(p: &str, locale: Locale) -> SResult<Fmt> {
        Fmt::of_pattern0(p, locale).map_err(|e| {
            let dtfb = "java.time.format.DateTimeFormatterBuilder";
            let m = e.msg.as_str();
            let inner: Vec<Fr> = if m.starts_with("Unknown pattern letter") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 2020)]
            } else if m.starts_with("Too many pattern letters") {
                let letter = m.chars().last().unwrap_or(' ');
                let line = if "MLQqEecGaBz".contains(letter) { jf!("java.time.format.DateTimeFormatterBuilder", "parseField", "DateTimeFormatterBuilder", 2118) } else { jf!("java.time.format.DateTimeFormatterBuilder", "parseField", "DateTimeFormatterBuilder", 2158) };
                vec![line, DTFB_PARSE_PATTERN_FIELD]
            } else if m.starts_with("Invalid pattern") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parseField", "DateTimeFormatterBuilder", 2082), DTFB_PARSE_PATTERN_FIELD]
            } else if m.starts_with("Pattern includes reserved character") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 2056)]
            } else if m.starts_with("Pattern invalid as it contains ]") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 2051)]
            } else if m.starts_with("Pattern ends with an incomplete") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 2037)]
            } else if m.starts_with("Pattern letter count must be 2") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 1953)]
            } else if m.starts_with("Pattern letter count must be 1 or 4") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 1980)]
            } else if m.starts_with("Wrong number of pattern letters") {
                vec![jf!("java.time.format.DateTimeFormatterBuilder", "parsePattern", "DateTimeFormatterBuilder", 1962)]
            } else {
                let _ = dtfb;
                Vec::new()
            };
            let mut v = inner;
            v.push(DTFB_APPEND_PATTERN);
            frs(e, &v)
        })
    }

    fn of_pattern0(p: &str, locale: Locale) -> SResult<Fmt> {
        let chars: Vec<char> = p.chars().collect();
        let mut stack: Vec<Builder> = vec![Builder { els: Vec::new(), active: None }];
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_ascii_alphabetic() {
                let mut n = 1;
                while chars.get(i + n) == Some(&c) {
                    n += 1;
                }
                i += n;
                let b = stack.last_mut().unwrap();
                let n = n as u8;
                let too_many = || illegal_arg(format!("Too many pattern letters: {c}"));
                match c {
                    'u' | 'y' | 'Y' => {
                        let f = match c {
                            'u' => F::Year,
                            'y' => F::YearOfEra,
                            _ => F::LocWeekBasedYear,
                        };
                        if n == 2 {
                            b.push_value(El::Reduced { f, width: 2, base: 2000, sub: 0 });
                        } else if n < 4 {
                            b.push_value(num(f, n, 19, Sign::Normal));
                        } else {
                            b.push_value(num(f, n, 19, Sign::ExceedsPad));
                        }
                    }
                    'M' | 'L' | 'Q' | 'q' => {
                        let f = if c == 'M' || c == 'L' { F::MonthOfYear } else { F::QuarterOfYear };
                        let standalone = c == 'L' || c == 'q';
                        match n {
                            1 => b.push_value(num(f, 1, 19, Sign::Normal)),
                            2 => b.push_value(fixed(f, 2)),
                            3 => b.push(El::Text { f, style: if standalone { 3 } else { 2 } }),
                            4 => b.push(El::Text { f, style: if standalone { 1 } else { 0 } }),
                            5 => b.push(El::Text { f, style: if standalone { 5 } else { 4 } }),
                            _ => return Err(too_many()),
                        }
                    }
                    'E' => match n {
                        1..=3 => b.push(El::Text { f: F::DayOfWeek, style: 2 }),
                        4 => b.push(El::Text { f: F::DayOfWeek, style: 0 }),
                        5 => b.push(El::Text { f: F::DayOfWeek, style: 4 }),
                        _ => return Err(too_many()),
                    },
                    'e' | 'c' => match n {
                        1 => b.push_value(num(F::LocDayOfWeek, 1, 19, Sign::Normal)),
                        2 if c == 'e' => b.push_value(fixed(F::LocDayOfWeek, 2)),
                        2 => return Err(illegal_arg("Invalid pattern \"cc\"")),
                        3 => b.push(El::Text { f: F::DayOfWeek, style: if c == 'c' { 3 } else { 2 } }),
                        4 => b.push(El::Text { f: F::DayOfWeek, style: if c == 'c' { 1 } else { 0 } }),
                        5 => b.push(El::Text { f: F::DayOfWeek, style: if c == 'c' { 5 } else { 4 } }),
                        _ => return Err(too_many()),
                    },
                    'a' => {
                        if n == 1 {
                            b.push(El::Text { f: F::AmPmOfDay, style: 2 });
                        } else {
                            return Err(too_many());
                        }
                    }
                    'B' => match n {
                        1 | 4 | 5 => b.push(El::DayPeriod),
                        _ => return Err(too_many()),
                    },
                    'G' => match n {
                        1..=3 => b.push(El::Text { f: F::Era, style: 2 }),
                        4 => b.push(El::Text { f: F::Era, style: 0 }),
                        5 => b.push(El::Text { f: F::Era, style: 4 }),
                        _ => return Err(too_many()),
                    },
                    'S' => b.push(El::Frac { min: n, max: n, point: false }),
                    'F' => {
                        if n == 1 {
                            b.push_value(num(F::AlignedDayOfWeekInMonth, 1, 19, Sign::Normal));
                        } else {
                            return Err(too_many());
                        }
                    }
                    'd' | 'h' | 'H' | 'k' | 'K' | 'm' | 's' => {
                        let f = match c {
                            'd' => F::DayOfMonth,
                            'h' => F::ClockHourOfAmPm,
                            'H' => F::HourOfDay,
                            'k' => F::ClockHourOfDay,
                            'K' => F::HourOfAmPm,
                            'm' => F::MinuteOfHour,
                            _ => F::SecondOfMinute,
                        };
                        match n {
                            1 => b.push_value(num(f, 1, 19, Sign::Normal)),
                            2 => b.push_value(fixed(f, 2)),
                            _ => return Err(too_many()),
                        }
                    }
                    'D' => match n {
                        1 => b.push_value(num(F::DayOfYear, 1, 19, Sign::Normal)),
                        2 | 3 => b.push_value(num(F::DayOfYear, n, 3, Sign::NotNegative)),
                        _ => return Err(too_many()),
                    },
                    'A' | 'N' | 'n' | 'g' => {
                        let f = match c {
                            'A' => F::MilliOfDay,
                            'N' => F::NanoOfDay,
                            'n' => F::NanoOfSecond,
                            _ => F::EpochDay,
                        };
                        b.push_value(num(f, n, 19, Sign::NotNegative));
                    }
                    'w' => match n {
                        1 => b.push_value(num(F::LocWeekOfYear, 1, 2, Sign::NotNegative)),
                        2 => b.push_value(fixed(F::LocWeekOfYear, 2)),
                        _ => return Err(too_many()),
                    },
                    'W' => {
                        if n == 1 {
                            b.push_value(num(F::LocWeekOfMonth, 1, 1, Sign::NotNegative));
                        } else {
                            return Err(too_many());
                        }
                    }
                    'z' => match n {
                        1..=3 => b.push(El::ZoneText(false)),
                        4 => b.push(El::ZoneText(true)),
                        _ => return Err(too_many()),
                    },
                    'v' => match n {
                        1 => b.push(El::GenericZone(false)),
                        4 => b.push(El::GenericZone(true)),
                        _ => return Err(illegal_arg(format!("Wrong number of pattern letters: {c}"))),
                    },
                    'V' => {
                        if n != 2 {
                            return Err(illegal_arg(format!("Pattern letter count must be 2: {c}")));
                        }
                        b.push(El::ZoneId { region_only: false });
                    }
                    'Z' => match n {
                        1..=3 => b.push(El::Offset { pat: "+HHMM", no_off: "+0000" }),
                        4 => b.push(El::LocOffset(true)),
                        5 => b.push(El::Offset { pat: "+HH:MM:ss", no_off: "Z" }),
                        _ => return Err(too_many()),
                    },
                    'O' => match n {
                        1 => b.push(El::LocOffset(false)),
                        4 => b.push(El::LocOffset(true)),
                        _ => return Err(illegal_arg(format!("Pattern letter count must be 1 or 4: {c}"))),
                    },
                    'X' | 'x' => {
                        let pats = ["+HHmm", "+HHMM", "+HH:MM", "+HHMMss", "+HH:MM:ss"];
                        let no_x = ["+00", "+0000", "+00:00", "+0000", "+00:00"];
                        if n > 5 {
                            return Err(too_many());
                        }
                        b.push(El::Offset { pat: pats[n as usize - 1], no_off: if c == 'X' { "Z" } else { no_x[n as usize - 1] } });
                    }
                    'p' => return Err(illegal_arg("Pad letter 'p' must be followed by valid pad pattern")),
                    other => return Err(illegal_arg(format!("Unknown pattern letter: {other}"))),
                }
                continue;
            }
            match c {
                '\'' => {
                    let start = i;
                    i += 1;
                    let mut lit = String::new();
                    loop {
                        match chars.get(i) {
                            None => return Err(illegal_arg(format!("Pattern ends with an incomplete string literal: {p}"))),
                            Some('\'') => {
                                if chars.get(i + 1) == Some(&'\'') {
                                    lit.push('\'');
                                    i += 2;
                                } else {
                                    i += 1;
                                    break;
                                }
                            }
                            Some(ch) => {
                                lit.push(*ch);
                                i += 1;
                            }
                        }
                    }
                    let b = stack.last_mut().unwrap();
                    if lit.is_empty() {
                        if i - start == 2 {
                            b.push(El::Lit("'".into()));
                        }
                    } else {
                        b.push(El::Lit(lit));
                    }
                }
                '[' => {
                    stack.push(Builder { els: Vec::new(), active: None });
                    i += 1;
                }
                ']' => {
                    if stack.len() == 1 {
                        return Err(illegal_arg("Pattern invalid as it contains ] without previous ["));
                    }
                    let inner = stack.pop().unwrap();
                    stack.last_mut().unwrap().push(El::Opt(inner.els));
                    i += 1;
                }
                '{' | '}' | '#' => return Err(illegal_arg(format!("Pattern includes reserved character: '{c}'"))),
                other => {
                    stack.last_mut().unwrap().push(El::Lit(other.to_string()));
                    i += 1;
                }
            }
        }
        // Unclosed optional sections are closed implicitly (Java: toFormatter closes them).
        while stack.len() > 1 {
            let inner = stack.pop().unwrap();
            stack.last_mut().unwrap().push(El::Opt(inner.els));
        }
        let mut f = Fmt::new(stack.pop().unwrap().els);
        f.locale = locale;
        Ok(f)
    }

    /// Localized `FormatStyle` formatter (English CLDR patterns).
    fn localized(date: Option<u8>, time: Option<u8>) -> SResult<Fmt> {
        let dp = date.map(|s| match s {
            0 => "EEEE, MMMM d, y",
            1 => "MMMM d, y",
            2 => "MMM d, y",
            _ => "M/d/yy",
        });
        let tp = time.map(|s| match s {
            0 => "h:mm:ss\u{202f}a zzzz",
            1 => "h:mm:ss\u{202f}a z",
            2 => "h:mm:ss\u{202f}a",
            _ => "h:mm\u{202f}a",
        });
        let pat = match (dp, tp) {
            (Some(d), Some(t)) => {
                if matches!(date, Some(0) | Some(1)) {
                    format!("{d} 'at' {t}")
                } else {
                    format!("{d}, {t}")
                }
            }
            (Some(d), None) => d.to_string(),
            (None, Some(t)) => t.to_string(),
            (None, None) => String::new(),
        };
        let inner = Fmt::of_pattern(&pat, Locale::en())?;
        Ok(Fmt::new(vec![El::Localized(date, time, inner.els)]))
    }

    fn with_locale(&self, l: Locale) -> Fmt {
        Fmt { els: self.els.clone(), locale: l, zone: self.zone.clone(), resolver: self.resolver }
    }
    fn with_zone(&self, z: Option<Zone>) -> Fmt {
        Fmt { els: self.els.clone(), locale: self.locale.clone(), zone: z, resolver: self.resolver }
    }
    fn with_resolver(&self, r: u8) -> Fmt {
        Fmt { els: self.els.clone(), locale: self.locale.clone(), zone: self.zone.clone(), resolver: r }
    }

    // ---- formatting -------------------------------------------------------

    fn format(&self, v: &Val) -> SResult<String> {
        if v.is_null() {
            return Err(npe_req("temporal", &[jf!("java.time.format.DateTimeFormatter", "format", "DateTimeFormatter", 1900)]));
        }
        let mut acc = Acc::of_any(v).ok_or_else(|| cast_to(v, "java.time.temporal.TemporalAccessor"))?;
        // Override zone: instants are converted to that zone.
        if let Some(z) = &self.zone {
            if let Some(i) = acc.inst {
                let zd = Zoned::of_instant(i, z.clone())?;
                acc.date = Some(zd.ldt.date);
                acc.time = Some(zd.ldt.nod);
                acc.off = Some(zd.off);
                acc.zone = Some(zd.zone);
            } else if acc.zone.is_none() {
                acc.zone = Some(z.clone());
            }
        }
        let mut out = String::new();
        let iso = self.resolver == 0;
        let wrapped = iso && acc.date.is_none() && !acc.partial;
        self.format_els(&self.els, &acc, &mut out, false).map_err(|e| {
            if e.frames.is_empty() {
                return e;
            }
            // Receiver `getLong` chain, print-context frame(s), then the formatter tail.
            let mut v: Vec<Fr> = Vec::new();
            let zone_err = e.msg.starts_with("Unable to extract ZoneId");
            if !zone_err {
                let field_name = e.msg.rsplit(' ').next().unwrap_or("");
                let time_based = CHRONO_FIELDS.iter().find(|f| f.name() == field_name).is_some_and(|f| f.is_time_based());
                v.extend_from_slice(&getlong_chain(acc.kind, time_based));
                if wrapped {
                    v.push(DTPC_WRAPPED);
                }
                v.push(DTPC_GET_VALUE);
            } else {
                v.push(DTPC_GET_ZONE);
            }
            v.extend_from_slice(e.frames);
            v.extend_from_slice(&DTF_FORMAT_TAIL);
            frs(e, &v)
        })?;
        Ok(out)
    }

    fn format_els(&self, els: &[El], acc: &Acc, out: &mut String, optional: bool) -> SResult<()> {
        let get = |f: F| -> SResult<i64> { acc.get(f, &self.locale).ok_or_else(|| frs(unsupported_field(f), &[])) };
        let pp = |e: Box<ScriptErr>, frame: Fr| frs(e, &[frame]);
        const PP_NUM: Fr = jf!("java.time.format.DateTimeFormatterBuilder$NumberPrinterParser", "format", "DateTimeFormatterBuilder", 2920);
        const PP_TEXT: Fr = jf!("java.time.format.DateTimeFormatterBuilder$TextPrinterParser", "format", "DateTimeFormatterBuilder", 3712);
        const PP_NANOS: Fr = jf!("java.time.format.DateTimeFormatterBuilder$NanosPrinterParser", "format", "DateTimeFormatterBuilder", 3375);
        const PP_FRACTION: Fr = jf!("java.time.format.DateTimeFormatterBuilder$FractionPrinterParser", "format", "DateTimeFormatterBuilder", 3488);
        const PP_OFFSET: Fr = jf!("java.time.format.DateTimeFormatterBuilder$OffsetIdPrinterParser", "format", "DateTimeFormatterBuilder", 3993);
        const PP_LOC_OFFSET: Fr = jf!("java.time.format.DateTimeFormatterBuilder$LocalizedOffsetIdPrinterParser", "format", "DateTimeFormatterBuilder", 4292);
        const PP_ZONE_ID: Fr = jf!("java.time.format.DateTimeFormatterBuilder$ZoneIdPrinterParser", "format", "DateTimeFormatterBuilder", 4632);
        const PP_ZONE_TEXT: Fr = jf!("java.time.format.DateTimeFormatterBuilder$ZoneTextPrinterParser", "format", "DateTimeFormatterBuilder", 4506);
        const PP_INSTANT: Fr = jf!("java.time.format.DateTimeFormatterBuilder$InstantPrinterParser", "format", "DateTimeFormatterBuilder", 3808);
        const PP_DAY_PERIOD: Fr = jf!("java.time.format.DateTimeFormatterBuilder$DayPeriodPrinterParser", "format", "DateTimeFormatterBuilder", 5369);
        for el in els {
            match el {
                El::Lit(s) => out.push_str(s),
                El::Num { f, min, max, sign, .. } => {
                    let v = get(*f).map_err(|e| pp(e, PP_NUM))?;
                    push_number(out, *f, v, *min, *max, *sign).map_err(|e| pp(e, PP_NUM))?;
                }
                El::Reduced { f, width, base, .. } => {
                    let v = get(*f).map_err(|e| pp(e, PP_NUM))?;
                    let range = 10i64.pow(*width as u32);
                    let abs = v.abs();
                    let shown = if v >= *base && v < base + range { abs % range } else { abs % range };
                    let _ = write!(out, "{:0w$}", shown, w = *width as usize);
                }
                El::Text { f, style } => {
                    let v = get(*f).map_err(|e| pp(e, PP_TEXT))?;
                    match text_names(*f, *style, &self.locale).into_iter().find(|(k, _)| *k == v) {
                        Some((_, s)) => out.push_str(&s),
                        None => {
                            let _ = write!(out, "{v}");
                        }
                    }
                }
                El::TextMap(f) => {
                    let v = get(*f).map_err(|e| pp(e, PP_TEXT))?;
                    match text_names(*f, 2, &Locale::en()).into_iter().find(|(k, _)| *k == v) {
                        Some((_, s)) => out.push_str(&s),
                        None => {
                            let _ = write!(out, "{v}");
                        }
                    }
                }
                El::Frac { min, max, point } => {
                    let v = get(F::NanoOfSecond).map_err(|e| pp(e, if *point { PP_FRACTION } else { PP_NANOS }))?;
                    let digits = format!("{v:09}");
                    let trimmed = digits.trim_end_matches('0');
                    if trimmed.is_empty() {
                        if *min > 0 {
                            if *point {
                                out.push('.');
                            }
                            out.push_str(&"0".repeat(*min as usize));
                        }
                    } else {
                        let scale = trimmed.len().max(*min as usize).min(*max as usize);
                        if *point {
                            out.push('.');
                        }
                        out.push_str(&digits[..scale]);
                    }
                }
                El::Offset { pat, no_off } => {
                    let v = get(F::OffsetSeconds).map_err(|e| pp(e, PP_OFFSET))?;
                    out.push_str(&format_offset(v as i32, pat, no_off));
                }
                El::LocOffset(full) => {
                    let v = get(F::OffsetSeconds).map_err(|e| pp(e, PP_LOC_OFFSET))? as i32;
                    out.push_str("GMT");
                    out.push_str(&loc_offset(v, *full));
                }
                El::ZoneId { .. } => {
                    let z = acc.zone.clone().or_else(|| acc.off.map(Zone::Offset)).ok_or_else(|| frs(dte(format!("Unable to extract ZoneId from temporal {}", acc.desc)), &[PP_ZONE_ID]))?;
                    out.push_str(&z.id());
                }
                El::ZoneText(full) => {
                    let z = acc.zone.clone().or_else(|| acc.off.map(Zone::Offset)).ok_or_else(|| frs(dte(format!("Unable to extract ZoneId from temporal {}", acc.desc)), &[PP_ZONE_TEXT]))?;
                    let secs = acc.inst.map(|i| i.0).unwrap_or_else(|| crate::date::now_millis() / 1000);
                    out.push_str(&zone_text(&z, secs, *full));
                }
                El::GenericZone(full) => {
                    let z = acc.zone.clone().or_else(|| acc.off.map(Zone::Offset)).ok_or_else(|| frs(dte(format!("Unable to extract ZoneId from temporal {}", acc.desc)), &[PP_ZONE_TEXT]))?;
                    out.push_str(&z.display_name(*full));
                }
                El::Instant => {
                    let secs = get(F::InstantSeconds).map_err(|e| pp(e, PP_INSTANT))?;
                    let nanos = acc.get(F::NanoOfSecond, &self.locale).unwrap_or(0);
                    inst_write(out, (secs, nanos as u32));
                }
                El::DayPeriod => {
                    let t = acc.time.ok_or_else(|| frs(unsupported_field(F::HourOfDay), &[PP_DAY_PERIOD]))?;
                    out.push_str(day_period(t));
                }
                El::Opt(inner) => {
                    let mut tmp = String::new();
                    if self.format_els(inner, acc, &mut tmp, true).is_ok() {
                        out.push_str(&tmp);
                    }
                }
                El::Group(inner) | El::Localized(_, _, inner) => self.format_els(inner, acc, out, optional)?,
                El::CaseSensitive(_) | El::Strict(_) => {}
            }
        }
        Ok(())
    }

    // ---- parsing ----------------------------------------------------------

    /// Parse the whole text into an unresolved field set.
    fn parse_unresolved(&self, text: &str) -> SResult<PState> {
        let chars: Vec<char> = text.chars().collect();
        let mut st = PState { fields: Vec::new(), zone: None, leap_second: false, ci: false, strict: true, locale: self.locale.clone() };
        let abbr = || if text.chars().count() > 64 { format!("{}...", text.chars().take(64).collect::<String>()) } else { text.to_string() };
        match self.parse_els(&self.els, &chars, 0, &mut st) {
            Ok(pos) => {
                if pos < chars.len() {
                    return Err(frs(dtpe(format!("Text '{}' could not be parsed, unparsed text found at index {pos}", abbr())), &[DTF_PARSE_UNPARSED]));
                }
            }
            Err(pos) => return Err(frs(dtpe(format!("Text '{}' could not be parsed at index {pos}", abbr())), &[DTF_PARSE_RESOLVED])),
        }
        Ok(st)
    }

    /// Parse and resolve (`DateTimeFormatter.parse(text)`).
    fn parse(&self, text: &str) -> SResult<Parsed> {
        let st = self.parse_unresolved(text)?;
        let abbr = if text.chars().count() > 64 { format!("{}...", text.chars().take(64).collect::<String>()) } else { text.to_string() };
        resolve(st, self.zone.clone(), self.resolver).map_err(|e| frs(dtpe(format!("Text '{abbr}' could not be parsed: {}", e.msg)), &[DTF_CREATE_ERROR]))
    }

    /// `X.parse(text, formatter)`.
    fn parse_to(&self, text: &str, target: &str) -> SResult<Val> {
        let p = self.parse(text)?;
        let abbr = if text.chars().count() > 64 { format!("{}...", text.chars().take(64).collect::<String>()) } else { text.to_string() };
        from_parsed(&p, target).map_err(|e| if e.kind == "date_time_parse_exception" { e } else { frs(dtpe(format!("Text '{abbr}' could not be parsed: {}", e.msg)), &[DTF_CREATE_ERROR]) })
    }

    fn parse_els(&self, els: &[El], text: &[char], mut pos: usize, st: &mut PState) -> Result<usize, usize> {
        let mut i = 0;
        while i < els.len() {
            let el = &els[i];
            i += 1;
            match el {
                El::Lit(s) => {
                    for c in s.chars() {
                        match text.get(pos) {
                            Some(t) if *t == c || (st.ci && t.to_lowercase().eq(c.to_lowercase())) => pos += 1,
                            _ => return Err(pos),
                        }
                    }
                }
                El::Num { f, min, max, sign, sub } => {
                    let (p, v) = parse_number(text, pos, *min, *max, *sign, *sub, st.strict)?;
                    st.set(*f, v);
                    pos = p;
                }
                El::Reduced { f, width, base, sub } => {
                    let (p, v) = parse_number(text, pos, *width, *width, Sign::NotNegative, *sub, st.strict)?;
                    let range = 10i64.pow(*width as u32);
                    let last = base % range;
                    let base_part = base - last;
                    let mut val = if *base > 0 { base_part + v } else { base_part - v };
                    if val < *base {
                        val += range;
                    }
                    st.set(*f, val);
                    pos = p;
                }
                El::Text { f, style } => {
                    let mut names = text_names(*f, *style, &st.locale);
                    if !st.strict {
                        for s in 0..6u8 {
                            if s != *style {
                                names.extend(text_names(*f, s, &st.locale));
                            }
                        }
                    }
                    let (p, v) = match_text(text, pos, &names, st.ci).ok_or(pos)?;
                    st.set(*f, v);
                    pos = p;
                }
                El::TextMap(f) => {
                    let mut names = text_names(*f, 2, &Locale::en());
                    names.extend(text_names(*f, 0, &Locale::en()));
                    let (p, v) = match_text(text, pos, &names, st.ci).ok_or(pos)?;
                    st.set(*f, v);
                    pos = p;
                }
                El::Frac { min, max, point } => {
                    let start = pos;
                    let eff_min = if st.strict { *min as usize } else { 0 };
                    if *point {
                        if text.get(pos) == Some(&'.') {
                            pos += 1;
                        } else if eff_min > 0 {
                            return Err(start);
                        } else {
                            continue;
                        }
                    }
                    let mut digits = String::new();
                    while digits.len() < *max as usize {
                        match text.get(pos) {
                            Some(c) if c.is_ascii_digit() => {
                                digits.push(*c);
                                pos += 1;
                            }
                            _ => break,
                        }
                    }
                    if digits.len() < eff_min {
                        return Err(start);
                    }
                    let mut nanos = 0i64;
                    for (k, c) in digits.chars().enumerate() {
                        nanos += (c as i64 - '0' as i64) * 10i64.pow(8 - k as u32);
                    }
                    if !digits.is_empty() || *point {
                        st.set(F::NanoOfSecond, nanos);
                    }
                }
                El::Offset { pat, no_off } => {
                    let (p, v) = parse_offset(text, pos, pat, no_off, !st.strict).ok_or(pos)?;
                    st.set(F::OffsetSeconds, v as i64);
                    pos = p;
                }
                El::LocOffset(full) => {
                    let (p, v) = parse_loc_offset(text, pos, *full).ok_or(pos)?;
                    st.set(F::OffsetSeconds, v as i64);
                    pos = p;
                }
                El::ZoneId { region_only } => {
                    let (p, z) = parse_zone(text, pos, *region_only).ok_or(pos)?;
                    st.zone = Some(z);
                    pos = p;
                }
                El::ZoneText(_) | El::GenericZone(_) => {
                    let (p, z) = parse_zone(text, pos, false).ok_or(pos)?;
                    st.zone = Some(z);
                    pos = p;
                }
                El::Instant => {
                    pos = parse_instant(text, pos, st)?;
                }
                El::DayPeriod => return Err(pos),
                El::Opt(inner) => {
                    let saved = st.snapshot();
                    match self.parse_els(inner, text, pos, st) {
                        Ok(p) => pos = p,
                        Err(_) => st.restore(saved),
                    }
                }
                El::Group(inner) | El::Localized(_, _, inner) => {
                    let (ci, strict) = (st.ci, st.strict);
                    pos = self.parse_els(inner, text, pos, st)?;
                    st.ci = ci;
                    st.strict = strict;
                }
                El::CaseSensitive(b) => st.ci = !*b,
                El::Strict(b) => st.strict = *b,
            }
        }
        Ok(pos)
    }
}

struct PState {
    fields: Vec<(F, i64)>,
    zone: Option<Zone>,
    leap_second: bool,
    ci: bool,
    strict: bool,
    locale: Locale,
}
impl PState {
    fn set(&mut self, f: F, v: i64) {
        if let Some(e) = self.fields.iter_mut().find(|(g, _)| *g == f) {
            e.1 = v;
        } else {
            self.fields.push((f, v));
        }
    }
    fn snapshot(&self) -> (usize, Option<Zone>, bool, bool, bool) {
        (self.fields.len(), self.zone.clone(), self.leap_second, self.ci, self.strict)
    }
    fn restore(&mut self, s: (usize, Option<Zone>, bool, bool, bool)) {
        self.fields.truncate(s.0);
        self.zone = s.1;
        self.leap_second = s.2;
        self.ci = s.3;
        self.strict = s.4;
    }
}

fn push_number(out: &mut String, f: F, v: i64, min: u8, max: u8, sign: Sign) -> SResult<()> {
    let s = v.unsigned_abs().to_string();
    if s.len() > max as usize {
        return Err(dte(format!("Field {} cannot be printed as the value {v} exceeds the maximum print width of {max}", f.name())));
    }
    if v >= 0 {
        match sign {
            Sign::ExceedsPad => {
                if min < 19 && v >= 10i64.pow(min as u32) {
                    out.push('+');
                }
            }
            Sign::Always => out.push('+'),
            _ => {}
        }
    } else {
        match sign {
            Sign::Normal | Sign::ExceedsPad | Sign::Always => out.push('-'),
            Sign::NotNegative => return Err(dte(format!("Field {} cannot be printed as the value {v} cannot be negative according to the SignStyle", f.name()))),
            Sign::Never => {}
        }
    }
    for _ in s.len()..min as usize {
        out.push('0');
    }
    out.push_str(&s);
    Ok(())
}

/// Java `NumberPrinterParser.parse`.
fn parse_number(text: &[char], pos: usize, min: u8, max: u8, sign: Sign, sub: i8, strict: bool) -> Result<(usize, i64), usize> {
    let start = pos;
    let mut p = pos;
    let (mut negative, mut positive) = (false, false);
    match text.get(p) {
        None => return Err(start),
        Some('+') => {
            if !sign.allows(true, strict) {
                return Err(start);
            }
            positive = true;
            p += 1;
        }
        Some('-') => {
            if !sign.allows(false, strict) {
                return Err(start);
            }
            negative = true;
            p += 1;
        }
        Some(_) => {
            if sign == Sign::Always && strict {
                return Err(start);
            }
        }
    }
    let fixed = sub == -1 || (sub > 0 && min == max && sign == Sign::NotNegative);
    let mut eff_max = (if fixed || strict { max as usize } else { 9 }) + sub.max(0) as usize;
    let digits_from = p;
    let parse = |eff_max: usize| -> (usize, i64) {
        let mut q = digits_from;
        let mut v: i64 = 0;
        while q < text.len() && q - digits_from < eff_max {
            match text[q] {
                c if c.is_ascii_digit() => {
                    v = v.saturating_mul(10).saturating_add(c as i64 - '0' as i64);
                    q += 1;
                }
                _ => break,
            }
        }
        (q, v)
    };
    let (mut q, mut v) = parse(eff_max);
    if q - digits_from < min as usize {
        return Err(start);
    }
    if sub > 0 && !fixed {
        let parse_len = q - digits_from;
        eff_max = (min as usize).max(parse_len.saturating_sub(sub as usize));
        let r = parse(eff_max);
        q = r.0;
        v = r.1;
        if q - digits_from < min as usize {
            return Err(start);
        }
    }
    if sign == Sign::ExceedsPad && strict {
        let parse_len = q - digits_from;
        if positive {
            if parse_len <= min as usize {
                return Err(start);
            }
        } else if parse_len > min as usize {
            return Err(start);
        }
    }
    Ok((q, if negative { -v } else { v }))
}

fn match_text(text: &[char], pos: usize, names: &[(i64, String)], ci: bool) -> Option<(usize, i64)> {
    let mut best: Option<(usize, i64)> = None;
    for (v, name) in names {
        let nc: Vec<char> = name.chars().collect();
        if pos + nc.len() > text.len() {
            continue;
        }
        let ok = nc.iter().enumerate().all(|(k, c)| {
            let t = text[pos + k];
            t == *c || (ci && t.to_lowercase().eq(c.to_lowercase()))
        });
        if ok && best.is_none_or(|(bp, _)| pos + nc.len() > bp) {
            best = Some((pos + nc.len(), *v));
        }
    }
    best
}

fn format_offset(secs: i32, pat: &str, no_off: &str) -> String {
    if secs == 0 {
        return no_off.to_string();
    }
    let colon = pat.contains(':');
    let body = &pat[1..];
    let parts: Vec<&str> = if colon { body.split(':').collect() } else { body.as_bytes().chunks(2).map(|c| std::str::from_utf8(c).unwrap_or("")).collect() };
    let min_mode = parts.get(1).map(|p| if p.chars().all(|c| c.is_uppercase()) { 2 } else { 1 }).unwrap_or(0);
    let sec_mode = parts.get(2).map(|p| if p.chars().all(|c| c.is_uppercase()) { 2 } else { 1 }).unwrap_or(0);
    let a = secs.abs();
    let (h, m, s) = (a / 3600, (a % 3600) / 60, a % 60);
    let mut out = format!("{}{:02}", if secs < 0 { '-' } else { '+' }, h);
    if min_mode == 2 || (min_mode == 1 && (m != 0 || (sec_mode == 1 && s != 0))) {
        if colon {
            out.push(':');
        }
        let _ = write!(out, "{m:02}");
        if sec_mode == 2 || (sec_mode == 1 && s != 0) {
            if colon {
                out.push(':');
            }
            let _ = write!(out, "{s:02}");
        }
    }
    out
}

/// Java `OffsetIdPrinterParser.parse` (strict or lenient).
fn parse_offset(text: &[char], pos: usize, pat: &str, no_off: &str, lenient: bool) -> Option<(usize, i32)> {
    let no: Vec<char> = no_off.chars().collect();
    if !no.is_empty() && text.len() >= pos + no.len() && text[pos..pos + no.len()].iter().zip(no.iter()).all(|(a, b)| a.eq_ignore_ascii_case(b)) {
        return Some((pos + no.len(), 0));
    }
    let sign = match text.get(pos)? {
        '+' => 1,
        '-' => -1,
        _ => return None,
    };
    let colon = pat.contains(':');
    let parts: Vec<&str> = if colon { pat[1..].split(':').collect() } else { pat[1..].as_bytes().chunks(2).map(|c| std::str::from_utf8(c).unwrap_or("")).collect() };
    let two = |p: usize| -> Option<i32> {
        let a = text.get(p)?;
        let b = text.get(p + 1)?;
        if a.is_ascii_digit() && b.is_ascii_digit() {
            Some((*a as i32 - '0' as i32) * 10 + (*b as i32 - '0' as i32))
        } else {
            None
        }
    };
    let mut p = pos + 1;
    let hours = match two(p) {
        Some(h) => {
            p += 2;
            h
        }
        None if lenient || parts.first().is_some_and(|h| h.len() == 1) => {
            let c = text.get(p)?;
            if !c.is_ascii_digit() {
                return None;
            }
            p += 1;
            *c as i32 - '0' as i32
        }
        None => return None,
    };
    let mut minutes = 0;
    let mut seconds = 0;
    let field = |p: &mut usize, mandatory: bool| -> Option<Option<i32>> {
        let mut q = *p;
        if colon {
            if text.get(q) == Some(&':') {
                q += 1;
            } else {
                return if mandatory && !lenient { None } else { Some(None) };
            }
        }
        match two(q) {
            Some(v) => {
                *p = q + 2;
                Some(Some(v))
            }
            None => {
                if mandatory && !lenient {
                    None
                } else {
                    Some(None)
                }
            }
        }
    };
    if parts.len() > 1 {
        let mandatory = parts[1].chars().all(|c| c.is_uppercase());
        match field(&mut p, mandatory)? {
            Some(v) => minutes = v,
            None => return Some((p, sign * hours * 3600)),
        }
        if parts.len() > 2 {
            let mandatory = parts[2].chars().all(|c| c.is_uppercase());
            if let Some(v) = field(&mut p, mandatory)? {
                seconds = v;
            }
        }
    }
    if hours > 18 || minutes > 59 || seconds > 59 {
        return None;
    }
    Some((p, sign * (hours * 3600 + minutes * 60 + seconds)))
}

fn parse_loc_offset(text: &[char], pos: usize, _full: bool) -> Option<(usize, i32)> {
    let gmt: Vec<char> = "GMT".chars().collect();
    if text.len() < pos + 3 || text[pos..pos + 3] != gmt[..] {
        return None;
    }
    let mut p = pos + 3;
    let sign = match text.get(p) {
        Some('+') => 1,
        Some('-') => -1,
        _ => return Some((p, 0)),
    };
    p += 1;
    let digits = |p: &mut usize, max: usize| -> Option<i32> {
        let mut v = 0;
        let mut n = 0;
        while n < max {
            match text.get(*p) {
                Some(c) if c.is_ascii_digit() => {
                    v = v * 10 + (*c as i32 - '0' as i32);
                    *p += 1;
                    n += 1;
                }
                _ => break,
            }
        }
        if n == 0 {
            None
        } else {
            Some(v)
        }
    };
    let h = digits(&mut p, 2)?;
    let mut m = 0;
    let mut s = 0;
    if text.get(p) == Some(&':') {
        p += 1;
        m = digits(&mut p, 2)?;
        if text.get(p) == Some(&':') {
            p += 1;
            s = digits(&mut p, 2)?;
        }
    }
    Some((p, sign * (h * 3600 + m * 60 + s)))
}

fn parse_zone(text: &[char], pos: usize, region_only: bool) -> Option<(usize, Zone)> {
    let mut end = pos;
    while end < text.len() && (text[end].is_ascii_alphanumeric() || matches!(text[end], '~' | '/' | '.' | '_' | '+' | '-' | ':')) {
        end += 1;
    }
    let mut e = end;
    while e > pos {
        let s: String = text[pos..e].iter().collect();
        if !(region_only && s.starts_with(['+', '-'])) {
            if let Ok(z) = Zone::parse(&s) {
                return Some((e, z));
            }
        }
        e -= 1;
    }
    None
}

/// `InstantPrinterParser.parse`.
fn parse_instant(text: &[char], pos: usize, st: &mut PState) -> Result<usize, usize> {
    let start = pos;
    let (p, year) = parse_number(text, pos, 4, 10, Sign::ExceedsPad, 0, true)?;
    let lit = |p: usize, c: char| -> Result<usize, usize> {
        match text.get(p) {
            Some(t) if t.eq_ignore_ascii_case(&c) => Ok(p + 1),
            _ => Err(p),
        }
    };
    let p = lit(p, '-')?;
    let (p, month) = parse_number(text, p, 2, 2, Sign::NotNegative, -1, true)?;
    let p = lit(p, '-')?;
    let (p, day) = parse_number(text, p, 2, 2, Sign::NotNegative, -1, true)?;
    let p = lit(p, 'T')?;
    let (p, mut hour) = parse_number(text, p, 2, 2, Sign::NotNegative, -1, true)?;
    let p = lit(p, ':')?;
    let (p, min) = parse_number(text, p, 2, 2, Sign::NotNegative, -1, true)?;
    let p = lit(p, ':')?;
    let (mut p, mut sec) = parse_number(text, p, 2, 2, Sign::NotNegative, -1, true)?;
    let mut nanos = 0i64;
    if text.get(p) == Some(&'.') {
        let mut q = p + 1;
        let mut digits = String::new();
        while digits.len() < 9 {
            match text.get(q) {
                Some(c) if c.is_ascii_digit() => {
                    digits.push(*c);
                    q += 1;
                }
                _ => break,
            }
        }
        for (k, c) in digits.chars().enumerate() {
            nanos += (c as i64 - '0' as i64) * 10i64.pow(8 - k as u32);
        }
        p = q;
    }
    let (p, off) = parse_offset(text, p, "+HH:MM:ss", "Z", true).ok_or(p)?;
    let mut days = 0;
    if hour == 24 && min == 0 && sec == 0 && nanos == 0 {
        hour = 0;
        days = 1;
    } else if hour == 23 && min == 59 && sec == 60 {
        st.leap_second = true;
        sec = 59;
    }
    let secs = (|| -> SResult<i64> {
        F::MonthOfYear.check(month)?;
        F::DayOfMonth.check(day)?;
        if day > month_len(month as u8, is_leap(year)) as i64 {
            return Err(dte("invalid date"));
        }
        let nod = time_of(hour, min, sec, 0)?;
        let ed = days_from_civil(year, month as u8, day as u8).checked_add(days).ok_or_else(overflow)?;
        let secs = ed.checked_mul(SPD).ok_or_else(overflow)?.checked_add(nod / NPS).ok_or_else(overflow)?.checked_sub(off as i64).ok_or_else(overflow)?;
        if !(INSTANT_MIN_SEC..=INSTANT_MAX_SEC).contains(&secs) {
            return Err(dte("Instant exceeds minimum or maximum instant"));
        }
        Ok(secs)
    })()
    .map_err(|_| start)?;
    st.set(F::MilliOfSecond, nanos / 1_000_000);
    st.set(F::InstantSeconds, secs);
    st.set(F::NanoOfSecond, nanos);
    st.set(F::MicroOfSecond, nanos / 1000);
    Ok(p)
}

// ---- resolution -----------------------------------------------------------

fn take(fields: &mut Vec<(F, i64)>, f: F) -> Option<i64> {
    let i = fields.iter().position(|(g, _)| *g == f)?;
    Some(fields.remove(i).1)
}
fn has(fields: &[(F, i64)], f: F) -> bool {
    fields.iter().any(|(g, _)| *g == f)
}
fn peek(fields: &[(F, i64)], f: F) -> Option<i64> {
    fields.iter().find(|(g, _)| *g == f).map(|e| e.1)
}

/// Java `Parsed.resolve` (ISO chronology).
fn resolve(st: PState, override_zone: Option<Zone>, resolver: u8) -> SResult<Parsed> {
    let mut fields = st.fields;
    let zone = st.zone.or(override_zone);
    let mut date: Option<LDate> = None;
    let mut time: Option<i64> = None;
    let mut excess_days = 0i64;
    let strict = resolver == 0;
    let lenient = resolver == 2;

    // Instant fields with a zone.
    if let Some(secs) = peek(&fields, F::InstantSeconds) {
        let z = zone.clone().or_else(|| peek(&fields, F::OffsetSeconds).map(|o| Zone::Offset(o as i32)));
        if let Some(z) = z {
            let nanos = peek(&fields, F::NanoOfSecond).unwrap_or(0);
            let zd = Zoned::of_instant((secs, nanos as u32), z)?;
            date = Some(zd.ldt.date);
            let sod = zd.ldt.nod / NPS;
            if let Some(v) = peek(&fields, F::SecondOfDay) {
                if v != sod {
                    return Err(dte(format!("Conflict found: SecondOfDay {v} differs from SecondOfDay {sod} while resolving  InstantSeconds")));
                }
            }
            fields.push((F::SecondOfDay, sod));
            if let Some(o) = peek(&fields, F::OffsetSeconds) {
                if o != zd.off as i64 {
                    return Err(dte(format!("Conflict found: OffsetSeconds {o} differs from OffsetSeconds {} while resolving  InstantSeconds", zd.off)));
                }
            }
        }
    }

    // Date fields.
    if let Some(ed) = take(&mut fields, F::EpochDay) {
        date = Some(LDate::from_epoch_day(ed)?);
    }
    if let Some(pm) = take(&mut fields, F::ProlepticMonth) {
        if !lenient {
            F::ProlepticMonth.check(pm)?;
        }
        fields.push((F::MonthOfYear, pm.rem_euclid(12) + 1));
        fields.push((F::Year, pm.div_euclid(12)));
    }
    if let Some(yoe) = peek(&fields, F::YearOfEra) {
        let era = take(&mut fields, F::Era);
        match era {
            None => {
                if !strict {
                    take(&mut fields, F::YearOfEra);
                    let year = peek(&fields, F::Year);
                    match year {
                        Some(y) => {
                            take(&mut fields, F::Year);
                            fields.push((F::Year, if y > 0 { yoe } else { 1 - yoe }));
                        }
                        None => fields.push((F::Year, yoe)),
                    }
                }
            }
            Some(1) => {
                take(&mut fields, F::YearOfEra);
                fields.push((F::Year, yoe));
            }
            Some(0) => {
                take(&mut fields, F::YearOfEra);
                fields.push((F::Year, 1 - yoe));
            }
            Some(e) => return Err(dte(format!("Invalid value for era: {e}"))),
        }
    } else if let Some(e) = take(&mut fields, F::Era) {
        if !lenient {
            F::Era.check(e)?;
        }
    }
    if date.is_none() && has(&fields, F::Year) {
        if has(&fields, F::MonthOfYear) && has(&fields, F::DayOfMonth) {
            let y = take(&mut fields, F::Year).unwrap_or(0);
            let moy = take(&mut fields, F::MonthOfYear).unwrap_or(1);
            let dom = take(&mut fields, F::DayOfMonth).unwrap_or(1);
            date = Some(resolve_ymd(y, moy, dom, resolver)?);
        } else if has(&fields, F::DayOfYear) {
            let y = take(&mut fields, F::Year).unwrap_or(0);
            let doy = take(&mut fields, F::DayOfYear).unwrap_or(1);
            date = Some(if lenient { LDate::of(y, 1, 1)?.plus_days(doy - 1)? } else { F::Year.check(y)?; LDate::of_year_day(y, doy)? });
        } else if has(&fields, F::MonthOfYear) && has(&fields, F::AlignedWeekOfMonth) && has(&fields, F::AlignedDayOfWeekInMonth) {
            let y = take(&mut fields, F::Year).unwrap_or(0);
            let moy = take(&mut fields, F::MonthOfYear).unwrap_or(1);
            let aw = take(&mut fields, F::AlignedWeekOfMonth).unwrap_or(1);
            let ad = take(&mut fields, F::AlignedDayOfWeekInMonth).unwrap_or(1);
            date = Some(LDate::of(y, moy, 1)?.plus_days((aw - 1) * 7 + (ad - 1))?);
        } else if has(&fields, F::MonthOfYear) && has(&fields, F::AlignedWeekOfMonth) && has(&fields, F::DayOfWeek) {
            let y = take(&mut fields, F::Year).unwrap_or(0);
            let moy = take(&mut fields, F::MonthOfYear).unwrap_or(1);
            let aw = take(&mut fields, F::AlignedWeekOfMonth).unwrap_or(1);
            let dow = take(&mut fields, F::DayOfWeek).unwrap_or(1);
            let base = LDate::of(y, moy, 1)?.plus_days((aw - 1) * 7)?;
            date = Some(adjust_next_or_same(&base, dow as u8)?);
        }
    }
    if date.is_none() && has(&fields, F::WeekBasedYear) && has(&fields, F::WeekOfWeekBasedYear) && has(&fields, F::DayOfWeek) {
        let wby = take(&mut fields, F::WeekBasedYear).unwrap_or(0);
        let week = take(&mut fields, F::WeekOfWeekBasedYear).unwrap_or(1);
        let dow = take(&mut fields, F::DayOfWeek).unwrap_or(1);
        let base = LDate::of(wby, 1, 4)?;
        let d = base.plus_weeks(week - 1)?.plus_days(dow - base.day_of_week() as i64)?;
        date = Some(d);
    }
    if date.is_none() && has(&fields, F::LocWeekBasedYear) && has(&fields, F::LocWeekOfYear) && has(&fields, F::LocDayOfWeek) {
        let (sow, min) = st.locale.week_fields();
        let wby = take(&mut fields, F::LocWeekBasedYear).unwrap_or(0);
        let week = take(&mut fields, F::LocWeekOfYear).unwrap_or(1);
        let ldow = take(&mut fields, F::LocDayOfWeek).unwrap_or(1);
        let jan1 = LDate::of(wby, 1, 1)?;
        let jan1_dow = loc_dow(&jan1, sow);
        let week_start_offset = start_of_week_offset(1, jan1_dow, min);
        let d = jan1.plus_days(-week_start_offset + (week - 1) * 7 + (ldow - 1))?;
        date = Some(d);
    }
    if let Some(d) = date {
        // Cross-check remaining date fields.
        for f in [F::DayOfWeek, F::MonthOfYear, F::DayOfMonth, F::DayOfYear, F::Year] {
            if let Some(v) = take(&mut fields, f) {
                if let Some(actual) = d.get(f) {
                    if actual != v {
                        return Err(dte(format!("Conflict found: Field {} {actual} differs from {} {v} derived from {}", f.name(), f.name(), {
                            let mut s = String::new();
                            d.write(&mut s);
                            s
                        })));
                    }
                }
            }
        }
    }

    // Time fields.
    if let Some(ch) = take(&mut fields, F::ClockHourOfDay) {
        if !lenient {
            F::ClockHourOfDay.check(ch)?;
        }
        fields.push((F::HourOfDay, if ch == 24 { 0 } else { ch }));
    }
    if let Some(ch) = take(&mut fields, F::ClockHourOfAmPm) {
        if !lenient {
            F::ClockHourOfAmPm.check(ch)?;
        }
        fields.push((F::HourOfAmPm, if ch == 12 { 0 } else { ch }));
    }
    if has(&fields, F::AmPmOfDay) && has(&fields, F::HourOfAmPm) {
        let ap = take(&mut fields, F::AmPmOfDay).unwrap_or(0);
        let hap = take(&mut fields, F::HourOfAmPm).unwrap_or(0);
        if !lenient {
            F::AmPmOfDay.check(ap)?;
            F::HourOfAmPm.check(hap)?;
        }
        fields.push((F::HourOfDay, ap * 12 + hap));
    }
    if let Some(nod) = take(&mut fields, F::NanoOfDay) {
        if !lenient {
            F::NanoOfDay.check(nod)?;
        }
        fields.push((F::HourOfDay, nod / (3600 * NPS)));
        fields.push((F::MinuteOfHour, (nod / (60 * NPS)) % 60));
        fields.push((F::SecondOfMinute, (nod / NPS) % 60));
        fields.push((F::NanoOfSecond, nod % NPS));
    }
    if let Some(v) = take(&mut fields, F::MicroOfDay) {
        if !lenient {
            F::MicroOfDay.check(v)?;
        }
        fields.push((F::SecondOfDay, v / 1_000_000));
        fields.push((F::MicroOfSecond, v % 1_000_000));
    }
    if let Some(v) = take(&mut fields, F::MilliOfDay) {
        if !lenient {
            F::MilliOfDay.check(v)?;
        }
        fields.push((F::SecondOfDay, v / 1000));
        fields.push((F::MilliOfSecond, v % 1000));
    }
    if let Some(v) = take(&mut fields, F::SecondOfDay) {
        if !lenient {
            F::SecondOfDay.check(v)?;
        }
        fields.push((F::HourOfDay, v / 3600));
        fields.push((F::MinuteOfHour, (v / 60) % 60));
        fields.push((F::SecondOfMinute, v % 60));
    }
    if let Some(v) = take(&mut fields, F::MinuteOfDay) {
        if !lenient {
            F::MinuteOfDay.check(v)?;
        }
        fields.push((F::HourOfDay, v / 60));
        fields.push((F::MinuteOfHour, v % 60));
    }
    if let Some(ms) = peek(&fields, F::MilliOfSecond) {
        if has(&fields, F::MicroOfSecond) {
            let us = take(&mut fields, F::MicroOfSecond).unwrap_or(0);
            take(&mut fields, F::MilliOfSecond);
            fields.push((F::MicroOfSecond, ms * 1000 + us % 1000));
        }
    }
    if has(&fields, F::MicroOfSecond) && !has(&fields, F::NanoOfSecond) {
        let us = take(&mut fields, F::MicroOfSecond).unwrap_or(0);
        fields.push((F::NanoOfSecond, us * 1000));
    } else if has(&fields, F::MilliOfSecond) && !has(&fields, F::NanoOfSecond) {
        let ms = take(&mut fields, F::MilliOfSecond).unwrap_or(0);
        fields.push((F::NanoOfSecond, ms * 1_000_000));
    }
    if let Some(hod) = take(&mut fields, F::HourOfDay) {
        let mut hod = hod;
        let moh = take(&mut fields, F::MinuteOfHour).unwrap_or(0);
        let som = take(&mut fields, F::SecondOfMinute).unwrap_or(0);
        let nos = take(&mut fields, F::NanoOfSecond).unwrap_or(0);
        if !lenient {
            if resolver == 1 && hod == 24 && moh == 0 && som == 0 && nos == 0 {
                hod = 0;
                excess_days = 1;
            }
            F::HourOfDay.check(hod)?;
            F::MinuteOfHour.check(moh)?;
            F::SecondOfMinute.check(som)?;
            F::NanoOfSecond.check(nos)?;
            time = Some(hod * 3600 * NPS + moh * 60 * NPS + som * NPS + nos);
        } else {
            let total = hod as i128 * 3600 * NPS as i128 + moh as i128 * 60 * NPS as i128 + som as i128 * NPS as i128 + nos as i128;
            excess_days = total.div_euclid(NPD as i128) as i64;
            time = Some(total.rem_euclid(NPD as i128) as i64);
        }
    }
    // Excess days roll into the date.
    if excess_days != 0 {
        if let Some(d) = date {
            date = Some(d.plus_days(excess_days)?);
            excess_days = 0;
        }
    }
    // Instant from date + time + offset/zone.
    if let (Some(d), Some(t)) = (date, time) {
        if !has(&fields, F::InstantSeconds) {
            if let Some(o) = peek(&fields, F::OffsetSeconds) {
                fields.insert(0, (F::InstantSeconds, LDT::of(d, t).epoch_second(o as i32)));
            } else if let Some(z) = &zone {
                let zd = Zoned::of_local(LDT::of(d, t), z.clone(), None)?;
                fields.insert(0, (F::InstantSeconds, zd.epoch_second()));
            }
        }
    }
    let _ = (excess_days, st.leap_second);
    Ok(Parsed { fields, zone, date, time })
}

fn adjust_next_or_same(d: &LDate, dow: u8) -> SResult<LDate> {
    let cal = d.day_of_week() as i64;
    let diff = cal - dow as i64;
    if diff == 0 {
        return Ok(*d);
    }
    d.plus_days(if diff >= 0 { 7 - diff } else { -diff })
}

fn resolve_ymd(y: i64, moy: i64, dom: i64, resolver: u8) -> SResult<LDate> {
    match resolver {
        2 => {
            let months = moy.checked_sub(1).ok_or_else(overflow)?;
            let days = dom.checked_sub(1).ok_or_else(overflow)?;
            LDate::of(y, 1, 1)?.plus_months(months)?.plus_days(days)
        }
        1 => {
            F::Year.check(y)?;
            F::MonthOfYear.check(moy)?;
            F::DayOfMonth.check(dom)?;
            let mut dom = dom;
            if matches!(moy, 4 | 6 | 9 | 11) && dom == 31 {
                dom = 30;
            } else if moy == 2 && dom > 28 {
                dom = dom.min(if is_leap(y) { 29 } else { 28 });
            }
            LDate::of(y, moy, dom)
        }
        _ => LDate::of(y, moy, dom),
    }
}

/// `X.from(parsed)` for the `parse(text, formatter)` entry points.
fn from_parsed(p: &Parsed, target: &str) -> SResult<Val> {
    let pv = t(JTime::Parsed(p.clone()));
    let unable_p = |what: &str| unable(what, &pv);
    match target {
        "LocalDate" => p.date.map(|d| t(JTime::LocalDate(d))).ok_or_else(|| unable_p("LocalDate")),
        "LocalTime" => p.time.map(|n| t(JTime::LocalTime(n))).ok_or_else(|| unable_p("LocalTime")),
        "LocalDateTime" => match (p.date, p.time) {
            (Some(d), Some(n)) => Ok(t(JTime::LocalDateTime(LDT::of(d, n)))),
            _ => Err(unable_p("LocalDateTime")),
        },
        "Instant" => {
            let secs = p.field(F::InstantSeconds).ok_or_else(|| unable_p("Instant"))?;
            let nanos = p.field(F::NanoOfSecond).unwrap_or(0);
            Ok(inst_to_val(inst_create(secs, nanos)?))
        }
        "ZonedDateTime" => {
            let zone = p.zone.clone().or_else(|| p.offset().map(Zone::Offset)).ok_or_else(|| unable_p("ZonedDateTime"))?;
            if let Some(secs) = p.field(F::InstantSeconds) {
                let nanos = p.field(F::NanoOfSecond).unwrap_or(0);
                return Ok(Zoned::of_instant(inst_create(secs, nanos)?, zone)?.to_val());
            }
            match (p.date, p.time) {
                (Some(d), Some(n)) => Ok(Zoned::of_local(LDT::of(d, n), zone, None)?.to_val()),
                _ => Err(unable_p("ZonedDateTime")),
            }
        }
        "OffsetDateTime" => {
            let off = p.offset().or_else(|| p.zone.as_ref().and_then(|z| z.fixed_offset().filter(|_| z.is_offset()))).ok_or_else(|| unable_p("OffsetDateTime"))?;
            match (p.date, p.time) {
                (Some(d), Some(n)) => Ok(t(JTime::OffsetDateTime(LDT::of(d, n), off))),
                _ => {
                    let secs = p.field(F::InstantSeconds).ok_or_else(|| unable_p("OffsetDateTime"))?;
                    let nanos = p.field(F::NanoOfSecond).unwrap_or(0);
                    let i = inst_create(secs, nanos)?;
                    Ok(t(JTime::OffsetDateTime(LDT::from_epoch(i.0, i.1, off)?, off)))
                }
            }
        }
        "OffsetTime" => {
            let off = p.offset().ok_or_else(|| unable_p("OffsetTime"))?;
            p.time.map(|n| t(JTime::OffsetTime(n, off))).ok_or_else(|| unable_p("OffsetTime"))
        }
        "YearMonth" => match (p.field(F::Year), p.field(F::MonthOfYear)) {
            (Some(y), Some(m)) => Ok(t(JTime::YearMonth(F::Year.check(y)? as i32, F::MonthOfYear.check(m)? as u8))),
            _ => Err(unable_p("YearMonth")),
        },
        "Year" => p.field(F::Year).map(|y| t(JTime::Year(y as i32))).ok_or_else(|| unable_p("Year")),
        "MonthDay" => match (p.field(F::MonthOfYear), p.field(F::DayOfMonth)) {
            (Some(m), Some(d)) => Ok(t(JTime::MonthDay(F::MonthOfYear.check(m)? as u8, F::DayOfMonth.check(d)? as u8))),
            _ => Err(unable_p("MonthDay")),
        },
        "DayOfWeek" => p.field(F::DayOfWeek).map(|d| Val::DayOfWeek(d as u8)).ok_or_else(|| unable_p("DayOfWeek")),
        "Month" => p.field(F::MonthOfYear).map(|m| Val::Month(m as u8)).ok_or_else(|| unable_p("Month")),
        "ZoneId" => p.zone.clone().or_else(|| p.offset().map(Zone::Offset)).map(|z| t(JTime::Zone(z))).ok_or_else(|| unable_p("ZoneId")),
        "ZoneOffset" => p.offset().map(|o| t(JTime::Zone(Zone::Offset(o)))).ok_or_else(|| unable_p("ZoneOffset")),
        _ => Ok(pv),
    }
}

/// The `DateTimeFormatter.ISO_*` / `RFC_1123_DATE_TIME` constants.
fn iso_formatter(name: &str) -> Option<Rc<Fmt>> {
    let date = || vec![num(F::Year, 4, 10, Sign::ExceedsPad), El::Lit("-".into()), fixed(F::MonthOfYear, 2), El::Lit("-".into()), fixed(F::DayOfMonth, 2)];
    let time = || {
        vec![
            fixed(F::HourOfDay, 2),
            El::Lit(":".into()),
            fixed(F::MinuteOfHour, 2),
            El::Opt(vec![El::Lit(":".into()), fixed(F::SecondOfMinute, 2), El::Opt(vec![El::Frac { min: 0, max: 9, point: true }])]),
        ]
    };
    let offset = || vec![El::Strict(false), El::Offset { pat: "+HH:MM:ss", no_off: "Z" }, El::Strict(true)];
    let ldt = || vec![El::CaseSensitive(false), El::Group(date()), El::Lit("T".into()), El::Group(time())];
    let odt = || {
        let mut v = vec![El::CaseSensitive(false), El::Group(ldt())];
        v.extend(offset());
        v
    };
    let region = || El::Opt(vec![El::Lit("[".into()), El::CaseSensitive(true), El::ZoneId { region_only: true }, El::Lit("]".into())]);
    let mut f = Fmt::new(match name {
        "ISO_LOCAL_DATE" => date(),
        "ISO_LOCAL_TIME" => time(),
        "ISO_LOCAL_DATE_TIME" => ldt(),
        "ISO_OFFSET_DATE_TIME" => odt(),
        "ISO_ZONED_DATE_TIME" => vec![El::Group(odt()), region()],
        "ISO_DATE_TIME" => {
            let mut inner = offset();
            inner.push(region());
            vec![El::Group(ldt()), El::Opt(inner)]
        }
        "ISO_INSTANT" => vec![El::CaseSensitive(false), El::Instant],
        "ISO_DATE" => vec![El::CaseSensitive(false), El::Group(date()), El::Opt(offset())],
        "ISO_TIME" => vec![El::CaseSensitive(false), El::Group(time()), El::Opt(offset())],
        "ISO_OFFSET_DATE" => {
            let mut v = vec![El::CaseSensitive(false), El::Group(date())];
            v.extend(offset());
            v
        }
        "ISO_OFFSET_TIME" => {
            let mut v = vec![El::CaseSensitive(false), El::Group(time())];
            v.extend(offset());
            v
        }
        "ISO_ORDINAL_DATE" => vec![El::CaseSensitive(false), num(F::Year, 4, 10, Sign::ExceedsPad), El::Lit("-".into()), fixed(F::DayOfYear, 3), El::Opt(offset())],
        "ISO_WEEK_DATE" => vec![
            El::CaseSensitive(false),
            num(F::WeekBasedYear, 4, 10, Sign::ExceedsPad),
            El::Lit("-W".into()),
            fixed(F::WeekOfWeekBasedYear, 2),
            El::Lit("-".into()),
            fixed(F::DayOfWeek, 1),
            El::Opt(offset()),
        ],
        "BASIC_ISO_DATE" => vec![
            El::CaseSensitive(false),
            El::Num { f: F::Year, min: 4, max: 4, sign: Sign::NotNegative, sub: 4 },
            El::Num { f: F::MonthOfYear, min: 2, max: 2, sign: Sign::NotNegative, sub: -1 },
            El::Num { f: F::DayOfMonth, min: 2, max: 2, sign: Sign::NotNegative, sub: -1 },
            El::Opt(vec![El::Strict(false), El::Offset { pat: "+HHMMss", no_off: "Z" }, El::Strict(true)]),
        ],
        "RFC_1123_DATE_TIME" => vec![
            El::CaseSensitive(false),
            El::Strict(false),
            El::Opt(vec![El::TextMap(F::DayOfWeek), El::Lit(", ".into())]),
            num(F::DayOfMonth, 1, 2, Sign::NotNegative),
            El::Lit(" ".into()),
            El::TextMap(F::MonthOfYear),
            El::Lit(" ".into()),
            fixed(F::Year, 4),
            El::Lit(" ".into()),
            fixed(F::HourOfDay, 2),
            El::Lit(":".into()),
            fixed(F::MinuteOfHour, 2),
            El::Opt(vec![El::Lit(":".into()), fixed(F::SecondOfMinute, 2)]),
            El::Lit(" ".into()),
            El::Offset { pat: "+HHMM", no_off: "GMT" },
        ],
        _ => return None,
    });
    f.resolver = 0;
    if name == "RFC_1123_DATE_TIME" {
        f.resolver = 1;
    }
    Some(Rc::new(f))
}

// ---------------------------------------------------------------------------
// `X.from(TemporalAccessor)` conversions

fn conv_parsed(v: &Val, target: &str) -> Option<SResult<Val>> {
    if let Val::Time(t) = v {
        if let JTime::Parsed(p) = &**t {
            return Some(from_parsed(p, target));
        }
    }
    None
}
fn conv_ldate(v: &Val) -> SResult<LDate> {
    if let Some(r) = conv_parsed(v, "LocalDate") {
        return ldate_arg(&r?);
    }
    match as_jtime(v) {
        Some(JTime::LocalDate(d)) => Ok(d),
        Some(JTime::LocalDateTime(d)) | Some(JTime::OffsetDateTime(d, _)) => Ok(d.date),
        Some(JTime::Zoned(z)) => Ok(z.ldt.date),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("LocalDate", v)),
    }
}
fn conv_ltime(v: &Val) -> SResult<i64> {
    if let Some(r) = conv_parsed(v, "LocalTime") {
        return ltime_arg(&r?);
    }
    match as_jtime(v) {
        Some(JTime::LocalTime(n)) | Some(JTime::OffsetTime(n, _)) => Ok(n),
        Some(JTime::LocalDateTime(d)) | Some(JTime::OffsetDateTime(d, _)) => Ok(d.nod),
        Some(JTime::Zoned(z)) => Ok(z.ldt.nod),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("LocalTime", v)),
    }
}
fn conv_ldt(v: &Val) -> SResult<LDT> {
    if let Some(r) = conv_parsed(v, "LocalDateTime") {
        return ldt_arg(&r?);
    }
    match as_jtime(v) {
        Some(JTime::LocalDateTime(d)) | Some(JTime::OffsetDateTime(d, _)) => Ok(d),
        Some(JTime::Zoned(z)) => Ok(z.ldt),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("LocalDateTime", v)),
    }
}
fn conv_instant(v: &Val) -> SResult<(i64, u32)> {
    if let Some(r) = conv_parsed(v, "Instant") {
        return instant_arg(&r?);
    }
    match as_jtime(v) {
        Some(JTime::Instant(s, n)) => Ok((s, n)),
        Some(JTime::OffsetDateTime(d, o)) => Ok((d.epoch_second(o), d.nanos())),
        Some(JTime::Zoned(z)) => Ok(z.instant()),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("Instant", v)),
    }
}
fn conv_zone(v: &Val) -> SResult<Zone> {
    if let Some(r) = conv_parsed(v, "ZoneId") {
        return zone_arg(&r?, "temporal");
    }
    match as_jtime(v) {
        Some(JTime::Zoned(z)) => Ok(z.zone),
        Some(JTime::OffsetDateTime(_, o)) | Some(JTime::OffsetTime(_, o)) => Ok(Zone::Offset(o)),
        Some(JTime::Zone(z)) => Ok(z),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("ZoneId", v)),
    }
}
fn conv_offset(v: &Val) -> SResult<i32> {
    if let Some(r) = conv_parsed(v, "ZoneOffset") {
        return offset_arg(&r?, "temporal");
    }
    match as_jtime(v) {
        Some(JTime::Zoned(z)) => Ok(z.off),
        Some(JTime::OffsetDateTime(_, o)) | Some(JTime::OffsetTime(_, o)) => Ok(o),
        Some(JTime::Zone(Zone::Offset(o))) => Ok(o),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("ZoneOffset", v)),
    }
}
fn conv_zoned(v: &Val) -> SResult<Zoned> {
    if let Some(r) = conv_parsed(v, "ZonedDateTime") {
        return conv_zoned(&r?);
    }
    match as_jtime(v) {
        Some(JTime::Zoned(z)) => Ok(z),
        Some(JTime::OffsetDateTime(d, o)) => Ok(Zoned { ldt: d, off: o, zone: Zone::Offset(o) }),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("ZonedDateTime", v)),
    }
}
fn conv_odt(v: &Val) -> SResult<(LDT, i32)> {
    if let Some(r) = conv_parsed(v, "OffsetDateTime") {
        return conv_odt(&r?);
    }
    match as_jtime(v) {
        Some(JTime::OffsetDateTime(d, o)) => Ok((d, o)),
        Some(JTime::Zoned(z)) => Ok((z.ldt, z.off)),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("OffsetDateTime", v)),
    }
}
fn conv_otime(v: &Val) -> SResult<(i64, i32)> {
    if let Some(r) = conv_parsed(v, "OffsetTime") {
        return conv_otime(&r?);
    }
    match as_jtime(v) {
        Some(JTime::OffsetTime(n, o)) => Ok((n, o)),
        Some(JTime::OffsetDateTime(d, o)) => Ok((d.nod, o)),
        Some(JTime::Zoned(z)) => Ok((z.ldt.nod, z.off)),
        _ if v.is_null() => Err(npe("temporal")),
        _ => Err(unable("OffsetTime", v)),
    }
}
fn conv_ym(v: &Val) -> SResult<(i32, u8)> {
    if let Some(r) = conv_parsed(v, "YearMonth") {
        return conv_ym(&r?);
    }
    match as_jtime(v) {
        Some(JTime::YearMonth(y, m)) => Ok((y, m)),
        _ => conv_ldate(v).map(|d| (d.y, d.m)).map_err(|_| unable("YearMonth", v)),
    }
}
fn conv_year(v: &Val) -> SResult<i32> {
    if let Some(r) = conv_parsed(v, "Year") {
        return conv_year(&r?);
    }
    match as_jtime(v) {
        Some(JTime::Year(y)) => Ok(y),
        Some(JTime::YearMonth(y, _)) => Ok(y),
        _ => conv_ldate(v).map(|d| d.y).map_err(|_| unable("Year", v)),
    }
}
fn conv_md(v: &Val) -> SResult<(u8, u8)> {
    if let Some(r) = conv_parsed(v, "MonthDay") {
        return conv_md(&r?);
    }
    match as_jtime(v) {
        Some(JTime::MonthDay(m, d)) => Ok((m, d)),
        _ => conv_ldate(v).map(|d| (d.m, d.d)).map_err(|_| unable("MonthDay", v)),
    }
}
fn conv_dow(v: &Val) -> SResult<u8> {
    if let Val::DayOfWeek(d) = v {
        return Ok(*d);
    }
    if let Some(r) = conv_parsed(v, "DayOfWeek") {
        return dow_arg(&r?);
    }
    conv_ldate(v).map(|d| d.day_of_week()).map_err(|_| unable("DayOfWeek", v))
}
fn conv_month(v: &Val) -> SResult<u8> {
    if let Val::Month(m) = v {
        return Ok(*m);
    }
    if let Some(r) = conv_parsed(v, "Month") {
        return match r? {
            Val::Month(m) => Ok(m),
            o => Err(cast_to(&o, "java.time.Month")),
        };
    }
    match as_jtime(v) {
        Some(JTime::YearMonth(_, m)) | Some(JTime::MonthDay(m, _)) => Ok(m),
        _ => conv_ldate(v).map(|d| d.m).map_err(|_| unable("Month", v)),
    }
}

// ---------------------------------------------------------------------------
// Generic temporal operations

#[derive(Clone, Debug)]
enum T {
    LD(LDate),
    LT(i64),
    LDT(LDT),
    ODT(LDT, i32),
    OT(i64, i32),
    ZDT(Zoned),
    INST(i64, u32),
    YM(i32, u8),
    Y(i32),
    MD(u8, u8),
}

impl T {
    fn of(v: &Val) -> Option<T> {
        Some(match as_jtime(v)? {
            JTime::LocalDate(d) => T::LD(d),
            JTime::LocalTime(n) => T::LT(n),
            JTime::LocalDateTime(d) => T::LDT(d),
            JTime::OffsetDateTime(d, o) => T::ODT(d, o),
            JTime::OffsetTime(n, o) => T::OT(n, o),
            JTime::Zoned(z) => T::ZDT(z),
            JTime::Instant(s, n) => T::INST(s, n),
            JTime::YearMonth(y, m) => T::YM(y, m),
            JTime::Year(y) => T::Y(y),
            JTime::MonthDay(m, d) => T::MD(m, d),
            _ => return None,
        })
    }
    fn to_val(self) -> Val {
        match self {
            T::LD(d) => t(JTime::LocalDate(d)),
            T::LT(n) => t(JTime::LocalTime(n)),
            T::LDT(d) => t(JTime::LocalDateTime(d)),
            T::ODT(d, o) => t(JTime::OffsetDateTime(d, o)),
            T::OT(n, o) => t(JTime::OffsetTime(n, o)),
            T::ZDT(z) => z.to_val(),
            T::INST(s, n) => inst_to_val((s, n)),
            T::YM(y, m) => t(JTime::YearMonth(y, m)),
            T::Y(y) => t(JTime::Year(y)),
            T::MD(m, d) => t(JTime::MonthDay(m, d)),
        }
    }
    /// `X.with(TemporalAdjuster)` frame.
    fn with_adjuster_frame(&self) -> Fr {
        match self {
            T::LD(_) => jt!("LocalDate", "with", 942),
            T::LT(_) => LT_WITH_781,
            T::LDT(_) => jt!("LocalDateTime", "with", 860),
            T::ZDT(_) => jt!("ZonedDateTime", "with", 1247),
            T::ODT(..) => jt!("OffsetDateTime", "with", 1063),
            T::OT(..) => jt!("OffsetTime", "with", 747),
            T::INST(..) => jt!("Instant", "with", 683),
            T::YM(..) => jt!("YearMonth", "with", 667),
            T::Y(_) => jt!("Year", "with", 596),
            T::MD(..) => jt!("MonthDay", "with", 563),
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            T::LD(_) => "LocalDate",
            T::LT(_) => "LocalTime",
            T::LDT(_) => "LocalDateTime",
            T::ODT(..) => "OffsetDateTime",
            T::OT(..) => "OffsetTime",
            T::ZDT(_) => "ZonedDateTime",
            T::INST(..) => "Instant",
            T::YM(..) => "YearMonth",
            T::Y(_) => "Year",
            T::MD(..) => "MonthDay",
        }
    }
    fn acc(&self) -> Acc {
        Acc::of(&self.clone().to_val()).unwrap_or_default()
    }
    fn get(&self, f: F) -> Option<i64> {
        self.acc().get(f, &Locale::en())
    }
    fn date(&self) -> Option<LDate> {
        match self {
            T::LD(d) => Some(*d),
            T::LDT(d) | T::ODT(d, _) => Some(d.date),
            T::ZDT(z) => Some(z.ldt.date),
            _ => None,
        }
    }
    /// `getLong(field)`.
    fn get_long(&self, f: F) -> SResult<i64> {
        if let T::INST(s, n) = self {
            return match f {
                F::NanoOfSecond => Ok(*n as i64),
                F::MicroOfSecond => Ok(*n as i64 / 1000),
                F::MilliOfSecond => Ok(*n as i64 / 1_000_000),
                F::InstantSeconds => Ok(*s),
                _ => Err(frs(unsupported_field(f), &[INST_GETLONG_603])),
            };
        }
        self.get(f).ok_or_else(|| frs(unsupported_field(f), &getlong_chain(self.kind(), f.is_time_based())))
    }
    /// `get(field)` (int-ranged fields only).
    fn get_int(&self, f: F) -> SResult<i64> {
        if let T::INST(..) = self {
            return match f {
                F::NanoOfSecond | F::MicroOfSecond | F::MilliOfSecond => self.get_long(f),
                _ => Err(frs(unsupported_field(f), &[INST_GET_566])),
            };
        }
        if f.is_chrono() && !f.is_int_range() {
            if self.get(f).is_some() {
                let e = utte(format!("Invalid field '{}' for get() method, use getLong() instead", f.name()));
                let inner: Vec<Fr> = match (self, f) {
                    (T::ZDT(_), F::InstantSeconds) => vec![ZDT_GET_820],
                    (_, F::InstantSeconds | F::OffsetSeconds) => vec![],
                    _ if f.is_time_based() => vec![LT_GET0_698, LT_GET_653],
                    (_, F::ProlepticMonth) => vec![LD_GET0_714, LD_GET_661],
                    _ => vec![LD_GET0_710, LD_GET_661],
                };
                return Err(frs(e, &get_chain_tail(self.kind(), inner)));
            }
            return Err(frs(unsupported_field(f), &get_chain_tail(self.kind(), get_chain_inner(self.kind(), f))));
        }
        self.get(f).ok_or_else(|| frs(unsupported_field(f), &get_chain_tail(self.kind(), get_chain_inner(self.kind(), f))))
    }
    fn range(&self, f: F) -> SResult<(i64, i64, i64, i64)> {
        if self.get(f).is_none() {
            return Err(frs(
                unsupported_field(f),
                &match self {
                    T::LD(_) => vec![jt!("LocalDate", "range", 626)],
                    T::LT(_) => vec![TA_RANGE, LT_RANGE_620],
                    T::INST(..) => vec![TA_RANGE, jt!("Instant", "range", 529)],
                    T::YM(..) => vec![TA_RANGE, jt!("YearMonth", "range", 446)],
                    T::Y(_) => vec![TA_RANGE, jt!("Year", "range", 443)],
                    T::MD(..) => vec![TA_RANGE, jt!("MonthDay", "range", 406)],
                    _ => vec![],
                },
            ));
        }
        if let Some(d) = self.date() {
            match f {
                F::DayOfMonth => return Ok((1, 1, d.length_of_month() as i64, d.length_of_month() as i64)),
                F::DayOfYear => return Ok((1, 1, d.length_of_year(), d.length_of_year())),
                F::AlignedWeekOfMonth => {
                    let m = if d.m == 2 && !d.leap() { 4 } else { 5 };
                    return Ok((1, 1, m, m));
                }
                F::YearOfEra => {
                    let m = if d.y <= 0 { MAX_YEAR + 1 } else { MAX_YEAR };
                    return Ok((1, 1, m, m));
                }
                F::DayOfQuarter => {
                    let q = quarter_len(&d);
                    return Ok((1, 1, q, q));
                }
                F::WeekOfWeekBasedYear => {
                    let m = iso_week_range_max(iso_week_based_year(&d));
                    return Ok((1, 1, m, m));
                }
                _ => {}
            }
        }
        if let T::YM(y, _) = self {
            if f == F::YearOfEra {
                let m = if *y <= 0 { MAX_YEAR + 1 } else { MAX_YEAR };
                return Ok((1, 1, m, m));
            }
        }
        let (a, b, c) = f.range();
        Ok((a, a, b, c))
    }

    fn plus_unit(&self, n: i64, unit: Unit) -> SResult<T> {
        Ok(match self {
            T::LD(d) => T::LD(d.plus_unit(n, unit)?),
            T::LT(t) => T::LT(time_plus_unit(*t, n, unit)?),
            T::LDT(d) => T::LDT(d.plus_unit(n, unit)?),
            T::ODT(d, o) => T::ODT(d.plus_unit(n, unit)?, *o),
            T::OT(t, o) => T::OT(time_plus_unit(*t, n, unit)?, *o),
            T::ZDT(z) => {
                let ldt = z.ldt.plus_unit(n, unit).map_err(|e| fr(e, &[ZDT_PLUS_1604]))?;
                if unit_is_date_based(unit) {
                    T::ZDT(z.with_ldt(ldt)?)
                } else {
                    T::ZDT(z.with_instant((ldt.epoch_second(z.off), ldt.nanos()))?)
                }
            }
            T::INST(s, nn) => {
                let r = inst_plus_unit((*s, *nn), n, unit)?;
                T::INST(r.0, r.1)
            }
            T::YM(y, m) => {
                let pm = *y as i64 * 12 + *m as i64 - 1;
                let months = match unit {
                    U_MONTHS => n,
                    U_YEARS => n.checked_mul(12).ok_or_else(overflow)?,
                    U_DECADES => n.checked_mul(120).ok_or_else(overflow)?,
                    U_CENTURIES => n.checked_mul(1200).ok_or_else(overflow)?,
                    U_MILLENNIA => n.checked_mul(12000).ok_or_else(overflow)?,
                    U_ERAS => return self.with_field(F::Era, era_of(*y).checked_add(n).ok_or_else(overflow)?),
                    u => return Err(frs(unsupported_unit(u), &[YM_PLUS_833])),
                };
                if months == 0 {
                    return Ok(self.clone());
                }
                let calc = pm.checked_add(months).ok_or_else(overflow)?;
                let ny = F::Year.check_int(calc.div_euclid(12)).map_err(|e| fr(e, &[if unit == U_MONTHS { jt!("YearMonth", "plusMonths", 871) } else { jt!("YearMonth", "plusYears", 852) }]))?;
                T::YM(ny as i32, (calc.rem_euclid(12) + 1) as u8)
            }
            T::Y(y) => {
                let years = match unit {
                    U_YEARS => n,
                    U_DECADES => n.checked_mul(10).ok_or_else(overflow)?,
                    U_CENTURIES => n.checked_mul(100).ok_or_else(overflow)?,
                    U_MILLENNIA => n.checked_mul(1000).ok_or_else(overflow)?,
                    U_ERAS => return self.with_field(F::Era, era_of(*y).checked_add(n).ok_or_else(overflow)?),
                    u => return Err(frs(unsupported_unit(u), &[Y_PLUS_720])),
                };
                if years == 0 {
                    return Ok(self.clone());
                }
                T::Y(F::Year.check_int((*y as i64).saturating_add(years)).map_err(|e| fr(e, &[jt!("Year", "plusYears", 739)]))? as i32)
            }
            T::MD(..) => return Err(unsupported("MonthDay is not a Temporal")),
        })
    }

    /// `plus(TemporalAmount)`.
    fn plus_amount(&self, amt: &Val, negate: bool) -> SResult<T> {
        let s = if negate { -1 } else { 1 };
        match amt {
            Val::Time(j) => match &**j {
                JTime::Period(y, m, d) => {
                    let months = period_total_months((*y, *m, *d)) * s;
                    let days = *d as i64 * s;
                    let mut r = self.clone();
                    if let T::ZDT(z) = self {
                        let ldt = z.ldt.plus_unit(months, U_MONTHS)?.plus_unit(days, U_DAYS)?;
                        return Ok(T::ZDT(z.with_ldt(ldt)?));
                    }
                    if months != 0 {
                        r = r.plus_unit(months, U_MONTHS)?;
                    }
                    if days != 0 {
                        r = r.plus_unit(days, U_DAYS)?;
                    }
                    Ok(r)
                }
                JTime::Duration(secs, nanos) => {
                    let mut r = self.clone();
                    let (secs, nanos) = (*secs, *nanos as i64);
                    if negate {
                        // Duration.subtractFrom: minus(seconds, SECONDS).minus(nanos, NANOS)
                        if secs != 0 {
                            r = r.plus_unit(secs.checked_neg().ok_or_else(overflow)?, U_SECONDS)?;
                        }
                        if nanos != 0 {
                            r = r.plus_unit(-nanos, U_NANOS)?;
                        }
                    } else {
                        if secs != 0 {
                            r = r.plus_unit(secs, U_SECONDS)?;
                        }
                        if nanos != 0 {
                            r = r.plus_unit(nanos, U_NANOS)?;
                        }
                    }
                    Ok(r)
                }
                _ => Err(cast_to(amt, "java.time.temporal.TemporalAmount")),
            },
            Val::Null => Err(npe(if negate { "amountToSubtract" } else { "amountToAdd" })),
            _ => Err(cast_to(amt, "java.time.temporal.TemporalAmount")),
        }
    }

    /// `until(end, unit)`.
    fn until(&self, end: &Val, unit: Unit) -> SResult<i64> {
        if end.is_null() {
            return Err(match self {
                T::LD(_) => npe_req("temporal", &[LD_FROM_419, jt!("LocalDate", "until", 1645)]),
                T::INST(..) => npe_req("temporal", &[INST_FROM_374, jt!("Instant", "until", 1174)]),
                T::Y(_) => npe_req("temporal", &[jt!("Year", "from", 256), jt!("Year", "until", 918)]),
                _ => npe_req("temporal", &[]),
            });
        }
        match self {
            T::LD(d) => d.until_unit(&conv_ldate(end).map_err(|e| fr(e, &[jt!("LocalDate", "until", 1645)]))?, unit),
            T::LT(n) => time_until(*n, conv_ltime(end).map_err(|e| fr(e, &[jt!("LocalTime", "until", 1418)]))?, unit),
            T::LDT(d) => d.until(&conv_ldt(end).map_err(|e| fr(e, &[jt!("LocalDateTime", "until", 1679)]))?, unit),
            T::ODT(d, o) => {
                let (ed, eo) = conv_odt(end).map_err(|e| fr(e, &[jt!("OffsetDateTime", "until", 1657)]))?;
                let ed = LDT::from_epoch(ed.epoch_second(eo), ed.nanos(), *o)?;
                d.until(&ed, unit).map_err(|e| fr(e, &[ODT_UNTIL_1666]))
            }
            T::OT(n, o) => {
                let (en, eo) = conv_otime(end)?;
                let en = time_plus(en, (*o as i64 - eo as i64) as i128 * NPS as i128);
                let d = en - n;
                Ok(match unit {
                    U_NANOS => d,
                    U_MICROS => d / 1000,
                    U_MILLIS => d / 1_000_000,
                    U_SECONDS => d / NPS,
                    U_MINUTES => d / (60 * NPS),
                    U_HOURS => d / (3600 * NPS),
                    U_HALF_DAYS => d / (12 * 3600 * NPS),
                    u => return Err(unsupported_unit(u)),
                })
            }
            T::ZDT(z) => {
                let e = conv_zoned(end).map_err(|e| fr(e, &[jt!("ZonedDateTime", "until", 2128)]))?;
                if unit_is_date_based(unit) {
                    let e = Zoned::of_instant(e.instant(), z.zone.clone())?;
                    z.ldt.until(&e.ldt, unit).map_err(|e| fr(e, &[ODT_UNTIL_1666, ZDT_UNTIL_2140]))
                } else {
                    let ed = LDT::from_epoch(e.epoch_second(), e.ldt.nanos(), z.off)?;
                    z.ldt.until(&ed, unit).map_err(|e| fr(e, &[ODT_UNTIL_1666, ZDT_UNTIL_2140]))
                }
            }
            T::INST(s, n) => inst_until((*s, *n), conv_instant(end).map_err(|e| fr(e, &[jt!("Instant", "until", 1174)]))?, unit),
            T::YM(y, m) => {
                let (ey, em) = conv_ym(end).map_err(|e| fr(e, &[jt!("YearMonth", "until", 1065)]))?;
                let months = (ey as i64 * 12 + em as i64) - (*y as i64 * 12 + *m as i64);
                Ok(match unit {
                    U_MONTHS => months,
                    U_YEARS => months / 12,
                    U_DECADES => months / 120,
                    U_CENTURIES => months / 1200,
                    U_MILLENNIA => months / 12000,
                    U_ERAS => era_of(ey) - era_of(*y),
                    u => return Err(frs(unsupported_unit(u), &[YM_UNTIL_1075])),
                })
            }
            T::Y(y) => {
                let ey = conv_year(end).map_err(|e| fr(e, &[jt!("Year", "until", 918)]))?;
                let years = ey as i64 - *y as i64;
                Ok(match unit {
                    U_YEARS => years,
                    U_DECADES => years / 10,
                    U_CENTURIES => years / 100,
                    U_MILLENNIA => years / 1000,
                    U_ERAS => era_of(ey) - era_of(*y),
                    u => return Err(frs(unsupported_unit(u), &[Y_UNTIL_927])),
                })
            }
            T::MD(..) => Err(unsupported("MonthDay is not a Temporal")),
        }
    }

    fn with_field(&self, f: F, v: i64) -> SResult<T> {
        Ok(match self {
            T::LD(d) => T::LD(d.with_field(f, v)?),
            T::LT(n) => T::LT(time_with(*n, f, v)?),
            T::LDT(d) => T::LDT(d.with_field(f, v)?),
            T::ODT(d, o) => match f {
                F::InstantSeconds => {
                    let i = inst_create(v, d.nanos() as i64)?;
                    T::ODT(LDT::from_epoch(i.0, i.1, *o)?, *o)
                }
                F::OffsetSeconds => T::ODT(*d, offset_of_total(f.check(v)?)?),
                _ => T::ODT(d.with_field(f, v)?, *o),
            },
            T::OT(n, o) => match f {
                F::OffsetSeconds => T::OT(*n, offset_of_total(f.check(v)?)?),
                _ => T::OT(time_with(*n, f, v)?, *o),
            },
            T::ZDT(z) => match f {
                F::InstantSeconds => T::ZDT(z.with_instant(inst_create(v, z.ldt.nanos() as i64)?)?),
                _ if f != F::OffsetSeconds && z.ldt.with_field(f, v).is_err() => return Err(fr(z.ldt.with_field(f, v).err().unwrap_or_else(|| unsupported_field(f)), &[ZDT_WITH_1311])),
                F::OffsetSeconds => {
                    let off = offset_of_total(f.check(v)?)?;
                    let (offs, _) = z.zone.valid_offsets(z.ldt.epoch_second(0));
                    if offs.contains(&off) {
                        T::ZDT(Zoned { ldt: z.ldt, off, zone: z.zone.clone() })
                    } else {
                        T::ZDT(z.clone())
                    }
                }
                _ => T::ZDT(z.with_ldt(z.ldt.with_field(f, v)?)?),
            },
            T::INST(s, n) => match f {
                F::MilliOfSecond => {
                    f.check(v).map_err(|e| fr(e, &[INST_WITH_707]))?;
                    T::INST(*s, (v * 1_000_000) as u32)
                }
                F::MicroOfSecond => {
                    f.check(v).map_err(|e| fr(e, &[jt!("Instant", "with", 711)]))?;
                    T::INST(*s, (v * 1000) as u32)
                }
                F::NanoOfSecond => {
                    f.check(v).map_err(|e| fr(e, &[jt!("Instant", "with", 715)]))?;
                    T::INST(*s, v as u32)
                }
                F::InstantSeconds => {
                    let i = inst_create(v, *n as i64)?;
                    T::INST(i.0, i.1)
                }
                _ => return Err(frs(unsupported_field(f), &[INST_WITH_719])),
            },
            T::YM(y, m) => match f {
                F::MonthOfYear => T::YM(*y, f.check(v).map_err(|e| fr(e, &[jt!("YearMonth", "withMonth", 742), jt!("YearMonth", "with", 695)]))? as u8),
                F::ProlepticMonth => self.plus_unit(v - (*y as i64 * 12 + *m as i64 - 1), U_MONTHS)?,
                F::YearOfEra => T::YM(f.check(if *y < 1 { 1 - v } else { v }).map_err(|e| fr(e, &[jt!("YearMonth", "withYear", 728), jt!("YearMonth", "with", 697)]))? as i32, *m),
                F::Year => T::YM(f.check(v).map_err(|e| fr(e, &[jt!("YearMonth", "withYear", 728), jt!("YearMonth", "with", 698)]))? as i32, *m),
                F::Era => {
                    f.check(v).map_err(|e| fr(e, &[jt!("YearMonth", "with", 699)]))?;
                    if era_of(*y) == v {
                        self.clone()
                    } else {
                        T::YM(1 - *y, *m)
                    }
                }
                _ => return Err(frs(unsupported_field(f), &[YM_WITH_711])),
            },
            T::Y(y) => match f {
                F::YearOfEra => T::Y(F::Year.check(if *y < 1 { 1 - v } else { v }).map_err(|e| fr(e, &[jt!("Year", "of", 229), jt!("Year", "with", 625)]))? as i32),
                F::Year => T::Y(f.check(v).map_err(|e| fr(e, &[jt!("Year", "of", 229), jt!("Year", "with", 626)]))? as i32),
                F::Era => {
                    f.check(v).map_err(|e| fr(e, &[jt!("Year", "with", 627)]))?;
                    if era_of(*y) == v {
                        self.clone()
                    } else {
                        T::Y(1 - *y)
                    }
                }
                _ => return Err(frs(unsupported_field(f), &[Y_WITH_631])),
            },
            T::MD(..) => return Err(unsupported("MonthDay is not a Temporal")),
        })
    }

    fn truncated(&self, unit: Unit) -> SResult<T> {
        Ok(match self {
            T::LT(n) => T::LT(time_truncate(*n, unit)?),
            T::LDT(d) => T::LDT(d.truncated(unit)?),
            T::ODT(d, o) => T::ODT(d.truncated(unit)?, *o),
            T::OT(n, o) => T::OT(time_truncate(*n, unit)?, *o),
            T::ZDT(z) => T::ZDT(z.with_ldt(z.ldt.truncated(unit).map_err(|e| fr(e, &[jt!("ZonedDateTime", "truncatedTo", 1528)]))?)?),
            T::INST(s, n) => {
                let r = inst_truncate((*s, *n), unit)?;
                T::INST(r.0, r.1)
            }
            _ => return Err(unsupported("truncatedTo")),
        })
    }

    /// `with(TemporalAdjuster)`.
    fn with_adjuster(&self, rt: &mut Rt, adj: &Val) -> SResult<T> {
        match adj {
            Val::Null => Err(frs(
                npe("Cannot invoke \"java.time.temporal.TemporalAdjuster.adjustInto(java.time.temporal.Temporal)\" because \"adjuster\" is null"),
                &[match self {
                    T::LD(_) => jt!("LocalDate", "with", 942),
                    T::ZDT(_) => jt!("ZonedDateTime", "with", 1247),
                    T::LT(_) => jt!("LocalTime", "with", 781),
                    T::LDT(_) => jt!("LocalDateTime", "with", 860),
                    _ => jt!("Instant", "with", 683),
                }],
            )),
            Val::DayOfWeek(d) => self.with_field(F::DayOfWeek, *d as i64),
            Val::Month(m) => self.with_field(F::MonthOfYear, *m as i64),
            Val::Lambda(_) | Val::MethodRef(_) => {
                let r = rt.call_fn(adj, &[self.clone().to_val()])?;
                T::of(&r).ok_or_else(|| cast_to(&r, &format!("java.time.{}", self.kind())))
            }
            Val::Time(j) => match &**j {
                JTime::Adjuster(a) => apply_adjuster(a, self, rt).map_err(|e| fr(e, &[self.with_adjuster_frame()])),
                JTime::LocalDate(d) => {
                    if let T::ZDT(z) = self {
                        return Ok(T::ZDT(z.with_ldt(LDT::of(*d, z.ldt.nod))?));
                    }
                    self.with_field(F::EpochDay, d.epoch_day()).map_err(|e| fr(e, &[LT_WITH_127, jf!("java.time.chrono.ChronoLocalDate", "adjustInto", "ChronoLocalDate", 552), jt!("LocalDate", "adjustInto", 1593), self.with_adjuster_frame()]))
                }
                JTime::LocalTime(n) => {
                    if let T::ZDT(z) = self {
                        return Ok(T::ZDT(z.with_ldt(LDT::of(z.ldt.date, *n))?));
                    }
                    self.with_field(F::NanoOfDay, *n)
                }
                JTime::LocalDateTime(d) => {
                    if let T::ZDT(z) = self {
                        return Ok(T::ZDT(z.with_ldt(*d)?));
                    }
                    self.with_field(F::EpochDay, d.date.epoch_day())?.with_field(F::NanoOfDay, d.nod)
                }
                JTime::OffsetDateTime(d, o) => {
                    if let T::ZDT(z) = self {
                        return Ok(T::ZDT(Zoned::of_local(*d, z.zone.clone(), Some(*o))?));
                    }
                    self.with_field(F::EpochDay, d.date.epoch_day())?.with_field(F::NanoOfDay, d.nod)?.with_field(F::OffsetSeconds, *o as i64)
                }
                JTime::Instant(s, n) => {
                    if let T::ZDT(z) = self {
                        return Ok(T::ZDT(z.with_instant((*s, *n))?));
                    }
                    self.with_field(F::InstantSeconds, *s)?.with_field(F::NanoOfSecond, *n as i64)
                }
                JTime::Zone(Zone::Offset(o)) => self.with_field(F::OffsetSeconds, *o as i64),
                JTime::YearMonth(y, m) => self.with_field(F::ProlepticMonth, *y as i64 * 12 + *m as i64 - 1),
                JTime::Year(y) => self.with_field(F::Year, *y as i64),
                JTime::MonthDay(m, d) => {
                    let r = self.with_field(F::MonthOfYear, *m as i64)?;
                    let max = r.range(F::DayOfMonth)?.3;
                    r.with_field(F::DayOfMonth, (*d as i64).min(max))
                }
                _ => Err(cast_to(adj, "java.time.temporal.TemporalAdjuster")),
            },
            _ => Err(cast_to(adj, "java.time.temporal.TemporalAdjuster")),
        }
    }
}

const ADJ_LAST_DAY_OF_MONTH: Fr = jf!("java.time.temporal.TemporalAdjusters", "lambda$lastDayOfMonth$0", "TemporalAdjusters", 189);

fn apply_adjuster(a: &Adj, t0: &T, rt: &mut Rt) -> SResult<T> {
    let dow_of = |t: &T| -> SResult<i64> { t.get(F::DayOfWeek).ok_or_else(|| unsupported_field(F::DayOfWeek)) };
    Ok(match a {
        Adj::FirstDayOfMonth => t0.with_field(F::DayOfMonth, 1)?,
        Adj::LastDayOfMonth => {
            let max = t0.range(F::DayOfMonth).map_err(|e| fr(e, &[ADJ_LAST_DAY_OF_MONTH]))?.3;
            t0.with_field(F::DayOfMonth, max)?
        }
        Adj::FirstDayOfNextMonth => t0.with_field(F::DayOfMonth, 1)?.plus_unit(1, U_MONTHS)?,
        Adj::FirstDayOfYear => t0.with_field(F::DayOfYear, 1)?,
        Adj::LastDayOfYear => {
            let max = t0.range(F::DayOfYear)?.3;
            t0.with_field(F::DayOfYear, max)?
        }
        Adj::FirstDayOfNextYear => t0.with_field(F::DayOfYear, 1)?.plus_unit(1, U_YEARS)?,
        Adj::DayOfWeekInMonth(ordinal, dow) => {
            let dow = *dow as i64;
            if *ordinal >= 0 {
                let temp = t0.with_field(F::DayOfMonth, 1)?;
                let cur = dow_of(&temp)?;
                let mut diff = (dow - cur + 7) % 7;
                diff += (ordinal - 1) * 7;
                temp.plus_unit(diff, U_DAYS)?
            } else {
                let max = t0.range(F::DayOfMonth)?.3;
                let temp = t0.with_field(F::DayOfMonth, max)?;
                let cur = dow_of(&temp)?;
                let mut diff = dow - cur;
                diff = if diff == 0 {
                    0
                } else if diff > 0 {
                    diff - 7
                } else {
                    diff
                };
                diff -= (-ordinal - 1) * 7;
                temp.plus_unit(diff, U_DAYS)?
            }
        }
        Adj::Next(dow) | Adj::NextOrSame(dow) => {
            let cal = dow_of(t0)?;
            if matches!(a, Adj::NextOrSame(_)) && cal == *dow as i64 {
                return Ok(t0.clone());
            }
            let diff = cal - *dow as i64;
            t0.plus_unit(if diff >= 0 { 7 - diff } else { -diff }, U_DAYS)?
        }
        Adj::Previous(dow) | Adj::PreviousOrSame(dow) => {
            let cal = dow_of(t0)?;
            if matches!(a, Adj::PreviousOrSame(_)) && cal == *dow as i64 {
                return Ok(t0.clone());
            }
            let diff = *dow as i64 - cal;
            t0.plus_unit(-(if diff >= 0 { 7 - diff } else { -diff }), U_DAYS)?
        }
        Adj::DateAdjuster(f) => {
            let d = conv_ldate(&t0.clone().to_val()).map_err(|e| fr(e, &[jf!("java.time.temporal.TemporalAdjusters", "lambda$ofDateAdjuster$0", "TemporalAdjusters", 142)]))?;
            let r = rt.call_fn(f, &[t(JTime::LocalDate(d))])?;
            let nd = ldate_arg(&r).map_err(|_| frs(class_cast(format!("class {} cannot be cast to class java.time.LocalDate ({} and java.time.LocalDate are in module java.base of loader 'bootstrap')", r.type_name(), r.type_name())), &[jf!("java.time.temporal.TemporalAdjusters", "lambda$ofDateAdjuster$0", "TemporalAdjusters", 143)]))?;
            t0.with_field(F::EpochDay, nd.epoch_day())?
        }
    })
}

// ---------------------------------------------------------------------------
// Duration / Period text forms

fn parse_duration(s: &str) -> SResult<(i64, u32)> {
    let err = || dtpe("Text cannot be parsed to a Duration");
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut negate = false;
    if matches!(b.first(), Some('+') | Some('-')) {
        negate = b[0] == '-';
        i += 1;
    }
    if !matches!(b.get(i), Some('P') | Some('p')) {
        return Err(err());
    }
    i += 1;
    let number = |i: &mut usize| -> Option<(bool, String)> {
        let start = *i;
        let mut j = *i;
        let mut neg = false;
        if matches!(b.get(j), Some('+') | Some('-')) {
            neg = b[j] == '-';
            j += 1;
        }
        let ds = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == ds {
            *i = start;
            return None;
        }
        *i = j;
        Some((neg, b[ds..j].iter().collect()))
    };
    let to_i64 = |neg: bool, digits: &str, what: &str| -> SResult<i64> {
        let v: i64 = digits.parse::<i64>().map_err(|_| dtpe(format!("Text cannot be parsed to a Duration: {what}")))?;
        if neg {
            v.checked_neg().ok_or_else(|| dtpe(format!("Text cannot be parsed to a Duration: {what}")))
        } else {
            Ok(v)
        }
    };
    let mut any = false;
    let mut days = 0i64;
    let (mut hours, mut minutes, mut seconds, mut nanos) = (0i64, 0i64, 0i64, 0i64);
    let save = i;
    if let Some((neg, d)) = number(&mut i) {
        if matches!(b.get(i), Some('D') | Some('d')) {
            i += 1;
            days = to_i64(neg, &d, "days")?;
            any = true;
        } else {
            i = save;
        }
    }
    if matches!(b.get(i), Some('T') | Some('t')) {
        i += 1;
        let mut tany = false;
        let save = i;
        if let Some((neg, d)) = number(&mut i) {
            if matches!(b.get(i), Some('H') | Some('h')) {
                i += 1;
                hours = to_i64(neg, &d, "hours")?;
                tany = true;
            } else {
                i = save;
            }
        }
        let save = i;
        if let Some((neg, d)) = number(&mut i) {
            if matches!(b.get(i), Some('M') | Some('m')) {
                i += 1;
                minutes = to_i64(neg, &d, "minutes")?;
                tany = true;
            } else {
                i = save;
            }
        }
        let save = i;
        if let Some((neg, d)) = number(&mut i) {
            let mut frac = String::new();
            if matches!(b.get(i), Some('.') | Some(',')) {
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_digit() && frac.len() < 9 {
                    frac.push(b[j]);
                    j += 1;
                }
                i = j;
            }
            if matches!(b.get(i), Some('S') | Some('s')) {
                i += 1;
                seconds = to_i64(neg, &d, "seconds")?;
                if !frac.is_empty() {
                    let mut f = frac.clone();
                    while f.len() < 9 {
                        f.push('0');
                    }
                    nanos = f.parse::<i64>().map_err(|_| dtpe("Text cannot be parsed to a Duration: fraction"))?;
                    if neg {
                        nanos = -nanos;
                    }
                }
                tany = true;
            } else {
                i = save;
            }
        }
        if !tany {
            return Err(err());
        }
        any = true;
    }
    if !any || i != b.len() {
        return Err(err());
    }
    let ovf = || dtpe("Text cannot be parsed to a Duration: overflow");
    let mut secs = days.checked_mul(SPD).ok_or_else(ovf)?;
    secs = secs.checked_add(hours.checked_mul(3600).ok_or_else(ovf)?).ok_or_else(ovf)?;
    secs = secs.checked_add(minutes.checked_mul(60).ok_or_else(ovf)?).ok_or_else(ovf)?;
    secs = secs.checked_add(seconds).ok_or_else(ovf)?;
    let d = dur_create(secs, nanos).map_err(|_| ovf())?;
    if negate {
        return dur_negate(d);
    }
    Ok(d)
}
fn dur_negate(d: (i64, u32)) -> SResult<(i64, u32)> {
    dur_from_nanos(-dur_total_nanos(d))
}

fn parse_period(s: &str) -> SResult<(i32, i32, i32)> {
    let err = || dtpe("Text cannot be parsed to a Period");
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut negate = false;
    if matches!(b.first(), Some('+') | Some('-')) {
        negate = b[0] == '-';
        i += 1;
    }
    if !matches!(b.get(i), Some('P') | Some('p')) {
        return Err(err());
    }
    i += 1;
    let mut vals = [0i64; 4];
    let mut any = false;
    for (k, letter) in ['Y', 'M', 'W', 'D'].iter().enumerate() {
        let save = i;
        let mut j = i;
        let mut neg = false;
        if matches!(b.get(j), Some('+') | Some('-')) {
            neg = b[j] == '-';
            j += 1;
        }
        let ds = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > ds && b.get(j).is_some_and(|c| c.eq_ignore_ascii_case(letter)) {
            let digits: String = b[ds..j].iter().collect();
            let v: i64 = digits.parse().map_err(|_| err())?;
            if v > i32::MAX as i64 + 1 {
                return Err(err());
            }
            let v = if neg { -v } else { v };
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err(err());
            }
            vals[k] = v;
            i = j + 1;
            any = true;
        } else {
            i = save;
        }
    }
    if !any || i != b.len() {
        return Err(err());
    }
    let sgn = if negate { -1 } else { 1 };
    let days = vals[2].checked_mul(7).and_then(|w| w.checked_add(vals[3])).ok_or_else(err)?;
    let c = |v: i64| i32::try_from(v * sgn).map_err(|_| err());
    Ok((c(vals[0])?, c(vals[1])?, c(days)?))
}

// ---------------------------------------------------------------------------
// Statics

fn enum_array(class: &str, items: Vec<Val>) -> Val {
    Val::new_array(Ty::Ref(class.to_string()), items)
}

fn short_ids() -> Val {
    let pairs = [
        ("ACT", "Australia/Darwin"),
        ("AET", "Australia/Sydney"),
        ("AGT", "America/Argentina/Buenos_Aires"),
        ("ART", "Africa/Cairo"),
        ("AST", "America/Anchorage"),
        ("BET", "America/Sao_Paulo"),
        ("BST", "Asia/Dhaka"),
        ("CAT", "Africa/Harare"),
        ("CNT", "America/St_Johns"),
        ("CST", "America/Chicago"),
        ("CTT", "Asia/Shanghai"),
        ("EAT", "Africa/Addis_Ababa"),
        ("ECT", "Europe/Paris"),
        ("IET", "America/Indiana/Indianapolis"),
        ("IST", "Asia/Kolkata"),
        ("JST", "Asia/Tokyo"),
        ("MIT", "Pacific/Apia"),
        ("NET", "Asia/Yerevan"),
        ("NST", "Pacific/Auckland"),
        ("PLT", "Asia/Karachi"),
        ("PNT", "America/Phoenix"),
        ("PRT", "America/Puerto_Rico"),
        ("PST", "America/Los_Angeles"),
        ("SST", "Pacific/Guadalcanal"),
        ("VST", "Asia/Ho_Chi_Minh"),
        ("EST", "America/Panama"),
        ("MST", "America/Phoenix"),
        ("HST", "Pacific/Honolulu"),
    ];
    let mut m = indexmap::IndexMap::new();
    for (k, v) in pairs {
        m.insert(super::value::Key::str(k), Val::str(v));
    }
    Val::new_map(m)
}

/// `Class.FIELD`; `None` falls back to the built-in static fields.
pub fn static_field(class: &str, name: &str) -> Option<Val> {
    Some(match (class, name) {
        ("LocalDate", "MAX") => t(JTime::LocalDate(LDate { y: MAX_YEAR as i32, m: 12, d: 31 })),
        ("LocalDate", "MIN") => t(JTime::LocalDate(LDate { y: MIN_YEAR as i32, m: 1, d: 1 })),
        ("LocalTime", "MAX") => t(JTime::LocalTime(NPD - 1)),
        ("LocalTime", "MIN" | "MIDNIGHT") => t(JTime::LocalTime(0)),
        ("LocalTime", "NOON") => t(JTime::LocalTime(12 * 3600 * NPS)),
        ("LocalDateTime", "MAX") => t(JTime::LocalDateTime(LDT::of(LDate { y: MAX_YEAR as i32, m: 12, d: 31 }, NPD - 1))),
        ("LocalDateTime", "MIN") => t(JTime::LocalDateTime(LDT::of(LDate { y: MIN_YEAR as i32, m: 1, d: 1 }, 0))),
        ("OffsetDateTime", "MAX") => t(JTime::OffsetDateTime(LDT::of(LDate { y: MAX_YEAR as i32, m: 12, d: 31 }, NPD - 1), -18 * 3600)),
        ("OffsetDateTime", "MIN") => t(JTime::OffsetDateTime(LDT::of(LDate { y: MIN_YEAR as i32, m: 1, d: 1 }, 0), 18 * 3600)),
        ("OffsetTime", "MAX") => t(JTime::OffsetTime(NPD - 1, -18 * 3600)),
        ("OffsetTime", "MIN") => t(JTime::OffsetTime(0, 18 * 3600)),
        ("Instant", "EPOCH") => t(JTime::Instant(0, 0)),
        ("Instant", "MAX") => t(JTime::Instant(INSTANT_MAX_SEC, 999_999_999)),
        ("Instant", "MIN") => t(JTime::Instant(INSTANT_MIN_SEC, 0)),
        ("Duration", "ZERO") => t(JTime::Duration(0, 0)),
        ("Period", "ZERO") => t(JTime::Period(0, 0, 0)),
        ("ZoneOffset", "UTC") => t(JTime::Zone(Zone::Offset(0))),
        ("ZoneOffset", "MAX") => t(JTime::Zone(Zone::Offset(18 * 3600))),
        ("ZoneOffset", "MIN") => t(JTime::Zone(Zone::Offset(-18 * 3600))),
        ("ZoneId", "SHORT_IDS") => short_ids(),
        ("Year", "MAX_VALUE") => Val::Int(MAX_YEAR as i32),
        ("Year", "MIN_VALUE") => Val::Int(MIN_YEAR as i32),
        ("DateTimeFormatter", f) => Val::Time(Rc::new(JTime::Formatter(iso_formatter(f)?))),
        ("TextStyle", s) => t(JTime::TextStyle(TEXT_STYLES.iter().position(|x| *x == s)? as u8)),
        ("ResolverStyle", s) => t(JTime::ResolverStyle(RESOLVER_STYLES.iter().position(|x| *x == s)? as u8)),
        ("FormatStyle", s) => t(JTime::FormatStyle(FORMAT_STYLES.iter().position(|x| *x == s)? as u8)),
        ("Locale", l) => t(JTime::Locale(Locale::constant(l)?)),
        ("ChronoField", f) => t(JTime::Field(F::from_enum_name(f)?)),
        ("IsoFields", "DAY_OF_QUARTER") => t(JTime::Field(F::DayOfQuarter)),
        ("IsoFields", "QUARTER_OF_YEAR") => t(JTime::Field(F::QuarterOfYear)),
        ("IsoFields", "WEEK_OF_WEEK_BASED_YEAR") => t(JTime::Field(F::WeekOfWeekBasedYear)),
        ("IsoFields", "WEEK_BASED_YEAR") => t(JTime::Field(F::WeekBasedYear)),
        ("IsoFields", "QUARTER_YEARS") => t(JTime::Unit(U_QUARTER_YEARS)),
        ("IsoFields", "WEEK_BASED_YEARS") => t(JTime::Unit(U_WEEK_BASED_YEARS)),
        ("IsoChronology", "INSTANCE") => t(JTime::Chronology),
        _ => return None,
    })
}

fn ints3(args: &[Val]) -> SResult<(i64, i64, i64)> {
    Ok((int_arg(args, 0)?, int_arg(args, 1)?, int_arg(args, 2)?))
}

fn ldt_of_args(args: &[Val]) -> SResult<LDT> {
    let (y, mo, d) = ints3(args)?;
    let h = int_arg(args, 3)?;
    let mi = int_arg(args, 4)?;
    let s = if args.len() > 5 { int_arg(args, 5)? } else { 0 };
    let n = if args.len() > 6 { int_arg(args, 6)? } else { 0 };
    let date = LDate::of(y, mo, d)?;
    let nod = time_of_at(h, mi, s, n, match args.len() { 5 => &LT_OF_2, 6 => &LT_OF_3, _ => &LT_OF_4 })?;
    Ok(LDT::of(date, nod))
}

fn parse_with(args: &[Val], default: &str, target: &str) -> SResult<Val> {
    let text = str_arg(args, 0, "text")?;
    let f = if args.len() > 1 { fmt_arg(&args[1], "formatter")? } else { iso_formatter(default).expect("iso formatter") };
    f.parse_to(&text, target)
}
fn pattern_formatter(p: &str, strict: bool) -> Rc<Fmt> {
    let mut f = Fmt::of_pattern(p, Locale::en()).unwrap_or_else(|_| Fmt::new(Vec::new()));
    if strict {
        f.resolver = 0;
    }
    Rc::new(f)
}
fn parse_with_pattern(args: &[Val], pattern: &str, target: &str) -> SResult<Val> {
    let text = str_arg(args, 0, "text")?;
    let f = if args.len() > 1 { fmt_arg(&args[1], "formatter")? } else { pattern_formatter(pattern, false) };
    f.parse_to(&text, target)
}

/// `Class.method(args)` for the classes in [`CLASSES`]; `Ok(None)` falls
/// back to the built-in statics.
pub fn call_static(rt: &mut Rt, class: &str, method: &str, args: &[Val]) -> SResult<Option<Val>> {
    call_static0(rt, class, method, args).map_err(|e| static_frames(class, method, args, e))
}

/// Outer JDK frames of a static method failure.
fn static_frames(class: &str, method: &str, args: &[Val], e: Box<ScriptErr>) -> Box<ScriptErr> {
    let n = args.len();
    let m = e.msg.as_str();
    let parse_outer = |a: Fr, b: Fr| -> Vec<Fr> {
        let mut v = vec![if m.contains("could not be parsed:") { DTF_PARSE_Q_ERR } else { DTF_PARSE_Q }, a];
        if n == 1 {
            v.push(b);
        }
        v
    };
    let outer: Vec<Fr> = match (class, method) {
        ("LocalDate", "parse") => parse_outer(jt!("LocalDate", "parse", 455), jt!("LocalDate", "parse", 440)),
        ("LocalDateTime", "parse") => parse_outer(jt!("LocalDateTime", "parse", 496), jt!("LocalDateTime", "parse", 481)),
        ("LocalTime", "parse") => parse_outer(jt!("LocalTime", "parse", 475), jt!("LocalTime", "parse", 460)),
        ("Instant", "parse") => vec![if m.contains("could not be parsed:") { DTF_PARSE_Q_ERR } else { DTF_PARSE_Q }, jt!("Instant", "parse", 398)],
        ("ZonedDateTime", "parse") => parse_outer(jt!("ZonedDateTime", "parse", 602), jt!("ZonedDateTime", "parse", 587)),
        ("OffsetDateTime", "parse") => parse_outer(jt!("OffsetDateTime", "parse", 407), jt!("OffsetDateTime", "parse", 392)),
        ("OffsetTime", "parse") => parse_outer(jt!("OffsetTime", "parse", 331), jt!("OffsetTime", "parse", 316)),
        ("YearMonth", "parse") => parse_outer(jt!("YearMonth", "parse", 315), jt!("YearMonth", "parse", 300)),
        ("Year", "parse") => parse_outer(jt!("Year", "parse", 294), jt!("Year", "parse", 279)),
        ("MonthDay", "parse") => parse_outer(jt!("MonthDay", "parse", 329), jt!("MonthDay", "parse", 314)),
        ("DateTimeFormatter", "ofPattern") => vec![if n == 1 { DTF_OF_PATTERN_1 } else { DTF_OF_PATTERN_2 }],
        ("LocalDateTime", "of") if n >= 5 => {
            let date_part = m.contains("Year") || m.contains("MonthOfYear") || m.contains("DayOfMonth") || m.starts_with("Invalid date");
            vec![match (n, date_part) {
                (5, true) => LDT_OF_315,
                (5, false) => LDT_OF_316,
                (6, true) => LDT_OF_340,
                (6, false) => LDT_OF_341,
                (_, true) => LDT_OF_365,
                _ => LDT_OF_366,
            }]
        }
        ("OffsetDateTime", "of") if n == 8 => vec![jt!("OffsetDateTime", "of", 311)],
        ("ZonedDateTime", "of") if n == 8 => vec![jt!("ZonedDateTime", "of", 343)],
        ("OffsetTime", "of") if n == 5 => vec![jt!("OffsetTime", "of", 240)],
        ("LocalTime", "ofSecondOfDay") => vec![jt!("LocalTime", "ofSecondOfDay", 392)],
        ("LocalTime", "ofNanoOfDay") => vec![jt!("LocalTime", "ofNanoOfDay", 410)],
        ("YearMonth", "of") => vec![if m.contains("for Year") { jt!("YearMonth", "of", 246) } else { jt!("YearMonth", "of", 247) }],
        ("Year", "of") => vec![jt!("Year", "of", 229)],
        ("MonthDay", "of") => {
            if m.starts_with("Invalid value for MonthOfYear") {
                vec![MD_OF_264]
            } else if m.starts_with("Illegal value") {
                vec![MD_OF_242, MD_OF_264]
            } else {
                vec![MD_OF_239, MD_OF_264]
            }
        }
        ("Month", "of") => vec![MON_OF_187],
        ("DayOfWeek", "of") => vec![DOW_OF_165],
        ("Month", "valueOf") => vec![if m == "Name is null" { ENUM_VALUEOF_NULL } else { ENUM_VALUEOF }, jt!("Month", "valueOf", 106)],
        ("DayOfWeek", "valueOf") => vec![if m == "Name is null" { ENUM_VALUEOF_NULL } else { ENUM_VALUEOF }, jt!("DayOfWeek", "valueOf", 109)],
        ("ChronoUnit", "valueOf") => vec![ENUM_VALUEOF, jf!("java.time.temporal.ChronoUnit", "valueOf", "ChronoUnit", 77)],
        ("ChronoField", "valueOf") => vec![ENUM_VALUEOF, jf!("java.time.temporal.ChronoField", "valueOf", "ChronoField", 102)],
        ("TextStyle", "valueOf") => vec![ENUM_VALUEOF, jf!("java.time.format.TextStyle", "valueOf", "TextStyle", 88)],
        ("ResolverStyle", "valueOf") => vec![ENUM_VALUEOF, jf!("java.time.format.ResolverStyle", "valueOf", "ResolverStyle", 77)],
        ("FormatStyle", "valueOf") => vec![ENUM_VALUEOF, jf!("java.time.format.FormatStyle", "valueOf", "FormatStyle", 75)],
        ("Instant", "ofEpochSecond") => vec![if n == 1 { jt!("Instant", "ofEpochSecond", 308) } else { jt!("Instant", "ofEpochSecond", 334) }],
        ("Instant", "ofEpochMilli") => vec![jt!("Instant", "ofEpochMilli", 349)],
        ("Duration", "ofDays") => vec![jt!("Duration", "ofDays", 214)],
        ("Duration", "ofHours") => vec![jt!("Duration", "ofHours", 229)],
        ("Duration", "ofMinutes") => vec![jt!("Duration", "ofMinutes", 244)],
        ("Duration", "of") => vec![jt!("Duration", "of", 343)],
        ("Duration", "from") => vec![jt!("Duration", "from", 371)],
        ("Duration", "between") => vec![DUR_BETWEEN],
        ("Duration", "parse") => {
            if m == "text" {
                vec![jt!("Duration", "parse", 422)]
            } else if m.ends_with(": seconds") {
                vec![jt!("Duration", "parseNumber", 466), jt!("Duration", "parse", 439)]
            } else if m.ends_with(": days") {
                vec![jt!("Duration", "parseNumber", 466), jt!("Duration", "parse", 436)]
            } else if m.ends_with(": hours") {
                vec![jt!("Duration", "parseNumber", 466), jt!("Duration", "parse", 437)]
            } else if m.ends_with(": minutes") {
                vec![jt!("Duration", "parseNumber", 466), jt!("Duration", "parse", 438)]
            } else {
                vec![jt!("Duration", "parse", 450)]
            }
        }
        ("Period", "parse") => {
            if m == "text" {
                vec![jt!("Period", "parse", 328)]
            } else if e.frames.is_empty() && m.ends_with("Period") && args.first().is_some_and(|a| a.to_java_string().chars().filter(|c| c.is_ascii_digit()).count() > 9) {
                vec![jt!("Period", "parse", 345)]
            } else {
                vec![jt!("Period", "parse", 349)]
            }
        }
        ("Period", "from") => vec![jt!("Period", "from", 281)],
        ("Period", "ofWeeks") => vec![jt!("Period", "ofWeeks", 208)],
        ("Period", "between") => vec![jt!("LocalDate", "until", 1706), jt!("Period", "between", 389)],
        ("ZoneId", "of") if m == "zoneId" => vec![jt!("ZoneId", "of", 403), ZID_OF_359],
        ("ZoneId", "ofOffset") => vec![jt!("ZoneId", "ofOffset", 383)],
        ("ZoneOffset", "of") if m == "offsetId" => vec![jt!("ZoneOffset", "of", 222)],
        ("ZoneOffset", "ofHours") => vec![jt!("ZoneOffset", "ofHours", 302)],
        ("ZoneOffset", "ofHoursMinutes") => vec![jt!("ZoneOffset", "ofHoursMinutes", 319)],
        ("ZonedDateTime", "ofInstant") if n == 2 => vec![ZDT_OF_INSTANT_412],
        ("ZonedDateTime", "of") if n == 2 && e.kind == "null_pointer_exception" => vec![if m == "zone" { ZDT_OF_LOCAL_372 } else { jt!("ZonedDateTime", "ofLocal", 371) }, ZDT_OF_297],
        ("ZonedDateTime", "ofLocal") if m == "zone" => vec![ZDT_OF_LOCAL_372],
        ("ZonedDateTime", "from") if e.kind == "null_pointer_exception" => vec![jt!("ZoneId", "from", 463), jt!("ZonedDateTime", "from", 558)],
        ("Clock", "fixed") => vec![if m == "zone" { jt!("Clock", "fixed", 349) } else { jt!("Clock", "fixed", 348) }],
        ("Clock", "offset") => vec![jt!("Clock", "offset", 374)],
        ("TemporalAdjusters", "next") => vec![jf!("java.time.temporal.TemporalAdjusters", "next", "TemporalAdjusters", 387)],
        ("TemporalAdjusters", "dayOfWeekInMonth") => vec![jf!("java.time.temporal.TemporalAdjusters", "dayOfWeekInMonth", "TemporalAdjusters", 347)],
        ("TemporalAdjusters", "ofDateAdjuster") => vec![jf!("java.time.temporal.TemporalAdjusters", "ofDateAdjuster", "TemporalAdjusters", 140)],
        ("DateTimeFormatter", "ofLocalizedDate") => vec![jf!("java.time.format.DateTimeFormatter", "ofLocalizedDate", "DateTimeFormatter", 620)],
        _ => Vec::new(),
    };
    let e = if e.kind == "null_pointer_exception" && e.frames.is_empty() && !outer.is_empty() && matches!(method, "parse" | "fixed" | "offset" | "dayOfWeekInMonth" | "ofDateAdjuster" | "ofLocalizedDate" | "ofPattern" | "ofOffset") == false && (m == "text" || m == "zone" || m == "instant" || m == "zoneId" || m == "offsetId" || m == "localDateTime") {
        fr(e, &[REQ_NON_NULL])
    } else if e.kind == "null_pointer_exception" && e.frames.is_empty() && (m == "text" || m == "fixedInstant" || m == "zone" || m == "baseClock" || m == "dayOfWeek" || m == "dateBasedAdjuster" || m == "dateStyle") && !outer.is_empty() {
        fr(e, &[REQ_NON_NULL])
    } else {
        e
    };
    fr(e, &outer)
}

fn call_static0(_rt: &mut Rt, class: &str, method: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    let a = |i: usize| at(args, i);
    let v = match (class, method, n) {
        // ---- LocalDate
        ("LocalDate", "of", 3) => {
            let (y, m, d) = ints3(args)?;
            t(JTime::LocalDate(LDate::of(y, m, d)?))
        }
        ("LocalDate", "ofEpochDay", 1) => t(JTime::LocalDate(LDate::from_epoch_day(long_arg(args, 0)?)?)),
        ("LocalDate", "ofYearDay", 2) => t(JTime::LocalDate(LDate::of_year_day(int_arg(args, 0)?, int_arg(args, 1)?)?)),
        ("LocalDate", "parse", 1 | 2) => parse_with(args, "ISO_LOCAL_DATE", "LocalDate")?,
        ("LocalDate", "from", 1) if a(0).is_null() => return Err(unable_null("LocalDate")),
        ("LocalDate", "from", 1) => t(JTime::LocalDate(conv_ldate(&a(0))?)),
        // ---- LocalDateTime
        ("LocalDateTime", "of", 5..=7) => t(JTime::LocalDateTime(ldt_of_args(args)?)),
        ("LocalDateTime", "of", 2) => t(JTime::LocalDateTime(LDT::of(ldate_arg(&a(0))?, ltime_arg(&a(1))?))),
        ("LocalDateTime", "ofEpochSecond", 3) => {
            let secs = long_arg(args, 0)?;
            let nanos = F::NanoOfSecond.check(int_arg(args, 1)?)?;
            let off = offset_arg(&a(2), "offset")?;
            t(JTime::LocalDateTime(LDT::from_epoch(secs, nanos as u32, off)?))
        }
        ("LocalDateTime", "ofInstant", 2) => {
            let i = instant_arg(&a(0))?;
            let z = zone_arg(&a(1), "zone")?;
            t(JTime::LocalDateTime(Zoned::of_instant(i, z)?.ldt))
        }
        ("LocalDateTime", "parse", 1 | 2) => parse_with(args, "ISO_LOCAL_DATE_TIME", "LocalDateTime")?,
        ("LocalDateTime", "from", 1) => t(JTime::LocalDateTime(conv_ldt(&a(0))?)),
        // ---- LocalTime
        ("LocalTime", "of", 2..=4) => {
            let h = int_arg(args, 0)?;
            let m = int_arg(args, 1)?;
            let s = if n > 2 { int_arg(args, 2)? } else { 0 };
            let ns = if n > 3 { int_arg(args, 3)? } else { 0 };
            t(JTime::LocalTime(time_of_at(h, m, s, ns, match n { 2 => &LT_OF_2, 3 => &LT_OF_3, _ => &LT_OF_4 })?))
        }
        ("LocalTime", "ofNanoOfDay", 1) => t(JTime::LocalTime(F::NanoOfDay.check(long_arg(args, 0)?)?)),
        ("LocalTime", "ofSecondOfDay", 1) => t(JTime::LocalTime(F::SecondOfDay.check(long_arg(args, 0)?)? * NPS)),
        ("LocalTime", "parse", 1 | 2) => parse_with(args, "ISO_LOCAL_TIME", "LocalTime")?,
        ("LocalTime", "from", 1) => t(JTime::LocalTime(conv_ltime(&a(0))?)),
        // ---- OffsetDateTime
        ("OffsetDateTime", "of", 2) => t(JTime::OffsetDateTime(ldt_arg(&a(0))?, offset_arg(&a(1), "offset")?)),
        ("OffsetDateTime", "of", 3) => t(JTime::OffsetDateTime(LDT::of(ldate_arg(&a(0))?, ltime_arg(&a(1))?), offset_arg(&a(2), "offset")?)),
        ("OffsetDateTime", "of", 8) => t(JTime::OffsetDateTime(ldt_of_args(&args[..7])?, offset_arg(&a(7), "offset")?)),
        ("OffsetDateTime", "ofInstant", 2) => {
            let i = instant_arg(&a(0))?;
            let z = zone_arg(&a(1), "zone")?;
            let zd = Zoned::of_instant(i, z)?;
            t(JTime::OffsetDateTime(zd.ldt, zd.off))
        }
        ("OffsetDateTime", "parse", 1 | 2) => parse_with(args, "ISO_OFFSET_DATE_TIME", "OffsetDateTime")?,
        ("OffsetDateTime", "from", 1) => {
            let (d, o) = conv_odt(&a(0))?;
            t(JTime::OffsetDateTime(d, o))
        }
        // ---- OffsetTime
        ("OffsetTime", "of", 2) => t(JTime::OffsetTime(ltime_arg(&a(0))?, offset_arg(&a(1), "offset")?)),
        ("OffsetTime", "of", 5) => {
            let nod = time_of(int_arg(args, 0)?, int_arg(args, 1)?, int_arg(args, 2)?, int_arg(args, 3)?)?;
            t(JTime::OffsetTime(nod, offset_arg(&a(4), "offset")?))
        }
        ("OffsetTime", "ofInstant", 2) => {
            let i = instant_arg(&a(0))?;
            let z = zone_arg(&a(1), "zone")?;
            let zd = Zoned::of_instant(i, z)?;
            t(JTime::OffsetTime(zd.ldt.nod, zd.off))
        }
        ("OffsetTime", "parse", 1 | 2) => parse_with(args, "ISO_OFFSET_TIME", "OffsetTime")?,
        ("OffsetTime", "from", 1) => {
            let (nod, o) = conv_otime(&a(0))?;
            t(JTime::OffsetTime(nod, o))
        }
        // ---- ZonedDateTime
        ("ZonedDateTime", "of", 2) => Zoned::of_local(ldt_arg(&a(0))?, zone_arg(&a(1), "zone")?, None)?.to_val(),
        ("ZonedDateTime", "of", 3) => Zoned::of_local(LDT::of(ldate_arg(&a(0))?, ltime_arg(&a(1))?), zone_arg(&a(2), "zone")?, None)?.to_val(),
        ("ZonedDateTime", "of", 8) => Zoned::of_local(ldt_of_args(&args[..7])?, zone_arg(&a(7), "zone")?, None)?.to_val(),
        ("ZonedDateTime", "ofInstant", 2) => Zoned::of_instant(instant_arg(&a(0))?, zone_arg(&a(1), "zone")?)?.to_val(),
        ("ZonedDateTime", "ofInstant", 3) => {
            let ldt = ldt_arg(&a(0))?;
            let off = offset_arg(&a(1), "offset")?;
            let z = zone_arg(&a(2), "zone")?;
            Zoned::of_instant((ldt.epoch_second(off), ldt.nanos()), z)?.to_val()
        }
        ("ZonedDateTime", "ofLocal", 3) => {
            let ldt = ldt_arg(&a(0))?;
            let z = zone_arg(&a(1), "zone")?;
            let pref = if a(2).is_null() { None } else { Some(offset_arg(&a(2), "preferredOffset")?) };
            Zoned::of_local(ldt, z, pref)?.to_val()
        }
        ("ZonedDateTime", "ofStrict", 3) => Zoned::of_strict(ldt_arg(&a(0))?, offset_arg(&a(1), "offset")?, zone_arg(&a(2), "zone")?)?.to_val(),
        ("ZonedDateTime", "parse", 1 | 2) => parse_with(args, "ISO_ZONED_DATE_TIME", "ZonedDateTime")?,
        ("ZonedDateTime", "from", 1) if a(0).is_null() => return Err(npe("Cannot invoke \"java.time.temporal.TemporalAccessor.query(java.time.temporal.TemporalQuery)\" because \"temporal\" is null")),
        ("ZonedDateTime", "from", 1) => conv_zoned(&a(0))?.to_val(),
        // ---- Instant
        ("Instant", "ofEpochMilli", 1) => {
            let ms = long_arg(args, 0)?;
            inst_to_val(inst_create(ms.div_euclid(1000), ms.rem_euclid(1000) * 1_000_000)?)
        }
        ("Instant", "ofEpochSecond", 1) => inst_to_val(inst_create(long_arg(args, 0)?, 0)?),
        ("Instant", "ofEpochSecond", 2) => inst_to_val(inst_create(long_arg(args, 0)?, long_arg(args, 1)?)?),
        ("Instant", "parse", 1) => parse_with(args, "ISO_INSTANT", "Instant")?,
        ("Instant", "from", 1) if a(0).is_null() => return Err(unable_null("Instant")),
        ("Instant", "from", 1) => inst_to_val(conv_instant(&a(0))?),
        // ---- Duration
        ("Duration", "between", 2) => {
            let start = T::of(&a(0)).ok_or_else(|| {
                if a(0).is_null() {
                    npe("Cannot invoke \"java.time.temporal.Temporal.until(java.time.temporal.Temporal, java.time.temporal.TemporalUnit)\" because \"startInclusive\" is null")
                } else {
                    cast_to(&a(0), "java.time.temporal.Temporal")
                }
            })?;
            match start.until(&a(1), U_NANOS) {
                Ok(nanos) => t(JTime::Duration(nanos.div_euclid(NPS), nanos.rem_euclid(NPS) as u32)),
                Err(_) => {
                    let mut secs = start.until(&a(1), U_SECONDS)?;
                    let end = T::of(&a(1));
                    let nanos = match (start.get(F::NanoOfSecond), end.and_then(|e| e.get(F::NanoOfSecond))) {
                        (Some(s), Some(e)) => {
                            let d = e - s;
                            if secs > 0 && d < 0 {
                                secs += 1;
                            } else if secs < 0 && d > 0 {
                                secs -= 1;
                            }
                            d
                        }
                        _ => 0,
                    };
                    let d = dur_create(secs, nanos)?;
                    t(JTime::Duration(d.0, d.1))
                }
            }
        }
        ("Duration", "from", 1) => match a(0) {
            Val::Time(j) => match &*j {
                JTime::Duration(s, nn) => t(JTime::Duration(*s, *nn)),
                JTime::Period(..) => return Err(utte("Unit must not have an estimated duration")),
                _ => return Err(cast_to(&a(0), "java.time.temporal.TemporalAmount")),
            },
            Val::Null => return Err(npe("amount")),
            o => return Err(cast_to(&o, "java.time.temporal.TemporalAmount")),
        },
        ("Duration", "of", 2) => {
            let d = dur_plus_unit((0, 0), long_arg(args, 0)?, unit_arg(&a(1))?)?;
            t(JTime::Duration(d.0, d.1))
        }
        ("Duration", "ofDays" | "ofHours" | "ofMinutes" | "ofSeconds" | "ofMillis" | "ofNanos", 1) => {
            let v = long_arg(args, 0)?;
            let d = match method {
                "ofDays" => (v.checked_mul(SPD).ok_or_else(overflow_mul)?, 0),
                "ofHours" => (v.checked_mul(3600).ok_or_else(overflow_mul)?, 0),
                "ofMinutes" => (v.checked_mul(60).ok_or_else(overflow_mul)?, 0),
                "ofSeconds" => (v, 0),
                "ofMillis" => (v.div_euclid(1000), (v.rem_euclid(1000) * 1_000_000) as u32),
                _ => dur_create(0, v)?,
            };
            t(JTime::Duration(d.0, d.1))
        }
        ("Duration", "ofSeconds", 2) => {
            let d = dur_create(long_arg(args, 0)?, long_arg(args, 1)?)?;
            t(JTime::Duration(d.0, d.1))
        }
        ("Duration", "parse", 1) => {
            let d = parse_duration(&str_arg(args, 0, "text")?)?;
            t(JTime::Duration(d.0, d.1))
        }
        // ---- Period
        ("Period", "between", 2) => {
            let s = ldate_arg(&a(0))?;
            let e = ldate_arg(&a(1))?;
            let p = s.period_until(&e)?;
            t(JTime::Period(p.0, p.1, p.2))
        }
        ("Period", "from", 1) => match a(0) {
            Val::Time(j) => match &*j {
                JTime::Period(y, m, d) => t(JTime::Period(*y, *m, *d)),
                JTime::Duration(..) => return Err(dte("Unit must be Years, Months or Days, but was Seconds")),
                _ => return Err(cast_to(&a(0), "java.time.temporal.TemporalAmount")),
            },
            Val::Null => return Err(npe("amount")),
            o => return Err(cast_to(&o, "java.time.temporal.TemporalAmount")),
        },
        ("Period", "of", 3) => {
            let (y, m, d) = ints3(args)?;
            t(JTime::Period(y as i32, m as i32, d as i32))
        }
        ("Period", "ofDays", 1) => t(JTime::Period(0, 0, int_arg(args, 0)? as i32)),
        ("Period", "ofMonths", 1) => t(JTime::Period(0, int_arg(args, 0)? as i32, 0)),
        ("Period", "ofYears", 1) => t(JTime::Period(int_arg(args, 0)? as i32, 0, 0)),
        ("Period", "ofWeeks", 1) => {
            let w = int_arg(args, 0)? as i32;
            t(JTime::Period(0, 0, w.checked_mul(7).ok_or_else(|| frs(arith("integer overflow"), &[MATH_MUL_INT]))?))
        }
        ("Period", "parse", 1) => {
            let p = parse_period(&str_arg(args, 0, "text")?)?;
            t(JTime::Period(p.0, p.1, p.2))
        }
        // ---- ZoneId / ZoneOffset
        ("ZoneId", "of", 1) => t(JTime::Zone(Zone::parse(&str_arg(args, 0, "zoneId")?)?)),
        ("ZoneId", "of", 2) => {
            let id = str_arg(args, 0, "zoneId")?;
            let alias = match a(1) {
                Val::Map(m) => m.borrow().get(&super::value::Key::str(&id)).map(|v| v.to_java_string()),
                Val::Null => return Err(npe("aliasMap")),
                o => return Err(cast_to(&o, "java.util.Map")),
            };
            t(JTime::Zone(Zone::parse(&alias.unwrap_or(id))?))
        }
        ("ZoneId", "ofOffset", 2) => {
            let prefix = str_arg(args, 0, "prefix")?;
            let off = offset_arg(&a(1), "offset")?;
            if !matches!(prefix.as_str(), "" | "GMT" | "UTC" | "UT") {
                return Err(illegal_arg(format!("prefix should be GMT, UTC or UT, is: {prefix}")));
            }
            t(JTime::Zone(Zone::of_offset(&prefix, off)))
        }
        ("ZoneId", "from", 1) => t(JTime::Zone(conv_zone(&a(0))?)),
        ("ZoneId", "systemDefault", 0) => t(JTime::Zone(Zone::Fixed("UTC".into(), 0))),
        ("ZoneId", "getAvailableZoneIds", 0) => {
            let items: Vec<Val> = chrono_tz::TZ_VARIANTS.iter().map(|z| Val::str(z.name())).collect();
            Val::new_set(super::value::CollKind::HashSet, items)
        }
        ("ZoneOffset", "of", 1) => t(JTime::Zone(Zone::Offset(parse_offset_id(&str_arg(args, 0, "offsetId")?)?))),
        ("ZoneOffset", "ofHours", 1) => t(JTime::Zone(Zone::Offset(offset_of_hms(int_arg(args, 0)? as i32, 0, 0)?))),
        ("ZoneOffset", "ofHoursMinutes", 2) => t(JTime::Zone(Zone::Offset(offset_of_hms(int_arg(args, 0)? as i32, int_arg(args, 1)? as i32, 0)?))),
        ("ZoneOffset", "ofHoursMinutesSeconds", 3) => {
            let (h, m, s) = ints3(args)?;
            t(JTime::Zone(Zone::Offset(offset_of_hms(h as i32, m as i32, s as i32)?)))
        }
        ("ZoneOffset", "ofTotalSeconds", 1) => t(JTime::Zone(Zone::Offset(offset_of_total(int_arg(args, 0)?)?))),
        ("ZoneOffset", "from", 1) => t(JTime::Zone(Zone::Offset(conv_offset(&a(0))?))),
        // ---- YearMonth / Year / MonthDay
        ("YearMonth", "of", 2) => {
            let y = F::Year.check(int_arg(args, 0)?)?;
            let m = F::MonthOfYear.check(int_arg(args, 1)?)?;
            t(JTime::YearMonth(y as i32, m as u8))
        }
        ("YearMonth", "parse", 1 | 2) => parse_with_pattern(args, "uuuu-MM", "YearMonth")?,
        ("YearMonth", "from", 1) => {
            let (y, m) = conv_ym(&a(0))?;
            t(JTime::YearMonth(y, m))
        }
        ("Year", "of", 1) => t(JTime::Year(F::Year.check(int_arg(args, 0)?)? as i32)),
        ("Year", "isLeap", 1) => Val::Bool(is_leap(long_arg(args, 0)?)),
        ("Year", "parse", 1 | 2) => {
            let text = str_arg(args, 0, "text")?;
            let f = if n > 1 { fmt_arg(&args[1], "formatter")? } else { Rc::new(Fmt::new(vec![num(F::Year, 4, 10, Sign::ExceedsPad)])) };
            f.parse_to(&text, "Year")?
        }
        ("Year", "from", 1) => t(JTime::Year(conv_year(&a(0))?)),
        ("MonthDay", "of", 2) => {
            let m = F::MonthOfYear.check(int_arg(args, 0)?)?;
            let d = F::DayOfMonth.check(int_arg(args, 1)?)?;
            if d > month_len(m as u8, true) as i64 {
                return Err(dte(format!("Illegal value for DayOfMonth field, value {d} is not valid for month {}", MONTHS[m as usize - 1])));
            }
            t(JTime::MonthDay(m as u8, d as u8))
        }
        ("MonthDay", "parse", 1 | 2) => parse_with_pattern(args, "--MM-dd", "MonthDay")?,
        ("MonthDay", "from", 1) => {
            let (m, d) = conv_md(&a(0))?;
            t(JTime::MonthDay(m, d))
        }
        // ---- DateTimeFormatter
        ("DateTimeFormatter", "ofPattern", 1) => t(JTime::Formatter(Rc::new(Fmt::of_pattern(&str_arg(args, 0, "pattern")?, Locale::en())?))),
        ("DateTimeFormatter", "ofPattern", 2) => {
            let p = str_arg(args, 0, "pattern")?;
            let l = locale_arg(&a(1))?;
            t(JTime::Formatter(Rc::new(Fmt::of_pattern(&p, l)?)))
        }
        ("DateTimeFormatter", "ofLocalizedDate", 1) => t(JTime::Formatter(Rc::new(Fmt::localized(Some(format_style_arg(&a(0))?), None)?))),
        ("DateTimeFormatter", "ofLocalizedTime", 1) => t(JTime::Formatter(Rc::new(Fmt::localized(None, Some(format_style_arg(&a(0))?))?))),
        ("DateTimeFormatter", "ofLocalizedDateTime", 1) => {
            let s = format_style_arg(&a(0))?;
            t(JTime::Formatter(Rc::new(Fmt::localized(Some(s), Some(s))?)))
        }
        ("DateTimeFormatter", "ofLocalizedDateTime", 2) => t(JTime::Formatter(Rc::new(Fmt::localized(Some(format_style_arg(&a(0))?), Some(format_style_arg(&a(1))?))?))),
        // ---- TemporalAdjusters
        ("TemporalAdjusters", m, _) => {
            let adj = match (m, n) {
                ("firstDayOfMonth", 0) => Adj::FirstDayOfMonth,
                ("lastDayOfMonth", 0) => Adj::LastDayOfMonth,
                ("firstDayOfNextMonth", 0) => Adj::FirstDayOfNextMonth,
                ("firstDayOfYear", 0) => Adj::FirstDayOfYear,
                ("lastDayOfYear", 0) => Adj::LastDayOfYear,
                ("firstDayOfNextYear", 0) => Adj::FirstDayOfNextYear,
                ("firstInMonth", 1) => Adj::DayOfWeekInMonth(1, dow_arg(&a(0))?),
                ("lastInMonth", 1) => Adj::DayOfWeekInMonth(-1, dow_arg(&a(0))?),
                ("dayOfWeekInMonth", 2) => Adj::DayOfWeekInMonth(int_arg(args, 0)?, dow_arg(&a(1))?),
                ("next", 1) if a(0).is_null() => return Err(npe("Cannot invoke \"java.time.DayOfWeek.getValue()\" because \"dayOfWeek\" is null")),
                ("next", 1) => Adj::Next(dow_arg(&a(0))?),
                ("nextOrSame", 1) => Adj::NextOrSame(dow_arg(&a(0))?),
                ("previous", 1) => Adj::Previous(dow_arg(&a(0))?),
                ("previousOrSame", 1) => Adj::PreviousOrSame(dow_arg(&a(0))?),
                ("ofDateAdjuster", 1) => {
                    if a(0).is_null() {
                        return Err(npe("dateBasedAdjuster"));
                    }
                    Adj::DateAdjuster(a(0))
                }
                _ => return Ok(None),
            };
            t(JTime::Adjuster(adj))
        }
        // ---- enums
        ("TextStyle", "valueOf", 1) => {
            let s = str_arg(args, 0, "name")?;
            t(JTime::TextStyle(TEXT_STYLES.iter().position(|x| *x == s).ok_or_else(|| illegal_arg(format!("No enum constant java.time.format.TextStyle.{s}")))? as u8))
        }
        ("TextStyle", "values", 0) => enum_array("TextStyle", (0..6).map(|i| t(JTime::TextStyle(i))).collect()),
        ("ResolverStyle", "valueOf", 1) => {
            let s = str_arg(args, 0, "name")?;
            t(JTime::ResolverStyle(RESOLVER_STYLES.iter().position(|x| *x == s).ok_or_else(|| illegal_arg(format!("No enum constant java.time.format.ResolverStyle.{s}")))? as u8))
        }
        ("ResolverStyle", "values", 0) => enum_array("ResolverStyle", (0..3).map(|i| t(JTime::ResolverStyle(i))).collect()),
        ("FormatStyle", "valueOf", 1) => {
            let s = str_arg(args, 0, "name")?;
            t(JTime::FormatStyle(FORMAT_STYLES.iter().position(|x| *x == s).ok_or_else(|| illegal_arg(format!("No enum constant java.time.format.FormatStyle.{s}")))? as u8))
        }
        ("FormatStyle", "values", 0) => enum_array("FormatStyle", (0..4).map(|i| t(JTime::FormatStyle(i))).collect()),
        ("ChronoField", "valueOf", 1) => {
            let s = str_arg(args, 0, "name")?;
            t(JTime::Field(F::from_enum_name(&s).ok_or_else(|| illegal_arg(format!("No enum constant java.time.temporal.ChronoField.{s}")))?))
        }
        ("ChronoField", "values", 0) => enum_array("ChronoField", CHRONO_FIELDS.iter().map(|f| t(JTime::Field(*f))).collect()),
        ("ChronoUnit", "valueOf", 1) => {
            let s = str_arg(args, 0, "name")?;
            Val::ChronoUnit(CHRONO_UNITS.iter().position(|x| *x == s).ok_or_else(|| illegal_arg(format!("No enum constant java.time.temporal.ChronoUnit.{s}")))? as u8)
        }
        ("ChronoUnit", "values", 0) => enum_array("ChronoUnit", (0..16).map(Val::ChronoUnit).collect()),
        ("DayOfWeek", "of", 1) => {
            let v = int_arg(args, 0)?;
            if !(1..=7).contains(&v) {
                return Err(dte(format!("Invalid value for DayOfWeek: {v}")));
            }
            Val::DayOfWeek(v as u8)
        }
        ("DayOfWeek", "valueOf", 1) if a(0).is_null() => return Err(npe("Name is null")),
        ("DayOfWeek", "valueOf", 1) => {
            let s = str_arg(args, 0, "name")?;
            Val::DayOfWeek(DAYS.iter().position(|x| *x == s).ok_or_else(|| illegal_arg(format!("No enum constant java.time.DayOfWeek.{s}")))? as u8 + 1)
        }
        ("DayOfWeek", "values", 0) => enum_array("DayOfWeek", (1..=7).map(Val::DayOfWeek).collect()),
        ("DayOfWeek", "from", 1) => Val::DayOfWeek(conv_dow(&a(0))?),
        ("Month", "of", 1) => {
            let v = int_arg(args, 0)?;
            if !(1..=12).contains(&v) {
                return Err(dte(format!("Invalid value for MonthOfYear: {v}")));
            }
            Val::Month(v as u8)
        }
        ("Month", "valueOf", 1) if a(0).is_null() => return Err(npe("Name is null")),
        ("Month", "valueOf", 1) => {
            let s = str_arg(args, 0, "name")?;
            Val::Month(MONTHS.iter().position(|x| *x == s).ok_or_else(|| illegal_arg(format!("No enum constant java.time.Month.{s}")))? as u8 + 1)
        }
        ("Month", "values", 0) => enum_array("Month", (1..=12).map(Val::Month).collect()),
        ("Month", "from", 1) => Val::Month(conv_month(&a(0))?),
        // ---- Locale
        ("Locale", "forLanguageTag", 1) if a(0).is_null() => {
            return Err(frs(
                npe("Cannot invoke \"String.length()\" because \"s\" is null"),
                &[jf!("sun.util.locale.LocaleUtils", "toLowerString", "LocaleUtils", 89), jf!("sun.util.locale.LanguageTag", "parse", "LanguageTag", 111), jf!("java.util.Locale", "forLanguageTag", "Locale", 1880)],
            ))
        }
        ("Locale", "forLanguageTag", 1) => t(JTime::Locale(Locale::for_tag(&str_arg(args, 0, "languageTag")?))),
        ("Locale", "getDefault", 0 | 1) => t(JTime::Locale(Locale::en())),
        ("Locale", "getAvailableLocales", 0) => enum_array("Locale", ["", "en", "en_US", "en_GB", "fr", "fr_FR", "de", "de_DE", "es", "es_ES", "it", "it_IT", "ja", "ja_JP", "zh", "zh_CN", "ko", "ko_KR"].iter().map(|s| t(JTime::Locale(Locale::for_tag(s)))).collect()),
        // ---- Clock
        ("Clock", "fixed", 2) => {
            if a(0).is_null() {
                return Err(npe("fixedInstant"));
            }
            let i = instant_arg(&a(0))?;
            t(JTime::Clock(i.0, i.1, zone_arg(&a(1), "zone")?))
        }
        ("Clock", "offset", 2) => match a(0) {
            Val::Time(j) if matches!(&*j, JTime::Clock(..) | JTime::OffsetClock(..)) => {
                let d = duration_arg(&a(1))?;
                t(JTime::OffsetClock(j.clone(), d.0, d.1))
            }
            Val::Null => return Err(npe("baseClock")),
            o => return Err(cast_to(&o, "java.time.Clock")),
        },
        _ => return Ok(None),
    };
    Ok(Some(v))
}

fn format_style_arg(v: &Val) -> SResult<u8> {
    match v {
        Val::Time(t) => match &**t {
            JTime::FormatStyle(s) => Ok(*s),
            _ => Err(cast_to(v, "java.time.format.FormatStyle")),
        },
        Val::Null => Err(npe("dateStyle")),
        _ => Err(cast_to(v, "java.time.format.FormatStyle")),
    }
}

// ---------------------------------------------------------------------------
// Instance methods

/// Argument that must be the same temporal class as the receiver.
fn same_kind_arg(tv: &T, v: &Val, target: &str) -> SResult<T> {
    if v.is_null() {
        return Err(npe(null_other_msg(tv, false)));
    }
    match T::of(v) {
        Some(o) if std::mem::discriminant(&o) == std::mem::discriminant(tv) => Ok(o),
        _ => Err(cast_to(v, target)),
    }
}

/// NPE message when `other` is null in `compareTo`/`isBefore`/... .
fn null_other_msg(tv: &T, compare: bool) -> String {
    match tv {
        T::LD(_) => "Cannot invoke \"java.time.chrono.ChronoLocalDate.toEpochDay()\" because \"other\" is null".into(),
        T::LDT(_) => "Cannot invoke \"java.time.chrono.ChronoLocalDateTime.toLocalDate()\" because \"other\" is null".into(),
        T::LT(_) => "Cannot read field \"hour\" because \"other\" is null".into(),
        T::INST(..) => "Cannot read field \"seconds\" because \"otherInstant\" is null".into(),
        T::ZDT(_) => "Cannot invoke \"java.time.chrono.ChronoZonedDateTime.toEpochSecond()\" because \"other\" is null".into(),
        T::ODT(..) => {
            if compare {
                "Cannot invoke \"java.time.OffsetDateTime.getOffset()\" because \"other\" is null".into()
            } else {
                "Cannot invoke \"java.time.OffsetDateTime.toEpochSecond()\" because \"other\" is null".into()
            }
        }
        T::OT(..) => "Cannot invoke \"java.time.OffsetTime.getOffset()\" because \"other\" is null".into(),
        T::YM(..) | T::Y(_) => "Cannot read field \"year\" because \"other\" is null".into(),
        T::MD(..) => "Cannot read field \"month\" because \"other\" is null".into(),
    }
}

/// `isBefore`/`isAfter`/`isEqual` ordering (instant-based for offset types).
fn time_line_cmp(a: &T, b: &T) -> Ordering {
    let inst = |t: &T| -> Option<(i64, i64)> {
        match t {
            T::ODT(d, o) => Some((d.epoch_second(*o), d.nanos() as i64)),
            T::ZDT(z) => Some((z.epoch_second(), z.ldt.nanos() as i64)),
            T::OT(n, o) => Some((0, n - *o as i64 * NPS)),
            _ => None,
        }
    };
    match (inst(a), inst(b)) {
        (Some(x), Some(y)) => x.cmp(&y),
        _ => compare_to(&a.clone().to_val(), &b.clone().to_val()).map(sign).unwrap_or(Ordering::Equal),
    }
}

fn query_val(rt: &mut Rt, recv: &Val, q: &Val) -> SResult<Val> {
    match q {
        Val::Null => Err(npe("query")),
        Val::MethodRef(r) if r.1 == "from" => {
            let target = r.0.as_str();
            if let Some(res) = conv_parsed(recv, target) {
                return res;
            }
            Ok(match target {
                "LocalDate" => t(JTime::LocalDate(conv_ldate(recv)?)),
                "LocalTime" => t(JTime::LocalTime(conv_ltime(recv)?)),
                "LocalDateTime" => t(JTime::LocalDateTime(conv_ldt(recv)?)),
                "Instant" => inst_to_val(conv_instant(recv)?),
                "ZonedDateTime" => conv_zoned(recv)?.to_val(),
                "OffsetDateTime" => {
                    let (d, o) = conv_odt(recv)?;
                    t(JTime::OffsetDateTime(d, o))
                }
                "ZoneId" => t(JTime::Zone(conv_zone(recv)?)),
                "ZoneOffset" => t(JTime::Zone(Zone::Offset(conv_offset(recv)?))),
                "YearMonth" => {
                    let (y, m) = conv_ym(recv)?;
                    t(JTime::YearMonth(y, m))
                }
                "Year" => t(JTime::Year(conv_year(recv)?)),
                "MonthDay" => {
                    let (m, d) = conv_md(recv)?;
                    t(JTime::MonthDay(m, d))
                }
                "DayOfWeek" => Val::DayOfWeek(conv_dow(recv)?),
                "Month" => Val::Month(conv_month(recv)?),
                _ => rt.call_fn(q, &[recv.clone()])?,
            })
        }
        Val::Time(j) => match &**j {
            JTime::Query(name) => {
                let acc = Acc::of_any(recv).ok_or_else(|| cast_to(recv, "java.time.temporal.TemporalAccessor"))?;
                Ok(match *name {
                    "ZoneId" => acc.zone.map(|z| t(JTime::Zone(z))).unwrap_or(Val::Null),
                    "Zone" => acc.zone.or_else(|| acc.off.map(Zone::Offset)).map(|z| t(JTime::Zone(z))).unwrap_or(Val::Null),
                    "Offset" => acc.off.map(|o| t(JTime::Zone(Zone::Offset(o)))).unwrap_or(Val::Null),
                    "LocalDate" => acc.date.map(|d| t(JTime::LocalDate(d))).unwrap_or(Val::Null),
                    "LocalTime" => acc.time.map(|n| t(JTime::LocalTime(n))).unwrap_or(Val::Null),
                    "Chronology" => {
                        if acc.date.is_some() || acc.partial {
                            t(JTime::Chronology)
                        } else {
                            Val::Null
                        }
                    }
                    _ => match T::of(recv) {
                        Some(T::LD(_)) | Some(T::MD(..)) => Val::ChronoUnit(U_DAYS),
                        Some(T::YM(..)) => Val::ChronoUnit(U_MONTHS),
                        Some(T::Y(_)) => Val::ChronoUnit(U_YEARS),
                        Some(_) => Val::ChronoUnit(U_NANOS),
                        None => match recv {
                            Val::Month(_) => Val::ChronoUnit(U_MONTHS),
                            Val::DayOfWeek(_) => Val::ChronoUnit(U_DAYS),
                            _ => Val::Null,
                        },
                    },
                })
            }
            _ => Err(cast_to(q, "java.time.temporal.TemporalQuery")),
        },
        Val::Lambda(_) | Val::MethodRef(_) => rt.call_fn(q, &[recv.clone()]),
        _ => Err(cast_to(q, "java.time.temporal.TemporalQuery")),
    }
}

/// `isSupported(field)`, `get`, `getLong`, `range`, `query` for any accessor value.
fn accessor_method(rt: &mut Rt, recv: &Val, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("isSupported", 1) => {
            if let Val::ChronoUnit(_) = args[0] {
                return Err(cast_to(&args[0], "java.time.temporal.TemporalField"));
            }
            let f = field_arg(&args[0])?;
            match T::of(recv) {
                Some(tv) => Val::Bool(tv.get_long(f).is_ok()),
                None => Val::Bool(Acc::of_any(recv).is_some_and(|a| a.supports(f, &Locale::en()))),
            }
        }
        ("get", 1) => {
            let f = field_arg(&args[0])?;
            match T::of(recv) {
                Some(tv) => Val::Int(tv.get_int(f)? as i32),
                None => {
                    let a = Acc::of_any(recv).ok_or_else(|| unsupported_field(f))?;
                    let v = a.get(f, &Locale::en()).ok_or_else(|| {
                        frs(
                            unsupported_field(f),
                            &match a.kind {
                                "DayOfWeek" => vec![TA_RANGE, DOW_RANGE_285, TA_GET, DOW_GET_318],
                                "Month" => vec![TA_RANGE, MON_RANGE_312, TA_GET, MON_GET_345],
                                _ => vec![],
                            },
                        )
                    })?;
                    if !f.is_int_range() {
                        return Err(utte(format!("Invalid field '{}' for get() method, use getLong() instead", f.name())));
                    }
                    Val::Int(v as i32)
                }
            }
        }
        ("getLong", 1) => {
            let f = field_arg(&args[0])?;
            match T::of(recv) {
                Some(tv) => Val::Long(tv.get_long(f)?),
                None => {
                    let a = Acc::of_any(recv).ok_or_else(|| unsupported_field(f))?;
                    Val::Long(a.get(f, &Locale::en()).ok_or_else(|| frs(unsupported_field(f), &getlong_chain(a.kind, f.is_time_based())))?)
                }
            }
        }
        ("range", 1) => {
            let f = field_arg(&args[0])?;
            match T::of(recv) {
                Some(tv) => {
                    let r = tv.range(f)?;
                    t(JTime::Range(r.0, r.1, r.2, r.3))
                }
                None => {
                    let a = Acc::of_any(recv).ok_or_else(|| unsupported_field(f))?;
                    if !a.supports(f, &Locale::en()) {
                        return Err(frs(
                            unsupported_field(f),
                            &match a.kind {
                                "DayOfWeek" => vec![TA_RANGE, DOW_RANGE_285],
                                "Month" => vec![TA_RANGE, MON_RANGE_312],
                                _ => vec![],
                            },
                        ));
                    }
                    let (lo, ms, ml) = f.range();
                    t(JTime::Range(lo, lo, ms, ml))
                }
            }
        }
        ("query", 1) => query_val(rt, recv, &args[0])?,
        ("getMillis" | "toEpochMilli", 0) => {
            let tv = T::of(recv).ok_or_else(|| unsupported_field(F::InstantSeconds))?;
            let secs = tv.get_long(F::InstantSeconds)?;
            let ms = tv.get_long(F::MilliOfSecond)?;
            Val::Long(secs.checked_mul(1000).and_then(|v| v.checked_add(ms)).ok_or_else(overflow)?)
        }
        _ => return Ok(None),
    }))
}

fn temporal_method(rt: &mut Rt, recv: &Val, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let Some(tv) = T::of(recv) else { return Ok(None) };
    temporal_method0(rt, recv, &tv, name, args).map_err(|e| method_frames(&tv, name, args, e))
}

/// Outer JDK frames of an instance method failure on a temporal.
fn method_frames(tv: &T, name: &str, args: &[Val], e: Box<ScriptErr>) -> Box<ScriptErr> {
    let n = args.len();
    let m = e.msg.as_str();
    let k = tv.kind();
    let npe_arg = e.kind == "null_pointer_exception" && e.frames.is_empty() && args.iter().any(Val::is_null);
    let outer: Vec<Fr> = match (k, name, n) {
        // format(formatter)
        (_, "format", 1) if npe_arg => vec![
            REQ_NON_NULL,
            match k {
                "LocalDate" => jt!("LocalDate", "format", 1816),
                "LocalDateTime" => jt!("LocalDateTime", "format", 1747),
                "LocalTime" => jt!("LocalTime", "format", 1445),
                "ZonedDateTime" => jt!("ZonedDateTime", "format", 2157),
                "OffsetDateTime" => jt!("OffsetDateTime", "format", 1761),
                "YearMonth" => jt!("YearMonth", "format", 1091),
                "Year" => jt!("Year", "format", 943),
                "MonthDay" => jt!("MonthDay", "format", 665),
                _ => jt!("OffsetTime", "format", 1206),
            },
        ],
        (_, "format", 1) if !e.frames.is_empty() => vec![match k {
            "LocalDate" => jt!("LocalDate", "format", 1817),
            "LocalDateTime" => jt!("LocalDateTime", "format", 1748),
            "LocalTime" => jt!("LocalTime", "format", 1446),
            "ZonedDateTime" => jt!("ZonedDateTime", "format", 2158),
            "OffsetDateTime" => jt!("OffsetDateTime", "format", 1762),
            "YearMonth" => jt!("YearMonth", "format", 1092),
            "Year" => jt!("Year", "format", 944),
            "MonthDay" => jt!("MonthDay", "format", 666),
            _ => jt!("OffsetTime", "format", 1207),
        }],
        // plus/minus (amount)
        ("LocalDate", "plus", 1) if npe_arg => vec![REQ_NON_NULL, jt!("LocalDate", "plus", 1176)],
        ("ZonedDateTime", "plus", 1) if npe_arg => vec![REQ_NON_NULL, jt!("ZonedDateTime", "plus", 1557)],
        ("LocalDate", "plus", 1) if m.starts_with("Unsupported unit") => vec![jt!("LocalDate", "plus", 143), DUR_ADD_TO, jt!("LocalDate", "plus", 1177)],
        ("LocalDate", "minus", 1) if m.starts_with("Unsupported unit") => vec![jt!("LocalDate", "minus", 1449), jt!("LocalDate", "minus", 143), DUR_SUB_FROM, jt!("LocalDate", "minus", 1425)],
        ("LocalTime", "plus", 1) if m.starts_with("Unsupported unit") => vec![LT_PLUS_127, PER_ADD_TO_DAYS, jt!("LocalTime", "plus", 1021)],
        ("Instant", "plus", 1) if m.starts_with("Unsupported unit") => vec![jt!("Instant", "plus", 209), PER_ADD_TO_MONTHS, jt!("Instant", "plus", 788)],
        ("Instant", "plus", 1) if m.starts_with("Instant exceeds") => vec![INST_PLUS_882, jt!("Instant", "plus", 209), DUR_ADD_TO, jt!("Instant", "plus", 788)],
        // plus/minus (amount, unit)
        ("LocalDate", "plus", 2) if npe_arg => vec![jt!("LocalDate", "plus", 1276)],
        ("LocalDate", "minus", 2) if m.starts_with("Unsupported unit") => vec![jt!("LocalDate", "minus", 1449)],
        ("LocalDate", "plus", 2) if m == "long overflow" => vec![jt!("LocalDate", "plus", 1265)],
        ("LocalDate", "plus", 2) if m.starts_with("Invalid value for Year") => vec![if matches!(args.get(1), Some(Val::ChronoUnit(U_YEARS))) { jt!("LocalDate", "plus", 1268) } else { jt!("LocalDate", "plus", 1269) }],
        ("LocalDateTime", "plus", 2) if m == "long overflow" => vec![jt!("LocalDate", "plus", 1265), LDT_PLUS_1190],
        // plusX / minusX
        ("LocalDate", "minusDays", 1) if m == "long overflow" => vec![jt!("LocalDate", "minusDays", 1536)],
        ("LocalDateTime", "plusDays", 1) => vec![jt!("LocalDateTime", "plusDays", 1283)],
        ("ZonedDateTime", "plusDays", 1) => vec![jt!("LocalDateTime", "plusDays", 1283), jt!("ZonedDateTime", "plusDays", 1696)],
        ("LocalDateTime", "plusHours", 1) if m.starts_with("Invalid value for EpochDay") => vec![jt!("LocalDateTime", "plusWithOverflow", 1567), jt!("LocalDateTime", "plusHours", 1298)],
        // withX
        ("LocalDate" | "LocalDateTime" | "ZonedDateTime", "withHour", 1) => {
            let mut v = vec![jt!("LocalTime", "withHour", 906)];
            if k != "LocalTime" {
                v.push(jt!("LocalDateTime", "withHour", 1052));
            }
            if k == "ZonedDateTime" {
                v.push(jt!("ZonedDateTime", "withHour", 1428));
            }
            v
        }
        ("LocalTime", "withHour", 1) => vec![jt!("LocalTime", "withHour", 906)],
        ("LocalDateTime", "withDayOfMonth", 1) => vec![jt!("LocalDateTime", "withDayOfMonth", 1022)],
        // until
        ("LocalDate", "until", 1) if npe_arg => vec![REQ_NON_NULL, LD_FROM_419, jt!("LocalDate", "until", 1706)],
        // atX
        ("LocalDate", "atStartOfDay", 1) if npe_arg => vec![REQ_NON_NULL, jt!("LocalDate", "atStartOfDay", 1938)],
        ("LocalDate", "atTime", 1) if npe_arg => vec![REQ_NON_NULL, jt!("LocalDateTime", "of", 379), jt!("LocalDate", "atTime", 1832)],
        ("LocalDate", "atTime", 2) => vec![jt!("LocalDate", "atTime", 1850)],
        ("LocalDate", "atTime", 3) => vec![jt!("LocalDate", "atTime", 1869)],
        ("LocalDate", "atTime", 4) => vec![jt!("LocalDate", "atTime", 1888)],
        ("LocalDateTime", "atZone", 1) if npe_arg => vec![REQ_NON_NULL, ZDT_OF_LOCAL_372, ZDT_OF_297, jt!("LocalDateTime", "atZone", 1796)],
        ("LocalDateTime", "atOffset", 1) if npe_arg => vec![REQ_NON_NULL, jt!("OffsetDateTime", "<init>", 419), jt!("OffsetDateTime", "of", 281), jt!("LocalDateTime", "atOffset", 1762)],
        ("LocalDateTime", "toEpochSecond", 1) if npe_arg => vec![REQ_NON_NULL, jf!("java.time.chrono.ChronoLocalDateTime", "toEpochSecond", "ChronoLocalDateTime", 475)],
        ("LocalDateTime", "toInstant", 1) if npe_arg => vec![REQ_NON_NULL, jf!("java.time.chrono.ChronoLocalDateTime", "toEpochSecond", "ChronoLocalDateTime", 475), jf!("java.time.chrono.ChronoLocalDateTime", "toInstant", "ChronoLocalDateTime", 457)],
        ("Instant", "atZone", 1) if npe_arg => vec![REQ_NON_NULL, ZDT_OF_INSTANT_412, jt!("Instant", "atZone", 1288)],
        ("Instant", "atOffset", 1) if npe_arg => vec![REQ_NON_NULL, jt!("OffsetDateTime", "ofInstant", 330), jt!("Instant", "atOffset", 1270)],
        ("ZonedDateTime", "withZoneSameInstant", 1) if npe_arg => vec![REQ_NON_NULL, jt!("ZonedDateTime", "withZoneSameInstant", 994)],
        ("ZonedDateTime", "withZoneSameLocal", 1) if npe_arg => vec![REQ_NON_NULL, jt!("ZonedDateTime", "withZoneSameLocal", 971)],
        ("Year", "atMonthDay", 1) if npe_arg => vec![jt!("Year", "atMonthDay", 1013)],
        ("YearMonth", "atDay", 1) => vec![jt!("YearMonth", "atDay", 1114)],
        ("Year", "atDay", 1) => vec![jt!("Year", "atDay", 961)],
        ("Year", "atMonth", 1) => vec![jt!("YearMonth", "of", 247), jt!("Year", "atMonth", 998)],
        ("MonthDay", "atYear", 1) => vec![jt!("MonthDay", "atYear", 685)],
        ("MonthDay", "withDayOfMonth", 1) => vec![jt!("MonthDay", "withDayOfMonth", 588)],
        ("MonthDay", "withMonth", 1) => vec![jt!("MonthDay", "withMonth", 547)],
        ("MonthDay", "with", 1) if npe_arg => vec![REQ_NON_NULL, jt!("MonthDay", "with", 563)],
        // compareTo / isBefore / isAfter / isEqual with null
        (_, "compareTo" | "isBefore" | "isAfter" | "isEqual", 1) if npe_arg => match k {
            "LocalDate" => vec![
                match name {
                    "compareTo" => jf!("java.time.chrono.ChronoLocalDate", "compareTo", "ChronoLocalDate", 711),
                    "isBefore" => jf!("java.time.chrono.ChronoLocalDate", "isBefore", "ChronoLocalDate", 751),
                    "isAfter" => jf!("java.time.chrono.ChronoLocalDate", "isAfter", "ChronoLocalDate", 733),
                    _ => jf!("java.time.chrono.ChronoLocalDate", "isEqual", "ChronoLocalDate", 769),
                },
                match name {
                    "compareTo" => jt!("LocalDate", "compareTo", 2022),
                    "isBefore" => jt!("LocalDate", "isBefore", 2091),
                    "isAfter" => jt!("LocalDate", "isAfter", 2062),
                    _ => jt!("LocalDate", "isEqual", 2120),
                },
            ],
            "LocalDateTime" => vec![
                match name {
                    "compareTo" => jf!("java.time.chrono.ChronoLocalDateTime", "compareTo", "ChronoLocalDateTime", 515),
                    "isBefore" => jf!("java.time.chrono.ChronoLocalDateTime", "isBefore", "ChronoLocalDateTime", 562),
                    "isAfter" => jf!("java.time.chrono.ChronoLocalDateTime", "isAfter", "ChronoLocalDateTime", 543),
                    _ => jf!("java.time.chrono.ChronoLocalDateTime", "isEqual", "ChronoLocalDateTime", 581),
                },
                match name {
                    "compareTo" => jt!("LocalDateTime", "compareTo", 1823),
                    "isBefore" => jt!("LocalDateTime", "isBefore", 1889),
                    "isAfter" => jt!("LocalDateTime", "isAfter", 1866),
                    _ => jt!("LocalDateTime", "isEqual", 1912),
                },
            ],
            "LocalTime" => {
                let mut v = vec![jt!("LocalTime", "compareTo", 1543)];
                if name == "isBefore" {
                    v.push(jt!("LocalTime", "isBefore", 1577));
                } else if name == "isAfter" {
                    v.push(jt!("LocalTime", "isAfter", 1562));
                }
                v
            }
            "Instant" => {
                let mut v = vec![jt!("Instant", "compareTo", 1333)];
                if name == "isBefore" {
                    v.push(jt!("Instant", "isBefore", 1363));
                } else if name == "isAfter" {
                    v.push(jt!("Instant", "isAfter", 1348));
                }
                v
            }
            "ZonedDateTime" => vec![match name {
                "compareTo" => jf!("java.time.chrono.ChronoZonedDateTime", "compareTo", "ChronoZonedDateTime", 583),
                "isBefore" => jf!("java.time.chrono.ChronoZonedDateTime", "isBefore", "ChronoZonedDateTime", 614),
                "isAfter" => jf!("java.time.chrono.ChronoZonedDateTime", "isAfter", "ChronoZonedDateTime", 598),
                _ => jf!("java.time.chrono.ChronoZonedDateTime", "isEqual", "ChronoZonedDateTime", 630),
            }],
            "OffsetDateTime" => vec![match name {
                "compareTo" => jt!("OffsetDateTime", "compareTo", 1814),
                "isBefore" => jt!("OffsetDateTime", "isBefore", 1857),
                "isAfter" => jt!("OffsetDateTime", "isAfter", 1838),
                _ => jt!("OffsetDateTime", "isEqual", 1876),
            }],
            "YearMonth" => {
                let mut v = vec![jt!("YearMonth", "compareTo", 1150)];
                if name == "isAfter" {
                    v.push(jt!("YearMonth", "isAfter", 1164));
                } else if name == "isBefore" {
                    v.push(jt!("YearMonth", "isBefore", 1175));
                }
                v
            }
            "Year" => vec![jt!("Year", "compareTo", 1031)],
            "MonthDay" => vec![jt!("MonthDay", "compareTo", 703)],
            _ => vec![],
        },
        // get/getLong/range/with with null field
        (_, "get", 1) if npe_arg && k == "LocalDate" => vec![jt!("LocalDate", "range", 628), TA_GET, jt!("LocalDate", "get", 663)],
        (_, "getLong", 1) if npe_arg && k == "LocalDate" => vec![jt!("LocalDate", "getLong", 700)],
        (_, "range", 1) if npe_arg && k == "LocalDate" => vec![jt!("LocalDate", "range", 628)],
        (_, "with", 2) if npe_arg && k == "LocalDate" => vec![LD_WITH_1070],
        _ => Vec::new(),
    };
    fr(e, &outer)
}

fn temporal_method0(rt: &mut Rt, recv: &Val, tv: &T, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let tv = tv.clone();
    let n = args.len();
    if let Some(v) = accessor_method(rt, recv, name, args)? {
        return ok(v);
    }
    let a = |i: usize| at(args, i);
    let cmp_target = match &tv {
        T::LD(_) => "java.time.chrono.ChronoLocalDate",
        T::LT(_) => "java.time.LocalTime",
        T::LDT(_) => "java.time.chrono.ChronoLocalDateTime",
        T::ODT(..) => "java.time.OffsetDateTime",
        T::OT(..) => "java.time.OffsetTime",
        T::ZDT(_) => "java.time.chrono.ChronoZonedDateTime",
        T::INST(..) => "java.time.Instant",
        T::YM(..) => "java.time.YearMonth",
        T::Y(_) => "java.time.Year",
        T::MD(..) => "java.time.MonthDay",
    };
    // Methods shared by all temporals.
    let shared = match (name, n) {
        ("compareTo", 1) => Some(Val::Int(compare_to(recv, &a(0))? as i32)),
        ("isBefore" | "isAfter" | "isEqual", 1) if !matches!(tv, T::LDT(_) | T::Y(_)) || name != "isEqual" => {
            if matches!(tv, T::Y(_)) && name == "isBefore" {
                None
            } else {
                let o = same_kind_arg(&tv, &a(0), cmp_target)?;
                let c = time_line_cmp(&tv, &o);
                Some(Val::Bool(match name {
                    "isBefore" => c == Ordering::Less,
                    "isAfter" => c == Ordering::Greater,
                    _ => c == Ordering::Equal,
                }))
            }
        }
        ("plus", 2) if !matches!(tv, T::MD(..)) => Some(tv.plus_unit(long_arg(args, 0)?, unit_arg(&a(1))?)?.to_val()),
        ("minus", 2) if !matches!(tv, T::MD(..)) => {
            let amt = long_arg(args, 0)?;
            let u = unit_arg(&a(1))?;
            Some(if amt == i64::MIN { tv.plus_unit(i64::MAX, u)?.plus_unit(1, u)?.to_val() } else { tv.plus_unit(-amt, u)?.to_val() })
        }
        ("plus", 1) if !matches!(tv, T::MD(..)) => Some(tv.plus_amount(&a(0), false)?.to_val()),
        ("minus", 1) if !matches!(tv, T::MD(..)) => Some(tv.plus_amount(&a(0), true)?.to_val()),
        ("with", 2) if !matches!(tv, T::MD(..)) => Some(tv.with_field(field_arg(&a(0))?, long_arg(args, 1)?)?.to_val()),
        ("with", 1) if !matches!(tv, T::MD(..)) => Some(tv.with_adjuster(rt, &a(0))?.to_val()),
        ("until", 2) if !matches!(tv, T::MD(..)) => Some(Val::Long(tv.until(&a(0), unit_arg(&a(1))?)?)),
        ("format", 1) => Some(Val::string(fmt_arg(&a(0), "formatter")?.format(recv)?)),
        ("truncatedTo", 1) if matches!(tv, T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_) | T::INST(..)) => Some(tv.truncated(unit_arg(&a(0))?)?.to_val()),
        ("adjustInto", 1) => {
            let target = T::of(&a(0)).ok_or_else(|| if a(0).is_null() { npe("temporal") } else { cast_to(&a(0), "java.time.temporal.Temporal") })?;
            Some(target.with_adjuster(rt, recv)?.to_val())
        }
        _ => None,
    };
    if let Some(v) = shared {
        return ok(v);
    }
    let int = |v: i64| Val::Int(v as i32);
    let getter = |f: F| -> SResult<Val> { Ok(int(tv.get_long(f)?)) };
    let plus_named = |unit: Unit| -> SResult<Val> {
        let amt = long_arg(args, 0)?;
        let amt = if name.starts_with("minus") { amt.checked_neg().unwrap_or(i64::MAX) } else { amt };
        Ok(tv.plus_unit(amt, unit)?.to_val())
    };
    let v = match (&tv, name, n) {
        // ---- field getters shared by date/time types
        (_, "getYear", 0) => getter(F::Year)?,
        (_, "getMonthValue" | "getMonthOfYear", 0) => getter(F::MonthOfYear)?,
        (_, "getMonth", 0) => Val::Month(tv.get_long(F::MonthOfYear)? as u8),
        (_, "getDayOfMonth", 0) => getter(F::DayOfMonth)?,
        (_, "getDayOfYear", 0) => getter(F::DayOfYear)?,
        (_, "getDayOfWeek" | "getDayOfWeekEnum", 0) => Val::DayOfWeek(tv.get_long(F::DayOfWeek)? as u8),
        (_, "getHour" | "getHourOfDay", 0) => getter(F::HourOfDay)?,
        (_, "getMinute" | "getMinuteOfHour", 0) => getter(F::MinuteOfHour)?,
        (_, "getSecond" | "getSecondOfMinute", 0) => getter(F::SecondOfMinute)?,
        (_, "getNano", 0) => getter(F::NanoOfSecond)?,
        (T::ZDT(_), "getMillisOfSecond", 0) => getter(F::MilliOfSecond)?,
        (T::ZDT(_), "getYearOfCentury", 0) => return Err(unsupported("[getYearOfCentury] is no longer available; use [get(ChronoField.YEAR_OF_ERA) % 100] instead")),
        (T::ZDT(_), "getWeekOfWeekyear", 0) => return Err(unsupported("[getWeekOfWeekyear] is no longer available; use [get(IsoFields.WEEK_OF_WEEK_BASED_YEAR)] instead")),
        (T::ZDT(_), "getMillisOfDay", 0) => return Err(unsupported("[getMillisOfDay] is no longer available; use [get(ChronoField.MILLI_OF_DAY)] instead")),
        (T::ZDT(_), "getEra", 0) => return Err(unsupported("[getEra] is no longer available; use [get(ChronoField.ERA)] instead")),
        (T::ZDT(_), "getCenturyOfEra", 0) => return Err(unsupported("[getCenturyOfEra] is no longer available; use [get(ChronoField.YEAR_OF_ERA) / 100] instead")),
        (T::ZDT(_), "getYearOfEra", 0) => return Err(unsupported("[getYearOfEra] is no longer available; use [get(ChronoField.YEAR_OF_ERA)] instead")),
        (T::ZDT(_), "getWeekyear", 0) => return Err(unsupported("[getWeekyear] is no longer available; use [get(IsoFields.WEEK_BASED_YEAR)] instead")),
        (T::ZDT(_), "getMinuteOfDay", 0) => return Err(unsupported("[getMinuteOfDay] is no longer available; use [get(ChronoField.MINUTE_OF_DAY)] instead")),
        (T::ZDT(_), "getSecondOfDay", 0) => return Err(unsupported("[getSecondOfDay] is no longer available; use [get(ChronoField.SECOND_OF_DAY)] instead")),
        // ---- plus/minus by unit
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_), "plusDays" | "minusDays", 1) => plus_named(U_DAYS)?,
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_), "plusWeeks" | "minusWeeks", 1) => plus_named(U_WEEKS)?,
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_) | T::YM(..), "plusMonths" | "minusMonths", 1) => plus_named(U_MONTHS)?,
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_) | T::YM(..) | T::Y(_), "plusYears" | "minusYears", 1) => plus_named(U_YEARS)?,
        (T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_), "plusHours" | "minusHours", 1) => plus_named(U_HOURS)?,
        (T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_), "plusMinutes" | "minusMinutes", 1) => plus_named(U_MINUTES)?,
        (T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_) | T::INST(..), "plusSeconds" | "minusSeconds", 1) => plus_named(U_SECONDS)?,
        (T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_) | T::INST(..), "plusNanos" | "minusNanos", 1) => plus_named(U_NANOS)?,
        (T::INST(..), "plusMillis" | "minusMillis", 1) => plus_named(U_MILLIS)?,
        // ---- with*
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_) | T::YM(..), "withYear", 1) => tv.with_field(F::Year, int_arg(args, 0)?)?.to_val(),
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_) | T::YM(..), "withMonth", 1) => tv.with_field(F::MonthOfYear, int_arg(args, 0)?)?.to_val(),
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_), "withDayOfMonth", 1) => tv.with_field(F::DayOfMonth, int_arg(args, 0)?)?.to_val(),
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_), "withDayOfYear", 1) => tv.with_field(F::DayOfYear, int_arg(args, 0)?)?.to_val(),
        (T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_), "withHour", 1) => tv.with_field(F::HourOfDay, int_arg(args, 0)?)?.to_val(),
        (T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_), "withMinute", 1) => tv.with_field(F::MinuteOfHour, int_arg(args, 0)?)?.to_val(),
        (T::LT(_) | T::LDT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_), "withSecond", 1) => tv.with_field(F::SecondOfMinute, int_arg(args, 0)?)?.to_val(),
        (T::LT(_) | T::ODT(..) | T::OT(..) | T::ZDT(_), "withNano", 1) => tv.with_field(F::NanoOfSecond, int_arg(args, 0)?)?.to_val(),
        // ---- LocalDate
        (T::LD(d), "isLeapYear", 0) => Val::Bool(d.leap()),
        (T::LD(d), "lengthOfMonth", 0) => Val::Int(d.length_of_month() as i32),
        (T::LD(d), "lengthOfYear", 0) => Val::Int(d.length_of_year() as i32),
        (T::LD(d), "toEpochDay", 0) => Val::Long(d.epoch_day()),
        (T::LD(_) | T::LDT(_) | T::ODT(..) | T::ZDT(_) | T::YM(..), "getChronology", 0) => t(JTime::Chronology),
        (T::LD(d), "getEra", 0) => t(JTime::Era(era_of(d.y) as u8)),
        (T::LD(d), "atStartOfDay", 0) => t(JTime::LocalDateTime(LDT::of(*d, 0))),
        (T::LD(d), "atStartOfDay", 1) => Zoned::of_local(LDT::of(*d, 0), zone_arg(&a(0), "zone")?, None)?.to_val(),
        (T::LD(d), "atTime", 1) => t(JTime::LocalDateTime(LDT::of(*d, ltime_arg(&a(0))?))),
        (T::LD(d), "atTime", 2..=4) => {
            let h = int_arg(args, 0)?;
            let m = int_arg(args, 1)?;
            let s = if n > 2 { int_arg(args, 2)? } else { 0 };
            let ns = if n > 3 { int_arg(args, 3)? } else { 0 };
            t(JTime::LocalDateTime(LDT::of(*d, time_of(h, m, s, ns)?)))
        }
        (T::LD(d), "until", 1) => {
            let e = conv_ldate(&a(0))?;
            let p = d.period_until(&e)?;
            t(JTime::Period(p.0, p.1, p.2))
        }
        // ---- LocalTime / OffsetTime
        (T::LT(nod) | T::OT(nod, _), "toSecondOfDay", 0) => Val::Int((nod / NPS) as i32),
        (T::LT(nod) | T::OT(nod, _), "toNanoOfDay", 0) => Val::Long(*nod),
        (T::LT(nod), "atDate", 1) => t(JTime::LocalDateTime(LDT::of(ldate_arg(&a(0))?, *nod))),
        (T::LT(nod), "atOffset", 1) => t(JTime::OffsetTime(*nod, offset_arg(&a(0), "offset")?)),
        (T::OT(nod, o), "atDate", 1) => t(JTime::OffsetDateTime(LDT::of(ldate_arg(&a(0))?, *nod), *o)),
        (T::OT(nod, _), "toLocalTime", 0) => t(JTime::LocalTime(*nod)),
        (T::OT(_, o) | T::ODT(_, o), "getOffset", 0) => t(JTime::Zone(Zone::Offset(*o))),
        (T::OT(nod, o), "withOffsetSameInstant", 1) => {
            let no = offset_arg(&a(0), "offset")?;
            t(JTime::OffsetTime(time_plus(*nod, (no as i64 - *o as i64) as i128 * NPS as i128), no))
        }
        (T::OT(nod, _), "withOffsetSameLocal", 1) => t(JTime::OffsetTime(*nod, offset_arg(&a(0), "offset")?)),
        // ---- LocalDateTime
        (T::LDT(d) | T::ODT(d, _), "toLocalDate", 0) => t(JTime::LocalDate(d.date)),
        (T::LDT(d) | T::ODT(d, _), "toLocalTime", 0) => t(JTime::LocalTime(d.nod)),
        (T::ZDT(z), "toLocalDate", 0) => t(JTime::LocalDate(z.ldt.date)),
        (T::ZDT(z), "toLocalTime", 0) => t(JTime::LocalTime(z.ldt.nod)),
        (T::ZDT(z), "toLocalDateTime", 0) => t(JTime::LocalDateTime(z.ldt)),
        (T::ODT(d, _), "toLocalDateTime", 0) => t(JTime::LocalDateTime(*d)),
        (T::LDT(d), "atOffset", 1) => t(JTime::OffsetDateTime(*d, offset_arg(&a(0), "offset")?)),
        (T::LDT(d), "atZone", 1) => Zoned::of_local(*d, zone_arg(&a(0), "zone")?, None)?.to_val(),
        (T::LDT(d), "toEpochSecond", 1) => Val::Long(d.epoch_second(offset_arg(&a(0), "offset")?)),
        (T::LDT(d), "toInstant", 1) => inst_to_val((d.epoch_second(offset_arg(&a(0), "offset")?), d.nanos())),
        // ---- OffsetDateTime
        (T::ODT(d, o), "toInstant", 0) => inst_to_val((d.epoch_second(*o), d.nanos())),
        (T::ODT(d, o), "toEpochSecond", 0) => Val::Long(d.epoch_second(*o)),
        (T::ODT(d, o), "toOffsetTime", 0) => t(JTime::OffsetTime(d.nod, *o)),
        (T::ODT(d, o), "toZonedDateTime", 0) => Zoned { ldt: *d, off: *o, zone: Zone::Offset(*o) }.to_val(),
        (T::ODT(d, o), "atZoneSameInstant", 1) => Zoned::of_instant((d.epoch_second(*o), d.nanos()), zone_arg(&a(0), "zone")?)?.to_val(),
        (T::ODT(d, o), "atZoneSimilarLocal", 1) => Zoned::of_local(*d, zone_arg(&a(0), "zone")?, Some(*o))?.to_val(),
        (T::ODT(d, o), "withOffsetSameInstant", 1) => {
            let no = offset_arg(&a(0), "offset")?;
            t(JTime::OffsetDateTime(LDT::from_epoch(d.epoch_second(*o), d.nanos(), no)?, no))
        }
        (T::ODT(d, _), "withOffsetSameLocal", 1) => t(JTime::OffsetDateTime(*d, offset_arg(&a(0), "offset")?)),
        // ---- ZonedDateTime
        (T::ZDT(z), "getZone", 0) => t(JTime::Zone(z.zone.clone())),
        (T::ZDT(z), "getOffset", 0) => t(JTime::Zone(Zone::Offset(z.off))),
        (T::ZDT(z), "toInstant", 0) => inst_to_val(z.instant()),
        (T::ZDT(z), "toEpochSecond", 0) => Val::Long(z.epoch_second()),
        (T::ZDT(z), "toOffsetDateTime", 0) => t(JTime::OffsetDateTime(z.ldt, z.off)),
        (T::ZDT(z), "withZoneSameInstant", 1) => Zoned::of_instant(z.instant(), zone_arg(&a(0), "zone")?)?.to_val(),
        (T::ZDT(z), "withZoneSameLocal", 1) => Zoned::of_local(z.ldt, zone_arg(&a(0), "zone")?, Some(z.off))?.to_val(),
        (T::ZDT(z), "withFixedOffsetZone", 0) => Zoned { ldt: z.ldt, off: z.off, zone: Zone::Offset(z.off) }.to_val(),
        (T::ZDT(z), "withEarlierOffsetAtOverlap" | "withLaterOffsetAtOverlap", 0) => {
            let (offs, _) = z.zone.valid_offsets(z.ldt.epoch_second(0));
            if offs.len() == 2 {
                let off = if name == "withEarlierOffsetAtOverlap" { offs[0] } else { offs[1] };
                Zoned { ldt: z.ldt, off, zone: z.zone.clone() }.to_val()
            } else {
                recv.clone()
            }
        }
        // ---- Instant
        (T::INST(s, _), "getEpochSecond", 0) => Val::Long(*s),
        (T::INST(s, nn), "atZone", 1) => Zoned::of_instant((*s, *nn), zone_arg(&a(0), "zone")?)?.to_val(),
        (T::INST(s, nn), "atOffset", 1) => {
            let o = offset_arg(&a(0), "offset")?;
            t(JTime::OffsetDateTime(LDT::from_epoch(*s, *nn, o)?, o))
        }
        // ---- YearMonth
        (T::YM(y, m), "atDay", 1) => t(JTime::LocalDate(LDate::of(*y as i64, *m as i64, int_arg(args, 0)?)?)),
        (T::YM(y, m), "atEndOfMonth", 0) => t(JTime::LocalDate(LDate { y: *y, m: *m, d: month_len(*m, is_leap(*y as i64)) })),
        (T::YM(y, _), "isLeapYear", 0) => Val::Bool(is_leap(*y as i64)),
        (T::YM(y, m), "isValidDay", 1) => {
            let d = int_arg(args, 0)?;
            Val::Bool(d >= 1 && d <= month_len(*m, is_leap(*y as i64)) as i64)
        }
        (T::YM(y, m), "lengthOfMonth", 0) => Val::Int(month_len(*m, is_leap(*y as i64)) as i32),
        (T::YM(y, _), "lengthOfYear", 0) => Val::Int(if is_leap(*y as i64) { 366 } else { 365 }),
        // ---- Year
        (T::Y(y), "getValue", 0) => Val::Int(*y),
        (T::Y(y), "isLeap", 0) => Val::Bool(is_leap(*y as i64)),
        (T::Y(y), "length", 0) => Val::Int(if is_leap(*y as i64) { 366 } else { 365 }),
        (T::Y(y), "atDay", 1) => t(JTime::LocalDate(LDate::of_year_day(*y as i64, int_arg(args, 0)?)?)),
        (T::Y(y), "atMonth", 1) => t(JTime::YearMonth(*y, F::MonthOfYear.check(int_arg(args, 0)?)? as u8)),
        (T::Y(y), "atMonthDay", 1) => {
            let (m, d) = match a(0) {
                Val::Time(j) => match &*j {
                    JTime::MonthDay(m, d) => (*m, *d),
                    _ => return Err(cast_to(&a(0), "java.time.MonthDay")),
                },
                Val::Null => return Err(npe("monthDay")),
                o => return Err(cast_to(&o, "java.time.MonthDay")),
            };
            t(JTime::LocalDate(md_at_year(m, d, *y as i64)?))
        }
        (T::Y(y), "isValidMonthDay", 1) => match a(0) {
            Val::Time(j) => match &*j {
                JTime::MonthDay(m, d) => Val::Bool(!(*m == 2 && *d == 29 && !is_leap(*y as i64))),
                _ => return Err(cast_to(&a(0), "java.time.MonthDay")),
            },
            Val::Null => Val::Bool(false),
            o => return Err(cast_to(&o, "java.time.MonthDay")),
        },
        // ---- MonthDay
        (T::MD(m, d), "atYear", 1) => t(JTime::LocalDate(md_at_year(*m, *d, int_arg(args, 0)?)?)),
        (T::MD(_, d), "isValidYear", 1) => {
            let y = int_arg(args, 0)?;
            Val::Bool(!(*d == 29 && !is_leap(y)))
        }
        (T::MD(m, d), "with", 1) => match a(0) {
            Val::Month(nm) => t(JTime::MonthDay(nm, (*d).min(month_len(nm, true)))),
            Val::Null => return Err(npe("month")),
            o => {
                let _ = m;
                return Err(cast_to(&o, "java.time.Month"));
            }
        },
        (T::MD(m, _), "withDayOfMonth", 1) => {
            let d = F::DayOfMonth.check(int_arg(args, 0)?)?;
            if d > month_len(*m, true) as i64 {
                return Err(dte(format!("Illegal value for DayOfMonth field, value {d} is not valid for month {}", MONTHS[*m as usize - 1])));
            }
            t(JTime::MonthDay(*m, d as u8))
        }
        (T::MD(_, d), "withMonth", 1) => {
            let m = F::MonthOfYear.check(int_arg(args, 0)?)? as u8;
            t(JTime::MonthDay(m, (*d).min(month_len(m, true))))
        }
        _ => return Ok(None),
    };
    ok(v)
}

fn md_at_year(m: u8, d: u8, y: i64) -> SResult<LDate> {
    let d = if m == 2 && d == 29 && !is_leap(y) { 28 } else { d };
    LDate::of(y, m as i64, d as i64)
}

fn display_name_args(args: &[Val]) -> SResult<(u8, Locale)> {
    let style = style_arg(&at(args, 0))?;
    let locale = locale_arg(&at(args, 1))?;
    Ok((style, locale))
}
/// `getDisplayName(style, locale)` of `DayOfWeek`/`Month`/`IsoEra` with JDK frames.
fn enum_display_name(args: &[Val], owner: Fr) -> SResult<(u8, Locale)> {
    if at(args, 0).is_null() {
        return Err(npe_req("textStyle", &[jf!("java.time.format.DateTimeFormatterBuilder", "appendText", "DateTimeFormatterBuilder", 805), owner]));
    }
    let style = style_arg(&at(args, 0))?;
    if at(args, 1).is_null() {
        return Err(npe_req("locale", &[jf!("java.time.format.DateTimeFormatterBuilder", "toFormatter", "DateTimeFormatterBuilder", 2439), jf!("java.time.format.DateTimeFormatterBuilder", "toFormatter", "DateTimeFormatterBuilder", 2417), owner]));
    }
    let locale = locale_arg(&at(args, 1))?;
    Ok((style, locale))
}

fn enum_method(recv: &Val, name: &str, args: &[Val], ordinal: i64, enum_name: &str) -> Option<Val> {
    Some(match (name, args.len()) {
        ("name", 0) => Val::str(enum_name),
        ("ordinal", 0) => Val::Int(ordinal as i32),
        ("compareTo", 1) => {
            let o = match (recv, &args[0]) {
                (Val::DayOfWeek(_), Val::DayOfWeek(d)) => *d as i64 - 1,
                (Val::Month(_), Val::Month(m)) => *m as i64 - 1,
                (Val::ChronoUnit(_), Val::ChronoUnit(u)) => *u as i64,
                (Val::Time(a), Val::Time(b)) => match (&**a, &**b) {
                    (JTime::Field(_), JTime::Field(f)) => CHRONO_FIELDS.iter().position(|x| x == f).map(|p| p as i64)?,
                    (JTime::TextStyle(_), JTime::TextStyle(s)) | (JTime::ResolverStyle(_), JTime::ResolverStyle(s)) | (JTime::FormatStyle(_), JTime::FormatStyle(s)) => *s as i64,
                    (JTime::Era(_), JTime::Era(e)) => *e as i64,
                    _ => return None,
                },
                _ => return None,
            };
            Val::Int((ordinal - o) as i32)
        }
        _ => return None,
    })
}

fn dow_method(recv: &Val, d: u8, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    if let Some(v) = enum_method(recv, name, args, d as i64 - 1, DAYS[d as usize - 1]) {
        return ok(v);
    }
    Ok(Some(match (name, n) {
        ("getValue", 0) => Val::Int(d as i32),
        ("plus" | "minus", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minus" { -(k % 7) } else { k % 7 };
            Val::DayOfWeek(((d as i64 - 1 + k).rem_euclid(7) + 1) as u8)
        }
        ("getDisplayName", 2) => {
            let (style, locale) = enum_display_name(args, jt!("DayOfWeek", "getDisplayName", 227))?;
            Val::string(day_name(d, style, &locale))
        }
        ("adjustInto", 1) => {
            let target = T::of(&args[0]).ok_or_else(|| if args[0].is_null() { npe("temporal") } else { cast_to(&args[0], "java.time.temporal.Temporal") })?;
            target.with_field(F::DayOfWeek, d as i64).map_err(|e| fr(e, &[LT_WITH_127, jt!("DayOfWeek", "adjustInto", 454)]))?.to_val()
        }
        _ => return Ok(None),
    }))
}

fn month_method(recv: &Val, m: u8, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    if let Some(v) = enum_method(recv, name, args, m as i64 - 1, MONTHS[m as usize - 1]) {
        return ok(v);
    }
    Ok(Some(match (name, n) {
        ("getValue", 0) => Val::Int(m as i32),
        ("plus" | "minus", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minus" { -(k % 12) } else { k % 12 };
            Val::Month(((m as i64 - 1 + k).rem_euclid(12) + 1) as u8)
        }
        ("getDisplayName", 2) => {
            let (style, locale) = enum_display_name(args, jt!("Month", "getDisplayName", 254))?;
            Val::string(month_name(m, style, &locale))
        }
        ("length", 1) => Val::Int(month_len(m, bool_arg(args, 0)?) as i32),
        ("maxLength", 0) => Val::Int(month_len(m, true) as i32),
        ("minLength", 0) => Val::Int(month_len(m, false) as i32),
        ("firstDayOfYear", 1) => Val::Int(first_day_of_year(m, bool_arg(args, 0)?) as i32),
        ("firstMonthOfQuarter", 0) => Val::Month((m - 1) / 3 * 3 + 1),
        ("adjustInto", 1) => {
            let target = T::of(&args[0]).ok_or_else(|| if args[0].is_null() { npe("temporal") } else { cast_to(&args[0], "java.time.temporal.Temporal") })?;
            target.with_field(F::MonthOfYear, m as i64).map_err(|e| fr(e, &[LT_WITH_127, jt!("Month", "adjustInto", 582)]))?.to_val()
        }
        _ => return Ok(None),
    }))
}

fn unit_method(recv: &Val, u: Unit, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    if u < 16 {
        if let Some(v) = enum_method(recv, name, args, u as i64, CHRONO_UNITS[u as usize]) {
            return ok(v);
        }
    }
    Ok(Some(match (name, n) {
        ("between", 2) => {
            let a = T::of(&args[0]).ok_or_else(|| {
                if args[0].is_null() {
                    frs(npe("Cannot invoke \"java.time.temporal.Temporal.until(java.time.temporal.Temporal, java.time.temporal.TemporalUnit)\" because \"temporal1Inclusive\" is null"), &[CU_BETWEEN])
                } else {
                    cast_to(&args[0], "java.time.temporal.Temporal")
                }
            })?;
            Val::Long(a.until(&args[1], u).map_err(|e| fr(e, &[CU_BETWEEN]))?)
        }
        ("addTo", 2) => {
            let a = T::of(&args[0]).ok_or_else(|| {
                if args[0].is_null() {
                    frs(npe("Cannot invoke \"java.time.temporal.Temporal.plus(long, java.time.temporal.TemporalUnit)\" because \"temporal\" is null"), &[CU_ADD_TO])
                } else {
                    cast_to(&args[0], "java.time.temporal.Temporal")
                }
            })?;
            a.plus_unit(long_arg(args, 1)?, u).map_err(|e| fr(e, &[if matches!(a, T::LT(_)) { LT_PLUS_127 } else { LD_PLUS_1273 }, CU_ADD_TO]))?.to_val()
        }
        ("getDuration", 0) => {
            let (s, nn) = unit_duration(u);
            t(JTime::Duration(s, nn))
        }
        ("isDateBased", 0) => Val::Bool(unit_is_date_based(u)),
        ("isTimeBased", 0) => Val::Bool(unit_is_time_based(u)),
        ("isDurationEstimated", 0) => Val::Bool(unit_is_estimated(u)),
        ("isSupportedBy", 1) => {
            let a = T::of(&args[0]).ok_or_else(|| cast_to(&args[0], "java.time.temporal.Temporal"))?;
            Val::Bool(a.plus_unit(1, u).is_ok())
        }
        _ => return Ok(None),
    }))
}

fn field_method(rt: &mut Rt, recv: &Val, f: F, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    if f.is_chrono() {
        let ord = CHRONO_FIELDS.iter().position(|x| *x == f).unwrap_or(0) as i64;
        if let Some(v) = enum_method(recv, name, args, ord, &f.enum_name()) {
            return ok(v);
        }
    }
    Ok(Some(match (name, n) {
        ("checkValidValue", 1) => Val::Long(f.check(long_arg(args, 0)?)?),
        ("checkValidIntValue", 1) => Val::Int(f.check_int(long_arg(args, 0)?)? as i32),
        ("getBaseUnit", 0) => unit_val(f.base_unit()),
        ("getRangeUnit", 0) => unit_val(f.range_unit()),
        ("isDateBased", 0) => Val::Bool(f.is_date_based()),
        ("isTimeBased", 0) => Val::Bool(f.is_time_based()),
        ("range", 0) => {
            let (lo, ms, ml) = f.range();
            t(JTime::Range(lo, lo, ms, ml))
        }
        ("getDisplayName", 1) => {
            locale_arg(&args[0])?;
            Val::str(f.name())
        }
        ("getFrom", 1) => {
            let v = accessor_method(rt, &args[0], "getLong", &[recv.clone()]).map_err(|e| fr(e, &[jf!("java.time.temporal.ChronoField", "getFrom", "ChronoField", 756)]))?;
            v.ok_or_else(|| cast_to(&args[0], "java.time.temporal.TemporalAccessor"))?
        }
        ("isSupportedBy", 1) => Val::Bool(Acc::of_any(&args[0]).is_some_and(|a| a.supports(f, &Locale::en()))),
        ("rangeRefinedBy", 1) => {
            let tv = T::of(&args[0]).ok_or_else(|| cast_to(&args[0], "java.time.temporal.TemporalAccessor"))?;
            let r = tv.range(f).map_err(|e| fr(e, &[jf!("java.time.temporal.ChronoField", "rangeRefinedBy", "ChronoField", 751)]))?;
            t(JTime::Range(r.0, r.1, r.2, r.3))
        }
        ("adjustInto", 2) => {
            let tv = T::of(&args[0]).ok_or_else(|| cast_to(&args[0], "java.time.temporal.Temporal"))?;
            tv.with_field(f, long_arg(args, 1)?).map_err(|e| fr(e, &[if matches!(tv, T::LT(_)) { LT_WITH_127 } else { LD_WITH_1070 }, jf!("java.time.temporal.ChronoField", "adjustInto", "ChronoField", 762)]))?.to_val()
        }
        _ => return Ok(None),
    }))
}

fn duration_method(d: (i64, u32), name: &str, args: &[Val]) -> SResult<Option<Val>> {
    duration_method0(d, name, args).map_err(|e| {
        let n = args.len();
        let m = e.msg.as_str();
        let outer: Vec<Fr> = match (name, n) {
            ("plusSeconds" | "minusSeconds", 1) => vec![DUR_PLUS_SECONDS_835],
            ("plusMillis" | "minusMillis", 1) => vec![jt!("Duration", "plusMillis", 848)],
            ("plusNanos" | "minusNanos", 1) => vec![jt!("Duration", "plusNanos", 861)],
            ("plusDays" | "minusDays", 1) => vec![jt!("Duration", "plusDays", 787)],
            ("plusHours" | "minusHours", 1) => vec![jt!("Duration", "plusHours", 803)],
            ("plusMinutes" | "minusMinutes", 1) => vec![jt!("Duration", "plusMinutes", 819)],
            ("plus", 1) if args[0].is_null() => vec![jt!("Duration", "plus", 739)],
            ("minus", 1) if args[0].is_null() => vec![jt!("Duration", "minus", 896)],
            ("plus", 1) => vec![jt!("Duration", "plus", 739)],
            ("plus", 2) if args[1].is_null() => vec![REQ_NON_NULL, jt!("Duration", "plus", 759)],
            ("compareTo", 1) if args[0].is_null() => vec![jt!("Duration", "compareTo", 1461)],
            ("toMillis", 0) => vec![MATH_MUL, MATH_MUL2, jt!("Duration", "toMillis", 1274)],
            ("toNanos", 0) => vec![MATH_MUL, jt!("Duration", "toNanos", 1297)],
            ("multipliedBy", 1) => vec![jt!("Duration", "multipliedBy", 1026)],
            ("dividedBy", 1) if m == "Cannot divide by zero" => vec![jt!("Duration", "dividedBy", 1040)],
            ("get", 1) => vec![jt!("Duration", "get", 587)],
            ("withNanos", 1) => vec![jt!("Duration", "withNanos", 724)],
            ("addTo", 1) if args[0].is_null() => vec![DUR_ADD_TO],
            _ => Vec::new(),
        };
        fr(e, &outer)
    })
}

fn duration_method0(d: (i64, u32), name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    let dv = |x: (i64, u32)| t(JTime::Duration(x.0, x.1));
    let total = dur_total_nanos(d);
    Ok(Some(match (name, n) {
        ("getSeconds", 0) => Val::Long(d.0),
        ("getNano", 0) => Val::Int(d.1 as i32),
        ("isZero", 0) => Val::Bool(d == (0, 0)),
        ("isNegative", 0) => Val::Bool(d.0 < 0),
        ("toDays", 0) => Val::Long(d.0 / SPD),
        ("toHours", 0) => Val::Long(d.0 / 3600),
        ("toMinutes", 0) => Val::Long(d.0 / 60),
        ("toMillis", 0) => Val::Long(i64::try_from(total.div_euclid(1_000_000)).map_err(|_| arith("long overflow"))?),
        ("toNanos", 0) => Val::Long(i64::try_from(total).map_err(|_| arith("long overflow"))?),
        ("abs", 0) => {
            if d.0 < 0 {
                dv(dur_negate(d)?)
            } else {
                dv(d)
            }
        }
        ("negated", 0) => dv(dur_negate(d)?),
        ("multipliedBy", 1) => {
            let k = long_arg(args, 0)?;
            dv(dur_from_nanos(total * k as i128)?)
        }
        ("dividedBy", 1) => {
            let k = long_arg(args, 0)?;
            if k == 0 {
                return Err(arith("Cannot divide by zero"));
            }
            dv(dur_from_nanos(total / k as i128)?)
        }
        ("plus" | "minus", 1) if args[0].is_null() => return Err(npe("Cannot invoke \"java.time.Duration.getSeconds()\" because \"duration\" is null")),
        ("plus", 2) if args[1].is_null() => return Err(npe("unit")),
        ("addTo" | "subtractFrom", 1) if args[0].is_null() => return Err(npe("Cannot invoke \"java.time.temporal.Temporal.plus(long, java.time.temporal.TemporalUnit)\" because \"temporal\" is null")),
        ("plus", 1) => {
            let o = duration_arg(&args[0])?;
            dv(dur_plus(d, o.0, o.1 as i64)?)
        }
        ("minus", 1) => {
            let o = duration_arg(&args[0])?;
            if o.0 == i64::MIN {
                dv(dur_plus(dur_plus(d, i64::MAX, -(o.1 as i64))?, 1, 0)?)
            } else {
                dv(dur_plus(d, -o.0, -(o.1 as i64))?)
            }
        }
        ("plus", 2) => dv(dur_plus_unit(d, long_arg(args, 0)?, unit_arg(&args[1])?)?),
        ("minus", 2) => {
            let amt = long_arg(args, 0)?;
            let u = unit_arg(&args[1])?;
            if amt == i64::MIN {
                dv(dur_plus_unit(dur_plus_unit(d, i64::MAX, u)?, 1, u)?)
            } else {
                dv(dur_plus_unit(d, -amt, u)?)
            }
        }
        ("plusDays" | "minusDays", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusDays" { k.checked_neg().ok_or_else(overflow)? } else { k };
            dv(dur_plus(d, k.checked_mul(SPD).ok_or_else(overflow)?, 0)?)
        }
        ("plusHours" | "minusHours", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusHours" { k.checked_neg().ok_or_else(overflow)? } else { k };
            dv(dur_plus(d, k.checked_mul(3600).ok_or_else(overflow)?, 0)?)
        }
        ("plusMinutes" | "minusMinutes", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusMinutes" { k.checked_neg().ok_or_else(overflow)? } else { k };
            dv(dur_plus(d, k.checked_mul(60).ok_or_else(overflow)?, 0)?)
        }
        ("plusSeconds" | "minusSeconds", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusSeconds" { k.checked_neg().ok_or_else(overflow)? } else { k };
            dv(dur_plus(d, k, 0)?)
        }
        ("plusMillis" | "minusMillis", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusMillis" { k.checked_neg().ok_or_else(overflow)? } else { k };
            dv(dur_plus(d, k / 1000, (k % 1000) * 1_000_000)?)
        }
        ("plusNanos" | "minusNanos", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusNanos" { k.checked_neg().ok_or_else(overflow)? } else { k };
            dv(dur_plus(d, 0, k)?)
        }
        ("withSeconds", 1) => dv((long_arg(args, 0)?, d.1)),
        ("withNanos", 1) => dv((d.0, F::NanoOfSecond.check_int(int_arg(args, 0)?)? as u32)),
        ("getUnits", 0) => Val::new_list(vec![Val::ChronoUnit(U_SECONDS), Val::ChronoUnit(U_NANOS)]),
        ("get", 1) => match unit_arg(&args[0])? {
            U_SECONDS => Val::Long(d.0),
            U_NANOS => Val::Long(d.1 as i64),
            u => return Err(unsupported_unit(u)),
        },
        ("addTo" | "subtractFrom", 1) => {
            let tv = T::of(&args[0]).ok_or_else(|| if args[0].is_null() { npe("temporal") } else { cast_to(&args[0], "java.time.temporal.Temporal") })?;
            tv.plus_amount(&t(JTime::Duration(d.0, d.1)), name == "subtractFrom")?.to_val()
        }
        ("compareTo", 1) => {
            let o = duration_arg(&args[0])?;
            let c = d.0.cmp(&o.0) as i64;
            Val::Int(if c != 0 { c as i32 } else { d.1 as i32 - o.1 as i32 })
        }
        _ => return Ok(None),
    }))
}

fn period_method(p: (i32, i32, i32), name: &str, args: &[Val]) -> SResult<Option<Val>> {
    period_method0(p, name, args).map_err(|e| {
        let n = args.len();
        let outer: Vec<Fr> = match (name, n) {
            ("plus", 1) if args[0].is_null() => vec![REQ_NON_NULL, jt!("Period", "from", 268), jt!("Period", "plus", 629)],
            ("minus", 1) if args[0].is_null() => vec![REQ_NON_NULL, jt!("Period", "from", 268), jt!("Period", "minus", 718)],
            ("plus", 1) => vec![jt!("Period", "from", 281), jt!("Period", "plus", 629)],
            ("minus", 1) => vec![jt!("Period", "from", 281), jt!("Period", "minus", 718)],
            ("get", 1) => vec![jt!("Period", "get", 443)],
            ("plusYears" | "minusYears", 1) => vec![jt!("Period", "plusYears", 653)],
            ("plusMonths" | "minusMonths", 1) => vec![jt!("Period", "plusMonths", 676)],
            ("plusDays" | "minusDays", 1) => vec![jt!("Period", "plusDays", 699)],
            ("multipliedBy", 1) => vec![jt!("Period", "multipliedBy", 796)],
            ("negated", 0) => vec![jt!("Period", "multipliedBy", 796), jt!("Period", "negated", 811)],
            ("normalized", 0) => vec![jt!("Period", "normalized", 843)],
            ("addTo" | "subtractFrom", 1) if args[0].is_null() => vec![REQ_NON_NULL, jt!("Period", "validateChrono", 971), jt!("Period", "addTo", 897)],
            ("addTo", 1) => vec![if p.2 != 0 && e.msg.ends_with("Days") { PER_ADD_TO_DAYS } else { PER_ADD_TO_MONTHS }],
            _ => Vec::new(),
        };
        fr(e, &outer)
    })
}

fn period_method0(p: (i32, i32, i32), name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    let pv = |x: (i32, i32, i32)| t(JTime::Period(x.0, x.1, x.2));
    let iov = || arith("integer overflow");
    let i32_arg = |i: usize| -> SResult<i32> { int_arg(args, i).map(|v| v as i32) };
    Ok(Some(match (name, n) {
        ("getYears", 0) => Val::Int(p.0),
        ("getMonths", 0) => Val::Int(p.1),
        ("getDays", 0) => Val::Int(p.2),
        ("getUnits", 0) => Val::new_list(vec![Val::ChronoUnit(U_YEARS), Val::ChronoUnit(U_MONTHS), Val::ChronoUnit(U_DAYS)]),
        ("get", 1) => match unit_arg(&args[0])? {
            U_YEARS => Val::Long(p.0 as i64),
            U_MONTHS => Val::Long(p.1 as i64),
            U_DAYS => Val::Long(p.2 as i64),
            u => return Err(unsupported_unit(u)),
        },
        ("getChronology", 0) => t(JTime::Chronology),
        ("isZero", 0) => Val::Bool(p == (0, 0, 0)),
        ("isNegative", 0) => Val::Bool(p.0 < 0 || p.1 < 0 || p.2 < 0),
        ("toTotalMonths", 0) => Val::Long(period_total_months(p)),
        ("normalized", 0) => {
            let total = period_total_months(p);
            let y = i32::try_from(total / 12).map_err(|_| int_overflow())?;
            pv((y, (total % 12) as i32, p.2))
        }
        ("negated", 0) => pv((p.0.checked_neg().ok_or_else(|| frs(iov(), &[MATH_MUL_INT]))?, p.1.checked_neg().ok_or_else(|| frs(iov(), &[MATH_MUL_INT]))?, p.2.checked_neg().ok_or_else(|| frs(iov(), &[MATH_MUL_INT]))?)),
        ("multipliedBy", 1) => {
            let k = i32_arg(0)?;
            let mul = |a: i32| a.checked_mul(k).ok_or_else(|| frs(iov(), &[MATH_MUL_INT]));
            pv((mul(p.0)?, mul(p.1)?, mul(p.2)?))
        }
        ("plus" | "minus", 1) => {
            let o = match &args[0] {
                Val::Time(j) => match &**j {
                    JTime::Period(y, m, d) => (*y, *m, *d),
                    JTime::Duration(..) => return Err(dte("Unit must be Years, Months or Days, but was Seconds")),
                    _ => return Err(cast_to(&args[0], "java.time.temporal.TemporalAmount")),
                },
                Val::Null => return Err(npe("amount")),
                o => return Err(cast_to(o, "java.time.temporal.TemporalAmount")),
            };
            let _ = &iov;
            if name == "plus" {
                pv((p.0.checked_add(o.0).ok_or_else(iov)?, p.1.checked_add(o.1).ok_or_else(iov)?, p.2.checked_add(o.2).ok_or_else(iov)?))
            } else {
                pv((p.0.checked_sub(o.0).ok_or_else(iov)?, p.1.checked_sub(o.1).ok_or_else(iov)?, p.2.checked_sub(o.2).ok_or_else(iov)?))
            }
        }
        ("plusYears" | "minusYears", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusYears" { -k } else { k };
            pv((i32::try_from(p.0 as i64 + k).map_err(|_| int_overflow())?, p.1, p.2))
        }
        ("plusMonths" | "minusMonths", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusMonths" { -k } else { k };
            pv((p.0, i32::try_from(p.1 as i64 + k).map_err(|_| int_overflow())?, p.2))
        }
        ("plusDays" | "minusDays", 1) => {
            let k = long_arg(args, 0)?;
            let k = if name == "minusDays" { -k } else { k };
            pv((p.0, p.1, i32::try_from(p.2 as i64 + k).map_err(|_| int_overflow())?))
        }
        ("withYears", 1) => pv((i32_arg(0)?, p.1, p.2)),
        ("withMonths", 1) => pv((p.0, i32_arg(0)?, p.2)),
        ("withDays", 1) => pv((p.0, p.1, i32_arg(0)?)),
        ("addTo" | "subtractFrom", 1) => {
            let tv = T::of(&args[0]).ok_or_else(|| if args[0].is_null() { npe("temporal") } else { cast_to(&args[0], "java.time.temporal.Temporal") })?;
            tv.plus_amount(&pv(p), name == "subtractFrom")?.to_val()
        }
        _ => return Ok(None),
    }))
}

fn zone_method(z: &Zone, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("getId", 0) => Val::string(z.id()),
        ("getRules", 0) => t(JTime::ZoneRules(z.clone())),
        ("normalized", 0) => t(JTime::Zone(z.normalized())),
        ("getDisplayName", 2) => {
            let (style, _locale) = display_name_args(args)?;
            Val::string(z.display_name(style <= 1))
        }
        ("getTotalSeconds", 0) => match z {
            Zone::Offset(o) => Val::Int(*o),
            _ => return Ok(None),
        },
        ("compareTo", 1) => match z {
            Zone::Offset(o) => Val::Int(offset_arg(&args[0], "other")? - *o),
            _ => return Ok(None),
        },
        ("adjustInto", 1) => match z {
            Zone::Offset(o) => {
                let tv = T::of(&args[0]).ok_or_else(|| cast_to(&args[0], "java.time.temporal.Temporal"))?;
                tv.with_field(F::OffsetSeconds, *o as i64)?.to_val()
            }
            _ => return Ok(None),
        },
        _ => return Ok(None),
    }))
}

fn find_transition(z: &Zone, from: i64, forward: bool) -> Option<(i64, i32, i32)> {
    if z.fixed_offset().is_some() {
        return None;
    }
    let step: i64 = 3600;
    let limit = 400 * SPD * 3;
    let mut prev = z.offset_at(from);
    let mut t0 = from;
    let mut travelled = 0;
    while travelled < limit {
        let t1 = if forward { t0 + step } else { t0 - step };
        let cur = z.offset_at(t1);
        if cur != prev {
            // Binary search for the exact second.
            let (mut lo, mut hi) = if forward { (t0, t1) } else { (t1, t0) };
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if z.offset_at(mid) == z.offset_at(lo) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some((hi, z.offset_at(lo), z.offset_at(hi)));
        }
        prev = cur;
        t0 = t1;
        travelled += step;
    }
    None
}

fn rules_method(z: &Zone, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("isFixedOffset", 0) => Val::Bool(z.fixed_offset().is_some()),
        ("getOffset", 1) => {
            let i = instant_arg(&args[0])?;
            t(JTime::Zone(Zone::Offset(z.offset_at(i.0))))
        }
        ("getStandardOffset", 1) => {
            let i = instant_arg(&args[0])?;
            t(JTime::Zone(Zone::Offset(z.standard_offset_at(i.0))))
        }
        ("getDaylightSavings", 1) => {
            let i = instant_arg(&args[0])?;
            t(JTime::Duration((z.offset_at(i.0) - z.standard_offset_at(i.0)) as i64, 0))
        }
        ("isDaylightSavings", 1) => {
            let i = instant_arg(&args[0])?;
            Val::Bool(z.is_dst_at(i.0))
        }
        ("getValidOffsets", 1) => {
            let ldt = ldt_arg(&args[0])?;
            let (offs, _) = z.valid_offsets(ldt.epoch_second(0));
            Val::new_list(offs.into_iter().map(|o| t(JTime::Zone(Zone::Offset(o)))).collect())
        }
        ("isValidOffset", 2) => {
            let ldt = ldt_arg(&args[0])?;
            let o = offset_arg(&args[1], "offset")?;
            Val::Bool(z.valid_offsets(ldt.epoch_second(0)).0.contains(&o))
        }
        ("getTransition", 1) => {
            let ldt = ldt_arg(&args[0])?;
            let (offs, gap) = z.valid_offsets(ldt.epoch_second(0));
            let pair = match (offs.len(), gap) {
                (_, Some(g)) => Some(g),
                (2, None) => Some((offs[0], offs[1])),
                _ => None,
            };
            match pair {
                Some((before, _)) => {
                    let local = ldt.epoch_second(0);
                    match find_transition(z, local - before as i64 - SPD, true) {
                        Some((secs, b, a)) => t(JTime::Transition(secs, b, a)),
                        None => Val::Null,
                    }
                }
                None => Val::Null,
            }
        }
        ("nextTransition", 1) => {
            let i = instant_arg(&args[0])?;
            find_transition(z, i.0, true).map(|(s, b, a)| t(JTime::Transition(s, b, a))).unwrap_or(Val::Null)
        }
        ("previousTransition", 1) => {
            let i = instant_arg(&args[0])?;
            find_transition(z, i.0, false).map(|(s, b, a)| t(JTime::Transition(s, b, a))).unwrap_or(Val::Null)
        }
        _ => return Ok(None),
    }))
}

fn transition_method(tr: (i64, i32, i32), name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let (secs, before, after) = tr;
    Ok(Some(match (name, args.len()) {
        ("getInstant", 0) => inst_to_val((secs, 0)),
        ("toEpochSecond", 0) => Val::Long(secs),
        ("getOffsetBefore", 0) => t(JTime::Zone(Zone::Offset(before))),
        ("getOffsetAfter", 0) => t(JTime::Zone(Zone::Offset(after))),
        ("getDateTimeBefore", 0) => t(JTime::LocalDateTime(LDT::from_epoch(secs, 0, before)?)),
        ("getDateTimeAfter", 0) => t(JTime::LocalDateTime(LDT::from_epoch(secs, 0, after)?)),
        ("getDuration", 0) => t(JTime::Duration((after - before) as i64, 0)),
        ("isGap", 0) => Val::Bool(after > before),
        ("isOverlap", 0) => Val::Bool(after < before),
        ("compareTo", 1) => match &args[0] {
            Val::Time(j) => match &**j {
                JTime::Transition(s, _, _) => Val::Int(secs.cmp(s) as i32),
                _ => return Err(cast_to(&args[0], "java.time.zone.ZoneOffsetTransition")),
            },
            o => return Err(cast_to(o, "java.time.zone.ZoneOffsetTransition")),
        },
        _ => return Ok(None),
    }))
}

fn clock_instant(j: &JTime) -> (i64, u32) {
    match j {
        JTime::Clock(s, n, _) => (*s, *n),
        JTime::OffsetClock(base, s, n) => {
            let b = clock_instant(base);
            inst_plus(b, *s, *n as i64).unwrap_or(b)
        }
        _ => (0, 0),
    }
}
fn clock_zone(j: &JTime) -> Zone {
    match j {
        JTime::Clock(_, _, z) => z.clone(),
        JTime::OffsetClock(base, _, _) => clock_zone(base),
        _ => Zone::Offset(0),
    }
}
fn clock_method(j: &JTime, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    Ok(Some(match (name, args.len()) {
        ("instant", 0) => inst_to_val(clock_instant(j)),
        ("millis", 0) => {
            let i = clock_instant(j);
            Val::Long(i.0.checked_mul(1000).and_then(|v| v.checked_add(i.1 as i64 / 1_000_000)).ok_or_else(overflow)?)
        }
        ("getZone", 0) => t(JTime::Zone(clock_zone(j))),
        _ => return Ok(None),
    }))
}

fn chronology_method(rt: &mut Rt, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    chronology_method0(rt, name, args).map_err(|e| {
        let outer: Vec<Fr> = match (name, args.len()) {
            ("date", 3) => vec![jf!("java.time.chrono.IsoChronology", "date", "IsoChronology", 212)],
            ("date", 1) => vec![jf!("java.time.chrono.IsoChronology", "date", "IsoChronology", 270)],
            ("dateYearDay", 2) => vec![jf!("java.time.chrono.IsoChronology", "dateYearDay", "IsoChronology", 241)],
            ("dateEpochDay", 1) => vec![jf!("java.time.chrono.IsoChronology", "dateEpochDay", "IsoChronology", 255)],
            ("eraOf", 1) => vec![jf!("java.time.chrono.IsoEra", "of", "IsoEra", 134), jf!("java.time.chrono.IsoChronology", "eraOf", "IsoChronology", 491)],
            ("localDateTime", 1) => vec![jf!("java.time.chrono.IsoChronology", "localDateTime", "IsoChronology", 372)],
            ("zonedDateTime", 1) => vec![jf!("java.time.chrono.IsoChronology", "zonedDateTime", "IsoChronology", 386)],
            ("zonedDateTime", 2) if args[1].is_null() => vec![REQ_NON_NULL, ZDT_OF_INSTANT_412, jf!("java.time.chrono.IsoChronology", "zonedDateTime", "IsoChronology", 401)],
            _ => Vec::new(),
        };
        fr(e, &outer)
    })
}

fn chronology_method0(rt: &mut Rt, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("getId", 0) => Val::str("ISO"),
        ("getCalendarType", 0) => Val::str("iso8601"),
        ("isLeapYear", 1) => Val::Bool(is_leap(long_arg(args, 0)?)),
        ("date", 3) => {
            let (y, m, d) = ints3(args)?;
            t(JTime::LocalDate(LDate::of(y, m, d)?))
        }
        ("date", 4) => {
            let era = match &args[0] {
                Val::Time(j) => match &**j {
                    JTime::Era(e) => *e as i64,
                    _ => return Err(cast_to(&args[0], "java.time.chrono.Era")),
                },
                Val::Null => return Err(npe("era")),
                o => return Err(cast_to(o, "java.time.chrono.Era")),
            };
            let yoe = int_arg(args, 1)?;
            let y = if era == 1 { yoe } else { 1 - yoe };
            t(JTime::LocalDate(LDate::of(y, int_arg(args, 2)?, int_arg(args, 3)?)?))
        }
        ("date", 1) => t(JTime::LocalDate(conv_ldate(&args[0])?)),
        ("dateEpochDay", 1) => t(JTime::LocalDate(LDate::from_epoch_day(long_arg(args, 0)?)?)),
        ("dateYearDay", 2) => t(JTime::LocalDate(LDate::of_year_day(int_arg(args, 0)?, int_arg(args, 1)?)?)),
        ("eraOf", 1) => {
            let v = int_arg(args, 0)?;
            if !(0..=1).contains(&v) {
                return Err(dte(format!("Invalid era: {v}")));
            }
            t(JTime::Era(v as u8))
        }
        ("eras", 0) => Val::new_list(vec![t(JTime::Era(0)), t(JTime::Era(1))]),
        ("period", 3) => {
            let (y, m, d) = ints3(args)?;
            t(JTime::Period(y as i32, m as i32, d as i32))
        }
        ("range", 1) => {
            let f = field_arg(&args[0])?;
            let (lo, ms, ml) = f.range();
            t(JTime::Range(lo, lo, ms, ml))
        }
        ("localDateTime", 1) => t(JTime::LocalDateTime(conv_ldt(&args[0])?)),
        ("zonedDateTime", 1) => conv_zoned(&args[0])?.to_val(),
        ("zonedDateTime", 2) => Zoned::of_instant(instant_arg(&args[0])?, zone_arg(&args[1], "zone")?)?.to_val(),
        ("prolepticYear", 2) => {
            let era = match &args[0] {
                Val::Time(j) => match &**j {
                    JTime::Era(e) => *e as i64,
                    _ => return Err(cast_to(&args[0], "java.time.chrono.Era")),
                },
                o => return Err(cast_to(o, "java.time.chrono.Era")),
            };
            let yoe = int_arg(args, 1)?;
            Val::Int(if era == 1 { yoe } else { 1 - yoe } as i32)
        }
        ("compareTo", 1) => Val::Int(0),
        ("resolveDate", 2) => {
            let _ = rt;
            return Ok(None);
        }
        _ => return Ok(None),
    }))
}

fn range_method(r: (i64, i64, i64, i64), name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let (min_s, min_l, max_s, max_l) = r;
    let valid = |v: i64| v >= min_s && v <= max_l;
    let is_int = min_s >= i32::MIN as i64 && max_l <= i32::MAX as i64;
    Ok(Some(match (name, args.len()) {
        ("getMinimum", 0) => Val::Long(min_s),
        ("getLargestMinimum", 0) => Val::Long(min_l),
        ("getSmallestMaximum", 0) => Val::Long(max_s),
        ("getMaximum", 0) => Val::Long(max_l),
        ("isFixed", 0) => Val::Bool(min_s == min_l && max_s == max_l),
        ("isIntValue", 0) => Val::Bool(is_int),
        ("isValidValue", 1) => Val::Bool(valid(long_arg(args, 0)?)),
        ("isValidIntValue", 1) => Val::Bool(is_int && valid(long_arg(args, 0)?)),
        ("checkValidValue" | "checkValidIntValue", 2) => {
            let v = long_arg(args, 0)?;
            let field = match &args[1] {
                Val::Null => None,
                f => Some(field_arg(f)?),
            };
            if !valid(v) || (name == "checkValidIntValue" && !is_int) {
                let rs = range_str(min_s, min_l, max_s, max_l);
                return Err(frs(
                    dte(match field {
                        Some(f) => format!("Invalid value for {} (valid values {rs}): {v}", f.name()),
                        None => format!("Invalid value (valid values {rs}): {v}"),
                    }),
                    &[if name == "checkValidIntValue" { VR_CHECK_INT } else { VR_CHECK }],
                ));
            }
            if name == "checkValidIntValue" {
                Val::Int(v as i32)
            } else {
                Val::Long(v)
            }
        }
        _ => return Ok(None),
    }))
}

fn locale_method(l: &Locale, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    Ok(Some(match (name, args.len()) {
        ("getLanguage", 0) => Val::string(l.lang.clone()),
        ("getCountry", 0) => Val::string(l.country.clone()),
        ("getVariant", 0) => Val::string(l.variant.clone()),
        ("getScript", 0) => Val::str(""),
        ("toLanguageTag", 0) => Val::string(l.tag()),
        ("getDisplayName", 0 | 1) => Val::string(l.display_name()),
        ("getDisplayLanguage", 0 | 1) => Val::string(Locale::new(&l.lang, "").display_name()),
        ("getDisplayCountry", 0 | 1) => Val::string(Locale::new("", &l.country).display_name()),
        ("stripExtensions", 0) => t(JTime::Locale(l.clone())),
        _ => return Ok(None),
    }))
}

fn formatter_method(rt: &mut Rt, f: &Rc<Fmt>, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    Ok(Some(match (name, n) {
        ("format", 1) => Val::string(f.format(&args[0])?),
        ("formatTo", 2) => {
            let s = f.format(&args[0])?;
            match &args[1] {
                Val::Builder(b) => {
                    b.borrow_mut().extend(s.encode_utf16());
                    Val::Null
                }
                o => return Err(cast_to(o, "java.lang.Appendable")),
            }
        }
        ("parse", 1) => {
            let text = str_arg(args, 0, "text")?;
            t(JTime::Parsed(f.parse(&text).map_err(|e| {
                let resolved = e.msg.contains("could not be parsed:");
                fr(e, &[if resolved { DTF_PARSE_1_ERR } else { DTF_PARSE_1 }])
            })?))
        }
        ("parse", 2) => {
            if args[1].is_null() {
                return Err(npe_req("query", &[jf!("java.time.format.DateTimeFormatter", "parse", "DateTimeFormatter", 2031)]));
            }
            let text = str_arg(args, 0, "text")?;
            let p = f.parse(&text).map_err(|e| {
                let resolved = e.msg.contains("could not be parsed:");
                fr(e, &[if resolved { DTF_PARSE_Q_ERR } else { DTF_PARSE_Q }])
            })?;
            let pv = t(JTime::Parsed(p));
            let abbr = if text.chars().count() > 64 { format!("{}...", text.chars().take(64).collect::<String>()) } else { text.clone() };
            match query_val(rt, &pv, &args[1]) {
                Ok(v) => v,
                Err(e) if e.kind == "date_time_exception" || e.kind == "unsupported_temporal_type_exception" => return Err(frs(dtpe(format!("Text '{abbr}' could not be parsed: {}", e.msg)), &[DTF_CREATE_ERROR, DTF_PARSE_Q_ERR])),
                Err(e) => return Err(e),
            }
        }
        ("withLocale", 1) if args[0].is_null() => return Err(npe_req("locale", &[jf!("java.time.format.DateTimeFormatter", "<init>", "DateTimeFormatter", 1505), jf!("java.time.format.DateTimeFormatter", "withLocale", "DateTimeFormatter", 1547)])),
        ("withResolverStyle", 1) if args[0].is_null() => return Err(npe_req("resolverStyle", &[jf!("java.time.format.DateTimeFormatter", "withResolverStyle", "DateTimeFormatter", 1768)])),
        ("withLocale", 1) => t(JTime::Formatter(Rc::new(f.with_locale(locale_arg(&args[0])?)))),
        ("withZone", 1) => {
            let z = if args[0].is_null() { None } else { Some(zone_arg(&args[0], "zone")?) };
            t(JTime::Formatter(Rc::new(f.with_zone(z))))
        }
        ("withResolverStyle", 1) => {
            let r = match &args[0] {
                Val::Time(j) => match &**j {
                    JTime::ResolverStyle(r) => *r,
                    _ => return Err(cast_to(&args[0], "java.time.format.ResolverStyle")),
                },
                Val::Null => return Err(npe("resolverStyle")),
                o => return Err(cast_to(o, "java.time.format.ResolverStyle")),
            };
            t(JTime::Formatter(Rc::new(f.with_resolver(r))))
        }
        ("withChronology", 1) | ("withDecimalStyle", 1) | ("withResolverFields", _) => t(JTime::Formatter(f.clone())),
        ("getLocale", 0) => t(JTime::Locale(f.locale.clone())),
        ("getZone", 0) => f.zone.clone().map(|z| t(JTime::Zone(z))).unwrap_or(Val::Null),
        ("getResolverStyle", 0) => t(JTime::ResolverStyle(f.resolver)),
        ("getChronology", 0) | ("getResolverFields", 0) => Val::Null,
        _ => return Ok(None),
    }))
}

/// Instance methods of `Val::Time`, `Val::Date`, `Val::Instant`,
/// `Val::DayOfWeek`, `Val::Month`, `Val::ChronoUnit` (consulted first;
/// `Ok(None)` falls back to the built-in methods).
pub fn call_method(rt: &mut Rt, recv: &Val, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let n = args.len();
    match (name, n) {
        ("toString", 0) => return ok(Val::string(recv.to_java_string())),
        ("equals", 1) => return ok(Val::Bool(recv.equals(&args[0]))),
        ("hashCode", 0) => {
            return ok(Val::Int(match as_jtime(recv) {
                Some(j) => j.hash(),
                None => recv.java_hash(),
            }));
        }
        _ => {}
    }
    match recv {
        Val::DayOfWeek(d) => {
            if let Some(v) = accessor_method(rt, recv, name, args)? {
                return ok(v);
            }
            dow_method(recv, *d, name, args)
        }
        Val::Month(m) => {
            if let Some(v) = accessor_method(rt, recv, name, args)? {
                return ok(v);
            }
            month_method(recv, *m, name, args)
        }
        Val::ChronoUnit(u) => unit_method(recv, *u, name, args),
        Val::Date(_) | Val::Instant(_) => temporal_method(rt, recv, name, args),
        Val::Time(j) => match &**j {
            JTime::Duration(s, nn) => duration_method((*s, *nn), name, args),
            JTime::Period(y, m, d) => period_method((*y, *m, *d), name, args),
            JTime::Zone(z) => zone_method(z, name, args),
            JTime::Formatter(f) => formatter_method(rt, f, name, args),
            JTime::Adjuster(a) => match (name, n) {
                ("adjustInto", 1) => {
                    let tv = T::of(&args[0]).ok_or_else(|| {
                        if args[0].is_null() {
                            frs(npe("Cannot invoke \"java.time.temporal.Temporal.range(java.time.temporal.TemporalField)\" because \"temporal\" is null"), &[ADJ_LAST_DAY_OF_MONTH])
                        } else {
                            cast_to(&args[0], "java.time.temporal.Temporal")
                        }
                    })?;
                    ok(apply_adjuster(a, &tv, rt)?.to_val())
                }
                _ => Ok(None),
            },
            JTime::TextStyle(s) => Ok(enum_method(recv, name, args, *s as i64, TEXT_STYLES[*s as usize]).or(match (name, n) {
                ("asStandalone", 0) => Some(t(JTime::TextStyle(*s | 1))),
                ("asNormal", 0) => Some(t(JTime::TextStyle(*s & !1))),
                ("isStandalone", 0) => Some(Val::Bool(*s & 1 == 1)),
                _ => None,
            })),
            JTime::ResolverStyle(s) => Ok(enum_method(recv, name, args, *s as i64, RESOLVER_STYLES[*s as usize])),
            JTime::FormatStyle(s) => Ok(enum_method(recv, name, args, *s as i64, FORMAT_STYLES[*s as usize])),
            JTime::Locale(l) => locale_method(l, name, args),
            JTime::Field(f) => field_method(rt, recv, *f, name, args),
            JTime::Unit(u) => unit_method(recv, *u, name, args),
            JTime::Clock(..) | JTime::OffsetClock(..) => clock_method(j, name, args),
            JTime::ZoneRules(z) => rules_method(z, name, args),
            JTime::Chronology => chronology_method(rt, name, args),
            JTime::Era(e) => Ok(enum_method(recv, name, args, *e as i64, if *e == 1 { "CE" } else { "BCE" }).or(match (name, n) {
                ("getValue", 0) => Some(Val::Int(*e as i32)),
                ("getDisplayName", 2) => {
                    let (style, locale) = display_name_args(args)?;
                    text_names(F::Era, style, &locale).into_iter().find(|(k, _)| *k == *e as i64).map(|(_, s)| Val::string(s))
                }
                _ => None,
            })),
            JTime::Range(a, b, c, d) => range_method((*a, *b, *c, *d), name, args),
            JTime::Parsed(_) => accessor_method(rt, recv, name, args),
            JTime::Query(_) => match (name, n) {
                ("queryFrom", 1) => ok(query_val(rt, &args[0], recv)?),
                _ => Ok(None),
            },
            JTime::Transition(s, b, a) => transition_method((*s, *b, *a), name, args),
            _ => temporal_method(rt, recv, name, args),
        },
        _ => Ok(None),
    }
}

/// `new X(args)` for java.time classes. `None`: not a class handled here.
pub fn construct(class: &str, args: &[Val]) -> Option<SResult<Val>> {
    Some(match (class, args.len()) {
        ("Locale", 1..=3) if args.iter().any(Val::is_null) => Err(frs(npe(""), &[jf!("java.util.Locale", "<init>", "Locale", 804), jf!("java.util.Locale", "<init>", "Locale", 861)])),
        ("Locale", 1..=3) if args.iter().any(|v|matches!(v,Val::Utf16(_))) => {
            return super::jmisc::construct(class,args);
        }
        ("Locale", 1..=3) => {
            let s = |i: usize| args.get(i).map(Val::to_java_string).unwrap_or_default();
            let mut l = Locale::new(&s(0), &s(1));
            l.variant = s(2);
            Ok(t(JTime::Locale(l)))
        }
        ("DateTimeException", 1 | 2) => Ok(Val::Exception(Rc::new(("DateTimeException".to_string(), args[0].to_java_string())))),
        ("DateTimeParseException", 3 | 4) => Ok(Val::Exception(Rc::new(("DateTimeParseException".to_string(), args[0].to_java_string())))),
        _ => return None,
    })
}

const KNOWN_TYPES: &[&str] = &[
    "Object", "Comparable", "Serializable", "Cloneable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "TemporalAmount", "TemporalUnit", "TemporalField", "TemporalQuery",
    "ChronoLocalDate", "ChronoLocalDateTime", "ChronoZonedDateTime", "ChronoPeriod", "Chronology", "AbstractChronology", "IsoChronology", "Era", "IsoEra", "LocalDate", "LocalDateTime",
    "LocalTime", "OffsetDateTime", "OffsetTime", "ZonedDateTime", "JodaCompatibleZonedDateTime", "Instant", "Duration", "Period", "ZoneId", "ZoneOffset", "YearMonth", "Year", "MonthDay",
    "DateTimeFormatter", "TextStyle", "Locale", "ChronoField", "ChronoUnit", "ResolverStyle", "FormatStyle", "Clock", "ZoneRules", "ValueRange", "ZoneOffsetTransition", "Parsed",
];

/// `v instanceof class` / reference-type assignment check for values
/// handled here; `None` when `v` is not a java.time value.
pub fn instance_of(v: &Val, class: &str) -> Option<bool> {
    if !KNOWN_TYPES.contains(&class) {
        return None;
    }
    let j = match v {
        Val::Date(ms) => JTime::Zoned(Zoned::from_ms(*ms)),
        Val::Instant(_) => JTime::Instant(0, 0),
        Val::Time(t) => return Some(instance_names(t).contains(&class)),
        _ => return None,
    };
    Some(instance_names(&j).contains(&class))
}

fn instance_names(j: &JTime) -> &'static [&'static str] {
    match j {
        JTime::LocalDate(_) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "ChronoLocalDate", "LocalDate"],
        JTime::LocalDateTime(_) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "ChronoLocalDateTime", "LocalDateTime"],
        JTime::LocalTime(_) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "LocalTime"],
        JTime::OffsetDateTime(..) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "OffsetDateTime"],
        JTime::OffsetTime(..) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "OffsetTime"],
        JTime::Zoned(_) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "ChronoZonedDateTime", "ZonedDateTime", "JodaCompatibleZonedDateTime"],
        JTime::Instant(..) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "Instant"],
        JTime::Duration(..) => &["Object", "Comparable", "Serializable", "TemporalAmount", "Duration"],
        JTime::Period(..) => &["Object", "Serializable", "TemporalAmount", "ChronoPeriod", "Period"],
        JTime::Zone(Zone::Offset(_)) => &["Object", "Comparable", "Serializable", "TemporalAccessor", "TemporalAdjuster", "ZoneId", "ZoneOffset"],
        JTime::Zone(_) => &["Object", "Serializable", "ZoneId"],
        JTime::YearMonth(..) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "YearMonth"],
        JTime::Year(_) => &["Object", "Comparable", "Serializable", "Temporal", "TemporalAccessor", "TemporalAdjuster", "Year"],
        JTime::MonthDay(..) => &["Object", "Comparable", "Serializable", "TemporalAccessor", "TemporalAdjuster", "MonthDay"],
        JTime::Formatter(_) => &["Object", "DateTimeFormatter"],
        JTime::Adjuster(_) => &["Object", "TemporalAdjuster"],
        JTime::TextStyle(_) => &["Object", "Comparable", "Serializable", "TextStyle"],
        JTime::Locale(_) => &["Object", "Cloneable", "Serializable", "Locale"],
        JTime::Field(f) => {
            if f.is_chrono() {
                &["Object", "Comparable", "Serializable", "TemporalField", "ChronoField"]
            } else {
                &["Object", "TemporalField"]
            }
        }
        JTime::Unit(_) => &["Object", "TemporalUnit"],
        JTime::ResolverStyle(_) => &["Object", "Comparable", "Serializable", "ResolverStyle"],
        JTime::FormatStyle(_) => &["Object", "Comparable", "Serializable", "FormatStyle"],
        JTime::Clock(..) | JTime::OffsetClock(..) => &["Object", "Clock"],
        JTime::ZoneRules(_) => &["Object", "Serializable", "ZoneRules"],
        JTime::Chronology => &["Object", "Comparable", "Chronology", "AbstractChronology", "IsoChronology"],
        JTime::Era(_) => &["Object", "Comparable", "Serializable", "TemporalAccessor", "TemporalAdjuster", "Era", "IsoEra"],
        JTime::Range(..) => &["Object", "Serializable", "ValueRange"],
        JTime::Parsed(_) => &["Object", "TemporalAccessor", "Parsed"],
        JTime::Query(_) => &["Object", "TemporalQuery"],
        JTime::Transition(..) => &["Object", "Comparable", "Serializable", "ZoneOffsetTransition"],
    }
}
