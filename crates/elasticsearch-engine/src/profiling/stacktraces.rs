//! `POST /_profiling/stacktraces` (ES `GetStackTracesRequest`,
//! `TransportGetStackTracesAction`, `StackTrace`, `StackFrame`): the
//! base of the flamegraph and topN APIs too.

use super::costs::{Calculator, CostParams, HostMetadata};
use super::java::{JConcurrentKeys, JHashMap};
use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts};
use crate::search::Target;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};

const DEFAULT_SAMPLE_SIZE: i64 = 20_000;
const MAX_TRACE_EVENTS_RESULT_SIZE: i64 = 150_000;
const CUSTOM_EVENT_SUB_AGGREGATION_NAME: &str = "custom_event_group_";
/// `xpack.profiling.kv_index.overlap` default (6h).
const KV_INDEX_OVERLAP_MS: i64 = 6 * 3600 * 1000;
pub const NATIVE_FRAME_TYPE: i32 = 3;
pub const KERNEL_FRAME_TYPE: i32 = 4;

// ------------------------------------------------------------------ request

#[derive(Default, Debug, Clone)]
pub struct StackTracesRequest {
    pub query: Option<Value>,
    pub sample_size: i64,
    pub limit: Option<i64>,
    pub indices: Option<Vec<String>>,
    pub user_provided_indices: bool,
    pub stacktrace_ids_field: Option<String>,
    pub aggregation_field: Option<String>,
    pub aggregation_fields: Option<Vec<String>>,
    pub requested_duration: Option<f64>,
    pub costs: CostParams,
    /// Flamegraph / topN up-scale counts by the sampling rate.
    pub adjust_sample_count: bool,
}

impl StackTracesRequest {
    pub fn aggregation_fields_effective(&self) -> Option<Vec<String>> {
        match &self.aggregation_field {
            Some(f) => Some(vec![f.clone()]),
            None => self.aggregation_fields.clone(),
        }
    }
    pub fn is_legacy_aggregation_field(&self) -> bool {
        self.aggregation_field.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    StartObject,
    EndObject,
    StartArray,
    EndArray,
    Field(String),
    Str(String),
    Num(String),
    Bool(bool),
    Null,
}

impl Tok {
    fn name(&self) -> &'static str {
        match self {
            Tok::StartObject => "START_OBJECT",
            Tok::EndObject => "END_OBJECT",
            Tok::StartArray => "START_ARRAY",
            Tok::EndArray => "END_ARRAY",
            Tok::Field(_) => "FIELD_NAME",
            Tok::Str(_) => "VALUE_STRING",
            Tok::Num(_) => "VALUE_NUMBER",
            Tok::Bool(_) => "VALUE_BOOLEAN",
            Tok::Null => "VALUE_NULL",
        }
    }
    fn is_value(&self) -> bool {
        matches!(self, Tok::Str(_) | Tok::Num(_) | Tok::Bool(_) | Tok::Null)
    }
}

/// (token, line, col, byte offset of the token start).
type Spanned = (Tok, u64, u64, usize);

/// Minimal JSON tokenizer with Jackson-style token locations (1-based).
fn tokenize(src: &str) -> Result<Vec<Spanned>> {
    let b = src.as_bytes();
    let (mut i, mut line, mut line_start) = (0usize, 1u64, 0usize);
    let mut out = Vec::new();
    // Whether the next string inside an object is a field name.
    let mut stack: Vec<bool> = Vec::new();
    let mut expect_key = false;
    let err = |l: u64, c: u64, m: &str| EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] {m}"));
    while i < b.len() {
        let c = b[i];
        let col = (i - line_start + 1) as u64;
        match c {
            b'\n' => {
                line += 1;
                line_start = i + 1;
                i += 1;
            }
            b' ' | b'\t' | b'\r' | b',' | b':' => i += 1,
            b'{' => {
                out.push((Tok::StartObject, line, col, i));
                stack.push(true);
                expect_key = true;
                i += 1;
            }
            b'}' => {
                out.push((Tok::EndObject, line, col, i));
                stack.pop();
                expect_key = stack.last().copied().unwrap_or(false);
                i += 1;
            }
            b'[' => {
                out.push((Tok::StartArray, line, col, i));
                stack.push(false);
                expect_key = false;
                i += 1;
            }
            b']' => {
                out.push((Tok::EndArray, line, col, i));
                stack.pop();
                expect_key = stack.last().copied().unwrap_or(false);
                i += 1;
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                if i >= b.len() {
                    return Err(err(line, col, "Unexpected end-of-input in a String value"));
                }
                i += 1;
                let s: String = serde_json::from_str(&src[start..i]).map_err(|e| err(line, col, &e.to_string()))?;
                let in_obj = stack.last().copied().unwrap_or(false);
                if in_obj && expect_key {
                    out.push((Tok::Field(s), line, col, start));
                    expect_key = false;
                } else {
                    out.push((Tok::Str(s), line, col, start));
                    expect_key = in_obj;
                }
            }
            _ => {
                let start = i;
                while i < b.len() && !matches!(b[i], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t' | b':') {
                    i += 1;
                }
                let word = &src[start..i];
                let t = match word {
                    "true" => Tok::Bool(true),
                    "false" => Tok::Bool(false),
                    "null" => Tok::Null,
                    w if w.parse::<f64>().is_ok() => Tok::Num(w.to_string()),
                    w => return Err(err(line, col, &format!("Unrecognized token '{w}': was expecting (JSON String, Number, Array, Object or token 'null', 'true' or 'false')"))),
                };
                out.push((t, line, col, start));
                expect_key = stack.last().copied().unwrap_or(false);
            }
        }
    }
    Ok(out)
}

fn parsing(msg: String, line: u64, col: u64) -> EsError {
    EsError::parsing_at(msg, line, col)
}

/// End index (exclusive) of the value starting at token `i`.
fn value_end(toks: &[Spanned], i: usize) -> usize {
    let mut depth = 0i32;
    let mut j = i;
    loop {
        match toks[j].0 {
            Tok::StartObject | Tok::StartArray => depth += 1,
            Tok::EndObject | Tok::EndArray => depth -= 1,
            _ => {}
        }
        j += 1;
        if depth <= 0 || j >= toks.len() {
            return j;
        }
    }
}

/// `GetStackTracesRequest.parseXContent` (+ defaults); `None` body = all
/// defaults.
pub fn parse_request(body: Option<&str>) -> Result<StackTracesRequest> {
    let mut r = StackTracesRequest { sample_size: DEFAULT_SAMPLE_SIZE, ..Default::default() };
    let Some(src) = body.filter(|s| !s.trim().is_empty()) else { return Ok(r) };
    let toks = tokenize(src)?;
    let mut i = 0;
    match toks.first() {
        Some((Tok::StartObject, ..)) => i += 1,
        Some((t, l, c, _)) => return Err(parsing(format!("Expected [START_OBJECT] but found [{}].", t.name()), *l, *c)),
        None => return Ok(r),
    }
    let mut current: Option<String> = None;
    let int_of = |t: &Tok, l: u64, c: u64, name: &str| -> Result<i64> {
        match t {
            Tok::Num(n) => Ok(n.parse::<f64>().map(|f| f as i64).unwrap_or(0)),
            Tok::Str(s) => s.trim().parse::<f64>().map(|f| f as i64).map_err(|_| EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] [{name}] failed to parse field"))),
            other => Err(EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] Current token ({}) not numeric, can not use numeric value accessors", other.name()))),
        }
    };
    let dbl_of = |t: &Tok, l: u64, c: u64, name: &str| -> Result<f64> {
        match t {
            Tok::Num(n) => Ok(n.parse::<f64>().unwrap_or(0.0)),
            Tok::Str(s) => s.trim().parse::<f64>().map_err(|_| EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] [{name}] failed to parse field"))),
            other => Err(EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] Current token ({}) not numeric, can not use numeric value accessors", other.name()))),
        }
    };
    let text_of = |t: &Tok| -> String {
        match t {
            Tok::Str(s) | Tok::Num(s) => s.clone(),
            Tok::Bool(b) => b.to_string(),
            _ => String::new(),
        }
    };
    while i < toks.len() {
        let (t, l, c, off) = toks[i].clone();
        match &t {
            Tok::EndObject => {
                i += 1;
                break;
            }
            Tok::Field(f) => {
                current = Some(f.clone());
                i += 1;
            }
            t if t.is_value() => {
                let name = current.clone().unwrap_or_default();
                match name.as_str() {
                    "sample_size" => r.sample_size = int_of(t, l, c, &name)?,
                    "limit" => r.limit = Some(int_of(t, l, c, &name)?),
                    "stacktrace_ids_field" => r.stacktrace_ids_field = Some(text_of(t)),
                    "aggregation_field" => r.aggregation_field = Some(text_of(t)),
                    "requested_duration" => r.requested_duration = Some(dbl_of(t, l, c, &name)?),
                    "aws_cost_factor" => r.costs.aws_cost_factor = Some(dbl_of(t, l, c, &name)?),
                    "azure_cost_factor" => r.costs.azure_cost_factor = Some(dbl_of(t, l, c, &name)?),
                    "co2_per_kwh" => r.costs.co2_per_kwh = Some(dbl_of(t, l, c, &name)?),
                    "datacenter_pue" => r.costs.datacenter_pue = Some(dbl_of(t, l, c, &name)?),
                    "per_core_watt_x86" => r.costs.per_core_watt_x86 = Some(dbl_of(t, l, c, &name)?),
                    "per_core_watt_arm64" => r.costs.per_core_watt_arm64 = Some(dbl_of(t, l, c, &name)?),
                    "cost_per_core_hour" => r.costs.cost_per_core_hour = Some(dbl_of(t, l, c, &name)?),
                    _ => return Err(parsing(format!("Unknown key for a {} in [{name}].", t.name()), l, c)),
                }
                i += 1;
            }
            Tok::StartObject => {
                if current.as_deref() == Some("query") {
                    let end = value_end(&toks, i);
                    let end_off = toks[end - 1].3 + 1;
                    let q: Value = serde_json::from_str(&src[off..end_off]).map_err(|e| EsError::json_parse(&e))?;
                    r.query = Some(q);
                    i = end;
                } else {
                    // ES does not skip an unknown object: its fields are read
                    // as if they were top-level fields.
                    i += 1;
                }
            }
            Tok::StartArray => {
                let name = current.clone().unwrap_or_default();
                if name != "indices" && name != "aggregation_fields" {
                    return Err(parsing(format!("Unexpected token {} in [{name}].", t.name()), l, c));
                }
                let mut vals = Vec::new();
                i += 1;
                loop {
                    let Some((x, xl, xc, _)) = toks.get(i).cloned() else { break };
                    match x {
                        Tok::EndArray => {
                            i += 1;
                            break;
                        }
                        Tok::Str(s) => {
                            vals.push(s);
                            i += 1;
                        }
                        other => return Err(parsing(format!("Expected [VALUE_STRING] but found [{}] in [{name}].", other.name()), xl, xc)),
                    }
                }
                if name == "indices" {
                    r.indices = Some(vals);
                    r.user_provided_indices = true;
                } else {
                    r.aggregation_fields = Some(vals);
                }
            }
            other => return Err(parsing(format!("Unknown key for a {} in [{}].", other.name(), current.clone().unwrap_or_default()), l, c)),
        }
    }
    if let Some((t, l, c, _)) = toks.get(i) {
        return Err(parsing(format!("Unexpected token [{}] found after the main object.", t.name()), *l, *c));
    }
    if r.indices.as_ref().is_some_and(|v| v.is_empty()) {
        r.user_provided_indices = true;
    }
    Ok(r)
}

fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        if v.is_sign_negative() || v != 0.0 {
            return format!("{}", v as i64);
        }
        return "0".into();
    }
    format!("{v}")
}

/// `GetStackTracesRequest.validate`.
pub fn validate(r: &StackTracesRequest) -> Result<()> {
    let mut errs: Vec<String> = Vec::new();
    if r.user_provided_indices {
        if r.stacktrace_ids_field.as_deref().is_none_or(str::is_empty) {
            errs.push("[stacktrace_ids_field] is mandatory".into());
        }
    } else if r.stacktrace_ids_field.is_some() {
        errs.push("[stacktrace_ids_field] must not be set".into());
    }
    if r.aggregation_field.is_some() && r.aggregation_fields.is_some() {
        errs.push("[aggregation_field] must not be set when [aggregation_fields] is also set".into());
    }
    if let Some(f) = &r.aggregation_fields {
        if f.is_empty() || f.len() > 2 {
            errs.push(format!("[aggregation_fields] must contain either one or two elements but contains [{}] elements.", f.len()));
        }
    }
    if r.aggregation_field.as_deref().is_some_and(|f| f.trim().is_empty()) {
        errs.push("[aggregation_field] must be non-empty".into());
    }
    let mut positive = |name: &str, v: Option<f64>, int: bool| {
        if let Some(v) = v {
            if v <= 0.0 {
                let shown = if int { fmt_num(v) } else { java_double(v) };
                errs.push(format!("[{name}] must be greater than 0, got: {shown}"));
            }
        }
    };
    positive("sample_size", Some(r.sample_size as f64), true);
    positive("limit", r.limit.map(|v| v as f64), true);
    positive("requested_duration", r.requested_duration, false);
    positive("aws_cost_factor", r.costs.aws_cost_factor, false);
    positive("azure_cost_factor", r.costs.azure_cost_factor, false);
    positive("co2_per_kwh", r.costs.co2_per_kwh, false);
    positive("datacenter_pue", r.costs.datacenter_pue, false);
    positive("per_core_watt_x86", r.costs.per_core_watt_x86, false);
    positive("per_core_watt_arm64", r.costs.per_core_watt_arm64, false);
    positive("cost_per_core_hour", r.costs.cost_per_core_hour, false);
    if errs.is_empty() {
        Ok(())
    } else {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        Err(EsError::action_validation(&refs))
    }
}

/// `Double.toString` for the values a validation message shows.
fn java_double(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e7 {
        format!("{:.1}", v)
    } else {
        format!("{v}")
    }
}

// ------------------------------------------------------------------ model

#[derive(Clone, Debug)]
pub struct SubGroup {
    pub name: String,
    pub count: Option<i64>,
    pub legacy: bool,
    pub subgroups: JHashMap<SubGroup>,
}

impl SubGroup {
    pub fn root(name: &str, legacy: bool) -> SubGroup {
        SubGroup { name: name.into(), count: None, legacy, subgroups: JHashMap::default() }
    }
    fn add_count(&mut self, name: &str, count: i64) {
        match self.subgroups.get_mut(name) {
            Some(s) => s.count = Some(s.count.unwrap_or(0) + count),
            None => self.subgroups.put(name.to_string(), SubGroup { name: name.into(), count: Some(count), legacy: self.legacy, subgroups: JHashMap::default() }),
        }
    }
    fn get_or_add_child(&mut self, name: &str) -> &mut SubGroup {
        if !self.subgroups.contains_key(name) {
            self.subgroups.put(name.to_string(), SubGroup { name: name.into(), count: None, legacy: self.legacy, subgroups: JHashMap::default() });
        }
        self.subgroups.get_mut(name).unwrap()
    }
    pub fn copy(&self) -> SubGroup {
        let mut m = JHashMap::with_capacity(self.subgroups.len());
        for (k, v) in self.subgroups.iter() {
            m.put(k.clone(), v.copy());
        }
        SubGroup { name: self.name.clone(), count: self.count, legacy: self.legacy, subgroups: m }
    }
    pub fn merge(&mut self, s: &SubGroup) {
        if self.name != s.name {
            return;
        }
        match (self.count, s.count) {
            (Some(a), Some(b)) => self.count = Some(a + b),
            (None, b) => self.count = b,
            _ => {}
        }
        for sub in s.subgroups.values() {
            match self.subgroups.get_mut(&sub.name) {
                Some(mine) => mine.merge(sub),
                None => self.subgroups.put(sub.name.clone(), sub.copy()),
            }
        }
    }
    /// `toXContent` into the enclosing object.
    pub fn render_into(&self, out: &mut Map<String, Value>) {
        if self.legacy {
            for s in self.subgroups.values() {
                out.insert(s.name.clone(), json!(s.count));
            }
        } else {
            let mut o = Map::new();
            if let Some(c) = self.count {
                o.insert("count".into(), json!(c));
            }
            for s in self.subgroups.values() {
                s.render_into(&mut o);
            }
            out.insert(self.name.clone(), Value::Object(o));
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TraceEvent {
    pub count: i64,
    pub annual_co2_tons: f64,
    pub annual_costs_usd: f64,
    pub sub_groups: Option<SubGroup>,
}

#[derive(Clone, Debug)]
pub struct StackTrace {
    pub address_or_lines: Vec<i32>,
    pub file_ids: Vec<String>,
    pub frame_ids: Vec<String>,
    pub type_ids: Vec<i32>,
    pub annual_co2_tons: f64,
    pub annual_costs_usd: f64,
    pub count: i64,
    pub sub_groups: Option<SubGroup>,
}

const SAFE_BASE64_ENCODER: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234456789-_";

const SAFE_BASE64_DECODER: [i32; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0, 0,
    0, 0, 63, 0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0, 0, 0, 0, 0,
];

fn char_code_at(s: &[u16], i: usize) -> i32 {
    SAFE_BASE64_DECODER[(s.get(i).copied().unwrap_or(0) & 0x7f) as usize]
}

/// `StackTrace.runLengthDecodeBase64Url`.
pub fn run_length_decode(input: &str, capacity: usize) -> Vec<i32> {
    let s: Vec<u16> = input.encode_utf16().collect();
    let size = s.len();
    let mut out = vec![0i32; capacity];
    let mut j = 0usize;
    let mut fill = |out: &mut Vec<i32>, j: &mut usize, count: i32, value: i32| {
        for _ in 0..count.max(0) {
            if *j < out.len() {
                out[*j] = value;
            }
            *j += 1;
        }
    };
    let ch = |i: usize| char_code_at(&s, i);
    let mut i = 0usize;
    while i < (size / 8) * 8 {
        let n = (ch(i) << 26) | (ch(i + 1) << 20) | (ch(i + 2) << 14) | (ch(i + 3) << 8) | (ch(i + 4) << 2) | (ch(i + 5) >> 4);
        fill(&mut out, &mut j, (n >> 24) & 0xff, (n >> 16) & 0xff);
        fill(&mut out, &mut j, (n >> 8) & 0xff, n & 0xff);
        let n = ((ch(i + 5) & 0xf) << 12) | (ch(i + 6) << 6) | ch(i + 7);
        fill(&mut out, &mut j, (n >> 8) & 0xff, n & 0xff);
        i += 8;
    }
    match size % 8 {
        6 => {
            let n = (ch(i) << 26) | (ch(i + 1) << 20) | (ch(i + 2) << 14) | (ch(i + 3) << 8) | (ch(i + 4) << 2) | (ch(i + 5) >> 4);
            fill(&mut out, &mut j, (n >> 24) & 0xff, (n >> 16) & 0xff);
            fill(&mut out, &mut j, (n >> 8) & 0xff, n & 0xff);
        }
        3 => {
            let n = ((ch(i) << 12) | (ch(i + 1) << 6) | ch(i + 2)) >> 2;
            fill(&mut out, &mut j, (n >> 8) & 0xff, n & 0xff);
        }
        _ => {}
    }
    out
}

/// `StackTrace.getAddressFromStackFrameID`.
pub fn address_from_frame_id(frame_id: &str) -> i32 {
    let s: Vec<u16> = frame_id.encode_utf16().collect();
    let mut a = char_code_at(&s, 21) & 0xf;
    for i in 22..=31 {
        a = a.wrapping_shl(6).wrapping_add(char_code_at(&s, i));
    }
    a
}

/// `StackTrace.getFileIDFromStackFrameID`.
pub fn file_id_from_frame_id(frame_id: &str) -> String {
    let s: Vec<u16> = frame_id.encode_utf16().collect();
    let prefix = String::from_utf16_lossy(&s[..21.min(s.len())]);
    let c = s.get(21).copied().unwrap_or(0) as usize & 0x30;
    format!("{prefix}{}", SAFE_BASE64_ENCODER[c] as char)
}

/// Value at a dotted path: the nested form, else the dotted key.
fn path<'a>(src: &'a Value, dotted: &str) -> Option<&'a Value> {
    let mut cur = src;
    let mut ok = true;
    for p in dotted.split('.') {
        match cur.get(p) {
            Some(v) => cur = v,
            None => {
                ok = false;
                break;
            }
        }
    }
    if ok {
        return Some(cur);
    }
    src.get(dotted)
}

/// `StackTrace.fromSource`.
pub fn stack_trace_from_source(src: &Value) -> StackTrace {
    let get = |k: &str| -> String {
        let nested = {
            let mut cur = Some(src);
            for p in k.split('.') {
                cur = cur.and_then(|c| c.get(p));
            }
            cur.and_then(Value::as_str).map(str::to_string)
        };
        nested.or_else(|| src.get(k).and_then(Value::as_str).map(str::to_string)).unwrap_or_default()
    };
    let ids = get("Stacktrace.frame.ids");
    let types = get("Stacktrace.frame.types");
    let n = ids.len() / 32;
    let (mut file_ids, mut frame_ids, mut addrs) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let f = &ids[i * 32..i * 32 + 32];
        frame_ids.push(f.to_string());
        file_ids.push(file_id_from_frame_id(f));
        addrs.push(address_from_frame_id(f));
    }
    let type_ids = run_length_decode(&types, n);
    StackTrace { address_or_lines: addrs, file_ids, frame_ids, type_ids, annual_co2_tons: 0.0, annual_costs_usd: 0.0, count: 0, sub_groups: None }
}

/// `StackFrame`: list-valued fields (a scalar becomes a one-element list).
#[derive(Clone, Debug, Default)]
pub struct StackFrame {
    pub file_name: Vec<Value>,
    pub function_name: Vec<Value>,
    pub function_offset: Vec<Value>,
    pub line_number: Vec<Value>,
}

fn list_of(v: Option<&Value>) -> Vec<Value> {
    match v {
        Some(Value::Array(a)) => a.clone(),
        Some(Value::Null) | None => vec![],
        Some(o) => vec![o.clone()],
    }
}

pub struct Frame {
    pub file_name: String,
    pub function_name: String,
    pub function_offset: i32,
    pub line_number: i32,
    pub inline: bool,
    pub last: bool,
}

fn as_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

impl StackFrame {
    pub fn from_source(src: &Value) -> StackFrame {
        StackFrame {
            file_name: list_of(src.get("Stackframe.file.name")),
            function_name: list_of(src.get("Stackframe.function.name")),
            function_offset: list_of(src.get("Stackframe.function.offset")),
            line_number: list_of(src.get("Stackframe.line.number")),
        }
    }
    pub fn size(&self) -> usize {
        self.function_name.len()
    }
    pub fn frames(&self) -> Vec<Frame> {
        let n = self.size();
        (0..n)
            .map(|i| Frame {
                file_name: self.file_name.get(i).map(as_text).unwrap_or_default(),
                function_name: as_text(&self.function_name[i]),
                function_offset: self.function_offset.get(i).and_then(Value::as_i64).unwrap_or(0) as i32,
                line_number: self.line_number.get(i).and_then(Value::as_i64).unwrap_or(0) as i32,
                inline: i > 0,
                last: i == n - 1,
            })
            .collect()
    }
    /// `StackFrame.EMPTY_STACKFRAME` (one empty frame).
    pub fn empty() -> StackFrame {
        StackFrame { file_name: vec![json!("")], function_name: vec![json!("")], function_offset: vec![json!(0)], line_number: vec![json!(0)] }
    }
    fn to_json(&self) -> Value {
        json!({"file_name": self.file_name, "function_name": self.function_name, "function_offset": self.function_offset, "line_number": self.line_number})
    }
}

/// One ES 9 trace event: samples of a stacktrace in one executable / thread.
#[derive(Clone, Debug, Default)]
pub struct TraceEventRow {
    pub executable: String,
    pub thread: String,
    pub stacktrace: String,
    pub count: i64,
    pub annual_co2_tons: f64,
    pub annual_costs_usd: f64,
}

/// The result of the stacktraces action (`GetStackTracesResponse`).
#[derive(Default)]
pub struct StackTracesResponse {
    /// In the iteration order of ES's `ConcurrentHashMap`.
    pub stack_traces: Option<Vec<(String, StackTrace)>>,
    pub stack_frames: Option<Vec<(String, StackFrame)>>,
    pub executables: Option<Vec<(String, String)>>,
    pub events: Option<BTreeMap<String, TraceEvent>>,
    /// ES 9 `TraceEventID`s: per (executable, thread, stacktrace) samples
    /// and their CO2 / costs, sorted by (stacktrace, executable, thread).
    pub trace_events: Vec<TraceEventRow>,
    pub total_frames: i64,
    pub sampling_rate: f64,
    pub total_samples: i64,
}

impl StackTracesResponse {
    pub fn frame(&self, id: &str) -> Option<&StackFrame> {
        self.stack_frames.as_ref()?.iter().find(|(k, _)| k == id).map(|(_, v)| v)
    }
    pub fn to_json(&self) -> Value {
        let mut o = Map::new();
        if let Some(st) = &self.stack_traces {
            let m: Map<String, Value> = st
                .iter()
                .map(|(k, s)| {
                    (
                        k.clone(),
                        json!({
                            "address_or_lines": s.address_or_lines, "file_ids": s.file_ids, "frame_ids": s.frame_ids, "type_ids": s.type_ids,
                            "annual_co2_tons": s.annual_co2_tons, "annual_costs_usd": s.annual_costs_usd, "count": s.count,
                        }),
                    )
                })
                .collect();
            o.insert("stack_traces".into(), Value::Object(m));
        }
        if let Some(sf) = &self.stack_frames {
            o.insert("stack_frames".into(), Value::Object(sf.iter().map(|(k, f)| (k.clone(), f.to_json())).collect()));
        }
        if let Some(ex) = &self.executables {
            o.insert("executables".into(), Value::Object(ex.iter().map(|(k, f)| (k.clone(), json!(f))).collect()));
        }
        if let Some(ev) = &self.events {
            o.insert("stack_trace_events".into(), Value::Object(ev.iter().map(|(k, e)| (k.clone(), json!(e.count))).collect()));
        }
        o.insert("total_frames".into(), json!(self.total_frames));
        o.insert("sampling_rate".into(), json!(self.sampling_rate));
        Value::Object(o)
    }
}

// ------------------------------------------------------------------ action

fn resolve(node: &Node, expr: &str) -> Result<Vec<Target>> {
    // STRICT_EXPAND_OPEN
    node.resolve(expr, ResolveOpts { ignore_unavailable: false, allow_no_indices: true, ..Default::default() })
}

thread_local! {
    /// The running [`execute_on_cluster`]'s coordinator.
    static COORDINATOR: std::cell::RefCell<Option<crate::transform::RestCall>> = const { std::cell::RefCell::new(None) };
}

/// [`execute`] on a cluster: its searches and K/V document lookups go
/// through the coordinator (`rest`), which reads every shard of the events
/// and K/V indices from a started copy wherever it is. Index resolution and
/// mappings stay on this node's metadata mirror.
pub fn execute_on_cluster(node: &Node, r: &StackTracesRequest, rest: Option<crate::transform::RestCall>) -> Result<StackTracesResponse> {
    struct Reset(Option<crate::transform::RestCall>);
    impl Drop for Reset {
        fn drop(&mut self) {
            let previous = self.0.take();
            COORDINATOR.with(|c| *c.borrow_mut() = previous);
        }
    }
    let _reset = Reset(COORDINATOR.with(|c| std::mem::replace(&mut *c.borrow_mut(), rest)));
    execute(node, r)
}

fn coordinator() -> Option<crate::transform::RestCall> {
    COORDINATOR.with(|c| c.borrow().clone())
}

/// Percent-encode an index name for a coordinator URI.
fn uri_component(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~*,".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// The coordinator `_search` URI over the concrete indices of `targets`.
fn search_uri(targets: &[Target]) -> String {
    let mut names: Vec<&str> = targets.iter().map(|t| t.svc.name.as_str()).collect();
    names.sort();
    names.dedup();
    format!("/{}/_search", names.iter().map(|n| uri_component(n)).collect::<Vec<_>>().join(","))
}

fn search(node: &Node, targets: &[Target], body: &Value) -> Result<Value> {
    match coordinator() {
        Some(rest) if !targets.is_empty() => rest("POST", &search_uri(targets), Some(body.clone())),
        _ => crate::transform::pivot::search(node, targets, body),
    }
}

fn query_of(r: &StackTracesRequest) -> Value {
    r.query.clone().unwrap_or_else(|| json!({"match_all": {}}))
}

fn agg_value(resp: &Value, name: &str) -> Option<f64> {
    resp["aggregations"][name]["value"].as_f64()
}

/// Terms sub-aggregations for `aggregation_fields` (`SubGroupCollector`).
fn subgroup_aggs(fields: &[String]) -> Option<Value> {
    let mut agg: Option<Value> = None;
    for f in fields.iter().rev() {
        let mut t = json!({"terms": {"field": f}});
        if let Some(inner) = agg.take() {
            t["aggs"] = inner;
        }
        agg = Some(json!({ format!("{CUSTOM_EVENT_SUB_AGGREGATION_NAME}{f}"): t }));
    }
    agg
}

fn collect_subgroups(fields: &[String], legacy: bool, bucket: &Value, event: &mut TraceEvent) {
    if fields.is_empty() {
        return;
    }
    let root = event.sub_groups.get_or_insert_with(|| SubGroup::root(&fields[0], legacy));
    fn walk(fields: &[String], i: usize, agg_parent: &Value, group: &mut SubGroup) {
        if i == fields.len() {
            return;
        }
        let name = format!("{CUSTOM_EVENT_SUB_AGGREGATION_NAME}{}", fields[i]);
        for b in agg_parent[&name]["buckets"].as_array().cloned().unwrap_or_default() {
            let key = b.get("key_as_string").or_else(|| b.get("key")).map(as_text).unwrap_or_default();
            group.add_count(&key, b["doc_count"].as_i64().unwrap_or(0));
            if i + 1 < fields.len() {
                let cur = group.subgroups.get_mut(&key).unwrap();
                let child = cur.get_or_add_child(&fields[i + 1]);
                walk(fields, i + 1, &b, child);
            }
        }
    }
    walk(fields, 0, bucket, root);
}

/// Java `Random` (for the resampler).
struct JavaRandom(i64);
impl JavaRandom {
    fn new(seed: i64) -> Self {
        JavaRandom((seed ^ 0x5DEECE66D) & ((1 << 48) - 1))
    }
    fn next(&mut self, bits: u32) -> i32 {
        self.0 = (self.0.wrapping_mul(0x5DEECE66D).wrapping_add(0xB)) & ((1 << 48) - 1);
        (self.0 >> (48 - bits)) as i32
    }
    fn next_double(&mut self) -> f64 {
        ((((self.next(26) as i64) << 27) + self.next(27) as i64) as f64) * (1.0 / (1i64 << 53) as f64)
    }
}

fn request_seed(r: &StackTracesRequest) -> i64 {
    use super::java::{objects_hash, J};
    let q = r.query.as_ref().map(|q| q.to_string()).unwrap_or_else(|| "null".into());
    let idx = r.indices.as_ref().map(|v| v.iter().fold(1i32, |h, s| h.wrapping_mul(31).wrapping_add(super::java::string_hash(s)))).unwrap_or(0);
    objects_hash(&[J::S(&q), J::I(r.sample_size as i32), J::I(idx), J::S(r.stacktrace_ids_field.as_deref().unwrap_or(""))]) as i64
}

/// `Resampler`.
struct Resampler {
    rng: Option<JavaRandom>,
    p: f64,
    adjusted_rate: f64,
}

impl Resampler {
    fn new(r: &StackTracesRequest, rate: f64, total: i64) -> Self {
        let (rng, p) = if total as f64 > r.sample_size as f64 * 1.1 { (Some(JavaRandom::new(request_seed(r))), r.sample_size as f64 / total as f64) } else { (None, 1.0) };
        Resampler { rng, p, adjusted_rate: if r.adjust_sample_count { rate } else { 1.0 } }
    }
    fn adjust(&mut self, original: i32) -> i32 {
        let raw = match &mut self.rng {
            Some(rng) => (0..original).filter(|_| rng.next_double() < self.p).count() as f64,
            None => original as f64,
        };
        (raw / (self.p * self.adjusted_rate)).floor() as i32
    }
}

struct Collected {
    events: BTreeMap<String, TraceEvent>,
    /// (host id, stacktrace id, count, executable name, thread name).
    host_counts: Vec<(String, String, i64, String, String)>,
    min_time: i64,
    max_time: i64,
}

/// `searchEventGroupedByStackTrace` over a profiling-events data stream.
fn collect_profiling_events(node: &Node, r: &StackTracesRequest, exp: i32, out: &mut StackTracesResponse) -> Result<Option<Collected>> {
    out.sampling_rate = super::events_sample_rate(exp);
    let targets = match resolve(node, &super::events_index_name(exp)) {
        Ok(t) => t,
        Err(e) if super::not_found(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let fields = r.aggregation_fields_effective().unwrap_or_default();
    let mut by_trace = json!({"terms": {"field": "Stacktrace.id", "size": MAX_TRACE_EVENTS_RESULT_SIZE, "execution_hint": "map"}, "aggs": {"count": {"sum": {"field": "Stacktrace.count"}}}});
    if let Some(sub) = subgroup_aggs(&fields) {
        for (k, v) in sub.as_object().unwrap() {
            by_trace["aggs"][k] = v.clone();
        }
    }
    let body = json!({
        "size": 0, "track_total_hits": false, "query": query_of(r),
        "aggs": {
            "min_time": {"min": {"field": "@timestamp"}},
            "max_time": {"max": {"field": "@timestamp"}},
            // ES 9 groups events by (executable, thread, host, stacktrace).
            "group_by": {"terms": {"field": "host.id", "size": MAX_TRACE_EVENTS_RESULT_SIZE, "execution_hint": "map"}, "aggs": {
                "exe": {"terms": {"field": "process.executable.name", "missing": "", "size": MAX_TRACE_EVENTS_RESULT_SIZE, "execution_hint": "map"}, "aggs": {
                    "thread": {"terms": {"field": "process.thread.name", "missing": "", "size": MAX_TRACE_EVENTS_RESULT_SIZE, "execution_hint": "map"}, "aggs": {"group_by": by_trace}}
                }}
            }},
            "total_count": {"sum": {"field": "Stacktrace.count"}},
        }
    });
    let resp = search(node, &targets, &body)?;
    let total_count = super::java_round(agg_value(&resp, "total_count").unwrap_or(0.0));
    let mut resampler = Resampler::new(r, out.sampling_rate, total_count);
    let mut events: BTreeMap<String, TraceEvent> = BTreeMap::new();
    let mut host_counts = Vec::new();
    let mut total_final = 0i64;
    for hb in resp["aggregations"]["group_by"]["buckets"].as_array().cloned().unwrap_or_default() {
        let host = hb.get("key").map(as_text).unwrap_or_default();
        for eb in hb["exe"]["buckets"].as_array().cloned().unwrap_or_default() {
            let exe = eb.get("key").map(as_text).unwrap_or_default();
            for tb in eb["thread"]["buckets"].as_array().cloned().unwrap_or_default() {
                let thread = tb.get("key").map(as_text).unwrap_or_default();
                for sb in tb["group_by"]["buckets"].as_array().cloned().unwrap_or_default() {
                    let c = sb["count"]["value"].as_f64().unwrap_or(0.0) as i32;
                    let fc = resampler.adjust(c);
                    if fc <= 0 {
                        continue;
                    }
                    total_final += fc as i64;
                    let id = sb.get("key").map(as_text).unwrap_or_default();
                    host_counts.push((host.clone(), id.clone(), fc as i64, exe.clone(), thread.clone()));
                    let ev = events.entry(id).or_default();
                    ev.count += fc as i64;
                    collect_subgroups(&fields, r.is_legacy_aggregation_field(), &sb, ev);
                }
            }
        }
    }
    out.total_samples = total_final;
    Ok(Some(Collected {
        events,
        host_counts,
        min_time: super::java_round(agg_value(&resp, "min_time").unwrap_or(0.0)),
        max_time: super::java_round(agg_value(&resp, "max_time").unwrap_or(0.0)),
    }))
}

/// Mapping type of `field` in an index (walks `properties`, dotted names
/// first).
fn field_type(svc: &crate::index::IndexService, field: &str) -> Option<String> {
    let m = svc.meta().mappings.to_json();
    fn find(props: &Value, field: &str) -> Option<String> {
        if let Some(t) = props.get(field).and_then(|f| f.get("type")).and_then(Value::as_str) {
            return Some(t.to_string());
        }
        for (i, _) in field.match_indices('.') {
            if let Some(sub) = props.get(&field[..i]).and_then(|f| f.get("properties")) {
                if let Some(t) = find(sub, &field[i + 1..]) {
                    return Some(t);
                }
            }
        }
        None
    }
    find(&m["properties"], field)
}

/// `searchGenericEvents`: user indices, stacktrace ids in a (counted)
/// keyword field, counted with a random sampler.
fn collect_generic_events(node: &Node, r: &StackTracesRequest, out: &mut StackTracesResponse) -> Result<Option<Collected>> {
    let expr = r.indices.clone().unwrap_or_default().join(",");
    let targets = resolve(node, &expr)?;
    let count_body = json!({"size": 0, "track_total_hits": true, "query": query_of(r)});
    let total = search(node, &targets, &count_body)?["hits"]["total"]["value"].as_i64().unwrap_or(0);
    let requested = r.sample_size;
    out.sampling_rate = if total <= requested * 2 { 1.0 } else { requested as f64 / total as f64 };
    if total == 0 {
        return Ok(None);
    }
    let field = r.stacktrace_ids_field.clone().unwrap_or_default();
    let fields = r.aggregation_fields_effective().unwrap_or_default();
    let mut src_fields = vec![field.clone()];
    src_fields.extend(fields.iter().cloned());
    let body = json!({
        "size": total.min(1_000_000), "track_total_hits": false, "query": query_of(r), "_source": src_fields,
        "aggs": {"min_time": {"min": {"field": "@timestamp"}}, "max_time": {"max": {"field": "@timestamp"}}},
    });
    let resp = search(node, &targets, &body)?;
    let counted: HashMap<String, bool> =
        targets.iter().map(|t| (t.svc.name.clone(), field_type(&t.svc, &field).as_deref() == Some("counted_keyword"))).collect();
    let mut rng = (out.sampling_rate < 1.0).then(|| JavaRandom::new(request_seed(r)));
    // stacktrace id -> (count, per-subgroup doc values)
    let mut counts: HashMap<String, i64> = HashMap::new();
    let mut sub_docs: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    for h in resp["hits"]["hits"].as_array().cloned().unwrap_or_default() {
        if let Some(rng) = rng.as_mut() {
            if rng.next_double() >= out.sampling_rate {
                continue;
            }
        }
        let src = &h["_source"];
        let mut vals: Vec<String> = list_of(path(src, &field)).iter().map(as_text).collect();
        if !counted.get(h["_index"].as_str().unwrap_or("")).copied().unwrap_or(false) {
            vals.sort();
            vals.dedup();
        }
        for v in vals {
            *counts.entry(v.clone()).or_insert(0) += 1;
            if !fields.is_empty() {
                let keys: Vec<String> = fields.iter().map(|f| list_of(path(src, f)).first().map(as_text).unwrap_or_default()).collect();
                sub_docs.entry(v).or_default().push(keys);
            }
        }
    }
    // counted_terms bucket order: count desc, key asc; size 150000.
    let mut buckets: Vec<(String, i64)> = counts.into_iter().collect();
    buckets.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    buckets.truncate(MAX_TRACE_EVENTS_RESULT_SIZE as usize);
    let mut events: BTreeMap<String, TraceEvent> = BTreeMap::new();
    let mut host_counts = Vec::new();
    let mut total_samples = 0i64;
    for (id, c) in buckets {
        total_samples += c;
        host_counts.push(("unknown".to_string(), id.clone(), c, String::new(), String::new()));
        let ev = events.entry(id.clone()).or_default();
        ev.count += c;
        if !fields.is_empty() {
            let bucket = synth_buckets(&fields, 0, sub_docs.get(&id).map(Vec::as_slice).unwrap_or(&[]));
            collect_subgroups(&fields, r.is_legacy_aggregation_field(), &bucket, ev);
        }
    }
    out.total_samples = total_samples;
    Ok(Some(Collected {
        events,
        host_counts,
        min_time: super::java_round(agg_value(&resp, "min_time").unwrap_or(0.0)),
        max_time: super::java_round(agg_value(&resp, "max_time").unwrap_or(0.0)),
    }))
}

/// Terms-aggregation-shaped buckets (count desc, key asc, size 10) built
/// from per-document sub-group keys.
fn synth_buckets(fields: &[String], i: usize, docs: &[Vec<String>]) -> Value {
    if i == fields.len() {
        return json!({});
    }
    let mut groups: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
    for d in docs {
        if !d[i].is_empty() {
            groups.entry(d[i].clone()).or_default().push(d.clone());
        }
    }
    let mut v: Vec<(String, Vec<Vec<String>>)> = groups.into_iter().collect();
    v.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    v.truncate(10);
    let buckets: Vec<Value> = v
        .into_iter()
        .map(|(k, ds)| {
            let mut b = synth_buckets(fields, i + 1, &ds);
            b["key"] = json!(k);
            b["doc_count"] = json!(ds.len());
            b
        })
        .collect();
    json!({ format!("{CUSTOM_EVENT_SUB_AGGREGATION_NAME}{}", fields[i]): {"buckets": buckets} })
}

/// `KvIndexResolver.resolve`: the indices behind a K/V alias whose
/// validity period overlaps `[start, end]`.
fn kv_indices(node: &Node, alias: &str, start: i64, end: i64) -> Result<Vec<std::sync::Arc<crate::index::IndexService>>> {
    let targets = resolve(node, alias)?;
    let mut svcs: Vec<_> = targets.into_iter().map(|t| t.svc).collect();
    svcs.dedup_by(|a, b| a.name == b.name);
    let mut out = Vec::new();
    if svcs.len() > 1 {
        let mut with_time: Vec<_> = svcs
            .iter()
            .map(|s| {
                let t = s
                    .setting("index.lifecycle.origination_date")
                    .and_then(|v| v.parse::<i64>().ok())
                    .unwrap_or_else(|| s.setting("index.creation_date").and_then(|v| v.parse().ok()).unwrap_or(0));
                (s.clone(), t)
            })
            .collect();
        with_time.sort_by(|a, b| b.1.cmp(&a.1));
        let mut interval_end = i64::MAX;
        for (s, t) in with_time {
            if t < end && interval_end > start {
                out.push(s);
            }
            interval_end = t.saturating_add(KV_INDEX_OVERLAP_MS);
        }
    }
    if out.is_empty() {
        out = svcs;
    }
    Ok(out)
}

fn get_source(svc: &crate::index::IndexService, id: &str) -> Result<Option<Value>> {
    if let Some(rest) = coordinator() {
        let doc = rest("GET", &format!("/{}/_doc/{}", uri_component(&svc.name), uri_component(id)), None)?;
        return Ok(doc.get("_source").filter(|_| doc["found"] == true).cloned());
    }
    Ok(svc.get_routed(id, None, true)?.and_then(|g| g.source).and_then(|s| serde_json::from_str(&s).ok()))
}

/// Run the stacktraces action.
pub fn execute(node: &Node, r: &StackTracesRequest) -> Result<StackTracesResponse> {
    super::check_license(node)?;
    super::ensure_resources(node)?;
    let mut out = StackTracesResponse::default();
    let collected = if r.user_provided_indices {
        collect_generic_events(node, r, &mut out)?
    } else {
        // Pick the down-sampled events index from the count in 5pow06.
        let medium = 6;
        let exp = match resolve(node, &super::events_index_name(medium)) {
            Ok(t) => {
                let body = json!({"size": 0, "track_total_hits": true, "query": query_of(r)});
                let n = search(node, &t, &body)?["hits"]["total"]["value"].as_i64().unwrap_or(0);
                super::resampled_exponent(r.sample_size, n, medium)
            }
            Err(e) if super::not_found(&e) => 0,
            Err(e) => return Err(e),
        };
        collect_profiling_events(node, r, exp, &mut out)?
    };
    let Some(c) = collected else { return Ok(out) };
    if c.events.is_empty() {
        return Ok(out);
    }
    let (start, end) = (c.min_time, c.max_time);
    let mut events = c.events;
    // A missing K/V index ends the request with the events only.
    let st_indices = match kv_indices(node, "profiling-stacktraces", start, end) {
        Ok(v) => v,
        Err(e) if super::not_found(&e) => {
            out.events = Some(events);
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    let event_ids: Vec<String> = events.keys().cloned().collect();
    let mut st_keys = JConcurrentKeys::with_capacity(event_ids.len());
    let mut traces: HashMap<String, StackTrace> = HashMap::new();
    let mut frame_ids = JConcurrentKeys::with_capacity(event_ids.len() * 5);
    let mut exe_ids = JConcurrentKeys::with_capacity(event_ids.len());
    let mut total_frames = 0i64;
    for svc in &st_indices {
        for id in &event_ids {
            if st_keys.contains(id) {
                continue;
            }
            if let Some(src) = get_source(svc, id)? {
                let st = stack_trace_from_source(&src);
                st_keys.insert(id);
                total_frames += st.frame_ids.len() as i64;
                for f in &st.frame_ids {
                    frame_ids.insert(f);
                }
                for (i, t) in st.type_ids.iter().enumerate() {
                    if *t == NATIVE_FRAME_TYPE || *t == KERNEL_FRAME_TYPE {
                        exe_ids.insert(&st.file_ids[i]);
                    }
                }
                traces.insert(id.clone(), st);
            }
        }
    }
    // Host metadata.
    let mut unique_hosts: Vec<String> = c.host_counts.iter().map(|h| h.0.clone()).collect();
    unique_hosts.sort();
    unique_hosts.dedup();
    let mut hosts: HashMap<String, HostMetadata> = HashMap::new();
    if !unique_hosts.is_empty() {
        let targets = resolve(node, "profiling-hosts")?;
        let body = json!({
            "track_total_hits": false, "from": 0,
            "query": {"bool": {"filter": [
                {"range": {"@timestamp": {"gte": start - 6 * 3600 * 1000, "lt": end, "format": "epoch_millis"}}},
                {"terms": {"host.id": unique_hosts}},
            ]}},
            "collapse": {"field": "host.id"},
            "sort": [{"@timestamp": {"order": "desc"}}],
        });
        let resp = search(node, &targets, &body)?;
        for h in resp["hits"]["hits"].as_array().cloned().unwrap_or_default() {
            if let Some(src) = h["_source"].as_object() {
                let m = HostMetadata::from_source(src);
                hosts.insert(m.host_id.clone(), m);
            }
        }
    }
    // CO2 and costs per (host, stacktrace).
    let duration = r.requested_duration.unwrap_or_else(|| (end.div_euclid(1000) - start.div_euclid(1000)) as f64);
    let calc = Calculator::new(&hosts, duration, &r.costs);
    let mut rows: BTreeMap<(String, String, String), TraceEventRow> = BTreeMap::new();
    for (host, id, count, exe, thread) in &c.host_counts {
        let (co2, costs) = (calc.annual_co2_tons(host, *count), calc.annual_costs_usd(host, *count as f64));
        if let Some(ev) = events.get_mut(id) {
            ev.annual_co2_tons += co2;
            ev.annual_costs_usd += costs;
        }
        let row = rows.entry((id.clone(), exe.clone(), thread.clone())).or_insert_with(|| TraceEventRow { executable: exe.clone(), thread: thread.clone(), stacktrace: id.clone(), ..Default::default() });
        row.count += count;
        row.annual_co2_tons += co2;
        row.annual_costs_usd += costs;
    }
    out.trace_events = rows.into_values().collect();
    // Stack frames and executables.
    let sf_indices = kv_indices(node, "profiling-stackframes", start, end)?;
    let ex_indices = kv_indices(node, "profiling-executables", start, end)?;
    let frame_order = frame_ids.keys();
    let mut sf_keys = JConcurrentKeys::with_capacity(frame_order.len());
    let mut frames: HashMap<String, StackFrame> = HashMap::new();
    let mut inline_frames = 0i64;
    for svc in &sf_indices {
        for id in &frame_order {
            if frames.contains_key(id) {
                continue;
            }
            if let Some(src) = get_source(svc, id)? {
                let f = StackFrame::from_source(&src);
                if f.size() > 0 {
                    sf_keys.insert(id);
                    inline_frames += f.size() as i64 - 1;
                    frames.insert(id.clone(), f);
                }
            }
        }
    }
    let exe_order = exe_ids.keys();
    let mut ex_keys = JConcurrentKeys::with_capacity(exe_order.len());
    let mut exes: HashMap<String, String> = HashMap::new();
    for svc in &ex_indices {
        for id in &exe_order {
            if exes.contains_key(id) {
                continue;
            }
            if let Some(src) = get_source(svc, id)? {
                let name = {
                    let nested = src.get("Executable").and_then(|e| e.get("file")).and_then(|f| f.get("name")).and_then(Value::as_str);
                    nested.or_else(|| src.get("Executable.file.name").and_then(Value::as_str)).map(str::to_string)
                };
                ex_keys.insert(id);
                exes.insert(id.clone(), name.unwrap_or_else(|| "<missing>".into()));
            }
        }
    }
    // Merge trace events into the stack traces (`GetStackTracesResponseBuilder.build`).
    let mut st_out = Vec::new();
    for k in st_keys.keys() {
        let mut st = traces.remove(&k).unwrap();
        if let Some(ev) = events.get(&k) {
            st.count = ev.count;
            if ev.sub_groups.is_some() {
                st.sub_groups = ev.sub_groups.clone();
            }
            st.annual_co2_tons = ev.annual_co2_tons;
            st.annual_costs_usd = ev.annual_costs_usd;
        }
        st_out.push((k, st));
    }
    out.stack_traces = Some(st_out);
    out.stack_frames = Some(sf_keys.keys().into_iter().map(|k| { let f = frames.remove(&k).unwrap(); (k, f) }).collect());
    out.executables = Some(ex_keys.keys().into_iter().map(|k| { let e = exes.remove(&k).unwrap(); (k, e) }).collect());
    out.events = Some(events);
    out.total_frames = total_frames + inline_frames;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_types() {
        assert_eq!(run_length_decode("BQADAg", 8), vec![0, 0, 0, 0, 0, 2, 2, 2]);
    }

    #[test]
    fn frame_id_parts() {
        assert_eq!(file_id_from_frame_id("634wiWh6F21tPpXr0Zz3mgAAAAAAEfFi"), "634wiWh6F21tPpXr0Zz3mg");
        assert_eq!(address_from_frame_id("634wiWh6F21tPpXr0Zz3mgAAAAAAEfFi"), 1175906);
        assert_eq!(address_from_frame_id("AAAAAAAAV4sAAAAAAAAAHezOBBlhpr8q"), 1638317866);
    }

    #[test]
    fn parse_errors() {
        let e = parse_request(Some("{\"foo\":1}")).unwrap_err();
        assert_eq!(e.reason, "Unknown key for a VALUE_NUMBER in [foo].");
        let r = parse_request(Some("{\"indices\":[\"a\"],\"sample_size\":-1,\"aggregation_fields\":[\"a\",\"b\",\"c\"]}")).unwrap();
        let e = validate(&r).unwrap_err();
        assert!(e.reason.contains("1: [stacktrace_ids_field] is mandatory;2: [aggregation_fields]"), "{}", e.reason);
    }
}

#[cfg(test)]
mod coordinator_uri_tests {
    use super::*;

    #[test]
    fn index_names_are_encoded_for_the_coordinator() {
        assert_eq!(uri_component(".profiling-stackframes-v001-000001"), ".profiling-stackframes-v001-000001");
        assert_eq!(uri_component("a b/c"), "a%20b%2Fc");
    }
}
