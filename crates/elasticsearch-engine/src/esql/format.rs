//! Response formats shared by ES|QL and SQL: json (row/columnar), csv,
//! tsv, txt, yaml, cbor.

use super::exec::Table;
use std::collections::HashMap;
use super::Val;
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

/// `columns` + `values` (row-major or columnar), honouring
/// `drop_null_columns` (dropped columns are listed in `all_columns`).
pub fn columns_values(t: &Table, columnar: bool, drop_null_columns: bool, out: &mut Map<String, Value>) {
    columns_values_all(t, None, columnar, drop_null_columns, out)
}

/// [`columns_values`] of a result that may already hold only its non-null
/// columns (`all`: [`super::exec::EsqlResult::all_columns`]).
pub fn columns_values_all(t: &Table, all: Option<&[super::exec::Column]>, columnar: bool, drop_null_columns: bool, out: &mut Map<String, Value>) {
    let keep = columns_only_all(t, all, drop_null_columns, out);
    let types: Vec<&str> = t.cols.iter().map(|c| c.ty.as_str()).collect();
    let values: Vec<Value> = if columnar {
        keep.iter().map(|i| Value::Array(t.rows.iter().map(|r| cell_json(&r[*i], types[*i])).collect())).collect()
    } else {
        t.rows.iter().map(|r| Value::Array(keep.iter().map(|i| cell_json(&r[*i], types[*i])).collect())).collect()
    };
    out.insert("values".into(), Value::Array(values));
}

/// A `keep` entry ([`columns_only`]) of a column the table does not hold
/// (all null, see [`columns_expanded`]): written as nulls.
pub const NULL_COLUMN: usize = usize::MAX;

/// The response columns of a table holding only the non-null columns of
/// `all` (in `all`'s order) when nulls are not dropped: every column of
/// `all` is listed as a `column`, and the returned `keep` maps each to its
/// table column or [`NULL_COLUMN`].
pub fn columns_expanded(t: &Table, all: &[super::exec::Column], out: &mut Map<String, Value>) -> Vec<usize> {
    let at: HashMap<&str, usize> = t.cols.iter().enumerate().map(|(i, c)| (c.name.as_str(), i)).collect();
    out.insert("columns".into(), Value::Array(all.iter().map(column_json).collect()));
    all.iter().map(|c| at.get(c.name.as_str()).copied().unwrap_or(NULL_COLUMN)).collect()
}

/// The full table of a table holding only the non-null columns of `all`.
pub fn expanded_table(t: &Table, all: &[super::exec::Column]) -> Table {
    let at: HashMap<&str, usize> = t.cols.iter().enumerate().map(|(i, c)| (c.name.as_str(), i)).collect();
    let map: Vec<Option<usize>> = all.iter().map(|c| at.get(c.name.as_str()).copied()).collect();
    Table {
        cols: all.to_vec(),
        rows: t.rows.iter().map(|r| map.iter().map(|m| m.map_or(Val::Null, |i| r[i].clone())).collect()).collect(),
    }
}

/// The `all_columns` / `columns` part of [`columns_values`]: the indices of
/// the columns the values keep.
/// [`columns_only`] of a result that may already hold only its non-null
/// columns: `all` lists every column (`all_columns`), and the table's
/// columns are those `drop_null_columns` keeps.
pub fn columns_only_all(t: &Table, all: Option<&[super::exec::Column]>, drop_null_columns: bool, out: &mut Map<String, Value>) -> Vec<usize> {
    match all {
        Some(all) if drop_null_columns => {
            out.insert("all_columns".into(), Value::Array(all.iter().map(column_json).collect()));
            out.insert("columns".into(), Value::Array(t.cols.iter().map(column_json).collect()));
            (0..t.cols.len()).collect()
        }
        _ => columns_only(t, drop_null_columns, out),
    }
}

pub fn columns_only(t: &Table, drop_null_columns: bool, out: &mut Map<String, Value>) -> Vec<usize> {
    let keep: Vec<usize> = if drop_null_columns { non_null_columns(t) } else { (0..t.cols.len()).collect() };
    if drop_null_columns {
        out.insert("all_columns".into(), Value::Array(t.cols.iter().map(column_json).collect()));
    }
    out.insert("columns".into(), Value::Array(keep.iter().map(|i| column_json(&t.cols[*i])).collect()));
    keep
}

/// The columns with a non-null value in some row, ascending: one row-major
/// pass (a column-by-column scan strides across every row of a wide table
/// for each all-null column), stopping once every column has a value.
pub fn non_null_columns(t: &Table) -> Vec<usize> {
    let mut seen = vec![false; t.cols.len()];
    let mut left = t.cols.len();
    for row in &t.rows {
        if left == 0 { break; }
        for (slot, value) in seen.iter_mut().zip(row) {
            if !*slot && !value.is_null() {
                *slot = true;
                left -= 1;
            }
        }
    }
    (0..t.cols.len()).filter(|&i| seen[i]).collect()
}

/// A response object whose `values_key` entry is the table's values,
/// serialized straight from the rows: the same JSON as the tree
/// [`columns_values`] builds (one `Value` per cell), without building it.
/// `meta` holds a placeholder at `values_key`, so the key order is the tree's.
pub struct StreamedResponse<'a> {
    pub meta: &'a Map<String, Value>,
    pub values_key: &'a str,
    pub table: &'a Table,
    /// Columns kept ([`columns_only`]).
    pub keep: &'a [usize],
    pub columnar: bool,
}

impl serde::Serialize for StreamedResponse<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(self.meta.len()))?;
        for (k, v) in self.meta {
            if k == self.values_key {
                map.serialize_entry(k, &StreamedValues { table: self.table, keep: self.keep, columnar: self.columnar })?;
            } else {
                map.serialize_entry(k, v)?;
            }
        }
        map.end()
    }
}

impl StreamedResponse<'_> {
    /// The bytes `serde_json::to_writer` writes for `self`, the values by a
    /// null-run aware encoder: a wide ES|QL table is mostly nulls (the Rally
    /// `esql` track's 10,000 rows x ~2,500 columns are ~125 MB of JSON),
    /// written here as copies of pre-rendered `,null` runs instead of one
    /// serializer call per cell. Compact JSON only (no pretty printing).
    pub fn write_json(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        w.write_all(b"{")?;
        for (n, (k, v)) in self.meta.iter().enumerate() {
            if n > 0 { w.write_all(b",")?; }
            serde_json::to_writer(&mut *w, k).map_err(std::io::Error::other)?;
            w.write_all(b":")?;
            if k == self.values_key {
                write_values(self.table, self.keep, self.columnar, w)?;
            } else {
                serde_json::to_writer(&mut *w, v).map_err(std::io::Error::other)?;
            }
        }
        w.write_all(b"}")
    }
}

/// `,null` repeated: a run of null cells is copied from it.
static NULL_RUN: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(|| b",null".repeat(1024));

/// Append the JSON array of `cells` (value, column type) to `buf`, exactly
/// as the serializer writes it: runs of nulls copied, other cells
/// serialized ([`Cell`]).
fn push_cells<'a>(buf: &mut Vec<u8>, cells: impl Iterator<Item = (&'a Val, &'a str)>) -> std::io::Result<()> {
    fn nulls(buf: &mut Vec<u8>, first: &mut bool, mut k: usize) {
        if k == 0 { return; }
        if *first {
            buf.extend_from_slice(b"null");
            *first = false;
            k -= 1;
        }
        while k > 0 {
            let take = k.min(1024);
            buf.extend_from_slice(&NULL_RUN[..take * 5]);
            k -= take;
        }
    }
    buf.push(b'[');
    let mut first = true;
    let mut run = 0usize;
    for (v, ty) in cells {
        if matches!(v, Val::Null) {
            run += 1;
            continue;
        }
        nulls(buf, &mut first, run);
        run = 0;
        if !first { buf.push(b','); }
        first = false;
        serde_json::to_writer(&mut *buf, &Cell(v, ty)).map_err(std::io::Error::other)?;
    }
    nulls(buf, &mut first, run);
    buf.push(b']');
    Ok(())
}

/// The values array of [`StreamedResponse::write_json`] (row-major, or one
/// array per kept column when `columnar`).
fn write_values(t: &Table, keep: &[usize], columnar: bool, w: &mut dyn std::io::Write) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
    buf.push(b'[');
    let flush = |buf: &mut Vec<u8>, w: &mut dyn std::io::Write| -> std::io::Result<()> {
        if buf.len() >= 1 << 16 {
            w.write_all(buf)?;
            buf.clear();
        }
        Ok(())
    };
    static NULL: Val = Val::Null;
    if columnar {
        for (n, &i) in keep.iter().enumerate() {
            if n > 0 { buf.push(b','); }
            if i == NULL_COLUMN {
                push_cells(&mut buf, t.rows.iter().map(|_| (&NULL, "")))?;
            } else {
                let ty = t.cols[i].ty.as_str();
                push_cells(&mut buf, t.rows.iter().map(|r| (&r[i], ty)))?;
            }
            flush(&mut buf, w)?;
        }
    } else {
        for (n, r) in t.rows.iter().enumerate() {
            if n > 0 { buf.push(b','); }
            push_cells(&mut buf, keep.iter().map(|&i| if i == NULL_COLUMN { (&NULL, "") } else { (&r[i], t.cols[i].ty.as_str()) }))?;
            flush(&mut buf, w)?;
        }
    }
    buf.push(b']');
    w.write_all(&buf)
}

struct StreamedValues<'a> {
    table: &'a Table,
    keep: &'a [usize],
    columnar: bool,
}

impl serde::Serialize for StreamedValues<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        struct Row<'a>(&'a [Val], &'a [usize], &'a [super::exec::Column]);
        impl serde::Serialize for Row<'_> {
            fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
                let mut seq = s.serialize_seq(Some(self.1.len()))?;
                for &i in self.1 {
                    if i == NULL_COLUMN {
                        seq.serialize_element(&Cell(&Val::Null, ""))?;
                    } else {
                        seq.serialize_element(&Cell(&self.0[i], &self.2[i].ty))?;
                    }
                }
                seq.end()
            }
        }
        struct Col<'a>(&'a [Vec<Val>], usize, &'a str);
        impl serde::Serialize for Col<'_> {
            fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
                let mut seq = s.serialize_seq(Some(self.0.len()))?;
                for r in self.0 {
                    if self.1 == NULL_COLUMN {
                        seq.serialize_element(&Cell(&Val::Null, ""))?;
                    } else {
                        seq.serialize_element(&Cell(&r[self.1], self.2))?;
                    }
                }
                seq.end()
            }
        }
        let t = self.table;
        if self.columnar {
            let mut seq = s.serialize_seq(Some(self.keep.len()))?;
            for &i in self.keep {
                let ty = if i == NULL_COLUMN { "" } else { t.cols[i].ty.as_str() };
                seq.serialize_element(&Col(&t.rows, i, ty))?;
            }
            seq.end()
        } else {
            let mut seq = s.serialize_seq(Some(t.rows.len()))?;
            for r in &t.rows {
                seq.serialize_element(&Row(r, self.keep, &t.cols))?;
            }
            seq.end()
        }
    }
}

/// [`cell_json`] serialized: the common cells directly, the others through
/// their JSON value.
struct Cell<'a>(&'a Val, &'a str);

impl serde::Serialize for Cell<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match (self.0, self.1) {
            (Val::Null, _) => s.serialize_unit(),
            (Val::Str(v), ty) if ty != "_source" && ty != "flattened" => s.serialize_str(v),
            (Val::Int(v) | Val::Long(v), _) => s.serialize_i64(*v),
            (Val::Bool(v), _) => s.serialize_bool(*v),
            (v, ty) => serde::Serialize::serialize(&cell_json(v, ty), s),
        }
    }
}

/// `{"name", "type"}` of a column; union-type columns (`unsupported:a,b`)
/// list their `original_types`.
pub fn column_json(c: &super::exec::Column) -> Value {
    match c.ty.strip_prefix("unsupported") {
        Some(rest) => {
            let types: Vec<&str> = rest.strip_prefix(':').map(|r| r.split(',').collect()).unwrap_or_default();
            let mut o = json!({"name": c.name, "type": "unsupported"});
            if !types.is_empty() {
                o["original_types"] = json!(types);
                if let Some(s) = suggested_cast(&types) {
                    o["suggested_cast"] = json!(s);
                }
            }
            o
        }
        None => json!({"name": c.name, "type": c.ty}),
    }
}

/// ES|QL type names (`DataType.typeName()`).
const ESQL_TYPES: &[&str] = &[
    "boolean", "integer", "long", "unsigned_long", "double", "float", "half_float", "scaled_float", "byte", "short", "keyword", "text", "datetime", "date_nanos", "ip", "version",
    "geo_point", "geo_shape", "cartesian_point", "cartesian_shape", "counter_long", "counter_integer", "counter_double", "aggregate_metric_double", "dense_vector",
    "date_range", "exponential_histogram", "tdigest", "histogram", "geohash", "geotile", "geohex",
];

/// `DataType.suggestedCast` of a union-type column's original types.
fn suggested_cast(types: &[&str]) -> Option<&'static str> {
    if types.is_empty() || types.iter().any(|t| !ESQL_TYPES.contains(t)) {
        return None;
    }
    let numeric = |t: &str| matches!(t, "integer" | "long" | "unsigned_long" | "double" | "float" | "half_float" | "scaled_float" | "byte" | "short");
    if types.len() == 2 && types.contains(&"date_nanos") && types.contains(&"datetime") {
        return Some("date_nanos");
    }
    if types.contains(&"aggregate_metric_double") && types.iter().all(|t| numeric(t) || *t == "aggregate_metric_double") {
        return Some("aggregate_metric_double");
    }
    Some("keyword")
}

/// JSON of a cell of a column of ES|QL type `ty`: `_source` is the document
/// object, `unsigned_long` values beyond `i64` print as exact integers.
pub fn cell_json(v: &Val, ty: &str) -> Value {
    match (ty, v) {
        ("_source" | "flattened", Val::Str(s)) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        ("unsigned_long", Val::Double(d)) if *d >= 0.0 && d.fract() == 0.0 && *d < 1.9e19 => json!(*d as u64),
        (_, Val::Multi(m)) if ty == "unsigned_long" || ty == "_source" => Value::Array(m.iter().map(|x| cell_json(x, ty)).collect()),
        _ => v.to_json(),
    }
}

fn cell(v: &Val) -> String {
    match v {
        Val::Null => String::new(),
        Val::Multi(m) => m.iter().map(cell).collect::<Vec<_>>().join(","),
        Val::Double(d) => super::fmt_double(*d),
        other => other.to_text(),
    }
}

fn txt_cell(v: &Val) -> String {
    match v {
        Val::Null => "null".into(),
        Val::Multi(m) => format!("[{}]", m.iter().map(txt_cell).collect::<Vec<_>>().join(", ")),
        Val::Double(d) => super::fmt_double(*d),
        other => other.to_text(),
    }
}

fn csv_escape(s: &str, delim: char) -> String {
    if s.contains(delim) || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Content type for a text/binary format; `None` for json.
pub fn content_type(format: &str) -> Option<&'static str> {
    Some(match format {
        "csv" => "text/csv; charset=utf-8; header=present",
        "tsv" => "text/tab-separated-values; charset=utf-8",
        "txt" => "text/plain; charset=utf-8",
        "yaml" => "application/yaml",
        "cbor" => "application/cbor",
        "smile" => "application/smile",
        "arrow" => "application/vnd.apache.arrow.stream",
        _ => return None,
    })
}

/// Resolve the response format from `?format=` or the Accept header.
pub fn pick_format(param: Option<&str>, accept: Option<&str>) -> Result<String> {
    if let Some(f) = param {
        let f = f.to_ascii_lowercase();
        if !matches!(f.as_str(), "json" | "csv" | "tsv" | "txt" | "yaml" | "cbor" | "smile" | "arrow") {
            return Err(EsError::illegal_argument(format!("Invalid format [{f}], expected one of [json, csv, tsv, txt, yaml, cbor, smile, arrow]")));
        }
        return Ok(f);
    }
    let a = accept.unwrap_or("").to_ascii_lowercase();
    Ok(if a.contains("text/csv") {
        "csv"
    } else if a.contains("text/tab-separated-values") {
        "tsv"
    } else if a.contains("text/plain") {
        "txt"
    } else if a.contains("yaml") {
        "yaml"
    } else if a.contains("cbor") {
        "cbor"
    } else if a.contains("smile") {
        "smile"
    } else {
        "json"
    }
    .to_string())
}

/// Text table formats (csv/tsv/txt) of a table.
pub fn text_table(t: &Table, format: &str, delimiter: Option<&str>, header: bool, drop_null_columns: bool) -> Result<String> {
    let keep: Vec<usize> = if drop_null_columns { non_null_columns(t) } else { (0..t.cols.len()).collect() };
    let mut out = String::new();
    match format {
        "csv" => {
            let d = delimiter.and_then(|d| d.chars().next()).unwrap_or(',');
            if header {
                out.push_str(&keep.iter().map(|i| csv_escape(&t.cols[*i].name, d)).collect::<Vec<_>>().join(&d.to_string()));
                out.push_str("\r\n");
            }
            for r in &t.rows {
                out.push_str(&keep.iter().map(|i| csv_escape(&cell(&r[*i]), d)).collect::<Vec<_>>().join(&d.to_string()));
                out.push_str("\r\n");
            }
        }
        "tsv" => {
            let esc = |s: &str| s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n").replace('\r', "\\r");
            if header {
                out.push_str(&keep.iter().map(|i| esc(&t.cols[*i].name)).collect::<Vec<_>>().join("\t"));
                out.push('\n');
            }
            for r in &t.rows {
                out.push_str(&keep.iter().map(|i| esc(&cell(&r[*i]))).collect::<Vec<_>>().join("\t"));
                out.push('\n');
            }
        }
        "txt" => {
            let cells: Vec<Vec<String>> = t.rows.iter().map(|r| keep.iter().map(|i| txt_cell(&r[*i])).collect()).collect();
            let widths: Vec<usize> = keep
                .iter()
                .enumerate()
                .map(|(k, i)| {
                    let w = cells.iter().map(|r| r[k].chars().count()).max().unwrap_or(0);
                    w.max(t.cols[*i].name.chars().count()).max(15)
                })
                .collect();
            if header {
                let mut line = Vec::new();
                for (k, i) in keep.iter().enumerate() {
                    let name = &t.cols[*i].name;
                    let pad = widths[k] - name.chars().count();
                    let left = pad / 2;
                    line.push(format!("{}{}{}", " ".repeat(left), name, " ".repeat(pad - left)));
                }
                out.push_str(&line.join("|"));
                out.push('\n');
                out.push_str(&widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("+"));
                out.push('\n');
            }
            for r in &cells {
                let line: Vec<String> = r.iter().enumerate().map(|(k, c)| format!("{}{}", c, " ".repeat(widths[k] - c.chars().count()))).collect();
                out.push_str(&line.join("|"));
                out.push('\n');
            }
        }
        other => return Err(EsError::illegal_argument(format!("format [{other}] is not a text format"))),
    }
    Ok(out)
}

/// Encode a JSON value as YAML.
pub fn to_yaml(v: &Value) -> String {
    let mut out = String::from("---\n");
    yaml_into(v, 0, &mut out, false);
    out
}

fn yaml_scalar(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        other => other.to_string(),
    }
}

fn yaml_into(v: &Value, indent: usize, out: &mut String, inline_first: bool) {
    let pad = " ".repeat(indent);
    match v {
        Value::Object(m) if !m.is_empty() => {
            for (i, (k, x)) in m.iter().enumerate() {
                if !(inline_first && i == 0) {
                    out.push_str(&pad);
                }
                out.push_str(k);
                out.push(':');
                match x {
                    Value::Object(mm) if !mm.is_empty() => {
                        out.push('\n');
                        yaml_into(x, indent + 2, out, false);
                    }
                    Value::Array(a) if !a.is_empty() => {
                        out.push('\n');
                        yaml_into(x, indent, out, false);
                    }
                    Value::Object(_) => out.push_str(" {}\n"),
                    Value::Array(_) => out.push_str(" []\n"),
                    s => {
                        out.push(' ');
                        out.push_str(&yaml_scalar(s));
                        out.push('\n');
                    }
                }
            }
        }
        Value::Array(a) if !a.is_empty() => {
            for x in a {
                out.push_str(&pad);
                out.push_str("- ");
                match x {
                    Value::Object(mm) if !mm.is_empty() => yaml_into(x, indent + 2, out, true),
                    Value::Array(aa) if !aa.is_empty() => {
                        out.push('\n');
                        yaml_into(x, indent + 2, out, false);
                    }
                    s => {
                        out.push_str(&yaml_scalar(s));
                        out.push('\n');
                    }
                }
            }
        }
        s => {
            out.push_str(&pad);
            out.push_str(&yaml_scalar(s));
            out.push('\n');
        }
    }
}

/// Encode a JSON value as CBOR.
pub fn to_cbor(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    cbor_into(v, &mut out);
    out
}

fn cbor_head(major: u8, n: u64, out: &mut Vec<u8>) {
    let m = major << 5;
    if n < 24 {
        out.push(m | n as u8);
    } else if n <= u8::MAX as u64 {
        out.push(m | 24);
        out.push(n as u8);
    } else if n <= u16::MAX as u64 {
        out.push(m | 25);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n <= u32::MAX as u64 {
        out.push(m | 26);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(m | 27);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

fn cbor_into(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Null => out.push(0xf6),
        Value::Bool(b) => out.push(if *b { 0xf5 } else { 0xf4 }),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                cbor_head(0, u, out);
            } else if let Some(i) = n.as_i64() {
                cbor_head(1, (-1 - i) as u64, out);
            } else {
                out.push(0xfb);
                out.extend_from_slice(&n.as_f64().unwrap_or(0.0).to_be_bytes());
            }
        }
        Value::String(s) => {
            cbor_head(3, s.len() as u64, out);
            out.extend_from_slice(s.as_bytes());
        }
        Value::Array(a) => {
            cbor_head(4, a.len() as u64, out);
            for x in a {
                cbor_into(x, out);
            }
        }
        Value::Object(m) => {
            cbor_head(5, m.len() as u64, out);
            for (k, x) in m {
                cbor_head(3, k.len() as u64, out);
                out.extend_from_slice(k.as_bytes());
                cbor_into(x, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::exec::{Column, Table};
    use super::{columns_only, columns_values, StreamedResponse, Val};
    use serde_json::{json, Map, Value};

    /// The streamed body is byte for byte the tree's: every cell kind,
    /// row-major and columnar, `drop_null_columns`, SQL's `rows` key, and
    /// the meta keys around the values.
    #[test]
    fn streamed_response_matches_value_tree() {
        let col = |name: &str, ty: &str| Column { name: name.into(), ty: ty.into() };
        let table = Table {
            cols: vec![col("k", "keyword"), col("l", "long"), col("i", "integer"), col("d", "double"), col("b", "boolean"), col("t", "datetime"),
                col("u", "unsigned_long"), col("s", "_source"), col("f", "flattened"), col("n", "keyword"), col("x", "unsupported:keyword,long"),
                col("dur", "time_duration"), col("m", "keyword")],
            rows: vec![
                vec![Val::Str("a\"\u{1}é".into()), Val::Long(-3), Val::Int(7), Val::Double(1.5), Val::Bool(true), Val::Date(86_400_000),
                    Val::Double(1.8e19), Val::Str(r#"{"z":1,"a":[1,2]}"#.into()), Val::Str(r#"{"q":"r"}"#.into()), Val::Null, Val::Null,
                    Val::Dur(1500), Val::Multi(vec![Val::Str("x".into()), Val::Str("y".into())])],
                vec![Val::Null, Val::Null, Val::Null, Val::Double(f64::NAN), Val::Bool(false), Val::Null,
                    Val::Double(12.0), Val::Str("not json".into()), Val::Null, Val::Null, Val::Null,
                    Val::Null, Val::Multi(vec![Val::Long(1), Val::Double(0.1)])],
                vec![Val::Str(String::new()), Val::Long(i64::MAX), Val::Int(0), Val::Double(-0.0), Val::Null, Val::Date(-1),
                    Val::Multi(vec![Val::Double(1.0e19), Val::Double(2.0)]), Val::Multi(vec![Val::Str("{}".into())]), Val::Null, Val::Null, Val::Null,
                    Val::Null, Val::Null],
            ],
        };
        let empty = Table { cols: table.cols.clone(), rows: Vec::new() };
        for t in [&table, &empty] {
            for columnar in [false, true] {
                for drop_null in [false, true] {
                    for key in ["values", "rows"] {
                        let mut meta = Map::new();
                        meta.insert("took".into(), json!(3));
                        meta.insert("is_partial".into(), json!(false));
                        let mut tree = meta.clone();
                        columns_values(t, columnar, drop_null, &mut tree);
                        if key != "values" && !columnar {
                            let v = tree.remove("values").unwrap();
                            tree.insert(key.into(), v);
                        }
                        tree.insert("documents_found".into(), json!(9));
                        let values_key = if key != "values" && !columnar { key } else { "values" };
                        let keep = columns_only(t, drop_null, &mut meta);
                        meta.insert(values_key.into(), Value::Null);
                        meta.insert("documents_found".into(), json!(9));
                        let response = StreamedResponse { meta: &meta, values_key, table: t, keep: &keep, columnar };
                        let streamed = serde_json::to_string(&response).unwrap();
                        assert_eq!(streamed, serde_json::to_string(&Value::Object(tree)).unwrap(), "columnar={columnar} drop_null={drop_null} key={key}");
                        let mut written = Vec::new();
                        response.write_json(&mut written).unwrap();
                        assert_eq!(String::from_utf8(written).unwrap(), streamed, "write_json columnar={columnar} drop_null={drop_null} key={key}");
                    }
                }
            }
        }
    }

    /// The null-run writer on a wide, mostly-null table (runs longer than
    /// the pre-rendered buffer, rows of nulls only, nulls first and last)
    /// writes what the serializer writes, row-major and columnar.
    #[test]
    fn null_runs_write_like_the_serializer() {
        let width = 2600;
        let cols: Vec<Column> = (0..width).map(|i| Column { name: format!("f{i}"), ty: if i % 7 == 0 { "long".into() } else { "keyword".into() } }).collect();
        let rows: Vec<Vec<Val>> = (0..40).map(|r| (0..width).map(|c| match (r % 5, c) {
            (0, _) => Val::Null,
            (1, c) if c == 0 || c == width - 1 => Val::Str(format!("edge \"{r}\"")),
            (2, c) if c % 1100 == 3 => if c % 7 == 0 { Val::Long(c as i64) } else { Val::Str(format!("v{c}")) },
            (3, c) if c == 1500 => Val::Multi(vec![Val::Str("a".into()), Val::Null, Val::Long(2)]),
            (4, c) if c % 2 == 0 => Val::Str("x".into()),
            _ => Val::Null,
        }).collect()).collect();
        let table = Table { cols, rows };
        for columnar in [false, true] {
            for drop_null in [false, true] {
                let mut meta = Map::new();
                let keep = columns_only(&table, drop_null, &mut meta);
                meta.insert("values".into(), Value::Null);
                let response = StreamedResponse { meta: &meta, values_key: "values", table: &table, keep: &keep, columnar };
                let mut written = Vec::new();
                response.write_json(&mut written).unwrap();
                assert!(written == serde_json::to_vec(&response).unwrap(), "columnar={columnar} drop_null={drop_null}");
            }
        }
    }
}
