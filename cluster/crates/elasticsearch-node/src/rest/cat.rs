//! `_cat` table rendering, following ES's `RestTable`: `v`, `h` (names,
//! aliases, wildcards; the header shows what was asked for), `s` (sort,
//! `:asc`/`:desc`, unknown keys rejected), `help`, `format=json|yaml|smile|cbor`
//! (text otherwise), `bytes`/`time` units. Text output pads every column but
//! the last; right-aligned (numeric) columns are right-aligned in the header
//! too, as ES copies the column's attributes onto the header cell.

use crate::front::{RestReq, RestResp};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub type Row = Vec<(&'static str, Option<String>)>;

/// One column: name, aliases, description (for `help`), whether it is shown
/// by default and whether it is right-aligned (numbers, sizes).
pub struct Col {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub desc: &'static str,
    pub default: bool,
    pub right: bool,
}

/// Shorthand for column tables: `(name, aliases, desc, default, right)`.
pub type ColDef = (&'static str, &'static [&'static str], &'static str, bool, bool);

pub fn cols(defs: &[ColDef]) -> Vec<Col> {
    defs.iter().map(|&(name, aliases, desc, default, right)| Col { name, aliases, desc, default, right }).collect()
}

fn wildcard(p: &str, s: &str) -> bool {
    match p.split_once('*') {
        None => p == s,
        Some((a, rest)) => {
            if !s.starts_with(a) {
                return false;
            }
            let s = &s[a.len()..];
            if rest.is_empty() {
                return true;
            }
            (0..=s.len()).filter(|i| s.is_char_boundary(*i)).any(|i| wildcard(rest, &s[i..]))
        }
    }
}

fn find<'a>(cols: &'a [Col], key: &str) -> Option<&'a Col> {
    cols.iter().find(|c| c.name == key).or_else(|| cols.iter().filter(|c| c.aliases.contains(&key)).last())
}

/// Resolve `h=` against the available columns: (column name, header shown).
/// Mirrors `RestTable.buildDisplayHeaders`: wildcards expand to column
/// names, an alias shows as the alias (the last column declaring it wins),
/// unknown names are dropped, `ts=false` hides `epoch`/`timestamp`.
fn select(params: &HashMap<String, String>, cols: &[Col]) -> Vec<(&'static str, String)> {
    let ts = params.get("ts").map(|v| v != "false").unwrap_or(true);
    let ts_ok = |n: &str| ts || (n != "epoch" && n != "timestamp");
    match params.get("h") {
        None => cols.iter().filter(|c| c.default && ts_ok(c.name)).map(|c| (c.name, c.name.to_string())).collect(),
        Some(h) => {
            let mut wanted: Vec<String> = Vec::new();
            for want in h.split(',').map(str::trim).filter(|w| !w.is_empty()) {
                if want.contains('*') {
                    for c in cols {
                        if (wildcard(want, c.name) || c.aliases.iter().any(|a| wildcard(want, a))) && !wanted.iter().any(|w| w == c.name) {
                            wanted.push(c.name.to_string());
                        }
                    }
                } else if !wanted.iter().any(|w| w == want) {
                    wanted.push(want.to_string());
                }
            }
            let mut out = Vec::new();
            for w in wanted {
                let hit = match cols.iter().find(|c| c.name == w) {
                    Some(c) => Some(c.name),
                    None => cols.iter().filter(|c| c.aliases.contains(&w.as_str())).last().map(|c| c.name),
                };
                if let Some(name) = hit.filter(|n| ts_ok(n)) {
                    out.push((name, w));
                }
            }
            out
        }
    }
}

fn flag(params: &HashMap<String, String>, k: &str) -> bool {
    params.get(k).map(|v| v != "false").unwrap_or(false)
}

/// Sort cells: numbers numerically (including sizes/durations with units
/// via a leading number), otherwise lexically; empty cells sort last.
fn cmp_cells(x: Option<&str>, y: Option<&str>) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (x.filter(|s| !s.is_empty()), y.filter(|s| !s.is_empty())) {
        (None, None) => Equal,
        (None, Some(_)) => Greater,
        (Some(_), None) => Less,
        (Some(a), Some(b)) => match (sort_number(a), sort_number(b)) {
            (Some(p), Some(q)) => p.partial_cmp(&q).unwrap_or(Equal),
            _ => a.cmp(b),
        },
    }
}

fn sort_number(s: &str) -> Option<f64> {
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let num_end = s.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-')).unwrap_or(s.len());
    let (n, unit) = s.split_at(num_end);
    let n: f64 = n.parse().ok()?;
    let mult = match unit {
        "b" => 1.0,
        "kb" => 1024.0,
        "mb" => 1024.0 * 1024.0,
        "gb" => 1024.0 * 1024.0 * 1024.0,
        "tb" => 1024f64.powi(4),
        "pb" => 1024f64.powi(5),
        "nanos" => 1e-6,
        "micros" => 1e-3,
        "ms" => 1.0,
        "s" => 1e3,
        "m" => 6e4,
        "h" => 3.6e6,
        "d" => 8.64e7,
        "%" => 1.0,
        _ => return None,
    };
    Some(n * mult)
}

pub fn render(req: &RestReq, cols: &[Col], rows: Vec<Row>) -> RestResp {
    let params = req.params();
    if flag(&params, "help") {
        let w = cols.iter().map(|c| c.name.len()).max().unwrap_or(0);
        let wa = cols.iter().map(|c| c.aliases.join(",").len()).max().unwrap_or(0);
        let mut s = String::new();
        for c in cols {
            let desc = if c.desc.is_empty() { "not available" } else { c.desc };
            s.push_str(&format!("{:<w$} | {:<wa$} | {}\n", c.name, c.aliases.join(","), desc));
        }
        return RestResp::text(200, s);
    }
    let selected = select(&params, cols);
    let mut rows = rows;
    let get = |r: &Row, name: &str| r.iter().find(|(n, _)| *n == name).and_then(|(_, v)| v.clone());
    if let Some(sort) = params.get("s").filter(|s| !s.is_empty()) {
        let mut keys: Vec<(&'static str, bool)> = Vec::new();
        for k in sort.split(',').map(str::trim) {
            let (name, desc) = match k.rsplit_once(':') {
                Some((n, "desc")) => (n, true),
                Some((n, "asc")) => (n, false),
                Some((n, _)) => (n, false),
                None => (k, false),
            };
            match find(cols, name) {
                Some(c) => keys.push((c.name, desc)),
                None => return RestResp::error(400, "unsupported_operation_exception", &format!("Unable to sort by unknown sort key `{name}`")),
            }
        }
        rows.sort_by(|a, b| {
            for (k, desc) in &keys {
                let (x, y) = (get(a, k), get(b, k));
                let ord = cmp_cells(x.as_deref(), y.as_deref());
                let ord = if *desc { ord.reverse() } else { ord };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    let format = params.get("format").map(|f| f.to_ascii_lowercase()).or_else(|| {
        let accept = req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("accept")).map(|(_, v)| v.to_ascii_lowercase()).unwrap_or_default();
        if accept.contains("json") {
            Some("json".into())
        } else if accept.contains("yaml") {
            Some("yaml".into())
        } else {
            None
        }
    });
    let objects = || -> Vec<Value> {
        rows.iter()
            .map(|r| {
                let mut m = Map::new();
                for (name, header) in &selected {
                    m.insert(header.clone(), get(r, name).map(Value::String).unwrap_or(Value::Null));
                }
                Value::Object(m)
            })
            .collect()
    };
    match format.as_deref() {
        Some("json") | Some("smile") | Some("cbor") => RestResp::json(200, &Value::Array(objects())),
        Some("yaml") => {
            let mut r = RestResp::text(200, serde_yaml::to_string(&Value::Array(objects())).map(|s| format!("---\n{}", s.trim_start_matches("---\n"))).unwrap_or_default());
            r.replace_headers(vec![("content-type".into(), "application/yaml".into())]);
            r
        }
        _ => {
            let verbose = flag(&params, "v");
            let mut table: Vec<Vec<String>> = Vec::new();
            if verbose {
                table.push(selected.iter().map(|(_, h)| h.clone()).collect());
            }
            for r in &rows {
                table.push(selected.iter().map(|(n, _)| get(r, n).unwrap_or_default()).collect());
            }
            let ncols = selected.len();
            let widths: Vec<usize> = (0..ncols).map(|i| table.iter().map(|row| row[i].chars().count()).max().unwrap_or(0)).collect();
            let right: Vec<bool> = selected.iter().map(|(n, _)| cols.iter().find(|c| c.name == *n).map(|c| c.right).unwrap_or(false)).collect();
            let mut out = String::new();
            for row in &table {
                for (i, cell) in row.iter().enumerate() {
                    let last = i + 1 == ncols;
                    let pad = widths[i].saturating_sub(cell.chars().count());
                    if right[i] {
                        out.push_str(&" ".repeat(pad));
                        out.push_str(cell);
                    } else {
                        out.push_str(cell);
                        if !last {
                            out.push_str(&" ".repeat(pad));
                        }
                    }
                    if !last {
                        out.push(' ');
                    }
                }
                out.push('\n');
            }
            RestResp::text(200, out)
        }
    }
}

/// Byte formatting for `_cat` (`bytes=` unit or ES's human form).
pub fn bytes(v: u64, unit: Option<&str>) -> String {
    let div = |d: f64| ((v as f64) / d).floor() as u64;
    match unit {
        Some("b") => v.to_string(),
        Some("k") | Some("kb") => div(1024.0).to_string(),
        Some("m") | Some("mb") => div(1024.0 * 1024.0).to_string(),
        Some("g") | Some("gb") => div(1024f64.powi(3)).to_string(),
        Some("t") | Some("tb") => div(1024f64.powi(4)).to_string(),
        Some("p") | Some("pb") => div(1024f64.powi(5)).to_string(),
        _ => human_bytes(v),
    }
}

/// ES `ByteSizeValue.toString()`: one decimal, truncated, trailing `.0`
/// dropped (the engine's formatter).
pub fn human_bytes(v: u64) -> String {
    elasticsearch_engine::memory::byte_size(v)
}

/// ES `TimeValue.toString()` for a millisecond duration (`time=` unit aware).
pub fn time_ms(ms: u64, unit: Option<&str>) -> String {
    match unit {
        Some("nanos") => (ms * 1_000_000).to_string(),
        Some("micros") => (ms * 1000).to_string(),
        Some("ms") => ms.to_string(),
        Some("s") => (ms / 1000).to_string(),
        Some("m") => (ms / 60_000).to_string(),
        Some("h") => (ms / 3_600_000).to_string(),
        Some("d") => (ms / 86_400_000).to_string(),
        _ => {
            let (v, u) = if ms >= 86_400_000 {
                (ms as f64 / 86_400_000.0, "d")
            } else if ms >= 3_600_000 {
                (ms as f64 / 3_600_000.0, "h")
            } else if ms >= 60_000 {
                (ms as f64 / 60_000.0, "m")
            } else if ms >= 1000 {
                (ms as f64 / 1000.0, "s")
            } else if ms == 0 {
                return "0s".into();
            } else {
                return format!("{ms}ms");
            };
            let s = format!("{v:.1}");
            format!("{}{u}", s.strip_suffix(".0").unwrap_or(&s))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human() {
        assert_eq!(human_bytes(10), "10b");
        assert_eq!(human_bytes(1536), "1.5kb");
        assert_eq!(human_bytes(1024 * 1024), "1mb");
        assert_eq!(time_ms(1500, None), "1.5s");
        assert_eq!(time_ms(12, None), "12ms");
        assert_eq!(time_ms(0, None), "0s");
    }

    #[test]
    fn wildcards() {
        assert!(wildcard("disk.*", "disk.used"));
        assert!(wildcard("*", "x"));
        assert!(wildcard("*.max", "heap.max"));
        assert!(wildcard("h*p*", "heap.percent"));
        assert!(!wildcard("disk.*", "node"));
    }

    #[test]
    fn sort_numbers() {
        assert_eq!(cmp_cells(Some("2kb"), Some("10b")), std::cmp::Ordering::Greater);
        assert_eq!(cmp_cells(Some("9"), Some("10")), std::cmp::Ordering::Less);
        assert_eq!(cmp_cells(Some("b"), Some("a")), std::cmp::Ordering::Greater);
    }
}
