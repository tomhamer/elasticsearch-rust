//! Allocation settings and the small ES value formats they use.
//!
//! Cluster settings resolve the way ES resolves dynamic settings: transient
//! cluster settings, then persistent, then the node's static settings
//! (`elasticsearch.yml` / `-E`), then the built-in default. Index settings
//! are read from `IndexMetadata.settings` (flat keys). Values may be stored as
//! JSON strings, numbers or booleans; everything is read through
//! [`value_str`]. Unparseable values fall back to the default (validation is
//! the settings API's job, not allocation's).
//!
//! Formatting helpers reproduce `ByteSizeValue#toString`,
//! `TimeValue#toString` and `strict_date_optional_time` (UTC), because ES
//! embeds those renderings in decider explanations and `_cat` output.

use crate::state::{IndexMetadata, Metadata};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

// ----------------------------------------------------------- value access --

/// A setting value as a string (`"1"`, `1` and `true` all read naturally;
/// arrays are joined with commas, which is how list settings are stored flat).
pub fn value_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Array(a) => Some(a.iter().filter_map(value_str).collect::<Vec<_>>().join(",")),
        _ => None,
    }
}

pub fn index_setting(meta: &IndexMetadata, key: &str) -> Option<String> {
    meta.settings.get(key).and_then(value_str)
}

pub fn index_bool(meta: &IndexMetadata, key: &str, default: bool) -> bool {
    index_setting(meta, key).and_then(|s| parse_bool(&s)).unwrap_or(default)
}

pub fn index_i64(meta: &IndexMetadata, key: &str, default: i64) -> i64 {
    index_setting(meta, key).and_then(|s| s.trim().parse().ok()).unwrap_or(default)
}

pub fn parse_bool(s: &str) -> Option<bool> {
    match s.trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Split a comma-delimited list setting, trimming and dropping empties.
pub fn split_list(s: &str) -> Vec<String> {
    s.split(',').map(str::trim).filter(|v| !v.is_empty()).map(String::from).collect()
}

// --------------------------------------------------------------- parsing --

/// Parse an ES time value (`"1m"`, `"30s"`, `"500ms"`, `"0"`, `"-1"`) into
/// milliseconds.
pub fn parse_time_millis(s: &str) -> Option<i64> {
    let s = s.trim().to_ascii_lowercase();
    if s == "0" || s == "-1" {
        return s.parse().ok();
    }
    let units: [(&str, f64); 7] = [("nanos", 1e-6), ("micros", 1e-3), ("ms", 1.0), ("s", 1e3), ("m", 60e3), ("h", 3600e3), ("d", 86400e3)];
    for (suffix, mult) in units {
        if let Some(num) = s.strip_suffix(suffix) {
            // "ms" also ends with "s": only accept a purely numeric prefix.
            if let Ok(n) = num.trim().parse::<f64>() {
                return Some((n * mult).round() as i64);
            }
        }
    }
    None
}

/// Parse an ES byte size (`"500mb"`, `"10gb"`, `"1024b"`, `"1.5k"`).
pub fn parse_bytes(s: &str) -> Option<u64> {
    let s = s.trim().to_ascii_lowercase();
    let units: [(&str, f64); 11] = [
        ("pb", 1125899906842624.0),
        ("tb", 1099511627776.0),
        ("gb", 1073741824.0),
        ("mb", 1048576.0),
        ("kb", 1024.0),
        ("p", 1125899906842624.0),
        ("t", 1099511627776.0),
        ("g", 1073741824.0),
        ("m", 1048576.0),
        ("k", 1024.0),
        ("b", 1.0),
    ];
    for (suffix, mult) in units {
        if let Some(num) = s.strip_suffix(suffix) {
            if let Ok(n) = num.trim().parse::<f64>() {
                if n >= 0.0 {
                    return Some((n * mult) as u64);
                }
            }
            return None;
        }
    }
    if s == "0" { Some(0) } else { None }
}

/// `RatioValue.parseRatioValue`: `"85%"` or `"0.85"` → 85.0 (percent).
pub fn parse_ratio_percent(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Some(p) = s.strip_suffix('%') {
        let v: f64 = p.trim().parse().ok()?;
        return (0.0..=100.0).contains(&v).then_some(v);
    }
    let v: f64 = s.parse().ok()?;
    if (0.0..=1.0).contains(&v) { Some(v * 100.0) } else { None }
}

// ------------------------------------------------------------ formatting --

/// `Strings.format1Decimals`: one decimal, truncated, `.0` dropped.
pub fn format1(value: f64, suffix: &str) -> String {
    let p = format!("{value}");
    match p.find('.') {
        None => format!("{p}{suffix}"),
        Some(ix) => {
            let frac = p[ix + 1..].chars().next().unwrap_or('0');
            if frac == '0' { format!("{}{suffix}", &p[..ix]) } else { format!("{}.{frac}{suffix}", &p[..ix]) }
        }
    }
}

/// `ByteSizeValue#toString` (`"7.3gb"`, `"200gb"`, `"512b"`).
pub fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [(&str, f64); 5] = [("pb", 1125899906842624.0), ("tb", 1099511627776.0), ("gb", 1073741824.0), ("mb", 1048576.0), ("kb", 1024.0)];
    let b = bytes as f64;
    for (suffix, size) in UNITS {
        if b >= size {
            return format1(b / size, suffix);
        }
    }
    format!("{bytes}b")
}

/// Render bytes in an explicit `_cat` `bytes=` unit (integer, truncated).
pub fn fmt_bytes_in(bytes: u64, unit: &str) -> Option<String> {
    let div: u64 = match unit {
        "b" => 1,
        "k" | "kb" => 1 << 10,
        "m" | "mb" => 1 << 20,
        "g" | "gb" => 1 << 30,
        "t" | "tb" => 1 << 40,
        "p" | "pb" => 1 << 50,
        _ => return None,
    };
    Some((bytes / div).to_string())
}

/// `TimeValue#toString` for a millisecond value (`"1m"`, `"59.9s"`, `"0s"`).
pub fn fmt_time_millis(ms: i64) -> String {
    if ms < 0 {
        return "-1".into();
    }
    let m = ms as f64;
    const UNITS: [(&str, f64); 4] = [("d", 86400e3), ("h", 3600e3), ("m", 60e3), ("s", 1e3)];
    for (suffix, size) in UNITS {
        if m >= size {
            return format1(m / size, suffix);
        }
    }
    if ms == 0 { "0s".into() } else { format!("{ms}ms") }
}

/// Days since the epoch → (year, month, day), proleptic Gregorian.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `strict_date_optional_time` in UTC with millis: `2017-01-04T18:08:16.600Z`.
pub fn fmt_date_millis(ms: i64) -> String {
    let days = ms.div_euclid(86_400_000);
    let rem = ms.rem_euclid(86_400_000);
    let (y, mo, d) = civil_from_days(days);
    let (h, mi, s, milli) = (rem / 3_600_000, rem / 60_000 % 60, rem / 1000 % 60, rem % 1000);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{milli:03}Z")
}

/// `HH:mm:ss` UTC, the `_cat/health` `timestamp` column.
pub fn fmt_hms(ms: i64) -> String {
    let rem = ms.rem_euclid(86_400_000);
    format!("{:02}:{:02}:{:02}", rem / 3_600_000, rem / 60_000 % 60, rem / 1000 % 60)
}

// ------------------------------------------------------- setting enums ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnableAllocation {
    All,
    Primaries,
    NewPrimaries,
    None,
}

impl EnableAllocation {
    pub fn parse(s: &str) -> Option<EnableAllocation> {
        match s.trim().to_ascii_lowercase().as_str() {
            "all" => Some(EnableAllocation::All),
            "primaries" => Some(EnableAllocation::Primaries),
            "new_primaries" => Some(EnableAllocation::NewPrimaries),
            "none" => Some(EnableAllocation::None),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            EnableAllocation::All => "all",
            EnableAllocation::Primaries => "primaries",
            EnableAllocation::NewPrimaries => "new_primaries",
            EnableAllocation::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnableRebalance {
    All,
    Primaries,
    Replicas,
    None,
}

impl EnableRebalance {
    pub fn parse(s: &str) -> Option<EnableRebalance> {
        match s.trim().to_ascii_lowercase().as_str() {
            "all" => Some(EnableRebalance::All),
            "primaries" => Some(EnableRebalance::Primaries),
            "replicas" => Some(EnableRebalance::Replicas),
            "none" => Some(EnableRebalance::None),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            EnableRebalance::All => "all",
            EnableRebalance::Primaries => "primaries",
            EnableRebalance::Replicas => "replicas",
            EnableRebalance::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllowRebalance {
    Always,
    IndicesPrimariesActive,
    IndicesAllActive,
}

impl AllowRebalance {
    pub fn parse(s: &str) -> Option<AllowRebalance> {
        match s.trim().to_ascii_lowercase().as_str() {
            "always" => Some(AllowRebalance::Always),
            "indices_primaries_active" => Some(AllowRebalance::IndicesPrimariesActive),
            "indices_all_active" => Some(AllowRebalance::IndicesAllActive),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            AllowRebalance::Always => "always",
            AllowRebalance::IndicesPrimariesActive => "indices_primaries_active",
            AllowRebalance::IndicesAllActive => "indices_all_active",
        }
    }
}

// ------------------------------------------------------------ watermarks --

#[derive(Clone, Debug, PartialEq)]
pub enum WatermarkValue {
    /// Maximum used percentage (`"85%"`, `"0.85"`).
    UsedPercent(f64),
    /// Minimum free bytes (`"500mb"`).
    FreeBytes(u64),
}

/// One disk watermark (`low`, `high` or `flood_stage`) with its optional
/// `max_headroom` (ES 8.5+: defaults to 200gb/150gb/100gb, but only when the
/// watermark itself is left at its default percentage).
#[derive(Clone, Debug, PartialEq)]
pub struct Watermark {
    pub key: &'static str,
    pub raw: String,
    pub value: WatermarkValue,
    pub headroom: Option<u64>,
    pub headroom_raw: String,
}

impl Watermark {
    /// Free bytes a node of `total` bytes must keep, and whether the
    /// headroom (not the percentage) is the binding constraint.
    pub fn required_free(&self, total: u64) -> (u64, bool) {
        match self.value {
            WatermarkValue::FreeBytes(b) => (b, false),
            WatermarkValue::UsedPercent(p) => {
                let by_pct = (total as f64 * (100.0 - p) / 100.0).max(0.0) as u64;
                match self.headroom {
                    Some(h) if h < by_pct => (h, true),
                    _ => (by_pct, false),
                }
            }
        }
    }

    /// `cluster.routing.allocation.disk.watermark.low=85%` (or the headroom
    /// setting when that is what binds) as ES prints it in explanations.
    pub fn describe(&self, headroom_binding: bool) -> String {
        if headroom_binding { format!("{}.max_headroom={}", self.key, self.headroom_raw) } else { format!("{}={}", self.key, self.raw) }
    }
}

// ------------------------------------------------------- node filters -----

/// `DiscoveryNodeFilters`: attribute → accepted values (wildcards allowed).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeFilters(pub BTreeMap<String, Vec<String>>);

impl NodeFilters {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Build from flat settings under `prefix` (e.g. `index.routing.allocation.require.`).
    pub fn from_entries<'a, I: IntoIterator<Item = (&'a String, String)>>(prefix: &str, entries: I) -> NodeFilters {
        let mut out = BTreeMap::new();
        for (k, v) in entries {
            if let Some(attr) = k.strip_prefix(prefix) {
                // `_tier_preference` belongs to the data tier decider.
                if attr.is_empty() || attr == "_tier_preference" {
                    continue;
                }
                let vals = split_list(&v);
                if !vals.is_empty() {
                    out.insert(attr.to_string(), vals);
                }
            }
        }
        NodeFilters(out)
    }

    /// `DiscoveryNodeFilters#toString`: `_name:"a OR b",zone:"z1"`.
    pub fn describe(&self) -> String {
        self.0.iter().map(|(k, v)| format!("{k}:\"{}\"", v.join(" OR "))).collect::<Vec<_>>().join(",")
    }
}

// --------------------------------------------------------- the settings --

/// Effective allocation settings for one allocation round.
#[derive(Clone, Debug)]
pub struct AllocSettings {
    pub enable: EnableAllocation,
    pub rebalance_enable: EnableRebalance,
    pub allow_rebalance: AllowRebalance,
    pub cluster_concurrent_rebalance: i64,
    pub node_concurrent_incoming: i64,
    pub node_concurrent_outgoing: i64,
    pub node_initial_primaries_recoveries: i64,
    pub balance_shard: f32,
    pub balance_index: f32,
    pub balance_threshold: f32,
    pub cluster_total_shards_per_node: i64,
    pub awareness_attributes: Vec<String>,
    pub forced_awareness: BTreeMap<String, Vec<String>>,
    pub require: NodeFilters,
    pub include: NodeFilters,
    pub exclude: NodeFilters,
    pub disk_threshold_enabled: bool,
    pub low: Watermark,
    pub high: Watermark,
    pub flood: Watermark,
}

/// Layered lookup: transient > persistent > node settings.
pub struct SettingsView<'a> {
    pub metadata: &'a Metadata,
    pub node: &'a Map<String, Value>,
}

impl SettingsView<'_> {
    pub fn get(&self, key: &str) -> Option<String> {
        self.metadata.setting(key).or_else(|| self.node.get(key)).and_then(value_str)
    }

    pub fn is_set(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Every key under `prefix`, highest-precedence value winning.
    pub fn with_prefix(&self, prefix: &str) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for layer in [self.node, &self.metadata.persistent_settings, &self.metadata.transient_settings] {
            for (k, v) in layer {
                if k.starts_with(prefix) {
                    match value_str(v) {
                        Some(s) => {
                            out.insert(k.clone(), s);
                        }
                        // an explicit null in a higher layer resets the key
                        None => {
                            out.remove(k);
                        }
                    }
                }
            }
        }
        out
    }

    fn parse<T>(&self, key: &str, f: impl Fn(&str) -> Option<T>, default: T) -> T {
        self.get(key).and_then(|s| f(&s)).unwrap_or(default)
    }
}

pub const LOW_KEY: &str = "cluster.routing.allocation.disk.watermark.low";
pub const HIGH_KEY: &str = "cluster.routing.allocation.disk.watermark.high";
pub const FLOOD_KEY: &str = "cluster.routing.allocation.disk.watermark.flood_stage";

impl AllocSettings {
    pub fn resolve(view: &SettingsView) -> AllocSettings {
        let recoveries = view.parse("cluster.routing.allocation.node_concurrent_recoveries", |s| s.trim().parse().ok(), 2i64);
        let awareness_attributes = view.get("cluster.routing.allocation.awareness.attributes").map(|s| split_list(&s)).unwrap_or_default();
        let mut forced_awareness = BTreeMap::new();
        for (k, v) in view.with_prefix("cluster.routing.allocation.awareness.force.") {
            if let Some(attr) = k.strip_prefix("cluster.routing.allocation.awareness.force.").and_then(|r| r.strip_suffix(".values")) {
                let vals = split_list(&v);
                if !vals.is_empty() {
                    forced_awareness.insert(attr.to_string(), vals);
                }
            }
        }
        let filters = |kind: &str| {
            let prefix = format!("cluster.routing.allocation.{kind}.");
            let entries = view.with_prefix(&prefix);
            NodeFilters::from_entries(&prefix, entries.iter().map(|(k, v)| (k, v.clone())))
        };
        AllocSettings {
            enable: view.parse("cluster.routing.allocation.enable", EnableAllocation::parse, EnableAllocation::All),
            rebalance_enable: view.parse("cluster.routing.rebalance.enable", EnableRebalance::parse, EnableRebalance::All),
            allow_rebalance: view.parse("cluster.routing.allocation.allow_rebalance", AllowRebalance::parse, AllowRebalance::IndicesAllActive),
            cluster_concurrent_rebalance: view.parse("cluster.routing.allocation.cluster_concurrent_rebalance", |s| s.trim().parse().ok(), 2),
            node_concurrent_incoming: view.parse("cluster.routing.allocation.node_concurrent_incoming_recoveries", |s| s.trim().parse().ok(), recoveries),
            node_concurrent_outgoing: view.parse("cluster.routing.allocation.node_concurrent_outgoing_recoveries", |s| s.trim().parse().ok(), recoveries),
            node_initial_primaries_recoveries: view.parse("cluster.routing.allocation.node_initial_primaries_recoveries", |s| s.trim().parse().ok(), 4),
            balance_shard: view.parse("cluster.routing.allocation.balance.shard", |s| s.trim().parse().ok(), 0.45),
            balance_index: view.parse("cluster.routing.allocation.balance.index", |s| s.trim().parse().ok(), 0.55),
            balance_threshold: view.parse("cluster.routing.allocation.balance.threshold", |s| s.trim().parse().ok(), 1.0),
            cluster_total_shards_per_node: view.parse("cluster.routing.allocation.total_shards_per_node", |s| s.trim().parse().ok(), -1),
            awareness_attributes,
            forced_awareness,
            require: filters("require"),
            include: filters("include"),
            exclude: filters("exclude"),
            disk_threshold_enabled: view.parse("cluster.routing.allocation.disk.threshold_enabled", parse_bool, true),
            low: watermark(view, LOW_KEY, "85%", "200gb"),
            high: watermark(view, HIGH_KEY, "90%", "150gb"),
            flood: watermark(view, FLOOD_KEY, "95%", "100gb"),
        }
    }
}

fn watermark(view: &SettingsView, key: &'static str, default: &str, default_headroom: &str) -> Watermark {
    let explicit = view.get(key);
    let raw = explicit.clone().unwrap_or_else(|| default.to_string());
    let value = parse_ratio_percent(&raw)
        .map(WatermarkValue::UsedPercent)
        .or_else(|| parse_bytes(&raw).map(WatermarkValue::FreeBytes))
        .unwrap_or_else(|| WatermarkValue::UsedPercent(parse_ratio_percent(default).unwrap_or(90.0)));
    let headroom_key = format!("{key}.max_headroom");
    let (headroom, headroom_raw) = match view.get(&headroom_key) {
        Some(h) if h.trim() == "-1" => (None, h),
        Some(h) => (parse_bytes(&h), h),
        None if explicit.is_none() => (parse_bytes(default_headroom), default_headroom.to_string()),
        None => (None, "-1".into()),
    };
    Watermark { key, raw, value, headroom, headroom_raw }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(fmt_bytes(0), "0b");
        assert_eq!(fmt_bytes(1023), "1023b");
        assert_eq!(fmt_bytes(1024), "1kb");
        assert_eq!(fmt_bytes(1536), "1.5kb");
        assert_eq!(fmt_bytes(7_900_000_000), "7.3gb");
        assert_eq!(fmt_bytes(200 * (1 << 30)), "200gb");
        assert_eq!(fmt_time_millis(60_000), "1m");
        assert_eq!(fmt_time_millis(59_999), "59.9s");
        assert_eq!(fmt_time_millis(0), "0s");
        assert_eq!(fmt_time_millis(250), "250ms");
        assert_eq!(fmt_date_millis(1483553296600), "2017-01-04T18:08:16.600Z");
        assert_eq!(fmt_date_millis(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(fmt_hms(1475871424000), "20:17:04");
        assert_eq!(format1(86.25, "%"), "86.2%");
    }

    #[test]
    fn parsing() {
        assert_eq!(parse_time_millis("1m"), Some(60_000));
        assert_eq!(parse_time_millis("500ms"), Some(500));
        assert_eq!(parse_time_millis("2s"), Some(2000));
        assert_eq!(parse_time_millis("0"), Some(0));
        assert_eq!(parse_time_millis("abc"), None);
        assert_eq!(parse_bytes("500mb"), Some(500 << 20));
        assert_eq!(parse_bytes("1gb"), Some(1 << 30));
        assert_eq!(parse_bytes("10b"), Some(10));
        assert_eq!(parse_bytes("85%"), None);
        assert_eq!(parse_ratio_percent("85%"), Some(85.0));
        assert_eq!(parse_ratio_percent("0.5"), Some(50.0));
        assert_eq!(parse_ratio_percent("500mb"), None);
    }

    #[test]
    fn watermark_headroom_defaults() {
        let meta = Metadata::default();
        let node = Map::new();
        let s = AllocSettings::resolve(&SettingsView { metadata: &meta, node: &node });
        // small disk: percentage binds
        assert_eq!(s.low.required_free(100 << 30), (15 << 30, false));
        // huge disk: 200gb headroom binds
        let tb10 = 10u64 << 40;
        assert_eq!(s.low.required_free(tb10), (200 << 30, true));
        assert_eq!(s.low.describe(true), "cluster.routing.allocation.disk.watermark.low.max_headroom=200gb");
        // explicit percentage disables the default headroom
        let mut meta = Metadata::default();
        meta.persistent_settings.insert(LOW_KEY.into(), "80%".into());
        let s = AllocSettings::resolve(&SettingsView { metadata: &meta, node: &node });
        assert_eq!(s.low.required_free(tb10).1, false);
        assert_eq!(s.low.describe(false), "cluster.routing.allocation.disk.watermark.low=80%");
    }

    #[test]
    fn precedence_and_filters() {
        let mut meta = Metadata::default();
        let mut node = Map::new();
        node.insert("cluster.routing.allocation.enable".into(), "none".into());
        meta.persistent_settings.insert("cluster.routing.allocation.enable".into(), "primaries".into());
        meta.persistent_settings.insert("cluster.routing.allocation.exclude._name".into(), "a, b".into());
        meta.transient_settings.insert("cluster.routing.allocation.awareness.force.zone.values".into(), "z1,z2".into());
        let s = AllocSettings::resolve(&SettingsView { metadata: &meta, node: &node });
        assert_eq!(s.enable, EnableAllocation::Primaries);
        assert_eq!(s.exclude.describe(), "_name:\"a OR b\"");
        assert_eq!(s.forced_awareness.get("zone").unwrap(), &vec!["z1".to_string(), "z2".to_string()]);
        meta.transient_settings.insert("cluster.routing.allocation.enable".into(), "all".into());
        let s = AllocSettings::resolve(&SettingsView { metadata: &meta, node: &node });
        assert_eq!(s.enable, EnableAllocation::All);
    }
}
