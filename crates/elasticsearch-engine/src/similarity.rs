//! Per-index scoring similarities (ES `index.similarity.*` settings and the
//! per-field `similarity` mapping parameter), mapped onto the core
//! [`Similarity`] implementations.
//!
//! ES semantics: `index.similarity.default` replaces the BM25 default for
//! every field without an explicit `similarity`; `index.similarity.<name>`
//! defines a named one a field can refer to; the built-in names `BM25` and
//! `boolean` always resolve. `discount_overlaps` (BM25, DFR, IB, LM*) also
//! shapes the length norm written at index time, so it is exposed per
//! similarity for the mapping to apply.

use crate::error::{EsError, Result};
use crate::settings::FlatSettings;
use rustic_core::query::similarity::{
    AfterEffect, BasicModel, Bm25, BooleanSim, Distribution, DfrSimilarity, IbSimilarity, Lambda, LmDirichlet, LmJelinekMercer, Normalization, Similarity,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// One configured similarity.
#[derive(Clone, Debug)]
pub struct SimDef {
    pub sim: Arc<dyn Similarity>,
    /// Whether position-overlapping tokens are left out of the length norm.
    pub discount_overlaps: bool,
}

impl SimDef {
    fn bm25(k1: f32, b: f32, discount_overlaps: bool) -> SimDef {
        SimDef { sim: Arc::new(Bm25 { k1, b, discount_overlaps }), discount_overlaps }
    }
}

/// The similarities of one index.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Used by fields without an explicit `similarity`.
    pub default: SimDef,
    named: BTreeMap<String, SimDef>,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig { default: SimDef::bm25(1.2, 0.75, true), named: BTreeMap::new() }
    }
}

fn param<'a>(p: &'a BTreeMap<String, String>, k: &str) -> Option<&'a str> {
    p.get(k).map(String::as_str)
}

fn float(p: &BTreeMap<String, String>, k: &str, default: f32, name: &str) -> Result<f32> {
    match param(p, k) {
        None => Ok(default),
        Some(s) => s.parse::<f32>().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [index.similarity.{name}.{k}]"))),
    }
}

/// Java's `Float.toString` for the values a message shows (`2.0`, `-0.5`,
/// `NaN`, `Infinity`; scientific form from 1e7 as Java does).
pub(crate) fn java_float(x: f32) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x != 0.0 && (x.abs() >= 1e7 || x.abs() < 1e-3) {
        let s = format!("{x:e}");
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let m = if m.contains('.') { m.to_string() } else { format!("{m}.0") };
        return format!("{m}E{e}");
    }
    let s = format!("{x}");
    if s.contains('.') { s } else { format!("{s}.0") }
}

fn bool_param(p: &BTreeMap<String, String>, k: &str, default: bool) -> bool {
    param(p, k).map(|s| s == "true").unwrap_or(default)
}

fn normalization(p: &BTreeMap<String, String>, name: &str) -> Result<Normalization> {
    let c = |k: &str, d: f32| float(p, k, d, name);
    Ok(match param(p, "normalization").unwrap_or("no") {
        "no" => Normalization::None,
        "h1" => Normalization::H1 { c: c("normalization.h1.c", 1.0)? },
        "h2" => Normalization::H2 { c: c("normalization.h2.c", 1.0)? },
        "h3" => Normalization::H3 { mu: c("normalization.h3.c", 800.0)? },
        "z" => Normalization::Z { z: c("normalization.z.z", 0.3)? },
        other => return Err(EsError::illegal_argument(format!("Unsupported Normalization [{other}]"))),
    })
}

/// Build one similarity from its flattened `index.similarity.<name>.*` parameters.
fn build(name: &str, p: &BTreeMap<String, String>) -> Result<SimDef> {
    let ty = param(p, "type").ok_or_else(|| EsError::illegal_argument(format!("Similarity [{name}] must have an associated type")))?;
    let discount = bool_param(p, "discount_overlaps", true);
    Ok(match ty {
        // Ranges: [`check_ranges`] (create and update only; an existing
        // index out of range still opens, unpruned).
        "BM25" => SimDef::bm25(float(p, "k1", 1.2, name)?, float(p, "b", 0.75, name)?, discount),
        "boolean" => SimDef { sim: Arc::new(BooleanSim), discount_overlaps: true },
        "DFR" => {
            let basic = match param(p, "basic_model") {
                Some("g") => BasicModel::G,
                Some("if") => BasicModel::IF,
                Some("in") => BasicModel::In,
                Some("ine") => BasicModel::Ine,
                Some(o) => return Err(EsError::illegal_argument(format!("Unsupported BasicModel [{o}]"))),
                None => return Err(EsError::illegal_argument("Basic Model [null] is not supported".to_string())),
            };
            let after = match param(p, "after_effect") {
                Some("no") => AfterEffect::None,
                Some("b") => AfterEffect::B,
                Some("l") => AfterEffect::L,
                Some(o) => return Err(EsError::illegal_argument(format!("Unsupported AfterEffect [{o}]"))),
                None => return Err(EsError::illegal_argument("After effect [null] is not supported".to_string())),
            };
            SimDef { sim: Arc::new(DfrSimilarity::new(basic, after, normalization(p, name)?)), discount_overlaps: discount }
        }
        "IB" => {
            let dist = match param(p, "distribution") {
                Some("ll") => Distribution::LL,
                Some("spl") => Distribution::SPL,
                Some(o) => return Err(EsError::illegal_argument(format!("Unsupported Distribution [{o}]"))),
                None => return Err(EsError::illegal_argument("Distribution [null] is not supported".to_string())),
            };
            let lambda = match param(p, "lambda") {
                Some("df") => Lambda::DF,
                Some("ttf") => Lambda::TTF,
                Some(o) => return Err(EsError::illegal_argument(format!("Unsupported Lambda [{o}]"))),
                None => return Err(EsError::illegal_argument("Lambda [null] is not supported".to_string())),
            };
            SimDef { sim: Arc::new(IbSimilarity::new(dist, lambda, normalization(p, name)?)), discount_overlaps: discount }
        }
        "LMDirichlet" => SimDef { sim: Arc::new(LmDirichlet::new(float(p, "mu", 2000.0, name)?)), discount_overlaps: discount },
        "LMJelinekMercer" => SimDef { sim: Arc::new(LmJelinekMercer::new(float(p, "lambda", 0.1, name)?)), discount_overlaps: discount },
        other => return Err(EsError::illegal_argument(format!("Unknown Similarity type [{other}] for [{name}]"))),
    })
}

/// Lucene `BM25Similarity`'s `k1` / `b` ranges for the BM25 similarities of
/// `flat`: out of range, scores stop growing with tf and shrinking with
/// length (and the pruning bounds stop holding). Checked where ES builds the
/// similarity for a new or changed index (create, settings update); an
/// index already on disk with such values (an older build accepted them)
/// opens, restores and is mirrored all the same, and its scorers are never
/// pruned (`Bm25Weight::monotonic`).
pub fn check_ranges(flat: &FlatSettings) -> Result<()> {
    for (name, p) in &groups_of(flat) {
        if param(p, "type") != Some("BM25") {
            continue;
        }
        let (k1, b) = (float(p, "k1", 1.2, name)?, float(p, "b", 0.75, name)?);
        if !k1.is_finite() || k1 < 0.0 {
            return Err(EsError::illegal_argument(format!("illegal k1 value: {}, must be a non-negative finite value", java_float(k1))));
        }
        if b.is_nan() || !(0.0..=1.0).contains(&b) {
            return Err(EsError::illegal_argument(format!("illegal b value: {}, must be between 0 and 1", java_float(b))));
        }
    }
    Ok(())
}

/// [`check_ranges`] of only the similarities that `changed` keys define or
/// change (a restore's `index_settings`, a resize's or settings update's
/// body): the others of an existing index may predate the check.
pub fn check_changed_ranges<'a>(merged: &FlatSettings, changed: impl IntoIterator<Item = &'a String>) -> Result<()> {
    let names: std::collections::BTreeSet<&str> =
        changed.into_iter().filter_map(|k| k.strip_prefix("index.similarity.")).filter_map(|r| r.split_once('.').map(|(n, _)| n)).collect();
    if names.is_empty() {
        return Ok(());
    }
    let scoped: FlatSettings = merged.iter().filter(|(k, _)| k.strip_prefix("index.similarity.").and_then(|r| r.split_once('.')).is_some_and(|(n, _)| names.contains(n))).map(|(k, v)| (k.clone(), v.clone())).collect();
    check_ranges(&scoped)
}

/// `index.similarity.<name>.<key>` of `flat`, by name.
fn groups_of(flat: &FlatSettings) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut groups: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (k, v) in flat {
        let Some(rest) = k.strip_prefix("index.similarity.") else { continue };
        let Some((name, key)) = rest.split_once('.') else { continue };
        let s = match v {
            Value::String(s) => s.clone(),
            Value::Null => continue,
            other => other.to_string(),
        };
        groups.entry(name.to_string()).or_default().insert(key.to_string(), s);
    }
    groups
}

impl SimConfig {
    /// Parse `index.similarity.*` out of the flattened index settings.
    pub fn from_flat(flat: &FlatSettings) -> Result<SimConfig> {
        let groups = groups_of(flat);
        let mut cfg = SimConfig::default();
        for (name, p) in &groups {
            let def = build(name, p)?;
            if name == "default" {
                cfg.default = def;
            } else {
                cfg.named.insert(name.clone(), def);
            }
        }
        Ok(cfg)
    }

    /// Whether a field may name `sim` as its `similarity`.
    pub fn knows(&self, sim: &str) -> bool {
        matches!(sim, "BM25" | "boolean") || self.named.contains_key(sim)
    }

    /// The similarity a field with mapping parameter `similarity` uses.
    pub fn resolve(&self, sim: Option<&str>) -> SimDef {
        match sim {
            None => self.default.clone(),
            Some("BM25") => SimDef::bm25(1.2, 0.75, true),
            Some("boolean") => SimDef { sim: Arc::new(BooleanSim), discount_overlaps: true },
            Some(n) => self.named.get(n).cloned().unwrap_or_else(|| self.default.clone()),
        }
    }

    /// Per core field similarity overrides for the fields of `lookup` that
    /// differ from the default.
    pub fn field_overrides(&self, fields: impl Iterator<Item = (u16, Option<String>)>) -> HashMap<u16, Arc<dyn Similarity>> {
        let mut out = HashMap::new();
        for (id, sim) in fields {
            if sim.is_some() {
                out.insert(id, self.resolve(sim.as_deref()).sim);
            }
        }
        out
    }
}

/// The default and per-field similarities a search over `svc` scores with.
pub fn options_for(svc: &crate::index::IndexService, lookup: &crate::mapping::FieldLookup) -> (Arc<dyn Similarity>, HashMap<u16, Arc<dyn Similarity>>) {
    let cfg = svc.sims();
    let fields = lookup.fields.values().filter_map(|f| {
        // Only text fields name a scoring similarity (vector fields reuse the parameter).
        let id = f.core_id?;
        ((f.ty.is_text() || f.ty.is_keyword_like()) && f.similarity != "cosine").then(|| (id, Some(f.similarity.clone())))
    });
    let over = cfg.field_overrides(fields);
    (cfg.default.sim, over)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flat(pairs: &[(&str, &str)]) -> FlatSettings {
        pairs.iter().map(|(k, v)| (k.to_string(), json!(v))).collect()
    }

    #[test]
    fn parses_named_and_default() {
        let cfg = SimConfig::from_flat(&flat(&[
            ("index.similarity.my.type", "BM25"),
            ("index.similarity.my.k1", "2.0"),
            ("index.similarity.my.b", "0.3"),
            ("index.similarity.default.type", "boolean"),
        ]))
        .unwrap();
        assert!(cfg.knows("my") && cfg.knows("BM25") && cfg.knows("boolean") && !cfg.knows("zzz"));
        assert_eq!(cfg.resolve(Some("my")).sim.as_bm25().unwrap().k1, 2.0);
        assert_eq!(cfg.resolve(None).sim.name(), "Boolean");
        assert_eq!(cfg.resolve(Some("BM25")).sim.as_bm25().unwrap().b, 0.75);
        assert!(SimConfig::from_flat(&flat(&[("index.similarity.x.type", "nope")])).is_err());
        let nd = SimConfig::from_flat(&flat(&[("index.similarity.x.type", "BM25"), ("index.similarity.x.discount_overlaps", "false")])).unwrap();
        assert!(!nd.resolve(Some("x")).discount_overlaps);
    }

    /// Lucene `BM25Similarity`'s ranges and messages.
    #[test]
    fn bm25_k1_and_b_ranges() {
        let bm25 = |k1: &str, b: &str| {
            let f = flat(&[("index.similarity.x.type", "BM25"), ("index.similarity.x.k1", k1), ("index.similarity.x.b", b)]);
            let cfg = SimConfig::from_flat(&f).unwrap();
            check_ranges(&f).map(|_| cfg)
        };
        for (k1, b) in [("0", "0"), ("0", "1"), ("100", "0.5")] {
            bm25(k1, b).unwrap();
        }
        for (k1, b, msg) in [
            ("-1", "0.75", "illegal k1 value: -1.0, must be a non-negative finite value"),
            ("-0.5", "0.75", "illegal k1 value: -0.5, must be a non-negative finite value"),
            ("Infinity", "0.75", "illegal k1 value: Infinity, must be a non-negative finite value"),
            ("NaN", "0.75", "illegal k1 value: NaN, must be a non-negative finite value"),
            ("1.2", "1.5", "illegal b value: 1.5, must be between 0 and 1"),
            ("1.2", "-0.1", "illegal b value: -0.1, must be between 0 and 1"),
            ("1.2", "NaN", "illegal b value: NaN, must be between 0 and 1"),
        ] {
            let e = bm25(k1, b).unwrap_err();
            assert_eq!((e.status, e.kind.as_str(), e.reason.as_str()), (400, "illegal_argument_exception", msg), "{k1} {b}");
        }
        assert_eq!(java_float(2.0), "2.0");
        assert_eq!(java_float(1.0e7), "1.0E7");
        // Only the similarities a change touches are checked.
        let f = flat(&[("index.similarity.old.type", "BM25"), ("index.similarity.old.b", "2"), ("index.similarity.new.type", "BM25"), ("index.similarity.new.k1", "1")]);
        check_changed_ranges(&f, &["index.similarity.new.k1".to_string()]).unwrap();
        assert!(check_changed_ranges(&f, &["index.similarity.old.b".to_string()]).is_err());
        assert!(check_changed_ranges(&f, &["index.refresh_interval".to_string()]).is_ok());
        // An existing index out of range still builds (unpruned).
        assert!(SimConfig::from_flat(&f).is_ok());
    }
}
