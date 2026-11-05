//! The `enrich` ingest processor (ES `EnrichProcessorFactory`,
//! `MatchProcessor` for match/range policies, `GeoMatchProcessor`).

use crate::error::{EsError, Result};
use crate::ingest::{Cfg, PResult, PipelineContext, Processor};
use crate::ingest::IngestDocument;
use serde_json::Value;

pub struct EnrichP {
    policy: String,
    field: String,
    target: String,
    max_matches: usize,
    shape_relation: String,
    override_: bool,
    ignore_missing: bool,
}

/// Build from a processor config (`policy_name`, `field`, `target_field`,
/// `max_matches`, `override`, `ignore_missing`, `shape_relation`).
pub fn build(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let policy = cfg.req_str("policy_name")?;
    // ES resolves the policy's enrich index when the processor is created.
    let kind: Option<Option<String>> = super::with_current(|n| {
        if !super::has_enrich_index(n, &policy) {
            return None;
        }
        Some(super::store(n).get(&policy).map(|p| p.kind).unwrap_or_else(|| "match".into()))
    });
    let kind = match kind {
        Some(None) => return Err(EsError::illegal_argument(format!("no enrich index exists for policy with name [{policy}]"))),
        Some(Some(k)) => Some(k),
        None => None,
    };
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    let override_ = cfg.bool_or("override", true)?;
    let field = cfg.req_str("field")?;
    let target = cfg.req_str("target_field")?;
    let max_matches = cfg.opt_i64("max_matches")?.unwrap_or(1);
    if !(1..=128).contains(&max_matches) {
        return Err(cfg.err("max_matches", "should be between 1 and 128"));
    }
    // `shape_relation` is only read for geo_match policies (unknown otherwise).
    let geo = kind.as_deref().map(|k| k == "geo_match").unwrap_or(cfg.has("shape_relation"));
    let shape_relation = if geo {
        let r = cfg.str_or("shape_relation", "intersects")?.to_lowercase();
        if !matches!(r.as_str(), "intersects" | "disjoint" | "within" | "contains") {
            return Err(EsError::illegal_argument(format!("Unknown ShapeRelation type [{r}]")));
        }
        r
    } else {
        "intersects".into()
    };
    Ok(Box::new(EnrichP { policy, field, target, max_matches: max_matches as usize, shape_relation, override_, ignore_missing }))
}

impl Processor for EnrichP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let field = doc.render(&self.field);
        let value = match doc.get(&field)? {
            None => {
                if self.ignore_missing {
                    return Ok(());
                }
                return Err(doc.require(&field).err().unwrap_or_else(|| EsError::illegal_argument(format!("field [{field}] not present as part of path [{field}]"))).into());
            }
            Some(Value::Null) => return Ok(()),
            Some(v) => v.clone(),
        };
        let target = doc.render(&self.target);
        if !self.override_ && doc.has(&target) {
            return Ok(());
        }
        let hits = super::with_current(|n| -> Result<_> {
            let p = super::store(n).get(&self.policy);
            let (kind, mf) = match p {
                Some(p) => (p.kind, p.match_field),
                None => return Err(EsError::index_not_found(&super::alias_name(&self.policy))),
            };
            let q = super::lookup_query(&kind, &mf, &value, &self.shape_relation)?;
            super::lookup(n, &self.policy, q, self.max_matches)
        })
        .ok_or_else(|| EsError::new(500, "illegal_state_exception", "enrich processor is not attached to a node"))??;
        if hits.is_empty() {
            return Ok(());
        }
        let v = if self.max_matches == 1 { hits[0].clone() } else { Value::Array(hits.iter().cloned().collect()) };
        doc.set(&target, v)?;
        Ok(())
    }
}
