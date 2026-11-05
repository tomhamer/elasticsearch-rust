//! `pinned` query: the listed documents first (in listed order), then the
//! organic matches. ES scores the i-th of n pinned documents with the float
//! whose bits are `0x7f000000 + (n - i) + 1`.

use super::dsl::{parse_query, QueryContext};
use crate::error::{EsError, Result};
use crate::mapping::ID_FIELD;
use rustic_core::fieldtypes::filter::FilterQuery;
use rustic_core::query::{CustomQuery, Query};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub fn pinned(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let o = body.as_object().ok_or_else(|| EsError::parsing("[pinned] query malformed, no start_object after query name"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "ids" | "docs" | "organic" | "boost" | "_name") {
            return Err(EsError::parsing(format!("[pinned] unknown field [{k}]")));
        }
    }
    let organic = o.get("organic").ok_or_else(|| EsError::parsing("Required [organic]"))?;
    // (id, index restriction) in pinned order; a `docs` item with `_index`
    // pins only in that index (ES `Item`, `_index` optional since 8.11).
    let items: Vec<(String, Option<String>)> = match (o.get("ids"), o.get("docs")) {
        (Some(_), Some(_)) => return Err(EsError::parsing("[pinned] failed to parse field [organic]")),
        (Some(Value::Array(a)), None) => a.iter().map(|x| (super::dsl::scalar_string(x), None)).collect(),
        (None, Some(Value::Array(a))) => a.iter().filter_map(|d| d.get("_id").map(|id| (super::dsl::scalar_string(id), d.get("_index").and_then(Value::as_str).map(str::to_string)))).collect(),
        _ => return Err(EsError::parsing("Failed to build [pinned] after last required field arrived")),
    };
    let organic = parse_query(organic, ctx)?;
    if items.is_empty() {
        return Ok(organic);
    }
    // Scores follow the position in the whole list, whichever index each item names.
    let n = items.len() as u32;
    let here = |ix: &str| ix == ctx.index_name || ctx.aliases.iter().any(|a| a == ix) || crate::mapping::wildcard_match(ix, ctx.index_name);
    let mut pos: HashMap<String, f32> = HashMap::new();
    for (i, (id, ix)) in items.iter().enumerate() {
        if ix.as_deref().is_none_or(here) {
            pos.entry(id.clone()).or_insert_with(|| f32::from_bits(0x7f00_0000 + (n - i as u32) + 1));
        }
    }
    if pos.is_empty() {
        return Ok(organic);
    }
    let ids: Vec<String> = pos.keys().cloned().collect();
    let candidates = Query::TermSet { field: ID_FIELD, terms: ids.iter().map(|s| s.clone().into_bytes()).collect(), boost: 1.0 };
    let pos = Arc::new(pos);
    let check_pos = pos.clone();
    let inner = FilterQuery::new(
        "pinned",
        Some(candidates),
        Arc::new(move |leaf, d| leaf.keyword_value(ID_FIELD, d).and_then(|id| check_pos.get(&id).copied())),
    );
    let pinned_q = Query::Custom(Arc::new(PinnedDocs { inner, pos }));
    // `DisjunctionMaxQuery([DisjunctionMaxQuery(pins), CappedScoreQuery(organic)], 0)`:
    // a pinned doc scores its pin, whatever its organic score.
    let q = Query::DisMax { queries: vec![pinned_q, organic], tie_breaker: 0.0 };
    let b = o.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    Ok(if (b - 1.0).abs() > f32::EPSILON { Query::Boost { query: Box::new(q), boost: b } } else { q })
}

/// The pinned documents of a `pinned` query (Lucene: a `DisjunctionMaxQuery`
/// of `BoostQuery(ConstantScoreQuery(_id:<id>), pinScore)` per document).
#[derive(Debug)]
struct PinnedDocs {
    inner: FilterQuery,
    pos: Arc<HashMap<String, f32>>,
}

impl CustomQuery for PinnedDocs {
    fn scorer(&self, leaf: &rustic_core::query::LeafReader<'_>, ctx: &rustic_core::query::SearchContext) -> Option<Box<dyn rustic_core::query::Scorer>> {
        self.inner.scorer(leaf, ctx)
    }
    fn name(&self) -> &'static str {
        "pinned"
    }
    fn explain(&self, leaf: &rustic_core::query::LeafReader<'_>, _ctx: &rustic_core::query::SearchContext, doc: rustic_core::types::DocId, score: f32) -> rustic_core::query::Explanation {
        use rustic_core::query::Explanation;
        let id = leaf.keyword_value(ID_FIELD, doc).unwrap_or_default();
        let pin = self.pos.get(&id).copied().unwrap_or(score);
        let clause = Explanation::leaf(pin, format!("ConstantScore(_id:({}))^{}", lucene_term_text(&encode_uid(&id)), crate::ingest::jtext::java_float(pin)));
        Explanation::new(score, "max of:", vec![clause])
    }
}

/// ES `Uid.encodeId`: the indexed form of an `_id`.
pub fn encode_uid(id: &str) -> Vec<u8> {
    const UTF8: u8 = 0xff;
    const NUMERIC: u8 = 0xfe;
    const BASE64_ESCAPE: u8 = 0xfd;
    if !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()) {
        let d: Vec<u8> = id.bytes().map(|b| b - b'0').collect();
        let mut out = vec![NUMERIC];
        for pair in d.chunks(2) {
            out.push((pair[0] << 4) | pair.get(1).copied().unwrap_or(0x0f));
        }
        return out;
    }
    let base64_ok = {
        let last = id.chars().last().unwrap_or('A');
        let tail_ok = match id.len() & 3 {
            0 => true,
            1 => false,
            2 => matches!(last, 'A' | 'Q' | 'g' | 'w'),
            _ => matches!(last, 'A' | 'E' | 'I' | 'M' | 'Q' | 'U' | 'Y' | 'c' | 'g' | 'k' | 'o' | 's' | 'w' | '0' | '4' | '8'),
        };
        tail_ok && !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    };
    if base64_ok {
        use base64::Engine;
        if let Ok(mut b) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(id) {
            if b.first().is_some_and(|&f| f >= BASE64_ESCAPE) {
                b.insert(0, BASE64_ESCAPE);
            }
            return b;
        }
    }
    let mut out = vec![UTF8];
    out.extend_from_slice(id.as_bytes());
    out
}

/// Lucene `Term.toString(BytesRef)`: the UTF-8 text, or `[hex bytes]` when
/// the bytes are not valid UTF-8.
pub fn lucene_term_text(b: &[u8]) -> String {
    match std::str::from_utf8(b) {
        Ok(s) => s.to_string(),
        Err(_) => format!("[{}]", b.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(" ")),
    }
}
