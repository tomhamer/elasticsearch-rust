//! Range verification over encoded keyword doc values. Decode dictionary
//! entries once and seek document ordinals lazily, so selective AND queries
//! do not materialize the entire range result before intersecting it.
use super::{filter::FilterQuery, range::{self, Relation}};
use crate::{codec::fastfields::OwnedKeywordColumn, query::{CustomQuery, DocSet, LeafReader, Query, Scorer, SearchContext, TERMINATED}, DocId};
use std::sync::Arc;

/// Query-independent parsed bounds. Most range kinds fit in 64-bit keys;
/// only IPv6 ranges require the wider representation. Invalid dictionary
/// values remain absent, matching the generic decoder's behavior.
pub(crate) struct RangeDictionary {
    bounds: RangeBounds,
    valid: Vec<bool>,
    _memory:crate::admission::Reservation,
    /// Accepted-ordinal masks of recent queries (most recent last): a
    /// repeated range query (every Rally iteration, every search of a
    /// dashboard) skips the scan of the whole dictionary.
    masks: parking_lot::Mutex<Vec<CachedMask>>,
}
struct CachedMask {
    bounds: (u128,u128),
    relation: Relation,
    range: Arc<CachedRange>,
    _memory: crate::admission::Reservation,
}

/// One range's per-segment state, cached with its mask (see
/// [`RangeDictionary::matching_cached`]): what every repeat of the query
/// would otherwise recompute per segment, each an O(distinct values) scan.
pub(crate) struct CachedRange {
    /// Per ordinal: whether its value matches.
    pub(crate) accepted: Arc<Vec<bool>>,
    /// Some ordinal matches.
    any: bool,
    /// Docs holding a matching value (from the ordinal index), once known.
    doc_count: std::sync::OnceLock<u64>,
    /// The matching docs as a bitset, once a conjunction found collecting
    /// them cheaper than verifying (`OrdinalRange::set_lead_cost`); later
    /// scorers of the range use it whatever their lead. `None`: refused by
    /// the memory budget.
    bits: std::sync::OnceLock<Option<(Vec<u64>, crate::admission::Reservation)>>,
}
/// Masks kept per segment field.
const CACHED_MASKS: usize = 8;
enum RangeBounds { Narrow(Vec<(u64,u64)>), Wide(Vec<(u128,u128)>) }
impl RangeDictionary {
    pub(crate) fn new(column: &OwnedKeywordColumn) -> crate::error::Result<Self> {
        Self::with_budget(column,&crate::admission::shared())
    }
    fn with_budget(column:&OwnedKeywordColumn,budget:&Arc<crate::admission::Budget>)->crate::error::Result<Self> {
        // Narrow and wide buffers can coexist during widening; admit that
        // peak before constructing either and retain the final allocation.
        let mut memory=budget.reserve((column.num_terms() as u64).saturating_mul(128).saturating_add(512),"range dictionary")?;
        let mut narrow=Vec::with_capacity(column.num_terms());
        let mut wide:Option<Vec<(u128,u128)>>=None;
        let mut valid=Vec::with_capacity(column.num_terms());
        for ordinal in 0..column.num_terms() {
            let parsed=column.term(ordinal as u32).as_deref().and_then(range::decode);
            valid.push(parsed.is_some());
            let (lo,hi)=parsed.unwrap_or((0,0));
            if lo>u64::MAX as u128 || hi>u64::MAX as u128 {
                if wide.is_none() { wide=Some(narrow.drain(..).map(|(l,h)|(l as u128,h as u128)).collect()); }
            }
            if let Some(values)=wide.as_mut() {values.push((lo,hi));} else {narrow.push((lo as u64,hi as u64));}
        }
        let bounds=wide.map(RangeBounds::Wide).unwrap_or(RangeBounds::Narrow(narrow));
        let bytes=match &bounds {RangeBounds::Narrow(v)=>v.capacity().saturating_mul(16),RangeBounds::Wide(v)=>v.capacity().saturating_mul(32)}.saturating_add(valid.capacity()).saturating_add(512) as u64;
        memory.shrink(memory.bytes().saturating_sub(bytes));
        Ok(Self {bounds,valid,_memory:memory,masks:parking_lot::Mutex::new(Vec::new())})
    }
    /// [`Self::matching`], from the cache when this range was asked for
    /// recently; the mask's memory is admitted once, while it is cached.
    fn matching_cached(&self, bounds:(u128,u128), relation:Relation) -> crate::error::Result<Arc<CachedRange>> {
        {
            let mut masks=self.masks.lock();
            if let Some(i)=masks.iter().position(|m| m.bounds==bounds && m.relation==relation) {
                let entry=masks.remove(i);
                let range=entry.range.clone();
                masks.push(entry);
                return Ok(range);
            }
        }
        let memory=crate::admission::shared().reserve((self.valid.len() as u64).saturating_add(128),"range matching mask")?;
        let accepted=Arc::new(self.matching(bounds,relation));
        let any=accepted.iter().any(|&b| b);
        let range=Arc::new(CachedRange{accepted,any,doc_count:std::sync::OnceLock::new(),bits:std::sync::OnceLock::new()});
        let mut masks=self.masks.lock();
        if let Some(m)=masks.iter().find(|m| m.bounds==bounds && m.relation==relation) {
            return Ok(m.range.clone());
        }
        if masks.len()>=CACHED_MASKS { masks.remove(0); }
        masks.push(CachedMask{bounds,relation,range:range.clone(),_memory:memory});
        Ok(range)
    }
    fn matching(&self, bounds:(u128,u128), relation:Relation) -> Vec<bool> {
        match &self.bounds {
            RangeBounds::Narrow(values)=>values.iter().zip(&self.valid).map(|(&(lo,hi),&valid)|valid && range::matches((lo as u128,hi as u128),bounds,relation)).collect(),
            RangeBounds::Wide(values)=>values.iter().zip(&self.valid).map(|(&v,&valid)|valid && range::matches(v,bounds,relation)).collect(),
        }
    }
}

#[derive(Debug)]
pub struct RangeRelationQuery {
    pub field: u16,
    pub bounds: (u128, u128),
    pub relation: Relation,
}
impl RangeRelationQuery {
    pub fn into_query(self) -> Query { Query::Custom(Arc::new(self)) }
}
impl CustomQuery for RangeRelationQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        if let LeafReader::Segment(segment) = leaf {
            if let Some(column) = segment.owned_keyword_column(self.field) {
                let dictionary = match segment.range_dictionary(self.field, &column) {
                    Ok(dictionary)=>dictionary,
                    Err(error)=>{ctx.record_memory_error(error.to_string());return None;}
                };
                let range = match dictionary.matching_cached(self.bounds, self.relation) {
                    Ok(range)=>range,Err(error)=>{ctx.record_memory_error(error.to_string());return None;}
                };
                if !range.any { return None; }
                let accepted = range.accepted.clone();
                let deleted = leaf.deletions_only();
                // The docs of each value (the segment's ordinal index, built
                // once and shared): the range's exact match count, and the
                // docs to collect when that is cheaper than verifying a
                // conjunction's candidates. Only when the index is admitted.
                // (No gate on the share of distinct values accepted: a range
                // can accept many distinct values held by few docs, as
                // noaa's TRANGE [-20, -10] does.)
                let index = segment.keyword_ord_index(self.field).ok().flatten()
                    .filter(|index| index.starts.len() == accepted.len() + 1);
                // A range whose docs an earlier conjunction collected reads
                // its bitset, in any clause.
                let use_bits = range.bits.get().is_some_and(|b| b.is_some());
                return Some(Box::new(OrdinalRange {
                    column, accepted, count: leaf.num_docs(), next: 0, current: TERMINATED,
                    deleted, index, range, use_bits,
                }));
            }
        }
        // Preserve the existing path for memtables and multivalued columns.
        let (field, bounds, relation) = (self.field, self.bounds, self.relation);
        FilterQuery::constant("range", Some(Query::Exists { field }), move |leaf, doc| {
            leaf.keyword_values(field, doc).iter().filter_map(|s| range::decode(s))
                .any(|v| range::matches(v, bounds, relation))
        }).scorer(leaf, ctx)
    }
    fn is_constant_score(&self) -> bool { true }
    fn scores_uniformly(&self) -> bool { true }
    fn name(&self) -> &'static str { "range" }
}
#[cfg(test)]
thread_local! {
    /// Range doc bitsets built (`set_lead_cost`), for tests.
    static BITS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Lucene `IndexOrDocValuesQuery`: the index side (here the range's docs
/// from the ordinal index) runs when its cost is at most this many times
/// the leading clause's.
const INDEX_LEAD_FACTOR: u64 = 8;
/// The range's docs are collected only behind a lead of at most
/// `1/SELECTIVE_LEAD` of the segment's docs (see `set_lead_cost`).
const SELECTIVE_LEAD: u64 = 16;

struct OrdinalRange {
    column: OwnedKeywordColumn,
    accepted: Arc<Vec<bool>>,
    count: u32,
    next: u32,
    current: DocId,
    /// Deleted docs (nested children are live), as `undeleted_docs`.
    deleted: Option<Arc<crate::postings::BitSet>>,
    /// The segment's docs grouped by value (see `RangeRelationQuery::scorer`).
    index: Option<Arc<crate::admission::Allocation<crate::codec::fastfields::OrdIndex>>>,
    /// The range's cached per-segment state (its bitset, doc count).
    range: Arc<CachedRange>,
    /// Match through `range`'s bitset (built and present).
    use_bits: bool,
}
impl OrdinalRange {
    #[inline]
    fn live(&self, doc: DocId) -> bool {
        self.deleted.as_ref().is_none_or(|b| !b.get(doc))
    }

    /// The number of docs holding an accepted value (computed once per
    /// segment and range).
    fn index_cost(&self) -> Option<u64> {
        let index = self.index.as_ref()?;
        Some(*self.range.doc_count.get_or_init(|| self.accepted.iter().enumerate().filter(|(_, a)| **a).map(|(o, _)| (index.starts[o + 1] - index.starts[o]) as u64).sum()))
    }

    #[inline]
    fn bit(&self, doc: DocId) -> bool {
        match self.range.bits.get() {
            Some(Some((bits, _))) => bits.get(doc as usize >> 6).is_some_and(|w| (w >> (doc & 63)) & 1 == 1),
            _ => false,
        }
    }

    /// Set the bits of the accepted values' docs (`None` when the bitset's
    /// memory is not admitted: the doc-value verifier stays).
    fn collect_bits(&self) -> Option<(Vec<u64>, crate::admission::Reservation)> {
        let index = self.index.as_ref()?;
        let words = (self.count as usize).div_ceil(64);
        let memory = crate::admission::shared().reserve((words as u64).saturating_mul(8).saturating_add(64), "range field doc bitset").ok()?;
        let mut bits = vec![0u64; words];
        for (ord, _) in self.accepted.iter().enumerate().filter(|(_, a)| **a) {
            for &doc in &index.docs[index.starts[ord] as usize..index.starts[ord + 1] as usize] {
                if let Some(w) = bits.get_mut(doc as usize >> 6) {
                    *w |= 1u64 << (doc & 63);
                }
            }
        }
        Some((bits, memory))
    }
}
impl Scorer for OrdinalRange {
    fn score(&mut self) -> f32 { 1.0 }
    fn score_upper_bound(&self) -> Option<f32> { Some(1.0) }
}
impl DocSet for OrdinalRange {
    fn has_fast_exact_seek(&self) -> bool { true }
    fn set_lead_cost(&mut self, lead_cost: u32) {
        // `IndexOrDocValuesQuery`: collect the range's docs when that costs
        // at most 8x the lead, else verify each candidate's value. And only
        // when one side is small (at most 1/16 of the segment): a broad lead
        // over a broad range (noaa's US term with TRANGE 0..30) usually
        // stops early at the hit count, so collecting millions of range docs
        // first would be all extra work. The bitset is kept with the
        // segment's cached range, for every later query of it.
        if self.use_bits || self.next != 0 {
            return;
        }
        let n = self.count as u64;
        let small_lead = (lead_cost as u64).saturating_mul(SELECTIVE_LEAD) <= n;
        let Some(cost) = self.index_cost() else { return };
        if cost <= INDEX_LEAD_FACTOR.saturating_mul(lead_cost as u64) && (small_lead || cost.saturating_mul(SELECTIVE_LEAD) <= n) {
            let built = self.range.bits.get_or_init(|| {
                #[cfg(test)]
                BITS_BUILT.with(|c| c.set(c.get() + 1));
                self.collect_bits()
            });
            self.use_bits = built.is_some();
        }
    }
    fn seek_exact(&mut self, target: DocId) -> bool {
        self.next = self.next.max(target.saturating_add(1));
        let in_range = if self.use_bits { self.bit(target) } else { self.accepted.get(self.column.ord(target) as usize).copied().unwrap_or(false) };
        let matched = target < self.count
            && in_range
            && self.live(target);
        self.current = if matched { target } else { TERMINATED };
        matched
    }
    fn advance(&mut self) -> DocId {
        if self.use_bits {
            if let Some(Some((bits, _))) = self.range.bits.get() {
                let mut w = self.next as usize >> 6;
                let mut word = bits.get(w).map_or(0, |x| x & (u64::MAX << (self.next & 63)));
                loop {
                    while word == 0 {
                        w += 1;
                        if w >= bits.len() { self.next = self.count; self.current = TERMINATED; return TERMINATED; }
                        word = bits[w];
                    }
                    let doc = (w * 64 + word.trailing_zeros() as usize) as DocId;
                    word &= word - 1;
                    if doc >= self.count { self.next = self.count; self.current = TERMINATED; return TERMINATED; }
                    self.next = doc + 1;
                    if self.live(doc) { self.current = doc; return doc; }
                }
            }
        }
        // `next_value` skips a sparse column's docs without a value.
        while self.next < self.count {
            let Some((doc, ord)) = self.column.next_value(self.next) else { break };
            self.next = doc + 1;
            if self.accepted.get(ord as usize).copied().unwrap_or(false) && self.live(doc) {
                self.current = doc;
                return doc;
            }
        }
        self.next = self.count;
        self.current = TERMINATED;
        TERMINATED
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target { return self.current; }
        self.next = self.next.max(target);
        self.advance()
    }
    fn doc(&self) -> DocId { self.current }
    fn size_hint(&self) -> u32 { self.count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{codec::directory::RamDirectory, segment::SegmentWriter, types::{Document, Field, FieldType}};

    #[test]
    fn range_dictionary_admission_precedes_growth_and_follows_last_reader() {
        let mut writer=SegmentWriter::new(0);let field=Field::new(0,"range",FieldType::Keyword);
        for i in 0..100 {let mut doc=Document::new();doc.add_keyword(0,&range::encode(i,i+1));writer.add_document(&doc,std::slice::from_ref(&field));}
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();let column=segment.owned_keyword_column(0).unwrap();
        let small=crate::admission::Budget::new(64);
        assert!(matches!(RangeDictionary::with_budget(&column,&small),Err(crate::RusticError::Memory(_))));
        assert_eq!(small.used(),0);
        let budget=crate::admission::Budget::new(1<<20);
        let dictionary=Arc::new(RangeDictionary::with_budget(&column,&budget).unwrap());
        let reader=dictionary.clone();drop(dictionary);assert!(budget.used()>0);
        drop(reader);assert_eq!(budget.used(),0);
        let ctx=SearchContext::default();ctx.record_memory_error("range dictionary admission");
        assert!(matches!(ctx.execution_result(),Err(crate::RusticError::Memory(_))));
    }

    /// A repeated range reuses its accepted-ordinal mask; at most
    /// `CACHED_MASKS` masks are kept, least recently used dropped first,
    /// and a cached mask equals a fresh scan.
    #[test]
    fn matching_masks_are_cached_per_range_and_relation() {
        let mut writer=SegmentWriter::new(0);let field=Field::new(0,"range",FieldType::Keyword);
        for i in 0..200 {let mut doc=Document::new();doc.add_keyword(0,&range::encode(i,i+3));writer.add_document(&doc,std::slice::from_ref(&field));}
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();let column=segment.owned_keyword_column(0).unwrap();
        let dictionary=RangeDictionary::new(&column).unwrap();
        let a=dictionary.matching_cached((10,20),Relation::Intersects).unwrap();
        let b=dictionary.matching_cached((10,20),Relation::Intersects).unwrap();
        assert!(Arc::ptr_eq(&a,&b));
        assert_eq!(*a.accepted,dictionary.matching((10,20),Relation::Intersects));
        let c=dictionary.matching_cached((10,20),Relation::Within).unwrap();
        assert!(!Arc::ptr_eq(&a,&c));
        assert_eq!(*c.accepted,dictionary.matching((10,20),Relation::Within));
        for k in 0..CACHED_MASKS as u128 { dictionary.matching_cached((100+k,110+k),Relation::Intersects).unwrap(); }
        assert_eq!(dictionary.masks.lock().len(),CACHED_MASKS);
        let d=dictionary.matching_cached((10,20),Relation::Intersects).unwrap();
        assert!(!Arc::ptr_eq(&a,&d),"evicted after {CACHED_MASKS} newer ranges");
        assert_eq!(*a.accepted,*d.accepted);
    }

    /// NOAA `range_field_conjunction_small_range_*`: a conjunction whose
    /// selective lead costs at least 1/8 of the range's docs collects the
    /// range's docs as a bitset (Lucene `IndexOrDocValuesQuery`); a more
    /// selective lead, or a broad one, keeps verifying through doc values. Same docs either way, with deletes and missing values.
    #[test]
    fn range_conjunction_picks_bitset_or_doc_values_by_lead_cost() {
        let mut field=Field::new(0,"range",FieldType::Keyword);
        field.indexed=false;
        let mut writer=SegmentWriter::new(0);
        let n=6000u32;
        // 600 distinct values, 10 docs each (minus missing ones).
        let value=|d:u32| (d%600) as u128;
        for d in 0..n {
            let mut doc=Document::new();
            if d%13!=0 {doc.add_keyword(0,&range::encode(value(d),value(d)+2));}
            writer.add_document(&doc,std::slice::from_ref(&field));
        }
        for d in [4u32,100,2501,5999] {writer.delete_doc(d);}
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let leaf=LeafReader::Segment(&segment);
        let ctx=SearchContext::default();
        let matching=|bounds:(u128,u128),d:u32| d%13!=0 && !leaf.is_deleted(d) && range::matches((value(d),value(d)+2),bounds,Relation::Intersects);
        // (bounds, lead stride, expect a bitset built): small ranges (~65
        // docs, at most 1/16 of the segment) collect when they cost at most
        // 8x the lead: leads of 60 or 120 docs, and of 1500 or 6000 too (a
        // small range is cheap to collect whatever the lead); a lead of 6 is
        // too selective. A half-wide range (~2,800 docs) collects only
        // behind a small lead costing at least 1/8 of it: none here.
        // Each case its own range (a built bitset stays cached with the
        // segment's range); the last repeats (100, 104) behind a broad lead:
        // it reuses that range's bitset, building nothing.
        for (bounds,stride,bits) in [((100,104),100u32,true),((200,204),50,true),((300,304),1000,false),((400,404),4,true),((500,504),1,true),((0,299),1,false),((0,299),20,false),((0,299),100,false),((100,104),4,false)] {
            let driver:Vec<u32>=(0..n).step_by(stride as usize).collect();
            let want:Vec<u32>=driver.iter().copied().filter(|&d| matching(bounds,d)).collect();
            let query=RangeRelationQuery {field:0,bounds,relation:Relation::Intersects};
            let before=BITS_BUILT.with(|c| c.get());
            let children:Vec<Box<dyn Scorer>>=vec![Box::new(crate::query::ConstScorer::new(crate::query::docset::VecDocSet::new(driver.clone()),1.0)),query.scorer(&leaf,&ctx).unwrap()];
            let mut and=crate::query::boolean::IntersectionN::new(children);
            assert_eq!(BITS_BUILT.with(|c| c.get())-before, usize::from(bits), "{bounds:?} lead {}", driver.len());
            let mut got=Vec::new();
            loop {let d=and.advance();if d==TERMINATED {break} got.push(d);}
            assert_eq!(got,want,"{bounds:?} lead {}",driver.len());
        }
    }

    fn check(leaf: LeafReader<'_>, values: &[Vec<(u128,u128)>]) {
        for bounds in [(0,0),(0,u128::MAX),(1,20),(20,20),(u128::MAX-1,u128::MAX)] {
            for relation in [Relation::Intersects,Relation::Contains,Relation::Within,Relation::Disjoint] {
                let query=RangeRelationQuery {field:0,bounds,relation};
                let expected:Vec<u32>=values.iter().enumerate().filter(|(d,vs)| !leaf.is_deleted(*d as u32) && vs.iter().any(|&v|range::matches(v,bounds,relation))).map(|(d,_)|d as u32).collect();
                let ctx=SearchContext::default();
                let mut actual=Vec::new();
                if let Some(mut s)=query.scorer(&leaf,&ctx) {
                    loop { let d=s.advance();if d==TERMINATED {break} if !leaf.is_deleted(d) {assert_eq!(s.score(),1.0);actual.push(d);} }
                    assert_eq!(s.advance(),TERMINATED);
                    assert_eq!(s.seek(TERMINATED),TERMINATED);
                }
                assert_eq!(actual,expected,"{bounds:?}/{relation:?}");
                // Independent fresh seek tests at every boundary, including missing/deleted docs.
                for target in 0..=values.len() as u32 {
                    let mut s=query.scorer(&leaf,&ctx);
                    let got=s.as_mut().map(|s| {
                        let mut d=s.seek(target);
                        while d!=TERMINATED && leaf.is_deleted(d) {d=s.advance();}
                        d
                    }).unwrap_or(TERMINATED);
                    assert_eq!(got,expected.iter().copied().find(|&d|d>=target).unwrap_or(TERMINATED));
                }
            }
        }
    }
    fn fixture(multi:bool) {
        let mut field=Field::new(0,"range",FieldType::Keyword);
        field.indexed=false;
        let mut writer=SegmentWriter::new(0);
        let values:Vec<Vec<(u128,u128)>>=(0..51).map(|d| {
            if d%7==0 {vec![]} else if d%13==0 {vec![(0,u128::MAX)]} else if multi && d%3==0 {vec![(d,d+10),(u128::MAX-2,u128::MAX)]} else {vec![(d,d+10)]}
        }).collect();
        for vs in &values {
            let mut doc=Document::new();
            for &(lo,hi) in vs {doc.add_keyword(0,&range::encode(lo,hi));}
            writer.add_document(&doc,std::slice::from_ref(&field));
        }
        writer.delete_doc(1);writer.delete_doc(26);
        check(LeafReader::Memtable(&writer, None),&values);
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        check(LeafReader::Segment(&segment),&values);
    }
    #[test]
    fn single_range_ordinals_match_all_relations_with_missing_values_and_deletes() {fixture(false);}
    #[test]
    fn multivalued_range_fallback_matches_all_relations() {fixture(true);}
    #[test]
    fn exact_range_seek_handles_misses_deletes_and_forward_iteration() {
        let mut writer=SegmentWriter::new(0);
        let field=Field::new(0,"range",FieldType::Keyword);
        for d in 0..100 {
            let mut doc=Document::new();
            if d%7!=0 {doc.add_keyword(0,&range::encode(d,d+3));}
            writer.add_document(&doc,std::slice::from_ref(&field));
        }
        writer.delete_doc(16);
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let leaf=LeafReader::Segment(&segment);let ctx=SearchContext::default();
        // Exercise the full parser-style wrapper, including its scorer-to-docset adapter.
        let q=Query::ConstScore {query:Box::new(RangeRelationQuery{field:0,bounds:(15,20),relation:Relation::Intersects}.into_query()),score:1.0, point_set: false};
        let expected:Vec<u32>=(0..100).filter(|&d|d%7!=0 && d!=16 && d+3>=15 && d<=20).collect();
        for target in 0..=100 {
            let mut scorer=q.scorer(&leaf,&ctx).unwrap();
            assert!(scorer.has_fast_exact_seek());
            assert_eq!(scorer.seek_exact(target),expected.contains(&target));
            if expected.contains(&target) {assert_eq!(scorer.doc(),target);assert_eq!(scorer.score(),1.0);}
            assert_eq!(scorer.advance(),expected.iter().copied().find(|&d|d>target).unwrap_or(TERMINATED));
        }
        let mut scorer=q.scorer(&leaf,&ctx).unwrap();
        for target in [0,12,14,15,16,17,20,21,99,TERMINATED] {
            assert_eq!(scorer.seek_exact(target),expected.contains(&target));
        }
        assert_eq!(scorer.advance(),TERMINATED);
    }

    #[test]
    fn dictionary_cache_preserves_invalid_values_and_wide_query_bounds() {
        let mut writer=SegmentWriter::new(0);
        let field=Field::new(0,"range",FieldType::Keyword);
        for value in [range::encode(0,10),range::encode(u64::MAX as u128-1,u64::MAX as u128),"invalid".into(),"z".repeat(64)] {
            let mut doc=Document::new();doc.add_keyword(0,&value);writer.add_document(&doc,std::slice::from_ref(&field));
        }
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let column=segment.owned_keyword_column(0).unwrap();
        let first=segment.range_dictionary(0,&column).unwrap();
        let again=segment.range_dictionary(0,&column).unwrap();
        assert!(Arc::ptr_eq(&first,&again));
        assert!(matches!(first.bounds,RangeBounds::Narrow(_)));
        for bounds in [(0,u128::MAX),(u128::MAX,u128::MAX),(0,0),(u64::MAX as u128,u128::MAX)] {
            for relation in [Relation::Intersects,Relation::Contains,Relation::Within,Relation::Disjoint] {
                let mask=first.matching(bounds,relation);
                for (ord,&actual) in mask.iter().enumerate() {
                    let expected=column.term(ord as u32).as_deref().and_then(range::decode).is_some_and(|v|range::matches(v,bounds,relation));
                    assert_eq!(actual,expected,"{bounds:?}/{relation:?}/{ord}");
                }
            }
        }
    }
    #[test]
    fn range_top_hits_match_exhaustive_collection_at_count_thresholds() {
        use crate::query::{Searcher,SearchOptions,TrackTotalHits};
        let mut writer=SegmentWriter::new(0);
        let field=Field::new(0,"range",FieldType::Keyword);
        for d in 0..300 {let mut doc=Document::new();doc.add_keyword(0,&range::encode(d%20,d%20+10));writer.add_document(&doc,std::slice::from_ref(&field));}
        for d in (0..300).step_by(17) {writer.delete_doc(d);}
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let searcher=Searcher::new(vec![LeafReader::Segment(&segment)]);
        let fast=RangeRelationQuery {field:0,bounds:(5,15),relation:Relation::Intersects}.into_query();
        let slow=FilterQuery::constant("range",Some(Query::Exists {field:0}),|l,d|l.keyword_values(0,d).iter().filter_map(|v|range::decode(v)).any(|v|range::matches(v,(5,15),Relation::Intersects))).into_query();
        assert!(fast.scores_uniformly());assert!(!slow.scores_uniformly());
        for count in [TrackTotalHits::True,TrackTotalHits::False,TrackTotalHits::UpTo(20)] {
            for from in [0,12] {for min_score in [None,Some(2.0)] {
                let opts=SearchOptions {top_k:10,from,min_score,track_total_hits:count,..Default::default()};
                let a=searcher.search(&fast,&opts);let b=searcher.search(&slow,&opts);
                assert_eq!((a.total_hits,a.total_hits_relation),(b.total_hits,b.total_hits_relation));
                let hits=|r:crate::query::TopDocs|r.hits.into_iter().map(|h|(h.leaf,h.doc,h.score)).collect::<Vec<_>>();
                assert_eq!(hits(a),hits(b));
            }}
        }
    }
    #[test]
    fn bounded_boolean_range_hits_equal_exhaustive_across_leaves() {
        use crate::query::{BoolBuilder, Searcher, SearchOptions, TrackTotalHits};
        let mut segments=Vec::new();
        let dir=RamDirectory::new();
        for leaf in 0..2 {
            let mut writer=SegmentWriter::new(leaf);
            let mut fields=vec![Field::new(0,"range",FieldType::Keyword),Field::new(1,"term",FieldType::Keyword)];
            fields[0].indexed=false; fields[1].norms=false;
            for d in 0..5000 {
                let mut doc=Document::new();
                if d%11!=0 {doc.add_keyword(0,&range::encode(d%100,d%100+10));}
                if d%7==0 {doc.add_keyword(1,"yes");}
                writer.add_document(&doc,&fields);
            }
            for d in (0..5000).step_by(19) {writer.delete_doc(d);}
            segments.push(writer.write_to(&dir).unwrap());
        }
        let searcher=Searcher::new(segments.iter().map(LeafReader::Segment).collect());
        let make=|fast,mode,boost| {
            let r=if fast {RangeRelationQuery {field:0,bounds:(10,80),relation:Relation::Intersects}.into_query()}
                else {FilterQuery::constant("range",Some(Query::Exists {field:0}),|l,d|l.keyword_values(0,d).iter().filter_map(|v|range::decode(v)).any(|v|range::matches(v,(10,80),Relation::Intersects))).into_query()};
            // Match the engine parser, including its constant-score wrapper.
            let r=Query::ConstScore {query:Box::new(r),score:1.0, point_set: false};
            let t=Query::term(1,"yes");
            let q=match mode {
                0=>BoolBuilder::default().must(r).must(t).build(),
                1=>BoolBuilder::default().should(r).should(t).build(),
                _=>BoolBuilder::default().should(r).should(t).should(Query::term(1,"absent")).build(),
            };
            if boost {q.boost(-1.0)} else {q}
        };
        for mode in 0..3 {for boost in [false,true] {
            for track in [TrackTotalHits::False,TrackTotalHits::UpTo(0),TrackTotalHits::UpTo(50),TrackTotalHits::True] {
                for from in [0,30] {for min_score in [None,Some(1.1)] {
                    let opts=SearchOptions {top_k:20,from,track_total_hits:track,min_score,..Default::default()};
                    let a=searcher.search(&make(true,mode,boost),&opts);
                    let b=searcher.search(&make(false,mode,boost),&opts);
                    assert_eq!((a.total_hits,a.total_hits_relation),(b.total_hits,b.total_hits_relation));
                    let hits=|r:crate::query::TopDocs|r.hits.into_iter().map(|h|(h.leaf,h.doc,h.score.to_bits())).collect::<Vec<_>>();
                    assert_eq!(hits(a),hits(b),"mode={mode} boost={boost} from={from} min={min_score:?} track={track:?}");
                }}
            }
        }}
    }
    #[test]
    fn constant_score_wrappers_preserve_only_proven_safe_bounds() {
        let mut writer=SegmentWriter::new(0);
        let field=Field::new(0,"range",FieldType::Keyword);
        let mut doc=Document::new();doc.add_keyword(0,&range::encode(0,10));
        writer.add_document(&doc,std::slice::from_ref(&field));
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let leaf=LeafReader::Segment(&segment);let ctx=SearchContext::default();
        let fast=RangeRelationQuery {field:0,bounds:(0,10),relation:Relation::Intersects}.into_query();
        let general=FilterQuery::constant("general",None,|_,_|true).into_query();
        for score in [0.0,1.0,4.5,-1.0,f32::NAN,f32::INFINITY] {
            let wrapped=Query::ConstScore {query:Box::new(fast.clone()),score, point_set: false};
            let bound=wrapped.scorer(&leaf,&ctx).unwrap().score_upper_bound();
            assert_eq!(bound,if score.is_finite() && score>=0.0 {Some(score)} else {None});
            let unknown=Query::ConstScore {query:Box::new(general.clone()),score, point_set: false};
            assert!(unknown.scorer(&leaf,&ctx).unwrap().score_upper_bound().is_none());
        }
    }
    #[test]
    fn seek_never_rewinds_and_stays_exhausted() {
        let mut writer=SegmentWriter::new(0);
        let field=Field::new(0,"range",FieldType::Keyword);
        for _ in 0..10 {let mut d=Document::new();d.add_keyword(0,&range::encode(0,10));writer.add_document(&d,std::slice::from_ref(&field));}
        let dir=RamDirectory::new();let segment=writer.write_to(&dir).unwrap();
        let q=RangeRelationQuery {field:0,bounds:(0,10),relation:Relation::Intersects};
        let mut s=q.scorer(&LeafReader::Segment(&segment),&SearchContext::default()).unwrap();
        assert_eq!(s.seek(5),5);assert_eq!(s.seek(5),5);assert_eq!(s.seek(3),5);
        assert_eq!(s.advance(),6);assert_eq!(s.seek(9),9);assert_eq!(s.advance(),TERMINATED);
        assert_eq!(s.seek(0),TERMINATED);assert_eq!(s.advance(),TERMINATED);
    }
}
