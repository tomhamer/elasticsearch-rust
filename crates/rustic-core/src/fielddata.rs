//! Uninverted fielddata (Lucene / ES `fielddata: true` on `text` fields).
//!
//! Text fields have no doc-values column; sorting and aggregating on them
//! "uninverts" the inverted index: walk every term of the field once and
//! record, per doc, the ordinals of the terms it holds. The result is
//! immutable for a sealed segment, so it is built lazily on first use and
//! cached on the segment (dropped with it), like ES's per-segment fielddata
//! cache.
//!
//! Layout: the field's terms in sorted order (ordinal = rank) packed into
//! one byte buffer, plus a CSR doc → ordinals table. A doc's ordinals are
//! ascending, so its smallest term is the first and its largest the last.

use crate::types::DocId;

/// One field's uninverted terms over one leaf.
#[derive(Debug)]
pub struct Uninverted {
    /// Term bytes, concatenated in ordinal (= sorted) order.
    term_bytes: Vec<u8>,
    /// `term_offsets[o]..term_offsets[o + 1]` is term `o` (`num_terms + 1` entries).
    term_offsets: Vec<u32>,
    /// `ords[doc_starts[d]..doc_starts[d + 1]]` are doc `d`'s ordinals,
    /// ascending (`num_docs + 1` entries).
    doc_starts: Vec<u32>,
    ords: Vec<u32>,
    /// Node-wide fielddata accounting, released with the table.
    charge: crate::accounting::Charge,
    memory:crate::admission::Reservation,
}

impl Uninverted {
    /// Docs covered (the leaf's doc count when built).
    pub fn num_docs(&self) -> u32 {
        self.doc_starts.len().saturating_sub(1) as u32
    }

    /// Distinct terms.
    pub fn num_terms(&self) -> u32 {
        self.term_offsets.len().saturating_sub(1) as u32
    }

    /// Bytes of term `ord`.
    #[inline]
    pub fn term(&self, ord: u32) -> &[u8] {
        let o = ord as usize;
        &self.term_bytes[self.term_offsets[o] as usize..self.term_offsets[o + 1] as usize]
    }

    /// Term `ord` as an owned string (terms of text fields are UTF-8).
    #[inline]
    pub fn term_string(&self, ord: u32) -> String {
        String::from_utf8_lossy(self.term(ord)).into_owned()
    }

    /// Ordinals of `doc`'s terms, ascending (empty: the doc has none).
    #[inline]
    pub fn ords(&self, doc: DocId) -> &[u32] {
        let d = doc as usize;
        match (self.doc_starts.get(d), self.doc_starts.get(d + 1)) {
            (Some(&a), Some(&b)) => &self.ords[a as usize..b as usize],
            _ => &[],
        }
    }

    /// Ordinal of `doc`'s smallest term.
    #[inline]
    pub fn min_ord(&self, doc: DocId) -> Option<u32> {
        self.ords(doc).first().copied()
    }

    /// Ordinal of `doc`'s largest term.
    #[inline]
    pub fn max_ord(&self, doc: DocId) -> Option<u32> {
        self.ords(doc).last().copied()
    }

    /// Approximate heap use.
    pub fn heap_bytes(&self) -> usize {
        self.term_bytes.capacity() + 4 * (self.term_offsets.capacity() + self.doc_starts.capacity() + self.ords.capacity())
    }
}

/// Builds an [`Uninverted`] from terms fed in ascending order.
pub struct UninvertedBuilder {
    num_docs: u32,
    term_bytes: Vec<u8>,
    term_offsets: Vec<u32>,
    /// `(doc, ord)` in ord order; bucketed by doc in [`Self::finish`].
    pairs: Vec<(u32, u32)>,
    memory:crate::admission::Reservation,
}

impl UninvertedBuilder {
    /// A builder for a leaf of `num_docs` docs; `postings_hint` sizes the
    /// pair buffer (sum of doc freqs, if known).
    pub fn new(num_docs: u32, _postings_hint: usize) -> crate::error::Result<Self> {
        Self::with_budget(num_docs,&crate::admission::shared())
    }
    fn with_budget(num_docs:u32,budget:&std::sync::Arc<crate::admission::Budget>)->crate::error::Result<Self> {
        let memory=budget.reserve((num_docs as u64).saturating_mul(12).saturating_add(512),"fielddata doc starts")?;
        Ok(Self { num_docs, term_bytes: Vec::new(), term_offsets: vec![0], pairs: Vec::new(),memory })
    }

    /// Add the next term (strictly greater than the previous one); returns its ordinal.
    pub fn add_term(&mut self, text: &[u8]) -> crate::error::Result<u32> {
        // Vec growth may keep the old allocation alive while allocating its
        // replacement. The retained 2x-length bound covers the replacement;
        // separately admit the old capacity for that temporary overlap.
        let overlap=if text.len()>self.term_bytes.capacity().saturating_sub(self.term_bytes.len()) {
            self.term_bytes.capacity() as u64
        } else { 0 };
        self.memory.grow((text.len() as u64).saturating_mul(2).saturating_add(16).saturating_add(overlap),"fielddata term")?;
        let ord = self.term_offsets.len() as u32 - 1;
        self.term_bytes.extend_from_slice(text);
        self.memory.shrink(overlap);
        self.term_offsets.push(self.term_bytes.len() as u32);
        Ok(ord)
    }

    /// `doc` holds term `ord` (each pair at most once).
    #[inline]
    pub fn add_doc(&mut self, ord: u32, doc: DocId) -> crate::error::Result<()> {
        if doc < self.num_docs {
            self.memory.grow(24,"fielddata ordinal")?;
            self.pairs.push((doc, ord));
        }
        Ok(())
    }

    pub fn finish(self) -> Uninverted {
        let n = self.num_docs as usize;
        // Counting sort by doc. Pairs arrive in ord order and the scatter is
        // stable, so each doc's ordinals come out ascending.
        let mut doc_starts = vec![0u32; n + 1];
        for &(d, _) in &self.pairs {
            doc_starts[d as usize + 1] += 1;
        }
        for i in 0..n {
            doc_starts[i + 1] += doc_starts[i];
        }
        let mut cursor: Vec<u32> = doc_starts[..n].to_vec();
        let mut ords = vec![0u32; self.pairs.len()];
        for &(d, o) in &self.pairs {
            let c = &mut cursor[d as usize];
            ords[*c as usize] = o;
            *c += 1;
        }
        let term_bytes = self.term_bytes;
        let term_offsets = self.term_offsets;
        drop(cursor);drop(self.pairs);
        let mut u = Uninverted { term_bytes, term_offsets, doc_starts, ords, charge: Default::default(),memory:self.memory };
        u.memory.shrink(u.memory.bytes().saturating_sub((u.heap_bytes() as u64).saturating_add(512)));
        u.charge = crate::accounting::Charge::new(crate::accounting::Category::Fielddata, u.heap_bytes() as u64);
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fielddata_refuses_growth_before_mutation_and_retains_last_owner() {
        let budget=crate::admission::Budget::new(1024);
        assert!(UninvertedBuilder::with_budget(1000,&budget).is_err());assert_eq!(budget.used(),0);
        let mut builder=UninvertedBuilder::with_budget(1,&budget).unwrap();
        assert!(builder.add_term(&vec![b'x';1024]).is_err());
        assert!(builder.term_bytes.is_empty());assert_eq!(builder.term_offsets.len(),1);
        let ord=builder.add_term(b"ok").unwrap();builder.add_doc(ord,0).unwrap();
        let value=std::sync::Arc::new(builder.finish());let reader=value.clone();drop(value);
        assert!(budget.used()>0);assert_eq!(reader.term(0),b"ok");drop(reader);assert_eq!(budget.used(),0);
    }

    #[test]
    fn fielddata_term_reallocation_admits_old_and_new_buffers() {
        let budget=crate::admission::Budget::new(3000);
        let mut builder=UninvertedBuilder::with_budget(0,&budget).unwrap();
        builder.add_term(&vec![b'a';1024]).unwrap();
        assert_eq!(builder.term_bytes.len(),builder.term_bytes.capacity());
        let charged=budget.used();
        assert!(builder.add_term(b"b").is_err());
        assert_eq!(builder.term_bytes.len(),1024);
        assert_eq!(builder.term_offsets.len(),2);
        assert_eq!(budget.used(),charged);
        drop(builder);assert_eq!(budget.used(),0);
    }

    #[test]
    fn uninverted_tables_charge_their_heap() {
        let mut b = UninvertedBuilder::new(3, 0).unwrap();
        let a = b.add_term(b"apple").unwrap();
        b.add_doc(a, 1).unwrap();
        let u = b.finish();
        assert_eq!(u.charge.bytes(), u.heap_bytes() as u64);
        assert!(u.charge.bytes() > 0);
    }

    #[test]
    fn builder_buckets_ords_per_doc_ascending() {
        let mut b = UninvertedBuilder::new(4, 0).unwrap();
        let a = b.add_term(b"apple").unwrap();
        let z = b.add_term(b"zebra").unwrap();
        b.add_doc(a, 2).unwrap();
        b.add_doc(a, 0).unwrap();
        b.add_doc(z, 0).unwrap();
        b.add_doc(z, 3).unwrap();
        b.add_doc(z, 99).unwrap(); // beyond num_docs: ignored
        let u = b.finish();
        assert_eq!(u.num_docs(), 4);
        assert_eq!(u.num_terms(), 2);
        assert_eq!(u.ords(0), &[0, 1]);
        assert_eq!(u.ords(1), &[] as &[u32]);
        assert_eq!(u.min_ord(3), Some(1));
        assert_eq!(u.max_ord(0), Some(1));
        assert_eq!(u.term(1), b"zebra");
        assert_eq!(u.ords(7), &[] as &[u32]);
    }
}
