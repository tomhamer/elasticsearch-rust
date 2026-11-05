//! Durable, packed quantized blocks. Calibration and correction terms are part
//! of the block: reopening must not refit quantizers from a changed live set.
use super::*;
use crate::{codec::crc32, error::{Result, RusticError}};

fn corrupt(message: &str) -> RusticError { RusticError::Corrupted(format!("quantized block: {message}")) }
fn put(out: &mut Vec<u8>, n: u32) { out.extend_from_slice(&n.to_le_bytes()); }
struct Input<'a> { bytes: &'a [u8] }
impl<'a> Input<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.bytes.len() { return Err(corrupt("truncated")); }
        let (head, tail) = self.bytes.split_at(n); self.bytes = tail; Ok(head)
    }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn float(&mut self) -> Result<f32> {
        let v = f32::from_bits(self.u32()?);
        if !v.is_finite() { return Err(corrupt("nonfinite calibration/correction")); }
        Ok(v)
    }
}
impl EsCodes {
    /// Encode exact calibration, packed codes and correction terms with a CRC.
    /// Graph identity and document ownership belong to the enclosing manifest.
    pub(crate) fn checkpoint_len(&self)->Result<usize> {
        let (count,bits,header,correction)=match &self.q {
            EsQuantizer::Lucene99(q)=>(self.offsets.len(),q.bits,32usize,4usize),
            EsQuantizer::Osq(q)=>{
                if q.centroid.len()!=self.dims{return Err(corrupt("centroid shape"))}
                (self.corrections.len(),q.doc_bits,36usize.checked_add(self.dims.checked_mul(4).ok_or_else(||corrupt("centroid length overflow"))?).ok_or_else(||corrupt("header length overflow"))?,16usize)
            }
        };
        if self.dims==0 || count.checked_mul(Self::row_bytes_for(&self.q,self.dims))!=Some(self.codes.len()){return Err(corrupt("shape"))}
        if !(1..=8).contains(&bits){return Err(corrupt("code width"))}
        let packed=count.checked_mul(self.dims).and_then(|n|n.checked_mul(bits as usize)).ok_or_else(||corrupt("code length overflow"))?.div_ceil(8);
        header.checked_add(packed).and_then(|n|count.checked_mul(correction).and_then(|c|n.checked_add(c))).and_then(|n|n.checked_add(4)).ok_or_else(||corrupt("checkpoint length overflow"))
    }
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>> {
        let length=self.checkpoint_len()?;
        let mut out=Vec::new();out.try_reserve_exact(length).map_err(|_|corrupt("checkpoint allocation"))?;
        self.append_checkpoint(&mut out)?;Ok(out)
    }
    /// Caller has admitted and reserved the whole destination. Never grows it.
    pub(crate) fn append_checkpoint(&self,out:&mut Vec<u8>)->Result<()> {
        let length=self.checkpoint_len()?;
        if out.capacity().saturating_sub(out.len())<length{return Err(corrupt("checkpoint destination capacity"))}
        let beginning=out.len();
        let count = match &self.q { EsQuantizer::Lucene99(_) => self.offsets.len(), EsQuantizer::Osq(_) => self.corrections.len() };
        out.extend_from_slice(b"RQCB\x01\0\0\0");
        put(out, u32::try_from(self.dims).map_err(|_| corrupt("dimensions overflow"))?);
        put(out, u32::try_from(count).map_err(|_| corrupt("row count overflow"))?);
        let bits = match &self.q {
            EsQuantizer::Lucene99(q) => {
                put(out, 0); put(out, q.bits as u32);
                put(out, q.min.to_bits()); put(out, q.max.to_bits()); q.bits
            }
            EsQuantizer::Osq(q) => {
                if q.centroid.len() != self.dims { return Err(corrupt("centroid shape")); }
                put(out, 1); put(out, q.doc_bits as u32);
                put(out, q.query_bits as u32);
                put(out, match q.sim { OsqSim::Euclidean => 0, OsqSim::Dot => 1, OsqSim::MaxInnerProduct => 2 });
                put(out, q.centroid_dp.to_bits());
                for &v in &q.centroid { put(out, v.to_bits()); }
                q.doc_bits
            }
        };
        if !(1..=8).contains(&bits) { return Err(corrupt("code width")); }
        let total_bits = count.checked_mul(self.dims).and_then(|cells| cells.checked_mul(bits as usize)).ok_or_else(|| corrupt("code length overflow"))?;
        let start = out.len();
        out.resize(start.checked_add(total_bits.div_ceil(8)).ok_or_else(|| corrupt("block length overflow"))?, 0);
        for i in 0..count * self.dims {
            let code = self.code_at(i / self.dims, i % self.dims);
            if u16::from(code) >= 1u16 << bits { return Err(corrupt("code out of range")); }
            let pos = i * bits as usize;
            let value = (code as u16) << (pos % 8);
            out[start + pos / 8] |= value as u8;
            if pos % 8 + bits as usize > 8 { out[start + pos / 8 + 1] |= (value >> 8) as u8; }
        }
        match &self.q {
            EsQuantizer::Lucene99(_) => for &v in &self.offsets { put(out, v.to_bits()); },
            EsQuantizer::Osq(_) => for c in &self.corrections {
                put(out, c.lower.to_bits()); put(out, c.upper.to_bits());
                put(out, c.additional.to_bits()); put(out, c.component_sum as u32);
            }
        }
        let checksum = crc32(&out[beginning..]); put(out, checksum);
        debug_assert_eq!(out.len()-beginning,length);
        Ok(())
    }

    /// Decode only after checking checksum and all lengths. Int4 and BBQ
    /// restore directly into packed resident rows without a whole-block
    /// expanded-code allocation.
    pub fn from_checkpoint(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 32 { return Err(corrupt("header")); }
        let (body, checksum) = bytes.split_at(bytes.len() - 4);
        if crc32(body) != u32::from_le_bytes(checksum.try_into().unwrap()) { return Err(corrupt("checksum")); }
        let mut input = Input { bytes: body };
        if input.take(8)? != b"RQCB\x01\0\0\0" { return Err(corrupt("version")); }
        let dims = input.u32()? as usize;
        let count = input.u32()? as usize;
        let kind = input.u32()?;
        let bits = input.u32()?;
        if dims == 0 || !(1..=8).contains(&bits) { return Err(corrupt("shape/code width")); }
        let q = match kind {
            0 => {
                let min = input.float()?; let max = input.float()?;
                if max <= min { return Err(corrupt("clip interval")); }
                EsQuantizer::Lucene99(UniformQuantizer { min, max, bits: bits as u8 })
            }
            1 => {
                let query_bits = input.u32()?;
                if !(1..=8).contains(&query_bits) { return Err(corrupt("query code width")); }
                let sim = match input.u32()? { 0 => OsqSim::Euclidean, 1 => OsqSim::Dot, 2 => OsqSim::MaxInnerProduct, _ => return Err(corrupt("similarity")) };
                let centroid_dp = input.float()?;
                if centroid_dp < 0.0 { return Err(corrupt("negative centroid norm")); }
                if dims > input.bytes.len() / 4 { return Err(corrupt("centroid length")); }
                let mut centroid = Vec::with_capacity(dims);
                for _ in 0..dims { centroid.push(input.float()?); }
                EsQuantizer::Osq(Arc::new(OsqQuantizer { sim, centroid, centroid_dp, doc_bits: bits as u8, query_bits: query_bits as u8 }))
            }
            _ => return Err(corrupt("quantizer kind")),
        };
        let cells = count.checked_mul(dims).ok_or_else(|| corrupt("shape overflow"))?;
        let packed_len = cells.checked_mul(bits as usize).ok_or_else(|| corrupt("code length overflow"))?.div_ceil(8);
        let correction_len = count.checked_mul(if kind == 0 {4} else {16}).ok_or_else(|| corrupt("correction length overflow"))?;
        if packed_len.checked_add(correction_len) != Some(input.bytes.len()) { return Err(corrupt("length/trailing data")); }
        let packed = input.take(packed_len)?;
        let mut codes = Vec::new();
        codes.try_reserve_exact(count.checked_mul(Self::row_bytes_for(&q, dims)).ok_or_else(|| corrupt("resident code length"))?)
            .map_err(|_| corrupt("code allocation"))?;
        let mut row = Vec::new();
        if count > 0 { row.try_reserve_exact(dims).map_err(|_| corrupt("decode row allocation"))?; }
        for r in 0..count {
            row.clear();
            for d in 0..dims {
                let pos = (r * dims + d) * bits as usize;
                let pair = u16::from(packed[pos / 8]) | (u16::from(*packed.get(pos / 8 + 1).unwrap_or(&0)) << 8);
                row.push(((pair >> (pos % 8)) & ((1u16 << bits) - 1)) as u8);
            }
            Self::append_codes(&q, &row, &mut codes);
        }
        let mut offsets = Vec::new(); let mut corrections = Vec::new();
        if kind == 0 {
            offsets.try_reserve_exact(count).map_err(|_| corrupt("offset allocation"))?;
            for _ in 0..count { offsets.push(input.float()?); }
        } else {
            corrections.try_reserve_exact(count).map_err(|_| corrupt("correction allocation"))?;
            for _ in 0..count {
                let lower = input.float()?; let upper = input.float()?; let additional = input.float()?;
                let component_sum = input.u32()? as i32;
                if upper < lower || component_sum < 0 || component_sum as u64 > dims as u64 * ((1u64 << bits) - 1) {
                    return Err(corrupt("invalid correction"));
                }
                corrections.push(OsqCorrections { lower, upper, additional, component_sum });
            }
        }
        Ok(Self { q, dims, codes, offsets, corrections, admission: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_and_scores_survive_packed_checkpoint_without_refitting() {
        let rows: Vec<Vec<f32>> = (0..17).map(|i| (0..13).map(|j| ((i * 13 + j * 7) % 31) as f32 / 19.0 - 0.5).collect()).collect();
        for sim in [Similarity::L2, Similarity::Cosine, Similarity::DotProduct, Similarity::MaxInnerProduct] {
            for (bits, osq) in [(7,false), (4,false), (7,true), (4,true), (1,true)] {
                let spec = EsQuant { bits, osq, confidence_interval: None };
                let original = EsCodes::build(spec.fit(sim, rows.len(), |i| &rows[i]), sim, rows.len(), 13, |i| &rows[i]);
                let bytes = original.checkpoint_bytes().unwrap();
                assert_eq!(bytes.len(),original.checkpoint_len().unwrap());
                let mut short=Vec::new();
                assert!(original.append_checkpoint(&mut short).is_err());
                assert!(short.is_empty());
                let mut prefixed=Vec::with_capacity(bytes.len()+7);prefixed.extend_from_slice(b"prefix!");
                let capacity=prefixed.capacity();original.append_checkpoint(&mut prefixed).unwrap();
                assert_eq!(prefixed.capacity(),capacity);
                assert_eq!(&prefixed[7..],bytes.as_slice(),"embedded CRC excludes prefix");
                let restored = EsCodes::from_checkpoint(&bytes).unwrap();
                assert_eq!(restored.checkpoint_bytes().unwrap(), bytes);
                for query in &rows {
                    let before = EsQuantScorer::new(original.q.clone(), sim, query);
                    let after = EsQuantScorer::new(restored.q.clone(), sim, query);
                    for i in 0..rows.len() { assert_eq!(before.score_row(&original,i).to_bits(), after.score_row(&restored,i).to_bits()); }
                }
                for end in 0..bytes.len() { assert!(EsCodes::from_checkpoint(&bytes[..end]).is_err()); }
                // Header attacks with a valid checksum must fail before any
                // allocation based on the claimed dimensions/row count.
                for (offset, value) in [(8, u32::MAX), (12, u32::MAX), (20, 0), (20, 9)] {
                    let mut damaged = bytes.clone();
                    damaged[offset..offset+4].copy_from_slice(&value.to_le_bytes());
                    let end = damaged.len() - 4;
                    let checksum = crc32(&damaged[..end]);
                    damaged[end..].copy_from_slice(&checksum.to_le_bytes());
                    assert!(EsCodes::from_checkpoint(&damaged).is_err());
                }
                let mut damaged = bytes; damaged[24] ^= 1;
                assert!(EsCodes::from_checkpoint(&damaged).is_err());
            }
        }
    }
}
