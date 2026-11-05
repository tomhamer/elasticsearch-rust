//! Just enough of `Lucene90DocValuesFormat` to find which documents carry a
//! value for `__soft_deletes` (Elasticsearch's soft-delete marker). The
//! metadata file holds every doc-values field of the segment in sequence, so
//! the other entry kinds are parsed only to be skipped.

use super::io::{expect_codec, read_index_header, In, FOOTER_LEN};
use super::{FieldInfo, SegmentCommit, SegmentFiles};
use anyhow::{anyhow, bail, Result};

const NUMERIC: u8 = 0;
const BINARY: u8 = 1;
const SORTED: u8 = 2;
const SORTED_SET: u8 = 3;
const SORTED_NUMERIC: u8 = 4;
const TERMS_DICT_BLOCK_LZ4_SHIFT: u32 = 6;

struct DocsWithField {
    offset: i64,
    length: i64,
    dense_rank_power: i8,
}

fn read_numeric(m: &mut In) -> Result<DocsWithField> {
    let offset = m.le_i64()?;
    let length = m.le_i64()?;
    let _jump_table_entries = m.le_u16()?;
    let dense_rank_power = m.u8()? as i8;
    let _num_values = m.le_i64()?;
    let table_size = m.le_i32()?;
    if table_size > 256 {
        bail!("doc values: table size {} > 256", table_size);
    }
    for _ in 0..table_size.max(0) {
        m.le_i64()?;
    }
    let _bpv = m.u8()?;
    let _min = m.le_i64()?;
    let _gcd = m.le_i64()?;
    let _values_offset = m.le_i64()?;
    let _values_length = m.le_i64()?;
    let _value_jump_table_offset = m.le_i64()?;
    Ok(DocsWithField { offset, length, dense_rank_power })
}

fn skip_monotonic_meta(m: &mut In, num_values: u64, block_shift: u32) -> Result<()> {
    if block_shift > 62 {
        bail!("doc values: bad block shift {}", block_shift);
    }
    let mut blocks = num_values >> block_shift;
    if blocks << block_shift < num_values {
        blocks += 1;
    }
    m.skip(blocks as usize * 21)
}

fn skip_binary(m: &mut In) -> Result<()> {
    m.skip(8 * 4 + 2 + 1)?;
    let num_docs = m.le_i32()? as u64;
    let min_len = m.le_i32()?;
    let max_len = m.le_i32()?;
    if min_len < max_len {
        m.le_i64()?;
        let shift = m.vint()?;
        skip_monotonic_meta(m, num_docs + 1, shift)?;
        m.le_i64()?;
    }
    Ok(())
}

fn skip_term_dict(m: &mut In) -> Result<()> {
    let size = m.vlong()?;
    let block_shift = m.le_i32()? as u32;
    let addresses = (size + (1 << TERMS_DICT_BLOCK_LZ4_SHIFT) - 1) >> TERMS_DICT_BLOCK_LZ4_SHIFT;
    skip_monotonic_meta(m, addresses, block_shift)?;
    m.le_i32()?; // max term length
    m.le_i32()?; // max block length
    m.skip(8 * 4)?;
    let index_shift = m.le_i32()? as u32;
    let index_size = (size + (1u64 << index_shift) - 1) >> index_shift;
    skip_monotonic_meta(m, 1 + index_size, block_shift)?;
    m.skip(8 * 4)
}

fn skip_sorted_numeric(m: &mut In) -> Result<()> {
    let start = m.pos();
    read_numeric(m)?;
    // numValues sits after docsWithField offset/length, jump table count and rank power
    let num_values = In::at(m.buf(), start + 8 + 8 + 2 + 1).le_i64()?;
    let num_docs = m.le_i32()? as i64;
    if num_docs != num_values {
        m.le_i64()?;
        let shift = m.vint()?;
        skip_monotonic_meta(m, num_docs as u64 + 1, shift)?;
        m.le_i64()?;
    }
    Ok(())
}

fn skip_sorted(m: &mut In) -> Result<()> {
    read_numeric(m)?;
    skip_term_dict(m)
}

fn skip_sorted_set(m: &mut In) -> Result<()> {
    match m.u8()? {
        0 => skip_sorted(m),
        1 => {
            skip_sorted_numeric(m)?;
            skip_term_dict(m)
        }
        x => bail!("doc values: bad sorted-set multiValued flag {}", x),
    }
}

/// Docs that have a `__soft_deletes` value in this segment as of the commit.
pub fn soft_deleted_docs(
    files: &SegmentFiles,
    sc: &SegmentCommit,
    field: &FieldInfo,
    fields: &[FieldInfo],
    max_doc: u32,
) -> Result<Vec<u32>> {
    let format = field.attributes.get("PerFieldDocValuesFormat.format").map(String::as_str).unwrap_or("Lucene90");
    let suffix = field.attributes.get("PerFieldDocValuesFormat.suffix").map(String::as_str).unwrap_or("0");
    if format != "Lucene90" {
        bail!("{} uses doc values format {}; only Lucene90 is supported", field.name, format);
    }
    let tail = format!("_{}_{}", format, suffix);
    let (meta, data) = match sc.dv_update_files.get(&(field.number as i32)) {
        Some(update) => {
            let dvm = update.iter().find(|f| f.ends_with(&format!("{}.dvm", tail)));
            let dvd = update.iter().find(|f| f.ends_with(&format!("{}.dvd", tail)));
            match (dvm, dvd) {
                (Some(m), Some(d)) => (files.open_plain(m)?, files.open_plain(d)?),
                _ => bail!("segment {}: doc-values update files for {} not found in {:?}", sc.name, field.name, update),
            }
        }
        None => (files.open(&format!("{}.dvm", tail))?, files.open(&format!("{}.dvd", tail))?),
    };
    let mb = meta.as_slice();
    let mut m = In::new(mb);
    let h = read_index_header(&mut m)?;
    expect_codec(&h, "doc values meta", &["Lucene90DocValuesMetadata"])?;
    loop {
        let num = m.le_i32()?;
        if num == -1 {
            break;
        }
        let ty = m.u8()?;
        let fi = fields.iter().find(|f| f.number as i32 == num);
        if fi.map_or(false, |f| f.has_dv_skip_index) {
            m.skip(8 * 4 + 4 + 4)?;
        }
        if num as u32 == field.number {
            if ty != NUMERIC {
                bail!("{} has doc-values type {} (expected numeric)", field.name, ty);
            }
            let d = read_numeric(&mut m)?;
            return docs_with_field(data.as_slice(), &d, max_doc);
        }
        match ty {
            NUMERIC => {
                read_numeric(&mut m)?;
            }
            BINARY => skip_binary(&mut m)?,
            SORTED => skip_sorted(&mut m)?,
            SORTED_SET => skip_sorted_set(&mut m)?,
            SORTED_NUMERIC => skip_sorted_numeric(&mut m)?,
            x => bail!("doc values: unknown entry type {} for field number {}", x, num),
        }
    }
    Err(anyhow!("segment {}: no doc-values entry for {}", sc.name, field.name))
}

fn docs_with_field(data: &[u8], d: &DocsWithField, max_doc: u32) -> Result<Vec<u32>> {
    match d.offset {
        -2 => return Ok(Vec::new()),
        -1 => return Ok((0..max_doc).collect()),
        _ => {}
    }
    let start = d.offset as usize;
    let end = start + d.length as usize;
    if end > data.len().saturating_sub(FOOTER_LEN) + FOOTER_LEN {
        bail!("doc values: IndexedDISI slice outside data file");
    }
    indexed_disi(&data[start..end], d.dense_rank_power)
}

/// Decodes an `IndexedDISI` doc set: 65536-doc blocks, each ALL, DENSE
/// (optional rank table + 1024-word bitset) or SPARSE (sorted u16 list).
pub fn indexed_disi(b: &[u8], dense_rank_power: i8) -> Result<Vec<u32>> {
    let mut inp = In::new(b);
    let mut out = Vec::new();
    while inp.remaining() >= 4 {
        let block = inp.le_u16()? as u32;
        let card = inp.le_u16()? as u32 + 1;
        if block == 0x7fff {
            break;
        }
        let base = block << 16;
        if card == 65536 {
            out.extend(base..base + 65536);
        } else if card > 4095 {
            if dense_rank_power != -1 {
                if !(7..=15).contains(&dense_rank_power) {
                    bail!("IndexedDISI: bad dense rank power {}", dense_rank_power);
                }
                inp.skip(1024 >> (dense_rank_power - 7))?;
            }
            let before = out.len();
            for w in 0..1024u32 {
                let mut word = inp.le_u64()?;
                while word != 0 {
                    let t = word.trailing_zeros();
                    out.push(base + w * 64 + t);
                    word &= word - 1;
                }
            }
            if (out.len() - before) as u32 != card {
                bail!("IndexedDISI: dense block {} has {} docs, header says {}", block, out.len() - before, card);
            }
        } else {
            for _ in 0..card {
                out.push(base + inp.le_u16()? as u32);
            }
        }
    }
    Ok(out)
}
