//! `dense_vector` index / node stats (ES 9 `DenseVectorStats` with the
//! `off_heap` section): the number of indexed vectors and, per field, the
//! off-heap bytes of each vector "file" ES would hold (`vec` raw vectors,
//! `vex` HNSW graph, `veq` scalar-quantized codes, `veb` binary-quantized
//! codes, `cenivf` / `clivf` DiskBBQ centroids and clusters).
//!
//! `vec` is the raw vectors' size in their element type (as Lucene stores
//! them; our payloads keep f32 for `float` / `bfloat16` and packed bytes
//! for `byte` / `bit`). An HNSW field's `vex` is its shared
//! mutable graph's real adjacency bytes, and its `veq` / `veb` the graph's
//! code log bytes once it has one; a `bbq_disk` field's `cenivf` / `clivf`
//! are its shared index's centroid and posting file bytes once it has one.
//! Everything else is estimated from the vector counts and the field's
//! index options as the equivalent Lucene files.

use crate::index::IndexService;
use crate::mapping::{EsType, ResolvedField};
use serde_json::{json, Map, Value};

/// Off-heap bytes of one field, by ES file extension.
#[derive(Default, Clone, Copy, Debug)]
struct FieldBytes {
    vec: u64,
    vex: Option<u64>,
    veq: u64,
    veb: u64,
    cenivf: u64,
    clivf: u64,
}

fn field_bytes(f: &ResolvedField, n: u64, has_graph: bool) -> FieldBytes {
    let et = crate::vectors::element_type(f);
    let dims = f.dims.unwrap_or(0) as u64;
    let elem_bytes = match et {
        crate::vectors::ElementType::Float => dims * 4,
        crate::vectors::ElementType::BFloat16 => dims * 2,
        crate::vectors::ElementType::Byte => dims,
        crate::vectors::ElementType::Bit => dims / 8,
    };
    let io = f.ext.as_ref().and_then(|e| e.params.get("index_options"));
    let t = io.and_then(|o| o.get("type")).and_then(Value::as_str).unwrap_or("hnsw");
    let threshold = io.and_then(|o| o.get("flat_index_threshold")).and_then(Value::as_i64).unwrap_or(-1);
    let m = io.and_then(|o| o.get("m")).and_then(Value::as_u64).unwrap_or(16);
    let mut b = FieldBytes { vec: n * elem_bytes, ..Default::default() };
    if n == 0 {
        return b;
    }
    if t.ends_with("hnsw") {
        // A graph is built for every segment with `flat_index_threshold: 0`,
        // else (dynamic) only once segments grow past the flat threshold.
        let graph = threshold == 0 || has_graph;
        b.vex = Some(if graph { n * (1 + (n - 1).min(2 * m)) + 4 } else { 0 });
    }
    match t {
        "int8_hnsw" | "int8_flat" => b.veq = n * (dims + 16),
        "int4_hnsw" | "int4_flat" => b.veq = n * (dims.div_ceil(2) + 16),
        "bbq_hnsw" | "bbq_flat" => b.veb = n * (dims.div_ceil(8) + 16),
        "bbq_disk" => {
            let cluster = io.and_then(|o| o.get("cluster_size")).and_then(Value::as_u64).unwrap_or(384).max(1);
            let clusters = n.div_ceil(cluster);
            let bits = io.and_then(|o| o.get("bits")).and_then(Value::as_u64).unwrap_or(1);
            b.cenivf = clusters * (dims * 4 + 16) + 82;
            b.clivf = n * ((dims * bits).div_ceil(8) + 16) + 12;
        }
        _ => {}
    }
    b
}

/// The `dense_vector` stats of an index: `value_count` (indexed vectors),
/// `off_heap` totals and, with `per_field`, the `fielddata` breakdown.
pub fn index_stats(svc: &IndexService, per_field: bool) -> Value {
    let lk = svc.lookup();
    let mut fields: Vec<&ResolvedField> = lk.fields.values().map(|f| f.as_ref()).filter(|f| f.declared == EsType::DenseVector && f.core_id.is_some() && crate::vectors::field_options(f).is_some()).collect();
    fields.sort_by(|a, b| a.name.cmp(&b.name));
    let mut count = 0u64;
    let mut per: Vec<(String, FieldBytes)> = Vec::new();
    let shards = if svc.is_open() { svc.local_shards() } else { Vec::new() };
    for f in fields {
        let id = f.core_id.unwrap_or(0);
        let (mut n, mut graph) = (0u64, false);
        // DiskBBQ: the shared indexes' actual centroid and posting bytes.
        let mut ivf: Option<(u64, u64)> = None;
        // The shared mutable graphs: (adjacency, code log) bytes.
        let mut mutable: Option<(u64, u64)> = None;
        for sh in &shards {
            // The store handle under a short read; memtables from the
            // published view (no lock while counting).
            let store = sh.shard.core.read().vector_store().clone();
            n += store.num_alive(id) as u64;
            graph |= store.segments(id).iter().any(|s| s.has_graph());
            if let Some((graph_bytes, code_bytes)) = store.mutable_off_heap(id) {
                let (g, c) = mutable.unwrap_or((0, 0));
                mutable = Some((g + graph_bytes, c + code_bytes));
            }
            if let Some(state) = store.ivf_state(id) {
                let (c, p) = ivf.unwrap_or((0, 0));
                ivf = Some((c + state.centroid_bytes() as u64, p + state.posting_bytes() as u64));
            }
            // Refreshed but unflushed docs live in the memtable reader.
            let view = sh.shard.reader_view();
            for leaf in view.leaves() {
                if let rustic_core::query::LeafReader::Memtable(writer, ..) = leaf {
                    // Presence needs no owned copy of every vector's values.
                    n += (0..writer.num_docs()).filter(|d| !writer.is_deleted(*d) && writer.vector(id, *d).is_some()).count() as u64;
                }
            }
        }
        count += n;
        if n > 0 {
            let mut bytes = field_bytes(f, n, graph);
            if let Some((cenivf, clivf)) = ivf {
                bytes.cenivf = cenivf;
                bytes.clivf = clivf;
            }
            if let Some((graph_bytes, code_bytes)) = mutable {
                apply_mutable(&mut bytes, graph_bytes, code_bytes);
            }
            per.push((f.name.clone(), bytes));
        }
    }
    stats_json(count, &per, per_field)
}

/// Real sizes of an HNSW field's shared mutable graph in place of the
/// estimates: the graph is always built (there is no per-segment flat
/// threshold), so `vex` is its adjacency even for few vectors; the code
/// log, when it holds anything, is the quantized codes (`veb` for BBQ).
fn apply_mutable(bytes: &mut FieldBytes, graph_bytes: u64, code_bytes: u64) {
    if bytes.vex.is_some() {
        bytes.vex = Some(graph_bytes);
    }
    if code_bytes > 0 {
        if bytes.veb > 0 {
            bytes.veb = code_bytes;
        } else if bytes.veq > 0 {
            bytes.veq = code_bytes;
        }
    }
}

fn stats_json(count: u64, per: &[(String, FieldBytes)], per_field: bool) -> Value {
    let sum = |g: &dyn Fn(&FieldBytes) -> u64| per.iter().map(|(_, b)| g(&b)).sum::<u64>();
    let (veb, vec, veq, vex, cen, cli) = (sum(&|b| b.veb), sum(&|b| b.vec), sum(&|b| b.veq), sum(&|b| b.vex.unwrap_or(0)), sum(&|b| b.cenivf), sum(&|b| b.clivf));
    let mut off = Map::new();
    off.insert("total_size_bytes".into(), json!(veb + vec + veq + vex + cen + cli));
    off.insert("total_veb_size_bytes".into(), json!(veb));
    off.insert("total_vec_size_bytes".into(), json!(vec));
    off.insert("total_veq_size_bytes".into(), json!(veq));
    off.insert("total_vex_size_bytes".into(), json!(vex));
    if cen + cli > 0 {
        off.insert("total_cenivf_size_bytes".into(), json!(cen));
        off.insert("total_clivf_size_bytes".into(), json!(cli));
    }
    if per_field && !per.is_empty() {
        let mut fd = Map::new();
        for (name, b) in per {
            // Keys in name order, only the files the field has.
            let mut e = Map::new();
            if b.cenivf > 0 {
                e.insert("cenivf_size_bytes".into(), json!(b.cenivf));
                e.insert("clivf_size_bytes".into(), json!(b.clivf));
            }
            e.insert("vec_size_bytes".into(), json!(b.vec));
            if b.veb > 0 {
                e.insert("veb_size_bytes".into(), json!(b.veb));
            }
            if b.veq > 0 {
                e.insert("veq_size_bytes".into(), json!(b.veq));
            }
            if let Some(x) = b.vex {
                e.insert("vex_size_bytes".into(), json!(x));
            }
            let mut sorted: Vec<(String, Value)> = e.into_iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            fd.insert(name.clone(), Value::Object(sorted.into_iter().collect()));
        }
        off.insert("fielddata".into(), Value::Object(fd));
    }
    json!({ "value_count": count, "off_heap": Value::Object(off) })
}

/// The empty stats (an index without vectors, the start of a sum).
pub fn zero() -> Value {
    stats_json(0, &[], false)
}

/// Drop the per-field breakdown from summed stats (`_all`, node level).
pub fn strip_fielddata(v: &mut Value) {
    if let Some(o) = v.get_mut("off_heap").and_then(Value::as_object_mut) {
        o.remove("fielddata");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutable_graph_sizes_replace_the_estimates() {
        // A default (dynamic threshold) HNSW field with no segment graph
        // used to report vex 0; the shared graph's bytes are reported.
        let mut b = FieldBytes { vec: 800, vex: Some(0), veq: 0, ..Default::default() };
        apply_mutable(&mut b, 1234, 0);
        assert_eq!(b.vex, Some(1234));
        assert_eq!(b.veq, 0, "no codes are invented for an f32 field");
        let mut q = FieldBytes { vec: 800, vex: Some(0), veq: 240, ..Default::default() };
        apply_mutable(&mut q, 99, 512);
        assert_eq!((q.vex, q.veq, q.veb), (Some(99), 512, 0));
        let mut bbq = FieldBytes { vec: 800, vex: Some(0), veb: 40, ..Default::default() };
        apply_mutable(&mut bbq, 99, 64);
        assert_eq!((bbq.veq, bbq.veb), (0, 64));
        // Flat fields report no graph file at all.
        let mut flat = FieldBytes { vec: 800, vex: None, veq: 240, ..Default::default() };
        apply_mutable(&mut flat, 99, 512);
        assert_eq!((flat.vex, flat.veq), (None, 512));
    }
}
