//! `frequent_item_sets` (deprecated alias `frequent_items`): closed
//! frequent item sets over the values of several fields (port of ES ML
//! `EclatMapReducer` / `CountingItemSetTraverser` / `FrequentItemSetCollector`:
//! the same pruning, traversal order and top-k collection).

use super::super::{key_json, Ctx, Docs, FieldRef, Key, Matcher};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// An item set as a bit set over item positions (ES `ItemSetBitSet`).
#[derive(Clone, PartialEq, Eq, Debug)]
struct Bits {
    words: Vec<u64>,
    card: usize,
}

impl Bits {
    fn from_positions(ps: &[usize]) -> Bits {
        let mut words: Vec<u64> = Vec::new();
        for &p in ps {
            let w = p / 64;
            if words.len() <= w {
                words.resize(w + 1, 0);
            }
            words[w] |= 1u64 << (p % 64);
        }
        while words.last() == Some(&0) {
            words.pop();
        }
        Bits { card: ps.len(), words }
    }
    fn contains_all(&self, other: &Bits) -> bool {
        other.words.iter().enumerate().all(|(i, w)| self.words.get(i).copied().unwrap_or(0) & w == *w)
    }
    fn positions(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (i, w) in self.words.iter().enumerate() {
            for b in 0..64 {
                if w & (1u64 << b) != 0 {
                    out.push(i * 64 + b);
                }
            }
        }
        out
    }
    /// ES `ItemSetBitSet.compare`.
    fn compare(a: &Bits, b: &Bits) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        if a.card != b.card {
            return if a.card > b.card { Greater } else { Less };
        }
        if a.words.len() != b.words.len() {
            return if a.words.len() < b.words.len() { Greater } else { Less };
        }
        match a.words.iter().zip(&b.words).find(|(x, y)| x != y) {
            None => Equal,
            Some((x, y)) => {
                if (*x as i64) < (*y as i64) {
                    Greater
                } else {
                    Less
                }
            }
        }
    }
}

#[derive(Clone)]
struct Cand {
    seq: u64,
    bits: Bits,
    count: u64,
}

/// ES `FrequentItemSetPriorityQueue.lessThan`.
fn less_than(a: &Cand, b: &Cand) -> bool {
    if a.count == b.count {
        Bits::compare(&a.bits, &b.bits) == std::cmp::Ordering::Less
    } else {
        a.count < b.count
    }
}

/// ES `FrequentItemSetCollector`: the top `size` closed item sets.
struct Collector {
    size: usize,
    min: u64,
    queue: Vec<Cand>,
    by_count: HashMap<u64, Vec<Cand>>,
    seq: u64,
}

impl Collector {
    fn top(&self) -> Option<&Cand> {
        self.queue.iter().reduce(|a, b| if less_than(b, a) { b } else { a })
    }
    fn threshold(&self) -> u64 {
        if self.queue.len() < self.size {
            self.min
        } else {
            self.top().map(|t| t.count).unwrap_or(self.min)
        }
    }
    fn remove_by_count(&mut self, c: &Cand) {
        if let Some(v) = self.by_count.get_mut(&c.count) {
            v.retain(|x| x.seq != c.seq);
        }
    }
    fn add(&mut self, bits: &Bits, count: u64) -> u64 {
        if self.queue.len() == self.size {
            if let Some(t) = self.top() {
                if count < t.count || (count == t.count && bits.card <= t.bits.card) {
                    return t.count;
                }
            }
        }
        if let Some(list) = self.by_count.get(&count).cloned() {
            for other in list {
                if bits.card > other.bits.card && bits.contains_all(&other.bits) {
                    // A superset with the same count replaces its subset.
                    self.remove_by_count(&other);
                    self.queue.retain(|x| x.seq != other.seq);
                    break;
                } else if bits.card < other.bits.card && other.bits.contains_all(bits) {
                    return self.threshold();
                }
            }
        }
        let cand = Cand { seq: self.seq, bits: bits.clone(), count };
        self.seq += 1;
        if self.queue.len() < self.size {
            self.queue.push(cand.clone());
        } else if let Some(t) = self.top().cloned() {
            if !less_than(&cand, &t) {
                self.queue.retain(|x| x.seq != t.seq);
                self.remove_by_count(&t);
                self.queue.push(cand.clone());
            } else {
                self.remove_by_count(&cand);
            }
        }
        self.by_count.entry(count).or_default().push(cand);
        self.threshold()
    }
}

/// ES `CountingItemSetTraverser` over `ItemSetTraverser`: depth-first over
/// item positions (most frequent first).
struct Traverser<'t> {
    n: usize,
    /// Per transaction: (ascending positions, multiplicity).
    trans: &'t [(Vec<usize>, u64)],
    item_counts: &'t [u64],
    /// `iters[d]`: next position the iterator of depth `d` returns.
    iters: Vec<usize>,
    stack: Vec<usize>,
    counts: Vec<u64>,
    visited: Vec<bool>,
    sp: isize,
}

impl Traverser<'_> {
    fn items(&self) -> usize {
        self.sp.max(0) as usize
    }
    fn next(&mut self, min_count: u64) -> bool {
        if self.sp == -1 {
            return false;
        }
        let pos = loop {
            let d = self.sp as usize;
            if self.iters[d] < self.n {
                let p = self.iters[d];
                self.iters[d] += 1;
                break p;
            }
            self.sp -= 1;
            if self.sp == -1 {
                return false;
            }
            self.stack.pop();
        };
        let d = self.sp as usize;
        if self.iters.len() == d + 1 {
            self.iters.push(pos + 1);
        } else {
            self.iters[d + 1] = pos + 1;
        }
        self.stack.push(pos);
        self.sp += 1;
        let depth = self.sp as usize;
        let single = self.item_counts[pos];
        if self.counts.len() < depth {
            self.counts.resize(depth, 0);
            self.visited.resize(depth, false);
        }
        if depth == 1 {
            // ES leaves the visited flag of depth 1 untouched here.
            self.counts[0] = single;
            return true;
        }
        let c = if single < min_count {
            single
        } else {
            let bits = &self.stack;
            self.trans.iter().filter(|(t, _)| bits.iter().all(|p| t.binary_search(p).is_ok())).map(|(_, m)| *m).sum()
        };
        self.counts[depth - 1] = c;
        self.visited[depth - 1] = false;
        true
    }
    fn count(&self) -> u64 {
        if self.items() > 0 { self.counts[self.items() - 1] } else { 0 }
    }
    fn parent_count(&self) -> u64 {
        if self.items() > 1 { self.counts[self.items() - 2] } else { 0 }
    }
    fn visited(&self) -> bool {
        if self.items() > 0 { self.visited[self.items() - 1] } else { true }
    }
    fn parent_visited(&self) -> bool {
        if self.items() > 1 { self.visited[self.items() - 2] } else { true }
    }
    fn set_parent_visited(&mut self) {
        if self.items() > 1 {
            let i = self.items() - 2;
            self.visited[i] = true;
        }
    }
    fn at_leaf(&self) -> bool {
        self.sp != -1 && self.iters[self.sp as usize] >= self.n
    }
    fn prune(&mut self) {
        if self.sp == -1 {
            return;
        }
        self.sp -= 1;
        if self.sp == -1 {
            return;
        }
        self.stack.pop();
    }
    fn prune_to_next_main_branch(&mut self) {
        let this = self.count();
        while self.items() > 1 && self.count() == this {
            self.prune();
        }
    }
    fn bits(&self) -> Bits {
        Bits::from_positions(&self.stack)
    }
    fn parent_bits(&self) -> Bits {
        Bits::from_positions(&self.stack[..self.stack.len().saturating_sub(1)])
    }
}

/// ES `EclatMapReducer.eclat`: the reported item sets, best first.
fn eclat(n_items: usize, item_counts: &[u64], trans: &[(Vec<usize>, u64)], min_count: u64, min_size: usize, size: usize) -> Result<Vec<(Vec<usize>, u64)>> {
    let mut min_count = min_count;
    let mut col = Collector { size, min: min_count, queue: Vec::new(), by_count: HashMap::new(), seq: 0 };
    let mut tr = Traverser { n: n_items, trans, item_counts, iters: vec![0], stack: Vec::new(), counts: Vec::new(), visited: Vec::new(), sp: 0 };
    let mut budget: u64 = 5_000_000;
    while tr.next(min_count) {
        budget -= 1;
        if budget == 0 {
            return Err(EsError::illegal_argument("frequent_item_sets traversal limit exceeded; increase minimum_support or restrict the query"));
        }
        if tr.count() < min_count {
            tr.prune();
            if tr.at_leaf() && !tr.visited() && tr.count() >= min_count && tr.items() >= min_size {
                min_count = col.add(&tr.bits(), tr.count());
            }
            if tr.at_leaf() {
                tr.prune_to_next_main_branch();
            }
            continue;
        }
        if !tr.parent_visited() && tr.items() > min_size && tr.count() < tr.parent_count() {
            min_count = col.add(&tr.parent_bits(), tr.parent_count());
        }
        tr.set_parent_visited();
        if tr.at_leaf() && tr.items() >= min_size {
            min_count = col.add(&tr.bits(), tr.count());
        }
        if tr.at_leaf() {
            tr.prune();
            tr.prune_to_next_main_branch();
        }
    }
    let mut q = col.queue;
    q.sort_by(|a, b| if less_than(a, b) { std::cmp::Ordering::Greater } else if less_than(b, a) { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal });
    Ok(q.into_iter().map(|c| (c.bits.positions(), c.count)).collect())
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub(super) fn frequent_item_sets(&self, name: &str, t: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let fields = o.get("fields").and_then(Value::as_array).ok_or_else(|| EsError::illegal_argument("Required [fields]"))?;
        if fields.len() < 2 && t == "frequent_item_sets" && fields.is_empty() {
            return Err(EsError::illegal_argument(format!("[fields] must contain at least 1 field for aggregation [{name}]")));
        }
        let min_support = o.get("minimum_support").and_then(Value::as_f64).unwrap_or(0.1);
        if !(0.0..=1.0).contains(&min_support) {
            return Err(EsError::illegal_argument(format!("[minimum_support] must be in range [0.0, 1.0], got [{min_support}]")));
        }
        let min_size = o.get("minimum_set_size").and_then(Value::as_i64).unwrap_or(1);
        if min_size < 1 {
            return Err(EsError::illegal_argument(format!("[minimum_set_size] must be greater than 0, got [{min_size}]")));
        }
        let size = o.get("size").and_then(Value::as_i64).unwrap_or(10);
        if size < 1 {
            return Err(EsError::illegal_argument(format!("[size] must be greater than 0, got [{size}]")));
        }
        struct Src {
            name: String,
            f: FieldRef,
            include: Option<Matcher>,
            exclude: Option<Matcher>,
        }
        let mut srcs: Vec<Src> = Vec::new();
        for fo in fields {
            let fo = fo.as_object().ok_or_else(|| EsError::parsing_at("[fields] entries must be objects", 1, 1))?;
            let fname = fo.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [field]"))?;
            if let Some(rf) = self.segs.iter().find_map(|s| s.lookup.get(fname)) {
                let ok = matches!(rf.ty, EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Flattened | EsType::Ip | EsType::Boolean)
                    || rf.ty.is_numeric()
                    || rf.ty.is_date();
                if !ok || crate::fieldtypes::is_histogram(rf.declared) {
                    return Err(EsError::illegal_argument(format!("Field [{fname}] of type [{}] is not supported for aggregation [{t}]", rf.declared.name())));
                }
            }
            let f = self.source_field(fname)?;
            srcs.push(Src { name: fname.to_string(), f, include: Matcher::parse(fo.get("include"))?, exclude: Matcher::parse(fo.get("exclude"))? });
        }
        // The optional `filter` restricts the transactions, not the total.
        let total = super::super::count(docs);
        let mined: Docs = match o.get("filter") {
            Some(q) => self.filter_docs(q, docs)?,
            None => docs.clone(),
        };
        // Items: (field index, value) -> id, by descending frequency.
        let mut item_ids: HashMap<(usize, String), usize> = HashMap::new();
        let mut items: Vec<(usize, Key, usize, usize)> = Vec::new(); // (field, key, count, first seen)
        let mut raw: HashMap<Vec<usize>, u64> = HashMap::new();
        for (s, v) in mined.iter().enumerate() {
            for &(d, _) in v {
                let mut tr = Vec::new();
                for (fi, src) in srcs.iter().enumerate() {
                    let mut ks = self.values(&src.f, s, d);
                    ks.sort_by(super::super::cmp_key);
                    ks.dedup();
                    for k in ks {
                        if src.include.as_ref().map(|m| !m.matches(&k)).unwrap_or(false) || src.exclude.as_ref().map(|m| m.matches(&k)).unwrap_or(false) {
                            continue;
                        }
                        let id = *item_ids.entry((fi, format!("{k:?}"))).or_insert_with(|| {
                            items.push((fi, k.clone(), 0, items.len()));
                            items.len() - 1
                        });
                        items[id].2 += 1;
                        tr.push(id);
                    }
                }
                tr.sort_unstable();
                tr.dedup();
                *raw.entry(tr).or_default() += 1;
            }
        }
        let state = State {
            total: total as u64,
            filtered: (total - super::super::count(&mined)) as u64,
            fields: srcs.iter().map(|s| s.name.clone()).collect(),
            items: items.into_iter().map(|(fi, k, _, _)| {
                let value = match &k {
                    Key::Bool(b) => json!(b),
                    Key::Long(x) if srcs[fi].f.ty.map(|t| t.is_date()).unwrap_or(false) =>
                        json!(srcs[fi].f.format.clone().unwrap_or_default().format(*x, crate::date::Tz::UTC)),
                    other => key_json(other),
                };
                (fi, k, value)
            }).collect(),
            transactions: raw.into_iter().collect(),
            profile: self.env.profile,
        };
        if super::super::partial::active() || crate::search::aggregation_shard_response() {
            return Ok(json!({"buckets": [], "__p": state}));
        }
        let result = finish(state, o)?;
        self.add_buckets(result["buckets"].as_array().map_or(0, Vec::len))?;
        Ok(result)
    }
}

/// Lossless, deduplicated transactions. Local support/top-k pruning is unsafe:
/// a locally discarded item can belong to a globally selected closed set.
#[derive(serde::Serialize, serde::Deserialize)]
struct State {
    total: u64,
    filtered: u64,
    fields: Vec<String>,
    items: Vec<(usize, Key, Value)>,
    transactions: Vec<(Vec<usize>, u64)>,
    profile: bool,
}

/// Merge raw transactions, retaining state through intermediate node reductions.
pub fn reduce(o: &Map<String, Value>, vals: &[&Value], final_reduce: bool) -> Result<Value> {
    let mut merged = State { total: 0, filtered: 0, fields: Vec::new(), items: Vec::new(), transactions: Vec::new(), profile: false };
    let mut ids = HashMap::new();
    let mut transactions: HashMap<Vec<usize>, u64> = HashMap::new();
    for v in vals {
        let state: State = serde_json::from_value(v.get("__p").cloned().ok_or_else(|| EsError::illegal_argument("frequent_item_sets requires mergeable transaction state from every shard"))?)?;
        if merged.fields.is_empty() { merged.fields = state.fields.clone(); }
        if merged.fields != state.fields { return Err(EsError::illegal_argument("incompatible frequent_item_sets fields across shards")); }
        merged.total += state.total;
        merged.filtered += state.filtered;
        merged.profile |= state.profile;
        let mut remap = Vec::with_capacity(state.items.len());
        for (fi, key, value) in state.items {
            if fi >= merged.fields.len() { return Err(EsError::illegal_argument("invalid frequent_item_sets field ordinal")); }
            let id = *ids.entry((fi, format!("{key:?}"))).or_insert_with(|| {
                let id = merged.items.len();
                merged.items.push((fi, key, value));
                id
            });
            remap.push(id);
        }
        for (tr, count) in state.transactions {
            let mut mapped = tr.into_iter().map(|i| remap.get(i).copied().ok_or_else(|| EsError::illegal_argument("invalid frequent_item_sets item ordinal"))).collect::<Result<Vec<_>>>()?;
            mapped.sort_unstable();
            mapped.dedup();
            *transactions.entry(mapped).or_default() += count;
        }
    }
    merged.transactions = transactions.into_iter().collect();
    if !final_reduce { return Ok(json!({"buckets": [], "__p": merged})); }
    finish(merged, o)
}

fn finish(state: State, o: &Map<String, Value>) -> Result<Value> {
    let total = state.total;
    let min_support = o.get("minimum_support").and_then(Value::as_f64).unwrap_or(0.1);
    let min_size = o.get("minimum_set_size").and_then(Value::as_u64).unwrap_or(1);
    let size = o.get("size").and_then(Value::as_u64).unwrap_or(10);
    let min_count = ((total as f64 * min_support).ceil() as u64).max(1);
    let mut items: Vec<_> = state.items.iter().enumerate().map(|(i, (fi, key, _))| (*fi, key.clone(), 0u64, i)).collect();
    for (tr, count) in &state.transactions {
        for &i in tr { items[i].2 += count; }
    }
    // ES prunes items seen at most floor(support * total) times, then
    // orders the rest by count (ties: ES uses hash order; field then
    // value order here, which agrees on the common cases).
    let prune_min = (min_support * total as f64) as u64;
    let mut order: Vec<usize> = (0..items.len()).filter(|i| items[*i].2 as u64 > prune_min).collect();
    order.sort_by(|a, b| items[*b].2.cmp(&items[*a].2).then(items[*a].0.cmp(&items[*b].0)).then_with(|| super::super::cmp_key(&items[*a].1, &items[*b].1)));
    let mut remap = vec![usize::MAX; items.len()];
    for (new, old) in order.iter().enumerate() {
        remap[*old] = new;
    }
    let mut merged: HashMap<Vec<usize>, u64> = HashMap::new();
    let mut first: Vec<Vec<usize>> = Vec::new();
    for (t, multiplicity) in &state.transactions {
        let mut v: Vec<usize> = t.iter().map(|i| remap[*i]).filter(|x| *x != usize::MAX).collect();
        v.sort();
        v.dedup();
        if v.is_empty() {
            continue;
        }
        let e = merged.entry(v.clone()).or_insert(0);
        if *e == 0 {
            first.push(v);
        }
        *e += multiplicity;
    }
    let mut trans: Vec<(Vec<usize>, u64)> = first.into_iter().map(|v| {
        let c = merged[&v];
        (v, c)
    }).collect();
    trans.sort_by(|a, b| b.1.cmp(&a.1));
    let item_counts: Vec<u64> = order.iter().map(|i| items[*i].2 as u64).collect();
    let sets = if order.is_empty() { Vec::new() } else { eclat(order.len(), &item_counts, &trans, min_count, min_size as usize, size as usize)? };
    let mut buckets = Vec::new();
    for (set, c) in sets {
        let mut per_field: Vec<(String, Vec<Value>)> = Vec::new();
        for id in set {
            let (fi, _, _, _) = &items[order[id]];
            let fname = &state.fields[*fi];
            let v = state.items[order[id]].2.clone();
            match per_field.iter_mut().find(|(n, _)| n == fname) {
                Some((_, vs)) => vs.push(v),
                None => per_field.push((fname.clone(), vec![v])),
            }
        }
        per_field.sort_by(|a, b| a.0.cmp(&b.0));
        let key: Map<String, Value> = per_field.into_iter().map(|(k, v)| (k, Value::Array(v))).collect();
        buckets.push(json!({
            "key": key,
            "doc_count": c,
            "support": crate::values::java_double(c as f64 / total.max(1) as f64),
        }));
    }
    if state.profile {
        let total_items: u64 = state.transactions.iter().map(|(t, n)| t.len() as u64 * n).sum();
        let pruned_items: u64 = trans.iter().map(|(v, c)| v.len() as u64 * c).sum();
        return Ok(json!({"buckets": buckets, "profile": {
            "total_items_after_reduce": total_items,
            "total_transactions_after_reduce": total,
            "filtered_transactions_after_reduce": state.filtered,
            "unique_items_after_reduce": items.len(),
            "unique_transactions_after_reduce": state.transactions.len(),
            "total_items_after_prune": pruned_items,
            "total_transactions_after_prune": total,
            "filtered_transactions_after_prune": state.filtered,
            "unique_items_after_prune": order.len(),
            "unique_transactions_after_prune": trans.len(),
            "start_min_count_eclat": min_count
        }}));
    }
    Ok(json!({"buckets": buckets}))
}

/// Whether the aggregation tree contains a transaction reducer.
pub fn present(spec: &Value) -> bool {
    spec.as_object().is_some_and(|m| m.values().any(|v| v.get("frequent_item_sets").is_some()
        || v.get("frequent_items").is_some()
        || v.get("aggs").or_else(|| v.get("aggregations")).is_some_and(present)))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn part(rows: &[(&[usize], u64)], total: u64) -> Value {
        json!({"buckets": [], "__p": State {
            total, filtered: 0, fields: vec!["tags".into()], profile: true,
            items: ["a", "b", "c"].into_iter().map(|s| (0, Key::Str(s.into()), json!(s))).collect(),
            transactions: rows.iter().map(|(t, n)| (t.to_vec(), *n)).collect(),
        }})
    }
    #[test]
    fn frequent_items_global_threshold_and_intermediate_state() {
        // a is below support on shard one, but above it globally. Its
        // occurrence on shard one must not disappear during partial reduce.
        let a = part(&[(&[0], 1), (&[1, 2], 9)], 10);
        let b = part(&[(&[0], 10)], 10);
        let cfg = json!({"minimum_support": 0.5, "size": 1});
        let cfg = cfg.as_object().unwrap();
        let node = reduce(cfg, &[&a], false).unwrap();
        let result = reduce(cfg, &[&node, &b], true).unwrap();
        assert_eq!(result["buckets"], json!([{"key":{"tags":["a"]},"doc_count":11,"support":0.55}]));
        assert_eq!(result["profile"]["total_transactions_after_reduce"], 20);
        assert!(result.get("__p").is_none());
        assert_eq!(result, reduce(cfg, &[&b, &a], true).unwrap());
    }
    #[test]
    fn frequent_items_refuses_nonmergeable_partial() {
        assert!(reduce(&Map::new(), &[&json!({"buckets": []})], true).is_err());
    }
}
