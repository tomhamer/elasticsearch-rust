//! Partial `STATS` states for shards read on other nodes.
//!
//! A `FROM ... | WHERE ... | STATS aggs BY keys` whose aggregates are all
//! decomposable (`COUNT`, `SUM`, `MIN`, `MAX`, `AVG`, `COUNT_DISTINCT` over
//! plain columns, grouped by plain columns) does not ship rows: every shard
//! groups its own rows into [`PartialGroup`]s ([`partial_groups`]), and the
//! node running the query merges them ([`merge_groups`]) and finishes each
//! state ([`AggState::finish`]). The result is the one `STATS` computes over
//! all rows in shard order: groups appear in first-seen order (shards are
//! merged in search order), `MIN` keeps the first and `MAX` the last of
//! equal values. Numeric states are ES|QL's: long sums with `addExact`
//! (overflow reads as null, with a warning) and compensated double sums
//! ([`Kahan`]) whose intermediate value and delta merge like ES's
//! intermediate states.

use super::{cmp_vals, Val};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// A decomposable aggregate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggKind {
    /// `COUNT(*)` (no column) or `COUNT(col)`.
    Count,
    Sum,
    Min,
    Max,
    /// `AVG` of a double or long column (a compensated sum of doubles).
    Avg,
    /// `AVG` of an integer column (a long sum).
    AvgInt,
    CountDistinct,
}

impl AggKind {
    pub fn of(name: &str) -> Option<AggKind> {
        Some(match name {
            "COUNT" => AggKind::Count,
            "SUM" => AggKind::Sum,
            "MIN" => AggKind::Min,
            "MAX" => AggKind::Max,
            "AVG" => AggKind::Avg,
            "COUNT_DISTINCT" => AggKind::CountDistinct,
            _ => return None,
        })
    }
}

/// One aggregate of a pushed-down `STATS`: its kind and the column it reads
/// (`None`: `COUNT(*)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggSpec {
    pub kind: AggKind,
    pub col: Option<String>,
}

/// What a data node aggregates for a pushed-down `STATS`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggPlan {
    /// The `FROM` index patterns (the schema is resolved the same way there).
    pub indices: Vec<String>,
    /// `unmapped_fields=load`.
    #[serde(default)]
    pub unmapped_load: bool,
    /// Grouping columns.
    pub by: Vec<String>,
    pub aggs: Vec<AggSpec>,
    /// ES|QL type of every column read (by name), checked by the data node.
    pub types: Vec<(String, String)>,
}

/// A value on the wire (doubles as bits: JSON has no NaN or infinity).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WireVal {
    N,
    B(bool),
    I(i64),
    L(i64),
    D(u64),
    S(String),
    Dt(i64),
    Du(i64),
    P(i64, i64),
    M(Vec<WireVal>),
}

impl From<&Val> for WireVal {
    fn from(v: &Val) -> WireVal {
        match v {
            Val::Null => WireVal::N,
            Val::Bool(b) => WireVal::B(*b),
            Val::Int(i) => WireVal::I(*i),
            Val::Long(i) => WireVal::L(*i),
            Val::Double(d) => WireVal::D(d.to_bits()),
            Val::Str(s) => WireVal::S(s.clone()),
            Val::Date(d) => WireVal::Dt(*d),
            Val::Dur(d) => WireVal::Du(*d),
            Val::Period(m, d) => WireVal::P(*m, *d),
            Val::Multi(m) => WireVal::M(m.iter().map(WireVal::from).collect()),
        }
    }
}

impl From<&WireVal> for Val {
    fn from(v: &WireVal) -> Val {
        match v {
            WireVal::N => Val::Null,
            WireVal::B(b) => Val::Bool(*b),
            WireVal::I(i) => Val::Int(*i),
            WireVal::L(i) => Val::Long(*i),
            WireVal::D(d) => Val::Double(f64::from_bits(*d)),
            WireVal::S(s) => Val::Str(s.clone()),
            WireVal::Dt(d) => Val::Date(*d),
            WireVal::Du(d) => Val::Dur(*d),
            WireVal::P(m, d) => Val::Period(*m, *d),
            WireVal::M(m) => Val::Multi(m.iter().map(Val::from).collect()),
        }
    }
}

/// Serde for a [`Val`] field as its [`WireVal`] (doubles as bits).
pub(super) mod wire_val {
    use super::{Val, WireVal};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Val, s: S) -> Result<S::Ok, S::Error> {
        WireVal::from(v).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Val, D::Error> {
        Ok(Val::from(&WireVal::deserialize(d)?))
    }
}

/// Serde for named [`Val`]s (map literals) as [`WireVal`]s.
pub(super) mod wire_pairs {
    use super::{Val, WireVal};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &[(String, Val)], s: S) -> Result<S::Ok, S::Error> {
        v.iter().map(|(k, x)| (k.clone(), WireVal::from(x))).collect::<Vec<_>>().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<(String, Val)>, D::Error> {
        Ok(Vec::<(String, WireVal)>::deserialize(d)?.iter().map(|(k, x)| (k.clone(), Val::from(x))).collect())
    }
}

/// ES|QL's compensated (Kahan) double sum (`CompensatedSum`,
/// `SumDoubleAggregator`): a running value and its correction. Partial
/// sums merge by adding another state's value and delta.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Kahan {
    pub value: f64,
    pub delta: f64,
}

impl Kahan {
    /// `CompensatedSum.add(value, delta)`.
    pub fn add(&mut self, value: f64, delta: f64) {
        // Infinity and NaN join the running value and end the summation.
        if !value.is_finite() {
            self.value += value;
        }
        if self.value.is_finite() {
            let corrected = value + (self.delta + delta);
            let updated = self.value + corrected;
            self.delta = corrected - (updated - self.value);
            self.value = updated;
        }
    }

    /// The compensated sum of `xs` as ES|QL reports it: one driver's state
    /// combined into the final (initially empty) state, which folds the
    /// delta into the value.
    pub fn sum(xs: impl IntoIterator<Item = f64>) -> f64 {
        let mut k = Kahan::default();
        for x in xs {
            k.add(x, 0.0);
        }
        let mut last = Kahan::default();
        last.add(k.value, k.delta);
        last.value
    }

    fn wire(&self) -> (u64, u64) {
        (self.value.to_bits(), self.delta.to_bits())
    }

    fn from_wire(w: (u64, u64)) -> Kahan {
        Kahan { value: f64::from_bits(w.0), delta: f64::from_bits(w.1) }
    }
}

/// `Math.addExact` over a running long sum: `None` once it overflowed
/// (ES|QL then reads the sum as null, with a warning).
fn add_exact(sum: Option<i64>, v: i64) -> Option<i64> {
    sum.and_then(|s| s.checked_add(v))
}

/// One aggregate's partial state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AggState {
    /// Rows (`COUNT(*)`) or non-null values.
    Count(i64),
    /// Values seen, whether all were integers, their long sum (`None`:
    /// overflowed) and their compensated double sum (value, delta bits).
    Sum { n: i64, ints: bool, lsum: Option<i64>, kahan: (u64, u64) },
    Min(Option<WireVal>),
    Max(Option<WireVal>),
    /// Numeric values seen, their long sum (`AVG` of integers) and their
    /// compensated double sum.
    Avg { n: i64, lsum: Option<i64>, kahan: (u64, u64) },
    /// Distinct value keys ([`Val::key`]).
    Distinct(Vec<String>),
}

/// Accumulator for one aggregate in one group.
#[derive(Clone, Debug)]
pub(super) enum Acc {
    Count(i64),
    Sum { n: i64, ints: bool, lsum: Option<i64>, kahan: Kahan },
    Min(Option<Val>),
    Max(Option<Val>),
    Avg { n: i64, lsum: Option<i64>, kahan: Kahan },
    Distinct(HashSet<String>),
}

impl Acc {
    pub(super) fn new(kind: AggKind) -> Acc {
        match kind {
            AggKind::Count => Acc::Count(0),
            AggKind::Sum => Acc::Sum { n: 0, ints: true, lsum: Some(0), kahan: Kahan::default() },
            AggKind::Min => Acc::Min(None),
            AggKind::Max => Acc::Max(None),
            AggKind::Avg | AggKind::AvgInt => Acc::Avg { n: 0, lsum: Some(0), kahan: Kahan::default() },
            AggKind::CountDistinct => Acc::Distinct(HashSet::new()),
        }
    }

    /// Variable-width state retained by MIN/MAX (scalar input values).
    pub(super) fn retained_bytes(&self) -> usize {
        match self {
            Acc::Min(Some(Val::Str(s))) | Acc::Max(Some(Val::Str(s))) => s.capacity(),
            _ => 0,
        }
    }

    /// Add one non-null value.
    pub(super) fn add(&mut self, v: &Val) {
        match self {
            Acc::Count(c) => *c += 1,
            Acc::Sum { n, ints, lsum, kahan } => {
                *n += 1;
                match v {
                    Val::Int(i) | Val::Long(i) => *lsum = add_exact(*lsum, *i),
                    _ => *ints = false,
                }
                if let Some(x) = v.as_f64() {
                    kahan.add(x, 0.0);
                }
            }
            Acc::Min(m) => {
                // The first of equal minimums (`Iterator::min_by`).
                if m.as_ref().is_none_or(|cur| cmp_vals(v, cur) == Ordering::Less) {
                    *m = Some(v.clone());
                }
            }
            Acc::Max(m) => {
                // The last of equal maximums (`Iterator::max_by`).
                if m.as_ref().is_none_or(|cur| cmp_vals(v, cur) != Ordering::Less) {
                    *m = Some(v.clone());
                }
            }
            Acc::Avg { n, lsum, kahan } => {
                if let Some(x) = v.as_f64() {
                    *n += 1;
                    kahan.add(x, 0.0);
                    if let Some(i) = v.as_i64() {
                        *lsum = add_exact(*lsum, i);
                    }
                }
            }
            Acc::Distinct(s) => {
                s.insert(v.key());
            }
        }
    }

    pub(super) fn state(self) -> AggState {
        match self {
            Acc::Count(c) => AggState::Count(c),
            Acc::Sum { n, ints, lsum, kahan } => AggState::Sum { n, ints, lsum, kahan: kahan.wire() },
            Acc::Min(m) => AggState::Min(m.as_ref().map(WireVal::from)),
            Acc::Max(m) => AggState::Max(m.as_ref().map(WireVal::from)),
            Acc::Avg { n, lsum, kahan } => AggState::Avg { n, lsum, kahan: kahan.wire() },
            Acc::Distinct(s) => {
                let mut v: Vec<String> = s.into_iter().collect();
                v.sort();
                AggState::Distinct(v)
            }
        }
    }
}

impl AggState {
    /// Fold `other` (a later shard's state) into this one, the way ES|QL
    /// combines intermediate states: long sums with `addExact`, double sums
    /// by adding the other value and delta.
    fn merge(&mut self, other: AggState) {
        match (self, other) {
            (AggState::Count(a), AggState::Count(b)) => *a += b,
            (AggState::Sum { n, ints, lsum, kahan }, AggState::Sum { n: n2, ints: i2, lsum: l2, kahan: k2 }) => {
                if n2 == 0 {
                    return;
                }
                *n += n2;
                *ints &= i2;
                *lsum = lsum.zip(l2).and_then(|(a, b)| a.checked_add(b));
                let mut k = Kahan::from_wire(*kahan);
                let o = Kahan::from_wire(k2);
                k.add(o.value, o.delta);
                *kahan = k.wire();
            }
            (AggState::Min(a), AggState::Min(b)) => {
                if let Some(b) = b {
                    if a.as_ref().is_none_or(|cur| cmp_vals(&Val::from(&b), &Val::from(cur)) == Ordering::Less) {
                        *a = Some(b);
                    }
                }
            }
            (AggState::Max(a), AggState::Max(b)) => {
                if let Some(b) = b {
                    if a.as_ref().is_none_or(|cur| cmp_vals(&Val::from(&b), &Val::from(cur)) != Ordering::Less) {
                        *a = Some(b);
                    }
                }
            }
            (AggState::Avg { n, lsum, kahan }, AggState::Avg { n: n2, lsum: l2, kahan: k2 }) => {
                if n2 == 0 {
                    return;
                }
                *n += n2;
                *lsum = lsum.zip(l2).and_then(|(a, b)| a.checked_add(b));
                let mut k = Kahan::from_wire(*kahan);
                let o = Kahan::from_wire(k2);
                k.add(o.value, o.delta);
                *kahan = k.wire();
            }
            (AggState::Distinct(a), AggState::Distinct(b)) => {
                let mut s: HashSet<String> = a.drain(..).collect();
                s.extend(b);
                let mut v: Vec<String> = s.into_iter().collect();
                v.sort();
                *a = v;
            }
            _ => {}
        }
    }

    /// The aggregate's value, and whether a long sum overflowed (the value
    /// is then null and ES|QL warns).
    pub fn finish(&self, kind: AggKind) -> (Val, bool) {
        match self {
            AggState::Count(c) => (Val::Long(*c), false),
            AggState::Sum { n, ints, lsum, kahan } => {
                if *n == 0 {
                    (Val::Null, false)
                } else if *ints {
                    match lsum {
                        Some(s) => (Val::Long(*s), false),
                        None => (Val::Null, true),
                    }
                } else {
                    (Val::Double(Kahan::from_wire(*kahan).value), false)
                }
            }
            AggState::Min(m) | AggState::Max(m) => (m.as_ref().map(Val::from).unwrap_or(Val::Null), false),
            AggState::Avg { n, lsum, kahan } => {
                if *n == 0 {
                    (Val::Null, false)
                } else if kind == AggKind::AvgInt {
                    // `SUM(int)` (a long) divided by the count.
                    match lsum {
                        Some(s) => (Val::Double(*s as f64 / *n as f64), false),
                        None => (Val::Null, true),
                    }
                } else {
                    (Val::Double(Kahan::from_wire(*kahan).value / *n as f64), false)
                }
            }
            AggState::Distinct(s) => (Val::Long(s.len() as i64), false),
        }
    }
}

/// One group of one shard: its key and aggregate states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartialGroup {
    pub key: Vec<WireVal>,
    pub states: Vec<AggState>,
}

/// Group `rows` by the columns `by` (a multi-valued key puts the row in a
/// group per value, like `STATS`) and aggregate. `aggs` pairs each kind
/// with the column it reads (`None`: count rows). Without `by`, there is
/// one group even for no rows.
pub fn partial_groups(rows: &[Vec<Val>], by: &[usize], aggs: &[(AggKind, Option<usize>)]) -> Vec<PartialGroup> {
    let mut order: Vec<(Vec<Val>, Vec<Acc>)> = Vec::new();
    let mut index: HashMap<Vec<String>, usize> = HashMap::new();
    let fresh = || aggs.iter().map(|(k, _)| Acc::new(*k)).collect::<Vec<_>>();
    if by.is_empty() {
        order.push((Vec::new(), fresh()));
    }
    for r in rows {
        let groups: Vec<usize> = if by.is_empty() {
            vec![0]
        } else {
            let mut combos: Vec<Vec<Val>> = vec![vec![]];
            for &g in by {
                let opts: Vec<Val> = match &r[g] {
                    Val::Multi(m) => m.clone(),
                    v => vec![v.clone()],
                };
                let mut next = Vec::with_capacity(combos.len() * opts.len());
                for c in &combos {
                    for o in &opts {
                        let mut c2 = c.clone();
                        c2.push(o.clone());
                        next.push(c2);
                    }
                }
                combos = next;
            }
            combos
                .into_iter()
                .map(|c| {
                    let k: Vec<String> = c.iter().map(Val::key).collect();
                    *index.entry(k).or_insert_with(|| {
                        order.push((c, fresh()));
                        order.len() - 1
                    })
                })
                .collect()
        };
        for gi in groups {
            for (acc, (kind, col)) in order[gi].1.iter_mut().zip(aggs) {
                match col {
                    None => acc.add(&Val::Null),
                    Some(c) => match &r[*c] {
                        Val::Null => {}
                        Val::Multi(m) => m.iter().filter(|x| !x.is_null()).for_each(|x| acc.add(x)),
                        v => acc.add(v),
                    },
                }
                let _ = kind;
            }
        }
    }
    order.into_iter().map(|(key, accs)| PartialGroup { key: key.iter().map(WireVal::from).collect(), states: accs.into_iter().map(Acc::state).collect() }).collect()
}

/// Merge per-shard groups given in shard (search) order: groups keep their
/// first-seen order. Without grouping (`grouped == false`) the result is
/// one group, even when no shard returned one.
pub fn merge_groups(shards: Vec<Vec<PartialGroup>>, grouped: bool, aggs: &[AggKind]) -> Vec<PartialGroup> {
    let mut out: Vec<PartialGroup> = Vec::new();
    let mut index: HashMap<Vec<String>, usize> = HashMap::new();
    if !grouped {
        out.push(PartialGroup { key: Vec::new(), states: aggs.iter().map(|k| Acc::new(*k).state()).collect() });
    }
    for groups in shards {
        for g in groups {
            let k: Vec<String> = g.key.iter().map(|w| Val::from(w).key()).collect();
            let gi = if grouped {
                match index.get(&k) {
                    Some(i) => *i,
                    None => {
                        index.insert(k, out.len());
                        out.push(PartialGroup { key: g.key.clone(), states: aggs.iter().map(|k| Acc::new(*k).state()).collect() });
                        out.len() - 1
                    }
                }
            } else {
                0
            };
            for (s, o) in out[gi].states.iter_mut().zip(g.states) {
                s.merge(o);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esql::exec::{run_on_table, Column, Table};
    use crate::esql::parse::{parse_query, Params};

    /// Deterministic pseudo-random numbers (xorshift).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn table(rng: &mut Rng, n: usize) -> Table {
        let cols = vec![
            Column { name: "k".into(), ty: "keyword".into() },
            Column { name: "j".into(), ty: "integer".into() },
            Column { name: "v".into(), ty: "long".into() },
            Column { name: "d".into(), ty: "double".into() },
            Column { name: "s".into(), ty: "keyword".into() },
        ];
        let mut rows = Vec::new();
        for _ in 0..n {
            let k = match rng.below(6) {
                0 => Val::Null,
                1 => Val::Multi(vec![Val::Str("a".into()), Val::Str("c".into())]),
                x => Val::Str(["a", "b", "c", "d"][(x - 2) as usize].into()),
            };
            let j = if rng.below(9) == 0 { Val::Null } else { Val::Int(rng.below(3) as i64) };
            let v = match rng.below(7) {
                0 => Val::Null,
                1 => Val::Multi(vec![Val::Long(rng.below(100) as i64), Val::Long(-(rng.below(100) as i64))]),
                _ => Val::Long(rng.below(1_000_000) as i64 - 500_000),
            };
                    let d = match rng.below(5) {
                0 => Val::Null,
                _ => Val::Double((rng.below(1_000_000) as f64) / 7.0),
            };
            let s = if rng.below(4) == 0 { Val::Null } else { Val::Str(format!("s{}", rng.below(40))) };
            rows.push(vec![k, j, v, d, s]);
        }
        Table { cols, rows }
    }

    const QUERIES: &[&str] = &[
        "STATS c = COUNT(*), cv = COUNT(v), s = SUM(v), mn = MIN(v), mx = MAX(s), a = AVG(v), cd = COUNT_DISTINCT(s) BY k",
        "STATS c = COUNT(*), sd = SUM(d), ad = AVG(d), mn = MIN(d), mx = MAX(d), aj = AVG(j), sj = SUM(j) BY k, j",
        "STATS c = COUNT(*), s = SUM(v), a = AVG(d), cd = COUNT_DISTINCT(k)",
    ];

    fn plan_of(q: &str) -> (Vec<String>, Vec<(String, AggKind, Option<String>)>) {
        let cmds = parse_query(&format!("ROW x = 1 | {q}"), &Params::default()).unwrap();
        let crate::esql::Cmd::Stats { aggs, by, .. } = &cmds[1] else { panic!() };
        let by = by.iter().map(|f| f.name.clone()).collect();
        let aggs = aggs
            .iter()
            .map(|a| match &a.expr {
                crate::esql::Expr::Call(n, args, _) => (
                    a.name.clone(),
                    match (AggKind::of(n).unwrap(), &args[0]) {
                        // `j` is the integer column.
                        (AggKind::Avg, crate::esql::Expr::Col(c, _)) if c == "j" => AggKind::AvgInt,
                        (k, _) => k,
                    },
                    match &args[0] {
                        crate::esql::Expr::Col(c, _) => Some(c.clone()),
                        _ => None,
                    },
                ),
                _ => panic!(),
            })
            .collect();
        (by, aggs)
    }

    /// Values equal, doubles (compensated sums merged per shard, as ES
    /// merges intermediate states) to within rounding.
    fn same(a: &[Vec<Val>], b: &[Vec<Val>]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| {
                x.len() == y.len()
                    && x.iter().zip(y).all(|(u, v)| match (u, v) {
                        (Val::Double(p), Val::Double(q)) => p == q || ((p - q).abs() <= 1e-9 * p.abs().max(q.abs())),
                        _ => u == v,
                    })
            })
    }

    fn finish_all(merged: &[PartialGroup], kinds: &[AggKind]) -> Vec<Vec<Val>> {
        merged.iter().map(|g| g.states.iter().zip(kinds).map(|(s, k)| s.finish(*k).0).chain(g.key.iter().map(Val::from)).collect()).collect()
    }

    /// Partial states over any split of the rows into shards equal `STATS`
    /// over all of them, values and group order.
    #[test]
    fn partial_stats_over_random_splits_equal_stats_over_all_rows() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for q in QUERIES {
            for round in 0..20 {
                let n = 1 + rng.below(300) as usize;
                let t = table(&mut rng, n);
                let cmds = parse_query(&format!("ROW x = 1 | {q}"), &Params::default()).unwrap();
                let want = run_on_table(q, &cmds[1..], t.clone()).unwrap();
                let (by, aggs) = plan_of(q);
                let idx = |n: &str| t.cols.iter().position(|c| c.name == n).unwrap();
                let by_idx: Vec<usize> = by.iter().map(|b| idx(b)).collect();
                let agg_idx: Vec<(AggKind, Option<usize>)> = aggs.iter().map(|(_, k, c)| (*k, c.as_deref().map(idx))).collect();
                // Cut the rows into 1..=5 contiguous shards (shard order = row order).
                let nshards = 1 + rng.below(5) as usize;
                let mut cuts: Vec<usize> = (0..nshards - 1).map(|_| rng.below(t.rows.len() as u64 + 1) as usize).collect();
                cuts.push(0);
                cuts.push(t.rows.len());
                cuts.sort();
                let shards: Vec<Vec<PartialGroup>> = cuts.windows(2).map(|w| partial_groups(&t.rows[w[0]..w[1]], &by_idx, &agg_idx)).collect();
                let kinds: Vec<AggKind> = aggs.iter().map(|a| a.1).collect();
                let merged = merge_groups(shards, !by.is_empty(), &kinds);
                let got = finish_all(&merged, &kinds);
                assert!(same(&got, &want.rows), "{q} round {round} shards {nshards}:\n{got:?}\n{:?}", want.rows);
                // One shard: the same sequential sums, bit for bit.
                let one = merge_groups(vec![partial_groups(&t.rows, &by_idx, &agg_idx)], !by.is_empty(), &kinds);
                assert_eq!(finish_all(&one, &kinds), want.rows, "{q} round {round} one shard");
            }
        }
    }

    #[test]
    fn no_rows_without_grouping_is_one_row() {
        let merged = merge_groups(vec![], false, &[AggKind::Count, AggKind::Sum, AggKind::Avg, AggKind::Min, AggKind::CountDistinct]);
        let kinds = [AggKind::Count, AggKind::Sum, AggKind::Avg, AggKind::Min, AggKind::CountDistinct];
        let vals: Vec<Val> = merged[0].states.iter().zip(kinds).map(|(s, k)| s.finish(k).0).collect();
        assert_eq!(vals, vec![Val::Long(0), Val::Null, Val::Null, Val::Null, Val::Long(0)]);
        assert!(merge_groups(vec![vec![], vec![]], true, &[AggKind::Count]).is_empty());
    }

    #[test]
    fn compensated_sums_follow_es() {
        // CompensatedSum: 0.1 ten times is exactly 1.0 (a plain sum gives 0.9999999999999999).
        assert_eq!(Kahan::sum([0.1; 10]), 1.0);
        assert_eq!([0.1f64; 10].iter().sum::<f64>(), 0.9999999999999999);
        assert!(Kahan::sum([f64::INFINITY, 1.0]).is_infinite());
        assert!(Kahan::sum([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert_eq!(Kahan::sum([1e308, 1e308, -1e308]), f64::INFINITY);
        assert_eq!(Kahan::sum([]), 0.0);
        // Shards merge as ES merges intermediate states: value + delta.
        let (a, b) = ([1e16, 1.0, 1.0, 3.3], [-1e16, 0.7, 1e-3]);
        let mut ka = Kahan::default();
        a.iter().for_each(|x| ka.add(*x, 0.0));
        let mut kb = Kahan::default();
        b.iter().for_each(|x| kb.add(*x, 0.0));
        let mut fin = Kahan::default();
        fin.add(ka.value, ka.delta);
        fin.add(kb.value, kb.delta);
        let rows: Vec<Vec<Val>> = a.iter().chain(b.iter()).map(|x| vec![Val::Double(*x)]).collect();
        let shards = vec![partial_groups(&rows[..4], &[], &[(AggKind::Sum, Some(0))]), partial_groups(&rows[4..], &[], &[(AggKind::Sum, Some(0))])];
        let merged = merge_groups(shards, false, &[AggKind::Sum]);
        assert_eq!(merged[0].states[0].finish(AggKind::Sum).0, Val::Double(fin.value));
    }

    #[test]
    fn long_overflow_part_way_through_reads_as_null_like_es() {
        let rows = vec![vec![Val::Long(i64::MAX)], vec![Val::Long(1)], vec![Val::Long(-1)]];
        let sum = |shards: &[std::ops::Range<usize>]| {
            let parts = shards.iter().map(|r| partial_groups(&rows[r.clone()], &[], &[(AggKind::Sum, Some(0))])).collect();
            merge_groups(parts, false, &[AggKind::Sum])[0].states[0].finish(AggKind::Sum)
        };
        // One shard: `addExact` overflows at the second value, although the total fits.
        assert_eq!(sum(&[0..3]), (Val::Null, true));
        // Split [MAX] | [1, -1]: no partial sum overflows (as ES with that split).
        assert_eq!(sum(&[0..1, 1..3]), (Val::Long(i64::MAX), false));
        // A total that does not fit overflows in the merge.
        let big = vec![vec![Val::Long(i64::MAX)], vec![Val::Long(i64::MAX)]];
        let parts = vec![partial_groups(&big[..1], &[], &[(AggKind::Sum, Some(0))]), partial_groups(&big[1..], &[], &[(AggKind::Sum, Some(0))])];
        assert_eq!(merge_groups(parts, false, &[AggKind::Sum])[0].states[0].finish(AggKind::Sum), (Val::Null, true));
        // AVG of integers is a long sum too.
        let ints = vec![vec![Val::Int(3)], vec![Val::Int(4)]];
        assert_eq!(partial_groups(&ints, &[], &[(AggKind::AvgInt, Some(0))])[0].states[0].finish(AggKind::AvgInt), (Val::Double(3.5), false));
    }

    #[test]
    fn single_node_sum_overflow_is_null_with_es_warnings() {
        use crate::esql::exec::{run_cmds, Ctx};
        let q = "ROW x = [9223372036854775807, 1, -1] | MV_EXPAND x | STATS s = SUM(x), n = COUNT(*)";
        let cmds = parse_query(q, &Params::default()).unwrap();
        let mut ctx = crate::esql::exec::new_ctx_local(q);
        let t = run_cmds(&mut ctx, &cmds).unwrap();
        assert_eq!(t.rows, vec![vec![Val::Null, Val::Long(3)]]);
        let w = ctx.fctx.warnings.borrow().list.clone();
        assert_eq!(w, vec![
            "Line 1:64: evaluation of [SUM(x)] failed, treating result as null. Only first 20 failures recorded.".to_string(),
            "Line 1:64: java.lang.ArithmeticException: long overflow".to_string(),
        ]);
        let _: fn(&mut Ctx, &[crate::esql::Cmd]) -> crate::error::Result<Table> = run_cmds;
    }

    #[test]
    fn composite_groups_preserve_boundaries_across_shards() {
        let a = vec![Val::Str("x\u{2}sy".into()), Val::Str("z".into())];
        let b = vec![Val::Str("x".into()), Val::Str("y\u{2}sz".into())];
        let rows = vec![a.clone(), b.clone(), a];
        let aggs = [(AggKind::Count, None)];
        let single = partial_groups(&rows, &[0, 1], &aggs);
        let merged = merge_groups(vec![
            partial_groups(&rows[..1], &[0, 1], &aggs),
            partial_groups(&rows[1..], &[0, 1], &aggs),
        ], true, &[AggKind::Count]);
        assert_eq!(merged, single);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].states[0].finish(AggKind::Count).0, Val::Long(2));
        assert_eq!(merged[1].states[0].finish(AggKind::Count).0, Val::Long(1));
    }

    #[test]
    fn states_round_trip_as_json() {
        let g = partial_groups(&[vec![Val::Double(f64::NAN), Val::Str("x".into())]], &[1], &[(AggKind::Max, Some(0)), (AggKind::Sum, Some(0))]);
        let back: Vec<PartialGroup> = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert!(matches!(back[0].states[0].finish(AggKind::Max).0, Val::Double(x) if x.is_nan()));
        assert!(matches!(back[0].states[1].finish(AggKind::Sum).0, Val::Double(x) if x.is_nan()));
        assert_eq!(back[0].key, g[0].key);
    }
}
