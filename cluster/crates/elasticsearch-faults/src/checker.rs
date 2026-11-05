//! Checkers for the guarantees in `docs/CONSISTENCY.md`, run on the
//! recorded history + the final state of every shard copy + the monitor's
//! observations of every node's applied cluster state.

use crate::cluster::Cluster;
use crate::history::{Kind, Op, Outcome};
use crate::monitor::Obs;
use crate::nemesis::{Event, Window};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Doc {
    pub seq_no: i64,
    pub version: i64,
    pub value: Option<i64>,
}

/// One shard copy's final contents (keys we ever touched) + its doc count.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Copy {
    pub node: usize,
    pub index: String,
    pub primary: bool,
    pub docs: BTreeMap<String, Doc>,
    pub count: Option<u64>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Violation {
    pub guarantee: String,
    pub kind: String,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<(String, String)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Counts {
    pub attempted: usize,
    pub ok: usize,
    pub fail: usize,
    pub info: usize,
    pub by_kind: BTreeMap<String, [usize; 3]>,
    pub writes_acked: usize,
    pub lost: usize,
    pub info_writes_present: usize,
    pub info_writes_absent: usize,
    pub reads_ok: usize,
    pub stale_reads_in_fault: usize,
    pub minority_writes: usize,
    pub minority_writes_rejected_block: usize,
    /// Minority-side writes acknowledged while that side held a quorum of
    /// the voting configuration (legitimate: see `side_voting_quorum`).
    pub minority_writes_with_voting_quorum: usize,
    pub minority_write_outcomes: BTreeMap<String, usize>,
    pub paused_node_acks_after_resume: usize,
    /// Per partition with a minority: seconds from partition start to the
    /// first write rejected by a minority node with cluster_block_exception.
    pub minority_block_latency: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Election {
    pub term: i64,
    pub master: String,
    pub first_seen: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Verdict {
    pub counts: Counts,
    pub violations: Vec<Violation>,
    pub warnings: Vec<Violation>,
    pub elections: Vec<Election>,
    /// Seconds from a fault on the master to the first observation of a
    /// new master (new term).
    pub failover_latencies: Vec<f64>,
    pub copies_compared: usize,
    pub guarantees: BTreeMap<String, String>,
}

pub struct Input<'a> {
    pub ops: &'a [Op],
    pub copies: &'a [Copy],
    pub obs: &'a [Obs],
    pub samples: &'a BTreeMap<(i64, u64), String>,
    pub events: &'a [Event],
    pub windows: &'a [Window],
    /// Fault-detection grace after a partition starts before the minority
    /// must reject writes (and after a fault ends before reads must be fresh).
    pub grace: f64,
    pub converged: Result<f64, String>,
    pub crashes: &'a [(usize, f64, String)],
}

fn v(g: &str, kind: &str, detail: String, key: Option<(&str, &str)>, t: Option<f64>) -> Violation {
    Violation { guarantee: g.into(), kind: kind.into(), detail, key: key.map(|(a, b)| (a.to_string(), b.to_string())), t }
}

fn is_write(k: Kind) -> bool {
    matches!(k, Kind::Init | Kind::Create | Kind::BulkCreate | Kind::Cas | Kind::Delete)
}

fn op_brief(o: &Op) -> String {
    format!(
        "#{} w{} n{} {:?} {}/{} v={:?}{} [{:.3}-{:.3}] {:?} {}{}",
        o.id,
        o.worker,
        o.node + 1,
        o.kind,
        o.index,
        o.key,
        o.value,
        o.if_seq_no.map(|s| format!(" if_seq_no={s} if_term={}", o.if_primary_term.unwrap_or(-1))).unwrap_or_default(),
        o.t_invoke,
        o.t_complete,
        o.outcome,
        o.seq_no.map(|s| format!("seq={s} term={} ", o.primary_term.unwrap_or(-1))).unwrap_or_default(),
        o.error.clone().unwrap_or_default()
    )
}

/// Whether the node that acknowledged `o` was, at that moment, on a side of
/// partition `w` that holds a quorum of the voting configuration.
///
/// A partition's `minority` is counted in nodes, but Elasticsearch commits
/// with a quorum of the *voting configuration*, which need not be all the
/// nodes: with `cluster.auto_shrink_voting_configuration` (default true) a
/// master that removes two of five nodes moves to a 3-node configuration,
/// and grows it again only once they rejoin. If the next partition starts
/// before they do, the "minority" by node count can hold the voting quorum
/// (and the "majority" cannot elect): that side keeps a legitimate master
/// and acknowledges writes. Such an ack is a G8 violation only if the
/// acknowledging node's master could not reach a quorum of both the
/// committed and the accepted configuration (an unreachable voter, a
/// bootstrap placeholder or an unknown id counts against the quorum).
///
/// The state is the acknowledging node's latest applied state observed by
/// the monitor before the ack completed or, since the monitor polls every
/// few hundred milliseconds, the first one observed within
/// [`ACK_STATE_LAG`] after it: a write held by the no-master block runs as
/// soon as the node applies the state naming its new master, and can be
/// acknowledged before the next poll sees that state (an elected master
/// applies its own first state only when that publication completes, up to
/// `cluster.publish.timeout` after its followers). Returns a short
/// description when the side holds the quorum.
fn side_voting_quorum(inp: &Input, w: &Window, o: &Op) -> Option<String> {
    let before = inp.obs.iter().filter(|b| b.node == o.node && b.t <= o.t_complete).last();
    let after = inp.obs.iter().find(|b| b.node == o.node && b.t > o.t_complete && b.t <= o.t_complete + ACK_STATE_LAG);
    before.into_iter().chain(after).find_map(|ob| state_holds_voting_quorum(w, o, ob))
}

/// Seconds after an ack within which the monitor's first observation of the
/// acknowledging node still stands for the state the write ran under.
const ACK_STATE_LAG: f64 = 1.0;

fn state_holds_voting_quorum(w: &Window, o: &Op, ob: &Obs) -> Option<String> {
    let master = Cluster::index_of_name(ob.master.as_deref()?)?;
    let linked = |a: usize, b: usize| a == b || !w.drops.iter().any(|&(x, y)| (x == a && y == b) || (x == b && y == a));
    // Without the dropped links (older artifacts), the side is the minority.
    let with_master = |b: usize| if w.drops.is_empty() { w.minority.contains(&b) } else { linked(master, b) };
    if !with_master(o.node) || (w.drops.is_empty() && !w.minority.contains(&master)) {
        return None;
    }
    let quorum = |cfg: &[String]| !cfg.is_empty() && 2 * cfg.iter().filter(|n| Cluster::index_of_name(n).is_some_and(|i| with_master(i))).count() > cfg.len();
    if !quorum(&ob.committed_config) || !quorum(&ob.accepted_config) {
        return None;
    }
    Some(format!("master n{} with voting configuration {:?} (observed at {:.2}s)", master + 1, ob.committed_config, ob.t))
}

pub fn check(inp: &Input) -> Verdict {
    let mut out = Verdict::default();
    let mut viol: Vec<Violation> = Vec::new();
    let mut warn: Vec<Violation> = Vec::new();

    // ---------------------------------------------------------- counts
    let c = &mut out.counts;
    for o in inp.ops.iter().filter(|o| o.kind != Kind::Init) {
        c.attempted += 1;
        let e = c.by_kind.entry(format!("{:?}", o.kind)).or_insert([0; 3]);
        match o.outcome {
            Outcome::Ok => {
                c.ok += 1;
                e[0] += 1;
            }
            Outcome::Fail => {
                c.fail += 1;
                e[1] += 1;
            }
            _ => {
                c.info += 1;
                e[2] += 1;
            }
        }
        if is_write(o.kind) && o.outcome == Outcome::Ok {
            c.writes_acked += 1;
        }
    }

    // --------------------------------------------- G5: copies identical
    let mut finals: BTreeMap<String, &Copy> = BTreeMap::new();
    let mut by_index: BTreeMap<&str, Vec<&Copy>> = BTreeMap::new();
    for cp in inp.copies {
        by_index.entry(cp.index.as_str()).or_default().push(cp);
    }
    for (ix, copies) in &by_index {
        out.copies_compared += copies.len();
        for cp in copies {
            if let Some(e) = &cp.error {
                viol.push(v("G5", "copy-unreadable", format!("{ix} copy on n{}: {e}", cp.node + 1), None, None));
            }
        }
        let good: Vec<&&Copy> = copies.iter().filter(|c| c.error.is_none()).collect();
        let Some(reference) = good.iter().find(|c| c.primary).or(good.first()) else {
            viol.push(v("G5", "no-readable-copy", format!("no readable copy of {ix}"), None, None));
            continue;
        };
        finals.insert(ix.to_string(), reference);
        let keys: BTreeSet<&String> = good.iter().flat_map(|c| c.docs.keys()).collect();
        let mut diverged = 0;
        for k in keys {
            let vals: Vec<Option<&Doc>> = good.iter().map(|c| c.docs.get(k)).collect();
            if vals.windows(2).any(|w| w[0] != w[1]) {
                diverged += 1;
                if diverged <= 20 {
                    let d: Vec<String> = good.iter().map(|c| format!("n{}{}={:?}", c.node + 1, if c.primary { "(p)" } else { "" }, c.docs.get(k))).collect();
                    viol.push(v("G5", "copies-diverge", format!("{ix}/{k}: {}", d.join(" ")), Some((ix, k)), None));
                }
            }
        }
        if diverged > 20 {
            viol.push(v("G5", "copies-diverge", format!("{ix}: {} more diverging keys", diverged - 20), None, None));
        }
        for cp in &good {
            if let Some(cnt) = cp.count {
                if cnt != cp.docs.len() as u64 {
                    viol.push(v("G5", "unknown-docs", format!("{ix} copy on n{} counts {cnt} docs but holds {} of the ids the clients used", cp.node + 1, cp.docs.len()), None, None));
                }
            }
        }
    }

    // ------------------------------------------- per-key write checks (G3)
    let mut by_key: BTreeMap<(&str, &str), Vec<&Op>> = BTreeMap::new();
    for o in inp.ops {
        by_key.entry((o.index.as_str(), o.key.as_str())).or_default().push(o);
    }
    let final_doc = |ix: &str, key: &str| -> Option<Option<&Doc>> { finals.get(ix).map(|c| c.docs.get(key)) };

    for ((ix, key), ops) in &by_key {
        let Some(fin) = final_doc(ix, key) else { continue };
        let writes: Vec<&&Op> = ops.iter().filter(|o| is_write(o.kind)).collect();
        let excerpt = || ops.iter().map(|o| op_brief(o)).collect::<Vec<_>>().join("\n      ");
        let value_op = |val: i64| writes.iter().find(|o| o.value == Some(val) && o.kind != Kind::Delete);
        // The final value must have been written by someone, and not by a
        // write that definitely failed.
        if let Some(d) = fin {
            match d.value.and_then(value_op) {
                None => viol.push(v("G3", "phantom-value", format!("{ix}/{key} final {:?} written by no client op\n      {}", d, excerpt()), Some((ix, key)), None)),
                Some(o) if o.outcome == Outcome::Fail => viol.push(v("G3", "failed-write-visible", format!("{ix}/{key} final {:?} comes from a definitely-failed op\n      {}", d, excerpt()), Some((ix, key)), None)),
                _ => {}
            }
        }
        let first = key.chars().next().unwrap_or(' ');
        match first {
            'c' | 'b' | 'p' => {
                for o in &writes {
                    match o.outcome {
                        Outcome::Ok => {
                            if fin.is_none() {
                                viol.push(v("G3", "lost-acked-write", format!("{ix}/{key} acked but absent from final state\n      {}", excerpt()), Some((ix, key)), Some(o.t_complete)));
                            } else if fin.and_then(|d| d.value) != o.value {
                                viol.push(v("G3", "acked-value-replaced", format!("{ix}/{key} final {:?}\n      {}", fin, excerpt()), Some((ix, key)), Some(o.t_complete)));
                            }
                        }
                        Outcome::Info | Outcome::Pending => {
                            if fin.is_some() {
                                out.counts.info_writes_present += 1;
                            } else {
                                out.counts.info_writes_absent += 1;
                            }
                        }
                        Outcome::Fail => {}
                    }
                }
            }
            'd' => {
                let create = writes.iter().find(|o| o.kind == Kind::Create);
                let delete = writes.iter().find(|o| o.kind == Kind::Delete);
                if let Some(cr) = create {
                    if cr.outcome == Outcome::Ok {
                        match delete.map(|d| d.outcome) {
                            Some(Outcome::Ok) => {
                                let d = delete.expect("delete");
                                if fin.is_some() {
                                    viol.push(v("G3", "acked-delete-resurrected", format!("{ix}/{key}\n      {}", excerpt()), Some((ix, key)), Some(d.t_complete)));
                                }
                                if d.read == Some(None) && cr.t_complete < d.t_invoke {
                                    viol.push(v("G3", "acked-create-not-found-by-delete", format!("{ix}/{key}: delete after acked create returned not_found\n      {}", excerpt()), Some((ix, key)), Some(d.t_complete)));
                                }
                            }
                            Some(Outcome::Fail) | None => {
                                if fin.is_none() {
                                    viol.push(v("G3", "lost-acked-write", format!("{ix}/{key} acked create absent (no successful delete)\n      {}", excerpt()), Some((ix, key)), Some(cr.t_complete)));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            'r' => {
                if fin.is_none() {
                    viol.push(v("G3", "lost-register", format!("{ix}/{key} register missing\n      {}", excerpt()), Some((ix, key)), None));
                    continue;
                }
                let d = fin.expect("doc");
                let acked: Vec<&&&Op> = writes.iter().filter(|o| o.outcome == Outcome::Ok).collect();
                if let Some(latest) = acked.iter().filter(|o| o.seq_no.is_some()).max_by_key(|o| o.seq_no) {
                    let ls = latest.seq_no.unwrap_or(-1);
                    if d.seq_no < ls {
                        viol.push(v("G3", "lost-acked-cas", format!("{ix}/{key} final seq {} < acked write seq {ls}\n      {}", d.seq_no, excerpt()), Some((ix, key)), Some(latest.t_complete)));
                    } else if d.seq_no == ls && d.value != latest.value {
                        viol.push(v("G3", "acked-cas-value-mismatch", format!("{ix}/{key} final {:?} vs acked {:?}\n      {}", d, latest.value, excerpt()), Some((ix, key)), Some(latest.t_complete)));
                    }
                }
                // At most one CAS succeeds per precondition.
                let mut by_pre: BTreeMap<(i64, i64), Vec<&Op>> = BTreeMap::new();
                for o in acked.iter().filter(|o| o.kind == Kind::Cas) {
                    by_pre.entry((o.if_seq_no.unwrap_or(-1), o.if_primary_term.unwrap_or(-1))).or_default().push(o);
                }
                for ((s, t), os) in by_pre {
                    if os.len() > 1 {
                        viol.push(v("G3", "double-cas-success", format!("{ix}/{key}: {} CAS ops with if_seq_no={s} if_primary_term={t} all acked (lost update)\n      {}", os.len(), os.iter().map(|o| op_brief(o)).collect::<Vec<_>>().join("\n      ")), Some((ix, key)), Some(os[0].t_complete)));
                    }
                }
                // Real-time order of acked writes matches seq_no order.
                for a in &acked {
                    for b in &acked {
                        if a.t_complete < b.t_invoke && a.seq_no.is_some() && b.seq_no.is_some() && a.seq_no >= b.seq_no {
                            viol.push(v("G3", "seq-order-violates-realtime", format!("{ix}/{key}: #{} completed before #{} was invoked but has seq {:?} >= {:?}", a.id, b.id, a.seq_no, b.seq_no), Some((ix, key)), Some(b.t_complete)));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Acked writes never share a seq_no within a shard (acked ops are never
    // rolled back, so their slots are never reused).
    let mut seqs: BTreeMap<(&str, i64), Vec<&Op>> = BTreeMap::new();
    for o in inp.ops.iter().filter(|o| is_write(o.kind) && o.outcome == Outcome::Ok) {
        if let Some(s) = o.seq_no {
            seqs.entry((o.index.as_str(), s)).or_default().push(o);
        }
    }
    for ((ix, s), os) in seqs {
        if os.len() > 1 {
            viol.push(v("G3", "seq-no-reused-by-acked-writes", format!("{ix} seq_no {s} acked for {} different ops:\n      {}", os.len(), os.iter().map(|o| op_brief(o)).collect::<Vec<_>>().join("\n      ")), None, Some(os[0].t_complete)));
        }
    }

    // ------------------------------------------------- G7: reads
    let fault_active = |t: f64| inp.windows.iter().any(|w| w.start - 1.0 <= t && w.end.map(|e| t <= e + inp.grace).unwrap_or(true));
    for ((ix, key), ops) in &by_key {
        let writes: Vec<&&Op> = ops.iter().filter(|o| is_write(o.kind)).collect();
        for r in ops.iter().filter(|o| o.kind == Kind::Read && o.outcome == Outcome::Ok) {
            out.counts.reads_ok += 1;
            let rv = r.read.flatten();
            if let Some(val) = rv {
                match writes.iter().find(|o| o.value == Some(val) && o.kind != Kind::Delete) {
                    None => viol.push(v("G7", "phantom-read", format!("{ix}/{key}: {}", op_brief(r)), Some((ix, key)), Some(r.t_complete))),
                    Some(w) if w.outcome == Outcome::Fail => viol.push(v("G7", "read-of-failed-write", format!("{ix}/{key}: {}\n      writer: {}", op_brief(r), op_brief(w)), Some((ix, key)), Some(r.t_complete))),
                    Some(w) if w.t_invoke > r.t_complete => viol.push(v("G7", "read-from-future", format!("{ix}/{key}: {}\n      writer: {}", op_brief(r), op_brief(w)), Some((ix, key)), Some(r.t_complete))),
                    _ => {}
                }
            }
            // Staleness: an acked write completed before the read began.
            let newer = writes.iter().filter(|w| w.outcome == Outcome::Ok && w.t_complete < r.t_invoke).find(|w| match (w.kind, rv, r.seq_no) {
                (Kind::Delete, _, _) => false,
                (_, None, _) => !writes.iter().any(|d| d.kind == Kind::Delete && d.outcome != Outcome::Fail && d.t_invoke < r.t_complete),
                (_, Some(_), Some(rs)) => w.seq_no.map(|ws| ws > rs).unwrap_or(false),
                _ => false,
            });
            if let Some(w) = newer {
                if fault_active(r.t_invoke) {
                    out.counts.stale_reads_in_fault += 1;
                } else {
                    viol.push(v("G7", "stale-read-no-fault", format!("{ix}/{key}: {}\n      newer acked: {}", op_brief(r), op_brief(w)), Some((ix, key)), Some(r.t_complete)));
                }
            }
        }
    }

    // ------------------------------------------------- G8 / G4
    for w in inp.windows.iter().filter(|w| w.kind == "partition" && !w.minority.is_empty()) {
        let Some(end) = w.end else { continue };
        let mut quorum_acks: Vec<(&Op, String)> = Vec::new();
        if let Some(first) = inp.ops.iter().filter(|o| w.minority.contains(&o.node) && o.t_invoke >= w.start && o.t_complete <= end && o.error.as_deref().map(|e| e.starts_with("cluster_block_exception")).unwrap_or(false)).map(|o| o.t_complete).reduce(f64::min) {
            out.counts.minority_block_latency.push(first - w.start);
        }
        for o in inp.ops.iter().filter(|o| is_write(o.kind) && o.kind != Kind::Init && w.minority.contains(&o.node)) {
            if o.t_invoke >= w.start + inp.grace && o.t_complete <= end && o.outcome != Outcome::Pending {
                out.counts.minority_writes += 1;
                let key = format!("{:?}:{}{}", o.outcome, o.status.map(|s| s.to_string()).unwrap_or_default(), o.error.as_deref().map(|e| format!(" {}", e.split(':').next().unwrap_or(""))).unwrap_or_default());
                *out.counts.minority_write_outcomes.entry(key).or_default() += 1;
                if o.error.as_deref().map(|e| e.starts_with("cluster_block_exception")).unwrap_or(false) {
                    out.counts.minority_writes_rejected_block += 1;
                }
                let quorum = if o.outcome == Outcome::Ok { side_voting_quorum(inp, w, o) } else { None };
                if let Some(why) = quorum {
                    out.counts.minority_writes_with_voting_quorum += 1;
                    quorum_acks.push((o, why));
                } else if o.outcome == Outcome::Ok {
                    viol.push(v("G8", "minority-acked-write", format!("write sent to minority node n{} {:.1}s after the partition began was acknowledged: {}", o.node + 1, o.t_invoke - w.start, op_brief(o)), Some((&o.index, &o.key)), Some(o.t_complete)));
                }
            }
        }
        if let Some((o, why)) = quorum_acks.first() {
            let minority: Vec<String> = w.minority.iter().map(|i| format!("n{}", i + 1)).collect();
            warn.push(v("G8", "minority-side-held-voting-quorum", format!("{} write(s) acknowledged on the node-count minority [{}] of the partition at {:.1}s, which held a quorum of the voting configuration ({why}); first: {}", quorum_acks.len(), minority.join(","), w.start, op_brief(o)), None, Some(o.t_complete)));
        }
    }
    for w in inp.windows.iter().filter(|w| w.kind == "pause") {
        let Some(end) = w.end else { continue };
        out.counts.paused_node_acks_after_resume += inp.ops.iter().filter(|o| is_write(o.kind) && w.nodes.contains(&o.node) && o.t_invoke >= w.start && o.t_invoke < end && o.t_complete >= end && o.outcome == Outcome::Ok).count();
    }

    // ------------------------------------------------- G1: masters / forks
    let mut masters_by_term: BTreeMap<i64, BTreeMap<String, (f64, usize)>> = BTreeMap::new();
    let mut uuid_by_version: BTreeMap<i64, BTreeMap<String, Vec<usize>>> = BTreeMap::new();
    let mut hash_by_version: BTreeMap<i64, BTreeSet<u64>> = BTreeMap::new();
    for o in inp.obs.iter().filter(|o| o.master.is_some()) {
        let m = o.master.clone().expect("master");
        masters_by_term.entry(o.term).or_default().entry(m).or_insert((o.t, o.node));
        uuid_by_version.entry(o.version).or_default().entry(o.state_uuid.clone()).or_default().push(o.node);
        hash_by_version.entry(o.version).or_default().insert(o.hash);
    }
    for (term, ms) in &masters_by_term {
        if ms.len() > 1 {
            viol.push(v("G1", "two-masters-one-term", format!("term {term}: masters {:?}", ms), None, ms.values().map(|x| x.0).reduce(f64::min)));
        }
        if let Some((m, (t, _))) = ms.iter().min_by(|a, b| a.1 .0.total_cmp(&b.1 .0)) {
            out.elections.push(Election { term: *term, master: m.clone(), first_seen: *t });
        }
    }
    for (ver, uuids) in &uuid_by_version {
        if uuids.len() > 1 {
            viol.push(v("G1", "state-fork", format!("version {ver} applied with different state_uuids: {:?}", uuids), None, None));
        } else if hash_by_version.get(ver).map(|h| h.len() > 1).unwrap_or(false) {
            let hs: Vec<String> = hash_by_version[ver].iter().filter_map(|h| inp.samples.get(&(*ver, *h))).map(|s| s.chars().take(600).collect()).collect();
            warn.push(v("G1", "same-version-different-content", format!("version {ver}: one state_uuid, {} renderings: {:?}", hs.len(), hs), None, None));
        }
    }
    // Per node, applied versions never go backwards within one process lifetime.
    let restarts: Vec<(usize, f64)> = inp.events.iter().filter(|e| matches!(e.kind.as_str(), "start" | "kill" | "stop" | "kill-all" | "wipe")).flat_map(|e| e.nodes.iter().map(move |n| (*n, e.t))).collect();
    let mut last: BTreeMap<usize, (f64, i64, i64)> = BTreeMap::new();
    for o in inp.obs {
        if let Some((t, ver, term)) = last.get(&o.node) {
            let restarted = restarts.iter().any(|(n, rt)| *n == o.node && *rt >= *t - 0.5 && *rt <= o.t);
            if !restarted && (o.version < *ver || o.term < *term) {
                viol.push(v("G2", "applied-state-went-backwards", format!("n{}: version {ver}/term {term} at {t:.2}s then version {}/term {} at {:.2}s", o.node + 1, o.version, o.term, o.t), None, Some(o.t)));
            }
        }
        last.insert(o.node, (o.t, o.version, o.term));
    }
    // Failover latency after faults that hit the master.
    for e in inp.events.iter().filter(|e| e.roles.iter().any(|r| r == "master" || r.starts_with("master="))) {
        let term_before = inp.obs.iter().filter(|o| o.t <= e.t && o.master.is_some()).map(|o| o.term).max().unwrap_or(-1);
        if let Some(el) = out.elections.iter().find(|el| el.term > term_before && el.first_seen > e.t) {
            out.failover_latencies.push(el.first_seen - e.t);
        }
    }

    // ------------------------------------------------- liveness / crashes
    if let Err(e) = &inp.converged {
        viol.push(v("liveness", "no-convergence", format!("cluster did not return to green with one master after all faults healed: {e}"), None, None));
    }
    for (n, t, st) in inp.crashes {
        viol.push(v("liveness", "node-crashed", format!("n{} exited on its own ({st})", n + 1), None, Some(*t)));
    }

    out.counts.lost = viol.iter().filter(|x| x.kind.starts_with("lost")).count();
    let g = |name: &str| -> String {
        let n = viol.iter().filter(|x| x.guarantee == name).count();
        if n == 0 { "pass".into() } else { format!("FAIL ({n})") }
    };
    for name in ["G1", "G2", "G3", "G5", "G7", "G8", "liveness"] {
        out.guarantees.insert(name.into(), g(name));
    }
    if out.counts.minority_writes == 0 {
        out.guarantees.insert("G8".into(), format!("{} (not exercised)", g("G8")));
    }
    out.violations = viol;
    out.warnings = warn;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::blank;

    fn w(id: u64, kind: Kind, key: &str, val: i64, t: (f64, f64), out: Outcome, seq: Option<i64>) -> Op {
        let mut o = blank(0, 0, kind, "i", key);
        o.id = id;
        o.value = Some(val);
        o.t_invoke = t.0;
        o.t_complete = t.1;
        o.outcome = out;
        o.seq_no = seq;
        o.primary_term = seq.map(|_| 1);
        o
    }
    fn copy(node: usize, primary: bool, docs: &[(&str, i64, i64)]) -> Copy {
        Copy { node, index: "i".into(), primary, docs: docs.iter().map(|(k, s, v)| (k.to_string(), Doc { seq_no: *s, version: 1, value: Some(*v) })).collect(), count: Some(docs.len() as u64), error: None }
    }
    fn run(ops: &[Op], copies: &[Copy], obs: &[Obs], windows: &[Window]) -> Verdict {
        let samples = BTreeMap::new();
        check(&Input { ops, copies, obs, samples: &samples, events: &[], windows, grace: 5.0, converged: Ok(0.0), crashes: &[] })
    }
    fn kinds(v: &Verdict) -> Vec<String> {
        v.violations.iter().map(|x| x.kind.clone()).collect()
    }

    #[test]
    fn clean_history_passes() {
        let ops = vec![w(0, Kind::Create, "c1", 10, (0.0, 1.0), Outcome::Ok, Some(0)), w(1, Kind::Create, "c2", 11, (0.0, 1.0), Outcome::Info, None), w(2, Kind::Create, "c3", 12, (0.0, 1.0), Outcome::Fail, None)];
        let cs = vec![copy(0, true, &[("c1", 0, 10)]), copy(1, false, &[("c1", 0, 10)])];
        let v = run(&ops, &cs, &[], &[]);
        assert!(v.violations.is_empty(), "{:?}", v.violations);
        assert_eq!(v.counts.info_writes_absent, 1);
    }

    #[test]
    fn detects_lost_and_failed_visible_and_divergence() {
        let ops = vec![w(0, Kind::Create, "c1", 10, (0.0, 1.0), Outcome::Ok, Some(0)), w(1, Kind::Create, "c2", 11, (0.0, 1.0), Outcome::Fail, None)];
        let cs = vec![copy(0, true, &[("c2", 1, 11)]), copy(1, false, &[("c1", 0, 10), ("c2", 1, 11)])];
        let k = kinds(&run(&ops, &cs, &[], &[]));
        assert!(k.contains(&"lost-acked-write".to_string()), "{k:?}");
        assert!(k.contains(&"failed-write-visible".to_string()), "{k:?}");
        assert!(k.contains(&"copies-diverge".to_string()), "{k:?}");
    }

    #[test]
    fn detects_double_cas_and_lost_cas() {
        let mut a = w(1, Kind::Cas, "r0", 5, (1.0, 2.0), Outcome::Ok, Some(3));
        a.if_seq_no = Some(0);
        a.if_primary_term = Some(1);
        let mut b = w(2, Kind::Cas, "r0", 6, (1.0, 2.0), Outcome::Ok, Some(4));
        b.if_seq_no = Some(0);
        b.if_primary_term = Some(1);
        let ops = vec![w(0, Kind::Init, "r0", 0, (0.0, 0.1), Outcome::Ok, Some(0)), a, b];
        let cs = vec![copy(0, true, &[("r0", 3, 5)])];
        let k = kinds(&run(&ops, &cs, &[], &[]));
        assert!(k.contains(&"double-cas-success".to_string()), "{k:?}");
        assert!(k.contains(&"lost-acked-cas".to_string()), "{k:?}");
    }

    #[test]
    fn detects_seq_reuse_and_stale_and_dirty_reads() {
        let mut r = w(3, Kind::Read, "c1", 0, (5.0, 5.1), Outcome::Ok, None);
        r.value = None;
        r.read = Some(None);
        let mut r2 = w(4, Kind::Read, "c2", 0, (5.0, 5.1), Outcome::Ok, Some(1));
        r2.value = None;
        r2.read = Some(Some(11));
        let ops = vec![w(0, Kind::Create, "c1", 10, (0.0, 1.0), Outcome::Ok, Some(0)), w(1, Kind::Create, "c2", 11, (0.0, 1.0), Outcome::Fail, None), w(2, Kind::Create, "c3", 12, (0.0, 1.0), Outcome::Ok, Some(0)), r, r2];
        let cs = vec![copy(0, true, &[("c1", 0, 10), ("c3", 0, 12)])];
        let k = kinds(&run(&ops, &cs, &[], &[]));
        assert!(k.contains(&"seq-no-reused-by-acked-writes".to_string()), "{k:?}");
        assert!(k.contains(&"stale-read-no-fault".to_string()), "{k:?}");
        assert!(k.contains(&"read-of-failed-write".to_string()), "{k:?}");
        // The same stale read during a fault window is only counted.
        let win = vec![Window { kind: "partition".into(), start: 4.0, end: Some(6.0), nodes: vec![0], minority: vec![], drops: vec![] }];
        let v = run(&ops, &cs, &[], &win);
        assert!(!kinds(&v).contains(&"stale-read-no-fault".to_string()));
        assert_eq!(v.counts.stale_reads_in_fault, 1);
    }

    #[test]
    fn detects_two_masters_fork_and_minority_ack() {
        let ob = |node: usize, term: i64, version: i64, uuid: &str, m: &str| Obs { t: 1.0, node, term, version, state_uuid: uuid.into(), master: Some(m.into()), committed_config: vec![], accepted_config: vec![], hash: 0 };
        let obs = vec![ob(0, 2, 10, "a", "node-1"), ob(1, 2, 10, "b", "node-2")];
        let mut o = w(0, Kind::Create, "c1", 10, (20.0, 21.0), Outcome::Ok, Some(0));
        o.node = 4;
        let win = vec![Window { kind: "partition".into(), start: 10.0, end: Some(30.0), nodes: vec![0, 1, 2, 3, 4], minority: vec![3, 4], drops: vec![] }];
        let cs = vec![copy(0, true, &[("c1", 0, 10)])];
        let k = kinds(&run(&[o], &cs, &obs, &win));
        assert!(k.contains(&"two-masters-one-term".to_string()), "{k:?}");
        assert!(k.contains(&"state-fork".to_string()), "{k:?}");
        assert!(k.contains(&"minority-acked-write".to_string()), "{k:?}");
    }
    /// Majority-minority seed 2: n5 won term 9 on {n2, n5}
    /// (configuration {n2, n4, n5}) at 301.5 s but applied its own state
    /// only when that publication timed out on n4, 30 s later; writes held
    /// on n5 by the no-master block were acknowledged at 331.44 s and the
    /// monitor's next poll of n5 saw the state at 331.56 s.
    #[test]
    fn an_ack_may_be_judged_by_the_state_observed_just_after_it() {
        let minority = vec![1, 4];
        let drops: Vec<(usize, usize)> = crate::net::split(&[minority.clone(), vec![0, 2, 3]]).into_iter().collect();
        let win = vec![Window { kind: "partition".into(), start: 268.8, end: Some(350.2), nodes: vec![0, 1, 2, 3, 4], minority, drops }];
        let cfg: Vec<String> = ["node-2", "node-4", "node-5"].iter().map(|n| n.to_string()).collect();
        let ob = |t: f64, master: Option<&str>| Obs { t, node: 4, term: 9, version: 31, state_uuid: "u".into(), master: master.map(str::to_string), committed_config: cfg.clone(), accepted_config: cfg.clone(), hash: 0 };
        let mut o = w(0, Kind::Create, "d1-383", 10, (319.9, 331.44), Outcome::Ok, Some(3347));
        o.node = 4;
        let cs = vec![copy(4, true, &[("d1-383", 3347, 10)])];
        let v = run(&[o.clone()], &cs, &[ob(331.28, None), ob(331.56, Some("node-5"))], &win);
        assert!(!kinds(&v).contains(&"minority-acked-write".to_string()), "{:?}", v.violations);
        assert_eq!(v.counts.minority_writes_with_voting_quorum, 1);
        // An observation long after the ack does not stand for its state.
        let v = run(&[o], &cs, &[ob(331.28, None), ob(333.0, Some("node-5"))], &win);
        assert!(kinds(&v).contains(&"minority-acked-write".to_string()), "{:?}", v.violations);
    }

    /// majority-minority seed 2: the master n5 removed n2 and n4 during an
    /// earlier partition and auto-shrank the voting configuration to
    /// {n1, n3, n5}; the next partition split {n1, n5} from {n2, n3, n4}
    /// before n2 and n4 rejoined. {n1, n5} is the node-count minority but
    /// holds the voting quorum, so its acks are legitimate (ES semantics);
    /// with the full 5-node configuration they are a violation.
    #[test]
    fn minority_ack_is_legitimate_only_with_a_voting_quorum() {
        let minority = vec![0, 4];
        let drops: Vec<(usize, usize)> = crate::net::split(&[minority.clone(), vec![1, 2, 3]]).into_iter().collect();
        let win = vec![Window { kind: "partition".into(), start: 76.0, end: Some(133.0), nodes: vec![0, 1, 2, 3, 4], minority, drops }];
        let names = |ns: &[&str]| ns.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        let ob = |cfg: &[&str]| Obs { t: 120.0, node: 0, term: 3, version: 21, state_uuid: "u".into(), master: Some("node-5".into()), committed_config: names(cfg), accepted_config: names(cfg), hash: 0 };
        let mut o = w(0, Kind::Create, "c1", 10, (126.6, 126.7), Outcome::Ok, Some(0));
        o.node = 0;
        let cs = vec![copy(0, true, &[("c1", 0, 10)])];

        let v = run(&[o.clone()], &cs, &[ob(&["node-1", "node-3", "node-5"])], &win);
        assert!(!kinds(&v).contains(&"minority-acked-write".to_string()), "{:?}", v.violations);
        assert_eq!(v.counts.minority_writes_with_voting_quorum, 1);
        assert!(v.warnings.iter().any(|x| x.kind == "minority-side-held-voting-quorum"), "{:?}", v.warnings);

        // The full configuration: {n1, n5} is 2 of 5, not a quorum.
        let v = run(&[o.clone()], &cs, &[ob(&["node-1", "node-2", "node-3", "node-4", "node-5"])], &win);
        assert!(kinds(&v).contains(&"minority-acked-write".to_string()), "{:?}", v.violations);
        // A placeholder or unknown id never counts towards the quorum.
        let v = run(&[o.clone()], &cs, &[ob(&["node-1", "{bootstrap-placeholder}-node-5", "node-3"])], &win);
        assert!(kinds(&v).contains(&"minority-acked-write".to_string()), "{:?}", v.violations);
        // No configuration observed (older binary): a violation, as before.
        let v = run(&[o.clone()], &cs, &[ob(&[])], &win);
        assert!(kinds(&v).contains(&"minority-acked-write".to_string()), "{:?}", v.violations);
        // A master across the partition from the acking node is no excuse.
        let mut far = ob(&["node-2", "node-3", "node-4"]);
        far.master = Some("node-3".into());
        let v = run(&[o], &cs, &[far], &win);
        assert!(kinds(&v).contains(&"minority-acked-write".to_string()), "{:?}", v.violations);
    }
}
