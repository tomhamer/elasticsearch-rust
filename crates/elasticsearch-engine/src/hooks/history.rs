//! Per-shard history state kept beside the translog without changing its
//! format (`cluster/` parses translog files directly):
//!
//! * **term history** (`<shard>/primary_terms.json`, durable): the current
//!   primary term and the seq_no from which each term started, so the term
//!   of any op assigned on this copy is known (hook 4);
//! * **history sidecar** (`<shard>/history/side-<gen:08>.log`, one per
//!   translog generation, same record framing as the translog): no-ops and
//!   the primary term of replicated ops whose term differs from the term
//!   history. Trimmed together with its translog generation;
//! * **seq_no state** ([`SeqNoState`], persisted inside
//!   `shard_state.json` at every commit): local checkpoint, processed
//!   seq_nos above it, delete tombstones above it and the max seq_no of
//!   each translog generation (for retention-aware trimming).

use crate::error::{EsError, Result};
use crate::translog::crc32;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const TERMS_FILE: &str = "primary_terms.json";
pub const HISTORY_DIR: &str = "history";

/// Primary terms of one shard copy.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TermHistory {
    /// The copy's operating primary term (only ever increases).
    pub current: u64,
    /// `(first seq_no, term)`, ascending in both: ops with seq_no in
    /// `[ranges[i].0, ranges[i+1].0)` were assigned in `ranges[i].1`.
    pub ranges: Vec<(i64, u64)>,
}

impl Default for TermHistory {
    fn default() -> Self {
        TermHistory { current: 1, ranges: vec![(0, 1)] }
    }
}

impl TermHistory {
    /// Term of the op with `seq_no` according to the ranges.
    pub fn term_of(&self, seq_no: i64) -> u64 {
        let i = self.ranges.partition_point(|r| r.0 <= seq_no);
        if i == 0 { self.ranges.first().map(|r| r.1).unwrap_or(1) } else { self.ranges[i - 1].1 }
    }

    /// Move to `term` (if higher) for ops from `next_seq_no` on.
    pub(crate) fn bump(&mut self, term: u64, next_seq_no: i64) -> bool {
        if term <= self.current {
            return false;
        }
        self.current = term;
        while self.ranges.last().is_some_and(|r| r.0 >= next_seq_no) {
            self.ranges.pop();
        }
        self.ranges.push((next_seq_no.max(0), term));
        true
    }

    pub(crate) fn load(shard_path: &Path) -> Result<TermHistory> {
        match std::fs::read(shard_path.join(TERMS_FILE)) {
            Ok(b) => serde_json::from_slice(&b).map_err(|e| EsError::internal(format!("corrupt {TERMS_FILE}: {e}"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(TermHistory::default()),
            Err(e) => Err(e.into()),
        }
    }

    pub(crate) fn persist(&self, shard_path: &Path) -> Result<()> {
        super::write_durable(&shard_path.join(TERMS_FILE), &serde_json::to_vec(self)?)?;
        Ok(())
    }
}

/// One history-sidecar record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SideRec {
    pub seq_no: i64,
    pub term: u64,
    /// `Some(reason)` for a no-op.
    pub noop: Option<String>,
}

pub(crate) fn side_file_name(generation: u64) -> String {
    format!("side-{generation:08}.log")
}

fn parse_side(name: &str) -> Option<u64> {
    name.strip_prefix("side-")?.strip_suffix(".log")?.parse().ok()
}

/// Decode a sidecar file: records and the valid prefix length.
fn read_side(path: &Path) -> Result<(Vec<SideRec>, usize)> {
    let mut data = Vec::new();
    match File::open(path) {
        Ok(mut f) => {
            f.read_to_end(&mut data)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(e.into()),
    }
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
        let Some(end) = (pos + 8).checked_add(len) else { break };
        if end > data.len() || crc32(&data[pos + 8..end]) != crc {
            break;
        }
        let Ok(r) = bincode::deserialize::<SideRec>(&data[pos + 8..end]) else { break };
        out.push(r);
        pos = end;
    }
    Ok((out, pos))
}

/// The history sidecar of one shard.
pub(crate) struct SideLog {
    dir: PathBuf,
    open: Option<(u64, File)>,
    /// Records appended to the open file since its last fsync.
    unsynced: bool,
    /// Test seam ([`SideLog::fail_next_appends`]): appends still to fail.
    #[cfg(test)]
    fail_appends: usize,
    /// seq_no → (term, generation) for ops whose term is not the term
    /// history's.
    pub terms: HashMap<i64, (u64, u64)>,
    /// seq_no → (term, reason, generation).
    pub noops: BTreeMap<i64, (u64, String, u64)>,
}

impl SideLog {
    pub fn open(shard_path: &Path) -> Result<SideLog> {
        let dir = shard_path.join(HISTORY_DIR);
        let mut s = SideLog { dir, open: None, unsynced: false, terms: HashMap::new(), noops: BTreeMap::new(),
            #[cfg(test)]
            fail_appends: 0 };
        for (g, p) in s.files()? {
            for r in read_side(&p)?.0 {
                match r.noop {
                    Some(reason) => {
                        s.noops.insert(r.seq_no, (r.term, reason, g));
                    }
                    None => {
                        s.terms.insert(r.seq_no, (r.term, g));
                    }
                }
            }
        }
        Ok(s)
    }

    /// Sidecar files, ascending by generation.
    pub fn files(&self) -> Result<Vec<(u64, PathBuf)>> {
        let mut v = Vec::new();
        match std::fs::read_dir(&self.dir) {
            Ok(rd) => {
                for e in rd.flatten() {
                    if let Some(g) = e.file_name().to_str().and_then(parse_side) {
                        v.push((g, e.path()));
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        v.sort();
        Ok(v)
    }

    /// Captured recovery sources borrow this owner until their independent
    /// retained source-map reservation has taken ownership of copied paths.
    pub(crate) fn files_admitted(&self) -> Result<rustic_core::admission::Allocation<Vec<(u64, PathBuf)>>> {
        crate::translog::list_named_generations_admitted(&self.dir, parse_side, true)
    }

    /// Append records to generation `generation`'s sidecar.
    pub fn append(&mut self, generation: u64, recs: &[SideRec], sync: bool) -> Result<()> {
        if recs.is_empty() {
            return Ok(());
        }
        #[cfg(test)]
        if self.fail_appends > 0 {
            self.fail_appends -= 1;
            return Err(EsError::internal("injected history side log failure"));
        }
        if self.open.as_ref().map(|(g, _)| *g) != Some(generation) {
            // The file being left keeps what was appended to it.
            self.sync()?;
            std::fs::create_dir_all(&self.dir)?;
            let path = self.dir.join(side_file_name(generation));
            let (_, good) = read_side(&path)?;
            let f = OpenOptions::new().create(true).append(true).open(&path)?;
            if f.metadata()?.len() != good as u64 {
                f.set_len(good as u64)?;
            }
            super::sync_dir(Some(&self.dir));
            self.open = Some((generation, f));
        }
        let mut buf = Vec::new();
        for r in recs {
            let body = bincode::serialize(r).map_err(|e| EsError::internal(format!("history sidecar: {e}")))?;
            buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
            buf.extend_from_slice(&crc32(&body).to_le_bytes());
            buf.extend_from_slice(&body);
        }
        let f = &mut self.open.as_mut().unwrap().1;
        f.write_all(&buf)?;
        if sync {
            f.sync_data()?;
            self.unsynced = false;
        } else {
            self.unsynced = true;
        }
        for r in recs {
            match &r.noop {
                Some(reason) => {
                    self.noops.insert(r.seq_no, (r.term, reason.clone(), generation));
                }
                None => {
                    self.terms.insert(r.seq_no, (r.term, generation));
                }
            }
        }
        Ok(())
    }

    /// Test seam: the next `count` non-empty appends fail.
    #[cfg(test)]
    pub(crate) fn fail_next_appends(&mut self, count: usize) {
        self.fail_appends = count;
    }

    pub fn sync(&mut self) -> Result<()> {
        if let Some((_, f)) = &mut self.open {
            f.sync_data()?;
        }
        self.unsynced = false;
        Ok(())
    }

    /// Whether records were appended since the last fsync.
    pub fn has_unsynced(&self) -> bool {
        self.unsynced
    }

    /// Drop every generation below `generation`.
    pub fn trim_below(&mut self, generation: u64) -> Result<()> {
        for (g, p) in self.files()? {
            if g < generation {
                if self.open.as_ref().map(|(og, _)| *og) == Some(g) {
                    self.open = None;
                }
                std::fs::remove_file(p)?;
            }
        }
        self.terms.retain(|_, (_, g)| *g >= generation);
        self.noops.retain(|_, (_, _, g)| *g >= generation);
        Ok(())
    }
}

/// Seq_no bookkeeping persisted with each commit (inside `shard_state.json`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SeqNoState {
    pub local_checkpoint: i64,
    /// Processed seq_nos above the local checkpoint (out-of-order replica ops).
    #[serde(default)]
    pub processed_above: Vec<i64>,
    /// `(id, seq_no, version)` of deletes above the local checkpoint.
    #[serde(default)]
    pub tombstones: Vec<(String, i64, u64)>,
    /// Upper bound of the seq_nos in each translog generation (recorded at roll).
    #[serde(default)]
    pub gen_max_seq: BTreeMap<u64, i64>,
}

/// In-memory history state of a shard (lives under the shard write lock).
pub(crate) struct ShardHistory {
    pub checkpoint: i64,
    pub above: BTreeSet<i64>,
    pub tombstones: HashMap<String, (i64, u64)>,
    pub gen_max_seq: BTreeMap<u64, i64>,
    pub side: SideLog,
}

impl ShardHistory {
    /// From the commit's state (legacy commits: everything up to the
    /// committed seq_no is processed).
    pub fn new(side: SideLog, state: Option<&SeqNoState>, committed_seq_no: i64) -> ShardHistory {
        let mut h = match state {
            Some(s) => ShardHistory {
                checkpoint: s.local_checkpoint,
                above: s.processed_above.iter().copied().collect(),
                tombstones: s.tombstones.iter().map(|(id, s, v)| (id.clone(), (*s, *v))).collect(),
                gen_max_seq: s.gen_max_seq.clone(),
                side,
            },
            None => ShardHistory { checkpoint: committed_seq_no, above: BTreeSet::new(), tombstones: HashMap::new(), gen_max_seq: BTreeMap::new(), side },
        };
        let noops: Vec<i64> = h.side.noops.keys().copied().filter(|s| *s > h.checkpoint).collect();
        for s in noops {
            h.mark(s);
        }
        h
    }

    pub fn is_processed(&self, seq_no: i64) -> bool {
        seq_no <= self.checkpoint || self.above.contains(&seq_no)
    }

    pub fn mark(&mut self, seq_no: i64) {
        if seq_no <= self.checkpoint {
            return;
        }
        if seq_no == self.checkpoint + 1 {
            self.checkpoint = seq_no;
            while self.above.remove(&(self.checkpoint + 1)) {
                self.checkpoint += 1;
            }
        } else {
            self.above.insert(seq_no);
        }
    }

    /// Forget tombstones at or below the local checkpoint (any op for that
    /// id older than the tombstone is then a duplicate).
    pub fn prune_tombstones(&mut self) {
        let cp = self.checkpoint;
        if !self.tombstones.is_empty() {
            self.tombstones.retain(|_, (s, _)| *s > cp);
        }
    }

    pub fn state(&self) -> SeqNoState {
        SeqNoState {
            local_checkpoint: self.checkpoint,
            processed_above: self.above.iter().copied().collect(),
            tombstones: self.tombstones.iter().map(|(id, (s, v))| (id.clone(), *s, *v)).collect(),
            gen_max_seq: self.gen_max_seq.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_ranges() {
        let mut t = TermHistory::default();
        assert_eq!(t.term_of(0), 1);
        assert!(t.bump(3, 10));
        assert!(!t.bump(2, 12));
        assert_eq!((t.term_of(9), t.term_of(10), t.term_of(1000)), (1, 3, 3));
        assert!(t.bump(4, 10)); // no ops under term 3: its range is replaced
        assert_eq!(t.ranges, vec![(0, 1), (10, 4)]);
    }

    #[test]
    fn tracker_marks_out_of_order() {
        let tmp = tempfile::tempdir().unwrap();
        let mut h = ShardHistory::new(SideLog::open(tmp.path()).unwrap(), None, -1);
        for s in [0, 2, 3, 5] {
            h.mark(s);
        }
        assert_eq!(h.checkpoint, 0);
        assert!(h.is_processed(3) && !h.is_processed(1) && !h.is_processed(4));
        h.mark(1);
        assert_eq!(h.checkpoint, 3);
        h.mark(4);
        assert_eq!(h.checkpoint, 5);
        assert!(h.above.is_empty());
    }

    #[test]
    fn sidecar_roundtrip_and_trim() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let mut s = SideLog::open(tmp.path()).unwrap();
            s.append(1, &[SideRec { seq_no: 3, term: 2, noop: None }, SideRec { seq_no: 4, term: 2, noop: Some("gap".into()) }], true).unwrap();
            s.append(2, &[SideRec { seq_no: 9, term: 3, noop: Some("x".into()) }], true).unwrap();
        }
        let mut s = SideLog::open(tmp.path()).unwrap();
        assert_eq!(s.terms.get(&3), Some(&(2, 1)));
        assert_eq!(s.noops.len(), 2);
        s.trim_below(2).unwrap();
        assert!(s.terms.is_empty());
        assert_eq!(s.noops.keys().copied().collect::<Vec<_>>(), vec![9]);
        assert_eq!(SideLog::open(tmp.path()).unwrap().noops.len(), 1);
    }
}
