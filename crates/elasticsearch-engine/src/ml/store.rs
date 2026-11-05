//! ML configuration and task state of a node.
//!
//! ES keeps job / datafeed / filter / calendar configs in `.ml-config`,
//! calendar events in `.ml-meta` and job / datafeed task states in the
//! cluster state's persistent tasks. Here everything lives in
//! `<data>/_state/ml.json`. Open jobs and started datafeeds are also
//! registered as running tasks (`xpack/ml/job[c]`, `xpack/ml/datafeed[c]`)
//! so `_tasks` / `_cat/tasks` list them, except in upgrade mode (ES
//! unassigns every ML persistent task then).

use crate::error::Result;
use crate::node::Node;
use crate::tasks::Task;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Runtime state of an anomaly detection job (the job persistent task).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct JobRt {
    /// `opening`, `opened`, `closing`, `closed`, `failed`.
    pub state: String,
    /// When the job was (last) opened.
    pub open_time: Option<i64>,
    /// `false` while a lazily opened job waits for a node it fits on.
    #[serde(default)]
    pub assigned: bool,
    /// 7b: data counts / model size stats of the native process. Kept so a
    /// future analytics process can fill them in; empty means zeros.
    #[serde(default)]
    pub data_counts: Map<String, Value>,
}

/// Runtime state of a datafeed (the datafeed persistent task).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatafeedRt {
    /// `starting`, `started`, `stopping`, `stopped`.
    pub state: String,
    pub start: Option<i64>,
    pub end: Option<i64>,
    #[serde(default)]
    pub search_count: u64,
    #[serde(default)]
    pub total_search_time_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct State {
    /// Job configs as ES renders them (datafeed not embedded).
    #[serde(default)]
    pub jobs: BTreeMap<String, Value>,
    #[serde(default)]
    pub job_rt: BTreeMap<String, JobRt>,
    #[serde(default)]
    pub datafeeds: BTreeMap<String, Value>,
    #[serde(default)]
    pub datafeed_rt: BTreeMap<String, DatafeedRt>,
    #[serde(default)]
    pub calendars: BTreeMap<String, Value>,
    /// Scheduled events (each with `calendar_id` and `event_id`), in
    /// insertion order.
    #[serde(default)]
    pub events: Vec<Value>,
    #[serde(default)]
    pub filters: BTreeMap<String, Value>,
    #[serde(default)]
    pub upgrade_mode: bool,
    /// Ids of jobs being deleted / reset (ES `deleting` / `blocked`).
    #[serde(default)]
    pub blocked: BTreeMap<String, String>,
    /// Model snapshot upgrade tasks (`<job>#<snapshot>` -> status), kept
    /// after they finish as ES keeps the persistent task's last state.
    #[serde(default)]
    pub snapshot_upgrades: BTreeMap<String, Value>,
}

impl State {
    pub fn job_state(&self, id: &str) -> &str {
        self.job_rt.get(id).map(|r| r.state.as_str()).filter(|s| !s.is_empty()).unwrap_or("closed")
    }
    pub fn datafeed_state(&self, id: &str) -> &str {
        self.datafeed_rt.get(id).map(|r| r.state.as_str()).filter(|s| !s.is_empty()).unwrap_or("stopped")
    }
    /// The datafeed feeding job `job_id`.
    pub fn datafeed_for_job(&self, job_id: &str) -> Option<String> {
        self.datafeeds.iter().find(|(_, d)| d["job_id"].as_str() == Some(job_id)).map(|(k, _)| k.clone())
    }
    /// Every group name used by a job.
    pub fn groups(&self) -> Vec<String> {
        let mut g: Vec<String> = self.jobs.values().flat_map(|j| j["groups"].as_array().cloned().unwrap_or_default()).filter_map(|v| v.as_str().map(str::to_string)).collect();
        g.sort();
        g.dedup();
        g
    }
    /// Jobs whose id or groups match the expression token `tok`.
    pub fn jobs_matching(&self, tok: &str) -> Vec<String> {
        let wild = tok == "_all" || tok.contains('*');
        self.jobs
            .iter()
            .filter(|(id, j)| {
                let m = |s: &str| if wild { tok == "_all" || crate::mapping::wildcard_match(tok, s) } else { tok == s };
                m(id) || j["groups"].as_array().map(|g| g.iter().any(|x| x.as_str().map(m).unwrap_or(false))).unwrap_or(false)
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
}

pub struct MlService {
    path: PathBuf,
    st: Mutex<State>,
    /// Running persistent tasks (`job-<id>`, `datafeed-<id>`).
    tasks: Mutex<HashMap<String, Arc<Task>>>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<MlService>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<MlService>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The node's ML service (loaded on first use).
pub fn service(node: &Node) -> Arc<MlService> {
    let mut r = registry().lock();
    if let Some(s) = r.get(&node.data) {
        return s.clone();
    }
    let path = node.data.join("_state").join("ml.json");
    let st: State = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let svc = Arc::new(MlService { path, st: Mutex::new(st), tasks: Mutex::new(HashMap::new()) });
    r.insert(node.data.clone(), svc.clone());
    drop(r);
    svc.sync_tasks(node);
    svc
}

impl MlService {
    pub fn reset(&self, node: &Node) -> Result<()> {
        let state = self.read();
        for id in state.datafeeds.keys() {
            super::feed::stop_and_wait(node, id, std::time::Duration::from_secs(30))?;
        }
        for id in state.jobs.keys() { super::process::drop_process(node, id); }
        self.mutate(|state| { *state = State::default(); Ok(()) })?;
        self.sync_tasks(node);
        super::dfa::reset(node)?;
        crate::inference::models::reset(&node.data.join("_state"))
    }

    pub fn upgrade_mode(&self) -> bool {
        self.st.lock().upgrade_mode
    }

    /// A snapshot of the state.
    pub fn read(&self) -> State {
        self.st.lock().clone()
    }

    /// Read under the lock without cloning.
    pub fn with<R>(&self, f: impl FnOnce(&State) -> R) -> R {
        f(&self.st.lock())
    }

    /// Mutate the state; persisted when `f` succeeds.
    pub fn mutate<R>(&self, f: impl FnOnce(&mut State) -> Result<R>) -> Result<R> {
        let mut st = self.st.lock();
        let mut draft = st.clone();
        let out = f(&mut draft)?;
        *st = draft;
        self.persist(&st)?;
        Ok(out)
    }

    fn persist(&self, st: &State) -> Result<()> {
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(st)?)?;
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    /// Make the registered tasks match the job / datafeed states (none in
    /// upgrade mode).
    pub fn sync_tasks(&self, node: &Node) {
        let st = self.st.lock();
        let mut want: BTreeMap<String, (&'static str, String)> = BTreeMap::new();
        if !st.upgrade_mode {
            for (id, rt) in &st.job_rt {
                if matches!(rt.state.as_str(), "opened" | "opening" | "closing") {
                    want.insert(format!("job-{id}"), ("xpack/ml/job[c]", format!("job-{id}")));
                }
            }
            for (id, rt) in &st.datafeed_rt {
                if matches!(rt.state.as_str(), "started" | "starting" | "stopping") {
                    want.insert(format!("datafeed-{id}"), ("xpack/ml/datafeed[c]", format!("datafeed-{id}")));
                }
            }
        }
        drop(st);
        let mut tasks = self.tasks.lock();
        tasks.retain(|k, t| {
            let keep = want.contains_key(k);
            if !keep {
                node.tasks.finish(t, Ok(Value::Null));
            }
            keep
        });
        for (k, (action, desc)) in want {
            tasks.entry(k).or_insert_with(|| node.tasks.register(action, Some(desc), true, false, Map::new()));
        }
    }
}
