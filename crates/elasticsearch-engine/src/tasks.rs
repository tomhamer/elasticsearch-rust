//! Task manager behind `_tasks`: ids are `<node_id>:<n>`, running tasks
//! live in a registry (with a cancellation flag and a status object the
//! task updates as it goes), and completed tasks that were started with
//! `wait_for_completion=false` keep their result (ES stores these in the
//! `.tasks` system index; here they are kept in memory and are lost on
//! restart).

use crate::error::{EsError, Result};
use crate::node::NODE_ID;
use crossbeam_skiplist::SkipMap;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A task description built when a listing asks for it (ES builds a
/// search's description lazily too: rendering its source per search would
/// cost every search for the rare `_tasks?detailed`).
pub type Describe = Arc<dyn Fn() -> String + Send + Sync>;

pub struct Task {
    pub id: u64,
    pub action: String,
    pub description: Option<String>,
    /// [`Describe`]: the description when `description` is `None`.
    describe: Option<Describe>,
    pub start_millis: i64,
    start: Instant,
    pub cancellable: bool,
    /// Its cancellation (a child task's follows its parent's when the
    /// parent runs in this process). Search loops poll it.
    cancel: Arc<rustic_core::cancel::Token>,
    status: Mutex<Option<Value>>,
    pub parent: Option<String>,
    pub headers: Map<String, Value>,
    /// Keep the outcome after completion (`wait_for_completion=false`).
    pub store_result: bool,
    /// Bulk-by-scroll `requests_per_second` (f64 bits; `-1.0` = unthrottled), changed by `_rethrottle`.
    rps_bits: AtomicU64,
    _metadata_owner: Option<crate::memory::Reservation>,
}

impl Task {
    pub fn task_id(&self) -> String {
        format!("{NODE_ID}:{}", self.id)
    }

    /// The description (built now for a lazily described task).
    pub fn description_text(&self) -> Option<String> {
        self.description.clone().or_else(|| self.describe.as_ref().map(|d| d()))
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Cancellation without taking the task registry lock (used by admitted
    /// registry visitors when their output allocation is refused).
    pub fn request_cancel(&self){if self.cancellable{self.cancel.cancel("by user request");}}

    /// Cancel with `reason` (ES `CancellableTask.cancel`: `task cancelled
    /// [<reason>]`), e.g. when the client that started it went away.
    pub fn cancel_with(&self, reason: &str) {
        if self.cancellable {
            self.cancel.cancel(reason);
        }
    }

    /// The task's cancellation token, for the work it runs (see
    /// [`rustic_core::cancel::search_scope`]).
    pub fn cancel_token(&self) -> &Arc<rustic_core::cancel::Token> {
        &self.cancel
    }

    /// Current `requests_per_second` of a bulk-by-scroll task (`-1.0` = unthrottled).
    pub fn rps(&self) -> f64 {
        f64::from_bits(self.rps_bits.load(Ordering::Relaxed))
    }

    /// Set the rate (start, or `_rethrottle`) and patch the visible status: the new
    /// `requests_per_second` (split evenly across `slices`) and no pending delay.
    pub fn set_rps(&self, rps: f64) {
        self.rps_bits.store(rps.to_bits(), Ordering::Relaxed);
        if let Some(Value::Object(m)) = self.status.lock().as_mut() {
            m.insert("requests_per_second".into(), json!(rps));
            m.insert("throttled_until_millis".into(), json!(0));
            if let Some(Value::Array(sl)) = m.get_mut("slices") {
                let n = sl.len().max(1) as f64;
                for s in sl.iter_mut().filter_map(Value::as_object_mut) {
                    s.insert("requests_per_second".into(), json!(if rps < 0.0 { rps } else { rps / n }));
                    s.insert("throttled_until_millis".into(), json!(0));
                }
            }
        }
    }

    pub fn set_status(&self, v: Value) {
        *self.status.lock() = Some(v);
    }

    pub fn status(&self) -> Option<Value> {
        self.status.lock().clone()
    }

    /// Borrowed task projection for admitted wire encoders. No metadata/status
    /// tree is cloned before the caller's serializer admits its output.
    pub fn serialize_info<S:serde::Serializer>(&self,detailed:bool,node:&str,id:u64,parent:Option<&str>,serializer:S)->std::result::Result<S::Ok,S::Error>{
        use serde::ser::SerializeMap;
        let status=self.status.lock();
        let mut m=serializer.serialize_map(None)?;
        m.serialize_entry("node",node)?;m.serialize_entry("id",&id)?;m.serialize_entry("type","transport")?;m.serialize_entry("action",&self.action)?;
        if detailed {if let Some(status)=status.as_ref(){m.serialize_entry("status",status)?;}if let Some(description)=self.description_text(){m.serialize_entry("description",&description)?;}}
        m.serialize_entry("start_time_in_millis",&self.start_millis)?;m.serialize_entry("running_time_in_nanos",&(self.start.elapsed().as_nanos()as u64))?;
        m.serialize_entry("cancellable",&self.cancellable)?;if self.cancellable{m.serialize_entry("cancelled",&self.is_cancelled())?;}
        if let Some(parent)=parent{m.serialize_entry("parent_task_id",parent)?;}
        m.serialize_entry("headers",&self.headers)?;m.end()
    }

    /// ES `TaskInfo` JSON.
    pub fn info(&self, detailed: bool) -> Value {
        self.info_with_runtime(detailed, self.start.elapsed())
    }

    fn info_with_runtime(&self, detailed: bool, running: Duration) -> Value {
        let mut m = Map::new();
        m.insert("node".into(), json!(NODE_ID));
        m.insert("id".into(), json!(self.id));
        m.insert("type".into(), json!("transport"));
        m.insert("action".into(), json!(self.action));
        if detailed {
            if let Some(s) = self.status() {
                m.insert("status".into(), s);
            }
            if let Some(d) = self.description_text() {
                m.insert("description".into(), json!(d));
            }
        }
        m.insert("start_time_in_millis".into(), json!(self.start_millis));
        m.insert("running_time_in_nanos".into(), json!(running.as_nanos() as u64));
        m.insert("cancellable".into(), json!(self.cancellable));
        if self.cancellable {
            m.insert("cancelled".into(), json!(self.is_cancelled()));
        }
        if let Some(p) = &self.parent {
            m.insert("parent_task_id".into(), json!(p));
        }
        m.insert("headers".into(), Value::Object(self.headers.clone()));
        Value::Object(m)
    }
}

/// The stored outcome of a finished task.
#[derive(Clone)]
pub struct TaskResult {
    pub task: Value,
    /// `Ok(response)` or `Err(error object)`.
    pub outcome: std::result::Result<Value, Value>,
}

/// Lock-free: every search registers and finishes tasks, so the registry
/// is a concurrent skip list (ordered by id, as `_tasks` lists them), never
/// a lock shared by all requests.
///
/// A removed skip-list entry is freed only once its epoch is reclaimed,
/// which on a quiet node can be much later. So a running entry's slot gives
/// up its task when the task finishes: what the task holds (a search's
/// request and its breaker reservation, its metadata reservation) is
/// released then, never left to the reclamation.
pub struct TaskManager {
    next: AtomicU64,
    running: SkipMap<u64, Slot>,
    done: SkipMap<u64, TaskResult>,
}

impl Default for TaskManager {
    fn default() -> Self {
        TaskManager { next: AtomicU64::new(0), running: SkipMap::new(), done: SkipMap::new() }
    }
}

/// A running entry's task: given up when the task finishes (see
/// [`TaskManager`]). A plain lock, not an `ArcSwapOption`: emptying an
/// `ArcSwapOption` pays every thread's outstanding debts (a walk over a
/// node per thread that ever read one), which every finishing search
/// task paid (~3% of a hybrid search node's CPU). Uncontended: only the
/// task's own finish writes it; readers (`_tasks`, parent lookups) clone.
struct Slot(parking_lot::Mutex<Option<Arc<Task>>>);

impl Slot {
    fn new(task: Arc<Task>) -> Slot {
        Slot(parking_lot::Mutex::new(Some(task)))
    }
    fn load_full(&self) -> Option<Arc<Task>> {
        self.0.lock().clone()
    }
    /// The task, taken out (dropped by the caller outside the lock).
    fn take(&self) -> Option<Arc<Task>> {
        self.0.lock().take()
    }
}

/// Breaker charge for one registry entry: a skip-list node (header, key,
/// value and its tower of next pointers). Each task carries its own, so it
/// is released with the task and needs no shared accounting.
const REGISTRY_NODE_BYTES: u64 = 128;

/// Parse `node:n`.
pub fn parse_task_id(s: &str) -> Result<(String, u64)> {
    let bad = || EsError::illegal_argument(format!("malformed task id {s}"));
    let (n, i) = s.split_once(':').ok_or_else(bad)?;
    let id = i.parse::<u64>().map_err(|_| bad())?;
    if n.is_empty() {
        return Err(bad());
    }
    Ok((n.to_string(), id))
}

impl TaskManager {
    pub fn register(&self, action: &str, description: Option<String>, cancellable: bool, store_result: bool, headers: Map<String, Value>) -> Arc<Task> {
        self.register_with_parent(action, description, cancellable, store_result, headers, None)
    }

    /// [`Self::register`] for a child task (`parent_task_id`).
    pub fn register_with_parent(&self, action: &str, description: Option<String>, cancellable: bool, store_result: bool, headers: Map<String, Value>, parent: Option<String>) -> Arc<Task> {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let t = Arc::new(Task {
            id,
            action: action.to_string(),
            description,
            describe: None,
            start_millis: crate::date::now_millis(),
            start: Instant::now(),
            cancellable,
            cancel: rustic_core::cancel::Token::new(),
            status: Mutex::new(None),
            parent,
            headers,
            store_result,
            rps_bits: AtomicU64::new((-1.0f64).to_bits()),
            _metadata_owner: None,
        });
        self.running.insert(id, Slot::new(t.clone()));
        t
    }

    /// Borrowed transport metadata is copied only after admission. Its owner
    /// lives in the task, including when the task outlives the incoming request.
    pub fn try_register_borrowed(&self, action:&str, description:Option<&str>, cancellable:bool, store_result:bool, headers:&[(String,String)], parent:Option<&str>)->Result<Arc<Task>> {
        self.try_register_linked(action, description, None, cancellable, store_result, headers, parent, None)
    }

    /// [`Self::try_register_borrowed`] with a lazily built description
    /// (`describe`, used when `description` is `None`), for a child of a
    /// task running in this process: its cancellation follows
    /// `parent_token` (a search's shard task follows its coordinating task).
    #[allow(clippy::too_many_arguments)]
    pub fn try_register_linked(&self, action:&str, description:Option<&str>, describe:Option<Describe>, cancellable:bool, store_result:bool, headers:&[(String,String)], parent:Option<&str>, parent_token:Option<Arc<rustic_core::cancel::Token>>)->Result<Arc<Task>> {
        let mut memory=crate::memory::breakers().reserve_native(0,"task metadata")?;
        fn text(memory:&mut crate::memory::Reservation,s:&str)->Result<String>{
            memory.grow(s.len() as u64,"task metadata text")?;
            let mut out=String::new();out.try_reserve_exact(s.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
            out.push_str(s);Ok(out)
        }
        let action=text(&mut memory,action)?;
        let description=description.map(|s|text(&mut memory,s)).transpose()?;
        let parent=parent.map(|s|text(&mut memory,s)).transpose()?;
        let mut owned_headers=Map::new();
        for (key,value) in headers {
            // Covers either Map backend, including replacement-table overlap.
            memory.grow((2*(12*std::mem::size_of::<(String,Value)>()+16*std::mem::size_of::<usize>()))as u64,"task header map")?;
            owned_headers.insert(text(&mut memory,key)?,Value::String(text(&mut memory,value)?));
        }
        // Arc metadata and this task's registry node are charged to the
        // task's own reservation (skip-list nodes are freed with their entry).
        memory.grow((std::mem::size_of::<Task>()+2*std::mem::size_of::<usize>())as u64,"task registration")?;
        memory.grow(REGISTRY_NODE_BYTES,"task registry node")?;
        let id=self.next.fetch_add(1,Ordering::Relaxed)+1;
        let cancel=if parent_token.is_some(){rustic_core::cancel::Token::child_of(parent_token)}else{rustic_core::cancel::Token::new()};
        let task=Arc::new(Task{id,action,description,describe,start_millis:crate::date::now_millis(),start:Instant::now(),cancellable,cancel,status:Mutex::new(None),parent,headers:owned_headers,store_result,rps_bits:AtomicU64::new((-1.0f64).to_bits()),_metadata_owner:Some(memory.into_result_owner())});
        self.running.insert(id,Slot::new(task.clone()));
        Ok(task)
    }

    /// Mark a task finished; keeps the result when the task asked for it.
    pub fn finish(&self, task: &Task, outcome: std::result::Result<Value, Value>) {
        let elapsed = task.start.elapsed();
        if task.store_result {
            let info = task.info_with_runtime(true, elapsed);
            self.done.insert(task.id, TaskResult { task: info, outcome });
        }
        if let Some(entry) = self.running.remove(&task.id) {
            drop(entry.value().take());
        }
    }

    pub fn try_visit_running<E>(&self,mut visit:impl FnMut(&Arc<Task>)->std::result::Result<(),E>)->std::result::Result<(),E>{
        for entry in self.running.iter(){if let Some(task)=entry.value().load_full(){visit(&task)?;}}Ok(())
    }
    pub fn with_result<T>(&self,id:u64,visit:impl FnOnce(Option<&TaskResult>)->T)->T{let entry=self.done.get(&id);visit(entry.as_ref().map(|e|e.value()))}

    pub fn running(&self) -> Vec<Arc<Task>> {
        self.running.iter().filter_map(|e| e.value().load_full()).collect()
    }

    pub fn get_running(&self, id: u64) -> Option<Arc<Task>> {
        self.running.get(&id).and_then(|e| e.value().load_full())
    }

    pub fn get_result(&self, id: u64) -> Option<TaskResult> {
        self.done.get(&id).map(|e| e.value().clone())
    }

    /// Cancel a running task. `Ok(None)` = not running.
    pub fn cancel(&self, id: u64) -> Result<Option<Arc<Task>>> {
        let Some(t) = self.get_running(id) else { return Ok(None) };
        if !t.cancellable {
            return Err(EsError::illegal_argument(format!("task [{}] doesn't support cancellation", t.task_id())));
        }
        t.cancel.cancel("by user request");
        Ok(Some(t))
    }

    /// Wait until task `id` is no longer running (or the timeout passes).
    pub fn wait(&self, id: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.get_running(id).is_none() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_cancel_finish() {
        let tm = TaskManager::default();
        let t = tm.register("indices:data/write/reindex", Some("reindex".into()), true, true, Map::new());
        assert_eq!(t.task_id(), format!("{NODE_ID}:1"));
        assert_eq!(tm.running().len(), 1);
        assert!(tm.cancel(t.id).unwrap().is_some());
        assert!(t.is_cancelled());
        tm.finish(&t, Ok(json!({"ok": true})));
        assert!(tm.running().is_empty());
        assert_eq!(tm.get_result(t.id).unwrap().outcome.unwrap()["ok"], true);
        let nc = tm.register("x", None, false, false, Map::new());
        assert_eq!(tm.cancel(nc.id).err().unwrap().kind, "illegal_argument_exception");
        assert_eq!(parse_task_id("abc").unwrap_err().reason, "malformed task id abc");
        assert_eq!(parse_task_id("n:7").unwrap(), ("n".to_string(), 7));
    }

    /// Concurrent registration, listing, lookup and finishing from many
    /// threads (the registry takes no lock): every task is seen while it
    /// runs, ids stay unique, listings stay id-ordered, none is left over.
    #[test]
    fn concurrent_register_list_finish() {
        let tm = Arc::new(TaskManager::default());
        let threads: Vec<_> = (0..16).map(|_| {
            let tm = tm.clone();
            std::thread::spawn(move || {
                let mut ids = Vec::new();
                for i in 0..2_000 {
                    let t = tm.try_register_borrowed("indices:data/read/search", None, true, i % 100 == 0, &[], None).unwrap();
                    assert!(tm.get_running(t.id).is_some());
                    if i % 64 == 0 {
                        let listed = tm.running();
                        assert!(listed.windows(2).all(|w| w[0].id < w[1].id), "listing is id-ordered");
                    }
                    ids.push(t.id);
                    tm.finish(&t, Ok(json!({"i": i})));
                    assert!(tm.get_running(t.id).is_none());
                }
                ids
            })
        }).collect();
        let mut all: Vec<u64> = threads.into_iter().flat_map(|t| t.join().unwrap()).collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "task ids are unique");
        assert!(tm.running().is_empty(), "no task left registered");
        let stored = all.iter().filter(|id| tm.get_result(**id).is_some()).count();
        assert_eq!(stored, 16 * 20, "results of store_result tasks are kept");
    }

    /// A finished task is released when it finishes, not when the skip
    /// list reclaims its removed entry: what it holds (here its lazy
    /// description's capture, for a search its request's breaker
    /// reservation) goes with the last handle outside the registry.
    #[test]
    fn finish_releases_what_the_task_holds() {
        let tm = TaskManager::default();
        let held = Arc::new(());
        let capture = held.clone();
        let describe: Describe = Arc::new(move || {
            let _ = &capture;
            "search".to_string()
        });
        let t = tm.try_register_linked("indices:data/read/search", None, Some(describe), true, false, &[], None, None).unwrap();
        assert_eq!(tm.get_running(t.id).unwrap().description_text().as_deref(), Some("search"));
        tm.finish(&t, Ok(Value::Null));
        drop(t);
        assert_eq!(Arc::strong_count(&held), 1, "the registry still holds the finished task");
    }
}
