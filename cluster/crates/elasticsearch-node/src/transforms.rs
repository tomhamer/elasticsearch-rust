//! The cluster side of transforms (ES `TransformPersistentTasksExecutor`):
//! the engine's transform service stores configs, state and checkpoints in
//! the `.transform-internal-007` system index through this node's
//! coordinator, and runs a started transform as a persistent task in the
//! `persistent_tasks` metadata custom, assigned to the elected master. Every
//! second [`tick`] lets the elected master load the transforms and take over
//! tasks assigned to another node (new allocation id), and makes any other
//! node stop executing them. See `elasticsearch_engine::transform::cluster`.

use crate::node::ClusterNode;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::transform::cluster::{TasksChange, TransformCluster, TASKS_CUSTOM};
use crate::front::BackgroundSubject;
use elasticsearch_engine::transform::{RestCall, RunAs};
use serde_json::Value;
use std::sync::{Arc, Weak};
use std::time::Duration;

/// How long a task change may take to reach this node's applied state.
const APPLY_WAIT: Duration = Duration::from_secs(30);
/// Attempts of a task change that lost a compare-and-set race.
const PUBLISH_ATTEMPTS: usize = 5;

pub struct ClusterTransforms {
    node: Weak<ClusterNode>,
    runtime: tokio::runtime::Handle,
}

impl ClusterTransforms {
    pub fn new(node: Weak<ClusterNode>, runtime: tokio::runtime::Handle) -> ClusterTransforms {
        ClusterTransforms { node, runtime }
    }

    fn node(&self) -> Result<Arc<ClusterNode>> {
        self.node.upgrade().ok_or_else(|| EsError::new(503, "node_closed_exception", "node is shutting down"))
    }

    /// Run `f` to completion from engine code (a blocking-pool or worker
    /// thread; never an async task of this runtime).
    fn block<F: std::future::Future>(&self, f: F) -> F::Output {
        tokio::task::block_in_place(|| self.runtime.block_on(f))
    }

    fn rest(&self, subject: Option<RunAs>) -> RestCall {
        let (weak, runtime) = (self.node.clone(), self.runtime.clone());
        // Encoded once: every call of this coordinator carries it.
        let subject = subject.map(|s| match s {
            RunAs::Identity(id) => (true, id.encode()),
            RunAs::User(u) => (false, u),
        });
        Arc::new(move |method, uri, body| {
            let node = weak.upgrade().ok_or_else(|| EsError::new(503, "node_closed_exception", "node is shutting down"))?;
            let who = match &subject {
                None => BackgroundSubject::System,
                Some((true, encoded)) => BackgroundSubject::Identity(encoded),
                Some((false, user)) => BackgroundSubject::RunAsUser(user),
            };
            let call = crate::front::background_cluster_rest(node, runtime.clone(), who)?;
            call(method, uri, body)
        })
    }
}

impl TransformCluster for ClusterTransforms {
    fn system_rest(&self) -> RestCall {
        self.rest(None)
    }

    fn rest_as(&self, subject: Option<&RunAs>) -> RestCall {
        self.rest(subject.cloned())
    }

    fn local_node(&self) -> String {
        self.node.upgrade().map(|n| n.local.id.clone()).unwrap_or_default()
    }

    fn master_term(&self) -> Option<u64> {
        let node = self.node.upgrade()?;
        node.is_master().then(|| node.state().coordination.term)
    }

    fn persistent_tasks(&self) -> Value {
        self.node.upgrade().and_then(|n| n.state().metadata.customs.get(TASKS_CUSTOM).cloned()).unwrap_or(Value::Null)
    }

    fn update_persistent_tasks(&self, f: TasksChange) -> Result<Value> {
        let node = self.node()?;
        // Computed from the custom as the batch publishes it (a change by an
        // earlier job of the same batch included) and published
        // compare-and-set; a stale base is computed again on the newer state.
        let mut last_err = String::new();
        for _ in 0..PUBLISH_ATTEMPTS {
            let out: Arc<parking_lot::Mutex<Option<Value>>> = Arc::new(parking_lot::Mutex::new(None));
            let (o, change) = (out.clone(), f.clone());
            let compute: Box<dyn FnOnce(&ClusterNode, &elasticsearch_cluster::state::ClusterState, &mut crate::mirror::MetadataDelta) + Send> =
                Box::new(move |_, base, delta| {
                    let current = match delta.customs.get(TASKS_CUSTOM) {
                        Some(Some(v)) => v.clone(),
                        Some(None) => Value::Null,
                        None => base.metadata.customs.get(TASKS_CUSTOM).cloned().unwrap_or(Value::Null),
                    };
                    match change(&current) {
                        Some(v) => {
                            delta.customs.insert(TASKS_CUSTOM.to_string(), Some(v.clone()));
                            *o.lock() = Some(v);
                        }
                        None => *o.lock() = Some(current),
                    }
                });
            match self.block(node.metadata_service().pure(compute)) {
                Ok(version) => {
                    self.block(node.wait_applied_here(version, APPLY_WAIT));
                    return Ok(out.lock().take().unwrap_or(Value::Null));
                }
                Err(e) if elasticsearch_cluster::coordination::CoordinationError::is_not_master(&e) => {
                    return Err(EsError::new(503, "not_master_exception", e));
                }
                Err(e) if e.starts_with(crate::master::STALE_BASE) => last_err = e,
                Err(e) => return Err(EsError::new(503, "failed_to_commit_cluster_state_exception", e)),
            }
        }
        Err(EsError::new(503, "failed_to_commit_cluster_state_exception", format!("persistent task change not published: {last_err}")))
    }
}

/// Install the cluster binding into the node's transform service.
pub fn install(node: &Arc<ClusterNode>) {
    let svc = elasticsearch_engine::transform::service(&node.engine);
    svc.set_cluster(Some(Arc::new(ClusterTransforms::new(Arc::downgrade(node), tokio::runtime::Handle::current()))));
}

/// One maintenance pass (blocking: run on the blocking pool).
pub fn tick(node: &Arc<ClusterNode>) {
    elasticsearch_engine::transform::cluster_tick(&node.engine);
}
