//! Reloading SSL contexts when their files change, as ES's
//! `SSLConfigurationReloader` (HTTP, transport) and `ReindexSslConfig`
//! (`reindex.ssl`): every file a context reads (certificate, key, CAs,
//! keystore, truststore) is watched on the resource watcher's `HIGH`
//! frequency (`resource.reload.interval.high`, default `5s`). When one is
//! created or changes, the context is rebuilt from all its files and swapped
//! in: new connections handshake with the new material, established ones
//! keep theirs. A rebuild that fails (a half-written key, a certificate that
//! doesn't match the key) is logged and the previous context stays in use
//! until the next change.
//!
//! The current config is behind an [`ArcSwap`]: the accept / connect paths
//! take it with [`Reloadable::load`] per new connection, without a lock.

use arc_swap::ArcSwap;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime};

/// What the watcher compares: ES's `FileWatcher` looks at the length and
/// modification time; the content digest also catches a same-size rewrite
/// within the file system's timestamp granularity.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FileState {
    modified: Option<SystemTime>,
    len: u64,
    digest: [u8; 32],
}

/// `None`: the file is missing or unreadable.
fn file_state(path: &Path) -> Option<FileState> {
    let meta = std::fs::metadata(path).ok()?;
    let bytes = std::fs::read(path).ok()?;
    Some(FileState { modified: meta.modified().ok(), len: meta.len(), digest: Sha256::digest(&bytes).into() })
}

/// The result of one [`Reload::check`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReloadOutcome {
    /// No watched file was created or changed.
    Unchanged,
    /// The context was rebuilt and swapped in.
    Reloaded,
    /// A file changed but the rebuild failed; the previous context stays.
    Failed(String),
}

/// A context the watcher can poll.
pub trait Reload: Send + Sync {
    /// The context name (`xpack.security.http.ssl`).
    fn context(&self) -> &str;
    /// Compare the files with the last check; rebuild and swap on a change.
    fn check(&self) -> ReloadOutcome;
}

type Builder<T> = Box<dyn Fn() -> Result<Arc<T>, String> + Send + Sync>;

/// A config (`rustls::ServerConfig`, `ClientConfig`, or anything built from
/// files) that follows changes to the files it was built from.
pub struct Reloadable<T> {
    context: String,
    files: Vec<PathBuf>,
    build: Option<Builder<T>>,
    current: ArcSwap<T>,
    /// The files' states as of the last check (or the initial build).
    seen: Mutex<Vec<Option<FileState>>>,
}

impl<T> std::fmt::Debug for Reloadable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reloadable").field("context", &self.context).field("files", &self.files).finish_non_exhaustive()
    }
}

impl<T: Send + Sync + 'static> Reloadable<T> {
    /// Build the config now (an error is the caller's start failure) and
    /// rebuild it with `build` whenever one of `files` changes.
    pub fn new(context: impl Into<String>, files: Vec<PathBuf>, build: impl Fn() -> Result<Arc<T>, String> + Send + Sync + 'static) -> Result<Reloadable<T>, String> {
        // The states before the build: a change made while it reads the
        // files is seen by the next check.
        let seen = files.iter().map(|f| file_state(f)).collect();
        let first = build()?;
        Ok(Reloadable { context: context.into(), files, build: Some(Box::new(build)), current: ArcSwap::new(first), seen: Mutex::new(seen) })
    }

    /// A config that never reloads (built from settings without files, or
    /// handed in by a caller that manages it).
    pub fn fixed(context: impl Into<String>, config: Arc<T>) -> Reloadable<T> {
        Reloadable { context: context.into(), files: Vec::new(), build: None, current: ArcSwap::new(config), seen: Mutex::new(Vec::new()) }
    }

    /// The current config, for one new connection.
    pub fn load(&self) -> Arc<T> {
        self.current.load_full()
    }

    /// The watched files.
    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }
}

impl<T: Send + Sync + 'static> Reload for Reloadable<T> {
    fn context(&self) -> &str {
        &self.context
    }

    fn check(&self) -> ReloadOutcome {
        let Some(build) = &self.build else { return ReloadOutcome::Unchanged };
        let mut seen = self.seen.lock().unwrap_or_else(|p| p.into_inner());
        let now: Vec<Option<FileState>> = self.files.iter().map(|f| file_state(f)).collect();
        if now == *seen {
            return ReloadOutcome::Unchanged;
        }
        // ES reloads on a file's creation or change, not on its deletion
        // (a missing file would only fail the rebuild).
        let changed: Vec<&PathBuf> = self.files.iter().zip(now.iter().zip(seen.iter())).filter(|(_, (n, s))| n.is_some() && n != s).map(|(f, _)| f).collect();
        // As ES's file watcher, the new states are recorded whether or not
        // the rebuild succeeds: a failed reload is retried on the next change.
        let changed: Vec<String> = changed.into_iter().map(|f| f.display().to_string()).collect();
        *seen = now;
        if changed.is_empty() {
            return ReloadOutcome::Unchanged;
        }
        match build() {
            Ok(config) => {
                self.current.store(config);
                for file in &changed {
                    tracing::info!(context = %self.context, "updated ssl contexts using changed file [{file}]");
                }
                ReloadOutcome::Reloaded
            }
            Err(error) => {
                tracing::warn!(context = %self.context, changed = ?changed, "failed to reload ssl configuration [{}]: {error}", self.context);
                ReloadOutcome::Failed(error)
            }
        }
    }
}

/// Poll `contexts` every `interval` on a background thread (ES's resource
/// watcher). The thread holds them weakly and ends once all are dropped.
pub fn watch(contexts: Vec<Arc<dyn Reload>>, interval: Duration) -> std::io::Result<()> {
    let weak: Vec<Weak<dyn Reload>> = contexts.iter().map(Arc::downgrade).collect();
    drop(contexts);
    if weak.is_empty() {
        return Ok(());
    }
    std::thread::Builder::new().name("ssl-config-reloader".into()).spawn(move || loop {
        std::thread::sleep(interval);
        let mut alive = false;
        for context in &weak {
            if let Some(context) = context.upgrade() {
                alive = true;
                context.check();
            }
        }
        if !alive {
            return;
        }
    })?;
    Ok(())
}

/// `resource.reload.enabled` and `resource.reload.interval.high` (ES
/// `ResourceWatcherService`): `Ok(None)` when reloading is disabled, else
/// the interval SSL files are checked at. `time_ms` parses an ES time value
/// into milliseconds.
pub fn reload_interval(get: &dyn Fn(&str) -> Option<String>, time_ms: &dyn Fn(&str) -> Option<i64>) -> Result<Option<Duration>, String> {
    const ENABLED: &str = "resource.reload.enabled";
    const HIGH: &str = "resource.reload.interval.high";
    match get(ENABLED).as_deref() {
        None | Some("true") => {}
        Some("false") => return Ok(None),
        Some(other) => return Err(format!("Failed to parse value [{other}] as only [true] or [false] are allowed.")),
    }
    let Some(value) = get(HIGH) else { return Ok(Some(Duration::from_secs(5))) };
    match time_ms(&value) {
        Some(ms) if ms > 0 => Ok(Some(Duration::from_millis(ms as u64))),
        Some(ms) if ms < 0 => Err(format!("failed to parse setting [{HIGH}] with value [{value}] as a time value: negative durations are not supported")),
        Some(_) =>Err(format!("failed to parse setting [{HIGH}] with value [{value}]: the resource watcher interval must be positive")),
        None => Err(format!("failed to parse setting [{HIGH}] with value [{value}] as a time value: unit is missing or unrecognized")),
    }
}
