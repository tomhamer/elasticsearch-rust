//! Authentication at the HTTP edge, before the request body is read
//! (production review #17 / #18; ES authenticates on the request headers in
//! its HTTP header validator, before the content is aggregated).
//!
//! Deciding most credentials is cheap (a cached success, an unknown
//! principal, a cheap hash). Uncached bcrypt / PBKDF2 verification is not:
//! it never runs on the networking runtime, but on a bounded pool of
//! dedicated crypto threads (ES `security-crypto`: `(processors + 1) / 2`
//! threads, queue 1000) with admission (a full queue answers 429), behind a
//! failed-authentication limiter per user and per client address. Only
//! successes are cached (ES `CachingUsernamePasswordRealm`).

use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{AuthFail, Authn, Credentials, Security};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant};

/// The edge's verdict on a request's credentials.
pub enum EdgeAuth {
    /// Authenticated (or security disabled): read the body, dispatch.
    Authenticated,
    /// Known-invalid credentials: answer 401 without reading the body.
    Failed(AuthFail),
    /// Not verified: the crypto pool is saturated or the client / user has
    /// failed too often (429, without reading the body).
    Rejected(EsError),
}

impl EdgeAuth {
    pub fn authenticated(&self) -> bool {
        matches!(self, EdgeAuth::Authenticated)
    }
}

type Job = Box<dyn FnOnce() + Send>;

/// Bounded pool of dedicated threads for expensive password verification.
pub struct CryptoPool {
    sender: Mutex<Option<SyncSender<Job>>>,
    pub threads: usize,
    pub queue: usize,
    rejected: AtomicU64,
    completed: AtomicU64,
}

impl CryptoPool {
    pub fn new(threads: usize, queue: usize) -> Arc<CryptoPool> {
        let threads = threads.max(1);
        // `sync_channel(queue)` holds `queue` waiting jobs; each thread holds
        // the one it runs.
        let (sender, receiver) = sync_channel::<Job>(queue);
        let receiver = Arc::new(std::sync::Mutex::new(receiver));
        for i in 0..threads {
            let receiver = receiver.clone();
            let _ = std::thread::Builder::new().name(format!("security-crypto-{i}")).spawn(move || loop {
                let job = match receiver.lock() {
                    Ok(r) => r.recv(),
                    Err(_) => return,
                };
                match job {
                    Ok(job) => job(),
                    Err(_) => return, // pool replaced / dropped
                }
            });
        }
        Arc::new(CryptoPool { sender: Mutex::new(Some(sender)), threads, queue, rejected: AtomicU64::new(0), completed: AtomicU64::new(0) })
    }

    /// Queue `f` on the pool without waiting; `Err` (429) when its queue is full.
    pub fn submit<T: Send + 'static>(&self, f: impl FnOnce() -> T + Send + 'static) -> Result<tokio::sync::oneshot::Receiver<T>, EsError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let job: Job = Box::new(move || {
            let _ = tx.send(f());
        });
        let submitted = match self.sender.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            Some(sender) => sender.try_send(job),
            None => Err(TrySendError::Disconnected(job)),
        };
        if submitted.is_err() {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(EsError::new(429, "es_rejected_execution_exception", format!(
                "rejected execution of authentication on the [security-crypto] pool (queue capacity {}, {} threads)", self.queue, self.threads)));
        }
        Ok(rx)
    }

    /// Run `f` on the pool; `Err` (429) when its queue is full.
    pub async fn run<T: Send + 'static>(&self, f: impl FnOnce() -> T + Send + 'static) -> Result<T, EsError> {
        let rx = self.submit(f)?;
        let r = rx.await.map_err(|_| EsError::internal("authentication worker failed"))?;
        self.completed.fetch_add(1, Ordering::Relaxed);
        Ok(r)
    }

    pub fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }
}

impl Drop for CryptoPool {
    fn drop(&mut self) {
        self.sender.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}

static POOL: LazyLock<RwLock<Arc<CryptoPool>>> = LazyLock::new(|| {
    let processors = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    RwLock::new(CryptoPool::new(processors.div_ceil(2), 1000))
});

/// The edge's crypto pool.
pub fn crypto_pool() -> Arc<CryptoPool> {
    POOL.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Replace the crypto pool (node settings, tests).
pub fn set_crypto_pool(threads: usize, queue: usize) {
    *POOL.write().unwrap_or_else(|e| e.into_inner()) = CryptoPool::new(threads, queue);
}

/// Failed expensive verifications per key within a window.
pub struct FailureLimiter {
    entries: Mutex<HashMap<String, (Instant, u32)>>,
    pub per_user: u32,
    pub per_address: u32,
    pub window: Duration,
}

/// Distinct keys tracked before stale ones are swept.
const LIMITER_KEYS: usize = 65_536;

impl FailureLimiter {
    pub fn new(per_user: u32, per_address: u32, window: Duration) -> Self {
        FailureLimiter { entries: Mutex::new(HashMap::new()), per_user, per_address, window }
    }

    fn keys(user: Option<&str>, address: Option<IpAddr>) -> Vec<(String, bool)> {
        let mut keys = Vec::with_capacity(2);
        if let Some(user) = user {
            keys.push((format!("u\u{0}{user}"), true));
        }
        if let Some(address) = address {
            keys.push((format!("a\u{0}{address}"), false));
        }
        keys
    }

    /// Whether a new expensive verification for `user` from `address` is
    /// refused (too many recent failures).
    pub fn blocked(&self, user: Option<&str>, address: Option<IpAddr>) -> bool {
        let now = Instant::now();
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        Self::keys(user, address).iter().any(|(key, is_user)| {
            entries.get(key).is_some_and(|(start, count)| now.duration_since(*start) < self.window && *count >= if *is_user { self.per_user } else { self.per_address })
        })
    }

    pub fn failed(&self, user: Option<&str>, address: Option<IpAddr>) {
        let now = Instant::now();
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.len() >= LIMITER_KEYS {
            let window = self.window;
            entries.retain(|_, (start, _)| now.duration_since(*start) < window);
            if entries.len() >= LIMITER_KEYS {
                entries.clear();
            }
        }
        for (key, _) in Self::keys(user, address) {
            let e = entries.entry(key).or_insert((now, 0));
            if now.duration_since(e.0) >= self.window {
                *e = (now, 0);
            }
            e.1 = e.1.saturating_add(1);
        }
    }
}

static LIMITER: LazyLock<RwLock<Arc<FailureLimiter>>> = LazyLock::new(|| RwLock::new(Arc::new(FailureLimiter::new(100, 300, Duration::from_secs(60)))));

pub fn failure_limiter() -> Arc<FailureLimiter> {
    LIMITER.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Replace the failed-authentication limits (node settings, tests).
pub fn set_failure_limits(per_user: u32, per_address: u32, window: Duration) {
    *LIMITER.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(FailureLimiter::new(per_user, per_address, window));
}

/// Authenticate a request's `Authorization` header at the edge, without
/// ever running expensive verification on the calling (async) thread.
pub async fn authenticate(node: &Arc<Node>, authorization: Option<&str>, address: Option<IpAddr>) -> EdgeAuth {
    if !node.security.enabled() {
        return EdgeAuth::Authenticated;
    }
    match verify(node, super::credentials_from(authorization), address).await {
        Verified::Authenticated(_) => EdgeAuth::Authenticated,
        Verified::Failed(fail) => EdgeAuth::Failed(fail),
        Verified::Rejected(error) => EdgeAuth::Rejected(error),
    }
}

/// The outcome of [`verify`].
pub enum Verified {
    Authenticated(Authn),
    Failed(AuthFail),
    /// Not verified: crypto pool saturated or too many recent failures (429).
    Rejected(EsError),
}

/// Authenticate `creds` from any async context (the client edge, the
/// security layer of forwarded / in-process requests, the password grant
/// APIs), never running expensive verification on the calling thread:
/// cheap verdicts inline, uncached bcrypt / PBKDF2 on the crypto pool behind
/// the failed-authentication limiter.
pub async fn verify(node: &Arc<Node>, creds: Credentials, address: Option<IpAddr>) -> Verified {
    match node.security.authenticate_nonblocking(&creds) {
        Some(Ok(a)) => return Verified::Authenticated(a),
        Some(Err(fail)) => return Verified::Failed(fail),
        None => {}
    }
    let user = Security::credential_principal(&creds).map(str::to_owned);
    let limiter = failure_limiter();
    if limiter.blocked(user.as_deref(), address) {
        return Verified::Rejected(EsError::new(429, "es_rejected_execution_exception", "too many failed authentication attempts; retry later"));
    }
    let verifier = node.clone();
    let outcome: Result<Result<Authn, AuthFail>, EsError> = crypto_pool().run(move || verifier.security.authenticate(creds)).await;
    match outcome {
        Ok(Ok(a)) => Verified::Authenticated(a),
        Ok(Err(fail)) => {
            limiter.failed(user.as_deref(), address);
            Verified::Failed(fail)
        }
        Err(rejected) => Verified::Rejected(rejected),
    }
}

/// Password grants carried in a request body (`grant_type: password`:
/// OAuth2 token, API key grant, profile activation). The handlers decide
/// synchronously, so the credentials are verified here first, on the crypto
/// pool: a success is then a cache hit for the handler, and a failure is
/// never re-verified on the runtime ([`password_grant`]). `Err`: refused
/// before verification (429).
pub async fn preverify_password_grant(node: &Arc<Node>, body: &[u8]) -> Result<(), EsError> {
    if body.is_empty() || memchr::memmem::find(body, b"password").is_none() {
        return Ok(());
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else { return Ok(()) };
    if value["grant_type"].as_str() != Some("password") {
        return Ok(());
    }
    let (Some(user), Some(password)) = (value["username"].as_str(), value["password"].as_str()) else { return Ok(()) };
    match verify(node, Credentials::Basic(user.to_owned(), password.to_owned()), None).await {
        Verified::Rejected(error) => Err(error),
        _ => Ok(()),
    }
}

/// A body password grant's verdict inside a (synchronous) handler, after
/// [`preverify_password_grant`]: decided without expensive hashing on this
/// thread; an expensive verdict not established as a success is a failure.
pub fn password_grant(security: &Security, user: &str, password: &str) -> Result<Authn, AuthFail> {
    match security.authenticate_nonblocking(&Credentials::Basic(user.to_owned(), password.to_owned())) {
        Some(result) => result,
        None => Err(AuthFail::User(user.to_owned())),
    }
}

/// The 401 the security layer answers for `fail` on `method path?query`
/// (same reason / `WWW-Authenticate` as after dispatch), rendered:
/// (status, headers, JSON body).
pub fn failure_response(node: &Node, fail: AuthFail, method: &axum::http::Method, path: &str, query: Option<&str>) -> Option<(u16, Vec<(String, String)>, Vec<u8>)> {
    let mut metadata = crate::input_metadata::Builder::new(true).ok()?;
    let params = metadata.parameters(query).ok()?;
    let path = metadata.text(path).ok()?;
    let req = crate::Req {
        method: method.clone(),
        segs: path.split('/').filter(|s| !s.is_empty()).map(str::to_owned).collect(),
        path,
        params,
        body: Default::default(),
        headers: Default::default(),
        access: None,
        extensions: Default::default(),
        api_version: crate::RestApiVersion::CURRENT,
    };
    let resp = super::auth_failure(node, fail, &req);
    let body = match &resp.body {
        crate::RespBody::Json(v) => serde_json::to_vec(v).ok()?,
        _ => return None,
    };
    let mut headers: Vec<(String, String)> = resp.headers.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    headers.push(("content-type".into(), "application/json".into()));
    Some((resp.status, headers, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_counts_failures_per_user_and_address_within_a_window() {
        let l = FailureLimiter::new(3, 5, Duration::from_millis(200));
        let a: IpAddr = "10.0.0.1".parse().unwrap();
        let b: IpAddr = "10.0.0.2".parse().unwrap();
        for _ in 0..3 {
            assert!(!l.blocked(Some("u"), Some(a)));
            l.failed(Some("u"), Some(a));
        }
        assert!(l.blocked(Some("u"), Some(b)), "per-user limit applies from any address");
        assert!(!l.blocked(Some("v"), Some(a)), "address below its own limit");
        l.failed(Some("v"), Some(a));
        l.failed(Some("w"), Some(a));
        assert!(l.blocked(Some("x"), Some(a)), "per-address limit");
        std::thread::sleep(Duration::from_millis(250));
        assert!(!l.blocked(Some("u"), Some(a)), "window expired");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pool_rejects_beyond_its_queue_and_never_blocks_the_runtime() {
        use std::sync::atomic::AtomicBool;
        let pool = CryptoPool::new(1, 2);
        let release = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let blocker = |release: Arc<AtomicBool>, started: Option<std::sync::mpsc::Sender<()>>| move || {
            if let Some(s) = started { let _ = s.send(()); }
            while !release.load(Ordering::Acquire) { std::thread::sleep(Duration::from_millis(1)); }
            1
        };
        // The running job: wait until the crypto thread has taken it.
        let mut held = vec![pool.submit(blocker(release.clone(), Some(started_tx))).unwrap()];
        tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(5)).unwrap()).await.unwrap();
        // Two queued jobs fill the queue; the next is refused at once.
        for _ in 0..2 {
            held.push(pool.submit(blocker(release.clone(), None)).unwrap());
        }
        let err = pool.submit(|| 2).unwrap_err();
        assert_eq!(err.status, 429);
        assert!(pool.rejected() >= 1);
        // The async workers stay free while the crypto thread is blocked.
        let t = Instant::now();
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(t.elapsed() < Duration::from_millis(200));
        release.store(true, Ordering::Release);
        for h in held {
            assert_eq!(tokio::time::timeout(Duration::from_secs(5), h).await.unwrap().unwrap(), 1);
        }
    }
}
