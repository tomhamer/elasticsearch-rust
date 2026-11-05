//! `/_snapshot/*`: repositories, snapshots, restore, clone, status.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::snapshot::{repo_missing, SnapshotService};
use serde_json::{json, Value};
use std::sync::Arc;

fn flag_default(req: &Req, k: &str, d: bool) -> Result<bool, EsError> {
    Ok(req.bool_param(k)?.unwrap_or(d))
}

/// Repositories named by `expr` (`_all`, wildcards, comma lists).
fn repos_for(node: &Node, expr: &str) -> Result<Vec<String>, EsError> {
    let v = node.snapshots.get_repositories(Some(expr))?;
    Ok(v.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default())
}

/// `repositories.default_repository` (transient over persistent; unset when empty).
pub(crate) fn default_repository(node: &Node) -> Option<String> {
    const KEY: &str = "repositories.default_repository";
    let cs = node.cluster_settings.read();
    cs.1.get(KEY).or(cs.0.get(KEY)).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string)
}

fn readonly_setting(settings: &Value) -> bool {
    matches!(settings.get("readonly"), Some(Value::Bool(true))) || settings.get("readonly").and_then(Value::as_str).is_some_and(|v| v == "true")
}

/// The default repository cannot be created or marked read-only.
fn check_default_readonly(node: &Node, repo: &str, body: &Value) -> Result<(), EsError> {
    if default_repository(node).as_deref() != Some(repo) || !readonly_setting(body.get("settings").unwrap_or(&Value::Null)) {
        return Ok(());
    }
    if !node.snapshots.has_repository(repo) {
        return Err(EsError::illegal_argument(format!(
            "Repository [{repo}] is the default repository. Cannot create the default repository as read-only. Please change the default repository setting before creating this repository as read-only."
        )));
    }
    if !node.snapshots.is_read_only(repo) {
        return Err(EsError::illegal_argument(format!(
            "Repository [{repo}] is the default repository. Cannot mark the default repository as read-only. Please change the default repository setting before marking this repository as read-only."
        )));
    }
    Ok(())
}

/// `RepositoriesService.validateDefaultRepository` (the setting's update validator).
pub(crate) fn validate_default_repository(node: &Node, value: &str) -> Result<(), EsError> {
    let current = default_repository(node).unwrap_or_default();
    if value.is_empty() || value == current {
        return Ok(());
    }
    let cause = if node.snapshots.registered_readonly(value).is_none() {
        format!("Repository [{value}] is not registered. Cannot set as default repository. Please register the repository prior to setting as default using PUT /_snapshot/{value}")
    } else if node.snapshots.registered_readonly(value) == Some(true) {
        format!("Repository [{value}] is marked as read-only. Cannot set a read-only repository as the default repository.")
    } else {
        return Ok(());
    };
    Err(EsError::illegal_argument(format!("illegal value can't update [repositories.default_repository] from [{current}] to [{value}]")).caused_by(EsError::illegal_argument(cause)))
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let repo = req.seg(1).to_string();
    let snap = req.seg(2).to_string();
    let sub = req.seg(3).to_string();
    let m = req.method.clone();
    let put = m == Method::PUT || m == Method::POST;
    match (repo.as_str(), snap.as_str(), sub.as_str()) {
        ("", _, _) => match m {
            Method::GET => blocking(move || Ok(Resp::ok(node.snapshots.get_repositories(None)?))).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("_status", "", _) if m == Method::GET => blocking(move || Ok(Resp::ok(json!({"snapshots": []})))).await,
        (_, "", _) => match m {
            Method::GET => blocking(move || Ok(Resp::ok(node.snapshots.get_repositories(Some(&repo))?))).await,
            _ if put => blocking(move || {
                // `?verify=false` skips the post-registration verification.
                let verify = flag_default(&req, "verify", true)?;
                let body = req.json()?;
                check_default_readonly(&node, &repo, &body)?;
                node.snapshots.put_repository(&repo, &body)?;
                if verify {
                    node.snapshots.verify(&repo, &node.name)?;
                }
                Ok(Resp::ack())
            })
            .await,
            Method::DELETE => blocking(move || {
                // A repository backing searchable snapshot indices cannot be unregistered.
                for name in repos_for(&node, &repo)? {
                    // `RepositoriesService.ensureRepositoryNotDefault`.
                    if default_repository(&node).as_deref() == Some(name.as_str()) {
                        return Err(EsError::new(
                            409,
                            "repository_conflict_exception",
                            format!("[{name}] trying to modify or unregister repository that is currently used (cannot delete the default repository. Please change the default repository cluster setting repositories.default_repository before deleting this repository.)"),
                        ));
                    }
                    let used = node.snapshots.indices_using_repository(&node, &name);
                    if !used.is_empty() {
                        let shown: Vec<String> = used.iter().take(5).map(|s| format!("[{}/{}]", s.name, s.uuid)).collect();
                        let more = if used.len() > 5 { ",..." } else { "" };
                        return Err(EsError::new(
                            409,
                            "repository_conflict_exception",
                            format!("[{name}] trying to modify or unregister repository that is currently used (found {} searchable snapshots indices that use the repository: {}{more})", used.len(), shown.join(",")),
                        ));
                    }
                }
                node.snapshots.delete_repository(&repo)?;
                Ok(Resp::ack())
            })
            .await,
            _ => method_not_allowed(&req, &["GET", "PUT", "POST", "DELETE"]),
        },
        (_, "_analyze", "") if m == Method::POST => blocking(move || analyze(&node, &req, &repo)).await,
        (_, "_verify_integrity", "") if m == Method::POST => blocking(move || verify_integrity(&node, &req, &repo)).await,
        (_, "_verify", "") if put => blocking(move || Ok(Resp::ok(node.snapshots.verify(&repo, &node.name)?))).await,
        (_, "_cleanup", "") if put => blocking(move || {
            node.snapshots.verify(&repo, &node.name)?;
            Ok(Resp::ok(json!({"results": {"deleted_bytes": 0, "deleted_blobs": 0}})))
        })
        .await,
        (_, "_status", "") if m == Method::GET => blocking(move || status(&node, &req, &repo, "_all")).await,
        (_, _, "_status") if m == Method::GET => blocking(move || status(&node, &req, &repo, &snap)).await,
        (_, _, "_restore") if put => blocking(move || {
            let body = req.json()?;
            let r = node.snapshots.restore(&node, &repo, &snap, &body)?;
            if flag_default(&req, "wait_for_completion", false)? {
                Ok(Resp::ok(r))
            } else {
                Ok(Resp::json(202, json!({"accepted": true})))
            }
        })
        .await,
        (_, _, "_mount") if m == Method::POST => blocking(move || mount(&node, &req, &repo, &snap)).await,
        (_, _, "") => match m {
            Method::GET => blocking(move || get(&node, &req, &repo, &snap)).await,
            _ if put => blocking(move || {
                let body = req.json()?;
                let meta = node.snapshots.create_snapshot(&node, &repo, &snap, &body)?;
                if flag_default(&req, "wait_for_completion", false)? {
                    Ok(Resp::ok(json!({"snapshot": SnapshotService::info_json(&meta, true, false, true, false)})))
                } else {
                    Ok(Resp::ok(json!({"accepted": true})))
                }
            })
            .await,
            Method::DELETE => blocking(move || {
                node.snapshots.delete_snapshots(&repo, &snap)?;
                Ok(Resp::ack())
            })
            .await,
            _ => method_not_allowed(&req, &["GET", "PUT", "POST", "DELETE"]),
        },
        _ => crate::not_found_handler(&req),
    }
}

/// `POST /_snapshot/{repo}/{snap}/_mount` (`RestMountSearchableSnapshotAction`).
fn mount(node: &Node, req: &Req, repo: &str, snap: &str) -> R {
    let storage = req.param("storage").unwrap_or("full_copy").to_string();
    let partial = match storage.to_uppercase().as_str() {
        "FULL_COPY" => false,
        "SHARED_CACHE" => true,
        other => {
            return Err(EsError::illegal_argument(format!(
                "No enum constant org.elasticsearch.xpack.core.searchablesnapshots.MountSearchableSnapshotRequest.Storage.{other}"
            )))
        }
    };
    let wait = flag_default(req, "wait_for_completion", false)?;
    let body = req.json()?;
    let obj = body.as_object().cloned().unwrap_or_default();
    let parse_err = |msg: String| EsError::x_content_parse(format!("[1:2] {msg}"));
    for k in obj.keys() {
        if !matches!(k.as_str(), "index" | "renamed_index" | "index_settings" | "ignore_index_settings" | "ignored_index_settings") {
            return Err(parse_err(format!("[mount_searchable_snapshot] unknown field [{k}]")));
        }
    }
    let index = obj.get("index").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [index]"))?.to_string();
    let mounted = obj.get("renamed_index").and_then(Value::as_str).unwrap_or(&index).to_string();
    let settings = match obj.get("index_settings") {
        Some(Value::Object(o)) => o.clone(),
        Some(Value::Null) | None => serde_json::Map::new(),
        Some(_) => return Err(parse_err("[mount_searchable_snapshot] failed to parse field [index_settings]".into())),
    };
    let ignore: Vec<String> = match obj.get("ignore_index_settings") {
        Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    };
    let r = node.snapshots.mount(node, repo, snap, &index, &mounted, &settings, &ignore, partial)?;
    if wait {
        Ok(Resp::ok(r))
    } else {
        Ok(Resp::json(202, json!({"accepted": true})))
    }
}

/// `/_snapshot/{repo}/{snap}/_clone/{target}` (one more path segment).
pub async fn handle_clone(node: Arc<Node>, req: Req) -> R {
    let (repo, snap, target) = (req.seg(1).to_string(), req.seg(2).to_string(), req.seg(4).to_string());
    if req.method != Method::PUT && req.method != Method::POST {
        return method_not_allowed(&req, &["PUT", "POST"]);
    }
    blocking(move || {
        node.snapshots.clone_snapshot(&repo, &snap, &target, &req.json()?)?;
        Ok(Resp::ack())
    })
    .await
}

fn get(node: &Node, req: &Req, repo: &str, snap: &str) -> R {
    let verbose = flag_default(req, "verbose", true)?;
    let index_details = flag_default(req, "index_details", false)?;
    let index_names = flag_default(req, "index_names", true)?;
    let include_repo = flag_default(req, "include_repository", true)?;
    let ignore = flag_default(req, "ignore_unavailable", false)?;
    // `RestGetSnapshotsAction`: `?state=` (comma list of `SnapshotState`s).
    let states: Option<Vec<&str>> = match req.param("state") {
        None => None,
        Some(s) if s.trim().is_empty() => return Err(EsError::illegal_argument("[state] parameter must not be empty")),
        Some(s) => {
            let list: Vec<&str> = s.split(',').collect();
            if let Some(bad) = list.iter().find(|x| !matches!(**x, "IN_PROGRESS" | "SUCCESS" | "FAILED" | "PARTIAL" | "INCOMPATIBLE")) {
                return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.snapshots.SnapshotState.{bad}")));
            }
            Some(list)
        }
    };
    let mut out = Vec::new();
    let repos = repos_for(node, repo)?;
    for r in &repos {
        for meta in node.snapshots.get_snapshots(r, snap, ignore)? {
            let info = SnapshotService::info_json(&meta, verbose, index_details, index_names, include_repo);
            if states.as_ref().is_none_or(|st| info.get("state").and_then(Value::as_str).is_some_and(|x| st.contains(&x))) {
                out.push(info);
            }
        }
    }
    if repos.is_empty() && !repo.contains('*') && repo != "_all" {
        return Err(repo_missing(repo));
    }
    let n = out.len();
    Ok(Resp::ok(json!({"snapshots": out, "total": n, "remaining": 0})))
}

fn status(node: &Node, req: &Req, repo: &str, snap: &str) -> R {
    let ignore = flag_default(req, "ignore_unavailable", false)?;
    let mut out: Vec<Value> = Vec::new();
    if snap == "_all" || snap == "_current" {
        // Only running snapshots are reported; snapshots here complete synchronously.
        node.snapshots.verify(repo, &node.name)?;
        return Ok(Resp::ok(json!({"snapshots": out})));
    }
    for meta in node.snapshots.get_snapshots(repo, snap, ignore)? {
        out.push(node.snapshots.status_json(repo, &meta)?);
    }
    Ok(Resp::ok(json!({"snapshots": out})))
}

fn num_param<T: std::str::FromStr>(req: &Req, k: &str, kind: &str, d: T) -> Result<T, EsError> {
    match req.param(k) {
        None => Ok(d),
        Some(v) => v.parse::<T>().map_err(|_| {
            EsError::illegal_argument(format!("Failed to parse {kind} parameter [{k}] with value [{v}]"))
                .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{v}\"")))
        }),
    }
}

fn bytes_param(req: &Req, k: &str, d: &str) -> Result<(String, i64), EsError> {
    let v = req.param(k).unwrap_or(d);
    let n = elasticsearch_engine::ingest::util::parse_bytes(v).ok_or_else(|| {
        EsError::new(400, "parse_exception", format!("failed to parse setting [{k}] with value [{v}] as a size in bytes: unit is missing or unrecognized"))
    })?;
    Ok((crate::util::human_bytes(n.max(0) as u64), n))
}

fn time_param(req: &Req, k: &str) -> Result<(), EsError> {
    if let Some(v) = req.param(k) {
        elasticsearch_engine::date::parse_time_value(v)
            .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{v}] as a time value: unit is missing or unrecognized")))?;
    }
    Ok(())
}

fn positive(name: &str, v: i64) -> Result<(), EsError> {
    if v <= 0 {
        return Err(EsError::illegal_argument(format!("{name} must be >0, but was [{v}]")));
    }
    Ok(())
}

/// `POST /_snapshot/{repo}/_analyze` (repository analysis). Parameters are
/// parsed and validated as ES does; the analysis itself is a single-node
/// simulation (no blob I/O is performed against the repository).
fn analyze(node: &Node, req: &Req, repo: &str) -> R {
    let blob_count: i64 = num_param(req, "blob_count", "int", 100)?;
    let concurrency: i64 = num_param(req, "concurrency", "int", 10)?;
    let read_node_count: i64 = num_param(req, "read_node_count", "int", 10)?;
    let early_read_node_count: i64 = num_param(req, "early_read_node_count", "int", 2)?;
    let register_operation_count: i64 = num_param(req, "register_operation_count", "int", 10)?;
    let seed: i64 = num_param(req, "seed", "long", (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0) % 100_000_000) as i64)?;
    let rare: f64 = num_param(req, "rare_action_probability", "double", 0.02)?;
    let (max_blob_h, max_blob) = bytes_param(req, "max_blob_size", "10mb")?;
    let (max_total_h, max_total) = bytes_param(req, "max_total_data_size", "1gb")?;
    time_param(req, "timeout")?;
    let detailed = flag_default(req, "detailed", false)?;
    flag_default(req, "rarely_abort_writes", true)?;
    positive("blobCount", blob_count)?;
    positive("concurrency", concurrency)?;
    positive("readNodeCount", read_node_count)?;
    positive("registerOperationCount", register_operation_count)?;
    if max_blob <= 0 {
        return Err(EsError::illegal_argument(format!("maxBlobSize must be >0, but was [{max_blob_h}]")));
    }
    if !(0.0..=1.0).contains(&rare) {
        return Err(EsError::illegal_argument(format!("rareActionProbability must be between 0 and 1, but was [{rare:?}]")));
    }
    // One byte per blob plus one max-sized blob.
    let min_total = blob_count - 1 + max_blob;
    if max_total < min_total {
        return Err(EsError::illegal_argument(format!(
            "cannot satisfy max total bytes [{max_total}B/{max_total_h}]: must write at least one byte per blob and at least one max-sized blob which is [{min_total}B] in total"
        )));
    }
    if node.snapshots.get_repositories(Some(repo)).ok().and_then(|v| v.as_object().map(|o| o.contains_key(repo))) != Some(true) {
        return Err(repo_missing(repo));
    }
    if node.snapshots.is_read_only(repo) {
        return Err(EsError::illegal_argument(format!("repository [{repo}] is read-only")));
    }
    let timeout_s = req.param("timeout").unwrap_or("30s").to_string();
    let timeout_ms = elasticsearch_engine::date::parse_time_value(&timeout_s).unwrap_or(30_000).max(0) as u64;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    // `max_snapshot_bytes_per_sec` throttles writes (0 / unset: unthrottled).
    let rate = |k: &str| -> f64 {
        node.snapshots
            .repository_setting(repo, k)
            .and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_i64().map(|n| n.to_string())))
            .and_then(|v| elasticsearch_engine::ingest::util::parse_bytes(&v))
            .map(|n| n as f64)
            .unwrap_or(0.0)
    };
    let (write_rate, read_rate) = (rate("max_snapshot_bytes_per_sec"), rate("max_restore_bytes_per_sec"));
    // Deterministic blob sizes from the seed (xorshift), capped by the totals.
    let mut x = (seed as u64) ^ 0x9E37_79B9_7F4A_7C15;
    let mut total = 0i64;
    let mut sizes = Vec::new();
    for i in 0..blob_count {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let left = max_total - total - (blob_count - 1 - i);
        let size = if i == 0 { max_blob.min(left) } else { (1 + (x % max_blob as u64) as i64).min(left).max(1) };
        total += size;
        sizes.push(size);
    }
    let blob_path = format!("temp-analysis-{}", elasticsearch_engine::index::new_uuid());
    let dir = node.snapshots.repository_dir(repo)?.join(&blob_path);
    let timed_out = |path: &str| {
        EsError::new(
            500,
            "repository_verification_exception",
            format!(
                "[{repo}] Repository analysis timed out. Consider specifying a longer timeout using the [?timeout] request parameter. See [https://www.elastic.co/guide/en/elasticsearch/reference/8.19/repo-analysis-api.html] for more information. Elasticsearch attempted to remove the data it wrote at [{path}] but may have left some behind. If so, please now remove this data manually."
            ),
        )
        .caused_by(EsError::new(500, "repository_verification_exception", format!("[{repo}] analysis timed out after [{timeout_s}]")))
    };
    let io_err = |e: std::io::Error| EsError::new(500, "repository_verification_exception", format!("[{repo}] {e}"));
    std::fs::create_dir_all(&dir).map_err(io_err)?;
    // Throttle `done` bytes since `start` to `bytes_per_sec`; false on timeout.
    let throttle = |start: std::time::Instant, done: usize, bytes_per_sec: f64, throttled: &mut u128| -> bool {
        if bytes_per_sec <= 0.0 {
            return std::time::Instant::now() < deadline;
        }
        let due = start + std::time::Duration::from_secs_f64((done as f64 / bytes_per_sec).min(1e9));
        let now = std::time::Instant::now();
        if due > now {
            let wait = (due.min(deadline)).saturating_duration_since(now);
            std::thread::sleep(wait);
            *throttled += wait.as_nanos();
            if due > deadline {
                return false;
            }
        }
        std::time::Instant::now() < deadline
    };
    const CHUNK: usize = 64 * 1024;
    let (mut w_elapsed, mut w_throttled, mut r_elapsed, mut r_throttled) = (0u128, 0u128, 0u128, 0u128);
    let mut details = Vec::new();
    let mut failure = None;
    'blobs: for (i, size) in sizes.iter().enumerate() {
        let name = format!("test-blob-{i}");
        let path = dir.join(&name);
        // Write.
        let start = std::time::Instant::now();
        let mut throttled = 0u128;
        let mut rng = (seed as u64).wrapping_add(i as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
        let mut hasher: u64 = 0xcbf2_9ce4_8422_2325;
        let mut f = match std::fs::File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                failure = Some(io_err(e));
                break;
            }
        };
        let mut written = 0usize;
        while written < *size as usize {
            let n = CHUNK.min(*size as usize - written);
            let mut buf = vec![0u8; n];
            for b in buf.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                *b = rng as u8;
            }
            hasher = fnv(hasher, &buf);
            if let Err(e) = std::io::Write::write_all(&mut f, &buf) {
                failure = Some(io_err(e));
                break 'blobs;
            }
            written += n;
            if !throttle(start, written, write_rate, &mut throttled) {
                failure = Some(timed_out(&blob_path));
                break 'blobs;
            }
        }
        drop(f);
        let write_nanos = start.elapsed().as_nanos();
        w_elapsed += write_nanos;
        w_throttled += throttled;
        let expected = hasher;
        // Read back and verify.
        let start = std::time::Instant::now();
        let mut throttled = 0u128;
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                failure = Some(io_err(e));
                break;
            }
        };
        if !throttle(start, data.len(), read_rate, &mut throttled) {
            failure = Some(timed_out(&blob_path));
            break;
        }
        if data.len() != *size as usize || fnv(0xcbf2_9ce4_8422_2325, &data) != expected {
            failure = Some(EsError::new(500, "repository_verification_exception", format!("[{repo}] failure processing [{name}]: checksum mismatch")));
            break;
        }
        let read_nanos = start.elapsed().as_nanos();
        r_elapsed += read_nanos;
        r_throttled += throttled;
        details.push(json!({
            "blob": {"name": name, "size": crate::util::human_bytes(*size as u64), "size_bytes": size, "read_start": 0, "read_end": size, "read_early": false, "overwritten": false},
            "node": {"id": elasticsearch_engine::node::NODE_ID, "name": node.name},
            "write_elapsed": crate::util::human_time_nanos(write_nanos as u64), "write_elapsed_nanos": write_nanos as u64,
            "write_throttled": crate::util::human_time_nanos(throttled as u64), "write_throttled_nanos": throttled as u64,
            "reads": [{
                "node": {"id": elasticsearch_engine::node::NODE_ID, "name": node.name},
                "first_byte_time": crate::util::human_time_nanos(read_nanos as u64), "first_byte_time_nanos": read_nanos as u64,
                "elapsed": crate::util::human_time_nanos(read_nanos as u64), "elapsed_nanos": read_nanos as u64,
                "throttled": crate::util::human_time_nanos(throttled as u64), "throttled_nanos": throttled as u64,
            }],
        }));
    }
    // Listing, then deleting what was written (also after a failure).
    let list_start = std::time::Instant::now();
    let listed = std::fs::read_dir(&dir).map(|rd| rd.count()).unwrap_or(0);
    let listing_nanos = list_start.elapsed().as_nanos() as u64;
    let del_start = std::time::Instant::now();
    let _ = std::fs::remove_dir_all(&dir);
    let delete_nanos = del_start.elapsed().as_nanos() as u64;
    if let Some(e) = failure {
        return Err(e);
    }
    if listed != sizes.len() {
        return Err(EsError::new(500, "repository_verification_exception", format!("[{repo}] expected to list [{}] blobs but found [{listed}]", sizes.len())));
    }
    let dur = |nanos: u128| crate::util::human_time_nanos(nanos as u64);
    let mut out = json!({
        "coordinating_node": {"id": elasticsearch_engine::node::NODE_ID, "name": node.name},
        "repository": repo,
        "blob_count": blob_count,
        "concurrency": concurrency,
        "read_node_count": read_node_count,
        "early_read_node_count": early_read_node_count,
        "max_blob_size": max_blob_h,
        "max_blob_size_bytes": max_blob,
        "max_total_data_size": max_total_h,
        "max_total_data_size_bytes": max_total,
        "seed": seed,
        "rare_action_probability": rare,
        "blob_path": blob_path,
        "issues_detected": [],
        "summary": {
            "write": {"count": blob_count, "total_size": crate::util::human_bytes(total as u64), "total_size_bytes": total, "total_throttled": dur(w_throttled), "total_throttled_nanos": w_throttled as u64, "total_elapsed": dur(w_elapsed), "total_elapsed_nanos": w_elapsed as u64},
            "read": {"count": blob_count, "total_size": crate::util::human_bytes(total as u64), "total_size_bytes": total, "total_wait": "0s", "total_wait_nanos": 0, "max_wait": "0s", "max_wait_nanos": 0, "total_throttled": dur(r_throttled), "total_throttled_nanos": r_throttled as u64, "total_elapsed": dur(r_elapsed), "total_elapsed_nanos": r_elapsed as u64},
        },
        "listing_elapsed": dur(listing_nanos as u128),
        "listing_elapsed_nanos": listing_nanos,
        "delete_elapsed": dur(delete_nanos as u128),
        "delete_elapsed_nanos": delete_nanos,
    });
    if detailed {
        out["details"] = Value::Array(details);
    }
    // `human` defaults to true here; without it only raw values remain.
    if req.bool_param("human")? == Some(false) {
        fn strip(v: &mut Value) {
            match v {
                Value::Object(m) => {
                    let raw: Vec<String> = m.keys().filter_map(|k| k.strip_suffix("_nanos").or_else(|| k.strip_suffix("_bytes")).map(str::to_string)).collect();
                    for k in raw {
                        m.shift_remove(&k);
                    }
                    m.values_mut().for_each(strip);
                }
                Value::Array(a) => a.iter_mut().for_each(strip),
                _ => {}
            }
        }
        strip(&mut out);
    }
    Ok(Resp::ok(out))
}

/// FNV-1a over `b`, continuing from `h` (blob content check).
fn fnv(mut h: u64, b: &[u8]) -> u64 {
    for x in b {
        h ^= *x as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// `POST /_snapshot/{repo}/_verify_integrity`: checks every snapshot's
/// metadata is readable; blob contents are not re-read.
fn verify_integrity(node: &Node, req: &Req, repo: &str) -> R {
    for k in ["meta_thread_pool_concurrency", "blob_thread_pool_concurrency", "snapshot_verification_concurrency", "index_verification_concurrency", "index_snapshot_verification_concurrency", "max_failed_shard_snapshots"] {
        let v: i64 = num_param(req, k, "int", 0)?;
        if v < 0 {
            return Err(EsError::illegal_argument(format!("argument [{k}] must be at least [0]")));
        }
    }
    let verify_contents = flag_default(req, "verify_blob_contents", false)?;
    bytes_param(req, "max_bytes_per_sec", "40mb")?;
    let repos = node.snapshots.get_repositories(Some(repo)).ok();
    let Some(repo_meta) = repos.as_ref().and_then(|v| v.get(repo)).cloned() else { return Err(repo_missing(repo)) };
    let snaps = node.snapshots.get_snapshots(repo, "_all", true)?;
    if snaps.is_empty() {
        return Err(EsError::illegal_argument("repository is empty, cannot verify its integrity"));
    }
    let now = chrono::Utc::now().timestamp_millis();
    let dir = node.snapshots.repository_dir(repo)?;
    let mut log = Vec::new();
    // index -> (snapshots containing it, restorable ones)
    let mut per_index: std::collections::BTreeMap<String, (usize, usize)> = std::collections::BTreeMap::new();
    let (mut index_snapshots, mut index_snapshots_ok, mut blobs, mut anomalies) = (0usize, 0usize, 0u64, 0usize);
    let mut snaps_ok = 0usize;
    // Every file of a snapshotted index is a blob: present, and (with
    // `verify_blob_contents`) fully readable.
    fn check_blobs(p: &std::path::Path, read: bool, n: &mut u64, bad: &mut Vec<String>) {
        match std::fs::read_dir(p) {
            Ok(rd) => {
                for e in rd.flatten() {
                    let path = e.path();
                    if path.is_dir() {
                        check_blobs(&path, read, n, bad);
                    } else {
                        *n += 1;
                        if read && std::fs::read(&path).is_err() {
                            bad.push(path.to_string_lossy().into_owned());
                        }
                    }
                }
            }
            Err(_) => bad.push(p.to_string_lossy().into_owned()),
        }
    }
    for meta in &snaps {
        let snap = meta["snapshot"].as_str().unwrap_or("");
        let info = SnapshotService::info_json(meta, true, false, true, true);
        let failed: Vec<&str> = meta["failures"].as_array().into_iter().flatten().filter_map(|f| f["index"].as_str()).collect();
        let mut snap_ok = true;
        for i in meta["indices"].as_array().into_iter().flatten().filter_map(Value::as_str) {
            index_snapshots += 1;
            let e = per_index.entry(i.to_string()).or_default();
            e.0 += 1;
            if failed.contains(&i) {
                continue;
            }
            let idir = dir.join(format!("snap-{snap}")).join(i);
            let mut bad = Vec::new();
            let meta_ok = std::fs::read(idir.join("index_meta.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).is_some();
            if !meta_ok {
                bad.push(idir.join("index_meta.json").to_string_lossy().into_owned());
            }
            check_blobs(&idir, verify_contents, &mut blobs, &mut bad);
            if bad.is_empty() {
                e.1 += 1;
                index_snapshots_ok += 1;
            } else {
                snap_ok = false;
                for b in bad {
                    anomalies += 1;
                    log.push(json!({"timestamp_in_millis": now, "anomaly": "missing or unreadable blob", "snapshot": info.clone(), "index": {"name": i}, "blob_name": b}));
                }
            }
        }
        if snap_ok {
            snaps_ok += 1;
        }
        log.push(json!({"timestamp_in_millis": now, "snapshot": info}));
    }
    for (name, (n, ok)) in &per_index {
        log.push(json!({"timestamp_in_millis": now, "index": {"name": name}, "snapshot_restorability": {"total_snapshot_count": n, "restorable_snapshot_count": ok}}));
    }
    let indices_ok = per_index.values().filter(|(n, ok)| n == ok).count();
    Ok(Resp::ok(json!({
        "log": log,
        "results": {
            "status": {
                "repository": {"name": repo, "uuid": repo_meta.get("uuid").cloned().unwrap_or(json!("_na_")), "generation": snaps.len() as i64 - 1},
                "snapshots": {"verified": snaps_ok, "total": snaps.len()},
                "indices": {"verified": indices_ok, "total": per_index.len()},
                "index_snapshots": {"verified": index_snapshots_ok, "total": index_snapshots},
                "blobs": {"verified": blobs}
            },
            "final_repository_generation": snaps.len() as i64 - 1,
            "total_anomalies": anomalies,
            "result": if anomalies == 0 { "pass" } else { "fail" }
        }
    })))
}
