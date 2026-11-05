// ---------------------------------------------------------------------------
// Appended to es-yaml-runner's `exec.rs` by es-cluster-yaml's build.rs (same
// module, so the private `Ctx` API is usable unchanged): run a test section
// against an external cluster instead of a fresh in-process node.
// ---------------------------------------------------------------------------

static BASES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
static NEXT_BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Use only with a matching module-specific file-realm server fixture.
pub static MODULE_SECURITY_FIXTURES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// File-realm REST-test credentials when --security is selected.
pub fn external_auth_header() -> Option<String> { default_auth() }

/// HTTP base URLs of every node; requests round-robin over them.
pub fn set_bases(b: Vec<String>) {
    let _ = BASES.set(b);
}

#[allow(dead_code)]
fn pick_base(default: &str) -> String {
    match BASES.get() {
        Some(b) if !b.is_empty() => b[NEXT_BASE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % b.len()].clone(),
        _ => default.to_string(),
    }
}

/// `http://h:p/path?q` → `http://h:p`.
#[allow(dead_code)]
fn base_of(url: &str) -> &str {
    let after = url.find("://").map(|i| i + 3).unwrap_or(0);
    match url[after..].find('/') {
        Some(j) => &url[..after + j],
        None => url,
    }
}

static REFERENCE_BASE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Reference gates determine required coverage without changing product features.
pub fn set_reference_base(base: String) { let _ = REFERENCE_BASE.set(base); }

async fn reference_capability_supported(base: &str, entry: &Value) -> Result<bool, String> {
    let method = entry["method"].as_str().unwrap_or("GET");
    let path = entry["path"].as_str().unwrap_or("/");
    let params = str_list(&entry["parameters"]).join(",");
    let caps = str_list(&entry["capabilities"]).join(",");
    let response = reqwest::Client::new().get(format!("{base}/_capabilities"))
        .query(&[("method", method), ("path", path), ("parameters", &params), ("capabilities", &caps)])
        .timeout(Duration::from_secs(30)).send().await.map_err(|e| format!("reference capability request failed: {e}"))?;
    if !response.status().is_success() { return Err(format!("reference capability request returned {}", response.status())); }
    let value = response.json::<Value>().await.map_err(|e| format!("invalid reference capability response: {e}"))?;
    value["supported"].as_bool().ok_or_else(|| "reference capability response has no boolean supported field".into())
}

static CAPS_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>> = std::sync::OnceLock::new();

/// Ask the cluster's `GET /_capabilities` whether one `capabilities:` entry of
/// a `requires`/`skip` clause is supported. Answers are cached for the run.
/// Anything but an explicit `"supported": true` (error, `null`, old node
/// without the API) counts as unsupported, as in ES's test runner.
async fn capability_supported(base: &str, entry: &Value) -> (String, bool) {
    let method = entry["method"].as_str().unwrap_or("GET").to_string();
    let path = entry["path"].as_str().unwrap_or("/").to_string();
    let params = str_list(&entry["parameters"]).join(",");
    let caps = str_list(&entry["capabilities"]).join(",");
    let label = format!("{method} {path} params=[{params}] caps=[{caps}]");
    // Match the in-process runner's explicit diagnostic mode. Normal runs
    // still query the product; forced runs cannot establish conformance.
    if claim_all() {
        return (label, true);
    }
    let cache = CAPS_CACHE.get_or_init(Default::default);
    let cache_key = format!("{base} {label}");
    if let Some(v) = cache.lock().unwrap().get(&cache_key) {
        return (label, *v);
    }
    let mut q: Vec<(&str, &str)> = vec![("method", &method), ("path", &path)];
    if !params.is_empty() {
        q.push(("parameters", &params));
    }
    if !caps.is_empty() {
        q.push(("capabilities", &caps));
    }
    let client = reqwest::Client::new();
    let mut request = client.get(format!("{base}/_capabilities")).query(&q).timeout(Duration::from_secs(30));
    if let Some(auth) = default_auth() { request = request.header("Authorization", auth); }
    let ok = match request.send().await {
        Ok(r) if r.status().is_success() => r.json::<Value>().await.ok().map(|v| v["supported"] == Value::Bool(true)).unwrap_or(false),
        _ => false,
    };
    cache.lock().unwrap().insert(cache_key, ok);
    (label, ok)
}

/// `prereq_skip` plus `capabilities` clauses resolved against the cluster
/// (the in-process runner skips every test that names one).
async fn cluster_prereq_skip(ctx: &mut Ctx<'_>, step: &Value, base: &str, required_missing: &mut Vec<String>) -> Result<Option<String>, String> {
    let Some(obj) = step.as_object() else { return Ok(None) };
    for (op, body) in obj {
        if op != "skip" && op != "requires" {
            continue;
        }
        let entries: Vec<Value> = match &body["capabilities"] {
            Value::Array(a) => a.clone(),
            Value::Null => vec![],
            other => vec![other.clone()],
        };
        // Everything except `capabilities` keeps the shared runner's semantics.
        let mut rest = body.clone();
        if let Some(m) = rest.as_object_mut() {
            m.remove("capabilities");
        }
        let mut one = Map::new();
        one.insert(op.clone(), rest);
        // Unsupported runner features skip, as in ES's runner.
        if let Some(r) = ctx.prereq_skip(&Value::Object(one)).await {
            return Ok(Some(r));
        }
        if entries.is_empty() {
            continue;
        }
        let mut missing = Vec::new();
        for e in &entries {
            let (label, ok) = capability_supported(base, e).await;
            if !ok {
                if op == "requires" {
                    let needed = match REFERENCE_BASE.get() {
                        Some(reference) => reference_capability_supported(reference, e).await?,
                        None => crate::compat::reference_capability(e) != Some(false),
                    };
                    if needed {
                        required_missing.push(label);
                        continue;
                    }
                }
                missing.push(label);
            }
        }
        if op == "requires" && !missing.is_empty() {
            return Ok(Some(format!("requires capabilities not supported: {}", missing.join("; "))));
        }
        if op == "skip" && missing.is_empty() {
            return Ok(Some("skip capabilities supported".into()));
        }
    }
    Ok(None)
}

/// Export the same module realms used by the shared runner for real external fixtures.
pub fn external_fixture(module: &str) -> Value {
    let mut cfg = match module {
        // x-pack main's `inference/` tests authenticate as `inference_user`
        // (the shared runner switches realms per directory; one fixture
        // serves the whole suite here).
        "main" => {
            let mut c = security_config();
            let inf = inference_security_config().0;
            c.file_users.extend(inf.file_users);
            c.file_user_roles.extend(inf.file_user_roles);
            c.file_roles.extend(inf.file_roles);
            c
        }
        "sec" => security_config(),
        "ent-search" => entsearch_security_config().0,
        "fleet" => fleet_security_config().0,
        "ccr" => ccr_security_config().0,
        "inference" => inference_security_config().0,
        "searchable-snapshots" => admin_superuser_config().0,
        "watcher-security" => watcher_security_config(),
        _ => elasticsearch_engine::security::SecConfig::default(),
    };
    // A stable file administrator handles cleanup; native feature reset cannot remove it.
    let admin = security_config();
    cfg.file_users.extend(admin.file_users);
    cfg.file_user_roles.extend(admin.file_user_roles);
    cfg.file_service_tokens.extend(admin.file_service_tokens);
    let (stack, apm, otel, prometheus) = if matches!(module, "main" | "sec" | "oss") {
        (true, true, true, true)
    } else { builtin_registries(module) };
    let settings = serde_json::json!({
        "xpack.security.enabled":cfg.enabled.to_string(),
        "xpack.security.authc.api_key.enabled":cfg.api_key_enabled.to_string(),
        "xpack.security.authc.token.enabled":cfg.token_enabled.to_string(),
        // `LicenseClientYamlTestSuiteIT` (rank-rrf `license/`) runs on the default basic license.
        "xpack.license.self_generated.type":if matches!(module, "fleet" | "rank-rrf-license") { "basic" } else { "trial" },
        "stack.templates.enabled":stack.to_string(),
        "xpack.apm_data.enabled":apm.to_string(),
        "xpack.otel_data.enabled":otel.to_string(),
        (elasticsearch_engine::registry::QUERYLOG_REGISTRY_ENABLED):stack.to_string(),
        (elasticsearch_engine::registry::PROMETHEUS_REGISTRY_ENABLED):prometheus.to_string(),
        (elasticsearch_engine::datastream::LOGSDB_PLUGIN_SETTING):(prometheus || module == "logsdb").to_string(),
        (elasticsearch_engine::warnings::TEST_MODULES_SETTING):if module == "async-search" { "x-pack-test-deprecated-query" } else { "" },
    });
    let mut settings = settings;
    if module == "stateless" {
        // StatelessPlugin.additionalSettings supplies this node override.
        settings["data_streams.failure_store.refresh_interval"] = serde_json::json!("30s");
    }
    if matches!(module, "gpu-single" | "gpu-multi" | "gpu-mixed" | "gpu-disabled" | "gpu-unsupported") {
        settings["vectors.indexing.use_gpu"] = serde_json::json!(if matches!(module, "gpu-disabled" | "gpu-unsupported") { "false" } else { "true" });
    }
    // ES's x-pack REST test cluster (`XPackRestIT`) runs transport TLS with
    // the `testnode` certificate and its passphrase-protected key (keystore
    // `secure_key_passphrase`), verification mode `certificate`.
    // `ES_FIXTURE_TRANSPORT_TLS=1` runs any module's fixture over TLS (a
    // check that a suite passes with an encrypted transport).
    let mut config_files = serde_json::Map::new();
    // Keystore entries (`elasticsearch-keystore add`): secure settings are
    // refused in `elasticsearch.yml` / `-E`, as ES refuses them.
    let mut secure_settings = serde_json::Map::new();
    if module == "main" || std::env::var("ES_FIXTURE_TRANSPORT_TLS").is_ok_and(|v| v == "1") {
        for (k, v) in [
            ("xpack.security.transport.ssl.enabled", "true"),
            ("xpack.security.transport.ssl.key", "testnode.pem"),
            ("xpack.security.transport.ssl.certificate", "testnode.crt"),
            ("xpack.security.transport.ssl.verification_mode", "certificate"),
        ] {
            settings[k] = serde_json::json!(v);
        }
        secure_settings.insert("xpack.security.transport.ssl.secure_key_passphrase".into(), serde_json::json!("testnode"));
        config_files.insert("testnode.crt".into(), serde_json::json!(include_str!("../../../../crates/es-yaml-runner/resources/testnode.crt")));
        config_files.insert("testnode.pem".into(), serde_json::json!(include_str!("../../../../crates/es-yaml-runner/resources/testnode.pem")));
    }
    if module == "searchable-snapshots" {
        // `SearchableSnapshotsClientYamlTestSuiteIT`'s shared cache.
        settings["xpack.searchable.snapshot.shared_cache.size"] = serde_json::json!("16MB");
        settings["xpack.searchable.snapshot.shared_cache.region_size"] = serde_json::json!("256KB");
    }
    // `StatelessYamlTestSuiteIT` (`StatelessElasticsearchCluster`,
    // `DefaultStatelessLocalConfigProvider`): stateless nodes over an fs
    // object store, an index node and a search node.
    let mut per_node_settings = Vec::new();
    if module == "stateless" {
        for (k, v) in [
            ("stateless.enabled", "true"),
            ("stateless.object_store.type", "fs"),
            ("stateless.object_store.bucket", "stateless"),
            ("stateless.object_store.base_path", "base_path"),
            ("xpack.ml.enabled", "false"),
            ("xpack.watcher.enabled", "false"),
        ] {
            settings[k] = serde_json::json!(v);
        }
        per_node_settings.push(serde_json::json!({"node.roles": "master,remote_cluster_client,ingest,index"}));
        per_node_settings.push(serde_json::json!({"node.roles": "remote_cluster_client,search"}));
    }
    if module == "gpu-mixed" {
        per_node_settings = vec![serde_json::json!({}), serde_json::json!({"vectors.indexing.use_gpu":"false"})];
    }
    if module == "watcher-security" {
        // `WatcherYamlRestWithSecurityIT`: settings to test settings filtering on
        // (the keystore `secure_password` is filtered either way).
        for (k, v) in [("xpack.notification.email.account._email.smtp.host", "host.domain"), ("xpack.notification.email.account._email.smtp.port", "587"), ("xpack.notification.email.account._email.smtp.user", "_user")] {
            settings[k] = serde_json::json!(v);
        }
    }
    serde_json::json!({"module":module,"file_users":cfg.file_users,"file_user_roles":cfg.file_user_roles,
        "file_roles":cfg.file_roles,"file_service_tokens":cfg.file_service_tokens,
        // Node counts follow each module's ES test cluster where its tests
        // assert on topology: `inference/qa/multi-node` runs 3 nodes;
        // `AutoscalingRestIT` declares none (ES's default single node) and
        // asserts the policy's total capacity equals one node's.
        "settings":settings,"security":cfg.enabled,"nodes":match module { "inference-multi-node" => 3, "autoscaling" | "watcher-single" | "gpu-single" | "gpu-disabled" | "gpu-unsupported" => 1, _ => 2 },
        "inference_test_services":matches!(module,"inference" | "inference-2" | "inference-multi-node" | "rank-rrf" | "rank-rrf-license" | "ent-search" | "main"),
        "per_node_settings":per_node_settings,
        "config_files":config_files,
        "secure_settings":secure_settings,
        "network_tls":false})
}

/// Run one test section (setup, test, teardown) against the cluster at `base`.
/// Cleaning up between tests is the caller's job (`wipe`).
pub async fn run_test(specs: &Specs, file: &TestFile, steps: &[Value], base: &str, timeout: Duration) -> Outcome {
    let mut ctx = Ctx::new(specs, base.to_string());
    if MODULE_SECURITY_FIXTURES.load(std::sync::atomic::Ordering::Relaxed) {
        ctx.auth = match file.dir.as_str() {
            "ent-search" => Some(entsearch_security_config().1),
            "inference" => Some(inference_security_config().1),
            "ccr" => Some(ccr_security_config().1),
            "fleet" => Some(fleet_security_config().1),
            "searchable-snapshots" => Some(admin_superuser_config().1),
            _ if is_watcher_security(file) => Some(basic("watcher_manager", XPACK_PASSWORD)),
            _ => default_auth(),
        };
    }
    let mut required_missing = Vec::new();
    for s in file.setup.iter().chain(file.teardown.iter()).chain(steps.iter()) {
        match cluster_prereq_skip(&mut ctx, s, base, &mut required_missing).await {
            Ok(Some(reason)) => return Outcome::Skip(reason),
            Ok(None) => {},
            Err(msg) => return Outcome::Fail { msg, category: "runner: reference prerequisite error".into() },
        }
    }
    required_missing.extend(ctx.required_coverage.iter().cloned());
    required_missing.sort();
    required_missing.dedup();
    let fut = async {
        if let Err(f) = ctx.run_steps(&file.setup).await {
            return Outcome::Fail { msg: format!("setup: {}", f.msg), category: format!("setup: {}", f.category) };
        }
        ctx.last = None;
        ctx.stash.insert("_project_id_prefix_".into(), Value::String(String::new()));
        let r = ctx.run_steps(steps).await;
        let _ = ctx.run_steps(&file.teardown).await;
        match r {
            Ok(()) if required_missing.is_empty() => Outcome::Pass,
            Ok(()) => Outcome::Fail {
                msg: format!("Assertions passed, but product capability discovery is missing reference-required gates: {}", required_missing.join("; ")),
                category: "product: missing required capability discovery".into(),
            },
            Err(f) => Outcome::Fail { msg: f.msg, category: f.category },
        }
    };
    match tokio::time::timeout(timeout, fut).await {
        Ok(o) => o,
        Err(_) => Outcome::Fail { msg: format!("test timed out after {}s", timeout.as_secs()), category: "runner: test timeout".into() },
    }
}

#[cfg(test)]
mod reference_gate_tests {
    use super::*;
    async fn endpoint(body: &str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            stream.read(&mut request).await.unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{address}")
    }

    #[test]
    fn module_fixture_exports_realms_and_module_security_modes() {
        let fleet = external_fixture("fleet");
        assert_eq!(fleet["settings"]["xpack.license.self_generated.type"], "basic");
        assert!(fleet["file_users"].get("elastic_admin").is_some());
        assert!(fleet["file_users"].get("fleet_unprivileged_secrets").is_some());
        assert_eq!(fleet["file_user_roles"]["fleet_unprivileged_secrets"], serde_json::json!(["unprivileged_secrets"]));
        for (module, user) in [("ccr", "ccr-user"), ("inference", "inference_user"), ("ent-search", "entsearch-admin")] {
            let fixture = external_fixture(module);
            assert_eq!(fixture["security"], true);
            assert!(fixture["file_users"].get(user).is_some());
            assert!(fixture["file_users"].get("x_pack_rest_user").is_some());
        }
        assert_eq!(external_fixture("stateless")["settings"]["data_streams.failure_store.refresh_interval"], "30s");
        assert!(external_fixture("oss")["settings"].get("data_streams.failure_store.refresh_interval").is_none());
        let stateless = external_fixture("stateless");
        assert_eq!(stateless["settings"]["stateless.enabled"], "true");
        assert_eq!(stateless["per_node_settings"][0]["node.roles"], "master,remote_cluster_client,ingest,index");
        assert_eq!(stateless["per_node_settings"][1]["node.roles"], "remote_cluster_client,search");
        assert_eq!(external_fixture("gpu-single")["nodes"], 1);
        assert_eq!(external_fixture("gpu-multi")["nodes"], 2);
        assert_eq!(external_fixture("gpu-single")["settings"]["vectors.indexing.use_gpu"], "true");
        assert_eq!(external_fixture("gpu-disabled")["settings"]["vectors.indexing.use_gpu"], "false");
        assert_eq!(external_fixture("gpu-mixed")["per_node_settings"][1]["vectors.indexing.use_gpu"], "false");
        // Secure settings travel as keystore entries, never as settings.
        let main = external_fixture("main");
        assert_eq!(main["secure_settings"]["xpack.security.transport.ssl.secure_key_passphrase"], "testnode");
        assert!(main["settings"].as_object().unwrap().keys().all(|k| !elasticsearch_engine::settings_filter::is_secure_only(k)));
        let core = external_fixture("oss");
        assert_eq!(core["settings"]["stack.templates.enabled"], "true");
        assert_eq!(core["settings"]["xpack.apm_data.enabled"], "true");
        let plain = external_fixture("inference-2");
        assert_eq!(plain["security"], false);
        assert_eq!(plain["inference_test_services"], true);
        assert_eq!(plain["nodes"], 2);
        let multi = external_fixture("inference-multi-node");
        assert_eq!(multi["inference_test_services"], true);
        assert_eq!(multi["nodes"], 3);
        assert_eq!(external_fixture("autoscaling")["nodes"], 1);
        assert_eq!(external_fixture("watcher-single")["nodes"], 1);
        assert!(external_fixture("main")["file_users"].get("inference_user").is_some());
    }

    #[tokio::test]
    async fn default_gate_policy_executes_required_cases_but_preserves_reference_exclusions() {
        let temp = std::env::temp_dir().join(format!("es-yaml-gates-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&temp).unwrap();
        let specs = Specs::load(&temp).unwrap();
        std::fs::remove_dir(&temp).unwrap();
        for (entry, expected_skip) in [
            (serde_json::json!({"method":"POST","path":"/{index}/_doc","capabilities":["failure_store_status"]}), false),
            (serde_json::json!({"method":"POST","path":"/_query","capabilities":["optional_fields_v2"]}), true),
        ] {
            let base = endpoint(r#"{"supported":false}"#).await;
            let mut ctx = Ctx::new(&specs, base.clone());
            let step = serde_json::json!({"requires":{"capabilities":[entry]}});
            let mut gaps = Vec::new();
            let skipped = cluster_prereq_skip(&mut ctx, &step, &base, &mut gaps).await.unwrap();
            assert_eq!(skipped.is_some(), expected_skip);
            assert_eq!(gaps.is_empty(), expected_skip);
        }
    }

    #[tokio::test]
    async fn capability_cache_is_scoped_to_server_and_reference_errors_are_explicit() {
        let entry = serde_json::json!({"method":"POST","path":"/_search","capabilities":["reference_gate_regression"]});
        let supported = endpoint(r#"{"supported":true}"#).await;
        let unsupported = endpoint(r#"{"supported":false}"#).await;
        assert!(capability_supported(&supported, &entry).await.1);
        assert!(!capability_supported(&unsupported, &entry).await.1);
        let reference = endpoint(r#"{"supported":true}"#).await;
        assert!(reference_capability_supported(&reference, &entry).await.unwrap());
        let invalid = endpoint(r#"{"supported":null}"#).await;
        assert!(reference_capability_supported(&invalid, &entry).await.is_err());
    }
}
