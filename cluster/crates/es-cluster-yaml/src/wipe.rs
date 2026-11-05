//! Between-test cleanup modelled on ES's `ESRestTestCase.wipeCluster()`:
//! snapshots + repositories, data streams, indices (incl. hidden/closed),
//! composable / component / legacy templates, ingest pipelines, persistent
//! and transient cluster settings, voting config exclusions, node shutdown
//! records; then wait until no shards initialize and no tasks are pending.

use serde_json::{json, Map, Value};
use std::time::Duration;

/// ES9.5.5 ESRestTestCase.wipeAllIndices excludes these history backing
/// indices: background services can recreate them after data-stream deletion.
/// This is cleanup policy only; cases exercising them still execute normally.
fn preserved_during_index_cleanup(name: &str) -> bool {
    name.starts_with(".security") || [
        ".ds-ilm-history-", ".ds-.slm-history-", ".ds-.watcher-history-", ".ds-.triggered_watches-",
    ].iter().any(|prefix| name.starts_with(prefix))
}

pub struct Wiper {
    client: reqwest::Client,
    base: String,
}

impl Wiper {
    pub fn new(base: &str) -> Wiper {
        let client = reqwest::Client::builder().timeout(Duration::from_secs(90)).build().expect("http client");
        Wiper { client, base: base.trim_end_matches('/').to_string() }
    }

    async fn call(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<(u16, Value), String> {
        let mut rb = self.client.request(method, format!("{}{}", self.base, path));
        if let Some(auth) = crate::exec::external_auth_header() {
            rb = rb.header("Authorization", auth);
        }
        if let Some(b) = body {
            rb = rb.header("content-type", "application/json").body(b.to_string());
        }
        let r = rb.send().await.map_err(|e| format!("{path}: {e}"))?;
        let status = r.status().as_u16();
        let text = r.text().await.unwrap_or_default();
        Ok((status, serde_json::from_str(&text).unwrap_or(Value::String(text))))
    }

    async fn get(&self, path: &str) -> Value {
        self.call(reqwest::Method::GET, path, None).await.map(|(_, v)| v).unwrap_or(Value::Null)
    }

    async fn delete(&self, path: &str) -> Result<u16, String> {
        let (status, body) = self.call(reqwest::Method::DELETE, path, None).await?;
        if status >= 300 && status != 404 { return Err(format!("{path}: HTTP {status}, {body}")); }
        Ok(status)
    }

    /// Wait until the cluster answers and has a master (used at start-up).
    pub async fn wait_ready(&self, nodes: usize) -> Result<(), String> {
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        loop {
            if let Ok((200, v)) = self.call(reqwest::Method::GET, &format!("/_cluster/health?wait_for_nodes={nodes}&timeout=5s"), None).await {
                if v["timed_out"] == json!(false) {
                    return Ok(());
                }
            }
            if std::time::Instant::now() > deadline {
                return Err(format!("cluster at {} not ready with {nodes} nodes", self.base));
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Returns a list of problems (empty = clean).
    pub async fn wipe(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let mut note = |r: Result<u16, String>, what: &str| match r {
            Ok(s) if s < 300 || s == 404 => {}
            Ok(s) => problems.push(format!("{what}: HTTP {s}")),
            Err(e) => problems.push(format!("{what}: {e}")),
        };

        // ESRestTestCase.wipeRollupJobs: stop persistent jobs before their
        // indices disappear, then delete the configurations.
        if let Ok((200, jobs)) = self.call(reqwest::Method::GET, "/_rollup/job/_all", None).await {
            let jobs: Vec<_> = jobs["jobs"].as_array().into_iter().flatten()
                .filter_map(|job| job["config"]["id"].as_str().map(str::to_string)).collect();
            for id in &jobs {
                let mut url = reqwest::Url::parse(&self.base).expect("validated base URL");
                url.path_segments_mut().expect("HTTP URL").clear().push("_rollup").push("job").push(id).push("_stop");
                note(self.call(reqwest::Method::POST, &format!("{}?wait_for_completion=true&timeout=10s", url.path()), None).await.map(|(s, _)| s), "stop rollup job");
            }
            for id in jobs {
                let mut url = reqwest::Url::parse(&self.base).expect("validated base URL");
                url.path_segments_mut().expect("HTTP URL").clear().push("_rollup").push("job").push(&id);
                note(self.delete(url.path()).await, "delete rollup job");
            }
        }

        // Stop policies before deleting their snapshots/repositories.
        if let Ok((200, Value::Object(policies))) = self.call(reqwest::Method::GET, "/_slm/policy", None).await {
            for id in policies.keys() {
                let mut url = reqwest::Url::parse(&self.base).expect("validated base URL");
                url.path_segments_mut().expect("HTTP URL").clear().push("_slm").push("policy").push(id);
                note(self.delete(url.path()).await, "delete SLM policy");
            }
        }

        // ESRestTestCase.wipeSearchableSnapshotsIndices: mounted indices go
        // before their repositories (which refuse to go while in use).
        let md = self.get("/_cluster/state/metadata").await;
        let mounted: Vec<String> = md["metadata"]["indices"].as_object().map(|m| {
            m.iter().filter(|(_, im)| im["settings"]["index"]["store"]["type"] == "snapshot" || im["settings"]["index.store.type"] == "snapshot").map(|(n, _)| n.clone()).collect()
        }).unwrap_or_default();
        for chunk in mounted.chunks(50) {
            note(self.delete(&format!("/{}?expand_wildcards=all&ignore_unavailable=true", chunk.join(","))).await, "delete searchable snapshot indices");
        }

        // Snapshots and repositories.
        if let Ok((200, Value::Object(repos))) = self.call(reqwest::Method::GET, "/_snapshot/_all", None).await {
            for (repo, def) in &repos {
                if def["type"].is_string() {
                    let snaps = self.get(&format!("/_snapshot/{repo}/_all")).await;
                    for s in snaps["snapshots"].as_array().cloned().unwrap_or_default() {
                        if let Some(name) = s["snapshot"].as_str() {
                            note(self.delete(&format!("/_snapshot/{repo}/{name}")).await, "delete snapshot");
                        }
                    }
                }
                note(self.delete(&format!("/_snapshot/{repo}")).await, "delete repository");
            }
        }

        // ESRestTestCase.wipeCluster resets feature states before ordinary
        // indices. Keeping security indices alone otherwise leaks keys/users
        // between tests; the file-realm test administrator survives reset.
        note(self.call(reqwest::Method::POST, "/_features/_reset", None).await.and_then(|(status, body)| {
            if status >= 300 && status != 404 {
                Err(format!("/_features/_reset: HTTP {status}, {body}"))
            } else {
                Ok(status)
            }
        }), "reset feature states");

        note(self.delete("/_data_stream/*?expand_wildcards=all").await, "delete data streams");

        // Indices: by name (wildcard deletes may be refused by
        // `action.destructive_requires_name`).
        let md = self.get("/_cluster/state/metadata").await;
        let names: Vec<String> = md["metadata"]["indices"].as_object().map(|m| m.keys().filter(|n| !preserved_during_index_cleanup(n)).cloned().collect()).unwrap_or_default();
        for chunk in names.chunks(50) {
            note(self.delete(&format!("/{}?expand_wildcards=all&ignore_unavailable=true", chunk.join(","))).await, "delete indices");
        }

        if !names.is_empty() {
            let md = self.get("/_cluster/state/metadata").await;
            if let Some(m) = md["metadata"]["indices"].as_object() {
                let left: Vec<&String> = m.keys().filter(|n| !preserved_during_index_cleanup(n)).collect();
                if !left.is_empty() { note(Ok(599), &format!("indices left after delete: {left:?}")); }
            }
        }

        // ESRestTestCase.wipeAllViews: view metadata survives index and
        // feature-state cleanup and must be removed separately.
        if let Ok((200, views)) = self.call(reqwest::Method::GET, "/_query/view", None).await {
            for view in views["views"].as_array().into_iter().flatten() {
                if let Some(name) = view["name"].as_str() {
                    note(self.delete(&format!("/_query/view/{name}")).await, "delete ES|QL view");
                }
            }
        }

        // Match ESRestTestCase's preserve-XPack-template cleanup. Registry
        // templates can be recreated between wildcard deletes, leaving their
        // components referenced. Delete only test-owned templates, in order.
        for (path, array) in [("/_index_template", Some("index_templates")),
                              ("/_component_template", Some("component_templates")),
                              ("/_template", None)] {
            match self.call(reqwest::Method::GET, path, None).await {
                Ok((200, body)) => {
                    let names: Option<Vec<&str>> = match array {
                        Some(field) => body[field].as_array().and_then(|items|
                            items.iter().map(|item| item["name"].as_str()).collect()),
                        None => body.as_object().map(|items| items.keys().map(String::as_str).collect()),
                    };
                    if let Some(names) = names {
                        for name in names.into_iter().filter(|name| !is_xpack_template(name)) {
                            let mut url = reqwest::Url::parse(&self.base).expect("validated base URL");
                            url.path_segments_mut().expect("HTTP URL").clear().push(path.trim_start_matches('/')).push(name);
                            note(self.delete(url.path()).await, "delete test template");
                        }
                    } else { note(Err(format!("{path}: invalid template listing: {body}")), "list templates"); }
                }
                Ok((status, body)) => note(Err(format!("{path}: HTTP {status}, {body}")), "list templates"),
                Err(error) => note(Err(error), "list templates"),
            }
        }
        note(self.delete("/_ingest/pipeline/*").await, "delete pipelines");
        // Synonym sets (ES keeps them in the `.synonyms` system index, which
        // its test cleanup resets; here they are cluster metadata).
        if let Ok((200, v)) = self.call(reqwest::Method::GET, "/_synonyms?size=1000", None).await {
            for s in v["results"].as_array().cloned().unwrap_or_default() {
                if let Some(n) = s["synonyms_set"].as_str() {
                    note(self.delete(&format!("/_synonyms/{n}")).await, "delete synonyms set");
                }
            }
        }

        // Cluster settings → null.
        let s = self.get("/_cluster/settings?flat_settings=true").await;
        let mut reset = Map::new();
        for scope in ["persistent", "transient"] {
            if let Some(m) = s[scope].as_object() {
                if !m.is_empty() {
                    reset.insert(scope.into(), Value::Object(m.keys().map(|k| (k.clone(), Value::Null)).collect()));
                }
            }
        }
        if !reset.is_empty() {
            note(self.call(reqwest::Method::PUT, "/_cluster/settings", Some(Value::Object(reset))).await.map(|(s, _)| s), "reset cluster settings");
        }

        note(self.delete("/_cluster/voting_config_exclusions?wait_for_removal=false").await, "clear voting exclusions");
        let sd = self.get("/_nodes/shutdown").await;
        for n in sd["nodes"].as_array().cloned().unwrap_or_default() {
            if let Some(id) = n["node_id"].as_str() {
                note(self.delete(&format!("/_nodes/{id}/shutdown")).await, "delete shutdown record");
            }
        }
        // ESRestTestCase.deleteAllNodeShutdownMetadata's sibling: desired
        // nodes are cluster metadata the tests leave behind.
        note(self.delete("/_internal/desired_nodes").await, "clear desired nodes");

        // ensureNoInitializingShards + waitForClusterStateUpdatesToFinish.
        let h = self.call(reqwest::Method::GET, "/_cluster/health?wait_for_no_initializing_shards=true&wait_for_no_relocating_shards=true&timeout=70s", None).await;
        match h {
            Ok((s, health)) if s != 200 || health["timed_out"] == true => problems.push(format!("health after wipe: HTTP {s}, {health}")),
            Err(error) => problems.push(format!("health after wipe: {error}")),
            _ => {}
        }
        for _ in 0..50 {
            let pt = self.get("/_cluster/pending_tasks").await;
            if pt["tasks"].as_array().map(|a| a.is_empty()).unwrap_or(true) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        problems
    }
}

// ES 9.5.5 ESRestTestCase.isXPackTemplate. In particular @custom is user
// owned and must be removed; a generic "contains @" exclusion is incorrect.
fn is_xpack_template(name: &str) -> bool {
    [".monitoring-", ".watch", ".triggered_watches", ".data-frame-", ".ml-",
     ".transform-", ".deprecation-", ".fleet-", "behavioral_analytics-",
     "profiling-", "elastic-connectors", "apm@", "apm-", "traces-apm",
     "metrics-apm", "logs-apm", ".slm-history", "ilm-history",
     "logs-elasticsearch.querylog"].iter().any(|prefix| name.starts_with(prefix))
    || (name.contains('@') && !name.ends_with("@custom"))
    || matches!(name, ".watches" | "security_audit_log" | ".async-search" |
        ".profiling-ilm-lock" | "saml-service-provider" | "logs" | "logs-settings" |
        "logs-mappings" | "metrics" | "metrics-settings" | "metrics-tsdb-settings" |
        "metrics-mappings" | "synthetics" | "agentless" | "synthetics-settings" |
        "synthetics-mappings" | ".snapshot-blob-cache" | "logstash-index-template" |
        "security-index-template" | "data-streams-mappings" | "search-acl-filter" |
        ".kibana-reporting")
}

#[cfg(test)]
mod template_tests {
    use super::*;
    #[test]
    fn cleanup_preserves_only_reference_history_prefixes_and_existing_security_policy() {
        for name in [".ds-ilm-history-7-2026.10.24-000001", ".ds-.slm-history-7-2026.10.24-000001",
            ".ds-.watcher-history-17-2026.10.24-000001", ".ds-.triggered_watches-1-000001", ".security-7"] {
            assert!(preserved_during_index_cleanup(name), "{name}");
        }
        for name in [".ds-logs-test-000001", "ilm-history-test", ".ds-ilm-history", ".ds-.slm-history",
            ".ds-custom-history-000001", "ordinary"] {
            assert!(!preserved_during_index_cleanup(name), "{name}");
        }
    }

    #[test]
    fn registry_templates_survive_but_custom_resources_do_not() {
        for name in ["elastic-connectors-settings", ".deprecation-indexing-template-9", "logs@settings", "logs"] {
            assert!(is_xpack_template(name), "{name}");
        }
        for name in ["test-template", "logs@custom", "test@custom", "custom-component"] {
            assert!(!is_xpack_template(name), "{name}");
        }
    }
}
