//! The Rally `k8s_metrics` track's data stream (`metrics-kubernetes.pod-rally`,
//! synthetic `_source`, `index.codec: best_compression`, composed from the
//! `metrics-kubernetes.pod@package` component template) with a generator of
//! its documents: ~10 KB pod-metrics docs whose `host.ip` (213 values, 109
//! distinct) and `host.mac` (107 values) arrays and node labels repeat in
//! every doc, a few dimensions (`kubernetes.pod.uid`, `service.address`,
//! `orchestrator.cluster.url`), gauges / counters (some sent as strings, as
//! the corpus does) and `@timestamp`. On c32b the track stored 42.83 GB vs ES
//! 10.33 GB (8.64M docs: 4,957 vs 1,196 B/doc) and indexed 6,156 vs 11,239
//! docs/s.
//!
//! Tests: a TSDB data stream created from a composable template gets the
//! `time_series` mode, synthetic source and the `_tsid` / `@timestamp` index
//! sort; synthetic `_source` round-trips (memtable, sealed zstd segments,
//! merged); best-compression stores fields in zstd chunks.
//!
//! Microbenches (ignored):
//! * `k8s_store_microbench`: bytes per doc by section and by field,
//!   force-merged to one segment.
//!   `cargo test --release -p elasticsearch-engine --lib k8s_store_microbench -- --ignored --nocapture`
//!   (`K8S_DOCS`, default 200,000; `K8S_CODEC=default` for the default codec;
//!   `K8S_TSDB=1` for the same stream in `time_series` mode).
//! * `k8s_ingest_microbench`: docs/s of `K8S_CLIENTS` (default 8) clients
//!   each writing `K8S_BULK` (default 1,000) docs per bulk and refreshing
//!   after every bulk (the track's `bulk-index-metrics-with-refresh`).
//!   `cargo test --release -p elasticsearch-engine --lib k8s_ingest_microbench -- --ignored --nocapture`
//!   (`K8S_DOCS` default 400,000, `K8S_CODEC`, `K8S_TSDB` as above).
//! Both take `K8S_MODE` (`standard`, `logsdb`, `time_series`) and
//! `K8S_CORPUS` (a file of real corpus docs, one per line).

use crate::index::{IndexService, WriteReq};
use crate::node::Node;
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub(crate) const DS: &str = "metrics-kubernetes.pod-rally";

fn kw() -> Value {
    json!({"type": "keyword", "ignore_above": 1024})
}

fn gauge(ty: &str) -> Value {
    if ty == "scaled_float" {
        json!({"type": "scaled_float", "scaling_factor": 1000, "time_series_metric": "gauge", "meta": {"unit": "percent"}})
    } else {
        json!({"type": ty, "time_series_metric": "gauge", "meta": {"unit": "byte"}})
    }
}

fn counter() -> Value {
    json!({"type": "long", "time_series_metric": "counter", "meta": {}})
}

/// The track's `metrics-kubernetes.pod@package` mapping (the fields the
/// corpus documents use, with the same types and parameters, and its
/// dynamic templates).
pub(crate) fn mappings() -> Value {
    let pct = || gauge("scaled_float");
    json!({
        "dynamic_templates": [
            {"container.labels": {"path_match": "container.labels.*", "mapping": {"type": "keyword"}, "match_mapping_type": "string"}},
            {"kubernetes.node.labels.*": {"path_match": "kubernetes.node.labels.*", "mapping": {"type": "keyword"}, "match_mapping_type": "*"}},
            {"kubernetes.labels.*": {"path_match": "kubernetes.labels.*", "mapping": {"type": "keyword"}, "match_mapping_type": "*"}},
            {"kubernetes.namespace_labels.*": {"path_match": "kubernetes.namespace_labels.*", "mapping": {"type": "keyword"}, "match_mapping_type": "*"}}
        ],
        "properties": {
            "@timestamp": {"type": "date", "ignore_malformed": false},
            "cloud": {"properties": {
                "availability_zone": kw(), "provider": kw(), "region": kw(),
                "image": {"properties": {"id": kw()}},
                "instance": {"properties": {"name": kw(), "id": kw()}},
                "machine": {"properties": {"type": kw()}},
                "project": {"properties": {"id": kw()}},
                "account": {"properties": {"id": kw()}}
            }},
            "container": {"properties": {
                "image": {"properties": {"name": kw()}}, "name": kw(), "id": kw(),
                "network": {"properties": {"ingress": {"properties": {"bytes": counter()}}, "egress": {"properties": {"bytes": counter()}}}}
            }},
            "kubernetes": {"properties": {
                "container": {"properties": {"image": kw(), "name": kw()}},
                "node": {"properties": {"uid": kw(), "hostname": kw(), "name": kw()}},
                "pod": {"properties": {
                    "uid": {"type": "keyword", "time_series_dimension": true},
                    "start_time": {"type": "date"},
                    "ip": {"type": "ip"},
                    "name": kw(),
                    "memory": {"properties": {
                        "rss": {"properties": {"bytes": gauge("long")}},
                        "major_page_faults": counter(),
                        "usage": {"properties": {"node": {"properties": {"pct": pct()}}, "bytes": gauge("long"), "limit": {"properties": {"pct": pct()}}}},
                        "available": {"properties": {"bytes": gauge("long")}},
                        "page_faults": counter(),
                        "working_set": {"properties": {"bytes": gauge("long"), "limit": {"properties": {"pct": pct()}}}}
                    }},
                    "cpu": {"properties": {"usage": {"properties": {"node": {"properties": {"pct": pct()}}, "nanocores": gauge("long"), "limit": {"properties": {"pct": pct()}}}}}},
                    "network": {"properties": {
                        "tx": {"properties": {"bytes": counter(), "errors": counter()}},
                        "rx": {"properties": {"bytes": counter(), "errors": counter()}}
                    }}
                }},
                "statefulset": {"properties": {"name": kw()}},
                "namespace": kw(), "namespace_uid": kw(),
                "replicaset": {"properties": {"name": kw()}},
                "daemonset": {"properties": {"name": kw()}},
                "job": {"properties": {"name": kw()}},
                "cronjob": {"properties": {"name": kw()}},
                "deployment": {"properties": {"name": kw()}}
            }},
            "orchestrator": {"properties": {"cluster": {"properties": {"name": kw(), "url": {"type": "keyword", "time_series_dimension": true}}}}},
            "ecs": {"properties": {"version": kw()}},
            "data_stream": {"properties": {"namespace": {"type": "constant_keyword"}, "type": {"type": "constant_keyword"}, "dataset": {"type": "constant_keyword"}}},
            "service": {"properties": {"address": {"type": "keyword", "time_series_dimension": true}, "type": kw()}},
            "host": {"properties": {
                "hostname": kw(),
                "os": {"properties": {
                    "build": kw(), "kernel": kw(), "codename": kw(),
                    "name": {"type": "keyword", "ignore_above": 1024, "fields": {"text": {"type": "text"}}},
                    "family": kw(), "version": kw(), "platform": kw()
                }},
                "domain": kw(), "ip": {"type": "ip"}, "containerized": {"type": "boolean"},
                "name": kw(), "id": kw(), "type": kw(), "mac": kw(), "architecture": kw()
            }}
        }
    })
}

/// Install the track's component and composable templates (`codec`: the
/// `index.codec`; `tsdb`: add `index.mode: time_series`, as a Fleet TSDB
/// package does, routing on the mapped dimensions).
pub(crate) fn install_templates(node: &Node, codec: &str, tsdb: bool) {
    install_templates_mode(node, codec, if tsdb { "time_series" } else { "standard" });
}

/// [`install_templates`] with `index.mode` `mode` (`standard`,
/// `time_series` or `logsdb`).
pub(crate) fn install_templates_mode(node: &Node, codec: &str, mode: &str) {
    let mut index = json!({
        "codec": codec,
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "mapping": {"source.mode": "synthetic", "total_fields": {"limit": "10000"}}
    });
    if mode != "standard" {
        index["mode"] = json!(mode);
    }
    let mut t = node.templates.write();
    t.component.insert("metrics-kubernetes.pod@package".into(), json!({"template": {"settings": {"index": index}, "mappings": mappings()}}));
    t.index.insert(
        "metrics-kubernetes.pod-template".into(),
        json!({
            "index_patterns": ["metrics-kubernetes.pod*"],
            "composed_of": ["metrics-kubernetes.pod@package"],
            "priority": 200,
            "data_stream": {"hidden": false, "allow_custom_routing": false},
            "template": {"settings": {"index": {}}, "mappings": {"_meta": {"package": {"name": "kubernetes"}, "managed_by": "fleet", "managed": true}}}
        }),
    );
}

/// Create the data stream; its write (backing) index.
pub(crate) fn create_stream(node: &Arc<Node>) -> Arc<IndexService> {
    let ds = node.create_data_stream(DS).unwrap();
    node.get_index(ds.write_index()).unwrap()
}

pub(crate) struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// What every document of the corpus repeats: the node's `host.ip` (213
/// values: two singles, then 105 pairs of the bridge address and a pod's
/// link-local address), `host.mac` (107 addresses) and node labels.
pub(crate) struct Shared {
    ips: Vec<String>,
    macs: Vec<String>,
    labels: Map<String, Value>,
}

impl Shared {
    pub fn new() -> Self {
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        let mut ips = vec!["10.10.0.186".to_string(), "fe80::4001:aff:fe0a:ba".to_string(), "169.254.123.1".to_string()];
        for _ in 0..105 {
            ips.push("10.232.83.1".to_string());
            ips.push(format!("fe80::{:x}:{:x}ff:fe{:02x}:{:x}", r.below(0xffff), r.below(0xff), r.below(0xff), r.below(0xffff)));
        }
        let mut macs: Vec<String> = (0..107).map(|_| (0..6).map(|_| format!("{:02X}", r.below(256))).collect::<Vec<_>>().join("-")).collect();
        macs.sort();
        let mut labels = Map::new();
        for (k, v) in [
            ("cloud_google_com/machine-family", "e2"), ("cloud_google_com/gke-nodepool", "kubernetes-scale-nl"),
            ("kubernetes_io/hostname", "kubernetes-scale-123456-6f4f4c15-93fe-4fe7-a1bf-3bbe5d165c22"), ("cloud_google_com/gke-os-distribution", "cos"),
            ("topology_kubernetes_io/zone", "europe-west1-d"), ("topology_gke_io/zone", "europe-west1-d"), ("topology_kubernetes_io/region", "europe-west1"),
            ("kubernetes_io/arch", "amd64"), ("cloud_google_com/gke-cpu-scaling-level", "4"), ("env", "kubernetes-scale"),
            ("failure-domain_beta_kubernetes_io/region", "europe-west1"), ("cloud_google_com/gke-max-pods-per-node", "110"),
            ("cloud_google_com/gke-container-runtime", "containerd"), ("beta_kubernetes_io/instance-type", "e2-standard-4"),
            ("failure-domain_beta_kubernetes_io/zone", "europe-west1-d"), ("node_kubernetes_io/instance-type", "e2-standard-4"),
            ("beta_kubernetes_io/os", "linux"), ("cloud_google_com/gke-boot-disk", "pd-balanced"), ("kubernetes_io/os", "linux"),
            ("cloud_google_com/private-node", "false"), ("cloud_google_com/gke-logging-variant", "DEFAULT"), ("beta_kubernetes_io/arch", "amd64"),
        ] {
            labels.insert(k.into(), json!(v));
        }
        Shared { ips, macs, labels }
    }
}

fn uuid(r: &mut Rng) -> String {
    let a = r.next();
    let b = r.next();
    format!("{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}", a >> 32, (a >> 16) & 0xffff, a & 0xfff, (b >> 48) & 0xffff, b & 0xffff_ffff_ffff)
}

/// `epoch ms` as the corpus prints timestamps (microseconds, `+03:00`).
fn ts(ms: i64) -> String {
    let offset = chrono::FixedOffset::east_opt(3 * 3600).unwrap();
    chrono::DateTime::from_timestamp_millis(ms).unwrap().with_timezone(&offset).format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string()
}

/// Document `i`: pod `i % pods` at tick `i / pods` (one sample per pod per
/// tick, all pods of a tick sharing its `@timestamp`, as the corpus does).
pub(crate) fn k8s_doc(i: u64, pods: u64, base_ms: i64, tick_ms: i64, shared: &Shared, r: &mut Rng) -> Value {
    let pod = i % pods;
    let tick = (i / pods) as i64;
    let at = ts(base_ms + tick * tick_ms + 590);
    // Stable per-pod identities.
    let mut pr = Rng(0x9e37_79b9_7f4a_7c15 ^ (pod + 1).wrapping_mul(0x1000_0000_01b3));
    let pod_uid = uuid(&mut pr);
    let node_uid = uuid(&mut pr);
    let n = 1000 + pod * 3 + 7;
    let rx = 1000 + (tick as u64) * (50 + pod % 17) + r.below(100);
    let tx = 500 + (tick as u64) * (20 + pod % 13) + r.below(50);
    json!({
        "@timestamp": at,
        "container": {"network": {"ingress": {"bytes": rx}, "egress": {"bytes": tx}}},
        "kubernetes": {
            "node": {"uid": node_uid, "hostname": "kubernetes-scale-123456.c.elastic-obs-integrations-dev.internal", "name": format!("kubernetes-scale-123456-{n}"), "labels": shared.labels},
            "pod": {
                "uid": pod_uid, "start_time": ts(base_ms + 590),
                "memory": {
                    "rss": {"bytes": (700_000 + r.below(800_000)).to_string()}, "major_page_faults": 0,
                    "usage": {"node": {"pct": "0"}, "bytes": (2_000_000 + r.below(300_000)).to_string(), "limit": {"pct": "0"}},
                    "available": {"bytes": 0}, "page_faults": 1386,
                    "working_set": {"bytes": (1_300_000 + r.below(1_600_000)).to_string(), "limit": {"pct": "0"}}
                },
                "ip": format!("120.{}.{}.{}", pod % 250, (pod / 250) % 250, 2 + pod % 200),
                "name": format!("demo-deployment-{n}"),
                "cpu": {"usage": {"node": {"pct": 0}, "nanocores": 0, "limit": {"pct": 0}}},
                "network": {"tx": {"bytes": tx, "errors": 0}, "rx": {"bytes": rx, "errors": 0}}
            },
            "namespace": format!("demo-{n}"), "namespace_uid": format!("demo-{n}"),
            "replicaset": {"name": format!("demo-deployment-{n}")},
            "namespace_labels": {"kubernetes_io/metadata_name": format!("demo-{n}")},
            "labels": {"app": "demo", "pod-template-hash": n.to_string(), "app-2": "demo-2", "app-1": "demo-1"},
            "deployment": {"name": format!("demo-deployment-{n}")}
        },
        "cloud": {
            "provider": "gcp", "availability_zone": "europe-west1-d",
            "instance": {"name": "kubernetes-scale-123456", "id": "12f376ef-5186-4e8b-a175-70f1140a8f30"},
            "machine": {"type": "e2-standard-4"}, "service": {"name": "GCE"},
            "project": {"id": "elastic-obs-integrations-dev"}, "account": {"id": "elastic-obs-integrations-dev"}
        },
        "orchestrator": {"cluster": {"name": "kubernetes-scale", "url": "https://246.116.233.113"}},
        "service": {"address": "https://kubernetes-scale-123456:10250/stats/summary", "type": "kubernetes"},
        "data_stream": {"namespace": "default", "type": "metrics", "dataset": "kubernetes.pod"},
        "ecs": {"version": "8.2.0"},
        "agent": {"id": "12f376ef-5186-4e8b-a175-70f1140a8f30", "name": "kubernetes-scale-123456", "type": "metricbeat", "version": "8.7.0", "ephemeral_id": "f94220b0-2ca6-4809-8656-eb478a66c541"},
        "elastic_agent": {"id": "12f376ef-5186-4e8b-a175-70f1140a8f30", "version": "8.7.0", "snapshot": true},
        "metricset": {"period": "10000", "name": "pod"},
        "event": {"duration": (2_300_000 + r.below(100_000)).to_string(), "agent_id_status": "verified", "ingested": at, "module": "kubernetes", "dataset": "kubernetes.pod"},
        "host": {
            "hostname": "kubernetes-scale-123456",
            "os": {"kernel": "5.10.161+", "codename": "focal", "name": "Ubuntu", "type": "linux", "family": "debian", "version": "20.04.5 LTS (Focal Fossa)", "platform": "ubuntu"},
            "containerized": false, "ip": shared.ips, "name": "kubernetes-scale-123456",
            "id": "6f4f4c15-93fe-4fe7-a1bf-3bbe5d165c22", "mac": shared.macs, "architecture": "x86_64"
        }
    })
}

/// Timestamps for `n` docs over `pods` pods: the corpus's 10 s ticks from a
/// fixed day, or, for a time series stream (whose backing index accepts the
/// last two hours), ticks squeezed into the 100 minutes before now.
pub(crate) fn clock(n: u64, pods: u64, tsdb: bool) -> (i64, i64) {
    if !tsdb {
        return (1_684_098_000_000, 10_000);
    }
    let span = 100 * 60_000i64;
    let ticks = (n / pods).max(1) as i64;
    (crate::date::now_millis() - span, (span / ticks).clamp(1, 10_000))
}

/// Start the node's license service as an HTTP / cluster node does: the
/// license is hardwired to trial, so the template's synthetic source stays
/// synthetic (the Rally baseline ran ES on a trial).
fn license_env(node: &Node) -> &'static str {
    crate::license::start(node);
    "trial (hardwired; synthetic _source)"
}

/// Write segments as a cluster at this build's format does (the Rally
/// node's floor), not the standalone default (v11): process-wide, so only
/// the ignored microbenches call it.
fn cluster_format() {
    rustic_core::codec::apply_cluster_segment_format(rustic_core::codec::VERSION);
}

/// `K8S_CORPUS`: a file of the track's documents (one JSON per line, e.g.
/// the head of `doc-ds-metrics-kubernetes.pod.json`) to index instead of
/// generated ones; its first `n` lines.
fn corpus(n: u64) -> Option<Vec<String>> {
    use std::io::BufRead;
    let path = std::env::var("K8S_CORPUS").ok()?;
    let f = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let docs: Vec<String> = f.lines().take(n as usize).map(|l| l.unwrap()).collect();
    assert_eq!(docs.len() as u64, n, "K8S_CORPUS holds fewer than {n} docs");
    Some(docs)
}

fn env(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.replace('_', "").parse().ok()).unwrap_or(d)
}

fn codec_env() -> String {
    std::env::var("K8S_CODEC").unwrap_or_else(|_| "best_compression".into())
}

fn tsdb_env() -> bool {
    std::env::var("K8S_TSDB").is_ok_and(|v| v == "1" || v == "true")
}

/// `K8S_MODE` (`standard`, `time_series`, `logsdb`); `K8S_TSDB=1` is
/// `time_series`.
fn mode_env() -> String {
    std::env::var("K8S_MODE").unwrap_or_else(|_| if tsdb_env() { "time_series".into() } else { "standard".into() })
}

fn search(node: &Arc<Node>, body: Value) -> Value {
    let request = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
    let targets = node.resolve(DS, Default::default()).unwrap();
    let options = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    serde_json::from_str(&crate::search::execute(&targets, &request, &options).unwrap().json).unwrap()
}

fn write_all(svc: &IndexService, docs: Vec<String>) -> Vec<String> {
    let reqs = docs.into_iter().map(|source| WriteReq::Index { id: None, source, routing: None, cond: Default::default() }).collect();
    svc.write(reqs, false).into_iter().map(|r| r.unwrap().id).collect()
}

#[test]
fn tsdb_stream_from_composable_template_is_time_series_synthetic_with_tsid_sort() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    install_templates(&node, "best_compression", true);
    let svc = create_stream(&node);
    // The template's mode, start / end bounds around now, synthetic source.
    assert_eq!(svc.setting("index.mode").as_deref(), Some("time_series"));
    assert!(svc.setting(crate::tsdb::START_TIME).is_some() && svc.setting(crate::tsdb::END_TIME).is_some());
    assert_eq!(svc.setting("index.codec").as_deref(), Some("best_compression"));
    let meta = svc.meta();
    assert_eq!(
        crate::synthetic::source_mode(crate::settings::get_str(&meta.settings, "index.mapping.source.mode"), &meta.mappings, crate::settings::get_str(&meta.settings, "index.mode")),
        crate::synthetic::SourceMode::Synthetic
    );
    // `IndexSortConfigDefaults` for time series, reported and applied.
    let defaults = crate::settings::index_defaults(&meta.settings);
    assert_eq!(defaults.get("index.sort.field"), Some(&json!(["_tsid", "@timestamp"])));
    assert_eq!(defaults.get("index.sort.order"), Some(&json!(["asc", "desc"])));
    let lookup = svc.lookup();
    let tsid = lookup.get("_tsid").and_then(|f| f.core_id).expect("hidden _tsid keyword");
    let timestamp = lookup.get("@timestamp").and_then(|f| f.core_id).unwrap();
    drop(meta);
    let shared = Shared::new();
    let (pods, n) = (40u64, 2000u64);
    let (base, tick) = clock(n, pods, true);
    let mut r = Rng(7);
    // Two bulks with a refresh each (two sealed sorted segments), then one
    // merged segment.
    for half in [0..n / 2, n / 2..n] {
        let docs = half.map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r).to_string()).collect();
        write_all(&svc, docs);
        svc.refresh().unwrap();
    }
    // The mode's default sort resolves to `_tsid` asc, `@timestamp` desc
    // (applied to the core with `RUSTIC_MODE_DEFAULT_SORT`).
    let sort = crate::indexsort::core_sort_with(&svc.meta().settings, &svc.lookup(), true).expect("time series default sort");
    let fields: Vec<(u16, bool, bool)> = sort.fields.iter().map(|f| (f.field, f.reverse, f.max_mode)).collect();
    assert_eq!(fields, vec![(tsid, false, false), (timestamp, true, true)]);
    assert!(crate::indexsort::core_sort_with(&svc.meta().settings, &svc.lookup(), false).is_none());
    let sorted = crate::indexsort::mode_default_sort_enabled();
    for sh in svc.local_shards() {
        let core = sh.shard.core.read();
        assert_eq!(core.index_sort(), sorted.then_some(&sort));
        assert_eq!(core.stored_compression(), rustic_core::codec::store::StoredCompression::Best);
    }
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    // `_doc` order is `_tsid` ascending, then `@timestamp` descending.
    let out = search(&node, json!({"size": n, "sort": ["_doc"], "fields": ["_tsid", "@timestamp"]}));
    let hits = out["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len() as u64, n);
    let keys: Vec<(String, String)> = hits
        .iter()
        .map(|h| {
            let tsid = h["fields"]["_tsid"][0].as_str().or_else(|| h["_tsid"].as_str()).unwrap_or_default().to_string();
            (crate::tsdb::tsid_stored(&tsid), h["fields"]["@timestamp"][0].as_str().unwrap_or_default().to_string())
        })
        .collect();
    assert!(keys.iter().all(|(t, _)| !t.is_empty()), "every hit has a _tsid");
    assert_eq!(keys.iter().map(|k| &k.0).collect::<std::collections::BTreeSet<_>>().len() as u64, pods, "one series per pod");
    if sorted {
        for w in keys.windows(2) {
            assert!(w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 >= w[1].1), "{:?} before {:?}", w[0], w[1]);
        }
    }
    // Stored fields are zstd chunks once the segment format allows them (a
    // cluster's floor; this standalone test process writes the default v11).
    let zstd = rustic_core::codec::write_version() >= rustic_core::codec::ZSTD_STORED_SINCE;
    for sh in svc.local_shards() {
        let core = sh.shard.core.read();
        for seg in core.segments() {
            assert_eq!(seg.store().unwrap().any_zstd(), zstd, "best_compression stores zstd chunks from v26");
        }
    }
}

/// What ES returns for one document of the stream: the canonical form of
/// synthetic `_source` (keys sorted; keyword and ip arrays sorted and
/// deduplicated, ips in address order; numbers sent as strings come back
/// as numbers; scaled floats as doubles; dates in the field's format, UTC).
#[test]
fn k8s_synthetic_source_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    install_templates(&node, "best_compression", false);
    let svc = create_stream(&node);
    let shared = Shared::new();
    let (pods, n) = (25u64, 600u64);
    let (base, tick) = clock(n, pods, false);
    let mut r = Rng(11);
    let sources: Vec<Value> = (0..n).map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r)).collect();
    let ids = write_all(&svc, sources.iter().map(|v| v.to_string()).collect());
    let expected: Vec<Value> = {
        let meta = svc.meta();
        let lk = svc.lookup();
        let keep = crate::synthetic::Keep::of_index(&meta.settings);
        sources.iter().map(|v| crate::synthetic::synthesize(v, &meta.mappings, &lk, keep).unwrap().0).collect()
    };
    let get = |id: &str| -> Value { serde_json::from_str(&svc.get(id).unwrap().unwrap().source.unwrap()).unwrap() };
    // Spot checks of ES's synthetic rendering on the first document.
    let first = &expected[0];
    let ips: Vec<&str> = first["host"]["ip"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut want: Vec<std::net::IpAddr> = shared.ips.iter().map(|s| s.parse().unwrap()).collect();
    let v6 = |a: &std::net::IpAddr| match a {
        std::net::IpAddr::V4(v) => v.to_ipv6_mapped().octets(),
        std::net::IpAddr::V6(v) => v.octets(),
    };
    want.sort_by_key(v6);
    want.dedup();
    assert_eq!(ips.len(), want.len(), "ips deduplicated");
    assert_eq!(ips.iter().map(|s| s.parse::<std::net::IpAddr>().unwrap()).collect::<Vec<_>>(), want, "ips in address order");
    let macs: Vec<&str> = first["host"]["mac"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(macs.windows(2).all(|w| w[0] < w[1]), "keyword arrays sorted and unique");
    assert!(first["kubernetes"]["pod"]["memory"]["rss"]["bytes"].is_u64(), "a long sent as a string comes back a number");
    assert_eq!(first["kubernetes"]["pod"]["memory"]["usage"]["node"]["pct"], json!(0.0));
    assert!(first["@timestamp"].as_str().unwrap().ends_with('Z'), "dates in UTC: {}", first["@timestamp"]);
    // Memtable (realtime get), then sealed and merged segments.
    for (i, id) in ids.iter().enumerate().step_by(37) {
        assert_eq!(get(id), expected[i], "memtable doc {i}");
    }
    svc.refresh().unwrap();
    svc.flush().unwrap();
    for (i, id) in ids.iter().enumerate().step_by(23) {
        assert_eq!(get(id), expected[i], "sealed doc {i}");
    }
    let more: Vec<Value> = (n..2 * n).map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r)).collect();
    write_all(&svc, more.iter().map(|v| v.to_string()).collect());
    svc.refresh().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    for (i, id) in ids.iter().enumerate() {
        let got = get(id);
        assert_eq!(got, expected[i], "merged doc {i}");
        // The stored form is already canonical: synthesizing it again is
        // the identity.
        if i % 50 == 0 {
            let meta = svc.meta();
            let lk = svc.lookup();
            assert_eq!(crate::synthetic::synthesize(&got, &meta.mappings, &lk, crate::synthetic::Keep::of_index(&meta.settings)).unwrap().0, got);
        }
    }
    // Searches return the same `_source`.
    let out = search(&node, json!({"size": 5, "query": {"ids": {"values": [ids[3], ids[4]]}}}));
    for h in out["hits"]["hits"].as_array().unwrap() {
        let i = ids.iter().position(|x| x == h["_id"].as_str().unwrap()).unwrap();
        assert_eq!(h["_source"], expected[i]);
    }
}

/// Under a started basic license the template's synthetic source falls
/// back to stored (ES 9.5.5 `LogsdbLicenseService`). Unreachable: licensing
/// is hardwired to trial.
#[test]
#[ignore = "licence hardwired to trial"]
fn basic_license_falls_back_to_stored_source() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    crate::license::start(&node);
    install_templates(&node, "best_compression", false);
    let svc = create_stream(&node);
    assert_eq!(svc.setting("index.mapping.source.mode").as_deref(), Some("STORED"));
}

/// A fresh node's license is the hardwired trial: with no `start_trial`, a
/// synthetic-source data stream keeps synthetic source, and requests for a
/// basic license (node setting, `start_basic`, `PUT` / `DELETE /_license`)
/// leave every feature enterprise.
#[test]
fn fresh_node_is_trial_and_keeps_synthetic_source() {
    use crate::license::{self, Mode};
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("basic"));
    license::start(&node);
    let now = crate::date::now_millis();
    let l = license::current(&node);
    assert_eq!((l.kind.as_str(), l.status(now)), ("trial", "active"));
    assert!(l.expiry_date_in_millis.unwrap() > now + 50 * 365 * 86_400_000);
    install_templates(&node, "best_compression", false);
    let svc = create_stream(&node);
    let meta = svc.meta();
    assert_eq!(
        crate::synthetic::source_mode(crate::settings::get_str(&meta.settings, "index.mapping.source.mode"), &meta.mappings, crate::settings::get_str(&meta.settings, "index.mode")),
        crate::synthetic::SourceMode::Synthetic
    );
    // Asking for basic changes nothing effective.
    let svc_l = license::service(&node);
    assert!(matches!(svc_l.start_basic(&node, true), license::Outcome::Done));
    assert!(matches!(svc_l.start_trial(&node, "trial", true), license::Outcome::Done));
    assert!(!svc_l.eligible_for_trial() && svc_l.eligible_for_basic(&node));
    for (feature, min) in license::XPACK_FEATURES {
        assert!(license::current(&node).effective_mode(now).allows(*min), "{feature} needs {min:?}");
    }
    assert!(license::check(&node, "ml", Mode::Enterprise).is_ok());
    assert_eq!(license::current(&node).kind, "trial");
    let svc2 = node.create_index("plain-synthetic", &json!({"settings": {"index.mapping.source.mode": "synthetic"}})).unwrap();
    assert_eq!(svc2.setting("index.mapping.source.mode").as_deref(), Some("synthetic"));
}

/// Synthetic source stores only the residual its columns do not rebuild,
/// and every read path returns the canonical form: realtime and
/// non-realtime gets, search hits, after flush and after a merge, for
/// keyword / ip / numeric / date / boolean fields alone and in arrays,
/// objects and object arrays, `ignore_above`, `null_value`, `copy_to`,
/// custom date formats, scaled / half floats, nested objects, text and
/// unmapped fields.
#[test]
fn synthetic_residual_rebuilds_canonical_source() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let mapping = json!({"properties": {
        "k": {"type": "keyword"}, "kn": {"type": "keyword", "ignore_above": 5, "null_value": "NULL"},
        "ip": {"type": "ip"}, "l": {"type": "long"}, "i": {"type": "integer"}, "f": {"type": "float"}, "h": {"type": "half_float"},
        "sf": {"type": "scaled_float", "scaling_factor": 100}, "d": {"type": "double"}, "u": {"type": "unsigned_long"},
        "dt": {"type": "date"}, "dc": {"type": "date", "format": "yyyy/MM/dd"}, "b": {"type": "boolean"},
        "t": {"type": "text", "fields": {"raw": {"type": "keyword"}}}, "src": {"type": "keyword", "copy_to": "dst"}, "dst": {"type": "keyword"},
        "o": {"properties": {"a": {"type": "keyword"}, "n": {"type": "long"}, "deep": {"properties": {"x": {"type": "keyword"}}}}},
        "ns": {"type": "nested", "properties": {"v": {"type": "keyword"}}}
    }});
    node.create_index("syn", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.mapping.source.mode": "synthetic", "codec": "best_compression"}, "mappings": mapping})).unwrap();
    let svc = node.get_index("syn").unwrap();
    let docs = vec![
        json!({"k": "b", "l": 3, "o": {"a": "x"}}),
        json!({"k": ["z", "a", "z"], "kn": ["toolongvalue", "ok", null], "ip": ["10.0.0.2", "::1", "10.0.0.2"], "l": [5, -1, 5], "i": 7,
               "f": 1.1, "h": 3.3333, "sf": 12.345, "d": [2.5, 1.25], "u": 18446744073709551615u64, "dt": "2023-05-15T00:00:00.590091+03:00",
               "dc": "2024/01/02", "b": [true, false], "t": "Hello World", "src": "s", "o": [{"a": "q", "n": 1}, {"a": "p", "n": [3, 2], "deep": {"x": "y"}}],
               "ns": [{"v": "n1"}, {"v": "n2"}], "extra": {"free": ["b", "a", "b"]}, "zz": "unmapped"}),
        json!({"b": "true", "l": "42", "dt": 1700000000000i64, "o.a": "dotted", "o": {"deep.x": "dd"}}),
        json!({"k": "", "f": 0.0, "d": -0.0, "sf": 0.004}),
    ];
    let ids = write_all(&svc, docs.iter().map(|v| v.to_string()).collect());
    let expected: Vec<Value> = {
        let meta = svc.meta();
        let lk = svc.lookup();
        docs.iter().map(|v| crate::synthetic::synthesize(v, &meta.mappings, &lk, crate::synthetic::Keep::of_index(&meta.settings)).unwrap().0).collect()
    };
    let check = |when: &str| {
        for (i, id) in ids.iter().enumerate() {
            let got: Value = serde_json::from_str(&svc.get(id).unwrap().unwrap().source.unwrap()).unwrap();
            assert_eq!(got.to_string(), expected[i].to_string(), "{when}: doc {i}");
        }
        let out = search_index(&node, "syn", json!({"size": 10, "sort": ["_doc"]}));
        for h in out["hits"]["hits"].as_array().unwrap() {
            let Some(i) = ids.iter().position(|x| x == h["_id"].as_str().unwrap()) else { continue };
            assert_eq!(h["_source"].to_string(), expected[i].to_string(), "{when}: hit {i}");
        }
    };
    check("memtable");
    svc.refresh().unwrap();
    check("refreshed");
    svc.flush().unwrap();
    check("flushed");
    // Most docs store a residual, not the whole canonical source.
    let residuals = {
        let sh = svc.local_shards().remove(0);
        let core = sh.shard.core.read();
        let mut n = 0;
        for seg in core.segments() {
            for d in 0..seg.num_docs {
                if let Ok(Some(doc)) = seg.get_doc(d) {
                    if doc.get(crate::mapping::SOURCE_FIELD).and_then(|v| v.as_text()).is_some_and(crate::synthetic::is_residual) {
                        n += 1;
                    }
                }
            }
        }
        n
    };
    assert!(residuals >= 3, "{residuals} residual sources");
    write_all(&svc, vec![json!({"k": "later"}).to_string()]);
    svc.refresh().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    check("merged");
}

/// logsdb documents shaped like http_logs and big5: a `match_only_text` /
/// `text` value comes back from its keyword multi-field's column and an
/// `epoch_second` (or other formatted) date from its column, so the stored
/// residual keeps neither; values the columns cannot render exactly (over
/// `ignore_above`, arrays out of order or repeated, a normalized or stored
/// sub-field, malformed dates) stay in it. Every source is byte-identical
/// to the canonical synthetic source (gets and hits, memtable, refreshed,
/// flushed, merged).
#[test]
fn logsdb_residual_rebuilds_text_and_dates_from_columns() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let mapping = json!({"properties": {
        "@timestamp": {"type": "date", "format": "epoch_second"},
        "ts2": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||epoch_millis"},
        "tsn": {"type": "date_nanos"},
        "bad": {"type": "date", "ignore_malformed": true},
        "request": {"type": "match_only_text", "fields": {"raw": {"type": "keyword", "ignore_above": 32}}},
        "message": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "norm": {"type": "text", "fields": {"lc": {"type": "keyword", "normalizer": "lowercase"}}},
        "stored": {"type": "text", "store": true, "fields": {"k": {"type": "keyword"}}},
        "plain": {"type": "match_only_text"},
        "status": {"type": "integer"}
    }});
    node.create_index("ldb", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.mode": "logsdb"}, "mappings": mapping})).unwrap();
    let svc = node.get_index("ldb").unwrap();
    let mut docs = Vec::new();
    for i in 0..40i64 {
        docs.push(json!({"@timestamp": 894225602 + i, "request": format!("GET /images/s{}.gif HTTP/1.0", i % 7), "message": format!("line {i}"), "status": 200, "ts2": "2024-01-02 03:04:05"}));
    }
    docs.extend([
        json!({"@timestamp": "894225700", "request": "GET /a/very/long/path/that/is/over/the/limit.html HTTP/1.0", "message": ["b", "a", "b"]}),
        json!({"@timestamp": [894225800, 894225799], "request": ["GET /z HTTP/1.0", "GET /a HTTP/1.0"], "message": ["a", "b"]}),
        json!({"@timestamp": 894225801.5, "request": "", "message": "", "ts2": 1700000000000i64, "tsn": "2023-01-01T00:00:00.123456789Z"}),
        json!({"@timestamp": 894225802, "norm": "MiXeD Case", "stored": "kept", "plain": "no keyword here", "bad": "not a date"}),
        json!({"@timestamp": 894225803, "request": ["same", "same"], "bad": "2024-01-01"}),
    ]);
    let ids = write_all(&svc, docs.iter().map(|v| v.to_string()).collect());
    let expected: Vec<Value> = {
        let meta = svc.meta();
        let lk = svc.lookup();
        docs.iter().map(|v| crate::synthetic::synthesize(v, &meta.mappings, &lk, crate::synthetic::Keep::of_index(&meta.settings)).unwrap().0).collect()
    };
    let check = |when: &str| {
        for (i, id) in ids.iter().enumerate() {
            let got: Value = serde_json::from_str(&svc.get(id).unwrap().unwrap().source.unwrap()).unwrap();
            assert_eq!(got.to_string(), expected[i].to_string(), "{when}: doc {i}");
        }
        let out = search_index(&node, "ldb", json!({"size": 100, "sort": ["_doc"]}));
        let hits = out["hits"]["hits"].as_array().unwrap();
        if when != "memtable" { assert!(hits.len() >= docs.len(), "{when}"); }
        for h in hits {
            let Some(i) = ids.iter().position(|x| x == h["_id"].as_str().unwrap()) else { continue };
            assert_eq!(h["_source"].to_string(), expected[i].to_string(), "{when}: hit {i}");
        }
    };
    check("memtable");
    svc.refresh().unwrap();
    check("refreshed");
    svc.flush().unwrap();
    check("flushed");
    // The plain http_logs-shaped docs keep neither the request nor the
    // timestamp in their stored residual.
    let stored: Vec<String> = {
        let sh = svc.local_shards().remove(0);
        let core = sh.shard.core.read();
        let mut out = Vec::new();
        for seg in core.segments() {
            for d in 0..seg.num_docs {
                if let Ok(Some(doc)) = seg.get_doc(d) {
                    // Every stored field: a value may be kept beside the residual.
                    out.push(doc.fields.iter().filter_map(|(_, v)| v.as_text()).collect::<Vec<_>>().join("|"));
                }
            }
        }
        out
    };
    let lean = stored.iter().filter(|s| !s.contains("GET /images") && !s.contains("8942256") && !s.contains("line ")).count();
    assert!(lean >= 40, "{lean} lean residuals: {:?}", &stored[..3.min(stored.len())]);
    // What the columns cannot render stays (the sources above came back
    // byte for byte): a request over `ignore_above`.
    assert!(stored.iter().any(|s| s.contains("over/the/limit")));
    write_all(&svc, vec![json!({"@timestamp": 894225900, "request": "GET /later HTTP/1.0"}).to_string()]);
    svc.refresh().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    check("merged");
}

/// Synthetic source across index modes (TSDB, logsdb, standard), plain
/// documents (one walk to the stored forms) and others (the two passes):
/// the source a realtime GET serves (the translog's canonical form), the
/// one rebuilt from the residual and the columns after a refresh, a flush
/// and a merge, and after reopening the node (translog replay of the
/// unflushed docs) all equal the general walk's canonical source.
#[test]
fn synthetic_forms_match_across_modes_plain_and_not() {
    let tmp = tempfile::tempdir().unwrap();
    let ts = "2024-05-01T10:00:00.000Z";
    let common = json!({
        "@timestamp": {"type": "date"}, "host": {"properties": {"name": {"type": "keyword"}}},
        "dim": {"type": "keyword", "time_series_dimension": true}, "k": {"type": "keyword", "ignore_above": 8},
        "l": {"type": "long"}, "sf": {"type": "scaled_float", "scaling_factor": 1000}, "d": {"type": "double"}, "ip": {"type": "ip"},
        "b": {"type": "boolean"}, "o": {"properties": {"a": {"type": "keyword"}, "n": {"type": "long"}, "e": {"properties": {"x": {"type": "keyword"}}}}},
        "t": {"type": "text"}, "kept": {"type": "keyword", "synthetic_source_keep": "all"}
    });
    let modes = [
        ("tsdb", json!({"index.mode": "time_series", "index.routing_path": ["dim"], "index.time_series.start_time": "2000-01-01T00:00:00Z", "index.time_series.end_time": "2099-01-01T00:00:00Z"})),
        ("logs", json!({"index.mode": "logsdb"})),
        ("std", json!({"index.mapping.source.mode": "synthetic"})),
    ];
    let docs = |i: usize| -> Vec<Value> { vec![
        // Plain: mapped keys, no arrays.
        json!({"@timestamp": ts, "host": {"name": "h1"}, "dim": format!("d{i}"), "k": "abc", "l": 7, "sf": 0.1234, "d": 2.5, "ip": "10.0.0.1", "b": true, "o": {"a": "x", "n": 3, "e": {"x": "y"}}}),
        json!({"@timestamp": ts, "host": {"name": "h2"}, "dim": format!("e{i}"), "k": "toolongvalue", "l": "42", "b": "false", "o": {"e": {}}}),
        json!({"@timestamp": ts, "host": {"name": "h3"}, "dim": format!("f{i}"), "t": "free text", "kept": "v", "d": -0.0}),
        // Not plain: arrays, dotted keys, unmapped fields, nulls.
        json!({"@timestamp": ts, "host": {"name": "h4"}, "dim": format!("g{i}"), "k": ["z", "a", "z"], "l": [3, 1], "o.a": "dotted"}),
        json!({"@timestamp": ts, "host": {"name": "h5"}, "dim": format!("h{i}"), "zz": {"free": [2, 1]}, "k": null, "o": [{"a": "p"}, {"a": "q"}]}),
    ] };
    let mut expected: Vec<(String, Vec<(String, String)>)> = Vec::new();
    {
        let node = Node::open(tmp.path()).unwrap();
        for (i, (name, settings)) in modes.iter().enumerate() {
            let mut settings = settings.clone();
            settings["number_of_shards"] = json!(1);
            node.create_index(name, &json!({"settings": settings, "mappings": {"properties": common}})).unwrap();
            let svc = node.get_index(name).unwrap();
            let docs = docs(i);
            let plain = crate::synthetic::PLAIN_FORMS.with(|n| n.get());
            let ids = write_all(&svc, docs.iter().map(|v| v.to_string()).collect());
            assert!(crate::synthetic::PLAIN_FORMS.with(|n| n.get()) - plain >= 2, "{name}: plain documents take the one walk");
            let meta = svc.meta();
            let lk = svc.lookup();
            crate::synthetic::FORCE_GENERAL.with(|f| f.set(true));
            let want: Vec<String> = docs.iter().map(|v| crate::synthetic::synthesize(v, &meta.mappings, &lk, crate::synthetic::Keep::of_index(&meta.settings)).unwrap().0.to_string()).collect();
            crate::synthetic::FORCE_GENERAL.with(|f| f.set(false));
            let check = |when: &str| {
                for (j, id) in ids.iter().enumerate() {
                    let got: Value = serde_json::from_str(&svc.get(id).unwrap().unwrap().source.unwrap()).unwrap();
                    assert_eq!(got.to_string(), want[j], "{name} {when}: doc {j}");
                }
                if when == "realtime" {
                    return;
                }
                let out = search_index(&node, name, json!({"size": 20, "sort": ["_doc"]}));
                let hits = out["hits"]["hits"].as_array().unwrap();
                assert_eq!(hits.len(), ids.len(), "{name} {when}");
                for h in hits {
                    let j = ids.iter().position(|x| x == h["_id"].as_str().unwrap()).unwrap();
                    assert_eq!(h["_source"].to_string(), want[j], "{name} {when}: hit {j}");
                }
            };
            check("realtime");
            svc.refresh().unwrap();
            check("refreshed");
            if i == 0 {
                // TSDB: flushed and merged too; the others stay in the
                // translog for the replay below.
                svc.flush().unwrap();
                check("flushed");
                svc.force_merge(1, false).unwrap();
                svc.refresh().unwrap();
                check("merged");
            }
            expected.push((name.to_string(), ids.into_iter().zip(want).collect()));
        }
    }
    // Reopened: unflushed docs replay from the translog.
    let node = Node::open(tmp.path()).unwrap();
    for (name, docs) in &expected {
        let svc = node.get_index(name).unwrap();
        svc.refresh().unwrap();
        for (id, want) in docs {
            let got: Value = serde_json::from_str(&svc.get(id).unwrap().unwrap().source.unwrap()).unwrap();
            assert_eq!(&got.to_string(), want, "{name} reopened: {id}");
        }
    }
}

fn search_index(node: &Arc<Node>, index: &str, body: Value) -> Value {
    let request = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
    let targets = node.resolve(index, Default::default()).unwrap();
    let options = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    serde_json::from_str(&crate::search::execute(&targets, &request, &options).unwrap().json).unwrap()
}

/// Per-section and per-field bytes of every segment of `svc`.
pub(crate) fn store_report(svc: &IndexService, n: u64, source_bytes: u64) -> u64 {
    let names: std::collections::HashMap<u16, String> = svc.lookup().fields.iter().filter_map(|(name, f)| Some((f.core_id?, name.clone()))).collect();
    let core_names = |id: u16| match id {
        crate::mapping::SOURCE_FIELD => Some("_source".to_string()),
        crate::mapping::IGNORED_SOURCE_FIELD => Some("_ignored_source".to_string()),
        u16::MAX => Some("(unattributed)".to_string()),
        _ => None,
    };
    let mut total = 0u64;
    let (mut terms, mut postings, mut fast, mut store, mut extra, mut vectors) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut per: std::collections::BTreeMap<String, [u64; 8]> = Default::default();
    let mut segments = 0;
    let (mut chunks, mut zstd_chunks, mut stored_raw) = (0usize, 0usize, 0u64);
    for sh in svc.local_shards() {
        let core = sh.shard.core.read();
        for seg in core.segments() {
            segments += 1;
            let du = seg.disk_usage().unwrap();
            let st = seg.store().unwrap();
            chunks += st.num_chunks();
            zstd_chunks += (0..st.num_chunks()).filter(|&c| st.chunk_zstd(c)).count();
            stored_raw += du.fields.values().map(|u| u.stored_raw).sum::<u64>();
            terms += du.terms;
            postings += du.postings;
            fast += du.fast;
            store += du.store;
            extra += du.extra;
            vectors += du.vectors;
            for (id, u) in &du.fields {
                let name = core_names(*id).or_else(|| names.get(id).cloned()).unwrap_or_else(|| format!("#{id}"));
                let e = per.entry(name).or_default();
                for (k, v) in [u.terms, u.postings, u.doc_values, u.doc_values_terms, u.points, u.norms, u.stored, u.geo + u.shape + u.other_extra].into_iter().enumerate() {
                    e[k] += v;
                }
            }
        }
    }
    total += terms + postings + fast + store + extra + vectors;
    let b = |x: u64| x as f64 / n as f64;
    eprintln!("{n} docs ({segments} segment(s)), JSON {:.0} B/doc: total {:.1} B/doc (track: ES 1,196, c32b 4,957)", b(source_bytes), b(total));
    eprintln!("  terms {:.1}  postings {:.1}  fast (doc values, points, norms) {:.1}  stored {:.1} (raw {:.1}, zstd chunks {zstd_chunks} of {chunks})  extra {:.1}  B/doc", b(terms), b(postings), b(fast), b(store), b(stored_raw), b(extra));
    let mut rows: Vec<(String, [u64; 8])> = per.into_iter().collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.iter().sum::<u64>()));
    eprintln!("  {:<44} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}  (B/doc)", "field", "terms", "post", "dv", "dvterm", "points", "norms", "stored", "extra");
    for (name, v) in rows.iter().take(40) {
        eprintln!("  {:<44} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}", name, b(v[0]), b(v[1]), b(v[2]), b(v[3]), b(v[4]), b(v[5]), b(v[6]), b(v[7]));
    }
    total
}

#[test]
#[ignore]
fn k8s_store_microbench() {
    let n = env("K8S_DOCS", 200_000);
    let pods = env("K8S_PODS", 1000);
    let (codec, mode) = (codec_env(), mode_env());
    let tsdb = mode == "time_series";
    cluster_format();
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let license = license_env(&node);
    install_templates_mode(&node, &codec, &mode);
    let svc = create_stream(&node);
    let shared = Shared::new();
    let (base, tick) = clock(n, pods, tsdb);
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let mut source_bytes = 0u64;
    let started = std::time::Instant::now();
    let mut i = 0;
    let mut real = corpus(n).map(|v| v.into_iter());
    while i < n {
        let docs: Vec<String> = match real.as_mut() {
            Some(it) => it.take(1000).collect(),
            None => (i..(i + 1000).min(n)).map(|j| k8s_doc(j, pods, base, tick, &shared, &mut r).to_string()).collect(),
        };
        source_bytes += docs.iter().map(|d| d.len() as u64).sum::<u64>();
        i += docs.len() as u64;
        write_all(&svc, docs);
    }
    let indexed = started.elapsed();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    eprintln!("license {license}, codec {codec}, {mode} mode: indexed in {:.1}s ({:.0} docs/s), force-merged in {:.1}s", indexed.as_secs_f64(), n as f64 / indexed.as_secs_f64(), (started.elapsed() - indexed).as_secs_f64());
    store_report(&svc, n, source_bytes);
}

#[test]
#[ignore]
fn k8s_ingest_microbench() {
    let n = env("K8S_DOCS", 400_000);
    let pods = env("K8S_PODS", 1000);
    let clients = env("K8S_CLIENTS", 8).max(1);
    let bulk = env("K8S_BULK", 1000).max(1);
    let (codec, mode) = (codec_env(), mode_env());
    let tsdb = mode == "time_series";
    cluster_format();
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let license = license_env(&node);
    install_templates_mode(&node, &codec, &mode);
    let svc = create_stream(&node);
    let shared = Shared::new();
    let (base, tick) = clock(n, pods, tsdb);
    // Sources are generated before the clock starts (Rally reads them from
    // its corpus file).
    let mut r = Rng(0x51_7cc1_b727_220a);
    let docs: Vec<String> = corpus(n).unwrap_or_else(|| (0..n).map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r).to_string()).collect());
    let bytes: u64 = docs.iter().map(|d| d.len() as u64).sum();
    let bulks: Vec<Vec<String>> = docs.chunks(bulk as usize).map(|c| c.to_vec()).collect();
    drop(docs);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let latencies = parking_lot::Mutex::new(Vec::with_capacity(bulks.len()));
    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..clients {
            s.spawn(|| loop {
                let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(batch) = bulks.get(b) else { break };
                let t = std::time::Instant::now();
                write_all(&svc, batch.clone());
                // `refresh=true` on every bulk request.
                svc.refresh().unwrap();
                latencies.lock().push(t.elapsed().as_secs_f64() * 1000.0);
            });
        }
    });
    let secs = started.elapsed().as_secs_f64();
    let mut l = latencies.into_inner();
    l.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| l[((l.len() - 1) as f64 * p) as usize];
    let segments: usize = svc.local_shards().iter().map(|sh| sh.shard.core.read().segments().len()).sum();
    eprintln!(
        "license {license}, codec {codec}, {mode} mode, {clients} clients x {bulk}-doc bulks + refresh: {n} docs ({:.0} B/doc JSON) in {secs:.1}s = {:.0} docs/s (track: ES 11,239, c32b 6,156); bulk p50 {:.0} ms p90 {:.0} ms p99 {:.0} ms (track p50: ES 668, c32b 1,278); {segments} segments",
        bytes as f64 / n as f64,
        n as f64 / secs,
        pct(0.5),
        pct(0.9),
        pct(0.99)
    );
    store_report(&svc, n, bytes);
}

/// ES `NumberFieldMapper` in a time_series index: `counter` / `gauge`
/// metrics default to `index: false` (doc values only, no points index);
/// range queries on them run on doc values and match the brute-force count.
#[test]
fn tsdb_metrics_are_doc_values_only_and_range_queries_run_on_them() {
    let metrics = ["kubernetes.pod.memory.rss.bytes", "kubernetes.pod.network.tx.bytes", "kubernetes.pod.cpu.usage.node.pct"];
    // A standard stream keeps them indexed.
    {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        install_templates(&node, "default", false);
        let svc = create_stream(&node);
        let lookup = svc.lookup();
        for f in metrics {
            assert!(lookup.get(f).unwrap().indexed, "standard {f}");
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    install_templates(&node, "default", true);
    let svc = create_stream(&node);
    let lookup = svc.lookup();
    let schema = lookup.core_schema();
    for f in metrics {
        let rf = lookup.get(f).unwrap();
        assert!(!rf.indexed && rf.doc_values && rf.skipper, "{f}");
        let core = schema.iter().find(|c| Some(c.id) == rf.core_id).unwrap();
        assert!(!core.points, "{f}: no points index");
    }
    // ES 9.5 with time series skippers: keyword / ip / numeric fields are
    // doc values only too; `@timestamp` stays indexed.
    assert!(lookup.get("@timestamp").unwrap().indexed);
    for f in ["kubernetes.pod.name", "kubernetes.pod.uid", "kubernetes.pod.ip"] {
        let rf = lookup.get(f).unwrap();
        assert!(!rf.indexed && rf.doc_values, "{f}: doc values only");
    }
    drop(lookup);
    let shared = Shared::new();
    let (pods, n) = (20u64, 1200u64);
    let (base, tick) = clock(n, pods, true);
    let mut r = Rng(11);
    let docs: Vec<Value> = (0..n).map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r)).collect();
    write_all(&svc, docs[..n as usize / 2].iter().map(Value::to_string).collect());
    svc.refresh().unwrap();
    write_all(&svc, docs[n as usize / 2..].iter().map(Value::to_string).collect());
    svc.refresh().unwrap();
    let num = |v: &Value| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap();
    let count = |field: &str, lo: f64, hi: f64| -> u64 {
        let out = search(&node, json!({"size": 0, "track_total_hits": true, "query": {"range": {field: {"gte": lo, "lt": hi}}}}));
        out["hits"]["total"]["value"].as_u64().unwrap()
    };
    for merged in [false, true] {
        if merged {
            svc.force_merge(1, false).unwrap();
            svc.refresh().unwrap();
        }
        // Term queries on a doc-values-only keyword dimension.
        let name = docs[3]["kubernetes"]["pod"]["name"].as_str().unwrap().to_string();
        let want = docs.iter().filter(|d| d["kubernetes"]["pod"]["name"] == name.as_str()).count() as u64;
        let out = search(&node, json!({"size": 0, "track_total_hits": true, "query": {"term": {"kubernetes.pod.name": name}}}));
        assert_eq!(out["hits"]["total"]["value"].as_u64().unwrap(), want, "term on a doc-values keyword, merged={merged}");
        // Query parsers' term leaves on it too (the tsdb / data stream
        // YAML suites' `query_string: '+dim:value'`).
        for q in [
            json!({"query_string": {"query": format!("+kubernetes.pod.name:\"{name}\"")}}),
            json!({"query_string": {"query": format!("\"{name}\""), "default_field": "kubernetes.pod.name"}}),
            json!({"simple_query_string": {"query": format!("\"{name}\""), "fields": ["kubernetes.pod.name"]}}),
            json!({"multi_match": {"query": name, "fields": ["kubernetes.pod.name"]}}),
            json!({"bool": {"filter": [{"query_string": {"query": format!("kubernetes.pod.name:\"{name}\"")}}]}}),
        ] {
            let out = search(&node, json!({"size": 0, "track_total_hits": true, "query": q}));
            assert_eq!(out["hits"]["total"]["value"].as_u64().unwrap(), want, "{q} merged={merged}");
        }
        let rss: Vec<f64> = docs.iter().map(|d| num(&d["kubernetes"]["pod"]["memory"]["rss"]["bytes"])).collect();
        let tx: Vec<f64> = docs.iter().map(|d| num(&d["kubernetes"]["pod"]["network"]["tx"]["bytes"])).collect();
        for (field, values, lo, hi) in [("kubernetes.pod.memory.rss.bytes", &rss, 900_000.0, 1_200_000.0), ("kubernetes.pod.network.tx.bytes", &tx, 600.0, 2_000.0)] {
            let want = values.iter().filter(|&&v| v >= lo && v < hi).count() as u64;
            assert!(want > 0 && want < n, "{field}: a selective range");
            assert_eq!(count(field, lo, hi), want, "{field} merged={merged}");
        }
    }
}

/// ES `TimeSeriesIdFieldMapper`: `_tsid` is doc values only (no terms or
/// postings in any segment), still sorts the index, aggregates and sorts,
/// and stays unsearchable; metric columns carry no points index.
#[test]
fn tsdb_tsid_is_doc_values_only() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    install_templates(&node, "default", true);
    let svc = create_stream(&node);
    let lookup = svc.lookup();
    let tsid = lookup.get(crate::tsdb::TSID_FIELD).unwrap();
    assert!(!tsid.indexed && tsid.doc_values);
    let tsid_id = tsid.core_id.unwrap();
    let rss = lookup.get("kubernetes.pod.memory.rss.bytes").and_then(|f| f.core_id).unwrap();
    drop(lookup);
    let shared = Shared::new();
    let (pods, n) = (25u64, 1500u64);
    let (base, tick) = clock(n, pods, true);
    let mut r = Rng(5);
    for half in [0..n / 2, n / 2..n] {
        write_all(&svc, half.map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r).to_string()).collect());
        svc.refresh().unwrap();
    }
    let check = |label: &str| {
        let sh = svc.local_shards().remove(0);
        let core = sh.shard.core.read();
        assert!(!core.segments().is_empty(), "{label}");
        for seg in core.segments() {
            let usage = seg.disk_usage().unwrap();
            let t = usage.fields.get(&tsid_id).cloned().unwrap_or_default();
            assert_eq!((t.terms, t.term_count, t.postings), (0, 0, 0), "{label}: _tsid postings");
            assert!(t.doc_values + t.doc_values_terms > 0, "{label}: _tsid column");
            assert_eq!(usage.fields.get(&rss).map_or(0, |f| f.points), 0, "{label}: metric points");
        }
    };
    check("sealed");
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    check("merged");
    let out = search(&node, json!({"size": 0, "aggs": {"t": {"terms": {"field": "_tsid", "size": 100}}}}));
    assert_eq!(out["aggregations"]["t"]["buckets"].as_array().unwrap().len() as u64, pods);
    let out = search(&node, json!({"size": 3, "sort": [{"_tsid": "asc"}, {"@timestamp": "desc"}]}));
    assert_eq!(out["hits"]["hits"].as_array().unwrap().len(), 3);
    let request = crate::search::parse_request(&json!({"query": {"term": {"_tsid": "x"}}}), &std::collections::HashMap::new()).unwrap();
    let targets = node.resolve(DS, Default::default()).unwrap();
    let options = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    assert!(crate::search::execute(&targets, &request, &options).is_err(), "_tsid stays unsearchable");
}

/// ES `index.mapping.synthetic_id` (time series): sealed segments keep no
/// `_id` postings and no stored `_id`; GET, `ids` queries, hit `_id`s,
/// duplicate detection and deletes resolve through `_tsid` + `@timestamp`
/// (memtables, sealed segments and the merged segment alike).
#[test]
fn tsdb_synthetic_id_is_neither_indexed_nor_stored() {
    use crate::mapping::ID_FIELD;
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    install_templates(&node, "default", true);
    let svc = create_stream(&node);
    assert!(svc.lookup().registry.synthetic_id, "synthetic _id with the default codec and the tsid sort");
    let shared = Shared::new();
    let (pods, n) = (12u64, 720u64);
    let (base, tick) = clock(n, pods, true);
    let mut r = Rng(3);
    let docs: Vec<String> = (0..n).map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r).to_string()).collect();
    let mut ids = write_all(&svc, docs[..n as usize / 2].to_vec());
    let get_id = |id: &str| -> Option<String> {
        svc.get(id).unwrap().map(|g| g.doc.get(ID_FIELD).and_then(|v| v.as_text()).map(str::to_string).unwrap_or_default())
    };
    // Realtime GET from the memtable.
    assert_eq!(get_id(&ids[7]).as_deref(), Some(ids[7].as_str()));
    svc.refresh().unwrap();
    ids.extend(write_all(&svc, docs[n as usize / 2..].to_vec()));
    svc.refresh().unwrap();
    let total = || search(&node, json!({"size": 0, "track_total_hits": true}))["hits"]["total"]["value"].as_u64().unwrap();
    let check = |label: &str, deleted: &[&str]| {
        let sh = svc.local_shards().remove(0);
        {
            let core = sh.shard.core.read();
            for seg in core.segments() {
                let u = seg.disk_usage().unwrap().fields.get(&ID_FIELD).cloned().unwrap_or_default();
                assert_eq!((u.terms, u.term_count, u.postings, u.stored_raw), (0, 0, 0, 0), "{label}: no _id postings or stored values");
            }
        }
        for i in [0usize, 5, 359, 360, 719] {
            let id = &ids[i];
            if deleted.contains(&id.as_str()) {
                assert_eq!(get_id(id), None, "{label}: deleted {id}");
            } else {
                assert_eq!(get_id(id).as_deref(), Some(id.as_str()), "{label}: GET {id}");
            }
        }
        let wanted: Vec<&str> = [1usize, 100, 600].iter().map(|&i| ids[i].as_str()).collect();
        let out = search(&node, json!({"size": 10, "query": {"ids": {"values": wanted}}}));
        let mut got: Vec<String> = out["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        got.sort();
        let mut want: Vec<String> = wanted.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(got, want, "{label}: ids query and hit _id");
    };
    check("sealed", &[]);
    assert_eq!(total(), n);
    // Indexing the same documents again finds their ids: no duplicates.
    write_all(&svc, docs[..50].to_vec());
    svc.refresh().unwrap();
    assert_eq!(total(), n, "re-indexed documents are not duplicated");
    // Delete by id.
    let gone = ids[5].clone();
    let out = svc.write(vec![WriteReq::Delete { id: gone.clone(), routing: None, cond: Default::default() }], false);
    assert!(out[0].is_ok(), "{:?}", out[0].as_ref().err());
    svc.refresh().unwrap();
    assert_eq!(total(), n - 1);
    check("after delete", &[&gone]);
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    check("merged", &[&gone]);
    assert_eq!(total(), n - 1);
    // Update by id: as ES, a time series index refuses it or applies it in place.
    match svc.update(&ids[0], &json!({"doc": {"kubernetes": {"pod": {"network": {"tx": {"errors": 7}}}}}}), None, 0, Default::default()) {
        Err(e) => assert!(e.reason.contains("time_series") || e.reason.contains("sequence numbers disabled"), "{}", e.reason),
        Ok(_) => {
            svc.refresh().unwrap();
            assert_eq!(get_id(&ids[0]).as_deref(), Some(ids[0].as_str()));
            assert_eq!(total(), n - 1);
        }
    }
}

/// What a time series pod document's stored `_source` residual holds (the
/// canonical synthetic source minus the leaves its columns rebuild), for
/// real corpus docs and the live mapping:
/// `TSQ_MAPPING=pod_mapping.json TSQ_CORPUS=doc-ds-metrics-kubernetes.pod.json
///  cargo test --profile fast -p elasticsearch-engine --lib tsq_residual_probe -- --ignored --nocapture`
#[test]
#[ignore]
fn tsq_residual_probe() {
    use std::io::BufRead;
    let mapping: Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("TSQ_MAPPING").unwrap()).unwrap()).unwrap();
    let mappings = mapping.as_object().unwrap().values().next().unwrap()["mappings"].clone();
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    {
        let mut t = node.templates.write();
        t.index.insert("tsq".into(), json!({"index_patterns": ["k8s-probe*"], "data_stream": {}, "priority": 500,
            "template": {"settings": {"index": {"mode": "time_series", "number_of_shards": 1, "number_of_replicas": 0, "time_series": {"start_time": "2000-01-01T00:00:00Z"}}}, "mappings": mappings}}));
    }
    let ds = node.create_data_stream("k8s-probe").unwrap();
    let svc = node.get_index(ds.write_index()).unwrap();
    let f = std::io::BufReader::new(std::fs::File::open(std::env::var("TSQ_CORPUS").unwrap()).unwrap());
    let docs: Vec<String> = f.lines().take(2000).map(|l| l.unwrap()).collect();
    let ids = write_all(&svc, docs);
    svc.refresh().unwrap();
    let sh = svc.local_shards().remove(0);
    let core = sh.shard.core.read();
    let mut total = 0usize;
    let mut keys: std::collections::BTreeMap<String, usize> = Default::default();
    let mut shown = 0;
    for seg in core.segments() {
        for d in 0..seg.num_docs {
            let Ok(Some(doc)) = seg.get_doc(d) else { continue };
            let Some(raw) = doc.get(crate::mapping::SOURCE_FIELD).and_then(|v| v.as_text()) else { continue };
            total += raw.len();
            let json = raw.rsplit('\u{1}').next().unwrap_or(raw);
            if shown < 2 { println!("residual: {raw}"); shown += 1; }
            if let Ok(v) = serde_json::from_str::<Value>(json) {
                fn walk(v: &Value, p: &str, out: &mut std::collections::BTreeMap<String, usize>) {
                    match v {
                        Value::Object(m) => for (k, x) in m { walk(x, &if p.is_empty() { k.clone() } else { format!("{p}.{k}") }, out) },
                        _ => *out.entry(p.to_string()).or_default() += v.to_string().len(),
                    }
                }
                walk(&v, "", &mut keys);
            }
        }
    }
    println!("docs {} residual bytes {total}", ids.len());
    let mut by: Vec<_> = keys.into_iter().collect();
    by.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in by.iter().take(25) { println!("{n:>9} {k}"); }
}

/// A time series doc's stored `_source` residual lists its column ids as
/// runs and keeps no `constant_keyword` values (rebuilt from the mapping);
/// `_source` reads back the same.
#[test]
fn tsdb_residual_is_compact_and_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    install_templates(&node, "default", true);
    let svc = create_stream(&node);
    let shared = Shared::new();
    let (pods, n) = (5u64, 200u64);
    let (base, tick) = clock(n, pods, true);
    let mut r = Rng(9);
    let docs: Vec<Value> = (0..n).map(|i| k8s_doc(i, pods, base, tick, &shared, &mut r)).collect();
    // The first document sets the constant_keyword values in the mapping.
    let mut ids = write_all(&svc, vec![docs[0].to_string()]);
    svc.refresh().unwrap();
    ids.extend(write_all(&svc, docs[1..].iter().map(Value::to_string).collect()));
    svc.refresh().unwrap();
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    {
        let sh = svc.local_shards().remove(0);
        let core = sh.shard.core.read();
        let mut seen = 0;
        let mut with_constants = 0;
        for seg in core.segments() {
            for d in 0..seg.num_docs {
                let doc = seg.get_doc(d).unwrap().unwrap();
                let raw = doc.get(crate::mapping::SOURCE_FIELD).and_then(|v| v.as_text()).unwrap().to_string();
                assert!(crate::synthetic::is_residual(&raw), "residual");
                with_constants += raw.contains("data_stream") as usize;
                // A projection of a few roots equals the whole source's.
                let lookup = svc.lookup();
                let leaf = rustic_core::query::LeafReader::Segment(seg);
                let cols = crate::synthetic::LeafColumns { leaf: &leaf, doc: d };
                let full: Value = serde_json::from_str(&crate::synthetic::rehydrate(&raw, &lookup, &cols).unwrap()).unwrap();
                for roots in [vec!["kubernetes", "@timestamp"], vec!["data_stream"], vec!["host", "event", "nope"]] {
                    let part = crate::synthetic::rehydrate_roots(&raw, &lookup, &cols, &|k| roots.contains(&k)).expect("projected");
                    let want: serde_json::Map<String, Value> = full.as_object().unwrap().iter().filter(|(k, _)| roots.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
                    assert_eq!(part, Value::Object(want), "{roots:?}");
                }
                assert!(raw.contains('-'), "id runs: {raw}");
                seen += 1;
            }
        }
        assert_eq!(seen, n);
        assert!(with_constants <= 1, "constant_keyword values are not stored ({with_constants} docs keep them)");
    }
    for (i, id) in ids.iter().enumerate().step_by(37) {
        let got: Value = serde_json::from_str(&svc.get(id).unwrap().unwrap().source.unwrap()).unwrap();
        assert_eq!(got["data_stream"]["dataset"], "kubernetes.pod", "{id}");
        assert_eq!(got["kubernetes"]["pod"]["name"], docs[i]["kubernetes"]["pod"]["name"]);
    }
    let out = search(&node, json!({"size": 3, "query": {"term": {"data_stream.dataset": "kubernetes.pod"}}}));
    assert_eq!(out["hits"]["hits"][0]["_source"]["data_stream"]["type"], "metrics");
}
