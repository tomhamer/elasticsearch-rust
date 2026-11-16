//! Trained models survive a change of master (ES keeps configs and
//! definitions in the replicated `.ml-inference-*` system indices and
//! deployments in cluster-state metadata): a tiny PyTorch model uploaded and
//! deployed through one node still answers the same inference after the
//! elected master is stopped, with no files copied by hand.
mod common;
use base64::Engine as _;
use common::procs::start_procs;
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const FAST_DETECTION: [(&str, &str); 3] = [("cluster.fault_detection.follower_check.interval", "500ms"),
    ("cluster.fault_detection.follower_check.timeout", "2s"), ("cluster.fault_detection.follower_check.retry_count", "2")];

/// A one-entry stored (uncompressed) zip archive.
fn zip_stored(name: &str, data: &[u8]) -> Vec<u8> {
    let le16 = |v: usize| (v as u16).to_le_bytes();
    let le32 = |v: usize| (v as u32).to_le_bytes();
    let mut z = Vec::new();
    z.extend(le32(0x0403_4b50));
    z.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    z.extend(le32(data.len()));
    z.extend(le32(data.len()));
    z.extend(le16(name.len()));
    z.extend(le16(0));
    z.extend(name.as_bytes());
    z.extend(data);
    let cd = z.len();
    z.extend(le32(0x0201_4b50));
    z.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    z.extend(le32(data.len()));
    z.extend(le32(data.len()));
    z.extend(le16(name.len()));
    z.extend([0; 12]);
    z.extend(le32(0));
    z.extend(name.as_bytes());
    let cd_len = z.len() - cd;
    z.extend(le32(0x0605_4b50));
    z.extend([0, 0, 0, 0, 1, 0, 1, 0]);
    z.extend(le32(cd_len));
    z.extend(le32(cd));
    z.extend([0, 0]);
    z
}

const CODE: &str = "class Tiny(Module):\n  def forward(self: __torch__.Tiny,\n    input_ids: Tensor) -> Tensor:\n    torch.manual_seed(torch.hash(str(input_ids)))\n    return torch.rand([1, 1, 8])\n";

async fn wait_green(pl: &common::procs::Procs, via: usize, index: &str) {
    pl.wait_health(via, index, "green", 120).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uploaded_model_infers_after_master_change() {
    let mut pl = start_procs(3, &FAST_DETECTION).await;
    let master = pl.master(0).await;
    let other = (master + 1) % 3;
    let archive = zip_stored("tiny/code/__torch__.py", CODE.as_bytes());
    let b64 = base64::engine::general_purpose::STANDARD.encode(&archive);
    // Upload and deploy through a node that is not the master.
    let (st, v) = pl.call(other, Method::PUT, "/_ml/trained_models/tiny", Some(json!({
        "model_type": "pytorch", "input": {"field_names": ["text_field"]},
        "inference_config": {"text_embedding": {"tokenization": {"bert": {"with_special_tokens": false}}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(other, Method::PUT, "/_ml/trained_models/tiny/definition/0", Some(json!({
        "definition": b64, "total_definition_length": archive.len(), "total_parts": 1}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(other, Method::PUT, "/_ml/trained_models/tiny/vocabulary", Some(json!({
        "vocabulary": ["[PAD]", "[UNK]", "these", "are", "my", "words"]}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = pl.call(other, Method::POST, "/_ml/trained_models/tiny/deployment/_start?wait_for=started", None).await;
    assert_eq!(st, 200, "{v}");
    let doc = json!({"docs": [{"text_field": "these are my words"}]});
    let (st, before) = pl.call(other, Method::POST, "/_ml/trained_models/tiny/_infer", Some(doc.clone())).await;
    assert_eq!(st, 200, "{before}");
    let embedding = before["inference_results"][0]["predicted_value"].clone();
    assert_eq!(embedding.as_array().map(Vec::len), Some(8), "{before}");

    // The model lives in the replicated system indices, fully replicated.
    for index in [".ml-inference-000005", ".ml-inference-native-000002"] {
        wait_green(&pl, other, index).await;
        let shards = pl.shards(other, index).await;
        assert_eq!(shards.iter().filter(|s| s["state"] == "STARTED").count(), 2, "{index}: {shards:?}");
    }

    // Stop the master; another node is elected.
    pl.stop(master, 60).await;
    let survivor = (master + 1) % 3;
    let deadline = Instant::now() + Duration::from_secs(90);
    let new_master = loop {
        let m = pl.master(survivor).await;
        if m != master {
            break m;
        }
        assert!(Instant::now() < deadline, "no new master");
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    pl.wait_nodes(survivor, 2, 90).await;
    for index in [".ml-inference-000005", ".ml-inference-native-000002"] {
        pl.wait_health(survivor, index, "yellow", 90).await;
    }

    // The same model, deployment and inference on the new master.
    for via in [new_master, survivor] {
        let (st, v) = pl.call(via, Method::GET, "/_ml/trained_models/tiny", None).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["trained_model_configs"][0]["model_id"], "tiny", "{v}");
        let (st, v) = pl.call(via, Method::GET, "/_ml/trained_models/tiny/_stats", None).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["trained_model_stats"][0]["deployment_stats"]["state"], "started", "{v}");
        let (st, after) = pl.call(via, Method::POST, "/_ml/trained_models/tiny/_infer", Some(doc.clone())).await;
        assert_eq!(st, 200, "{after}");
        assert_eq!(after["inference_results"][0]["predicted_value"], embedding, "{after}");
    }
    let _: Value = Value::Null;
}
