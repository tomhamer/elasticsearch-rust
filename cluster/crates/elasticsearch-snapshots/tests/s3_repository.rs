//! S3 integration test (ignored by default). Run with:
//!
//! ```sh
//! ES_TEST_S3_BUCKET=my-bucket AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
//! [AWS_SESSION_TOKEN=...] [AWS_REGION=us-east-1] \
//! [ES_TEST_S3_ENDPOINT=http://localhost:9000 ES_TEST_S3_PATH_STYLE=true]  # MinIO
//! cargo test -p elasticsearch-snapshots --test s3_repository -- --ignored --nocapture
//! ```
//!
//! Uses a unique random `base_path` and deletes everything under it afterwards.

mod common;

use common::*;
use elasticsearch_snapshots::{RepositoryContext, RepositoryMetadata, S3ClientSettings, SnapshotsService};
use serde_json::json;
use std::collections::BTreeMap;

#[tokio::test]
#[ignore = "needs a real S3 bucket: set ES_TEST_S3_BUCKET + AWS credentials"]
async fn s3_full_cycle() {
    let bucket = std::env::var("ES_TEST_S3_BUCKET").expect("ES_TEST_S3_BUCKET must be set");
    let region = std::env::var("AWS_REGION").ok().filter(|r| !r.is_empty()).unwrap_or_else(|| "us-east-1".into());
    let mut client = S3ClientSettings { region: Some(region), ..Default::default() };
    if let Ok(ep) = std::env::var("ES_TEST_S3_ENDPOINT") {
        client.endpoint = Some(ep);
    }
    client.path_style_access = std::env::var("ES_TEST_S3_PATH_STYLE").map(|v| v == "true").unwrap_or(false);
    let mut clients = BTreeMap::new();
    clients.insert("default".to_string(), client);
    let svc = SnapshotsService::new(RepositoryContext { s3_clients: clients, ..Default::default() });

    let base_path = format!("es-snapshots-test/{}", uuid_like());
    // 5mb is the minimum S3 chunk size; with a 5mb chunk the 15mb+ file is
    // stored in 4 parts and buffer_size 5mb forces multipart uploads.
    let chunk = 5 * 1024 * 1024;
    svc.put_repository(
        RepositoryMetadata::new(
            "s3repo",
            "s3",
            json!({
                "bucket": bucket,
                "base_path": base_path,
                "chunk_size": "5mb",
                "buffer_size": "5mb",
                "compress": true,
                "max_snapshot_bytes_per_sec": "0",
            }),
        ),
        true,
    )
    .await
    .expect("put s3 repository");

    let work = tempfile::tempdir().unwrap();
    let result = tokio::spawn({
        let svc = svc.clone();
        let path = work.path().to_path_buf();
        async move { full_cycle(&svc, "s3repo", &path, chunk).await }
    })
    .await;

    // Always clean up the base path.
    let repo = svc.repository("s3repo").unwrap();
    let cleaned = repo.root().delete().await;
    eprintln!("cleanup of s3://{bucket}/{base_path}: {cleaned:?}");
    if let Err(e) = result {
        std::panic::resume_unwind(e.into_panic());
    }
    assert!(repo.root().list_blobs().await.unwrap().is_empty());
}

fn uuid_like() -> String {
    let b = random_bytes(8);
    b.iter().map(|x| format!("{x:02x}")).collect()
}
