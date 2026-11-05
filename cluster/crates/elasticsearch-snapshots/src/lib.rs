//! Snapshot / restore for the `elasticsearch` cluster layer.
//!
//! Storage only (no engine dependency): the node crate hands over shard
//! directories + file lists and index/global metadata JSON, and executes
//! restore plans.
//!
//! * [`blobstore`]: `BlobStore` / `BlobContainer` with `fs`, `s3` and `url` implementations.
//! * [`repository`]: ES `BlobStoreRepository` layout, generations, incremental
//!   shard snapshots, delete / clone / cleanup / verify.
//! * [`service`]: [`service::SnapshotsService`] — repository registry and the
//!   `_snapshot` operations with ES REST response shapes.
//! * [`restore`]: restore request parsing/validation and planning.
//! * [`slm`]: snapshot lifecycle policies, cron/interval schedules, date-math
//!   names, retention, `_slm` response shapes.

pub mod blobstore;
pub mod error;
pub mod repository;
pub mod restore;
pub mod service;
pub mod slm;
pub mod snapshot;
pub mod util;

pub use error::{Result, SnapshotError};
pub use repository::{Repository, RepositoryContext, RepositoryMetadata, S3ClientSettings};
pub use restore::{ExistingIndex, RestoreIndexPlan, RestorePlan, RestoreRequest};
pub use service::{
    CreateSnapshotParams, CreateSnapshotRequest, GetSnapshotsRequest, IndexSnapshotInput, RemoteShard, RemoteShardRequest,
    RemoteShardSnapshots, RemoteUploader, ShardSnapshotInput, SnapshotsService, SortBy,
};
pub use snapshot::{RenderOptions, SnapshotInfo, SnapshotState};
pub use util::CancelToken;
