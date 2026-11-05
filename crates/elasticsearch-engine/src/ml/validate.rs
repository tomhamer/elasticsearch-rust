//! `_validate` / `_validate/detector` (the parsing + validation of
//! [`super::job`] without the cluster-state checks of a put).

pub use super::job::{validate_detector_body as detector, validate_job_body as job};
