//! Kubernetes operator for the `elasticsearch` cluster: an ECK-compatible
//! subset (`elasticsearch.k8s.elastic.co/v1` `Elasticsearch`).
//!
//! Layout:
//! * [`crd`] — the custom resource types and CRD generation;
//! * [`names`] — ECK-compatible object names, labels, annotations;
//! * [`settings`] — spec validation and `elasticsearch.yml` rendering;
//! * [`resources`] — pure builders for StatefulSets, Services, ConfigMaps, PDB;
//! * [`es_client`] — the [`es_client::EsClient`] trait (+ reqwest implementation);
//! * [`scale`] — upscale / safe-downscale planner (pure);
//! * [`rolling`] — rolling-restart planner (pure);
//! * [`executor`] — runs planner output against an `EsClient`;
//! * [`status`] — status/conditions computation (pure);
//! * [`controller`] — kube-rs controller wiring.

pub mod controller;
pub mod crd;
pub mod es_client;
pub mod executor;
pub mod names;
pub mod resources;
pub mod rolling;
pub mod scale;
pub mod settings;
pub mod status;

#[cfg(test)]
mod sim;
