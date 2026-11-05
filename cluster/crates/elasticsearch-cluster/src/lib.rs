//! Cluster-level logic for the `elasticsearch` rewrite. Everything here is
//! deterministic and I/O free so it can be tested by simulation; the
//! `elasticsearch-node` crate drives it over the network.

pub mod allocation;
pub mod coordination;
pub mod compatibility;
pub mod health;
pub mod routing;
pub mod state;
pub mod wire;
