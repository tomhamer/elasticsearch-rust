//! Query DSL translation.
pub mod dsl;
pub mod strict;
pub mod kql;
pub mod query_string;
pub mod cross_fields;
pub mod combined_fields;
pub mod synonym;
pub mod geo_decay;

pub use dsl::{parse_knn_query, parse_query, QueryContext};
pub mod positional;
pub mod extra;
pub mod docvalues;
pub mod geo;
pub mod terms_set;
pub mod regexp;
pub mod pinned;

pub mod limits;
