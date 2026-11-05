pub mod aggs;
pub mod h3;
pub mod slice;
pub mod analysis;
pub mod date;
pub mod document;
pub mod synthetic;
pub mod stored;
pub mod error;
pub mod memory;
pub mod indexing_pressure;
pub mod jsonpos;
pub mod source_json;
pub mod search_strict;
pub mod search_ids;
pub mod highlight;
pub mod termvectors;
pub mod index;
pub mod mapping;
pub mod query;
pub mod script;
pub mod search;
pub mod suggest;
pub mod collapse;
pub mod segsort;
pub mod explain_fmt;
pub mod similarity;
pub(crate) mod parse_depth;
pub use rustic_core::sloppy_math;
pub mod warnings;
pub mod settings;
pub mod settings_filter;
pub mod tsdb;
pub mod indexsort;
pub mod shard;
pub mod refresh;
pub mod translog;
pub mod values;
pub mod node;
pub mod ccs_stats;
pub mod fielddata_cache;
pub mod processors;
pub mod system_indices;
pub mod version;
pub mod tasks;
pub mod datastream;
pub mod migrate;
pub mod async_search;
pub mod reindex;
pub mod remote;
pub mod field_caps;
pub mod fieldtypes;
pub mod field_usage;
pub mod ingest;
pub use rustic_core;
pub mod esql;
pub mod sql;
pub mod eql;
pub mod template;
pub mod search_ext;
pub mod retriever;
pub mod vectors;
pub mod gpu;
pub mod stateless;
pub mod vector_source;
pub mod vector_stats;
pub mod rank_eval;
pub mod enrich;
pub mod ccr;
pub mod transform;
pub mod connector;
pub mod fleet;
pub mod text_structure;
pub mod hooks;
pub mod security;
pub mod ilm;
pub mod inference;
pub mod snapshot;
pub mod watcher;
pub mod license;
pub mod ml;
pub mod coord;
pub mod profiling;
pub mod registry;

#[cfg(test)]
mod geo_count_aggs_tests;
#[cfg(test)]
mod geo_suite_microbench;
#[cfg(test)]
mod k8s_metrics_microbench;
#[cfg(test)]
mod merge_microbench;
#[cfg(test)]
mod nested_microbench;
#[cfg(test)]
mod noaa_store_microbench;
#[cfg(test)]
mod par_count_microbench;
#[cfg(test)]
mod text_fielddata_tests;

/// Runs last when the suite runs on one thread (tests run in name order):
/// every reservation on the process's breakers has been released by then,
/// or something leaked one. `cargo test -p elasticsearch-engine --lib --
/// --test-threads=1`. On several threads other tests may still hold theirs:
/// it only reports.
#[cfg(test)]
mod zzz_suite_end {
    #[test]
    fn process_reservations_return_to_zero() {
        let breakers = crate::memory::process_breakers();
        let args: Vec<String> = std::env::args().collect();
        let serial = args.iter().any(|a| a == "--test-threads=1") || args.windows(2).any(|w| w[0] == "--test-threads" && w[1] == "1")
            || std::env::var("RUST_TEST_THREADS").is_ok_and(|v| v == "1");
        let held = crate::memory::Kind::ALL.iter().any(|k| { let b = breakers.get(*k); b.reserved() != 0 || b.live() != 0 });
        eprintln!("process breakers at the end of the suite: {}", breakers.held_summary());
        if serial {
            assert!(!held, "process reservations outlived the suite: {}", breakers.held_summary());
        }
    }
}
