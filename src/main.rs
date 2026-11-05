//! `elasticsearch`: an Elasticsearch 8.x-compatible server.
//!
//! Single-node with no configuration; multi-node when
//! `discovery.seed_hosts` / `cluster.initial_master_nodes` are set.
//! Settings: `$ES_PATH_CONF/elasticsearch.yml`, dotted env vars, `-E k=v`.
//! See `elasticsearch_node::cli`.

fn main() {
    elasticsearch_node::cli::main();
}
