//! `elasticsearch-keystore`: manage `$ES_PATH_CONF/elasticsearch.keystore`
//! (ES's keystore tool: `create`, `list`, `add`, `add-file`, `remove`,
//! `show`, `upgrade`, `passwd`, `has-passwd`). See
//! `elasticsearch_node::keystore::cli`.

fn main() {
    elasticsearch_node::keystore::cli::main();
}
