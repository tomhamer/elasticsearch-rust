//! `elasticsearch-create-enrollment-token`: create an enrollment token for
//! a new node (`-s node`) or a Kibana instance (`-s kibana`) on a running,
//! auto-configured node (ES `CreateEnrollmentTokenTool`). See
//! `elasticsearch_node::create_enrollment_token`.

fn main() {
    elasticsearch_node::create_enrollment_token::main();
}
