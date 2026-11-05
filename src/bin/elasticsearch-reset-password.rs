//! `elasticsearch-reset-password`: reset the password of a native or
//! built-in user on a running node (ES `ResetPasswordTool`). See
//! `elasticsearch_node::reset_password`.

fn main() {
    elasticsearch_node::reset_password::main();
}
