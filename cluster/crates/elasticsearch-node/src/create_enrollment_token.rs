//! `elasticsearch-create-enrollment-token`, as ES's
//! `CreateEnrollmentTokenTool` on `BaseRunAsSuperuserCommand` with
//! `ExternalEnrollmentTokenGenerator`: as a temporary file-realm superuser
//! (see `reset_password`), create an API key whose only privilege is
//! `cluster:admin/xpack/security/enroll/{node,kibana}` (expiring in 30
//! minutes), read the node's HTTP addresses from `GET /_nodes/_local/http`,
//! and print the token (the key, the addresses and the fingerprint of the
//! node's HTTP CA).
//!
//! Options: `-s/--scope node|kibana` (required), `--url`, `-f/--force`,
//! `-E key=value`, `-v/--verbose`, `--silent`.

use crate::config::{self, get_str, Settings};
use crate::enrollment::{self, EnrollmentToken};
use crate::reset_password::{
    add_file_superuser, call, check_cluster_health, create_url, default_url, ensure_file_realm_enabled, exit_codes, http_client, remove_file_user, CliError,
    StdTerminal, Terminal,
};
use crate::security_autoconfig::{generate_password, random_alphanumeric};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// ES `ExitCodes.CANT_CREATE`.
pub const CANT_CREATE: i32 = 73;
/// ES `CreateEnrollmentTokenTool.ALLOWED_SCOPES`.
pub const ALLOWED_SCOPES: [&str; 2] = ["node", "kibana"];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    pub scope: String,
    pub force: bool,
    pub url: Option<String>,
    /// `-E key=value` arguments, as `-Ekey=value`.
    pub settings_args: Vec<String>,
    pub verbose: bool,
    pub silent: bool,
    pub help: bool,
}

pub const USAGE: &str = "Creates enrollment tokens for elasticsearch nodes and kibana instances\n\n\
Option               Description\n\
------               -----------\n\
-E <KeySetting>      Configure a setting\n\
-f, --force          Use this option to force execution of the command against a cluster that is currently unhealthy.\n\
-h, --help           Show help\n\
-s, --scope          The scope of this enrollment token, can be either \"node\" or \"kibana\"\n\
--silent             Show minimal output\n\
--url                the URL where the elasticsearch node listens for connections.\n\
-v, --verbose        Show verbose output";

/// Parse the tool's arguments (joptsimple forms: `-s x`, `-sx`, `--scope x`,
/// `--scope=x`).
pub fn parse_args(args: &[String]) -> Result<Options, CliError> {
    let mut o = Options::default();
    let mut scope = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut value = |name: &str| -> Result<String, CliError> {
            i += 1;
            args.get(i).cloned().ok_or_else(|| CliError::new(exit_codes::USAGE, format!("Option {name} requires an argument")))
        };
        match a {
            "-f" | "--force" => o.force = true,
            "-v" | "--verbose" => o.verbose = true,
            "--silent" => o.silent = true,
            "-h" | "--help" => o.help = true,
            "-s" | "--scope" => scope = Some(value("[s/scope]")?),
            "--url" => o.url = Some(value("[url]")?),
            "-E" => o.settings_args.push(format!("-E{}", value("[E]")?)),
            _ => {
                if let Some(v) = a.strip_prefix("--scope=") {
                    scope = Some(v.to_string());
                } else if let Some(v) = a.strip_prefix("--url=") {
                    o.url = Some(v.to_string());
                } else if let Some(v) = a.strip_prefix("-E") {
                    o.settings_args.push(format!("-E{v}"));
                } else if let Some(v) = a.strip_prefix("-s").filter(|v| !v.is_empty() && !a.starts_with("--")) {
                    scope = Some(v.to_string());
                } else {
                    let name = a.trim_start_matches('-');
                    return Err(CliError::new(exit_codes::USAGE, format!("{name} is not a recognized option")));
                }
            }
        }
        i += 1;
    }
    if o.help {
        return Ok(o);
    }
    o.scope = scope.ok_or_else(|| CliError::new(exit_codes::USAGE, "Missing required option(s) [s/scope]"))?;
    Ok(o)
}

/// ES `CreateEnrollmentTokenTool.validate`.
fn validate(o: &Options, settings: &Settings, term: &mut dyn Terminal) -> Result<(), CliError> {
    if !enrollment::enrollment_enabled(settings) {
        return Err(CliError::new(exit_codes::CONFIG, "[xpack.security.enrollment.enabled] must be set to `true` to create an enrollment token"));
    }
    if !ALLOWED_SCOPES.contains(&o.scope.as_str()) {
        term.eprintln(&format!("The scope of this enrollment token, can only be one of [{}]", ALLOWED_SCOPES.join(", ")));
        return Err(CliError::new(exit_codes::USAGE, "Invalid scope"));
    }
    Ok(())
}

/// ES `ExternalEnrollmentTokenGenerator.getBoundAddresses`: the local
/// node's `http.bound_address` and the IP of its `http.publish_address`.
pub fn bound_addresses(nodes_info: &Value) -> Vec<String> {
    let Some(http) = nodes_info["nodes"].as_object().and_then(|n| n.values().next()).map(|n| &n["http"]) else { return Vec::new() };
    let mut out: Vec<String> = http["bound_address"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect();
    if let Some(p) = http["publish_address"].as_str() {
        out.push(enrollment::ip_from_publish_address(p));
    }
    out
}

/// ES `ExternalEnrollmentTokenGenerator.create`.
async fn create_token(settings: &Settings, client: &reqwest::Client, base: &str, user: &str, password: &str, action: &str) -> Result<EnrollmentToken, String> {
    if !enrollment::enrollment_enabled(settings) {
        return Err("[xpack.security.enrollment.enabled] must be set to `true` to create an enrollment token".into());
    }
    let fingerprint = enrollment::https_ca_fingerprint(settings)?;
    let key_url = create_url(base, "/_security/api_key", "");
    let body = json!({
        "name": format!("enrollment_token_API_key_{}", elasticsearch_engine::security::hash::random_b64url(15)),
        "expiration": enrollment::ENROLL_API_KEY_EXPIRATION,
        "role_descriptors": {"create_enrollment_token": {"cluster": [action]}},
    });
    let (status, key) = call(client, reqwest::Method::POST, &key_url, user, password, Some(body)).await?;
    if status != 200 {
        tracing::error!("Error {status}when calling GET {key_url}. ResponseBody: {key}");
        return Err(format!("Unexpected response code [{status}] from calling POST {key_url}"));
    }
    let text = |v: &Value| match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let (api_key, id) = (text(&key["api_key"]), text(&key["id"]));
    if api_key.is_empty() || id.is_empty() {
        return Err("Could not create an api key.".into());
    }
    let info_url = create_url(base, "/_nodes/_local/http", "");
    let (status, info) = call(client, reqwest::Method::GET, &info_url, user, password, None).await?;
    if status != 200 {
        return Err(format!("Unexpected response code [{status}] from calling GET {info_url}"));
    }
    let addresses = bound_addresses(&info);
    if addresses.is_empty() {
        return Err(format!("No bound addresses found in response from calling GET {info_url}"));
    }
    let filtered = enrollment::filtered_addresses(&addresses);
    let version = info["nodes"].as_object().and_then(|n| n.values().next()).map(|n| text(&n["version"])).unwrap_or_default();
    if version.is_empty() {
        return Err("Could not retrieve the version.".into());
    }
    Ok(EnrollmentToken::new(format!("{id}:{api_key}"), fingerprint, filtered))
}

/// Run the tool (`env` holds `ES_PATH_CONF`); prints the token.
pub async fn run(o: &Options, env: &BTreeMap<String, String>, term: &mut dyn Terminal) -> Result<(), CliError> {
    let settings = config::load(&o.settings_args, env).map_err(|e| CliError::new(exit_codes::CONFIG, e.0))?;
    validate(o, &settings, term)?;
    ensure_file_realm_enabled(&settings)?;
    let conf = PathBuf::from(get_str(&settings, "path.conf").unwrap_or("config"));
    let base = match &o.url {
        Some(u) => u.clone(),
        None => default_url(&settings)?,
    };
    let client = http_client(&settings).map_err(|e| CliError::new(exit_codes::CONFIG, e))?;
    let hasher = elasticsearch_engine::security::hash::Hasher::resolve(get_str(&settings, "xpack.security.authc.password_hashing.algorithm").unwrap_or("bcrypt"))
        .map_err(|e| CliError::new(exit_codes::CONFIG, e))?;
    let username = format!("autogenerated_{}", random_alphanumeric(8));
    let password = generate_password(14);
    let action = if o.scope == "node" { enrollment::NODE_ENROLLMENT_ACTION } else { enrollment::KIBANA_ENROLLMENT_ACTION };
    let outcome = async {
        add_file_superuser(&conf, &username, &hasher.hash(&password))?;
        check_cluster_health(&client, &base, &username, &password, 5, o.force, o.verbose, term).await?;
        match create_token(&settings, &client, &base, &username, &password, action).await {
            Ok(token) => {
                term.print(&format!("{}\n", token.encoded()));
                Ok(())
            }
            Err(e) => {
                term.eprintln(&format!("Unable to create enrollment token for scope [{}]", o.scope));
                Err(CliError::new(CANT_CREATE, e))
            }
        }
    }
    .await;
    let cleanup = remove_file_user(&conf, &username);
    outcome?;
    cleanup
}

/// `elasticsearch-create-enrollment-token` entry point.
pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env: BTreeMap<String, String> = std::env::vars().collect();
    let mut term = StdTerminal;
    let fail = |e: CliError| {
        if e.code == exit_codes::USAGE {
            eprintln!("{USAGE}");
        }
        eprintln!();
        eprintln!("ERROR: {e}");
        e.code
    };
    let code = match parse_args(&args) {
        Ok(o) if o.help => {
            println!("{USAGE}");
            0
        }
        Ok(o) => {
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio runtime");
            match rt.block_on(run(&o, &env, &mut term)) {
                Ok(()) => 0,
                Err(e) => fail(e),
            }
        }
        Err(e) => fail(e),
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn options_follow_es() {
        let o = parse_args(&args(&["-s", "node", "--url", "https://h:9200", "-f", "-Ea=b"])).unwrap();
        assert_eq!((o.scope.as_str(), o.url.as_deref(), o.force), ("node", Some("https://h:9200"), true));
        assert_eq!(o.settings_args, vec!["-Ea=b"]);
        assert_eq!(parse_args(&args(&["--scope=kibana"])).unwrap().scope, "kibana");
        assert_eq!(parse_args(&args(&["-skibana"])).unwrap().scope, "kibana");
        let e = parse_args(&args(&["-f"])).unwrap_err();
        assert_eq!((e.code, e.message.as_str()), (64, "Missing required option(s) [s/scope]"));
        assert_eq!(parse_args(&args(&["-s", "node", "--nope"])).unwrap_err().code, 64);
    }

    struct Quiet(String);
    impl Terminal for Quiet {
        fn print(&mut self, text: &str) {
            self.0.push_str(text);
        }
        fn eprintln(&mut self, text: &str) {
            self.0.push_str(text);
            self.0.push('\n');
        }
        fn read_line(&mut self, _: &str) -> Option<String> {
            None
        }
        fn read_secret(&mut self, _: &str) -> Option<String> {
            None
        }
    }

    #[test]
    fn validation_follows_es() {
        let mut s = Settings::new();
        let mut t = Quiet(String::new());
        let o = parse_args(&args(&["-s", "node"])).unwrap();
        assert_eq!(validate(&o, &s, &mut t).unwrap_err(), CliError::new(78, "[xpack.security.enrollment.enabled] must be set to `true` to create an enrollment token"));
        s.insert("xpack.security.enrollment.enabled".into(), json!("true"));
        assert!(validate(&o, &s, &mut t).is_ok());
        let o = parse_args(&args(&["-s", "beats"])).unwrap();
        assert_eq!(validate(&o, &s, &mut t).unwrap_err(), CliError::new(64, "Invalid scope"));
        assert_eq!(t.0, "The scope of this enrollment token, can only be one of [node, kibana]\n");
    }

    #[test]
    fn bound_addresses_follow_es() {
        let info = json!({"nodes": {"x": {"version": "9.5.5", "http": {"bound_address": ["[::1]:9200", "127.0.0.1:9200"], "publish_address": "host/10.0.0.3:9200"}}}});
        assert_eq!(bound_addresses(&info), vec!["[::1]:9200", "127.0.0.1:9200", "10.0.0.3:9200"]);
        assert_eq!(enrollment::filtered_addresses(&bound_addresses(&info)), vec!["10.0.0.3:9200"]);
        assert!(bound_addresses(&json!({"nodes": {}})).is_empty());
    }
}
