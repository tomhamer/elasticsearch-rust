//! `elasticsearch-reset-password`, as ES's `ResetPasswordTool` on
//! `BaseRunAsSuperuserCommand`: create a temporary file-realm superuser in
//! `$ES_PATH_CONF/users` + `users_roles`, check the cluster health as that
//! user (retrying while the node reloads its file realm), reset the user's
//! password through `POST /_security/user/<user>/_password`, and remove the
//! temporary user again.
//!
//! Options: `-u/--username` (required), `-a/--auto` (default: a generated
//! password, printed), `-i/--interactive` (prompted), `-b/--batch` (no
//! confirmation), `-f/--force` (run against a RED cluster), `--url`,
//! `-E key=value`, `-v/--verbose`, `-s/--silent`. HTTPS trusts the node's
//! HTTP certificate chain (`xpack.security.http.ssl.certificate`) and
//! `xpack.security.http.ssl.certificate_authorities`, as ES's CLI trusts the
//! HTTP SSL profile.

use crate::config::{self, get_list, get_str, Settings};
use crate::security_autoconfig::{generate_password, random_alphanumeric};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// ES `ExitCodes`.
pub mod exit_codes {
    pub const OK: i32 = 0;
    pub const USAGE: i32 = 64;
    pub const DATA_ERROR: i32 = 65;
    pub const UNAVAILABLE: i32 = 69;
    pub const TEMP_FAILURE: i32 = 75;
    pub const CONFIG: i32 = 78;
}

/// ES `UserException`: printed as `ERROR: <message>, with exit code <code>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError {
    pub code: i32,
    pub message: String,
}

impl CliError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        CliError { code, message: message.into() }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, with exit code {}", self.message, self.code)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    pub username: String,
    pub interactive: bool,
    pub auto: bool,
    pub batch: bool,
    pub force: bool,
    pub url: Option<String>,
    /// `-E key=value` arguments, as `-Ekey=value`.
    pub settings_args: Vec<String>,
    pub verbose: bool,
    pub silent: bool,
    pub help: bool,
}

/// Parse the tool's arguments (joptsimple forms: `-u x`, `-ux`,
/// `--username x`, `--username=x`).
pub fn parse_args(args: &[String]) -> Result<Options, CliError> {
    let mut o = Options::default();
    let mut username = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut value = |name: &str, inline: Option<&str>| -> Result<String, CliError> {
            if let Some(v) = inline {
                return Ok(v.to_string());
            }
            i += 1;
            args.get(i).cloned().ok_or_else(|| CliError::new(exit_codes::USAGE, format!("Option {name} requires an argument")))
        };
        match a {
            "-i" | "--interactive" => o.interactive = true,
            "-a" | "--auto" => o.auto = true,
            "-b" | "--batch" => o.batch = true,
            "-f" | "--force" => o.force = true,
            "-v" | "--verbose" => o.verbose = true,
            "-s" | "--silent" => o.silent = true,
            "-h" | "--help" => o.help = true,
            "-u" | "--username" => username = Some(value("[u/username]", None)?),
            "--url" => o.url = Some(value("[url]", None)?),
            "-E" => o.settings_args.push(format!("-E{}", value("[E]", None)?)),
            _ => {
                if let Some(v) = a.strip_prefix("--username=") {
                    username = Some(v.to_string());
                } else if let Some(v) = a.strip_prefix("--url=") {
                    o.url = Some(v.to_string());
                } else if let Some(v) = a.strip_prefix("-E") {
                    o.settings_args.push(format!("-E{v}"));
                } else if let Some(v) = a.strip_prefix("-u").filter(|v| !v.is_empty() && !a.starts_with("--")) {
                    username = Some(v.to_string());
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
    o.username = username.ok_or_else(|| CliError::new(exit_codes::USAGE, "Missing required option(s) [u/username]"))?;
    if o.interactive && o.auto {
        return Err(CliError::new(exit_codes::USAGE, "You can only run the tool in one of [auto] or [interactive] modes"));
    }
    Ok(o)
}

pub const USAGE: &str = "Resets the password of users in the native realm and built-in users.\n\n\
Option             Description\n\
------             -----------\n\
-E <KeySetting>    Configure a setting\n\
-a, --auto\n\
-b, --batch\n\
-f, --force        Use this option to force execution of the command against a cluster that is currently unhealthy.\n\
-h, --help         Show help\n\
-i, --interactive\n\
-s, --silent       Show minimal output\n\
-u, --username     The username of the user whose password will be reset\n\
--url              the URL where the elasticsearch node listens for connections.\n\
-v, --verbose      Show verbose output";

/// The console the tool talks to (stdout/stderr/stdin, or a script in tests).
pub trait Terminal {
    fn print(&mut self, text: &str);
    fn eprintln(&mut self, text: &str);
    /// Read a line after printing `prompt` (`None` at end of input).
    fn read_line(&mut self, prompt: &str) -> Option<String>;
    /// Read a secret (no echo) after printing `prompt`.
    fn read_secret(&mut self, prompt: &str) -> Option<String>;
}

/// ES `Terminal.promptYesNo(prompt, false)`.
fn prompt_yes_no(term: &mut dyn Terminal, prompt: &str) -> bool {
    loop {
        let Some(answer) = term.read_line(&format!("{prompt} [y/N]")) else { return false };
        let answer = answer.trim().to_lowercase();
        if answer.is_empty() {
            return false;
        }
        if answer == "y" {
            return true;
        }
        if answer == "n" {
            return false;
        }
        term.eprintln(&format!("Did not understand answer '{answer}'"));
    }
}

/// ES `CommandLineHttpClient.getDefaultURL`.
pub fn default_url(settings: &Settings) -> Result<String, CliError> {
    let fail = |e: String| CliError::new(exit_codes::DATA_ERROR, format!("unable to determine default URL from settings, please use the -u option to explicitly provide the url ({e})"));
    let scheme = if get_str(settings, "xpack.security.http.ssl.enabled") == Some("true") { "https" } else { "http" };
    let mut hosts = get_list(settings, "http.publish_host");
    if hosts.is_empty() {
        hosts = get_list(settings, "network.publish_host");
    }
    let host = hosts.first().map(String::as_str).unwrap_or("_local_");
    let ip = config::resolve_host(host).map_err(|e| fail(e.0))?[0];
    let mut port: i64 = get_str(settings, "http.publish_port").and_then(|p| p.parse().ok()).unwrap_or(-1);
    if port <= 0 {
        let ports = config::parse_port_range(get_str(settings, "http.port").unwrap_or("9200-9300")).map_err(|e| fail(e.0))?;
        port = ports.first().map(|p| *p as i64).unwrap_or(0);
        if port <= 0 {
            return Err(fail("unable to determine http port from settings".into()));
        }
    }
    let host = match ip {
        std::net::IpAddr::V6(v6) => format!("[{v6}]"),
        v4 => v4.to_string(),
    };
    Ok(format!("{scheme}://{host}:{port}"))
}

/// ES `CommandLineHttpClient.createURL`: `base` + `path` (slashes collapsed) + `query`.
pub fn create_url(base: &str, path: &str, query: &str) -> String {
    let (origin, base_path) = match base.find("://").map(|i| i + 3) {
        Some(start) => match base[start..].find('/') {
            Some(slash) => (&base[..start + slash], &base[start + slash..]),
            None => (base, ""),
        },
        None => (base, ""),
    };
    let mut joined = format!("{base_path}/{path}");
    while joined.contains("//") {
        joined = joined.replace("//", "/");
    }
    format!("{origin}{joined}{query}")
}

/// ES `CommandLineHttpClient.getErrorCause`.
pub fn error_cause(body: &Value) -> Option<String> {
    let error = body.get("error")?;
    match error {
        Value::Object(e) => {
            if let Some(reason) = e.get("reason").filter(|r| !r.is_null()) {
                return Some(reason.as_str().map(str::to_string).unwrap_or_else(|| reason.to_string()));
            }
            if let Some(Value::Object(root)) = e.get("root_cause") {
                if let Some(reason) = root.get("reason").filter(|r| !r.is_null()) {
                    return Some(reason.as_str().map(str::to_string).unwrap_or_else(|| reason.to_string()));
                }
                if let Some(t) = root.get("type").and_then(Value::as_str) {
                    return Some(t.to_string());
                }
            }
            Some(e.get("type").and_then(Value::as_str).unwrap_or("null").to_string())
        }
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

fn conf_path(settings: &Settings, p: &str) -> PathBuf {
    let p = PathBuf::from(p);
    if p.is_absolute() { p } else { PathBuf::from(get_str(settings, "path.conf").unwrap_or("config")).join(p) }
}

/// The HTTP client of the tool: trusts the node's HTTP chain and CAs.
pub fn http_client(settings: &Settings) -> Result<reqwest::Client, String> {
    let mut b = reqwest::Client::builder().timeout(std::time::Duration::from_secs(35));
    let mut pem_files: Vec<PathBuf> = get_list(settings, "xpack.security.http.ssl.certificate_authorities").iter().map(|p| conf_path(settings, p)).collect();
    if let Some(cert) = get_str(settings, "xpack.security.http.ssl.certificate") {
        pem_files.push(conf_path(settings, cert));
    }
    for f in pem_files {
        let Ok(bytes) = std::fs::read(&f) else { continue };
        for cert in reqwest::Certificate::from_pem_bundle(&bytes).map_err(|e| format!("cannot read certificates from [{}]: {e}", f.display()))? {
            b = b.add_root_certificate(cert);
        }
    }
    match get_str(settings, "xpack.security.http.ssl.verification_mode") {
        Some("none") => b = b.danger_accept_invalid_certs(true),
        Some("certificate") => b = b.danger_accept_invalid_hostnames(true),
        _ => {}
    }
    b.build().map_err(|e| e.to_string())
}

// ------------------------------------------------------- file realm files

fn read_lines(path: &Path) -> Result<Vec<String>, CliError> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(t.lines().map(str::to_string).collect()),
        // ES's distribution ships empty `users` / `users_roles` files; a
        // missing one is treated as empty and created.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(CliError::new(exit_codes::DATA_ERROR, format!("File realm configuration file [{}] cannot be read: {e}", path.display()))),
    }
}

fn write_lines(path: &Path, lines: &[String]) -> Result<(), CliError> {
    let mut text = lines.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    let io = if path.exists() {
        crate::security_autoconfig::replace_file(path, text.as_bytes())
    } else {
        std::fs::write(path, text.as_bytes()).and_then(|_| {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o660))?;
            }
            Ok(())
        })
    };
    io.map_err(|e| CliError::new(exit_codes::DATA_ERROR, format!("Failed to write file realm configuration file [{}]: {e}", path.display())))
}

fn is_entry(line: &str) -> bool {
    let t = line.trim();
    !t.is_empty() && !t.starts_with('#')
}

/// Add `user` (with `hash`) as a superuser: `users_roles` first, then `users`.
pub fn add_file_superuser(conf: &Path, user: &str, hash: &str) -> Result<(), CliError> {
    let roles_file = conf.join("users_roles");
    let mut roles = read_lines(&roles_file)?;
    match roles.iter_mut().find(|l| is_entry(l) && l.split_once(':').is_some_and(|(r, _)| r.trim() == "superuser")) {
        Some(line) => {
            let users = line.split_once(':').map(|(_, u)| u.trim().to_string()).unwrap_or_default();
            *line = if users.is_empty() { format!("superuser:{user}") } else { format!("superuser:{users},{user}") };
        }
        None => roles.push(format!("superuser:{user}")),
    }
    write_lines(&roles_file, &roles)?;
    let users_file = conf.join("users");
    let mut users = read_lines(&users_file)?;
    users.retain(|l| !(is_entry(l) && l.split_once(':').is_some_and(|(u, _)| u.trim() == user)));
    users.push(format!("{user}:{hash}"));
    write_lines(&users_file, &users)
}

/// Remove `user` from `users` and `users_roles` (ES `cleanup`).
pub fn remove_file_user(conf: &Path, user: &str) -> Result<(), CliError> {
    let mut errors = Vec::new();
    let users_file = conf.join("users");
    match read_lines(&users_file) {
        Ok(mut users) => {
            let before = users.len();
            users.retain(|l| !(is_entry(l) && l.split_once(':').is_some_and(|(u, _)| u.trim() == user)));
            if users.len() != before {
                if let Err(e) = write_lines(&users_file, &users) {
                    errors.push(e.message);
                }
            }
        }
        Err(e) => errors.push(e.message),
    }
    let roles_file = conf.join("users_roles");
    match read_lines(&roles_file) {
        Ok(roles) => {
            let mut changed = false;
            let mut out = Vec::with_capacity(roles.len());
            for line in roles {
                let parsed: Option<(String, Vec<String>)> = if is_entry(&line) {
                    line.split_once(':').map(|(r, u)| (r.trim().to_string(), u.split(',').map(|x| x.trim().to_string()).collect()))
                } else {
                    None
                };
                match parsed {
                    Some((role, users)) if users.iter().any(|u| u == user) => {
                        changed = true;
                        let rest: Vec<String> = users.into_iter().filter(|u| !u.is_empty() && u != user).collect();
                        if !rest.is_empty() {
                            out.push(format!("{role}:{}", rest.join(",")));
                        }
                    }
                    _ => out.push(line),
                }
            }
            if changed {
                if let Err(e) = write_lines(&roles_file, &out) {
                    errors.push(e.message);
                }
            }
        }
        Err(e) => errors.push(e.message),
    }
    if errors.is_empty() { Ok(()) } else { Err(CliError::new(exit_codes::CONFIG, errors.join(" , "))) }
}

/// ES `ensureFileRealmEnabled`: a single configured file realm must not be disabled.
pub fn ensure_file_realm_enabled(settings: &Settings) -> Result<(), CliError> {
    let names: std::collections::BTreeSet<&str> = settings.keys()
        .filter_map(|k| k.strip_prefix("xpack.security.authc.realms.file.")).filter_map(|r| r.split('.').next()).collect();
    if names.len() == 1 {
        let name = names.iter().next().expect("one realm");
        if get_str(settings, &format!("xpack.security.authc.realms.file.{name}.enabled")) == Some("false") {
            return Err(CliError::new(exit_codes::CONFIG, "File realm must be enabled"));
        }
    }
    Ok(())
}

pub(crate) async fn call(client: &reqwest::Client, method: reqwest::Method, url: &str, user: &str, password: &str, body: Option<Value>) -> Result<(u16, Value), String> {
    let mut req = client.request(method, url).basic_auth(user, Some(password)).header("Content-Type", "application/json");
    if let Some(b) = body {
        req = req.body(b.to_string());
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    Ok((status, serde_json::from_str(&text).unwrap_or(Value::Null)))
}

/// ES `checkClusterHealthWithRetries`.
pub(crate) async fn check_cluster_health(client: &reqwest::Client, base: &str, user: &str, password: &str, mut retries: u32, force: bool, verbose: bool, term: &mut dyn Terminal) -> Result<(), CliError> {
    let url = create_url(base, "_cluster/health", "?pretty");
    loop {
        let (status, body) = call(client, reqwest::Method::GET, &url, user, password, None).await
            .map_err(|e| CliError::new(exit_codes::UNAVAILABLE, format!("Failed to determine the health of the cluster. {e}")))?;
        if status != 200 {
            if (status == 401 || status == 403) && retries > 0 {
                if verbose {
                    term.print(&format!("Unexpected http status [{status}] while attempting to determine cluster health. Will retry at most {retries} more times.\n"));
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                retries -= 1;
                continue;
            }
            return Err(CliError::new(exit_codes::DATA_ERROR, format!("Failed to determine the health of the cluster. Unexpected http status [{status}]")));
        }
        let health = body.get("status").and_then(Value::as_str).unwrap_or("");
        if health.is_empty() {
            return Err(CliError::new(exit_codes::DATA_ERROR, "Failed to determine the health of the cluster. Cluster health API did not return a status value."));
        }
        if health.eq_ignore_ascii_case("red") && !force {
            term.eprintln("Failed to determine the health of the cluster. Cluster health is currently RED.");
            term.eprintln("This means that some cluster data is unavailable and your cluster is not fully functional.");
            term.eprintln("The cluster logs (https://www.elastic.co/docs/deploy-manage/monitor/logging-configuration/update-elasticsearch-logging?version=9.5) might contain information/indications for the underlying cause");
            term.eprintln("It is recommended that you resolve the issues with your cluster before continuing");
            term.eprintln("It is very likely that the command will fail when run against an unhealthy cluster.");
            term.eprintln("");
            term.eprintln("If you still want to attempt to execute this command against an unhealthy cluster, you can pass the `-f` parameter.");
            return Err(CliError::new(exit_codes::UNAVAILABLE, "Failed to determine the health of the cluster. Cluster health is currently RED."));
        }
        return Ok(());
    }
}

/// ES `Validation.Users.validatePassword`.
fn validate_password(p: &str) -> Option<&'static str> {
    if p.chars().count() < 6 { Some("passwords must be at least [6] characters long") } else { None }
}

fn prompt_for_password(term: &mut dyn Terminal, user: &str) -> Result<String, CliError> {
    loop {
        let Some(p1) = term.read_secret(&format!("Enter password for [{user}]: ")) else {
            return Err(CliError::new(exit_codes::OK, "User cancelled operation"));
        };
        if let Some(err) = validate_password(&p1) {
            term.eprintln(err);
            term.eprintln("Try again.");
            continue;
        }
        let Some(p2) = term.read_secret(&format!("Re-enter password for [{user}]: ")) else {
            return Err(CliError::new(exit_codes::OK, "User cancelled operation"));
        };
        if p1 != p2 {
            term.eprintln("Passwords do not match.");
            term.eprintln("Try again.");
            continue;
        }
        return Ok(p1);
    }
}

/// ES `ResetPasswordTool.executeCommand`.
async fn execute(o: &Options, settings: &Settings, client: &reqwest::Client, base: &str, user: &str, password: &str, term: &mut dyn Terminal) -> Result<(), CliError> {
    let target = &o.username;
    let new_password = if o.interactive {
        if !o.batch {
            if !o.silent {
                term.print(&format!("This tool will reset the password of the [{target}] user.\nYou will be prompted to enter the password.\n"));
            }
            let go = prompt_yes_no(term, "Please confirm that you would like to continue");
            term.print("\n\n");
            if !go {
                return Err(CliError::new(exit_codes::OK, "User cancelled operation"));
            }
        }
        prompt_for_password(term, target)?
    } else {
        if !o.batch {
            if !o.silent {
                term.print(&format!("This tool will reset the password of the [{target}] user to an autogenerated value.\nThe password will be printed in the console.\n"));
            }
            let go = prompt_yes_no(term, "Please confirm that you would like to continue");
            term.print("\n\n");
            if !go {
                return Err(CliError::new(exit_codes::OK, "User cancelled operation"));
            }
        }
        generate_password(20)
    };
    let _ = settings;
    let url = create_url(base, &format!("_security/user/{target}/_password"), "?pretty");
    let failed = || CliError::new(exit_codes::TEMP_FAILURE, format!("Failed to reset password for the [{target}] user"));
    let (status, body) = call(client, reqwest::Method::POST, &url, user, password, Some(json!({"password": new_password}))).await.map_err(|_| failed())?;
    if status != 200 {
        // ES wraps this in "Failed to reset password for the [user] user";
        // the cause is shown with -v.
        if o.verbose {
            let mut message = format!("Failed to reset password for the [{target}] user. Unexpected http status [{status}].");
            if let Some(cause) = error_cause(&body) {
                message.push_str(&format!(" Cause was {cause}"));
            }
            term.eprintln(&message);
        }
        return Err(failed());
    }
    if !o.silent {
        term.print(&format!("Password for the [{target}] user successfully reset.\n"));
    }
    if !o.interactive {
        if !o.silent {
            term.print("New value: ");
        }
        term.print(&format!("{new_password}\n"));
    }
    Ok(())
}

/// Run the tool (`args` without the program name; `env` the process
/// environment, which holds `ES_PATH_CONF`).
pub async fn run(o: &Options, env: &BTreeMap<String, String>, term: &mut dyn Terminal) -> Result<(), CliError> {
    let settings = config::load(&o.settings_args, env).map_err(|e| CliError::new(exit_codes::CONFIG, e.0))?;
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
    let outcome = async {
        add_file_superuser(&conf, &username, &hasher.hash(&password))?;
        check_cluster_health(&client, &base, &username, &password, 5, o.force, o.verbose, term).await?;
        execute(o, &settings, &client, &base, &username, &password, term).await
    }.await;
    let cleanup = remove_file_user(&conf, &username);
    outcome?;
    cleanup
}

/// The real console.
pub struct StdTerminal;

impl Terminal for StdTerminal {
    fn print(&mut self, text: &str) {
        use std::io::Write;
        let mut out = std::io::stdout();
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }

    fn eprintln(&mut self, text: &str) {
        eprintln!("{text}");
    }

    fn read_line(&mut self, prompt: &str) -> Option<String> {
        self.print(prompt);
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_string()),
        }
    }

    fn read_secret(&mut self, prompt: &str) -> Option<String> {
        self.print(prompt);
        #[cfg(unix)]
        let saved = {
            // SAFETY: termios of stdin, restored below.
            unsafe {
                let mut t: libc::termios = std::mem::zeroed();
                if libc::isatty(libc::STDIN_FILENO) == 1 && libc::tcgetattr(libc::STDIN_FILENO, &mut t) == 0 {
                    let saved = t;
                    t.c_lflag &= !libc::ECHO;
                    libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t);
                    Some(saved)
                } else {
                    None
                }
            }
        };
        let mut line = String::new();
        let read = std::io::stdin().read_line(&mut line);
        #[cfg(unix)]
        {
            if let Some(saved) = saved {
                // SAFETY: restores the attributes read above.
                unsafe {
                    libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &saved);
                }
                self.print("\n");
            }
        }
        match read {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_string()),
        }
    }
}

/// `elasticsearch-reset-password` entry point.
pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let env: BTreeMap<String, String> = std::env::vars().collect();
    let mut term = StdTerminal;
    let code = match parse_args(&args) {
        Ok(o) if o.help => {
            println!("{USAGE}");
            0
        }
        Ok(o) => {
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio runtime");
            match rt.block_on(run(&o, &env, &mut term)) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!();
                    eprintln!("ERROR: {e}");
                    e.code
                }
            }
        }
        Err(e) => {
            if e.code == exit_codes::USAGE {
                eprintln!("{USAGE}");
            }
            eprintln!();
            eprintln!("ERROR: {e}");
            e.code
        }
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
        let o = parse_args(&args(&["-u", "elastic", "-b", "--url", "https://h:9200", "-E", "a.b=c", "-Ex.y=z"])).unwrap();
        assert_eq!((o.username.as_str(), o.batch, o.interactive, o.url.as_deref()), ("elastic", true, false, Some("https://h:9200")));
        assert_eq!(o.settings_args, vec!["-Ea.b=c", "-Ex.y=z"]);
        assert_eq!(parse_args(&args(&["--username=kibana_system", "-i"])).unwrap().username, "kibana_system");
        assert_eq!(parse_args(&args(&["-uelastic"])).unwrap().username, "elastic");
        let e = parse_args(&args(&["-u", "elastic", "-i", "-a"])).unwrap_err();
        assert_eq!(e, CliError::new(64, "You can only run the tool in one of [auto] or [interactive] modes"));
        assert_eq!(parse_args(&args(&["-b"])).unwrap_err().message, "Missing required option(s) [u/username]");
        assert_eq!(parse_args(&args(&["-u", "e", "--nope"])).unwrap_err().code, 64);
        assert_eq!(e.to_string(), "You can only run the tool in one of [auto] or [interactive] modes, with exit code 64");
    }

    #[test]
    fn default_url_and_create_url() {
        let s = |pairs: &[(&str, &str)]| -> Settings { pairs.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect() };
        assert_eq!(default_url(&s(&[])).unwrap(), "http://127.0.0.1:9200");
        assert_eq!(default_url(&s(&[("xpack.security.http.ssl.enabled", "true"), ("http.port", "9201-9300")])).unwrap(), "https://127.0.0.1:9201");
        assert_eq!(default_url(&s(&[("http.publish_port", "9443"), ("http.publish_host", "10.1.2.3")])).unwrap(), "http://10.1.2.3:9443");
        assert_eq!(create_url("https://h:9200", "_security/user/elastic/_password", "?pretty"), "https://h:9200/_security/user/elastic/_password?pretty");
        assert_eq!(create_url("https://h:9200/base/", "/_cluster/health", "?pretty"), "https://h:9200/base/_cluster/health?pretty");
    }

    #[test]
    fn error_cause_follows_es() {
        assert_eq!(error_cause(&json!({"error": {"reason": "nope", "type": "x"}})).as_deref(), Some("nope"));
        assert_eq!(error_cause(&json!({"error": {"type": "security_exception"}})).as_deref(), Some("security_exception"));
        assert_eq!(error_cause(&json!({"error": "plain"})).as_deref(), Some("plain"));
        assert_eq!(error_cause(&json!({"ok": true})), None);
    }

    #[test]
    fn temporary_superuser_is_added_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("users"), "alice:hash\n").unwrap();
        std::fs::write(dir.path().join("users_roles"), "# roles\nsuperuser:alice\nviewer:bob\n").unwrap();
        add_file_superuser(dir.path(), "autogenerated_x", "h").unwrap();
        assert_eq!(std::fs::read_to_string(dir.path().join("users")).unwrap(), "alice:hash\nautogenerated_x:h\n");
        assert_eq!(std::fs::read_to_string(dir.path().join("users_roles")).unwrap(), "# roles\nsuperuser:alice,autogenerated_x\nviewer:bob\n");
        remove_file_user(dir.path(), "autogenerated_x").unwrap();
        assert_eq!(std::fs::read_to_string(dir.path().join("users")).unwrap(), "alice:hash\n");
        assert_eq!(std::fs::read_to_string(dir.path().join("users_roles")).unwrap(), "# roles\nsuperuser:alice\nviewer:bob\n");
        // Missing files are created (and emptied again).
        let empty = tempfile::tempdir().unwrap();
        add_file_superuser(empty.path(), "t", "h").unwrap();
        assert_eq!(std::fs::read_to_string(empty.path().join("users_roles")).unwrap(), "superuser:t\n");
        remove_file_user(empty.path(), "t").unwrap();
        assert_eq!(std::fs::read_to_string(empty.path().join("users")).unwrap(), "");
        assert_eq!(std::fs::read_to_string(empty.path().join("users_roles")).unwrap(), "");
    }

    #[test]
    fn disabled_file_realm_is_refused() {
        let mut s = Settings::new();
        s.insert("xpack.security.authc.realms.file.f1.enabled".into(), Value::String("false".into()));
        assert_eq!(ensure_file_realm_enabled(&s).unwrap_err(), CliError::new(78, "File realm must be enabled"));
        s.insert("xpack.security.authc.realms.file.f1.enabled".into(), Value::String("true".into()));
        assert!(ensure_file_realm_enabled(&s).is_ok());
    }
}
