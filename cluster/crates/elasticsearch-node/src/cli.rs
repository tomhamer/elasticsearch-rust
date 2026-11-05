//! Entry point shared by the `elasticsearch` binary (root package) and
//! `elasticsearch-node`: the cluster-aware server; single-node by default.
//!
//! Settings come from `$ES_PATH_CONF/elasticsearch.yml`, dotted environment
//! variables and `-E key=value` (see `config.rs`). SIGTERM/SIGINT flush
//! every local shard through the gated path and exit 0.

use crate::config::{self, NodeConfig};
use std::collections::BTreeMap;

/// Parse args/env/config, start the node and block until shutdown.
pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("Version: {}, Build: rust", crate::node::ES_VERSION);
        return;
    }
    let mut env: BTreeMap<String, String> = std::env::vars().collect();
    // ES's launcher exports HOSTNAME (the auto-configured
    // `cluster.initial_master_nodes: ["${HOSTNAME}"]` refers to it).
    env.entry("HOSTNAME".into()).or_insert_with(config::hostname);
    // ES `ServerCli`: `--enrollment-token <token>` (once) enrolls this new
    // node into an existing secured cluster before it starts.
    let enrollment_token = match enrollment_token(&args) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: {e}, with exit code {}", crate::security_autoconfig::exit_codes::USAGE);
            std::process::exit(crate::security_autoconfig::exit_codes::USAGE);
        }
    };
    let load = |args: &[String], env: &BTreeMap<String, String>| {
        let settings = match config::load(args, env) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR: {e}");
                std::process::exit(78);
            }
        };
        // ES `SettingsModule`: an unknown (or index-level) node setting is
        // an IllegalArgumentException while the node is built, which ES
        // reports as "fatal exception while booting Elasticsearch" (exit 1).
        if let Err(e) = crate::settings_validation::validate_node_settings(&settings) {
            eprintln!("fatal exception while booting Elasticsearch: java.lang.IllegalArgumentException: {e}");
            std::process::exit(1);
        }
        settings
    };
    let mut settings = load(&args, &env);
    // ES `ServerCli.autoConfigureSecurity`: first-start security
    // auto-configuration, then the settings are read again.
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    let conf_dir = std::path::PathBuf::from(config::get_str(&settings, "path.conf").unwrap_or("config"));
    let outcome = match &enrollment_token {
        Some(token) => crate::security_autoconfig::enroll_blocking(&settings, &conf_dir, token, verbose),
        None => crate::security_autoconfig::run(&settings, &conf_dir),
    };
    match outcome {
        Ok(crate::security_autoconfig::Outcome::Configured { .. }) => settings = load(&args, &env),
        Ok(crate::security_autoconfig::Outcome::Skipped(skip)) => {
            if !skip.verbose || verbose {
                println!("{}", skip.message);
            }
        }
        // Enrolling, every failure stops the start.
        Err(e) if e.start_anyway() && enrollment_token.is_none() => eprintln!("{}", e.message),
        Err(e) => {
            eprintln!("ERROR: {e}, with exit code {}", e.exit_code);
            std::process::exit(e.exit_code);
        }
    }
    // Secure settings: `$ES_PATH_CONF/elasticsearch.keystore` (see `keystore`).
    // After validation (secure-only keys are known there, and
    // `keystore::reject_insecure` reports them) and auto-configuration.
    if let Err(e) = crate::keystore::load_into_settings(&mut settings, &env) {
        eprintln!("ERROR: {e}");
        std::process::exit(78);
    }
    let cfg = match NodeConfig::from_settings(settings) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(78);
        }
    };
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into());
    // `ES_LOG_FORMAT=json` (or `LOG_FORMAT=json`, as the operator uses, or
    // `--log-format=json`): one JSON object per line for log collectors.
    // Colour only when stdout is a terminal (never under docker/kubectl).
    let json_logs = args.iter().any(|a| a == "--log-format=json")
        || ["ES_LOG_FORMAT", "LOG_FORMAT"].iter().any(|k| env.get(*k).is_some_and(|v| v.eq_ignore_ascii_case("json")));
    if json_logs {
        tracing_subscriber::fmt().with_env_filter(filter).json().flatten_event(true).init();
    } else {
        let ansi = std::io::IsTerminal::is_terminal(&std::io::stdout()) && std::env::var_os("NO_COLOR").is_none();
        tracing_subscriber::fmt().with_env_filter(filter).with_ansi(ansi).init();
    }
    // What the JVM does when it starts (`-XX:+MaxFDLimit`): the soft file
    // descriptor limit goes up to the hard one.
    crate::natives::raise_file_descriptor_limit();
    // ES `Elasticsearch.initializeNatives`: `bootstrap.memory_lock` locks
    // memory before the node is built (an unparsable value is an
    // IllegalArgumentException while booting, exit 1).
    match crate::bootstrap_checks::memory_lock_requested(&cfg.settings) {
        Ok(true) => {
            let locked = crate::natives::try_lock_memory();
            elasticsearch_http::set_memory_locked(locked);
        }
        Ok(false) => {}
        Err(e) => {
            eprintln!("fatal exception while booting Elasticsearch: java.lang.IllegalArgumentException: {e}");
            std::process::exit(1);
        }
    }
    // `node.store.allow_mmap: false` (ES `IndexModule.NODE_STORE_ALLOW_MMAP`):
    // no store file is memory-mapped; files are read into memory instead.
    // Set before any index opens.
    match crate::bootstrap_checks::bool_setting(&cfg.settings, "node.store.allow_mmap", true) {
        Ok(allowed) => {
            // Store files then live in memory, counted on the breakers.
            if !allowed {
                elasticsearch_engine::memory::install_store_file_accounting();
            }
            elasticsearch_engine::rustic_core::codec::directory::set_mmap_allowed(allowed);
        }
        Err(e) => {
            eprintln!("fatal exception while booting Elasticsearch: java.lang.IllegalArgumentException: {e}");
            std::process::exit(1);
        }
    }
    // The segment format this process writes follows the cluster's
    // negotiated format floor (every member reads it); an explicit
    // `RUSTIC_SEGMENT_WRITE_VERSION` only caps it.
    // Diagnostics: `RUSTIC_MEMORY_REPORT_SECS=n` logs the memory accounting
    // categories, the shared admission budget and the code logs every n s.
    if let Some(secs) = std::env::var("RUSTIC_MEMORY_REPORT_SECS").ok().and_then(|v| v.parse::<u64>().ok()).filter(|s| *s > 0) {
        let _ = std::thread::Builder::new().name("memory-report".into()).spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            tracing::info!(target: "memory_report", "{}", elasticsearch_engine::rustic_core::accounting::report());
        });
    }
    let segment_read_max = elasticsearch_engine::rustic_core::codec::VERSION;
    match elasticsearch_engine::rustic_core::codec::write_version_cap() {
        Some(cap) => tracing::info!(segment_read_max, segment_write_cap = cap, "segment formats: reads through v{segment_read_max}; writes the cluster's negotiated format, capped at v{cap}"),
        None => tracing::info!(segment_read_max, "segment formats: reads through v{segment_read_max}; writes the cluster's negotiated format"),
    }
    // Idle workers give their freed pages back while the parent breaker is
    // over its limit (a relaxed load otherwise); when it goes over, the
    // workers already parked are woken to do so.
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().on_thread_park(elasticsearch_engine::memory::idle_collect).build().expect("tokio runtime");
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Handle> = std::sync::OnceLock::new();
    let _ = RUNTIME.set(rt.handle().clone());
    // One task per worker, each holding its worker until all have started
    // (at most 20 ms): a worker holding one cannot take the next, so the
    // tasks spread over the parked workers instead of one worker draining
    // them all, and each worker parks (and collects) after its own. Rare:
    // the breakers wake at most once a second, and only while over.
    fn wake_runtime() {
        let Some(handle) = RUNTIME.get() else { return };
        let workers = handle.metrics().num_workers();
        let gate = std::sync::Arc::new((std::sync::Mutex::new(0usize), std::sync::Condvar::new()));
        for _ in 0..workers {
            let gate = gate.clone();
            handle.spawn(async move {
                let (started, all) = &*gate;
                let mut n = started.lock().unwrap_or_else(|e| e.into_inner());
                *n += 1;
                if *n >= workers {
                    all.notify_all();
                    return;
                }
                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(20);
                while *n < workers {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        break;
                    }
                    n = all.wait_timeout(n, left).unwrap_or_else(|e| e.into_inner()).0;
                }
            });
        }
    }
    elasticsearch_engine::memory::add_wake_hook(wake_runtime);
    rt.block_on(async move {
        tokio::select! {
            r = crate::run(cfg) => {
                if let Err(e) = r {
                    // ES `NodeValidationException` (bootstrap checks): exit 78.
                    if let Some(v) = e.downcast_ref::<crate::bootstrap_checks::NodeValidationError>() {
                        tracing::error!("node validation exception\n{v}");
                        eprintln!("ERROR: {v}");
                        std::process::exit(78);
                    }
                    tracing::error!("fatal: {e}");
                    std::process::exit(1);
                }
            }
            _ = shutdown_signal() => {
                tracing::info!("shutting down");
                crate::shutdown(std::time::Duration::from_secs(5)).await;
            }
        }
    });
    // Exit without dropping the runtime: its drop cancels every task (a
    // waiter on a cancelled blocking task panicked, aborting the process)
    // and then waits for running blocking tasks without bound (the process,
    // and its listeners, outlived the signal). Nothing is left to do here.
    tracing::info!("stopped");
    std::process::exit(0);
}

/// The `--enrollment-token` value (`--enrollment-token t` or
/// `--enrollment-token=t`); more than one is a usage error, as in ES.
pub fn enrollment_token(args: &[String]) -> Result<Option<String>, String> {
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--enrollment-token" {
            i += 1;
            match args.get(i) {
                Some(t) => tokens.push(t.clone()),
                None => return Err("Option enrollment-token requires an argument".into()),
            }
        } else if let Some(t) = a.strip_prefix("--enrollment-token=") {
            tokens.push(t.to_string());
        }
        i += 1;
    }
    if tokens.len() > 1 {
        return Err("Multiple --enrollment-token parameters are not allowed".into());
    }
    Ok(tokens.pop())
}

/// Ctrl-C (and Ctrl-Break / console close on Windows).
#[cfg(windows)]
async fn shutdown_signal() {
    use tokio::signal::windows::{ctrl_break, ctrl_close};
    let mut brk = ctrl_break().expect("Ctrl-Break handler");
    let mut close = ctrl_close().expect("console close handler");
    tokio::select! {
        _ = brk.recv() => {}
        _ = close.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    tokio::select! {
        _ = term.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enrollment_token_option() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(enrollment_token(&a(&["-Ea=b"])).unwrap(), None);
        assert_eq!(enrollment_token(&a(&["--enrollment-token", "tok", "-v"])).unwrap().as_deref(), Some("tok"));
        assert_eq!(enrollment_token(&a(&["--enrollment-token=tok"])).unwrap().as_deref(), Some("tok"));
        assert_eq!(enrollment_token(&a(&["--enrollment-token", "a", "--enrollment-token", "b"])).unwrap_err(), "Multiple --enrollment-token parameters are not allowed");
        assert!(enrollment_token(&a(&["--enrollment-token"])).is_err());
    }
}
