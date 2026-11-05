//! `elasticsearch-operator` binary.
//!
//! ```text
//! elasticsearch-operator crd                 # print the CRD YAML
//! elasticsearch-operator [run] [options]     # run the controller
//!   --namespace <ns>                 watch one namespace (env WATCH_NAMESPACE; default: all)
//!   --default-image-repository <r>   image used when spec.image is unset: <r>:<spec.version>
//!                                    (env DEFAULT_IMAGE_REPOSITORY; default elasticsearch-rs)
//!   --es-timeout-secs <n>            per-request timeout to Elasticsearch (default 10)
//!   --health-addr <addr>             liveness/readiness endpoint (default 0.0.0.0:8080)
//! ```

use elasticsearch_operator::{controller, crd, resources};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn usage() -> ! {
    eprintln!("usage: elasticsearch-operator [crd | run [--namespace NS] [--default-image-repository REPO] [--es-timeout-secs N] [--health-addr ADDR]]");
    std::process::exit(2)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match args.first().map(String::as_str) {
        Some("crd") => {
            print!("{}", crd::crd_yaml());
            return Ok(());
        }
        Some("run") => ("run", &args[1..]),
        Some("-h") | Some("--help") => usage(),
        _ => ("run", &args[..]),
    };
    debug_assert_eq!(cmd, "run");

    let mut namespace = std::env::var("WATCH_NAMESPACE").ok().filter(|s| !s.is_empty());
    let mut repo = std::env::var("DEFAULT_IMAGE_REPOSITORY").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| resources::DEFAULT_IMAGE_REPOSITORY.to_string());
    let mut timeout = 10u64;
    let mut health_addr = "0.0.0.0:8080".to_string();
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| usage());
        match a.as_str() {
            "--namespace" => namespace = Some(val()),
            "--default-image-repository" => repo = val(),
            "--es-timeout-secs" => timeout = val().parse().unwrap_or_else(|_| usage()),
            "--health-addr" => health_addr = val(),
            _ => usage(),
        }
    }

    let json_logs = std::env::var("LOG_FORMAT").is_ok_and(|v| v == "json");
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,kube=warn".into());
    if json_logs {
        tracing_subscriber::fmt().with_env_filter(filter).json().init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    // Minimal health endpoint for the Deployment's probes.
    let listener = tokio::net::TcpListener::bind(&health_addr).await?;
    tokio::spawn(async move {
        loop {
            if let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok").await;
                });
            }
        }
    });

    let client = kube::Client::try_default().await?;
    tracing::info!(namespace = ?namespace, default_image_repository = %repo, "starting elasticsearch-operator");
    let ctx = controller::Context { client, default_image_repository: repo, es_client: controller::http_es_client_factory(Duration::from_secs(timeout)) };
    controller::run(ctx, namespace).await;
    Ok(())
}
