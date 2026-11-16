//! The AWS default credential provider chain, as ES `repository-s3` uses it
//! when a client has no keystore credentials:
//!
//! 1. environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//!    `AWS_SESSION_TOKEN`);
//! 2. web identity (IRSA / EKS pod identity tokens): `AWS_WEB_IDENTITY_TOKEN_FILE`
//!    + `AWS_ROLE_ARN` exchanged at STS `AssumeRoleWithWebIdentity`;
//! 3. container credentials (ECS task role):
//!    `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` or `..._FULL_URI`, with
//!    `AWS_CONTAINER_AUTHORIZATION_TOKEN[_FILE]`;
//! 4. the EC2 instance profile through IMDSv2 (session token first).
//!
//! Temporary credentials are cached and refreshed before they expire
//! ([`ChainConfig::refresh_before`]); a refresh that fails while the cached
//! credentials are still valid keeps using them. A request rejected for an
//! expired token invalidates the cache ([`CredentialChain::invalidate`]).

use super::sigv4::Credentials;
use crate::error::{Result, SnapshotError};
use std::time::{Duration, SystemTime};

/// Where the chain looks; [`ChainConfig::from_env`] reads the standard AWS
/// environment variables, tests point the endpoints at mocks.
#[derive(Debug, Clone)]
pub struct ChainConfig {
    pub region: String,
    pub use_env: bool,
    pub web_identity_token_file: Option<String>,
    pub role_arn: Option<String>,
    pub role_session_name: String,
    /// STS endpoint (default `https://sts.<region>.amazonaws.com`).
    pub sts_endpoint: Option<String>,
    /// Full container credentials URL.
    pub container_uri: Option<String>,
    pub container_token: Option<String>,
    pub container_token_file: Option<String>,
    /// `None`: IMDS disabled.
    pub imds_endpoint: Option<String>,
    /// Refresh temporary credentials this long before they expire.
    pub refresh_before: Duration,
    /// Limit on each metadata/STS exchange.
    pub timeout: Duration,
}

impl ChainConfig {
    pub fn from_env(region: &str) -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let container_uri = var("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            .or_else(|| var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").map(|p| format!("http://169.254.170.2{p}")));
        let imds_disabled = var("AWS_EC2_METADATA_DISABLED").is_some_and(|v| v.eq_ignore_ascii_case("true"));
        Self {
            region: region.to_string(),
            use_env: true,
            web_identity_token_file: var("AWS_WEB_IDENTITY_TOKEN_FILE"),
            role_arn: var("AWS_ROLE_ARN"),
            role_session_name: var("AWS_ROLE_SESSION_NAME").unwrap_or_else(|| "elasticsearch-repository-s3".into()),
            sts_endpoint: var("AWS_ENDPOINT_URL_STS"),
            container_uri,
            container_token: var("AWS_CONTAINER_AUTHORIZATION_TOKEN"),
            container_token_file: var("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"),
            imds_endpoint: if imds_disabled {
                None
            } else {
                Some(var("AWS_EC2_METADATA_SERVICE_ENDPOINT").unwrap_or_else(|| "http://169.254.169.254".into()))
            },
            refresh_before: Duration::from_secs(5 * 60),
            timeout: Duration::from_secs(5),
        }
    }

    /// Only the given endpoints, nothing from the environment.
    pub fn empty(region: &str) -> Self {
        Self {
            region: region.to_string(),
            use_env: false,
            web_identity_token_file: None,
            role_arn: None,
            role_session_name: "elasticsearch-repository-s3".into(),
            sts_endpoint: None,
            container_uri: None,
            container_token: None,
            container_token_file: None,
            imds_endpoint: None,
            refresh_before: Duration::from_secs(5 * 60),
            timeout: Duration::from_secs(5),
        }
    }
}

#[derive(Clone)]
struct Cached {
    creds: Credentials,
    /// `None`: long-lived (environment).
    expires: Option<SystemTime>,
    source: &'static str,
}

/// The default provider chain with its cache.
pub struct CredentialChain {
    cfg: ChainConfig,
    http: reqwest::Client,
    cache: tokio::sync::Mutex<Option<Cached>>,
}

impl std::fmt::Debug for CredentialChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CredentialChain").field("cfg", &self.cfg).finish()
    }
}

fn err(msg: impl Into<String>) -> SnapshotError {
    SnapshotError::new("sdk_client_exception", 500, msg)
}

impl CredentialChain {
    pub fn new(cfg: ChainConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(cfg.timeout)
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| SnapshotError::io("building credentials client", e))?;
        Ok(Self { cfg, http, cache: tokio::sync::Mutex::new(None) })
    }

    /// Current credentials, refreshed when they are within
    /// `refresh_before` of expiring.
    pub async fn credentials(&self) -> Result<Credentials> {
        let mut cache = self.cache.lock().await;
        let now = SystemTime::now();
        if let Some(c) = cache.as_ref() {
            let fresh = match c.expires {
                None => true,
                Some(exp) => exp.duration_since(now).map(|left| left > self.cfg.refresh_before).unwrap_or(false),
            };
            if fresh {
                return Ok(c.creds.clone());
            }
        }
        match self.resolve().await {
            Ok(c) => {
                tracing::debug!("loaded s3 credentials from {} (expires {:?})", c.source, c.expires);
                let creds = c.creds.clone();
                *cache = Some(c);
                Ok(creds)
            }
            Err(e) => match cache.as_ref() {
                // Refresh failed but the cached credentials still work.
                Some(c) if c.expires.is_some_and(|exp| exp > now) => {
                    tracing::warn!("refreshing s3 credentials from {} failed, using the cached ones until they expire: {e}", c.source);
                    Ok(c.creds.clone())
                }
                _ => Err(e),
            },
        }
    }

    /// Drop the cached credentials (the service rejected them).
    pub async fn invalidate(&self) {
        *self.cache.lock().await = None;
    }

    async fn resolve(&self) -> Result<Cached> {
        if self.cfg.use_env {
            if let Some(creds) = Credentials::from_env() {
                return Ok(Cached { creds, expires: None, source: "environment" });
            }
        }
        let mut tried = Vec::new();
        if let (Some(file), Some(role)) = (&self.cfg.web_identity_token_file, &self.cfg.role_arn) {
            match self.web_identity(file, role).await {
                Ok(c) => return Ok(c),
                Err(e) => tried.push(format!("web identity: {}", e.reason)),
            }
        }
        if let Some(uri) = &self.cfg.container_uri {
            match self.container(uri).await {
                Ok(c) => return Ok(c),
                Err(e) => tried.push(format!("container: {}", e.reason)),
            }
        }
        if let Some(ep) = &self.cfg.imds_endpoint {
            match self.imds(ep).await {
                Ok(c) => return Ok(c),
                Err(e) => tried.push(format!("instance profile: {}", e.reason)),
            }
        }
        Err(err(format!(
            "Unable to load AWS credentials from any provider in the chain: set s3.client.<name>.access_key/secret_key, \
             AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, a web identity, container or instance role [{}]",
            tried.join("; ")
        )))
    }

    async fn web_identity(&self, token_file: &str, role_arn: &str) -> Result<Cached> {
        let token = tokio::fs::read_to_string(token_file).await.map_err(|e| err(format!("reading [{token_file}]: {e}")))?;
        let endpoint = self.cfg.sts_endpoint.clone().unwrap_or_else(|| format!("https://sts.{}.amazonaws.com", self.cfg.region));
        let form = [
            ("Action", "AssumeRoleWithWebIdentity"),
            ("Version", "2011-06-15"),
            ("RoleArn", role_arn),
            ("RoleSessionName", self.cfg.role_session_name.as_str()),
            ("WebIdentityToken", token.trim()),
        ];
        let body = form
            .iter()
            .map(|(k, v)| format!("{k}={}", super::sigv4::uri_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        let r = self
            .http
            .post(format!("{}/", endpoint.trim_end_matches('/')))
            .header("content-type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| err(format!("STS [{endpoint}]: {e}")))?;
        let status = r.status().as_u16();
        let text = r.text().await.map_err(|e| err(format!("STS [{endpoint}]: {e}")))?;
        if !(200..300).contains(&status) {
            return Err(err(format!("STS AssumeRoleWithWebIdentity returned {status}: {}", super::xml::tag(&text, "Message").unwrap_or_default())));
        }
        let get = |t: &str| super::xml::tag(&text, t).ok_or_else(|| err(format!("STS response without <{t}>")));
        Ok(Cached {
            creds: Credentials { access_key: get("AccessKeyId")?, secret_key: get("SecretAccessKey")?, session_token: Some(get("SessionToken")?) },
            expires: Some(parse_expiration(&get("Expiration")?)?),
            source: "web identity",
        })
    }

    async fn container(&self, uri: &str) -> Result<Cached> {
        let mut rb = self.http.get(uri);
        let token = match (&self.cfg.container_token_file, &self.cfg.container_token) {
            (Some(f), _) => Some(tokio::fs::read_to_string(f).await.map_err(|e| err(format!("reading [{f}]: {e}")))?.trim().to_string()),
            (None, t) => t.clone(),
        };
        if let Some(t) = token {
            rb = rb.header("authorization", t);
        }
        let r = rb.send().await.map_err(|e| err(format!("[{uri}]: {e}")))?;
        let status = r.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(err(format!("[{uri}] returned {status}")));
        }
        let v: serde_json::Value = r.json().await.map_err(|e| err(format!("[{uri}]: {e}")))?;
        json_credentials(&v, "container")
    }

    async fn imds(&self, endpoint: &str) -> Result<Cached> {
        let base = endpoint.trim_end_matches('/');
        // IMDSv2: a session token is required for every metadata read.
        let r = self
            .http
            .put(format!("{base}/latest/api/token"))
            .header("x-aws-ec2-metadata-token-ttl-seconds", "21600")
            .send()
            .await
            .map_err(|e| err(format!("IMDS token: {e}")))?;
        if !r.status().is_success() {
            return Err(err(format!("IMDS token request returned {}", r.status().as_u16())));
        }
        let token = r.text().await.map_err(|e| err(format!("IMDS token: {e}")))?;
        let get = |path: String| {
            let rb = self.http.get(format!("{base}{path}")).header("x-aws-ec2-metadata-token", token.clone());
            async move {
                let r = rb.send().await.map_err(|e| err(format!("IMDS [{path}]: {e}")))?;
                if !r.status().is_success() {
                    return Err(err(format!("IMDS [{path}] returned {}", r.status().as_u16())));
                }
                r.text().await.map_err(|e| err(format!("IMDS [{path}]: {e}")))
            }
        };
        let roles = get("/latest/meta-data/iam/security-credentials/".into()).await?;
        let role = roles.lines().map(str::trim).find(|l| !l.is_empty()).ok_or_else(|| err("IMDS: no instance profile role"))?.to_string();
        let body = get(format!("/latest/meta-data/iam/security-credentials/{role}")).await?;
        let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| err(format!("IMDS credentials: {e}")))?;
        json_credentials(&v, "instance profile")
    }
}

fn json_credentials(v: &serde_json::Value, source: &'static str) -> Result<Cached> {
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
    let (Some(access_key), Some(secret_key)) = (s("AccessKeyId"), s("SecretAccessKey")) else {
        return Err(err(format!("{source} credentials response without AccessKeyId/SecretAccessKey")));
    };
    let expires = s("Expiration").map(|e| parse_expiration(&e)).transpose()?;
    Ok(Cached { creds: Credentials { access_key, secret_key, session_token: s("Token").or_else(|| s("SessionToken")) }, expires, source })
}

fn parse_expiration(s: &str) -> Result<SystemTime> {
    let t = chrono::DateTime::parse_from_rfc3339(s.trim()).map_err(|e| err(format!("bad credential expiration [{s}]: {e}")))?;
    Ok(SystemTime::from(t.with_timezone(&chrono::Utc)))
}
