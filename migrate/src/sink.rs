//! Where the import goes: a running server (REST) or a directory of request
//! files that can be inspected or replayed later.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

pub enum Sink {
    Http { base: String, auth: Option<String>, agent: ureq::Agent },
    Dir { out: PathBuf },
}

pub struct BulkOutcome {
    pub items: usize,
    pub conflicts: usize,
    pub errors: Vec<String>,
}

impl Sink {
    pub fn http(base: &str, user: Option<&str>) -> Sink {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).timeout(Duration::from_secs(600)).build();
        let auth = user.map(|u| format!("Basic {}", crate::es::base64(u.as_bytes(), false)));
        Sink::Http { base: base.trim_end_matches('/').to_string(), auth, agent }
    }

    pub fn dir(out: PathBuf) -> Result<Sink> {
        std::fs::create_dir_all(&out).with_context(|| format!("create {}", out.display()))?;
        Ok(Sink::Dir { out })
    }

    fn append(path: PathBuf) -> Result<File> {
        Ok(OpenOptions::new().create(true).append(true).open(&path).with_context(|| format!("open {}", path.display()))?)
    }

    fn send(&self, method: &str, path: &str, ctype: &str, body: Option<&str>) -> Result<(u16, Value)> {
        let Sink::Http { base, auth, agent } = self else { unreachable!() };
        let mut req = agent.request(method, &format!("{}/{}", base, path.trim_start_matches('/')));
        if let Some(a) = auth {
            req = req.set("Authorization", a);
        }
        let res = match body {
            Some(b) => req.set("Content-Type", ctype).send_string(b),
            None => req.call(),
        };
        let resp = match res {
            Ok(r) => r,
            Err(ureq::Error::Status(_, r)) => r,
            Err(e) => bail!("{} {}: {}", method, path, e),
        };
        let status = resp.status();
        let text = resp.into_string().unwrap_or_default();
        Ok((status, serde_json::from_str(&text).unwrap_or(Value::String(text))))
    }

    /// A metadata request. In directory mode it is recorded and reported as 200.
    pub fn request(&mut self, method: &str, path: &str, body: Option<&Value>) -> Result<(u16, Value)> {
        match self {
            Sink::Dir { out } => {
                let mut f = Self::append(out.join("requests.ndjson"))?;
                let line = serde_json::json!({"method": method, "path": path, "body": body});
                writeln!(f, "{}", line)?;
                Ok((200, Value::Null))
            }
            Sink::Http { .. } => {
                let b = body.map(|b| b.to_string());
                self.send(method, path, "application/json", b.as_deref())
            }
        }
    }

    pub fn exists(&mut self, path: &str) -> Result<bool> {
        match self {
            Sink::Dir { .. } => Ok(false),
            Sink::Http { .. } => Ok(self.send("GET", path, "application/json", None)?.0 == 200),
        }
    }

    pub fn bulk(&mut self, label: &str, query: &str, body: &str) -> Result<BulkOutcome> {
        match self {
            Sink::Dir { out } => {
                let mut f = Self::append(out.join(format!("{}.bulk.ndjson", label)))?;
                f.write_all(body.as_bytes())?;
                Ok(BulkOutcome { items: body.lines().count() / 2, conflicts: 0, errors: vec![] })
            }
            Sink::Http { .. } => {
                let (status, v) = self.send("POST", &format!("_bulk{}", query), "application/x-ndjson", Some(body))?;
                if status != 200 {
                    bail!("_bulk returned {}: {}", status, truncate(&v.to_string(), 500));
                }
                let items = v.get("items").and_then(|i| i.as_array()).cloned().unwrap_or_default();
                let mut out = BulkOutcome { items: items.len(), conflicts: 0, errors: vec![] };
                if v.get("errors").and_then(|e| e.as_bool()) == Some(true) {
                    for it in &items {
                        let Some(r) = it.as_object().and_then(|o| o.values().next()) else { continue };
                        if let Some(err) = r.get("error") {
                            if r.get("status").and_then(|s| s.as_u64()) == Some(409) {
                                out.conflicts += 1;
                            } else {
                                out.errors.push(format!("[{}] {}", r.get("_id").and_then(|i| i.as_str()).unwrap_or("?"), truncate(&err.to_string(), 300)));
                            }
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    pub fn is_dry(&self) -> bool {
        matches!(self, Sink::Dir { .. })
    }
}

pub fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}
