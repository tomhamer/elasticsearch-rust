//! Concurrent clients. Every request goes to a uniformly random node (the
//! client does not know which nodes are down, paused or partitioned, exactly
//! like a real client behind a load balancer). Nothing is retried.

use crate::api::{call, classify_write, client, HttpErr};
use crate::cluster::Cluster;
use crate::history::{blank, History, Kind, Outcome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct Workload {
    pub indices: Vec<String>,
    /// Register keys `r0..rN` per index.
    pub registers: usize,
    pub hist: Arc<History>,
    pub cluster: Arc<Cluster>,
    pub op_timeout: Duration,
    pub stop: Arc<AtomicBool>,
    pub bulk_seq: AtomicU64,
}

pub fn register_key(k: usize) -> String {
    format!("r{k}")
}

fn value_of(v: &Value) -> Option<i64> {
    v["_source"]["v"].as_i64()
}

impl Workload {
    pub async fn run_worker(self: Arc<Self>, wid: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(wid as u64));
        let http = client(self.op_timeout);
        let mut counter: i64 = 0;
        let mut acked: Vec<(String, String)> = Vec::new();
        while !self.stop.load(Ordering::SeqCst) {
            let roll = rng.gen_range(0..100);
            let node = rng.gen_range(0..self.cluster.n);
            let index = self.indices[rng.gen_range(0..self.indices.len())].clone();
            counter += 1;
            let value = (wid as i64 + 1) * 1_000_000_000 + counter;
            if roll < 30 {
                let key = format!("c{wid}-{counter}");
                if self.create(&http, wid, node, &index, &key, value).await == Outcome::Ok {
                    acked.push((index, key));
                    if acked.len() > 64 {
                        acked.remove(0);
                    }
                }
            } else if roll < 60 {
                let key = register_key(rng.gen_range(0..self.registers));
                if let Some((seq, term)) = self.read(&http, wid, node, &index, &key).await.and_then(|r| r.1) {
                    let node2 = rng.gen_range(0..self.cluster.n);
                    self.cas(&http, wid, node2, &index, &key, value, seq, term).await;
                }
            } else if roll < 70 {
                let m = rng.gen_range(2..8usize);
                let items: Vec<(String, String, i64)> = (0..m)
                    .map(|j| {
                        counter += 1;
                        let ix = self.indices[rng.gen_range(0..self.indices.len())].clone();
                        (ix, format!("b{wid}-{counter}-{j}"), (wid as i64 + 1) * 1_000_000_000 + counter)
                    })
                    .collect();
                self.bulk(&http, wid, node, &items).await;
            } else if roll < 80 {
                let key = format!("d{wid}-{counter}");
                if self.create(&http, wid, node, &index, &key, value).await == Outcome::Ok {
                    let node2 = rng.gen_range(0..self.cluster.n);
                    self.delete(&http, wid, node2, &index, &key).await;
                }
            } else {
                let (ix, key) = if !acked.is_empty() && rng.gen_bool(0.4) {
                    acked[rng.gen_range(0..acked.len())].clone()
                } else {
                    (index, register_key(rng.gen_range(0..self.registers)))
                };
                self.read(&http, wid, node, &ix, &key).await;
            }
            tokio::time::sleep(Duration::from_millis(rng.gen_range(2..20))).await;
        }
    }

    fn url(&self, node: usize, path: &str) -> String {
        format!("{}{}", self.cluster.url(node), path)
    }

    async fn write_single(&self, http: &reqwest::Client, id: u64, method: &str, url: String, body: Option<String>) -> Outcome {
        let r = call(http, method, &url, body, None).await;
        let mut outcome = Outcome::Info;
        self.hist.complete(id, |op| match r {
            Ok((status, v)) => {
                outcome = classify_write(status, &v);
                op.outcome = outcome;
                op.status = Some(status);
                if outcome == Outcome::Ok {
                    op.seq_no = v["_seq_no"].as_i64();
                    op.primary_term = v["_primary_term"].as_i64();
                    op.version = v["_version"].as_i64();
                    if v["result"] == "not_found" {
                        op.read = Some(None);
                    }
                } else {
                    op.error = Some(format!("{}: {}", crate::api::error_type(&v), v["error"]["reason"].as_str().unwrap_or("")));
                }
            }
            Err(HttpErr::NotSent(m)) => {
                outcome = Outcome::Fail;
                op.outcome = outcome;
                op.error = Some(format!("not sent: {m}"));
            }
            Err(HttpErr::Unknown(m)) => {
                op.outcome = Outcome::Info;
                op.error = Some(format!("no response: {m}"));
            }
        });
        outcome
    }

    async fn create(&self, http: &reqwest::Client, wid: usize, node: usize, index: &str, key: &str, value: i64) -> Outcome {
        let mut op = blank(wid, node, Kind::Create, index, key);
        op.value = Some(value);
        let id = self.hist.invoke(op);
        let body = json!({"k": key, "v": value}).to_string();
        self.write_single(http, id, "PUT", self.url(node, &format!("/{index}/_create/{key}")), Some(body)).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn cas(&self, http: &reqwest::Client, wid: usize, node: usize, index: &str, key: &str, value: i64, seq: i64, term: i64) -> Outcome {
        let mut op = blank(wid, node, Kind::Cas, index, key);
        op.value = Some(value);
        op.if_seq_no = Some(seq);
        op.if_primary_term = Some(term);
        let id = self.hist.invoke(op);
        let body = json!({"k": key, "v": value}).to_string();
        self.write_single(http, id, "PUT", self.url(node, &format!("/{index}/_doc/{key}?if_seq_no={seq}&if_primary_term={term}")), Some(body)).await
    }

    async fn delete(&self, http: &reqwest::Client, wid: usize, node: usize, index: &str, key: &str) -> Outcome {
        let op = blank(wid, node, Kind::Delete, index, key);
        let id = self.hist.invoke(op);
        self.write_single(http, id, "DELETE", self.url(node, &format!("/{index}/_doc/{key}")), None).await
    }

    /// Realtime GET. Returns `(value, Some((seq_no, primary_term)))` when
    /// the read completed.
    async fn read(&self, http: &reqwest::Client, wid: usize, node: usize, index: &str, key: &str) -> Option<(Option<i64>, Option<(i64, i64)>)> {
        let op = blank(wid, node, Kind::Read, index, key);
        let id = self.hist.invoke(op);
        let r = call(http, "GET", &self.url(node, &format!("/{index}/_doc/{key}")), None, None).await;
        let mut res = None;
        self.hist.complete(id, |op| match r {
            Ok((200, v)) if v["found"] == true => {
                op.outcome = Outcome::Ok;
                op.status = Some(200);
                op.read = Some(value_of(&v));
                op.seq_no = v["_seq_no"].as_i64();
                op.primary_term = v["_primary_term"].as_i64();
                op.version = v["_version"].as_i64();
                let st = match (op.seq_no, op.primary_term) {
                    (Some(s), Some(t)) => Some((s, t)),
                    _ => None,
                };
                res = Some((value_of(&v), st));
            }
            Ok((404, v)) if v["found"] == false => {
                op.outcome = Outcome::Ok;
                op.status = Some(404);
                op.read = Some(None);
                res = Some((None, None));
            }
            Ok((s, v)) => {
                op.outcome = Outcome::Fail;
                op.status = Some(s);
                op.error = Some(format!("{}: {}", crate::api::error_type(&v), v["error"]["reason"].as_str().unwrap_or("")));
            }
            Err(e) => {
                op.outcome = Outcome::Fail;
                op.error = Some(e.msg());
            }
        });
        res
    }

    async fn bulk(&self, http: &reqwest::Client, wid: usize, node: usize, items: &[(String, String, i64)]) {
        let bulk_id = self.bulk_seq.fetch_add(1, Ordering::SeqCst);
        let mut body = String::new();
        let mut ids = Vec::new();
        for (ix, key, value) in items {
            let mut op = blank(wid, node, Kind::BulkCreate, ix, key);
            op.value = Some(*value);
            op.bulk_id = Some(bulk_id);
            ids.push(self.hist.invoke(op));
            body.push_str(&json!({"create": {"_index": ix, "_id": key}}).to_string());
            body.push('\n');
            body.push_str(&json!({"k": key, "v": value}).to_string());
            body.push('\n');
        }
        let r = call(http, "POST", &self.url(node, "/_bulk"), Some(body), None).await;
        for (pos, id) in ids.iter().enumerate() {
            self.hist.complete(*id, |op| match &r {
                Ok((200, v)) => {
                    let item = &v["items"][pos]["create"];
                    let st = item["status"].as_u64().unwrap_or(0) as u16;
                    op.status = Some(st);
                    op.outcome = if item.is_null() { Outcome::Info } else { classify_write(st, item) };
                    if op.outcome == Outcome::Ok {
                        op.seq_no = item["_seq_no"].as_i64();
                        op.primary_term = item["_primary_term"].as_i64();
                        op.version = item["_version"].as_i64();
                    } else {
                        op.error = Some(format!("{}: {}", crate::api::error_type(item), item["error"]["reason"].as_str().unwrap_or("")));
                    }
                }
                Ok((s, v)) => {
                    op.status = Some(*s);
                    // A non-200 bulk response from the front may still
                    // follow partial execution: only a block is definite.
                    op.outcome = if *s == 503 && crate::api::error_type(v) == "cluster_block_exception" { Outcome::Fail } else { Outcome::Info };
                    op.error = Some(format!("{}: {}", crate::api::error_type(v), v["error"]["reason"].as_str().unwrap_or("")));
                }
                Err(HttpErr::NotSent(m)) => {
                    op.outcome = Outcome::Fail;
                    op.error = Some(format!("not sent: {m}"));
                }
                Err(HttpErr::Unknown(m)) => {
                    op.outcome = Outcome::Info;
                    op.error = Some(format!("no response: {m}"));
                }
            });
        }
    }
}
