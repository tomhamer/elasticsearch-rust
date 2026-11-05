//! Network partitions between processes that all live on loopback.
//!
//! Mechanism (verified on Linux 6.x / iptables-nft 1.8):
//!
//! * Each node listens on its own address `127.0.P.i` (`network.host`).
//! * Each node runs in its own cgroup; a `nat POSTROUTING -o lo -m cgroup
//!   --path <cg> -j SNAT --to-source 127.0.P.i` rule rewrites the source of
//!   every connection the node *opens*, so on the wire every inter-node
//!   packet carries `src = sender's address, dst = receiver's address` —
//!   in both directions of the TCP connection.
//! * Drops live in the `raw` table (before conntrack/NAT): `raw OUTPUT`
//!   sees reply packets before the SNAT reverse-mapping rewrites their
//!   destination back to 127.0.0.1, and `raw PREROUTING` sees original-
//!   direction packets after SNAT. A rule `-s A -d B -j DROP` in a chain
//!   jumped to from both hooks therefore drops *every* packet from node A
//!   to node B, whoever opened the connection.
//! * The harness itself talks to nodes from 127.0.0.1 (not in any node
//!   cgroup), so clients are never partitioned from the node they talk to;
//!   only inter-node traffic is affected. Rules match only our 127.0.P.x
//!   addresses, so other processes on the host are unaffected.
//!
//! All rules carry the comment `es-faults-<port_base>` and live in chains
//! named `ESF<port_base>` / `ESF<port_base>_SNAT`.

use crate::cluster::{cgroup_root, node_ip};
use parking_lot::Mutex;
use std::collections::BTreeSet;
use std::io::Write;
use std::process::{Command, Stdio};

pub type Drops = BTreeSet<(usize, usize)>;

pub struct Net {
    pub port_base: u16,
    pub n: usize,
    current: Mutex<Drops>,
    netem: Mutex<bool>,
}

fn comment(pb: u16) -> String {
    format!("es-faults-{pb}")
}
fn chain(pb: u16) -> String {
    format!("ESF{pb}")
}
fn snat_chain(pb: u16) -> String {
    format!("ESF{pb}_SNAT")
}

fn ipt(args: &[&str]) -> Result<String, String> {
    let out = Command::new("sudo").arg("-n").arg("iptables").arg("-w").args(args).output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("iptables {:?}: {}", args, String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn tc(args: &[&str]) -> Result<String, String> {
    let out = Command::new("sudo").arg("-n").arg("tc").args(args).output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("tc {:?}: {}", args, String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Remove every rule/chain/qdisc a run on `port_base` may have installed.
pub fn teardown(port_base: u16) {
    let c = comment(port_base);
    let ch = chain(port_base);
    let sn = snat_chain(port_base);
    for (table, hook, target) in [("raw", "PREROUTING", &ch), ("raw", "OUTPUT", &ch), ("nat", "POSTROUTING", &sn)] {
        for _ in 0..32 {
            if ipt(&["-t", table, "-D", hook, "-m", "comment", "--comment", &c, "-j", target]).is_err() {
                break;
            }
        }
    }
    let _ = ipt(&["-t", "raw", "-F", &ch]);
    let _ = ipt(&["-t", "raw", "-X", &ch]);
    let _ = ipt(&["-t", "nat", "-F", &sn]);
    let _ = ipt(&["-t", "nat", "-X", &sn]);
    netem_teardown(port_base);
}

/// The netem qdisc is installed only when no root qdisc other than the
/// default `noqueue` exists on lo; we remember that by the handle `1:` *and*
/// a marker file, and only ever delete our own.
fn netem_marker(port_base: u16) -> String {
    format!("/tmp/es-faults-{port_base}.netem")
}

fn netem_teardown(port_base: u16) {
    if std::path::Path::new(&netem_marker(port_base)).exists() {
        let _ = tc(&["qdisc", "del", "dev", "lo", "root", "handle", "1:"]);
        let _ = std::fs::remove_file(netem_marker(port_base));
    }
}

impl Net {
    pub fn setup(port_base: u16, n: usize) -> Result<Net, String> {
        teardown(port_base);
        let c = comment(port_base);
        let ch = chain(port_base);
        let sn = snat_chain(port_base);
        ipt(&["-t", "nat", "-N", &sn])?;
        ipt(&["-t", "nat", "-I", "POSTROUTING", "-m", "comment", "--comment", &c, "-j", &sn])?;
        for i in 0..n {
            let cg = format!("{}/n{}", cgroup_root(port_base), i + 1);
            let ip = node_ip(port_base, i).to_string();
            ipt(&["-t", "nat", "-A", &sn, "-o", "lo", "-m", "cgroup", "--path", &cg, "-m", "comment", "--comment", &c, "-j", "SNAT", "--to-source", &ip])?;
        }
        ipt(&["-t", "raw", "-N", &ch])?;
        ipt(&["-t", "raw", "-I", "PREROUTING", "-m", "comment", "--comment", &c, "-j", &ch])?;
        ipt(&["-t", "raw", "-I", "OUTPUT", "-m", "comment", "--comment", &c, "-j", &ch])?;
        Ok(Net { port_base, n, current: Mutex::new(Drops::new()), netem: Mutex::new(false) })
    }

    pub fn current(&self) -> Drops {
        self.current.lock().clone()
    }

    /// Atomically replace the drop set (`(a, b)` = drop every packet a→b).
    pub fn apply(&self, drops: &Drops) -> Result<(), String> {
        let ch = chain(self.port_base);
        let c = comment(self.port_base);
        let mut rules = format!("*raw\n-F {ch}\n");
        for (a, b) in drops {
            rules.push_str(&format!("-A {ch} -s {}/32 -d {}/32 -m comment --comment {c} -j DROP\n", node_ip(self.port_base, *a), node_ip(self.port_base, *b)));
        }
        rules.push_str("COMMIT\n");
        let mut child = Command::new("sudo")
            .args(["-n", "iptables-restore", "--noflush"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        child.stdin.take().expect("stdin").write_all(rules.as_bytes()).map_err(|e| e.to_string())?;
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err(format!("iptables-restore: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        *self.current.lock() = drops.clone();
        Ok(())
    }

    pub fn heal(&self) -> Result<(), String> {
        self.apply(&Drops::new())
    }

    /// Delay/loss on every packet *sent by* our nodes (egress of their
    /// addresses on lo). Matching by source only is deliberate: reply
    /// packets have their destination rewritten back to 127.0.0.1 before
    /// the qdisc sees them.
    pub fn netem_on(&self, delay_ms: u32, jitter_ms: u32, loss_pct: f64) -> Result<(), String> {
        let mut on = self.netem.lock();
        if *on {
            return Ok(());
        }
        let show = tc(&["qdisc", "show", "dev", "lo"])?;
        if !show.trim().starts_with("qdisc noqueue") {
            return Err(format!("lo already has a non-default qdisc, refusing to touch it: {}", show.trim()));
        }
        tc(&["qdisc", "add", "dev", "lo", "root", "handle", "1:", "prio", "bands", "4", "priomap", "1", "2", "2", "2", "1", "2", "0", "0", "1", "1", "1", "1", "1", "1", "1", "1"])?;
        let _ = std::fs::write(netem_marker(self.port_base), "1");
        let delay = format!("{delay_ms}ms");
        let jitter = format!("{jitter_ms}ms");
        let loss = format!("{loss_pct}%");
        tc(&["qdisc", "add", "dev", "lo", "parent", "1:4", "handle", "40:", "netem", "delay", &delay, &jitter, "loss", &loss])?;
        for i in 0..self.n {
            let src = format!("{}/32", node_ip(self.port_base, i));
            tc(&["filter", "add", "dev", "lo", "parent", "1:", "protocol", "ip", "prio", "1", "u32", "match", "ip", "src", &src, "flowid", "1:4"])?;
        }
        *on = true;
        Ok(())
    }

    pub fn netem_off(&self) {
        let mut on = self.netem.lock();
        if *on {
            netem_teardown(self.port_base);
            *on = false;
        }
    }
}

impl Drop for Net {
    fn drop(&mut self) {
        teardown(self.port_base);
    }
}

// ------------------------------------------------------------ topologies

/// Drop everything between nodes of different groups (both directions).
pub fn split(groups: &[Vec<usize>]) -> Drops {
    let mut d = Drops::new();
    for (gi, g) in groups.iter().enumerate() {
        for (hi, h) in groups.iter().enumerate() {
            if gi != hi {
                for a in g {
                    for b in h {
                        d.insert((*a, *b));
                    }
                }
            }
        }
    }
    d
}

pub fn isolate(n: usize, x: usize) -> Drops {
    let rest: Vec<usize> = (0..n).filter(|i| *i != x).collect();
    split(&[vec![x], rest])
}

/// `left | bridge | right` where the bridge node sees everyone.
pub fn bridge(left: &[usize], right: &[usize]) -> Drops {
    split(&[left.to_vec(), right.to_vec()])
}

/// One direction only: drop packets from every node in `from` to every
/// node in `to`.
pub fn one_way(from: &[usize], to: &[usize]) -> Drops {
    let mut d = Drops::new();
    for a in from {
        for b in to {
            if a != b {
                d.insert((*a, *b));
            }
        }
    }
    d
}

/// Nodes that can exchange packets in both directions with fewer than a
/// quorum of nodes (counting themselves): they must not be able to elect a
/// master or acknowledge writes.
pub fn minority_nodes(n: usize, drops: &Drops) -> Vec<usize> {
    let quorum = n / 2 + 1;
    (0..n)
        .filter(|a| {
            let reach = (0..n).filter(|b| a == b || (!drops.contains(&(*a, *b)) && !drops.contains(&(*b, *a)))).count();
            reach < quorum
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topologies() {
        let d = split(&[vec![0, 1], vec![2, 3, 4]]);
        assert_eq!(d.len(), 12);
        assert_eq!(minority_nodes(5, &d), vec![0, 1]);
        let b = bridge(&[0, 1], &[3, 4]);
        assert_eq!(b.len(), 8);
        assert!(minority_nodes(5, &b).is_empty());
        let i = isolate(5, 2);
        assert_eq!(minority_nodes(5, &i), vec![2]);
        let o = one_way(&[0], &[1, 2, 3, 4]);
        assert_eq!(o.len(), 4);
        assert_eq!(minority_nodes(5, &o), vec![0]);
    }
}
