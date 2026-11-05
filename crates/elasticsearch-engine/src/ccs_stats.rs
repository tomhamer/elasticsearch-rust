//! Lifetime CCS request counters, owned by the node rather than a global cache.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Default, Clone, Serialize, Deserialize)]
struct Distribution {
    count: u64,
    sum: u128,
    max: u64,
    // Eight significant binary bits bound memory independently of request count.
    buckets: BTreeMap<u64, u64>,
}

impl Distribution {
    fn record(&mut self, ms: u64) {
        self.count += 1;
        self.sum += ms as u128;
        self.max = self.max.max(ms);
        let shift = (64 - ms.leading_zeros()).saturating_sub(8);
        *self.buckets.entry((ms >> shift) << shift).or_default() += 1;
    }

    fn merge(&mut self, o: &Distribution) {
        self.count += o.count;
        self.sum += o.sum;
        self.max = self.max.max(o.max);
        for (v, n) in &o.buckets {
            *self.buckets.entry(*v).or_default() += n;
        }
    }

    fn json(&self) -> Value {
        let rank = (self.count * 9).div_ceil(10);
        let mut seen = 0;
        let p90 = self.buckets.iter().find_map(|(value, n)| {
            seen += n;
            (seen >= rank).then_some(*value)
        }).unwrap_or(0);
        json!({"max": self.max, "avg": if self.count == 0 { 0.0 } else { self.sum as f64 / self.count as f64 }, "p90": p90})
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct ClusterUsage {
    total: u64,
    skipped: u64,
    took: Distribution,
}

/// Serializable so a cluster can merge every node's counters
/// (`CCSTelemetrySnapshot.add`).
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct CcsStats {
    total: u64,
    success: u64,
    skipped_requests: u64,
    took: Distribution,
    mrt_true: Distribution,
    mrt_false: Distribution,
    remote_max: u64,
    remote_sum: u64,
    failures: BTreeMap<String, u64>,
    features: BTreeMap<String, u64>,
    clusters: BTreeMap<String, ClusterUsage>,
}

impl CcsStats {
    pub fn record(&mut self, ms: u64, aliases: &[String], skipped: &[String], success: bool, minimize_roundtrips: bool, wildcard: bool) {
        self.total += 1;
        self.success += u64::from(success);
        if !success { *self.failures.entry("other".into()).or_default() += 1; }
        if success {
            self.took.record(ms);
            if minimize_roundtrips { self.mrt_true.record(ms); } else { self.mrt_false.record(ms); }
        }
        if !skipped.is_empty() { self.skipped_requests += 1; }
        if minimize_roundtrips { *self.features.entry("mrt_on".into()).or_default() += 1; }
        if wildcard { *self.features.entry("wildcards".into()).or_default() += 1; }
        self.remote_max = self.remote_max.max(aliases.len() as u64);
        self.remote_sum += aliases.len() as u64;
        for alias in aliases {
            if skipped.contains(alias) {
                self.clusters.entry(alias.clone()).or_default().skipped += 1;
            } else if success {
                let c = self.clusters.entry(alias.clone()).or_default();
                c.total += 1;
                c.took.record(ms);
            }
        }
    }

    /// Add another node's counters into these.
    pub fn merge(&mut self, o: &CcsStats) {
        self.total += o.total;
        self.success += o.success;
        self.skipped_requests += o.skipped_requests;
        self.took.merge(&o.took);
        self.mrt_true.merge(&o.mrt_true);
        self.mrt_false.merge(&o.mrt_false);
        self.remote_max = self.remote_max.max(o.remote_max);
        self.remote_sum += o.remote_sum;
        for (k, n) in &o.failures {
            *self.failures.entry(k.clone()).or_default() += n;
        }
        for (k, n) in &o.features {
            *self.features.entry(k.clone()).or_default() += n;
        }
        for (alias, c) in &o.clusters {
            let mine = self.clusters.entry(alias.clone()).or_default();
            mine.total += c.total;
            mine.skipped += c.skipped;
            mine.took.merge(&c.took);
        }
    }

    pub fn json(&self) -> Value {
        let clusters: BTreeMap<_, _> = self.clusters.iter().map(|(alias, c)| (alias, json!({"total": c.total, "skipped": c.skipped, "took": c.took.json()}))).collect();
        json!({"total": self.total, "success": self.success, "skipped": self.skipped_requests,
            "took": self.took.json(), "took_mrt_true": self.mrt_true.json(), "took_mrt_false": self.mrt_false.json(),
            "remotes_per_search_max": self.remote_max, "remotes_per_search_avg": if self.total == 0 { 0.0 } else { self.remote_sum as f64 / self.total as f64 },
            "failure_reasons": self.failures, "features": self.features, "clients": {}, "clusters": clusters})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_adds_counts_and_keeps_distributions() {
        let (mut a, mut b) = (CcsStats::default(), CcsStats::default());
        a.record(10, &["one".into()], &[], true, true, false);
        b.record(30, &["one".into(), "two".into()], &["two".into()], true, false, true);
        // Round-trip through the wire form, as a remote node's counters arrive.
        let b: CcsStats = serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
        a.merge(&b);
        let j = a.json();
        assert_eq!(j["total"], 2);
        assert_eq!(j["success"], 2);
        assert_eq!(j["skipped"], 1);
        assert_eq!(j["took"]["max"], 30);
        assert_eq!(j["took"]["avg"], 20.0);
        assert_eq!(j["remotes_per_search_max"], 2);
        assert_eq!(j["remotes_per_search_avg"], 1.5);
        assert_eq!(j["clusters"]["one"]["total"], 2);
        assert_eq!(j["clusters"]["two"]["skipped"], 1);
        assert_eq!(j["features"]["wildcards"], 1);
    }
}
