//! SLM retention (ES `SnapshotRetentionConfiguration`).

use crate::error::{Result, SnapshotError};
use crate::snapshot::{SnapshotInfo, SnapshotState};
use crate::util::parse_time_value;
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RetentionConfig {
    /// Original string (`30d`) and its millis.
    pub expire_after: Option<(String, i64)>,
    pub min_count: Option<u32>,
    pub max_count: Option<u32>,
}

impl RetentionConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| SnapshotError::illegal_argument("retention must be an object"))?;
        let mut r = Self::default();
        for (k, v) in obj {
            match k.as_str() {
                "expire_after" => {
                    let s = v.as_str().ok_or_else(|| SnapshotError::illegal_argument("expire_after must be a string"))?;
                    r.expire_after = Some((s.to_string(), parse_time_value("expire_after", s)?));
                }
                "min_count" => {
                    r.min_count = Some(
                        v.as_i64()
                            .filter(|n| *n >= 1)
                            .ok_or_else(|| SnapshotError::illegal_argument("minimum snapshot count must be at least 1, but was: ".to_string() + &v.to_string()))?
                            as u32,
                    )
                }
                "max_count" => {
                    r.max_count = Some(
                        v.as_i64()
                            .filter(|n| *n >= 1)
                            .ok_or_else(|| SnapshotError::illegal_argument("maximum snapshot count must be at least 1, but was: ".to_string() + &v.to_string()))?
                            as u32,
                    )
                }
                other => {
                    return Err(SnapshotError::new(
                        "x_content_parse_exception",
                        400,
                        format!("[snapshot_retention] unknown field [{other}]"),
                    ));
                }
            }
        }
        if let (Some(min), Some(max)) = (r.min_count, r.max_count) {
            if max < min {
                return Err(SnapshotError::illegal_argument(format!(
                    "minimum snapshot count {min} cannot be larger than maximum snapshot count {max}"
                )));
            }
        }
        Ok(r)
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        if let Some((s, _)) = &self.expire_after {
            m.insert("expire_after".into(), json!(s));
        }
        if let Some(n) = self.min_count {
            m.insert("min_count".into(), json!(n));
        }
        if let Some(n) = self.max_count {
            m.insert("max_count".into(), json!(n));
        }
        Value::Object(m)
    }

    pub fn is_empty(&self) -> bool {
        self.expire_after.is_none() && self.min_count.is_none() && self.max_count.is_none()
    }

    /// ES `isSnapshotEligibleForDeletion` for one snapshot among `all`
    /// (all completed snapshots of the policy in the repository).
    pub fn is_eligible_for_deletion(&self, si: &SnapshotInfo, all: &[SnapshotInfo], now_millis: i64) -> bool {
        let unsuccessful = |s: &SnapshotInfo| matches!(s.state, Some(SnapshotState::Failed) | Some(SnapshotState::Partial));
        let success = |s: &SnapshotInfo| s.state == Some(SnapshotState::Success);
        let mut sorted: Vec<&SnapshotInfo> = all.iter().collect();
        sorted.sort_by_key(|s| (s.start_time, s.snapshot.clone()));
        let successful_count = all.iter().filter(|s| success(s)).count();
        let newest_successful = all.iter().filter(|s| success(s)).map(|s| s.start_time).max().unwrap_or(i64::MIN);
        let same = |a: &SnapshotInfo| a.uuid == si.uuid;

        // No expire_after and a newer successful snapshot: failed ones go.
        if self.expire_after.is_none() && unsuccessful(si) && newest_successful > si.start_time {
            return true;
        }
        // max_count: the oldest (successful - max) successful snapshots go.
        if let Some(max) = self.max_count {
            let max = max as usize;
            if successful_count > max {
                let to_delete = successful_count - max;
                let mut seen = 0;
                for s in &sorted {
                    if success(s) {
                        seen += 1;
                    }
                    if seen > to_delete {
                        break;
                    }
                    if same(s) {
                        return true;
                    }
                }
            }
        }
        // min_count is a floor regardless of age (for successful snapshots).
        if let Some(min) = self.min_count {
            if successful_count <= min as usize && !unsuccessful(si) {
                return false;
            }
        }
        if let Some((_, expire)) = self.expire_after {
            if let Some(min) = self.min_count {
                let maybe_eligible = if success(si) {
                    sorted
                        .iter()
                        .filter(|s| success(s))
                        .take(successful_count.saturating_sub(min as usize))
                        .any(|s| same(s))
                } else {
                    unsuccessful(si)
                };
                if !maybe_eligible {
                    return false;
                }
            }
            return now_millis - si.start_time > expire;
        }
        false
    }

    /// Snapshots of `policy` (by `metadata.policy`) that retention would delete.
    pub fn snapshots_to_delete(&self, policy_id: &str, snapshots: &[SnapshotInfo], now_millis: i64) -> Vec<SnapshotInfo> {
        if self.is_empty() {
            return vec![];
        }
        let mine: Vec<SnapshotInfo> = snapshots
            .iter()
            .filter(|s| s.slm_policy() == Some(policy_id))
            .filter(|s| s.state.map(|st| st.completed()).unwrap_or(false))
            .cloned()
            .collect();
        let mut out: Vec<SnapshotInfo> =
            mine.iter().filter(|s| self.is_eligible_for_deletion(s, &mine, now_millis)).cloned().collect();
        out.sort_by_key(|s| (s.start_time, s.snapshot.clone()));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const DAY: i64 = 86_400_000;

    fn snap(name: &str, start: i64, state: SnapshotState, policy: &str) -> SnapshotInfo {
        let mut md = Map::new();
        md.insert("policy".into(), json!(policy));
        SnapshotInfo {
            repository: "r".into(),
            snapshot: name.into(),
            uuid: format!("u-{name}"),
            version_id: None,
            version: None,
            indices: vec![],
            data_streams: vec![],
            feature_states: vec![],
            include_global_state: None,
            metadata: Some(md),
            state: Some(state),
            reason: None,
            start_time: start,
            end_time: start + 1,
            total_shards: 0,
            successful_shards: 0,
            failures: vec![],
            index_details: BTreeMap::new(),
        }
    }

    fn names(v: Vec<SnapshotInfo>) -> Vec<String> {
        v.into_iter().map(|s| s.snapshot).collect()
    }

    #[test]
    fn max_count() {
        let r = RetentionConfig::from_json(&json!({"max_count": 2})).unwrap();
        let s: Vec<_> = (0..5).map(|i| snap(&format!("s{i}"), i * DAY, SnapshotState::Success, "p")).collect();
        assert_eq!(names(r.snapshots_to_delete("p", &s, 10 * DAY)), vec!["s0", "s1", "s2"]);
        // other policy's snapshots untouched
        assert!(r.snapshots_to_delete("other", &s, 10 * DAY).is_empty());
    }

    #[test]
    fn expire_after_with_min_count() {
        let r = RetentionConfig::from_json(&json!({"expire_after": "2d", "min_count": 2})).unwrap();
        let s: Vec<_> = (0..5).map(|i| snap(&format!("s{i}"), i * DAY, SnapshotState::Success, "p")).collect();
        // now = day 10: all expired, but keep newest 2.
        assert_eq!(names(r.snapshots_to_delete("p", &s, 10 * DAY)), vec!["s0", "s1", "s2"]);
        // now = day 5: s0,s1,s2 are older than 2d (5-2=3 → s0..s2 have age > 2d).
        assert_eq!(names(r.snapshots_to_delete("p", &s, 5 * DAY)), vec!["s0", "s1", "s2"]);
        // now = day 4.5: only s0..s2 (ages 4.5, 3.5, 2.5).
        assert_eq!(names(r.snapshots_to_delete("p", &s, 4 * DAY + DAY / 2)), vec!["s0", "s1", "s2"]);
        // now = day 3.5: s0, s1 expired.
        assert_eq!(names(r.snapshots_to_delete("p", &s, 3 * DAY + DAY / 2)), vec!["s0", "s1"]);
    }

    #[test]
    fn min_count_floor() {
        let r = RetentionConfig::from_json(&json!({"expire_after": "1d", "min_count": 5})).unwrap();
        let s: Vec<_> = (0..3).map(|i| snap(&format!("s{i}"), i * DAY, SnapshotState::Success, "p")).collect();
        assert!(r.snapshots_to_delete("p", &s, 100 * DAY).is_empty());
    }

    #[test]
    fn failed_snapshots() {
        // No expire_after: failed snapshot older than a successful one is deleted.
        let r = RetentionConfig::from_json(&json!({"max_count": 10})).unwrap();
        let s = vec![
            snap("f0", 0, SnapshotState::Failed, "p"),
            snap("s1", DAY, SnapshotState::Success, "p"),
            snap("f2", 2 * DAY, SnapshotState::Partial, "p"),
        ];
        assert_eq!(names(r.snapshots_to_delete("p", &s, 3 * DAY)), vec!["f0"]);
        // With expire_after + min_count: failed snapshots expire even below min_count.
        let r = RetentionConfig::from_json(&json!({"expire_after": "1d", "min_count": 3})).unwrap();
        assert_eq!(names(r.snapshots_to_delete("p", &s, 5 * DAY)), vec!["f0", "f2"]);
    }

    #[test]
    fn parse_errors() {
        assert!(RetentionConfig::from_json(&json!({"min_count": 0})).is_err());
        let e = RetentionConfig::from_json(&json!({"min_count": 5, "max_count": 2})).unwrap_err();
        assert_eq!(e.reason, "minimum snapshot count 5 cannot be larger than maximum snapshot count 2");
        assert!(RetentionConfig::from_json(&json!({"expire_after": "forever"})).is_err());
        assert_eq!(
            RetentionConfig::from_json(&json!({"expire_after": "30d", "max_count": 3})).unwrap().to_json(),
            json!({"expire_after":"30d","max_count":3})
        );
    }
}
