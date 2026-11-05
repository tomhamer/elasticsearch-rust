//! ES's `Reconfigurator`: chooses the voting configuration the leader
//! should move to, given the live master-eligible nodes.
//!
//! Rules (identical to ES):
//! * candidates are the live, non-retired master-eligible nodes plus the
//!   non-live, non-retired members of the current configuration;
//! * they are ordered live first, then members of the current
//!   configuration (stability), then the current master, then by id;
//! * with `cluster.auto_shrink_voting_configuration: true` the target size
//!   is the number of live non-retired candidates rounded *down* to an odd
//!   number, but never below 3 once the configuration has 3 or more
//!   members (and never below 1);
//! * with auto-shrink disabled, the configuration never shrinks below the
//!   number of its non-retired members (it can still grow);
//! * a configuration that would not have a quorum of live nodes is never
//!   proposed — the current one is kept instead.
//!
//! Retired nodes are those named in the voting configuration exclusions.

use crate::state::VotingConfiguration;
use std::collections::BTreeSet;

fn round_down_to_odd(n: usize) -> usize {
    if n % 2 == 0 { n.saturating_sub(1) } else { n }
}

/// Compute the new configuration. `live` are the ids of live
/// master-eligible nodes (for ES: those in the cluster state that have
/// voted for the current master in its term, plus the master itself).
pub fn reconfigure(live: &BTreeSet<String>, retired: &BTreeSet<String>, current_master: &str, current: &VotingConfiguration, auto_shrink: bool) -> VotingConfiguration {
    // (live, in_current_config, is_master, id) sorted with "true" first.
    let mut candidates: Vec<(bool, bool, bool, String)> = Vec::new();
    for id in live.iter().filter(|id| !retired.contains(*id)) {
        candidates.push((true, current.0.contains(id), id == current_master, id.clone()));
    }
    for id in current.0.iter().filter(|id| !live.contains(*id) && !retired.contains(*id)) {
        candidates.push((false, true, false, id.clone()));
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)).then(a.3.cmp(&b.3)));

    let non_retired_live = candidates.iter().filter(|c| c.0).count();
    let minimum = if current.0.len() < 3 { 1 } else { 3 };
    let mut target = round_down_to_odd(non_retired_live).max(minimum);
    if !auto_shrink {
        let non_retired_in_config = current.0.iter().filter(|id| !retired.contains(*id)).count();
        target = target.max(non_retired_in_config);
    }
    let new_config = VotingConfiguration::new(candidates.into_iter().take(target).map(|c| c.3));
    if new_config.has_quorum(live.iter()) { new_config } else { current.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }
    fn cfg(ids: &[&str]) -> VotingConfiguration {
        VotingConfiguration::new(ids.iter().map(|s| s.to_string()))
    }

    #[test]
    fn grows_to_odd_sizes() {
        let none = BTreeSet::new();
        assert_eq!(reconfigure(&set(&["a", "b"]), &none, "a", &cfg(&["a"]), true), cfg(&["a"]));
        assert_eq!(reconfigure(&set(&["a", "b", "c"]), &none, "a", &cfg(&["a"]), true), cfg(&["a", "b", "c"]));
        assert_eq!(reconfigure(&set(&["a", "b", "c", "d"]), &none, "a", &cfg(&["a", "b", "c"]), true), cfg(&["a", "b", "c"]));
        assert_eq!(reconfigure(&set(&["a", "b", "c", "d", "e"]), &none, "a", &cfg(&["a", "b", "c"]), true), cfg(&["a", "b", "c", "d", "e"]));
    }

    #[test]
    fn shrinks_but_not_below_three() {
        let none = BTreeSet::new();
        let five = cfg(&["a", "b", "c", "d", "e"]);
        assert_eq!(reconfigure(&set(&["a", "b", "c"]), &none, "a", &five, true), cfg(&["a", "b", "c"]));
        // 2 live of 3: keep 3 (dead one stays), quorum still held.
        assert_eq!(reconfigure(&set(&["a", "b"]), &none, "a", &cfg(&["a", "b", "c"]), true), cfg(&["a", "b", "c"]));
        // Without auto-shrink the 5-node config is kept.
        assert_eq!(reconfigure(&set(&["a", "b", "c"]), &none, "a", &five, false), five);
    }

    #[test]
    fn prefers_master_and_current_members() {
        let none = BTreeSet::new();
        // 4 live, current {b,c,d}: keep current members.
        assert_eq!(reconfigure(&set(&["a", "b", "c", "d"]), &none, "a", &cfg(&["b", "c", "d"]), true), cfg(&["b", "c", "d"]));
        // 4 live, none in current config: master d first, then ids a, b.
        assert_eq!(reconfigure(&set(&["a", "b", "c", "d"]), &none, "d", &cfg(&["x"]), true), cfg(&["a", "b", "d"]));
    }

    #[test]
    fn exclusions_and_no_quorum() {
        let five = cfg(&["a", "b", "c", "d", "e"]);
        assert_eq!(reconfigure(&set(&["a", "b", "c", "d", "e"]), &set(&["d", "e"]), "a", &five, true), cfg(&["a", "b", "c"]));
        // Would not have a live quorum: keep current.
        assert_eq!(reconfigure(&set(&["a"]), &BTreeSet::new(), "a", &cfg(&["a", "b", "c"]), true), cfg(&["a", "b", "c"]));
        // Placeholders are replaced by live nodes.
        let p = cfg(&["a", "b", "{bootstrap-placeholder}-n3"]);
        assert_eq!(reconfigure(&set(&["a", "b", "c"]), &BTreeSet::new(), "a", &p, true), cfg(&["a", "b", "c"]));
    }
}
