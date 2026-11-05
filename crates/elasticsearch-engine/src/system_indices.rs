//! Indices owned by a system feature (ES `SystemIndices`): hidden from
//! wildcard expansion and never auto-followed. A dot prefix alone does not
//! make an index hidden (ES `IndexNameExpressionResolver` only skips
//! `index.hidden` and system indices).

const SYSTEM: &[&str] = &[
    ".security", ".tasks", ".kibana", ".async-search", ".enrich-", ".ml-", ".fleet-", ".transform-internal", ".geoip_databases", ".watches",
    ".triggered_watches", ".watcher-history", ".snapshot-blob-cache", ".inference", ".synonyms", ".connector", ".search-app", ".logstash",
    ".apm-", ".reporting-", ".slo-", ".secrets-inference", ".query-rules", ".profiling-",
];

pub fn is_system(name: &str) -> bool {
    name.starts_with('.') && SYSTEM.iter().any(|p| name.starts_with(p))
}

#[cfg(test)]
mod tests {
    #[test]
    fn system_by_feature_prefix_not_by_dot() {
        assert!(super::is_system(".security-7"));
        assert!(super::is_system(".tasks"));
        assert!(!super::is_system(".monitoring-es-7-2026.10.15"));
        assert!(!super::is_system(".my-dot-index"));
        assert!(!super::is_system("security"));
    }
}
