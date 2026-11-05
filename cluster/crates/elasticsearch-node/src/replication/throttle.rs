//! Recovery throttling (ES `RecoverySettings`): `indices.recovery.*` as of
//! now, and the node-wide byte rate limiter every recovery on a node shares
//! (operation batches here; file chunks reuse it).

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

/// `indices.recovery.max_bytes_per_sec` (ES default `40mb`; `0` / `-1`:
/// unthrottled). Dynamic.
pub const MAX_BYTES_PER_SEC: &str = "indices.recovery.max_bytes_per_sec";
/// `indices.recovery.chunk_size` (ES default `512kb`).
pub const CHUNK_SIZE: &str = "indices.recovery.chunk_size";
/// `indices.recovery.max_concurrent_file_chunks` (ES default 2, at most 8).
pub const MAX_CONCURRENT_FILE_CHUNKS: &str = "indices.recovery.max_concurrent_file_chunks";
/// A file chunk is never larger than this, whatever `chunk_size` says.
pub const MAX_CHUNK: u64 = 16 << 20;
/// The longest one pause may be (well below a recovery batch request's
/// 120s timeout).
pub const MAX_PAUSE: Duration = Duration::from_secs(30);

/// The recovery settings in effect (ES `RecoverySettings`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecoverySettings {
    /// 0: unthrottled.
    pub max_bytes_per_sec: u64,
    pub chunk_size: u64,
    pub max_concurrent_file_chunks: usize,
}

impl Default for RecoverySettings {
    fn default() -> Self {
        RecoverySettings { max_bytes_per_sec: 40 << 20, chunk_size: 512 << 10, max_concurrent_file_chunks: 2 }
    }
}

impl RecoverySettings {
    /// `get`: the effective value of a setting (cluster transient, then
    /// persistent, then the node's).
    pub fn resolve(get: impl Fn(&str) -> Option<String>) -> RecoverySettings {
        let d = RecoverySettings::default();
        // Validated when set (`settings::validate_cluster_update`); a value
        // that still does not parse (a node setting) keeps the default.
        let bytes = |key: &str| get(key).and_then(|v| elasticsearch_engine::settings::parse_byte_size(&v));
        RecoverySettings {
            max_bytes_per_sec: bytes(MAX_BYTES_PER_SEC).map_or(d.max_bytes_per_sec, |b| b.max(0) as u64),
            chunk_size: bytes(CHUNK_SIZE).filter(|b| *b > 0).map_or(d.chunk_size, |b| (b as u64).min(MAX_CHUNK)),
            max_concurrent_file_chunks: get(MAX_CONCURRENT_FILE_CHUNKS).and_then(|v| v.trim().parse::<usize>().ok()).map_or(d.max_concurrent_file_chunks, |n| n.clamp(1, 8)),
        }
    }
}

/// A node-wide byte rate limiter (Lucene `SimpleRateLimiter`, ES
/// `RecoverySettings.rateLimiter()`): every recovery on the node shares
/// it. [`RateLimiter::delay`] returns how long the caller waits before
/// moving `bytes` (it sleeps outside the lock).
#[derive(Default)]
pub struct RateLimiter {
    last: Mutex<Option<Instant>>,
}

impl RateLimiter {
    /// As `SimpleRateLimiter.pause`: the target time is the last one plus
    /// `bytes` at the rate; one already past is no pause (and no credit:
    /// the clock restarts at now).
    pub fn delay(&self, bytes: u64, bytes_per_sec: u64) -> Duration {
        self.delay_capped(bytes, bytes_per_sec, Duration::MAX)
    }

    /// [`RateLimiter::delay`], at most `cap` from now: the clock is not
    /// pushed further either (one oversized transfer, or a rate lowered
    /// since, never queues the others behind it for longer).
    pub fn delay_capped(&self, bytes: u64, bytes_per_sec: u64, cap: Duration) -> Duration {
        if bytes_per_sec == 0 || bytes == 0 {
            return Duration::ZERO;
        }
        let now = Instant::now();
        let step = Duration::from_secs_f64(bytes as f64 / bytes_per_sec as f64);
        let mut last = self.last.lock();
        match last.map(|l| l + step) {
            Some(target) if target > now => {
                let target = if target - now > cap { now + cap } else { target };
                *last = Some(target);
                target - now
            }
            _ => {
                *last = Some(now);
                Duration::ZERO
            }
        }
    }
}

/// The rate limiter of node `node_id` (one per node, in-process clusters
/// included).
pub fn node_limiter(node_id: &str) -> Arc<RateLimiter> {
    static LIMITERS: LazyLock<Mutex<HashMap<String, Arc<RateLimiter>>>> = LazyLock::new(Default::default);
    LIMITERS.lock().entry(node_id.to_string()).or_default().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_resolve_with_es_defaults() {
        assert_eq!(RecoverySettings::resolve(|_| None), RecoverySettings::default());
        assert_eq!(RecoverySettings::resolve(|k| (k == MAX_BYTES_PER_SEC).then(|| "4mb".to_string())).max_bytes_per_sec, 4 << 20);
        assert_eq!(RecoverySettings::resolve(|k| (k == MAX_BYTES_PER_SEC).then(|| "4m".to_string())).max_bytes_per_sec, 4 << 20);
        assert_eq!(RecoverySettings::resolve(|k| (k == MAX_BYTES_PER_SEC).then(|| "0".to_string())).max_bytes_per_sec, 0);
        assert_eq!(RecoverySettings::resolve(|k| (k == MAX_BYTES_PER_SEC).then(|| "-1".to_string())).max_bytes_per_sec, 0);
        assert_eq!(RecoverySettings::resolve(|k| (k == CHUNK_SIZE).then(|| "1gb".to_string())).chunk_size, MAX_CHUNK);
    }

    #[test]
    fn the_limiter_paces_bytes_at_the_rate() {
        let l = RateLimiter::default();
        assert_eq!(l.delay(1 << 20, 0), Duration::ZERO, "0: unthrottled");
        assert_eq!(l.delay(1 << 20, 1 << 20), Duration::ZERO, "the first batch goes at once");
        let d = l.delay(1 << 20, 1 << 20);
        assert!(d > Duration::from_millis(900) && d <= Duration::from_secs(1), "{d:?}");
        let d2 = l.delay(1 << 19, 1 << 20);
        assert!(d2 > d, "pauses accumulate: {d:?} then {d2:?}");
        // An oversized transfer at a low rate waits at most the cap, and
        // the next one is not queued behind the uncapped time.
        let c = RateLimiter::default();
        c.delay_capped(1, 1, MAX_PAUSE);
        assert!(c.delay_capped(1 << 30, 1 << 10, MAX_PAUSE) <= MAX_PAUSE);
        assert!(c.delay_capped(1 << 10, 1 << 10, MAX_PAUSE) <= MAX_PAUSE + Duration::from_secs(2));
    }
}
