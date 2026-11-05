//! Small helpers shared by the crate: ES byte-size / time-value parsing,
//! random ids, date rendering, wildcard matching, a rate limiter and a
//! cancellation token.

use crate::error::{Result, SnapshotError};
use base64::Engine;
use parking_lot::Mutex;
use rand::RngCore;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// `UUIDs.randomBase64UUID()` equivalent: 16 random bytes, URL-safe base64, no padding (22 chars).
pub fn random_base64_uuid() -> String {
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    // ES sets version/variant bits like a type-4 UUID.
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

pub fn now_millis() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// `strict_date_optional_time` rendering used by ES for snapshot timestamps:
/// `2020-07-06T21:55:18.129Z`.
pub fn format_millis(millis: i64) -> String {
    match chrono::DateTime::<chrono::Utc>::from_timestamp_millis(millis) {
        Some(dt) => dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
        None => String::new(),
    }
}

/// Human readable duration like ES `TimeValue.toString()` (`1.2s`, `5m`, `150ms`).
pub fn format_duration_millis(millis: i64) -> String {
    let ms = millis.max(0) as f64;
    let (v, unit) = if ms >= 86_400_000.0 {
        (ms / 86_400_000.0, "d")
    } else if ms >= 3_600_000.0 {
        (ms / 3_600_000.0, "h")
    } else if ms >= 60_000.0 {
        (ms / 60_000.0, "m")
    } else if ms >= 1000.0 {
        (ms / 1000.0, "s")
    } else {
        return format!("{}ms", millis.max(0));
    };
    let s = format!("{:.1}", v);
    let s = s.strip_suffix(".0").unwrap_or(&s).to_string();
    format!("{s}{unit}")
}

/// Parse an ES `ByteSizeValue` (`40mb`, `1gb`, `512`, `100b`, `1.5gb`).
/// `-1` parses to `None` (unbounded).
pub fn parse_byte_size(setting: &str, value: &str) -> Result<Option<u64>> {
    let v = value.trim().to_ascii_lowercase();
    if v == "-1" {
        return Ok(None);
    }
    let units: [(&str, u64); 11] = [
        ("pb", 1 << 50),
        ("tb", 1 << 40),
        ("gb", 1 << 30),
        ("mb", 1 << 20),
        ("kb", 1 << 10),
        ("p", 1 << 50),
        ("t", 1 << 40),
        ("g", 1 << 30),
        ("m", 1 << 20),
        ("k", 1 << 10),
        ("b", 1),
    ];
    let err = || {
        SnapshotError::new(
            "parse_exception",
            400,
            format!(
                "failed to parse setting [{setting}] with value [{value}] as a size in bytes: unit is missing or unrecognized"
            ),
        )
    };
    for (suffix, mult) in units {
        if let Some(num) = v.strip_suffix(suffix) {
            let num = num.trim();
            if let Ok(n) = num.parse::<u64>() {
                return Ok(Some(n.saturating_mul(mult)));
            }
            if let Ok(f) = num.parse::<f64>() {
                if f >= 0.0 {
                    return Ok(Some((f * mult as f64) as u64));
                }
            }
            return Err(err());
        }
    }
    if v == "0" {
        return Ok(Some(0));
    }
    // ES requires a unit except for 0 / -1, but accept bare integers (as bytes).
    v.parse::<u64>().map(Some).map_err(|_| err())
}

/// Parse an ES `TimeValue` (`30m`, `1h`, `10s`, `500ms`, `7d`) into milliseconds.
pub fn parse_time_value(setting: &str, value: &str) -> Result<i64> {
    let v = value.trim().to_ascii_lowercase();
    let units: [(&str, f64); 7] = [
        ("nanos", 1e-6),
        ("micros", 1e-3),
        ("ms", 1.0),
        ("s", 1000.0),
        ("m", 60_000.0),
        ("h", 3_600_000.0),
        ("d", 86_400_000.0),
    ];
    for (suffix, mult) in units {
        if let Some(num) = v.strip_suffix(suffix) {
            if let Ok(n) = num.trim().parse::<f64>() {
                if n >= 0.0 || v == "-1" {
                    return Ok((n * mult) as i64);
                }
            }
            break;
        }
    }
    if v == "-1" {
        return Ok(-1);
    }
    if v == "0" {
        return Ok(0);
    }
    Err(SnapshotError::new(
        "parse_exception",
        400,
        format!("failed to parse setting [{setting}] with value [{value}] as a time value: unit is missing or unrecognized"),
    ))
}

/// ES `Regex.simpleMatch`: `*` matches any sequence.
pub fn simple_match(pattern: &str, s: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == s;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let first = parts[0];
    let last = parts[parts.len() - 1];
    if !s.starts_with(first) {
        return false;
    }
    let mut pos = first.len();
    for mid in &parts[1..parts.len() - 1] {
        match s[pos..].find(mid) {
            Some(i) => pos += i + mid.len(),
            None => return false,
        }
    }
    s.len() >= pos + last.len() && s[pos..].ends_with(last)
}

pub fn is_wildcard(s: &str) -> bool {
    s.contains('*')
}

/// Resolve a comma-separated / list of name expressions (`_all`, `*`,
/// wildcards, `-exclusions`) against available names. Returns names in the
/// order of `available`. Unknown concrete names are returned in `missing`.
pub fn resolve_names(expressions: &[String], available: &[String]) -> (Vec<String>, Vec<String>) {
    let mut selected: Vec<bool> = vec![false; available.len()];
    let mut missing = Vec::new();
    let exprs: Vec<&str> = expressions.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if exprs.is_empty() || (exprs.len() == 1 && exprs[0] == "_all") {
        return (available.to_vec(), missing);
    }
    let mut seen_inclusion = false;
    for e in exprs {
        if let Some(neg) = e.strip_prefix('-').filter(|_| seen_inclusion) {
            for (i, a) in available.iter().enumerate() {
                if simple_match(neg, a) {
                    selected[i] = false;
                }
            }
            continue;
        }
        seen_inclusion = true;
        if e == "_all" || e == "*" {
            selected.iter_mut().for_each(|s| *s = true);
        } else if is_wildcard(e) {
            for (i, a) in available.iter().enumerate() {
                if simple_match(e, a) {
                    selected[i] = true;
                }
            }
        } else {
            match available.iter().position(|a| a == e) {
                Some(i) => selected[i] = true,
                None => missing.push(e.to_string()),
            }
        }
    }
    let out = available.iter().zip(selected).filter(|(_, s)| *s).map(|(a, _)| a.clone()).collect();
    (out, missing)
}

/// Token-bucket-ish rate limiter equivalent to Lucene's `SimpleRateLimiter`
/// used by ES for `max_snapshot_bytes_per_sec` / `max_restore_bytes_per_sec`.
#[derive(Debug)]
pub struct RateLimiter {
    bytes_per_sec: Option<u64>,
    next: Mutex<Instant>,
    throttled_nanos: std::sync::atomic::AtomicU64,
}

impl RateLimiter {
    /// `None` or `Some(0)` = unlimited.
    pub fn new(bytes_per_sec: Option<u64>) -> Self {
        Self {
            bytes_per_sec: bytes_per_sec.filter(|b| *b > 0),
            next: Mutex::new(Instant::now()),
            throttled_nanos: Default::default(),
        }
    }

    pub fn bytes_per_sec(&self) -> Option<u64> {
        self.bytes_per_sec
    }

    /// Total time spent throttled so far.
    pub fn throttled(&self) -> Duration {
        Duration::from_nanos(self.throttled_nanos.load(Ordering::Relaxed))
    }

    /// Account for `bytes` and sleep if we are ahead of the allowed rate.
    pub async fn acquire(&self, bytes: u64) {
        let Some(rate) = self.bytes_per_sec else { return };
        let wake = {
            let mut next = self.next.lock();
            let now = Instant::now();
            let start = if *next > now { *next } else { now };
            let cost = Duration::from_secs_f64(bytes as f64 / rate as f64);
            *next = start + cost;
            *next
        };
        let now = Instant::now();
        // Allow a small burst (5ms) before sleeping, like Lucene's MIN_PAUSE_CHECK_MSEC.
        if wake > now + Duration::from_millis(5) {
            let d = wake - now;
            self.throttled_nanos.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
            tokio::time::sleep(d).await;
        }
    }
}

/// Cooperative cancellation flag with a reason.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    inner: Arc<CancelInner>,
}

#[derive(Debug, Default)]
struct CancelInner {
    cancelled: AtomicBool,
    reason: Mutex<Option<String>>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self, reason: impl Into<String>) {
        *self.inner.reason.lock() = Some(reason.into());
        self.inner.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    pub fn reason(&self) -> Option<String> {
        self.inner.reason.lock().clone()
    }

    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(SnapshotError::cancelled(self.reason().unwrap_or_else(|| "cancelled".into())))
        } else {
            Ok(())
        }
    }
}

/// sha256 hex digest.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(data))
}

/// Slicing-by-8 tables of the reflected IEEE polynomial (0xEDB88320).
const CRC_TABLES: [[u32; 256]; 8] = {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 {
            t[s][i] = (t[s - 1][i] >> 8) ^ t[0][(t[s - 1][i] & 0xff) as usize];
            i += 1;
        }
        s += 1;
    }
    t
};

/// Streaming CRC-32 (IEEE): start from `0xFFFF_FFFF`, feed the pieces in
/// order, invert at the end ([`crc32`]). The same value as the engine's
/// file checksums (`rustic_core::codec::crc32`), so a restore can hand the
/// engine checksums computed while the files were downloaded.
pub fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    let t = &CRC_TABLES;
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let lo = crc ^ u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let hi = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
        crc = t[7][(lo & 0xff) as usize] ^ t[6][((lo >> 8) & 0xff) as usize] ^ t[5][((lo >> 16) & 0xff) as usize] ^ t[4][(lo >> 24) as usize]
            ^ t[3][(hi & 0xff) as usize] ^ t[2][((hi >> 8) & 0xff) as usize] ^ t[1][((hi >> 16) & 0xff) as usize] ^ t[0][(hi >> 24) as usize];
    }
    for &b in chunks.remainder() {
        crc = t[0][((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc
}

/// CRC-32 (IEEE) of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    !crc32_update(0xFFFF_FFFF, data)
}

/// Render settings the way ES renders repository settings: flat keys, string values.
pub fn stringify_settings(settings: &serde_json::Map<String, serde_json::Value>) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    fn walk(prefix: &str, v: &serde_json::Value, out: &mut serde_json::Map<String, serde_json::Value>) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, v) in m {
                    let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    walk(&key, v, out);
                }
            }
            serde_json::Value::String(s) => {
                out.insert(prefix.to_string(), serde_json::Value::String(s.clone()));
            }
            serde_json::Value::Null => {}
            serde_json::Value::Array(a) => {
                out.insert(
                    prefix.to_string(),
                    serde_json::Value::Array(
                        a.iter()
                            .map(|x| match x {
                                serde_json::Value::String(s) => serde_json::Value::String(s.clone()),
                                other => serde_json::Value::String(other.to_string()),
                            })
                            .collect(),
                    ),
                );
            }
            other => {
                out.insert(prefix.to_string(), serde_json::Value::String(other.to_string()));
            }
        }
    }
    walk("", &serde_json::Value::Object(settings.clone()), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_is_ieee_and_streams() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        let data: Vec<u8> = (0..10_007u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let whole = crc32(&data);
        for split in [0, 1, 7, 8, 9, 4096, 10_006] {
            let (a, b) = data.split_at(split);
            assert_eq!(!crc32_update(crc32_update(0xFFFF_FFFF, a), b), whole, "split at {split}");
        }
        // Bitwise reference.
        let mut c = 0xFFFF_FFFFu32;
        for &b in &data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        assert_eq!(!c, whole);
    }

    #[test]
    fn byte_sizes() {
        assert_eq!(parse_byte_size("x", "40mb").unwrap(), Some(40 << 20));
        assert_eq!(parse_byte_size("x", "1GB").unwrap(), Some(1 << 30));
        assert_eq!(parse_byte_size("x", "100b").unwrap(), Some(100));
        assert_eq!(parse_byte_size("x", "-1").unwrap(), None);
        assert_eq!(parse_byte_size("x", "1.5kb").unwrap(), Some(1536));
        assert!(parse_byte_size("x", "abc").is_err());
    }

    #[test]
    fn time_values() {
        assert_eq!(parse_time_value("x", "30m").unwrap(), 1_800_000);
        assert_eq!(parse_time_value("x", "500ms").unwrap(), 500);
        assert_eq!(parse_time_value("x", "7d").unwrap(), 7 * 86_400_000);
        assert!(parse_time_value("x", "7").is_err());
    }

    #[test]
    fn wildcard_and_resolution() {
        assert!(simple_match("snap-*", "snap-1"));
        assert!(simple_match("*-1", "snap-1"));
        assert!(simple_match("s*p*1", "snap-1"));
        assert!(!simple_match("x*", "snap-1"));
        let avail: Vec<String> = ["a1", "a2", "b1"].iter().map(|s| s.to_string()).collect();
        let (sel, missing) = resolve_names(&["a*".into(), "-a2".into(), "zz".into()], &avail);
        assert_eq!(sel, vec!["a1"]);
        assert_eq!(missing, vec!["zz"]);
    }

    #[test]
    fn dates_and_ids() {
        assert_eq!(format_millis(1594072518129), "2020-07-06T21:55:18.129Z");
        assert_eq!(random_base64_uuid().len(), 22);
        assert_eq!(format_duration_millis(1500), "1.5s");
        assert_eq!(format_duration_millis(120_000), "2m");
    }

    #[tokio::test]
    async fn rate_limiter_throttles() {
        let rl = RateLimiter::new(Some(1_000_000));
        let start = Instant::now();
        for _ in 0..5 {
            rl.acquire(20_000).await;
        }
        // 100KB at 1MB/s ≈ 100ms (minus the first slice).
        assert!(start.elapsed() >= Duration::from_millis(60), "{:?}", start.elapsed());
    }
}

/// ES `ByteSizeValue.toString()`: `1.2kb`, `10mb`, `100b`.
pub fn human_bytes(b: u64) -> String {
    let units = [("pb", 1u64 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)];
    for (u, m) in units {
        if b >= m {
            let s = format!("{:.1}", b as f64 / m as f64);
            let s = s.strip_suffix(".0").unwrap_or(&s).to_string();
            return format!("{s}{u}");
        }
    }
    format!("{b}b")
}
