//! Real OS / process numbers for `_nodes/stats`, `_cluster/stats` and the
//! `_cat` APIs (libc; Linux `/proc` or macOS `sysctl`). Anything the
//! platform cannot tell us falls back to a conservative estimate.

use std::sync::Mutex;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct Sys {
    pub total_mem: u64,
    pub free_mem: u64,
    pub swap_total: u64,
    pub swap_free: u64,
    pub load: [f64; 3],
    pub cpus: usize,
    /// Process CPU time (user + system), millis.
    pub cpu_millis: u64,
    /// Process CPU percent since the previous sample.
    pub process_cpu_percent: u64,
    /// System CPU percent (load-average based estimate).
    pub os_cpu_percent: u64,
    pub rss: u64,
    /// Linux resident classes; absent when the OS does not expose them.
    pub rss_anon: Option<u64>,
    pub rss_file: Option<u64>,
    pub rss_shmem: Option<u64>,
    pub virt: u64,
    pub open_fds: u64,
    pub max_fds: u64,
    pub threads: u64,
    pub os_name: String,
    pub os_pretty_name: String,
    pub os_version: String,
}

#[derive(Clone, Debug, Default)]
pub struct Fs {
    pub total: u64,
    pub free: u64,
    pub available: u64,
    pub mount: String,
    pub fs_type: String,
    /// Block device backing the mount (`/dev/nvme0n1p1`), when known.
    pub device: Option<String>,
}

/// Cumulative `/proc/diskstats` counters of one block device.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DiskIo {
    pub read_ops: u64,
    pub write_ops: u64,
    pub read_kb: u64,
    pub write_kb: u64,
    pub io_time_ms: u64,
}

fn diskstats(dev: &str) -> Option<DiskIo> {
    let t = read("/proc/diskstats")?;
    for l in t.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() >= 13 && f[2] == dev {
            let n = |i: usize| f[i].parse::<u64>().unwrap_or(0);
            return Some(DiskIo { read_ops: n(3), read_kb: n(5) / 2, write_ops: n(7), write_kb: n(9) / 2, io_time_ms: n(12) });
        }
    }
    None
}

static DISK_BASE: Mutex<Option<(String, DiskIo)>> = Mutex::new(None);

/// I/O counters of `device` since the node started (ES `FsInfo.IoStats`),
/// as `(device name, counters)`. `None` when the platform has no
/// `/proc/diskstats` entry for it.
pub fn disk_io(device: &str) -> Option<(String, DiskIo)> {
    let name = device.rsplit('/').next().unwrap_or(device).to_string();
    let cur = diskstats(&name)?;
    let mut base = DISK_BASE.lock().unwrap_or_else(|e| e.into_inner());
    let b = match &*base {
        Some((n, b)) if *n == name => *b,
        _ => {
            *base = Some((name.clone(), cur));
            cur
        }
    };
    Some((
        name,
        DiskIo {
            read_ops: cur.read_ops.saturating_sub(b.read_ops),
            write_ops: cur.write_ops.saturating_sub(b.write_ops),
            read_kb: cur.read_kb.saturating_sub(b.read_kb),
            write_kb: cur.write_kb.saturating_sub(b.write_kb),
            io_time_ms: cur.io_time_ms.saturating_sub(b.io_time_ms),
        },
    ))
}

/// Linux control-group figures (`_nodes/stats` `os.cgroup`), cgroup v2 or v1.
#[derive(Clone, Debug, Default)]
pub struct Cgroup {
    pub cpuacct_group: String,
    pub usage_nanos: u64,
    pub cpu_group: String,
    pub cfs_period_micros: i64,
    pub cfs_quota_micros: i64,
    pub elapsed_periods: u64,
    pub times_throttled: u64,
    pub throttled_nanos: u64,
    pub memory_group: String,
    pub memory_limit: String,
    pub memory_usage: String,
}

pub fn cgroup() -> Option<Cgroup> {
    let text = read("/proc/self/cgroup")?;
    let kv = |s: &str| -> std::collections::HashMap<String, u64> {
        s.lines().filter_map(|l| l.split_once(' ').and_then(|(k, v)| v.trim().parse().ok().map(|v| (k.to_string(), v)))).collect()
    };
    // cgroup v2: a single `0::/path` line.
    if let Some(path) = text.lines().find_map(|l| l.strip_prefix("0::")) {
        let base = format!("/sys/fs/cgroup{}", if path == "/" { "" } else { path });
        let stat = kv(&read(&format!("{base}/cpu.stat")).unwrap_or_default());
        let (quota, period) = read(&format!("{base}/cpu.max"))
            .and_then(|s| {
                let mut it = s.split_whitespace();
                let q = it.next()?;
                let p = it.next()?.parse::<i64>().ok()?;
                Some((if q == "max" { -1 } else { q.parse().unwrap_or(-1) }, p))
            })
            .unwrap_or((-1, 100_000));
        let g = path.to_string();
        return Some(Cgroup {
            cpuacct_group: g.clone(),
            usage_nanos: stat.get("usage_usec").copied().unwrap_or(0) * 1000,
            cpu_group: g.clone(),
            cfs_period_micros: period,
            cfs_quota_micros: quota,
            elapsed_periods: stat.get("nr_periods").copied().unwrap_or(0),
            times_throttled: stat.get("nr_throttled").copied().unwrap_or(0),
            throttled_nanos: stat.get("throttled_usec").copied().unwrap_or(0) * 1000,
            memory_group: g,
            memory_limit: read(&format!("{base}/memory.max")).map(|s| s.trim().to_string()).unwrap_or_else(|| "max".into()),
            memory_usage: read(&format!("{base}/memory.current")).map(|s| s.trim().to_string()).unwrap_or_else(|| "0".into()),
        });
    }
    // cgroup v1: `id:controllers:path` lines.
    let mut groups = std::collections::HashMap::new();
    for l in text.lines() {
        let mut it = l.splitn(3, ':');
        let (_, ctl, p) = (it.next()?, it.next()?, it.next()?);
        for c in ctl.split(',') {
            groups.insert(c.to_string(), p.to_string());
        }
    }
    let g = |c: &str| groups.get(c).cloned().unwrap_or_else(|| "/".into());
    let v1 = |ctl: &str, f: &str| read(&format!("/sys/fs/cgroup/{ctl}{}/{f}", if g(ctl) == "/" { String::new() } else { g(ctl) })).map(|s| s.trim().to_string());
    let stat = kv(&v1("cpu", "cpu.stat").unwrap_or_default());
    Some(Cgroup {
        cpuacct_group: g("cpuacct"),
        usage_nanos: v1("cpuacct", "cpuacct.usage").and_then(|s| s.parse().ok()).unwrap_or(0),
        cpu_group: g("cpu"),
        cfs_period_micros: v1("cpu", "cpu.cfs_period_us").and_then(|s| s.parse().ok()).unwrap_or(100_000),
        cfs_quota_micros: v1("cpu", "cpu.cfs_quota_us").and_then(|s| s.parse().ok()).unwrap_or(-1),
        elapsed_periods: stat.get("nr_periods").copied().unwrap_or(0),
        times_throttled: stat.get("nr_throttled").copied().unwrap_or(0),
        throttled_nanos: stat.get("throttled_time").copied().unwrap_or(0),
        memory_group: g("memory"),
        memory_limit: v1("memory", "memory.limit_in_bytes").unwrap_or_else(|| "max".into()),
        memory_usage: v1("memory", "memory.usage_in_bytes").unwrap_or_else(|| "0".into()),
    })
}

fn read(p: &str) -> Option<String> {
    std::fs::read_to_string(p).ok()
}

#[cfg(windows)]
fn page_size() -> u64 {
    elasticsearch_engine::rustic_core::platform::page_size()
}

#[cfg(unix)]
fn page_size() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 { p as u64 } else { 4096 }
}

#[cfg(target_os = "macos")]
fn sysctl_raw(name: &str, buf: &mut [u8]) -> Option<usize> {
    let c = std::ffi::CString::new(name).ok()?;
    let mut len = buf.len();
    // SAFETY: buf/len describe a valid writable buffer.
    let r = unsafe { libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr() as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) };
    if r == 0 { Some(len) } else { None }
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let mut b = [0u8; 8];
    let n = sysctl_raw(name, &mut b)?;
    Some(match n {
        4 => u32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as u64,
        8 => u64::from_ne_bytes(b),
        _ => return None,
    })
}

#[cfg(unix)]
fn meminfo(key: &str, text: &str) -> Option<u64> {
    text.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).map(|kb| kb * 1024)
}

/// Keep absent fields distinct from a measured zero (older/non-Linux kernels).
fn resident_classes(status: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    (meminfo("RssAnon:", status), meminfo("RssFile:", status), meminfo("RssShmem:", status))
}

fn memory() -> (u64, u64, u64, u64) {
    #[cfg(target_os = "macos")]
    {
        let total = sysctl_u64("hw.memsize").unwrap_or(0);
        let ps = page_size();
        let free = sysctl_u64("vm.page_free_count").unwrap_or(0) * ps + sysctl_u64("vm.page_speculative_count").unwrap_or(0) * ps;
        let mut b = [0u8; 32];
        let (st, sf) = match sysctl_raw("vm.swapusage", &mut b) {
            Some(n) if n >= 24 => {
                let t = u64::from_ne_bytes(b[0..8].try_into().unwrap());
                let a = u64::from_ne_bytes(b[8..16].try_into().unwrap());
                (t, a)
            }
            _ => (0, 0),
        };
        return (total, free.min(total), st, sf);
    }
    #[cfg(windows)]
    {
        return elasticsearch_engine::rustic_core::platform::memory().unwrap_or_default();
    }
    #[allow(unreachable_code)]
    #[cfg(unix)]
    {
        if let Some(t) = read("/proc/meminfo") {
            let total = meminfo("MemTotal:", &t).unwrap_or(0);
            let free = meminfo("MemAvailable:", &t).or_else(|| meminfo("MemFree:", &t)).unwrap_or(0);
            return (total, free, meminfo("SwapTotal:", &t).unwrap_or(0), meminfo("SwapFree:", &t).unwrap_or(0));
        }
        // SAFETY: sysconf has no preconditions.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let total = if pages > 0 { pages as u64 * page_size() } else { 0 };
        (total, total / 2, 0, 0)
    }
}

#[cfg(windows)]
fn load_average() -> [f64; 3] {
    [0.0; 3]
}

#[cfg(unix)]
fn load_average() -> [f64; 3] {
    let mut l = [0f64; 3];
    // SAFETY: l has room for 3 doubles.
    let n = unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    if n < 3 { [0.0; 3] } else { l }
}

/// CPU milliseconds and peak resident bytes.
#[cfg(windows)]
fn rusage() -> (u64, u64) {
    (elasticsearch_engine::rustic_core::platform::process_cpu_millis(), elasticsearch_engine::rustic_core::platform::process_memory().1)
}

#[cfg(unix)]
fn rusage() -> (u64, u64) {
    // SAFETY: zeroed rusage is a valid out-parameter.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    let ms = (u.ru_utime.tv_sec as u64 + u.ru_stime.tv_sec as u64) * 1000 + (u.ru_utime.tv_usec as u64 + u.ru_stime.tv_usec as u64) / 1000;
    let maxrss = if cfg!(target_os = "macos") { u.ru_maxrss as u64 } else { u.ru_maxrss as u64 * 1024 };
    (ms, maxrss)
}

#[cfg(windows)]
fn fds() -> (u64, u64) {
    (0, 0)
}

#[cfg(unix)]
fn fds() -> (u64, u64) {
    let open = std::fs::read_dir("/proc/self/fd").or_else(|_| std::fs::read_dir("/dev/fd")).map(|d| d.count() as u64).unwrap_or(0);
    // SAFETY: zeroed rlimit is a valid out-parameter.
    let mut r: libc::rlimit = unsafe { std::mem::zeroed() };
    let max = if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut r) } == 0 { r.rlim_cur as u64 } else { 0 };
    (open, if max > i64::MAX as u64 { i64::MAX as u64 } else { max })
}

#[cfg(windows)]
fn os_names() -> (String, String, String) {
    ("Windows".into(), "Windows".into(), String::new())
}

#[cfg(unix)]
fn os_names() -> (String, String, String) {
    // SAFETY: zeroed utsname is a valid out-parameter.
    let mut u: libc::utsname = unsafe { std::mem::zeroed() };
    let release = if unsafe { libc::uname(&mut u) } == 0 {
        // SAFETY: uname NUL-terminates the field.
        unsafe { std::ffi::CStr::from_ptr(u.release.as_ptr()) }.to_string_lossy().into_owned()
    } else {
        String::new()
    };
    if cfg!(target_os = "macos") {
        return ("Mac OS X".into(), "macOS".into(), release);
    }
    let pretty = read("/etc/os-release")
        .and_then(|t| t.lines().find(|l| l.starts_with("PRETTY_NAME=")).map(|l| l["PRETTY_NAME=".len()..].trim_matches('"').to_string()))
        .unwrap_or_else(|| "Linux".into());
    ("Linux".into(), pretty, release)
}

static LAST_CPU: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
/// Apply the node's `node.processors` (see `elasticsearch_engine::processors`).
pub fn configure_processors(node_settings: &serde_json::Map<String, serde_json::Value>) {
    elasticsearch_engine::processors::configure(node_settings);
}

/// Whether `bootstrap.memory_lock` locked this process's memory (ES
/// `NativeAccess.isMemoryLocked`, `_nodes` `process.mlockall`); set by the
/// node launcher after `mlockall`.
static MEMORY_LOCKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_memory_locked(locked: bool) {
    MEMORY_LOCKED.store(locked, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn memory_locked() -> bool {
    MEMORY_LOCKED.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn processors() -> usize {
    elasticsearch_engine::processors::get()
}
static OS_NAMES: OnceLock<(String, String, String)> = OnceLock::new();

/// [`sys`], refreshed at most once a second (ES's `monitor.os` /
/// `monitor.process` / `monitor.jvm` stats are probed on a 1s refresh
/// interval too): `_nodes/stats` does not re-read `/proc` per request.
pub fn sys_cached() -> Sys {
    static CACHE: std::sync::Mutex<Option<(Instant, Sys)>> = std::sync::Mutex::new(None);
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((t, s)) = c.as_ref() {
        if t.elapsed() < std::time::Duration::from_secs(1) {
            return s.clone();
        }
    }
    let s = sys();
    *c = Some((Instant::now(), s.clone()));
    s
}

/// ES's `monitor.fs.refresh_interval` / `monitor.os.refresh_interval`
/// (default and minimum 1s): `FsService` and `OsService` keep their probes'
/// last result in a `SingleObjectCache` for that long.
pub const MONITOR_REFRESH: std::time::Duration = std::time::Duration::from_secs(1);

/// `slot`'s value when it was probed less than `max_age` before `now`, else
/// a fresh `probe()` stored with `now` (ES `SingleObjectCache.getOrRefresh`).
fn cached<T: Clone>(slot: &mut Option<(Instant, T)>, now: Instant, max_age: std::time::Duration, probe: impl FnOnce() -> T) -> T {
    if let Some((at, value)) = slot.as_ref() {
        if now.saturating_duration_since(*at) < max_age {
            return value.clone();
        }
    }
    let value = probe();
    *slot = Some((now, value.clone()));
    value
}

/// [`fs`] of `path` and its device's [`disk_io`], probed at most once per
/// [`MONITOR_REFRESH`] per path (ES `FsService`'s cached `FsInfo`): node
/// stats does not `statvfs`, read `/proc/mounts` and `/proc/diskstats` per
/// request.
pub fn fs_cached(path: &Path) -> (Fs, Option<(String, DiskIo)>) {
    type Entry = Option<(Instant, (Fs, Option<(String, DiskIo)>))>;
    static CACHE: OnceLock<Mutex<std::collections::HashMap<std::path::PathBuf, Entry>>> = OnceLock::new();
    let mut map = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    let slot = map.entry(path.to_path_buf()).or_default();
    cached(slot, Instant::now(), MONITOR_REFRESH, || {
        let f = fs(path);
        let io = f.device.as_deref().and_then(disk_io);
        (f, io)
    })
}

/// [`cgroup`], re-read at most once per [`MONITOR_REFRESH`] (part of ES's
/// cached `OsStats`).
pub fn cgroup_cached() -> Option<Cgroup> {
    static CACHE: Mutex<Option<(Instant, Option<Cgroup>)>> = Mutex::new(None);
    let mut slot = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cached(&mut *slot, Instant::now(), MONITOR_REFRESH, cgroup)
}

pub fn sys() -> Sys {
    let (total_mem, free_mem, swap_total, swap_free) = memory();
    let load = load_average();
    let cpus = processors();
    let (cpu_millis, maxrss) = rusage();
    let ps = page_size();
    let (mut rss, mut virt) = (maxrss, maxrss);
    let mut threads = (cpus + 4) as u64;
    let (mut rss_anon, mut rss_file, mut rss_shmem) = (None, None, None);
    if let Some(s) = read("/proc/self/statm") {
        let f: Vec<u64> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if f.len() >= 2 {
            virt = f[0] * ps;
            rss = f[1] * ps;
        }
    }
    if let Some(s) = read("/proc/self/status") {
        (rss_anon, rss_file, rss_shmem) = resident_classes(&s);
        if let Some(t) = s.lines().find(|l| l.starts_with("Threads:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()) {
            threads = t;
        }
    }
    let (open_fds, max_fds) = fds();
    let now = Instant::now();
    let process_cpu_percent = {
        let mut last = LAST_CPU.lock().unwrap_or_else(|e| e.into_inner());
        let pct = match *last {
            Some((t, c)) => {
                let wall = now.duration_since(t).as_millis() as u64;
                if wall == 0 { 0 } else { ((cpu_millis.saturating_sub(c)) * 100 / wall / cpus as u64).min(100) }
            }
            None => 0,
        };
        *last = Some((now, cpu_millis));
        pct
    };
    let os_cpu_percent = ((load[0] / cpus as f64) * 100.0).round().clamp(0.0, 100.0) as u64;
    let (os_name, os_pretty_name, os_version) = OS_NAMES.get_or_init(os_names).clone();
    Sys {
        total_mem,
        free_mem,
        swap_total,
        swap_free,
        load,
        cpus,
        cpu_millis,
        process_cpu_percent,
        os_cpu_percent,
        rss,
        rss_anon,
        rss_file,
        rss_shmem,
        virt,
        open_fds,
        max_fds,
        threads,
        os_name,
        os_pretty_name,
        os_version,
    }
}

/// Filesystem holding `path`.
pub fn fs(path: &Path) -> Fs {
    let mut out = Fs { mount: "/".into(), fs_type: if cfg!(target_os = "macos") { "apfs".into() } else if cfg!(windows) { "NTFS".into() } else { "ext4".into() }, ..Default::default() };
    #[cfg(windows)]
    {
        if let Some((total, free, available)) = elasticsearch_engine::rustic_core::platform::disk_space(path) {
            out.total = total;
            out.free = free;
            out.available = available;
        }
    }
    #[cfg(unix)]
    {
        let Ok(c) = std::ffi::CString::new(path.to_string_lossy().as_bytes().to_vec()) else { return out };
        // SAFETY: zeroed statvfs is a valid out-parameter; c is NUL-terminated.
        let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0 {
            let f = s.f_frsize as u64;
            out.total = s.f_blocks as u64 * f;
            out.free = s.f_bfree as u64 * f;
            out.available = s.f_bavail as u64 * f;
        }
        if let Some(m) = read("/proc/mounts") {
            // Longest mount point that prefixes the path.
            let p = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            let mut best: Option<(String, String, String)> = None;
            for l in m.lines() {
                let parts: Vec<&str> = l.split_whitespace().collect();
                if parts.len() >= 3 && p.starts_with(parts[1]) && best.as_ref().map(|b| parts[1].len() > b.0.len()).unwrap_or(true) {
                    best = Some((parts[1].to_string(), parts[2].to_string(), parts[0].to_string()));
                }
            }
            if let Some((mnt, ty, dev)) = best {
                out.mount = mnt;
                out.fs_type = ty;
                out.device = Some(dev);
            }
        }
    }
    out
}

/// Compatibility heap fields: this native process has no JVM heap. Report
/// anonymous resident bytes without scaling or clipping (including usage above
/// the configured compatibility maximum). On platforms without a resident
/// breakdown, the fallback is explicitly the engine's accounted allocations.
/// `process.mem` exposes the OS resident classes and actual native budget.
pub fn heap(s: &Sys) -> (u64, u64) {
    (s.rss_anon.unwrap_or_else(|| elasticsearch_engine::memory::breakers().accounted()),
        elasticsearch_engine::memory::es_heap())
}

#[cfg(test)]
mod resident_tests {
    use super::*;
    #[test]
    fn compatibility_heap_never_scales_or_clips_native_anonymous_bytes() {
        let sample = Sys { rss: 8 << 30, rss_anon: Some(u64::MAX / 2), ..Sys::default() };
        assert_eq!(heap(&sample).0, u64::MAX / 2);
        let file_only = Sys { rss: 8 << 30, rss_anon: Some(0), ..Sys::default() };
        assert_eq!(heap(&file_only).0, 0);
    }
    #[test]
    fn resident_classes_are_bytes_and_unknown_is_not_zero() {
        assert_eq!(resident_classes("RssAnon: 128 kB\nRssFile: 2048 kB\nRssShmem: 0 kB\n"),
            (Some(128 * 1024), Some(2048 * 1024), Some(0)));
        assert_eq!(resident_classes("VmRSS: 4096 kB\n"), (None, None, None));
    }
}

#[cfg(test)]
mod monitor_cache_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cached_probe_is_reused_until_the_refresh_interval_passes() {
        let mut slot: Option<(Instant, u32)> = None;
        let mut probes = 0u32;
        let t0 = Instant::now();
        let mut probe = |now: Instant, slot: &mut Option<(Instant, u32)>| cached(slot, now, MONITOR_REFRESH, || { probes += 1; probes });
        assert_eq!(probe(t0, &mut slot), 1);
        // Within the interval: the first sample, not a new probe.
        assert_eq!(probe(t0 + Duration::from_millis(999), &mut slot), 1);
        // A clock read before the sample (another thread's `now`) reuses it too.
        assert_eq!(probe(t0, &mut slot), 1);
        // At the interval: probed again, and that sample is kept.
        assert_eq!(probe(t0 + MONITOR_REFRESH, &mut slot), 2);
        assert_eq!(probe(t0 + MONITOR_REFRESH + Duration::from_millis(500), &mut slot), 2);
        assert_eq!(probe(t0 + 3 * MONITOR_REFRESH, &mut slot), 3);
    }

    #[test]
    fn cached_filesystem_probe_matches_the_direct_probe() {
        let dir = tempfile::tempdir().unwrap();
        let direct = fs(dir.path());
        let (first, io) = fs_cached(dir.path());
        let (again, io_again) = fs_cached(dir.path());
        // Same mount, type, device and size as an uncached probe.
        assert_eq!((first.mount.as_str(), first.fs_type.as_str(), first.device.as_deref(), first.total), (direct.mount.as_str(), direct.fs_type.as_str(), direct.device.as_deref(), direct.total));
        // Within the refresh interval the sample (free space, io counters) is reused.
        assert_eq!((first.free, first.available), (again.free, again.available));
        assert_eq!(io, io_again);
        // Another path is probed for itself.
        let other = tempfile::tempdir().unwrap();
        assert_eq!(fs_cached(other.path()).0.total, fs(other.path()).total);
    }

    #[test]
    fn cached_cgroup_matches_the_direct_read() {
        let direct = cgroup().map(|c| (c.cpu_group, c.memory_group, c.cfs_quota_micros));
        let cached = cgroup_cached().map(|c| (c.cpu_group, c.memory_group, c.cfs_quota_micros));
        assert_eq!(cached, direct);
    }
}
