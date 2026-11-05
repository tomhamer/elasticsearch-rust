//! OS / process / filesystem statistics for `_nodes/stats`, `_cat/nodes`,
//! `_cat/allocation` and the disk-threshold allocation decider.
//!
//! Linux (the production target, incl. Kubernetes) is read from `/proc`
//! and `statvfs`; elsewhere we fall back to what portable APIs give and
//! report zeros, which ES itself does when a probe is unavailable.

use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DiskUsage {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub available_bytes: u64,
}

impl DiskUsage {
    pub fn used_percent(&self) -> f64 {
        if self.total_bytes == 0 { 0.0 } else { 100.0 * (self.total_bytes - self.available_bytes.min(self.total_bytes)) as f64 / self.total_bytes as f64 }
    }
}

/// `statvfs` of the filesystem holding `path`.
#[cfg(unix)]
pub fn disk_usage(path: &Path) -> DiskUsage {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = CString::new(path.as_os_str().as_bytes()) else { return DiskUsage::default() };
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: valid NUL-terminated path and an out-pointer to a zeroed struct.
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return DiskUsage::default();
    }
    let frsize = s.f_frsize as u64;
    DiskUsage { total_bytes: s.f_blocks as u64 * frsize, free_bytes: s.f_bfree as u64 * frsize, available_bytes: s.f_bavail as u64 * frsize }
}

/// The volume holding `path` (`GetDiskFreeSpaceExW`).
#[cfg(windows)]
pub fn disk_usage(path: &Path) -> DiskUsage {
    let Some((total_bytes, free_bytes, available_bytes)) = elasticsearch_engine::rustic_core::platform::disk_space(path) else { return DiskUsage::default() };
    DiskUsage { total_bytes, free_bytes, available_bytes }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MemInfo {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub available_bytes: u64,
}

#[cfg(target_os = "linux")]
pub fn mem_info() -> MemInfo {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(k)).and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok()).unwrap_or(0) * 1024;
    let mut m = MemInfo { total_bytes: get("MemTotal:"), free_bytes: get("MemFree:"), available_bytes: get("MemAvailable:") };
    // Inside a container the cgroup limit is what matters.
    if let Some(limit) = cgroup_memory_limit() {
        if limit < m.total_bytes {
            let used = cgroup_memory_usage().unwrap_or(0);
            m.total_bytes = limit;
            m.available_bytes = limit.saturating_sub(used);
            m.free_bytes = m.available_bytes;
        }
    }
    m
}

#[cfg(target_os = "linux")]
fn cgroup_memory_limit() -> Option<u64> {
    let v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok();
    let v1 = || std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes").ok();
    v2.or_else(v1)?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn cgroup_memory_usage() -> Option<u64> {
    let v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.current").ok();
    let v1 = || std::fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes").ok();
    v2.or_else(v1)?.trim().parse().ok()
}

#[cfg(windows)]
pub fn mem_info() -> MemInfo {
    let (total, available, _, _) = elasticsearch_engine::rustic_core::platform::memory().unwrap_or_default();
    MemInfo { total_bytes: total, free_bytes: available, available_bytes: available }
}

#[cfg(all(unix, not(target_os = "linux")))]
pub fn mem_info() -> MemInfo {
    let total = sysctl_u64("hw.memsize").unwrap_or(0);
    MemInfo { total_bytes: total, free_bytes: 0, available_bytes: 0 }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn sysctl_u64(name: &str) -> Option<u64> {
    let out = std::process::Command::new("sysctl").arg("-n").arg(name).output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// 1m / 5m / 15m load averages.
#[cfg(unix)]
pub fn load_average() -> [f64; 3] {
    let mut l = [0f64; 3];
    // SAFETY: getloadavg writes at most 3 doubles into the buffer.
    let n = unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    if n < 3 { [0.0; 3] } else { l }
}

/// No load average on Windows (ES reports none there either).
#[cfg(windows)]
pub fn load_average() -> [f64; 3] {
    [0.0; 3]
}

/// Resident set size and open file descriptors of this process.
pub fn process_resources() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        let rss = std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<u64>().ok()))
            .map(|pages| pages * 4096)
            .unwrap_or(0);
        let fds = std::fs::read_dir("/proc/self/fd").map(|d| d.count() as u64).unwrap_or(0);
        (rss, fds)
    }
    #[cfg(windows)]
    {
        (elasticsearch_engine::rustic_core::platform::process_memory().0, 0)
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: out-pointer to a zeroed rusage.
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        // macOS reports ru_maxrss in bytes.
        (ru.ru_maxrss as u64, 0)
    }
}

#[cfg(unix)]
pub fn max_file_descriptors() -> u64 {
    let mut r: libc::rlimit = unsafe { std::mem::zeroed() };
    // SAFETY: out-pointer to a zeroed rlimit.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut r) } == 0 { r.rlim_cur as u64 } else { 0 }
}

/// No descriptor limit on Windows.
#[cfg(windows)]
pub fn max_file_descriptors() -> u64 {
    0
}

/// Total CPU time (user + system) used by this process, in milliseconds.
#[cfg(windows)]
pub fn process_cpu_millis() -> u64 {
    elasticsearch_engine::rustic_core::platform::process_cpu_millis()
}

/// Total CPU time (user + system) used by this process, in milliseconds.
#[cfg(unix)]
pub fn process_cpu_millis() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: out-pointer to a zeroed rusage.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let ms = |t: libc::timeval| t.tv_sec as u64 * 1000 + t.tv_usec as u64 / 1000;
    ms(ru.ru_utime) + ms(ru.ru_stime)
}

/// Turns cumulative CPU time into a percentage between successive calls,
/// as ES's `process.cpu.percent` / `os.cpu.percent` do.
pub struct CpuSampler {
    last: parking_lot::Mutex<(Instant, u64, u8)>,
}

impl Default for CpuSampler {
    fn default() -> Self {
        CpuSampler { last: parking_lot::Mutex::new((Instant::now(), process_cpu_millis(), 0)) }
    }
}

impl CpuSampler {
    pub fn percent(&self) -> u8 {
        let mut g = self.last.lock();
        let now = Instant::now();
        let elapsed = now.duration_since(g.0).as_millis() as u64;
        if elapsed < 500 {
            return g.2;
        }
        let cpu = process_cpu_millis();
        let cores = std::thread::available_parallelism().map(|n| n.get() as u64).unwrap_or(1);
        let pct = ((cpu.saturating_sub(g.1)) * 100 / (elapsed * cores).max(1)).min(100) as u8;
        *g = (now, cpu, pct);
        pct
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_return_sane_values() {
        let d = disk_usage(Path::new("."));
        assert!(d.total_bytes > 0 && d.available_bytes <= d.total_bytes);
        assert!((0.0..=100.0).contains(&d.used_percent()));
        assert!(max_file_descriptors() > 0);
        let (rss, _) = process_resources();
        assert!(rss > 0);
        let s = CpuSampler::default();
        assert!(s.percent() <= 100);
        assert!(load_average().iter().all(|x| *x >= 0.0));
    }
}
