//! Operating-system calls that differ between Unix and Windows.
//!
//! Linux is the production target; macOS and Windows are supported for
//! development. Everything here either does the same thing on every
//! platform or reports "unavailable" (`None`) where a platform has no
//! equivalent, as ES does for probes it cannot run.

use std::path::Path;

/// Make a rename or a new entry in `dir` durable (fsync the directory).
/// Windows has no directory handle to fsync (NTFS journals the directory
/// change itself), so there it only checks that `dir` exists.
#[cfg(unix)]
pub fn sync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
pub fn sync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::metadata(dir).map(|_| ())
}

/// fsync the file at `path`. Windows flushes only through a handle opened
/// for writing, so the file is opened that way there.
#[cfg(unix)]
pub fn sync_file(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

#[cfg(not(unix))]
pub fn sync_file(path: &Path) -> std::io::Result<()> {
    std::fs::OpenOptions::new().write(true).open(path)?.sync_all()
}

/// Bytes `(total, free, available to this process)` of the filesystem
/// holding `path`.
#[cfg(windows)]
pub fn disk_space(path: &Path) -> Option<(u64, u64, u64)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let (mut available, mut total, mut free) = (0u64, 0u64, 0u64);
    // SAFETY: `wide` is NUL-terminated; the out-pointers are valid u64s.
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut available, &mut total, &mut free) };
    (ok != 0).then_some((total, free, available))
}

/// Physical memory `(total, available)` and page file `(total, available)`.
#[cfg(windows)]
pub fn memory() -> Option<(u64, u64, u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: a zeroed MEMORYSTATUSEX with its length set is a valid argument.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) };
    (ok != 0).then_some((m.ullTotalPhys, m.ullAvailPhys, m.ullTotalPageFile, m.ullAvailPageFile))
}

/// The page size.
#[cfg(windows)]
pub fn page_size() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    // SAFETY: GetSystemInfo fills the zeroed out-parameter.
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    unsafe { GetSystemInfo(&mut info) };
    if info.dwPageSize > 0 { info.dwPageSize as u64 } else { 4096 }
}

/// CPU time (user + kernel) used by this process, in milliseconds.
#[cfg(windows)]
pub fn process_cpu_millis() -> u64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: the pseudo-handle of the current process; valid out-pointers.
    let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    if ok == 0 {
        return 0;
    }
    // FILETIMEs count 100 ns ticks.
    let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    (ticks(kernel) + ticks(user)) / 10_000
}

/// This process's resident memory `(current, peak)` in bytes.
#[cfg(windows)]
pub fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: a zeroed counters struct with its size passed as `cb`.
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, size) };
    if ok == 0 { (0, 0) } else { (c.WorkingSetSize as u64, c.PeakWorkingSetSize as u64) }
}
