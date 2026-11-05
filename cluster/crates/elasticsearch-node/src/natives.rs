//! ES `NativeAccess` (`PosixNativeAccess`, `LinuxNativeAccess`,
//! `MacNativeAccess`) and the JVM start-up behaviour the bootstrap checks
//! read: process limits (`getrlimit`), `vm.max_map_count`, the file
//! descriptor limit the JVM raises at start (`-XX:+MaxFDLimit`), the user
//! name (`user.name`) and memory locking (`bootstrap.memory_lock`).
//!
//! Values follow ES's `ProcessLimits`: [`UNKNOWN`] (-1) when the platform
//! cannot tell, [`UNLIMITED`] (`Long.MAX_VALUE`) for `RLIM_INFINITY`.

use std::sync::atomic::{AtomicBool, Ordering};

/// ES `ProcessLimits.UNKNOWN`.
pub const UNKNOWN: i64 = -1;
/// ES `ProcessLimits.UNLIMITED` (`Long.MAX_VALUE`).
pub const UNLIMITED: i64 = i64::MAX;

/// The platforms ES distinguishes (`Constants.LINUX` / `Constants.MAC_OS_X`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Linux,
    MacOs,
    Other,
}

impl Platform {
    pub fn current() -> Platform {
        if cfg!(target_os = "linux") {
            Platform::Linux
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else {
            Platform::Other
        }
    }
}

/// ES `ProcessLimits`: `maxThreads`, `maxVirtualMemorySize`, `maxFileSize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessLimits {
    pub max_threads: i64,
    pub max_virtual_memory_size: i64,
    pub max_file_size: i64,
}

/// `isMemoryLocked`: set once `mlockall` succeeded.
static MEMORY_LOCKED: AtomicBool = AtomicBool::new(false);

pub fn is_memory_locked() -> bool {
    MEMORY_LOCKED.load(Ordering::Relaxed)
}

/// The OS message for an errno without Rust's ` (os error N)` suffix
/// (ES logs `strerror(errno)`).
#[cfg(unix)]
fn strerror(err: &std::io::Error) -> String {
    let s = err.to_string();
    match s.rfind(" (os error ") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

/// `PosixNativeAccess.getRLimit`: the soft limit, [`UNLIMITED`] for
/// `RLIM_INFINITY`, [`UNKNOWN`] (with ES's warning) when `getrlimit` fails.
#[cfg(unix)]
fn rlimit_cur(read: impl FnOnce(&mut libc::rlimit) -> libc::c_int, description: &str) -> i64 {
    let mut r = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    if read(&mut r) == 0 {
        if r.rlim_cur == libc::RLIM_INFINITY {
            UNLIMITED
        } else {
            i64::try_from(r.rlim_cur).unwrap_or(UNLIMITED)
        }
    } else {
        tracing::warn!("unable to retrieve {description} [{}]", strerror(&std::io::Error::last_os_error()));
        UNKNOWN
    }
}

/// `PosixNativeAccess`'s `ProcessLimits`: `RLIMIT_NPROC` on Linux only (it
/// counts threads there; on BSD/macOS it counts processes, so ES reports
/// [`UNKNOWN`]), `RLIMIT_AS` and `RLIMIT_FSIZE` on every POSIX platform.
pub fn process_limits() -> ProcessLimits {
    #[cfg(unix)]
    {
        #[cfg(target_os = "linux")]
        let max_threads = rlimit_cur(|r| unsafe { libc::getrlimit(libc::RLIMIT_NPROC, r) }, "max number of threads");
        #[cfg(not(target_os = "linux"))]
        let max_threads = UNKNOWN;
        ProcessLimits {
            max_threads,
            max_virtual_memory_size: rlimit_cur(|r| unsafe { libc::getrlimit(libc::RLIMIT_AS, r) }, "max size virtual memory"),
            max_file_size: rlimit_cur(|r| unsafe { libc::getrlimit(libc::RLIMIT_FSIZE, r) }, "max file size"),
        }
    }
    #[cfg(not(unix))]
    {
        ProcessLimits { max_threads: UNKNOWN, max_virtual_memory_size: UNKNOWN, max_file_size: UNKNOWN }
    }
}

/// ES `ProcessProbe.getMaxFileDescriptorCount`: the soft `RLIMIT_NOFILE`
/// (-1 where the platform has none).
pub fn max_file_descriptors() -> i64 {
    #[cfg(unix)]
    {
        let mut r = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut r) } != 0 {
            return UNKNOWN;
        }
        if r.rlim_cur == libc::RLIM_INFINITY {
            return UNLIMITED;
        }
        i64::try_from(r.rlim_cur).unwrap_or(UNLIMITED)
    }
    #[cfg(not(unix))]
    {
        UNKNOWN
    }
}

/// HotSpot's start-up `-XX:+MaxFDLimit` (on by default), which every ES
/// node runs under: raise the soft `RLIMIT_NOFILE` to the hard limit (on
/// macOS to at most `OPEN_MAX`, 10240). ES's file descriptor check and
/// `max_file_descriptors` therefore see the raised value.
pub fn raise_file_descriptor_limit() {
    #[cfg(unix)]
    {
        let mut r = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut r) } != 0 {
            return;
        }
        #[cfg(target_os = "macos")]
        let target = r.rlim_max.min(10240);
        #[cfg(not(target_os = "macos"))]
        let target = r.rlim_max;
        if target > r.rlim_cur {
            let raised = libc::rlimit { rlim_cur: target, rlim_max: r.rlim_max };
            if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &raised) } != 0 {
                tracing::debug!("could not raise the file descriptor limit to [{target}]: {}", std::io::Error::last_os_error());
            }
        }
    }
}

/// ES `MaxMapCountCheck.getMaxMapCount`: the first line of
/// `/proc/sys/vm/max_map_count`, or -1 (with ES's warnings) when it cannot
/// be read or parsed.
pub fn max_map_count() -> i64 {
    const PATH: &str = "/proc/sys/vm/max_map_count";
    match std::fs::read_to_string(PATH) {
        Ok(text) => {
            let line = text.lines().next().unwrap_or("");
            parse_max_map_count(line).unwrap_or_else(|| {
                tracing::warn!("unable to parse vm.max_map_count [{line}]");
                UNKNOWN
            })
        }
        Err(e) => {
            tracing::warn!("I/O exception while trying to read [{PATH}]: {e}");
            UNKNOWN
        }
    }
}

/// `Long.parseLong` of the line (no surrounding whitespace allowed).
pub fn parse_max_map_count(line: &str) -> Option<i64> {
    line.parse::<i64>().ok()
}

/// The JVM's `user.name`: the login name of the real user id.
pub fn user_name() -> String {
    #[cfg(unix)]
    {
        let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
        let mut buf = vec![0 as libc::c_char; 4096];
        let mut out: *mut libc::passwd = std::ptr::null_mut();
        let rc = unsafe { libc::getpwuid_r(libc::getuid(), &mut pwd, buf.as_mut_ptr(), buf.len(), &mut out) };
        if rc == 0 && !out.is_null() && !pwd.pw_name.is_null() {
            return unsafe { std::ffi::CStr::from_ptr(pwd.pw_name) }.to_string_lossy().into_owned();
        }
    }
    std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_else(|_| "?".into())
}

/// ES `PosixNativeAccess.tryLockMemory` (`bootstrap.memory_lock: true`,
/// called at start before the node is built): `mlockall(MCL_CURRENT)` and,
/// on failure, ES's warnings (with the `RLIMIT_MEMLOCK` advice on
/// `ENOMEM`). Returns whether memory is now locked.
///
/// Only `MCL_CURRENT`, as ES: `MCL_FUTURE` would also lock every later
/// mapping, including the memory-mapped segment files (`rustic-core`'s
/// directory maps them), so an index larger than `RLIMIT_MEMLOCK` would
/// fail to open and one smaller would be pinned in RAM in full. ES pins
/// the heap it pre-touched at start, never the mapped index files.
pub fn try_lock_memory() -> bool {
    #[cfg(unix)]
    {
        if unsafe { libc::mlockall(libc::MCL_CURRENT) } == 0 {
            MEMORY_LOCKED.store(true, Ordering::Relaxed);
            return true;
        }
        let err = std::io::Error::last_os_error();
        let errno = err.raw_os_error().unwrap_or(0);
        tracing::warn!("Unable to lock JVM Memory: error={errno}, reason={}", strerror(&err));
        tracing::warn!("This can result in part of the JVM being swapped out.");
        if errno == libc::ENOMEM {
            let mut r = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
            if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut r) } == 0 {
                let show = |v: libc::rlim_t| if v == libc::RLIM_INFINITY { "unlimited".to_string() } else { v.to_string() };
                tracing::warn!("Increase RLIMIT_MEMLOCK, soft limit: {}, hard limit: {}", show(r.rlim_cur), show(r.rlim_max));
                if cfg!(target_os = "linux") {
                    let user = user_name();
                    tracing::warn!(
                        "These can be adjusted by modifying /etc/security/limits.conf, for example:\n\t# allow user '{user}' mlockall\n\t{user} soft memlock unlimited\n\t{user} hard memlock unlimited"
                    );
                    tracing::warn!("If you are logged in interactively, you will have to re-login for the new limits to take effect.");
                }
            } else {
                tracing::warn!("Unable to retrieve resource limits: {}", strerror(&std::io::Error::last_os_error()));
            }
        }
        false
    }
    #[cfg(not(unix))]
    {
        tracing::warn!("Unable to lock JVM Memory: memory locking is not supported on this platform");
        false
    }
}
