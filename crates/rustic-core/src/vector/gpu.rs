//! GPU-built HNSW graphs (Elasticsearch `x-pack/plugin/gpu`).
//!
//! CUDA/cuVS CAGRA supplies graph proposals. Unquantized ES fields admit
//! them through the shared mutable owner's ordinary reciprocal publication
//! and cancellation path. Existing int8 segment formats can encode a GPU
//! graph while retaining their quantized scoring. Text merges never build a
//! graph for a registered mutable owner.
//!
//! * **Build-time opt-in**: the CUDA backend exists only with the `gpu`
//!   cargo feature. Without it [`info`] is always `None` and nothing here
//!   touches CUDA. Even with it the binary has no link-time CUDA dependency:
//!   the driver (`libcuda.so.1`), cuVS (`libcuvs_c.so`) and the CUDA runtime
//!   are `dlopen`ed on first use, so the same binary runs on hosts without
//!   a GPU and reports GPU indexing as unsupported, as ES does.
//! * **Run-time detection** ([`info`], ES `CuVSGPUSupport`): the first GPU
//!   with compute capability >= 7.0 and >= 7.5 GiB of memory, and a
//!   loadable cuVS.
//! * **Parameters**: CAGRA supplies degree `2m` at level zero; upper
//!   levels use nested subsets. Mutable admission applies its own diversity
//!   and reciprocal-link rules. Recall and performance require validation
//!   of the integrated path; donor measurements are not acceptance evidence.
//! * **Who may use it** is the caller's policy ([`GpuGate`]: node setting
//!   `vectors.indexing.use_gpu` + licence, decided by the engine).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use super::hnsw::HnswGraph;
#[cfg(feature = "gpu")]
use super::hnsw::{HnswBuilder, HnswParams, KnnLevel, VectorSource};
use super::Similarity;

/// ES `CuVSGPUSupport.MIN_DEVICE_MEMORY_IN_BYTES` (7.5 GiB: 8 GB cards
/// report a little less than 8 GiB).
pub const MIN_DEVICE_MEMORY_IN_BYTES: u64 = 8_053_063_680;
/// cuvs-java `GPUInfoProvider.MIN_COMPUTE_CAPABILITY_{MAJOR,MINOR}`.
pub const MIN_COMPUTE_CAPABILITY: (i32, i32) = (7, 0);
/// ES `ES92GpuHnswVectorsFormat.MIN_NUM_VECTORS_FOR_GPU_BUILD`.
pub const MIN_NUM_VECTORS_FOR_GPU_BUILD: usize = 2;
/// ES `ES92GpuHnswVectorsWriter.MAX_NUM_VECTORS_FOR_NN_DESCENT`.
pub const MAX_NUM_VECTORS_FOR_NN_DESCENT: usize = 5_000_000;
/// ES `CuVSResourceManager.GPU_COMPUTATION_MEMORY_FACTOR`.
pub const GPU_COMPUTATION_MEMORY_FACTOR: f64 = 2.0;

/// The GPU this process builds graphs on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    /// Device name (`"Tesla T4"`), ES `GpuNodeStats.type`.
    pub name: String,
    /// Total device memory, ES `memory_in_bytes`.
    pub total_memory: u64,
    pub compute_capability: (i32, i32),
    /// CUDA device ordinal.
    pub ordinal: i32,
}

/// One device as the driver reports it (input of [`select_device`]).
#[derive(Clone, Debug)]
pub struct DeviceProbe {
    pub ordinal: i32,
    pub name: String,
    pub total_memory: u64,
    pub compute_capability: (i32, i32),
}

/// ES `CuVSGPUSupport.initializeGpuInfo`: the first device with the minimum
/// compute capability and memory; the reason when there is none.
pub fn select_device(devices: &[DeviceProbe]) -> Result<GpuInfo, String> {
    if devices.is_empty() {
        return Err("No GPU found".into());
    }
    let mut last = String::new();
    for d in devices {
        let (major, minor) = d.compute_capability;
        let (min_major, min_minor) = MIN_COMPUTE_CAPABILITY;
        let capable = major > min_major || (major == min_major && minor >= min_minor);
        if !capable {
            last = format!(
                "GPU [{}] does not have the minimum compute capabilities (required: [{min_major}.{min_minor}], found: [{major}.{minor}])",
                d.name
            );
        } else if d.total_memory < MIN_DEVICE_MEMORY_IN_BYTES {
            last = format!("GPU [{}] does not have minimum memory required (required: [{MIN_DEVICE_MEMORY_IN_BYTES}], found: [{}])", d.name, d.total_memory);
        } else {
            return Ok(GpuInfo { name: d.name.clone(), total_memory: d.total_memory, compute_capability: d.compute_capability, ordinal: d.ordinal });
        }
    }
    Err(last)
}

/// CAGRA graph degree for HNSW `m`: the HNSW layer-0 degree `2m`.
pub fn cagra_graph_degree(m: usize) -> usize {
    (2 * m).max(2)
}

/// CAGRA intermediate (pre-pruning) degree: twice the graph degree, cuVS's
/// default ratio (64 -> 128).
pub fn cagra_intermediate_graph_degree(m: usize) -> usize {
    2 * cagra_graph_degree(m)
}

/// cuVS's default NN-descent iteration count.
pub const NN_DESCENT_ITERATIONS: usize = 20;

/// Upper levels with at most this many nodes are built as a CPU HNSW
/// graph (cheap at this size); larger ones are CAGRA graphs.
pub const CPU_TOP_LEVEL_MAX_NODES: usize = 4096;

/// HNSW level of each of `n` ordinals for `m` (Malkov & Yashunin: level
/// `floor(-ln(u) / ln(m))`), from a fixed hash so builds are repeatable.
pub fn node_levels(n: usize, m: usize) -> Vec<u8> {
    let ml = if m > 1 { 1.0 / (m as f64).ln() } else { 0.0 };
    (0..n as u64)
        .map(|i| {
            let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            ((-u.ln() * ml).floor() as u64).min(31) as u8
        })
        .collect()
}

/// The rows of `members` of a dense column.
#[cfg(feature = "gpu")]
struct Members<'a> {
    data: &'a [f32],
    dim: usize,
    member: Vec<bool>,
}

#[cfg(feature = "gpu")]
impl VectorSource for Members<'_> {
    fn dim(&self) -> u32 {
        self.dim as u32
    }
    fn num_docs(&self) -> u32 {
        self.member.len() as u32
    }
    fn get(&self, doc: crate::DocId) -> Option<&[f32]> {
        let d = doc as usize;
        self.member.get(d).copied().unwrap_or(false).then(|| &self.data[d * self.dim..(d + 1) * self.dim])
    }
}

/// ES `ES92GpuHnswVectorsFormat.cagraGraphDegree` (for reference: see the
/// module docs for why the graph degree differs).
pub fn es_cagra_graph_degree(m: usize) -> usize {
    2 + m * 2 / 3
}

/// cuVS `cuvsDistanceType` values ES uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum CagraMetric {
    L2Expanded = 0,
    CosineExpanded = 2,
    InnerProduct = 6,
}

/// cuVS `cuvsCagraGraphBuildAlgo`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum CagraAlgo {
    IvfPq = 1,
    NnDescent = 2,
}

/// The CAGRA build parameters of one graph (ES
/// `ES92GpuHnswVectorsWriter.createCagraIndexParams`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CagraParams {
    pub metric: CagraMetric,
    pub graph_degree: usize,
    pub intermediate_graph_degree: usize,
    pub algo: CagraAlgo,
    pub nn_descent_iterations: usize,
}

impl CagraParams {
    /// Parameters for `n` float vectors of `dim` dims built with HNSW
    /// `m`, on a device of `total_memory` bytes.
    pub fn new(similarity: Similarity, n: usize, dim: usize, m: usize, total_memory: u64) -> CagraParams {
        let graph_degree = cagra_graph_degree(m);
        // CAGRA requires the intermediate degree to exceed the graph degree.
        let intermediate_graph_degree = cagra_intermediate_graph_degree(m).max(graph_degree + 1);
        let metric = match similarity {
            Similarity::Cosine => CagraMetric::CosineExpanded,
            Similarity::L2 => CagraMetric::L2Expanded,
            Similarity::DotProduct | Similarity::MaxInnerProduct => CagraMetric::InnerProduct,
            packed => unreachable!("packed similarity {packed:?} is never built on the GPU"),
        };
        let nn_descent_memory = GPU_COMPUTATION_MEMORY_FACTOR * n as f64 * dim as f64 * 4.0;
        let algo = if n >= MAX_NUM_VECTORS_FOR_NN_DESCENT || (total_memory > 0 && nn_descent_memory > total_memory as f64) {
            CagraAlgo::IvfPq
        } else {
            CagraAlgo::NnDescent
        };
        CagraParams { metric, graph_degree, intermediate_graph_degree, algo, nn_descent_iterations: NN_DESCENT_ITERATIONS }
    }
}

/// The ordinal nearest the centroid of `data` (`n` rows of `dim`): the
/// entry point of a one-level graph, so a search starts near the middle of
/// the data instead of at an arbitrary row.
pub fn central_ordinal(data: &[f32], dim: usize, similarity: Similarity) -> u32 {
    use rayon::prelude::*;
    let n = if dim == 0 { 0 } else { data.len() / dim };
    if n <= 1 {
        return 0;
    }
    let mut centroid = data
        .par_chunks(dim * 4096)
        .map(|block| {
            let mut s = vec![0f64; dim];
            for row in block.chunks_exact(dim) {
                for (a, &x) in s.iter_mut().zip(row) {
                    *a += x as f64;
                }
            }
            s
        })
        .reduce(|| vec![0f64; dim], |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        })
        .into_iter()
        .map(|x| (x / n as f64) as f32)
        .collect::<Vec<f32>>();
    if matches!(similarity, Similarity::DotProduct | Similarity::MaxInnerProduct) {
        // Inner-product "distance" rewards long rows, not central ones.
        let len = super::norm(&centroid);
        if len > 0.0 {
            centroid.iter_mut().for_each(|x| *x /= len);
        }
    }
    let metric = if similarity == Similarity::L2 { Similarity::L2 } else { Similarity::Cosine };
    data.par_chunks_exact(dim)
        .enumerate()
        .map(|(i, row)| (metric.distance(&centroid, row), i as u32))
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
        .map_or(0, |(_, i)| i)
}

/// GPU graph builds of this process since start (ES `GPUSupport`
/// `GPU_USAGE_COUNT`, reported as `index_build_count`).
static USAGE: AtomicU64 = AtomicU64::new(0);

/// Phases of the last GPU graph build of this process: level-0 CAGRA,
/// upper levels, whole build (diagnostics and benchmarks).
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildTimings {
    pub level0: std::time::Duration,
    pub upper: std::time::Duration,
    pub total: std::time::Duration,
}

static LAST_TIMINGS: parking_lot::Mutex<Option<BuildTimings>> = parking_lot::Mutex::new(None);

/// See [`BuildTimings`].
pub fn last_build_timings() -> Option<BuildTimings> {
    *LAST_TIMINGS.lock()
}

/// ES `GPUSupport.getUsageCount`.
pub fn usage_count() -> u64 {
    USAGE.load(Ordering::Relaxed)
}

/// The compatible GPU of this host, detected once; `None` without the
/// `gpu` feature, without a compatible device or without cuVS (ES
/// `CuVSGPUSupport.isSupported`).
pub fn info() -> Option<&'static GpuInfo> {
    static INFO: OnceLock<Option<GpuInfo>> = OnceLock::new();
    INFO.get_or_init(|| {
        #[cfg(feature = "gpu")]
        {
            match backend::detect() {
                Ok(i) => {
                    tracing::info!("Found compatible GPU [{}] (id: [{}])", i.name, i.ordinal);
                    Some(i)
                }
                Err(e) => {
                    tracing::warn!("GPU based vector indexing is not supported on this node; {e}");
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    })
    .as_ref()
}

/// Whether a seal may build its graph on the GPU: set by the engine on a
/// node whose GPU indexing is enabled, evaluated at each build (the
/// licence can change). The default gate never allows it.
#[derive(Clone, Default)]
pub struct GpuGate(Option<Arc<dyn Fn() -> bool + Send + Sync>>);

impl GpuGate {
    pub fn new(allowed: impl Fn() -> bool + Send + Sync + 'static) -> GpuGate {
        GpuGate(Some(Arc::new(allowed)))
    }

    /// A GPU is present and the policy allows using it now.
    pub fn allows(&self) -> bool {
        self.0.as_ref().is_some_and(|f| info().is_some() && f())
    }

    /// A gate is installed (the node opted in), whatever it answers now.
    pub fn is_set(&self) -> bool {
        self.0.is_some()
    }
}

impl std::fmt::Debug for GpuGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "GpuGate(set)" } else { "GpuGate(none)" })
    }
}

/// Exclusive use of the GPU for graph builds. One build at a time: the
/// build saturates the device, and serializing keeps peak device memory to
/// one dataset.
pub struct Lease {
    #[cfg(feature = "gpu")]
    guard: parking_lot::MutexGuard<'static, Option<backend::Ctx>>,
}

#[cfg(feature = "gpu")]
fn device_lock() -> &'static parking_lot::Mutex<Option<backend::Ctx>> {
    static LOCK: parking_lot::Mutex<Option<backend::Ctx>> = parking_lot::Mutex::new(None);
    &LOCK
}

/// Wait for the GPU (`None` when there is no usable GPU). ES `acquire`
/// (merges).
pub fn lease() -> Option<Lease> {
    info()?;
    #[cfg(feature = "gpu")]
    {
        Some(Lease { guard: device_lock().lock() })
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// The GPU if it is free now (`None` when busy or absent). ES `tryAcquire`
/// (flushes fall back to the CPU when the GPU is busy).
pub fn try_lease() -> Option<Lease> {
    try_lease_for(std::time::Duration::ZERO)
}

/// The GPU if it frees up within `wait` (`None` when still busy, or
/// absent). ES keeps a pool of four GPU resources, so a flush only falls
/// back when four builds are running; one device lock with a short wait
/// gives refreshes the same chance of building on the GPU.
pub fn try_lease_for(wait: std::time::Duration) -> Option<Lease> {
    info()?;
    #[cfg(feature = "gpu")]
    {
        device_lock().try_lock_for(wait).map(|guard| Lease { guard })
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = wait;
        None
    }
}

impl Lease {
    /// Build the graph of `data` (`n` rows of `dim` floats, ordinal order).
    ///
    /// Level 0 is a CAGRA graph over every row (degree `2m`). A one-level
    /// graph searched from a single entry point cannot leave the region it
    /// starts in (CAGRA graphs of clustered data have one component per
    /// cluster; CAGRA's own search starts from many random seeds), so the
    /// upper levels are built as in HNSW over nested random subsets (each
    /// node's HNSW level): CAGRA graphs (degree `m`) while a level has more
    /// than [`CPU_TOP_LEVEL_MAX_NODES`] nodes, then an HNSW graph over the
    /// remaining few on the CPU. A search then descends to the right
    /// region before the layer-0 beam, exactly like an HNSW search.
    ///
    /// Errors (device out of memory, cuVS failures) leave the caller to
    /// build on the CPU.
    pub fn build_graph(&mut self, dim: usize, data: &[f32], similarity: Similarity, m: usize, ef_construction: usize) -> Result<HnswGraph, String> {
        let n = if dim == 0 { 0 } else { data.len() / dim };
        if n < MIN_NUM_VECTORS_FOR_GPU_BUILD {
            return Err(format!("[{n}] vectors is below the GPU minimum of [{MIN_NUM_VECTORS_FOR_GPU_BUILD}]"));
        }
        let gpu = info().ok_or("no GPU")?;
        let params = CagraParams::new(similarity, n, dim, m, gpu.total_memory);
        #[cfg(feature = "gpu")]
        {
            let started = std::time::Instant::now();
            let (degree, rows) = backend::build(&mut self.guard, gpu, data, n, dim, &params)?;
            let level0 = started.elapsed();
            let mut levels = vec![KnnLevel { nodes: (0..n as u32).collect(), degree, rows }];
            let node_level = node_levels(n, m);
            let mut top = None;
            for level in 1u8.. {
                let members: Vec<u32> = (0..n as u32).filter(|&i| node_level[i as usize] >= level).collect();
                if members.is_empty() {
                    break;
                }
                if members.len() <= CPU_TOP_LEVEL_MAX_NODES || members.len() < MIN_NUM_VECTORS_FOR_GPU_BUILD {
                    let mut member = vec![false; n];
                    members.iter().for_each(|&i| member[i as usize] = true);
                    let src = Members { data, dim, member };
                    let p = HnswParams { m, ef_construction, threads: 0, ..HnswParams::default() };
                    top = Some(HnswBuilder::build(dim as u32, &src, n as u32, similarity, p));
                    break;
                }
                let mut sub = Vec::with_capacity(members.len() * dim);
                for &i in &members {
                    sub.extend_from_slice(&data[i as usize * dim..(i as usize + 1) * dim]);
                }
                let mut p = CagraParams::new(similarity, members.len(), dim, m, gpu.total_memory);
                p.graph_degree = m.max(2);
                p.intermediate_graph_degree = 2 * p.graph_degree;
                let (degree, local) = backend::build(&mut self.guard, gpu, &sub, members.len(), dim, &p)?;
                let rows = local.into_iter().map(|v| members.get(v as usize).copied().unwrap_or(u32::MAX)).collect();
                levels.push(KnnLevel { nodes: members, degree, rows });
            }
            let entry = if top.is_none() { central_ordinal(data, dim, similarity) } else { 0 };
            let graph = HnswGraph::from_levels(levels, top, entry);
            *LAST_TIMINGS.lock() = Some(BuildTimings { level0, upper: started.elapsed() - level0, total: started.elapsed() });
            tracing::debug!("graph of [{n}] x [{dim}] built on the GPU in [{:?}] ({params:?}, {} levels)", started.elapsed(), graph.num_layers());
            USAGE.fetch_add(1, Ordering::Relaxed);
            Ok(graph)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (params, ef_construction);
            Err("built without the `gpu` feature".into())
        }
    }
}

#[cfg(feature = "gpu")]
mod backend {
    //! CUDA driver / runtime and cuVS C API, `dlopen`ed (no link-time
    //! dependency).
    use super::{CagraAlgo, CagraParams, DeviceProbe, GpuInfo};
    use std::ffi::{c_char, c_int, c_void, CStr, CString};

    const RTLD_NOW: c_int = 2;
    const RTLD_LOCAL: c_int = 0;
    const RTLD_NOLOAD: c_int = 4;
    const RTLD_GLOBAL: c_int = 0x100;
    /// Never unmap the object, even when its last handle is closed.
    #[cfg(target_os = "macos")]
    const RTLD_NODELETE: c_int = 0x80;
    #[cfg(not(target_os = "macos"))]
    const RTLD_NODELETE: c_int = 0x1000;

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *mut c_char;
    }

    fn dl_error(name: &str) -> String {
        let e = unsafe { dlerror() };
        if e.is_null() { format!("cannot load [{name}]") } else { unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned() }
    }

    fn open_any(names: &[&str]) -> Result<*mut c_void, String> {
        open_accepted(names, |_| Ok(()))
    }

    /// The first of `names` that loads and that `accept` (given its handle)
    /// accepts, each tried in turn. Probed `RTLD_LOCAL`, so a rejected
    /// library's symbols never join the global namespace. A rejected
    /// library stays mapped (`RTLD_NODELETE`, and its handle is never
    /// closed): its constructors have run and may have registered `atexit`
    /// handlers, TLS destructors or CUDA runtime callbacks that point into
    /// its code, so unmapping it would crash the process at exit. The
    /// accepted one is then promoted to `RTLD_GLOBAL`, as it was loaded
    /// before.
    fn open_accepted(names: &[&str], accept: impl Fn(*mut c_void) -> Result<(), String>) -> Result<*mut c_void, String> {
        let mut errors = Vec::new();
        for name in names {
            let c = CString::new(*name).unwrap();
            let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) };
            if h.is_null() {
                errors.push(dl_error(name));
                continue;
            }
            if let Err(error) = accept(h) {
                // Deliberately leaked (see above).
                errors.push(format!("[{name}]: {error}"));
                continue;
            }
            // Same object, now global (one more reference, kept for the
            // process like the first).
            if unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL) }.is_null() {
                errors.push(dl_error(name));
                // Deliberately leaked (see above).
                continue;
            }
            return Ok(h);
        }
        Err(errors.join("; "))
    }

    unsafe fn sym<T: Copy>(h: *mut c_void, name: &str) -> Result<T, String> {
        let c = CString::new(name).unwrap();
        let p = unsafe { dlsym(h, c.as_ptr()) };
        if p.is_null() {
            return Err(format!("missing symbol [{name}]"));
        }
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
    }

    // DLPack (v0.8 / v1.0 `DLManagedTensor`).
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DLDevice {
        device_type: i32,
        device_id: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DLDataType {
        code: u8,
        bits: u8,
        lanes: u16,
    }
    #[repr(C)]
    struct DLTensor {
        data: *mut c_void,
        device: DLDevice,
        ndim: i32,
        dtype: DLDataType,
        shape: *mut i64,
        strides: *mut i64,
        byte_offset: u64,
    }
    #[repr(C)]
    struct DLManagedTensor {
        dl_tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
    }
    const KDL_CPU: i32 = 1;
    const KDL_FLOAT: u8 = 2;

    /// Leading fields of cuVS `struct cuvsCagraIndexParams` (stable since
    /// 24.x; the struct is allocated by cuVS, we only write these).
    #[repr(C)]
    struct CagraIndexParams {
        metric: i32,
        intermediate_graph_degree: usize,
        graph_degree: usize,
        build_algo: i32,
        nn_descent_niter: usize,
    }

    type CuResult = c_int;
    type CuvsError = c_int; // CUVS_ERROR = 0, CUVS_SUCCESS = 1
    const CUVS_SUCCESS: CuvsError = 1;

    struct Driver {
        init: unsafe extern "C" fn(u32) -> CuResult,
        device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
        device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
        device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult,
        device_total_mem: unsafe extern "C" fn(*mut usize, c_int) -> CuResult,
        device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    }

    pub(super) struct Cuvs {
        resources_create: unsafe extern "C" fn(*mut usize) -> CuvsError,
        last_error: unsafe extern "C" fn() -> *const c_char,
        params_create: unsafe extern "C" fn(*mut *mut CagraIndexParams) -> CuvsError,
        params_destroy: unsafe extern "C" fn(*mut CagraIndexParams) -> CuvsError,
        index_create: unsafe extern "C" fn(*mut *mut c_void) -> CuvsError,
        index_destroy: unsafe extern "C" fn(*mut c_void) -> CuvsError,
        build: unsafe extern "C" fn(usize, *mut CagraIndexParams, *mut DLManagedTensor, *mut c_void) -> CuvsError,
        graph_degree: unsafe extern "C" fn(*mut c_void, *mut i64) -> CuvsError,
        size: unsafe extern "C" fn(*mut c_void, *mut i64) -> CuvsError,
        get_graph: unsafe extern "C" fn(*mut c_void, *mut DLManagedTensor) -> CuvsError,
        stream_sync: unsafe extern "C" fn(usize) -> CuvsError,
        set_device: unsafe extern "C" fn(c_int) -> c_int,
        memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
    }
    // Function pointers into process-global libraries.
    unsafe impl Send for Cuvs {}
    unsafe impl Sync for Cuvs {}

    static CUVS: std::sync::OnceLock<Result<Cuvs, String>> = std::sync::OnceLock::new();

    // The SONAME does not identify the CAGRA dataset ABI: 26.08 changed
    // cuvsCagraBuild's third argument from DLManagedTensor* to cuvsDataset_t.
    // Only this exact release has been audited against these handwritten C
    // layouts. Unknown/older releases must be audited before adding them.
    fn require_tensor_abi(version: [u16; 3]) -> Result<(), String> {
        if version == [26, 6, 0] {
            Ok(())
        } else {
            Err(format!(
                "unsupported cuVS C ABI [{}.{:02}.{}]; this backend requires audited 26.06.0 DLManagedTensor ABI",
                version[0], version[1], version[2]
            ))
        }
    }

    fn cuvs() -> Result<&'static Cuvs, String> {
        CUVS.get_or_init(|| unsafe {
            // Every candidate is tried until one passes the version check:
            // a mismatched `libcuvs_c.so` must not hide a valid `.so.26`.
            // This scalar version function has the same published signature
            // in 26.06/26.08/26.10. Resolve and check it before any resource,
            // params, index, dataset or build calls. Missing markers fail closed.
            let h = open_accepted(&["libcuvs_c.so", "libcuvs_c.so.26", "libcuvs_c.so.25"], |h| {
                let version_get: unsafe extern "C" fn(*mut u16, *mut u16, *mut u16) -> CuvsError =
                    sym(h, "cuvsVersionGet")?;
                let (mut major, mut minor, mut patch) = (0u16, 0u16, 0u16);
                if version_get(&mut major, &mut minor, &mut patch) != CUVS_SUCCESS {
                    return Err("cuVS version query failed; CAGRA ABI was not accepted".into());
                }
                require_tensor_abi([major, minor, patch])
            })?;
            // cuVS links the CUDA runtime: this returns the loaded copy.
            let rt = open_any(&["libcudart.so.12", "libcudart.so.13", "libcudart.so"])?;
            Ok(Cuvs {
                resources_create: sym(h, "cuvsResourcesCreate")?,
                last_error: sym(h, "cuvsGetLastErrorText")?,
                params_create: sym(h, "cuvsCagraIndexParamsCreate")?,
                params_destroy: sym(h, "cuvsCagraIndexParamsDestroy")?,
                index_create: sym(h, "cuvsCagraIndexCreate")?,
                index_destroy: sym(h, "cuvsCagraIndexDestroy")?,
                build: sym(h, "cuvsCagraBuild")?,
                graph_degree: sym(h, "cuvsCagraIndexGetGraphDegree")?,
                size: sym(h, "cuvsCagraIndexGetSize")?,
                get_graph: sym(h, "cuvsCagraIndexGetGraph")?,
                stream_sync: sym(h, "cuvsStreamSync")?,
                set_device: sym(rt, "cudaSetDevice")?,
                memcpy: sym(rt, "cudaMemcpy")?,
            })
        })
        .as_ref()
        .map_err(Clone::clone)
    }

    #[test]
    fn cagra_dataset_abi_is_not_inferred_from_soname() {
        assert!(require_tensor_abi([26, 6, 0]).is_ok());
        // These releases retain SONAME 26 but changed the dataset parameter.
        for version in [[26, 8, 1], [26, 10, 0], [26, 6, 1], [25, 12, 0], [0, 0, 0]] {
            assert!(require_tensor_abi(version).is_err(), "{version:?}");
        }
    }

    fn probe_devices() -> Result<Vec<DeviceProbe>, String> {
        unsafe {
            let h = open_any(&["libcuda.so.1", "libcuda.so"])?;
            let d = Driver {
                init: sym(h, "cuInit")?,
                device_get_count: sym(h, "cuDeviceGetCount")?,
                device_get: sym(h, "cuDeviceGet")?,
                device_get_name: sym(h, "cuDeviceGetName")?,
                device_total_mem: sym(h, "cuDeviceTotalMem_v2")?,
                device_get_attribute: sym(h, "cuDeviceGetAttribute")?,
            };
            let rc = (d.init)(0);
            if rc != 0 {
                return Err(format!("cuInit failed with CUDA error [{rc}]"));
            }
            let mut count = 0;
            if (d.device_get_count)(&mut count) != 0 {
                return Err("cuDeviceGetCount failed".into());
            }
            let mut out = Vec::new();
            for ordinal in 0..count {
                let mut dev = 0;
                if (d.device_get)(&mut dev, ordinal) != 0 {
                    continue;
                }
                let mut name = [0 as c_char; 256];
                (d.device_get_name)(name.as_mut_ptr(), name.len() as c_int, dev);
                let mut mem = 0usize;
                (d.device_total_mem)(&mut mem, dev);
                let (mut major, mut minor) = (0, 0);
                // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR.
                (d.device_get_attribute)(&mut major, 75, dev);
                (d.device_get_attribute)(&mut minor, 76, dev);
                out.push(DeviceProbe {
                    ordinal,
                    name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(),
                    total_memory: mem as u64,
                    compute_capability: (major, minor),
                });
            }
            Ok(out)
        }
    }

    pub(super) fn detect() -> Result<GpuInfo, String> {
        let info = super::select_device(&probe_devices()?)?;
        cuvs().map_err(|e| format!("cuVS is not available: {e}"))?;
        Ok(info)
    }

    /// Per-process cuVS resources (a CUDA stream and handles), created on
    /// first build and reused under the device lock.
    pub(super) struct Ctx {
        res: usize,
    }

    fn err(lib: &Cuvs, what: &str) -> String {
        let p = unsafe { (lib.last_error)() };
        let detail = if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() };
        format!("{what} failed: {detail}")
    }

    /// Build a CAGRA graph; `(degree, n * degree neighbour ids)`.
    pub(super) fn build(ctx: &mut Option<Ctx>, gpu: &GpuInfo, data: &[f32], n: usize, dim: usize, p: &CagraParams) -> Result<(usize, Vec<u32>), String> {
        let lib = cuvs()?;
        unsafe {
            if (lib.set_device)(gpu.ordinal) != 0 {
                return Err(format!("cudaSetDevice({}) failed", gpu.ordinal));
            }
            if ctx.is_none() {
                let mut res = 0usize;
                if (lib.resources_create)(&mut res) != CUVS_SUCCESS {
                    return Err(err(lib, "cuvsResourcesCreate"));
                }
                *ctx = Some(Ctx { res });
            }
            let res = ctx.as_ref().unwrap().res;
            let mut params: *mut CagraIndexParams = std::ptr::null_mut();
            if (lib.params_create)(&mut params) != CUVS_SUCCESS {
                return Err(err(lib, "cuvsCagraIndexParamsCreate"));
            }
            (*params).metric = p.metric as i32;
            (*params).intermediate_graph_degree = p.intermediate_graph_degree;
            (*params).graph_degree = p.graph_degree;
            (*params).build_algo = p.algo as i32;
            (*params).nn_descent_niter = p.nn_descent_iterations;
            if p.algo == CagraAlgo::IvfPq {
                // cuVS derives the IVF-PQ parameters from the dataset shape
                // when the build algorithm asks for IVF-PQ without them.
            }
            let mut index: *mut c_void = std::ptr::null_mut();
            if (lib.index_create)(&mut index) != CUVS_SUCCESS {
                (lib.params_destroy)(params);
                return Err(err(lib, "cuvsCagraIndexCreate"));
            }
            let mut shape = [n as i64, dim as i64];
            let mut dataset = DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data.as_ptr() as *mut c_void,
                    device: DLDevice { device_type: KDL_CPU, device_id: 0 },
                    ndim: 2,
                    dtype: DLDataType { code: KDL_FLOAT, bits: 32, lanes: 1 },
                    shape: shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };
            let built = (lib.build)(res, params, &mut dataset, index);
            (lib.params_destroy)(params);
            let result = if built != CUVS_SUCCESS {
                Err(err(lib, "cuvsCagraBuild"))
            } else {
                read_graph(lib, res, index, n)
            };
            (lib.index_destroy)(index);
            result
        }
    }

    unsafe fn read_graph(lib: &Cuvs, res: usize, index: *mut c_void, n: usize) -> Result<(usize, Vec<u32>), String> {
        unsafe {
            let (mut degree, mut size) = (0i64, 0i64);
            if (lib.graph_degree)(index, &mut degree) != CUVS_SUCCESS || (lib.size)(index, &mut size) != CUVS_SUCCESS {
                return Err(err(lib, "cuvsCagraIndexGetGraphDegree"));
            }
            if size as usize != n {
                return Err(format!("CAGRA index has [{size}] rows, expected [{n}]"));
            }
            let mut graph = std::mem::zeroed::<DLManagedTensor>();
            if (lib.get_graph)(index, &mut graph) != CUVS_SUCCESS {
                return Err(err(lib, "cuvsCagraIndexGetGraph"));
            }
            (lib.stream_sync)(res);
            let degree = degree as usize;
            let mut out = vec![0u32; n * degree];
            let src = (graph.dl_tensor.data as *const u8).add(graph.dl_tensor.byte_offset as usize) as *const c_void;
            let bytes = out.len() * 4;
            let rc = if graph.dl_tensor.device.device_type == KDL_CPU {
                std::ptr::copy_nonoverlapping(src as *const u32, out.as_mut_ptr(), out.len());
                0
            } else {
                // cudaMemcpyDeviceToHost; the runtime figures out the source.
                (lib.memcpy)(out.as_mut_ptr() as *mut c_void, src, bytes, 2)
            };
            if let Some(del) = graph.deleter {
                del(&mut graph);
            }
            if rc != 0 {
                return Err(format!("cudaMemcpy of the CAGRA graph failed with [{rc}]"));
            }
            Ok((degree, out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::hnsw::KnnLevel;

    fn probe(ordinal: i32, cc: (i32, i32), mem: u64) -> DeviceProbe {
        DeviceProbe { ordinal, name: format!("gpu{ordinal}"), total_memory: mem, compute_capability: cc }
    }

    #[test]
    fn device_selection_follows_es_minimums() {
        assert_eq!(select_device(&[]).unwrap_err(), "No GPU found");
        // Too old, then too small, then the first compatible one wins.
        let devices = [probe(0, (6, 1), 16 << 30), probe(1, (7, 5), 4 << 30), probe(2, (7, 0), MIN_DEVICE_MEMORY_IN_BYTES), probe(3, (8, 0), 80 << 30)];
        assert_eq!(select_device(&devices).unwrap().ordinal, 2);
        let err = select_device(&devices[..1]).unwrap_err();
        assert!(err.contains("minimum compute capabilities (required: [7.0], found: [6.1])"), "{err}");
        let err = select_device(&devices[1..2]).unwrap_err();
        assert!(err.contains("minimum memory required (required: [8053063680], found: [4294967296])"), "{err}");
    }

    #[test]
    fn cagra_parameters() {
        // m = 16: the HNSW layer-0 degree 32 (ES would use 12).
        assert_eq!((cagra_graph_degree(16), cagra_intermediate_graph_degree(16), es_cagra_graph_degree(16)), (32, 64, 12));
        let p = CagraParams::new(Similarity::L2, 1000, 24, 16, 16 << 30);
        assert_eq!(p, CagraParams { metric: CagraMetric::L2Expanded, graph_degree: 32, intermediate_graph_degree: 64, algo: CagraAlgo::NnDescent, nn_descent_iterations: 20 });
        let p = CagraParams::new(Similarity::Cosine, 1000, 24, 1, 16 << 30);
        assert_eq!((p.graph_degree, p.intermediate_graph_degree, p.metric), (2, 4, CagraMetric::CosineExpanded));
        // ES's metrics.
        assert_eq!(CagraParams::new(Similarity::DotProduct, 10, 4, 16, 0).metric, CagraMetric::InnerProduct);
        assert_eq!(CagraParams::new(Similarity::MaxInnerProduct, 10, 4, 16, 0).metric, CagraMetric::InnerProduct);
        // IVF-PQ from 5M vectors, or when NN-descent would not fit (ES).
        assert_eq!(CagraParams::new(Similarity::L2, MAX_NUM_VECTORS_FOR_NN_DESCENT, 4, 16, 1 << 40).algo, CagraAlgo::IvfPq);
        assert_eq!(CagraParams::new(Similarity::L2, 1_000_000, 768, 16, 4 << 30).algo, CagraAlgo::IvfPq);
        assert_eq!(CagraParams::new(Similarity::L2, 1_000_000, 768, 16, 16 << 30).algo, CagraAlgo::NnDescent);
    }

    #[test]
    fn knn_rows_become_a_one_level_graph() {
        // Node 0 lists itself, a duplicate and an out-of-range id.
        let knn = [0, 1, 1, 9, 0, 2, 0, 2, 1, 0, 1, 2];
        let g = HnswGraph::from_knn_rows(4, 3, &knn, 2);
        assert_eq!((g.num_layers(), g.num_nodes(), g.entry()), (1, 4, Some(2)));
        assert_eq!(g.max_neighbors(0), Some(3));
        let bytes = g.encode();
        let back = HnswGraph::open(&bytes).unwrap();
        assert_eq!(back.encode(), bytes);
        assert!((g.mean_degree() - 2.0).abs() < 1e-9, "{}", g.mean_degree());
        assert_eq!(HnswGraph::from_knn_rows(0, 3, &[], 0).num_nodes(), 0);
    }

    #[test]
    fn node_levels_follow_the_hnsw_distribution() {
        let levels = node_levels(200_000, 16);
        assert_eq!(levels, node_levels(200_000, 16), "repeatable");
        let at_least = |l: u8| levels.iter().filter(|&&x| x >= l).count() as f64 / levels.len() as f64;
        // P(level >= l) = m^-l.
        assert!((at_least(1) - 1.0 / 16.0).abs() < 0.003, "{}", at_least(1));
        assert!((at_least(2) - 1.0 / 256.0).abs() < 0.001, "{}", at_least(2));
        assert!(node_levels(10, 1).iter().all(|&l| l == 0));
    }

    #[test]
    fn stacked_levels_nest_into_one_graph() {
        // Level 0 over 0..6, level 1 over {1, 4} (ids off-level dropped).
        let l0 = KnnLevel { nodes: (0..6).collect(), degree: 2, rows: vec![1, 2, 0, 2, 1, 3, 2, 4, 3, 5, 4, 3] };
        let l1 = KnnLevel { nodes: vec![1, 4], degree: 2, rows: vec![4, 0, 1, 5] };
        let g = HnswGraph::from_levels(vec![l0, l1], None, 4);
        assert_eq!((g.num_layers(), g.num_nodes(), g.entry()), (2, 6, Some(4)));
        let back = HnswGraph::open(&g.encode()).unwrap();
        assert_eq!(back.encode(), g.encode());
    }

    #[test]
    fn entry_is_the_row_nearest_the_centroid() {
        let data = [0.0, 0.0, 10.0, 10.0, 4.0, 6.0, 9.0, 1.0];
        assert_eq!(central_ordinal(&data, 2, Similarity::L2), 2);
        assert_eq!(central_ordinal(&data[..2], 2, Similarity::L2), 0);
    }

    /// ES builds every segment of >= 2 vectors on the GPU (its YAML tests
    /// index 3 to 10 vectors): CAGRA must cope with degrees above `n`.
    #[test]
    fn tiny_segments_build_on_the_gpu() {
        let Some(lease) = lease() else { return };
        check_tiny_gpu_segments(lease);
    }

    /// Explicit deployment gate: unlike the portable unit test above this
    /// must fail when CUDA/cuVS or a supported physical GPU is unavailable.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a supported physical CUDA GPU and cuVS; run explicitly in the GPU gate"]
    fn hardware_required_builds_graphs() {
        let lease = lease().expect("GPU gate requires supported CUDA hardware and loadable cuVS");
        check_tiny_gpu_segments(lease);
    }

    fn check_tiny_gpu_segments(mut lease: Lease) {
        for n in [2usize, 3, 10, 100, 1000] {
            let data: Vec<f32> = (0..n * 5).map(|i| ((i * 37 % 11) as f32) - 5.0 + (i / 5) as f32 * 0.01).collect();
            let before = usage_count();
            let g = lease.build_graph(5, &data, Similarity::L2, 16, 100).unwrap_or_else(|e| panic!("n={n}: {e}"));
            assert_eq!(g.num_nodes(), n);
            assert_eq!(usage_count(), before + 1);
            assert!(g.mean_degree() >= 1.0, "n={n}: {}", g.mean_degree());
        }
        assert!(lease.build_graph(5, &[0.0; 5], Similarity::L2, 16, 100).is_err(), "one vector is not a GPU build");
    }

    #[test]
    fn without_a_gate_or_gpu_nothing_is_allowed() {
        assert!(!GpuGate::default().allows());
        assert!(!GpuGate::default().is_set());
        if info().is_none() {
            assert!(!GpuGate::new(|| true).allows());
            assert!(lease().is_none() && try_lease().is_none());
        }
    }
}
