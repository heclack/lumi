use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

// ─── Device allocation accounting ────────────────────────────────────────────
// Every device allocation in this module funnels through GpuBuf/Bf16Buf, so a
// pair of counters here gives an exact footprint without duplicating the sizing
// arithmetic (which would drift from the real allocations).
static GPU_BYTES_LIVE: AtomicUsize = AtomicUsize::new(0);
static GPU_BYTES_PEAK: AtomicUsize = AtomicUsize::new(0);

/// Device bytes currently held by GpuBuf/Bf16Buf allocations.
pub fn gpu_bytes_live() -> usize {
    GPU_BYTES_LIVE.load(Ordering::Relaxed)
}

/// High-water mark of device bytes held since process start.
pub fn gpu_bytes_peak() -> usize {
    GPU_BYTES_PEAK.load(Ordering::Relaxed)
}

/// Human-readable byte count, e.g. "12.34 GiB".
pub fn format_bytes(b: usize) -> String {
    const GIB: f64 = (1usize << 30) as f64;
    const MIB: f64 = (1usize << 20) as f64;
    let bf = b as f64;
    if bf >= GIB {
        format!("{:.2} GiB", bf / GIB)
    } else {
        format!("{:.1} MiB", bf / MIB)
    }
}

fn track_alloc(bytes: usize) {
    let live = GPU_BYTES_LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    GPU_BYTES_PEAK.fetch_max(live, Ordering::Relaxed);
}

fn track_free(bytes: usize) {
    GPU_BYTES_LIVE.fetch_sub(bytes, Ordering::Relaxed);
}

/// Panic with the size that failed and how much was already held. A bare error
/// code makes an out-of-memory look like a driver fault; this makes it obvious
/// when the config simply does not fit the card.
fn alloc_failed(err: i32, bytes: usize, what: &str) -> ! {
    panic!(
        "GPU allocation failed ({what}, error {err}): requested {}, already holding {} \
         across live buffers (peak {}). If this is out-of-memory, reduce batch_size or \
         seq_len, or enable mixed_precision / bf16_activations.",
        format_bytes(bytes),
        format_bytes(gpu_bytes_live()),
        format_bytes(gpu_bytes_peak()),
    )
}

/// GPU buffer — owns a device pointer and knows its size.
pub struct GpuBuf {
    pub ptr: *mut f32,
    pub len: usize, // number of f32 elements
}

impl GpuBuf {
    pub fn alloc(n: usize) -> Self {
        let mut ptr: *mut f32 = ptr::null_mut();
        let bytes = n * std::mem::size_of::<f32>();
        unsafe {
            let err = cuda_malloc(&mut ptr as *mut *mut f32 as *mut *mut std::ffi::c_void, bytes);
            if err != 0 {
                alloc_failed(err, bytes, "f32");
            }
        }
        track_alloc(bytes);
        Self { ptr, len: n }
    }

    pub fn empty() -> Self {
        Self { ptr: ptr::null_mut(), len: 0 }
    }

    pub fn zero(&self) {
        unsafe {
            cuda_memset(self.ptr as *mut std::ffi::c_void, 0,
                        self.len * std::mem::size_of::<f32>());
        }
    }

    /// Copy from host Vec to device.
    pub fn from_host(data: &[f32]) -> Self {
        let buf = Self::alloc(data.len());
        unsafe {
            cuda_memcpy(buf.ptr as *mut std::ffi::c_void,
                        data.as_ptr() as *const std::ffi::c_void,
                        data.len() * 4, 1); // cudaMemcpyHostToDevice = 1
        }
        buf
    }

    /// Copy device to host Vec.
    pub fn to_host(&self) -> Vec<f32> {
        let mut data = vec![0.0f32; self.len];
        unsafe {
            cuda_memcpy(data.as_mut_ptr() as *mut std::ffi::c_void,
                        self.ptr as *const std::ffi::c_void,
                        self.len * 4, 2); // cudaMemcpyDeviceToHost = 2
        }
        data
    }

    /// Copy host data into existing allocation (no realloc — preserves sub-pointers).
    pub fn copy_from_host(&self, data: &[f32]) {
        assert!(data.len() == self.len, "copy_from_host size mismatch: {} vs {}", data.len(), self.len);
        unsafe {
            cuda_memcpy(self.ptr as *mut std::ffi::c_void,
                        data.as_ptr() as *const std::ffi::c_void,
                        data.len() * 4, 1); // cudaMemcpyHostToDevice = 1
        }
    }

    /// Offset pointer (for splitting projected buffer into sub-tensors).
    pub fn offset(&self, offset: usize) -> *mut f32 {
        assert!(offset < self.len, "offset {} >= len {}", offset, self.len);
        unsafe { self.ptr.add(offset) }
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_free(self.ptr as *mut std::ffi::c_void); }
            track_free(self.len * std::mem::size_of::<f32>());
            self.ptr = ptr::null_mut();
        }
    }
}

/// GPU buffer for BF16 data (2 bytes per element). Used for mixed-precision scratch space.
pub struct Bf16Buf {
    pub ptr: *mut u16,
    pub len: usize, // number of bf16 elements
}

impl Bf16Buf {
    pub fn alloc(n: usize) -> Self {
        let mut ptr: *mut u16 = ptr::null_mut();
        let bytes = n * std::mem::size_of::<u16>();
        unsafe {
            let err = cuda_malloc(&mut ptr as *mut *mut u16 as *mut *mut std::ffi::c_void, bytes);
            if err != 0 {
                alloc_failed(err, bytes, "bf16");
            }
        }
        track_alloc(bytes);
        Self { ptr, len: n }
    }

    /// Allocate a zero-sized dummy (no GPU memory). Used when mixed_precision is off.
    pub fn empty() -> Self {
        Self { ptr: ptr::null_mut(), len: 0 }
    }
}

impl Drop for Bf16Buf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_free(self.ptr as *mut std::ffi::c_void); }
            track_free(self.len * std::mem::size_of::<u16>());
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for Bf16Buf {}
unsafe impl Sync for Bf16Buf {}

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

// Raw GPU-runtime FFI for memory management.
//
// CUDA and HIP expose the same four entry points with identical signatures, and
// the memcpy `kind` enum shares its numeric values across both (HostToDevice=1,
// DeviceToHost=2), so only the symbol names differ. `link_name` absorbs that
// difference and every call site below stays backend-agnostic.
#[cfg(all(feature = "cuda", not(feature = "hip")))]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                  count: usize, kind: i32) -> i32;
    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
}

#[cfg(feature = "hip")]
extern "C" {
    #[link_name = "hipMalloc"]
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    #[link_name = "hipFree"]
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    #[link_name = "hipMemcpy"]
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                  count: usize, kind: i32) -> i32;
    #[link_name = "hipMemset"]
    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
}

unsafe fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32 {
    cudaMalloc(ptr, size)
}
unsafe fn cuda_free(ptr: *mut std::ffi::c_void) -> i32 {
    cudaFree(ptr)
}
pub unsafe fn cuda_memcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                      count: usize, kind: i32) -> i32 {
    cudaMemcpy(dst, src, count, kind)
}
pub unsafe fn cuda_memset(ptr: *mut std::ffi::c_void, value: i32, count: usize) -> i32 {
    cudaMemset(ptr, value, count)
}
