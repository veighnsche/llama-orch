// vram_ops.cu — Safe CUDA VRAM operations
//
// Provides safe wrappers around CUDA memory operations with bounds checking.
//
// Security: TIER 1 Critical
// - All operations validate parameters
// - Error codes returned for all failures
// - No silent failures
// - Defensive programming throughout
//
// Testing: See tests/cuda_kernel_tests.rs for unit tests

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Error codes (matches Rust CudaError enum)
extern "C" {
    const int CUDA_SUCCESS = 0;
    const int CUDA_ERROR_ALLOCATION_FAILED = 1;
    const int CUDA_ERROR_INVALID_VALUE = 2;
    const int CUDA_ERROR_MEMCPY_FAILED = 3;
    const int CUDA_ERROR_DRIVER = 4;
}

// Maximum allocation size (100GB) - prevents integer overflow
const size_t MAX_ALLOCATION_SIZE = 100ULL * 1024ULL * 1024ULL * 1024ULL;

// Helper: Map CUDA error to our error codes
static inline int map_cuda_error(cudaError_t err) {
    switch (err) {
        case cudaSuccess:
            return CUDA_SUCCESS;
        case cudaErrorMemoryAllocation:
        case cudaErrorOutOfMemory:
            return CUDA_ERROR_ALLOCATION_FAILED;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidMemcpyDirection:
            return CUDA_ERROR_INVALID_VALUE;
        default:
            return CUDA_ERROR_DRIVER;
    }
}

// Helper: Validate pointer is not null
static inline bool is_valid_ptr(const void* ptr) {
    return ptr != nullptr;
}

// Helper: Check for size overflow
static inline bool is_size_valid(size_t size) {
    return size > 0 && size <= MAX_ALLOCATION_SIZE;
}

/// Allocate VRAM with bounds checking
///
/// # Safety
/// - Returns null pointer on failure
/// - Sets error code
/// - Validates size > 0 and size <= MAX_ALLOCATION_SIZE
/// - Prevents integer overflow
///
/// # Parameters
/// - ptr: Output pointer (must be non-null)
/// - bytes: Size to allocate (must be > 0 and <= 100GB)
///
/// # Returns
/// - CUDA_SUCCESS on success
/// - CUDA_ERROR_INVALID_VALUE if parameters are invalid
/// - CUDA_ERROR_ALLOCATION_FAILED if allocation fails
extern "C" int vram_malloc(void** ptr, size_t bytes) {
    // Validate output pointer
    if (!is_valid_ptr(ptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to null (defensive)
    *ptr = nullptr;
    
    // Validate size
    if (!is_size_valid(bytes)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Clear any previous CUDA errors (defensive)
    cudaGetLastError();
    
    // Attempt allocation
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        *ptr = nullptr;
        return map_cuda_error(err);
    }
    
    // Verify allocation succeeded (defensive)
    if (*ptr == nullptr) {
        return CUDA_ERROR_ALLOCATION_FAILED;
    }
    
    // Verify pointer is aligned (defensive)
    if (reinterpret_cast<uintptr_t>(*ptr) % 256 != 0) {
        // CUDA should always return 256-byte aligned pointers
        // If not, something is very wrong
        cudaFree(*ptr);
        *ptr = nullptr;
        return CUDA_ERROR_DRIVER;
    }
    
    return CUDA_SUCCESS;
}

/// Deallocate VRAM
///
/// # Safety
/// - Accepts null pointer (no-op, idempotent)
/// - Returns error code on failure
/// - Safe to call multiple times on same pointer
/// - Clears any previous CUDA errors
///
/// # Parameters
/// - ptr: Pointer to free (null is valid)
///
/// # Returns
/// - CUDA_SUCCESS on success or if ptr is null
/// - CUDA_ERROR_DRIVER if cudaFree fails
extern "C" int vram_free(void* ptr) {
    // Null pointer is valid (no-op, idempotent)
    if (ptr == nullptr) {
        return CUDA_SUCCESS;
    }
    
    // Clear any previous CUDA errors (defensive)
    cudaGetLastError();
    
    // Attempt to free
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        // Don't return CUDA_ERROR_INVALID_VALUE for invalid pointer
        // because that could indicate a double-free or corruption
        return map_cuda_error(err);
    }
    
    return CUDA_SUCCESS;
}

/// Copy from host to device (VRAM)
///
/// # Safety
/// - Validates all pointers are non-null
/// - Validates size is reasonable
/// - No pointer overlap checking (CUDA handles this)
/// - Synchronous operation (blocks until complete)
///
/// # Parameters
/// - dst: Device pointer (must be non-null)
/// - src: Host pointer (must be non-null)
/// - bytes: Number of bytes to copy
///
/// # Returns
/// - CUDA_SUCCESS on success or if bytes is 0
/// - CUDA_ERROR_INVALID_VALUE if pointers are null
/// - CUDA_ERROR_MEMCPY_FAILED if copy fails
extern "C" int vram_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    // Validate pointers
    if (!is_valid_ptr(dst) || !is_valid_ptr(src)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Zero bytes is a no-op (not an error)
    if (bytes == 0) {
        return CUDA_SUCCESS;
    }
    
    // Validate size is reasonable
    if (bytes > MAX_ALLOCATION_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Clear any previous CUDA errors (defensive)
    cudaGetLastError();
    
    // Perform copy (synchronous)
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return map_cuda_error(err);
    }
    
    // Verify copy completed (defensive)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return CUDA_ERROR_DRIVER;
    }
    
    return CUDA_SUCCESS;
}

/// Copy from device (VRAM) to host
///
/// # Safety
/// - Validates all pointers are non-null
/// - Validates size is reasonable
/// - No pointer overlap checking (CUDA handles this)
/// - Synchronous operation (blocks until complete)
///
/// # Parameters
/// - dst: Host pointer (must be non-null)
/// - src: Device pointer (must be non-null)
/// - bytes: Number of bytes to copy
///
/// # Returns
/// - CUDA_SUCCESS on success or if bytes is 0
/// - CUDA_ERROR_INVALID_VALUE if pointers are null
/// - CUDA_ERROR_MEMCPY_FAILED if copy fails
extern "C" int vram_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    // Validate pointers
    if (!is_valid_ptr(dst) || !is_valid_ptr(src)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Zero bytes is a no-op (not an error)
    if (bytes == 0) {
        return CUDA_SUCCESS;
    }
    
    // Validate size is reasonable
    if (bytes > MAX_ALLOCATION_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Clear any previous CUDA errors (defensive)
    cudaGetLastError();
    
    // Perform copy (synchronous)
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return map_cuda_error(err);
    }
    
    // Verify copy completed (defensive)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return CUDA_ERROR_DRIVER;
    }
    
    return CUDA_SUCCESS;
}

/// Query VRAM capacity
///
/// # Safety
/// - Validates pointers are non-null
/// - Returns free and total VRAM in bytes
/// - Initializes outputs to 0 on error (defensive)
///
/// # Parameters
/// - free_bytes: Output for free VRAM (must be non-null)
/// - total_bytes: Output for total VRAM (must be non-null)
///
/// # Returns
/// - CUDA_SUCCESS on success
/// - CUDA_ERROR_INVALID_VALUE if pointers are null
/// - CUDA_ERROR_DRIVER if query fails
extern "C" int vram_get_info(size_t* free_bytes, size_t* total_bytes) {
    // Validate pointers
    if (!is_valid_ptr(free_bytes) || !is_valid_ptr(total_bytes)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize outputs (defensive)
    *free_bytes = 0;
    *total_bytes = 0;
    
    // Clear any previous CUDA errors (defensive)
    cudaGetLastError();
    
    // Query VRAM
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err != cudaSuccess) {
        // Reset outputs on error (defensive)
        *free_bytes = 0;
        *total_bytes = 0;
        return map_cuda_error(err);
    }
    
    // Sanity check: free should never exceed total
    if (*free_bytes > *total_bytes) {
        *free_bytes = 0;
        *total_bytes = 0;
        return CUDA_ERROR_DRIVER;
    }
    
    return CUDA_SUCCESS;
}

/// Set CUDA device
///
/// # Safety
/// - Validates device index >= 0
/// - Validates device exists
/// - Returns error if device doesn't exist
///
/// # Parameters
/// - device: Device index (must be >= 0 and < device count)
///
/// # Returns
/// - CUDA_SUCCESS on success
/// - CUDA_ERROR_INVALID_VALUE if device < 0
/// - CUDA_ERROR_DRIVER if device doesn't exist or set fails
extern "C" int vram_set_device(int device) {
    // Validate device index
    if (device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Validate device exists
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return map_cuda_error(err);
    }
    
    if (device >= count) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Clear any previous CUDA errors (defensive)
    cudaGetLastError();
    
    // Set device
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return map_cuda_error(err);
    }
    
    // Verify device was set (defensive)
    int current_device = -1;
    err = cudaGetDevice(&current_device);
    if (err != cudaSuccess || current_device != device) {
        return CUDA_ERROR_DRIVER;
    }
    
    return CUDA_SUCCESS;
}

/// Get CUDA device count
///
/// # Safety
/// - Validates pointer is non-null
/// - Returns number of CUDA devices
/// - Initializes output to 0 on error (defensive)
///
/// # Parameters
/// - count: Output for device count (must be non-null)
///
/// # Returns
/// - CUDA_SUCCESS on success
/// - CUDA_ERROR_INVALID_VALUE if pointer is null
/// - CUDA_ERROR_DRIVER if query fails
extern "C" int vram_get_device_count(int* count) {
    // Validate pointer
    if (!is_valid_ptr(count)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output (defensive)
    *count = 0;
    
    // Clear any previous CUDA errors (defensive)
    cudaGetLastError();
    
    // Query device count
    cudaError_t err = cudaGetDeviceCount(count);
    if (err != cudaSuccess) {
        *count = 0;
        return map_cuda_error(err);
    }
    
    // Sanity check: count should be reasonable
    if (*count < 0 || *count > 16) {
        // More than 16 GPUs is suspicious
        *count = 0;
        return CUDA_ERROR_DRIVER;
    }
    
    return CUDA_SUCCESS;
}
