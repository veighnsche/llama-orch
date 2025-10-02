// Mock CUDA implementation for testing without GPU
//
// This provides stub implementations of CUDA functions for testing.
// These are only used when CUDA is not available.

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

// Error codes (matches vram_ops.cu)
const int CUDA_SUCCESS = 0;
const int CUDA_ERROR_ALLOCATION_FAILED = 1;
const int CUDA_ERROR_INVALID_VALUE = 2;
const int CUDA_ERROR_MEMCPY_FAILED = 3;
const int CUDA_ERROR_DRIVER = 4;

// Maximum allocation size (100GB) - matches vram_ops.cu
const size_t MAX_ALLOCATION_SIZE = 100ULL * 1024ULL * 1024ULL * 1024ULL;

// Allocation tracking
#define MAX_ALLOCATIONS 1024

typedef struct {
    void* ptr;
    size_t size;
} AllocationEntry;

static AllocationEntry allocations[MAX_ALLOCATIONS];
static size_t allocation_count = 0;
static size_t mock_allocated_bytes = 0;

// Mock VRAM allocation (uses malloc with alignment)
int vram_malloc(void** ptr, size_t bytes) {
    if (ptr == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to null (defensive)
    *ptr = NULL;
    
    if (bytes == 0 || bytes > MAX_ALLOCATION_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Allocate with 256-byte alignment (like real CUDA)
    #ifdef _WIN32
        *ptr = _aligned_malloc(bytes, 256);
    #else
        if (posix_memalign(ptr, 256, bytes) != 0) {
            *ptr = NULL;
        }
    #endif
    
    if (*ptr == NULL) {
        return CUDA_ERROR_ALLOCATION_FAILED;
    }
    
    // Verify alignment
    if (((uintptr_t)*ptr) % 256 != 0) {
        free(*ptr);
        *ptr = NULL;
        return CUDA_ERROR_DRIVER;
    }
    
    // Track allocation
    if (allocation_count < MAX_ALLOCATIONS) {
        allocations[allocation_count].ptr = *ptr;
        allocations[allocation_count].size = bytes;
        allocation_count++;
        mock_allocated_bytes += bytes;
    }
    
    return CUDA_SUCCESS;
}

// Mock VRAM free
int vram_free(void* ptr) {
    if (ptr == NULL) {
        return CUDA_SUCCESS;
    }
    
    // Find and remove allocation
    for (size_t i = 0; i < allocation_count; i++) {
        if (allocations[i].ptr == ptr) {
            // Track deallocation
            size_t freed_bytes = allocations[i].size;
            mock_allocated_bytes -= freed_bytes;
            
            // Remove from tracking (shift remaining entries)
            for (size_t j = i; j < allocation_count - 1; j++) {
                allocations[j] = allocations[j + 1];
            }
            allocation_count--;
            
            free(ptr);
            return CUDA_SUCCESS;
        }
    }
    
    // Pointer not found - still free it (defensive)
    free(ptr);
    return CUDA_SUCCESS;
}

// Mock host-to-device copy (just memcpy)
int vram_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    if (dst == NULL || src == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    if (bytes == 0) {
        return CUDA_SUCCESS;
    }
    
    memcpy(dst, src, bytes);
    return CUDA_SUCCESS;
}

// Mock device-to-host copy (just memcpy)
int vram_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    if (dst == NULL || src == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    if (bytes == 0) {
        return CUDA_SUCCESS;
    }
    
    memcpy(dst, src, bytes);
    return CUDA_SUCCESS;
}

// Mock VRAM info (returns fake values with tracking)
int vram_get_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == NULL || total_bytes == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Mock: Use environment variable or default to 24GB
    // Can be configured in MB (MOCK_VRAM_MB) or GB (MOCK_VRAM_GB)
    // For testing, can be set to any value (no minimum enforced in mock)
    const char* vram_mb_env = getenv("MOCK_VRAM_MB");
    const char* vram_gb_env = getenv("MOCK_VRAM_GB");
    
    if (vram_mb_env) {
        // Use MB for fine-grained control (useful for testing)
        size_t vram_mb = (size_t)atoi(vram_mb_env);
        *total_bytes = vram_mb * 1024ULL * 1024ULL;
    } else if (vram_gb_env) {
        // Use GB for production-like sizes
        size_t vram_gb = (size_t)atoi(vram_gb_env);
        *total_bytes = vram_gb * 1024ULL * 1024ULL * 1024ULL;
    } else {
        // Default to 24GB
        *total_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
    }
    
    // Free = total - allocated
    if (mock_allocated_bytes > *total_bytes) {
        *free_bytes = 0;
    } else {
        *free_bytes = *total_bytes - mock_allocated_bytes;
    }
    
    return CUDA_SUCCESS;
}

// Mock set device (no-op)
int vram_set_device(int device) {
    if (device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    return CUDA_SUCCESS;
}

// Mock get device count (returns 1)
int vram_get_device_count(int* count) {
    if (count == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    *count = 1;
    return CUDA_SUCCESS;
}

// Reset mock VRAM state (for testing)
// WARNING: This leaks memory but is safe for BDD tests
// We reset the tracking counters but don't free the memory because
// Rust's SafeCudaPtr will free it later when dropped
void vram_reset_mock_state(void) {
    // Reset tracking counters (but don't free - Rust will do that)
    allocation_count = 0;
    mock_allocated_bytes = 0;
}
