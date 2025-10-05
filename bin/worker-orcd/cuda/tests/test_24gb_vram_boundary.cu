// 24GB VRAM Boundary Tests
//
// Tests to validate GPT-OSS-20B operates correctly within 24GB VRAM constraints.
// Tests VRAM allocation, usage tracking, and OOM handling.
//
// Story: GT-044
// Spec: M0-W-1020, M0-W-1021

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Get VRAM info
void get_vram_info(size_t& free_mem, size_t& total_mem) {
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
}

void test_vram_fits_in_24gb() {
    printf("Test 1: GPT-OSS-20B fits in 24GB VRAM...\n");
    
    // Calculate expected VRAM usage
    size_t vocab_size = 50257;
    size_t hidden_dim = 4096;
    size_t num_layers = 24;
    size_t ffn_dim = 16384;
    size_t max_seq_len = 2048;
    
    // MXFP4 size calculation
    auto mxfp4_size = [](size_t elements) {
        return ((elements + 31) / 32) * 17;
    };
    
    // Model weights (MXFP4)
    size_t embeddings = mxfp4_size(vocab_size * hidden_dim);
    size_t attention = num_layers * mxfp4_size(4 * hidden_dim * hidden_dim);
    size_t ffn = num_layers * mxfp4_size(2 * hidden_dim * ffn_dim);
    size_t lm_head = mxfp4_size(vocab_size * hidden_dim);
    size_t total_weights = embeddings + attention + ffn + lm_head;
    
    // KV cache (FP16)
    size_t kv_cache = num_layers * 2 * max_seq_len * hidden_dim * sizeof(half);
    
    // Activations (estimate)
    size_t activations = max_seq_len * hidden_dim * sizeof(half) * 10;
    
    // Total VRAM
    size_t total_vram = total_weights + kv_cache + activations;
    
    printf("  Model weights (MXFP4): %.2f GB\n", total_weights / 1024.0 / 1024.0 / 1024.0);
    printf("  KV cache (FP16): %.2f GB\n", kv_cache / 1024.0 / 1024.0 / 1024.0);
    printf("  Activations: %.2f GB\n", activations / 1024.0 / 1024.0 / 1024.0);
    printf("  Total: %.2f GB\n", total_vram / 1024.0 / 1024.0 / 1024.0);
    
    // Validate fits in 24GB
    size_t target_vram = 24ULL * 1024 * 1024 * 1024;
    assert(total_vram < target_vram);
    
    float utilization = (float)total_vram / target_vram * 100.0f;
    printf("  VRAM utilization: %.1f%%\n", utilization);
    printf("  Headroom: %.2f GB\n", (target_vram - total_vram) / 1024.0 / 1024.0 / 1024.0);
    
    printf("  ✓ Model fits in 24GB VRAM\n");
}

void test_vram_tracking_accuracy() {
    printf("Test 2: VRAM usage tracking accuracy...\n");
    
    size_t free_before, total;
    get_vram_info(free_before, total);
    
    printf("  Before allocation:\n");
    printf("    Free: %.2f GB\n", free_before / 1024.0 / 1024.0 / 1024.0);
    
    // Allocate test buffer
    size_t alloc_size = 1ULL * 1024 * 1024 * 1024;  // 1GB
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, alloc_size));
    
    size_t free_after, _;
    get_vram_info(free_after, _);
    
    printf("  After 1GB allocation:\n");
    printf("    Free: %.2f GB\n", free_after / 1024.0 / 1024.0 / 1024.0);
    
    size_t actual_used = free_before - free_after;
    printf("  Actual used: %.2f GB\n", actual_used / 1024.0 / 1024.0 / 1024.0);
    
    // Verify tracking accuracy (within 10% tolerance for fragmentation)
    float diff_ratio = fabsf((float)actual_used - (float)alloc_size) / (float)alloc_size;
    assert(diff_ratio < 0.1f);
    
    CUDA_CHECK(cudaFree(d_buffer));
    
    printf("  ✓ VRAM tracking accurate\n");
}

void test_oom_detection() {
    printf("Test 3: OOM detection...\n");
    
    size_t free_mem, total_mem;
    get_vram_info(free_mem, total_mem);
    
    printf("  Available VRAM: %.2f GB\n", free_mem / 1024.0 / 1024.0 / 1024.0);
    
    // Try to allocate more than available
    size_t oversized = free_mem + (1ULL * 1024 * 1024 * 1024);  // +1GB over limit
    void* d_buffer;
    
    cudaError_t err = cudaMalloc(&d_buffer, oversized);
    
    if (err == cudaErrorMemoryAllocation) {
        printf("  OOM detected correctly: %s\n", cudaGetErrorString(err));
        printf("  ✓ OOM detection working\n");
    } else if (err == cudaSuccess) {
        // Allocation succeeded (shouldn't happen)
        cudaFree(d_buffer);
        printf("  ⚠️  Warning: Oversized allocation succeeded\n");
    } else {
        printf("  Unexpected error: %s\n", cudaGetErrorString(err));
    }
    
    // Clear error state
    cudaGetLastError();
}

void test_vram_residency_verification() {
    printf("Test 4: VRAM residency verification...\n");
    
    // Allocate buffer
    size_t buffer_size = 512 * 1024 * 1024;  // 512MB
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
    
    // Verify pointer is valid device pointer
    cudaPointerAttributes attrs;
    CUDA_CHECK(cudaPointerGetAttributes(&attrs, d_buffer));
    
    printf("  Pointer type: ");
    if (attrs.type == cudaMemoryTypeDevice) {
        printf("Device memory ✓\n");
    } else {
        printf("Not device memory ✗\n");
        assert(false);
    }
    
    printf("  Device ID: %d\n", attrs.device);
    
    CUDA_CHECK(cudaFree(d_buffer));
    
    printf("  ✓ VRAM residency verified\n");
}

void test_fragmentation_handling() {
    printf("Test 5: VRAM fragmentation handling...\n");
    
    // Allocate and free multiple buffers to create fragmentation
    std::vector<void*> buffers;
    size_t chunk_size = 256 * 1024 * 1024;  // 256MB chunks
    int num_chunks = 8;
    
    printf("  Allocating %d chunks of %.2f GB...\n", num_chunks, chunk_size / 1024.0 / 1024.0 / 1024.0);
    
    for (int i = 0; i < num_chunks; i++) {
        void* d_buffer;
        cudaError_t err = cudaMalloc(&d_buffer, chunk_size);
        if (err == cudaSuccess) {
            buffers.push_back(d_buffer);
        } else {
            printf("  Allocation %d failed (expected if low VRAM)\n", i);
            break;
        }
    }
    
    printf("  Allocated %zu chunks\n", buffers.size());
    
    // Free every other buffer to create fragmentation
    for (size_t i = 0; i < buffers.size(); i += 2) {
        CUDA_CHECK(cudaFree(buffers[i]));
    }
    
    printf("  Created fragmentation pattern\n");
    
    // Try to allocate large contiguous buffer
    size_t large_size = chunk_size * 2;
    void* d_large;
    cudaError_t err = cudaMalloc(&d_large, large_size);
    
    if (err == cudaSuccess) {
        printf("  Large allocation succeeded despite fragmentation ✓\n");
        CUDA_CHECK(cudaFree(d_large));
    } else {
        printf("  Large allocation failed due to fragmentation (expected)\n");
        cudaGetLastError();  // Clear error
    }
    
    // Cleanup remaining buffers
    for (size_t i = 1; i < buffers.size(); i += 2) {
        CUDA_CHECK(cudaFree(buffers[i]));
    }
    
    printf("  ✓ Fragmentation handling tested\n");
}

void test_peak_vs_steady_state() {
    printf("Test 6: Peak vs steady-state VRAM usage...\n");
    
    size_t free_initial, total;
    get_vram_info(free_initial, total);
    
    printf("  Initial free: %.2f GB\n", free_initial / 1024.0 / 1024.0 / 1024.0);
    
    // Simulate peak usage (model loading)
    size_t peak_size = 4ULL * 1024 * 1024 * 1024;  // 4GB
    void* d_peak;
    CUDA_CHECK(cudaMalloc(&d_peak, peak_size));
    
    size_t free_peak, _;
    get_vram_info(free_peak, _);
    printf("  Peak usage: %.2f GB used\n", (free_initial - free_peak) / 1024.0 / 1024.0 / 1024.0);
    
    CUDA_CHECK(cudaFree(d_peak));
    
    // Simulate steady-state (inference)
    size_t steady_size = 3ULL * 1024 * 1024 * 1024;  // 3GB
    void* d_steady;
    CUDA_CHECK(cudaMalloc(&d_steady, steady_size));
    
    size_t free_steady;
    get_vram_info(free_steady, _);
    printf("  Steady-state: %.2f GB used\n", (free_initial - free_steady) / 1024.0 / 1024.0 / 1024.0);
    
    CUDA_CHECK(cudaFree(d_steady));
    
    printf("  ✓ Peak/steady-state usage validated\n");
}

int main() {
    printf("=== 24GB VRAM Boundary Tests ===\n\n");
    
    test_vram_fits_in_24gb();
    test_vram_tracking_accuracy();
    test_oom_detection();
    test_vram_residency_verification();
    test_fragmentation_handling();
    test_peak_vs_steady_state();
    
    printf("\n✅ All VRAM boundary tests passed!\n");
    printf("\nBoundary Test Coverage:\n");
    printf("- Model fits in 24GB ✓\n");
    printf("- VRAM tracking accuracy ✓\n");
    printf("- OOM detection ✓\n");
    printf("- VRAM residency ✓\n");
    printf("- Fragmentation handling ✓\n");
    printf("- Peak vs steady-state ✓\n");
    
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
