# FT-015: Embedding Lookup Kernel

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: M (2 days)  
**Days**: 28 - 29  
**Spec Ref**: M0-W-1430, CUDA-5030

---

## Story Description

Implement CUDA kernel for embedding lookup that retrieves token embeddings from weight matrix. This is the first layer of transformer inference and is shared across all model architectures.

---

## Acceptance Criteria

- [ ] CUDA kernel performs embedding lookup: `embeddings[i] = weight_matrix[token_ids[i]]`
- [ ] Supports batch processing (multiple tokens)
- [ ] Handles edge cases (invalid token IDs, empty input)
- [ ] Optimized memory access pattern (coalesced reads)
- [ ] Unit tests validate correctness with known inputs
- [ ] Integration tests validate with real model weights
- [ ] Kernel launch parameters optimized for GPU utilization
- [ ] Error handling for out-of-bounds token IDs
- [ ] Support for FP16 and FP32 embeddings

---

## Dependencies

### Upstream (Blocks This Story)
- FT-013: Device memory RAII (Expected completion: Day 26)

### Downstream (This Story Blocks)
- FT-024: HTTP-FFI-CUDA integration needs embedding kernel
- Llama/GPT teams need embedding kernel for forward pass

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/embedding.cu` - Embedding kernel
- `bin/worker-orcd/cuda/kernels/embedding.cuh` - Kernel header
- `bin/worker-orcd/cuda/tests/embedding_test.cu` - Unit tests

### Key Interfaces
```cuda
// embedding.cuh
#ifndef WORKER_EMBEDDING_CUH
#define WORKER_EMBEDDING_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace kernels {

/**
 * Embedding lookup kernel (FP16).
 * 
 * Retrieves embeddings for input token IDs from weight matrix.
 * 
 * @param token_ids Input token IDs [batch_size]
 * @param weight_matrix Embedding weight matrix [vocab_size, hidden_dim]
 * @param embeddings Output embeddings [batch_size, hidden_dim]
 * @param batch_size Number of tokens
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 */
__global__ void embedding_lookup_fp16(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size
);

/**
 * Embedding lookup kernel (FP32).
 */
__global__ void embedding_lookup_fp32(
    const int* token_ids,
    const float* weight_matrix,
    float* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size
);

/**
 * Launch embedding lookup kernel.
 * 
 * @param token_ids Device pointer to token IDs
 * @param weight_matrix Device pointer to embedding weights
 * @param embeddings Device pointer to output embeddings
 * @param batch_size Number of tokens
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream
 */
void launch_embedding_lookup(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace worker

#endif // WORKER_EMBEDDING_CUH

// embedding.cu
#include "embedding.cuh"
#include <stdio.h>

namespace worker {
namespace kernels {

__global__ void embedding_lookup_fp16(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size
) {
    // Each thread handles one element of one embedding
    int token_idx = blockIdx.x;  // Which token in batch
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;  // Which dimension
    
    if (token_idx >= batch_size || dim_idx >= hidden_dim) {
        return;
    }
    
    // Get token ID
    int token_id = token_ids[token_idx];
    
    // Bounds check
    if (token_id < 0 || token_id >= vocab_size) {
        // Invalid token ID, set to zero
        embeddings[token_idx * hidden_dim + dim_idx] = __float2half(0.0f);
        return;
    }
    
    // Lookup embedding
    // weight_matrix layout: [vocab_size, hidden_dim]
    // Coalesced memory access: consecutive threads access consecutive elements
    half value = weight_matrix[token_id * hidden_dim + dim_idx];
    embeddings[token_idx * hidden_dim + dim_idx] = value;
}

__global__ void embedding_lookup_fp32(
    const int* token_ids,
    const float* weight_matrix,
    float* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= batch_size || dim_idx >= hidden_dim) {
        return;
    }
    
    int token_id = token_ids[token_idx];
    
    if (token_id < 0 || token_id >= vocab_size) {
        embeddings[token_idx * hidden_dim + dim_idx] = 0.0f;
        return;
    }
    
    float value = weight_matrix[token_id * hidden_dim + dim_idx];
    embeddings[token_idx * hidden_dim + dim_idx] = value;
}

void launch_embedding_lookup(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
) {
    // Kernel launch configuration
    // Grid: (batch_size, ceil(hidden_dim / 256))
    // Block: 256 threads
    int threads_per_block = 256;
    int blocks_y = (hidden_dim + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_y);
    dim3 block(threads_per_block);
    
    embedding_lookup_fp16<<<grid, block, 0, stream>>>(
        token_ids,
        weight_matrix,
        embeddings,
        batch_size,
        hidden_dim,
        vocab_size
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Embedding kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels
} // namespace worker

// Unit tests
// cuda/tests/embedding_test.cu
#include <gtest/gtest.h>
#include "embedding.cuh"
#include <cuda_runtime.h>
#include <vector>

using namespace worker::kernels;

class EmbeddingKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Allocate device memory
        cudaMalloc(&d_token_ids, batch_size * sizeof(int));
        cudaMalloc(&d_weight_matrix, vocab_size * hidden_dim * sizeof(half));
        cudaMalloc(&d_embeddings, batch_size * hidden_dim * sizeof(half));
    }
    
    void TearDown() override {
        cudaFree(d_token_ids);
        cudaFree(d_weight_matrix);
        cudaFree(d_embeddings);
    }
    
    int batch_size = 4;
    int hidden_dim = 128;
    int vocab_size = 1000;
    
    int* d_token_ids = nullptr;
    half* d_weight_matrix = nullptr;
    half* d_embeddings = nullptr;
};

TEST_F(EmbeddingKernelTest, BasicLookup) {
    // Host data
    std::vector<int> h_token_ids = {0, 1, 2, 3};
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim);
    
    // Initialize weights with token_id * 0.1 for easy verification
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            h_weight_matrix[i * hidden_dim + j] = __float2half(i * 0.1f);
        }
    }
    
    // Copy to device
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_embedding_lookup(d_token_ids, d_weight_matrix, d_embeddings, batch_size, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    // Copy result back
    std::vector<half> h_embeddings(batch_size * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings, batch_size * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify
    for (int i = 0; i < batch_size; ++i) {
        float expected = i * 0.1f;
        float actual = __half2float(h_embeddings[i * hidden_dim]);
        EXPECT_NEAR(actual, expected, 0.01f) << "Token " << i;
    }
}

TEST_F(EmbeddingKernelTest, OutOfBoundsTokenID) {
    std::vector<int> h_token_ids = {0, vocab_size + 10, -1, 1};  // Invalid IDs
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim, __float2half(1.0f));
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup(d_token_ids, d_weight_matrix, d_embeddings, batch_size, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(batch_size * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings, batch_size * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Token 0 should be valid (1.0)
    EXPECT_NEAR(__half2float(h_embeddings[0]), 1.0f, 0.01f);
    
    // Token 1 (out of bounds) should be zero
    EXPECT_NEAR(__half2float(h_embeddings[hidden_dim]), 0.0f, 0.01f);
    
    // Token 2 (negative) should be zero
    EXPECT_NEAR(__half2float(h_embeddings[2 * hidden_dim]), 0.0f, 0.01f);
    
    // Token 3 should be valid (1.0)
    EXPECT_NEAR(__half2float(h_embeddings[3 * hidden_dim]), 1.0f, 0.01f);
}

TEST_F(EmbeddingKernelTest, LargeHiddenDim) {
    // Test with hidden_dim > 256 (multiple blocks per token)
    hidden_dim = 1024;
    
    cudaFree(d_weight_matrix);
    cudaFree(d_embeddings);
    cudaMalloc(&d_weight_matrix, vocab_size * hidden_dim * sizeof(half));
    cudaMalloc(&d_embeddings, batch_size * hidden_dim * sizeof(half));
    
    std::vector<int> h_token_ids = {5, 10, 15, 20};
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim);
    
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            h_weight_matrix[i * hidden_dim + j] = __float2half(i * 0.1f);
        }
    }
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup(d_token_ids, d_weight_matrix, d_embeddings, batch_size, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(batch_size * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings, batch_size * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify all dimensions are correct
    for (int i = 0; i < batch_size; ++i) {
        float expected = h_token_ids[i] * 0.1f;
        for (int j = 0; j < hidden_dim; ++j) {
            float actual = __half2float(h_embeddings[i * hidden_dim + j]);
            EXPECT_NEAR(actual, expected, 0.01f) << "Token " << i << ", dim " << j;
        }
    }
}
```

### Implementation Notes
- Coalesced memory access: consecutive threads read consecutive memory locations
- Grid configuration: (batch_size, ceil(hidden_dim / 256))
- Block size: 256 threads (optimal for most GPUs)
- Bounds checking prevents crashes on invalid token IDs
- FP16 for memory efficiency, FP32 for higher precision if needed
- Kernel launch errors checked immediately after launch
- Supports arbitrary hidden dimensions (not limited to 256)

---

## Testing Strategy

### Unit Tests
- Test basic embedding lookup with known values
- Test out-of-bounds token IDs return zero
- Test negative token IDs return zero
- Test large hidden dimensions (>256)
- Test batch size = 1 (single token)
- Test empty batch (batch_size = 0)

### Integration Tests
- Test with real model embedding weights
- Test with Qwen2.5-0.5B vocabulary (151936 tokens)
- Test memory access pattern is coalesced (use nvprof)

### Manual Verification
1. Run unit tests: `./build/tests/embedding_test`
2. Profile kernel: `nvprof --metrics gld_efficiency ./build/tests/embedding_test`
3. Verify coalesced memory access (>80% efficiency)

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing (3+ tests)
- [ ] Documentation updated (kernel docs, launch function docs)
- [ ] Kernel profiled for memory efficiency
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.4 Required Kernels (M0-W-1430, CUDA-5030)
- Related Stories: FT-013 (device memory), FT-016 (cuBLAS GEMM)
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

## 🔍 Testing Requirements

**Added by**: Testing Team (test-harness/TEAM_RESPONSIBILITIES.md)

### Unit Tests (MUST implement)

**Critical Path Coverage**:
- **Test basic embedding lookup with known values** (M0-W-1430)
  - Given: Weight matrix with token_id i → embedding value i*0.1
  - When: launch_embedding_lookup([0,1,2,3], weights, embeddings, 4, 128, 1000)
  - Then: embeddings[0] = 0.0, embeddings[128] = 0.1, embeddings[256] = 0.2, embeddings[384] = 0.3
  - **Why critical**: Core kernel functionality must be correct

- **Test out-of-bounds token IDs return zero** (M0-W-1430)
  - Given: vocab_size = 1000, token_ids = [0, 1001, -1, 1]
  - When: Kernel executes
  - Then: embeddings[0] valid, embeddings[128] = 0.0 (OOB), embeddings[256] = 0.0 (negative), embeddings[384] valid
  - **Why critical**: Prevents crashes and undefined behavior

- **Test negative token IDs return zero**
  - Given: token_ids = [-1, -100]
  - When: Kernel executes
  - Then: All embeddings = 0.0
  - **Why critical**: Defensive programming

- **Test large hidden dimensions (>256)** (M0-W-1430)
  - Given: hidden_dim = 1024 (requires multiple blocks per token)
  - When: Kernel executes
  - Then: All 1024 dimensions populated correctly
  - **Why critical**: Real models have large hidden dims

- **Test batch size = 1 (single token)**
  - Given: batch_size = 1
  - When: Kernel executes
  - Then: Single embedding retrieved correctly
  - **Why critical**: Edge case for inference

- **Test empty batch (batch_size = 0)**
  - Given: batch_size = 0
  - When: Kernel executes
  - Then: No crash, no-op
  - **Why critical**: Defensive programming

### Integration Tests (MUST implement)

- **Test with real model embedding weights** (M0-W-1430)
  - Given: Qwen2.5-0.5B embedding weights loaded
  - When: Lookup token IDs from real prompt
  - Then: Embeddings match expected values (validated against reference)
  - **Why critical**: Real-world validation

- **Test with Qwen2.5-0.5B vocabulary (151936 tokens)**
  - Given: vocab_size = 151936, hidden_dim = 896
  - When: Kernel executes
  - Then: Correct embeddings retrieved, no crashes
  - **Why critical**: Real vocabulary size

- **Test memory access pattern is coalesced** (performance)
  - Given: Kernel execution
  - When: Profiled with nvprof --metrics gld_efficiency
  - Then: Global load efficiency > 80%
  - **Why critical**: Performance optimization validation

### Property Tests (SHOULD implement)

- **Property: Embedding lookup is deterministic**
  - Given: Same token_ids, same weights
  - When: Kernel executed multiple times
  - Then: Identical embeddings every time
  - **Why valuable**: Validates determinism requirement

### BDD Scenarios (VERY IMPORTANT - MUST implement)

**Feature**: Embedding Lookup Kernel

```gherkin
Scenario: Worker retrieves token embeddings for inference
  Given a worker with Qwen2.5-0.5B model loaded
  When the worker performs embedding lookup for token IDs [1, 2, 3]
  Then the kernel retrieves 3 embeddings of dimension 896
  And each embedding matches the model's weight matrix
  And no out-of-bounds access occurs

Scenario: Worker handles invalid token IDs gracefully
  Given a worker with vocab_size = 1000
  When the worker performs embedding lookup with token ID 1500
  Then the kernel returns zero embedding for invalid token
  And no crash occurs
  And valid tokens are processed correctly

Scenario: Worker processes large batches efficiently
  Given a worker with hidden_dim = 1024
  When the worker performs embedding lookup for 128 tokens
  Then all 128 embeddings are retrieved correctly
  And memory access is coalesced (>80% efficiency)
```

### Test Artifacts (MUST produce)

- **Unit test report**: Pass/fail for each test
- **Kernel profiling**: Memory efficiency metrics (gld_efficiency)
- **BDD scenario results**: Pass/fail with embedding traces
- **Performance baseline**: Latency for common batch sizes

### Acceptance Criteria for Testing

- ✅ All unit tests pass (6+ tests covering critical paths and edge cases)
- ✅ All integration tests pass (3+ tests with real model data)
- ✅ All BDD scenarios pass (3 scenarios validating kernel behavior)
- ✅ Memory coalescing efficiency > 80% (verified with nvprof)
- ✅ All tests produce verifiable artifacts

### False Positive Prevention

**CRITICAL**: Tests MUST validate actual kernel output, not mock data.

❌ **FORBIDDEN**:
```cpp
// Pre-filling embeddings that kernel should compute
std::fill(embeddings.begin(), embeddings.end(), 0.5f);
launch_embedding_lookup(...);
assert(embeddings[0] == 0.5f);  // FALSE POSITIVE: kernel didn't run
```

✅ **REQUIRED**:
```cpp
// Let kernel compute embeddings, verify correctness
std::fill(embeddings.begin(), embeddings.end(), -999.0f);  // Sentinel
launch_embedding_lookup(token_ids, weights, embeddings, ...);
assert(embeddings[0] != -999.0f);  // Kernel ran
assert(embeddings[0] == expected_from_weights);  // Correct value
```

### Test Execution Commands

```bash
# Unit tests
./build/tests/embedding_test

# Kernel profiling
nvprof --metrics gld_efficiency ./build/tests/embedding_test

# Integration tests
cargo test --features cuda --test embedding_integration

# BDD scenarios
cargo run --bin bdd-runner -- --features embedding_lookup
```

### Dependencies for Testing

- **Upstream**: FT-013 (DeviceMemory)
- **Test infrastructure**: Google Test (C++), nvprof, BDD runner
- **Test data**: Qwen2.5-0.5B embedding weights (for integration tests)

---
**Testing requirements added by Testing Team 🔍**

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team (v0.2.0)

Hey Foundation Team! 👋 We're here to help you make kernel operations **delightfully debuggable**!

### Quick Start (v0.2.0 Builder API)

We just shipped v0.2.0 with a **builder pattern** that's 43% less boilerplate:

```rust
use observability_narration_core::{Narration, ACTOR_INFERENCE_ENGINE};

// In your kernel launch code:
Narration::new(ACTOR_INFERENCE_ENGINE, "kernel_launch", "embedding_lookup")
    .human(format!("Launching embedding lookup kernel (batch={}, hidden_dim={})", 
                   batch_size, hidden_dim))
    .device(format!("GPU{}", device_id))
    .emit();
```

The builder automatically adds `emitted_by`, `emitted_at_ms`, and secret redaction!

### Events to Narrate

1. **Kernel launched**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "kernel_launch",
       target: "embedding_lookup".to_string(),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Launching embedding lookup kernel (batch={}, hidden_dim={})", batch_size, hidden_dim),
       ..Default::default()
   });
   ```

2. **Kernel completed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "kernel_complete",
       target: "embedding_lookup".to_string(),
       device: Some(format!("GPU{}", device_id)),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("Embedding lookup completed ({} ms)", elapsed.as_millis()),
       ..Default::default()
   });
   ```

3. **Invalid token ID detected**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "kernel_launch",
       target: "embedding_lookup".to_string(),
       error_kind: Some("invalid_token_id".to_string()),
       human: format!("Invalid token ID {} (vocab_size={})", token_id, vocab_size),
       ..Default::default()
   });
   ```

**Why this matters**: Embedding lookup is the first transformer layer. Narration helps track kernel performance and diagnose invalid token IDs.

### Testing Your Narration

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_kernel_launch_narrates() {
    let adapter = CaptureAdapter::install();
    
    // Your kernel launch
    launch_embedding_lookup(...);
    
    adapter.assert_includes("embedding lookup");
    adapter.assert_field("actor", "inference-engine");
}
```

Run with: `cargo test --features test-support`

### Need Help?

- **Full docs**: `bin/shared-crates/narration-core/README.md`
- **Quick start**: `bin/shared-crates/narration-core/QUICKSTART.md`
- **Field reference**: See README section "NarrationFields Reference"

We're watching your narration with ❤️!

---
*Narration guidance added by Narration-Core Team v0.2.0 🎀*
