# FT-017: Temperature Scaling Kernel

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: S (1 day)  
**Days**: 32 - 32  
**Spec Ref**: M0-W-1032, M0-W-1421, KERNEL-SAMPLE-003

---

## Story Description

Implement CUDA kernel for temperature scaling of logits before sampling. This controls randomness in token generation: temp=0 for greedy (testing), temp>0 for stochastic (production).

---

## Acceptance Criteria

- [ ] CUDA kernel divides logits by temperature: `logit[i] /= temperature`
- [ ] Handles temperature = 0.0 (greedy sampling, no division)
- [ ] Handles temperature range 0.0-2.0
- [ ] Unit tests validate scaling correctness
- [ ] Integration tests validate with sampling pipeline
- [ ] Kernel optimized for memory bandwidth
- [ ] Error handling for invalid temperature values
- [ ] Support for FP16 and FP32 logits

---

## Dependencies

### Upstream (Blocks This Story)
- FT-016: cuBLAS GEMM wrapper (Expected completion: Day 31)

### Downstream (This Story Blocks)
- FT-018: Greedy sampling needs temperature scaling
- FT-019: Stochastic sampling needs temperature scaling

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/sampling.cu` - Sampling kernels
- `bin/worker-orcd/cuda/kernels/sampling.cuh` - Kernel header
- `bin/worker-orcd/cuda/tests/sampling_test.cu` - Unit tests

### Key Interfaces
```cuda
// sampling.cuh
#ifndef WORKER_SAMPLING_CUH
#define WORKER_SAMPLING_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace kernels {

/**
 * Apply temperature scaling to logits.
 * 
 * For temperature > 0: logits[i] /= temperature
 * For temperature == 0: no-op (greedy sampling)
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param temperature Sampling temperature (0.0-2.0)
 */
__global__ void temperature_scale_fp32(
    float* logits,
    int vocab_size,
    float temperature
);

__global__ void temperature_scale_fp16(
    half* logits,
    int vocab_size,
    float temperature
);

/**
 * Launch temperature scaling kernel.
 */
void launch_temperature_scale(
    float* logits,
    int vocab_size,
    float temperature,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace worker

#endif // WORKER_SAMPLING_CUH

// sampling.cu
#include "sampling.cuh"

namespace worker {
namespace kernels {

__global__ void temperature_scale_fp32(
    float* logits,
    int vocab_size,
    float temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= vocab_size) {
        return;
    }
    
    // Temperature == 0 means greedy sampling (no scaling)
    if (temperature == 0.0f) {
        return;
    }
    
    // Validate temperature range (defensive)
    if (temperature < 0.0f || temperature > 2.0f) {
        // Invalid temperature, skip scaling
        return;
    }
    
    // Apply temperature scaling
    logits[idx] /= temperature;
}

__global__ void temperature_scale_fp16(
    half* logits,
    int vocab_size,
    float temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= vocab_size) {
        return;
    }
    
    if (temperature == 0.0f) {
        return;
    }
    
    if (temperature < 0.0f || temperature > 2.0f) {
        return;
    }
    
    // Convert to float, scale, convert back
    float logit_f = __half2float(logits[idx]);
    logit_f /= temperature;
    logits[idx] = __float2half(logit_f);
}

void launch_temperature_scale(
    float* logits,
    int vocab_size,
    float temperature,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    
    temperature_scale_fp32<<<num_blocks, threads_per_block, 0, stream>>>(
        logits,
        vocab_size,
        temperature
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Temperature scale kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

} // namespace kernels
} // namespace worker

// Unit tests
// cuda/tests/sampling_test.cu
#include <gtest/gtest.h>
#include "sampling.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace worker::kernels;

class TemperatureScaleTest : public ::testing::Test {
protected:
    void SetUp() override {
        vocab_size = 1000;
        cudaMalloc(&d_logits, vocab_size * sizeof(float));
    }
    
    void TearDown() override {
        cudaFree(d_logits);
    }
    
    int vocab_size;
    float* d_logits = nullptr;
};

TEST_F(TemperatureScaleTest, TemperatureOne) {
    // Temperature = 1.0 should not change logits
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale(d_logits, vocab_size, 1.0f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], h_logits[i], 0.001f);
    }
}

TEST_F(TemperatureScaleTest, TemperatureHalf) {
    // Temperature = 0.5 should double logits
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = 1.0f;
    }
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale(d_logits, vocab_size, 0.5f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], 2.0f, 0.001f);
    }
}

TEST_F(TemperatureScaleTest, TemperatureTwo) {
    // Temperature = 2.0 should halve logits
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = 4.0f;
    }
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale(d_logits, vocab_size, 2.0f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], 2.0f, 0.001f);
    }
}

TEST_F(TemperatureScaleTest, TemperatureZero) {
    // Temperature = 0.0 should not change logits (greedy mode)
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale(d_logits, vocab_size, 0.0f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], h_logits[i], 0.001f);
    }
}

TEST_F(TemperatureScaleTest, NegativeLogits) {
    // Test with negative logits
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = -2.0f;
    }
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale(d_logits, vocab_size, 0.5f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(vocab_size);
    cudaMemcpy(h_result.data(), d_logits, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_result[i], -4.0f, 0.001f);
    }
}

TEST_F(TemperatureScaleTest, LargeVocabulary) {
    // Test with realistic vocabulary size (e.g., Qwen 151936)
    int large_vocab = 151936;
    float* d_large_logits;
    cudaMalloc(&d_large_logits, large_vocab * sizeof(float));
    
    std::vector<float> h_logits(large_vocab, 1.0f);
    cudaMemcpy(d_large_logits, h_logits.data(), large_vocab * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    launch_temperature_scale(d_large_logits, large_vocab, 0.7f);
    cudaDeviceSynchronize();
    
    std::vector<float> h_result(large_vocab);
    cudaMemcpy(h_result.data(), d_large_logits, large_vocab * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    float expected = 1.0f / 0.7f;
    for (int i = 0; i < large_vocab; ++i) {
        EXPECT_NEAR(h_result[i], expected, 0.01f);
    }
    
    cudaFree(d_large_logits);
}
```

### Implementation Notes
- Simple element-wise operation (memory-bound, not compute-bound)
- Temperature = 0.0 is special case for greedy sampling (no scaling)
- Defensive validation of temperature range (0.0-2.0)
- FP16 version converts to FP32 for division, then back to FP16
- Kernel launch with 256 threads per block (optimal for memory operations)
- No shared memory needed (simple element-wise operation)

---

## Testing Strategy

### Unit Tests
- Test temperature = 1.0 (no change)
- Test temperature = 0.5 (doubles logits)
- Test temperature = 2.0 (halves logits)
- Test temperature = 0.0 (greedy mode, no change)
- Test negative logits
- Test large vocabulary (151936 tokens)

### Integration Tests
- Test with sampling pipeline (temperature → softmax → sample)
- Test determinism with temperature = 0.0

### Manual Verification
1. Run unit tests: `./build/tests/sampling_test`
2. Verify all tests pass
3. Profile memory bandwidth: `nvprof --metrics gld_throughput ./build/tests/sampling_test`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing (2+ tests)
- [ ] Documentation updated (kernel docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §3.2 Temperature Scaling (M0-W-1032)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.3 Token Sampling (M0-W-1421, KERNEL-SAMPLE-003)
- Related Stories: FT-018 (greedy sampling), FT-019 (stochastic sampling)

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
