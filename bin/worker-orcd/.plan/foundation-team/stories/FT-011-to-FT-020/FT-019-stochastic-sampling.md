# FT-019: Stochastic Sampling

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: M (2 days)  
**Days**: 34 - 35  
**Spec Ref**: M0-W-1421, M0-W-1032

---

## Story Description

Implement stochastic sampling from probability distribution for production inference with temperature>0. This enables creative, varied text generation.

---

## Acceptance Criteria

- [ ] Softmax kernel converts logits to probabilities
- [ ] Sampling kernel selects token from probability distribution
- [ ] Uses provided seed for reproducibility
- [ ] Handles temperature range 0.1-2.0
- [ ] Unit tests validate sampling distribution
- [ ] Integration tests validate with temperature scaling
- [ ] Kernel optimized for numerical stability (log-sum-exp trick)
- [ ] Support for FP16 and FP32 logits

---

## Dependencies

### Upstream (Blocks This Story)
- FT-017: Temperature scaling (Expected completion: Day 32)
- FT-018: Greedy sampling (Expected completion: Day 33)

### Downstream (This Story Blocks)
- FT-020: Seeded RNG needs stochastic sampling
- Production inference needs stochastic sampling

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/sampling.cu` - Add stochastic sampling kernels
- `bin/worker-orcd/cuda/kernels/sampling.cuh` - Add stochastic sampling interface
- `bin/worker-orcd/cuda/tests/sampling_test.cu` - Add stochastic sampling tests

### Key Interfaces
```cuda
// sampling.cuh (additions)
namespace worker {
namespace kernels {

/**
 * Softmax kernel with numerical stability (log-sum-exp trick).
 * 
 * Converts logits to probabilities: probs[i] = exp(logits[i]) / sum(exp(logits))
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param probs Device pointer to output probabilities [vocab_size]
 * @param vocab_size Vocabulary size
 */
__global__ void softmax_fp32(
    const float* logits,
    float* probs,
    int vocab_size
);

/**
 * Sample token from probability distribution.
 * 
 * Uses cumulative distribution function (CDF) and random number.
 * 
 * @param probs Device pointer to probabilities [vocab_size]
 * @param vocab_size Vocabulary size
 * @param random_value Random value in [0, 1)
 * @param token_id Output parameter for selected token ID
 */
__global__ void sample_from_distribution(
    const float* probs,
    int vocab_size,
    float random_value,
    int* token_id
);

/**
 * Launch stochastic sampling pipeline.
 * 
 * @param logits Device pointer to logits
 * @param vocab_size Vocabulary size
 * @param random_value Random value from RNG
 * @return Selected token ID
 */
int launch_stochastic_sample(
    const float* logits,
    int vocab_size,
    float random_value,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace worker

// sampling.cu (additions)
namespace worker {
namespace kernels {

// Softmax with numerical stability
__global__ void softmax_fp32(
    const float* logits,
    float* probs,
    int vocab_size
) {
    // Shared memory for reduction
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Find max (for numerical stability)
    float local_max = (idx < vocab_size) ? logits[idx] : -INFINITY;
    
    // Reduce to find global max
    __shared__ float block_max[256];
    block_max[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            block_max[tid] = fmaxf(block_max[tid], block_max[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        shared_max = block_max[0];
    }
    __syncthreads();
    
    // Phase 2: Compute exp(logit - max) and sum
    float local_exp = 0.0f;
    if (idx < vocab_size) {
        local_exp = expf(logits[idx] - shared_max);
    }
    
    __shared__ float block_sum[256];
    block_sum[tid] = local_exp;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            block_sum[tid] += block_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        shared_sum = block_sum[0];
    }
    __syncthreads();
    
    // Phase 3: Normalize
    if (idx < vocab_size) {
        probs[idx] = local_exp / shared_sum;
    }
}

// Sample from CDF using binary search
__global__ void sample_from_distribution(
    const float* probs,
    int vocab_size,
    float random_value,
    int* token_id
) {
    // Single thread performs sampling
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    
    // Build CDF and sample
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (random_value < cumsum) {
            *token_id = i;
            return;
        }
    }
    
    // Fallback (should not reach here if probs sum to 1.0)
    *token_id = vocab_size - 1;
}

// Optimized version with parallel prefix sum
__global__ void compute_cdf(
    const float* probs,
    float* cdf,
    int vocab_size
) {
    // Parallel prefix sum (scan)
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    __shared__ float shared_probs[512];
    if (idx < vocab_size) {
        shared_probs[tid] = probs[idx];
    } else {
        shared_probs[tid] = 0.0f;
    }
    __syncthreads();
    
    // Up-sweep (reduce)
    for (int d = 1; d < blockDim.x; d *= 2) {
        int mask = (d * 2) - 1;
        if ((tid & mask) == mask) {
            shared_probs[tid] += shared_probs[tid - d];
        }
        __syncthreads();
    }
    
    // Down-sweep (scan)
    if (tid == blockDim.x - 1) {
        shared_probs[tid] = 0.0f;
    }
    __syncthreads();
    
    for (int d = blockDim.x / 2; d > 0; d /= 2) {
        int mask = (d * 2) - 1;
        if ((tid & mask) == mask) {
            float temp = shared_probs[tid - d];
            shared_probs[tid - d] = shared_probs[tid];
            shared_probs[tid] += temp;
        }
        __syncthreads();
    }
    
    // Write CDF
    if (idx < vocab_size) {
        cdf[idx] = shared_probs[tid] + probs[idx];
    }
}

__global__ void sample_from_cdf(
    const float* cdf,
    int vocab_size,
    float random_value,
    int* token_id
) {
    // Binary search in CDF
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    
    int left = 0;
    int right = vocab_size - 1;
    
    while (left < right) {
        int mid = (left + right) / 2;
        if (cdf[mid] < random_value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    *token_id = left;
}

int launch_stochastic_sample(
    const float* logits,
    int vocab_size,
    float random_value,
    cudaStream_t stream
) {
    // Allocate temporary storage
    float* d_probs;
    float* d_cdf;
    int* d_token_id;
    
    cudaMalloc(&d_probs, vocab_size * sizeof(float));
    cudaMalloc(&d_cdf, vocab_size * sizeof(float));
    cudaMalloc(&d_token_id, sizeof(int));
    
    // Phase 1: Softmax
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    
    softmax_fp32<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, d_probs, vocab_size
    );
    
    // Phase 2: Compute CDF
    compute_cdf<<<num_blocks, threads_per_block, 0, stream>>>(
        d_probs, d_cdf, vocab_size
    );
    
    // Phase 3: Sample from CDF
    sample_from_cdf<<<1, 1, 0, stream>>>(
        d_cdf, vocab_size, random_value, d_token_id
    );
    
    // Copy result back
    int h_token_id;
    cudaMemcpyAsync(&h_token_id, d_token_id, sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Free temporary storage
    cudaFree(d_probs);
    cudaFree(d_cdf);
    cudaFree(d_token_id);
    
    return h_token_id;
}

} // namespace kernels
} // namespace worker

// Unit tests (additions)
TEST(StochasticSamplingTest, SoftmaxNormalization) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 1.0f);
    std::vector<float> h_probs(vocab_size);
    
    float *d_logits, *d_probs;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_probs, vocab_size * sizeof(float));
    
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    softmax_fp32<<<4, 256>>>(d_logits, d_probs, vocab_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_probs.data(), d_probs, vocab_size * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Sum of probabilities should be 1.0
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        sum += h_probs[i];
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // Each probability should be 1/vocab_size
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_NEAR(h_probs[i], 1.0f / vocab_size, 0.0001f);
    }
    
    cudaFree(d_logits);
    cudaFree(d_probs);
}

TEST(StochasticSamplingTest, SamplingDistribution) {
    // Test that sampling follows distribution
    int vocab_size = 10;
    std::vector<float> h_logits = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f
    };
    
    float *d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Sample many times with different random values
    std::vector<int> counts(vocab_size, 0);
    int num_samples = 1000;
    
    for (int i = 0; i < num_samples; ++i) {
        float random_value = static_cast<float>(i) / num_samples;
        int token_id = launch_stochastic_sample(d_logits, vocab_size, random_value);
        counts[token_id]++;
    }
    
    // Higher logits should be sampled more often
    // (This is a weak test, just checking general trend)
    EXPECT_GT(counts[9], counts[0]);
    
    cudaFree(d_logits);
}

TEST(StochasticSamplingTest, DeterministicWithSeed) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i % 100);
    }
    
    float *d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Same random value should give same result
    float random_value = 0.5f;
    int token_id1 = launch_stochastic_sample(d_logits, vocab_size, random_value);
    int token_id2 = launch_stochastic_sample(d_logits, vocab_size, random_value);
    
    EXPECT_EQ(token_id1, token_id2);
    
    cudaFree(d_logits);
}
```

### Implementation Notes
- Softmax uses log-sum-exp trick for numerical stability
- CDF computed via parallel prefix sum (scan)
- Binary search in CDF for efficient sampling
- Deterministic given same random value
- Handles large vocabularies efficiently
- FP32 for numerical precision in softmax

---

## Testing Strategy

### Unit Tests
- Test softmax normalization (sum = 1.0)
- Test sampling distribution (higher logits → more samples)
- Test determinism with same random value
- Test numerical stability with large logits

### Integration Tests
- Test with temperature scaling pipeline
- Test with seeded RNG for reproducibility

### Manual Verification
1. Run unit tests: `./build/tests/sampling_test`
2. Verify softmax numerical stability
3. Profile prefix sum efficiency

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (4+ tests)
- [ ] Integration tests passing (2+ tests)
- [ ] Documentation updated (kernel docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.3 Token Sampling (M0-W-1421)
- Related Stories: FT-017 (temperature), FT-020 (seeded RNG)
- Parallel Prefix Sum: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
