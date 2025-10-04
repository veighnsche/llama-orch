# FT-018: Greedy Sampling

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: S (1 day)  
**Days**: 33 - 33  
**Spec Ref**: M0-W-1032, M0-W-1421

---

## Story Description

Implement greedy sampling (argmax) for deterministic token selection when temperature=0. This is used for testing reproducibility and ensures identical outputs across runs.

---

## Acceptance Criteria

- [ ] CUDA kernel finds argmax of logits: `token_id = argmax(logits)`
- [ ] Handles large vocabulary sizes efficiently (e.g., 151936 tokens)
- [ ] Unit tests validate correctness with known inputs
- [ ] Integration tests validate determinism (same input → same output)
- [ ] Kernel optimized with parallel reduction
- [ ] Error handling for empty logits
- [ ] Support for FP16 and FP32 logits

---

## Dependencies

### Upstream (Blocks This Story)
- FT-017: Temperature scaling (Expected completion: Day 32)

### Downstream (This Story Blocks)
- FT-024: HTTP-FFI-CUDA integration needs greedy sampling for tests
- Reproducibility tests need greedy sampling

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/sampling.cu` - Add greedy sampling kernel
- `bin/worker-orcd/cuda/kernels/sampling.cuh` - Add greedy sampling interface
- `bin/worker-orcd/cuda/tests/sampling_test.cu` - Add greedy sampling tests

### Key Interfaces
```cuda
// sampling.cuh (additions)
namespace worker {
namespace kernels {

/**
 * Greedy sampling: find argmax of logits.
 * 
 * Uses parallel reduction for efficiency.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param token_id Output parameter for selected token ID
 */
__global__ void greedy_sample_fp32(
    const float* logits,
    int vocab_size,
    int* token_id
);

/**
 * Launch greedy sampling kernel.
 * 
 * @return Selected token ID
 */
int launch_greedy_sample(
    const float* logits,
    int vocab_size,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace worker

// sampling.cu (additions)
namespace worker {
namespace kernels {

// Parallel reduction to find argmax
__global__ void greedy_sample_fp32(
    const float* logits,
    int vocab_size,
    int* token_id
) {
    // Shared memory for reduction
    __shared__ float shared_max[256];
    __shared__ int shared_idx[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with first element or -inf
    float local_max = (idx < vocab_size) ? logits[idx] : -INFINITY;
    int local_idx = (idx < vocab_size) ? idx : 0;
    
    // Each thread finds max in its stride
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }
    
    // Store in shared memory
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result from block 0, thread 0
    if (blockIdx.x == 0 && tid == 0) {
        // Find global max across all blocks
        float global_max = shared_max[0];
        int global_idx = shared_idx[0];
        
        // If multiple blocks, need to reduce across blocks
        // For simplicity, we use single block for now
        *token_id = global_idx;
    }
}

// Optimized version for large vocabularies
__global__ void greedy_sample_reduce(
    const float* logits,
    int vocab_size,
    float* block_max,
    int* block_idx,
    int num_blocks
) {
    __shared__ float shared_max[256];
    __shared__ int shared_idx[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = (idx < vocab_size) ? logits[idx] : -INFINITY;
    int local_idx = (idx < vocab_size) ? idx : 0;
    
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }
    
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_max[blockIdx.x] = shared_max[0];
        block_idx[blockIdx.x] = shared_idx[0];
    }
}

__global__ void greedy_sample_final(
    const float* block_max,
    const int* block_idx,
    int num_blocks,
    int* token_id
) {
    float global_max = block_max[0];
    int global_idx = block_idx[0];
    
    for (int i = 1; i < num_blocks; ++i) {
        if (block_max[i] > global_max) {
            global_max = block_max[i];
            global_idx = block_idx[i];
        }
    }
    
    *token_id = global_idx;
}

int launch_greedy_sample(
    const float* logits,
    int vocab_size,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = std::min(256, (vocab_size + threads_per_block - 1) / threads_per_block);
    
    // Allocate temporary storage for block results
    float* d_block_max;
    int* d_block_idx;
    int* d_token_id;
    
    cudaMalloc(&d_block_max, num_blocks * sizeof(float));
    cudaMalloc(&d_block_idx, num_blocks * sizeof(int));
    cudaMalloc(&d_token_id, sizeof(int));
    
    // Phase 1: Reduce within blocks
    greedy_sample_reduce<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, vocab_size, d_block_max, d_block_idx, num_blocks
    );
    
    // Phase 2: Final reduction
    greedy_sample_final<<<1, 1, 0, stream>>>(
        d_block_max, d_block_idx, num_blocks, d_token_id
    );
    
    // Copy result back
    int h_token_id;
    cudaMemcpyAsync(&h_token_id, d_token_id, sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Free temporary storage
    cudaFree(d_block_max);
    cudaFree(d_block_idx);
    cudaFree(d_token_id);
    
    return h_token_id;
}

} // namespace kernels
} // namespace worker

// Unit tests (additions to sampling_test.cu)
TEST(GreedySamplingTest, SimpleArgmax) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set token 500 to highest value
    h_logits[500] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 500);
    
    cudaFree(d_logits);
}

TEST(GreedySamplingTest, FirstToken) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set token 0 to highest value
    h_logits[0] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 0);
    
    cudaFree(d_logits);
}

TEST(GreedySamplingTest, LastToken) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set last token to highest value
    h_logits[vocab_size - 1] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, vocab_size - 1);
    
    cudaFree(d_logits);
}

TEST(GreedySamplingTest, NegativeLogits) {
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size, -10.0f);
    
    // Set token 250 to least negative (highest)
    h_logits[250] = -1.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 250);
    
    cudaFree(d_logits);
}

TEST(GreedySamplingTest, LargeVocabulary) {
    // Test with Qwen vocabulary size
    int vocab_size = 151936;
    std::vector<float> h_logits(vocab_size, 0.0f);
    
    // Set token 100000 to highest value
    h_logits[100000] = 10.0f;
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    int token_id = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id, 100000);
    
    cudaFree(d_logits);
}

TEST(GreedySamplingTest, Determinism) {
    // Test that greedy sampling is deterministic
    int vocab_size = 1000;
    std::vector<float> h_logits(vocab_size);
    
    // Random logits
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i % 100);
    }
    
    float* d_logits;
    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Run multiple times
    int token_id1 = launch_greedy_sample(d_logits, vocab_size);
    int token_id2 = launch_greedy_sample(d_logits, vocab_size);
    int token_id3 = launch_greedy_sample(d_logits, vocab_size);
    
    EXPECT_EQ(token_id1, token_id2);
    EXPECT_EQ(token_id2, token_id3);
    
    cudaFree(d_logits);
}
```

### Implementation Notes
- Two-phase reduction for large vocabularies (>256 elements)
- Phase 1: Parallel reduction within blocks
- Phase 2: Final reduction across block results
- Shared memory used for efficient reduction
- Handles edge cases (first token, last token, negative logits)
- Deterministic (same input always produces same output)
- Optimized for large vocabularies (151936 tokens)

---

## Testing Strategy

### Unit Tests
- Test simple argmax (token in middle)
- Test first token is max
- Test last token is max
- Test negative logits
- Test large vocabulary (151936 tokens)
- Test determinism (multiple runs → same result)

### Integration Tests
- Test with temperature=0 pipeline (scale → greedy sample)
- Test reproducibility across multiple inference runs

### Manual Verification
1. Run unit tests: `./build/tests/sampling_test`
2. Verify all tests pass
3. Profile reduction efficiency: `nvprof --metrics shared_efficiency ./build/tests/sampling_test`

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
- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.3 Token Sampling (M0-W-1421)
- Related Stories: FT-017 (temperature scaling), FT-019 (stochastic sampling)
- CUDA Reduction: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Token sampled (greedy)**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "token_sample",
       target: "greedy".to_string(),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Sampled token {} (greedy, logit={})", token_id, max_logit),
       ..Default::default()
   });
   ```

**Why this matters**: Greedy sampling is deterministic. Narration helps verify determinism and track sampled tokens.

---
*Narration guidance added by Narration-Core Team 🎀*
