# FT-019: Stochastic Sampling (Extended with Advanced Parameters)

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: L (3 days) — **EXPANDED from M (2 days)**  
**Days**: 34 - 36  
**Spec Ref**: M0-W-1421, M0-W-1032, GENERATION_PARAMETERS_ANALYSIS.md

---

## Story Description

Implement stochastic sampling from probability distribution for production inference with temperature>0. This enables creative, varied text generation.

**EXPANDED SCOPE**: This story now includes advanced sampling parameters (top-p, top-k, repetition_penalty, stop sequences) to achieve competitive parity with OpenAI/llama.cpp/LM Studio. These parameters are critical for user expectations and structured output generation.

**Rationale**: Analysis shows M0 is missing 9+ standard generation parameters. Adding them now (in Sprint 3) is more efficient than deferring to M1, as the sampling infrastructure is already being built.

---

## Acceptance Criteria

### Core Sampling (Original)
- [ ] Softmax kernel converts logits to probabilities
- [ ] Sampling kernel selects token from probability distribution
- [ ] Uses provided seed for reproducibility
- [ ] Handles temperature range 0.1-2.0
- [ ] Unit tests validate sampling distribution
- [ ] Integration tests validate with temperature scaling
- [ ] Kernel optimized for numerical stability (log-sum-exp trick)
- [ ] Support for FP16 and FP32 logits

### Advanced Parameters (NEW)
- [ ] **Top-P (nucleus sampling)**: Filter tokens by cumulative probability
- [ ] **Top-K sampling**: Keep only top K tokens by probability
- [ ] **Repetition penalty**: Penalize already-generated tokens
- [ ] **Stop sequences**: Terminate generation on match (up to 4 sequences)
- [ ] **Min-P sampling**: Minimum probability threshold (optional, low priority)
- [ ] HTTP API extended with new parameters (optional, default values)
- [ ] Validation for all new parameters
- [ ] Unit tests for each sampling strategy
- [ ] Integration tests with combined parameters (e.g., temp + top-p + top-k)

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
 * Sampling configuration with advanced parameters.
 */
struct SamplingConfig {
    float temperature = 1.0f;
    float top_p = 1.0f;           // Nucleus sampling (0.0-1.0, 1.0=disabled)
    int top_k = 0;                // Top-k sampling (0=disabled)
    float repetition_penalty = 1.0f;  // Repetition penalty (1.0=disabled)
    float min_p = 0.0f;           // Min-p sampling (0.0=disabled)
    
    // Stop sequences (tokenized)
    const int* stop_sequences[4] = {nullptr, nullptr, nullptr, nullptr};
    int stop_sequence_lengths[4] = {0, 0, 0, 0};
};

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
 * Apply top-k filtering to logits (zero out tokens outside top k).
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param top_k Keep only top k tokens
 */
__global__ void apply_top_k(
    float* logits,
    int vocab_size,
    int top_k
);

/**
 * Apply top-p (nucleus) filtering to logits.
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param top_p Cumulative probability threshold
 */
__global__ void apply_top_p(
    float* logits,
    int vocab_size,
    float top_p
);

/**
 * Apply repetition penalty to logits.
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param history Device pointer to generated token IDs
 * @param history_length Number of tokens in history
 * @param penalty Repetition penalty factor (>1.0 = penalize, <1.0 = encourage)
 */
__global__ void apply_repetition_penalty(
    float* logits,
    int vocab_size,
    const int* history,
    int history_length,
    float penalty
);

/**
 * Apply min-p filtering to logits.
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param min_p Minimum probability threshold
 */
__global__ void apply_min_p(
    float* logits,
    int vocab_size,
    float min_p
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
 * Check if generated sequence matches any stop sequence.
 * 
 * @param generated_tokens Device pointer to generated token IDs
 * @param num_generated Number of generated tokens
 * @param config Sampling configuration with stop sequences
 * @return True if any stop sequence matched
 */
bool check_stop_sequences(
    const int* generated_tokens,
    int num_generated,
    const SamplingConfig& config
);

/**
 * Launch advanced stochastic sampling pipeline with all parameters.
 * 
 * @param logits Device pointer to logits
 * @param vocab_size Vocabulary size
 * @param config Sampling configuration
 * @param history Device pointer to generated token history (for repetition penalty)
 * @param history_length Number of tokens in history
 * @param random_value Random value from RNG
 * @return Selected token ID
 */
int launch_advanced_sample(
    const float* logits,
    int vocab_size,
    const SamplingConfig& config,
    const int* history,
    int history_length,
    float random_value,
    cudaStream_t stream = 0
);

/**
 * Launch stochastic sampling pipeline (basic version, backward compatible).
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

#### Core Sampling
- Softmax uses log-sum-exp trick for numerical stability
- CDF computed via parallel prefix sum (scan)
- Binary search in CDF for efficient sampling
- Deterministic given same random value
- Handles large vocabularies efficiently
- FP32 for numerical precision in softmax

#### Advanced Parameters Implementation

**Top-K Sampling**:
```cpp
__global__ void apply_top_k(float* logits, int vocab_size, int top_k) {
    // Partial sort to find top k
    // Zero out logits outside top k
    // Implementation: Use thrust::sort or custom parallel partial sort
}
```

**Top-P (Nucleus) Sampling**:
```cpp
__global__ void apply_top_p(float* logits, int vocab_size, float top_p) {
    // 1. Sort logits descending
    // 2. Compute cumulative softmax
    // 3. Find cutoff where cumsum >= top_p
    // 4. Zero out logits below cutoff
}
```

**Repetition Penalty**:
```cpp
__global__ void apply_repetition_penalty(
    float* logits, int vocab_size,
    const int* history, int history_length,
    float penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;
    
    // Check if token is in history
    for (int i = 0; i < history_length; ++i) {
        if (history[i] == idx) {
            // Apply penalty
            if (logits[idx] > 0) {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
            break;
        }
    }
}
```

**Stop Sequences**:
```cpp
bool check_stop_sequences(
    const int* generated_tokens,
    int num_generated,
    const SamplingConfig& config
) {
    for (int seq_idx = 0; seq_idx < 4; ++seq_idx) {
        if (config.stop_sequences[seq_idx] == nullptr) continue;
        
        int seq_len = config.stop_sequence_lengths[seq_idx];
        if (num_generated < seq_len) continue;
        
        // Check if last seq_len tokens match stop sequence
        bool match = true;
        for (int i = 0; i < seq_len; ++i) {
            int gen_idx = num_generated - seq_len + i;
            if (generated_tokens[gen_idx] != config.stop_sequences[seq_idx][i]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}
```

**Advanced Sampling Pipeline**:
```cpp
int launch_advanced_sample(
    const float* logits,
    int vocab_size,
    const SamplingConfig& config,
    const int* history,
    int history_length,
    float random_value,
    cudaStream_t stream
) {
    // Copy logits to temporary buffer (we'll modify them)
    float* d_logits_temp;
    cudaMalloc(&d_logits_temp, vocab_size * sizeof(float));
    cudaMemcpyAsync(d_logits_temp, logits, vocab_size * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Apply filters in order
    if (config.repetition_penalty != 1.0f && history != nullptr) {
        apply_repetition_penalty<<<grid, block, 0, stream>>>(
            d_logits_temp, vocab_size, history, history_length,
            config.repetition_penalty
        );
    }
    
    if (config.top_k > 0) {
        apply_top_k<<<grid, block, 0, stream>>>(
            d_logits_temp, vocab_size, config.top_k
        );
    }
    
    if (config.top_p < 1.0f) {
        apply_top_p<<<grid, block, 0, stream>>>(
            d_logits_temp, vocab_size, config.top_p
        );
    }
    
    if (config.min_p > 0.0f) {
        apply_min_p<<<grid, block, 0, stream>>>(
            d_logits_temp, vocab_size, config.min_p
        );
    }
    
    // Sample from filtered distribution
    int token_id = launch_stochastic_sample(
        d_logits_temp, vocab_size, random_value, stream
    );
    
    cudaFree(d_logits_temp);
    return token_id;
}
```

#### HTTP API Extension

**Extended Request Schema**:
```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku about GPU computing",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "stop": ["\n\n", "END"],
  "seed": 42
}
```

**Validation Rules**:
- `top_p`: 0.0-1.0, default 1.0 (disabled)
- `top_k`: 0-vocab_size, default 0 (disabled)
- `repetition_penalty`: 0.0-2.0, default 1.0 (disabled)
- `stop`: array of strings, max 4 sequences, each max 32 tokens
- `min_p`: 0.0-1.0, default 0.0 (disabled)

**Backward Compatibility**: All new parameters are optional with defaults that disable them. Existing M0 requests (temperature only) continue to work.

---

## Testing Strategy

### Unit Tests (Core Sampling)
- Test softmax normalization (sum = 1.0)
- Test sampling distribution (higher logits → more samples)
- Test determinism with same random value
- Test numerical stability with large logits

### Unit Tests (Advanced Parameters)
- **Top-K**: Test that only top k tokens are kept
- **Top-P**: Test nucleus filtering (cumsum cutoff)
- **Repetition Penalty**: Test penalty applied to history tokens
- **Stop Sequences**: Test early termination on match
- **Min-P**: Test minimum probability threshold
- **Combined**: Test top-k + top-p + repetition_penalty together

### Integration Tests
- Test with temperature scaling pipeline
- Test with seeded RNG for reproducibility
- Test HTTP API with all new parameters
- Test backward compatibility (old requests still work)
- Test parameter validation (reject invalid values)

### Manual Verification
1. Run unit tests: `./build/tests/sampling_test`
2. Verify softmax numerical stability
3. Profile prefix sum efficiency
4. Test HTTP API with curl:
   ```bash
   curl -X POST http://localhost:8080/execute \
     -H "Content-Type: application/json" \
     -d '{
       "job_id": "test-1",
       "prompt": "Write a haiku",
       "temperature": 0.7,
       "top_p": 0.9,
       "top_k": 50,
       "repetition_penalty": 1.1,
       "stop": ["\n\n"],
       "seed": 42
     }'
   ```

---

## Definition of Done

### Core Requirements
- [ ] All acceptance criteria met (core + advanced)
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (10+ tests: 4 core + 6 advanced)
- [ ] Integration tests passing (5+ tests)
- [ ] Documentation updated (kernel docs + HTTP API docs)
- [ ] Story marked complete in day-tracker.md

### Advanced Parameters Checklist
- [ ] Top-K sampling implemented and tested
- [ ] Top-P (nucleus) sampling implemented and tested
- [ ] Repetition penalty implemented and tested
- [ ] Stop sequences implemented and tested
- [ ] Min-P sampling implemented and tested (optional, low priority)
- [ ] HTTP API extended with new parameters
- [ ] Parameter validation implemented
- [ ] Backward compatibility verified (old requests work)
- [ ] All new parameters have sensible defaults

---

## References

- **Primary Spec**: `bin/.specs/01_M0_worker_orcd.md` §9.3 Token Sampling (M0-W-1421)
- **Analysis Document**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md` (comprehensive gap analysis)
- **Related Stories**: FT-017 (temperature), FT-020 (seeded RNG)
- **Technical References**:
  - Parallel Prefix Sum: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
  - Nucleus Sampling Paper: Holtzman et al. 2019 "The Curious Case of Neural Text Degeneration"
  - Top-K Sampling: Fan et al. 2018 "Hierarchical Neural Story Generation"

---

## Summary: Why This Expansion Matters

**Problem**: M0 was missing 9+ standard generation parameters that users expect from LLM APIs (OpenAI, Anthropic, llama.cpp, LM Studio).

**Solution**: Expand FT-019 to include advanced sampling parameters (top-p, top-k, repetition_penalty, stop sequences) in Sprint 3.

**Benefits**:
1. **Competitive Parity**: Match industry standards (OpenAI: 10 params, llama.cpp: 12 params, M0: now 8 params)
2. **User Expectations**: Users coming from other APIs expect these parameters
3. **Structured Output**: Stop sequences are critical for JSON, code, and structured generation
4. **Efficiency**: Adding now (during sampling implementation) is more efficient than deferring to M1
5. **Backward Compatible**: All new parameters are optional with sensible defaults

**Trade-off**: Story expanded from M (2 days) to L (3 days), but eliminates need for separate M1 story.

**Comparison**:
| Feature | M0 (before) | M0 (after) | OpenAI | llama.cpp |
|---------|-------------|------------|--------|-----------|
| Parameters | 3 | 8 | 10 | 12 |
| Top-P | ❌ | ✅ | ✅ | ✅ |
| Top-K | ❌ | ✅ | ❌ | ✅ |
| Repetition Penalty | ❌ | ✅ | ❌ | ✅ |
| Stop Sequences | ❌ | ✅ | ✅ | ✅ |

---

## 🔍 Testing Requirements

**Added by**: Testing Team (test-harness/TEAM_RESPONSIBILITIES.md)

### Unit Tests (MUST implement)

**Critical Path Coverage**:
- **Test softmax normalization (sum = 1.0)** (M0-W-1421)
  - Given: Uniform logits [1.0, 1.0, ..., 1.0]
  - When: softmax_fp32(logits, probs, vocab_size)
  - Then: sum(probs) = 1.0, each prob = 1/vocab_size
  - **Why critical**: Softmax must normalize correctly

- **Test sampling distribution (higher logits → more samples)**
  - Given: logits [0, 1, 2, ..., 9]
  - When: Sample 1000 times with uniform random values
  - Then: Token 9 sampled more than token 0
  - **Why critical**: Sampling must follow probability distribution

- **Test determinism with same random value** (M0-W-1030)
  - Given: Same logits, same random_value
  - When: Sampled twice
  - Then: Identical token_id both times
  - **Why critical**: Reproducibility with seeded RNG

- **Test numerical stability with large logits**
  - Given: logits with values > 100
  - When: Softmax computed
  - Then: No overflow, sum(probs) = 1.0
  - **Why critical**: Log-sum-exp trick must work

### Integration Tests (MUST implement)

- **Test with temperature scaling pipeline** (M0-W-1032)
  - Given: Temperature scaling → softmax → sample
  - When: Pipeline executes with temp=0.7
  - Then: Sampling distribution reflects temperature
  - **Why critical**: End-to-end validation

- **Test with seeded RNG for reproducibility** (M0-W-1030)
  - Given: Same seed, same logits
  - When: Sampled twice
  - Then: Identical token sequences
  - **Why critical**: Reproducibility requirement

### BDD Scenarios (VERY IMPORTANT - MUST implement)

**Feature**: Stochastic Sampling

```gherkin
Scenario: Worker performs stochastic sampling for creative generation
  Given a worker with temperature = 0.7
  When the worker samples from logits
  Then tokens are sampled from probability distribution
  And higher probability tokens are sampled more often
  And output is varied (not deterministic)

Scenario: Worker ensures reproducibility with same seed
  Given a worker with seed = 42
  And same model and prompt
  When inference is run twice with same seed
  Then both runs produce identical token sequences
  And stochastic sampling is used

Scenario: Worker handles numerical stability in softmax
  Given a worker with very large logit values (>100)
  When softmax is computed
  Then no overflow occurs
  And probabilities sum to 1.0
  And sampling works correctly
```

### Test Artifacts (MUST produce)

- **Unit test report**: Pass/fail for each test
- **Sampling distribution analysis**: Histogram of sampled tokens
- **Determinism proof**: Same seed → same outputs
- **BDD scenario results**: Pass/fail with sampling traces

### Acceptance Criteria for Testing

- ✅ All unit tests pass (4+ tests covering critical paths)
- ✅ All integration tests pass (2+ tests with temperature pipeline)
- ✅ All BDD scenarios pass (3 scenarios validating stochastic behavior)
- ✅ Numerical stability verified (no overflow with large logits)
- ✅ All tests produce verifiable artifacts

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

Hey Foundation Team! 👋 We're here to help you make stochastic sampling **delightfully debuggable**!

### Quick Start (v0.2.0 Builder API)

We just shipped v0.2.0 with a **builder pattern** that's 43% less boilerplate:

```rust
use observability_narration_core::{Narration, ACTOR_INFERENCE_ENGINE};

// In your stochastic sampling code:
Narration::new(ACTOR_INFERENCE_ENGINE, "token_sample", "stochastic")
    .human(format!("Sampled token {} (stochastic, seed={}, temp={})", 
                   token_id, seed, temperature))
    .device(format!("GPU{}", device_id))
    .emit();
```

The builder automatically adds `emitted_by`, `emitted_at_ms`, and secret redaction!

### Events to Narrate

1. **Token sampled (stochastic)**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "token_sample",
       target: "stochastic".to_string(),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Sampled token {} (stochastic, seed={}, temp={})", token_id, seed, temperature),
       ..Default::default()
   });
   ```

**Why this matters**: Stochastic sampling uses RNG. Narration helps verify reproducibility with same seed and track sampling behavior.

### Testing Your Narration

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_stochastic_sampling_narrates() {
    let adapter = CaptureAdapter::install();
    
    // Your stochastic sampling
    let token_id = launch_stochastic_sample(logits, vocab_size, random_value);
    
    adapter.assert_includes("Sampled token");
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
