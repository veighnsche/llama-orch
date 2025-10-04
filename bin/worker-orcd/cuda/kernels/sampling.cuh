/**
 * Sampling Kernels Header
 * 
 * Public interface for token sampling operations.
 * 
 * Spec: M0-W-1032, M0-W-1421, KERNEL-SAMPLE-003
 * Story: FT-017, FT-018, FT-019
 */

#ifndef WORKER_SAMPLING_CUH
#define WORKER_SAMPLING_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace kernels {

/**
 * Apply temperature scaling to logits (FP32).
 * 
 * Temperature controls randomness in sampling:
 * - temperature = 0.0: Greedy sampling (no scaling)
 * - temperature < 1.0: More deterministic (sharper distribution)
 * - temperature = 1.0: No change (identity)
 * - temperature > 1.0: More random (flatter distribution)
 * 
 * Formula: logits[i] /= temperature (for temperature > 0)
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param temperature Sampling temperature (0.0-2.0)
 */
__global__ void temperature_scale_fp32(
    float* logits,
    int vocab_size,
    float temperature
);

/**
 * Apply temperature scaling to logits (FP16).
 * 
 * Same as FP32 version but with half precision.
 * Converts to FP32 for division, then back to FP16.
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param temperature Sampling temperature (0.0-2.0)
 */
__global__ void temperature_scale_fp16(
    half* logits,
    int vocab_size,
    float temperature
);

/**
 * Launch temperature scaling kernel (FP32).
 * 
 * Configures grid/block and launches kernel.
 * 
 * Grid configuration:
 * - Grid: ceil(vocab_size / 256)
 * - Block: 256 threads
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size (must be > 0)
 * @param temperature Sampling temperature (0.0-2.0)
 * @param stream CUDA stream (default: 0)
 * 
 * Error handling:
 * - Validates vocab_size > 0
 * - Validates temperature >= 0.0 and <= 2.0
 * - Checks kernel launch errors
 * - temperature = 0.0 is valid (greedy mode, no scaling)
 */
void launch_temperature_scale_fp32(
    float* logits,
    int vocab_size,
    float temperature,
    cudaStream_t stream = 0
);

/**
 * Launch temperature scaling kernel (FP16).
 * 
 * Same as FP32 version but with half precision.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size (must be > 0)
 * @param temperature Sampling temperature (0.0-2.0)
 * @param stream CUDA stream (default: 0)
 */
void launch_temperature_scale_fp16(
    half* logits,
    int vocab_size,
    float temperature,
    cudaStream_t stream = 0
);

/**
 * Greedy sampling: find argmax of logits.
 * 
 * Uses two-phase parallel reduction for efficiency:
 * 1. Parallel reduction within blocks
 * 2. Final reduction across block results
 * 
 * This is deterministic: same logits always produce same token ID.
 * Used for temperature=0.0 (greedy mode) in testing.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size (must be > 0)
 * @param stream CUDA stream (default: 0)
 * @return Selected token ID (index of max logit), or -1 on error
 * 
 * Error handling:
 * - Returns -1 if vocab_size <= 0
 * - Returns -1 if logits is nullptr
 * - Returns -1 on kernel launch failure
 * 
 * Performance:
 * - Handles large vocabularies efficiently (e.g., 151936 tokens)
 * - Uses 256 threads per block, up to 256 blocks
 * - Shared memory reduction for optimal performance
 */
int launch_greedy_sample(
    const float* logits,
    int vocab_size,
    cudaStream_t stream = 0
);

/**
 * Softmax kernel with numerical stability (log-sum-exp trick).
 * 
 * Converts logits to probabilities: probs[i] = exp(logits[i]) / sum(exp(logits))
 * Uses max subtraction to prevent overflow: exp(x - max) / sum(exp(x - max))
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
 * Sample token from probability distribution using CDF.
 * 
 * Uses linear scan through CDF to find token where cumsum >= random_value.
 * Single-threaded for simplicity (sampling is fast compared to softmax).
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
 * Stochastic sampling: sample token from probability distribution.
 * 
 * Two-phase pipeline:
 * 1. Softmax: Convert logits to probabilities (with numerical stability)
 * 2. Sample: Select token from probability distribution using CDF
 * 
 * This is stochastic: different random values produce different tokens.
 * Used for temperature > 0.0 (creative generation).
 * 
 * Deterministic given same random_value (for reproducibility with seeded RNG).
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size (must be > 0)
 * @param random_value Random value from RNG [0, 1)
 * @param stream CUDA stream (default: 0)
 * @return Selected token ID (index of sampled token), or -1 on error
 * 
 * Error handling:
 * - Returns -1 if vocab_size <= 0
 * - Returns -1 if logits is nullptr
 * - Returns -1 if random_value not in [0, 1)
 * - Returns -1 on kernel launch failure
 * 
 * Performance:
 * - Handles large vocabularies efficiently (e.g., 151936 tokens)
 * - Uses log-sum-exp trick for numerical stability
 * - Softmax: up to 256 blocks × 256 threads
 * - Sampling: single-threaded (fast compared to softmax)
 */
int launch_stochastic_sample(
    const float* logits,
    int vocab_size,
    float random_value,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace worker

#endif // WORKER_SAMPLING_CUH

// ---
// Built by Foundation-Alpha 🏗️
