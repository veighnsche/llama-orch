/**
 * Sampling Kernels Header
 * 
 * Public interface for token sampling operations.
 * 
 * Spec: M0-W-1032, M0-W-1421, KERNEL-SAMPLE-003
 * Story: FT-017
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

} // namespace kernels
} // namespace worker

#endif // WORKER_SAMPLING_CUH

// ---
// Built by Foundation-Alpha 🏗️
