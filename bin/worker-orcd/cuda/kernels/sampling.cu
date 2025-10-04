/**
 * Sampling Kernels
 * 
 * Implements token sampling operations:
 * - Temperature scaling (controls randomness)
 * - Greedy sampling (argmax) - TODO
 * - Top-k sampling - TODO
 * 
 * Spec: M0-W-1032, M0-W-1421, KERNEL-SAMPLE-003
 * Story: FT-017
 */

#include "sampling.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

namespace worker {
namespace kernels {

/**
 * Temperature scaling kernel (FP32).
 * 
 * Divides logits by temperature to control sampling randomness.
 * Each thread handles one logit value.
 * 
 * Special cases:
 * - temperature = 0.0: No scaling (greedy mode)
 * - temperature < 0.0 or > 2.0: No scaling (invalid, defensive)
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param temperature Sampling temperature (0.0-2.0)
 */
__global__ void temperature_scale_fp32(
    float* logits,
    int vocab_size,
    float temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx >= vocab_size) {
        return;
    }
    
    // Temperature == 0.0 means greedy sampling (no scaling)
    if (temperature == 0.0f) {
        return;
    }
    
    // Validate temperature range (defensive)
    if (temperature < 0.0f || temperature > 2.0f) {
        // Invalid temperature, skip scaling
        return;
    }
    
    // Apply temperature scaling: logits[i] /= temperature
    logits[idx] /= temperature;
}

/**
 * Temperature scaling kernel (FP16).
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
 * @param vocab_size Vocabulary size
 * @param temperature Sampling temperature (0.0-2.0)
 * @param stream CUDA stream (default: 0)
 */
void launch_temperature_scale_fp32(
    float* logits,
    int vocab_size,
    float temperature,
    cudaStream_t stream
) {
    // Validate inputs
    if (vocab_size <= 0) {
        fprintf(stderr, "Invalid vocab_size: %d\n", vocab_size);
        return;
    }
    
    if (logits == nullptr) {
        fprintf(stderr, "Null logits pointer\n");
        return;
    }
    
    // Kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    temperature_scale_fp32<<<num_blocks, threads_per_block, 0, stream>>>(
        logits,
        vocab_size,
        temperature
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Temperature scale kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

/**
 * Launch temperature scaling kernel (FP16).
 * 
 * Same as FP32 version but with half precision.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param temperature Sampling temperature (0.0-2.0)
 * @param stream CUDA stream (default: 0)
 */
void launch_temperature_scale_fp16(
    half* logits,
    int vocab_size,
    float temperature,
    cudaStream_t stream
) {
    if (vocab_size <= 0) {
        fprintf(stderr, "Invalid vocab_size: %d\n", vocab_size);
        return;
    }
    
    if (logits == nullptr) {
        fprintf(stderr, "Null logits pointer\n");
        return;
    }
    
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    
    temperature_scale_fp16<<<num_blocks, threads_per_block, 0, stream>>>(
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

// ---
// Built by Foundation-Alpha 🏗️

