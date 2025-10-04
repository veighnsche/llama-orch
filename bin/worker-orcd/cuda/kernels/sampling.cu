/**
 * Sampling Kernels
 * 
 * Implements token sampling operations:
 * - Temperature scaling (controls randomness)
 * - Greedy sampling (argmax)
 * - Top-k sampling - TODO
 * 
 * Spec: M0-W-1032, M0-W-1421, KERNEL-SAMPLE-003
 * Story: FT-017, FT-018
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

/**
 * Greedy sampling kernel: find argmax of logits.
 * 
 * Uses parallel reduction within a single block for efficiency.
 * Each thread finds the max in its stride, then reduces in shared memory.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param token_id Output parameter for selected token ID
 */
__global__ void greedy_sample_reduce(
    const float* logits,
    int vocab_size,
    float* block_max,
    int* block_idx
) {
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
    
    // Write block result
    if (tid == 0) {
        block_max[blockIdx.x] = shared_max[0];
        block_idx[blockIdx.x] = shared_idx[0];
    }
}

/**
 * Final reduction kernel: find global max across block results.
 * 
 * Runs on a single thread to find the maximum across all block results.
 * 
 * @param block_max Device pointer to block max values [num_blocks]
 * @param block_idx Device pointer to block max indices [num_blocks]
 * @param num_blocks Number of blocks
 * @param token_id Output parameter for selected token ID
 */
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

/**
 * Launch greedy sampling kernel.
 * 
 * Two-phase reduction:
 * 1. Parallel reduction within blocks
 * 2. Final reduction across block results
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream (default: 0)
 * @return Selected token ID
 */
int launch_greedy_sample(
    const float* logits,
    int vocab_size,
    cudaStream_t stream
) {
    // Validate inputs
    if (vocab_size <= 0) {
        fprintf(stderr, "Invalid vocab_size: %d\n", vocab_size);
        return -1;
    }
    
    if (logits == nullptr) {
        fprintf(stderr, "Null logits pointer\n");
        return -1;
    }
    
    // Kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    // Cap at 256 blocks for efficiency
    if (num_blocks > 256) {
        num_blocks = 256;
    }
    
    // Allocate temporary storage for block results
    float* d_block_max;
    int* d_block_idx;
    int* d_token_id;
    
    cudaMalloc(&d_block_max, num_blocks * sizeof(float));
    cudaMalloc(&d_block_idx, num_blocks * sizeof(int));
    cudaMalloc(&d_token_id, sizeof(int));
    
    // Phase 1: Reduce within blocks
    greedy_sample_reduce<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, vocab_size, d_block_max, d_block_idx
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Greedy sample reduce kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_block_max);
        cudaFree(d_block_idx);
        cudaFree(d_token_id);
        return -1;
    }
    
    // Phase 2: Final reduction
    greedy_sample_final<<<1, 1, 0, stream>>>(
        d_block_max, d_block_idx, num_blocks, d_token_id
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Greedy sample final kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_block_max);
        cudaFree(d_block_idx);
        cudaFree(d_token_id);
        return -1;
    }
    
    // Copy result back to host
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

// ---
// Built by Foundation-Alpha 🏗️

