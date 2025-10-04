/**
 * Sampling Kernels
 * 
 * Implements token sampling operations:
 * - Temperature scaling (controls randomness)
 * - Greedy sampling (argmax)
 * - Stochastic sampling (probability distribution)
 * - Top-k sampling (keep top k tokens)
 * - Top-p sampling (nucleus sampling)
 * 
 * Spec: M0-W-1032, M0-W-1421, KERNEL-SAMPLE-003
 * Story: FT-017, FT-018, FT-019, FT-019-EXT-1
 */

#include "sampling.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>

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

/**
 * Softmax kernel with numerical stability (log-sum-exp trick).
 * 
 * Converts logits to probabilities: probs[i] = exp(logits[i]) / sum(exp(logits))
 * Uses max subtraction to prevent overflow: exp(x - max) / sum(exp(x - max))
 * 
 * Single-block implementation for simplicity (assumes vocab_size <= 65536).
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param probs Device pointer to output probabilities [vocab_size]
 * @param vocab_size Vocabulary size
 */
__global__ void softmax_fp32(
    const float* logits,
    float* probs,
    int vocab_size
) {
    // Shared memory for reduction
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Find max (for numerical stability)
    float local_max = (idx < vocab_size) ? logits[idx] : -INFINITY;
    
    // Grid-stride loop to handle large vocabularies
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, logits[i]);
    }
    
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = shared_max[0];
    __syncthreads();
    
    // Phase 2: Compute exp(logit - max) and sum
    float local_exp = 0.0f;
    if (idx < vocab_size) {
        local_exp = expf(logits[idx] - global_max);
    }
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        local_exp += expf(logits[i] - global_max);
    }
    
    shared_sum[tid] = local_exp;
    __syncthreads();
    
    // Reduce to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float global_sum = shared_sum[0];
    __syncthreads();
    
    // Phase 3: Normalize
    if (idx < vocab_size) {
        probs[idx] = expf(logits[idx] - global_max) / global_sum;
    }
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        probs[i] = expf(logits[i] - global_max) / global_sum;
    }
}

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

/**
 * Launch stochastic sampling pipeline.
 * 
 * Three-phase pipeline:
 * 1. Softmax: Convert logits to probabilities
 * 2. Sample: Select token from probability distribution
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param random_value Random value from RNG [0, 1)
 * @param stream CUDA stream (default: 0)
 * @return Selected token ID
 */
int launch_stochastic_sample(
    const float* logits,
    int vocab_size,
    float random_value,
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
    
    if (random_value < 0.0f || random_value >= 1.0f) {
        fprintf(stderr, "Invalid random_value: %f (must be in [0, 1))\n", random_value);
        return -1;
    }
    
    // Allocate temporary storage
    float* d_probs;
    int* d_token_id;
    
    cudaMalloc(&d_probs, vocab_size * sizeof(float));
    cudaMalloc(&d_token_id, sizeof(int));
    
    // Kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    // Cap at 256 blocks for efficiency
    if (num_blocks > 256) {
        num_blocks = 256;
    }
    
    // Phase 1: Softmax
    softmax_fp32<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, d_probs, vocab_size
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Softmax kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_probs);
        cudaFree(d_token_id);
        return -1;
    }
    
    // Phase 2: Sample from distribution
    sample_from_distribution<<<1, 1, 0, stream>>>(
        d_probs, vocab_size, random_value, d_token_id
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Sample kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_probs);
        cudaFree(d_token_id);
        return -1;
    }
    
    // Copy result back to host
    int h_token_id;
    cudaMemcpyAsync(&h_token_id, d_token_id, sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Free temporary storage
    cudaFree(d_probs);
    cudaFree(d_token_id);
    
    return h_token_id;
}

/**
 * Top-K filtering kernel using Thrust for sorting.
 * 
 * Keeps only the top k tokens by logit value. All other tokens are set to -INFINITY.
 * 
 * Algorithm:
 * 1. Create indices array [0, 1, 2, ..., vocab_size-1]
 * 2. Sort logits descending (keeping track of indices)
 * 3. Zero out logits outside top k
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param top_k Keep only top k tokens
 */
__global__ void apply_top_k(
    float* logits,
    int vocab_size,
    int top_k
) {
    // This kernel is a placeholder - actual work done in launch function
    // using Thrust for sorting efficiency
}

/**
 * Launch top-k filtering kernel.
 * 
 * Uses Thrust library for efficient sorting.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param top_k Keep only top k tokens (0 = disabled)
 * @param stream CUDA stream
 */
void launch_top_k(
    float* logits,
    int vocab_size,
    int top_k,
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
    
    // If top_k disabled or >= vocab_size, no filtering
    if (top_k <= 0 || top_k >= vocab_size) {
        return;
    }
    
    // Create device vectors for sorting
    thrust::device_ptr<float> d_logits_ptr(logits);
    thrust::device_vector<int> indices(vocab_size);
    thrust::sequence(thrust::device, indices.begin(), indices.end());
    
    // Sort by logits descending (keep track of original indices)
    thrust::device_vector<float> logits_copy(d_logits_ptr, d_logits_ptr + vocab_size);
    thrust::sort_by_key(
        thrust::device,
        logits_copy.begin(),
        logits_copy.end(),
        indices.begin(),
        thrust::greater<float>()
    );
    
    // Zero out logits outside top k
    // Create a kernel to set filtered logits to -INFINITY
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    
    // Lambda kernel to filter
    auto filter_kernel = [=] __device__ (int idx) {
        if (idx >= vocab_size) return;
        
        // Check if this index is in top k
        bool in_top_k = false;
        for (int i = 0; i < top_k; ++i) {
            if (thrust::raw_pointer_cast(indices.data())[i] == idx) {
                in_top_k = true;
                break;
            }
        }
        
        if (!in_top_k) {
            logits[idx] = -INFINITY;
        }
    };
    
    // Alternative: use a simple kernel
    // For now, copy sorted indices and filter on CPU side is more reliable
    thrust::host_vector<int> h_top_k_indices(top_k);
    thrust::copy(indices.begin(), indices.begin() + top_k, h_top_k_indices.begin());
    
    // Create a mask on device
    thrust::device_vector<bool> mask(vocab_size, false);
    for (int i = 0; i < top_k; ++i) {
        mask[h_top_k_indices[i]] = true;
    }
    
    // Apply mask to logits
    thrust::transform(
        thrust::device,
        d_logits_ptr,
        d_logits_ptr + vocab_size,
        mask.begin(),
        d_logits_ptr,
        [] __device__ (float logit, bool keep) {
            return keep ? logit : -INFINITY;
        }
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Top-k filtering failed: %s\n", cudaGetErrorString(err));
    }
}

/**
 * Top-P (nucleus) filtering kernel using Thrust.
 * 
 * Keeps tokens whose cumulative probability <= top_p.
 * 
 * Algorithm:
 * 1. Sort logits descending
 * 2. Compute softmax on sorted logits
 * 3. Compute cumulative sum
 * 4. Find cutoff where cumsum > top_p
 * 5. Zero out logits below cutoff
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param top_p Cumulative probability threshold (0.0-1.0)
 */
__global__ void apply_top_p(
    float* logits,
    int vocab_size,
    float top_p
) {
    // Placeholder - actual work done in launch function
}

/**
 * Launch top-p (nucleus) filtering kernel.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param top_p Cumulative probability threshold (0.0-1.0)
 * @param stream CUDA stream
 */
void launch_top_p(
    float* logits,
    int vocab_size,
    float top_p,
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
    
    if (top_p < 0.0f || top_p > 1.0f) {
        fprintf(stderr, "Invalid top_p: %f (must be in [0.0, 1.0])\n", top_p);
        return;
    }
    
    // If top_p >= 1.0, no filtering
    if (top_p >= 1.0f) {
        return;
    }
    
    // Special case: top_p = 0.0 means keep only the max token (greedy-like)
    if (top_p <= 0.0f) {
        // Find max logit
        thrust::device_ptr<float> d_logits_ptr(logits);
        auto max_iter = thrust::max_element(thrust::device, d_logits_ptr, d_logits_ptr + vocab_size);
        int max_idx = max_iter - d_logits_ptr;
        float max_val = *max_iter;
        
        // Set all except max to -INFINITY
        thrust::transform(
            thrust::device,
            d_logits_ptr,
            d_logits_ptr + vocab_size,
            thrust::counting_iterator<int>(0),
            d_logits_ptr,
            [max_idx, max_val] __device__ (float logit, int idx) {
                return (idx == max_idx) ? max_val : -INFINITY;
            }
        );
        return;
    }
    
    // Create device vectors
    thrust::device_ptr<float> d_logits_ptr(logits);
    thrust::device_vector<float> sorted_logits(d_logits_ptr, d_logits_ptr + vocab_size);
    thrust::device_vector<int> indices(vocab_size);
    thrust::sequence(thrust::device, indices.begin(), indices.end());
    
    // Sort logits descending
    thrust::sort_by_key(
        thrust::device,
        sorted_logits.begin(),
        sorted_logits.end(),
        indices.begin(),
        thrust::greater<float>()
    );
    
    // Optimization: Only copy first N tokens to host (top-p rarely needs more than 1000 tokens)
    // This dramatically reduces host-device transfer time for large vocabularies
    int max_copy = std::min(vocab_size, 1000);
    thrust::host_vector<float> h_sorted_logits(sorted_logits.begin(), sorted_logits.begin() + max_copy);
    
    // Compute softmax normalization factor (only for copied portion)
    float max_logit = h_sorted_logits[0];
    float sum = 0.0f;
    for (int i = 0; i < max_copy; ++i) {
        sum += expf(h_sorted_logits[i] - max_logit);
    }
    
    // Find cutoff where cumsum > top_p
    float cumsum = 0.0f;
    int cutoff = 1;  // Always keep at least the max token
    for (int i = 0; i < max_copy; ++i) {
        float prob = expf(h_sorted_logits[i] - max_logit) / sum;
        cumsum += prob;
        if (cumsum > top_p) {
            cutoff = i + 1;  // Keep tokens up to and including this one
            break;
        }
    }
    
    // If we didn't find cutoff in first max_copy tokens, keep all of them
    // (This is rare - means distribution is very flat)
    if (cutoff == 1 && cumsum <= top_p && max_copy < vocab_size) {
        cutoff = max_copy;
    }
    
    // Create mask: keep tokens in top cutoff positions
    // Only copy the indices we need (not all vocab_size)
    thrust::host_vector<int> h_indices(indices.begin(), indices.begin() + cutoff);
    thrust::device_vector<bool> mask(vocab_size, false);
    for (int i = 0; i < cutoff; ++i) {
        mask[h_indices[i]] = true;
    }
    
    // Apply mask to logits
    thrust::transform(
        thrust::device,
        d_logits_ptr,
        d_logits_ptr + vocab_size,
        mask.begin(),
        d_logits_ptr,
        [] __device__ (float logit, bool keep) {
            return keep ? logit : -INFINITY;
        }
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Top-p filtering failed: %s\n", cudaGetErrorString(err));
    }
}

/**
 * Repetition penalty kernel.
 * 
 * Applies penalty to tokens that appear in generation history.
 * Each thread handles one token in the vocabulary.
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param history Device pointer to generated token IDs
 * @param history_length Number of tokens in history
 * @param penalty Repetition penalty factor
 */
__global__ void apply_repetition_penalty(
    float* logits,
    int vocab_size,
    const int* history,
    int history_length,
    float penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= vocab_size) {
        return;
    }
    
    // If penalty disabled or no history, skip
    if (penalty == 1.0f || history == nullptr || history_length == 0) {
        return;
    }
    
    // Check if this token is in history
    bool in_history = false;
    for (int i = 0; i < history_length; ++i) {
        if (history[i] == idx) {
            in_history = true;
            break;
        }
    }
    
    // Apply penalty if in history
    if (in_history) {
        if (logits[idx] > 0.0f) {
            logits[idx] /= penalty;
        } else {
            logits[idx] *= penalty;
        }
    }
}

/**
 * Launch repetition penalty kernel.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param history Device pointer to generated token IDs
 * @param history_length Number of tokens in history
 * @param penalty Repetition penalty factor
 * @param stream CUDA stream
 */
void launch_repetition_penalty(
    float* logits,
    int vocab_size,
    const int* history,
    int history_length,
    float penalty,
    cudaStream_t stream
) {
    // Validate inputs
    if (penalty == 1.0f || history == nullptr || history_length == 0) {
        return;  // No penalty to apply
    }
    
    if (vocab_size <= 0 || logits == nullptr) {
        fprintf(stderr, "Invalid inputs to repetition penalty\n");
        return;
    }
    
    // Kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    
    apply_repetition_penalty<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, vocab_size, history, history_length, penalty
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Repetition penalty kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

/**
 * Min-P filtering kernel.
 * 
 * Filters tokens where prob < min_p * max_prob.
 * Uses parallel reduction to find max logit, then filters.
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param min_p Minimum probability threshold
 */
__global__ void apply_min_p(
    float* logits,
    int vocab_size,
    float min_p
) {
    __shared__ float shared_max[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Find max logit
    float local_max = (idx < vocab_size) ? logits[idx] : -INFINITY;
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, logits[i]);
    }
    
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = shared_max[0];
    __syncthreads();
    
    // Phase 2: Compute threshold and filter
    // threshold_logit = log(min_p) + max_logit
    float threshold_logit = logf(min_p) + global_max;
    
    // Filter tokens below threshold
    if (idx < vocab_size) {
        if (logits[idx] < threshold_logit) {
            logits[idx] = -INFINITY;
        }
    }
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        if (logits[i] < threshold_logit) {
            logits[i] = -INFINITY;
        }
    }
}

/**
 * Launch min-p filtering kernel.
 * 
 * @param logits Device pointer to logits [vocab_size]
 * @param vocab_size Vocabulary size
 * @param min_p Minimum probability threshold (0.0-1.0)
 * @param stream CUDA stream
 */
void launch_min_p(
    float* logits,
    int vocab_size,
    float min_p,
    cudaStream_t stream
) {
    if (min_p <= 0.0f) {
        return;  // Disabled
    }
    
    if (vocab_size <= 0 || logits == nullptr) {
        fprintf(stderr, "Invalid inputs to min-p\n");
        return;
    }
    
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 256) {
        num_blocks = 256;
    }
    
    apply_min_p<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, vocab_size, min_p
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Min-p kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

/**
 * Check if generated sequence matches any stop sequence.
 * 
 * CPU-side implementation for pattern matching.
 * 
 * @param generated_tokens Host pointer to generated token IDs
 * @param num_generated Number of generated tokens
 * @param stop_sequences Array of stop sequence pointers (up to 4)
 * @param stop_lengths Array of stop sequence lengths
 * @param num_stop_sequences Number of stop sequences
 * @return True if any stop sequence matched
 */
bool check_stop_sequences(
    const int* generated_tokens,
    int num_generated,
    const int* stop_sequences[4],
    const int stop_lengths[4],
    int num_stop_sequences
) {
    // Check each stop sequence
    for (int seq_idx = 0; seq_idx < num_stop_sequences && seq_idx < 4; ++seq_idx) {
        const int* seq = stop_sequences[seq_idx];
        int seq_len = stop_lengths[seq_idx];
        
        // Skip if sequence is empty or nullptr
        if (seq == nullptr || seq_len == 0) {
            continue;
        }
        
        // Need at least seq_len tokens to match
        if (num_generated < seq_len) {
            continue;
        }
        
        // Check if last seq_len tokens match stop sequence
        bool match = true;
        for (int i = 0; i < seq_len; ++i) {
            int gen_idx = num_generated - seq_len + i;
            if (generated_tokens[gen_idx] != seq[i]) {
                match = false;
                break;
            }
        }
        
        if (match) {
            return true;  // Stop sequence matched
        }
    }
    
    return false;  // No match
}

} // namespace kernels
} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️

