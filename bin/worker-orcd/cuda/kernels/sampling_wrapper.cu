// sampling_wrapper.cu — Unified Sampling Interface
//
// Provides extern "C" wrapper for sampling operations
// Combines temperature, top-k, top-p, and random sampling
//
// Spec: M0-W-1032

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <stdio.h>
#include <algorithm>

// Forward declarations from sampling.cu
namespace worker {
namespace kernels {
    void launch_temperature_scale_fp32(float* logits, int vocab_size, float temperature, cudaStream_t stream);
    void launch_top_k(float* logits, int vocab_size, int top_k, cudaStream_t stream);
    void launch_top_p(float* logits, int vocab_size, float top_p, cudaStream_t stream);
}
}

/**
 * Softmax kernel for converting logits to probabilities
 */
__global__ void softmax_kernel(
    const float* logits,
    float* probs,
    int vocab_size
) {
    // Single block, single thread for simplicity (vocab_size is large but manageable)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Find max for numerical stability
        float max_logit = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_logit && !isinf(logits[i])) {
                max_logit = logits[i];
            }
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (isinf(logits[i]) && logits[i] < 0) {
                probs[i] = 0.0f;  // Filtered out token
            } else {
                probs[i] = expf(logits[i] - max_logit);
                sum += probs[i];
            }
        }
        
        // Normalize
        if (sum > 0.0f) {
            for (int i = 0; i < vocab_size; i++) {
                probs[i] /= sum;
            }
        }
    }
}

/**
 * Sample from probability distribution using cuRAND
 */
__global__ void sample_kernel(
    const float* probs,
    int vocab_size,
    uint64_t seed,
    int* output_token
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize cuRAND state
        curandState state;
        curand_init(seed, 0, 0, &state);
        
        // Generate random number [0, 1)
        float rand_val = curand_uniform(&state);
        
        // Sample using cumulative probability
        float cumsum = 0.0f;
        int selected_token = 0;
        
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i];
            if (rand_val <= cumsum) {
                selected_token = i;
                break;
            }
        }
        
        *output_token = selected_token;
    }
}

/**
 * Greedy sampling (argmax)
 */
__global__ void argmax_kernel(
    const float* logits,
    int vocab_size,
    int* output_token
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_val = -INFINITY;
        int max_idx = 0;
        
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        
        // Debug: Print first few logits and max
        static int call_count = 0;
        if (call_count < 15) {  // Increased to see generation phase
            printf("🔍 [ARGMAX DEBUG #%d] First 10 logits: ", call_count);
            for (int i = 0; i < 10 && i < vocab_size; i++) {
                printf("%.2f ", logits[i]);
            }
            printf("\n");
            printf("🔍 [ARGMAX DEBUG #%d] Max: %.2f at token_id=%d (vocab_size=%d)\n", call_count, max_val, max_idx, vocab_size);
            call_count++;
        }
        
        *output_token = max_idx;
    }
}

extern "C" {

/**
 * Unified sampling function
 * 
 * Applies temperature, top-k, top-p filtering, then samples
 * 
 * @param logits Input logits [vocab_size] (FP32)
 * @param vocab_size Vocabulary size
 * @param temperature Sampling temperature (0.0 = greedy, >0 = stochastic)
 * @param top_k Keep only top k tokens (0 = disabled)
 * @param top_p Nucleus sampling threshold (0.0-1.0, 0 = disabled)
 * @param seed Random seed
 * @return Sampled token ID
 */
int cuda_sample_token(
    float* logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint64_t seed
) {
    // Allocate device memory for intermediate results
    float* d_probs;
    int* d_token;
    cudaMalloc(&d_probs, vocab_size * sizeof(float));
    cudaMalloc(&d_token, sizeof(int));
    
    // Greedy sampling (temperature = 0)
    if (temperature == 0.0f) {
        argmax_kernel<<<1, 1>>>(logits, vocab_size, d_token);
    } else {
        // Apply temperature scaling
        worker::kernels::launch_temperature_scale_fp32(
            logits, vocab_size, temperature, nullptr
        );
        
        // Apply top-k filtering
        if (top_k > 0 && top_k < vocab_size) {
            worker::kernels::launch_top_k(
                logits, vocab_size, top_k, nullptr
            );
        }
        
        // Apply top-p filtering
        if (top_p > 0.0f && top_p < 1.0f) {
            worker::kernels::launch_top_p(
                logits, vocab_size, top_p, nullptr
            );
        }
        
        // Compute softmax
        softmax_kernel<<<1, 1>>>(logits, d_probs, vocab_size);
        
        // Sample from distribution
        sample_kernel<<<1, 1>>>(d_probs, vocab_size, seed, d_token);
    }
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_token, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_probs);
    cudaFree(d_token);
    
    return result;
}

/**
 * Simplified sampling for testing
 * Always uses greedy (argmax)
 */
int cuda_sample_token_greedy(
    const float* logits,
    uint32_t vocab_size
) {
    int* d_token;
    cudaMalloc(&d_token, sizeof(int));
    
    argmax_kernel<<<1, 1>>>(logits, vocab_size, d_token);
    
    int result;
    cudaMemcpy(&result, d_token, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_token);
    
    return result;
}

} // extern "C"

// ---
// Crafted by GPT-Gamma 🤖
