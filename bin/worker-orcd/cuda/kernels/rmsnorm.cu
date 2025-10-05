// rmsnorm.cu — RMSNorm layer normalization - LT-013
//
// Implements RMSNorm used in Llama models (pre/post layer normalization).
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
//
// Spec: M0-W-1214, M0-W-1430

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

/**
 * RMSNorm kernel - normalizes activations using root mean square
 * 
 * Formula:
 *   rms = sqrt(mean(x^2) + eps)
 *   output = (x / rms) * weight
 */
__global__ void rmsnorm_kernel(
    half* output,
    const half* input,
    const half* weight,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps
) {
    int token_idx = blockIdx.x;  // Token index (batch * seq_len)
    int tid = threadIdx.x;       // Thread index
    
    if (token_idx >= batch_size * seq_len) return;
    
    const half* x = input + token_idx * hidden_dim;
    half* y = output + token_idx * hidden_dim;
    
    // Shared memory for parallel reduction
    __shared__ float sum_sq[256];
    
    // 1. Compute sum of squares (parallel reduction)
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        thread_sum += val * val;
    }
    sum_sq[tid] = thread_sum;
    __syncthreads();
    
    // 2. Reduce sum_sq to single value
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_sq[tid] += sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // 3. Compute RMS
    float mean_sq = sum_sq[0] / hidden_dim;
    float rms = sqrtf(mean_sq + eps);
    
    // 4. Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        y[i] = __float2half((val / rms) * w);
    }
}

extern "C" {

/**
 * Apply RMSNorm to input tensor
 * 
 * @param output Output tensor [batch, seq_len, hidden_dim]
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param weight Weight tensor [hidden_dim]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param eps Epsilon for numerical stability (default 1e-6)
 * @return 0 on success, error code on failure
 */
int cuda_rmsnorm_forward(
    half* output,
    const half* input,
    const half* weight,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps
) {
    // Validate dimensions
    if (batch_size <= 0 || seq_len <= 0 || hidden_dim <= 0) {
        fprintf(stderr, "RMSNorm: Invalid dimensions\n");
        return -1;
    }
    
    if (eps <= 0.0f) {
        fprintf(stderr, "RMSNorm: eps must be positive\n");
        return -1;
    }
    
    // Launch kernel
    // One block per token, 256 threads per block
    int num_tokens = batch_size * seq_len;
    dim3 grid(num_tokens);
    dim3 block(256);
    
    rmsnorm_kernel<<<grid, block>>>(
        output, input, weight,
        batch_size, seq_len, hidden_dim, eps
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RMSNorm kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/**
 * Wrapper for transformer - single token inference
 * 
 * This matches the signature expected by QwenTransformer.
 * For single token generation, seq_len = 1.
 */
void cuda_rmsnorm_forward(
    const void* input,
    const void* weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    float eps,
    cudaStream_t stream
) {
    // For single token generation, seq_len = 1
    const int seq_len = 1;
    
    cuda_rmsnorm_forward(
        reinterpret_cast<half*>(output),
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(weight),
        static_cast<int>(batch_size),
        seq_len,
        static_cast<int>(hidden_dim),
        eps
    );
    
    // Note: stream parameter ignored for now (using default stream)
}

} // extern "C"
