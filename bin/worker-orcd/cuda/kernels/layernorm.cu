// LayerNorm kernel for GPT architecture
//
// Implements standard LayerNorm with mean and variance normalization:
// y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
//
// This differs from Llama's RMSNorm which only normalizes by RMS without mean centering.
//
// Spec: M0-W-1432, M0-W-1215
// Story: GT-009, GT-010

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// LayerNorm kernel with mean and variance normalization
// 
// Args:
//   output: Output tensor [batch_size, seq_len, hidden_size]
//   input: Input tensor [batch_size, seq_len, hidden_size]
//   gamma: Scale parameter [hidden_size]
//   beta: Bias parameter [hidden_size]
//   batch_size: Batch size
//   seq_len: Sequence length
//   hidden_size: Hidden dimension (d_model)
//   epsilon: Small constant for numerical stability (typically 1e-5)
//
// Each block processes one sequence position (batch_idx, seq_idx)
// Threads within block cooperatively compute mean, variance, and normalize
__global__ void layernorm_kernel(
    half* output,
    const half* input,
    const half* gamma,
    const half* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon
) {
    // Each block handles one (batch, seq) position
    int batch_idx = blockIdx.x / seq_len;
    int seq_idx = blockIdx.x % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    // Input/output offset for this position
    int offset = (batch_idx * seq_len + seq_idx) * hidden_size;
    const half* x = input + offset;
    half* y = output + offset;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = shared_mem + blockDim.x;
    
    // Step 1: Compute mean
    // Each thread sums a subset of elements
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += __half2float(x[i]);
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduce to compute total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / hidden_size;
    __syncthreads();
    
    // Step 2: Compute variance
    // variance = E[(x - mean)^2]
    float thread_sq_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = __half2float(x[i]) - mean;
        thread_sq_sum += diff * diff;
    }
    shared_sq_sum[threadIdx.x] = thread_sq_sum;
    __syncthreads();
    
    // Reduce to compute total squared sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / hidden_size;
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    __syncthreads();
    
    // Step 3: Normalize and apply affine transformation
    // y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (__half2float(x[i]) - mean) * inv_std;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        y[i] = __float2half(scaled);
    }
}

// Host function to launch LayerNorm kernel
extern "C" void cuda_layernorm(
    half* output,
    const half* input,
    const half* gamma,
    const half* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon,
    cudaStream_t stream
) {
    // One block per (batch, seq) position
    int num_blocks = batch_size * seq_len;
    
    // Threads per block (tune for performance)
    int threads_per_block = 256;
    if (hidden_size < 256) {
        threads_per_block = 128;
    }
    
    // Shared memory: 2 arrays of size threads_per_block for sum and sq_sum
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    layernorm_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        output, input, gamma, beta,
        batch_size, seq_len, hidden_size, epsilon
    );
}

// Fused LayerNorm + Residual kernel
// Computes: y = LayerNorm(x + residual)
//
// This is a common pattern in transformers where we add residual connection
// before normalization. Fusing saves memory bandwidth.
__global__ void layernorm_residual_kernel(
    half* output,
    const half* input,
    const half* residual,
    const half* gamma,
    const half* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon
) {
    int batch_idx = blockIdx.x / seq_len;
    int seq_idx = blockIdx.x % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    int offset = (batch_idx * seq_len + seq_idx) * hidden_size;
    const half* x = input + offset;
    const half* r = residual + offset;
    half* y = output + offset;
    
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = shared_mem + blockDim.x;
    
    // Step 1: Compute mean of (x + residual)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]) + __half2float(r[i]);
        thread_sum += val;
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / hidden_size;
    __syncthreads();
    
    // Step 2: Compute variance
    float thread_sq_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]) + __half2float(r[i]);
        float diff = val - mean;
        thread_sq_sum += diff * diff;
    }
    shared_sq_sum[threadIdx.x] = thread_sq_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / hidden_size;
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    __syncthreads();
    
    // Step 3: Normalize
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]) + __half2float(r[i]);
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        y[i] = __float2half(scaled);
    }
}

extern "C" void cuda_layernorm_residual(
    half* output,
    const half* input,
    const half* residual,
    const half* gamma,
    const half* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon,
    cudaStream_t stream
) {
    int num_blocks = batch_size * seq_len;
    int threads_per_block = 256;
    if (hidden_size < 256) {
        threads_per_block = 128;
    }
    
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    layernorm_residual_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        output, input, residual, gamma, beta,
        batch_size, seq_len, hidden_size, epsilon
    );
}

// ---
// Crafted by GPT-Gamma 🤖
