// GELU activation kernel for GPT architecture
//
// Implements exact GELU (Gaussian Error Linear Unit) activation:
// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// where Φ(x) is the cumulative distribution function of the standard normal distribution.
//
// This differs from Llama's SwiGLU activation.
//
// Spec: M0-W-1433, M0-W-1215
// Story: GT-012

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Constants for GELU
#define GELU_SQRT_2 1.41421356237f
#define GELU_HALF 0.5f

// GELU activation kernel (exact formula)
//
// Applies GELU activation element-wise:
// y = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// Args:
//   output: Output tensor [batch_size, seq_len, hidden_size]
//   input: Input tensor [batch_size, seq_len, hidden_size]
//   size: Total number of elements
__global__ void gelu_kernel(
    half* output,
    const half* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = __half2float(input[idx]);
        
        // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        float erf_arg = x / GELU_SQRT_2;
        float erf_val = erff(erf_arg);  // CUDA intrinsic for error function
        float gelu_val = x * GELU_HALF * (1.0f + erf_val);
        
        output[idx] = __float2half(gelu_val);
    }
}

// GELU activation using tanh approximation (faster but less accurate)
//
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// This is faster than exact GELU but introduces ~0.1% error.
// Use for performance-critical paths if accuracy loss is acceptable.
__global__ void gelu_tanh_approx_kernel(
    half* output,
    const half* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = __half2float(input[idx]);
        
        // Constants for tanh approximation
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
        const float coeff = 0.044715f;
        
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanhf(tanh_arg);
        float gelu_val = GELU_HALF * x * (1.0f + tanh_val);
        
        output[idx] = __float2half(gelu_val);
    }
}

// Host function to launch GELU kernel (exact formula)
extern "C" void cuda_gelu(
    half* output,
    const half* input,
    int size,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    gelu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output, input, size
    );
}

// Host function to launch GELU kernel (tanh approximation)
extern "C" void cuda_gelu_tanh_approx(
    half* output,
    const half* input,
    int size,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    gelu_tanh_approx_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output, input, size
    );
}

// In-place GELU (modifies input tensor)
// Useful for memory-constrained scenarios
__global__ void gelu_inplace_kernel(
    half* data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = __half2float(data[idx]);
        float erf_arg = x / GELU_SQRT_2;
        float erf_val = erff(erf_arg);
        float gelu_val = x * GELU_HALF * (1.0f + erf_val);
        data[idx] = __float2half(gelu_val);
    }
}

extern "C" void cuda_gelu_inplace(
    half* data,
    int size,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    gelu_inplace_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        data, size
    );
}

// Fused GELU + scale kernel
// Computes: y = GELU(x) * scale
// Useful for combining GELU with layer-specific scaling
__global__ void gelu_scale_kernel(
    half* output,
    const half* input,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = __half2float(input[idx]);
        float erf_arg = x / GELU_SQRT_2;
        float erf_val = erff(erf_arg);
        float gelu_val = x * GELU_HALF * (1.0f + erf_val);
        output[idx] = __float2half(gelu_val * scale);
    }
}

extern "C" void cuda_gelu_scale(
    half* output,
    const half* input,
    float scale,
    int size,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    
    gelu_scale_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output, input, scale, size
    );
}

// ---
// Crafted by GPT-Gamma 🤖
