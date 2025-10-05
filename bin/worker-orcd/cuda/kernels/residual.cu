// residual.cu — Residual Connection Kernel - LT-014
//
// Implements residual connections for Llama transformer blocks.
// output = input + residual (element-wise addition)
//
// Spec: M0-W-1214

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdio.h>

/**
 * Residual connection kernel - element-wise addition
 * 
 * Formula:
 *   output[i] = input[i] + residual[i]
 */
__global__ void residual_kernel(
    half* output,
    const half* input,
    const half* residual,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = __hadd(input[idx], residual[idx]);
    }
}

/**
 * Vectorized residual kernel using half2 for better throughput
 */
__global__ void residual_kernel_vectorized(
    half2* output,
    const half2* input,
    const half2* residual,
    int total_elements_half2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements_half2) {
        output[idx] = __hadd2(input[idx], residual[idx]);
    }
}

extern "C" {

/**
 * Apply residual connection
 * 
 * @param output Output tensor [batch, seq_len, hidden_dim]
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param residual Residual tensor [batch, seq_len, hidden_dim]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param in_place If true, output += residual (input ignored)
 * @return 0 on success, error code on failure
 */
int cuda_residual_forward(
    half* output,
    const half* input,
    const half* residual,
    int batch_size,
    int seq_len,
    int hidden_dim,
    bool in_place
) {
    // Validate dimensions
    if (batch_size <= 0 || seq_len <= 0 || hidden_dim <= 0) {
        fprintf(stderr, "Residual: Invalid dimensions\n");
        return -1;
    }
    
    int total_elements = batch_size * seq_len * hidden_dim;
    
    // Use vectorized kernel if hidden_dim is even
    if (hidden_dim % 2 == 0) {
        int total_half2 = total_elements / 2;
        int threads = 256;
        int blocks = (total_half2 + threads - 1) / threads;
        
        if (in_place) {
            residual_kernel_vectorized<<<blocks, threads>>>(
                (half2*)output,
                (const half2*)output,
                (const half2*)residual,
                total_half2
            );
        } else {
            residual_kernel_vectorized<<<blocks, threads>>>(
                (half2*)output,
                (const half2*)input,
                (const half2*)residual,
                total_half2
            );
        }
    } else {
        // Fall back to non-vectorized kernel
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        if (in_place) {
            residual_kernel<<<blocks, threads>>>(
                output,
                output,
                residual,
                total_elements
            );
        } else {
            residual_kernel<<<blocks, threads>>>(
                output,
                input,
                residual,
                total_elements
            );
        }
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Residual kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/**
 * Wrapper for transformer compatibility
 * Maps cuda_residual_add to cuda_residual_forward
 */
void cuda_residual_add(
    const void* input,
    const void* residual,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    cudaStream_t stream
) {
    // Calculate total elements
    int total_elements = batch_size * hidden_dim;
    
    // Cast to half pointers
    const half* input_half = reinterpret_cast<const half*>(input);
    const half* residual_half = reinterpret_cast<const half*>(residual);
    half* output_half = reinterpret_cast<half*>(output);
    
    // Use vectorized kernel if possible
    if (hidden_dim % 2 == 0) {
        int total_half2 = total_elements / 2;
        int threads = 256;
        int blocks = (total_half2 + threads - 1) / threads;
        
        residual_kernel_vectorized<<<blocks, threads, 0, stream>>>(
            (half2*)output_half,
            (const half2*)input_half,
            (const half2*)residual_half,
            total_half2
        );
    } else {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        residual_kernel<<<blocks, threads, 0, stream>>>(
            output_half,
            input_half,
            residual_half,
            total_elements
        );
    }
}

} // extern "C"
