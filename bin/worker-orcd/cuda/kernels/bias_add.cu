// bias_add.cu — Bias Addition Kernel
//
// Adds bias vector to matrix rows (for attention Q/K/V projections)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdio.h>

/**
 * Add bias to each row of a matrix
 * 
 * output[i, j] = input[i, j] + bias[j]
 */
__global__ void bias_add_kernel(
    half* output,
    const half* input,
    const half* bias,
    int batch_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dim;
    
    if (idx < total) {
        int col = idx % dim;
        output[idx] = __hadd(input[idx], bias[col]);
    }
}

extern "C" {

/**
 * Add bias vector to matrix (in-place or out-of-place)
 * 
 * @param output Output matrix [batch, dim]
 * @param input Input matrix [batch, dim] (can be same as output for in-place)
 * @param bias Bias vector [dim]
 * @param batch_size Number of rows
 * @param dim Dimension of each row
 * @param stream CUDA stream
 */
void cuda_bias_add(
    void* output,
    const void* input,
    const void* bias,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    half* output_half = reinterpret_cast<half*>(output);
    const half* input_half = reinterpret_cast<const half*>(input);
    const half* bias_half = reinterpret_cast<const half*>(bias);
    
    int total = batch_size * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    bias_add_kernel<<<blocks, threads, 0, stream>>>(
        output_half,
        input_half,
        bias_half,
        batch_size,
        dim
    );
}

} // extern "C"
