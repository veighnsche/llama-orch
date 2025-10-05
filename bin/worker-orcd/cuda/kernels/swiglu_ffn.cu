// swiglu_ffn.cu — Full SwiGLU Feed-Forward Network
//
// Implements complete SwiGLU FFN with gate/up/down projections
// FFN(x) = down(silu(gate(x)) * up(x))
//
// Spec: M0-W-1217

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>

// External cuBLAS GEMM function
extern "C" void gemm_fp16(
    cublasHandle_t handle,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    int lda,
    const half* B,
    int ldb,
    float beta,
    half* C,
    int ldc
);

// External SwiGLU activation kernel
extern "C" int cuda_swiglu_activation(
    half* output,
    const half* gate,
    const half* up,
    int batch_size,
    int seq_len,
    int ffn_dim
);

extern "C" {

/**
 * Full SwiGLU FFN forward pass
 * 
 * Computes: output = down(silu(gate(input)) * up(input))
 * 
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param gate_weight Gate projection weight [ffn_dim, hidden_dim]
 * @param up_weight Up projection weight [ffn_dim, hidden_dim]
 * @param down_weight Down projection weight [hidden_dim, ffn_dim]
 * @param output Output tensor [batch, seq_len, hidden_dim]
 * @param batch_size Batch size
 * @param hidden_dim Hidden dimension
 * @param ffn_dim FFN intermediate dimension
 * @param stream CUDA stream
 */
void cuda_swiglu_forward(
    const void* input,
    const void* gate_weight,
    const void* up_weight,
    const void* down_weight,
    void* output,
    uint32_t batch_size,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    cudaStream_t stream
) {
    const half* input_half = reinterpret_cast<const half*>(input);
    const half* gate_weight_half = reinterpret_cast<const half*>(gate_weight);
    const half* up_weight_half = reinterpret_cast<const half*>(up_weight);
    const half* down_weight_half = reinterpret_cast<const half*>(down_weight);
    half* output_half = reinterpret_cast<half*>(output);
    
    // Allocate intermediate buffers
    half* gate_out;
    half* up_out;
    half* swiglu_out;
    
    size_t intermediate_size = batch_size * ffn_dim * sizeof(half);
    cudaMalloc(&gate_out, intermediate_size);
    cudaMalloc(&up_out, intermediate_size);
    cudaMalloc(&swiglu_out, intermediate_size);
    
    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
    
    // 1. Gate projection: gate_out = input @ gate_weight^T
    //    input: [batch, hidden_dim]
    //    gate_weight: [ffn_dim, hidden_dim]
    //    gate_out: [batch, ffn_dim]
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,  // Transpose gate_weight
        CUBLAS_OP_N,  // No transpose input
        ffn_dim,      // M
        batch_size,   // N
        hidden_dim,   // K
        &alpha,
        gate_weight_half, CUDA_R_16F, hidden_dim,
        input_half, CUDA_R_16F, hidden_dim,
        &beta,
        gate_out, CUDA_R_16F, ffn_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // 2. Up projection: up_out = input @ up_weight^T
    status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        ffn_dim,
        batch_size,
        hidden_dim,
        &alpha,
        up_weight_half, CUDA_R_16F, hidden_dim,
        input_half, CUDA_R_16F, hidden_dim,
        &beta,
        up_out, CUDA_R_16F, ffn_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // 3. SwiGLU activation: swiglu_out = silu(gate_out) * up_out
    cuda_swiglu_activation(
        swiglu_out,
        gate_out,
        up_out,
        batch_size,
        1,  // seq_len = 1 for single token
        ffn_dim
    );
    
    // 4. Down projection: output = swiglu_out @ down_weight^T
    //    swiglu_out: [batch, ffn_dim]
    //    down_weight: [hidden_dim, ffn_dim]
    //    output: [batch, hidden_dim]
    status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        hidden_dim,
        batch_size,
        ffn_dim,
        &alpha,
        down_weight_half, CUDA_R_16F, ffn_dim,
        swiglu_out, CUDA_R_16F, ffn_dim,
        &beta,
        output_half, CUDA_R_16F, hidden_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // Cleanup
    cudaFree(gate_out);
    cudaFree(up_out);
    cudaFree(swiglu_out);
    cublasDestroy(cublas_handle);
}

} // extern "C"

// ---
// Crafted by GPT-Gamma 🤖
