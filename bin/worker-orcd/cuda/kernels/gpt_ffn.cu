// GPT Feed-Forward Network (FFN) kernel
//
// Implements standard FFN with two linear projections and GELU activation:
// FFN(x) = down_proj(GELU(up_proj(x)))
//
// This differs from Llama's SwiGLU FFN which uses gated activation.
//
// Spec: M0-W-1433, M0-W-1215
// Story: GT-014

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Forward declarations of GELU kernel
extern "C" void cuda_gelu(
    half* output,
    const half* input,
    int size,
    cudaStream_t stream
);

// Add bias kernel (used after GEMM operations)
__global__ void add_bias_kernel(
    half* output,
    const half* bias,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size;
    
    if (idx < total_elements) {
        int hidden_idx = idx % hidden_size;
        output[idx] = __hadd(output[idx], bias[hidden_idx]);
    }
}

// Host function to add bias
extern "C" void cuda_add_bias(
    half* output,
    const half* bias,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = batch_size * seq_len * hidden_size;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    add_bias_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output, bias, batch_size, seq_len, hidden_size
    );
}

// GPT FFN forward pass
//
// Implements: output = W_down @ GELU(W_up @ input + b_up) + b_down
//
// Args:
//   input: Input tensor [batch_size, seq_len, d_model]
//   w_up: Up projection weights [d_model, ffn_dim]
//   b_up: Up projection bias [ffn_dim]
//   w_down: Down projection weights [ffn_dim, d_model]
//   b_down: Down projection bias [d_model]
//   output: Output tensor [batch_size, seq_len, d_model]
//   workspace: Temporary buffer [batch_size, seq_len, ffn_dim]
//   batch_size: Batch size
//   seq_len: Sequence length
//   d_model: Model dimension
//   ffn_dim: FFN intermediate dimension (typically 4 * d_model)
//   cublas_handle: cuBLAS handle for GEMM operations
//   stream: CUDA stream
extern "C" void cuda_gpt_ffn_forward(
    const half* input,
    const half* w_up,
    const half* b_up,
    const half* w_down,
    const half* b_down,
    half* output,
    half* workspace,
    int batch_size,
    int seq_len,
    int d_model,
    int ffn_dim,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    // Set cuBLAS stream
    cublasSetStream(cublas_handle, stream);
    
    // Constants for GEMM
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    int m = batch_size * seq_len;  // Number of input vectors
    int n_up = ffn_dim;             // Up projection output dimension
    int k_up = d_model;             // Up projection input dimension
    
    // Step 1: Up projection - workspace = input @ w_up
    // input: [m, k_up], w_up: [k_up, n_up], workspace: [m, n_up]
    cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,    // w_up is not transposed
        CUBLAS_OP_N,    // input is not transposed
        n_up,           // Number of rows of w_up and workspace
        m,              // Number of columns of input and workspace
        k_up,           // Number of columns of w_up and rows of input
        &alpha,
        w_up, CUDA_R_16F, n_up,
        input, CUDA_R_16F, k_up,
        &beta,
        workspace, CUDA_R_16F, n_up,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // Step 2: Add bias - workspace = workspace + b_up
    if (b_up != nullptr) {
        cuda_add_bias(workspace, b_up, batch_size, seq_len, ffn_dim, stream);
    }
    
    // Step 3: GELU activation - workspace = GELU(workspace)
    int workspace_size = batch_size * seq_len * ffn_dim;
    cuda_gelu(workspace, workspace, workspace_size, stream);
    
    // Step 4: Down projection - output = workspace @ w_down
    // workspace: [m, ffn_dim], w_down: [ffn_dim, d_model], output: [m, d_model]
    int n_down = d_model;
    int k_down = ffn_dim;
    
    cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,    // w_down is not transposed
        CUBLAS_OP_N,    // workspace is not transposed
        n_down,         // Number of rows of w_down and output
        m,              // Number of columns of workspace and output
        k_down,         // Number of columns of w_down and rows of workspace
        &alpha,
        w_down, CUDA_R_16F, n_down,
        workspace, CUDA_R_16F, k_down,
        &beta,
        output, CUDA_R_16F, n_down,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // Step 5: Add bias - output = output + b_down
    if (b_down != nullptr) {
        cuda_add_bias(output, b_down, batch_size, seq_len, d_model, stream);
    }
}

// Fused FFN + Residual kernel
// Computes: output = input + FFN(input)
//
// This is a common pattern in transformers where we add the residual
// connection after the FFN. Fusing saves memory bandwidth.
// NOTE: Temporarily disabled due to linkage issues - use separate calls
// extern "C" void cuda_gpt_ffn_forward_residual(...) { ... }

// Workspace size calculation helper
// Returns the size in bytes needed for the workspace buffer
extern "C" size_t cuda_gpt_ffn_workspace_size(
    int batch_size,
    int seq_len,
    int ffn_dim
) {
    // Workspace needs to hold intermediate activations: [batch_size, seq_len, ffn_dim]
    return batch_size * seq_len * ffn_dim * sizeof(half);
}

// Validate FFN dimensions
extern "C" bool cuda_gpt_ffn_validate_dims(
    int batch_size,
    int seq_len,
    int d_model,
    int ffn_dim
) {
    if (batch_size <= 0 || seq_len <= 0 || d_model <= 0 || ffn_dim <= 0) {
        return false;
    }
    
    // Typical constraint: ffn_dim = 4 * d_model (but not strictly required)
    // Just check reasonable bounds
    if (ffn_dim > 10 * d_model || ffn_dim < d_model / 2) {
        return false;
    }
    
    return true;
}

// ---
// Crafted by GPT-Gamma 🤖
