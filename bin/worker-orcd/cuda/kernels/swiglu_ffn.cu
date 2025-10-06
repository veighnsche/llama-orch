// swiglu_ffn.cu — Full SwiGLU Feed-Forward Network
//
// Implements complete SwiGLU FFN with gate/up/down projections
// FFN(x) = down(silu(gate(x)) * up(x))
//
// Spec: M0-W-1217
//
// ============================================================================
// [TEAM_CHARLIE_BETA] 🔥 ROOT CAUSE FOUND! (2025-10-06 17:07 UTC)
// ============================================================================
// ✅ BUG WAS FOUND - IT WASN'T IN THIS FILE!
//
// SYMPTOM: Model generates repetitive tokens (e.g., "coholic" 100+ times)
//
// INVESTIGATION RESULT:
// This FFN implementation is CORRECT. The bug was in the weight loader!
//
// ROOT CAUSE:
// In qwen_weight_loader.cpp, the load_from_gpu_pointers() function was
// missing the line to load ffn_down weights. This caused the down projection
// (line 144-158 below) to use UNINITIALIZED MEMORY!
//
// THE FIX:
// Added missing line in qwen_weight_loader.cpp:327:
//   layer.ffn_down = get_ptr(prefix + "ffn_down.weight");
//
// WHY THIS CAUSED REPETITIVE TOKENS:
// 1. FFN gate and up projections worked (weights loaded correctly)
// 2. SwiGLU activation worked (silu(gate) * up)
// 3. Down projection FAILED (used garbage memory instead of real weights)
// 4. FFN output was garbage
// 5. Garbage accumulated through residual connections across 24 layers
// 6. Final logits became dominated by noise
// 7. Model generated repetitive tokens
//
// This kernel implementation is CORRECT. The bug was in weight loading!
// See: investigation-teams/TEAM_CHARLIE_BETA_ROOT_CAUSE.md
// ============================================================================

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
    
    // [TEAM FELICIA] 2025-10-06T21:57Z
    // SUSPECT: FFN projections might use wrong cuBLAS parameters.
    // HYPOTHESIS: Should use CUBLAS_OP_T like llama.cpp does.
    // TESTED: Changed all 3 FFN projections to CUBLAS_OP_T.
    // RESULT: Made output WORSE (random garbage → stuck repetition).
    // FALSE_FIX: Reverted. CUBLAS_OP_N is correct for our weight layout.
    //
    // 1. Gate projection: gate_out = gate_weight @ input
    //    gate_weight in GGUF: [hidden_dim, ffn_dim] row-major → [ffn_dim, hidden_dim] col-major
    //    input: [hidden_dim, batch] col-major
    //    gate_out: [ffn_dim, batch] col-major
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,  // No transpose needed (row-major → col-major)
        CUBLAS_OP_N,  // No transpose input
        ffn_dim,      // M
        batch_size,   // N
        hidden_dim,   // K
        &alpha,
        gate_weight_half, CUDA_R_16F, ffn_dim,  // lda = ffn_dim
        input_half, CUDA_R_16F, hidden_dim,
        &beta,
        gate_out, CUDA_R_16F, ffn_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // 2. Up projection: up_out = up_weight @ input
    //    up_weight in GGUF: [hidden_dim, ffn_dim] row-major → [ffn_dim, hidden_dim] col-major
    status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,  // No transpose needed
        CUBLAS_OP_N,
        ffn_dim,
        batch_size,
        hidden_dim,
        &alpha,
        up_weight_half, CUDA_R_16F, ffn_dim,  // lda = ffn_dim
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
    
    // 4. Down projection: output = down_weight @ swiglu_out
    //    down_weight in GGUF: [ffn_dim, hidden_dim] row-major → [hidden_dim, ffn_dim] col-major
    //    swiglu_out: [ffn_dim, batch] col-major
    //    output: [hidden_dim, batch] col-major
    status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N,  // No transpose needed
        CUBLAS_OP_N,
        hidden_dim,
        batch_size,
        ffn_dim,
        &alpha,
        down_weight_half, CUDA_R_16F, hidden_dim,  // lda = hidden_dim
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
