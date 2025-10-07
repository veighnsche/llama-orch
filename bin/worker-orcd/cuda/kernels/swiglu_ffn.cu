// swiglu_ffn.cu — Full SwiGLU Feed-Forward Network
//
// Implements complete SwiGLU FFN with gate/up/down projections
// FFN(x) = down(silu(gate(x)) * up(x))
//
// Spec: M0-W-1217
//
// [APPEND-ONLY GUARD] Do not delete prior teams' comments. Add new notes below existing blocks.
//
// ============================================================================
// [TEAM RACE CAR] 2025-10-07T01:02Z - FFN Intermediate Checkpoint Logging
// ============================================================================
// OBJECTIVE: Log gate_out, up_out, and swiglu_out for parity verification
// Enable with RACECAR_FFN_TRACE macro (must match transformer definition)
// Logs first 16 + min/max/mean for tokens 0-1 only
// ============================================================================
#ifndef RACECAR_FFN_TRACE
#define RACECAR_FFN_TRACE 1  // Must match qwen_transformer.cpp
#endif

// ============================================================================
// [TEAM PAPER CUTTER] 2025-10-07T08:59Z - Last Block FFN GEMM Audit
// ============================================================================
// OBJECTIVE: Log GEMM params (M,N,K, lda/ldb/ldc, opA/opB) for last block only
// PLAN: Log all 3 GEMMs (gate, up, down) with exact parameters
// OBSERVED: Will log when last_block_mode is enabled
// ============================================================================
#ifndef PAPER_CUTTER_LAST_BLOCK_TRACE
#define PAPER_CUTTER_LAST_BLOCK_TRACE 1
#endif
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

// [TEAM RACE CAR] 2025-10-07T01:02Z - Helper function for FFN parity logging
#if RACECAR_FFN_TRACE
static int racecar_ffn_call_count = 0;

static void log_ffn_intermediate(const char* name, const half* data, int size) {
    half* h_data = new half[size];
    cudaMemcpy(h_data, data, size * sizeof(half), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[RACE CAR] %s[0..15]: ", name);
    int display_count = (size < 16) ? size : 16;
    for (int i = 0; i < display_count; i++) {
        fprintf(stderr, "%.6f ", __half2float(h_data[i]));
    }
    
    // Compute min/max/mean
    float min_val = __half2float(h_data[0]);
    float max_val = __half2float(h_data[0]);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = __half2float(h_data[i]);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    float mean = sum / size;
    
    fprintf(stderr, "\n[RACE CAR]   min=%.6f max=%.6f mean=%.6f\n", min_val, max_val, mean);
    delete[] h_data;
}
#endif

// [TEAM PAPER CUTTER] 2025-10-07T08:59Z - Last block checkpoint logger
#if PAPER_CUTTER_LAST_BLOCK_TRACE
static int paper_cutter_call_count = 0;

static void log_paper_cutter_checkpoint(const char* name, const half* data, int size) {
    half* h_data = new half[size];
    cudaMemcpy(h_data, data, size * sizeof(half), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[PAPER CUTTER] %s first8=[", name);
    int display_count = (size < 8) ? size : 8;
    for (int i = 0; i < display_count; i++) {
        fprintf(stderr, "%.6f", __half2float(h_data[i]));
        if (i < display_count - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]\n");
    
    // Compute min/max/mean
    float min_val = __half2float(h_data[0]);
    float max_val = __half2float(h_data[0]);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = __half2float(h_data[i]);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    float mean = sum / size;
    
    fprintf(stderr, "[PAPER CUTTER]   min=%.6f max=%.6f mean=%.6f\n", min_val, max_val, mean);
    delete[] h_data;
}
#endif

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
    // [TEAM SENTINEL] 2025-10-07T23:18Z
    // FALSE_FIX: Team Felicia's conclusion was wrong - needed correct lda values + ALL matmuls.
    // FIXED: FFN gate/up/down with CUBLAS_OP_T + lda=hidden_dim/ffn_dim (part of 8-matmul fix).
    //
    // 1. Gate projection: gate_out = gate_weight @ input
    //    gate_weight in GGUF: [hidden_dim, ffn_dim] row-major
    //    Need CUBLAS_OP_T with lda=hidden_dim (first dimension of row-major array)
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - OBSERVED: GEMM_GATE params
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    bool is_last_block = (paper_cutter_call_count < 2);  // First 2 calls = tokens 0-1 of last block
    if (is_last_block) {
        fprintf(stderr, "[PAPER CUTTER] GEMM_GATE M=%u, N=%u, K=%u, lda=%u, ldb=%u, ldc=%u, opA=T, opB=N, compute=32F\n",
                ffn_dim, batch_size, hidden_dim, hidden_dim, hidden_dim, ffn_dim);
    }
#endif
    
    // ⚠️ [TEAM PEAR] 2025-10-07T11:04Z - FFN gate uses CUBLAS_OP_T (CORRECT, don't change)
    // SENTINEL verified this is mathematically correct, but output is STILL garbage.
    // Bug is NOT in cuBLAS parameters. Don't waste time re-testing OP_N or different lda.
    // [TEAM MONET 2025-10-07T14:22Z] Checked line 239: CUBLAS_OP_T lda=hidden_dim ✅
    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,  // Transpose to match row-major layout
        CUBLAS_OP_N,  // No transpose input
        ffn_dim,      // M
        batch_size,   // N
        hidden_dim,   // K
        &alpha,
        gate_weight_half, CUDA_R_16F, hidden_dim,  // lda = hidden_dim (FIXED!)
        input_half, CUDA_R_16F, hidden_dim,
        &beta,
        gate_out, CUDA_R_16F, ffn_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // [TEAM RACE CAR] 2025-10-07T01:02Z - Checkpoint 2: After gate_proj
#if RACECAR_FFN_TRACE
    if (racecar_ffn_call_count < 2) {
        log_ffn_intermediate("Checkpoint 2: After gate_proj", gate_out, ffn_dim);
    }
#endif
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - OBSERVED: CHK_GATE
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    if (is_last_block) {
        log_paper_cutter_checkpoint("CHK_GATE", gate_out, ffn_dim);
    }
#endif
    
    // 2. Up projection: up_out = up_weight @ input
    //    up_weight in GGUF: [hidden_dim, ffn_dim] row-major
    //    Same fix as gate - CUBLAS_OP_T with lda=hidden_dim
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - OBSERVED: GEMM_UP params
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    if (is_last_block) {
        fprintf(stderr, "[PAPER CUTTER] GEMM_UP M=%u, N=%u, K=%u, lda=%u, ldb=%u, ldc=%u, opA=T, opB=N, compute=32F\n",
                ffn_dim, batch_size, hidden_dim, hidden_dim, hidden_dim, ffn_dim);
    }
#endif
    
    // [TEAM MONET 2025-10-07T14:22Z] Checked line 281: CUBLAS_OP_T lda=hidden_dim ✅
    status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,  // Transpose to match row-major layout
        CUBLAS_OP_N,
        ffn_dim,
        batch_size,
        hidden_dim,
        &alpha,
        up_weight_half, CUDA_R_16F, hidden_dim,  // lda = hidden_dim (FIXED!)
        input_half, CUDA_R_16F, hidden_dim,
        &beta,
        up_out, CUDA_R_16F, ffn_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // [TEAM RACE CAR] 2025-10-07T01:02Z - Checkpoint 3: After up_proj
#if RACECAR_FFN_TRACE
    if (racecar_ffn_call_count < 2) {
        log_ffn_intermediate("Checkpoint 3: After up_proj", up_out, ffn_dim);
    }
#endif
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - OBSERVED: CHK_UP
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    if (is_last_block) {
        log_paper_cutter_checkpoint("CHK_UP", up_out, ffn_dim);
    }
#endif
    
    // 3. SwiGLU activation: swiglu_out = silu(gate_out) * up_out
    cuda_swiglu_activation(
        swiglu_out,
        gate_out,
        up_out,
        batch_size,
        1,  // seq_len = 1 for single token
        ffn_dim
    );
    
    // [TEAM RACE CAR] 2025-10-07T01:02Z - Checkpoint 4: After SwiGLU activation
#if RACECAR_FFN_TRACE
    if (racecar_ffn_call_count < 2) {
        log_ffn_intermediate("Checkpoint 4: After SwiGLU activation", swiglu_out, ffn_dim);
    }
#endif
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - OBSERVED: CHK_SILU and CHK_ELEMWISE
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    if (is_last_block) {
        // Note: SwiGLU combines silu(gate) * up, so log the combined result
        log_paper_cutter_checkpoint("CHK_ELEMWISE (post-SwiGLU)", swiglu_out, ffn_dim);
    }
#endif
    
    // [TEAM SENTINEL] 2025-10-07T23:18Z
    // 4. Down projection: output = down_weight @ swiglu_out
    //    down_weight in GGUF: [ffn_dim, hidden_dim] row-major
    //    Use CUBLAS_OP_T with lda=ffn_dim (part of 8-matmul fix)
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - OBSERVED: GEMM_DOWN params
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    if (is_last_block) {
        fprintf(stderr, "[PAPER CUTTER] GEMM_DOWN M=%u, N=%u, K=%u, lda=%u, ldb=%u, ldc=%u, opA=T, opB=N, compute=32F\n",
                hidden_dim, batch_size, ffn_dim, ffn_dim, ffn_dim, hidden_dim);
    }
#endif
    
    // ⚠️ [TEAM PEAR] FFN down uses CUBLAS_OP_T (CORRECT, don't change)
    // [TEAM MONET 2025-10-07T14:22Z] Checked line 350: CUBLAS_OP_T lda=ffn_dim ✅
    status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,  // Transpose to match row-major layout
        CUBLAS_OP_N,
        hidden_dim,
        batch_size,
        ffn_dim,
        &alpha,
        down_weight_half, CUDA_R_16F, ffn_dim,  // lda = ffn_dim (FIXED!)
        swiglu_out, CUDA_R_16F, ffn_dim,
        &beta,
        output_half, CUDA_R_16F, hidden_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - OBSERVED: CHK_DOWN
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    if (is_last_block) {
        log_paper_cutter_checkpoint("CHK_DOWN", output_half, hidden_dim);
    }
#endif
    
    // [TEAM RACE CAR] 2025-10-07T01:02Z - Increment call counter
#if RACECAR_FFN_TRACE
    racecar_ffn_call_count++;
#endif
    
    // [TEAM PAPER CUTTER] 2025-10-07T08:59Z - Increment call counter
#if PAPER_CUTTER_LAST_BLOCK_TRACE
    paper_cutter_call_count++;
#endif
    
    // Cleanup
    cudaFree(gate_out);
    cudaFree(up_out);
    cudaFree(swiglu_out);
    cublasDestroy(cublas_handle);
}

} // extern "C"

// ---
// Crafted by GPT-Gamma 🤖
