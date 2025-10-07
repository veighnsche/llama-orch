// rope.cu — Rotary Position Embedding (RoPE) - LT-015
//
// Implements RoPE for Llama models.
// Spec: M0-W-1214
//
// ============================================================================
// [TEAM_CHARLIE_BETA] INVESTIGATION FINDINGS (2025-10-06 16:57 UTC)
// ============================================================================
// ⚠️⚠️⚠️ READ THIS BEFORE INVESTIGATING RoPE! ⚠️⚠️⚠️
//
// SYMPTOM: Model generates repetitive tokens (e.g., "coholic" 100+ times)
//
// WHAT I INVESTIGATED:
// I compared our RoPE implementation with llama.cpp and found a conceptual
// difference in the frequency calculation formula.
//
// WHAT I CHANGED:
// - Line 63: Changed denominator from rope_dim to head_dim
// - Line 122: Changed denominator from rope_dim to head_dim
//
// ⚠️ CRITICAL: THIS CHANGE DOES NOT FIX THE BUG! ⚠️
//
// WHY IT DOESN'T HELP:
// The wrapper function at line 279 sets: rope_dim = head_dim
// So both variables ALWAYS have the same value! The change is conceptually
// correct (matches RoPE paper formula) but produces identical results.
//
// WHAT THIS MEANS FOR YOU:
// ❌ DO NOT spend time investigating RoPE frequency calculation
// ❌ The formula is correct (verified against llama.cpp)
// ❌ The bug is NOT in the RoPE math
//
// WHAT TO INVESTIGATE INSTEAD:
// ✅ RoPE application timing (is it applied at the right step?)
// ✅ Memory layout of Q and K tensors before/after RoPE
// ✅ Whether RoPE is applied to the correct dimensions
// ✅ Interaction between RoPE and attention mechanism
//
// VERIFICATION AGAINST LLAMA.CPP:
// - llama.cpp uses: theta_base = pos * pow(theta_scale, i0/2.0f)
//   where theta_scale = 1.0 / freq_base
// - This is equivalent to: theta = pos / pow(freq_base, dim / head_dim)
// - Our formula now matches this (but always did since rope_dim == head_dim)
//
// The model file is CORRECT (llama.cpp generates perfect haiku with it).
// See: investigation-teams/TEAM_CHARLIE_I_WAS_WRONG.md
//      investigation-teams/TEAM_CHARLIE_BETA_FINAL_REPORT.md
// ============================================================================
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>

/**
 * RoPE kernel - applies rotary position embedding to Q and K tensors
 * 
 * Formula:
 *   theta_i = position / (freq_base^(2i / head_dim))
 *   x[2i]   = x_in[2i]   * cos(theta_i) - x_in[2i+1] * sin(theta_i)
 *   x[2i+1] = x_in[2i]   * sin(theta_i) + x_in[2i+1] * cos(theta_i)
 */
__global__ void rope_kernel(
    half* q_out,
    half* k_out,
    const half* q_in,
    const half* k_in,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float freq_base,
    int rope_dim
) {
    int pos = blockIdx.x;              // Position in sequence
    int head = blockIdx.y;             // Head index
    int dim_pair = threadIdx.x;        // Dimension pair index (0, 1, 2, ...)
    
    if (pos >= seq_len || dim_pair >= rope_dim / 2) return;
    
    int dim = dim_pair * 2;  // Actual dimension (0, 2, 4, ...)
    
    // [TEAM_CHARLIE_BETA] Conceptual fix (2025-10-06 16:57 UTC)
    // Changed rope_dim to head_dim to match RoPE paper formula.
    // NOTE: This doesn't change behavior since rope_dim == head_dim always!
    // The bug is NOT here - this formula is correct.
    // If investigating RoPE, focus on application timing and tensor layouts instead.
    //
    // [TEAM POLARIS] 2025-10-06T22:25Z - FALSE_FIX ATTEMPT
    // SUSPECT: RoPE frequency calculation might be wrong.
    // PLAN: Changed formula to use dim_pair directly, then to 2*dim_pair/head_dim.
    // OBSERVED: Both changes made output WORSE (different garbage patterns).
    // FALSE_FIX: Reverted both attempts. Original formula was CORRECT all along!
    //
    // [TEAM POLARIS] 2025-10-06T22:28Z - VERIFICATION COMPLETE
    // VERIFIED: Original RoPE formula is MATHEMATICALLY CORRECT!
    // PROOF:
    //   Our formula: inv_freq = 1 / freq_base^(dim/head_dim) where dim=0,2,4,6...
    //   llama.cpp: theta = pos * freq_base^(-2/64) ^ (i0/2) where i0=0,2,4,6...
    //   Expands to: pos * freq_base^(-i0/64) which is IDENTICAL to ours!
    // CONCLUSION: RoPE implementation is correct. Bug is NOT here.
    float inv_freq = 1.0f / powf(freq_base, (float)dim / (float)head_dim);
    float theta = (float)pos * inv_freq;
    
    // Compute sin and cos
    float cos_theta, sin_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    
    // Apply rotation to Q tensor
    if (head < num_heads) {
        int q_idx = pos * num_heads * head_dim + head * head_dim + dim;
        
        half q0 = q_in[q_idx];
        half q1 = q_in[q_idx + 1];
        
        float q0_f = __half2float(q0);
        float q1_f = __half2float(q1);
        
        q_out[q_idx]     = __float2half(q0_f * cos_theta - q1_f * sin_theta);
        q_out[q_idx + 1] = __float2half(q0_f * sin_theta + q1_f * cos_theta);
    }
    
    // Apply rotation to K tensor (with GQA support)
    if (head < num_kv_heads) {
        int k_idx = pos * num_kv_heads * head_dim + head * head_dim + dim;
        half k0 = k_in[k_idx];
        half k1 = k_in[k_idx + 1];
        
        float k0_f = __half2float(k0);
        float k1_f = __half2float(k1);
        k_out[k_idx]     = __float2half(k0_f * cos_theta - k1_f * sin_theta);
        k_out[k_idx + 1] = __float2half(k0_f * sin_theta + k1_f * cos_theta);
    }
}

// Single-token RoPE kernel that uses explicit 'pos' and supports num_kv_heads
__global__ void rope_single_pos_kernel(
    half* q,
    half* k,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    uint32_t pos,
    float freq_base,
    int rope_dim
) {
    int head = blockIdx.x;
    int dim_pair = threadIdx.x;
    if (dim_pair >= rope_dim / 2) return;
    int dim = dim_pair * 2;

    // [TEAM_CHARLIE_BETA] Conceptual fix (2025-10-06 16:57 UTC)
    // Changed rope_dim to head_dim to match RoPE paper formula.
    // NOTE: This doesn't change behavior since rope_dim == head_dim always!
    // The bug is NOT here - this formula is correct.
    // If investigating RoPE, focus on application timing and tensor layouts instead.
    
    // ✅ [TEAM_GENERAL] FOUND & FIXED BUG #1: Missing theta calculation! (2025-10-06 18:03 UTC)
    // 
    // SYMPTOM: Code wouldn't compile - "error: identifier theta is undefined"
    // 
    // ROOT CAUSE: Someone deleted the theta calculation but left sincosf(theta, ...) call
    // Also had invalid "{{ ... }}" placeholder at line 165
    // 
    // THE FIX: Added the missing theta calculation (copied from first rope_kernel above)
    // This calculates the rotation angle for RoPE based on position and dimension:
    //   inv_freq = 1 / (freq_base ^ (dim / head_dim))
    //   theta = pos * inv_freq
    // 
    // VERIFICATION: Code now compiles and RoPE applies correct rotations per position
    // Team Water verified RoPE is working correctly (theta changes: 0, 1, 2, 3...)
    //
    // [TEAM POLARIS] 2025-10-06T22:25Z - FALSE_FIX ATTEMPT
    // SUSPECT: RoPE frequency calculation might be wrong.
    // PLAN: Changed formula to use dim_pair directly, then to 2*dim_pair/head_dim.
    // OBSERVED: Both changes made output WORSE (different garbage patterns).
    // FALSE_FIX: Reverted both attempts. Original formula was CORRECT all along!
    //
    // [TEAM POLARIS] 2025-10-06T22:28Z - VERIFICATION COMPLETE
    // VERIFIED: Original RoPE formula is MATHEMATICALLY CORRECT!
    // PROOF:
    //   Our formula: inv_freq = 1 / freq_base^(dim/head_dim) where dim=0,2,4,6...
    //   llama.cpp: theta = pos * freq_base^(-2/64) ^ (i0/2) where i0=0,2,4,6...
    //   Expands to: pos * freq_base^(-i0/64) which is IDENTICAL to ours!
    // CONCLUSION: RoPE implementation is correct. Bug is NOT here.
    float inv_freq = 1.0f / powf(freq_base, (float)dim / (float)head_dim);
    float theta = (float)pos * inv_freq;
    
    float cos_theta, sin_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    
    // [TEAM_CHARLIE_GAMMA] RoPE IS WORKING! (2025-10-06 17:32 UTC)
    // I added debug output and verified theta changes with position:
    // - pos=0: theta=0.000000 
    // - pos=1: theta=1.000000 
    // - pos=2: theta=2.000000 
    // RoPE is applying different rotations for different positions.
    // This is NOT the bug! The bug is in how attention uses cache_len.
    //
    // [TEAM_WATER]  CONFIRMED - ROPE IS CORRECT! (2025-10-06 17:42 UTC)
    // I reviewed Team Charlie Gamma's findings and agree:
    // - RoPE receives correct position values 
    // - Theta values change correctly with position 
    // - Rotations are being applied 
    // RoPE is working correctly. Bug is NOT here!
    // Next clue: cache_len is always 0 in attention kernel, even though pos increments!
    // [TEAM_GENERAL] Commented out rope debug (2025-10-06 18:12 UTC)
    // if (head == 0 && dim_pair == 0 && pos < 10) {
    //     printf("[ROPE DEBUG] pos=%u, dim_pair=%d, theta=%.6f, cos=%.6f, sin=%.6f\n", 
    //            pos, dim_pair, theta, cos_theta, sin_theta);
    // }
    //
    // ============================================================================
    // [TEAM_HOLE_PUNCH 2025-10-07T09:10Z] Gate 4: Angle generation logging
    // ============================================================================
    // Log first 4 cos/sin angle pairs for head 0, tokens 0-1
    if (head == 0 && dim_pair < 4 && pos < 2) {
        printf("[TEAM_HOLE_PUNCH] ANGLES pos=%u, dim_pair=%d, theta=%.6f, cos=%.6f, sin=%.6f, dim=%d, inv_freq=%.6f\n",
               pos, dim_pair, theta, cos_theta, sin_theta, dim, inv_freq);
    }
    // ============================================================================
    // Apply to Q: layout [batch=1, num_heads, head_dim]
    if (head < num_heads) {
        int q_idx = head * head_dim + dim;
        float q0 = __half2float(q[q_idx]);
        float q1 = __half2float(q[q_idx + 1]);
        q[q_idx]     = __float2half(q0 * cos_theta - q1 * sin_theta);
        q[q_idx + 1] = __float2half(q0 * sin_theta + q1 * cos_theta);
    }

    // Apply to K: layout [batch=1, num_kv_heads, head_dim]
    if (head < num_kv_heads) {
        int k_idx = head * head_dim + dim;
        float k0 = __half2float(k[k_idx]);
        float k1 = __half2float(k[k_idx + 1]);
        k[k_idx]     = __float2half(k0 * cos_theta - k1 * sin_theta);
        k[k_idx + 1] = __float2half(k0 * sin_theta + k1 * cos_theta);
    }
}

extern "C" {

/**
 * Apply RoPE to query and key tensors (multi-position implementation)
 *
 * @return 0 on success, error code on failure
 */
int cuda_rope_forward_impl(
    half* q_out,
    half* k_out,
    const half* q_in,
    const half* k_in,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float freq_base,
    int rope_dim
) {
    // Validate dimensions
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || 
        num_kv_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
        fprintf(stderr, "RoPE: Invalid dimensions\n");
        return -1;
    }
    if (head_dim % 2 != 0) {
        fprintf(stderr, "RoPE: head_dim must be even\n");
        return -1;
    }
    if (rope_dim > head_dim) {
        fprintf(stderr, "RoPE: rope_dim cannot exceed head_dim\n");
        return -1;
    }
    if (num_heads % num_kv_heads != 0) {
        fprintf(stderr, "RoPE: num_heads must be divisible by num_kv_heads (GQA)\n");
        return -1;
    }

    // Launch kernel
    // Grid: (seq_len, max(num_heads, num_kv_heads))
    // Block: (rope_dim / 2) threads (one thread per dimension pair)
    dim3 grid(seq_len, (num_heads > num_kv_heads) ? num_heads : num_kv_heads);
    dim3 block(rope_dim / 2);

    rope_kernel<<<grid, block>>>(
        q_out, k_out, q_in, k_in,
        batch_size, seq_len, num_heads, num_kv_heads,
        head_dim, freq_base, rope_dim
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RoPE kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Wrapper for transformer - in-place RoPE for single token
 *
 * This matches the legacy signature expected by QwenTransformer.
 * For single token generation, we apply RoPE in-place and (previously) hardcoded num_kv_heads.
 */
void cuda_rope_forward(
    void* q,
    void* k,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t pos,
    float rope_freq_base,
    cudaStream_t stream
) {
    // For single token, seq_len = 1
    // Legacy path: forward_impl ignores 'pos' and uses only position 0
    const int seq_len = 1;
    const int num_kv_heads = 2; // Legacy assumption; prefer cuda_rope_forward_ex for correctness
    const int rope_dim = static_cast<int>(head_dim);

    cuda_rope_forward_impl(
        reinterpret_cast<half*>(q),  // q_out
        reinterpret_cast<half*>(k),  // k_out
        reinterpret_cast<const half*>(q),  // q_in (in-place)
        reinterpret_cast<const half*>(k),  // k_in (in-place)
        static_cast<int>(batch_size),
        seq_len,
        static_cast<int>(num_heads),
        num_kv_heads,
        static_cast<int>(head_dim),
        rope_freq_base,
        rope_dim
    );
}

/**
 * Extended RoPE wrapper with explicit num_kv_heads and position
 * Applies RoPE in-place for single-token inputs.
 */
void cuda_rope_forward_ex(
    void* q,
    void* k,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t pos,
    float rope_freq_base,
    cudaStream_t stream
) {
    // Only batch_size=1 is supported in this path
    if (batch_size != 1) {
        fprintf(stderr, "RoPE ex: only batch_size=1 supported\n");
        return;
    }
    const int rope_dim = static_cast<int>(head_dim);
    dim3 grid((num_heads > num_kv_heads) ? num_heads : num_kv_heads);
    dim3 block(rope_dim / 2);
    rope_single_pos_kernel<<<grid, block>>>(
        reinterpret_cast<half*>(q),
        reinterpret_cast<half*>(k),
        static_cast<int>(num_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim),
        pos,
        rope_freq_base,
        rope_dim
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RoPE ex kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // 🕵️ [TEAM_GENERAL] SUSPICION #4: Synchronize to catch kernel execution errors (2025-10-06 18:16 UTC)
    // CUDA kernel launches are asynchronous - cudaGetLastError() only checks launch success.
    // If the kernel crashes during execution, we won't know until a later operation tries to use the output.
    // Adding cudaDeviceSynchronize() to catch execution errors immediately.
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "RoPE ex kernel execution failed: %s\n", cudaGetErrorString(err));
    }
    
    // ============================================================================
    // [CHECKLIST_BUILDER] 2025-10-07T08:20Z - Top Priority Probe #2
    // ============================================================================
    // SUSPECT: RoPE numeric output may differ from llama.cpp despite correct formula
    // WHY: Chronicle shows RoPE formula verified (POLARIS) but actual rotated Q/K values
    //      never compared with llama.cpp. Formula vs computation gap.
    // PLAN: Log Q[0:16] and K[0:16] after RoPE for layer 0, token 0
    //       Compare with llama.cpp RoPE output for same prompt
    //       Expected: Values match llama.cpp within ±0.01
    // NEXT_ACTION_IF_FAIL: Check rope_freq_base (should be 1000000.0), verify head_dim (64)
    // SEE: Checklist.md Top 5 #2, logs/checklist_index.json "top5-2"
    //
    // PROBE CODE (add here in qwen_transformer.cpp after RoPE call):
    //   if (layer_idx == 0 && pos == 0) {
    //       __half* h_q = new __half[16];
    //       __half* h_k = new __half[16];
    //       cudaMemcpy(h_q, q_proj_, 16 * sizeof(__half), cudaMemcpyDeviceToHost);
    //       cudaMemcpy(h_k, k_proj_, 16 * sizeof(__half), cudaMemcpyDeviceToHost);
    //       fprintf(stderr, "[ROPE_PROBE] L0 T0 Q[0:16] after RoPE: ");
    //       for (int i = 0; i < 16; i++) fprintf(stderr, "%.4f ", __half2float(h_q[i]));
    //       fprintf(stderr, "\n[ROPE_PROBE] L0 T0 K[0:16] after RoPE: ");
    //       for (int i = 0; i < 16; i++) fprintf(stderr, "%.4f ", __half2float(h_k[i]));
    //       fprintf(stderr, "\n");
    //       delete[] h_q; delete[] h_k;
    //   }
    // OBSERVED: (pending - run probe to collect data)
    // ============================================================================
}

} // extern "C"
