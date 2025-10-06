// rope.cu — Rotary Position Embedding (RoPE) - LT-014
//
// Implements RoPE for positional encoding in transformer models.
// Used in Llama, Qwen, and other modern LLMs.
//
// Spec: M0-W-1215, M0-W-1431

#include <cuda_runtime.h>
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
    
    // Calculate rotation angle
    float inv_freq = 1.0f / powf(freq_base, (float)dim / (float)rope_dim);
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

    float inv_freq = 1.0f / powf(freq_base, (float)dim / (float)rope_dim);
    float theta = (float)pos * inv_freq;

    float cos_theta, sin_theta;
    sincosf(theta, &sin_theta, &cos_theta);

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
}

} // extern "C"
