// swiglu.cu — SwiGLU Feed-Forward Network - LT-017
//
// SwiGLU: output = silu(gate) * up
//
// Spec: M0-W-1217
//
// ============================================================================
// [TEAM_CHARLIE_BETA]  BUG FOUND - NOT IN THIS FILE! (2025-10-06 17:07 UTC)
// ============================================================================
//  THIS KERNEL IS CORRECT!
//
// SYMPTOM: Model generates repetitive tokens (e.g., "alcoholic" 100+ times)
//
// INVESTIGATION RESULT:
// This SwiGLU activation kernel is implemented correctly:
//  SiLU formula: x * sigmoid(x) = x / (1 + exp(-x)) - CORRECT
//  Element-wise multiply: silu(gate) * up - CORRECT
//  Vectorized implementation using half2 - CORRECT
//
// ROOT CAUSE (found in different file):
// The bug was in qwen_weight_loader.cpp - the ffn_down weight was never loaded!
// This caused the down projection to use uninitialized memory, making FFN
// output garbage even though this activation kernel worked correctly.
//
// THE FIX (in qwen_weight_loader.cpp:327):
//   layer.ffn_down = get_ptr(prefix + "ffn_down.weight");
//
// This kernel is CORRECT. The bug was in weight loading!
// See: investigation-teams/TEAM_CHARLIE_BETA_ROOT_CAUSE.md
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

/**
 * SiLU (Swish) activation with element-wise multiply
 * 
 * Formula:
 *   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *   output = silu(gate) * up
 */
__global__ void swiglu_activation_kernel(
    half* output,
    const half* gate,
    const half* up,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        
        // SiLU: g * sigmoid(g)
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sigmoid_g;
        
        // Element-wise multiply
        output[idx] = __float2half(silu_g * u);
    }
}

/**
 * Vectorized SwiGLU activation using half2
 */
__global__ void swiglu_activation_kernel_vectorized(
    half2* output,
    const half2* gate,
    const half2* up,
    int total_elements_half2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements_half2) {
        half2 g = gate[idx];
        half2 u = up[idx];
        
        float g0 = __half2float(__low2half(g));
        float g1 = __half2float(__high2half(g));
        float u0 = __half2float(__low2half(u));
        float u1 = __half2float(__high2half(u));
        
        // SiLU for both elements
        float sigmoid_g0 = 1.0f / (1.0f + expf(-g0));
        float sigmoid_g1 = 1.0f / (1.0f + expf(-g1));
        float silu_g0 = g0 * sigmoid_g0;
        float silu_g1 = g1 * sigmoid_g1;
        
        // Element-wise multiply
        half2 result = __halves2half2(
            __float2half(silu_g0 * u0),
            __float2half(silu_g1 * u1)
        );
        
        output[idx] = result;
    }
}

extern "C" {

/**
 * SwiGLU activation (fused SiLU + element-wise multiply)
 * 
 * @param output Output tensor [batch, seq_len, ffn_dim]
 * @param gate Gate projection [batch, seq_len, ffn_dim]
 * @param up Up projection [batch, seq_len, ffn_dim]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param ffn_dim FFN intermediate dimension
 * @return 0 on success, error code on failure
 */
int cuda_swiglu_activation(
    half* output,
    const half* gate,
    const half* up,
    int batch_size,
    int seq_len,
    int ffn_dim
) {
    // Validate dimensions
    if (batch_size <= 0 || seq_len <= 0 || ffn_dim <= 0) {
        fprintf(stderr, "SwiGLU: Invalid dimensions\n");
        return -1;
    }
    
    int total_elements = batch_size * seq_len * ffn_dim;
    
    // Use vectorized kernel if ffn_dim is even
    if (ffn_dim % 2 == 0) {
        int total_half2 = total_elements / 2;
        int threads = 256;
        int blocks = (total_half2 + threads - 1) / threads;
        
        swiglu_activation_kernel_vectorized<<<blocks, threads>>>(
            (half2*)output,
            (const half2*)gate,
            (const half2*)up,
            total_half2
        );
    } else {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        swiglu_activation_kernel<<<blocks, threads>>>(
            output, gate, up, total_elements
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "SwiGLU activation kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

} // extern "C"
