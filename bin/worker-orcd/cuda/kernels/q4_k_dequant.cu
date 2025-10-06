// Q4_K Dequantization Kernel
//
// Converts Q4_K quantized weights to FP16 format on GPU.
// Implements GGML Q4_K format specification:
// - Block size: 256 elements
// - Block bytes: 144 bytes
// - Sub-blocks: 8 × 32 elements
//
// Block structure:
// - d: fp16 (2 bytes) - super scale
// - dmin: fp16 (2 bytes) - super min-scale
// - scales: 12 bytes - packed 6-bit indices for 8 sub-blocks
// - qs: 128 bytes - 256 nibbles (4-bit values)
//
// Dequantization formula:
//   scale_s = float(d) * sc_s
//   min_s = float(dmin) * m_s
//   y[i] = scale_s * q[i] + min_s
//   where q[i] ∈ [0..15]
//
// Performance: Each thread processes one element with coalesced reads.
//
// Spec: Based on GGML Q4_K format
// Story: Performance optimization - move dequant to GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Constants
constexpr int Q4K_BLOCK_SIZE = 256;
constexpr int Q4K_BLOCK_BYTES = 144;
constexpr int Q4K_NUM_SUB_BLOCKS = 8;
constexpr int Q4K_SUB_BLOCK_SIZE = 32;

// Q4_K block structure (256 elements = 144 bytes)
// Matches Rust implementation for compatibility
struct Q4KBlock {
    uint16_t d;           // fp16 super scale (2 bytes)
    uint16_t dmin;        // fp16 super min-scale (2 bytes)
    uint8_t scales[12];   // packed 6-bit scale/min indices (12 bytes)
    uint8_t qs[128];      // 256 × 4-bit quantized values (128 bytes)
} __attribute__((packed));

// [TEAM VANGUARD] 2025-10-07T20:20Z - CRITICAL BUG FIX #2
// OLD decode_scales_and_mins was COMPLETELY WRONG!
// Tested against llama.cpp: ALL 8 indices had mismatches
// Example: idx=0 llama(d=60,m=31) vs ours(d=60,m=16) - WRONG!
//
// FIXED: Use llama.cpp's get_scale_min_k4 logic exactly
// Source: llama.cpp/ggml/src/ggml-cuda/convert.cu lines 189-196
//
// Decode 6-bit scale and min for a specific index j (0-7)
// This matches llama.cpp's get_scale_min_k4 function EXACTLY
__device__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Q4_K dequantization kernel
//
// Each thread processes one element from a 256-element block.
// Threads are organized in blocks of 256 to match the Q4_K block size.
// This ensures coalesced memory access and efficient GPU utilization.
//
// Grid: (num_blocks, 1, 1)
// Block: (256, 1, 1)
//
// Args:
//   output: Output FP16 tensor [num_blocks * 256]
//   input: Input Q4_K data [num_blocks * 144 bytes]
//   num_blocks: Number of Q4_K blocks
__global__ void q4k_dequant_kernel(
    half* output,
    const uint8_t* input,
    int num_blocks
) {
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;  // 0-255
    
    if (block_idx >= num_blocks) {
        return;
    }
    
    // Calculate which sub-block (s) and position (j) this thread handles
    int s = thread_idx / Q4K_SUB_BLOCK_SIZE;  // 0-7
    int j = thread_idx % Q4K_SUB_BLOCK_SIZE;  // 0-31
    
    // Load block structure
    const Q4KBlock* block = reinterpret_cast<const Q4KBlock*>(
        input + block_idx * Q4K_BLOCK_BYTES
    );
    
    // Load super scales (fp16 -> float)
    half d_half = __ushort_as_half(block->d);
    half dmin_half = __ushort_as_half(block->dmin);
    float dall = __half2float(d_half);
    float dmin = __half2float(dmin_half);
    
    // [TEAM VANGUARD] 2025-10-07T20:22Z
    // FIXED: Use llama.cpp's get_scale_min_k4 instead of broken decode_scales_and_mins
    // Get scale and min for this sub-block using llama.cpp's exact logic
    uint8_t sc, m;
    get_scale_min_k4(s, block->scales, sc, m);
    
    // Calculate scale and min for this sub-block
    // Formula from llama.cpp: d1 = dall * sc, m1 = dmin * m
    float scale = dall * static_cast<float>(sc);
    float min_val = dmin * static_cast<float>(m);
    
    // Load quantized value (4-bit nibble)
    // Each byte contains 2 nibbles: [high 4 bits | low 4 bits]
    int qs_offset = s * 16 + (j / 2);  // 16 bytes per sub-block
    uint8_t packed = block->qs[qs_offset];
    
    // Extract nibble (low or high 4 bits)
    uint8_t q;
    if (j % 2 == 0) {
        q = packed & 0x0F;  // Low nibble
    } else {
        q = (packed >> 4) & 0x0F;  // High nibble
    }
    
    // [TEAM VANGUARD] 2025-10-07T20:12Z
    // CRITICAL BUG FOUND: Dequantization formula was WRONG!
    // SUSPECT: We used "scale * q + min_val" but llama.cpp uses "scale * q - min_val"
    // PLAN: Compare our formula with llama.cpp/ggml/src/ggml-cuda/convert.cu line 224
    // OBSERVED: llama.cpp: y[l + 0] = d1 * (q[l] & 0xF) - m1;
    // OBSERVED: Our code:  result = scale * q + min_val;
    // CONTRADICTION: Sign is OPPOSITE! Should be MINUS, not PLUS!
    // FIXED: Changed + to - to match llama.cpp
    // 
    // This bug would cause ALL dequantized weights to be shifted incorrectly,
    // explaining why the model outputs garbage despite correct algorithms.
    //
    // Dequantize: y = scale * q - min  (FIXED: was +, now -)
    float result = scale * static_cast<float>(q) - min_val;
    
    // Write to output (coalesced access)
    int output_idx = block_idx * Q4K_BLOCK_SIZE + thread_idx;
    output[output_idx] = __float2half(result);
}

// Host function to launch Q4_K dequantization
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q4_K data [num_blocks * 144 bytes]
//   num_elements: Total number of elements (must be multiple of 256)
//   stream: CUDA stream for async execution
//
// Returns:
//   cudaSuccess on success, error code otherwise
extern "C" cudaError_t q4k_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
) {
    // Validate input
    if (num_elements % Q4K_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }
    
    int num_blocks = num_elements / Q4K_BLOCK_SIZE;
    
    // Launch kernel: 1 thread per element, 256 threads per block
    dim3 grid(num_blocks, 1, 1);
    dim3 block(Q4K_BLOCK_SIZE, 1, 1);
    
    q4k_dequant_kernel<<<grid, block, 0, stream>>>(
        output,
        input,
        num_blocks
    );
    
    return cudaGetLastError();
}

// Synchronous version for testing
extern "C" cudaError_t q4k_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
) {
    cudaError_t err = q4k_dequant_launch(output, input, num_elements, 0);
    if (err != cudaSuccess) {
        return err;
    }
    return cudaDeviceSynchronize();
}
