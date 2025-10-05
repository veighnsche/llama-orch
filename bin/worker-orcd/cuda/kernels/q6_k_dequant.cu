// Q6_K Dequantization Kernel
//
// Converts Q6_K quantized weights to FP16 format on GPU.
// Implements GGML Q6_K format specification:
// - Block size: 256 elements
// - Block bytes: 210 bytes
// - Sub-blocks: 16 × 16 elements
//
// Block structure:
// - d: fp16 (2 bytes) - super scale
// - ql: 128 bytes - low 4 bits (2 elements per byte)
// - qh: 64 bytes - two high bitplanes (2 bits per element)
// - scales: 16 bytes - uint8 scale per 16-element sub-block
//
// Dequantization formula:
//   q6_u = low4 | (hi2 << 4)  // unsigned [0..63]
//   q6_s = q6_u - 32          // signed [-32..31]
//   scale_g = d * scales[g]
//   y[i] = scale_g * q6_s
//
// Performance: Each thread processes one element with coalesced reads.
//
// Spec: Based on GGML Q6_K format
// Story: Performance optimization - move dequant to GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Constants
constexpr int Q6K_BLOCK_SIZE = 256;
constexpr int Q6K_BLOCK_BYTES = 210;
constexpr int Q6K_NUM_SUB_BLOCKS = 16;
constexpr int Q6K_SUB_BLOCK_SIZE = 16;

// Q6_K block structure (256 elements = 210 bytes)
// Matches Rust implementation for compatibility
struct Q6KBlock {
    uint16_t d;           // fp16 super scale (2 bytes)
    uint8_t ql[128];      // low 4 bits (128 bytes)
    uint8_t qh[64];       // two high bitplanes (64 bytes)
    uint8_t scales[16];   // sub-block scales (16 bytes)
} __attribute__((packed));

// Get low 4 bits for element at (sub-block g, position t)
__device__ __forceinline__ uint8_t get_low4(const uint8_t* ql, int g, int t) {
    int idx = g * 8 + (t >> 1);
    if ((t & 1) == 0) {
        return ql[idx] & 0x0F;
    } else {
        return (ql[idx] >> 4) & 0x0F;
    }
}

// Get high 2 bits for element at (sub-block g, position t)
__device__ __forceinline__ uint8_t get_hi2(const uint8_t* qh, int g, int t) {
    int base = g * 4 + (t & 7);
    int shift = t >> 3;  // 0 or 1
    
    uint8_t b0 = (qh[base] >> (shift + 0)) & 0x1;
    uint8_t b1 = (qh[base] >> (shift + 2)) & 0x1;
    
    return b0 | (b1 << 1);
}

// Q6_K dequantization kernel
//
// Each thread processes one element from a 256-element block.
// Threads are organized in blocks of 256 to match the Q6_K block size.
// This ensures coalesced memory access and efficient GPU utilization.
//
// Grid: (num_blocks, 1, 1)
// Block: (256, 1, 1)
//
// Args:
//   output: Output FP16 tensor [num_blocks * 256]
//   input: Input Q6_K data [num_blocks * 210 bytes]
//   num_blocks: Number of Q6_K blocks
__global__ void q6k_dequant_kernel(
    half* output,
    const uint8_t* input,
    int num_blocks
) {
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;  // 0-255
    
    if (block_idx >= num_blocks) {
        return;
    }
    
    // Calculate which sub-block (g) and position (t) this thread handles
    int g = thread_idx / Q6K_SUB_BLOCK_SIZE;  // 0-15
    int t = thread_idx % Q6K_SUB_BLOCK_SIZE;  // 0-15
    
    // Load block structure
    const Q6KBlock* block = reinterpret_cast<const Q6KBlock*>(
        input + block_idx * Q6K_BLOCK_BYTES
    );
    
    // Load super scale (fp16 -> float)
    half d_half = __ushort_as_half(block->d);
    float d = __half2float(d_half);
    
    // Load sub-block scale
    float scale_g = d * static_cast<float>(block->scales[g]);
    
    // Get low 4 bits and high 2 bits for this element
    uint8_t low4 = get_low4(block->ql, g, t);
    uint8_t hi2 = get_hi2(block->qh, g, t);
    
    // Combine to 6-bit unsigned [0..63]
    uint8_t q6_u = low4 | (hi2 << 4);
    
    // Convert to signed [-32..31]
    int q6_s = static_cast<int>(q6_u) - 32;
    
    // Dequantize: y = scale_g * q6_s
    float result = scale_g * static_cast<float>(q6_s);
    
    // Write to output (coalesced access)
    int output_idx = block_idx * Q6K_BLOCK_SIZE + thread_idx;
    output[output_idx] = __float2half(result);
}

// Host function to launch Q6_K dequantization
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q6_K data [num_blocks * 210 bytes]
//   num_elements: Total number of elements (must be multiple of 256)
//   stream: CUDA stream for async execution
//
// Returns:
//   cudaSuccess on success, error code otherwise
extern "C" cudaError_t q6k_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
) {
    // Validate input
    if (num_elements % Q6K_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }
    
    int num_blocks = num_elements / Q6K_BLOCK_SIZE;
    
    // Launch kernel: 1 thread per element, 256 threads per block
    dim3 grid(num_blocks, 1, 1);
    dim3 block(Q6K_BLOCK_SIZE, 1, 1);
    
    q6k_dequant_kernel<<<grid, block, 0, stream>>>(
        output,
        input,
        num_blocks
    );
    
    return cudaGetLastError();
}

// Synchronous version for testing
extern "C" cudaError_t q6k_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
) {
    cudaError_t err = q6k_dequant_launch(output, input, num_elements, 0);
    if (err != cudaSuccess) {
        return err;
    }
    return cudaDeviceSynchronize();
}
