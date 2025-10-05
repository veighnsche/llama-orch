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

// Decode 6-bit scale and min indices from packed 12-byte array
//
// The 12 bytes encode 8 pairs of (scale, min), each 6 bits.
// Based on GGML's get_scale_min_k4 logic.
__device__ void decode_scales_and_mins(const uint8_t* scales, uint8_t* sc, uint8_t* m) {
    // Unpack 6-bit values from the packed byte array
    // This follows the GGML bit-packing scheme
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = ((scales[2] & 0x0F) << 2) | ((scales[0] >> 6) & 0x03);
    sc[3] = ((scales[3] & 0x0F) << 2) | ((scales[1] >> 6) & 0x03);
    sc[4] = ((scales[4] & 0x0F) << 2) | ((scales[2] >> 4) & 0x03);
    sc[5] = ((scales[5] & 0x0F) << 2) | ((scales[3] >> 4) & 0x03);
    sc[6] = ((scales[6] & 0x0F) << 2) | ((scales[4] >> 4) & 0x03);
    sc[7] = ((scales[7] & 0x0F) << 2) | ((scales[5] >> 4) & 0x03);
    
    m[0] = ((scales[8] & 0x0F) << 2) | ((scales[6] >> 4) & 0x03);
    m[1] = ((scales[9] & 0x0F) << 2) | ((scales[7] >> 4) & 0x03);
    m[2] = ((scales[10] & 0x0F) << 2) | ((scales[8] >> 4) & 0x03);
    m[3] = ((scales[11] & 0x0F) << 2) | ((scales[9] >> 4) & 0x03);
    m[4] = (scales[10] >> 4) & 0x0F;
    m[5] = (scales[11] >> 4) & 0x0F;
    m[6] = scales[2] >> 4;
    m[7] = scales[3] >> 4;
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
    float d = __half2float(d_half);
    float dmin = __half2float(dmin_half);
    
    // Decode scale and min indices (shared across threads in same sub-block)
    // Use shared memory to avoid redundant computation
    __shared__ uint8_t sc[Q4K_NUM_SUB_BLOCKS];
    __shared__ uint8_t m[Q4K_NUM_SUB_BLOCKS];
    
    // Only first thread of each sub-block decodes scales
    if (j == 0) {
        decode_scales_and_mins(block->scales, sc, m);
    }
    __syncthreads();
    
    // Calculate scale and min for this sub-block
    float scale = d * static_cast<float>(sc[s]);
    float min_val = dmin * static_cast<float>(m[s]);
    
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
    
    // Dequantize: y = scale * q + min
    float result = scale * static_cast<float>(q) + min_val;
    
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
