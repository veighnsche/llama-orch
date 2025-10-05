// Q5_0 Dequantization Kernel
//
// Converts Q5_0 quantized weights to FP16 format on GPU.
// Implements GGML Q5_0 format specification:
// - Block size: 32 elements
// - Block bytes: 22 bytes
// - Structure: d (fp16) + qh (4 bytes high bits) + qs (16 bytes low nibbles)
//
// Each element is 5-bit signed:
// - Low 4 bits from qs (nibbles)
// - High 1 bit from qh (bitmask)
// - Signed by subtracting 16
//
// Dequantization formula:
//   q5_u = low4 | (high1 << 4)  // unsigned [0..31]
//   q5_s = q5_u - 16            // signed [-16..15]
//   y[i] = d * q5_s
//
// Performance: Each thread processes one element with coalesced reads.
//
// Spec: Based on GGML Q5_0 format
// Story: Performance optimization - move dequant to GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Constants
constexpr int Q5_0_BLOCK_SIZE = 32;
constexpr int Q5_0_BLOCK_BYTES = 22;

// Q5_0 block structure (32 elements = 22 bytes)
// Matches Rust implementation for compatibility
struct Q5_0Block {
    uint16_t d;        // fp16 scale (2 bytes)
    uint8_t qh[4];     // high bits bitmask (4 bytes = 32 bits)
    uint8_t qs[16];    // low 4-bit nibbles (16 bytes = 32 nibbles)
} __attribute__((packed));

// Get low 4 bits for element i
__device__ __forceinline__ uint8_t get_low4(const uint8_t* qs, int i) {
    if (i < 16) {
        return qs[i] & 0x0F;
    } else {
        return (qs[i - 16] >> 4) & 0x0F;
    }
}

// Q5_0 dequantization kernel
//
// Each thread processes one element from a 32-element block.
// Threads are organized in blocks of 32 to match the Q5_0 block size.
// This ensures coalesced memory access and efficient GPU utilization.
//
// Grid: (num_blocks, 1, 1)
// Block: (32, 1, 1)
//
// Args:
//   output: Output FP16 tensor [num_blocks * 32]
//   input: Input Q5_0 data [num_blocks * 22 bytes]
//   num_blocks: Number of Q5_0 blocks
__global__ void q5_0_dequant_kernel(
    half* output,
    const uint8_t* input,
    int num_blocks
) {
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;  // 0-31
    
    if (block_idx >= num_blocks) {
        return;
    }
    
    // Load block structure
    const Q5_0Block* block = reinterpret_cast<const Q5_0Block*>(
        input + block_idx * Q5_0_BLOCK_BYTES
    );
    
    // Load scale (fp16 -> float)
    half d_half = __ushort_as_half(block->d);
    float d = __half2float(d_half);
    
    // Load qh bits as 32-bit value
    uint32_t qh_bits = *reinterpret_cast<const uint32_t*>(block->qh);
    
    // Get low 4 bits for this element
    uint8_t low4 = get_low4(block->qs, thread_idx);
    
    // Get high 1 bit for this element
    uint8_t high1 = (qh_bits >> thread_idx) & 0x1;
    
    // Combine to 5-bit unsigned [0..31]
    uint8_t q5_u = low4 | (high1 << 4);
    
    // Convert to signed [-16..15]
    int q5_s = static_cast<int>(q5_u) - 16;
    
    // Dequantize: y = d * q5_s
    float result = d * static_cast<float>(q5_s);
    
    // Write to output (coalesced access)
    int output_idx = block_idx * Q5_0_BLOCK_SIZE + thread_idx;
    output[output_idx] = __float2half(result);
}

// Host function to launch Q5_0 dequantization
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q5_0 data [num_blocks * 22 bytes]
//   num_elements: Total number of elements (must be multiple of 32)
//   stream: CUDA stream for async execution
//
// Returns:
//   cudaSuccess on success, error code otherwise
extern "C" cudaError_t q5_0_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
) {
    // Validate input
    if (num_elements % Q5_0_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }
    
    int num_blocks = num_elements / Q5_0_BLOCK_SIZE;
    
    // Launch kernel: 1 thread per element, 32 threads per block
    dim3 grid(num_blocks, 1, 1);
    dim3 block(Q5_0_BLOCK_SIZE, 1, 1);
    
    q5_0_dequant_kernel<<<grid, block, 0, stream>>>(
        output,
        input,
        num_blocks
    );
    
    return cudaGetLastError();
}

// Synchronous version for testing
extern "C" cudaError_t q5_0_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
) {
    cudaError_t err = q5_0_dequant_launch(output, input, num_elements, 0);
    if (err != cudaSuccess) {
        return err;
    }
    return cudaDeviceSynchronize();
}
