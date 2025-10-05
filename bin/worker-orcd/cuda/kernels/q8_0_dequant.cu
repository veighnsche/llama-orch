// Q8_0 Dequantization Kernel
//
// Converts Q8_0 quantized weights to FP16 format on GPU.
// Implements GGML Q8_0 format specification:
// - Block size: 32 elements
// - Block bytes: 34 bytes
// - Structure: d (fp16) + qs (32 × int8)
//
// Dequantization formula:
//   y[i] = d * qs[i]
//   where qs[i] ∈ int8 [-128..127]
//
// Performance: Each thread processes one element with coalesced reads.
//
// Spec: Based on GGML Q8_0 format
// Story: Performance optimization - move dequant to GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Constants
constexpr int Q8_0_BLOCK_SIZE = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;

// Q8_0 block structure (32 elements = 34 bytes)
// Matches Rust implementation for compatibility
struct Q8_0Block {
    uint16_t d;       // fp16 scale (2 bytes)
    int8_t qs[32];    // 32 signed 8-bit values (32 bytes)
} __attribute__((packed));

// Q8_0 dequantization kernel
//
// Each thread processes one element from a 32-element block.
// Threads are organized in blocks of 32 to match the Q8_0 block size.
// This ensures coalesced memory access and efficient GPU utilization.
//
// Grid: (num_blocks, 1, 1)
// Block: (32, 1, 1)
//
// Args:
//   output: Output FP16 tensor [num_blocks * 32]
//   input: Input Q8_0 data [num_blocks * 34 bytes]
//   num_blocks: Number of Q8_0 blocks
__global__ void q8_0_dequant_kernel(
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
    const Q8_0Block* block = reinterpret_cast<const Q8_0Block*>(
        input + block_idx * Q8_0_BLOCK_BYTES
    );
    
    // Load scale (fp16 -> float)
    half d_half = __ushort_as_half(block->d);
    float d = __half2float(d_half);
    
    // Load quantized value (signed int8)
    int8_t q = block->qs[thread_idx];
    
    // Dequantize: y = d * q
    float result = d * static_cast<float>(q);
    
    // Write to output (coalesced access)
    int output_idx = block_idx * Q8_0_BLOCK_SIZE + thread_idx;
    output[output_idx] = __float2half(result);
}

// Host function to launch Q8_0 dequantization
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q8_0 data [num_blocks * 34 bytes]
//   num_elements: Total number of elements (must be multiple of 32)
//   stream: CUDA stream for async execution
//
// Returns:
//   cudaSuccess on success, error code otherwise
extern "C" cudaError_t q8_0_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
) {
    // Validate input
    if (num_elements % Q8_0_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }
    
    int num_blocks = num_elements / Q8_0_BLOCK_SIZE;
    
    // Launch kernel: 1 thread per element, 32 threads per block
    dim3 grid(num_blocks, 1, 1);
    dim3 block(Q8_0_BLOCK_SIZE, 1, 1);
    
    q8_0_dequant_kernel<<<grid, block, 0, stream>>>(
        output,
        input,
        num_blocks
    );
    
    return cudaGetLastError();
}

// Synchronous version for testing
extern "C" cudaError_t q8_0_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
) {
    cudaError_t err = q8_0_dequant_launch(output, input, num_elements, 0);
    if (err != cudaSuccess) {
        return err;
    }
    return cudaDeviceSynchronize();
}
