// GGUF Dequantization Kernels Header
//
// C interface for GGUF quantization format dequantization kernels.
// Provides FFI-safe functions for Rust integration.
//
// Supported formats:
// - Q4_K: 4-bit quantization with 8 sub-blocks (256 elements/block)
// - Q5_0: 5-bit quantization (32 elements/block)
// - Q6_K: 6-bit quantization with 16 sub-blocks (256 elements/block)
// - Q8_0: 8-bit quantization (32 elements/block)
//
// All kernels dequantize to FP16 format for GPU inference.
//
// Spec: Based on GGML quantization formats
// Story: Performance optimization - GPU dequantization

#ifndef GGUF_DEQUANT_CUH
#define GGUF_DEQUANT_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Q4_K Dequantization
//
// Block size: 256 elements
// Block bytes: 144 bytes
// Format: d (fp16) + dmin (fp16) + scales (12B) + qs (128B)
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q4_K data [num_blocks * 144 bytes]
//   num_elements: Total number of elements (must be multiple of 256)
//   stream: CUDA stream for async execution (0 for default stream)
//
// Returns:
//   cudaSuccess on success, error code otherwise
cudaError_t q4k_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
);

// Synchronous Q4_K dequantization (for testing)
cudaError_t q4k_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
);

// Q6_K Dequantization
//
// Block size: 256 elements
// Block bytes: 210 bytes
// Format: d (fp16) + ql (128B) + qh (64B) + scales (16B)
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q6_K data [num_blocks * 210 bytes]
//   num_elements: Total number of elements (must be multiple of 256)
//   stream: CUDA stream for async execution (0 for default stream)
//
// Returns:
//   cudaSuccess on success, error code otherwise
cudaError_t q6k_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
);

// Synchronous Q6_K dequantization (for testing)
cudaError_t q6k_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
);

// Q5_0 Dequantization
//
// Block size: 32 elements
// Block bytes: 22 bytes
// Format: d (fp16) + qh (4B high bits) + qs (16B low nibbles)
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q5_0 data [num_blocks * 22 bytes]
//   num_elements: Total number of elements (must be multiple of 32)
//   stream: CUDA stream for async execution (0 for default stream)
//
// Returns:
//   cudaSuccess on success, error code otherwise
cudaError_t q5_0_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
);

// Synchronous Q5_0 dequantization (for testing)
cudaError_t q5_0_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
);

// Q8_0 Dequantization
//
// Block size: 32 elements
// Block bytes: 34 bytes
// Format: d (fp16) + qs (32 × int8)
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input Q8_0 data [num_blocks * 34 bytes]
//   num_elements: Total number of elements (must be multiple of 32)
//   stream: CUDA stream for async execution (0 for default stream)
//
// Returns:
//   cudaSuccess on success, error code otherwise
cudaError_t q8_0_dequant_launch(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
);

// Synchronous Q8_0 dequantization (for testing)
cudaError_t q8_0_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
);

#ifdef __cplusplus
}
#endif

#endif // GGUF_DEQUANT_CUH
