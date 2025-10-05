// MXFP4 Dequantization Kernel
//
// Implements software dequantization for MXFP4 (Microscaling FP4) format.
// MXFP4: 32 FP4 values + 1 FP8 scale = 17 bytes per block
//
// Dequantization: fp16 = fp4_mantissa * fp8_scale
//
// Spec: M0-W-1201, M0-W-1820
// Story: GT-029

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP4 mantissa lookup table
// Maps 4-bit values (0-15) to normalized float values
__device__ __constant__ float FP4_MANTISSA_TABLE[16] = {
    0.0f,   // 0000
    0.5f,   // 0001
    1.0f,   // 0010
    1.5f,   // 0011
    2.0f,   // 0100
    2.5f,   // 0101
    3.0f,   // 0110
    3.5f,   // 0111
    -0.0f,  // 1000 (negative zero)
    -0.5f,  // 1001
    -1.0f,  // 1010
    -1.5f,  // 1011
    -2.0f,  // 1100
    -2.5f,  // 1101
    -3.0f,  // 1110
    -3.5f   // 1111
};

// Convert FP8 E8M0 scale to float
// E8M0: 8-bit exponent, no mantissa
// Value = 2^(exponent - 127)
__device__ __forceinline__ float fp8_e8m0_to_float(uint8_t fp8) {
    if (fp8 == 0x00) return 0.0f;
    if (fp8 == 0xFF) return INFINITY;
    
    int exponent = (int)fp8 - 127;
    return powf(2.0f, (float)exponent);
}

// Dequantize a single MXFP4 block (32 elements)
//
// Block layout (17 bytes):
// - Bytes 0-15: 32 FP4 values (2 per byte, 16 bytes total)
// - Byte 16: FP8 E8M0 scale factor
__device__ void dequant_mxfp4_block(
    half* output,
    const uint8_t* block_data,
    int block_idx
) {
    // Load scale factor (last byte of block)
    uint8_t fp8_scale = block_data[16];
    float scale = fp8_e8m0_to_float(fp8_scale);
    
    // Dequantize 32 FP4 values
    for (int i = 0; i < 32; i++) {
        // Get byte containing this FP4 value (2 FP4 per byte)
        int byte_idx = i / 2;
        uint8_t packed_byte = block_data[byte_idx];
        
        // Extract 4-bit mantissa (low nibble for even i, high nibble for odd i)
        uint8_t fp4_mantissa;
        if (i % 2 == 0) {
            fp4_mantissa = packed_byte & 0x0F;  // Low nibble
        } else {
            fp4_mantissa = (packed_byte >> 4) & 0x0F;  // High nibble
        }
        
        // Dequantize: fp16 = fp4_mantissa * scale
        float mantissa_val = FP4_MANTISSA_TABLE[fp4_mantissa];
        float result = mantissa_val * scale;
        
        output[i] = __float2half(result);
    }
}

// MXFP4 dequantization kernel
//
// Dequantizes MXFP4 blocks to FP16 tensors.
// Each block processes 32 elements (17 bytes input, 64 bytes output).
//
// Args:
//   output: Output FP16 tensor [num_elements]
//   input: Input MXFP4 data [num_blocks * 17 bytes]
//   num_blocks: Number of MXFP4 blocks
__global__ void mxfp4_dequant_kernel(
    half* output,
    const uint8_t* input,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_idx < num_blocks) {
        // Input: 17 bytes per block
        const uint8_t* block_data = input + block_idx * 17;
        
        // Output: 32 FP16 values per block
        half* block_output = output + block_idx * 32;
        
        // Dequantize this block
        dequant_mxfp4_block(block_output, block_data, block_idx);
    }
}

// Optimized MXFP4 dequantization using shared memory
//
// Each block cooperatively dequantizes multiple MXFP4 blocks.
// Uses shared memory to reduce global memory traffic.
__global__ void mxfp4_dequant_kernel_shared(
    half* output,
    const uint8_t* input,
    int num_blocks
) {
    // Shared memory for block data
    __shared__ uint8_t shared_block[17 * 32];  // 32 blocks worth
    
    int blocks_per_grid = blockDim.x;
    int base_block_idx = blockIdx.x * blocks_per_grid;
    int thread_block_idx = base_block_idx + threadIdx.x;
    
    // Cooperatively load block data to shared memory
    if (thread_block_idx < num_blocks) {
        const uint8_t* src = input + thread_block_idx * 17;
        uint8_t* dst = shared_block + threadIdx.x * 17;
        
        for (int i = 0; i < 17; i++) {
            dst[i] = src[i];
        }
    }
    __syncthreads();
    
    // Dequantize
    if (thread_block_idx < num_blocks) {
        const uint8_t* block_data = shared_block + threadIdx.x * 17;
        half* block_output = output + thread_block_idx * 32;
        
        dequant_mxfp4_block(block_output, block_data, thread_block_idx);
    }
}

// Host function to launch MXFP4 dequantization
extern "C" void cuda_mxfp4_dequant(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
) {
    // Calculate number of blocks (round up)
    int num_blocks = (num_elements + 31) / 32;
    
    // Launch configuration
    int threads_per_block = 256;
    int blocks_per_grid = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    mxfp4_dequant_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output, input, num_blocks
    );
}

// Optimized version using shared memory
extern "C" void cuda_mxfp4_dequant_optimized(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
) {
    int num_blocks = (num_elements + 31) / 32;
    
    // Use 32 threads per block (one per MXFP4 block)
    int threads_per_block = 32;
    int blocks_per_grid = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    mxfp4_dequant_kernel_shared<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output, input, num_blocks
    );
}

// Calculate MXFP4 storage size for given number of elements
extern "C" size_t cuda_mxfp4_storage_size(int num_elements) {
    int num_blocks = (num_elements + 31) / 32;
    return num_blocks * 17;  // 17 bytes per block
}

// Validate MXFP4 block data
extern "C" bool cuda_mxfp4_validate_block(
    const uint8_t* block_data,
    int block_size
) {
    if (block_size != 17) {
        return false;
    }
    
    // Check scale factor is valid (not reserved values)
    uint8_t scale = block_data[16];
    if (scale == 0xFF) {
        // Infinity/NaN - may be invalid depending on use case
        return false;
    }
    
    return true;
}

// Batch dequantization for weight tensors
//
// Dequantizes multiple weight tensors in sequence.
// Useful for loading model weights.
extern "C" void cuda_mxfp4_dequant_batch(
    half** outputs,
    const uint8_t** inputs,
    const int* num_elements,
    int num_tensors,
    cudaStream_t stream
) {
    for (int i = 0; i < num_tensors; i++) {
        cuda_mxfp4_dequant(outputs[i], inputs[i], num_elements[i], stream);
    }
}

// In-place dequantization (output buffer must be large enough)
//
// WARNING: This requires output buffer to be at least 3.76x larger than input.
// Not recommended for production use due to memory waste.
extern "C" void cuda_mxfp4_dequant_inplace(
    void* buffer,
    int num_elements,
    cudaStream_t stream
) {
    // Cast buffer as both input and output
    const uint8_t* input = (const uint8_t*)buffer;
    half* output = (half*)buffer;
    
    // Dequantize (output will overwrite input)
    cuda_mxfp4_dequant(output, input, num_elements, stream);
}

// ---
// Crafted by GPT-Gamma 🤖
