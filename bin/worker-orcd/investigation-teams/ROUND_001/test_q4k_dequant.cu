// Test Q4_K Dequantization Against llama.cpp Reference
// Team VANGUARD - Find remaining dequantization bugs

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

// llama.cpp's get_scale_min_k4 implementation
__device__ void llama_get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Our decode_scales_and_mins implementation
__device__ void our_decode_scales_and_mins(const uint8_t* scales, uint8_t* sc, uint8_t* m) {
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

// Test kernel to compare both approaches
__global__ void compare_scale_decoding(const uint8_t* scales) {
    int tid = threadIdx.x;
    if (tid >= 8) return;
    
    // Method 1: llama.cpp's approach
    uint8_t llama_d, llama_m;
    llama_get_scale_min_k4(tid, scales, llama_d, llama_m);
    
    // Method 2: Our approach
    __shared__ uint8_t our_sc[8];
    __shared__ uint8_t our_m[8];
    if (tid == 0) {
        our_decode_scales_and_mins(scales, our_sc, our_m);
    }
    __syncthreads();
    
    uint8_t our_d = our_sc[tid];
    uint8_t our_min = our_m[tid];
    
    // Compare
    if (llama_d != our_d || llama_m != our_min) {
        printf("[VANGUARD MISMATCH] idx=%d: llama(d=%u, m=%u) vs ours(d=%u, m=%u)\n",
               tid, llama_d, llama_m, our_d, our_min);
    } else {
        printf("[VANGUARD OK] idx=%d: d=%u, m=%u\n", tid, llama_d, llama_m);
    }
}

int main() {
    printf("[VANGUARD] Q4_K Scale/Min Decoding Test\n\n");
    
    // Test data: typical scales array from a Q4_K block
    uint8_t h_scales[12] = {
        0x3C, 0x2A, 0x15, 0x28,  // scales[0-3]
        0x1F, 0x32, 0x0D, 0x19,  // scales[4-7]
        0x24, 0x1B, 0x2E, 0x17   // scales[8-11]
    };
    
    // Copy to device
    uint8_t* d_scales;
    cudaMalloc(&d_scales, 12);
    cudaMemcpy(d_scales, h_scales, 12, cudaMemcpyHostToDevice);
    
    // Run comparison
    compare_scale_decoding<<<1, 8>>>(d_scales);
    cudaDeviceSynchronize();
    
    cudaFree(d_scales);
    
    printf("\n[VANGUARD] If all OK, our decoding matches llama.cpp\n");
    printf("[VANGUARD] If MISMATCH, we have the bug location!\n");
    
    return 0;
}

// Compile: nvcc -o test_q4k test_q4k_dequant.cu
// Run: ./test_q4k
