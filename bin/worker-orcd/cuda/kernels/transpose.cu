// ============================================================================
// [TEAM DICKINSON] 2025-10-08T00:23Z - Matrix Transpose Kernel
// ============================================================================
// PURPOSE: Transpose weight matrices from GGUF column-major to row-major
// 
// HYPOTHESIS: GGUF stores matrices in column-major order (like Fortran/BLAS)
//             but our code assumes row-major order (like C/PyTorch/Candle)
// 
// EVIDENCE:
//   1. gguf_dump.py shows: token_embd.weight = [896, 151936] (hidden × vocab)
//   2. Candle expects: [vocab_size, hidden_size] = [151936, 896]
//   3. Candle source (candle-nn/src/linear.rs:49): `let w = self.weight.t()?;`
//      → Candle transposes weights in EVERY linear layer forward pass!
// 
// IMPLEMENTATION: Transpose at load time (one-time cost, no inference penalty)
// 
// TEST PLAN:
//   1. Apply transpose to ALL weight matrices (embedding, FFN, attention, output)
//   2. Run haiku test
//   3. Check if output is coherent English
// 
// EXPECTED OUTCOMES:
//   SUCCESS: Output is coherent English → This was THE bug! 🎉
//   FAILURE: Output still garbage → Transpose helps but other bugs remain
//            OR transpose hurts → GGUF is actually row-major (unlikely)
// 
// IF THIS FAILS:
//   - Check if GGUF dimensions are correct (maybe [151936, 896] already?)
//   - Check if we need to transpose SOME but not ALL matrices
//   - Check if cuBLAS parameters need adjustment WITH transpose
//   - Document findings for next team to combine with other theories
// 
// See: investigation-teams/ROOT_CAUSE_FOUND.md
//      investigation-teams/GGUF_TRANSPOSE_ANALYSIS.md
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Transpose kernel for FP16 matrices
// Input:  [rows, cols] in column-major order
// Output: [cols, rows] in row-major order (transposed)
__global__ void transpose_fp16_kernel(
    const half* input,
    half* output,
    int rows,
    int cols
) {
    // Each thread handles one element
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // Row index in input
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Col index in input
    
    if (i < rows && j < cols) {
        // input[i, j] in column-major = input[i * cols + j]
        // output[j, i] in row-major = output[j * rows + i]
        output[j * rows + i] = input[i * cols + j];
    }
}

// Optimized transpose using shared memory (for larger matrices)
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_fp16_tiled_kernel(
    const half* input,
    half* output,
    int rows,
    int cols
) {
    __shared__ half tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile from input (column-major)
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * cols + x];
        }
    }
    
    __syncthreads();
    
    // Write transposed tile to output (row-major)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < rows && (y + i) < cols) {
            output[(y + i) * rows + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// Host function to transpose a matrix
extern "C" void cuda_transpose_fp16(
    const half* d_input,
    half* d_output,
    int rows,
    int cols
) {
    // Choose kernel based on matrix size
    if (rows * cols < 1024 * 1024) {
        // Small matrix: use simple kernel
        dim3 block(16, 16);
        dim3 grid((cols + 15) / 16, (rows + 15) / 16);
        transpose_fp16_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    } else {
        // Large matrix: use tiled kernel with shared memory
        dim3 block(TILE_DIM, BLOCK_ROWS);
        dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
        transpose_fp16_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[TRANSPOSE] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Transpose FP32 matrices (for biases, norms, etc.)
__global__ void transpose_fp32_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}

extern "C" void cuda_transpose_fp32(
    const float* d_input,
    float* d_output,
    int rows,
    int cols
) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    transpose_fp32_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[TRANSPOSE] CUDA error: %s\n", cudaGetErrorString(err));
    }
}
