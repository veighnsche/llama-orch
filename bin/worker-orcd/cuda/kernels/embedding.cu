/**
 * Embedding Lookup Kernel
 * 
 * Implements token embedding retrieval from weight matrix.
 * This is the first layer of transformer inference, shared across all architectures.
 * 
 * Features:
 * - Coalesced memory access for optimal GPU performance
 * - Bounds checking for invalid token IDs
 * - Support for FP16 and FP32 precision
 * - Handles arbitrary hidden dimensions (not limited to 256)
 * 
 * Spec: M0-W-1430, CUDA-5030
 * Story: FT-015
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdio.h>

namespace worker {
namespace kernels {

/**
 * Embedding lookup kernel (FP16).
 * 
 * Retrieves embeddings for input token IDs from weight matrix.
 * Each thread handles one element of one embedding.
 * 
 * Memory layout:
 * - weight_matrix: [vocab_size, hidden_dim] row-major
 * - embeddings: [batch_size, hidden_dim] row-major
 * 
 * Grid configuration:
 * - Grid: (batch_size, ceil(hidden_dim / 256))
 * - Block: 256 threads
 * 
 * Memory access pattern:
 * - Coalesced: consecutive threads read consecutive memory locations
 * - Each block processes 256 dimensions of one token's embedding
 * 
 * @param token_ids Input token IDs [batch_size]
 * @param weight_matrix Embedding weight matrix [vocab_size, hidden_dim]
 * @param embeddings Output embeddings [batch_size, hidden_dim]
 * @param batch_size Number of tokens
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 */
__global__ void embedding_lookup_fp16(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size
) {
    // Each thread handles one element of one embedding
    int token_idx = blockIdx.x;  // Which token in batch
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;  // Which dimension
    
    // Bounds check
    if (token_idx >= batch_size || dim_idx >= hidden_dim) {
        return;
    }
    
    // Get token ID
    int token_id = token_ids[token_idx];
    
    // Validate token ID (bounds check)
    if (token_id < 0 || token_id >= vocab_size) {
        // Invalid token ID, set to zero
        embeddings[token_idx * hidden_dim + dim_idx] = __float2half(0.0f);
        return;
    }
    
    // Lookup embedding from weight matrix
    // weight_matrix layout: [vocab_size, hidden_dim]
    // Coalesced memory access: consecutive threads access consecutive elements
    half value = weight_matrix[token_id * hidden_dim + dim_idx];
    embeddings[token_idx * hidden_dim + dim_idx] = value;
}

/**
 * Embedding lookup kernel (FP32).
 * 
 * Same as FP16 version but with single-precision floats.
 * Use for higher precision requirements or when FP16 not available.
 * 
 * @param token_ids Input token IDs [batch_size]
 * @param weight_matrix Embedding weight matrix [vocab_size, hidden_dim]
 * @param embeddings Output embeddings [batch_size, hidden_dim]
 * @param batch_size Number of tokens
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 */
__global__ void embedding_lookup_fp32(
    const int* token_ids,
    const float* weight_matrix,
    float* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= batch_size || dim_idx >= hidden_dim) {
        return;
    }
    
    int token_id = token_ids[token_idx];
    
    if (token_id < 0 || token_id >= vocab_size) {
        embeddings[token_idx * hidden_dim + dim_idx] = 0.0f;
        return;
    }
    
    float value = weight_matrix[token_id * hidden_dim + dim_idx];
    embeddings[token_idx * hidden_dim + dim_idx] = value;
}

/**
 * Launch embedding lookup kernel (FP16).
 * 
 * Configures grid/block dimensions and launches kernel.
 * 
 * Grid configuration:
 * - Grid X: batch_size (one block per token)
 * - Grid Y: ceil(hidden_dim / 256) (multiple blocks if hidden_dim > 256)
 * - Block: 256 threads
 * 
 * Example: batch_size=4, hidden_dim=1024
 * - Grid: (4, 4) = 16 blocks
 * - Block: 256 threads
 * - Total threads: 4096
 * 
 * @param token_ids Device pointer to token IDs [batch_size]
 * @param weight_matrix Device pointer to embedding weights [vocab_size, hidden_dim]
 * @param embeddings Device pointer to output embeddings [batch_size, hidden_dim]
 * @param batch_size Number of tokens
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream (0 = default stream)
 */
void launch_embedding_lookup_fp16(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
) {
    // Validate inputs
    if (batch_size <= 0 || hidden_dim <= 0 || vocab_size <= 0) {
        fprintf(stderr, "Invalid dimensions: batch_size=%d, hidden_dim=%d, vocab_size=%d\n",
                batch_size, hidden_dim, vocab_size);
        return;
    }
    
    if (token_ids == nullptr || weight_matrix == nullptr || embeddings == nullptr) {
        fprintf(stderr, "Null pointer in embedding lookup\n");
        return;
    }
    
    // Kernel launch configuration
    // Grid: (batch_size, ceil(hidden_dim / 256))
    // Block: 256 threads
    int threads_per_block = 256;
    int blocks_y = (hidden_dim + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_y);
    dim3 block(threads_per_block);
    
    // Launch kernel
    embedding_lookup_fp16<<<grid, block, 0, stream>>>(
        token_ids,
        weight_matrix,
        embeddings,
        batch_size,
        hidden_dim,
        vocab_size
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Embedding kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

/**
 * Launch embedding lookup kernel (FP32).
 * 
 * Same as FP16 version but with single-precision floats.
 * 
 * @param token_ids Device pointer to token IDs [batch_size]
 * @param weight_matrix Device pointer to embedding weights [vocab_size, hidden_dim]
 * @param embeddings Device pointer to output embeddings [batch_size, hidden_dim]
 * @param batch_size Number of tokens
 * @param hidden_dim Embedding dimension
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream (0 = default stream)
 */
void launch_embedding_lookup_fp32(
    const int* token_ids,
    const float* weight_matrix,
    float* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream
) {
    if (batch_size <= 0 || hidden_dim <= 0 || vocab_size <= 0) {
        fprintf(stderr, "Invalid dimensions: batch_size=%d, hidden_dim=%d, vocab_size=%d\n",
                batch_size, hidden_dim, vocab_size);
        return;
    }
    
    if (token_ids == nullptr || weight_matrix == nullptr || embeddings == nullptr) {
        fprintf(stderr, "Null pointer in embedding lookup\n");
        return;
    }
    
    int threads_per_block = 256;
    int blocks_y = (hidden_dim + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, blocks_y);
    dim3 block(threads_per_block);
    
    embedding_lookup_fp32<<<grid, block, 0, stream>>>(
        token_ids,
        weight_matrix,
        embeddings,
        batch_size,
        hidden_dim,
        vocab_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Embedding kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels
} // namespace worker

// Extern C wrapper for transformer
extern "C" {
    void cuda_embedding_lookup(
        const uint32_t* token_ids,
        const void* embedding_table,
        void* output,
        uint32_t batch_size,
        uint32_t vocab_size,
        uint32_t hidden_dim,
        cudaStream_t stream
    ) {
        worker::kernels::launch_embedding_lookup_fp16(
            reinterpret_cast<const int*>(token_ids),
            reinterpret_cast<const half*>(embedding_table),
            reinterpret_cast<half*>(output),
            batch_size,
            hidden_dim,
            vocab_size,
            stream
        );
    }
}

// ---
// Crafted by GPT-Gamma 🤖function-Alpha 🏗️
