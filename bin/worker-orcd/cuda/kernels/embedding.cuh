/**
 * Embedding Lookup Kernel Header
 * 
 * Public interface for embedding lookup kernels.
 * 
 * Spec: M0-W-1430, CUDA-5030
 * Story: FT-015
 */

#ifndef WORKER_EMBEDDING_CUH
#define WORKER_EMBEDDING_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace kernels {

/**
 * Embedding lookup kernel (FP16).
 * 
 * Retrieves embeddings for input token IDs from weight matrix.
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
);

/**
 * Embedding lookup kernel (FP32).
 * 
 * Same as FP16 version but with single-precision floats.
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
);

/**
 * Launch embedding lookup kernel (FP16).
 * 
 * Configures grid/block dimensions and launches kernel.
 * 
 * Grid configuration:
 * - Grid: (batch_size, ceil(hidden_dim / 256))
 * - Block: 256 threads
 * 
 * Memory access pattern:
 * - Coalesced: consecutive threads access consecutive memory locations
 * - Optimal for GPU memory bandwidth
 * 
 * @param token_ids Device pointer to token IDs [batch_size]
 * @param weight_matrix Device pointer to embedding weights [vocab_size, hidden_dim]
 * @param embeddings Device pointer to output embeddings [batch_size, hidden_dim]
 * @param batch_size Number of tokens (must be > 0)
 * @param hidden_dim Embedding dimension (must be > 0)
 * @param vocab_size Vocabulary size (must be > 0)
 * @param stream CUDA stream (default: 0)
 * 
 * Error handling:
 * - Validates dimensions (batch_size, hidden_dim, vocab_size > 0)
 * - Validates pointers (not nullptr)
 * - Checks kernel launch errors
 * - Invalid token IDs produce zero embeddings
 */
void launch_embedding_lookup_fp16(
    const int* token_ids,
    const half* weight_matrix,
    half* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream = 0
);

/**
 * Launch embedding lookup kernel (FP32).
 * 
 * Same as FP16 version but with single-precision floats.
 * 
 * @param token_ids Device pointer to token IDs [batch_size]
 * @param weight_matrix Device pointer to embedding weights [vocab_size, hidden_dim]
 * @param embeddings Device pointer to output embeddings [batch_size, hidden_dim]
 * @param batch_size Number of tokens (must be > 0)
 * @param hidden_dim Embedding dimension (must be > 0)
 * @param vocab_size Vocabulary size (must be > 0)
 * @param stream CUDA stream (default: 0)
 */
void launch_embedding_lookup_fp32(
    const int* token_ids,
    const float* weight_matrix,
    float* embeddings,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace worker

#endif // WORKER_EMBEDDING_CUH

// ---
// Built by Foundation-Alpha 🏗️
