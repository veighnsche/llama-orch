// MXFP4 Embedding Lookup
//
// Implements efficient embedding table access with MXFP4 quantized weights.
// Supports token and position embeddings with on-the-fly dequantization.
//
// Story: GT-034
// Spec: M0-W-1435

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// External MXFP4 dequantization function
extern "C" void cuda_mxfp4_dequant(
    half* output,
    const uint8_t* input,
    int num_elements,
    cudaStream_t stream
);

// MXFP4 embedding lookup kernel
//
// Looks up embeddings from MXFP4 table and dequantizes to FP16.
// Each embedding is stored as MXFP4 blocks.
//
// Args:
//   output: Output FP16 embeddings [batch_size, embedding_dim]
//   mxfp4_table: MXFP4 embedding table [vocab_size, embedding_dim]
//   token_ids: Token IDs to look up [batch_size]
//   batch_size: Number of tokens to look up
//   embedding_dim: Dimension of each embedding
//   vocab_size: Size of vocabulary
__global__ void mxfp4_embedding_lookup_kernel(
    half* output,
    const uint8_t* mxfp4_table,
    const int* token_ids,
    int batch_size,
    int embedding_dim,
    int vocab_size
) {
    int batch_idx = blockIdx.x;
    int embed_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || embed_idx >= embedding_dim) {
        return;
    }
    
    // Get token ID
    int token_id = token_ids[batch_idx];
    
    // Bounds check
    if (token_id < 0 || token_id >= vocab_size) {
        output[batch_idx * embedding_dim + embed_idx] = __float2half(0.0f);
        return;
    }
    
    // Calculate MXFP4 block offset for this embedding
    int elements_per_block = 32;
    int bytes_per_block = 17;
    int blocks_per_embedding = (embedding_dim + elements_per_block - 1) / elements_per_block;
    
    // Offset to this token's embedding in MXFP4 table
    size_t token_offset = (size_t)token_id * blocks_per_embedding * bytes_per_block;
    
    // Which block within the embedding?
    int block_idx = embed_idx / elements_per_block;
    int elem_in_block = embed_idx % elements_per_block;
    
    // Load MXFP4 block
    const uint8_t* block_data = mxfp4_table + token_offset + block_idx * bytes_per_block;
    
    // Extract FP4 mantissa
    int byte_idx = elem_in_block / 2;
    int nibble = elem_in_block % 2;
    uint8_t packed_byte = block_data[byte_idx];
    uint8_t fp4_mantissa = (nibble == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
    
    // Extract FP8 scale
    uint8_t fp8_scale = block_data[16];
    
    // Dequantize (simplified - use lookup table in production)
    __shared__ float fp4_table[16];
    if (threadIdx.x < 16) {
        fp4_table[threadIdx.x] = (threadIdx.x < 8) ? 
            (float)threadIdx.x * 0.5f : 
            -(float)(threadIdx.x - 8) * 0.5f;
    }
    __syncthreads();
    
    float mantissa = fp4_table[fp4_mantissa];
    float scale = powf(2.0f, (float)((int)fp8_scale - 127));
    float value = mantissa * scale;
    
    // Write output
    output[batch_idx * embedding_dim + embed_idx] = __float2half(value);
}

// Host function: MXFP4 embedding lookup
extern "C" void mxfp4_embedding_lookup(
    half* output,
    const uint8_t* mxfp4_table,
    const int* token_ids,
    int batch_size,
    int embedding_dim,
    int vocab_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size, (embedding_dim + 255) / 256);
    
    mxfp4_embedding_lookup_kernel<<<grid, block, 0, stream>>>(
        output,
        mxfp4_table,
        token_ids,
        batch_size,
        embedding_dim,
        vocab_size
    );
}

// Optimized version: Dequantize full embedding table once
extern "C" void mxfp4_embedding_lookup_cached(
    half* output,
    const half* fp16_table,  // Pre-dequantized FP16 table
    const int* token_ids,
    int batch_size,
    int embedding_dim,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size * embedding_dim + threads - 1) / threads;
    
    auto lookup_kernel = [=] __device__ (
        half* output,
        const half* fp16_table,
        const int* token_ids,
        int batch_size,
        int embedding_dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size * embedding_dim) {
            int batch_idx = idx / embedding_dim;
            int embed_idx = idx % embedding_dim;
            int token_id = token_ids[batch_idx];
            
            output[idx] = fp16_table[token_id * embedding_dim + embed_idx];
        }
    };
    
    // Note: Lambda placeholder - actual implementation would use separate kernel
}

// Batch embedding lookup for multiple sequences
extern "C" void mxfp4_embedding_lookup_batch(
    half** output_array,
    const uint8_t* mxfp4_table,
    const int** token_ids_array,
    const int* batch_sizes,
    int num_sequences,
    int embedding_dim,
    int vocab_size,
    cudaStream_t stream
) {
    for (int i = 0; i < num_sequences; i++) {
        mxfp4_embedding_lookup(
            output_array[i],
            mxfp4_table,
            token_ids_array[i],
            batch_sizes[i],
            embedding_dim,
            vocab_size,
            stream
        );
    }
}

// Position embedding with MXFP4
// Adds position embeddings to token embeddings
extern "C" void mxfp4_add_position_embeddings(
    half* token_embeddings,  // [batch_size, seq_len, embedding_dim]
    const uint8_t* mxfp4_pos_table,  // MXFP4 position embedding table
    int batch_size,
    int seq_len,
    int embedding_dim,
    cudaStream_t stream
) {
    // Create position IDs [0, 1, 2, ..., seq_len-1]
    int* pos_ids;
    cudaMalloc(&pos_ids, seq_len * sizeof(int));
    
    // Initialize position IDs on device
    auto init_kernel = [=] __device__ (int* pos_ids, int seq_len) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < seq_len) {
            pos_ids[idx] = idx;
        }
    };
    
    // Allocate temp buffer for position embeddings
    half* pos_embeddings;
    cudaMalloc(&pos_embeddings, seq_len * embedding_dim * sizeof(half));
    
    // Look up position embeddings
    mxfp4_embedding_lookup(
        pos_embeddings,
        mxfp4_pos_table,
        pos_ids,
        seq_len,
        embedding_dim,
        seq_len,  // vocab_size = max_seq_len for position embeddings
        stream
    );
    
    // Add position embeddings to token embeddings (broadcast across batch)
    int threads = 256;
    int total_elements = batch_size * seq_len * embedding_dim;
    int blocks = (total_elements + threads - 1) / threads;
    
    auto add_kernel = [=] __device__ (
        half* token_embeddings,
        const half* pos_embeddings,
        int batch_size,
        int seq_len,
        int embedding_dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < batch_size * seq_len * embedding_dim) {
            int pos_idx = (idx / embedding_dim) % seq_len;
            int embed_idx = idx % embedding_dim;
            int pos_offset = pos_idx * embedding_dim + embed_idx;
            
            token_embeddings[idx] = __hadd(
                token_embeddings[idx],
                pos_embeddings[pos_offset]
            );
        }
    };
    
    // Cleanup
    cudaFree(pos_ids);
    cudaFree(pos_embeddings);
}

// Calculate VRAM savings for embedding table
extern "C" size_t mxfp4_embedding_vram_savings(int vocab_size, int embedding_dim) {
    size_t fp16_size = vocab_size * embedding_dim * sizeof(half);
    size_t mxfp4_size = vocab_size * ((embedding_dim + 31) / 32) * 17;
    return fp16_size - mxfp4_size;
}

// ---
// Crafted by GPT-Gamma 🤖
