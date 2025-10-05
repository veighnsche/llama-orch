// Absolute positional embedding kernel for GPT architecture
//
// Implements learned absolute positional embeddings:
// output = token_embeddings + position_embeddings
//
// This differs from Llama's RoPE (Rotary Position Embedding) which applies
// rotations to query/key vectors.
//
// Spec: M0-W-1434, M0-W-1215
// Story: GT-008

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Add absolute positional embeddings to token embeddings
//
// Args:
//   output: Output tensor [batch_size, seq_len, hidden_size]
//   token_emb: Token embeddings [batch_size, seq_len, hidden_size]
//   pos_emb: Position embeddings [max_seq_len, hidden_size]
//   batch_size: Batch size
//   seq_len: Sequence length
//   hidden_size: Hidden dimension (d_model)
//
// Each thread processes one element
__global__ void add_positional_embedding_kernel(
    half* output,
    const half* token_emb,
    const half* pos_emb,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size;
    
    if (idx < total_elements) {
        // Decompose index into (batch, seq, hidden)
        int hidden_idx = idx % hidden_size;
        int seq_idx = (idx / hidden_size) % seq_len;
        int batch_idx = idx / (seq_len * hidden_size);
        
        // Position embedding index (same for all batches)
        int pos_emb_idx = seq_idx * hidden_size + hidden_idx;
        
        // Add token embedding + position embedding
        float token_val = __half2float(token_emb[idx]);
        float pos_val = __half2float(pos_emb[pos_emb_idx]);
        output[idx] = __float2half(token_val + pos_val);
    }
}

// Host function to launch positional embedding kernel
extern "C" void cuda_add_positional_embedding(
    half* output,
    const half* token_emb,
    const half* pos_emb,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = batch_size * seq_len * hidden_size;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    add_positional_embedding_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output, token_emb, pos_emb,
        batch_size, seq_len, hidden_size
    );
}

// In-place version (modifies token_emb directly)
// Useful for memory-constrained scenarios
__global__ void add_positional_embedding_inplace_kernel(
    half* token_emb,
    const half* pos_emb,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size;
    
    if (idx < total_elements) {
        int hidden_idx = idx % hidden_size;
        int seq_idx = (idx / hidden_size) % seq_len;
        
        int pos_emb_idx = seq_idx * hidden_size + hidden_idx;
        
        float token_val = __half2float(token_emb[idx]);
        float pos_val = __half2float(pos_emb[pos_emb_idx]);
        token_emb[idx] = __float2half(token_val + pos_val);
    }
}

extern "C" void cuda_add_positional_embedding_inplace(
    half* token_emb,
    const half* pos_emb,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = batch_size * seq_len * hidden_size;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    add_positional_embedding_inplace_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        token_emb, pos_emb,
        batch_size, seq_len, hidden_size
    );
}

// Optimized version using vectorized loads (half2)
// Processes 2 elements per thread for better memory bandwidth
__global__ void add_positional_embedding_vectorized_kernel(
    half2* output,
    const half2* token_emb,
    const half2* pos_emb,
    int batch_size,
    int seq_len,
    int hidden_size_div2  // hidden_size / 2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size_div2;
    
    if (idx < total_elements) {
        int hidden_idx = idx % hidden_size_div2;
        int seq_idx = (idx / hidden_size_div2) % seq_len;
        
        int pos_emb_idx = seq_idx * hidden_size_div2 + hidden_idx;
        
        // Load 2 elements at once
        half2 token_val = token_emb[idx];
        half2 pos_val = pos_emb[pos_emb_idx];
        
        // Add using half2 intrinsics
        output[idx] = __hadd2(token_val, pos_val);
    }
}

extern "C" void cuda_add_positional_embedding_vectorized(
    half* output,
    const half* token_emb,
    const half* pos_emb,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    // Only use vectorized version if hidden_size is even
    if (hidden_size % 2 != 0) {
        cuda_add_positional_embedding(output, token_emb, pos_emb,
                                     batch_size, seq_len, hidden_size, stream);
        return;
    }
    
    int hidden_size_div2 = hidden_size / 2;
    int total_elements = batch_size * seq_len * hidden_size_div2;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    add_positional_embedding_vectorized_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<half2*>(output),
        reinterpret_cast<const half2*>(token_emb),
        reinterpret_cast<const half2*>(pos_emb),
        batch_size, seq_len, hidden_size_div2
    );
}

// Extract position embeddings for a specific position range
// Useful for incremental decoding where we only need new positions
__global__ void extract_position_embeddings_kernel(
    half* output,
    const half* pos_emb,
    int start_pos,
    int num_positions,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_positions * hidden_size;
    
    if (idx < total_elements) {
        int hidden_idx = idx % hidden_size;
        int pos_offset = idx / hidden_size;
        int pos_idx = start_pos + pos_offset;
        
        int pos_emb_idx = pos_idx * hidden_size + hidden_idx;
        output[idx] = pos_emb[pos_emb_idx];
    }
}

extern "C" void cuda_extract_position_embeddings(
    half* output,
    const half* pos_emb,
    int start_pos,
    int num_positions,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = num_positions * hidden_size;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    extract_position_embeddings_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        output, pos_emb, start_pos, num_positions, hidden_size
    );
}

// ---
// Crafted by GPT-Gamma 🤖
