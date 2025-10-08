//! Simple KV cache for Llama-2
//!
//! Checkpoint 3 validation target
//!
//! IMPLEMENTATION: Keep simple for MVP
//! STRUCTURE: Room to grow (this is why cache/ is top-level)
//!
//! For Llama-2:
//! - 32 layers (not 24 like GPT-2)
//! - 32 heads (not 16)
//! - Head dim: 128 (not 64)
//! - Max context: 4096 (not 2048)
//!
//! Created by: TEAM-000

use ndarray::Array3;

/// Simple KV cache for Llama-2
pub struct KVCache {
    k_cache: Option<Array3<f32>>,
    v_cache: Option<Array3<f32>>,
    max_seq_len: usize,
}

impl KVCache {
    /// Create new KV cache
    ///
    /// For Llama-2:
    /// - n_heads: 32
    /// - head_dim: 128
    /// - max_seq_len: 4096
    pub fn new(n_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        todo!("Implement in Checkpoint 3")
    }

    /// Update cache with new K, V tensors
    ///
    /// Returns: (full_k, full_v) including cached values
    pub fn update(
        &mut self,
        k: Array3<f32>,
        v: Array3<f32>,
        start_pos: usize,
    ) -> (Array3<f32>, Array3<f32>) {
        todo!("Implement in Checkpoint 3")
    }
}
