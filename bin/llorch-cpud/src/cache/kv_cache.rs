//! KV Cache Implementation
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 3 (KV Cache)

use ndarray::{Array3, Array4};

/// KV Cache for autoregressive generation
///
/// Stores keys and values across generation steps to avoid recomputation.
/// Simple implementation for MVP - room to grow (paged attention later).
pub struct KVCache {
    /// Cache storage: [2, batch, max_seq, n_heads, head_dim]
    /// Index 0 = keys, Index 1 = values
    cache: Option<Array4<f32>>,
    _max_seq_len: usize,
    n_heads: usize,
    head_dim: usize,
}

impl KVCache {
    /// Create new KV cache
    pub fn new(max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        Self { cache: None, _max_seq_len: max_seq_len, n_heads, head_dim }
    }

    /// Update cache with new K, V at position start_pos
    ///
    /// # Arguments
    /// * `k` - Keys [batch, seq, n_heads, head_dim]
    /// * `v` - Values [batch, seq, n_heads, head_dim]
    /// * `start_pos` - Position to insert at
    pub fn update(&mut self, _k: &Array3<f32>, _v: &Array3<f32>, _start_pos: usize) {
        // TODO: Implement cache update (Checkpoint 3)
        // For now, just a stub
    }

    /// Retrieve cached K, V up to end_pos
    ///
    /// # Returns
    /// (keys, values) each [batch, seq, n_heads, head_dim]
    pub fn get(&self, _end_pos: usize) -> (Array3<f32>, Array3<f32>) {
        // TODO: Implement cache retrieval (Checkpoint 3)
        // Placeholder
        let k = Array3::zeros((1, self.n_heads, self.head_dim));
        let v = Array3::zeros((1, self.n_heads, self.head_dim));
        (k, v)
    }

    /// Clear cache (for new sequence)
    pub fn clear(&mut self) {
        self.cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_initialization() {
        let cache = KVCache::new(2048, 16, 64);
        let (cached_k, cached_v) = cache.get(2);

        // Placeholder returns simple shape
        assert_eq!(cached_k.shape()[1], 16); // n_heads
        assert_eq!(cached_v.shape()[2], 64); // head_dim
    }
}
