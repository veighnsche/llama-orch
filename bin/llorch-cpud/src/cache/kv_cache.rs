//! KV Cache Implementation
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 3 (KV Cache)

use ndarray::{Array3, ArrayD};

/// KV Cache for autoregressive generation
///
/// Stores keys and values across generation steps to avoid recomputation.
/// Simple implementation for MVP - room to grow (paged attention later).
pub struct KVCache {
    /// Cache storage: [2, max_seq, n_heads, head_dim]
    /// Index 0 = keys, Index 1 = values
    cache: Option<ArrayD<f32>>,
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
    /// * `k` - Keys [batch_seq, n_heads, head_dim]
    /// * `v` - Values [batch_seq, n_heads, head_dim]
    /// * `start_pos` - Position to insert at
    pub fn update(&mut self, k: &Array3<f32>, v: &Array3<f32>, start_pos: usize) {
        let batch_seq = k.shape()[0];
        
        // Initialize cache on first use
        if self.cache.is_none() {
            self.cache = Some(ArrayD::zeros(vec![
                2,
                self._max_seq_len,
                self.n_heads,
                self.head_dim,
            ]));
        }
        
        // Update cache at position start_pos
        if let Some(ref mut cache) = self.cache {
            // Store keys at cache[0, start_pos:start_pos+batch_seq, :, :]
            for s in 0..batch_seq {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        cache[[0, start_pos + s, h, d]] = k[[s, h, d]];
                        cache[[1, start_pos + s, h, d]] = v[[s, h, d]];
                    }
                }
            }
        }
    }

    /// Retrieve cached K, V up to end_pos
    ///
    /// # Returns
    /// (keys, values) each [seq, n_heads, head_dim]
    pub fn get(&self, end_pos: usize) -> (Array3<f32>, Array3<f32>) {
        if let Some(ref cache) = self.cache {
            // Extract K and V up to end_pos
            let mut k = Array3::zeros((end_pos, self.n_heads, self.head_dim));
            let mut v = Array3::zeros((end_pos, self.n_heads, self.head_dim));
            
            for s in 0..end_pos {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        k[[s, h, d]] = cache[[0, s, h, d]];
                        v[[s, h, d]] = cache[[1, s, h, d]];
                    }
                }
            }
            
            (k, v)
        } else {
            // Empty cache - return zeros
            let k = Array3::zeros((end_pos, self.n_heads, self.head_dim));
            let v = Array3::zeros((end_pos, self.n_heads, self.head_dim));
            (k, v)
        }
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
