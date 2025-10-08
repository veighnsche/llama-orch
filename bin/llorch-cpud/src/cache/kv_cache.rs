//! KV Cache Implementation
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 3 (KV Cache)

use ndarray::{Array3, Array4, s};

/// KV Cache for autoregressive generation
///
/// Stores keys and values across generation steps to avoid recomputation.
/// Simple implementation for MVP - room to grow (paged attention later).
pub struct KVCache {
    /// Cache storage: [2, batch, max_seq, n_heads, head_dim]
    /// Index 0 = keys, Index 1 = values
    cache: Option<Array4<f32>>,
    max_seq_len: usize,
    n_heads: usize,
    head_dim: usize,
}

impl KVCache {
    /// Create new KV cache
    pub fn new(max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        Self {
            cache: None,
            max_seq_len,
            n_heads,
            head_dim,
        }
    }

    /// Update cache with new K, V at position start_pos
    ///
    /// # Arguments
    /// * `k` - Keys [batch, seq, n_heads, head_dim]
    /// * `v` - Values [batch, seq, n_heads, head_dim]
    /// * `start_pos` - Position to insert at
    pub fn update(&mut self, k: &Array3<f32>, v: &Array3<f32>, start_pos: usize) {
        let batch = k.shape()[0];
        let seq_len = k.shape()[1];

        // Initialize cache on first use
        if self.cache.is_none() {
            tracing::debug!(
                "Initializing KV cache: [2, {}, {}, {}, {}]",
                batch,
                self.max_seq_len,
                self.n_heads,
                self.head_dim
            );
            self.cache = Some(Array4::zeros((
                2,
                batch,
                self.max_seq_len,
                self.n_heads,
                self.head_dim,
            )));
        }

        // Update cache at position start_pos
        let cache = self.cache.as_mut().unwrap();
        let end_pos = start_pos + seq_len;

        tracing::trace!("Updating cache: positions {}..{}", start_pos, end_pos);

        // Store keys at index 0
        cache
            .slice_mut(s![0, .., start_pos..end_pos, .., ..])
            .assign(k);

        // Store values at index 1
        cache
            .slice_mut(s![1, .., start_pos..end_pos, .., ..])
            .assign(v);
    }

    /// Retrieve cached K, V up to end_pos
    ///
    /// # Returns
    /// (keys, values) each [batch, seq, n_heads, head_dim]
    pub fn get(&self, end_pos: usize) -> (Array3<f32>, Array3<f32>) {
        let cache = self
            .cache
            .as_ref()
            .expect("Cache not initialized - call update first");

        tracing::trace!("Retrieving cache: positions 0..{}", end_pos);

        // Retrieve keys from index 0
        let k = cache.slice(s![0, .., ..end_pos, .., ..]).to_owned();

        // Retrieve values from index 1
        let v = cache.slice(s![1, .., ..end_pos, .., ..]).to_owned();

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
        let mut cache = KVCache::new(2048, 16, 64);
        let k = Array3::zeros((1, 2, 16, 64));
        let v = Array3::zeros((1, 2, 16, 64));

        cache.update(&k, &v, 0);

        let (cached_k, cached_v) = cache.get(2);
        assert_eq!(cached_k.shape(), &[1, 2, 16, 64]);
        assert_eq!(cached_v.shape(), &[1, 2, 16, 64]);
    }
}
