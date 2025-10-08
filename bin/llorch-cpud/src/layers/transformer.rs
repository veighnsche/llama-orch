//! Transformer Block
//!
//! IMPORTS: Internal crate imports only (LayerNorm, Attention, FFN, KVCache)
//! CHECKPOINT: 7 (First Block)

use crate::cache::KVCache;
use crate::layers::{Attention, FFN, LayerNorm};
use ndarray::Array2;

/// Transformer Block
///
/// Pre-norm architecture:
/// 1. h = x + attention(ln_1(x))
/// 2. h = h + ffn(ln_2(h))
pub struct TransformerBlock {
    /// First layer norm (before attention)
    ln_1: LayerNorm,
    /// Attention layer
    attn: Attention,
    /// Second layer norm (before FFN)
    ln_2: LayerNorm,
    /// Feedforward network
    ffn: FFN,
}

impl TransformerBlock {
    /// Create new transformer block
    pub fn new(ln_1: LayerNorm, attn: Attention, ln_2: LayerNorm, ffn: FFN) -> Self {
        Self {
            ln_1,
            attn,
            ln_2,
            ffn,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input [batch, seq, dim]
    /// * `cache` - KV cache
    /// * `start_pos` - Starting position
    ///
    /// # Returns
    /// Output [batch, seq, dim]
    pub fn forward(&mut self, x: &Array2<f32>, _cache: &mut KVCache, _start_pos: usize) -> Array2<f32> {
        // TODO: Implement transformer block (Checkpoint 7)
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block_shape() {
        // TODO: Create transformer block and test
        // This requires creating all components (ln_1, attn, ln_2, ffn)
    }
}
