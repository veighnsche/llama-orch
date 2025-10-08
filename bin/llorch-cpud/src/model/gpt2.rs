//! GPT-2 Model Implementation
//!
//! IMPORTS: worker-models (GPTConfig), internal crate modules
//! CHECKPOINTS: 8-12

use worker_common::SamplingConfig;
use worker_models::GPTConfig;

use crate::cache::KVCache;
use crate::error::{Error, Result};
use crate::layers::{Embedding, LayerNorm, TransformerBlock};
use ndarray::{Array1, Array2};

/// GPT-2 Model
pub struct GPT2Model {
    /// Model configuration
    _config: GPTConfig,
    /// Token and position embeddings
    _embedding: Embedding,
    /// Transformer blocks (24 for GPT-2 Medium)
    _blocks: Vec<TransformerBlock>,
    /// Final layer norm
    _ln_f: LayerNorm,
    /// Language model head (weight tied with token embeddings)
    _lm_head: Array2<f32>,
}

impl GPT2Model {
    /// Load model from path
    pub fn load(model_path: &str) -> Result<Self> {
        tracing::info!("Loading GPT-2 model weights from: {}", model_path);

        // TODO: Load model weights from GGUF
        // 1. Load config
        // 2. Load embeddings
        // 3. Load transformer blocks (24 blocks)
        // 4. Load final layer norm
        // 5. Load lm_head (or tie with token embeddings)

        Err(Error::ModelLoad("Not implemented yet".to_string()))
    }

    /// Generate tokens
    ///
    /// # Arguments
    /// * `input_tokens` - Input token IDs
    /// * `_config` - Sampling configuration
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate(&self, input_tokens: &[usize], _config: &SamplingConfig) -> Result<Vec<usize>> {
        // TODO: Implement generation loop (Checkpoints 8-12)
        Ok(input_tokens.to_vec())
    }

    /// Forward pass (single step)
    fn forward(
        &mut self,
        token_ids: &Array1<usize>,
        _cache: &mut KVCache,
        _start_pos: usize,
    ) -> Array2<f32> {
        // TODO: Implement forward pass (Checkpoints 8-9)
        let seq_len = token_ids.len();
        let vocab_size = self._config.vocab_size;
        Array2::zeros((seq_len, vocab_size))
    }

    /// Sample next token
    fn sample(&self, _logits: &Array1<f32>, _config: &SamplingConfig) -> usize {
        // TODO: Implement sampling (Checkpoints 10-11)
        0
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_model_load() {
        // TODO: Test model loading
        // This requires a test model file
    }
}
