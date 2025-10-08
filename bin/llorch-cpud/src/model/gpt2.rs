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
    config: GPTConfig,
    /// Token and position embeddings
    embedding: Embedding,
    /// Transformer blocks (24 for GPT-2 Medium)
    blocks: Vec<TransformerBlock>,
    /// Final layer norm
    ln_f: LayerNorm,
    /// Language model head (weight tied with token embeddings)
    lm_head: Array2<f32>,
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
    /// * `config` - Sampling configuration
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate(&self, input_tokens: &[usize], config: &SamplingConfig) -> Result<Vec<usize>> {
        tracing::debug!("Generating with {} input tokens", input_tokens.len());

        // TODO: Implement generation loop
        // 1. Initialize cache
        // 2. Process prompt (all tokens at once)
        // 3. Generate tokens one by one (autoregressive)
        // 4. Apply sampling (temperature, top_k, top_p)
        // 5. Stop at max_tokens or EOS

        // CHECKPOINTS:
        // - Checkpoint 8: Full logits after all blocks
        // - Checkpoint 9: Selected logits (last position)
        // - Checkpoint 10: Argmax sampling (temperature=0)
        // - Checkpoint 11: Softmax probabilities (temperature>0)
        // - Checkpoint 12: End-to-end generation

        // Placeholder
        Ok(input_tokens.to_vec())
    }

    /// Forward pass (single step)
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs [seq]
    /// * `cache` - KV cache
    /// * `start_pos` - Starting position
    ///
    /// # Returns
    /// Logits [seq, vocab_size]
    fn forward(&mut self, token_ids: &Array1<usize>, cache: &mut KVCache, start_pos: usize) -> Array2<f32> {
        // TODO: Implement forward pass
        // 1. Embeddings (token + position)
        // 2. Pass through all transformer blocks
        // 3. Final layer norm
        // 4. LM head projection

        // Placeholder
        let seq_len = token_ids.len();
        let vocab_size = self.config.vocab_size;
        Array2::zeros((seq_len, vocab_size))
    }

    /// Sample next token
    fn sample(&self, logits: &Array1<f32>, config: &SamplingConfig) -> usize {
        // TODO: Implement sampling
        // - temperature=0: argmax (deterministic)
        // - temperature>0: sample from softmax distribution
        // - Apply top_k, top_p if specified

        // Placeholder: return first token
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load() {
        // TODO: Test model loading
        // This requires a test model file
    }
}
