//! Main inference backend implementation
//!
//! Created by: TEAM-015 (refactored from `candle_backend.rs`)
//! Original code by: TEAM-000, TEAM-009, TEAM-011, TEAM-014
//! Modified by: TEAM-017 (added multi-model support with enum pattern)

use super::models::{self, Model};
use super::sampling;
use super::tokenizer_loader;
use crate::common::{InferenceResult, SamplingConfig};
use crate::http::InferenceBackend;
use crate::token_output_stream::TokenOutputStream;
use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

/// Candle inference backend using candle-transformers models
///
/// TEAM-009: Complete rewrite to use Candle's models directly
/// instead of building layers from scratch.
/// TEAM-015: Refactored into focused modules
/// TEAM-017: Changed to enum pattern for Candle idiomaticity
pub struct CandleInferenceBackend {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    model_size_bytes: u64,
}

impl CandleInferenceBackend {
    /// Load model from `SafeTensors` with auto-detected architecture
    ///
    /// TEAM-009: Uses candle-transformers models directly
    /// TEAM-015: Delegates to `model_loader` module
    /// TEAM-017: Uses model factory with enum pattern (Candle-idiomatic)
    pub fn load(model_path: &str, device: Device) -> Result<Self> {
        let path = Path::new(model_path);

        // TEAM-017: Load model using model factory (returns Model enum)
        let model = models::load_model(model_path, &device)?;
        let model_size_bytes = models::calculate_model_size(model_path)?;

        // TEAM-017: Load tokenizer with auto-detection
        let tokenizer = tokenizer_loader::load_tokenizer(path)?;

        tracing::info!(
            architecture = model.architecture(),
            vocab_size = model.vocab_size(),
            tokenizer_vocab = tokenizer.get_vocab_size(true),
            model_size_mb = model_size_bytes / 1_000_000,
            "Model and tokenizer loaded successfully"
        );

        Ok(Self { model, tokenizer, device, model_size_bytes })
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        self.model_size_bytes
    }

    /// Warmup GPU with dummy inference
    ///
    /// TEAM-014: Eliminates cold start by running a single token generation.
    /// This initializes CUDA kernels and caches, preventing 9s overhead on first request.
    /// TEAM-017: Updated to use Model enum
    pub fn warmup(&mut self) -> Result<()> {
        tracing::info!("Starting GPU warmup...");
        let start = std::time::Instant::now();

        // Use a simple prompt for warmup
        let warmup_prompt = "Hello";

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(warmup_prompt, true)
            .map_err(|e| anyhow::anyhow!("Warmup tokenization failed: {e}"))?;
        let tokens = encoding.get_ids();

        // Create input tensor
        let input_ids = Tensor::new(tokens, &self.device)
            .context("Failed to create warmup tensor")?
            .unsqueeze(0)
            .context("Failed to unsqueeze warmup tensor")?;

        // TEAM-017: Single forward pass using Model enum (delegates to specific model)
        let _logits = self.model.forward(&input_ids, 0).context("Warmup forward pass failed")?;

        let duration = start.elapsed();
        tracing::info!(duration_ms = duration.as_millis(), "GPU warmup complete");

        Ok(())
    }
}

#[async_trait]
impl InferenceBackend for CandleInferenceBackend {
    /// Execute inference with streaming token generation
    ///
    /// TEAM-009: Complete implementation using candle-transformers
    /// TEAM-014: Added warmup support, `LogitsProcessor`, `TokenOutputStream`
    /// TEAM-015: Refactored into focused modules
    /// TEAM-017: Updated to use Model enum (Candle-idiomatic)
    async fn execute(
        &mut self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        tracing::debug!(
            prompt_len = prompt.len(),
            max_tokens = config.max_tokens,
            temperature = config.temperature,
            "Starting inference"
        );

        // Tokenize prompt
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenization failed: {e}"))?;
        let mut tokens = encoding.get_ids().to_vec();

        tracing::debug!(prompt_tokens = tokens.len(), "Prompt tokenized");

        // TEAM-014: Create LogitsProcessor for proper sampling
        // TEAM-015: Delegates to sampling module
        let mut logits_processor = sampling::create_logits_processor(config);

        // TEAM-014: Create TokenOutputStream for proper space handling
        let mut token_stream = TokenOutputStream::new(self.tokenizer.clone());

        // Generate tokens
        let mut generated_tokens = Vec::new();
        let mut generated_text = Vec::new();
        let start_time = std::time::Instant::now();

        for pos in 0..config.max_tokens {
            let pos_usize = pos as usize;

            // TEAM-011: Prepare input tensor with correct shape [batch_size, seq_len]
            let input_ids = if pos == 0 {
                // First iteration: use all prompt tokens
                Tensor::new(&tokens[..], &self.device)
                    .map_err(|e| format!("Failed to create input tensor: {e}"))?
                    .unsqueeze(0) // Add batch dimension: [seq_len] -> [1, seq_len]
                    .map_err(|e| format!("Failed to unsqueeze input tensor: {e}"))?
            } else {
                // Subsequent iterations: only last token
                Tensor::new(&[tokens[tokens.len() - 1]], &self.device)
                    .map_err(|e| format!("Failed to create input tensor: {e}"))?
                    .unsqueeze(0) // Add batch dimension: [1] -> [1, 1]
                    .map_err(|e| format!("Failed to unsqueeze input tensor: {e}"))?
            };

            // TEAM-009: Verify device residency (log only, no comparison since Device doesn't impl PartialEq)
            if pos == 0 {
                tracing::debug!(
                    input_device = ?input_ids.device(),
                    expected_device = ?self.device,
                    "Device residency check: input tensor"
                );
            }

            // TEAM-017: Forward pass using Model enum (delegates to specific model)
            let logits =
                self.model.forward(&input_ids, pos_usize).map_err(|e| format!("Forward pass failed: {e}"))?;

            // TEAM-009: Log output device residency
            if pos == 0 {
                tracing::debug!(
                    output_device = ?logits.device(),
                    expected_device = ?self.device,
                    "Device residency check: output tensor"
                );
            }

            // Get logits for last position
            let logits =
                logits.squeeze(0).map_err(|e| format!("Failed to squeeze logits: {e}"))?;
            let logits = if logits.dims().len() > 1 {
                logits
                    .get(logits.dims()[0] - 1)
                    .map_err(|e| format!("Failed to get last logits: {e}"))?
            } else {
                logits
            };

            // TEAM-014: Sample next token using Candle's LogitsProcessor
            let next_token =
                logits_processor.sample(&logits).map_err(|e| format!("Sampling failed: {e}"))?;

            // TEAM-017: Check for EOS - try tokenizer first (Candle-idiomatic), fallback to model
            let is_eos = self
                .tokenizer
                .token_to_id("</s>")
                .map(|eos_id| next_token == eos_id)
                .unwrap_or_else(|| next_token == self.model.eos_token_id());

            if is_eos {
                tracing::debug!("EOS token generated");
                break;
            }

            // TEAM-014: Use TokenOutputStream for proper streaming decode with spaces
            if let Some(token_str) = token_stream
                .next_token(next_token)
                .map_err(|e| format!("Detokenization failed: {e}"))?
            {
                generated_text.push(token_str);
            }

            generated_tokens.push(next_token);
            tokens.push(next_token);

            // Log progress
            if (pos + 1) % 10 == 0 {
                tracing::debug!(tokens_generated = pos + 1, "Generation progress");
            }
        }

        // TEAM-014: Get any remaining decoded bytes from token stream
        if let Some(rest) =
            token_stream.decode_rest().map_err(|e| format!("Failed to decode rest: {e}"))?
        {
            generated_text.push(rest);
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        let tokens_per_sec =
            if duration_ms > 0 { (generated_tokens.len() as u64 * 1000) / duration_ms } else { 0 };

        tracing::info!(
            tokens_generated = generated_tokens.len(),
            duration_ms = duration_ms,
            tokens_per_sec = tokens_per_sec,
            "Inference completed"
        );

        Ok(InferenceResult::max_tokens(generated_text, generated_tokens, config.seed, duration_ms))
    }

    /// Cancel inference (not implemented for single-threaded)
    async fn cancel(&self, _job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    /// Get VRAM usage
    fn vram_usage(&self) -> u64 {
        #[cfg(feature = "cuda")]
        {
            self.model_size_bytes
        }

        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Check if backend is healthy
    fn is_healthy(&self) -> bool {
        true
    }
}
