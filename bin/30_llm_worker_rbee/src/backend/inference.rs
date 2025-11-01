// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Inference implementation with proper error handling

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
use crate::narration::{
    ACTION_CACHE_RESET, ACTION_INFERENCE_COMPLETE, ACTION_INFERENCE_START, ACTION_MODEL_LOAD,
    ACTION_TOKENIZE, ACTION_TOKEN_GENERATE, ACTION_WARMUP,
};
use crate::token_output_stream::TokenOutputStream;
use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use observability_narration_core::n;
use std::path::Path;
use tokenizers::Tokenizer;

/// Candle inference backend using candle-transformers models
///
/// TEAM-009: Complete rewrite to use Candle's models directly
/// instead of building layers from scratch.
/// TEAM-015: Refactored into focused modules
/// TEAM-017: Changed to enum pattern for Candle idiomaticity
/// TEAM-149: Made fields pub(crate) for `generation_engine` access
pub struct CandleInferenceBackend {
    pub(crate) model: Model,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) device: Device,
    model_size_bytes: u64,
}

impl CandleInferenceBackend {
    /// Load model from `SafeTensors` with auto-detected architecture
    ///
    /// TEAM-009: Uses candle-transformers models directly
    /// TEAM-015: Delegates to `model_loader` module
    /// TEAM-017: Uses model factory with enum pattern (Candle-idiomatic)
    /// TEAM-NARRATION-FIX: Device is now compile-time determined by feature flags
    #[cfg(feature = "cpu")]
    pub fn load(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        let device = Device::Cpu;

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

        n!(ACTION_MODEL_LOAD, "Loaded {} model ({} MB, vocab: {})", model.architecture(), model_size_bytes / 1_000_000, model.vocab_size());

        Ok(Self { model, tokenizer, device, model_size_bytes })
    }

    #[cfg(feature = "cuda")]
    pub fn load(model_path: &str, gpu_id: usize) -> Result<Self> {
        let path = Path::new(model_path);
        let device = Device::new_cuda(gpu_id)?;

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

        n!(ACTION_MODEL_LOAD, "Loaded {} model ({} MB, vocab: {})", model.architecture(), model_size_bytes / 1_000_000, model.vocab_size());

        Ok(Self { model, tokenizer, device, model_size_bytes })
    }

    #[cfg(feature = "metal")]
    pub fn load(model_path: &str, gpu_id: usize) -> Result<Self> {
        let path = Path::new(model_path);
        let device = Device::new_metal(gpu_id)?;

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

        n!(ACTION_MODEL_LOAD, "Loaded {} model ({} MB, vocab: {})", model.architecture(), model_size_bytes / 1_000_000, model.vocab_size());

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
    /// TEAM-021: Warmup uses inference cache, will be reset before actual inference
    ///
    /// 🎯 TEAM-021: Warmup doesn't pollute inference - cache reset handles it!
    pub fn warmup(&mut self) -> Result<()> {
        tracing::info!("Starting GPU warmup...");

        n!(ACTION_WARMUP, "Starting GPU warmup");

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
        // TEAM-021: This uses the inference cache, but execute() will reset it before use
        let _logits = self.model.forward(&input_ids, 0).context("Warmup forward pass failed")?;

        let duration = start.elapsed();
        tracing::info!(
            duration_ms = duration.as_millis(),
            "GPU warmup complete (cache will be reset before inference)"
        );

        n!(ACTION_WARMUP, "GPU warmup complete ({} ms)", duration.as_millis());

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

        n!(ACTION_INFERENCE_START, "Starting inference (prompt: {} chars, max_tokens: {}, temp: {})", prompt.len(), config.max_tokens, config.temperature);

        // Tokenize prompt
        let encoding =
            self.tokenizer.encode(prompt, true).map_err(|e| format!("Tokenization failed: {e}"))?;
        let mut tokens = encoding.get_ids().to_vec();

        tracing::debug!(prompt_tokens = tokens.len(), "Prompt tokenized");

        n!(ACTION_TOKENIZE, "Tokenized prompt ({} tokens)", tokens.len());

        // TEAM-021: Reset cache to clear warmup pollution
        // Warmup leaves KV pairs in cache, causing mask broadcasting errors
        // 🎯 TEAM-021 Victory: Clean cache = no mask mismatch!
        self.model.reset_cache().context("Failed to reset cache before inference")?;
        tracing::debug!("Cache reset before inference to clear warmup pollution");

        n!(ACTION_CACHE_RESET, "Reset KV cache before inference to clear warmup pollution");

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
            let logits = self
                .model
                .forward(&input_ids, pos_usize)
                .map_err(|e| format!("Forward pass failed: {e}"))?;

            // TEAM-009: Log output device residency
            if pos == 0 {
                tracing::debug!(
                    output_device = ?logits.device(),
                    expected_device = ?self.device,
                    "Device residency check: output tensor"
                );
            }

            // Get logits for last position
            let logits = logits.squeeze(0).map_err(|e| format!("Failed to squeeze logits: {e}"))?;
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

            // TEAM-095: Debug logging for zero-token bug
            tracing::info!(
                pos = pos,
                next_token = next_token,
                model_eos = self.model.eos_token_id(),
                "Sampled token"
            );

            // TEAM-017: Check for EOS - try tokenizer first (Candle-idiomatic), fallback to model
            let tokenizer_eos_id = self.tokenizer.token_to_id("</s>");
            let is_eos = tokenizer_eos_id.map_or_else(
                || next_token == self.model.eos_token_id(),
                |eos_id| next_token == eos_id,
            );

            // TEAM-095: Debug EOS detection
            tracing::info!(
                pos = pos,
                next_token = next_token,
                tokenizer_eos = ?tokenizer_eos_id,
                model_eos = self.model.eos_token_id(),
                is_eos = is_eos,
                "EOS check result"
            );

            if is_eos {
                tracing::warn!(
                    pos = pos,
                    next_token = next_token,
                    "EOS token detected - stopping generation"
                );
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

                n!(ACTION_TOKEN_GENERATE, "Generated {} tokens", pos + 1);
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

        // TEAM-089: Join generated text for logging
        let full_text = generated_text.join("");
        let text_preview = if full_text.len() > 100 {
            format!("{}...", &full_text[..100])
        } else {
            full_text.clone()
        };

        tracing::info!(
            tokens_generated = generated_tokens.len(),
            duration_ms = duration_ms,
            tokens_per_sec = tokens_per_sec,
            text_preview = %text_preview,
            "Inference completed"
        );

        // TEAM-089: Narrate the actual answer (CRITICAL for debugging)
        n!(ACTION_INFERENCE_COMPLETE, "Generated: \"{}\" ({} tokens, {} ms, {} tok/s)", text_preview, generated_tokens.len(), duration_ms, tokens_per_sec);

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
