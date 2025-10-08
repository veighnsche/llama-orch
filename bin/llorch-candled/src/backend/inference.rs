//! Main inference backend implementation
//!
//! Created by: TEAM-015 (refactored from candle_backend.rs)
//! Original code by: TEAM-000, TEAM-009, TEAM-011, TEAM-014

use async_trait::async_trait;
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor};
use candle_transformers::models::llama::{Llama, Config, Cache};
use tokenizers::Tokenizer;
use std::path::Path;
use crate::token_output_stream::TokenOutputStream;
use super::model_loader;
use super::sampling;

/// Candle inference backend using candle-transformers Llama
///
/// TEAM-009: Complete rewrite to use Candle's Llama directly
/// instead of building layers from scratch.
/// TEAM-015: Refactored into focused modules
pub struct CandleInferenceBackend {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
    model_size_bytes: u64,
}

impl CandleInferenceBackend {
    /// Load Llama model from SafeTensors or GGUF
    ///
    /// TEAM-009: Uses candle-transformers Llama directly
    /// TEAM-015: Delegates to model_loader module
    pub fn load(model_path: &str, device: Device) -> Result<Self> {
        let path = Path::new(model_path);
        
        // Load model using model_loader module
        let (model, config, model_size_bytes) = model_loader::load_model(model_path, &device)?;

        // TEAM-011: Load tokenizer from model directory (not path.parent())
        let tokenizer_path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join("tokenizer.json")
        };
        
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

        tracing::info!(
            vocab_size = tokenizer.get_vocab_size(true),
            model_size_mb = model_size_bytes / 1_000_000,
            "Model and tokenizer loaded successfully"
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            model_size_bytes,
        })
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        self.model_size_bytes
    }

    /// Warmup GPU with dummy inference
    ///
    /// TEAM-014: Eliminates cold start by running a single token generation.
    /// This initializes CUDA kernels and caches, preventing 9s overhead on first request.
    pub fn warmup(&self) -> Result<()> {
        tracing::info!("Starting GPU warmup...");
        let start = std::time::Instant::now();

        // Use a simple prompt for warmup
        let warmup_prompt = "Hello";
        
        // Tokenize
        let encoding = self.tokenizer.encode(warmup_prompt, true)
            .map_err(|e| anyhow::anyhow!("Warmup tokenization failed: {}", e))?;
        let tokens = encoding.get_ids();
        
        // Create input tensor
        let input_ids = Tensor::new(tokens, &self.device)
            .context("Failed to create warmup tensor")?
            .unsqueeze(0)
            .context("Failed to unsqueeze warmup tensor")?;
        
        // Initialize cache
        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)
            .context("Failed to create warmup cache")?;
        
        // Single forward pass
        let _logits = self.model.forward(&input_ids, 0, &mut cache)
            .context("Warmup forward pass failed")?;
        
        let duration = start.elapsed();
        tracing::info!(
            duration_ms = duration.as_millis(),
            "GPU warmup complete"
        );
        
        Ok(())
    }
}

#[async_trait]
impl InferenceBackend for CandleInferenceBackend {
    /// Execute inference with streaming token generation
    ///
    /// TEAM-009: Complete implementation using candle-transformers
    /// TEAM-014: Added warmup support, LogitsProcessor, TokenOutputStream
    /// TEAM-015: Refactored into focused modules
    async fn execute(
        &self,
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
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        let mut tokens = encoding.get_ids().to_vec();
        
        tracing::debug!(
            prompt_tokens = tokens.len(),
            "Prompt tokenized"
        );

        // Initialize cache
        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)
            .map_err(|e| format!("Failed to create cache: {}", e))?;

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
                    .map_err(|e| format!("Failed to create input tensor: {}", e))?
                    .unsqueeze(0)  // Add batch dimension: [seq_len] -> [1, seq_len]
                    .map_err(|e| format!("Failed to unsqueeze input tensor: {}", e))?
            } else {
                // Subsequent iterations: only last token
                Tensor::new(&[tokens[tokens.len() - 1]], &self.device)
                    .map_err(|e| format!("Failed to create input tensor: {}", e))?
                    .unsqueeze(0)  // Add batch dimension: [1] -> [1, 1]
                    .map_err(|e| format!("Failed to unsqueeze input tensor: {}", e))?
            };

            // TEAM-009: Verify device residency (log only, no comparison since Device doesn't impl PartialEq)
            if pos == 0 {
                tracing::debug!(
                    input_device = ?input_ids.device(),
                    expected_device = ?self.device,
                    "Device residency check: input tensor"
                );
            }

            // Forward pass
            let logits = self.model.forward(&input_ids, pos_usize, &mut cache)
                .map_err(|e| format!("Forward pass failed: {}", e))?;
            
            // TEAM-009: Log output device residency
            if pos == 0 {
                tracing::debug!(
                    output_device = ?logits.device(),
                    expected_device = ?self.device,
                    "Device residency check: output tensor"
                );
            }

            // Get logits for last position
            let logits = logits.squeeze(0)
                .map_err(|e| format!("Failed to squeeze logits: {}", e))?;
            let logits = if logits.dims().len() > 1 {
                logits.get(logits.dims()[0] - 1)
                    .map_err(|e| format!("Failed to get last logits: {}", e))?
            } else {
                logits
            };

            // TEAM-014: Sample next token using Candle's LogitsProcessor
            let next_token = logits_processor.sample(&logits)
                .map_err(|e| format!("Sampling failed: {}", e))?;

            // Check for EOS
            if next_token == self.tokenizer.token_to_id("</s>").unwrap_or(2) {
                tracing::debug!("EOS token generated");
                break;
            }

            // TEAM-014: Use TokenOutputStream for proper streaming decode with spaces
            if let Some(token_str) = token_stream.next_token(next_token)
                .map_err(|e| format!("Detokenization failed: {}", e))? {
                generated_text.push(token_str);
            }

            generated_tokens.push(next_token);
            tokens.push(next_token);

            // Log progress
            if (pos + 1) % 10 == 0 {
                tracing::debug!(
                    tokens_generated = pos + 1,
                    "Generation progress"
                );
            }
        }

        // TEAM-014: Get any remaining decoded bytes from token stream
        if let Some(rest) = token_stream.decode_rest()
            .map_err(|e| format!("Failed to decode rest: {}", e))? {
            generated_text.push(rest);
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        let tokens_per_sec = if duration_ms > 0 {
            (generated_tokens.len() as u64 * 1000) / duration_ms
        } else {
            0
        };

        tracing::info!(
            tokens_generated = generated_tokens.len(),
            duration_ms = duration_ms,
            tokens_per_sec = tokens_per_sec,
            "Inference completed"
        );

        Ok(InferenceResult::max_tokens(
            generated_text,
            generated_tokens,
            config.seed,
            duration_ms,
        ))
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
