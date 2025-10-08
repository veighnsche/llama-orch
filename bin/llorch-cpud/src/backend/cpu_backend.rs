//! CPU Inference Backend Implementation
//!
//! IMPORTS: worker-http, worker-common, worker-tokenizer
//! CHECKPOINT: 0 (Foundation)

use async_trait::async_trait;
use worker_common::{InferenceResult, SamplingConfig};
use worker_http::InferenceBackend;
use worker_tokenizer::Tokenizer;

use crate::error::{Error, Result};
use crate::model::GPT2Model;

pub struct CpuInferenceBackend {
    model: GPT2Model,
    tokenizer: Tokenizer,
}

impl CpuInferenceBackend {
    /// Load model from path
    pub fn load(model_path: &str) -> Result<Self> {
        tracing::info!("Loading GPT-2 model from: {}", model_path);

        // TODO: Load tokenizer from GGUF or HuggingFace
        let tokenizer = Tokenizer::from_gguf(model_path)
            .map_err(|e| Error::ModelLoad(format!("Failed to load tokenizer: {}", e)))?;

        // TODO: Load model weights
        let model = GPT2Model::load(model_path)?;

        tracing::info!("Model loaded successfully");

        Ok(Self { model, tokenizer })
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        // TODO: Calculate actual memory usage
        // For GPT-2 Medium: ~1.5GB
        1_500_000_000
    }
}

#[async_trait]
impl InferenceBackend for CpuInferenceBackend {
    async fn execute(&self, prompt: &str, config: &SamplingConfig) -> Result<InferenceResult> {
        tracing::debug!("Executing inference for prompt: {}", prompt);

        // 1. Tokenize (worker-tokenizer)
        let tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| Error::Inference(format!("Tokenization failed: {}", e)))?;

        tracing::debug!("Tokenized to {} tokens", tokens.len());

        // 2. Generate (YOUR implementation via checkpoints)
        let output_tokens = self.model.generate(&tokens, config)?;

        tracing::debug!("Generated {} tokens", output_tokens.len());

        // 3. Decode (worker-tokenizer)
        let text = self
            .tokenizer
            .decode(&output_tokens)
            .map_err(|e| Error::Inference(format!("Decoding failed: {}", e)))?;

        // 4. Return (worker-common)
        Ok(InferenceResult::max_tokens(
            tokens,
            output_tokens,
            config.seed.unwrap_or(0),
            0, // generation_time_ms - TODO: track actual time
        ))
    }

    async fn cancel(&self, _request_id: &str) -> Result<()> {
        // CPU worker doesn't support cancellation (single-threaded)
        Ok(())
    }

    fn vram_usage(&self) -> u64 {
        // CPU worker doesn't use VRAM
        0
    }

    fn is_healthy(&self) -> bool {
        // TODO: Add health checks
        true
    }
}
