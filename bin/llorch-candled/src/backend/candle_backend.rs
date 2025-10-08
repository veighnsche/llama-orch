//! Candle-based inference backend
//!
//! Implements InferenceBackend trait from worker-http for Llama inference.
//! Uses candle-transformers Llama directly (TEAM-008 recommendation).
//!
//! Created by: TEAM-000
//! Modified by: TEAM-009

use async_trait::async_trait;
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use anyhow::{Result, Context, bail};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, Config, Cache};
use tokenizers::Tokenizer;
use std::path::Path;
use rand::Rng;

/// Candle inference backend using candle-transformers Llama
///
/// TEAM-009: Complete rewrite to use Candle's Llama directly
/// instead of building layers from scratch.
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
    pub fn load(model_path: &str, device: Device) -> Result<Self> {
        let path = Path::new(model_path);
        
        // Determine model format
        let is_gguf = path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

        tracing::info!(
            path = %model_path,
            format = if is_gguf { "gguf" } else { "safetensors" },
            device = ?device,
            "Loading Llama model"
        );

        // Load model based on format
        let (model, config, model_size_bytes) = if is_gguf {
            Self::load_gguf(model_path, &device)?
        } else {
            Self::load_safetensors(model_path, &device)?
        };

        // Load tokenizer (assume tokenizer.json in same directory)
        let tokenizer_path = path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer.json");
        
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

    /// Load GGUF format model
    ///
    /// TEAM-009: GGUF support deferred - use SafeTensors for now
    fn load_gguf(_path: &str, _device: &Device) -> Result<(Llama, Config, u64)> {
        bail!("GGUF support not yet implemented - use SafeTensors format instead");
    }

    /// Load SafeTensors format model
    ///
    /// TEAM-009: Uses VarBuilder + candle-transformers Llama
    fn load_safetensors(path: &str, device: &Device) -> Result<(Llama, Config, u64)> {
        let path = Path::new(path);
        
        // Find all safetensors files
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut safetensor_files = Vec::new();
        
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            safetensor_files.push(path.to_path_buf());
        } else if path.is_dir() {
            // Scan directory for .safetensors files
            for entry in std::fs::read_dir(parent)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    safetensor_files.push(path);
                }
            }
        }

        if safetensor_files.is_empty() {
            bail!("No safetensors files found at {}", path.display());
        }

        // Calculate total size
        let model_size_bytes: u64 = safetensor_files.iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();

        // Load config (assume config.json in same directory)
        let config_path = parent.join("config.json");
        let _config_json: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {:?}", config_path))?
        )?;
        
        // Parse config manually since Config doesn't implement Deserialize
        // TEAM-009: Default to 7B for now - TODO: Parse _config_json to determine actual model size
        let config = Config::config_7b_v2(false);

        // Create VarBuilder from safetensors
        let dtype = DType::F32; // Use F32 for CPU, can be F16 for GPU
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)?
        };

        // Load Llama model
        let model = Llama::load(vb, &config)
            .context("Failed to load Llama model from safetensors")?;

        Ok((model, config, model_size_bytes))
    }

    /// Sample next token from logits
    ///
    /// TEAM-009: Simple sampling implementation
    fn sample_token(&self, logits: &Tensor, config: &SamplingConfig) -> Result<u32> {
        let logits = logits.to_vec1::<f32>()?;
        
        if config.temperature == 0.0 {
            // Greedy sampling
            let token = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            return Ok(token);
        }

        // Temperature sampling
        let logits: Vec<f32> = logits.iter()
            .map(|&l| l / config.temperature)
            .collect();

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter()
            .map(|&l| (l - max_logit).exp())
            .collect();
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum).collect();

        // Sample from distribution
        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();
        let mut cumsum = 0.0;
        
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample < cumsum {
                return Ok(idx as u32);
            }
        }

        Ok((probs.len() - 1) as u32)
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        self.model_size_bytes
    }
}

#[async_trait]
impl InferenceBackend for CandleInferenceBackend {
    /// Execute inference with streaming token generation
    ///
    /// TEAM-009: Complete implementation using candle-transformers
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

        // Generate tokens
        let mut generated_tokens = Vec::new();
        let mut generated_text = Vec::new();
        let start_time = std::time::Instant::now();

        for pos in 0..config.max_tokens {
            let pos_usize = pos as usize;
            
            // Prepare input tensor
            let input_ids = if pos == 0 {
                // First iteration: use all prompt tokens
                Tensor::new(&tokens[..], &self.device)
                    .map_err(|e| format!("Failed to create input tensor: {}", e))?
            } else {
                // Subsequent iterations: only last token
                Tensor::new(&[tokens[tokens.len() - 1]], &self.device)
                    .map_err(|e| format!("Failed to create input tensor: {}", e))?
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

            // Sample next token
            let next_token = self.sample_token(&logits, config)
                .map_err(|e| format!("Sampling failed: {}", e))?;

            // Check for EOS
            if next_token == self.tokenizer.token_to_id("</s>").unwrap_or(2) {
                tracing::debug!("EOS token generated");
                break;
            }

            // Decode token
            let token_str = self.tokenizer.decode(&[next_token], true)
                .map_err(|e| format!("Detokenization failed: {}", e))?;

            generated_tokens.push(next_token);
            generated_text.push(token_str);
            tokens.push(next_token);

            // Log progress
            if (pos + 1) % 10 == 0 {
                tracing::debug!(
                    tokens_generated = pos + 1,
                    "Generation progress"
                );
            }
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
