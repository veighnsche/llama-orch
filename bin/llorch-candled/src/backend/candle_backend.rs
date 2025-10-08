//! Candle-based inference backend
//!
//! Implements InferenceBackend trait from worker-http for Llama inference.
//! Uses candle-transformers Llama directly (TEAM-008 recommendation).
//!
//! Created by: TEAM-000
//! Modified by: TEAM-009
//! Modified by: TEAM-014 (GPU warmup, LogitsProcessor, TokenOutputStream)

use async_trait::async_trait;
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use anyhow::{Result, Context, bail};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, Config, Cache, LlamaEosToks};
use candle_transformers::generation::{LogitsProcessor, Sampling}; // TEAM-014: For proper sampling
use tokenizers::Tokenizer;
use std::path::Path;
use crate::token_output_stream::TokenOutputStream; // TEAM-014: For proper space handling

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
        
        // TEAM-011: Fixed directory scanning bug - was using parent instead of path
        let (parent, safetensor_files) = if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            // Single file: use its parent directory
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            (parent, vec![path.to_path_buf()])
        } else if path.is_dir() {
            // Directory: scan for .safetensors files
            let mut files = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    files.push(entry_path);
                }
            }
            (path, files)
        } else {
            bail!("Path must be a .safetensors file or directory containing .safetensors files");
        };

        if safetensor_files.is_empty() {
            bail!("No safetensors files found at {}", path.display());
        }

        // Calculate total size
        let model_size_bytes: u64 = safetensor_files.iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();

        // TEAM-011: Parse config.json to determine model architecture
        let config_path = parent.join("config.json");
        let config_json: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {:?}", config_path))?
        )?;
        
        // Parse config manually since Config doesn't implement Deserialize
        let hidden_size = config_json["hidden_size"].as_u64()
            .context("config.json missing hidden_size")?;
        let intermediate_size = config_json["intermediate_size"].as_u64()
            .context("config.json missing intermediate_size")?;
        let num_hidden_layers = config_json["num_hidden_layers"].as_u64()
            .context("config.json missing num_hidden_layers")?;
        let num_attention_heads = config_json["num_attention_heads"].as_u64()
            .context("config.json missing num_attention_heads")?;
        let num_key_value_heads = config_json["num_key_value_heads"].as_u64()
            .unwrap_or(num_attention_heads); // Default to MHA if not specified
        let vocab_size = config_json["vocab_size"].as_u64()
            .context("config.json missing vocab_size")?;
        let rms_norm_eps = config_json["rms_norm_eps"].as_f64()
            .unwrap_or(1e-5);
        let rope_theta = config_json["rope_theta"].as_f64()
            .unwrap_or(10000.0);
        let max_position_embeddings = config_json["max_position_embeddings"].as_u64()
            .unwrap_or(2048);
        let bos_token_id = config_json["bos_token_id"].as_u64()
            .unwrap_or(1);
        let eos_token_id = config_json["eos_token_id"].as_u64()
            .unwrap_or(2);
        let tie_word_embeddings = config_json["tie_word_embeddings"].as_bool()
            .unwrap_or(false);
        
        tracing::info!(
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            num_layers = num_hidden_layers,
            num_heads = num_attention_heads,
            num_kv_heads = num_key_value_heads,
            vocab_size = vocab_size,
            "Parsed model config"
        );
        
        // Build Config from parsed values
        let config = Config {
            hidden_size: hidden_size as usize,
            intermediate_size: intermediate_size as usize,
            vocab_size: vocab_size as usize,
            num_hidden_layers: num_hidden_layers as usize,
            num_attention_heads: num_attention_heads as usize,
            num_key_value_heads: num_key_value_heads as usize,
            rms_norm_eps,
            rope_theta: rope_theta as f32,
            max_position_embeddings: max_position_embeddings as usize,
            bos_token_id: Some(bos_token_id as u32),
            eos_token_id: Some(LlamaEosToks::Single(eos_token_id as u32)),
            rope_scaling: None,
            tie_word_embeddings,
            use_flash_attn: false, // CPU doesn't support flash attention
        };

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

    /// Create LogitsProcessor from SamplingConfig
    ///
    /// TEAM-014: Use Candle's battle-tested LogitsProcessor instead of custom sampling
    fn create_logits_processor(&self, config: &SamplingConfig) -> LogitsProcessor {
        let temperature = if config.temperature == 0.0 {
            None
        } else {
            Some(config.temperature as f64)
        };

        let sampling = match (temperature, config.top_k, config.top_p) {
            (None, _, _) => Sampling::ArgMax,
            (Some(temp), 0, p) if p >= 1.0 => Sampling::All { temperature: temp },
            (Some(temp), 0, p) => Sampling::TopP { p: p as f64, temperature: temp },
            (Some(temp), k, p) if p >= 1.0 => Sampling::TopK { k: k as usize, temperature: temp },
            (Some(temp), k, p) => Sampling::TopKThenTopP { k: k as usize, p: p as f64, temperature: temp },
        };

        LogitsProcessor::from_sampling(config.seed, sampling)
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
    /// TEAM-014: Added warmup support
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
        let mut logits_processor = self.create_logits_processor(config);

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
