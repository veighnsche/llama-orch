//! CUDA InferenceBackend implementation
//!
//! Implements the worker-http InferenceBackend trait for CUDA models.
//! Uses real GPU inference via QwenTransformer.

use worker_http::backend::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use crate::cuda::{Model, RealInference};
use crate::inference_executor::InferenceExecutor;
use async_trait::async_trait;
use std::sync::Arc;
use worker_gguf::GGUFMetadata;
use worker_tokenizer::{Tokenizer, TokenizerBackend};

/// CUDA-based inference backend with real GPU inference
pub struct CudaInferenceBackend {
    model: Arc<Model>,
    model_path: String,
    metadata: GGUFMetadata,
    tokenizer: Tokenizer,
}

impl CudaInferenceBackend {
    /// Create new CUDA backend with real inference
    ///
    /// # Arguments
    ///
    /// * `model` - Loaded CUDA model with weights in VRAM
    /// * `model_path` - Path to GGUF file (for metadata and tokenizer)
    ///
    /// # Errors
    ///
    /// Returns error if metadata parsing or tokenizer loading fails
    pub fn new(model: Model, model_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Parse GGUF metadata for model config
        let metadata = GGUFMetadata::from_file(model_path)
            .map_err(|e| format!("Failed to parse GGUF metadata: {}", e))?;
        
        // Load tokenizer from GGUF
        let tokenizer = Tokenizer::from_gguf(model_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        
        Ok(Self {
            model: Arc::new(model),
            model_path: model_path.to_string(),
            metadata,
            tokenizer,
        })
    }
}

#[async_trait]
impl InferenceBackend for CudaInferenceBackend {
    async fn execute(
        &self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        // Encode prompt to token IDs
        let token_ids = self.tokenizer.encode(prompt, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        
        if token_ids.is_empty() {
            return Err("Empty token sequence".into());
        }
        
        // Get model configuration from metadata
        let vocab_size = self.metadata.vocab_size()? as u32;
        let hidden_dim = self.metadata.hidden_dim()? as u32;
        let num_layers = self.metadata.num_layers()? as u32;
        let num_heads = self.metadata.num_heads()? as u32;
        let num_kv_heads = self.metadata.num_kv_heads()? as u32;
        let context_length = self.metadata.context_length()? as u32;
        
        // Calculate head_dim and ffn_dim
        let head_dim = hidden_dim / num_heads;
        let ffn_dim = hidden_dim * 4; // Standard transformer FFN ratio
        
        // Initialize real inference context
        let mut inference = RealInference::init(
            &self.model,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_dim,
            context_length,
        )?;
        
        // Process prompt tokens (prefill phase)
        let mut current_token = token_ids[0];
        for &token_id in &token_ids[1..] {
            current_token = inference.generate_token(
                current_token,
                0.0, // Greedy for prefill
                0,
                1.0,
                config.seed,
            )?;
            // Verify we're following the prompt
            if current_token != token_id {
                // This is expected during prefill - we're feeding the prompt
                current_token = token_id;
            }
        }
        
        // Generate new tokens (decode phase)
        let mut executor = InferenceExecutor::new(config.clone());
        let mut token_idx = 0;
        let eos_token_id = self.metadata.eos_token_id().unwrap_or(151643); // Qwen2.5 EOS
        
        while token_idx < config.max_tokens {
            // Generate next token
            let next_token_id = inference.generate_token(
                current_token,
                config.temperature,
                50, // top_k
                0.95, // top_p
                config.seed.wrapping_add(token_idx as u64),
            )?;
            
            // Check for EOS
            if next_token_id == eos_token_id {
                break;
            }
            
            // Decode token to text
            let token_text = self.tokenizer.decode(&[next_token_id], false)
                .map_err(|e| format!("Detokenization failed: {}", e))?;
            
            executor.add_token(token_text, token_idx);
            current_token = next_token_id;
            token_idx += 1;
        }
        
        Ok(executor.finalize())
    }
    
    async fn cancel(&self, _job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement cancellation
        Ok(())
    }
    
    fn vram_usage(&self) -> u64 {
        self.model.vram_bytes()
    }
    
    fn is_healthy(&self) -> bool {
        // TODO: Implement health check
        true
    }
}
