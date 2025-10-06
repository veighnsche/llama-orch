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
use worker_tokenizer::Tokenizer;

/// CUDA-based inference backend with real GPU inference
pub struct CudaInferenceBackend {
    model: Arc<Model>,
    #[allow(dead_code)]
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
        tracing::info!("🔧 Creating CudaInferenceBackend with REAL inference");
        tracing::info!("   Model path: {}", model_path);
        
        // Parse GGUF metadata for model config
        let metadata = GGUFMetadata::from_file(model_path)
            .map_err(|e| {
                tracing::error!("❌ Failed to parse GGUF metadata: {}", e);
                format!("Failed to parse GGUF metadata: {}", e)
            })?;
        
        tracing::info!("✅ GGUF metadata parsed");
        
        // Load tokenizer from GGUF
        let tokenizer = Tokenizer::from_gguf(model_path)
            .map_err(|e| {
                tracing::error!("❌ Failed to load tokenizer: {}", e);
                format!("Failed to load tokenizer: {}", e)
            })?;
        
        tracing::info!("✅ Tokenizer loaded");
        tracing::info!("🎉 CudaInferenceBackend created successfully - REAL INFERENCE ENABLED");
        
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
        tracing::info!("🚀 REAL INFERENCE STARTING");
        tracing::info!("   Prompt: {}", prompt);
        
        // Encode prompt to token IDs
        let token_ids = self.tokenizer.encode(prompt, true)
            .map_err(|e| {
                tracing::error!("❌ Tokenization failed: {}", e);
                format!("Tokenization failed: {}", e)
            })?;
        
        tracing::info!("✅ Tokenized to {} tokens", token_ids.len());
        
        if token_ids.is_empty() {
            return Err("Empty token sequence".into());
        }
        
        // Get model configuration from metadata
        let vocab_size = match self.metadata.vocab_size() {
            Ok(size) => size as u32,
            Err(_) => {
                // Fallback: derive from token_embd.weight tensor
                tracing::warn!("tokenizer.ggml.tokens not found, deriving vocab_size from token_embd.weight");
                let tensors = worker_gguf::GGUFMetadata::parse_tensors(&self.model_path)
                    .map_err(|e| format!("Failed to parse tensors: {}", e))?;
                
                tensors.iter()
                    .find(|t| t.name == "token_embd.weight")
                    .and_then(|t| t.dimensions.last())
                    .map(|&d| d as u32)
                    .ok_or_else(|| "Cannot determine vocab_size".to_string())?
            }
        };
        let hidden_dim = self.metadata.hidden_dim()? as u32;
        let num_layers = self.metadata.num_layers()? as u32;
        let num_heads = self.metadata.num_heads()? as u32;
        let num_kv_heads = self.metadata.num_kv_heads()? as u32;
        let context_length = self.metadata.context_length()? as u32;
        
        tracing::info!("Model config: vocab={}, hidden={}, layers={}, heads={}, kv_heads={}", 
            vocab_size, hidden_dim, num_layers, num_heads, num_kv_heads);
        
        // Calculate head_dim and derive ffn_dim from GGUF tensors (do not assume 4x)
        let head_dim = hidden_dim / num_heads;
        let ffn_dim = match worker_gguf::GGUFMetadata::parse_tensors(&self.model_path) {
            Ok(tensors) => {
                // Prefer ffn_up.weight; fall back to ffn_gate.weight
                let mut derived: Option<u32> = None;
                for t in &tensors {
                    if t.name == "blk.0.ffn_up.weight" || t.name == "blk.0.ffn_gate.weight" {
                        if let Some(&d0) = t.dimensions.first() { derived = Some(d0 as u32); break; }
                    }
                }
                derived.unwrap_or(hidden_dim * 4)
            }
            Err(_) => hidden_dim * 4,
        };
        
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
        // Feed all prompt tokens through the transformer to build KV cache
        // We call generate_token() to run the forward pass, but we ignore the sampled output
        // and feed the next prompt token instead (teacher forcing)
        for (i, &token_id) in token_ids.iter().enumerate() {
            if i < token_ids.len() - 1 {
                // Prefill: run forward pass with this token, ignore sampled output
                let _ = inference.generate_token(
                    token_id,
                    0.0, // Greedy (doesn't matter, we ignore output)
                    0,
                    1.0,
                    config.seed,
                )?;
                // Continue with next prompt token (teacher forcing)
            }
        }
        
        // Start generation from the last prompt token
        let mut current_token = *token_ids.last().unwrap();
        
        // Generate new tokens (decode phase)
        let mut executor = InferenceExecutor::new(config.clone());
        let mut token_idx = 0;
        let eos_token_id = self.metadata.eos_token_id().unwrap_or(151643); // Qwen2.5 EOS
        
        eprintln!("\n🎨 GENERATING TOKENS:");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        while token_idx < config.max_tokens {
            // Generate next token
            let next_token_id = inference.generate_token(
                current_token,
                0.8, // Slightly higher temperature for more creativity
                40, // Lower top_k to reduce repetition
                0.9, // Lower top_p for more focused sampling
                config.seed.wrapping_add(token_idx as u64),
            )?;
            
            // Check for EOS
            if next_token_id == eos_token_id {
                break;
            }
            
            // Decode token to text
            let token_text = self.tokenizer.decode(&[next_token_id], false)
                .map_err(|e| format!("Detokenization failed: {}", e))?;
            
            // Debug: show token ID for first few tokens
            if token_idx < 5 {
                eprintln!("\n[Token {}] ID: {} = {:?}", token_idx, next_token_id, token_text);
            }
            
            // Print token to console in real-time
            eprint!("{}", token_text);
            
            executor.add_token(token_text, token_idx);
            current_token = next_token_id;
            token_idx += 1;
        }
        
        eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("✅ Generated {} tokens\n", token_idx);
        
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
