//! CUDA InferenceBackend implementation
//!
//! Implements the worker-http InferenceBackend trait for CUDA models.

use worker_http::backend::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use crate::cuda::Model;
use crate::inference_executor::InferenceExecutor;
use async_trait::async_trait;
use std::sync::Arc;

/// CUDA-based inference backend
pub struct CudaInferenceBackend {
    model: Arc<Model>,
}

impl CudaInferenceBackend {
    pub fn new(model: Model) -> Self {
        Self {
            model: Arc::new(model),
        }
    }
}

#[async_trait]
impl InferenceBackend for CudaInferenceBackend {
    async fn execute(
        &self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        // Start CUDA inference
        let mut inference = self.model.start_inference(
            prompt,
            config.max_tokens as u32,
            config.temperature,
            config.seed,
        )?;
        
        // Generate tokens
        let mut executor = InferenceExecutor::new(config.clone());
        let mut token_idx = 0;
        
        while let Ok(Some((token, _id))) = inference.next_token() {
            executor.add_token(token, token_idx);
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
