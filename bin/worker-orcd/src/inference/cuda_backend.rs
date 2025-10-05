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
        _prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        // Create executor with config
        let executor = InferenceExecutor::new(config.clone());
        
        // TODO: Implement actual CUDA inference
        // For now, return a stub result
        let result = executor.finalize();
        
        Ok(result)
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
