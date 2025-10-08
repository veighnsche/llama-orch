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
    _model: GPT2Model,
    _tokenizer: Tokenizer,
}

impl CpuInferenceBackend {
    /// Load model from path
    pub fn load(_model_path: &str) -> Result<Self> {
        // TODO: Implement model loading
        Err(Error::ModelLoad("Not implemented yet".to_string()))
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
    async fn execute(
        &self,
        _prompt: &str,
        _config: &SamplingConfig,
    ) -> std::result::Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement inference
        Err(Box::new(Error::Inference("Not implemented yet".to_string())))
    }

    async fn cancel(
        &self,
        _job_id: &str,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
