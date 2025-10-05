//! World state for worker-compute BDD tests

use cucumber::World;
use worker_compute::ComputeError;

#[derive(Debug, Default, World)]
pub struct ComputeWorld {
    // Device initialization
    pub device_id: Option<i32>,
    pub init_error: Option<ComputeError>,
    pub init_success: bool,
    
    // Model loading
    pub model_path: Option<String>,
    pub model_loaded: bool,
    pub model_memory: Option<u64>,
    pub load_error: Option<ComputeError>,
    
    // Inference
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub inference_started: bool,
    pub tokens_generated: Vec<String>,
    pub inference_error: Option<ComputeError>,
}
