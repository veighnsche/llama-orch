//! World state for worker-common BDD tests

use cucumber::World;
use worker_common::{SamplingConfig, WorkerError};

#[derive(Debug, Default, World)]
pub struct CommonWorld {
    // Sampling config
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub min_p: Option<f32>,
    pub sampling_config: Option<SamplingConfig>,
    pub sampling_mode: Option<String>,
    pub has_advanced_sampling: Option<bool>,

    // Error handling
    pub error_type: Option<String>,
    pub worker_error: Option<WorkerError>,
    pub is_retriable: Option<bool>,
    pub status_code: Option<u16>,

    // Ready callback
    pub has_callback: bool,
    pub memory_bytes: Option<u64>,
    pub memory_architecture: Option<String>,
    pub callback_sent: bool,
}
