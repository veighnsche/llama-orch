//! World state for worker-common BDD tests

use cucumber::World;

#[derive(Debug, Default, World)]
pub struct CommonWorld {
    // Sampling config
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub has_advanced_sampling: Option<bool>,
    
    // Error handling
    pub error_type: Option<String>,
    pub is_retriable: Option<bool>,
    
    // Ready callback
    pub has_callback: bool,
    pub memory_bytes: Option<u64>,
    pub memory_architecture: Option<String>,
    pub callback_sent: bool,
}
