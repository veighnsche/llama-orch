//! VRAM configuration types
//!
//! Configuration for VramManager initialization.

/// Configuration for VramManager
#[derive(Debug, Clone)]
pub struct VramConfig {
    /// Worker API token (for seal key derivation)
    pub worker_api_token: String,
    
    /// GPU device index
    pub gpu_device: u32,
    
    /// Maximum model size (bytes)
    pub max_model_size: usize,
    
    /// Total VRAM capacity (bytes)
    pub total_vram: usize,
}

impl VramConfig {
    /// Create new configuration
    pub fn new(worker_api_token: String, gpu_device: u32) -> Self {
        Self {
            worker_api_token,
            gpu_device,
            max_model_size: 100 * 1024 * 1024 * 1024, // 100GB default
            total_vram: 24 * 1024 * 1024 * 1024,       // 24GB default
        }
    }
    
    /// Set maximum model size
    pub fn with_max_model_size(mut self, size: usize) -> Self {
        self.max_model_size = size;
        self
    }
    
    /// Set total VRAM capacity
    pub fn with_total_vram(mut self, size: usize) -> Self {
        self.total_vram = size;
        self
    }
}

impl Default for VramConfig {
    fn default() -> Self {
        Self {
            worker_api_token: String::new(),
            gpu_device: 0,
            max_model_size: 100 * 1024 * 1024 * 1024,
            total_vram: 24 * 1024 * 1024 * 1024,
        }
    }
}
