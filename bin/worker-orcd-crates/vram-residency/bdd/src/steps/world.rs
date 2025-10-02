//! BDD World for vram-residency tests

use cucumber::World;
use gpu_info::GpuInfo;
use std::collections::HashMap;
use std::fmt;
use vram_residency::{SealedShard, VramError, VramManager};

#[derive(World)]
#[world(init = Self::new)]
pub struct BddWorld {
    /// VramManager instance (auto-detects GPU)
    pub manager: Option<VramManager>,

    /// GPU information (if available)
    pub gpu_info: Option<GpuInfo>,

    /// Sealed shards by ID
    pub shards: HashMap<String, SealedShard>,

    /// Last operation result
    pub last_result: Option<Result<(), VramError>>,

    /// Last error message
    pub last_error: Option<String>,

    /// Test data
    pub model_data: Vec<u8>,
    pub shard_id: String,
    pub gpu_device: u32,
    pub vram_capacity: usize,

    /// Expected values for assertions
    pub expected_error_type: Option<String>,
    pub expected_vram_bytes: Option<usize>,
}

impl BddWorld {
    pub fn new() -> Self {
        Self {
            manager: None,
            gpu_info: None,
            shards: HashMap::new(),
            last_result: None,
            last_error: None,
            model_data: Vec::new(),
            shard_id: String::new(),
            gpu_device: 0,
            vram_capacity: 10 * 1024 * 1024, // 10MB default
            expected_error_type: None,
            expected_vram_bytes: None,
        }
    }

    /// Store operation result
    pub fn store_result(&mut self, result: Result<(), VramError>) {
        match result {
            Ok(()) => {
                self.last_result = Some(Ok(()));
                self.last_error = None;
            }
            Err(e) => {
                let error_msg = e.to_string();
                self.last_result = Some(Err(e));
                self.last_error = Some(error_msg);
            }
        }
    }

    /// Check if last operation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }

    /// Check if last operation failed
    pub fn last_failed(&self) -> bool {
        matches!(self.last_result, Some(Err(_)))
    }

    /// Get last error message
    pub fn get_last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.gpu_info.as_ref().map_or(false, |info| info.available)
    }

    /// Get test mode description
    pub fn test_mode(&self) -> &str {
        if self.has_gpu() {
            "🎮 GPU"
        } else {
            "💻 Mock"
        }
    }
}

impl fmt::Debug for BddWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BddWorld")
            .field("has_manager", &self.manager.is_some())
            .field("has_gpu", &self.has_gpu())
            .field("shards_count", &self.shards.len())
            .field("last_succeeded", &self.last_succeeded())
            .finish()
    }
}

impl Default for BddWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BddWorld {
    fn drop(&mut self) {
        // Explicitly clear shards first
        self.shards.clear();
        
        // Force drop of manager to free VRAM immediately
        if let Some(manager) = self.manager.take() {
            std::mem::drop(manager);
        }
    }
}
