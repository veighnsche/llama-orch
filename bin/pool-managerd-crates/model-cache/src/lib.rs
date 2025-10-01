//! model-cache — Model caching

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

pub struct ModelCache;

impl ModelCache {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}
