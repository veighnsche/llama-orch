// TEAM-269: Vendor trait for extensibility

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

/// Trait for model vendor implementations
///
/// Vendors handle downloading models from different sources:
/// - HuggingFace Hub
/// - Ollama registry
/// - Local filesystem
/// - Custom registries
///
/// Each vendor implements:
/// 1. Model ID format detection (supports_model)
/// 2. Download logic (download_model)
#[async_trait]
pub trait ModelVendor: Send + Sync {
    /// Check if this vendor supports the given model_id format
    ///
    /// # Examples
    /// - HuggingFace: "meta-llama/Llama-2-7b" (contains '/')
    /// - Ollama: "ollama:llama2" (starts with "ollama:")
    /// - Local: "file:/path/to/model" (starts with "file:")
    fn supports_model(&self, model_id: &str) -> bool;

    /// Download model to destination path
    ///
    /// # Arguments
    /// * `job_id` - Job ID for narration/SSE routing
    /// * `model_id` - Model identifier in vendor-specific format
    /// * `dest` - Destination directory for model files
    ///
    /// # Returns
    /// Total size in bytes of downloaded model
    ///
    /// # Errors
    /// - Network errors
    /// - Authentication errors
    /// - Filesystem errors
    /// - Model not found
    async fn download_model(
        &self,
        job_id: &str,
        model_id: &str,
        dest: &Path,
    ) -> Result<u64>;

    /// Get vendor name for logging/debugging
    fn vendor_name(&self) -> &'static str;
}
