// TEAM-269: Model provisioner with vendor support
//
// This module provides model provisioning (download/install) with support
// for multiple vendors (HuggingFace, Ollama, local files, etc.).
//
// Architecture:
// - ModelProvisioner: Main entry point, routes to appropriate vendor
// - ModelVendor trait: Interface for vendor implementations
// - Vendor implementations: HuggingFace (first), others future

use crate::{ModelCatalog, ModelEntry};
use anyhow::{anyhow, Result};
use std::sync::Arc;

mod traits;
mod huggingface;

pub use traits::ModelVendor;
pub use huggingface::HuggingFaceVendor;

/// Model provisioner with vendor routing
///
/// Routes model download requests to appropriate vendor based on model_id format.
/// Currently supports:
/// - HuggingFace: model_id contains '/' (e.g., "meta-llama/Llama-2-7b")
///
/// Future vendors:
/// - Ollama: model_id starts with "ollama:" (e.g., "ollama:llama2")
/// - Local: model_id starts with "file:" (e.g., "file:/path/to/model")
pub struct ModelProvisioner {
    catalog: Arc<ModelCatalog>,
    vendors: Vec<Box<dyn ModelVendor>>,
}

impl ModelProvisioner {
    /// Create a new provisioner with default vendors
    pub fn new(catalog: Arc<ModelCatalog>) -> Self {
        let vendors: Vec<Box<dyn ModelVendor>> = vec![
            Box::new(HuggingFaceVendor::default()), // First vendor
            // Future: Box::new(OllamaVendor::default()),
            // Future: Box::new(LocalVendor::default()),
        ];

        Self { catalog, vendors }
    }

    /// Download a model using appropriate vendor
    ///
    /// # Arguments
    /// * `job_id` - Job ID for narration/SSE routing
    /// * `model_id` - Model identifier (format determines vendor)
    ///
    /// # Returns
    /// Model ID on success
    ///
    /// # Errors
    /// - No vendor supports the model_id format
    /// - Download failed
    /// - Catalog registration failed
    pub async fn download_model(&self, job_id: &str, model_id: &str) -> Result<String> {
        // Find vendor that supports this model
        let vendor = self
            .vendors
            .iter()
            .find(|v| v.supports_model(model_id))
            .ok_or_else(|| {
                anyhow!(
                    "No vendor supports model '{}'. \
                     Supported formats: HuggingFace (contains '/'), \
                     Future: Ollama ('ollama:'), Local ('file:')",
                    model_id
                )
            })?;

        // Determine download destination
        let model_path = self.catalog.model_path(model_id);

        // Download using vendor
        let size = vendor.download_model(job_id, model_id, &model_path).await?;

        // Register in catalog
        let model = ModelEntry::new(
            model_id.to_string(),
            model_id.to_string(), // Use model_id as name for now
            model_path,
            size,
        );
        self.catalog.add(model)?;

        Ok(model_id.to_string())
    }

    /// Check if a model_id is supported by any vendor
    pub fn is_supported(&self, model_id: &str) -> bool {
        self.vendors.iter().any(|v| v.supports_model(model_id))
    }

    /// Find local model path (checks if already downloaded)
    pub fn find_local_model(&self, model_id: &str) -> Option<ModelEntry> {
        self.catalog.get(model_id).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_supported() {
        // TEAM-269: Test vendor detection without needing catalog
        let vendor = HuggingFaceVendor::default();
        assert!(vendor.supports_model("meta-llama/Llama-2-7b"));
        assert!(vendor.supports_model("mistralai/Mistral-7B-v0.1"));
    }

    #[test]
    fn test_unsupported_format() {
        // TEAM-269: Test vendor detection without needing catalog
        let vendor = HuggingFaceVendor::default();
        assert!(!vendor.supports_model("random-model-name"));
        assert!(!vendor.supports_model("ollama:llama2")); // Future vendor
    }
}
