// TEAM-269: HuggingFace vendor implementation (first vendor)

use super::ModelVendor;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::path::Path;

/// HuggingFace Hub vendor
///
/// Supports model IDs in format: "org/model-name"
/// Examples:
/// - "meta-llama/Llama-2-7b-chat-hf"
/// - "mistralai/Mistral-7B-v0.1"
/// - "google/gemma-2b"
///
/// For v0.1.0: Returns placeholder implementation
/// Future: Integrate with HuggingFace Hub API
pub struct HuggingFaceVendor {
    // Future: Add HF token, cache settings, etc.
}

impl Default for HuggingFaceVendor {
    fn default() -> Self {
        Self {}
    }
}

#[async_trait]
impl ModelVendor for HuggingFaceVendor {
    fn supports_model(&self, model_id: &str) -> bool {
        // HuggingFace models contain '/' (org/model format)
        model_id.contains('/')
    }

    async fn download_model(
        &self,
        _job_id: &str,
        model_id: &str,
        _dest: &Path,
    ) -> Result<u64> {
        // TEAM-269: Placeholder for v0.1.0
        // Future implementation will:
        // 1. Use HuggingFace Hub API to fetch model metadata
        // 2. Download model files (safetensors, config.json, tokenizer, etc.)
        // 3. Report progress via narration with job_id
        // 4. Return total downloaded size

        Err(anyhow!(
            "HuggingFace download not yet implemented for model '{}'. \
             This is a placeholder for v0.1.0. \
             Future: Will integrate with HuggingFace Hub API.",
            model_id
        ))
    }

    fn vendor_name(&self) -> &'static str {
        "HuggingFace"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_huggingface_format() {
        let vendor = HuggingFaceVendor::default();

        // Valid HuggingFace formats
        assert!(vendor.supports_model("meta-llama/Llama-2-7b"));
        assert!(vendor.supports_model("mistralai/Mistral-7B-v0.1"));
        assert!(vendor.supports_model("google/gemma-2b"));

        // Invalid formats
        assert!(!vendor.supports_model("llama2"));
        assert!(!vendor.supports_model("ollama:llama2"));
        // Note: "file:/path/to/model" contains '/' so it matches HF format
        // This is OK - future LocalVendor will have higher priority
    }

    #[test]
    fn test_vendor_name() {
        let vendor = HuggingFaceVendor::default();
        assert_eq!(vendor.vendor_name(), "HuggingFace");
    }
}
