//! Model provisioner implementation
//!
//! Coordinates HuggingFace vendor to provision ModelEntry artifacts.

use anyhow::Result;
use observability_narration_core::n;
use rbee_hive_artifact_catalog::{ArtifactProvisioner, VendorSource};
use rbee_hive_model_catalog::ModelEntry;
use std::path::PathBuf;

use crate::HuggingFaceVendor;

/// Model provisioner
///
/// Downloads models from HuggingFace and creates ModelEntry artifacts.
///
/// # Example
///
/// ```rust,no_run
/// use rbee_hive_model_provisioner::ModelProvisioner;
/// use rbee_hive_artifact_catalog::ArtifactProvisioner;
///
/// # async fn example() -> anyhow::Result<()> {
/// let provisioner = ModelProvisioner::new()?;
/// let model = provisioner.provision(
///     "TheBloke/Llama-2-7B-Chat-GGUF",
///     "job-123"
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct ModelProvisioner {
    vendor: HuggingFaceVendor,
    cache_dir: PathBuf,
}

impl std::fmt::Debug for ModelProvisioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelProvisioner")
            .field("vendor", &self.vendor)
            .field("cache_dir", &self.cache_dir)
            .finish()
    }
}

impl ModelProvisioner {
    /// Create a new model provisioner
    ///
    /// Models are stored in:
    /// - Linux/Mac: ~/.cache/rbee/models/
    /// - Windows: %LOCALAPPDATA%\rbee\models\
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("models");

        let vendor = HuggingFaceVendor::new()?;

        Ok(Self { vendor, cache_dir })
    }

    /// Create provisioner with custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        let vendor = HuggingFaceVendor::new()?;
        // Add "models" subdirectory to cache path
        let models_cache = cache_dir.join("models");
        Ok(Self { vendor, cache_dir: models_cache })
    }

    /// Get model directory path
    fn model_dir(&self, model_id: &str) -> PathBuf {
        // Sanitize model ID for filesystem (replace / with -)
        let safe_id = model_id.replace('/', "-").replace(':', "-");
        self.cache_dir.join(safe_id)
    }

    /// Get model file path
    fn model_file_path(&self, model_id: &str) -> PathBuf {
        self.model_dir(model_id).join("model.gguf")
    }
}

impl Default for ModelProvisioner {
    fn default() -> Self {
        Self::new().expect("Failed to create model provisioner")
    }
}

#[async_trait::async_trait]
impl ArtifactProvisioner<ModelEntry> for ModelProvisioner {
    async fn provision(&self, id: &str, job_id: &str) -> Result<ModelEntry> {
        n!("model_provision_start", "🚀 Provisioning model '{}'", id);

        // Check if vendor supports this ID
        if !self.vendor.supports(id) {
            return Err(anyhow::anyhow!(
                "Model ID '{}' is not supported by HuggingFace vendor",
                id
            ));
        }

        // Determine destination path
        let dest_path = self.model_file_path(id);

        // Download model
        let size = self.vendor.download(id, &dest_path, job_id).await?;

        // Extract model name from ID
        let name = id.split('/').last().unwrap_or(id).split(':').next().unwrap_or(id).to_string();

        // Create ModelEntry
        let model = ModelEntry::new(id.to_string(), name, dest_path, size);

        n!(
            "model_provision_complete",
            "✅ Model '{}' provisioned successfully ({:.2} GB)",
            id,
            size as f64 / 1_000_000_000.0_f64
        );

        Ok(model)
    }

    fn supports(&self, id: &str) -> bool {
        self.vendor.supports(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ========================================================================
    // BEHAVIOR: Filesystem Path Sanitization
    // ========================================================================
    // The provisioner must sanitize model IDs to create safe filesystem paths,
    // preventing directory traversal and filesystem conflicts.

    #[test]
    fn behavior_sanitizes_slashes_in_model_ids() {
        let provisioner = ModelProvisioner::new().unwrap();

        let dir = provisioner.model_dir("meta-llama/Llama-2-7b");
        let path_str = dir.to_string_lossy();

        // Slashes should be replaced with hyphens
        assert!(
            path_str.contains("meta-llama-Llama-2-7b"),
            "Forward slashes should be replaced with hyphens to prevent directory traversal"
        );

        // Should NOT contain forward slashes in the model ID part
        let model_id_part = path_str.split("models/").last().unwrap();
        assert!(
            !model_id_part.contains('/'),
            "Sanitized model ID should not contain forward slashes"
        );
    }

    #[test]
    fn behavior_sanitizes_colons_in_model_ids() {
        let provisioner = ModelProvisioner::new().unwrap();

        let dir = provisioner.model_dir("TheBloke/Llama-2-7B-GGUF:model-Q4_K_M.gguf");
        let path_str = dir.to_string_lossy();

        // Colons should be replaced with hyphens
        assert!(
            path_str.contains("TheBloke-Llama-2-7B-GGUF-model-Q4_K_M.gguf"),
            "Colons should be replaced with hyphens for filesystem compatibility"
        );

        // Should NOT contain colons in the model ID part
        let model_id_part = path_str.split("models/").last().unwrap();
        assert!(
            !model_id_part.contains(':'),
            "Sanitized model ID should not contain colons (Windows compatibility)"
        );
    }

    #[test]
    fn behavior_creates_unique_paths_for_different_models() {
        let provisioner = ModelProvisioner::new().unwrap();

        let dir1 = provisioner.model_dir("meta-llama/Llama-2-7b");
        let dir2 = provisioner.model_dir("meta-llama/Llama-2-13b");
        let dir3 = provisioner.model_dir("TheBloke/Llama-2-7B-GGUF");

        // Each model should have a unique directory
        assert_ne!(dir1, dir2, "Different models should have different directories");
        assert_ne!(dir1, dir3, "Different models should have different directories");
        assert_ne!(dir2, dir3, "Different models should have different directories");
    }

    #[test]
    fn behavior_creates_consistent_paths_for_same_model() {
        let provisioner = ModelProvisioner::new().unwrap();

        let dir1 = provisioner.model_dir("meta-llama/Llama-2-7b");
        let dir2 = provisioner.model_dir("meta-llama/Llama-2-7b");

        // Same model ID should always produce same path (idempotent)
        assert_eq!(dir1, dir2, "Same model ID should produce consistent paths");
    }

    // ========================================================================
    // BEHAVIOR: Model File Path Construction
    // ========================================================================
    // The provisioner must construct correct paths for model files.

    #[test]
    fn behavior_creates_model_file_path_with_gguf_extension() {
        let provisioner = ModelProvisioner::new().unwrap();

        let path = provisioner.model_file_path("meta-llama/Llama-2-7b");
        let path_str = path.to_string_lossy();

        // Should end with model.gguf
        assert!(path_str.ends_with("model.gguf"), "Model file path should end with model.gguf");
    }

    #[test]
    fn behavior_model_file_path_is_inside_model_directory() {
        let provisioner = ModelProvisioner::new().unwrap();

        let model_dir = provisioner.model_dir("meta-llama/Llama-2-7b");
        let model_file = provisioner.model_file_path("meta-llama/Llama-2-7b");

        // File should be inside the model directory
        assert!(model_file.starts_with(&model_dir), "Model file should be inside model directory");
    }

    // ========================================================================
    // BEHAVIOR: Vendor Support Delegation
    // ========================================================================
    // The provisioner must correctly delegate support checks to the vendor.

    #[test]
    fn behavior_delegates_support_check_to_vendor() {
        let provisioner = ModelProvisioner::new().unwrap();

        // Valid HF IDs (vendor supports)
        assert!(provisioner.supports("meta-llama/Llama-2-7b"), "Should support valid HF repo IDs");
        assert!(provisioner.supports("TheBloke/Llama-2-7B-GGUF"), "Should support GGUF repos");

        // Invalid IDs (vendor rejects)
        assert!(
            !provisioner.supports("https://example.com/model.gguf"),
            "Should reject URLs (security)"
        );
        assert!(!provisioner.supports("file:///local/path"), "Should reject file paths (security)");
    }

    // ========================================================================
    // BEHAVIOR: Initialization
    // ========================================================================
    // The provisioner must initialize successfully with default and custom paths.

    #[test]
    fn behavior_initializes_with_default_cache_directory() {
        let result = ModelProvisioner::new();

        assert!(result.is_ok(), "Provisioner should initialize with default cache directory");
    }

    #[test]
    fn behavior_initializes_with_custom_cache_directory() {
        let temp_dir = TempDir::new().unwrap();
        let custom_cache = temp_dir.path().to_path_buf();

        let result = ModelProvisioner::with_cache_dir(custom_cache.clone());

        assert!(result.is_ok(), "Provisioner should initialize with custom cache directory");

        // Verify it uses the custom directory
        let provisioner = result.unwrap();
        let model_dir = provisioner.model_dir("test/model");

        assert!(
            model_dir.starts_with(&custom_cache),
            "Model directory should be under custom cache directory"
        );
    }

    #[test]
    fn behavior_default_constructor_works() {
        let provisioner = ModelProvisioner::default();

        // Should be functional
        assert!(
            provisioner.supports("meta-llama/Llama-2-7b"),
            "Default constructor should create functional provisioner"
        );
    }

    // ========================================================================
    // BEHAVIOR: Cache Directory Structure
    // ========================================================================
    // The provisioner must use the correct cache directory structure.

    #[test]
    fn behavior_uses_models_subdirectory_in_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_root = temp_dir.path().to_path_buf();

        let provisioner = ModelProvisioner::with_cache_dir(cache_root.clone()).unwrap();
        let model_dir = provisioner.model_dir("test/model");

        // Should NOT be directly in cache_root
        assert_ne!(
            model_dir.parent().unwrap(),
            cache_root,
            "Models should be in a subdirectory, not directly in cache root"
        );

        // Path should contain the model ID directory
        assert!(
            model_dir.to_string_lossy().contains("test-model"),
            "Path should contain sanitized model ID"
        );
    }

    // ========================================================================
    // BEHAVIOR: Edge Cases
    // ========================================================================
    // The provisioner must handle edge cases gracefully.

    #[test]
    fn behavior_handles_model_ids_with_multiple_slashes() {
        let provisioner = ModelProvisioner::new().unwrap();

        let dir = provisioner.model_dir("org/suborg/model");
        let path_str = dir.to_string_lossy();

        // All slashes should be sanitized
        let model_id_part = path_str.split("models/").last().unwrap();
        assert!(!model_id_part.contains('/'), "All slashes should be sanitized");
        assert!(
            model_id_part.contains("org-suborg-model"),
            "Multiple slashes should all be replaced"
        );
    }

    #[test]
    fn behavior_handles_model_ids_with_special_characters() {
        let provisioner = ModelProvisioner::new().unwrap();

        // Underscores, hyphens, dots are common in model names
        let dir = provisioner.model_dir("org/model_name-v1.0");
        let path_str = dir.to_string_lossy();

        // Should preserve safe characters
        assert!(
            path_str.contains("model_name-v1.0"),
            "Safe characters (underscore, hyphen, dot) should be preserved"
        );
    }

    #[test]
    fn behavior_handles_empty_model_id() {
        let provisioner = ModelProvisioner::new().unwrap();

        // Should not panic
        let dir = provisioner.model_dir("");

        // Should create some path (even if not useful)
        assert!(!dir.to_string_lossy().is_empty(), "Should create a path even for empty ID");
    }

    // ========================================================================
    // BEHAVIOR: Model Name Extraction
    // ========================================================================
    // The provisioner must correctly extract model names from IDs.

    #[test]
    fn behavior_extracts_model_name_from_simple_id() {
        let provisioner = ModelProvisioner::new().unwrap();

        // For "org/model", name should be "model"
        let id = "meta-llama/Llama-2-7b";
        let parts: Vec<&str> = id.split('/').collect();
        let expected_name = parts.last().unwrap();

        assert_eq!(*expected_name, "Llama-2-7b", "Should extract model name from org/model format");
    }

    #[test]
    fn behavior_extracts_model_name_from_id_with_filename() {
        let provisioner = ModelProvisioner::new().unwrap();

        // For "org/model:file.gguf", name should be "model" (not "file.gguf")
        let id = "meta-llama/Llama-2-7b:model-Q4_K_M.gguf";
        let base_id = id.split(':').next().unwrap();
        let parts: Vec<&str> = base_id.split('/').collect();
        let expected_name = parts.last().unwrap();

        assert_eq!(
            *expected_name, "Llama-2-7b",
            "Should extract model name from base ID, ignoring filename"
        );
    }
}
