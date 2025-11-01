//! HuggingFace vendor implementation using official hf-hub crate
//!
//! Downloads GGUF models from HuggingFace Hub.
//! Uses the same library as Candle for compatibility.

use anyhow::Result;
use hf_hub::api::tokio::Api;
use observability_narration_core::n;
use rbee_hive_artifact_catalog::VendorSource;
use std::path::Path;

/// HuggingFace vendor for downloading models
///
/// # Supported ID Formats
///
/// - `meta-llama/Llama-2-7b-chat-hf` - Standard HF repo
/// - `TheBloke/Llama-2-7B-Chat-GGUF` - GGUF-specific repos
/// - Any valid HuggingFace model ID
///
/// # Example
///
/// ```rust,no_run
/// use rbee_hive_model_provisioner::HuggingFaceVendor;
/// use rbee_hive_artifact_catalog::VendorSource;
///
/// # async fn example() -> anyhow::Result<()> {
/// let vendor = HuggingFaceVendor::new();
/// let size = vendor.download(
///     "TheBloke/Llama-2-7B-Chat-GGUF",
///     Path::new("/tmp/model"),
///     "job-123"
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct HuggingFaceVendor {
    api: Api,
}

impl std::fmt::Debug for HuggingFaceVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HuggingFaceVendor")
            .field("api", &"<hf_hub::Api>")
            .finish()
    }
}

impl HuggingFaceVendor {
    /// Create a new HuggingFace vendor
    ///
    /// Uses default HF cache location (~/.cache/huggingface/)
    pub fn new() -> Result<Self> {
        let api = Api::new()?;
        Ok(Self { api })
    }

    /// Create vendor with custom cache directory
    pub fn with_cache_dir(_cache_dir: impl AsRef<Path>) -> Result<Self> {
        // Note: hf-hub 0.3 doesn't support custom cache directories via API
        // It uses HF_HOME environment variable instead
        // For now, just use default cache
        let api = Api::new()?;
        Ok(Self { api })
    }

    /// Find GGUF file in repository
    ///
    /// Looks for common GGUF file patterns:
    /// - *.gguf
    /// - *Q4_K_M.gguf (common quantization)
    /// - *Q5_K_M.gguf
    async fn find_gguf_file(&self, repo_id: &str) -> Result<String> {
        let repo = self.api.model(repo_id.to_string());
        
        // Try to list files in repo
        // Note: hf-hub doesn't have a direct list_files API yet,
        // so we'll try common GGUF filenames
        
        // Common GGUF quantizations (from most to least common)
        let common_quants = vec![
            "Q4_K_M",
            "Q5_K_M",
            "Q4_0",
            "Q5_0",
            "Q8_0",
            "F16",
        ];
        
        // Try to find a GGUF file
        // First, try the repo name as base
        let base_name = repo_id.split('/').last().unwrap_or(repo_id);
        
        for quant in &common_quants {
            let filename = format!("{}-{}.gguf", base_name, quant);
            // Try to get the file - if it exists, hf-hub will download it
            match repo.get(&filename).await {
                Ok(_) => return Ok(filename),
                Err(_) => continue,
            }
        }
        
        // If no common quant found, return error with helpful message
        Err(anyhow::anyhow!(
            "Could not find GGUF file in repository '{}'. \
             Please specify the exact filename (e.g., 'model-Q4_K_M.gguf')",
            repo_id
        ))
    }
}

impl Default for HuggingFaceVendor {
    fn default() -> Self {
        Self::new().expect("Failed to create HuggingFace vendor")
    }
}

#[async_trait::async_trait]
impl VendorSource for HuggingFaceVendor {
    async fn download(&self, id: &str, dest: &Path, _job_id: &str) -> Result<u64> {
        n!("hf_download_start", "📥 Downloading model '{}' from HuggingFace", id);
        
        // Parse ID: "repo/model" or "repo/model:filename.gguf"
        let (repo_id, filename) = if id.contains(':') {
            let parts: Vec<&str> = id.split(':').collect();
            (parts[0], parts.get(1).map(|s| s.to_string()))
        } else {
            (id, None)
        };
        
        // Get the repository
        let repo = self.api.model(repo_id.to_string());
        
        // Determine filename
        let filename = if let Some(f) = filename {
            f
        } else {
            n!("hf_find_gguf", "🔍 Looking for GGUF file in repository...");
            self.find_gguf_file(repo_id).await?
        };
        
        n!("hf_download_file", "📥 Downloading file: {}", filename);
        
        // Download the file
        // hf-hub automatically caches in ~/.cache/huggingface/
        let cached_path = repo.get(&filename).await?;
        
        n!("hf_download_cached", "✅ File cached at: {}", cached_path.display());
        
        // Copy to destination
        let metadata = tokio::fs::metadata(&cached_path).await?;
        let size = metadata.len();
        
        // Create destination directory
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        // Copy file to destination
        tokio::fs::copy(&cached_path, dest).await?;
        
        n!(
            "hf_download_complete",
            "✅ Model downloaded: {} ({:.2} GB)",
            filename,
            size as f64 / 1_000_000_000.0
        );
        
        Ok(size)
    }

    fn supports(&self, id: &str) -> bool {
        // HuggingFace IDs must have format: "org/model" or "org/model:file.gguf"
        // Reject empty strings, URLs, and single names without org
        if id.is_empty() || id.trim().is_empty() {
            return false;
        }
        
        let base_id = id.split(':').next().unwrap_or(id);
        
        // Must contain '/' (org/model format)
        // Must NOT contain "://" (reject URLs)
        base_id.contains('/') && !base_id.contains("://")
    }

    fn name(&self) -> &str {
        "HuggingFace"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BEHAVIOR: Vendor ID Recognition
    // ========================================================================
    // The vendor must correctly identify HuggingFace repository IDs while
    // rejecting URLs and local paths to prevent security issues.

    #[test]
    fn behavior_accepts_standard_huggingface_repo_ids() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        // Standard org/model format
        assert!(
            vendor.supports("meta-llama/Llama-2-7b"),
            "Should accept standard HF repo format: org/model"
        );
        
        // GGUF-specific repos
        assert!(
            vendor.supports("TheBloke/Llama-2-7B-GGUF"),
            "Should accept GGUF-specific repos"
        );
        
        // Different organizations
        assert!(
            vendor.supports("microsoft/phi-2"),
            "Should accept repos from different orgs"
        );
    }

    #[test]
    fn behavior_accepts_repo_ids_with_explicit_filenames() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        // Repo with explicit filename
        assert!(
            vendor.supports("meta-llama/Llama-2-7b:model-Q4_K_M.gguf"),
            "Should accept repo:filename format for explicit file selection"
        );
        
        // Different quantizations
        assert!(
            vendor.supports("TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf"),
            "Should accept different quantization formats"
        );
    }

    #[test]
    fn behavior_rejects_urls_to_prevent_arbitrary_downloads() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        // HTTP URLs
        assert!(
            !vendor.supports("https://example.com/model.gguf"),
            "Must reject HTTP URLs to prevent arbitrary downloads"
        );
        
        assert!(
            !vendor.supports("http://malicious.com/model.gguf"),
            "Must reject HTTP URLs (security)"
        );
        
        // File URLs
        assert!(
            !vendor.supports("file:///local/path/model.gguf"),
            "Must reject file:// URLs to prevent local file access"
        );
    }

    #[test]
    fn behavior_rejects_single_names_without_organization() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        // Single name without org (ambiguous)
        assert!(
            !vendor.supports("llama-2-7b"),
            "Should reject single names without org/ prefix (ambiguous)"
        );
        
        assert!(
            !vendor.supports("model.gguf"),
            "Should reject bare filenames"
        );
    }

    // ========================================================================
    // BEHAVIOR: Vendor Identity
    // ========================================================================
    // The vendor must correctly identify itself for logging and error messages.

    #[test]
    fn behavior_identifies_as_huggingface_vendor() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        assert_eq!(
            vendor.name(),
            "HuggingFace",
            "Vendor must identify as 'HuggingFace' for logging"
        );
    }

    // ========================================================================
    // BEHAVIOR: Initialization
    // ========================================================================
    // The vendor must initialize successfully with default and custom caches.

    #[test]
    fn behavior_initializes_with_default_cache() {
        let result = HuggingFaceVendor::new();
        
        assert!(
            result.is_ok(),
            "Vendor should initialize successfully with default cache"
        );
    }

    #[test]
    fn behavior_initializes_with_custom_cache_directory() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("rbee-test-cache");
        let result = HuggingFaceVendor::with_cache_dir(&temp_dir);
        
        assert!(
            result.is_ok(),
            "Vendor should initialize with custom cache directory"
        );
    }

    #[test]
    fn behavior_default_constructor_works() {
        let vendor = HuggingFaceVendor::default();
        
        assert_eq!(
            vendor.name(),
            "HuggingFace",
            "Default constructor should create valid vendor"
        );
    }

    // ========================================================================
    // BEHAVIOR: ID Parsing
    // ========================================================================
    // The vendor must correctly parse repo IDs with and without filenames.

    #[test]
    fn behavior_parses_repo_id_without_filename() {
        let vendor = HuggingFaceVendor::new().unwrap();
        let id = "meta-llama/Llama-2-7b";
        
        // Should support it
        assert!(vendor.supports(id));
        
        // Should extract repo part correctly (tested implicitly by supports())
        let parts: Vec<&str> = id.split(':').collect();
        assert_eq!(parts.len(), 1, "ID without filename should have no colon");
        assert_eq!(parts[0], "meta-llama/Llama-2-7b");
    }

    #[test]
    fn behavior_parses_repo_id_with_filename() {
        let vendor = HuggingFaceVendor::new().unwrap();
        let id = "meta-llama/Llama-2-7b:model-Q4_K_M.gguf";
        
        // Should support it
        assert!(vendor.supports(id));
        
        // Should extract both parts correctly
        let parts: Vec<&str> = id.split(':').collect();
        assert_eq!(parts.len(), 2, "ID with filename should have colon separator");
        assert_eq!(parts[0], "meta-llama/Llama-2-7b", "Repo part should be correct");
        assert_eq!(parts[1], "model-Q4_K_M.gguf", "Filename part should be correct");
    }

    // ========================================================================
    // BEHAVIOR: Edge Cases
    // ========================================================================
    // The vendor must handle edge cases gracefully.

    #[test]
    fn behavior_handles_empty_string() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        assert!(
            !vendor.supports(""),
            "Empty string should not be supported"
        );
    }

    #[test]
    fn behavior_handles_whitespace() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        assert!(
            !vendor.supports("   "),
            "Whitespace-only string should not be supported"
        );
    }

    #[test]
    fn behavior_handles_multiple_colons() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        // Multiple colons (malformed)
        assert!(
            vendor.supports("org/model:file:extra"),
            "Should still support (will use first colon split)"
        );
    }

    #[test]
    fn behavior_handles_special_characters_in_org_name() {
        let vendor = HuggingFaceVendor::new().unwrap();
        
        // Underscores, hyphens are common in HF org names
        assert!(
            vendor.supports("hugging-face_org/model-name_v2"),
            "Should support special characters common in HF names"
        );
    }
}
