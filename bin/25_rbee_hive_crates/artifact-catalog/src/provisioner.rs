// TEAM-273: Generic provisioner abstraction
use crate::types::Artifact;
use anyhow::Result;
use std::path::Path;

/// Vendor source trait
///
/// Implemented by HuggingFaceVendor, GitHubReleaseVendor, LocalBuildVendor, etc.
#[async_trait::async_trait]
pub trait VendorSource: Send + Sync {
    /// Download/provision artifact from this vendor
    ///
    /// # Arguments
    /// * `id` - Artifact identifier (e.g., "meta-llama/Llama-2-7b", "v0.1.0")
    /// * `dest` - Destination path
    /// * `job_id` - Job ID for narration routing
    ///
    /// # Returns
    /// Size in bytes
    async fn download(&self, id: &str, dest: &Path, job_id: &str) -> Result<u64>;

    /// Check if this vendor supports the given artifact ID
    fn supports(&self, id: &str) -> bool;

    /// Vendor name for logging/narration
    fn name(&self) -> &str;
}

/// Artifact provisioner trait
///
/// Coordinates multiple vendor sources to provision artifacts.
#[async_trait::async_trait]
pub trait ArtifactProvisioner<T: Artifact>: Send + Sync {
    /// Provision an artifact
    ///
    /// # Arguments
    /// * `id` - Artifact identifier
    /// * `job_id` - Job ID for narration routing
    ///
    /// # Returns
    /// Provisioned artifact
    async fn provision(&self, id: &str, job_id: &str) -> Result<T>;

    /// Check if any vendor supports this artifact
    fn supports(&self, id: &str) -> bool;
}

/// Multi-vendor provisioner implementation
///
/// Routes provisioning requests to appropriate vendor based on ID format.
pub struct MultiVendorProvisioner<T: Artifact + Send + Sync> {
    vendors: Vec<Box<dyn VendorSource>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Artifact + Send + Sync> MultiVendorProvisioner<T> {
    /// Create a new multi-vendor provisioner
    pub fn new(vendors: Vec<Box<dyn VendorSource>>) -> Self {
        Self { vendors, _phantom: std::marker::PhantomData }
    }

    /// Find vendor that supports the given ID
    fn find_vendor(&self, id: &str) -> Option<&dyn VendorSource> {
        self.vendors.iter().find(|v| v.supports(id)).map(|v| v.as_ref())
    }
}

#[async_trait::async_trait]
impl<T: Artifact + Send + Sync> ArtifactProvisioner<T> for MultiVendorProvisioner<T> {
    async fn provision(&self, id: &str, _job_id: &str) -> Result<T> {
        let _vendor = self
            .find_vendor(id)
            .ok_or_else(|| anyhow::anyhow!("No vendor supports artifact '{}'", id))?;

        // Delegate to vendor
        // Note: Actual artifact creation is vendor-specific
        // This is a placeholder - concrete implementations will override
        Err(anyhow::anyhow!(
            "MultiVendorProvisioner::provision must be overridden by concrete implementation"
        ))
    }

    fn supports(&self, id: &str) -> bool {
        self.find_vendor(id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockVendor {
        name: String,
        prefix: String,
    }

    #[async_trait::async_trait]
    impl VendorSource for MockVendor {
        async fn download(&self, _id: &str, _dest: &Path, _job_id: &str) -> Result<u64> {
            Ok(1024)
        }

        fn supports(&self, id: &str) -> bool {
            id.starts_with(&self.prefix)
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_vendor_routing() {
        let vendors: Vec<Box<dyn VendorSource>> = vec![
            Box::new(MockVendor { name: "HuggingFace".to_string(), prefix: "HF:".to_string() }),
            Box::new(MockVendor { name: "GitHub".to_string(), prefix: "GH:".to_string() }),
        ];

        let provisioner = MultiVendorProvisioner::<()>::new(vendors);

        assert!(provisioner.supports("HF:meta-llama/Llama-2-7b"));
        assert!(provisioner.supports("GH:owner/repo"));
        assert!(!provisioner.supports("unknown:artifact"));
    }
}
