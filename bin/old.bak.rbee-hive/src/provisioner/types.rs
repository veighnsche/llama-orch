//! Model provisioner types
//!
//! Created by: TEAM-033

use std::path::PathBuf;

/// Model provisioner - handles model downloads
pub struct ModelProvisioner {
    pub(super) base_dir: PathBuf,
}

/// Download progress information
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub stage: String,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub speed_mbps: f64,
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub reference: String,
    pub model_dir: PathBuf,
    pub total_size: i64,
    pub file_count: usize,
    pub gguf_files: Vec<PathBuf>,
}

impl ModelProvisioner {
    /// Create new model provisioner
    ///
    /// # Arguments
    /// * `base_dir` - Base directory for model storage
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// Get the base directory
    pub fn base_dir(&self) -> &PathBuf {
        &self.base_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provisioner_creation() {
        let base_dir = PathBuf::from("/tmp/models");
        let provisioner = ModelProvisioner::new(base_dir.clone());
        assert_eq!(provisioner.base_dir, base_dir);
    }

    #[test]
    fn test_provisioner_base_dir_getter() {
        let base_dir = PathBuf::from("/tmp/test");
        let provisioner = ModelProvisioner::new(base_dir.clone());
        assert_eq!(provisioner.base_dir(), &base_dir);
    }

    #[test]
    fn test_download_progress_creation() {
        let progress = DownloadProgress {
            stage: "downloading".to_string(),
            bytes_downloaded: 1024,
            bytes_total: 2048,
            speed_mbps: 10.5,
        };
        assert_eq!(progress.stage, "downloading");
        assert_eq!(progress.bytes_downloaded, 1024);
    }

    #[test]
    fn test_model_info_creation() {
        let info = ModelInfo {
            reference: "test/model".to_string(),
            model_dir: PathBuf::from("/tmp/test"),
            total_size: 1000,
            file_count: 5,
            gguf_files: vec![PathBuf::from("/tmp/test/model.gguf")],
        };
        assert_eq!(info.reference, "test/model");
        assert_eq!(info.file_count, 5);
        assert_eq!(info.gguf_files.len(), 1);
    }
}
