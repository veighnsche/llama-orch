//! Core types for model loading

use std::path::Path;
use std::net::IpAddr;

/// Model load request
///
/// Specifies what to load and how to validate it.
#[derive(Debug, Clone)]
pub struct LoadRequest<'a> {
    /// Path to model file
    pub model_path: &'a Path,
    
    /// Expected SHA-256 hash (64 hex characters)
    /// If provided, hash verification is performed
    pub expected_hash: Option<&'a str>,
    
    /// Maximum allowed file size (bytes)
    pub max_size: usize,
    
    // Actor context for audit logging and narration
    /// Worker ID (who is loading the model)
    pub worker_id: Option<String>,
    
    /// Source IP address (where the request came from)
    pub source_ip: Option<IpAddr>,
    
    /// Correlation ID for request tracking
    pub correlation_id: Option<String>,
    
    // TODO(Post-M0): Add signature verification fields
    // pub signature: Option<&'a [u8]>,
    // pub public_key: Option<&'a PublicKey>,
}

impl<'a> LoadRequest<'a> {
    /// Create a new load request with default max size (100GB)
    pub fn new(model_path: &'a Path) -> Self {
        Self {
            model_path,
            expected_hash: None,
            max_size: 100_000_000_000, // 100GB
            worker_id: None,
            source_ip: None,
            correlation_id: None,
        }
    }
    
    /// Set expected hash for verification
    pub fn with_hash(mut self, hash: &'a str) -> Self {
        self.expected_hash = Some(hash);
        self
    }
    
    /// Set maximum file size
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }
    
    /// Set worker ID for audit logging
    pub fn with_worker_id(mut self, worker_id: String) -> Self {
        self.worker_id = Some(worker_id);
        self
    }
    
    /// Set source IP for audit logging
    pub fn with_source_ip(mut self, source_ip: IpAddr) -> Self {
        self.source_ip = Some(source_ip);
        self
    }
    
    /// Set correlation ID for request tracking
    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }
}

// TODO(Post-M0): Add GGUF metadata types per 30_dependencies.md §1.4
// #[cfg(feature = "metadata-extraction")]
// pub struct GgufMetadata {
//     pub version: u32,
//     pub tensor_count: u64,
//     pub metadata_kv: HashMap<String, MetadataValue>,
// }
//
// #[cfg(feature = "metadata-extraction")]
// pub enum MetadataValue {
//     String(String),
//     U64(u64),
//     F64(f64),
//     Bool(bool),
//     Array(Vec<MetadataValue>),
// }

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_load_request_builder() {
        let path = PathBuf::from("/models/test.gguf");
        let request = LoadRequest::new(&path)
            .with_hash("abc123")
            .with_max_size(1000);
        
        assert_eq!(request.model_path, &path);
        assert_eq!(request.expected_hash, Some("abc123"));
        assert_eq!(request.max_size, 1000);
    }
}
