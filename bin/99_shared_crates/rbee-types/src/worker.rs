//! Worker types and management
//!
//! Created by: TEAM-022

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
}

impl Backend {
    #[allow(clippy::should_implement_trait)] // TEAM-022: Not implementing FromStr trait to keep simple API
    pub fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Backend::Cpu),
            "metal" => Ok(Backend::Metal),
            "cuda" => Ok(Backend::Cuda),
            _ => Err(crate::PoolError::InvalidBackend(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub backend: Backend,
    pub model_id: String,
    pub gpu_id: Option<u32>,
    pub port: u16,
    pub pid: u32,
    pub started_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_from_str() {
        assert_eq!(Backend::from_str("cpu").unwrap(), Backend::Cpu);
        assert_eq!(Backend::from_str("metal").unwrap(), Backend::Metal);
        assert_eq!(Backend::from_str("cuda").unwrap(), Backend::Cuda);
        assert_eq!(Backend::from_str("CPU").unwrap(), Backend::Cpu);
        assert!(Backend::from_str("invalid").is_err());
    }

    #[test]
    fn test_backend_as_str() {
        assert_eq!(Backend::Cpu.as_str(), "cpu");
        assert_eq!(Backend::Metal.as_str(), "metal");
        assert_eq!(Backend::Cuda.as_str(), "cuda");
    }
}
