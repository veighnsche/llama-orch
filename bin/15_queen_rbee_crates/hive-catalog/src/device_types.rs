//! Device types for hive catalog
//!
//! Created by: TEAM-158
//!
//! Stores device capabilities (CPU, CUDA, Metal) for each hive.
//! These types match the device-detection crate's Backend enum.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Device backend type
///
/// TEAM-158: Matches device-detection crate's Backend enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceBackend {
    /// CPU (always available)
    Cpu,
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal
    Metal,
}

impl DeviceBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceBackend::Cpu => "cpu",
            DeviceBackend::Cuda => "cuda",
            DeviceBackend::Metal => "metal",
        }
    }
}

impl std::fmt::Display for DeviceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for DeviceBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "cpu" => Ok(DeviceBackend::Cpu),
            "cuda" => Ok(DeviceBackend::Cuda),
            "metal" => Ok(DeviceBackend::Metal),
            _ => Err(anyhow::anyhow!("Invalid device backend: {}", s)),
        }
    }
}

/// CPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuDevice {
    pub cores: u32,
    pub ram_gb: u32,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub vram_gb: u32,
    pub backend: DeviceBackend, // cuda or metal
}

/// Device capabilities for a hive
///
/// TEAM-158: Stores what devices are available on this hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub cpu: Option<CpuDevice>,
    pub gpus: Vec<GpuDevice>,
}

impl DeviceCapabilities {
    /// Create empty capabilities
    pub fn none() -> Self {
        Self { cpu: None, gpus: Vec::new() }
    }

    /// Check if hive has CPU
    pub fn has_cpu(&self) -> bool {
        self.cpu.is_some()
    }

    /// Check if hive has any GPUs
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// Check if hive has specific backend
    pub fn has_backend(&self, backend: DeviceBackend) -> bool {
        match backend {
            DeviceBackend::Cpu => self.has_cpu(),
            DeviceBackend::Cuda => self.gpus.iter().any(|g| g.backend == DeviceBackend::Cuda),
            DeviceBackend::Metal => self.gpus.iter().any(|g| g.backend == DeviceBackend::Metal),
        }
    }

    /// Get GPU count for specific backend
    pub fn gpu_count(&self, backend: DeviceBackend) -> usize {
        match backend {
            DeviceBackend::Cpu => 0,
            DeviceBackend::Cuda => {
                self.gpus.iter().filter(|g| g.backend == DeviceBackend::Cuda).count()
            }
            DeviceBackend::Metal => {
                self.gpus.iter().filter(|g| g.backend == DeviceBackend::Metal).count()
            }
        }
    }

    /// Get total device count (CPU + all GPUs)
    pub fn total_devices(&self) -> usize {
        let cpu_count = if self.has_cpu() { 1 } else { 0 };
        cpu_count + self.gpus.len()
    }

    /// Convert to JSON for storage
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize devices: {}", e))
    }

    /// Parse from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow::anyhow!("Failed to parse devices: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_backend_str() {
        assert_eq!(DeviceBackend::Cpu.as_str(), "cpu");
        assert_eq!(DeviceBackend::Cuda.as_str(), "cuda");
        assert_eq!(DeviceBackend::Metal.as_str(), "metal");
    }

    #[test]
    fn test_device_backend_from_str() {
        assert_eq!(DeviceBackend::from_str("cpu").unwrap(), DeviceBackend::Cpu);
        assert_eq!(DeviceBackend::from_str("cuda").unwrap(), DeviceBackend::Cuda);
        assert_eq!(DeviceBackend::from_str("metal").unwrap(), DeviceBackend::Metal);
        assert!(DeviceBackend::from_str("invalid").is_err());
    }

    #[test]
    fn test_device_capabilities_none() {
        let caps = DeviceCapabilities::none();
        assert!(!caps.has_cpu());
        assert!(!caps.has_gpu());
        assert_eq!(caps.total_devices(), 0);
    }

    #[test]
    fn test_device_capabilities_has_backend() {
        let mut caps = DeviceCapabilities::none();
        caps.cpu = Some(CpuDevice { cores: 8, ram_gb: 32 });
        caps.gpus.push(GpuDevice {
            index: 0,
            name: "RTX 3060".to_string(),
            vram_gb: 12,
            backend: DeviceBackend::Cuda,
        });

        assert!(caps.has_backend(DeviceBackend::Cpu));
        assert!(caps.has_backend(DeviceBackend::Cuda));
        assert!(!caps.has_backend(DeviceBackend::Metal));
    }

    #[test]
    fn test_device_capabilities_json_roundtrip() {
        let mut caps = DeviceCapabilities::none();
        caps.cpu = Some(CpuDevice { cores: 8, ram_gb: 32 });
        caps.gpus.push(GpuDevice {
            index: 0,
            name: "RTX 3060".to_string(),
            vram_gb: 12,
            backend: DeviceBackend::Cuda,
        });

        let json = caps.to_json().unwrap();
        let parsed = DeviceCapabilities::from_json(&json).unwrap();

        assert!(parsed.has_cpu());
        assert_eq!(parsed.gpus.len(), 1);
        assert_eq!(parsed.gpus[0].name, "RTX 3060");
    }
}
