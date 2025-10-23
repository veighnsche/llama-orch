//! Backend detection and capabilities
//!
//! Created by: TEAM-052
//!
//! Detects available compute backends (CUDA, Metal, CPU) and their device counts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;

/// Supported compute backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal
    Metal,
    /// CPU fallback
    Cpu,
}

impl Backend {
    /// Get backend name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Backend capabilities for a machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Available backends
    pub backends: Vec<Backend>,
    /// Device count per backend
    pub devices: HashMap<Backend, u32>,
}

impl BackendCapabilities {
    /// Create empty capabilities
    pub fn none() -> Self {
        Self {
            backends: vec![Backend::Cpu], // CPU always available
            devices: [(Backend::Cpu, 1)].iter().cloned().collect(),
        }
    }

    /// Check if backend is available
    pub fn has_backend(&self, backend: Backend) -> bool {
        self.backends.contains(&backend)
    }

    /// Get device count for backend
    pub fn device_count(&self, backend: Backend) -> u32 {
        self.devices.get(&backend).copied().unwrap_or(0)
    }

    /// Get total device count across all backends
    pub fn total_devices(&self) -> u32 {
        self.devices.values().sum()
    }

    /// Convert to JSON-compatible format for registry
    pub fn to_json_strings(&self) -> (String, String) {
        let backends_json =
            serde_json::to_string(&self.backends).unwrap_or_else(|_| "[]".to_string());
        let devices_json =
            serde_json::to_string(&self.devices).unwrap_or_else(|_| "{}".to_string());
        (backends_json, devices_json)
    }

    /// Parse from JSON strings
    ///
    /// # Errors
    ///
    /// Returns an error if JSON parsing fails
    pub fn from_json_strings(backends_json: &str, devices_json: &str) -> Result<Self> {
        let backends: Vec<Backend> = serde_json::from_str(backends_json).map_err(|e| {
            crate::error::GpuError::Other(format!("Failed to parse backends: {}", e))
        })?;
        let devices: HashMap<Backend, u32> = serde_json::from_str(devices_json).map_err(|e| {
            crate::error::GpuError::Other(format!("Failed to parse devices: {}", e))
        })?;
        Ok(Self { backends, devices })
    }
}

/// Detect all available backends on this machine
pub fn detect_backends() -> BackendCapabilities {
    let mut backends = Vec::new();
    let mut devices = HashMap::new();

    // Always have CPU
    backends.push(Backend::Cpu);
    devices.insert(Backend::Cpu, 1);

    // Detect CUDA
    if let Some(cuda_count) = detect_cuda_devices() {
        if cuda_count > 0 {
            backends.push(Backend::Cuda);
            devices.insert(Backend::Cuda, cuda_count);
        }
    }

    // Detect Metal (macOS only)
    #[cfg(target_os = "macos")]
    {
        if let Some(metal_count) = detect_metal_devices() {
            if metal_count > 0 {
                backends.push(Backend::Metal);
                devices.insert(Backend::Metal, metal_count);
            }
        }
    }

    BackendCapabilities { backends, devices }
}

/// Detect CUDA device count
fn detect_cuda_devices() -> Option<u32> {
    // Try nvidia-smi
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let count = stdout.lines().filter(|line| !line.trim().is_empty()).count();
    Some(count as u32)
}

/// Detect Metal device count (macOS only)
#[cfg(target_os = "macos")]
fn detect_metal_devices() -> Option<u32> {
    // On macOS, Metal is available if we can run system_profiler
    let output =
        std::process::Command::new("system_profiler").args(["SPDisplaysDataType"]).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Count "Chipset Model:" entries which indicate GPUs
    let count = stdout.lines().filter(|line| line.contains("Chipset Model:")).count();

    if count > 0 {
        Some(count as u32)
    } else {
        // Fallback: assume 1 Metal device if system_profiler succeeded
        Some(1)
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_metal_devices() -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_as_str() {
        assert_eq!(Backend::Cuda.as_str(), "cuda");
        assert_eq!(Backend::Metal.as_str(), "metal");
        assert_eq!(Backend::Cpu.as_str(), "cpu");
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", Backend::Cuda), "cuda");
        assert_eq!(format!("{}", Backend::Metal), "metal");
        assert_eq!(format!("{}", Backend::Cpu), "cpu");
    }

    #[test]
    fn test_capabilities_none() {
        let caps = BackendCapabilities::none();
        assert_eq!(caps.backends.len(), 1);
        assert!(caps.has_backend(Backend::Cpu));
        assert_eq!(caps.device_count(Backend::Cpu), 1);
    }

    #[test]
    fn test_capabilities_has_backend() {
        let mut caps = BackendCapabilities::none();
        assert!(caps.has_backend(Backend::Cpu));
        assert!(!caps.has_backend(Backend::Cuda));

        caps.backends.push(Backend::Cuda);
        caps.devices.insert(Backend::Cuda, 2);
        assert!(caps.has_backend(Backend::Cuda));
    }

    #[test]
    fn test_capabilities_device_count() {
        let mut caps = BackendCapabilities::none();
        caps.backends.push(Backend::Cuda);
        caps.devices.insert(Backend::Cuda, 2);

        assert_eq!(caps.device_count(Backend::Cpu), 1);
        assert_eq!(caps.device_count(Backend::Cuda), 2);
        assert_eq!(caps.device_count(Backend::Metal), 0);
    }

    #[test]
    fn test_capabilities_total_devices() {
        let mut caps = BackendCapabilities::none();
        caps.backends.push(Backend::Cuda);
        caps.devices.insert(Backend::Cuda, 2);

        assert_eq!(caps.total_devices(), 3); // 1 CPU + 2 CUDA
    }

    #[test]
    fn test_capabilities_json_roundtrip() {
        let mut caps = BackendCapabilities::none();
        caps.backends.push(Backend::Cuda);
        caps.devices.insert(Backend::Cuda, 2);

        let (backends_json, devices_json) = caps.to_json_strings();
        let parsed = BackendCapabilities::from_json_strings(&backends_json, &devices_json).unwrap();

        assert_eq!(parsed.backends.len(), caps.backends.len());
        assert_eq!(parsed.device_count(Backend::Cpu), 1);
        assert_eq!(parsed.device_count(Backend::Cuda), 2);
    }

    #[test]
    fn test_detect_backends_has_cpu() {
        let caps = detect_backends();
        // CPU should always be available
        assert!(caps.has_backend(Backend::Cpu));
        assert!(caps.device_count(Backend::Cpu) >= 1);
    }
}
