//! GPU information types

use serde::{Deserialize, Serialize};

use crate::error::{GpuError, Result};

/// Complete GPU information for all detected devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// True if any GPU detected
    pub available: bool,

    /// Number of GPUs detected
    pub count: usize,

    /// Per-device information
    pub devices: Vec<GpuDevice>,
}

impl GpuInfo {
    /// Create empty GpuInfo (no GPUs detected)
    pub fn none() -> Self {
        Self { available: false, count: 0, devices: Vec::new() }
    }

    /// Create GpuInfo from device list
    pub fn from_devices(devices: Vec<GpuDevice>) -> Self {
        Self { available: !devices.is_empty(), count: devices.len(), devices }
    }

    /// Get total VRAM across all GPUs (in bytes)
    pub fn total_vram_bytes(&self) -> usize {
        self.devices.iter().map(|d| d.vram_total_bytes).sum()
    }

    /// Get total free VRAM across all GPUs (in bytes)
    pub fn total_free_vram_bytes(&self) -> usize {
        self.devices.iter().map(|d| d.vram_free_bytes).sum()
    }

    /// Get GPU with most free VRAM
    pub fn best_gpu_for_workload(&self) -> Option<&GpuDevice> {
        self.devices.iter().max_by_key(|d| d.vram_free_bytes)
    }

    /// Validate device index and return device
    pub fn validate_device(&self, device: u32) -> Result<&GpuDevice> {
        self.devices.get(device as usize).ok_or(GpuError::InvalidDevice(device, self.count))
    }

    /// Check if GPU is available for operations
    pub fn can_run_gpu_operations(&self) -> bool {
        self.available && self.count > 0
    }
}

/// Information for a single GPU device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Device index (0, 1, 2, ...)
    pub index: u32,

    /// GPU name (e.g., "NVIDIA GeForce RTX 3090")
    pub name: String,

    /// Total VRAM in bytes
    pub vram_total_bytes: usize,

    /// Free VRAM in bytes
    pub vram_free_bytes: usize,

    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),

    /// PCI bus ID (e.g., "0000:01:00.0")
    pub pci_bus_id: String,
}

impl GpuDevice {
    /// Get VRAM utilization (0.0 to 1.0)
    pub fn vram_utilization(&self) -> f64 {
        if self.vram_total_bytes == 0 {
            return 0.0;
        }
        let used = self.vram_total_bytes.saturating_sub(self.vram_free_bytes);
        used as f64 / self.vram_total_bytes as f64
    }

    /// Get free VRAM in GB
    pub fn vram_free_gb(&self) -> f64 {
        self.vram_free_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    }

    /// Get total VRAM in GB
    pub fn vram_total_gb(&self) -> f64 {
        self.vram_total_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    }

    /// Get used VRAM in GB
    pub fn vram_used_gb(&self) -> f64 {
        let used = self.vram_total_bytes.saturating_sub(self.vram_free_bytes);
        used as f64 / 1024.0 / 1024.0 / 1024.0
    }

    /// Check if GPU has sufficient free VRAM
    pub fn has_free_vram(&self, required_bytes: usize) -> bool {
        self.vram_free_bytes >= required_bytes
    }

    /// Check if GPU supports compute capability
    pub fn supports_compute_capability(&self, major: u32, minor: u32) -> bool {
        self.compute_capability.0 > major
            || (self.compute_capability.0 == major && self.compute_capability.1 >= minor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device(index: u32, vram_total_mb: usize, vram_free_mb: usize) -> GpuDevice {
        GpuDevice {
            index,
            name: format!("Test GPU {}", index),
            vram_total_bytes: vram_total_mb * 1024 * 1024,
            vram_free_bytes: vram_free_mb * 1024 * 1024,
            compute_capability: (8, 6),
            pci_bus_id: format!("0000:0{}:00.0", index),
        }
    }

    #[test]
    fn test_gpu_info_none() {
        let info = GpuInfo::none();
        assert!(!info.available);
        assert_eq!(info.count, 0);
        assert!(info.devices.is_empty());
    }

    #[test]
    fn test_gpu_info_from_devices() {
        let devices =
            vec![create_test_device(0, 24576, 20000), create_test_device(1, 12288, 10000)];

        let info = GpuInfo::from_devices(devices);
        assert!(info.available);
        assert_eq!(info.count, 2);
    }

    #[test]
    fn test_total_vram_bytes() {
        let devices = vec![
            create_test_device(0, 24576, 20000), // 24GB
            create_test_device(1, 12288, 10000), // 12GB
        ];

        let info = GpuInfo::from_devices(devices);
        let expected = (24576 + 12288) * 1024 * 1024;
        assert_eq!(info.total_vram_bytes(), expected);
    }

    #[test]
    fn test_best_gpu_for_workload() {
        let devices = vec![
            create_test_device(0, 24576, 10000), // 10GB free
            create_test_device(1, 12288, 11000), // 11GB free (best)
        ];

        let info = GpuInfo::from_devices(devices);
        let best = info.best_gpu_for_workload().unwrap();
        assert_eq!(best.index, 1);
    }

    #[test]
    fn test_validate_device() {
        let devices =
            vec![create_test_device(0, 24576, 20000), create_test_device(1, 12288, 10000)];

        let info = GpuInfo::from_devices(devices);

        // Valid device
        assert!(info.validate_device(0).is_ok());
        assert!(info.validate_device(1).is_ok());

        // Invalid device
        assert!(info.validate_device(2).is_err());
        assert!(info.validate_device(999).is_err());
    }

    #[test]
    fn test_vram_utilization() {
        let device = create_test_device(0, 24576, 12288); // 50% used

        let utilization = device.vram_utilization();
        assert!((utilization - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_vram_gb_conversions() {
        let device = create_test_device(0, 24576, 12288);

        assert!((device.vram_total_gb() - 24.0).abs() < 0.1);
        assert!((device.vram_free_gb() - 12.0).abs() < 0.1);
        assert!((device.vram_used_gb() - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_has_free_vram() {
        let device = create_test_device(0, 24576, 12288); // 12GB free

        // Has enough
        assert!(device.has_free_vram(10 * 1024 * 1024 * 1024)); // 10GB

        // Not enough
        assert!(!device.has_free_vram(15 * 1024 * 1024 * 1024)); // 15GB
    }

    #[test]
    fn test_compute_capability() {
        let device = GpuDevice {
            index: 0,
            name: "RTX 3090".to_string(),
            vram_total_bytes: 24 * 1024 * 1024 * 1024,
            vram_free_bytes: 20 * 1024 * 1024 * 1024,
            compute_capability: (8, 6),
            pci_bus_id: "0000:01:00.0".to_string(),
        };

        // Supports 8.6
        assert!(device.supports_compute_capability(8, 6));

        // Supports lower versions
        assert!(device.supports_compute_capability(7, 5));
        assert!(device.supports_compute_capability(8, 0));

        // Does not support higher versions
        assert!(!device.supports_compute_capability(9, 0));
        assert!(!device.supports_compute_capability(8, 7));
    }
}
