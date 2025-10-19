//! Resource monitoring and limits
//!
//! Monitors system resources (memory, VRAM, disk) and enforces limits
//! to prevent resource exhaustion.
//!
//! Created by: TEAM-115

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, warn};

/// System resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// Total system memory in bytes
    pub memory_total_bytes: u64,
    /// Available system memory in bytes
    pub memory_available_bytes: u64,
    /// Total disk space in bytes
    pub disk_total_bytes: u64,
    /// Available disk space in bytes
    pub disk_available_bytes: u64,
}

/// Memory limit configuration
#[derive(Debug, Clone)]
pub struct MemoryLimits {
    /// Maximum memory per worker in bytes (default: 8GB)
    pub max_worker_memory_bytes: u64,
    /// Minimum free memory to maintain in bytes (default: 2GB)
    pub min_free_memory_bytes: u64,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_worker_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            min_free_memory_bytes: 2 * 1024 * 1024 * 1024,   // 2GB
        }
    }
}

/// Disk space limits configuration
#[derive(Debug, Clone)]
pub struct DiskLimits {
    /// Minimum free disk space in bytes (default: 10GB)
    pub min_free_disk_bytes: u64,
}

impl Default for DiskLimits {
    fn default() -> Self {
        Self {
            min_free_disk_bytes: 10 * 1024 * 1024 * 1024, // 10GB
        }
    }
}

/// Check if system has sufficient memory for a worker
///
/// # Arguments
/// * `required_bytes` - Memory required by the worker
/// * `limits` - Memory limit configuration
///
/// # Returns
/// * `Ok(())` if sufficient memory available
/// * `Err` if insufficient memory
pub fn check_memory_available(required_bytes: u64, limits: &MemoryLimits) -> Result<()> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_memory();

    let available_memory = sys.available_memory();
    let total_memory = sys.total_memory();

    debug!(
        available_mb = available_memory / 1024 / 1024,
        total_mb = total_memory / 1024 / 1024,
        required_mb = required_bytes / 1024 / 1024,
        "Checking memory availability"
    );

    // Check if worker would exceed per-worker limit
    if required_bytes > limits.max_worker_memory_bytes {
        anyhow::bail!(
            "Worker requires {}MB but max per-worker limit is {}MB",
            required_bytes / 1024 / 1024,
            limits.max_worker_memory_bytes / 1024 / 1024
        );
    }

    // Check if spawning worker would leave enough free memory
    let memory_after_spawn = available_memory.saturating_sub(required_bytes);
    if memory_after_spawn < limits.min_free_memory_bytes {
        anyhow::bail!(
            "Insufficient memory: {}MB available, {}MB required, {}MB minimum free required",
            available_memory / 1024 / 1024,
            required_bytes / 1024 / 1024,
            limits.min_free_memory_bytes / 1024 / 1024
        );
    }

    Ok(())
}

/// Get current system resource information
pub fn get_resource_info() -> Result<ResourceInfo> {
    use sysinfo::{Disks, System};

    let mut sys = System::new_all();
    sys.refresh_all();

    let memory_total_bytes = sys.total_memory();
    let memory_available_bytes = sys.available_memory();

    // Get disk info for root filesystem
    let disks = Disks::new_with_refreshed_list();
    let root_disk = disks
        .iter()
        .find(|d| d.mount_point() == Path::new("/"))
        .context("Failed to find root filesystem")?;

    let disk_total_bytes = root_disk.total_space();
    let disk_available_bytes = root_disk.available_space();

    Ok(ResourceInfo {
        memory_total_bytes,
        memory_available_bytes,
        disk_total_bytes,
        disk_available_bytes,
    })
}

/// Check if sufficient disk space is available
///
/// # Arguments
/// * `required_bytes` - Disk space required
/// * `limits` - Disk limit configuration
///
/// # Returns
/// * `Ok(())` if sufficient disk space available
/// * `Err` if insufficient disk space
pub fn check_disk_space_available(required_bytes: u64, limits: &DiskLimits) -> Result<()> {
    use sysinfo::Disks;

    let disks = Disks::new_with_refreshed_list();
    let root_disk = disks
        .iter()
        .find(|d| d.mount_point() == Path::new("/"))
        .context("Failed to find root filesystem")?;

    let available_space = root_disk.available_space();

    debug!(
        available_gb = available_space / 1024 / 1024 / 1024,
        required_gb = required_bytes / 1024 / 1024 / 1024,
        "Checking disk space availability"
    );

    // Check if download would leave enough free space
    let space_after_download = available_space.saturating_sub(required_bytes);
    if space_after_download < limits.min_free_disk_bytes {
        anyhow::bail!(
            "Insufficient disk space: {}GB available, {}GB required, {}GB minimum free required",
            available_space / 1024 / 1024 / 1024,
            required_bytes / 1024 / 1024 / 1024,
            limits.min_free_disk_bytes / 1024 / 1024 / 1024
        );
    }

    Ok(())
}

/// Monitor worker memory usage and kill if exceeds limit
///
/// # Arguments
/// * `pid` - Process ID to monitor
/// * `limit_bytes` - Memory limit in bytes
///
/// # Returns
/// * `Ok(true)` if worker is within limits
/// * `Ok(false)` if worker exceeds limits (should be killed)
/// * `Err` if unable to check memory usage
pub fn check_worker_memory_usage(pid: u32, limit_bytes: u64) -> Result<bool> {
    use sysinfo::{Pid, System};

    let mut sys = System::new();
    sys.refresh_processes();

    let pid_obj = Pid::from_u32(pid);
    let process = sys.process(pid_obj).context("Process not found")?;

    let memory_usage = process.memory();

    debug!(
        pid = pid,
        memory_mb = memory_usage / 1024 / 1024,
        limit_mb = limit_bytes / 1024 / 1024,
        "Checking worker memory usage"
    );

    if memory_usage > limit_bytes {
        warn!(
            pid = pid,
            memory_mb = memory_usage / 1024 / 1024,
            limit_mb = limit_bytes / 1024 / 1024,
            "Worker exceeds memory limit"
        );
        return Ok(false);
    }

    Ok(true)
}

/// VRAM limit configuration
#[derive(Debug, Clone)]
pub struct VramLimits {
    /// Minimum free VRAM to maintain in bytes (default: 1GB)
    pub min_free_vram_bytes: u64,
}

impl Default for VramLimits {
    fn default() -> Self {
        Self {
            min_free_vram_bytes: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Check if GPU has sufficient VRAM for a worker
///
/// # Arguments
/// * `device` - GPU device index
/// * `required_bytes` - VRAM required by the worker
/// * `limits` - VRAM limit configuration
///
/// # Returns
/// * `Ok(())` if sufficient VRAM available
/// * `Err` if insufficient VRAM or GPU not available
pub fn check_vram_available(device: u32, required_bytes: u64, limits: &VramLimits) -> Result<()> {
    use gpu_info::detect_gpus;

    let gpu_info = detect_gpus();
    if !gpu_info.available {
        anyhow::bail!("No GPU detected");
    }

    let gpu_device = gpu_info
        .validate_device(device)
        .map_err(|e| anyhow::anyhow!("Invalid GPU device {}: {}", device, e))?;

    let free_vram = gpu_device.vram_free_bytes;

    debug!(
        device = device,
        free_vram_gb = free_vram as f64 / 1024.0 / 1024.0 / 1024.0,
        required_gb = required_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        "Checking VRAM availability"
    );

    // Check if worker would leave enough free VRAM
    let vram_after_spawn = (free_vram as u64).saturating_sub(required_bytes);
    if vram_after_spawn < limits.min_free_vram_bytes {
        anyhow::bail!(
            "Insufficient VRAM on GPU {}: {}GB free, {}GB required, {}GB minimum free required",
            device,
            free_vram as f64 / 1024.0 / 1024.0 / 1024.0,
            required_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            limits.min_free_vram_bytes as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }

    Ok(())
}

/// Estimate model VRAM requirements based on model size
///
/// This is a rough estimate. Actual VRAM usage depends on:
/// - Model architecture
/// - Quantization level
/// - Batch size
/// - Context length
///
/// # Arguments
/// * `model_size_bytes` - Model file size in bytes
///
/// # Returns
/// Estimated VRAM requirement in bytes
pub fn estimate_model_vram_bytes(model_size_bytes: u64) -> u64 {
    // Rule of thumb: VRAM = model_size * 1.2 (20% overhead for activations)
    // For GGUF quantized models, this is usually sufficient
    (model_size_bytes as f64 * 1.2) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_limits_default() {
        let limits = MemoryLimits::default();
        assert_eq!(limits.max_worker_memory_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(limits.min_free_memory_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_disk_limits_default() {
        let limits = DiskLimits::default();
        assert_eq!(limits.min_free_disk_bytes, 10 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_get_resource_info() {
        let info = get_resource_info().unwrap();
        assert!(info.memory_total_bytes > 0);
        assert!(info.memory_available_bytes > 0);
        assert!(info.disk_total_bytes > 0);
        assert!(info.disk_available_bytes > 0);
        assert!(info.memory_available_bytes <= info.memory_total_bytes);
        assert!(info.disk_available_bytes <= info.disk_total_bytes);
    }

    #[test]
    fn test_check_memory_available_within_limits() {
        let limits = MemoryLimits {
            max_worker_memory_bytes: 1024 * 1024, // 1MB
            min_free_memory_bytes: 512 * 1024,    // 512KB
        };

        // Request very small amount (should succeed on any system)
        let result = check_memory_available(100 * 1024, &limits); // 100KB
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_memory_available_exceeds_worker_limit() {
        let limits = MemoryLimits {
            max_worker_memory_bytes: 1024 * 1024, // 1MB
            min_free_memory_bytes: 512 * 1024,    // 512KB
        };

        // Request more than per-worker limit
        let result = check_memory_available(2 * 1024 * 1024, &limits); // 2MB
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max per-worker limit"));
    }

    #[test]
    fn test_check_disk_space_available() {
        let limits = DiskLimits {
            min_free_disk_bytes: 100 * 1024, // 100KB
        };

        // Request very small amount (should succeed on any system)
        let result = check_disk_space_available(10 * 1024, &limits); // 10KB
        assert!(result.is_ok());
    }

    #[test]
    fn test_vram_limits_default() {
        let limits = VramLimits::default();
        assert_eq!(limits.min_free_vram_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_estimate_model_vram_bytes() {
        // 4GB model should require ~4.8GB VRAM (4GB * 1.2)
        let model_size = 4 * 1024 * 1024 * 1024;
        let estimated_vram = estimate_model_vram_bytes(model_size);
        let expected = (model_size as f64 * 1.2) as u64;
        assert_eq!(estimated_vram, expected);
        
        // Check it's roughly 20% more
        assert!(estimated_vram > model_size);
        assert!(estimated_vram < model_size + model_size / 4); // Less than 25% overhead
    }

    #[test]
    fn test_check_vram_available_no_gpu() {
        let limits = VramLimits::default();
        // This test will fail if no GPU is available (expected)
        // On systems without GPU, this should return an error
        let result = check_vram_available(0, 1024 * 1024 * 1024, &limits);
        // We can't assert success/failure since it depends on hardware
        // Just verify it doesn't panic
        let _ = result;
    }
}
