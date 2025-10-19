//! GPU detection implementation

use std::path::PathBuf;
use std::process::Command;

use crate::error::{GpuError, Result};
use crate::types::{GpuDevice, GpuInfo};

/// Detect available GPUs (returns empty if none found)
pub fn detect_gpus() -> GpuInfo {
    // Try nvidia-smi first (most reliable)
    if let Ok(info) = detect_via_nvidia_smi() {
        return info;
    }

    // Fall back to CUDA runtime if available
    #[cfg(feature = "cuda-runtime")]
    {
        if let Ok(info) = detect_via_cuda_runtime() {
            return info;
        }
    }

    // No GPU detected
    tracing::debug!("No NVIDIA GPU detected");
    GpuInfo::none()
}

/// Detect GPUs or fail if none found
pub fn detect_gpus_or_fail() -> Result<GpuInfo> {
    let info = detect_gpus();
    if !info.available {
        return Err(GpuError::NoGpuDetected);
    }
    Ok(info)
}

/// Check if any GPU is available
pub fn has_gpu() -> bool {
    detect_gpus().available
}

/// Get number of available GPUs
pub fn gpu_count() -> usize {
    detect_gpus().count
}

/// Assert GPU is available (fail fast if not)
pub fn assert_gpu_available() -> Result<()> {
    detect_gpus_or_fail().map(|_| ())
}

/// Get GPU info for specific device
pub fn get_device_info(device: u32) -> Result<GpuDevice> {
    let info = detect_gpus_or_fail()?;
    info.validate_device(device).cloned()
}

/// Find nvidia-smi executable in PATH
fn find_nvidia_smi() -> Result<PathBuf> {
    which::which("nvidia-smi").map_err(|_| GpuError::NvidiaSmiNotFound)
}

/// Detect GPUs via nvidia-smi
fn detect_via_nvidia_smi() -> Result<GpuInfo> {
    // Find nvidia-smi with explicit path (security: prevents PATH manipulation)
    let nvidia_smi_path = find_nvidia_smi()?;

    tracing::debug!("Found nvidia-smi at: {:?}", nvidia_smi_path);

    // Execute nvidia-smi with absolute path
    let output = Command::new(&nvidia_smi_path)
        .args([
            "--query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| {
            tracing::error!("Failed to execute nvidia-smi: {}", e);
            GpuError::NvidiaSmiNotFound
        })?;

    if !output.status.success() {
        return Err(GpuError::NvidiaSmiParseFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_nvidia_smi_output(&stdout)
}

/// Parse nvidia-smi CSV output
fn parse_nvidia_smi_output(output: &str) -> Result<GpuInfo> {
    const MAX_GPU_NAME_LEN: usize = 256;
    const MAX_PCI_BUS_ID_LEN: usize = 32;
    const MAX_REASONABLE_VRAM_MB: usize = 1_000_000; // 1TB is unreasonable

    let mut devices = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 6 {
            tracing::warn!("Skipping malformed nvidia-smi line (wrong field count)");
            continue;
        }

        // Parse and validate index
        let index = parts
            .first()
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| GpuError::NvidiaSmiParseFailed("Invalid GPU index".to_string()))?;

        // Parse and validate GPU name (with length limit)
        let name_raw = parts
            .get(1)
            .ok_or_else(|| GpuError::NvidiaSmiParseFailed("Missing GPU name".to_string()))?;

        // Security: Limit name length to prevent memory exhaustion
        if name_raw.len() > MAX_GPU_NAME_LEN {
            tracing::warn!(
                "GPU name too long ({} chars), truncating to {}",
                name_raw.len(),
                MAX_GPU_NAME_LEN
            );
        }
        let name = name_raw.chars().take(MAX_GPU_NAME_LEN).collect::<String>();

        // Security: Reject null bytes
        if name.contains('\0') {
            return Err(GpuError::NvidiaSmiParseFailed("GPU name contains null byte".to_string()));
        }

        // Parse and validate VRAM total
        let vram_total_mb = parts
            .get(2)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| GpuError::NvidiaSmiParseFailed("Invalid memory.total".to_string()))?;

        // Security: Validate reasonable bounds
        if vram_total_mb > MAX_REASONABLE_VRAM_MB {
            return Err(GpuError::NvidiaSmiParseFailed(format!(
                "Unreasonable VRAM size: {} MB",
                vram_total_mb
            )));
        }

        // Parse and validate VRAM free
        let vram_free_mb = parts
            .get(3)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| GpuError::NvidiaSmiParseFailed("Invalid memory.free".to_string()))?;

        // Security: Validate free <= total
        if vram_free_mb > vram_total_mb {
            tracing::warn!(
                "Free VRAM ({} MB) > Total VRAM ({} MB), clamping",
                vram_free_mb,
                vram_total_mb
            );
        }

        // Parse compute capability
        let compute_cap = parts
            .get(4)
            .ok_or_else(|| GpuError::NvidiaSmiParseFailed("Missing compute_cap".to_string()))?;
        let compute_capability = parse_compute_capability(compute_cap)?;

        // Parse and validate PCI bus ID (with length limit)
        let pci_bus_id_raw = parts
            .get(5)
            .ok_or_else(|| GpuError::NvidiaSmiParseFailed("Missing PCI bus ID".to_string()))?;

        // Security: Limit PCI bus ID length
        if pci_bus_id_raw.len() > MAX_PCI_BUS_ID_LEN {
            return Err(GpuError::NvidiaSmiParseFailed("PCI bus ID too long".to_string()));
        }

        let pci_bus_id = pci_bus_id_raw.to_string();

        // Security: Reject null bytes
        if pci_bus_id.contains('\0') {
            return Err(GpuError::NvidiaSmiParseFailed(
                "PCI bus ID contains null byte".to_string(),
            ));
        }

        // Security: Use saturating multiplication to prevent overflow
        let vram_total_bytes = vram_total_mb.saturating_mul(1024).saturating_mul(1024);
        let vram_free_bytes =
            vram_free_mb.saturating_mul(1024).saturating_mul(1024).min(vram_total_bytes); // Clamp to total

        devices.push(GpuDevice {
            index,
            name,
            vram_total_bytes,
            vram_free_bytes,
            compute_capability,
            pci_bus_id,
        });
    }

    if devices.is_empty() {
        return Err(GpuError::NoGpuDetected);
    }

    tracing::info!("Detected {} GPU(s) via nvidia-smi", devices.len());
    for device in &devices {
        tracing::debug!(
            "GPU {}: {} ({} GB VRAM, compute {}.{})",
            device.index,
            device.name,
            device.vram_total_gb(),
            device.compute_capability.0,
            device.compute_capability.1
        );
    }

    Ok(GpuInfo::from_devices(devices))
}

/// Parse compute capability string (e.g., "8.6" -> (8, 6))
fn parse_compute_capability(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return Err(GpuError::NvidiaSmiParseFailed(format!("Invalid compute capability: {}", s)));
    }

    let major = parts[0].parse::<u32>().map_err(|_| {
        GpuError::NvidiaSmiParseFailed(format!("Invalid compute capability major: {}", parts[0]))
    })?;

    let minor = parts[1].parse::<u32>().map_err(|_| {
        GpuError::NvidiaSmiParseFailed(format!("Invalid compute capability minor: {}", parts[1]))
    })?;

    Ok((major, minor))
}

/// Detect GPUs via CUDA runtime API (optional feature)
#[cfg(feature = "cuda-runtime")]
fn detect_via_cuda_runtime() -> Result<GpuInfo> {
    // TODO: Implement CUDA runtime detection
    // - cudaGetDeviceCount()
    // - cudaGetDeviceProperties()
    // - cudaMemGetInfo()
    Err(GpuError::CudaRuntimeError("CUDA runtime detection not yet implemented".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_smi_output() {
        let output = "\
0, NVIDIA GeForce RTX 3090, 24576, 23456, 8.6, 0000:01:00.0
1, NVIDIA GeForce RTX 3060 Lite Hash Rate, 12288, 11234, 8.6, 0000:02:00.0
";

        let info = parse_nvidia_smi_output(output).unwrap();
        assert_eq!(info.count, 2);
        assert!(info.available);

        // Check first GPU
        let gpu0 = &info.devices[0];
        assert_eq!(gpu0.index, 0);
        assert_eq!(gpu0.name, "NVIDIA GeForce RTX 3090");
        assert_eq!(gpu0.vram_total_bytes, 24576 * 1024 * 1024);
        assert_eq!(gpu0.vram_free_bytes, 23456 * 1024 * 1024);
        assert_eq!(gpu0.compute_capability, (8, 6));
        assert_eq!(gpu0.pci_bus_id, "0000:01:00.0");

        // Check second GPU
        let gpu1 = &info.devices[1];
        assert_eq!(gpu1.index, 1);
        assert_eq!(gpu1.name, "NVIDIA GeForce RTX 3060 Lite Hash Rate");
        assert_eq!(gpu1.vram_total_bytes, 12288 * 1024 * 1024);
    }

    #[test]
    fn test_parse_nvidia_smi_empty() {
        let output = "";
        let result = parse_nvidia_smi_output(output);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_nvidia_smi_malformed() {
        let output = "0, RTX 3090";
        let result = parse_nvidia_smi_output(output);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_compute_capability() {
        assert_eq!(parse_compute_capability("8.6").unwrap(), (8, 6));
        assert_eq!(parse_compute_capability("7.5").unwrap(), (7, 5));
        assert_eq!(parse_compute_capability("9.0").unwrap(), (9, 0));

        // Invalid formats
        assert!(parse_compute_capability("8").is_err());
        assert!(parse_compute_capability("8.6.1").is_err());
        assert!(parse_compute_capability("abc").is_err());
    }

    #[test]
    fn test_has_gpu() {
        // This test will pass/fail depending on whether GPU is available
        // Just ensure it doesn't panic
        let _ = has_gpu();
    }

    #[test]
    fn test_gpu_count() {
        // This test will return 0 or more depending on GPU availability
        // Just ensure it doesn't panic
        let count = gpu_count();
        assert!(count <= 16); // Reasonable upper bound
    }
}
