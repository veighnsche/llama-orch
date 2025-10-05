//! Preflight checks for GPU-only enforcement and environment validations.
//!
//! Status: IMPLEMENTED (using gpu-info crate for detection).
//! Spec: ORCH-1102 (GPU-only policy), ORCH-3202 (preflight checks).
//! Usage: Called by provisioners or pool-managerd daemon at startup to enforce GPU-only policy.
//! Integration: engine-provisioner calls this before building/starting engines.
//! Tests: Unit tests exist (preflight.rs:68-98).
//! TODO: Expand to check for other required tools (git, cmake, make, nvcc) per ORCH-3203.
//! TODO: Add preflight for container runtime (podman/docker) if container provisioning is used.
//! TODO: Add preflight for model tooling (huggingface-cli, aws, oras) per ORCH-3203.

use anyhow::{Context, Result};
use gpu_info::{assert_gpu_available, has_gpu};

/// Return true if CUDA appears available (GPU detected via nvidia-smi).
///
/// Uses shared gpu-info crate for detection.
pub fn cuda_available() -> bool {
    has_gpu()
}

/// Fail fast if GPU is not available or CUDA toolchain missing when GPU-only is required.
///
/// Uses shared gpu-info crate for detection and validation.
pub fn assert_gpu_only() -> Result<()> {
    assert_gpu_available()
        .context("GPU-only enforcement: CUDA toolkit or NVIDIA driver not detected")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::{env, fs};

    // Guard PATH environment mutation across parallel tests
    static PATH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_isolated_path<F: FnOnce(&PathBuf)>(f: F) {
        let _guard = PATH_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        let orig_path = env::var_os("PATH");
        let dir = env::temp_dir().join(format!(
            "pool_managerd_preflight_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
        ));
        let _ = fs::create_dir_all(&dir);
        env::set_var("PATH", &dir);
        f(&dir);
        // restore and cleanup
        if let Some(p) = orig_path {
            env::set_var("PATH", p);
        } else {
            env::remove_var("PATH");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    fn create_fake_nvidia_smi(dir: &PathBuf) {
        let file = dir.join(if cfg!(windows) { "nvidia-smi.exe" } else { "nvidia-smi" });
        // Create a fake nvidia-smi that outputs valid CSV
        let script =
            b"#!/bin/sh\necho '0, NVIDIA GeForce RTX 3090, 24576, 23456, 8.6, 0000:01:00.0'\n";
        let _ = fs::write(&file, script);
        #[cfg(unix)]
        {
            if let Ok(meta) = fs::metadata(&file) {
                let mut perms = meta.permissions();
                perms.set_mode(0o755);
                let _ = fs::set_permissions(&file, perms);
            }
        }
    }

    #[test]
    fn test_cuda_available_false_when_path_empty() {
        with_isolated_path(|_dir| {
            assert!(!cuda_available());
        });
    }

    #[test]
    fn test_cuda_available_true_when_nvidia_smi_on_path() {
        with_isolated_path(|dir| {
            create_fake_nvidia_smi(dir);
            assert!(cuda_available());
        });
    }

    #[test]
    fn test_assert_gpu_only_err_when_unavailable() {
        with_isolated_path(|_dir| {
            let err = assert_gpu_only().unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("GPU-only enforcement"));
        });
    }

    #[test]
    fn test_assert_gpu_only_ok_when_available() {
        with_isolated_path(|dir| {
            create_fake_nvidia_smi(dir);
            assert!(assert_gpu_only().is_ok());
        });
    }
}
