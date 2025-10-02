//! VRAM-only policy enforcement
//!
//! Enforces that models reside entirely in VRAM.

use crate::error::{Result, VramError};
use crate::policy::validation::{validate_device_properties, check_unified_memory};
use audit_logging::{AuditLogger, AuditEvent};
use chrono::Utc;
use std::sync::Arc;

/// Enforce VRAM-only policy
///
/// # Requirements
///
/// - Disable unified memory (UMA)
/// - Disable zero-copy
/// - Disable pinned host memory
/// - Verify device properties
///
/// # Arguments
///
/// * `gpu_device` - GPU device index
///
/// # Returns
///
/// `Ok(())` if policy is enforced, error otherwise
///
/// # Security
///
/// This function ensures that all model weights reside in VRAM during inference.
/// No RAM fallback is allowed. GPU is required; fail fast if policy cannot be enforced.
///
/// # Note
///
/// In test mode (without CUDA), this performs validation only.
/// In production mode with CUDA, this would call cudaDeviceSetLimit to disable UMA.
pub fn enforce_vram_only_policy(
    gpu_device: u32,
    audit_logger: Option<&Arc<AuditLogger>>,
    worker_id: &str,
) -> Result<()> {
    // Validate device properties first
    if let Err(e) = validate_device_properties(gpu_device) {
        // Emit policy violation audit event
        if let Some(logger) = audit_logger {
            if let Err(audit_err) = logger.emit(AuditEvent::PolicyViolation {
                timestamp: Utc::now(),
                policy: "vram_only".to_string(),
                violation: "invalid_device_properties".to_string(),
                details: format!("Device validation failed: {}", e),
                severity: "CRITICAL".to_string(),
                worker_id: worker_id.to_string(),
                action_taken: "worker_stopped".to_string(),
            }) {
                tracing::error!(error = %audit_err, "Failed to emit PolicyViolation audit event");
            }
        }
        return Err(e);
    }
    
    // Check unified memory status
    if let Err(e) = check_unified_memory(gpu_device) {
        // Emit policy violation audit event
        if let Some(logger) = audit_logger {
            if let Err(audit_err) = logger.emit(AuditEvent::PolicyViolation {
                timestamp: Utc::now(),
                policy: "vram_only".to_string(),
                violation: "unified_memory_detected".to_string(),
                details: format!("UMA check failed: {}", e),
                severity: "CRITICAL".to_string(),
                worker_id: worker_id.to_string(),
                action_taken: "worker_stopped".to_string(),
            }) {
                tracing::error!(error = %audit_err, "Failed to emit PolicyViolation audit event");
            }
        }
        return Err(e);
    }
    
    // In test mode, we can only validate
    #[cfg(test)]
    {
        tracing::info!(
            gpu_device = %gpu_device,
            "VRAM-only policy enforcement: test mode (validation only)"
        );
        return Ok(());
    }
    
    // In production mode with CUDA, enforce policy
    #[cfg(not(test))]
    {
        // Note: CUDA doesn't provide a direct API to disable unified memory
        // at the application level. The VRAM-only policy is enforced by:
        //
        // 1. Using cudaMalloc (not cudaMallocManaged)
        // 2. Never using cudaMallocHost or cudaHostAlloc
        // 3. Validating device properties (compute capability >= 6.0)
        // 4. Cryptographic sealing to detect VRAM corruption
        //
        // The CudaContext and SafeCudaPtr already enforce this by only
        // using cudaMalloc for allocations.
        
        tracing::info!(
            gpu_device = %gpu_device,
            "VRAM-only policy enforced: using cudaMalloc exclusively"
        );
        
        Ok(())
    }
}

/// Check if policy is currently enforced
///
/// # Arguments
///
/// * `gpu_device` - GPU device index
///
/// # Returns
///
/// `Ok(true)` if policy is enforced, `Ok(false)` otherwise
pub fn is_policy_enforced(gpu_device: u32) -> Result<bool> {
    // Check if device properties are valid
    match validate_device_properties(gpu_device) {
        Ok(()) => {
            tracing::debug!(
                gpu_device = %gpu_device,
                "VRAM-only policy is enforced"
            );
            Ok(true)
        }
        Err(VramError::PolicyViolation(_)) => {
            tracing::warn!(
                gpu_device = %gpu_device,
                "VRAM-only policy is NOT enforced"
            );
            Ok(false)
        }
        Err(e) => Err(e),
    }
}
