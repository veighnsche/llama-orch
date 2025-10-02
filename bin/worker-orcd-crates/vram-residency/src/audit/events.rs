//! Audit event emission
//!
//! Emits audit events for security-critical operations using the audit-logging crate.
//!
//! # Security
//!
//! All VRAM operations are audited for:
//! - Compliance (GDPR, SOC2, ISO 27001)
//! - Forensic investigation
//! - Tamper detection
//! - Capacity tracking

use crate::types::SealedShard;
use audit_logging::{AuditEvent, AuditLogger};
use chrono::Utc;

/// Emit VramSealed audit event
///
/// Records when a model shard is sealed in VRAM with cryptographic signature.
///
/// # Parameters
///
/// * `audit_logger` - The audit logger instance
/// * `shard` - The sealed shard
/// * `worker_id` - Worker identifier
///
/// # Security
///
/// This is a security-critical event that creates an immutable audit trail entry.
pub async fn emit_vram_sealed(
    audit_logger: &AuditLogger,
    shard: &SealedShard,
    worker_id: &str,
) -> Result<(), audit_logging::AuditError> {
    audit_logger
        .emit(AuditEvent::VramSealed {
            timestamp: Utc::now(),
            shard_id: shard.shard_id.clone(),
            gpu_device: shard.gpu_device,
            vram_bytes: shard.vram_bytes,
            digest: shard.digest.clone(),
            worker_id: worker_id.to_string(),
        })
        .await
}

/// Emit SealVerified audit event
///
/// Records successful seal verification before execution.
///
/// # Parameters
///
/// * `audit_logger` - The audit logger instance
/// * `shard` - The verified shard
/// * `worker_id` - Worker identifier
///
/// # Security
///
/// This event confirms VRAM integrity before inference execution.
pub async fn emit_seal_verified(
    audit_logger: &AuditLogger,
    shard: &SealedShard,
    worker_id: &str,
) -> Result<(), audit_logging::AuditError> {
    audit_logger
        .emit(AuditEvent::SealVerified {
            timestamp: Utc::now(),
            shard_id: shard.shard_id.clone(),
            worker_id: worker_id.to_string(),
        })
        .await
}

/// Emit SealVerificationFailed audit event (CRITICAL)
///
/// Records seal verification failure - indicates VRAM corruption or tampering.
///
/// # Parameters
///
/// * `audit_logger` - The audit logger instance
/// * `shard` - The shard that failed verification
/// * `reason` - Why verification failed
/// * `expected_digest` - Expected digest value
/// * `actual_digest` - Actual digest value
/// * `worker_id` - Worker identifier
///
/// # Security
///
/// **CRITICAL SECURITY EVENT**: This indicates:
/// - VRAM corruption (hardware failure)
/// - Tampering attempt (malicious actor)
/// - Seal forgery (cryptographic attack)
///
/// Worker MUST transition to Stopped state when this occurs.
pub async fn emit_seal_verification_failed(
    audit_logger: &AuditLogger,
    shard: &SealedShard,
    reason: &str,
    expected_digest: &str,
    actual_digest: &str,
    worker_id: &str,
) -> Result<(), audit_logging::AuditError> {
    audit_logger
        .emit(AuditEvent::SealVerificationFailed {
            timestamp: Utc::now(),
            shard_id: shard.shard_id.clone(),
            reason: reason.to_string(),
            expected_digest: expected_digest.to_string(),
            actual_digest: actual_digest.to_string(),
            worker_id: worker_id.to_string(),
            severity: "CRITICAL".to_string(),
        })
        .await
}

/// Emit VramAllocated audit event
///
/// Records successful VRAM allocation.
///
/// # Parameters
///
/// * `audit_logger` - The audit logger instance
/// * `requested_bytes` - Requested allocation size
/// * `allocated_bytes` - Actual allocated size
/// * `available_bytes` - Available VRAM before allocation
/// * `used_bytes` - Total VRAM used after allocation
/// * `gpu_device` - GPU device index
/// * `worker_id` - Worker identifier
pub async fn emit_vram_allocated(
    audit_logger: &AuditLogger,
    requested_bytes: usize,
    allocated_bytes: usize,
    available_bytes: usize,
    used_bytes: usize,
    gpu_device: u32,
    worker_id: &str,
) -> Result<(), audit_logging::AuditError> {
    audit_logger
        .emit(AuditEvent::VramAllocated {
            timestamp: Utc::now(),
            requested_bytes,
            allocated_bytes,
            available_bytes,
            used_bytes,
            gpu_device,
            worker_id: worker_id.to_string(),
        })
        .await
}

/// Emit VramAllocationFailed audit event
///
/// Records failed VRAM allocation attempt.
///
/// # Parameters
///
/// * `audit_logger` - The audit logger instance
/// * `requested_bytes` - Requested allocation size
/// * `available_bytes` - Available VRAM
/// * `gpu_device` - GPU device index
/// * `worker_id` - Worker identifier
pub async fn emit_vram_allocation_failed(
    audit_logger: &AuditLogger,
    requested_bytes: usize,
    available_bytes: usize,
    gpu_device: u32,
    worker_id: &str,
) -> Result<(), audit_logging::AuditError> {
    audit_logger
        .emit(AuditEvent::VramAllocationFailed {
            timestamp: Utc::now(),
            requested_bytes,
            available_bytes,
            gpu_device,
            reason: format!("Insufficient VRAM: requested {} bytes, available {}bytes", requested_bytes, available_bytes),
            worker_id: worker_id.to_string(),
        })
        .await
}

/// Emit VramDeallocated audit event
///
/// Records when VRAM is deallocated (shard dropped).
///
/// # Parameters
///
/// * `audit_logger` - The audit logger instance
/// * `shard_id` - ID of the deallocated shard
/// * `freed_bytes` - Size of deallocated VRAM
/// * `remaining_used` - Total VRAM still in use after deallocation
/// * `gpu_device` - GPU device index
/// * `worker_id` - Worker identifier
pub async fn emit_vram_deallocated(
    audit_logger: &AuditLogger,
    shard_id: &str,
    freed_bytes: usize,
    remaining_used: usize,
    gpu_device: u32,
    worker_id: &str,
) -> Result<(), audit_logging::AuditError> {
    audit_logger
        .emit(AuditEvent::VramDeallocated {
            timestamp: Utc::now(),
            shard_id: shard_id.to_string(),
            freed_bytes,
            remaining_used,
            gpu_device,
            worker_id: worker_id.to_string(),
        })
        .await
}

/// Emit PolicyViolation audit event (CRITICAL)
///
/// Records VRAM-only policy violation.
///
/// # Parameters
///
/// * `audit_logger` - The audit logger instance
/// * `violation` - Description of the policy violation
/// * `details` - Additional details about the violation
/// * `action_taken` - Action taken in response
/// * `worker_id` - Worker identifier
///
/// # Security
///
/// **CRITICAL SECURITY EVENT**: This indicates the VRAM-only policy
/// cannot be enforced. Worker MUST NOT start if this occurs.
pub async fn emit_policy_violation(
    audit_logger: &AuditLogger,
    violation: &str,
    details: &str,
    action_taken: &str,
    worker_id: &str,
) -> Result<(), audit_logging::AuditError> {
    audit_logger
        .emit(AuditEvent::PolicyViolation {
            timestamp: Utc::now(),
            policy: "vram_only".to_string(),
            violation: violation.to_string(),
            details: details.to_string(),
            severity: "CRITICAL".to_string(),
            worker_id: worker_id.to_string(),
            action_taken: action_taken.to_string(),
        })
        .await
}
