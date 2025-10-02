//! Audit event emission
//!
//! Emits audit events for security-critical operations.
//!
//! # Note
//!
//! This module currently uses structured logging via `tracing`.
//! Future integration with `audit-logging` crate will provide:
//! - Tamper-evident audit logs
//! - Cryptographic integrity verification
//! - Structured audit event types
//!
//! See: `bin/shared-crates/audit-logging/README.md`

use crate::types::SealedShard;

/// Emit VramSealed audit event
///
/// Records when a model shard is sealed in VRAM with cryptographic signature.
///
/// # Arguments
///
/// * `shard` - The sealed shard
///
/// # Security
///
/// This is a security-critical event that should be logged to the audit trail.
pub fn emit_vram_sealed(shard: &SealedShard) {
    tracing::info!(
        event = "VramSealed",
        shard_id = %shard.shard_id,
        gpu_device = %shard.gpu_device,
        vram_bytes = %shard.vram_bytes,
        digest = %&shard.digest[..16],
        sealed_at = ?shard.sealed_at,
        "Model shard sealed in VRAM with cryptographic signature"
    );
}

/// Emit SealVerified audit event
///
/// Records successful seal verification before execution.
///
/// # Arguments
///
/// * `shard` - The verified shard
///
/// # Security
///
/// This event confirms VRAM integrity before inference execution.
pub fn emit_seal_verified(shard: &SealedShard) {
    tracing::info!(
        event = "SealVerified",
        shard_id = %shard.shard_id,
        gpu_device = %shard.gpu_device,
        "Seal verification passed - VRAM integrity confirmed"
    );
}

/// Emit SealVerificationFailed audit event (CRITICAL)
///
/// Records seal verification failure - indicates VRAM corruption or tampering.
///
/// # Arguments
///
/// * `shard` - The shard that failed verification
/// * `reason` - Reason for failure
///
/// # Security
///
/// **CRITICAL SECURITY EVENT**: This indicates either:
/// - VRAM corruption (hardware failure)
/// - Tampering attempt (security incident)
///
/// Worker MUST transition to Stopped state when this occurs.
pub fn emit_seal_verification_failed(shard: &SealedShard, reason: &str) {
    tracing::error!(
        event = "SealVerificationFailed",
        severity = "CRITICAL",
        shard_id = %shard.shard_id,
        gpu_device = %shard.gpu_device,
        reason = %reason,
        "SECURITY INCIDENT: Seal verification failed - VRAM corruption or tampering detected"
    );
}

/// Emit VramDeallocated audit event
///
/// Records when VRAM is deallocated (shard dropped).
///
/// # Arguments
///
/// * `shard_id` - ID of the deallocated shard
/// * `vram_bytes` - Size of deallocated VRAM
pub fn emit_vram_deallocated(shard_id: &str, vram_bytes: usize) {
    tracing::info!(
        event = "VramDeallocated",
        shard_id = %shard_id,
        vram_bytes = %vram_bytes,
        "VRAM deallocated"
    );
}

/// Emit PolicyViolation audit event (CRITICAL)
///
/// Records VRAM-only policy violation.
///
/// # Arguments
///
/// * `reason` - Reason for policy violation
///
/// # Security
///
/// **CRITICAL SECURITY EVENT**: This indicates the VRAM-only policy
/// cannot be enforced. Worker MUST NOT start if this occurs.
pub fn emit_policy_violation(reason: &str) {
    tracing::error!(
        event = "PolicyViolation",
        severity = "CRITICAL",
        reason = %reason,
        "SECURITY INCIDENT: VRAM-only policy violation detected"
    );
}
