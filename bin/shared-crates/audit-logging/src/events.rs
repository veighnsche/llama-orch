//! Audit event types
//!
//! Defines all 32 audit event types across 7 categories.
//! See `.specs/01_event-types.md` for complete documentation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

/// Actor information (WHO performed the action)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorInfo {
    /// User ID or token fingerprint (e.g., "admin@example.com" or "token:a3f2c1")
    pub user_id: String,

    /// Source IP address
    pub ip: Option<IpAddr>,

    /// Authentication method used
    pub auth_method: AuthMethod,

    /// Session ID for correlation
    pub session_id: Option<String>,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    BearerToken,
    ApiKey,
    MTls,
    Internal,
}

/// Resource information (WHAT was affected)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// Resource type (e.g., "pool", "node", "job", "shard")
    pub resource_type: String,

    /// Resource ID (e.g., "pool-123", "shard-abc123")
    pub resource_id: String,

    /// Parent resource ID (e.g., "node-1" for a pool)
    pub parent_id: Option<String>,
}

/// Audit result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditResult {
    Success,
    Failure { reason: String },
    PartialSuccess { details: String },
}

/// Audit event types
///
/// All 35 event types across 7 categories:
/// - Authentication (4 types)
/// - Authorization (3 types)
/// - Resource Operations (8 types)
/// - VRAM Operations (6 types)
/// - Security Incidents (8 types)
/// - Data Access (3 types)
/// - Compliance (3 types)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum AuditEvent {
    // ========== Authentication Events (4) ==========
    /// Successful authentication
    AuthSuccess {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        method: AuthMethod,
        path: String,
        service_id: String,
    },

    /// Failed authentication attempt
    AuthFailure {
        timestamp: DateTime<Utc>,
        attempted_user: Option<String>,
        reason: String,
        ip: IpAddr,
        path: String,
        service_id: String,
    },

    /// API token created
    TokenCreated {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        token_fingerprint: String,
        scope: Vec<String>,
        expires_at: Option<DateTime<Utc>>,
    },

    /// API token revoked
    TokenRevoked {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        token_fingerprint: String,
        reason: String,
    },

    // ========== Authorization Events (3) ==========
    /// Authorization granted
    AuthorizationGranted {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        resource: ResourceInfo,
        action: String,
    },

    /// Authorization denied
    AuthorizationDenied {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        resource: ResourceInfo,
        action: String,
        reason: String,
    },

    /// Permission changed
    PermissionChanged {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        subject: String,
        old_permissions: Vec<String>,
        new_permissions: Vec<String>,
    },

    // ========== Resource Operation Events (8) ==========
    /// Pool created
    PoolCreated {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        pool_id: String,
        model_ref: String,
        node_id: String,
        replicas: u32,
        gpu_devices: Vec<u32>,
    },

    /// Pool deleted
    PoolDeleted {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        pool_id: String,
        model_ref: String,
        node_id: String,
        reason: String,
        replicas_terminated: u32,
    },

    /// Pool modified
    PoolModified {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        pool_id: String,
        changes: serde_json::Value,
    },

    /// Node registered
    NodeRegistered {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        node_id: String,
        gpu_count: u32,
        total_vram_gb: u64,
        capabilities: Vec<String>,
    },

    /// Node deregistered
    NodeDeregistered {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        node_id: String,
        reason: String,
        pools_affected: Vec<String>,
    },

    /// Task submitted
    TaskSubmitted {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        task_id: String,
        model_ref: String,
        prompt_length: usize,
        prompt_hash: String,
        max_tokens: u32,
    },

    /// Task completed
    TaskCompleted {
        timestamp: DateTime<Utc>,
        task_id: String,
        worker_id: String,
        tokens_generated: u32,
        duration_ms: u64,
        result: AuditResult,
    },

    /// Task canceled
    TaskCanceled { timestamp: DateTime<Utc>, actor: ActorInfo, task_id: String, reason: String },

    // ========== VRAM Operation Events (6) ==========
    /// Model sealed in VRAM
    VramSealed {
        timestamp: DateTime<Utc>,
        shard_id: String,
        gpu_device: u32,
        vram_bytes: usize,
        digest: String,
        worker_id: String,
    },

    /// Seal verified successfully
    SealVerified { timestamp: DateTime<Utc>, shard_id: String, worker_id: String },

    /// Seal verification failed (CRITICAL)
    SealVerificationFailed {
        timestamp: DateTime<Utc>,
        shard_id: String,
        reason: String,
        expected_digest: String,
        actual_digest: String,
        worker_id: String,
        severity: String,
    },

    /// VRAM allocated
    VramAllocated {
        timestamp: DateTime<Utc>,
        requested_bytes: usize,
        allocated_bytes: usize,
        available_bytes: usize,
        used_bytes: usize,
        gpu_device: u32,
        worker_id: String,
    },

    /// VRAM allocation failed
    VramAllocationFailed {
        timestamp: DateTime<Utc>,
        requested_bytes: usize,
        available_bytes: usize,
        reason: String,
        gpu_device: u32,
        worker_id: String,
    },

    /// VRAM deallocated
    VramDeallocated {
        timestamp: DateTime<Utc>,
        shard_id: String,
        freed_bytes: usize,
        remaining_used: usize,
        gpu_device: u32,
        worker_id: String,
    },

    // ========== Security Incident Events (8) ==========
    /// Rate limit exceeded
    RateLimitExceeded {
        timestamp: DateTime<Utc>,
        ip: IpAddr,
        endpoint: String,
        limit: u32,
        actual: u32,
        window_seconds: u32,
    },

    /// Path traversal attempt
    PathTraversalAttempt {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        attempted_path: String,
        endpoint: String,
    },

    /// Invalid token used
    InvalidTokenUsed {
        timestamp: DateTime<Utc>,
        ip: IpAddr,
        token_prefix: String,
        endpoint: String,
    },

    /// Security policy violation
    PolicyViolation {
        timestamp: DateTime<Utc>,
        policy: String,
        violation: String,
        details: String,
        severity: String,
        worker_id: String,
        action_taken: String,
    },

    /// Suspicious activity detected
    SuspiciousActivity {
        timestamp: DateTime<Utc>,
        actor: ActorInfo,
        activity_type: String,
        details: String,
        risk_score: u32,
    },

    /// Integrity violation (hash mismatch, supply chain attack)
    IntegrityViolation {
        timestamp: DateTime<Utc>,
        resource_type: String,
        resource_id: String,
        expected_hash: String,
        actual_hash: String,
        severity: String,
        action_taken: String,
        worker_id: Option<String>,
    },

    /// Malformed model rejected (potential exploit attempt)
    MalformedModelRejected {
        timestamp: DateTime<Utc>,
        model_ref: String,
        validation_error: String,
        severity: String,
        action_taken: String,
        worker_id: Option<String>,
    },

    /// Resource limit violation (DoS attempt)
    ResourceLimitViolation {
        timestamp: DateTime<Utc>,
        resource_type: String,
        limit_type: String,
        limit_value: u64,
        actual_value: u64,
        severity: String,
        action_taken: String,
        worker_id: Option<String>,
    },

    // ========== Data Access Events (3) ==========
    /// Inference executed (GDPR)
    InferenceExecuted {
        timestamp: DateTime<Utc>,
        customer_id: String,
        job_id: String,
        model_ref: String,
        tokens_processed: u32,
        provider_id: Option<String>,
        result: AuditResult,
    },

    /// Model accessed (GDPR)
    ModelAccessed {
        timestamp: DateTime<Utc>,
        customer_id: String,
        model_ref: String,
        access_type: String,
        provider_id: Option<String>,
    },

    /// Data deleted (GDPR)
    DataDeleted {
        timestamp: DateTime<Utc>,
        customer_id: String,
        data_types: Vec<String>,
        reason: String,
    },

    // ========== Compliance Events (3) ==========
    /// GDPR data access request
    GdprDataAccessRequest {
        timestamp: DateTime<Utc>,
        customer_id: String,
        requester: String,
        scope: Vec<String>,
    },

    /// GDPR data export
    GdprDataExport {
        timestamp: DateTime<Utc>,
        customer_id: String,
        data_types: Vec<String>,
        export_format: String,
        file_hash: String,
    },

    /// GDPR right to erasure
    GdprRightToErasure {
        timestamp: DateTime<Utc>,
        customer_id: String,
        completed_at: DateTime<Utc>,
        data_types_deleted: Vec<String>,
    },
}

impl AuditEvent {
    /// Check if this event is security-critical
    ///
    /// Critical events should be flushed immediately (no batching) to ensure
    /// they are never lost in case of crash or power failure.
    ///
    /// # Critical Events
    ///
    /// - `AuthFailure` — Security incident (failed login attempt)
    /// - `TokenRevoked` — Security action (token invalidated)
    /// - `PolicyViolation` — Security breach (policy enforcement)
    /// - `SealVerificationFailed` — VRAM security failure
    /// - `PathTraversalAttempt` — Attack attempt
    /// - `InvalidTokenUsed` — Attack attempt
    /// - `SuspiciousActivity` — Anomaly detection
    /// - `IntegrityViolation` — Hash mismatch, supply chain attack
    /// - `MalformedModelRejected` — Potential exploit attempt
    /// - `ResourceLimitViolation` — DoS attempt
    ///
    /// # Routine Events (Can Batch)
    ///
    /// - `AuthSuccess` — Normal operation
    /// - `TaskSubmitted`, `TaskCompleted`, `TaskCanceled` — Normal operation
    /// - `PoolCreated`, `PoolDeleted`, `PoolModified` — Admin action
    /// - `VramSealed`, `SealVerified`, `VramAllocated`, `VramDeallocated` — Normal operation
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            // Security incidents (always critical)
            AuditEvent::AuthFailure { .. }
                | AuditEvent::TokenRevoked { .. }
                | AuditEvent::PolicyViolation { .. }
                | AuditEvent::SealVerificationFailed { .. }
                | AuditEvent::PathTraversalAttempt { .. }
                | AuditEvent::InvalidTokenUsed { .. }
                | AuditEvent::SuspiciousActivity { .. }
                | AuditEvent::IntegrityViolation { .. }
                | AuditEvent::MalformedModelRejected { .. }
                | AuditEvent::ResourceLimitViolation { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_failure_is_critical() {
        let event = AuditEvent::AuthFailure {
            timestamp: Utc::now(),
            attempted_user: Some("attacker@evil.com".to_string()),
            reason: "invalid_password".to_string(),
            ip: "10.0.0.1".parse().unwrap(),
            path: "/v2/tasks".to_string(),
            service_id: "orchestratord".to_string(),
        };
        assert!(event.is_critical(), "AuthFailure should be critical");
    }

    #[test]
    fn test_auth_success_is_not_critical() {
        let event = AuditEvent::AuthSuccess {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "user@example.com".to_string(),
                ip: Some("127.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            method: AuthMethod::BearerToken,
            path: "/v2/tasks".to_string(),
            service_id: "orchestratord".to_string(),
        };
        assert!(!event.is_critical(), "AuthSuccess should not be critical");
    }

    #[test]
    fn test_policy_violation_is_critical() {
        let event = AuditEvent::PolicyViolation {
            timestamp: Utc::now(),
            policy: "vram_isolation".to_string(),
            violation: "cross_shard_access".to_string(),
            details: "Shard A accessed Shard B memory".to_string(),
            severity: "high".to_string(),
            worker_id: "worker-gpu-0".to_string(),
            action_taken: "shard_terminated".to_string(),
        };
        assert!(event.is_critical(), "PolicyViolation should be critical");
    }

    #[test]
    fn test_pool_created_is_not_critical() {
        let event = AuditEvent::PoolCreated {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "admin@example.com".to_string(),
                ip: Some("127.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            pool_id: "pool-123".to_string(),
            model_ref: "llama-3.1-8b".to_string(),
            node_id: "node-1".to_string(),
            replicas: 1,
            gpu_devices: vec![0],
        };
        assert!(!event.is_critical(), "PoolCreated should not be critical");
    }
}
