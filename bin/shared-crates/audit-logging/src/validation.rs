//! Input validation for audit events
//!
//! Integrates with `input-validation` crate to prevent log injection attacks.
//!
//! **CRITICAL SECURITY REQUIREMENT**: All user-controlled data MUST be sanitized
//! before being included in audit events.

use crate::error::{AuditError, Result};
use crate::events::{ActorInfo, AuditEvent, ResourceInfo};
use std::borrow::Cow;

/// Validate and sanitize audit event
///
/// Applies input validation to all user-controlled fields:
/// - Actor user_id
/// - Resource IDs
/// - Reason strings
/// - Details strings
/// - Any other user-supplied data
///
/// # Security
///
/// This prevents:
/// - ANSI escape sequence injection
/// - Control character injection
/// - Unicode directional override attacks
/// - Null byte injection
pub fn validate_event(event: &mut AuditEvent) -> Result<()> {
    match event {
        // Authentication events
        AuditEvent::AuthSuccess { actor, path, .. } => {
            validate_actor(actor)?;
            validate_string_field(path, "path")?;
        }

        AuditEvent::AuthFailure { attempted_user, path, .. } => {
            if let Some(user) = attempted_user {
                let sanitized = sanitize(user)?;
                if let Cow::Owned(s) = sanitized {
                    *user = s;
                }
            }
            validate_string_field(path, "path")?;
        }

        AuditEvent::TokenCreated { actor, .. } => {
            validate_actor(actor)?;
        }

        AuditEvent::TokenRevoked { actor, reason, .. } => {
            validate_actor(actor)?;
            validate_string_field(reason, "reason")?;
        }

        // Authorization events
        AuditEvent::AuthorizationGranted { actor, resource, action, .. } => {
            validate_actor(actor)?;
            validate_resource(resource)?;
            validate_string_field(action, "action")?;
        }

        AuditEvent::AuthorizationDenied { actor, resource, action, reason, .. } => {
            validate_actor(actor)?;
            validate_resource(resource)?;
            validate_string_field(action, "action")?;
            validate_string_field(reason, "reason")?;
        }

        AuditEvent::PermissionChanged { actor, subject, .. } => {
            validate_actor(actor)?;
            validate_string_field(subject, "subject")?;
        }

        // Resource operations
        AuditEvent::PoolCreated { actor, pool_id, model_ref, node_id, .. } => {
            validate_actor(actor)?;
            validate_string_field(pool_id, "pool_id")?;
            validate_string_field(model_ref, "model_ref")?;
            validate_string_field(node_id, "node_id")?;
        }

        AuditEvent::PoolDeleted { actor, pool_id, model_ref, node_id, reason, .. } => {
            validate_actor(actor)?;
            validate_string_field(pool_id, "pool_id")?;
            validate_string_field(model_ref, "model_ref")?;
            validate_string_field(node_id, "node_id")?;
            validate_string_field(reason, "reason")?;
        }

        AuditEvent::PoolModified { actor, pool_id, .. } => {
            validate_actor(actor)?;
            validate_string_field(pool_id, "pool_id")?;
        }

        AuditEvent::NodeRegistered { actor, node_id, .. } => {
            validate_actor(actor)?;
            validate_string_field(node_id, "node_id")?;
        }

        AuditEvent::NodeDeregistered { actor, node_id, reason, .. } => {
            validate_actor(actor)?;
            validate_string_field(node_id, "node_id")?;
            validate_string_field(reason, "reason")?;
        }

        AuditEvent::TaskSubmitted { actor, task_id, model_ref, .. } => {
            validate_actor(actor)?;
            validate_string_field(task_id, "task_id")?;
            validate_string_field(model_ref, "model_ref")?;
        }

        AuditEvent::TaskCompleted { task_id, worker_id, .. } => {
            validate_string_field(task_id, "task_id")?;
            validate_string_field(worker_id, "worker_id")?;
        }

        AuditEvent::TaskCanceled { actor, task_id, reason, .. } => {
            validate_actor(actor)?;
            validate_string_field(task_id, "task_id")?;
            validate_string_field(reason, "reason")?;
        }

        // VRAM operations
        AuditEvent::VramSealed { shard_id, worker_id, .. } => {
            validate_string_field(shard_id, "shard_id")?;
            validate_string_field(worker_id, "worker_id")?;
        }

        AuditEvent::SealVerified { shard_id, worker_id, .. } => {
            validate_string_field(shard_id, "shard_id")?;
            validate_string_field(worker_id, "worker_id")?;
        }

        AuditEvent::SealVerificationFailed { shard_id, reason, worker_id, .. } => {
            validate_string_field(shard_id, "shard_id")?;
            validate_string_field(reason, "reason")?;
            validate_string_field(worker_id, "worker_id")?;
        }

        AuditEvent::VramAllocated { worker_id, .. } => {
            validate_string_field(worker_id, "worker_id")?;
        }

        AuditEvent::VramAllocationFailed { reason, worker_id, .. } => {
            validate_string_field(reason, "reason")?;
            validate_string_field(worker_id, "worker_id")?;
        }

        AuditEvent::VramDeallocated { shard_id, worker_id, .. } => {
            validate_string_field(shard_id, "shard_id")?;
            validate_string_field(worker_id, "worker_id")?;
        }

        // Security incidents
        AuditEvent::RateLimitExceeded { endpoint, .. } => {
            validate_string_field(endpoint, "endpoint")?;
        }

        AuditEvent::PathTraversalAttempt { actor, attempted_path, endpoint, .. } => {
            validate_actor(actor)?;
            validate_string_field(attempted_path, "attempted_path")?;
            validate_string_field(endpoint, "endpoint")?;
        }

        AuditEvent::InvalidTokenUsed { endpoint, .. } => {
            validate_string_field(endpoint, "endpoint")?;
        }

        AuditEvent::PolicyViolation {
            policy, violation, details, worker_id, action_taken, ..
        } => {
            validate_string_field(policy, "policy")?;
            validate_string_field(violation, "violation")?;
            validate_string_field(details, "details")?;
            validate_string_field(worker_id, "worker_id")?;
            validate_string_field(action_taken, "action_taken")?;
        }

        AuditEvent::SuspiciousActivity { actor, activity_type, details, .. } => {
            validate_actor(actor)?;
            validate_string_field(activity_type, "activity_type")?;
            validate_string_field(details, "details")?;
        }

        AuditEvent::IntegrityViolation {
            resource_type,
            resource_id,
            expected_hash,
            actual_hash,
            severity,
            action_taken,
            worker_id,
            ..
        } => {
            validate_string_field(resource_type, "resource_type")?;
            validate_string_field(resource_id, "resource_id")?;
            validate_string_field(expected_hash, "expected_hash")?;
            validate_string_field(actual_hash, "actual_hash")?;
            validate_string_field(severity, "severity")?;
            validate_string_field(action_taken, "action_taken")?;
            if let Some(wid) = worker_id {
                validate_string_field(wid, "worker_id")?;
            }
        }

        AuditEvent::MalformedModelRejected {
            model_ref,
            validation_error,
            severity,
            action_taken,
            worker_id,
            ..
        } => {
            validate_string_field(model_ref, "model_ref")?;
            validate_string_field(validation_error, "validation_error")?;
            validate_string_field(severity, "severity")?;
            validate_string_field(action_taken, "action_taken")?;
            if let Some(wid) = worker_id {
                validate_string_field(wid, "worker_id")?;
            }
        }

        AuditEvent::ResourceLimitViolation {
            resource_type,
            limit_type,
            severity,
            action_taken,
            worker_id,
            ..
        } => {
            validate_string_field(resource_type, "resource_type")?;
            validate_string_field(limit_type, "limit_type")?;
            validate_string_field(severity, "severity")?;
            validate_string_field(action_taken, "action_taken")?;
            if let Some(wid) = worker_id {
                validate_string_field(wid, "worker_id")?;
            }
        }

        // Data access
        AuditEvent::InferenceExecuted { customer_id, job_id, model_ref, .. } => {
            validate_string_field(customer_id, "customer_id")?;
            validate_string_field(job_id, "job_id")?;
            validate_string_field(model_ref, "model_ref")?;
        }

        AuditEvent::ModelAccessed { customer_id, model_ref, access_type, .. } => {
            validate_string_field(customer_id, "customer_id")?;
            validate_string_field(model_ref, "model_ref")?;
            validate_string_field(access_type, "access_type")?;
        }

        AuditEvent::DataDeleted { customer_id, reason, .. } => {
            validate_string_field(customer_id, "customer_id")?;
            validate_string_field(reason, "reason")?;
        }

        // Compliance
        AuditEvent::GdprDataAccessRequest { customer_id, requester, .. } => {
            validate_string_field(customer_id, "customer_id")?;
            validate_string_field(requester, "requester")?;
        }

        AuditEvent::GdprDataExport { customer_id, export_format, .. } => {
            validate_string_field(customer_id, "customer_id")?;
            validate_string_field(export_format, "export_format")?;
        }

        AuditEvent::GdprRightToErasure { customer_id, .. } => {
            validate_string_field(customer_id, "customer_id")?;
        }
    }

    Ok(())
}

/// Validate actor information
fn validate_actor(actor: &mut ActorInfo) -> Result<()> {
    let sanitized = sanitize(&actor.user_id)?;
    if let Cow::Owned(s) = sanitized {
        actor.user_id = s; // Only update if changed
    }

    if let Some(session_id) = &actor.session_id {
        let sanitized = sanitize(session_id)?;
        if let Cow::Owned(s) = sanitized {
            actor.session_id = Some(s);
        }
    }

    Ok(())
}

/// Validate resource information
fn validate_resource(resource: &mut ResourceInfo) -> Result<()> {
    let sanitized = sanitize(&resource.resource_type)?;
    if let Cow::Owned(s) = sanitized {
        resource.resource_type = s;
    }

    let sanitized = sanitize(&resource.resource_id)?;
    if let Cow::Owned(s) = sanitized {
        resource.resource_id = s;
    }

    if let Some(parent_id) = &resource.parent_id {
        let sanitized = sanitize(parent_id)?;
        if let Cow::Owned(s) = sanitized {
            resource.parent_id = Some(s);
        }
    }

    Ok(())
}

/// Validate and sanitize a string field
fn validate_string_field(field: &mut String, field_name: &'static str) -> Result<()> {
    // Check length limits
    const MAX_FIELD_LEN: usize = 1024;
    if field.len() > MAX_FIELD_LEN {
        return Err(AuditError::FieldTooLong(field_name));
    }

    // Sanitize (only update if changed)
    let sanitized = sanitize(field)?;
    if let Cow::Owned(s) = sanitized {
        *field = s;
    }

    Ok(())
}

/// Sanitize string using input-validation crate
///
/// Removes:
/// - ANSI escape sequences
/// - Control characters (except newline in structured fields)
/// - Unicode directional overrides
/// - Null bytes
///
/// Returns Cow<'a, str> to avoid allocation when input is already valid.
/// Finding 2 optimization: Zero-copy when sanitization doesn't change the string.
fn sanitize<'a>(input: &'a str) -> Result<Cow<'a, str>> {
    input_validation::sanitize_string(input)
        .map(|s| {
            // Only allocate if sanitization changed the string
            if s.as_ptr() == input.as_ptr() && s.len() == input.len() {
                Cow::Borrowed(input) // Zero allocation
            } else {
                Cow::Owned(s.to_string()) // Allocate only if changed
            }
        })
        .map_err(|e| AuditError::InvalidInput(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{ActorInfo, AuditEvent, AuthMethod};
    use chrono::Utc;
    use std::net::IpAddr;

    #[test]
    fn test_sanitize_rejects_ansi_escapes() {
        let input = "\x1b[31mFAKE ERROR\x1b[0m";
        let result = sanitize(input);
        assert!(result.is_err(), "ANSI escapes should be rejected");
    }

    #[test]
    fn test_sanitize_rejects_control_chars() {
        let input = "normal\x07text"; // Bell character
        let result = sanitize(input);
        assert!(result.is_err(), "Control chars should be rejected");
    }

    #[test]
    fn test_sanitize_allows_newlines() {
        let input = "line1\nline2";
        let result = sanitize(input);
        assert!(result.is_ok(), "Newlines should be allowed");
    }

    #[test]
    fn test_validate_auth_success() {
        let mut event = AuditEvent::AuthSuccess {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "admin@example.com".to_string(),
                ip: Some("192.168.1.1".parse::<IpAddr>().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            method: AuthMethod::BearerToken,
            path: "/v2/tasks".to_string(),
            service_id: "orchestratord".to_string(),
        };

        assert!(validate_event(&mut event).is_ok());
    }

    #[test]
    fn test_field_too_long() {
        let long_string = "A".repeat(2000);
        let result = validate_string_field(&mut long_string.clone(), "test_field");
        assert!(matches!(result, Err(AuditError::FieldTooLong("test_field"))));
    }

    #[test]
    fn test_sanitize_rejects_null_bytes() {
        let input = "test\0data";
        let result = sanitize(input);
        assert!(result.is_err(), "Null bytes should be rejected");
    }

    #[test]
    fn test_sanitize_rejects_unicode_overrides() {
        let input = "test\u{202E}evil\u{202D}";
        let result = sanitize(input);
        assert!(result.is_err(), "Unicode overrides should be rejected");
    }

    #[test]
    fn test_validate_token_created() {
        let mut event = AuditEvent::TokenCreated {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "admin@example.com".to_string(),
                ip: Some("192.168.1.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            token_fingerprint: "abc123".to_string(),
            scope: vec!["read".to_string()],
            expires_at: None,
        };
        assert!(validate_event(&mut event).is_ok());
    }

    #[test]
    fn test_validate_pool_created() {
        let mut event = AuditEvent::PoolCreated {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "admin@example.com".to_string(),
                ip: Some("192.168.1.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            pool_id: "pool-123".to_string(),
            model_ref: "meta-llama/Llama-3.1-8B".to_string(),
            node_id: "node-gpu-0".to_string(),
            replicas: 1,
            gpu_devices: vec![0],
        };
        assert!(validate_event(&mut event).is_ok());
    }

    #[test]
    fn test_validate_vram_sealed() {
        let mut event = AuditEvent::VramSealed {
            timestamp: Utc::now(),
            shard_id: "shard-abc123".to_string(),
            gpu_device: 0,
            vram_bytes: 1024,
            digest: "abc123".to_string(),
            worker_id: "worker-gpu-0".to_string(),
        };
        assert!(validate_event(&mut event).is_ok());
    }

    #[test]
    fn test_validate_path_traversal_attempt() {
        let mut event = AuditEvent::PathTraversalAttempt {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "attacker@evil.com".to_string(),
                ip: Some("10.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            attempted_path: "../../../etc/passwd".to_string(),
            endpoint: "/v2/files".to_string(),
        };
        assert!(validate_event(&mut event).is_ok());
    }

    #[test]
    fn test_validate_gdpr_data_access_request() {
        let mut event = AuditEvent::GdprDataAccessRequest {
            timestamp: Utc::now(),
            customer_id: "customer-123".to_string(),
            requester: "user@example.com".to_string(),
            scope: vec!["all".to_string()],
        };
        assert!(validate_event(&mut event).is_ok());
    }

    #[test]
    fn test_rejects_ansi_escape_in_user_id() {
        let mut event = AuditEvent::AuthSuccess {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "\x1b[31mFAKE\x1b[0m@example.com".to_string(),
                ip: Some("127.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            method: AuthMethod::BearerToken,
            path: "/test".to_string(),
            service_id: "test".to_string(),
        };

        // Validation should reject events with ANSI escapes
        assert!(
            validate_event(&mut event).is_err(),
            "ANSI escapes should cause validation to fail"
        );
    }

    #[test]
    fn test_rejects_control_chars_in_reason() {
        let mut event = AuditEvent::TokenRevoked {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "admin@example.com".to_string(),
                ip: Some("127.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            token_fingerprint: "abc123".to_string(),
            reason: "revoked\x07[FAKE] log".to_string(), // Bell character
        };

        // Validation should reject events with control chars
        assert!(
            validate_event(&mut event).is_err(),
            "Control chars should cause validation to fail"
        );
    }

    #[test]
    fn test_validate_actor_rejects_bad_session_id() {
        let mut actor = ActorInfo {
            user_id: "user@example.com".to_string(),
            ip: Some("127.0.0.1".parse().unwrap()),
            auth_method: AuthMethod::BearerToken,
            session_id: Some("session\x1b[31m123".to_string()),
        };

        assert!(validate_actor(&mut actor).is_err(), "Bad session ID should be rejected");
    }

    #[test]
    fn test_validate_resource_rejects_bad_fields() {
        let mut resource = ResourceInfo {
            resource_type: "pool\x1b[0m".to_string(),
            resource_id: "pool-123".to_string(),
            parent_id: None,
        };

        assert!(validate_resource(&mut resource).is_err(), "Bad resource type should be rejected");
    }

    #[test]
    fn test_field_length_limit_enforced() {
        let mut long_field = "A".repeat(1500);
        let result = validate_string_field(&mut long_field, "test_field");
        assert!(result.is_err());
        assert!(matches!(result, Err(AuditError::FieldTooLong("test_field"))));
    }

    #[test]
    fn test_field_at_max_length_accepted() {
        let mut field = "A".repeat(1024);
        let result = validate_string_field(&mut field, "test_field");
        assert!(result.is_ok());
    }
}
