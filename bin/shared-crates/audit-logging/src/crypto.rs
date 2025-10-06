//! Cryptographic operations for audit logging
//!
//! Provides:
//! - SHA-256 hashing for hash chains
//! - HMAC-SHA256 signatures (platform mode)
//! - Ed25519 signatures (platform mode, optional)

use crate::error::{AuditError, Result};
use crate::storage::AuditEventEnvelope;
use sha2::{Digest, Sha256};

/// Compute SHA-256 hash of audit event
///
/// Hash includes:
/// - audit_id
/// - timestamp (RFC3339)
/// - service_id
/// - event (JSON serialized)
/// - prev_hash
///
/// # Security
///
/// Uses SHA-256 (FIPS 140-2 approved algorithm).
/// Hash is deterministic and collision-resistant.
///
/// # Errors
///
/// Returns `Serialization` error if event cannot be serialized to JSON.
pub fn compute_event_hash(envelope: &AuditEventEnvelope) -> Result<String> {
    let mut hasher = Sha256::new();

    // Hash audit_id
    hasher.update(envelope.audit_id.as_bytes());

    // Hash timestamp (RFC3339 format for determinism)
    hasher.update(envelope.timestamp.to_rfc3339().as_bytes());

    // Hash service_id
    hasher.update(envelope.service_id.as_bytes());

    // Hash event (JSON serialized)
    // Note: serde_json serialization is deterministic for our event types
    let event_json =
        serde_json::to_string(&envelope.event).map_err(AuditError::Serialization)?;
    hasher.update(event_json.as_bytes());

    // Hash prev_hash
    hasher.update(envelope.prev_hash.as_bytes());

    // Return hex-encoded hash
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify hash chain integrity
///
/// Verifies:
/// 1. Each event's hash matches computed hash
/// 2. Each event's prev_hash matches previous event's hash
///
/// # Errors
///
/// Returns `InvalidChain` if any hash doesn't match.
/// Returns `BrokenChain` if prev_hash link is broken.
pub fn verify_hash_chain(events: &[AuditEventEnvelope]) -> Result<()> {
    for (i, event) in events.iter().enumerate() {
        // Verify event hash
        let computed_hash = compute_event_hash(event)?;
        if computed_hash != event.hash {
            return Err(AuditError::InvalidChain(format!(
                "Event {} hash mismatch: expected {}, got {}",
                event.audit_id, event.hash, computed_hash
            )));
        }

        // Verify chain link (skip first event)
        if i > 0 {
            let prev_event = &events[i.wrapping_sub(1)];
            if event.prev_hash != prev_event.hash {
                return Err(AuditError::BrokenChain(format!(
                    "Chain broken at event {}: prev_hash {} doesn't match previous event hash {}",
                    event.audit_id, event.prev_hash, prev_event.hash
                )));
            }
        }
    }

    Ok(())
}

/// Sign audit event with HMAC-SHA256
#[cfg(feature = "platform")]
pub fn sign_event_hmac(_envelope: &AuditEventEnvelope, _key: &[u8]) -> String {
    // TODO: Implement with hmac crate
    // use hmac::{Hmac, Mac};
    // use sha2::Sha256;
    todo!("Implement HMAC-SHA256 signing")
}

/// Verify HMAC-SHA256 signature
#[cfg(feature = "platform")]
pub fn verify_signature_hmac(
    _envelope: &AuditEventEnvelope,
    _signature: &str,
    _key: &[u8],
) -> bool {
    // TODO: Implement HMAC verification
    todo!("Implement HMAC verification")
}

/// Sign audit event with Ed25519
#[cfg(feature = "platform")]
pub fn sign_event_ed25519(
    _envelope: &AuditEventEnvelope,
    _keypair: &[u8], // TODO: Use ed25519_dalek::Keypair
) -> String {
    // TODO: Implement with ed25519-dalek crate
    todo!("Implement Ed25519 signing")
}

/// Verify Ed25519 signature
#[cfg(feature = "platform")]
pub fn verify_signature_ed25519(
    _envelope: &AuditEventEnvelope,
    _signature: &str,
    _public_key: &[u8], // TODO: Use ed25519_dalek::PublicKey
) -> bool {
    // TODO: Implement Ed25519 verification
    todo!("Implement Ed25519 verification")
}

/// Compute file checksum (SHA-256)
pub fn compute_file_checksum(_path: &std::path::Path) -> Result<String> {
    // TODO: Implement file checksum
    todo!("Implement file checksum computation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{ActorInfo, AuditEvent, AuthMethod};
    use chrono::Utc;

    fn create_test_envelope(audit_id: &str, prev_hash: &str) -> AuditEventEnvelope {
        AuditEventEnvelope {
            audit_id: audit_id.to_string(),
            timestamp: Utc::now(),
            service_id: "test-service".to_string(),
            event: AuditEvent::AuthSuccess {
                timestamp: Utc::now(),
                actor: ActorInfo {
                    user_id: "test@example.com".to_string(),
                    ip: Some("127.0.0.1".parse().unwrap()),
                    auth_method: AuthMethod::BearerToken,
                    session_id: None,
                },
                method: AuthMethod::BearerToken,
                path: "/test".to_string(),
                service_id: "test-service".to_string(),
            },
            prev_hash: prev_hash.to_string(),
            hash: String::new(),
            signature: None,
        }
    }

    #[test]
    fn test_compute_event_hash_deterministic() {
        let envelope = create_test_envelope("audit-001", "0000");

        let hash1 = compute_event_hash(&envelope).unwrap();
        let hash2 = compute_event_hash(&envelope).unwrap();

        assert_eq!(hash1, hash2, "Hash should be deterministic");
        assert_eq!(hash1.len(), 64, "SHA-256 hash should be 64 hex chars");
    }

    #[test]
    fn test_compute_event_hash_different_for_different_events() {
        let envelope1 = create_test_envelope("audit-001", "0000");
        let envelope2 = create_test_envelope("audit-002", "0000");

        let hash1 = compute_event_hash(&envelope1).unwrap();
        let hash2 = compute_event_hash(&envelope2).unwrap();

        assert_ne!(hash1, hash2, "Different events should have different hashes");
    }

    #[test]
    fn test_compute_event_hash_includes_prev_hash() {
        let envelope1 = create_test_envelope("audit-001", "0000");
        let envelope2 = create_test_envelope("audit-001", "1111");

        let hash1 = compute_event_hash(&envelope1).unwrap();
        let hash2 = compute_event_hash(&envelope2).unwrap();

        assert_ne!(hash1, hash2, "Different prev_hash should result in different hash");
    }

    #[test]
    fn test_verify_hash_chain_valid() {
        let mut envelope1 = create_test_envelope(
            "audit-001",
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        envelope1.hash = compute_event_hash(&envelope1).unwrap();

        let mut envelope2 = create_test_envelope("audit-002", &envelope1.hash);
        envelope2.hash = compute_event_hash(&envelope2).unwrap();

        let mut envelope3 = create_test_envelope("audit-003", &envelope2.hash);
        envelope3.hash = compute_event_hash(&envelope3).unwrap();

        let events = vec![envelope1, envelope2, envelope3];

        assert!(verify_hash_chain(&events).is_ok(), "Valid chain should pass verification");
    }

    #[test]
    fn test_verify_hash_chain_detects_tampering() {
        let mut envelope1 = create_test_envelope(
            "audit-001",
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        envelope1.hash = compute_event_hash(&envelope1).unwrap();

        let mut envelope2 = create_test_envelope("audit-002", &envelope1.hash);
        envelope2.hash = compute_event_hash(&envelope2).unwrap();

        // Tamper with envelope1's hash
        envelope1.hash =
            "tampered_hash_0000000000000000000000000000000000000000000000000".to_string();

        let events = vec![envelope1, envelope2];

        assert!(verify_hash_chain(&events).is_err(), "Tampered hash should be detected");
    }

    #[test]
    fn test_verify_hash_chain_detects_broken_link() {
        let mut envelope1 = create_test_envelope(
            "audit-001",
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        envelope1.hash = compute_event_hash(&envelope1).unwrap();

        let mut envelope2 = create_test_envelope("audit-002", "wrong_prev_hash");
        envelope2.hash = compute_event_hash(&envelope2).unwrap();

        let events = vec![envelope1, envelope2];

        assert!(verify_hash_chain(&events).is_err(), "Broken chain link should be detected");
    }

    #[test]
    fn test_verify_hash_chain_empty() {
        let events: Vec<AuditEventEnvelope> = vec![];
        assert!(verify_hash_chain(&events).is_ok(), "Empty chain should be valid");
    }

    #[test]
    fn test_verify_hash_chain_single_event() {
        let mut envelope = create_test_envelope(
            "audit-001",
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        envelope.hash = compute_event_hash(&envelope).unwrap();

        let events = vec![envelope];

        assert!(verify_hash_chain(&events).is_ok(), "Single event chain should be valid");
    }

    #[test]
    fn test_hash_includes_all_fields() {
        let mut envelope = create_test_envelope("audit-001", "0000");
        let hash1 = compute_event_hash(&envelope).unwrap();

        // Change service_id
        envelope.service_id = "different-service".to_string();
        let hash2 = compute_event_hash(&envelope).unwrap();

        assert_ne!(hash1, hash2, "Hash should include service_id");
    }
}
