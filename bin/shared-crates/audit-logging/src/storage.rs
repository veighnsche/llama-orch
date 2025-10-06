//! Storage layer for audit events
//!
//! Handles tamper-evident storage with hash chains.

use crate::error::Result;
use crate::events::AuditEvent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Audit event envelope with tamper-evidence
///
/// Each event includes:
/// - Unique audit ID
/// - Timestamp
/// - Service ID
/// - Event data
/// - Hash chain (prev_hash, hash)
/// - Optional signature (platform mode)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEventEnvelope {
    /// Unique audit event ID
    pub audit_id: String,

    /// Event timestamp (ISO 8601 UTC)
    pub timestamp: DateTime<Utc>,

    /// Service that emitted the event
    pub service_id: String,

    /// The actual audit event
    pub event: AuditEvent,

    /// Hash of previous event (blockchain-style chain)
    pub prev_hash: String,

    /// Hash of this event
    pub hash: String,

    /// Signature (platform mode only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl AuditEventEnvelope {
    /// Create new envelope (hash will be computed by writer)
    #[must_use] 
    pub fn new(
        audit_id: String,
        timestamp: DateTime<Utc>,
        service_id: String,
        event: AuditEvent,
        prev_hash: String,
    ) -> Self {
        Self {
            audit_id,
            timestamp,
            service_id,
            event,
            prev_hash,
            hash: String::new(), // Computed by writer
            signature: None,
        }
    }
}

/// Manifest entry for audit file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Filename
    pub filename: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Close timestamp (None if still open)
    pub closed_at: Option<DateTime<Utc>>,

    /// Number of events in file
    pub event_count: usize,

    /// First audit ID in file
    pub first_audit_id: String,

    /// Last audit ID in file
    pub last_audit_id: String,

    /// SHA-256 checksum of file (None if still open)
    pub sha256: Option<String>,

    /// File size in bytes
    pub size_bytes: u64,
}

/// Manifest file (index of all audit files)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Service ID
    pub service_id: String,

    /// Manifest creation timestamp
    pub created_at: DateTime<Utc>,

    /// List of audit files
    pub files: Vec<ManifestEntry>,
}

impl Manifest {
    /// Create new manifest
    pub fn new(service_id: String) -> Self {
        Self { service_id, created_at: Utc::now(), files: Vec::new() }
    }

    /// Add file entry
    pub fn add_file(&mut self, entry: ManifestEntry) {
        self.files.push(entry);
    }

    /// Load manifest from file
    pub fn load(_path: &std::path::Path) -> Result<Self> {
        // TODO: Implement
        todo!("Load manifest from file")
    }

    /// Save manifest to file
    pub fn save(&self, _path: &std::path::Path) -> Result<()> {
        // TODO: Implement
        todo!("Save manifest to file")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{ActorInfo, AuditEvent, AuthMethod};

    #[test]
    fn test_envelope_new() {
        let event = AuditEvent::AuthSuccess {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "test@example.com".to_string(),
                ip: Some("127.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            method: AuthMethod::BearerToken,
            path: "/test".to_string(),
            service_id: "test".to_string(),
        };

        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            event,
            "prev_hash".to_string(),
        );

        assert_eq!(envelope.audit_id, "audit-001");
        assert_eq!(envelope.service_id, "test-service");
        assert_eq!(envelope.prev_hash, "prev_hash");
        assert_eq!(envelope.hash, "");
        assert!(envelope.signature.is_none());
    }

    #[test]
    fn test_envelope_serialization() {
        let event = AuditEvent::AuthSuccess {
            timestamp: Utc::now(),
            actor: ActorInfo {
                user_id: "test@example.com".to_string(),
                ip: Some("127.0.0.1".parse().unwrap()),
                auth_method: AuthMethod::BearerToken,
                session_id: None,
            },
            method: AuthMethod::BearerToken,
            path: "/test".to_string(),
            service_id: "test".to_string(),
        };

        let envelope = AuditEventEnvelope::new(
            "audit-001".to_string(),
            Utc::now(),
            "test-service".to_string(),
            event,
            "prev_hash".to_string(),
        );

        let json = serde_json::to_string(&envelope).unwrap();
        assert!(json.contains("audit-001"));
        assert!(json.contains("test-service"));
    }

    #[test]
    fn test_envelope_deserialization() {
        let json = r#"{
            "audit_id": "audit-001",
            "timestamp": "2024-01-01T00:00:00Z",
            "service_id": "test-service",
            "event": {
                "event_type": "auth_success",
                "timestamp": "2024-01-01T00:00:00Z",
                "actor": {
                    "user_id": "test@example.com",
                    "ip": "127.0.0.1",
                    "auth_method": "bearer_token"
                },
                "method": "bearer_token",
                "path": "/test",
                "service_id": "test"
            },
            "prev_hash": "prev",
            "hash": "current"
        }"#;

        let envelope: AuditEventEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(envelope.audit_id, "audit-001");
        assert_eq!(envelope.service_id, "test-service");
    }

    #[test]
    fn test_manifest_new() {
        let manifest = Manifest::new("test-service".to_string());
        assert_eq!(manifest.service_id, "test-service");
        assert!(manifest.files.is_empty());
    }

    #[test]
    fn test_manifest_add_file() {
        let mut manifest = Manifest::new("test-service".to_string());

        let entry = ManifestEntry {
            filename: "2024-01-01.audit".to_string(),
            created_at: Utc::now(),
            closed_at: None,
            event_count: 100,
            first_audit_id: "audit-001".to_string(),
            last_audit_id: "audit-100".to_string(),
            sha256: None,
            size_bytes: 10240,
        };

        manifest.add_file(entry);
        assert_eq!(manifest.files.len(), 1);
        assert_eq!(manifest.files[0].filename, "2024-01-01.audit");
    }
}
