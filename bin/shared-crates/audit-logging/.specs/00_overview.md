# Audit Logging — Overview & Core Specification

**Crate**: `bin/shared-crates/audit-logging`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Executive Summary

`audit-logging` provides **immutable, tamper-evident security audit trails** for compliance (GDPR, SOC2, ISO 27001) and forensic investigation across all llama-orch services. This is distinct from `narration-core`, which provides developer-focused observability.

**Key Distinction**: Audit logging answers **WHO did WHAT WHEN** (security), while narration answers **WHY things happened** (observability).

---

## 1. Purpose & Scope

### 1.1 Purpose

Audit logging provides an **immutable record of security-critical events** for:

- **Compliance**: GDPR data processing records, SOC2 audit trails, ISO 27001 security event logging
- **Forensics**: Security incident investigation, dispute resolution
- **Accountability**: Provider accountability in platform mode, customer trust
- **Legal liability**: Immutable proof of what happened when

### 1.2 Scope

**In Scope**:
- Authentication & authorization events
- Resource operations (admin actions)
- Data access events (GDPR compliance)
- Security incidents (rate limits, policy violations)
- VRAM operations (seal/verify/allocate/deallocate)
- Compliance events (GDPR requests, data erasure)

**Out of Scope**:
- Developer observability (use `narration-core`)
- Performance metrics (use metrics crates)
- Debug logging (use `tracing`)
- User-facing error messages

---

## 2. Audit Logging vs Narration

| Feature | **audit-logging** (Security) | **narration-core** (Observability) |
|---------|------------------------------|-------------------------------------|
| **Purpose** | Compliance, forensics | Debugging, performance |
| **Audience** | Auditors, security teams | Developers, SREs |
| **Format** | Machine-readable events | Human-readable narratives |
| **Content** | WHO did WHAT WHEN | WHY things happened, causality |
| **Retention** | Years (regulatory) | Days/weeks |
| **Mutability** | Immutable, append-only | Can rotate/delete |
| **Volume** | Low (critical events only) | High (verbose flows) |
| **Query** | Structured queries | Text search |

**Example: User deletes a pool**

**Narration** (observability):
```json
{
  "event": "pool_lifecycle",
  "narration": "Pool pool-123 (llama-3.1-8b) drained and deleted",
  "correlation_id": "orchd-a1b2c3",
  "duration_ms": 1200,
  "replicas_terminated": 4
}
```

**Audit Log** (security):
```json
{
  "timestamp": "2025-10-01T16:48:05Z",
  "event_type": "pool.delete",
  "actor": {
    "user_id": "admin@llorch.io",
    "ip": "192.168.1.100",
    "auth_method": "bearer_token"
  },
  "resource": {
    "pool_id": "pool-123",
    "model_ref": "llama-3.1-8b",
    "node_id": "gpu-node-1"
  },
  "result": "success",
  "audit_id": "audit-2025-1001-164805-abc123"
}
```

---

## 3. Architecture

### 3.1 Single-Node Mode (Local Audit)

In single-node deployments, audit logs are stored locally:

```
orchestratord/pool-managerd/worker-orcd
         ↓
    audit-logging crate
         ↓
    Local storage (.audit/)
    - Append-only file
    - Hash-chained entries
    - Encrypted at rest
```

**Storage location**: `/var/lib/llorch/audit/{service-name}/`

### 3.2 Platform Mode (Centralized Audit)

In platform/marketplace mode, providers emit audit events to the central platform:

```
Provider's orchestratord
         ↓
    audit-logging crate
         ↓
    audit-client (HTTP)
         ↓
Platform Audit Service (audit.llama-orch-platform.com)
         ↓
    Immutable Storage (S3, TimescaleDB)
         ↓
    Audit Query API (customers, regulators)
```

**Why centralized?**
- **Legal liability**: Platform is liable for GDPR/SOC2, not providers
- **Customer trust**: One audit trail, not scattered across 50 providers
- **Provider accountability**: Detect malicious providers
- **Dispute resolution**: Immutable proof of what happened

---

## 4. Core Principles

### 4.1 Immutability

**Audit logs MUST be append-only**:
- No updates to existing events
- No deletions (except for retention policy)
- Tamper-evident storage (hash chain)

### 4.2 Tamper Evidence

**Each event MUST include hash of previous event**:
```rust
pub struct AuditEventEnvelope {
    pub prev_hash: String,  // SHA-256 of previous event
    pub hash: String,       // SHA-256 of this event
}
```

**Verification**:
```rust
impl AuditLogger {
    pub fn verify_integrity(&self) -> Result<(), AuditError> {
        // Verify hash chain
        // Detect tampering
    }
}
```

### 4.3 Sensitive Data Redaction

**MUST NOT log**:
- Raw API tokens (use fingerprints: `token:a3f2c1`)
- VRAM pointers
- Seal secret keys
- Raw model bytes
- Internal memory addresses
- Prompt content (log length/hash only)
- Customer PII (unless required for GDPR)

**Example**:
```rust
// ❌ FORBIDDEN
AuditEvent::VramSealed {
    vram_ptr: 0x7f8a4c000000,  // Never log pointers
}

// ✅ CORRECT
AuditEvent::VramSealed {
    shard_id: "shard-abc123",  // Opaque ID only
}
```

### 4.4 Access Control

**Audit logs MUST be read-only**:
- Only administrators can read
- No API for updates/deletes
- Audit log access is itself audited

---

## 5. Event Categories

### 5.1 Authentication & Authorization

```rust
AuditEvent::AuthSuccess { actor, method, ip }
AuditEvent::AuthFailure { attempted_user, reason, ip }
AuditEvent::TokenCreated { actor, scope, expires_at }
AuditEvent::TokenRevoked { actor, token_id }
```

### 5.2 Resource Operations (Admin Actions)

```rust
AuditEvent::PoolCreated { actor, pool_id, model_ref, node_id }
AuditEvent::PoolDeleted { actor, pool_id, reason }
AuditEvent::NodeRegistered { actor, node_id, capacity }
AuditEvent::NodeDeregistered { actor, node_id, reason }
```

### 5.3 Data Access (GDPR)

```rust
AuditEvent::InferenceExecuted { customer_id, model_ref, tokens, provider_id }
AuditEvent::ModelAccessed { customer_id, model_ref, provider_id }
AuditEvent::DataDeleted { customer_id, reason: "gdpr_erasure" }
```

### 5.4 Security Incidents

```rust
AuditEvent::RateLimitExceeded { ip, endpoint, limit }
AuditEvent::PathTraversalAttempt { actor, attempted_path }
AuditEvent::InvalidTokenUsed { ip, token_prefix }
AuditEvent::SealVerificationFailed { shard_id, reason, severity }
```

### 5.5 VRAM Operations (Security-Critical)

```rust
AuditEvent::VramSealed { shard_id, gpu_device, vram_bytes, digest, worker_id }
AuditEvent::SealVerified { shard_id, result, worker_id }
AuditEvent::SealVerificationFailed { shard_id, reason, expected_digest, actual_digest, severity }
AuditEvent::VramAllocated { requested_bytes, allocated_bytes, available_bytes, gpu_device }
AuditEvent::VramDeallocated { shard_id, freed_bytes, remaining_used, gpu_device }
AuditEvent::PolicyViolation { policy, violation, details, severity, action_taken }
```

### 5.6 Compliance Events

```rust
AuditEvent::GdprDataAccessRequest { customer_id, requester, scope }
AuditEvent::GdprDataExport { customer_id, data_types }
AuditEvent::GdprRightToErasure { customer_id, completed_at }
```

---

## 6. Standard Event Format

### 6.1 Envelope Structure

**All events MUST include**:
```rust
pub struct AuditEventEnvelope {
    pub audit_id: String,           // Unique ID: "audit-2025-1001-164805-abc123"
    pub timestamp: DateTime<Utc>,   // ISO 8601 UTC
    pub event_type: String,         // "vram.sealed", "auth.success", etc.
    pub service_id: String,         // "orchestratord", "worker-gpu-0"
    pub event: AuditEvent,          // Event-specific data
    pub prev_hash: String,          // SHA-256 of previous event (tamper evidence)
    pub hash: String,               // SHA-256 of this event
    pub signature: Option<String>,  // HMAC signature (platform mode)
}
```

### 6.2 Actor Information

```rust
pub struct ActorInfo {
    pub user_id: String,           // "admin@example.com" or "customer-123"
    pub ip: Option<IpAddr>,        // Source IP address
    pub auth_method: AuthMethod,   // BearerToken, ApiKey, mTLS
    pub session_id: Option<String>, // For correlation
}
```

### 6.3 Resource Information

```rust
pub struct ResourceInfo {
    pub resource_type: String,  // "pool", "node", "job", "shard"
    pub resource_id: String,    // "pool-123", "shard-abc123"
    pub parent_id: Option<String>, // "node-1" (parent of pool)
}
```

### 6.4 Result Types

```rust
pub enum AuditResult {
    Success,
    Failure { reason: String },
    PartialSuccess { details: String },
}
```

---

## 7. Security Requirements

### 7.1 Cryptographic Requirements

**Hash algorithm**: SHA-256 (FIPS 140-2 approved)  
**Signature algorithm** (platform mode): HMAC-SHA256 or Ed25519  
**Random ID generation**: Cryptographically secure (not timestamp-based)

### 7.2 Storage Requirements

**Local mode**:
- Append-only files with `.audit` extension
- Daily rotation: `2025-10-01.audit`
- Checksums: `2025-10-01.audit.sha256`
- Manifest: `manifest.json` (file index)

**Platform mode**:
- S3 with Object Lock (WORM mode)
- Glacier for long-term archival
- Cross-region replication
- Encrypted at rest (KMS)

### 7.3 Retention Requirements

| Regulation | Retention | Notes |
|------------|-----------|-------|
| **GDPR** | 1-7 years | Must prove data handling |
| **SOC2** | 7 years | Auditor access required |
| **ISO 27001** | 3 years | Security event records |
| **HIPAA** | 6 years | If handling health data |

### 7.4 Performance Requirements

**Audit MUST NOT block operations**:
- Async emission (non-blocking)
- Buffering and batching (1 second or 100 events)
- Flush on critical events (seal verification failure)
- Flush on service shutdown

---

## 8. API Surface

### 8.1 Logger Initialization

**Local mode**:
```rust
use audit_logging::AuditLogger;

let logger = AuditLogger::local("./audit/")?;
```

**Platform mode**:
```rust
use audit_logging::{AuditLogger, PlatformConfig};

let logger = AuditLogger::platform(PlatformConfig {
    endpoint: "https://audit.llama-orch-platform.com".to_string(),
    provider_id: "provider-a".to_string(),
    provider_key: load_provider_key()?,
    batch_size: 100,
    flush_interval: Duration::from_secs(10),
})?;
```

### 8.2 Event Emission

**Emit audit event**:
```rust
logger.emit(AuditEvent::PoolDeleted {
    timestamp: Utc::now(),
    actor: ActorInfo {
        user_id: "admin@example.com".to_string(),
        ip: Some("192.168.1.100".parse()?),
        auth_method: AuthMethod::BearerToken,
    },
    resource: ResourceInfo {
        pool_id: "pool-123".to_string(),
        model_ref: "llama-3.1-8b".to_string(),
        node_id: "gpu-node-1".to_string(),
    },
    result: AuditResult::Success,
    metadata: json!({"replicas": 4, "reason": "user_requested"}),
}).await?;
```

### 8.3 Querying Audit Logs

```rust
use audit_logging::{AuditQuery, AuditLogger};

let logger = AuditLogger::local("./audit/")?;

let events = logger.query(AuditQuery {
    actor: Some("admin@example.com".to_string()),
    start_time: Some(Utc::now() - Duration::days(7)),
    end_time: Some(Utc::now()),
    event_types: vec!["pool.delete", "pool.create"],
    limit: 100,
}).await?;
```

### 8.4 Integrity Verification

```rust
// Verify hash chain integrity
logger.verify_integrity().await?;

// Verify signature (platform mode)
logger.verify_signatures().await?;
```

---

## 9. Dependencies

### 9.1 Internal Dependencies

- `contracts/api-types` — Audit event types
- `libs/auth-min` — Actor authentication info
- `libs/narration-core` — (separate, for observability)

### 9.2 External Dependencies

- `serde` — Serialization
- `tokio` — Async I/O
- `chrono` — Timestamps
- `sha2` — Checksumming and hash chains
- `ed25519-dalek` — Signatures (platform mode)
- `reqwest` — HTTP client (platform mode)
- `thiserror` — Error types

---

## 10. Implementation Phases

### Phase 1: M0 Essentials (Current Sprint)

1. ✅ Event types for core services
2. ✅ Local file-based audit logger
3. ✅ Async event emission
4. ✅ Basic tamper evidence (hash chain)
5. ⬜ VRAM operation events (for vram-residency)
6. ⬜ Input validation integration

### Phase 2: Production Hardening (Post-M0)

7. ⬜ Platform mode support
8. ⬜ Event buffering and batching
9. ⬜ Integrity verification API
10. ⬜ Query API with filters
11. ⬜ Signature verification
12. ⬜ Retention policy enforcement

### Phase 3: Advanced Features (Future)

13. ⬜ Export to SIEM (Splunk, ELK)
14. ⬜ Real-time anomaly detection
15. ⬜ Automated compliance reports (SOC2, GDPR)
16. ⬜ Audit event analytics dashboard
17. ⬜ Multi-region audit replication

---

## 11. References

**Security Documentation**:
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Existing codebase vulnerabilities
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Worker-orcd security requirements
- `.docs/security/SECURITY_OVERSEER_SUMMARY.md` — Security posture assessment

**Specifications**:
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — VRAM security requirements
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer expectations
- `bin/shared-crates/audit-logging/.specs/11_worker_vram_residency.md` — VRAM audit requirements

**Standards**:
- GDPR — Data processing records
- SOC2 — Audit trail requirements
- ISO 27001 — Security event logging
- FIPS 140-2 — Cryptographic standards

---

## 12. Refinement Opportunities

### 12.1 Immediate Improvements

1. **Add structured event types** for all services (orchestratord, pool-managerd, worker-orcd)
2. **Implement file-based storage** with append-only guarantees
3. **Add hash chain verification** to detect tampering
4. **Integrate with input-validation** to prevent log injection

### 12.2 Medium-Term Enhancements

5. **Add platform mode** with HTTP client and signature support
6. **Implement query API** with time-range and actor filters
7. **Add retention policy** with automatic archival/deletion
8. **Create compliance reports** (GDPR, SOC2) from audit logs

### 12.3 Long-Term Vision

9. **Real-time streaming** to SIEM systems
10. **Anomaly detection** using ML on audit patterns
11. **Blockchain-backed** tamper evidence for high-security deployments
12. **Multi-tenant isolation** with per-customer audit trails

---

**End of Overview Specification**
