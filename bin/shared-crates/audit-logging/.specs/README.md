# Audit Logging — Specification Index

**Crate**: `bin/shared-crates/audit-logging`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## Overview

This directory contains the complete specification for the `audit-logging` crate, which provides **immutable, tamper-evident security audit trails** for compliance (GDPR, SOC2, ISO 27001) and forensic investigation.

**Key Distinction**: Audit logging is the **serious cousin** of `narration-core`:
- **audit-logging**: WHO did WHAT WHEN (security, compliance, forensics)
- **narration-core**: WHY things happened (observability, debugging)

---

## Specification Documents

### Core Specifications

| Document | Description | Status |
|----------|-------------|--------|
| **[00-overview.md](./00-overview.md)** | Overview, architecture, core principles | ✅ Complete |
| **[10-event-types.md](./10-event-types.md)** | All audit event types and schemas | ✅ Complete |
| **[20-storage-and-tamper-evidence.md](./20-storage-and-tamper-evidence.md)** | Storage, hash chains, integrity verification | ✅ Complete |
| **[30-security-and-api.md](./30-security-and-api.md)** | Security requirements, access control, API | ✅ Complete |

### Consumer-Specific Requirements

| Document | Description | Consumer | Status |
|----------|-------------|----------|--------|
| **[11_worker_vram_residency.md](./11_worker_vram_residency.md)** | VRAM operation audit requirements | `vram-residency` | ✅ Complete |

---

## Quick Start

### For Implementers

**Read in this order**:
1. **00-overview.md** — Understand purpose, architecture, and principles
2. **10-event-types.md** — Learn event schemas and categories
3. **20-storage-and-tamper-evidence.md** — Understand storage and integrity
4. **30-security-and-api.md** — Implement security and API

### For Consumers

**Read in this order**:
1. **00-overview.md** — Understand what audit logging provides
2. **10-event-types.md** — Find events relevant to your service
3. **30-security-and-api.md** — Learn how to emit events
4. **Consumer-specific requirements** (if applicable)

---

## Key Concepts

### 1. Immutability

Audit logs are **append-only**:
- ✅ New events can be appended
- ❌ Existing events CANNOT be modified
- ❌ Existing events CANNOT be deleted (except by retention policy)

### 2. Tamper Evidence

Every event includes:
- **prev_hash**: SHA-256 of previous event (blockchain-style chain)
- **hash**: SHA-256 of this event
- **signature**: HMAC/Ed25519 signature (platform mode)

Tampering breaks the hash chain and is immediately detectable.

### 3. Sensitive Data Redaction

**Never log**:
- Full API tokens (use fingerprints: `token:a3f2c1`)
- VRAM pointers
- Prompt content (log length/hash only)
- Raw passwords or keys

### 4. Non-Blocking Performance

Audit logging **must not block operations**:
- Async event emission
- Buffering (1000 events or 10MB)
- Batching (flush every 1 second or 100 events)
- Graceful degradation (drop events if buffer full)

---

## Event Categories

| Category | Event Count | Priority | Consumers |
|----------|-------------|----------|-----------|
| **Authentication** | 4 | P0 | All services |
| **Authorization** | 3 | P0 | All services |
| **Resource Operations** | 8 | P1 | rbees-orcd, pool-managerd |
| **VRAM Operations** | 6 | P0 | worker-orcd, vram-residency |
| **Data Access** | 3 | P1 | rbees-orcd (GDPR) |
| **Security Incidents** | 5 | P0 | All services |
| **Compliance** | 3 | P2 | Platform mode |

**Total**: 32 event types

See **[10-event-types.md](./10-event-types.md)** for complete list.

---

## Architecture Modes

### Local Mode (Single-Node)

```
rbees-orcd/pool-managerd/worker-orcd
         ↓
    audit-logging crate
         ↓
    Local storage (.audit/)
    - Append-only files
    - Hash-chained entries
    - Daily rotation
```

**Storage**: `/var/lib/llorch/audit/{service-name}/`

### Platform Mode (Marketplace)

```
Provider's service
         ↓
    audit-logging crate
         ↓
    audit-client (HTTP)
         ↓
Platform Audit Service
         ↓
    Immutable Storage (S3, TimescaleDB)
```

**Why centralized?**
- Legal liability (platform is liable for GDPR/SOC2)
- Customer trust (one audit trail, not scattered)
- Provider accountability (detect malicious providers)

---

## Implementation Phases

### Phase 1: M0 Essentials (Current Sprint)

- [x] Event types for core services
- [x] Local file-based audit logger
- [x] Async event emission
- [x] Basic tamper evidence (hash chain)
- [ ] VRAM operation events (for vram-residency)
- [ ] Input validation integration

### Phase 2: Production Hardening (Post-M0)

- [ ] Platform mode support
- [ ] Event buffering and batching
- [ ] Integrity verification API
- [ ] Query API with filters
- [ ] Signature verification
- [ ] Retention policy enforcement

### Phase 3: Advanced Features (Future)

- [ ] Export to SIEM (Splunk, ELK)
- [ ] Real-time anomaly detection
- [ ] Automated compliance reports (SOC2, GDPR)
- [ ] Multi-region audit replication

---

## Security Considerations

### Threat Model

| Threat | Mitigation |
|--------|------------|
| **Unauthorized access** | Access control, encryption |
| **Tampering** | Hash chains, signatures |
| **Deletion** | Append-only storage, backups |
| **Injection** | Input validation, sanitization |
| **Denial of service** | Buffering, async writes |
| **Exfiltration** | Encryption, redaction |

See **[30-security-and-api.md](./30-security-and-api.md)** for details.

---

## Compliance Requirements

### Retention Periods

| Regulation | Minimum Retention | Recommended |
|------------|-------------------|-------------|
| **GDPR** | 1 year | 3 years |
| **SOC2** | 7 years | 7 years |
| **ISO 27001** | 3 years | 5 years |
| **HIPAA** | 6 years | 6 years |

**Default**: 7 years (SOC2 requirement)

### Regulatory Standards

- **GDPR**: Data processing records (Article 30)
- **SOC2**: Audit trail requirements (CC7.2)
- **ISO 27001**: Security event logging (A.12.4.1)
- **FIPS 140-2**: Cryptographic standards (SHA-256, AES-256)

---

## API Quick Reference

### Initialization

```rust
use audit_logging::{AuditLogger, AuditConfig};

let logger = AuditLogger::new(AuditConfig {
    mode: AuditMode::Local {
        base_dir: PathBuf::from("/var/lib/llorch/audit/rbees-orcd"),
    },
    service_id: "rbees-orcd".to_string(),
    rotation_policy: RotationPolicy::Daily,
    retention_policy: RetentionPolicy::default(),
})?;
```

### Event Emission

```rust
logger.emit(AuditEvent::AuthSuccess {
    timestamp: Utc::now(),
    actor: ActorInfo {
        user_id: "admin@llorch.io".to_string(),
        ip: Some("192.168.1.100".parse()?),
        auth_method: AuthMethod::BearerToken,
        session_id: Some("sess-abc123".to_string()),
    },
    method: AuthMethod::BearerToken,
    path: "/v2/tasks".to_string(),
    service_id: "rbees-orcd".to_string(),
}).await?;
```

### Query

```rust
let events = logger.query(AuditQuery {
    actor: Some("admin@example.com".to_string()),
    start_time: Some(Utc::now() - Duration::days(7)),
    end_time: Some(Utc::now()),
    event_types: vec!["pool.delete", "pool.create"],
    limit: 100,
}).await?;
```

### Integrity Verification

```rust
let result = logger.verify_integrity(VerifyOptions {
    mode: VerifyMode::LastN(1000),
}).await?;
```

See **[30-security-and-api.md](./30-security-and-api.md)** for complete API.

---

## References

### Internal Documentation

- **[README.md](../README.md)** — Crate overview and usage
- **[src/lib.rs](../src/lib.rs)** — Implementation
- **[.docs/security/](../../../../.docs/security/)** — Security audit documents

### Security Audits

- **SECURITY_AUDIT_EXISTING_CODEBASE.md** — Existing codebase vulnerabilities
- **SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md** — Worker-orcd security requirements
- **SECURITY_OVERSEER_SUMMARY.md** — Security posture assessment

### Standards & Regulations

- **GDPR** — General Data Protection Regulation (EU)
- **SOC2** — Service Organization Control 2 (AICPA)
- **ISO 27001** — Information Security Management (ISO)
- **FIPS 140-2** — Cryptographic Module Validation (NIST)

---

## Consumer Integration

### rbees-orcd

**Events to emit**:
- `auth.success`, `auth.failure` (authentication)
- `task.submitted`, `task.completed`, `task.canceled` (task lifecycle)
- `node.registered`, `node.deregistered` (node management)

**Integration points**:
- Authentication middleware
- Task submission handler
- Node registration handler

### pool-managerd

**Events to emit**:
- `pool.created`, `pool.deleted`, `pool.modified` (pool lifecycle)
- `auth.success`, `auth.failure` (authentication)

**Integration points**:
- Pool creation/deletion handlers
- Authentication middleware

### worker-orcd

**Events to emit**:
- `vram.sealed`, `seal.verified`, `seal.verification_failed` (VRAM security)
- `vram.allocated`, `vram.allocation_failed`, `vram.deallocated` (VRAM operations)
- `security.policy_violation` (policy enforcement)

**Integration points**:
- VRAM seal operations
- VRAM allocation/deallocation
- Policy enforcement checks

See **[11_worker_vram_residency.md](./11_worker_vram_residency.md)** for detailed VRAM requirements.

---

## Testing Strategy

### Unit Tests

- Hash chain verification
- Tampering detection
- Event serialization/deserialization
- Input validation

### Integration Tests

- End-to-end audit flow
- Multi-service audit correlation
- Query API functionality
- Integrity verification

### Security Tests

- Log injection attempts
- Unauthorized access attempts
- Tampering detection
- Signature verification

---

## Refinement Opportunities

### Immediate (P0)

1. Implement file-based storage with append-only guarantees
2. Add hash chain verification on startup
3. Integrate with input-validation crate
4. Add VRAM operation events for vram-residency

### Medium-Term (P1)

5. Add platform mode with HTTP client and signatures
6. Implement query API with filters and pagination
7. Add retention policy enforcement
8. Create compliance reports (GDPR, SOC2)

### Long-Term (P2)

9. Real-time streaming to SIEM systems
10. Anomaly detection using ML
11. Blockchain integration for ultimate tamper evidence
12. Multi-tenant isolation with per-customer audit trails

---

## Contributing

When adding new event types:

1. **Add to event taxonomy** in `10-event-types.md`
2. **Define schema** with all required fields
3. **Document security considerations** (what NOT to log)
4. **Add usage examples** for consumers
5. **Update event type registry** table

When modifying storage:

1. **Maintain backward compatibility** with existing logs
2. **Preserve hash chain integrity**
3. **Update integrity verification** if needed
4. **Test tampering detection**

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Priority**: P0 for platform mode, P1 for single-node

**Current Implementation**: Basic event types and tracing-based logging. File-based storage and tamper evidence pending.

---

**For questions or clarifications, see the individual specification documents or the security audit documents in `.docs/security/`.**
