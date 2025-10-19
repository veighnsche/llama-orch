# Team Audit Logging — Responsibilities

**Who We Are**: The serious, security-focused sibling of narration-core  
**What We Do**: Immutable, tamper-evident audit trails for compliance and forensics  
**Our Mood**: Vigilant, uncompromising, and deeply paranoid (in a good way)

---

## Our Mission

We exist to provide **legally defensible proof** of what happened in llama-orch. Every security-critical action gets an immutable audit record that can withstand:

- **Regulatory audits** (GDPR, SOC2, ISO 27001)
- **Forensic investigation** (security incidents, data breaches)
- **Legal proceedings** (dispute resolution, compliance violations)
- **Customer trust** (prove we handled their data correctly)

We are the **last line of defense** when someone asks "what happened?" and the answer matters legally.

---

## Our Relationship with narration-core

We are **siblings, not competitors**. We serve different masters:

| Aspect | **audit-logging** (Us) | **narration-core** (Our Sibling) |
|--------|------------------------|-----------------------------------|
| **Purpose** | Legal compliance, forensics | Developer debugging, observability |
| **Audience** | Auditors, regulators, lawyers | Developers, SREs, operators |
| **Tone** | Formal, machine-readable | Human-readable, sometimes cute 🎀 |
| **Content** | WHO did WHAT WHEN | WHY things happened, causality |
| **Retention** | 7 years (regulatory requirement) | Days/weeks (operational need) |
| **Mutability** | **NEVER** — append-only, immutable | Can rotate, delete, archive |
| **Volume** | Low (critical events only) | High (verbose debugging flows) |
| **Security** | Tamper-evident, signed, encrypted | Redacted secrets, correlation IDs |

**Example: User deletes a pool**

**narration-core** says:
```
"Pool pool-123 (llama-3.1-8b) drained and deleted — all replicas stopped gracefully! 👋"
```

**We** say:
```json
{
  "audit_id": "audit-2025-1001-164805-abc123",
  "timestamp": "2025-10-01T16:48:05Z",
  "event_type": "pool.delete",
  "actor": {
    "user_id": "token:a3f2c1",
    "ip": "192.168.1.100",
    "auth_method": "bearer_token"
  },
  "resource": {
    "pool_id": "pool-123",
    "model_ref": "llama-3.1-8b",
    "node_id": "gpu-node-1"
  },
  "result": "success"
}
```

**Both are necessary.** narration-core helps developers debug. We help lawyers defend.

---

## What We Provide to Other Crates

### Core Capabilities

**1. Immutable Event Recording**
- Append-only audit trail with blockchain-style hash chains
- 32 pre-defined event types across 7 categories
- Synchronous, non-blocking emission (won't slow down operations)
- Automatic buffering and batching (1 second or 100 events)

**2. Compliance-Ready Storage**
- GDPR, SOC2, ISO 27001 compliant retention (7 years default)
- Local mode (single-node) or Platform mode (marketplace)
- Encrypted at rest, checksummed files
- Daily rotation with integrity verification

**3. Security-First Design**
- Automatic sensitive data redaction (no tokens, passwords, or pointers)
- Integration with `input-validation` for log injection prevention
- Cryptographic signatures (platform mode)
- Tamper detection via hash chain verification

**4. Query & Verification APIs**
- Query by actor, event type, resource, or time range
- Hash chain integrity verification
- File checksum verification
- Compliance report generation

### Event Types We Offer

**For queen-rbee**:
- `AuthSuccess`, `AuthFailure` — Authentication events
- `TaskSubmitted`, `TaskCompleted`, `TaskCanceled` — Task lifecycle
- `NodeRegistered`, `NodeDeregistered` — Node management

**For pool-managerd**:
- `PoolCreated`, `PoolDeleted`, `PoolModified` — Pool lifecycle

**For worker-orcd / vram-residency**:
- `VramSealed`, `SealVerified`, `SealVerificationFailed` — VRAM security
- `VramAllocated`, `VramAllocationFailed`, `VramDeallocated` — VRAM resources
- `PolicyViolation` — Security policy enforcement

**For all services**:
- `RateLimitExceeded`, `PathTraversalAttempt`, `InvalidTokenUsed` — Security incidents
- `SuspiciousActivity` — Anomaly detection

### Integration Pattern

**1. Initialize at startup**:
```rust
use audit_logging::{AuditLogger, AuditConfig, AuditMode};

let audit_logger = AuditLogger::new(AuditConfig {
    mode: AuditMode::Local {
        base_dir: PathBuf::from("/var/lib/llorch/audit/queen-rbee"),
    },
    service_id: "queen-rbee".to_string(),
    rotation_policy: RotationPolicy::Daily,
    retention_policy: RetentionPolicy::default(),
})?;
```

**2. Emit events (synchronous, non-blocking)**:
```rust
// Works from both sync and async contexts (no .await needed)
audit_logger.emit(AuditEvent::AuthSuccess {
    timestamp: Utc::now(),
    actor: ActorInfo {
        user_id: format!("token:{}", token_fp),  // Use fingerprint, not full token
        ip: Some(extract_ip(&req)),
        auth_method: AuthMethod::BearerToken,
        session_id: None,
    },
    method: AuthMethod::BearerToken,
    path: req.uri().path().to_string(),
    service_id: "queen-rbee".to_string(),
})?;
```

**3. Always sanitize user input**:
```rust
use input_validation::sanitize_string;

// CRITICAL: Always sanitize before logging
let safe_pool_id = sanitize_string(&pool_id)?;
let safe_reason = sanitize_string(&reason)?;

audit_logger.emit(AuditEvent::PoolDeleted {
    pool_id: safe_pool_id,  // ✅ Protected from log injection
    reason: safe_reason,
    ...
})?;
```

**4. Flush on shutdown**:
```rust
audit_logger.flush().await?;
```

---

## Our Guarantees

### Performance Guarantees

- **Non-blocking**: Synchronous emission with background writer task
- **Buffered**: Up to 1000 events or 10MB in memory
- **Batched**: Flush every 1 second or 100 events (whichever comes first)
- **Graceful degradation**: Drops events if buffer full (logs warning)
- **Critical events**: Immediate flush for security incidents

### Security Guarantees

- **Immutable**: Append-only, no updates or deletes
- **Tamper-evident**: Hash chain (blockchain-style) detects modifications
- **Redacted**: Never logs full tokens, passwords, VRAM pointers, or prompt content
- **Sanitized**: Integration with `input-validation` prevents log injection
- **Signed**: HMAC/Ed25519 signatures in platform mode

### Compliance Guarantees

- **GDPR**: 7-year retention, data access records, right to erasure tracking
- **SOC2**: Auditor access, 7-year retention, security event logging
- **ISO 27001**: 3-year retention, security incident records

---

## What We Are NOT

### We Are NOT narration-core

- **No cute messages** — We are formal and machine-readable
- **No emojis** — We are serious business
- **No "why" explanations** — We record facts, not narratives
- **No developer debugging** — Use narration-core for that

### We Are NOT a General-Purpose Logger

- **No debug logs** — Use `tracing` for that
- **No performance metrics** — Use metrics crates for that
- **No user-facing messages** — Use error types for that

### We Are NOT Optional

If you handle:
- **Authentication** → You MUST emit audit events
- **Authorization** → You MUST emit audit events
- **Data access** → You MUST emit audit events (GDPR requirement)
- **Admin actions** → You MUST emit audit events
- **Security incidents** → You MUST emit audit events

**Compliance is not negotiable.**

---

## What We NEVER Log

### Forbidden Content

- ❌ **Full API tokens** — Use `auth_min::fingerprint_token()` instead
- ❌ **Raw passwords or keys** — Never, ever, ever
- ❌ **VRAM pointers** (`0x7f8a4c000000`) — Security risk
- ❌ **Prompt content** — Use length and hash instead
- ❌ **Customer PII** — Unless required for GDPR compliance

### What We DO Log

- ✅ **Token fingerprints** (`token:a3f2c1`)
- ✅ **User IDs** (opaque identifiers)
- ✅ **Resource IDs** (pool_id, shard_id, task_id)
- ✅ **IP addresses** (for security monitoring)
- ✅ **Timestamps** (UTC, ISO 8601)
- ✅ **Action outcomes** (success/failure)

---

## Our Security Posture

### We Are Paranoid (By Design)

**Threat Model**: Audit logging is a **high-value target** for attackers because:
- Destroying audit logs covers tracks of intrusions
- Tampering with logs creates false evidence
- Injecting fake events frames innocent parties
- Exfiltrating logs reveals system behavior

**Our Defense**:
- **Input validation** — All fields sanitized against log injection
- **Tamper detection** — Hash chains detect any modification
- **Immutability** — Append-only storage, no updates/deletes
- **Encryption** — At rest (local mode) and in transit (platform mode)
- **Signatures** — Cryptographic proof of authenticity (platform mode)

### Attack Vectors We Defend Against

**1. Log Injection Attacks**:
- ANSI escape sequence injection (`\x1b[31mFAKE ERROR\x1b[0m`)
- Control character injection (`\r\n[CRITICAL] Fake log`)
- Null byte injection (`admin\0malicious`)
- Unicode directional override (`\u{202E}evil\u{202D}`)

**2. Tampering Attacks**:
- Modifying existing entries (detected by hash chain)
- Deleting entries (detected by hash chain)
- Inserting fake entries (detected by signatures in platform mode)

**3. Denial of Service**:
- Unbounded memory (bounded buffer: 1000 events max)
- Disk exhaustion (rotation policy, retention limits)
- CPU exhaustion (batched writes, rate limiting)

**4. Path Traversal**:
- Directory traversal in pool IDs (`pool-../../../etc/passwd`)
- Detected and rejected by `input-validation`

---

## Our Responsibilities

### What We Own

**1. Event Type Definitions**
- All 32 audit event types
- Event schemas and validation rules
- Event versioning and migration

**2. Storage & Integrity**
- Append-only file format
- Hash chain implementation
- Checksum verification
- Rotation and retention policies

**3. Security Hardening**
- Input validation integration
- Sensitive data redaction
- Tamper detection
- Cryptographic signatures (platform mode)

**4. Compliance Support**
- GDPR data access records
- SOC2 audit trail generation
- ISO 27001 security event logging
- Retention policy enforcement

**5. Query & Verification APIs**
- Query by actor, event type, resource, time range
- Hash chain integrity verification
- Compliance report generation

### What We Do NOT Own

**1. Authentication/Authorization Logic**
- Owned by `auth-min` and `auth-core`
- We only **record** auth events, not enforce them

**2. Input Validation Logic**
- Owned by `input-validation`
- We **use** it, but don't implement it

**3. Developer Observability**
- Owned by `narration-core`
- We are **complementary**, not competitive

**4. Metrics & Performance Monitoring**
- Owned by metrics crates
- We record **security events**, not performance data

---

## Our Standards

### We Are Uncompromising

**No exceptions. No shortcuts. No "just this once."**

- **Immutability**: Once written, audit logs are **NEVER** modified or deleted (except by retention policy)
- **Validation**: All input is **ALWAYS** sanitized, no matter the source
- **Redaction**: Secrets are **ALWAYS** redacted, no opt-out
- **Integrity**: Hash chains are **ALWAYS** verified on read
- **Retention**: Regulatory retention is **ALWAYS** enforced

### We Are Thorough

**BDD Coverage**: 25+ scenarios testing critical attack vectors
- ANSI escape injection (4 scenarios)
- Control character injection (4 scenarios)
- Null byte injection (4 scenarios)
- Path traversal (1 scenario)
- Log line injection (3 scenarios)

**Event Type Coverage**: 32 event types across 7 categories
- Authentication (4 types)
- Authorization (3 types)
- Resource operations (6 types)
- Task lifecycle (3 types)
- VRAM operations (6 types)
- Security incidents (4 types)
- Compliance (6 types)

### We Are Documented

**Specifications**: 11 spec documents totaling 200+ pages
- `00_overview.md` — Architecture and principles
- `01_event-types.md` — All 32 event types with schemas
- `02_storage-and-tamper-evidence.md` — Storage format and integrity
- `03_security-and-api.md` — API reference and security requirements
- `10_expectations.md` — What consumers can expect from us
- `11_worker_vram_residency.md` — VRAM-specific audit requirements
- `20_security.md` — Attack surface analysis (23 attack vectors)
- `21_security_verification.md` — Security verification plan
- `30_dependencies.md` — Dependency security analysis
- `41_property_testing.md` — Property-based testing strategy

---

## Our Philosophy

### Security Is Not Optional

Audit logging is **foundational to trust**. Without it:
- We cannot prove GDPR compliance
- We cannot pass SOC2 audits
- We cannot investigate security incidents
- We cannot resolve customer disputes
- We cannot defend against legal claims

**Every security-critical action MUST be audited.**

### Immutability Is Sacred

Once an audit event is written, it is **permanent**. We do not:
- Update events (append new events instead)
- Delete events (retention policy only)
- Modify timestamps (use original timestamp)
- Rewrite history (append corrections)

**The audit trail is the source of truth.**

### Compliance Is Non-Negotiable

Regulatory requirements are **hard constraints**, not suggestions:
- **GDPR**: 7-year retention, data access records, right to erasure tracking
- **SOC2**: Auditor access, security event logging, 7-year retention
- **ISO 27001**: Security incident records, 3-year retention

**We meet or exceed all requirements.**

---

## Our Message to Other Teams

### Dear queen-rbee, pool-managerd, worker-orcd, and vram-residency,

We built you the **legally defensible audit trail** you need for compliance and security. Please use it correctly:

**DO**:
- ✅ Emit audit events for all security-critical actions
- ✅ Use token fingerprints, not full tokens
- ✅ Sanitize all user input before logging
- ✅ Flush audit logger on graceful shutdown
- ✅ Handle `BufferFull` errors gracefully

**DON'T**:
- ❌ Log sensitive data (tokens, passwords, VRAM pointers)
- ❌ Skip audit events "because it's just a test"
- ❌ Modify or delete audit logs manually
- ❌ Ignore audit emission errors
- ❌ Use audit logging for debugging (use narration-core)

**We are here to protect you** — from regulators, from attackers, from legal liability. But we can only protect you if you use us correctly.

With vigilance and zero tolerance for shortcuts,  
**The Audit Logging Team** 🔒

---

## Current Status

- **Version**: 0.1.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha (security-hardened, production-ready validation)
- **Priority**: P0 for platform mode, P1 for single-node

### Recent Changes

- ✅ **v0.1.0**: Made `emit()` synchronous (breaking change, simpler API)
- ✅ **BDD Suite**: 25+ scenarios covering critical attack vectors
- ✅ **Input Validation**: Integration with `input-validation` crate
- ✅ **Documentation**: 11 spec documents, 200+ pages

### Next Steps

- ⬜ **Service Integration**: queen-rbee, pool-managerd, worker-orcd
- ⬜ **VRAM Events**: Full integration with vram-residency
- ⬜ **Platform Mode**: Centralized audit service, signatures
- ⬜ **Query API**: REST API for compliance reports

---

## Fun Facts (Well, Serious Facts)

- We have **32 event types** across 7 categories
- We defend against **23 attack vectors**
- We wrote **25+ BDD scenarios** for security validation
- We have **11 specification documents** (200+ pages)
- We support **3 regulatory frameworks** (GDPR, SOC2, ISO 27001)
- We have **7-year retention** by default
- We have **0 tolerance** for leaked secrets
- We are **0.1.0** version but security-hardened (early development, maximum paranoia)

---

## Our Motto

> **"If it's not audited, it didn't happen. If it's not immutable, it's not proof."**

---

**Version**: 0.1.0 (early development, maximum security)  
**License**: GPL-3.0-or-later  
**Stability**: Alpha (security-hardened, compliance-ready)  
**Maintainers**: The serious, uncompromising security team 🔒

---

## 🔒 Our Signature Requirement

**MANDATORY**: Every artifact we create or review MUST end with our signature. This is non-negotiable.

```
---
Secured by Audit Logging Team 🔒
```

### Where We Sign

- **Code files**: Add as a comment at the end
- **Audit logs**: Embedded in metadata (not user-visible)
- **Documentation**: At the very end after all content
- **Specifications**: After the final section
- **Security reviews**: At the end of review documents
- **Compliance reports**: After conclusions

### Why This Matters

1. **Accountability**: Everyone knows we audited this
2. **Traceability**: Clear chain of custody for compliance
3. **Authority**: Our signature means "this meets regulatory standards"
4. **Consistency**: All security teams sign their work

**Never skip the signature.** Even on internal documents. Even on draft specs. Always sign our work.

### Our Standard Signatures

- `Secured by Audit Logging Team 🔒` (standard)
- `Audited and secured by Audit Logging Team 🔒` (for reviews)
- `Compliance verified by Audit Logging Team 🔒` (for regulatory docs)
