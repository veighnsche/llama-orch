# audit-logging

**Security audit logging for compliance and forensics**

`bin/shared-crates/audit-log` — Immutable, tamper-evident audit trail for security-critical events across all llama-orch services.

---

## Purpose

Audit logging provides an **immutable record of security events** for compliance (GDPR, SOC2, ISO 27001) and forensic investigation. This is distinct from narration-core, which provides developer-focused observability.

### Audit Logging vs Narration

| Feature | **audit-log** (Security) | **narration-core** (Observability) |
|---------|--------------------------|-------------------------------------|
| **Purpose** | Compliance, forensics | Debugging, performance |
| **Audience** | Auditors, security teams | Developers, SREs |
| **Format** | Machine-readable events | Human-readable narratives |
| **Content** | WHO did WHAT WHEN | WHY things happened, causality |
| **Retention** | Years (regulatory) | Days/weeks |
| **Mutability** | Immutable, append-only | Can rotate/delete |
| **Volume** | Low (critical events only) | High (verbose flows) |

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

## Architecture

### Single-Node Mode (Local Audit)

In single-node deployments, audit logs are stored locally:

```
orchestratord/pool-managerd
         ↓
    audit-log crate
         ↓
    Local storage (.audit/)
    - Append-only file
    - Checksummed entries
    - Encrypted at rest
```

### Platform Mode (Centralized Audit)

In platform/marketplace mode, providers emit audit events to the central platform:

```
Provider's orchestratord
         ↓
    audit-log crate
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

## What Gets Audited

### Critical Security Events

**Authentication & Authorization**:
```rust
AuditEvent::AuthSuccess { actor, method, ip }
AuditEvent::AuthFailure { attempted_user, reason, ip }
AuditEvent::TokenCreated { actor, scope, expires_at }
AuditEvent::TokenRevoked { actor, token_id }
```

**Resource Operations** (admin actions):
```rust
AuditEvent::PoolCreated { actor, pool_id, model_ref, node_id }
AuditEvent::PoolDeleted { actor, pool_id, reason }
AuditEvent::NodeRegistered { actor, node_id, capacity }
AuditEvent::NodeDeregistered { actor, node_id, reason }
```

**Data Access** (GDPR):
```rust
AuditEvent::InferenceExecuted { customer_id, model_ref, tokens, provider_id }
AuditEvent::ModelAccessed { customer_id, model_ref, provider_id }
AuditEvent::DataDeleted { customer_id, reason: "gdpr_erasure" }
```

**Security Incidents**:
```rust
AuditEvent::RateLimitExceeded { ip, endpoint, limit }
AuditEvent::PathTraversalAttempt { actor, attempted_path }
AuditEvent::InvalidTokenUsed { ip, token_prefix }
```

**Compliance Events**:
```rust
AuditEvent::GdprDataAccessRequest { customer_id, requester, scope }
AuditEvent::GdprDataExport { customer_id, data_types }
AuditEvent::GdprRightToErasure { customer_id, completed_at }
```

---

## Usage

### Basic Audit Emission

```rust
use audit_log::{AuditLogger, AuditEvent};

// Initialize logger (single-node mode)
let logger = AuditLogger::local("./audit/")?;

// Emit audit event
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

### Platform Mode (Provider)

```rust
use audit_log::{AuditLogger, PlatformConfig};

// Initialize logger (platform mode)
let logger = AuditLogger::platform(PlatformConfig {
    endpoint: "https://audit.llama-orch-platform.com".to_string(),
    provider_id: "provider-a".to_string(),
    provider_key: load_provider_key()?,
    batch_size: 100,
    flush_interval: Duration::from_secs(10),
})?;

// Emit event (buffered, batched, signed)
logger.emit(AuditEvent::InferenceExecuted {
    timestamp: Utc::now(),
    customer_id: "customer-123".to_string(),
    job_id: "job-456".to_string(),
    model_ref: "llama-3.1-70b".to_string(),
    tokens_processed: 1500,
    provider_id: "provider-a".to_string(),
    result: AuditResult::Success,
}).await?;
```

### Querying Audit Logs

```rust
use audit_log::{AuditQuery, AuditLogger};

let logger = AuditLogger::local("./audit/")?;

// Query by actor
let events = logger.query(AuditQuery {
    actor: Some("admin@example.com".to_string()),
    start_time: Some(Utc::now() - Duration::days(7)),
    end_time: Some(Utc::now()),
    event_types: vec!["pool.delete", "pool.create"],
    limit: 100,
}).await?;

for event in events {
    println!("{:?}", event);
}
```

---

## Event Format

### Standard Fields (All Events)

```rust
pub struct AuditEvent {
    /// Unique audit event ID (never reused)
    pub audit_id: String,  // "audit-2025-1001-164805-abc123"
    
    /// Event timestamp (ISO 8601, UTC)
    pub timestamp: DateTime<Utc>,
    
    /// Event type (e.g., "pool.delete", "auth.success")
    pub event_type: String,
    
    /// WHO performed the action
    pub actor: ActorInfo,
    
    /// WHAT resource was affected
    pub resource: Option<ResourceInfo>,
    
    /// Result (success/failure)
    pub result: AuditResult,
    
    /// Additional metadata (JSON)
    pub metadata: serde_json::Value,
    
    /// Signature (platform mode only)
    pub signature: Option<String>,
}

pub struct ActorInfo {
    pub user_id: String,           // "admin@example.com" or "customer-123"
    pub ip: Option<IpAddr>,        // Source IP address
    pub auth_method: AuthMethod,   // BearerToken, ApiKey, mTLS
    pub session_id: Option<String>, // For correlation
}

pub struct ResourceInfo {
    pub resource_type: String,  // "pool", "node", "job"
    pub resource_id: String,    // "pool-123"
    pub parent_id: Option<String>, // "node-1" (parent of pool)
}

pub enum AuditResult {
    Success,
    Failure { reason: String },
    PartialSuccess { details: String },
}
```

---

## Immutability & Tamper-Evidence

### Local Storage

**Append-only file format**:
```
.audit/
  ├─ 2025-10-01.audit       # Daily log file
  ├─ 2025-10-01.audit.sha256 # Checksum
  └─ manifest.json          # File index
```

**Each entry is checksummed**:
```json
{
  "audit_id": "audit-2025-1001-164805-abc123",
  "event": { ... },
  "prev_hash": "sha256:deadbeef...",  // Hash of previous entry
  "hash": "sha256:cafebabe..."         // Hash of this entry
}
```

**Tamper detection**:
- Each entry includes hash of previous entry (blockchain-style)
- Modifying any entry breaks the hash chain
- Verification detects tampering

### Platform Storage

**Provider signs events**:
```rust
let event_hash = sha256(serialize(&event));
let signature = provider_key.sign(&event_hash);

event.signature = Some(signature);
```

**Platform validates**:
1. Signature matches provider's registered public key
2. Event data hasn't been tampered with
3. Detects anomalies (e.g., provider claims 10M tokens but platform only routed 5M)

---

## Retention & Compliance

### Regulatory Requirements

| Regulation | Retention | Notes |
|------------|-----------|-------|
| **GDPR** | 1-7 years | Must prove data handling |
| **SOC2** | 7 years | Auditor access required |
| **ISO 27001** | 3 years | Security event records |
| **HIPAA** | 6 years | If handling health data |

### Storage Recommendations

**Local deployment**:
- Use encrypted filesystem (LUKS, dm-crypt)
- Regular backups to separate location
- Write-once media for long-term retention

**Platform deployment**:
- S3 with Object Lock (WORM mode)
- Glacier for long-term archival
- Cross-region replication
- Encrypted at rest (KMS)

---

## Access Control

### Who Can Access Audit Logs?

**Single-Node Mode**:
- Administrators only
- Read-only API (no updates/deletes)
- Audit log access is itself audited

**Platform Mode**:
- **Customers**: Query their own events only
- **Regulators**: Query for compliance (with approval)
- **Platform admins**: Full access (audited)
- **Providers**: Cannot access (security risk)

### Query API Permissions

```rust
// Customer queries their own events
GET /v1/audit/events?customer_id=customer-123
Authorization: Bearer customer-token

// Platform admin queries all events
GET /v1/audit/events?start_time=2025-10-01
Authorization: Bearer admin-token
X-Admin-Reason: "SOC2 audit preparation"
```

---

## Platform Mode: Centralized Audit

### Why Providers Must Send Audit to Platform

**Legal liability**:
- Platform is liable for GDPR/SOC2, not providers
- If customer data is mishandled, platform gets sued
- Need immutable proof of compliance

**Customer trust**:
- Customers trust the platform, not individual providers
- Need to prove "data never left EU" (GDPR)
- One audit API, not 50 different providers

**Provider accountability**:
- Detect malicious providers accessing data they shouldn't
- Verify billing accuracy (provider claims vs. actual usage)
- Enforce compliance (provider violates terms = removal)

**Dispute resolution**:
- Customer: "Provider X lost my data!"
- Platform: *checks audit* "Actually, you deleted it at 14:32 UTC"

### Architecture: audit-client

**Provider-side library** for sending events to platform:

```rust
use audit_client::PlatformAuditClient;

let client = PlatformAuditClient::new(
    "https://audit.llama-orch-platform.com",
    provider_credentials,
);

// Events buffered, batched, signed, sent
client.emit(event).await?;
```

**Features**:
- Batching (every 10s or 100 events)
- Retry on network failure
- Provider signature (tamper-evident)
- Buffer if platform unreachable
- Rate limiting

---

## Security Considerations

### Event Signing (Platform Mode)

**Prevents provider forgery**:
```rust
// Provider signs each event
let event_hash = sha256(&event);
let signature = provider_key.sign(&event_hash);

// Platform validates
let valid = provider_pubkey.verify(&event_hash, &signature);
if !valid {
    reject_event("Invalid signature");
}
```

### Anomaly Detection

**Platform validates event plausibility**:
```rust
// Provider claims 10M tokens served
audit_event.tokens = 10_000_000;

// But platform only routed 5M tokens to that provider
if audit_event.tokens > platform_routing_stats * 1.1 {
    flag_anomaly("Provider over-reporting usage");
}
```

### Sensitive Data Redaction

**Never log sensitive data**:
```rust
// ❌ BAD: Logs prompt content
AuditEvent { prompt: "My credit card is 1234..." }

// ✅ GOOD: Only log metadata
AuditEvent { 
    prompt_length: 50,
    prompt_hash: "sha256:abc...",  // For correlation
}
```

---

## Dependencies

### Internal

- `contracts/api-types` — Audit event types
- `libs/auth-min` — Actor authentication info
- `libs/narration-core` — (separate, for observability)

### External

- `serde` — Serialization
- `tokio` — Async I/O
- `chrono` — Timestamps
- `sha2` — Checksumming
- `ed25519-dalek` — Signatures (platform mode)
- `reqwest` — HTTP client (platform mode)

---

## Specifications

Implements requirements from:
- AUDIT-1001: Immutable audit trail
- AUDIT-1002: Tamper-evident storage
- AUDIT-1003: WHO/WHAT/WHEN logging
- AUDIT-1004: Platform centralization
- AUDIT-1005: GDPR compliance
- AUDIT-1006: 7-year retention

See `.specs/` for full requirements.

---

## Testing

### Unit Tests

```bash
# Test local audit logging
cargo test -p audit-log -- local_emit

# Test platform audit client
cargo test -p audit-log -- platform_batch

# Test tamper detection
cargo test -p audit-log -- tamper_detection
```

### Integration Tests

```bash
# Test full platform audit flow
cargo test -p audit-log -- e2e_platform_audit
```

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Priority**: P0 for platform mode, P1 for single-node

---

## Future Enhancements

- [ ] Export to SIEM (Splunk, ELK)
- [ ] Real-time anomaly detection
- [ ] Automated compliance reports (SOC2, GDPR)
- [ ] Audit event analytics dashboard
- [ ] Multi-region audit replication
- [ ] Blockchain-backed tamper-evidence

---

**For questions**: See `.docs/SECURITY_AUDIT_EXISTING_CODEBASE.md`
