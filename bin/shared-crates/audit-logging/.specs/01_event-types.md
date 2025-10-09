# Audit Logging — Event Types Specification

**Crate**: `bin/shared-crates/audit-logging`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document specifies all audit event types that `audit-logging` must support across llama-orch services. Events are categorized by security domain and include detailed field specifications.

---

## 1. Event Type Taxonomy

### 1.1 Event Categories

| Category | Event Count | Priority | Consumers |
|----------|-------------|----------|-----------|
| **Authentication** | 4 | P0 | queen-rbee, pool-managerd, worker-orcd |
| **Authorization** | 3 | P0 | queen-rbee, pool-managerd |
| **Resource Operations** | 8 | P1 | queen-rbee, pool-managerd |
| **VRAM Operations** | 6 | P0 | worker-orcd, vram-residency |
| **Data Access** | 3 | P1 | queen-rbee (GDPR) |
| **Security Incidents** | 5 | P0 | All services |
| **Compliance** | 3 | P2 | Platform mode |

---

## 2. Authentication Events

### 2.1 AuthSuccess

**Purpose**: Record successful authentication attempts.

**Event Type**: `auth.success`

**Fields**:
```rust
pub struct AuthSuccess {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub method: AuthMethod,  // BearerToken, ApiKey, mTLS
    pub path: String,        // Endpoint accessed
    pub service_id: String,  // "queen-rbee", "pool-managerd"
}
```

**Example**:
```json
{
  "audit_id": "audit-2025-1001-164805-abc123",
  "timestamp": "2025-10-01T16:48:05Z",
  "event_type": "auth.success",
  "service_id": "queen-rbee",
  "event": {
    "actor": {
      "user_id": "admin@llorch.io",
      "ip": "192.168.1.100",
      "auth_method": "bearer_token",
      "session_id": "sess-abc123"
    },
    "method": "bearer_token",
    "path": "/v2/tasks"
  }
}
```

---

### 2.2 AuthFailure

**Purpose**: Record failed authentication attempts (security monitoring).

**Event Type**: `auth.failure`

**Fields**:
```rust
pub struct AuthFailure {
    pub timestamp: DateTime<Utc>,
    pub attempted_user: Option<String>,  // If provided
    pub reason: String,                  // "invalid_token", "expired_token", "missing_header"
    pub ip: IpAddr,
    pub path: String,
    pub service_id: String,
}
```

**Example**:
```json
{
  "audit_id": "audit-2025-1001-164810-def456",
  "timestamp": "2025-10-01T16:48:10Z",
  "event_type": "auth.failure",
  "service_id": "queen-rbee",
  "event": {
    "attempted_user": null,
    "reason": "invalid_token",
    "ip": "203.0.113.42",
    "path": "/v2/tasks"
  }
}
```

**Security Note**: Log IP for rate limiting and intrusion detection.

---

### 2.3 TokenCreated

**Purpose**: Record API token creation (admin action).

**Event Type**: `token.created`

**Fields**:
```rust
pub struct TokenCreated {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub token_fingerprint: String,  // First 6 hex chars of SHA-256
    pub scope: Vec<String>,         // ["read:pools", "write:tasks"]
    pub expires_at: Option<DateTime<Utc>>,
}
```

**Example**:
```json
{
  "audit_id": "audit-2025-1001-164815-ghi789",
  "timestamp": "2025-10-01T16:48:15Z",
  "event_type": "token.created",
  "service_id": "queen-rbee",
  "event": {
    "actor": {
      "user_id": "admin@llorch.io",
      "ip": "192.168.1.100",
      "auth_method": "bearer_token"
    },
    "token_fingerprint": "a3f2c1",
    "scope": ["read:pools", "write:tasks"],
    "expires_at": "2025-11-01T16:48:15Z"
  }
}
```

**Security Note**: Never log full token, only fingerprint.

---

### 2.4 TokenRevoked

**Purpose**: Record API token revocation (security event).

**Event Type**: `token.revoked`

**Fields**:
```rust
pub struct TokenRevoked {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub token_fingerprint: String,
    pub reason: String,  // "compromised", "user_requested", "expired"
}
```

---

## 3. Authorization Events

### 3.1 AuthorizationGranted

**Purpose**: Record successful authorization checks.

**Event Type**: `authz.granted`

**Fields**:
```rust
pub struct AuthorizationGranted {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub resource: ResourceInfo,
    pub action: String,  // "read", "write", "delete"
}
```

---

### 3.2 AuthorizationDenied

**Purpose**: Record failed authorization checks (security monitoring).

**Event Type**: `authz.denied`

**Fields**:
```rust
pub struct AuthorizationDenied {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub resource: ResourceInfo,
    pub action: String,
    pub reason: String,  // "insufficient_permissions", "resource_not_found"
}
```

**Security Note**: High frequency of denials may indicate attack.

---

### 3.3 PermissionChanged

**Purpose**: Record permission changes (admin action).

**Event Type**: `authz.permission_changed`

**Fields**:
```rust
pub struct PermissionChanged {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub subject: String,      // User/token whose permissions changed
    pub old_permissions: Vec<String>,
    pub new_permissions: Vec<String>,
}
```

---

## 4. Resource Operation Events

### 4.1 PoolCreated

**Purpose**: Record pool creation (admin action).

**Event Type**: `pool.created`

**Fields**:
```rust
pub struct PoolCreated {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub pool_id: String,
    pub model_ref: String,
    pub node_id: String,
    pub replicas: u32,
    pub gpu_devices: Vec<u32>,
}
```

---

### 4.2 PoolDeleted

**Purpose**: Record pool deletion (admin action).

**Event Type**: `pool.deleted`

**Fields**:
```rust
pub struct PoolDeleted {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub pool_id: String,
    pub model_ref: String,
    pub node_id: String,
    pub reason: String,  // "user_requested", "node_failure", "policy_violation"
    pub replicas_terminated: u32,
}
```

---

### 4.3 PoolModified

**Purpose**: Record pool configuration changes.

**Event Type**: `pool.modified`

**Fields**:
```rust
pub struct PoolModified {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub pool_id: String,
    pub changes: serde_json::Value,  // {"replicas": {"old": 2, "new": 4}}
}
```

---

### 4.4 NodeRegistered

**Purpose**: Record GPU node registration.

**Event Type**: `node.registered`

**Fields**:
```rust
pub struct NodeRegistered {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub node_id: String,
    pub gpu_count: u32,
    pub total_vram_gb: u64,
    pub capabilities: Vec<String>,  // ["cuda", "tensor_cores"]
}
```

---

### 4.5 NodeDeregistered

**Purpose**: Record GPU node removal.

**Event Type**: `node.deregistered`

**Fields**:
```rust
pub struct NodeDeregistered {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub node_id: String,
    pub reason: String,  // "maintenance", "failure", "decommissioned"
    pub pools_affected: Vec<String>,
}
```

---

### 4.6 TaskSubmitted

**Purpose**: Record inference task submission.

**Event Type**: `task.submitted`

**Fields**:
```rust
pub struct TaskSubmitted {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub task_id: String,
    pub model_ref: String,
    pub prompt_length: usize,  // Characters (not content)
    pub prompt_hash: String,   // SHA-256 for correlation
    pub max_tokens: u32,
}
```

**Security Note**: Never log prompt content (PII risk).

---

### 4.7 TaskCompleted

**Purpose**: Record inference task completion.

**Event Type**: `task.completed`

**Fields**:
```rust
pub struct TaskCompleted {
    pub timestamp: DateTime<Utc>,
    pub task_id: String,
    pub worker_id: String,
    pub tokens_generated: u32,
    pub duration_ms: u64,
    pub result: AuditResult,  // Success, Failure
}
```

---

### 4.8 TaskCanceled

**Purpose**: Record inference task cancellation.

**Event Type**: `task.canceled`

**Fields**:
```rust
pub struct TaskCanceled {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub task_id: String,
    pub reason: String,  // "user_requested", "timeout", "worker_failure"
}
```

---

## 5. VRAM Operation Events

### 5.1 VramSealed

**Purpose**: Record model sealing in VRAM (security-critical).

**Event Type**: `vram.sealed`

**Fields**:
```rust
pub struct VramSealed {
    pub timestamp: DateTime<Utc>,
    pub shard_id: String,
    pub gpu_device: u32,
    pub vram_bytes: usize,
    pub digest: String,      // SHA-256 of model bytes
    pub worker_id: String,
}
```

**Example**:
```json
{
  "audit_id": "audit-2025-1001-200532-abc123",
  "timestamp": "2025-10-01T20:05:32Z",
  "event_type": "vram.sealed",
  "service_id": "worker-gpu-0",
  "event": {
    "shard_id": "shard-abc123",
    "gpu_device": 0,
    "vram_bytes": 8000000000,
    "digest": "abc123def456789012345678901234567890123456789012345678901234",
    "worker_id": "worker-gpu-0"
  }
}
```

**Security Note**: This event proves which models were loaded when (forensics).

---

### 5.2 SealVerified

**Purpose**: Record successful seal verification.

**Event Type**: `seal.verified`

**Fields**:
```rust
pub struct SealVerified {
    pub timestamp: DateTime<Utc>,
    pub shard_id: String,
    pub worker_id: String,
}
```

---

### 5.3 SealVerificationFailed

**Purpose**: Record seal verification failure (SECURITY INCIDENT).

**Event Type**: `seal.verification_failed`

**Fields**:
```rust
pub struct SealVerificationFailed {
    pub timestamp: DateTime<Utc>,
    pub shard_id: String,
    pub reason: String,           // "digest_mismatch", "seal_expired"
    pub expected_digest: String,
    pub actual_digest: String,
    pub worker_id: String,
    pub severity: String,         // "critical"
}
```

**Security Note**: This indicates VRAM corruption or tampering. Immediate investigation required.

---

### 5.4 VramAllocated

**Purpose**: Track VRAM resource usage (capacity planning, DoS detection).

**Event Type**: `vram.allocated`

**Fields**:
```rust
pub struct VramAllocated {
    pub timestamp: DateTime<Utc>,
    pub requested_bytes: usize,
    pub allocated_bytes: usize,
    pub available_bytes: usize,
    pub used_bytes: usize,
    pub gpu_device: u32,
    pub worker_id: String,
}
```

---

### 5.5 VramAllocationFailed

**Purpose**: Record VRAM allocation failure (OOM detection).

**Event Type**: `vram.allocation_failed`

**Fields**:
```rust
pub struct VramAllocationFailed {
    pub timestamp: DateTime<Utc>,
    pub requested_bytes: usize,
    pub available_bytes: usize,
    pub reason: String,  // "insufficient_vram", "fragmentation"
    pub gpu_device: u32,
    pub worker_id: String,
}
```

**Security Note**: Repeated failures may indicate DoS attack.

---

### 5.6 VramDeallocated

**Purpose**: Track VRAM cleanup (leak detection).

**Event Type**: `vram.deallocated`

**Fields**:
```rust
pub struct VramDeallocated {
    pub timestamp: DateTime<Utc>,
    pub shard_id: String,
    pub freed_bytes: usize,
    pub remaining_used: usize,
    pub gpu_device: u32,
    pub worker_id: String,
}
```

---

## 6. Security Incident Events

### 6.1 RateLimitExceeded

**Purpose**: Record rate limit violations (DoS detection).

**Event Type**: `security.rate_limit_exceeded`

**Fields**:
```rust
pub struct RateLimitExceeded {
    pub timestamp: DateTime<Utc>,
    pub ip: IpAddr,
    pub endpoint: String,
    pub limit: u32,           // Requests per minute
    pub actual: u32,          // Actual request count
    pub window_seconds: u32,
}
```

---

### 6.2 PathTraversalAttempt

**Purpose**: Record path traversal attempts (security incident).

**Event Type**: `security.path_traversal_attempt`

**Fields**:
```rust
pub struct PathTraversalAttempt {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub attempted_path: String,
    pub endpoint: String,
}
```

---

### 6.3 InvalidTokenUsed

**Purpose**: Record invalid token usage (security monitoring).

**Event Type**: `security.invalid_token_used`

**Fields**:
```rust
pub struct InvalidTokenUsed {
    pub timestamp: DateTime<Utc>,
    pub ip: IpAddr,
    pub token_prefix: String,  // First 6 chars only
    pub endpoint: String,
}
```

---

### 6.4 PolicyViolation

**Purpose**: Record security policy violations.

**Event Type**: `security.policy_violation`

**Fields**:
```rust
pub struct PolicyViolation {
    pub timestamp: DateTime<Utc>,
    pub policy: String,        // "vram_only", "gpu_required"
    pub violation: String,     // "unified_memory_detected"
    pub details: String,
    pub severity: String,      // "critical", "high", "medium"
    pub worker_id: String,
    pub action_taken: String,  // "worker_stopped", "request_rejected"
}
```

---

### 6.5 SuspiciousActivity

**Purpose**: Record anomalous behavior patterns.

**Event Type**: `security.suspicious_activity`

**Fields**:
```rust
pub struct SuspiciousActivity {
    pub timestamp: DateTime<Utc>,
    pub actor: ActorInfo,
    pub activity_type: String,  // "repeated_failures", "unusual_pattern"
    pub details: String,
    pub risk_score: u32,        // 0-100
}
```

---

## 7. Data Access Events (GDPR)

### 7.1 InferenceExecuted

**Purpose**: Record inference execution (GDPR data processing).

**Event Type**: `data.inference_executed`

**Fields**:
```rust
pub struct InferenceExecuted {
    pub timestamp: DateTime<Utc>,
    pub customer_id: String,
    pub job_id: String,
    pub model_ref: String,
    pub tokens_processed: u32,
    pub provider_id: Option<String>,  // Platform mode
    pub result: AuditResult,
}
```

---

### 7.2 ModelAccessed

**Purpose**: Record model access (GDPR data processing).

**Event Type**: `data.model_accessed`

**Fields**:
```rust
pub struct ModelAccessed {
    pub timestamp: DateTime<Utc>,
    pub customer_id: String,
    pub model_ref: String,
    pub access_type: String,  // "inference", "download", "inspect"
    pub provider_id: Option<String>,
}
```

---

### 7.3 DataDeleted

**Purpose**: Record data deletion (GDPR right to erasure).

**Event Type**: `data.deleted`

**Fields**:
```rust
pub struct DataDeleted {
    pub timestamp: DateTime<Utc>,
    pub customer_id: String,
    pub data_types: Vec<String>,  // ["inference_logs", "model_cache"]
    pub reason: String,            // "gdpr_erasure", "retention_policy"
}
```

---

## 8. Compliance Events

### 8.1 GdprDataAccessRequest

**Purpose**: Record GDPR data access requests.

**Event Type**: `compliance.gdpr_data_access_request`

**Fields**:
```rust
pub struct GdprDataAccessRequest {
    pub timestamp: DateTime<Utc>,
    pub customer_id: String,
    pub requester: String,
    pub scope: Vec<String>,  // ["inference_logs", "audit_logs"]
}
```

---

### 8.2 GdprDataExport

**Purpose**: Record GDPR data export.

**Event Type**: `compliance.gdpr_data_export`

**Fields**:
```rust
pub struct GdprDataExport {
    pub timestamp: DateTime<Utc>,
    pub customer_id: String,
    pub data_types: Vec<String>,
    pub export_format: String,  // "json", "csv"
    pub file_hash: String,
}
```

---

### 8.3 GdprRightToErasure

**Purpose**: Record GDPR right to erasure completion.

**Event Type**: `compliance.gdpr_right_to_erasure`

**Fields**:
```rust
pub struct GdprRightToErasure {
    pub timestamp: DateTime<Utc>,
    pub customer_id: String,
    pub completed_at: DateTime<Utc>,
    pub data_types_deleted: Vec<String>,
}
```

---

## 9. Event Type Registry

### 9.1 Complete Event Type List

| Event Type | Category | Priority | Consumer |
|------------|----------|----------|----------|
| `auth.success` | Authentication | P0 | All services |
| `auth.failure` | Authentication | P0 | All services |
| `token.created` | Authentication | P1 | queen-rbee |
| `token.revoked` | Authentication | P1 | queen-rbee |
| `authz.granted` | Authorization | P1 | All services |
| `authz.denied` | Authorization | P0 | All services |
| `authz.permission_changed` | Authorization | P1 | queen-rbee |
| `pool.created` | Resource Ops | P1 | pool-managerd |
| `pool.deleted` | Resource Ops | P1 | pool-managerd |
| `pool.modified` | Resource Ops | P2 | pool-managerd |
| `node.registered` | Resource Ops | P1 | queen-rbee |
| `node.deregistered` | Resource Ops | P1 | queen-rbee |
| `task.submitted` | Resource Ops | P1 | queen-rbee |
| `task.completed` | Resource Ops | P1 | queen-rbee |
| `task.canceled` | Resource Ops | P2 | queen-rbee |
| `vram.sealed` | VRAM Ops | P0 | worker-orcd |
| `seal.verified` | VRAM Ops | P0 | worker-orcd |
| `seal.verification_failed` | VRAM Ops | P0 | worker-orcd |
| `vram.allocated` | VRAM Ops | P1 | worker-orcd |
| `vram.allocation_failed` | VRAM Ops | P1 | worker-orcd |
| `vram.deallocated` | VRAM Ops | P2 | worker-orcd |
| `security.rate_limit_exceeded` | Security | P0 | All services |
| `security.path_traversal_attempt` | Security | P0 | All services |
| `security.invalid_token_used` | Security | P0 | All services |
| `security.policy_violation` | Security | P0 | worker-orcd |
| `security.suspicious_activity` | Security | P1 | All services |
| `data.inference_executed` | GDPR | P1 | queen-rbee |
| `data.model_accessed` | GDPR | P1 | queen-rbee |
| `data.deleted` | GDPR | P1 | queen-rbee |
| `compliance.gdpr_data_access_request` | Compliance | P2 | Platform |
| `compliance.gdpr_data_export` | Compliance | P2 | Platform |
| `compliance.gdpr_right_to_erasure` | Compliance | P2 | Platform |

---

## 10. Implementation Guidelines

### 10.1 Event Naming Convention

**Pattern**: `{category}.{action}`

**Examples**:
- `auth.success` (not `authentication_success`)
- `vram.sealed` (not `vram_seal_event`)
- `security.rate_limit_exceeded` (not `rate_limit`)

### 10.2 Field Naming Convention

**Use snake_case** for all field names:
- `user_id` (not `userId`)
- `gpu_device` (not `gpuDevice`)
- `expected_digest` (not `expectedDigest`)

### 10.3 Timestamp Format

**Always use ISO 8601 UTC**:
```rust
use chrono::{DateTime, Utc};

let timestamp = Utc::now();
// Serializes to: "2025-10-01T16:48:05Z"
```

### 10.4 Sensitive Data Handling

**Never log**:
- Full API tokens (use fingerprints)
- Raw passwords
- VRAM pointers
- Prompt content (use length/hash)
- Customer PII (unless required for GDPR)

**Always log**:
- Timestamps (UTC)
- Actor identity (user_id or token fingerprint)
- Resource identifiers (pool_id, shard_id)
- Action outcomes (success/failure)

---

## 11. Refinement Opportunities

### 11.1 Immediate Improvements

1. **Add event versioning** to support schema evolution
2. **Define event severity levels** (info, warning, critical)
3. **Add correlation fields** for multi-event flows
4. **Create event validation** to ensure required fields present

### 11.2 Medium-Term Enhancements

5. **Add event aggregation** for high-frequency events (e.g., token generation per minute)
6. **Define event retention policies** per category
7. **Create event templates** for common patterns
8. **Add event enrichment** (e.g., geo-location from IP)

### 11.3 Long-Term Vision

9. **Machine-readable event schemas** (JSON Schema, Protobuf)
10. **Event streaming** to real-time analytics
11. **Event replay** for forensic investigation
12. **Event anonymization** for privacy-preserving analytics

---

**End of Event Types Specification**
