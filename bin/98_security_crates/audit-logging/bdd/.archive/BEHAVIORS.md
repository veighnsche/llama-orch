# Audit Logging BDD Behaviors

**Comprehensive behavior documentation for audit-logging validation**

This document catalogs all BDD scenarios testing the audit-logging crate's security validation.

---

## Overview

**Purpose**: Ensure audit events are validated against log injection attacks and data integrity violations.

**Security Focus**: TIER 1 (security-critical) — Audit logs are high-value targets for attackers.

**Attack Vectors Covered**:
- ANSI escape sequence injection
- Control character injection
- Null byte injection
- Unicode directional override attacks
- Path traversal attempts
- Log line injection
- Field length violations

---

## Authentication Events

### AuthSuccess Event Validation

**Event Type**: `AuditEvent::AuthSuccess`

**Fields Validated**:
- `actor.user_id` — User identifier
- `path` — Request path
- `service_id` — Service identifier

#### Scenarios

1. **Accept valid AuthSuccess event**
   - **Given**: Valid user ID and path
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject AuthSuccess with ANSI escape in user ID**
   - **Given**: User ID contains `\x1b[31mFAKE ERROR\x1b[0m`
   - **When**: Create and validate event
   - **Then**: Validation rejects ANSI escape sequences
   - **Security**: Prevents terminal manipulation attacks

3. **Reject AuthSuccess with control characters in path**
   - **Given**: Path contains `\r\n[CRITICAL] Fake log`
   - **When**: Create and validate event
   - **Then**: Validation rejects control characters
   - **Security**: Prevents log line injection

---

### AuthFailure Event Validation

**Event Type**: `AuditEvent::AuthFailure`

**Fields Validated**:
- `attempted_user` — User who attempted authentication
- `reason` — Failure reason
- `path` — Request path

#### Scenarios

1. **Accept valid AuthFailure event**
   - **Given**: Valid user ID, IP, path, and reason
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject AuthFailure with null byte in user ID**
   - **Given**: User ID contains `admin\0malicious`
   - **When**: Create and validate event
   - **Then**: Validation rejects null bytes
   - **Security**: Prevents C string truncation attacks

3. **Reject AuthFailure with log injection in reason**
   - **Given**: Reason contains `failed\n[ERROR] System compromised`
   - **When**: Create and validate event
   - **Then**: Validation rejects log injection
   - **Security**: Prevents fake log entries

---

## Resource Operation Events

### PoolCreated Event Validation

**Event Type**: `AuditEvent::PoolCreated`

**Fields Validated**:
- `actor.user_id` — User who created pool
- `pool_id` — Pool identifier
- `model_ref` — Model reference
- `node_id` — Node identifier

#### Scenarios

1. **Accept valid PoolCreated event**
   - **Given**: Valid user, pool ID, model ref, and node ID
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject PoolCreated with path traversal in pool ID**
   - **Given**: Pool ID contains `pool-../../../etc/passwd`
   - **When**: Create and validate event
   - **Then**: Validation fails
   - **Security**: Prevents directory traversal attacks

3. **Reject PoolCreated with ANSI escape in model reference**
   - **Given**: Model ref contains `\x1b[31mmalicious-model\x1b[0m`
   - **When**: Create and validate event
   - **Then**: Validation rejects ANSI escape sequences
   - **Security**: Prevents terminal manipulation

---

### PoolDeleted Event Validation

**Event Type**: `AuditEvent::PoolDeleted`

**Fields Validated**:
- `actor.user_id` — User who deleted pool
- `pool_id` — Pool identifier
- `model_ref` — Model reference
- `node_id` — Node identifier
- `reason` — Deletion reason

#### Scenarios

1. **Accept valid PoolDeleted event**
   - **Given**: Valid user, pool ID, model ref, node ID, and reason
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject PoolDeleted with control characters in reason**
   - **Given**: Reason contains `deleted\r\n[CRITICAL] Unauthorized access`
   - **When**: Create and validate event
   - **Then**: Validation rejects control characters
   - **Security**: Prevents log line injection

---

### TaskSubmitted Event Validation

**Event Type**: `AuditEvent::TaskSubmitted`

**Fields Validated**:
- `actor.user_id` — User who submitted task
- `task_id` — Task identifier
- `model_ref` — Model reference

#### Scenarios

1. **Accept valid TaskSubmitted event**
   - **Given**: Valid user, task ID, and model ref
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject TaskSubmitted with null byte in task ID**
   - **Given**: Task ID contains `task-123\0malicious`
   - **When**: Create and validate event
   - **Then**: Validation rejects null bytes
   - **Security**: Prevents C string truncation

---

## VRAM Operation Events

### VramSealed Event Validation

**Event Type**: `AuditEvent::VramSealed`

**Fields Validated**:
- `shard_id` — Shard identifier
- `worker_id` — Worker identifier

**Security Criticality**: HIGHEST — VRAM sealing is a core security primitive

#### Scenarios

1. **Accept valid VramSealed event**
   - **Given**: Valid shard ID and worker ID
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject VramSealed with ANSI escape in shard ID**
   - **Given**: Shard ID contains `\x1b[31mshard-fake\x1b[0m`
   - **When**: Create and validate event
   - **Then**: Validation rejects ANSI escape sequences
   - **Security**: Prevents terminal manipulation in security logs

3. **Reject VramSealed with null byte in worker ID**
   - **Given**: Worker ID contains `worker-gpu-0\0malicious`
   - **When**: Create and validate event
   - **Then**: Validation rejects null bytes
   - **Security**: Prevents C string truncation

4. **Reject VramSealed with control characters in shard ID**
   - **Given**: Shard ID contains `shard-123\r\nFAKE LOG`
   - **When**: Create and validate event
   - **Then**: Validation rejects control characters
   - **Security**: Prevents log line injection

---

## Security Incident Events

### PathTraversalAttempt Event Validation

**Event Type**: `AuditEvent::PathTraversalAttempt`

**Fields Validated**:
- `actor.user_id` — User who attempted traversal
- `attempted_path` — Path that was attempted
- `endpoint` — API endpoint

**Security Note**: This event records an attack attempt, so the `attempted_path` may contain malicious content. However, other fields must still be sanitized.

#### Scenarios

1. **Accept valid PathTraversalAttempt event**
   - **Given**: Valid user, path, and endpoint
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject PathTraversalAttempt with ANSI escape in endpoint**
   - **Given**: Endpoint contains `/v2/files\x1b[31mFAKE\x1b[0m`
   - **When**: Create and validate event
   - **Then**: Validation rejects ANSI escape sequences
   - **Security**: Prevents terminal manipulation even in security logs

---

### PolicyViolation Event Validation

**Event Type**: `AuditEvent::PolicyViolation`

**Fields Validated**:
- `worker_id` — Worker that violated policy
- `details` — Violation details
- `policy` — Policy name
- `violation` — Violation type
- `action_taken` — Action taken

#### Scenarios

1. **Accept valid PolicyViolation event**
   - **Given**: Valid worker ID and details
   - **When**: Create and validate event
   - **Then**: Validation succeeds
   - **Security**: Baseline positive test

2. **Reject PolicyViolation with control characters in details**
   - **Given**: Details contains `violation\r\n[ERROR] System compromised`
   - **When**: Create and validate event
   - **Then**: Validation rejects control characters
   - **Security**: Prevents log line injection

3. **Reject PolicyViolation with null byte in worker ID**
   - **Given**: Worker ID contains `worker-gpu-0\0malicious`
   - **When**: Create and validate event
   - **Then**: Validation rejects null bytes
   - **Security**: Prevents C string truncation

---

## Event Serialization

### JSON Serialization Tests

**Purpose**: Ensure validated events can be serialized to JSON for storage.

#### Scenarios

1. **Serialize valid AuthSuccess event**
   - **Given**: Valid AuthSuccess event
   - **When**: Validate and serialize to JSON
   - **Then**: Event is serializable
   - **Security**: Ensures sanitized data doesn't break serialization

2. **Serialize valid PoolCreated event**
   - **Given**: Valid PoolCreated event
   - **When**: Validate and serialize to JSON
   - **Then**: Event is serializable
   - **Security**: Ensures sanitized data doesn't break serialization

3. **Serialize valid VramSealed event**
   - **Given**: Valid VramSealed event
   - **When**: Validate and serialize to JSON
   - **Then**: Event is serializable
   - **Security**: Ensures sanitized data doesn't break serialization

4. **Sanitized data should be serializable**
   - **Given**: Valid event with sanitized data
   - **When**: Validate and serialize to JSON
   - **Then**: Event contains sanitized data and is serializable
   - **Security**: Meta-test ensuring validation pipeline works end-to-end

---

## Attack Vector Coverage

### ANSI Escape Sequence Injection

**Attack**: Inject terminal control sequences to manipulate log display.

**Example**: `\x1b[31mFAKE ERROR\x1b[0m`

**Impact**:
- Fake error messages in logs
- Terminal manipulation when viewing logs
- Log parsing tools confused

**Scenarios Testing This**:
- AuthSuccess with ANSI escape in user ID
- PoolCreated with ANSI escape in model reference
- VramSealed with ANSI escape in shard ID
- PathTraversalAttempt with ANSI escape in endpoint

---

### Control Character Injection

**Attack**: Inject control characters to create fake log lines.

**Example**: `deleted\r\n[CRITICAL] Unauthorized access`

**Impact**:
- Fake log lines injected
- Log analysis tools deceived
- False security alerts

**Scenarios Testing This**:
- AuthSuccess with control characters in path
- PoolDeleted with control characters in reason
- VramSealed with control characters in shard ID
- PolicyViolation with control characters in details

---

### Null Byte Injection

**Attack**: Inject null bytes to truncate strings in C-based tools.

**Example**: `admin\0malicious`

**Impact**:
- Truncates logs in C-based tools
- Bypasses length checks
- Hides malicious content

**Scenarios Testing This**:
- AuthFailure with null byte in user ID
- TaskSubmitted with null byte in task ID
- VramSealed with null byte in worker ID
- PolicyViolation with null byte in worker ID

---

### Path Traversal

**Attack**: Use directory traversal sequences to escape directories.

**Example**: `pool-../../../etc/passwd`

**Impact**:
- Audit logs written to arbitrary directories
- Overwrite system files
- Privilege escalation

**Scenarios Testing This**:
- PoolCreated with path traversal in pool ID

---

### Log Line Injection

**Attack**: Inject newlines to create fake log entries.

**Example**: `failed\n[ERROR] System compromised`

**Impact**:
- Fake log entries created
- Log analysis deceived
- False security alerts

**Scenarios Testing This**:
- AuthFailure with log injection in reason
- PoolDeleted with control characters in reason
- PolicyViolation with control characters in details

---

## Security Requirements Verified

### From .specs/20_security.md

| Requirement | Attack Vector | BDD Coverage |
|-------------|---------------|--------------|
| **SEC-AUDIT-001** | ANSI escape injection | ✅ 4 scenarios |
| **SEC-AUDIT-002** | Control character injection | ✅ 4 scenarios |
| **SEC-AUDIT-003** | Null byte injection | ✅ 4 scenarios |
| **SEC-AUDIT-004** | Unicode directional overrides | ⬜ Not yet implemented |
| **SEC-AUDIT-005** | Path traversal | ✅ 1 scenario |
| **SEC-AUDIT-006** | Log line injection | ✅ 3 scenarios |
| **SEC-AUDIT-007** | Field length limits | ⬜ Not yet implemented |

**Total Scenarios**: 25+ scenarios covering critical attack vectors

---

## Test Execution

### Running All Tests

```bash
cargo test -p audit-logging-bdd -- --nocapture
```

### Running Specific Feature

```bash
LLORCH_BDD_FEATURE_PATH=tests/features/authentication_events.feature \
cargo test -p audit-logging-bdd -- --nocapture
```

### Expected Output

```
Feature: Authentication Event Validation
  ✓ Accept valid AuthSuccess event
  ✓ Reject AuthSuccess with ANSI escape in user ID
  ✓ Reject AuthSuccess with control characters in path
  ✓ Reject AuthFailure with null byte in user ID
  ✓ Accept valid AuthFailure event
  ✓ Reject AuthFailure with log injection in reason

6 scenarios (6 passed)
18 steps (18 passed)
```

---

## Future Enhancements

### Additional Attack Vectors to Test

1. **Unicode directional overrides** (`\u{202E}`, `\u{202D}`)
2. **Field length violations** (oversized fields)
3. **Format string injection** (though low risk in Rust)
4. **Integer overflow** in numeric fields
5. **Timestamp manipulation** (past/future dates)

### Additional Event Types to Test

Currently testing 5 event types. Remaining 27 event types:
- TokenCreated, TokenRevoked
- AuthorizationGranted, AuthorizationDenied, PermissionChanged
- PoolModified, NodeRegistered, NodeDeregistered, TaskCompleted, TaskCanceled
- SealVerified, SealVerificationFailed, VramAllocated, VramAllocationFailed, VramDeallocated
- RateLimitExceeded, InvalidTokenUsed, SuspiciousActivity
- InferenceExecuted, ModelAccessed, DataDeleted
- GdprDataAccessRequest, GdprDataExport, GdprRightToErasure

---

## References

- **Parent Crate**: `bin/shared-crates/audit-logging`
- **Specifications**: `.specs/README.md`, `.specs/20_security.md`
- **Security Audit**: `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` (Vulnerability #18)
- **Input Validation**: `bin/shared-crates/input-validation`

---

**Status**: Alpha — Security-critical validation with maximum robustness

**Maintainers**: @llama-orch-maintainers
