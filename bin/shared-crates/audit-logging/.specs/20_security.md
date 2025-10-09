# Audit Logging — Security Attack Surface Analysis

**Crate**: `bin/shared-crates/audit-logging`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Executive Summary

This document provides a **comprehensive security attack surface analysis** for the `audit-logging` crate. As a security-critical component responsible for immutable audit trails, this crate must be hardened against all attack vectors.

**Threat Model**: Audit logging is a **high-value target** for attackers because:
- Destroying audit logs covers tracks of intrusions
- Tampering with logs creates false evidence
- Injecting fake events frames innocent parties
- Exfiltrating logs reveals system behavior and credentials

**Security Posture**: TIER 1 (security-critical)

---

## 1. Attack Surface Overview

### 1.1 Attack Surface Categories

| Category | Attack Vectors | Severity | Status |
|----------|----------------|----------|--------|
| **Input Validation** | Log injection, path traversal, format string | CRITICAL | ⬜ Needs hardening |
| **Storage Security** | Tampering, deletion, unauthorized access | CRITICAL | ⬜ Needs implementation |
| **Memory Safety** | Buffer overflows, integer overflows, panics | HIGH | ⬜ Needs Clippy enforcement |
| **Cryptographic** | Weak hashing, signature forgery, timing attacks | HIGH | ⬜ Needs implementation |
| **Denial of Service** | Unbounded memory, disk exhaustion, CPU exhaustion | HIGH | ⬜ Needs limits |
| **Access Control** | Unauthorized reads, privilege escalation | MEDIUM | ⬜ Needs implementation |
| **Side Channels** | Timing leaks, error message leaks | MEDIUM | ⬜ Needs review |
| **Dependencies** | Supply chain, transitive deps | LOW | ✅ Minimal deps |

**Total Attack Vectors Identified**: 23

---

## 2. Input Validation Attack Surface

### 2.1 Log Injection Attacks

**Attack Vector**: Attacker injects malicious content into audit events.

**Threat Scenarios**:

**A. ANSI Escape Sequence Injection**:
```rust
// ❌ VULNERABLE
AuditEvent::AuthFailure {
    attempted_user: Some("\x1b[31mFAKE ERROR: System compromised\x1b[0m".to_string()),
    ...
}
```

**Impact**: 
- Fake error messages in logs
- Terminal manipulation when viewing logs
- Log parsing tools confused

**B. Control Character Injection**:
```rust
// ❌ VULNERABLE
AuditEvent::PoolDeleted {
    reason: "user_requested\r\n[CRITICAL] Unauthorized access detected".to_string(),
    ...
}
```

**Impact**:
- Fake log lines injected
- Log analysis tools deceived
- False security alerts

**C. Unicode Directional Override**:
```rust
// ❌ VULNERABLE
AuditEvent::TaskSubmitted {
    task_id: "task-123\u{202E}evil\u{202D}".to_string(),  // Right-to-left override
    ...
}
```

**Impact**:
- Text displayed in reverse
- Hides malicious content
- Bypasses visual inspection

**D. Null Byte Injection**:
```rust
// ❌ VULNERABLE
AuditEvent::NodeRegistered {
    node_id: "node-1\0malicious-data".to_string(),
    ...
}
```

**Impact**:
- Truncates logs in C-based tools
- Bypasses length checks
- Hides malicious content

**Mitigation**: **REQUIRED** — Integrate with `input-validation` crate:
```rust
use input_validation::sanitize_string;

// ✅ PROTECTED
let safe_user = sanitize_string(&attempted_user)?;
audit_logger.emit(AuditEvent::AuthFailure {
    attempted_user: Some(safe_user),
    ...
}).await?;
```

**Validation rules**:
- Remove ANSI escape sequences (`\x1b[...m`)
- Remove control characters (except newline in structured fields)
- Remove Unicode directional overrides (`\u{202E}`, `\u{202D}`)
- Remove null bytes (`\0`)
- Limit string length (max 1024 chars per field)

**Status**: ⬜ **NOT IMPLEMENTED** — Critical vulnerability

**References**: 
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #18 (Log Injection)
- `bin/shared-crates/input-validation/.specs/10_expectations.md` §4.5

---

### 2.2 Path Traversal Attacks

**Attack Vector**: Attacker manipulates file paths in audit configuration.

**Threat Scenarios**:

**A. Directory Traversal in Audit Directory**:
```rust
// ❌ VULNERABLE
let config = AuditConfig {
    mode: AuditMode::Local {
        base_dir: PathBuf::from("../../../../etc/cron.d"),  // Escape to system dir
    },
    ...
};
```

**Impact**:
- Audit logs written to arbitrary directories
- Overwrite system files
- Privilege escalation

**B. Symlink Attack**:
```bash
# Attacker creates symlink
ln -s /etc/passwd /var/lib/llorch/audit/queen-rbee/2025-10-01.audit

# Audit logger overwrites /etc/passwd
```

**Impact**:
- Overwrite critical system files
- Denial of service
- Privilege escalation

**Mitigation**: **REQUIRED** — Validate and canonicalize paths:
```rust
use std::path::{Path, PathBuf};

pub fn validate_audit_dir(path: &Path) -> Result<PathBuf, AuditError> {
    // Canonicalize to resolve .. and symlinks
    let canonical = path.canonicalize()
        .map_err(|_| AuditError::InvalidPath("Cannot canonicalize path".into()))?;
    
    // Check path is absolute
    if !canonical.is_absolute() {
        return Err(AuditError::InvalidPath("Path must be absolute".into()));
    }
    
    // Check path is within allowed directory
    let allowed_root = PathBuf::from("/var/lib/llorch/audit");
    if !canonical.starts_with(&allowed_root) {
        return Err(AuditError::InvalidPath("Path outside allowed directory".into()));
    }
    
    // Check path is a directory
    if !canonical.is_dir() {
        return Err(AuditError::InvalidPath("Path is not a directory".into()));
    }
    
    Ok(canonical)
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

**References**: 
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #9 (Path Traversal)

---

### 2.3 Format String Attacks

**Attack Vector**: Attacker injects format specifiers into logged strings.

**Threat Scenarios**:

**A. Format String in Event Data**:
```rust
// ❌ VULNERABLE (if using format! incorrectly)
let reason = user_input;  // Contains "%s%s%s%n"
tracing::info!("Pool deleted: {}", reason);  // Safe (Rust)
```

**Impact**: 
- **Low risk in Rust** (no printf-style vulnerabilities)
- But can confuse log parsers expecting specific formats

**Mitigation**: **RECOMMENDED** — Validate format:
```rust
// Reject strings containing format specifiers
if reason.contains('%') || reason.contains('{') {
    return Err(AuditError::InvalidInput("Format specifiers not allowed".into()));
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — Low priority

---

## 3. Storage Security Attack Surface

### 3.1 Tampering Attacks

**Attack Vector**: Attacker modifies audit log files.

**Threat Scenarios**:

**A. Direct File Modification**:
```bash
# Attacker with file access
sed -i 's/admin@evil.com/admin@good.com/g' /var/lib/llorch/audit/2025-10-01.audit
```

**Impact**:
- Evidence destroyed
- False evidence created
- Compliance violation

**B. Hash Chain Break**:
```json
// Attacker modifies event
{"audit_id": "audit-002", "prev_hash": "abc123", "hash": "def456", ...}
// But doesn't update next event's prev_hash
{"audit_id": "audit-003", "prev_hash": "abc123", ...}  // Should be "def456"
```

**Impact**:
- Tampering detected by verification
- But attacker can delete entire chain

**Mitigation**: **REQUIRED** — Implement hash chain:
```rust
pub struct AuditEventEnvelope {
    pub audit_id: String,
    pub timestamp: DateTime<Utc>,
    pub event: AuditEvent,
    pub prev_hash: String,  // SHA-256 of previous event
    pub hash: String,       // SHA-256 of this event
}

pub fn verify_hash_chain(events: &[AuditEventEnvelope]) -> Result<(), AuditError> {
    for (i, event) in events.iter().enumerate() {
        // Verify event hash
        let computed_hash = compute_event_hash(event);
        if computed_hash != event.hash {
            return Err(AuditError::InvalidChain(format!("Event {} hash mismatch", event.audit_id)));
        }
        
        // Verify chain link
        if i > 0 {
            let prev_event = &events[i - 1];
            if event.prev_hash != prev_event.hash {
                return Err(AuditError::BrokenChain(format!("Chain broken at event {}", event.audit_id)));
            }
        }
    }
    Ok(())
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — Critical vulnerability

**References**: 
- `.specs/02_storage-and-tamper-evidence.md` §2.1

---

### 3.2 Deletion Attacks

**Attack Vector**: Attacker deletes audit log files.

**Threat Scenarios**:

**A. File Deletion**:
```bash
# Attacker with file access
rm /var/lib/llorch/audit/2025-10-01.audit
```

**Impact**:
- Evidence destroyed
- Compliance violation
- Incident investigation impossible

**B. Selective Event Deletion**:
```bash
# Attacker removes specific events from file
sed -i '/auth.failure/d' /var/lib/llorch/audit/2025-10-01.audit
```

**Impact**:
- Specific evidence destroyed
- Hash chain broken (detectable)

**Mitigation**: **REQUIRED** — Multiple layers:

**1. File permissions**:
```bash
# Audit files owned by dedicated user
chown llorch-audit:llorch-audit /var/lib/llorch/audit/
chmod 700 /var/lib/llorch/audit/

# Services can write but not read
# Admins can read but not write
```

**2. Append-only files** (Linux):
```bash
# Set immutable flag after rotation
chattr +a /var/lib/llorch/audit/2025-10-01.audit
```

**3. Manifest tracking**:
```json
{
  "files": [
    {
      "filename": "2025-10-01.audit",
      "event_count": 15234,
      "sha256": "abc123...",
      "created_at": "2025-10-01T00:00:00Z",
      "closed_at": "2025-10-02T00:00:00Z"
    }
  ]
}
```

**4. Remote backup**:
- Real-time replication to separate system
- Write-once storage (S3 Object Lock)

**Status**: ⬜ **NOT IMPLEMENTED** — Critical vulnerability

---

### 3.3 Unauthorized Access Attacks

**Attack Vector**: Attacker reads audit logs without authorization.

**Threat Scenarios**:

**A. File Permission Bypass**:
```bash
# Attacker gains access to audit files
cat /var/lib/llorch/audit/2025-10-01.audit | grep "password"
```

**Impact**:
- Sensitive data leaked (if not redacted)
- System behavior revealed
- Attack planning enabled

**B. Query API Abuse**:
```rust
// Attacker queries all events
let events = audit_logger.query(AuditQuery {
    actor: None,  // All actors
    start_time: None,  // All time
    limit: 1000000,  // Everything
    ...
}).await?;
```

**Impact**:
- Mass data exfiltration
- Privacy violation
- Compliance violation

**Mitigation**: **REQUIRED** — Access control:

**1. File permissions**:
```bash
# Read-only for admins only
chmod 400 /var/lib/llorch/audit/2025-10-01.audit
chown llorch-audit:llorch-audit /var/lib/llorch/audit/2025-10-01.audit
```

**2. Query API authorization**:
```rust
pub async fn query(
    &self,
    query: AuditQuery,
    requester: &ActorInfo,
) -> Result<Vec<AuditEventEnvelope>, AuditError> {
    // Check requester has permission
    if !self.authorize_query(requester, &query).await? {
        return Err(AuditError::Unauthorized);
    }
    
    // Apply row-level security
    let filtered_query = self.apply_row_security(query, requester);
    
    self.execute_query(filtered_query).await
}
```

**3. Audit log access is audited**:
```rust
// Log who accessed audit logs
audit_logger.emit(AuditEvent::AuditLogAccessed {
    timestamp: Utc::now(),
    accessor: requester.clone(),
    query: query.clone(),
}).await?;
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

---

## 4. Memory Safety Attack Surface

### 4.1 Buffer Overflow Attacks

**Attack Vector**: Attacker triggers buffer overflows in event processing.

**Threat Scenarios**:

**A. Unbounded String Fields**:
```rust
// ❌ VULNERABLE
AuditEvent::TaskSubmitted {
    prompt_hash: "A".repeat(1_000_000_000),  // 1GB string
    ...
}
```

**Impact**:
- Memory exhaustion
- Denial of service
- Potential buffer overflow in C-based log parsers

**Mitigation**: **REQUIRED** — Field length limits:
```rust
const MAX_STRING_FIELD_LEN: usize = 1024;
const MAX_HASH_LEN: usize = 64;  // SHA-256 hex
const MAX_ID_LEN: usize = 256;

pub fn validate_event(event: &AuditEvent) -> Result<(), AuditError> {
    // Validate all string fields
    match event {
        AuditEvent::TaskSubmitted { prompt_hash, task_id, model_ref, .. } => {
            if prompt_hash.len() > MAX_HASH_LEN {
                return Err(AuditError::FieldTooLong("prompt_hash"));
            }
            if task_id.len() > MAX_ID_LEN {
                return Err(AuditError::FieldTooLong("task_id"));
            }
            if model_ref.len() > MAX_STRING_FIELD_LEN {
                return Err(AuditError::FieldTooLong("model_ref"));
            }
        }
        // ... validate all variants
    }
    Ok(())
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

---

### 4.2 Integer Overflow Attacks

**Attack Vector**: Attacker triggers integer overflows in size calculations.

**Threat Scenarios**:

**A. Event Count Overflow**:
```rust
// ❌ VULNERABLE
let total_events = self.event_count + 1;  // Overflow if event_count = u64::MAX
```

**Impact**:
- Incorrect event counts
- Manifest corruption
- Potential panic

**B. Buffer Size Overflow**:
```rust
// ❌ VULNERABLE
let buffer_size = event_count * size_per_event;  // Overflow
```

**Impact**:
- Incorrect memory allocation
- Buffer overflow
- Denial of service

**Mitigation**: **REQUIRED** — Use checked arithmetic:
```rust
// ✅ PROTECTED
let total_events = self.event_count.checked_add(1)
    .ok_or(AuditError::IntegerOverflow("event_count"))?;

let buffer_size = event_count.checked_mul(size_per_event)
    .ok_or(AuditError::IntegerOverflow("buffer_size"))?;
```

**Clippy enforcement**:
```rust
#![deny(clippy::integer_arithmetic)]
```

**Status**: ⬜ **NOT IMPLEMENTED** — Medium vulnerability

---

### 4.3 Panic-Induced Denial of Service

**Attack Vector**: Attacker triggers panics to crash service.

**Threat Scenarios**:

**A. Unwrap on Error**:
```rust
// ❌ VULNERABLE
let timestamp = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap();  // Panics if time goes backwards
```

**Impact**:
- Service crash
- Audit logging unavailable
- Evidence loss

**B. Index Out of Bounds**:
```rust
// ❌ VULNERABLE
let first_event = events[0];  // Panics if events is empty
```

**Impact**:
- Service crash
- Denial of service

**Mitigation**: **REQUIRED** — Clippy enforcement:
```rust
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
```

**Status**: ✅ **IMPLEMENTED** — Clippy config in place (src/lib.rs)

**References**: 
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #1 (Mutex Poisoning)

---

## 5. Cryptographic Attack Surface

### 5.1 Weak Hashing Attacks

**Attack Vector**: Attacker exploits weak hash algorithms.

**Threat Scenarios**:

**A. MD5/SHA-1 Usage**:
```rust
// ❌ VULNERABLE
use md5::Md5;
let hash = Md5::digest(event_bytes);  // Collision attacks possible
```

**Impact**:
- Hash collisions
- Tamper evidence bypassed
- Compliance violation

**Mitigation**: **REQUIRED** — Use SHA-256 or stronger:
```rust
// ✅ PROTECTED
use sha2::{Sha256, Digest};

pub fn compute_event_hash(event: &AuditEventEnvelope) -> String {
    let mut hasher = Sha256::new();
    hasher.update(event.audit_id.as_bytes());
    hasher.update(event.timestamp.to_rfc3339().as_bytes());
    hasher.update(serde_json::to_string(&event.event).unwrap().as_bytes());
    hasher.update(event.prev_hash.as_bytes());
    format!("{:x}", hasher.finalize())
}
```

**Algorithm requirements**:
- **Hash**: SHA-256 minimum (FIPS 140-2 approved)
- **Signature**: HMAC-SHA256 or Ed25519
- **No**: MD5, SHA-1, or custom algorithms

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

---

### 5.2 Signature Forgery Attacks

**Attack Vector**: Attacker forges event signatures (platform mode).

**Threat Scenarios**:

**A. Weak Signature Algorithm**:
```rust
// ❌ VULNERABLE
let signature = format!("{:x}", md5::compute(event_bytes));  // Not cryptographic
```

**Impact**:
- Fake events accepted
- Provider accountability bypassed
- Compliance violation

**B. Key Leakage**:
```rust
// ❌ VULNERABLE
tracing::info!("Signing with key: {}", hex::encode(&signing_key));  // Logs key
```

**Impact**:
- Signing key compromised
- All signatures forgeable

**Mitigation**: **REQUIRED** — Strong signatures:
```rust
// ✅ PROTECTED (HMAC-SHA256)
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub fn sign_event(event: &AuditEventEnvelope, key: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(key).unwrap();
    mac.update(event.audit_id.as_bytes());
    mac.update(event.timestamp.to_rfc3339().as_bytes());
    mac.update(serde_json::to_string(&event.event).unwrap().as_bytes());
    let result = mac.finalize();
    hex::encode(result.into_bytes())
}

pub fn verify_signature(event: &AuditEventEnvelope, signature: &str, key: &[u8]) -> bool {
    let expected = sign_event(event, key);
    constant_time_eq(expected.as_bytes(), signature.as_bytes())
}
```

**Or Ed25519** (asymmetric, more secure):
```rust
use ed25519_dalek::{Keypair, Signature, Signer, Verifier};

pub fn sign_event_ed25519(event: &AuditEventEnvelope, keypair: &Keypair) -> String {
    let message = serialize_for_signing(event);
    let signature = keypair.sign(message.as_bytes());
    hex::encode(signature.to_bytes())
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability (platform mode)

---

### 5.3 Timing Attack Vulnerabilities

**Attack Vector**: Attacker uses timing differences to extract secrets.

**Threat Scenarios**:

**A. Non-Constant-Time Comparison**:
```rust
// ❌ VULNERABLE
if signature == expected_signature {  // Early return on mismatch
    return Ok(());
}
```

**Impact**:
- Signature can be brute-forced byte-by-byte
- Authentication bypass

**Mitigation**: **REQUIRED** — Constant-time comparison:
```rust
// ✅ PROTECTED
use subtle::ConstantTimeEq;

pub fn verify_signature(signature: &[u8], expected: &[u8]) -> bool {
    signature.ct_eq(expected).into()
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — Medium vulnerability

---

## 6. Denial of Service Attack Surface

### 6.1 Unbounded Memory Attacks

**Attack Vector**: Attacker exhausts memory via unbounded buffers.

**Threat Scenarios**:

**A. Unbounded Event Buffer**:
```rust
// ❌ VULNERABLE
let mut buffer = Vec::new();
loop {
    buffer.push(event);  // No limit
}
```

**Impact**:
- Memory exhaustion
- OOM killer terminates service
- Denial of service

**B. Large Event Spam**:
```rust
// Attacker sends massive events
for _ in 0..1_000_000 {
    audit_logger.emit(AuditEvent::TaskSubmitted {
        prompt_hash: "A".repeat(1024),
        ...
    }).await?;
}
```

**Impact**:
- Buffer fills up
- Memory exhaustion
- Denial of service

**Mitigation**: **REQUIRED** — Bounded buffers:
```rust
const MAX_BUFFER_EVENTS: usize = 1000;
const MAX_BUFFER_BYTES: usize = 10_485_760;  // 10MB

pub struct AuditLogger {
    tx: mpsc::Sender<AuditEvent>,
}

impl AuditLogger {
    pub fn new(config: AuditConfig) -> Self {
        let (tx, rx) = mpsc::channel(MAX_BUFFER_EVENTS);  // Bounded
        
        tokio::spawn(audit_writer_task(rx, config));
        
        Self { tx }
    }
    
    pub async fn emit(&self, event: AuditEvent) -> Result<(), AuditError> {
        // Non-blocking send with backpressure
        self.tx.try_send(event)
            .map_err(|_| AuditError::BufferFull)
    }
}
```

**Graceful degradation**:
```rust
// Drop events if buffer full (log warning)
if let Err(AuditError::BufferFull) = audit_logger.emit(event).await {
    tracing::warn!("Audit buffer full, dropping event");
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

**References**: 
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #5, #6 (Unbounded memory)

---

### 6.2 Disk Exhaustion Attacks

**Attack Vector**: Attacker fills disk with audit logs.

**Threat Scenarios**:

**A. Event Spam**:
```python
# Attacker floods with events
while True:
    for i in range(1000):
        emit_audit_event(f"spam-{i}")
```

**Impact**:
- Disk fills up
- Service crashes (can't write logs)
- System-wide denial of service

**B. No Rotation**:
```rust
// ❌ VULNERABLE
// Audit file grows forever
```

**Impact**:
- Single file grows to gigabytes
- Disk fills up
- Performance degradation

**Mitigation**: **REQUIRED** — Rotation and limits:
```rust
pub enum RotationPolicy {
    Daily,
    SizeLimit(usize),  // Rotate at 100MB
    Both { daily: bool, size_limit: usize },
}

pub fn should_rotate(&self) -> bool {
    match &self.rotation_policy {
        RotationPolicy::SizeLimit(limit) => {
            self.current_file_size() >= *limit
        }
        RotationPolicy::Daily => {
            let current_date = Utc::now().format("%Y-%m-%d").to_string();
            current_date != self.current_file_date()
        }
        RotationPolicy::Both { daily, size_limit } => {
            // Rotate on either condition
        }
    }
}
```

**Retention policy**:
```rust
pub struct RetentionPolicy {
    pub min_retention_days: u32,  // 2555 days = 7 years
    pub archive_after_days: u32,  // 90 days
    pub delete_after_days: u32,   // 2555 days
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

---

### 6.3 CPU Exhaustion Attacks

**Attack Vector**: Attacker triggers expensive operations.

**Threat Scenarios**:

**A. Hash Chain Verification Spam**:
```rust
// Attacker repeatedly triggers verification
for _ in 0..1000 {
    audit_logger.verify_integrity(VerifyOptions {
        mode: VerifyMode::All,  // Verify all events
    }).await?;
}
```

**Impact**:
- CPU exhaustion
- Service unresponsive
- Denial of service

**B. Expensive Query**:
```rust
// Attacker queries all events
audit_logger.query(AuditQuery {
    start_time: Some(Utc::now() - Duration::days(365 * 10)),  // 10 years
    limit: 10_000_000,
    ...
}).await?;
```

**Impact**:
- CPU exhaustion
- Memory exhaustion
- Denial of service

**Mitigation**: **REQUIRED** — Rate limiting:
```rust
use governor::{Quota, RateLimiter};

pub struct AuditLogger {
    query_limiter: RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>,
}

impl AuditLogger {
    pub async fn query(
        &self,
        query: AuditQuery,
        requester: &str,
    ) -> Result<Vec<AuditEventEnvelope>, AuditError> {
        // Rate limit queries per requester
        self.query_limiter.check_key(&requester.to_string())
            .map_err(|_| AuditError::RateLimitExceeded)?;
        
        // Limit query size
        if query.limit > 10_000 {
            return Err(AuditError::QueryTooLarge);
        }
        
        self.execute_query(query).await
    }
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — Medium vulnerability

---

## 7. Access Control Attack Surface

### 7.1 Privilege Escalation Attacks

**Attack Vector**: Attacker gains unauthorized access to audit logs.

**Threat Scenarios**:

**A. File Permission Bypass**:
```bash
# Attacker exploits misconfigured permissions
chmod 777 /var/lib/llorch/audit/  # Oops
cat /var/lib/llorch/audit/2025-10-01.audit
```

**Impact**:
- Unauthorized access to audit logs
- Privacy violation
- Compliance violation

**B. Query API Authorization Bypass**:
```rust
// ❌ VULNERABLE
pub async fn query(&self, query: AuditQuery) -> Result<Vec<AuditEventEnvelope>> {
    // No authorization check
    self.execute_query(query).await
}
```

**Impact**:
- Any user can query all events
- Privacy violation
- Compliance violation

**Mitigation**: **REQUIRED** — Strict access control:

**1. File permissions**:
```bash
# Audit directory
drwx------  2 llorch-audit llorch-audit  4096 Oct  1 16:48 /var/lib/llorch/audit/

# Audit files (read-only after rotation)
-r--------  1 llorch-audit llorch-audit  52428800 Oct  1 23:59 2025-10-01.audit
```

**2. Query authorization**:
```rust
pub async fn query(
    &self,
    query: AuditQuery,
    requester: &ActorInfo,
) -> Result<Vec<AuditEventEnvelope>, AuditError> {
    // Check requester has admin role
    if !requester.has_role("audit_admin") {
        // Non-admins can only query their own events
        if query.actor.as_ref() != Some(&requester.user_id) {
            return Err(AuditError::Unauthorized);
        }
    }
    
    self.execute_query(query).await
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

---

### 7.2 Row-Level Security Bypass

**Attack Vector**: Attacker queries events they shouldn't see.

**Threat Scenarios**:

**A. Cross-Customer Data Access**:
```rust
// Customer A queries Customer B's events
let events = audit_logger.query(AuditQuery {
    actor: Some("customer-b@example.com".to_string()),  // Different customer
    ...
}).await?;
```

**Impact**:
- Privacy violation
- GDPR violation
- Compliance violation

**Mitigation**: **REQUIRED** — Row-level security:
```rust
pub fn apply_row_security(
    query: AuditQuery,
    requester: &ActorInfo,
) -> AuditQuery {
    if requester.has_role("admin") {
        // Admins can query all
        return query;
    }
    
    // Non-admins can only query their own events
    AuditQuery {
        actor: Some(requester.user_id.clone()),
        ..query
    }
}
```

**Status**: ⬜ **NOT IMPLEMENTED** — High vulnerability

---

## 8. Side Channel Attack Surface

### 8.1 Timing Side Channels

**Attack Vector**: Attacker infers secrets from timing differences.

**Threat Scenarios**:

**A. Signature Verification Timing**:
```rust
// ❌ VULNERABLE
if signature == expected {  // Early return
    return Ok(());
}
```

**Impact**:
- Signature can be brute-forced
- Authentication bypass

**Mitigation**: **REQUIRED** — Constant-time operations (covered in §5.3)

**Status**: ⬜ **NOT IMPLEMENTED**

---

### 8.2 Error Message Information Leakage

**Attack Vector**: Attacker learns secrets from error messages.

**Threat Scenarios**:

**A. Detailed Error Messages**:
```rust
// ❌ VULNERABLE
return Err(AuditError::InvalidSignature {
    expected: hex::encode(&expected_signature),  // Leaks expected signature
    actual: hex::encode(&signature),
});
```

**Impact**:
- Signature leaked
- Attack planning enabled

**B. Path Disclosure**:
```rust
// ❌ VULNERABLE
return Err(AuditError::FileNotFound(format!(
    "Cannot open /var/lib/llorch/audit/secret-file.audit"  // Leaks internal paths
)));
```

**Impact**:
- Internal paths revealed
- Attack surface mapping

**Mitigation**: **REQUIRED** — Generic error messages:
```rust
// ✅ PROTECTED
pub enum AuditError {
    InvalidSignature,  // No details
    FileNotFound,      // No path
    Unauthorized,      // No reason
}

// Log details internally
tracing::error!(
    expected = %hex::encode(&expected_signature),
    actual = %hex::encode(&signature),
    "Signature verification failed"
);

// Return generic error to user
Err(AuditError::InvalidSignature)
```

**Status**: ⬜ **NOT IMPLEMENTED** — Medium vulnerability

---

## 9. Dependency Attack Surface

### 9.1 Supply Chain Attacks

**Attack Vector**: Attacker compromises dependencies.

**Current Dependencies**:
```toml
[dependencies]
thiserror.workspace = true
serde = { workspace = true }
serde_json.workspace = true
tracing.workspace = true
chrono = { workspace = true }
```

**Planned Dependencies**:
- `sha2` — Hashing (REQUIRED)
- `hmac` — Signatures (REQUIRED)
- `ed25519-dalek` — Signatures (platform mode)
- `tokio` — Async I/O (REQUIRED)
- `reqwest` — HTTP client (platform mode)

**Threat Scenarios**:

**A. Malicious Dependency**:
- Compromised crate on crates.io
- Backdoor in dependency
- Data exfiltration

**B. Transitive Dependencies**:
- Deep dependency tree
- Unvetted dependencies
- Supply chain attack

**Mitigation**: **REQUIRED** — Dependency management:

**1. Minimize dependencies**:
- Only essential crates
- Prefer workspace-managed versions
- Avoid large dependency trees

**2. Audit dependencies**:
```bash
cargo audit
cargo tree
```

**3. Lock file**:
- Commit `Cargo.lock`
- Review dependency updates
- CI checks for vulnerabilities

**4. Vendoring** (optional):
```bash
cargo vendor
```

**Current Status**: ✅ **GOOD** — Minimal dependencies (5 crates, all workspace-managed)

**References**: 
- `bin/shared-crates/input-validation/.specs/21_security_verification.md` §6

---

## 10. Attack Surface Summary

### 10.1 Critical Vulnerabilities (MUST FIX for M0)

| # | Vulnerability | Attack Vector | Mitigation | Status |
|---|---------------|---------------|------------|--------|
| 1 | Log injection | ANSI/control chars | `input-validation` integration | ⬜ Not implemented |
| 2 | Path traversal | Directory escape | Path validation | ⬜ Not implemented |
| 3 | Tampering | File modification | Hash chain | ⬜ Not implemented |
| 4 | Deletion | File removal | Permissions + manifest | ⬜ Not implemented |
| 5 | Unbounded memory | Event spam | Bounded buffers | ⬜ Not implemented |
| 6 | Disk exhaustion | Log spam | Rotation + retention | ⬜ Not implemented |

### 10.2 High Priority Vulnerabilities (Fix for Production)

| # | Vulnerability | Attack Vector | Mitigation | Status |
|---|---------------|---------------|------------|--------|
| 7 | Weak hashing | MD5/SHA-1 | SHA-256 minimum | ⬜ Not implemented |
| 8 | Signature forgery | Weak algorithm | HMAC-SHA256/Ed25519 | ⬜ Not implemented |
| 9 | Unauthorized access | No access control | File permissions + API authz | ⬜ Not implemented |
| 10 | Buffer overflow | Unbounded fields | Field length limits | ⬜ Not implemented |
| 11 | Integer overflow | Unchecked arithmetic | Checked arithmetic | ⬜ Not implemented |
| 12 | Privilege escalation | Weak authz | Row-level security | ⬜ Not implemented |

### 10.3 Medium Priority Vulnerabilities (Post-M0)

| # | Vulnerability | Attack Vector | Mitigation | Status |
|---|---------------|---------------|------------|--------|
| 13 | Timing attacks | Non-constant-time | Constant-time comparison | ⬜ Not implemented |
| 14 | Error leakage | Detailed errors | Generic error messages | ⬜ Not implemented |
| 15 | CPU exhaustion | Expensive queries | Rate limiting | ⬜ Not implemented |
| 16 | Format string | Format specifiers | Format validation | ⬜ Not implemented |

### 10.4 Implemented Protections

| Protection | Status | Reference |
|------------|--------|-----------|
| Clippy security lints | ✅ Implemented | src/lib.rs |
| Minimal dependencies | ✅ Implemented | Cargo.toml |
| Input validation reminder | ✅ Documented | src/lib.rs |

---

## 11. Security Testing Requirements

### 11.1 Unit Tests

**Required tests**:
- [ ] Log injection prevention (ANSI, control chars, Unicode)
- [ ] Path traversal rejection
- [ ] Hash chain verification
- [ ] Tampering detection
- [ ] Field length limits
- [ ] Integer overflow prevention
- [ ] Signature verification
- [ ] Constant-time comparison

### 11.2 Integration Tests

**Required tests**:
- [ ] End-to-end audit flow with tampering attempt
- [ ] Unauthorized access rejection
- [ ] Buffer overflow prevention
- [ ] Disk exhaustion handling
- [ ] Query authorization enforcement

### 11.3 Fuzzing

**Fuzz targets**:
- [ ] Event deserialization
- [ ] Path validation
- [ ] Hash computation
- [ ] Signature verification

### 11.4 Security Audit

**Required audits**:
- [ ] Code review by security team
- [ ] Penetration testing
- [ ] Compliance review (GDPR, SOC2)
- [ ] Dependency audit (`cargo audit`)

---

## 12. Compliance Requirements

### 12.1 GDPR

**Requirements**:
- [ ] Data minimization (no PII in logs)
- [ ] Right to erasure (retention policy)
- [ ] Data access requests (query API)
- [ ] Breach notification (integrity verification)

### 12.2 SOC2

**Requirements**:
- [ ] 7-year retention
- [ ] Immutable audit trail
- [ ] Access control
- [ ] Tamper evidence

### 12.3 ISO 27001

**Requirements**:
- [ ] Security event logging
- [ ] Log protection
- [ ] Access control
- [ ] Integrity verification

---

## 13. Refinement Opportunities

### 13.1 Immediate (M0)

1. **Implement input validation** — Integrate with `input-validation` crate
2. **Implement hash chain** — Tamper-evident storage
3. **Implement bounded buffers** — Prevent memory exhaustion
4. **Implement path validation** — Prevent directory traversal
5. **Add field length limits** — Prevent buffer overflow
6. **Implement file rotation** — Prevent disk exhaustion

### 13.2 Medium-Term (Production)

7. **Implement event signatures** — HMAC-SHA256 or Ed25519
8. **Implement access control** — File permissions + API authorization
9. **Implement rate limiting** — Prevent CPU exhaustion
10. **Add constant-time operations** — Prevent timing attacks
11. **Implement row-level security** — Prevent cross-customer access
12. **Add security tests** — Unit, integration, fuzzing

### 13.3 Long-Term (Hardening)

13. **Hardware security module** — Key management
14. **Blockchain integration** — Ultimate tamper evidence
15. **Zero-knowledge proofs** — Privacy-preserving verification
16. **Formal verification** — Mathematical proof of security

---

## 14. References

**Internal Documentation**:
- `.specs/00_overview.md` — Architecture and principles
- `.specs/02_storage-and-tamper-evidence.md` — Storage security
- `.specs/03_security-and-api.md` — API security
- `.specs/10_expectations.md` — Consumer expectations

**Security Audits**:
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Existing vulnerabilities
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Worker-orcd security
- `.docs/security/SECURITY_OVERSEER_SUMMARY.md` — Security posture

**Standards**:
- GDPR — General Data Protection Regulation
- SOC2 — Service Organization Control 2
- ISO 27001 — Information Security Management
- FIPS 140-2 — Cryptographic Module Validation

---

**End of Security Attack Surface Analysis**
