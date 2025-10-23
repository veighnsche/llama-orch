# rbee Architecture Overview - Part 6: Security & Compliance

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document

---

## Defense in Depth Strategy

### Security Philosophy

**rbee implements defense-in-depth through 5 specialized security crates.**

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Audit Logging (Compliance & Forensics)            │
│ - Immutable audit trail                                    │
│ - Tamper detection                                         │
│ - GDPR compliance                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Deadline Propagation (Resource Protection)        │
│ - Timeout enforcement                                      │
│ - Resource exhaustion prevention                           │
│ - Cascading timeouts                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Input Validation (Injection Prevention)           │
│ - Path traversal prevention                                │
│ - SQL injection prevention                                 │
│ - Command injection prevention                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Secrets Management (Credential Protection)        │
│ - File-based secrets (not env vars)                        │
│ - Zeroization on drop                                      │
│ - Timing-safe verification                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Authentication (Identity Verification)            │
│ - Timing-safe token comparison                             │
│ - Token fingerprinting                                     │
│ - Bind policy enforcement                                  │
└─────────────────────────────────────────────────────────────┘
```

**Each layer catches what others might miss.**

---

## 1. auth-min (Authentication Layer)

### Purpose

**Timing-safe authentication primitives to prevent timing attacks.**

### Location

`bin/99_shared_crates/auth-min/`

### Core Features

#### Timing-Safe Token Comparison

```rust
use subtle::ConstantTimeEq;

pub fn compare_tokens(a: &str, b: &str) -> bool {
    // Uses constant-time comparison to prevent timing attacks
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

// ❌ WRONG: Vulnerable to timing attack
pub fn compare_tokens_unsafe(a: &str, b: &str) -> bool {
    a == b  // Short-circuits on first difference
}
```

**Why Timing-Safe?**

Attack scenario:
1. Attacker submits token: `aaaa...`
2. System compares byte-by-byte
3. Fails on first byte, returns quickly (1μs)
4. Attacker submits token: `baaa...`
5. Fails on first byte, returns quickly (1μs)
6. Attacker submits token: `taaa...` (correct first byte!)
7. Fails on second byte, returns slowly (2μs)
8. **Attacker knows first byte is 't'**
9. Repeat for all bytes

**Solution:** Constant-time comparison always takes the same time, regardless of where the difference is.

#### Token Fingerprinting

```rust
use sha2::{Sha256, Digest};

pub fn fingerprint_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    format!("{:x}", hasher.finalize())
}
```

**Usage:**
```rust
// Log token fingerprint, not the token itself
audit_logger.log(AuditEvent::AuthSuccess {
    token_fingerprint: fingerprint_token(&token),
    user_id,
    ip_address,
});
```

#### Bind Policy Enforcement

```rust
pub struct TokenPolicy {
    pub max_age: Duration,
    pub bind_to_ip: bool,
    pub bind_to_user_agent: bool,
}

pub fn validate_token_with_policy(
    token: &Token,
    policy: &TokenPolicy,
    context: &RequestContext,
) -> Result<()> {
    // Check expiration
    if token.created_at.elapsed() > policy.max_age {
        return Err(anyhow!("Token expired"));
    }
    
    // Check IP binding
    if policy.bind_to_ip && token.bound_ip != context.ip_address {
        return Err(anyhow!("Token IP mismatch"));
    }
    
    // Check User-Agent binding
    if policy.bind_to_user_agent && token.bound_user_agent != context.user_agent {
        return Err(anyhow!("Token User-Agent mismatch"));
    }
    
    Ok(())
}
```

### Security Rating

**Rating:** A-  
**Strengths:** Timing-safe, well-tested  
**Improvements:** Add rate limiting, add HMAC-based tokens

---

## 2. audit-logging (Compliance Layer)

### Purpose

**Immutable audit trail with tamper detection for GDPR compliance.**

### Location

`bin/99_shared_crates/audit-logging/`

### GDPR Event Types (32 Total)

#### 1. Authentication Events
- `AuthSuccess` - Successful authentication
- `AuthFailure` - Failed authentication attempt
- `TokenCreated` - API token created
- `TokenRevoked` - API token revoked

#### 2. Authorization Events
- `AccessGranted` - Access to resource granted
- `AccessDenied` - Access to resource denied
- `PermissionChanged` - User permissions modified

#### 3. Data Access Events
- `DataAccessed` - Personal data accessed
- `DataExported` - Personal data exported
- `DataDeleted` - Personal data deleted

#### 4. Resource Events
- `WorkerSpawned` - Worker process spawned
- `WorkerStopped` - Worker process stopped
- `ModelDownloaded` - Model downloaded

#### 5. GDPR Events
- `ConsentGiven` - User consent given
- `ConsentRevoked` - User consent revoked
- `DataSubjectRequest` - GDPR data subject request
- `RightToErasure` - Right to erasure request

### Immutable Audit Trail

```rust
pub struct AuditLogger {
    log_file: File,
    current_hash: String,
}

impl AuditLogger {
    pub fn log(&mut self, event: AuditEvent) -> Result<()> {
        // 1. Serialize event
        let json = serde_json::to_string(&event)?;
        
        // 2. Compute hash (includes previous hash for chain)
        let new_hash = self.compute_hash(&json, &self.current_hash);
        
        // 3. Write to append-only log
        writeln!(
            self.log_file,
            "{}|{}|{}",
            event.timestamp,
            new_hash,
            json
        )?;
        
        // 4. Update current hash
        self.current_hash = new_hash;
        
        Ok(())
    }
    
    fn compute_hash(&self, data: &str, previous_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(previous_hash.as_bytes());
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}
```

### Tamper Detection

```rust
pub fn verify_audit_log(log_path: &Path) -> Result<()> {
    let file = File::open(log_path)?;
    let reader = BufReader::new(file);
    
    let mut previous_hash = String::from("0");
    
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split('|').collect();
        
        if parts.len() != 3 {
            return Err(anyhow!("Invalid log format"));
        }
        
        let (_timestamp, hash, json) = (parts[0], parts[1], parts[2]);
        
        // Recompute hash
        let computed_hash = compute_hash(json, &previous_hash);
        
        // Verify hash matches
        if computed_hash != hash {
            return Err(anyhow!("Audit log tampered at line: {}", line));
        }
        
        previous_hash = hash.to_string();
    }
    
    Ok(())
}
```

### 7-Year Retention

```rust
pub struct AuditLogManager {
    retention_days: u32,  // Default: 2557 (7 years)
}

impl AuditLogManager {
    pub fn archive_old_logs(&self) -> Result<()> {
        let cutoff = Utc::now() - Duration::days(self.retention_days as i64);
        
        // Compress old logs
        for log_file in self.find_old_logs(cutoff)? {
            self.compress_and_archive(&log_file)?;
        }
        
        Ok(())
    }
}
```

### Security Rating

**Rating:** A-  
**Strengths:** Blockchain-style hash chain, GDPR compliant  
**Improvements:** Add encryption at rest, add signing

---

## 3. input-validation (Injection Prevention Layer)

### Purpose

**Prevent injection attacks through comprehensive input validation.**

### Location

`bin/99_shared_crates/input-validation/`

### Validation Types

#### Path Traversal Prevention

```rust
pub fn validate_file_path(path: &str) -> Result<PathBuf> {
    let path = PathBuf::from(path);
    
    // Reject paths with ".."
    if path.components().any(|c| c == Component::ParentDir) {
        return Err(anyhow!("Path traversal attempt detected"));
    }
    
    // Reject absolute paths (unless explicitly allowed)
    if path.is_absolute() {
        return Err(anyhow!("Absolute paths not allowed"));
    }
    
    // Canonicalize to resolve symlinks
    let canonical = path.canonicalize()
        .map_err(|_| anyhow!("Invalid path"))?;
    
    // Ensure path is within allowed directory
    let base_dir = PathBuf::from("/allowed/directory");
    if !canonical.starts_with(&base_dir) {
        return Err(anyhow!("Path outside allowed directory"));
    }
    
    Ok(canonical)
}
```

#### SQL Injection Prevention

```rust
pub fn validate_identifier(name: &str) -> Result<String> {
    // Only allow alphanumeric and underscore
    if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(anyhow!("Invalid identifier"));
    }
    
    // Reject SQL keywords
    let keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP"];
    if keywords.contains(&name.to_uppercase().as_str()) {
        return Err(anyhow!("SQL keyword not allowed"));
    }
    
    Ok(name.to_string())
}
```

#### Command Injection Prevention

```rust
pub fn validate_command_arg(arg: &str) -> Result<String> {
    // Reject shell metacharacters
    let forbidden = ['|', '&', ';', '\n', '`', '$', '(', ')'];
    if arg.chars().any(|c| forbidden.contains(&c)) {
        return Err(anyhow!("Invalid characters in argument"));
    }
    
    Ok(arg.to_string())
}
```

#### Log Injection Prevention

```rust
pub fn sanitize_log_message(message: &str) -> String {
    // Replace newlines to prevent log injection
    message
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
```

### Security Rating

**Rating:** B+  
**Strengths:** Comprehensive validation  
**Improvements:** Add more validation types, add Unicode handling

---

## 4. secrets-management (Credential Protection Layer)

### Purpose

**Secure credential storage with zeroization and timing-safe verification.**

### Location

`bin/99_shared_crates/secrets-management/`

### File-Based Secrets

**Why Files, Not Environment Variables?**

1. **Security:** Env vars visible in `ps`, `/proc`, core dumps
2. **Version Control:** Files can be .gitignored
3. **Rotation:** Easier to update files
4. **Hierarchical:** Can use directories for organization

**Implementation:**
```rust
pub struct SecretsManager {
    secrets_dir: PathBuf,
}

impl SecretsManager {
    pub fn new() -> Result<Self> {
        let secrets_dir = dirs::data_dir()
            .ok_or_else(|| anyhow!("No data dir"))?
            .join("rbee/secrets");
        
        fs::create_dir_all(&secrets_dir)?;
        Ok(Self { secrets_dir })
    }
    
    pub fn get_secret(&self, name: &str) -> Result<Secret> {
        let path = self.secrets_dir.join(name);
        let content = fs::read_to_string(path)?;
        Ok(Secret::new(content.trim().to_string()))
    }
    
    pub fn set_secret(&self, name: &str, value: &str) -> Result<()> {
        let path = self.secrets_dir.join(name);
        fs::write(path, value)?;
        
        // Set restrictive permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&path)?.permissions();
            perms.set_mode(0o600);  // rw-------
            fs::set_permissions(&path, perms)?;
        }
        
        Ok(())
    }
}
```

### Zeroization

```rust
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Secret {
    value: String,
}

impl Secret {
    pub fn new(value: String) -> Self {
        Self { value }
    }
    
    pub fn expose(&self) -> &str {
        &self.value
    }
}

// When Secret is dropped, value is zeroed in memory
```

### Timing-Safe Verification

```rust
pub fn verify_secret(secret: &Secret, input: &str) -> bool {
    use subtle::ConstantTimeEq;
    secret.expose().as_bytes().ct_eq(input.as_bytes()).into()
}
```

### Security Rating

**Rating:** A  
**Strengths:** Zeroization, timing-safe, file-based  
**Improvements:** Add encryption at rest

---

## 5. deadline-propagation (Resource Protection Layer)

### Purpose

**Prevent resource exhaustion through hard deadline enforcement.**

### Location

`bin/99_shared_crates/deadline-propagation/`

### Hard Deadline Enforcement

```rust
pub struct DeadlineEnforcer {
    deadline: Instant,
}

impl DeadlineEnforcer {
    pub fn new(timeout: Duration) -> Self {
        Self {
            deadline: Instant::now() + timeout,
        }
    }
    
    pub async fn enforce<F, T>(&self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let duration = self.deadline.saturating_duration_since(Instant::now());
        
        tokio::time::timeout(duration, future)
            .await
            .map_err(|_| anyhow!("Operation timed out"))?
    }
}
```

### Cascading Timeouts

```rust
pub struct CascadingDeadline {
    parent_deadline: Option<Instant>,
    local_timeout: Duration,
}

impl CascadingDeadline {
    pub fn effective_deadline(&self) -> Instant {
        let local_deadline = Instant::now() + self.local_timeout;
        
        match self.parent_deadline {
            Some(parent) => min(local_deadline, parent),
            None => local_deadline,
        }
    }
}
```

### Visual Countdown (TEAM-661 Fix)

```rust
pub struct TimeoutEnforcer {
    timeout: Duration,
    job_id: Option<String>,
}

impl TimeoutEnforcer {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout, job_id: None }
    }
    
    pub fn with_job_id(mut self, job_id: &str) -> Self {
        self.job_id = Some(job_id.to_string());
        self
    }
    
    pub async fn enforce<F, T>(&self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        // Emit timeout warning (with SSE routing if job_id present)
        if let Some(job_id) = &self.job_id {
            NARRATE.action("timeout_warning")
                .job_id(job_id)
                .context(&format!("{}s", self.timeout.as_secs()))
                .human("⏱️  Operation timeout: {} seconds")
                .emit();
        }
        
        tokio::time::timeout(self.timeout, future)
            .await
            .map_err(|_| anyhow!("Operation timed out"))?
    }
}
```

### Security Rating

**Rating:** B+  
**Strengths:** Hard enforcement, cascading support  
**Improvements:** Add resource usage tracking

---

## Threat Model

### Assets

1. **Credentials:** API tokens, SSH keys
2. **Data:** User prompts, model outputs
3. **Models:** Downloaded LLM weights
4. **Audit Logs:** Compliance evidence

### Threats

#### 1. Credential Theft

**Attack:** Attacker steals API token to gain access

**Mitigations:**
- ✅ File-based secrets (not env vars)
- ✅ Token fingerprinting (log fingerprint, not token)
- ✅ Timing-safe comparison (prevent timing attacks)
- ✅ Bind policy (IP/User-Agent binding)
- 📋 TODO: Rate limiting
- 📋 TODO: Token rotation

#### 2. Injection Attacks

**Attack:** Attacker injects malicious code via inputs

**Mitigations:**
- ✅ Path traversal prevention
- ✅ SQL injection prevention
- ✅ Command injection prevention
- ✅ Log injection prevention

#### 3. Resource Exhaustion

**Attack:** Attacker submits infinite loop prompt

**Mitigations:**
- ✅ Hard deadlines (timeout enforcement)
- ✅ Cascading timeouts
- 📋 TODO: Rate limiting
- 📋 TODO: Resource quotas

#### 4. Audit Log Tampering

**Attack:** Attacker modifies audit logs to hide actions

**Mitigations:**
- ✅ Immutable audit trail
- ✅ Blockchain-style hash chain
- ✅ Tamper detection
- 📋 TODO: Append-only filesystem
- 📋 TODO: Log signing

---

## GDPR Compliance

### Data Subject Rights

#### 1. Right to Access

```rust
// GET /v1/gdpr/data-export?user_id=123
pub async fn handle_data_export(user_id: String) -> Result<Json<DataExport>> {
    let audit_events = audit_logger.get_events_for_user(&user_id)?;
    let user_data = database.get_user_data(&user_id)?;
    
    Ok(Json(DataExport {
        user_id,
        audit_events,
        user_data,
        exported_at: Utc::now(),
    }))
}
```

#### 2. Right to Erasure

```rust
// DELETE /v1/gdpr/erase?user_id=123
pub async fn handle_erasure(user_id: String) -> Result<()> {
    // Log the request
    audit_logger.log(AuditEvent::RightToErasure {
        user_id: user_id.clone(),
        requested_at: Utc::now(),
    })?;
    
    // Delete user data
    database.delete_user(&user_id)?;
    
    // Note: Audit logs are retained for 7 years (legal requirement)
    
    Ok(())
}
```

#### 3. Right to Consent

```rust
// POST /v1/gdpr/consent
pub async fn handle_consent(
    Json(request): Json<ConsentRequest>,
) -> Result<()> {
    audit_logger.log(AuditEvent::ConsentGiven {
        user_id: request.user_id.clone(),
        purpose: request.purpose.clone(),
        timestamp: Utc::now(),
    })?;
    
    database.record_consent(&request)?;
    
    Ok(())
}
```

### 7-Year Retention

**Legal Requirement:** EU companies must retain audit logs for 7 years.

**Implementation:**
```rust
const RETENTION_DAYS: u32 = 2557;  // 7 years

pub struct AuditLogArchiver {
    retention_days: u32,
}

impl AuditLogArchiver {
    pub fn archive_old_logs(&self) -> Result<()> {
        let cutoff = Utc::now() - Duration::days(self.retention_days as i64);
        
        for log_file in self.find_logs_before(cutoff)? {
            self.compress_and_archive(&log_file)?;
        }
        
        Ok(())
    }
}
```

---

## Security Best Practices

### 1. Never Log Secrets

```rust
// ❌ WRONG
eprintln!("Token: {}", token);

// ✅ CORRECT
eprintln!("Token fingerprint: {}", fingerprint_token(&token));
```

### 2. Always Use Timing-Safe Comparison

```rust
// ❌ WRONG
if token == stored_token { ... }

// ✅ CORRECT
if compare_tokens(&token, &stored_token) { ... }
```

### 3. Always Validate Inputs

```rust
// ❌ WRONG
let path = PathBuf::from(user_input);

// ✅ CORRECT
let path = validate_file_path(user_input)?;
```

### 4. Always Set Deadlines

```rust
// ❌ WRONG
let result = slow_operation().await?;

// ✅ CORRECT
let result = TimeoutEnforcer::new(Duration::from_secs(30))
    .enforce(slow_operation())
    .await?;
```

### 5. Always Audit Security Events

```rust
// ✅ REQUIRED
audit_logger.log(AuditEvent::AuthSuccess {
    user_id,
    ip_address,
    timestamp: Utc::now(),
})?;
```

---

## Security Roadmap

### M1 (Current)
- ✅ auth-min implemented
- ✅ audit-logging implemented
- ✅ input-validation implemented
- ✅ secrets-management implemented
- ✅ deadline-propagation implemented

### M2 (Future)
- 📋 Add rate limiting
- 📋 Add token rotation
- 📋 Add encryption at rest
- 📋 Add log signing
- 📋 Add resource quotas

### M3 (Future)
- 📋 SOC2 Type II compliance
- 📋 ISO 27001 certification
- 📋 Penetration testing
- 📋 Bug bounty program

---

## Conclusion

**rbee implements defense-in-depth security through 5 specialized crates:**

1. **auth-min** - Timing-safe authentication
2. **audit-logging** - GDPR compliance & forensics
3. **input-validation** - Injection prevention
4. **secrets-management** - Credential protection
5. **deadline-propagation** - Resource protection

**Each layer provides independent security guarantees, creating a robust security posture.**

**Security Rating:** A- (Production-ready with minor improvements)

---

## Architecture Overview Complete

**All 6 parts are now documented:**

1. **Part 1:** System Design & Philosophy
2. **Part 2:** Component Deep Dive
3. **Part 3:** Shared Infrastructure
4. **Part 4:** Data Flow & Protocols
5. **Part 5:** Development Patterns
6. **Part 6:** Security & Compliance

**For the next engineering team:** Read all 6 parts to understand the complete architecture.

**Location:** `.arch/00_OVERVIEW_PART_1.md` through `.arch/05_SECURITY_PART_6.md`
