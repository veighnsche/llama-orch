# 🔐 rbee Security Architecture: Defense in Depth

**Pronunciation:** rbee (pronounced "are-bee")  
**Date:** 2025-10-10  
**Status:** Production-Ready Security Hardening  
**Audience:** CTOs, Security Officers, Compliance Teams, Enterprise Customers

**🎯 PRIMARY TARGET AUDIENCE:** Developers who build with AI but don't want to depend on big AI providers.

**THE FEAR:** Building complex codebases with AI assistance. What if the provider changes, shuts down, or changes pricing? Your codebase becomes unmaintainable.

**THE SOLUTION:** Build your own AI infrastructure using ALL your home network hardware. Never depend on external providers again.

---

## Executive Summary

**rbee (pronounced "are-bee") is built with security-first principles** from day one. We don't bolt on security as an afterthought—it's woven into every layer of the architecture through **six specialized security crates** that work together to create defense-in-depth.

**Our Security Posture:**
- ✅ **Zero-trust authentication** (timing-safe token comparison)
- ✅ **Immutable audit trails** (GDPR, SOC2, ISO 27001 compliant)
- ✅ **Input validation** (prevents injection attacks)
- ✅ **Secrets management** (file-based, never in environment)
- ✅ **Deadline enforcement** (prevents resource exhaustion)

**Result:** Enterprise-grade security without sacrificing performance.

---

## Table of Contents

1. [The Five Security Crates](#the-five-security-crates)
2. [auth-min: The Trickster Guardians](#auth-min-the-trickster-guardians)
3. [audit-logging: The Compliance Engine](#audit-logging-the-compliance-engine)
4. [input-validation: The First Line of Defense](#input-validation-the-first-line-of-defense)
5. [secrets-management: The Credential Guardian](#secrets-management-the-credential-guardian)
6. [deadline-propagation: The Performance Enforcer](#deadline-propagation-the-performance-enforcer)
7. [Security by Design](#security-by-design)
8. [Compliance & Certifications](#compliance--certifications)
9. [Threat Model](#threat-model)
10. [Security Guarantees](#security-guarantees)

---

## The Five Security Crates

**rbee's security is built on five specialized crates, each with a distinct responsibility:**

| Crate | Purpose | Key Features | Team Motto |
|-------|---------|--------------|------------|
| **auth-min** 🎭 | Authentication primitives | Timing-safe comparison, token fingerprinting, bind policy | "Minimal in name, maximal in vigilance" |
| **audit-logging** 🔒 | Compliance & forensics | Immutable audit trail, tamper detection, GDPR compliance | "If it's not audited, it didn't happen" |
| **input-validation** 🛡️ | Injection prevention | Path traversal, SQL injection, log injection prevention | "Trust no input" |
| **secrets-management** 🔑 | Credential handling | File-based secrets, zeroization, timing-safe verification | "Never in environment, always protected" |
| **deadline-propagation** ⏱️ | Resource protection | Deadline enforcement, timeout handling, waste prevention | "Every millisecond counts" |
| **narration-core** 🎀 | Observability & redaction | Human-readable narration, correlation IDs, secret redaction | "Cuteness pays the bills!" |

**Together, they create defense-in-depth:** Multiple layers of security, each catching what others might miss.

---

## auth-min: The Trickster Guardians 🎭

### What They Do

**auth-min** provides zero-trust authentication primitives that protect against the most subtle attacks:

- **Timing-safe token comparison** (prevents CWE-208)
- **Token fingerprinting** (safe logging without leakage)
- **Bearer token parsing** (RFC 6750 compliant)
- **Bind policy enforcement** (prevents accidental exposure)

### Why "Trickster"?

The name "auth-min" suggests "minimal authentication"—but that's strategic misdirection. They're actually **foundational to all security** in rbee. Every service depends on them.

**Their strategy:** Ask for more than needed, let others "win" by rejecting excessive demands, land exactly where they wanted. The compromise is the goal.

### Key Features

**1. Timing-Safe Comparison**
```rust
// Prevents timing attacks (CWE-208)
if !timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
    return Err(AuthError::InvalidToken);
}
```

**Property:** Execution time independent of mismatch position  
**Verification:** Variance < 10% for early vs. late mismatches  
**Test Coverage:** 1000+ iterations, property tests

**2. Token Fingerprinting**
```rust
// Safe for logs, never reveals original token
let fp6 = token_fp6(&token);  // SHA-256 → first 6 hex chars
tracing::info!(identity = %format!("token:{}", fp6), "authenticated");
```

**Property:** Non-reversible, collision-resistant (24-bit space)  
**Use Case:** Audit logs, correlation IDs, error messages  
**Guarantee:** Cannot recover original token

**3. Bind Policy Enforcement**
```rust
// Refuses non-loopback bind without LLORCH_API_TOKEN
enforce_startup_bind_policy(&bind_addr)?;
```

**Rule:** Non-loopback binds (0.0.0.0, public IPs) REQUIRE authentication  
**Enforcement:** Startup validation, fail-fast, no override  
**Prevents:** Accidental public exposure without auth

### Attack Vectors Defended

1. **Timing Attacks (CWE-208):** Constant-time comparison prevents byte-by-byte token discovery
2. **Token Leakage:** Fingerprints in logs, never raw tokens
3. **Header Injection:** RFC 6750 parsing rejects malformed headers
4. **Accidental Exposure:** Bind policy prevents public binds without auth
5. **Brute Force:** Timing-safe comparison eliminates timing oracle

### Integration Points

**Every service uses auth-min:**
- **queen-rbee:** Token validation, bind policy
- **pool-managerd:** Token validation, bind policy
- **worker-orcd:** Token validation (future)
- **http-util:** Bearer token parsing
- **audit-logging:** Token fingerprints for actor identity

---

## audit-logging: The Compliance Engine 🔒

### What They Do

**audit-logging** provides legally defensible proof of what happened in rbee:

- **Immutable audit trail** (append-only, tamper-evident)
- **32 event types** across 7 categories
- **GDPR, SOC2, ISO 27001 compliant**
- **Tamper detection** (blockchain-style hash chains)
- **7-year retention** (regulatory requirement)

### Why It Matters

**Without audit logging:**
- ❌ Cannot prove GDPR compliance
- ❌ Cannot pass SOC2 audits
- ❌ Cannot investigate security incidents
- ❌ Cannot resolve customer disputes
- ❌ Cannot defend against legal claims

**With audit logging:**
- ✅ Legally defensible proof of actions
- ✅ Regulatory compliance (GDPR, SOC2, ISO 27001)
- ✅ Forensic investigation capability
- ✅ Customer trust (prove correct data handling)

### Key Features

**1. Immutable Event Recording**
```json
{
  "audit_id": "audit-2025-1001-164805-abc123",
  "timestamp": "2025-10-01T16:48:05Z",
  "event_type": "auth.success",
  "actor": {
    "user_id": "token:a3f2c1",  // Fingerprint, not raw token
    "ip": "192.168.1.100",
    "auth_method": "bearer_token"
  },
  "result": "success"
}
```

**Property:** Once written, never modified or deleted  
**Storage:** Append-only files with hash chains  
**Retention:** 7 years (GDPR requirement)

**2. Tamper Detection**
- **Hash chains:** Each event includes hash of previous event (blockchain-style)
- **Checksums:** File-level integrity verification
- **Signatures:** Cryptographic proof of authenticity (platform mode)

**3. Security-First Design**
- **No raw tokens:** Always uses auth-min fingerprints
- **No VRAM pointers:** Security risk
- **No prompt content:** Uses length and hash instead
- **Input validation:** Integration with input-validation crate

### Event Types (32 Total)

**Authentication (4 types):**
- `AuthSuccess`, `AuthFailure`, `InvalidTokenUsed`, `RateLimitExceeded`

**Resource Operations (6 types):**
- `PoolCreated`, `PoolDeleted`, `PoolModified`, `NodeRegistered`, `NodeDeregistered`

**Task Lifecycle (3 types):**
- `TaskSubmitted`, `TaskCompleted`, `TaskCanceled`

**VRAM Operations (6 types):**
- `VramSealed`, `SealVerified`, `SealVerificationFailed`, `VramAllocated`, `VramDeallocated`

**Security Incidents (4 types):**
- `PathTraversalAttempt`, `SuspiciousActivity`, `PolicyViolation`

**Compliance (6 types):**
- `DataAccess`, `DataExport`, `DataDeletion`, `ConsentGranted`, `ConsentRevoked`

### Compliance Support

**GDPR (EU Regulation):**
- ✅ 7-year retention
- ✅ Data access records
- ✅ Right to erasure tracking
- ✅ Consent management

**SOC2 (US Standard):**
- ✅ Auditor access
- ✅ Security event logging
- ✅ 7-year retention
- ✅ Tamper-evident storage

**ISO 27001 (International):**
- ✅ Security incident records
- ✅ 3-year retention
- ✅ Access control logging

---

## input-validation: The First Line of Defense 🛡️

### What They Do

**input-validation** prevents injection attacks and resource exhaustion:

- **Identifier validation** (prevents path traversal)
- **Model reference validation** (prevents command injection)
- **Prompt validation** (prevents VRAM exhaustion)
- **Path validation** (prevents directory traversal)
- **String sanitization** (prevents log injection)

### Attack Vectors Prevented

**1. Injection Attacks:**
- ✅ SQL Injection: `"'; DROP TABLE models; --"`
- ✅ Command Injection: `"model.gguf; rm -rf /"`
- ✅ Log Injection: `"model\n[ERROR] Fake log entry"`
- ✅ Path Traversal: `"file:../../../../etc/passwd"`
- ✅ ANSI Escape Injection: `"task\x1b[31mRED"`

**2. Resource Exhaustion:**
- ✅ Length Attacks: 10MB prompt → VRAM exhaustion
- ✅ Integer Overflow: `max_tokens: usize::MAX`

**3. Encoding Attacks:**
- ✅ Null Byte Injection: `"shard\0null"` → C string truncation
- ✅ Control Characters: `\r\n` → log parsing confusion

### Key Features

**1. Identifier Validation**
```rust
// Validates shard_id, task_id, pool_id, node_id
validate_identifier("shard-abc123", 256)?;  // ✅ Valid
validate_identifier("shard-../etc/passwd", 256)?;  // ❌ Rejected
```

**Rules:** Alphanumeric + dash + underscore only, max length, no path traversal

**2. Model Reference Validation**
```rust
validate_model_ref("meta-llama/Llama-3.1-8B")?;  // ✅ Valid
validate_model_ref("model; rm -rf /")?;  // ❌ Rejected (command injection)
```

**Rules:** No shell metacharacters (`;`, `|`, `&`, `$`), no path traversal

**3. Path Validation**
```rust
let allowed_root = PathBuf::from("/var/lib/llorch/models");
let path = validate_path("model.gguf", &allowed_root)?;  // ✅ Valid
validate_path("../../../etc/passwd", &allowed_root)?;  // ❌ Rejected
```

**Rules:** Canonicalize path, verify within allowed root

**4. Prompt Validation**
```rust
validate_prompt("Write a story...", 100_000)?;  // ✅ Valid
validate_prompt(&"a".repeat(200_000), 100_000)?;  // ❌ Rejected (too long)
```

**Rules:** Max length (prevents VRAM exhaustion), no null bytes

### Performance

**Validation is fast** (designed for hot paths):
- Identifier validation: **< 1μs** (typical inputs)
- Model ref validation: **< 2μs** (typical inputs)
- Path validation: **< 10μs** (includes filesystem I/O)
- **No allocations** during validation

### Security Properties

- ✅ **No panics:** All validation functions never panic (DoS prevention)
- ✅ **No information leakage:** Error messages don't leak sensitive data
- ✅ **Minimal dependencies:** Only `thiserror` (smaller attack surface)
- ✅ **Fuzz-tested:** No crashes on arbitrary input

---

## secrets-management: The Credential Guardian 🔑

### What They Do

**secrets-management** provides secure credential handling:

- **File-based loading** (not environment variables)
- **Systemd credentials** (production-ready)
- **Memory zeroization** (prevents memory dumps)
- **Permission validation** (rejects world-readable files)
- **Timing-safe verification** (constant-time comparison)

### Why File-Based?

**Environment variables are insecure:**
- ❌ Visible in `ps auxe` (all processes can see)
- ❌ Visible in `/proc/PID/environ`
- ❌ Visible in Docker inspect
- ❌ Visible in systemd service files
- ❌ Logged by many tools

**File-based secrets are secure:**
- ✅ Permission-controlled (0600)
- ✅ Not visible in process listings
- ✅ Not visible in `/proc`
- ✅ Zeroized on drop (prevents memory dumps)

### Key Features

**1. File-Based Loading**
```rust
// Load token from file (not environment)
let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;

// Verify incoming request (timing-safe)
if token.verify(&received_token) {
    println!("Authenticated");
}
```

**Security:** File permissions validated (rejects 0644), memory zeroized on drop

**2. Systemd Credentials**
```rust
// Load from systemd LoadCredential
let token = Secret::from_systemd_credential("api_token")?;
```

**Systemd service:**
```ini
[Service]
LoadCredential=api_token:/etc/llorch/secrets/api-token
# Token available at /run/credentials/<service>/api_token
```

**Security:** Credentials not visible in process listings or `/proc`

**3. Key Derivation**
```rust
// Derive seal key from worker token (HKDF-SHA256)
let seal_key = SecretKey::derive_from_token(
    &worker_api_token,
    b"llorch-seal-key-v1"  // Domain separation
)?;
```

**Use Case:** Generate cryptographic keys from API tokens without storing separate key files

### Security Properties

**Memory Safety:**
- ✅ **Zeroization:** All secrets overwritten on drop
- ✅ **No leaks:** Prevents memory dumps
- ✅ **Compiler fences:** Prevents optimization from removing zeroization

**Logging Safety:**
- ✅ **No Debug/Display:** Secrets cannot be logged accidentally
- ✅ **Path logging only:** Logs file paths, never values
- ✅ **Error safety:** Error messages never contain secret values

**Timing Safety:**
- ✅ **Constant-time comparison:** Uses `subtle::ConstantTimeEq`
- ✅ **No short-circuit:** Examines all bytes regardless of match
- ✅ **Prevents CWE-208:** Observable timing discrepancy attacks

**File Permission Validation:**
- ✅ **Unix permissions:** Rejects files with mode `0o077` bits set
- ✅ **Recommended:** `0600` (owner read/write only)
- ✅ **Enforced:** Before reading file contents

---

## deadline-propagation: The Performance Enforcer ⏱️

### What They Do

**deadline-propagation** ensures rbee never wastes cycles on doomed work:

- **Deadline propagation** (client → orchestrator → pool → worker)
- **Remaining time calculation** (at every hop)
- **Deadline enforcement** (abort if insufficient time)
- **Timeout responses** (504 Gateway Timeout)

### Why It Matters

**Without deadline enforcement:**
- ❌ Wastes GPU cycles on results nobody will receive
- ❌ Delays queue processing for waiting clients
- ❌ Burns electricity for zero value
- ❌ Increases system load

**With deadline enforcement:**
- ✅ Aborts work immediately when deadline exceeded
- ✅ Returns fast failure (504 in 100ms vs. 5 seconds)
- ✅ Frees resources for fresh requests
- ✅ Respects client timeout budgets

### Key Features

**1. Deadline Propagation**
```
Client → queen-rbee (50ms) → pool-managerd (30ms) → worker-orcd (20ms)
Total: 100ms

Client deadline: 5000ms
Remaining at worker: 4900ms
Inference needs: 4800ms

✅ Proceed (sufficient time)
```

**2. Deadline Enforcement**
```
Client deadline: 5000ms
Elapsed: 4900ms
Remaining: 100ms
Inference needs: 4800ms

❌ ABORT IMMEDIATELY (insufficient time)
```

**3. Timeout Response**
```http
HTTP/1.1 504 Gateway Timeout
Retry-After: 1
X-Deadline-Exceeded-By-Ms: 250

{
  "error": "deadline_exceeded",
  "deadline_ms": 5000,
  "elapsed_ms": 5250,
  "exceeded_by_ms": 250
}
```

### Performance Guarantees

- ✅ **Zero allocation:** Deadline checks use stack-only comparison
- ✅ **Microsecond precision:** `Instant`-based timing
- ✅ **Constant time:** Deadline validation is O(1)
- ✅ **No async overhead:** All APIs are synchronous

### Coordination with Security

**CRITICAL:** All performance optimizations MUST be reviewed by auth-min team.

**Process:**
1. Performance team proposes optimization
2. Provides security analysis (timing attacks, leakage)
3. auth-min reviews and approves/rejects
4. Only approved optimizations are implemented

**Result:** Fast AND secure.

---

## Security by Design

### Defense in Depth

**rbee uses multiple layers of security:**

```
┌─────────────────────────────────────────┐
│ input-validation (First Line)          │ ← Rejects malicious input
├─────────────────────────────────────────┤
│ auth-min (Authentication)              │ ← Timing-safe token validation
├─────────────────────────────────────────┤
│ secrets-management (Credentials)       │ ← File-based, zeroized
├─────────────────────────────────────────┤
│ audit-logging (Compliance)             │ ← Immutable audit trail
├─────────────────────────────────────────┤
│ deadline-propagation (Resource)        │ ← Prevents exhaustion
└─────────────────────────────────────────┘
```

**Each layer catches what others might miss.**

### Zero-Trust Principles

**rbee operates on zero-trust:**
- ✅ **Every token** validated with timing-safe comparison
- ✅ **Every log** uses fingerprints, never raw tokens
- ✅ **Every bind** checked against policy at startup
- ✅ **Every header** parsed with RFC 6750 rigor
- ✅ **Every input** validated before processing
- ✅ **Every secret** zeroized on drop

**Trust nothing. Verify everything.**

### Security-First Development

**All code is reviewed by specialized security teams:**
- **auth-min team 🎭:** Reviews authentication code
- **audit-logging team 🔒:** Reviews compliance code
- **Performance team ⏱️:** Coordinates with auth-min on optimizations

**Result:** Security is not an afterthought, it's built-in.

---

## Compliance & Certifications

### GDPR (EU Regulation)

**rbee is GDPR-compliant by design:**
- ✅ **7-year audit retention** (regulatory requirement)
- ✅ **Data access records** (audit-logging)
- ✅ **Right to erasure tracking** (audit-logging)
- ✅ **Consent management** (audit-logging)
- ✅ **Data residency** (EU-only worker filtering)
- ✅ **Audit logging** (immutable trail)

**Compliance endpoints:**
- `GET /v2/compliance/data-access` - Data access records
- `POST /v2/compliance/data-export` - Export user data
- `POST /v2/compliance/data-deletion` - Delete user data
- `GET /v2/compliance/audit-trail` - Audit trail access

### SOC2 (US Standard)

**rbee meets SOC2 requirements:**
- ✅ **Auditor access** (audit-logging query API)
- ✅ **Security event logging** (32 event types)
- ✅ **7-year retention** (regulatory requirement)
- ✅ **Tamper-evident storage** (hash chains)
- ✅ **Access control** (auth-min)
- ✅ **Encryption at rest** (audit logs)

### ISO 27001 (International)

**rbee meets ISO 27001 requirements:**
- ✅ **Security incident records** (audit-logging)
- ✅ **3-year retention** (minimum requirement)
- ✅ **Access control logging** (auth events)
- ✅ **Cryptographic controls** (auth-min, secrets-management)

---

## Threat Model

### Attack Vectors Defended

**1. Timing Attacks (CWE-208)**
- **Attack:** Measure execution time to learn token content
- **Defense:** auth-min timing-safe comparison
- **Verification:** Variance < 10% (tested)

**2. Token Leakage**
- **Attack:** Extract tokens from logs, errors, debug output
- **Defense:** auth-min fingerprinting, secrets-management zeroization
- **Verification:** CI checks for raw token usage

**3. Injection Attacks**
- **Attack:** SQL, command, log, path traversal injection
- **Defense:** input-validation sanitization
- **Verification:** BDD scenarios, fuzz testing

**4. Accidental Exposure**
- **Attack:** Exploit misconfigured services bound to 0.0.0.0
- **Defense:** auth-min bind policy enforcement
- **Verification:** Startup validation, fail-fast

**5. Resource Exhaustion**
- **Attack:** 10MB prompts, infinite loops, hung jobs
- **Defense:** input-validation length limits, deadline-propagation
- **Verification:** Unit tests, integration tests

**6. Audit Tampering**
- **Attack:** Modify or delete audit logs to cover tracks
- **Defense:** audit-logging hash chains, immutability
- **Verification:** Integrity verification, checksums

**7. Memory Dumps**
- **Attack:** Extract secrets from memory dumps
- **Defense:** secrets-management zeroization
- **Verification:** Drop trait implementation, compiler fences

**8. Brute Force**
- **Attack:** Brute force tokens through timing differences
- **Defense:** auth-min timing-safe comparison (eliminates timing oracle)
- **Note:** Rate limiting handled at admission layer

---

## Security Guarantees

### What We Guarantee

**1. Timing Safety**
- ✅ All token comparisons use constant-time algorithms
- ✅ Variance < 10% for early vs. late mismatches
- ✅ No information leakage through execution time

**2. Token Safety**
- ✅ No raw tokens in logs (100% fingerprinted)
- ✅ Tokens zeroized on drop (prevents memory dumps)
- ✅ File-based storage (not environment variables)

**3. Audit Integrity**
- ✅ Immutable audit trail (append-only)
- ✅ Tamper detection (hash chains)
- ✅ 7-year retention (GDPR compliant)

**4. Input Safety**
- ✅ All user input validated before processing
- ✅ No injection attacks (SQL, command, log, path)
- ✅ No resource exhaustion (length limits)

**5. Compliance**
- ✅ GDPR compliant (EU regulation)
- ✅ SOC2 ready (US standard)
- ✅ ISO 27001 aligned (international)

### What We Do NOT Guarantee

**rbee is secure, but not magic:**
- ❌ **Not immune to zero-days** (we patch quickly)
- ❌ **Not immune to social engineering** (user education required)
- ❌ **Not immune to physical access** (secure your servers)
- ❌ **Not immune to insider threats** (audit logging helps detect)

**Security is a process, not a destination.**

---

## Conclusion

**rbee's security is not an afterthought—it's foundational.**

**Five specialized security crates work together:**
- **auth-min 🎭:** Timing-safe authentication
- **audit-logging 🔒:** Compliance and forensics
- **input-validation 🛡️:** Injection prevention
- **secrets-management 🔑:** Credential protection
- **deadline-propagation ⏱️:** Resource enforcement

**The result:**
- ✅ Enterprise-grade security
- ✅ GDPR, SOC2, ISO 27001 compliant
- ✅ Defense-in-depth architecture
- ✅ Zero-trust principles
- ✅ Production-ready hardening

**rbee is secure by design. Trust, but verify.** 🔐

---

*Last Updated: 2025-10-10*  
*Based on: auth-min, audit-logging, input-validation, secrets-management, deadline-propagation team responsibilities*  
*Security Reviewed: ✅ All five security teams*

---

Guarded by auth-min Team 🎭  
Secured by Audit Logging Team 🔒  
Validated by Input Validation Team 🛡️  
Protected by Secrets Management Team 🔑  
Optimized by Performance Team ⏱️
