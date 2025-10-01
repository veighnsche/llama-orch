# Audit Logging — Dependencies & Library Choices

**Crate**: `bin/shared-crates/audit-logging`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document specifies all dependencies for the `audit-logging` crate, including rationale, alternatives considered, and security implications. As a security-critical crate (TIER 1), dependency choices must be carefully justified.

**Dependency Philosophy**:
- **Minimize attack surface** — Fewer dependencies = smaller attack surface
- **Battle-tested only** — Use widely-adopted, well-audited libraries
- **Workspace-managed** — Prefer workspace versions for consistency
- **No custom crypto** — Use industry-standard cryptographic libraries

---

## 1. Dependency Categories

### 1.1 Dependency Tiers

| Tier | Purpose | Scrutiny Level | Examples |
|------|---------|----------------|----------|
| **Core** | Essential functionality | High | `serde`, `tokio`, `chrono` |
| **Security** | Cryptography, validation | Critical | `sha2`, `hmac`, `input-validation` |
| **Infrastructure** | File I/O, async runtime | High | `tracing-appender`, `tokio` |
| **Optional** | Platform mode, advanced features | Medium | `reqwest`, `ed25519-dalek` |

---

## 2. Core Dependencies (M0)

### 2.1 Serialization

**Dependency**: `serde` + `serde_json`

**Purpose**: Event serialization to JSON format

**Rationale**:
- Industry standard for Rust serialization
- Well-audited, widely used
- Workspace-managed version ensures consistency

**Version**: Workspace-managed

**Alternatives Considered**:
- ❌ `bincode` — Binary format not human-readable
- ❌ `rmp-serde` (MessagePack) — Less tooling support
- ❌ Custom JSON — Reinventing the wheel

**Security Considerations**:
- ✅ No known vulnerabilities in recent versions
- ✅ Handles untrusted input safely
- ⚠️ Large dependency tree (via `serde_derive`)

**Status**: ✅ **APPROVED** — Already in use

```toml
[dependencies]
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
```

---

### 2.2 Async Runtime

**Dependency**: `tokio`

**Purpose**: Async I/O, background writer task, buffering

**Rationale**:
- De facto standard for async Rust
- Non-blocking event emission required
- Workspace-managed version

**Version**: Workspace-managed

**Features Required**:
- `rt` — Runtime
- `fs` — Async file I/O
- `sync` — Channels for buffering
- `time` — Periodic flushing

**Alternatives Considered**:
- ❌ `async-std` — Less ecosystem support
- ❌ Sync I/O — Would block operations (unacceptable)

**Security Considerations**:
- ✅ Well-audited, production-proven
- ✅ Large community, quick security patches
- ⚠️ Large dependency (but unavoidable)

**Status**: ✅ **APPROVED** — Already in use

```toml
[dependencies]
tokio = { workspace = true, features = ["rt", "fs", "sync", "time"] }
```

---

### 2.3 Timestamps

**Dependency**: `chrono`

**Purpose**: UTC timestamps, ISO 8601 formatting

**Rationale**:
- Industry standard for Rust date/time
- ISO 8601 compliance (audit requirement)
- Workspace-managed version

**Version**: Workspace-managed

**Alternatives Considered**:
- ❌ `time` crate — Less ergonomic API
- ❌ `std::time::SystemTime` — No ISO 8601 formatting

**Security Considerations**:
- ✅ Well-audited
- ⚠️ Past vulnerabilities (localtime_r) — fixed in recent versions
- ✅ Workspace version ensures patched version

**Status**: ✅ **APPROVED** — Already in use

```toml
[dependencies]
chrono = { workspace = true, features = ["serde"] }
```

---

### 2.4 Error Handling

**Dependency**: `thiserror`

**Purpose**: Ergonomic error types with `#[derive(Error)]`

**Rationale**:
- Standard for Rust error handling
- Zero runtime overhead
- Workspace-managed version

**Version**: Workspace-managed

**Alternatives Considered**:
- ❌ `anyhow` — Not suitable for library crates
- ❌ Manual `impl Error` — Too verbose

**Security Considerations**:
- ✅ Zero dependencies (proc macro only)
- ✅ No runtime code
- ✅ Cannot introduce vulnerabilities

**Status**: ✅ **APPROVED** — Already in use

```toml
[dependencies]
thiserror = { workspace = true }
```

---

### 2.5 Logging

**Dependency**: `tracing`

**Purpose**: Internal logging (not audit events)

**Rationale**:
- Standard for Rust structured logging
- Workspace-managed version
- Used for operational logs (errors, warnings)

**Version**: Workspace-managed

**Alternatives Considered**:
- ❌ `log` crate — Less structured
- ❌ No logging — Need operational visibility

**Security Considerations**:
- ✅ Well-audited
- ⚠️ Must not log sensitive data (use carefully)

**Status**: ✅ **APPROVED** — Already in use

```toml
[dependencies]
tracing = { workspace = true }
```

---

## 3. Security Dependencies (M0)

### 3.1 Cryptographic Hashing

**Dependency**: `sha2`

**Purpose**: SHA-256 for hash chains, event digests

**Rationale**:
- **FIPS 140-2 approved** algorithm
- Part of RustCrypto project (well-audited)
- Industry standard for integrity verification
- Fast, secure, battle-tested

**Version**: `0.10` (latest stable)

**Alternatives Considered**:
- ❌ `md5` — Cryptographically broken (collision attacks)
- ❌ `sha1` — Deprecated (collision attacks)
- ❌ `blake3` — Faster but less standardized
- ❌ `ring` — More complex API, larger dependency

**Security Considerations**:
- ✅ **RECOMMENDED** by NIST for integrity
- ✅ No known vulnerabilities
- ✅ Constant-time implementation
- ✅ Part of RustCrypto (security-focused)

**Usage**:
```rust
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

**Status**: ✅ **APPROVED** — Required for M0

```toml
[dependencies]
sha2 = "0.10"
```

---

### 3.2 Input Validation

**Dependency**: `input-validation` (internal crate)

**Purpose**: Sanitize user input before logging (prevent log injection)

**Rationale**:
- **Critical security requirement** — Prevents ANSI/control char injection
- Already built and tested
- Zero external dependencies
- Workspace-managed

**Version**: Workspace path dependency

**Alternatives Considered**:
- ❌ Manual sanitization — Error-prone, not reusable
- ❌ External crate — None exist with our requirements
- ✅ Internal crate — Already built, tested, documented

**Security Considerations**:
- ✅ **REQUIRED** for security (see `.specs/20_security.md` §2.1)
- ✅ Removes ANSI escape sequences
- ✅ Removes control characters
- ✅ Removes Unicode directional overrides
- ✅ Removes null bytes

**Usage**:
```rust
use input_validation::sanitize_string;

// Sanitize before logging
let safe_user_id = sanitize_string(&user_id)?;
audit_logger.emit(AuditEvent::AuthFailure {
    attempted_user: Some(safe_user_id),
    ...
}).await?;
```

**Status**: ✅ **APPROVED** — Required for M0

```toml
[dependencies]
input-validation = { path = "../input-validation" }
```

---

### 3.3 File Rotation

**Dependency**: `tracing-appender`

**Purpose**: Async file writing with daily rotation

**Rationale**:
- **Battle-tested** — Used by tracing ecosystem
- Handles rotation, async I/O, buffering
- Non-blocking writes
- Integrates with tokio

**Version**: `0.2` (latest stable)

**Alternatives Considered**:
- ❌ Manual file rotation — Complex, error-prone
- ❌ `log4rs` — Overkill, not async-native
- ❌ `flexi_logger` — Not tokio-native
- ✅ `tracing-appender` — Perfect fit

**Security Considerations**:
- ✅ Well-audited (part of tracing ecosystem)
- ✅ Handles file permissions correctly
- ✅ Atomic file operations
- ⚠️ Not tamper-evident (we add hash chain on top)

**Usage**:
```rust
use tracing_appender::rolling::{RollingFileAppender, Rotation};

let file_appender = RollingFileAppender::new(
    Rotation::DAILY,
    "/var/lib/llorch/audit/orchestratord",
    "audit.log"
);
```

**Status**: ✅ **APPROVED** — Recommended for M0

```toml
[dependencies]
tracing-appender = "0.2"
```

---

## 4. Optional Dependencies (Production)

### 4.1 HMAC Signatures

**Dependency**: `hmac`

**Purpose**: HMAC-SHA256 signatures for event authentication (platform mode)

**Rationale**:
- Part of RustCrypto project
- Industry standard for message authentication
- Simple, secure, well-audited

**Version**: `0.12` (latest stable)

**Alternatives Considered**:
- ❌ Custom HMAC — Never roll your own crypto
- ❌ `ring` — More complex, larger dependency
- ✅ `hmac` — Standard choice

**Security Considerations**:
- ✅ **FIPS 140-2 approved** (when used with SHA-256)
- ✅ Constant-time comparison
- ✅ No known vulnerabilities

**Usage**:
```rust
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
```

**Status**: ✅ **APPROVED** — Required for platform mode

```toml
[dependencies]
hmac = { version = "0.12", optional = true }
```

---

### 4.2 Ed25519 Signatures

**Dependency**: `ed25519-dalek`

**Purpose**: Asymmetric signatures for platform mode (provider authentication)

**Rationale**:
- Modern, fast, secure signature algorithm
- Asymmetric (public key verification)
- Better than HMAC for multi-party scenarios

**Version**: `2.0` (latest stable)

**When to Use**:
- **Platform mode** — Providers sign events, platform verifies
- **Multi-party** — Different keys per provider
- **Public verification** — Anyone can verify with public key

**Alternatives Considered**:
- ❌ RSA — Slower, more complex
- ❌ ECDSA — More complex, potential timing issues
- ✅ Ed25519 — Modern, fast, simple

**Security Considerations**:
- ✅ Modern algorithm (2011)
- ✅ Constant-time implementation
- ✅ No known vulnerabilities
- ⚠️ Larger dependency than HMAC

**Usage**:
```rust
use ed25519_dalek::{Keypair, Signature, Signer, Verifier};

pub fn sign_event_ed25519(event: &AuditEventEnvelope, keypair: &Keypair) -> String {
    let message = serialize_for_signing(event);
    let signature = keypair.sign(message.as_bytes());
    hex::encode(signature.to_bytes())
}

pub fn verify_signature_ed25519(
    event: &AuditEventEnvelope,
    signature: &str,
    public_key: &ed25519_dalek::PublicKey
) -> bool {
    let message = serialize_for_signing(event);
    let sig_bytes = hex::decode(signature).unwrap();
    let signature = Signature::from_bytes(&sig_bytes).unwrap();
    public_key.verify(message.as_bytes(), &signature).is_ok()
}
```

**Status**: ✅ **APPROVED** — Optional for platform mode

```toml
[dependencies]
ed25519-dalek = { version = "2.0", optional = true }
```

---

### 4.3 HTTP Client

**Dependency**: `reqwest`

**Purpose**: HTTP client for platform mode (send events to central service)

**Rationale**:
- Industry standard for Rust HTTP clients
- Async, tokio-native
- TLS support built-in

**Version**: `0.11` (latest stable)

**Features Required**:
- `json` — JSON serialization
- `rustls-tls` — TLS support (prefer rustls over openssl)

**Alternatives Considered**:
- ❌ `hyper` — Lower-level, more complex
- ❌ `ureq` — Sync only
- ✅ `reqwest` — High-level, async, perfect fit

**Security Considerations**:
- ✅ TLS 1.3 support
- ✅ Certificate validation
- ⚠️ Large dependency tree
- ⚠️ Only needed for platform mode

**Usage**:
```rust
use reqwest::Client;

pub async fn send_events_to_platform(
    events: Vec<AuditEventEnvelope>,
    config: &PlatformConfig,
) -> Result<(), AuditError> {
    let client = Client::new();
    let response = client
        .post(&config.endpoint)
        .json(&events)
        .send()
        .await?;
    
    response.error_for_status()?;
    Ok(())
}
```

**Status**: ✅ **APPROVED** — Optional for platform mode

```toml
[dependencies]
reqwest = { version = "0.11", features = ["json", "rustls-tls"], optional = true }
```

---

### 4.4 Rate Limiting

**Dependency**: `governor`

**Purpose**: Rate limit query API (prevent CPU exhaustion)

**Rationale**:
- Standard rate limiting library for Rust
- Efficient, well-tested
- Integrates with async

**Version**: `0.6` (latest stable)

**Alternatives Considered**:
- ❌ `tower-governor` — Tied to tower/axum (we're a library)
- ❌ Manual rate limiting — Complex, error-prone
- ✅ `governor` — Perfect for library use

**Security Considerations**:
- ✅ Prevents DoS via expensive queries
- ✅ Per-actor rate limiting
- ✅ No known vulnerabilities

**Usage**:
```rust
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;

pub struct AuditLogger {
    query_limiter: RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>,
}

impl AuditLogger {
    pub async fn query(
        &self,
        query: AuditQuery,
        requester: &str,
    ) -> Result<Vec<AuditEventEnvelope>, AuditError> {
        // Rate limit: 10 queries per second per requester
        self.query_limiter.check_key(&requester.to_string())
            .map_err(|_| AuditError::RateLimitExceeded)?;
        
        self.execute_query(query).await
    }
}
```

**Status**: ✅ **APPROVED** — Optional for production

```toml
[dependencies]
governor = { version = "0.6", optional = true }
```

---

## 5. Rejected Dependencies

### 5.1 Database Libraries

**Rejected**: `diesel`, `sqlx`, `rusqlite`

**Rationale**:
- Overkill for append-only logs
- Flat files sufficient for audit logs
- Adds complexity and attack surface
- Not needed for compliance requirements

**Alternative**: Flat files with hash chains

---

### 5.2 Embedded Databases

**Rejected**: `sled`, `redb`, `rocksdb`

**Rationale**:
- Overkill for append-only logs
- Not designed for audit trail use case
- Adds complexity
- Flat files simpler and more transparent

**Alternative**: Flat files with `tracing-appender`

---

### 5.3 Custom Crypto

**Rejected**: Any custom cryptographic implementations

**Rationale**:
- **Never roll your own crypto**
- Use battle-tested libraries only
- Security-critical code requires expert review

**Alternative**: RustCrypto libraries (`sha2`, `hmac`, `ed25519-dalek`)

---

### 5.4 `log` Crate

**Rejected**: `log` crate

**Rationale**:
- Already using `tracing` (workspace standard)
- `log` is less structured
- No benefit to adding both

**Alternative**: `tracing` (already in use)

---

## 6. Dependency Summary

### 6.1 M0 Dependencies (Minimal Viable)

```toml
[dependencies]
# Core (already have)
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
tokio = { workspace = true, features = ["rt", "fs", "sync", "time"] }
chrono = { workspace = true, features = ["serde"] }
thiserror = { workspace = true }
tracing = { workspace = true }

# Add for M0
sha2 = "0.10"                                      # Hash chains
input-validation = { path = "../input-validation" }  # Sanitization
tracing-appender = "0.2"                           # File rotation
```

**Total new dependencies**: 3 (`sha2`, `input-validation`, `tracing-appender`)

---

### 6.2 Production Dependencies (Full Security)

```toml
[dependencies]
# M0 dependencies +
hmac = { version = "0.12", optional = true }       # Signatures
ed25519-dalek = { version = "2.0", optional = true }  # Asymmetric signatures
reqwest = { version = "0.11", features = ["json", "rustls-tls"], optional = true }  # Platform mode
governor = { version = "0.6", optional = true }    # Rate limiting

[features]
default = []
platform = ["hmac", "ed25519-dalek", "reqwest"]
rate-limiting = ["governor"]
full = ["platform", "rate-limiting"]
```

**Total production dependencies**: 7 additional (optional)

---

### 6.3 Dependency Tree Analysis

**Direct dependencies**: 9 (M0) + 4 (optional) = 13 total

**Transitive dependencies** (estimated):
- `serde`: ~5 transitive deps
- `tokio`: ~20 transitive deps
- `chrono`: ~5 transitive deps
- `sha2`: ~2 transitive deps
- `reqwest`: ~30 transitive deps (platform mode only)

**Total dependency count**: ~50-60 crates (mostly from tokio/reqwest)

**Security posture**: ✅ **ACCEPTABLE**
- All dependencies are workspace-managed or well-audited
- No custom crypto
- Minimal attack surface for M0
- Optional features for advanced use cases

---

## 7. Dependency Security

### 7.1 Security Audit Process

**Required for all dependencies**:
1. ✅ Check `cargo audit` for known vulnerabilities
2. ✅ Review dependency tree (`cargo tree`)
3. ✅ Verify crate ownership (trusted maintainers)
4. ✅ Check recent activity (not abandoned)
5. ✅ Review security advisories

**Audit schedule**:
- **Before adding**: Full security review
- **Weekly**: `cargo audit` in CI
- **Monthly**: Dependency update review
- **Quarterly**: Full dependency tree audit

---

### 7.2 Vulnerability Response

**If vulnerability discovered**:
1. **Assess impact** — Does it affect audit-logging?
2. **Update immediately** — Patch to fixed version
3. **Test thoroughly** — Ensure no breaking changes
4. **Document** — Update this spec with lessons learned

**Escalation**:
- **Critical**: Update within 24 hours
- **High**: Update within 1 week
- **Medium**: Update in next release
- **Low**: Update when convenient

---

### 7.3 Supply Chain Security

**Mitigations**:
1. ✅ **Cargo.lock committed** — Reproducible builds
2. ✅ **Workspace versions** — Centralized dependency management
3. ✅ **Minimal dependencies** — Smaller attack surface
4. ✅ **Trusted sources** — RustCrypto, tokio-rs, serde-rs
5. ⬜ **Vendoring** (optional) — Offline builds, supply chain isolation

**CI checks**:
```bash
# Check for vulnerabilities
cargo audit

# Check for outdated dependencies
cargo outdated

# Review dependency tree
cargo tree
```

---

## 8. Provenance Metadata

### 8.1 Current Approach (M0)

**Minimal provenance** — Just enough for forensics:
```rust
pub struct AuditEventEnvelope {
    pub audit_id: String,           // Unique ID
    pub timestamp: DateTime<Utc>,   // When
    pub service_id: String,         // Who (orchestratord, pool-managerd, worker-gpu-0)
    pub event: AuditEvent,
    pub prev_hash: String,
    pub hash: String,
}
```

**Rationale**:
- Sufficient for single-node deployments
- Simple, no extra dependencies
- Easy to understand and verify

---

### 8.2 Enhanced Provenance (Production)

**Optional provenance metadata** — For distributed systems:
```rust
pub struct AuditEventEnvelope {
    pub audit_id: String,
    pub timestamp: DateTime<Utc>,
    pub service_id: String,
    
    // Optional provenance (add later)
    pub provenance: Option<ProvenanceInfo>,
    
    pub event: AuditEvent,
    pub prev_hash: String,
    pub hash: String,
}

pub struct ProvenanceInfo {
    pub service_name: String,     // "orchestratord"
    pub service_version: String,  // "0.1.0"
    pub hostname: String,         // "gpu-node-1"
    pub process_id: u32,          // 12345
    pub correlation_id: Option<String>,  // Link to narration events
    pub code_location: Option<String>,   // "src/api/auth.rs:42" (debug builds only)
}
```

**When to add**:
- **Distributed deployments** — Multiple nodes, need to identify source
- **Forensic investigation** — Need detailed event origin
- **Compliance requirements** — Prove which system processed data

**Dependencies**: None (uses std only)

**Status**: ⬜ **DEFERRED** — Not needed for M0

---

## 9. Refinement Opportunities

### 9.1 Immediate (M0)

1. **Add `sha2` dependency** — Required for hash chains
2. **Add `input-validation` dependency** — Required for sanitization
3. **Add `tracing-appender` dependency** — Recommended for file rotation
4. **Update Cargo.toml** — Add M0 dependencies

### 9.2 Medium-Term (Production)

5. **Add `hmac` dependency** — Required for platform mode
6. **Add `ed25519-dalek` dependency** — Optional for asymmetric signatures
7. **Add `reqwest` dependency** — Required for platform mode
8. **Add `governor` dependency** — Optional for rate limiting
9. **Add feature flags** — `platform`, `rate-limiting`, `full`

### 9.3 Long-Term (Hardening)

10. **Evaluate vendoring** — Supply chain isolation
11. **Add dependency fuzzing** — Fuzz critical dependencies
12. **Implement SBOM** — Software Bill of Materials for compliance
13. **Add dependency pinning** — Pin critical dependencies to specific versions

---

## 10. References

**Internal Documentation**:
- `.specs/00_overview.md` — Architecture overview
- `.specs/20_security.md` — Security attack surface analysis
- `.specs/02_storage-and-tamper-evidence.md` — Storage requirements
- `bin/shared-crates/input-validation/.specs/21_security_verification.md` — Input validation security

**External Resources**:
- RustCrypto: https://github.com/RustCrypto
- Tokio: https://tokio.rs/
- Tracing: https://tracing.rs/
- Cargo Audit: https://github.com/rustsec/rustsec

**Security Standards**:
- FIPS 140-2 — Cryptographic Module Validation
- NIST SP 800-57 — Key Management Recommendations
- OWASP Dependency Check — Supply chain security

---

**End of Dependencies Specification**
