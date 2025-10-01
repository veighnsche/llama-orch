# vram-residency

**Cryptographically sealed VRAM-resident model shards with integrity verification**

`bin/worker-orcd-crates/vram-residency` — Enforces VRAM-only inference policy and provides tamper-evident sealed shard handles with HMAC-SHA256 integrity verification. Prevents RAM fallback, detects VRAM corruption, and provides cryptographic attestation of model residency.

---

## What This Crate Offers

`vram-residency` provides **VRAM-only inference guarantees** with cryptographic integrity verification for worker-orcd. Here's what we offer to other crates:

### 🔒 Core Capabilities

**1. Sealed Shard Contract**
- Cryptographically sealed VRAM-resident model shards
- HMAC-SHA256 signatures prevent seal forgery
- Tamper-evident: detects VRAM corruption or modification
- Immutable once sealed (ready for immediate inference)

**2. VRAM-Only Policy Enforcement**
- GPU is required; fail fast if insufficient VRAM
- Disables unified memory (UMA), zero-copy, and pinned host memory
- Detects and rejects RAM inference attempts
- Cryptographic attestation of VRAM residency

**3. Integrity Verification**
- SHA-256 digest computation on seal
- Re-verification before each Execute operation (TOCTOU prevention)
- Seal signature verification with timing-safe comparison
- Automatic security incident logging on verification failure

**4. Security-First Design**
- TIER 1 security (no panics, no unwrap, no unsafe without bounds checking)
- VRAM pointers never exposed in API or logs
- Seal keys derived from worker token (HKDF-SHA256)
- Automatic key zeroization on drop
- Integration with `audit-logging` for security trail

---

## What You Get

### For worker-api (HTTP endpoints)

```rust
use vram_residency::{VramManager, SealedShard};

// Plan endpoint: Check VRAM capacity
let available = vram_manager.available_vram();
if model_size > available {
    return Err(ErrW::InsufficientVram);
}

// Commit endpoint: Seal model in VRAM
let sealed_shard = vram_manager.seal_model(
    shard_id,
    gpu_device,
    model_bytes,
)?;

// Ready endpoint: Verify seal and return handle
vram_manager.verify_sealed(&sealed_shard)?;
let handle = ModelShardHandle {
    shard_id: sealed_shard.shard_id,
    gpu_device: sealed_shard.gpu_device,
    vram_bytes: sealed_shard.vram_bytes,
    digest: sealed_shard.digest,
    sealed_at: sealed_shard.sealed_at,
    // vram_ptr is private (never exposed)
};

// Execute endpoint: Verify seal before inference
vram_manager.verify_sealed(&sealed_shard)?;
// Now safe to execute inference
```

### For model-loader

```rust
use vram_residency::VramManager;

// Load and seal model
let model_bytes = load_model_from_catalog(&model_ref)?;
let sealed_shard = vram_manager.seal_model(
    generate_shard_id(),
    gpu_device,
    &model_bytes,
)?;

// Shard is now:
// - Resident in VRAM (no RAM fallback)
// - Cryptographically sealed (tamper-evident)
// - Ready for immediate inference
```

### For scheduler

```rust
use vram_residency::VramManager;

// Check VRAM capacity before accepting job
let capacity = vram_manager.available_vram();
if job.estimated_vram > capacity {
    return Err(SchedulerError::InsufficientVram);
}

// Track VRAM usage across workers
let metrics = vram_manager.metrics();
tracing::info!(
    vram_used = metrics.used_bytes,
    vram_available = metrics.available_bytes,
    "VRAM capacity check"
);
```

### For worker-orcd binary

```rust
use vram_residency::{VramManager, VramConfig};

// Initialize at startup
let vram_manager = VramManager::new(VramConfig {
    worker_api_token: config.worker_token.clone(),
    gpu_device: config.gpu_device,
    max_model_size: 100 * 1024 * 1024 * 1024, // 100GB
    audit_logger: audit_logger.clone(),
})?;

// Enforce VRAM-only policy
vram_manager.enforce_vram_only_policy()?;

// Add to app state
let state = WorkerState {
    vram_manager: Arc::new(Mutex::new(vram_manager)),
    // ... other fields
};
```

---

## API Reference

### Core Types

#### `SealedShard`

Cryptographically sealed VRAM-resident model shard.

```rust
pub struct SealedShard {
    pub shard_id: String,           // Unique opaque identifier
    pub gpu_device: u32,            // CUDA device index
    pub vram_bytes: usize,          // Size in VRAM
    pub digest: String,             // SHA-256 hex string
    pub sealed_at: SystemTime,      // Seal timestamp
    // vram_ptr: private (MUST NOT be exposed)
    // signature: private (HMAC-SHA256 seal proof)
}
```

**Security properties**:
- VRAM pointer is private (never exposed in API, logs, or serialization)
- Digest is SHA-256 (FIPS 140-2 approved)
- Signature is HMAC-SHA256 with per-worker secret key
- Immutable once sealed

#### `ModelShardHandle`

Public API type for sealed shard handles (type alias or wrapper).

```rust
pub type ModelShardHandle = SealedShard;
```

**Usage**: Returned in Commit and Ready endpoint responses.

---

### VramManager API

#### Allocation

```rust
pub fn seal_model(
    &mut self,
    shard_id: String,
    gpu_device: u32,
    model_bytes: &[u8],
) -> Result<SealedShard>
```

**What it does**:
1. Validates inputs (shard_id, gpu_device, model_bytes size)
2. Checks available VRAM capacity
3. Allocates VRAM via CUDA FFI (or mock for M0)
4. Copies model bytes to VRAM
5. Computes SHA-256 digest
6. Generates HMAC-SHA256 seal signature
7. Emits `AuditEvent::VramSealed`
8. Returns sealed shard with `sealed: true`

**Error cases**:
- `VramError::InvalidInput` — Invalid shard_id, gpu_device, or size
- `VramError::InsufficientVram(needed, available)` — Not enough VRAM
- `VramError::CudaAllocationFailed` — CUDA malloc failed
- `VramError::IntegrityViolation` — Digest computation failed

---

#### Verification

```rust
pub fn verify_sealed(&self, shard: &SealedShard) -> Result<()>
```

**What it does**:
1. Re-computes digest from VRAM contents
2. Verifies HMAC-SHA256 signature (timing-safe comparison)
3. Optionally checks seal timestamp freshness
4. Emits `AuditEvent::SealVerified` or `AuditEvent::SealVerificationFailed`
5. Returns Ok(()) if valid, Err otherwise

**Error cases**:
- `VramError::SealVerificationFailed` — Digest mismatch (tampering detected)
- `VramError::NotSealed` — Shard not properly sealed
- `VramError::IntegrityViolation` — VRAM corruption detected

**CRITICAL**: Worker transitions to Stopped state on verification failure.

---

#### Capacity Query

```rust
pub fn available_vram(&self) -> usize;
pub fn total_vram(&self) -> usize;
pub fn used_vram(&self) -> usize;
```

**Usage**:
- Plan endpoint checks if model fits before accepting
- Scheduler validates job can be accepted
- Telemetry reports VRAM usage metrics

---

#### Policy Enforcement

```rust
pub fn enforce_vram_only_policy(&self) -> Result<()>
```

**What it does**:
1. Disables unified memory (UMA) at initialization
2. Disables zero-copy and pinned host memory modes
3. Verifies no host memory fallback
4. Emits `AuditEvent::PolicyViolation` if policy cannot be enforced

**Error cases**:
- `VramError::PolicyViolation` — UMA detected or policy cannot be enforced

---

#### Deallocation

```rust
impl Drop for SealedShard {
    fn drop(&mut self) {
        // MUST release VRAM via CUDA FFI
        // MUST emit AuditEvent::VramDeallocated
        // MUST NOT panic (WORKER-4083)
    }
}
```

**Automatic cleanup**: Shard is deallocated when it goes out of scope.

---

## Security Guarantees

### TIER 1 Security Configuration

```rust
// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
```

**What this means**:
- ✅ Never panics (all functions return Result)
- ✅ No unwrap/expect (explicit error handling)
- ✅ Bounds checking on all array access
- ✅ Saturating/checked arithmetic (no overflow)
- ✅ Safe pointer operations (validated offsets)

---

### Cryptographic Properties

**Seal signatures** (HMAC-SHA256):
- Per-worker secret keys (derived from worker API token)
- Covers `(shard_id, digest, sealed_at, gpu_device)`
- Timing-safe verification (prevents timing attacks)
- Detects seal forgery and tampering

**Digest computation** (SHA-256):
- FIPS 140-2 approved hash function
- Computed on seal and re-verified before Execute
- Detects VRAM corruption or modification

**Key management**:
- Keys derived via HKDF-SHA256 (domain separation)
- Automatic zeroization on drop (prevents memory dumps)
- Never logged or exposed in API

---

### VRAM-Only Policy

**Enforced at runtime**:
- ✅ Model weights MUST reside entirely in GPU VRAM during inference
- ✅ Unified memory (UMA) disabled
- ✅ Zero-copy and pinned host memory disabled
- ✅ Fail fast if VRAM capacity insufficient
- ✅ Detect and reject RAM inference attempts

**Cryptographic attestation**:
- Seal signature proves VRAM residency
- Verification before each Execute operation
- Tamper-evident (detects policy bypass)

---

## Integration Pattern

### 1. Initialize at Startup

```rust
use vram_residency::{VramManager, VramConfig};
use audit_logging::AuditLogger;

let audit_logger = AuditLogger::new(audit_config)?;

let vram_manager = VramManager::new(VramConfig {
    worker_api_token: load_worker_token()?,
    gpu_device: 0,
    max_model_size: 100 * 1024 * 1024 * 1024, // 100GB
    audit_logger: Arc::new(audit_logger),
})?;

// Enforce VRAM-only policy
vram_manager.enforce_vram_only_policy()?;
```

---

### 2. Seal Model (Commit Endpoint)

```rust
use input_validation::{validate_identifier, validate_range};

// Validate inputs
validate_identifier(&shard_id, 256)?;
validate_range(gpu_device, 0, gpu_count)?;

// Seal model in VRAM
let sealed_shard = vram_manager.seal_model(
    shard_id,
    gpu_device,
    &model_bytes,
)?;

// Return handle in response
Ok(CommitResponse {
    shard_handle: ModelShardHandle {
        shard_id: sealed_shard.shard_id,
        gpu_device: sealed_shard.gpu_device,
        vram_bytes: sealed_shard.vram_bytes,
        digest: sealed_shard.digest,
        sealed_at: sealed_shard.sealed_at,
    },
})
```

---

### 3. Verify Seal (Execute Endpoint)

```rust
// CRITICAL: Verify seal before EVERY execution
vram_manager.verify_sealed(&sealed_shard)?;

// Now safe to execute inference
let output = run_inference(&sealed_shard)?;
```

---

### 4. Handle Verification Failure

```rust
match vram_manager.verify_sealed(&sealed_shard) {
    Ok(()) => {
        // Seal valid, proceed with inference
    }
    Err(VramError::SealVerificationFailed) => {
        // SECURITY INCIDENT: VRAM corruption or tampering detected
        // Worker MUST transition to Stopped state
        // Audit event already emitted
        transition_to_stopped_state()?;
        return Err(ErrW::SecurityIncident);
    }
    Err(e) => return Err(e.into()),
}
```

---

## Performance Characteristics

**Seal operation** (seal_model):
- VRAM allocation: O(1) CUDA call
- Memory copy: O(n) where n = model size
- SHA-256 digest: O(n) where n = model size
- HMAC signature: O(1) (operates on digest)
- **Total**: O(n) dominated by memory copy

**Verification operation** (verify_sealed):
- Digest re-computation: O(n) where n = model size
- HMAC verification: O(1) (timing-safe comparison)
- **Total**: O(n) dominated by digest computation

**Capacity query** (available_vram):
- O(1) — Simple arithmetic on tracked state

**Policy enforcement** (enforce_vram_only_policy):
- O(1) — CUDA device property queries

---

## Error Handling

### VramError Enum

```rust
pub enum VramError {
    InvalidInput(String),                    // Input validation failed
    InsufficientVram(usize, usize),          // (needed, available)
    SealVerificationFailed,                  // Digest mismatch (tampering)
    NotSealed,                               // Shard not properly sealed
    IntegrityViolation,                      // VRAM corruption detected
    CudaAllocationFailed(String),            // CUDA malloc failed
    CudaDeallocationFailed(String),          // CUDA free failed
    PolicyViolation(String),                 // VRAM-only policy violated
}
```

**Error classification**:
- **Retriable**: `InsufficientVram`, `CudaAllocationFailed`
- **Fatal**: `SealVerificationFailed`, `IntegrityViolation`, `PolicyViolation`
- **Invalid request**: `InvalidInput`, `NotSealed`

---

## Dependencies

### Production Dependencies

```toml
[dependencies]
# Cryptography - HMAC-SHA256 seal signatures
hmac = "0.12"
sha2 = "0.10"

# Shared crates - Security
input-validation = { path = "../../shared-crates/input-validation" }
secrets-management = { path = "../../shared-crates/secrets-management" }
audit-logging = { path = "../../shared-crates/audit-logging" }

# Core infrastructure
thiserror.workspace = true
serde = { workspace = true, features = ["derive"] }
tracing.workspace = true
chrono = { workspace = true }
```

**Why these dependencies?**
- `hmac`, `sha2` — RustCrypto (professionally audited, FIPS 140-2)
- `input-validation` — Centralized security boundary (TIER 2)
- `secrets-management` — Seal key derivation and zeroization (TIER 1)
- `audit-logging` — Security audit trail (TIER 1)

See `.specs/30_dependencies.md` for complete dependency analysis.

---

## Specifications

Implements requirements from:
- **WORKER-4100 to WORKER-4122**: VRAM residency and seal integrity
- **MS-001 to MS-007**: Memory safety requirements
- **CI-001 to CI-007**: Cryptographic integrity requirements
- **VP-001 to VP-006**: VRAM-only policy enforcement
- **IV-001 to IV-005**: Input validation requirements
- **RP-001 to RP-005**: Resource protection requirements

See `.specs/` for full requirements:
- `00_vram-residency.md` — Functional specification
- `10_expectations.md` — Consumer expectations
- `20_security.md` — Security specification
- `30_dependencies.md` — Dependency analysis
- `31_dependency_verification.md` — Shared crate verification

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p vram-residency

# Specific test suites
cargo test -p vram-residency seal      # Seal operations
cargo test -p vram-residency verify    # Verification
cargo test -p vram-residency security  # Security tests
```

### BDD Tests

```bash
# Run BDD test suite
cd bin/worker-orcd-crates/vram-residency/bdd
cargo test
```

### Security Tests

**Required coverage** (per `.specs/20_security.md` §6.1):
- ✅ VRAM pointer not exposed
- ✅ Seal forgery rejected
- ✅ Integer overflow prevented
- ✅ Bounds checking enforced
- ✅ Seal key not logged
- ✅ Timing-safe verification

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Security Tier**: TIER 1 (Critical)
- **Priority**: P0 (blocking for worker-orcd)

---

## Roadmap

### Phase 1: M0 Essentials (Current)
- ✅ `SealedShard` struct with public fields
- ✅ `VramManager::seal_model()` with mock VRAM allocation
- ✅ `VramManager::verify_sealed()` stub
- ✅ `VramError` enum with core variants
- ⬜ Add HMAC-SHA256 seal signature computation
- ⬜ Integrate input validation
- ⬜ Integrate audit logging
- ⬜ Implement capacity query methods

### Phase 2: API Integration (Next)
- ⬜ Wire into worker-api Commit endpoint
- ⬜ Wire into worker-api Ready endpoint
- ⬜ Wire into worker-api Execute endpoint (verification)
- ⬜ Add structured logging for all operations
- ⬜ Implement Drop for SealedShard with audit logging

### Phase 3: CUDA Integration (Post-M0)
- ⬜ Replace mock VRAM with real CUDA FFI
- ⬜ Implement cuda_malloc/cuda_free wrappers
- ⬜ Add CUDA device property queries
- ⬜ Implement digest re-verification from VRAM contents
- ⬜ Add VRAM-only policy enforcement checks

### Phase 4: Production Hardening (Post-M0)
- ⬜ Add metrics emission
- ⬜ Implement tensor-parallel multi-shard support
- ⬜ Add seal timestamp freshness checks
- ⬜ Comprehensive unit tests with proof bundles
- ⬜ Integration tests with real GPU

---

## Contributing

**Before implementing**:
1. Read `.specs/00_vram-residency.md` — Functional specification
2. Read `.specs/20_security.md` — Security requirements
3. Read `.specs/10_expectations.md` — Consumer expectations
4. Follow TIER 1 Clippy configuration (no panics, no unwrap)

**Testing requirements**:
- Unit tests for all public APIs
- Security tests for all vulnerabilities
- BDD tests for consumer scenarios
- Property tests for invariants

---

## For Questions

See:
- `.specs/` — Complete specifications
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Security context
- `bin/worker-orcd/.specs/00_worker-orcd.md` — Parent specification